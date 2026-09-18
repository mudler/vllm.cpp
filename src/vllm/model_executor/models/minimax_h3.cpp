// MiniMax-H3 DiT — config parse, weight contract, packed forward, denoise loop.
// Port of vllm-project/vllm-omni, vllm_omni/diffusion/models/minimax_h3/
// minimax_h3_transformer.py (the DiT) and denoise_loop.py (the CFG-distilled loop).
//
// ─── SCOPE OF THIS TU (read before trusting it for performance) ──────────────
// This is the CORRECTNESS forward: the two structurally hard pieces route through
// the SHARED vt:: ops — every projection is `vt::MatmulBT` and the packed
// varlen NON-CAUSAL attention is `vt::DFlashBlockAttention(causal=false)`, whose
// per-document bidirectional contract is exactly upstream's `cu_seqlens` varlen FA
// call (minimax_h3_transformer.py:288-317, 383-419) — while the elementwise glue
// (RMSNorm, SiLU, 3D RoPE, AdaLN modulate/gate, the indexed scatter/gather) runs as
// explicit host loops so the math can be read against upstream line by line and
// gated exactly. It therefore requires a CPU device.
//
// The DEVICE-RESIDENT forward is a separate, tracked brick (H3-2b in
// .agents/specs/minimax-h3.md): it folds the glue onto `vt::FusedChain` recipes and
// a fused AdaLN modulate op, which is also where the speed work belongs (upstream
// reports the DiT at 88% of request latency). Nothing here claims a speed result.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/dtype.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vllm/model_executor/models/dit_lora.h"

namespace vllm {
namespace {

using vt::DType;
using vt::Tensor;

// ---------------------------------------------------------------------------
// Small host helpers. Every one names the upstream lines it mirrors.
// ---------------------------------------------------------------------------

// nn.RMSNorm with fp32 variance accumulation (minimax_h3_transformer.py:171-175).
void RmsNormRows(const float* in, const float* weight, float* out, int64_t rows, int64_t width,
                 double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * width;
    double sum = 0.0;
    for (int64_t i = 0; i < width; ++i) sum += static_cast<double>(src[i]) * src[i];
    const float inv = static_cast<float>(1.0 / std::sqrt(sum / static_cast<double>(width) + eps));
    float* dst = out + r * width;
    for (int64_t i = 0; i < width; ++i) dst[i] = src[i] * inv * weight[i];
  }
}

float Silu(float x) { return x / (1.0f + std::exp(-x)); }

// Round through bfloat16 (round-to-nearest-even, matching torch's .to(bf16)).
// The production H3 stream is bf16 with fp32 islands; rounding IN PLACE rather
// than switching storage keeps this forward comparing the same CAST POINTS as
// upstream instead of a different accumulation strategy.
float RoundBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u) {  // inf/nan pass through
    bits &= 0xFFFF0000u;
  } else {
    const uint32_t lsb = (bits >> 16) & 1u;
    bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// The block-stream dtype policy for one forward.
struct StreamDtype {
  bool bf16 = false;
  float operator()(float value) const { return bf16 ? RoundBf16(value) : value; }
  void Apply(float* data, int64_t count) const {
    if (!bf16) return;
    for (int64_t i = 0; i < count; ++i) data[i] = RoundBf16(data[i]);
  }
  void Apply(std::vector<float>& buffer) const {
    Apply(buffer.data(), static_cast<int64_t>(buffer.size()));
  }
};

// vt::MatmulBT + optional bias. Weight is [out_features, in_features], matching
// every {Column,Row,QKV,MergedColumn}ParallelLinear at TP=1.
void Linear(vt::Queue& q, const float* in, int64_t rows, int64_t in_features,
            const Tensor& weight, const Tensor* bias, float* out,
            const DitRuntimeLoraLayer* lora = nullptr) {
  const int64_t out_features = weight.shape[0];
  VT_CHECK(weight.rank == 2 && weight.shape[1] == in_features,
           "minimax_h3 linear: weight shape does not match input width");
  VT_CHECK(weight.dtype == DType::kF32, "minimax_h3 linear: reference forward expects f32 weights");
  Tensor a = Tensor::Contiguous(const_cast<float*>(in), DType::kF32, weight.device,
                                {rows, in_features});
  Tensor o = Tensor::Contiguous(out, DType::kF32, weight.device, {rows, out_features});
  vt::MatmulBT(q, o, a, weight);
  if (bias != nullptr && bias->data != nullptr) {
    const float* b = bias->Ptr<float>();
    for (int64_t r = 0; r < rows; ++r) {
      float* dst = out + r * out_features;
      for (int64_t i = 0; i < out_features; ++i) dst[i] += b[i];
    }
  }
  if (lora != nullptr) {
    DitApplyRuntimeLoraDelta(q, a, out, rows, out_features, lora);
  }
}

// MiniMaxH3Rope.forward (minimax_h3_transformer.py:222-230).
// img_position_ids [S,3] fp64 -> freqs [S, 6*L] fp32, laid out as
// [t|h|w|t|h|w] with L frequencies per axis.
std::vector<float> RopeFreqs(const double* position_ids, int64_t seq_len,
                             const float* inv_freq, int64_t inv_freq_len) {
  const int64_t half = 3 * inv_freq_len;
  std::vector<float> freqs(static_cast<size_t>(seq_len * 2 * half));
  for (int64_t s = 0; s < seq_len; ++s) {
    float* dst = freqs.data() + s * 2 * half;
    for (int64_t axis = 0; axis < 3; ++axis) {
      const float pos = static_cast<float>(position_ids[s * 3 + axis]);
      for (int64_t i = 0; i < inv_freq_len; ++i) {
        dst[axis * inv_freq_len + i] = pos * inv_freq[i];
      }
    }
    std::memcpy(dst + half, dst, static_cast<size_t>(half) * sizeof(float));
  }
  return freqs;
}

// _apply_rope (minimax_h3_transformer.py:233-244): rotate the first rot_dim head
// dims with rotate_half, pass the rest through.
void ApplyRope(float* x, int64_t rows, int64_t heads, int64_t head_dim, const float* freqs,
               int64_t rot_dim) {
  const int64_t half = rot_dim / 2;
  for (int64_t r = 0; r < rows; ++r) {
    const float* f = freqs + r * rot_dim;
    for (int64_t h = 0; h < heads; ++h) {
      float* v = x + (r * heads + h) * head_dim;
      for (int64_t i = 0; i < half; ++i) {
        const float cos_lo = std::cos(f[i]), sin_lo = std::sin(f[i]);
        const float cos_hi = std::cos(f[i + half]), sin_hi = std::sin(f[i + half]);
        const float lo = v[i], hi = v[i + half];
        // rotate_half: (-x2, x1) => out_lo = lo*cos - hi*sin, out_hi = hi*cos + lo*sin
        v[i] = lo * cos_lo - hi * sin_lo;
        v[i + half] = hi * cos_hi + lo * sin_hi;
      }
    }
  }
}

// _modulate_scale_shift (minimax_h3_transformer.py:183-192).
void ModulateScaleShift(float* x, int64_t rows, int64_t width, const float* shift,
                        const float* scale, const int64_t* indices) {
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t idx = indices[r];
    float* dst = x + r * width;
    const float* s = scale + idx * width;
    const float* h = shift + idx * width;
    for (int64_t i = 0; i < width; ++i) dst[i] = dst[i] * (1.0f + s[i]) + h[i];
  }
}

// _modulate_gate (minimax_h3_transformer.py:195-204).
void ModulateGate(float* residual, int64_t rows, int64_t width, const float* gate,
                  const float* other, const int64_t* indices) {
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t idx = indices[r];
    float* dst = residual + r * width;
    const float* g = gate + idx * width;
    const float* o = other + r * width;
    for (int64_t i = 0; i < width; ++i) dst[i] += g[i] * o[i];
  }
}

// MiniMaxH3AdalnProj.forward (minimax_h3_transformer.py:555-561):
// silu(t_emb) -> linear -> view(M*modality_num, expand*H) -> chunk(expand).
// Returns the flat [M*modality_num, expand*H] buffer; chunk c of row m starts at
// (m * expand + c) * H within the caller's indexing scheme below.
// `apply_silu` is FALSE for the pruned/curve form: the curve table already
// carries the activated embedding, so upstream skips the SiLU
// (comfy/ldm/minimax/model.py:197, with apply_silu = not use_adaln_curves at :421).
std::vector<float> AdalnProject(vt::Queue& q, const float* t_emb, int64_t m, int64_t time_embed_dim,
                                const Tensor& weight, const Tensor& bias, const StreamDtype& dt,
                                bool apply_silu,
                                const DitRuntimeLoraLayer* lora = nullptr) {
  std::vector<float> activated(static_cast<size_t>(m * time_embed_dim));
  for (int64_t i = 0; i < m * time_embed_dim; ++i) {
    activated[static_cast<size_t>(i)] = apply_silu ? Silu(t_emb[i]) : t_emb[i];
  }
  const int64_t out_features = weight.shape[0];
  // silu(t_emb) is fp32, then cast to the BF16 linear's dtype before the GEMM.
  dt.Apply(activated);
  std::vector<float> out(static_cast<size_t>(m * out_features));
  Linear(q, activated.data(), m, time_embed_dim, weight, &bias, out.data(), lora);
  dt.Apply(out);
  return out;
}

// One chunk of an AdaLN projection, materialized as [M*modality_num, H] so the
// per-row `index_select` in the modulate helpers is a plain row lookup.
std::vector<float> AdalnChunk(const std::vector<float>& projected, int64_t m, int64_t modality_num,
                              int64_t hidden_size, int64_t expand_ratio, int64_t chunk) {
  const int64_t rows = m * modality_num;
  const int64_t row_width = expand_ratio * hidden_size;
  std::vector<float> out(static_cast<size_t>(rows * hidden_size));
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = projected.data() + r * row_width + chunk * hidden_size;
    std::memcpy(out.data() + r * hidden_size, src, static_cast<size_t>(hidden_size) * sizeof(float));
  }
  return out;
}

struct AttentionBlockWeights {
  const Tensor* qkv;
  const Tensor* q_norm;
  const Tensor* k_norm;
  const Tensor* out_proj;
};

// MiniMaxH3Attention.forward (minimax_h3_transformer.py:421-467): fused qkv ->
// per-head q/k RMSNorm -> RoPE -> varlen non-causal attention -> output projection.
void AttentionForward(vt::Queue& q, const MiniMaxH3DitParams& params,
                      const AttentionBlockWeights& w, const float* in, int64_t rows,
                      const float* rope_freqs, const int32_t* cu_seqlens, int num_reqs,
                      const StreamDtype& dt, float* out,
                      const DitRuntimeLoraLayer* qkv_lora = nullptr,
                      const DitRuntimeLoraLayer* out_lora = nullptr) {
  const int64_t heads = params.num_attention_heads;
  const int64_t head_dim = params.attention_head_dim;
  const int64_t inner = heads * head_dim;
  const int64_t hidden = params.hidden_size;

  std::vector<float> qkv(static_cast<size_t>(rows * 3 * inner));
  Linear(q, in, rows, hidden, *w.qkv, nullptr, qkv.data(), qkv_lora);
  dt.Apply(qkv);

  std::vector<float> qbuf(static_cast<size_t>(rows * inner));
  std::vector<float> kbuf(static_cast<size_t>(rows * inner));
  std::vector<float> vbuf(static_cast<size_t>(rows * inner));
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = qkv.data() + r * 3 * inner;
    std::memcpy(qbuf.data() + r * inner, src, static_cast<size_t>(inner) * sizeof(float));
    std::memcpy(kbuf.data() + r * inner, src + inner, static_cast<size_t>(inner) * sizeof(float));
    std::memcpy(vbuf.data() + r * inner, src + 2 * inner, static_cast<size_t>(inner) * sizeof(float));
  }

  // Per-head RMSNorm over head_dim: [rows, heads, head_dim] -> [rows*heads, head_dim].
  std::vector<float> qn(qbuf.size()), kn(kbuf.size());
  RmsNormRows(qbuf.data(), w.q_norm->Ptr<float>(), qn.data(), rows * heads, head_dim,
              params.qk_norm_eps);
  RmsNormRows(kbuf.data(), w.k_norm->Ptr<float>(), kn.data(), rows * heads, head_dim,
              params.qk_norm_eps);

  dt.Apply(qn);
  dt.Apply(kn);
  if (rope_freqs != nullptr) {
    ApplyRope(qn.data(), rows, heads, head_dim, rope_freqs, params.rope_rot_dim());
    ApplyRope(kn.data(), rows, heads, head_dim, rope_freqs, params.rope_rot_dim());
    dt.Apply(qn);
    dt.Apply(kn);
  }

  // The SHARED packed bidirectional attention op. `cu_seqlens` carries the packed
  // document bounds, so attention never crosses into the alignment padding.
  const vt::Device device = w.qkv->device;
  Tensor tq = Tensor::Contiguous(qn.data(), DType::kF32, device, {rows, heads, head_dim});
  Tensor tk = Tensor::Contiguous(kn.data(), DType::kF32, device, {rows, heads, head_dim});
  Tensor tv = Tensor::Contiguous(vbuf.data(), DType::kF32, device, {rows, heads, head_dim});
  std::vector<float> attn(static_cast<size_t>(rows * inner));
  Tensor ta = Tensor::Contiguous(attn.data(), DType::kF32, device, {rows, heads, head_dim});
  vt::DFlashBlockAttentionArgs args;
  args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  args.causal = false;  // bidirectional within each packed document
  args.sliding_window = 0;
  args.cu_seqlens = cu_seqlens;
  args.num_reqs = num_reqs;
  vt::DFlashBlockAttention(q, ta, tq, tk, tv, args);
  dt.Apply(attn);

  Linear(q, attn.data(), rows, inner, *w.out_proj, nullptr, out, out_lora);
  dt.Apply(out, rows * hidden);
}

// MiniMaxH3MLP.forward (minimax_h3_transformer.py:512-517): silu(gate) * up.
void MlpForward(vt::Queue& q, const MiniMaxH3DitParams& params, const Tensor& fc1,
                const Tensor& fc2, const float* in, int64_t rows, const StreamDtype& dt,
                float* out, const DitRuntimeLoraLayer* fc1_lora = nullptr,
                const DitRuntimeLoraLayer* fc2_lora = nullptr) {
  const int64_t ffn = params.ffn_hidden_size;
  std::vector<float> hidden(static_cast<size_t>(rows * 2 * ffn));
  Linear(q, in, rows, params.hidden_size, fc1, nullptr, hidden.data(), fc1_lora);
  dt.Apply(hidden);
  std::vector<float> act(static_cast<size_t>(rows * ffn));
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = hidden.data() + r * 2 * ffn;
    float* dst = act.data() + r * ffn;
    for (int64_t i = 0; i < ffn; ++i) dst[i] = Silu(src[i]) * src[ffn + i];
  }
  dt.Apply(act);
  Linear(q, act.data(), rows, ffn, fc2, nullptr, out, fc2_lora);
  dt.Apply(out, rows * params.hidden_size);
}

AttentionBlockWeights AttnOf(const MiniMaxH3DitBlockWeights& b) {
  return AttentionBlockWeights{&b.qkv_proj, &b.q_norm, &b.k_norm, &b.out_proj};
}

}  // namespace

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// MiniMaxH3DiTArchConfig.from_mapping (minimax_h3_transformer.py:69-78): take the
// keys the mapping actually carries, keep the shipped defaults for the rest.
MiniMaxH3DitParams ParseMiniMaxH3DitParams(const nlohmann::json& config) {
  MiniMaxH3DitParams p;
  auto get_int = [&](const char* key, int64_t& slot) {
    if (config.contains(key) && config.at(key).is_number()) slot = config.at(key).get<int64_t>();
  };
  auto get_double = [&](const char* key, double& slot) {
    if (config.contains(key) && config.at(key).is_number()) slot = config.at(key).get<double>();
  };
  get_int("num_layers", p.num_layers);
  get_int("token_refiner_num_layers", p.token_refiner_num_layers);
  get_int("hidden_size", p.hidden_size);
  get_int("num_attention_heads", p.num_attention_heads);
  get_int("attention_head_dim", p.attention_head_dim);
  get_int("ffn_hidden_size", p.ffn_hidden_size);
  get_int("latents_dim", p.latents_dim);
  get_int("audio_latents_dim", p.audio_latents_dim);
  get_int("text_dim", p.text_dim);
  get_int("timestep_input_dim", p.timestep_input_dim);
  get_int("time_embed_hidden_size", p.time_embed_hidden_size);
  get_int("time_embed_dim", p.time_embed_dim);
  get_int("adaln_out_features", p.adaln_out_features);
  get_int("final_adaln_out_features", p.final_adaln_out_features);
  get_int("rope_inv_freq_len", p.rope_inv_freq_len);
  get_double("norm_eps", p.norm_eps);
  get_double("qk_norm_eps", p.qk_norm_eps);
  get_double("final_norm_eps", p.final_norm_eps);
  if (config.contains("patch_size")) {
    const nlohmann::json& patch = config.at("patch_size");
    VT_CHECK(patch.is_array() && patch.size() == 3,
             "minimax_h3: patch_size must contain three values");
    p.patch_size_t = patch[0].get<int64_t>();
    p.patch_size_h = patch[1].get<int64_t>();
    p.patch_size_w = patch[2].get<int64_t>();
  }
  // The invariants the upstream model asserts at construction
  // (minimax_h3_transformer.py:539-541, 810-828, 852-859).
  VT_CHECK(p.num_attention_heads > 0 && p.hidden_size > 0 && p.attention_head_dim > 0 &&
               p.ffn_hidden_size > 0,
           "minimax_h3: geometry scalars must be positive");
  VT_CHECK(p.adaln_out_features == 6 * p.hidden_size * kMiniMaxH3AdalnModalityNum,
           "minimax_h3: adaln_out_features must be 6 * hidden_size * 3");
  VT_CHECK(p.final_adaln_out_features == 2 * p.hidden_size,
           "minimax_h3: final_adaln_out_features must be 2 * hidden_size");
  VT_CHECK(p.rope_rot_dim() <= p.attention_head_dim,
           "minimax_h3: 6 * rope_inv_freq_len must not exceed attention_head_dim");
  return p;
}

// ---------------------------------------------------------------------------
// Weight contract
// ---------------------------------------------------------------------------

// The DiT loads by EXACT checkpoint name (minimax_h3_transformer.py:906-922), so
// this enumeration IS the contract. FP32 names come from
// MINIMAX_H3_FP32_PARAM_NAMES / _BUFFER_NAMES (:85-101), which post_load_weights
// re-asserts (:898-904).
std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3DitTensors(const MiniMaxH3DitParams& p) {
  std::vector<MiniMaxH3TensorSpec> out;
  const int64_t inner = p.num_attention_heads * p.attention_head_dim;
  const int64_t video_patch_dim = p.video_row_width();

  out.push_back({"video_patch_proj.weight", {p.hidden_size, video_patch_dim}, true});
  out.push_back({"video_patch_proj.bias", {p.hidden_size}, true});
  out.push_back({"audio_patch_proj.weight", {p.hidden_size, p.audio_latents_dim}, true});
  out.push_back({"audio_patch_proj.bias", {p.hidden_size}, true});
  out.push_back({"condition_proj.weight", {p.hidden_size, p.text_dim}, false});
  out.push_back({"condition_proj.bias", {p.hidden_size}, false});
  // PRUNED (curve) checkpoints carry `adaln_t_table` INSTEAD of the four time
  // embedder tensors (comfy/ldm/minimax/model.py:428-432). The two forms are
  // mutually exclusive, which is exactly what makes the contract checkable.
  if (p.use_adaln_curves()) {
    out.push_back({"adaln_t_table", {p.adaln_curve_grid, p.time_embed_dim}, true});
  } else {
    out.push_back({"time_embedder.proj_in.weight", {p.time_embed_hidden_size, p.timestep_input_dim}, true});
    out.push_back({"time_embedder.proj_in.bias", {p.time_embed_hidden_size}, true});
    out.push_back({"time_embedder.proj_out.weight", {p.time_embed_dim, p.time_embed_hidden_size}, true});
    out.push_back({"time_embedder.proj_out.bias", {p.time_embed_dim}, true});
  }
  out.push_back({"rope.inv_freq", {p.rope_inv_freq_len}, true});

  auto push_attn_and_mlp = [&](const std::string& prefix) {
    // The checkpoint stores qkv GROUPED per query group; the reorder at load time
    // turns it into [q_all, k_all, v_all] without changing the row count.
    out.push_back({prefix + ".attn.qkv_proj.weight", {3 * inner, p.hidden_size}, false});
    out.push_back({prefix + ".attn.q_norm.weight", {p.attention_head_dim}, false});
    out.push_back({prefix + ".attn.k_norm.weight", {p.attention_head_dim}, false});
    out.push_back({prefix + ".attn.out_proj.weight", {p.hidden_size, inner}, false});
    // fc1 rows split evenly into [gate; up] at load (minimax_h3_transformer.py:497-510).
    out.push_back({prefix + ".mlp.fc1.weight", {2 * p.ffn_hidden_size, p.hidden_size}, false});
    out.push_back({prefix + ".mlp.fc2.weight", {p.hidden_size, p.ffn_hidden_size}, false});
  };

  for (int64_t i = 0; i < p.token_refiner_num_layers; ++i) {
    const std::string prefix = "token_refiner.blocks." + std::to_string(i);
    out.push_back({prefix + ".norm1.weight", {p.hidden_size}, false});
    out.push_back({prefix + ".norm2.weight", {p.hidden_size}, false});
    push_attn_and_mlp(prefix);
  }
  out.push_back({"token_refiner.final_norm.weight", {p.hidden_size}, false});

  for (int64_t i = 0; i < p.num_layers; ++i) {
    const std::string prefix = "blocks." + std::to_string(i);
    out.push_back({prefix + ".norm1.weight", {p.hidden_size}, false});
    out.push_back({prefix + ".norm2.weight", {p.hidden_size}, false});
    push_attn_and_mlp(prefix);
    out.push_back({prefix + ".adaln_proj.linear.weight", {p.adaln_out_features, p.time_embed_dim}, false});
    out.push_back({prefix + ".adaln_proj.linear.bias", {p.adaln_out_features}, false});
  }

  out.push_back({"final_layer.norm.weight", {p.hidden_size}, false});
  out.push_back(
      {"final_layer.adaln_proj.linear.weight", {p.final_adaln_out_features, p.time_embed_dim}, false});
  out.push_back({"final_layer.adaln_proj.linear.bias", {p.final_adaln_out_features}, false});
  out.push_back({"final_layer.video_out.weight", {video_patch_dim, p.hidden_size}, true});
  out.push_back({"final_layer.video_out.bias", {video_patch_dim}, true});
  out.push_back({"final_layer.audio_out.weight", {p.audio_latents_dim, p.hidden_size}, true});
  out.push_back({"final_layer.audio_out.bias", {p.audio_latents_dim}, true});
  return out;
}

// MiniMaxH3TimeEmbedder.forward (minimax_h3_transformer.py:272-285).
std::vector<float> MiniMaxH3SinusoidalTimeEmbed(vt::Device device, const MiniMaxH3DitParams& params,
                                                const MiniMaxH3DitWeights& weights,
                                                const float* timesteps, int64_t m) {
  vt::Queue queue{device, nullptr};
  const int64_t half = params.timestep_input_dim / 2;
  std::vector<float> t_freq(static_cast<size_t>(m * params.timestep_input_dim));
  for (int64_t r = 0; r < m; ++r) {
    for (int64_t i = 0; i < half; ++i) {
      const double freq =
          std::exp(-std::log(10000.0) * static_cast<double>(i) / static_cast<double>(half));
      const double arg = static_cast<double>(timesteps[r]) * freq;
      // Cosine values are concatenated BEFORE sine values.
      t_freq[static_cast<size_t>(r * params.timestep_input_dim + i)] =
          static_cast<float>(std::cos(arg));
      t_freq[static_cast<size_t>(r * params.timestep_input_dim + half + i)] =
          static_cast<float>(std::sin(arg));
    }
  }
  std::vector<float> mid(static_cast<size_t>(m * params.time_embed_hidden_size));
  Linear(queue, t_freq.data(), m, params.timestep_input_dim, weights.time_proj_in_w,
         &weights.time_proj_in_b, mid.data());
  for (float& value : mid) value = Silu(value);
  std::vector<float> out(static_cast<size_t>(m * params.time_embed_dim));
  Linear(queue, mid.data(), m, params.time_embed_hidden_size, weights.time_proj_out_w,
         &weights.time_proj_out_b, out.data());
  return out;
}

// comfy/ldm/minimax/model.py:612-615, verbatim.
std::vector<float> MiniMaxH3AdalnCurveEmbed(const float* table, int64_t grid, int64_t dim,
                                            const float* timesteps, int64_t m) {
  VT_CHECK(grid >= 2, "minimax_h3 curve: adaln_t_table needs at least two rows to interpolate");
  std::vector<float> out(static_cast<size_t>(m * dim));
  for (int64_t r = 0; r < m; ++r) {
    // t in [0,1] -> fractional grid index; out-of-range t clamps to the ends.
    const float t = std::min(1.0f, std::max(0.0f, timesteps[r]));
    const float pos = t * static_cast<float>(grid - 1);
    // The max-clamp keeps t = 1.0 on the LAST interval instead of reading past
    // the table.
    int64_t i0 = static_cast<int64_t>(std::floor(pos));
    if (i0 > grid - 2) i0 = grid - 2;
    if (i0 < 0) i0 = 0;
    const float frac = pos - static_cast<float>(i0);
    const float* lo = table + i0 * dim;
    const float* hi = table + (i0 + 1) * dim;
    for (int64_t d = 0; d < dim; ++d) {
      // torch.lerp(a, b, w) == a + w * (b - a).
      out[static_cast<size_t>(r * dim + d)] = lo[d] + frac * (hi[d] - lo[d]);
    }
  }
  return out;
}

// _reorder_grouped_qkv_to_qkv (minimax_h3_transformer.py:139-168).
std::vector<float> MiniMaxH3ReorderGroupedQkv(const std::vector<float>& weight,
                                              int64_t num_query_groups, int64_t heads_per_group,
                                              int64_t head_dim, int64_t in_features) {
  const int64_t per_group = (heads_per_group + 2) * head_dim;
  const int64_t expected_out = num_query_groups * per_group;
  VT_CHECK(static_cast<int64_t>(weight.size()) == expected_out * in_features,
           "minimax_h3 qkv reorder: weight has incompatible output dim for the grouped layout");
  const int64_t q_rows = num_query_groups * heads_per_group * head_dim;
  const int64_t kv_rows = num_query_groups * head_dim;
  std::vector<float> out(weight.size());
  for (int64_t g = 0; g < num_query_groups; ++g) {
    const int64_t src_base = g * per_group;
    for (int64_t r = 0; r < heads_per_group * head_dim; ++r) {
      const int64_t dst = g * heads_per_group * head_dim + r;
      std::memcpy(out.data() + dst * in_features, weight.data() + (src_base + r) * in_features,
                  static_cast<size_t>(in_features) * sizeof(float));
    }
    for (int64_t r = 0; r < head_dim; ++r) {
      const int64_t src_k = src_base + heads_per_group * head_dim + r;
      const int64_t dst_k = q_rows + g * head_dim + r;
      std::memcpy(out.data() + dst_k * in_features, weight.data() + src_k * in_features,
                  static_cast<size_t>(in_features) * sizeof(float));
      const int64_t src_v = src_k + head_dim;
      const int64_t dst_v = q_rows + kv_rows + g * head_dim + r;
      std::memcpy(out.data() + dst_v * in_features, weight.data() + src_v * in_features,
                  static_cast<size_t>(in_features) * sizeof(float));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

MiniMaxH3DitOutputs MiniMaxH3DitForward(vt::Device device, const MiniMaxH3DitParams& params,
                                        const MiniMaxH3DitWeights& weights,
                                        const MiniMaxH3DitInputs& inputs, DType compute_dtype,
                                        const DitRuntimeLoraState* lora_state) {
  VT_CHECK(device.type == vt::DeviceType::kCPU,
           "minimax_h3: the reference DiT forward is CPU-only; the device-resident "
           "forward is brick H3-2b (.agents/specs/minimax-h3.md)");
  VT_CHECK(compute_dtype == DType::kF32 || compute_dtype == DType::kBF16,
           "minimax_h3: the reference DiT forward computes in f32 (parity) or bf16 "
           "(the production stream policy)");
  // kBF16 reproduces upstream's PRODUCTION dtype policy: the block stream is bf16
  // while the patch projections, the time embedder, and both output heads stay
  // fp32 islands (minimax_h3_transformer.py:85-101). Both dtypes run the SAME
  // code; only the rounding points differ.
  const StreamDtype dt{compute_dtype == DType::kBF16};
  VT_CHECK(static_cast<int64_t>(weights.blocks.size()) == params.num_layers,
           "minimax_h3: block weight count does not match num_layers");
  VT_CHECK(static_cast<int64_t>(weights.refiner.size()) == params.token_refiner_num_layers,
           "minimax_h3: refiner weight count does not match token_refiner_num_layers");
  VT_CHECK(inputs.num_cu_seqlens >= 2, "minimax_h3: packed_seq_params.cu_seqlens is required");

  const bool has_lora = lora_state != nullptr && !lora_state->empty();
  auto lora_find = [&](const std::string& target) -> const DitRuntimeLoraLayer* {
    return has_lora ? lora_state->Find(target) : nullptr;
  };

  vt::Queue q{device, nullptr};
  const int64_t seq_len = inputs.seq_len;
  const int64_t hidden = params.hidden_size;
  const int64_t video_width = params.video_row_width();
  const int64_t m = inputs.num_unique_timesteps;

  // --- RoPE over the full packed sequence (minimax_h3_transformer.py:1040-1041) ---
  const std::vector<float> inv_freq(weights.rope_inv_freq.Ptr<float>(),
                                    weights.rope_inv_freq.Ptr<float>() + params.rope_inv_freq_len);
  const std::vector<float> freqs =
      RopeFreqs(inputs.img_position_ids, seq_len, inv_freq.data(), params.rope_inv_freq_len);

  // --- _embed (minimax_h3_transformer.py:944-984) ---
  std::vector<float> video_rows(static_cast<size_t>(inputs.num_img_pos * video_width));
  for (int64_t r = 0; r < inputs.num_img_pos; ++r) {
    std::memcpy(video_rows.data() + r * video_width, inputs.x + inputs.img_pos[r] * video_width,
                static_cast<size_t>(video_width) * sizeof(float));
  }
  std::vector<float> video_embed(static_cast<size_t>(inputs.num_img_pos * hidden));
  Linear(q, video_rows.data(), inputs.num_img_pos, video_width, weights.video_patch_proj_w,
         &weights.video_patch_proj_b, video_embed.data(),
         lora_find("video_patch_proj.weight"));

  const int64_t audio_width = params.audio_latents_dim;
  std::vector<float> audio_rows(static_cast<size_t>(inputs.num_audio_pos * audio_width));
  for (int64_t r = 0; r < inputs.num_audio_pos; ++r) {
    std::memcpy(audio_rows.data() + r * audio_width,
                inputs.audio_x + inputs.audio_pos[r] * audio_width,
                static_cast<size_t>(audio_width) * sizeof(float));
  }
  std::vector<float> audio_embed(static_cast<size_t>(inputs.num_audio_pos * hidden));
  Linear(q, audio_rows.data(), inputs.num_audio_pos, audio_width, weights.audio_patch_proj_w,
         &weights.audio_patch_proj_b, audio_embed.data(),
         lora_find("audio_patch_proj.weight"));

  std::vector<float> text_embed(static_cast<size_t>(inputs.num_text_pos * hidden));
  {
    // text rows enter as the stream dtype before the BF16 condition projection.
    std::vector<float> text_rows(inputs.prompt_embeds,
                                 inputs.prompt_embeds + inputs.num_text_pos * params.text_dim);
    dt.Apply(text_rows);
    Linear(q, text_rows.data(), inputs.num_text_pos, params.text_dim, weights.condition_proj_w,
           &weights.condition_proj_b, text_embed.data(),
           lora_find("condition_proj.weight"));
    dt.Apply(text_embed);
  }

  // Token refiner: a plain pre-norm stack, no AdaLN and no RoPE
  // (minimax_h3_transformer.py:564-623). It runs on the REPLICATED text rows, so
  // it uses the refiner's own cu_seqlens.
  {
    const int64_t rows = inputs.num_text_pos;
    std::vector<float> normed(text_embed.size()), tmp(text_embed.size());
    for (size_t ri = 0; ri < weights.refiner.size(); ++ri) {
      const MiniMaxH3DitBlockWeights& block = weights.refiner[ri];
      const std::string rp = "token_refiner.blocks." + std::to_string(ri);
      RmsNormRows(text_embed.data(), block.norm1.Ptr<float>(), normed.data(), rows, hidden,
                  params.norm_eps);
      dt.Apply(normed.data(), rows * hidden);
      AttentionForward(q, params, AttnOf(block), normed.data(), rows, nullptr,
                       inputs.refiner_cu_seqlens,
                       static_cast<int>(inputs.num_refiner_cu_seqlens - 1), dt, tmp.data(),
                       lora_find(rp + ".attn.qkv_proj.weight"),
                       lora_find(rp + ".attn.out_proj.weight"));
      for (size_t i = 0; i < text_embed.size(); ++i) text_embed[i] = dt(text_embed[i] + tmp[i]);
      RmsNormRows(text_embed.data(), block.norm2.Ptr<float>(), normed.data(), rows, hidden,
                  params.norm_eps);
      dt.Apply(normed.data(), rows * hidden);
      MlpForward(q, params, block.fc1, block.fc2, normed.data(), rows, dt, tmp.data(),
                 lora_find(rp + ".mlp.fc1.weight"),
                 lora_find(rp + ".mlp.fc2.weight"));
      for (size_t i = 0; i < text_embed.size(); ++i) text_embed[i] = dt(text_embed[i] + tmp[i]);
    }
    std::vector<float> final_normed(text_embed.size());
    RmsNormRows(text_embed.data(), weights.refiner_final_norm.Ptr<float>(), final_normed.data(),
                rows, hidden, params.final_norm_eps);
    dt.Apply(final_normed);
    text_embed.swap(final_normed);
  }

  // index_add_ scatter of the three modality embeddings into the packed stream.
  std::vector<float> stream(static_cast<size_t>(seq_len * hidden), 0.0f);
  auto scatter = [&](const int64_t* pos, int64_t count, const std::vector<float>& src) {
    for (int64_t r = 0; r < count; ++r) {
      float* dst = stream.data() + pos[r] * hidden;
      const float* s = src.data() + r * hidden;
      for (int64_t i = 0; i < hidden; ++i) dst[i] += s[i];
    }
  };
  // The patch projections are fp32 islands; their outputs enter the bf16 stream
  // only at this indexed scatter (minimax_h3_transformer.py:978-981).
  dt.Apply(text_embed);
  dt.Apply(video_embed);
  dt.Apply(audio_embed);
  scatter(inputs.text_pos, inputs.num_text_pos, text_embed);
  scatter(inputs.img_pos, inputs.num_img_pos, video_embed);
  scatter(inputs.audio_pos, inputs.num_audio_pos, audio_embed);
  dt.Apply(stream);

  // --- time embedding (minimax_h3_transformer.py:272-285) ---
  //
  // PRUNED/curve form: the sinusoidal + 2-layer-MLP embedder does not exist in
  // the checkpoint; t_emb is interpolated out of `adaln_t_table`
  // (comfy/ldm/minimax/model.py:610-615). Same t, same [m, time_embed_dim]
  // result, so nothing downstream of this block changes.
  std::vector<float> t_emb(static_cast<size_t>(m * params.time_embed_dim));
  if (params.use_adaln_curves()) {
    VT_CHECK(weights.adaln_t_table.data != nullptr,
             "minimax_h3: a curve-form checkpoint must bind adaln_t_table");
    t_emb = MiniMaxH3AdalnCurveEmbed(weights.adaln_t_table.Ptr<float>(), params.adaln_curve_grid,
                                     params.time_embed_dim, inputs.unique_timesteps, m);
  } else {
    t_emb = MiniMaxH3SinusoidalTimeEmbed(device, params, weights, inputs.unique_timesteps, m);
  }

  // combined_indices = inverse_indices * modality_num + token_tags.clamp(min=0)
  // (minimax_h3_transformer.py:1057).
  std::vector<int64_t> combined(static_cast<size_t>(seq_len));
  for (int64_t i = 0; i < seq_len; ++i) {
    const int64_t tag = inputs.token_tags[i] < 0 ? 0 : inputs.token_tags[i];
    combined[static_cast<size_t>(i)] = inputs.inverse_indices[i] * kMiniMaxH3AdalnModalityNum + tag;
  }

  // --- the DiT block stack (minimax_h3_transformer.py:645-688) ---
  const int num_reqs = static_cast<int>(inputs.num_cu_seqlens - 1);
  std::vector<float> normed(stream.size()), tmp(stream.size());
  for (size_t bi = 0; bi < weights.blocks.size(); ++bi) {
    const MiniMaxH3DitBlockWeights& block = weights.blocks[bi];
    const std::string bp = "blocks." + std::to_string(bi);
    const std::vector<float> projected =
        AdalnProject(q, t_emb.data(), m, params.time_embed_dim, block.adaln_w, block.adaln_b, dt,
                     !params.use_adaln_curves(),
                     lora_find(bp + ".adaln_proj.linear.weight"));
    const std::vector<float> shift_msa =
        AdalnChunk(projected, m, kMiniMaxH3AdalnModalityNum, hidden, 6, 0);
    const std::vector<float> scale_msa =
        AdalnChunk(projected, m, kMiniMaxH3AdalnModalityNum, hidden, 6, 1);
    const std::vector<float> gate_msa =
        AdalnChunk(projected, m, kMiniMaxH3AdalnModalityNum, hidden, 6, 2);
    const std::vector<float> shift_mlp =
        AdalnChunk(projected, m, kMiniMaxH3AdalnModalityNum, hidden, 6, 3);
    const std::vector<float> scale_mlp =
        AdalnChunk(projected, m, kMiniMaxH3AdalnModalityNum, hidden, 6, 4);
    const std::vector<float> gate_mlp =
        AdalnChunk(projected, m, kMiniMaxH3AdalnModalityNum, hidden, 6, 5);

    RmsNormRows(stream.data(), block.norm1.Ptr<float>(), normed.data(), seq_len, hidden,
                params.norm_eps);
    dt.Apply(normed);
    ModulateScaleShift(normed.data(), seq_len, hidden, shift_msa.data(), scale_msa.data(),
                       combined.data());
    dt.Apply(normed);  // _modulate_scale_shift casts its result to the stream dtype
    AttentionForward(q, params, AttnOf(block), normed.data(), seq_len, freqs.data(),
                     inputs.cu_seqlens, num_reqs, dt, tmp.data(),
                     lora_find(bp + ".attn.qkv_proj.weight"),
                     lora_find(bp + ".attn.out_proj.weight"));
    ModulateGate(stream.data(), seq_len, hidden, gate_msa.data(), tmp.data(), combined.data());
    dt.Apply(stream);

    RmsNormRows(stream.data(), block.norm2.Ptr<float>(), normed.data(), seq_len, hidden,
                params.norm_eps);
    dt.Apply(normed);
    ModulateScaleShift(normed.data(), seq_len, hidden, shift_mlp.data(), scale_mlp.data(),
                       combined.data());
    dt.Apply(normed);
    MlpForward(q, params, block.fc1, block.fc2, normed.data(), seq_len, dt, tmp.data(),
               lora_find(bp + ".mlp.fc1.weight"),
               lora_find(bp + ".mlp.fc2.weight"));
    ModulateGate(stream.data(), seq_len, hidden, gate_mlp.data(), tmp.data(), combined.data());
    dt.Apply(stream);
  }

  // --- final layer (minimax_h3_transformer.py:724-743) ---
  const std::vector<float> final_projected =
      AdalnProject(q, t_emb.data(), m, params.time_embed_dim, weights.final_adaln_w,
                   weights.final_adaln_b, dt, !params.use_adaln_curves(),
                   lora_find("final_layer.adaln_proj.linear.weight"));
  const std::vector<float> final_shift = AdalnChunk(final_projected, m, 1, hidden, 2, 0);
  const std::vector<float> final_scale = AdalnChunk(final_projected, m, 1, hidden, 2, 1);
  RmsNormRows(stream.data(), weights.final_norm.Ptr<float>(), normed.data(), seq_len, hidden,
              params.final_norm_eps);
  dt.Apply(normed);
  // The final layer is single-modality, so it indexes by inverse_indices directly.
  std::vector<int64_t> inverse(inputs.inverse_indices, inputs.inverse_indices + seq_len);
  ModulateScaleShift(normed.data(), seq_len, hidden, final_shift.data(), final_scale.data(),
                     inverse.data());
  // Cast UP before both output heads: they are fp32 islands.
  dt.Apply(normed);

  std::vector<float> video_all(static_cast<size_t>(seq_len * video_width));
  Linear(q, normed.data(), seq_len, hidden, weights.video_out_w, &weights.video_out_b,
         video_all.data(), lora_find("final_layer.video_out.weight"));
  std::vector<float> audio_all(static_cast<size_t>(seq_len * audio_width));
  Linear(q, normed.data(), seq_len, hidden, weights.audio_out_w, &weights.audio_out_b,
         audio_all.data(), lora_find("final_layer.audio_out.weight"));

  // Select the inference-output rows, then zero the pinned condition rows
  // (minimax_h3_transformer.py:1087-1101).
  MiniMaxH3DitOutputs out;
  out.video_logits.resize(static_cast<size_t>(inputs.num_infer_out_pos * video_width));
  for (int64_t r = 0; r < inputs.num_infer_out_pos; ++r) {
    std::memcpy(out.video_logits.data() + r * video_width,
                video_all.data() + inputs.infer_out_pos[r] * video_width,
                static_cast<size_t>(video_width) * sizeof(float));
  }
  out.audio_logits.resize(static_cast<size_t>(inputs.num_audio_pos * audio_width));
  for (int64_t r = 0; r < inputs.num_audio_pos; ++r) {
    std::memcpy(out.audio_logits.data() + r * audio_width,
                audio_all.data() + inputs.audio_pos[r] * audio_width,
                static_cast<size_t>(audio_width) * sizeof(float));
  }
  if (!inputs.skip_mask_out_condition) {
    VT_CHECK(inputs.update_mask != nullptr, "minimax_h3: update_mask is required");
    for (int64_t r = 0; r < inputs.num_infer_out_pos; ++r) {
      if (inputs.update_mask[r]) continue;
      std::fill_n(out.video_logits.data() + r * video_width, video_width, 0.0f);
    }
    if (inputs.audio_update_mask != nullptr) {
      for (int64_t r = 0; r < inputs.num_audio_pos; ++r) {
        if (inputs.audio_update_mask[r]) continue;
        std::fill_n(out.audio_logits.data() + r * audio_width, audio_width, 0.0f);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Denoise loop (denoise_loop.py:129-239)
// ---------------------------------------------------------------------------

MiniMaxH3DenoiseResult MiniMaxH3DenoiseLoop(
    vt::Device device, const MiniMaxH3DitParams& params, const MiniMaxH3DitWeights& weights,
    const MiniMaxH3DenoiseBranch& branch, const std::vector<float>& initial_video_rows,
    const std::vector<float>& initial_audio_rows, const std::vector<float>& keyframe_cond_rows,
    const std::vector<float>& audio_ref_rows, const std::vector<double>& sigmas_video,
    const std::vector<double>& sigmas_audio, DType compute_dtype,
    const MiniMaxH3DitDeviceWeights* prestaged,
    const DitRuntimeLoraState* lora_state) {
  VT_CHECK(sigmas_video.size() == sigmas_audio.size(),
           "minimax_h3 denoise: video/audio sigma schedules must have equal length");
  VT_CHECK(sigmas_video.size() >= 2, "minimax_h3 denoise: sigma schedules need at least 2 entries");

  // DEVICE PATH. On a non-CPU device the per-step forward runs device-resident, and
  // the weights are staged ONCE here rather than per step — a 50-step loop that
  // re-uploaded ~16 GB every iteration would be dominated by the transfer. This is
  // exactly what StageMiniMaxH3DitWeights exists for.
  const bool on_device = device.type != vt::DeviceType::kCPU;
  // VT_H3_PROGRESS=1 traces the phases to stderr. A 50-step loop over a real
  // checkpoint spends minutes in ONE of {weight staging, per-step forward}, and
  // guessing which is exactly the trap this avoids.
  const bool trace = std::getenv("VT_H3_PROGRESS") != nullptr;
  auto now = [] { return std::chrono::steady_clock::now(); };
  vt::Queue device_queue{device, nullptr};
  MiniMaxH3DitDeviceWeights staged;
  if (on_device) {
    vt::Backend& backend = vt::GetBackend(device.type);
    device_queue = backend.CreateQueue();
    const auto t0 = now();
    if (prestaged == nullptr) {
      staged = StageMiniMaxH3DitWeights(device_queue, params, weights, compute_dtype);
    }
    if (trace) {
      std::fprintf(stderr, "[h3] staged weights to device in %.1f s\n",
                   std::chrono::duration<double>(now() - t0).count());
    }
  } else if (trace) {
    std::fprintf(stderr, "[h3] CPU reference forward (device.type == kCPU)\n");
  }

  const MiniMaxH3PackedSequence& packed = branch.packed;
  const int64_t seq_len = packed.seq_len;
  const int64_t video_width = params.video_row_width();
  const int64_t audio_width = params.audio_latents_dim;
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());

  VT_CHECK(static_cast<int64_t>(initial_video_rows.size()) == num_img * video_width,
           "minimax_h3 denoise: initial video rows do not match the layout");
  VT_CHECK(static_cast<int64_t>(initial_audio_rows.size()) == num_audio * audio_width,
           "minimax_h3 denoise: initial audio rows do not match the layout");

  std::vector<uint8_t> audio_update = packed.audio_update_mask;
  if (audio_update.empty()) audio_update.assign(static_cast<size_t>(num_audio), 1);

  MiniMaxH3DenoiseResult result;
  result.video_rows = initial_video_rows;
  result.audio_rows = initial_audio_rows;

  // Pin the conditioning anchors before the first step.
  int64_t cond_index = 0;
  for (int64_t r = 0; r < num_img; ++r) {
    if (packed.update_mask[static_cast<size_t>(r)]) continue;
    VT_CHECK(static_cast<int64_t>(keyframe_cond_rows.size()) >= (cond_index + 1) * video_width,
             "minimax_h3 denoise: keyframe_cond_rows shorter than the layout's condition rows");
    std::memcpy(result.video_rows.data() + r * video_width,
                keyframe_cond_rows.data() + cond_index * video_width,
                static_cast<size_t>(video_width) * sizeof(float));
    ++cond_index;
  }
  int64_t audio_ref_index = 0;
  for (int64_t r = 0; r < num_audio; ++r) {
    if (audio_update[static_cast<size_t>(r)]) continue;
    VT_CHECK(static_cast<int64_t>(audio_ref_rows.size()) >= (audio_ref_index + 1) * audio_width,
             "minimax_h3 denoise: audio_ref_rows shorter than the layout's reference rows");
    std::memcpy(result.audio_rows.data() + r * audio_width,
                audio_ref_rows.data() + audio_ref_index * audio_width,
                static_cast<size_t>(audio_width) * sizeof(float));
    ++audio_ref_index;
  }

  // DIAGNOSTIC (env-gated, byte-identical when unset): VT_H3_TRACE_MOTION prints
  // per-step velocity + latent-motion stats to stderr; VT_H3_DUMP_DIR writes the
  // initial and final video latent rows as raw f32 so runs can be byte-compared.
  // The rectified-flow Euler integration telescopes to (sigma0 - sigmaN) * v ~= v,
  // so a velocity that does NOT evolve across steps yields a step-count-INVARIANT
  // result -- this trace measures exactly that (see H3 render-coherence bisection).
  const bool trace_motion = std::getenv("VT_H3_TRACE_MOTION") != nullptr;
  const char* dump_dir = std::getenv("VT_H3_DUMP_DIR");
  auto rms_absmax = [](const float* p, int64_t n, double* rms, double* absmax, double* mean) {
    double s2 = 0.0, amax = 0.0, sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const double v = static_cast<double>(p[i]);
      s2 += v * v;
      sum += v;
      if (std::fabs(v) > amax) amax = std::fabs(v);
    }
    const double denom = n > 0 ? static_cast<double>(n) : 1.0;
    *rms = std::sqrt(s2 / denom);
    *absmax = amax;
    *mean = sum / denom;
  };
  auto dump_rows = [&](const char* name, const std::vector<float>& v) {
    if (dump_dir == nullptr) return;
    std::string path = std::string(dump_dir) + "/" + name;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return;
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
  };
  dump_rows("init_video_rows.f32", initial_video_rows);
  if (trace_motion) {
    double r = 0, a = 0, mn = 0;
    rms_absmax(initial_video_rows.data(), static_cast<int64_t>(initial_video_rows.size()), &r, &a,
               &mn);
    std::fprintf(stderr, "[h3-motion] INIT video_rows n=%lld rms=%.6g absmax=%.6g mean=%.6g\n",
                 static_cast<long long>(initial_video_rows.size()), r, a, mn);
  }

  const int64_t num_steps = static_cast<int64_t>(sigmas_video.size()) - 1;
  for (int64_t step = 0; step < num_steps; ++step) {
    const double s_v = sigmas_video[static_cast<size_t>(step)];
    const double s_v_next = sigmas_video[static_cast<size_t>(step + 1)];
    const double s_a = sigmas_audio[static_cast<size_t>(step)];
    const double s_a_next = sigmas_audio[static_cast<size_t>(step + 1)];
    const double t_v = 1.0 - s_v, t_a = 1.0 - s_a;
    const double imgvid_cond_t = std::max(t_v, kMiniMaxH3ImgVidCondTimestep);
    const double audio_ref_cond_t = std::max(t_a, kMiniMaxH3AudioRefCondTimestep);

    // MiniMaxH3DenoiseBranch.forward_kwargs (denoise_loop.py:91-126): scatter the
    // current rows back into the packed stream and rebuild the per-row timesteps.
    std::vector<float> x(static_cast<size_t>(seq_len * video_width), 0.0f);
    for (int64_t r = 0; r < num_img; ++r) {
      std::memcpy(x.data() + packed.img_pos[static_cast<size_t>(r)] * video_width,
                  result.video_rows.data() + r * video_width,
                  static_cast<size_t>(video_width) * sizeof(float));
    }
    std::vector<float> audio_x(static_cast<size_t>(seq_len * audio_width), 0.0f);
    for (int64_t r = 0; r < num_audio; ++r) {
      std::memcpy(audio_x.data() + packed.audio_pos[static_cast<size_t>(r)] * audio_width,
                  result.audio_rows.data() + r * audio_width,
                  static_cast<size_t>(audio_width) * sizeof(float));
    }

    // Non-media rows (text and padding) inherit the current VIDEO timestep.
    std::vector<float> timesteps(static_cast<size_t>(seq_len), static_cast<float>(t_v));
    for (int64_t r = 0; r < num_img; ++r) {
      timesteps[static_cast<size_t>(packed.img_pos[static_cast<size_t>(r)])] =
          packed.update_mask[static_cast<size_t>(r)] ? static_cast<float>(t_v)
                                                     : static_cast<float>(imgvid_cond_t);
    }
    for (int64_t r = 0; r < num_audio; ++r) {
      timesteps[static_cast<size_t>(packed.audio_pos[static_cast<size_t>(r)])] =
          audio_update[static_cast<size_t>(r)] ? static_cast<float>(t_a)
                                               : static_cast<float>(audio_ref_cond_t);
    }

    // torch.unique(sorted=True, return_inverse=True).
    std::vector<float> unique = timesteps;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    std::vector<int64_t> inverse(static_cast<size_t>(seq_len));
    for (int64_t i = 0; i < seq_len; ++i) {
      inverse[static_cast<size_t>(i)] = static_cast<int64_t>(
          std::lower_bound(unique.begin(), unique.end(), timesteps[static_cast<size_t>(i)]) -
          unique.begin());
    }

    MiniMaxH3DitInputs in;
    in.seq_len = seq_len;
    in.x = x.data();
    in.audio_x = audio_x.data();
    in.img_position_ids = packed.img_position_ids.data();
    in.unique_timesteps = unique.data();
    in.num_unique_timesteps = static_cast<int64_t>(unique.size());
    in.inverse_indices = inverse.data();
    in.token_tags = branch.token_tags.data();
    in.prompt_embeds = branch.text_embeddings.data();
    in.img_pos = packed.img_pos.data();
    in.num_img_pos = num_img;
    in.audio_pos = packed.audio_pos.data();
    in.num_audio_pos = num_audio;
    in.text_pos = packed.text_pos.data();
    in.num_text_pos = static_cast<int64_t>(packed.text_pos.size());
    in.infer_out_pos = packed.img_pos.data();
    in.num_infer_out_pos = num_img;
    in.update_mask = packed.update_mask.data();
    in.audio_update_mask = packed.audio_update_mask.empty() ? nullptr : audio_update.data();
    in.cu_seqlens = packed.cu_seqlens.data();
    in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
    const int32_t text_len = static_cast<int32_t>(packed.text_pos.size());
    const std::vector<int32_t> refiner_cu = {0, text_len, text_len};
    in.refiner_cu_seqlens = refiner_cu.data();
    in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());

    // DIAGNOSTIC (env-gated, byte-identical when unset): VT_H3_DUMP_INPUTS=<dir>
    // dumps every DiT input at STEP 0 as raw little-endian binary plus a text
    // manifest, so the real-scale driver's DiT inputs can be diffed EXACTLY against
    // upstream pipeline_minimax_h3.py's construction (packed layout, fp64 position
    // grid, per-token modality tags, per-token timestep -> unique/inverse ->
    // combined AdaLN index) and the encoder conditioning statistically. This is the
    // S1 surface #70/#74 never isolated: the ladder fed RANDOM inputs; the real
    // render's INPUTS are the untested corner. Only step 0 (the layout and the
    // timestep partition are the same shape every step).
    if (step == 0) {
      if (const char* dump_inputs_dir = std::getenv("VT_H3_DUMP_INPUTS")) {
        const std::string dir(dump_inputs_dir);
        auto wr = [&](const char* name, const void* p, size_t bytes) {
          std::FILE* f = std::fopen((dir + "/" + name).c_str(), "wb");
          if (f == nullptr) return;
          std::fwrite(p, 1, bytes, f);
          std::fclose(f);
        };
        // per-token pre-unique timesteps (before torch.unique) and combined AdaLN idx
        std::vector<int64_t> combined_dump(static_cast<size_t>(seq_len));
        for (int64_t i = 0; i < seq_len; ++i) {
          const int64_t tag = branch.token_tags[static_cast<size_t>(i)] < 0
                                  ? 0
                                  : branch.token_tags[static_cast<size_t>(i)];
          combined_dump[static_cast<size_t>(i)] =
              inverse[static_cast<size_t>(i)] * kMiniMaxH3AdalnModalityNum + tag;
        }
        wr("timesteps_pretoken.f32", timesteps.data(), timesteps.size() * sizeof(float));
        wr("unique_timesteps.f32", unique.data(), unique.size() * sizeof(float));
        wr("inverse_indices.i64", inverse.data(), inverse.size() * sizeof(int64_t));
        wr("combined_indices.i64", combined_dump.data(), combined_dump.size() * sizeof(int64_t));
        wr("token_tags.i64", branch.token_tags.data(),
           branch.token_tags.size() * sizeof(int64_t));
        wr("img_position_ids.f64", packed.img_position_ids.data(),
           packed.img_position_ids.size() * sizeof(double));
        wr("input_ids.i64", packed.input_ids.data(), packed.input_ids.size() * sizeof(int64_t));
        wr("image_mask.u8", packed.image_mask.data(), packed.image_mask.size());
        wr("audio_mask.u8", packed.audio_mask.data(), packed.audio_mask.size());
        wr("img_pos.i64", packed.img_pos.data(), packed.img_pos.size() * sizeof(int64_t));
        wr("audio_pos.i64", packed.audio_pos.data(), packed.audio_pos.size() * sizeof(int64_t));
        wr("text_pos.i64", packed.text_pos.data(), packed.text_pos.size() * sizeof(int64_t));
        wr("update_mask.u8", packed.update_mask.data(), packed.update_mask.size());
        wr("audio_update_mask.u8", audio_update.data(), audio_update.size());
        wr("cu_seqlens.i32", packed.cu_seqlens.data(),
           packed.cu_seqlens.size() * sizeof(int32_t));
        wr("document_id.i64", packed.document_id.data(),
           packed.document_id.size() * sizeof(int64_t));
        wr("sigmas_video.f64", sigmas_video.data(), sigmas_video.size() * sizeof(double));
        wr("sigmas_audio.f64", sigmas_audio.data(), sigmas_audio.size() * sizeof(double));
        wr("prompt_embeds.f32", branch.text_embeddings.data(),
           branch.text_embeddings.size() * sizeof(float));
        std::FILE* mf = std::fopen((dir + "/manifest.txt").c_str(), "wb");
        if (mf != nullptr) {
          std::fprintf(mf,
                       "seq_len=%lld\nnum_unique_timesteps=%lld\nnum_img_pos=%lld\n"
                       "num_audio_pos=%lld\nnum_text_pos=%lld\ntext_dim=%lld\n"
                       "video_row_width=%lld\naudio_latents_dim=%lld\nnum_steps=%lld\n"
                       "s_v0=%.17g\ns_a0=%.17g\nt_v0=%.17g\nt_a0=%.17g\n"
                       "imgvid_cond_t0=%.17g\naudio_ref_cond_t0=%.17g\n"
                       "prompt_embeds_rows=%lld\n",
                       static_cast<long long>(seq_len),
                       static_cast<long long>(unique.size()),
                       static_cast<long long>(num_img),
                       static_cast<long long>(num_audio),
                       static_cast<long long>(packed.text_pos.size()),
                       static_cast<long long>(params.text_dim),
                       static_cast<long long>(video_width),
                       static_cast<long long>(audio_width),
                       static_cast<long long>(num_steps), s_v, s_a, t_v, t_a, imgvid_cond_t,
                       audio_ref_cond_t,
                       static_cast<long long>(branch.text_embeddings.size() /
                                              (params.text_dim > 0 ? params.text_dim : 1)));
          std::fclose(mf);
        }
        std::fprintf(stderr, "[h3-dump-inputs] wrote DiT step-0 inputs to %s (seq_len=%lld)\n",
                     dir.c_str(), static_cast<long long>(seq_len));
        std::fflush(stderr);
      }
    }

    const auto step_t0 = now();
    const MiniMaxH3DitOutputs velocity =
        on_device ? MiniMaxH3DitForwardDevice(
                        device_queue, params,
                        prestaged != nullptr ? prestaged->weights : staged.weights, in,
                        compute_dtype, lora_state)
                  : MiniMaxH3DitForward(device, params, weights, in, compute_dtype, lora_state);
    if (trace) {
      std::fprintf(stderr, "[h3] step %d/%d forward %.2f s (seq_len=%lld)\n",
                   static_cast<int>(step + 1), static_cast<int>(sigmas_video.size() - 1),
                   std::chrono::duration<double>(now() - step_t0).count(),
                   static_cast<long long>(seq_len));
      std::fflush(stderr);
    }

    // Chain only the TARGET rows; pinned rows are reset to their anchors after
    // every step (denoise_loop.py:210-235).
    auto advance = [&](std::vector<float>& rows, const std::vector<float>& v,
                       const std::vector<uint8_t>& mask, int64_t count, int64_t width, double t,
                       double sigma_curr, double sigma_next) {
      std::vector<float> target, target_v;
      target.reserve(static_cast<size_t>(count * width));
      target_v.reserve(static_cast<size_t>(count * width));
      for (int64_t r = 0; r < count; ++r) {
        if (!mask[static_cast<size_t>(r)]) continue;
        target.insert(target.end(), rows.begin() + r * width, rows.begin() + (r + 1) * width);
        target_v.insert(target_v.end(), v.begin() + r * width, v.begin() + (r + 1) * width);
      }
      const std::vector<float> x0 = MiniMaxH3RfVToX0(target, target_v, t);
      const std::vector<float> stepped = MiniMaxH3EulerEta0Step(target, x0, sigma_curr, sigma_next);
      int64_t k = 0;
      for (int64_t r = 0; r < count; ++r) {
        if (!mask[static_cast<size_t>(r)]) continue;
        std::memcpy(rows.data() + r * width, stepped.data() + k * width,
                    static_cast<size_t>(width) * sizeof(float));
        ++k;
      }
    };

    std::vector<float> motion_prev;
    if (trace_motion) motion_prev = result.video_rows;
    advance(result.video_rows, velocity.video_logits, packed.update_mask, num_img, video_width, t_v,
            s_v, s_v_next);
    cond_index = 0;
    for (int64_t r = 0; r < num_img; ++r) {
      if (packed.update_mask[static_cast<size_t>(r)]) continue;
      std::memcpy(result.video_rows.data() + r * video_width,
                  keyframe_cond_rows.data() + cond_index * video_width,
                  static_cast<size_t>(video_width) * sizeof(float));
      ++cond_index;
    }

    advance(result.audio_rows, velocity.audio_logits, audio_update, num_audio, audio_width, t_a,
            s_a, s_a_next);
    audio_ref_index = 0;
    for (int64_t r = 0; r < num_audio; ++r) {
      if (audio_update[static_cast<size_t>(r)]) continue;
      std::memcpy(result.audio_rows.data() + r * audio_width,
                  audio_ref_rows.data() + audio_ref_index * audio_width,
                  static_cast<size_t>(audio_width) * sizeof(float));
      ++audio_ref_index;
    }

    if (trace_motion) {
      double v_rms = 0, v_amax = 0, v_mean = 0;
      rms_absmax(velocity.video_logits.data(),
                 static_cast<int64_t>(velocity.video_logits.size()), &v_rms, &v_amax, &v_mean);
      // Motion this step, over the UPDATE (denoise-target) rows only.
      double d2 = 0.0;
      int64_t dn = 0;
      for (int64_t rr = 0; rr < num_img; ++rr) {
        if (!packed.update_mask[static_cast<size_t>(rr)]) continue;
        for (int64_t c = 0; c < video_width; ++c) {
          const size_t idx = static_cast<size_t>(rr * video_width + c);
          const double diff =
              static_cast<double>(result.video_rows[idx]) - static_cast<double>(motion_prev[idx]);
          d2 += diff * diff;
          ++dn;
        }
      }
      const double drows_rms = std::sqrt(d2 / (dn > 0 ? static_cast<double>(dn) : 1.0));
      double rows_rms = 0, rows_amax = 0, rows_mean = 0;
      rms_absmax(result.video_rows.data(), static_cast<int64_t>(result.video_rows.size()),
                 &rows_rms, &rows_amax, &rows_mean);
      std::fprintf(stderr,
                   "[h3-motion] step %lld/%lld sig_v=%.5f->%.5f v_rms=%.6g v_amax=%.6g "
                   "v_mean=%.6g drows_rms=%.6g rows_rms=%.6g rows_amax=%.6g\n",
                   static_cast<long long>(step + 1), static_cast<long long>(num_steps),
                   sigmas_video[static_cast<size_t>(step)],
                   sigmas_video[static_cast<size_t>(step + 1)], v_rms, v_amax, v_mean, drows_rms,
                   rows_rms, rows_amax);
      std::fflush(stderr);
    }
  }

  dump_rows("final_video_rows.f32", result.video_rows);
  if (trace_motion) {
    double r = 0, a = 0, mn = 0;
    rms_absmax(result.video_rows.data(), static_cast<int64_t>(result.video_rows.size()), &r, &a,
               &mn);
    // Total displacement from the initial noise, over all rows.
    double d2 = 0.0;
    const size_t n = std::min(result.video_rows.size(), initial_video_rows.size());
    for (size_t i = 0; i < n; ++i) {
      const double diff =
          static_cast<double>(result.video_rows[i]) - static_cast<double>(initial_video_rows[i]);
      d2 += diff * diff;
    }
    const double disp_rms = std::sqrt(d2 / (n > 0 ? static_cast<double>(n) : 1.0));
    std::fprintf(stderr,
                 "[h3-motion] FINAL video_rows rms=%.6g absmax=%.6g mean=%.6g "
                 "disp_from_init_rms=%.6g\n",
                 r, a, mn, disp_rms);
    std::fflush(stderr);
  }
  return result;
}

}  // namespace vllm
