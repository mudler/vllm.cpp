// H3-Encoder — the DEVICE-resident text tower, running from ggml blocks.
//
// This is what makes real text conditioning possible on a box that cannot hold the
// encoder in f32: the tower is 32B (~128 GB expanded, ~13.2 GiB kept quantized), so
// the projections stay in their blocks and go through vt::MatmulBT, which dispatches
// kMatmulBTQuant on a block-typed weight. No dequantization anywhere on this path.
//
// Structure is 1:1 with MiniMaxH3EncoderTextForward (the gated host reference), and
// the three H3 deltas are preserved:
//   * LAYER TRUNCATION to min(num_hidden_layers, selected_layer) — 50, not 64;
//   * the output is UNNORMALIZED (no final RMSNorm), which is why `norm.weight` is
//     not even loaded;
//   * DeepStack injection is the caller's business (text-only prompts have none).
//
// The M-RoPE looks like it needs a bespoke kernel and does not. Upstream builds
// `emb = cat(freqs, freqs)`, so cos and sin REPEAT across the two halves — exactly
// vt::RopeFromCache's [cos_half | sin_half] layout. Only the ANGLES are unusual
// (three axes interleaved as [THW THW ...] over the frequency slots), and those are
// computed host-side once per prompt.
#include "vllm/model_executor/models/minimax_h3.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {

const vt::Tensor& MiniMaxH3EncoderDeviceWeights::Get(const std::string& name) const {
  const auto it = views.find(name);
  VT_CHECK(it != views.end(), "minimax_h3 encoder device: missing tensor (by name)");
  return it->second;
}

MiniMaxH3EncoderDeviceWeights StageMiniMaxH3EncoderWeights(
    vt::Queue& queue, const MiniMaxH3EncoderQuantWeights& host) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  MiniMaxH3EncoderDeviceWeights out;
  for (const auto& kv : host.views) {
    const vt::Tensor& src = kv.second;
    // Block-quant tensors go up VERBATIM; f32 norms go up as-is. Either way the
    // GEMM must see DEVICE memory — a host-byte view reads as zeros on the GPU and
    // a CPU-only gate cannot catch it.
    const size_t bytes = vt::IsBlockQuant(src.dtype)
                             ? static_cast<size_t>(src.shape[0]) *
                                   vt::RowSizeBytes(src.dtype, src.shape[1])
                             : static_cast<size_t>(src.Numel()) * vt::SizeOf(src.dtype);
    void* p = backend.Alloc(bytes);
    std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
    backend.Copy(queue, p, src.data, bytes);
    std::vector<int64_t> shape(src.shape, src.shape + src.rank);
    out.views[kv.first] = dense_attn::MakeTensor(p, src.dtype, queue.device, shape);
    out.storage.push_back(std::move(owner));
  }
  backend.Synchronize(queue);
  return out;
}

namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using vt::DType;
using vt::Tensor;

// The M-RoPE cos/sin cache in vt::RopeFromCache's layout: [seq, head_dim] with the
// first half cos and the second half sin. Upstream's cat(freqs, freqs) is what makes
// that layout sufficient — the halves repeat, so one entry per frequency slot serves
// both. The axis interleave ([TTT HHH WWW] chunks reshuffled to [THW THW ...]) is
// transcribed from the host reference; getting it wrong rotates the wrong axis and
// silently mis-positions every token.
std::vector<float> BuildEncoderRopeCache(const int64_t* positions, int64_t seq,
                                         int64_t head_dim, double rope_theta,
                                         const std::vector<int64_t>& mrope_section) {
  const int64_t half = head_dim / 2;
  std::vector<float> cache(static_cast<size_t>(seq * head_dim));
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t i = 0; i < half; ++i) {
      int64_t axis = 0;
      if (i % 3 == 1 && i < 3 * mrope_section[1]) {
        axis = 1;
      } else if (i % 3 == 2 && i < 3 * mrope_section[2]) {
        axis = 2;
      }
      const double inv_freq =
          1.0 / std::pow(rope_theta, static_cast<double>(2 * i) / static_cast<double>(head_dim));
      const double angle = static_cast<double>(positions[axis * seq + s]) * inv_freq;
      cache[static_cast<size_t>(s * head_dim + i)] = static_cast<float>(std::cos(angle));
      cache[static_cast<size_t>(s * head_dim + half + i)] = static_cast<float>(std::sin(angle));
    }
  }
  return cache;
}

}  // namespace

std::vector<float> MiniMaxH3EncoderTextForwardDevice(
    vt::Queue& queue, const MiniMaxH3EncoderConfig& config,
    const MiniMaxH3EncoderDeviceWeights& weights, const std::vector<float>& inputs_embeds,
    const int64_t* positions, int64_t seq, const uint8_t* visual_pos_mask,
    const std::vector<std::vector<float>>& deepstack) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};

  const int64_t hidden = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t kv_heads = config.num_key_value_heads;
  const int64_t head_dim = config.head_dim;
  const int64_t q_width = heads * head_dim;
  const int64_t kv_width = kv_heads * head_dim;
  VT_CHECK(static_cast<int64_t>(inputs_embeds.size()) == seq * hidden,
           "minimax_h3 encoder device: inputs_embeds size does not match [seq, hidden]");
  VT_CHECK(heads % kv_heads == 0,
           "minimax_h3 encoder device: heads must be a multiple of kv_heads");

  const std::vector<float> rope_host =
      BuildEncoderRopeCache(positions, seq, head_dim, config.rope_theta, config.mrope_section);
  DBuf rope(d, DType::kF32, {seq, head_dim}, rope_host.data());
  std::vector<int32_t> arange(static_cast<size_t>(seq));
  for (int64_t i = 0; i < seq; ++i) arange[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  DBuf rope_pos(d, DType::kI32, {seq}, arange.data());

  // The whole prompt is ONE causal document.
  const std::vector<int32_t> cu_seqlens = {0, static_cast<int32_t>(seq)};

  DBuf h(d, DType::kF32, {seq, hidden}, inputs_embeds.data());
  DBuf normed(d, DType::kF32, {seq, hidden});
  DBuf attn_out(d, DType::kF32, {seq, hidden});

  vt::RmsNormArgs norm_args;
  norm_args.eps = static_cast<float>(config.rms_norm_eps);

  // GEMM WEIGHT ACCESS. A block-quant weight (the GGUF arm) goes to MatmulBT
  // untouched — it dispatches kMatmulBTQuant, which takes the f32 activation as
  // it is. A BF16 weight (the unquantized safetensors arm) cannot: MatmulBT
  // requires BOTH operands in the same dtype and these activations are f32.
  //
  // So a bf16 weight is WIDENED here, immediately before its GEMM, into a scratch
  // buffer keyed by element count and reused across all 50 layers. bf16 -> f32 is
  // EXACT, so the GEMM sees bit-identical inputs to what an f32-staged tower would
  // have given it — the widening is a residency trick, not a numerics one. It has
  // to be: the 50 layers H3 runs are 48.8 GiB in bf16 and 97.5 GiB in f32, and the
  // pool is 122 GiB shared with the host. Peak cost is ONE layer's projections
  // (~2 GiB), not the model.
  std::map<int64_t, DBuf> widen_scratch;
  auto weight = [&](const std::string& name) -> Tensor {
    const Tensor& w = weights.Get(name);
    if (w.dtype != DType::kBF16) return w;
    const int64_t numel = w.Numel();
    auto it = widen_scratch.find(numel);
    if (it == widen_scratch.end()) {
      it = widen_scratch.emplace(numel, DBuf(d, DType::kF32, {numel})).first;
    }
    // Same allocation every layer, and the casts and GEMMs are enqueued on ONE
    // stream, so layer L's GEMM has consumed it before layer L+1's cast writes it.
    Tensor flat_src = dense_attn::Reshape(w, {numel});
    vt::CastF32(d.q, it->second.t(), flat_src);
    return dense_attn::Reshape(it->second.t(),
                               std::vector<int64_t>(w.shape, w.shape + w.rank));
  };

  const int64_t num_layers =
      MiniMaxH3EncoderNumLayers(config.num_hidden_layers, config.selected_layer);
  for (int64_t layer = 0; layer < num_layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer) + ".";
    vt::RmsNorm(d.q, normed.t(), h.t(), weights.Get(p + "input_layernorm.weight"), norm_args);

    DBuf q(d, DType::kF32, {seq, q_width});
    DBuf k(d, DType::kF32, {seq, kv_width});
    DBuf v(d, DType::kF32, {seq, kv_width});
    if (weights.Has(p + "self_attn.qkv_proj.weight")) {
      // Uniform-encoding checkpoint: one GEMM then split.
      DBuf qkv(d, DType::kF32, {seq, q_width + 2 * kv_width});
      vt::MatmulBT(d.q, qkv.t(), normed.t(), weight(p + "self_attn.qkv_proj.weight"));
      vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
    } else {
      // MIXED-encoding checkpoint (the shipped Q4_K_M keeps v_proj at Q6_K), so the
      // group was never fused: three GEMMs, each in its own encoding. Costs launches,
      // not precision.
      vt::MatmulBT(d.q, q.t(), normed.t(), weight(p + "self_attn.q_proj.weight"));
      vt::MatmulBT(d.q, k.t(), normed.t(), weight(p + "self_attn.k_proj.weight"));
      vt::MatmulBT(d.q, v.t(), normed.t(), weight(p + "self_attn.v_proj.weight"));
    }

    // Per-head q/k RMSNorm over head_dim, THEN RoPE — that order is upstream's.
    Tensor qn = dense_attn::Reshape(q.t(), {seq * heads, head_dim});
    Tensor kn = dense_attn::Reshape(k.t(), {seq * kv_heads, head_dim});
    vt::RmsNorm(d.q, qn, qn, weights.Get(p + "self_attn.q_norm.weight"), norm_args);
    vt::RmsNorm(d.q, kn, kn, weights.Get(p + "self_attn.k_norm.weight"), norm_args);

    Tensor q3 = dense_attn::Reshape(q.t(), {seq, heads, head_dim});
    Tensor k3 = dense_attn::Reshape(k.t(), {seq, kv_heads, head_dim});
    vt::RopeArgs rope_args;
    rope_args.rotary_dim = static_cast<int>(head_dim);
    rope_args.is_neox_style = true;
    vt::RopeFromCache(d.q, q3, &k3, rope_pos.t(), rope.t(), rope_args);

    // CAUSAL GQA attention (upstream passes is_causal=True); the shared op
    // broadcasts kv heads across their query group.
    Tensor tv = dense_attn::Reshape(v.t(), {seq, kv_heads, head_dim});
    DBuf attn(d, DType::kF32, {seq, heads, head_dim});
    vt::DFlashBlockAttentionArgs args;
    args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    args.causal = true;
    args.sliding_window = 0;
    args.cu_seqlens = cu_seqlens.data();
    args.num_reqs = 1;
    vt::DFlashBlockAttention(d.q, attn.t(), q3, k3, tv, args);

    Tensor flat = dense_attn::Reshape(attn.t(), {seq, q_width});
    vt::MatmulBT(d.q, attn_out.t(), flat, weight(p + "self_attn.o_proj.weight"));
    vt::Add(d.q, h.t(), h.t(), attn_out.t());

    vt::RmsNorm(d.q, normed.t(), h.t(), weights.Get(p + "post_attention_layernorm.weight"),
                norm_args);
    const int64_t ffn = config.intermediate_size;
    DBuf act(d, DType::kF32, {seq, ffn});
    if (weights.Has(p + "mlp.gate_up_proj.weight")) {
      DBuf gate_up(d, DType::kF32, {seq, 2 * ffn});
      vt::MatmulBT(d.q, gate_up.t(), normed.t(), weight(p + "mlp.gate_up_proj.weight"));
      vt::SiluAndMul(d.q, act.t(), gate_up.t());
    } else {
      DBuf gate(d, DType::kF32, {seq, ffn});
      DBuf up(d, DType::kF32, {seq, ffn});
      vt::MatmulBT(d.q, gate.t(), normed.t(), weight(p + "mlp.gate_proj.weight"));
      vt::MatmulBT(d.q, up.t(), normed.t(), weight(p + "mlp.up_proj.weight"));
      // SiluAndMul wants [gate | up] contiguous, so stage the pair once.
      DBuf gate_up(d, DType::kF32, {seq, 2 * ffn});
      for (int64_t r = 0; r < seq; ++r) {
        backend.Copy(d.q, static_cast<float*>(gate_up.ptr()) + r * 2 * ffn,
                     static_cast<float*>(gate.ptr()) + r * ffn,
                     static_cast<size_t>(ffn) * sizeof(float));
        backend.Copy(d.q, static_cast<float*>(gate_up.ptr()) + r * 2 * ffn + ffn,
                     static_cast<float*>(up.ptr()) + r * ffn,
                     static_cast<size_t>(ffn) * sizeof(float));
      }
      vt::SiluAndMul(d.q, act.t(), gate_up.t());
    }
    vt::MatmulBT(d.q, attn_out.t(), act.t(), weight(p + "mlp.down_proj.weight"));
    vt::Add(d.q, h.t(), h.t(), attn_out.t());

    // DeepStack: ADD the visual features into the visual-token rows, for the FIRST
    // len(deepstack) layers only (encoder.py:792-799; host mirror
    // MiniMaxH3EncoderTextForward). Upstream `_deepstack_process` does
    // `hidden_states[visual_pos_masks, :] += visual_embeds`; we scatter the block
    // into a [seq, hidden] additive buffer at the masked rows and add on device.
    // Text-only prompts leave `deepstack` empty and this never runs.
    if (layer < static_cast<int64_t>(deepstack.size())) {
      VT_CHECK(visual_pos_mask != nullptr,
               "minimax_h3 encoder device: deepstack embeds require a visual position mask");
      const std::vector<float>& embeds = deepstack[static_cast<size_t>(layer)];
      std::vector<float> add_host(static_cast<size_t>(seq * hidden), 0.0f);
      int64_t visual_index = 0;
      for (int64_t s = 0; s < seq; ++s) {
        if (!visual_pos_mask[s]) continue;
        VT_CHECK((visual_index + 1) * hidden <= static_cast<int64_t>(embeds.size()),
                 "minimax_h3 encoder device: deepstack block has fewer rows than masked positions");
        for (int64_t i = 0; i < hidden; ++i) {
          add_host[static_cast<size_t>(s * hidden + i)] =
              embeds[static_cast<size_t>(visual_index * hidden + i)];
        }
        ++visual_index;
      }
      VT_CHECK(visual_index * hidden == static_cast<int64_t>(embeds.size()),
               "minimax_h3 encoder device: deepstack block row count != masked positions");
      DBuf add_dev(d, DType::kF32, {seq, hidden}, add_host.data());
      vt::Add(d.q, h.t(), h.t(), add_dev.t());
    }
  }

  // NO final RMSNorm: H3 reads the UNNORMALIZED truncated output.
  std::vector<float> out(static_cast<size_t>(seq * hidden));
  h.Download(d, out.data());
  return out;
}

}  // namespace vllm
