// dots3-note W3 — the FULL-attention layer, as a portable host reference.
// Issue #699, spec `.agents/specs/dots3-note.md`. The header carries the scope
// statement, the upstream anchors and the reason this is host code; read it
// first. Upstream read at `origin/main` = `06ecec7a84` (2026-08-25); paths are
// relative to `${VLLM_SOURCE}` = /home/mudler/_git/vllm.

#include "vllm/model_executor/models/dots3_note_attn.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm::dots3_note {
namespace {

// y[r, o] = sum_i x[r, i] * w[o, i]  — torch `nn.Linear` with `bias=False`,
// whose `.weight` is [out_features, in_features] row-major. Every projection in
// `_forward_note_mla` is bias-free (model.py:292-298 for `g_proj`;
// `deepseek_v2.py` builds the rest with `bias=False`).
std::vector<double> Linear(const std::vector<double>& x,
                           const std::vector<double>& w, int64_t rows,
                           int64_t in_features, int64_t out_features) {
  std::vector<double> y(static_cast<size_t>(rows * out_features), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    const double* xr = x.data() + r * in_features;
    double* yr = y.data() + r * out_features;
    for (int64_t o = 0; o < out_features; ++o) {
      const double* wo = w.data() + o * in_features;
      double acc = 0.0;
      for (int64_t i = 0; i < in_features; ++i) acc += xr[i] * wo[i];
      yr[o] = acc;
    }
  }
  return y;
}

std::vector<float> ToFloat(const std::vector<double>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i]);
  return out;
}

void RequireSize(const std::vector<double>& v, int64_t want, const char* name) {
  VT_CHECK(static_cast<int64_t>(v.size()) == want,
           std::string("dots3-note full attention: weight ") + name + " has " +
               std::to_string(v.size()) + " elements, geometry needs " +
               std::to_string(want) +
               " — a truncated projection reads as a plausible model. See "
               ".agents/specs/dots3-note.md and issue #699");
}

}  // namespace

// ── primitives ───────────────────────────────────────────────────────────────

std::vector<double> RmsNorm(const std::vector<double>& x,
                            const std::vector<double>& weight, int64_t rows,
                            int64_t cols, double eps) {
  VT_CHECK(static_cast<int64_t>(x.size()) == rows * cols,
           "dots3-note RmsNorm: input is " + std::to_string(x.size()) +
               " elements, want " + std::to_string(rows * cols));
  VT_CHECK(static_cast<int64_t>(weight.size()) == cols,
           "dots3-note RmsNorm: weight is " + std::to_string(weight.size()) +
               " elements, want " + std::to_string(cols));
  std::vector<double> y(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    const double* xr = x.data() + r * cols;
    double sumsq = 0.0;
    for (int64_t c = 0; c < cols; ++c) sumsq += xr[c] * xr[c];
    // `variance = x.pow(2).mean(-1)`, then `x * rsqrt(variance + epsilon)`
    // (ir/ops/layernorm.py:17-18). The epsilon is INSIDE the sqrt.
    const double inv = 1.0 / std::sqrt(sumsq / static_cast<double>(cols) + eps);
    double* yr = y.data() + r * cols;
    for (int64_t c = 0; c < cols; ++c) yr[c] = xr[c] * inv * weight[c];
  }
  return y;
}

std::vector<double> LayerNorm(const std::vector<double>& x,
                              const std::vector<double>& weight,
                              const std::vector<double>& bias, int64_t rows,
                              int64_t cols, double eps) {
  VT_CHECK(static_cast<int64_t>(x.size()) == rows * cols,
           "dots3-note LayerNorm: input is " + std::to_string(x.size()) +
               " elements, want " + std::to_string(rows * cols));
  VT_CHECK(static_cast<int64_t>(weight.size()) == cols &&
               static_cast<int64_t>(bias.size()) == cols,
           "dots3-note LayerNorm: weight/bias must both be " +
               std::to_string(cols) + " elements (torch.nn.LayerNorm is affine "
               "by default, and the released checkpoint ships BOTH "
               "indexer.k_norm.weight and indexer.k_norm.bias)");
  std::vector<double> y(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    const double* xr = x.data() + r * cols;
    double mean = 0.0;
    for (int64_t c = 0; c < cols; ++c) mean += xr[c];
    mean /= static_cast<double>(cols);
    double var = 0.0;
    for (int64_t c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
    // torch.nn.LayerNorm uses the BIASED variance (divide by N, not N-1).
    var /= static_cast<double>(cols);
    const double inv = 1.0 / std::sqrt(var + eps);
    double* yr = y.data() + r * cols;
    for (int64_t c = 0; c < cols; ++c) {
      yr[c] = (xr[c] - mean) * inv * weight[c] + bias[c];
    }
  }
  return y;
}

std::vector<double> RopeCosSinCache(double base, int64_t rotary_dim,
                                    int64_t rows) {
  VT_CHECK(rotary_dim > 0 && rotary_dim % 2 == 0,
           "dots3-note RopeCosSinCache: rotary_dim must be positive and even, "
           "got " + std::to_string(rotary_dim));
  const int64_t half = rotary_dim / 2;
  std::vector<double> cache(static_cast<size_t>(rows * rotary_dim));
  for (int64_t p = 0; p < rows; ++p) {
    double* row = cache.data() + p * rotary_dim;
    for (int64_t j = 0; j < half; ++j) {
      // inv_freq = 1 / base ** (arange(0, rotary_dim, 2) / rotary_dim)
      // (base.py:86-91).
      const double inv_freq =
          1.0 / std::pow(base, static_cast<double>(2 * j) /
                                   static_cast<double>(rotary_dim));
      const double angle = static_cast<double>(p) * inv_freq;
      row[j] = std::cos(angle);         // cat((cos, sin), -1) (base.py:102)
      row[half + j] = std::sin(angle);
    }
  }
  return cache;
}

void ApplyRopeInPlace(std::vector<double>& x,
                      const std::vector<int32_t>& positions,
                      const std::vector<double>& cos_sin_cache,
                      int64_t num_tokens, int64_t num_heads,
                      int64_t head_stride, int64_t lane_offset,
                      int64_t rotary_dim, bool is_neox_style) {
  VT_CHECK(static_cast<int64_t>(positions.size()) == num_tokens,
           "dots3-note ApplyRopeInPlace: positions is " +
               std::to_string(positions.size()) + " entries, want " +
               std::to_string(num_tokens));
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * num_heads * head_stride,
           "dots3-note ApplyRopeInPlace: tensor is " + std::to_string(x.size()) +
               " elements, want " +
               std::to_string(num_tokens * num_heads * head_stride));
  VT_CHECK(lane_offset >= 0 && lane_offset + rotary_dim <= head_stride,
           "dots3-note ApplyRopeInPlace: the rotated lanes [" +
               std::to_string(lane_offset) + ", " +
               std::to_string(lane_offset + rotary_dim) +
               ") do not fit a head of " + std::to_string(head_stride) +
               " lanes");
  const int64_t half = rotary_dim / 2;
  for (int64_t t = 0; t < num_tokens; ++t) {
    const int64_t pos = positions[static_cast<size_t>(t)];
    VT_CHECK(pos >= 0 && (pos + 1) * rotary_dim <=
                             static_cast<int64_t>(cos_sin_cache.size()),
             "dots3-note ApplyRopeInPlace: position " + std::to_string(pos) +
                 " is outside the cos/sin cache");
    const double* cos = cos_sin_cache.data() + pos * rotary_dim;
    const double* sin = cos + half;
    for (int64_t h = 0; h < num_heads; ++h) {
      double* head = x.data() + (t * num_heads + h) * head_stride + lane_offset;
      for (int64_t j = 0; j < half; ++j) {
        // GPT-J takes ADJACENT lanes, NeoX takes the two halves
        // (common.py:169-181). This pairing is §4 trap 2's polarity; the
        // `lane_offset` above is #1846's, and they are independent.
        const int64_t i1 = is_neox_style ? j : 2 * j;
        const int64_t i2 = is_neox_style ? half + j : 2 * j + 1;
        const double x1 = head[i1];
        const double x2 = head[i2];
        head[i1] = x1 * cos[j] - x2 * sin[j];
        head[i2] = x2 * cos[j] + x1 * sin[j];
      }
    }
  }
}

// ── geometry ─────────────────────────────────────────────────────────────────

double FullAttnDims::softmax_scale() const {
  return std::pow(static_cast<double>(qk_head_dim()), -0.5);
}

void FullAttnDims::Validate() const {
  VT_CHECK(hidden_size > 0 && num_heads > 0 && qk_nope_head_dim > 0 &&
               qk_rope_head_dim > 0 && v_head_dim > 0 && q_lora_rank > 0 &&
               kv_lora_rank > 0,
           "dots3-note full attention: the geometry has a non-positive "
           "dimension");
  VT_CHECK(qk_rope_head_dim % 2 == 0,
           "dots3-note full attention: qk_rope_head_dim must be even for a "
           "rotary pairing, got " + std::to_string(qk_rope_head_dim));
  VT_CHECK(index_n_heads > 0 && index_head_dim > 0 && index_topk > 0,
           "dots3-note full attention: the DSA indexer geometry has a "
           "non-positive dimension — the full layers are ALWAYS sparse "
           "(model.py:171, and the sliding class is what sets is_sparse False "
           "at :432-434)");
  VT_CHECK(index_head_dim >= qk_rope_head_dim,
           "dots3-note full attention: index_head_dim " +
               std::to_string(index_head_dim) +
               " is narrower than the rope slice qk_rope_head_dim " +
               std::to_string(qk_rope_head_dim) +
               " it must carry (deepseek_v2.py:804)");
  VT_CHECK(rope_theta > 0.0,
           "dots3-note full attention: rope_theta must be positive");
  VT_CHECK(attention_gate_type == "headwise",
           "dots3-note full attention: attention_gate_type='" +
               attention_gate_type +
               "' is not ported. Upstream's lane-wise arm exists "
               "(model.py:198-200), but W1's config parse already refuses every "
               "non-headwise value (dots3_note.cpp:376-381), so a ported arm "
               "would be production code no input reaches. See "
               ".agents/specs/dots3-note.md §4.4 and issue #699");
}

FullAttnDims Dots3NoteFullAttnDimsFrom(const Dots3NoteParams& params) {
  bool has_full_layer = false;
  for (const Dots3NoteLayerKind kind : params.layer_types) {
    if (kind == Dots3NoteLayerKind::kFullAttention) has_full_layer = true;
  }
  VT_CHECK(has_full_layer,
           "dots3-note full attention: this config's layer_types schedule has "
           "no full_attention layer, so the full geometry is not the one any "
           "layer runs — the sliding geometry is W4's and differs in the head "
           "count, TWO lora ranks, the NoPE width, the rope theta and the "
           "absence of an indexer. See .agents/specs/dots3-note.md §1.1");
  VT_CHECK(params.full.has_indexer,
           "dots3-note full attention: the full geometry reports no DSA "
           "indexer, but is_sparse is what makes a layer the FULL arm "
           "(model.py:171)");

  FullAttnDims d;
  d.hidden_size = params.hidden_size;
  d.num_heads = params.full.num_attention_heads;
  d.qk_nope_head_dim = params.full.qk_nope_head_dim;
  d.qk_rope_head_dim = params.full.qk_rope_head_dim;
  d.v_head_dim = params.full.v_head_dim;
  d.q_lora_rank = params.full.q_lora_rank;
  d.kv_lora_rank = params.full.kv_lora_rank;
  d.rms_norm_eps = params.rms_norm_eps;
  d.rope_theta = params.full.rope_theta;
  d.rope_is_neox_style = params.full.rope_is_neox_style;
  d.q_lora_scale = params.full.q_lora_scale;
  d.kv_lora_scale = params.full.kv_lora_scale;
  d.attention_gate_type = params.full.attention_gate_type;
  d.index_n_heads = params.index_n_heads;
  d.index_head_dim = params.index_head_dim;
  d.index_topk = params.index_topk;
  d.indexer_rope_is_neox_style = params.indexer_rope_is_neox_style();
  d.Validate();
  return d;
}

int64_t IndexerRopeOffset(const FullAttnDims& dims) {
  // LEADING. `deepseek_v2.py`::Indexer.forward:804-805 splits
  // `[rope_dim, head_dim - rope_dim]` off the FRONT of the index head and
  // :825 concatenates `[q_pe, q_nope]` back in that order, so lane 0 is
  // where the rotated slice lives. The released shard index says the same in
  // its own metadata (`indexer_rope_layout: "leading"`), and W2 pinned that
  // string; this is the CONSUMER of it (#1846).
  (void)dims;
  return 0;
}

// ── weights ──────────────────────────────────────────────────────────────────

void FullAttnWeights::Validate(const FullAttnDims& dims) const {
  const int64_t h = dims.hidden_size;
  RequireSize(q_a_proj, dims.q_lora_rank * h, "q_a_proj.weight");
  RequireSize(kv_a_proj_with_mqa,
              (dims.kv_lora_rank + dims.qk_rope_head_dim) * h,
              "kv_a_proj_with_mqa.weight");
  RequireSize(q_a_layernorm, dims.q_lora_rank, "q_a_layernorm.weight");
  RequireSize(kv_a_layernorm, dims.kv_lora_rank, "kv_a_layernorm.weight");
  RequireSize(k_rope_only_layernorm, dims.qk_rope_head_dim,
              "k_rope_only_layernorm.weight");
  RequireSize(q_b_proj, dims.num_heads * dims.qk_head_dim() * dims.q_lora_rank,
              "q_b_proj.weight");
  RequireSize(kv_b_proj,
              dims.num_heads * (dims.qk_nope_head_dim + dims.v_head_dim) *
                  dims.kv_lora_rank,
              "kv_b_proj.weight");
  RequireSize(o_proj, h * dims.num_heads * dims.v_head_dim, "o_proj.weight");
  // headwise => [num_heads, hidden]; the released checkpoint ships
  // [128, 5120] (model.py:286-291, and the committed shard-index fixture).
  RequireSize(g_proj, dims.num_heads * h, "g_proj.weight");
  RequireSize(indexer_wq_b, dims.index_n_heads * dims.index_head_dim *
                                dims.q_lora_rank,
              "indexer.wq_b.weight");
  RequireSize(indexer_wk, dims.index_head_dim * h, "indexer.wk.weight");
  RequireSize(indexer_weights_proj, dims.index_n_heads * h,
              "indexer.weights_proj.weight");
  RequireSize(indexer_k_norm_weight, dims.index_head_dim,
              "indexer.k_norm.weight");
  RequireSize(indexer_k_norm_bias, dims.index_head_dim, "indexer.k_norm.bias");
}

// ── the gate ─────────────────────────────────────────────────────────────────

std::vector<double> ApplyHeadwiseGate(const std::vector<double>& attn_out,
                                      const std::vector<double>& gate_logits,
                                      int64_t num_tokens, int64_t num_heads,
                                      int64_t v_head_dim) {
  VT_CHECK(static_cast<int64_t>(attn_out.size()) ==
               num_tokens * num_heads * v_head_dim,
           "dots3-note headwise gate: attention output is " +
               std::to_string(attn_out.size()) + " elements, want " +
               std::to_string(num_tokens * num_heads * v_head_dim));
  VT_CHECK(static_cast<int64_t>(gate_logits.size()) == num_tokens * num_heads,
           "dots3-note headwise gate: g_proj produced " +
               std::to_string(gate_logits.size()) +
               " logits, the headwise arm wants ONE PER HEAD, i.e. " +
               std::to_string(num_tokens * num_heads) +
               " (model.py:192-194)");
  std::vector<double> out(attn_out.size());
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t h = 0; h < num_heads; ++h) {
      // `torch.sigmoid(gate.float()).to(attn_out.dtype)` (model.py:196). The
      // FP32 cast is upstream's, not ours: on a bf16 activation path the
      // sigmoid is computed WIDE and only the result is narrowed. This
      // reference is double throughout, which is wider still; the DEVICE layer
      // owes the exact bf16-in / fp32-sigmoid / bf16-out shape (porting.md).
      const double g = 1.0 / (1.0 + std::exp(-gate_logits[t * num_heads + h]));
      const int64_t base = (t * num_heads + h) * v_head_dim;
      for (int64_t d = 0; d < v_head_dim; ++d) out[base + d] = attn_out[base + d] * g;
    }
  }
  return out;
}

// ── the layer ────────────────────────────────────────────────────────────────

std::vector<double> ForwardFullAttention(const FullAttnDims& dims,
                                         const FullAttnWeights& w,
                                         const std::vector<double>& hidden,
                                         const std::vector<int32_t>& positions,
                                         int64_t num_tokens,
                                         FullAttnTrace* trace) {
  dims.Validate();
  w.Validate(dims);
  const int64_t H = dims.hidden_size;
  const int64_t T = num_tokens;
  const int64_t N = dims.num_heads;
  const int64_t P = dims.qk_nope_head_dim;
  const int64_t R = dims.qk_rope_head_dim;
  const int64_t V = dims.v_head_dim;
  const int64_t QK = dims.qk_head_dim();
  const int64_t L = dims.kv_lora_rank;
  VT_CHECK(T > 0, "dots3-note full attention: num_tokens must be positive");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == T * H,
           "dots3-note full attention: hidden_states is " +
               std::to_string(hidden.size()) + " elements, want " +
               std::to_string(T * H));
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "dots3-note full attention: positions is " +
               std::to_string(positions.size()) + " entries, want " +
               std::to_string(T));
  int32_t max_pos = 0;
  for (const int32_t p : positions) max_pos = std::max(max_pos, p);

  // (1) `qkv_lora = fused_qkv_a_proj(hidden_states)` then the two splits
  //     (model.py:147-158). Upstream FUSES `q_a_proj` and `kv_a_proj_with_mqa`
  //     into one GEMM at load time (`DeepSeekV2FusedQkvAProjLinear`,
  //     deepseek_v2.py:1034-1039) and splits the result; concatenating the two
  //     weights' ROWS and splitting the output computes exactly the per-output
  //     dot products the two separate GEMMs do, so the numbers are identical
  //     and the checkpoint's two tensors stay two tensors here.
  std::vector<double> q_c = Linear(hidden, w.q_a_proj, T, H, dims.q_lora_rank);
  const std::vector<double> kv_lora =
      Linear(hidden, w.kv_a_proj_with_mqa, T, H, L + R);

  // (2) `q_c = q_a_layernorm(q_c) * q_lora_scale` (model.py:155). §4 trap 5.
  q_c = RmsNorm(q_c, w.q_a_layernorm, T, dims.q_lora_rank, dims.rms_norm_eps);
  for (double& v : q_c) v *= dims.q_lora_scale;

  // (3) `kv_c, k_pe = kv_lora.split([kv_lora_rank, qk_rope_head_dim])`
  //     (model.py:156-158).
  std::vector<double> kv_c(static_cast<size_t>(T * L));
  std::vector<double> k_pe(static_cast<size_t>(T * R));
  for (int64_t t = 0; t < T; ++t) {
    const double* row = kv_lora.data() + t * (L + R);
    std::copy(row, row + L, kv_c.begin() + t * L);
    std::copy(row + L, row + L + R, k_pe.begin() + t * R);
  }

  // (4) `kv_c_normed = kv_a_layernorm(kv_c) * kv_lora_scale` (model.py:159).
  std::vector<double> kv_c_normed =
      RmsNorm(kv_c, w.kv_a_layernorm, T, L, dims.rms_norm_eps);
  for (double& v : kv_c_normed) v *= dims.kv_lora_scale;

  // (5) `k_pe = k_rope_only_layernorm(k_pe)` (model.py:160) — the extra RMSNorm
  //     over the 64-wide rope-only slice of k that DeepSeek does NOT have. It
  //     runs BEFORE the rotation, so it is not absorbable into anything.
  k_pe = RmsNorm(k_pe, w.k_rope_only_layernorm, T, R, dims.rms_norm_eps);

  // (6) `q = q_b_proj(q_c).view(-1, num_heads, qk_head_dim)` (model.py:162-166).
  std::vector<double> q = Linear(q_c, w.q_b_proj, T, dims.q_lora_rank, N * QK);

  // (7) the decoupled MLA RoPE over q[..., qk_nope:] and the single shared k_pe
  //     head (model.py:167-169). GPT-J on both dots3 geometries (§4 item 6,
  //     corrected at W1 — #1804), at the FULL layers' own theta.
  const std::vector<double> mla_cos_sin =
      RopeCosSinCache(dims.rope_theta, R, static_cast<int64_t>(max_pos) + 1);
  ApplyRopeInPlace(q, positions, mla_cos_sin, T, N, QK, /*lane_offset=*/P, R,
                   dims.rope_is_neox_style);
  ApplyRopeInPlace(k_pe, positions, mla_cos_sin, T, /*num_heads=*/1, R,
                   /*lane_offset=*/0, R, dims.rope_is_neox_style);

  // (8) the DSA lightning indexer, which runs ONLY because this arm is sparse
  //     (model.py:171). It reads `q_c` — the layernormed AND RESCALED one from
  //     step (2) — and the same `hidden_states` the gate reads.
  //     `deepseek_v2.py`::Indexer.forward:751-842.
  //
  //     §4 trap 5 does NOT reach the selection, and an earlier draft of this
  //     comment claimed it did. A mutation that fed the indexer the UNRESCALED
  //     `q_c` came back GREEN, and the reason is a real invariance rather than
  //     a hole in the gate: the logit is
  //     `sum_h weights[t,h] * ReLU(dot(q[t,h,:], k[s,:]))`, so a POSITIVE
  //     rescale of `q_c` multiplies every logit in a row by the same constant
  //     and leaves the argmax alone. `q_lora_scale` therefore reaches the
  //     output only through the MLA scores. The invariance is asserted rather
  //     than merely written down — see the `q_c rescale` case in
  //     `tests/vllm/models/test_dots3_note_attn.cpp`.
  const int64_t IH = dims.index_n_heads;
  const int64_t ID = dims.index_head_dim;
  std::vector<double> iq =
      Linear(q_c, w.indexer_wq_b, T, dims.q_lora_rank, IH * ID);
  std::vector<double> ik = Linear(hidden, w.indexer_wk, T, H, ID);
  const std::vector<double> iweights =
      Linear(hidden, w.indexer_weights_proj, T, H, IH);
  // `k = self.k_norm(k)` (:810) — a LayerNorm, not an RMSNorm, and it lands
  // BEFORE the rope split.
  ik = LayerNorm(ik, w.indexer_k_norm_weight, w.indexer_k_norm_bias, T, ID,
                 dims.indexer_k_norm_eps);
  // The indexer's own rope: same cache construction, its OWN pairing (§4 trap
  // 2) and the LEADING lane offset (#1846). Both are numerically silent.
  const std::vector<double> idx_cos_sin =
      RopeCosSinCache(dims.rope_theta, R, static_cast<int64_t>(max_pos) + 1);
  const int64_t lane = IndexerRopeOffset(dims);
  ApplyRopeInPlace(iq, positions, idx_cos_sin, T, IH, ID, lane, R,
                   dims.indexer_rope_is_neox_style);
  ApplyRopeInPlace(ik, positions, idx_cos_sin, T, /*num_heads=*/1, ID, lane, R,
                   dims.indexer_rope_is_neox_style);

  // The SELECTION math is the shared DSA port, not a second copy of it:
  // `deepseek_v4::DsaIndexerWeightFold` / `DsaIndexerLogits` / `DsaTopkSelect`
  // are ports of `layers/sparse_attn_indexer.py`:203-206 / :509-518 and
  // `v1/attention/ops/triton_fp8_mqa_logits.py`:120-156, which is the same
  // machinery dots3-note reaches through `deepseek_v2.py::Indexer`. Those take
  // float; the narrowing is on the LOGITS, whose only consumer is an argmax, so
  // it can move a selection only on a tie tighter than float epsilon. The gate
  // asserts the realised selection MARGIN rather than assuming that.
  std::vector<int64_t> win_start(static_cast<size_t>(T), 0);
  std::vector<int64_t> win_end(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) win_end[static_cast<size_t>(t)] = t + 1;
  const std::vector<float> folded =
      deepseek_v4::DsaIndexerWeightFold(ToFloat(iweights), T, IH, ID);
  const std::vector<float> logits = deepseek_v4::DsaIndexerLogits(
      ToFloat(iq), ToFloat(ik), folded, win_start, win_end, T, T, IH, ID);
  const std::vector<int64_t> topk =
      deepseek_v4::DsaTopkSelect(logits, win_start, win_end, T, T,
                                 dims.index_topk);

  // (9) the MLA attention itself, materialized. `mla_attn(q, kv_c_normed, k_pe,
  //     output_shape=(T, num_heads*v_head_dim))` (model.py:179-188). The
  //     unabsorbed form up-projects the latent through `kv_b_proj` into per-head
  //     K/V; the absorbed decode form our device seam uses computes the same
  //     function (mla_attention.h documents the identity), and the reference
  //     deliberately takes the form that needs no absorption to be right.
  const std::vector<double> kv =
      Linear(kv_c_normed, w.kv_b_proj, T, L, N * (P + V));
  const double scale = dims.softmax_scale();
  std::vector<double> attn_out(static_cast<size_t>(T * N * V), 0.0);
  std::vector<double> scores(static_cast<size_t>(T));
  std::vector<char> allowed(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    // The sparse mask IS the indexer's top-k row. `DsaTopkSelect` already
    // restricted the candidates to the causal window [0, t+1) and pads with -1,
    // so membership in the row is the whole mask.
    std::fill(allowed.begin(), allowed.end(), static_cast<char>(0));
    int64_t num_allowed = 0;
    for (int64_t j = 0; j < dims.index_topk; ++j) {
      const int64_t s = topk[static_cast<size_t>(t * dims.index_topk + j)];
      if (s < 0) continue;
      VT_CHECK(s <= t, "dots3-note full attention: the indexer selected key " +
                           std::to_string(s) + " for query " +
                           std::to_string(t) + ", which is not causal");
      allowed[static_cast<size_t>(s)] = 1;
      ++num_allowed;
    }
    VT_CHECK(num_allowed > 0,
             "dots3-note full attention: query " + std::to_string(t) +
                 " selected no keys at all — a token that attends to nothing "
                 "produces a NaN softmax");
    for (int64_t h = 0; h < N; ++h) {
      const double* qh = q.data() + (t * N + h) * QK;
      double best = -std::numeric_limits<double>::infinity();
      for (int64_t s = 0; s < T; ++s) {
        if (!allowed[static_cast<size_t>(s)]) continue;
        const double* k_nope = kv.data() + (s * N + h) * (P + V);
        const double* k_rope = k_pe.data() + s * R;
        double dot = 0.0;
        for (int64_t d = 0; d < P; ++d) dot += qh[d] * k_nope[d];
        // The rope half of the key is ONE head shared by all N query heads —
        // that is what makes MLA multi-query in its rotated coordinates.
        for (int64_t d = 0; d < R; ++d) dot += qh[P + d] * k_rope[d];
        scores[static_cast<size_t>(s)] = dot * scale;
        best = std::max(best, scores[static_cast<size_t>(s)]);
      }
      double denom = 0.0;
      for (int64_t s = 0; s < T; ++s) {
        if (!allowed[static_cast<size_t>(s)]) continue;
        const double e = std::exp(scores[static_cast<size_t>(s)] - best);
        scores[static_cast<size_t>(s)] = e;
        denom += e;
      }
      double* out_head = attn_out.data() + (t * N + h) * V;
      for (int64_t s = 0; s < T; ++s) {
        if (!allowed[static_cast<size_t>(s)]) continue;
        const double p = scores[static_cast<size_t>(s)] / denom;
        const double* v_head = kv.data() + (s * N + h) * (P + V) + P;
        for (int64_t d = 0; d < V; ++d) out_head[d] += p * v_head[d];
      }
    }
  }

  // (10) the headwise gate (model.py:190-197). `g_proj` reads the SAME
  //      `hidden_states` the attention did, not the attention output.
  const std::vector<double> gate_logits = Linear(hidden, w.g_proj, T, H, N);
  const std::vector<double> gated =
      ApplyHeadwiseGate(attn_out, gate_logits, T, N, V);

  // (11) `return attention.o_proj(attn_out)[0]` (model.py:201).
  std::vector<double> out = Linear(gated, w.o_proj, T, N * V, H);

  if (trace != nullptr) {
    trace->q_c = q_c;
    trace->kv_c_normed = kv_c_normed;
    trace->k_pe = k_pe;
    trace->q = q;
    trace->indexer_q = iq;
    trace->indexer_k = ik;
    trace->indexer_logits.assign(logits.begin(), logits.end());
    trace->topk = topk;
    trace->attn_out = attn_out;
    trace->gate.resize(static_cast<size_t>(T * N));
    for (int64_t i = 0; i < T * N; ++i) {
      trace->gate[static_cast<size_t>(i)] =
          1.0 / (1.0 + std::exp(-gate_logits[static_cast<size_t>(i)]));
    }
    trace->gated = gated;
  }
  return out;
}

}  // namespace vllm::dots3_note
