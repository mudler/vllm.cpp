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

// ═════════════════════════════════════════════════════════════════════════════
// W4b-1 — the SLIDING arm, the §2.3 machinery and the padded KV row.
// The header carries the scope statement, the split argument and every upstream
// anchor; read it first. Upstream re-read at `origin/main` = `d9fbe526c0`
// (2026-08-25), whose `vllm/models/dots3_note/` is byte-identical to the
// `06ecec7a84` W3 and W4a read.
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// `torch.repeat_interleave` + `torch.cumsum` are the only two torch primitives
// `_build_sliding_window_metadata` uses that are not plain arithmetic, and both
// are three lines here; they are written out rather than hidden in a helper so
// the transcription is checkable line by line against attention.py:228-236.
void RequireSlidingSize(const std::vector<double>& v, int64_t want,
                        const char* name) {
  VT_CHECK(static_cast<int64_t>(v.size()) == want,
           std::string("dots3-note sliding attention: weight ") + name +
               " has " + std::to_string(v.size()) + " elements, geometry needs " +
               std::to_string(want) +
               " — a truncated projection reads as a plausible model. See "
               ".agents/specs/dots3-note.md and issue #699");
}

}  // namespace

// ── the sliding geometry ─────────────────────────────────────────────────────

double SlidingAttnDims::softmax_scale() const {
  // model.py:446 — `scale=qk_head_dim**-0.5`, with NO YaRN mscale, because the
  // sliding rope is built with `rope_type="default"` (model.py:404-407).
  return std::pow(static_cast<double>(qk_head_dim()), -0.5);
}

void SlidingAttnDims::Validate() const {
  VT_CHECK(hidden_size > 0 && num_heads > 0 && qk_nope_head_dim > 0 &&
               qk_rope_head_dim > 0 && v_head_dim > 0 && q_lora_rank > 0 &&
               kv_lora_rank > 0,
           "dots3-note sliding attention: the geometry has a non-positive "
           "dimension");
  VT_CHECK(qk_rope_head_dim % 2 == 0,
           "dots3-note sliding attention: qk_rope_head_dim must be even for a "
           "rotary pairing, got " + std::to_string(qk_rope_head_dim));
  VT_CHECK(rope_theta > 0.0,
           "dots3-note sliding attention: swa_rope_theta must be positive");
  VT_CHECK(sliding_window > 0,
           "dots3-note sliding attention: sliding_window_size must be positive "
           "— a window of 0 is what a FULL layer has, and the two classes are "
           "chosen by `config.layer_types` (model.py:499-503), never by the "
           "window being absent");
  VT_CHECK(physical_latent_row >= latent_row(),
           "dots3-note sliding attention: the physical MLA cache row " +
               std::to_string(physical_latent_row) +
               " is narrower than the logical row " +
               std::to_string(latent_row()) +
               " this arm reads — upstream asserts exactly this at "
               "model.py:210 (`physical_head_size >= self.head_size`)");
  VT_CHECK(attention_gate_type == "headwise",
           "dots3-note sliding attention: swa_attention_gate_type='" +
               attention_gate_type +
               "' is not ported. Upstream's lane-wise arm exists "
               "(model.py:198-200), but W1's config parse already refuses every "
               "non-headwise value, so a ported arm would be production code no "
               "input reaches. See .agents/specs/dots3-note.md §4.4 and issue "
               "#699");
}

SlidingAttnDims Dots3NoteSlidingAttnDimsFrom(const Dots3NoteParams& params) {
  bool has_sliding_layer = false;
  for (const Dots3NoteLayerKind kind : params.layer_types) {
    if (kind == Dots3NoteLayerKind::kSlidingAttention) has_sliding_layer = true;
  }
  VT_CHECK(has_sliding_layer,
           "dots3-note sliding attention: this config's layer_types schedule "
           "has no sliding_attention layer, so the sliding geometry is not the "
           "one any layer runs — the full geometry is W3/W4a's and differs in "
           "the head count, TWO lora ranks, the NoPE width, the rope theta and "
           "the presence of an indexer. See .agents/specs/dots3-note.md §1.1");
  VT_CHECK(!params.swa.has_indexer,
           "dots3-note sliding attention: the sliding geometry reports a DSA "
           "indexer, but `self.indexer = None` / `self.is_sparse = False` "
           "(model.py:432-434) is what makes a layer the SLIDING arm — the "
           "indexer runs only under `attention.is_sparse` (model.py:171)");

  SlidingAttnDims d;
  d.hidden_size = params.hidden_size;
  d.num_heads = params.swa.num_attention_heads;
  d.qk_nope_head_dim = params.swa.qk_nope_head_dim;
  d.qk_rope_head_dim = params.swa.qk_rope_head_dim;
  d.v_head_dim = params.swa.v_head_dim;
  d.q_lora_rank = params.swa.q_lora_rank;
  d.kv_lora_rank = params.swa.kv_lora_rank;
  d.rms_norm_eps = params.rms_norm_eps;
  d.rope_theta = params.swa.rope_theta;
  d.rope_is_neox_style = params.swa.rope_is_neox_style;
  d.q_lora_scale = params.swa.q_lora_scale;
  d.kv_lora_scale = params.swa.kv_lora_scale;
  d.attention_gate_type = params.swa.attention_gate_type;
  d.sliding_window = params.swa.sliding_window;
  d.physical_latent_row = params.physical_latent_row();
  d.Validate();
  return d;
}

// ── §2.3, mechanism by mechanism ─────────────────────────────────────────────

int64_t SwaGatherLen(int64_t sliding_window, int64_t query_len) {
  VT_CHECK(sliding_window > 0 && query_len > 0,
           "dots3-note SwaGatherLen: sliding_window and query_len must both be "
           "positive (attention.py:484)");
  // `(self.sliding_window + query_len - 1 + 7) // 8 * 8` — attention.py:484.
  return (sliding_window + query_len - 1 + 7) / 8 * 8;
}

std::vector<SlidingWindowChunk> BuildSlidingWindowMetadata(
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& query_start_loc, int64_t sliding_window,
    int64_t workspace_size) {
  const int64_t n_reqs = static_cast<int64_t>(seq_lens.size());
  VT_CHECK(n_reqs > 0,
           "dots3-note SWA metadata: the prefill sub-batch is empty");
  VT_CHECK(static_cast<int64_t>(query_start_loc.size()) == n_reqs + 1,
           "dots3-note SWA metadata: query_start_loc must be [n_reqs + 1] with "
           "a leading 0, got " + std::to_string(query_start_loc.size()) +
               " for " + std::to_string(n_reqs) + " requests");
  VT_CHECK(query_start_loc[0] == 0,
           "dots3-note SWA metadata: query_start_loc must start at 0 — "
           "upstream rebases it with `- query_start_loc_cpu[0]` before the "
           "build (attention.py:369)");
  VT_CHECK(sliding_window > 0 && workspace_size > 0,
           "dots3-note SWA metadata: sliding_window and workspace_size must "
           "both be positive");

  // `query_lens = query_start_loc[1:] - query_start_loc[:-1]` (:203-205),
  // `kv_lens = minimum(seq_lens, query_lens + sliding_window - 1)` (:206),
  // `starts = seq_lens - kv_lens` (:207).
  std::vector<int64_t> query_lens(static_cast<size_t>(n_reqs));
  std::vector<int64_t> kv_lens(static_cast<size_t>(n_reqs));
  std::vector<int64_t> starts(static_cast<size_t>(n_reqs));
  for (int64_t i = 0; i < n_reqs; ++i) {
    const int64_t ql = static_cast<int64_t>(query_start_loc[static_cast<size_t>(i + 1)]) -
                       static_cast<int64_t>(query_start_loc[static_cast<size_t>(i)]);
    VT_CHECK(ql > 0,
             "dots3-note SWA metadata: request " + std::to_string(i) +
                 " has a non-positive query length");
    const int64_t sl = static_cast<int64_t>(seq_lens[static_cast<size_t>(i)]);
    VT_CHECK(sl >= ql,
             "dots3-note SWA metadata: request " + std::to_string(i) +
                 " has seq_len " + std::to_string(sl) + " below its query_len " +
                 std::to_string(ql) +
                 " — the queries are the TAIL of the sequence "
                 "(attention.py:142), so a shorter sequence than query is not "
                 "representable");
    query_lens[static_cast<size_t>(i)] = ql;
    kv_lens[static_cast<size_t>(i)] = std::min(sl, ql + sliding_window - 1);
    starts[static_cast<size_t>(i)] = sl - kv_lens[static_cast<size_t>(i)];
  }

  std::vector<SlidingWindowChunk> chunks;
  int64_t req_start = 0;
  while (req_start < n_reqs) {
    // The packing loop, attention.py:209-224. Note the ORDER of the two tests:
    // the "does it fit at all" check runs AFTER the "start a new chunk" check,
    // so a request that exceeds the workspace alone still raises rather than
    // looping forever on an empty chunk.
    int64_t req_end = req_start;
    int64_t num_kv_tokens = 0;
    while (req_end < n_reqs) {
      const int64_t next_len = kv_lens[static_cast<size_t>(req_end)];
      if (num_kv_tokens != 0 && num_kv_tokens + next_len > workspace_size) break;
      VT_CHECK(next_len <= workspace_size,
               "Dots3 NOTE SWA prefill window exceeds the MLA workspace: " +
                   std::to_string(next_len) + " > " +
                   std::to_string(workspace_size));
      num_kv_tokens += next_len;
      ++req_end;
    }

    SlidingWindowChunk c;
    c.req_start = req_start;
    c.req_end = req_end;
    c.query_start = static_cast<int64_t>(query_start_loc[static_cast<size_t>(req_start)]);
    c.query_end = static_cast<int64_t>(query_start_loc[static_cast<size_t>(req_end)]);
    const int64_t n = req_end - req_start;
    c.cu_seq_lens_q.assign(static_cast<size_t>(n + 1), 0);
    c.cu_seq_lens_k.assign(static_cast<size_t>(n + 1), 0);
    for (int64_t i = 0; i < n; ++i) {
      c.cu_seq_lens_q[static_cast<size_t>(i + 1)] =
          c.cu_seq_lens_q[static_cast<size_t>(i)] +
          static_cast<int32_t>(query_lens[static_cast<size_t>(req_start + i)]);
      c.cu_seq_lens_k[static_cast<size_t>(i + 1)] =
          c.cu_seq_lens_k[static_cast<size_t>(i)] +
          static_cast<int32_t>(kv_lens[static_cast<size_t>(req_start + i)]);
      c.starts.push_back(static_cast<int32_t>(starts[static_cast<size_t>(req_start + i)]));
      c.max_seq_len_q = std::max(c.max_seq_len_q, query_lens[static_cast<size_t>(req_start + i)]);
      c.max_seq_len_k = std::max(c.max_seq_len_k, kv_lens[static_cast<size_t>(req_start + i)]);
      for (int64_t j = 0; j < kv_lens[static_cast<size_t>(req_start + i)]; ++j) {
        c.token_to_seq.push_back(static_cast<int32_t>(i));
      }
    }
    c.num_kv_tokens = num_kv_tokens;
    chunks.push_back(std::move(c));
    req_start = req_end;
  }
  return chunks;
}

SwaGatherResult GatherSwaKv(const std::vector<double>& cache,
                            const std::vector<int32_t>& block_table,
                            const std::vector<int32_t>& seq_lens, int64_t n_reqs,
                            int64_t blocks_per_req, int64_t num_blocks,
                            int64_t page_size, int64_t physical_row,
                            int64_t kv_dim, int64_t gather_len) {
  VT_CHECK(n_reqs > 0 && blocks_per_req > 0 && num_blocks > 0 && page_size > 0 &&
               gather_len > 0,
           "dots3-note GatherSwaKv: every extent must be positive");
  VT_CHECK(kv_dim > 0 && physical_row >= kv_dim,
           "dots3-note GatherSwaKv: the logical row " + std::to_string(kv_dim) +
               " must fit inside the physical row " +
               std::to_string(physical_row) +
               " (attention.py:701 asserts the same thing)");
  VT_CHECK(static_cast<int64_t>(cache.size()) == num_blocks * page_size * physical_row,
           "dots3-note GatherSwaKv: the cache is " + std::to_string(cache.size()) +
               " elements, the [num_blocks, page_size, physical_row] shape wants " +
               std::to_string(num_blocks * page_size * physical_row));
  VT_CHECK(static_cast<int64_t>(block_table.size()) == n_reqs * blocks_per_req,
           "dots3-note GatherSwaKv: block_table must be [n_reqs, "
           "blocks_per_req]");
  VT_CHECK(static_cast<int64_t>(seq_lens.size()) == n_reqs,
           "dots3-note GatherSwaKv: seq_lens must be [n_reqs]");

  SwaGatherResult r;
  r.kv.assign(static_cast<size_t>(n_reqs * gather_len * kv_dim), 0.0);
  r.valid.assign(static_cast<size_t>(n_reqs * gather_len), 0);
  for (int64_t req = 0; req < n_reqs; ++req) {
    const int64_t seq_len = static_cast<int64_t>(seq_lens[static_cast<size_t>(req)]);
    // `gather_start = max(seq_len - GATHER_LEN, 0)` (attention.py:76).
    const int64_t gather_start = std::max<int64_t>(seq_len - gather_len, 0);
    for (int64_t slot = 0; slot < gather_len; ++slot) {
      const int64_t logical_token = gather_start + slot;
      // `logical_valid = (offsets < GATHER_LEN) & (logical_tokens < seq_len)`
      // (attention.py:79).
      if (logical_token >= seq_len) continue;
      const int64_t logical_page = logical_token / page_size;
      const int64_t page_offset = logical_token - logical_page * page_size;
      VT_CHECK(logical_page < blocks_per_req,
               "dots3-note GatherSwaKv: request " + std::to_string(req) +
                   " needs logical page " + std::to_string(logical_page) +
                   " but its block table has only " +
                   std::to_string(blocks_per_req));
      const int64_t physical_page =
          static_cast<int64_t>(block_table[static_cast<size_t>(req * blocks_per_req + logical_page)]);
      // `physical_valid = logical_valid & (physical_pages >= 0)`
      // (attention.py:86): a NEGATIVE block-table entry is an unmapped page,
      // and the slot stays ZERO and INVALID rather than reading page -1.
      if (physical_page < 0) continue;
      VT_CHECK(physical_page < num_blocks,
               "dots3-note GatherSwaKv: block table entry " +
                   std::to_string(physical_page) + " is outside the " +
                   std::to_string(num_blocks) + "-block cache");
      // THE PADDING IS HERE: the row base advances by `physical_row`, and only
      // the leading `kv_dim` of it is read (`KV_DIM`, attention.py:91-97).
      const int64_t src = (physical_page * page_size + page_offset) * physical_row;
      const int64_t dst = (req * gather_len + slot) * kv_dim;
      for (int64_t c = 0; c < kv_dim; ++c) {
        r.kv[static_cast<size_t>(dst + c)] = cache[static_cast<size_t>(src + c)];
      }
      r.valid[static_cast<size_t>(req * gather_len + slot)] = 1;
    }
  }
  return r;
}

// attention.py:161. `-FLT_MAX` and not `-inf`, deliberately: see the header.
const double kSwaMaskedScore = -3.4028234663852886e38;

void ApplySwaScoreMask(std::vector<double>& scores,
                       const std::vector<int32_t>& seq_lens,
                       const std::vector<char>& valid, int64_t n_reqs,
                       int64_t num_heads, int64_t query_len, int64_t gather_len,
                       int64_t window_size) {
  VT_CHECK(n_reqs > 0 && num_heads > 0 && query_len > 0 && gather_len > 0,
           "dots3-note ApplySwaScoreMask: every extent must be positive");
  VT_CHECK(window_size > 0,
           "dots3-note ApplySwaScoreMask: WINDOW_SIZE must be positive — a "
           "window of 0 masks every key and every softmax row becomes NaN");
  VT_CHECK(static_cast<int64_t>(scores.size()) ==
               n_reqs * num_heads * query_len * gather_len,
           "dots3-note ApplySwaScoreMask: scores must be [n_reqs, num_heads, "
           "query_len, gather_len]");
  VT_CHECK(static_cast<int64_t>(valid.size()) == n_reqs * gather_len,
           "dots3-note ApplySwaScoreMask: valid must be [n_reqs, gather_len]");
  VT_CHECK(static_cast<int64_t>(seq_lens.size()) == n_reqs,
           "dots3-note ApplySwaScoreMask: seq_lens must be [n_reqs]");

  for (int64_t req = 0; req < n_reqs; ++req) {
    const int64_t seq_len = static_cast<int64_t>(seq_lens[static_cast<size_t>(req)]);
    const int64_t gather_start = std::max<int64_t>(seq_len - gather_len, 0);
    for (int64_t q = 0; q < query_len; ++q) {
      // `query_position = seq_len - QUERY_LEN + query_idx` (attention.py:142):
      // the queries are the TAIL of the sequence, so causality is decided on
      // ABSOLUTE positions and not on the token's index in the batch.
      const int64_t query_position = seq_len - query_len + q;
      for (int64_t slot = 0; slot < gather_len; ++slot) {
        const int64_t kv_position = gather_start + slot;
        const bool keep =
            valid[static_cast<size_t>(req * gather_len + slot)] != 0 &&
            kv_position <= query_position &&
            kv_position >= query_position - window_size + 1 &&
            query_position >= 0;
        if (keep) continue;
        for (int64_t h = 0; h < num_heads; ++h) {
          scores[static_cast<size_t>(((req * num_heads + h) * query_len + q) *
                                         gather_len +
                                     slot)] = kSwaMaskedScore;
        }
      }
    }
  }
}

// ── the padded / heterogeneous KV spec ───────────────────────────────────────

void PaddedMlaCacheSpec::Validate() const {
  VT_CHECK(num_blocks > 0 && page_size > 0 && logical_row > 0,
           "dots3-note padded MLA cache: every extent must be positive");
  VT_CHECK(physical_row >= logical_row,
           "dots3-note padded MLA cache: the physical row " +
               std::to_string(physical_row) +
               " is narrower than the logical row " + std::to_string(logical_row) +
               " a layer reads — `Dots3NotePaddedMLAAttention.__init__` asserts "
               "exactly this (model.py:210)");
}

void WritePaddedMlaCache(std::vector<double>& cache,
                         const PaddedMlaCacheSpec& spec,
                         const std::vector<double>& kv_c_normed,
                         const std::vector<double>& k_pe, int64_t kv_lora_rank,
                         int64_t qk_rope_head_dim,
                         const std::vector<int64_t>& slot_mapping,
                         int64_t num_tokens) {
  spec.Validate();
  VT_CHECK(kv_lora_rank + qk_rope_head_dim == spec.logical_row,
           "dots3-note padded MLA cache: the write is " +
               std::to_string(kv_lora_rank + qk_rope_head_dim) +
               " wide but the spec's logical row is " +
               std::to_string(spec.logical_row));
  VT_CHECK(static_cast<int64_t>(cache.size()) == spec.slots() * spec.physical_row,
           "dots3-note padded MLA cache: the cache buffer is " +
               std::to_string(cache.size()) + " elements, the spec wants " +
               std::to_string(spec.slots() * spec.physical_row));
  VT_CHECK(static_cast<int64_t>(kv_c_normed.size()) == num_tokens * kv_lora_rank &&
               static_cast<int64_t>(k_pe.size()) == num_tokens * qk_rope_head_dim,
           "dots3-note padded MLA cache: kv_c_normed / k_pe do not match "
           "num_tokens");
  VT_CHECK(static_cast<int64_t>(slot_mapping.size()) == num_tokens,
           "dots3-note padded MLA cache: slot_mapping must have one slot per "
           "token");
  for (int64_t t = 0; t < num_tokens; ++t) {
    const int64_t slot = slot_mapping[static_cast<size_t>(t)];
    VT_CHECK(slot >= 0 && slot < spec.slots(),
             "dots3-note padded MLA cache: slot " + std::to_string(slot) +
                 " is outside the " + std::to_string(spec.slots()) +
                 "-slot cache");
    // The row base advances by the PHYSICAL row; only the leading logical_row
    // is written, and the tail keeps whatever the other class put there.
    double* row = cache.data() + slot * spec.physical_row;
    for (int64_t c = 0; c < kv_lora_rank; ++c) {
      row[c] = kv_c_normed[static_cast<size_t>(t * kv_lora_rank + c)];
    }
    for (int64_t c = 0; c < qk_rope_head_dim; ++c) {
      row[kv_lora_rank + c] = k_pe[static_cast<size_t>(t * qk_rope_head_dim + c)];
    }
  }
}

std::vector<double> NarrowLogicalCacheRows(const std::vector<double>& cache,
                                           const PaddedMlaCacheSpec& spec) {
  spec.Validate();
  VT_CHECK(static_cast<int64_t>(cache.size()) == spec.slots() * spec.physical_row,
           "dots3-note _logical_cache: the cache buffer is " +
               std::to_string(cache.size()) + " elements, the spec wants " +
               std::to_string(spec.slots() * spec.physical_row));
  std::vector<double> out(static_cast<size_t>(spec.slots() * spec.logical_row));
  for (int64_t s = 0; s < spec.slots(); ++s) {
    const double* src = cache.data() + s * spec.physical_row;
    double* dst = out.data() + s * spec.logical_row;
    for (int64_t c = 0; c < spec.logical_row; ++c) dst[c] = src[c];
  }
  return out;
}

// ── the sliding layer ────────────────────────────────────────────────────────

void SlidingAttnWeights::Validate(const SlidingAttnDims& dims) const {
  const int64_t h = dims.hidden_size;
  RequireSlidingSize(q_a_proj, dims.q_lora_rank * h, "q_a_proj.weight");
  RequireSlidingSize(kv_a_proj_with_mqa,
                     (dims.kv_lora_rank + dims.qk_rope_head_dim) * h,
                     "kv_a_proj_with_mqa.weight");
  RequireSlidingSize(q_a_layernorm, dims.q_lora_rank, "q_a_layernorm.weight");
  RequireSlidingSize(kv_a_layernorm, dims.kv_lora_rank, "kv_a_layernorm.weight");
  RequireSlidingSize(k_rope_only_layernorm, dims.qk_rope_head_dim,
                     "k_rope_only_layernorm.weight");
  RequireSlidingSize(q_b_proj,
                     dims.num_heads * dims.qk_head_dim() * dims.q_lora_rank,
                     "q_b_proj.weight");
  RequireSlidingSize(kv_b_proj,
                     dims.num_heads *
                         (dims.qk_nope_head_dim + dims.v_head_dim) *
                         dims.kv_lora_rank,
                     "kv_b_proj.weight");
  RequireSlidingSize(o_proj, h * dims.num_heads * dims.v_head_dim,
                     "o_proj.weight");
  RequireSlidingSize(g_proj, dims.num_heads * h, "g_proj.weight");
}

std::vector<double> ForwardSlidingAttention(const SlidingAttnDims& dims,
                                            const SlidingAttnWeights& w,
                                            const std::vector<double>& hidden,
                                            const std::vector<int32_t>& positions,
                                            int64_t num_tokens,
                                            const SlidingPaging& paging,
                                            SlidingAttnTrace* trace) {
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
  const int64_t LR = dims.latent_row();
  VT_CHECK(T > 0, "dots3-note sliding attention: num_tokens must be positive");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == T * H,
           "dots3-note sliding attention: hidden_states is " +
               std::to_string(hidden.size()) + " elements, want " +
               std::to_string(T * H));
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "dots3-note sliding attention: positions is " +
               std::to_string(positions.size()) + " entries, want " +
               std::to_string(T));
  VT_CHECK(paging.page_size > 0,
           "dots3-note sliding attention: the paging needs a positive page "
           "size — the gather addresses the cache by [page, offset] "
           "(attention.py:80-85)");
  const int64_t page_size = paging.page_size;
  const int64_t blocks_per_req = static_cast<int64_t>(paging.block_table.size());
  VT_CHECK(blocks_per_req * page_size >= T,
           "dots3-note sliding attention: the block table holds " +
               std::to_string(blocks_per_req * page_size) +
               " slots but the request has " + std::to_string(T) + " tokens");
  int32_t max_pos = 0;
  for (const int32_t p : positions) max_pos = std::max(max_pos, p);

  // (1)-(2) `q_c = q_a_layernorm(q_c) * q_lora_scale` — model.py:147-155, at
  //         the SLIDING ranks. Identical shape to the full arm; different
  //         numbers, and §4 trap 5's two scales are equal here on the released
  //         model because both swa ranks are 1024.
  std::vector<double> q_c = Linear(hidden, w.q_a_proj, T, H, dims.q_lora_rank);
  q_c = RmsNorm(q_c, w.q_a_layernorm, T, dims.q_lora_rank, dims.rms_norm_eps);
  for (double& v : q_c) v *= dims.q_lora_scale;

  // (3)-(5) the kv split, `kv_a_layernorm(kv_c) * kv_lora_scale` and the extra
  //         `k_rope_only_layernorm` over the rope-only slice — model.py:156-160.
  const std::vector<double> kv_lora =
      Linear(hidden, w.kv_a_proj_with_mqa, T, H, L + R);
  std::vector<double> kv_c(static_cast<size_t>(T * L));
  std::vector<double> k_pe(static_cast<size_t>(T * R));
  for (int64_t t = 0; t < T; ++t) {
    const double* row = kv_lora.data() + t * (L + R);
    std::copy(row, row + L, kv_c.begin() + t * L);
    std::copy(row + L, row + L + R, k_pe.begin() + t * R);
  }
  std::vector<double> kv_c_normed =
      RmsNorm(kv_c, w.kv_a_layernorm, T, L, dims.rms_norm_eps);
  for (double& v : kv_c_normed) v *= dims.kv_lora_scale;
  k_pe = RmsNorm(k_pe, w.k_rope_only_layernorm, T, R, dims.rms_norm_eps);

  // (6)-(7) `q_b_proj` then the decoupled RoPE, at the SLIDING theta
  //         (model.py:406) and GPT-J (model.py:408). Three orders of magnitude
  //         from the full arm's theta, and nothing about the shapes says so.
  std::vector<double> q = Linear(q_c, w.q_b_proj, T, dims.q_lora_rank, N * QK);
  const std::vector<double> cos_sin =
      RopeCosSinCache(dims.rope_theta, R, static_cast<int64_t>(max_pos) + 1);
  ApplyRopeInPlace(q, positions, cos_sin, T, N, QK, /*lane_offset=*/P, R,
                   dims.rope_is_neox_style);
  ApplyRopeInPlace(k_pe, positions, cos_sin, T, /*num_heads=*/1, R,
                   /*lane_offset=*/0, R, dims.rope_is_neox_style);

  // NO INDEXER. `attention.indexer and attention.is_sparse` (model.py:171) is
  // False on this arm because the class sets both to None/False at :432-434.
  // The absence is the mechanism, so it is stated here rather than left as a
  // missing block: a sliding layer that ran the DSA top-k would prune a window
  // that is already 513 wide, and `Dots3NoteSlidingAttnDimsFrom` refuses a
  // params object that claims one.

  // (8) the MLA cache write, into the PADDED physical row every layer shares
  //     (model.py:283 -> :216). `slot = block_table[t / page_size] * page_size
  //     + t % page_size` is the slot mapping the engine hands the cache write.
  PaddedMlaCacheSpec spec;
  spec.page_size = page_size;
  spec.physical_row = dims.physical_latent_row;
  spec.logical_row = LR;
  int64_t max_page = 0;
  for (const int32_t b : paging.block_table) {
    VT_CHECK(b >= 0,
             "dots3-note sliding attention: this request's own block table "
             "carries an unmapped page — the negative entry the gather tolerates "
             "(attention.py:86) is for slots BEYOND seq_len, not for pages the "
             "request is writing");
    max_page = std::max<int64_t>(max_page, b);
  }
  spec.num_blocks = max_page + 1;
  spec.Validate();
  std::vector<double> cache(static_cast<size_t>(spec.slots() * spec.physical_row), 0.0);
  std::vector<int64_t> slot_mapping(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    const int64_t page = paging.block_table[static_cast<size_t>(t / page_size)];
    slot_mapping[static_cast<size_t>(t)] = page * page_size + t % page_size;
  }
  WritePaddedMlaCache(cache, spec, kv_c_normed, k_pe, L, R, slot_mapping, T);

  // (9) `_forward_swa_mqa` (attention.py:470-563), at `num_reqs == 1`,
  //     `seq_len == query_len == T`.
  //
  //     (9a) the ABSORBED query. `mqa_ql_nope = bmm(q_nope, W_UK_T)` then
  //     concat with `q_pe` (mla_attention.py's absorbed decode form, which is
  //     what `_forward_swa_mqa` is handed): W_UK[h] is rows
  //     [h*(P+V), h*(P+V)+P) of `kv_b_proj`, each a [kv_lora_rank] vector.
  std::vector<double> q_absorbed(static_cast<size_t>(T * N * LR), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < N; ++h) {
      double* dst = q_absorbed.data() + (t * N + h) * LR;
      const double* qh = q.data() + (t * N + h) * QK;
      for (int64_t l = 0; l < L; ++l) {
        double acc = 0.0;
        for (int64_t d = 0; d < P; ++d) {
          acc += qh[d] * w.kv_b_proj[static_cast<size_t>((h * (P + V) + d) * L + l)];
        }
        dst[l] = acc;
      }
      for (int64_t d = 0; d < R; ++d) dst[L + d] = qh[P + d];
    }
  }

  //     (9b) the gather. `gather_len` is the SAME function the workspace
  //     reservation uses (attention.py:321 / :484).
  const int64_t gather_len = SwaGatherLen(dims.sliding_window, T);
  const std::vector<int32_t> seq_lens{static_cast<int32_t>(T)};
  const SwaGatherResult gathered =
      GatherSwaKv(cache, paging.block_table, seq_lens, /*n_reqs=*/1,
                  blocks_per_req, spec.num_blocks, page_size, spec.physical_row,
                  LR, gather_len);
  // `gather_start` is 0 whenever `gather_len >= seq_len`, which holds by
  // construction here (`gather_len >= window + T - 1 >= T` for window >= 1), so
  // every query keeps at least its own key and no softmax row is all-masked.
  // Stated as a check rather than as a comment, because an all-masked row is
  // exactly the `exp(-FLT_MAX - -FLT_MAX) == 1` degenerate the mask literal was
  // chosen to keep finite, and a silent one would be worse than a throw.
  VT_CHECK(gather_len >= T,
           "dots3-note sliding attention: gather_len " +
               std::to_string(gather_len) + " is below the sequence length " +
               std::to_string(T) +
               " — a query earlier than gather_start would attend to nothing");

  //     (9c) the scores, in `scores_by_head` layout [1, N, T, gather_len]
  //     (attention.py:532-540), then the window mask (:541-556).
  const double scale = dims.softmax_scale();
  std::vector<double> scores(static_cast<size_t>(N * T * gather_len), 0.0);
  for (int64_t h = 0; h < N; ++h) {
    for (int64_t t = 0; t < T; ++t) {
      const double* qa = q_absorbed.data() + (t * N + h) * LR;
      double* row = scores.data() + (h * T + t) * gather_len;
      for (int64_t g = 0; g < gather_len; ++g) {
        const double* kv = gathered.kv.data() + g * LR;
        double dot = 0.0;
        for (int64_t c = 0; c < LR; ++c) dot += qa[c] * kv[c];
        row[g] = dot * scale;
      }
    }
  }
  ApplySwaScoreMask(scores, seq_lens, gathered.valid, /*n_reqs=*/1, N, T,
                    gather_len, dims.sliding_window);

  //     (9d) softmax, then `bmm(probs, kv_latent[..., :kv_lora_rank])` and the
  //     `W_UV` up-projection (attention.py:557-563 + the v_up_proj the caller
  //     applies). `attn_out` is [T, N*V], which is what the gate reads.
  std::vector<double> attn_out(static_cast<size_t>(T * N * V), 0.0);
  int64_t pruned_rows = 0;
  for (int64_t t = 0; t < T; ++t) {
    // How many causal keys this query LOSES to the window. Reported so the gate
    // can refuse to be vacuous: at T <= window nothing is pruned and every
    // window assertion would pass on a plain causal answer.
    if (t + 1 > dims.sliding_window) ++pruned_rows;
  }
  std::vector<double> probs(static_cast<size_t>(gather_len));
  std::vector<double> latent_out(static_cast<size_t>(L));
  for (int64_t h = 0; h < N; ++h) {
    for (int64_t t = 0; t < T; ++t) {
      const double* row = scores.data() + (h * T + t) * gather_len;
      double best = -std::numeric_limits<double>::infinity();
      for (int64_t g = 0; g < gather_len; ++g) best = std::max(best, row[g]);
      double denom = 0.0;
      for (int64_t g = 0; g < gather_len; ++g) {
        probs[static_cast<size_t>(g)] = std::exp(row[g] - best);
        denom += probs[static_cast<size_t>(g)];
      }
      std::fill(latent_out.begin(), latent_out.end(), 0.0);
      for (int64_t g = 0; g < gather_len; ++g) {
        const double p = probs[static_cast<size_t>(g)] / denom;
        if (p == 0.0) continue;
        const double* kv = gathered.kv.data() + g * LR;
        for (int64_t l = 0; l < L; ++l) latent_out[static_cast<size_t>(l)] += p * kv[l];
      }
      double* dst = attn_out.data() + (t * N + h) * V;
      for (int64_t v = 0; v < V; ++v) {
        double acc = 0.0;
        for (int64_t l = 0; l < L; ++l) {
          acc += latent_out[static_cast<size_t>(l)] *
                 w.kv_b_proj[static_cast<size_t>((h * (P + V) + P + v) * L + l)];
        }
        dst[v] = acc;
      }
    }
  }

  // (10)-(11) the headwise gate and `o_proj` — model.py:190-201, shared with
  //           the full arm because `_forward_note_mla` is ONE function.
  const std::vector<double> gate_logits = Linear(hidden, w.g_proj, T, H, N);
  const std::vector<double> gated =
      ApplyHeadwiseGate(attn_out, gate_logits, T, N, V);
  std::vector<double> out = Linear(gated, w.o_proj, T, N * V, H);

  if (trace != nullptr) {
    trace->q_c = q_c;
    trace->kv_c_normed = kv_c_normed;
    trace->k_pe = k_pe;
    trace->q = q;
    trace->q_absorbed = q_absorbed;
    trace->cache = cache;
    trace->gathered = gathered.kv;
    trace->gather_valid = gathered.valid;
    trace->masked_scores = scores;
    trace->attn_out = attn_out;
    trace->gate.resize(static_cast<size_t>(T * N));
    for (int64_t i = 0; i < T * N; ++i) {
      trace->gate[static_cast<size_t>(i)] =
          1.0 / (1.0 + std::exp(-gate_logits[static_cast<size_t>(i)]));
    }
    trace->gated = gated;
    trace->gather_len = gather_len;
    trace->rows_pruned_by_the_window = pruned_rows;
  }
  return out;
}

}  // namespace vllm::dots3_note
