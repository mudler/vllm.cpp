// DeepSeek-V4-Flash W3 primitives — host reference implementations.
// See deepseek_v4_dsa.h for the full port map (file:line on both sides).
#include "vllm/model_executor/models/deepseek_v4_dsa.h"

// For `HostBf16ToF32`, the inlined carried-tower widening `Wf` uses below.
// No cycle: `deepseek_v4_dsa.h` includes only <cstdint> and <vector>, and
// `deepseek_v4.h` does not include it back.
#include "vllm/model_executor/models/deepseek_v4.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm::deepseek_v4 {

std::vector<float> DsaIndexerWeightFold(const std::vector<float>& weights_proj,
                                        int64_t num_tokens, int64_t index_n_heads,
                                        int64_t index_head_dim) {
  VT_CHECK(index_n_heads > 0 && index_head_dim > 0, "bad indexer dims");
  VT_CHECK(static_cast<int64_t>(weights_proj.size()) == num_tokens * index_n_heads,
           "weights_proj size mismatch");
  // softmax_scale = index_head_dim**-0.5 (attention.py:735);
  // head_scale    = index_n_heads**-0.5 (attention.py:843).
  const float softmax_scale =
      1.0f / std::sqrt(static_cast<float>(index_head_dim));
  const float head_scale = 1.0f / std::sqrt(static_cast<float>(index_n_heads));
  const float fold = softmax_scale * head_scale;
  std::vector<float> out(weights_proj.size());
  for (size_t i = 0; i < weights_proj.size(); ++i) out[i] = weights_proj[i] * fold;
  return out;
}

std::vector<float> DsaIndexerLogits(const std::vector<float>& q,
                                    const std::vector<float>& k,
                                    const std::vector<float>& folded_weights,
                                    const std::vector<int64_t>& win_start,
                                    const std::vector<int64_t>& win_end,
                                    int64_t num_tokens, int64_t num_keys,
                                    int64_t index_n_heads, int64_t index_head_dim) {
  const int64_t H = index_n_heads, D = index_head_dim;
  VT_CHECK(static_cast<int64_t>(q.size()) == num_tokens * H * D, "q size mismatch");
  VT_CHECK(static_cast<int64_t>(k.size()) == num_keys * D, "k size mismatch");
  VT_CHECK(static_cast<int64_t>(folded_weights.size()) == num_tokens * H,
           "folded_weights size mismatch");
  VT_CHECK(static_cast<int64_t>(win_start.size()) == num_tokens &&
               static_cast<int64_t>(win_end.size()) == num_tokens,
           "window arrays size mismatch");

  const float kNegInf = -std::numeric_limits<float>::infinity();
  std::vector<float> logits(static_cast<size_t>(num_tokens) * num_keys, kNegInf);

  for (int64_t t = 0; t < num_tokens; ++t) {
    const int64_t s0 = std::max<int64_t>(0, win_start[t]);
    const int64_t s1 = std::min<int64_t>(num_keys, win_end[t]);
    for (int64_t s = s0; s < s1; ++s) {
      float acc = 0.0f;
      for (int64_t h = 0; h < H; ++h) {
        // dot(q[t,h,:], k[s,:])
        float dot = 0.0f;
        const float* qp = &q[((t * H) + h) * D];
        const float* kp = &k[s * D];
        for (int64_t d = 0; d < D; ++d) dot += qp[d] * kp[d];
        // kv_scale == 1 in the fp32 reference; ReLU is load-bearing
        // (triton_fp8_mqa_logits.py:129).
        const float relu = dot > 0.0f ? dot : 0.0f;
        acc += folded_weights[t * H + h] * relu;
      }
      logits[t * num_keys + s] = acc;
    }
  }
  return logits;
}

std::vector<int64_t> DsaTopkSelect(const std::vector<float>& logits,
                                   const std::vector<int64_t>& win_start,
                                   const std::vector<int64_t>& win_end,
                                   int64_t num_tokens, int64_t num_keys,
                                   int64_t topk) {
  VT_CHECK(topk > 0, "topk must be positive");
  VT_CHECK(static_cast<int64_t>(logits.size()) == num_tokens * num_keys,
           "logits size mismatch");
  std::vector<int64_t> out(static_cast<size_t>(num_tokens) * topk, -1);

  for (int64_t t = 0; t < num_tokens; ++t) {
    const int64_t s0 = std::max<int64_t>(0, win_start[t]);
    const int64_t s1 = std::min<int64_t>(num_keys, win_end[t]);
    const int64_t n = std::max<int64_t>(0, s1 - s0);
    int64_t* dst = &out[t * topk];

    if (n <= topk) {
      // Short-context: EVERY candidate selected, ascending key order
      // (attention.py:70-86 _fill_short_context_topk_indices / :813-831).
      int64_t w = 0;
      for (int64_t s = s0; s < s1; ++s) dst[w++] = s;
      continue;  // remaining slots stay -1
    }

    // Full top-k: pick the `topk` keys with the largest logits; ties resolved
    // toward the SMALLER key index (stable). Sort a candidate index list by
    // (logit desc, index asc), take the first `topk`, then emit them in
    // ASCENDING key order (top_k_per_row writes indices, order-agnostic for the
    // downstream gather; ascending keeps the reference deterministic).
    std::vector<int64_t> cand(static_cast<size_t>(n));
    std::iota(cand.begin(), cand.end(), s0);
    std::stable_sort(cand.begin(), cand.end(), [&](int64_t a, int64_t b) {
      const float la = logits[t * num_keys + a];
      const float lb = logits[t * num_keys + b];
      if (la != lb) return la > lb;  // larger logit first
      return a < b;                  // tie -> smaller index
    });
    cand.resize(static_cast<size_t>(topk));
    std::sort(cand.begin(), cand.end());  // ascending key order
    for (int64_t i = 0; i < topk; ++i) dst[i] = cand[static_cast<size_t>(i)];
  }
  return out;
}

std::vector<float> SoftmaxWithSink(const std::vector<float>& scores, float sink) {
  const int64_t n = static_cast<int64_t>(scores.size());
  std::vector<float> prob(static_cast<size_t>(n), 0.0f);
  if (n == 0) return prob;

  float m = sink;
  for (float s : scores) m = std::max(m, s);
  // A fully -inf row (no keys finite AND sink -inf) would give 0/0; guard it.
  if (m == -std::numeric_limits<float>::infinity()) return prob;

  float denom = std::exp(sink - m);  // sink contributes to the denominator only
  for (int64_t j = 0; j < n; ++j) {
    const float e = std::exp(scores[static_cast<size_t>(j)] - m);
    prob[static_cast<size_t>(j)] = e;
    denom += e;
  }
  for (int64_t j = 0; j < n; ++j) prob[static_cast<size_t>(j)] /= denom;
  return prob;
}

// `Wf` is the ONE place the weight dtype is widened. On `float` it is the identity,
// so the f32 arm compiles to exactly what it compiled to before W1d; on `uint16_t`
// it is `HostBf16ToF32`, the INLINE header helper (`deepseek_v4.h`) that performs
// `vt::BF16ToF32`'s bit operation without the out-of-line call this build cannot
// inline away -- see W1d-5, and the exhaustive 65536-pattern agreement case.
inline float Wf(float w) { return w; }
inline float Wf(uint16_t w) { return HostBf16ToF32(w); }

// One body, two weight dtypes (W1d, #2186). `Wf` widens a bf16 bit pattern the
// same way `vt::BF16ToF32` does and is the identity on f32, so the f32 arm is
// BYTE-FOR-BYTE the pre-W1d function and the bf16 arm differs only in the weight
// elements themselves. Accumulators and reduction order are shared.
template <typename W>
std::vector<float> GroupedOutputLora(const std::vector<float>& o,
                                     const std::vector<W>& wo_a,
                                     const std::vector<W>& wo_b,
                                     int64_t num_tokens, int64_t n_heads,
                                     int64_t head_dim, int64_t n_groups,
                                     int64_t o_lora_rank, int64_t hidden_size) {
  VT_CHECK(n_groups > 0 && n_heads % n_groups == 0,
           "n_heads must be divisible by n_groups");
  const int64_t in_per_group = n_heads * head_dim / n_groups;  // heads_per_group*head_dim
  const int64_t z_dim = n_groups * o_lora_rank;
  VT_CHECK(static_cast<int64_t>(o.size()) == num_tokens * n_heads * head_dim,
           "o size mismatch");
  VT_CHECK(static_cast<int64_t>(wo_a.size()) == n_groups * o_lora_rank * in_per_group,
           "wo_a size mismatch");
  VT_CHECK(static_cast<int64_t>(wo_b.size()) == hidden_size * z_dim,
           "wo_b size mismatch");

  std::vector<float> out(static_cast<size_t>(num_tokens) * hidden_size, 0.0f);
  std::vector<float> z(static_cast<size_t>(z_dim));

  for (int64_t t = 0; t < num_tokens; ++t) {
    // o for token t is [n_heads, head_dim] contiguous == [n_groups, in_per_group]
    // contiguous (heads_per_group heads packed per group).
    const float* o_t = &o[t * n_heads * head_dim];
    // z[g, d] = sum_r wo_a[g, d, r] * o_group[g, r]  (per-group einsum "bhr,hdr->bhd")
    for (int64_t g = 0; g < n_groups; ++g) {
      const float* o_g = o_t + g * in_per_group;
      const W* wa_g = &wo_a[g * o_lora_rank * in_per_group];
      float* z_g = &z[g * o_lora_rank];
      for (int64_t d = 0; d < o_lora_rank; ++d) {
        float acc = 0.0f;
        const W* wa_gd = wa_g + d * in_per_group;
        for (int64_t r = 0; r < in_per_group; ++r) acc += Wf(wa_gd[r]) * o_g[r];
        z_g[d] = acc;
      }
    }
    // out[t, :] = wo_b @ z    ( [hidden_size x z_dim] @ [z_dim] )
    float* out_t = &out[t * hidden_size];
    for (int64_t h = 0; h < hidden_size; ++h) {
      float acc = 0.0f;
      const W* wb_h = &wo_b[h * z_dim];
      for (int64_t c = 0; c < z_dim; ++c) acc += Wf(wb_h[c]) * z[static_cast<size_t>(c)];
      out_t[h] = acc;
    }
  }
  return out;
}

// Both arms are instantiated HERE rather than left implicit, so a caller that
// needs a third weight dtype fails to link instead of silently instantiating a
// body whose numerics nobody reviewed.
template std::vector<float> GroupedOutputLora<float>(const std::vector<float>&,
                                                     const std::vector<float>&,
                                                     const std::vector<float>&, int64_t,
                                                     int64_t, int64_t, int64_t, int64_t,
                                                     int64_t);
template std::vector<float> GroupedOutputLora<uint16_t>(const std::vector<float>&,
                                                        const std::vector<uint16_t>&,
                                                        const std::vector<uint16_t>&,
                                                        int64_t, int64_t, int64_t, int64_t,
                                                        int64_t, int64_t);

}  // namespace vllm::deepseek_v4
