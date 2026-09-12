// DeepSeek-V4-Flash W6 — the sqrtsoftplus + hash-routed MoE host reference.
// Ported 1:1 from the pinned vLLM eager reference @ 555967922:
//   - sqrtsoftplus score + router: vllm/model_executor/layers/fused_moe/router/
//     fused_topk_bias_router.py:75-118 (`_topk_softplus_sqrt_torch`) + the hash
//     branch :100-106; cross-checked SGLang v0.5.15
//     python/sglang/srt/layers/moe/{topk.py:1013-1014, hash_topk.py:137-180}.
//   - clamped SwiGLU: vllm/model_executor/layers/activation.py:197-201
//     (`SiluAndMulWithClamp.forward_native`), used by DeepseekV4MLP
//     nvidia/model.py:126-133.
// See deepseek_v4_moe.h for the full semantics + honest-scope note.
#include "vllm/model_executor/models/deepseek_v4_moe.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vllm::deepseek_v4 {

namespace {
inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace

float SqrtSoftplus(float x) {
  // softplus(x) = log(1 + exp(x)), numerically stable as
  //   max(x,0) + log1p(exp(-|x|))   (== log(1+exp(x)) for all x)
  // then the outer sqrt. Matches torch.sqrt(F.softplus(x.float()))
  // (fused_topk_bias_router.py:88).
  const float sp = std::max(x, 0.0f) + std::log1p(std::exp(-std::fabs(x)));
  return std::sqrt(sp);
}

MoeRouteResult SqrtSoftplusRouteTopk(const std::vector<float>& gating, int64_t num_tokens,
                                     int64_t num_experts, int64_t topk,
                                     const std::vector<float>& e_score_correction_bias,
                                     bool renormalize, float routed_scaling_factor,
                                     const std::vector<int64_t>& input_tokens,
                                     const std::vector<int32_t>& hash_indices_table,
                                     int64_t vocab_size,
                                     const std::vector<float>& vision_bias,
                                     const std::vector<char>& is_media_token) {
  const bool has_bias = !e_score_correction_bias.empty();
  const bool is_hash = !hash_indices_table.empty() && !input_tokens.empty();
  const bool any_media = !is_media_token.empty() && !vision_bias.empty();

  MoeRouteResult out;
  out.topk_ids.assign(static_cast<size_t>(num_tokens * topk), 0);
  out.topk_weights.assign(static_cast<size_t>(num_tokens * topk), 0.0f);

  std::vector<float> scores(static_cast<size_t>(num_experts));
  std::vector<float> scores_for_choice(static_cast<size_t>(num_experts));
  std::vector<int64_t> order(static_cast<size_t>(num_experts));

  for (int64_t t = 0; t < num_tokens; ++t) {
    const float* g = gating.data() + t * num_experts;
    // scores = sqrt(softplus(gating))  (UNBIASED — the weight source).
    for (int64_t e = 0; e < num_experts; ++e) scores[e] = SqrtSoftplus(g[e]);

    int32_t* ids = out.topk_ids.data() + t * topk;
    float* w = out.topk_weights.data() + t * topk;

    // MODEL-MM-deepseek-v4 W4 (#2411): an IMAGE row takes the vision bias and
    // the learned top-k route, on EVERY layer. On a hash layer that replaces
    // the `tid2eid` lookup rather than adding to it, because the row has no
    // token identifier to hash -- which is the same answer llama.cpp gives by
    // skipping its hash branch for a whole media ubatch.
    const bool media =
        any_media && is_media_token[static_cast<size_t>(t)] != 0;

    if (is_hash && !media) {
      // Hash MoE: experts are predetermined by the tid2eid lookup on the token
      // id; the bias is NOT used (a hash layer carries none). Weights are
      // gathered from the UNBIASED scores (fused_topk_bias_router.py:100-106,
      // SGLang hash_topk.py:176-179).
      const int64_t tok = input_tokens[static_cast<size_t>(t)];
      const int32_t* row = hash_indices_table.data() + (tok % vocab_size) * topk;
      for (int64_t j = 0; j < topk; ++j) {
        ids[j] = row[j];
        w[j] = scores[row[j]];
      }
    } else {
      // scores_for_choice = scores + bias  (SELECTION ONLY).
      const std::vector<float>& row_bias =
          media ? vision_bias : e_score_correction_bias;
      const bool row_has_bias = media ? true : has_bias;
      for (int64_t e = 0; e < num_experts; ++e) {
        scores_for_choice[e] = row_has_bias ? scores[e] + row_bias[e] : scores[e];
      }
      // top-k by scores_for_choice, descending; ties → smaller expert index
      // (a stable partial sort — mirrors torch.topk(sorted=True) with a
      // deterministic tie-break, fused_topk_bias_router.py:109).
      std::iota(order.begin(), order.end(), int64_t{0});
      std::partial_sort(
          order.begin(), order.begin() + topk, order.end(),
          [&](int64_t a, int64_t b) {
            if (scores_for_choice[a] != scores_for_choice[b])
              return scores_for_choice[a] > scores_for_choice[b];
            return a < b;
          });
      // GATHER weights from the UNBIASED scores (NOT scores_for_choice).
      for (int64_t j = 0; j < topk; ++j) {
        ids[j] = static_cast<int32_t>(order[static_cast<size_t>(j)]);
        w[j] = scores[order[static_cast<size_t>(j)]];
      }
    }

    // renormalize: weights /= max(Σ weights, 1e-20)  (fused_topk_bias_router.py
    // :114-115). Then scale by routed_scaling_factor (:117).
    if (renormalize) {
      float sum = 0.0f;
      for (int64_t j = 0; j < topk; ++j) sum += w[j];
      const float denom = std::max(sum, 1e-20f);
      for (int64_t j = 0; j < topk; ++j) w[j] /= denom;
    }
    for (int64_t j = 0; j < topk; ++j) w[j] *= routed_scaling_factor;
  }
  return out;
}

std::vector<float> ClampedSwiGLU(const std::vector<float>& gate_up, int64_t d,
                                 float limit, float alpha, float beta) {
  // gate = clamp(gate_up[:d], max=limit) ; up = clamp(gate_up[d:], -limit, limit)
  // out  = gate * sigmoid(alpha*gate) * (up + beta)   (activation.py:197-201).
  std::vector<float> out(static_cast<size_t>(d));
  for (int64_t i = 0; i < d; ++i) {
    const float gate = std::min(gate_up[i], limit);                          // MAX only
    const float up = std::min(std::max(gate_up[d + i], -limit), limit);      // BOTH sides
    out[i] = gate * Sigmoid(alpha * gate) * (up + beta);
  }
  return out;
}

}  // namespace vllm::deepseek_v4
