// Ported from: vllm/v1/sample/logits_processor/builtin.py @ e24d1b24
// (MinTokensLogitsProcessor, LogitBiasLogitsProcessor, MinPLogitsProcessor).
//
// The three T0 builtin logits processors, ported as direct functions over the
// logits [num_reqs, vocab] (row-major f32) rather than the plugin object graph
// (see the metadata.h deviation note). Each carries its upstream
// `is_argmax_invariant()` as a constexpr — this drives the Sampler pipeline
// placement (M1.7 Task 4): non-argmax-invariant procs run BEFORE the greedy
// argmax snapshot; argmax-invariant procs run inside sample() after temperature.
#ifndef VLLM_V1_SAMPLE_LOGITS_PROCESSOR_BUILTIN_H_
#define VLLM_V1_SAMPLE_LOGITS_PROCESSOR_BUILTIN_H_

#include <cstdint>
#include <map>
#include <vector>

#include "vllm/v1/sample/metadata.h"
#include "vt/tensor.h"

namespace vllm::v1 {

// MinTokensLogitsProcessor.is_argmax_invariant() -> False (censoring stop tokens
// can change the greedy argmax).
inline constexpr bool kMinTokensArgmaxInvariant = false;
// LogitBiasLogitsProcessor.is_argmax_invariant() -> False (bias can rebalance the
// argmax).
inline constexpr bool kLogitBiasArgmaxInvariant = false;
// MinPLogitsProcessor.is_argmax_invariant() -> True (min-p never masks the
// max-prob token, so greedy is unaffected).
inline constexpr bool kMinPArgmaxInvariant = true;

// MinTokensLogitsProcessor.apply. For each tracked request whose generated
// length is still below its min_tokens floor, mask every id in its stop-token
// set (eos + stop_token_ids) to -inf. `output_token_ids[i]` gives the generated
// length for request i.
void apply_min_tokens(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, MinTokensState>& min_tokens,
                      const std::vector<std::vector<int32_t>>& output_token_ids);

// LogitBiasLogitsProcessor.apply. Adds the per-(request, token) additive bias:
// logits[i, tok] += bias for every entry in `logit_bias`.
void apply_logit_bias(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, std::map<int32_t, float>>& logit_bias);

// MinPLogitsProcessor.apply. Per row: mask tokens whose softmax probability is
// below min_p[i] * max_prob to -inf. Rows with min_p[i] == 0 are unaffected.
void apply_min_p(vt::Queue& q, vt::Tensor& logits, const std::vector<float>& min_p);

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_LOGITS_PROCESSOR_BUILTIN_H_
