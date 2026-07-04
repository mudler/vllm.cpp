// Ported from: vllm/v1/sample/ops/bad_words.py (apply_bad_words,
// _apply_bad_words_single_batch) @ e24d1b24. The allowed-token-ids mask has no
// dedicated upstream file — the Sampler applies it inline
// (vllm/v1/sample/sampler.py:396-397 `logits.masked_fill_(mask, -inf)`); it is
// grouped here with bad_words as the pair of logits exclusion masks the Sampler
// runs before the non-argmax-invariant procs (M1.7 plan Task 3 groups them).
#ifndef VLLM_V1_SAMPLE_OPS_BAD_WORDS_H_
#define VLLM_V1_SAMPLE_OPS_BAD_WORDS_H_

#include <cstdint>
#include <map>
#include <vector>

#include "vt/tensor.h"

namespace vllm::v1 {

// apply_bad_words. For each request `i` with bad-words n-grams, when the last
// (len-1) tokens of the request's output (past_tokens_ids[i]) match a n-gram's
// prefix, the n-gram's final token is blocked (logit set to -inf) for request i.
// n-grams longer than len(output)+1 can never match and are skipped.
void apply_bad_words(vt::Queue& q, vt::Tensor& logits,
                     const std::map<int, std::vector<std::vector<int32_t>>>& bad_words_token_ids,
                     const std::vector<std::vector<int32_t>>& past_tokens_ids);

// apply_allowed_token_ids. mask is [num_reqs][vocab] (row-major), TRUE (!=0) for
// tokens to EXCLUDE (gpu_input_batch fills the row True then clears the allowed
// ids to False). Sets logits[i,j] = -inf wherever mask[i][j] != 0.
void apply_allowed_token_ids(vt::Queue& q, vt::Tensor& logits,
                             const std::vector<std::vector<uint8_t>>& mask);

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_OPS_BAD_WORDS_H_
