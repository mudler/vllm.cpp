// Ported from: vllm/v1/core/sched/utils.py @ e24d1b24
//
// Scope (M1.4 Task 4): the token-level stop check the scheduler runs after each
// sampled token. check_stop is the only piece of sched/utils.py the V1 engine's
// update_from_output path needs at T0.
//
// DEFERRED (marked; the upstream file's other helpers are not needed at T0):
//   - _has_repeating_pattern / check_sequence_repetition (the
//     repetition-detection path — sampling_params.repetition_detection is a
//     deferred SamplingParams field, so check_stop's repetition branch is
//     omitted here).
//   - remove_all (the list-removal helper) — update_from_output removes stopped
//     requests from the running vector / waiting queue directly.
//
// IMPORTANT (stop precedence, verified against the pin — NOT the summary in the
// plan): the pinned check_stop order is
//   1. min_tokens gate: num_output_tokens < min_tokens -> return false FIRST
//      (this gates *everything*, including the length cap, at this pin);
//   2. eos: last token == eos_token_id -> FINISHED_STOPPED;
//   3. stop_token_ids: last token in stop_token_ids -> FINISHED_STOPPED
//      (stop_reason = the matched token id);
//   4. length cap: num_tokens >= max_model_len OR num_output_tokens >= max_tokens
//      -> FINISHED_LENGTH_CAPPED;
//   5. repetition (DEFERRED).
#ifndef VLLM_V1_CORE_SCHED_UTILS_H_
#define VLLM_V1_CORE_SCHED_UTILS_H_

#include "vllm/v1/request.h"

namespace vllm::v1 {

// check_stop(request, max_model_len): returns true and sets request.status
// (and, for a stop_token_ids match, request.stop_reason) when the request has
// hit a stop condition after its latest sampled token. Mirrors
// vllm/v1/core/sched/utils.py::check_stop for the T0 subset (pooling and
// repetition-detection branches deferred). Mutates the request exactly as
// upstream does.
bool check_stop(Request& request, int max_model_len);

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_SCHED_UTILS_H_
