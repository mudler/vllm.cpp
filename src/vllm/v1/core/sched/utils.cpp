// Ported from: vllm/v1/core/sched/utils.py @ e24d1b24
// See include/vllm/v1/core/sched/utils.h for the T0 scope + the (verified) stop
// precedence.
#include "vllm/v1/core/sched/utils.h"

#include <cassert>

#include "vllm/sampling_params.h"

namespace vllm::v1 {

bool check_stop(Request& request, int max_model_len) {
  // assert not request.pooling_params — pooling is deferred at T0, so there is
  // nothing to assert against here.
  const SamplingParams& sampling_params = request.sampling_params;

  // (1) min_tokens gate — FIRST at this pin, so it gates the length cap too:
  // do not stop before at least min_tokens have been generated.
  if (request.NumOutputTokens() < sampling_params.min_tokens) {
    return false;
  }

  const int last_token_id = request.output_token_ids.back();

  // (2) EOS. Upstream compares against sampling_params.eos_token_id, which is a
  // *property* returning None when ignore_eos is set (sampling_params.py only
  // assigns _eos_token_id `if not self.ignore_eos`). Our SamplingParams stores
  // the raw eos id plus a separate ignore_eos flag, so we replicate the
  // property's ignore_eos gate here — behaviorally identical to the pin.
  if (!sampling_params.ignore_eos && sampling_params.eos_token_id.has_value() &&
      last_token_id == *sampling_params.eos_token_id) {
    request.status = RequestStatus::kFinishedStopped;
    return true;
  }

  // (3) stop_token_ids. On a match, carry the matched token id as stop_reason
  // (upstream `request.stop_reason = last_token_id`).
  for (const int32_t stop_token_id : sampling_params.stop_token_ids) {
    if (last_token_id == stop_token_id) {
      request.status = RequestStatus::kFinishedStopped;
      request.stop_reason = last_token_id;
      return true;
    }
  }

  // (4) length cap. request.max_tokens == sampling_params.max_tokens upstream
  // (Request.__init__ asserts it is not None for sampling requests), so assert
  // it here too rather than inventing a fallback.
  assert(sampling_params.max_tokens.has_value() &&
         "sampling request must have max_tokens set (Request.max_tokens)");
  if (request.NumTokens() >= max_model_len ||
      request.NumOutputTokens() >= *sampling_params.max_tokens) {
    request.status = RequestStatus::kFinishedLengthCapped;
    return true;
  }

  // (5) repetition detection — DEFERRED (sampling_params.repetition_detection is
  // a deferred field). Upstream would check check_sequence_repetition here.

  return false;
}

}  // namespace vllm::v1
