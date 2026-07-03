// Ported from: vllm/v1/request.py @ e24d1b24
// (FinishReason / _FINISHED_REASON_MAP from vllm/v1/engine/__init__.py @
// e24d1b24). See include/vllm/v1/request.h for the deferred-field / deviation
// notes.
#include "vllm/v1/request.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vllm::v1 {

// RequestStatus.is_finished: finished iff status > PREEMPTED. Relies on the
// enum ordering (every value after PREEMPTED is a FINISHED_* status).
bool IsFinished(RequestStatus status) {
  return static_cast<int>(status) > static_cast<int>(RequestStatus::kPreempted);
}

// RequestStatus.get_finished_reason: _FINISHED_REASON_MAP.get(status).
// NOTE: the ignored requests are those whose prompt lengths exceed the model's
// length cap, so their reason is "length" (as in the OpenAI API).
std::optional<FinishReason> GetFinishedReason(RequestStatus status) {
  switch (status) {
    case RequestStatus::kFinishedStopped:
      return FinishReason::kStop;
    case RequestStatus::kFinishedLengthCapped:
      return FinishReason::kLength;
    case RequestStatus::kFinishedAborted:
      return FinishReason::kAbort;
    case RequestStatus::kFinishedIgnored:
      return FinishReason::kLength;
    case RequestStatus::kFinishedError:
      return FinishReason::kError;
    case RequestStatus::kFinishedRepetition:
      return FinishReason::kRepetition;
    case RequestStatus::kWaitingForStreamingReq:
      return FinishReason::kStop;
    // Statuses not present in _FINISHED_REASON_MAP -> nullopt (dict.get default).
    case RequestStatus::kWaiting:
    case RequestStatus::kWaitingForStructuredOutputGrammar:
    case RequestStatus::kWaitingForRemoteKvs:
    case RequestStatus::kRunning:
    case RequestStatus::kPreempted:
      return std::nullopt;
  }
  return std::nullopt;
}

Request::Request(std::string request_id,
                 std::vector<int32_t> prompt_token_ids,
                 SamplingParams sampling_params,
                 std::optional<int32_t> eos_token_id, double arrival_time)
    : request_id(std::move(request_id)),
      prompt_token_ids(std::move(prompt_token_ids)),
      sampling_params(std::move(sampling_params)),
      num_computed_tokens(0),
      status(RequestStatus::kWaiting),
      eos_token_id(eos_token_id),
      arrival_time(arrival_time),
      num_prompt_tokens(static_cast<int>(this->prompt_token_ids.size())) {}

// num_tokens: len(_all_token_ids) == prompt + output.
int Request::NumTokens() const {
  return num_prompt_tokens + static_cast<int>(output_token_ids.size());
}

// num_output_tokens: len(_output_token_ids).
int Request::NumOutputTokens() const {
  return static_cast<int>(output_token_ids.size());
}

// append_output_token_ids(int).
void Request::AppendOutputToken(int32_t token_id) {
  output_token_ids.push_back(token_id);
}

// append_output_token_ids(list[int]).
void Request::AppendOutputToken(const std::vector<int32_t>& token_ids) {
  output_token_ids.insert(output_token_ids.end(), token_ids.begin(),
                          token_ids.end());
}

bool Request::IsFinished() const { return vllm::v1::IsFinished(status); }

std::optional<FinishReason> Request::GetFinishedReason() const {
  return vllm::v1::GetFinishedReason(status);
}

}  // namespace vllm::v1
