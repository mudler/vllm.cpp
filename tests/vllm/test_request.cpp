// Tests for the Request / RequestStatus / FinishReason port
// (vllm/v1/request.py @ e24d1b24).
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "vllm/v1/request.h"

using vllm::v1::FinishReason;
using vllm::v1::Request;
using vllm::v1::RequestStatus;

TEST_CASE("RequestStatus int values mirror upstream IntEnum ordering") {
  // enum.auto() starts at 1; ordering is load-bearing for is_finished.
  CHECK(static_cast<int>(RequestStatus::kWaiting) == 1);
  CHECK(static_cast<int>(RequestStatus::kWaitingForStructuredOutputGrammar) == 2);
  CHECK(static_cast<int>(RequestStatus::kWaitingForRemoteKvs) == 3);
  CHECK(static_cast<int>(RequestStatus::kWaitingForStreamingReq) == 4);
  CHECK(static_cast<int>(RequestStatus::kRunning) == 5);
  CHECK(static_cast<int>(RequestStatus::kPreempted) == 6);
  CHECK(static_cast<int>(RequestStatus::kFinishedStopped) == 7);
  CHECK(static_cast<int>(RequestStatus::kFinishedLengthCapped) == 8);
  CHECK(static_cast<int>(RequestStatus::kFinishedAborted) == 9);
  CHECK(static_cast<int>(RequestStatus::kFinishedIgnored) == 10);
  CHECK(static_cast<int>(RequestStatus::kFinishedError) == 11);
  CHECK(static_cast<int>(RequestStatus::kFinishedRepetition) == 12);
}

TEST_CASE("RequestStatus::IsFinished matches upstream (status > PREEMPTED)") {
  // Non-finished (<= PREEMPTED).
  CHECK_FALSE(vllm::v1::IsFinished(RequestStatus::kWaiting));
  CHECK_FALSE(vllm::v1::IsFinished(
      RequestStatus::kWaitingForStructuredOutputGrammar));
  CHECK_FALSE(vllm::v1::IsFinished(RequestStatus::kWaitingForRemoteKvs));
  CHECK_FALSE(
      vllm::v1::IsFinished(RequestStatus::kWaitingForStreamingReq));
  CHECK_FALSE(vllm::v1::IsFinished(RequestStatus::kRunning));
  CHECK_FALSE(vllm::v1::IsFinished(RequestStatus::kPreempted));
  // Finished (> PREEMPTED).
  CHECK(vllm::v1::IsFinished(RequestStatus::kFinishedStopped));
  CHECK(vllm::v1::IsFinished(RequestStatus::kFinishedLengthCapped));
  CHECK(vllm::v1::IsFinished(RequestStatus::kFinishedAborted));
  CHECK(vllm::v1::IsFinished(RequestStatus::kFinishedIgnored));
  CHECK(vllm::v1::IsFinished(RequestStatus::kFinishedError));
  CHECK(vllm::v1::IsFinished(RequestStatus::kFinishedRepetition));
}

TEST_CASE("FinishReason int values mirror upstream IntEnum") {
  CHECK(static_cast<int>(FinishReason::kStop) == 0);
  CHECK(static_cast<int>(FinishReason::kLength) == 1);
  CHECK(static_cast<int>(FinishReason::kAbort) == 2);
  CHECK(static_cast<int>(FinishReason::kError) == 3);
  CHECK(static_cast<int>(FinishReason::kRepetition) == 4);
}

TEST_CASE("RequestStatus::GetFinishedReason mirrors _FINISHED_REASON_MAP") {
  // Finished statuses map per upstream.
  CHECK(vllm::v1::GetFinishedReason(RequestStatus::kFinishedStopped) ==
        FinishReason::kStop);
  CHECK(vllm::v1::GetFinishedReason(
            RequestStatus::kFinishedLengthCapped) == FinishReason::kLength);
  CHECK(vllm::v1::GetFinishedReason(RequestStatus::kFinishedAborted) ==
        FinishReason::kAbort);
  // FINISHED_IGNORED -> LENGTH (prompt longer than the model cap).
  CHECK(vllm::v1::GetFinishedReason(RequestStatus::kFinishedIgnored) ==
        FinishReason::kLength);
  CHECK(vllm::v1::GetFinishedReason(RequestStatus::kFinishedError) ==
        FinishReason::kError);
  CHECK(vllm::v1::GetFinishedReason(RequestStatus::kFinishedRepetition) ==
        FinishReason::kRepetition);
  // WAITING_FOR_STREAMING_REQ is IN the map (-> STOP) though NOT finished.
  CHECK(vllm::v1::GetFinishedReason(
            RequestStatus::kWaitingForStreamingReq) == FinishReason::kStop);
  // Non-mapped statuses return nullopt.
  CHECK_FALSE(
      vllm::v1::GetFinishedReason(RequestStatus::kRunning).has_value());
  CHECK_FALSE(
      vllm::v1::GetFinishedReason(RequestStatus::kWaiting).has_value());
  CHECK_FALSE(vllm::v1::GetFinishedReason(RequestStatus::kPreempted)
                  .has_value());
}

namespace {
Request MakeRequest(std::vector<int32_t> prompt) {
  vllm::SamplingParams params;
  // The model's EOS token id rides on sampling_params (upstream
  // SamplingParams._eos_token_id), engine-populated before construction.
  params.eos_token_id = 2;
  return Request("req-0", std::move(prompt), params,
                 /*arrival_time=*/123.0);
}
}  // namespace

TEST_CASE("Request construction sets defaults and prompt count") {
  Request req = MakeRequest({10, 11, 12, 13});
  CHECK(req.request_id == "req-0");
  CHECK(req.num_prompt_tokens == 4);
  CHECK(req.prompt_token_ids.size() == 4);
  CHECK(req.output_token_ids.empty());
  CHECK(req.num_computed_tokens == 0);
  CHECK(req.status == RequestStatus::kWaiting);
  // EOS lives on sampling_params, not on Request (upstream fidelity).
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 2);
  CHECK(req.arrival_time == doctest::Approx(123.0));
  CHECK_FALSE(req.IsFinished());
}

TEST_CASE("Request token counting: NumTokens = prompt + output") {
  Request req = MakeRequest({1, 2, 3});
  CHECK(req.NumTokens() == 3);
  CHECK(req.NumOutputTokens() == 0);

  req.AppendOutputToken(42);
  CHECK(req.NumOutputTokens() == 1);
  CHECK(req.NumTokens() == 4);
  CHECK(req.output_token_ids.back() == 42);

  req.AppendOutputToken(43);
  CHECK(req.NumOutputTokens() == 2);
  CHECK(req.NumTokens() == 5);
}

TEST_CASE("Request AppendOutputToken (list form) extends output") {
  Request req = MakeRequest({7});
  req.AppendOutputToken(std::vector<int32_t>{100, 101, 102});
  CHECK(req.NumOutputTokens() == 3);
  CHECK(req.NumTokens() == 4);
  CHECK(req.output_token_ids[0] == 100);
  CHECK(req.output_token_ids[2] == 102);
}

TEST_CASE("Request IsFinished / GetFinishedReason track status") {
  Request req = MakeRequest({1, 2});
  CHECK_FALSE(req.IsFinished());
  CHECK_FALSE(req.GetFinishedReason().has_value());

  req.status = RequestStatus::kRunning;
  CHECK_FALSE(req.IsFinished());
  CHECK_FALSE(req.GetFinishedReason().has_value());

  req.status = RequestStatus::kFinishedStopped;
  CHECK(req.IsFinished());
  REQUIRE(req.GetFinishedReason().has_value());
  CHECK(*req.GetFinishedReason() == FinishReason::kStop);

  req.status = RequestStatus::kFinishedLengthCapped;
  CHECK(req.IsFinished());
  CHECK(*req.GetFinishedReason() == FinishReason::kLength);

  req.status = RequestStatus::kFinishedAborted;
  CHECK(req.IsFinished());
  CHECK(*req.GetFinishedReason() == FinishReason::kAbort);
}
