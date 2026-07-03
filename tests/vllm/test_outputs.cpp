// Tests for the public request-result types (vllm/outputs.py @ e24d1b24):
// CompletionOutput / RequestOutput.
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/outputs.h"
#include "vllm/v1/request.h"

using vllm::CompletionOutput;
using vllm::FinishReasonToString;
using vllm::RequestOutput;
using vllm::v1::FinishReason;

TEST_CASE("CompletionOutput defaults mirror upstream dataclass") {
  CompletionOutput out;
  out.index = 0;
  out.text = "hello";
  out.token_ids = {1, 2, 3};
  CHECK(out.index == 0);
  CHECK(out.text == "hello");
  CHECK(out.token_ids.size() == 3);
  // Defaulted optionals start disengaged.
  CHECK_FALSE(out.cumulative_logprob.has_value());
  CHECK_FALSE(out.logprobs.has_value());
  CHECK_FALSE(out.finish_reason.has_value());
  CHECK_FALSE(out.stop_reason.has_value());
  // finished() == finish_reason is not None.
  CHECK_FALSE(out.Finished());
}

TEST_CASE("CompletionOutput.finished() tracks finish_reason") {
  CompletionOutput out;
  CHECK_FALSE(out.Finished());
  out.finish_reason = "stop";
  CHECK(out.Finished());
  out.finish_reason = std::nullopt;
  CHECK_FALSE(out.Finished());
}

TEST_CASE("FinishReason -> string matches upstream FINISH_REASON_STRINGS") {
  CHECK(FinishReasonToString(FinishReason::kStop) == "stop");
  CHECK(FinishReasonToString(FinishReason::kLength) == "length");
  CHECK(FinishReasonToString(FinishReason::kAbort) == "abort");
  CHECK(FinishReasonToString(FinishReason::kError) == "error");
  CHECK(FinishReasonToString(FinishReason::kRepetition) == "repetition");
}

TEST_CASE("CompletionOutput.SetFinishReason uses upstream string mapping") {
  CompletionOutput out;
  out.SetFinishReason(FinishReason::kStop);
  REQUIRE(out.finish_reason.has_value());
  CHECK(*out.finish_reason == "stop");
  CHECK(out.Finished());

  out.SetFinishReason(FinishReason::kLength);
  CHECK(*out.finish_reason == "length");

  out.SetFinishReason(FinishReason::kAbort);
  CHECK(*out.finish_reason == "abort");
}

TEST_CASE("CompletionOutput carries cumulative_logprob and stop_reason") {
  CompletionOutput out;
  out.cumulative_logprob = -1.5;
  out.stop_reason = "</s>";
  REQUIRE(out.cumulative_logprob.has_value());
  CHECK(*out.cumulative_logprob == doctest::Approx(-1.5));
  REQUIRE(out.stop_reason.has_value());
  CHECK(*out.stop_reason == "</s>");
}

TEST_CASE("RequestOutput default construction (generate path fields)") {
  RequestOutput req;
  req.request_id = "req-0";
  req.prompt = "hi";
  req.prompt_token_ids = {10, 11};
  req.finished = false;
  CHECK(req.request_id == "req-0");
  REQUIRE(req.prompt.has_value());
  CHECK(*req.prompt == "hi");
  CHECK(req.prompt_token_ids.size() == 2);
  CHECK(req.outputs.empty());
  // Deferred logprobs placeholder starts disengaged (nullopt), mirroring
  // CompletionOutput.logprobs.
  CHECK_FALSE(req.prompt_logprobs.has_value());
  CHECK_FALSE(req.finished);
  CHECK_FALSE(req.Finished());
}

TEST_CASE("RequestOutput.Finished mirrors the finished flag") {
  RequestOutput req;
  req.finished = false;
  CHECK_FALSE(req.Finished());
  req.finished = true;
  CHECK(req.Finished());
}

TEST_CASE("RequestOutput n>1 shape: outputs indexed 0..n-1") {
  const int n = 3;
  RequestOutput req;
  req.request_id = "req-multi";
  req.prompt_token_ids = {1, 2, 3};
  req.finished = true;
  for (int i = 0; i < n; ++i) {
    CompletionOutput out;
    out.index = i;
    out.text = "seq" + std::to_string(i);
    out.token_ids = {static_cast<int32_t>(100 + i)};
    out.SetFinishReason(FinishReason::kStop);
    req.outputs.push_back(out);
  }
  REQUIRE(req.outputs.size() == static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    CHECK(req.outputs[static_cast<std::size_t>(i)].index == i);
    CHECK(req.outputs[static_cast<std::size_t>(i)].Finished());
    CHECK(*req.outputs[static_cast<std::size_t>(i)].finish_reason == "stop");
  }
  CHECK(req.Finished());
}

TEST_CASE("finish_reason nullopt -> CompletionOutput not finished") {
  CompletionOutput out;
  out.index = 0;
  out.text = "";
  // No finish_reason set: still generating.
  CHECK_FALSE(out.finish_reason.has_value());
  CHECK_FALSE(out.Finished());
}
