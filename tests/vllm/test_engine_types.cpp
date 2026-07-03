// Tests for the EngineCore I/O types
// (vllm/v1/engine/__init__.py + vllm/v1/outputs.py @ e24d1b24).
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/request.h"

using vllm::SamplingParams;
using vllm::v1::EngineCoreOutput;
using vllm::v1::EngineCoreOutputs;
using vllm::v1::EngineCoreRequest;
using vllm::v1::FinishReason;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::Request;
using vllm::v1::RequestStatus;
using vllm::v1::SamplerOutput;

TEST_CASE("EngineCoreRequest carries the frontend->core T0 fields") {
  SamplingParams params;
  params.eos_token_id = 2;
  EngineCoreRequest req{
      /*request_id=*/"req-0",
      /*prompt_token_ids=*/{10, 11, 12},
      /*sampling_params=*/params,
      /*arrival_time=*/123.5,
  };
  CHECK(req.request_id == "req-0");
  CHECK(req.prompt_token_ids.size() == 3);
  CHECK(req.prompt_token_ids[0] == 10);
  CHECK(req.arrival_time == doctest::Approx(123.5));
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 2);
}

TEST_CASE("Request::FromEngineCoreRequest mirrors upstream construction") {
  SamplingParams params;
  params.eos_token_id = 7;
  params.max_tokens = 32;
  EngineCoreRequest core_req{
      /*request_id=*/"req-1",
      /*prompt_token_ids=*/{1, 2, 3, 4},
      /*sampling_params=*/params,
      /*arrival_time=*/9.0,
  };

  Request req = Request::FromEngineCoreRequest(core_req);

  CHECK(req.request_id == "req-1");
  CHECK(req.prompt_token_ids.size() == 4);
  CHECK(req.prompt_token_ids[3] == 4);
  CHECK(req.num_prompt_tokens == 4);
  CHECK(req.status == RequestStatus::kWaiting);
  CHECK(req.num_computed_tokens == 0);
  CHECK(req.output_token_ids.empty());
  CHECK(req.arrival_time == doctest::Approx(9.0));
  // sampling_params copied through (incl. engine-populated eos_token_id).
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 7);
  CHECK(req.sampling_params.max_tokens == 32);
  CHECK_FALSE(req.IsFinished());
}

TEST_CASE("SamplerOutput holds a per-request row of sampled token ids") {
  // [num_reqs, max_num_generated_tokens]; T0 non-spec is [num_reqs, 1].
  SamplerOutput out;
  out.sampled_token_ids = {{42}, {43}, {44}};
  CHECK(out.sampled_token_ids.size() == 3);  // num_reqs
  CHECK(out.sampled_token_ids[0].size() == 1);
  CHECK(out.sampled_token_ids[0][0] == 42);
  CHECK(out.sampled_token_ids[2][0] == 44);
  CHECK_FALSE(out.logprobs_tensors.has_value());
}

TEST_CASE("ModelRunnerOutput carries req_ids order + sampled ids + index map") {
  ModelRunnerOutput out;
  out.req_ids = {"a", "b"};
  out.req_id_to_index = {{"a", 0}, {"b", 1}};
  out.sampled_token_ids = {{100}, {200}};

  REQUIRE(out.req_ids.size() == 2);
  CHECK(out.req_ids[0] == "a");
  CHECK(out.req_id_to_index.at("b") == 1);
  REQUIRE(out.sampled_token_ids.size() == 2);
  CHECK(out.sampled_token_ids[1][0] == 200);
}

TEST_CASE("EngineCoreOutput: finished() is (finish_reason has value)") {
  EngineCoreOutput out;
  out.request_id = "req-2";
  out.new_token_ids = {5, 6};
  CHECK(out.request_id == "req-2");
  CHECK(out.new_token_ids.size() == 2);
  CHECK_FALSE(out.finish_reason.has_value());
  CHECK_FALSE(out.Finished());

  out.finish_reason = FinishReason::kStop;
  CHECK(out.Finished());
  CHECK(*out.finish_reason == FinishReason::kStop);
}

TEST_CASE("EngineCoreOutputs collects per-request outputs") {
  EngineCoreOutputs outs;
  CHECK(outs.engine_index == 0);
  CHECK(outs.outputs.empty());

  EngineCoreOutput a;
  a.request_id = "a";
  a.new_token_ids = {1};
  a.finish_reason = FinishReason::kLength;
  outs.outputs.push_back(a);

  EngineCoreOutput b;
  b.request_id = "b";
  b.new_token_ids = {2, 3};
  outs.outputs.push_back(b);

  REQUIRE(outs.outputs.size() == 2);
  CHECK(outs.outputs[0].Finished());
  CHECK_FALSE(outs.outputs[1].Finished());
}
