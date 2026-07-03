// Tests for the SchedulerOutput / NewRequestData / CachedRequestData port
// (vllm/v1/core/sched/output.py @ e24d1b24).
//
// Upstream output.py has no dedicated unit test; these types are exercised via
// tests/v1/core/test_scheduler.py (the new-vs-cached diff shape the model runner
// consumes). Ported here as direct construct-and-check value-carrier tests: the
// diff protocol (full NewRequestData vs diff-only CachedRequestData), the
// from_request field copy, make_empty, and the SchedulerOutput envelope.
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/request.h"

using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::NewRequestData;
using vllm::v1::Request;
using vllm::v1::SchedulerOutput;

namespace {

Request MakeRequest(const std::string& id,
                    std::vector<int32_t> prompt = {1, 2, 3, 4}) {
  SamplingParams params;
  params.max_tokens = 16;
  Request req(id, std::move(prompt), params, /*arrival_time=*/0.0);
  req.num_computed_tokens = 2;
  return req;
}

}  // namespace

TEST_CASE("NewRequestData::from_request copies the T0 fields + block_ids") {
  Request req = MakeRequest("req-0", {10, 11, 12});
  // Per-group block ids (one group here).
  std::vector<std::vector<int>> block_ids = {{7, 8, 9}};

  NewRequestData data = NewRequestData::from_request(req, block_ids);

  CHECK(data.req_id == "req-0");
  REQUIRE(data.prompt_token_ids.has_value());
  CHECK(data.prompt_token_ids.value() == std::vector<int32_t>{10, 11, 12});
  REQUIRE(data.sampling_params.has_value());
  CHECK(data.sampling_params->max_tokens == 16);
  CHECK(data.num_computed_tokens == 2);
  REQUIRE(data.block_ids.size() == 1);
  CHECK(data.block_ids[0] == std::vector<int>{7, 8, 9});
}

TEST_CASE("NewRequestData carries multi-group block_ids by group") {
  Request req = MakeRequest("req-mg");
  std::vector<std::vector<int>> block_ids = {{1, 2}, {3, 4, 5}};

  NewRequestData data = NewRequestData::from_request(req, block_ids);

  REQUIRE(data.block_ids.size() == 2);
  CHECK(data.block_ids[0] == std::vector<int>{1, 2});
  CHECK(data.block_ids[1] == std::vector<int>{3, 4, 5});
}

TEST_CASE("CachedRequestData::make_empty is the empty diff") {
  CachedRequestData cached = CachedRequestData::make_empty();

  CHECK(cached.num_reqs() == 0);
  CHECK(cached.req_ids.empty());
  CHECK(cached.resumed_req_ids.empty());
  CHECK(cached.new_token_ids.empty());
  CHECK(cached.all_token_ids.empty());
  CHECK(cached.new_block_ids.empty());
  CHECK(cached.num_computed_tokens.empty());
  CHECK(cached.num_output_tokens.empty());
}

TEST_CASE("CachedRequestData diff shape: parallel arrays over req_ids") {
  CachedRequestData cached;
  cached.req_ids = {"a", "b"};
  // "b" is resumed from preemption -> its block table is REPLACED.
  cached.resumed_req_ids = {"b"};
  // Per request: newly allocated per-group block ids; nullopt = none this step.
  cached.new_block_ids = {
      std::optional<std::vector<std::vector<int>>>{{{20, 21}}},  // a: append
      std::optional<std::vector<std::vector<int>>>{{{30}}},      // b: replace
  };
  cached.num_computed_tokens = {5, 8};
  cached.num_output_tokens = {1, 0};

  CHECK(cached.num_reqs() == 2);
  // "a" is appended (not resumed); "b" is resumed (replace).
  CHECK(cached.resumed_req_ids.count("a") == 0);
  CHECK(cached.resumed_req_ids.count("b") == 1);

  REQUIRE(cached.new_block_ids.size() == 2);
  REQUIRE(cached.new_block_ids[0].has_value());
  CHECK(cached.new_block_ids[0].value()[0] == std::vector<int>{20, 21});
  REQUIRE(cached.new_block_ids[1].has_value());
  CHECK(cached.new_block_ids[1].value()[0] == std::vector<int>{30});

  CHECK(cached.num_computed_tokens == std::vector<int>{5, 8});
  CHECK(cached.num_output_tokens == std::vector<int>{1, 0});
}

TEST_CASE("CachedRequestData new_block_ids nullopt = no new blocks this step") {
  CachedRequestData cached;
  cached.req_ids = {"only"};
  cached.new_block_ids = {std::nullopt};
  cached.num_computed_tokens = {3};
  cached.num_output_tokens = {2};

  REQUIRE(cached.new_block_ids.size() == 1);
  CHECK_FALSE(cached.new_block_ids[0].has_value());
}

TEST_CASE("CachedRequestData::is_context_phase reflects num_output_tokens") {
  CachedRequestData cached;
  cached.req_ids = {"prefill", "decode"};
  cached.num_output_tokens = {0, 4};

  // prefill: still 0 output tokens -> context (prefill) phase.
  CHECK(cached.is_context_phase("prefill"));
  // decode: has output tokens -> not context phase.
  CHECK_FALSE(cached.is_context_phase("decode"));
  // unknown req_id -> false.
  CHECK_FALSE(cached.is_context_phase("missing"));
}

TEST_CASE("SchedulerOutput::make_empty is an empty step") {
  SchedulerOutput out = SchedulerOutput::make_empty();

  CHECK(out.scheduled_new_reqs.empty());
  CHECK(out.scheduled_cached_reqs.num_reqs() == 0);
  CHECK(out.num_scheduled_tokens.empty());
  CHECK(out.total_num_scheduled_tokens == 0);
  CHECK(out.scheduled_spec_decode_tokens.empty());
  CHECK(out.scheduled_encoder_inputs.empty());
  CHECK(out.num_common_prefix_blocks.empty());
  CHECK(out.finished_req_ids.empty());
  CHECK(out.free_encoder_mm_hashes.empty());
}

TEST_CASE("SchedulerOutput carries new + cached reqs, token map, finished ids") {
  Request new_req = MakeRequest("new-1", {100, 101});

  SchedulerOutput out;
  out.scheduled_new_reqs.push_back(
      NewRequestData::from_request(new_req, {{42}}));

  CachedRequestData cached;
  cached.req_ids = {"cached-1"};
  cached.new_block_ids = {std::optional<std::vector<std::vector<int>>>{{{43}}}};
  cached.num_computed_tokens = {6};
  cached.num_output_tokens = {3};
  out.scheduled_cached_reqs = cached;

  out.num_scheduled_tokens = {{"new-1", 2}, {"cached-1", 1}};
  out.total_num_scheduled_tokens = 3;
  out.num_common_prefix_blocks = {0};
  out.finished_req_ids = {"done-1"};

  // New reqs carry FULL data.
  REQUIRE(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "new-1");
  CHECK(out.scheduled_new_reqs[0].block_ids[0] == std::vector<int>{42});

  // Cached reqs carry only the DIFF.
  CHECK(out.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(out.scheduled_cached_reqs.req_ids[0] == "cached-1");

  // Token accounting: total == sum of the per-request map.
  int sum = 0;
  for (const auto& [id, n] : out.num_scheduled_tokens) sum += n;
  CHECK(sum == out.total_num_scheduled_tokens);
  CHECK(out.num_scheduled_tokens.at("new-1") == 2);
  CHECK(out.num_scheduled_tokens.at("cached-1") == 1);

  CHECK(out.finished_req_ids.count("done-1") == 1);
}
