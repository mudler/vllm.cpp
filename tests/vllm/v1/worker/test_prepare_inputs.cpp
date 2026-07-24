// Tests for the step-input build (M1.5 Task 3) — update_states (apply a
// SchedulerOutput to the persistent InputBatch) + prepare_inputs (build the
// flattened per-step inputs the attention op / sampler consume).
//
// Ported from vllm/v1/worker/gpu_model_runner.py @ e24d1b24. The update_states
// admission/removal cases mirror tests/v1/worker/test_gpu_model_runner.py
// (test_update_states_new_request / _request_unscheduled: the
// _schedule_new_request + _is_req_scheduled / _is_req_added pattern, and the
// finished/unscheduled removal). The upstream _prepare_inputs itself runs on
// GPU; here — host arrays, no CUDA — the prefill-chunk + decode oracle is
// derived directly from the pinned _prepare_inputs algorithm (query_start_loc =
// [0] ++ cumsum, positions = num_computed + arange, slot = block_id*block_size +
// offset, seq_lens = num_computed + num_scheduled, logits_indices =
// query_start_loc[1:] - 1). See prepare_inputs.h for the deferred paths.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/worker/gpu/input_batch.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::CachedRequestState;
using vllm::v1::InputBatch;
using vllm::v1::NewRequestData;
using vllm::v1::prepare_inputs;
using vllm::v1::SchedulerOutput;
using vllm::v1::StepInputs;
using vllm::v1::update_states;

namespace {

constexpr int kBlockSize = 16;

InputBatch make_batch(int max_num_reqs = 8, int max_model_len = 64) {
  return InputBatch(/*max_num_reqs=*/max_num_reqs,
                    /*max_model_len=*/max_model_len,
                    /*max_num_batched_tokens=*/max_model_len,
                    /*vocab_size=*/1024, /*block_sizes=*/{kBlockSize},
                    /*kernel_block_sizes=*/{kBlockSize});
}

SamplingParams default_params() {
  SamplingParams sp;      // temperature 1.0 => random (M1.1 contract).
  sp.PostInit();          // Feed a PostInit-normalized params object.
  return sp;
}

// A new-request payload (MRV2 contract): prefill_token_ids == prompt ++ output.
NewRequestData make_new_req(const std::string& req_id,
                            std::vector<int32_t> prompt,
                            std::vector<int32_t> output,
                            std::vector<int> block_ids, int num_computed) {
  NewRequestData nr;
  nr.req_id = req_id;
  std::vector<int32_t> all_ids = prompt;
  all_ids.insert(all_ids.end(), output.begin(), output.end());
  nr.prompt_token_ids = std::move(prompt);
  nr.sampling_params = default_params();
  nr.block_ids = {std::move(block_ids)};
  nr.num_computed_tokens = num_computed;
  nr.prefill_token_ids = std::move(all_ids);
  return nr;
}

// Seed a CachedRequestState directly into the batch (prompt then output).
int add_direct(InputBatch& batch, const std::string& req_id,
               std::vector<int32_t> prompt, std::vector<int32_t> output,
               std::vector<int> block_ids, int num_computed) {
  CachedRequestState state;
  state.req_id = req_id;
  state.prompt_token_ids = std::move(prompt);
  state.output_token_ids = std::move(output);
  state.sampling_params = default_params();
  state.block_ids = {std::move(block_ids)};
  state.num_computed_tokens = num_computed;
  state.finalize();
  return batch.add_request(state);
}

SchedulerOutput make_output(std::map<std::string, int> num_scheduled) {
  SchedulerOutput so;
  so.scheduled_cached_reqs = CachedRequestData::make_empty();
  int total = 0;
  for (const auto& [req_id, n] : num_scheduled) {
    total += n;
  }
  so.num_scheduled_tokens = std::move(num_scheduled);
  so.total_num_scheduled_tokens = total;
  return so;
}

}  // namespace

// ─── update_states ──────────────────────────────────────────────────────────

TEST_CASE("update_states admits a new request (from prefill_token_ids)") {
  InputBatch batch = make_batch();
  SchedulerOutput so = make_output({{"req_0", 3}});
  so.scheduled_new_reqs.push_back(
      make_new_req("req_0", {1, 2, 3}, {}, {0}, /*num_computed=*/0));

  update_states(batch, so);

  CHECK(batch.num_reqs() == 1);
  REQUIRE(batch.req_id_to_index.count("req_0") == 1);
  CHECK(batch.req_id_to_index.at("req_0") == 0);
  CHECK(batch.token_id(0, 0) == 1);
  CHECK(batch.token_id(0, 2) == 3);
  CHECK(batch.num_computed_tokens_cpu[0] == 0);
  CHECK(batch.block_table[0].num_blocks_per_row[0] == 1);
  CHECK(batch.block_table[0].cpu_block_id(0, 0) == 0);
}

TEST_CASE("update_states removes finished requests and condenses") {
  InputBatch batch = make_batch();
  add_direct(batch, "a", {1}, {}, {10}, 1);
  add_direct(batch, "b", {2}, {}, {20}, 1);
  add_direct(batch, "c", {3}, {}, {30}, 1);
  REQUIRE(batch.num_reqs() == 3);

  // Finish "b"; "a" and "c" stay scheduled so they are not treated as
  // unscheduled.
  SchedulerOutput so = make_output({{"a", 1}, {"c", 1}});
  so.finished_req_ids = {"b"};

  update_states(batch, so);

  CHECK(batch.num_reqs() == 2);
  CHECK(batch.req_id_to_index.count("b") == 0);
  CHECK(batch.req_id_to_index.at("a") == 0);
  // "c" slides down into the freed slot 1.
  CHECK(batch.req_id_to_index.at("c") == 1);
  REQUIRE(batch.req_ids[1].has_value());
  CHECK(*batch.req_ids[1] == "c");
  CHECK(batch.block_table[0].cpu_block_id(1, 0) == 30);
}

TEST_CASE("update_states removes unscheduled requests") {
  InputBatch batch = make_batch();
  add_direct(batch, "a", {1}, {}, {10}, 1);
  add_direct(batch, "b", {2}, {}, {20}, 1);

  // Only "a" is scheduled this step; "b" is unscheduled and removed.
  SchedulerOutput so = make_output({{"a", 1}});
  update_states(batch, so);

  CHECK(batch.num_reqs() == 1);
  CHECK(batch.req_id_to_index.at("a") == 0);
  CHECK(batch.req_id_to_index.count("b") == 0);
}

TEST_CASE("update_states applies a cached diff (num_computed + new_block_ids)") {
  InputBatch batch = make_batch();
  add_direct(batch, "a", {1, 2, 3}, {}, {0}, /*num_computed=*/0);
  REQUIRE(batch.block_table[0].num_blocks_per_row[0] == 1);

  SchedulerOutput so = make_output({{"a", 1}});
  CachedRequestData cached;
  cached.req_ids = {"a"};
  cached.num_computed_tokens = {3};
  cached.num_output_tokens = {0};
  std::vector<std::vector<int>> appended_blocks = {{5}};  // one group, block 5
  cached.new_block_ids.emplace_back(std::move(appended_blocks));
  so.scheduled_cached_reqs = cached;

  update_states(batch, so);

  CHECK(batch.num_computed_tokens_cpu[0] == 3);
  // Appended block 5 to the existing row [0] -> [0, 5].
  CHECK(batch.block_table[0].num_blocks_per_row[0] == 2);
  CHECK(batch.block_table[0].cpu_block_id(0, 0) == 0);
  CHECK(batch.block_table[0].cpu_block_id(0, 1) == 5);
}

// ─── prepare_inputs: prefill chunk + decode oracle ──────────────────────────

TEST_CASE("prepare_inputs: prefill chunk (N=4) + decode (1) oracle") {
  InputBatch batch = make_batch();

  // Req "A": chunked prefill, prompt of 10 tokens, num_computed=0, computing
  // the first N=4 tokens (positions 0..3). Block [3].
  add_direct(batch, "A", {100, 101, 102, 103, 104, 105, 106, 107, 108, 109}, {},
             {3}, /*num_computed=*/0);
  // Req "B": decode, prompt of 5 + 1 already-produced output token (206 cached
  // at column 5), num_computed=5, computing 1 token (position 5). Block [7].
  add_direct(batch, "B", {200, 201, 202, 203, 204}, {205}, {7},
             /*num_computed=*/5);
  REQUIRE(batch.num_reqs() == 2);

  // A no-op update (both scheduled, nothing new/finished) then the build.
  const int kN = 4;
  SchedulerOutput so = make_output({{"A", kN}, {"B", 1}});
  update_states(batch, so);
  REQUIRE(batch.num_reqs() == 2);

  StepInputs step = prepare_inputs(batch, so);

  // num_scheduled_tokens in batch order.
  CHECK(step.num_scheduled_tokens == std::vector<int32_t>{4, 1});

  // query_start_loc = [0, N, N+1].
  CHECK(step.query_start_loc == std::vector<int32_t>{0, 4, 5});

  // seq_lens = [num_computed+N, num_computed+1] = [0+4, 5+1].
  CHECK(step.seq_lens == std::vector<int32_t>{4, 6});

  // positions = per-req arange from num_computed: A -> 0..3, B -> 5.
  CHECK(step.positions == std::vector<int64_t>{0, 1, 2, 3, 5});

  // input_token_ids = the scheduled tokens gathered from token_ids_cpu.
  CHECK(step.input_token_ids ==
        std::vector<int32_t>{100, 101, 102, 103, 205});

  // logits_indices = query_start_loc[1:] - 1 = [N-1, N] = [3, 4].
  CHECK(step.logits_indices == std::vector<int32_t>{3, 4});

  // slot_mapping (single group) = block_id*block_size + within-block offset.
  // A: block 3 -> 3*16 + {0,1,2,3} = 48,49,50,51; B: block 7 -> 7*16 + 5 = 117.
  REQUIRE(step.slot_mapping.size() == 1);
  CHECK(step.slot_mapping[0] ==
        std::vector<int64_t>{48, 49, 50, 51, 117});
}

TEST_CASE("prepare_inputs: single decode step (num_scheduled=1)") {
  InputBatch batch = make_batch();
  // Decode of a request that has computed 3 tokens; next token cached at col 3.
  add_direct(batch, "r", {10, 11, 12}, {13}, {2}, /*num_computed=*/3);

  SchedulerOutput so = make_output({{"r", 1}});
  update_states(batch, so);
  StepInputs step = prepare_inputs(batch, so);

  CHECK(step.query_start_loc == std::vector<int32_t>{0, 1});
  CHECK(step.seq_lens == std::vector<int32_t>{4});
  CHECK(step.positions == std::vector<int64_t>{3});
  CHECK(step.input_token_ids == std::vector<int32_t>{13});
  CHECK(step.logits_indices == std::vector<int32_t>{0});
  // block 2 -> 2*16 + 3 = 35.
  REQUIRE(step.slot_mapping.size() == 1);
  CHECK(step.slot_mapping[0] == std::vector<int64_t>{35});
}

// ─── logits expansion under spec decode (SPEC-REJECTION I3) ─────────────────
// Ported from gpu/model_runner.py:866-898 (the cu_num_logits build) + the
// logits-index formula of _combine_sampled_and_draft_tokens_kernel:317-327.
// The DEFAULT-OFF byte-identity property is asserted first: with no
// scheduled_spec_decode_tokens the expansion reduces to the pre-I3 arrays.

TEST_CASE("prepare_inputs: no drafts -> cu_num_logits == arange, logits_indices unchanged") {
  InputBatch batch = make_batch();
  add_direct(batch, "A", {100, 101, 102, 103}, {}, {3}, /*num_computed=*/0);
  add_direct(batch, "B", {200, 201, 202, 203, 204}, {205}, {7}, /*num_computed=*/5);

  SchedulerOutput so = make_output({{"A", 4}, {"B", 1}});
  update_states(batch, so);
  StepInputs step = prepare_inputs(batch, so);

  // The scheduler populated NO drafts (the production default).
  CHECK(so.scheduled_spec_decode_tokens.empty());
  CHECK(step.num_draft_tokens == 0);
  CHECK(step.num_draft_tokens_per_req.empty());
  // cu_num_logits = arange(num_reqs + 1) (model_runner.py:872-875).
  CHECK(step.cu_num_logits == std::vector<int32_t>{0, 1, 2});
  // logits_indices == query_start_loc[1:] - 1 — the pre-I3 array, unchanged.
  CHECK(step.logits_indices == std::vector<int32_t>{3, 4});
  REQUIRE(step.logits_indices.size() == static_cast<size_t>(batch.num_reqs()));
  for (int i = 0; i < batch.num_reqs(); ++i) {
    CHECK(step.logits_indices[static_cast<size_t>(i)] ==
          step.query_start_loc[static_cast<size_t>(i) + 1] - 1);
  }
}

TEST_CASE("prepare_inputs: drafts expand logits to 1+k_i rows per request") {
  InputBatch batch = make_batch();
  // "A" carries 2 drafts => scheduled for 1 + 2 = 3 tokens.
  add_direct(batch, "A", {100, 101, 102}, {103, 104, 105}, {3}, /*num_computed=*/3);
  // "B" carries 1 draft => scheduled for 2 tokens.
  add_direct(batch, "B", {200, 201}, {202, 203}, {7}, /*num_computed=*/2);

  SchedulerOutput so = make_output({{"A", 3}, {"B", 2}});
  so.scheduled_spec_decode_tokens["A"] = {104, 105};
  so.scheduled_spec_decode_tokens["B"] = {203};
  update_states(batch, so);
  StepInputs step = prepare_inputs(batch, so);

  CHECK(step.num_draft_tokens == 3);
  CHECK(step.num_draft_tokens_per_req == std::vector<int32_t>{2, 1});
  // num_logits = k_i + 1 => [3, 2]; cu = [0, 3, 5].
  CHECK(step.cu_num_logits == std::vector<int32_t>{0, 3, 5});
  // query_start_loc = [0, 3, 5]; per request logits_start = query_end -
  // num_logits, then num_logits consecutive rows.
  CHECK(step.query_start_loc == std::vector<int32_t>{0, 3, 5});
  CHECK(step.logits_indices == std::vector<int32_t>{0, 1, 2, 3, 4});
  // Every request's LAST expanded row is still its last scheduled token (the
  // bonus position) — the non-spec logits_indices entry.
  for (int i = 0; i < batch.num_reqs(); ++i) {
    const int last = step.cu_num_logits[static_cast<size_t>(i) + 1] - 1;
    CHECK(step.logits_indices[static_cast<size_t>(last)] ==
          step.query_start_loc[static_cast<size_t>(i) + 1] - 1);
  }
}

TEST_CASE("prepare_inputs: a request without drafts keeps 1 logit row in a mixed batch") {
  InputBatch batch = make_batch();
  add_direct(batch, "A", {100, 101}, {102, 103}, {3}, /*num_computed=*/2);
  add_direct(batch, "B", {200, 201, 202}, {203}, {7}, /*num_computed=*/3);

  // Only "A" carries a draft; "B" is a plain decode row in the same step.
  SchedulerOutput so = make_output({{"A", 2}, {"B", 1}});
  so.scheduled_spec_decode_tokens["A"] = {103};
  update_states(batch, so);
  StepInputs step = prepare_inputs(batch, so);

  CHECK(step.num_draft_tokens == 1);
  CHECK(step.num_draft_tokens_per_req == std::vector<int32_t>{1, 0});
  CHECK(step.cu_num_logits == std::vector<int32_t>{0, 2, 3});
  CHECK(step.query_start_loc == std::vector<int32_t>{0, 2, 3});
  // A: rows 0,1 (draft position + bonus). B: row 2 (its single last token).
  CHECK(step.logits_indices == std::vector<int32_t>{0, 1, 2});
}
