// Tests for the persistent InputBatch + CachedRequestState (M1.5 Task 2) — the
// num_reqs-major per-slot arrays the model runner keeps alive across steps.
//
// Ported from vllm/v1/worker/gpu_input_batch.py @ e24d1b24, with the behavioral
// oracle taken from tests/v1/worker/test_gpu_input_batch.py (the
// add_request / _remove_requests / condense / _compare_objs pattern:
// test_sampling_metadata_in_input_batch adds a batch, removes a subset, then
// condense()s and checks the dense state). Those upstream tests drive a random
// batch through SamplingMetadata; here — SamplingMetadata is not yet landed
// (M1.7) — we assert the same underlying per-slot array + req_id_to_index +
// block-table densification directly, plus the MRV2-contract from_new_request
// seed (prefill_token_ids). See the header for the deferred slot state.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/worker/gpu/input_batch.h"

using vllm::SamplingParams;
using vllm::v1::CachedRequestState;
using vllm::v1::InputBatch;
using vllm::v1::NewRequestData;

namespace {

// Build a single-group CachedRequestState directly (the V1 add-request input).
CachedRequestState make_req(const std::string& req_id,
                            std::vector<int32_t> prompt,
                            std::vector<int32_t> output,
                            std::vector<int> block_ids,
                            SamplingParams sp = SamplingParams{}) {
  CachedRequestState state;
  state.req_id = req_id;
  state.prompt_token_ids = std::move(prompt);
  state.output_token_ids = std::move(output);
  state.sampling_params = sp;
  state.block_ids = {std::move(block_ids)};
  state.num_computed_tokens = state.num_prompt_tokens;  // overwritten below
  state.finalize();
  return state;
}

InputBatch make_batch(int max_num_reqs = 8, int max_model_len = 64) {
  return InputBatch(/*max_num_reqs=*/max_num_reqs,
                    /*max_model_len=*/max_model_len,
                    /*max_num_batched_tokens=*/max_model_len,
                    /*vocab_size=*/1024, /*block_sizes=*/{16},
                    /*kernel_block_sizes=*/{16});
}

}  // namespace

TEST_CASE("add_request fills a slot: token ids, counts, block rows, sampling") {
  InputBatch batch = make_batch();

  CachedRequestState req = make_req("r0", {10, 11, 12}, {20, 21}, {3, 7});
  req.num_computed_tokens = 3;
  const int idx = batch.add_request(req);

  CHECK(idx == 0);
  CHECK(batch.num_reqs() == 1);
  CHECK(batch.req_id_to_index.at("r0") == 0);
  CHECK(batch.req_ids[0].has_value());
  CHECK(*batch.req_ids[0] == "r0");

  // token_ids_cpu seeded prompt then output (== prefill_token_ids).
  CHECK(batch.token_id(0, 0) == 10);
  CHECK(batch.token_id(0, 1) == 11);
  CHECK(batch.token_id(0, 2) == 12);
  CHECK(batch.token_id(0, 3) == 20);
  CHECK(batch.token_id(0, 4) == 21);

  CHECK(batch.num_prompt_tokens[0] == 3);
  CHECK(batch.num_tokens_no_spec[0] == 5);  // 3 prompt + 2 output
  CHECK(batch.num_computed_tokens_cpu[0] == 3);

  // Block-table rows added for group 0.
  CHECK(batch.block_table[0].num_blocks_per_row[0] == 2);
  CHECK(batch.block_table[0].cpu_block_id(0, 0) == 3);
  CHECK(batch.block_table[0].cpu_block_id(0, 1) == 7);

  // Default sampling params (temperature 1.0) => random, no top_p/top_k.
  CHECK(batch.all_random());
  CHECK_FALSE(batch.all_greedy());
  CHECK(batch.no_top_p());
  CHECK(batch.no_top_k());
  CHECK(batch.no_penalties());
}

TEST_CASE("add_request assigns sequential slots and tracks num_reqs") {
  InputBatch batch = make_batch();
  CHECK(batch.add_request(make_req("a", {1}, {}, {0})) == 0);
  CHECK(batch.add_request(make_req("b", {2}, {}, {1})) == 1);
  CHECK(batch.add_request(make_req("c", {3}, {}, {2})) == 2);
  CHECK(batch.num_reqs() == 3);
}

TEST_CASE("sampling predicates reflect per-request params") {
  InputBatch batch = make_batch();

  SamplingParams greedy;
  greedy.temperature = 0.0;  // Type() => greedy
  SamplingParams topk;
  topk.top_k = 5;
  SamplingParams topp;
  topp.top_p = 0.9;
  SamplingParams pen;
  pen.presence_penalty = 0.5;

  batch.add_request(make_req("g", {1}, {}, {0}, greedy));
  CHECK(batch.all_greedy());
  CHECK_FALSE(batch.all_random());
  CHECK(batch.temperature_cpu[0] == doctest::Approx(0.0));

  batch.add_request(make_req("k", {2}, {}, {1}, topk));
  CHECK_FALSE(batch.no_top_k());
  CHECK(batch.top_k_cpu[1] == 5);
  // Mixed greedy + random now.
  CHECK_FALSE(batch.all_greedy());
  CHECK_FALSE(batch.all_random());

  batch.add_request(make_req("p", {3}, {}, {2}, topp));
  CHECK_FALSE(batch.no_top_p());
  CHECK(batch.top_p_cpu[2] == doctest::Approx(0.9));

  batch.add_request(make_req("n", {4}, {}, {3}, pen));
  CHECK_FALSE(batch.no_penalties());
}

TEST_CASE("remove_request frees the slot and returns its index") {
  InputBatch batch = make_batch();
  batch.add_request(make_req("a", {1}, {}, {5}));
  batch.add_request(make_req("b", {2}, {}, {6}));

  const std::optional<int> removed = batch.remove_request("a");
  REQUIRE(removed.has_value());
  CHECK(*removed == 0);
  CHECK(batch.num_reqs() == 1);
  CHECK(batch.req_id_to_index.count("a") == 0);
  CHECK_FALSE(batch.req_ids[0].has_value());
  // Block row cleared.
  CHECK(batch.block_table[0].num_blocks_per_row[0] == 0);

  // Removing an unknown req returns nullopt.
  CHECK_FALSE(batch.remove_request("zzz").has_value());
}

TEST_CASE("condense: remove the middle request then densify [0, num_reqs)") {
  InputBatch batch = make_batch();
  // Three requests with distinct blocks + token ids.
  CachedRequestState r0 = make_req("r0", {100}, {}, {10});
  r0.num_computed_tokens = 1;
  CachedRequestState r1 = make_req("r1", {200, 201}, {}, {20, 21});
  r1.num_computed_tokens = 2;
  CachedRequestState r2 = make_req("r2", {300}, {}, {30});
  r2.num_computed_tokens = 1;
  batch.add_request(r0);
  batch.add_request(r1);
  batch.add_request(r2);
  REQUIRE(batch.num_reqs() == 3);

  // Remove the middle request (slot 1).
  const std::optional<int> removed = batch.remove_request("r1");
  REQUIRE(removed.has_value());
  CHECK(*removed == 1);
  CHECK(batch.num_reqs() == 2);

  batch.condense();

  // r2 (was slot 2) slides down into the freed slot 1; [0, 2) is now dense.
  CHECK(batch.num_reqs() == 2);
  CHECK(static_cast<int>(batch.req_ids.size()) == 2);
  CHECK(batch.req_id_to_index.at("r0") == 0);
  CHECK(batch.req_id_to_index.at("r2") == 1);
  CHECK(batch.req_id_to_index.count("r1") == 0);
  REQUIRE(batch.req_ids[1].has_value());
  CHECK(*batch.req_ids[1] == "r2");

  // r2's per-slot arrays moved into slot 1.
  CHECK(batch.token_id(1, 0) == 300);
  CHECK(batch.num_prompt_tokens[1] == 1);
  CHECK(batch.num_tokens_no_spec[1] == 1);
  CHECK(batch.num_computed_tokens_cpu[1] == 1);

  // r2's block-table row moved into slot 1.
  CHECK(batch.block_table[0].num_blocks_per_row[1] == 1);
  CHECK(batch.block_table[0].cpu_block_id(1, 0) == 30);

  // Slot 0 (r0) is untouched.
  CHECK(*batch.req_ids[0] == "r0");
  CHECK(batch.token_id(0, 0) == 100);
  CHECK(batch.block_table[0].cpu_block_id(0, 0) == 10);
}

// ─── async-scheduling per-slot state (ENG-ASYNC-SCHED W3 runner leaf) ────────
TEST_CASE("add_request seeds prefill_len; last_sampled only for resumed reqs") {
  InputBatch batch = make_batch();

  // Fresh prefill (num_computed == 0): prefill_len == num_tokens (prompt+output),
  // last_sampled stays 0 (combine never reads it during prefill).
  CachedRequestState fresh = make_req("fresh", {10, 11, 12}, {}, {3});
  fresh.num_computed_tokens = 0;
  const int i0 = batch.add_request(fresh);
  CHECK(batch.prefill_len[static_cast<size_t>(i0)] == 3);        // 3 prompt
  CHECK(batch.last_sampled_tokens[static_cast<size_t>(i0)] == 0);

  // Resumed / PD-disagg (0 < num_computed <= prefill_len): last_sampled seeded
  // with the token at num_computed-1 so the first decode reads the right id.
  CachedRequestState resumed = make_req("resumed", {20, 21, 22}, {23, 24}, {7});
  resumed.num_computed_tokens = 4;  // prefill_len = 5 (prompt3 + output2)
  const int i1 = batch.add_request(resumed);
  CHECK(batch.prefill_len[static_cast<size_t>(i1)] == 5);
  // token at index num_computed-1 == 3 -> the seed row is [20,21,22,23,24] -> 23.
  CHECK(batch.last_sampled_tokens[static_cast<size_t>(i1)] == 23);
}

TEST_CASE("condense moves last_sampled_tokens + prefill_len with the request") {
  InputBatch batch = make_batch();
  CachedRequestState r0 = make_req("r0", {100}, {}, {10});
  r0.num_computed_tokens = 1;
  CachedRequestState r1 = make_req("r1", {200, 201}, {}, {20});
  r1.num_computed_tokens = 2;
  // r2 resumed so its last_sampled is a distinctive non-zero seed.
  CachedRequestState r2 = make_req("r2", {300, 301}, {302}, {30});
  r2.num_computed_tokens = 3;  // prefill_len 3 -> seed token at idx 2 == 302
  batch.add_request(r0);
  batch.add_request(r1);
  batch.add_request(r2);
  REQUIRE(batch.last_sampled_tokens[2] == 302);
  REQUIRE(batch.prefill_len[2] == 3);

  batch.remove_request("r1");
  batch.condense();  // r2 (slot 2) slides into freed slot 1

  CHECK(batch.req_id_to_index.at("r2") == 1);
  CHECK(batch.last_sampled_tokens[1] == 302);  // moved with the request
  CHECK(batch.prefill_len[1] == 3);
  // Slot 0 untouched.
  CHECK(batch.prefill_len[0] == 1);
}

TEST_CASE("swap_states swaps last_sampled_tokens + prefill_len") {
  InputBatch batch = make_batch();
  CachedRequestState a = make_req("a", {1, 2}, {3}, {10});
  a.num_computed_tokens = 3;  // prefill_len 3, seed idx2 == 3
  CachedRequestState b = make_req("b", {5}, {}, {20});
  b.num_computed_tokens = 0;  // prefill_len 1, last_sampled 0
  batch.add_request(a);  // slot 0
  batch.add_request(b);  // slot 1
  REQUIRE(batch.last_sampled_tokens[0] == 3);
  REQUIRE(batch.prefill_len[0] == 3);

  batch.swap_states(0, 1);

  // The reorder primitive must carry the async state with the row.
  CHECK(batch.req_id_to_index.at("a") == 1);
  CHECK(batch.req_id_to_index.at("b") == 0);
  CHECK(batch.last_sampled_tokens[1] == 3);
  CHECK(batch.prefill_len[1] == 3);
  CHECK(batch.last_sampled_tokens[0] == 0);
  CHECK(batch.prefill_len[0] == 1);
}

TEST_CASE("condense is a no-op when only the last request was removed") {
  InputBatch batch = make_batch();
  batch.add_request(make_req("a", {1}, {}, {0}));
  batch.add_request(make_req("b", {2}, {}, {1}));

  // Remove the LAST slot: no active request lives above the hole, so condense
  // only trims (no move); req_id_to_index for the survivor is unchanged.
  batch.remove_request("b");
  batch.condense();
  CHECK(batch.num_reqs() == 1);
  CHECK(batch.req_id_to_index.at("a") == 0);
  CHECK(static_cast<int>(batch.req_ids.size()) == 1);
}

TEST_CASE("add after remove fills the freed hole before appending") {
  InputBatch batch = make_batch();
  batch.add_request(make_req("a", {1}, {}, {0}));
  batch.add_request(make_req("b", {2}, {}, {1}));
  batch.add_request(make_req("c", {3}, {}, {2}));

  batch.remove_request("b");  // frees slot 1
  // Next add fills slot 1 (the hole), not slot 3.
  const int idx = batch.add_request(make_req("d", {4}, {}, {9}));
  CHECK(idx == 1);
  CHECK(batch.req_id_to_index.at("d") == 1);
  CHECK(batch.block_table[0].cpu_block_id(1, 0) == 9);
  // No stray hole remains for condense to fill.
  batch.condense();
  CHECK(batch.num_reqs() == 3);
}

TEST_CASE("CachedRequestState.from_new_request seeds output from prefill (MRV2)") {
  // MRV2 contract: prefill_token_ids == all_token_ids (prompt + output). The
  // state's output_token_ids is its tail beyond the prompt.
  NewRequestData nr;
  nr.req_id = "r";
  nr.prompt_token_ids = std::vector<int32_t>{1, 2, 3};
  nr.sampling_params = SamplingParams{};
  nr.block_ids = {{7}};
  nr.num_computed_tokens = 3;
  nr.prefill_token_ids = std::vector<int32_t>{1, 2, 3, 4, 5};  // prompt + 2 out

  CachedRequestState state = CachedRequestState::from_new_request(nr);
  CHECK(state.req_id == "r");
  CHECK(state.num_prompt_tokens == 3);
  CHECK(state.output_token_ids == std::vector<int32_t>{4, 5});
  CHECK(state.num_tokens() == 5);
  CHECK(state.num_computed_tokens == 3);

  // Feeding it through add_request reproduces the prefill_token_ids seed.
  InputBatch batch = make_batch();
  batch.add_request(state);
  for (int i = 0; i < 5; ++i) {
    CHECK(batch.token_id(0, i) == i + 1);
  }
  CHECK(batch.block_table[0].cpu_block_id(0, 0) == 7);
}

TEST_CASE("CachedRequestState.get_token_id and num_tokens") {
  CachedRequestState state;
  state.prompt_token_ids = {10, 11};
  state.output_token_ids = {12};
  state.finalize();
  CHECK(state.num_prompt_tokens == 2);
  CHECK(state.num_tokens() == 3);
  CHECK(state.get_token_id(0) == 10);
  CHECK(state.get_token_id(1) == 11);
  CHECK(state.get_token_id(2) == 12);
  CHECK(state.get_token_id(3) == -1);  // past the end
}
