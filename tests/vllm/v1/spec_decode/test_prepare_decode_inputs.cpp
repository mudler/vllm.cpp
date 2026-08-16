// SPEC-MTP-K-GT-1 (#81) — the draft-DECODE input preparation.
//
// Ported from vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py @
// 5559679229bc961848b121ccdeaa8fa5d79bec98:
//   * `_prepare_decode_inputs_kernel` :597-645 + `prepare_decode_inputs` :648-671
//   * `_update_draft_inputs_kernel`   :674-738 + `update_draft_inputs`   :741-771
//
// Upstream has no unit test for either kernel: they are covered only through the
// GPU e2e speculative suite (tests/v1/e2e/spec_decode/), which needs a device and
// two checkpoints. These cases pin the arithmetic that e2e depends on, including
// the two `max_model_len` clamps, which an e2e run only reaches at the very tail
// of a sequence and therefore almost never exercises.
//
// The sibling for the k=1 half is test_prepare_prefill_inputs.cpp.
#include "vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_decode_inputs.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using vllm::v1::draft_decode_slot_mapping;
using vllm::v1::prepare_decode_inputs;
using vllm::v1::SpecDecodeInputs;
using vllm::v1::update_draft_inputs;

TEST_CASE("prepare_decode_inputs: the step-1 entry state, per request") {
  // Two requests. r0 had one draft rejected, r1 had none.
  const std::vector<int32_t> draft0 = {41, 42};
  const std::vector<int32_t> target_seq_lens = {20, 30};
  const std::vector<int32_t> num_rejected = {1, 0};
  const std::vector<int32_t> prefill_positions = {17, 29};

  const SpecDecodeInputs in = prepare_decode_inputs(
      draft0, target_seq_lens, num_rejected, prefill_positions,
      /*max_model_len=*/4096);

  // :630-631 — the step-0 draft token is what the first decode step drafts FROM.
  CHECK(in.input_ids == std::vector<int32_t>{41, 42});
  // :637-639 — one past the prefill's sampled row.
  CHECK(in.positions == std::vector<int32_t>{18, 30});
  // :641-645 — the target length less the rejected tail, plus this step's token.
  CHECK(in.seq_lens == std::vector<int32_t>{20, 31});
  // :617-621 — exactly one query token per request.
  CHECK(in.query_start_loc == std::vector<int32_t>{0, 1, 2});
  CHECK(in.num_reqs() == 2);
}

TEST_CASE("prepare_decode_inputs: both max_model_len clamps fire") {
  // :635-636 / :638 — the position clamp is to max_model_len - 1 (it indexes),
  // and :644 the seq_len clamp is to max_model_len (it counts). The two bounds
  // differ by one, and getting that wrong is an out-of-range block lookup at the
  // tail of a sequence, which is exactly where an e2e run rarely goes.
  const SpecDecodeInputs in = prepare_decode_inputs(
      /*draft_tokens_step0=*/{7}, /*target_seq_lens=*/{32},
      /*num_rejected=*/{0}, /*prefill_positions=*/{31}, /*max_model_len=*/32);
  CHECK(in.positions == std::vector<int32_t>{31});  // NOT 32
  CHECK(in.seq_lens == std::vector<int32_t>{32});   // NOT 33
}

TEST_CASE("update_draft_inputs: records the step, then feeds it forward") {
  SpecDecodeInputs in = prepare_decode_inputs(
      /*draft_tokens_step0=*/{41, 42}, /*target_seq_lens=*/{20, 30},
      /*num_rejected=*/{1, 0}, /*prefill_positions=*/{17, 29},
      /*max_model_len=*/4096);
  const int k = 3;
  std::vector<int32_t> drafts(2 * static_cast<size_t>(k), -1);

  // Step 0 records the prefill's token and advances, because two steps remain.
  CHECK(update_draft_inputs({41, 42}, /*step=*/0, k, drafts, in,
                            /*max_model_len=*/4096));
  CHECK(drafts[0] == 41);
  CHECK(drafts[3] == 42);
  CHECK(in.input_ids == std::vector<int32_t>{41, 42});
  CHECK(in.positions == std::vector<int32_t>{19, 31});  // advanced once
  CHECK(in.seq_lens == std::vector<int32_t>{21, 32});

  // Step 1 likewise, and the drafted token becomes the next input.
  CHECK(update_draft_inputs({51, 52}, /*step=*/1, k, drafts, in,
                            /*max_model_len=*/4096));
  CHECK(drafts[1] == 51);
  CHECK(drafts[4] == 52);
  CHECK(in.input_ids == std::vector<int32_t>{51, 52});
  CHECK(in.positions == std::vector<int32_t>{20, 32});
  CHECK(in.seq_lens == std::vector<int32_t>{22, 33});

  // :704-706 — the FINAL step records its token and feeds NOTHING forward. The
  // returned false is the loop bound, so a wrong answer here either drops the
  // deepest draft or runs one forward too many.
  CHECK_FALSE(update_draft_inputs({61, 62}, /*step=*/2, k, drafts, in,
                                  /*max_model_len=*/4096));
  CHECK(drafts == std::vector<int32_t>{41, 51, 61, 42, 52, 62});
  CHECK(in.input_ids == std::vector<int32_t>{51, 52});   // untouched
  CHECK(in.positions == std::vector<int32_t>{20, 32});   // untouched
  CHECK(in.seq_lens == std::vector<int32_t>{22, 33});    // untouched
}

TEST_CASE("update_draft_inputs: k=1 records the only draft and stops at once") {
  // The degenerate depth. The whole loop must collapse to "record and return
  // false", which is what makes k=1 byte-identical to the pre-depth path.
  SpecDecodeInputs in = prepare_decode_inputs({9}, {12}, {0}, {11}, 4096);
  std::vector<int32_t> drafts(1, -1);
  CHECK_FALSE(update_draft_inputs({9}, /*step=*/0, /*k=*/1, drafts, in, 4096));
  CHECK(drafts == std::vector<int32_t>{9});
}

TEST_CASE("update_draft_inputs: the advance clamps at max_model_len too") {
  // :732-734 / :736-737 — the same two bounds as prepare_decode_inputs, applied
  // between steps rather than at entry.
  SpecDecodeInputs in = prepare_decode_inputs({7}, {32}, {0}, {31},
                                              /*max_model_len=*/32);
  std::vector<int32_t> drafts(3, -1);
  CHECK(update_draft_inputs({7}, /*step=*/0, /*k=*/3, drafts, in,
                            /*max_model_len=*/32));
  CHECK(in.positions == std::vector<int32_t>{31});
  CHECK(in.seq_lens == std::vector<int32_t>{32});
}

TEST_CASE("update_draft_inputs: a step outside [0, k) is refused") {
  SpecDecodeInputs in = prepare_decode_inputs({1}, {4}, {0}, {3}, 4096);
  std::vector<int32_t> drafts(2, -1);
  CHECK_THROWS(update_draft_inputs({1}, /*step=*/2, /*k=*/2, drafts, in, 4096));
  CHECK_THROWS(update_draft_inputs({1}, /*step=*/-1, /*k=*/2, drafts, in, 4096));
}

TEST_CASE("draft_decode_slot_mapping: block_id * block_size + offset") {
  // The world-size-1 reduction of BlockTable::compute_slot_mapping, which is
  // what upstream's compute_slot_mappings (speculator.py:392-397) computes for
  // the draft's one-token-per-request decode batch.
  const std::vector<int32_t> block_table = {
      5, 6, 7,   // request 0's blocks
      9, 1, 2,   // request 1's blocks
  };
  // r0: position 3 -> block index 0 (block id 5), offset 3.
  // r1: position 8 -> block index 2 (block id 2), offset 0. Picking the block by
  // position / block_size rather than by request order is the whole point: r1's
  // third block is a different physical page from r0's first.
  const std::vector<int32_t> positions = {3, 8};
  const std::vector<int64_t> slots = draft_decode_slot_mapping(
      block_table, /*block_table_num_cols=*/3, positions, /*block_size=*/4);
  CHECK(slots == std::vector<int64_t>{5 * 4 + 3, 2 * 4 + 0});
}

TEST_CASE("draft_decode_slot_mapping: a position past the block table throws") {
  // The draft advances positions the TARGET has not reached, so it is the one
  // caller that can run off the end of a request's allocated blocks. Refusing
  // loudly beats reading a neighbouring request's KV.
  CHECK_THROWS(draft_decode_slot_mapping(/*block_table=*/{5, 6},
                                         /*block_table_num_cols=*/2,
                                         /*positions=*/{9},
                                         /*block_size=*/4));
}
