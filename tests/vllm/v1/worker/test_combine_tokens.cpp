// ENG-ASYNC-SCHED W3 runner leaf — combine_sampled_and_draft_tokens, the
// async-scheduling device-input path that rebuilds each decode row's input token
// id from the GPU-resident-analog last_sampled_tokens instead of the host
// token_ids_cpu read (so step N+1 need not wait on step N's sampled token to
// cross to the host — the ~3.25 ms/step idle).
//
// Ported from vllm/v1/worker/gpu/input_batch.py::combine_sampled_and_draft_tokens
// + _combine_sampled_and_draft_tokens_kernel @ e24d1b24 (T0 non-spec subset).
// The upstream kernel runs on GPU; here — host arrays, no CUDA — the oracle is
// derived directly from the kernel algorithm (per batch row: logits_indices =
// query_end - num_logits (+block); if seq_len > prefill_len write
// last_sampled_tokens[idx_mapping[b]] at query_end - num_logits, else leave the
// prompt token).
//
// RED→GREEN: each case first records what prepare_inputs' host read produced (the
// STALE value the async path must not use — either the pre-write 0 or the prior
// prompt token) and then asserts combine splices the fresh last_sampled id for
// decode rows while leaving prefill / chunked-prefill rows untouched. Without
// combine (the sync no-op) the decode rows keep the stale value and every
// "== fresh id" assertion fails — that is the RED.
#include <doctest/doctest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::v1::combine_sampled_and_draft_tokens;

namespace {
// Identity idx_mapping (our persistent batch is condensed dense).
std::vector<int32_t> Identity(int num_reqs) {
  std::vector<int32_t> m(static_cast<size_t>(num_reqs));
  std::iota(m.begin(), m.end(), 0);
  return m;
}
}  // namespace

// ─── pure decode ────────────────────────────────────────────────────────────
TEST_CASE("combine: pure decode overwrites each row's input id with last_sampled") {
  // Two decode requests, one scheduled token each. query_start_loc = [0,1,2];
  // input_token_ids seeded with the STALE host values (say a re-read of the
  // previous input), prefill_len below seq_len so both are decodes.
  std::vector<int32_t> input_ids = {111, 222};  // stale host values
  const std::vector<int32_t> last_sampled = {700, 800};  // fresh sampler output
  const std::vector<int32_t> qsl = {0, 1, 2};
  const std::vector<int32_t> seq_lens = {6, 9};    // > prefill_len -> decode
  const std::vector<int32_t> prefill_len = {5, 8};

  // RED baseline: before combine the input ids are the stale host values.
  REQUIRE(input_ids[0] == 111);
  REQUIRE(input_ids[1] == 222);

  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, Identity(2), last_sampled, qsl, seq_lens, prefill_len);

  // GREEN: each decode row's input id is now the fresh sampled token.
  CHECK(input_ids == std::vector<int32_t>{700, 800});
  // logits_indices = query_start_loc[1:] - 1.
  CHECK(li == std::vector<int32_t>{0, 1});
}

// ─── reads last_sampled, NOT the host buffer ────────────────────────────────
TEST_CASE("combine: decode input id comes from last_sampled, not token_ids_cpu") {
  // Simulate the async D2H-skip: the host input ids at the decode positions are
  // STALE (zeroed — token_ids_cpu was never written this step). combine must
  // still produce the correct id purely from last_sampled.
  std::vector<int32_t> input_ids = {0, 0, 0};  // stale/zero host reads
  const std::vector<int32_t> last_sampled = {41, 42, 43};
  const std::vector<int32_t> qsl = {0, 1, 2, 3};
  const std::vector<int32_t> seq_lens = {2, 2, 2};
  const std::vector<int32_t> prefill_len = {1, 1, 1};  // all past prefill

  combine_sampled_and_draft_tokens(input_ids, Identity(3), last_sampled, qsl,
                                   seq_lens, prefill_len);

  CHECK(input_ids == std::vector<int32_t>{41, 42, 43});
}

// ─── prefill chunk: untouched ───────────────────────────────────────────────
TEST_CASE("combine: prefill / chunked-prefill rows keep their prompt token") {
  // Req in mid-prefill: seq_len (num_computed+num_scheduled) < prefill_len, so
  // the last scheduled token is a PROMPT token, not a sampled one — leave it.
  // Multi-token prefill chunk of 4 tokens (query_start_loc jumps by 4).
  std::vector<int32_t> input_ids = {100, 101, 102, 103};  // prompt chunk
  const std::vector<int32_t> last_sampled = {999};        // must NOT be spliced
  const std::vector<int32_t> qsl = {0, 4};
  const std::vector<int32_t> seq_lens = {4};       // < prefill_len (10) -> prefill
  const std::vector<int32_t> prefill_len = {10};

  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, Identity(1), last_sampled, qsl, seq_lens, prefill_len);

  // Unchanged prompt chunk.
  CHECK(input_ids == std::vector<int32_t>{100, 101, 102, 103});
  // logits_indices still the last scheduled position (query_end - 1 == 3).
  CHECK(li == std::vector<int32_t>{3});
}

// ─── chunked-prefill transition boundary ────────────────────────────────────
TEST_CASE("combine: prefill-completing chunk (seq_len == prefill_len) is prefill; "
          "first decode (seq_len == prefill_len+1) splices") {
  const std::vector<int32_t> last_sampled = {555};

  SUBCASE("seq_len == prefill_len: the chunk that exactly finishes prefill") {
    // The last prompt token is the input; the sampled first output token belongs
    // to the NEXT step. seq_len <= prefill_len -> no splice.
    std::vector<int32_t> input_ids = {70, 71, 72};  // last is the final prompt id
    const std::vector<int32_t> qsl = {0, 3};
    const std::vector<int32_t> seq_lens = {8};
    const std::vector<int32_t> prefill_len = {8};  // seq_len == prefill_len

    combine_sampled_and_draft_tokens(input_ids, Identity(1), last_sampled, qsl,
                                     seq_lens, prefill_len);
    CHECK(input_ids == std::vector<int32_t>{70, 71, 72});  // untouched
  }
  SUBCASE("seq_len == prefill_len + 1: the first true decode step") {
    std::vector<int32_t> input_ids = {0};  // stale host read at the decode pos
    const std::vector<int32_t> qsl = {0, 1};
    const std::vector<int32_t> seq_lens = {9};
    const std::vector<int32_t> prefill_len = {8};  // seq_len > prefill_len

    combine_sampled_and_draft_tokens(input_ids, Identity(1), last_sampled, qsl,
                                     seq_lens, prefill_len);
    CHECK(input_ids == std::vector<int32_t>{555});  // spliced
  }
}

// ─── mixed batch (decode-first order) ───────────────────────────────────────
TEST_CASE("combine: mixed decode+prefill batch splices only the decode rows") {
  // Decode-first order (the runner reorders before prepare_inputs): row 0 decode
  // (1 token), row 1 a 3-token prefill chunk. query_start_loc = [0,1,4].
  std::vector<int32_t> input_ids = {0, 200, 201, 202};  // [decode-stale, prompt x3]
  const std::vector<int32_t> last_sampled = {321, 999};  // row1's must NOT splice
  const std::vector<int32_t> qsl = {0, 1, 4};
  const std::vector<int32_t> seq_lens = {5, 3};   // row0 decode, row1 prefill
  const std::vector<int32_t> prefill_len = {4, 9};

  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, Identity(2), last_sampled, qsl, seq_lens, prefill_len);

  // Only the decode row (index 0) is rewritten; the prefill chunk is untouched.
  CHECK(input_ids == std::vector<int32_t>{321, 200, 201, 202});
  // logits_indices = [query_end0-1, query_end1-1] = [0, 3].
  CHECK(li == std::vector<int32_t>{0, 3});
}

// ─── idx_mapping indirection (abort / finish reorders req_states) ───────────
TEST_CASE("combine: idx_mapping selects the correct req_state after churn") {
  // After an abort/finish + condense, the dense batch row need not equal the
  // req_state slot. idx_mapping[b] -> req_state carries last_sampled / prefill_len
  // per req_state while query_start_loc / seq_lens are per batch row. Here batch
  // row 0 maps to req_state 2 and batch row 1 to req_state 0.
  std::vector<int32_t> input_ids = {0, 0};
  // last_sampled / prefill_len indexed by req_state (size >= max slot + 1).
  const std::vector<int32_t> last_sampled = {10, 11, 12};  // [rs0, rs1, rs2]
  const std::vector<int32_t> prefill_len = {5, 5, 5};
  const std::vector<int32_t> idx_mapping = {2, 0};  // batch->req_state
  const std::vector<int32_t> qsl = {0, 1, 2};
  const std::vector<int32_t> seq_lens = {6, 6};  // both decodes

  combine_sampled_and_draft_tokens(input_ids, idx_mapping, last_sampled, qsl,
                                   seq_lens, prefill_len);

  // Row 0 got req_state 2's token (12); row 1 got req_state 0's token (10).
  CHECK(input_ids == std::vector<int32_t>{12, 10});
}

// ─── draft-only step (num_new_sampled_tokens == 0) ──────────────────────────
TEST_CASE("combine: num_new_sampled_tokens==0 writes no sampled id, empty logits") {
  // The bonus-token-less path (excl. accepted drafts). At T0 there are no draft
  // tokens either, so this splices nothing and emits no logit indices.
  std::vector<int32_t> input_ids = {77};
  const std::vector<int32_t> last_sampled = {999};
  const std::vector<int32_t> qsl = {0, 1};
  const std::vector<int32_t> seq_lens = {6};
  const std::vector<int32_t> prefill_len = {5};

  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, Identity(1), last_sampled, qsl, seq_lens, prefill_len,
      /*num_new_sampled_tokens=*/0);

  CHECK(input_ids == std::vector<int32_t>{77});  // untouched
  CHECK(li.empty());                             // no logits this step
}
