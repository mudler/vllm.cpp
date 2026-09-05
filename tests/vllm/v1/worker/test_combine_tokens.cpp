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

// cu_num_logits on the NON-SPECULATIVE path: arange(num_reqs + 1), i.e. one
// logit per request (gpu/model_runner.py:872-875, and prepare_inputs.cpp's own
// no-draft branch). num_logits per row is then 1 == num_new_sampled_tokens, so
// every case below that predates SPEC-DFLASH2 A2-1 keeps its exact meaning.
std::vector<int32_t> CuArange(int num_reqs) {
  std::vector<int32_t> c(static_cast<size_t>(num_reqs) + 1);
  std::iota(c.begin(), c.end(), 0);
  return c;
}

// No request has drafts this step: upstream still passes a tensor, we pass an
// empty buffer with a zero stride (nothing indexes it when num_draft_tokens==0).
const std::vector<int32_t> kNoDrafts;
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
      input_ids, Identity(2), last_sampled, qsl, seq_lens, prefill_len, kNoDrafts,
      /*draft_tokens_stride=*/0, CuArange(2));

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
                                   seq_lens, prefill_len, kNoDrafts,
                                   /*draft_tokens_stride=*/0, CuArange(3));

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
      input_ids, Identity(1), last_sampled, qsl, seq_lens, prefill_len, kNoDrafts,
      /*draft_tokens_stride=*/0, CuArange(1));

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
                                     seq_lens, prefill_len, kNoDrafts,
                                     /*draft_tokens_stride=*/0, CuArange(1));
    CHECK(input_ids == std::vector<int32_t>{70, 71, 72});  // untouched
  }
  SUBCASE("seq_len == prefill_len + 1: the first true decode step") {
    std::vector<int32_t> input_ids = {0};  // stale host read at the decode pos
    const std::vector<int32_t> qsl = {0, 1};
    const std::vector<int32_t> seq_lens = {9};
    const std::vector<int32_t> prefill_len = {8};  // seq_len > prefill_len

    combine_sampled_and_draft_tokens(input_ids, Identity(1), last_sampled, qsl,
                                     seq_lens, prefill_len, kNoDrafts,
                                     /*draft_tokens_stride=*/0, CuArange(1));
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
      input_ids, Identity(2), last_sampled, qsl, seq_lens, prefill_len, kNoDrafts,
      /*draft_tokens_stride=*/0, CuArange(2));

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
                                   seq_lens, prefill_len, kNoDrafts,
                                   /*draft_tokens_stride=*/0, CuArange(2));

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

  // num_new_sampled_tokens == 0 means the step asks for NO logits at all, so
  // cu_num_logits is all-zero (num_logits == 0 for the one row).
  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, Identity(1), last_sampled, qsl, seq_lens, prefill_len, kNoDrafts,
      /*draft_tokens_stride=*/0, /*cu_num_logits=*/{0, 0},
      /*num_new_sampled_tokens=*/0);

  CHECK(input_ids == std::vector<int32_t>{77});  // untouched
  CHECK(li.empty());                             // no logits this step
}

// ════════════════════════════════════════════════════════════════════════════
// G2 — SPEC-DFLASH2 A2-1 (#2644): the draft tokens that reach the verify step
// must EQUAL the drafts the proposer produced, asserted on the TOKEN IDS.
//
// This is the gate `.agents/specs/dflash2-async-spec-sampler.md` §Gates names,
// and its `## Owed` says A2-1 owes it as the wave's red-before test. It is
// written on the token ids and NOT on an acceptance ratio, NOT on the emitted
// text and NOT on a golden, for a measured reason: speculative decoding is
// lossless, so a destroyed draft costs acceptance and nothing else. The spec's
// recorded mutation run corrupted the last draft of EVERY verify block in every
// arm and the emitted tokens did not move, the identity assertions passed, and
// the synthetic head's acceptance signal had zero dynamic range (`ns=1 acc=0`
// in the baseline too). Reason A shipped once already as #1366. Only a
// comparison against the proposer's own ids discriminates.
//
// The instrument drives the draft-aware combine DIRECTLY rather than through the
// runner, because the runner cannot reach it: `async_input_combine_` is vetoed
// for every speculative engine at BOTH GPUModelRunner constructors
// (src/vllm/v1/worker/gpu/runner.cpp:480 and :553 — the assignments themselves,
// not the comments above them), and wave A2-5 owns that flip, which the spec
// says is not a judgement call to make ahead of these waves. Driving the
// function is therefore the only honest way to write G2 today; the price is that
// the draft lane lands UNREACHED, which the commit body, the pull request body
// and the spec's `## Owed` all disclose.
//
// SHAPE OF A VERIFY STEP, which is what every case below builds. A request that
// was given k drafts is scheduled 1 + k tokens: its committed token (last step's
// sampled id) followed by the k drafts. So num_logits == 1 + k comes from
// cu_num_logits, logits_start == query_end - (1 + k), and the drafts occupy
// [query_end - k, query_end). Reading num_logits as num_new_sampled_tokens there
// puts the committed token in the LAST DRAFT SLOT — reason A exactly.
namespace {

// The proposer's output as GPUModelRunner::propose_drafts stashes it
// (DraftTokenIds: one row of ids per request), flattened into upstream's 2-D
// [num_req_states, num_speculative_steps] draft_tokens tensor. `pad` is what a
// request shorter than the stride leaves behind: the scatter must never read it,
// because it takes its count from cu_num_logits and not from the stride.
std::vector<int32_t> FlattenDrafts(
    const std::vector<std::vector<int32_t>>& per_req_state, int stride,
    int32_t pad) {
  std::vector<int32_t> flat(per_req_state.size() * static_cast<size_t>(stride),
                            pad);
  for (size_t r = 0; r < per_req_state.size(); ++r) {
    for (size_t j = 0; j < per_req_state[r].size(); ++j) {
      flat[r * static_cast<size_t>(stride) + j] = per_req_state[r][j];
    }
  }
  return flat;
}

// The scheduler's async placeholder (async_scheduler.py:43-45). Seeding the
// draft slots with it rather than with the drafts themselves is what keeps the
// assertion non-vacuous: a combine that writes nothing leaves -1 behind.
constexpr int32_t kPlaceholder = -1;

}  // namespace

// ─── G2.1: one request, k=2 ─────────────────────────────────────────────────
TEST_CASE("G2: the drafts reaching the verify step are the proposer's, by id") {
  // req_state 0, prompt of 5 tokens, already decoded past it. The proposer
  // produced two drafts; the previous step committed token 900.
  const std::vector<int32_t> proposed = {31, 32};
  const int k = static_cast<int>(proposed.size());

  std::vector<int32_t> input_ids(static_cast<size_t>(1 + k), kPlaceholder);
  const std::vector<int32_t> last_sampled = {900};
  const std::vector<int32_t> qsl = {0, 1 + k};
  const std::vector<int32_t> prefill_len = {5};
  // seq_len 8: positions 0-4 are the prompt, 5 is the committed token, 6-7 the
  // drafts. first_logit_seq_pos == 5 == prefill_len, so the committed store is
  // in the generated region and runs.
  const std::vector<int32_t> seq_lens = {8};
  const std::vector<int32_t> cu_num_logits = {0, 1 + k};
  const int stride = k;
  const std::vector<int32_t> draft_tokens =
      FlattenDrafts({proposed}, stride, /*pad=*/7777);

  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, Identity(1), last_sampled, qsl, seq_lens, prefill_len,
      draft_tokens, stride, cu_num_logits);

  // The verify step scores num_logits == 1 + k rows, in query order.
  REQUIRE(li.size() == static_cast<size_t>(1 + k));
  // Row 0 is the committed token; rows 1..k are the drafts, and THOSE are what
  // this gate compares against the proposer, id by id.
  CHECK(input_ids[static_cast<size_t>(li[0])] == 900);
  for (int j = 0; j < k; ++j) {
    CHECK(input_ids[static_cast<size_t>(li[static_cast<size_t>(1 + j)])] ==
          proposed[static_cast<size_t>(j)]);
  }
  CHECK(input_ids == std::vector<int32_t>{900, 31, 32});
  CHECK(li == std::vector<int32_t>{0, 1, 2});
}

// ─── G2.2: reason A, stated as an assertion ─────────────────────────────────
TEST_CASE("G2: the committed token does not land in the last draft slot") {
  // The corruption the spec records: with num_logits read as 1 the committed
  // token is written at query_end - 1, which on a verify step is the last DRAFT
  // position, and the last draft is replaced by the previous step's emit. The
  // emitted tokens never move, so only this comparison sees it.
  const std::vector<int32_t> proposed = {18, 6};
  const int k = 2;
  std::vector<int32_t> input_ids(3, kPlaceholder);
  const std::vector<int32_t> last_sampled = {5};  // the previous step's emit
  const std::vector<int32_t> qsl = {0, 3};
  const std::vector<int32_t> seq_lens = {8};
  const std::vector<int32_t> prefill_len = {5};
  const std::vector<int32_t> cu_num_logits = {0, 3};

  combine_sampled_and_draft_tokens(input_ids, Identity(1), last_sampled, qsl,
                                   seq_lens, prefill_len,
                                   FlattenDrafts({proposed}, k, /*pad=*/7777), k,
                                   cu_num_logits);

  const size_t last_draft_slot = static_cast<size_t>(qsl[1] - 1);
  CHECK(input_ids[last_draft_slot] == proposed[1]);
  CHECK(input_ids[last_draft_slot] != last_sampled[0]);
}

// ─── G2.3: ragged k, idx_mapping indirection, stride > k ────────────────────
TEST_CASE("G2: per-request k comes from cu_num_logits and the draft row from "
          "idx_mapping") {
  // Two batch rows after a churn reorder: row 0 -> req_state 2 (k=3), row 1 ->
  // req_state 0 (k=2). The buffer's stride is the speculator's max draft length
  // (3), so req_state 0's row carries one PAD id the scatter must not read.
  constexpr int32_t kPad = 7777;
  const int stride = 3;
  const std::vector<std::vector<int32_t>> proposed = {
      {70, 71},      // req_state 0, k=2 (one pad slot)
      {0, 0, 0},     // req_state 1, not in this batch
      {80, 81, 82},  // req_state 2, k=3
  };
  const std::vector<int32_t> draft_tokens =
      FlattenDrafts(proposed, stride, kPad);

  std::vector<int32_t> input_ids(7, kPlaceholder);
  const std::vector<int32_t> idx_mapping = {2, 0};
  const std::vector<int32_t> last_sampled = {500, 501, 502};  // per req_state
  const std::vector<int32_t> prefill_len = {4, 4, 6};         // per req_state
  const std::vector<int32_t> qsl = {0, 4, 7};       // 1+3 tokens, then 1+2
  const std::vector<int32_t> seq_lens = {10, 7};    // per BATCH row
  const std::vector<int32_t> cu_num_logits = {0, 4, 7};

  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, idx_mapping, last_sampled, qsl, seq_lens, prefill_len,
      draft_tokens, stride, cu_num_logits);

  // Row 0 got req_state 2's committed id and its three drafts; row 1 got
  // req_state 0's committed id and its two.
  CHECK(input_ids == std::vector<int32_t>{502, 80, 81, 82, 500, 70, 71});
  CHECK(li == std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6});

  // The stride's padding is never read: the count comes from cu_num_logits.
  for (const int32_t id : input_ids) CHECK(id != kPad);

  // The same comparison as G2.1, walked over both requests' verify rows.
  const std::vector<int> batch_to_state = {2, 0};
  size_t p = 0;
  for (size_t b = 0; b < batch_to_state.size(); ++b) {
    const std::vector<int32_t>& rows =
        proposed[static_cast<size_t>(batch_to_state[b])];
    const size_t num_logits =
        static_cast<size_t>(cu_num_logits[b + 1] - cu_num_logits[b]);
    CHECK(input_ids[static_cast<size_t>(li[p])] ==
          last_sampled[static_cast<size_t>(batch_to_state[b])]);
    ++p;
    for (size_t j = 1; j < num_logits; ++j, ++p) {
      CHECK(input_ids[static_cast<size_t>(li[p])] == rows[j - 1]);
    }
  }
}

// ─── G2.4: the first_logit_seq_pos guard (input_batch.py:344-348) ───────────
TEST_CASE("G2: a logits window reaching over the prompt tail keeps the prompt "
          "token and still scatters the drafts") {
  // seq_len > prefill_len (so this is not the prefill early-return), but
  // first_logit_seq_pos == seq_len - num_logits == 4 < prefill_len == 6, so
  // logits_start addresses a PROMPT slot. Upstream skips the committed-token
  // store there and leaves the prompt id alone (:344-348). The draft scatter is
  // deliberately NOT under that guard (:350-361), so it still runs.
  const std::vector<int32_t> proposed = {41, 42};
  const int stride = 2;
  std::vector<int32_t> input_ids = {1004, 1005, kPlaceholder};
  const std::vector<int32_t> last_sampled = {777};
  const std::vector<int32_t> qsl = {0, 3};
  const std::vector<int32_t> seq_lens = {7};
  const std::vector<int32_t> prefill_len = {6};
  const std::vector<int32_t> cu_num_logits = {0, 3};

  combine_sampled_and_draft_tokens(
      input_ids, Identity(1), last_sampled, qsl, seq_lens, prefill_len,
      FlattenDrafts({proposed}, stride, /*pad=*/7777), stride, cu_num_logits);

  CHECK(input_ids == std::vector<int32_t>{1004, 41, 42});
  // The committed id was NOT written anywhere.
  for (const int32_t id : input_ids) CHECK(id != 777);
}

// ─── G2.5: draft-only step (num_new_sampled_tokens == 0) ────────────────────
TEST_CASE("G2: num_new_sampled_tokens==0 scatters k drafts and no committed id") {
  // num_draft_tokens == num_logits - 0 == num_logits, so the whole logits window
  // is drafts and nothing is spliced from last_sampled.
  const std::vector<int32_t> proposed = {13, 14};
  const int stride = 2;
  std::vector<int32_t> input_ids(2, kPlaceholder);
  const std::vector<int32_t> last_sampled = {666};
  const std::vector<int32_t> qsl = {0, 2};
  const std::vector<int32_t> seq_lens = {9};
  const std::vector<int32_t> prefill_len = {5};
  const std::vector<int32_t> cu_num_logits = {0, 2};

  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, Identity(1), last_sampled, qsl, seq_lens, prefill_len,
      FlattenDrafts({proposed}, stride, /*pad=*/7777), stride, cu_num_logits,
      /*num_new_sampled_tokens=*/0);

  CHECK(input_ids == std::vector<int32_t>{13, 14});
  CHECK(li == std::vector<int32_t>{0, 1});
}

// ─── G2.6: a prefill row is still untouched, drafts or not ──────────────────
TEST_CASE("G2: a prefill row with a draft row in the buffer is left alone") {
  // seq_len <= prefill_len returns before both stores (input_batch.py:337-341),
  // so a stale draft row for that req_state cannot leak into a prompt chunk.
  const int stride = 2;
  std::vector<int32_t> input_ids = {200, 201, 202};
  const std::vector<int32_t> last_sampled = {999};
  const std::vector<int32_t> qsl = {0, 3};
  const std::vector<int32_t> seq_lens = {9};
  const std::vector<int32_t> prefill_len = {9};  // seq_len == prefill_len
  const std::vector<int32_t> cu_num_logits = {0, 3};

  combine_sampled_and_draft_tokens(
      input_ids, Identity(1), last_sampled, qsl, seq_lens, prefill_len,
      FlattenDrafts({{55, 56}}, stride, /*pad=*/7777), stride, cu_num_logits);

  CHECK(input_ids == std::vector<int32_t>{200, 201, 202});
}

// ─── G2.7: the stride's padding cannot leak past a request's draft count ─────
TEST_CASE("G2: a ragged verify row does not write the stride's pad over the "
          "next row's prompt token") {
  // G2.3 already SAYS the count comes from cu_num_logits and never from
  // draft_tokens_stride, but its own shape cannot FALSIFY that: replacing the
  // scatter's bound with draft_tokens_stride there writes one slot past the end
  // of a 7-element input_ids, and `for (const int32_t id : input_ids)` iterates
  // size() and never looks at it. Swapping its rows does not help either — the
  // over-write then lands on query_end_i, which the next row's committed store
  // immediately overwrites.
  //
  // This is the shape that CATCHES it. A ragged verify row (stride 3, k 2) is
  // followed by a PREFILL row, which returns at seq_len <= prefill_len and
  // writes nothing at all, so a pad landing in its prompt chunk stays there and
  // is read by the model as a prompt token.
  //
  //   correct:  500 70 71 900 901   the prompt slot holds 900
  //   b < stride: 500 70 71 7777 901   the pad has replaced a PROMPT token
  constexpr int32_t kPad = 7777;
  const int stride = 3;  // the speculator's max draft length
  const std::vector<std::vector<int32_t>> proposed = {
      {70, 71},  // req_state 0, k=2 -> one pad slot at the end of its row
      {},        // req_state 1 is the prefill row; it has no drafts
  };
  const std::vector<int32_t> draft_tokens =
      FlattenDrafts(proposed, stride, kPad);

  // Row 0 occupies [0,3): committed id + 2 drafts. Row 1 is a 2-token prompt
  // chunk at [3,5), seeded with real prompt ids and not with the placeholder,
  // because what this case watches is a PROMPT token surviving.
  std::vector<int32_t> input_ids = {kPlaceholder, kPlaceholder, kPlaceholder,
                                    900, 901};
  const std::vector<int32_t> last_sampled = {500, 999};
  const std::vector<int32_t> prefill_len = {5, 10};
  const std::vector<int32_t> qsl = {0, 3, 5};
  const std::vector<int32_t> seq_lens = {8, 4};  // row 0 verify, row 1 prefill
  const std::vector<int32_t> cu_num_logits = {0, 3, 4};

  const std::vector<int32_t> li = combine_sampled_and_draft_tokens(
      input_ids, Identity(2), last_sampled, qsl, seq_lens, prefill_len,
      draft_tokens, stride, cu_num_logits);

  CHECK(input_ids == std::vector<int32_t>{500, 70, 71, 900, 901});
  // Said once more on the slot the pad would land in, so the failure names it.
  CHECK(input_ids[3] == 900);
  for (const int32_t id : input_ids) CHECK(id != kPad);
  // Row 1 still scores its single prefill logit at query_end - 1.
  CHECK(li == std::vector<int32_t>{0, 1, 2, 4});
}
