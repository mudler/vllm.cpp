// ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710) — the combine's row predicate, and the
// per-STEP reduction the refusal turns on.
//
// WHAT THIS PROTECTS. `ModelRegistry::Forward` refuses a step whose HOST token
// identifiers are stale when the registered forward does not read the device
// ones. "Stale" is not "the mirror is on": the runner's combine splices a row
// only when that row has moved past its known prefill, so an all-prefill step
// leaves the host vector perfectly good. Getting that term wrong in either
// direction is a defect:
//
//   too NARROW  -> the guard misses a stale step and the model decodes from
//                  token id 0, which is the whole defect class this row exists
//                  to close (#1305, #2496, #2544).
//   too WIDE    -> the guard refuses `ForwardLlamaModelEmbedding`, a pooling
//                  forward that is CORRECT today because every request it serves
//                  is prefill-only. A refusal that breaks a working path is
//                  worse than the bug.
//
// WHY THE GRANULARITY IS THE POINT OF THIS FILE. Staleness is a per-ROW fact and
// the decision it feeds is a per-STEP one. A per-REQUEST reading of the same rule
// lets a MIXED step through on account of its prefill rows while its decode row
// reads identifiers the runner never wrote. This repository has shipped exactly
// that shape before — a per-request refusal paired with a per-step route veto —
// and it survived 27 mutations because every test used `num_reqs == 1`, so the
// two readings agreed on every test input. The DISAGREEMENT case below is the
// one input that separates them, and it is why this file is not redundant with a
// single-request test.
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::v1::AnyRowSplicedByCombine;
using vllm::v1::CombineSplicesRow;

TEST_CASE("the row predicate is the combine's own boundary") {
  // A decode row: the sequence has moved past everything known at admission, so
  // the previous step's sampled token is spliced over the host id.
  CHECK(CombineSplicesRow(/*seq_len=*/9, /*prefill_len=*/8));

  // A prefill row: still consuming known prompt tokens.
  CHECK_FALSE(CombineSplicesRow(/*seq_len=*/5, /*prefill_len=*/8));

  // THE BOUNDARY, and it is the one a re-derivation gets wrong. The chunk that
  // EXACTLY completes prefill is still prefill: its input is the prompt token,
  // there is no sampled token to splice, and the combine leaves it alone. A `>=`
  // here would mark a correct step stale and refuse a forward that is fine.
  CHECK_FALSE(CombineSplicesRow(/*seq_len=*/8, /*prefill_len=*/8));
}

TEST_CASE("the step reduction is an OR over rows, not a per-request answer") {
  // ── THE DISAGREEMENT CASE ────────────────────────────────────────────────
  // Three requests in one step. Rows 0 and 1 are prefill; row 2 is a decode row.
  //
  // A PER-REQUEST reading answers "not stale" for rows 0 and 1 and would let the
  // step proceed on their account — serving row 2 from a host identifier the
  // runner never wrote. The PER-STEP reading answers "stale" for the whole step,
  // because the forward is handed all three rows at once and either reads the
  // device buffer or does not.
  //
  // This input is the only kind that separates the two readings, which is why a
  // `num_reqs == 1` suite cannot gate this rule at all.
  const std::vector<int32_t> seq_lens{5, 5, 9};
  const std::vector<int32_t> prefill_len{8, 8, 8};

  CHECK(AnyRowSplicedByCombine(seq_lens, prefill_len, /*idx_mapping=*/nullptr,
                               /*num_reqs=*/3));

  // The two readings, made executable side by side. Every INDIVIDUAL row answer
  // is available, and the majority of them are false; the step answer is true.
  // If these ever agreed for all rows, the case above would have stopped
  // discriminating and this assertion says so.
  CHECK_FALSE(CombineSplicesRow(seq_lens[0], prefill_len[0]));
  CHECK_FALSE(CombineSplicesRow(seq_lens[1], prefill_len[1]));
  CHECK(CombineSplicesRow(seq_lens[2], prefill_len[2]));
}

TEST_CASE("an all-prefill step is NOT stale, which is what keeps pooling alive") {
  // Every row still inside its prompt: the combine splices nothing and the host
  // vector is authoritative. `ForwardLlamaModelEmbedding` serves only steps of
  // this shape, so a predicate that answered true here would refuse a correct
  // model — the failure mode this term exists to prevent.
  const std::vector<int32_t> seq_lens{4, 8, 2};
  const std::vector<int32_t> prefill_len{8, 8, 8};
  CHECK_FALSE(AnyRowSplicedByCombine(seq_lens, prefill_len,
                                     /*idx_mapping=*/nullptr, /*num_reqs=*/3));
}

TEST_CASE("an all-decode step is stale, and one row is enough") {
  const std::vector<int32_t> all_decode{9, 10, 11};
  const std::vector<int32_t> prefill_len{8, 8, 8};
  CHECK(AnyRowSplicedByCombine(all_decode, prefill_len, /*idx_mapping=*/nullptr,
                               /*num_reqs=*/3));

  // A SINGLE decode row, in the LAST position, with everything before it prefill.
  // A reduction that returned on the first row's answer rather than OR-ing would
  // pass this and hand the step to a model that cannot read it.
  const std::vector<int32_t> tail_only{5, 5, 5, 12};
  const std::vector<int32_t> pf4{8, 8, 8, 8};
  CHECK(AnyRowSplicedByCombine(tail_only, pf4, /*idx_mapping=*/nullptr,
                               /*num_reqs=*/4));
}

TEST_CASE("an empty step is not stale") {
  const std::vector<int32_t> none;
  const std::vector<int32_t> prefill_len{8};
  CHECK_FALSE(AnyRowSplicedByCombine(none, prefill_len, /*idx_mapping=*/nullptr,
                                     /*num_reqs=*/0));
}

TEST_CASE("prefill_len is read through idx_mapping, as the combine reads it") {
  // After an abort or a finish reorders `req_states` against the dense batch,
  // batch row `i` is not req_state slot `i`. The combine indexes `prefill_len` by
  // the MAPPED slot, so this must too: indexing by the batch row would compare
  // one request's sequence length against another request's prefill.
  //
  // Batch row 0 -> req_state 1, batch row 1 -> req_state 0.
  //   mapped:   seq 9 vs prefill_len[1] == 20 -> prefill;
  //             seq 5 vs prefill_len[0] == 4  -> decode.  => stale
  //   unmapped: seq 9 vs prefill_len[0] == 4  -> decode.  => also stale, but for
  //             the wrong row, so the STALE answer alone cannot discriminate.
  const std::vector<int32_t> seq_lens{9, 5};
  const std::vector<int32_t> prefill_len{4, 20};
  const int32_t mapping[2] = {1, 0};
  CHECK(AnyRowSplicedByCombine(seq_lens, prefill_len, mapping, /*num_reqs=*/2));

  // The discriminating pair: under the MAPPED reading nothing is spliced, while
  // an unmapped reading would call row 0 a decode row and refuse a clean step.
  const std::vector<int32_t> seq2{9, 5};
  const std::vector<int32_t> pf2{4, 20};
  const int32_t identity_like[2] = {1, 1};
  //   row 0: seq 9 vs prefill_len[1] == 20 -> prefill
  //   row 1: seq 5 vs prefill_len[1] == 20 -> prefill
  CHECK_FALSE(AnyRowSplicedByCombine(seq2, pf2, identity_like, /*num_reqs=*/2));
}
