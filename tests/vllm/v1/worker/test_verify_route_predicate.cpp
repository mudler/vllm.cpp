// SPEC-DFLASH2 A2-2 (#2802) — the spec-decode VERIFY route predicate, and the
// per-STEP reduction both sampling entry points and one refusal turn on.
//
// WHAT THIS PROTECTS. Upstream has ONE sampling entry point and asks the
// question once (`vllm/v1/worker/gpu/model_runner.py:1129` @ pin 5559679229:
// `if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None`).
// We have two — `GPUModelRunner::sample_tokens` and
// `GPUModelRunner::sample_tokens_async` — and a third site, the async input
// combine, refuses on the NEGATION of the same rule. Three readings of one rule
// is how a route and its refusal drift apart.
//
// WHY THE GRANULARITY IS THE POINT OF THIS FILE. `num_draft_tokens` is the batch
// TOTAL, `sum(num_draft_tokens_per_req)`. The forward produced ONE expanded
// logits tensor for the whole step, with `Σ(1 + k_i)` rows, and either the
// rejection sampler consumes it or the plain sampler does. A per-REQUEST reading
// of the same rule — "row i drafted nothing, so it is an ordinary decode row" —
// answers differently on a MIXED step, and would hand the plain sampler a tensor
// whose rows are not one per request.
//
// This repository has shipped exactly that shape before: a per-request refusal
// paired with a per-step route predicate (#2710,
// `tests/vllm/v1/worker/test_combine_row_predicate.cpp`). It survived 27
// mutations because every test used `num_reqs == 1`, so the two readings agreed
// on every input.
//
// WHAT ACTUALLY PREVENTS THE #2710 SHAPE HERE IS THE SIGNATURE, NOT THIS FILE,
// and an earlier version of this comment claimed otherwise. `StepRoutesToVerify`
// takes ONE `int32_t` (`prepare_inputs.h`). There is no per-request vector in
// scope inside it, so the per-row mutation that comment described
// (`return per_req[0] > 0`) cannot be written at all — a reader who tried it
// would find the file does not compile, which is not a red and not evidence.
// The mutation this file DOES detect is one on the same scalar: widen the
// boundary to `!= 0` and the negative-total case goes red; invert it and every
// case goes red.
//
// AND THE TWO READINGS ARE NOT INDEPENDENT. For non-negative per-request counts
// — the only kind `prepare_inputs` can build, since they are sizes —
// `any_i RowCarriesDraftTokens(k_i)` is IDENTICALLY equal to
// `StepRoutesToVerify(Σ k_i)`: a sum of non-negative numbers is positive exactly
// when one term is. The DISAGREEMENT case below asserts that equality rather
// than a disagreement, and what it pins is narrower and still worth pinning: the
// MAJORITY of the per-row answers is false on a mixed step, so a re-derivation
// that took the per-row answer for the row it happens to be looking at — which
// is what #2710 did — answers wrongly for the step, and the tensor the sampler
// is handed has Σ(1 + k_i) rows rather than one per request. That is the fact
// this file exists to keep executable.
#include <doctest/doctest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::v1::RowCarriesDraftTokens;
using vllm::v1::StepRoutesToVerify;

namespace {

// The step total, exactly as `prepare_inputs` builds `StepInputs::num_draft_
// tokens`: sum(num_draft_tokens_per_req).
int32_t StepTotal(const std::vector<int32_t>& per_req) {
  return std::accumulate(per_req.begin(), per_req.end(), 0);
}

}  // namespace

TEST_CASE("the verify route is the step's draft TOTAL, and zero means the plain sampler") {
  // The production default, on every step, forever: no speculator, so the
  // scheduler populated no `scheduled_spec_decode_tokens` and the total is 0.
  // This is the branch that must stay byte-identical.
  CHECK_FALSE(StepRoutesToVerify(0));

  // One request with one draft is the smallest verify step: 2 expanded rows.
  CHECK(StepRoutesToVerify(1));
  CHECK(StepRoutesToVerify(24));

  // The total is a count and is never negative; a `!= 0` re-derivation would
  // route a negative total to the verify arm, so the boundary is a strict `>`.
  CHECK_FALSE(StepRoutesToVerify(-1));
}

TEST_CASE("the row predicate answers about ONE row and never about the step") {
  CHECK_FALSE(RowCarriesDraftTokens(0));
  CHECK(RowCarriesDraftTokens(1));
  CHECK(RowCarriesDraftTokens(3));
}

TEST_CASE("THE MIXED STEP: rows that drafted nothing still route to verify") {
  // ── Three requests in one step. Rows 0 and 2 drafted NOTHING this step (a
  // request that was just admitted, or whose drafts were all rolled back); row 1
  // carries two drafts.
  //
  // A PER-REQUEST reading answers "no drafts" for rows 0 and 2 and would send
  // them to the plain sampler. But there is no per-row choice to make: the
  // forward produced ONE expanded tensor of Σ(1 + k_i) = 1 + 3 + 1 = 5 rows for
  // this step, and the plain sampler expects one row per request. The PER-STEP
  // reading answers "verify" for the whole step, which is the only answer that
  // matches the tensor the sampler is handed.
  //
  // This case does NOT separate `any_i RowCarriesDraftTokens(k_i)` from
  // `StepRoutesToVerify(Σ k_i)`; for non-negative counts nothing can, and the
  // last block below asserts they agree. What it separates is a reading that
  // answers for ONE row from a reading that answers for the step, which is the
  // mistake #2710 made and which a `num_reqs == 1` suite cannot see.
  const std::vector<int32_t> per_req{0, 2, 0};
  const int32_t total = StepTotal(per_req);
  CHECK(total == 2);
  CHECK(StepRoutesToVerify(total));

  // The two readings, side by side and executable. The MAJORITY of the per-row
  // answers is false while the step answer is true, so a re-derivation that
  // reads one row and calls it the step's answer is wrong here two times in
  // three. That is the discriminating fact, and it is a fact about the ROWS,
  // not about the aggregate — which agrees, as the last block states.
  CHECK_FALSE(RowCarriesDraftTokens(per_req[0]));
  CHECK(RowCarriesDraftTokens(per_req[1]));
  CHECK_FALSE(RowCarriesDraftTokens(per_req[2]));

  // The expanded-row count the verify arm is handed, spelled out: it is NOT
  // num_reqs, which is exactly what the plain sampler would assume.
  int32_t expanded = 0;
  for (const int32_t k : per_req) expanded += 1 + k;
  CHECK(expanded == 5);
  CHECK(expanded != static_cast<int32_t>(per_req.size()));
}

TEST_CASE("a step where EVERY row drafted nothing does not route to verify") {
  // The all-zero mixed shape: `num_draft_tokens_per_req` is present and sized,
  // and every entry is 0, so the expanded tensor IS one row per request and the
  // plain sampler is correct. A predicate that keyed off "the per-request vector
  // is non-empty" rather than off the total would route this to the verify arm
  // and verify nothing.
  const std::vector<int32_t> per_req{0, 0, 0};
  CHECK(StepTotal(per_req) == 0);
  CHECK_FALSE(StepRoutesToVerify(StepTotal(per_req)));
}

TEST_CASE("the refusal at the async input combine fires on the MIXED step too") {
  // `runner.cpp`'s async input-combine site refuses a step that scheduled draft
  // tokens, because the draft buffer its combine scatters from is not wired yet
  // (A2-3, #2644). It asks `!StepRoutesToVerify(step.num_draft_tokens)` — the
  // route's own function, negated, rather than a second reading of the rule.
  //
  // NOTHING HERE CAN PROVE THAT SHARING, and this case does not pretend to: two
  // source sites calling one function is a property of the source, and the
  // instrument for it is the reviewer's mutation (change one site's predicate
  // and this file must go red for the route half; the refusal half is
  // unreachable today and is recorded as owed in the row's spec). What this case
  // DOES pin is that route and refusal PARTITION the input: on the mixed step
  // the route fires and the refusal does not, and there is no value of the total
  // for which both or neither hold. The refusal site carries one check the route
  // does not — a separate `>= 0` on the total — because negating `> 0` admits a
  // negative total that the pre-A2-2 `== 0` refusal rejected; that check is a
  // structural guard beside the predicate, not a second reading of it.
  const std::vector<int32_t> mixed{0, 2, 0};
  const bool routes = StepRoutesToVerify(StepTotal(mixed));
  const bool refuses = !StepRoutesToVerify(StepTotal(mixed));
  CHECK(routes);
  CHECK_FALSE(refuses);

  // The per-request reading a re-derivation would have used, made explicit. The
  // OR over the rows agrees with the step answer — it must, for non-negative
  // counts — so the danger was never that the aggregate differs. It is that the
  // rows individually say "no drafts" for the majority of this step.
  bool any_row = false;
  for (const int32_t k : mixed) any_row = any_row || RowCarriesDraftTokens(k);
  CHECK(any_row == routes);
  CHECK_FALSE(RowCarriesDraftTokens(mixed[0]));
  CHECK_FALSE(RowCarriesDraftTokens(mixed[2]));
}
