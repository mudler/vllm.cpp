// SPEC-DFLASH2 W10 (#1857) — the PagedAttention lane-classification seam.
//
// `include/vt/paged_attn_route.h` is the host half of the CUDA dispatch's
// prefill/decode class split, factored out so a box without a GPU can pin the
// ROUTING of a uniform-qlen speculative verify: classified + admissible ⇒ the
// DECODE class (split-KV), everything else ⇒ byte-identical to the shipped
// `num_tokens > num_reqs` predicate.
//
// Mutation targets: the shape guard's `> 1` (a pure decode must never be
// "uniform spec"), the `q * num_reqs` product (a stale field over a rewritten
// batch must never route), and the is-prefill polarity (dropping the
// spec-as-decode conjunct re-reds the verify-routing case).
#include <doctest/doctest.h>

#include "vt/paged_attn_route.h"

using vt::PagedAttnIsPrefill;
using vt::PagedAttnUniformSpecShape;

TEST_CASE("route: the classified uniform verify shape is admitted") {
  // The #1857 measured shape: 1 request, q=9 verify (k=8).
  CHECK(PagedAttnUniformSpecShape(/*num_tokens=*/9, /*num_reqs=*/1, /*uq=*/9));
  // Batched verify: 4 requests at q=3.
  CHECK(PagedAttnUniformSpecShape(/*num_tokens=*/12, /*num_reqs=*/4, /*uq=*/3));
}

TEST_CASE("route: unclassified, stale and degenerate shapes are refused") {
  // Not classified (the default 0): never a spec batch.
  CHECK_FALSE(PagedAttnUniformSpecShape(9, 1, 0));
  // q == 1 is a pure decode, which routes decode by shape already.
  CHECK_FALSE(PagedAttnUniformSpecShape(4, 4, 1));
  // STALE FIELD over a padded pure-decode rebuild: the BuildPaddedDecode
  // rewrites make num_tokens == num_reqs, and S == q*S only at q == 1.
  CHECK_FALSE(PagedAttnUniformSpecShape(/*num_tokens=*/16, /*num_reqs=*/16, /*uq=*/9));
  // Shape/classification mismatch (a batch that shrank or grew).
  CHECK_FALSE(PagedAttnUniformSpecShape(/*num_tokens=*/5, /*num_reqs=*/1, /*uq=*/9));
  CHECK_FALSE(PagedAttnUniformSpecShape(/*num_tokens=*/9, /*num_reqs=*/2, /*uq=*/9));
  // Degenerate batch.
  CHECK_FALSE(PagedAttnUniformSpecShape(0, 0, 9));
}

TEST_CASE("route: a spec-as-decode batch takes the DECODE class") {
  // THE #1857 ROUTING PIN: the uniform-qlen verify must NOT be prefill when the
  // decode lane admits it. Before W10 the dispatch had no spec term and this
  // batch rode the prefill flash ladder at num_splits=1.
  CHECK_FALSE(PagedAttnIsPrefill(/*num_tokens=*/9, /*num_reqs=*/1,
                                 /*spec_as_decode=*/true));
  CHECK_FALSE(PagedAttnIsPrefill(/*num_tokens=*/12, /*num_reqs=*/4,
                                 /*spec_as_decode=*/true));
}

TEST_CASE("route: without the spec admission the shipped predicate holds verbatim") {
  // Prefill: more tokens than requests.
  CHECK(PagedAttnIsPrefill(9, 1, /*spec_as_decode=*/false));
  CHECK(PagedAttnIsPrefill(5, 2, /*spec_as_decode=*/false));
  // Pure decode: one token per request.
  CHECK_FALSE(PagedAttnIsPrefill(4, 4, /*spec_as_decode=*/false));
  CHECK_FALSE(PagedAttnIsPrefill(1, 1, /*spec_as_decode=*/false));
}
