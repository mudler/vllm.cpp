// SPEC-DFLASH2 W10 (#1857) — the spec-as-decode reorder-threshold policy.
//
// Mirrors `AttentionMetadataBuilder._init_reorder_batch_threshold`
// (vllm/v1/attention/backend.py:718-736 @ b389ac2946; byte-identical at the
// parity pin 5559679229):
//
//   max_num_queries_for_spec =
//       1 + (2 if parallel_drafting else 1) * num_speculative_tokens
//
// and FlashInfer's classification (flashinfer.py:852-860): a uniform batch
// whose query length is at most the threshold stays on the DECODE kernel.
//
// The boundary cases are the mutation targets: an off-by-one in the threshold
// formula, a dropped parallel-drafting factor, or an inverted comparison each
// red a case below.
#include <doctest/doctest.h>

#include "vllm/v1/attention/backend.h"

using vllm::v1::SpecAsDecodeQueryLen;
using vllm::v1::SpecAsDecodeReorderThreshold;

TEST_CASE("spec-as-decode threshold: 1 + 2K for a parallel-drafting speculator") {
  // DFlash/DFlash2/DSpark are parallel_drafting (speculative.py:1064-1065).
  CHECK(SpecAsDecodeReorderThreshold(/*k=*/8, /*parallel=*/true) == 17);
  CHECK(SpecAsDecodeReorderThreshold(/*k=*/3, /*parallel=*/true) == 7);
  CHECK(SpecAsDecodeReorderThreshold(/*k=*/1, /*parallel=*/true) == 3);
}

TEST_CASE("spec-as-decode threshold: 1 + K for an autoregressive drafter") {
  CHECK(SpecAsDecodeReorderThreshold(/*k=*/8, /*parallel=*/false) == 9);
  CHECK(SpecAsDecodeReorderThreshold(/*k=*/1, /*parallel=*/false) == 2);
}

TEST_CASE("spec-as-decode threshold: speculation off keeps the plain decode threshold") {
  // Upstream's reorder_batch_threshold default is 1; the spec raise only ever
  // applies over a configured num_speculative_tokens.
  CHECK(SpecAsDecodeReorderThreshold(/*k=*/0, /*parallel=*/true) == 1);
  CHECK(SpecAsDecodeReorderThreshold(/*k=*/0, /*parallel=*/false) == 1);
  CHECK(SpecAsDecodeReorderThreshold(/*k=*/-3, /*parallel=*/true) == 1);
}

TEST_CASE("spec-as-decode classification: the 1 + 2K boundary is EXACT") {
  // q == threshold classifies; q == threshold + 1 does not. The DFlash2
  // measured shape (#1857): k=8 verify at q=9 sits inside 1+2*8=17.
  CHECK(SpecAsDecodeQueryLen(/*q=*/9, /*k=*/8, /*parallel=*/true) == 9);
  CHECK(SpecAsDecodeQueryLen(/*q=*/17, /*k=*/8, /*parallel=*/true) == 17);
  CHECK(SpecAsDecodeQueryLen(/*q=*/18, /*k=*/8, /*parallel=*/true) == 0);
  // The fixture shape (k=3, parallel): threshold 7, verify width 4.
  CHECK(SpecAsDecodeQueryLen(/*q=*/4, /*k=*/3, /*parallel=*/true) == 4);
  CHECK(SpecAsDecodeQueryLen(/*q=*/7, /*k=*/3, /*parallel=*/true) == 7);
  CHECK(SpecAsDecodeQueryLen(/*q=*/8, /*k=*/3, /*parallel=*/true) == 0);
}

TEST_CASE("spec-as-decode classification: the autoregressive boundary is 1 + K") {
  CHECK(SpecAsDecodeQueryLen(/*q=*/9, /*k=*/8, /*parallel=*/false) == 9);
  CHECK(SpecAsDecodeQueryLen(/*q=*/10, /*k=*/8, /*parallel=*/false) == 0);
}

TEST_CASE("spec-as-decode classification: q <= 1 and spec-off never classify") {
  // A pure decode routes decode by shape already; 0 is "not classified".
  CHECK(SpecAsDecodeQueryLen(/*q=*/1, /*k=*/8, /*parallel=*/true) == 0);
  CHECK(SpecAsDecodeQueryLen(/*q=*/0, /*k=*/8, /*parallel=*/true) == 0);
  CHECK(SpecAsDecodeQueryLen(/*q=*/-2, /*k=*/8, /*parallel=*/true) == 0);
  // Speculation off: threshold is 1, so no q > 1 can classify.
  CHECK(SpecAsDecodeQueryLen(/*q=*/2, /*k=*/0, /*parallel=*/true) == 0);
  CHECK(SpecAsDecodeQueryLen(/*q=*/2, /*k=*/0, /*parallel=*/false) == 0);
}
