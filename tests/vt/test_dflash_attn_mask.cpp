// SPEC-DFLASH2 (#2784, discharging #1900). The DFlash attention MASK BOUND,
// gated on the CPU because it is compiled into nine CUDA kernels this machine
// has no toolchain for, and because the property that matters — a NON-CAUSAL
// layer carrying a window still attends within it — is invisible from any token
// gate in this repository. The verify is lossless, so a draft that attends to
// the wrong keys changes no output token; only ACCEPTANCE falls, which is how
// #2784 reached 0.06 accepted-per-16 at 8,159 prompt tokens with every suite
// green.
#include <doctest/doctest.h>

#include <cstdint>
#include <initializer_list>

#include "vt/dflash_attn_mask.h"

using vt::DFlashMaskSpan;
using vt::DFlashMaskSpanOf;

namespace {

// The bound this file replaced, transcribed EXACTLY as it stood at every one of
// the eleven sites before #2784. It is here as the reference for the causal
// byte-identity claim ONLY — the claim that the fix cannot move a DFlash1
// checkpoint — and it is deliberately NOT used to check the non-causal arm,
// where it is the defect.
DFlashMaskSpan LegacyBound(int64_t ii, int64_t num_keys, bool causal, int64_t window) {
  DFlashMaskSpan s;
  s.hi = causal ? ii : (num_keys - 1);
  s.lo = 0;
  if (causal && window > 0) s.lo = ii - (window - 1) > 0 ? ii - (window - 1) : 0;
  return s;
}

}  // namespace

// THE PORTED RULE. `_maybe_symmetrize_window`
// (vllm/v1/attention/backends/flash_attn.py:319-330 @ the parity pin
// 5559679229) makes a causal `(w, 0)` window symmetric `(w, w)` when attention
// is non-causal. So a non-causal query row sees [ii-(w-1), ii+(w-1)], clipped
// to the axis — not the whole axis, which is what this tree did.
TEST_CASE("dflash mask: a NON-CAUSAL layer with a window attends SYMMETRICALLY") {
  // ii = 40 in a 46-key axis, window 8. Symmetric bound is [33, 47] -> [33, 45].
  const DFlashMaskSpan s = DFlashMaskSpanOf(/*ii=*/40, /*num_keys=*/46,
                                            /*causal=*/false, /*window=*/8);
  CHECK(s.lo == 33);
  CHECK(s.hi == 45);  // ii + 7 = 47, clipped to num_keys - 1

  // The RIGHT half must bind too when the axis is long enough to expose it.
  const DFlashMaskSpan t = DFlashMaskSpanOf(/*ii=*/40, /*num_keys=*/200,
                                            /*causal=*/false, /*window=*/8);
  CHECK(t.lo == 33);
  CHECK(t.hi == 47);
}

// The regression this fix must not cause: a DFlash1 checkpoint declares no
// `is_causal`, so its SWA layers resolve causal and must keep the exact bound
// they had. Swept rather than spot-checked, because the claim is byte identity
// over the whole domain and one example cannot carry it.
TEST_CASE("dflash mask: the CAUSAL arm is byte-identical to the pre-#2784 bound") {
  int checked = 0;
  for (int64_t num_keys = 1; num_keys <= 40; ++num_keys) {
    for (int64_t ii = 0; ii < num_keys; ++ii) {
      for (int64_t window = 0; window <= 45; ++window) {
        const DFlashMaskSpan a = DFlashMaskSpanOf(ii, num_keys, /*causal=*/true, window);
        const DFlashMaskSpan b = LegacyBound(ii, num_keys, /*causal=*/true, window);
        REQUIRE(a.lo == b.lo);
        REQUIRE(a.hi == b.hi);
        ++checked;
      }
    }
  }
  CHECK(checked == 40 * 41 / 2 * 46);  // the sweep RAN, and over the shape claimed
}

// A full-attention layer carries window 0 and must be untouched on both arms —
// upstream's "leaves full-attention (-1, -1) ... untouched".
TEST_CASE("dflash mask: window 0 is full attention on both arms") {
  const DFlashMaskSpan nc = DFlashMaskSpanOf(/*ii=*/3, /*num_keys=*/10,
                                             /*causal=*/false, /*window=*/0);
  CHECK(nc.lo == 0);
  CHECK(nc.hi == 9);
  const DFlashMaskSpan c = DFlashMaskSpanOf(/*ii=*/3, /*num_keys=*/10,
                                            /*causal=*/true, /*window=*/0);
  CHECK(c.lo == 0);
  CHECK(c.hi == 3);
}

// A window wider than the axis binds nothing, on either arm. A case built at
// that shape would go green whatever the guard did, so this pins the boundary
// rather than standing in for the cases above.
TEST_CASE("dflash mask: a window wider than the axis is inert") {
  for (int64_t ii = 0; ii < 6; ++ii) {
    const DFlashMaskSpan nc = DFlashMaskSpanOf(ii, /*num_keys=*/6,
                                               /*causal=*/false, /*window=*/64);
    CHECK(nc.lo == 0);
    CHECK(nc.hi == 5);
  }
}

// The bound the MMA kernel's block-level STAGING range depends on: `lo` and
// `hi` are both non-decreasing in `ii`, so the union over a block's rows is
// [span(first).lo, span(last).hi] and the kernel needs no third formula.
TEST_CASE("dflash mask: both bounds are MONOTONE in the query row") {
  for (bool causal : {false, true}) {
    for (int64_t window : {0, 1, 3, 8, 64}) {
      DFlashMaskSpan prev = DFlashMaskSpanOf(0, 50, causal, window);
      for (int64_t ii = 1; ii < 50; ++ii) {
        const DFlashMaskSpan cur = DFlashMaskSpanOf(ii, 50, causal, window);
        REQUIRE(cur.lo >= prev.lo);
        REQUIRE(cur.hi >= prev.hi);
        prev = cur;
      }
    }
  }
}
