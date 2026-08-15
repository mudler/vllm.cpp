// vllm.cpp original (test harness); no upstream mirror.
//
// ONE max|diff| golden-comparison helper for every model gate, hardened against
// the defect that issue #449 records: the two obvious spellings of the reduction
// are both NaN-BLIND, so a brick emitting all-NaN compared against entirely
// correct goldens reports `max|diff| = 0.0` and PASSES every bound.
//
//   form A:  if (d > worst) worst = d;                  // NaN is never > anything
//   form B:  worst = std::max(worst, std::abs(...));    // std::max(a,b) is
//                                                       // `a < b ? b : a`, and
//                                                       // `a < NaN` is false, so
//                                                       // it returns `a`
//
// Both leave `worst` at 0.0 for an all-NaN input. The comparison helper could not
// see the one defect class it exists to catch. On LTX-2.5 phase L5 that concealed
// a real `Ltx2SigmaSchedule(1, ...)` returning `{-nan, 0}` against a correct
// golden, reported green.
//
// The fix, and its polarity: a non-finite operand on EITHER side is a FAILURE,
// never a zero. `MaxAbsDiffScan` reports the first offending index and returns
// +infinity, which fails every `< tol` bound; `MaxAbsDiff` additionally raises a
// doctest failure naming the index and both values, so the `> tol` "these must
// differ" callers cannot read a NaN as a difference either.
//
// NON-FINITE GOLDENS. No golden compared through this helper legitimately holds
// Inf or NaN. Saturating-but-FINITE values do exist and are fine: LTX-2.5's
// `additive_mask` carries -FLT_MAX (-3.40282347e+38), and it is compared with an
// EXACT comparison, not through here. A future golden that genuinely needs a
// non-finite value takes an explicit, documented carve-out at its own call site;
// it never relaxes this helper.
#pragma once

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace vllm_test {

// The reduction, with no doctest side effect, so a test can gate the guard
// itself without turning its own suite red.
struct MaxAbsDiffScanResult {
  static constexpr size_t kNone = static_cast<size_t>(-1);

  // max|got - want|, or +infinity when any operand is non-finite.
  double worst = 0.0;
  // Index of the FIRST non-finite operand, or kNone when both sides are finite.
  size_t bad_index = kNone;
  float bad_got = 0.0f;
  float bad_want = 0.0f;

  bool ok() const { return bad_index == kNone; }
};

inline MaxAbsDiffScanResult MaxAbsDiffScan(const float* got, const float* want, size_t count) {
  MaxAbsDiffScanResult r;
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
      r.worst = std::numeric_limits<double>::infinity();
      r.bad_index = i;
      r.bad_got = got[i];
      r.bad_want = want[i];
      return r;
    }
    // Safe here, unlike form A: both operands are known finite by the branch
    // above, and the difference of two finite floats cannot overflow a double,
    // so `d` is never NaN and the comparison is never the blind one.
    const double d = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    if (d > worst) worst = d;
  }
  r.worst = worst;
  return r;
}

// The doctest-facing helper the model gates call. Fails loudly on a non-finite
// operand and still returns +infinity, so a bound check either way reports red.
inline double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t count) {
  REQUIRE(got.size() == count);
  const MaxAbsDiffScanResult r = MaxAbsDiffScan(got.data(), want, count);
  if (!r.ok()) {
    FAIL_CHECK("max|diff|: NON-FINITE operand at index "
               << r.bad_index << " (got = " << r.bad_got << ", want = " << r.bad_want
               << "). A NaN here used to reduce to 0.0 and PASS — see issue #449.");
  }
  return r.worst;
}

inline double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  return MaxAbsDiff(a, b.data(), b.size());
}

}  // namespace vllm_test
