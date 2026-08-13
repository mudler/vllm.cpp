// The gate on the golden-comparison helper itself (issue #449).
//
// vllm.cpp original (test harness); no upstream mirror. Every LTX-2.5 and
// MiniMax-H3 golden comparison, and both DeepSeek-V4 forward gates, reduce to
// `max|got - want|`. Until this file existed, that reduction was NaN-BLIND in
// both of its spellings, so the instrument could not see the defect class it
// exists to catch: an all-NaN brick against correct goldens reported
// `max|diff| = 0.0` and passed every bound.
//
// The first test case below pins the blindness of the two historical spellings,
// so the reason this helper is written the way it is cannot be lost. The rest
// gate the replacement.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "support/max_abs_diff.h"

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// A correct golden and a correct result, so the ONLY thing under test is what
// the reduction does with a non-finite operand.
constexpr float kWant[] = {0.25f, -1.5f, 3.0f, 0.0f};
constexpr size_t kN = std::size(kWant);

std::vector<float> Correct() { return {0.25f, -1.5f, 3.0f, 0.0f}; }

}  // namespace

TEST_CASE("max|diff| helper: BOTH historical spellings are NaN-blind (issue #449)") {
  // An all-NaN result against the correct golden — the worst input there is.
  const std::vector<float> got(kN, kNaN);

  // form A, as tests/vllm/models/test_ltx2.cpp:189 had it. NaN is never > x.
  double form_a = 0.0;
  for (size_t i = 0; i < kN; ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(kWant[i]));
    if (d > form_a) form_a = d;
  }

  // form B, as test_ltx2_vae.cpp:183 / test_ltx2_text_encoder.cpp:152 /
  // test_minimax_h3.cpp:143 had it. std::max(a, b) is `a < b ? b : a`, and
  // `a < NaN` is false, so it returns `a`.
  double form_b = 0.0;
  for (size_t i = 0; i < kN; ++i) {
    form_b = std::max(form_b,
                      std::abs(static_cast<double>(got[i]) - static_cast<double>(kWant[i])));
  }

  CHECK(form_a == 0.0);
  CHECK(form_b == 0.0);
  // ...and so every bound a gate could state was satisfied by an all-NaN result.
  CHECK(form_a < 1e-5);
  CHECK(form_b < 1e-5);
}

TEST_CASE("max|diff| scan: a finite comparison is unchanged") {
  const std::vector<float> got = Correct();
  const vllm_test::MaxAbsDiffScanResult same = vllm_test::MaxAbsDiffScan(got.data(), kWant, kN);
  CHECK(same.ok());
  CHECK(same.worst == 0.0);

  std::vector<float> off = Correct();
  off[2] += 0.125f;
  const vllm_test::MaxAbsDiffScanResult moved = vllm_test::MaxAbsDiffScan(off.data(), kWant, kN);
  CHECK(moved.ok());
  // Exact, not Approx: doctest's Approx carries a 1.19e-5 ABSOLUTE floor by
  // default, and 3.125f - 3.0f is exact in binary anyway.
  CHECK(moved.worst == 0.125);
}

TEST_CASE("max|diff| scan: a NaN in the RESULT is a failure, not a zero") {
  const std::vector<float> all_nan(kN, kNaN);
  const vllm_test::MaxAbsDiffScanResult r =
      vllm_test::MaxAbsDiffScan(all_nan.data(), kWant, kN);
  CHECK_FALSE(r.ok());
  CHECK(r.bad_index == 0);
  CHECK(std::isnan(r.bad_got));
  CHECK(std::isinf(r.worst));
  // The bound the gate actually states must now REFUSE.
  CHECK_FALSE(r.worst < 1e-5);

  // One NaN buried in an otherwise perfect result is caught at its own index.
  std::vector<float> one = Correct();
  one[2] = kNaN;
  const vllm_test::MaxAbsDiffScanResult buried =
      vllm_test::MaxAbsDiffScan(one.data(), kWant, kN);
  CHECK_FALSE(buried.ok());
  CHECK(buried.bad_index == 2);
  CHECK(std::isinf(buried.worst));
}

TEST_CASE("max|diff| scan: a non-finite GOLDEN is a failure too") {
  // A generator that emits NaN into a golden is the same instrument failure seen
  // from the other side, and `NaN - NaN` is NaN, so it reduced to 0.0 as well.
  const float nan_golden[] = {0.25f, kNaN, 3.0f, 0.0f};
  const std::vector<float> got = Correct();
  const vllm_test::MaxAbsDiffScanResult r =
      vllm_test::MaxAbsDiffScan(got.data(), nan_golden, kN);
  CHECK_FALSE(r.ok());
  CHECK(r.bad_index == 1);
  CHECK(std::isnan(r.bad_want));
  CHECK(std::isinf(r.worst));
}

TEST_CASE("max|diff| scan: Inf on either side is a failure") {
  std::vector<float> pos = Correct();
  pos[1] = kInf;
  CHECK_FALSE(vllm_test::MaxAbsDiffScan(pos.data(), kWant, kN).ok());

  std::vector<float> neg = Correct();
  neg[3] = -kInf;
  CHECK_FALSE(vllm_test::MaxAbsDiffScan(neg.data(), kWant, kN).ok());

  const float inf_golden[] = {0.25f, -1.5f, kInf, 0.0f};
  const std::vector<float> got = Correct();
  CHECK_FALSE(vllm_test::MaxAbsDiffScan(got.data(), inf_golden, kN).ok());

  // Matching infinities are NOT a pass: `Inf - Inf` is NaN, so the old reduction
  // read two saturated tensors as identical.
  std::vector<float> both = Correct();
  both[2] = kInf;
  CHECK_FALSE(vllm_test::MaxAbsDiffScan(both.data(), inf_golden, kN).ok());
}

TEST_CASE("max|diff| scan: -FLT_MAX is FINITE and must still compare") {
  // LTX-2.5's `additive_mask` legitimately carries -FLT_MAX (-3.40282347e+38,
  // ltx2_text_goldens.inc:471). Saturating is not non-finite, and hardening the
  // guard must not turn a legitimate magnitude into a refusal. The subtraction
  // runs in double, so it cannot overflow either.
  const float lo = -std::numeric_limits<float>::max();
  const float mask_golden[] = {lo, 0.0f, lo, 0.0f};
  const std::vector<float> got = {lo, 0.0f, lo, 0.0f};
  const vllm_test::MaxAbsDiffScanResult r =
      vllm_test::MaxAbsDiffScan(got.data(), mask_golden, kN);
  CHECK(r.ok());
  CHECK(r.worst == 0.0);

  // And a real difference at that magnitude is still measured, not saturated.
  const std::vector<float> drifted = {0.0f, 0.0f, lo, 0.0f};
  const vllm_test::MaxAbsDiffScanResult moved =
      vllm_test::MaxAbsDiffScan(drifted.data(), mask_golden, kN);
  CHECK(moved.ok());
  CHECK(moved.worst > 1e38);
}

// Named so that the `*RAISES*` CTest filter below matches exactly one case —
// doctest's name matching is case-INSENSITIVE.
TEST_CASE("max|diff| wrapper: a finite comparison reports nothing") {
  const std::vector<float> got = Correct();
  CHECK(vllm_test::MaxAbsDiff(got, kWant, kN) == 0.0);
  const std::vector<float> want(kWant, kWant + kN);
  CHECK(vllm_test::MaxAbsDiff(got, want) == 0.0);
}

// The doctest-facing wrapper must RAISE on a non-finite operand, not merely
// return +infinity: the `CHECK(worst > tol)` "these must differ" callers
// (test_deepseek_v4_forward.cpp, test_deepseek_v4_mtp.cpp) would otherwise read a
// NaN as a difference and pass. A raised failure cannot be gated from inside the
// same run without counting as a failure, so this case is skipped by default and
// run by the `test_max_abs_diff_nan_raises` CTest entry, which requires a
// NON-ZERO exit. See tests/CMakeLists.txt.
TEST_CASE("max|diff| wrapper: RAISES on a NaN operand" * doctest::skip()) {
  const std::vector<float> all_nan(kN, kNaN);
  const double worst = vllm_test::MaxAbsDiff(all_nan, kWant, kN);
  CHECK(std::isinf(worst));
}
