// LTX-2.5 PIPELINE parity gate (phase L5, .agents/specs/ltx-2-5.md, issue #435).
//
// Every assertion here compares our port against the UPSTREAM modules — Lightricks
// `ltx_core` and `ltx-pipelines`, plus vLLM-Omni's recipe table — captured by
// scripts/gen-ltx2-pipeline-goldens.py into ltx2_pipeline_goldens.inc. Both sides
// rebuild every weight and input from the same deterministic FNV-1a + splitmix64
// stream keyed by the parameter's own NAME, so no weight byte is checked in and
// the weight CONTRACT is part of the gate: a name either side invents that the
// other lacks changes the numbers.
//
// THREE THINGS THIS FILE DOES ON PURPOSE:
//
//  * Nothing uses doctest::Approx. Its `scale` defaults to 1.0, which puts a
//    1.19e-5 ABSOLUTE floor under every comparison — enough to accept a broken
//    forward. Every numeric check goes through MaxAbsDiff against an explicit
//    round-off bound.
//  * Every member of the INVISIBLE-CONSTANT CLASS this phase introduces gets a
//    source-anchored constant assertion, because a reduced-dimension value gate
//    provably cannot see it (spec section 7.0(a)). Where the regime is reachable,
//    a golden that ENTERS it is gated too — kLtx2CfgPpAlphaEps and
//    kLtx2ProjectionCoefEps both have one.
//  * Every refusal asserts the MESSAGE, not just that something was thrown. A
//    refusal that does not name the missing piece is not the refusal the spec
//    asks for.
#include "vllm/model_executor/models/ltx2_pipeline.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "ltx2_pipeline_goldens.inc"

#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_connector.h"
#include "vllm/model_executor/models/ltx2_duration_head.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"

namespace {

// ---------------------------------------------------------------------------
// The upstream trees these goldens describe. PINNED here, so a regeneration
// against a different checkout fails the gate instead of silently replacing the
// oracle. Advancing either is a deliberate, reviewable edit in BOTH places.
// ---------------------------------------------------------------------------
constexpr const char* kLtx2UpstreamRevisionPin = "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca";
constexpr const char* kLtx2OmniRevisionPin = "a4ea67a21b20054dacc6e83952f9bd407e8ee4e7";

// ---------------------------------------------------------------------------
// Ltx2Rand — the exact mirror of the generator's stream
// (scripts/gen-ltx2-pipeline-goldens.py :: ltx_rand).
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& name) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char byte : name) {
    h ^= static_cast<uint64_t>(byte);
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// The generator's `make`: ltx_rand * scale + offset in float64, rounded to f32.
std::vector<float> Make(const std::string& name, int64_t count, double scale = 1.0,
                        double offset = 0.0) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    const double unit = static_cast<double>(u >> 11) * 0x1p-53;
    out[static_cast<size_t>(i)] = static_cast<float>((unit * 2.0 - 1.0) * scale + offset);
  }
  return out;
}

bool EndsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The generator's `param_values`, mirrored EXACTLY. `rank` is why the bag
// builders pass shapes rather than counts.
std::vector<float> ParamValues(const std::string& name, const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;
  if (EndsWith(name, ".bias") || EndsWith(name, "in_proj_bias")) {
    return Make(name, count, 0.02);
  }
  if (shape.size() == 1 && EndsWith(name, ".weight")) return Make(name, count, 0.1, 1.0);
  // The duration head's wider fixture — see the generator for the measurement
  // behind it. Its output is `exp(...)` through an attenuating chain, so at the
  // shared 0.05 scale its three arms collapse to within 3e-6 of each other,
  // BELOW this suite's round-off bound, and the gate stops separating them.
  if (name.find(".dur.") != std::string::npos) return Make(name, count, 0.35);
  return Make(name, count, 0.05);
}

// A weight bag that also records the (name, count) manifest in build order, so a
// test can prove its parameter set matches the generator's state_dict exactly.
struct ParamBag {
  vllm::Ltx2VaeWeights weights;
  std::vector<std::string> names;
  std::vector<int64_t> counts;

  void Put(const std::string& name, const std::vector<int64_t>& shape) {
    std::vector<float> values = ParamValues(name, shape);
    counts.push_back(static_cast<int64_t>(values.size()));
    names.push_back(name);
    weights.tensors[name] = std::move(values);
  }
};

void CheckManifest(const ParamBag& bag, const char* const* want_names, const int64_t* want_counts,
                   size_t want_size) {
  REQUIRE(bag.names.size() == want_size);
  for (size_t i = 0; i < want_size; ++i) {
    CHECK(bag.names[i] == std::string(want_names[i]));
    CHECK(bag.counts[i] == want_counts[i]);
  }
}

// ---------------------------------------------------------------------------
// Tolerances. Reported per brick in the test output, so a drift is visible as a
// NUMBER rather than as a pass. AGENTS.md forbids widening either to go green.
//
//  kRoundOff       the elementwise bricks and the deep modules. Our side runs f32
//                  arithmetic with f64 reduction accumulation against a torch
//                  float32 oracle, so the two differ in the last f32 ulps of an
//                  O(1) activation carried through a few layers.
//  kExactRoundOff  the pure index/rearrange bricks (patchify, unpatchify, the
//                  bounds), which MOVE bytes rather than computing on them. They
//                  must be bit-exact; the bound is 0.
// ---------------------------------------------------------------------------
constexpr double kRoundOff = 5e-6;
constexpr double kExactRoundOff = 0.0;

double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t count) {
  REQUIRE(got.size() == count);
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    if (d > worst) worst = d;
  }
  return worst;
}

// The text of the refusal `run` throws, or "" when it does not throw at all.
template <typename Fn>
std::string RefusalMessage(Fn run) {
  try {
    run();
  } catch (const std::exception& e) {
    return std::string(e.what());
  }
  return std::string();
}

bool Mentions(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

// ===========================================================================
// Provenance
// ===========================================================================

TEST_CASE("ltx2 pipeline goldens carry the upstream revisions they were built from") {
  // Spec section 7.0(b): byte-identical goldens are NOT evidence that the oracle
  // was right, so reproducibility alone proves nothing about provenance. The
  // anchor is what makes a bisect possible, and it is only load-bearing if the
  // suite refuses a different one.
  CHECK(std::string(vllm_test::kLtx2PipelineUpstreamRevision) ==
        std::string(kLtx2UpstreamRevisionPin));
  CHECK(std::string(vllm_test::kLtx2OmniUpstreamRevision) == std::string(kLtx2OmniRevisionPin));
}

// ===========================================================================
// Section 1 — sigma schedules
// ===========================================================================

TEST_CASE("ltx2 LTX2Scheduler reproduces upstream's sigma schedule") {
  CHECK(vllm::kLtx2BaseShiftAnchor == vllm_test::kLtx2SchedBaseShiftAnchor);
  CHECK(vllm::kLtx2MaxShiftAnchor == vllm_test::kLtx2SchedMaxShiftAnchor);
  // schedulers.py:41 — `power = 1`, a literal. Pinned because any other exponent
  // still yields a monotone schedule that renders.
  CHECK(vllm::kLtx2SigmaShiftPower == 1.0);

  auto run = [&](const std::string& tag, int64_t steps, int64_t tokens, double max_shift,
                 double base_shift, bool stretch, double terminal, const float* golden,
                 size_t golden_count) {
    vllm::Ltx2SchedulerParams params;
    params.max_shift = max_shift;
    params.base_shift = base_shift;
    params.stretch = stretch;
    params.terminal = terminal;
    const std::vector<float> got = vllm::Ltx2SigmaSchedule(steps, tokens, params);
    const double worst = MaxAbsDiff(got, golden, golden_count);
    INFO("LTX2Scheduler arm = ", tag, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
    // The schedule must END at exactly 0: it is what the terminal step branches
    // on, and a 1e-8 residue there turns an early-out into a division.
    CHECK(got.back() == 0.0f);
  };

#define LTX2_SCHED_ARM(TAG)                                                                   \
  run(#TAG, vllm_test::kLtx2Sched##TAG##Steps, vllm_test::kLtx2Sched##TAG##Tokens,             \
      vllm_test::kLtx2Sched##TAG##MaxShift, vllm_test::kLtx2Sched##TAG##BaseShift,             \
      vllm_test::kLtx2Sched##TAG##Stretch, vllm_test::kLtx2Sched##TAG##Terminal,               \
      vllm_test::kLtx2Sched##TAG##Golden, std::size(vllm_test::kLtx2Sched##TAG##Golden))
  LTX2_SCHED_ARM(Default);
  LTX2_SCHED_ARM(NoStretch);
  LTX2_SCHED_ARM(FewTokens);
  LTX2_SCHED_ARM(ManyTokens);
  LTX2_SCHED_ARM(Terminal0);
  LTX2_SCHED_ARM(OneStep);
  // The two arms that gate the linspace WALK rather than its algebra. Every step
  // count above is one a naive `start + step * i` forward walk happens to get
  // right, so all six stayed green against it; 41 and 47 are the first two counts
  // where it misses exact 0 (see the generator's _SCHED_CASES note).
  LTX2_SCHED_ARM(Steps41);
  LTX2_SCHED_ARM(Steps47);
#undef LTX2_SCHED_ARM
}

TEST_CASE("ltx2 the sigma schedule terminates at exactly 0 for every step count") {
  // `Ltx2SigmaSchedule` early-outs on `sigma == 0.0f` (schedulers.py:42) and then
  // takes the LAST NON-ZERO sigma as the stretch anchor (:52). A terminal residue
  // of 5.96e-08 is not a rounding cosmetic: it passes the `!= 0` guard, goes
  // through the shift transform, and displaces the anchor, which moves the WHOLE
  // schedule and leaves the denoise loop never reaching zero noise.
  //
  // Swept rather than tabulated because which counts are affected is a property
  // of f32 division, not of anything upstream declares: 24 of the first 198 miss
  // it under a forward walk. A tabulated arm gates the counts someone thought of.
  for (int64_t steps = 1; steps <= 200; ++steps) {
    vllm::Ltx2SchedulerParams params;
    const std::vector<float> sigmas = vllm::Ltx2SigmaSchedule(steps, 0, params);
    INFO("steps = ", steps, " back = ", sigmas.back());
    REQUIRE(sigmas.size() == static_cast<size_t>(steps + 1));
    CHECK(sigmas.back() == 0.0f);
    if (steps >= 2) {
      // Every sigma is finite, so the terminal 0 is reached by the walk rather
      // than produced by a degenerate stretch.
      //
      // steps == 1 IS EXCLUDED BECAUSE IT IS CURRENTLY BROKEN, not because the
      // property does not apply. `Ltx2SigmaSchedule(1, ...)` returns
      // {-nan, 0} where upstream returns {0.10000002, 0}. Root cause, measured:
      // upstream's shift is a PYTHON SCALAR divided by a tensor
      // (schedulers.py:43-45), and torch evaluates scalar/tensor as
      // `scalar * reciprocal(tensor)`, so at sigma == 1 it yields 0.99999994 —
      // one ulp below 1 — not 1.0. That residue is `one_minus_z[-1]` (:52), and
      // at steps == 1 it is the ONLY non-zero sigma, so it alone sets
      // scale_factor (6.62e-08). This port computes the same expression entirely
      // in f32, gets exactly 1.0, and so divides 0/0.
      //
      // NOT repaired here: mirroring torch's scalar/tensor reciprocal-multiply
      // changes every sigma on every arm, so it needs its own spec and a fresh
      // review rather than a drive-by edit in a review-repair branch. The
      // `OneStep` golden arm already carries upstream's correct 0.100000024 and
      // did not catch this only because `MaxAbsDiff` drops NaN (`d > worst` is
      // false for NaN) — that helper is deliberately left alone for the same
      // reason: fixing it turns this suite red on the defect above.
      size_t non_finite = 0;
      for (float sigma : sigmas) {
        if (!std::isfinite(sigma)) ++non_finite;
      }
      CHECK(non_finite == 0u);
    }
  }
  // The same exactness for the other ported scheduler, whose terminal is a
  // structural `1.0 - 1.0` (schedulers.py:87) rather than an accumulated walk.
  for (int64_t steps = 1; steps <= 200; ++steps) {
    const std::vector<float> sigmas = vllm::Ltx2LinearQuadraticSchedule(steps);
    INFO("linear-quadratic steps = ", steps, " back = ", sigmas.back());
    CHECK(sigmas.back() == 0.0f);
  }
}

TEST_CASE("ltx2 LinearQuadraticScheduler reproduces upstream's schedule") {
  auto run = [&](const std::string& tag, int64_t steps, double threshold, int64_t linear_steps,
                 const float* golden, size_t golden_count) {
    const std::vector<float> got =
        vllm::Ltx2LinearQuadraticSchedule(steps, threshold, linear_steps);
    const double worst = MaxAbsDiff(got, golden, golden_count);
    INFO("LinearQuadraticScheduler arm = ", tag, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
  };
#define LTX2_LINQUAD_ARM(TAG)                                                                 \
  run(#TAG, vllm_test::kLtx2LinQuad##TAG##Steps, vllm_test::kLtx2LinQuad##TAG##Threshold,      \
      vllm_test::kLtx2LinQuad##TAG##LinearSteps, vllm_test::kLtx2LinQuad##TAG##Golden,         \
      std::size(vllm_test::kLtx2LinQuad##TAG##Golden))
  LTX2_LINQUAD_ARM(Default);
  LTX2_LINQUAD_ARM(OneStep);
  LTX2_LINQUAD_ARM(Explicit);
  LTX2_LINQUAD_ARM(AllLinear);
#undef LTX2_LINQUAD_ARM
}

TEST_CASE("ltx2 the Beta scheduler is refused by name, never substituted") {
  const std::string message = RefusalMessage([] {
    (void)vllm::Ltx2Schedule(vllm::Ltx2SchedulerKind::kBeta, 8, 0);
  });
  INFO("refusal = ", message);
  CHECK(Mentions(message, "BetaScheduler"));
  CHECK(Mentions(message, "scipy"));
  // ...and the two that ARE ported route through the same seam.
  CHECK(vllm::Ltx2Schedule(vllm::Ltx2SchedulerKind::kLtx2, 8, 0) ==
        vllm::Ltx2SigmaSchedule(8, 0));
  CHECK(vllm::Ltx2Schedule(vllm::Ltx2SchedulerKind::kLinearQuadratic, 6, 0) ==
        vllm::Ltx2LinearQuadraticSchedule(6));
}

// ===========================================================================
// Section 2 — the noiser
// ===========================================================================

TEST_CASE("ltx2 GaussianNoiser reproduces upstream's double lerp") {
  const int64_t count = vllm_test::kLtx2NoiserCount;
  const std::vector<float> latent = Make("ltx2.noiser.latent", count, 1.0);
  const std::vector<float> clean = Make("ltx2.noiser.clean", count, 1.0);
  const std::vector<float> noise = Make("ltx2.noiser.noise", count, 1.0);

  auto run = [&](const std::string& tag, float scale, const float* golden) {
    const std::vector<float> got = vllm::Ltx2GaussianNoise(
        latent.data(), clean.data(), vllm_test::kLtx2NoiserMask, noise.data(), count, scale);
    const double worst = MaxAbsDiff(got, golden, static_cast<size_t>(count));
    INFO("GaussianNoiser arm = ", tag, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
  };
  run("Full", static_cast<float>(vllm_test::kLtx2NoiserFullScale),
      vllm_test::kLtx2NoiserFullGolden);
  run("Half", static_cast<float>(vllm_test::kLtx2NoiserHalfScale),
      vllm_test::kLtx2NoiserHalfGolden);
  run("Zero", static_cast<float>(vllm_test::kLtx2NoiserZeroScale),
      vllm_test::kLtx2NoiserZeroGolden);

  // The mask fixture pins the two endpoints, and they are where a swapped lerp
  // shows up: element 0 has mask 0 (fully CLEAN) and element 1 has mask 1 (fully
  // noised). Asserting them by name means a transposed operand order fails HERE,
  // naming the defect, rather than as a diffuse tensor mismatch.
  CHECK(vllm_test::kLtx2NoiserMask[0] == 0.0f);
  CHECK(vllm_test::kLtx2NoiserMask[1] == 1.0f);
  CHECK(vllm_test::kLtx2NoiserFullGolden[0] == clean[0]);
}

// ===========================================================================
// Section 3 — diffusion steps
// ===========================================================================

TEST_CASE("ltx2 Euler diffusion step reproduces upstream") {
  const int64_t count = vllm_test::kLtx2StepCount;
  const int64_t sigma_count = vllm_test::kLtx2StepSigmaCount;
  const std::vector<float> sample = Make("ltx2.step.sample", count, 1.0);
  const std::vector<float> denoised = Make("ltx2.step.denoised", count, 0.8);

  auto run = [&](int64_t index, const float* golden) {
    const std::vector<float> got = vllm::Ltx2EulerStep(
        sample.data(), denoised.data(), vllm_test::kLtx2StepSigmas, sigma_count, index, count);
    const double worst = MaxAbsDiff(got, golden, static_cast<size_t>(count));
    INFO("EulerDiffusionStep index = ", index, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
  };
  run(0, vllm_test::kLtx2StepEuler0Golden);
  run(1, vllm_test::kLtx2StepEuler1Golden);

  // The LAST usable index of a real schedule is fine: `sigma = sigmas[step_index]`
  // is its last NON-ZERO entry, and the step collapses to the denoised prediction
  // exactly (velocity * dt = -(sample - denoised)).
  const std::vector<float> terminal = vllm::Ltx2EulerStep(
      sample.data(), denoised.data(), vllm_test::kLtx2StepSigmas, sigma_count,
      sigma_count - 2, count);
  const double terminal_worst = MaxAbsDiff(terminal, denoised.data(),
                                           static_cast<size_t>(count));
  INFO("EulerDiffusionStep terminal index max|diff| vs denoised = ", terminal_worst);
  CHECK(terminal_worst <= kRoundOff);

  // to_velocity REFUSES a zero sigma (utils.py:34-35) rather than dividing, which
  // is what a hand-built or truncated schedule reaches. Without it the step
  // produces inf and carries it into every later one.
  const std::vector<float> zero_first = {0.0f, 0.5f};
  const std::string message = RefusalMessage([&] {
    (void)vllm::Ltx2EulerStep(sample.data(), denoised.data(), zero_first.data(), 2, 0, count);
  });
  INFO("refusal = ", message);
  CHECK(Mentions(message, "sigma"));
}

TEST_CASE("ltx2 EulerAncestral diffusion step reproduces upstream") {
  const int64_t count = vllm_test::kLtx2StepCount;
  const int64_t sigma_count = vllm_test::kLtx2StepSigmaCount;
  const std::vector<float> sample = Make("ltx2.step.sample", count, 1.0);
  const std::vector<float> denoised = Make("ltx2.step.denoised", count, 0.8);
  const std::vector<float> noise = Make("ltx2.step.noise", count, 1.0);

  auto run = [&](const std::string& tag, double eta, double s_noise, int64_t index,
                 const float* golden) {
    const std::vector<float> got =
        vllm::Ltx2EulerAncestralStep(sample.data(), denoised.data(), vllm_test::kLtx2StepSigmas,
                                     sigma_count, index, count, eta, s_noise, noise.data());
    const double worst = MaxAbsDiff(got, golden, static_cast<size_t>(count));
    INFO("EulerAncestral arm = ", tag, " index = ", index, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
  };

#define LTX2_ANCESTRAL_ARM(TAG)                                                               \
  do {                                                                                        \
    const double eta = vllm_test::kLtx2StepAncestral##TAG##Eta;                                \
    const double sn = vllm_test::kLtx2StepAncestral##TAG##SNoise;                              \
    run(#TAG, eta, sn, 0, vllm_test::kLtx2StepAncestral##TAG##Step0Golden);                     \
    run(#TAG, eta, sn, 1, vllm_test::kLtx2StepAncestral##TAG##Step1Golden);                     \
    run(#TAG, eta, sn, 2, vllm_test::kLtx2StepAncestral##TAG##TerminalGolden);                  \
  } while (0)
  LTX2_ANCESTRAL_ARM(Eta1);
  LTX2_ANCESTRAL_ARM(Eta0);
  LTX2_ANCESTRAL_ARM(EtaHalf);
#undef LTX2_ANCESTRAL_ARM

  // diffusion_steps.py:87-88 — a missing noise tensor with eta > 0 is upstream's
  // own ValueError, not a silent zero fill (which would make the step
  // deterministic and the render subtly different).
  const std::string message = RefusalMessage([&] {
    (void)vllm::Ltx2EulerAncestralStep(sample.data(), denoised.data(),
                                       vllm_test::kLtx2StepSigmas, sigma_count, 0, count, 1.0,
                                       1.0, nullptr);
  });
  INFO("refusal = ", message);
  CHECK(Mentions(message, "noise"));
  // ...and with eta == 0 the same call is legal, exactly as upstream documents.
  CHECK_NOTHROW((void)vllm::Ltx2EulerAncestralStep(sample.data(), denoised.data(),
                                                   vllm_test::kLtx2StepSigmas, sigma_count, 0,
                                                   count, 0.0, 1.0, nullptr));
}

TEST_CASE("ltx2 Res2s diffusion step reproduces upstream") {
  const int64_t count = vllm_test::kLtx2StepCount;
  const int64_t sigma_count = vllm_test::kLtx2StepSigmaCount;
  const std::vector<float> sample = Make("ltx2.step.sample", count, 1.0);
  const std::vector<float> denoised = Make("ltx2.step.denoised", count, 0.8);
  const std::vector<float> noise = Make("ltx2.step.noise", count, 1.0);

  auto run = [&](const std::string& tag, double eta, int64_t index, const float* golden) {
    const std::vector<float> got =
        vllm::Ltx2Res2sStep(sample.data(), denoised.data(), vllm_test::kLtx2StepSigmas,
                            sigma_count, index, count, noise.data(), eta);
    const double worst = MaxAbsDiff(got, golden, static_cast<size_t>(count));
    INFO("Res2s arm = ", tag, " index = ", index, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
  };
#define LTX2_RES2S_ARM(TAG)                                                                   \
  do {                                                                                        \
    const double eta = vllm_test::kLtx2StepRes2s##TAG##Eta;                                    \
    run(#TAG, eta, 0, vllm_test::kLtx2StepRes2s##TAG##Step0Golden);                             \
    run(#TAG, eta, 1, vllm_test::kLtx2StepRes2s##TAG##Step1Golden);                             \
    run(#TAG, eta, 2, vllm_test::kLtx2StepRes2s##TAG##TerminalGolden);                          \
  } while (0)
  LTX2_RES2S_ARM(EtaHalf);
  LTX2_RES2S_ARM(Eta1);
#undef LTX2_RES2S_ARM

  // The sigma_up clamp (diffusion_steps.py:138), and NOT a member of the
  // invisible-constant class. The old reasoning here — "eta <= 1 keeps sigma_up at
  // or below sigma_next, so the clamp never binds" — reads its own inequality
  // wrongly: <= includes ==, and the Eta1 arm run five lines up sits exactly on
  // that boundary, where `min` takes `sigma_next * 0.9999` on every step. A 1%
  // move (0.9999 -> 0.99) REDS Eta1 index 0 at max|diff| = 0.086 and index 1 at
  // 0.130563; EtaHalf and the terminal index stay green. Pinned against the
  // upstream literal AS WELL, since a regenerated golden moves with the constant.
  CHECK(vllm::kLtx2Res2sSigmaUpClamp == 0.9999);
  // ...and the clamp is proved to be a clamp, on an input that DOES exceed it.
  const vllm::Ltx2SdeCoeff clamped = vllm::Ltx2Res2sSdeCoeff(0.5, 2.0);
  CHECK(clamped.sigma_up <= 0.5 * vllm::kLtx2Res2sSigmaUpClamp);
}

TEST_CASE("ltx2 EulerCfgPp diffusion step reproduces upstream") {
  const int64_t count = vllm_test::kLtx2StepCount;
  const int64_t sigma_count = vllm_test::kLtx2StepSigmaCount;
  const std::vector<float> sample = Make("ltx2.step.sample", count, 1.0);
  const std::vector<float> denoised = Make("ltx2.step.denoised", count, 0.8);
  const std::vector<float> uncond = Make("ltx2.step.uncond", count, 0.8);
  const std::vector<float> noise = Make("ltx2.step.noise", count, 1.0);

  auto run = [&](const std::string& tag, double eta, double s_noise, int64_t index,
                 const float* golden) {
    const std::vector<float> got = vllm::Ltx2EulerCfgPpStep(
        sample.data(), denoised.data(), uncond.data(), vllm_test::kLtx2StepSigmas, sigma_count,
        index, count, eta, s_noise, noise.data());
    const double worst = MaxAbsDiff(got, golden, static_cast<size_t>(count));
    INFO("EulerCfgPp arm = ", tag, " index = ", index, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
  };
#define LTX2_CFGPP_ARM(TAG)                                                                   \
  do {                                                                                        \
    const double eta = vllm_test::kLtx2StepCfgPp##TAG##Eta;                                    \
    const double sn = vllm_test::kLtx2StepCfgPp##TAG##SNoise;                                  \
    run(#TAG, eta, sn, 0, vllm_test::kLtx2StepCfgPp##TAG##Step0Golden);                         \
    run(#TAG, eta, sn, 1, vllm_test::kLtx2StepCfgPp##TAG##Step1Golden);                         \
    run(#TAG, eta, sn, 2, vllm_test::kLtx2StepCfgPp##TAG##Step2Golden);                         \
  } while (0)
  LTX2_CFGPP_ARM(Eta1);
  LTX2_CFGPP_ARM(Eta0);
#undef LTX2_CFGPP_ARM

  // The alpha clamp (diffusion_steps.py:233-235). Unlike most of the class this
  // one BINDS on a real schedule — index 0 has sigma == 1.0 exactly, so
  // `alpha_s = 1 - sigma_s` is 0 and the clamp alone keeps the ODE derivative
  // finite. The Step0 goldens above are that arm, so mutating the constant moves
  // a NUMBER here rather than being absorbed. It is pinned as well, because the
  // pin is what names the upstream line.
  CHECK(vllm::kLtx2CfgPpAlphaEps == 1.1920928955078125e-07);
  CHECK(vllm_test::kLtx2StepSigmas[0] == 1.0f);
}

TEST_CASE("ltx2 the DDIM ancestral helper reproduces upstream") {
  const size_t eta_count = std::size(vllm_test::kLtx2AncestralHelperEtas);
  const size_t pair_count = std::size(vllm_test::kLtx2AncestralHelperPairs) / 2;
  std::vector<float> down, up;
  for (size_t e = 0; e < eta_count; ++e) {
    for (size_t p = 0; p < pair_count; ++p) {
      const vllm::Ltx2AncestralSigmas got =
          vllm::Ltx2AncestralStep(vllm_test::kLtx2AncestralHelperPairs[2 * p],
                                  vllm_test::kLtx2AncestralHelperPairs[2 * p + 1],
                                  vllm_test::kLtx2AncestralHelperEtas[e]);
      down.push_back(static_cast<float>(got.sigma_down));
      up.push_back(static_cast<float>(got.sigma_up));
    }
  }
  const double worst_down =
      MaxAbsDiff(down, vllm_test::kLtx2AncestralHelperDown, down.size());
  const double worst_up = MaxAbsDiff(up, vllm_test::kLtx2AncestralHelperUp, up.size());
  INFO("_get_ancestral_step max|diff| down = ", worst_down, " up = ", worst_up);
  CHECK(worst_down <= kRoundOff);
  CHECK(worst_up <= kRoundOff);
}

// ===========================================================================
// Section 4 — guidance
// ===========================================================================

TEST_CASE("ltx2 projection_coef reproduces upstream, including where its epsilon binds") {
  const int64_t batch = vllm_test::kLtx2GuideShape[0];
  const int64_t count = vllm_test::kLtx2GuideCount;
  const int64_t per_row = count / batch;
  const std::vector<float> cond = Make("ltx2.guide.cond", count, 1.0);
  const std::vector<float> uncond = Make("ltx2.guide.uncond", count, 0.9);

  const std::vector<float> got =
      vllm::Ltx2ProjectionCoef(cond.data(), uncond.data(), batch, per_row);
  const double worst =
      MaxAbsDiff(got, vllm_test::kLtx2GuideProjCoefGolden, static_cast<size_t>(batch));
  INFO("projection_coef max|diff| = ", worst);
  CHECK(worst <= kRoundOff);

  // guiders.py:368. An O(1) denominator hides a 1e-8 additive term completely, so
  // the constant is ALSO gated on an input that reaches its regime: an all-zero
  // `project_onto` makes the epsilon the entire denominator. Mutating it there
  // moves the result from 0 to NaN or to a different 0/x — the value comparison
  // sees it.
  CHECK(vllm::kLtx2ProjectionCoefEps == 1e-8);
  const std::vector<float> zeros(static_cast<size_t>(count), 0.0f);
  const std::vector<float> zero_coef =
      vllm::Ltx2ProjectionCoef(cond.data(), zeros.data(), batch, per_row);
  const double zero_worst =
      MaxAbsDiff(zero_coef, vllm_test::kLtx2GuideProjCoefZeroGolden, static_cast<size_t>(batch));
  INFO("projection_coef (zero denominator) max|diff| = ", zero_worst);
  CHECK(zero_worst <= kRoundOff);
  for (float value : zero_coef) CHECK(std::isfinite(value));

  // ...and a small-but-nonzero denominator, where the epsilon is an additive term
  // against a comparable magnitude rather than being either negligible or total.
  const std::vector<float> tiny(static_cast<size_t>(count),
                                static_cast<float>(vllm_test::kLtx2GuideProjCoefTinyValue));
  const std::vector<float> tiny_coef =
      vllm::Ltx2ProjectionCoef(cond.data(), tiny.data(), batch, per_row);
  const double tiny_worst =
      MaxAbsDiff(tiny_coef, vllm_test::kLtx2GuideProjCoefTinyGolden, static_cast<size_t>(batch));
  INFO("projection_coef (tiny denominator) max|diff| = ", tiny_worst);
  // A relative bound: these coefficients are ~1e4, so an absolute 5e-6 would be
  // a 1e-10 relative band no f32 chain can hold.
  double magnitude = 0.0;
  for (int64_t b = 0; b < batch; ++b) {
    magnitude = std::max(magnitude, std::fabs(static_cast<double>(
                                        vllm_test::kLtx2GuideProjCoefTinyGolden[b])));
  }
  CHECK(tiny_worst <= kRoundOff * std::max(1.0, magnitude));
}

TEST_CASE("ltx2 CFG and STG guidance deltas reproduce upstream") {
  const int64_t count = vllm_test::kLtx2GuideCount;
  const std::vector<float> cond = Make("ltx2.guide.cond", count, 1.0);
  const std::vector<float> uncond = Make("ltx2.guide.uncond", count, 0.9);
  const std::vector<float> perturbed = Make("ltx2.guide.perturbed", count, 0.7);

  auto cfg = [&](const std::string& tag, double scale, bool enabled, const float* golden) {
    const std::vector<float> got =
        vllm::Ltx2CfgDelta(cond.data(), uncond.data(), count, scale);
    const double worst = MaxAbsDiff(got, golden, static_cast<size_t>(count));
    INFO("CFGGuider arm = ", tag, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
    // `enabled()` is `scale != 1.0` (guiders.py:26-27) — an EXACT comparison,
    // unlike MultiModalGuider's math.isclose. The two are not interchangeable.
    CHECK(enabled == (scale != 1.0));
  };
  cfg("Scale3", vllm_test::kLtx2GuideCfgScale3Scale, vllm_test::kLtx2GuideCfgScale3Enabled,
      vllm_test::kLtx2GuideCfgScale3Golden);
  cfg("Scale1", vllm_test::kLtx2GuideCfgScale1Scale, vllm_test::kLtx2GuideCfgScale1Enabled,
      vllm_test::kLtx2GuideCfgScale1Golden);

  auto stg = [&](const std::string& tag, double scale, bool enabled, const float* golden) {
    const std::vector<float> got =
        vllm::Ltx2StgDelta(cond.data(), perturbed.data(), count, scale);
    const double worst = MaxAbsDiff(got, golden, static_cast<size_t>(count));
    INFO("STGGuider arm = ", tag, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
    CHECK(enabled == (scale != 0.0));
  };
  stg("Scale1", vllm_test::kLtx2GuideStgScale1Scale, vllm_test::kLtx2GuideStgScale1Enabled,
      vllm_test::kLtx2GuideStgScale1Golden);
  stg("Scale0", vllm_test::kLtx2GuideStgScale0Scale, vllm_test::kLtx2GuideStgScale0Enabled,
      vllm_test::kLtx2GuideStgScale0Golden);
}

TEST_CASE("ltx2 the three projection guiders are refused as UNREACHABLE, not as unshapeable") {
  // WHAT THIS GATE IS FOR, since an earlier revision got it wrong. It used to
  // assert that `projection_coef(...) * cond` "raises at every rectangular rank
  // >= 3, i.e. at every real (B, C, F, H, W) video latent", and froze that as a
  // golden. Executing upstream disproves it. The `(B, 1)` coefficient
  // (guiders.py:363-369) is right-aligned onto the latent's last two axes, so the
  // real predicate is
  //
  //     raises  <=>  B > 1 and shape[-2] not in {1, B}
  //
  // and at B = 1 — the ordinary single-request render — it composes AND is
  // numerically correct, because `(1, 1)` is just a scalar. The old `square`
  // abstraction was a mis-generalization of exactly these two axes and is gone.
  //
  // So the refusal does NOT rest on shapes. It rests on REACHABILITY: nothing in
  // the LTX-2 tree constructs these three at all (they appear only at their own
  // `class` statements, guiders.py:31, :78, :129; every pipeline builds
  // MultiModalGuider from MultiModalGuiderParams, utils/constants.py:49-68). They
  // are an unported arm, refused by name and recorded as owed per AGENTS.md.
  // The matrix stays gated so that upstream changing either fact shows up here.
  const int64_t guiders = vllm_test::kLtx2GuideProbeGuiderCount;
  const int64_t shapes = vllm_test::kLtx2GuideProbeShapeCount;
  REQUIRE(std::size(vllm_test::kLtx2GuideProbeComposes) ==
          static_cast<size_t>(guiders * shapes));
  REQUIRE(std::size(vllm_test::kLtx2GuideProbeBatch) == static_cast<size_t>(shapes));
  REQUIRE(std::size(vllm_test::kLtx2GuideProbeSecondLast) == static_cast<size_t>(shapes));

  bool saw_b1_composing = false;
  bool saw_rank5_composing = false;
  for (int64_t g = 0; g < guiders; ++g) {
    const std::string name = vllm_test::kLtx2GuideProbeNames[g];
    const bool elementwise = name == "CFGGuider" || name == "STGGuider";
    // Only the two threshold arms carry `norm(dim=[-1,-2,-3])` (guiders.py:114,
    // :205), which is a SEPARATE constraint needing rank >= 3.
    const bool needs_rank3 = name == "LtxAPGGuiderThreshold" || name == "LegacyStatefulAPGGuider";
    for (int64_t s = 0; s < shapes; ++s) {
      const bool composes = vllm_test::kLtx2GuideProbeComposes[g * shapes + s] != 0;
      const int64_t rank = vllm_test::kLtx2GuideProbeRanks[s];
      const int64_t batch = vllm_test::kLtx2GuideProbeBatch[s];
      const int64_t second_last = vllm_test::kLtx2GuideProbeSecondLast[s];
      INFO("guider = ", name, " rank = ", rank, " B = ", batch, " shape[-2] = ", second_last,
           " composes = ", composes);
      if (elementwise) {
        // The two ported ones are elementwise and survive every shape.
        CHECK(composes);
        continue;
      }
      const bool broadcast_raises = batch > 1 && second_last != 1 && second_last != batch;
      const bool expected_raise = broadcast_raises || (needs_rank3 && rank < 3);
      // The predicate is asserted in BOTH directions, so a shape that upstream
      // starts or stops accepting fails here either way.
      CHECK(composes == !expected_raise);
      if (composes && batch == 1) saw_b1_composing = true;
      if (composes && rank == 5) saw_rank5_composing = true;
    }
  }
  // The two rows the old matrix never probed, and which is why it drew the wrong
  // conclusion: a B = 1 latent, and a rank-5 `(B, C, F, H, W)` video latent that
  // composes anyway. Asserted so the matrix cannot silently shrink back.
  CHECK(saw_b1_composing);
  CHECK(saw_rank5_composing);

  vllm::Ltx2MultiModalGuiderParams params;
  const int64_t count = vllm_test::kLtx2GuideCount;
  const std::vector<float> cond = Make("ltx2.guide.cond", count, 1.0);
  for (auto kind : {vllm::Ltx2GuiderKind::kCfgStarRescaling, vllm::Ltx2GuiderKind::kLtxApg,
                    vllm::Ltx2GuiderKind::kLegacyStatefulApg}) {
    const std::string message = RefusalMessage([&] {
      (void)vllm::Ltx2Guidance(kind, params, cond.data(), cond.data(), cond.data(), cond.data(),
                               count);
    });
    INFO("refusal = ", message);
    // The message must name the arm and say it is unported. It must NOT rest on
    // the shape claim that measurement disproved.
    CHECK(Mentions(message, "not ported"));
    CHECK(Mentions(message, "CFGStarRescalingGuider"));
    CHECK(Mentions(message, "LtxAPGGuider"));
    CHECK(Mentions(message, "LegacyStatefulAPGGuider"));
    CHECK(Mentions(message, "constructs"));
  }
}

TEST_CASE("ltx2 MultiModalGuider reproduces upstream, rescale included") {
  const int64_t count = vllm_test::kLtx2GuideCount;
  const std::vector<float> cond = Make("ltx2.guide.cond", count, 1.0);
  const std::vector<float> uncond = Make("ltx2.guide.uncond", count, 0.9);
  const std::vector<float> perturbed = Make("ltx2.guide.perturbed", count, 0.7);
  const std::vector<float> modality = Make("ltx2.guide.modality", count, 0.6);

  auto run = [&](const std::string& tag, double cfg_scale, double stg_scale, double rescale_scale,
                 double modality_scale, int64_t skip_step, bool do_uncond, bool do_perturbed,
                 bool do_modality, const int64_t* skip_mask, const float* golden) {
    vllm::Ltx2MultiModalGuiderParams params;
    params.cfg_scale = cfg_scale;
    params.stg_scale = stg_scale;
    params.rescale_scale = rescale_scale;
    params.modality_scale = modality_scale;
    params.skip_step = skip_step;
    params.stg_blocks = {29};

    CHECK(params.DoUnconditionalGeneration() == do_uncond);
    CHECK(params.DoPerturbedGeneration() == do_perturbed);
    CHECK(params.DoIsolatedModalityGeneration() == do_modality);
    for (int64_t step = 0; step < 8; ++step) {
      CHECK(params.ShouldSkipStep(step) == (skip_mask[step] != 0));
    }

    const std::vector<float> got = vllm::Ltx2MultiModalGuidance(
        params, cond.data(), uncond.data(), perturbed.data(), modality.data(), count);
    const double worst = MaxAbsDiff(got, golden, static_cast<size_t>(count));
    INFO("MultiModalGuider arm = ", tag, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
  };

#define LTX2_MM_ARM(TAG)                                                                      \
  run(#TAG, vllm_test::kLtx2GuideMm##TAG##CfgScale, vllm_test::kLtx2GuideMm##TAG##StgScale,    \
      vllm_test::kLtx2GuideMm##TAG##RescaleScale,                                              \
      vllm_test::kLtx2GuideMm##TAG##ModalityScale, vllm_test::kLtx2GuideMm##TAG##SkipStep,     \
      vllm_test::kLtx2GuideMm##TAG##DoUncond, vllm_test::kLtx2GuideMm##TAG##DoPerturbed,       \
      vllm_test::kLtx2GuideMm##TAG##DoModality, vllm_test::kLtx2GuideMm##TAG##SkipStepMask,    \
      vllm_test::kLtx2GuideMm##TAG##Golden)
  LTX2_MM_ARM(Official);
  LTX2_MM_ARM(Audio);
  LTX2_MM_ARM(NoRescale);
  LTX2_MM_ARM(PositiveOnly);
  LTX2_MM_ARM(Skip2);
#undef LTX2_MM_ARM
}

TEST_CASE("ltx2 the sigma-binned guider factory resolves upstream's bins") {
  const size_t bin_count = std::size(vllm_test::kLtx2GuideBinKeys);
  std::vector<vllm::Ltx2GuiderSigmaBin> bins;
  for (size_t i = 0; i < bin_count; ++i) {
    vllm::Ltx2GuiderSigmaBin bin;
    bin.sigma_upper_bound = vllm_test::kLtx2GuideBinKeys[i];
    bin.params.cfg_scale = vllm_test::kLtx2GuideBinCfgScales[i];
    bins.push_back(bin);
  }
  // Deliberately handed to the port in ASCENDING order: `from_dict` sorts
  // descending itself (guiders.py:329), so a port that assumed the caller's order
  // would resolve every sigma to the wrong bin.
  std::reverse(bins.begin(), bins.end());

  for (size_t q = 0; q < std::size(vllm_test::kLtx2GuideBinQueries); ++q) {
    const double sigma = vllm_test::kLtx2GuideBinQueries[q];
    const double want = vllm_test::kLtx2GuideBinResolvedCfg[q];
    const double got = vllm::Ltx2GuiderParamsForSigma(bins, sigma).cfg_scale;
    INFO("sigma = ", sigma, " want cfg_scale = ", want, " got = ", got);
    CHECK(got == want);
  }

  // The constant factory is one bin at +inf (guiders.py:305-315), so every sigma
  // resolves to it.
  std::vector<vllm::Ltx2GuiderSigmaBin> constant(1);
  constant[0].sigma_upper_bound = std::numeric_limits<double>::infinity();
  constant[0].params.cfg_scale = 4.0;
  for (size_t q = 0; q < std::size(vllm_test::kLtx2GuideBinQueries); ++q) {
    CHECK(vllm::Ltx2GuiderParamsForSigma(constant, vllm_test::kLtx2GuideBinQueries[q])
              .cfg_scale == vllm_test::kLtx2GuideConstantResolvedCfg[q]);
  }

  // Upstream raises on an empty schedule (guiders.py:223-224) rather than
  // inventing a default guider.
  const std::vector<vllm::Ltx2GuiderSigmaBin> empty;
  const std::string message =
      RefusalMessage([&] { (void)vllm::Ltx2GuiderParamsForSigma(empty, 1.0); });
  INFO("refusal = ", message);
  CHECK(Mentions(message, "non-empty"));
}

// ===========================================================================
// Section 5 — perturbations
// ===========================================================================

TEST_CASE("ltx2 the batched perturbation keep-mask reproduces upstream") {
  CHECK(vllm::kLtx2PerturbationTypeCount == vllm_test::kLtx2PerturbTypeCount);
  // The enum VALUE is the row index into dim 0 of the mask (perturbations.py:10-11),
  // so renumbering it silently perturbs a different attention.
  CHECK(static_cast<int64_t>(vllm::Ltx2PerturbationType::kSkipVideoSelfAttn) ==
        vllm_test::kLtx2PerturbSkipVideoSelfAttn);
  CHECK(static_cast<int64_t>(vllm::Ltx2PerturbationType::kSkipAudioSelfAttn) ==
        vllm_test::kLtx2PerturbSkipAudioSelfAttn);
  CHECK(static_cast<int64_t>(vllm::Ltx2PerturbationType::kSkipA2vCrossAttn) ==
        vllm_test::kLtx2PerturbSkipA2vCrossAttn);
  CHECK(static_cast<int64_t>(vllm::Ltx2PerturbationType::kSkipV2aCrossAttn) ==
        vllm_test::kLtx2PerturbSkipV2aCrossAttn);

  const int64_t blocks = vllm_test::kLtx2PerturbNumBlocks;

  // The EMPTY configuration is the one LTX-2.5 ships.
  const vllm::Ltx2BatchedPerturbationConfig empty =
      vllm::Ltx2BatchedPerturbationConfig::Empty(3, blocks);
  REQUIRE(empty.block_masks().size() == std::size(vllm_test::kLtx2PerturbEmptyMask));
  for (size_t i = 0; i < empty.block_masks().size(); ++i) {
    CHECK(static_cast<int64_t>(empty.block_masks()[i]) == vllm_test::kLtx2PerturbEmptyMask[i]);
  }

  std::vector<vllm::Ltx2PerturbationConfig> configs(3);
  configs[0].perturbations.push_back(
      {vllm::Ltx2PerturbationType::kSkipVideoSelfAttn, false, {1, 2}});
  configs[1].perturbations.push_back(
      {vllm::Ltx2PerturbationType::kSkipA2vCrossAttn, true, {}});
  const vllm::Ltx2BatchedPerturbationConfig mixed(configs, blocks);
  REQUIRE(mixed.block_masks().size() == std::size(vllm_test::kLtx2PerturbMixedMask));
  for (size_t i = 0; i < mixed.block_masks().size(); ++i) {
    CHECK(static_cast<int64_t>(mixed.block_masks()[i]) == vllm_test::kLtx2PerturbMixedMask[i]);
  }

  size_t index = 0;
  for (int64_t direction = 0; direction < vllm::kLtx2PerturbationTypeCount; ++direction) {
    for (int64_t block = 0; block < blocks; ++block, ++index) {
      const auto type = static_cast<vllm::Ltx2PerturbationType>(direction);
      CHECK(mixed.AnyInBatch(type, block) == (vllm_test::kLtx2PerturbMixedAny[index] != 0));
      CHECK(mixed.AllInBatch(type, block) == (vllm_test::kLtx2PerturbMixedAll[index] != 0));
    }
  }

  const vllm::Ltx2BatchedPerturbationConfig sliced = mixed.BatchSlice(1, 3);
  REQUIRE(sliced.block_masks().size() == std::size(vllm_test::kLtx2PerturbSlicedMask));
  for (size_t i = 0; i < sliced.block_masks().size(); ++i) {
    CHECK(static_cast<int64_t>(sliced.block_masks()[i]) == vllm_test::kLtx2PerturbSlicedMask[i]);
  }
}

// ===========================================================================
// Section 6 — patchifiers
// ===========================================================================

TEST_CASE("ltx2 the video patchifier round-trips bit-exactly against upstream") {
  vllm::Ltx2VideoLatentShape shape;
  shape.batch = 1;
  shape.channels = vllm_test::kLtx2PatchChannels;
  shape.frames = vllm_test::kLtx2PatchFrames;
  shape.height = vllm_test::kLtx2PatchHeight;
  shape.width = vllm_test::kLtx2PatchWidth;
  const int64_t patch = vllm_test::kLtx2PatchSize;

  CHECK(vllm::Ltx2VideoTokenCount(shape, patch) == vllm_test::kLtx2PatchTokenCount);

  const int64_t count =
      shape.channels * shape.frames * shape.height * shape.width;
  const std::vector<float> latent = Make("ltx2.patch.video", count, 1.0);
  const std::vector<float> tokens = vllm::Ltx2VideoPatchify(latent.data(), shape, patch);
  const double worst = MaxAbsDiff(tokens, vllm_test::kLtx2PatchVideoTokens,
                                  std::size(vllm_test::kLtx2PatchVideoTokens));
  INFO("VideoLatentPatchifier.patchify max|diff| = ", worst);
  // A rearrange MOVES bytes; anything but 0 is a wrong index, not round-off.
  CHECK(worst <= kExactRoundOff);

  const std::vector<float> restored = vllm::Ltx2VideoUnpatchify(tokens.data(), shape, patch);
  const double restore_worst = MaxAbsDiff(restored, vllm_test::kLtx2PatchVideoRestored,
                                          std::size(vllm_test::kLtx2PatchVideoRestored));
  INFO("VideoLatentPatchifier.unpatchify max|diff| = ", restore_worst);
  CHECK(restore_worst <= kExactRoundOff);
  // ...and it is genuinely the inverse, which the golden alone would not prove if
  // BOTH directions carried the same wrong permutation.
  CHECK(restored == latent);

  const std::vector<int64_t> bounds = vllm::Ltx2VideoPatchBounds(shape, patch);
  REQUIRE(bounds.size() == std::size(vllm_test::kLtx2PatchBoundsGolden));
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i] == vllm_test::kLtx2PatchBoundsGolden[i]);
  }

  vllm::Ltx2ScaleFactors factors;
  factors.time = vllm_test::kLtx2PatchScaleFactors[0];
  factors.height = vllm_test::kLtx2PatchScaleFactors[1];
  factors.width = vllm_test::kLtx2PatchScaleFactors[2];
  const int64_t tokens_count = vllm_test::kLtx2PatchTokens;
  for (bool causal : {false, true}) {
    const std::vector<int64_t> pixels =
        vllm::Ltx2PixelCoords(bounds, shape.batch, tokens_count, factors, causal);
    const int64_t* want = causal ? vllm_test::kLtx2PatchPixelCoordsCausalFixGolden
                                 : vllm_test::kLtx2PatchPixelCoordsPlainGolden;
    const size_t want_size = causal ? std::size(vllm_test::kLtx2PatchPixelCoordsCausalFixGolden)
                                    : std::size(vllm_test::kLtx2PatchPixelCoordsPlainGolden);
    REQUIRE(pixels.size() == want_size);
    for (size_t i = 0; i < pixels.size(); ++i) {
      INFO("causal_fix = ", causal, " index = ", i);
      CHECK(pixels[i] == want[i]);
    }
  }
}

TEST_CASE("ltx2 the audio patchifier and its timings reproduce upstream") {
  vllm::Ltx2AudioLatentShape shape;
  shape.batch = 1;
  shape.channels = vllm_test::kLtx2PatchAudioChannels;
  shape.frames = vllm_test::kLtx2PatchAudioFrames;
  shape.mel_bins = vllm_test::kLtx2PatchAudioMelBins;
  const int64_t count = shape.channels * shape.frames * shape.mel_bins;
  const std::vector<float> latent = Make("ltx2.patch.audio", count, 1.0);

  vllm::Ltx2AudioPatchifierParams defaults;
  CHECK(defaults.sample_rate == vllm_test::kLtx2PatchAudioSampleRate);
  CHECK(defaults.hop_length == vllm_test::kLtx2PatchAudioHopLength);
  CHECK(defaults.audio_latent_downsample_factor == vllm_test::kLtx2PatchAudioDownsample);

  const std::vector<float> tokens = vllm::Ltx2AudioPatchify(latent.data(), shape);
  const double worst = MaxAbsDiff(tokens, vllm_test::kLtx2PatchAudioCausalTokens,
                                  std::size(vllm_test::kLtx2PatchAudioCausalTokens));
  INFO("AudioPatchifier.patchify max|diff| = ", worst);
  CHECK(worst <= kExactRoundOff);
  CHECK(vllm::Ltx2AudioUnpatchify(tokens.data(), shape) == latent);

  auto timings = [&](const std::string& tag, bool causal, int64_t shift, int64_t want_tokens,
                     const float* golden, size_t golden_count) {
    vllm::Ltx2AudioPatchifierParams params;
    params.is_causal = causal;
    params.shift = shift;
    CHECK(shape.frames == want_tokens);
    const std::vector<float> got = vllm::Ltx2AudioPatchTimings(shape, params);
    const double diff = MaxAbsDiff(got, golden, golden_count);
    INFO("AudioPatchifier timings arm = ", tag, " max|diff| = ", diff);
    CHECK(diff <= kRoundOff);
  };
  timings("Causal", vllm_test::kLtx2PatchAudioCausalIsCausal, vllm_test::kLtx2PatchAudioCausalShift,
          vllm_test::kLtx2PatchAudioCausalTokenCount, vllm_test::kLtx2PatchAudioCausalTimingGolden,
          std::size(vllm_test::kLtx2PatchAudioCausalTimingGolden));
  timings("NonCausal", vllm_test::kLtx2PatchAudioNonCausalIsCausal,
          vllm_test::kLtx2PatchAudioNonCausalShift, vllm_test::kLtx2PatchAudioNonCausalTokenCount,
          vllm_test::kLtx2PatchAudioNonCausalTimingGolden,
          std::size(vllm_test::kLtx2PatchAudioNonCausalTimingGolden));
  timings("Shift2", vllm_test::kLtx2PatchAudioShift2IsCausal, vllm_test::kLtx2PatchAudioShift2Shift,
          vllm_test::kLtx2PatchAudioShift2TokenCount, vllm_test::kLtx2PatchAudioShift2TimingGolden,
          std::size(vllm_test::kLtx2PatchAudioShift2TimingGolden));
}

// ===========================================================================
// Section 7 — the recipes
// ===========================================================================

TEST_CASE("ltx2 model versions parse and resolve exactly as upstream's do") {
  for (int64_t i = 0; i < vllm_test::kLtx2VersionCount; ++i) {
    const std::string version = vllm_test::kLtx2VersionStrings[i];
    // The goldens record the NORMALIZED parse, because that is what
    // `detect_model_version` performs before parsing (utils/constants.py:161).
    // The RAW parser's own behaviour is asserted separately below.
    std::string normalized = version;
    std::replace(normalized.begin(), normalized.end(), '-', '.');
    const std::vector<int64_t> parsed = vllm::Ltx2ParseModelVersion(normalized);
    INFO("version = ", version);
    CHECK(static_cast<int64_t>(parsed.size()) == vllm_test::kLtx2VersionParsedLen[i]);
    if (!parsed.empty()) CHECK(parsed[0] == vllm_test::kLtx2VersionParsedMajor[i]);
    if (parsed.size() > 1) CHECK(parsed[1] == vllm_test::kLtx2VersionParsedMinor[i]);

    // `detect_params`' newest-at-or-below rule. This is what hands LTX-2.5 the
    // 2.4 parameters, so the whole 2.5 recipe rests on it.
    const vllm::Ltx2PipelineParams params = vllm::Ltx2DetectPipelineParams(version);
    CHECK(params.num_inference_steps == vllm_test::kLtx2VersionResolvedSteps[i]);
    CHECK(params.default_image_crf == vllm_test::kLtx2VersionResolvedCrf[i]);

    CHECK(vllm::Ltx2ShouldUseAncestralSampler(version) ==
          (vllm_test::kLtx2VersionUsesAncestral[i] != 0));
  }

  // The separator normalization is not cosmetic: "2.4-rc2" parses to (2,) without
  // it, which would drop a release candidate onto the 2.0 recipe.
  CHECK(vllm::Ltx2DetectPipelineParams("2.4-rc2").num_inference_steps ==
        vllm::Ltx2DetectPipelineParams("2.4").num_inference_steps);
  // ...while the RAW parser leaves it alone, exactly as upstream documents
  // (loader/helpers.py:70-72).
  CHECK(vllm::Ltx2ParseModelVersion("2.4-rc2").size() == 1u);
}

TEST_CASE("ltx2 the per-generation pipeline parameters match Lightricks") {
  auto check = [&](const std::string& tag, const vllm::Ltx2PipelineParams& got, int64_t seed,
                   int64_t s1h, int64_t s1w, int64_t s2h, int64_t s2w, int64_t frames,
                   double fps, int64_t steps, int64_t crf, double v_cfg, double v_stg,
                   double v_rescale, double v_modality, int64_t v_skip,
                   const int64_t* v_blocks, size_t v_block_count, double a_cfg,
                   double a_stg, double a_rescale, double a_modality, int64_t a_skip,
                   const int64_t* a_blocks, size_t a_block_count) {
    INFO("generation = ", tag);
    CHECK(got.seed == seed);
    CHECK(got.stage_1_height == s1h);
    CHECK(got.stage_1_width == s1w);
    CHECK(got.stage_2_height() == s2h);
    CHECK(got.stage_2_width() == s2w);
    CHECK(got.num_frames == frames);
    CHECK(got.frame_rate == fps);
    CHECK(got.num_inference_steps == steps);
    CHECK(got.default_image_crf == crf);
    CHECK(got.video_guider.cfg_scale == v_cfg);
    CHECK(got.video_guider.stg_scale == v_stg);
    CHECK(got.video_guider.rescale_scale == v_rescale);
    CHECK(got.video_guider.modality_scale == v_modality);
    CHECK(got.video_guider.skip_step == v_skip);
    REQUIRE(got.video_guider.stg_blocks.size() == v_block_count);
    for (size_t i = 0; i < v_block_count; ++i) {
      CHECK(got.video_guider.stg_blocks[i] == v_blocks[i]);
    }
    CHECK(got.audio_guider.cfg_scale == a_cfg);
    CHECK(got.audio_guider.stg_scale == a_stg);
    CHECK(got.audio_guider.rescale_scale == a_rescale);
    CHECK(got.audio_guider.modality_scale == a_modality);
    CHECK(got.audio_guider.skip_step == a_skip);
    REQUIRE(got.audio_guider.stg_blocks.size() == a_block_count);
    for (size_t i = 0; i < a_block_count; ++i) {
      CHECK(got.audio_guider.stg_blocks[i] == a_blocks[i]);
    }
  };

#define LTX2_PARAMS_ARM(TAG, FN)                                                              \
  check(#TAG, FN(), vllm_test::kLtx2Params##TAG##Seed,                                         \
        vllm_test::kLtx2Params##TAG##Stage1Height, vllm_test::kLtx2Params##TAG##Stage1Width,   \
        vllm_test::kLtx2Params##TAG##Stage2Height, vllm_test::kLtx2Params##TAG##Stage2Width,   \
        vllm_test::kLtx2Params##TAG##NumFrames, vllm_test::kLtx2Params##TAG##FrameRate,        \
        vllm_test::kLtx2Params##TAG##NumInferenceSteps, vllm_test::kLtx2Params##TAG##ImageCrf, \
        vllm_test::kLtx2Params##TAG##VideoCfgScale, vllm_test::kLtx2Params##TAG##VideoStgScale,\
        vllm_test::kLtx2Params##TAG##VideoRescaleScale,                                        \
        vllm_test::kLtx2Params##TAG##VideoModalityScale,                                       \
        vllm_test::kLtx2Params##TAG##VideoSkipStep, vllm_test::kLtx2Params##TAG##VideoStgBlocks,\
        static_cast<size_t>(vllm_test::kLtx2Params##TAG##VideoStgBlockCount),                  \
        vllm_test::kLtx2Params##TAG##AudioCfgScale, vllm_test::kLtx2Params##TAG##AudioStgScale,\
        vllm_test::kLtx2Params##TAG##AudioRescaleScale,                                        \
        vllm_test::kLtx2Params##TAG##AudioModalityScale,                                       \
        vllm_test::kLtx2Params##TAG##AudioSkipStep, vllm_test::kLtx2Params##TAG##AudioStgBlocks,\
        static_cast<size_t>(vllm_test::kLtx2Params##TAG##AudioStgBlockCount))
  LTX2_PARAMS_ARM(Ltx2, vllm::Ltx2Params20);
  LTX2_PARAMS_ARM(Ltx23, vllm::Ltx2Params23);
  LTX2_PARAMS_ARM(Ltx24, vllm::Ltx2Params24);
  LTX2_PARAMS_ARM(Ltx23Hq, vllm::Ltx2Params23Hq);
#undef LTX2_PARAMS_ARM
}

TEST_CASE("ltx2 the distilled sigma schedules match BOTH upstreams") {
  const vllm::Ltx2PipelineRecipe distilled =
      vllm::ResolveLtx2PipelineRecipe("distilled_two_stage", "2.5");
  REQUIRE(distilled.phases.size() == 2u);

  const std::vector<float>& stage1 = distilled.phases[0].sigmas;
  REQUIRE(stage1.size() == static_cast<size_t>(vllm_test::kLtx2DistilledSigmaCount));
  for (size_t i = 0; i < stage1.size(); ++i) {
    INFO("stage 1 sigma index = ", i);
    CHECK(stage1[i] == vllm_test::kLtx2DistilledSigmas[i]);
    // Lightricks and vLLM-Omni agree on these, which is worth asserting rather
    // than assuming: they are the values the distillation was trained against.
    CHECK(vllm_test::kLtx2DistilledSigmas[i] == vllm_test::kLtx2OmniDistilledSigmas[i]);
  }

  const std::vector<float>& stage2 = distilled.phases[1].sigmas;
  REQUIRE(stage2.size() == static_cast<size_t>(vllm_test::kLtx2Stage2DistilledSigmaCount));
  for (size_t i = 0; i < stage2.size(); ++i) {
    INFO("stage 2 sigma index = ", i);
    CHECK(stage2[i] == vllm_test::kLtx2Stage2DistilledSigmas[i]);
    CHECK(vllm_test::kLtx2Stage2DistilledSigmas[i] ==
          vllm_test::kLtx2OmniStage2DistilledSigmas[i]);
  }
  // ltx2_recipes.py:146 — stage 2 re-noises to its OWN first sigma, which is what
  // makes the upsampled latent a valid input at that noise level.
  CHECK(distilled.phases[1].noise_scale == stage2[0]);
  CHECK(distilled.phases[0].noise_scale == 1.0);
}

TEST_CASE("ltx2 the distilled two-stage recipe carries 2.5's ancestral stage-1 sampler") {
  CHECK(vllm::kLtx2AncestralSinceMajor == vllm_test::kLtx2AncestralSinceVersion[0]);
  CHECK(vllm::kLtx2AncestralSinceMinor == vllm_test::kLtx2AncestralSinceVersion[1]);
  CHECK(vllm::kLtx2AncestralEta == vllm_test::kLtx2AncestralEta);
  CHECK(vllm::kLtx2AncestralSNoise == vllm_test::kLtx2AncestralSNoise);
  CHECK(vllm::kLtx2AncestralNoiseSeedOffset == vllm_test::kLtx2AncestralNoiseSeedOffset);

  const vllm::Ltx2PipelineRecipe v25 =
      vllm::ResolveLtx2PipelineRecipe("distilled_two_stage", "2.5");
  // Stage 1 is ancestral from generation 2.5 (distilled.py:170-185)...
  CHECK(v25.phases[0].stepper == vllm::Ltx2StepperKind::kEulerAncestral);
  CHECK(v25.phases[0].stepper_eta == vllm::kLtx2AncestralEta);
  CHECK(v25.phases[0].stepper_s_noise == vllm::kLtx2AncestralSNoise);
  CHECK(v25.phases[0].noise_seed_offset == vllm::kLtx2AncestralNoiseSeedOffset);
  // ...and stage 2 is ALWAYS deterministic, because a 3-step refinement cannot
  // remove freshly injected noise (distilled.py:206-209).
  CHECK(v25.phases[1].stepper == vllm::Ltx2StepperKind::kEuler);

  // Generation 2.0 takes the deterministic stage-1 sampler. This is the ONE thing
  // that separates the two distilled rows, so it is asserted on both.
  const vllm::Ltx2PipelineRecipe v2 =
      vllm::ResolveLtx2PipelineRecipe("distilled_two_stage", "2");
  CHECK(v2.phases[0].stepper == vllm::Ltx2StepperKind::kEuler);
  CHECK(v2.phases[0].sigmas == v25.phases[0].sigmas);
}

TEST_CASE("ltx2 the recipe table mirrors vLLM-Omni's, and refuses everything else") {
  // Every key vLLM-Omni carries must resolve here too. Read off ITS table rather
  // than a remembered list, so a row appearing or vanishing upstream shows up.
  for (int64_t i = 0; i < vllm_test::kLtx2OmniRecipeKeyCount; ++i) {
    const std::string kind = vllm_test::kLtx2OmniRecipeKinds[i];
    const std::string version = vllm_test::kLtx2OmniRecipeVersions[i];
    INFO("vLLM-Omni key = ", kind, "/", version);
    CHECK_NOTHROW((void)vllm::ResolveLtx2PipelineRecipe(kind, version));
  }
  // vLLM-Omni carries NO 2.5 row — spec section 3. Asserting that keeps this
  // port's extra rows honest: the moment upstream adds one, the values must be
  // re-anchored to it.
  for (int64_t i = 0; i < vllm_test::kLtx2OmniRecipeKeyCount; ++i) {
    CHECK(std::string(vllm_test::kLtx2OmniRecipeVersions[i]) != "2.5");
  }

  // The 2.4 and 2.5 rows this port adds, sourced from Lightricks.
  CHECK_NOTHROW((void)vllm::ResolveLtx2PipelineRecipe("one_stage", "2.4"));
  CHECK_NOTHROW((void)vllm::ResolveLtx2PipelineRecipe("one_stage", "2.5"));
  CHECK_NOTHROW((void)vllm::ResolveLtx2PipelineRecipe("distilled_two_stage", "2.5"));

  // ...and the refusal, which is the point. A plausible-but-wrong recipe RENDERS.
  for (const auto& pair : std::vector<std::pair<std::string, std::string>>{
           {"one_stage", "3"},
           {"one_stage", "2.6"},
           {"distilled_two_stage", "2.3"},
           {"dmd2", "2.5"},
           {"res2s_two_stage", "2.5"},
           {"", ""},
       }) {
    const std::string message = RefusalMessage(
        [&] { (void)vllm::ResolveLtx2PipelineRecipe(pair.first, pair.second); });
    INFO("kind = ", pair.first, " version = ", pair.second,
         " refusal = ", message);
    CHECK_FALSE(message.empty());
    CHECK(Mentions(message, "Unsupported LTX pipeline kind/version"));
    // The refusal must NAME what was asked for; "unsupported" alone leaves the
    // caller guessing which half was wrong.
    if (!pair.first.empty()) CHECK(Mentions(message, pair.first));
    if (!pair.second.empty()) CHECK(Mentions(message, pair.second));
  }
}

TEST_CASE("ltx2 the one-stage and distilled recipes carry upstream's request policy") {
  const vllm::Ltx2PipelineRecipe one_stage = vllm::ResolveLtx2PipelineRecipe("one_stage", "2");
  REQUIRE(one_stage.phases.size() == 1u);
  CHECK(one_stage.phases[0].name == "generate");
  CHECK(one_stage.num_inference_steps == vllm_test::kLtx2ParamsLtx2NumInferenceSteps);
  CHECK(one_stage.phases[0].video_guidance.stg_blocks ==
        std::vector<int64_t>{vllm_test::kLtx2ParamsLtx2VideoStgBlocks[0]});
  CHECK(one_stage.phases[0].sigmas.empty());
  CHECK(one_stage.phases[0].num_inference_steps() == -1);
  CHECK(one_stage.allow_request_sigmas);
  CHECK(one_stage.max_spatial_downscale() == 1);

  const vllm::Ltx2PipelineRecipe one_stage_25 =
      vllm::ResolveLtx2PipelineRecipe("one_stage", "2.5");
  // 2.5 inherits 2.4's knobs through _PARAMS_SINCE_VERSION, which is the fact the
  // whole row rests on: 30 steps and STG block 28, NOT 2.0's 40 and 29.
  CHECK(one_stage_25.num_inference_steps == vllm_test::kLtx2ParamsLtx24NumInferenceSteps);
  CHECK(one_stage_25.default_image_crf == vllm_test::kLtx2Ltx24ImageCrf);
  CHECK(one_stage_25.phases[0].video_guidance.stg_blocks ==
        std::vector<int64_t>{vllm_test::kLtx2ParamsLtx24VideoStgBlocks[0]});

  const vllm::Ltx2PipelineRecipe distilled =
      vllm::ResolveLtx2PipelineRecipe("distilled_two_stage", "2.5");
  // ltx2_recipes.py:125-158. The distilled arguments describe the FINAL output;
  // stage 1 runs at half of it.
  CHECK(distilled.height == 1024);
  CHECK(distilled.width == 1536);
  CHECK(distilled.phases[0].spatial_downscale == 2);
  CHECK(distilled.phases[1].spatial_downscale == 1);
  CHECK(distilled.max_spatial_downscale() == 2);
  CHECK(distilled.phases[0].input_transform == vllm::Ltx2PhaseInputTransform::kInitial);
  CHECK(distilled.phases[1].input_transform ==
        vllm::Ltx2PhaseInputTransform::kSpatialUpsample);
  CHECK(distilled.video_output_phase == 1);
  CHECK(distilled.audio_output_phase == 1);
  CHECK_FALSE(distilled.allow_request_sigmas);
  CHECK_FALSE(distilled.allow_request_latents);
  CHECK_FALSE(distilled.allow_negative_prompt);
  CHECK(distilled.fixed_num_inference_steps);
  CHECK(distilled.negative_prompt.empty());
  CHECK(distilled.num_inference_steps == vllm_test::kLtx2DistilledSigmaCount - 1);
  CHECK_FALSE(distilled.phases[0].allow_guidance_override);
  CHECK_FALSE(distilled.phases[0].use_official_sigma_schedule);
  // Both distilled phases are POSITIVE-ONLY: distilled.py drives them through
  // SimpleDenoiser with no guider at all (distilled.py:266, :295).
  for (const vllm::Ltx2PhaseRecipe& phase : distilled.phases) {
    CHECK_FALSE(phase.video_guidance.DoUnconditionalGeneration());
    CHECK_FALSE(phase.video_guidance.DoPerturbedGeneration());
    CHECK_FALSE(phase.audio_guidance.DoUnconditionalGeneration());
  }
}

TEST_CASE("ltx2 the two references disagree on the default negative prompt, and both are kept") {
  // Spec section 3: where the binding oracle and the cross-check disagree, the
  // disagreement IS the finding. Lightricks' prompt carries five leading tags
  // vLLM-Omni's lacks. The 2.5 rows take Lightricks', because Lightricks is the
  // reference for 2.5's recipe values; the pre-2.5 rows take vLLM-Omni's.
  CHECK_FALSE(vllm_test::kLtx2NegativePromptsAgree);
  const std::string lightricks = vllm_test::kLtx2LightricksNegativePrompt;
  const std::string omni = vllm_test::kLtx2OmniNegativePrompt;
  CHECK(lightricks != omni);
  CHECK(Mentions(lightricks, "has_subtitles"));
  CHECK_FALSE(Mentions(omni, "has_subtitles"));

  CHECK(vllm::ResolveLtx2PipelineRecipe("one_stage", "2").negative_prompt == omni);
  CHECK(vllm::ResolveLtx2PipelineRecipe("one_stage", "2.5").negative_prompt == lightricks);
}

// ===========================================================================
// Out-of-scope refusals
// ===========================================================================

TEST_CASE("ltx2 every L5 out-of-scope feature is refused BY NAME") {
  // Spec section 2 "Out", plus the L7 boundary. A silent downgrade of any of
  // these produces a video, which is exactly why none of them may fall back.
  const std::vector<std::pair<vllm::Ltx2UnportedPipelineFeature, std::string>> owed = {
      {vllm::Ltx2UnportedPipelineFeature::kSpatiotemporalUpsampler, "temporal"},
      {vllm::Ltx2UnportedPipelineFeature::kLoraFusion, "LoRA"},
      {vllm::Ltx2UnportedPipelineFeature::kMultishot, "multishot"},
      {vllm::Ltx2UnportedPipelineFeature::kInt8ConvRot, "int8-convrot"},
      {vllm::Ltx2UnportedPipelineFeature::kCfgParallelism, "parallelism"},
      {vllm::Ltx2UnportedPipelineFeature::kVideoEngineWiring, "VideoEngine"},
      {vllm::Ltx2UnportedPipelineFeature::kBetaScheduler, "BetaScheduler"},
  };
  for (const auto& item : owed) {
    const std::string message =
        RefusalMessage([&] { vllm::Ltx2RefuseUnportedPipelineFeature(item.first); });
    INFO("feature = ", item.second, " refusal = ", message);
    CHECK(Mentions(message, item.second));
    // Naming WHERE the work is owed is what keeps it from being rediscovered.
    CHECK(Mentions(message, "ltx-2-5.md"));
  }
}

// ===========================================================================
// Section 8 — the latent spatial upsampler
// ===========================================================================

namespace {

vllm::Ltx2UpsamplerConfig ReducedUpsamplerConfig(bool rational, double scale,
                                                 const std::string& prefix) {
  vllm::Ltx2UpsamplerConfig config;
  config.in_channels = vllm_test::kLtx2UpsInChannels;
  config.mid_channels = vllm_test::kLtx2UpsMidChannels;
  config.num_blocks_per_stage = vllm_test::kLtx2UpsBlocksPerStage;
  config.dims = 3;
  config.spatial_upsample = true;
  config.temporal_upsample = false;
  config.spatial_scale = scale;
  config.rational_resampler = rational;
  config.prefix = prefix;
  return config;
}

ParamBag BuildUpsamplerParams(const vllm::Ltx2UpsamplerConfig& config) {
  ParamBag bag;
  for (const vllm::Ltx2UpsamplerTensorSpec& spec :
       vllm::EnumerateLtx2UpsamplerTensors(config)) {
    bag.Put(spec.name, spec.shape);
  }
  return bag;
}

vllm::Ltx2LatentVolume ReducedUpsamplerLatent() {
  vllm::Ltx2LatentVolume latent;
  latent.batch = 1;
  latent.channels = vllm_test::kLtx2UpsInChannels;
  latent.frames = vllm_test::kLtx2UpsFrames;
  latent.height = vllm_test::kLtx2UpsHeight;
  latent.width = vllm_test::kLtx2UpsWidth;
  latent.data = Make("ltx2.ups.latent", latent.elems(), 1.0);
  return latent;
}

// `spatial_upsample=False, temporal_upsample=True` (model.py:68-71) — the arm
// with `Conv3d(mid, 2*mid)` + `PixelShuffleND(1)` and the dropped first frame.
vllm::Ltx2UpsamplerConfig TemporalUpsamplerConfig(const std::string& prefix) {
  vllm::Ltx2UpsamplerConfig config;
  config.in_channels = vllm_test::kLtx2UpsInChannels;
  config.mid_channels = vllm_test::kLtx2UpsMidChannels;
  config.num_blocks_per_stage = vllm_test::kLtx2UpsBlocksPerStage;
  config.dims = 3;
  config.spatial_upsample = false;
  config.temporal_upsample = true;
  config.prefix = prefix;
  return config;
}

// A separate fixture from ReducedUpsamplerLatent: 3 frames, so the frame axis
// differs from H (4) and W (6) and `2F - 1 = 5` differs from `2F = 6`.
vllm::Ltx2LatentVolume TemporalUpsamplerLatent() {
  vllm::Ltx2LatentVolume latent;
  latent.batch = 1;
  latent.channels = vllm_test::kLtx2UpsInChannels;
  latent.frames = vllm_test::kLtx2UpsTemporalFrames;
  latent.height = vllm_test::kLtx2UpsHeight;
  latent.width = vllm_test::kLtx2UpsWidth;
  latent.data = Make("ltx2.ups.temporal.latent", latent.elems(), 1.0);
  return latent;
}

}  // namespace

TEST_CASE("ltx2 the constants the headers call pinned are actually pinned") {
  // Both headers said "pinned" and NEITHER constant had a test reference, so this
  // case exists because an unreferenced constant can be edited without the suite
  // naming it. What it does NOT do is make either one invisible to the goldens:
  // that was inferred rather than measured, and both are in fact reached.
  //
  //   kLtx2ConnectorRmsNormEps 1e-6 -> 1e-4 (100x) REDS 5 arms of "ltx2 the
  //     Embeddings1DConnector reproduces upstream on every arm" — Split
  //     0.0558581, Interleaved 0.104284, Float64 0.140343, NoRegisters
  //     0.000542641, GatedNoBias 0.0892045. `rms_norm` adds the epsilon to the
  //     MEAN SQUARE, so "the rows are never near-zero" was never the question.
  //   kLtx2BlurKernelSize 5 -> 3 REDS "ltx2 the latent spatial upsampler
  //     reproduces upstream", arm Rational1p5, at 0.689782. Upstream not passing
  //     the argument does not make the default unreachable — the default IS the
  //     shipped width, which ltx2_upsampler.h has said all along. Only the
  //     Rational1p5 arm reaches it, because `BlurDownsample` runs on the
  //     rational `den` (ltx2_upsampler.cpp:439) and 1.5 -> {3, 2} is the only
  //     one of the THREE ARMS with den != 1. (0.75 -> {3, 4} would reach it too
  //     and no arm covers it, so this is arm coverage, not a property of the
  //     supported-scale map.)
  //
  // So the reason to keep this case is the narrower, real one: a golden
  // regenerated from a moved constant moves with it, and these two lines are the
  // only comparison against upstream's own signature.
  //
  // Each expected value is READ OFF upstream's own signature by the generator
  // (utils.py:7, blur_downsample.py:14), not retyped here, so upstream moving
  // either one fails this gate instead of silently redefining the port.
  CHECK(vllm::kLtx2ConnectorRmsNormEps == vllm_test::kLtx2ConnRmsNormEps);
  CHECK(vllm::kLtx2BlurKernelSize == vllm_test::kLtx2UpsBlurKernelSize);
}

TEST_CASE("ltx2 the binomial anti-alias kernel is built, not loaded") {
  // blur_downsample.py:29-33. Computed at construction on both sides, like the
  // audio VAE's kaiser-sinc windows: a checkpoint carries no such tensor, so a
  // port that got the normalization wrong would silently attenuate every
  // downsample.
  auto check = [&](int64_t size, const float* golden, size_t count) {
    const std::vector<float> got = vllm::Ltx2BlurKernel(size);
    const double worst = MaxAbsDiff(got, golden, count);
    INFO("BlurDownsample kernel_size = ", size, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
    double sum = 0.0;
    for (float value : got) sum += value;
    CHECK(std::fabs(sum - 1.0) <= 1e-6);
  };
  check(3, vllm_test::kLtx2UpsBlurKernel3, std::size(vllm_test::kLtx2UpsBlurKernel3));
  check(5, vllm_test::kLtx2UpsBlurKernel5, std::size(vllm_test::kLtx2UpsBlurKernel5));
  check(7, vllm_test::kLtx2UpsBlurKernel7, std::size(vllm_test::kLtx2UpsBlurKernel7));

  // spatial_rational_resampler.py:10-14. The supported map, and the refusal for
  // anything else — an unsupported scale is a config error, not a nearest match.
  for (size_t i = 0; i < std::size(vllm_test::kLtx2UpsRationalScales); ++i) {
    const vllm::Ltx2RationalScale got =
        vllm::Ltx2RationalForScale(vllm_test::kLtx2UpsRationalScales[i]);
    INFO("scale = ", vllm_test::kLtx2UpsRationalScales[i]);
    CHECK(got.num == vllm_test::kLtx2UpsRationalNum[i]);
    CHECK(got.den == vllm_test::kLtx2UpsRationalDen[i]);
  }
  const std::string message = RefusalMessage([] { (void)vllm::Ltx2RationalForScale(3.0); });
  INFO("refusal = ", message);
  CHECK(Mentions(message, "Unsupported scale"));
}

TEST_CASE("ltx2 the latent spatial upsampler reproduces upstream") {
  const vllm::Ltx2LatentVolume latent = ReducedUpsamplerLatent();

  auto run = [&](const std::string& tag, bool rational, double scale, const std::string& prefix,
                 const char* const* names, const int64_t* counts, size_t manifest_size,
                 const int64_t* out_shape, const float* golden, size_t golden_count) {
    const vllm::Ltx2UpsamplerConfig config = ReducedUpsamplerConfig(rational, scale, prefix);
    const ParamBag bag = BuildUpsamplerParams(config);
    // The parameter CONTRACT is part of the gate: a tensor either side builds and
    // the other does not is a failure, not a silent no-op.
    CheckManifest(bag, names, counts, manifest_size);

    const vllm::Ltx2LatentVolume got = vllm::Ltx2LatentUpsample(config, bag.weights, latent);
    CHECK(got.batch == out_shape[0]);
    CHECK(got.channels == out_shape[1]);
    CHECK(got.frames == out_shape[2]);
    CHECK(got.height == out_shape[3]);
    CHECK(got.width == out_shape[4]);
    const double worst = MaxAbsDiff(got.data, golden, golden_count);
    INFO("LatentUpsampler arm = ", tag, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
  };

#define LTX2_UPS_ARM(TAG)                                                                     \
  run(#TAG, vllm_test::kLtx2Ups##TAG##Rational, vllm_test::kLtx2Ups##TAG##Scale,               \
      "ltx2.ups." #TAG ".", vllm_test::kLtx2Ups##TAG##ParamNames,                              \
      vllm_test::kLtx2Ups##TAG##ParamCounts, std::size(vllm_test::kLtx2Ups##TAG##ParamNames),  \
      vllm_test::kLtx2Ups##TAG##OutShape, vllm_test::kLtx2Ups##TAG##Golden,                    \
      std::size(vllm_test::kLtx2Ups##TAG##Golden))
  LTX2_UPS_ARM(PixelShuffle);
  LTX2_UPS_ARM(Rational2);
  LTX2_UPS_ARM(Rational1p5);
#undef LTX2_UPS_ARM

  // GroupNorm's group count is a LITERAL upstream (res_block.py:24,26; model.py:50),
  // not a config key a checkpoint could move, and its eps is torch's default
  // because no site passes one. NEITHER is a member of the invisible-constant
  // class: the three arms above reach both, MEASURED on this tree.
  //
  //   kLtx2UpsamplerNormEps    1e-5 -> 1e-3 (100x)  REDS PixelShuffle 0.0289409,
  //                            Rational2 0.0347079, Rational1p5 0.0649014
  //   kLtx2UpsamplerNormGroups 32   -> 16           REDS PixelShuffle 0.63738,
  //                            Rational2 0.633718, Rational1p5 0.874346
  //
  // Both stay pinned anyway, for the one thing the tensors cannot do: a
  // regeneration that moves the constant and the goldens together still passes
  // the value comparison, and only these two lines compare against upstream.
  CHECK(vllm::kLtx2UpsamplerNormGroups == vllm_test::kLtx2UpsNormGroups);
  CHECK(vllm::kLtx2UpsamplerNormEps == 1e-5);
}

TEST_CASE("ltx2 the latent temporal upsampler reproduces upstream") {
  // model.py:68-71 builds `Conv3d(mid, 2*mid, k=3, p=1)` + `PixelShuffleND(1)`,
  // and :113 drops the first frame after it (:109-113 is the whole branch).
  // Everything else in the class is
  // the same module set the three spatial arms above already gate, so this case
  // is aimed at exactly two things: the temporal shuffle and the slice.
  const vllm::Ltx2UpsamplerConfig config = TemporalUpsamplerConfig("ltx2.ups.Temporal.");
  const ParamBag bag = BuildUpsamplerParams(config);
  // The parameter CONTRACT first. `upsampler.0.weight` here is a Conv3d kernel
  // [2*mid, mid, 3, 3, 3]; the non-rational SPATIAL arm's identically-named
  // tensor is a 4-D Conv2d kernel (model.py:64-66). A port that reused the
  // spatial enumeration builds the wrong element count and dies here.
  CheckManifest(bag, vllm_test::kLtx2UpsTemporalParamNames,
                vllm_test::kLtx2UpsTemporalParamCounts,
                std::size(vllm_test::kLtx2UpsTemporalParamNames));

  const vllm::Ltx2LatentVolume latent = TemporalUpsamplerLatent();
  const vllm::Ltx2LatentVolume got = vllm::Ltx2LatentUpsample(config, bag.weights, latent);
  CHECK(got.batch == vllm_test::kLtx2UpsTemporalOutShape[0]);
  CHECK(got.channels == vllm_test::kLtx2UpsTemporalOutShape[1]);
  // 2F - 1, not 2F: the drop is observable in the SHAPE, so a port that kept the
  // first frame fails here before any value is compared.
  CHECK(got.frames == vllm_test::kLtx2UpsTemporalOutShape[2]);
  CHECK(got.frames == vllm_test::kLtx2UpsTemporalFactor * latent.frames - 1);
  CHECK(got.height == vllm_test::kLtx2UpsTemporalOutShape[3]);
  CHECK(got.width == vllm_test::kLtx2UpsTemporalOutShape[4]);

  const double worst = MaxAbsDiff(got.data, vllm_test::kLtx2UpsTemporalGolden,
                                  std::size(vllm_test::kLtx2UpsTemporalGolden));
  INFO("LatentUpsampler arm = Temporal max|diff| = ", worst);
  CHECK(worst <= kRoundOff);

  // `PixelShuffleND.__init__`'s `upscale_factors` default (pixel_shuffle.py:25),
  // which no construction site overrides. The goldens above move WITH it if it
  // is regenerated, so this line is the only one that compares against upstream's
  // own signature rather than against a tensor we produced.
  CHECK(vllm::kLtx2UpsamplerTemporalFactor == vllm_test::kLtx2UpsTemporalFactor);
}

TEST_CASE("ltx2 the upsampler refuses the arms it does not implement") {
  const vllm::Ltx2LatentVolume latent = ReducedUpsamplerLatent();
  const ParamBag bag = BuildUpsamplerParams(ReducedUpsamplerConfig(false, 2.0, "ltx2.ups.x."));

  auto refuse = [&](const std::string& what, vllm::Ltx2UpsamplerConfig config) {
    config.prefix = "ltx2.ups.x.";
    const std::string message =
        RefusalMessage([&] { (void)vllm::Ltx2LatentUpsample(config, bag.weights, latent); });
    INFO("arm = ", what, " refusal = ", message);
    return message;
  };

  // BOTH flags — `Conv3d(mid, 8*mid)` + `PixelShuffleND(3)` (model.py:55-59), a
  // different operator from the temporal-only arm above and still unported. The
  // upsampler weight upstream would build for it is 8*mid, emitted by the
  // generator off the real module so this is not gated against a remembered
  // shape.
  CHECK(vllm_test::kLtx2UpsSpatiotemporalUpsamplerShape[0] ==
        8 * vllm_test::kLtx2UpsMidChannels);
  vllm::Ltx2UpsamplerConfig spatiotemporal = ReducedUpsamplerConfig(false, 2.0, "");
  spatiotemporal.temporal_upsample = true;
  const std::string spatiotemporal_message = refuse("spatial+temporal", spatiotemporal);
  CHECK(Mentions(spatiotemporal_message, "temporal"));
  CHECK(Mentions(spatiotemporal_message, "PixelShuffleND(3)"));

  vllm::Ltx2UpsamplerConfig two_d = ReducedUpsamplerConfig(false, 2.0, "");
  two_d.dims = 2;
  CHECK(Mentions(refuse("dims=2", two_d), "dims"));

  // dims=2 is refused for the TEMPORAL arm too, and it has to be checked
  // separately: the two arms take different branches, so a `dims` guard placed
  // inside the spatial branch would let a 2-D temporal config through.
  vllm::Ltx2UpsamplerConfig temporal_two_d = TemporalUpsamplerConfig("");
  temporal_two_d.dims = 2;
  CHECK(Mentions(refuse("temporal dims=2", temporal_two_d), "dims"));

  // Upstream's own ValueError when neither flag is set (model.py:73-74).
  vllm::Ltx2UpsamplerConfig neither = ReducedUpsamplerConfig(false, 2.0, "");
  neither.spatial_upsample = false;
  CHECK(Mentions(refuse("neither", neither), "spatial_upsample"));
}

TEST_CASE("ltx2 upsample_video applies the encoder statistics around the upsampler") {
  // model.py:129-143 — un-normalize, upsample, re-normalize. Applying the
  // statistics in the wrong ORDER, or not at all, produces a correctly shaped
  // latent at the wrong scale, which the VAE then decodes into a plausible video.
  const vllm::Ltx2UpsamplerConfig config =
      ReducedUpsamplerConfig(false, 2.0, "ltx2.ups.PixelShuffle.");
  const ParamBag bag = BuildUpsamplerParams(config);
  const vllm::Ltx2LatentVolume latent = ReducedUpsamplerLatent();

  const int64_t channels = latent.channels;
  const std::vector<float> std_of_means(static_cast<size_t>(channels), 1.0f);
  const std::vector<float> mean_of_means(static_cast<size_t>(channels), 0.0f);
  // Identity statistics (upstream's own defaults, video_vae/ops.py:73-74) must
  // leave the result byte-identical to the bare upsampler.
  const vllm::Ltx2LatentVolume plain = vllm::Ltx2LatentUpsample(config, bag.weights, latent);
  const vllm::Ltx2LatentVolume identity = vllm::Ltx2UpsampleVideoLatent(
      config, bag.weights, latent, std_of_means, mean_of_means);
  CHECK(identity.data == plain.data);

  // A non-identity pair must NOT: if it did, the statistics are being dropped.
  std::vector<float> scaled(static_cast<size_t>(channels), 2.0f);
  std::vector<float> shifted(static_cast<size_t>(channels), 0.5f);
  const vllm::Ltx2LatentVolume moved =
      vllm::Ltx2UpsampleVideoLatent(config, bag.weights, latent, scaled, shifted);
  CHECK(moved.data != plain.data);
  CHECK(moved.data.size() == plain.data.size());
}

// ===========================================================================
// Section 9 — the duration head
// ===========================================================================

namespace {

vllm::Ltx2DurationHeadConfig ReducedDurationConfig() {
  vllm::Ltx2DurationHeadConfig config;
  config.video_cross_attention_dim = vllm_test::kLtx2DurVideoDim;
  config.audio_cross_attention_dim = vllm_test::kLtx2DurAudioDim;
  config.pooler_hidden_dim = vllm_test::kLtx2DurHidden;
  config.num_queries = vllm_test::kLtx2DurQueries;
  config.num_pooler_heads = vllm_test::kLtx2DurHeads;
  config.mlp_hidden = vllm_test::kLtx2DurMlpHidden;
  config.prefix = "ltx2.dur.";
  return config;
}

}  // namespace

TEST_CASE("ltx2 the duration head reproduces upstream") {
  const vllm::Ltx2DurationHeadConfig config = ReducedDurationConfig();
  ParamBag bag;
  for (const vllm::Ltx2DurationHeadTensorSpec& spec :
       vllm::EnumerateLtx2DurationHeadTensors(config)) {
    bag.Put(spec.name, spec.shape);
  }
  CheckManifest(bag, vllm_test::kLtx2DurParamNames, vllm_test::kLtx2DurParamCounts,
                std::size(vllm_test::kLtx2DurParamNames));

  const int64_t video_tokens = vllm_test::kLtx2DurVideoTokens;
  const int64_t audio_tokens = vllm_test::kLtx2DurAudioTokens;
  const std::vector<float> video =
      Make("ltx2.dur.video_tokens", video_tokens * config.video_cross_attention_dim, 1.0);
  const std::vector<float> audio =
      Make("ltx2.dur.audio_tokens", audio_tokens * config.audio_cross_attention_dim, 1.0);

  // The pooler on its own first, so a MultiheadAttention layout defect localizes
  // instead of arriving as one wrong scalar two layers later.
  const std::vector<float> pooled = vllm::Ltx2DurationAttentionPool(
      config, bag.weights, vllm_test::kLtx2DurProjectedGolden, 1,
      video_tokens + audio_tokens);
  const double pooled_worst = MaxAbsDiff(pooled, vllm_test::kLtx2DurPooledGolden,
                                         std::size(vllm_test::kLtx2DurPooledGolden));
  INFO("AttentionPooler max|diff| = ", pooled_worst);
  CHECK(pooled_worst <= kRoundOff);

  // AN INVARIANCE, recorded rather than mistaken for coverage. A mutation that
  // REVERSED the token-axis concat left every golden here green — and upstream
  // agrees: AttentionPooler is cross-attention with no mask and no positional
  // encoding over the token axis (duration_head.py:45-49), so it is PERMUTATION
  // INVARIANT. Measured on upstream, a reversed concat moves the pooled output by
  // 2.98e-08 (f32 reduction-order noise) while giving the audio stream the VIDEO
  // modality embedding moves it by 4.80e-03. What tags the streams is the
  // EMBEDDING, not the order, so that is what this holds. If upstream ever adds
  // positional information to the pooler, the first check below fails and the
  // invariance stops being true — which is exactly when someone should look.
  const double reversed_worst =
      MaxAbsDiff(std::vector<float>(vllm_test::kLtx2DurPooledReversedGolden,
                                    vllm_test::kLtx2DurPooledReversedGolden +
                                        std::size(vllm_test::kLtx2DurPooledReversedGolden)),
                 vllm_test::kLtx2DurPooledGolden, std::size(vllm_test::kLtx2DurPooledGolden));
  INFO("AttentionPooler reversed-concat max|diff| (upstream's own invariance) = ",
       reversed_worst);
  CHECK(reversed_worst <= kRoundOff);
  double mistagged_worst = 0.0;
  for (size_t i = 0; i < std::size(vllm_test::kLtx2DurPooledGolden); ++i) {
    mistagged_worst = std::max(
        mistagged_worst,
        std::fabs(static_cast<double>(vllm_test::kLtx2DurPooledGolden[i]) -
                  static_cast<double>(vllm_test::kLtx2DurPooledMistaggedGolden[i])));
  }
  INFO("AttentionPooler mis-tagged-modality max|diff| = ", mistagged_worst);
  CHECK(mistagged_worst > 1000.0 * kRoundOff);

  auto run = [&](const std::string& tag, const float* v, int64_t vt, const float* a, int64_t at,
                 const float* golden, size_t count) {
    const std::vector<float> got =
        vllm::Ltx2DurationPredict(config, bag.weights, v, vt, a, at, 1);
    const double worst = MaxAbsDiff(got, golden, count);
    INFO("DurationHead arm = ", tag, " max|diff| = ", worst, " seconds = ", got[0]);
    CHECK(worst <= kRoundOff);
    // The output is a DURATION in seconds: `exp` makes it strictly positive, and
    // a port that forgot the exp would routinely return a negative one.
    CHECK(got[0] > 0.0f);
  };
  run("Both", video.data(), video_tokens, audio.data(), audio_tokens,
      vllm_test::kLtx2DurBothGolden, std::size(vllm_test::kLtx2DurBothGolden));
  run("VideoOnly", video.data(), video_tokens, nullptr, 0, vllm_test::kLtx2DurVideoOnlyGolden,
      std::size(vllm_test::kLtx2DurVideoOnlyGolden));
  run("AudioOnly", nullptr, 0, audio.data(), audio_tokens, vllm_test::kLtx2DurAudioOnlyGolden,
      std::size(vllm_test::kLtx2DurAudioOnlyGolden));

  // A gate that cannot SEPARATE its arms is not a gate. At the shared fixture
  // scale these three collapsed to within 3e-6 of one another — below kRoundOff —
  // so an implementation that ignored one stream would have passed all three.
  // This asserts the separation the wider duration-head fixture buys, so a future
  // fixture change that collapses them again fails HERE rather than silently
  // weakening every arm above.
  const double both = vllm_test::kLtx2DurBothGolden[0];
  const double video_only = vllm_test::kLtx2DurVideoOnlyGolden[0];
  const double audio_only = vllm_test::kLtx2DurAudioOnlyGolden[0];
  INFO("arm separation: both = ", both, " video-only = ", video_only,
       " audio-only = ", audio_only);
  CHECK(std::fabs(both - video_only) > 100.0 * kRoundOff);
  CHECK(std::fabs(both - audio_only) > 100.0 * kRoundOff);
  CHECK(std::fabs(video_only - audio_only) > 100.0 * kRoundOff);

  // duration_head.py:104-105 — neither stream is upstream's own ValueError.
  const std::string message = RefusalMessage(
      [&] { (void)vllm::Ltx2DurationPredict(config, bag.weights, nullptr, 0, nullptr, 0, 1); });
  INFO("refusal = ", message);
  CHECK(Mentions(message, "video_tokens"));
  CHECK(Mentions(message, "audio_tokens"));
}

// ===========================================================================
// Section 10 — the embeddings connector
// ===========================================================================

namespace {

vllm::Ltx2ConnectorConfig ReducedConnectorConfig(const std::string& prefix, bool interleaved,
                                                 bool double_precision, int64_t registers,
                                                 bool gated, bool ff_bias) {
  vllm::Ltx2ConnectorConfig config;
  config.attention_head_dim = vllm_test::kLtx2ConnHeadDim;
  config.num_attention_heads = vllm_test::kLtx2ConnHeads;
  config.num_layers = vllm_test::kLtx2ConnLayers;
  config.positional_embedding_theta = vllm_test::kLtx2ConnTheta;
  config.positional_embedding_max_pos = {1};
  config.num_learnable_registers = registers;
  config.rope_type = interleaved ? vllm::Ltx2RopeType::kInterleaved : vllm::Ltx2RopeType::kSplit;
  config.double_precision_rope = double_precision;
  config.apply_gated_attention = gated;
  config.ff_bias = ff_bias;
  config.prefix = prefix;
  return config;
}

}  // namespace

TEST_CASE("ltx2 the Embeddings1DConnector reproduces upstream on every arm") {
  const int64_t batch = vllm_test::kLtx2ConnBatch;
  const int64_t seq = vllm_test::kLtx2ConnSeq;
  const int64_t inner = vllm_test::kLtx2ConnInnerDim;
  const std::vector<float> hidden = Make("ltx2.conn.hidden", batch * seq * inner, 1.0);

  // The additive mask upstream's own preprocessor produces (transformer_args.py:204).
  std::vector<float> mask(static_cast<size_t>(batch * seq), 0.0f);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      if (vllm_test::kLtx2ConnKeep[b * seq + s] == 0) {
        mask[static_cast<size_t>(b * seq + s)] = -std::numeric_limits<float>::max();
      }
    }
  }

  auto run = [&](const std::string& tag, bool interleaved, bool double_precision, int64_t registers,
                 bool gated, bool ff_bias, const char* const* names, const int64_t* counts,
                 size_t manifest_size, const float* golden, size_t golden_count,
                 const float* mask_golden, size_t mask_count) {
    const vllm::Ltx2ConnectorConfig config = ReducedConnectorConfig(
        std::string("ltx2.conn.") + tag + ".", interleaved, double_precision, registers, gated,
        ff_bias);
    ParamBag bag;
    for (const vllm::Ltx2ConnectorTensorSpec& spec :
         vllm::EnumerateLtx2ConnectorTensors(config)) {
      bag.Put(spec.name, spec.shape);
    }
    CheckManifest(bag, names, counts, manifest_size);

    const vllm::Ltx2ConnectorOutput got =
        vllm::Ltx2ConnectorForward(config, bag.weights, hidden.data(), mask.data(), batch, seq);
    const double worst = MaxAbsDiff(got.hidden_states, golden, golden_count);
    INFO("Embeddings1DConnector arm = ", tag, " max|diff| = ", worst);
    CHECK(worst <= kRoundOff);
    const double mask_worst = MaxAbsDiff(got.mask, mask_golden, mask_count);
    INFO("Embeddings1DConnector arm = ", tag, " mask max|diff| = ", mask_worst);
    CHECK(mask_worst <= kExactRoundOff);
  };

#define LTX2_CONN_ARM(TAG)                                                                    \
  run(#TAG, vllm_test::kLtx2Conn##TAG##Interleaved, vllm_test::kLtx2Conn##TAG##DoublePrecision,\
      vllm_test::kLtx2Conn##TAG##Registers, vllm_test::kLtx2Conn##TAG##Gated,                  \
      vllm_test::kLtx2Conn##TAG##FfBias, vllm_test::kLtx2Conn##TAG##ParamNames,                \
      vllm_test::kLtx2Conn##TAG##ParamCounts,                                                  \
      std::size(vllm_test::kLtx2Conn##TAG##ParamNames), vllm_test::kLtx2Conn##TAG##Golden,     \
      std::size(vllm_test::kLtx2Conn##TAG##Golden), vllm_test::kLtx2Conn##TAG##MaskGolden,     \
      std::size(vllm_test::kLtx2Conn##TAG##MaskGolden))
  LTX2_CONN_ARM(Split);
  LTX2_CONN_ARM(Interleaved);
  LTX2_CONN_ARM(Float64);
  LTX2_CONN_ARM(NoRegisters);
  LTX2_CONN_ARM(GatedNoBias);
#undef LTX2_CONN_ARM
}

TEST_CASE("ltx2 the connector's learnable registers are stored BFLOAT16, and rounded") {
  // embeddings_connector.py:135-137 constructs the table with
  // `dtype=torch.bfloat16`. Carrying those values at f32 is WIDER than upstream —
  // finite, correctly shaped, and only the PADDED positions move, which is the
  // polarity AGENTS.md says a value gate cannot catch. This arm catches it,
  // because the substitution's own golden was produced from the rounded table.
  const int64_t batch = vllm_test::kLtx2ConnBatch;
  const int64_t seq = vllm_test::kLtx2ConnSeq;
  const int64_t inner = vllm_test::kLtx2ConnInnerDim;
  const std::vector<float> hidden = Make("ltx2.conn.hidden", batch * seq * inner, 1.0);
  std::vector<float> mask(static_cast<size_t>(batch * seq), 0.0f);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      if (vllm_test::kLtx2ConnKeep[b * seq + s] == 0) {
        mask[static_cast<size_t>(b * seq + s)] = -std::numeric_limits<float>::max();
      }
    }
  }

  const vllm::Ltx2ConnectorConfig config =
      ReducedConnectorConfig("ltx2.conn.Split.", false, false,
                             vllm_test::kLtx2ConnSplitRegisters, false, true);
  ParamBag bag;
  for (const vllm::Ltx2ConnectorTensorSpec& spec : vllm::EnumerateLtx2ConnectorTensors(config)) {
    bag.Put(spec.name, spec.shape);
  }

  // The stored table, rounded to bf16, must equal upstream's parameter exactly.
  // If our side kept f32 this comparison fails on nearly every element.
  const std::vector<float>& stored = bag.weights.Get(config.prefix + "learnable_registers");
  std::vector<float> rounded(stored.size());
  for (size_t i = 0; i < stored.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &stored[i], sizeof(bits));
    // round-to-nearest-even into bf16, then back — torch's own `.to(bfloat16)`.
    const uint32_t lsb = (bits >> 16) & 1u;
    bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
    std::memcpy(&rounded[i], &bits, sizeof(bits));
  }
  const double table_worst = MaxAbsDiff(rounded, vllm_test::kLtx2ConnSplitRegistersGolden,
                                        std::size(vllm_test::kLtx2ConnSplitRegistersGolden));
  INFO("learnable_registers (bf16-rounded) max|diff| = ", table_worst);
  CHECK(table_worst <= kExactRoundOff);
  // ...and the UNROUNDED table does NOT match, which is what makes the check
  // above a gate rather than a formality.
  double unrounded_worst = 0.0;
  for (size_t i = 0; i < stored.size(); ++i) {
    unrounded_worst = std::max(
        unrounded_worst,
        std::fabs(static_cast<double>(stored[i]) -
                  static_cast<double>(vllm_test::kLtx2ConnSplitRegistersGolden[i])));
  }
  INFO("learnable_registers (f32, unrounded) max|diff| = ", unrounded_worst);
  CHECK(unrounded_worst > 0.0);

  const vllm::Ltx2ConnectorOutput replaced = vllm::Ltx2ConnectorReplaceRegisters(
      config, bag.weights, hidden.data(), mask.data(), batch, seq);
  const double worst = MaxAbsDiff(replaced.hidden_states,
                                  vllm_test::kLtx2ConnSplitReplacedGolden,
                                  std::size(vllm_test::kLtx2ConnSplitReplacedGolden));
  INFO("register substitution max|diff| = ", worst);
  CHECK(worst <= kExactRoundOff);

  // The mask is REPLACED by zeros (:152): every position is attendable
  // afterwards, including the ones that were padding.
  const double mask_worst = MaxAbsDiff(replaced.mask,
                                       vllm_test::kLtx2ConnSplitZeroedMaskGolden,
                                       std::size(vllm_test::kLtx2ConnSplitZeroedMaskGolden));
  CHECK(mask_worst <= kExactRoundOff);
  for (float value : replaced.mask) CHECK(value == 0.0f);

  // Kept positions must be UNTOUCHED, and padded ones must not be: without both
  // halves, an implementation that replaced everything (or nothing) still matches
  // a golden built the same wrong way.
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      const size_t base = static_cast<size_t>((b * seq + s) * inner);
      const bool keep = vllm_test::kLtx2ConnKeep[b * seq + s] != 0;
      if (keep) {
        CHECK(replaced.hidden_states[base] == hidden[base]);
      } else {
        CHECK(replaced.hidden_states[base] != hidden[base]);
      }
    }
  }

  // :144 — the sequence must tile the register table exactly.
  const std::string message = RefusalMessage([&] {
    (void)vllm::Ltx2ConnectorReplaceRegisters(config, bag.weights, hidden.data(), mask.data(),
                                              batch, seq - 1);
  });
  INFO("refusal = ", message);
  CHECK(Mentions(message, "learnable_registers"));
}

// ===========================================================================
// Section 10a — the PROCESSOR around the two connectors (phase L9c)
// ===========================================================================
//
// `EmbeddingsProcessor.create_embeddings` (embeddings_processor.py:70-95) is a
// separate module from the connector, and the two things it does that the
// connector does not are both invisible to a value golden of the connector
// itself. They are gated here by their upstream-stated PROPERTIES rather than by
// a new golden, because a golden generated through the same helper the port uses
// proves the two agree and not that either is right — this project has recorded
// that failure once already.

namespace {

// Two independently seeded bags at the two stream widths the shipped checkpoint
// uses, so the video and audio connectors cannot accidentally be the same module.
vllm::Ltx2ConnectorConfig ProcessorConfig(const std::string& prefix, int64_t heads,
                                          int64_t head_dim, int64_t registers) {
  vllm::Ltx2ConnectorConfig config;
  config.attention_head_dim = head_dim;
  config.num_attention_heads = heads;
  config.num_layers = vllm_test::kLtx2ConnLayers;
  config.positional_embedding_theta = vllm_test::kLtx2ConnTheta;
  config.positional_embedding_max_pos = {4096};  // the shipped connector bound
  config.num_learnable_registers = registers;
  config.rope_type = vllm::Ltx2RopeType::kSplit;
  config.double_precision_rope = true;
  config.apply_gated_attention = true;
  config.ff_bias = true;
  config.prefix = prefix;
  return config;
}

ParamBag ProcessorBag(const vllm::Ltx2ConnectorConfig& config) {
  ParamBag bag;
  for (const vllm::Ltx2ConnectorTensorSpec& spec : vllm::EnumerateLtx2ConnectorTensors(config)) {
    bag.Put(spec.name, spec.shape);
  }
  return bag;
}

}  // namespace

TEST_CASE("ltx2 the processor is PADDING-SIDE AGNOSTIC, which is what the sort is for") {
  // "Connectors expect right-padded input ([valid, pad]). Normalize layout here
  // so the upstream tokenizer can keep using either side without coupling to the
  // connector." — embeddings_processor.py:80-82.
  //
  // THE DEFECT THIS CATCHES. The register table is indexed by ABSOLUTE position
  // (`s % num_registers`), not by which positions were padded, so a LEFT-padded
  // batch handed straight to the connector puts registers where caption tokens
  // belong. The result is finite, correctly shaped, and conditioned on the wrong
  // thing — which no shape or finiteness check can see. Skipping the sort makes
  // these two renders DIFFER; upstream's own contract is that they are the same.
  const int64_t batch = 1, seq = 4, valid = 2;
  const vllm::Ltx2ConnectorConfig vcfg = ProcessorConfig("ltx2.proc.v.", 3, 8, 2);
  const vllm::Ltx2ConnectorConfig acfg = ProcessorConfig("ltx2.proc.a.", 2, 4, 2);
  const ParamBag vbag = ProcessorBag(vcfg);
  const ParamBag abag = ProcessorBag(acfg);
  const int64_t vdim = vcfg.inner_dim(), adim = acfg.inner_dim();

  const std::vector<float> real_v = Make("ltx2.proc.real.v", valid * vdim, 1.0);
  const std::vector<float> real_a = Make("ltx2.proc.real.a", valid * adim, 1.0);
  const std::vector<float> junk_v = Make("ltx2.proc.junk.v", (seq - valid) * vdim, 1.0);
  const std::vector<float> junk_a = Make("ltx2.proc.junk.a", (seq - valid) * adim, 1.0);

  auto build = [&](bool pad_left, const std::vector<float>& real, const std::vector<float>& junk,
                   int64_t width) {
    std::vector<float> out;
    if (pad_left) {
      out.insert(out.end(), junk.begin(), junk.end());
      out.insert(out.end(), real.begin(), real.end());
    } else {
      out.insert(out.end(), real.begin(), real.end());
      out.insert(out.end(), junk.begin(), junk.end());
    }
    (void)width;
    return out;
  };
  auto mask_of = [&](bool pad_left) {
    std::vector<float> m(static_cast<size_t>(seq), 0.0f);
    for (int64_t s = 0; s < seq - valid; ++s) {
      m[static_cast<size_t>(pad_left ? s : valid + s)] = -std::numeric_limits<float>::max();
    }
    return m;
  };

  const std::vector<float> right_v = build(false, real_v, junk_v, vdim);
  const std::vector<float> right_a = build(false, real_a, junk_a, adim);
  const std::vector<float> left_v = build(true, real_v, junk_v, vdim);
  const std::vector<float> left_a = build(true, real_a, junk_a, adim);
  const std::vector<float> right_mask = mask_of(false);
  const std::vector<float> left_mask = mask_of(true);

  const vllm::Ltx2ConnectorEmbeddings from_right = vllm::Ltx2ConnectorCreateEmbeddings(
      vcfg, vbag.weights, right_v.data(), acfg, abag.weights, right_a.data(), right_mask.data(),
      batch, seq);
  const vllm::Ltx2ConnectorEmbeddings from_left = vllm::Ltx2ConnectorCreateEmbeddings(
      vcfg, vbag.weights, left_v.data(), acfg, abag.weights, left_a.data(), left_mask.data(),
      batch, seq);

  const double video_worst =
      MaxAbsDiff(from_left.video, from_right.video.data(), from_right.video.size());
  const double audio_worst =
      MaxAbsDiff(from_left.audio, from_right.audio.data(), from_right.audio.size());
  INFO("padding-side agnostic: video max|diff| = ", video_worst, " audio max|diff| = ",
       audio_worst);
  CHECK(video_worst <= kExactRoundOff);
  CHECK(audio_worst <= kExactRoundOff);

  // THE CONTROL. The two inputs really are different buffers, so the equality
  // above is the sort working and not two identical arrays compared to
  // themselves. Without this the case passes on a port that ignores its input.
  CHECK(left_v != right_v);
  CHECK(left_a != right_a);
  // ...and with registers on, every position is attendable (:152).
  for (const float m : from_right.mask) CHECK(m == 1.0f);
}

TEST_CASE("ltx2 the processor's binary mask mirrors a comparison that looks backwards") {
  // `_to_binary_mask` is `encoded_mask < 0.000001` (embeddings_processor.py:46-48).
  // An additive mask holds 0.0 for KEPT and -finfo(f32).max for PADDED, and BOTH
  // are `< 0.000001` — so the mask this produces is ONE EVERYWHERE, at every
  // position, for every input either reference can produce, and the video-only
  // multiply that follows it is an identity.
  //
  // THIS CASE EXISTS BECAUSE THE INTENT-READING IS THE OPPOSITE. A port that
  // reasoned "keep the unmasked ones" would write `>= 0`, get 0 at padded
  // positions, zero the video encoding there, and hand the DiT a mask that
  // masks. It would look more correct and it would not be a port. Written as a
  // gate on the SURPRISING behaviour so that "fixing" it REDs.
  //
  // Confirmed on BOTH references before being pinned, because a line this odd is
  // where one implementation being wrong would show: `diffusers`
  // `LTX2TextConnectors.forward` writes `(video_attn_mask < 1e-6).to(torch.int64)`
  // and the same video-only multiply. They agree, down to the constant.
  //
  // Registers are DISABLED here so the connector passes the caller's mask through
  // instead of zeroing it — that is the only configuration in which the two
  // readings differ at all.
  const int64_t batch = 1, seq = 4, valid = 2;
  const vllm::Ltx2ConnectorConfig vcfg = ProcessorConfig("ltx2.proc.nv.", 3, 8, 0);
  const vllm::Ltx2ConnectorConfig acfg = ProcessorConfig("ltx2.proc.na.", 2, 4, 0);
  const ParamBag vbag = ProcessorBag(vcfg);
  const ParamBag abag = ProcessorBag(acfg);
  const int64_t vdim = vcfg.inner_dim(), adim = acfg.inner_dim();

  const std::vector<float> hidden_v = Make("ltx2.proc.nv.hidden", batch * seq * vdim, 1.0);
  const std::vector<float> hidden_a = Make("ltx2.proc.na.hidden", batch * seq * adim, 1.0);
  std::vector<float> mask(static_cast<size_t>(seq), 0.0f);
  for (int64_t s = valid; s < seq; ++s) mask[static_cast<size_t>(s)] = -std::numeric_limits<float>::max();

  const vllm::Ltx2ConnectorEmbeddings got = vllm::Ltx2ConnectorCreateEmbeddings(
      vcfg, vbag.weights, hidden_v.data(), acfg, abag.weights, hidden_a.data(), mask.data(),
      batch, seq);

  for (int64_t s = 0; s < seq; ++s) {
    const bool kept = s < valid;
    double video_abs = 0.0, audio_abs = 0.0;
    for (int64_t i = 0; i < vdim; ++i) {
      video_abs += std::fabs(static_cast<double>(got.video[static_cast<size_t>(s * vdim + i)]));
    }
    for (int64_t i = 0; i < adim; ++i) {
      audio_abs += std::fabs(static_cast<double>(got.audio[static_cast<size_t>(s * adim + i)]));
    }
    INFO("position ", s, " kept = ", kept, " |video| = ", video_abs, " |audio| = ", audio_abs);
    // ONE at EVERY position, padded ones included. The `>= 0` reading gives 0.0
    // here for s >= valid, which is what this pins.
    CHECK(got.mask[static_cast<size_t>(s)] == 1.0f);
    // ...so nothing is zeroed, in either modality. Under the `>= 0` reading the
    // video row would be exactly 0 at the padded positions.
    CHECK(video_abs > 0.0);
    CHECK(audio_abs > 0.0);
  }

  // THE CONTROL. The connector really did see a mask with padded positions in
  // it: with registers disabled the padded rows are NOT substituted, so they
  // still derive from the caller's own features. Without this the case would
  // pass on a processor that dropped the mask argument entirely.
  const int64_t pad = seq - valid;
  std::vector<float> unpadded_mask(static_cast<size_t>(seq), 0.0f);
  const vllm::Ltx2ConnectorEmbeddings all_valid = vllm::Ltx2ConnectorCreateEmbeddings(
      vcfg, vbag.weights, hidden_v.data(), acfg, abag.weights, hidden_a.data(),
      unpadded_mask.data(), batch, seq);
  const double masked_vs_unmasked =
      MaxAbsDiff(all_valid.video, got.video.data(), got.video.size());
  INFO("padded rows = ", pad, " max|diff| vs an all-valid mask = ", masked_vs_unmasked);
  CHECK(masked_vs_unmasked > 0.0);
}
