// GLM-5.3-Flash W4 gate — the manifold hyper-connection (mHC) residual topology
// and its UNWEIGHTED head collapse (#2098, row
// MODEL-MM-glm5-next-glm5-next-for-conditional-generation,
// `.agents/specs/glm5-next-flash.md` §W4).
//
// THE ORACLE IS RUN, NOT TRANSCRIBED. Every golden in
// `fixtures/glm5_next_mhc_goldens.inc` is the return value of an unmodified
// `transformers` v5.16.1 module — `Glm5NextTextHyperConnection.forward` and
// `Glm5NextTextHyperHead.forward` — captured by
// `fixtures/gen_glm5_next_mhc_goldens.py`. vLLM registers no `glm5_next` at any
// revision, so under AGENTS.md "When vLLM has no implementation" transformers is
// the reference for this surface. W0 (#2096) owns recording the lane revision.
//
// WHAT THIS GATE IS FOR. Three of the four mHC pieces are DeepSeek-V4's and are
// reused; the fourth is not, and reusing it is the defect this file exists to
// detect. `Glm5NextTextHyperHead.forward` is `hidden_streams.mean(dim=2)` and its
// own docstring says "Unlike DeepSeek-V4". `deepseek_v4::HcHeadCollapse` is V4's
// gated collapse. Swapping one for the other yields a model that runs and
// produces fluent text through a wrong final projection, which no end-to-end
// token gate on this row could ever catch — there is none (spec §Gates: the
// smallest published artifact is 181.32 GiB against ~119.63 GiB on GB10).
//
// So `head_is_not_v4_gated_collapse` below is a DISCRIMINATING case, and it is
// hand-derivable: with `fn == 0` and `base == 0`, V4's gate is
// `sigmoid(0) + hc_eps` on every stream, so its output is
// `(0.5 + 1e-6) * sum_m x[m]` = `(2 + 4e-6)x` the mean at `hc_mult == 4`. A port
// that reuses `HcHeadCollapse` is off by a factor of two and never crashes.
#include "vllm/model_executor/models/glm5_next_mhc.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_mhc.h"
#include "vllm/model_executor/models/glm5_next.h"

#include "glm5_next_mhc_goldens.inc"

namespace g = glm5_next_mhc_goldens;

namespace {

constexpr int64_t kHc = g::kHcMult;
constexpr int64_t kHidden = g::kHidden;
constexpr int64_t kSeq = g::kSeq;

// The tolerance for "our f32 host reduction agrees with the oracle's f32 torch
// reduction". Both are float32; only the summation ORDER differs, so the gap is
// accumulated rounding over a 32-wide dot product, not an algorithmic gap.
constexpr float kTol = 2e-6f;

vllm::Glm5NextMhcParams Mhc() {
  vllm::Glm5NextMhcParams p;
  p.mult = g::kHcMult;
  p.sinkhorn_iters = g::kHcSinkhornIters;
  p.eps = static_cast<double>(g::kHcEps);
  return p;
}

vllm::glm5_next::HcSite Site() {
  vllm::glm5_next::HcSite s;
  s.fn.assign(std::begin(g::kFn), std::end(g::kFn));
  s.base.assign(std::begin(g::kBase), std::end(g::kBase));
  s.scale.assign(std::begin(g::kScale), std::end(g::kScale));
  return s;
}

// Token `t`'s [hc, hidden] slice of the stream manifold.
std::vector<float> Streams(int64_t t) {
  const int64_t n = kHc * kHidden;
  return std::vector<float>(g::kStreams + t * n, g::kStreams + (t + 1) * n);
}

std::vector<float> SublayerOut(int64_t t) {
  return std::vector<float>(g::kSublayerOut + t * kHidden,
                            g::kSublayerOut + (t + 1) * kHidden);
}

float MaxAbsDiff(const std::vector<float>& a, const float* b, size_t n) {
  float worst = 0.0f;
  for (size_t i = 0; i < n; ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

}  // namespace

TEST_CASE("glm5_next mHC goldens come from the pinned reference, at hc_mult 4") {
  // An oracle whose identity is not asserted is an oracle nobody can reproduce.
  CHECK(std::string(g::kOracle) == "transformers 5.16.1");
  // The published checkpoint's mHC constants. `hc_eps` is a DIFFERENT constant
  // from `rms_norm_eps` and a port that collapses the two passes every shape
  // check and every token gate.
  CHECK(g::kHcMult == 4);
  CHECK(g::kHcSinkhornIters == 20);
  // Exact float equality, deliberately. `doctest::Approx` adds a scale term with
  // a ~1.19e-5 floor, which is LARGER than either constant, so an Approx compare
  // here cannot distinguish 1e-6 from 1e-5 -- or from zero.
  CHECK(g::kHcEps == 1e-6f);
  CHECK(g::kRmsNormEps == 1e-5f);
  CHECK(g::kHcEps != g::kRmsNormEps);
}

TEST_CASE("glm5_next mHC head collapse is the reference's unweighted mean") {
  for (int64_t t = 0; t < kSeq; ++t) {
    CAPTURE(t);
    const std::vector<float> x = Streams(t);
    const std::vector<float> got = vllm::glm5_next::HcHeadCollapseMean(x, kHc, kHidden);
    REQUIRE(got.size() == static_cast<size_t>(kHidden));

    // (a) against the ORACLE's own output.
    const float* want = g::kHeadOut + t * kHidden;
    for (int64_t h = 0; h < kHidden; ++h) {
      CAPTURE(h);
      CHECK(std::abs(got[h] - want[h]) <= kTol);
    }

    // (b) against an independent DOUBLE-precision recompute of the definition,
    // so an agreeing (a) is not a transcription of the same rounding.
    for (int64_t h = 0; h < kHidden; ++h) {
      double acc = 0.0;
      for (int64_t m = 0; m < kHc; ++m) acc += static_cast<double>(x[m * kHidden + h]);
      CAPTURE(h);
      CHECK(std::abs(static_cast<double>(got[h]) - acc / static_cast<double>(kHc)) <= 1e-6);
    }
  }
}

TEST_CASE("glm5_next mHC head collapse is NOT DeepSeek-V4's gated collapse") {
  const std::vector<float> x = Streams(0);
  const std::vector<float> mean = vllm::glm5_next::HcHeadCollapseMean(x, kHc, kHidden);

  SUBCASE("hand-derived: a zeroed V4 gate returns (2 + 4e-6)x the mean") {
    // fn == 0 makes every projection zero, so V4's gate is
    // `sigmoid(0 * scale + 0) + hc_eps` = 0.5 + 1e-6 on all four streams and its
    // output is `(0.5 + 1e-6) * sum_m x[m]`. At hc_mult 4 the mean is
    // `0.25 * sum_m x[m]`, so the ratio is exactly `2 + 4e-6`. No tuning: the
    // factor of two is what the V4 formula does at its own neutral point.
    const std::vector<float> zero_fn(static_cast<size_t>(kHc) * kHc * kHidden, 0.0f);
    const std::vector<float> zero_base(static_cast<size_t>(kHc), 0.0f);
    const std::vector<float> v4 = vllm::deepseek_v4::HcHeadCollapse(
        x, zero_fn, /*scale=*/1.0f, zero_base, kHc, kHidden, g::kRmsNormEps, g::kHcEps);
    REQUIRE(v4.size() == mean.size());
    for (int64_t h = 0; h < kHidden; ++h) {
      CAPTURE(h);
      CHECK(std::abs(v4[h] - (2.0f + 4e-6f) * mean[h]) <= 1e-5f);
    }
    // And therefore the two disagree wherever the mean is not ~zero.
    float worst = 0.0f;
    for (int64_t h = 0; h < kHidden; ++h) worst = std::max(worst, std::abs(v4[h] - mean[h]));
    CAPTURE(worst);
    CHECK(worst > 0.1f);
  }

  SUBCASE("randomised V4 weights disagree with the mean by a wide margin") {
    // A seeded, ordinary parameterisation — nothing adversarial. `hc_head_fn` is
    // [hc, hc*hidden] and `hc_head_scale` is a scalar, per deepseek_v4_mhc.h.
    std::mt19937 rng(2098);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> fn(static_cast<size_t>(kHc) * kHc * kHidden);
    for (auto& v : fn) v = nd(rng);
    std::vector<float> base(static_cast<size_t>(kHc));
    for (auto& v : base) v = nd(rng);

    const std::vector<float> v4 = vllm::deepseek_v4::HcHeadCollapse(
        x, fn, /*scale=*/1.25f, base, kHc, kHidden, g::kRmsNormEps, g::kHcEps);
    float worst = 0.0f;
    for (int64_t h = 0; h < kHidden; ++h) worst = std::max(worst, std::abs(v4[h] - mean[h]));
    CAPTURE(worst);
    CHECK(worst > 0.1f);
  }
}

TEST_CASE("glm5_next mHC pre reuses DeepSeek-V4's at this row's constants") {
  const vllm::Glm5NextMhcParams mhc = Mhc();
  const vllm::glm5_next::HcSite site = Site();

  for (int64_t t = 0; t < kSeq; ++t) {
    CAPTURE(t);
    const std::vector<float> x = Streams(t);
    const vllm::deepseek_v4::MhcPreResult got =
        vllm::glm5_next::MhcPre(x, site, mhc, kHidden, g::kRmsNormEps);

    REQUIRE(got.post_mix.size() == static_cast<size_t>(kHc));
    REQUIRE(got.comb_mix.size() == static_cast<size_t>(kHc * kHc));
    REQUIRE(got.layer_input.size() == static_cast<size_t>(kHidden));

    CHECK(MaxAbsDiff(got.post_mix, g::kPost + t * kHc, static_cast<size_t>(kHc)) <= kTol);
    CHECK(MaxAbsDiff(got.comb_mix, g::kComb + t * kHc * kHc,
                     static_cast<size_t>(kHc * kHc)) <= kTol);
    CHECK(MaxAbsDiff(got.layer_input, g::kCollapsed + t * kHidden,
                     static_cast<size_t>(kHidden)) <= kTol);

    // `post` is `2 * sigmoid(...)`, so it lives in (0, 2). A port that dropped
    // the alpha would still match a golden captured at a gate near 0.5, so pin
    // that at least one entry is above 1.0 — outside a plain sigmoid's range.
    float hi = 0.0f;
    for (float v : got.post_mix) hi = std::max(hi, v);
    CAPTURE(hi);
    CHECK(hi > 1.0f);
  }
}

TEST_CASE("glm5_next mHC post folds the sublayer output back, comb TRANSPOSED") {
  const vllm::Glm5NextMhcParams mhc = Mhc();
  const vllm::glm5_next::HcSite site = Site();

  for (int64_t t = 0; t < kSeq; ++t) {
    CAPTURE(t);
    const std::vector<float> x = Streams(t);
    const std::vector<float> y = SublayerOut(t);
    const vllm::deepseek_v4::MhcPreResult pre =
        vllm::glm5_next::MhcPre(x, site, mhc, kHidden, g::kRmsNormEps);

    // The golden's own comb must be ASYMMETRIC, or this case cannot see a
    // transposed port at all and the assertion below is vacuous.
    const float* comb = g::kComb + t * kHc * kHc;
    float asym = 0.0f;
    for (int64_t i = 0; i < kHc; ++i) {
      for (int64_t j = 0; j < kHc; ++j) {
        asym = std::max(asym, std::abs(comb[i * kHc + j] - comb[j * kHc + i]));
      }
    }
    CAPTURE(asym);
    CHECK(asym > 1e-3f);

    const std::vector<float> got = vllm::glm5_next::MhcPost(y, x, pre, kHc, kHidden);
    REQUIRE(got.size() == static_cast<size_t>(kHc * kHidden));
    CHECK(MaxAbsDiff(got, g::kMixed + t * kHc * kHidden,
                     static_cast<size_t>(kHc * kHidden)) <= kTol);
  }
}
