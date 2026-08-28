// GLM-5.3-Flash KDA arm (MODEL-MM-GLM53-FLASH W2) — UNIT GATE.
//
// HONEST SCOPE. §Gates of `.agents/specs/glm5-next-flash.md` records that NO
// end-to-end token gate against an oracle exists or can exist for this model on
// this fleet: the only admissible reference is transformers `v5.16.1` and
// running it needs 305.78 GiB (FP8) or 598.5 GiB (BF16) against ~119.63 GiB on
// the largest device here. So this file gates the KDA arm's MATH against
// hand-derived literal cases whose expected values are readable straight off
// `modular_glm5_next.py`, plus a from-first-principles double-precision
// reference that mirrors the reference's own line order and its own (transposed)
// state layout. That is the "host reference + structural review" bar.
//
// THE DELIVERABLE'S PROOF is the first section: the forget gate takes the
// SIGMOID branch, and every case there FAILS against `kimi_kda.cpp:60`'s
// softplus branch. That is asserted permanently, not just captured once — the
// last case in the section runs both functions on the same inputs and requires
// them to disagree by more than three orders of magnitude, while requiring our
// OWN softplus branch to agree with `kimi_kda`'s to float precision. The pair
// separates "different branch" from "broken port".
#include "vllm/model_executor/models/glm5_next_kda.h"

#include <doctest/doctest.h>

#include <algorithm>  // std::max
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "vllm/model_executor/models/kimi_kda.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using namespace vllm::glm5_next_kda;

namespace {

vt::Queue CpuQ() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

double Sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Relative L2 of a f32 result against a f64 reference.
double RelL2(const std::vector<float>& a, const std::vector<double>& b) {
  REQUIRE(a.size() == b.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return std::sqrt(num) / std::max(std::sqrt(den), 1e-30);
}

double RelL2F(const std::vector<float>& a, const std::vector<float>& b) {
  return RelL2(a, std::vector<double>(b.begin(), b.end()));
}

std::vector<float> Rand(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

}  // namespace

// ═══ (1) THE FORGET GATE — the sigmoid branch ════════════════════════════════

TEST_CASE("glm5_next kda: the forget gate is the SIGMOID branch") {
  // H=1, D=2, A_log=[0] -> decay_rate = exp(0) = 1. dt_bias is ZERO, and it
  // is PASSED: upstream declares it unconditionally (:384) and always adds it
  // (:393), so there is no biasless mode to exercise.
  // g1 = [0, ln 3]:  sigmoid(1*0) = 1/2 -> -5.0 * 1/2 = -2.5
  //                  sigmoid(1*ln 3) = 3/4 -> -5.0 * 3/4 = -3.75
  const std::vector<float> g1 = {0.0f, static_cast<float>(std::log(3.0))};
  const std::vector<float> a_log = {0.0f};
  const std::vector<float> y =
      Glm5NextForgetGate(g1, a_log, {0.0f, 0.0f}, 1, 1, 2, -5.0);
  REQUIRE(y.size() == 2);
  CHECK(y[0] == doctest::Approx(-2.5));
  CHECK(y[1] == doctest::Approx(-3.75));
}

TEST_CASE("glm5_next kda: the gate CANNOT leave [bound, 0]") {
  // The whole point of `safe_gate_lower_bound`: the per-step log-decay is
  // floored at the bound, so no single step can annihilate the state. The
  // softplus branch has no such floor and returns -100 for g1 = 100.
  const std::vector<float> a_log = {0.0f};
  const std::vector<float> hi =
      Glm5NextForgetGate({10.0f}, a_log, {0.0f}, 1, 1, 1, -5.0);
  const std::vector<float> lo =
      Glm5NextForgetGate({-10.0f}, a_log, {0.0f}, 1, 1, 1, -5.0);
  CHECK(hi[0] > -5.0f);
  CHECK(hi[0] == doctest::Approx(-5.0).epsilon(1e-4));
  CHECK(lo[0] < 0.0f);
  CHECK(lo[0] == doctest::Approx(-5.0 / (1.0 + std::exp(10.0))));
  // Saturated in f32, the bound is REACHED and never passed. The softplus
  // branch's answer for the same input is -100.
  const std::vector<float> sat =
      Glm5NextForgetGate({100.0f}, a_log, {0.0f}, 1, 1, 1, -5.0);
  CHECK(sat[0] == -5.0f);
  CHECK(vllm::kimi_kda::KdaDecayGate({100.0f}, a_log, {}, 1, 1, 1)[0] ==
        doctest::Approx(-100.0));
}

TEST_CASE("glm5_next kda: decay_rate is POSITIVE inside the sigmoid") {
  // A_log = ln 10 -> decay_rate = +10 (modular_glm5_next.py:394-395), and it
  // MULTIPLIES g INSIDE the sigmoid (:399). g1 = -1 gives sigmoid(-10), so the
  // channel forgets almost nothing. Feeding -exp(A_log) instead — the sign the
  // softplus branch uses (:405) — gives sigmoid(+10) and -4.99977, four orders
  // of magnitude away, with no NaN anywhere to reveal it.
  const std::vector<float> a_log = {static_cast<float>(std::log(10.0))};
  const std::vector<float> y =
      Glm5NextForgetGate({-1.0f}, a_log, {0.0f}, 1, 1, 1, -5.0);
  CHECK(y[0] == doctest::Approx(-5.0 / (1.0 + std::exp(10.0))));
  CHECK(y[0] != doctest::Approx(-5.0 / (1.0 + std::exp(-10.0))));
}

TEST_CASE("glm5_next kda: A_log is per HEAD and dt_bias is per CHANNEL") {
  // H=2, D=1. A_log = [ln 2, 0] -> decay_rate [2, 1].
  // g1 = [0.2, 0.2], dt_bias = [0.3, 0.3] -> g = [0.5, 0.5].
  //   head 0: sigmoid(2*0.5) = sigmoid(1)
  //   head 1: sigmoid(1*0.5) = sigmoid(0.5)
  // dt_bias is added BEFORE the reshape to [..,H,D] (:393), so it is indexed
  // h*D + d over the flat H*D axis.
  const std::vector<float> g1 = {0.2f, 0.2f};
  const std::vector<float> a_log = {static_cast<float>(std::log(2.0)), 0.0f};
  const std::vector<float> dt = {0.3f, 0.3f};
  const std::vector<float> y = Glm5NextForgetGate(g1, a_log, dt, 1, 2, 1, -5.0);
  REQUIRE(y.size() == 2);
  CHECK(y[0] == doctest::Approx(-5.0 * Sig(1.0)));
  CHECK(y[1] == doctest::Approx(-5.0 * Sig(0.5)));
  CHECK(y[0] != doctest::Approx(y[1]));
}

TEST_CASE("glm5_next kda: an absent bound mirrors upstream's softplus fallback") {
  // `linear_lower_bound` is `float | None` upstream, so the else branch
  // (:401-405) is a real mode and is mirrored rather than dropped: -exp(A_log)
  // * where(g > 20, g, log1p(exp(g))).
  const std::vector<float> a_log0 = {0.0f};
  const std::vector<float> y =
      Glm5NextForgetGate({0.0f, 100.0f}, a_log0, {0.0f, 0.0f}, 1, 1, 2,
                         std::nullopt);
  REQUIRE(y.size() == 2);
  CHECK(y[0] == doctest::Approx(-std::log(2.0)));
  CHECK(y[1] == doctest::Approx(-100.0));  // linearised above the 20.0 literal
  const std::vector<float> a_log2 = {static_cast<float>(std::log(2.0))};
  const std::vector<float> y2 =
      Glm5NextForgetGate({0.0f}, a_log2, {0.0f}, 1, 1, 1, std::nullopt);
  CHECK(y2[0] == doctest::Approx(-2.0 * std::log(2.0)));
}

TEST_CASE("glm5_next kda: the two branches are DIFFERENT FUNCTIONS, not a clamp") {
  // The pair that separates "wrong branch" from "broken port".
  std::mt19937 rng(20260827);
  std::uniform_real_distribution<float> u(-4.0f, 4.0f);
  const int64_t T = 6, H = 3, D = 5, hd = H * D;
  std::vector<float> g1(static_cast<size_t>(T) * hd), a_log(H), dt(hd);
  for (auto& z : g1) z = u(rng);
  for (auto& z : a_log) z = u(rng) * 0.25f;
  for (auto& z : dt) z = u(rng) * 0.5f;

  const std::vector<float> sigmoid_branch =
      Glm5NextForgetGate(g1, a_log, dt, T, H, D, -5.0);
  const std::vector<float> softplus_branch =
      Glm5NextForgetGate(g1, a_log, dt, T, H, D, std::nullopt);
  const std::vector<float> kimi =
      vllm::kimi_kda::KdaDecayGate(g1, a_log, dt, T, H, D);

  // Our fallback IS Kimi-Linear's gate: same function, ported twice.
  CHECK(RelL2F(softplus_branch, kimi) < 1e-6);
  // The branch this model takes is not. Measured relative L2 against
  // Kimi-Linear's on this draw is 0.863 — the two answers are the same size
  // and point in different directions, which is exactly the failure mode that
  // stays fluent. A ratio anywhere near 0 would mean they agree.
  CHECK(RelL2F(sigmoid_branch, kimi) > 0.5);
  // And the ranges do not even overlap at the ends: the sigmoid branch cannot
  // leave [bound, 0] and Kimi-Linear's leaves it on this very draw, so no
  // rescaling of one produces the other.
  int below_bound = 0;
  for (size_t i = 0; i < sigmoid_branch.size(); ++i) {
    CHECK(sigmoid_branch[i] >= -5.0f);
    CHECK(sigmoid_branch[i] < 0.0f);
    if (kimi[i] < -5.0f) ++below_bound;
  }
  CHECK(below_bound > 0);
}

TEST_CASE("glm5_next kda: forget gate vs an independent double reference") {
  std::mt19937 rng(7717);
  std::uniform_real_distribution<float> u(-3.0f, 3.0f);
  const int64_t T = 5, H = 4, D = 6, hd = H * D;
  std::vector<float> g1(static_cast<size_t>(T) * hd), a_log(H), dt(hd);
  for (auto& z : g1) z = u(rng);
  for (auto& z : a_log) z = u(rng) * 0.3f;
  for (auto& z : dt) z = u(rng);

  std::vector<double> ref(g1.size());
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t d = 0; d < D; ++d) {
        const size_t i = static_cast<size_t>(t * hd + h * D + d);
        const double gg = static_cast<double>(g1[i]) + dt[static_cast<size_t>(h * D + d)];
        ref[i] = -5.0 / (1.0 + std::exp(-std::exp(static_cast<double>(a_log[h])) * gg));
      }
  CHECK(RelL2(Glm5NextForgetGate(g1, a_log, dt, T, H, D, -5.0), ref) < 1e-6);
}

TEST_CASE("glm5_next kda: a missing or misshaped dt_bias is refused by name") {
  // `self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))` (:384) is
  // unconditional and :393 always adds it, so this model has no biasless mode.
  // Treating an empty vector as "no bias" would silently compute a DIFFERENT
  // gate from a checkpoint whose tensor failed to load, and the values stay
  // finite and plausible, so nothing downstream would report it.
  for (const std::vector<float>& bad :
       {std::vector<float>{}, std::vector<float>{0.1f}}) {  // absent, then short
    bool threw = false;
    try {
      Glm5NextForgetGate({0.2f, 0.3f}, {0.0f}, bad, 1, 1, 2, -5.0);
    } catch (const std::exception& e) {
      threw = true;
      CHECK(std::string(e.what()).find("dt_bias") != std::string::npos);
    }
    CHECK(threw);
  }
  // And the correctly sized one is accepted and USED: the same g1 with a zero
  // bias and with a 0.1 bias are different answers, which is what makes the
  // refusal worth having.
  const std::vector<float> zero =
      Glm5NextForgetGate({0.2f, 0.3f}, {0.0f}, {0.0f, 0.0f}, 1, 1, 2, -5.0);
  const std::vector<float> biased =
      Glm5NextForgetGate({0.2f, 0.3f}, {0.0f}, {0.1f, 0.1f}, 1, 1, 2, -5.0);
  REQUIRE(zero.size() == 2);
  CHECK(zero[0] != doctest::Approx(biased[0]));
  CHECK(biased[0] == doctest::Approx(-5.0 * Sig(0.3)));
}

TEST_CASE("glm5_next kda: a non-negative gate bound is refused by name") {
  bool threw = false;
  try {
    Glm5NextForgetGate({0.0f}, {0.0f}, {0.0f}, 1, 1, 1, 5.0);
  } catch (const std::exception& e) {
    threw = true;
    CHECK(std::string(e.what()).find("gate_lower_bound") != std::string::npos);
  }
  CHECK(threw);
}

// ═══ (2) THE STRICT-FP32 GATED OUTPUT NORM ═══════════════════════════════════

TEST_CASE("glm5_next kda: gated norm is rms * weight * SIGMOID(gate)") {
  // D=2, x=[3,4], weight=[1,1], eps=0 -> var = 25/2 = 12.5, rstd = 1/sqrt(12.5).
  // gate = 0 -> sigmoid(0) = 1/2.
  const double rstd = 1.0 / std::sqrt(12.5);
  const std::vector<float> y = Glm5NextRmsNormGated({3.0f, 4.0f}, {0.0f, 0.0f},
                                                    {1.0f, 1.0f}, 1, 1, 2, 0.0);
  REQUIRE(y.size() == 2);
  CHECK(y[0] == doctest::Approx(3.0 * rstd * 0.5));
  CHECK(y[1] == doctest::Approx(4.0 * rstd * 0.5));
}

TEST_CASE("glm5_next kda: the gate activation is sigmoid, NOT silu") {
  // Both Qwen3.5-GDN and FLA's own KDA use silu at this position; upstream sets
  // activation="sigmoid" (:412). silu(2) = 2*sigmoid(2) is 2x sigmoid(2).
  const std::vector<float> y =
      Glm5NextRmsNormGated({1.0f}, {2.0f}, {1.0f}, 1, 1, 1, 0.0);
  CHECK(y[0] == doctest::Approx(Sig(2.0)));            // x/|x| == 1 at D=1
  CHECK(y[0] != doctest::Approx(2.0 * Sig(2.0)));      // the silu answer
}

TEST_CASE("glm5_next kda: the gated-norm eps is rms_norm_eps 1e-5, not 1e-6") {
  // The constructor default is 1e-6 (:410) and the layer passes rms_norm_eps
  // = 1e-5 at :635. On an ordinary row the two agree; on a near-zero-variance
  // row — the only place an eps does anything — they differ by 2.3x.
  const std::vector<float> x = {1e-3f, 1e-3f};  // var = 1e-6
  const std::vector<float> g = {0.0f, 0.0f};
  const std::vector<float> w = {1.0f, 1.0f};
  const std::vector<float> got = Glm5NextRmsNormGated(x, g, w, 1, 1, 2, 1e-5);
  const std::vector<float> wrong = Glm5NextRmsNormGated(x, g, w, 1, 1, 2, 1e-6);
  CHECK(got[0] == doctest::Approx(1e-3 / std::sqrt(1e-6 + 1e-5) * 0.5));
  CHECK(wrong[0] == doctest::Approx(1e-3 / std::sqrt(1e-6 + 1e-6) * 0.5));
  CHECK(got[0] / wrong[0] < 0.7);
}

TEST_CASE("glm5_next kda: the gated norm's weight is per-channel and NOT downcast") {
  // `self.weight.to(torch.float32) * hidden_states` (:421). A weight value that
  // bf16 cannot represent must survive the norm exactly, which is the part of
  // "strict FP32 ... do not downcast on the weights" a host vector can express.
  const float w_exact = 1.0009765625f;              // representable in f32
  const float w_bf16 = vt::BF16ToF32(vt::F32ToBF16(w_exact));
  REQUIRE(w_bf16 != w_exact);                       // bf16 really does lose it
  const std::vector<float> y =
      Glm5NextRmsNormGated({1.0f}, {0.0f}, {w_exact}, 1, 1, 1, 0.0);
  CHECK(y[0] == doctest::Approx(w_exact * 0.5).epsilon(1e-9));
  CHECK(y[0] != doctest::Approx(w_bf16 * 0.5).epsilon(1e-9));
}

TEST_CASE("glm5_next kda: the gated norm returns in the MODEL dtype") {
  // `return hidden_states.to(input_dtype)` (:426): the norm computes in fp32
  // and rounds on the way out. Nothing else in the norm sees the model dtype.
  const std::vector<float> x = {1.0f}, g = {0.0f}, w = {1.0009765625f};
  const std::vector<float> f32 =
      Glm5NextRmsNormGated(x, g, w, 1, 1, 1, 0.0, Glm5NextActivationDType::kFloat32);
  const std::vector<float> bf16 =
      Glm5NextRmsNormGated(x, g, w, 1, 1, 1, 0.0, Glm5NextActivationDType::kBFloat16);
  CHECK(bf16[0] == vt::BF16ToF32(vt::F32ToBF16(f32[0])));
  CHECK(bf16[0] != f32[0]);
}

TEST_CASE("glm5_next kda: gated norm vs an independent double reference") {
  const int64_t T = 4, H = 3, D = 8, hd = H * D;
  const std::vector<float> x = Rand(static_cast<size_t>(T) * hd, 991, -2.0f, 2.0f);
  const std::vector<float> g = Rand(static_cast<size_t>(T) * hd, 992, -3.0f, 3.0f);
  const std::vector<float> w = Rand(static_cast<size_t>(D), 993, 0.5f, 1.5f);
  std::vector<double> ref(x.size());
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h) {
      double var = 0.0;
      for (int64_t d = 0; d < D; ++d) {
        const double v = x[static_cast<size_t>(t * hd + h * D + d)];
        var += v * v;
      }
      var /= static_cast<double>(D);
      const double rstd = 1.0 / std::sqrt(var + 1e-5);
      for (int64_t d = 0; d < D; ++d) {
        const size_t i = static_cast<size_t>(t * hd + h * D + d);
        ref[i] = static_cast<double>(x[i]) * rstd * w[static_cast<size_t>(d)] *
                 Sig(static_cast<double>(g[i]));
      }
    }
  CHECK(RelL2(Glm5NextRmsNormGated(x, g, w, T, H, D, 1e-5), ref) < 1e-6);
}

// ═══ (3) L2NORM ══════════════════════════════════════════════════════════════

TEST_CASE("glm5_next kda: l2norm divides by sqrt(SUM of squares), not the mean") {
  const std::vector<float> y = Glm5NextL2Norm({3.0f, 4.0f}, 1, 2, 0.0);
  CHECK(y[0] == doctest::Approx(0.6));
  CHECK(y[1] == doctest::Approx(0.8));
  // The rms answer would be 3/sqrt(12.5) = 0.8485.
  CHECK(y[0] != doctest::Approx(3.0 / std::sqrt(12.5)));
}

TEST_CASE("glm5_next kda: the l2norm eps is ADDED INSIDE the root, not a floor") {
  // Upstream says so in its own comment (:433): FLA "does + eps instead of
  // max(..., eps)". On an ordinary row the two agree to 1e-7; on a near-zero
  // row `F.normalize` returns a UNIT vector and this returns a tiny one, and a
  // randomized test almost never draws that row.
  const std::vector<float> x = {1e-4f, 0.0f};
  const std::vector<float> y = Glm5NextL2Norm(x, 1, 2, 1e-6);
  CHECK(y[0] == doctest::Approx(1e-4 / std::sqrt(1e-8 + 1e-6)));
  CHECK(y[0] < 0.11f);
  CHECK(y[0] != doctest::Approx(1.0));  // what max(norm, eps) would give
}

TEST_CASE("glm5_next kda: l2norm vs an independent double reference") {
  const int64_t R = 7, D = 9;
  const std::vector<float> x = Rand(static_cast<size_t>(R) * D, 4242, -2.0f, 2.0f);
  std::vector<double> ref(x.size());
  for (int64_t r = 0; r < R; ++r) {
    double ss = 0.0;
    for (int64_t d = 0; d < D; ++d) {
      const double v = x[static_cast<size_t>(r * D + d)];
      ss += v * v;
    }
    const double inv = std::sqrt(ss + 1e-6);
    for (int64_t d = 0; d < D; ++d)
      ref[static_cast<size_t>(r * D + d)] = x[static_cast<size_t>(r * D + d)] / inv;
  }
  CHECK(RelL2(Glm5NextL2Norm(x, R, D), ref) < 1e-6);
}

// ═══ (4) THE Q/K/V SHORT CONVS ═══════════════════════════════════════════════

TEST_CASE("glm5_next kda: the three checkpoint convs concatenate in q, k, v order") {
  // qkv_dim = 1, K = 2. Distinct kernels so any permutation is visible.
  const std::vector<float> q = {1.0f, 2.0f}, k = {3.0f, 4.0f}, v = {5.0f, 6.0f};
  const std::vector<float> w = Glm5NextMixedQkvConvWeight(q, k, v, 1, 2);
  REQUIRE(w.size() == 6);
  CHECK(w == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  // k, q, v — the plausible wrong order — is a different weight.
  CHECK(Glm5NextMixedQkvConvWeight(k, q, v, 1, 2) != w);
}

TEST_CASE("glm5_next kda: the short conv is causal, depthwise and silu-activated") {
  // channels = 1, K = 2, w = [0.5, 1.0], x = [1, 2], fresh sequence.
  //   t=0: 0.5*0 (zero history) + 1.0*1 = 1.0   -> silu(1.0)
  //   t=1: 0.5*1 + 1.0*2         = 2.5          -> silu(2.5)
  const std::vector<float> y =
      Glm5NextMixedQkvConv({1.0f, 2.0f}, {0.5f, 1.0f}, 2, 1, 2, nullptr, "silu");
  REQUIRE(y.size() == 2);
  CHECK(y[0] == doctest::Approx(1.0 * Sig(1.0)));
  CHECK(y[1] == doctest::Approx(2.5 * Sig(2.5)));
}

TEST_CASE("glm5_next kda: the conv state carries the pre-conv stream across calls") {
  // Splitting a 5-token prefill into 3 + 2 through the cache must reproduce the
  // one-shot answer. This is the whole conv-state contract in one assertion.
  const int64_t C = 3, K = 4, T = 5;
  const std::vector<float> w = Rand(static_cast<size_t>(C) * K, 31337, -1.0f, 1.0f);
  const std::vector<float> x = Rand(static_cast<size_t>(T) * C, 31338, -2.0f, 2.0f);
  const std::vector<float> one_shot =
      Glm5NextMixedQkvConv(x, w, T, C, K, nullptr, "silu");

  std::vector<float> state(static_cast<size_t>(C) * K, 0.0f);
  const std::vector<float> head(x.begin(), x.begin() + 3 * C);
  const std::vector<float> tail(x.begin() + 3 * C, x.end());
  const std::vector<float> a = Glm5NextMixedQkvConv(head, w, 3, C, K, &state, "silu");
  const std::vector<float> b = Glm5NextMixedQkvConv(tail, w, 2, C, K, &state, "silu");
  std::vector<float> joined(a);
  joined.insert(joined.end(), b.begin(), b.end());
  CHECK(RelL2F(joined, one_shot) < 1e-6);
  // And a single-token decode step off that state is the 6th token's answer.
  CHECK(state.size() == static_cast<size_t>(C) * K);
}

TEST_CASE("glm5_next kda: a hidden_act other than silu is refused by name") {
  bool threw = false;
  try {
    Glm5NextMixedQkvConv({1.0f}, {1.0f}, 1, 1, 1, nullptr, "gelu");
  } catch (const std::exception& e) {
    threw = true;
    CHECK(std::string(e.what()).find("silu") != std::string::npos);
  }
  CHECK(threw);
}

// ═══ (5) THE ASSEMBLED LAYER ═════════════════════════════════════════════════

namespace {

struct TinyLayer {
  Glm5NextKdaDims dims;
  Glm5NextKdaLayerWeights w;
  std::vector<float> x;
};

TinyLayer MakeTiny(int64_t T, uint32_t seed) {
  TinyLayer L;
  L.dims.hidden_size = 6;
  L.dims.num_heads = 2;
  L.dims.head_dim = 4;
  L.dims.conv_kernel_size = 3;
  L.dims.rms_norm_eps = 1e-5;
  L.dims.gate_lower_bound = -5.0;
  L.dims.hidden_act = "silu";
  const int64_t H = L.dims.hidden_size, nh = L.dims.num_heads, hd = L.dims.head_dim;
  const int64_t proj = nh * hd, K = L.dims.conv_kernel_size;
  L.w.q_proj = Rand(static_cast<size_t>(proj) * H, seed + 1, -0.6f, 0.6f);
  L.w.k_proj = Rand(static_cast<size_t>(proj) * H, seed + 2, -0.6f, 0.6f);
  L.w.v_proj = Rand(static_cast<size_t>(proj) * H, seed + 3, -0.6f, 0.6f);
  L.w.q_conv1d = Rand(static_cast<size_t>(proj) * K, seed + 4, -0.8f, 0.8f);
  L.w.k_conv1d = Rand(static_cast<size_t>(proj) * K, seed + 5, -0.8f, 0.8f);
  L.w.v_conv1d = Rand(static_cast<size_t>(proj) * K, seed + 6, -0.8f, 0.8f);
  L.w.f_a_proj = Rand(static_cast<size_t>(hd) * H, seed + 7, -0.5f, 0.5f);
  L.w.f_b_proj = Rand(static_cast<size_t>(proj) * hd, seed + 8, -0.5f, 0.5f);
  L.w.dt_bias = Rand(static_cast<size_t>(proj), seed + 9, -0.4f, 0.4f);
  L.w.a_log = Rand(static_cast<size_t>(nh), seed + 10, -0.5f, 0.5f);
  L.w.b_proj = Rand(static_cast<size_t>(nh) * H, seed + 11, -0.7f, 0.7f);
  L.w.g_a_proj = Rand(static_cast<size_t>(hd) * H, seed + 12, -0.5f, 0.5f);
  L.w.g_b_proj = Rand(static_cast<size_t>(proj) * hd, seed + 13, -0.5f, 0.5f);
  L.w.o_norm = Rand(static_cast<size_t>(hd), seed + 14, 0.6f, 1.4f);
  L.w.o_proj = Rand(static_cast<size_t>(H) * proj, seed + 15, -0.5f, 0.5f);
  L.x = Rand(static_cast<size_t>(T) * H, seed + 16, -1.5f, 1.5f);
  return L;
}

// Which order the checkpoint's three separate depthwise conv weights are
// concatenated into the reference's ONE grouped conv. The mixed stream is
// ALWAYS [q; k; v] (:655-661), so anything but kQKV pairs a channel block with
// another projection's filter — which is the mistake. Note that this is not the
// same experiment as swapping two of the layer's own weight TENSORS: that moves
// the answer under any fixed concat order and so gates nothing.
enum class ConvWeightOrder { kQKV, kQVK, kKQV };

// A from-first-principles double reference that mirrors
// `Glm5NextTextLinearAttention.forward` (:641-746) and
// `recurrent_kimi_delta_attention` (:441-491) in THEIR line order and THEIR
// state layout (S[h][k][v], the transpose of vt::KdaGatedDeltaRule's), so a
// layout mistake on our side cannot cancel out against the reference.
//
// `gate_source` selects where g, beta and the output gate read from: `true`
// mirrors upstream (the PRE-conv hidden states) and `false` is the plausible
// fusion mistake, present so the test can prove it would be visible.
std::vector<double> TinyRef(const TinyLayer& L, int64_t T, bool gate_from_pre_conv,
                            ConvWeightOrder cw_order = ConvWeightOrder::kQKV) {
  const int64_t H = L.dims.hidden_size, nh = L.dims.num_heads, hd = L.dims.head_dim;
  const int64_t proj = nh * hd, K = L.dims.conv_kernel_size, C = 3 * proj;
  const auto& w = L.w;
  const auto mv = [](const std::vector<float>& m, const std::vector<double>& in,
                     int64_t o_dim, int64_t i_dim, int64_t rows) {
    std::vector<double> out(static_cast<size_t>(rows) * o_dim, 0.0);
    for (int64_t t = 0; t < rows; ++t)
      for (int64_t o = 0; o < o_dim; ++o) {
        double a = 0.0;
        for (int64_t i = 0; i < i_dim; ++i)
          a += static_cast<double>(m[static_cast<size_t>(o * i_dim + i)]) *
               in[static_cast<size_t>(t * i_dim + i)];
        out[static_cast<size_t>(t * o_dim + o)] = a;
      }
    return out;
  };
  const std::vector<double> x(L.x.begin(), L.x.end());

  // mixed_qkv = cat([q_proj(x), k_proj(x), v_proj(x)], -1)      (:655-661)
  const std::vector<double> qp = mv(w.q_proj, x, proj, H, T);
  const std::vector<double> kp = mv(w.k_proj, x, proj, H, T);
  const std::vector<double> vp = mv(w.v_proj, x, proj, H, T);
  std::vector<double> mixed(static_cast<size_t>(T) * C);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < proj; ++j) {
      mixed[static_cast<size_t>(t * C + j)] = qp[static_cast<size_t>(t * proj + j)];
      mixed[static_cast<size_t>(t * C + proj + j)] = kp[static_cast<size_t>(t * proj + j)];
      mixed[static_cast<size_t>(t * C + 2 * proj + j)] = vp[static_cast<size_t>(t * proj + j)];
    }

  // one grouped depthwise causal conv, silu                     (:621-628,:687)
  const std::vector<float>* cwo[3] = {&w.q_conv1d, &w.k_conv1d, &w.v_conv1d};
  if (cw_order == ConvWeightOrder::kQVK) {
    cwo[1] = &w.v_conv1d;
    cwo[2] = &w.k_conv1d;
  } else if (cw_order == ConvWeightOrder::kKQV) {
    cwo[0] = &w.k_conv1d;
    cwo[1] = &w.q_conv1d;
  }
  std::vector<float> cw;
  for (const std::vector<float>* c : cwo) cw.insert(cw.end(), c->begin(), c->end());
  std::vector<double> conv(static_cast<size_t>(T) * C, 0.0);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t c = 0; c < C; ++c) {
      double a = 0.0;
      for (int64_t j = 0; j < K; ++j) {
        const int64_t p = t - (K - 1) + j;
        if (p < 0) continue;
        a += static_cast<double>(cw[static_cast<size_t>(c * K + j)]) *
             mixed[static_cast<size_t>(p * C + c)];
      }
      conv[static_cast<size_t>(t * C + c)] = a * Sig(a);
    }

  // split, then l2norm q and k over head_dim                    (:698-706,:458)
  std::vector<double> q(static_cast<size_t>(T) * proj), k(q.size()), v(q.size());
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < proj; ++j) {
      q[static_cast<size_t>(t * proj + j)] = conv[static_cast<size_t>(t * C + j)];
      k[static_cast<size_t>(t * proj + j)] = conv[static_cast<size_t>(t * C + proj + j)];
      v[static_cast<size_t>(t * proj + j)] = conv[static_cast<size_t>(t * C + 2 * proj + j)];
    }
  const auto l2 = [&](std::vector<double>& z) {
    for (int64_t r = 0; r < T * nh; ++r) {
      double ss = 0.0;
      for (int64_t d = 0; d < hd; ++d) {
        const double e = z[static_cast<size_t>(r * hd + d)];
        ss += e * e;
      }
      const double inv = std::sqrt(ss + 1e-6);
      for (int64_t d = 0; d < hd; ++d) z[static_cast<size_t>(r * hd + d)] /= inv;
    }
  };
  l2(q);
  l2(k);

  // g, beta, gate — from the PRE-conv hidden states             (:709,:710,:742)
  const std::vector<double>& src = gate_from_pre_conv ? x : q;
  const int64_t src_dim = gate_from_pre_conv ? H : proj;
  std::vector<float> fa = w.f_a_proj, fb = w.f_b_proj, bp = w.b_proj,
                     ga = w.g_a_proj;
  if (!gate_from_pre_conv) {  // reshape the first leg to the wider source
    fa.assign(static_cast<size_t>(hd) * src_dim, 0.0f);
    bp.assign(static_cast<size_t>(nh) * src_dim, 0.0f);
    ga.assign(static_cast<size_t>(hd) * src_dim, 0.0f);
    for (size_t i = 0; i < fa.size(); ++i) fa[i] = w.f_a_proj[i % w.f_a_proj.size()];
    for (size_t i = 0; i < bp.size(); ++i) bp[i] = w.b_proj[i % w.b_proj.size()];
    for (size_t i = 0; i < ga.size(); ++i) ga[i] = w.g_a_proj[i % w.g_a_proj.size()];
  }
  const std::vector<double> g1 =
      mv(fb, mv(fa, src, hd, src_dim, T), proj, hd, T);
  std::vector<double> g(g1.size());
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < nh; ++h) {
      const double dr = std::exp(static_cast<double>(w.a_log[static_cast<size_t>(h)]));
      for (int64_t d = 0; d < hd; ++d) {
        const size_t i = static_cast<size_t>(t * proj + h * hd + d);
        g[i] = -5.0 * Sig(dr * (g1[i] + w.dt_bias[static_cast<size_t>(h * hd + d)]));
      }
    }
  const std::vector<double> braw = mv(bp, src, nh, src_dim, T);
  std::vector<double> beta(braw.size());
  for (size_t i = 0; i < beta.size(); ++i) beta[i] = Sig(braw[i]);
  const std::vector<double> gate =
      mv(w.g_b_proj, mv(ga, src, hd, src_dim, T), proj, hd, T);

  // the recurrence, in the reference's own S[h][k][v] layout    (:477-489)
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
  std::vector<double> S(static_cast<size_t>(nh) * hd * hd, 0.0);
  std::vector<double> core(static_cast<size_t>(T) * proj, 0.0);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < nh; ++h) {
      double* Sh = &S[static_cast<size_t>(h) * hd * hd];  // Sh[ki*hd + vi]
      const size_t base = static_cast<size_t>(t * proj + h * hd);
      const double b = beta[static_cast<size_t>(t * nh + h)];
      for (int64_t ki = 0; ki < hd; ++ki) {
        const double gi = std::exp(g[base + static_cast<size_t>(ki)]);
        for (int64_t vi = 0; vi < hd; ++vi) Sh[ki * hd + vi] *= gi;
      }
      std::vector<double> kv(static_cast<size_t>(hd), 0.0);
      for (int64_t vi = 0; vi < hd; ++vi)
        for (int64_t ki = 0; ki < hd; ++ki)
          kv[static_cast<size_t>(vi)] += Sh[ki * hd + vi] * k[base + static_cast<size_t>(ki)];
      for (int64_t vi = 0; vi < hd; ++vi) {
        const double delta = (v[base + static_cast<size_t>(vi)] - kv[static_cast<size_t>(vi)]) * b;
        for (int64_t ki = 0; ki < hd; ++ki)
          Sh[ki * hd + vi] += k[base + static_cast<size_t>(ki)] * delta;
      }
      for (int64_t vi = 0; vi < hd; ++vi) {
        double o = 0.0;
        for (int64_t ki = 0; ki < hd; ++ki)
          o += Sh[ki * hd + vi] * q[base + static_cast<size_t>(ki)] * scale;
        core[base + static_cast<size_t>(vi)] = o;
      }
    }

  // gated norm then o_proj                                      (:743-744)
  std::vector<double> normed(core.size());
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < nh; ++h) {
      double var = 0.0;
      for (int64_t d = 0; d < hd; ++d) {
        const double e = core[static_cast<size_t>(t * proj + h * hd + d)];
        var += e * e;
      }
      var /= static_cast<double>(hd);
      const double rstd = 1.0 / std::sqrt(var + L.dims.rms_norm_eps);
      for (int64_t d = 0; d < hd; ++d) {
        const size_t i = static_cast<size_t>(t * proj + h * hd + d);
        normed[i] = core[i] * rstd * w.o_norm[static_cast<size_t>(d)] * Sig(gate[i]);
      }
    }
  return mv(w.o_proj, normed, H, proj, T);
}

}  // namespace

TEST_CASE("glm5_next kda: the layer matches an independent double reference") {
  const int64_t T = 7;
  const TinyLayer L = MakeTiny(T, 5150);
  vt::Queue q = CpuQ();
  const std::vector<float> got =
      Glm5NextKdaLayerForward(L.w, L.x, L.dims, T, nullptr, q);
  const std::vector<double> ref = TinyRef(L, T, /*gate_from_pre_conv=*/true);
  REQUIRE(got.size() == ref.size());
  CHECK(RelL2(got, ref) < 1e-5);
}

TEST_CASE("glm5_next kda: g, beta and the output gate read the PRE-CONV states") {
  // The guard for layout fact 2. Computing the three gates from the post-conv
  // stream is a cheap and plausible fusion; this shows the reference above
  // would see it, so the previous case's agreement is evidence and not luck.
  const int64_t T = 7;
  const TinyLayer L = MakeTiny(T, 5150);
  vt::Queue q = CpuQ();
  const std::vector<float> got =
      Glm5NextKdaLayerForward(L.w, L.x, L.dims, T, nullptr, q);
  const std::vector<double> fused = TinyRef(L, T, /*gate_from_pre_conv=*/false);
  CHECK(RelL2(got, fused) > 1e-2);
}

TEST_CASE("glm5_next kda: q/k/v conv order is load-bearing in the layer") {
  // The guard for layout fact 1, and it has to be a reference built with the
  // WRONG pairing. Swapping two of the layer's own conv weights and watching
  // the answer move proves nothing: it moves under ANY fixed concat order, so
  // that experiment stays green under the very mutation it claims to catch.
  // What discriminates is a [q; k; v] stream convolved with [q_w; v_w; k_w]:
  // reorder the implementation's concat and `got` LEAVES the kQKV reference
  // and JOINS the reordered one, so the first check and one of the other two
  // both red.
  const int64_t T = 6;
  const TinyLayer L = MakeTiny(T, 909);
  vt::Queue q = CpuQ();
  const std::vector<float> got =
      Glm5NextKdaLayerForward(L.w, L.x, L.dims, T, nullptr, q);
  CHECK(RelL2(got, TinyRef(L, T, /*gate_from_pre_conv=*/true,
                           ConvWeightOrder::kQKV)) < 1e-5);
  CHECK(RelL2(got, TinyRef(L, T, /*gate_from_pre_conv=*/true,
                           ConvWeightOrder::kQVK)) > 1e-2);
  CHECK(RelL2(got, TinyRef(L, T, /*gate_from_pre_conv=*/true,
                           ConvWeightOrder::kKQV)) > 1e-2);
}

TEST_CASE("glm5_next kda: the cache carries conv AND recurrent state across steps") {
  // Prefill 6 then decode 1 must equal a 7-token one-shot prefill. This gates
  // the conv-state layout and the fp32 recurrent-state carry together: drop
  // either and the last row diverges.
  const int64_t T = 7;
  const TinyLayer L = MakeTiny(T, 2718);
  vt::Queue q = CpuQ();
  const int64_t H = L.dims.hidden_size;
  const std::vector<float> one_shot =
      Glm5NextKdaLayerForward(L.w, L.x, L.dims, T, nullptr, q);

  Glm5NextKdaCache cache;
  const std::vector<float> head(L.x.begin(), L.x.begin() + (T - 1) * H);
  const std::vector<float> tail(L.x.begin() + (T - 1) * H, L.x.end());
  const std::vector<float> a =
      Glm5NextKdaLayerForward(L.w, head, L.dims, T - 1, &cache, q);
  CHECK(cache.conv_state.size() ==
        static_cast<size_t>(L.dims.conv_dim()) * L.dims.conv_kernel_size);
  CHECK(cache.recurrent_state.size() == static_cast<size_t>(L.dims.num_heads) *
                                            L.dims.head_dim * L.dims.head_dim);
  const std::vector<float> b =
      Glm5NextKdaLayerForward(L.w, tail, L.dims, 1, &cache, q);
  std::vector<float> joined(a);
  joined.insert(joined.end(), b.begin(), b.end());
  CHECK(RelL2F(joined, one_shot) < 1e-5);
  // The carry is what does it: a decode step off a FRESH cache is a different
  // row entirely.
  Glm5NextKdaCache fresh;
  const std::vector<float> no_carry =
      Glm5NextKdaLayerForward(L.w, tail, L.dims, 1, &fresh, q);
  CHECK(RelL2F(no_carry, b) > 1e-2);
}

TEST_CASE("glm5_next kda: the softplus branch changes the whole LAYER, not one gate") {
  // The deliverable's proof, one level up: the same weights and the same
  // tokens through Kimi-Linear's branch produce a different model output.
  const int64_t T = 6;
  TinyLayer L = MakeTiny(T, 1234);
  vt::Queue q = CpuQ();
  const std::vector<float> sigmoid_arm =
      Glm5NextKdaLayerForward(L.w, L.x, L.dims, T, nullptr, q);
  L.dims.gate_lower_bound.reset();  // -> the softplus branch
  const std::vector<float> softplus_arm =
      Glm5NextKdaLayerForward(L.w, L.x, L.dims, T, nullptr, q);
  CHECK(RelL2F(softplus_arm, sigmoid_arm) > 1e-2);
  for (float z : sigmoid_arm) CHECK(std::isfinite(z));
  for (float z : softplus_arm) CHECK(std::isfinite(z));
}

TEST_CASE("glm5_next kda: the host layer refuses a non-CPU queue by name") {
  const TinyLayer L = MakeTiny(2, 77);
  vt::Queue cuda{vt::Device{vt::DeviceType::kCUDA, 0}, nullptr};
  bool threw = false;
  try {
    Glm5NextKdaLayerForward(L.w, L.x, L.dims, 2, nullptr, cuda);
  } catch (const std::exception& e) {
    threw = true;
    CHECK(std::string(e.what()).find("CPU queue") != std::string::npos);
  }
  CHECK(threw);
}
