// Qwen4-Exp (Qwen3.8-Flash-Next) W3 UNIT GATE — the 4-branch GATED-RESIDUAL
// hyper-connection stream and the grouped RMSNorm it stands on.
// Issue #1988, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT THE GOLDENS ARE. `qwen4_exp_hc_goldens.inc` is dumped by
// `scripts/gen-qwen4-exp-hc-goldens.py`, which lifts `Qwen4ExpTextRMSNorm`
// (:158-181) and `Qwen4ExpTextGatedResidual` (:941-969) VERBATIM by line range
// out of the lane-pinned oracle — transformers **v5.16.0**
// `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`, sha256
// 77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459 — and
// EXECUTES them under torch. Nothing was retyped, and the script re-checks that
// sha256 before it runs. The one documented harness adaptation is a four-field
// stand-in for `Qwen4ExpTextConfig`: the real dataclass drags in the whole
// transformers package, and the classes read only `hidden_size`, `hc_count`,
// `hc_lowrank` and `rms_norm_eps`. The write-back goldens replay the two
// verbatim lines of `Qwen4ExpTextDecoderLayer.forward`.
//
// WHY A DISCRIMINATION SWEEP AND NOT ONLY A GOLDEN COMPARE. Five of this
// module's guarantees are single-character defects that a plausible port makes
// silently, and a golden that cannot see them is decoration. `Variant` below is
// an INDEPENDENT double-precision reference with one flag per defect; the test
// asserts the correct spelling matches the oracle AND that each single-flag flip
// does NOT, which measures the goldens' discriminating power rather than
// assuming it (AGENTS.md "Gates", `.agents/verification.md`).
//
// SCOPE, HONESTLY. Host reference only. No token claim, no speed claim: no
// `qwen4_exp` arm runs anywhere yet (the smallest published checkpoint is
// ~128 GB against a ~119 GB budget, spec `## Hardware`), and W1 config
// registration is still in review. What this gate proves is the arithmetic.
#include "vllm/model_executor/models/qwen4_exp_hc.h"

#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "support/max_abs_diff.h"
#include "vllm/model_executor/models/deepseek_v4_mhc.h"

using namespace vllm::qwen4_exp;
using vllm_test::MaxAbsDiff;

namespace {

#include "qwen4_exp_hc_goldens.inc"  // NOLINT — golden literals

// The goldens are fp32 out of torch; this reference is fp32 with double-precision
// reductions (the `deepseek_v4_mhc.cpp` house convention for a host reference).
constexpr double kTol = 1e-5;

// The floor a Variant flip has to clear. It is not a round number picked to be
// safe: it is 6.6x under the SMALLEST separation actually measured, and 100x
// over kTol, so the two bands cannot touch. Measured on case A, max|diff| of the
// flipped double reference against the oracle goldens, per token:
//
//   flip              t=0        t=1        t=2      affects
//   silu-div-after    1.79e-2    6.63e-3    9.54e-3  mixed_input
//   div-on-up         1.97e-2    1.95e-2    1.81e-2  mixed_input
//   sum-not-mean      1.03e+0    6.91e-1    1.17e+0  mixed_input
//   raw-not-normed    4.77e-1    3.39e-1    5.66e-1  mixed_input
//   inject-no-div     5.22e-1    2.82e-1    5.49e-1  injection_weights
//
// The SiLU-division placement is the narrowest of the five, which is exactly the
// reason it is worth a gate: it is a one-token-deep edit that moves the answer
// by well under a percent at some tokens and would survive a loose eyeball.
constexpr double kFlipFloor = 1e-3;

std::vector<float> Load(const float* p, size_t n) { return std::vector<float>(p, p + n); }

// ─── The independent double-precision reference, with one flag per known trap ──
struct Variant {
  bool div_before_silu = true;   // silu(down(x)/hc), NOT silu(down(x))/hc
  bool div_on_up = false;        // there is NO division on the up-projection
  bool reduce_is_mean = true;    // .mean(dim=-2), NOT .sum
  bool multiply_normed = true;   // against hyper_input_normed, NOT hyper_input
  bool inject_div = true;        // 2*sigmoid(inject(x)/hc)
  bool norm_is_grouped = true;   // hc independent reductions, NOT one global one
  bool norm_weight_plus_one = true;  // (1.0 + w_hf)
};

double SigmoidD(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::vector<double> NormRefD(const std::vector<float>& x, const std::vector<float>& w_hf,
                             int64_t hc, int64_t hidden, double eps, const Variant& v) {
  std::vector<double> out(x.size());
  const int64_t groups = v.norm_is_grouped ? hc : 1;
  const int64_t gsize = v.norm_is_grouped ? hidden : hc * hidden;
  for (int64_t g = 0; g < groups; ++g) {
    double ss = 0.0;
    for (int64_t d = 0; d < gsize; ++d) {
      const double t = x[g * gsize + d];
      ss += t * t;
    }
    const double r = 1.0 / std::sqrt(ss / static_cast<double>(gsize) + eps);
    for (int64_t d = 0; d < gsize; ++d) {
      const double scale = v.norm_weight_plus_one ? 1.0 + w_hf[g * gsize + d] : w_hf[g * gsize + d];
      out[g * gsize + d] = x[g * gsize + d] * r * scale;
    }
  }
  return out;
}

std::vector<double> LinearD(const float* w, const std::vector<double>& x, int64_t out_dim,
                            int64_t in_dim) {
  std::vector<double> y(static_cast<size_t>(out_dim), 0.0);
  for (int64_t o = 0; o < out_dim; ++o) {
    double acc = 0.0;
    for (int64_t i = 0; i < in_dim; ++i) acc += static_cast<double>(w[o * in_dim + i]) * x[i];
    y[o] = acc;
  }
  return y;
}

struct RefOut {
  std::vector<double> normed;
  std::vector<double> mixed;
  std::vector<double> injection;
};

RefOut ForwardRefD(const std::vector<float>& hyper, const std::vector<float>& w_hf,
                   const float* down, const float* up, const float* inject, int64_t hc,
                   int64_t hidden, int64_t rank, double eps, const Variant& v) {
  const int64_t flat = hc * hidden;
  RefOut o;
  o.normed = NormRefD(hyper, w_hf, hc, hidden, eps, v);

  std::vector<double> lo = LinearD(down, o.normed, rank, flat);
  for (int64_t r = 0; r < rank; ++r) {
    const double a = v.div_before_silu ? lo[r] / static_cast<double>(hc) : lo[r];
    const double s = a * SigmoidD(a);
    lo[r] = v.div_before_silu ? s : s / static_cast<double>(hc);
  }
  std::vector<double> gate = LinearD(up, lo, flat, rank);
  for (int64_t p = 0; p < flat; ++p) {
    gate[p] = SigmoidD(v.div_on_up ? gate[p] / static_cast<double>(hc) : gate[p]);
  }

  o.mixed.assign(static_cast<size_t>(hidden), 0.0);
  for (int64_t j = 0; j < hc; ++j) {
    for (int64_t h = 0; h < hidden; ++h) {
      const double stream = v.multiply_normed ? o.normed[j * hidden + h] : hyper[j * hidden + h];
      o.mixed[h] += gate[j * hidden + h] * stream;
    }
  }
  if (v.reduce_is_mean) {
    for (int64_t h = 0; h < hidden; ++h) o.mixed[h] /= static_cast<double>(hc);
  }

  if (inject != nullptr) {
    std::vector<double> logits = LinearD(inject, o.normed, hc, flat);
    o.injection.resize(static_cast<size_t>(hc));
    for (int64_t j = 0; j < hc; ++j) {
      o.injection[j] =
          2.0 * SigmoidD(v.inject_div ? logits[j] / static_cast<double>(hc) : logits[j]);
    }
  }
  return o;
}

// Compare an fp32 result against the double reference directly, for the cases
// where no oracle golden exists (a deliberately huge eps).
double MaxAbsDV(const std::vector<float>& got, const std::vector<double>& want) {
  REQUIRE(got.size() == want.size());
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - want[i]));
  }
  return worst;
}

double MaxAbsD(const std::vector<double>& got, const float* want) {
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(got[i] - static_cast<double>(want[i])));
  }
  return worst;
}

}  // namespace

// ── 1. The grouped RMSNorm, against vLLM's `RMSNormGated(group_size=)` form ────
// vLLM `model_executor/layers/layernorm.py` @ origin/main 6a5e8f5979: `RMSNorm`
// opens at :37 and has no group knob; `RMSNormGated` opens at :172 with
// `group_size` at :187, and its grouped branch is :258-264 —
// `rearrange(x, "... (g d) -> ... g d")`, `variance = x_group.pow(2).mean(-1)`,
// `x_normed = x_group * rsqrt(variance + eps)`, `out = flatten * weight`. That is
// the op we mirror; the SEMANTICS (which weight, which eps) come from
// transformers, and the two differ by exactly the `1 + w` transform below.
TEST_CASE("qwen4_exp grouped RMSNorm mirrors RMSNormGated(group_size) at the lane pin") {
  const int64_t hc = 4, hidden = 6;
  const std::vector<float> hyper = Load(kA_hyper, static_cast<size_t>(hc * hidden));
  const std::vector<float> w_hf = Load(kA_norm_w_hf, static_cast<size_t>(hc * hidden));

  const std::vector<float> w = HcNormWeightFromHf(w_hf);
  const std::vector<float> got = GroupedRmsNorm(hyper, w, hidden, 1e-6f);
  CHECK(MaxAbsDiff(got, kA_normed, hyper.size()) < kTol);

  // A GLOBAL reduction over all 24 elements is the defect a plain RMSNorm port
  // makes. It must be visibly different, or the case above proves nothing.
  Variant global;
  global.norm_is_grouped = false;
  CHECK(MaxAbsD(NormRefD(hyper, w_hf, hc, hidden, 1e-6, global), kA_normed) > 1e-2);

  // eps lives INSIDE the rsqrt, added to the mean square. At the model's real
  // 1e-6 the two placements differ by ~1e-6, which is UNDER this suite's fp32
  // tolerance — so an eps-placement defect is invisible to every golden case
  // above, and a case at an eps large enough to separate them is the only thing
  // that gates it. `rsqrt(m + 4)` and `1/(sqrt(m) + 4)` are far apart; the
  // double reference puts eps inside, so this pins the placement rather than
  // merely observing that eps does something.
  const std::vector<float> big_eps = GroupedRmsNorm(hyper, w, hidden, 4.0f);
  CHECK(MaxAbsDiff(big_eps, got) > 1e-2);
  CHECK(MaxAbsDV(big_eps, NormRefD(hyper, w_hf, hc, hidden, 4.0, Variant{})) < kTol);
}

TEST_CASE("qwen4_exp hc_norm weight is 1 + w_hf, applied exactly once") {
  const std::vector<float> w_hf = Load(kA_norm_w_hf, 24);
  const std::vector<float> w = HcNormWeightFromHf(w_hf);
  REQUIRE(w.size() == w_hf.size());
  for (size_t i = 0; i < w.size(); ++i) CHECK(w[i] == doctest::Approx(1.0f + w_hf[i]));

  const std::vector<float> hyper = Load(kA_hyper, 24);
  // The HF weight used raw is the "forgot the +1" defect: near-zero scale, which
  // reads as a checkpoint bug rather than a port bug (spec, trap 2).
  CHECK(MaxAbsDiff(GroupedRmsNorm(hyper, w_hf, 6, 1e-6f), kA_normed, 24) > 1e-1);
  // Adding 1 a SECOND time is the defect a reader of the published GGUF makes,
  // because the convert step has already folded it (issue #1988, trap 2).
  CHECK(MaxAbsDiff(GroupedRmsNorm(hyper, HcNormWeightFromHf(w), 6, 1e-6f), kA_normed, 24) > 1e-1);
}

TEST_CASE("qwen4_exp grouped RMSNorm refuses an indivisible width, as upstream does") {
  // The MESSAGE is asserted, not merely the type. Every guard in this TU throws
  // `std::invalid_argument`, so a type-only assertion passes when the guard that
  // fires is a DIFFERENT one further down — which is exactly what a mutation
  // sweep found: deleting a refusal left the suite green because a later bounds
  // check threw the same type. Pinning the text makes each guard its own gate.
  const std::vector<float> x(10, 1.0f), w(10, 1.0f);
  CHECK_THROWS_WITH_AS(GroupedRmsNorm(x, w, 4, 1e-6f),
                       "qwen4_exp: hidden_size (10) must be divisible by group_size (4).",
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(GroupedRmsNorm(x, std::vector<float>(9, 1.0f), 5, 1e-6f),
                       "qwen4_exp: hc_norm weight has 9 elements, expected 10.",
                       std::invalid_argument);
}

// ── 2. The gated-residual READ, both arms ─────────────────────────────────────
TEST_CASE("qwen4_exp GatedResidual read matches transformers v5.16.0, use_combine=true") {
  const int64_t hc = 4, hidden = 6, rank = 5, flat = hc * hidden;
  GatedResidualWeights w;
  w.hc_norm_weight = HcNormWeightFromHf(Load(kA_norm_w_hf, static_cast<size_t>(flat)));
  w.mix_down = Load(kA_down, static_cast<size_t>(rank * flat));
  w.mix_up = Load(kA_up, static_cast<size_t>(flat * rank));
  w.block_inject = Load(kA_inject, static_cast<size_t>(hc * flat));

  for (int64_t t = 0; t < 3; ++t) {
    const std::vector<float> hyper = Load(kA_hyper + t * flat, static_cast<size_t>(flat));
    const GatedResidualResult r = GatedResidualForward(hyper, w, hc, hidden, 1e-6f);
    CHECK(MaxAbsDiff(r.hyper_input_normed, kA_normed + t * flat, static_cast<size_t>(flat)) <
          kTol);
    CHECK(MaxAbsDiff(r.mixed_input, kA_mixed + t * hidden, static_cast<size_t>(hidden)) < kTol);
    CHECK(MaxAbsDiff(r.injection_weights, kA_inj_w + t * hc, static_cast<size_t>(hc)) < kTol);
    // Range of 2*sigmoid is (0, 2) with 1.0 at a zero logit.
    for (float v : r.injection_weights) CHECK((v > 0.0f && v < 2.0f));
  }
}

TEST_CASE("qwen4_exp GatedResidual read matches transformers v5.16.0 at hc_count=3") {
  // hc_count is a config value, not the constant 4. A port that folds 4 into the
  // two divisions passes case A and fails here.
  const int64_t hc = 3, hidden = 5, rank = 7, flat = hc * hidden;
  GatedResidualWeights w;
  w.hc_norm_weight = HcNormWeightFromHf(Load(kB_norm_w_hf, static_cast<size_t>(flat)));
  w.mix_down = Load(kB_down, static_cast<size_t>(rank * flat));
  w.mix_up = Load(kB_up, static_cast<size_t>(flat * rank));
  w.block_inject = Load(kB_inject, static_cast<size_t>(hc * flat));

  for (int64_t t = 0; t < 2; ++t) {
    const std::vector<float> hyper = Load(kB_hyper + t * flat, static_cast<size_t>(flat));
    const GatedResidualResult r = GatedResidualForward(hyper, w, hc, hidden, 1e-5f);
    CHECK(MaxAbsDiff(r.hyper_input_normed, kB_normed + t * flat, static_cast<size_t>(flat)) <
          kTol);
    CHECK(MaxAbsDiff(r.mixed_input, kB_mixed + t * hidden, static_cast<size_t>(hidden)) < kTol);
    CHECK(MaxAbsDiff(r.injection_weights, kB_inj_w + t * hc, static_cast<size_t>(hc)) < kTol);

    const std::vector<float> block = Load(kB_block_out + t * hidden, static_cast<size_t>(hidden));
    CHECK(MaxAbsDiff(GatedResidualWriteBack(hyper, block, r.injection_weights, hc, hidden),
                     kB_written + t * flat, static_cast<size_t>(flat)) < kTol);
  }
}

TEST_CASE("qwen4_exp GatedResidual with use_combine=false is the final mixer") {
  const int64_t hc = 4, hidden = 6, rank = 5, flat = hc * hidden;
  GatedResidualWeights w;
  w.hc_norm_weight = HcNormWeightFromHf(Load(kC_norm_w_hf, static_cast<size_t>(flat)));
  w.mix_down = Load(kC_down, static_cast<size_t>(rank * flat));
  w.mix_up = Load(kC_up, static_cast<size_t>(flat * rank));
  w.block_inject.clear();  // == `block_inject_weight = None`

  for (int64_t t = 0; t < 2; ++t) {
    const std::vector<float> hyper = Load(kC_hyper + t * flat, static_cast<size_t>(flat));
    const GatedResidualResult r = GatedResidualForward(hyper, w, hc, hidden, 1e-6f);
    CHECK(MaxAbsDiff(r.hyper_input_normed, kC_normed + t * flat, static_cast<size_t>(flat)) <
          kTol);
    CHECK(MaxAbsDiff(r.mixed_input, kC_mixed + t * hidden, static_cast<size_t>(hidden)) < kTol);
    // `Qwen4ExpTextGatedResidual.forward` returns EARLY here: no injection branch
    // is computed at all, and there is no trailing model norm after it either
    // (`Qwen4ExpTextModel.forward` goes straight to `lm_head` — spec, trap 1).
    CHECK(r.injection_weights.empty());
    CHECK(r.mixed_input.size() == static_cast<size_t>(hidden));
  }
}

TEST_CASE("qwen4_exp GatedResidual refuses a stream that is not hc_count * hidden_size") {
  GatedResidualWeights w;
  w.hc_norm_weight.assign(24, 1.0f);
  w.mix_down.assign(5 * 24, 0.0f);
  w.mix_up.assign(24 * 5, 0.0f);
  // Message-pinned for the reason given on the grouped-norm refusal above: with
  // a type-only assertion the 23- and 25-wide cases were caught downstream by
  // GroupedRmsNorm's divisibility guard, and deleting THIS refusal changed
  // nothing the suite could see. A 30-wide stream is divisible by hidden_size,
  // so it walks past that guard too and reaches the weight-size check instead.
  CHECK_THROWS_WITH_AS(GatedResidualForward(std::vector<float>(23, 1.0f), w, 4, 6, 1e-6f),
                       "qwen4_exp: expected 24 hyper-connection features, got 23.",
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(GatedResidualForward(std::vector<float>(25, 1.0f), w, 4, 6, 1e-6f),
                       "qwen4_exp: expected 24 hyper-connection features, got 25.",
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(GatedResidualForward(std::vector<float>(30, 1.0f), w, 4, 6, 1e-6f),
                       "qwen4_exp: expected 24 hyper-connection features, got 30.",
                       std::invalid_argument);
}

// ── 3. The rank-1 write-back ──────────────────────────────────────────────────
TEST_CASE("qwen4_exp write-back is hyper_input RAW plus outer(block_out, injection)") {
  const int64_t hc = 4, hidden = 6, flat = hc * hidden;
  for (int64_t t = 0; t < 3; ++t) {
    const std::vector<float> hyper = Load(kA_hyper + t * flat, static_cast<size_t>(flat));
    const std::vector<float> block = Load(kA_block_out + t * hidden, static_cast<size_t>(hidden));
    const std::vector<float> inj = Load(kA_inj_w + t * hc, static_cast<size_t>(hc));
    const std::vector<float> got = GatedResidualWriteBack(hyper, block, inj, hc, hidden);
    CHECK(MaxAbsDiff(got, kA_written + t * flat, static_cast<size_t>(flat)) < kTol);

    // The un-normed stream is what gets written. Feeding the NORMED stream here
    // is the natural mistake and must be visible.
    const std::vector<float> normed = Load(kA_normed + t * flat, static_cast<size_t>(flat));
    CHECK(MaxAbsDiff(GatedResidualWriteBack(normed, block, inj, hc, hidden),
                     kA_written + t * flat, static_cast<size_t>(flat)) > 1e-2);

    // The in-place form is the fusion seam: it must agree bit-for-bit with the
    // allocating one, so a device arm can replace one with the other.
    std::vector<float> inplace = hyper;
    GatedResidualWriteBackInPlace(inplace.data(), block.data(), inj.data(), hc, hidden);
    for (size_t i = 0; i < inplace.size(); ++i) CHECK(inplace[i] == got[i]);
  }
}

// The spec asserts our DeepSeek-V4 `MhcPost` is a bit-exact bring-up bridge for
// this write-back once its comb matrix is the identity. Verified rather than
// trusted, because it is the reviewer's free mutation target. It holds on finite
// inputs; it is NOT an identity for a negative-zero residual (0.0f + (-0.0f)
// is +0.0f) or a non-finite one (0.0f * inf is NaN), and neither case is
// reachable on this path.
TEST_CASE("qwen4_exp write-back equals MhcPost with an identity comb, bit for bit") {
  const int64_t hc = 4, hidden = 6, flat = hc * hidden;
  std::vector<float> comb(static_cast<size_t>(hc * hc), 0.0f);
  for (int64_t j = 0; j < hc; ++j) comb[j * hc + j] = 1.0f;

  for (int64_t t = 0; t < 3; ++t) {
    const std::vector<float> hyper = Load(kA_hyper + t * flat, static_cast<size_t>(flat));
    const std::vector<float> block = Load(kA_block_out + t * hidden, static_cast<size_t>(hidden));
    const std::vector<float> inj = Load(kA_inj_w + t * hc, static_cast<size_t>(hc));
    const std::vector<float> ours = GatedResidualWriteBack(hyper, block, inj, hc, hidden);
    const std::vector<float> bridge =
        vllm::deepseek_v4::MhcPost(block, hyper, inj, comb, hc, hidden);
    REQUIRE(bridge.size() == ours.size());
    for (size_t i = 0; i < ours.size(); ++i) CHECK(bridge[i] == ours[i]);
  }
}

// ── 4. Discriminating power: each trap opens a gap the goldens can see ────────
TEST_CASE("qwen4_exp goldens separate every known gated-residual trap") {
  const int64_t hc = 4, hidden = 6, rank = 5, flat = hc * hidden;
  const std::vector<float> w_hf = Load(kA_norm_w_hf, static_cast<size_t>(flat));

  for (int64_t t = 0; t < 3; ++t) {
    const std::vector<float> hyper = Load(kA_hyper + t * flat, static_cast<size_t>(flat));
    const RefOut ok = ForwardRefD(hyper, w_hf, kA_down, kA_up, kA_inject, hc, hidden, rank, 1e-6,
                                  Variant{});
    // The independent double reference reproduces the oracle. Two references
    // agreeing is what makes a single-flag flip attributable to the flag.
    CHECK(MaxAbsD(ok.mixed, kA_mixed + t * hidden) < kTol);
    CHECK(MaxAbsD(ok.injection, kA_inj_w + t * hc) < kTol);

    struct Case {
      Variant v;
      bool hits_mixed;
      bool hits_injection;
    };
    Variant a;
    a.div_before_silu = false;  // silu(x)/hc instead of silu(x/hc)
    Variant b;
    b.div_on_up = true;  // a division upstream does NOT have
    Variant c;
    c.reduce_is_mean = false;  // sum instead of mean
    Variant d;
    d.multiply_normed = false;  // the raw stream instead of the normed one
    Variant e;
    e.inject_div = false;  // 2*sigmoid(logits) instead of 2*sigmoid(logits/hc)
    const Case cases[] = {{a, true, false}, {b, true, false}, {c, true, false},
                          {d, true, false}, {e, false, true}};

    for (const Case& k : cases) {
      const RefOut bad =
          ForwardRefD(hyper, w_hf, kA_down, kA_up, kA_inject, hc, hidden, rank, 1e-6, k.v);
      if (k.hits_mixed) CHECK(MaxAbsD(bad.mixed, kA_mixed + t * hidden) > kFlipFloor);
      if (k.hits_injection) CHECK(MaxAbsD(bad.injection, kA_inj_w + t * hc) > kFlipFloor);
    }
  }
}
