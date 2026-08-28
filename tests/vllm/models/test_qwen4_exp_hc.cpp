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
#include <cstdint>
#include <cstring>
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
//
// `kTol` IS A PROPERTY OF THESE SHAPES AND NOTHING ELSE. Say it here because the
// natural next reader is W5, wiring the module at hidden_size 2560, and reusing
// this number there is a red that is arithmetic noise. At the widths below —
// flat = 24 and 15 — the fp32 implementation is bit-identical or within one ulp
// of the oracle, measured against the pinned oracle itself: max|diff| over every
// golden array of case A, B and C is 2.384e-07, so kTol carries a 42x margin and
// is unconstrained. It does NOT survive a rescale, and the last two cases in this
// file are the measurement that says so rather than a warning that asks to be
// believed. See `kRealWidthNormTol` and `kRealWidthMixedTol`.
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

// EVERY max|diff| in this file goes through `vllm_test::MaxAbsDiff`, including
// the double-sided ones. This file used to carry two local helpers spelled
// `worst = std::max(worst, std::abs(...))` for the fp32-vs-double and
// double-vs-golden comparisons, which is verbatim the NaN-blind "form B" that
// `support/max_abs_diff.h` exists to prevent (issue #449): `std::max(a, NaN)`
// returns `a`, so an all-NaN result reduces to 0.0 and PASSES every `< tol`
// bound. It was not hypothetical — poisoning `GroupedRmsNorm`'s large-eps path
// to all-NaN left the `< kTol` assertion below GREEN, and the suite only went
// red because a SIBLING assertion happened to use the hardened helper. The
// shared scan is now templated on both operand types so there is no reason to
// write the reduction again here, and no place left to write it wrong.

// "Bit for bit" means the BIT PATTERN. `==` is VALUE equality, and the one
// divergence the MhcPost caveat below names — +0.0f against -0.0f — is exactly
// the case `==` cannot see. Returns the index of the first differing pattern,
// or `a.size()` when every pattern matches.
size_t FirstBitDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    uint32_t x = 0, y = 0;
    std::memcpy(&x, &a[i], sizeof(x));
    std::memcpy(&y, &b[i], sizeof(y));
    if (x != y) return i;
  }
  return a.size();
}

// ─── Real-width scaffolding ───────────────────────────────────────────────────
// `Qwen/Qwen3.8-Flash-Next`'s own values (spec `## Architecture`): hc_count = 4,
// hidden_size = 2560, hc_lowrank = 320, so the residual is 10240 wide and the
// grouped norm reduces over 2560 elements. No oracle golden exists at this width
// and none should: dumping the IO of one token is 26 MB of `.inc`. What the two
// cases at the end of this file gate is the arithmetic's behaviour at SCALE,
// against the same independent double reference the sweep uses.
constexpr int64_t kModelHc = 4;
constexpr int64_t kModelHidden = 2560;
constexpr int64_t kModelRank = 320;

// A deterministic generator written out in full rather than `std::mt19937` plus
// a distribution: `std::uniform_real_distribution` is NOT specified to produce
// the same sequence across standard libraries, and a gate whose INPUT depends on
// libstdc++ versus libc++ is not a reproducible measurement.
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  // Uniform in [-1, 1), from the top 24 bits, which are the good ones.
  float Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<uint32_t>(s >> 40)) / 8388608.0f - 1.0f;
  }
  // Approximately unit-variance normal, four uniforms deep (Irwin-Hall). It
  // matters that this is not uniform: the goldens are generated from
  // `torch.randn * k`, the errors below are compared against a measurement made
  // on THAT data, and a bounded uniform has visibly lighter tails, which shows
  // up as a smaller dot-product error over 10240 terms. Var(U[-1,1)) = 1/3, so
  // four of them scaled by sqrt(3)/2 is unit variance.
  float Normal() { return 0.8660254f * (Next() + Next() + Next() + Next()); }
};

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
  CHECK(MaxAbsDiff(NormRefD(hyper, w_hf, hc, hidden, 1e-6, global), kA_normed,
                   hyper.size()) > 1e-2);

  // Same for the missing `1 + w` fold, through the SWEEP's reference rather than
  // only through the direct call in the next case. `Variant` advertises a flag
  // per known defect and this one was declared, read, and never once set false,
  // so the struct claimed coverage the file did not have.
  Variant raw_gamma;
  raw_gamma.norm_weight_plus_one = false;
  CHECK(MaxAbsDiff(NormRefD(hyper, w_hf, hc, hidden, 1e-6, raw_gamma), kA_normed,
                   hyper.size()) > 1e-1);

  // eps lives INSIDE the rsqrt, added to the mean square. At the model's real
  // 1e-6 the two placements differ by ~1e-6, which is UNDER this suite's fp32
  // tolerance — so an eps-placement defect is invisible to every golden case
  // above, and a case at an eps large enough to separate them is the only thing
  // that gates it. `rsqrt(m + 4)` and `1/(sqrt(m) + 4)` are far apart; the
  // double reference puts eps inside, so this pins the placement rather than
  // merely observing that eps does something.
  const std::vector<float> big_eps = GroupedRmsNorm(hyper, w, hidden, 4.0f);
  CHECK(MaxAbsDiff(big_eps, got) > 1e-2);
  CHECK(MaxAbsDiff(big_eps, NormRefD(hyper, w_hf, hc, hidden, 4.0, Variant{})) < kTol);
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

TEST_CASE("qwen4_exp GatedResidual at a SMALL magnitude, where eps is observable") {
  // ADDED BY W5b-2 (#2031). It SHARPENS this file; it does not close a hole in
  // it. The `big_eps = 4.0f` probe in "qwen4_exp grouped RMSNorm mirrors
  // RMSNormGated(group_size) at the lane pin" above already gates the
  // placement: on a reconstructed pre-repair tree (this file and the goldens at
  // `origin/main`, no case D) the eps mutation reds it at
  // `CHECK( 0.802185 < 1e-05 )`, 2 of 14 cases and 2 assertions. The hole was
  // the DEVICE arm's, which had no such probe. Case D is driven by BOTH suites
  // and takes this file from those 2 red assertions to 10 (mutation M16).
  // `eps` goes INSIDE the rsqrt, added to the MEAN SQUARE
  // (`torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`, :170); the
  // plausible slip adds it to the norm instead. Cases A, B and C all draw the
  // stream at `hyper_scale = 1.7`, where the mean square is O(1) and eps = 1e-6
  // moves the answer by about 5e-7 — a third of `kTol`, so the wrong spelling
  // passes all three, and the `Variant` sweep below has no flag for it either.
  //
  // MEASURED on the device arm of the same arithmetic: the W5b-2 mutation
  // battery moved `+ eps` outside the rsqrt in `cpu_qwen4_exp.cpp` and it
  // SURVIVED every A/B/C comparison. Golden case D draws the stream at
  // `hyper_scale = 0.01`, where the mean square is ~1e-4 and eps is 1% of it, so
  // the two spellings separate by ~0.5% — three orders over `kTol`.
  const int64_t hc = 4, hidden = 6, rank = 5, flat = hc * hidden;
  GatedResidualWeights w;
  w.hc_norm_weight = HcNormWeightFromHf(Load(kD_norm_w_hf, static_cast<size_t>(flat)));
  w.mix_down = Load(kD_down, static_cast<size_t>(rank * flat));
  w.mix_up = Load(kD_up, static_cast<size_t>(flat * rank));
  w.block_inject = Load(kD_inject, static_cast<size_t>(hc * flat));

  for (int64_t t = 0; t < 2; ++t) {
    const std::vector<float> hyper = Load(kD_hyper + t * flat, static_cast<size_t>(flat));
    const GatedResidualResult r = GatedResidualForward(hyper, w, hc, hidden, 1e-6f);
    CHECK(MaxAbsDiff(r.hyper_input_normed, kD_normed + t * flat, static_cast<size_t>(flat)) <
          kTol);
    CHECK(MaxAbsDiff(r.mixed_input, kD_mixed + t * hidden, static_cast<size_t>(hidden)) < kTol);
    CHECK(MaxAbsDiff(r.injection_weights, kD_inj_w + t * hc, static_cast<size_t>(hc)) < kTol);
    const std::vector<float> block = Load(kD_block_out + t * hidden, static_cast<size_t>(hidden));
    const std::vector<float> got =
        GatedResidualWriteBack(hyper, block, r.injection_weights, hc, hidden);
    CHECK(MaxAbsDiff(got, kD_written + t * flat, static_cast<size_t>(flat)) < kTol);
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

// Every OTHER refusal this module makes. Split out from the two cases above
// because those two were, until this change, the ONLY refusals under any
// assertion: a mutation sweep deleted eight of the eleven guards one at a time
// and the suite stayed 280/280 green each time, including all three `RequireSize`
// calls in `GatedResidualWriteBack`.
//
// Three of those eight are not hygiene. `GatedResidualWriteBackInPlace` takes RAW
// POINTERS and the header declares it the fusion seam that a device kernel
// replaces, so the wrapper's three size checks are the only bounds check between
// a caller and an out-of-bounds read of `block_out[h]` or `injection_weights[j]`.
// What that class of guard prevents is not a tidy error: deleting the
// divisibility guard in the same sweep produced `Fatal glibc error:
// malloc.c:2599` and `Aborted (core dumped)`, with no doctest summary printed at
// all.
//
// The MESSAGE is pinned, never the type. Every guard in this translation unit
// throws `std::invalid_argument`, so `CHECK_THROWS_AS` cannot tell a deleted
// guard from a different one firing further down — which is the defect the
// hyper_input case above already documents finding.
TEST_CASE("qwen4_exp refuses every malformed shape BY NAME, not merely by type") {
  const std::vector<float> ones10(10, 1.0f);

  SUBCASE("grouped norm: a non-positive group size") {
    // Deleting this guard is not a wrong answer, it is `x.size() % 0` and the
    // process dies on SIGFPE. Re-run and confirmed: with the guard removed this
    // case takes the whole binary down mid-run.
    CHECK_THROWS_WITH_AS(GroupedRmsNorm(ones10, ones10, 0, 1e-6f),
                         "qwen4_exp: group_size must be positive, got 0.", std::invalid_argument);
    CHECK_THROWS_WITH_AS(GroupedRmsNorm(ones10, ones10, -3, 1e-6f),
                         "qwen4_exp: group_size must be positive, got -3.", std::invalid_argument);
  }

  SUBCASE("gated residual: hc_count <= 1 and hidden_size <= 0") {
    // `configuration_qwen4_exp.py:196-197` raises `Qwen4-Exp requires hc_count >
    // 1`; this is that refusal, and until now nothing held it.
    GatedResidualWeights w;
    w.hc_norm_weight.assign(24, 1.0f);
    w.mix_down.assign(5 * 24, 0.0f);
    w.mix_up.assign(24 * 5, 0.0f);
    CHECK_THROWS_WITH_AS(GatedResidualForward(std::vector<float>(6, 1.0f), w, 1, 6, 1e-6f),
                         "qwen4_exp: hc_count must be > 1 and hidden_size > 0, got 1 and 6.",
                         std::invalid_argument);
    CHECK_THROWS_WITH_AS(GatedResidualForward(std::vector<float>(24, 1.0f), w, 4, 0, 1e-6f),
                         "qwen4_exp: hc_count must be > 1 and hidden_size > 0, got 4 and 0.",
                         std::invalid_argument);
  }

  SUBCASE("gated residual: the low-rank weights") {
    // `rank` is INFERRED from `mix_down.size() / flat`, so a ragged `mix_down` is
    // not a bad input, it is a bad rank silently propagated into two more GEMMs.
    const std::vector<float> hyper(24, 1.0f);
    GatedResidualWeights w;
    w.hc_norm_weight.assign(24, 1.0f);
    w.mix_up.assign(24 * 5, 0.0f);

    w.mix_down.clear();
    CHECK_THROWS_WITH_AS(GatedResidualForward(hyper, w, 4, 6, 1e-6f),
                         "qwen4_exp: input_mix_weight_down is not a multiple of 24 (got 0).",
                         std::invalid_argument);
    w.mix_down.assign(23, 0.0f);
    CHECK_THROWS_WITH_AS(GatedResidualForward(hyper, w, 4, 6, 1e-6f),
                         "qwen4_exp: input_mix_weight_down is not a multiple of 24 (got 23).",
                         std::invalid_argument);

    w.mix_down.assign(5 * 24, 0.0f);  // rank 5, so mix_up must be 120
    w.mix_up.assign(119, 0.0f);
    CHECK_THROWS_WITH_AS(GatedResidualForward(hyper, w, 4, 6, 1e-6f),
                         "qwen4_exp: input_mix_weight_up has 119 elements, expected 120.",
                         std::invalid_argument);

    w.mix_up.assign(120, 0.0f);
    w.block_inject.assign(95, 0.0f);  // must be hc * flat = 96
    CHECK_THROWS_WITH_AS(GatedResidualForward(hyper, w, 4, 6, 1e-6f),
                         "qwen4_exp: block_inject_weight has 95 elements, expected 96.",
                         std::invalid_argument);
  }

  SUBCASE("write-back: the fusion seam's only bounds check") {
    const std::vector<float> hyper24(24, 1.0f), block6(6, 1.0f), inj4(4, 1.0f);
    CHECK_THROWS_WITH_AS(
        GatedResidualWriteBack(std::vector<float>(23, 1.0f), block6, inj4, 4, 6),
        "qwen4_exp: hyper_input has 23 elements, expected 24.", std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        GatedResidualWriteBack(hyper24, std::vector<float>(5, 1.0f), inj4, 4, 6),
        "qwen4_exp: block output has 5 elements, expected 6.", std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        GatedResidualWriteBack(hyper24, block6, std::vector<float>(3, 1.0f), 4, 6),
        "qwen4_exp: injection_weights has 3 elements, expected 4.", std::invalid_argument);
  }
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

    // The in-place form is THE FUSION SEAM (`qwen4_exp_hc.h`), so it is gated
    // against the ORACLE directly and not against its own wrapper. Checking it
    // against `got` alone is a tautology: `GatedResidualWriteBack` copies and
    // then calls this very function, so the comparison holds under any mutation
    // of it. Proven — mutating the in-place body (`+=` to `=`, and
    // `injection_weights[j]` to `[0]`) reddened three other assertions and never
    // this one. The golden compare below is the assertion that actually holds
    // the primitive; the bit check that follows holds the WRAPPER's claim to be
    // free of its own arithmetic.
    std::vector<float> inplace = hyper;
    GatedResidualWriteBackInPlace(inplace.data(), block.data(), inj.data(), hc, hidden);
    CHECK(MaxAbsDiff(inplace, kA_written + t * flat, static_cast<size_t>(flat)) < kTol);
    CHECK(FirstBitDiff(inplace, got) == inplace.size());
  }
}

// The bit helper's own polarity, pinned. `==` reports these two equal and the
// pattern comparison does not, which is the entire reason the two "bit for bit"
// claims in this file stopped using `==`.
TEST_CASE("qwen4_exp bit-for-bit means the pattern: +0.0f and -0.0f are == but not equal") {
  const std::vector<float> pos{0.0f, 1.5f};
  const std::vector<float> neg{-0.0f, 1.5f};
  CHECK(pos[0] == neg[0]);
  CHECK(FirstBitDiff(pos, neg) == 0);
  CHECK(FirstBitDiff(pos, pos) == pos.size());
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
    // The PATTERN, not `==`. The negative-zero caveat in the comment above is
    // precisely the divergence value equality cannot see, so asserting the claim
    // with `==` asserted something weaker than the claim.
    CHECK(FirstBitDiff(bridge, ours) == ours.size());
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
    CHECK(MaxAbsDiff(ok.mixed, kA_mixed + t * hidden, static_cast<size_t>(hidden)) < kTol);
    CHECK(MaxAbsDiff(ok.injection, kA_inj_w + t * hc, static_cast<size_t>(hc)) < kTol);

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
      if (k.hits_mixed) {
        CHECK(MaxAbsDiff(bad.mixed, kA_mixed + t * hidden, static_cast<size_t>(hidden)) >
              kFlipFloor);
      }
      if (k.hits_injection) {
        CHECK(MaxAbsDiff(bad.injection, kA_inj_w + t * hc, static_cast<size_t>(hc)) > kFlipFloor);
      }
    }
  }
}

// ── 5. Real width: the two things the toy shapes cannot measure ───────────────
//
// Everything above runs at hidden_size 6 or 5 and hc_count 4 or 3. Two of this
// module's documented properties are invisible at those widths, and one of them
// is a tolerance the next wave will otherwise reuse.

// THE DOUBLE ACCUMULATOR. `qwen4_exp_hc.h` "PRECISION" states the per-group sum
// of squares is accumulated in `double`. At group sizes 5 and 6 that claim has
// no discriminating power whatsoever: replacing `double ss` with `float ss`
// moves nothing any tolerance in this file can see, so it was a documented
// convention with zero coverage. At the model's real group size it is worth
// three orders of magnitude.
//
// The input is magnitude-separated on purpose — one element at 4096, the rest at
// 1.0 — and that is not cherry-picking, it is the only shape that CAN separate
// the two accumulators. A sum of 2560 POSITIVE squares of similar size loses
// about sqrt(N)*u either way and the two agree to within 3x. The loss appears
// when a partial sum grows past the point where the next term falls under its
// own ulp: 4096^2 is exactly 2^24, the float integer ceiling, so every one of the
// 2559 following `+1.0` terms rounds away and `float ss` misses the tail
// ENTIRELY, by 2559 out of 16779775.
//
// Measured on exactly the data below, and byte-identical at -O0 and -O2, so it
// does not rest on vectorisation. It cannot rest on FMA contraction either: this
// tree compiles with `-ffp-contract=off` everywhere.
//
//   max|reference|                    5.20e+01
//   ours, double accumulator          3.168430e-06   (6.1e-08 relative: one ulp)
//   the same code with `float ss`     2.352230e-03   (742x worse)
//
// The bound sits 31.6x above the first and 23.5x below the second, which is a
// band rather than a fitted number. Both ends are re-run mutations, not
// estimates: `double ss` -> `float ss` in `GroupedRmsNorm` reddens this case and
// nothing else in the file.
constexpr double kRealWidthNormTol = 1e-4;

TEST_CASE("qwen4_exp GroupedRmsNorm needs its double accumulator at group_size 2560") {
  const int64_t flat = kModelHc * kModelHidden;
  std::vector<float> x(static_cast<size_t>(flat)), w(static_cast<size_t>(flat));
  const float dominant[4] = {4096.0f, 2048.0f, 8192.0f, 1024.0f};
  Lcg rng(12345);
  for (int64_t j = 0; j < kModelHc; ++j) {
    for (int64_t d = 0; d < kModelHidden; ++d) {
      x[j * kModelHidden + d] = (d == 0) ? dominant[j] : 1.0f;
      w[j * kModelHidden + d] = 1.0f + 0.5f * rng.Next();
    }
  }

  const std::vector<float> got = GroupedRmsNorm(x, w, kModelHidden, 1e-6f);

  // The reference, in full double, written out here rather than reusing
  // `NormRefD`: that one takes the HF gamma and applies the `1 + w` fold, and
  // this case is about the ACCUMULATOR, so the weight goes in already folded.
  std::vector<double> want(static_cast<size_t>(flat));
  for (int64_t j = 0; j < kModelHc; ++j) {
    double ss = 0.0;
    for (int64_t d = 0; d < kModelHidden; ++d) {
      const double v = x[j * kModelHidden + d];
      ss += v * v;
    }
    const double r = 1.0 / std::sqrt(ss / static_cast<double>(kModelHidden) + 1e-6);
    for (int64_t d = 0; d < kModelHidden; ++d) {
      want[j * kModelHidden + d] = static_cast<double>(x[j * kModelHidden + d]) * r *
                                   static_cast<double>(w[j * kModelHidden + d]);
    }
  }
  CHECK(MaxAbsDiff(got, want) < kRealWidthNormTol);

  // And the separation is real rather than asserted: the same reduction with a
  // float accumulator, over the same data, is on the far side of the bound. If
  // this stops holding, the case above has stopped measuring anything.
  float ss_f = 0.0f;
  double ss_d = 0.0;
  for (int64_t d = 0; d < kModelHidden; ++d) {
    const double v = x[d];
    ss_f += static_cast<float>(v * v);
    ss_d += v * v;
  }
  const double r_f = 1.0 / std::sqrt(static_cast<double>(ss_f) / kModelHidden + 1e-6);
  const double r_d = 1.0 / std::sqrt(ss_d / kModelHidden + 1e-6);
  CHECK(std::abs(r_f - r_d) * dominant[0] * w[0] > 10.0 * kRealWidthNormTol);
}

// WHAT THE fp32 INTERIOR COSTS, AND WHY `kTol` DOES NOT TRAVEL. At flat = 24 the
// implementation is bit-identical to the oracle, so kTol measures nothing there.
// At flat = 10240 it is not, and the reason is not ours alone. Measured against
// the PINNED ORACLE itself (transformers v5.16.0, the same lift-and-exec path
// `scripts/gen-qwen4-exp-hc-goldens.py` uses, at hidden=2560 hc=4 lowrank=320
// eps=1e-6, two tokens), max|diff| on `mixed_input`. These are ONE draw of
// random inputs, and the ratios below move from draw to draw; the ordering and
// the conclusion do not.
//
//                             t=0          t=1
//   ours (fp32)  vs oracle    2.325e-05    2.137e-05
//   double ref   vs oracle    1.360e-05    5.431e-06
//   ours (fp32)  vs double    3.684e-05    1.606e-05
//
// Two conclusions, and the second is the one that matters. Our fp32 interior is
// 2.1x to 2.3x over kTol at model width, driven by `LinearNoBias`'s sequential
// fp32 accumulation over 10240 terms. And the ORACLE is itself OF THE SAME
// ORDER AS kTol against an exact evaluation of its own algorithm — 1.36x on the
// draw tabulated above, 0.91x and 0.82x on an independent draw taken during
// fresh review — because torch runs this in fp32 too, so widening OUR
// accumulator cannot rescue a 1e-5 absolute bound here. No
// fp32 implementation of this function meets kTol at hidden_size 2560, and the
// tolerance is the thing that is wrong at that width, not the arithmetic.
//
// The bound below is therefore RELATIVE to the reference's own magnitude, and
// the case compares against the double reference because committing an oracle
// golden at this width is 26 MB of `.inc`. It is derived rather than fitted: a
// sequential fp32 dot of length K accumulates a relative error that grows as a
// random walk, about sqrt(K)*u, with u = 2^-24 = 5.96e-08 the fp32 unit
// roundoff. At K = 10240 that is 6.03e-06, and 4e-05 is 6.6x it. The three
// measurements that exist all sit inside that: 6.79e-06 relative on this case's
// own data, and 7.32e-06 / 1.68e-05 on the two oracle tokens tabulated above.
//
// What this gates is that the fp32 interior stays within a small multiple of the
// random-walk bound for its accumulation length. What it does NOT gate is
// agreement with the ORACLE at this width — that needs a real checkpoint, and it
// is recorded under the spec's `## Owed`.
constexpr double kRealWidthMixedRel = 4e-5;

TEST_CASE("qwen4_exp gated residual at the model's real width, hidden_size 2560") {
  const int64_t flat = kModelHc * kModelHidden;
  Lcg rng(20260826);
  std::vector<float> w_hf(static_cast<size_t>(flat)), hyper(static_cast<size_t>(flat));
  std::vector<float> down(static_cast<size_t>(kModelRank * flat));
  std::vector<float> up(static_cast<size_t>(flat * kModelRank));
  std::vector<float> inject(static_cast<size_t>(kModelHc * flat));
  // The same standard deviations `scripts/gen-qwen4-exp-hc-goldens.py` uses, so
  // the numbers this case measures are comparable with the oracle table above.
  for (float& v : w_hf) v = 0.5f * rng.Normal();
  for (float& v : hyper) v = 1.7f * rng.Normal();
  for (float& v : down) v = 0.3f * rng.Normal();
  for (float& v : up) v = 0.3f * rng.Normal();
  for (float& v : inject) v = 0.3f * rng.Normal();

  GatedResidualWeights w;
  w.hc_norm_weight = HcNormWeightFromHf(w_hf);
  w.mix_down = down;
  w.mix_up = up;
  w.block_inject = inject;

  const GatedResidualResult got = GatedResidualForward(hyper, w, kModelHc, kModelHidden, 1e-6f);
  const RefOut ref = ForwardRefD(hyper, w_hf, down.data(), up.data(), inject.data(), kModelHc,
                                 kModelHidden, kModelRank, 1e-6, Variant{});

  double scale = 0.0;
  for (double v : ref.mixed) scale = std::max(scale, std::abs(v));
  REQUIRE(scale > 0.1);  // the data is not degenerate, so the ratio means something

  CHECK(MaxAbsDiff(got.hyper_input_normed, ref.normed) < kRealWidthMixedRel * scale);
  CHECK(MaxAbsDiff(got.mixed_input, ref.mixed) < kRealWidthMixedRel * scale);
  CHECK(MaxAbsDiff(got.injection_weights, ref.injection) < kRealWidthMixedRel * 2.0);

  // The write-back at real width too: it is the seam, and 10240 is the width the
  // fused kernel will see.
  std::vector<float> block(static_cast<size_t>(kModelHidden));
  for (float& v : block) v = 0.9f * rng.Normal();
  const std::vector<float> written =
      GatedResidualWriteBack(hyper, block, got.injection_weights, kModelHc, kModelHidden);
  REQUIRE(written.size() == static_cast<size_t>(flat));
  for (int64_t j = 0; j < kModelHc; ++j) {
    for (int64_t h = 0; h < kModelHidden; h += 512) {
      const size_t i = static_cast<size_t>(j * kModelHidden + h);
      CHECK(written[i] == doctest::Approx(hyper[i] + block[h] * got.injection_weights[j]));
    }
  }
}
