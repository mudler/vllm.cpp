// LTX-2.5 PIPELINE — see include/vllm/model_executor/models/ltx2_pipeline.h for
// the upstream mapping, the recipe provenance, and why the refusals exist.
//
// EVERY numeric expression below mirrors upstream's ARITHMETIC WIDTH, not just
// its algebra, because the two schedulers do not agree on it and a port that
// picked one width for both would be wrong on one of them:
//
//   LTX2Scheduler          torch float32 tensors throughout (schedulers.py:33-57),
//                          with the shift term alone computed as a Python double
//                          before torch sees it (:37-44).
//   LinearQuadraticScheduler  PYTHON FLOATS — i.e. double — for the whole
//                          schedule, cast to float32 once by `torch.FloatTensor`
//                          at the very end (:75-88).
//
// The diffusion steps are float32 (each casts `.to(torch.float32)` on entry), and
// their scalar coefficients are 0-dim float32 tensors, so `eta`, `s_noise` and
// friends round to f32 at every binary op rather than staying double.
//
// ONE KNOWN EXCEPTION TO THE "EVERY" ABOVE, found 2026-08-12 and NOT yet
// repaired. `Ltx2SigmaSchedule`'s shift (:105-109) is written as f32/f32, but
// upstream's is a PYTHON SCALAR divided by a tensor (schedulers.py:43-45), and
// torch evaluates scalar/tensor as `scalar * reciprocal(tensor)` — so at
// sigma == 1 upstream yields 0.99999994, one ulp below 1, where this file yields
// exactly 1.0. It is invisible for steps >= 2 (the stretch renormalizes it away,
// within the 5e-06 gate), but at steps == 1 that residue is the ONLY non-zero
// sigma and therefore the whole stretch anchor: upstream returns
// {0.10000002, 0} and this returns {-nan, 0}. The `OneStep` golden already
// carries upstream's correct value and did not catch it because the suite's
// `MaxAbsDiff` drops NaN. Left unrepaired deliberately: mirroring torch's
// reciprocal-multiply moves every sigma on every arm, so it owes its own
// red-first change and a fresh review rather than a drive-by edit. Gated as far
// as it can be without the fix by the "terminates at exactly 0" case in
// tests/vllm/models/test_ltx2_pipeline.cpp, which sweeps steps 1..200.
#include "vllm/model_executor/models/ltx2_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& message) { throw std::runtime_error(message); }

void Require(bool condition, const std::string& message) {
  if (!condition) Refuse(message);
}

// torch.linspace on CPU (aten RangeFactories.cpp): `step` is computed in the
// tensor's own dtype and the SECOND HALF is walked backwards from `end`, which is
// what makes the terminal value exactly `end` instead of an accumulated
// near-zero. The schedule's last sigma being exactly 0 is what every terminal
// early-out branches on, so this is mirrored rather than approximated.
std::vector<float> LinspaceF32(float start, float end, int64_t count) {
  std::vector<float> out(static_cast<size_t>(count));
  if (count == 1) {
    out[0] = start;
    return out;
  }
  const float step = (end - start) / static_cast<float>(count - 1);
  const int64_t halfway = count / 2;
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = i < halfway
                                      ? start + step * static_cast<float>(i)
                                      : end - step * static_cast<float>(count - i - 1);
  }
  return out;
}

// math.isclose's defaults: rel_tol = 1e-9, abs_tol = 0.0. MultiModalGuider uses
// it where CFGGuider uses `!=` (guiders.py:26 vs :277), and the two are not
// interchangeable — the difference decides whether an extra full DiT forward runs.
bool IsClose(double a, double b) {
  const double tol = 1e-9 * std::max(std::fabs(a), std::fabs(b));
  return std::fabs(a - b) <= tol;
}

// Lexicographic tuple comparison, matching Python's: a shorter tuple that is a
// prefix of a longer one compares LESS, which is what makes `()` (an unset or
// unparseable version) fall below every real generation.
int CompareVersion(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const size_t common = std::min(a.size(), b.size());
  for (size_t i = 0; i < common; ++i) {
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  }
  if (a.size() == b.size()) return 0;
  return a.size() < b.size() ? -1 : 1;
}

std::string NormalizeVersionSeparator(const std::string& version) {
  // utils/constants.py:159-161. Pre-release tags come both dot- and
  // hyphen-separated; without this "2.4-rc2" parses to (2,) and lands a release
  // candidate on the 2.0 recipe.
  std::string normalized = version;
  std::replace(normalized.begin(), normalized.end(), '-', '.');
  return normalized;
}

}  // namespace

// ---------------------------------------------------------------------------
// Sigma schedules
// ---------------------------------------------------------------------------

std::vector<float> Ltx2SigmaSchedule(int64_t steps, int64_t tokens,
                                     const Ltx2SchedulerParams& params) {
  Require(steps >= 1, "ltx2 scheduler: steps must be >= 1, got " + std::to_string(steps));
  const int64_t resolved_tokens = tokens > 0 ? tokens : params.default_number_of_tokens;

  std::vector<float> sigmas = LinspaceF32(1.0f, 0.0f, steps + 1);

  // schedulers.py:35-39, all Python doubles before torch sees them.
  const double x1 = static_cast<double>(kLtx2BaseShiftAnchor);
  const double x2 = static_cast<double>(kLtx2MaxShiftAnchor);
  const double mm = (params.max_shift - params.base_shift) / (x2 - x1);
  const double b = params.base_shift - mm * x1;
  const double sigma_shift = static_cast<double>(resolved_tokens) * mm + b;
  const float shift_exp = static_cast<float>(std::exp(sigma_shift));

  // schedulers.py:42-46 — `torch.where(sigmas != 0, ..., 0)`. The zero branch is
  // taken structurally rather than by evaluating 1/0 and relying on inf.
  for (float& sigma : sigmas) {
    if (sigma == 0.0f) continue;
    const float shifted = std::pow(1.0f / sigma - 1.0f, static_cast<float>(kLtx2SigmaShiftPower));
    sigma = shift_exp / (shift_exp + shifted);
  }

  if (params.stretch) {
    // schedulers.py:49-55. `one_minus_z[-1]` is the LAST non-zero entry, which
    // for a linspace(1, 0) is the one just before the terminal 0.
    int64_t last_non_zero = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(sigmas.size()); ++i) {
      if (sigmas[static_cast<size_t>(i)] != 0.0f) last_non_zero = i;
    }
    if (last_non_zero >= 0) {
      const float terminal_gap = static_cast<float>(1.0 - params.terminal);
      const float scale_factor = (1.0f - sigmas[static_cast<size_t>(last_non_zero)]) / terminal_gap;
      for (float& sigma : sigmas) {
        if (sigma == 0.0f) continue;
        sigma = 1.0f - (1.0f - sigma) / scale_factor;
      }
    }
  }
  return sigmas;
}

std::vector<float> Ltx2LinearQuadraticSchedule(int64_t steps, double threshold_noise,
                                               int64_t linear_steps) {
  Require(steps >= 1,
          "ltx2 linear-quadratic scheduler: steps must be >= 1, got " + std::to_string(steps));
  // schedulers.py:70-71 — the one-step schedule is a literal, not a formula.
  if (steps == 1) return {1.0f, 0.0f};

  const int64_t linear = linear_steps < 0 ? steps / 2 : linear_steps;
  Require(linear > 0,
          "ltx2 linear-quadratic scheduler: linear_steps must be > 0 (upstream divides by it "
          "at schedulers.py:75), got " +
              std::to_string(linear));

  // Python floats, i.e. DOUBLE, for the whole schedule. Only `torch.FloatTensor`
  // at :88 narrows it.
  std::vector<double> schedule;
  schedule.reserve(static_cast<size_t>(steps + 1));
  for (int64_t i = 0; i < linear; ++i) {
    schedule.push_back(static_cast<double>(i) * threshold_noise / static_cast<double>(linear));
  }
  const double step_diff = static_cast<double>(linear) - threshold_noise * static_cast<double>(steps);
  const int64_t quadratic_steps = steps - linear;
  if (quadratic_steps > 0) {
    const double qs = static_cast<double>(quadratic_steps);
    const double quadratic_coef = step_diff / (static_cast<double>(linear) * qs * qs);
    const double linear_coef =
        threshold_noise / static_cast<double>(linear) - 2.0 * step_diff / (qs * qs);
    const double constant = quadratic_coef * static_cast<double>(linear) * static_cast<double>(linear);
    for (int64_t i = linear; i < steps; ++i) {
      const double x = static_cast<double>(i);
      schedule.push_back(quadratic_coef * x * x + linear_coef * x + constant);
    }
  }
  schedule.push_back(1.0);

  std::vector<float> out(schedule.size());
  for (size_t i = 0; i < schedule.size(); ++i) {
    out[i] = static_cast<float>(1.0 - schedule[i]);
  }
  return out;
}

std::vector<float> Ltx2Schedule(Ltx2SchedulerKind kind, int64_t steps, int64_t tokens,
                                const Ltx2SchedulerParams& params) {
  switch (kind) {
    case Ltx2SchedulerKind::kLtx2:
      return Ltx2SigmaSchedule(steps, tokens, params);
    case Ltx2SchedulerKind::kLinearQuadratic:
      return Ltx2LinearQuadraticSchedule(steps);
    case Ltx2SchedulerKind::kBeta:
      Ltx2RefuseUnportedPipelineFeature(Ltx2UnportedPipelineFeature::kBetaScheduler);
  }
  Refuse("ltx2 scheduler: unknown Ltx2SchedulerKind");
}

// ---------------------------------------------------------------------------
// The noiser
// ---------------------------------------------------------------------------

std::vector<float> Ltx2GaussianNoise(const float* latent, const float* clean_latent,
                                     const float* denoise_mask, const float* noise,
                                     int64_t count, float noise_scale) {
  Require(latent != nullptr && clean_latent != nullptr && denoise_mask != nullptr &&
              noise != nullptr,
          "ltx2 noiser: latent, clean_latent, denoise_mask and noise are all required");
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const size_t k = static_cast<size_t>(i);
    // noisers.py:32 — lerp toward the NOISE by noise_scale...
    const float noised = latent[k] + noise_scale * (noise[k] - latent[k]);
    // ...then :33 — lerp from the CLEAN latent toward that, by the denoise mask.
    // Mask 0 keeps the clean value (a conditioned token); mask 1 takes the noised
    // one. Swapping these two operands still renders.
    out[k] = clean_latent[k] + denoise_mask[k] * (noised - clean_latent[k]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Diffusion steps
// ---------------------------------------------------------------------------

namespace {

void RequireStepIndex(int64_t sigma_count, int64_t step_index) {
  Require(step_index >= 0 && step_index + 1 < sigma_count,
          "ltx2 diffusion step: step_index " + std::to_string(step_index) +
              " is outside a " + std::to_string(sigma_count) + "-sigma schedule");
}

}  // namespace

std::vector<float> Ltx2EulerStep(const float* sample, const float* denoised,
                                 const float* sigmas, int64_t sigma_count, int64_t step_index,
                                 int64_t count) {
  RequireStepIndex(sigma_count, step_index);
  const float sigma = sigmas[step_index];
  const float sigma_next = sigmas[step_index + 1];
  // to_velocity (utils.py:34-35) REFUSES a zero sigma rather than dividing.
  Require(sigma != 0.0f, "ltx2 Euler step: sigma can't be 0.0 (utils.py:34-35)");
  const float dt = sigma_next - sigma;

  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const size_t k = static_cast<size_t>(i);
    const float velocity = (sample[k] - denoised[k]) / sigma;
    out[k] = sample[k] + velocity * dt;
  }
  return out;
}

std::vector<float> Ltx2EulerAncestralStep(const float* sample, const float* denoised,
                                          const float* sigmas, int64_t sigma_count,
                                          int64_t step_index, int64_t count, double eta,
                                          double s_noise, const float* noise) {
  RequireStepIndex(sigma_count, step_index);
  const float sigma = sigmas[step_index];
  const float sigma_next = sigmas[step_index + 1];
  // diffusion_steps.py:85-86 — the terminal step returns the denoised prediction
  // outright. This is checked BEFORE the noise requirement, exactly as upstream
  // orders it.
  if (sigma_next == 0.0f) {
    return std::vector<float>(denoised, denoised + count);
  }
  Require(!(eta > 0.0 && noise == nullptr),
          "ltx2 EulerAncestral step: requires a noise tensor when eta > 0 "
          "(diffusion_steps.py:87-88)");

  const float eta_f = static_cast<float>(eta);
  const float downstep_ratio = 1.0f + (sigma_next / sigma - 1.0f) * eta_f;
  const float sigma_down = sigma_next * downstep_ratio;
  const float sigma_down_ratio = sigma_down / sigma;

  float renoise_coeff = 0.0f;
  float alpha_scale = 1.0f;
  if (eta > 0.0) {
    const float alpha_next = 1.0f - sigma_next;
    const float alpha_down = 1.0f - sigma_down;
    const float inner = sigma_next * sigma_next -
                        sigma_down * sigma_down * alpha_next * alpha_next /
                            (alpha_down * alpha_down);
    renoise_coeff = std::sqrt(std::max(inner, 0.0f));
    alpha_scale = alpha_next / alpha_down;
  }
  const float s_noise_f = static_cast<float>(s_noise);

  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const size_t k = static_cast<size_t>(i);
    float value = sigma_down_ratio * sample[k] + (1.0f - sigma_down_ratio) * denoised[k];
    if (eta > 0.0) {
      value = alpha_scale * value + noise[k] * s_noise_f * renoise_coeff;
    }
    out[k] = value;
  }
  return out;
}

namespace {

// `Res2sDiffusionStep.get_sde_coeff` (diffusion_steps.py:136-155), the
// `sigma_up is not None` arm — the only one `step` reaches (:179).
//
// TEMPLATED ON THE SIGMA TYPE, because upstream's has no dtype of its own and
// the res_2s loop reaches it at TWO precisions: float64 from the substep's
// `[sigma, sub_sigma]` pair (samplers.py:342) and float32 from the loop's own
// schedule at step level (samplers.py:415). Instantiating one formula twice is
// what keeps that from becoming a second copy.
template <typename Sigma>
Ltx2SdeCoeff Res2sSdeCoeffImpl(double sigma_next, double sigma_up) {
  const Sigma next = static_cast<Sigma>(sigma_next);
  Sigma up = static_cast<Sigma>(sigma_up);
  up = std::min(up, next * static_cast<Sigma>(kLtx2Res2sSigmaUpClamp));

  const Sigma sigma_signal = static_cast<Sigma>(1) - next;  // `sigmax` defaults to ones_like
  const Sigma residual = std::sqrt(std::max(next * next - up * up, static_cast<Sigma>(0)));
  Sigma alpha_ratio = sigma_signal + residual;
  Sigma down = residual / alpha_ratio;

  // :149-153 — the NaN scrubbing, which is what keeps a degenerate schedule from
  // poisoning the whole latent.
  if (std::isnan(up)) up = static_cast<Sigma>(0);
  if (std::isnan(down)) down = next;
  if (std::isnan(alpha_ratio)) alpha_ratio = static_cast<Sigma>(1);

  Ltx2SdeCoeff coeff;
  coeff.alpha_ratio = alpha_ratio;
  coeff.sigma_down = down;
  coeff.sigma_up = up;
  return coeff;
}

// `Res2sDiffusionStep.step` (diffusion_steps.py:157-190). `Sigma` is the width
// the SCHEDULE and therefore the coefficients are computed at; `Value` is the
// width of the sample, the noise and the result. Upstream reaches three
// combinations across this port's call sites and they are the three
// instantiations below.
template <typename Sigma, typename Value>
std::vector<Value> Res2sStepImpl(const Value* sample, const Value* denoised,
                                 const Sigma* sigmas, int64_t sigma_count, int64_t step_index,
                                 int64_t count, const Value* noise, double eta) {
  RequireStepIndex(sigma_count, step_index);
  const Sigma sigma = sigmas[step_index];
  const Sigma sigma_next = sigmas[step_index + 1];
  const Ltx2SdeCoeff coeff = Res2sSdeCoeffImpl<Sigma>(
      static_cast<double>(sigma_next),
      static_cast<double>(sigma_next * static_cast<Sigma>(eta)));

  // :181-182 — returned UNCHANGED, not cast, when either is zero.
  if (coeff.sigma_up == 0.0 || sigma_next == static_cast<Sigma>(0)) {
    return std::vector<Value>(denoised, denoised + count);
  }
  Require(noise != nullptr, "ltx2 Res2s step: requires a noise tensor");

  const Sigma alpha_ratio = static_cast<Sigma>(coeff.alpha_ratio);
  const Sigma sigma_down = static_cast<Sigma>(coeff.sigma_down);
  const Sigma sigma_up = static_cast<Sigma>(coeff.sigma_up);
  // The SUBTRACTION happens at the schedule's own width, which is what upstream
  // does: `sigma - sigma_next` is a tensor op between two schedule entries
  // before the f64 numerator ever divides by it (diffusion_steps.py:185).
  const Sigma denom = sigma - sigma_next;

  std::vector<Value> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const size_t k = static_cast<size_t>(i);
    const Value eps_next = (sample[k] - denoised[k]) / static_cast<Value>(denom);
    const Value denoised_next = sample[k] - static_cast<Value>(sigma) * eps_next;
    out[k] = static_cast<Value>(alpha_ratio) *
                 (denoised_next + static_cast<Value>(sigma_down) * eps_next) +
             static_cast<Value>(sigma_up) * noise[k];
  }
  return out;
}

}  // namespace

Ltx2SdeCoeff Ltx2Res2sSdeCoeff(double sigma_next, double sigma_up) {
  return Res2sSdeCoeffImpl<float>(sigma_next, sigma_up);
}

Ltx2SdeCoeff Ltx2Res2sSdeCoeffHp(double sigma_next, double sigma_up) {
  return Res2sSdeCoeffImpl<double>(sigma_next, sigma_up);
}

std::vector<float> Ltx2Res2sStep(const float* sample, const float* denoised,
                                 const float* sigmas, int64_t sigma_count, int64_t step_index,
                                 int64_t count, const float* noise, double eta) {
  return Res2sStepImpl<float, float>(sample, denoised, sigmas, sigma_count, step_index, count,
                                     noise, eta);
}

std::vector<double> Ltx2Res2sStepHp(const double* sample, const double* denoised,
                                    const double* sigmas, int64_t sigma_count,
                                    int64_t step_index, int64_t count, const double* noise,
                                    double eta, Ltx2Res2sScheduleWidth width) {
  if (width == Ltx2Res2sScheduleWidth::kF64Schedule) {
    return Res2sStepImpl<double, double>(sample, denoised, sigmas, sigma_count, step_index,
                                         count, noise, eta);
  }
  // The step-level arm. The schedule really is float32 upstream, so it is
  // narrowed HERE rather than at the call site: narrowing at the call site would
  // put the conversion one frame away from the arithmetic it changes, and the
  // next reader would have to reconstruct which of the two widths ran.
  std::vector<float> narrowed(static_cast<size_t>(sigma_count));
  for (int64_t i = 0; i < sigma_count; ++i) {
    narrowed[static_cast<size_t>(i)] = static_cast<float>(sigmas[i]);
  }
  return Res2sStepImpl<float, double>(sample, denoised, narrowed.data(), sigma_count,
                                      step_index, count, noise, eta);
}

Ltx2AncestralSigmas Ltx2AncestralStep(double sigma_from, double sigma_to, double eta) {
  Ltx2AncestralSigmas result;
  // :17-18 — `if not eta`, i.e. exactly 0.0, short-circuits before any division.
  if (eta == 0.0) {
    result.sigma_down = sigma_to;
    result.sigma_up = 0.0;
    return result;
  }
  const float from = static_cast<float>(sigma_from);
  const float to = static_cast<float>(sigma_to);
  const float variance = to * to * std::max(from * from - to * to, 0.0f) / (from * from);
  float up = static_cast<float>(eta) * std::sqrt(variance);
  up = std::min(up, to);
  const float down = std::sqrt(std::max(to * to - up * up, 0.0f));
  result.sigma_down = down;
  result.sigma_up = up;
  return result;
}

std::vector<float> Ltx2EulerCfgPpStep(const float* sample, const float* denoised,
                                      const float* uncond_denoised, const float* sigmas,
                                      int64_t sigma_count, int64_t step_index, int64_t count,
                                      double eta, double s_noise, const float* noise) {
  RequireStepIndex(sigma_count, step_index);
  Require(uncond_denoised != nullptr,
          "ltx2 CFG++ step: requires an unconditioned prediction (diffusion_steps.py:214)");
  const float sigma_s = sigmas[step_index];
  const float sigma_t = sigmas[step_index + 1];
  // :233-235 — the clamp that keeps `alpha = 1 - sigma` off zero when sigma is
  // EXACTLY 1.0, which the first step of an unstretched schedule is.
  const float eps = static_cast<float>(kLtx2CfgPpAlphaEps);
  const float alpha_s = std::max(1.0f - sigma_s, eps);
  const float alpha_t = std::max(1.0f - sigma_t, eps);

  const Ltx2AncestralSigmas ancestral =
      Ltx2AncestralStep(static_cast<double>(sigma_s / alpha_s),
                        static_cast<double>(sigma_t / alpha_t), eta);
  const float sigma_down = alpha_t * static_cast<float>(ancestral.sigma_down);
  const float sigma_up = static_cast<float>(ancestral.sigma_up);
  const bool renoise = noise != nullptr && eta > 0.0 && s_noise > 0.0;
  const float s_noise_f = static_cast<float>(s_noise);

  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const size_t k = static_cast<size_t>(i);
    // :243 — the derivative uses the UNCONDITIONED prediction. That is the whole
    // CFG++ correction; using the conditioned one is a plain Euler step.
    const float d = (sample[k] - alpha_s * uncond_denoised[k]) / sigma_s;
    float value = alpha_t * denoised[k] + sigma_down * d;
    if (renoise) value += alpha_t * noise[k] * s_noise_f * sigma_up;
    out[k] = value;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Guidance
// ---------------------------------------------------------------------------

std::vector<float> Ltx2ProjectionCoef(const float* to_project, const float* project_onto,
                                      int64_t batch, int64_t count) {
  std::vector<float> out(static_cast<size_t>(batch));
  for (int64_t b = 0; b < batch; ++b) {
    double dot = 0.0;
    double squared_norm = 0.0;
    for (int64_t i = 0; i < count; ++i) {
      const size_t k = static_cast<size_t>(b * count + i);
      dot += static_cast<double>(to_project[k]) * static_cast<double>(project_onto[k]);
      squared_norm += static_cast<double>(project_onto[k]) * static_cast<double>(project_onto[k]);
    }
    const float denominator = static_cast<float>(squared_norm) +
                              static_cast<float>(kLtx2ProjectionCoefEps);
    out[static_cast<size_t>(b)] = static_cast<float>(dot) / denominator;
  }
  return out;
}

std::vector<float> Ltx2CfgDelta(const float* cond, const float* uncond, int64_t count,
                                double scale) {
  // guiders.py:24 — `(self.scale - 1)` is evaluated in Python (double) and only
  // then meets the tensor.
  const float gain = static_cast<float>(scale - 1.0);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const size_t k = static_cast<size_t>(i);
    out[k] = gain * (cond[k] - uncond[k]);
  }
  return out;
}

std::vector<float> Ltx2StgDelta(const float* cond, const float* perturbed, int64_t count,
                                double scale) {
  const float gain = static_cast<float>(scale);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const size_t k = static_cast<size_t>(i);
    out[k] = gain * (cond[k] - perturbed[k]);
  }
  return out;
}

bool Ltx2MultiModalGuiderParams::DoUnconditionalGeneration() const {
  return !IsClose(cfg_scale, 1.0);
}
bool Ltx2MultiModalGuiderParams::DoPerturbedGeneration() const {
  return !IsClose(stg_scale, 0.0);
}
bool Ltx2MultiModalGuiderParams::DoIsolatedModalityGeneration() const {
  return !IsClose(modality_scale, 1.0);
}
bool Ltx2MultiModalGuiderParams::ShouldSkipStep(int64_t step) const {
  // guiders.py:287-291. `skip_step == 0` means "never skip"; otherwise every
  // `skip_step + 1`-th step runs and the rest are skipped.
  if (skip_step == 0) return false;
  return step % (skip_step + 1) != 0;
}

std::vector<float> Ltx2MultiModalGuidance(const Ltx2MultiModalGuiderParams& params,
                                          const float* cond, const float* uncond_text,
                                          const float* uncond_perturbed,
                                          const float* uncond_modality, int64_t count) {
  Require(cond != nullptr, "ltx2 MultiModalGuider: `cond` is required");
  // A null stands for upstream's `float` union member at its only reachable
  // value: the 0.0 a disabled arm passes (guiders.py:247-249).
  auto at = [](const float* buffer, size_t k) {
    return buffer != nullptr ? buffer[k] : 0.0f;
  };
  const float cfg_gain = static_cast<float>(params.cfg_scale - 1.0);
  const float stg_gain = static_cast<float>(params.stg_scale);
  const float modality_gain = static_cast<float>(params.modality_scale - 1.0);

  std::vector<float> pred(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const size_t k = static_cast<size_t>(i);
    // guiders.py:261-266, summed left to right exactly as written.
    float value = cond[k];
    value += cfg_gain * (cond[k] - at(uncond_text, k));
    value += stg_gain * (cond[k] - at(uncond_perturbed, k));
    value += modality_gain * (cond[k] - at(uncond_modality, k));
    pred[k] = value;
  }

  if (params.rescale_scale != 0.0) {
    // :268-271. torch's `std` is the UNBIASED (N-1) estimator by default; the
    // biased one would be a small, everywhere, resolution-dependent gain error.
    auto unbiased_std = [count](const float* buffer) {
      double mean = 0.0;
      for (int64_t i = 0; i < count; ++i) mean += static_cast<double>(buffer[i]);
      mean /= static_cast<double>(count);
      double sum_sq = 0.0;
      for (int64_t i = 0; i < count; ++i) {
        const double d = static_cast<double>(buffer[i]) - mean;
        sum_sq += d * d;
      }
      return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count - 1)));
    };
    const float factor_raw = unbiased_std(cond) / unbiased_std(pred.data());
    const float factor = static_cast<float>(params.rescale_scale) * factor_raw +
                         static_cast<float>(1.0 - params.rescale_scale);
    for (float& value : pred) value *= factor;
  }
  return pred;
}

std::vector<float> Ltx2Guidance(Ltx2GuiderKind kind, const Ltx2MultiModalGuiderParams& params,
                                const float* cond, const float* uncond_text,
                                const float* uncond_perturbed, const float* uncond_modality,
                                int64_t count) {
  switch (kind) {
    case Ltx2GuiderKind::kCfg:
      return Ltx2CfgDelta(cond, uncond_text, count, params.cfg_scale);
    case Ltx2GuiderKind::kStg:
      return Ltx2StgDelta(cond, uncond_perturbed, count, params.stg_scale);
    case Ltx2GuiderKind::kMultiModal:
      return Ltx2MultiModalGuidance(params, cond, uncond_text, uncond_perturbed,
                                    uncond_modality, count);
    case Ltx2GuiderKind::kCfgStarRescaling:
    case Ltx2GuiderKind::kLtxApg:
    case Ltx2GuiderKind::kLegacyStatefulApg:
      break;
  }
  // UNREACHABLE, not unshapeable — see the header. Nothing in the LTX-2 tree
  // constructs these three; they appear only at their own `class` statements.
  // (An earlier revision refused them on a shape argument instead. That premise
  // was measured and is FALSE: at B = 1 the (B, 1) coefficient is a scalar and
  // composes correctly on any rank. The real predicate is
  // `B > 1 && shape[-2] not in {1, B}` and it is gated in the test, not here.)
  Refuse(
      "ltx2 guidance: CFGStarRescalingGuider / LtxAPGGuider / LegacyStatefulAPGGuider are not "
      "ported. Nothing upstream constructs them: all three appear in the LTX-2 tree only at "
      "their own class statements (guiders.py:31, 78, 129), and every pipeline builds "
      "MultiModalGuider from MultiModalGuiderParams (ltx-pipelines utils/constants.py:49-68). "
      "This arm is owed and recorded in .agents/specs/ltx-2-5.md phase L5 and "
      ".agents/porting-inventory.md 9.18(b).");
}

const Ltx2MultiModalGuiderParams& Ltx2GuiderParamsForSigma(
    const std::vector<Ltx2GuiderSigmaBin>& bins, double sigma) {
  Require(!bins.empty(), "ltx2 guider factory: params_by_sigma must be non-empty");
  // guiders.py:226-230. Keys sorted DESCENDING; the bin is the smallest key that
  // is still >= sigma, and a sigma above every key falls back to the largest.
  // Sorting here rather than trusting the caller mirrors `from_dict` (:329).
  const Ltx2GuiderSigmaBin* largest = &bins[0];
  const Ltx2GuiderSigmaBin* chosen = nullptr;
  for (const Ltx2GuiderSigmaBin& bin : bins) {
    if (bin.sigma_upper_bound > largest->sigma_upper_bound) largest = &bin;
    if (bin.sigma_upper_bound >= sigma) {
      if (chosen == nullptr || bin.sigma_upper_bound < chosen->sigma_upper_bound) chosen = &bin;
    }
  }
  return chosen != nullptr ? chosen->params : largest->params;
}

// ---------------------------------------------------------------------------
// Perturbations
// ---------------------------------------------------------------------------

bool Ltx2PerturbationConfig::IsPerturbed(Ltx2PerturbationType type, int64_t block) const {
  for (const Ltx2Perturbation& perturbation : perturbations) {
    if (perturbation.type != type) continue;
    if (perturbation.all_blocks) return true;
    if (std::find(perturbation.blocks.begin(), perturbation.blocks.end(), block) !=
        perturbation.blocks.end()) {
      return true;
    }
  }
  return false;
}

Ltx2BatchedPerturbationConfig::Ltx2BatchedPerturbationConfig(
    const std::vector<Ltx2PerturbationConfig>& configs, int64_t num_blocks)
    : batch_(static_cast<int64_t>(configs.size())), num_blocks_(num_blocks) {
  // perturbations.py:77-84 — [type, block, sample], 1 = KEEP.
  masks_.assign(static_cast<size_t>(kLtx2PerturbationTypeCount * num_blocks * batch_), 1);
  for (int64_t direction = 0; direction < kLtx2PerturbationTypeCount; ++direction) {
    for (int64_t block = 0; block < num_blocks; ++block) {
      for (int64_t sample = 0; sample < batch_; ++sample) {
        const bool perturbed = configs[static_cast<size_t>(sample)].IsPerturbed(
            static_cast<Ltx2PerturbationType>(direction), block);
        masks_[static_cast<size_t>((direction * num_blocks + block) * batch_ + sample)] =
            perturbed ? 0 : 1;
      }
    }
  }
}

Ltx2BatchedPerturbationConfig Ltx2BatchedPerturbationConfig::Empty(int64_t batch,
                                                                   int64_t num_blocks) {
  return Ltx2BatchedPerturbationConfig(
      std::vector<Ltx2PerturbationConfig>(static_cast<size_t>(batch)), num_blocks);
}

std::vector<int32_t> Ltx2BatchedPerturbationConfig::Mask(Ltx2PerturbationType type,
                                                         int64_t block) const {
  const int64_t base = (static_cast<int64_t>(type) * num_blocks_ + block) * batch_;
  return std::vector<int32_t>(masks_.begin() + base, masks_.begin() + base + batch_);
}

bool Ltx2BatchedPerturbationConfig::AnyInBatch(Ltx2PerturbationType type, int64_t block) const {
  const std::vector<int32_t> mask = Mask(type, block);
  return std::any_of(mask.begin(), mask.end(), [](int32_t v) { return v == 0; });
}

bool Ltx2BatchedPerturbationConfig::AllInBatch(Ltx2PerturbationType type, int64_t block) const {
  const std::vector<int32_t> mask = Mask(type, block);
  return std::all_of(mask.begin(), mask.end(), [](int32_t v) { return v == 0; });
}

Ltx2BatchedPerturbationConfig Ltx2BatchedPerturbationConfig::BatchSlice(int64_t start,
                                                                       int64_t end) const {
  Require(start >= 0 && end <= batch_ && start <= end,
          "ltx2 perturbation config: batch slice [" + std::to_string(start) + ", " +
              std::to_string(end) + ") is outside a batch of " + std::to_string(batch_));
  Ltx2BatchedPerturbationConfig sliced;
  sliced.batch_ = end - start;
  sliced.num_blocks_ = num_blocks_;
  sliced.masks_.reserve(
      static_cast<size_t>(kLtx2PerturbationTypeCount * num_blocks_ * sliced.batch_));
  for (int64_t direction = 0; direction < kLtx2PerturbationTypeCount; ++direction) {
    for (int64_t block = 0; block < num_blocks_; ++block) {
      const int64_t base = (direction * num_blocks_ + block) * batch_;
      for (int64_t sample = start; sample < end; ++sample) {
        sliced.masks_.push_back(masks_[static_cast<size_t>(base + sample)]);
      }
    }
  }
  return sliced;
}

// ---------------------------------------------------------------------------
// Patchifiers
// ---------------------------------------------------------------------------

int64_t Ltx2VideoTokenCount(const Ltx2VideoLatentShape& shape, int64_t patch_size) {
  // patchifiers.py:24-25 — `prod(shape[2:]) // prod(patch_size)`, and the
  // temporal patch is always 1.
  return shape.frames * shape.height * shape.width / (patch_size * patch_size);
}

namespace {

void RequireVideoGeometry(const Ltx2VideoLatentShape& shape, int64_t patch_size) {
  Require(patch_size >= 1, "ltx2 video patchifier: patch_size must be >= 1");
  Require(shape.height % patch_size == 0 && shape.width % patch_size == 0,
          "ltx2 video patchifier: height " + std::to_string(shape.height) + " and width " +
              std::to_string(shape.width) + " must both be divisible by patch_size " +
              std::to_string(patch_size));
}

}  // namespace

std::vector<float> Ltx2VideoPatchify(const float* latent, const Ltx2VideoLatentShape& shape,
                                     int64_t patch_size) {
  RequireVideoGeometry(shape, patch_size);
  const int64_t gh = shape.height / patch_size;
  const int64_t gw = shape.width / patch_size;
  const int64_t tokens = shape.frames * gh * gw;
  const int64_t token_dim = shape.channels * patch_size * patch_size;
  std::vector<float> out(static_cast<size_t>(shape.batch * tokens * token_dim));

  // `b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)` with p1 == 1.
  for (int64_t b = 0; b < shape.batch; ++b) {
    for (int64_t f = 0; f < shape.frames; ++f) {
      for (int64_t h = 0; h < gh; ++h) {
        for (int64_t w = 0; w < gw; ++w) {
          const int64_t token = (f * gh + h) * gw + w;
          for (int64_t c = 0; c < shape.channels; ++c) {
            for (int64_t p2 = 0; p2 < patch_size; ++p2) {
              for (int64_t p3 = 0; p3 < patch_size; ++p3) {
                const int64_t feature = (c * patch_size + p2) * patch_size + p3;
                const int64_t src =
                    (((b * shape.channels + c) * shape.frames + f) * shape.height +
                     h * patch_size + p2) *
                        shape.width +
                    w * patch_size + p3;
                out[static_cast<size_t>((b * tokens + token) * token_dim + feature)] =
                    latent[src];
              }
            }
          }
        }
      }
    }
  }
  return out;
}

std::vector<float> Ltx2VideoUnpatchify(const float* tokens, const Ltx2VideoLatentShape& shape,
                                       int64_t patch_size) {
  RequireVideoGeometry(shape, patch_size);
  const int64_t gh = shape.height / patch_size;
  const int64_t gw = shape.width / patch_size;
  const int64_t token_count = shape.frames * gh * gw;
  const int64_t token_dim = shape.channels * patch_size * patch_size;
  std::vector<float> out(
      static_cast<size_t>(shape.batch * shape.channels * shape.frames * shape.height *
                          shape.width));

  // `b (f h w) (c p q) -> b c f (h p) (w q)`. `p` takes HEIGHT and `q` takes
  // WIDTH; they are not interchangeable (ops.py's unpatchify makes the same
  // mistake available with r and q).
  for (int64_t b = 0; b < shape.batch; ++b) {
    for (int64_t f = 0; f < shape.frames; ++f) {
      for (int64_t h = 0; h < gh; ++h) {
        for (int64_t w = 0; w < gw; ++w) {
          const int64_t token = (f * gh + h) * gw + w;
          for (int64_t c = 0; c < shape.channels; ++c) {
            for (int64_t p = 0; p < patch_size; ++p) {
              for (int64_t q = 0; q < patch_size; ++q) {
                const int64_t feature = (c * patch_size + p) * patch_size + q;
                const int64_t dst =
                    (((b * shape.channels + c) * shape.frames + f) * shape.height +
                     h * patch_size + p) *
                        shape.width +
                    w * patch_size + q;
                out[static_cast<size_t>(dst)] =
                    tokens[(b * token_count + token) * token_dim + feature];
              }
            }
          }
        }
      }
    }
  }
  return out;
}

std::vector<int64_t> Ltx2VideoPatchBounds(const Ltx2VideoLatentShape& shape,
                                          int64_t patch_size) {
  RequireVideoGeometry(shape, patch_size);
  Require(shape.frames > 0 && shape.height > 0 && shape.width > 0 && shape.batch > 0,
          "ltx2 video patchifier: frames, height, width and batch must all be positive");
  const int64_t gh = shape.height / patch_size;
  const int64_t gw = shape.width / patch_size;
  const int64_t tokens = shape.frames * gh * gw;
  // [batch, 3, tokens, 2], axis order (frame, height, width).
  std::vector<int64_t> out(static_cast<size_t>(shape.batch * 3 * tokens * 2));
  const int64_t sizes[3] = {1, patch_size, patch_size};

  for (int64_t f = 0; f < shape.frames; ++f) {
    for (int64_t h = 0; h < gh; ++h) {
      for (int64_t w = 0; w < gw; ++w) {
        const int64_t token = (f * gh + h) * gw + w;
        const int64_t starts[3] = {f, h * patch_size, w * patch_size};
        for (int64_t axis = 0; axis < 3; ++axis) {
          for (int64_t b = 0; b < shape.batch; ++b) {
            const size_t base = static_cast<size_t>(((b * 3 + axis) * tokens + token) * 2);
            out[base] = starts[axis];
            out[base + 1] = starts[axis] + sizes[axis];
          }
        }
      }
    }
  }
  return out;
}

std::vector<int64_t> Ltx2PixelCoords(const std::vector<int64_t>& latent_coords, int64_t batch,
                                     int64_t tokens, const Ltx2ScaleFactors& factors,
                                     bool causal_fix) {
  Require(latent_coords.size() == static_cast<size_t>(batch * 3 * tokens * 2),
          "ltx2 pixel coords: latent_coords must be [batch, 3, tokens, 2]");
  const int64_t scale[3] = {factors.time, factors.height, factors.width};
  std::vector<int64_t> out(latent_coords.size());
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t axis = 0; axis < 3; ++axis) {
      for (int64_t t = 0; t < tokens * 2; ++t) {
        const size_t index = static_cast<size_t>(((b * 3 + axis) * tokens * 2) + t);
        int64_t value = latent_coords[index] * scale[axis];
        // patchifiers.py:166-169 — the TEMPORAL axis only, and clamped at 0. The
        // VAE's stride for the very first frame is 1, not `time`.
        if (causal_fix && axis == 0) value = std::max<int64_t>(value + 1 - factors.time, 0);
        out[index] = value;
      }
    }
  }
  return out;
}

std::vector<float> Ltx2AudioPatchify(const float* latent, const Ltx2AudioLatentShape& shape) {
  // `b c t f -> b t (c f)`.
  const int64_t token_dim = shape.channels * shape.mel_bins;
  std::vector<float> out(static_cast<size_t>(shape.batch * shape.frames * token_dim));
  for (int64_t b = 0; b < shape.batch; ++b) {
    for (int64_t t = 0; t < shape.frames; ++t) {
      for (int64_t c = 0; c < shape.channels; ++c) {
        for (int64_t f = 0; f < shape.mel_bins; ++f) {
          const int64_t src = ((b * shape.channels + c) * shape.frames + t) * shape.mel_bins + f;
          out[static_cast<size_t>((b * shape.frames + t) * token_dim + c * shape.mel_bins + f)] =
              latent[src];
        }
      }
    }
  }
  return out;
}

std::vector<float> Ltx2AudioUnpatchify(const float* tokens, const Ltx2AudioLatentShape& shape) {
  const int64_t token_dim = shape.channels * shape.mel_bins;
  std::vector<float> out(
      static_cast<size_t>(shape.batch * shape.channels * shape.frames * shape.mel_bins));
  for (int64_t b = 0; b < shape.batch; ++b) {
    for (int64_t t = 0; t < shape.frames; ++t) {
      for (int64_t c = 0; c < shape.channels; ++c) {
        for (int64_t f = 0; f < shape.mel_bins; ++f) {
          const int64_t dst = ((b * shape.channels + c) * shape.frames + t) * shape.mel_bins + f;
          out[static_cast<size_t>(dst)] =
              tokens[(b * shape.frames + t) * token_dim + c * shape.mel_bins + f];
        }
      }
    }
  }
  return out;
}

std::vector<float> Ltx2AudioPatchTimings(const Ltx2AudioLatentShape& shape,
                                         const Ltx2AudioPatchifierParams& params) {
  // _get_audio_latent_time_in_sec (patchifiers.py:216-249), evaluated twice — once
  // from `shift` and once from `shift + 1` — and stacked into [start, end).
  auto seconds = [&params](int64_t latent_frame) {
    float mel = static_cast<float>(latent_frame) *
                static_cast<float>(params.audio_latent_downsample_factor);
    if (params.is_causal) {
      // The "+1" is what makes the timestamp the first FULLY available sample.
      mel = std::max(mel + 1.0f - static_cast<float>(params.audio_latent_downsample_factor),
                     0.0f);
    }
    return mel * static_cast<float>(params.hop_length) / static_cast<float>(params.sample_rate);
  };

  // [batch, 1, frames, 2].
  std::vector<float> out(static_cast<size_t>(shape.batch * shape.frames * 2));
  for (int64_t b = 0; b < shape.batch; ++b) {
    for (int64_t t = 0; t < shape.frames; ++t) {
      const size_t base = static_cast<size_t>((b * shape.frames + t) * 2);
      out[base] = seconds(params.shift + t);
      out[base + 1] = seconds(params.shift + 1 + t);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The recipes
// ---------------------------------------------------------------------------

std::vector<int64_t> Ltx2ParseModelVersion(const std::string& version) {
  // loader/helpers.py:74-81. Parsing stops at the first dot-separated component
  // that is not a plain integer, so "2.3.rc1" is (2, 3) and "banana" is {}.
  std::vector<int64_t> parts;
  if (version.empty()) return parts;
  size_t start = 0;
  while (start <= version.size()) {
    const size_t dot = version.find('.', start);
    const std::string part =
        version.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
    if (part.empty() ||
        !std::all_of(part.begin(), part.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) {
      break;
    }
    parts.push_back(std::stoll(part));
    if (dot == std::string::npos) break;
    start = dot + 1;
  }
  return parts;
}

Ltx2PipelineParams Ltx2Params20() {
  // utils/constants.py:40-80 (PipelineParams' own defaults).
  Ltx2PipelineParams params;
  params.video_guider.cfg_scale = 3.0;
  params.video_guider.stg_scale = 1.0;
  params.video_guider.rescale_scale = 0.7;
  params.video_guider.modality_scale = 3.0;
  params.video_guider.skip_step = 0;
  params.video_guider.stg_blocks = {29};
  params.audio_guider.cfg_scale = 7.0;
  params.audio_guider.stg_scale = 1.0;
  params.audio_guider.rescale_scale = 0.7;
  params.audio_guider.modality_scale = 3.0;
  params.audio_guider.skip_step = 0;
  params.audio_guider.stg_blocks = {29};
  return params;
}

Ltx2PipelineParams Ltx2Params23() {
  // utils/constants.py:83-88 — 2.0's params with 30 steps and STG on block 28.
  Ltx2PipelineParams params = Ltx2Params20();
  params.num_inference_steps = 30;
  params.video_guider.stg_blocks = {28};
  params.audio_guider.stg_blocks = {28};
  return params;
}

Ltx2PipelineParams Ltx2Params24() {
  // utils/constants.py:124 — 2.3's lineage, moving only the image CRF. Deriving
  // it from 2.0 instead would silently hand a 2.4 checkpoint the 2.0 step count
  // and STG block, which upstream's comment calls out explicitly.
  Ltx2PipelineParams params = Ltx2Params23();
  params.default_image_crf = 18;
  return params;
}

Ltx2PipelineParams Ltx2Params23Hq() {
  // utils/constants.py:95-115. A plain constant upstream, not a `replace` of
  // anything: it overrides every knob that varies between generations.
  Ltx2PipelineParams params;
  params.num_inference_steps = 15;
  params.stage_1_height = 1088 / 2;
  params.stage_1_width = 1920 / 2;
  params.video_guider.cfg_scale = 3.0;
  params.video_guider.stg_scale = 0.0;
  params.video_guider.rescale_scale = 0.45;
  params.video_guider.modality_scale = 3.0;
  params.video_guider.skip_step = 0;
  params.video_guider.stg_blocks = {};
  params.audio_guider.cfg_scale = 7.0;
  params.audio_guider.stg_scale = 0.0;
  params.audio_guider.rescale_scale = 1.0;
  params.audio_guider.modality_scale = 3.0;
  params.audio_guider.skip_step = 0;
  params.audio_guider.stg_blocks = {};
  return params;
}

Ltx2PipelineParams Ltx2DetectPipelineParams(const std::string& version) {
  // utils/constants.py:130-179 — the newest generation the version is at or
  // above, so an unrecognised NEWER version inherits the closest known one
  // instead of falling back to 2.0. This is what gives LTX-2.5 the 2.4 params.
  const std::vector<int64_t> parsed =
      Ltx2ParseModelVersion(NormalizeVersionSeparator(version));
  if (CompareVersion(parsed, {2, 4}) >= 0) return Ltx2Params24();
  if (CompareVersion(parsed, {2, 3}) >= 0) return Ltx2Params23();
  return Ltx2Params20();
}

bool Ltx2ShouldUseAncestralSampler(const std::string& version) {
  // distilled.py:76-84, through `detect_model_version`'s separator normalization.
  const std::vector<int64_t> parsed =
      Ltx2ParseModelVersion(NormalizeVersionSeparator(version));
  return CompareVersion(parsed, {kLtx2AncestralSinceMajor, kLtx2AncestralSinceMinor}) >= 0;
}

int64_t Ltx2PhaseRecipe::num_inference_steps() const {
  // ltx2_recipes.py:48-50 — `None` when the schedule is not explicit.
  return sigmas.empty() ? -1 : static_cast<int64_t>(sigmas.size()) - 1;
}

void Ltx2AssertResolution(int64_t height, int64_t width, int64_t divisor) {
  Require(divisor >= 1, "ltx2 resolution: the divisor must be at least 1, got " +
                            std::to_string(divisor));
  // Upstream checks both axes against one divisor and names both in one message
  // (helpers.py:546-551). Naming the OFFENDING axis as well, because a caller who
  // passed two numbers cannot tell from "(80x64) is not divisible by 64" which of
  // them to change — and the fix is a different number on each axis.
  //
  // The axis phrase carries its own verb rather than being a bare noun dropped
  // into one template. A shared "; the X is not" tail reads as a constant to a
  // test: `msg.find("width")` is satisfied by the "(width x height)" label this
  // message always carries, so a needle spelt that way stays green with the two
  // names SWAPPED. The needle a test can hold has to be the phrase, not the word.
  if (height % divisor != 0 || width % divisor != 0) {
    const std::string bad =
        (height % divisor != 0)
            ? (width % divisor != 0 ? "the width and height are not" : "the height is not")
            : "the width is not";
    // A suggestion has to be a size the caller can actually pass. Flooring an axis
    // that is BELOW the divisor yields 0, and 0 is refused three lines further on
    // by the lower bound, so the old wording sent a width-32 caller from one
    // refusal to another. When either axis floors away there is no legal size at
    // or below the request at all, and saying so — with the smallest legal size —
    // is the honest answer.
    const int64_t floor_w = (width / divisor) * divisor;
    const int64_t floor_h = (height / divisor) * divisor;
    const std::string suggestion =
        (floor_w == 0 || floor_h == 0)
            ? ("No legal size at or below the request exists: an axis under " +
               std::to_string(divisor) + " floors to zero latent cells. Smallest legal size: " +
               std::to_string(divisor) + "x" + std::to_string(divisor))
            : ("Nearest legal size at or below the request: " + std::to_string(floor_w) + "x" +
               std::to_string(floor_h));
    Refuse("ltx-2.5 video: the requested resolution " + std::to_string(width) + "x" +
           std::to_string(height) + " (width x height) is not divisible by " +
           std::to_string(divisor) + "; " + bad + ". That divisor is this " +
           "recipe's worst phase downscale times the VAE's spatial factor, so both axes " +
           "must be multiples of " + std::to_string(divisor) + ". " + suggestion +
           " (`assert_resolution`, ltx-pipelines utils/helpers.py:540-551). Rounding it " +
           "here would render a clip at a size nobody asked for");
  }
}

int64_t Ltx2PipelineRecipe::max_spatial_downscale() const {
  int64_t worst = 1;
  for (const Ltx2PhaseRecipe& phase : phases) {
    worst = std::max(worst, phase.spatial_downscale);
  }
  return worst;
}

namespace {

// vLLM-Omni's LTX_DEFAULT_NEGATIVE_PROMPT (ltx2_recipes.py:11-23).
const char* const kOmniNegativePrompt =
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing "
    "facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, "
    "inconsistent perspective, camera shake, incorrect depth of field, background too sharp, "
    "background clutter, distracting reflections, harsh shadows, inconsistent lighting "
    "direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, "
    "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong "
    "gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic "
    "voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, "
    "repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural "
    "transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, "
    "cinematic oversaturation, stylized filters, or AI artifacts.";

// Lightricks' DEFAULT_NEGATIVE_PROMPT (utils/constants.py:186-199). It is
// vLLM-Omni's with FIVE leading tags the other lacks. The two references
// disagree; spec section 3 says the disagreement is the finding, so both are
// kept and each recipe row takes its own source's value.
const char* const kLightricksNegativePromptPrefix =
    "has_subtitles, has_blurbox, transition from black, transition to black, "
    "speech_ending_short, ";

std::string LightricksNegativePrompt() {
  return std::string(kLightricksNegativePromptPrefix) + kOmniNegativePrompt;
}

// utils/constants.py:17-23 / ltx2_recipes.py:25-26. Both references carry these
// byte-for-byte; the suite asserts that agreement rather than assuming it.
const std::vector<float>& DistilledSigmas() {
  static const std::vector<float> sigmas = {1.0f,      0.99375f, 0.9875f,   0.98125f, 0.975f,
                                            0.909375f, 0.725f,   0.421875f, 0.0f};
  return sigmas;
}
const std::vector<float>& Stage2DistilledSigmas() {
  static const std::vector<float> sigmas = {0.909375f, 0.725f, 0.421875f, 0.0f};
  return sigmas;
}

// `_official_guidance` (ltx2_recipes.py:90-106) built from a resolved
// PipelineParams, so the one_stage rows and the Lightricks constants cannot
// drift apart.
Ltx2PhaseRecipe OneStagePhase(const Ltx2PipelineParams& params) {
  Ltx2PhaseRecipe phase;
  phase.name = "generate";
  phase.video_guidance = params.video_guider;
  phase.audio_guidance = params.audio_guider;
  // #1013. This was left at the struct's 0.0 default, and 0.0 is not "no extra
  // noise": `Ltx2GaussianNoise` is `latent + noise_scale * (noise - latent)`, so
  // at 0.0 the state stays exactly as `create_initial_state` wrote it, which
  // with no initial latent is ALL ZEROS. A one_stage render therefore denoised a
  // zero tensor.
  //
  // Upstream's `ModalitySpec.noise_scale` defaults to 1.0
  // (ltx-pipelines/utils/types.py:110) and `TI2VidOneStagePipeline.__call__`
  // constructs both specs without it (ti2vid_one_stage.py:233-239), so 1.0 is
  // what reaches `GaussianNoiser.__call__`'s `torch.lerp(latent, noise,
  // noise_scale)` (components/noisers.py:31). The two neighbouring recipes
  // already set it explicitly, which is what made the omission legible.
  //
  // No gate saw it because every end-to-end test loads `distilled_two_stage`,
  // and a zero-initialized denoise still returns a finite clip of the right
  // size, frame count and sample rate.
  phase.noise_scale = 1.0;
  return phase;
}

Ltx2PipelineRecipe OneStageRecipe(const Ltx2PipelineParams& params,
                                  const std::string& negative_prompt) {
  Ltx2PipelineRecipe recipe;
  recipe.phases = {OneStagePhase(params)};
  recipe.height = params.stage_1_height;
  recipe.width = params.stage_1_width;
  recipe.num_frames = params.num_frames;
  recipe.frame_rate = params.frame_rate;
  recipe.num_inference_steps = params.num_inference_steps;
  recipe.default_image_crf = params.default_image_crf;
  recipe.negative_prompt = negative_prompt;
  return recipe;
}

// `T2AOneStagePipeline` (t2a_one_stage.py:43). Built FROM `OneStageRecipe`
// rather than beside it, because upstream's difference between the two is not in
// the schedule: both hard-code `LTX2Scheduler()` (`:67` against
// ti2vid_one_stage.py:81) and both take `num_inference_steps` from the same
// `PipelineParams`. What differs is that there is no video.
//
// The geometry fields are left at the params table's values and are DEAD on this
// recipe — upstream fills the same slots with a 512x512 placeholder whose height
// and width it documents as unused (t2a_one_stage.py:37-40). Only `num_frames`
// and `frame_rate` are read, and they are read to derive the audio DURATION
// (`AudioLatentShape.from_video_pixel_shape`, types.py:184-200).
//
// The VIDEO guider is deliberately left at its default and is never consumed:
// upstream's T2A CLI constructs ONE `MultiModalGuiderParams` and it is the audio
// one (`:196-205`). Zeroing it here would look tidier and would be a fabricated
// value; leaving the params table's own entry says "this recipe does not read
// it" without inventing a number.
Ltx2PipelineRecipe T2aOneStageRecipe(const Ltx2PipelineParams& params,
                                     const std::string& negative_prompt) {
  Ltx2PipelineRecipe recipe = OneStageRecipe(params, negative_prompt);
  recipe.audio_only = true;
  // `video_output_phase` is already -1 on a fresh recipe; restated because on
  // THIS recipe it is a statement rather than a default, and a later edit that
  // gave the field a real value would otherwise silently ask for a video output
  // from a pipeline that produces none.
  recipe.video_output_phase = -1;
  // `modality_scale=1.0` — the CLI pins it, and says why: "Audio-only generation
  // has no video modality, so the video->audio (v2a) cross-modal guidance is
  // meaningless here. 1.0 disables it" (t2a_one_stage.py:200-202). It is the ONE
  // guider field T2A overrides against the params table's 3.0, and 1.0 is exactly
  // the value `do_isolated_modality_generation` reads as OFF
  // (guiders.py:283-285). Applied at the recipe rather than at the call site so
  // no caller can reach the isolated-modality forward this port does not have.
  recipe.phases[0].audio_guidance.modality_scale = 1.0;
  return recipe;
}

// LTX_POSITIVE_ONLY_RECIPE (ltx2_recipes.py:116-124): every guidance knob at its
// no-op value, and the official sigma schedule turned OFF.
Ltx2PipelineRecipe PositiveOnlyRecipe() {
  Ltx2PipelineRecipe recipe;
  Ltx2PhaseRecipe phase;
  phase.name = "generate";
  phase.use_official_sigma_schedule = false;
  recipe.phases = {phase};
  recipe.negative_prompt = kOmniNegativePrompt;
  return recipe;
}

// LTX2_DISTILLED_TWO_STAGE_RECIPE (ltx2_recipes.py:125-158), plus — for
// generation 2.5 and later — `should_use_ancestral_sampler` (distilled.py:76-84),
// which is the SINGLE thing distilled.py changes for 2.5.
Ltx2PipelineRecipe DistilledTwoStageRecipe(const std::string& version) {
  Ltx2PipelineRecipe recipe;
  const Ltx2PipelineParams params = Ltx2DetectPipelineParams(version);

  Ltx2PhaseRecipe stage1;
  stage1.name = "generate_lowres";
  stage1.spatial_downscale = 2;
  stage1.sigmas = DistilledSigmas();
  stage1.noise_scale = 1.0;
  stage1.allow_guidance_override = false;
  stage1.use_official_sigma_schedule = false;
  if (Ltx2ShouldUseAncestralSampler(version)) {
    stage1.stepper = Ltx2StepperKind::kEulerAncestral;
    stage1.stepper_eta = kLtx2AncestralEta;
    stage1.stepper_s_noise = kLtx2AncestralSNoise;
    stage1.noise_seed_offset = kLtx2AncestralNoiseSeedOffset;
  }

  Ltx2PhaseRecipe stage2;
  stage2.name = "refine";
  stage2.sigmas = Stage2DistilledSigmas();
  // ltx2_recipes.py:146 / distilled.py:305 — stage 2 re-noises to its OWN first
  // sigma, which is what makes the upsampled latent valid at that noise level.
  stage2.noise_scale = Stage2DistilledSigmas().front();
  stage2.input_transform = Ltx2PhaseInputTransform::kSpatialUpsample;
  stage2.allow_guidance_override = false;
  stage2.use_official_sigma_schedule = false;
  // Always deterministic: a 3-step refinement cannot remove freshly injected
  // noise (distilled.py:206-209).
  stage2.stepper = Ltx2StepperKind::kEuler;

  recipe.phases = {stage1, stage2};
  // The distilled arguments describe the FINAL output; stage 1 runs at half.
  recipe.height = params.stage_2_height();
  recipe.width = params.stage_2_width();
  recipe.num_frames = params.num_frames;
  recipe.frame_rate = params.frame_rate;
  recipe.num_inference_steps = static_cast<int64_t>(DistilledSigmas().size()) - 1;
  recipe.default_image_crf = params.default_image_crf;
  recipe.negative_prompt = "";
  recipe.video_output_phase = 1;
  recipe.audio_output_phase = 1;
  recipe.allow_request_sigmas = false;
  recipe.allow_request_latents = false;
  recipe.allow_negative_prompt = false;
  recipe.fixed_num_inference_steps = true;
  return recipe;
}

// DFRPipeline (ltx-pipelines/dfr_pipeline.py:155-561). Row LTX25-DFR-PIPELINE,
// issue #986.
//
// THE SCHEDULE IS THE DISTILLED TWO-STAGE ONE, EXACTLY, and that is a finding
// rather than a shortcut. `DFRPipeline.__call__` defaults `stage_1_sigmas` to
// `DISTILLED_SIGMAS` and `stage_2_sigmas` to `STAGE_2_DISTILLED_SIGMAS`
// (:281-282), runs stage 1 at `width // 2, height // 2` (:319), and re-noises
// stage 2 to `stage_2_sigmas[0]` (:386-391). Every one of those is what
// `DistilledTwoStageRecipe` already carries. DFR differs from the distilled
// two-stage pipeline in its CONDITIONING and its rounds loop, not in its
// schedule — recorded here so a reader does not go looking for a DFR-specific
// sigma set that upstream does not have.
//
// What the recipe DOES carry differently is the phase NAMES, because they are
// what the engine's refusals quote back to a caller, and "generate_lowres" would
// describe the wrong thing on a pipeline whose first stage also invents keyframe
// slots.
Ltx2PipelineRecipe DfrRecipe(const std::string& version) {
  Ltx2PipelineRecipe recipe = DistilledTwoStageRecipe(version);
  if (recipe.phases.size() != 2) {
    Refuse("The DFR recipe is the distilled two-stage recipe with DFR's phase names, and that "
           "recipe just returned " + std::to_string(recipe.phases.size()) +
           " phases. DFR's stage 1 and stage 2 are addressed by INDEX below, so a changed phase "
           "count would rename the wrong ones.");
  }
  recipe.phases[0].name = "dfr_base";
  recipe.phases[1].name = "dfr_detail";
  return recipe;
}

// `RetakePipeline` (retake.py:53). ONE stage at the SOURCE clip's own
// resolution, distilled sigmas, plain Euler, no guidance — see the table comment
// in the header for the line behind each of those, and in particular for why
// this is not `DistilledTwoStageRecipe` with one phase removed.
Ltx2PipelineRecipe RetakeRecipe(const std::string& version) {
  Ltx2PipelineRecipe recipe;
  const Ltx2PipelineParams params = Ltx2DetectPipelineParams(version);

  Ltx2PhaseRecipe stage;
  stage.name = "retake";
  // retake.py:317-318 passes `output_shape.width` / `.height` through, so there
  // is no downscale and no input transform. The engine seeds this phase's
  // initial latent with the encoded source clip.
  stage.spatial_downscale = 1;
  stage.sigmas = DistilledSigmas();
  stage.noise_scale = 1.0;
  stage.allow_guidance_override = false;
  stage.use_official_sigma_schedule = false;
  // NOT the ancestral sampler. `Ltx2ShouldUseAncestralSampler` is
  // `distilled.py:76-84` and reaches retake through nothing:
  // `DiffusionStage.__call__` defaults `stepper` to `EulerDiffusionStep()`
  // (utils/blocks.py:526-527) and `retake.py:313-324` passes neither `stepper`
  // nor `loop`. Selecting it here because the SIGMAS are the distilled ones
  // would be inferring a sampler from a schedule, and the two are chosen by
  // different upstream files.
  stage.stepper = Ltx2StepperKind::kEuler;

  recipe.phases = {stage};
  // Defaults only — the engine overrides all four from the source clip
  // (retake.py:220 -> :317-320). They are populated rather than left at zero so
  // a `retake` recipe inspected on its own reports this checkpoint's geometry
  // instead of nothing.
  recipe.height = params.stage_2_height();
  recipe.width = params.stage_2_width();
  recipe.num_frames = params.num_frames;
  recipe.frame_rate = params.frame_rate;
  recipe.num_inference_steps = static_cast<int64_t>(DistilledSigmas().size()) - 1;
  recipe.default_image_crf = params.default_image_crf;
  // `SimpleDenoiser` (retake.py:291-294) takes no negative context, and
  // `prompts_to_encode` is `[prompt]` alone in the distilled arm (:259).
  recipe.negative_prompt = "";
  recipe.video_output_phase = 0;
  recipe.audio_output_phase = 0;
  recipe.allow_request_sigmas = false;
  recipe.allow_request_latents = false;
  recipe.allow_negative_prompt = false;
  recipe.fixed_num_inference_steps = true;
  return recipe;
}

// `TI2VidTwoStagesHQPipeline` (ti2vid_two_stages_hq.py:59, `__call__` at :174).
// Row LTX25-RES2S-LOOP, issue #921.
//
// ─── WHAT MAKES IT HQ, VERIFIED RATHER THAN INHERITED ────────────────────────
// Two of the differences from `TI2VidTwoStagesPipeline` are the SAMPLER:
// `stepper=Res2sDiffusionStep()` (:258) and
// `loop=res2s_audio_video_denoising_loop` passed to BOTH stages (:292, :335).
// A third is `LTX_2_3_HQ_PARAMS` (utils/constants.py:95-115). THEY ARE NOT THE
// ONLY THREE, and this comment said they were until 2026-08-17. Diffing the two
// files at `fd4ded7f` also shows: stage 1 loads the distilled LoRA at
// `distilled_lora_strength_stage_1` (:92-101, :151-154) where the plain
// pipeline loads none on that stage; stage 1's schedule is derived as
// `execute(latent=empty_latent, steps=...)` (:260-267) against the plain
// pipeline's `execute(steps=...)`, which `schedulers.py:32` turns into a
// resolution-dependent shift instead of the 4096-token default; and
// `GuidedDenoiser` (:271-281) replaces `FactoryGuidedDenoiser`.
//
// Two of those four are ALREADY what this recipe does and one is out of scope.
// The schedule: this engine always derives from `target_tokens`
// (`ltx2_video.cpp`'s `Ltx2SigmaSchedule` call), which is the latent-aware form,
// so stage 1 coincides with upstream here — the divergence, if any, is on the
// PLAIN two-stage arm and is not this recipe's to move. The denoiser: stage 1's
// `video_guidance` below reaches `Ltx2GuidedDenoise`, which is
// `_guided_denoise` — the one function `GuidedDenoiser` and
// `FactoryGuidedDenoiser` share (utils/denoisers.py:61-211) — so the difference
// between the two upstream classes is WHERE the params come from and not what
// runs. The distilled LoRA per stage is out of scope for every LTX row here and
// is named in the row's spec section 2 rather than silently absent.
//
// So this recipe is NOT the distilled two-stage one with different numbers. The
// res_2s loop evaluates the transformer TWICE per step, which is the whole
// reason the preset can afford 15 steps against the 2.4 lineage's 30. A recipe
// that carried these guidance scales and this step count on `kEuler` would
// render a finished, plausible, correctly-sized clip at half the model
// evaluations it was tuned for, and no output check could tell.
//
// ─── 2.5 ONLY, AND NOT BY ANALOGY WITH THE ONE-STAGE ROWS ────────────────────
// `LTX_2_3_HQ_PARAMS` is a plain constant, not a `replace` of a neighbour, and
// upstream says why in its own comment (constants.py:91-94): "it overrides every
// knob that varies between generations, so there is nothing for it to inherit
// from a detected checkpoint". There is therefore no `detect_params` lineage to
// spread this across versions the way `one_stage` and `t2a_one_stage` are
// spread, and the one generation-dependent value it does NOT carry —
// `default_image_crf` — is resolved from the checkpoint by the pipeline's own
// `ImageConditioner` (the same comment), which is `Ltx2DetectPipelineParams`
// here.
Ltx2PipelineRecipe Res2sTwoStageRecipe(const std::string& version) {
  Ltx2PipelineRecipe recipe;
  const Ltx2PipelineParams params = Ltx2Params23Hq();

  Ltx2PhaseRecipe stage1;
  stage1.name = "generate_lowres_hq";
  // :238-243 — `width // 2, height // 2`, the same halving the distilled
  // two-stage arm applies.
  stage1.spatial_downscale = 2;
  // :260-267 — `stage_1_sigmas` defaults to None and is then built by
  // `LTX2Scheduler().execute(latent, steps=num_inference_steps)`. So this phase
  // has NO frozen schedule: it is derived, and 15 steps is what derives it. This
  // is the one place this recipe differs in KIND from the distilled two-stage
  // one, whose stage 1 carries `DISTILLED_SIGMAS` and cannot honour a step
  // override.
  stage1.noise_scale = 1.0;
  // :271-281 — a `GuidedDenoiser` with a negative context and the HQ guider
  // params, so guidance is live and a request may override it.
  stage1.video_guidance = params.video_guider;
  stage1.audio_guidance = params.audio_guider;
  stage1.stepper = Ltx2StepperKind::kRes2s;

  Ltx2PhaseRecipe stage2;
  stage2.name = "refine_hq";
  // :193 — `stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS`, a DEFAULT
  // ARGUMENT, so the schedule is frozen for this phase even though stage 1's is
  // not.
  stage2.sigmas = Stage2DistilledSigmas();
  // :327, :332 — both modality specs re-noise to `stage_2_sigmas[0].item()`,
  // which is what makes the upsampled latent valid at that noise level.
  stage2.noise_scale = Stage2DistilledSigmas().front();
  // :297 — `self.upsampler(video_state.latent[:1])`.
  stage2.input_transform = Ltx2PhaseInputTransform::kSpatialUpsample;
  // :316 — `SimpleDenoiser`, "single transformer call, no guidance"
  // (utils/denoisers.py:215). Nothing a request sends can turn guidance back on.
  stage2.allow_guidance_override = false;
  stage2.use_official_sigma_schedule = false;
  // :319/:335 — the SAME stepper and the SAME loop as stage 1. This is where the
  // HQ pipeline parts company with the distilled two-stage one, whose stage 2 is
  // always deterministic Euler because a 3-step refinement cannot remove freshly
  // injected noise (distilled.py:206-209). That argument does not transfer: the
  // HQ pipeline passes `stepper` and `loop` to `self.stage_2` explicitly, and
  // "the schedule is short" is not a reason this port may substitute a different
  // sampler than the one upstream hands it.
  stage2.stepper = Ltx2StepperKind::kRes2s;

  recipe.phases = {stage1, stage2};
  // `assert_resolution(is_two_stage=True)` (:199), and the arguments describe
  // the FINAL output — stage 1 runs at half of it.
  recipe.height = params.stage_2_height();
  recipe.width = params.stage_2_width();
  recipe.num_frames = params.num_frames;
  recipe.frame_rate = params.frame_rate;
  // constants.py:96 — 15, against the 2.4 lineage's 30. Half the steps, and
  // twice the evaluations per step.
  recipe.num_inference_steps = params.num_inference_steps;
  // NOT from the HQ params: constants.py:91-94 says `default_image_crf` is the
  // one generation-dependent value this preset does not fix, and that the
  // pipeline resolves it from the checkpoint instead.
  recipe.default_image_crf = Ltx2DetectPipelineParams(version).default_image_crf;
  // :210 — `self.prompt_encoder([prompt, negative_prompt], ...)`, and stage 1's
  // guider consumes the negative encoding. Unlike the distilled arm, this
  // pipeline HAS a negative prompt.
  recipe.negative_prompt = LightricksNegativePrompt();
  recipe.video_output_phase = 1;
  // :313-314 — "Stage 2 refines video only; discard its audio", so the audio
  // that leaves is STAGE 1's. `video_state, _ = self.stage_2(...)` at :315 is
  // the discard, and `self.audio_decoder(audio_state.latent)` at :339 reads the
  // name stage 1 bound. Writing 1 here would decode the audio the pipeline
  // throws away — a soundtrack that is finite, the right length and the wrong
  // take.
  recipe.audio_output_phase = 0;
  // Stage 1's schedule really is derived from `num_inference_steps`, so a
  // request may set it. Stage 2's is frozen by its own default argument and is
  // unaffected either way, exactly as upstream's two parameters are.
  recipe.allow_request_sigmas = true;
  recipe.allow_request_latents = true;
  recipe.allow_negative_prompt = true;
  recipe.fixed_num_inference_steps = false;
  return recipe;
}

}  // namespace

Ltx2PipelineRecipe ResolveLtx2PipelineRecipe(const std::string& pipeline_kind,
                                             const std::string& model_version) {
  // Keyed on the EXACT pair, never defaulted — see the header. Rows sourced from
  // vLLM-Omni carry its negative prompt; the 2.4 / 2.5 rows are Lightricks' and
  // carry Lightricks'.
  if (pipeline_kind == "one_stage") {
    if (model_version == "2") return OneStageRecipe(Ltx2Params20(), kOmniNegativePrompt);
    if (model_version == "2.3") return OneStageRecipe(Ltx2Params23(), kOmniNegativePrompt);
    if (model_version == "2.4") {
      return OneStageRecipe(Ltx2DetectPipelineParams("2.4"), LightricksNegativePrompt());
    }
    if (model_version == "2.5") {
      return OneStageRecipe(Ltx2DetectPipelineParams("2.5"), LightricksNegativePrompt());
    }
  } else if (pipeline_kind == "distilled_two_stage") {
    if (model_version == "2" || model_version == "2.5") {
      return DistilledTwoStageRecipe(model_version);
    }
  } else if (pipeline_kind == "dfr") {
    // 2.5 only, and deliberately not 2.0. DFR's whole base stage rests on
    // generated keyframe slots, which need a checkpoint declaring
    // `use_keyframes_abs_pos_embedding` (blocks.py:395-419); the 2.0 distilled
    // row exists here for a generation that predates that parameter, so
    // resolving DFR onto it would build a recipe whose first stage the engine
    // must then refuse at load. Refusing at the recipe table names the version.
    if (model_version == "2.5") return DfrRecipe(model_version);
  } else if (pipeline_kind == "res2s_two_stage") {
    // 2.5 only — see `Res2sTwoStageRecipe`: `LTX_2_3_HQ_PARAMS` is a plain
    // constant with no `detect_params` lineage, so there is no second version to
    // resolve it onto.
    if (model_version == "2.5") return Res2sTwoStageRecipe(model_version);
  } else if (pipeline_kind == "dmd2") {
    if (model_version == "2" || model_version == "2.3") return PositiveOnlyRecipe();
  } else if (pipeline_kind == "retake") {
    if (model_version == "2" || model_version == "2.5") return RetakeRecipe(model_version);
  } else if (pipeline_kind == "t2a_one_stage") {
    if (model_version == "2") return T2aOneStageRecipe(Ltx2Params20(), kOmniNegativePrompt);
    if (model_version == "2.3") return T2aOneStageRecipe(Ltx2Params23(), kOmniNegativePrompt);
    if (model_version == "2.4") {
      return T2aOneStageRecipe(Ltx2DetectPipelineParams("2.4"), LightricksNegativePrompt());
    }
    if (model_version == "2.5") {
      return T2aOneStageRecipe(Ltx2DetectPipelineParams("2.5"), LightricksNegativePrompt());
    }
  }
  Refuse("Unsupported LTX pipeline kind/version: '" + pipeline_kind + "'/'" + model_version +
         "'. Recipes are resolved from an EXACT (kind, version) table "
         "(vllm_omni/diffusion/models/ltx2/ltx2_recipes.py:170-175) and never defaulted: a "
         "plausible-but-wrong sigma schedule or guidance scale renders a video rather than "
         "failing.");
}

void Ltx2RefuseUnportedPipelineFeature(Ltx2UnportedPipelineFeature feature) {
  const std::string owed =
      " Recorded as owed in .agents/specs/ltx-2-5.md; grounded against Lightricks/LTX-2 "
      "fd4ded7f in .agents/specs/ltx25-retire-dead-arms.md.";
  // A marker is not a refusal a caller can trip, and saying so is the point: this
  // enum used to read as six live refusals when only ONE has a product call site.
  // "Two" is what this said until review found that the second, `kBetaScheduler`,
  // sits inside `Ltx2Schedule`, which nothing calls — see the case below its
  // enumerator in the header, and `test_ltx2_pipeline`'s "the reachable/marker
  // split matches the source", which derives that from the tree.
  const std::string marker =
      " DECLARED, NOT REQUESTABLE: no request field or load extra asks for this, so nothing "
      "but the out-of-scope ledger reaches this message.";
  switch (feature) {
    case Ltx2UnportedPipelineFeature::kSpatiotemporalUpsampler:
      Refuse("ltx2: the SPATIOTEMPORAL latent upsampler (spatial_upsample AND "
             "temporal_upsample, model/upsampler/model.py:55-59) is not ported. It is a "
             "different operator from the temporal-only arm, which IS ported: 8 * "
             "mid_channels out of its Conv3d and PixelShuffleND(3), against 2 * mid_channels "
             "and PixelShuffleND(1). Owed and recorded in "
             ".agents/specs/ltx25-temporal-upsampler.md section 2, under the campaign "
             ".agents/specs/ltx-2-5.md.");
    case Ltx2UnportedPipelineFeature::kBetaScheduler:
      // A MARKER, not a reachable refusal, and the correction is upstream's. This
      // case label is inside `Ltx2Schedule`, which no product code calls: the
      // engine calls `Ltx2SigmaSchedule` directly, in `ltx2_video.cpp`'s phase
      // driver, and no ABI field, load extra or CLI flag carries a kind. That mirrors LTX-2
      // @ fd4ded7f, where `BetaScheduler` is DEFINED at ltx-core
      // components/schedulers.py:91 and CONSTRUCTED nowhere: all seven pipelines
      // hard-code `LTX2Scheduler()` (ti2vid_one_stage.py:81, ti2vid_two_stages.py:87,
      // ti2vid_two_stages_hq.py:90, a2vid_two_stage.py:78, t2a_one_stage.py:67,
      // keyframe_interpolation.py:82, retake.py:96), and vLLM-Omni @ a4ea67a21 has
      // zero hits for the name. Publishing this as reachable would have promised a
      // selection surface that upstream does not have.
      Refuse("ltx2: BetaScheduler (ltx-core components/schedulers.py:91-120) is not ported. It "
             "inverts a Beta CDF through scipy.stats.beta.ppf. Upstream constructs it nowhere — "
             "every ltx-pipelines entry point hard-codes LTX2Scheduler() — so mirroring upstream "
             "means this port has no scheduler-kind field either." +
             marker + owed);
    case Ltx2UnportedPipelineFeature::kInt8ConvRot:
      // VERIFIED UNREACHABLE so nobody re-audits it, and stated as UNREACHABLE rather
      // than ABSENT because absent is what the first version of this message claimed
      // and it was false. At LTX-2 @ fd4ded7f: `convrot` / `conv_rot` / `quarot` /
      // `spinquant` really are 0 hits, and the four inference quantization kinds are
      // exhaustive (quantization_factory.py:23-26, `assert_never` at :50). But int8
      // is NOT trainer-only. `ltx-kernels` — an inference package — carries a per-row
      // int8 quantize kernel with fp32 scales (blockwise/triton_ops.py:25-50, out
      // dtype `torch.int8` at :43), aliased `rowwise_int_quantize_triton` at :436.
      // That alias is its ONLY reference: blockwise/functional.py:12-18 re-exports
      // five names and not this one, so nothing constructs it. The package is a fork
      // of Lightricks' int8 kernel library retargeted to fp8/fp6/nvfp4 — its custom-op
      // namespace is still literally `q8_kernels_ops` (functional.py:25) — and the
      // int8 half is what was left behind. Nothing wired reaches int8, which is why
      // the disposition is unchanged; only the sentence was wrong.
      Refuse("ltx2: the int8-convrot quantization is out of scope. It is a ComfyUI-ecosystem "
             "format, not an LTX-2 arm: upstream's own inference kinds are fp8-cast, "
             "fp8-scaled-mm, nvfp4-cast and nvfp4-prequant (ltx-pipelines/utils/"
             "quantization_factory.py:23-26), and int8 is UNREACHABLE upstream — trainer-only "
             "for anything wired (ltx-trainer gemma_8bit.py:33-36, quantization.py:11-15), plus "
             "one DEAD per-row int8 quantize kernel in the ltx-kernels inference package "
             "(blockwise/triton_ops.py:35,43, reached only by its own alias at :436)." +
             marker + owed);
    case Ltx2UnportedPipelineFeature::kMultiGpuParallelism:
      // The old spelling was `kCfgParallelism`, which named something upstream does
      // not do. There is no CFG pass to split here in the first place: the distilled
      // recipe denoises with SimpleDenoiser at both stages (distilled.py:266,295),
      // "single transformer call, no guidance" (utils/denoisers.py:3).
      //
      // THIS MESSAGE SAID "three forms and none of them is CFG batching", AND BOTH
      // HALVES WERE WRONG. The count missed `BatchParallelGemmaBuilder`
      // (multigpu/bp_gemma_builder.py:42), a fourth `BuilderProtocol` in the very
      // directory cited above. And the CFG half rested on a grep PATH-FILTERED to
      // the two source trees, which excluded `ltx-pipelines/docs/multigpu/`: re-run
      // over `-- '*multigpu*'` (33 files, the control) it is 5 hits, not 0, two of
      // them prose about CFG at docs/multigpu/gemma.md:103-104.
      //
      // The disposition did not move; it got stronger. gemma.md:104 says the
      // distilled pipeline runs "without CFG", so the one form that WOULD batch a
      // CFG pair is the one upstream tells you not to use for the recipe this port
      // runs. Stating the reason beats asserting an absence — §1.3 of the row spec.
      Refuse("ltx2: single-node multi-GPU parallelism (ltx-pipelines/multigpu) is out of "
             "scope. Upstream has four forms: sequence-parallel "
             "(multigpu/sp_builder.py:25), tiled data parallel "
             "(multigpu/tdp_builder.py:25, upscale stage only), distributed VAE decode "
             "(ltx-core multigpu/vae/distributed_decoder.py:204-256) and batch-parallel Gemma "
             "encoding (multigpu/bp_gemma_builder.py:42 BatchParallelGemmaBuilder), which "
             "partitions a PROMPT LIST across ranks. None is CFG batching, and the fourth is "
             "the closest thing to it: upstream's own docs/multigpu/gemma.md:103-104 calls a "
             "positive+negative pair 'the typical CFG case' and then records that the DISTILLED "
             "pipeline — the one this port runs — takes no negative_prompt and so 'runs without "
             "CFG', leaving nothing to partition. It is a LATENCY tool, "
             "not a memory tool (docs/multigpu/README.md:5-16), and this port targets one "
             "GB10." +
             marker + owed);
  }
  Refuse("ltx2: unknown unported pipeline feature." + owed);
}

}  // namespace vllm
