// LTX-2.5 SAMPLERS — the res_2s second-order denoising loop.
//
// Row: LTX25-RES2S-LOOP. Spec: .agents/specs/ltx25-res2s-loop.md. Issue #921.
// Ported from Lightricks/LTX-2 @ fd4ded7f,
// packages/ltx-pipelines/src/ltx_pipelines/utils/{res2s,samplers}.py.
//
// The header carries the port map, the dtype argument and the warning about
// `phi`. This file carries the arithmetic, anchored line by line.
#include "vllm/model_executor/models/ltx2_samplers.h"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/models/ltx2_pipeline.h"

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& message) { throw std::runtime_error(message); }

void Require(bool condition, const std::string& message) {
  if (!condition) Refuse(message);
}

// `math.factorial` over the only two values `get_res2s_coefficients` reaches.
double Factorial(int64_t j) {
  double out = 1.0;
  for (int64_t k = 2; k <= j; ++k) out *= static_cast<double>(k);
  return out;
}

std::vector<double> ToHp(const std::vector<float>& x) {
  return std::vector<double>(x.begin(), x.end());
}

// `.to(model_dtype)` (samplers.py:370, :375, :431, :433, :442, :445).
std::vector<float> ToModelDtype(const std::vector<double>& x) {
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = static_cast<float>(x[i]);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// The exponential integrator (utils/res2s.py:4-62)
// ---------------------------------------------------------------------------

double Ltx2Phi(int64_t j, double neg_h) {
  Require(j >= 1, "ltx2 res2s phi: j must be >= 1, got " + std::to_string(j));
  // res2s.py:13-16. The threshold is EXACTLY 1e-10 and the comparison is
  // STRICT, so -1e-10 itself takes the formula branch below and returns
  // upstream's cancelled value rather than 1/j!. See the header: this is a
  // guard against dividing by zero, NOT a series expansion, and replacing it
  // with one would make this port disagree with the model's own runtime.
  if (std::fabs(neg_h) < 1e-10) return 1.0 / Factorial(j);

  // res2s.py:19 — the remainder sum_{k<j} z^k / k!, accumulated in the same
  // order Python's `sum` accumulates it (k ascending from 0).
  double remainder = 0.0;
  double power = 1.0;  // neg_h ** 0
  for (int64_t k = 0; k < j; ++k) {
    remainder += power / Factorial(k);
    power *= neg_h;
  }
  // res2s.py:22. `power` now holds neg_h ** j, which is what `neg_h**j` is.
  return (std::exp(neg_h) - remainder) / power;
}

Ltx2Res2sCoefficients Ltx2GetRes2sCoefficients(double h, Ltx2PhiCache& phi_cache, double c2) {
  // res2s.py:37-44 — the cache, keyed on (j, neg_h) exactly as upstream keys it.
  const auto get_phi = [&phi_cache](int64_t j, double neg_h) {
    const std::pair<int64_t, double> key{j, neg_h};
    const auto hit = phi_cache.find(key);
    if (hit != phi_cache.end()) return hit->second;
    const double result = Ltx2Phi(j, neg_h);
    phi_cache.emplace(key, result);
    return result;
  };

  Ltx2Res2sCoefficients coeff;
  // res2s.py:48-50 — a21 = c2 * phi_1(-h * c2).
  const double neg_h_c2 = -h * c2;
  coeff.a21 = c2 * get_phi(1, neg_h_c2);
  // res2s.py:54-56 — b2 = phi_2(-h) / c2.
  const double neg_h_full = -h;
  coeff.b2 = get_phi(2, neg_h_full) / c2;
  // res2s.py:59-60 — b1 = phi_1(-h) - b2. IN THIS ORDER: b2 is computed first
  // upstream and b1 is defined against it, so an implementation that derived b2
  // from b1 would invert the dependency and reorder the cache insertions.
  coeff.b1 = get_phi(1, neg_h_full) - coeff.b2;
  return coeff;
}

// ---------------------------------------------------------------------------
// The noise (utils/samplers.py:155-170)
// ---------------------------------------------------------------------------

std::vector<double> Ltx2Res2sNormalizeNoise(std::vector<double> noise) {
  Require(noise.size() >= 2,
          "ltx2 res2s noise: the normalization divides by an UNBIASED standard "
          "deviation (torch's default), which is undefined for fewer than 2 elements");
  const auto normalize = [](std::vector<double>& x) {
    const double n = static_cast<double>(x.size());
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double sq = 0.0;
    for (const double v : x) sq += (v - mean) * (v - mean);
    // torch's `Tensor.std()` is UNBIASED by default: the denominator is n - 1.
    const double sd = std::sqrt(sq / (n - 1.0));
    for (double& v : x) v = (v - mean) / sd;
  };
  // samplers.py:169 — the global normalize...
  normalize(noise);
  // ...and :170 -> :160-161, `_channelwise_normalize` over the last two dims.
  // On this port's rank-2 [tokens, width] latent those two dims ARE every
  // element, so this repeats the operation above and is the identity up to
  // rounding. Applied anyway, in upstream's order: idempotence here is a
  // property of THIS port's rank, not of the function, and dropping the call
  // would be a divergence that a batched latent would make visible.
  normalize(noise);
  return noise;
}

// ---------------------------------------------------------------------------
// The loop (utils/samplers.py:208-447)
// ---------------------------------------------------------------------------

Ltx2Res2sLoopStats Ltx2Res2sDenoisingLoop(const std::vector<float>& sigmas_in,
                                          Ltx2Res2sModality& video,
                                          Ltx2Res2sModality& audio,
                                          const Ltx2Res2sHooks& hooks,
                                          const Ltx2Res2sLoopParams& params) {
  // samplers.py:257-259.
  Require(video.present || audio.present,
          "ltx2 res2s loop: at least one of video_state or audio_state must be provided "
          "(samplers.py:258-259)");
  Require(static_cast<bool>(hooks.denoise) && static_cast<bool>(hooks.post_process) &&
              static_cast<bool>(hooks.new_noise),
          "ltx2 res2s loop: the denoiser, post_process_latent and new_noise hooks are all "
          "required. Two of the three are upstream PARAMETERS — `denoiser` (samplers.py:214) and "
          "`new_noise_fn` (:220); `post_process_latent` is a module-level import upstream calls "
          "directly (:305, :390, :441), and it is a hook here only because the engine already "
          "owns the mask and the clean latent");
  Require(sigmas_in.size() >= 2,
          "ltx2 res2s loop: a schedule needs at least two sigmas, got " +
              std::to_string(sigmas_in.size()));

  Ltx2Res2sLoopStats stats;
  // samplers.py:279. TAKEN BEFORE THE INJECTION BELOW, which is why it still
  // counts the CALLER's steps after the schedule grows by one.
  stats.full_steps = static_cast<int64_t>(sigmas_in.size()) - 1;

  // samplers.py:280-282 — "inject minimal sigma value to avoid division by
  // zero". The zero is REPLACED by 0.0011 and a new zero appended, so the last
  // full step lands on 0.0011 and the final evaluation happens AT it rather
  // than at a sigma the model cannot be conditioned on.
  std::vector<float> sigmas = sigmas_in;
  const bool terminal_zero = sigmas.back() == 0.0f;
  if (terminal_zero) {
    sigmas.back() = kLtx2Res2sTerminalSigma;
    sigmas.push_back(0.0f);
  }

  // samplers.py:284 — step sizes in log space, on the MODIFIED schedule, with
  // the widening to hp BEFORE the division (`sigmas[1:].to(hp) /
  // sigmas[:-1].to(hp)`). The final entry is +inf when the schedule ends at 0;
  // it is computed anyway, exactly as upstream computes the whole vector, and
  // the loop never reads it.
  std::vector<double> hs(sigmas.size() - 1);
  for (size_t i = 0; i + 1 < sigmas.size(); ++i) {
    hs[i] = -std::log(static_cast<double>(sigmas[i + 1]) / static_cast<double>(sigmas[i]));
  }

  // samplers.py:287-288.
  Ltx2PhiCache phi_cache;
  const double c2 = params.c2;

  // The `sigmas` the step-level injection is handed (samplers.py:415, :425) is
  // this f32 schedule. `Ltx2Res2sStepHp` takes doubles and narrows internally
  // under `kF32Schedule`, so widen once here rather than per step.
  const std::vector<double> sigmas_hp = ToHp(sigmas);

  std::vector<float> denoised_v, denoised_a;
  std::vector<double> x_anchor_v, x_anchor_a, eps_1_v, eps_1_a, x_mid_v, x_mid_a;

  for (int64_t step_idx = 0; step_idx < stats.full_steps; ++step_idx) {
    const size_t s = static_cast<size_t>(step_idx);
    // samplers.py:291-292.
    const double sigma = static_cast<double>(sigmas[s]);
    const double sigma_next = static_cast<double>(sigmas[s + 1]);

    // samplers.py:294-296 — the anchor is the state as it stands, in hp.
    if (video.present) x_anchor_v = ToHp(video.latent);
    if (audio.present) x_anchor_a = ToHp(audio.latent);

    // ── STAGE 1: evaluate at the current point (samplers.py:298-307) ────────
    denoised_v.clear();
    denoised_a.clear();
    // :301 — the loop's OWN counter is this call's `step_index`.
    hooks.denoise(video.latent, audio.latent, sigma, step_idx, denoised_v, denoised_a);
    stats.evaluations += 1;
    stats.eval_sigmas.push_back(sigma);
    stats.eval_step_indices.push_back(step_idx);
    // :304-307 — post_process at the MODEL DTYPE, hence the narrowing back.
    if (video.present && !denoised_v.empty()) {
      denoised_v = ToModelDtype(hooks.post_process(ToHp(denoised_v), true));
    }
    if (audio.present && !denoised_a.empty()) {
      denoised_a = ToModelDtype(hooks.post_process(ToHp(denoised_a), false));
    }

    const double h = hs[s];  // :309
    // :311-312.
    const Ltx2Res2sCoefficients coeff = Ltx2GetRes2sCoefficients(h, phi_cache, c2);
    // :314-315 — "sqrt is a hardcode for c2 = 0.5".
    const double sub_sigma = std::sqrt(sigma * sigma_next);
    // `h * a21` is a scalar-scalar product upstream before it ever meets a
    // tensor (:322), so it is formed once here for the same reason.
    const double h_a21 = h * coeff.a21;

    // ── the substep point (samplers.py:317-332) ─────────────────────────────
    const auto build_mid = [&](bool present, const std::vector<float>& denoised,
                               const std::vector<double>& anchor, std::vector<double>& eps,
                               std::vector<double>& mid) {
      if (!present || denoised.empty()) {
        eps.clear();
        mid.clear();
        return;
      }
      eps.resize(anchor.size());
      mid.resize(anchor.size());
      for (size_t k = 0; k < anchor.size(); ++k) {
        eps[k] = static_cast<double>(denoised[k]) - anchor[k];
        mid[k] = anchor[k] + h_a21 * eps[k];
      }
    };
    build_mid(video.present, denoised_v, x_anchor_v, eps_1_v, x_mid_v);
    build_mid(audio.present, denoised_a, x_anchor_a, eps_1_a, x_mid_a);

    // ── SDE noise injection at the substep (samplers.py:334-352) ────────────
    //
    // VIDEO FIRST, THEN AUDIO, and the order is load bearing: both draws come
    // from ONE generator, so swapping them hands each modality the other's
    // noise. Upstream fixes the order at :337 and :345.
    //
    // eta is 0.5 here whatever the loop's own eta is — ":273-274, substep eta is
    // always default 0.5 for compatibility with the original implementation".
    // The schedule is the f64 pair [sigma, sub_sigma] at index 0 (:342, :350).
    const double substep_sigmas[2] = {sigma, sub_sigma};
    const auto inject = [&](bool present, std::vector<double>& x, bool is_video,
                            const std::vector<double>& sample, const double* sched,
                            int64_t sched_count, int64_t idx, double eta,
                            Ltx2Res2sScheduleWidth width, bool substep) {
      if (!present || x.empty()) return;
      const int64_t count = static_cast<int64_t>(x.size());
      // :187 — the noise is drawn over `state.latent`, i.e. the modality's own
      // element count, before the stepper is entered.
      const std::vector<double> noise = hooks.new_noise(count, is_video, substep);
      Require(static_cast<int64_t>(noise.size()) == count,
              "ltx2 res2s loop: the noise hook returned " + std::to_string(noise.size()) +
                  " values for a " + std::to_string(count) + "-element latent");
      x = Ltx2Res2sStepHp(sample.data(), x.data(), sched, sched_count, idx, count,
                          noise.data(), eta, width);
      // :202-203 — `legacy_mode` is TRUE on every reachable path, so the blend
      // happens AFTER the step rather than the sigmas being converted before it.
      x = hooks.post_process(std::move(x), is_video);
    };
    inject(video.present, x_mid_v, true, x_anchor_v, substep_sigmas, 2, 0,
           kLtx2Res2sSubstepEta, Ltx2Res2sScheduleWidth::kF64Schedule, true);
    inject(audio.present, x_mid_a, false, x_anchor_a, substep_sigmas, 2, 0,
           kLtx2Res2sSubstepEta, Ltx2Res2sScheduleWidth::kF64Schedule, true);

    // ── the bong iteration (samplers.py:354-364) ────────────────────────────
    //
    // A FIXED-POINT REFINEMENT OF THE ANCHOR, and both the anchor and eps_1 are
    // carried into the final combination below. The guard is upstream's, with
    // both comparisons STRICT: a schedule sitting at exactly sigma = 0.03 does
    // not refine.
    //
    // There is no early exit and `bongmath_max_iter` iterations always run.
    // Left as written: the map contracts with ratio `h * a21`, which the h < 0.5
    // guard bounds under 0.25, so it converges to machine precision long before
    // iteration 100 and an early exit would be numerically invisible — which is
    // exactly why removing the parameter would be untestable and is not done.
    if (params.bongmath && h < kLtx2Res2sBongMaxH && sigma > kLtx2Res2sBongMinSigma) {
      stats.bong_steps += 1;
      for (int64_t iter = 0; iter < params.bongmath_max_iter; ++iter) {
        if (!x_mid_v.empty() && !eps_1_v.empty()) {
          for (size_t k = 0; k < x_mid_v.size(); ++k) {
            x_anchor_v[k] = x_mid_v[k] - h_a21 * eps_1_v[k];
            eps_1_v[k] = static_cast<double>(denoised_v[k]) - x_anchor_v[k];
          }
        }
        if (!x_mid_a.empty() && !eps_1_a.empty()) {
          for (size_t k = 0; k < x_mid_a.size(); ++k) {
            x_anchor_a[k] = x_mid_a[k] - h_a21 * eps_1_a[k];
            eps_1_a[k] = static_cast<double>(denoised_a[k]) - x_anchor_a[k];
          }
        }
      }
    }

    // ── STAGE 2: evaluate at the substep point, WITH noise (samplers.py:366-392)
    //
    // THE SECOND EVALUATION. This is the half of the sampler that a token count,
    // a frame count, a shape check and a rendered pixel are all blind to, and
    // dropping it leaves a working-looking renderer running the first-order
    // method at the HQ preset's step count.
    const std::vector<float> mid_v =
        (video.present && !x_mid_v.empty()) ? ToModelDtype(x_mid_v) : video.latent;
    const std::vector<float> mid_a =
        (audio.present && !x_mid_a.empty()) ? ToModelDtype(x_mid_a) : audio.latent;
    std::vector<float> denoised_v2, denoised_a2;
    // A LITERAL ZERO, not `step_idx` (samplers.py:385). Upstream builds a
    // one-element schedule `torch.stack([sub_sigma])` for this call and indexes
    // it at 0, so the pair `(sigmas, step_index)` the denoiser receives is
    // `([sub_sigma], 0)` on EVERY step. The scalar sigma above carries the first
    // half of that; this carries the second, and it is not cosmetic: the
    // denoiser reads `step_index` through `should_skip_step`
    // (guiders.py:287-291), so `0 % (skip_step + 1) == 0` makes the substep
    // evaluation unskippable at any `skip_step`. Passing the loop counter here
    // would skip it on the same steps the first evaluation is skipped on, which
    // is a first-order trajectory wearing the second-order sampler's schedule.
    // Inert on the HQ preset itself, whose `skip_step` is 0 (constants.py:104,
    // :112), and live for a request that overrides it.
    hooks.denoise(mid_v, mid_a, sub_sigma, /*step_index=*/0, denoised_v2, denoised_a2);
    stats.evaluations += 1;
    stats.eval_sigmas.push_back(sub_sigma);
    stats.eval_step_indices.push_back(0);
    if (video.present && !denoised_v2.empty()) {
      denoised_v2 = ToModelDtype(hooks.post_process(ToHp(denoised_v2), true));
    }
    if (audio.present && !denoised_a2.empty()) {
      denoised_a2 = ToModelDtype(hooks.post_process(ToHp(denoised_a2), false));
    }

    // ── the final combination (samplers.py:394-407) ─────────────────────────
    //
    // `x_anchor + h * (b1 * eps_1 + b2 * eps_2)`, in that association: the two
    // weighted epsilons are summed and the sum is scaled by h, which is not the
    // same rounding as scaling each term.
    std::vector<double> x_next_v, x_next_a;
    const auto combine = [&](bool present, const std::vector<double>& anchor,
                             const std::vector<double>& eps1, const std::vector<float>& d2,
                             std::vector<double>& out) {
      if (!present || anchor.empty() || eps1.empty() || d2.empty()) {
        out.clear();
        return;
      }
      out.resize(anchor.size());
      for (size_t k = 0; k < anchor.size(); ++k) {
        const double eps_2 = static_cast<double>(d2[k]) - anchor[k];
        out[k] = anchor[k] + h * (coeff.b1 * eps1[k] + coeff.b2 * eps_2);
      }
    };
    combine(video.present, x_anchor_v, eps_1_v, denoised_v2, x_next_v);
    combine(audio.present, x_anchor_a, eps_1_a, denoised_a2, x_next_a);

    // ── SDE noise injection at the step level (samplers.py:409-427) ─────────
    //
    // The loop's OWN eta, and the loop's OWN float32 schedule at `step_idx`.
    // Both differ from the substep call above, and both differences are
    // upstream's.
    inject(video.present, x_next_v, true, x_anchor_v, sigmas_hp.data(),
           static_cast<int64_t>(sigmas_hp.size()), step_idx, params.eta,
           Ltx2Res2sScheduleWidth::kF32Schedule, false);
    inject(audio.present, x_next_a, false, x_anchor_a, sigmas_hp.data(),
           static_cast<int64_t>(sigmas_hp.size()), step_idx, params.eta,
           Ltx2Res2sScheduleWidth::kF32Schedule, false);

    // samplers.py:429-433.
    if (video.present && !x_next_v.empty()) video.latent = ToModelDtype(x_next_v);
    if (audio.present && !x_next_a.empty()) audio.latent = ToModelDtype(x_next_a);
  }

  // ── the final step (samplers.py:435-445) ──────────────────────────────────
  //
  // "Final step if we need to fully remove the noise." It runs at index
  // `n_full_steps`, which after the injection above is the 0.0011 entry, and its
  // prediction becomes the state OUTRIGHT — there is no stepper call, so nothing
  // re-noises the finished latent. This is the `+ 1` in `2 * n_full_steps + 1`.
  if (terminal_zero) {
    denoised_v.clear();
    denoised_a.clear();
    // :437 — `n_full_steps`, which is one past the last full step's index and is
    // the position the injected 0.0011 now occupies.
    hooks.denoise(video.latent, audio.latent,
                  static_cast<double>(sigmas[static_cast<size_t>(stats.full_steps)]),
                  stats.full_steps, denoised_v, denoised_a);
    stats.evaluations += 1;
    stats.eval_sigmas.push_back(static_cast<double>(sigmas[static_cast<size_t>(stats.full_steps)]));
    stats.eval_step_indices.push_back(stats.full_steps);
    if (video.present && !denoised_v.empty()) {
      video.latent = ToModelDtype(hooks.post_process(ToHp(denoised_v), true));
    }
    if (audio.present && !denoised_a.empty()) {
      audio.latent = ToModelDtype(hooks.post_process(ToHp(denoised_a), false));
    }
  }

  return stats;
}

}  // namespace vllm
