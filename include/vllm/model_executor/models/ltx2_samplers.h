// LTX-2.5 SAMPLERS — the res_2s second-order denoising loop.
//
// Row: LTX25-RES2S-LOOP. Spec: .agents/specs/ltx25-res2s-loop.md. Issue #921.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream: Lightricks/LTX-2 @ fd4ded7f,
//           packages/ltx-pipelines/src/ltx_pipelines/
//   OURS                            <-  UPSTREAM
//   Ltx2Phi                         <-  utils/res2s.py:4-22
//   Ltx2Res2sCoefficients           <-  utils/res2s.py:25-62
//   Ltx2Res2sNormalizeNoise         <-  utils/samplers.py:160-170
//   Ltx2Res2sDenoisingLoop          <-  utils/samplers.py:208-447
//
// ─── WHY THIS IS A SEPARATE TRANSLATION UNIT ─────────────────────────────────
// Upstream's own partition. A *stepper* advances one substep and lives in
// `ltx-core/components/diffusion_steps.py`, which this port mirrors in
// `ltx2_pipeline.{h,cpp}`. A *sampler* decides how many substeps there are, what
// is evaluated between them, and in what order, and lives in
// `ltx-pipelines/utils/samplers.py`. They are different packages upstream and
// they are different files here.
//
// ─── THE SAMPLER *IS* THE HQ VARIANT ─────────────────────────────────────────
// `TI2VidTwoStagesHQPipeline` differs from `TI2VidTwoStagesPipeline` in SEVERAL
// things, and this loop is two of them: `stepper=Res2sDiffusionStep()`
// (ti2vid_two_stages_hq.py:258) and `loop=res2s_audio_video_denoising_loop`
// passed to both stages (:292, :335). The others, measured by diffing the two
// files at `fd4ded7f` rather than asserted: `LTX_2_3_HQ_PARAMS`
// (utils/constants.py:95-115); stage 1 loads the distilled LoRA at
// `distilled_lora_strength_stage_1` where the plain pipeline loads none on that
// stage (:92-101 against ti2vid_two_stages.py:140); the stage-1 schedule is
// derived as `execute(latent=empty_latent, steps=...)` against the plain
// pipeline's `execute(steps=...)`, which `schedulers.py:32` makes a
// RESOLUTION-DEPENDENT shift rather than the 4096-token default; and
// `GuidedDenoiser` (:271-281) replaces `FactoryGuidedDenoiser`. This comment
// said "exactly three things" until 2026-08-17, and the count was wrong in a
// load-bearing way, because it was the argument for what this row had to port.
//
// So a build that served the HQ preset's 15 steps and 0.45 rescale on the Euler
// loop would render a plausible clip at HALF the denoiser calls the preset was
// tuned for, and there is no shape, frame count, sample rate or pixel that says
// so. The one observable that separates the two samplers is the number of
// denoiser evaluations, which is why `Ltx2Res2sLoopStats::evaluations` exists
// and why the suite asserts an exact number rather than a bound.
//
// AND THE COUNT OF EVALUATIONS CANNOT SEE THE OTHER HALF. Each evaluation on
// the HQ stage 1 is THREE transformer forwards, because `GuidedDenoiser` runs
// the conditional, unconditional and isolated-modality passes
// (denoisers.py:100-137) at cfg 3.0 and modality 3.0. An arm that ran this
// sampler around a bare unguided forward reports the same evaluation count this
// file gates. `Ltx2ConditioningTrace::dit_forwards` is the second counter, and
// it is what the engine's gate reads.
//
// ─── DTYPE, AND WHY IT IS NOT f32 HERE ───────────────────────────────────────
// This is the one LTX-2.5 path whose interior is DOUBLE, and that is upstream's
// own choice stated in upstream's own words: `hp = highest_precision_float(...)`
// with the comment "float64 on CUDA/CPU for ODE numerical stability"
// (samplers.py:261-262). Every anchor, epsilon, midpoint and combination below
// is `double`; the LATENT that enters and leaves is f32, which is this port's
// `model_dtype`, at the positions upstream writes `.to(model_dtype)` — :370,
// :375, :431, :433, :442 and :445.
//
// AND THE TWO `model_dtype`s ARE NOT THE SAME WIDTH. Upstream's loop declares
// `model_dtype: torch.dtype = torch.bfloat16` (samplers.py:221) and the HQ
// pipeline overrides nothing (`DiffusionStage.__call__` passes six keyword
// arguments, utils/blocks.py:566-573), so upstream stores this latent at bf16
// where this port stores it at f32 — twice the bytes on the largest buffer in
// the loop. That is a PORT-WIDE pre-existing choice, not this row's: every
// LTX-2.5 host path here is f32 (`ltx2.h`), and narrowing one loop's storage
// would put a bf16 tensor into an f32 pipeline. It is stated here because
// `AGENTS.md` "Inherit vLLM defaults" says a wider dtype is invisible to every
// correctness gate this project owns, so it has to be written down where the
// divergence lives rather than discovered later.
//
// The already-ported ANCESTRAL loop does the opposite and steps in float32
// (samplers.py:550 calls `.float()` on the SAMPLE; the denoised operand was
// already floated at :484, so only one `.float()` sits at the step call and a
// reader looking for two at :550-551 finds one). Two loops, two
// precisions, in one file. Neither is a widening choice made here.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace vllm {

// ---------------------------------------------------------------------------
// The exponential integrator (utils/res2s.py)
// ---------------------------------------------------------------------------

// `phi(j, neg_h)` (res2s.py:4-22).
//
//   phi_j(z) = (e^z - sum_{k<j} z^k / k!) / z^j,   with phi_j(0) = 1 / j!
//
// THE SMALL-z BRANCH IS A GUARD, NOT A SERIES EXPANSION, AND THE DIFFERENCE
// DECIDES WHETHER THIS PORT IS CORRECT. Upstream returns `1 / j!` only when
// `abs(neg_h) < 1e-10` (res2s.py:13-16); everywhere else it evaluates the
// quotient DIRECTLY, and just outside the guard that quotient cancels
// catastrophically. Upstream's own values, measured by running upstream's own
// `phi`:
//
//   phi(2, -1e-11) = 0.5                  (guarded)
//   phi(2, -1e-10) = 0.0                  (NOT guarded: e^z - (1 + z) is 0 in f64)
//   phi(2, -1e-9)  = 0.0
//   phi(2, -1e-8)  = 1.1102230246251563
//   phi(1, -1e-10) = 1.000000082740371
//
// So the numerically BETTER implementation — a Taylor series near zero — returns
// 0.5 where upstream returns 0.0 and 1.11, and is WRONG for this port's purpose.
// Mirroring means reproducing the cancellation and the exact 1e-10 threshold.
// `1e-10` is not `< 1e-10`, so `-1e-10` takes the formula branch; that boundary
// is observable and the suite pins it. Do not "fix" this function.
double Ltx2Phi(int64_t j, double neg_h);

// The `phi_cache` upstream threads through the loop (res2s.py:25 the parameter,
// :37-44 the lookup; `:29` is the docstring line for `h`, not the cache),
// keyed on `(j, neg_h)` exactly as upstream keys it (res2s.py:39). It changes no
// value — every hit returns what a recompute would — and it is mirrored because
// the loop OWNS one across iterations (samplers.py:287) and a reader comparing
// the two files should find the same object in the same place.
using Ltx2PhiCache = std::map<std::pair<int64_t, double>, double>;

// `get_res2s_coefficients` (res2s.py:25-62). `c2` is the substep position and is
// 0.5 on every reachable path (samplers.py:288).
struct Ltx2Res2sCoefficients {
  double a21 = 0.0;  // c2 * phi_1(-h * c2)      (res2s.py:48-50)
  double b1 = 0.0;   // phi_1(-h) - b2           (res2s.py:59-60)
  double b2 = 0.0;   // phi_2(-h) / c2           (res2s.py:54-56)
};
Ltx2Res2sCoefficients Ltx2GetRes2sCoefficients(double h, Ltx2PhiCache& phi_cache,
                                               double c2 = 0.5);

// ---------------------------------------------------------------------------
// The noise (utils/samplers.py:155-170)
// ---------------------------------------------------------------------------

// The normalization half of `_get_new_noise` (samplers.py:164-170): a global
// `(n - mean) / std`, then `_channelwise_normalize` (:160-161), which on this
// port's rank-2 [tokens, width] latent covers the same elements and is therefore
// the identity up to rounding. BOTH ARE APPLIED ANYWAY, in upstream's order,
// because "idempotent" is a property of the data this port happens to hand it
// and not of the function; a batched latent would make the second one real.
//
// THE DRAW ITSELF IS NOT HERE, and that is the honest boundary. Upstream draws
// `torch.randn` on a seeded `torch.Generator`; this port has `SplitMixGaussian`.
// The streams differ, so a res_2s render is not bit-comparable with upstream —
// exactly as the already-shipped ancestral arm is not. What IS mirrored is which
// noise function each loop uses, and that is not the same for the two:
// `euler_ancestral_denoising_loop` defaults to `_get_plain_noise`, a bare
// `randn` (samplers.py:574), and the res_2s loop defaults to `_get_new_noise`,
// which normalizes (samplers.py:220). Two loops, two noise functions, ten lines
// apart. Reading one off the other would drop this step silently.
//
// `std` is UNBIASED (torch's default, n-1 denominator), matching `Tensor.std()`.
std::vector<double> Ltx2Res2sNormalizeNoise(std::vector<double> noise);

// The two seeds upstream's loop declares (samplers.py:215-216, :265-266).
//
// `-1` IS A CONSTANT, NOT THE REQUEST'S SEED, and that is the fact most likely
// to be got wrong by analogy. `DiffusionStage.__call__` passes the loop SIX
// keyword arguments — sigmas, video_state, audio_state, stepper, transformer,
// denoiser — and no others (utils/blocks.py:566-573), so `noise_seed` keeps its
// declared default on every reachable path. The already-ported ancestral arm
// does the opposite and derives its seed from the pipeline's
// (distilled.py:69-73), which is why this is stated rather than assumed.
inline constexpr int64_t kLtx2Res2sNoiseSeed = -1;
inline constexpr int64_t kLtx2Res2sNoiseSeedSubstepOffset = 10000;

// ---------------------------------------------------------------------------
// The loop (utils/samplers.py:208-447)
// ---------------------------------------------------------------------------

// `_inject_sde_noise`'s substep call fixes eta at 0.5 "for compatibility with
// the original implementation" (samplers.py:273-274) regardless of the step-level
// eta. Step level takes the loop's `eta`, which is 0.5 by default (:217).
inline constexpr double kLtx2Res2sSubstepEta = 0.5;
inline constexpr double kLtx2Res2sEta = 0.5;
// samplers.py:218-219.
inline constexpr bool kLtx2Res2sBongMath = true;
inline constexpr int64_t kLtx2Res2sBongMathMaxIter = 100;
// samplers.py:288 — "Midpoint for res_2s".
inline constexpr double kLtx2Res2sC2 = 0.5;
// samplers.py:357 — the bong guard, `h < 0.5 and sigma > 0.03`. STRICT on both
// sides: a schedule sitting at exactly 0.03 does NOT refine.
inline constexpr double kLtx2Res2sBongMaxH = 0.5;
inline constexpr double kLtx2Res2sBongMinSigma = 0.03;
// samplers.py:281-282 — the minimal sigma injected in place of a terminal zero,
// "to avoid division by zero". It becomes a real schedule entry, so the loop's
// last full step lands on it and the final evaluation happens AT it.
inline constexpr float kLtx2Res2sTerminalSigma = 0.0011f;

// What the loop needs from its caller. Upstream's loop takes a `transformer` and
// a `Denoiser` callable (samplers.py:213-214) rather than reaching for a model,
// and mirroring that shape is also what makes the evaluation count gateable: a
// test supplies a counting denoiser and asserts an exact number.
struct Ltx2Res2sHooks {
  // `denoiser(transformer, video_state, audio_state, sigmas, step_index)`
  // (samplers.py:301, :380-386). Writes each modality's DENOISED prediction —
  // upstream's `X0Model` returns x0, not velocity (ltx-core
  // model/transformer/model.py:590-604 is the forward that converts;
  // utils/blocks.py:480-482 only shows that the loop is handed that TYPE, and
  // the `utils/` prefix matters because a second `blocks.py` exists under
  // ltx-core model/video_vae/transformer/) — at the
  // model dtype, which here is f32.
  //
  // A SCALAR SIGMA, not a schedule and an index into it, because all three
  // upstream call sites reduce to `sigmas[step_index]` inside the denoiser
  // (utils/denoisers.py:237) and the substep one already passes a ONE-element
  // schedule with index 0 (samplers.py:384-385). Handing a pair to this hook
  // would invite a caller to index it differently from upstream.
  //
  // `step_index` IS STILL PASSED, because it is a SECOND argument upstream's
  // `Denoiser` takes and the denoiser reads it for something other than the
  // sigma: `should_skip_step` is `step % (skip_step + 1) != 0`
  // (guiders.py:287-291). The three call sites pass three different things —
  // `step_idx` (samplers.py:301), a literal `0` (samplers.py:385) and
  // `n_full_steps` (samplers.py:437) — so the substep evaluation is never
  // skipped whatever the request's `skip_step` is. Deriving it here from the
  // loop counter instead would silently skip half of a step's evaluations on a
  // request that sets `skip_step`, and no rendered frame would show it.
  //
  // `double`, AND THE NARROWING BELONGS TO THE CALLER. The two evaluations are
  // handed different widths upstream: the first gets an entry of the float32
  // schedule (samplers.py:301) and the second gets `sub_sigma`, which is
  // float64 (`torch.stack([sub_sigma])`, samplers.py:384). This port's DiT
  // interface takes `const float*` for `Modality.sigma`, so a narrowing has to
  // happen somewhere; it happens at that interface, in the engine, and not here,
  // so the loop stays the shape upstream's is.
  std::function<void(const std::vector<float>& video_latent,
                     const std::vector<float>& audio_latent, double sigma, int64_t step_index,
                     std::vector<float>& denoised_video,
                     std::vector<float>& denoised_audio)>
      denoise;

  // `post_process_latent(x, denoise_mask, clean)` (utils/helpers.py:461-463),
  // per modality. Kept as a hook rather than taking the mask and the clean
  // latent as arguments because the engine already owns both inside its own
  // stream struct, and a second copy of the blend is the shape this project has
  // recorded going wrong.
  //
  // ONE `double` HOOK FOR BOTH OF UPSTREAM'S WIDTHS, and the reason is a
  // property of the data rather than of the function. Upstream calls
  // `post_process_latent` at the model dtype on a denoiser result
  // (samplers.py:305, :390, :441) and at `hp` on a sample inside
  // `_inject_sde_noise` (samplers.py:203). The blend is
  // `denoised * mask + clean * (1 - mask)`, and every LTX-2.5 denoise mask is
  // 0 or 1 — `create_initial_state` writes ones and a conditioning zeroes whole
  // token rows — so the result is exactly one operand or the other and no
  // rounding is reachable at either width. The loop still narrows the
  // model-dtype call sites back to f32 afterwards, mirroring
  // `.to(denoised.dtype)`, so a mask that ever stopped being 0/1 would show as a
  // difference rather than silently taking the wider path.
  std::function<std::vector<double>(std::vector<double> x, bool is_video)> post_process;

  // `new_noise_fn(state.latent, generator)` (samplers.py:220, :187). `substep`
  // selects between upstream's TWO generators (samplers.py:267-268), which are
  // seeded `noise_seed` and `noise_seed + 10000` so the substep draw is not
  // bit-identical to the step draw.
  std::function<std::vector<double>(int64_t count, bool is_video, bool substep)> new_noise;
};

// Both modalities' state, in and out. Upstream carries a `LatentState` per
// modality and allows either to be absent (samplers.py:231); `present` is that
// `None`.
struct Ltx2Res2sModality {
  std::vector<float> latent;  // model_dtype (f32 here)
  bool present = false;
};

// Reported so the caller can assert what happened, because nothing in the
// returned latents can. `evaluations` is the discriminator this row rests on.
struct Ltx2Res2sLoopStats {
  // Total denoiser calls. `2 * full_steps + 1` when the caller's schedule ends
  // at 0, `2 * full_steps` when it does not.
  int64_t evaluations = 0;
  // `n_full_steps` (samplers.py:279), taken BEFORE the terminal sigma injection.
  int64_t full_steps = 0;
  // Steps on which `bongmath and h < 0.5 and sigma > 0.03` held (samplers.py:357).
  int64_t bong_steps = 0;
  // The sigma each evaluation ran at, in call order. Every odd entry is
  // `sqrt(sigma * sigma_next)` (samplers.py:315), so a build that evaluated
  // twice at the SAME sigma is visible here and nowhere else.
  std::vector<double> eval_sigmas;
  // The `step_index` each evaluation was handed, in the same call order. It is
  // NOT the position in this vector and it is not the loop counter: upstream
  // passes `step_idx`, then a literal `0` for the substep, then `n_full_steps`
  // for the terminal evaluation (samplers.py:301, :385, :437). The denoiser
  // reads it through `should_skip_step` (guiders.py:287-291), so on a request
  // with `skip_step != 0` the sequence decides which evaluations run a forward
  // at all — and nothing in the returned latents, the evaluation count or a
  // rendered frame records which value was passed.
  std::vector<int64_t> eval_step_indices;
};

// Loop parameters, in upstream's own declaration order and with upstream's own
// defaults (samplers.py:208-223). They are defaults HERE for the same reason
// they are defaults THERE: `DiffusionStage.__call__` overrides none of them.
struct Ltx2Res2sLoopParams {
  double eta = kLtx2Res2sEta;
  bool bongmath = kLtx2Res2sBongMath;
  int64_t bongmath_max_iter = kLtx2Res2sBongMathMaxIter;
  double c2 = kLtx2Res2sC2;
};

// `res2s_audio_video_denoising_loop` (samplers.py:208-447). `legacy_mode` is
// TRUE on every reachable path (samplers.py:222 default, never overridden), so
// `_inject_sde_noise` hands the stepper the raw schedule and applies
// `post_process_latent` afterwards (:202-203) instead of converting sigmas
// through `timesteps_from_mask` (:188-192). The false arm is not built; nothing
// upstream selects it and a selection surface for it would be invented here.
Ltx2Res2sLoopStats Ltx2Res2sDenoisingLoop(const std::vector<float>& sigmas,
                                          Ltx2Res2sModality& video,
                                          Ltx2Res2sModality& audio,
                                          const Ltx2Res2sHooks& hooks,
                                          const Ltx2Res2sLoopParams& params = {});

}  // namespace vllm
