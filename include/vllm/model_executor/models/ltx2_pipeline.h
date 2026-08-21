// LTX-2.5 PIPELINE — the flow-matching schedule, the noiser, the diffusion
// steps, guidance, the patchifiers, and the recipe table.
//
// Row: MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model. Spec:
// .agents/specs/ltx-2-5.md (phase L5). Issue #435.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream A: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/
//   OURS                            <-  UPSTREAM
//   Ltx2SigmaSchedule               <-  components/schedulers.py:21-57  (LTX2Scheduler)
//   Ltx2LinearQuadraticSchedule     <-  components/schedulers.py:67-88
//   Ltx2GaussianNoise               <-  components/noisers.py:30-37
//   Ltx2EulerStep                   <-  components/diffusion_steps.py:32-40
//   Ltx2EulerAncestralStep          <-  components/diffusion_steps.py:63-106
//   Ltx2Res2sStep / Ltx2SdeCoeff    <-  components/diffusion_steps.py:118-190
//   Ltx2EulerCfgPpStep              <-  components/diffusion_steps.py:208-252
//   Ltx2AncestralStep               <-  components/diffusion_steps.py:7-22
//   Ltx2ProjectionCoef              <-  components/guiders.py:363-369
//   Ltx2CfgDelta / Ltx2StgDelta     <-  components/guiders.py:23-27, 70-74
//   Ltx2MultiModalGuidance          <-  components/guiders.py:244-291
//   Ltx2GuiderParamsForSigma        <-  components/guiders.py:214-230, 332-335
//   Ltx2BatchedPerturbationConfig   <-  guidance/perturbations.py:53-143
//   Ltx2VideoPatchify / …           <-  components/patchifiers.py:11-134
//   Ltx2PixelCoords                 <-  components/patchifiers.py:137-171
//   Ltx2AudioPatchify / …           <-  components/patchifiers.py:174-353
//
// Upstream B: Lightricks/LTX-2, packages/ltx-pipelines/src/ltx_pipelines/
//   Ltx2ParseModelVersion           <-  (ltx-core) loader/helpers.py:62-81
//   Ltx2DetectPipelineParams        <-  utils/constants.py:130-179
//   Ltx2ShouldUseAncestralSampler   <-  distilled.py:62-84
//   the distilled sigma constants   <-  utils/constants.py:17-25
//
// Upstream C: vLLM-Omni, vllm_omni/diffusion/models/ltx2/ltx2_recipes.py
//   Ltx2PipelineRecipe / …Phase…    <-  ltx2_recipes.py:29-87
//   ResolveLtx2PipelineRecipe       <-  ltx2_recipes.py:161-175
//
// ─── WHICH UPSTREAM OWNS WHICH VALUE ─────────────────────────────────────────
// vLLM-Omni is this project's BINDING oracle (spec section 3) and it carries NO
// 2.5 row — its table stops at 2.3. So the SHAPE of the recipe model and the
// values of every pre-2.5 row come from vLLM-Omni, and the values of the 2.4 and
// 2.5 rows come from Lightricks `ltx-pipelines`, which is the model author's own
// runtime. Each row below names its source. Where the two references DISAGREE the
// disagreement is recorded rather than resolved by preference: they ship
// different default negative prompts, and both strings are exposed
// (kLtx2LightricksNegativePrompt vs kLtx2OmniNegativePrompt in the goldens).
//
// ─── THE REFUSAL IS THE POINT ────────────────────────────────────────────────
// `resolve_ltx_pipeline_recipe` RAISES on an unknown (kind, version) rather than
// defaulting (ltx2_recipes.py:170-175), and this port mirrors that exactly. A
// sigma schedule or a guidance scale that is plausible but wrong does not fail —
// it renders. Defaulting an unknown checkpoint generation onto 2.0's 40-step,
// STG-block-29 recipe would produce a video, and nothing downstream could tell.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// Every buffer here is f32, and that is upstream's OWN width on these paths, not
// a widening: the noiser lerps in `.float()` (noisers.py:32-33), every diffusion
// step casts to `torch.float32` and back to the sample dtype
// (diffusion_steps.py:40, 83-84, 90-91, 231-240), and MultiModalGuider.calculate
// opens with `cond.float()` (guiders.py:256-260). The schedules are float32
// tensors upstream too (schedulers.py:57, 88). Sigma SHIFT arithmetic is the one
// double: `mm`, `b` and `math.exp(sigma_shift)` are Python floats before torch
// sees them (schedulers.py:37-44), so they are computed in double here and
// rounded to f32 at exactly the point torch rounds them.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

// ---------------------------------------------------------------------------
// Sigma schedules (components/schedulers.py)
// ---------------------------------------------------------------------------

// schedulers.py:10-11. The token axis the shift is fitted on. These are module
// constants, not arguments: they decide how a resolution maps to a shift, so a
// port that moved either would produce a valid-looking schedule for the wrong
// resolution.
inline constexpr int64_t kLtx2BaseShiftAnchor = 1024;
inline constexpr int64_t kLtx2MaxShiftAnchor = 4096;

// schedulers.py:41. The exponent applied to `(1/sigma - 1)`. A literal upstream,
// pinned here because `power != 1` would change every sigma while still
// producing a monotone schedule that renders.
inline constexpr double kLtx2SigmaShiftPower = 1.0;

// LTX2Scheduler.execute's keyword defaults (schedulers.py:21-31).
struct Ltx2SchedulerParams {
  double max_shift = 2.05;
  double base_shift = 0.95;
  bool stretch = true;
  double terminal = 0.1;
  // `default_number_of_tokens` — used when no latent is supplied.
  int64_t default_number_of_tokens = kLtx2MaxShiftAnchor;
};

// LTX2Scheduler.execute (schedulers.py:21-57). Returns `steps + 1` sigmas, from
// ~1 down to exactly 0. `tokens` is `math.prod(latent.shape[2:])` — pass 0 to
// take `params.default_number_of_tokens`, which is what a caller with no latent
// yet does. Throws when `steps < 1`.
std::vector<float> Ltx2SigmaSchedule(int64_t steps, int64_t tokens,
                                     const Ltx2SchedulerParams& params = {});

// LinearQuadraticScheduler.execute (schedulers.py:67-88). `linear_steps < 0`
// takes upstream's `steps // 2` default.
std::vector<float> Ltx2LinearQuadraticSchedule(int64_t steps, double threshold_noise = 0.025,
                                               int64_t linear_steps = -1);

// Which scheduler a caller is asking for. `kBeta` is upstream's third
// (schedulers.py:91-120) and is NOT ported: it inverts a Beta CDF through
// `scipy.stats.beta.ppf`, no ltx-pipelines entry point constructs it, and
// approximating an inverse incomplete beta would be a numerical port of its own.
// `Ltx2Schedule` REFUSES it by name rather than substituting LTX2Scheduler.
enum class Ltx2SchedulerKind { kLtx2, kLinearQuadratic, kBeta };

// The seam a caller would reach for if it held a configured kind. NOTHING IN
// `src/`, `include/` OR `examples/` CALLS THIS, and that is upstream's shape, not
// an omission: no ltx-pipelines entry point selects a scheduler either, so there
// is no request field to carry a kind and the engine calls `Ltx2SigmaSchedule`
// directly, in `ltx2_video.cpp`'s phase driver. Say "no caller" rather than "the
// seam a caller reaches for": that wording is what published `kBetaScheduler` as
// a reachable refusal (#889). No line number: that file moves on every merge, and
// this row has already shipped three anchors that went stale inside one branch.
//
// UNREACHED, AND DELIBERATELY SO UNTIL #893 DECIDES OTHERWISE. Under AGENTS.md
// `## Nothing lands dead` this is the "test-only driver" shape, and the rule asks
// for the unreached thing, its owning row and its issue to be named rather than
// left for the next reader to discover. Owning row LTX25-RETIRE-DEAD-ARMS, which
// lists it under `## Owed`. There is no wiring wave coming — upstream has no
// scheduler selection to mirror — so the open question is retire-or-keep, not
// when to wire it.
//
// Forwards to the two ported schedulers and throws for `kBeta`.
std::vector<float> Ltx2Schedule(Ltx2SchedulerKind kind, int64_t steps, int64_t tokens,
                                const Ltx2SchedulerParams& params = {});

// ---------------------------------------------------------------------------
// The noiser (components/noisers.py)
// ---------------------------------------------------------------------------

// GaussianNoiser.__call__ (noisers.py:30-37), elementwise over `count`:
//   latent = lerp(latent, noise, noise_scale)
//   latent = lerp(clean_latent, latent, denoise_mask)
// The SECOND lerp is the one a port gets backwards: `denoise_mask` selects
// toward the NOISED latent, so a conditioned token (mask 0) keeps its clean value
// and an unconditioned one (mask 1) is fully noised. Swapping the operands still
// produces a plausible video with the conditioning frames re-noised away.
//
// `noise` is drawn by the caller (upstream uses `torch.randn(generator=...)`,
// noisers.py:22-28); this port takes the draw as an argument so the stream stays
// the caller's, exactly as Ltx2NoiseStream does for the VAE.
std::vector<float> Ltx2GaussianNoise(const float* latent, const float* clean_latent,
                                     const float* denoise_mask, const float* noise,
                                     int64_t count, float noise_scale = 1.0f);

// ---------------------------------------------------------------------------
// Diffusion steps (components/diffusion_steps.py)
// ---------------------------------------------------------------------------

// EulerDiffusionStep.step (diffusion_steps.py:32-40): `x + to_velocity(...) * dt`.
// `to_velocity` (utils.py:21-36) REFUSES sigma == 0 rather than dividing, so a
// terminal step_index throws here too.
std::vector<float> Ltx2EulerStep(const float* sample, const float* denoised,
                                 const float* sigmas, int64_t sigma_count, int64_t step_index,
                                 int64_t count);

// EulerAncestralDiffusionStep.step (diffusion_steps.py:63-106). The
// rectified-flow parameterization (alpha = 1 - sigma), which is LTX-2's; it is
// deliberately NOT the DDIM one `Ltx2AncestralStep` implements, and the two agree
// only at eta == 0 (the class docstring says so at :51-56).
//
// `noise` may be null only when `eta == 0`; anything else throws, mirroring
// upstream's own ValueError (:87-88). When `sigmas[step_index + 1] == 0` the
// denoised prediction is returned unchanged (:85-86).
std::vector<float> Ltx2EulerAncestralStep(const float* sample, const float* denoised,
                                          const float* sigmas, int64_t sigma_count,
                                          int64_t step_index, int64_t count, double eta = 1.0,
                                          double s_noise = 1.0, const float* noise = nullptr);

// Res2sDiffusionStep.get_sde_coeff (diffusion_steps.py:118-155), the `sigma_up`
// arm — the only one `step` uses (:179). `sigma_up` is clamped IN to
// `sigma_next * kLtx2Res2sSigmaUpClamp` before anything else.
struct Ltx2SdeCoeff {
  double alpha_ratio = 1.0;
  double sigma_down = 0.0;
  double sigma_up = 0.0;
};

// diffusion_steps.py:138. What keeps `sqrt(sigma_next^2 - sigma_up^2)` off zero,
// and it BINDS on the ordinary eta = 1 schedule rather than only on a malformed
// one. `step` forms `sigma_up = sigma_next * eta`, so eta <= 1 gives
// sigma_up <= sigma_next — but <= includes ==, and at equality `min` takes
// `sigma_next * 0.9999`, which is the whole point: without it the residual is
// exactly 0 and `sigma_down` collapses. The earlier note reasoned from the
// inequality and skipped its boundary, calling the constant invisible; it is not.
// A 1% move (0.9999 -> 0.99) REDS the Eta1 arm the suite already runs, at
// max|diff| = 0.086 (index 0) and 0.130563 (index 1), because the residual scales
// as sqrt(1 - clamp^2) and that is 10x larger at 0.99. The EtaHalf arm stays green
// (0.5 * sigma_next is below the clamp) and Eta1 index 2 stays green
// (sigma_next == 0 returns the denoised prediction unchanged, :181-182). Pinned
// as well, because a regenerated golden would move with the constant.
inline constexpr double kLtx2Res2sSigmaUpClamp = 0.9999;

Ltx2SdeCoeff Ltx2Res2sSdeCoeff(double sigma_next, double sigma_up);

// Res2sDiffusionStep.step (diffusion_steps.py:157-190). Returns the denoised
// prediction unchanged when `sigma_up` or `sigma_next` is 0 (:181-182).
std::vector<float> Ltx2Res2sStep(const float* sample, const float* denoised,
                                 const float* sigmas, int64_t sigma_count, int64_t step_index,
                                 int64_t count, const float* noise, double eta = 0.5);

// ─── THE SAME STEP, AT THE PRECISION EACH CALL SITE ACTUALLY HANDS IT ────────
//
// `Res2sDiffusionStep.step` has no dtype of its own: it takes whatever its
// tensors carry, and the res_2s loop hands it two DIFFERENT combinations. Both
// are mirrored rather than unified onto one, because the difference is real
// arithmetic and putting the conversion where upstream puts it is the rule.
//
//   SUBSTEP (samplers.py:337-352). `sigmas = torch.stack([sigma, sub_sigma])`,
//   and both are `hp` (:291-292, :315). So `get_sde_coeff` runs in FLOAT64.
//
//   STEP (samplers.py:412-427). `sigmas` is the loop's own schedule, which
//   `DiffusionStage` created as FLOAT32 (ti2vid_two_stages_hq.py:268). So
//   `get_sde_coeff` runs in FLOAT32 — the residual `sqrt(sigma_next^2 -
//   sigma_up^2)`, `alpha_ratio` and `sigma_down` are all f32 quantities — while
//   the SAMPLE and the noise are still f64 and the result is f64.
//
// The values in both cases are f64, because `sample` is `x_anchor` (`hp`) and
// `output_dtype = denoised_sample.dtype` is `hp` too (diffusion_steps.py:180).
//
// One implementation, instantiated at the two scalar types; there is no second
// copy of the formula. The selection is an enum naming the two upstream call
// sites rather than a bare bool, so a reader can check the claim.
enum class Ltx2Res2sScheduleWidth {
  // samplers.py:415, :425 — the loop's float32 schedule.
  kF32Schedule,
  // samplers.py:342, :350 — the [sigma, sub_sigma] pair, both float64.
  kF64Schedule,
};

// `Res2sDiffusionStep.get_sde_coeff` computed in float64 rather than float32.
// The f32 arm stays `Ltx2Res2sSdeCoeff` above and keeps its goldens.
Ltx2SdeCoeff Ltx2Res2sSdeCoeffHp(double sigma_next, double sigma_up);

// `Res2sDiffusionStep.step` over float64 samples. `width` decides only the
// precision the SIGMAS and therefore the coefficients are computed at.
std::vector<double> Ltx2Res2sStepHp(const double* sample, const double* denoised,
                                    const double* sigmas, int64_t sigma_count,
                                    int64_t step_index, int64_t count, const double* noise,
                                    double eta, Ltx2Res2sScheduleWidth width);

// _get_ancestral_step (diffusion_steps.py:7-22): the DDIM / variance-exploding
// ancestral coefficients, in the rescaled `sigma / alpha` space. Used only by
// CFG++.
struct Ltx2AncestralSigmas {
  double sigma_down = 0.0;
  double sigma_up = 0.0;
};
Ltx2AncestralSigmas Ltx2AncestralStep(double sigma_from, double sigma_to, double eta = 1.0);

// diffusion_steps.py:233-235. `torch.finfo(torch.float32).eps`, the clamp that
// keeps `alpha = 1 - sigma` off zero when sigma is EXACTLY 1.0 — which the first
// step of every unstretched schedule is. Unlike most of the invisible-constant
// class this one DOES bind on a real schedule, and the goldens include the arm
// where it decides the numbers.
inline constexpr double kLtx2CfgPpAlphaEps = 1.1920928955078125e-07;

// EulerCfgPpDiffusionStep.step (diffusion_steps.py:208-252).
std::vector<float> Ltx2EulerCfgPpStep(const float* sample, const float* denoised,
                                      const float* uncond_denoised, const float* sigmas,
                                      int64_t sigma_count, int64_t step_index, int64_t count,
                                      double eta = 1.0, double s_noise = 1.0,
                                      const float* noise = nullptr);

// ---------------------------------------------------------------------------
// Guidance (components/guiders.py)
// ---------------------------------------------------------------------------

// guiders.py:368. The stabilizer under `squared_norm`. It is ~1e-8 relative
// against an O(1) denominator, so a tensor comparison cannot see it — except on
// the one arm where `project_onto` is all zeros, which the goldens carry.
inline constexpr double kLtx2ProjectionCoefEps = 1e-8;

// projection_coef (guiders.py:363-369): per BATCH ROW,
// `dot(to_project, project_onto) / (||project_onto||^2 + eps)`. `count` is the
// number of elements per row, i.e. the product of every axis after the batch.
std::vector<float> Ltx2ProjectionCoef(const float* to_project, const float* project_onto,
                                      int64_t batch, int64_t count);

// CFGGuider.delta (guiders.py:23-24) and STGGuider.delta (:70-71). Both are
// elementwise and rank-agnostic, which is why they are the two that survive
// contact with a real 5-D latent — see the refusal below.
std::vector<float> Ltx2CfgDelta(const float* cond, const float* uncond, int64_t count,
                                double scale);
std::vector<float> Ltx2StgDelta(const float* cond, const float* perturbed, int64_t count,
                                double scale);

// NOT PORTED, refused by name: CFGStarRescalingGuider (guiders.py:30-52),
// LtxAPGGuider (:77-125) and LegacyStatefulAPGGuider (:128-191).
//
// THE REASON IS REACHABILITY. Nothing in the LTX-2 tree constructs any of the
// three: they appear only at their own `class` statements (:31, :78, :129), and
// every pipeline builds MultiModalGuider from MultiModalGuiderParams
// (utils/constants.py:49-68). Porting an arm upstream cannot reach would be
// inventing behaviour, so `Ltx2Guidance` refuses them by name and they are
// recorded as owed (.agents/porting-inventory.md 9.18(b)).
//
// IT IS *NOT* BECAUSE THE SHAPES CANNOT WORK — an earlier revision of this
// comment said so and was wrong, and the correction is kept here because the
// wrong version had been frozen as a golden (spec §7.0(b)). All three do
// multiply `projection_coef`'s rank-2 `(B, 1)` result straight into the latent
// (`proj_coeff * cond`, :48, :118, :184) and torch does right-align it onto the
// LAST TWO axes, but the measured predicate is
//
//     raises  <=>  B > 1 and shape[-2] not in {1, B}
//
// so at B = 1 — the ordinary single-request video latent — it composes and is
// numerically CORRECT, `(1, 1)` being a scalar. Where it composes with B > 1 it
// is silently wrong, applying the per-batch coefficient along axis -2 rather
// than the batch axis. LtxAPG's `norm(dim=[-1, -2, -3])` (:114) is a SEPARATE
// constraint needing rank >= 3. The measured matrix is a golden
// (kLtx2GuideProbeComposes), now including B = 1 and shape[-2] == B rows, so
// upstream changing either the shapes or the reachability fails this gate
// instead of going unnoticed.
enum class Ltx2GuiderKind {
  kCfg,
  kStg,
  kMultiModal,
  kCfgStarRescaling,   // refused
  kLtxApg,             // refused
  kLegacyStatefulApg,  // refused
};

// MultiModalGuiderParams (guiders.py:194-211) — the ONLY guider parameter object
// any ltx-pipelines entry point constructs.
struct Ltx2MultiModalGuiderParams {
  double cfg_scale = 1.0;
  double stg_scale = 0.0;
  std::vector<int64_t> stg_blocks;
  double rescale_scale = 0.0;
  double modality_scale = 1.0;
  int64_t skip_step = 0;

  // guiders.py:275-291. `math.isclose` with its DEFAULT rel_tol of 1e-9, not
  // `!=`: a scale of 1.0 + 1e-12 is "no guidance" here and "guidance" under an
  // exact comparison, and the difference decides whether an extra full DiT
  // forward runs per step.
  bool DoUnconditionalGeneration() const;
  bool DoPerturbedGeneration() const;
  bool DoIsolatedModalityGeneration() const;
  bool ShouldSkipStep(int64_t step) const;
};

// MultiModalGuider.calculate (guiders.py:244-273). `uncond_*` may be null, which
// mirrors upstream's `torch.Tensor | float` union at its only reachable scalar
// value: a null stands for the float 0.0 that a disabled arm passes.
//
// The rescale (`:268-271`) divides by `pred.std()`. torch's `std` is the UNBIASED
// (N-1) estimator by default, and using the biased one instead is a small,
// everywhere, resolution-dependent gain error that no shape or finiteness check
// can see.
std::vector<float> Ltx2MultiModalGuidance(const Ltx2MultiModalGuiderParams& params,
                                          const float* cond, const float* uncond_text,
                                          const float* uncond_perturbed,
                                          const float* uncond_modality, int64_t count);

// The seam a caller reaches for when it holds a configured kind: forwards to the
// three ported guiders and throws by name for the other three.
std::vector<float> Ltx2Guidance(Ltx2GuiderKind kind, const Ltx2MultiModalGuiderParams& params,
                                const float* cond, const float* uncond_text,
                                const float* uncond_perturbed, const float* uncond_modality,
                                int64_t count);

// One bin of the sigma-dependent factory (guiders.py:294-342). `sigma_upper_bound`
// is the bin's INCLUSIVE upper edge; bin i is `(key[i+1], key[i]]`.
struct Ltx2GuiderSigmaBin {
  double sigma_upper_bound = 0.0;
  Ltx2MultiModalGuiderParams params;
};

// _params_for_sigma_from_sorted_dict (guiders.py:214-230). `bins` need not be
// sorted — this sorts descending exactly as `from_dict` does (:329) — but it must
// be non-empty, which upstream also requires (:223-224, :327-328).
const Ltx2MultiModalGuiderParams& Ltx2GuiderParamsForSigma(
    const std::vector<Ltx2GuiderSigmaBin>& bins, double sigma);

// ---------------------------------------------------------------------------
// Perturbations (guidance/perturbations.py)
// ---------------------------------------------------------------------------

// perturbations.py:8-16. The VALUE is the row index into the mask tensor's dim 0,
// so these are not free to renumber.
enum class Ltx2PerturbationType {
  kSkipVideoSelfAttn = 0,
  kSkipAudioSelfAttn = 1,
  kSkipA2vCrossAttn = 2,
  kSkipV2aCrossAttn = 3,
};
inline constexpr int64_t kLtx2PerturbationTypeCount = 4;

// perturbations.py:19-33. `all_blocks` is upstream's `blocks is None`.
struct Ltx2Perturbation {
  Ltx2PerturbationType type = Ltx2PerturbationType::kSkipVideoSelfAttn;
  bool all_blocks = false;
  std::vector<int64_t> blocks;
};

// perturbations.py:36-50. An empty list is "nothing perturbed"; it is what
// `PerturbationConfig.empty()` builds and what LTX-2.5 actually runs.
struct Ltx2PerturbationConfig {
  std::vector<Ltx2Perturbation> perturbations;
  bool IsPerturbed(Ltx2PerturbationType type, int64_t block) const;
};

// BatchedPerturbationConfig (perturbations.py:53-143): the per-block KEEP mask,
// 1 = keep and 0 = perturbed, indexed [type, block, sample]. The no-perturbation
// configuration — all ones — is the shipped LTX-2.5 path (spec section 2 routes
// STG through the guider), so it is the default here rather than a corner case.
class Ltx2BatchedPerturbationConfig {
 public:
  Ltx2BatchedPerturbationConfig() = default;
  Ltx2BatchedPerturbationConfig(const std::vector<Ltx2PerturbationConfig>& configs,
                                int64_t num_blocks);
  // `BatchedPerturbationConfig.empty` (:134-143).
  static Ltx2BatchedPerturbationConfig Empty(int64_t batch, int64_t num_blocks);

  int64_t batch() const { return batch_; }
  int64_t num_blocks() const { return num_blocks_; }
  // The whole [type, block, sample] mask, row-major.
  const std::vector<int32_t>& block_masks() const { return masks_; }
  // `mask` (:118-124): this block's per-sample keep mask for one type.
  std::vector<int32_t> Mask(Ltx2PerturbationType type, int64_t block) const;
  bool AnyInBatch(Ltx2PerturbationType type, int64_t block) const;
  bool AllInBatch(Ltx2PerturbationType type, int64_t block) const;
  // `batch_slice` (:111-116): a view over samples [start, end), rebuilt by
  // slicing rather than by re-reading the config list.
  Ltx2BatchedPerturbationConfig BatchSlice(int64_t start, int64_t end) const;

 private:
  int64_t batch_ = 0;
  int64_t num_blocks_ = 0;
  std::vector<int32_t> masks_;
};

// ---------------------------------------------------------------------------
// Patchifiers (components/patchifiers.py)
// ---------------------------------------------------------------------------

// VideoLatentShape (types.py:73-98) and AudioLatentShape (:134-160).
struct Ltx2VideoLatentShape {
  int64_t batch = 1, channels = 0, frames = 0, height = 0, width = 0;
};
struct Ltx2AudioLatentShape {
  int64_t batch = 1, channels = 0, frames = 0, mel_bins = 0;
};

// SpatioTemporalScaleFactors.default() (types.py:31-33) — the Conv VAE's
// downsampling, which is what turns latent bounds into pixel timestamps.
struct Ltx2ScaleFactors {
  int64_t time = 8, height = 32, width = 32;
};

// VideoLatentPatchifier (patchifiers.py:11-134). The temporal patch size is
// always 1 (:15) — `unpatchify` asserts it (:46) — so `patch_size` is the
// SPATIAL one only.
int64_t Ltx2VideoTokenCount(const Ltx2VideoLatentShape& shape, int64_t patch_size);
// `b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)` (:31-38). Output is
// [batch, tokens, channels * patch_size^2].
std::vector<float> Ltx2VideoPatchify(const float* latent, const Ltx2VideoLatentShape& shape,
                                     int64_t patch_size);
// `b (f h w) (c p q) -> b c f (h p) (w q)` (:52-60).
std::vector<float> Ltx2VideoUnpatchify(const float* tokens, const Ltx2VideoLatentShape& shape,
                                       int64_t patch_size);
// get_patch_grid_bounds (:64-134): [batch, 3, tokens, 2] of [start, end) bounds
// in (frame, height, width) order.
std::vector<int64_t> Ltx2VideoPatchBounds(const Ltx2VideoLatentShape& shape, int64_t patch_size);
// get_pixel_coords (:137-171). `causal_fix` rewrites the TEMPORAL axis only, and
// clamps at 0 — the first latent frame covers one pixel frame, not `time`.
std::vector<int64_t> Ltx2PixelCoords(const std::vector<int64_t>& latent_coords, int64_t batch,
                                     int64_t tokens, const Ltx2ScaleFactors& factors,
                                     bool causal_fix);

// AudioPatchifier (patchifiers.py:174-353). The three rates below are its
// constructor defaults (:177-180) and they set the seconds-per-latent-frame the
// DiT's audio RoPE is indexed by, so a wrong one is a silently mistimed
// soundtrack rather than an error.
struct Ltx2AudioPatchifierParams {
  int64_t sample_rate = 16000;
  int64_t hop_length = 160;
  int64_t audio_latent_downsample_factor = 4;
  bool is_causal = true;
  int64_t shift = 0;
};
// `b c t f -> b t (c f)` (:301-306).
std::vector<float> Ltx2AudioPatchify(const float* latent, const Ltx2AudioLatentShape& shape);
// `b t (c f) -> b c t f` (:325-331).
std::vector<float> Ltx2AudioUnpatchify(const float* tokens, const Ltx2AudioLatentShape& shape);
// get_patch_grid_bounds (:334-353): [batch, 1, frames, 2] of [start, end)
// timestamps in SECONDS.
std::vector<float> Ltx2AudioPatchTimings(const Ltx2AudioLatentShape& shape,
                                         const Ltx2AudioPatchifierParams& params);

// ---------------------------------------------------------------------------
// The recipes
// ---------------------------------------------------------------------------

// parse_model_version (ltx-core loader/helpers.py:62-81): the dot-separated
// numeric PREFIX, stopping at the first non-numeric component, so "2.3.rc1" is
// (2, 3) and "banana" is {}. Callers that want "2.4-rc2" to compare equal to its
// own generation normalize the separator first, exactly as `detect_model_version`
// does (utils/constants.py:161) — `Ltx2DetectPipelineParams` does that here.
std::vector<int64_t> Ltx2ParseModelVersion(const std::string& version);

// MultiModalGuiderParams for both streams plus the geometry defaults —
// PipelineParams (utils/constants.py:40-76).
struct Ltx2PipelineParams {
  int64_t seed = 10;
  int64_t stage_1_height = 512;
  int64_t stage_1_width = 768;
  int64_t num_frames = 121;
  double frame_rate = 24.0;
  int64_t num_inference_steps = 40;
  int64_t default_image_crf = 33;
  Ltx2MultiModalGuiderParams video_guider;
  Ltx2MultiModalGuiderParams audio_guider;

  int64_t stage_2_height() const { return stage_1_height * 2; }
  int64_t stage_2_width() const { return stage_1_width * 2; }
};

// The three generations upstream declares (utils/constants.py:80-124).
Ltx2PipelineParams Ltx2Params20();
Ltx2PipelineParams Ltx2Params23();
Ltx2PipelineParams Ltx2Params24();
Ltx2PipelineParams Ltx2Params23Hq();

// detect_params (utils/constants.py:166-179) applied to a version STRING: the
// params of the newest generation that version is at or above. This is the rule
// that gives LTX-2.5 its parameters — (2,5) >= (2,4), so 2.5 inherits
// LTX_2_4_PARAMS — and it is why a 2.5 recipe can be written from upstream at all
// rather than being invented here.
Ltx2PipelineParams Ltx2DetectPipelineParams(const std::string& version);

// distilled.py:62-84. Generation 2.5 and later sample STAGE 1 with the ancestral
// (SDE) Euler step; earlier ones use the deterministic one. Stage 2 is always
// deterministic — its 3-step refinement is too short to remove freshly injected
// noise (distilled.py:206-209).
inline constexpr int64_t kLtx2AncestralSinceMajor = 2;
inline constexpr int64_t kLtx2AncestralSinceMinor = 5;
inline constexpr double kLtx2AncestralEta = 1.0;
inline constexpr double kLtx2AncestralSNoise = 1.0;
// distilled.py:69-73. Offsets the loop's noise generator off the pipeline seed so
// its first draw is not bit-identical to the initial latent noise.
inline constexpr int64_t kLtx2AncestralNoiseSeedOffset = 10000;
bool Ltx2ShouldUseAncestralSampler(const std::string& version);

// ltx2_recipes.py:38 — how a phase builds its input.
enum class Ltx2PhaseInputTransform { kInitial, kSpatialUpsample };
// Which stepper a phase samples with (distilled.py:170-185).
//
// `kRes2s` is not only a stepper: it selects a whole SAMPLER. Upstream keeps the
// two choices separate — `DiffusionStage.__call__` takes `stepper` and `loop`
// independently (utils/blocks.py:512-513) — but they are not independently
// selectable in practice, because `res2s_audio_video_denoising_loop` REFUSES any
// stepper that is not a `Res2sDiffusionStep` (samplers.py:276-277) and no other
// loop constructs one. `TI2VidTwoStagesHQPipeline` passes both together, to both
// stages (ti2vid_two_stages_hq.py:285/:292 and :319/:335). One enumerator
// therefore carries both, and the alternative — a separate loop field whose only
// legal combination is this one — would publish a selection surface upstream
// does not have and three combinations that must then be refused.
//
// Row LTX25-RES2S-LOOP, issue #921. Spec .agents/specs/ltx25-res2s-loop.md.
enum class Ltx2StepperKind { kEuler, kEulerAncestral, kRes2s };

// Which denoiser upstream CONSTRUCTS for this phase — the two classes in
// ltx-pipelines `utils/denoisers.py`. `kGuided` is `GuidedDenoiser`, built from
// a `MultiModalGuider` per stream; `kSimple` is `SimpleDenoiser`, "single
// transformer call, no guidance" (`utils/denoisers.py:3`).
//
// THIS DOES NOT GATE THE SEAM, and reading it as if it did is the mistake worth
// naming here. `Ltx2GuidedDenoise` runs on EVERY phase, because a phase whose
// recipe sets no guidance keeps `Ltx2MultiModalGuiderParams`'s own defaults and
// those ARE `_POSITIVE_ONLY_GUIDER` (denoisers.py:25-28) — one pass, and a
// `calculate` whose every term is zero, which is `SimpleDenoiser`'s output.
// That equivalence is measured rather than argued; see
// .agents/specs/ltx25-guided-video.md section 10.
//
// What it DOES decide is where a request's guider override lands, and it exists
// because `allow_guidance_override` alone cannot express the a2vid case. That
// field answers "does this pipeline's CLI carry the guider flags at all":
// `distilled.py` selects `default_2_stage_distilled_arg_parser`
// (utils/args.py:1188), which never adds them, so an override there names a knob
// the pipeline has no surface for and is REFUSED. `a2vid_two_stage.py:311`
// selects `default_2_stage_arg_parser` (utils/args.py:1123), which DOES carry
// them (utils/args.py:947-1006, the six video-guider flags) — and they reach
// stage 1's guider alone (`:233-236`),
// because stage 2 constructs `SimpleDenoiser(v_context_p, a_context_p)`
// (`:278`) and takes no params at all. So on that phase the flag is legal and
// simply does not arrive. Neither value of a boolean says that: refusing would
// reject a request upstream accepts, and applying would switch on guidance
// upstream's stage 2 does not have.
enum class Ltx2PhaseDenoiser { kGuided, kSimple };

// LTXPhaseRecipe (ltx2_recipes.py:29-50).
// WHICH of the load's adapters a PHASE runs.
//
// Upstream states this by building a second `DiffusionStage` from the same
// checkpoint with a different `loras=` argument, read at Lightricks/LTX-2
// fd4ded7f: `a2vid_two_stage.py:107` against `:114`, `ti2vid_two_stages.py:140`
// against `:151`, `ti2vid_two_stages_hq.py:154` against `:165`, and — the mirror
// image — `ic_lora.py:108` against `:119`, where the adapter rides stage 1 and
// stage 2 runs bare.
//
// UPSTREAM NEEDS TWO PLACEMENTS, not one, which is why this is a SET and not a
// "does this phase get the distilled adapter" boolean.
// `ltx-pipelines/CLAUDE.md:48` scopes the adapter to "stage 2 only in
// TI2Vid/A2Vid/Keyframe", while `:49` has HQ apply it to BOTH stages and
// `:50-51` says the same of DFR. Stage 1 `kNoAdapters` with stage 2 defaulted is
// the first; both phases defaulted is the second.
//
// TWO ENUMERATORS, and two is the COMPLETE space rather than a boolean wearing
// an enum's clothes: `Ltx2ResolveLoraReferenceFactors` refuses more than one
// adapter by name (`ltx2_lora.h:167-172`, mirroring `dubit.py:364-365` and
// `hdr_ic_lora.py:271-272`), so the powerset of the load's adapters has exactly
// two members. "Some of them" has no spelling here because it has no spelling
// anywhere in this engine yet; the day that arity cap lifts, the third value
// goes here.
//
// AND UPSTREAM HOLDS ONE TRANSFORMER, not two. Both `from_checkpoint` calls name
// the same `model_paths.transformer()` (`a2vid_two_stage.py:104` and `:116`,
// `ti2vid_two_stages.py:137` and `:148`) and differ only in the adapter tuple.
// So a phase-scoped adapter over one resident DiT is what upstream does, and a
// second resident weight set would be a heavier architecture than the reference
// rather than a faithful port of it.
//
// NO PER-PHASE STRENGTH, deliberately. `ti2vid_two_stages_hq.py` needs one —
// 0.25 at `:92-96` and 0.5 at `:97-101` — and no recipe this tree ships would
// set it, so adding the field now lands a branch nothing can select. That is the
// argument `ltx2_lora.h:41-44` already makes for the second product form. Owed
// by https://github.com/mudler/vllm.cpp/issues/1144 — NOT #921, which was closed
// as completed the same day this landed and would have left the debt looking
// owned while owning nothing. The trap that makes it more than a new field is
// written beside `Ltx2RebindDitLoras` in `ltx2_loader.h`: that function's no-op
// test is a BOOLEAN, and HQ needs both stages fused at different strengths.
enum class Ltx2PhaseLoraScope {
  // Every adapter the load supplied. The DEFAULT, because `distilled.py:131`
  // builds ONE stage set and so every recipe that predates this field —
  // `distilled_two_stage`, `dfr`, `retake`, `one_stage`, `res2s`, `t2a_one_stage`
  // — is upstream-correct running the adapters on all of its phases. A different
  // default would silently move six gated arms.
  kAllAdapters,
  // The base weights. `ic_lora.py:119`'s `loras=()`, and
  // `a2vid_two_stage.py:107`'s stage 1 relative to the distilled adapter that
  // `requires_distilled_lora` identifies.
  kNoAdapters,
};

// WHICH token count the sigma SHIFT is fitted on, for a phase whose schedule is
// derived rather than frozen.
//
// `LTX2Scheduler.execute` takes an OPTIONAL latent and `schedulers.py:31` is
// `tokens = math.prod(latent.shape[2:]) if latent is not None else
// default_number_of_tokens`. So upstream selects between two anchors by passing
// a latent or not, and `default_number_of_tokens` is `MAX_SHIFT_ANCHOR` = 4096
// (`schedulers.py:11`, `:29`).
//
// SEVEN CALL SITES AT `fd4ded7f`, AND SIX OF THEM PASS NO LATENT. The
// population is `grep -rn '\.execute(' packages/ltx-pipelines/src/ltx_pipelines/`
// and it is small enough to list in full:
//
//   ti2vid_one_stage.py:207      no latent      our `one_stage` x4
//   t2a_one_stage.py:141         no latent      our `t2a_one_stage`
//   retake.py:287                no latent      our `retake`, non-distilled arm
//   a2vid_two_stage.py:226       no latent      our `a2vid_two_stage` stage 1
//   ti2vid_two_stages.py:244     no latent      our `ti2vid_two_stage` stage 1
//   keyframe_interpolation.py:200 no latent     our `keyframe_interpolation` stage 1
//   ti2vid_two_stages_hq.py:267  latent=empty_latent   our `res2s_two_stage`
//
// So the LATENT-DERIVED anchor is upstream's exception, not its rule — which is
// the opposite of how this engine has always behaved, since `ltx2_video.cpp`
// passes `target_tokens` on every phase. That divergence is
// https://github.com/mudler/vllm.cpp/issues/1150 and it is REAL rather than a
// rounding: at the recipe default geometry the target latent is 6144 tokens,
// giving a shift of 2.78 against upstream's 2.05, so every sigma in the
// schedule moves while the frame count, the shapes and the sample rate do not.
//
// THE DEFAULT IS TODAY'S BEHAVIOUR AND NOT UPSTREAM'S MAJORITY, deliberately.
// Flipping it would re-sample `one_stage` at four version keys,
// `a2vid_two_stage` stage 1 and `retake`, all shipped and gated, and rewrite
// their goldens — on a finding made inside a row scoped to add one recipe. #1150
// owns that flip and this enum is the seam it uses. The preserving default also
// cannot fail SILENTLY: an arm moves only where a line says so, whereas under
// the flip an arm nobody remembered to pin would move with nothing naming it.
//
// Read in exactly one place, the phase loop's schedule block, and only on the
// branch that derives a schedule at all. A phase carrying explicit `sigmas`
// never reaches it.
enum class Ltx2PhaseScheduleTokens {
  // `math.prod(latent.shape[2:])` of THIS phase's target grid, which is
  // `ti2vid_two_stages_hq.py:267`'s `latent=empty_latent`. The default.
  kTargetLatent,
  // `default_number_of_tokens`, i.e. 4096 — what the six call sites above get
  // by passing no latent at all.
  kSchedulerDefault,
};

struct Ltx2PhaseRecipe {
  std::string name;
  Ltx2MultiModalGuiderParams video_guidance;
  Ltx2MultiModalGuiderParams audio_guidance;
  int64_t spatial_downscale = 1;
  // Empty means "derive the schedule from the scheduler at run time"; a non-empty
  // one is an explicit distilled schedule and fixes `num_inference_steps`.
  std::vector<float> sigmas;
  double noise_scale = 0.0;
  Ltx2PhaseInputTransform input_transform = Ltx2PhaseInputTransform::kInitial;
  bool allow_guidance_override = true;
  // See `Ltx2PhaseDenoiser`. Read in exactly one place — where a request's
  // guider overrides are applied — and only AFTER the refusal above, so no
  // recipe that refuses an override can reach it.
  Ltx2PhaseDenoiser denoiser = Ltx2PhaseDenoiser::kGuided;
  bool use_official_sigma_schedule = true;
  // See `Ltx2PhaseScheduleTokens`. Only consulted when `sigmas` is empty.
  Ltx2PhaseScheduleTokens schedule_tokens = Ltx2PhaseScheduleTokens::kTargetLatent;
  // The adapter set this phase runs. Read in exactly one place — the phase
  // loop's rebind, immediately before the phase's first DiT forward — and
  // honoured by `Ltx2RebindDitLoras`, which re-materializes only the tensors an
  // adapter targets so that no second weight set ever exists.
  Ltx2PhaseLoraScope loras = Ltx2PhaseLoraScope::kAllAdapters;
  Ltx2StepperKind stepper = Ltx2StepperKind::kEuler;
  double stepper_eta = 0.0;
  double stepper_s_noise = 1.0;
  int64_t noise_seed_offset = 0;

  // ltx2_recipes.py:48-50: `None` when the schedule is not explicit. -1 here.
  int64_t num_inference_steps() const;
};

// WHICH OF UPSTREAM'S TWO IMAGE-CONDITIONING BUILDERS this recipe runs over its
// `images` list. Row LTX25-KEYFRAME-INTERP, issue #1096.
//
// The two functions live side by side in
// `ltx-pipelines/utils/helpers.py` and differ by exactly one branch:
//
//                       combined_image_conditionings   ..._by_adding_guiding_latent
//                       (:272-308)                     (:343-367)
//   frame_idx == 0      VideoConditionByLatentIndex    VideoConditionByKeyframeIndex
//   any other frame_idx VideoConditionByKeyframeIndex  VideoConditionByKeyframeIndex
//
// The second has no branch at all — one loop, one item type. And the two items
// do different things to the state: `VideoConditionByLatentIndex` REPLACES the
// clean tokens of latent frame 0 and the token count never changes
// (latent_cond.py:38-39), while `VideoConditionByKeyframeIndex` APPENDS a latent
// frame of tokens at the end (keyframe_cond.py:79-82).
//
// A RECIPE FIELD RATHER THAN A PHASE ONE, because upstream picks the builder per
// PIPELINE: `keyframe_interpolation.py` calls the same one for both of its
// stages (`:211` and `:260`, differing only in the height and width they pass).
// A per-phase field would offer a combination upstream has no call site for.
//
// AND NOT A `pipeline_kind` STRING COMPARE at the reader, for the reason
// `audio_only` and `requires_audio_input` give below: the recipe table is the
// one place that knows, and a string test at the call site is one more chance
// for the next recipe on this builder to be missed.
//
// WHY THIS IS INVISIBLE WITHOUT A DELIBERATE GATE. A `keyframe_interpolation`
// render built on the replace arm returns a clip of the right size, the right
// frame count and the right sample rate, with the supplied image pinned into it.
// It IS conditioned — it is conditioned as a different pipeline. The only
// observable is the sequence LENGTH the DiT ran over, which is
// `Ltx2ConditioningTrace::video_tokens`; no pixel comparison and no shape check
// reads it.
enum class Ltx2ImageConditioningBuilder {
  // `combined_image_conditionings` (helpers.py:272-308). Frame 0 REPLACES. Every
  // other pipeline in `ltx-pipelines`, so this is upstream's majority as well as
  // this engine's incumbent behaviour — the default moves nothing.
  kCombined,
  // `image_conditionings_by_adding_guiding_latent` (helpers.py:343-367). Frame 0
  // APPENDS, like every other frame. `keyframe_interpolation.py:211`, `:260`.
  kAddGuidingLatent,
};

// WHICH CLASS OF TRANSFORMER a load carries. Row LTX25-CHECKPOINT-CLASS,
// issue #1137.
//
// This is a CLAIM BY THE CALLER, not a detection, and the distinction is the
// whole row. Four real LTX-2.5 safetensors headers were read on 2026-08-20 and
// none of them carries a field that separates a distilled transformer from a
// full one:
//
//   - `__metadata__["config"]` is 2199 bytes and BYTE-IDENTICAL between
//     `ltx-2.5-22b-dev-transformer-bf16.safetensors` (full) and
//     `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` (distilled), and
//     `model_version` is `2.5.0` on both.
//   - The one structural difference between those two,
//     `keyframes_abs_pos_embedding`, is NOT a class marker. The distilled file's
//     own config still declares `"use_keyframes_abs_pos_embedding": true` while
//     the tensor is absent, and a THIRD file named distilled —
//     `vonkaiser/LTX-2.5-FP8-NVFP4`'s `ltx-2.5-22b-distilled-fp8.safetensors` —
//     carries it as `F8_E4M3 [1, 4096]` over a base tensor-name set that is
//     exactly the dev file's 4349 names.
//   - The two bf16 transformers are the SAME SIZE, 42,018,190,584 bytes each,
//     so `ls -l` does not separate them either.
//
// The measurement, its limits and the one file that would settle it are in
// `.agents/specs/ltx25-checkpoint-class.md` section 2. A detector built on any
// of those fields would be a guess, and a wrong detector is worse than none:
// it would refuse a correct load, or admit the wrong one with a green check
// beside it.
enum class Ltx2CheckpointClass {
  // "Full" in upstream's table. The undistilled base transformer.
  kFull,
  // "Distilled only". The transformer distilled onto the 8-sigma schedule.
  kDistilled,
  // `DFRPipeline`'s "Keyframe-slot SFT" base (dfr_pipeline.py:157, "on a
  // keyframe-slot-capable SFT base plus a distilled LoRA"). Neither of the
  // other two: it is a separate fine-tune.
  kKeyframeSlotSft,
};

// WHICH CLASS A RECIPE NEEDS — one enumerator per distinct checkpoint half of
// the `Model` column in `packages/ltx-pipelines/CLAUDE.md:17-30 @ fd4ded7f`.
//
// The LoRA half of that column is a DIFFERENT field. `Full + distilled LoRA` is
// a class (`Full`) and an adapter (`distilled LoRA`), and the adapter half is
// `requires_distilled_lora` below. Conflating the two would let a full-model arm
// pass its class check by carrying an adapter.
enum class Ltx2RequiredCheckpointClass {
  // "Full": `TI2VidOneStagePipeline`, `T2AOneStagePipeline`,
  // `TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`,
  // `A2VidPipelineTwoStage`, `KeyframeInterpolationPipeline`.
  // `t2a_one_stage.py:50` says it again in prose: "Assumes full non distilled
  // model is provided in the checkpoint_path."
  kFull,
  // "Distilled only": `DistilledPipeline`, and upstream's `ICLoraPipeline` and
  // `DubItPipeline`, which this tree does not ship.
  kDistilled,
  // "Keyframe-slot SFT": `DFRPipeline`.
  kKeyframeSlotSft,
  // "Full or distilled": `RetakePipeline`, and it is a CONDITION rather than a
  // hole. `retake.py:71-73` states it: "Set to ``True`` if using distilled model
  // or passing distillation lora with full model." This tree's `RetakeRecipe`
  // mirrors upstream's CLI, which hard-codes `distilled=True` (`retake.py:336`,
  // `:359`) and then takes `DISTILLED_SIGMAS` (`:287`), so a `full` checkpoint
  // with no adapter would run the distilled schedule on undistilled weights —
  // #1137 again, one enumerator down. `Ltx2CheckpointClassRefusal` therefore
  // requires an adapter on the `full` arm and accepts `distilled` outright.
  kFullOrDistilled,
  // NO REFERENCE STATES A CLASS. `dmd2` alone: it comes from vLLM-Omni's
  // `_PIPELINE_RECIPES` (`ltx2_recipes.py:160-167 @ a4ea67a2`), whose table has
  // no `Model` column, and it has no row in Lightricks' table at all. Recorded
  // as unstated rather than defaulted to a permissive value, because a silent
  // `kFullOrDistilled` would read as a decision somebody made. Owed in
  // `.agents/specs/ltx25-checkpoint-class.md`.
  kUnstated,
};

// LTXPipelineRecipe (ltx2_recipes.py:53-87).
struct Ltx2PipelineRecipe {
  std::vector<Ltx2PhaseRecipe> phases;
  int64_t height = 512;
  int64_t width = 768;
  int64_t num_frames = 121;
  double frame_rate = 24.0;
  int64_t num_inference_steps = 40;
  // Not a vLLM-Omni field: Lightricks resolves the conditioning-image CRF per
  // generation (utils/constants.py:36-37, 124) and it is a property of the model
  // generation, so it travels with the recipe rather than being re-derived.
  int64_t default_image_crf = 33;
  std::string negative_prompt;
  int64_t video_output_phase = -1;
  int64_t audio_output_phase = -1;
  bool allow_request_sigmas = true;
  bool allow_request_latents = true;
  bool allow_negative_prompt = true;
  bool fixed_num_inference_steps = false;

  // `T2AOneStagePipeline` (t2a_one_stage.py:43). TRUE means the pipeline passes
  // `video=None` to the stage (`:167`) and returns a waveform and nothing else
  // (`:172`) — there is no video latent, no video VAE decode and no frame.
  //
  // A FLAG ON THE RECIPE RATHER THAN A STRING COMPARE AT THE CALL SITE, because
  // the engine has to answer "is there a picture" in four places (geometry,
  // resolution guard, decode, artifacts) and four independent `kind ==
  // "t2a_one_stage"` tests are four chances for one of them to be missed on the
  // next audio-only recipe. The recipe table is the one place that knows.
  bool audio_only = false;

  // `A2VidPipelineTwoStage` (a2vid_two_stage.py:53). TRUE means a driving
  // waveform is not optional: `--audio-path` is `required=True` (`:312-317`) and
  // the whole pipeline is "denoise video AROUND this take", with the audio
  // stream frozen at both stages (`:251-256`, `:291-296`).
  //
  // FLAGS ON THE RECIPE, not `pipeline_kind` string compares at the two call
  // sites, for the reason `audio_only` gives above. The second one already has a
  // second user waiting: `ti2vid_two_stages` (#1093) and
  // `keyframe_interpolation` (#1096) both select a parser where
  // `--distilled-lora` is `required=True` (utils/args.py:1140-1155).
  //
  // WITHOUT THE TAKE the render still finishes. It returns a clip of the right
  // size, the right frame count and the right sample rate, with the soundtrack
  // generated rather than supplied — which is the ordinary joint-generation
  // behaviour and is indistinguishable from audio-to-video that ignored its
  // input.
  bool requires_audio_input = false;
  // `--distilled-lora` is `required=True` on the two-stage parser this pipeline
  // selects (utils/args.py:1140-1155, `default_2_stage_arg_parser` at `:1123`),
  // and stage 2's three-sigma refinement (`:164`) is what that adapter was
  // trained for. A recipe that fixes this flag cannot render on a checkpoint
  // carrying no adapter without running a distilled schedule on undistilled
  // weights.
  //
  // THE PLACEMENT IS NOT THIS FLAG'S JOB, and it is no longer missing. This
  // comment used to end "this engine fuses at load into one weight set", owed by
  // #1118. Row LTX25-PHASE-LORA closed that: `Ltx2PhaseRecipe::loras` carries
  // upstream's per-stage adapter set, and `A2VidTwoStageRecipe` gives stage 1
  // `kNoAdapters` (`a2vid_two_stage.py:107`) against stage 2's default
  // (`:114`, `stage_2_loras = (*loras, *distilled_lora)`).
  //
  // What this flag says is only that the load must CARRY an adapter, mirroring
  // `--distilled-lora required=True`. What the phase field says is which stage
  // runs it. The two were conflated while only one placement existed.
  bool requires_distilled_lora = false;

  // WHICH CLASS OF TRANSFORMER this pipeline can run — the checkpoint half of
  // upstream's `Model` column. Row LTX25-CHECKPOINT-CLASS, issue #1137.
  //
  // A FLAG ON THE RECIPE, not a `pipeline_kind` string compare at the refusal,
  // for the reason `audio_only` and `requires_distilled_lora` give above: the
  // table is the one place that knows, and the next recipe inherits the rule
  // instead of being missed.
  //
  // THE DEFAULT IS THE MOST DEMANDING VALUE any row takes, so a recipe added
  // later that forgets this field refuses rather than admits. The default is not
  // the gate: `test_ltx2_pipeline` walks every `(kind, version)` pair this table
  // resolves and pins each requirement against the upstream row.
  //
  // WITHOUT THIS FIELD the render still finishes. A distilled transformer on a
  // `Full` arm returns a clip of the right size, the right frame count and the
  // right sample rate, sampled in a regime the weights were never trained for —
  // and no pixel, RMS, windowed-energy or spectral check can see it, because the
  // model ran.
  Ltx2RequiredCheckpointClass checkpoint_class = Ltx2RequiredCheckpointClass::kFull;

  // See `Ltx2ImageConditioningBuilder`. Read in exactly one place — the phase
  // loop's first-frame arm, which is the only site where the two builders
  // disagree. The last-frame arm below it is unchanged, because `frame_idx != 0`
  // takes the keyframe item under BOTH of them.
  Ltx2ImageConditioningBuilder image_conditioning = Ltx2ImageConditioningBuilder::kCombined;

  int64_t max_spatial_downscale() const;
};

// The spelling of a class in the `checkpoint_class` load extra and in a refusal.
const char* Ltx2CheckpointClassName(Ltx2CheckpointClass value);

// The three accepted spellings, comma-separated, for a message that has to list
// them. Derived from the enum so a fourth class cannot be added without every
// message following it.
std::string Ltx2CheckpointClassSpellings();

// `true`, and writes `*out`, when `text` is one of the spellings above. `false`
// otherwise, and `*out` is untouched.
bool Ltx2ParseCheckpointClass(const std::string& text, Ltx2CheckpointClass* out);

// What a recipe needs, in the words of upstream's `Model` column, for a refusal
// to quote.
std::string Ltx2RequiredCheckpointClassName(Ltx2RequiredCheckpointClass value);

// THE DECISION, kept beside the recipe table it reads rather than in the engine,
// so the table and the rule that consumes it cannot drift into two files.
//
// Returns the EMPTY STRING when the load is acceptable, and the refusal text
// otherwise. `declared` is the raw `checkpoint_class` load extra, empty when the
// caller supplied none. `has_lora` says whether the load carries an adapter,
// which only `kFullOrDistilled` reads.
//
// Four refusals: an unparseable value, an absent declaration on a recipe that
// needs a specific class, a declared class the recipe cannot run, and a `full`
// declaration on `kFullOrDistilled` with no adapter.
std::string Ltx2CheckpointClassRefusal(const Ltx2PipelineRecipe& recipe,
                                       const std::string& pipeline_kind,
                                       const std::string& declared, bool has_lora);

// `assert_resolution` (ltx-pipelines utils/helpers.py:540-551). Upstream calls it
// at the top of a pipeline's `__call__`, before any work is paid for — NINE
// invocations, counted at the pin, among them ti2vid_two_stages.py:184 and
// ti2vid_two_stages_hq.py:199 (both `is_two_stage=True`) against
// ti2vid_one_stage.py:156 (`False`). Nine, and not the twenty-one lines a grep
// for the name returns: those are 9 invocations + 1 definition + 10 imports + 1
// `__all__` string. Nor is it every pipeline: 13 pipeline `__call__`s take a
// height and a width, and the four that do NOT call the guard are
// distilled_mgpu.py:143, ti2vid_two_stages_mgpu.py:163,
// ti2vid_two_stages_hq_mgpu.py:164 and hdr_ic_lora.py:352.
//
// Upstream spells the divisor as a literal 64 or 32 chosen by a bool. That pair
// is not two constants: it is the VAE spatial factor (32,
// ltx_core/types.py:31-33) times the worst spatial downscale any phase applies.
// A two-stage pipeline runs stage 1 at `width // 2`
// (ti2vid_two_stages.py:226-228), so the request must survive being halved and
// still divide the grid — hence 32 * 2. Taking the divisor as a parameter lets
// the caller derive it from the recipe it actually holds, which reproduces
// upstream's two numbers on the two shipped arms rather than restating them as
// literals.
//
// It reproduces them; it does not generalise past them, and the limit is stated
// rather than implied. `max_spatial_downscale()` takes the MAXIMUM, and the
// quantity a request must survive is the LEAST COMMON MULTIPLE of the phase
// downscales. The two agree on every shipped recipe, whose downscales are 1 and
// 2, and they part on a recipe with phases at 2 and 3: the max gives 96, a
// 96-wide request passes, and the downscale-2 phase then floors 48 onto one
// latent cell — the very defect this guard exists to stop. No shipped recipe has
// a non-power-of-two downscale, so the lcm form would change no behaviour any
// production entry point can reach and no test entering there could gate it.
// Recorded as a limitation in `.agents/specs/ltx25-resolution-envelope.md`
// instead of implemented unreached.
//
// This is a REFUSAL and not a rounding on purpose. Integer division is what the
// engine did before, and it renders a clip at a size nobody asked for
// (#919) — 80 became 64, successfully, with the wrong size reported back.
void Ltx2AssertResolution(int64_t height, int64_t width, int64_t divisor);

// resolve_ltx_pipeline_recipe (ltx2_recipes.py:170-175). Keyed on the EXACT
// (pipeline_kind, model_version) pair and throwing by name on anything else —
// never defaulting. The table:
//
//   ("one_stage",          "2")    vLLM-Omni LTX2_ONE_STAGE_RECIPE (:109-111)
//   ("one_stage",          "2.3")  vLLM-Omni LTX23_ONE_STAGE_RECIPE (:112-115)
//   ("one_stage",          "2.4")  Lightricks LTX_2_4_PARAMS (constants.py:124)
//   ("one_stage",          "2.5")  Lightricks, via _PARAMS_SINCE_VERSION (:130-133)
//   ("distilled_two_stage","2")    vLLM-Omni LTX2_DISTILLED_TWO_STAGE_RECIPE (:125-158)
//   ("distilled_two_stage","2.5")  Lightricks distilled.py + constants.py:17-23
//   ("res2s_two_stage",    "2.5")  Lightricks ti2vid_two_stages_hq.py:59-340 plus
//                                  LTX_2_3_HQ_PARAMS (constants.py:95-115). Row
//                                  LTX25-RES2S-LOOP, #921. The res_2s sampler on
//                                  BOTH stages, 15 steps, STG off. 2.5 only, and
//                                  not by analogy with the one_stage rows:
//                                  `LTX_2_3_HQ_PARAMS` is a plain constant that
//                                  overrides every generation-varying knob
//                                  (constants.py:91-94 says so), so there is no
//                                  `detect_params` lineage to spread it across
//                                  versions. THE SAMPLER IS THE PRESET: this
//                                  recipe on `kEuler` would render a finished,
//                                  correctly sized, plausible clip at half the
//                                  model evaluations 15 steps was tuned for
//   ("dmd2",               "2")    vLLM-Omni LTX_POSITIVE_ONLY_RECIPE (:116-124)
//   ("dmd2",               "2.3")  same
//   ("dfr",                "2.5")  Lightricks dfr_pipeline.py:155-561 (row
//                                  LTX25-DFR-PIPELINE, #986). The distilled
//                                  two-stage SCHEDULE with DFR's phase names —
//                                  upstream defaults stage 1 to DISTILLED_SIGMAS
//                                  and stage 2 to STAGE_2_DISTILLED_SIGMAS
//                                  (:281-282) and halves stage 1 (:319), so DFR
//                                  differs in its CONDITIONING and its rounds
//                                  loop rather than in its schedule. 2.5 only:
//                                  its base stage needs a checkpoint declaring
//                                  `use_keyframes_abs_pos_embedding`
//   ("retake",             "2")    Lightricks retake.py:85,287,290-294,313-324
//   ("retake",             "2.5")  same
//   ("a2vid_two_stage",    "2")    Lightricks a2vid_two_stage.py:53,143 (row
//   ("a2vid_two_stage",    "2.3")  LTX25-A2VID-RECIPE, #1117). Stage 1 denoises
//   ("a2vid_two_stage",    "2.4")  VIDEO at half resolution, guided by the
//   ("a2vid_two_stage",    "2.5")  params table's video row and a scheduler-
//                                  DERIVED schedule (:225-227), with the audio
//                                  stream frozen on the caller's own take
//                                  (:251-256); stage 2 upsamples 2x and refines
//                                  with STAGE_2_DISTILLED_SIGMAS and no guider
//                                  at all (:277-297). It is NOT
//                                  `distilled_two_stage` with a take attached:
//                                  that recipe fixes both stages' sigmas, fixes
//                                  its guidance, and samples stage 1 with the
//                                  ANCESTRAL stepper on 2.5, where A2Vid passes
//                                  no `stepper` and gets `EulerDiffusionStep()`
//                                  (utils/blocks.py:526-527)
//   ("ti2vid_two_stage",   "2")    Lightricks ti2vid_two_stages.py:61,159 (row
//   ("ti2vid_two_stage",   "2.3")  LTX25-TI2VID-RECIPE, #1093). Upstream's PLAIN
//   ("ti2vid_two_stage",   "2.4")  two-stage pipeline: stage 1 is the FULL model
//   ("ti2vid_two_stage",   "2.5")  under CFG at half resolution on a scheduler-
//                                  DERIVED schedule (:243-245) with the
//                                  distilled adapter withheld (:140), stage 2
//                                  spatially upsamples (:272) and refines on the
//                                  frozen STAGE_2_DISTILLED_SIGMAS (:178) with
//                                  the adapter (:151) and no guider (:290). It
//                                  is NOT `distilled_two_stage`, which builds
//                                  ONE stage set (distilled.py:131), freezes
//                                  stage 1's sigmas and samples 2.5 with the
//                                  ANCESTRAL stepper; nor `res2s_two_stage`,
//                                  which puts the adapter on BOTH stages at
//                                  0.25/0.5 and runs the second-order sampler.
//                                  Four keys for the a2vid reason: `main()`
//                                  takes whatever `resolve_cli_params()` read
//                                  off the checkpoint (:318-319). ALONE among
//                                  the derived arms it fits its sigma shift on
//                                  the 4096 anchor rather than the target grid,
//                                  because `execute(steps=...)` passes no latent
//                                  (schedulers.py:31) — see
//                                  `Ltx2PhaseScheduleTokens`
//   ("t2a_one_stage",      "2")    Lightricks t2a_one_stage.py:43,109 (row
//   ("t2a_one_stage",      "2.3")  LTX25-T2A-ONE-STAGE, #1005). The one_stage
//   ("t2a_one_stage",      "2.4")  rows' own schedule with `audio_only` set:
//   ("t2a_one_stage",      "2.5")  T2A hard-codes the SAME `LTX2Scheduler()`
//                                  (t2a_one_stage.py:67 against
//                                  ti2vid_one_stage.py:81) and the same
//                                  `detect_params` step count, and differs in
//                                  carrying no video stream at all
//
// The four `t2a_one_stage` rows mirror the four `one_stage` rows one for one, and
// the negative prompt follows the same split for the same reason: it travels with
// the GENERATION, not with the pipeline. There is no "which versions support
// text-to-audio" question upstream — `T2AOneStagePipeline` takes whatever
// `resolve_cli_params` read off the checkpoint (t2a_one_stage.py:178-179), so
// restricting these rows to 2.5 would be a local invention.
//
// The `retake` rows are Lightricks' `RetakePipeline` and have no vLLM-Omni
// counterpart at all. Every value on them is read off `retake.py` rather than
// adapted from a neighbouring recipe, because the neighbouring recipe is wrong
// in a way that renders: `distilled_two_stage` runs its first stage at
// `spatial_downscale = 2`, and retake seeds the video stream with a latent
// encoded from the source clip at FULL resolution (retake.py:317-318 passes
// `output_shape.width` / `.height` straight through). Riding that recipe would
// put a full-resolution latent into a half-resolution grid.
//
// ONE phase (one `DiffusionStage` call at retake.py:313-324), `DISTILLED_SIGMAS`
// (:287, because `distilled` defaults True at :85 and the CLI hard-codes it at
// :359), plain Euler — `DiffusionStage.__call__` defaults to
// `euler_denoising_loop` and `EulerDiffusionStep()` (utils/blocks.py:524-527)
// and retake overrides neither, so the ancestral sampler that `distilled.py`
// selects for 2.5 reaches retake through nothing — and no negative prompt,
// because the distilled arm builds a `SimpleDenoiser` (:290-294) and encodes
// `[prompt]` alone (:259).
//
// The geometry fields on a `retake` recipe are the params table's, and the
// engine OVERRIDES all four from the source clip, which is what upstream does
// (`output_shape` from `get_videostream_metadata`, retake.py:220, passed at
// :317-320).
//
// The 2.4 and 2.5 rows exist here and not upstream in vLLM-Omni, which carries no
// row past 2.3 (spec section 3). Their VALUES are Lightricks', not invented: the
// one_stage rows are `detect_params`' own resolution of that version, and the
// distilled 2.5 row is the 2.0 one plus `should_use_ancestral_sampler`, which is
// the single thing distilled.py changes for generation 2.5.
Ltx2PipelineRecipe ResolveLtx2PipelineRecipe(const std::string& pipeline_kind,
                                             const std::string& model_version);

// ---------------------------------------------------------------------------
// Out of scope, refused by name (spec section 2 "Out"; the 2026-08-13 grounding
// pass is .agents/specs/ltx25-retire-dead-arms.md, row LTX25-RETIRE-DEAD-ARMS)
// ---------------------------------------------------------------------------

// Each of these renders something plausible if it is silently downgraded, which
// is why none of them falls back. `Ltx2RefuseUnportedPipelineFeature` throws with
// a message naming the missing piece and the row that owes it.
//
// TWO KINDS live here, and conflating them overstated what this port refuses:
//
//   REACHABLE REFUSAL — a product path constructs the condition and throws, so a
//   caller CAN trip it. `kSpatiotemporalUpsampler` (ltx2_upsampler.cpp:465) is
//   the ONE. `ltx2_video.cpp` reaches it through `Ltx2UpsampleVideoLatent` when a
//   phase asks for the spatial-upsample transform. The TEMPORAL-ONLY x2
//   upsampler is NOT among them: it is ported (`2e9d95e74`, spec
//   .agents/specs/ltx25-temporal-upsampler.md), which is why the enumerator that
//   used to be spelled `kTemporalUpsampler` now names the spatiotemporal arm
//   only. Nothing shipped drives the ported arm yet, so it is gated, not served.
//
//   The definition above is a claim about CALLERS, and it takes a caller to
//   satisfy it. A `case` label is not one. `kBetaScheduler` was published here as
//   the second reachable arm for the whole of row LTX25-RETIRE-DEAD-ARMS, and it
//   is not: its call site `ltx2_pipeline.cpp:199` sits inside `Ltx2Schedule`,
//   which nothing calls (#889). Recorded rather than quietly moved, because the
//   row's subject is exactly this — a classification asserted instead of derived,
//   and AGENTS.md `## Nothing lands dead` now names the shape it took. The split
//   is gated by `test_ltx2_pipeline`'s "the reachable/marker split matches the
//   source", which walks src/, include/ and examples/ and carries two positive
//   controls in the same walk. That gate is the anti-tautological shape #691
//   asked for, for the beta arm; #691 stays open for the other three markers.
//
//   DECLARED-OUT-OF-SCOPE MARKER — no request field, load extra or CLI flag asks
//   for it, so nothing outside the ledger test reaches it. It is a record of what
//   upstream HAS and this port does NOT, which is worth keeping; calling it a
//   refusal is what was wrong. `kBetaScheduler`, `kInt8ConvRot` and
//   `kMultiGpuParallelism` are markers, and their messages say so.
//
// THREE ENUMERATORS HAVE BEEN RETIRED, recorded here because the retirement IS
// the record — a reader who finds them in git history needs to know they did not
// simply move. Two on 2026-08-13 because they were wrong, one on 2026-08-15
// because it came true:
//
//   `kLoraFusion` — RETIRED 2026-08-15 by row LTX25-IC-LORA (#923) because it
//   came TRUE. It said LoRA fusion was out of scope and carried the
//   DECLARED, NOT REQUESTABLE marker, which asserted that no request field or
//   load extra asks for it. The `lora_path` / `lora_strength` load extras now
//   do, and `Ltx2DitLoadOptions::loras` fuses the delta into every arm, so the
//   marker's own sentence had become false. #691 predicted this exact drift in
//   its own words — the ledger test gates the message TEXT and not the property,
//   so nothing here would have caught it — and that is why the enumerator is
//   REMOVED rather than reclassified: there is no longer an unported LoRA-fusion
//   feature to name, and a refusal for a served capability is worse than none.
//
//   `kMultishot` — FABRICATED. It refused "multishot generation" and cited
//   "ltx-pipelines multishot entry points". No such entry point, symbol or string
//   exists in Lightricks/LTX-2 @ fd4ded7f or huggingface/diffusers @ 3a2f35d4.
//   Searched as a SUBJECT rather than by our own phrasing: upstream's only sense
//   of "shot" is ONE camera take (duration_head.py:1,5 "predicts shot duration";
//   README.md:136 "a cinematographer describing a shot list"). `scene` has THREE
//   senses upstream and none is a generation mode: `scene-linear` HDR colour
//   (ltx-core color/hlg.py, hdr.py), PySceneDetect in the TRAINER — the only CODE
//   sense — and prompt-writing guidance, which ships at INFERENCE inside
//   `ltx-core`, in text_encoders/gemma/encoders/prompts/ as
//   gemma{3,4}_{i2v,t2v}_system_prompt.txt. That third sense is why the
//   retirement HOLDS rather than being undermined: those prompts tell the
//   enhancer NOT to describe scene cuts and to keep a "Single continuous take"
//   (gemma3_i2v:6,18, gemma3_t2v:24, gemma4_i2v:3). A defect in our record is not
//   a gap in our port, so there was nothing to owe.
//
//   Recorded because it is the row's own subject: this paragraph used to claim
//   that `scene` appeared upstream ONLY as PySceneDetect in the trainer. It was
//   an absence asserted from our own vocabulary with no positive control — #604 —
//   shipped in the header of the row that exists to retire #604 instances, and it
//   took a third review round to find. The derivation, with its positive control
//   in the same command, is .agents/specs/ltx25-retire-dead-arms.md §1.1.
//
//   `kVideoEngineWiring` — LANDED. It said the end-to-end composition through
//   `vllm::multimodal::VideoEngine` "is phase L7, not L5"; L7 shipped in
//   `cefacd2d0`. A refusal whose subject shipped is a false statement.
enum class Ltx2UnportedPipelineFeature {
  // Reachable refusal. Singular.
  // model/upsampler with BOTH flags set. The temporal-ONLY arm is ported
  // (.agents/specs/ltx25-temporal-upsampler.md); this one is a different
  // operator — `Conv3d(mid, 8*mid)` + `PixelShuffleND(3)`, model.py:55-59.
  kSpatiotemporalUpsampler,
  // Declared-out-of-scope markers.
  kBetaScheduler,        // ltx-core components/schedulers.py:91-120. A MARKER because
                         //   upstream constructs it nowhere: all seven ltx-pipelines entry
                         //   points hard-code `LTX2Scheduler()`, and vLLM-Omni @ a4ea67a21
                         //   has zero hits for the name. Mirroring that means no
                         //   scheduler-kind field here either, so nothing reaches the
                         //   refusal — `Ltx2Schedule`, which holds it, has no caller.
  kInt8ConvRot,          // ComfyUI-ecosystem quantization, and NOT an LTX-2 arm: the four
                         //   inference kinds upstream defines are fp8-cast / fp8-scaled-mm /
                         //   nvfp4-cast / nvfp4-prequant (quantization_factory.py:23-26).
                         //   `convrot` is nowhere at all; int8 is UNREACHABLE rather than
                         //   absent — trainer-only for anything wired, plus one DEAD kernel
                         //   in ltx-kernels (triton_ops.py:35,43). §1.2 of the row spec
  kMultiGpuParallelism,  // ltx-pipelines/multigpu — four forms: sequence-parallel,
                         //   tiled data parallel, distributed VAE decode, and
                         //   batch-parallel Gemma encoding (bp_gemma_builder.py:42,
                         //   `BatchParallelGemmaBuilder`), which partitions a prompt
                         //   list across ranks. None is CFG batching, and the reason
                         //   is upstream's own: docs/multigpu/gemma.md:103-104 calls a
                         //   positive+negative pair "the typical CFG case" and records
                         //   that the DISTILLED pipeline this port runs takes no
                         //   negative_prompt, so it "runs without CFG". This used to
                         //   assert instead that the string was absent from both
                         //   multigpu trees, which came from a PATH-FILTERED grep that
                         //   excluded the docs/ tree carrying the answer: 5 hits, not
                         //   0, against 33 files as the control (#892, §1.3 of the row
                         //   spec). The false sentence is not repeated here, because
                         //   the gate on it matches TEXT and cannot tell a quotation
                         //   from a claim
};
[[noreturn]] void Ltx2RefuseUnportedPipelineFeature(Ltx2UnportedPipelineFeature feature);

}  // namespace vllm
