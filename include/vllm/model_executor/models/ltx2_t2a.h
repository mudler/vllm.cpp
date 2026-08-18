// LTX-2.5 TEXT-TO-AUDIO — the port of `T2AOneStagePipeline`, and the first path
// in this tree that renders no picture at all.
//
// Row: LTX25-T2A-ONE-STAGE. Spec: .agents/specs/ltx25-t2a-one-stage.md.
// Issue #1005. Campaign #644 / #435.
//
// ─── WHAT THIS TU IS A PORT OF (file:line on BOTH sides) ─────────────────────
// Upstream: Lightricks/LTX-2 @ fd4ded7f
//   OURS                        <-  UPSTREAM
//   Ltx2T2aGenerate             <-  ltx-pipelines t2a_one_stage.py:109-172
//                                   (T2AOneStagePipeline.__call__)
//   the guided step             <-  ltx-core components/guiders.py:244-273
//                                   (MultiModalGuider.calculate), reached through
//                                   ltx-pipelines utils/denoisers.py:188-203
//                                   (FactoryGuidedDenoiser)
//   the x0 wrapper on each pass <-  ltx-core model/transformer/model.py:590-604
//                                   (X0Model.forward), which is what
//                                   ltx-pipelines utils/blocks.py:480-482 builds
//                                   and hands the denoiser
//   the STG pass                <-  ltx-core model/transformer/attention.py:552-577
//   the audio latent shape      <-  ltx-core types.py:164-200
//                                   (AudioLatentShape.from_video_pixel_shape)
//   the schedule                <-  ltx-core components/schedulers.py:21-57,
//                                   hard-coded as LTX2Scheduler() at
//                                   t2a_one_stage.py:67
//
// ─── WHY IT IS ITS OWN TRANSLATION UNIT ──────────────────────────────────────
//
// Upstream's T2A is its own FILE with its own `__call__`, and AGENTS.md
// §"Shared seams" requires mirroring that structure. It is reached only through
// `Ltx2VideoEngine::Generate`, it owns no weights, and every numeric it uses is
// an already-gated brick — so it is a composition, not a second path.
//
// What it deliberately does NOT do is thread an `is_t2a` flag through the joint
// phase driver in `ltx2_video.cpp`. That function is 1900 lines and roughly a
// third of it constructs a video stream that does not exist here; nine new
// branches inside it would be nine chances to leave one behind, and a missed one
// renders.
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY IF GUESSED ───────────────────────────
//
// 1. `video = nullptr`, NOT `video->enabled = false`. Upstream's predicate is
//    `run_v2a = run_ax and (video is not None and vx.numel() > 0)`
//    (transformer.py:269) — it tests PRESENCE. A disabled-but-present video
//    stream still feeds video->audio cross attention from a latent T2A never
//    meant to exist, and still returns a finished, playable waveform.
//
// 2. THE GUIDER IS NOT OPTIONAL HERE. `distilled_two_stage` builds a
//    `SimpleDenoiser` upstream and this engine's joint loop mirrors that with one
//    forward per step. T2A builds a `FactoryGuidedDenoiser`
//    (t2a_one_stage.py:154-161) whose CLI defaults are `cfg_scale=7.0` and
//    `stg_scale=1.0` (utils/constants.py:58-66 through :118), so
//    `do_unconditional_generation` and `do_perturbed_generation` are both TRUE
//    (guiders.py:275-281) and the default path is THREE forwards per step. A
//    single-forward T2A produces audio of exactly the right length on a
//    trajectory the model was not asked for.
//
// 3. `modality_scale` IS PINNED TO 1.0, and the pin is upstream's, at the CLI
//    layer: "Audio-only generation has no video modality, so the video->audio
//    (v2a) cross-modal guidance is meaningless here. 1.0 disables it"
//    (t2a_one_stage.py:200-202). The params table's own value is 3.0, so
//    inheriting it would turn on a fourth forward against a modality that is not
//    there.
//
// 4. THE GUIDER COMBINES X0, NOT VELOCITY (#1039). Upstream builds the
//    denoiser's transformer as `X0Model(...)` (ltx-pipelines
//    utils/blocks.py:480-482), so every pass `_guided_denoise` hands
//    `MultiModalGuider.calculate` has ALREADY been converted with
//    `to_denoised(latent, v, timesteps)` (model.py:590-604, `to_denoised` at
//    ltx-core utils.py:39-52) before it is combined
//    (utils/denoisers.py:188-203).
//
//    Combining raw velocities and converting once afterwards is the SAME
//    FUNCTION only while `rescale_scale == 0`, because `calculate`'s linear
//    terms are invariant under `x0 = latent - sigma*v`. The rescale branch is
//    not invariant: upstream computes `factor` from `std(x0_cond)/std(x0_pred)`
//    and scales the whole x0, giving `factor*(latent - sigma*v)`; scaling the
//    velocity instead gives `latent - sigma*factor*v`. The two differ by
//    `(factor - 1) * latent` — zero only where the latent is zero, which on
//    this path it never is (the state IS the unit-variance noise, item above).
//    `rescale_scale = 0.7` is the shipped T2A default (utils/constants.py:63,
//    utils/args.py:1101-1106), so this is the DEFAULT arm rather than an
//    exotic one, and nothing about the rendered waveform separates the two.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_audio_vae.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vt/device.h"

namespace vllm {

// Everything `T2AOneStagePipeline.__call__` reads, as borrowed pointers. Nothing
// here is owned: the engine holds the weights for its whole lifetime and this
// call runs inside its mutex.
struct Ltx2T2aRequest {
  vt::Device device;
  vt::DType compute_dtype = vt::DType::kF32;
  const Ltx2DitParams* dit_params = nullptr;
  const Ltx2DitWeights* dit_weights = nullptr;

  // The AUDIO conditioning rows, `[context_tokens, audio_cross_attention_dim]`.
  // `negative_context` is `ctx_n.audio_encoding` (t2a_one_stage.py:135) and is
  // null when the caller has no negative conditioning — which is a REFUSAL when
  // the guider asks for an unconditional pass, never a silent drop to
  // `uncond = 0`.
  const float* context = nullptr;
  const float* negative_context = nullptr;
  int64_t context_tokens = 0;

  int64_t num_frames = 0;
  double frame_rate = 0.0;
  int64_t steps = 0;  // <= 0 => the recipe's own count

  // `GaussianNoiser(generator=torch.Generator(...).manual_seed(seed))`
  // (t2a_one_stage.py:124-125). Supplied by the CALLER rather than constructed
  // here: the engine already owns the one Gaussian source in this tree, and a
  // second generator seeded the same way would be a parallel path whose
  // agreement with the first nothing gates.
  Ltx2NoiseStream* noise = nullptr;

  Ltx2MultiModalGuiderParams guidance;

  const Ltx2AudioDecoderConfig* audio_cfg = nullptr;
  const Ltx2VaeWeights* audio_weights = nullptr;
  const Ltx2VocoderBweConfig* vocoder_cfg = nullptr;
  const Ltx2VaeWeights* vocoder_weights = nullptr;
};

// The rendered soundtrack, plus the observability the render itself cannot be
// inspected for.
//
// EVERY COUNTER BELOW IS INCREMENTED AT THE FORWARD, not derived from the params
// that were supposed to drive it. A field written from `guidance.cfg_scale`
// would report a healthy uncond count on a build that computed the params and
// then ran one forward — which is the exact instrument failure
// `Ltx2ConditioningTrace::audio_frozen` already paid for on this campaign.
struct Ltx2T2aResult {
  // [channels, samples_per_channel], the vocoder's own layout.
  std::vector<float> waveform;
  int64_t channels = 0;
  int64_t samples_per_channel = 0;
  int64_t sample_rate = 0;

  int64_t audio_tokens = 0;
  int64_t latent_frames = 0;

  // Forwards actually issued, by arm.
  int64_t cond_forwards = 0;
  int64_t uncond_forwards = 0;
  int64_t perturbed_forwards = 0;

  // TRUE if any forward was handed a video stream. It must be FALSE on every
  // T2A render, and it is recorded rather than asserted in prose because
  // §"THE THREE THINGS" item 1 is invisible in the output: a run that passed a
  // present-but-disabled video stream produces a waveform of exactly the right
  // length, the right channel count and the right sample rate.
  bool video_stream_present = false;

  // The blocks the STG pass actually perturbed, read off the vector handed to
  // the forward. A count alone cannot tell "perturbed block 1" from "perturbed
  // block 0", and `stg_blocks` is what decides which.
  std::vector<int64_t> perturbed_blocks;

  // FNV-1a over the final audio latent's raw f32 bytes, and its max|x|. The
  // digest detects CHANGE; the absmax is the lower bound a digest cannot make,
  // because a latent that collapsed to zeros has a perfectly stable digest.
  uint64_t latent_digest = 0;
  double latent_absmax = 0.0;

  // EVERYTHING STEP 0 PRODUCED, in the order it produced it: the sampler's
  // input, EVERY GUIDANCE PASS as a (raw velocity, x0 prediction) pair, the
  // guider's result, and the latent the Euler step wrote. They exist because
  // #1039 is invisible in every other field here — combining the guidance passes
  // in VELOCITY space and converting once afterwards produces a waveform of the
  // right length, the right channel count, the right sample rate and a perfectly
  // healthy forward count — and together they make the question decidable by
  // arithmetic rather than by magnitude:
  //
  //     first_step_cond      == first_step_latent - sigma * first_step_velocity
  //     first_step_uncond    == first_step_latent - sigma * first_step_uncond_velocity
  //     first_step_perturbed == first_step_latent - sigma * first_step_perturbed_velocity
  //
  // holds when the guider is handed X0 PREDICTIONS, as `X0Model.forward` does
  // (model.py:590-604, over the `X0Model(...)` that utils/blocks.py:480-482
  // builds), and fails when it is handed the velocities. `first_step_cond` is
  // also upstream's own `DenoisedLatentResult.cond` (utils/denoisers.py:206),
  // rather than a field invented for a test.
  //
  // ONE PAIR PER ARM, AND NOT ONLY THE CONDITIONAL ONE. The default T2A arm runs
  // THREE forwards per step (header item 2), and a build that converts the
  // conditional pass and leaves either of the other two in velocity space
  // renders a different waveform through a guider whose cond term is impeccable.
  // A single recorded pair holds the claim "`to_denoised` on the way out of the
  // forward" for one third of the passes it is made about; the review that found
  // #1039 mutated exactly those other two arms and the gate stayed green.
  //
  // `first_step_next_latent` is what `Ltx2EulerStep` WROTE, and it is here so
  // that what the sampler CONSUMED is checkable rather than assumed:
  //
  //     first_step_next_latent
  //         == latent + (latent - first_step_denoised)/sigma * (sigma_next - sigma)
  //
  // A second `ToDenoised` applied to the guider's output on the way into the
  // step — the residue a partial #1039 repair leaves behind — moves this and
  // nothing else.
  //
  // STEP 0 SPECIFICALLY, because it is the one step whose inputs do not depend
  // on any earlier step, so two renders that differ only in a guider parameter
  // share a bit-identical step-0 latent and bit-identical DiT passes.
  //
  // The uncond and perturbed pairs stay EMPTY when the guider does not ask for
  // that arm (`cfg_scale == 1.0`, `stg_scale == 0.0`), because the forward did
  // not run. An empty vector is the honest record of a pass that never happened;
  // a zero-filled one of the right length would be indistinguishable from a
  // forward that returned zeros.
  std::vector<float> first_step_latent;
  std::vector<float> first_step_velocity;
  std::vector<float> first_step_cond;
  std::vector<float> first_step_uncond_velocity;
  std::vector<float> first_step_uncond;
  std::vector<float> first_step_perturbed_velocity;
  std::vector<float> first_step_perturbed;
  std::vector<float> first_step_denoised;
  std::vector<float> first_step_next_latent;
  double first_step_sigma = 0.0;
};

// `T2AOneStagePipeline.__call__` (t2a_one_stage.py:109-172). Throws
// std::runtime_error naming the problem; never renders a fallback.
Ltx2T2aResult Ltx2T2aGenerate(const Ltx2T2aRequest& req);

// `stg_blocks` -> the per-block vector `Ltx2DitForward` takes
// (guidance/perturbations.py:19-33, `blocks is None` meaning ALL). Exposed so a
// test can pin the mapping without reaching into the pipeline, and so the one
// place that turns a block LIST into a block MASK is named.
//
// A block index outside `[0, num_layers)` is REFUSED. Upstream indexes
// `self.transformer_blocks` by it and would raise; silently ignoring it would
// run an unperturbed pass and call it STG.
std::vector<uint8_t> Ltx2StgBlockMask(const std::vector<int64_t>& stg_blocks, int64_t num_layers);

}  // namespace vllm
