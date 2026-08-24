// LTX-2.5 denoisers — `ltx-pipelines/utils/denoisers.py` @ Lightricks/LTX-2
// fd4ded7f, in its own translation unit because upstream has its own file.
//
// Row LTX25-GUIDED-VIDEO, issue
// https://github.com/mudler/vllm.cpp/issues/1092. Spec
// .agents/specs/ltx25-guided-video.md.
//
// ── WHAT THIS FILE IS FOR ──────────────────────────────────────────────────
//
// Upstream has three denoisers (`SimpleDenoiser`, `GuidedDenoiser`,
// `FactoryGuidedDenoiser`) and they share ONE function: `_guided_denoise`
// (denoisers.py:61-211). That function is what this file ports. It is the piece
// four unported pipelines are each blocked on — `a2vid_two_stage.py:230`,
// `ti2vid_two_stages.py:248`, `ti2vid_two_stages_hq.py:271`,
// `keyframe_interpolation.py:232` — and it is what a `pipeline_kind = one_stage`
// render here was missing entirely: `ti2vid_one_stage.py:221-226` builds a
// `FactoryGuidedDenoiser` and this port ran one unguided forward per step.
//
// ── THE FOUR THINGS THAT ARE EASY TO GET WRONG AND STILL RENDER ────────────
//
// 1. THE SPACE. The transformer this seam is handed is upstream's `X0Model`
//    (built at blocks.py:480-482, forward at model.py:590-604), so every pass it
//    returns is ALREADY `latent - sigma * velocity` and the guider combines
//    denoised tensors. Combining velocities and converting once afterwards is a
//    DIFFERENT function whenever `rescale_scale != 0` (guiders.py:268-271), and
//    it is 0.7 on every video row of the params table. That defect shipped on
//    the audio arm of this tree and is #1039. The conversion therefore lives in
//    the caller's `Ltx2X0Model`, which is where upstream puts it, and the seam
//    combines `Ltx2X0Outputs::video` / `::audio`.
//
//    THAT IS CALLER DISCIPLINE AND NOT A TYPE GUARANTEE, and this comment
//    claimed the stronger thing until 2026-08-17. `Ltx2X0Outputs` carries the
//    raw velocity beside the denoised prediction (below), so a lambda that fills
//    `video` with what belongs in `video_velocity` type-checks and renders.
//    Nothing in the signature can stop it; the four per-arm invariants in
//    `test_ltx2_video` do, and mutations M1-M4 — one per arm, each handing the
//    seam a velocity — are red against them. A structural claim a type does not
//    enforce is worth less than a gate that catches the substitution, so the
//    gate is where this is argued.
//
// 2. ONE PASS LIST, TWO GUIDERS. `_guided_denoise` takes the UNION of what the
//    two guiders want — one `uncond` pass if either asks (`:102-109`), one `ptb`
//    pass carrying both modalities' `stg_blocks` (`:111-119`), one `mod` pass if
//    either asks (`:121-137`) — and then combines each modality with its OWN
//    guider over the same splits (`:203-204`). Running a per-modality pass list
//    instead would issue up to six forwards where upstream issues four, and
//    would hand the audio stream a different video state to cross-attend to on
//    the video-only passes. Both renders finish.
//
// 3. `post_process_latent` IS NOT PART OF THE DENOISER. It is applied by the
//    LOOP, to the guider's OUTPUT (`samplers.py:35`, `:484`), not to each arm on
//    the way out of the forward. Pinning the conditioned tokens per arm makes
//    every arm agree on those tokens, which silently zeroes the guidance delta
//    exactly where a keyframe or a reference clip is conditioning. This seam
//    therefore returns the raw guided prediction and the caller post-processes.
//
// 4. A SKIPPED STEP RUNS NO FORWARD. When both guiders skip, upstream returns
//    the PREVIOUS step's denoised pair (`:87-90`) rather than running the
//    conditional pass and using it. `skip_step` is 0 in every params table, so
//    this is reachable only through an explicit request; it is ported because
//    the request surface exposes it.
//
// ── WHAT IS NOT HERE ───────────────────────────────────────────────────────
//
// The BATCHING. Upstream concatenates the passes along the batch axis and issues
// ONE transformer call (`:141-186`); this issues one call per pass. That is a
// throughput difference and not a numerical one at `batch == 1`, and it is the
// same adaptation `Ltx2DitPerturbation` already records: every path in this port
// runs `Ltx2ModalityInput::batch == 1`.

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_pipeline.h"

namespace vllm {

// What one pass of the `X0Model` returns. `Ltx2DitOutputs` carries the raw
// velocity; this carries the denoised prediction AND the velocity it came from,
// because "which space was this combined in" is an arithmetic question between
// three tensors and cannot be answered from the denoised one alone. The velocity
// is what the trace records per arm and what the gate's per-arm invariant
// `x0 == latent - sigma * velocity` is checked against.
struct Ltx2X0Outputs {
  std::vector<float> video;           // `to_denoised(video.latent, vx, video.timesteps)`
  std::vector<float> audio;           // `to_denoised(audio.latent, ax, audio.timesteps)`
  std::vector<float> video_velocity;  // the DiT's own output, before the conversion
  std::vector<float> audio_velocity;
};

// `X0Model` (model.py:590-604), supplied by the caller.
//
// It is a callable and not a (params, weights) pair on purpose. Upstream's
// `_guided_denoise(transformer, ...)` takes the model the same way, the host and
// the device forward are two lambdas over one seam, and the four pipelines that
// will use this next supply their own conditioning without this file learning
// anything about keyframes, reference clips or two-stage schedules.
//
// EITHER STREAM MAY BE NULL, which is upstream's absent modality. `perturbations`
// is null on the passes that have none, which is upstream's
// `PerturbationConfig.empty()` reaching `model.py:509-511`.
using Ltx2X0Model = std::function<Ltx2X0Outputs(const Ltx2ModalityInput* video,
                                                const Ltx2ModalityInput* audio,
                                                const Ltx2DitPerturbation* perturbations)>;

// The pass names of `_guided_denoise` (`:100-137`), in the order it appends
// them. The order is not cosmetic: it is the batch order upstream splits back
// out at `:188-190`, and it is the order the perturbation config is built in.
enum class Ltx2DenoisePass {
  kCond = 0,
  kUncond = 1,
  kPerturbed = 2,
  kModality = 3,
};
inline constexpr int64_t kLtx2DenoisePassCount = 4;

struct Ltx2GuidedDenoiseInputs {
  // The two streams, exactly as they would be handed to the forward for the
  // conditional pass. The seam copies them per pass and overrides `context` and
  // `enabled`; it never touches the latent, the timesteps or the positions.
  const Ltx2ModalityInput* video = nullptr;
  const Ltx2ModalityInput* audio = nullptr;

  // `guider.negative_context` (guiders.py:236). A null is upstream's
  // `negative_context is None`, whose branch at `:107-108` falls back to the
  // POSITIVE context rather than refusing — and which `:104-106` refuses when
  // that modality's guider is the one asking. Both are mirrored.
  const float* video_negative_context = nullptr;
  const float* audio_negative_context = nullptr;

  Ltx2MultiModalGuiderParams video_guider;
  Ltx2MultiModalGuiderParams audio_guider;

  // `transformer.num_blocks` (denoisers.py:180). Needed to size the perturbation
  // masks, and refused when it disagrees with the DiT the caller's lambda drives
  // — a mask built for another block count perturbs a prefix and renders.
  int64_t num_blocks = 0;

  int64_t step_index = 0;
  // `force_uncond_pass` (declared at denoisers.py:74, read at `:102-103`). NO
  // upstream caller enables it at `fd4ded7f`. It is a CFG++ affordance that runs
  // the uncond pass when `cfg_scale` is 1.0 and the uncond prediction is still
  // needed for the ordinary differential equation derivative, described at
  // `packages/ltx-pipelines/CLAUDE.md:76`. Upstream declares the flag and
  // threads it through `GuidedDenoiser` (`:267`, stored at `:273`, forwarded at
  // `:297`) and `FactoryGuidedDenoiser` (`:313`, `:319`, `:357`), and all seven
  // construction sites in the upstream tree take the `False` default,
  // `RetakePipeline` (retake.py:305-310) included. This field mirrors that
  // unused plumbing, so nothing in this tree assigns it either.
  bool force_uncond_pass = false;

  // `_last_denoised_video` / `_last_denoised_audio` (denoisers.py:274-275). Null
  // on the first step. A skipped step with nothing to reuse is refused rather
  // than reading an empty vector.
  const std::vector<float>* last_denoised_video = nullptr;
  const std::vector<float>* last_denoised_audio = nullptr;
};

// Everything one call produced. The per-pass tensors are kept because a gate
// that can only see the combination cannot tell which arm was converted in which
// space, and because #1039's first gate covered one arm out of three and three
// mutations survived it.
struct Ltx2GuidedDenoiseResult {
  // `DenoisedLatentResult.denoised` (utils/types.py), per modality. Empty when
  // that modality was absent.
  std::vector<float> video_denoised;
  std::vector<float> audio_denoised;
  bool video_skipped = false;
  bool audio_skipped = false;

  // Which passes ran, and what each returned. Indexed by `Ltx2DenoisePass`.
  bool pass_ran[kLtx2DenoisePassCount] = {false, false, false, false};
  std::vector<float> video_pass[kLtx2DenoisePassCount];           // x0
  std::vector<float> audio_pass[kLtx2DenoisePassCount];
  std::vector<float> video_pass_velocity[kLtx2DenoisePassCount];  // raw
  std::vector<float> audio_pass_velocity[kLtx2DenoisePassCount];

  // The blocks the perturbed pass actually asked the DiT to skip, read off the
  // mask that was handed over rather than copied from the guider params. A
  // config that is BUILT and not HANDED OVER is invisible in the params.
  std::vector<int64_t> perturbed_video_blocks;
  std::vector<int64_t> perturbed_audio_blocks;
  // Whether the isolated-modality pass reached the DiT with BOTH cross
  // directions off, observed at the call for the same reason.
  bool modality_pass_skipped_a2v = false;
  bool modality_pass_skipped_v2a = false;
};

// `_guided_denoise` (denoisers.py:61-211).
Ltx2GuidedDenoiseResult Ltx2GuidedDenoise(const Ltx2X0Model& transformer,
                                          const Ltx2GuidedDenoiseInputs& in);

}  // namespace vllm
