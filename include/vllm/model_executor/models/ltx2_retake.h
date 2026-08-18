// LTX-2.5 RETAKE — regenerate a chosen time window of an existing clip, keeping
// everything outside it. Row LTX25-RETAKE (#924), spec
// .agents/specs/ltx25-retake.md.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
//   OURS                             <-  UPSTREAM
//   Ltx2TemporalRegionMaskVideo      <-  ltx-core conditioning/types/
//                                        noise_mask_cond.py:30-35,38-44
//   Ltx2TemporalRegionMaskAudio      <-  ltx-core conditioning/types/
//                                        noise_mask_cond.py:27-29,38-44
//   Ltx2ConformLatentLength          <-  ltx-pipelines utils/helpers.py:149-162
//   Ltx2RetakeAssertWindow           <-  ltx-pipelines retake.py:211-212
//   Ltx2RetakeAssertSourceGeometry   <-  ltx-pipelines retake.py:344-353
//   Ltx2RetakePlanModalities         <-  ltx-pipelines retake.py:268-283
//
// The pipeline that drives these is `RetakePipeline` (retake.py:53, called at
// :355): read the source clip, encode it through the video VAE, conform the
// latent's length, seed the video stream with it, write the denoise mask from
// the time window, denoise, and blend the untouched region back.
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * `causal_fix` DEFAULTS THE OTHER WAY AT THE CALL SITE. `get_pixel_coords`
//    declares `causal_fix: bool = False` (patchifiers.py:140), and
//    `TemporalRegionMask` calls it with `getattr(latent_tools, "causal_fix",
//    True)` (noise_mask_cond.py:33). Reading the function's default instead of
//    the call site's shifts every video boundary by `time - 1` = seven pixel
//    frames, which renders a clip whose regenerated window is in the wrong
//    place and is invisible to any shape or token check.
//  * THE TWO SIDES USE DIFFERENT COORDINATES. Video bounds come out of the
//    patchifier in LATENT frames and have to go through `get_pixel_coords` and
//    then a division by fps (noise_mask_cond.py:31-35). Audio bounds come out
//    ALREADY IN SECONDS (:27-29, via patchifiers.py:216-249) and must not be
//    scaled again. One shared code path here silently mistimes one modality.
//  * THE TEST IS OVERLAP, NOT CONTAINMENT. `in_region = (t_end > start_time) &
//    (t_start < end_time)` (noise_mask_cond.py:39) — half-open on both ends, so
//    a token STRADDLING either edge is inside. Containment would drop the
//    boundary tokens and leave a seam at each end of the regenerated window.
//  * THE MASK IS OVERWRITTEN, NOT COMBINED. `state.denoise_mask.copy_(mask_val)`
//    (noise_mask_cond.py:44) replaces whatever the mask held. ANDing it with an
//    earlier conditioning's mask would keep that conditioning's zeros and shrink
//    the regenerated region without saying so.
//
// ─── AND ONE THAT IS NOT SILENT, BUT IS SURPRISING ───────────────────────────
// `Ltx2ConformLatentLength` truncates OR ZERO-PADS. Its sibling on the
// audio-to-video path truncates and NEVER pads (a2vid_two_stage.py:202), and a
// short take there is an ERROR rather than a short latent because
// `create_initial_state` asserts the shape (ltx-core/tools.py:146-148) — which
// row LTX25-A2V-AUDIO-INPUT mirrored as a refusal, reasoned out at
// src/vllm/model_executor/models/ltx2_audio_input.cpp:209. The two polarities
// are both upstream's, on two callers, and one shared helper would have to pick
// one and be wrong for the other. That is why this is a second function with a
// different name rather than a parameter on the first.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_pipeline.h"  // shapes, scale factors, patchifier params

namespace vllm {

// ---------------------------------------------------------------------------
// TemporalRegionMask (noise_mask_cond.py:9-45)
// ---------------------------------------------------------------------------

// The VIDEO branch (noise_mask_cond.py:30-35). Returns one value per token, in
// the patchifier's `(f, h, w)` order, 1.0 inside the window and 0.0 outside.
//
// `causal_fix` is exposed rather than fixed at `true` because the two upstream
// defaults disagree (see the header banner); the engine passes the CALL SITE's
// default, and the parameter exists so a test can tell the two apart.
std::vector<float> Ltx2TemporalRegionMaskVideo(const Ltx2VideoLatentShape& shape,
                                               int64_t patch_size,
                                               const Ltx2ScaleFactors& factors, double fps,
                                               double start_time, double end_time,
                                               bool causal_fix);

// The AUDIO branch (noise_mask_cond.py:27-29). The patchifier's bounds are
// already seconds, so there is no scale factor and no fps here — passing one
// would be the mistiming the banner names.
std::vector<float> Ltx2TemporalRegionMaskAudio(const Ltx2AudioLatentShape& shape,
                                               const Ltx2AudioPatchifierParams& params,
                                               double start_time, double end_time);

// ---------------------------------------------------------------------------
// _conform_latent_length (utils/helpers.py:149-162)
// ---------------------------------------------------------------------------

// `latent` is the UNPATCHIFIED volume laid out channel-major as
// [channels][frames][per_frame], which is what `Ltx2VideoPatchify` and
// `Ltx2AudioPatchify` consume. `per_frame` is `height * width` for video and
// `mel_bins` for audio — the product of every axis after the frame axis, which
// is upstream's `shape[3:]` once `shape[2]` is the frame axis it slices.
//
// Longer than `expected_frames`: keep the LEADING frames (`latent[:, :, :n]`,
// helpers.py:152). Shorter: append zeros (`torch.cat([latent, pad], dim=2)`,
// helpers.py:156-161). Equal: returned unchanged.
std::vector<float> Ltx2ConformLatentLength(const std::vector<float>& latent, int64_t channels,
                                           int64_t frames, int64_t per_frame,
                                           int64_t expected_frames);

// ---------------------------------------------------------------------------
// The validations, at the layer upstream runs them from
// ---------------------------------------------------------------------------

// retake.py:211-212, the first statement of `__call__`. Upstream repeats it at
// the CLI stage (:340-341) with a shorter message; the pipeline's own message is
// the one with both values in it, so it is the one mirrored.
void Ltx2RetakeAssertWindow(double start_time, double end_time);

// retake.py:344-353 — the CLI-stage geometry validation, which runs BEFORE the
// pipeline is constructed so a bad clip costs no model load.
//
// `time_factor` is `SpatioTemporalScaleFactors.default().time` (retake.py:344,
// ltx-core types.py:31-33). The frame refusal NAMES THE SNAPPED VALUE
// (retake.py:348-351); a refusal that does not say what would have worked costs
// a round trip. The resolution refusal names both axes (:352-353).
void Ltx2RetakeAssertSourceGeometry(int64_t frames, int64_t height, int64_t width,
                                    int64_t time_factor);

// ---------------------------------------------------------------------------
// The source clip (utils/helpers.py:197-220, media_io/decode.py:213-224)
// ---------------------------------------------------------------------------

// Upstream reads a container with PyAV (`av.open`, decode.py:226) and NO
// demuxer is vendored in this tree — `src/vllm/entrypoints/openai/video_api.cpp`
// says so where it defines the reference-video field. But upstream carries a
// SECOND ingestion arm for exactly that case: a directory of frames, whose frame
// rate cannot be read from a container and must be supplied
// (decode.py:213-215), and which has no audio stream at all
// (utils/helpers.py:261-262). This tree's `frame_%06d.ppm` directory is that
// arm; the container is refused by name at the engine.
struct Ltx2RetakeSourceGeometry {
  int64_t frames = 0;
  int64_t height = 0;
  int64_t width = 0;
};

// `get_videostream_metadata`'s folder branch (decode.py:216-224): count the
// frames, and take height and width from the FIRST one. Refuses an empty
// directory, and refuses a frame whose size differs from frame 0 — upstream
// reads one spec and applies it to the whole folder, so a ragged folder would
// silently reinterpret later frames.
Ltx2RetakeSourceGeometry Ltx2ProbeFrameDirectory(const std::string& dir);

// `video_preprocess` over the folder (utils/helpers.py:205-220,228). Returns
// [channels, frames, height, width] in [-1, 1] — the layout
// `Ltx2ConvVideoEncode` takes — by running each frame through the SAME
// decode -> resize-and-centre-crop -> `/127.5 - 1` chain the image-conditioning
// path uses, which is upstream's own note at helpers.py:202 ("Center-crop
// matches the SDR video_preprocess path used for mp4 inputs").
//
// CRF is 0 and not a parameter: `crf` is a knob on an IMAGE conditioning input
// (`ImageConditioningInput.crf`, blocks.py:977-983) and upstream's video
// ingestion path never applies one.
std::vector<float> Ltx2ReadFrameDirectory(const std::string& dir, int64_t height, int64_t width);

// ---------------------------------------------------------------------------
// The four-way modality plan (retake.py:268-283)
// ---------------------------------------------------------------------------

// `conditioned` means the modality gets a `TemporalRegionMask`; `frozen` means
// it is held at its initial latent for the whole run (ModalitySpec.frozen,
// utils/types.py:104-106).
struct Ltx2RetakePlan {
  bool video_conditioned = false;
  bool video_frozen = false;
  bool audio_conditioned = false;
  bool audio_frozen = false;
};

// BOTH audio predicates are conjunctions with `initial_audio_latent is not None`
// (retake.py:279 and :282) and neither video predicate is (:271, :274). So with
// no audio latent — which is every frame-folder source, because
// `audio_latent_from_file` returns None for one before opening anything
// (utils/helpers.py:261-262) — the audio stream is neither conditioned nor
// frozen and `regenerate_audio` has NO OBSERVABLE EFFECT. That asymmetry is
// upstream's, it is mirrored rather than repaired, and it is pinned by a test so
// the next reader finds it asserted instead of having to re-derive it.
Ltx2RetakePlan Ltx2RetakePlanModalities(bool regenerate_video, bool regenerate_audio,
                                        bool has_audio_latent);

}  // namespace vllm
