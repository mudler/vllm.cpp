// LTX-2.5 CONDITIONING ITEMS — what the VAE encoders' output is FOR.
//
// An encoded image, keyframe, reference video or reference audio is not
// conditioning yet: it becomes conditioning when its tokens are placed into the
// denoise state, with the right RoPE positions and the right denoise mask. That
// placement is upstream's `ConditioningItem.apply_to` family, and this is the
// port of it.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f, packages/ltx-core/src/ltx_core/
//   OURS                                <-  UPSTREAM
//   Ltx2CreateVideoLatentState          <-  tools.py:139-186 (VideoLatentTools)
//   Ltx2CreateAudioLatentState          <-  tools.py:246-279 (AudioLatentTools)
//   Ltx2ConditionVideoByLatentIndex     <-  conditioning/types/latent_cond.py:22-43
//   Ltx2ConditionVideoByKeyframe        <-  conditioning/types/keyframe_cond.py:39-90
//   Ltx2ConditionVideoByReference       <-  conditioning/types/reference_video_cond.py:46-108
//   Ltx2ConditionAudioByReference       <-  conditioning/types/reference_audio_cond.py:33-65
//
// The patchifiers and the coordinate maps are NOT re-implemented here: they are
// `Ltx2VideoPatchify`, `Ltx2VideoPatchBounds`, `Ltx2PixelCoords`,
// `Ltx2AudioPatchify` and `Ltx2AudioPatchTimings` from ltx2_pipeline.h, already
// gated by phase L5.
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * `Ltx2ConditionVideoByLatentIndex` LEAVES THE NOISY TENSOR ALONE. Only
//    `clean_latent` and `denoise_mask` are written (latent_cond.py:38-39); the
//    noiser then composes them as
//    `lerp(clean, lerp(latent, noise, ns), mask)` (components/noisers.py:31-34).
//    diffusers instead writes the clean tokens into the noisy tensor too
//    (pipeline_ltx2_condition.py:1002) and reaches the same answer ONLY at
//    `noise_scale == 1`. Copying diffusers here would be a silent divergence at
//    every other noise scale.
//  * THE DENOISE MASK IS `1 - strength`, NOT `strength`. Mask 0 means "keep the
//    clean value"; strength 1 means fully conditioned. Inverting it renders an
//    unconditioned clip that looks like the feature not working.
//  * A KEYFRAME'S CAUSAL FIX IS DISABLED UNLESS `frame_idx == 0`
//    (keyframe_cond.py:45-50). The fix exists because the causal encoder's FIRST
//    latent frame spans one pixel frame; a keyframe placed later has no such
//    frame, so applying the fix anyway shifts it by up to `factors.time - 1`.
//  * THE REFERENCE ITEM'S TWO SCALE FACTORS EACH GUARD ON `!= 1`
//    (reference_video_cond.py:74, 80). Applying the temporal translation with
//    `temporal_scale_factor == 1` subtracts zero — harmless — but applying the
//    CLAMP unconditionally is not, and applying the spatial multiply when it is 1
//    is a no-op that hides a wrong branch.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"  // Ltx2LatentVolume

namespace vllm {

// LatentState (types.py:251-287) at batch 1, carrying only the fields a
// conditioning item touches. `attention_mask` is deliberately absent: every item
// ported here passes `attention_mask=None` and there is no pre-existing mask, so
// `update_attention_mask` returns None (conditioning/mask_utils.py:110-140). An
// item that DID need one would have to grow this struct rather than silently
// dropping it, which is why the omission is stated here.
struct Ltx2LatentState {
  int64_t tokens = 0;
  int64_t width = 0;     // channels per token
  int64_t pos_dims = 0;  // 3 for video (frame, height, width), 1 for audio
  std::vector<float> latent;     // [tokens, width] — the NOISY tensor
  std::vector<float> clean;      // [tokens, width]
  std::vector<float> mask;       // [tokens] — the denoise mask, 1 = fully denoised
  std::vector<float> positions;  // [pos_dims, tokens, 2] — [start, end)
};

// VideoLatentTools.create_initial_state (tools.py:139-186) at batch 1.
// `initial_latent` may be null, which is upstream's zero fill. `out_keyframes_mask`
// receives `_first_frame_keyframes_mask` (tools.py:186-197) — the target's FIRST
// latent frame marked, unconditionally, because the causal encoder makes it span a
// single pixel frame. May be null.
Ltx2LatentState Ltx2CreateVideoLatentState(const Ltx2VideoLatentShape& shape, int64_t patch_size,
                                           const Ltx2ScaleFactors& factors, double fps,
                                           bool causal_fix, const float* initial_latent = nullptr,
                                           std::vector<float>* out_keyframes_mask = nullptr);

// AudioLatentTools.create_initial_state (tools.py:246-279) at batch 1. The audio
// positions are the patch grid bounds THEMSELVES, in seconds — there is no
// `get_pixel_coords` and no division by fps on this side.
Ltx2LatentState Ltx2CreateAudioLatentState(const Ltx2AudioLatentShape& shape,
                                           const Ltx2AudioPatchifierParams& params,
                                           const float* initial_latent = nullptr);

// VideoConditionByLatentIndex (latent_cond.py:22-43): REPLACE the clean tokens of
// latent frame `latent_idx` with an encoded image or clip. This is the
// image-to-video / first-frame path, and `latent_idx = 0` is what
// `combined_image_conditionings` selects for `frame_idx == 0`
// (ltx-pipelines/utils/helpers.py:295-300).
//
// `conditioning` must match the target's batch, channels, height and width;
// upstream raises `ConditioningError` otherwise and so does this.
void Ltx2ConditionVideoByLatentIndex(Ltx2LatentState* state, const Ltx2VideoLatentShape& target,
                                     int64_t patch_size, const Ltx2LatentVolume& conditioning,
                                     double strength, int64_t latent_idx);

// VideoConditionByKeyframeIndex (keyframe_cond.py:39-90): APPEND keyframe tokens
// whose positions are offset to pixel frame `frame_idx`. The appended tokens are
// zero in the noisy tensor and the keyframe in the clean one.
void Ltx2ConditionVideoByKeyframe(Ltx2LatentState* state, const Ltx2LatentVolume& keyframes,
                                  int64_t patch_size, const Ltx2ScaleFactors& factors, double fps,
                                  int64_t frame_idx, double strength, int64_t num_pixel_frames,
                                  bool causal_fix);

// VideoConditionByReferenceLatent (reference_video_cond.py:46-108): APPEND a
// reference video's tokens, translated into the target's frame. `downscale_factor`
// is the target/reference SPATIAL ratio and `temporal_scale_factor` the temporal
// one; both must match what the IC-LoRA was trained with, which upstream stores in
// the LoRA metadata this project does not yet read — so a caller that guesses them
// gets a plausible, wrongly-placed reference.
void Ltx2ConditionVideoByReference(Ltx2LatentState* state, const Ltx2LatentVolume& reference,
                                   int64_t patch_size, const Ltx2ScaleFactors& factors, double fps,
                                   int64_t downscale_factor, int64_t temporal_scale_factor,
                                   double strength, bool causal_fix);

// AudioConditionByReferenceLatent (reference_audio_cond.py:33-65): APPEND already
// patchified reference-audio tokens with their own timings. Takes the patchified
// form because the caller holds the encoder's `[channels, frames, mel_bins]` output
// and `Ltx2AudioPatchify` is the gated way to flatten it.
void Ltx2ConditionAudioByReference(Ltx2LatentState* state, const std::vector<float>& tokens,
                                   int64_t token_count, int64_t width,
                                   const std::vector<float>& positions, double strength);

}  // namespace vllm
