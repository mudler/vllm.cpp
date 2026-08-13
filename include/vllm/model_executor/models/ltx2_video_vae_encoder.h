// LTX-2.5 CONV VIDEO VAE — the ENCODER half, which phase L4 recorded as owed.
//
// Without it there is no way to turn an input image, keyframe or reference video
// into latents, so every conditioning mode except prompt-embeds is unreachable:
// `combined_image_conditionings` calls `video_encoder(image)` and hands the
// result to `VideoConditionByLatentIndex` / `VideoConditionByKeyframeIndex`
// (ltx-pipelines/utils/helpers.py:272-308).
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f, packages/ltx-core/src/ltx_core/
//   OURS                          <-  UPSTREAM
//   Ltx2ConvVideoEncode           <-  model/video_vae/video_vae.py:264-336
//   (block construction)          <-  model/video_vae/video_vae.py:39-145
//   (SpaceToDepthDownsample)      <-  model/video_vae/sampling.py:12-65
//   (CausalConv3d, strided)       <-  model/video_vae/convolution.py:266-313
//   (ResnetBlock3D/UNetMidBlock3D)<-  model/video_vae/resnet.py:12-277
//   (AttnBlock3D)                 <-  model/video_vae/attention.py:11-69
//   (patchify)                    <-  model/video_vae/ops.py:6-32
//   (per-channel statistics)      <-  model/video_vae/ops.py:63-84
//   Ltx2VideoTemporalScaleFactor  <-  types.py:35-53 (SpatioTemporalScaleFactors)
//   Ltx2ConvVideoEncoderConfig    <-  model/video_vae/model_configurator.py:37-70
//
// The primitives are SHARED with the decoder rather than copied: this encoder is
// compiled into ltx2_video_vae.cpp so it calls the very same `CausalConv3d`,
// `PixelNorm`, `ApplyNorm`, `ResnetBlock3d` and `AttnBlock3d` the decoder is
// gated on. A second copy of a causal pad is the duplicate that goes wrong
// quietly, because each copy keeps its own green gate.
//
// ─── THE FIVE THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * THE ENCODER'S DEFAULT SPATIAL PADDING IS `zeros`; THE DECODER'S IS
//    `reflect` (model_configurator.py:63-67 vs :90). They read the SAME
//    checkpoint key `spatial_padding_mode` on a flat CausalVideoAutoencoder
//    config, so they only diverge when the key is ABSENT — and then they diverge
//    silently, by a half-pixel border, in opposite directions.
//  * `SpaceToDepthDownsample` DUPLICATES FRAME 0 BEFORE the space-to-depth fold
//    whenever the temporal stride is 2 (sampling.py:39-40), and it does that on
//    the SKIP path as well as the conv path. Skipping it shifts the whole clip by
//    one latent frame while every shape still checks out.
//  * ITS SKIP CONNECTION IS A GROUP MEAN, NOT A SLICE (sampling.py:50-51): the
//    folded `in_channels * prod(stride)` channels are cut into `out_channels`
//    contiguous groups of `group_size` and averaged. Taking the first channel of
//    each group instead produces a plausible latent.
//  * `patchify` DECOMPOSES SPATIAL AXES AS `(h q) (w r) -> (c p r q)` — r (WIDTH)
//    is the OUTER factor and q (HEIGHT) the inner one (ops.py:20-28). This is the
//    exact inverse of the decoder's `unpatchify`, and swapping r and q transposes
//    every patch.
//  * A STRIDED CausalConv3d STILL PREPENDS `k_t - 1` COPIES OF FRAME 0, not
//    `k_t - stride` (convolution.py:305-307). The padding is decided before the
//    stride is applied, so the output frame count is
//    `(T + k_t - 1 - k_t) / stride + 1`.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// f32 throughout, for exactly the reason ltx2_video_vae.cpp:20-44 gives for the
// decoder: this is the CPU REFERENCE arm and upstream instead runs in the
// CHECKPOINT's dtype. No gate here can catch a dtype that is merely too WIDE,
// because the generator casts every upstream parameter to f32 and the oracle
// therefore runs f32 too. The production arm is owed with the decoder's.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"    // Ltx2VaeWeights
#include "vllm/model_executor/models/ltx2_upsampler.h"    // Ltx2LatentVolume
#include "vllm/model_executor/models/ltx2_video_vae.h"    // Ltx2NormLayer, Ltx2PaddingMode

namespace vllm {

// video_vae/enums.py:9-13. What the last conv emits ALONGSIDE the means, and it
// changes the channel arithmetic rather than just the bookkeeping:
//   kPerChannel  conv_out emits 2 * out_channels; means are the first half.
//   kUniform     conv_out emits out_channels + 1; the single trailing logvar
//                channel is REPEATED out_channels times before the split.
//   kConstant    conv_out emits out_channels + 1; the trailing channel is
//                DISCARDED and replaced by a constant -30 block.
//   kNone        conv_out emits out_channels — and upstream then chunks that in
//                two, leaving HALF as many mean channels as the per-channel
//                statistics carry. See Ltx2ConvVideoEncode: it is refused by
//                name rather than reproduced, because upstream itself raises.
enum class Ltx2LogVarianceType { kPerChannel, kUniform, kConstant, kNone };

// `approx_ln_0` (video_vae.py:328) — the constant log-variance the kConstant arm
// substitutes, described upstream as "the minimal clamp value in
// DiagonalGaussianDistribution objects". Named so it can be pinned: it is a
// member of the invisible-constant class, because a golden that only ever reads
// the MEANS half of the split can never see it. The gate reads the constant
// directly instead.
inline constexpr double kLtx2EncoderApproxLnZero = -30.0;

// One entry in `encoder_blocks`, in CHECKPOINT order — the encoder walks it
// FORWARD, which is what makes it the mirror of the decoder's reversed walk.
// `multiplier` 0 means "the upstream default for this block kind": 2 for
// `res_x_y`, `compress_all_x_y`, `compress_all_res`, `compress_space_res` and
// `compress_time_res` (video_vae.py:62, 103, 114, 123, 132); the plain
// `compress_time` / `compress_space` / `compress_all` convolutions take NO
// multiplier at all and keep `in_channels` (video_vae.py:72-101).
struct Ltx2VideoEncoderBlock {
  std::string name;
  int64_t num_layers = 1;
  int64_t multiplier = 0;
};

struct Ltx2ConvVideoEncoderConfig {
  // Defaults mirror `_prepare_video_encoder_kwargs`
  // (video_vae/model_configurator.py:55-69).
  int64_t in_channels = 3;
  // The LATENT width. On a flat CausalVideoAutoencoder config this is
  // `latent_channels`, NOT the top-level `out_channels`, which is the decoder's
  // RGB count (model_configurator.py:41-43) — reading the wrong one builds an
  // encoder with 3 latent channels that still runs.
  int64_t out_channels = 128;
  std::vector<Ltx2VideoEncoderBlock> encoder_blocks;
  int64_t patch_size = 4;
  Ltx2NormLayer norm_layer = Ltx2NormLayer::kPixelNorm;
  Ltx2LogVarianceType latent_log_var = Ltx2LogVarianceType::kUniform;
  // ZEROS is the ENCODER's default and it is NOT the decoder's `reflect`; see the
  // header note above.
  Ltx2PaddingMode spatial_padding_mode = Ltx2PaddingMode::kZeros;
  int64_t norm_num_groups = 32;  // VideoEncoder._DEFAULT_NORM_NUM_GROUPS = 32.
  // HARDCODED upstream: `_make_encoder_block` passes `resnet_eps=1e-6` /
  // `eps=1e-6` literally (video_vae.py:56, 66) and `conv_norm_out` takes
  // `eps=1e-6` (video_vae.py:240). It is a field here only so the gate can pin
  // it; there is no checkpoint key that moves it.
  //
  // And norm3 is the reason it is LIVE on a PixelNorm checkpoint here too, for
  // the identical reason it is on the decoder's `norm_eps`: `ResnetBlock3D`
  // builds `norm3 = nn.GroupNorm(num_groups=1, ..., eps=eps)` whenever
  // `in_channels != out_channels` (resnet.py:93-97) and applies it to the
  // residual (resnet.py:178), and `norm_layer` does not gate that. So every
  // `res_x_y` encoder block reads this value even though `conv_norm_out` and
  // `ApplyNorm` take their PixelNorm branches.
  //
  // Both halves route through ONE line in the port — ltx2_video_vae.cpp:1051,1056
  // reach :405, the same line the decoder reaches from :693,700 — but a SHARED
  // LINE IS NOT AN ARGUMENT FOR LIVENESS, and this file previously offered it as
  // one. :405 sits behind the `input.channels != out_channels` guard at :400, so
  // even entering ResnetBlock3d is not reaching it — `res_x` passes `x.channels`
  // as `out_channels` at :1051 and the guard is false. Encoder arm B does not
  // reach it at all: all four blocks it holds are plain strided CausalConv3d
  // (:1060-1068), so it never enters ResnetBlock3d and stays green under every
  // mutation of this value. Liveness is per-arm and MEASURED — the field default
  // 1e-6 -> 1e-4 reds two encoder goldens at max|diff| = 4.38839e-05 — which makes
  // the numerical coverage real but PARTIAL, and is why the pin still carries the
  // arms the goldens do not.
  double norm_eps = 1e-6;
  // `PixelNorm()`'s bare DEFAULT (normalization.py:22), same as the decoder's.
  double pixel_norm_eps = 1e-8;
  std::string prefix;
};

// `SpatioTemporalScaleFactors.from_blocks` (types.py:35-53), the temporal half.
// Every block whose name starts with `compress_time` or `compress_all` halves
// time by 2 REGARDLESS of its channel multiplier, so the factor is 2^steps. This
// is what decides the frame-count crop below, which is why it is exposed: a
// caller that picks a frame count without it silently loses trailing frames.
int64_t Ltx2VideoTemporalScaleFactor(const std::vector<Ltx2VideoEncoderBlock>& blocks);

// The spatial half of the same rule: `patch_size * 2^steps` over blocks starting
// with `compress_space` or `compress_all` (types.py:52).
int64_t Ltx2VideoSpatialScaleFactor(const std::vector<Ltx2VideoEncoderBlock>& blocks,
                                    int64_t patch_size);

// VideoEncoder.forward at batch 1 (video_vae.py:264-336). `frames` is
// [in_channels, F, H, W] channel-major, in [-1, 1] pixel space.
//
// F MUST BE 1 + k * temporal_factor. Upstream WARNS and crops the trailing
// `(F - 1) % factor` frames rather than failing (video_vae.py:276-286); this
// mirrors the crop exactly, and `out_cropped_frames` reports how many were
// dropped so a caller can surface it instead of silently shipping a shorter
// clip. The pointer may be null.
//
// The returned latent is ALREADY NORMALIZED by `per_channel_statistics`
// (video_vae.py:336), i.e. it is in the same space the DiT and
// `Ltx2UpsampleVideoLatent` expect, and the same space `Ltx2ConvVideoDecode`
// de-normalizes on the way back.
Ltx2LatentVolume Ltx2ConvVideoEncode(const Ltx2ConvVideoEncoderConfig& config,
                                     const Ltx2VaeWeights& weights,
                                     const std::vector<float>& frames, int64_t channels,
                                     int64_t frame_count, int64_t height, int64_t width,
                                     int64_t* out_cropped_frames = nullptr);

}  // namespace vllm
