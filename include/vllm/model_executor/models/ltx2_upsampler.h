// LTX-2.5 LATENT SPATIAL UPSAMPLER — stage 2 of the distilled two-stage recipe.
//
// Row: MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model. Spec:
// .agents/specs/ltx-2-5.md (phase L5). Issue #435.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/model/upsampler/
//   OURS                        <-  UPSTREAM
//   Ltx2LatentUpsample          <-  model.py:82-126 (LatentUpsampler.forward)
//   Ltx2UpsamplerConfig         <-  model.py:25-80 + model_configurator.py:11-31
//   (ResBlock)                  <-  res_block.py:29-37
//   (SpatialRationalResampler)  <-  spatial_rational_resampler.py:40-47
//   (PixelShuffleND)            <-  pixel_shuffle.py:31-54
//   (BlurDownsample)            <-  blur_downsample.py:29-53
//   Ltx2RationalForScale        <-  spatial_rational_resampler.py:10-14
//   Ltx2UpsampleVideoLatent     <-  model.py:129-143 (upsample_video)
//
// ─── WHAT SEPARATES THIS FROM THE VAE'S CONVOLUTIONS ─────────────────────────
// These are plain `torch.nn.Conv3d`/`Conv2d` with `padding=1` — ZERO padding on
// every axis INCLUDING time. The Conv video VAE next door uses `CausalConv3d`,
// which prepends replicated copies of frame 0 (convolution.py:306-307). The two
// are not interchangeable and this file does not reuse that kernel: a causal pad
// here would shift the whole clip while still producing a correctly shaped,
// finite, plausible latent.
//
// Three more things that fail silently:
//  * GroupNorm's group count is HARDCODED to 32 (res_block.py:24,26; model.py:50),
//    not a config key. With mid_channels 512 that is 16 channels per group; a port
//    that normalized per channel or per tensor produces a valid-looking latent.
//  * The residual is added BEFORE the activation, not after: `activation(x + residual)`
//    (res_block.py:36). `activation(x) + residual` is the same shape and a
//    different function.
//  * PixelShuffleND unpacks `(c p1 p2)` with p1 taking HEIGHT and p2 taking WIDTH
//    (pixel_shuffle.py:41-47). Swapping them transposes every 2x2 block.
//
// ─── NOT PORTED, refused by name ─────────────────────────────────────────────
//  * `temporal_upsample` (model.py:55-72, 109-113) — spec section 2 puts the
//    temporal x2 upsampler out of scope; asking for it throws.
//  * `dims == 2` (model.py:85-100) — a checkpoint that sets it wants Conv2d
//    everywhere, i.e. no temporal convolution at all. LTX-2.5's upsampler is
//    dims=3; the 2-D arm is refused rather than approximated by the 3-D one.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// f32, because this is the CPU REFERENCE arm and the gate compares the ALGORITHM
// against upstream run in torch float32. Upstream runs the upsampler in the
// pipeline's bfloat16 (distilled.py:109, 219), so the bf16 arm is owed by phase
// L6 exactly as the VAEs' is.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"  // Ltx2VaeWeights

namespace vllm {

// res_block.py:24,26 and model.py:50 — `torch.nn.GroupNorm(32, channels)`, a
// literal on all three sites.
inline constexpr int64_t kLtx2UpsamplerNormGroups = 32;
// torch's `nn.GroupNorm` default `eps` (it is not passed at any of the three
// construction sites). NOT a member of the invisible-constant class. It was
// recorded as one on a mutation that happened not to move anything, and a
// mutation that moves nothing proves nothing; at the class's OWN 100x bar
// (1e-5 -> 1e-3) it REDS all three arms of
// "ltx2 the latent spatial upsampler reproduces upstream" —
// PixelShuffle 0.0289409, Rational2 0.0347079, Rational1p5 0.0649014. The pin
// below still earns its place, because a golden regenerated with a moved eps
// moves with it and only the pin compares against torch's own default.
inline constexpr double kLtx2UpsamplerNormEps = 1e-5;

// LatentUpsampler.__init__ defaults (model.py:25-35), which are also
// LatentUpsamplerConfigurator.from_metadata's `config.get` fallbacks
// (model_configurator.py:14-21).
struct Ltx2UpsamplerConfig {
  int64_t in_channels = 128;
  int64_t mid_channels = 512;
  int64_t num_blocks_per_stage = 4;
  int64_t dims = 3;
  bool spatial_upsample = true;
  bool temporal_upsample = false;
  double spatial_scale = 2.0;
  bool rational_resampler = false;
  std::string prefix;
};

// _rational_for_scale (spatial_rational_resampler.py:10-14): the up/down integer
// pair for a supported scale. Upstream RAISES on anything else (:12-13) and so
// does this — an unsupported scale is a config error, never a nearest match.
struct Ltx2RationalScale {
  int64_t num = 1;
  int64_t den = 1;
};
Ltx2RationalScale Ltx2RationalForScale(double scale);

// BlurDownsample's fixed separable binomial kernel (blur_downsample.py:29-33):
// the outer product of Pascal's row `kernel_size - 1`, normalized to sum 1. It is
// COMPUTED at construction, never loaded, so both sides must build it
// independently — the same rule the audio VAE's kaiser-sinc windows follow.
// Returns [kernel_size * kernel_size], row-major.
std::vector<float> Ltx2BlurKernel(int64_t kernel_size);
// `BlurDownsample.__init__`'s default (blur_downsample.py:14), which
// `SpatialRationalResampler` never overrides (:38) — so this default IS the
// shipped kernel width. Gated against upstream's own signature by
// test_ltx2_pipeline.cpp, case "the constants the headers call pinned are
// actually pinned", and reached NUMERICALLY as well: 5 -> 3 REDS the Rational1p5
// arm of "ltx2 the latent spatial upsampler reproduces upstream" at 0.689782.
// Only that arm, because the blur runs on the rational `den` and 1.5 -> {3, 2} is
// the one scale the suite covers with den != 1.
inline constexpr int64_t kLtx2BlurKernelSize = 5;

// The parameter contract: every tensor `LatentUpsampler(config)` creates, in
// `named_parameters()` order. This IS the layout — the parity suite compares it
// against the upstream module's own.
struct Ltx2UpsamplerTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};
std::vector<Ltx2UpsamplerTensorSpec> EnumerateLtx2UpsamplerTensors(
    const Ltx2UpsamplerConfig& config);

// A [batch, channels, frames, height, width] latent, row-major.
struct Ltx2LatentVolume {
  int64_t batch = 1;
  int64_t channels = 0;
  int64_t frames = 0;
  int64_t height = 0;
  int64_t width = 0;
  std::vector<float> data;

  int64_t elems() const { return batch * channels * frames * height * width; }
};

// LatentUpsampler.forward (model.py:82-126), the dims == 3 spatial arm. Throws by
// name for `dims == 2`, for `temporal_upsample`, and when neither upsample flag
// is set (upstream's own ValueError at :74).
Ltx2LatentVolume Ltx2LatentUpsample(const Ltx2UpsamplerConfig& config,
                                    const Ltx2VaeWeights& weights,
                                    const Ltx2LatentVolume& latent);

// upsample_video (model.py:129-143): un-normalize by the video encoder's
// per-channel statistics, upsample, re-normalize. `std_of_means` / `mean_of_means`
// are the encoder's `per_channel_statistics` (video_vae/ops.py:63-84), one value
// per LATENT channel.
//
// Exposed separately from `Ltx2LatentUpsample` because the statistics belong to
// the VAE, not to the upsampler, and a caller holding a latent that is already in
// un-normalized space must not apply them twice.
Ltx2LatentVolume Ltx2UpsampleVideoLatent(const Ltx2UpsamplerConfig& config,
                                         const Ltx2VaeWeights& weights,
                                         const Ltx2LatentVolume& latent,
                                         const std::vector<float>& std_of_means,
                                         const std::vector<float>& mean_of_means);

}  // namespace vllm
