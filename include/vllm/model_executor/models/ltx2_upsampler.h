// LTX-2.5 LATENT UPSAMPLER — the SPATIAL x2 arm (stage 2 of the distilled
// two-stage recipe) and the TEMPORAL x2 arm.
//
// Row: MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model. Specs:
// .agents/specs/ltx-2-5.md (phase L5, the spatial arm, issue #435) and
// .agents/specs/ltx25-temporal-upsampler.md (the temporal arm, issue #644 row E).
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/model/upsampler/
//   OURS                        <-  UPSTREAM
//   Ltx2LatentUpsample          <-  model.py:82-126 (LatentUpsampler.forward)
//   Ltx2UpsamplerConfig         <-  model.py:25-80 + model_configurator.py:11-31
//   (ResBlock)                  <-  res_block.py:29-37
//   (SpatialRationalResampler)  <-  spatial_rational_resampler.py:40-47
//   (PixelShuffleND, dims 2)    <-  pixel_shuffle.py:40-46
//   (PixelShuffleND, dims 1)    <-  pixel_shuffle.py:47-52
//   (the first-frame drop)      <-  model.py:111-113
//   (BlurDownsample)            <-  blur_downsample.py:29-53
//   Ltx2RationalForScale        <-  spatial_rational_resampler.py:10-14
//   Ltx2UpsampleVideoLatent     <-  model.py:129-143 (upsample_video)
//
// ─── WHAT THE TEMPORAL ARM IS REACHABLE FROM ─────────────────────────────────
// NOTHING, today, and that is stated here rather than only in the spec because a
// header is what the next reader opens. It is ported and gated against executed
// upstream at reduced dimensions, and `Ltx2ParseUpsamplerConfig`
// (ltx2_loader.cpp:1431-1444) reads `temporal_upsample` off a checkpoint. But the
// engine's ONE upsampler call site is the `kSpatialUpsample` phase input
// transform (multimodal/ltx2_video.cpp:1408-1466), which shape-checks the result
// against a SPATIALLY doubled latent and fails otherwise; and upstream's only
// consumer is `DFRPipeline`'s rounds loop (ltx-pipelines/dfr_pipeline.py:235-245,
// 402-407), which is not ported. The shipped temporal checkpoint
// (`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`,
// ltx-pipelines/docs/pipelines.md:176) is not on the NAS either, so no
// real-weight result exists.
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
//    (pixel_shuffle.py:41-47). Swapping them transposes every 2x2 block. The
//    dims=1 arm has the same trap on one axis: `(c p1)` -> `(f p1)` puts p1
//    FASTEST in both groupings (pixel_shuffle.py:47-52).
//  * The temporal arm DROPS THE FIRST FRAME after the shuffle (model.py:109-113),
//    so `f` frames in produce `2f - 1` out and not `2f`.
//
// ─── NOT PORTED, refused by name ─────────────────────────────────────────────
//  * `spatial_upsample AND temporal_upsample` (model.py:55-59) — a DIFFERENT
//    operator from the temporal-only arm: `Conv3d(mid, 8*mid)` + PixelShuffleND(3).
//    Asking for it throws.
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

// `PixelShuffleND.__init__`'s `upscale_factors` default (pixel_shuffle.py:25),
// element 0 — the only element the `dims == 1` arm reads (:47-52). No
// construction site in `ltx_core` passes one, so this default IS the shipped
// temporal upscale factor and it decides how many frames come out. Gated against
// upstream's own signature by test_ltx2_pipeline.cpp, case "ltx2 the latent
// temporal upsampler reproduces upstream".
inline constexpr int64_t kLtx2UpsamplerTemporalFactor = 2;

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

// LatentUpsampler.forward (model.py:82-126), the dims == 3 arms: spatial-only
// (`[b, c, f, h, w] -> [b, c, f, 2h, 2w]` at scale 2.0) and temporal-only
// (`-> [b, c, 2f-1, h, w]`). Throws by name for `dims == 2`, for both flags set
// at once, and when neither is set (upstream's own ValueError at :74).
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
