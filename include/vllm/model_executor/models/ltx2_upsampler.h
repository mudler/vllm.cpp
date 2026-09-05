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
// ─── WHAT EACH ARM IS REACHABLE FROM ─────────────────────────────────────────
// Stated here rather than only in the spec because a header is what the next
// reader opens. `Ltx2UpsampleVideoLatent` has THREE product call sites, all in
// multimodal/ltx2_video.cpp, and each one pins the frame axis:
//
//   :3521  the video latent, the `kSpatialUpsample` phase input transform.
//          Requires `up.frames == vshape.frames` at :3525-3531.
//   :3548  the generated keyframe slots, which take the SAME spatial upsampler
//          (dfr_pipeline.py:348). Requires `slot_positions.size()` at :3552-3563.
//   :5058  DFR's temporal-refinement rounds, the TEMPORAL arm. Reached through
//          the `temporal_upsample_rounds` load extra, and instrumented by
//          `trace.temporal_upsample_calls` (multimodal/ltx2_video.h:741).
//
// This paragraph said "NOTHING, today" of the temporal arm and "the engine's ONE
// upsampler call site" of the spatial one. Both were false by the time they were
// read: site :5058 drives the temporal arm and is not a phase input transform at
// all. Corrected under issue #2580; the count is what the dims=2 port's
// reachability argument rests on, so it is derived here rather than remembered.
// The shipped temporal checkpoint
// (`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`,
// ltx-pipelines/docs/pipelines.md:176) is not on the NAS, so no real-weight
// result exists for that arm.
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
// ─── THE dims=2 ARM (model.py:47, :85-100) ───────────────────────────────────
// PORTED. `conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d` (:47)
// reaches four parameter groups — `initial_conv`, both ResBlock stacks and
// `final_conv` — so each is a 4-D kernel where the 3-D arms build a 5-D one. The
// `upsampler` branch (:55-72) never reads `dims` and keeps its rank.
//
// The forward folds the frame axis into the BATCH (:86) and unfolds at :100,
// which has one consequence a shape check cannot see: GroupNorm normalises PER
// FRAME. This port reproduces the fold by running one frame at a time, so the
// existing reduction over `frames * height * width` gives the per-frame
// statistic without a second normaliser. Gated by "reproduces upstream on the
// dims=2 arm" (test_ltx2_pipeline) against the executed module, and reached
// end-to-end by "a dims=2 upsampler checkpoint RENDERS" (test_ltx2_video).
//
// Any `dims` that is not 2 builds Conv3d, which is upstream's own `else` at :47
// and is mirrored rather than narrowed to a refusal upstream does not raise.
//
// ─── NOT PORTED, refused by name ─────────────────────────────────────────────
//  * `spatial_upsample AND temporal_upsample` (model.py:55-59) — a DIFFERENT
//    operator from the temporal-only arm: `Conv3d(mid, 8*mid)` + PixelShuffleND(3).
//    Asking for it throws. Its operator is small; what keeps it out is that it
//    returns `[c, 2f-1, 2h, 2w]` and BOTH spatial call sites above require the
//    frame count back unchanged. Owed, with the measurement, in
//    .agents/specs/ltx25-upsampler-arms.md.
//  * `dims == 2` WITH `temporal_upsample` — not an arm but a contradiction:
//    upstream builds that upsampler as a Conv3d (:68-71) and the 2-D forward
//    hands it a 4-D tensor. It raises on the CHANNEL COUNT and not on the rank:
//    `Conv3d` reads a 4-D input as an unbatched 5-D one, so at 3 frames against
//    `mid_channels` 32 it reports "to have 32 channels, but got 3". At
//    `frames == mid_channels` the conv passes and `PixelShuffleND(1)` fails
//    instead. Two mechanisms, one contradiction — refused by name once.
//  * `dims == 2` WITH `rational_resampler` — the sibling, and the dangerous one:
//    every operator in that branch is per-frame, so this port would compute a
//    finite, plausible latent no shape check could fault. Upstream raises
//    `not enough values to unpack (expected 5, got 4)` at
//    spatial_rational_resampler.py:41. Refused by name.
//
//    Both are EXECUTED against the real module by
//    scripts/gen-ltx2-pipeline-goldens.py, which asserts each raises and what it
//    says, so neither refusal can drift into one upstream would serve.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// TWO ARMS since A24 wave 5 (row LTX25-A24-UPSAMPLER-BF16, #2857), and the arm
// is the WEIGHT BAG'S, never a caller's choice: `Ltx2VaeWeights` populates one
// of its two maps and its `dtype` says which (ltx2_audio_vae.h:71-85).
//
//   bf16  Upstream's own model dtype. `distilled.py:109` resolves ONE dtype for
//         the pipeline and `:138-141` hands it to the latent upsampler, so this
//         is what a shipped checkpoint runs at and it is the DEFAULT the engine
//         loads (multimodal/ltx2_video.cpp).
//   f32   The CPU parity arm every committed golden was measured against, kept
//         because the algorithm gate compares against upstream run in float32.
//
// The storage really is the width: every intermediate buffer holds bytes at
// `dtype` rather than f32 bytes carrying narrowed values, and that sentence is
// MEASURED rather than asserted -- `Ltx2UpsamplerStorage` below is the
// observable, because a build that keeps every dtype field correct and sizes
// every buffer by `sizeof(float)` passes every value gate and every dtype
// counter in this tree.
//
// The measurement is taken PER ARM, because one arm does not reach one file: it
// runs the three spatial arms, the temporal arm and the `dims == 2` fold, which
// between them allocate every buffer this stage allocates. The spatial arm alone
// reaches neither the anti-aliasing blur -- only a rational scale with `den > 1`
// does -- nor the two buffers of the `dims == 2` fold, and a widening confined to
// the blur was built and run green against a single-arm version of the gate.
//
// Six rounding rules separate the two arms, and each was measured against the
// executed module rather than read off it -- the GroupNorm affine's single
// rounding, the f32 epsilon, SiLU's single rounding, the residual add that
// rounds BEFORE the activation, the blur kernel's registered buffer, and
// `PerChannelStatistics`' narrowed buffers and two roundings.
// .agents/specs/ltx25-a24-upsampler-bf16.md section 3 tabulates them with the
// alternative each rejects and a separating count.
//
// The FP8 and NVFP4 arms are A22 and are refused by name. There is no CUDA arm:
// this file has no queue and no `vt::` kernel seam, so a device arm is a
// residency row and not a dtype row. Both are owed in the row's spec.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"  // Ltx2VaeWeights
#include "vt/dtype.h"

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
//
// `data` stays f32 because a latent is an INTERFACE value -- the same split
// `Ltx2ConvVideoEncode` makes on its own return. `dtype` reports the width the
// stage that produced it actually COMPUTED at, and it is not decoration: A24's
// deliverable is a dtype, a token gate cannot see one, and this field is what a
// production path reads to assert it. It is set by `Ltx2LatentUpsample` from the
// weight bag's own arm and defaults to f32 on a latent nobody has upsampled.
struct Ltx2LatentVolume {
  int64_t batch = 1;
  int64_t channels = 0;
  int64_t frames = 0;
  int64_t height = 0;
  int64_t width = 0;
  std::vector<float> data;
  vt::DType dtype = vt::DType::kF32;

  int64_t elems() const { return batch * channels * frames * height * width; }
};

// ─── THE STORAGE OBSERVABLE (A24 wave 5, row LTX25-A24-UPSAMPLER-BF16, #2857) ─
//
// A24's deliverable is a WIDTH, and `Ltx2LatentVolume::dtype` above reports the
// width a stage COMPUTED at, not the width it stored. Those are different
// claims and only the first of them had a gate. The review of this row built the
// difference and ran it: sizing every internal buffer by `sizeof(float)` while
// still rounding each stored value to bf16 keeps every value, every golden and
// every reported `dtype` bit-identical -- 9125 assertions, zero failures -- and
// moves twice the bytes. That is exactly the polarity AGENTS.md names when it
// says "a token gate cannot detect a dtype that is too wide".
//
// So the bytes are counted where they are actually allocated and read, and the
// gate compares TWO RUNS on the same input rather than quoting a number: the
// bf16 arm must be EXACTLY half the f32 arm's. This is the shape
// `Ltx2VaeWeights::Bytes()` documents for the weight bag (ltx2_audio_vae.h:104-107)
// applied to the one thing that bag does not cover, the upsampler's own
// intermediate volumes.
//
// TWO COUNTS, because each catches a widening the other cannot see:
//
//   `volumes`/`elems`/`bytes` are the intermediate VOLUMES -- what `Volume::Alloc`
//     really reserved. `bytes / elems` is the storage width, and it is 4 rather
//     than 2 on a bf16 arm that widened its buffers.
//   `param_views`/`param_elems`/`param_bytes` are the PARAMETERS, taken off the
//     same member `WeightView::operator[]` dispatches on, so a view that
//     materialises a widened f32 copy reports the width it reads THROUGH rather
//     than the width the checkpoint is stored at. That is the claim "it is a
//     view and not a widened copy" reduced to a number; at the shipped
//     `mid_channels = 512` one convolution weight alone is 7.1 M parameters.
//
// Read-and-CLEAR, and per thread: the caller brackets its own call, so two
// callers cannot read each other's bytes and a leftover cannot be counted twice.
// Nothing in this file threads work, so a whole upsample accumulates on the
// thread that asked for it.
struct Ltx2UpsamplerStorage {
  int64_t volumes = 0;
  int64_t elems = 0;
  int64_t bytes = 0;
  int64_t param_views = 0;
  int64_t param_elems = 0;
  int64_t param_bytes = 0;
};
Ltx2UpsamplerStorage Ltx2TakeUpsamplerStorage();

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
