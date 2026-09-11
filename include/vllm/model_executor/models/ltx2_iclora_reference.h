// LTX-2.5 IC-LoRA REFERENCE CONDITIONING — the reference clip's own geometry,
// and the conditioning attention mask that rides it.
//
// Row LTX25-IC-LORA-REF-VIDEO, issue #3020,
// spec .agents/specs/ltx25-ic-lora-ref-video.md (gaps A15 and A16).
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f
//   OURS                              <-  UPSTREAM
//   Ltx2IcLoraReferenceGeometry       <-  ltx-pipelines/iclora_utils.py:111-117
//   Ltx2TemporalSubsample             <-  ltx-pipelines/iclora_utils.py:87-90
//   Ltx2MaskVideoFromPixels           <-  ltx-pipelines/ic_lora.py:511-537
//   Ltx2DownsampleMaskVideoToLatent   <-  ltx-pipelines/iclora_utils.py:52-84
//   Ltx2ResolveCrossMask              <-  ltx-core/conditioning/mask_utils.py:13-73
//   Ltx2BuildAttentionMask            <-  ltx-core/conditioning/mask_utils.py:170-243
//   Ltx2UpdateAttentionMask           <-  ltx-core/conditioning/mask_utils.py:110-167
//
// The PIXEL path is not re-implemented here. `video_preprocess`
// (media_io/decode.py:82-103) is per frame `resize_and_center_crop` then
// `normalize_images`, concatenated on the frame axis, and this tree already has
// that as `Ltx2ReadFrameDirectory` (ltx2_retake.h) — which row LTX25-RETAKE
// ported and gated. This header only says at WHAT GEOMETRY to call it.
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * `temporal_subsample` KEEPS INDEX 1. It is `[0, *range(1, F, N)]` (`:89`),
//    not `range(0, F, N)`. At N=2 over 5 frames the kept set is {0, 1, 3} and
//    the plausible reading gives {0, 2, 4} — the same COUNT, so every shape
//    check agrees and the reference simply describes different moments.
//  * THE MASK'S FIRST LATENT FRAME IS PIXEL FRAME 0 ALONE (`:70`, `:80`), a
//    causal carve-out that mirrors the VAE's own first-frame rule. Pooling all
//    `f_pix` frames uniformly produces a correctly shaped mask of plausible
//    values.
//  * `build_attention_mask` PUTS THE CROSS BLOCK ON THE NOISY ROWS ONLY
//    (`:236`, `:240`), and leaves the prev-ref-to-new-ref blocks at ZERO
//    (`:242`). On a fixture with no PRIOR conditioning item that is elementwise
//    equal to "cross on all existing rows", so the wrong reading is invisible
//    until a second item exists.
//  * AN ALL-ONES MASK IS THE IDENTITY. A downsample that lost its values, a
//    strength that was never multiplied in, and a correct unmasked render are
//    all the same bytes. Nothing about the render's shape can see it.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// Upstream runs this whole path at bf16 (`self.dtype`, ic_lora.py:271-281, and
// `torch.bfloat16` at ic_lora.py:530). Every buffer here is f32, which is this
// tree's standing LTX host-path width and the A24 wave campaign's open
// divergence — NOT a decision this row made. Recorded in
// .agents/specs/ltx25-ic-lora-ref-video.md §4 R4 and in its `## Owed`, because a
// token gate and these goldens are both blind to a dtype that is too wide.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

// `iclora_utils.py:111-117`. The resolution at which the reference clip is READ,
// which is the target's divided by the adapter's `reference_downscale_factor`.
//
// THE REFUSAL IS UPSTREAM'S AND IT IS GUARDED ON `scale != 1` (`:112`), not on
// divisibility alone: at scale 1 no dimension is ever refused, whatever it is.
// Refusing there would reject every render that supplies no adapter, since an
// absent `reference_downscale_factor` reads as 1 (`:35`).
struct Ltx2IcLoraReferenceGeometry {
  int64_t height = 0;
  int64_t width = 0;
};
Ltx2IcLoraReferenceGeometry Ltx2ResolveIcLoraReferenceGeometry(int64_t height, int64_t width,
                                                               int64_t downscale_factor);

// `temporal_subsample` (`iclora_utils.py:87-90`) over a channel-major
// `[channels, frames, height, width]` volume — the layout `Ltx2ReadFrameDirectory`
// returns and `Ltx2ConvVideoEncode` takes.
//
// Returns the KEPT FRAME INDICES, and the caller gathers. The indices are the
// whole content of the function and a vector of them is the only thing a gate
// can compare against upstream without also re-gating the copy.
std::vector<int64_t> Ltx2TemporalSubsampleIndices(int64_t frames, int64_t temporal_scale_factor);

// Gather those indices out of a `[channels, frames, plane]` volume.
std::vector<float> Ltx2TemporalSubsample(const std::vector<float>& clip, int64_t channels,
                                         int64_t frames, int64_t plane,
                                         int64_t temporal_scale_factor);

// `_load_mask_video`'s ARITHMETIC half (`ic_lora.py:530-536`): the three-channel
// pixel volume in [-1, 1] that `Ltx2ReadFrameDirectory` returns becomes a
// one-channel mask in [0, 1] — mean over channels, then `(x + 1) / 2`, then
// clamp. The READ half is `Ltx2ReadFrameDirectory` itself, at the stage's own
// height and width, which is what `ic_lora.py:460-461` spells `args.height // 2`.
std::vector<float> Ltx2MaskVideoFromPixels(const std::vector<float>& pixels, int64_t channels,
                                           int64_t frames, int64_t plane);

// `downsample_mask_video_to_latent` (`iclora_utils.py:52-84`). A pixel-space
// mask `[frames, height, width]` becomes flattened latent token weights
// `[f_lat * h_lat * w_lat]`, in the token order the video patchifier uses.
//
// AREA interpolation spatially (`:63-67`) — for the shapes this engine reaches
// it is exact box averaging — then the causal split: latent frame 0 is pixel
// frame 0 alone (`:70`, `:80`) and the remaining `f_pix - 1` frames mean-pool in
// groups of `t = (f_pix - 1) / (f_lat - 1)` (`:73`, `:78-79`). Refuses a
// non-divisible pair with upstream's own assertion (`:74-77`), and degenerates
// to the first frame alone when either axis has one frame (`:81-82`).
std::vector<float> Ltx2DownsampleMaskVideoToLatent(const std::vector<float>& mask, int64_t f_pix,
                                                   int64_t h_pix, int64_t w_pix, int64_t f_lat,
                                                   int64_t h_lat, int64_t w_lat);

// `resolve_cross_mask` (`mask_utils.py:13-73`) at batch 1. `values` empty is the
// SCALAR arm (`:31-37`): `scalar` fills all `num_new_tokens`. Otherwise `values`
// is the 1-D `(M,)` form (`:49-54`) and its length must be `num_new_tokens`,
// refused by name exactly as `:50-53` refuses it.
std::vector<float> Ltx2ResolveCrossMask(const std::vector<float>& values, double scalar,
                                        int64_t num_new_tokens);

// `build_attention_mask` (`mask_utils.py:170-243`) at batch 1. Returns a dense
// `[(N+M) * (N+M)]` row-major mask in [0, 1].
//
//                  noisy(Nn)   prev_ref(N-Nn)   new_ref(M)
//     noisy        existing    existing         cross
//     prev_ref     existing    existing         0
//     new_ref      cross       0                1
//
// `existing` EMPTY is upstream's `existing_mask=None` and fills the top-left
// `N x N` block with ones (`:225-226`); otherwise it is preserved (`:224`),
// which is what keeps an earlier item's attenuation alive under a later one.
std::vector<float> Ltx2BuildAttentionMask(const std::vector<float>& existing,
                                          int64_t num_noisy_tokens, int64_t num_new_tokens,
                                          int64_t num_existing_tokens,
                                          const std::vector<float>& cross_mask);

// `update_attention_mask` (`mask_utils.py:110-167`) at batch 1, for the
// `attention_mask is None` arm alone (`:141-156`).
//
// `existing` empty returns empty — upstream's "no mask requested and none
// present, return None" (`:142-143`). `existing` NON-empty pads the new tokens
// with FULL attention (`:147`) rather than returning the mask unchanged, which
// is what keeps the mask's dimensions equal to the growing sequence. A mask left
// short of the sequence is not a shape error anywhere downstream: it is read as
// a broadcastable key-only form and silently masks the wrong axis.
std::vector<float> Ltx2PadAttentionMaskForUnmaskedTokens(const std::vector<float>& existing,
                                                         int64_t num_noisy_tokens,
                                                         int64_t num_new_tokens,
                                                         int64_t num_existing_tokens);

}  // namespace vllm
