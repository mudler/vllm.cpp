// LTX-2.5 DFR CANVAS LAYOUT — the keyframe segment grid, the temporal tile
// ranges and the latent stitch that `DFRPipeline` denoises over.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca,
// packages/ltx-pipelines/src/ltx_pipelines/
//   OURS                              <-  UPSTREAM
//   Ltx2DfrTileRange                  <-  dfr_layout.py:21-38  (TileRange)
//   Ltx2DfrChooseSegmentLength        <-  dfr_layout.py:40-57
//   Ltx2DfrResolveCanvas              <-  dfr_layout.py:60-81
//   Ltx2DfrPixelToLatentIndex         <-  dfr_layout.py:84-90
//   Ltx2DfrTileRanges                 <-  dfr_layout.py:137-182 (+ :93-134)
//   Ltx2DfrStitchTileLatents          <-  dfr_layout.py:185-208
//   Ltx2DfrRemapPositionsToLocal      <-  dfr_layout.py:211-213
//   Ltx2DfrSlotInitialsFromVideo      <-  dfr_pipeline.py:101-111
//   Ltx2DfrMergeCarryForwardKeyframes <-  dfr_pipeline.py:114-139
//   Ltx2DfrTrimToTargetFrames         <-  dfr_pipeline.py:531-540
//
// This module is PURE INDEX ARITHMETIC over a latent volume. It holds no
// weights, runs no kernel and reads no checkpoint, which is why it is its own
// translation unit: every one of its failure modes produces a correctly shaped,
// finite, plausible latent, so the only thing that can catch them is an exact
// gate on the indices themselves.
//
// ─── THE FIVE THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * THE SEGMENT TIE GOES TO THE LARGER CANDIDATE. `choose_segment_length`
//    (dfr_layout.py:55) takes a strictly smaller pad, and on a tie keeps the
//    LARGER segment. `SEGMENT_CANDIDATES` is (24, 32) and the loop seeds `best`
//    with the FIRST candidate (:50-51), so a port that seeded with the largest,
//    or that used `<=` instead of `<`, silently renders a different keyframe
//    grid on every frame count whose pads are equal — 120 content frames pads 0
//    both ways, and the two answers place 5 keyframes or 4.
//  * FRAME 0 IS NOT A KEYFRAME. `_build_tile` filters `if boundaries[index]`
//    (dfr_layout.py:129), which drops the leading 0 and ONLY the leading 0. A
//    port that took the whole boundary run would ask the carry-forward bag for a
//    keyframe at pixel frame 0 that no round ever generated, and
//    `_merge_carry_forward_keyframes` would then hand the next round an anchor
//    it has no latent for.
//  * THE NON-FIRST TILE DROPS ITS LEAD-IN PLUS ONE MORE LATENT.
//    `drop_latent_prefix` adds 1 when `own_lo > 0` (dfr_layout.py:120-122)
//    because the PREVIOUS tile keeps the seam latent — the 8-pixel-frame token
//    that ENDS on the keyframe mark — and this one resumes strictly after it.
//    Off by one here stitches a clip of the right length with one latent frame
//    duplicated or missing at every seam.
//  * THE SLOT SEED INDEX ROUNDS HALF-TO-EVEN. See `Ltx2DfrSlotInitialsFromVideo`.
//
// AND ONE THAT DOES **NOT** FAIL SILENTLY, recorded because this header claimed
// the opposite until a mutation refuted it. `tile_ranges` passes
// `lead_segments if index > 0 else 0` (dfr_layout.py:177), and it is natural to
// read that conditional as load-bearing. **It is not.** For the first tile
// `own_lo == 0`, so `window_lo = max(0, 0 - lead) = 0` for every lead, and
// `drop_latent_prefix = latent(boundaries[0]) - latent_start = 0 - 0 = 0` with
// no `+1` because `own_lo > 0` is false. The conditional is therefore REDUNDANT
// with the clamp beside it, and a port that dropped it would be byte-identical.
//
// MEASURED against executed upstream rather than argued: `_build_tile` called
// on the first tile with `lead_segments` 0, 1 and 5 returns three `TileRange`s
// that compare EQUAL. The mutation that removes the conditional here stays GREEN
// for that reason, and it stays green because there is nothing to detect — which
// is a different thing from a coverage hole, and the difference is why it is
// written down. `Ltx2DfrTileRanges` keeps the conditional anyway, to mirror
// upstream's text rather than its algebra, and the invariant is gated by
// `test_ltx2_dfr`'s "the first tile's window is lead-in invariant" so that the
// claim in this paragraph is checked rather than trusted.
//  * THE SLOT SEED INDEX ROUNDS, IT DOES NOT FLOOR, AND IT ROUNDS HALF-TO-EVEN.
//    `_slot_initials_from_video` (dfr_pipeline.py:109) is
//    `min(max(round(position / temporal_scale), 0), T-1)`, and Python's `round`
//    is half-to-even where C's `std::round` is half-away-from-zero. See
//    `Ltx2DfrSlotInitialsFromVideo` for why the distinction is unobservable on
//    the shipped segment candidates and is mirrored anyway.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/ltx2_pipeline.h"   // Ltx2ScaleFactors
#include "vllm/model_executor/models/ltx2_upsampler.h"  // Ltx2LatentVolume

namespace vllm {

// SEGMENT_CANDIDATES (dfr_layout.py:12). A tuple upstream, an array here; the
// ORDER is load-bearing, because `choose_segment_length` seeds its running best
// with element 0 and only a STRICTLY smaller pad displaces it.
inline constexpr int64_t kLtx2DfrSegmentCandidates[] = {24, 32};

// TILE_LEAD_SEGMENTS (dfr_layout.py:18). Upstream's own reason, kept because it
// is the only thing that explains the `+1` in `drop_latent_prefix`: a tile's
// local latent 0 is an IMAGE latent (one pixel frame) and its local latent 1 was
// denoised against that image, so neither may be spliced into the mid-canvas
// stream. One segment of lead-in puts both inside the discarded prefix while
// keeping the window start on a keyframe anchor, which the KeyFrame@0 lock
// requires.
inline constexpr int64_t kLtx2DfrTileLeadSegments = 1;

// The anchor keyframes carried between temporal rounds, pinned just short of
// fully clean so a tile can still settle its seam frame (dfr_pipeline.py:70-72).
// Upstream calls this value "ours" in that comment — it is Lightricks' own
// choice rather than a trained constant, which is why it is named here rather
// than spelled at the call site.
inline constexpr double kLtx2DfrAnchorKeyframeStrength = 0.95;

// The ancestral eta the temporal rounds sample with (dfr_pipeline.py:73, applied
// at :495). NOT `kLtx2AncestralEta`: the distilled stage-1 sampler runs at 1.0
// (distilled.py:62-84) and a round that reused it would inject twice the noise
// into a 4-step refinement that cannot remove it.
inline constexpr double kLtx2DfrTemporalAncestralEta = 0.5;

// dfr_pipeline.py:74-78. The CONDITIONING fps is capped independently of the
// playback fps, and upstream's reason is a statement about RoPE rather than a
// safety margin: RoPE time is `pixel_frame / fps`, so a 120 fps time base halves
// every token's temporal span against the trained distribution and the model can
// no longer lay out the 8 pixel frames inside one latent token — it decodes as a
// motion spike at each latent border followed by a stall. Playback fps is used
// for decoding only.
inline constexpr double kLtx2DfrMaxConditioningFps = 60.0;

// The rounds the CLI and the pipeline both accept (dfr_pipeline.py:284-285,
// utils/args.py-side `choices=(0, 1, 2)` at :587). An inclusive upper bound, and
// upstream refuses anything else by name rather than clamping.
inline constexpr int64_t kLtx2DfrMaxTemporalRounds = 2;

// TileRange (dfr_layout.py:21-38): one temporal denoise tile in GLOBAL pixel and
// latent coordinates.
//
// `pixel_start` / `pixel_end` are INCLUSIVE, `latent_end_exclusive` is half-open
// — upstream's own asymmetry, kept because the two coordinate systems are read
// by different consumers and collapsing them would need a conversion at every
// site. Non-first tiles start `kLtx2DfrTileLeadSegments` segments before the
// region they own, so the seam shared with the previous tile falls inside the
// window.
//
// `anchor_kf_global` are the seam keyframes in the window, which includes
// `pixel_start` on every tile but the first — frame 0 is not a keyframe, so the
// first tile's window start contributes no anchor. `slot_kf_global` are the
// mid-segment positions this window INVENTS.
struct Ltx2DfrTileRange {
  int64_t pixel_start = 0;
  int64_t pixel_end = 0;
  int64_t latent_start = 0;
  int64_t latent_end_exclusive = 0;
  std::vector<int64_t> anchor_kf_global;
  std::vector<int64_t> slot_kf_global;
  int64_t drop_latent_prefix = 0;
};

// `resolve_canvas`'s three return values (dfr_layout.py:60-81).
struct Ltx2DfrCanvas {
  // N': `content_padded + 1`. The canvas the pipeline actually denoises, which
  // may be LONGER than the caller asked for.
  int64_t num_frames = 0;
  // S: the keyframe segment length this canvas chose.
  int64_t segment = 0;
  // `[S, 2S, ..., N' - 1]`. Frame 0 is EXCLUDED — it is already a single-pixel-
  // frame token under causal encoding, so it needs no slot — and the terminal
  // frame IS included.
  std::vector<int64_t> positions;
};

// `choose_segment_length` (dfr_layout.py:40-57). `content_frames` is
// `num_frames - 1`. Picks the candidate that pads LEAST; ties keep the LARGER.
int64_t Ltx2DfrChooseSegmentLength(int64_t content_frames);

// `resolve_canvas` (dfr_layout.py:60-81). Pads `num_frames - 1` up to a multiple
// of the segment length.
//
// The `(num_frames - 1) % temporal_scale == 0` refusal is upstream's (:71-72)
// and it is NOT the same rule as the resolution envelope's frames axis (#919):
// that one documents the engine's FLOOR, this one refuses. DFR needs the
// stronger rule because every keyframe position it emits must land on a latent
// border, and a floored count moves the terminal position off one.
Ltx2DfrCanvas Ltx2DfrResolveCanvas(int64_t num_frames,
                                   int64_t temporal_scale = Ltx2ScaleFactors{}.time);

// `pixel_to_latent_index` (dfr_layout.py:84-90). Pixel frame 0 is legal and maps
// to latent 0; every other value must sit on the x`temporal_scale` border.
int64_t Ltx2DfrPixelToLatentIndex(int64_t pixel_frame,
                                  int64_t temporal_scale = Ltx2ScaleFactors{}.time);

// `tile_ranges` (dfr_layout.py:137-182). Partitions the canvas into `num_tiles`
// keyframe-seam tiles, GAPLESS, with a lead-in on every tile but the first.
// `num_tiles` is CLAMPED to the segment count (:171) rather than refused, which
// is what makes a 2-segment canvas legal at round 2.
std::vector<Ltx2DfrTileRange> Ltx2DfrTileRanges(
    const std::vector<int64_t>& seam_positions, int64_t num_frames, int64_t num_tiles,
    int64_t temporal_scale = Ltx2ScaleFactors{}.time,
    int64_t lead_segments = kLtx2DfrTileLeadSegments);

// `stitch_tile_latents` (dfr_layout.py:185-208). Concatenates the tile video
// latents along T, each tile contributing `latent[drop_latent_prefix:]`.
//
// Every tile must carry the SAME channels, height and width; upstream gets that
// from `torch.cat`'s own broadcast rules, and this port asserts it, because a
// tile at the wrong spatial size would otherwise be memcpy'd into a volume whose
// header says something else.
Ltx2LatentVolume Ltx2DfrStitchTileLatents(const std::vector<Ltx2LatentVolume>& tile_latents,
                                          const std::vector<Ltx2DfrTileRange>& ranges);

// `remap_positions_to_local` (dfr_layout.py:211-213): `local = global - start`.
//
// Trivial, and exposed anyway, because it is the seam a tile's conditioning is
// built through. Upstream applies it to the anchor bag (dfr_pipeline.py:465) and
// to the slot bag (:472) and to NEITHER the image conditioning nor the stitch,
// and a port that inlined `- pixel_start` at the two sites it belongs to would
// leave nothing for a test to pin when a third site grows.
std::vector<int64_t> Ltx2DfrRemapPositionsToLocal(const std::vector<int64_t>& positions,
                                                  int64_t pixel_start);

// `latent[:, :, start:end]` — the half-open frame slice upstream writes inline
// wherever it needs one: the tile's own window (`dfr_pipeline.py:423`), a single
// keyframe out of the carry-forward bag (`:460`), and a single slot out of a
// tile's return (`:524`).
//
// It is a named function rather than three inline loops because the volume is
// [B, C, T, H, W] ROW-MAJOR and a flat `data.begin() + start * plane` is wrong
// for any volume with more than one channel: it returns a correctly sized
// buffer holding channel 0's later frames instead of every channel's window.
// That is the same defect `Ltx2DfrStitchTileLatents` carries a comment about,
// and the two now share the reason as well as the layout.
Ltx2LatentVolume Ltx2DfrSliceLatentFrames(const Ltx2LatentVolume& latent, int64_t start,
                                          int64_t end_exclusive);

// `torch.cat(pieces, dim=2)` over single-frame volumes — the bag builder behind
// `dfr_pipeline.py:459-461`, `:523-525` and `:139`. Same reason as the slice:
// concatenating along T in a [B, C, T, H, W] buffer is a per-plane copy, and
// appending the pieces end to end interleaves the channels wrong while
// producing exactly the right number of floats.
Ltx2LatentVolume Ltx2DfrConcatLatentFrames(const std::vector<Ltx2LatentVolume>& pieces);

// `_slot_initials_from_video` (dfr_pipeline.py:101-111): stack the NEAREST video
// latent frames as `(B, C, K, H, W)` slot seeds.
//
// `positions` are TILE-LOCAL pixel frames and `video` is the tile's own latent
// window, which is why this takes the remapped bag rather than the global one.
//
// THE INDEX ROUNDS RATHER THAN FLOORS, and upstream's `round` is Python's, which
// is HALF-TO-EVEN. On the shipped segment candidates the quotient is always
// exact — every slot position is an odd multiple of S, S is 24 or 32, and both
// divide by the temporal scale 8 — so no shipped canvas can observe the
// difference between round-half-even, round-half-up and floor. It is mirrored
// exactly anyway, because "unobservable on the values we ship" is a statement
// about SEGMENT_CANDIDATES rather than about this function, and the next
// candidate added upstream would silently change which frame seeds every slot.
Ltx2LatentVolume Ltx2DfrSlotInitialsFromVideo(const Ltx2LatentVolume& video,
                                              const std::vector<int64_t>& positions,
                                              int64_t temporal_scale);

// `_merge_carry_forward_keyframes` (dfr_pipeline.py:114-139): the next round's
// anchor bag — the carried keyframe stills PLUS this round's denoised slots,
// keyed by position and sorted.
//
// Positions must already be on the CURRENT round's pixel grid; the caller remaps
// (x2) for the next round. A slot at the same position as an anchor WINS, which
// is upstream's insertion order (anchors first at :125, slots second at :126)
// and not an arbitrary tie-break: the slot was denoised this round and the
// anchor is a still carried from the last one.
struct Ltx2DfrCarryForward {
  std::vector<int64_t> positions;
  Ltx2LatentVolume keyframes;
};
Ltx2DfrCarryForward Ltx2DfrMergeCarryForwardKeyframes(
    const std::vector<int64_t>& anchor_positions, const Ltx2LatentVolume* anchor_latents,
    const std::vector<int64_t>& slot_positions, const Ltx2LatentVolume* slot_latents);

// The caller's frame contract (dfr_pipeline.py:531-540).
//
// The canvas may have padded its tail and each round maps `N -> 2(N - 1) + 1`,
// so what the caller is owed is `(requested - 1) * 2**rounds + 1`. `requested - 1`
// is a multiple of the VAE temporal scale, so the trim ALWAYS lands on a latent
// boundary — which is why this returns a latent-frame count rather than a pixel
// one and why a target ABOVE the canvas is a refusal rather than a pad.
int64_t Ltx2DfrTargetFrames(int64_t requested_frames, int64_t rounds);

}  // namespace vllm
