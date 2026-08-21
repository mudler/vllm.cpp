// LTX-2.5 TILED + STREAMING VIDEO DECODE — the interval algebra, the separable
// trapezoidal blend, and the temporal-chunk streaming decode.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca,
// packages/ltx-core/src/ltx_core/
//   OURS                              <-  UPSTREAM
//   Ltx2TrapezoidalMask1d             <-  tiling.py:13-49
//   Ltx2DimensionInterval(s)          <-  tiling.py:77-96
//   Ltx2SplitBySize                   <-  tiling.py:174-222
//   Ltx2SplitTemporalCausal           <-  tiling.py:225-250
//   Ltx2SplitByCount                  <-  tiling.py:304-361
//   (last-tile growth + validation)   <-  tiling.py:140-171
//   Ltx2MapTemporalSlice              <-  video_vae.py:549-555
//   Ltx2MapSpatialSlice               <-  video_vae.py:586-592
//   Ltx2Tile / Ltx2CreateTiles        <-  tiling.py:382-401, 494-543
//   Ltx2GroupTilesByTemporalSlice     <-  tiling.py:546-571
//   Ltx2MasksAreComplementary         <-  tiling.py:438-472
//   Ltx2ComputeSummedWeights          <-  tiling.py:475-491
//   Ltx2ScaleByMasks1d                <-  tiling.py:423-435
//   Ltx2DimensionSizeConfig           <-  tiling.py:619-645
//   Ltx2TileSizeConfig                <-  tiling.py:715-841
//   Ltx2AutoTileSizeConfig            <-  ltx_pipelines/utils/helpers.py:59-88
//   Ltx2VideoScaleFactorsFromBlocks   <-  types.py:36-53
//   Ltx2ConvVideoDecodeTiled          <-  conv_video_decoder.py:383-484, 508-557
//   Ltx2ConvVideoDecodeStreaming      <-  conv_video_decoder.py:486-506
//
// ─── THE THREE THINGS THAT FAIL SILENTLY ─────────────────────────────────────
//  * THE RAMPS MUST PARTITION UNITY, and the check is PER AXIS over the UNIQUE
//    out-slices on that axis (tiling.py:438-472). Summing over the cartesian
//    product of tiles multi-counts the same 1-D interval and never reaches 1, so
//    a port that checks the product concludes "not complementary" on a layout
//    that is, allocates a denominator the size of the video, and divides by it —
//    numerically fine, which is exactly why nothing catches it.
//  * THE MASKS ARE SEPARABLE 1-D, NEVER A DENSE N-D MASK (tiling.py:423-435).
//    `Tile.blend_mask` (tiling.py:403-420) materializes the dense product and is
//    a DEBUGGING property; using it in the accumulate loop allocates a second
//    tensor the size of the tile for no numerical gain.
//  * THE TEMPORAL MAPPING IS NOT THE SPATIAL MAPPING. `map_temporal_slice` sends
//    [begin, end) to [begin*s, 1 + (end-1)*s) and passes left_starts_from_0=TRUE;
//    `map_spatial_slice` sends it to [begin*s, end*s) with FALSE. The `+1` is the
//    first frame the causal decoder keeps and the flag moves the ramp's first
//    sample by one (tiling.py:38-43). Swapping them shifts every temporal chunk
//    by a frame and puts a zero-weight column at the left edge of every spatial
//    tile — a seam, which a shape check cannot see.
//
// ─── WHEN THIS ACTUALLY BINDS, MEASURED ──────────────────────────────────────
// Upstream's own AUTO layout (768px long side / 64px overlap, 80 frames / 24
// overlap) is a NO-OP below 768px on the long side and below 81 frames, because
// `split_by_size` returns ONE interval when `dim <= size` (tiling.py:199-200).
// Executed at the pinned SHA, the first size that tiles at all is 896x512 (4
// spatial tiles, still ONE temporal group) and the first that CHUNKS temporally
// is **81 frames** — 81, because `latent_t = (frames - 1) / 8 + 1` reaches 11
// there and `split_temporal_causal` short-circuits only while `latent_t <= 10`
// (tiling.py:239-240). NOT 121, which this line used to say four lines under the
// paragraph above already saying 81: 121 is merely the first frame count the
// original probe sweep sampled above 25. The row's own golden `kLtx2AutoCases`
// carries `768x768/81f -> t_intervals = 2, chunks = 2`, and the sweep now walks
// the axis one latent frame at a time and asserts where the transition is.
//
// THE CONSEQUENCE IS USER-VISIBLE, so it is stated and not implied: 81..120
// frames IS the tiled regime. `docs/USAGE.md` records the LTX-2.5 recipe default
// as 1024x1536 at 121 frames, so an ordinary request lands inside it, and a
// tiled render is not the same image as an untiled one (upstream's behaviour,
// not a defect — see the equivalence note in tests/vllm/models/test_ltx2_tiling.cpp).
// Measured on the shipped checkpoint at 64x64/81f by
// scripts/probe_ltx2_tiled_equivalence.cpp: 2 chunks, max|diff| 0.0503043234
// against an output whose own |max| is 0.7512672544 — 6.70% of the output's own
// range — with 962983 of 995328 channel values (96.75%) not bit-identical. The
// ONE-TILE CONTROL therefore makes routing safe below 81 frames and NOT within
// 81..120.
//
// THAT NUMBER WAS 0.716 UNTIL IT WAS RE-DERIVED, and the correction is recorded
// because the failure mode is reusable. The probe's first version reassembled the
// streamed chunks by appending their buffers end to end, but a chunk is
// [C, t, H, W] CHANNEL-MAJOR, so a flat append is not [C, T, H, W] once C > 1 and
// there is more than one chunk — both true here. It was comparing channel 1
// against channel 0's later frames, and the number it published was 14x the real
// gap. Same run, same binary, corrected reassembly: 0.0503; the flat append the
// probe still prints as a labelled diagnostic reproduces 0.716 exactly. The
// QUALITATIVE conclusion is unchanged — 96.75% of values still move and the
// one-tile control still does not cover 81..120 — but the magnitude was wrong.
//
// At 448x256/25f the latent is 8x14 against a 14x24 grid tile and
// 4 frames against a 10-latent-frame temporal tile, so upstream calls `forward`
// once on the whole volume there — see .agents/specs/ltx25-tiled-decode.md §0.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// f32 throughout, for the same reason ltx2_video_vae.h:47-54 gives: this is the
// CPU REFERENCE arm. Upstream's tiled buffer inherits `latent.dtype`
// (conv_video_decoder.py:427-431) while the masks stay float32 so a bf16/fp16
// tile PROMOTES (tiling.py:425). When phase L6 lands the checkpoint-dtype arm the
// buffer follows the latent and the masks do not; recorded here so it does not
// have to be rediscovered.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// `Ltx2ScaleFactors` (SpatioTemporalScaleFactors, types.py:19-33) already lives
// here, defaulted to 8/32/32. It is REUSED rather than redeclared: a second
// spelling of the same three integers is exactly the parallel path AGENTS.md
// §"Shared seams" forbids, and the compiler caught the first draft doing it.
#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"

namespace vllm {

// `SpatioTemporalScaleFactors.from_blocks` (types.py:36-53). Each `compress_*`
// block contributes a stride-2 step on the axes its NAME starts with,
// INDEPENDENT of any channel `multiplier`, and the patchify contributes an extra
// `patch_size` of spatial scale. Deriving it from the blocks rather than
// hardcoding 8/32/32 is what keeps the 16x16x4 VAE variant correct.
Ltx2ScaleFactors Ltx2VideoScaleFactorsFromBlocks(const std::vector<Ltx2VideoDecoderBlock>& blocks,
                                                 int64_t patch_size);

// One interval of one axis (tiling.py:77-83). `[start, end)`; the ramps are the
// lengths of the regions blended with the neighbouring intervals.
struct Ltx2DimensionInterval {
  int64_t start = 0;
  int64_t end = 0;
  int64_t left_ramp = 0;
  int64_t right_ramp = 0;
};

// `compute_trapezoidal_mask_1d` (tiling.py:13-49). `left_starts_from_0` selects
// which endpoint of the linear ramp is dropped: TRUE keeps the leading 0 (the
// temporal mapper's choice, because the first temporal tile is causal), FALSE
// drops it so the ramp starts at the first non-zero value.
std::vector<float> Ltx2TrapezoidalMask1d(int64_t length, int64_t ramp_left, int64_t ramp_right,
                                         bool left_starts_from_0);

// `split_by_size` (tiling.py:174-222). `min_tile_size < 0` means "none" — the
// legacy short-last-tile behaviour; >= 1 grows a short last tile leftward,
// widens the penultimate right ramp, and VALIDATES coverage and ramp/overlap
// consistency, throwing on an illegal layout exactly as upstream raises.
std::vector<Ltx2DimensionInterval> Ltx2SplitBySize(int64_t dimension_size, int64_t size,
                                                   int64_t overlap, int64_t min_tile_size = -1);

// `split_temporal_causal` (tiling.py:225-250). Every tile after the first is
// shifted back by ONE and its left ramp grown by ONE, which is what keeps the
// blend continuous across a boundary the causal convolution cannot see past.
std::vector<Ltx2DimensionInterval> Ltx2SplitTemporalCausal(int64_t dimension_size, int64_t size,
                                                           int64_t overlap,
                                                           int64_t min_tile_size = -1);

// `split_by_count` (tiling.py:304-361). The MGPU / explicit-count arm. Tile size
// is `(dim + overlap*(n-1)) // n` and the first `remainder` tiles each absorb one
// extra unit, so the tiles cover the axis exactly.
std::vector<Ltx2DimensionInterval> Ltx2SplitByCount(int64_t dimension_size, int64_t num_tiles,
                                                    int64_t overlap, int64_t min_tile_size = -1);

// `split_by_count_temporal_causal` (tiling.py:275-301).
std::vector<Ltx2DimensionInterval> Ltx2SplitByCountTemporalCausal(int64_t dimension_size,
                                                                  int64_t num_tiles,
                                                                  int64_t overlap,
                                                                  int64_t min_tile_size = -1);

// A half-open output range together with the 1-D blend mask over it. A mask of
// LENGTH 1 is the untiled axis and broadcasts, exactly as upstream's
// `untiled_mask_1d` (tiling.py:121-123) does.
struct Ltx2AxisMapping {
  int64_t start = 0;
  int64_t stop = 0;
  std::vector<float> mask;
};

// `map_temporal_slice` (video_vae.py:549-555) and `map_spatial_slice` (:586-592).
Ltx2AxisMapping Ltx2MapTemporalSlice(int64_t begin, int64_t end, int64_t left_ramp,
                                     int64_t right_ramp, int64_t scale);
Ltx2AxisMapping Ltx2MapSpatialSlice(int64_t begin, int64_t end, int64_t left_ramp,
                                    int64_t right_ramp, int64_t scale);

// `Tile` (tiling.py:382-401), restricted to the three axes a video latent tiles.
// Batch and channel are never split, so upstream's length-1 masks on those axes
// carry no information and are not represented.
struct Ltx2Tile {
  // Where to cut this tile out of the LATENT.
  int64_t in_t0 = 0, in_t1 = 0, in_h0 = 0, in_h1 = 0, in_w0 = 0, in_w1 = 0;
  // Where its decoded output belongs in the PIXEL volume, with the 1-D masks.
  Ltx2AxisMapping out_t, out_h, out_w;
};

// One tiled axis of the layout (tiling.py:619-645). `tile_size == 0` means the
// axis is NOT tiled and must carry `overlap == 0`.
struct Ltx2DimensionSizeConfig {
  int64_t tile_size = 0;
  int64_t overlap = 0;
  bool IsTiled() const { return tile_size > 0; }
};

// `TileSizeConfig` (tiling.py:715-841): sizes are in PIXEL / FRAME units and are
// converted to the latent grid by `ToSplitters`, never stored pre-divided.
struct Ltx2TileSizeConfig {
  Ltx2DimensionSizeConfig frames;
  Ltx2DimensionSizeConfig height;
  Ltx2DimensionSizeConfig width;

  // `TileSizeConfig.default()` (tiling.py:746-752): 80/24 frames, 768/64 h and w.
  static Ltx2TileSizeConfig Default();

  // `TileSizeConfig.from_long_side` (tiling.py:754-795). The short axis is scaled
  // in LATENT units — `round(size_lat * axis_lat / long_lat)` — and multiplied
  // back by the VAE factor. Doing the round in PIXEL space and snapping up biases
  // the short axis by almost a whole latent cell (upstream's own note: 680 -> 704
  // instead of 672).
  static Ltx2TileSizeConfig FromLongSide(const Ltx2DimensionSizeConfig& long_side, int64_t height,
                                         int64_t width, const Ltx2ScaleFactors& factors,
                                         const Ltx2DimensionSizeConfig& frames);

  // `TileSizeConfig.video_chunks_number` (tiling.py:836-841).
  int64_t VideoChunksNumber(int64_t num_frames) const;
};

// `AUTO_TILING` resolved for the CONV VAE (helpers.py:59-88): the aspect-coupled
// 768/64 long side plus the 80/24 temporal chunk. This is upstream's DEFAULT for
// a Conv checkpoint, not a vllm.cpp invention, and it is what the pipeline uses.
Ltx2TileSizeConfig Ltx2AutoTileSizeConfig(int64_t height, int64_t width,
                                          const Ltx2ScaleFactors& factors);

// `ConvVideoDecoder._prepare_tiles` (conv_video_decoder.py:359-381) + `create_tiles`
// (tiling.py:526-543). Tiles come out with the TEMPORAL axis varying SLOWEST,
// which is what `Ltx2GroupTilesByTemporalSlice` relies on.
std::vector<Ltx2Tile> Ltx2CreateTiles(int64_t latent_t, int64_t latent_h, int64_t latent_w,
                                      const Ltx2TileSizeConfig& config,
                                      const Ltx2ScaleFactors& factors);

// `group_tiles_by_temporal_slice` (tiling.py:546-571). Consecutive tiles sharing
// a temporal out-slice form one group; each group is one streamed chunk.
std::vector<std::vector<Ltx2Tile>> Ltx2GroupTilesByTemporalSlice(
    const std::vector<Ltx2Tile>& tiles);

// `masks_are_complementary` (tiling.py:438-472). TRUE means weighted accumulation
// needs NO denominator, and upstream then allocates none at all.
bool Ltx2MasksAreComplementary(const std::vector<Ltx2Tile>& tiles, int64_t full_t, int64_t full_h,
                               int64_t full_w, double atol = 1e-5);

// `compute_summed_weights` (tiling.py:475-491), the fallback denominator for a
// layout that does NOT partition unity. Upstream forces it to CPU float32 so a
// CUDA mask cannot place a multi-GB [F, H, W] tensor on the device; here it is
// simply the host, and it is clamped to 1e-8 the same way.
std::vector<float> Ltx2ComputeSummedWeights(const std::vector<Ltx2Tile>& tiles, int64_t full_t,
                                            int64_t full_h, int64_t full_w);

// One streamed temporal chunk. `first_frame` is its index in the full clip, so a
// consumer can write frame files without counting.
struct Ltx2VideoChunk {
  int64_t first_frame = 0;
  Ltx2VideoFrames frames;
};

// The consumer. Upstream is a GENERATOR (conv_video_decoder.py:383-484) and the
// caller streams chunks straight into the muxer (ti2vid_two_stages.py:369-376);
// a callback is the C++ spelling of the same "yield and drop" contract, and it is
// the property that keeps peak memory at about two temporal chunks instead of the
// whole video. Returning is the only way to release the chunk — it is destroyed
// as soon as the callback returns.
using Ltx2VideoChunkSink = std::function<void(const Ltx2VideoChunk&)>;

// `ConvVideoDecoder.tiled_decode` (conv_video_decoder.py:383-484) together with
// `_accumulate_temporal_group_into_buffer` (:508-557).
//
// MEMORY, stated precisely rather than as a slogan. At more than one tile the
// full pixel volume is never materialized: at most two temporal chunk buffers are
// live at once, which is the whole point of the generator upstream. At exactly
// ONE tile and ONE group — every size below 896x512 / 81 frames — the chunk
// buffer IS the full volume, the decoded tile is another copy of it, and the
// callback's consumer may hold a third. That is not a regression (the untiled
// path allocates the volume too) and at 448x256/25f it is 34.4 MiB, but "never
// materialized" is false there and this row is about memory, so it says so.
//
// THE NOISE STREAM IS SHARED ACROSS TILES, AND ITS DRAW COUNT IS OWED. This calls
// `Ltx2ConvVideoDecode` once per tile with the SAME `Ltx2NoiseStream*`, exactly as
// upstream calls `self.forward` per tile (conv_video_decoder.py:529) with the same
// generator. So on a checkpoint with `timestep_conditioning = true`
// (ltx2_video_vae.h:175 defaults it true) or a block with `inject_noise` (:100),
// the tiled and untiled paths draw a DIFFERENT number of tensors in a different
// order — upstream's behaviour, mirrored, not a local divergence. It is inert on
// the shipped `ltx-2.5-video-vae-conv` checkpoint, whose config sets
// `timestep_conditioning: false` with no `inject_noise` block, and the golden
// fixtures disable it for the same reason (see the generator's docstring). What is
// OWED is a gate over a noise-drawing config; recorded in
// .agents/specs/ltx25-tiled-decode.md's "What is owed", not left to be found.
//
// REFUSES an UNTILED FRAMES AXIS (`tiling.frames.tile_size == 0`) by name.
// `create_tiles` accepts it and maps the axis through `DEFAULT_MAPPING_OPERATION`
// (tiling.py:126-132), but upstream's `tiled_decode` then subtracts the `None`
// stop of `slice(0, None)` (conv_video_decoder.py:424) and raises TypeError —
// measured over all eight (frames, height, width) x (tiled, untiled) combinations
// at the pin. Pass a frames tile_size larger than the clip for one chunk. The
// SPATIAL axes are untiled-legal and gated end to end.
void Ltx2ConvVideoDecodeTiled(const Ltx2ConvVideoDecoderConfig& config,
                              const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                              int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                              int64_t latent_w, Ltx2NoiseStream* noise,
                              const Ltx2TileSizeConfig& tiling, const Ltx2VideoChunkSink& emit,
                              const double* timestep = nullptr, vt::Queue* queue = nullptr);

// `ConvVideoDecoder.decode_video` (conv_video_decoder.py:486-506) minus the [0,1]
// rescale, which upstream does in `to_rgb` and this project's frame writer does
// itself. A tiling config that tiles NOTHING still streams — through exactly one
// chunk — so the caller has one code path, and that is upstream's shape too.
//
// REFUSES the diffusion decoder by name, never downgrading, for the reason
// Ltx2VideoDecode gives.
//
// `queue` is forwarded verbatim to every per-tile `Ltx2ConvVideoDecode`, and it
// is WHERE THE CONVOLUTION RUNS (#1007, LTX25-DEVICE-RESIDENCY W5). NULL means
// the CPU queue and is byte-identical to the pre-seam host arm; see
// ltx2_video_vae.h for what is and is not on the device.
void Ltx2VideoDecodeStreaming(Ltx2VideoDecoderKind kind,
                              const Ltx2ConvVideoDecoderConfig& config,
                              const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                              int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                              int64_t latent_w, Ltx2NoiseStream* noise,
                              const Ltx2TileSizeConfig& tiling, const Ltx2VideoChunkSink& emit,
                              const double* timestep = nullptr, vt::Queue* queue = nullptr);

}  // namespace vllm
