// LTX-2.5 TILED + STREAMING CONV VIDEO DECODE — `ConvVideoDecoder.tiled_decode`
// (conv_video_decoder.py:383-484) and `_accumulate_temporal_group_into_buffer`
// (:508-557), ported 1:1 from Lightricks/LTX-2 @
// fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca.
//
// It lives in its own translation unit and calls the PUBLIC `Ltx2ConvVideoDecode`
// once per tile, exactly as upstream calls `self.forward` per tile
// (conv_video_decoder.py:529). That is deliberate: the untiled decode stays the
// gated reference, and the tiled path cannot drift from it by sharing an internal
// helper that only one of them exercises.
//
// ─── WHY A CALLBACK AND NOT A RETURNED VOLUME ────────────────────────────────
// Upstream's `tiled_decode` is a GENERATOR: it buffers ONE temporal group, blends
// the overlap with the previous chunk, `yield`s the non-overlapping part and DROPS
// it (conv_video_decoder.py:466-476). Peak memory is about two temporal chunks,
// and the consumer streams chunks straight into the muxer
// (ti2vid_two_stages.py:369-376). Returning an `Ltx2VideoFrames` here would
// materialize exactly the tensor the generator exists to avoid, so the C++
// spelling is a sink that is called once per chunk and whose return releases it.
//
// "TWO CHUNKS, NEVER THE WHOLE VIDEO" IS A CLAIM ABOUT THE TILED CASE ONLY. At
// one tile and one temporal group — every size below 896x512 / 81 frames, which
// is every size this project has rendered — the single chunk IS the whole pixel
// volume, the decoded tile is a second copy of it, and the `Concat`/emit path can
// hold a third. 34.4 MiB at 448x256/25f, so the magnitude is immaterial and it is
// no worse than the untiled path it replaced; but a row about memory does not get
// to state the tiled-case bound as if it were unconditional.
//
// ─── THE OVERLAP BOOKKEEPING, WHICH IS THE PART THAT LOOKS WRONG AND IS NOT ──
// Every tile is accumulated into its group's buffer ALREADY MULTIPLIED by its
// trapezoidal mask (:552). So the temporal blend at :453 is a bare `+=` of two
// already-weighted halves, not a lerp — and it is correct precisely because the
// two ramps sum to one there. When they do not (`complementary == false`), the
// weights buffer accumulates the same masks and the quotient is taken at
// :470-471. Reading :453 as a missing blend and "fixing" it to an average is the
// mistake this paragraph exists to prevent.
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_tiling.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
// W0 of LTX25-DEVICE-RESIDENCY (#1010). The one include in this file that is not
// about decoding video, and it is here deliberately: `decode.video` is the leaf
// W5's lever is measured against, and every other assertion the gate makes about
// that leaf is a RATIO taken against the leaf itself. A sub-scope opened in the
// DRIVER, two statements from the leaf's own `Open`, moves with the leaf and
// constrains nothing; a sub-scope opened HERE, around the tile decode the render
// actually spends the seconds in, does not. `render_phase_log.h` is a
// process-wide instrument rather than a multimodal model, and it is the first
// thing under `multimodal/` this directory includes.
#include "vllm/multimodal/render_phase_log.h"
#include "vt/dtype.h"

namespace vllm {

namespace {

// A [C, T, H, W] pixel buffer for ONE temporal group, always rebased so its
// frame 0 is the group's first frame (conv_video_decoder.py:521-522).
struct ChunkBuffer {
  int64_t channels = 0, t = 0, h = 0, w = 0;
  std::vector<float> data;

  size_t At(int64_t c, int64_t ti, int64_t hi, int64_t wi) const {
    return static_cast<size_t>(((c * t + ti) * h + hi) * w + wi);
  }
  void Allocate(int64_t c, int64_t frames, int64_t height, int64_t width) {
    channels = c;
    t = frames;
    h = height;
    w = width;
    data.assign(static_cast<size_t>(c * frames * height * width), 0.0f);
  }
};

float MaskAt(const Ltx2AxisMapping& m, int64_t i) {
  // A length-1 mask is the untiled axis and BROADCASTS (tiling.py:121-123, 431-434).
  return m.mask.size() == 1 ? m.mask[0] : m.mask[static_cast<size_t>(i)];
}

// `_accumulate_temporal_group_into_buffer` (conv_video_decoder.py:508-557).
// Returns the weights buffer, or an empty vector when `complementary` — upstream
// returns None there and allocates nothing, which is the whole point of the
// complementarity check.
std::vector<float> AccumulateTemporalGroup(const Ltx2ConvVideoDecoderConfig& config,
                                           const Ltx2VaeWeights& weights,
                                           const std::vector<float>& latent,
                                           int64_t latent_channels, int64_t latent_t,
                                           int64_t latent_h, int64_t latent_w,
                                           Ltx2NoiseStream* noise, const double* timestep,
                                           const std::vector<Ltx2Tile>& group, ChunkBuffer* buffer,
                                           bool complementary, vt::Queue* queue) {
  const int64_t group_start = group.front().out_t.start;
  std::vector<float> group_weights;
  if (!complementary) {
    group_weights.assign(buffer->data.size(), 0.0f);
  }

  for (const Ltx2Tile& tile : group) {
    // `latent[tile.in_coords]` — the crop, channel-major like the source.
    const int64_t ct = tile.in_t1 - tile.in_t0;
    const int64_t ch = tile.in_h1 - tile.in_h0;
    const int64_t cw = tile.in_w1 - tile.in_w0;
    std::vector<float> crop(static_cast<size_t>(latent_channels * ct * ch * cw));
    for (int64_t c = 0; c < latent_channels; ++c) {
      for (int64_t ti = 0; ti < ct; ++ti) {
        for (int64_t hi = 0; hi < ch; ++hi) {
          const size_t src = static_cast<size_t>(
              (((c * latent_t) + tile.in_t0 + ti) * latent_h + tile.in_h0 + hi) * latent_w +
              tile.in_w0);
          const size_t dst =
              static_cast<size_t>((((c * ct) + ti) * ch + hi) * cw);
          std::copy(latent.begin() + static_cast<ptrdiff_t>(src),
                    latent.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(cw)),
                    crop.begin() + static_cast<ptrdiff_t>(dst));
        }
      }
    }

    const Ltx2VideoFrames decoded =
        Ltx2ConvVideoDecode(config, weights, crop, latent_channels, ct, ch, cw, noise, timestep,
                            queue);

    const int64_t temporal_offset = tile.out_t.start - group_start;
    // conv_video_decoder.py:533-537: the OUT-COORD length is authoritative, and
    // the decoder is allowed to return a different frame count — take the
    // smallest of the three rather than trusting either.
    const int64_t expected = tile.out_t.stop - tile.out_t.start;
    const int64_t actual =
        std::min({expected, decoded.frames, buffer->t - temporal_offset});
    VT_CHECK(actual > 0, "ltx2 tiled decode: a tile contributed no frames to its group buffer");
    VT_CHECK(decoded.channels == buffer->channels,
             "ltx2 tiled decode: a decoded tile's channel count does not match the chunk buffer");
    const int64_t tile_h = tile.out_h.stop - tile.out_h.start;
    const int64_t tile_w = tile.out_w.stop - tile.out_w.start;
    VT_CHECK(decoded.height == tile_h && decoded.width == tile_w,
             "ltx2 tiled decode: a decoded tile's spatial extent (" +
                 std::to_string(decoded.height) + "x" + std::to_string(decoded.width) +
                 ") does not match the out-slice it maps to (" + std::to_string(tile_h) + "x" +
                 std::to_string(tile_w) + ")");

    // `buffer[chunk_coords] += scale_by_masks_1d(decoded_slice, masks)`
    // (conv_video_decoder.py:552) — SEPARABLE 1-D masks, never a dense N-D one.
    for (int64_t c = 0; c < buffer->channels; ++c) {
      for (int64_t ti = 0; ti < actual; ++ti) {
        const float mt = MaskAt(tile.out_t, ti);
        for (int64_t hi = 0; hi < tile_h; ++hi) {
          const float mth = mt * MaskAt(tile.out_h, hi);
          const size_t dst_row =
              buffer->At(c, temporal_offset + ti, tile.out_h.start + hi, tile.out_w.start);
          const size_t src_row = static_cast<size_t>(
              ((c * decoded.frames + ti) * decoded.height + hi) * decoded.width);
          for (int64_t wi = 0; wi < tile_w; ++wi) {
            const float m = mth * MaskAt(tile.out_w, wi);
            buffer->data[dst_row + static_cast<size_t>(wi)] +=
                decoded.data[src_row + static_cast<size_t>(wi)] * m;
            if (!complementary) {
              // conv_video_decoder.py:553-555: the weights accumulate the SAME
              // masks over a tensor of ones. The value does not depend on the
              // channel, but upstream's weights buffer is `zeros_like(buffer)`
              // and therefore channel-shaped, so the divide at :471 is elementwise
              // and this must be too.
              group_weights[dst_row + static_cast<size_t>(wi)] += m;
            }
          }
        }
      }
    }
  }
  return group_weights;
}

}  // namespace

void Ltx2ConvVideoDecodeTiled(const Ltx2ConvVideoDecoderConfig& config,
                              const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                              int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                              int64_t latent_w, Ltx2NoiseStream* noise,
                              const Ltx2TileSizeConfig& tiling, const Ltx2VideoChunkSink& emit,
                              const double* timestep, vt::Queue* queue) {
  VT_CHECK(latent_channels == config.in_channels,
           "ltx2 tiled decode: latent channel count does not match in_channels");
  VT_CHECK(static_cast<int64_t>(latent.size()) == latent_channels * latent_t * latent_h * latent_w,
           "ltx2 tiled decode: latent size does not match [C, T, H, W]");
  VT_CHECK(static_cast<bool>(emit), "ltx2 tiled decode: a chunk sink is required");
  // An UNTILED FRAMES AXIS IS REFUSED BY NAME, because upstream cannot run it.
  //
  // `tile_size = 0` is legal to `DimensionSizeConfig` (tiling.py:619-645) and
  // `create_tiles` maps the axis through `DEFAULT_MAPPING_OPERATION`
  // (tiling.py:126-132), which returns `slice(0, None)`. `tiled_decode` then
  // computes `curr_temporal_slice.stop - curr_temporal_slice.start`
  // (conv_video_decoder.py:424) and raises `TypeError: unsupported operand
  // type(s) for -: 'NoneType' and 'int'`. Measured at the pinned SHA over all
  // eight (frames, height, width) x (tiled, untiled) combinations: every arm
  // with frames untiled RAISES, and every arm with frames tiled RUNS.
  //
  // "Runs" is the claim, and it is not "reproduces `forward` exactly". It
  // reproduces `forward` exactly only where the frames tile exceeds the axis, so
  // that ONE temporal group comes out — the one-tile control and the
  // untiled-spatial control, both of which the goldens pin at max|diff| == 0
  // (`kLtx2TileDec*UpstreamControlVsUntiled`,
  // `kLtx2TileDec*UpstreamUntiledSpatialVsUntiled`). Where the frames axis
  // actually splits, upstream's own tiled decode differs from its own `forward`
  // by a lot: `kLtx2TileDecCausalUpstreamTiledVsUntiled` is 2.13274002 against an
  // `OutputSpan` of 2.31735897, and the non-causal arm 2.07932711 against
  // 2.14835119. That is upstream's behaviour and no bound is claimed over it. The
  // gate for the refusal itself is `kLtx2TileDec*UpstreamUntiledFramesRaises`
  // together with the raising file, line and message the generator records.
  //
  // Inventing a concrete stop here would be a behaviour upstream never produces,
  // on a public signature, with no oracle — so it refuses instead. The SPATIAL
  // axes are untiled-legal and are gated end to end.
  VT_CHECK(tiling.frames.IsTiled(),
           "ltx2 tiled decode: the frames axis must be tiled (tile_size > 0); upstream's "
           "tiled_decode cannot run an untiled temporal axis, because "
           "DEFAULT_MAPPING_OPERATION hands it slice(0, None) and conv_video_decoder.py:424 "
           "subtracts its None stop. Pass a frames tile_size larger than the clip to get one "
           "temporal chunk.");

  const Ltx2ScaleFactors factors =
      Ltx2VideoScaleFactorsFromBlocks(config.decoder_blocks, config.patch_size);
  const std::vector<Ltx2Tile> tiles =
      Ltx2CreateTiles(latent_t, latent_h, latent_w, tiling, factors);
  VT_CHECK(!tiles.empty(), "ltx2 tiled decode: the tiling produced no tiles");

  // `VideoLatentShape.upscale` (types.py:125-131) — the FULL pixel extent, which
  // is only ever used to size ONE temporal chunk, never allocated whole.
  const int64_t full_t = (latent_t - 1) * factors.time + 1;
  const int64_t full_h = latent_h * factors.height;
  const int64_t full_w = latent_w * factors.width;
  const bool complementary = Ltx2MasksAreComplementary(tiles, full_t, full_h, full_w);
  const std::vector<std::vector<Ltx2Tile>> groups = Ltx2GroupTilesByTemporalSlice(tiles);

  ChunkBuffer previous;
  std::vector<float> previous_weights;
  int64_t previous_start = 0;
  int64_t previous_stop = 0;
  bool have_previous = false;

  // conv_video_decoder.py:415-476.
  for (const std::vector<Ltx2Tile>& group : groups) {
    const int64_t curr_start = group.front().out_t.start;
    const int64_t curr_stop = group.front().out_t.stop;

    ChunkBuffer buffer;
    buffer.Allocate(config.out_channels, curr_stop - curr_start, full_h, full_w);
    // W0 (#1010): THE VIDEO DECODE'S OWN WORK, bounded by production events on
    // both ends. One record per temporal group, which is one record per chunk
    // the sink is handed, so the count is a quantity the instrument cannot move.
    // Nested, so the table's sum does not change.
    //
    // THE PLACEMENT IS GATED BY A COVERAGE FLOOR AND NOT BY THE COUNT ALONE.
    // Moving this scope one statement up, onto `buffer.Allocate`, keeps the
    // count, the `nested` flag and the containment in a chunk window — and a
    // fourth fresh review measured what that costs: `decode.video.vae = 0.000 s`
    // beside a five-millisecond `decode.video`, with the tile decode inside no
    // sub-scope, both gates green. `test_ltx2_video`'s containment case now
    // requires these records to cover at least half of the render's
    // `decode.video.chunk` seconds (measured 91.1%-98.6%), so an anchor that
    // sits BESIDE the work rather than ON it is a red.
    std::vector<float> curr_weights;
    {
      const ::vllm::multimodal::phase::Scope vae_phase("decode.video.vae");
      curr_weights =
          AccumulateTemporalGroup(config, weights, latent, latent_channels, latent_t, latent_h,
                                  latent_w, noise, timestep, group, &buffer, complementary, queue);
    }

    if (have_previous) {
      if (previous_stop > curr_start) {
        const int64_t overlap_len = previous_stop - curr_start;
        const int64_t prev_offset = curr_start - previous_start;
        const int64_t plane = previous.h * previous.w;
        VT_CHECK(previous.h == buffer.h && previous.w == buffer.w &&
                     previous.channels == buffer.channels,
                 "ltx2 tiled decode: consecutive temporal chunks disagree on their frame shape");
        for (int64_t c = 0; c < buffer.channels; ++c) {
          for (int64_t f = 0; f < overlap_len; ++f) {
            const size_t p = previous.At(c, prev_offset + f, 0, 0);
            const size_t b = buffer.At(c, f, 0, 0);
            for (int64_t i = 0; i < plane; ++i) {
              // conv_video_decoder.py:453 — a bare `+=` of two ALREADY-MASKED
              // halves, not a lerp. See the header note.
              previous.data[p + static_cast<size_t>(i)] += buffer.data[b + static_cast<size_t>(i)];
            }
            if (!complementary) {
              for (int64_t i = 0; i < plane; ++i) {
                previous_weights[p + static_cast<size_t>(i)] +=
                    curr_weights[b + static_cast<size_t>(i)];
              }
            }
            for (int64_t i = 0; i < plane; ++i) {
              buffer.data[b + static_cast<size_t>(i)] = previous.data[p + static_cast<size_t>(i)];
            }
            if (!complementary) {
              for (int64_t i = 0; i < plane; ++i) {
                curr_weights[b + static_cast<size_t>(i)] =
                    previous_weights[p + static_cast<size_t>(i)];
              }
            }
          }
        }
      }

      // conv_video_decoder.py:466-471: yield only the frames the NEXT chunk will
      // not touch again.
      const int64_t yield_len = curr_start - previous_start;
      // NOT REACHABLE THROUGH Ltx2CreateTiles, and said plainly rather than left
      // to look like a gate. `SplitAxis` always sends the frames axis through
      // `Ltx2SplitTemporalCausal`, whose -1 shift makes tile i+1 start at
      // `end_i - overlap - 1` in LATENT units; mapped, that is
      // `curr_start = (e_i - overlap - 1) * scale` against
      // `previous_stop = 1 + (e_i - overlap - 1) * scale`, i.e. one frame of
      // overlap even at overlap 0. So no layout this file can build leaves a gap.
      //
      // It stays because upstream's own spelling silently TRUNCATES instead:
      // `previous_chunk[:, :, :yield_len]` is clamped by torch, so a splitter that
      // did leave a gap would return a clip with frames missing from the middle
      // and no diagnostic. `split_temporal` (tiling.py:253-272) is exactly such a
      // splitter — it forces `right_ramp = 0` — and it is the ENCODER's, not
      // ported here. If it or another arrives, this refuses instead of shipping a
      // short render.
      VT_CHECK(yield_len > 0 && yield_len <= previous.t,
               "ltx2 tiled decode: consecutive temporal groups leave a GAP — group at frame " +
                   std::to_string(previous_start) + " covers " + std::to_string(previous.t) +
                   " frames but the next starts at " + std::to_string(curr_start) +
                   ", so " + std::to_string(yield_len - previous.t) +
                   " frames are covered by no tile. Give the frames axis a non-zero overlap "
                   "(upstream's default is 24 frames); refusing rather than returning a clip "
                   "with frames missing from the middle");
      Ltx2VideoChunk chunk;
      chunk.first_frame = previous_start;
      chunk.frames.channels = previous.channels;
      chunk.frames.frames = yield_len;
      chunk.frames.height = previous.h;
      chunk.frames.width = previous.w;
      chunk.frames.data.resize(
          static_cast<size_t>(previous.channels * yield_len * previous.h * previous.w));
      for (int64_t c = 0; c < previous.channels; ++c) {
        for (int64_t f = 0; f < yield_len; ++f) {
          const size_t src = previous.At(c, f, 0, 0);
          const size_t dst = static_cast<size_t>((c * yield_len + f) * previous.h * previous.w);
          for (int64_t i = 0; i < previous.h * previous.w; ++i) {
            const float v = previous.data[src + static_cast<size_t>(i)];
            // conv_video_decoder.py:470 clamps the denominator to 1e-8 BEFORE
            // dividing; a tile corner where every ramp reaches zero would
            // otherwise divide by zero and emit NaN into the muxer.
            chunk.frames.data[dst + static_cast<size_t>(i)] =
                complementary
                    ? v
                    : v / std::max(previous_weights[src + static_cast<size_t>(i)], 1e-8f);
          }
        }
      }
      emit(chunk);
    }

    previous = std::move(buffer);
    previous_weights = std::move(curr_weights);
    previous_start = curr_start;
    previous_stop = curr_stop;
    have_previous = true;
  }

  // conv_video_decoder.py:478-484 — the tail.
  VT_CHECK(have_previous, "ltx2 tiled decode: no temporal group produced a chunk");
  Ltx2VideoChunk chunk;
  chunk.first_frame = previous_start;
  chunk.frames.channels = previous.channels;
  chunk.frames.frames = previous.t;
  chunk.frames.height = previous.h;
  chunk.frames.width = previous.w;
  chunk.frames.data.resize(previous.data.size());
  for (size_t i = 0; i < previous.data.size(); ++i) {
    chunk.frames.data[i] =
        complementary ? previous.data[i] : previous.data[i] / std::max(previous_weights[i], 1e-8f);
  }
  emit(chunk);
}

void Ltx2VideoDecodeStreaming(Ltx2VideoDecoderKind kind,
                              const Ltx2ConvVideoDecoderConfig& config,
                              const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                              int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                              int64_t latent_w, Ltx2NoiseStream* noise,
                              const Ltx2TileSizeConfig& tiling, const Ltx2VideoChunkSink& emit,
                              const double* timestep, vt::Queue* queue) {
  // The same refusal Ltx2VideoDecode makes, for the same reason: a silent
  // downgrade to the conv decoder returns a worse render as if it were the
  // requested one, and no gate this project owns can detect that.
  VT_CHECK(kind != Ltx2VideoDecoderKind::kDiffusion,
           "ltx2 video vae: this checkpoint asks for the DIFFUSION video decoder "
           "(NADiffusionDecoder / DiffusionVideoDecoder), which is NOT implemented — it needs a "
           "neighborhood-attention kernel and has its own row. It is refused rather than "
           "downgraded to the Conv video VAE, which would silently return a worse render");
  Ltx2ConvVideoDecodeTiled(config, weights, latent, latent_channels, latent_t, latent_h, latent_w,
                           noise, tiling, emit, timestep, queue);
}

}  // namespace vllm
