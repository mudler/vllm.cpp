// LTX-2.5 tiling algebra — see include/vllm/model_executor/models/ltx2_tiling.h
// for the port table and the three silent-failure notes. Ported 1:1 from
// Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca,
// packages/ltx-core/src/ltx_core/tiling.py and .../model/video_vae/video_vae.py.
#include "vllm/model_executor/models/ltx2_tiling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

namespace {

bool StartsWith(const std::string& value, const char* prefix) {
  const std::string p(prefix);
  return value.size() >= p.size() && value.compare(0, p.size(), p) == 0;
}

// `torch.linspace(start, stop, steps)` — the ENDPOINT-INCLUSIVE spacing, which is
// what makes `[:-1]` / `[1:-1]` drop exactly one sample per side upstream.
std::vector<double> Linspace(double start, double stop, int64_t steps) {
  std::vector<double> out(static_cast<size_t>(std::max<int64_t>(steps, 0)));
  if (steps <= 0) return out;
  if (steps == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(steps - 1);
  for (int64_t i = 0; i < steps; ++i) {
    out[static_cast<size_t>(i)] = start + step * static_cast<double>(i);
  }
  // torch pins the last sample to `stop` exactly; reproduce that rather than
  // letting the accumulated step decide it.
  out[static_cast<size_t>(steps - 1)] = stop;
  return out;
}

// `_grow_last_tile_to_min` (tiling.py:140-154).
std::vector<Ltx2DimensionInterval> GrowLastTileToMin(std::vector<Ltx2DimensionInterval> intervals,
                                                     int64_t min_tile_size) {
  if (intervals.size() <= 1) return intervals;
  Ltx2DimensionInterval& last = intervals.back();
  if (last.end - last.start >= min_tile_size) return intervals;
  const int64_t new_start = last.end - min_tile_size;
  Ltx2DimensionInterval& prev = intervals[intervals.size() - 2];
  const int64_t new_overlap = prev.end - new_start;
  prev.right_ramp = new_overlap;
  last.start = new_start;
  last.left_ramp = new_overlap;
  return intervals;
}

// `_validate_tile_intervals` (tiling.py:157-171). Upstream raises ValueError;
// VT_CHECK is this project's spelling of the same refusal.
void ValidateTileIntervals(const std::vector<Ltx2DimensionInterval>& intervals, int64_t dim_size,
                           int64_t min_tile_size) {
  VT_CHECK(!intervals.empty() && intervals.front().start == 0 && intervals.back().end == dim_size,
           "ltx2 tiling: tiles must cover [0, " + std::to_string(dim_size) + ")");
  for (size_t i = 0; i < intervals.size(); ++i) {
    const Ltx2DimensionInterval& iv = intervals[i];
    const int64_t length = iv.end - iv.start;
    VT_CHECK(length >= min_tile_size, "ltx2 tiling: tile " + std::to_string(i) + " length " +
                                          std::to_string(length) + " is below min_tile_size=" +
                                          std::to_string(min_tile_size));
    VT_CHECK(iv.left_ramp >= 0 && iv.right_ramp >= 0 && iv.left_ramp <= length &&
                 iv.right_ramp <= length,
             "ltx2 tiling: tile " + std::to_string(i) + " has invalid ramps: left=" +
                 std::to_string(iv.left_ramp) + ", right=" + std::to_string(iv.right_ramp) +
                 ", length=" + std::to_string(length));
    if (i == 0) continue;
    const int64_t overlap = intervals[i - 1].end - iv.start;
    VT_CHECK(overlap >= 0 && intervals[i - 1].right_ramp == overlap && iv.left_ramp == overlap,
             "ltx2 tiling: tiles " + std::to_string(i - 1) + "/" + std::to_string(i) +
                 ": ramp/overlap mismatch (overlap=" + std::to_string(overlap) + ")");
  }
}

// `default_split_operation` (tiling.py:114-115).
std::vector<Ltx2DimensionInterval> DefaultSplit(int64_t length) {
  return {Ltx2DimensionInterval{0, length, 0, 0}};
}

// `_validate_size_axis` (tiling.py:888-898).
void ValidateSizeAxis(const Ltx2DimensionSizeConfig& cfg, int64_t factor, const char* axis) {
  if (!cfg.IsTiled()) return;
  const int64_t min_size = 2 * factor;
  VT_CHECK(cfg.tile_size >= min_size,
           "ltx2 tiling: " + std::string(axis) + ".tile_size must be at least " +
               std::to_string(min_size) + ", got " + std::to_string(cfg.tile_size));
  VT_CHECK(cfg.tile_size % factor == 0,
           "ltx2 tiling: " + std::string(axis) + ".tile_size must be divisible by " +
               std::to_string(factor) + ", got " + std::to_string(cfg.tile_size));
  VT_CHECK(cfg.overlap % factor == 0,
           "ltx2 tiling: " + std::string(axis) + ".overlap must be divisible by " +
               std::to_string(factor) + ", got " + std::to_string(cfg.overlap));
}

// `DimensionSizeConfig.__post_init__` (tiling.py:631-641).
void ValidateSizeConfig(const Ltx2DimensionSizeConfig& cfg, const char* axis) {
  VT_CHECK(cfg.tile_size >= 0, "ltx2 tiling: " + std::string(axis) + ".tile_size must be >= 0");
  VT_CHECK(cfg.overlap >= 0, "ltx2 tiling: " + std::string(axis) + ".overlap must be >= 0");
  if (cfg.tile_size == 0) {
    VT_CHECK(cfg.overlap == 0, "ltx2 tiling: untiled axis (tile_size=0) must have overlap=0, but " +
                                   std::string(axis) + " has overlap " +
                                   std::to_string(cfg.overlap));
    return;
  }
  VT_CHECK(cfg.overlap < cfg.tile_size,
           "ltx2 tiling: overlap must be less than tile size, got " + std::to_string(cfg.overlap) +
               " and " + std::to_string(cfg.tile_size));
}

// `TileSizeConfig.to_splitters`' per-axis body (tiling.py:811-828), returning the
// resolved LATENT-grid intervals directly rather than a closure.
std::vector<Ltx2DimensionInterval> SplitAxis(const Ltx2DimensionSizeConfig& cfg, int64_t factor,
                                             int64_t latent_extent, const char* axis,
                                             bool temporal) {
  ValidateSizeConfig(cfg, axis);
  if (!cfg.IsTiled()) return DefaultSplit(latent_extent);
  ValidateSizeAxis(cfg, factor, axis);
  const int64_t size = cfg.tile_size / factor;
  const int64_t overlap = cfg.overlap / factor;
  const int64_t lower_threshold = std::max<int64_t>(2, overlap + 1);
  const int64_t tile = std::max<int64_t>(lower_threshold, size);
  if (temporal) return Ltx2SplitTemporalCausal(latent_extent, tile, overlap);
  return Ltx2SplitBySize(latent_extent, tile, overlap);
}

}  // namespace

Ltx2ScaleFactors Ltx2VideoScaleFactorsFromBlocks(const std::vector<Ltx2VideoDecoderBlock>& blocks,
                                                 int64_t patch_size) {
  // types.py:45-53. The `multiplier` is a CHANNEL knob and never a stride; the
  // block NAME alone decides which axes gain a factor of two.
  int64_t spatial_steps = 0;
  int64_t temporal_steps = 0;
  for (const Ltx2VideoDecoderBlock& block : blocks) {
    if (StartsWith(block.name, "compress_space") || StartsWith(block.name, "compress_all")) {
      ++spatial_steps;
    }
    if (StartsWith(block.name, "compress_time") || StartsWith(block.name, "compress_all")) {
      ++temporal_steps;
    }
  }
  Ltx2ScaleFactors out;
  out.time = static_cast<int64_t>(1) << temporal_steps;
  out.height = patch_size * (static_cast<int64_t>(1) << spatial_steps);
  out.width = out.height;
  return out;
}

std::vector<float> Ltx2TrapezoidalMask1d(int64_t length, int64_t ramp_left, int64_t ramp_right,
                                         bool left_starts_from_0) {
  VT_CHECK(length > 0, "ltx2 tiling: mask length must be positive");
  ramp_left = std::max<int64_t>(0, std::min<int64_t>(ramp_left, length));
  ramp_right = std::max<int64_t>(0, std::min<int64_t>(ramp_right, length));

  std::vector<float> mask(static_cast<size_t>(length), 1.0f);
  if (ramp_left > 0) {
    // tiling.py:39-43. `left_starts_from_0` KEEPS the leading zero by asking for
    // one fewer point and dropping only the trailing 1.0; otherwise the leading
    // zero is dropped too, so the first sample is already above zero.
    const int64_t interval_length = left_starts_from_0 ? ramp_left + 1 : ramp_left + 2;
    std::vector<double> fade = Linspace(0.0, 1.0, interval_length);
    fade.pop_back();  // [:-1]
    if (!left_starts_from_0) fade.erase(fade.begin());  // [1:]
    VT_CHECK(static_cast<int64_t>(fade.size()) == ramp_left,
             "ltx2 tiling: left fade length does not match the ramp");
    for (int64_t i = 0; i < ramp_left; ++i) {
      mask[static_cast<size_t>(i)] *= static_cast<float>(fade[static_cast<size_t>(i)]);
    }
  }
  if (ramp_right > 0) {
    // tiling.py:46-47: linspace(1, 0, ramp_right + 2)[1:-1] — both endpoints go.
    std::vector<double> fade = Linspace(1.0, 0.0, ramp_right + 2);
    fade.pop_back();
    fade.erase(fade.begin());
    VT_CHECK(static_cast<int64_t>(fade.size()) == ramp_right,
             "ltx2 tiling: right fade length does not match the ramp");
    for (int64_t i = 0; i < ramp_right; ++i) {
      mask[static_cast<size_t>(length - ramp_right + i)] *=
          static_cast<float>(fade[static_cast<size_t>(i)]);
    }
  }
  for (float& v : mask) v = std::min(1.0f, std::max(0.0f, v));
  return mask;
}

std::vector<Ltx2DimensionInterval> Ltx2SplitBySize(int64_t dimension_size, int64_t size,
                                                   int64_t overlap, int64_t min_tile_size) {
  VT_CHECK(size > 0, "ltx2 tiling: size must be > 0, got " + std::to_string(size));
  VT_CHECK(overlap >= 0 && overlap < size,
           "ltx2 tiling: overlap must satisfy 0 <= overlap < size, got overlap=" +
               std::to_string(overlap) + ", size=" + std::to_string(size));
  VT_CHECK(min_tile_size < 0 || min_tile_size >= 1,
           "ltx2 tiling: min_tile_size must be >= 1, got " + std::to_string(min_tile_size));

  const bool has_min = min_tile_size >= 0;
  if (has_min && dimension_size < min_tile_size) return DefaultSplit(dimension_size);
  // tiling.py:199-200. THE SHORT-CIRCUIT that makes upstream's default layout a
  // no-op below one tile — the reason 448x256/25f is untiled upstream too.
  if (dimension_size <= size) return DefaultSplit(dimension_size);

  const int64_t amount = (dimension_size + size - 2 * overlap - 1) / (size - overlap);
  // Upstream's list literal (tiling.py:202-216) is first + middle + last with no
  // guard, which is well defined only because `dimension_size > size` forces
  // `amount >= 2`. Asserting it here keeps that reasoning in the tree rather than
  // in a reviewer's head.
  VT_CHECK(amount >= 2, "ltx2 tiling: split_by_size reached the tiled branch with amount=" +
                            std::to_string(amount));
  std::vector<Ltx2DimensionInterval> intervals;
  intervals.push_back(Ltx2DimensionInterval{0, size, 0, overlap});
  for (int64_t i = 1; i < amount - 1; ++i) {
    intervals.push_back(Ltx2DimensionInterval{i * (size - overlap), i * (size - overlap) + size,
                                              overlap, overlap});
  }
  intervals.push_back(
      Ltx2DimensionInterval{(amount - 1) * (size - overlap), dimension_size, overlap, 0});
  if (has_min) {
    intervals = GrowLastTileToMin(std::move(intervals), min_tile_size);
    ValidateTileIntervals(intervals, dimension_size, min_tile_size);
  }
  return intervals;
}

std::vector<Ltx2DimensionInterval> Ltx2SplitTemporalCausal(int64_t dimension_size, int64_t size,
                                                           int64_t overlap,
                                                           int64_t min_tile_size) {
  // tiling.py:238-240: the causal arm short-circuits on its OWN `size` check
  // before delegating, so a `min_tile_size` larger than the dimension cannot
  // reach the non-causal branch's separate short-circuit.
  if (dimension_size <= size) return DefaultSplit(dimension_size);
  std::vector<Ltx2DimensionInterval> intervals =
      Ltx2SplitBySize(dimension_size, size, overlap, min_tile_size);
  if (intervals.size() <= 1) return intervals;
  for (size_t i = 1; i < intervals.size(); ++i) {
    intervals[i].start -= 1;
    intervals[i].left_ramp += 1;
  }
  return intervals;
}

std::vector<Ltx2DimensionInterval> Ltx2SplitByCount(int64_t dimension_size, int64_t num_tiles,
                                                    int64_t overlap, int64_t min_tile_size) {
  VT_CHECK(num_tiles >= 1, "ltx2 tiling: num_tiles must be >= 1, got " + std::to_string(num_tiles));
  VT_CHECK(overlap >= 0, "ltx2 tiling: overlap must be >= 0, got " + std::to_string(overlap));
  VT_CHECK(min_tile_size < 0 || min_tile_size >= 1,
           "ltx2 tiling: min_tile_size must be >= 1, got " + std::to_string(min_tile_size));
  VT_CHECK(num_tiles <= dimension_size,
           "ltx2 tiling: num_tiles (" + std::to_string(num_tiles) + ") exceeds dim_size (" +
               std::to_string(dimension_size) + "). Cannot assign at least 1 unit per tile.");
  if (num_tiles == 1) return DefaultSplit(dimension_size);

  const int64_t total = dimension_size + overlap * (num_tiles - 1);
  const int64_t tile_size = total / num_tiles;
  VT_CHECK(tile_size > overlap,
           "ltx2 tiling: split_by_count produced size=" + std::to_string(tile_size) +
               " <= overlap=" + std::to_string(overlap) +
               " for dim_size=" + std::to_string(dimension_size) +
               ", num_tiles=" + std::to_string(num_tiles));
  const int64_t remainder = total % num_tiles;

  std::vector<Ltx2DimensionInterval> base =
      Ltx2SplitBySize(dimension_size - remainder, tile_size, overlap);
  std::vector<Ltx2DimensionInterval> intervals;
  intervals.reserve(base.size());
  for (size_t i = 0; i < base.size(); ++i) {
    const int64_t shift = std::min<int64_t>(static_cast<int64_t>(i), remainder);
    const int64_t grow = static_cast<int64_t>(i) < remainder ? 1 : 0;
    Ltx2DimensionInterval iv = base[i];
    iv.start += shift;
    iv.end += shift + grow;
    intervals.push_back(iv);
  }
  if (min_tile_size >= 0) {
    intervals = GrowLastTileToMin(std::move(intervals), min_tile_size);
    ValidateTileIntervals(intervals, dimension_size, min_tile_size);
  }
  return intervals;
}

std::vector<Ltx2DimensionInterval> Ltx2SplitByCountTemporalCausal(int64_t dimension_size,
                                                                  int64_t num_tiles,
                                                                  int64_t overlap,
                                                                  int64_t min_tile_size) {
  std::vector<Ltx2DimensionInterval> intervals =
      Ltx2SplitByCount(dimension_size, num_tiles, overlap, min_tile_size);
  if (intervals.size() <= 1) return intervals;
  for (size_t i = 1; i < intervals.size(); ++i) {
    intervals[i].start -= 1;
    intervals[i].left_ramp += 1;
  }
  return intervals;
}

Ltx2AxisMapping Ltx2MapTemporalSlice(int64_t begin, int64_t end, int64_t left_ramp,
                                     int64_t right_ramp, int64_t scale) {
  // video_vae.py:549-555. The `1 +` is the frame the causal decoder keeps, and
  // `left_starts_from_0=true` is what makes the FIRST temporal tile's ramp begin
  // at zero rather than at the first non-zero sample.
  Ltx2AxisMapping out;
  out.start = begin * scale;
  out.stop = 1 + (end - 1) * scale;
  const int64_t mapped_left = left_ramp == 0 ? 0 : 1 + (left_ramp - 1) * scale;
  const int64_t mapped_right = right_ramp * scale;
  out.mask = Ltx2TrapezoidalMask1d(out.stop - out.start, mapped_left, mapped_right, true);
  return out;
}

Ltx2AxisMapping Ltx2MapSpatialSlice(int64_t begin, int64_t end, int64_t left_ramp,
                                    int64_t right_ramp, int64_t scale) {
  // video_vae.py:586-592.
  Ltx2AxisMapping out;
  out.start = begin * scale;
  out.stop = end * scale;
  out.mask = Ltx2TrapezoidalMask1d(out.stop - out.start, left_ramp * scale, right_ramp * scale,
                                   false);
  return out;
}

Ltx2TileSizeConfig Ltx2TileSizeConfig::Default() {
  Ltx2TileSizeConfig out;
  out.frames = Ltx2DimensionSizeConfig{80, 24};
  out.height = Ltx2DimensionSizeConfig{768, 64};
  out.width = Ltx2DimensionSizeConfig{768, 64};
  return out;
}

Ltx2TileSizeConfig Ltx2TileSizeConfig::FromLongSide(const Ltx2DimensionSizeConfig& long_side,
                                                    int64_t height, int64_t width,
                                                    const Ltx2ScaleFactors& factors,
                                                    const Ltx2DimensionSizeConfig& frames) {
  VT_CHECK(height >= 1 && width >= 1, "ltx2 tiling: height/width must be >= 1, got " +
                                          std::to_string(height) + "x" + std::to_string(width));
  VT_CHECK(long_side.IsTiled(), "ltx2 tiling: long_side must be tiled (tile_size > 0)");
  VT_CHECK(factors.height >= 1 && factors.width >= 1,
           "ltx2 tiling: scale_factors height/width must be >= 1");
  const int64_t span = std::max(height, width);

  // tiling.py:779-789. The round happens in LATENT units; a pixel-space round
  // plus a ceil-snap biases the short axis up by almost one latent cell.
  auto axis_size = [&](int64_t axis_len, int64_t factor) {
    const int64_t axis_lat = axis_len / factor;
    const int64_t long_lat = span / factor;
    const int64_t size_lat = long_side.tile_size / factor;
    const int64_t overlap_lat = long_side.overlap / factor;
    const int64_t lower_threshold = std::max<int64_t>(2, overlap_lat + 1);
    const double scaled = static_cast<double>(size_lat) * static_cast<double>(axis_lat) /
                          static_cast<double>(long_lat);
    // Python's `round` on a float is ROUND-HALF-TO-EVEN, and so is `nearbyint`
    // under the default FE_TONEAREST. `std::round` is NOT — it rounds halves away
    // from zero, and would put the short axis one latent cell wider than upstream
    // on every exact tie. The tie cases are swept in test_ltx2_tiling.
    const int64_t tile_lat =
        std::max<int64_t>(lower_threshold, static_cast<int64_t>(std::nearbyint(scaled)));
    const int64_t tile_px = tile_lat * factor;
    const int64_t min_legal = std::max<int64_t>(2 * factor, long_side.overlap + factor);
    return std::max(tile_px, min_legal);
  };

  Ltx2TileSizeConfig out;
  out.frames = frames;
  out.height = Ltx2DimensionSizeConfig{axis_size(height, factors.height), long_side.overlap};
  out.width = Ltx2DimensionSizeConfig{axis_size(width, factors.width), long_side.overlap};
  return out;
}

int64_t Ltx2TileSizeConfig::VideoChunksNumber(int64_t num_frames) const {
  // tiling.py:836-841.
  if (!frames.IsTiled()) return 1;
  const int64_t frame_stride = frames.tile_size - frames.overlap;
  return (num_frames - 1 + frame_stride - 1) / frame_stride;
}

Ltx2TileSizeConfig Ltx2AutoTileSizeConfig(int64_t height, int64_t width,
                                          const Ltx2ScaleFactors& factors) {
  // helpers.py:59-63, 80-88. These are upstream's numbers for a CONV VAE, not
  // this project's: `_CONV_AUTO_LONG_SIDE = 768/64`, `_CONV_AUTO_FRAMES = 80/24`.
  const Ltx2DimensionSizeConfig long_side{768, 64};
  const Ltx2DimensionSizeConfig frames{80, 24};
  return Ltx2TileSizeConfig::FromLongSide(long_side, height, width, factors, frames);
}

std::vector<Ltx2Tile> Ltx2CreateTiles(int64_t latent_t, int64_t latent_h, int64_t latent_w,
                                      const Ltx2TileSizeConfig& config,
                                      const Ltx2ScaleFactors& factors) {
  VT_CHECK(latent_t >= 1 && latent_h >= 1 && latent_w >= 1,
           "ltx2 tiling: the latent must be non-empty on every axis");
  // conv_video_decoder.py:364-381. The mapper for an axis is replaced exactly
  // when the CONFIG declares the axis tiled — NOT when the splitter happens to
  // produce more than one interval — so an axis with tile_size > 0 whose split
  // short-circuits still goes through the mapping op, and gets a concrete
  // out-slice covering the whole axis instead of the untiled `slice(0, None)`.
  const std::vector<Ltx2DimensionInterval> t_intervals =
      SplitAxis(config.frames, factors.time, latent_t, "frames", /*temporal=*/true);
  const std::vector<Ltx2DimensionInterval> h_intervals =
      SplitAxis(config.height, factors.height, latent_h, "height", /*temporal=*/false);
  const std::vector<Ltx2DimensionInterval> w_intervals =
      SplitAxis(config.width, factors.width, latent_w, "width", /*temporal=*/false);

  auto map_t = [&](const Ltx2DimensionInterval& iv) {
    if (!config.frames.IsTiled()) {
      Ltx2AxisMapping m;
      m.start = 0;
      m.stop = (latent_t - 1) * factors.time + 1;
      m.mask = {1.0f};
      return m;
    }
    return Ltx2MapTemporalSlice(iv.start, iv.end, iv.left_ramp, iv.right_ramp, factors.time);
  };
  auto map_s = [&](const Ltx2DimensionInterval& iv, bool is_h) {
    const int64_t scale = is_h ? factors.height : factors.width;
    if (!(is_h ? config.height.IsTiled() : config.width.IsTiled())) {
      Ltx2AxisMapping m;
      m.start = 0;
      m.stop = (is_h ? latent_h : latent_w) * scale;
      m.mask = {1.0f};
      return m;
    }
    return Ltx2MapSpatialSlice(iv.start, iv.end, iv.left_ramp, iv.right_ramp, scale);
  };

  // `itertools.product` with the TEMPORAL axis first (tiling.py:515-523), which
  // is the ordering `group_tiles_by_temporal_slice` documents its assumption on.
  std::vector<Ltx2Tile> tiles;
  tiles.reserve(t_intervals.size() * h_intervals.size() * w_intervals.size());
  for (const Ltx2DimensionInterval& ti : t_intervals) {
    for (const Ltx2DimensionInterval& hi : h_intervals) {
      for (const Ltx2DimensionInterval& wi : w_intervals) {
        Ltx2Tile tile;
        tile.in_t0 = ti.start;
        tile.in_t1 = ti.end;
        tile.in_h0 = hi.start;
        tile.in_h1 = hi.end;
        tile.in_w0 = wi.start;
        tile.in_w1 = wi.end;
        tile.out_t = map_t(ti);
        tile.out_h = map_s(hi, /*is_h=*/true);
        tile.out_w = map_s(wi, /*is_h=*/false);
        tiles.push_back(std::move(tile));
      }
    }
  }
  return tiles;
}

std::vector<std::vector<Ltx2Tile>> Ltx2GroupTilesByTemporalSlice(
    const std::vector<Ltx2Tile>& tiles) {
  // tiling.py:546-571. Consecutive-equal grouping, NOT a sort: it relies on the
  // temporal axis varying slowest, which Ltx2CreateTiles guarantees.
  std::vector<std::vector<Ltx2Tile>> groups;
  if (tiles.empty()) return groups;
  int64_t start = tiles.front().out_t.start;
  int64_t stop = tiles.front().out_t.stop;
  std::vector<Ltx2Tile> current;
  for (const Ltx2Tile& tile : tiles) {
    if (tile.out_t.start == start && tile.out_t.stop == stop) {
      current.push_back(tile);
    } else {
      groups.push_back(std::move(current));
      current.clear();
      start = tile.out_t.start;
      stop = tile.out_t.stop;
      current.push_back(tile);
    }
  }
  if (!current.empty()) groups.push_back(std::move(current));
  return groups;
}

bool Ltx2MasksAreComplementary(const std::vector<Ltx2Tile>& tiles, int64_t full_t, int64_t full_h,
                               int64_t full_w, double atol) {
  // tiling.py:438-472. PER AXIS, over the UNIQUE out-slices on that axis — a sum
  // over the cartesian product of tiles multi-counts the same 1-D interval and
  // would report "not complementary" for a layout that is.
  if (tiles.empty()) return true;
  const int64_t lengths[3] = {full_t, full_h, full_w};
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t length = lengths[axis];
    std::vector<double> acc(static_cast<size_t>(length), 0.0);
    std::vector<std::pair<int64_t, int64_t>> seen;
    for (const Ltx2Tile& tile : tiles) {
      const Ltx2AxisMapping& m =
          axis == 0 ? tile.out_t : (axis == 1 ? tile.out_h : tile.out_w);
      const std::pair<int64_t, int64_t> key{m.start, m.stop};
      if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
      seen.push_back(key);
      VT_CHECK(m.start >= 0 && m.stop <= length && m.stop >= m.start,
               "ltx2 tiling: an out-slice leaves the full extent");
      const int64_t span = m.stop - m.start;
      const bool broadcast = static_cast<int64_t>(m.mask.size()) == 1;
      VT_CHECK(broadcast || static_cast<int64_t>(m.mask.size()) == span,
               "ltx2 tiling: a 1-D mask does not match its out-slice");
      for (int64_t i = 0; i < span; ++i) {
        acc[static_cast<size_t>(m.start + i)] +=
            static_cast<double>(broadcast ? m.mask[0] : m.mask[static_cast<size_t>(i)]);
      }
    }
    for (double v : acc) {
      // torch.allclose(acc, ones, atol=atol, rtol=0.0).
      if (!(std::abs(v - 1.0) <= atol)) return false;
    }
  }
  return true;
}

std::vector<float> Ltx2ComputeSummedWeights(const std::vector<Ltx2Tile>& tiles, int64_t full_t,
                                            int64_t full_h, int64_t full_w) {
  // tiling.py:475-491. Separable per-axis broadcasts, never `Tile.blend_mask`.
  std::vector<float> weights(static_cast<size_t>(full_t * full_h * full_w), 0.0f);
  for (const Ltx2Tile& tile : tiles) {
    const auto value = [](const Ltx2AxisMapping& m, int64_t i) {
      return m.mask.size() == 1 ? m.mask[0] : m.mask[static_cast<size_t>(i)];
    };
    for (int64_t t = 0; t < tile.out_t.stop - tile.out_t.start; ++t) {
      const float mt = value(tile.out_t, t);
      for (int64_t h = 0; h < tile.out_h.stop - tile.out_h.start; ++h) {
        const float mh = value(tile.out_h, h);
        for (int64_t w = 0; w < tile.out_w.stop - tile.out_w.start; ++w) {
          const size_t idx = static_cast<size_t>(
              ((tile.out_t.start + t) * full_h + tile.out_h.start + h) * full_w +
              tile.out_w.start + w);
          weights[idx] += mt * mh * value(tile.out_w, w);
        }
      }
    }
  }
  for (float& v : weights) v = std::max(v, 1e-8f);
  return weights;
}

}  // namespace vllm
