// LTX-2.5 TILED + STREAMING DECODE parity gate — the interval algebra, the
// separable trapezoidal blend, the AUTO layout, and `tiled_decode` itself,
// compared against the UPSTREAM `ltx_core` modules executed at reduced dimensions
// on CPU by scripts/gen-ltx2-tiling-goldens.py.
//
// ─── THE EQUIVALENCE CLAIM THIS SUITE DOES AND DOES NOT MAKE ─────────────────
// This row was dispatched with "tiled decode must be byte-equivalent (or
// bounded-equivalent) to untiled, and that equivalence IS the correctness gate".
// MEASURED, that property does not exist — not in this port and not upstream.
// Swept over tile sizes and both causality arms, upstream's own
// max|tiled - untiled| stays at 0.67-1.29 TIMES the whole output range and does
// not shrink as the tile grows, because the decoder's receptive field in LATENT
// units is wider than the overlap the layout uses (the shipped AUTO layout blends
// a 768px tile with a 64px overlap on a 32x grid — TWO latent cells). Upstream
// accepts that seam and blends it. Stating a bound for a quantity that is the size
// of the signal would be fabricating a gate, so this suite does not.
//
// What it gates instead, and what actually holds:
//
//   A. OUR tiled output == UPSTREAM's tiled output, to the same 5e-6 band every
//      other LTX-2.5 golden uses. That is the correctness gate.
//   B. The ONE-TILE CONTROL is EXACT. A tiling config whose splits all
//      short-circuit produces one tile, and `tiled_decode` then reproduces
//      `forward` BIT FOR BIT — upstream's own measured value is 0, emitted as a
//      golden rather than assumed. This is the property that makes routing the
//      pipeline through the streaming entry point safe at every size where the
//      AUTO layout does not tile, which today is every resolution this project
//      has run (see the 448x256/25f row of the AUTO sweep below). It is a bound,
//      not a blanket: the AUTO layout DOES tile at 896x512 and DOES chunk at
//      **81 frames**, so 81..120 frames — which contains the recipe default's
//      121 — is outside it, and a render there is a different image. Measured on
//      the shipped checkpoint in .agents/specs/ltx25-tiled-decode.md §0.
//   B'. The UNTILED-AXIS control is exact too, through the OTHER mapper branch
//      (`tile_size == 0` -> `DEFAULT_MAPPING_OPERATION`). Without it, zeroing the
//      broadcast mask renders a black clip and this whole suite stays green.
//   C. The tiled-vs-untiled gap is UPSTREAM's own number, held to the golden band.
//      A port that blends differently — a hard cut instead of a ramp, say — moves
//      A and C together, which is the mutation recorded in the spec.
//
// ─── WHAT THE FIXTURE IS NOT: IT CARRIES A `res_x_y`, THE SHIPPED LADDER DOES NOT
// `TilingFixtureConfig` below builds res_x / compress_all / **res_x_y** /
// compress_space / compress_time / res_x. `res_x_y` builds
// `norm3 = nn.GroupNorm(num_groups=1, ...)` (resnet.py:91-97) — a GLOBAL reduction
// over all of (C, T, H, W) — and the shipped `ltx-2.5-video-vae-conv` ladder has
// none (its blocks are res_x and compress_* only; see §0 of the row's spec, read
// from the checkpoint's own `__metadata__["config"]`).
//
// That matters for exactly one claim and no other: a decoder containing a global
// norm cannot have a local tiled approximation, so it INFLATES the tiled-vs-untiled
// gap (C) relative to the shipped ladder. It does not weaken A or B — our output
// is compared against upstream running the SAME fixture, and the one-tile and
// untiled-axis controls are exact on both. The res_x_y-free ladder is swept by
// scripts/probe_ltx2_tiled_vs_untiled_shipped_ladder.py and reaches the same
// conclusion, which is why the gap is reported as upstream's own number rather
// than bounded.
//
// Tolerances use `.scale(0.0)` nowhere because no doctest::Approx appears here;
// every comparison is an explicit max|diff| against a named band.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "support/max_abs_diff.h"
#include "vllm/model_executor/models/ltx2_tiling.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"

#include "ltx2_tiling_goldens.inc"

namespace {

using vllm_test::MaxAbsDiff;

// The band every LTX-2.5 tensor golden uses. Deliberately the SAME number as
// tests/vllm/models/test_ltx2_vae.cpp's kLtx2GoldenTol: these tensors come out of
// the same decoder through the same arithmetic, and a second, looser band for the
// tiled arm would let the tiled path drift from the untiled one it is supposed to
// be built out of.
constexpr double kLtx2GoldenTol = 5e-6;

// The upstream tree these goldens are only interpretable against. Advancing it is
// a deliberate edit HERE and in scripts/gen-ltx2-tiling-goldens.py, never a side
// effect of regenerating.
constexpr const char* kLtx2TilingUpstreamRevisionPin =
    "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca";

// ---------------------------------------------------------------------------
// The deterministic stream, mirroring scripts/gen-ltx2-vae-goldens.py :: ltx_rand
// exactly — a per-tensor FNV-1a seed plus a splitmix64 counter, so both sides
// build identical tensors from a NAME alone.
//
// This is the same stream tests/vllm/models/test_ltx2_vae.cpp implements, and it
// is duplicated rather than shared ON PURPOSE for now: that file is being edited
// concurrently by two other LTX-2.5 rows, and hoisting a 200-line block out of it
// is exactly the "two relocations auto-merge into a duplicate" hazard. Factoring
// both onto one fixture header is owed and is recorded in
// .agents/specs/ltx25-tiled-decode.md, not silently dropped.
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& name) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char byte : name) {
    h ^= static_cast<uint64_t>(byte);
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::vector<double> Ltx2Rand(const std::string& name, int64_t count) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<double> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    out[static_cast<size_t>(i)] = (static_cast<double>(u >> 11) * 0x1p-53) * 2.0 - 1.0;
  }
  return out;
}

std::vector<float> Ltx2Input(const std::string& name, int64_t count, double scale) {
  const std::vector<double> raw = Ltx2Rand(name, count);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(raw[static_cast<size_t>(i)] * scale);
  }
  return out;
}

bool EndsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The generator's `param_values` rule, restricted to the roles this fixture's
// decoder actually has. A role reached here that the generator spells differently
// shows up as a golden mismatch, not as a silent no-op.
std::vector<float> Ltx2Param(const std::string& name, const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;
  const size_t rank = shape.size();

  double scale = 0.1;
  double offset = 0.0;
  if (EndsWith(name, ".gamma")) {
    offset = 1.0;
  } else if (EndsWith(name, "std-of-means")) {
    offset = 1.0;
  } else if (EndsWith(name, ".bias")) {
    scale = 0.05;
  } else if (rank == 1 && EndsWith(name, ".weight")) {
    offset = 1.0;  // a 1-D `.weight` is an affine norm gain, initialized to ones
  }

  const std::vector<double> raw = Ltx2Rand(name, count);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] =
        static_cast<float>(raw[static_cast<size_t>(i)] * scale + offset);
  }
  return out;
}

struct ParamBag {
  vllm::Ltx2VaeWeights weights;
  std::vector<std::string> names;
  std::vector<int64_t> counts;

  void Put(const std::string& name, const std::vector<int64_t>& shape) {
    std::vector<float> values = Ltx2Param(name, shape);
    counts.push_back(static_cast<int64_t>(values.size()));
    names.push_back(name);
    weights.tensors[name] = std::move(values);
  }
};

void CheckManifest(const ParamBag& bag, const char* const* want_names, const int64_t* want_counts,
                   size_t want_size) {
  REQUIRE(bag.names.size() == want_size);
  REQUIRE(bag.counts.size() == want_size);
  for (size_t i = 0; i < want_size; ++i) {
    CHECK(bag.names[i] == std::string(want_names[i]));
    CHECK(bag.counts[i] == want_counts[i]);
  }
}

// scripts/gen-ltx2-tiling-goldens.py :: TILING_BLOCKS / TILING_DEC / TILING_LATENT.
// Two `compress_*` blocks touch space and two touch time, so with patch_size 2 the
// scale factors are time 4, height/width 8 (types.py:45-53) — which is what makes
// a 16px / 12-frame tile TWO latent cells and THREE latent frames.
vllm::Ltx2ConvVideoDecoderConfig TilingFixtureConfig(bool causal, const std::string& prefix) {
  vllm::Ltx2ConvVideoDecoderConfig cfg;
  cfg.prefix = prefix;
  cfg.in_channels = 6;
  cfg.out_channels = 3;
  cfg.patch_size = 2;
  cfg.base_channels = 8;
  cfg.causal = causal;
  cfg.timestep_conditioning = false;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kReflect;
  cfg.decoder_blocks = {
      {"res_x", 1, 0, false, false},
      {"compress_all", 1, 2, false, /*residual=*/true},
      {"res_x_y", 1, 2, false, false},
      {"compress_space", 1, 1, false, false},
      {"compress_time", 1, 1, false, false},
      {"res_x", 1, 0, false, false},
  };
  return cfg;
}

constexpr int64_t kLatentT = 5;
constexpr int64_t kLatentH = 4;
constexpr int64_t kLatentW = 4;

vllm::Ltx2TileSizeConfig FixtureTiling() {
  vllm::Ltx2TileSizeConfig cfg;
  cfg.frames = vllm::Ltx2DimensionSizeConfig{12, 4};
  cfg.height = vllm::Ltx2DimensionSizeConfig{16, 8};
  cfg.width = vllm::Ltx2DimensionSizeConfig{16, 8};
  return cfg;
}

// The generator's CONTROL_FRAMES / CONTROL_SPATIAL: every axis declares itself
// tiled, and every split short-circuits on `dim <= size` (tiling.py:199-200).
vllm::Ltx2TileSizeConfig ControlTiling() {
  vllm::Ltx2TileSizeConfig cfg;
  cfg.frames = vllm::Ltx2DimensionSizeConfig{10000, 0};
  cfg.height = vllm::Ltx2DimensionSizeConfig{10000, 0};
  cfg.width = vllm::Ltx2DimensionSizeConfig{10000, 0};
  return cfg;
}

// The generator's UNTILED_SPATIAL arm, and it is NOT the control above.
//
// `tile_size == 0` declares the axis UNTILED, so `Ltx2CreateTiles` replaces the
// real mapper with the broadcast one (`DEFAULT_MAPPING_OPERATION`,
// tiling.py:126-132). `tile_size == 10000` still declares the axis TILED and the
// split merely short-circuits, so the real mapper runs. Those are two different
// branches of `Ltx2CreateTiles`, and until this config existed nothing in the
// suite entered the untiled one — a mask of {0.0f} there multiplies the whole
// decoded volume by zero, i.e. renders a BLACK CLIP, and every assertion in this
// file still passed.
vllm::Ltx2TileSizeConfig UntiledSpatialTiling() {
  vllm::Ltx2TileSizeConfig cfg;
  cfg.frames = vllm::Ltx2DimensionSizeConfig{10000, 0};
  cfg.height = vllm::Ltx2DimensionSizeConfig{0, 0};
  cfg.width = vllm::Ltx2DimensionSizeConfig{0, 0};
  return cfg;
}

// All three axes untiled: legal to `create_tiles`, and FATAL to upstream's
// `tiled_decode`. See the refusal case below.
vllm::Ltx2TileSizeConfig UntiledAxesTiling() {
  vllm::Ltx2TileSizeConfig cfg;
  cfg.frames = vllm::Ltx2DimensionSizeConfig{0, 0};
  cfg.height = vllm::Ltx2DimensionSizeConfig{0, 0};
  cfg.width = vllm::Ltx2DimensionSizeConfig{0, 0};
  return cfg;
}

// The parameter bag for the fixture decoder, in torch's own state_dict order —
// _parameters before _modules, modules in registration order. The generator emits
// the same manifest, so an ordering error is a named failure.
ParamBag BuildFixtureParams(const vllm::Ltx2ConvVideoDecoderConfig& cfg) {
  ParamBag bag;
  // The SAME string the decoder will look its weights up under. Passing the
  // prefix separately is how the first draft of this file silently built a bag
  // the decoder could not read.
  const std::string p = cfg.prefix;

  int64_t multiplier = 1;
  for (const vllm::Ltx2VideoDecoderBlock& block : cfg.decoder_blocks) {
    if (block.name == "res_x_y") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 2;
    } else if (block.name != "res_x" && block.name != "attn") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 1;
    }
  }
  int64_t channels = cfg.base_channels * multiplier;

  bag.Put(p + "per_channel_statistics.std-of-means", {cfg.in_channels});
  bag.Put(p + "per_channel_statistics.mean-of-means", {cfg.in_channels});
  bag.Put(p + "conv_in.conv.weight", {channels, cfg.in_channels, 3, 3, 3});
  bag.Put(p + "conv_in.conv.bias", {channels});

  auto put_resnet3d = [&](const std::string& rp, int64_t in_ch, int64_t out_ch) {
    bag.Put(rp + ".conv1.conv.weight", {out_ch, in_ch, 3, 3, 3});
    bag.Put(rp + ".conv1.conv.bias", {out_ch});
    bag.Put(rp + ".conv2.conv.weight", {out_ch, out_ch, 3, 3, 3});
    bag.Put(rp + ".conv2.conv.bias", {out_ch});
    if (in_ch != out_ch) {
      bag.Put(rp + ".conv_shortcut.weight", {out_ch, in_ch, 1, 1, 1});
      bag.Put(rp + ".conv_shortcut.bias", {out_ch});
      bag.Put(rp + ".norm3.weight", {in_ch});
      bag.Put(rp + ".norm3.bias", {in_ch});
    }
  };

  int64_t index = 0;
  for (auto it = cfg.decoder_blocks.rbegin(); it != cfg.decoder_blocks.rend(); ++it, ++index) {
    const vllm::Ltx2VideoDecoderBlock& block = *it;
    const std::string bp = p + "up_blocks." + std::to_string(index);
    if (block.name == "res_x") {
      for (int64_t i = 0; i < block.num_layers; ++i) {
        put_resnet3d(bp + ".res_blocks." + std::to_string(i), channels, channels);
      }
    } else if (block.name == "res_x_y") {
      const int64_t out_ch = channels / (block.multiplier != 0 ? block.multiplier : 2);
      put_resnet3d(bp, channels, out_ch);
      channels = out_ch;
    } else {
      int64_t stride_product = 2;
      if (block.name == "compress_space") stride_product = 4;
      if (block.name == "compress_all") stride_product = 8;
      const int64_t reduction = block.multiplier != 0 ? block.multiplier : 1;
      bag.Put(bp + ".conv.conv.weight", {stride_product * channels / reduction, channels, 3, 3, 3});
      bag.Put(bp + ".conv.conv.bias", {stride_product * channels / reduction});
      channels /= reduction;
    }
  }

  bag.Put(p + "conv_out.conv.weight",
          {cfg.out_channels * cfg.patch_size * cfg.patch_size, channels, 3, 3, 3});
  bag.Put(p + "conv_out.conv.bias", {cfg.out_channels * cfg.patch_size * cfg.patch_size});
  return bag;
}

// The decoder never draws noise on this fixture (`timestep_conditioning` is off
// and no block sets `inject_noise`), which is what makes tiled and untiled
// comparable at all — see the generator's docstring. A stream that is REACHED is
// therefore a bug, and this one says so rather than returning zeros.
class NoNoise : public vllm::Ltx2NoiseStream {
 public:
  std::vector<float> Draw(int64_t count) override {
    FAIL("the tiling fixture drew " << count
                                    << " noise values; it is configured to draw none, and a "
                                       "tiled/untiled comparison across differing draw counts "
                                       "would be meaningless");
    return std::vector<float>(static_cast<size_t>(count), 0.0f);
  }
};

// Collect every streamed chunk back into one volume, purely so the test can
// compare against a golden. The PRODUCT path never does this — that is the whole
// point of the sink — so the concatenation lives here and not in src/.
struct Collected {
  std::vector<vllm::Ltx2VideoChunk> chunks;
  vllm::Ltx2VideoFrames Concat() const {
    vllm::Ltx2VideoFrames out;
    REQUIRE(!chunks.empty());
    out.channels = chunks.front().frames.channels;
    out.height = chunks.front().frames.height;
    out.width = chunks.front().frames.width;
    out.frames = 0;
    for (const vllm::Ltx2VideoChunk& c : chunks) out.frames += c.frames.frames;
    out.data.resize(static_cast<size_t>(out.channels * out.frames * out.height * out.width));
    const int64_t plane = out.height * out.width;
    int64_t written = 0;
    for (const vllm::Ltx2VideoChunk& c : chunks) {
      for (int64_t ch = 0; ch < out.channels; ++ch) {
        for (int64_t f = 0; f < c.frames.frames; ++f) {
          const size_t src = static_cast<size_t>((ch * c.frames.frames + f) * plane);
          const size_t dst = static_cast<size_t>((ch * out.frames + written + f) * plane);
          std::copy(c.frames.data.begin() + static_cast<ptrdiff_t>(src),
                    c.frames.data.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(plane)),
                    out.data.begin() + static_cast<ptrdiff_t>(dst));
        }
      }
      written += c.frames.frames;
    }
    return out;
  }
};

}  // namespace

TEST_CASE("ltx2 tiling: the goldens name the upstream tree they came from") {
  // A golden without an anchor cannot be bisected, and a regeneration against a
  // different checkout would silently replace the oracle.
  CHECK(std::string(vllm_test::kLtx2TilingUpstreamRevision) ==
        std::string(kLtx2TilingUpstreamRevisionPin));
}

TEST_CASE("ltx2 tiling: the trapezoidal blend mask matches upstream") {
  size_t cursor = 0;
  size_t golden = 0;
  for (int64_t c = 0; c < vllm_test::kLtx2MaskCaseCount; ++c) {
    const int64_t length = vllm_test::kLtx2MaskCaseParams[cursor + 0];
    const int64_t left = vllm_test::kLtx2MaskCaseParams[cursor + 1];
    const int64_t right = vllm_test::kLtx2MaskCaseParams[cursor + 2];
    const bool from0 = vllm_test::kLtx2MaskCaseParams[cursor + 3] != 0;
    cursor += 4;
    const std::vector<float> mask = vllm::Ltx2TrapezoidalMask1d(length, left, right, from0);
    REQUIRE(static_cast<int64_t>(mask.size()) == length);
    INFO("mask case length=" << length << " left=" << left << " right=" << right
                             << " left_starts_from_0=" << from0);
    const double err =
        MaxAbsDiff(mask, vllm_test::kLtx2MaskGolden + golden, static_cast<size_t>(length));
    CHECK(err <= kLtx2GoldenTol);
    golden += static_cast<size_t>(length);
  }
  CHECK(golden == std::size(vllm_test::kLtx2MaskGolden));
}

TEST_CASE("ltx2 tiling: the split operations match upstream") {
  size_t flat = 0;
  for (size_t c = 0; c < std::size(vllm_test::kLtx2SplitLabels); ++c) {
    const std::string label = vllm_test::kLtx2SplitLabels[c];
    // "kind:dim:a:b[:min]" — parsed rather than duplicated, so the label a human
    // reads on failure is the same string that selected the call.
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
      const size_t colon = label.find(':', start);
      parts.push_back(label.substr(start, colon == std::string::npos ? colon : colon - start));
      if (colon == std::string::npos) break;
      start = colon + 1;
    }
    REQUIRE(parts.size() >= 4);
    const int64_t dim = std::stoll(parts[1]);
    const int64_t a = std::stoll(parts[2]);
    const int64_t b = std::stoll(parts[3]);
    const int64_t min_tile = parts.size() >= 5 ? std::stoll(parts[4]) : -1;

    std::vector<vllm::Ltx2DimensionInterval> got;
    if (parts[0] == "by_size") {
      got = vllm::Ltx2SplitBySize(dim, a, b);
    } else if (parts[0] == "by_size_min") {
      got = vllm::Ltx2SplitBySize(dim, a, b, min_tile);
    } else if (parts[0] == "temporal_causal") {
      got = vllm::Ltx2SplitTemporalCausal(dim, a, b);
    } else if (parts[0] == "by_count") {
      got = vllm::Ltx2SplitByCount(dim, a, b);
    } else if (parts[0] == "by_count_causal") {
      got = vllm::Ltx2SplitByCountTemporalCausal(dim, a, b);
    } else {
      FAIL("unhandled split kind in golden label: " << label);
    }

    INFO("split case " << label);
    REQUIRE(static_cast<int64_t>(got.size()) == vllm_test::kLtx2SplitCounts[c]);
    for (const vllm::Ltx2DimensionInterval& iv : got) {
      CHECK(iv.start == vllm_test::kLtx2SplitFlat[flat + 0]);
      CHECK(iv.end == vllm_test::kLtx2SplitFlat[flat + 1]);
      CHECK(iv.left_ramp == vllm_test::kLtx2SplitFlat[flat + 2]);
      CHECK(iv.right_ramp == vllm_test::kLtx2SplitFlat[flat + 3]);
      flat += 4;
    }
  }
  CHECK(flat == std::size(vllm_test::kLtx2SplitFlat));
}

TEST_CASE("ltx2 tiling: the CONV AUTO layout matches upstream, and is a NO-OP below one tile") {
  const vllm::Ltx2ScaleFactors factors{8, 32, 32};
  bool saw_untiled_448 = false;
  bool saw_tiled_somewhere = false;
  for (int64_t c = 0; c < vllm_test::kLtx2AutoCaseCount; ++c) {
    const int64_t* row = vllm_test::kLtx2AutoCases + c * vllm_test::kLtx2AutoCaseStride;
    const int64_t width = row[0], height = row[1], frames = row[2];
    INFO("auto layout case " << width << "x" << height << "/" << frames << "f");
    const vllm::Ltx2TileSizeConfig cfg = vllm::Ltx2AutoTileSizeConfig(height, width, factors);
    CHECK(cfg.height.tile_size == row[3]);
    CHECK(cfg.height.overlap == row[4]);
    CHECK(cfg.width.tile_size == row[5]);
    CHECK(cfg.width.overlap == row[6]);
    CHECK(cfg.frames.tile_size == row[7]);
    CHECK(cfg.frames.overlap == row[8]);

    const int64_t latent_t = (frames - 1) / factors.time + 1;
    const int64_t latent_h = height / factors.height;
    const int64_t latent_w = width / factors.width;
    const std::vector<vllm::Ltx2Tile> tiles =
        vllm::Ltx2CreateTiles(latent_t, latent_h, latent_w, cfg, factors);
    const std::vector<std::vector<vllm::Ltx2Tile>> groups =
        vllm::Ltx2GroupTilesByTemporalSlice(tiles);
    CHECK(static_cast<int64_t>(groups.size()) == row[9]);
    CHECK(static_cast<int64_t>(tiles.size()) == row[9] * row[10] * row[11]);
    CHECK(cfg.VideoChunksNumber(frames) == row[12]);

    // Every layout the AUTO path produces must partition unity, or upstream would
    // be allocating a denominator the size of the video on its own defaults.
    const int64_t full_t = (latent_t - 1) * factors.time + 1;
    CHECK(vllm::Ltx2MasksAreComplementary(tiles, full_t, latent_h * factors.height,
                                          latent_w * factors.width));

    if (width == 448 && height == 256 && frames == 25) {
      // THE ROW THIS WHOLE SPEC TURNS ON. Upstream's own default resolves to ONE
      // tile here, so the size that was reported as failing for want of tiling is
      // a size upstream does not tile either.
      CHECK(tiles.size() == 1u);
      CHECK(groups.size() == 1u);
      saw_untiled_448 = true;
    }
    if (tiles.size() > 1u) saw_tiled_somewhere = true;
  }
  // Guard the guard: an assertion that never sees a tiled case would be true by
  // construction, and so would one that never sees the 448x256 row.
  CHECK(saw_untiled_448);
  CHECK(saw_tiled_somewhere);

  // AND THE AUTO LAYOUT NEVER DECLARES AN AXIS UNTILED, which is why the
  // untiled-mapper branch below needs a case of its own rather than a resolution.
  for (int64_t c = 0; c < vllm_test::kLtx2AutoCaseCount; ++c) {
    const int64_t* row = vllm_test::kLtx2AutoCases + c * vllm_test::kLtx2AutoCaseStride;
    const vllm::Ltx2TileSizeConfig cfg = vllm::Ltx2AutoTileSizeConfig(row[1], row[0], factors);
    CHECK(cfg.frames.IsTiled());
    CHECK(cfg.height.IsTiled());
    CHECK(cfg.width.IsTiled());
  }
}

TEST_CASE("ltx2 tiling: an UNTILED axis maps through the broadcast mask, not the ramp") {
  // Section 3b of the goldens: upstream's `DEFAULT_MAPPING_OPERATION`
  // (tiling.py:126-132), executed on an all-untiled `TileSizeConfig`. Its
  // out-slice is `slice(0, None)` and its mask is `untiled_mask_1d()` — length-1
  // ones that BROADCAST (tiling.py:121-123).
  //
  // This is the arm the shipped `Ltx2AutoTileSizeConfig` never reaches and that
  // `ControlTiling` (10000/0) does not reach either, because a huge tile still
  // declares the axis tiled. A `{0.0f}` mask here is a black clip.
  const vllm::Ltx2ConvVideoDecoderConfig cfg =
      TilingFixtureConfig(/*causal=*/false, "ltx2.tiledec.noncausal.");
  const vllm::Ltx2ScaleFactors factors =
      vllm::Ltx2VideoScaleFactorsFromBlocks(cfg.decoder_blocks, cfg.patch_size);

  const std::vector<vllm::Ltx2Tile> tiles =
      vllm::Ltx2CreateTiles(kLatentT, kLatentH, kLatentW, UntiledAxesTiling(), factors);
  REQUIRE(static_cast<int64_t>(tiles.size()) == vllm_test::kLtx2UntiledMapTileCount);

  // Upstream's `None` stop has no integer to compare against; -1 is emitted for
  // it and the concrete stop this port computes must be the FULL axis extent,
  // which is what `slice(0, None)` means for a tensor of that shape.
  const int64_t full[3] = {(kLatentT - 1) * factors.time + 1, kLatentH * factors.height,
                           kLatentW * factors.width};
  const vllm::Ltx2AxisMapping* axes[3] = {&tiles[0].out_t, &tiles[0].out_h, &tiles[0].out_w};
  const char* names[3] = {"frames", "height", "width"};
  for (int a = 0; a < 3; ++a) {
    INFO("untiled axis " << names[a]);
    CHECK(axes[a]->start == vllm_test::kLtx2UntiledMapStart[a]);
    REQUIRE(vllm_test::kLtx2UntiledMapStop[a] == -1);
    CHECK(axes[a]->stop == full[a]);
    CHECK(static_cast<int64_t>(axes[a]->mask.size()) == vllm_test::kLtx2UntiledMapMaskLen[a]);
    REQUIRE(!axes[a]->mask.empty());
    CHECK(axes[a]->mask[0] == vllm_test::kLtx2UntiledMapMaskValue[a]);
  }

  // Guard the guard: the golden must be the BROADCAST value, not zero, or the
  // three assertions above would accept the black-clip mutation they exist to
  // catch by simply moving with it.
  for (int a = 0; a < 3; ++a) {
    CHECK(vllm_test::kLtx2UntiledMapMaskLen[a] == 1);
    CHECK(vllm_test::kLtx2UntiledMapMaskValue[a] == 1.0f);
  }

  // And it is a DIFFERENT branch from the 10000/0 control: that one declares the
  // axis tiled, so the real mapper runs and the mask is a full-length ramp.
  const std::vector<vllm::Ltx2Tile> control =
      vllm::Ltx2CreateTiles(kLatentT, kLatentH, kLatentW, ControlTiling(), factors);
  REQUIRE(control.size() == 1u);
  CHECK(control[0].out_h.mask.size() > 1u);
  CHECK(control[0].out_w.mask.size() > 1u);
  CHECK(control[0].out_t.mask.size() > 1u);
}

TEST_CASE("ltx2 tiling: the temporal and spatial axis mappers are NOT interchangeable") {
  size_t t_mask = 0;
  size_t s_mask = 0;
  for (int64_t c = 0; c < vllm_test::kLtx2MapCaseCount; ++c) {
    const int64_t* p = vllm_test::kLtx2MapCaseParams + c * 5;
    INFO("mapper case begin=" << p[0] << " end=" << p[1] << " left=" << p[2] << " right=" << p[3]
                              << " scale=" << p[4]);
    const vllm::Ltx2AxisMapping t = vllm::Ltx2MapTemporalSlice(p[0], p[1], p[2], p[3], p[4]);
    CHECK(t.start == vllm_test::kLtx2MapTemporalBounds[2 * c + 0]);
    CHECK(t.stop == vllm_test::kLtx2MapTemporalBounds[2 * c + 1]);
    CHECK(MaxAbsDiff(t.mask, vllm_test::kLtx2MapTemporalMasks + t_mask, t.mask.size()) <=
          kLtx2GoldenTol);
    t_mask += t.mask.size();

    const vllm::Ltx2AxisMapping s = vllm::Ltx2MapSpatialSlice(p[0], p[1], p[2], p[3], p[4]);
    CHECK(s.start == vllm_test::kLtx2MapSpatialBounds[2 * c + 0]);
    CHECK(s.stop == vllm_test::kLtx2MapSpatialBounds[2 * c + 1]);
    CHECK(MaxAbsDiff(s.mask, vllm_test::kLtx2MapSpatialMasks + s_mask, s.mask.size()) <=
          kLtx2GoldenTol);
    s_mask += s.mask.size();
  }
  CHECK(t_mask == std::size(vllm_test::kLtx2MapTemporalMasks));
  CHECK(s_mask == std::size(vllm_test::kLtx2MapSpatialMasks));

  // And they DIFFER where it matters: same interval, same scale, different span
  // and a different first sample. If a port ever routed one through the other,
  // every assertion above would still be checkable — this one names the reason.
  const vllm::Ltx2AxisMapping t = vllm::Ltx2MapTemporalSlice(1, 5, 2, 0, 4);
  const vllm::Ltx2AxisMapping s = vllm::Ltx2MapSpatialSlice(1, 5, 2, 0, 4);
  CHECK(t.stop - t.start == 13);  // 1 + (5-1)*4 - 1*4
  CHECK(s.stop - s.start == 16);  // 5*4 - 1*4
  CHECK(t.mask.front() == 0.0f);  // left_starts_from_0 = true keeps the leading zero
  CHECK(s.mask.front() > 0.0f);   // ...and false drops it
}

TEST_CASE("ltx2 tiling: complementarity decides whether a denominator is allocated") {
  const vllm::Ltx2ConvVideoDecoderConfig cfg =
      TilingFixtureConfig(/*causal=*/true, "ltx2.tiledec.causal.");
  const vllm::Ltx2ScaleFactors factors =
      vllm::Ltx2VideoScaleFactorsFromBlocks(cfg.decoder_blocks, cfg.patch_size);
  CHECK(factors.time == 4);
  CHECK(factors.height == 8);
  CHECK(factors.width == 8);

  const std::vector<vllm::Ltx2Tile> tiles =
      vllm::Ltx2CreateTiles(kLatentT, kLatentH, kLatentW, FixtureTiling(), factors);
  const bool complementary = vllm::Ltx2MasksAreComplementary(
      tiles, (kLatentT - 1) * factors.time + 1, kLatentH * factors.height,
      kLatentW * factors.width);
  CHECK(complementary == (vllm_test::kLtx2ComplementaryShipped != 0));

  // The FALLBACK arm, which the shipped layouts never reach and which would
  // therefore rot unpinned: two length-4 out-slices overlapping by 2, both masks
  // all ones, so the overlap sums to 2 and a denominator IS needed.
  vllm::Ltx2Tile a;
  a.in_t0 = 0;
  a.in_t1 = 4;
  a.in_h0 = 0;
  a.in_h1 = 2;
  a.in_w0 = 0;
  a.in_w1 = 2;
  a.out_t = vllm::Ltx2AxisMapping{0, 4, std::vector<float>(4, 1.0f)};
  a.out_h = vllm::Ltx2AxisMapping{0, 2, std::vector<float>(2, 1.0f)};
  a.out_w = vllm::Ltx2AxisMapping{0, 2, std::vector<float>(2, 1.0f)};
  vllm::Ltx2Tile b = a;
  b.in_t0 = 2;
  b.in_t1 = 6;
  b.out_t = vllm::Ltx2AxisMapping{2, 6, std::vector<float>(4, 1.0f)};
  const std::vector<vllm::Ltx2Tile> bad{a, b};

  CHECK(vllm::Ltx2MasksAreComplementary(bad, 6, 2, 2) ==
        (vllm_test::kLtx2ComplementaryBad != 0));
  const std::vector<float> weights = vllm::Ltx2ComputeSummedWeights(bad, 6, 2, 2);
  // Upstream's denominator is `torch.zeros(*full_shape)` (tiling.py:485), i.e.
  // CHANNEL-SHAPED [B, C, T, H, W], even though no mask depends on the channel.
  // Ours is [T, H, W] because the blend never does either. So the golden is
  // checked BOTH ways: our plane against channel 0, and every other channel plane
  // of the golden against channel 0 — which is what proves dropping the axis is
  // lossless rather than merely convenient.
  const int64_t plane = vllm_test::kLtx2SummedWeightsShape[2] *
                        vllm_test::kLtx2SummedWeightsShape[3] *
                        vllm_test::kLtx2SummedWeightsShape[4];
  REQUIRE(static_cast<int64_t>(weights.size()) == plane);
  REQUIRE(static_cast<int64_t>(std::size(vllm_test::kLtx2SummedWeights)) ==
          vllm_test::kLtx2SummedWeightsShape[0] * vllm_test::kLtx2SummedWeightsShape[1] * plane);
  const double err =
      MaxAbsDiff(weights, vllm_test::kLtx2SummedWeights, static_cast<size_t>(plane));
  INFO("summed weights max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
  for (int64_t c = 1; c < vllm_test::kLtx2SummedWeightsShape[1]; ++c) {
    for (int64_t i = 0; i < plane; ++i) {
      CHECK(vllm_test::kLtx2SummedWeights[c * plane + i] == vllm_test::kLtx2SummedWeights[i]);
    }
  }
  // ...and the overlap really does sum to 2, so the case is not vacuous.
  CHECK(*std::max_element(weights.begin(), weights.end()) == doctest::Approx(2.0).scale(0.0));
}

namespace {

// One decode arm, run twice below with the two causality polarities.
void RunDecodeArm(bool causal, const std::string& prefix, const char* const* param_names,
                  const int64_t* param_counts, size_t param_size, int64_t out_c, int64_t out_t,
                  int64_t out_h, int64_t out_w, int64_t tile_count, int64_t group_count,
                  int64_t complementary, const int64_t* chunk_frames, int64_t chunk_count,
                  const float* untiled_golden, size_t untiled_size, const float* tiled_golden,
                  size_t tiled_size, double upstream_gap, double upstream_control_gap,
                  int64_t control_chunk_count, double upstream_untiled_spatial_gap,
                  int64_t untiled_spatial_chunk_count, int64_t upstream_untiled_frames_raises,
                  const char* upstream_untiled_frames_raise_file,
                  int64_t upstream_untiled_frames_raise_line,
                  const char* upstream_untiled_frames_raise_message) {
  const vllm::Ltx2ConvVideoDecoderConfig cfg = TilingFixtureConfig(causal, prefix);
  ParamBag bag = BuildFixtureParams(cfg);
  CheckManifest(bag, param_names, param_counts, param_size);

  const std::vector<float> latent =
      Ltx2Input(prefix + "input", cfg.in_channels * kLatentT * kLatentH * kLatentW, 1.0);
  NoNoise noise;

  // (B) the untiled reference, first — if this is wrong the tiled comparison
  // proves nothing about the tiling.
  const vllm::Ltx2VideoFrames untiled = vllm::Ltx2ConvVideoDecode(
      cfg, bag.weights, latent, cfg.in_channels, kLatentT, kLatentH, kLatentW, &noise);
  REQUIRE(untiled.channels == out_c);
  REQUIRE(untiled.frames == out_t);
  REQUIRE(untiled.height == out_h);
  REQUIRE(untiled.width == out_w);
  {
    const double err = MaxAbsDiff(untiled.data, untiled_golden, untiled_size);
    INFO("untiled max|diff| = " << err);
    CHECK(err <= kLtx2GoldenTol);
  }

  // The tile layout upstream built, asserted before the tensors so a shape drift
  // is reported as a shape drift.
  const vllm::Ltx2ScaleFactors factors =
      vllm::Ltx2VideoScaleFactorsFromBlocks(cfg.decoder_blocks, cfg.patch_size);
  const std::vector<vllm::Ltx2Tile> tiles =
      vllm::Ltx2CreateTiles(kLatentT, kLatentH, kLatentW, FixtureTiling(), factors);
  CHECK(static_cast<int64_t>(tiles.size()) == tile_count);
  CHECK(static_cast<int64_t>(vllm::Ltx2GroupTilesByTemporalSlice(tiles).size()) == group_count);
  CHECK(static_cast<int64_t>(vllm::Ltx2MasksAreComplementary(
            tiles, out_t, out_h, out_w)) == complementary);

  // (A) the correctness gate: our streamed chunks vs upstream's.
  Collected collected;
  vllm::Ltx2ConvVideoDecodeTiled(
      cfg, bag.weights, latent, cfg.in_channels, kLatentT, kLatentH, kLatentW, &noise,
      FixtureTiling(), [&](const vllm::Ltx2VideoChunk& c) { collected.chunks.push_back(c); });

  REQUIRE(static_cast<int64_t>(collected.chunks.size()) == chunk_count);
  int64_t expect_first = 0;
  for (int64_t i = 0; i < chunk_count; ++i) {
    INFO("chunk " << i);
    CHECK(collected.chunks[static_cast<size_t>(i)].frames.frames == chunk_frames[i]);
    // The chunk announces where it starts, so a consumer can write frame files
    // without counting; an off-by-one here would silently reorder the clip.
    CHECK(collected.chunks[static_cast<size_t>(i)].first_frame == expect_first);
    expect_first += chunk_frames[i];
  }
  // STREAMING, not batching: no single chunk is the whole clip when there is more
  // than one group. Without this, a port that decoded everything and emitted it in
  // one callback would pass every tensor assertion in this file.
  if (group_count > 1) {
    for (const vllm::Ltx2VideoChunk& c : collected.chunks) CHECK(c.frames.frames < out_t);
  }
  CHECK(expect_first == out_t);

  const vllm::Ltx2VideoFrames tiled = collected.Concat();
  REQUIRE(tiled.frames == out_t);
  {
    const double err = MaxAbsDiff(tiled.data, tiled_golden, tiled_size);
    INFO("tiled max|diff| vs upstream = " << err);
    CHECK(err <= kLtx2GoldenTol);
  }

  // (C) the tiled-vs-untiled gap is UPSTREAM's own number. It is NOT small — it is
  // the size of the signal — and that is the measured statement, not a bound.
  const double gap = MaxAbsDiff(tiled.data, untiled.data.data(), untiled.data.size());
  INFO("our max|tiled - untiled| = " << gap << ", upstream's = " << upstream_gap);
  CHECK(std::abs(gap - upstream_gap) <= kLtx2GoldenTol);

  // (B') THE ONE-TILE CONTROL, whose bound is ZERO and is measured. A tiling
  // config whose splits all short-circuit must reproduce the untiled decode
  // EXACTLY — this is what makes routing the pipeline through the streaming entry
  // point safe at every size the AUTO layout does not tile.
  CHECK(upstream_control_gap == 0.0);
  Collected control;
  vllm::Ltx2ConvVideoDecodeTiled(
      cfg, bag.weights, latent, cfg.in_channels, kLatentT, kLatentH, kLatentW, &noise,
      ControlTiling(), [&](const vllm::Ltx2VideoChunk& c) { control.chunks.push_back(c); });
  CHECK(static_cast<int64_t>(control.chunks.size()) == control_chunk_count);
  const vllm::Ltx2VideoFrames control_frames = control.Concat();
  REQUIRE(control_frames.data.size() == untiled.data.size());
  const double control_gap =
      MaxAbsDiff(control_frames.data, untiled.data.data(), untiled.data.size());
  INFO("one-tile control max|diff| vs untiled = " << control_gap);
  CHECK(control_gap == 0.0);

  // (B'') THE UNTILED-SPATIAL CONTROL — the same bound through the OTHER branch.
  //
  // Height and width declared untiled (`tile_size == 0`) send `Ltx2CreateTiles`
  // through the broadcast mapper instead of `Ltx2MapSpatialSlice`. Upstream runs
  // this config and reproduces `forward` exactly; a `{0.0f}` broadcast mask would
  // render the entire clip black and, before this arm existed, would still have
  // left every case in this file green.
  CHECK(upstream_untiled_spatial_gap == 0.0);
  Collected untiled_spatial;
  vllm::Ltx2ConvVideoDecodeTiled(
      cfg, bag.weights, latent, cfg.in_channels, kLatentT, kLatentH, kLatentW, &noise,
      UntiledSpatialTiling(),
      [&](const vllm::Ltx2VideoChunk& c) { untiled_spatial.chunks.push_back(c); });
  CHECK(static_cast<int64_t>(untiled_spatial.chunks.size()) == untiled_spatial_chunk_count);
  const vllm::Ltx2VideoFrames untiled_spatial_frames = untiled_spatial.Concat();
  REQUIRE(untiled_spatial_frames.data.size() == untiled.data.size());
  const double untiled_spatial_gap =
      MaxAbsDiff(untiled_spatial_frames.data, untiled.data.data(), untiled.data.size());
  INFO("untiled-spatial control max|diff| vs untiled = " << untiled_spatial_gap);
  CHECK(untiled_spatial_gap == upstream_untiled_spatial_gap);
  // The config really did take the broadcast branch, or the assertion above is
  // just the one-tile control wearing a different name.
  const std::vector<vllm::Ltx2Tile> untiled_spatial_tiles =
      vllm::Ltx2CreateTiles(kLatentT, kLatentH, kLatentW, UntiledSpatialTiling(), factors);
  REQUIRE(untiled_spatial_tiles.size() == 1u);
  CHECK(untiled_spatial_tiles[0].out_h.mask.size() == 1u);
  CHECK(untiled_spatial_tiles[0].out_w.mask.size() == 1u);

  // ...and the FRAMES axis is refused, because upstream's own `tiled_decode`
  // cannot run it: `DEFAULT_MAPPING_OPERATION` hands it `slice(0, None)` and
  // conv_video_decoder.py:424 subtracts that `None`. The golden records that
  // upstream raises, so this is a mirrored refusal and not a local policy.
  //
  // THE MECHANISM IS PINNED, NOT JUST THE EXCEPTION TYPE. `raises == 1` alone
  // stays 1 for any future `TypeError` from any line of `tiled_decode`, which
  // would leave the refusal citing a cause that had silently moved. The
  // generator now records the innermost raising frame, so the file, the line
  // conv_video_decoder.py:424 and the None-stop arithmetic in the message are
  // each asserted here — the same three facts the refusal message names.
  CHECK(upstream_untiled_frames_raises == 1);
  CHECK(std::string(upstream_untiled_frames_raise_file) == "conv_video_decoder.py");
  CHECK(upstream_untiled_frames_raise_line == 424);
  const std::string raise_message(upstream_untiled_frames_raise_message);
  INFO("upstream's untiled-frames TypeError: " << raise_message);
  CHECK(raise_message.find("unsupported operand type") != std::string::npos);
  CHECK(raise_message.find("NoneType") != std::string::npos);
  int64_t refused_emitted = 0;
  CHECK_THROWS(vllm::Ltx2ConvVideoDecodeTiled(
      cfg, bag.weights, latent, cfg.in_channels, kLatentT, kLatentH, kLatentW, &noise,
      UntiledAxesTiling(), [&](const vllm::Ltx2VideoChunk&) { ++refused_emitted; }));
  CHECK(refused_emitted == 0);
}

}  // namespace

TEST_CASE("ltx2 tiled decode matches upstream ltx_core — CAUSAL") {
  RunDecodeArm(
      /*causal=*/true, "ltx2.tiledec.causal.", vllm_test::kLtx2TileDecCausalParamNames,
      vllm_test::kLtx2TileDecCausalParamCounts,
      std::size(vllm_test::kLtx2TileDecCausalParamNames), vllm_test::kLtx2TileDecCausalOutC,
      vllm_test::kLtx2TileDecCausalOutT, vllm_test::kLtx2TileDecCausalOutH,
      vllm_test::kLtx2TileDecCausalOutW, vllm_test::kLtx2TileDecCausalTileCount,
      vllm_test::kLtx2TileDecCausalGroupCount, vllm_test::kLtx2TileDecCausalComplementary,
      vllm_test::kLtx2TileDecCausalChunkFrames, vllm_test::kLtx2TileDecCausalChunkCount,
      vllm_test::kLtx2TileDecCausalUntiled, std::size(vllm_test::kLtx2TileDecCausalUntiled),
      vllm_test::kLtx2TileDecCausalTiled, std::size(vllm_test::kLtx2TileDecCausalTiled),
      vllm_test::kLtx2TileDecCausalUpstreamTiledVsUntiled,
      vllm_test::kLtx2TileDecCausalUpstreamControlVsUntiled,
      vllm_test::kLtx2TileDecCausalControlChunkCount,
      vllm_test::kLtx2TileDecCausalUpstreamUntiledSpatialVsUntiled,
      vllm_test::kLtx2TileDecCausalUntiledSpatialChunkCount,
      vllm_test::kLtx2TileDecCausalUpstreamUntiledFramesRaises,
      vllm_test::kLtx2TileDecCausalUpstreamUntiledFramesRaiseFile,
      vllm_test::kLtx2TileDecCausalUpstreamUntiledFramesRaiseLine,
      vllm_test::kLtx2TileDecCausalUpstreamUntiledFramesRaiseMessage);
}

TEST_CASE("ltx2 tiled decode matches upstream ltx_core — NON-CAUSAL, the shipped polarity") {
  // `ltx-2.5-video-vae-conv-bf16.safetensors` sets `causal_decoder: false`, so
  // this is the arm the product actually runs; the causal one above is the arm
  // where the temporal padding is one-sided.
  RunDecodeArm(
      /*causal=*/false, "ltx2.tiledec.noncausal.", vllm_test::kLtx2TileDecNonCausalParamNames,
      vllm_test::kLtx2TileDecNonCausalParamCounts,
      std::size(vllm_test::kLtx2TileDecNonCausalParamNames), vllm_test::kLtx2TileDecNonCausalOutC,
      vllm_test::kLtx2TileDecNonCausalOutT, vllm_test::kLtx2TileDecNonCausalOutH,
      vllm_test::kLtx2TileDecNonCausalOutW, vllm_test::kLtx2TileDecNonCausalTileCount,
      vllm_test::kLtx2TileDecNonCausalGroupCount, vllm_test::kLtx2TileDecNonCausalComplementary,
      vllm_test::kLtx2TileDecNonCausalChunkFrames, vllm_test::kLtx2TileDecNonCausalChunkCount,
      vllm_test::kLtx2TileDecNonCausalUntiled, std::size(vllm_test::kLtx2TileDecNonCausalUntiled),
      vllm_test::kLtx2TileDecNonCausalTiled, std::size(vllm_test::kLtx2TileDecNonCausalTiled),
      vllm_test::kLtx2TileDecNonCausalUpstreamTiledVsUntiled,
      vllm_test::kLtx2TileDecNonCausalUpstreamControlVsUntiled,
      vllm_test::kLtx2TileDecNonCausalControlChunkCount,
      vllm_test::kLtx2TileDecNonCausalUpstreamUntiledSpatialVsUntiled,
      vllm_test::kLtx2TileDecNonCausalUntiledSpatialChunkCount,
      vllm_test::kLtx2TileDecNonCausalUpstreamUntiledFramesRaises,
      vllm_test::kLtx2TileDecNonCausalUpstreamUntiledFramesRaiseFile,
      vllm_test::kLtx2TileDecNonCausalUpstreamUntiledFramesRaiseLine,
      vllm_test::kLtx2TileDecNonCausalUpstreamUntiledFramesRaiseMessage);
}

TEST_CASE("ltx2 tiled decode: the diffusion decoder is REFUSED, never downgraded") {
  const vllm::Ltx2ConvVideoDecoderConfig cfg =
      TilingFixtureConfig(/*causal=*/false, "ltx2.tiledec.noncausal.");
  ParamBag bag = BuildFixtureParams(cfg);
  const std::vector<float> latent =
      Ltx2Input("ltx2.tiledec.noncausal.input",
                cfg.in_channels * kLatentT * kLatentH * kLatentW, 1.0);
  NoNoise noise;
  int64_t emitted = 0;
  CHECK_THROWS(vllm::Ltx2VideoDecodeStreaming(
      vllm::Ltx2VideoDecoderKind::kDiffusion, cfg, bag.weights, latent, cfg.in_channels, kLatentT,
      kLatentH, kLatentW, &noise, ControlTiling(),
      [&](const vllm::Ltx2VideoChunk&) { ++emitted; }));
  // A refusal that had already streamed a chunk would have handed the caller a
  // partial render from the WRONG decoder before failing.
  CHECK(emitted == 0);
}
