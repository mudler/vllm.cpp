// LTX-2.5 IMAGE CONDITIONING gate — row LTX25-IMAGE-COND, issue #644.
//
// Spec: .agents/specs/ltx25-image-conditioning.md §5.
//
// WHAT THIS GATES. The CHAIN a conditioning image travels, against EXECUTED
// upstream at reduced dimensions: PPM -> aspect-fill resize -> normalize ->
// `Ltx2ConvVideoEncode` -> `Ltx2ConditionVideoByLatentIndex` -> the noiser.
// Every link but the first two already had a golden in `test_ltx2_vae`; the
// CHAIN did not, and a chain of green links can still be wired in the wrong
// ORDER. It also gates the three things this row built to reach that chain from
// a checkpoint at all: the encoder key rules, the encoder config parser, and the
// CRF resolution that decides whether an image request is served or refused.
//
// WHAT IT CANNOT SHOW. Nothing here is a render-quality result. And one negative
// result is recorded rather than papered over: NO golden in this file can see a
// swap of the resize/normalize ORDER. `resize` is a convex combination and
// `normalize` is affine, so the two orders are equal in exact arithmetic and
// differ only by f32 rounding — measured at 1.94e-07 upstream
// (`kLtx2ImgPreOrderGap`), which is below the golden band AND below this port's
// own distance from torch. The order is mirrored because it is upstream's, and
// that fact is written down here instead of being assumed to be covered.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ltx2_image_cond_goldens.inc"
#include "vllm/model_executor/models/ltx2_conditioning.h"
#include "vllm/model_executor/models/ltx2_image_preprocess.h"
#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/model_executor/models/ltx2_video_vae_encoder.h"

namespace {

// ---------------------------------------------------------------------------
// TOLERANCES, DERIVED FROM MEASUREMENT rather than picked — a band that can
// never bind reports nothing, which is the criticism test_ltx2_vae.cpp:172-180
// already makes of its own history. Both were measured by setting them to 1e-12
// and reading the reported `worst` on this box (CPU Release, gcc, the tree's own
// `-ffp-contract=off`):
//
//   section 1, resize, 0..255 PIXEL space   6.10352e-05  (case 2, the identity, is 0)
//   section 2, preprocess, [-1, 1]          4.76837e-07
//   section 3, encoded latent               2.68221e-07
//   section 4, conditioned clean            2.68221e-07
//   section 5, noised latent                1.78814e-07
//
// `kLtx2ImgGoldenTol` covers everything in latent/normalized space at ~4x the
// worst of those. It is also the band scripts/gen-ltx2-image-cond-goldens.py
// PARSES out of this file, so the number has exactly one definition and the
// generator's own assertions are stated against the band the suite applies.
//
// `kLtx2ImgPixelTol` is section 1's, and it is two orders wider for a MEASURED
// reason worth stating because it looks like slack and is not. Torch's bilinear
// does not round the way any portable f32 expression does. The index map and the
// lambdas were probed directly with basis images and match this port BIT FOR
// BIT; the residual is in the ACCUMULATION, and it appears even on an output
// element whose width weights are exactly (1, 0) — a pure two-term
// `a*h0 + c*h1`, where dimension order cannot be the explanation. Plain-f32,
// f64-accumulate and premultiplied-weight orderings were all tried against torch
// and all three land 1 ulp away on the same elements, which is the signature of
// FMA contraction inside torch's kernel. This tree compiles with
// `-ffp-contract=off`, so it cannot reproduce that even in principle. 6.1e-05 is
// ~1 ulp at 255; the band is ~3x it, for a different libm and a different
// -march. A structural porting error moves this by orders of magnitude, not by
// ulps — a `round` instead of a `ceil` in the resize moves it to ~200.
// ---------------------------------------------------------------------------
constexpr double kLtx2ImgGoldenTol = 2e-6;
constexpr double kLtx2ImgPixelTol = 2e-4;

// The exact upstream tree the goldens came from. Regenerating against a
// DIFFERENT checkout fails here instead of silently replacing the oracle.
constexpr const char* kLtx2ImgCondUpstreamRevisionPin =
    "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca";

// ---------------------------------------------------------------------------
// The shared deterministic stream — the exact mirror of the generator's
// `ltx_rand` / `ltx_bytes` / `param_values`, and byte-for-byte what
// tests/vllm/models/test_ltx2_vae.cpp uses. It is duplicated here rather than
// shared because that file is a .cpp, not a header; the duplication is made SAFE
// by `CheckManifest` below, which asserts this file's parameter set is exactly
// the state_dict the generator filled — so a copy that drifted is a failure
// rather than a silently different model.
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

// The generator's `ltx_bytes`: uint8 codes, because a conditioning image IS
// uint8 out of the decoder and quantizing a float stream afterwards would gate a
// different input than a real PPM carries.
std::vector<uint8_t> Ltx2Bytes(const std::string& name, int64_t count) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<uint8_t> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] =
        static_cast<uint8_t>(Splitmix64(seed + static_cast<uint64_t>(i)) % 256U);
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

std::vector<float> Ltx2Param(const std::string& name, const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;
  const size_t rank = shape.size();
  double scale = 0.1;
  double offset = 0.0;
  if (EndsWith(name, "std-of-means")) {
    offset = 1.0;
  } else if (EndsWith(name, "mean-of-means")) {
    // scale 0.1, offset 0
  } else if (EndsWith(name, ".bias")) {
    scale = 0.05;
  } else if (rank == 1 && EndsWith(name, ".weight")) {
    offset = 1.0;
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

// The reduced encoder the generator built (`IMG_ENC_BLOCKS` / `IMG_ENC`).
vllm::Ltx2ConvVideoEncoderConfig ImageEncoderConfig() {
  vllm::Ltx2ConvVideoEncoderConfig cfg;
  cfg.in_channels = 3;
  cfg.out_channels = 4;
  cfg.patch_size = 2;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  cfg.latent_log_var = vllm::Ltx2LogVarianceType::kUniform;
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kZeros;
  cfg.encoder_blocks = {
      {"res_x", 1, 0},
      {"compress_space_res", 1, 2},
      {"compress_all_res", 1, 1},
  };
  cfg.prefix = "ltx2.imgenc.";
  return cfg;
}

// The parameter set `VideoEncoder(**IMG_ENC)` builds, in state_dict order.
ParamBag BuildImageEncoderParams(const vllm::Ltx2ConvVideoEncoderConfig& cfg) {
  ParamBag bag;
  const std::string p = cfg.prefix;
  bag.Put(p + "per_channel_statistics.std-of-means", {cfg.out_channels});
  bag.Put(p + "per_channel_statistics.mean-of-means", {cfg.out_channels});
  const int64_t patched_in = cfg.in_channels * cfg.patch_size * cfg.patch_size;
  bag.Put(p + "conv_in.conv.weight", {cfg.out_channels, patched_in, 3, 3, 3});
  bag.Put(p + "conv_in.conv.bias", {cfg.out_channels});

  int64_t feature = cfg.out_channels;
  for (size_t i = 0; i < cfg.encoder_blocks.size(); ++i) {
    const vllm::Ltx2VideoEncoderBlock& block = cfg.encoder_blocks[i];
    const std::string bp = p + "down_blocks." + std::to_string(i);
    const int64_t multiplier = block.multiplier != 0 ? block.multiplier : 2;
    if (block.name == "res_x") {
      for (int64_t j = 0; j < block.num_layers; ++j) {
        const std::string rp = bp + ".res_blocks." + std::to_string(j);
        bag.Put(rp + ".conv1.conv.weight", {feature, feature, 3, 3, 3});
        bag.Put(rp + ".conv1.conv.bias", {feature});
        bag.Put(rp + ".conv2.conv.weight", {feature, feature, 3, 3, 3});
        bag.Put(rp + ".conv2.conv.bias", {feature});
      }
    } else {
      // The *_res family: SpaceToDepthDownsample's conv emits
      // out_channels / prod(stride); the space-to-depth fold multiplies it back.
      const int64_t st = block.name == "compress_space_res" ? 1 : 2;
      const int64_t ss = block.name == "compress_time_res" ? 1 : 2;
      const int64_t out = feature * multiplier;
      const int64_t conv_out = out / (st * ss * ss);
      bag.Put(bp + ".conv.conv.weight", {conv_out, feature, 3, 3, 3});
      bag.Put(bp + ".conv.conv.bias", {conv_out});
      feature = out;
    }
  }
  bag.Put(p + "conv_out.conv.weight", {cfg.out_channels + 1, feature, 3, 3, 3});
  bag.Put(p + "conv_out.conv.bias", {cfg.out_channels + 1});
  return bag;
}

// NaN-hardened, for the reason issue #449 records: `std::max(worst, x)` is
// `worst < x ? x : worst`, and `worst < NaN` is false, so an all-NaN arm reduces
// to 0.0 and reports a perfect match.
double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t count) {
  REQUIRE(got.size() == count);
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double diff = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    if (!(diff <= worst)) worst = diff;  // NaN takes this branch
  }
  return worst;
}

// A binary PPM (P6) carrying `codes` as an h x w RGB payload — the container the
// engine actually reads, built here so section 2 gates the DECODER too and not
// just the arithmetic downstream of it.
std::string MakePpm(const std::vector<uint8_t>& codes, int64_t height, int64_t width) {
  std::string out = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
  out.append(reinterpret_cast<const char*>(codes.data()), codes.size());
  return out;
}

const std::string& ConditioningImagePpm() {
  static const std::string ppm = MakePpm(
      Ltx2Bytes("ltx2.imgcond.image",
                vllm_test::kLtx2ImgPreSrcH * vllm_test::kLtx2ImgPreSrcW * 3),
      vllm_test::kLtx2ImgPreSrcH, vllm_test::kLtx2ImgPreSrcW);
  return ppm;
}

// The latent state the generator built: the target shape, patch 1, causal fix.
vllm::Ltx2VideoLatentShape ConditioningTarget() {
  vllm::Ltx2VideoLatentShape target;
  target.batch = 1;
  target.channels = vllm_test::kLtx2ImgEncOutC;
  target.frames = vllm_test::kLtx2ImgCondTargetFrames;
  target.height = vllm_test::kLtx2ImgEncOutH;
  target.width = vllm_test::kLtx2ImgEncOutW;
  return target;
}

// The encoded conditioning image, as `Ltx2ConvVideoEncode` produces it.
vllm::Ltx2LatentVolume EncodeConditioningImage(const std::vector<float>& chw) {
  const vllm::Ltx2ConvVideoEncoderConfig cfg = ImageEncoderConfig();
  const ParamBag bag = BuildImageEncoderParams(cfg);
  return vllm::Ltx2ConvVideoEncode(cfg, bag.weights, chw, cfg.in_channels, /*frame_count=*/1,
                                   vllm_test::kLtx2ImgPreDstH, vllm_test::kLtx2ImgPreDstW,
                                   nullptr);
}

// The composition under test, factored so the mutation witnesses below can drive
// it with a deliberately WRONG `latent`, `strength` or ORDER and show the golden
// moves. Returns the noised latent.
struct Composition {
  std::vector<float> clean, mask, noised;
};

Composition ComposeConditioning(const vllm::Ltx2LatentVolume& conditioning, double strength,
                                bool condition_before_noise) {
  const vllm::Ltx2VideoLatentShape target = ConditioningTarget();
  const vllm::Ltx2ScaleFactors factors;
  vllm::Ltx2LatentState state = vllm::Ltx2CreateVideoLatentState(
      target, vllm_test::kLtx2ImgCondPatch, factors, vllm_test::kLtx2ImgCondFps,
      /*causal_fix=*/true);

  const std::vector<float> noise =
      Ltx2Input("ltx2.imgcond.noise", static_cast<int64_t>(state.latent.size()), 1.0);

  auto apply = [&]() {
    vllm::Ltx2ConditionVideoByLatentIndex(&state, target, vllm_test::kLtx2ImgCondPatch,
                                          conditioning, strength,
                                          vllm_test::kLtx2ImgCondLatentIdx);
  };
  auto noise_it = [&]() {
    std::vector<float> broadcast(state.latent.size());
    for (int64_t t = 0; t < state.tokens; ++t) {
      for (int64_t c = 0; c < state.width; ++c) {
        broadcast[static_cast<size_t>(t * state.width + c)] =
            state.mask[static_cast<size_t>(t)];
      }
    }
    state.latent = vllm::Ltx2GaussianNoise(
        state.latent.data(), state.clean.data(), broadcast.data(), noise.data(),
        static_cast<int64_t>(state.latent.size()),
        static_cast<float>(vllm_test::kLtx2ImgNoiseScale));
  };

  if (condition_before_noise) {
    apply();
    noise_it();
  } else {
    noise_it();
    apply();
  }
  return Composition{state.clean, state.mask, state.latent};
}

}  // namespace

// ─── the anchor ─────────────────────────────────────────────────────────────

TEST_CASE("ltx2 image cond: the goldens carry the PINNED upstream revision") {
  CHECK(std::string(vllm_test::kLtx2ImgCondUpstreamRevision) ==
        std::string(kLtx2ImgCondUpstreamRevisionPin));
}

// ─── section 1: the resize ──────────────────────────────────────────────────

TEST_CASE("ltx2 image cond: resize_and_center_crop matches executed upstream") {
  // Five shape pairs: upscale, downscale, identity, wider-than-target and
  // taller-than-target. Case 1 (32x24 -> 16x16) is the one where `ceil`
  // disagrees with BOTH `round` and `int`: 32 * (16/24) is 21.333, so upstream
  // resizes to 22 rows and crops 3, while a rounding port resizes to 21 and
  // crops 2 — every output value moves and every shape check still passes.
  struct Case {
    int64_t src_h, src_w, dst_h, dst_w;
    const float* golden;
    size_t golden_size;
    const char* name;
  };
  const Case cases[] = {
      {vllm_test::kLtx2ImgResize0SrcH, vllm_test::kLtx2ImgResize0SrcW,
       vllm_test::kLtx2ImgResize0DstH, vllm_test::kLtx2ImgResize0DstW,
       vllm_test::kLtx2ImgResize0Golden, std::size(vllm_test::kLtx2ImgResize0Golden),
       "12x20 -> 16x16"},
      {vllm_test::kLtx2ImgResize1SrcH, vllm_test::kLtx2ImgResize1SrcW,
       vllm_test::kLtx2ImgResize1DstH, vllm_test::kLtx2ImgResize1DstW,
       vllm_test::kLtx2ImgResize1Golden, std::size(vllm_test::kLtx2ImgResize1Golden),
       "32x24 -> 16x16 (the ceil case)"},
      {vllm_test::kLtx2ImgResize2SrcH, vllm_test::kLtx2ImgResize2SrcW,
       vllm_test::kLtx2ImgResize2DstH, vllm_test::kLtx2ImgResize2DstW,
       vllm_test::kLtx2ImgResize2Golden, std::size(vllm_test::kLtx2ImgResize2Golden),
       "16x16 -> 16x16 (identity)"},
      {vllm_test::kLtx2ImgResize3SrcH, vllm_test::kLtx2ImgResize3SrcW,
       vllm_test::kLtx2ImgResize3DstH, vllm_test::kLtx2ImgResize3DstW,
       vllm_test::kLtx2ImgResize3Golden, std::size(vllm_test::kLtx2ImgResize3Golden),
       "8x8 -> 16x24"},
      {vllm_test::kLtx2ImgResize4SrcH, vllm_test::kLtx2ImgResize4SrcW,
       vllm_test::kLtx2ImgResize4DstH, vllm_test::kLtx2ImgResize4DstW,
       vllm_test::kLtx2ImgResize4Golden, std::size(vllm_test::kLtx2ImgResize4Golden),
       "10x7 -> 16x16"},
  };
  REQUIRE(std::size(cases) == static_cast<size_t>(vllm_test::kLtx2ImgResizeCases));

  for (size_t i = 0; i < std::size(cases); ++i) {
    const Case& c = cases[i];
    INFO("resize case " << i << ": " << c.name);
    const std::vector<uint8_t> codes =
        Ltx2Bytes("ltx2.imgcond.resize" + std::to_string(i), c.src_h * c.src_w * 3);
    std::vector<float> hwc(codes.size());
    for (size_t k = 0; k < codes.size(); ++k) hwc[k] = static_cast<float>(codes[k]);

    const std::vector<float> got =
        vllm::Ltx2ResizeAndCenterCrop(hwc.data(), c.src_h, c.src_w, 3, c.dst_h, c.dst_w);
    REQUIRE(got.size() == c.golden_size);
    const double worst = MaxAbsDiff(got, c.golden, c.golden_size);
    CAPTURE(worst);
    CHECK(worst <= kLtx2ImgPixelTol);
  }
}

TEST_CASE("ltx2 image cond: an aspect-fill resize that cannot cover the target is refused") {
  // Not reachable through `Ltx2LoadImageAndPreprocess` — `scale = max(...)` makes
  // it unreachable by construction — which is exactly why it is asserted here:
  // a guard nobody can trip is a guard nobody notices has been deleted, and this
  // one is what stops a future caller with its own scale from silently indexing
  // a negative crop offset.
  const std::vector<float> tiny(3 * 4 * 4, 1.0F);
  CHECK_THROWS(vllm::Ltx2ResizeAndCenterCrop(tiny.data(), 0, 4, 3, 4, 4));
  CHECK_THROWS(vllm::Ltx2ResizeAndCenterCrop(nullptr, 4, 4, 3, 4, 4));
}

// ─── section 2: the whole preprocess, from PPM bytes ────────────────────────

TEST_CASE("ltx2 image cond: load_image_and_preprocess matches upstream at crf=0") {
  const std::vector<float> got = vllm::Ltx2LoadImageAndPreprocess(
      "first_frame", ConditioningImagePpm(), vllm_test::kLtx2ImgPreDstH,
      vllm_test::kLtx2ImgPreDstW, /*crf=*/0);
  REQUIRE(got.size() == std::size(vllm_test::kLtx2ImgPreGolden));
  const double worst =
      MaxAbsDiff(got, vllm_test::kLtx2ImgPreGolden, std::size(vllm_test::kLtx2ImgPreGolden));
  CAPTURE(worst);
  CHECK(worst <= kLtx2ImgGoldenTol);

  // The output space, asserted rather than assumed: `Ltx2ConvVideoEncode` takes
  // [-1, 1] pixels, and a port that forgot the `- 1.0` would still be finite,
  // still be the right shape, and still pass every structural check.
  const auto minmax = std::minmax_element(got.begin(), got.end());
  CHECK(*minmax.first >= -1.0F);
  CHECK(*minmax.second <= 1.0F);
  CHECK(*minmax.first < 0.0F);
}

TEST_CASE("ltx2 image cond: the resize/normalize ORDER is below every band here") {
  // THE NEGATIVE RESULT, asserted so it stays true rather than left as prose.
  // A convex combination commutes with an affine map, so the two orders differ
  // only by f32 rounding — `kLtx2ImgPreOrderGap` is that difference, measured
  // upstream. It is below the golden band, so no value comparison in this file
  // can catch a swap; and it is below `kLtx2ImgPixelTol` too, so tightening the
  // band would not help either, it would only make section 1 red.
  //
  // If this ever fails, the two operations stopped being the affine/convex pair
  // this reasoning rests on, and the ORDER becomes gateable — which is a finding
  // worth acting on, not an assertion to delete.
  CHECK(vllm_test::kLtx2ImgPreOrderGap < kLtx2ImgGoldenTol);
  CHECK(vllm_test::kLtx2ImgPreOrderGap * 127.5 < kLtx2ImgPixelTol);
}

// ─── the CRF refusal, and what resolves it ──────────────────────────────────

TEST_CASE("ltx2 image cond: a non-zero CRF is refused by name, never rendered") {
  CHECK_NOTHROW(vllm::Ltx2PreprocessImageCrf(0));
  for (const int64_t crf : {1, 18, 33, 51, -1}) {
    INFO("crf " << crf);
    CHECK_THROWS_WITH_AS(vllm::Ltx2PreprocessImageCrf(crf),
                         doctest::Contains("encode_single_frame"), std::runtime_error);
  }
  // And it is refused through the FULL entry point too, not only the helper —
  // an image request that named a CRF must not reach the encoder.
  CHECK_THROWS_WITH_AS(
      vllm::Ltx2LoadImageAndPreprocess("first_frame", ConditioningImagePpm(), 16, 16, 18),
      doctest::Contains("CRF 18"), std::runtime_error);
}

TEST_CASE("ltx2 image cond: an LTX-2.5 checkpoint RESOLVES crf 18, so the default refuses") {
  // constants.py:130-133 — the newest row at or below the version. This is what
  // makes the refusal above a real one: a caller who says nothing gets 18.
  CHECK(vllm::Ltx2ResolveDefaultImageCrf({2, 5}) == 18);
  CHECK(vllm::Ltx2ResolveDefaultImageCrf({2, 5, 0}) == 18);
  CHECK(vllm::Ltx2ResolveDefaultImageCrf({2, 4}) == 18);
  CHECK(vllm::Ltx2ResolveDefaultImageCrf({3}) == 18);  // newer inherits the closest known row
  CHECK(vllm::Ltx2ResolveDefaultImageCrf({2, 3}) == 33);
  CHECK(vllm::Ltx2ResolveDefaultImageCrf({2, 0}) == 33);
  CHECK(vllm::Ltx2ResolveDefaultImageCrf({2}) == 33);
  // `detect_model_version` returns () for an unset/unparseable version, which
  // "compares below every real version" (:138-139) — the OLDEST fallback.
  CHECK(vllm::Ltx2ResolveDefaultImageCrf({}) == 33);
}

// ─── the PPM decoder's own refusals ─────────────────────────────────────────

TEST_CASE("ltx2 image cond: a container this tree cannot read is refused by name") {
  int64_t h = 0, w = 0;
  CHECK_THROWS_WITH_AS(vllm::Ltx2DecodePpmRgb("first_frame", "\x89PNG\r\n\x1a\n", &h, &w),
                       doctest::Contains("binary PPM"), std::runtime_error);
  // P3 is ASCII PPM — a real format, and NOT the one that is read.
  CHECK_THROWS_AS(vllm::Ltx2DecodePpmRgb("first_frame", "P3\n2 2\n255\n0 0 0", &h, &w),
                  std::runtime_error);
  // A non-255 maxval is REFUSED rather than rescaled: PIL's rescale is what
  // upstream would apply and mirroring it is a separate port.
  CHECK_THROWS_WITH_AS(
      vllm::Ltx2DecodePpmRgb("first_frame", std::string("P6\n1 1\n15\n") + "\x01\x02\x03", &h, &w),
      doctest::Contains("maxval"), std::runtime_error);
  // A truncated payload must not be padded with whatever was in the buffer.
  CHECK_THROWS_WITH_AS(vllm::Ltx2DecodePpmRgb("first_frame", "P6\n4 4\n255\nshort", &h, &w),
                       doctest::Contains("truncated"), std::runtime_error);

  // The happy path reports the FILE's own geometry, not the caller's.
  const std::vector<uint8_t> rgb =
      vllm::Ltx2DecodePpmRgb("first_frame", ConditioningImagePpm(), &h, &w);
  CHECK(h == vllm_test::kLtx2ImgPreSrcH);
  CHECK(w == vllm_test::kLtx2ImgPreSrcW);
  CHECK(rgb.size() == static_cast<size_t>(h * w * 3));
}

// ─── the load path this row exists to build ─────────────────────────────────

TEST_CASE("ltx2 image cond: the ENCODER key rules are the encoder's, not the decoder's") {
  const std::vector<vllm::Ltx2VaeKeyRule> rules = vllm::Ltx2VideoVaeEncoderKeyRules();
  REQUIRE(rules.size() == 4);
  // model_configurator.py:267-276, rule for rule and IN ORDER — the `vae.`
  // spellings must precede their bare twins or a monolithic checkpoint's
  // `vae.encoder.*` matches nothing and is silently dropped.
  CHECK(rules[0].match_prefix == "vae.encoder.");
  CHECK(rules[0].replacement.empty());
  CHECK(rules[1].match_prefix == "vae.per_channel_statistics.");
  CHECK(rules[1].replacement == "per_channel_statistics.");
  CHECK(rules[2].match_prefix == "encoder.");
  CHECK(rules[2].replacement.empty());
  CHECK(rules[3].match_prefix == "per_channel_statistics.");
  CHECK(rules[3].replacement == "per_channel_statistics.");

  // The two filters must DISAGREE. Before this row the engine held the decoder's
  // alone, and asserting they differ is what says this one is not a copy.
  const std::vector<vllm::Ltx2VaeKeyRule> decoder = vllm::Ltx2VideoVaeDecoderKeyRules();
  REQUIRE(decoder.size() == rules.size());
  bool differs = false;
  for (size_t i = 0; i < rules.size(); ++i) {
    if (rules[i].match_prefix != decoder[i].match_prefix) differs = true;
  }
  CHECK(differs);
}

TEST_CASE("ltx2 image cond: a decoder-only checkpoint is REPORTED, not half-loaded") {
  CHECK(!vllm::Ltx2CheckpointHasVideoEncoder(
      {"vae.decoder.conv_in.conv.weight", "vae.per_channel_statistics.std-of-means"}));
  // per_channel_statistics alone is NOT evidence of an encoder: a Comfy-split
  // decoder file carries it too, and treating it as evidence would report every
  // one of them as encodable and then throw deep inside the encoder instead.
  CHECK(!vllm::Ltx2CheckpointHasVideoEncoder({"per_channel_statistics.mean-of-means"}));
  CHECK(vllm::Ltx2CheckpointHasVideoEncoder({"vae.encoder.conv_in.conv.weight"}));
  CHECK(vllm::Ltx2CheckpointHasVideoEncoder({"encoder.conv_in.conv.weight"}));
  CHECK(!vllm::Ltx2CheckpointHasVideoEncoder({}));
}

TEST_CASE("ltx2 image cond: the encoder config reads the LATENT width, not the RGB one") {
  nlohmann::json vae;
  vae["_class_name"] = "CausalVideoAutoencoder";
  vae["dims"] = 3;
  vae["in_channels"] = 3;
  vae["out_channels"] = 3;        // the DECODER's RGB count
  vae["latent_channels"] = 128;   // the ENCODER's latent width
  vae["patch_size"] = 4;
  vae["norm_layer"] = "pixel_norm";
  vae["encoder_blocks"] = nlohmann::json::array({
      nlohmann::json::array({"res_x", {{"num_layers", 2}}}),
      nlohmann::json::array({"compress_all_res", {{"multiplier", 2}}}),
  });
  nlohmann::json config;
  config["vae"] = vae;

  const vllm::Ltx2ConvVideoEncoderConfig cfg = vllm::Ltx2ParseConvVideoEncoderConfig(config);
  CHECK(cfg.out_channels == 128);  // NOT 3
  CHECK(cfg.in_channels == 3);
  CHECK(cfg.patch_size == 4);
  REQUIRE(cfg.encoder_blocks.size() == 2);
  CHECK(cfg.encoder_blocks[0].name == "res_x");
  CHECK(cfg.encoder_blocks[0].num_layers == 2);
  // An ABSENT multiplier stays at the 0 sentinel — becoming 1 would quietly
  // halve every widening block's output width.
  CHECK(cfg.encoder_blocks[0].multiplier == 0);
  CHECK(cfg.encoder_blocks[1].multiplier == 2);
  // THE DEFAULT THAT DIVERGES FROM THE DECODER'S. `spatial_padding_mode` is
  // absent here, so the encoder must take `zeros` while the decoder takes
  // `reflect` (model_configurator.py:63-68 vs :92).
  CHECK(cfg.spatial_padding_mode == vllm::Ltx2PaddingMode::kZeros);
  CHECK(cfg.latent_log_var == vllm::Ltx2LogVarianceType::kUniform);

  SUBCASE("a declared spatial_padding_mode wins over the default") {
    config["vae"]["spatial_padding_mode"] = "reflect";
    CHECK(vllm::Ltx2ParseConvVideoEncoderConfig(config).spatial_padding_mode ==
          vllm::Ltx2PaddingMode::kReflect);
  }
  SUBCASE("the top-level encoder_spatial_padding_mode is the second lookup") {
    config["vae"]["encoder_spatial_padding_mode"] = "replicate";
    CHECK(vllm::Ltx2ParseConvVideoEncoderConfig(config).spatial_padding_mode ==
          vllm::Ltx2PaddingMode::kReplicate);
  }
  SUBCASE("a NESTED CausalDiffusionVAE config reads encoder.out_channels") {
    nlohmann::json nested;
    nested["vae"]["latent_channels"] = 64;
    nested["vae"]["encoder"]["out_channels"] = 96;
    nested["vae"]["encoder"]["blocks"] = config["vae"]["encoder_blocks"];
    CHECK(vllm::Ltx2ParseConvVideoEncoderConfig(nested).out_channels == 96);
  }
  SUBCASE("no encoder_blocks is REFUSED, never defaulted to an empty list") {
    config["vae"].erase("encoder_blocks");
    CHECK_THROWS_WITH_AS(vllm::Ltx2ParseConvVideoEncoderConfig(config),
                         doctest::Contains("encoder_blocks"), std::runtime_error);
  }
  SUBCASE("a 2-D checkpoint is refused rather than built as 3-D") {
    config["vae"]["dims"] = 2;
    CHECK_THROWS_AS(vllm::Ltx2ParseConvVideoEncoderConfig(config), std::runtime_error);
  }
  SUBCASE("an unknown latent_log_var is refused rather than mapped to the nearest") {
    config["vae"]["latent_log_var"] = "gaussian";
    CHECK_THROWS_AS(vllm::Ltx2ParseConvVideoEncoderConfig(config), std::runtime_error);
  }
  SUBCASE("a res_x block WITHOUT num_layers is refused, as upstream's subscript is") {
    // `_make_encoder_block` reads `block_config["num_layers"]` — a SUBSCRIPT
    // (video_vae.py:55) — so a `res_x` block that omits it is a config upstream
    // cannot load either. This parser used to default it to 1, which silently
    // built a one-layer `UNetMidBlock3D` instead; two lines below it, an absent
    // `multiplier` is deliberately kept at a sentinel for exactly that reason.
    // The two are consistent as of the review of #657.
    config["vae"]["encoder_blocks"][0][1].erase("num_layers");
    CHECK_THROWS_WITH_AS(vllm::Ltx2ParseConvVideoEncoderConfig(config),
                         doctest::Contains("num_layers"), std::runtime_error);
  }
  SUBCASE("only res_x needs num_layers; no other block kind reads it") {
    // The other half of the mirror, and the half a bare "make it strict" would
    // get wrong: `_make_encoder_block`'s remaining branches (video_vae.py:61-145)
    // never touch `num_layers`, so requiring it everywhere would refuse configs
    // upstream loads.
    config["vae"]["encoder_blocks"][1][1].erase("multiplier");
    const vllm::Ltx2ConvVideoEncoderConfig lean =
        vllm::Ltx2ParseConvVideoEncoderConfig(config);
    REQUIRE(lean.encoder_blocks.size() == 2);
    CHECK(lean.encoder_blocks[1].name == "compress_all_res");
    CHECK(lean.encoder_blocks[1].num_layers == 1);
    CHECK(lean.encoder_blocks[1].multiplier == 0);
  }
}

// ─── sections 3-5: the chain ────────────────────────────────────────────────

TEST_CASE("ltx2 image cond: the encoder's parameter set IS the generator's state_dict") {
  const ParamBag bag = BuildImageEncoderParams(ImageEncoderConfig());
  REQUIRE(bag.names.size() == std::size(vllm_test::kLtx2ImgEncParamNames));
  REQUIRE(bag.counts.size() == std::size(vllm_test::kLtx2ImgEncParamCounts));
  for (size_t i = 0; i < bag.names.size(); ++i) {
    INFO("parameter " << i);
    CHECK(bag.names[i] == std::string(vllm_test::kLtx2ImgEncParamNames[i]));
    CHECK(bag.counts[i] == vllm_test::kLtx2ImgEncParamCounts[i]);
  }
}

TEST_CASE("ltx2 image cond: PPM to conditioned, noised tokens, against executed upstream") {
  // THE WHOLE CHAIN, driven from the container bytes rather than from any
  // intermediate golden — which is the one thing the per-brick goldens in
  // test_ltx2_vae cannot show.
  const std::vector<float> chw = vllm::Ltx2LoadImageAndPreprocess(
      "first_frame", ConditioningImagePpm(), vllm_test::kLtx2ImgPreDstH,
      vllm_test::kLtx2ImgPreDstW, /*crf=*/0);

  const vllm::Ltx2ConvVideoEncoderConfig cfg = ImageEncoderConfig();
  CHECK(vllm::Ltx2VideoTemporalScaleFactor(cfg.encoder_blocks) ==
        vllm_test::kLtx2ImgEncTemporalFactor);
  CHECK(vllm::Ltx2VideoSpatialScaleFactor(cfg.encoder_blocks, cfg.patch_size) ==
        vllm_test::kLtx2ImgEncSpatialFactor);

  const vllm::Ltx2LatentVolume encoded = EncodeConditioningImage(chw);
  CHECK(encoded.channels == vllm_test::kLtx2ImgEncOutC);
  CHECK(encoded.frames == vllm_test::kLtx2ImgEncOutT);
  CHECK(encoded.height == vllm_test::kLtx2ImgEncOutH);
  CHECK(encoded.width == vllm_test::kLtx2ImgEncOutW);
  {
    const double worst = MaxAbsDiff(encoded.data, vllm_test::kLtx2ImgEncGolden,
                                    std::size(vllm_test::kLtx2ImgEncGolden));
    CAPTURE(worst);
    CHECK(worst <= kLtx2ImgGoldenTol);
  }

  const Composition composed =
      ComposeConditioning(encoded, vllm_test::kLtx2ImgCondStrength, /*condition_before_noise=*/true);

  CHECK(static_cast<int64_t>(composed.mask.size()) == vllm_test::kLtx2ImgCondTokens);
  {
    const double worst = MaxAbsDiff(composed.clean, vllm_test::kLtx2ImgCondClean,
                                    std::size(vllm_test::kLtx2ImgCondClean));
    CAPTURE(worst);
    CHECK(worst <= kLtx2ImgGoldenTol);
  }
  {
    const double worst = MaxAbsDiff(composed.mask, vllm_test::kLtx2ImgCondMask,
                                    std::size(vllm_test::kLtx2ImgCondMask));
    CAPTURE(worst);
    CHECK(worst <= kLtx2ImgGoldenTol);
  }
  {
    const double worst = MaxAbsDiff(composed.noised, vllm_test::kLtx2ImgNoisedGolden,
                                    std::size(vllm_test::kLtx2ImgNoisedGolden));
    CAPTURE(worst);
    CHECK(worst <= kLtx2ImgGoldenTol);
  }
}

// ─── the goldens are LOAD-BEARING: witnesses, not assertions of shape ───────

TEST_CASE("ltx2 image cond: the conditioned latent is what the goldens depend on") {
  // THIS IS THE CLASS THE CAMPAIGN KEEPS FINDING. An earlier phase's
  // conditioning could be scaled x1.5 or row-REVERSED and every assertion in the
  // suite stayed green. Each witness below drives the SAME composition with a
  // deliberately wrong input and asserts the golden distance is enormous, so a
  // reader can see that the comparisons above are sensitive to the thing they
  // claim to be about — rather than inferring it.
  const std::vector<float> chw = vllm::Ltx2LoadImageAndPreprocess(
      "first_frame", ConditioningImagePpm(), vllm_test::kLtx2ImgPreDstH,
      vllm_test::kLtx2ImgPreDstW, /*crf=*/0);
  const vllm::Ltx2LatentVolume encoded = EncodeConditioningImage(chw);

  const size_t noised_n = std::size(vllm_test::kLtx2ImgNoisedGolden);
  const size_t clean_n = std::size(vllm_test::kLtx2ImgCondClean);
  const size_t mask_n = std::size(vllm_test::kLtx2ImgCondMask);

  SUBCASE("the encoded latent scaled x1.5") {
    vllm::Ltx2LatentVolume scaled = encoded;
    for (float& v : scaled.data) v *= 1.5F;
    const Composition c =
        ComposeConditioning(scaled, vllm_test::kLtx2ImgCondStrength, true);
    CHECK(MaxAbsDiff(c.clean, vllm_test::kLtx2ImgCondClean, clean_n) > kLtx2ImgGoldenTol);
    CHECK(MaxAbsDiff(c.noised, vllm_test::kLtx2ImgNoisedGolden, noised_n) > kLtx2ImgGoldenTol);
  }

  SUBCASE("the encoded latent's channels REVERSED — same values, wrong places") {
    vllm::Ltx2LatentVolume reversed = encoded;
    std::reverse(reversed.data.begin(), reversed.data.end());
    const Composition c =
        ComposeConditioning(reversed, vllm_test::kLtx2ImgCondStrength, true);
    CHECK(MaxAbsDiff(c.clean, vllm_test::kLtx2ImgCondClean, clean_n) > kLtx2ImgGoldenTol);
    CHECK(MaxAbsDiff(c.noised, vllm_test::kLtx2ImgNoisedGolden, noised_n) > kLtx2ImgGoldenTol);
  }

  SUBCASE("the mask as `strength` instead of `1 - strength`") {
    // 1 - 0.7 = 0.3 against 0.7. Mask 0 means KEEP the clean value, so inverting
    // it renders an unconditioned clip that looks like the feature not working.
    const Composition c = ComposeConditioning(encoded, 1.0 - vllm_test::kLtx2ImgCondStrength, true);
    CHECK(MaxAbsDiff(c.mask, vllm_test::kLtx2ImgCondMask, mask_n) > kLtx2ImgGoldenTol);
    CHECK(MaxAbsDiff(c.noised, vllm_test::kLtx2ImgNoisedGolden, noised_n) > kLtx2ImgGoldenTol);
    // ...and the CLEAN latent is IDENTICAL, which is why the mask needs its own
    // comparison: a suite that only checked `clean` would be blind to this.
    CHECK(MaxAbsDiff(c.clean, vllm_test::kLtx2ImgCondClean, clean_n) <= kLtx2ImgGoldenTol);
  }

  SUBCASE("the conditioning applied AFTER the noiser") {
    // Upstream conditions the CLEAN tensor and lets the noiser compose it
    // (latent_cond.py:38-39 + noisers.py:31-34). Applying it afterwards leaves
    // the noised latent pinned to noise instead of to the image — with the
    // identical clean tensor and the identical mask, so only section 5 sees it.
    const Composition c =
        ComposeConditioning(encoded, vllm_test::kLtx2ImgCondStrength, /*condition_before_noise=*/false);
    CHECK(MaxAbsDiff(c.clean, vllm_test::kLtx2ImgCondClean, clean_n) <= kLtx2ImgGoldenTol);
    CHECK(MaxAbsDiff(c.mask, vllm_test::kLtx2ImgCondMask, mask_n) <= kLtx2ImgGoldenTol);
    CHECK(MaxAbsDiff(c.noised, vllm_test::kLtx2ImgNoisedGolden, noised_n) > kLtx2ImgGoldenTol);
  }
}

TEST_CASE("ltx2 image cond: the noiser follows ltx_core, and diffusers is far away") {
  // `kLtx2ImgNoiseDivergence` is max|ltx_core - diffusers| at this NON-UNIT noise
  // scale, measured by the generator against both compositions. Asserting it is
  // orders of magnitude above the band is what makes section 5 a gate on the
  // CHOICE spec §3.3 makes, rather than a gate that would pass either way.
  CHECK(vllm_test::kLtx2ImgNoiseDivergence > 1000.0 * kLtx2ImgGoldenTol);
  CHECK(vllm_test::kLtx2ImgNoiseScale != 1.0);  // the only scale at which they agree
}
