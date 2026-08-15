// WaveNet (S2Mel DiT final layer) against upstream goldens. See wavenet.h.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/wavenet.h"
#include "wavenet_goldens.inc"

namespace {

// The generator's FNV-1a -> splitmix64 stream, repeated bit-for-bit so no
// weight bytes are committed.
std::vector<float> Rnd(const std::string& name, size_t n, double scale) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char ch : name) {
    h = (h ^ ch) * 0x100000001B3ULL;
  }
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    h += 0x9E3779B97F4A7C15ULL;
    uint64_t z = h;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    out[i] = static_cast<float>((u * 2.0 - 1.0) * scale);
  }
  return out;
}

using namespace wavenet_goldens;

vllm::models::wavenet::Config Cfg() {
  vllm::models::wavenet::Config c;
  c.hidden = kHidden;
  c.kernel = kKernel;
  c.dilation_rate = kDilationRate;
  c.layers = kLayers;
  c.gin = kGin;
  return c;
}

vllm::models::wavenet::ConvWeights Conv(const std::string& prefix, int64_t out_ch,
                                        int64_t in_ch, int64_t kernel) {
  vllm::models::wavenet::ConvWeights c;
  // torch weight_norm over dim 0: g is [out, 1, 1], v is [out, in, kernel].
  c.g = Rnd(prefix + ".weight_g", static_cast<size_t>(out_ch), 0.5);
  c.v = Rnd(prefix + ".weight_v", static_cast<size_t>(out_ch * in_ch * kernel), 0.5);
  c.bias = Rnd(prefix + ".bias", static_cast<size_t>(out_ch), 0.5);
  return c;
}

vllm::models::wavenet::Weights BuildWeights() {
  vllm::models::wavenet::Weights w;
  w.cond = Conv("wn.cond_layer.conv.conv", 2 * kHidden * kLayers, kGin, 1);
  for (int64_t i = 0; i < kLayers; ++i) {
    const std::string idx = std::to_string(i);
    w.in_layers.push_back(
        Conv("wn.in_layers." + idx + ".conv.conv", 2 * kHidden, kHidden, kKernel));
    const int64_t res_skip_ch = (i < kLayers - 1) ? 2 * kHidden : kHidden;
    w.res_skip_layers.push_back(
        Conv("wn.res_skip_layers." + idx + ".conv.conv", res_skip_ch, kHidden, 1));
  }
  return w;
}

}  // namespace

TEST_CASE("wavenet matches upstream WN with conditioning") {
  const std::vector<float> x = Rnd("wn.x", static_cast<size_t>(kHidden * kFrames), 1.0);
  const std::vector<float> g = Rnd("wn.g", static_cast<size_t>(kGin), 1.0);

  const std::vector<float> got =
      vllm::models::wavenet::Forward(Cfg(), BuildWeights(), x, kFrames, g, {});

  REQUIRE(got.size() == static_cast<size_t>(kHidden * kFrames));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(kOutCond[i]).epsilon(2e-6));
  }
}

TEST_CASE("wavenet masks the residual and the output, but not the conv input") {
  const std::vector<float> x = Rnd("wn.x", static_cast<size_t>(kHidden * kFrames), 1.0);
  const std::vector<float> g = Rnd("wn.g", static_cast<size_t>(kGin), 1.0);
  const std::vector<float> mask(kMask, kMask + kFrames);

  const std::vector<float> got =
      vllm::models::wavenet::Forward(Cfg(), BuildWeights(), x, kFrames, g, mask);

  REQUIRE(got.size() == static_cast<size_t>(kHidden * kFrames));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(kOutMasked[i]).epsilon(2e-6));
  }

  // The masked frames must be exactly zero, and the kept frames must NOT equal
  // the unmasked run: the mask feeds back through the residual, so it changes
  // frames it does not itself zero.
  bool kept_differs = false;
  for (int64_t c = 0; c < kHidden; ++c) {
    for (int64_t t = 0; t < kFrames; ++t) {
      const size_t i = static_cast<size_t>(c * kFrames + t);
      if (kMask[t] == 0.0F) {
        CHECK(got[i] == 0.0F);
      } else if (std::fabs(got[i] - kOutCond[i]) > 1e-6F) {
        kept_differs = true;
      }
    }
  }
  CHECK(kept_differs);
}

TEST_CASE("the last res_skip layer is narrower than the rest") {
  // Upstream: the final layer needs no residual half, so it emits `hidden`
  // rather than `2 * hidden`. Building it wide would silently read the wrong
  // slice, so the contract is asserted on the weights themselves.
  const vllm::models::wavenet::Weights w = BuildWeights();
  REQUIRE(w.res_skip_layers.size() == static_cast<size_t>(kLayers));
  for (int64_t i = 0; i < kLayers - 1; ++i) {
    CHECK(w.res_skip_layers[static_cast<size_t>(i)].g.size() ==
          static_cast<size_t>(2 * kHidden));
  }
  CHECK(w.res_skip_layers[static_cast<size_t>(kLayers - 1)].g.size() ==
        static_cast<size_t>(kHidden));
}
