// VocosBackbone parity gate — EnhancedCodec's ConvNeXt-1D encoder (#634).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/vocos.h"
#include "vocos_goldens.inc"

namespace {
using namespace vocos_goldens;

std::vector<float> Rand(const std::string& name, int64_t n, double scale) {
  uint64_t seed = 0xCBF29CE484222325ULL;
  for (char c : name) { seed ^= static_cast<unsigned char>(c); seed *= 0x100000001B3ULL; }
  std::vector<float> o(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    uint64_t x = seed + static_cast<uint64_t>(i);
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    o[static_cast<size_t>(i)] =
        static_cast<float>(((static_cast<double>(z >> 11) * 0x1.0p-53) * 2.0 - 1.0) * scale);
  }
  return o;
}

vllm::models::vocos::BackboneWeights Build() {
  std::map<std::string, std::vector<float>> m;
  for (int64_t i = 0; i < kManifestSize; ++i) {
    const auto& e = kManifest[i];
    const int64_t d[3] = {e.d0, e.d1, e.d2};
    int64_t n = 1;
    for (int64_t k = 0; k < e.rank; ++k) n *= d[k];
    m[e.name] = Rand(e.name, n, std::string(e.name).find("pwconv1.weight") != std::string::npos ? 1.0 : 0.3);
  }
  vllm::models::vocos::BackboneWeights w;
  w.embed_w = m.at("embed.weight"); w.embed_b = m.at("embed.bias");
  w.norm_gamma = m.at("norm.weight"); w.norm_beta = m.at("norm.bias");
  w.final_gamma = m.at("final_layer_norm.weight");
  w.final_beta = m.at("final_layer_norm.bias");
  for (int64_t i = 0; i < kLayers; ++i) {
    const std::string p = "convnext." + std::to_string(i) + ".";
    vllm::models::vocos::BlockWeights b;
    b.dw_weight = m.at(p + "dwconv.weight"); b.dw_bias = m.at(p + "dwconv.bias");
    b.ln_gamma = m.at(p + "norm.weight"); b.ln_beta = m.at(p + "norm.bias");
    b.pw1_w = m.at(p + "pwconv1.weight"); b.pw1_b = m.at(p + "pwconv1.bias");
    b.pw2_w = m.at(p + "pwconv2.weight"); b.pw2_b = m.at(p + "pwconv2.bias");
    b.gamma = m.at(p + "gamma");
    w.blocks.push_back(std::move(b));
  }
  return w;
}
}  // namespace

TEST_CASE("vocos backbone matches upstream and returns [T, dim]") {
  // The output orientation is the trap: the final layer norm runs on the
  // TRANSPOSED tensor and is never transposed back. Returning [dim, T] has the
  // same element count, so only the layout assertion and the values catch it.
  const auto w = Build();
  const std::vector<float> x(std::begin(kVx), std::end(kVx));
  const std::vector<float> got =
      vllm::models::vocos::Backbone(x, kInChannels, kFrames, kDim, kIntermediate, w, 1e-6);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kDim));
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - kVocosOut[i]));
  }
  INFO("max abs diff vs upstream VocosBackbone: ", worst);
  // 2e-6, not 2e-5. Measured: the exact-erf vs tanh-approx GELU difference
  // reaches only 1.41e-5 on this fixture, so a 2e-5 tolerance accepts the wrong
  // activation outright -- found by mutation. The correct port agrees far more
  // tightly than that, so the bound is set by what the DEFECT costs, not by
  // what felt safe.
  CHECK(worst < 2e-6);
}

TEST_CASE("vocos block residual keeps a zeroed layer scale as the identity") {
  // gamma is a LEARNED per-channel scale on the block's branch. With gamma = 0
  // the block must reduce to exactly its input -- which pins that the residual
  // is added AFTER the scale, not before.
  auto w = Build();
  for (float& g : w.blocks[0].gamma) g = 0.0F;
  const std::vector<float> x = Rand("blockx", kDim * kFrames, 4.0);
  const std::vector<float> got = vllm::models::vocos::ConvNeXtBlock(
      x, kDim, kFrames, kIntermediate, w.blocks[0], 1e-6);
  REQUIRE(got.size() == x.size());
  for (size_t i = 0; i < x.size(); ++i) CHECK(got[i] == x[i]);
}
