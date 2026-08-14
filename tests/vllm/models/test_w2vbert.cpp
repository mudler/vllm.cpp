// w2v-bert-2.0 Conformer parity gate — IndexTTS-2.5's semantic front end (#634).
//
// Gated against HuggingFace `transformers` executed DIRECTLY: the class
// `infer_v2_5.py:174` instantiates, so there is no restatement. Weights are
// rebuilt from the same FNV-1a -> splitmix64 stream the generator used, keyed by
// the upstream parameter NAME.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/w2vbert.h"
#include "w2vbert_goldens.inc"

namespace {
using namespace w2vbert_goldens;

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

std::map<std::string, std::vector<float>> Weights() {
  std::map<std::string, std::vector<float>> w;
  for (int64_t i = 0; i < kManifestSize; ++i) {
    const auto& e = kManifest[i];
    const int64_t d[3] = {e.d0, e.d1, e.d2};
    int64_t n = 1;
    for (int64_t k = 0; k < e.rank; ++k) n *= d[k];
    w[e.name] = Rand(e.name, n, 0.3);
  }
  return w;
}

double Worst(const std::vector<float>& got, const float* want, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(static_cast<double>(got[i]) - want[i]));
  return m;
}
}  // namespace

TEST_CASE("w2vbert FeedForward matches upstream (swish, not gelu)") {
  const auto w = Weights();
  const std::vector<float> x = Rand("hidden", kFrames * kHidden, 1.0);
  // ffn1 runs on the LAYER-NORMED input, mirroring the encoder layer.
  const std::vector<float> normed = vllm::models::w2vbert::LayerNorm(
      x, kFrames, kHidden, w.at("ffn1_layer_norm.weight"), w.at("ffn1_layer_norm.bias"),
      kLayerNormEps);
  const std::vector<float> got = vllm::models::w2vbert::FeedForward(
      normed, kFrames, kHidden, kIntermediate, w.at("ffn1.intermediate_dense.weight"),
      w.at("ffn1.intermediate_dense.bias"), w.at("ffn1.output_dense.weight"),
      w.at("ffn1.output_dense.bias"));
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  CHECK(Worst(got, kFfn1Out, got.size()) < 1e-5);
}

TEST_CASE("w2vbert ConvModule matches upstream, including the CAUSAL left pad") {
  // The pad is (kernel-1, 0). A symmetric pad yields the same shape while
  // letting every frame see the future -- no length or shape check can see it,
  // which is why this is gated on values against the real module.
  const auto w = Weights();
  vllm::models::w2vbert::ConvModuleWeights c;
  c.ln_gamma = w.at("conv_module.layer_norm.weight");
  c.ln_beta = w.at("conv_module.layer_norm.bias");
  c.pointwise1 = w.at("conv_module.pointwise_conv1.weight");
  c.depthwise = w.at("conv_module.depthwise_conv.weight");
  c.dw_ln_gamma = w.at("conv_module.depthwise_layer_norm.weight");
  c.dw_ln_beta = w.at("conv_module.depthwise_layer_norm.bias");
  c.pointwise2 = w.at("conv_module.pointwise_conv2.weight");

  const std::vector<float> x(std::begin(kConvIn), std::end(kConvIn));
  const std::vector<float> got =
      vllm::models::w2vbert::ConvModule(x, kFrames, kHidden, kConvKernel, c, kLayerNormEps);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  CHECK(Worst(got, kConvOut, got.size()) < 1e-5);
}

TEST_CASE("w2vbert conv module cannot see the future") {
  // Direct causality probe, independent of the goldens: perturbing the LAST
  // frame must leave every earlier output bit-identical.
  const auto w = Weights();
  vllm::models::w2vbert::ConvModuleWeights c;
  c.ln_gamma = w.at("conv_module.layer_norm.weight");
  c.ln_beta = w.at("conv_module.layer_norm.bias");
  c.pointwise1 = w.at("conv_module.pointwise_conv1.weight");
  c.depthwise = w.at("conv_module.depthwise_conv.weight");
  c.dw_ln_gamma = w.at("conv_module.depthwise_layer_norm.weight");
  c.dw_ln_beta = w.at("conv_module.depthwise_layer_norm.bias");
  c.pointwise2 = w.at("conv_module.pointwise_conv2.weight");

  std::vector<float> x(std::begin(kConvIn), std::end(kConvIn));
  const std::vector<float> base =
      vllm::models::w2vbert::ConvModule(x, kFrames, kHidden, kConvKernel, c, kLayerNormEps);
  x[static_cast<size_t>((kFrames - 1) * kHidden)] += 3.0F;
  const std::vector<float> moved =
      vllm::models::w2vbert::ConvModule(x, kFrames, kHidden, kConvKernel, c, kLayerNormEps);

  // NOTE: the module's leading layer_norm is per-frame, so a change to the last
  // frame cannot leak backwards through it either.
  for (size_t i = 0; i < static_cast<size_t>((kFrames - 1) * kHidden); ++i) {
    CHECK(base[i] == moved[i]);
  }
  double delta = 0.0;
  for (size_t i = static_cast<size_t>((kFrames - 1) * kHidden); i < base.size(); ++i) {
    delta = std::max(delta, std::fabs(static_cast<double>(base[i] - moved[i])));
  }
  CHECK(delta > 1e-6);  // the last frame MUST move, or the probe proves nothing
}
