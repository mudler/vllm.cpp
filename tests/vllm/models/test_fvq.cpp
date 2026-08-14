// EnhancedCodec quantizer parity gate (#634).
//
// The DISCRETE output here is the semantic code the talker consumes, so an
// index that is off by one entry is a different utterance -- indices are gated
// EXACTLY, not within a tolerance.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "fvq_goldens.inc"
#include "vllm/model_executor/models/fvq.h"
#include "vllm/model_executor/models/vocoder1d.h"

namespace {
using namespace fvq_goldens;

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

vllm::models::fvq::Weights BuildWeights() {
  std::map<std::string, std::vector<float>> m;
  for (int64_t i = 0; i < kManifestSize; ++i) {
    const auto& e = kManifest[i];
    const int64_t d[3] = {e.d0, e.d1, e.d2};
    int64_t n = 1;
    for (int64_t k = 0; k < e.rank; ++k) n *= d[k];
    m[e.name] = Rand(e.name, n, 0.3);
  }
  vllm::models::fvq::Weights w;
  w.in_g = m.at("in_project.weight_g");
  w.in_v = m.at("in_project.weight_v");
  w.in_bias = m.at("in_project.bias");
  w.out_g = m.at("out_project.weight_g");
  w.out_v = m.at("out_project.weight_v");
  w.out_bias = m.at("out_project.bias");
  w.codebook = m.at("codebook.weight");
  return w;
}
}  // namespace

TEST_CASE("fvq indices match upstream EXACTLY") {
  const vllm::models::fvq::Weights w = BuildWeights();
  const std::vector<float> z(std::begin(kZ), std::end(kZ));
  const auto r = vllm::models::fvq::Quantize(z, kFrames, kInputDim, kCodebookDim, kCodebookSize, w);
  REQUIRE(r.indices.size() == static_cast<size_t>(kFrames));
  for (int64_t t = 0; t < kFrames; ++t) {
    // An index off by one entry is a different utterance, not a small error.
    CHECK(r.indices[static_cast<size_t>(t)] == kIndices[t]);
  }
}

TEST_CASE("fvq dequantized output matches upstream") {
  // Distances are computed on NORMALIZED vectors, but the entry returned is the
  // RAW codebook row. Returning the normalized row keeps the indices identical
  // and only moves these values, so this case is what separates the two.
  const vllm::models::fvq::Weights w = BuildWeights();
  const std::vector<float> z(std::begin(kZ), std::end(kZ));
  const auto r = vllm::models::fvq::Quantize(z, kFrames, kInputDim, kCodebookDim, kCodebookSize, w);
  REQUIRE(r.z_q.size() == static_cast<size_t>(kInputDim * kFrames));
  double worst = 0.0;
  for (size_t i = 0; i < r.z_q.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(r.z_q[i]) - kZq[i]));
  }
  INFO("max abs diff vs upstream z_q: ", worst);
  CHECK(worst < 1e-5);
}

TEST_CASE("fvq weight-norm materialization is g * v / ||v||") {
  // The checkpoint stores (g, v), never the effective weight. Getting the norm
  // axis wrong still produces a weight of the right shape.
  const std::vector<float> g{2.0F, 3.0F};
  const std::vector<float> v{3.0F, 4.0F, 0.0F, 5.0F};  // rows [3,4] (norm 5), [0,5] (norm 5)
  const std::vector<float> w = vllm::vocoder1d::MaterializeWeightNorm(g, v, 2);
  REQUIRE(w.size() == 4U);
  CHECK(w[0] == doctest::Approx(2.0 * 3.0 / 5.0).epsilon(1e-6));
  CHECK(w[1] == doctest::Approx(2.0 * 4.0 / 5.0).epsilon(1e-6));
  CHECK(w[2] == doctest::Approx(0.0).epsilon(1e-6));
  CHECK(w[3] == doctest::Approx(3.0 * 5.0 / 5.0).epsilon(1e-6));
}
