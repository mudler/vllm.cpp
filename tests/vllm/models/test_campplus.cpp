// CAMPPlus primitives — W3 of #634, the speaker-style encoder on the MANDATORY
// reference-audio path (the talker is built with spk_cond_mode="campplus", so
// this feeds it).
//
// Gated against `indextts/s2mel/modules/campplus/layers.py` executed DIRECTLY:
// it has no vllm dependency, so the goldens come from the real classes rather
// than a restatement. Weights are rebuilt both sides from one FNV-1a ->
// splitmix64 stream.
//
// THREE THINGS HERE FAIL SILENTLY, so each has its own case:
//   1. std is UNBIASED (N-1). The biased form differs by ~0.2% at T=250 — small
//      enough to look like noise, large enough to move a style vector.
//   2. BatchNorm1d in EVAL uses RUNNING statistics. Using batch statistics still
//      normalizes, and is a different model.
//   3. seg_pooling uses ceil_mode, expands each segment back over seg_len frames
//      and TRUNCATES to the input length. T=250 gives 3 segments, the last one
//      partial, which is where an off-by-one lives.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "campplus_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/campplus.h"

namespace {
using namespace campplus_goldens;

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
double Worst(const std::vector<float>& got, const float* want, size_t n) {
  double w = 0.0;
  for (size_t i = 0; i < n; ++i) w = std::max(w, std::fabs(static_cast<double>(got[i]) - want[i]));
  return w;
}
}  // namespace

TEST_CASE("campplus StatsPool concatenates mean and UNBIASED std") {
  const std::vector<float> x = Rand("x", kChannels * kFrames, 1.0);
  const std::vector<float> got = vllm::models::campplus::StatsPool(x, kChannels, kFrames);
  REQUIRE(got.size() == static_cast<size_t>(2 * kChannels));
  const double w = Worst(got, kStats, got.size());
  INFO("max abs diff vs upstream StatsPool: ", w);
  CHECK(w < 1e-5);
}

TEST_CASE("campplus BatchNorm1d in eval uses RUNNING statistics") {
  const std::vector<float> x = Rand("x", kChannels * kFrames, 1.0);
  std::vector<float> gamma = Rand("bn.w", kChannels, 0.5);
  for (float& g : gamma) g += 1.0F;
  const std::vector<float> beta = Rand("bn.b", kChannels, 0.3);
  const std::vector<float> mean = Rand("bn.rm", kChannels, 0.4);
  std::vector<float> var = Rand("bn.rv", kChannels, 0.2);
  for (float& v : var) v = std::fabs(v) + 0.5F;

  const std::vector<float> got =
      vllm::models::campplus::BatchNorm1dEval(x, kChannels, kFrames, gamma, beta, mean, var, 1e-5);
  REQUIRE(got.size() == x.size());
  const double w = Worst(got, kBatchNormEval, got.size());
  INFO("max abs diff vs upstream BatchNorm1d(eval): ", w);
  CHECK(w < 1e-5);
}

TEST_CASE("campplus seg_pooling expands ceil-mode segments and truncates") {
  const std::vector<float> x = Rand("xb", kBnChannels * kFrames, 1.0);
  const std::vector<float> got =
      vllm::models::campplus::SegPooling(x, kBnChannels, kFrames, kSegLen);
  // The expansion must return the INPUT length, not a multiple of seg_len.
  REQUIRE(got.size() == x.size());
  const double w = Worst(got, kSegPooling, got.size());
  INFO("max abs diff vs upstream seg_pooling: ", w);
  CHECK(w < 1e-5);
}

TEST_CASE("campplus CAMLayer gates the local branch by the pooled context") {
  const std::vector<float> x = Rand("xb", kBnChannels * kFrames, 1.0);
  vllm::models::campplus::CamLayerWeights w;
  w.linear_local = Rand("cam.ll.w", kOutChannels * kBnChannels * kKernel, 0.3);
  w.linear1_weight = Rand("cam.l1.w", (kBnChannels / 2) * kBnChannels, 0.3);
  w.linear1_bias = Rand("cam.l1.b", kBnChannels / 2, 0.2);
  w.linear2_weight = Rand("cam.l2.w", kOutChannels * (kBnChannels / 2), 0.3);
  w.linear2_bias = Rand("cam.l2.b", kOutChannels, 0.2);

  const std::vector<float> got = vllm::models::campplus::CamLayer(
      x, kBnChannels, kFrames, kOutChannels, kKernel, kDilation, kSegLen, w);
  REQUIRE(got.size() == static_cast<size_t>(kOutChannels * kFrames));
  const double d = Worst(got, kCamOut, got.size());
  INFO("max abs diff vs upstream CAMLayer: ", d);
  CHECK(d < 1e-5);
}
