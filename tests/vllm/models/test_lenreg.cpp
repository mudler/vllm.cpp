// S2Mel length-regulator primitives (#634), gated against torch.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "lenreg_goldens.inc"
#include "vllm/model_executor/models/lenreg.h"

namespace {
using namespace lenreg_goldens;

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
double Worst(const std::vector<float>& g, const float* w, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(static_cast<double>(g[i]) - w[i]));
  return m;
}
}  // namespace

TEST_CASE("lenreg nearest interpolate matches torch at a NON-INTEGER ratio") {
  // 7 -> 17 is the case that separates floor(i*in/out) from a rounded or
  // half-offset rule; both alternatives agree at exact multiples.
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> got =
      vllm::models::lenreg::InterpolateNearest(x, kChannels, kInFrames, 17);
  REQUIRE(got.size() == static_cast<size_t>(kChannels * 17));
  // Exact: interpolation COPIES samples, it never blends them.
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == kInterp17[i]);
}

TEST_CASE("lenreg nearest interpolate matches torch at an exact multiple") {
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> got =
      vllm::models::lenreg::InterpolateNearest(x, kChannels, kInFrames, 14);
  REQUIRE(got.size() == static_cast<size_t>(kChannels * 14));
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == kInterp14[i]);
}

TEST_CASE("lenreg nearest interpolate DOWNsamples the same way") {
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> got =
      vllm::models::lenreg::InterpolateNearest(x, kChannels, kInFrames, 3);
  REQUIRE(got.size() == static_cast<size_t>(kChannels * 3));
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == kInterp3[i]);
}

TEST_CASE("lenreg GroupNorm shares statistics across a GROUP, not a channel") {
  // Per-channel statistics would also normalize, and would be a different
  // model; only the values distinguish them.
  const std::vector<float> x(std::begin(kX), std::end(kX));
  std::vector<float> gamma = Rand("gn.weight", kChannels, 0.5);
  for (float& g : gamma) g += 1.0F;
  const std::vector<float> beta = Rand("gn.bias", kChannels, 0.3);
  const std::vector<float> got = vllm::models::lenreg::GroupNorm(
      x, kChannels, kInFrames, kGroups, gamma, beta, 1e-5);
  REQUIRE(got.size() == x.size());
  CHECK(Worst(got, kGroupNorm, got.size()) < 1e-5);
}

TEST_CASE("lenreg Mish is x * tanh(softplus(x)), not SiLU") {
  const std::vector<float> in(std::begin(kMishIn), std::end(kMishIn));
  std::vector<float> got(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    got[i] = static_cast<float>(vllm::models::lenreg::Mish(static_cast<double>(in[i])));
  }
  CHECK(Worst(got, kMishOut, got.size()) < 1e-6);
}
