// adaLN / FinalLayer parity gate — how S2Mel's DiT is conditioned (#634).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "adaln_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/adaln.h"

namespace {
using namespace adaln_goldens;

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

vllm::models::adaln::FinalLayerWeights Build() {
  std::map<std::string, std::vector<float>> m;
  for (int64_t i = 0; i < kManifestSize; ++i) {
    const auto& e = kManifest[i];
    const int64_t d[3] = {e.d0, e.d1, e.d2};
    int64_t n = 1;
    for (int64_t k = 0; k < e.rank; ++k) n *= d[k];
    m[e.name] = Rand(e.name, n, 0.3);
  }
  vllm::models::adaln::FinalLayerWeights w;
  w.ada_w = m.at("adaLN_modulation.1.weight");
  w.ada_b = m.at("adaLN_modulation.1.bias");
  w.linear_g = m.at("linear.weight_g");
  w.linear_v = m.at("linear.weight_v");
  w.linear_bias = m.at("linear.bias");
  return w;
}
double Worst(const std::vector<float>& g, const float* w, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(static_cast<double>(g[i]) - w[i]));
  return m;
}
}  // namespace

TEST_CASE("adaln FinalLayer matches upstream") {
  const auto w = Build();
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> c(std::begin(kC), std::end(kC));
  const std::vector<float> got =
      vllm::models::adaln::FinalLayer(x, kFrames, kHidden, kOutCh, c, w, 1e-6);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kOutCh));
  const double worst = Worst(got, kFinalOut, got.size());
  INFO("max abs diff vs upstream FinalLayer: ", worst);
  CHECK(worst < 1e-5);
}

TEST_CASE("adaln modulate is x * (1 + scale) + shift") {
  // Hand-computed, independent of the goldens. With scale = 0 the modulation
  // must be the IDENTITY plus shift -- dropping the `1 +` would zero x instead,
  // which is the whole point of the case.
  const int64_t T = 2, H = 3;
  const std::vector<float> x{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<float> shift{10.0F, 20.0F, 30.0F};
  const std::vector<float> zero_scale{0.0F, 0.0F, 0.0F};
  const std::vector<float> got = vllm::models::adaln::Modulate(x, T, H, shift, zero_scale);
  const std::vector<float> want{11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F};
  for (size_t i = 0; i < want.size(); ++i) CHECK(got[i] == want[i]);

  // And with scale = 1 every element doubles before the shift.
  const std::vector<float> one_scale{1.0F, 1.0F, 1.0F};
  const std::vector<float> doubled = vllm::models::adaln::Modulate(x, T, H, shift, one_scale);
  CHECK(doubled[0] == 12.0F);   // 1*(1+1) + 10
  CHECK(doubled[4] == 30.0F);   // 5*(1+1) + 20
}

TEST_CASE("adaln layer norm applies NO affine parameters") {
  // elementwise_affine=False: the checkpoint has no gamma/beta for this norm, so
  // the output must be exactly zero-mean and unit-variance per frame.
  const std::vector<float> x = Rand("lnx", 4 * 8, 2.0);
  const std::vector<float> got = vllm::models::adaln::LayerNormNoAffine(x, 4, 8, 1e-6);
  for (int64_t t = 0; t < 4; ++t) {
    double mean = 0.0, sq = 0.0;
    for (int64_t h = 0; h < 8; ++h) {
      const double v = got[static_cast<size_t>(t * 8 + h)];
      mean += v; sq += v * v;
    }
    mean /= 8.0; sq /= 8.0;
    CHECK(std::fabs(mean) < 1e-5);
    CHECK(std::fabs(sq - 1.0) < 1e-4);
  }
}

TEST_CASE("adaln chunk order is shift THEN scale") {
  // The modulation Linear emits 2*hidden values chunking into [shift, scale].
  // Swapping them is invisible to shapes and yields a plausible model.
  //
  // Pinned by driving the halves to values whose EFFECT differs: with the whole
  // weight zeroed, shift and scale come from the bias alone. Set shift = 0 and
  // scale = -1, so `x * (1 + -1) + 0` collapses to ZERO and the output is
  // exactly the linear bias. Under the swapped order the modulation would be
  // `x * (1 + 0) + (-1)`, which is not zero and gives a different answer.
  auto w = Build();
  const int64_t H = kHidden;
  for (float& v : w.ada_w) v = 0.0F;
  for (int64_t i = 0; i < H; ++i) {
    w.ada_b[static_cast<size_t>(i)] = 0.0F;          // shift half
    w.ada_b[static_cast<size_t>(H + i)] = -1.0F;     // scale half
  }
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> c(static_cast<size_t>(H), 0.0F);
  const std::vector<float> got =
      vllm::models::adaln::FinalLayer(x, kFrames, kHidden, kOutCh, c, w, 1e-6);
  for (int64_t t = 0; t < kFrames; ++t) {
    for (int64_t o = 0; o < kOutCh; ++o) {
      CHECK(got[static_cast<size_t>(t * kOutCh + o)] ==
            doctest::Approx(w.linear_bias[static_cast<size_t>(o)]).epsilon(1e-6));
    }
  }
}
