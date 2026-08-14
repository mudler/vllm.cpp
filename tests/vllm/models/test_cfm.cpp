// S2Mel flow-matching scaffolding (#634).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "cfm_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/cfm.h"

namespace {
using namespace cfm_goldens;
double Worst(const std::vector<float>& g, const float* w, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(static_cast<double>(g[i]) - w[i]));
  return m;
}
}  // namespace

TEST_CASE("cfm timestep features put COSINE first and scale t by 1000") {
  // Swapping the halves yields an embedding that is still smooth and periodic,
  // and completely wrong; dropping the 1000x scale changes which part of the
  // frequency sweep each timestep lands on. Both are invisible to any shape or
  // finiteness check.
  const std::vector<float> t(std::begin(kT), std::end(kT));
  const std::vector<float> got =
      vllm::models::cfm::TimestepFeatures(t, kFreqDim, kMaxPeriod, kScale);
  REQUIRE(got.size() == static_cast<size_t>(kNumT * kFreqDim));
  const double worst = Worst(got, kRawEmbedding, got.size());
  INFO("max abs diff vs upstream timestep_embedding: ", worst);
  // 2e-5, and the reason is measured rather than guessed. With scale = 1000 the
  // arguments reach ~130, where float32's spacing is ~7.6e-6; upstream keeps the
  // frequency table AND the argument in float32, so torch's input to cos/sin
  // already carries that much representation error. A double-precision port
  // therefore cannot agree below ~6.5e-6 no matter how faithful it is. The bound
  // is set just above the measured floor -- still tight enough that a swapped
  // cos/sin ordering or a missing 1000x scale, which move values by O(1), fail
  // loudly. The ordering additionally has its own exact case below.
  CHECK(worst < 2e-5);
}

TEST_CASE("cfm timestep features at t=0 are cos=1 then sin=0") {
  // A direct structural read of the ordering, independent of the goldens: at
  // t = 0 every argument is 0, so the first half must be ALL ONES and the
  // second ALL ZEROS. Sine-first would invert that exactly.
  const std::vector<float> t{0.0F};
  const std::vector<float> got =
      vllm::models::cfm::TimestepFeatures(t, kFreqDim, kMaxPeriod, kScale);
  for (int64_t i = 0; i < kFreqDim / 2; ++i) {
    CHECK(got[static_cast<size_t>(i)] == 1.0F);
    CHECK(got[static_cast<size_t>(kFreqDim / 2 + i)] == 0.0F);
  }
}

TEST_CASE("cfm euler step applies CFG and RE-ZEROES the prompt region") {
  // dphi = (1+rate)*cond - rate*uncond, then x += dt*dphi, then the prompt
  // frames are zeroed. Skipping the zeroing still yields a mel of the right
  // shape while the solver integrates over frames the prompt was meant to pin.
  const int64_t C = 2, T = 5, PROMPT = 2;
  std::vector<float> x(static_cast<size_t>(C * T), 1.0F);
  std::vector<float> cond(static_cast<size_t>(C * T), 2.0F);
  std::vector<float> uncond(static_cast<size_t>(C * T), 0.5F);
  const double dt = 0.25, rate = 0.7;

  const std::vector<float> got =
      vllm::models::cfm::EulerStepCfg(x, cond, uncond, C, T, dt, rate, PROMPT);
  REQUIRE(got.size() == x.size());

  // Hand-computed: dphi = 1.7*2 - 0.7*0.5 = 3.4 - 0.35 = 3.05; x = 1 + 0.25*3.05 = 1.7625
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t t = 0; t < T; ++t) {
      const float v = got[static_cast<size_t>(c * T + t)];
      if (t < PROMPT) {
        CHECK(v == 0.0F);
      } else {
        CHECK(v == doctest::Approx(1.7625).epsilon(1e-6));
      }
    }
  }
}

TEST_CASE("cfm with rate 0 reduces to a plain Euler step") {
  const int64_t C = 1, T = 3;
  const std::vector<float> x(static_cast<size_t>(C * T), 2.0F);
  const std::vector<float> cond(static_cast<size_t>(C * T), 4.0F);
  const std::vector<float> uncond(static_cast<size_t>(C * T), 99.0F);  // must be ignored
  const std::vector<float> got =
      vllm::models::cfm::EulerStepCfg(x, cond, uncond, C, T, 0.5, 0.0, 0);
  for (const float v : got) CHECK(v == doctest::Approx(4.0).epsilon(1e-6));  // 2 + 0.5*4
}
