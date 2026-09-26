// GLiNER2.5-Decide head test: classification head forward + softmax gates.
//
// Tests the host-side classification head (Linear(H,2H) → ReLU → Linear(2H,1))
// and softmax/sigmoid activations. Uses a deterministic PRNG for reproducible
// weight generation (same pattern as test_kev.cpp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <doctest/doctest.h>
#include <numeric>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gliner25_decide.h"

namespace {

std::vector<float> Rand(const std::string& name, int64_t count, double scale) {
  uint64_t seed = 0xCBF29CE484222325ULL;
  for (const char c : name) {
    seed ^= static_cast<unsigned char>(c);
    seed *= 0x100000001B3ULL;
  }
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    uint64_t x = seed + static_cast<uint64_t>(i);
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = static_cast<double>(z >> 11) * 0x1.0p-53;
    out[static_cast<size_t>(i)] = static_cast<float>((u * 2.0 - 1.0) * scale);
  }
  return out;
}

bool ApproxEqual(const std::vector<float>& a, const std::vector<float>& b,
                 float tol) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > tol) return false;
  }
  return true;
}

vllm::gliner25_decide::DecideHeadParams TestParams() {
  vllm::gliner25_decide::DecideHeadParams p;
  p.hidden_size = 64;  // small H for fast tests
  p.temperature = 1.0F;
  return p;
}

vllm::gliner25_decide::DecideHeadWeights TestHead(
    const vllm::gliner25_decide::DecideHeadParams& p) {
  vllm::gliner25_decide::DecideHeadWeights hw;
  hw.w0 = Rand("classifier.0.weight", 2 * p.hidden_size * p.hidden_size, 0.3);
  hw.b0 = Rand("classifier.0.bias", 2 * p.hidden_size, 0.3);
  hw.w2 = Rand("classifier.2.weight", 2 * p.hidden_size, 0.3);
  hw.b2 = Rand("classifier.2.bias", 1, 0.3);
  return hw;
}

}  // namespace

// ── Classifier forward ──────────────────────────────────────────────────

TEST_CASE("gliner25_decide.Classifier.produces_correct_count") {
  auto params = TestParams();
  auto hw = TestHead(params);
  int64_t N = 5;
  auto embs = Rand("label_embs", N * params.hidden_size, 1.0);
  auto logits = vllm::gliner25_decide::ClassifierForward(hw, params, embs, N);
  CHECK(static_cast<int64_t>(logits.size()) == N);
}

TEST_CASE("gliner25_decide.Classifier.different_inputs_produce_different_outputs") {
  auto params = TestParams();
  auto hw = TestHead(params);
  int64_t N = 3;
  auto e1 = Rand("embs1", N * params.hidden_size, 1.0);
  auto e2 = Rand("embs2", N * params.hidden_size, 1.0);
  auto l1 = vllm::gliner25_decide::ClassifierForward(hw, params, e1, N);
  auto l2 = vllm::gliner25_decide::ClassifierForward(hw, params, e2, N);
  CHECK_FALSE(ApproxEqual(l1, l2, 1e-4F));
}

TEST_CASE("gliner25_decide.Classifier.zero_input_produces_finite") {
  auto params = TestParams();
  auto hw = TestHead(params);
  int64_t N = 2;
  std::vector<float> zero(static_cast<size_t>(N * params.hidden_size), 0.0F);
  auto logits = vllm::gliner25_decide::ClassifierForward(hw, params, zero, N);
  for (float v : logits) CHECK(std::isfinite(v));
}

// ── Softmax ─────────────────────────────────────────────────────────────

TEST_CASE("gliner25_decide.Softmax.sums_to_one") {
  std::vector<float> logits = {1.0F, 2.0F, 3.0F, 0.5F};
  auto probs = vllm::gliner25_decide::Softmax(logits, 1.0F);
  REQUIRE(probs.size() == 4);
  double sum = 0.0;
  for (float p : probs) sum += p;
  CHECK(std::abs(sum - 1.0) < 1e-5);
}

TEST_CASE("gliner25_decide.Softmax.temperature_scales") {
  std::vector<float> logits = {1.0F, 2.0F, 3.0F};
  auto p1 = vllm::gliner25_decide::Softmax(logits, 1.0F);
  auto p10 = vllm::gliner25_decide::Softmax(logits, 10.0F);
  // Higher temperature → more uniform
  double max_p1 = 0.0, max_p10 = 0.0;
  for (size_t i = 0; i < p1.size(); ++i) {
    max_p1 = std::max(max_p1, static_cast<double>(p1[i]));
    max_p10 = std::max(max_p10, static_cast<double>(p10[i]));
  }
  CHECK(max_p10 < max_p1);
}

TEST_CASE("gliner25_decide.Softmax.empty_input") {
  std::vector<float> empty;
  auto probs = vllm::gliner25_decide::Softmax(empty, 1.0F);
  CHECK(probs.empty());
}

// ── Sigmoid ────────────────────────────────────────────────────────────

TEST_CASE("gliner25_decide.Sigmoid.range_01") {
  std::vector<float> logits = {-5.0F, -1.0F, 0.0F, 1.0F, 5.0F};
  auto probs = vllm::gliner25_decide::Sigmoid(logits, 1.0F);
  for (float p : probs) {
    CHECK(p >= 0.0F);
    CHECK(p <= 1.0F);
  }
}

TEST_CASE("gliner25_decide.Sigmoid.zero_logit_is_half") {
  std::vector<float> logits = {0.0F};
  auto probs = vllm::gliner25_decide::Sigmoid(logits, 1.0F);
  REQUIRE(probs.size() == 1);
  CHECK(std::abs(probs[0] - 0.5F) < 1e-5F);
}

// ── Perturbation: missing ReLU ──────────────────────────────────────────

TEST_CASE("gliner25_decide.perturbation.relu_matters") {
  auto params = TestParams();
  auto hw = TestHead(params);
  int64_t N = 3;
  auto embs = Rand("embs", N * params.hidden_size, 2.0);

  // Normal forward (with ReLU)
  auto logits = vllm::gliner25_decide::ClassifierForward(hw, params, embs, N);

  // Without ReLU: manually compute with no activation
  const int64_t H = params.hidden_size;
  const int64_t H2 = H * 2;
  std::vector<float> no_relu(static_cast<size_t>(N));
  for (int64_t n = 0; n < N; ++n) {
    std::vector<float> emb(H);
    std::copy_n(&embs[static_cast<size_t>(n * H)], H, emb.data());
    // Linear(H, 2H)
    std::vector<float> hidden(H2, 0.0F);
    for (int64_t j = 0; j < H2; ++j) {
      double acc = 0.0;
      for (int64_t i = 0; i < H; ++i) {
        acc += emb[i] * hw.w0[static_cast<size_t>(j * H + i)];
      }
      hidden[static_cast<size_t>(j)] = static_cast<float>(acc + hw.b0[static_cast<size_t>(j)]);
    }
    // NO ReLU here — just Linear(2H, 1)
    double logit = 0.0;
    for (int64_t i = 0; i < H2; ++i) {
      logit += hidden[i] * hw.w2[static_cast<size_t>(i)];
    }
    no_relu[static_cast<size_t>(n)] = static_cast<float>(logit + hw.b2[0]);
  }

  CHECK_FALSE(ApproxEqual(logits, no_relu, 1e-4F));
}
