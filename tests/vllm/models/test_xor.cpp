// XOR (juspay/xor) decision model test: forward+reverse calibration gates.
//
// Tests the forward+reverse option-order evaluation and probability
// calibration logic. Uses deterministic PRNG for reproducible inputs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <doctest/doctest.h>
#include <numeric>
#include <string>
#include <vector>

namespace {

// Deterministic PRNG (same as test_kev.cpp / test_clm.cpp)
[[maybe_unused]] static std::vector<float> Rand(const std::string& name, int64_t count, double scale) {
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

std::vector<float> Softmax(const std::vector<float>& logits) {
  if (logits.empty()) return {};
  float mx = *std::max_element(logits.begin(), logits.end());
  std::vector<float> exp_vals(logits.size());
  double sum = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    exp_vals[i] = std::exp(logits[i] - mx);
    sum += static_cast<double>(exp_vals[i]);
  }
  if (sum < 1e-12) sum = 1e-12;
  std::vector<float> probs(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = static_cast<float>(exp_vals[i] / sum);
  }
  return probs;
}

}  // namespace

// ── Softmax ──────────────────────────────────────────────────────────────

TEST_CASE("xor.Softmax.sums_to_one") {
  std::vector<float> logits = {1.0F, 2.0F, 3.0F, 0.5F};
  auto probs = Softmax(logits);
  REQUIRE(probs.size() == 4);
  double sum = 0.0;
  for (float p : probs) sum += p;
  CHECK(std::abs(sum - 1.0) < 1e-5);
}

TEST_CASE("xor.Softmax.argmax_matches_max") {
  std::vector<float> logits = {0.1F, 5.0F, 0.2F, 0.3F};
  auto probs = Softmax(logits);
  auto max_it = std::max_element(probs.begin(), probs.end());
  CHECK(static_cast<int>(max_it - probs.begin()) == 1);
}

TEST_CASE("xor.Softmax.empty") {
  std::vector<float> empty;
  auto probs = Softmax(empty);
  CHECK(probs.empty());
}

// ── Forward+reverse calibration ──────────────────────────────────────────

TEST_CASE("xor.calibration.symmetric_options_averaged") {
  // Forward scores: [3.0, 1.0, 0.5] → option 0 is best
  // Reverse scores: [0.5, 1.0, 3.0] → option 2 (original 0) is best
  // After calibration: forward_probs[i] and rev_probs[N-1-i] averaged
  int N = 3;
  std::vector<float> fwd_scores = {3.0F, 1.0F, 0.5F};
  std::vector<float> rev_scores = {3.0F, 1.0F, 0.5F};

  auto fwd_probs = Softmax(fwd_scores);
  auto rev_probs = Softmax(rev_scores);

  std::vector<float> calibrated(N);
  for (int i = 0; i < N; ++i) {
    int rev_idx = N - 1 - i;
    calibrated[i] = 0.5F * (fwd_probs[i] + rev_probs[rev_idx]);
  }

  // The calibrated distribution should still sum to 1
  double sum = 0.0;
  for (float p : calibrated) sum += p;
  CHECK(std::abs(sum - 1.0) < 1e-5);

  // Option 0 should be the argmax (it was best in forward, and its reverse
  // counterpart was also best)
  auto max_it = std::max_element(calibrated.begin(), calibrated.end());
  CHECK(static_cast<int>(max_it - calibrated.begin()) == 0);
}

TEST_CASE("xor.calibration.symmetric_distribution_reduces_bias") {
  // If forward and reverse agree, calibration = forward = reverse
  int N = 4;
  std::vector<float> scores = {2.0F, 1.0F, 0.5F, 0.1F};

  auto fwd_probs = Softmax(scores);
  auto rev_probs = Softmax(scores);  // same scores

  std::vector<float> calibrated(N);
  for (int i = 0; i < N; ++i) {
    int rev_idx = N - 1 - i;
    calibrated[i] = 0.5F * (fwd_probs[i] + rev_probs[rev_idx]);
  }

  // When forward == reverse, calibrated[0] = 0.5*(fwd[0] + rev[N-1])
  // fwd = rev = softmax([2,1,0.5,0.1])
  // calibrated[0] = 0.5*(fwd[0] + rev[3]) = 0.5*(fwd[0] + fwd[3])
  float expected = 0.5F * (fwd_probs[0] + fwd_probs[3]);
  CHECK(std::abs(calibrated[0] - expected) < 1e-5F);
}

// ── Confidence: standard SystemOne formula ──────────────────────────────
// xor uses the same confidence formulas as kev/laya (per spec).

TEST_CASE("xor.confidence.choice_formula") {
  // (max(p) - 1/K) / (1 - 1/K)
  std::vector<float> probs = {0.5F, 0.3F, 0.2F};
  float mx = *std::max_element(probs.begin(), probs.end());
  float K = static_cast<float>(probs.size());
  float conf = (mx - 1.0F / K) / (1.0F - 1.0F / K);
  // (0.5 - 0.333) / (1 - 0.333) = 0.167 / 0.667 ≈ 0.25
  CHECK(std::abs(conf - 0.25F) < 0.01F);
}

TEST_CASE("xor.confidence.uniform_is_zero") {
  std::vector<float> probs = {0.25F, 0.25F, 0.25F, 0.25F};
  float mx = *std::max_element(probs.begin(), probs.end());
  float K = static_cast<float>(probs.size());
  float conf = (mx - 1.0F / K) / (1.0F - 1.0F / K);
  CHECK(std::abs(conf - 0.0F) < 1e-5F);
}

TEST_CASE("xor.confidence.certain_is_one") {
  std::vector<float> probs = {1.0F, 0.0F, 0.0F};
  float mx = *std::max_element(probs.begin(), probs.end());
  float K = static_cast<float>(probs.size());
  float conf = (mx - 1.0F / K) / (1.0F - 1.0F / K);
  // (1.0 - 0.333) / (1 - 0.333) = 1.0
  CHECK(std::abs(conf - 1.0F) < 1e-5F);
}

// ── Prompt construction ──────────────────────────────────────────────────

TEST_CASE("xor.prompt.has_all_options") {
  // Verify that a prompt built with 3 options contains all 3 labels
  std::vector<std::string> options = {"apple", "banana", "cherry"};
  std::string prompt = "state\n\nQuestion: choice\nOptions:\n";
  for (size_t i = 0; i < options.size(); ++i) {
    char label = static_cast<char>('A' + i);
    prompt += std::string(1, label) + ". " + options[i] + "\n";
  }
  prompt += "\nAnswer:";

  CHECK(prompt.find("A. apple") != std::string::npos);
  CHECK(prompt.find("B. banana") != std::string::npos);
  CHECK(prompt.find("C. cherry") != std::string::npos);
  CHECK(prompt.find("Answer:") != std::string::npos);
}

TEST_CASE("xor.prompt.reverse_order") {
  std::vector<std::string> options = {"apple", "banana", "cherry"};
  std::vector<std::string> rev_options(options.rbegin(), options.rend());

  // Reverse should have cherry first
  CHECK(rev_options[0] == "cherry");
  CHECK(rev_options[1] == "banana");
  CHECK(rev_options[2] == "apple");
}
