// kev PointerHead test: golden parity + perturbation gates.
//
// The generator (scripts/gen-kev-head-goldens.py) and this test rebuild the
// same weights from the same FNV-1a -> splitmix64 stream, so no weight byte
// is checked in.  The golden file carries only config constants, the input
// tensors, and the expected logits.
//
// The perturbation cases gate the three things this architecture gets wrong
// quietly: (1) the q/k projection order, (2) omitting bias terms,
// (3) the wrong scale factor.  Each is detected by a mutation that still
// runs but produces wrong logits.
#include <cmath>
#include <cstdint>
#include <doctest/doctest.h>
#include <string>
#include <vector>

#include "kev_head_goldens.inc"
#include "kev_lora_goldens.inc"
#include "vllm/model_executor/models/kev.h"

namespace {

// The generator's stream, byte-for-byte: values uniform in [-1, 1) derived
// from the tensor NAME alone.  A local copy by design -- if this drifts
// from the generator the goldens stop matching, which is the failure we
// want.
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

vllm::kev::HeadParams GoldenParams() {
  vllm::kev::HeadParams p;
  p.hidden_size = kev_head_goldens::kHiddenSize;
  p.head_dim = kev_head_goldens::kHeadDim;
  return p;
}

vllm::kev::HeadWeights GoldenWeights() {
  const int64_t d = kev_head_goldens::kHiddenSize;
  const int64_t dp = kev_head_goldens::kHeadDim;
  vllm::kev::HeadWeights w;
  w.q_weight = Rand("q.weight", d * dp, 0.3);
  w.q_bias = Rand("q.bias", dp, 0.3);
  w.k_weight = Rand("k.weight", d * dp, 0.3);
  w.k_bias = Rand("k.bias", dp, 0.3);
  return w;
}

bool ApproxEqual(const std::vector<float>& a, const std::vector<float>& b,
                 float tol) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > tol) return false;
  }
  return true;
}

}  // namespace

// ── Golden parity: logits match the PyTorch reference ──────────────────

TEST_CASE("kev.PointerHead.goldens.2_options") {
  const int64_t d = kev_head_goldens::kHiddenSize;
  auto params = GoldenParams();
  auto weights = GoldenWeights();
  std::vector<float> h_decide(
      kev_head_goldens::kHDecide1,
      kev_head_goldens::kHDecide1 + d);
  std::vector<float> h_opts(
      kev_head_goldens::kHOpts1,
      kev_head_goldens::kHOpts1 + 2 * d);
  auto logits = vllm::kev::PointerHeadForward(
      params, weights, h_decide, h_opts, 2);
  std::vector<float> expected(
      kev_head_goldens::kLogits1,
      kev_head_goldens::kLogits1 + 2);
  CHECK(ApproxEqual(logits, expected, 1e-4F));
}

TEST_CASE("kev.PointerHead.goldens.3_options") {
  const int64_t d = kev_head_goldens::kHiddenSize;
  auto params = GoldenParams();
  auto weights = GoldenWeights();
  std::vector<float> h_decide(
      kev_head_goldens::kHDecide2,
      kev_head_goldens::kHDecide2 + d);
  std::vector<float> h_opts(
      kev_head_goldens::kHOpts2,
      kev_head_goldens::kHOpts2 + 3 * d);
  auto logits = vllm::kev::PointerHeadForward(
      params, weights, h_decide, h_opts, 3);
  std::vector<float> expected(
      kev_head_goldens::kLogits2,
      kev_head_goldens::kLogits2 + 3);
  CHECK(ApproxEqual(logits, expected, 1e-4F));
}

TEST_CASE("kev.PointerHead.goldens.5_options") {
  const int64_t d = kev_head_goldens::kHiddenSize;
  auto params = GoldenParams();
  auto weights = GoldenWeights();
  std::vector<float> h_decide(
      kev_head_goldens::kHDecide3,
      kev_head_goldens::kHDecide3 + d);
  std::vector<float> h_opts(
      kev_head_goldens::kHOpts3,
      kev_head_goldens::kHOpts3 + 5 * d);
  auto logits = vllm::kev::PointerHeadForward(
      params, weights, h_decide, h_opts, 5);
  std::vector<float> expected(
      kev_head_goldens::kLogits3,
      kev_head_goldens::kLogits3 + 5);
  CHECK(ApproxEqual(logits, expected, 1e-4F));
}

// ── Softmax correctness ────────────────────────────────────────────────

TEST_CASE("kev.Softmax.sum_to_one") {
  std::vector<float> logits = {1.0F, 2.0F, 3.0F, 0.5F};
  auto probs = vllm::kev::Softmax(logits);
  REQUIRE(probs.size() == 4);
  float sum = 0.0F;
  for (float p : probs) sum += p;
  CHECK(std::abs(sum - 1.0F) < 1e-5F);
  // The largest logit should get the largest probability.
  CHECK(probs[2] > probs[1]);
  CHECK(probs[1] > probs[0]);
  CHECK(probs[0] > probs[3]);
}

TEST_CASE("kev.Softmax.empty") {
  auto probs = vllm::kev::Softmax({});
  CHECK(probs.empty());
}

TEST_CASE("kev.Softmax.single") {
  auto probs = vllm::kev::Softmax({42.0F});
  REQUIRE(probs.size() == 1);
  CHECK(std::abs(probs[0] - 1.0F) < 1e-5F);
}

// ── Perturbation gates: mutations that still run but produce wrong logits ─

TEST_CASE("kev.PointerHead.perturbation.qk_swap_detected") {
  // Swapping q and k weights changes the logits (different projections).
  const int64_t d = kev_head_goldens::kHiddenSize;
  auto params = GoldenParams();
  auto weights = GoldenWeights();
  std::vector<float> h_decide(
      kev_head_goldens::kHDecide1,
      kev_head_goldens::kHDecide1 + d);
  std::vector<float> h_opts(
      kev_head_goldens::kHOpts1,
      kev_head_goldens::kHOpts1 + 2 * d);

  auto correct = vllm::kev::PointerHeadForward(
      params, weights, h_decide, h_opts, 2);

  // Swap q and k weights.
  vllm::kev::HeadWeights swapped;
  swapped.q_weight = weights.k_weight;
  swapped.q_bias = weights.k_bias;
  swapped.k_weight = weights.q_weight;
  swapped.k_bias = weights.q_bias;
  auto mutated = vllm::kev::PointerHeadForward(
      params, swapped, h_decide, h_opts, 2);

  CHECK_FALSE(ApproxEqual(correct, mutated, 1e-4F));
}

TEST_CASE("kev.PointerHead.perturbation.bias_omission_detected") {
  // Omitting bias terms changes the logits.
  const int64_t d = kev_head_goldens::kHiddenSize;
  const int64_t dp = kev_head_goldens::kHeadDim;
  auto params = GoldenParams();
  auto weights = GoldenWeights();
  std::vector<float> h_decide(
      kev_head_goldens::kHDecide2,
      kev_head_goldens::kHDecide2 + d);
  std::vector<float> h_opts(
      kev_head_goldens::kHOpts2,
      kev_head_goldens::kHOpts2 + 3 * d);

  auto correct = vllm::kev::PointerHeadForward(
      params, weights, h_decide, h_opts, 3);

  // Zero out all biases.
  vllm::kev::HeadWeights no_bias;
  no_bias.q_weight = weights.q_weight;
  no_bias.q_bias = std::vector<float>(static_cast<size_t>(dp), 0.0F);
  no_bias.k_weight = weights.k_weight;
  no_bias.k_bias = std::vector<float>(static_cast<size_t>(dp), 0.0F);
  auto mutated = vllm::kev::PointerHeadForward(
      params, no_bias, h_decide, h_opts, 3);

  CHECK_FALSE(ApproxEqual(correct, mutated, 1e-4F));
}

TEST_CASE("kev.PointerHead.perturbation.wrong_scale_detected") {
  // Using the wrong scale (1/sqrt(d) instead of 1/sqrt(dp)) changes logits.
  const int64_t d = kev_head_goldens::kHiddenSize;
  auto params = GoldenParams();
  auto weights = GoldenWeights();
  std::vector<float> h_decide(
      kev_head_goldens::kHDecide3,
      kev_head_goldens::kHDecide3 + d);
  std::vector<float> h_opts(
      kev_head_goldens::kHOpts3,
      kev_head_goldens::kHOpts3 + 5 * d);

  auto correct = vllm::kev::PointerHeadForward(
      params, weights, h_decide, h_opts, 5);

  // Wrong scale: use hidden_size instead of head_dim.
  vllm::kev::HeadParams wrong = params;
  wrong.head_dim = params.hidden_size;  // scale = 1/sqrt(d) instead of 1/sqrt(dp)
  auto mutated = vllm::kev::PointerHeadForward(
      wrong, weights, h_decide, h_opts, 5);

  CHECK_FALSE(ApproxEqual(correct, mutated, 1e-4F));
}

// ── LoRA merge: golden parity + perturbation gates ────────────────────

TEST_CASE("kev.LoraMerge.goldens.case1") {
  const int64_t out = kev_lora_goldens::kOut1;
  const int64_t in = kev_lora_goldens::kIn1;
  const int64_t rank = kev_lora_goldens::kRank1;
  const float scaling = kev_lora_goldens::kScaling1;

  auto base = Rand("lora_base_1", out * in, 0.3);
  auto lora_a = Rand("lora_a_1", rank * in, 0.3);
  auto lora_b = Rand("lora_b_1", out * rank, 0.3);

  auto merged = vllm::kev::MergeLoraDelta(
      base, lora_a, lora_b, out, in, rank, scaling);

  std::vector<float> expected(
      kev_lora_goldens::kMerged1,
      kev_lora_goldens::kMerged1 + out * in);
  CHECK(ApproxEqual(merged, expected, 1e-5F));
}

TEST_CASE("kev.LoraMerge.goldens.case2") {
  const int64_t out = kev_lora_goldens::kOut2;
  const int64_t in = kev_lora_goldens::kIn2;
  const int64_t rank = kev_lora_goldens::kRank2;
  const float scaling = kev_lora_goldens::kScaling2;

  auto base = Rand("lora_base_2", out * in, 0.3);
  auto lora_a = Rand("lora_a_2", rank * in, 0.3);
  auto lora_b = Rand("lora_b_2", out * rank, 0.3);

  auto merged = vllm::kev::MergeLoraDelta(
      base, lora_a, lora_b, out, in, rank, scaling);

  std::vector<float> expected(
      kev_lora_goldens::kMerged2,
      kev_lora_goldens::kMerged2 + out * in);
  CHECK(ApproxEqual(merged, expected, 1e-5F));
}

TEST_CASE("kev.LoraMerge.perturbation.wrong_scaling_detected") {
  const int64_t out = kev_lora_goldens::kOut1;
  const int64_t in = kev_lora_goldens::kIn1;
  const int64_t rank = kev_lora_goldens::kRank1;
  const float scaling = kev_lora_goldens::kScaling1;

  auto base = Rand("lora_base_1", out * in, 0.3);
  auto lora_a = Rand("lora_a_1", rank * in, 0.3);
  auto lora_b = Rand("lora_b_1", out * rank, 0.3);

  auto correct = vllm::kev::MergeLoraDelta(
      base, lora_a, lora_b, out, in, rank, scaling);
  auto wrong = vllm::kev::MergeLoraDelta(
      base, lora_a, lora_b, out, in, rank, scaling * 2.0f);

  CHECK_FALSE(ApproxEqual(correct, wrong, 1e-5F));
}

TEST_CASE("kev.LoraMerge.perturbation.truncated_rank_detected") {
  const int64_t out = kev_lora_goldens::kOut2;
  const int64_t in = kev_lora_goldens::kIn2;
  const int64_t rank = kev_lora_goldens::kRank2;
  const float scaling = kev_lora_goldens::kScaling2;

  auto base = Rand("lora_base_2", out * in, 0.3);
  auto lora_a = Rand("lora_a_2", rank * in, 0.3);
  auto lora_b = Rand("lora_b_2", out * rank, 0.3);

  auto correct = vllm::kev::MergeLoraDelta(
      base, lora_a, lora_b, out, in, rank, scaling);
  auto truncated = vllm::kev::MergeLoraDelta(
      base, lora_a, lora_b, out, in, rank - 1, scaling);

  CHECK_FALSE(ApproxEqual(correct, truncated, 1e-5F));
}

TEST_CASE("kev.LoraMerge.perturbation.zero_delta_equals_base") {
  const int64_t out = kev_lora_goldens::kOut1;
  const int64_t in = kev_lora_goldens::kIn1;
  const int64_t rank = kev_lora_goldens::kRank1;
  const float scaling = kev_lora_goldens::kScaling1;

  auto base = Rand("lora_base_1", out * in, 0.3);
  std::vector<float> zero_a(static_cast<size_t>(rank * in), 0.0F);
  auto lora_b = Rand("lora_b_1", out * rank, 0.3);

  auto merged = vllm::kev::MergeLoraDelta(
      base, zero_a, lora_b, out, in, rank, scaling);

  CHECK(ApproxEqual(merged, base, 0.0F));
}

// ── bf16 conversion ───────────────────────────────────────────────────

TEST_CASE("kev.Bf16.roundtrip_representable_values") {
  const float values[] = {0.0f, 1.0f,  -1.0f, 2.0f,  -2.0f,
                          0.5f, -0.5f, 3.0f,  -3.0f, 256.0f};
  for (float v : values) {
    uint16_t bf16 = vllm::kev::F32ToBf16(v);
    float back = vllm::kev::Bf16ToF32(bf16);
    CHECK(back == v);
  }
}

TEST_CASE("kev.Bf16.roundtrip_within_precision") {
  auto vals = Rand("bf16_test", 64, 10.0);
  for (float v : vals) {
    uint16_t bf16 = vllm::kev::F32ToBf16(v);
    float back = vllm::kev::Bf16ToF32(bf16);
    CHECK(std::abs(back - v) <= std::abs(v) * 0.01f + 1e-3f);
  }
}
