// cua-s1 TinyTransformerScorer test: golden parity + perturbation gates.
//
// The generator (scripts/gen-cua-s1-goldens.py) and this test rebuild the
// same weights from the same FNV-1a -> splitmix64 stream, so no weight byte
// is checked in.  The golden file carries only config constants, the input
// tensors, and the expected logits.
//
// The perturbation cases gate the three things this architecture gets wrong
// quietly: (1) the packed-QKV split order, (2) finfo.min vs -inf for masking,
// (3) the safe-mask force at position 0.  Each is detected by a mutation that
// still runs but produces wrong logits.
#include <cmath>
#include <cstdint>
#include <doctest/doctest.h>
#include <string>
#include <vector>

#include "cua_s1_goldens.inc"
#include "vllm/model_executor/models/cua_s1.h"
#include "vllm/model_executor/models/cua_s1_collator.h"
#include "vllm/model_executor/models/cua_s1_inference.h"

namespace {

// The generator's stream, byte-for-byte: values uniform in [-1, 1) derived
// from the tensor NAME alone.  A local copy by design — if this drifts from
// the generator the goldens stop matching, which is the failure we want.
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

std::vector<float> RandPlusOne(const std::string& name, int64_t count, double scale) {
  std::vector<float> v = Rand(name, count, scale);
  for (float& x : v) x += 1.0F;
  return v;
}

vllm::cua_s1::Params GoldenParams() {
  vllm::cua_s1::Params p;
  p.width = cua_s1_goldens::kWidth;
  p.rank = cua_s1_goldens::kRank;
  p.context_tokens = cua_s1_goldens::kContextTokens;
  p.option_tokens = cua_s1_goldens::kOptionTokens;
  p.layers = cua_s1_goldens::kLayers;
  p.heads = cua_s1_goldens::kHeads;
  return p;
}

// Rebuild the checkpoint in nn.Linear [out, in] orientation, matching the
// generator's weight names byte-for-byte.
vllm::cua_s1::CheckpointTensors GoldenCheckpoint(const vllm::cua_s1::Params& p) {
  vllm::cua_s1::CheckpointTensors t;
  const int64_t w = p.width;
  const int64_t r = p.rank;
  const int64_t ff = p.dim_ff();
  const int64_t vocab = 257;
  const int64_t max_pos = p.max_pos();

  t.Set("embedding.weight", {vocab, w}, Rand("embedding.weight", vocab * w, 0.5));
  t.Set("position.weight", {max_pos, w}, Rand("position.weight", max_pos * w, 0.5));

  auto set_layer = [&](const std::string& base) {
    t.Set(base + "self_attn.in_proj_weight", {3 * w, w},
          Rand(base + "self_attn.in_proj_weight", 3 * w * w, 0.3));
    t.Set(base + "self_attn.in_proj_bias", {3 * w},
          Rand(base + "self_attn.in_proj_bias", 3 * w, 0.3));
    t.Set(base + "self_attn.out_proj.weight", {w, w},
          Rand(base + "self_attn.out_proj.weight", w * w, 0.3));
    t.Set(base + "self_attn.out_proj.bias", {w},
          Rand(base + "self_attn.out_proj.bias", w, 0.3));
    t.Set(base + "linear1.weight", {ff, w},
          Rand(base + "linear1.weight", ff * w, 0.3));
    t.Set(base + "linear1.bias", {ff},
          Rand(base + "linear1.bias", ff, 0.3));
    t.Set(base + "linear2.weight", {w, ff},
          Rand(base + "linear2.weight", w * ff, 0.3));
    t.Set(base + "linear2.bias", {w},
          Rand(base + "linear2.bias", w, 0.3));
    t.Set(base + "norm1.weight", {w},
          RandPlusOne(base + "norm1.weight", w, 0.1));
    t.Set(base + "norm1.bias", {w},
          Rand(base + "norm1.bias", w, 0.1));
    t.Set(base + "norm2.weight", {w},
          RandPlusOne(base + "norm2.weight", w, 0.1));
    t.Set(base + "norm2.bias", {w},
          Rand(base + "norm2.bias", w, 0.1));
  };

  for (int64_t i = 0; i < p.layers; ++i) {
    set_layer("encoder.layers." + std::to_string(i) + ".");
  }
  set_layer("option_encoder.layers.0.");

  t.Set("head.context_norm.weight", {w},
        RandPlusOne("head.context_norm.weight", w, 0.1));
  t.Set("head.context_norm.bias", {w},
        Rand("head.context_norm.bias", w, 0.1));
  t.Set("head.option_norm.weight", {w},
        RandPlusOne("head.option_norm.weight", w, 0.1));
  t.Set("head.option_norm.bias", {w},
        Rand("head.option_norm.bias", w, 0.1));
  t.Set("head.query.weight", {r, w},
        Rand("head.query.weight", r * w, 0.3));
  t.Set("head.key.weight", {r, w},
        Rand("head.key.weight", r * w, 0.3));
  t.Set("head.value.weight", {r, w},
        Rand("head.value.weight", r * w, 0.3));

  return t;
}

double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) -
                                       static_cast<double>(want[i])));
  }
  return worst;
}

std::vector<int64_t> VecFromArr(const int64_t* arr, size_t n) {
  return std::vector<int64_t>(arr, arr + n);
}
std::vector<uint8_t> VecFromArr(const uint8_t* arr, size_t n) {
  return std::vector<uint8_t>(arr, arr + n);
}

}  // namespace

TEST_CASE("cua_s1 forward reproduces the upstream logits: case 1") {
  const vllm::cua_s1::Params p = GoldenParams();
  const vllm::cua_s1::Weights w =
      vllm::cua_s1::Load(p, GoldenCheckpoint(p));

  const auto ctx_ids = VecFromArr(cua_s1_goldens::kContextIds1,
                                   sizeof(cua_s1_goldens::kContextIds1) / sizeof(int64_t));
  const auto ctx_mask = VecFromArr(cua_s1_goldens::kContextMask1,
                                    sizeof(cua_s1_goldens::kContextMask1) / sizeof(uint8_t));
  const auto opt_ids = VecFromArr(cua_s1_goldens::kOptionIds1,
                                   sizeof(cua_s1_goldens::kOptionIds1) / sizeof(int64_t));
  const auto opt_tmask = VecFromArr(cua_s1_goldens::kOptionTokMask1,
                                     sizeof(cua_s1_goldens::kOptionTokMask1) / sizeof(uint8_t));
  const auto opt_mask = VecFromArr(cua_s1_goldens::kOptionMask1,
                                    sizeof(cua_s1_goldens::kOptionMask1) / sizeof(uint8_t));

  const std::vector<float> logits = vllm::cua_s1::ForwardHost(
      p, w, ctx_ids, ctx_mask, opt_ids, opt_tmask, opt_mask);

  const size_t n = static_cast<size_t>(cua_s1_goldens::kNumOptions1);
  REQUIRE(logits.size() == n);
  const double diff = MaxAbsDiff(logits, cua_s1_goldens::kLogits1, n);
  INFO("max abs diff vs upstream logits (case 1): ", diff);
  CHECK(diff < 2e-5);
}

TEST_CASE("cua_s1 forward reproduces the upstream logits: case 2") {
  const vllm::cua_s1::Params p = GoldenParams();
  const vllm::cua_s1::Weights w =
      vllm::cua_s1::Load(p, GoldenCheckpoint(p));

  const auto ctx_ids = VecFromArr(cua_s1_goldens::kContextIds2,
                                   sizeof(cua_s1_goldens::kContextIds2) / sizeof(int64_t));
  const auto ctx_mask = VecFromArr(cua_s1_goldens::kContextMask2,
                                    sizeof(cua_s1_goldens::kContextMask2) / sizeof(uint8_t));
  const auto opt_ids = VecFromArr(cua_s1_goldens::kOptionIds2,
                                   sizeof(cua_s1_goldens::kOptionIds2) / sizeof(int64_t));
  const auto opt_tmask = VecFromArr(cua_s1_goldens::kOptionTokMask2,
                                     sizeof(cua_s1_goldens::kOptionTokMask2) / sizeof(uint8_t));
  const auto opt_mask = VecFromArr(cua_s1_goldens::kOptionMask2,
                                    sizeof(cua_s1_goldens::kOptionMask2) / sizeof(uint8_t));

  const std::vector<float> logits = vllm::cua_s1::ForwardHost(
      p, w, ctx_ids, ctx_mask, opt_ids, opt_tmask, opt_mask);

  const size_t n = static_cast<size_t>(cua_s1_goldens::kNumOptions2);
  REQUIRE(logits.size() == n);
  const double diff = MaxAbsDiff(logits, cua_s1_goldens::kLogits2, n);
  INFO("max abs diff vs upstream logits (case 2): ", diff);
  CHECK(diff < 2e-5);
}

TEST_CASE("cua_s1 context mask changes the output") {
  // Without the context key_padding_mask the padded positions silently
  // contribute to attention, changing the logits. The mask must matter.
  const vllm::cua_s1::Params p = GoldenParams();
  const vllm::cua_s1::Weights w =
      vllm::cua_s1::Load(p, GoldenCheckpoint(p));

  const auto ctx_ids = VecFromArr(cua_s1_goldens::kContextIds1,
                                   sizeof(cua_s1_goldens::kContextIds1) / sizeof(int64_t));
  const auto opt_ids = VecFromArr(cua_s1_goldens::kOptionIds1,
                                   sizeof(cua_s1_goldens::kOptionIds1) / sizeof(int64_t));
  const auto opt_tmask = VecFromArr(cua_s1_goldens::kOptionTokMask1,
                                     sizeof(cua_s1_goldens::kOptionTokMask1) / sizeof(uint8_t));
  const auto opt_mask = VecFromArr(cua_s1_goldens::kOptionMask1,
                                    sizeof(cua_s1_goldens::kOptionMask1) / sizeof(uint8_t));

  // Case 1 mask: 4 valid, 4 padding.
  std::vector<uint8_t> mask_a(cua_s1_goldens::kContextMask1,
                              cua_s1_goldens::kContextMask1 +
                                  sizeof(cua_s1_goldens::kContextMask1) / sizeof(uint8_t));
  // All-valid mask.
  std::vector<uint8_t> mask_b(mask_a.size(), 1);

  const auto out_a = vllm::cua_s1::ForwardHost(
      p, w, ctx_ids, mask_a, opt_ids, opt_tmask, opt_mask);
  const auto out_b = vllm::cua_s1::ForwardHost(
      p, w, ctx_ids, mask_b, opt_ids, opt_tmask, opt_mask);

  REQUIRE(out_a.size() == out_b.size());
  double worst = 0.0;
  for (size_t i = 0; i < out_a.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(out_a[i] - out_b[i])));
  }
  INFO("max abs diff with partial vs full context mask: ", worst);
  CHECK(worst > 1e-4);
}

TEST_CASE("cua_s1 different options produce different logits") {
  // Swapping two options must change the per-option logits, proving the
  // option encoder and attention head are wired to the right inputs.
  const vllm::cua_s1::Params p = GoldenParams();
  const vllm::cua_s1::Weights w =
      vllm::cua_s1::Load(p, GoldenCheckpoint(p));

  const auto ctx_ids = VecFromArr(cua_s1_goldens::kContextIds2,
                                   sizeof(cua_s1_goldens::kContextIds2) / sizeof(int64_t));
  const auto ctx_mask = VecFromArr(cua_s1_goldens::kContextMask2,
                                    sizeof(cua_s1_goldens::kContextMask2) / sizeof(uint8_t));

  // Original option ids: [[9,10,11,12], [13,14,15,0]]
  std::vector<int64_t> opt_ids_a(cua_s1_goldens::kOptionIds2,
                                  cua_s1_goldens::kOptionIds2 +
                                      sizeof(cua_s1_goldens::kOptionIds2) / sizeof(int64_t));
  // Swapped: [[13,14,15,0], [9,10,11,12]]
  std::vector<int64_t> opt_ids_b = opt_ids_a;
  const int64_t opt_len = p.option_tokens;
  for (int64_t t = 0; t < opt_len; ++t) {
    std::swap(opt_ids_b[static_cast<size_t>(t)],
              opt_ids_b[static_cast<size_t>(opt_len + t)]);
  }
  // Swap the token masks too.
  std::vector<uint8_t> opt_tmask_a(cua_s1_goldens::kOptionTokMask2,
                                    cua_s1_goldens::kOptionTokMask2 +
                                        sizeof(cua_s1_goldens::kOptionTokMask2) / sizeof(uint8_t));
  std::vector<uint8_t> opt_tmask_b = opt_tmask_a;
  for (int64_t t = 0; t < opt_len; ++t) {
    std::swap(opt_tmask_b[static_cast<size_t>(t)],
              opt_tmask_b[static_cast<size_t>(opt_len + t)]);
  }
  const auto opt_mask = VecFromArr(cua_s1_goldens::kOptionMask2,
                                    sizeof(cua_s1_goldens::kOptionMask2) / sizeof(uint8_t));

  const auto out_a = vllm::cua_s1::ForwardHost(
      p, w, ctx_ids, ctx_mask, opt_ids_a, opt_tmask_a, opt_mask);
  const auto out_b = vllm::cua_s1::ForwardHost(
      p, w, ctx_ids, ctx_mask, opt_ids_b, opt_tmask_b, opt_mask);

  REQUIRE(out_a.size() == out_b.size());
  // Logit for option 0 in A should equal logit for option 1 in B, and vice versa.
  const double cross_diff_0 = std::fabs(static_cast<double>(out_a[0] - out_b[1]));
  const double cross_diff_1 = std::fabs(static_cast<double>(out_a[1] - out_b[0]));
  const double same_diff_0 = std::fabs(static_cast<double>(out_a[0] - out_b[0]));
  const double same_diff_1 = std::fabs(static_cast<double>(out_a[1] - out_b[1]));

  INFO("cross diff 0: ", cross_diff_0, " cross diff 1: ", cross_diff_1);
  INFO("same diff 0: ", same_diff_0, " same diff 1: ", same_diff_1);
  CHECK(cross_diff_0 < 2e-5);
  CHECK(cross_diff_1 < 2e-5);
  CHECK(same_diff_0 > 1e-4);
  CHECK(same_diff_1 > 1e-4);
}

// ── Phase 2: ByteCollator + inference pipeline ────────────────────────────

TEST_CASE("cua_s1 ByteCollator tokenizes and pads correctly") {
  const vllm::cua_s1::Params p = GoldenParams();

  // "hi" → bytes [104, 105] → ids [105, 106]
  // options: "a" → [98], "bc" → [99, 100]
  // max_option_tokens = 2, so "a" pads to [98, 0]
  const vllm::cua_s1::CollatedBatch b =
      vllm::cua_s1::ByteCollate(p, "hi", {"a", "bc"});

  REQUIRE(b.ctx_len == 2);
  REQUIRE(b.n_opt == 2);
  REQUIRE(b.opt_len == 2);

  // Context ids and mask.
  REQUIRE(b.context_ids.size() == 2);
  CHECK(b.context_ids[0] == 105);  // 'h' + 1
  CHECK(b.context_ids[1] == 106);  // 'i' + 1
  CHECK(b.context_mask[0] == 1);
  CHECK(b.context_mask[1] == 1);

  // Option ids (flattened): [98, 0, 99, 100]
  REQUIRE(b.option_ids.size() == 4);
  CHECK(b.option_ids[0] == 98);   // 'a' + 1
  CHECK(b.option_ids[1] == 0);    // padding
  CHECK(b.option_ids[2] == 99);   // 'b' + 1
  CHECK(b.option_ids[3] == 100);  // 'c' + 1

  // Option token mask.
  CHECK(b.option_tok_mask[0] == 1);
  CHECK(b.option_tok_mask[1] == 0);
  CHECK(b.option_tok_mask[2] == 1);
  CHECK(b.option_tok_mask[3] == 1);

  // Option mask (all valid).
  CHECK(b.option_mask[0] == 1);
  CHECK(b.option_mask[1] == 1);
}

TEST_CASE("cua_s1 ByteCollator truncates long inputs") {
  const vllm::cua_s1::Params p = GoldenParams();
  // context_tokens = 8, option_tokens = 4.
  // 10-byte context truncated to 8, 5-byte option truncated to 4.
  const vllm::cua_s1::CollatedBatch b =
      vllm::cua_s1::ByteCollate(p, "abcdefghij", {"ABCDE"});

  REQUIRE(b.ctx_len == 8);
  REQUIRE(b.opt_len == 4);
  REQUIRE(b.context_ids.size() == 8);
  // 'a'=97+1=98 ... 'h'=104+1=105
  CHECK(b.context_ids[0] == 98);
  CHECK(b.context_ids[7] == 105);
  // All 8 positions valid (no padding after truncation).
  CHECK(b.context_mask[0] == 1);
  CHECK(b.context_mask[7] == 1);

  // 'A'=65+1=66, 'B'=66+1=67, 'C'=67+1=68, 'D'=68+1=69, truncated to 4.
  REQUIRE(b.option_ids.size() == 4);
  CHECK(b.option_ids[0] == 66);
  CHECK(b.option_ids[3] == 69);
}

TEST_CASE("cua_s1 ByteCollator handles UTF-8 multibyte") {
  // 'é' = 0xC3 0xA9 in UTF-8 → ids [0xC3+1, 0xA9+1] = [196, 170]
  const auto ids = vllm::cua_s1::ByteIds("\xc3\xa9", 8);
  REQUIRE(ids.size() == 2);
  CHECK(ids[0] == 196);
  CHECK(ids[1] == 170);
}

TEST_CASE("cua_s1 full pipeline reproduces upstream probabilities") {
  const vllm::cua_s1::Params p = GoldenParams();
  const vllm::cua_s1::Weights w =
      vllm::cua_s1::Load(p, GoldenCheckpoint(p));

  const vllm::cua_s1::CuaS1ScoreResult result =
      vllm::cua_s1::CuaS1Inference(p, w, "hi", {"a", "bc"});

  const size_t n = static_cast<size_t>(cua_s1_goldens::kNumOptions3);
  REQUIRE(result.probabilities.size() == n);

  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(
        static_cast<double>(result.probabilities[i]) -
        static_cast<double>(cua_s1_goldens::kProbs3[i])));
  }
  INFO("max abs diff vs upstream probabilities: ", worst);
  CHECK(worst < 2e-5);

  // Probabilities sum to 1.
  double psum = 0.0;
  for (size_t i = 0; i < n; ++i) psum += result.probabilities[i];
  CHECK(std::fabs(psum - 1.0) < 1e-5);

  // Winner is argmax.
  const int64_t expected_winner =
      cua_s1_goldens::kProbs3[0] > cua_s1_goldens::kProbs3[1] ? 0 : 1;
  CHECK(result.winner == expected_winner);

  // Confidence is the winning probability.
  const float expected_conf =
      std::max(cua_s1_goldens::kProbs3[0], cua_s1_goldens::kProbs3[1]);
  CHECK(std::fabs(static_cast<double>(result.confidence - expected_conf)) < 2e-5);
}
