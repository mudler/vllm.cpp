// Ported from:
//   tests/kernels/core/test_pos_encoding.py:66-193
//   tests/kernels/core/test_mrope.py:47-235
//   tests/kernels/core/test_apply_rotary_emb.py:43-203
//   tests/models/language/pooling/test_nomic_max_model_len.py:93-111
// and pinned class behavior in rotary_embedding/{__init__,base,common,
// yarn_scaling_rope,mrope,llama3_rope,phi3_long_rope_scaled_rope,
// dynamic_ntk_scaling_rope,dynamic_ntk_alpha_rope}.py
// @ e24d1b24fe96.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/layers/rotary_embedding/common.h"
#include "vllm/model_executor/layers/rotary_embedding/dynamic_ntk_alpha_rope.h"
#include "vllm/model_executor/layers/rotary_embedding/dynamic_ntk_scaling_rope.h"
#include "vllm/model_executor/layers/rotary_embedding/llama3_rope.h"
#include "vllm/model_executor/layers/rotary_embedding/mrope.h"
#include "vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.h"
#include "vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.h"
#include "vt/dtype.h"

namespace {

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }

float CacheValue(const vllm::RotaryEmbeddingBase& rope, int64_t row,
                 int64_t column) {
  const vt::Tensor cache = rope.cos_sin_cache();
  const int64_t offset = row * cache.shape[1] + column;
  if (cache.dtype == vt::DType::kF32) return cache.Ptr<float>()[offset];
  return vt::BF16ToF32(cache.Ptr<uint16_t>()[offset]);
}

vllm::RopeParameters YarnParameters() {
  vllm::RopeParameters params;
  params.rope_type = "yarn";
  // A small base leaves a fractional correction boundary inside this tiny
  // rotary dimension, so the truncate on/off test below is discriminating.
  params.rope_theta = 10.0;
  params.rope_dim = 8;
  params.factor = 4.0;
  params.original_max_position_embeddings = 32;
  return params;
}

vllm::RopeParameters Llama3Parameters() {
  vllm::RopeParameters params;
  params.rope_type = "llama3";
  params.rope_theta = 10000.0;
  params.rope_dim = 16;
  params.factor = 4.0;
  params.low_freq_factor = 1.0;
  params.high_freq_factor = 4.0;
  params.original_max_position_embeddings = 32;
  return params;
}

vllm::RopeParameters LongRoPEParameters() {
  vllm::RopeParameters params;
  params.rope_type = "longrope";
  params.rope_theta = 10000.0;
  params.rope_dim = 8;
  params.original_max_position_embeddings = 32;
  params.short_factor = {1.0, 1.1, 1.2, 1.3};
  params.long_factor = {2.0, 2.2, 2.4, 2.6};
  return params;
}

vllm::RopeParameters DynamicFactorParameters() {
  vllm::RopeParameters params;
  params.rope_type = "dynamic";
  params.rope_theta = 10000.0;
  params.rope_dim = 8;
  params.factor = 4.0;
  params.max_trained_positions = 32;
  return params;
}

}  // namespace

TEST_CASE("yarn common helpers mirror correction ramp and magnitude scaling") {
  CHECK(vllm::yarn_get_mscale(0.5) == doctest::Approx(1.0));
  CHECK(vllm::yarn_get_mscale(1.0) == doctest::Approx(1.0));
  CHECK(vllm::yarn_get_mscale(4.0) ==
        doctest::Approx(1.0 + 0.1 * std::log(4.0)));

  const auto equal_ramp = vllm::yarn_linear_ramp_mask(2.0, 2.0, 5);
  REQUIRE(equal_ramp.size() == 5);
  CHECK(equal_ramp[0] == 0.0F);
  CHECK(equal_ramp[2] == 0.0F);
  CHECK(equal_ramp[3] == 1.0F);

  const auto [low, high] =
      vllm::yarn_find_correction_range(32, 1, 8, 10000.0, 32, true);
  CHECK(low == doctest::Approx(0.0));
  CHECK(high <= 7.0);
  CHECK(high >= low);
}

TEST_CASE("get_rope builds and memoizes the typed YaRN cache") {
  vllm::RopeParameters params = YarnParameters();
  auto first = vllm::get_rope(16, 128, true, params, vt::DType::kF32);
  auto second = vllm::get_rope(16, 128, true, params, vt::DType::kF32);
  REQUIRE(first == second);
  CHECK(first->type_name() == "YaRNScalingRotaryEmbedding");
  CHECK(first->head_size() == 16);
  CHECK(first->rotary_dim() == 8);
  CHECK(first->max_position_embeddings() == 32);
  CHECK(first->cache_rows() == 128);
  CHECK(first->cos_sin_cache().shape[1] == 8);

  const float mscale =
      static_cast<float>(vllm::yarn_get_mscale(4.0));
  for (int64_t pair = 0; pair < 4; ++pair) {
    CHECK(CacheValue(*first, 0, pair) == doctest::Approx(mscale));
    CHECK(CacheValue(*first, 0, 4 + pair) == doctest::Approx(0.0));
  }

  params.truncate = false;
  auto untruncated =
      vllm::get_rope(16, 128, true, params, vt::DType::kF32);
  CHECK(untruncated != first);
  bool any_difference = false;
  for (int64_t column = 0; column < 8; ++column) {
    any_difference = any_difference ||
                     CacheValue(*first, 31, column) !=
                         CacheValue(*untruncated, 31, column);
  }
  CHECK(any_difference);
}

TEST_CASE("YaRN apply_yarn_scaling and dtype cache match pinned dispatch") {
  vllm::RopeParameters params = YarnParameters();
  params.attn_factor = 0.75;
  params.apply_yarn_scaling = false;
  auto no_yarn_scale =
      vllm::get_rope(16, 128, true, params, vt::DType::kF32);
  CHECK(CacheValue(*no_yarn_scale, 0, 0) == doctest::Approx(0.75));

  params.apply_yarn_scaling = true;
  auto bf16 = vllm::get_rope(16, 128, true, params, vt::DType::kBF16);
  CHECK(bf16->dtype() == vt::DType::kBF16);
  const float expected = vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(
      vllm::yarn_get_mscale(4.0) * params.attn_factor)));
  CHECK(CacheValue(*bf16, 0, 0) == expected);
  CHECK(bf16->cache_bytes() ==
        static_cast<size_t>(bf16->cache_rows() * bf16->rotary_dim() * 2));
}

TEST_CASE("get_rope YaRN mrope branch enlarges cache and drops apply flag") {
  vllm::RopeParameters params = YarnParameters();
  params.mrope_section = {1, 1, 2};
  params.mrope_interleaved = true;
  params.attn_factor = 0.5;
  params.apply_yarn_scaling = false;  // get_rope drops this for MRoPE.
  auto rope = vllm::get_rope(16, 128, true, params, vt::DType::kF32);
  REQUIRE(rope->type_name() == "MRotaryEmbedding");
  auto mrope = std::dynamic_pointer_cast<vllm::MRotaryEmbedding>(rope);
  REQUIRE(mrope != nullptr);
  CHECK(mrope->mrope_section() == std::vector<int64_t>{1, 1, 2});
  CHECK(mrope->mrope_interleaved());
  // Constructor receives original max 32, reserves 4x, then YaRN expands 4x.
  CHECK(mrope->max_position_embeddings() == 128);
  CHECK(mrope->cache_rows() == 512);
  CHECK(mrope->mscale() == doctest::Approx(
                                vllm::yarn_get_mscale(4.0) * 0.5));
}

TEST_CASE("default MRoPE text positions reduce to ordinary cached rotation") {
  vllm::RopeParameters params;
  params.rope_dim = 8;
  params.mrope_section = {1, 1, 2};
  auto rope = vllm::get_rope(10, 16, true, params, vt::DType::kF32);
  REQUIRE(rope->type_name() == "MRotaryEmbedding");
  CHECK(rope->cache_rows() == 64);  // pinned MRoPE reserves 4x.

  std::vector<int64_t> positions = {0, 1, 7};
  std::vector<float> query(3 * 2 * 10);
  std::vector<float> key(3 * 10);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = static_cast<float>(i + 1) / 17.0F;
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = -static_cast<float>(i + 1) / 19.0F;
  }
  vt::Tensor tq = vt::Tensor::Contiguous(query.data(), vt::DType::kF32,
                                         Cpu(), {3, 2, 10});
  vt::Tensor tk = vt::Tensor::Contiguous(key.data(), vt::DType::kF32, Cpu(),
                                         {3, 1, 10});
  vt::Tensor tp = vt::Tensor::Contiguous(positions.data(), vt::DType::kI64,
                                         Cpu(), {3});
  vt::Queue queue{Cpu(), nullptr};
  rope->forward_native(queue, tp, tq, &tk);
  for (float value : query) CHECK(std::isfinite(value));
  for (float value : key) CHECK(std::isfinite(value));
}

TEST_CASE("get_rope mirrors partial dimension and validation errors") {
  vllm::RopeParameters partial;
  partial.partial_rotary_factor = 0.5;
  auto rope = vllm::get_rope(16, 32, true, partial, vt::DType::kF32);
  CHECK(rope->rotary_dim() == 8);

  partial.partial_rotary_factor = 0.0;
  CHECK_THROWS_WITH_AS(vllm::get_rope(16, 32, true, partial),
                       doctest::Contains("partial_rotary_factor"),
                       std::invalid_argument);

  vllm::RopeParameters bad_section;
  bad_section.rope_dim = 8;
  bad_section.mrope_section = {1, 1, 1};
  CHECK_THROWS_WITH_AS(vllm::get_rope(16, 32, true, bad_section),
                       doctest::Contains("must equal"),
                       std::invalid_argument);

  vllm::RopeParameters missing;
  missing.rope_type = "yarn";
  CHECK_THROWS_WITH_AS(vllm::get_rope(16, 32, true, missing),
                       doctest::Contains("requires factor"),
                       std::invalid_argument);
}

TEST_CASE("Llama 3 cache covers unchanged smoothed and scaled bands") {
  const vllm::RopeParameters params = Llama3Parameters();
  auto rope = vllm::get_rope(24, 128, true, params, vt::DType::kF32);
  auto again = vllm::get_rope(24, 128, true, params, vt::DType::kF32);
  REQUIRE(rope == again);
  CHECK(rope->type_name() == "Llama3RotaryEmbedding");
  CHECK(rope->cache_rows() == 128);
  auto llama3 = std::dynamic_pointer_cast<vllm::Llama3RotaryEmbedding>(rope);
  REQUIRE(llama3 != nullptr);
  CHECK(llama3->scaling_factor() == doctest::Approx(4.0));
  CHECK(llama3->low_freq_factor() == doctest::Approx(1.0));
  CHECK(llama3->high_freq_factor() == doctest::Approx(4.0));
  CHECK(llama3->orig_max_position() == 32);

  int unchanged = 0;
  int smoothed = 0;
  int scaled = 0;
  constexpr int64_t kHalf = 8;
  for (int64_t pair = 0; pair < kHalf; ++pair) {
    const float exponent = static_cast<float>(2 * pair) / 16.0F;
    const float base_inv = 1.0F / std::pow(10000.0F, exponent);
    const float wave_len =
        static_cast<float>(2.0 * std::acos(-1.0)) / base_inv;
    const float got = std::atan2(CacheValue(*rope, 1, kHalf + pair),
                                 CacheValue(*rope, 1, pair));
    if (wave_len < 8.0F) {
      ++unchanged;
      CHECK(got == doctest::Approx(base_inv).epsilon(1e-5));
    } else if (wave_len > 32.0F) {
      ++scaled;
      CHECK(got == doctest::Approx(base_inv / 4.0F).epsilon(1e-5));
    } else {
      ++smoothed;
      CHECK(got > base_inv / 4.0F);
      CHECK(got < base_inv);
    }
  }
  CHECK(unchanged > 0);
  CHECK(smoothed > 0);
  CHECK(scaled > 0);
}

TEST_CASE("Llama 3 equal frequency factors avoid a singular smoothing path") {
  vllm::RopeParameters params = Llama3Parameters();
  params.low_freq_factor = 2.0;
  params.high_freq_factor = 2.0;
  auto rope = vllm::get_rope(24, 128, true, params, vt::DType::kF32);
  constexpr int64_t kHalf = 8;
  for (int64_t pair = 0; pair < kHalf; ++pair) {
    const float exponent = static_cast<float>(2 * pair) / 16.0F;
    const float base_inv = 1.0F / std::pow(10000.0F, exponent);
    const float wave_len =
        static_cast<float>(2.0 * std::acos(-1.0)) / base_inv;
    const float got = std::atan2(CacheValue(*rope, 1, kHalf + pair),
                                 CacheValue(*rope, 1, pair));
    REQUIRE(std::isfinite(got));
    const float expected = wave_len < 16.0F ? base_inv : base_inv / 4.0F;
    CHECK(got == doctest::Approx(expected).epsilon(1e-5));
  }

  vllm::RopeParameters missing = params;
  missing.low_freq_factor.reset();
  CHECK_THROWS_WITH_AS(vllm::get_rope(24, 128, true, missing),
                       doctest::Contains("requires factor"),
                       std::invalid_argument);
}

TEST_CASE("Phi-3 LongRoPE builds both caches and selects one globally") {
  const vllm::RopeParameters params = LongRoPEParameters();
  auto short_rope =
      vllm::get_rope(16, 128, true, params, vt::DType::kF32);
  auto short_again = vllm::get_rope(16, 128, true, params,
                                    vt::DType::kF32, 32);
  auto long_rope = vllm::get_rope(16, 128, true, params,
                                  vt::DType::kF32, 33);
  REQUIRE(short_rope == short_again);
  REQUIRE(short_rope != long_rope);
  REQUIRE(short_rope->type_name() ==
          "Phi3LongRoPEScaledRotaryEmbedding");
  CHECK(short_rope->cache_rows() == 160);
  CHECK(long_rope->cache_rows() == 160);

  auto short_phi = std::dynamic_pointer_cast<
      vllm::Phi3LongRoPEScaledRotaryEmbedding>(short_rope);
  auto long_phi = std::dynamic_pointer_cast<
      vllm::Phi3LongRoPEScaledRotaryEmbedding>(long_rope);
  REQUIRE(short_phi != nullptr);
  REQUIRE(long_phi != nullptr);
  CHECK_FALSE(short_phi->use_long_rope());
  CHECK(long_phi->use_long_rope());
  CHECK(short_phi->max_model_len() == 32);
  CHECK(long_phi->max_model_len() == 33);

  const double default_mscale =
      std::sqrt(1.0 + std::log(4.0) / std::log(32.0));
  CHECK(short_phi->short_mscale() == doctest::Approx(default_mscale));
  CHECK(short_phi->long_mscale() == doctest::Approx(default_mscale));
  CHECK(CacheValue(*short_rope, 0, 0) ==
        doctest::Approx(default_mscale));
  CHECK(CacheValue(*short_rope, 32, 0) ==
        doctest::Approx(default_mscale));

  for (int64_t pair = 0; pair < 4; ++pair) {
    const float exponent = static_cast<float>(2 * pair) / 8.0F;
    const float base_inv = 1.0F / std::pow(10000.0F, exponent);
    const float short_angle =
        std::atan2(CacheValue(*short_rope, 1, 4 + pair),
                   CacheValue(*short_rope, 1, pair));
    const float long_angle =
        std::atan2(CacheValue(*short_rope, 33, 4 + pair),
                   CacheValue(*short_rope, 33, pair));
    CHECK(short_angle == doctest::Approx(
                             base_inv / params.short_factor[pair])
                             .epsilon(1e-5));
    CHECK(long_angle == doctest::Approx(base_inv / params.long_factor[pair])
                            .epsilon(1e-5));
  }
}

TEST_CASE("Phi-3 LongRoPE mirrors mscale and validation boundaries") {
  vllm::RopeParameters params = LongRoPEParameters();
  params.short_mscale = 0.75;
  params.long_mscale = 1.25;
  auto rope = vllm::get_rope(16, 128, true, params, vt::DType::kF32, 128);
  auto phi = std::dynamic_pointer_cast<
      vllm::Phi3LongRoPEScaledRotaryEmbedding>(rope);
  REQUIRE(phi != nullptr);
  CHECK(phi->use_long_rope());
  CHECK(phi->short_mscale() == doctest::Approx(0.75));
  CHECK(phi->long_mscale() == doctest::Approx(1.25));
  CHECK(CacheValue(*rope, 0, 0) == doctest::Approx(0.75));
  CHECK(CacheValue(*rope, 32, 0) == doctest::Approx(1.25));

  std::vector<int64_t> positions = {0};
  std::vector<float> query(16, 1.0F);
  vt::Tensor tp = vt::Tensor::Contiguous(positions.data(), vt::DType::kI64,
                                         Cpu(), {1});
  vt::Tensor tq = vt::Tensor::Contiguous(query.data(), vt::DType::kF32,
                                         Cpu(), {1, 1, 16});
  vt::Queue queue{Cpu(), nullptr};
  CHECK_THROWS_WITH_AS(rope->forward_native(queue, tp, tq, nullptr),
                       doctest::Contains("requires a key tensor"),
                       std::invalid_argument);

  CHECK_THROWS_WITH_AS(
      vllm::get_rope(16, 128, false, params, vt::DType::kF32, 128),
      doctest::Contains("only supports neox_style"), std::invalid_argument);

  params.short_factor.pop_back();
  CHECK_THROWS_WITH_AS(
      vllm::get_rope(16, 128, true, params, vt::DType::kF32, 128),
      doctest::Contains("rotary_dim/2"), std::invalid_argument);

  params = LongRoPEParameters();
  params.long_factor.clear();
  CHECK_THROWS_WITH_AS(vllm::get_rope(16, 128, true, params),
                       doctest::Contains("requires short_factor"),
                       std::invalid_argument);
}

TEST_CASE("dynamic NTK factor mode mirrors trained-length base scaling") {
  const vllm::RopeParameters params = DynamicFactorParameters();
  auto rope = vllm::get_rope(16, 128, true, params, vt::DType::kF32);
  auto dynamic = std::dynamic_pointer_cast<
      vllm::DynamicNTKScalingRotaryEmbedding>(rope);
  REQUIRE(dynamic != nullptr);
  CHECK(dynamic->scaling_factor() == doctest::Approx(4.0));
  CHECK(dynamic->max_trained_positions() == 32);
  CHECK(rope->cache_rows() == 128);

  const double scale = 4.0 * 128.0 / 32.0 - 3.0;
  const double scaled_base = 10000.0 * std::pow(scale, 8.0 / 6.0);
  for (int64_t pair = 0; pair < 4; ++pair) {
    const float expected =
        1.0F / std::pow(static_cast<float>(scaled_base),
                        static_cast<float>(2 * pair) / 8.0F);
    const float got = std::atan2(CacheValue(*rope, 1, 4 + pair),
                                 CacheValue(*rope, 1, pair));
    CHECK(got == doctest::Approx(expected).epsilon(1e-5));
  }
}

TEST_CASE("dynamic NTK factor one reduces to default RoPE") {
  vllm::RopeParameters dynamic_params;
  dynamic_params.rope_type = "dynamic";
  dynamic_params.rope_dim = 8;
  dynamic_params.factor = 1.0;
  auto dynamic =
      vllm::get_rope(16, 128, true, dynamic_params, vt::DType::kF32);

  vllm::RopeParameters default_params;
  default_params.rope_dim = 8;
  auto ordinary =
      vllm::get_rope(16, 128, true, default_params, vt::DType::kF32);
  REQUIRE(dynamic->cache_bytes() == ordinary->cache_bytes());
  const auto* got = static_cast<const std::byte*>(dynamic->cache_data());
  const auto* want = static_cast<const std::byte*>(ordinary->cache_data());
  CHECK(std::equal(got, got + dynamic->cache_bytes(), want));
}

TEST_CASE("dynamic NTK alpha mode wins dispatch and guards dimensions") {
  vllm::RopeParameters params = DynamicFactorParameters();
  params.alpha = 2.0;
  auto rope = vllm::get_rope(16, 128, false, params, vt::DType::kF32);
  auto alpha = std::dynamic_pointer_cast<
      vllm::DynamicNTKAlphaRotaryEmbedding>(rope);
  REQUIRE(alpha != nullptr);
  CHECK(alpha->scaling_alpha() == doctest::Approx(2.0));
  CHECK(rope->is_neox_style() == false);

  const double scaled_base = 10000.0 * std::pow(2.0, 8.0 / 6.0);
  for (int64_t pair = 0; pair < 4; ++pair) {
    const float expected =
        1.0F / std::pow(static_cast<float>(scaled_base),
                        static_cast<float>(2 * pair) / 8.0F);
    const float got = std::atan2(CacheValue(*rope, 1, 4 + pair),
                                 CacheValue(*rope, 1, pair));
    CHECK(got == doctest::Approx(expected).epsilon(1e-5));
  }

  params.rope_dim = 2;
  CHECK_THROWS_WITH_AS(vllm::get_rope(16, 128, true, params),
                       doctest::Contains("greater than 2"),
                       std::invalid_argument);

  params = {};
  params.rope_type = "dynamic";
  CHECK_THROWS_WITH_AS(vllm::get_rope(16, 128, true, params),
                       doctest::Contains("either alpha or factor"),
                       std::invalid_argument);
}

TEST_CASE("mrope next text positions mirror three equal streams") {
  const auto positions =
      vllm::MRotaryEmbedding::get_next_input_positions(5, 3, 7);
  for (const auto& axis : positions) {
    CHECK(axis == std::vector<int64_t>{8, 9, 10, 11});
  }
}

TEST_CASE("mrope model-driven TP/CUDA cases remain tracked dependencies" *
          doctest::skip(true) *
          doctest::description(
              "MODEL-MM-qwen2-vl-qwen2-vlfor-conditional-generation and TP rows")) {
  MESSAGE("SKIP MODEL-MM-qwen2-vl-qwen2-vlfor-conditional-generation: "
          "multimodal position construction and model family are not ported; "
          "TP=2 remains owned by its scale-out row");
}

TEST_CASE("nomic YaRN feature-positive e2e remains a model dependency" *
          doctest::skip(true) *
          doctest::description(
              "MODEL-EMBED-bert-with-rope-nomic-bert-model")) {
  MESSAGE("SKIP MODEL-EMBED-bert-with-rope-nomic-bert-model: Nomic pooling "
          "model family is not ported; typed config and operator parity land in W5");
}

TEST_CASE("Llama 3 feature-positive e2e remains a model dependency" *
          doctest::skip(true) *
          doctest::description("MODEL-TEXT-llama-llama-for-causal-lm")) {
  MESSAGE("SKIP MODEL-TEXT-llama-llama-for-causal-lm: the Llama model family "
          "is not ported; W6 closes formula/cache/operator parity only");
}

TEST_CASE("Phi-3 LongRoPE feature-positive e2e remains a model dependency" *
          doctest::skip(true) *
          doctest::description("MODEL-TEXT-phi3-phi3-for-causal-lm")) {
  MESSAGE("SKIP MODEL-TEXT-phi3-phi3-for-causal-lm: the Phi-3 model family "
          "is not ported; W7 closes formula/cache/operator parity only");
}

TEST_CASE("dynamic NTK feature-positive e2e remains a model dependency" *
          doctest::skip(true) *
          doctest::description(
              "MODEL-TEXT-hunyuan-v1-hun-yuan-dense-v1-for-causal-lm")) {
  MESSAGE("SKIP MODEL-TEXT-hunyuan-v1-hun-yuan-dense-v1-for-causal-lm: "
          "the Hunyuan family is not ported; W8 closes formula/cache parity");
}
