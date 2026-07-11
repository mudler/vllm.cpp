// Ported from:
//   tests/kernels/core/test_pos_encoding.py:66-193
//   tests/kernels/core/test_mrope.py:47-235
//   tests/kernels/core/test_apply_rotary_emb.py:43-203
//   tests/models/language/pooling/test_nomic_max_model_len.py:93-111
// and pinned class behavior in rotary_embedding/{__init__,base,common,
// yarn_scaling_rope,mrope}.py @ e24d1b24fe96.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/layers/rotary_embedding/common.h"
#include "vllm/model_executor/layers/rotary_embedding/mrope.h"
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
