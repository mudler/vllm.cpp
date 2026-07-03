// Tests for the KV-cache SPEC hierarchy + config wrappers (M1.3 Task 1),
// ported from vllm/tests/v1/core/test_kv_cache_utils.py and
// vllm/tests/v1/worker/test_attn_utils.py @ e24d1b24.
//
// Ported oracles:
//   - FullAttentionSpec.page_size_bytes / real_page_size_bytes:
//       * the `new_kv_cache_spec()` default dims (block_size=16, num_kv_heads=2,
//         head_size=64, float32) used throughout test_kv_cache_utils.py
//         (page_size_bytes = 16384).
//       * the page_size_padded cases from test_attn_utils.py
//         test_reshape_padded_flash_attention_* (real=256, page=384) and
//         test_reshape_padded_hnd_* (real=768, page=1024).
//   - MambaSpec.page_size_bytes: the `new_mamba_spec()` default state shapes
//     ((2,512),(3,32,32), float32) from test_kv_cache_utils.py
//     (page_size_bytes = 16384), plus the page_size_padded override.
//   - KVCacheTensor / KVCacheGroupSpec / KVCacheConfig construction mirrors
//     test_get_kv_cache_configs_multiple_workers (num_blocks=10, per-layer
//     tensors of page_size_bytes*10, one group over [layer1, layer2]).
//   - KVCacheConfig.has_mamba_layers / needs_kv_cache_zeroing (upstream
//     properties).
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

using vllm::v1::FullAttentionSpec;
using vllm::v1::KVCacheConfig;
using vllm::v1::KVCacheGroupSpec;
using vllm::v1::KVCacheSpec;
using vllm::v1::KVCacheSpecKind;
using vllm::v1::KVCacheTensor;
using vllm::v1::KVQuantMode;
using vllm::v1::MambaSpec;
using vt::DType;

namespace {

// Mirrors upstream new_kv_cache_spec() defaults (block_size=16, num_kv_heads=2,
// head_size=64, dtype=float32).
std::shared_ptr<FullAttentionSpec> new_kv_cache_spec() {
  return std::make_shared<FullAttentionSpec>(/*block_size=*/16,
                                             /*num_kv_heads=*/2,
                                             /*head_size=*/64, DType::kF32);
}

// Mirrors upstream new_mamba_spec() defaults (shapes=((2,512),(3,32,32)),
// dtypes=(float32,float32), num_speculative_blocks=2).
std::shared_ptr<MambaSpec> new_mamba_spec() {
  return std::make_shared<MambaSpec>(
      /*block_size=*/16,
      std::vector<std::vector<int64_t>>{{2, 512}, {3, 32, 32}},
      std::vector<DType>{DType::kF32, DType::kF32},
      /*page_size_padded=*/std::nullopt, /*mamba_cache_mode=*/"none",
      /*num_speculative_blocks=*/2);
}

}  // namespace

TEST_CASE("FullAttentionSpec page_size_bytes: new_kv_cache_spec defaults") {
  auto spec = new_kv_cache_spec();
  // head_size_v defaults to head_size (upstream __post_init__).
  CHECK(spec->head_size_v == 64);
  // 16 * 2 * (64 + 64) * 4 = 16384.
  CHECK(spec->real_page_size_bytes() == 16384);
  CHECK(spec->page_size_bytes() == 16384);
  CHECK(spec->kind() == KVCacheSpecKind::kFullAttention);
}

TEST_CASE("FullAttentionSpec block_size / storage_block_size accessor") {
  auto spec = new_kv_cache_spec();
  CHECK(spec->block_size == 16);
  CHECK(spec->storage_block_size() == 16);
}

TEST_CASE("FullAttentionSpec real_page_size_bytes: padded flash-attn oracles") {
  // test_reshape_padded_flash_attention_kv_cache_strides_by_page.
  FullAttentionSpec spec(/*block_size=*/16, /*num_kv_heads=*/1, /*head_size=*/2,
                         DType::kF32, /*head_size_v=*/std::nullopt,
                         KVQuantMode::kNone, /*page_size_padded=*/384);
  CHECK(spec.real_page_size_bytes() == 256);  // 16 * 1 * (2 + 2) * 4
  CHECK(spec.page_size_bytes() == 384);        // padded

  // test_reshape_padded_hnd_flash_attention_kv_cache_strides_by_page.
  FullAttentionSpec hnd(/*block_size=*/16, /*num_kv_heads=*/3, /*head_size=*/2,
                        DType::kF32, /*head_size_v=*/std::nullopt,
                        KVQuantMode::kNone, /*page_size_padded=*/1024);
  CHECK(hnd.real_page_size_bytes() == 768);  // 16 * 3 * (2 + 2) * 4
  CHECK(hnd.page_size_bytes() == 1024);       // padded
}

TEST_CASE("FullAttentionSpec asymmetric head_size_v") {
  FullAttentionSpec spec(/*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64,
                         DType::kF32, /*head_size_v=*/32);
  CHECK(spec.head_size_v == 32);
  // 16 * 2 * (64 + 32) * 4 = 12288.
  CHECK(spec.real_page_size_bytes() == 12288);
  CHECK(spec.page_size_bytes() == 12288);
}

TEST_CASE("FullAttentionSpec page_size_padded must be >= real") {
  FullAttentionSpec spec(/*block_size=*/16, /*num_kv_heads=*/1, /*head_size=*/2,
                         DType::kF32, /*head_size_v=*/std::nullopt,
                         KVQuantMode::kNone, /*page_size_padded=*/100);
  CHECK_THROWS_AS(spec.page_size_bytes(), std::runtime_error);
}

TEST_CASE("FullAttentionSpec quantized page-size math is deferred") {
  FullAttentionSpec spec(/*block_size=*/16, /*num_kv_heads=*/1, /*head_size=*/4,
                         DType::kI8, /*head_size_v=*/std::nullopt,
                         KVQuantMode::kInt8PerTokenHead);
  CHECK_THROWS_AS(spec.real_page_size_bytes(), std::runtime_error);
}

TEST_CASE("MambaSpec page_size_bytes: new_mamba_spec defaults") {
  auto spec = new_mamba_spec();
  // prod(2,512)*4 + prod(3,32,32)*4 = 1024*4 + 3072*4 = 4096 + 12288 = 16384.
  CHECK(spec->page_size_bytes() == 16384);
  CHECK(spec->block_size == 16);
  CHECK(spec->num_speculative_blocks == 2);
  CHECK(spec->mamba_cache_mode == "none");
  CHECK(spec->kind() == KVCacheSpecKind::kMamba);
}

TEST_CASE("MambaSpec page_size_padded override") {
  MambaSpec spec(/*block_size=*/16,
                 std::vector<std::vector<int64_t>>{{2, 512}, {3, 32, 32}},
                 std::vector<DType>{DType::kF32, DType::kF32},
                 /*page_size_padded=*/20000);
  CHECK(spec.page_size_bytes() == 20000);

  MambaSpec too_small(/*block_size=*/16,
                      std::vector<std::vector<int64_t>>{{2, 512}},
                      std::vector<DType>{DType::kF32},
                      /*page_size_padded=*/1);
  CHECK_THROWS_AS(too_small.page_size_bytes(), std::runtime_error);
}

TEST_CASE("MambaSpec mixed dtypes contribute their own byte size") {
  // conv state in bf16 (2 bytes), ssm state in f32 (4 bytes).
  MambaSpec spec(/*block_size=*/16,
                 std::vector<std::vector<int64_t>>{{4, 128}, {8, 64}},
                 std::vector<DType>{DType::kBF16, DType::kF32});
  // prod(4,128)*2 + prod(8,64)*4 = 512*2 + 512*4 = 1024 + 2048 = 3072.
  CHECK(spec.page_size_bytes() == 3072);
}

TEST_CASE("KVCacheGroupSpec / KVCacheConfig construction (get_kv_cache_configs)") {
  auto ref = new_kv_cache_spec();
  const int64_t page = ref->page_size_bytes();  // 16384

  KVCacheConfig config{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/
      {KVCacheTensor{/*size=*/page * 10, /*shared_by=*/{"layer1"}},
       KVCacheTensor{/*size=*/page * 10, /*shared_by=*/{"layer2"}}},
      /*kv_cache_groups=*/
      {KVCacheGroupSpec{{"layer1", "layer2"}, ref}}};

  CHECK(config.num_blocks == 10);
  REQUIRE(config.kv_cache_tensors.size() == 2);
  CHECK(config.kv_cache_tensors[0].size == 163840);
  CHECK(config.kv_cache_tensors[0].shared_by == std::vector<std::string>{"layer1"});
  CHECK(config.kv_cache_tensors[1].shared_by == std::vector<std::string>{"layer2"});
  REQUIRE(config.kv_cache_groups.size() == 1);
  CHECK(config.kv_cache_groups[0].layer_names ==
        std::vector<std::string>{"layer1", "layer2"});
  CHECK(config.kv_cache_groups[0].kv_cache_spec->page_size_bytes() == page);
  CHECK(config.kv_cache_groups[0].kv_cache_spec->kind() ==
        KVCacheSpecKind::kFullAttention);
}

TEST_CASE("KVCacheConfig has_mamba_layers / needs_kv_cache_zeroing") {
  // Full-attention only: no mamba, no zeroing.
  KVCacheConfig attn_only{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/{},
      /*kv_cache_groups=*/{KVCacheGroupSpec{{"layer1"}, new_kv_cache_spec()}}};
  CHECK_FALSE(attn_only.has_mamba_layers());
  CHECK_FALSE(attn_only.needs_kv_cache_zeroing());

  // Hybrid GDN + full-attn (the gate models): has mamba => needs zeroing.
  KVCacheConfig hybrid{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/{},
      /*kv_cache_groups=*/
      {KVCacheGroupSpec{{"attn_layer"}, new_kv_cache_spec()},
       KVCacheGroupSpec{{"mamba_layer"}, new_mamba_spec()}}};
  CHECK(hybrid.has_mamba_layers());
  CHECK(hybrid.needs_kv_cache_zeroing());
}
