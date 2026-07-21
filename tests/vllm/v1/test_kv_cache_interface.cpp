// Tests for the KV-cache SPEC hierarchy + config wrappers (M1.3 Task 1),
// ported from vllm/tests/v1/core/test_kv_cache_utils.py,
// vllm/tests/v1/test_kv_cache_spec_registry.py, and
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
//   - ChunkedLocalAttentionSpec sizing/uniform/registry cases port
//     test_single_type_kv_cache_manager.py and test_kv_cache_spec_registry.py.
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
#include "vllm/v1/kv_cache_spec_registry.h"
#include "vt/dtype.h"

using vllm::v1::ChunkedLocalAttentionSpec;
using vllm::v1::FullAttentionSpec;
using vllm::v1::KVCacheConfig;
using vllm::v1::KVCacheGroupSpec;
using vllm::v1::KVCacheSpec;
using vllm::v1::KVCacheSpecKind;
using vllm::v1::KVCacheManagerKind;
using vllm::v1::KVCacheSpecRegistry;
using vllm::v1::KVCacheTensor;
using vllm::v1::KVQuantMode;
using vllm::v1::MambaSpec;
using vllm::v1::SlidingWindowSpec;
using vllm::v1::are_uniform_kv_cache_specs;
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

std::shared_ptr<SlidingWindowSpec> new_sliding_window_spec(
    int sliding_window = 1024, int block_size = 16) {
  return std::make_shared<SlidingWindowSpec>(
      block_size, /*num_kv_heads=*/2, /*head_size=*/64, DType::kF32,
      sliding_window);
}

std::shared_ptr<ChunkedLocalAttentionSpec> new_chunked_local_spec(
    int attention_chunk_size = 512, int block_size = 16) {
  return std::make_shared<ChunkedLocalAttentionSpec>(
      block_size, /*num_kv_heads=*/2, /*head_size=*/64, DType::kF32,
      attention_chunk_size);
}

struct CustomFullSpec : FullAttentionSpec {
  using FullAttentionSpec::FullAttentionSpec;
};

struct CustomChunkedLocalSpec : ChunkedLocalAttentionSpec {
  using ChunkedLocalAttentionSpec::ChunkedLocalAttentionSpec;
};

struct TrulyUnregisteredSpec : KVCacheSpec {
  TrulyUnregisteredSpec() : KVCacheSpec(/*block_size=*/16) {}
  int64_t page_size_bytes() const override { return 0; }
  KVCacheSpecKind kind() const override { return KVCacheSpecKind::kUnknown; }
};

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

// ─── MLAAttentionSpec (MLA campaign W1 — allocation metadata only) ───────────
//
// Upstream oracle: vllm/v1/kv_cache_interface.py:363 MLAAttentionSpec, page
// formula :380-398 =
//   storage_block_size * num_kv_heads * head_dim * dtype_size
// i.e. NO factor 2 and NO separate V, unlike every other attention spec.
// The DeepSeek geometry (confirmed against the real
// deepseek-ai/DeepSeek-V2-Lite config.json at W0) is kv_lora_rank=512 +
// qk_rope_head_dim=64 = a 576-wide latent with num_kv_heads == 1.
TEST_CASE("MLAAttentionSpec page_size_bytes: DeepSeek 576-wide latent") {
  // DeepSeek-V2-Lite / V3 / Kimi-Linear MLA layers, bf16, block 16.
  vllm::v1::MLAAttentionSpec spec(/*block_size=*/16, /*head_size=*/576,
                                  DType::kBF16);
  CHECK(spec.num_kv_heads == 1);
  CHECK(spec.head_size == 576);
  CHECK(spec.head_size_v == 576);
  CHECK(spec.kind() == KVCacheSpecKind::kMlaAttention);
  // 16 * 1 * 576 * 2 = 18432 — the number quoted in the campaign spike §4.1.
  CHECK(spec.real_page_size_bytes() == 18432);
  CHECK(spec.page_size_bytes() == 18432);
}

TEST_CASE("MLAAttentionSpec drops the K+V factor 2") {
  // Same dims through FullAttentionSpec would double the page — that doubling
  // is exactly what the MLA override removes.
  vllm::v1::MLAAttentionSpec mla(/*block_size=*/16, /*head_size=*/576,
                                 DType::kBF16);
  FullAttentionSpec full(/*block_size=*/16, /*num_kv_heads=*/1,
                         /*head_size=*/576, DType::kBF16);
  CHECK(full.real_page_size_bytes() == 2 * mla.real_page_size_bytes());
}

TEST_CASE("MLAAttentionSpec page_size_padded and quantized guard") {
  vllm::v1::MLAAttentionSpec padded(/*block_size=*/16, /*head_size=*/576,
                                    DType::kBF16, /*num_kv_heads=*/1,
                                    KVQuantMode::kNone,
                                    /*page_size_padded=*/20480);
  CHECK(padded.real_page_size_bytes() == 18432);
  CHECK(padded.page_size_bytes() == 20480);

  // fp8_ds_mla (V3.2 656 B/token, V4 584 B/token) and int4 per-token-head are
  // OUT OF SCOPE for this campaign and must throw, never silently mis-size.
  vllm::v1::MLAAttentionSpec quantized(/*block_size=*/16, /*head_size=*/576,
                                       DType::kBF16, /*num_kv_heads=*/1,
                                       KVQuantMode::kFp8PerTensor);
  CHECK_THROWS_AS(quantized.real_page_size_bytes(), std::runtime_error);
}

TEST_CASE("MLAAttentionSpec maps to the ORDINARY full-attention manager") {
  // Upstream vllm/v1/core/single_type_kv_cache_manager.py:1539 registers
  // MLAAttentionSpec -> FullAttentionManager with
  // uniform_type_base_spec=FullAttentionSpec: MLA-ness is a page-SIZE and
  // tensor-SHAPE concern only, so block table / prefix caching / eviction are
  // untouched.
  vllm::v1::MLAAttentionSpec spec(/*block_size=*/16, /*head_size=*/576,
                                  DType::kBF16);
  CHECK(KVCacheSpecRegistry::get_manager_kind(spec) ==
        KVCacheManagerKind::kFullAttention);
  CHECK(KVCacheSpecRegistry::get_uniform_type_base_spec(spec) ==
        std::optional<std::type_index>{typeid(FullAttentionSpec)});
}

TEST_CASE("SlidingWindowSpec page size, head_size_v, and admission cap") {
  SlidingWindowSpec spec(/*block_size=*/16, /*num_kv_heads=*/2,
                         /*head_size=*/64, DType::kF32,
                         /*sliding_window=*/1024);
  CHECK(spec.head_size_v == 64);
  CHECK(spec.real_page_size_bytes() == 16384);
  CHECK(spec.page_size_bytes() == 16384);
  CHECK(spec.kind() == KVCacheSpecKind::kSlidingWindow);

  // min(1024 - 1 + 256, 8192) = 1279; cdiv(1279, 16) + 1 = 81.
  CHECK(spec.max_admission_blocks_per_request(
            /*max_num_batched_tokens=*/256, /*max_model_len=*/8192) == 81);
  // The model-length clamp is part of the same upstream formula.
  CHECK(spec.max_admission_blocks_per_request(256, 1000) == 64);

  SlidingWindowSpec asymmetric(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64, DType::kF32,
      /*sliding_window=*/1024, /*head_size_v=*/32,
      KVQuantMode::kNone, /*page_size_padded=*/16000);
  CHECK(asymmetric.real_page_size_bytes() == 12288);
  CHECK(asymmetric.page_size_bytes() == 16000);
}

TEST_CASE("SlidingWindowSpec quantized page-size math is deferred") {
  SlidingWindowSpec spec(/*block_size=*/16, /*num_kv_heads=*/1,
                         /*head_size=*/4, DType::kI8,
                         /*sliding_window=*/32, std::nullopt,
                         KVQuantMode::kInt8PerTokenHead);
  CHECK_THROWS_AS(spec.real_page_size_bytes(), std::runtime_error);
}

TEST_CASE("ChunkedLocalAttentionSpec page size and admission cap") {
  ChunkedLocalAttentionSpec spec(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64, DType::kF32,
      /*attention_chunk_size=*/512, KVQuantMode::kNone,
      /*page_size_padded=*/20000);
  CHECK(spec.kind() == KVCacheSpecKind::kChunkedLocalAttention);
  // Inherits AttentionSpec's symmetric K+V page formula.
  CHECK(spec.real_page_size_bytes() == 16384);
  CHECK(spec.page_size_bytes() == 20000);
  // min(chunk + max_batch, max_model) = min(512 + 255, 8192) = 767;
  // cdiv(767, 16) = 48. Unlike SWA, no unaligned-window +1 is needed because
  // attention_chunk_size is block-aligned by the backend contract.
  CHECK(spec.max_admission_blocks_per_request(255, 8192) == 48);
  CHECK(spec.max_admission_blocks_per_request(255, 500) == 32);
}

TEST_CASE("KVCacheSpecRegistry built-ins and inherited custom specs") {
  auto full = new_kv_cache_spec();
  auto sliding = new_sliding_window_spec();
  auto chunked = new_chunked_local_spec();
  auto mamba = new_mamba_spec();

  CHECK(KVCacheSpecRegistry::get_manager_kind(*full) ==
        KVCacheManagerKind::kFullAttention);
  CHECK(KVCacheSpecRegistry::get_manager_kind(*sliding) ==
        KVCacheManagerKind::kSlidingWindow);
  CHECK(KVCacheSpecRegistry::get_manager_kind(*chunked) ==
        KVCacheManagerKind::kChunkedLocalAttention);
  CHECK(KVCacheSpecRegistry::get_manager_kind(*mamba) ==
        KVCacheManagerKind::kMamba);
  CHECK(KVCacheSpecRegistry::get_uniform_type_base_spec(*sliding) ==
        std::optional<std::type_index>{typeid(SlidingWindowSpec)});
  CHECK(KVCacheSpecRegistry::get_uniform_type_base_spec(*chunked) ==
        std::optional<std::type_index>{typeid(ChunkedLocalAttentionSpec)});

  // Explicit custom registration is idempotent with inherited built-in
  // behavior, and an unregistered subclass still resolves through its base.
  CustomFullSpec custom(/*block_size=*/16, /*num_kv_heads=*/2,
                        /*head_size=*/64, DType::kF32);
  CHECK(KVCacheSpecRegistry::get_manager_kind(custom) ==
        KVCacheManagerKind::kFullAttention);
  KVCacheSpecRegistry::register_spec<CustomFullSpec, FullAttentionSpec>(
      KVCacheManagerKind::kFullAttention);
  CHECK(KVCacheSpecRegistry::get_uniform_type_base_spec(custom) ==
        std::optional<std::type_index>{typeid(FullAttentionSpec)});

  CustomChunkedLocalSpec custom_chunked(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64, DType::kF32,
      /*attention_chunk_size=*/512);
  CHECK(KVCacheSpecRegistry::get_manager_kind(custom_chunked) ==
        KVCacheManagerKind::kChunkedLocalAttention);
  KVCacheSpecRegistry::register_spec<CustomChunkedLocalSpec,
                                     ChunkedLocalAttentionSpec>(
      KVCacheManagerKind::kChunkedLocalAttention);
  CHECK(KVCacheSpecRegistry::get_uniform_type_base_spec(custom_chunked) ==
        std::optional<std::type_index>{typeid(ChunkedLocalAttentionSpec)});
}

TEST_CASE("KVCacheSpecRegistry rejects a truly unregistered spec") {
  TrulyUnregisteredSpec unknown;
  CHECK_FALSE(KVCacheSpecRegistry::get_manager_kind(unknown).has_value());
  CHECK_FALSE(
      KVCacheSpecRegistry::get_uniform_type_base_spec(unknown).has_value());
  CHECK_THROWS_WITH_AS(
      KVCacheSpecRegistry::check_kv_cache_spec_registry(
          {{"layer_0", &unknown}}),
      "Unsupported KV cache spec type for layer layer_0",
      std::invalid_argument);
}

TEST_CASE("KVCacheSpecRegistry uniform-type rules include local-attention fields") {
  auto sliding_a = new_sliding_window_spec(/*sliding_window=*/1024);
  auto sliding_b = new_sliding_window_spec(/*sliding_window=*/1024);
  auto sliding_other = new_sliding_window_spec(/*sliding_window=*/256);
  auto full = new_kv_cache_spec();
  auto chunked_a = new_chunked_local_spec(/*attention_chunk_size=*/512);
  auto chunked_b = new_chunked_local_spec(/*attention_chunk_size=*/512);
  auto chunked_other = new_chunked_local_spec(/*attention_chunk_size=*/256);

  CHECK(are_uniform_kv_cache_specs({sliding_a.get(), sliding_b.get()}));
  CHECK_FALSE(
      are_uniform_kv_cache_specs({sliding_a.get(), sliding_other.get()}));
  CHECK_FALSE(are_uniform_kv_cache_specs({full.get(), sliding_a.get()}));
  CHECK(are_uniform_kv_cache_specs({chunked_a.get(), chunked_b.get()}));
  CHECK_FALSE(
      are_uniform_kv_cache_specs({chunked_a.get(), chunked_other.get()}));
  CHECK_FALSE(are_uniform_kv_cache_specs({sliding_a.get(), chunked_a.get()}));
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
