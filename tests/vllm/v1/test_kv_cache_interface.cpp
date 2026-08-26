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

using vllm::v1::AttentionSpec;
using vllm::v1::ChunkedLocalAttentionSpec;
using vllm::v1::FullAttentionSpec;
using vllm::v1::KVBytesPerBlock;
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

// ─── KVBytesPerBlock (ROAD-V1-MEM M2) ────────────────────────────────────────
// The marginal device bytes the paged KV pool grows by per additional block:
// group-aware, mirroring the runner's own `num_blocks * page_size_bytes()`
// per-attention-layer allocation, with GDN/Mamba state (per-sequence-slot, not
// per-block) contributing nothing.

TEST_CASE("KVBytesPerBlock: dense group weighted by layer count") {
  auto ref = new_kv_cache_spec();  // page_size_bytes == 16384
  // One FullAttention group spanning two layers -> 2 * 16384.
  KVCacheConfig config{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/{},
      /*kv_cache_groups=*/{KVCacheGroupSpec{{"layer1", "layer2"}, ref}}};
  CHECK(KVBytesPerBlock(config) == 2 * 16384);
}

TEST_CASE("KVBytesPerBlock: MLA drops the K+V factor 2") {
  // DeepSeek 576-wide latent, bf16, block 16 -> 18432 (from the MLA spec test).
  auto mla = std::make_shared<vllm::v1::MLAAttentionSpec>(
      /*block_size=*/16, /*head_size=*/576, DType::kBF16);
  KVCacheConfig config{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/{},
      /*kv_cache_groups=*/{KVCacheGroupSpec{{"mla_layer"}, mla}}};
  CHECK(KVBytesPerBlock(config) == 18432);
}

TEST_CASE("KVBytesPerBlock: hybrid excludes the Mamba group") {
  // The gate-model shape: one attention layer + one GDN/Mamba layer. Only the
  // attention layer scales with the block count in the runner.
  KVCacheConfig hybrid{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/{},
      /*kv_cache_groups=*/
      {KVCacheGroupSpec{{"attn_layer"}, new_kv_cache_spec()},
       KVCacheGroupSpec{{"mamba_layer"}, new_mamba_spec()}}};
  CHECK(KVBytesPerBlock(hybrid) == 16384);  // NOT 16384 + mamba page
}

TEST_CASE("KVBytesPerBlock: heterogeneous per-layer specs sum, GDN nulls skip") {
  // Gemma-4-style het-KV: per_layer_attn_specs published, one spec per non-GDN
  // layer, null for GDN layers. When populated it wins over the group specs.
  auto small = std::make_shared<FullAttentionSpec>(
      /*block_size=*/16, /*num_kv_heads=*/1, /*head_size=*/64, DType::kF32);
  // 16 * 1 * (64 + 64) * 4 = 8192.
  CHECK(small->page_size_bytes() == 8192);
  auto big = new_kv_cache_spec();  // 16384
  KVCacheConfig config{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/{},
      // Groups present but IGNORED because per_layer_attn_specs is populated.
      /*kv_cache_groups=*/{KVCacheGroupSpec{{"g"}, new_kv_cache_spec()}}};
  config.per_layer_attn_specs = {
      std::static_pointer_cast<AttentionSpec>(big),
      std::static_pointer_cast<AttentionSpec>(small),
      nullptr,  // a GDN/linear layer contributes nothing
  };
  CHECK(KVBytesPerBlock(config) == 16384 + 8192);
}

TEST_CASE("KVBytesPerBlock: the M1 absolute-bytes divisor") {
  // The exact arithmetic ResolveNumBlocks does for --kv-cache-memory: a budget
  // divided by the per-block bytes yields the block count (floored).
  KVCacheConfig config{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/{},
      /*kv_cache_groups=*/{KVCacheGroupSpec{{"layer1"}, new_kv_cache_spec()}}};
  const int64_t bpb = KVBytesPerBlock(config);  // 16384
  const int64_t budget = int64_t{16384} * 100 + 1;  // 100 blocks + a remainder
  CHECK(budget / bpb == 100);
}

// ─── KV-DSV4-MULTICACHE W1 (#1960) — the compressed, aligned MLA page ────────
//
// Upstream oracles, at the parity pin 5559679229bc961848b121ccdeaa8fa5d79bec98:
//   - MLAAttentionSpec fields + storage_block_size + real_page_size_bytes:
//       vllm/v1/kv_cache_interface.py:381-410
//   - SlidingWindowMLASpec:  :611-642
//   - _apply_alignment_padding / round_up:
//       :345-351 and vllm/utils/math_utils.py:20-22
//
// Every EXPECTED value below is derived from an upstream CONSTRUCTION site --
// the four DeepSeek-V4 declaration sites -- not from a prose restatement:
//   (a) DeepseekV4SWACache.get_kv_cache_spec
//         vllm/v1/attention/backends/mla/sparse_swa.py:86-101
//   (b) DeepseekV4Attention.get_kv_cache_spec
//         vllm/models/deepseek_v4/attention.py:626-645
//   (c) DeepseekV4IndexerCache.get_kv_cache_spec
//         vllm/models/deepseek_v4/attention.py:669-684
//   (d) CompressorStateCache.get_kv_cache_spec
//         vllm/models/deepseek_v4/compressor.py:188-200
//
// The configured cache_config.block_size on this geometry is 256. That is not
// assumed: sparse_swa.py:76-83 derives the SWA block size of 64 from "the C4A
// KV block shape [256//4, head_dim] = [64, head_dim]", and compressor.py:174-178
// repeats the same [256//4, head_dim] = [64, 584] derivation. Both hold only at
// 256.
//
// torch.uint8 (upstream's fp8_ds_mla storage dtype) maps to vt::DType::kI8
// here: 1 byte per element, the same mapping src/vllm/v1/kv_cache_interface.cpp
// already asserts for the fp8 KV store.

using vllm::v1::MLAAttentionSpec;
using vllm::v1::SlidingWindowMLASpec;

namespace {

// Upstream round_up (vllm/utils/math_utils.py:20-22), recomputed in the test so
// the expectation does not read the implementation it gates.
int64_t upstream_round_up(int64_t x, int64_t y) { return ((x + y - 1) / y) * y; }

}  // namespace

// ---------------------------------------------------------------------------
// BYTE-NEUTRALITY. Seven call sites construct MLAAttentionSpec today, all with
// exactly three positional arguments (deepseek_v2_registry.cpp:173,
// deepseek_v4_registry.cpp:145, glm4_moe_lite_registry.cpp:176,
// kimi_k3_registry.cpp:121, kimi_linear_registry.cpp:148,
// minicpm3_registry.cpp:117, dots3_note.cpp:697), each passing
// head_size = kv_lora_rank + qk_rope_head_dim. W1 adds four fields and a
// storage_block_size() override; if any of those changes what those seven
// allocate, that is a defect and not a refinement.
// ---------------------------------------------------------------------------
TEST_CASE("W1 byte-neutrality: the 3-argument MLA spec every model builds") {
  struct Case {
    int block_size;
    int head_size;
    DType dtype;
    int64_t expected_page;
  };
  // head_size 576 = kv_lora_rank 512 + qk_rope_head_dim 64, the DeepSeek-V2 /
  // V3 / V4 / Kimi-K3 / MiniCPM3 / GLM4-MoE-Lite / dots3-note geometry; 640 and
  // 512 stand for the other kv_lora_rank/rope splits the same accessor yields.
  const Case cases[] = {
      {16, 576, DType::kBF16, 16LL * 576 * 2},
      {16, 576, DType::kF32, 16LL * 576 * 4},
      {32, 576, DType::kBF16, 32LL * 576 * 2},
      {64, 576, DType::kBF16, 64LL * 576 * 2},
      {16, 640, DType::kBF16, 16LL * 640 * 2},
      {16, 512, DType::kBF16, 16LL * 512 * 2},
  };
  for (const Case& c : cases) {
    CAPTURE(c.block_size);
    CAPTURE(c.head_size);
    MLAAttentionSpec spec(c.block_size, c.head_size, c.dtype);
    // The four new fields sit at upstream's own defaults
    // (kv_cache_interface.py:383-387), which is what makes the page unchanged.
    CHECK(spec.compress_ratio == 1);
    CHECK_FALSE(spec.alignment.has_value());
    CHECK_FALSE(spec.cache_dtype_str.has_value());
    CHECK_FALSE(spec.model_version.has_value());
    // compress_ratio 1 => storage_block_size is block_size, so the formula is
    // the one that ran before W1.
    CHECK(spec.storage_block_size() == c.block_size);
    // No alignment => the padding helper wrote nothing, so page_size_bytes is
    // the raw page and NOT a rounded-up one.
    CHECK_FALSE(spec.page_size_padded.has_value());
    CHECK(spec.real_page_size_bytes() == c.expected_page);
    CHECK(spec.page_size_bytes() == c.expected_page);
  }
}

TEST_CASE("W1 byte-neutrality: the non-MLA specs are untouched") {
  // The header these four share is the file W1 edits, so their pages are pinned
  // here as literals beside the MLA ones.
  FullAttentionSpec full(/*block_size=*/16, /*num_kv_heads=*/2,
                         /*head_size=*/64, DType::kF32);
  CHECK(full.storage_block_size() == 16);
  CHECK(full.page_size_bytes() == 16384);  // 16 * 2 * (64+64) * 4

  SlidingWindowSpec sw(/*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64,
                       DType::kF32, /*sliding_window=*/1024);
  CHECK(sw.storage_block_size() == 16);
  CHECK(sw.page_size_bytes() == 16384);
  CHECK(sw.kind() == KVCacheSpecKind::kSlidingWindow);

  ChunkedLocalAttentionSpec chunked(/*block_size=*/16, /*num_kv_heads=*/2,
                                    /*head_size=*/64, DType::kF32,
                                    /*attention_chunk_size=*/1024);
  CHECK(chunked.storage_block_size() == 16);
  CHECK(chunked.page_size_bytes() == 16384);  // 2 * 16 * 2 * 64 * 4

  MambaSpec mamba(/*block_size=*/16, /*shapes=*/{{2, 512}, {3, 32, 32}},
                  /*dtypes=*/{DType::kF32, DType::kF32});
  CHECK(mamba.storage_block_size() == 16);
  CHECK(mamba.page_size_bytes() == 16384);
}

// ---------------------------------------------------------------------------
// storage_block_size: block_size // compress_ratio on BOTH classes
// (kv_cache_interface.py:393-395 and :623-625).
// ---------------------------------------------------------------------------
TEST_CASE("W1 storage_block_size divides by compress_ratio") {
  // (b) at cache_config.block_size = 256, the two DeepSeek-V4 compressed ratios.
  MLAAttentionSpec c4(/*block_size=*/256, /*head_size=*/512, DType::kI8,
                      /*num_kv_heads=*/1, KVQuantMode::kNone,
                      /*page_size_padded=*/std::nullopt,
                      /*indexes_kv_by_block_stride=*/false,
                      /*cache_dtype_str=*/std::nullopt,
                      /*alignment=*/std::nullopt, /*compress_ratio=*/4);
  CHECK(c4.storage_block_size() == 64);

  MLAAttentionSpec c128(/*block_size=*/256, /*head_size=*/512, DType::kI8,
                        /*num_kv_heads=*/1, KVQuantMode::kNone,
                        /*page_size_padded=*/std::nullopt,
                        /*indexes_kv_by_block_stride=*/false,
                        /*cache_dtype_str=*/std::nullopt,
                        /*alignment=*/std::nullopt, /*compress_ratio=*/128);
  CHECK(c128.storage_block_size() == 2);

  // The element formula is driven by storage_block_size, not block_size
  // (kv_cache_interface.py:406-410): 2 rows * 512 * 1 byte.
  CHECK(c128.real_page_size_bytes() == 2LL * 512);

  SlidingWindowMLASpec sw_c4(/*block_size=*/256, /*num_kv_heads=*/1,
                             /*head_size=*/512, DType::kI8,
                             /*sliding_window=*/128,
                             /*cache_dtype_str=*/std::nullopt,
                             /*alignment=*/std::nullopt, /*compress_ratio=*/4);
  CHECK(sw_c4.storage_block_size() == 64);
  CHECK(sw_c4.real_page_size_bytes() == 64LL * 1 * 512 * 1);
}

// ---------------------------------------------------------------------------
// The four DeepSeek-V4 declaration sites, page for page.
// ---------------------------------------------------------------------------
TEST_CASE("W1 (a) SWA cache: sparse_swa.py:86-101") {
  // block_size=64 (sparse_swa.py:79-82), num_kv_heads=1, head_size=head_dim=512,
  // dtype uint8, sliding_window=128, cache_dtype_str="fp8_ds_mla",
  // alignment=576, model_version="deepseek_v4",
  // kv_quant_mode=get_kv_quant_mode("fp8_ds_mla") == FP8_PER_TENSOR.
  SlidingWindowMLASpec spec(/*block_size=*/64, /*num_kv_heads=*/1,
                            /*head_size=*/512, DType::kI8,
                            /*sliding_window=*/128,
                            /*cache_dtype_str=*/std::string("fp8_ds_mla"),
                            /*alignment=*/576, /*compress_ratio=*/1,
                            /*model_version=*/std::string("deepseek_v4"),
                            KVQuantMode::kFp8PerTensor);
  CHECK(spec.kind() == KVCacheSpecKind::kSlidingWindowMla);
  CHECK(spec.sliding_window == 128);
  CHECK(spec.storage_block_size() == 64);
  // 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token
  // (kv_cache_interface.py:629-632).
  CHECK(spec.real_page_size_bytes() == 64LL * 584);
  CHECK(spec.real_page_size_bytes() == 37376);
  // _apply_alignment_padding wrote page_size_padded (:345-351).
  REQUIRE(spec.page_size_padded.has_value());
  CHECK(*spec.page_size_padded == upstream_round_up(37376, 576));
  CHECK(*spec.page_size_padded == 37440);
  CHECK(spec.page_size_bytes() == 37440);
}

TEST_CASE("W1 (b) compressed MLA latent: attention.py:631-645") {
  // C4A layer: block_size=cache_config.block_size=256, head_size=head_dim=512,
  // compress_ratio=4, uint8, fp8_ds_mla, alignment=576, deepseek_v4.
  MLAAttentionSpec c4(/*block_size=*/256, /*head_size=*/512, DType::kI8,
                      /*num_kv_heads=*/1, KVQuantMode::kFp8PerTensor,
                      /*page_size_padded=*/std::nullopt,
                      /*indexes_kv_by_block_stride=*/false,
                      /*cache_dtype_str=*/std::string("fp8_ds_mla"),
                      /*alignment=*/576, /*compress_ratio=*/4,
                      /*model_version=*/std::string("deepseek_v4"));
  CHECK(c4.kind() == KVCacheSpecKind::kMlaAttention);
  CHECK(c4.storage_block_size() == 64);
  CHECK(c4.real_page_size_bytes() == 64LL * 584);
  REQUIRE(c4.page_size_padded.has_value());
  CHECK(*c4.page_size_padded == 37440);
  CHECK(c4.page_size_bytes() == 37440);

  // C128A layer: same site, ratio 128.
  MLAAttentionSpec c128(/*block_size=*/256, /*head_size=*/512, DType::kI8,
                        /*num_kv_heads=*/1, KVQuantMode::kFp8PerTensor,
                        /*page_size_padded=*/std::nullopt,
                        /*indexes_kv_by_block_stride=*/false,
                        /*cache_dtype_str=*/std::string("fp8_ds_mla"),
                        /*alignment=*/576, /*compress_ratio=*/128,
                        /*model_version=*/std::string("deepseek_v4"));
  CHECK(c128.storage_block_size() == 2);
  CHECK(c128.real_page_size_bytes() == 2LL * 584);
  CHECK(c128.real_page_size_bytes() == 1168);
  REQUIRE(c128.page_size_padded.has_value());
  CHECK(*c128.page_size_padded == upstream_round_up(1168, 576));
  CHECK(*c128.page_size_padded == 1728);
}

TEST_CASE("W1 SWA and C4A pages MATCH: they share one physical tensor") {
  // sparse_swa.py:76-83 fixes the SWA block size at 64 *because* the SWA and
  // C4A blocks "share the same physical tensor, [so] they must use the same
  // page size". The two specs reach that page by different routes -- a
  // SlidingWindowMLASpec at block 64 / ratio 1 and an MLAAttentionSpec at
  // block 256 / ratio 4 -- so this equality holds only if both
  // storage_block_size overrides, both 584-byte branches and the shared 576B
  // padding are right at once.
  SlidingWindowMLASpec swa(/*block_size=*/64, /*num_kv_heads=*/1,
                           /*head_size=*/512, DType::kI8,
                           /*sliding_window=*/128,
                           /*cache_dtype_str=*/std::string("fp8_ds_mla"),
                           /*alignment=*/576, /*compress_ratio=*/1,
                           /*model_version=*/std::string("deepseek_v4"),
                           KVQuantMode::kFp8PerTensor);
  MLAAttentionSpec c4a(/*block_size=*/256, /*head_size=*/512, DType::kI8,
                       /*num_kv_heads=*/1, KVQuantMode::kFp8PerTensor,
                       /*page_size_padded=*/std::nullopt,
                       /*indexes_kv_by_block_stride=*/false,
                       /*cache_dtype_str=*/std::string("fp8_ds_mla"),
                       /*alignment=*/576, /*compress_ratio=*/4,
                       /*model_version=*/std::string("deepseek_v4"));
  CHECK(swa.storage_block_size() == c4a.storage_block_size());
  CHECK(swa.real_page_size_bytes() == c4a.real_page_size_bytes());
  CHECK(swa.page_size_bytes() == c4a.page_size_bytes());
  CHECK(swa.page_size_bytes() == 37440);
}

TEST_CASE("W1 (c) indexer key cache: attention.py:669-684") {
  // head_dim already carries the fp8 scale padding: head_dim +
  // head_dim//quant_block_size*4 = 128 + 4 = 132 (attention.py:756-759).
  // NOTE: this site passes NEITHER cache_dtype_str NOR model_version, so it
  // takes the ELEMENT formula, not the 584-byte branch, even though its dtype
  // is uint8 and its alignment is 576.
  MLAAttentionSpec idx(/*block_size=*/256, /*head_size=*/132, DType::kI8,
                       /*num_kv_heads=*/1, KVQuantMode::kNone,
                       /*page_size_padded=*/std::nullopt,
                       /*indexes_kv_by_block_stride=*/false,
                       /*cache_dtype_str=*/std::nullopt,
                       /*alignment=*/576, /*compress_ratio=*/4);
  CHECK(idx.storage_block_size() == 64);
  CHECK(idx.real_page_size_bytes() == 64LL * 1 * 132 * 1);
  CHECK(idx.real_page_size_bytes() == 8448);
  REQUIRE(idx.page_size_padded.has_value());
  CHECK(*idx.page_size_padded == upstream_round_up(8448, 576));
  CHECK(*idx.page_size_padded == 8640);

  // MXFP4 arm of the same site: head_dim//2 + head_dim//32 = 64 + 4 = 68
  // (attention.py:750-753).
  MLAAttentionSpec idx_mxfp4(/*block_size=*/256, /*head_size=*/68, DType::kI8,
                             /*num_kv_heads=*/1, KVQuantMode::kNone,
                             /*page_size_padded=*/std::nullopt,
                             /*indexes_kv_by_block_stride=*/false,
                             /*cache_dtype_str=*/std::nullopt,
                             /*alignment=*/576, /*compress_ratio=*/4);
  CHECK(idx_mxfp4.real_page_size_bytes() == 64LL * 68);
  CHECK(idx_mxfp4.page_size_bytes() == upstream_round_up(64LL * 68, 576));
}

TEST_CASE("W1 (d) compressor state caches: compressor.py:188-200") {
  // state_dim = 2 * coff * head_dim, coff = 1 + (compress_ratio == 4),
  // dtype f32, sliding_window = coff * compress_ratio, block_size 4 for ratio 4
  // and 8 for ratio 128 (compressor.py:170-186, :291-293). This site passes
  // NEITHER cache_dtype_str NOR model_version, so it takes the element formula.

  // The attention layer's own compressor at ratio 4, head_dim 512:
  // coff = 2, state_dim = 2 * 2 * 512 = 2048, sliding_window = 8.
  SlidingWindowMLASpec attn_c4(/*block_size=*/4, /*num_kv_heads=*/1,
                               /*head_size=*/2048, DType::kF32,
                               /*sliding_window=*/8,
                               /*cache_dtype_str=*/std::nullopt,
                               /*alignment=*/576);
  CHECK(attn_c4.kind() == KVCacheSpecKind::kSlidingWindowMla);
  CHECK(attn_c4.compress_ratio == 1);  // this site does NOT pass compress_ratio
  CHECK(attn_c4.storage_block_size() == 4);
  CHECK(attn_c4.real_page_size_bytes() == 4LL * 1 * 2048 * 4);
  CHECK(attn_c4.real_page_size_bytes() == 32768);
  REQUIRE(attn_c4.page_size_padded.has_value());
  CHECK(*attn_c4.page_size_padded == upstream_round_up(32768, 576));
  CHECK(*attn_c4.page_size_padded == 32832);

  // The indexer's compressor (attention.py:768-777) at head_dim 128:
  // state_dim = 2 * 2 * 128 = 512.
  SlidingWindowMLASpec idx_c4(/*block_size=*/4, /*num_kv_heads=*/1,
                              /*head_size=*/512, DType::kF32,
                              /*sliding_window=*/8,
                              /*cache_dtype_str=*/std::nullopt,
                              /*alignment=*/576);
  CHECK(idx_c4.real_page_size_bytes() == 4LL * 512 * 4);
  CHECK(idx_c4.real_page_size_bytes() == 8192);
  REQUIRE(idx_c4.page_size_padded.has_value());
  CHECK(*idx_c4.page_size_padded == 8640);

  // Ratio 128: coff = 1, state_dim = 2 * 1 * 512 = 1024, block_size 8,
  // sliding_window = 128.
  SlidingWindowMLASpec attn_c128(/*block_size=*/8, /*num_kv_heads=*/1,
                                 /*head_size=*/1024, DType::kF32,
                                 /*sliding_window=*/128,
                                 /*cache_dtype_str=*/std::nullopt,
                                 /*alignment=*/576);
  CHECK(attn_c128.sliding_window == 128);
  CHECK(attn_c128.storage_block_size() == 8);
  CHECK(attn_c128.real_page_size_bytes() == 8LL * 1024 * 4);
  CHECK(attn_c128.real_page_size_bytes() == 32768);
  REQUIRE(attn_c128.page_size_padded.has_value());
  CHECK(*attn_c128.page_size_padded == 32832);

  // The FlashInfer-sparse arm of the same sites uses alignment 512 instead of
  // 576 (compressor.py:196, sparse_swa.py:99). 32768 is already a multiple of
  // 512, so _apply_alignment_padding writes NOTHING (:348-351 "if
  // padded_page_size != actual_page_size").
  SlidingWindowMLASpec bf16_arm(/*block_size=*/4, /*num_kv_heads=*/1,
                                /*head_size=*/2048, DType::kF32,
                                /*sliding_window=*/8,
                                /*cache_dtype_str=*/std::nullopt,
                                /*alignment=*/512);
  CHECK(bf16_arm.real_page_size_bytes() == 32768);
  CHECK_FALSE(bf16_arm.page_size_padded.has_value());
  CHECK(bf16_arm.page_size_bytes() == 32768);
}

// ---------------------------------------------------------------------------
// The branch ORDER, which is the one thing a careless port gets wrong.
// ---------------------------------------------------------------------------
TEST_CASE("W1 fp8_ds_mla is tested BEFORE the quantization mode") {
  // attention.py:644 passes kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"), and
  // kv_cache_interface.py:70-71 maps any string starting with "fp8" to
  // FP8_PER_TENSOR. So every real DeepSeek-V4 MLA spec carries a NON-NONE quant
  // mode, and our port throws on non-NONE quant modes
  // (src/vllm/v1/kv_cache_interface.cpp:64-71). Upstream returns from the
  // cache_dtype_str branch (:398-405) before it ever reads kv_quant_mode, so
  // this must NOT throw.
  MLAAttentionSpec spec(/*block_size=*/256, /*head_size=*/512, DType::kI8,
                        /*num_kv_heads=*/1, KVQuantMode::kFp8PerTensor,
                        /*page_size_padded=*/std::nullopt,
                        /*indexes_kv_by_block_stride=*/false,
                        /*cache_dtype_str=*/std::string("fp8_ds_mla"),
                        /*alignment=*/576, /*compress_ratio=*/4,
                        /*model_version=*/std::string("deepseek_v4"));
  CHECK_NOTHROW(static_cast<void>(spec.real_page_size_bytes()));
  CHECK(spec.real_page_size_bytes() == 64LL * 584);

  // WITHOUT the fp8_ds_mla marker the same quant mode still throws, exactly as
  // it did before W1 -- the escape hatch is the layout string, not the mode.
  MLAAttentionSpec no_marker(/*block_size=*/256, /*head_size=*/512, DType::kI8,
                             /*num_kv_heads=*/1, KVQuantMode::kFp8PerTensor);
  CHECK_THROWS_AS(no_marker.real_page_size_bytes(), std::runtime_error);
}

TEST_CASE("W1 V3.2 main MLA: 656 bytes per token off block_size, not storage") {
  // kv_cache_interface.py:402-405 -- "V3.2 main MLA: 656-byte custom layout
  // (kv_lora_rank=512 + qk_rope_head_dim=64, head_size=576)". Selected by
  // cache_dtype_str == "fp8_ds_mla" with model_version NOT "deepseek_v4", and
  // it multiplies BLOCK_SIZE, not storage_block_size.
  MLAAttentionSpec v32(/*block_size=*/64, /*head_size=*/576, DType::kI8,
                       /*num_kv_heads=*/1, KVQuantMode::kFp8PerTensor,
                       /*page_size_padded=*/std::nullopt,
                       /*indexes_kv_by_block_stride=*/false,
                       /*cache_dtype_str=*/std::string("fp8_ds_mla"));
  CHECK(v32.compress_ratio == 1);
  CHECK(v32.real_page_size_bytes() == 64LL * 656);
  CHECK(v32.real_page_size_bytes() == 41984);

  // The 656 branch reads block_size even when a compress_ratio is set, so a
  // ratio must NOT shrink it (kv_cache_interface.py:404-405 has no
  // storage_block_size in it).
  MLAAttentionSpec v32_ratio(/*block_size=*/64, /*head_size=*/576, DType::kI8,
                            /*num_kv_heads=*/1, KVQuantMode::kFp8PerTensor,
                            /*page_size_padded=*/std::nullopt,
                            /*indexes_kv_by_block_stride=*/false,
                            /*cache_dtype_str=*/std::string("fp8_ds_mla"),
                            /*alignment=*/std::nullopt, /*compress_ratio=*/4);
  CHECK(v32_ratio.real_page_size_bytes() == 64LL * 656);
}

TEST_CASE("W1 SlidingWindowMLASpec refuses an unknown model_version") {
  // kv_cache_interface.py:634-636 asserts model_version in (None,
  // "deepseek_v4") before the element formula. With no alignment, upstream's
  // __post_init__ never queries the page, so the refusal lands on the first
  // page query and not on construction.
  SlidingWindowMLASpec spec(
      /*block_size=*/8, /*num_kv_heads=*/1, /*head_size=*/1024, DType::kF32,
      /*sliding_window=*/128, /*cache_dtype_str=*/std::nullopt,
      /*alignment=*/std::nullopt, /*compress_ratio=*/1,
      /*model_version=*/std::string("deepseek_v9"));
  CHECK_THROWS(static_cast<void>(spec.real_page_size_bytes()));

  // WITH an alignment the padding helper queries the page from the constructor
  // body, so the same refusal lands at construction — upstream's
  // __post_init__ -> _apply_alignment_padding -> real_page_size_bytes chain.
  CHECK_THROWS(SlidingWindowMLASpec(
      /*block_size=*/8, /*num_kv_heads=*/1, /*head_size=*/1024, DType::kF32,
      /*sliding_window=*/128, /*cache_dtype_str=*/std::nullopt,
      /*alignment=*/576, /*compress_ratio=*/1,
      /*model_version=*/std::string("deepseek_v9")));
}

TEST_CASE("W1 SlidingWindowMLASpec page formula is NOT the parent's K+V one") {
  // SlidingWindowSpec sums head_size + head_size_v (:552-558); the MLA subclass
  // holds ONE vector instead of K + V (compressor.py:194) and multiplies
  // head_size alone (:637-642). Same dims through the parent must therefore
  // DOUBLE the page.
  SlidingWindowMLASpec mla(/*block_size=*/8, /*num_kv_heads=*/1,
                           /*head_size=*/1024, DType::kF32,
                           /*sliding_window=*/128);
  SlidingWindowSpec parent(/*block_size=*/8, /*num_kv_heads=*/1,
                           /*head_size=*/1024, DType::kF32,
                           /*sliding_window=*/128);
  CHECK(parent.real_page_size_bytes() == 2 * mla.real_page_size_bytes());
}

TEST_CASE("W1 alignment padding: only when it changes the page") {
  // _apply_alignment_padding (:345-351): no alignment => no write; an aligned
  // page => no write; an unaligned page => round_up.
  MLAAttentionSpec unaligned(/*block_size=*/256, /*head_size=*/512, DType::kI8,
                             /*num_kv_heads=*/1, KVQuantMode::kFp8PerTensor,
                             /*page_size_padded=*/std::nullopt,
                             /*indexes_kv_by_block_stride=*/false,
                             /*cache_dtype_str=*/std::string("fp8_ds_mla"),
                             /*alignment=*/std::nullopt, /*compress_ratio=*/4,
                             /*model_version=*/std::string("deepseek_v4"));
  CHECK_FALSE(unaligned.page_size_padded.has_value());
  CHECK(unaligned.page_size_bytes() == 37376);  // the RAW page, not 37440

  // 576 * 65 = 37440 is already aligned to 576, so a spec whose real page IS
  // 37440 gets no write.
  SlidingWindowMLASpec exact(/*block_size=*/64, /*num_kv_heads=*/1,
                             /*head_size=*/585, DType::kI8,
                             /*sliding_window=*/128,
                             /*cache_dtype_str=*/std::nullopt,
                             /*alignment=*/8);
  CHECK(exact.real_page_size_bytes() == 64LL * 585);  // 37440
  CHECK_FALSE(exact.page_size_padded.has_value());
}

TEST_CASE("W1 refuses a compress_ratio or alignment that cannot divide") {
  // compress_ratio 0 is a ZeroDivisionError upstream; the model guards it with
  // max(1, config.compress_ratios[layer_id]) (attention.py:205-212). Refuse it
  // rather than divide by zero.
  CHECK_THROWS(MLAAttentionSpec(/*block_size=*/256, /*head_size=*/512,
                                DType::kI8, /*num_kv_heads=*/1,
                                KVQuantMode::kNone,
                                /*page_size_padded=*/std::nullopt,
                                /*indexes_kv_by_block_stride=*/false,
                                /*cache_dtype_str=*/std::nullopt,
                                /*alignment=*/std::nullopt,
                                /*compress_ratio=*/0));
  CHECK_THROWS(SlidingWindowMLASpec(
      /*block_size=*/8, /*num_kv_heads=*/1, /*head_size=*/1024, DType::kF32,
      /*sliding_window=*/128, /*cache_dtype_str=*/std::nullopt,
      /*alignment=*/0));
}

// ---------------------------------------------------------------------------
// Registry. Upstream registers SlidingWindowMLASpec against SlidingWindowManager
// with uniform_type_base_spec=SlidingWindowMLASpec -- ITSELF, not
// SlidingWindowSpec (single_type_kv_cache_manager.py:1834-1838), unlike
// MLAAttentionSpec which registers against FullAttentionSpec (:1824-1827).
// ---------------------------------------------------------------------------
TEST_CASE("W1 SlidingWindowMLASpec registers as its OWN uniform base") {
  SlidingWindowMLASpec spec(/*block_size=*/64, /*num_kv_heads=*/1,
                            /*head_size=*/512, DType::kI8,
                            /*sliding_window=*/128);
  CHECK(KVCacheSpecRegistry::get_manager_kind(spec) ==
        KVCacheManagerKind::kSlidingWindow);
  CHECK(KVCacheSpecRegistry::get_uniform_type_base_spec(spec) ==
        std::optional<std::type_index>{typeid(SlidingWindowMLASpec)});

  // ...so it is NOT uniform with a plain SlidingWindowSpec, which registers
  // against SlidingWindowSpec.
  SlidingWindowSpec plain(/*block_size=*/64, /*num_kv_heads=*/1,
                          /*head_size=*/512, DType::kI8,
                          /*sliding_window=*/128);
  CHECK_FALSE(are_uniform_kv_cache_specs({&spec, &plain}));
}

TEST_CASE("W1 SlidingWindowMLASpec uniformity keys on the window") {
  // is_uniform_with_collection (:679-686) requires the same sliding_window.
  SlidingWindowMLASpec a(/*block_size=*/64, /*num_kv_heads=*/1,
                         /*head_size=*/512, DType::kI8,
                         /*sliding_window=*/128);
  SlidingWindowMLASpec b(/*block_size=*/64, /*num_kv_heads=*/1,
                         /*head_size=*/512, DType::kI8,
                         /*sliding_window=*/128);
  SlidingWindowMLASpec other_window(/*block_size=*/64, /*num_kv_heads=*/1,
                                    /*head_size=*/512, DType::kI8,
                                    /*sliding_window=*/256);
  CHECK(are_uniform_kv_cache_specs({&a, &b}));
  CHECK_FALSE(are_uniform_kv_cache_specs({&a, &other_window}));
}

TEST_CASE("W1 KVBytesPerBlock reads a SlidingWindowMLASpec group's PADDED page") {
  // The allocator divides a byte budget by this, so it must see the padded page
  // and not the raw one.
  auto swa = std::make_shared<SlidingWindowMLASpec>(
      /*block_size=*/64, /*num_kv_heads=*/1, /*head_size=*/512, DType::kI8,
      /*sliding_window=*/128, /*cache_dtype_str=*/std::string("fp8_ds_mla"),
      /*alignment=*/576, /*compress_ratio=*/1,
      /*model_version=*/std::string("deepseek_v4"), KVQuantMode::kFp8PerTensor);
  KVCacheConfig config{
      /*num_blocks=*/10,
      /*kv_cache_tensors=*/{},
      /*kv_cache_groups=*/{KVCacheGroupSpec{{"l0.swa_cache", "l1.swa_cache"},
                                            swa}}};
  CHECK(KVBytesPerBlock(config) == 2 * 37440);
}
