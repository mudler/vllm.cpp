// Tests for the KV-cache COORDINATOR (M1.3 Task 3) — UnitaryKVCacheCoordinator
// (single group), HybridKVCacheCoordinator (the gate models: full-attn + GDN
// mamba group), and the get_kv_cache_coordinator factory.
//
// Ported from vllm/tests/v1/core/test_prefix_caching.py @ e24d1b24:
//   - the "full" + "mamba" 2-group hybrid config (_make_hybrid_kv_cache_config,
//     _HYBRID_MODEL_TEST_CASES "2g-full+mamba") drives the hybrid cases below.
//   - test_hybrid_cache_mamba_align_shared_prefix_detection ->
//     "cross-group find_longest_cache_hit ..." + "num_uncached_common_prefix
//     _tokens": the coordinator reports the intersection (min) hit and the
//     uncached-common-prefix delta when full attention cached a longer prefix
//     than the mamba group. Ported at the COORDINATOR level (Task 4's
//     KVCacheManager / Scheduler are not landed yet), constructing the per-group
//     cache-hit state directly against the owned BlockPool — the same
//     mock-the-block-pool pattern used by the Task 2 manager tests.
//   - test_eagle_with_sliding_window and pure-SWA retention/replay cases at
//     tests/v1/core/test_prefix_caching.py:2520-2584,3811-3909.
//
// Object-identity comparisons (`block is ...`) are ported as pointer equality.
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_coordinator.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::v1::BlockHash;
using vllm::v1::ChunkedLocalAttentionManager;
using vllm::v1::ChunkedLocalAttentionSpec;
using vllm::v1::get_kv_cache_coordinator;
using vllm::v1::get_request_block_hasher;
using vllm::v1::HybridKVCacheCoordinator;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlock;
using vllm::v1::KVCacheConfig;
using vllm::v1::KVCacheCoordinator;
using vllm::v1::KVCacheCoordinatorNoPrefixCache;
using vllm::v1::KVCacheGroupSpec;
using vllm::v1::KVCacheSpec;
using vllm::v1::FullAttentionSpec;
using vllm::v1::make_block_hash_with_group_id;
using vllm::v1::MambaSpec;
using vllm::v1::Request;
using vllm::v1::sha256_cbor;
using vllm::v1::SlidingWindowManager;
using vllm::v1::SlidingWindowSpec;
using vllm::v1::UnitaryKVCacheCoordinator;
using vt::DType;

namespace {

constexpr int kBlockSize = 2;

std::shared_ptr<FullAttentionSpec> MakeFullSpec() {
  return std::make_shared<FullAttentionSpec>(kBlockSize, /*num_kv_heads=*/1,
                                             /*head_size=*/1, DType::kF32);
}

std::shared_ptr<MambaSpec> MakeMambaSpec() {
  return std::make_shared<MambaSpec>(
      kBlockSize, std::vector<std::vector<int64_t>>{{1, 1}},
      std::vector<DType>{DType::kF32});
}

std::shared_ptr<SlidingWindowSpec> MakeSlidingSpec(
    int sliding_window = 4) {
  return std::make_shared<SlidingWindowSpec>(
      kBlockSize, /*num_kv_heads=*/1, /*head_size=*/1, DType::kF32,
      sliding_window);
}

std::shared_ptr<ChunkedLocalAttentionSpec> MakeChunkedLocalSpec(
    int attention_chunk_size = 4) {
  return std::make_shared<ChunkedLocalAttentionSpec>(
      kBlockSize, /*num_kv_heads=*/1, /*head_size=*/1, DType::kF32,
      attention_chunk_size);
}

// The 2-group gate-model config: group 0 = full attention, group 1 = GDN mamba.
KVCacheConfig MakeHybridConfig(int num_blocks = 100) {
  KVCacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"full0"},
                                   MakeFullSpec());
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"mamba0"},
                                   MakeMambaSpec());
  return cfg;
}

KVCacheConfig MakeUnitaryConfig(int num_blocks = 100) {
  KVCacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"full0"},
                                   MakeFullSpec());
  return cfg;
}

KVCacheConfig MakeSlidingConfig(int num_blocks = 100,
                                int sliding_window = 4) {
  KVCacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"swa0"},
                                   MakeSlidingSpec(sliding_window));
  return cfg;
}

KVCacheConfig MakeChunkedLocalConfig(int num_blocks = 100,
                                     int attention_chunk_size = 4) {
  KVCacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"chunked0"},
      MakeChunkedLocalSpec(attention_chunk_size));
  return cfg;
}

std::unique_ptr<KVCacheCoordinator> MakeCoordinator(KVCacheConfig cfg) {
  return get_kv_cache_coordinator(
      std::move(cfg), /*max_model_len=*/8192, /*max_num_batched_tokens=*/8192,
      /*use_eagle=*/false, /*enable_caching=*/true,
      /*enable_kv_cache_events=*/false, /*dcp_world_size=*/1,
      /*pcp_world_size=*/1, /*scheduler_block_size=*/kBlockSize,
      /*hash_block_size=*/kBlockSize);
}

// Cache `blk` for hash `bh` under a specific kv cache group (mirrors the Task 2
// manager tests' direct cached_block_hash_to_block insertion).
void MockCache(vllm::v1::BlockPool& pool, const BlockHash& bh, int group_id,
               KVCacheBlock* blk) {
  pool.cached_block_hash_to_block[make_block_hash_with_group_id(bh, group_id)]
                                 [blk->block_id] = blk;
}

Request MakeRequest(const std::string& id, int n_tokens) {
  std::vector<int32_t> tokens;
  for (int i = 0; i < n_tokens; ++i) tokens.push_back(i);
  return Request(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
                 get_request_block_hasher(kBlockSize, sha256_cbor));
}

}  // namespace

// ---------------------------------------------------------------------------
// Factory + construction
// ---------------------------------------------------------------------------

TEST_CASE("get_kv_cache_coordinator: 1 group -> Unitary, 2 groups -> Hybrid") {
  auto uni = MakeCoordinator(MakeUnitaryConfig());
  CHECK(dynamic_cast<UnitaryKVCacheCoordinator*>(uni.get()) != nullptr);
  CHECK(uni->single_type_managers.size() == 1);

  auto hyb = MakeCoordinator(MakeHybridConfig());
  CHECK(dynamic_cast<HybridKVCacheCoordinator*>(hyb.get()) != nullptr);
  CHECK(hyb->single_type_managers.size() == 2);
}

TEST_CASE("get_kv_cache_coordinator: SlidingWindowSpec uses registry manager and cap") {
  auto coord = get_kv_cache_coordinator(
      MakeSlidingConfig(), /*max_model_len=*/100,
      /*max_num_batched_tokens=*/4, /*use_eagle=*/false,
      /*enable_caching=*/true, /*enable_kv_cache_events=*/false,
      /*dcp_world_size=*/1, /*pcp_world_size=*/1,
      /*scheduler_block_size=*/kBlockSize,
      /*hash_block_size=*/kBlockSize);
  REQUIRE(dynamic_cast<UnitaryKVCacheCoordinator*>(coord.get()) != nullptr);
  REQUIRE(coord->single_type_managers.size() == 1);
  auto* sliding = dynamic_cast<SlidingWindowManager*>(
      coord->single_type_managers[0].get());
  REQUIRE(sliding != nullptr);
  // min(W-1 + max_batch, max_model) = 7; cdiv(7, 2) + 1 = 5.
  CHECK(sliding->_max_admission_blocks_per_request == std::optional<int>{5});

  vllm::v1::KVCacheBlocksTuple no_hits(1);
  CHECK(coord->get_num_blocks_to_allocate(
            "admit", /*num_tokens=*/100, no_hits,
            /*num_encoder_tokens=*/0, /*total_computed_tokens=*/0,
            /*num_tokens_main_model=*/100,
            /*apply_admission_cap=*/true) == 5);
}

TEST_CASE(
    "get_kv_cache_coordinator: ChunkedLocalAttentionSpec uses registry manager and cap") {
  auto coord = get_kv_cache_coordinator(
      MakeChunkedLocalConfig(), /*max_model_len=*/100,
      /*max_num_batched_tokens=*/3, /*use_eagle=*/false,
      /*enable_caching=*/true, /*enable_kv_cache_events=*/false,
      /*dcp_world_size=*/1, /*pcp_world_size=*/1,
      /*scheduler_block_size=*/kBlockSize,
      /*hash_block_size=*/kBlockSize);
  REQUIRE(dynamic_cast<UnitaryKVCacheCoordinator*>(coord.get()) != nullptr);
  REQUIRE(coord->single_type_managers.size() == 1);
  auto* chunked = dynamic_cast<ChunkedLocalAttentionManager*>(
      coord->single_type_managers[0].get());
  REQUIRE(chunked != nullptr);
  // min(chunk + max_batch, max_model) = 7; cdiv(7, 2) = 4.
  CHECK(chunked->_max_admission_blocks_per_request == std::optional<int>{4});

  vllm::v1::KVCacheBlocksTuple no_hits(1);
  CHECK(coord->get_num_blocks_to_allocate(
            "admit", /*num_tokens=*/100, no_hits,
            /*num_encoder_tokens=*/0, /*total_computed_tokens=*/0,
            /*num_tokens_main_model=*/100,
            /*apply_admission_cap=*/true) == 4);
}

TEST_CASE("KVCacheCoordinatorNoPrefixCache returns no hits for hybrid groups") {
  auto coord = get_kv_cache_coordinator(
      MakeHybridConfig(), 8192, 8192, false, /*enable_caching=*/false, false, 1,
      1, kBlockSize, kBlockSize);
  REQUIRE(dynamic_cast<KVCacheCoordinatorNoPrefixCache*>(coord.get()) !=
          nullptr);
  CHECK_FALSE(coord->enable_caching);

  std::vector<BlockHash> hashes = {"h0", "h1"};
  auto [hits, hit_length] =
      coord->find_longest_cache_hit(hashes, /*max_cache_hit_length=*/4);
  CHECK(hit_length == 0);
  REQUIRE(hits.size() == 2);
  CHECK(hits[0].empty());
  CHECK(hits[1].empty());
  CHECK(coord->get_num_common_prefix_blocks("running") ==
        std::vector<int>{0, 0});

  auto allocated = coord->allocate_new_blocks("running", 4, 4);
  REQUIRE(allocated.size() == 2);
  CHECK_FALSE(allocated[0].empty());
  CHECK_FALSE(allocated[1].empty());
  coord->free("running");
}

TEST_CASE("HybridKVCacheCoordinator: attention_groups puts full attention first") {
  // Config order is [full, mamba]; a config given as [mamba, full] must still
  // sort full first (upstream attention_groups.sort key).
  KVCacheConfig cfg;
  cfg.num_blocks = 100;
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"mamba0"},
                                   MakeMambaSpec());
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"full0"},
                                   MakeFullSpec());
  auto coord = MakeCoordinator(std::move(cfg));
  auto* hyb = dynamic_cast<HybridKVCacheCoordinator*>(coord.get());
  REQUIRE(hyb != nullptr);
  REQUIRE(hyb->attention_groups.size() == 2);
  CHECK(hyb->attention_groups[0].spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
  // The mamba group keeps its config-order id (1 here / 0 in this config).
  CHECK(hyb->attention_groups[0].group_ids == std::vector<int>{1});
  CHECK(hyb->attention_groups[1].group_ids == std::vector<int>{0});
}

TEST_CASE("HybridKVCacheCoordinator: equal sliding specs share one SpecGroup") {
  KVCacheConfig cfg;
  cfg.num_blocks = 100;
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"swa0"},
                                   MakeSlidingSpec());
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"full0"},
                                   MakeFullSpec());
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"swa1"},
                                   MakeSlidingSpec());
  auto coord = MakeCoordinator(std::move(cfg));
  auto* hyb = dynamic_cast<HybridKVCacheCoordinator*>(coord.get());
  REQUIRE(hyb != nullptr);
  REQUIRE(hyb->attention_groups.size() == 2);
  CHECK(hyb->attention_groups[0].spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
  CHECK(hyb->attention_groups[1].spec->kind() ==
        vllm::v1::KVCacheSpecKind::kSlidingWindow);
  CHECK(hyb->attention_groups[1].group_ids == std::vector<int>{0, 2});
}

TEST_CASE("HybridKVCacheCoordinator: equal chunked-local specs share one SpecGroup") {
  KVCacheConfig cfg;
  cfg.num_blocks = 100;
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"chunked0"},
                                   MakeChunkedLocalSpec());
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"full0"},
                                   MakeFullSpec());
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"chunked1"},
                                   MakeChunkedLocalSpec());
  auto coord = MakeCoordinator(std::move(cfg));
  auto* hyb = dynamic_cast<HybridKVCacheCoordinator*>(coord.get());
  REQUIRE(hyb != nullptr);
  REQUIRE(hyb->attention_groups.size() == 2);
  CHECK(hyb->attention_groups[0].spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
  CHECK(hyb->attention_groups[1].spec->kind() ==
        vllm::v1::KVCacheSpecKind::kChunkedLocalAttention);
  CHECK(hyb->attention_groups[1].group_ids == std::vector<int>{0, 2});
}

// ---------------------------------------------------------------------------
// Allocation across groups
// ---------------------------------------------------------------------------

TEST_CASE(
    "HybridKVCacheCoordinator: allocate_new_blocks across groups reduces pool "
    "free by the sum") {
  auto coord = MakeCoordinator(MakeHybridConfig());
  int free_before = coord->block_pool.get_num_free_blocks();

  auto blocks = coord->allocate_new_blocks("r1", /*num_tokens=*/6,
                                           /*num_tokens_main_model=*/6);
  REQUIRE(blocks.size() == 2);
  CHECK(blocks[0].size() == 3);  // full attention: cdiv(6, 2)
  CHECK(blocks[1].size() == 3);  // mamba (mode none): cdiv(6, 2)
  // 3 + 3 = 6 physical blocks pulled from the shared pool.
  CHECK(coord->block_pool.get_num_free_blocks() == free_before - 6);
}

TEST_CASE(
    "HybridKVCacheCoordinator: get_num_blocks_to_allocate sums across groups") {
  auto coord = MakeCoordinator(MakeHybridConfig());
  vllm::v1::KVCacheBlocksTuple no_computed = {{}, {}};
  int n = coord->get_num_blocks_to_allocate(
      "r1", /*num_tokens=*/6, no_computed, /*num_encoder_tokens=*/0,
      /*total_computed_tokens=*/0, /*num_tokens_main_model=*/6);
  // full cdiv(6,2)=3 + mamba cdiv(6,2)=3.
  CHECK(n == 6);
}

TEST_CASE("HybridKVCacheCoordinator: free across groups returns every block") {
  auto coord = MakeCoordinator(MakeHybridConfig());
  int free_before = coord->block_pool.get_num_free_blocks();
  coord->allocate_new_blocks("r1", 6, 6);
  CHECK(coord->block_pool.get_num_free_blocks() == free_before - 6);

  coord->free("r1");
  CHECK(coord->block_pool.get_num_free_blocks() == free_before);
  for (const auto& mgr : coord->single_type_managers) {
    CHECK(mgr->req_to_blocks.count("r1") == 0);
  }
}

TEST_CASE(
    "HybridKVCacheCoordinator: get_num_common_prefix_blocks per group (mamba is "
    "always 0)") {
  auto coord = MakeCoordinator(MakeHybridConfig());
  coord->allocate_new_blocks("r1", 6, 6);  // 3 full + 3 mamba, ref_cnt 1 each.

  auto common = coord->get_num_common_prefix_blocks("r1");
  REQUIRE(common.size() == 2);
  CHECK(common[0] == 3);  // full attention: all 3 blocks common (1 request).
  CHECK(common[1] == 0);  // mamba: cascade attention unsupported -> always 0.
}

// ---------------------------------------------------------------------------
// UnitaryKVCacheCoordinator
// ---------------------------------------------------------------------------

TEST_CASE(
    "UnitaryKVCacheCoordinator: find_longest_cache_hit delegates to the single "
    "group") {
  auto coord = MakeCoordinator(MakeUnitaryConfig());
  auto& pool = coord->block_pool;
  MockCache(pool, "h0", 0, &pool.blocks[10]);
  MockCache(pool, "h1", 0, &pool.blocks[11]);
  // h2 not cached -> prefix stops at 2 blocks.

  std::vector<BlockHash> bh = {"h0", "h1", "h2"};
  auto [blocks, hit_length] =
      coord->find_longest_cache_hit(bh, /*max_cache_hit_length=*/6);
  REQUIRE(blocks.size() == 1);
  CHECK(blocks[0].size() == 2);
  CHECK(blocks[0][0] == &pool.blocks[10]);
  CHECK(blocks[0][1] == &pool.blocks[11]);
  CHECK(hit_length == 2 * kBlockSize);  // 2 blocks * block_size
}

TEST_CASE(
    "UnitaryKVCacheCoordinator: sliding-window cache replay keeps the live tail") {
  init_none_hash(sha256_cbor);
  auto coord = MakeCoordinator(MakeSlidingConfig());

  Request original = MakeRequest("swa-a", /*n_tokens=*/6);
  auto allocated = coord->allocate_new_blocks("swa-a", 6, 6);
  REQUIRE(allocated.size() == 1);
  REQUIRE(allocated[0].size() == 3);
  coord->cache_blocks(original, /*num_computed_tokens=*/6);
  coord->free("swa-a");

  Request replay = MakeRequest("swa-b", /*n_tokens=*/8);
  auto [blocks, hit_length] =
      coord->find_longest_cache_hit(replay.block_hashes, 8);
  REQUIRE(blocks.size() == 1);
  REQUIRE(blocks[0].size() == 3);
  CHECK(blocks[0][0] == coord->block_pool.null_block);
  CHECK(blocks[0][1]->block_id == allocated[0][1]->block_id);
  CHECK(blocks[0][2]->block_id == allocated[0][2]->block_id);
  CHECK(hit_length == 3 * kBlockSize);
}

TEST_CASE(
    "UnitaryKVCacheCoordinator: chunked-local replay nulls old chunks and reuses current") {
  init_none_hash(sha256_cbor);
  auto coord = MakeCoordinator(MakeChunkedLocalConfig());

  Request original = MakeRequest("chunk-a", /*n_tokens=*/6);
  auto allocated = coord->allocate_new_blocks("chunk-a", 6, 6);
  REQUIRE(allocated.size() == 1);
  REQUIRE(allocated[0].size() == 3);
  coord->cache_blocks(original, /*num_computed_tokens=*/6);
  coord->free("chunk-a");

  Request replay = MakeRequest("chunk-b", /*n_tokens=*/8);
  auto [blocks, hit_length] =
      coord->find_longest_cache_hit(replay.block_hashes,
                                    /*max_cache_hit_length=*/6);
  REQUIRE(blocks.size() == 1);
  REQUIRE(blocks[0].size() == 3);
  CHECK(blocks[0][0] == coord->block_pool.null_block);
  CHECK(blocks[0][1] == coord->block_pool.null_block);
  CHECK(blocks[0][2]->block_id == allocated[0][2]->block_id);
  CHECK(hit_length == 3 * kBlockSize);
}

// ---------------------------------------------------------------------------
// HybridKVCacheCoordinator: THE CROSS-GROUP find_longest_cache_hit (the core)
// ---------------------------------------------------------------------------

TEST_CASE(
    "HybridKVCacheCoordinator: cross-group find_longest_cache_hit returns the "
    "intersection (min) of the per-group hits, truncating full attention") {
  auto coord = MakeCoordinator(MakeHybridConfig());
  auto* hyb = dynamic_cast<HybridKVCacheCoordinator*>(coord.get());
  REQUIRE(hyb != nullptr);
  auto& pool = coord->block_pool;

  std::vector<BlockHash> bh = {"h0", "h1", "h2", "h3"};

  SUBCASE("full attention cached a LONGER prefix than mamba -> hit = mamba's") {
    // Full attention (group 0) cached all 4 blocks.
    MockCache(pool, "h0", 0, &pool.blocks[10]);
    MockCache(pool, "h1", 0, &pool.blocks[11]);
    MockCache(pool, "h2", 0, &pool.blocks[12]);
    MockCache(pool, "h3", 0, &pool.blocks[13]);
    // Mamba (group 1) only cached a recurrent state at block index 1 (h1);
    // h2/h3 are NOT cached in the mamba group.
    MockCache(pool, "h1", 1, &pool.blocks[21]);

    auto [blocks, hit_length] =
        coord->find_longest_cache_hit(bh, /*max_cache_hit_length=*/8);

    // The coordinated hit is the MIN: mamba's 2-block (index 1) state hit.
    CHECK(hit_length == 2 * kBlockSize);
    REQUIRE(blocks.size() == 2);
    // Full attention truncated from 4 -> 2 blocks to match the intersection.
    REQUIRE(blocks[0].size() == 2);
    CHECK(blocks[0][0] == &pool.blocks[10]);
    CHECK(blocks[0][1] == &pool.blocks[11]);
    // Mamba: [null, state] — single tail recurrent state.
    REQUIRE(blocks[1].size() == 2);
    CHECK(blocks[1][0] == pool.null_block);
    CHECK(blocks[1][1] == &pool.blocks[21]);
    // Full attention cached a longer (4-block) prefix than the coordinated
    // 2-block hit -> an uncached common prefix across requests.
    CHECK(hyb->num_uncached_common_prefix_tokens == 2 * kBlockSize);
  }

  SUBCASE("both groups agree on a 3-block prefix -> full 3-block hit") {
    MockCache(pool, "h0", 0, &pool.blocks[10]);
    MockCache(pool, "h1", 0, &pool.blocks[11]);
    MockCache(pool, "h2", 0, &pool.blocks[12]);
    // Mamba cached its state at block index 2 (h2) -> hit length 3 blocks.
    MockCache(pool, "h2", 1, &pool.blocks[22]);

    auto [blocks, hit_length] =
        coord->find_longest_cache_hit(bh, /*max_cache_hit_length=*/6);
    CHECK(hit_length == 3 * kBlockSize);
    REQUIRE(blocks[0].size() == 3);
    REQUIRE(blocks[1].size() == 3);
    CHECK(blocks[1][2] == &pool.blocks[22]);
    CHECK(hyb->num_uncached_common_prefix_tokens == 0);
  }

  SUBCASE("no mamba hit -> coordinated hit is empty") {
    MockCache(pool, "h0", 0, &pool.blocks[10]);
    MockCache(pool, "h1", 0, &pool.blocks[11]);
    // Mamba group has nothing cached.
    auto [blocks, hit_length] =
        coord->find_longest_cache_hit(bh, /*max_cache_hit_length=*/8);
    CHECK(hit_length == 0);
    REQUIRE(blocks.size() == 2);
    CHECK(blocks[0].empty());  // full attention truncated to 0 blocks.
    CHECK(blocks[1].empty());
    // Full attention still cached a 2-block prefix -> uncached common prefix.
    CHECK(hyb->num_uncached_common_prefix_tokens == 2 * kBlockSize);
  }
}

// ---------------------------------------------------------------------------
// HybridKVCacheCoordinator: end-to-end cache_blocks -> find_longest_cache_hit
// ---------------------------------------------------------------------------

TEST_CASE(
    "HybridKVCacheCoordinator: cache_blocks then find_longest_cache_hit returns "
    "the shared prefix cached in BOTH groups (end-to-end)") {
  init_none_hash(sha256_cbor);
  auto coord = MakeCoordinator(MakeHybridConfig());

  // Request A: 6 prompt tokens -> 3 full blocks. Allocate across both groups and
  // cache the full prefix.
  Request reqA = MakeRequest("A", 6);
  REQUIRE(reqA.block_hashes.size() == 3);
  auto allocatedA = coord->allocate_new_blocks("A", 6, 6);
  REQUIRE(allocatedA[0].size() == 3);  // full attention
  REQUIRE(allocatedA[1].size() == 3);  // mamba
  coord->cache_blocks(reqA, /*num_computed_tokens=*/6);
  coord->new_step_starts();  // A's mamba state becomes reusable next step.

  // Request B shares A's 6-token prefix.
  Request reqB = MakeRequest("B", 8);
  auto [blocks, hit_length] =
      coord->find_longest_cache_hit(reqB.block_hashes, /*max_cache_hit_length=*/8);

  // Both groups cached the full 3-block prefix -> a coordinated 3-block hit.
  CHECK(hit_length == 3 * kBlockSize);
  REQUIRE(blocks.size() == 2);
  // Full attention returns the 3 cached blocks.
  REQUIRE(blocks[0].size() == 3);
  CHECK(blocks[0][0]->block_id == allocatedA[0][0]->block_id);
  CHECK(blocks[0][2]->block_id == allocatedA[0][2]->block_id);
  // Mamba returns [null, null, state] — only the single tail recurrent state.
  REQUIRE(blocks[1].size() == 3);
  CHECK(blocks[1][0] == coord->block_pool.null_block);
  CHECK(blocks[1][1] == coord->block_pool.null_block);
  CHECK(blocks[1][2]->block_id == allocatedA[1][2]->block_id);
}

TEST_CASE("SWA model-positive replay waits on attention and model rows" *
          doctest::skip(true) *
          doctest::description(
              "tests/v1/e2e/general/test_correctness_sliding_window.py needs "
              "ATTN-SLIDING-WINDOW and MODEL-TEXT-gemma3-gemma3-for-causal-lm")) {
  MESSAGE("SKIP ATTN-SLIDING-WINDOW + MODEL-TEXT-gemma3-gemma3-for-causal-lm");
}

TEST_CASE("chunked-local model-positive replay waits on W4 and Llama4" *
          doctest::skip(true) *
          doctest::description(
              "chunked-local e2e needs ATTN-CHUNKED-LOCAL and "
              "MODEL-TEXT-llama4-llama4-for-causal-lm")) {
  MESSAGE("SKIP ATTN-CHUNKED-LOCAL + MODEL-TEXT-llama4-llama4-for-causal-lm");
}
