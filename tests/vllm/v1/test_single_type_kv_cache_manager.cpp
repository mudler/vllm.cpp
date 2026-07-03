// Tests for the per-group KV-cache managers (M1.3 Task 2) — FullAttentionManager
// (standard paged K+V) and MambaManager (the GDN single-state manager).
//
// Ported from vllm/tests/v1/core/test_single_type_kv_cache_manager.py @ e24d1b24:
//   - test_get_num_blocks_to_allocate            -> "get_num_blocks_to_allocate
//       accounting" (adapted to FullAttentionManager: SlidingWindow is DEFERRED,
//       but full attention shares the base accounting and its
//       get_num_skipped_tokens is 0, so the 20 / 15 oracles carry over verbatim).
//   - test_evictable_cached_blocks_not_double_allocated -> "evictable cached
//       blocks are not double allocated" (same adaptation; base-class path).
// The upstream file only exercises the DEFERRED SWA/Chunked/RSWA managers for
// find_longest_cache_hit / remove_skipped_blocks; the FullAttention and Mamba
// find_longest_cache_hit / lifecycle behaviors below follow the upstream
// mock-the-block-pool pattern (block_pool.cached_block_hash_to_block insertion)
// applied to the T0 managers, asserting the upstream method semantics.
//
// Object-identity comparisons (`block is ...`) are ported as pointer equality.
#include <doctest/doctest.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/single_type_kv_cache_manager.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::v1::BlockHash;
using vllm::v1::BlockPool;
using vllm::v1::FullAttentionManager;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlock;
using vllm::v1::KVCacheSpec;
using vllm::v1::make_block_hash_with_group_id;
using vllm::v1::MambaManager;
using vllm::v1::MambaSpec;
using vllm::v1::Request;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

// block_size=2, num_kv_heads=1, head_size=1, float32 (upstream test dims).
std::shared_ptr<FullAttentionSpec> MakeFullSpec(int block_size = 2) {
  return std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                             /*head_size=*/1, DType::kF32);
}

std::shared_ptr<MambaSpec> MakeMambaSpec(int block_size = 2,
                                         const std::string& mode = "none",
                                         int num_speculative_blocks = 0) {
  return std::make_shared<MambaSpec>(
      block_size, std::vector<std::vector<int64_t>>{{2, 4}},
      std::vector<DType>{DType::kF32}, /*page_size_padded=*/std::nullopt, mode,
      num_speculative_blocks);
}

// Insert a cached block for `bh` in group 0 (mirrors upstream's direct
// cached_block_hash_to_block.insert used by the manager tests).
void MockCache(BlockPool& pool, const BlockHash& bh, KVCacheBlock* blk) {
  pool.cached_block_hash_to_block[make_block_hash_with_group_id(bh, 0)]
                                 [blk->block_id] = blk;
}

Request MakeRequest(const std::string& id, const std::vector<int32_t>& tokens,
                    int block_size) {
  return Request(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
                 get_request_block_hasher(block_size, sha256_cbor));
}

std::vector<int32_t> Iota(int n) {
  std::vector<int32_t> v;
  for (int i = 0; i < n; ++i) v.push_back(i);
  return v;
}

// Pointer-stable store for standalone KVCacheBlock objects (upstream's
// KVCacheBlock(id) test fixtures).
struct BlockStore {
  std::deque<KVCacheBlock> storage;
  KVCacheBlock* make(int id) {
    storage.emplace_back(id);
    return &storage.back();
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// FullAttentionManager
// ---------------------------------------------------------------------------

TEST_CASE(
    "FullAttentionManager: allocate_new_blocks reduces pool free count and "
    "records req_to_blocks") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, /*enable_caching=*/true,
                           /*kv_cache_group_id=*/0, /*scheduler_block_size=*/2);

  CHECK(pool.get_num_free_blocks() == 99);
  auto blocks = mgr.allocate_new_blocks("r1", /*num_tokens=*/6,
                                        /*num_tokens_main_model=*/6);
  CHECK(blocks.size() == 3);  // cdiv(6, 2)
  CHECK(mgr.req_to_blocks["r1"].size() == 3);
  CHECK(pool.get_num_free_blocks() == 96);

  // Full attention records new block ids for the worker's block-table update.
  auto ids = mgr.take_new_block_ids();
  CHECK(ids.size() == 3);
  CHECK(mgr.take_new_block_ids().empty());
}

TEST_CASE("FullAttentionManager: get_num_blocks_to_allocate accounting") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  BlockStore store;
  std::vector<KVCacheBlock*> cached_blocks_1;
  for (int i = 0; i < 10; ++i) cached_blocks_1.push_back(store.make(i + 1));
  std::vector<KVCacheBlock*> cached_blocks_2;
  for (int i = 0; i < 5; ++i) cached_blocks_2.push_back(pool.null_block);
  for (int i = 0; i < 5; ++i) cached_blocks_2.push_back(store.make(i + 1));

  // 10 required-suffix blocks + 10 evictable computed blocks.
  CHECK(mgr.get_num_blocks_to_allocate("1", /*num_tokens=*/40, cached_blocks_1,
                                       /*total_computed_tokens=*/0,
                                       /*num_tokens_main_model=*/40) == 20);
  // 5 of the computed blocks are null (not evictable) -> 10 + 5.
  CHECK(mgr.get_num_blocks_to_allocate("2", 40, cached_blocks_2, 0, 40) == 15);
}

TEST_CASE("FullAttentionManager: evictable cached blocks are not double allocated") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, /*enable_caching=*/true, 0, 2);

  KVCacheBlock* evictable_block = &pool.blocks[1];  // ref_cnt 0, eviction cand.

  int n = mgr.get_num_blocks_to_allocate(
      "req", /*num_tokens=*/4, {evictable_block},
      /*total_computed_tokens=*/2, /*num_tokens_main_model=*/4);
  // Free-capacity check counts the evictable block, but only 1 truly new block.
  CHECK(n == 2);

  mgr.add_local_computed_blocks("req", {evictable_block},
                                /*num_local_computed_tokens=*/2,
                                /*num_external_computed_tokens=*/0);
  auto new_blocks = mgr.allocate_new_blocks("req", 4, 4);
  CHECK(new_blocks.size() == 1);
  CHECK(mgr.req_to_blocks["req"].size() == 2);
}

TEST_CASE("FullAttentionManager: find_longest_cache_hit walks the multi-block prefix") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  std::vector<BlockHash> bh = {"h0", "h1", "h2"};

  SUBCASE("prefix hit stops at the first miss") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    // h2 not cached.
    auto computed = mgr.find_longest_cache_hit(
        bh, /*max_length=*/6, {0}, pool, *spec, /*drop_eagle_block=*/false,
        /*alignment_tokens=*/2)[0];
    CHECK(computed.size() == 2);
    CHECK(computed[0] == &pool.blocks[10]);
    CHECK(computed[1] == &pool.blocks[11]);
  }

  SUBCASE("full hit returns every block") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    MockCache(pool, "h2", &pool.blocks[12]);
    auto computed =
        mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, false, 2)[0];
    CHECK(computed.size() == 3);
    CHECK(computed[2] == &pool.blocks[12]);
  }

  SUBCASE("no hit returns empty") {
    auto computed =
        mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, false, 2)[0];
    CHECK(computed.empty());
  }
}

TEST_CASE(
    "FullAttentionManager: cache_blocks then find_longest_cache_hit returns the "
    "cached prefix (end-to-end)") {
  init_none_hash(sha256_cbor);
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  Request req = MakeRequest("r1", Iota(6), /*block_size=*/2);
  REQUIRE(req.block_hashes.size() == 3);

  auto allocated = mgr.allocate_new_blocks("r1", 6, 6);
  REQUIRE(allocated.size() == 3);
  mgr.cache_blocks(req, /*num_tokens=*/6);

  auto computed = mgr.find_longest_cache_hit(req.block_hashes, 6, {0}, pool,
                                             *spec, false, 2)[0];
  CHECK(computed.size() == 3);
  for (int i = 0; i < 3; ++i) {
    CHECK(computed[static_cast<size_t>(i)]->block_id == allocated[static_cast<size_t>(i)]->block_id);
  }
}

TEST_CASE("FullAttentionManager: free returns blocks to the pool") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  mgr.allocate_new_blocks("r1", 6, 6);
  CHECK(pool.get_num_free_blocks() == 96);
  mgr.free("r1");
  CHECK(pool.get_num_free_blocks() == 99);
  CHECK(mgr.req_to_blocks.count("r1") == 0);
  CHECK(mgr.num_cached_block.count("r1") == 0);
}

TEST_CASE("FullAttentionManager: get_num_common_prefix_blocks") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  mgr.allocate_new_blocks("r1", 6, 6);  // 3 blocks, ref_cnt 1 each.
  // Single request: all its blocks are common (ref_cnt == #requests == 1).
  CHECK(mgr.get_num_common_prefix_blocks("r1") == 3);

  // A second, unshared request drops the common count: r1's blocks have
  // ref_cnt 1 != 2.
  mgr.allocate_new_blocks("r2", 4, 4);
  CHECK(mgr.get_num_common_prefix_blocks("r1") == 0);
}

// ---------------------------------------------------------------------------
// MambaManager
// ---------------------------------------------------------------------------

TEST_CASE("MambaManager: find_longest_cache_hit keeps only the rightmost state") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, true, 0, 2);

  std::vector<BlockHash> bh = {"h0", "h1", "h2"};

  SUBCASE("all cached: [null, null, state] — single tail state") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    MockCache(pool, "h2", &pool.blocks[12]);
    auto computed = mgr.find_longest_cache_hit(bh, /*max_length=*/6, {0}, pool,
                                               *spec, false, 2)[0];
    REQUIRE(computed.size() == 3);
    CHECK(computed[0] == pool.null_block);
    CHECK(computed[1] == pool.null_block);
    CHECK(computed[2] == &pool.blocks[12]);
  }

  SUBCASE("rightmost cached is h1 (h2 miss): [null, state]") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    auto computed =
        mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, false, 2)[0];
    REQUIRE(computed.size() == 2);
    CHECK(computed[0] == pool.null_block);
    CHECK(computed[1] == &pool.blocks[11]);
  }

  SUBCASE("no hit returns empty") {
    auto computed =
        mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, false, 2)[0];
    CHECK(computed.empty());
  }
}

TEST_CASE("MambaManager: allocate_new_blocks (mode none) does not record block ids") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, true, 0, 2);

  auto blocks = mgr.allocate_new_blocks("r", 6, 6);
  CHECK(blocks.size() == 3);
  CHECK(mgr.req_to_blocks["r"].size() == 3);
  CHECK(pool.get_num_free_blocks() == 96);
  // Mamba is not a full-attention spec, so new_block_ids stays empty.
  CHECK(mgr.take_new_block_ids().empty());

  mgr.free("r");
  CHECK(pool.get_num_free_blocks() == 99);
}

TEST_CASE("MambaManager: get_num_common_prefix_blocks is always 0") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, true, 0, 2);
  mgr.allocate_new_blocks("r", 6, 6);
  CHECK(mgr.get_num_common_prefix_blocks("r") == 0);
}

TEST_CASE("MambaManager: remove_skipped_blocks frees all but the tail state") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, true, 0, 2);

  mgr.allocate_new_blocks("r", 6, 6);  // 3 blocks.
  int free_before = pool.get_num_free_blocks();

  // get_num_skipped_tokens(6) = 5 -> 5 // 2 = 2 leading blocks freed.
  mgr.remove_skipped_blocks("r", /*num_computed_tokens=*/6);
  auto& blocks = mgr.req_to_blocks["r"];
  CHECK(blocks[0] == pool.null_block);
  CHECK(blocks[1] == pool.null_block);
  CHECK(blocks[2] != pool.null_block);
  CHECK(pool.get_num_free_blocks() == free_before + 2);
}
