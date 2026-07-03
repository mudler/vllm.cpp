// Tests for BlockPool (M1.2 Task 3) — the physical KV block store with
// prefix-cache reuse + LRU eviction. Ported from the BlockPool-level cases of
// vllm/tests/v1/core/test_prefix_caching.py @ e24d1b24:
//   - test_cache_blocks            -> "cache_full_blocks caches full blocks"
//   - test_cache_blocks_multi_group-> "cache_full_blocks is group-aware ..."
//   - test_maybe_evict_cached_block-> "_maybe_evict_cached_block ..."
//   - test_evict (free order)      -> "free_blocks splits unhashed to the front"
// plus the allocate / touch / eviction-on-reuse / reset / null-block invariants
// upstream exercises through the pool (test_reset_prefix_cache, test_prefill).
//
// These drive the PINNED e24d1b24 BlockPool API 1:1: cache_full_blocks reads
// request.block_hashes, which the Request computes incrementally through the
// injected block hasher (get_request_block_hasher). The pool carries
// hash_block_size. free_blocks splits freed blocks: those WITHOUT a hash are
// prepended to the FRONT (immediate reuse), those WITH a hash go to the tail
// (LRU) — the pinned test_evict free-order oracle.
//
// Object-identity comparisons (`block is ...`) are ported as pointer equality.
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/request.h"

using vllm::v1::BlockHash;
using vllm::v1::BlockHashWithGroupId;
using vllm::v1::BlockPool;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlock;
using vllm::v1::make_block_hash_with_group_id;
using vllm::v1::Request;
using vllm::v1::sha256_cbor;

namespace {

// A Request over the given token ids, with the incremental block hasher injected
// (so request.block_hashes is populated exactly as upstream make_request does).
// init_none_hash MUST have been called first (the hasher runs in the ctor).
Request MakeRequest(const std::string& id, const std::vector<int32_t>& tokens,
                    int block_size) {
  return Request(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
                 get_request_block_hasher(block_size, sha256_cbor));
}

// Token ids 0..n-1, matching upstream's list(range(n)).
std::vector<int32_t> Iota(int n) {
  std::vector<int32_t> v;
  v.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v.push_back(i);
  return v;
}

std::vector<int> FreeBlockIds(const BlockPool& pool) {
  std::vector<int> ids;
  for (auto* b : pool.free_block_queue.get_all_free_blocks()) {
    ids.push_back(b->block_id);
  }
  return ids;
}

}  // namespace

TEST_CASE("BlockPool: construction pops the null block out of the free list") {
  BlockPool pool(/*num_gpu_blocks=*/10, /*enable_caching=*/true,
                 /*hash_block_size=*/16);
  // Block 0 is the null placeholder: is_null, never in the free list.
  CHECK(pool.null_block == &pool.blocks[0]);
  CHECK(pool.null_block->block_id == 0);
  CHECK(pool.null_block->is_null);
  // 10 total, minus the popped null block.
  CHECK(pool.get_num_free_blocks() == 9);
  CHECK(pool.get_usage() == doctest::Approx(0.0));
  // The null block is not among the free blocks.
  for (int id : FreeBlockIds(pool)) {
    CHECK(id != 0);
  }
}

TEST_CASE("BlockPool: get_new_blocks allocates distinct non-null blocks") {
  BlockPool pool(10, true, 16);
  auto blocks = pool.get_new_blocks(3);
  CHECK(blocks.size() == 3);
  CHECK(pool.get_num_free_blocks() == 6);
  CHECK(pool.get_usage() == doctest::Approx(1.0 - 6.0 / 9.0));

  std::vector<int> ids;
  for (auto* b : blocks) {
    CHECK(b->ref_cnt == 1);      // allocated -> ref_cnt 1
    CHECK(b->block_id != 0);     // never the null block
    ids.push_back(b->block_id);
  }
  // Distinct blocks.
  CHECK(ids[0] != ids[1]);
  CHECK(ids[1] != ids[2]);
  CHECK(ids[0] != ids[2]);
}

TEST_CASE("BlockPool: get_new_blocks throws when not enough free blocks") {
  BlockPool pool(4, true, 16);  // 3 allocatable (block 0 is null).
  CHECK(pool.get_num_free_blocks() == 3);
  CHECK_THROWS_AS(pool.get_new_blocks(4), std::runtime_error);
  auto blocks = pool.get_new_blocks(3);  // exhaust the pool
  CHECK(pool.get_num_free_blocks() == 0);
  CHECK_THROWS_AS(pool.get_new_blocks(1), std::runtime_error);
}

// Ported from test_cache_blocks: cache_full_blocks reads request.block_hashes.
TEST_CASE("BlockPool: cache_full_blocks caches full blocks for reuse") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  BlockPool pool(/*num_gpu_blocks=*/5, /*enable_caching=*/true, block_size);

  // Req: 14 tokens -> 3 full blocks of 4, trailing partial not hashed.
  Request req = MakeRequest("0", Iota(14), block_size);
  REQUIRE(req.block_hashes.size() == 3);

  // Standalone blocks with ids 0..2 (upstream builds KVCacheBlock(block_id=i)).
  // Reserve so addresses stay stable while the pool map holds pointers to them.
  std::vector<KVCacheBlock> owned;
  owned.reserve(3);
  owned.emplace_back(0);
  owned.emplace_back(1);
  std::vector<KVCacheBlock*> blocks{&owned[0], &owned[1]};

  // Cache 2 full blocks from the start.
  pool.cache_full_blocks(req, blocks, /*num_cached=*/0, /*num_full=*/2,
                         block_size, /*group=*/0);
  CHECK(pool.cached_block_hash_to_block.size() == 2);
  CHECK(owned[0].block_hash().has_value());
  CHECK(owned[1].block_hash().has_value());

  // Cache a block that does not start from the beginning.
  owned.emplace_back(2);
  blocks.push_back(&owned[2]);
  pool.cache_full_blocks(req, blocks, /*num_cached=*/2, /*num_full=*/3,
                         block_size, /*group=*/0);
  CHECK(pool.cached_block_hash_to_block.size() == 3);
  CHECK(owned[2].block_hash().has_value());

  // get_cached_block returns the cached block for a prefix hit.
  auto hit0 = pool.get_cached_block(req.block_hashes[0], /*groups=*/{0});
  REQUIRE(hit0.has_value());
  CHECK(hit0->size() == 1);
  CHECK(hit0->front() == &owned[0]);
  CHECK(pool.get_cached_block(req.block_hashes[2], {0}).has_value());
  // A hash that was never cached misses.
  const BlockHash never = "not-a-real-hash";
  CHECK_FALSE(pool.get_cached_block(never, {0}).has_value());
}

// Ported from test_cache_blocks_multi_group.
TEST_CASE("BlockPool: cache_full_blocks is group-aware") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  BlockPool pool(10, true, block_size);

  Request req = MakeRequest("0", Iota(14), block_size);
  REQUIRE(req.block_hashes.size() == 3);

  // Cache 2 blocks for group 0.
  std::vector<KVCacheBlock> g0;
  g0.reserve(2);
  g0.emplace_back(0);
  g0.emplace_back(1);
  std::vector<KVCacheBlock*> b0{&g0[0], &g0[1]};
  pool.cache_full_blocks(req, b0, 0, 2, block_size, /*group=*/0);
  CHECK(pool.cached_block_hash_to_block.size() == 2);
  CHECK(req.block_hashes.size() == 3);

  // Cache 3 blocks for group 1.
  std::vector<KVCacheBlock> g1;
  g1.reserve(3);
  g1.emplace_back(2);
  g1.emplace_back(3);
  g1.emplace_back(4);
  std::vector<KVCacheBlock*> b1{&g1[0], &g1[1], &g1[2]};
  pool.cache_full_blocks(req, b1, 0, 3, block_size, /*group=*/1);
  CHECK(pool.cached_block_hash_to_block.size() == 5);

  // Block hash 0/1: hit for group 0 and 1. Block hash 2: hit for group 1 only.
  CHECK(pool.get_cached_block(req.block_hashes[0], {0}).has_value());
  CHECK(pool.get_cached_block(req.block_hashes[1], {0}).has_value());
  CHECK_FALSE(pool.get_cached_block(req.block_hashes[2], {0}).has_value());
  CHECK(pool.get_cached_block(req.block_hashes[0], {1}).has_value());
  CHECK(pool.get_cached_block(req.block_hashes[1], {1}).has_value());
  CHECK(pool.get_cached_block(req.block_hashes[2], {1}).has_value());
  CHECK(pool.get_cached_block(req.block_hashes[0], {0, 1}).has_value());
  CHECK(pool.get_cached_block(req.block_hashes[1], {0, 1}).has_value());
  // Group 0 misses hash 2, so the multi-group lookup misses.
  CHECK_FALSE(pool.get_cached_block(req.block_hashes[2], {0, 1}).has_value());
}

// Blocks without a hash are prepended to the FRONT (immediate reuse). This is
// the F1 free_blocks fix.
TEST_CASE("BlockPool: free_blocks prepends unhashed blocks to the front") {
  BlockPool pool(6, true, 16);          // blocks 1..5 allocatable.
  auto blocks = pool.get_new_blocks(3);  // ids 1, 2, 3 (none cached => no hash)
  CHECK(FreeBlockIds(pool) == std::vector<int>{4, 5});

  // Free ordered [3, 2, 1]; all lack a hash, so all get prepended to the FRONT
  // in that order (ahead of the pre-existing free blocks 4, 5).
  pool.free_blocks({blocks[2], blocks[1], blocks[0]});
  CHECK(pool.get_num_free_blocks() == 5);
  CHECK(FreeBlockIds(pool) == std::vector<int>{3, 2, 1, 4, 5});
  for (auto* b : blocks) {
    CHECK(b->ref_cnt == 0);
  }
}

// Ported from test_evict's free-order oracle: unhashed (partial) blocks at the
// head, hashed (full) blocks at the tail — the pinned order shape
// [6, 10, 5, 4, 3, 2, 1] reproduced at the pool level.
TEST_CASE("BlockPool: free_blocks splits unhashed-to-front / hashed-to-tail") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  // 8 blocks: block 0 null, blocks 1..7 allocatable.
  BlockPool pool(8, true, block_size);

  // 5 full blocks + 1 partial (21 tokens) -> block_hashes has 5 entries.
  Request req = MakeRequest("0", Iota(5 * block_size + 1), block_size);
  REQUIRE(req.block_hashes.size() == 5);

  // Allocate 6 blocks (ids 1..6). One free block (id 7) is left pre-existing.
  auto allocated = pool.get_new_blocks(6);
  CHECK(FreeBlockIds(pool) == std::vector<int>{7});

  // Cache the first 5 as full blocks (they gain a hash). Block id 6 (the
  // partial) is never cached, so it stays hashless.
  std::vector<KVCacheBlock*> full5(allocated.begin(), allocated.begin() + 5);
  pool.cache_full_blocks(req, full5, 0, 5, block_size, /*group=*/0);
  for (int i = 0; i < 5; ++i) CHECK(allocated[i]->block_hash().has_value());
  CHECK(allocated[5]->block_id == 6);
  CHECK_FALSE(allocated[5]->block_hash().has_value());

  // Free in reverse block order [6, 5, 4, 3, 2, 1] (as KVCacheManager.free does).
  // unhashed=[6] -> prepended to the front; hashed=[5,4,3,2,1] -> appended to
  // the tail. Queue before: [7]. Result: [6, 7, 5, 4, 3, 2, 1].
  std::vector<KVCacheBlock*> rev(allocated.rbegin(), allocated.rend());
  pool.free_blocks(rev);
  CHECK(pool.get_num_free_blocks() == 7);
  CHECK(FreeBlockIds(pool) == std::vector<int>{6, 7, 5, 4, 3, 2, 1});
}

TEST_CASE("BlockPool: touch prevents a cached block from being evicted") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  BlockPool pool(4, true, block_size);  // blocks 1..3 allocatable.

  Request req = MakeRequest("0", Iota(4), block_size);  // one full block
  REQUIRE(req.block_hashes.size() == 1);

  // Allocate and cache block 1, then free it: it becomes an eviction candidate
  // (ref_cnt 0, in the free queue) that still carries its hash.
  KVCacheBlock* b = pool.get_new_blocks(1).front();
  std::vector<KVCacheBlock*> one{b};
  pool.cache_full_blocks(req, one, 0, 1, block_size, 0);
  pool.free_blocks(one);
  CHECK(b->ref_cnt == 0);
  CHECK(pool.get_cached_block(req.block_hashes[0], {0}).has_value());

  // Touch it (a prefix hit from another request): removed from the free queue,
  // ref_cnt back to 1, so it is no longer an eviction candidate.
  pool.touch(one);
  CHECK(b->ref_cnt == 1);
  CHECK(pool.get_num_free_blocks() == 2);  // blocks 2, 3

  // Exhaust the remaining free blocks. The touched block must survive.
  pool.get_new_blocks(2);
  CHECK(pool.get_cached_block(req.block_hashes[0], {0}).has_value());
  CHECK(b->block_hash().has_value());
}

TEST_CASE("BlockPool: get_new_blocks evicts the LRU cached block on reuse") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  BlockPool pool(4, true, block_size);  // blocks 1..3 allocatable.

  Request req = MakeRequest("0", Iota(4), block_size);
  REQUIRE(req.block_hashes.size() == 1);

  // Cache block 1 and free it -> eviction candidate at the tail with a hash.
  KVCacheBlock* b = pool.get_new_blocks(1).front();
  std::vector<KVCacheBlock*> one{b};
  pool.cache_full_blocks(req, one, 0, 1, block_size, 0);
  pool.free_blocks(one);
  CHECK(pool.get_cached_block(req.block_hashes[0], {0}).has_value());
  CHECK(pool.get_num_free_blocks() == 3);  // blocks 2, 3, and freed 1 (at tail)

  // Allocate all free blocks. Reusing the cached block evicts it from the cache
  // via _maybe_evict_cached_block.
  auto reused = pool.get_new_blocks(3);
  // The cached block is now reused and no longer serves a prefix hit.
  CHECK_FALSE(pool.get_cached_block(req.block_hashes[0], {0}).has_value());
  CHECK_FALSE(b->block_hash().has_value());  // hash was reset on eviction
  CHECK(pool.cached_block_hash_to_block.empty());
  // b is the last block reused (LRU tail), and it is now allocated.
  CHECK(reused.back() == b);
  CHECK(b->ref_cnt == 1);
}

TEST_CASE("BlockPool: reset_prefix_cache clears when all blocks are free") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  BlockPool pool(4, true, block_size);

  Request req = MakeRequest("0", Iota(4), block_size);
  REQUIRE(req.block_hashes.size() == 1);

  KVCacheBlock* b = pool.get_new_blocks(1).front();
  std::vector<KVCacheBlock*> one{b};
  pool.cache_full_blocks(req, one, 0, 1, block_size, 0);

  // A block is still in use (ref_cnt 1): reset must fail and be a no-op.
  CHECK_FALSE(pool.reset_prefix_cache());
  CHECK_FALSE(pool.cached_block_hash_to_block.empty());

  // Free it; now only the null block is "used" -> reset succeeds and clears.
  pool.free_blocks(one);
  CHECK_FALSE(pool.cached_block_hash_to_block.empty());
  CHECK(pool.reset_prefix_cache());
  CHECK(pool.cached_block_hash_to_block.empty());
  for (const auto& blk : pool.blocks) {
    CHECK_FALSE(blk.block_hash().has_value());
  }
}

// Ported from test_maybe_evict_cached_block: duplicate block hashes share a map
// entry; eviction removes only the given block and drops the key when its inner
// map empties.
TEST_CASE("BlockPool: _maybe_evict_cached_block handles duplicate hashes") {
  BlockPool pool(/*num_gpu_blocks=*/4, /*enable_caching=*/true,
                 /*hash_block_size=*/16);
  const BlockHashWithGroupId h0 = make_block_hash_with_group_id("10", 1000);
  const BlockHashWithGroupId h1 = make_block_hash_with_group_id("20", 2000);
  const BlockHashWithGroupId h2 = make_block_hash_with_group_id("30", 3000);
  // block 3 shares the exact same hash as block 0.
  const std::vector<BlockHashWithGroupId> hs{h0, h1, h2, h0};
  REQUIRE(pool.blocks.size() == 4);

  // Manually seed the cache (as upstream does).
  for (int i = 0; i < 4; ++i) {
    pool.blocks[i].set_block_hash(hs[i]);
    pool.cached_block_hash_to_block[hs[i]][pool.blocks[i].block_id] =
        &pool.blocks[i];
  }
  auto& cache = pool.cached_block_hash_to_block;
  CHECK(cache.size() == 3);
  CHECK(cache[h0].size() == 2);  // {block0, block3}
  CHECK(cache[h1].size() == 1);
  CHECK(cache[h2].size() == 1);

  // Evict block1: its key is dropped entirely.
  CHECK(pool._maybe_evict_cached_block(&pool.blocks[1]));
  CHECK(cache.size() == 2);
  CHECK(cache.find(h1) == cache.end());

  // Evict block0: block_hash0 stays because block3 still uses it.
  CHECK(pool._maybe_evict_cached_block(&pool.blocks[0]));
  CHECK(cache.size() == 2);
  REQUIRE(cache.find(h0) != cache.end());
  CHECK(cache[h0].size() == 1);
  CHECK(cache[h0].begin()->second == &pool.blocks[3]);

  // Evict block2.
  CHECK(pool._maybe_evict_cached_block(&pool.blocks[2]));
  CHECK(cache.size() == 1);
  CHECK(cache.find(h2) == cache.end());

  // Evict block3: last user of block_hash0, map becomes empty.
  CHECK(pool._maybe_evict_cached_block(&pool.blocks[3]));
  CHECK(cache.empty());

  // A block with no hash: eviction is a no-op returning false.
  CHECK_FALSE(pool._maybe_evict_cached_block(&pool.blocks[0]));
}

TEST_CASE("BlockPool: caching disabled skips eviction bookkeeping") {
  BlockPool pool(4, /*enable_caching=*/false, /*hash_block_size=*/16);
  auto blocks = pool.get_new_blocks(3);
  CHECK(blocks.size() == 3);
  for (auto* b : blocks) {
    CHECK(b->ref_cnt == 1);
    CHECK_FALSE(b->block_hash().has_value());
  }
  pool.free_blocks(blocks);
  CHECK(pool.get_num_free_blocks() == 3);
}

// The deferred align path (block_size != hash_block_size) and the deferred
// partial primitives are guarded 1:1 stubs.
TEST_CASE("BlockPool: deferred align / partial primitives throw") {
  init_none_hash(sha256_cbor, "seed42");
  const int hash_block_size = 4;
  BlockPool pool(8, true, hash_block_size);

  Request req = MakeRequest("0", Iota(16), hash_block_size);
  std::vector<KVCacheBlock> owned;
  owned.reserve(2);
  owned.emplace_back(0);
  owned.emplace_back(1);
  std::vector<KVCacheBlock*> blocks{&owned[0], &owned[1]};

  // block_size != hash_block_size -> align path, DEFERRED.
  CHECK_THROWS_AS(
      pool.cache_full_blocks(req, blocks, 0, 1, /*block_size=*/8, 0),
      std::runtime_error);
  // cache_partial_block / evict_blocks -> DEFERRED stubs.
  CHECK_THROWS_AS(pool.cache_partial_block(req, &owned[0], 4, 0, 8),
                  std::runtime_error);
  CHECK_THROWS_AS(pool.evict_blocks({1, 2}), std::runtime_error);
}
