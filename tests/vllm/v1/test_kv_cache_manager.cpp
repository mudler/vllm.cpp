// Tests for the TOP-LEVEL KVCacheManager (M1.3 Task 4) — get_computed_blocks
// (prefix reuse), allocate_slots (the central allocator + None-on-OOM), free
// (the LRU eviction order), reset_prefix_cache, and the hybrid full-attn+mamba
// allocation.
//
// Ported from vllm/tests/v1/core/test_prefix_caching.py @ e24d1b24:
//   - test_prefill: get_computed_blocks returns the cached prefix across two
//     requests (req1 hits req0's 3-block prefix) + allocate_slots success
//     returning ([1, 2, 3, 4],) and reducing the free-block count.
//   - test_evict: the LITERAL manager-level free order (M1.2 LRU carry-forward)
//     `[6, 10, 5, 4, 3, 2, 1]` then `[6, 10, 5, 4, 3, 2, 1, 9, 8, 7]`.
//   - test_prefill_hybrid_model: the 2-group (full-attn + GDN mamba) allocate.
// The allocate_slots OOM -> std::nullopt case (the signal the Scheduler uses to
// preempt) has no single dedicated upstream test; it is the return contract
// asserted directly here.
//
// Only sha256_cbor is ported (M1.2), so these use it; the block ids / free order
// are hash-function-independent.
#include <doctest/doctest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_manager.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlocks;
using vllm::v1::KVCacheConfig;
using vllm::v1::KVCacheManager;
using vllm::v1::MambaSpec;
using vllm::v1::Request;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

KVCacheConfig MakeFullConfig(int block_size, int num_blocks) {
  KVCacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return cfg;
}

KVCacheConfig MakeHybridConfig(int block_size, int num_blocks) {
  KVCacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"full"},
      std::make_shared<FullAttentionSpec>(block_size, 1, 1, DType::kF32));
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mamba"},
      std::make_shared<MambaSpec>(block_size,
                                  std::vector<std::vector<int64_t>>{{1, 1}},
                                  std::vector<DType>{DType::kF32}));
  return cfg;
}

std::unique_ptr<KVCacheManager> MakeManager(KVCacheConfig cfg, int block_size) {
  return std::make_unique<KVCacheManager>(
      std::move(cfg), /*max_model_len=*/8192,
      /*scheduler_block_size=*/block_size, /*hash_block_size=*/block_size);
}

// Build a request whose token ids are `tokens`, wired with the cbor block hasher
// so its block_hashes populate at block_size granularity.
Request MakeRequest(const std::string& id, const std::vector<int32_t>& tokens,
                    int block_size) {
  return Request(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
                 get_request_block_hasher(block_size, sha256_cbor));
}

std::vector<int32_t> Range(int lo, int hi) {
  std::vector<int32_t> v;
  for (int i = lo; i < hi; ++i) v.push_back(i);
  return v;
}

std::vector<int> FreeBlockIds(KVCacheManager& mgr) {
  std::vector<int> ids;
  for (const auto* blk : mgr.block_pool.free_block_queue.get_all_free_blocks()) {
    ids.push_back(blk->block_id);
  }
  return ids;
}

}  // namespace

// ---------------------------------------------------------------------------
// get_computed_blocks: prefix reuse across two requests (test_prefill)
// ---------------------------------------------------------------------------

TEST_CASE(
    "KVCacheManager: get_computed_blocks returns the cached prefix across two "
    "requests (ported test_prefill)") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeFullConfig(block_size, 11), block_size);

  // 3 full blocks (48 common tokens) + 7 unique = 55 tokens.
  std::vector<int32_t> common;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < block_size; ++j) common.push_back(i);
  std::vector<int32_t> tokens0 = common;
  for (int i = 0; i < 7; ++i) tokens0.push_back(3);

  Request req0 = MakeRequest("0", tokens0, block_size);
  REQUIRE(req0.block_hashes.size() == 3);
  auto [computed0, n0] = mgr->get_computed_blocks(req0);
  CHECK(computed0.blocks[0].empty());  // full cache miss
  CHECK(n0 == 0);

  int free_before = mgr->block_pool.get_num_free_blocks();
  auto blocks0 = mgr->allocate_slots(req0, /*num_new_tokens=*/55,
                                     /*num_new_computed_tokens=*/n0,
                                     std::optional<KVCacheBlocks>(computed0));
  REQUIRE(blocks0.has_value());
  // cdiv(55, 16) = 4 blocks: [1, 2, 3, 4].
  CHECK(blocks0->get_block_ids() == std::vector<std::vector<int>>{{1, 2, 3, 4}});
  CHECK(mgr->block_pool.get_num_free_blocks() == free_before - 4);

  // req1 shares the 3-block (48-token) prefix.
  std::vector<int32_t> tokens1 = common;
  for (int i = 0; i < 5; ++i) tokens1.push_back(3);
  Request req1 = MakeRequest("1", tokens1, block_size);
  auto [computed1, n1] = mgr->get_computed_blocks(req1);
  CHECK(computed1.get_block_ids() == std::vector<std::vector<int>>{{1, 2, 3}});
  CHECK(n1 == 3 * block_size);

  auto blocks1 = mgr->allocate_slots(req1, /*num_new_tokens=*/53 - 3 * 16, n1,
                                     std::optional<KVCacheBlocks>(computed1));
  REQUIRE(blocks1.has_value());
  CHECK(blocks1->get_block_ids() == std::vector<std::vector<int>>{{5}});
  // The shared prefix blocks are now referenced by both requests.
  for (auto* blk : computed1.blocks[0]) CHECK(blk->ref_cnt == 2);
}

// ---------------------------------------------------------------------------
// allocate_slots: OOM -> std::nullopt (the Scheduler preempts)
// ---------------------------------------------------------------------------

TEST_CASE("KVCacheManager: allocate_slots returns std::nullopt on OOM") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  // 4 usable blocks (num_blocks 5, minus the null block).
  auto mgr = MakeManager(MakeFullConfig(block_size, 5), block_size);

  // Ask for 10 blocks worth of tokens -> cannot fit -> nullopt.
  Request req = MakeRequest("0", Range(0, 10 * block_size), block_size);
  auto [computed, n] = mgr->get_computed_blocks(req);
  auto blocks = mgr->allocate_slots(req, /*num_new_tokens=*/10 * block_size, n,
                                    std::optional<KVCacheBlocks>(computed));
  CHECK(!blocks.has_value());
  // No blocks were consumed by the failed allocation.
  CHECK(mgr->block_pool.get_num_free_blocks() == 4);
}

// ---------------------------------------------------------------------------
// free: the LITERAL LRU eviction order (ported test_evict)
// ---------------------------------------------------------------------------

TEST_CASE(
    "KVCacheManager: free returns blocks in the pinned LRU order "
    "[6, 10, 5, 4, 3, 2, 1] (ported test_evict)") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeFullConfig(block_size, 11), block_size);

  int last = 5 * 16 + 7;  // 5 full + 1 partial block
  Request req0 = MakeRequest("0", Range(0, last), block_size);
  auto [c0, n0] = mgr->get_computed_blocks(req0);
  CHECK(c0.blocks[0].empty());
  CHECK(n0 == 0);
  auto b0 = mgr->allocate_slots(req0, last, n0, std::optional<KVCacheBlocks>(c0));
  REQUIRE(b0.has_value());
  CHECK(b0->blocks[0].size() == 6);  // 5 full + 1 partial

  Request req1 = MakeRequest("1", Range(last, last + 3 * 16), block_size);
  auto [c1, n1] = mgr->get_computed_blocks(req1);
  CHECK(c1.blocks[0].empty());
  CHECK(n1 == 0);
  auto b1 = mgr->allocate_slots(req1, 3 * 16, n1, std::optional<KVCacheBlocks>(c1));
  REQUIRE(b1.has_value());
  CHECK(b1->blocks[0].size() == 3);  // 3 full blocks

  // 10 usable - (6 + 3) == 1
  CHECK(mgr->block_pool.free_block_queue.num_free_blocks == 1);

  mgr->free(req0);
  // Partial block (unhashed, id 6) at head for immediate reuse; the pre-existing
  // free block (10) next; then the hashed blocks (5, 4, 3, 2, 1) at the tail.
  CHECK(FreeBlockIds(*mgr) == std::vector<int>{6, 10, 5, 4, 3, 2, 1});

  mgr->free(req1);
  CHECK(mgr->block_pool.free_block_queue.num_free_blocks == 10);
  CHECK(FreeBlockIds(*mgr) == std::vector<int>{6, 10, 5, 4, 3, 2, 1, 9, 8, 7});
}

// ---------------------------------------------------------------------------
// reset_prefix_cache
// ---------------------------------------------------------------------------

TEST_CASE("KVCacheManager: reset_prefix_cache clears cached prefixes") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeFullConfig(block_size, 11), block_size);

  std::vector<int32_t> tokens = Range(0, 3 * block_size);
  Request req0 = MakeRequest("0", tokens, block_size);
  auto [c0, n0] = mgr->get_computed_blocks(req0);
  auto b0 = mgr->allocate_slots(req0, 3 * block_size, n0,
                                std::optional<KVCacheBlocks>(c0));
  REQUIRE(b0.has_value());
  mgr->free(req0);

  // A second request now hits the cached 3-block prefix.
  Request req1 = MakeRequest("1", tokens, block_size);
  auto [c1, n1] = mgr->get_computed_blocks(req1);
  CHECK(n1 == 2 * block_size);  // last token withheld -> 2 full blocks hit

  // After a reset the prefix cache is empty (no in-use blocks -> succeeds).
  CHECK(mgr->reset_prefix_cache());
  Request req2 = MakeRequest("2", tokens, block_size);
  auto [c2, n2] = mgr->get_computed_blocks(req2);
  CHECK(n2 == 0);
  CHECK(c2.blocks[0].empty());
}

// ---------------------------------------------------------------------------
// Hybrid full-attn + mamba allocation (ported test_prefill_hybrid_model)
// ---------------------------------------------------------------------------

TEST_CASE("KVCacheManager: allocate_slots over a hybrid full-attn + mamba model") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeHybridConfig(block_size, 21), block_size);
  REQUIRE(mgr->num_kv_cache_groups == 2);

  // 3 full blocks (48 tokens) + 7 unique = 55 tokens.
  Request req0 = MakeRequest("0", Range(0, 55), block_size);
  auto [c0, n0] = mgr->get_computed_blocks(req0);
  CHECK(n0 == 0);

  int free_before = mgr->block_pool.get_num_free_blocks();
  auto b0 = mgr->allocate_slots(req0, 55, n0, std::optional<KVCacheBlocks>(c0));
  REQUIRE(b0.has_value());
  REQUIRE(b0->blocks.size() == 2);
  // Both groups allocate cdiv(55, 16) = 4 blocks.
  CHECK(b0->blocks[0].size() == 4);  // full attention
  CHECK(b0->blocks[1].size() == 4);  // mamba
  CHECK(mgr->block_pool.get_num_free_blocks() == free_before - 8);

  mgr->free(req0);
  CHECK(mgr->block_pool.get_num_free_blocks() == free_before);
}
