// Ported from: vllm/v1/core/block_pool.py @ e24d1b24
// See include/vllm/v1/core/block_pool.h for scope, the pinned-API fidelity
// note, and recorded deferrals / deviations.
#include "vllm/v1/core/block_pool.h"

#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vllm/v1/request.h"

namespace vllm::v1 {
namespace {

// Build num_gpu_blocks KVCacheBlocks with block_id = 0..n-1. Reserve up front so
// the buffer never reallocates: FreeKVCacheBlockQueue and cached_block_hash_to_block
// hold raw pointers into these elements, so their addresses must stay stable.
std::vector<KVCacheBlock> MakeBlocks(int64_t num_gpu_blocks) {
  std::vector<KVCacheBlock> blocks;
  if (num_gpu_blocks > 0) {
    blocks.reserve(static_cast<size_t>(num_gpu_blocks));
    for (int64_t idx = 0; idx < num_gpu_blocks; ++idx) {
      blocks.emplace_back(static_cast<int>(idx));
    }
  }
  return blocks;
}

// A KVCacheBlock* view over `blocks`, used to seed FreeKVCacheBlockQueue.
std::vector<KVCacheBlock*> BlockPtrs(std::vector<KVCacheBlock>& blocks) {
  std::vector<KVCacheBlock*> ptrs;
  ptrs.reserve(blocks.size());
  for (auto& block : blocks) {
    ptrs.push_back(&block);
  }
  return ptrs;
}

}  // namespace

BlockPool::BlockPool(int64_t num_gpu_blocks, bool enable_caching,
                     int hash_block_size, bool enable_kv_cache_events)
    : num_gpu_blocks(num_gpu_blocks),
      enable_caching(enable_caching),
      hash_block_size(hash_block_size),
      blocks(MakeBlocks(num_gpu_blocks)),
      free_block_queue(BlockPtrs(blocks)),
      null_block(nullptr),
      enable_kv_cache_events(enable_kv_cache_events) {
  assert(num_gpu_blocks > 0);

  // To represent a placeholder block with block_id=0. The ref_cnt of null_block
  // is not maintained; it is popped out of the free queue here so it can never
  // be allocated. (Special care is needed elsewhere to avoid freeing it.)
  null_block = free_block_queue.popleft();
  null_block->is_null = true;
}

std::optional<std::vector<KVCacheBlock*>> BlockPool::get_cached_block(
    const BlockHash& block_hash, const std::vector<int>& kv_cache_group_ids) {
  std::vector<KVCacheBlock*> cached_blocks;
  for (int group_id : kv_cache_group_ids) {
    const BlockHashWithGroupId block_hash_with_group_id =
        make_block_hash_with_group_id(block_hash,
                                      static_cast<uint32_t>(group_id));
    auto it = cached_block_hash_to_block.find(block_hash_with_group_id);
    if (it == cached_block_hash_to_block.end() || it->second.empty()) {
      return std::nullopt;
    }
    // If there are duplicated blocks, return the first block in the cache.
    cached_blocks.push_back(it->second.begin()->second);
  }
  return cached_blocks;
}

void BlockPool::cache_full_blocks(
    const Request& request, const std::vector<KVCacheBlock*>& blocks_arg,
    int num_cached_blocks, int num_full_blocks, int block_size,
    int kv_cache_group_id, const std::optional<std::vector<bool>>& block_mask) {
  if (num_cached_blocks >= num_full_blocks) {
    return;
  }
  // new_full_blocks = blocks[num_cached_blocks:num_full_blocks] (kept as an
  // index range into blocks_arg to avoid a copy).
  const int num_new_full = num_full_blocks - num_cached_blocks;
  assert(!block_mask.has_value() ||
         static_cast<int>(block_mask->size()) == num_new_full);

  // Common case: block_size == hash_block_size, block_hashes = request's list.
  // The align path (block_size a multiple of hash_block_size, upstream's
  // BlockHashListWithBlockSize) is DEFERRED (see header).
  if (block_size != hash_block_size) {
    throw std::runtime_error(
        "BlockPool::cache_full_blocks: align mode "
        "(block_size != hash_block_size) not yet ported");
  }
  const std::vector<BlockHash>& block_hashes = request.block_hashes;
  assert(static_cast<int>(block_hashes.size()) >= num_full_blocks);

  // new_block_hashes = block_hashes[num_cached_blocks:]; indexed as
  // block_hashes[num_cached_blocks + i] below.
  // new_hashes (KV-cache events) — DEFERRED (see header).
  for (int i = 0; i < num_new_full; ++i) {
    KVCacheBlock* blk = blocks_arg[num_cached_blocks + i];
    // Some blocks may be null or masked out (sparse/sliding-window attention or
    // mamba align mode). Skip them.
    if (blk->is_null || (block_mask.has_value() && !(*block_mask)[i])) {
      continue;
    }
    const BlockHash& block_hash = block_hashes[num_cached_blocks + i];
    const int num_hash_tokens = (num_cached_blocks + i + 1) * block_size;

    const BlockHashWithGroupId block_hash_with_group_id =
        make_block_hash_with_group_id(block_hash,
                                      static_cast<uint32_t>(kv_cache_group_id));
    if (blk->block_hash().has_value()) {
      // The only valid case where a "new full block" already has a hash is a
      // partial->full promotion of the same cache block — DEFERRED (requires the
      // partial primitives). See header.
      throw std::runtime_error(
          "BlockPool::cache_full_blocks: partial->full promotion not yet "
          "ported");
    }
    _insert_block_hash(block_hash_with_group_id, blk, num_hash_tokens);
    // enable_kv_cache_events (maybe_convert_block_hash append) — DEFERRED.
  }
  // enable_kv_cache_events (BlockStored emission) — DEFERRED (see header).
}

std::optional<BlockHashWithGroupId> BlockPool::cache_partial_block(
    const Request& /*request*/, KVCacheBlock* /*block*/, int /*num_tokens*/,
    int /*kv_cache_group_id*/, int /*block_size*/) {
  // DEFERRED 1:1 stub (partial prefix-cache primitives). See header.
  throw std::runtime_error(
      "BlockPool::cache_partial_block: partial primitives not yet ported");
}

void BlockPool::evict_blocks(const std::set<int>& block_ids) {
  // KV-OFFLOAD W4 (jointly with KV-BLOCK-POOL). 1:1 with block_pool.py:637-654:
  // evict the given blocks from the PREFIX CACHE hash table only. Blocks with
  // ref_cnt > 0 stay in the pool (not freed) — this drops only their cache
  // reachability, exactly as _maybe_evict_cached_block does. Used when a
  // connector reports blocks whose external copy has become authoritative.
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= static_cast<int>(blocks.size())) {
      // A connector should only report scheduler-allocated ids; upstream asserts
      // (block_pool.py:648-652). We THROW rather than corrupt on an out-of-range
      // id, which is the safe side.
      throw std::out_of_range(
          "BlockPool::evict_blocks: block_id out of range (connector bug)");
    }
    _maybe_evict_cached_block(&blocks[block_id]);
  }
}

std::vector<BlockHashWithGroupId> BlockPool::_remove_cached_block_hashes(
    KVCacheBlock* block) {
  std::vector<BlockHashWithGroupId> block_hashes;
  if (block->block_hash().has_value()) {
    block_hashes.push_back(block->block_hash().value());
  }
  auto by_block = cached_block_hashes_by_block.find(block->block_id);
  if (by_block != cached_block_hashes_by_block.end()) {
    block_hashes.insert(block_hashes.end(), by_block->second.begin(),
                        by_block->second.end());
    cached_block_hashes_by_block.erase(by_block);
  }
  if (block_hashes.empty()) {
    return {};
  }

  std::vector<BlockHashWithGroupId> removed_hashes;
  for (const BlockHashWithGroupId& block_hash : block_hashes) {
    // cached_block_hash_to_block.pop(block_hash, block_id): remove block_id from
    // the key's inner map; drop the key when its inner map empties; the pop
    // "succeeds" only when block_id was present.
    auto it = cached_block_hash_to_block.find(block_hash);
    if (it == cached_block_hash_to_block.end()) {
      continue;
    }
    auto inner = it->second.find(block->block_id);
    if (inner == it->second.end()) {
      continue;
    }
    it->second.erase(inner);
    if (it->second.empty()) {
      cached_block_hash_to_block.erase(it);
    }
    removed_hashes.push_back(block_hash);
  }
  block->reset_hash();
  return removed_hashes;
}

void BlockPool::_insert_block_hash(
    const BlockHashWithGroupId& block_hash_with_group_id, KVCacheBlock* block,
    std::optional<int> num_tokens) {
  if (block->block_hash().has_value() &&
      block->block_hash().value() == block_hash_with_group_id) {
    return;
  }
  auto it = cached_block_hash_to_block.find(block_hash_with_group_id);
  if (it != cached_block_hash_to_block.end() &&
      it->second.count(block->block_id) != 0) {
    return;
  }

  if (!block->block_hash().has_value()) {
    block->set_block_hash(block_hash_with_group_id, num_tokens);
  } else {
    cached_block_hashes_by_block[block->block_id].insert(
        block_hash_with_group_id);
  }
  cached_block_hash_to_block[block_hash_with_group_id][block->block_id] = block;
}

std::vector<KVCacheBlock*> BlockPool::get_new_blocks(int num_blocks) {
  if (num_blocks > get_num_free_blocks()) {
    throw std::runtime_error("Cannot get free blocks from the pool");
  }

  std::vector<KVCacheBlock*> ret = free_block_queue.popleft_n(num_blocks);

  // In order to only iterate the list once, we duplicate a bit of code.
  if (enable_caching) {
    for (KVCacheBlock* block : ret) {
      _maybe_evict_cached_block(block);
      assert(block->ref_cnt == 0);
      block->incr_ref();
      // metrics_collector.on_block_allocated — DEFERRED (see header).
    }
  } else {
    for (KVCacheBlock* block : ret) {
      assert(block->ref_cnt == 0);
      block->incr_ref();
      // metrics_collector.on_block_allocated — DEFERRED (see header).
    }
  }
  return ret;
}

bool BlockPool::_maybe_evict_cached_block(KVCacheBlock* block) {
  // metrics_collector.on_block_evicted — DEFERRED (see header).
  std::vector<BlockHashWithGroupId> evicted_hashes =
      _remove_cached_block_hashes(block);
  if (evicted_hashes.empty()) {
    // The block doesn't have a hash, eviction is not needed.
    return false;
  }
  // enable_kv_cache_events (BlockRemoved emission) — DEFERRED (see header).
  return true;
}

void BlockPool::touch(const std::vector<KVCacheBlock*>& blocks_arg) {
  for (KVCacheBlock* block : blocks_arg) {
    // ref_cnt=0 means this block is in the free list (i.e. an eviction
    // candidate), so remove it.
    if (block->ref_cnt == 0 && !block->is_null) {
      free_block_queue.remove(block);
    }
    block->incr_ref();
  }
}

void BlockPool::free_blocks(const std::vector<KVCacheBlock*>& ordered_blocks) {
  // Identify blocks with a hash (LRU cache candidates) and without one (they
  // can never match in APC, so they should be evicted first). Mirrors upstream
  // block_pool.py free_blocks 1:1.
  std::vector<KVCacheBlock*> blocks_with_hash;
  std::vector<KVCacheBlock*> blocks_without_hash;
  for (KVCacheBlock* block : ordered_blocks) {
    block->decr_ref();
    if (block->ref_cnt == 0 && !block->is_null) {
      if (!block->block_hash().has_value()) {
        blocks_without_hash.push_back(block);
      } else {
        blocks_with_hash.push_back(block);
      }
    }
  }

  // Blocks without a hash always get evicted first: prepend them to the FRONT
  // (immediate reuse). Blocks with a hash go to the tail (LRU order).
  free_block_queue.prepend_n(blocks_without_hash);
  free_block_queue.append_n(blocks_with_hash);
}

bool BlockPool::reset_prefix_cache() {
  const int64_t num_used_blocks = num_gpu_blocks - get_num_free_blocks();
  if (num_used_blocks != 1) {  // The null block is always marked as used.
    // Some blocks are still in use; cannot reset (upstream logs a warning).
    return false;
  }

  // Remove all hashes so that no new blocks will hit.
  cached_block_hash_to_block.clear();
  cached_block_hashes_by_block.clear();

  // Remove all hashes from all blocks.
  for (KVCacheBlock& block : blocks) {
    block.reset_hash();
  }

  // enable_kv_cache_events (AllBlocksCleared emission) — DEFERRED (see header).
  return true;
}

int BlockPool::get_num_free_blocks() const {
  return free_block_queue.num_free_blocks;
}

double BlockPool::get_usage() const {
  // Subtract 1 to account for the null block.
  const int64_t total_gpu_blocks = num_gpu_blocks - 1;
  if (total_gpu_blocks == 0) {
    return 0.0;
  }
  return 1.0 - (static_cast<double>(get_num_free_blocks()) /
                static_cast<double>(total_gpu_blocks));
}

std::vector<KVCacheEvent> BlockPool::take_events() {
  // DEFERRED: KV-cache events are not emitted (see file header). Upstream would
  // atomically drain kv_event_queue here.
  if (!enable_kv_cache_events) {
    return {};
  }
  std::vector<KVCacheEvent> events = std::move(kv_event_queue);
  kv_event_queue.clear();
  return events;
}

}  // namespace vllm::v1
