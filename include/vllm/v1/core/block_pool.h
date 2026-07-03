// Ported from: vllm/v1/core/block_pool.py @ e24d1b24
//
// Scope (M1.2 Task 3): the physical KV-cache block store — BlockPool. It owns
// the block array (block 0 is the null placeholder), the intrusive LRU
// FreeKVCacheBlockQueue (from kv_cache_utils.h), and the
// cached_block_hash_to_block map that powers prefix-cache reuse + LRU eviction.
// This is the correctness core M1.3's KVCacheManager builds on. Behavioral only
// (no CUDA, no model): the physical KV memory itself is not touched here, only
// the block bookkeeping.
//
// API FIDELITY: this mirrors the e24d1b24 BlockPool API 1:1 for the COMMON
// prefix-caching path. cache_full_blocks takes the pinned signature
// (request, blocks, num_cached_blocks, num_full_blocks, block_size,
// kv_cache_group_id, block_mask) and reads request.block_hashes — the per-block
// hashes the Request computes incrementally via its _block_hasher
// (get_request_block_hasher). The pool carries hash_block_size (per the pin).
//
// DEFERRED behind 1:1 stubs (documented; the gate models — text-only GDN/MoE —
// never exercise these, and later units fill them in without a call-site change):
//   - The align / multiple-block-size path in cache_full_blocks
//     (block_size != hash_block_size, upstream's BlockHashListWithBlockSize):
//     guarded with a throw. Only the block_size == hash_block_size common case
//     is ported.
//   - The partial->full promotion branch inside cache_full_blocks (a "new full
//     block" that already carries a hash): guarded with a throw. Requires the
//     deferred partial primitives.
//   - cache_partial_block / _get_partial_block_hash /
//     _get_partial_block_parent_hash_and_start: throw-if-called stubs (partial
//     prefix entries, not needed by the gate models).
//   - evict_blocks (KV-connector-driven cache eviction): throw-if-called stub.
//   - KV-cache events: enable_kv_cache_events, kv_event_queue, take_events(),
//     and the BlockStored / BlockRemoved / AllBlocksCleared emission inside
//     cache_full_blocks / _remove_cached_block_hashes / reset_prefix_cache.
//     KVCacheEvent is a placeholder struct; the emission branches are omitted
//     (marked in the .cpp). Mirrors upstream vllm/distributed/kv_events.py.
//   - metrics_collector (KVCacheMetricsCollector): the on_block_allocated /
//     _accessed / _evicted / reset hooks are omitted. Not part of the
//     correctness core.
//   - block_mask: the parameter is present and honored (masked blocks skipped),
//     but only ever passed None by the ported (non-SWA/non-mamba) call sites.
//
// DEVIATION, recorded: the BlockHashToBlockMap single-block-vs-dict union
// (upstream's GC micro-optimization) is NOT ported. cached_block_hash_to_block
// is the plain {BlockHashWithGroupId -> {block_id -> KVCacheBlock*}} nesting.
// Observable semantics are identical (get_one_block / contain / insert / pop
// map onto the nested-map operations); only the inner GC cost differs.
//
// DEVIATIONS, recorded:
//   - The null block (block_id 0) is made un-allocatable exactly as upstream:
//     it is popleft()'d out of the free queue at construction and marked
//     is_null, so it is simply never in the free list. Its ref_cnt is NOT
//     maintained (stays 0). (The task brief's "ref_cnt pre-incremented"
//     paraphrase does not match upstream; the pop-from-free-queue mechanism is
//     what get_usage / reset_prefix_cache's "num_used_blocks == 1" invariant
//     rely on, so upstream is followed.)
//   - get_cached_block returns the inner map's first entry (begin()->second,
//     i.e. the smallest block_id) when a hash maps to multiple duplicate
//     blocks. Upstream returns the first by dict insertion order. For a single
//     cached block per hash (the common and tested case) the two coincide; the
//     duplicate-block ordering is not observable to any ported test.
//   - BlockPool is non-copyable / non-movable: it owns the block array and the
//     FreeKVCacheBlockQueue holds raw pointers into it (and the queue is itself
//     non-movable), so the pool must have a stable address.
#ifndef VLLM_V1_CORE_BLOCK_POOL_H_
#define VLLM_V1_CORE_BLOCK_POOL_H_

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"

namespace vllm::v1 {

struct Request;

// KVCacheEvent — DEFERRED placeholder (see file header). Upstream events live in
// vllm/distributed/kv_events.py (BlockStored / BlockRemoved / AllBlocksCleared).
// Kept as an empty type so take_events() retains its 1:1 signature.
struct KVCacheEvent {};

// BlockPool manages KVCacheBlocks. It provides methods to allocate, free and
// cache the kv cache blocks. The free_block_queue stores the free blocks in
// eviction order to enable allocation, free, and cache eviction. The
// cached_block_hash_to_block maps between block hash and cached block to support
// finding cached blocks by their block hash.
class BlockPool {
 public:
  // Args:
  //   num_gpu_blocks: The number of blocks in the pool (must be > 0).
  //   enable_caching: Whether to enable prefix caching.
  //   hash_block_size: The block size at which block hashes are computed. The
  //     actual block size usually equals it; with differently-sized KV cache
  //     groups it can be a multiple (the align path, DEFERRED).
  //   enable_kv_cache_events: Whether to enable kv cache events (DEFERRED).
  BlockPool(int64_t num_gpu_blocks, bool enable_caching, int hash_block_size,
            bool enable_kv_cache_events = false);

  // Non-copyable / non-movable (see DEVIATIONS in the file header).
  BlockPool(const BlockPool&) = delete;
  BlockPool& operator=(const BlockPool&) = delete;
  BlockPool(BlockPool&&) = delete;
  BlockPool& operator=(BlockPool&&) = delete;

  // Get the cached block by the block hash for each group in
  // kv_cache_group_ids, or nullopt if cache miss for any group. If there are
  // duplicated blocks, returns the first block in the cache.
  std::optional<std::vector<KVCacheBlock*>> get_cached_block(
      const BlockHash& block_hash, const std::vector<int>& kv_cache_group_ids);

  // Cache a list of full blocks for prefix caching. Reads request.block_hashes
  // (computed incrementally by the Request's _block_hasher), updates each
  // newly-full block's hash metadata and inserts it into
  // cached_block_hash_to_block.
  //
  // request: the request whose block_hashes drive caching.
  // blocks: all blocks in the request (the range
  //   [num_cached_blocks, num_full_blocks) is cached).
  // num_cached_blocks: number of blocks already cached.
  // num_full_blocks: number of blocks that are full and should be cached now.
  // block_size: number of tokens per block.
  // kv_cache_group_id: id of the KV cache group.
  // block_mask: optional mask aligned with blocks[num_cached_blocks:
  //   num_full_blocks]; masked-off (false) blocks are skipped like null blocks.
  //   Only ever None from the ported call sites (see header DEFERRED note).
  //
  // Common path only (block_size == hash_block_size); the align path and the
  // partial->full promotion branch are DEFERRED (throw). See the header.
  void cache_full_blocks(
      const Request& request, const std::vector<KVCacheBlock*>& blocks,
      int num_cached_blocks, int num_full_blocks, int block_size,
      int kv_cache_group_id,
      const std::optional<std::vector<bool>>& block_mask = std::nullopt);

  // Register a partial prefix-cache entry for an existing block. DEFERRED 1:1
  // stub (partial primitives, not needed by the gate models): throws if called.
  std::optional<BlockHashWithGroupId> cache_partial_block(
      const Request& request, KVCacheBlock* block, int num_tokens,
      int kv_cache_group_id, int block_size);

  // Evict blocks from the prefix cache by their block IDs (KV-connector-driven).
  // DEFERRED 1:1 stub: throws if called.
  void evict_blocks(const std::set<int>& block_ids);

  // Get num_blocks new blocks from the free block pool. Does NOT check the
  // prefix cache. Throws std::runtime_error (upstream ValueError) if there are
  // not enough free blocks.
  std::vector<KVCacheBlock*> get_new_blocks(int num_blocks);

  // If a block is cached in cached_block_hash_to_block, reset its hash metadata
  // and evict it from the cache. Returns true iff the block was evicted.
  bool _maybe_evict_cached_block(KVCacheBlock* block);

  // Remove every hash key that points to `block` (its primary hash plus any
  // partial-alias hashes in cached_block_hashes_by_block), reset the block's
  // hash, and return the keys actually removed from the map. Mirrors upstream
  // _remove_cached_block_hashes.
  std::vector<BlockHashWithGroupId> _remove_cached_block_hashes(
      KVCacheBlock* block);

  // Insert a hash key for `block` into cached_block_hash_to_block. If the block
  // has no primary hash yet, the key becomes its primary hash (with num_tokens);
  // otherwise the key is tracked as a partial alias in
  // cached_block_hashes_by_block. Mirrors upstream _insert_block_hash.
  void _insert_block_hash(const BlockHashWithGroupId& block_hash_with_group_id,
                          KVCacheBlock* block, std::optional<int> num_tokens);

  // Touch a block: increase its reference count by 1, and remove it from the
  // free queue if it was an eviction candidate (ref_cnt == 0). Used when a
  // block is hit by another request with the same prefix.
  void touch(const std::vector<KVCacheBlock*>& blocks);

  // Free a list of blocks, ordered by eviction priority (first block will be
  // evicted first). Decrements ref counts; a block whose ref_cnt reaches 0 is
  // appended back to the free queue in the given order (order matters for LRU).
  void free_blocks(const std::vector<KVCacheBlock*>& ordered_blocks);

  // Reset prefix cache: clears the hash map and all block hashes. Returns true
  // on success; returns false (a no-op) if any non-null block is still in use.
  bool reset_prefix_cache();

  // Number of free blocks in the pool.
  int get_num_free_blocks() const;

  // KV cache usage in [0.0, 1.0] (the null block is excluded from the total).
  double get_usage() const;

  // Atomically take all events and clear the queue. DEFERRED: always returns an
  // empty list (see file header).
  std::vector<KVCacheEvent> take_events();

  // --- Public state (mirrors upstream's accessible attributes; the ported
  // tests inspect these directly). ---------------------------------------------

  int64_t num_gpu_blocks;
  bool enable_caching;
  int hash_block_size;

  // All kv-cache blocks, indexed by block_id (blocks[0] is the null block).
  std::vector<KVCacheBlock> blocks;

  // Free block queue over `blocks` (constructs / manipulates the doubly linked
  // list of free blocks, including eviction candidates when caching is enabled).
  FreeKVCacheBlockQueue free_block_queue;

  // {block_hash_with_group_id -> {block_id -> block}}. A cached block is a full
  // block with a block hash usable for prefix caching. The cached block may be
  // used by running requests or sit in the free queue as an eviction candidate.
  std::unordered_map<BlockHashWithGroupId, std::map<int, KVCacheBlock*>>
      cached_block_hash_to_block;

  // {block_id -> set of partial-alias hash keys}. Tracks the extra hash keys
  // (beyond a block's primary block_hash) that point to the block, so eviction /
  // reset / promotion can remove every key. Only populated by the deferred
  // partial primitives; empty on the common path. Mirrors upstream
  // cached_block_hashes_by_block.
  std::unordered_map<int, std::set<BlockHashWithGroupId>>
      cached_block_hashes_by_block;

  // The null placeholder block (block_id 0). Its ref_cnt is NOT maintained;
  // it is popped out of the free queue so it can never be allocated.
  KVCacheBlock* null_block;

  bool enable_kv_cache_events;
  std::vector<KVCacheEvent> kv_event_queue;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_BLOCK_POOL_H_
