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
// VERSION NOTE (which upstream is mirrored): the landed kv_cache_utils.{h,cpp}
// mirror the CLASSIC vLLM V1 block-hashing API — free functions
// hash_block_tokens / hash_request_tokens that take a pluggable hash_function,
// BlockHash as raw hash bytes, and make_block_hash_with_group_id packing a
// 4-byte group id. BlockPool is ported to match THOSE utils, i.e. the classic
// BlockPool shape:
//   - cached_block_hash_to_block is a plain
//     {BlockHashWithGroupId -> {block_id -> KVCacheBlock*}} nesting (the classic
//     defaultdict(dict)); the e24d1b24 BlockHashToBlockMap single-vs-dict union
//     optimization is intentionally NOT ported — it is a GC micro-optimization
//     with identical observable semantics.
//   - cache_full_blocks carries the classic (block_hashes, hash_fn) parameters:
//     the caller passes the precomputed per-block hashes (from
//     hash_request_tokens) and a hash function used to fill in any trailing
//     blocks whose hashes were not precomputed. The e24d1b24 signature instead
//     reads request.block_hashes (a Request field not present in the T0 Request
//     port) and takes hash_block_size on the pool; that path is not portable
//     against the landed utils, so the classic signature is used.
// The @ e24d1b24 pin marks the upstream tree tracked for the surrounding V1
// core; the BlockPool logic mirrored is upstream's classic pre-BlockHashToBlockMap
// form, consistent with the already-landed classic-hashing utils.
//
// DEFERRED, recorded (kept as stubs / no-ops so signatures stay 1:1 and the
// later units fill them in without a call-site change):
//   - KV-cache events: enable_kv_cache_events, kv_event_queue, take_events(),
//     and the BlockStored / BlockRemoved emission inside cache_full_blocks /
//     _maybe_evict_cached_block. KVCacheEvent is a placeholder struct; the
//     emission branches are omitted (marked in the .cpp). This mirrors upstream
//     vllm/distributed/kv_events.py, deferred with the rest of the V1 event bus.
//   - metrics_collector (KVCacheMetricsCollector): the on_block_allocated /
//     _accessed / _evicted hooks are omitted. Not part of the correctness core.
//   - cache_partial_block / evict_blocks: not in the classic shape and not
//     needed by the gate models; omitted.
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
  //   enable_kv_cache_events: Whether to enable kv cache events (DEFERRED).
  BlockPool(int64_t num_gpu_blocks, bool enable_caching,
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

  // Cache a list of full blocks for prefix caching. Updates each newly-full
  // block's hash metadata and inserts it into cached_block_hash_to_block.
  //
  // block_hashes: the request's per-block hashes; block_hashes[k] is the hash
  //   of the k-th full block. May be extended in place for trailing blocks
  //   whose hash was not precomputed (the else branch computes it via hash_fn
  //   and appends it — matching upstream, which caches it on the request for a
  //   possible future preemption).
  // num_cached_blocks: number of blocks already cached.
  // num_full_blocks: number of blocks that are full and should be cached now.
  // block_size: number of tokens per block.
  // kv_cache_group_id: id of the KV cache group.
  // hash_fn: the pluggable block-hash function (see kv_cache_utils.h).
  void cache_full_blocks(const Request& request,
                         const std::vector<KVCacheBlock*>& blocks,
                         std::vector<BlockHash>& block_hashes,
                         int num_cached_blocks, int num_full_blocks,
                         int block_size, int kv_cache_group_id,
                         const HashFn& hash_fn);

  // Get num_blocks new blocks from the free block pool. Does NOT check the
  // prefix cache. Throws std::runtime_error (upstream ValueError) if there are
  // not enough free blocks.
  std::vector<KVCacheBlock*> get_new_blocks(int num_blocks);

  // If a block is cached in cached_block_hash_to_block, reset its hash metadata
  // and evict it from the cache. Returns true iff the block was evicted.
  bool _maybe_evict_cached_block(KVCacheBlock* block);

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

  // The null placeholder block (block_id 0). Its ref_cnt is NOT maintained;
  // it is popped out of the free queue so it can never be allocated.
  KVCacheBlock* null_block;

  bool enable_kv_cache_events;
  std::vector<KVCacheEvent> kv_event_queue;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_BLOCK_POOL_H_
