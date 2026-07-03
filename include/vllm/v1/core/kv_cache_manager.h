// Ported from: vllm/v1/core/kv_cache_manager.py @ e24d1b24
//
// Scope (M1.3 Task 4): the TOP-LEVEL KV-cache manager the Scheduler (M1.4)
// drives to admit / preempt requests. Two ported types:
//   * KVCacheBlocks: the grouped-block wrapper returned by get_computed_blocks /
//     allocate_slots — the interface between Scheduler and KVCacheManager that
//     hides the coordinator's internal per-group block tables. `blocks[i][j]` is
//     the j-th block of the i-th kv cache group.
//   * KVCacheManager: owns the KVCacheCoordinator (M1.3 Task 3, built by
//     get_kv_cache_coordinator) + the block_pool (via the coordinator), and
//     exposes the three load-bearing operations:
//       - get_computed_blocks(request): the longest cached prefix across the
//         request's block_hashes (coordinator.find_longest_cache_hit), returned
//         as (KVCacheBlocks, num_new_computed_tokens).
//       - allocate_slots(...): THE central allocator. Ports the exact
//         comp|new_comp|ext_comp|new|lookahead block accounting and the
//         None-on-OOM contract (returns std::nullopt when the pool cannot supply
//         the needed blocks — this is the signal the Scheduler uses to preempt).
//       - free(request): return the request's blocks in reverse (tail-first)
//         eviction order — where the M1.2 LRU carry-forward
//         `[6, 10, 5, 4, 3, 2, 1]` free order is observable.
// Behavioral only: no CUDA, no model. Builds on M1.2 BlockPool and M1.3
// Tasks 1-3 (specs/config, per-group managers, coordinator).
//
// THE allocate_slots ACCOUNTING (ported line-by-line from upstream) — layout:
//   -----------------------------------------------------------------------
//   | < comp > | < new_comp > | < ext_comp > | < new > | < lookahead >     |
//   -----------------------------------------------------------------------
//   comp      = request.num_computed_tokens
//   new_comp  = num_new_computed_tokens (= len(new_computed_blocks)*block_size)
//   ext_comp  = num_external_computed_tokens (connector-cached)
//   new       = num_new_tokens (incl. unverified draft tokens)
//   lookahead = num_lookahead_tokens
//   num_local_computed_tokens = comp + new_comp
//   total_computed_tokens     = min(local + ext, max_model_len)
//   num_tokens_main_model     = total_computed_tokens + new
//   num_tokens_need_slot      = min(main_model + lookahead, max_model_len)
//   The pool must supply num_blocks_to_allocate (summed across groups by the
//   coordinator) PLUS the watermark, within (free - reserved_blocks); otherwise
//   allocate_slots returns std::nullopt (OOM -> the Scheduler preempts).
//   Only "finalized" tokens are cached: num_tokens_to_cache is capped at
//   request.num_tokens (draft tokens that could be rejected are excluded).
//
// DEVIATIONS, recorded:
//   - Upstream's `new_computed_blocks: KVCacheBlocks | None` (identity-compared
//     against self.empty_kv_cache_blocks.blocks) is taken here as
//     std::optional<KVCacheBlocks>; has_value() reproduces the "is not empty"
//     identity check that gates allocate_new_computed_blocks.
//   - Python tuples of per-group block lists are the coordinator's
//     KVCacheBlocksTuple (std::vector<std::vector<KVCacheBlock*>>);
//     KVCacheBlocks.blocks IS that type. get_block_ids returns
//     std::vector<std::vector<int>> (upstream tuple[list[int], ...]).
//   - The coordinator is heap-owned (std::unique_ptr) because it owns a
//     non-movable BlockPool the per-group managers reference; block_pool is
//     therefore exposed as a reference member into the coordinator (upstream
//     self.block_pool). KVCacheManager is consequently non-copyable/non-movable.
//   - metrics_collector (KVCacheMetricsCollector), dcp/pcp world sizes > 1, and
//     the watermark's admission-cap details ride the constructor for fidelity;
//     the gate models use the defaults (metrics_collector omitted, world sizes
//     == 1, watermark 0.0).
//
// DEFERRED (marked; the gate models never exercise these):
//   - PrefixCacheStats / make_prefix_cache_stats / the log_stats record() path
//     in get_computed_blocks: log_stats defaults false; prefix_cache_stats is
//     not materialized. Upstream vllm/v1/metrics/stats.py.
//   - request.skip_reading_prefix_cache (prompt-logprobs / all-pooling skip) and
//     request.num_preemptions: not on the T0 Request (deferred there); the
//     get_computed_blocks early-out is on enable_caching only.
//   - take_events()' kv_cache_event_metadata annotation (BlockStored group-idx
//     spec-kind / sliding-window tagging): events are deferred at the BlockPool
//     (M1.2), so take_events() forwards block_pool.take_events() unannotated.
//   - evict_blocks(): forwards to the deferred BlockPool.evict_blocks (throws).
#ifndef VLLM_V1_CORE_KV_CACHE_MANAGER_H_
#define VLLM_V1_CORE_KV_CACHE_MANAGER_H_

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_coordinator.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm::v1 {

struct Request;

// The allocation result of KVCacheManager: the interface between the Scheduler
// and the KVCacheManager, hiding the coordinator's internal data structure.
// (Upstream KVCacheBlocks dataclass.)
struct KVCacheBlocks {
  // blocks[i][j] refers to the i-th kv_cache_group and the j-th block of tokens.
  KVCacheBlocksTuple blocks;

  // Concatenate two KVCacheBlocks per group. (Upstream __add__.)
  KVCacheBlocks operator+(const KVCacheBlocks& other) const;

  // Convert to per-group block_ids (upstream get_block_ids, allow_none=False).
  std::vector<std::vector<int>> get_block_ids() const;

  // allow_none=True: returns nullopt when every group is empty.
  std::optional<std::vector<std::vector<int>>> get_block_ids(
      bool allow_none) const;

  // block_ids of unhashed blocks (single group only). (Upstream
  // get_unhashed_block_ids.)
  std::vector<int> get_unhashed_block_ids() const;

  // Per-group block_ids of unhashed, non-null blocks. (Upstream
  // get_unhashed_block_ids_all_groups.)
  std::vector<std::vector<int>> get_unhashed_block_ids_all_groups() const;

  // A new KVCacheBlocks with the same group count but no blocks.
  KVCacheBlocks new_empty() const;
};

// The top-level KV cache manager. (Upstream KVCacheManager.)
class KVCacheManager {
 public:
  // Upstream positional order: kv_cache_config, max_model_len,
  // scheduler_block_size, hash_block_size, then keyword defaults.
  // max_num_batched_tokens defaults to max_model_len when unset (nullopt).
  KVCacheManager(KVCacheConfig kv_cache_config, int max_model_len,
                 int scheduler_block_size, int hash_block_size,
                 std::optional<int> max_num_batched_tokens = std::nullopt,
                 bool enable_caching = true, bool use_eagle = false,
                 bool log_stats = false, bool enable_kv_cache_events = false,
                 int dcp_world_size = 1, int pcp_world_size = 1,
                 double watermark = 0.0);

  // Non-copyable / non-movable (owns the coordinator + a BlockPool reference).
  KVCacheManager(const KVCacheManager&) = delete;
  KVCacheManager& operator=(const KVCacheManager&) = delete;

  // KV cache usage in [0.0, 1.0]. (Upstream usage property.)
  double usage() const;

  // The computed (cached) blocks for the request + the number of computed
  // tokens. Skips the lookup when caching is disabled. The computed blocks are
  // always full; num_computed_tokens is block-size aligned.
  std::pair<KVCacheBlocks, int> get_computed_blocks(const Request& request);

  // THE central allocator. Returns the newly allocated blocks, or std::nullopt
  // when the pool cannot supply them (OOM -> the Scheduler preempts). See the
  // file header for the comp|new_comp|ext_comp|new|lookahead accounting.
  std::optional<KVCacheBlocks> allocate_slots(
      const Request& request, int num_new_tokens,
      int num_new_computed_tokens = 0,
      std::optional<KVCacheBlocks> new_computed_blocks = std::nullopt,
      int num_lookahead_tokens = 0, int num_external_computed_tokens = 0,
      bool delay_cache_blocks = false, int num_encoder_tokens = 0,
      bool full_sequence_must_fit = false, int reserved_blocks = 0,
      bool has_scheduled_reqs = true);

  // Free the request's blocks (reverse order so tail blocks are evicted first).
  void free(const Request& request);

  // Remove blocks no longer needed (e.g. outside the sliding window).
  void remove_skipped_blocks(const std::string& request_id,
                             int total_computed_tokens,
                             std::optional<int> num_prompt_tokens = std::nullopt);

  // Pop the request's bookkeeping and return its blocks WITHOUT returning them
  // to the pool (caller frees them in reverse order).
  std::vector<KVCacheBlock*> pop_blocks_for_free(const Request& request);

  // Evict blocks from the prefix cache by id. DEFERRED (BlockPool.evict_blocks
  // throws): carried for signature fidelity.
  void evict_blocks(const std::set<int>& block_ids);

  // Reset the prefix cache. Returns false (no-op) if any block is still in use.
  bool reset_prefix_cache();

  // Number of common prefix blocks per kv cache group for a running request.
  std::vector<int> get_num_common_prefix_blocks(
      const std::string& running_request_id);

  // Take the KV cache events from the block pool. DEFERRED annotation (see
  // header): forwards block_pool.take_events() unannotated.
  std::vector<KVCacheEvent> take_events();

  // The blocks of a request.
  KVCacheBlocks get_blocks(const std::string& request_id);

  // The per-group block ids of a request.
  std::vector<std::vector<int>> get_block_ids(const std::string& request_id);

  // Cache the request's blocks, if caching is enabled.
  void cache_blocks(const Request& request, int num_computed_tokens);

  // Wrap per-group block lists into a KVCacheBlocks, reusing the shared empty
  // instance when every group is empty.
  KVCacheBlocks create_kv_cache_blocks(const KVCacheBlocksTuple& blocks) const;

  // Drain new attention block ids across all managers (for zeroing).
  std::vector<int> take_new_block_ids();

  // Called when a new step is started.
  void new_step_starts();

  // --- Public state (mirrors upstream's accessible attributes; the ported
  // tests inspect these directly). -------------------------------------------

  int max_model_len;
  bool enable_caching;
  bool use_eagle;
  bool log_stats;
  int num_kv_cache_groups;
  KVCacheConfig kv_cache_config;
  // Minimum number of free blocks to keep when admitting waiting/preempted
  // requests (int(watermark * num_blocks)).
  int watermark_blocks;
  // The coordinator (heap-owned; owns the non-movable BlockPool).
  std::unique_ptr<KVCacheCoordinator> coordinator;
  // The BlockPool owned by the coordinator (upstream self.block_pool).
  BlockPool& block_pool;
  // Pre-constructed empty result (num_kv_cache_groups empty groups).
  KVCacheBlocks empty_kv_cache_blocks;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_KV_CACHE_MANAGER_H_
