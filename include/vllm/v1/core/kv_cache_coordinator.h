// Ported from: vllm/v1/core/kv_cache_coordinator.py @ e24d1b24
//
// Scope (M1.3 Task 3): the KV-cache COORDINATOR — the layer that fans the
// per-group SingleTypeKVCacheManagers (M1.3 Task 2) out across a model's KV
// cache groups and, crucially, coordinates the CROSS-GROUP prefix-cache hit the
// hybrid gate models depend on. Four concrete coordinators are ported:
//   * KVCacheCoordinator (abstract base): owns the BlockPool + one
//     SingleTypeKVCacheManager per kv cache group (built by a small
//     spec->manager factory), and the group-fanout shared methods
//     (get_num_blocks_to_allocate = sum across groups, allocate_new_blocks =
//     per-group, cache_blocks, free/pop, get_num_common_prefix_blocks,
//     remove_skipped_blocks, get_blocks, new_step_starts).
//   * UnitaryKVCacheCoordinator (exactly ONE group — pure full-attn OR pure
//     mamba): find_longest_cache_hit delegates to the single manager.
//   * KVCacheCoordinatorNoPrefixCache (any group count): returns no hits and no
//     common-prefix blocks while retaining the ordinary allocation/free fanout.
//   * HybridKVCacheCoordinator (>1 group — the gate models: a full-attn block
//     group + a GDN/mamba-state group): the CROSS-GROUP find_longest_cache_hit
//     — the highest-risk correctness code in M1.3.
// Behavioral only: no CUDA, no model. Builds on M1.2 BlockPool and M1.3 Tasks
// 1-2 (specs, config, per-group managers).
//
// THE CROSS-GROUP find_longest_cache_hit (the load-bearing algorithm) — ported
// verbatim from upstream HybridKVCacheCoordinator.find_longest_cache_hit:
//   A shared prefix is only reusable if it is cached in EVERY group. The hybrid
//   coordinator runs an iterative FIXED-POINT: it groups kv cache groups by spec
//   (FullAttention sorted FIRST because its left-to-right scan gives the
//   tightest initial bound), then repeatedly asks each spec-group for its
//   longest hit BOUNDED BY the current candidate length. Each group either
//   accepts the candidate or REDUCES it (never grows it, since every lookup is
//   capped at the running `curr_hit_length`); if any group reduces it, the pass
//   restarts. This converges because the length monotonically decreases and is
//   bounded below by 0. For the "simple hybrid" (1 full-attn + 1 other, the gate
//   models) ONE pass suffices, so it short-circuits. The final hit length is
//   thus the MIN (intersection) of the per-group hits: e.g. if full attention
//   cached a 4-block prefix but the mamba group only cached a recurrent state at
//   block index 1 (hit length 2 blocks), the coordinated hit is 2 blocks — and
//   the full-attention blocks are TRUNCATED (`del blks[num_blocks:]`) to match.
//   The excess (longest_hit_length - hit_length) is recorded as
//   `num_uncached_common_prefix_tokens` (an uncached common prefix across
//   requests).
//
// DEVIATIONS, recorded:
//   - Per-group managers are built through the registry-backed
//     `get_manager_for_kv_cache_spec`, including the recycling-aware admission
//     cap for SlidingWindowSpec and ChunkedLocalAttentionSpec. Out-of-tree
//     platform callbacks remain deferred in kv_cache_spec_registry.h.
//   - Task 2 ported `find_longest_cache_hit` as a VIRTUAL INSTANCE method (C++
//     has no abstract static methods). Upstream's Hybrid calls
//     `manager_cls.find_longest_cache_hit(...)` (a classmethod). Here each
//     SpecGroup holds a pointer to a REPRESENTATIVE manager instance
//     (`single_type_managers[group_ids[0]]`) and the lookup is dispatched
//     through it — equivalent, since the method reads only the passed-in args
//     and the group-invariant block_size / scheduler_block_size.
//   - Python tuples of per-group block lists are `std::vector<std::vector<
//     KVCacheBlock*>>` (KVCacheBlocksTuple); find_longest_cache_hit returns
//     `std::pair<KVCacheBlocksTuple, int>` (blocks, hit_length_in_tokens).
//   - The coordinator OWNS its BlockPool (constructs it internally exactly as
//     upstream `self.block_pool = BlockPool(...)`). BlockPool is non-movable and
//     the managers hold references into it, so the coordinator is likewise
//     non-copyable / non-movable and is always heap-allocated (the factory
//     returns std::unique_ptr).
//   - `_get_block_hashes` (upstream's BlockHashListWithBlockSize wrap for groups
//     whose block_size is a multiple of hash_block_size) is DEFERRED with the
//     BlockPool align path (M1.2): the coordinator asserts every group's
//     block_size == hash_block_size and passes block_hashes through unchanged.
//     The gate models use a single uniform block size.
//   - `retention_interval` (envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL) is fixed
//     to nullopt (dense caching — the only path the gate models exercise); the
//     env read + _validate_prefix_cache_retention_interval are DEFERRED.
//   - `metrics_collector` (KVCacheMetricsCollector) is omitted (not part of the
//     allocation correctness core).
//
// DEFERRED (marked; later tasks add these without reshaping the base):
//   - CrossAttentionManager special-casing in get_num_blocks_to_allocate /
//     allocate_new_blocks (num_encoder_tokens): cross-attention is a deferred
//     manager type (Task 2), so the encoder branches are omitted.
//   - allocate_new_computed_blocks' external-KV-connector path
//     (num_external_computed_tokens): the two-phase local+external allocation is
//     ported, but only the local path is exercised by the gate models / tests.
//   - EAGLE / MTP last-block drop: the eagle_group_ids machinery + the
//     `drop_eagle_block` / eagle_verified fixed-point bookkeeping are ported for
//     fidelity but T0 gate models set use_eagle == false everywhere.
//   - dcp_world_size / pcp_world_size > 1 (context parallelism): carried for
//     signature fidelity; Hybrid asserts both == 1 (as upstream does).
//   - find_longest_cache_hit_per_group (independent per-group hits, used by the
//     KV-connector): omitted (T1).
#ifndef VLLM_V1_CORE_KV_CACHE_COORDINATOR_H_
#define VLLM_V1_CORE_KV_CACHE_COORDINATOR_H_

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/single_type_kv_cache_manager.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm::v1 {

struct Request;

// A tuple of per-group block lists (upstream `tuple[list[KVCacheBlock], ...]`).
using KVCacheBlocksTuple = std::vector<std::vector<KVCacheBlock*>>;

// Coordinate the KV cache of different KV cache groups. (Upstream
// KVCacheCoordinator, abstract base.)
class KVCacheCoordinator {
 public:
  virtual ~KVCacheCoordinator() = default;

  // Non-copyable / non-movable (owns a BlockPool; managers reference it).
  KVCacheCoordinator(const KVCacheCoordinator&) = delete;
  KVCacheCoordinator& operator=(const KVCacheCoordinator&) = delete;

  // Sum, across all groups, of the blocks needed to allocate for the request.
  int get_num_blocks_to_allocate(
      const std::string& request_id, int num_tokens,
      const KVCacheBlocksTuple& new_computed_blocks, int num_encoder_tokens,
      int total_computed_tokens, int num_tokens_main_model,
      bool apply_admission_cap = false);

  // Add the new (prefix-hit) computed blocks to the request across all groups;
  // optionally allocate external computed blocks. Two-phase (issue #33775):
  // touch every group's local hit blocks first, then allocate external blocks.
  void allocate_new_computed_blocks(
      const std::string& request_id,
      const KVCacheBlocksTuple& new_computed_blocks,
      int num_local_computed_tokens, int num_external_computed_tokens);

  // Allocate new blocks per group so each has >= num_tokens slots.
  KVCacheBlocksTuple allocate_new_blocks(const std::string& request_id,
                                         int num_tokens,
                                         int num_tokens_main_model,
                                         int num_encoder_tokens = 0);

  // Cache the request's blocks in every group. (Hybrid overrides to align to
  // scheduler_block_size.)
  virtual void cache_blocks(const Request& request, int num_computed_tokens);

  // Free the request's blocks in every group.
  void free(const std::string& request_id);

  // Pop the request's bookkeeping from all managers and return its blocks
  // (NOT returned to the pool — caller frees them in reverse order).
  std::vector<KVCacheBlock*> pop_blocks_for_free(const std::string& request_id);

  // Number of common prefix blocks per kv cache group.
  virtual std::vector<int> get_num_common_prefix_blocks(
      const std::string& running_request_id);

  // Remove skipped blocks per group.
  void remove_skipped_blocks(const std::string& request_id,
                             int total_computed_tokens,
                             std::optional<int> num_prompt_tokens = std::nullopt);

  // The per-group block tables for the request.
  KVCacheBlocksTuple get_blocks(const std::string& request_id);

  // The longest cache hit coordinated across all groups: (per-group hit blocks,
  // hit length in tokens). Differs unitary vs hybrid.
  virtual std::pair<KVCacheBlocksTuple, int> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes, int max_cache_hit_length) = 0;

  // Per-step reset hook, forwarded to every manager.
  void new_step_starts();

  // --- Public state (mirrors upstream's accessible attributes). --------------

  KVCacheConfig kv_cache_config;
  int max_model_len;
  bool enable_caching;
  int scheduler_block_size;
  // The BlockPool owned + shared by all managers (upstream self.block_pool).
  BlockPool block_pool;
  // KV cache group indices that get the EAGLE last-block drop.
  std::set<int> eagle_group_ids;
  // One manager per kv cache group (built by the spec->manager factory).
  std::vector<std::unique_ptr<SingleTypeKVCacheManager>> single_type_managers;
  // Dense caching (nullopt) is the only path the gate models exercise.
  std::optional<int> retention_interval = std::nullopt;

 protected:
  KVCacheCoordinator(KVCacheConfig kv_cache_config, int max_model_len,
                     int max_num_batched_tokens, bool use_eagle,
                     bool enable_caching, bool enable_kv_cache_events,
                     int dcp_world_size, int pcp_world_size,
                     int scheduler_block_size, int hash_block_size);

  int max_num_batched_tokens_;
  int dcp_world_size_;
  int pcp_world_size_;
  int hash_block_size_;
};

// Coordinator used when prefix caching is disabled or unsupported. It accepts
// arbitrary group counts and deliberately exposes no cache hits, while all
// allocation, block-table and free behavior stays in the common base.
class KVCacheCoordinatorNoPrefixCache : public KVCacheCoordinator {
 public:
  KVCacheCoordinatorNoPrefixCache(
      KVCacheConfig kv_cache_config, int max_model_len,
      int max_num_batched_tokens, bool use_eagle,
      bool enable_kv_cache_events, int dcp_world_size, int pcp_world_size,
      int scheduler_block_size, int hash_block_size);

  std::vector<int> get_num_common_prefix_blocks(
      const std::string& running_request_id) override;

  std::pair<KVCacheBlocksTuple, int> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes,
      int max_cache_hit_length) override;

 private:
  std::size_t num_single_type_managers_;
};

// KV cache coordinator for models with exactly one KV cache group (all layers
// of one registered spec type). (Upstream UnitaryKVCacheCoordinator.)
class UnitaryKVCacheCoordinator : public KVCacheCoordinator {
 public:
  UnitaryKVCacheCoordinator(KVCacheConfig kv_cache_config, int max_model_len,
                            int max_num_batched_tokens, bool use_eagle,
                            bool enable_caching, bool enable_kv_cache_events,
                            int dcp_world_size, int pcp_world_size,
                            int scheduler_block_size, int hash_block_size);

  std::pair<KVCacheBlocksTuple, int> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes,
      int max_cache_hit_length) override;

 private:
  KVCacheSpec* kv_cache_spec_;
  int block_size_;
};

// KV cache groups that share one spec, batched together for a single cache-hit
// lookup. (Upstream SpecGroup NamedTuple.) `representative` is the manager
// instance the (virtual-instance) find_longest_cache_hit is dispatched through
// (see the header DEVIATIONS note).
struct SpecGroup {
  KVCacheSpec* spec;
  std::vector<int> group_ids;
  SingleTypeKVCacheManager* representative;
  bool use_eagle;
};

// KV cache coordinator for hybrid models with multiple KV cache types (and thus
// multiple kv cache groups) — the gate models. (Upstream
// HybridKVCacheCoordinator.)
class HybridKVCacheCoordinator : public KVCacheCoordinator {
 public:
  HybridKVCacheCoordinator(KVCacheConfig kv_cache_config, int max_model_len,
                           int max_num_batched_tokens, bool use_eagle,
                           bool enable_caching, bool enable_kv_cache_events,
                           int dcp_world_size, int pcp_world_size,
                           int scheduler_block_size, int hash_block_size);

  // Cache hits here are always a multiple of scheduler_block_size tokens; align
  // num_computed_tokens down before caching per group.
  void cache_blocks(const Request& request, int num_computed_tokens) override;

  // The cross-group fixed-point prefix match (see the header). The final hit is
  // the intersection (min) of the per-group hits; full-attention blocks are
  // truncated to match.
  std::pair<KVCacheBlocksTuple, int> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes,
      int max_cache_hit_length) override;

  // Groups sharing a spec, FullAttention first (upstream attention_groups).
  std::vector<SpecGroup> attention_groups;
  // longest_hit_length - hit_length after the last find_longest_cache_hit: an
  // uncached common prefix across requests (upstream attribute).
  int num_uncached_common_prefix_tokens = 0;

 private:
  void verify_and_split_kv_cache_groups();
  // NOTE: this class deliberately holds NO `hash_block_size` member. It used to
  // carry a private `hash_block_size_h_` copy that duplicated the protected
  // `KVCacheCoordinator::hash_block_size_` it already inherits and that nothing
  // ever read — Clang's -Wunused-private-field flagged it on macOS
  // (BACKEND-METAL-MLX W0 item 2). Removed rather than suppressed; use the
  // inherited `hash_block_size_` if this class ever needs the value.
};

// Select + construct the coordinator for a config. (Upstream
// get_kv_cache_coordinator.) Caching off -> NoPrefixCache; otherwise 1 group ->
// Unitary and >1 groups -> Hybrid.
std::unique_ptr<KVCacheCoordinator> get_kv_cache_coordinator(
    KVCacheConfig kv_cache_config, int max_model_len,
    int max_num_batched_tokens, bool use_eagle, bool enable_caching,
    bool enable_kv_cache_events, int dcp_world_size, int pcp_world_size,
    int scheduler_block_size, int hash_block_size);

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_KV_CACHE_COORDINATOR_H_
