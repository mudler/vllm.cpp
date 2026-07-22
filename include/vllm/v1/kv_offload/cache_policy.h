// Ported from: vllm/v1/kv_offload/cpu/policies/base.py @ e24d1b24
//               vllm/v1/kv_offload/cpu/policies/lru.py
//               vllm/v1/kv_offload/cpu/policies/arc.py
//
// Scope: the pluggable cache-replacement seam the CPU offload tier (and, via
// §B3, our bounded disk tier) uses. Upstream has ONE manager
// (CPUOffloadingManager) with pluggable CachePolicy objects — there is no
// `LRUOffloadingManager` at this pin (cpu/manager.py:30-33); the record
// previously implied otherwise and is corrected in this change.
//
// TWO CONTRACTS THAT ARE LOAD-BEARING AND EASY TO BREAK
// (.agents/specs/kv-persistence-lmcache.md §Risks R5 (a) and (b)):
//
//  1. `ref_cnt == -1` IS THE "NOT READY TO READ" SENTINEL, overloaded onto the
//     refcount field (base.py:20-25). It is NOT a separate boolean. A port that
//     adds `bool ready` alongside a non-negative count WILL desynchronize the
//     two, because every transition upstream writes only ref_cnt:
//       -1 -> 0  a store completed (block becomes readable AND evictable)
//        0 -> 1  a load pinned it (no longer evictable)
//        1 -> 0  the load finished (evictable again)
//     `is_ready()` is therefore defined as `ref_cnt >= 0` and there is no other
//     readiness state anywhere.
//
//  2. `evict(n, protected)` IS ATOMIC (base.py:70-73). It returns "no result"
//     and mutates NOTHING when n evictions cannot be satisfied. A partial
//     eviction is wrong: the caller uses the all-or-nothing answer to decide
//     whether the whole store is skipped, and a half-evicted cache would have
//     freed blocks for a store that never happens.
#ifndef VLLM_V1_KV_OFFLOAD_CACHE_POLICY_H_
#define VLLM_V1_KV_OFFLOAD_CACHE_POLICY_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vllm/v1/kv_offload/base.h"

namespace vllm::v1::kv_offload {

// Upstream: class BlockStatus (cpu/policies/base.py:10-33). A ctypes.Structure
// upstream purely to shrink Python object overhead; a plain struct here.
struct BlockStatus {
  explicit BlockStatus(int64_t block_id) : block_id(block_id) {}

  // The number of transfers currently using this block as a SOURCE.
  // -1 means "not yet ready to be read" — see the header note, contract 1.
  int32_t ref_cnt = -1;
  // Index of the physical backing slot (CPU buffer slot / file slot).
  int64_t block_id = 0;

  bool is_ready() const { return ref_cnt >= 0; }
};

// A protected-key set for evict(). unordered_set over the raw key bytes.
using OffloadKeySet = std::unordered_set<OffloadKey>;

// Upstream: class CachePolicy (cpu/policies/base.py:36-92). Encapsulates BOTH
// block organization and replacement decisions — upstream notes these cannot be
// split cleanly because ARC's ghost lists and target_t1_size sit at their
// intersection.
class CachePolicy {
 public:
  virtual ~CachePolicy() = default;

  // Find a block. Returns nullptr when absent. The returned pointer is owned by
  // the policy and stays valid until the key is removed/evicted/cleared.
  virtual BlockStatus* get(const OffloadKey& key) = 0;

  // Add a newly allocated block. (ARC also drops it from the ghost lists.)
  virtual void insert(const OffloadKey& key, BlockStatus block) = 0;

  // Remove a block (used to clean up after a FAILED store).
  virtual void remove(const OffloadKey& key) = 0;

  // Mark blocks as recently used. Processed in REVERSE order, mirroring
  // upstream's `for key in reversed(list(keys))` — this is what makes the
  // FIRST key of a prefix end up most-recently-used, so a shared prefix is the
  // last thing evicted.
  virtual void touch(const std::vector<OffloadKey>& keys) = 0;

  // Evict EXACTLY n blocks, skipping any in `protected_keys`. ATOMIC: returns
  // std::nullopt and mutates nothing when n evictions are impossible.
  virtual std::optional<std::vector<std::pair<OffloadKey, BlockStatus>>> evict(
      int64_t n, const OffloadKeySet& protected_keys) = 0;

  // Remove ALL blocks regardless of ref_cnt; reset adaptive state.
  virtual void clear() = 0;

  // Called when a block's ref_cnt transitions TO 0 (it became evictable).
  virtual void mark_evictable(const OffloadKey& key) { (void)key; }
  // Called when a block's ref_cnt transitions FROM 0 (it became pinned).
  virtual void mark_non_evictable(const OffloadKey& key) { (void)key; }
};

// Upstream: _CACHE_POLICIES (cpu/manager.py:29-33). Names: "lru", "arc".
// Throws std::invalid_argument on an unknown name, mirroring upstream's
// ValueError (cpu/manager.py:60-64).
std::unique_ptr<CachePolicy> make_cache_policy(const std::string& name,
                                               int64_t cache_capacity);

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_CACHE_POLICY_H_
