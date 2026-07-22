// Ported from: vllm/v1/kv_offload/base.py @ e24d1b24
//
// Scope: the core abstractions every KV-offload tier is built on — the offload
// KEY, the tri-state-plus-retry lookup result, the load/store specs, and the
// OffloadingManager contract.
//
// THE KEY IS ALREADY OURS. Upstream's `OffloadKey` is
// `block_hash_bytes + group_idx.to_bytes(4, "big")` (base.py:27-47) —
// deliberately raw bytes rather than a tuple to avoid Python GC overhead. Our
// `BlockHashWithGroupId` packs a 4-byte BIG-ENDIAN group id onto the 32-byte
// SHA-256 digest in exactly the same order
// (src/vllm/v1/core/kv_cache_utils.cpp:279-301), so the encoding is
// BYTE-IDENTICAL and `OffloadKey` is a type alias, not a conversion. This is
// the identity that makes an offload tier possible at all: the tier and the
// prefix cache key on the SAME Request::block_hashes
// (upstream asserts the agreement at kv_offload/base.py:536-549).
//
// SEMANTICS A COMPETENT PORT GETS WRONG (each mirrored deliberately; see
// .agents/specs/kv-persistence-lmcache.md §Risks R5):
//   - LookupResult has FOUR states. kHitPending ("present but a store is still
//     in flight") is not a hit, and kRetry ("ask again, a promotion is running")
//     is not a miss. Collapsing either loses correctness.
//   - PrepareStoreOutput being ABSENT is a normal CONTROL PATH meaning "skip
//     this store", not an error (upstream `prepare_store -> None`,
//     cpu/manager.py:192-194). std::optional models it; callers must not treat
//     nullopt as a failure.
//
// NOT PORTED, recorded: GPULoadStoreSpec's group_sizes/block_indices
// (base.py:362-398) carry the unaligned first/last offloaded block when the
// offloaded block is LARGER than a GPU block. Our block_size == hash_block_size
// today (the differing-size path throws at
// src/vllm/v1/core/block_pool.cpp:93-97), so there is no unaligned case to
// describe; the fields are omitted rather than carried as always-degenerate.
#ifndef VLLM_V1_KV_OFFLOAD_BASE_H_
#define VLLM_V1_KV_OFFLOAD_BASE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"

namespace vllm::v1::kv_offload {

// Upstream: OffloadKey (base.py:27-30). Byte-identical to our existing
// BlockHashWithGroupId — see the header note.
using OffloadKey = BlockHashWithGroupId;

// Upstream: make_offload_key / get_offload_block_hash / get_offload_group_idx
// (base.py:35-47). Thin aliases over the existing packing so there is exactly
// ONE implementation of the encoding in the tree.
inline OffloadKey make_offload_key(const BlockHash& block_hash,
                                   uint32_t group_idx) {
  return make_block_hash_with_group_id(block_hash, group_idx);
}
inline BlockHash get_offload_block_hash(const OffloadKey& key) {
  return get_block_hash(key);
}
inline uint32_t get_offload_group_idx(const OffloadKey& key) {
  return get_group_id(key);
}

// Upstream: class LookupResult (base.py:56-62).
enum class LookupResult : int {
  // Not present in this tier.
  kMiss = 0,
  // Present and READY to read.
  kHit = 1,
  // Present but a store is still in flight — NOT readable yet. Distinct from
  // kMiss: the block must not be re-stored.
  kHitPending = 2,
  // A promotion from a lower tier was started; ask again next step. Distinct
  // from kMiss: reporting a miss here would re-store a block already on its way
  // up.
  kRetry = 3,
};

// Upstream: class OffloadPolicy (base.py:65-71).
enum class OffloadPolicy : int {
  // Offload only newly-computed blocks; prefix hits (already offloaded by an
  // earlier request) are skipped.
  kBlockLevel = 0,
  // Offload every block of the request, including prefix hits.
  kRequestLevel = 1,
};

// Upstream: @dataclass ReqContext (base.py:50-53). kv_transfer_params is
// deferred with the connector config (KV-EXTERNAL-CACHE).
struct ReqContext {
  std::string req_id;
};

// Upstream: class LoadStoreSpec (base.py:88-101). Abstract metadata telling a
// worker where to load/store blocks.
class LoadStoreSpec {
 public:
  virtual ~LoadStoreSpec() = default;
  // A string naming the medium this spec targets ("CPU", "GPU", "FS").
  virtual const char* medium() const = 0;
};

// Upstream: class BlockIDsLoadStoreSpec (base.py:350-359).
class BlockIDsLoadStoreSpec : public LoadStoreSpec {
 public:
  explicit BlockIDsLoadStoreSpec(std::vector<int64_t> block_ids)
      : block_ids(std::move(block_ids)) {}
  std::vector<int64_t> block_ids;
};

// Upstream: class CPULoadStoreSpec (cpu/common.py:13-21).
class CPULoadStoreSpec : public BlockIDsLoadStoreSpec {
 public:
  using BlockIDsLoadStoreSpec::BlockIDsLoadStoreSpec;
  static const char* kMedium;
  const char* medium() const override { return kMedium; }
};

// Upstream: @dataclass PrepareStoreOutput (base.py:104-108).
struct PrepareStoreOutput {
  std::vector<OffloadKey> keys_to_store;
  std::shared_ptr<LoadStoreSpec> store_spec;
  std::vector<OffloadKey> evicted_keys;
};

// Upstream: @dataclass OffloadingEvent (base.py:111-116).
struct OffloadingEvent {
  std::vector<OffloadKey> keys;
  std::string medium;
  // True if the blocks were REMOVED, false if STORED.
  bool removed = false;
};

// Upstream: class OffloadingManager (base.py:177-347). The subset of the ABC
// the CPU and filesystem tiers need; the tiering/scheduler-facing hooks
// (on_new_request, on_schedule_end, has_pending_work, get_stats, shutdown) are
// owned by W4 and are not declared here so no caller can depend on an
// unimplemented contract.
class OffloadingManager {
 public:
  virtual ~OffloadingManager() = default;

  // Is this key present in this tier, and is it readable?
  virtual LookupResult lookup(const OffloadKey& key,
                              const ReqContext& req_context) = 0;

  // Pin the given (already-HIT) keys for reading and describe where they live.
  virtual std::shared_ptr<LoadStoreSpec> prepare_load(
      const std::vector<OffloadKey>& keys, const ReqContext& req_context) = 0;

  // Mark keys as recently used (recency/frequency bookkeeping only).
  virtual void touch(const std::vector<OffloadKey>& keys,
                     const ReqContext& req_context) = 0;

  // Release the read pins taken by prepare_load.
  virtual void complete_load(const std::vector<OffloadKey>& keys,
                             const ReqContext& req_context) = 0;

  // Reserve room for the given keys. Returns std::nullopt when the store must
  // be SKIPPED (eviction could not be satisfied) — a control path, NOT an
  // error. See the header note.
  virtual std::optional<PrepareStoreOutput> prepare_store(
      const std::vector<OffloadKey>& keys, const ReqContext& req_context) = 0;

  // Publish (success) or roll back (failure) a store reserved by prepare_store.
  virtual void complete_store(const std::vector<OffloadKey>& keys,
                              const ReqContext& req_context,
                              bool success = true) = 0;

  // Drop every cached block.
  virtual void reset_cache() = 0;

  // Drain the accumulated store/remove events (empty when events are off).
  virtual std::vector<OffloadingEvent> take_events() = 0;
};

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_BASE_H_
