// Ported from: vllm/v1/kv_offload/tiering/manager.py:123-706 @ e24d1b24
//               vllm/v1/kv_offload/tiering/base.py:52-64 (the tier model)
//
// Scope (KV-OFFLOAD W4): the TIERING MANAGER — ONE manager that coordinates the
// CPU primary tier (W2 CPUOffloadingManager) and the disk secondary tier
// (W3 FileSystemTier) with PROMOTION (disk -> CPU on a hit) and DEMOTION/cascade
// (CPU -> disk on a store). It implements the same OffloadingManager surface the
// CPU tier does, so the connector/scheduler half (kv_connector.h) drives ONE
// object without knowing how many tiers sit behind it.
//
// CORRECTED RECORD (the caching spike found "KV-OFFLOAD" cited an
// `LRUOffloadingManager` that does not exist): LRU/ARC are pluggable
// `CachePolicy` objects behind the ONE CPU manager; the tiering layer is a
// SEPARATE manager that OWNS a primary manager plus N secondary tiers. This file
// is that separate manager.
//
// THE TIER MODEL IS A STRUCTURAL CONSTRAINT, NOT A DETAIL
// (tiering/base.py:52-64): a secondary tier may NEVER touch device memory. Every
// byte moves GPU <-> CPU (primary) <-> secondary. So the disk tier is reached
// ONLY through the CPU tier's pinned backing store, addressed by CPU block id.
// The manager hands the secondary tier a byte view of the primary store
// (PrimaryByteView, mirroring upstream's `primary_kv_view` memoryview,
// tiering/base.py:66-81) and the secondary reads/writes it by slot id.
//
// THE ORDERING IS THE CORRECTNESS CONTENT (mirrored line-for-line; see the .cpp
// for the upstream anchors):
//
//  * PROMOTION (manager.py:282-353). A lookup that MISSES the primary but HITS a
//    secondary does NOT load inline. It: (1) `prepare_write`s a primary slot,
//    which sets ref_cnt == -1 so the slot reads as HIT_PENDING to any later
//    lookup THIS step and cannot be promoted twice; (2) defers a batched
//    `submit_load` into `pending_promotions_`, flushed only at
//    `on_schedule_end`; (3) returns RETRY, NOT a hit. The scheduler re-asks next
//    step, by which point the load has completed and the primary reports HIT.
//    Collapsing RETRY into MISS re-stores a block already on its way up;
//    collapsing it into HIT reads a slot that has no bytes yet.
//
//  * CASCADE / demotion (manager.py:497-556). On `complete_store` success the
//    manager `prepare_read`s the freshly-stored primary blocks (ref_cnt++ to pin
//    them) and `submit_store`s them to every secondary tier. The pin is released
//    (`complete_read`) only when that job is observed finished at a later
//    `_process_finished_jobs`, never inline — so a block being written to disk
//    cannot be evicted from CPU underneath the writer.
//
//  * RESET ORDERING (manager.py:642-681). `reset_cache` drains the secondary
//    tiers FIRST (a stuck tier blocks visibly rather than corrupting the
//    primary), processes finished jobs, clears the deferred promotions (their
//    reserved primary slots are about to be invalidated), and only THEN resets
//    the primary. The secondary tiers are DELIBERATELY NOT reset: a persistent
//    disk store survives a prefix-cache reset. That is the whole point of a
//    persisted tier.
//
// DEVIATION, recorded: upstream's secondary tiers are fully asynchronous
// (`submit_load`/`submit_store` return immediately; `get_finished_jobs` polls).
// Our FileSystemTier already runs its IO on a background dual-queue pool, but its
// only completion primitive is a blocking `wait(job_id)`. The manager therefore
// SUBMITS at flush/cascade time (async, off the scheduler thread) and FINALIZES
// by `wait`-ing at the next `process_finished_jobs()` — which is exactly the
// step boundary at which upstream expects the blocks to have become ready. The
// observable RETRY-then-HIT promotion semantics (the load-bearing part) are
// preserved; only the poll is a blocking wait instead of a non-blocking query.
#ifndef VLLM_V1_KV_OFFLOAD_TIERING_MANAGER_H_
#define VLLM_V1_KV_OFFLOAD_TIERING_MANAGER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/cpu_manager.h"
#include "vllm/v1/kv_offload/fs_tier.h"

namespace vllm::v1::kv_offload {

// A byte view over the primary CPU tier's pinned backing store. Block
// `cpu_block_id` occupies [slot(id), slot(id) + page_size_bytes). Mirrors
// upstream's `primary_kv_view` memoryview handed to secondary tiers
// (tiering/base.py:66-81): it is how a disk tier addresses "the bytes of CPU
// slot b" without ever touching device memory.
//
// The manager treats the payload as OPAQUE page_size_bytes (§Risks R9): one code
// path serves full attention (rank-4, interleaved [K|V]) and MLA (rank-3, one
// 576-wide latent, no V) because no layout is interpreted at this layer.
class PrimaryByteView {
 public:
  virtual ~PrimaryByteView() = default;
  virtual void* slot(int64_t cpu_block_id) = 0;
  virtual const void* slot(int64_t cpu_block_id) const = 0;
  virtual std::size_t page_size_bytes() const = 0;
};

// A plain heap-backed PrimaryByteView for CPU development and tests (the
// host-resident KV path, VT_DEVICE_KV_CACHE=0). Production backs this with the
// pinned CPUBackingStore (kv_block_transfer.h); the manager does not care which.
class HeapPrimaryByteView final : public PrimaryByteView {
 public:
  HeapPrimaryByteView(int64_t num_blocks, std::size_t page_size_bytes);
  void* slot(int64_t cpu_block_id) override;
  const void* slot(int64_t cpu_block_id) const override;
  std::size_t page_size_bytes() const override { return page_size_bytes_; }

 private:
  std::size_t page_size_bytes_ = 0;
  std::vector<std::uint8_t> data_;
};

struct TieringStats {
  // Fraction of primary-tier blocks currently holding non-evictable data.
  double cpu_cache_usage = 0.0;
  // Disk-tier residency.
  int64_t fs_num_blocks = 0;
  int64_t fs_bytes_used = 0;
  int64_t fs_num_evicted = 0;
  // Promotions (disk -> CPU) and cascades (CPU -> disk) completed since
  // construction.
  int64_t promotions = 0;
  int64_t cascades = 0;
  // Promotions that REFUSED a foreign/corrupt disk block (§B2 safety evidence).
  // A refused block is treated as ABSENT — never trusted.
  int64_t refusals = 0;
};

// ONE manager over a CPU primary tier and a single disk secondary tier. (Upstream
// supports N secondary tiers; we ship the one that answers the user's request —
// the disk tier — and the seam generalizes to N without a shape change.)
class TieringOffloadingManager final : public OffloadingManager {
 public:
  // `primary_view` must outlive the manager; it is the byte home the disk tier
  // reads/writes by CPU slot id.
  TieringOffloadingManager(std::unique_ptr<CPUOffloadingManager> primary,
                           std::unique_ptr<FileSystemTier> secondary,
                           PrimaryByteView& primary_view);
  ~TieringOffloadingManager() override;

  // --- OffloadingManager surface (drives promotion/cascade) ------------------
  LookupResult lookup(const OffloadKey& key,
                      const ReqContext& req_context) override;
  std::shared_ptr<LoadStoreSpec> prepare_load(
      const std::vector<OffloadKey>& keys,
      const ReqContext& req_context) override;
  void touch(const std::vector<OffloadKey>& keys,
             const ReqContext& req_context) override;
  void complete_load(const std::vector<OffloadKey>& keys,
                     const ReqContext& req_context) override;
  std::optional<PrepareStoreOutput> prepare_store(
      const std::vector<OffloadKey>& keys,
      const ReqContext& req_context) override;
  void complete_store(const std::vector<OffloadKey>& keys,
                      const ReqContext& req_context,
                      bool success = true) override;
  void reset_cache() override;
  std::vector<OffloadingEvent> take_events() override;

  // --- tiering/scheduler-facing hooks (manager.py:558-706) -------------------
  // Register a request; returns nothing observable today (no request-level
  // tiers), but the per-request state it creates gates cascade finalization.
  void on_new_request(const ReqContext& req_context);
  // A request has left the scheduler; finalize once its stores drain.
  void on_request_finished(const ReqContext& req_context);
  // Once per step: process finished jobs, reset the per-step gate, and FLUSH the
  // deferred promotion loads. THIS is where a promotion's disk read actually
  // happens; the block is HIT at the next step's lookup.
  void on_schedule_end();
  // True while any promotion/cascade job or deferred promotion is outstanding.
  bool has_pending_work() const;

  TieringStats get_stats();

  // Test/observability: force-drain every outstanding job. Public because the
  // restart-hit correctness test drives promotion to completion deterministically
  // rather than across simulated steps.
  void drain_jobs();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_TIERING_MANAGER_H_
