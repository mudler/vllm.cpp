// Ported from: vllm/v1/kv_offload/cpu/manager.py @ e24d1b24
//
// Scope: `CPUOffloadingManager` — the PRIMARY offload tier. It owns all the
// logic shared by every replacement policy (ref-counting, the block pool,
// event emission, and the prepare_store/complete_store skeletons) and delegates
// only block organization and eviction choice to a `CachePolicy`
// (include/vllm/v1/kv_offload/cache_policy.h).
//
// TIER MODEL, a structural constraint rather than a detail
// (vllm/v1/kv_offload/tiering/base.py:52-64): SECONDARY tiers may never touch
// device memory. A store goes GPU -> CPU -> secondary and a load comes back
// secondary -> CPU -> GPU. That is why the CPU tier is built FIRST and the disk
// tier is reached only through it — building the disk tier standalone would
// mean a structure we then have to unpick.
//
// `store_threshold` (default 1) and the capped reuse tracker: with a threshold
// >= 2 a block is only stored once it has been LOOKED UP that many times, so a
// one-shot prompt does not evict a genuinely shared prefix. The tracker is an
// insertion-ordered map bounded by `max_tracker_size` (default 64000), which
// drops its least-recently-seen entry rather than growing without limit.
#ifndef VLLM_V1_KV_OFFLOAD_CPU_MANAGER_H_
#define VLLM_V1_KV_OFFLOAD_CPU_MANAGER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/cache_policy.h"

namespace vllm::v1::kv_offload {

// Upstream: OffloadingConnectorStats' CPU gauges (cpu/common.py:7-10). We have
// no Prometheus route yet (SERVE-METRICS), so the numbers are returned
// directly.
struct CPUOffloadingStats {
  // Fraction of the tier's blocks currently holding non-evictable data.
  double cpu_cache_usage = 0.0;
  // Stores skipped by store_threshold since the last get_stats() call.
  int64_t stores_skipped = 0;
};

class CPUOffloadingManager final : public OffloadingManager {
 public:
  // `cache_policy` is "lru" or "arc" (make_cache_policy). Upstream signature:
  // CPUOffloadingManager(num_blocks, cache_policy="lru", enable_events=False,
  //                      store_threshold=1, max_tracker_size=64_000)
  // (cpu/manager.py:46-53). NOTE there is deliberately no `num_cpu_blocks`
  // flag upstream — the count is DERIVED from a byte budget by the spec layer.
  explicit CPUOffloadingManager(int64_t num_blocks,
                                const std::string& cache_policy = "lru",
                                bool enable_events = false,
                                int64_t store_threshold = 1,
                                int64_t max_tracker_size = 64000);
  ~CPUOffloadingManager() override;

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

  // Take (and reset the per-call counters of) the tier statistics.
  CPUOffloadingStats get_stats();

  // Total configured block count.
  int64_t num_blocks() const;
  // Blocks that could still be handed out without evicting anything.
  int64_t num_free_blocks() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_CPU_MANAGER_H_
