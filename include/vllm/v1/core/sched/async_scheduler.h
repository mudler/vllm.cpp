// Ported from: vllm/v1/core/sched/async_scheduler.py @ e24d1b24
// (class AsyncScheduler(Scheduler) :12-75).
//
// Scope (async-serving spec W3, row ENG-ASYNC-SCHED): the scheduler half of
// async/overlap scheduling. AsyncScheduler subclasses the synchronous Scheduler
// and adds ONLY the output-placeholder accounting that lets the engine schedule
// step N+1 before step N's sampled tokens have returned:
//   * update_after_schedule (async_scheduler.py:19-49): after the base advance,
//     reserve num_sampled_tokens_per_step output placeholders for each scheduled,
//     non-prefill-chunk running request (the in-flight token this step samples),
//     and flag pending_structured_output_tokens when a structured request still
//     has placeholders outstanding.
//   * update_request_with_output (async_scheduler.py:51-75): when a step's output
//     returns, first drop one stale frame per force-preemption (async_tokens_to_
//     discard), then run the base append/check_stop, drain the placeholder count
//     by the number of accepted tokens, and cache the request's blocks up to
//     (num_computed_tokens - num_output_placeholders).
//
// The base Scheduler already reads num_output_placeholders in its running-loop
// budget formula + max_tokens guard and in is_prefill_chunk (all INERT while the
// count is 0, i.e. under the synchronous Scheduler), so subclassing is the only
// delta — no base-class behavior changes for the sync path.
//
// DEFERRED (T0, matches upstream structure so re-adding is mechanical):
//   - spec-decode placeholders (_spec_token_placeholders / spec_token_ids;
//     num_spec_tokens == 0 at T0 so no draft placeholders are added),
//   - next_decode_eligible_step PP-microbatch cadence (pp_size == 1 → the base
//     guard stays inert),
//   - the diffusion num_sampled_tokens_per_step == 0 path (T0 is autoregressive,
//     so it is 1).
#ifndef VLLM_V1_CORE_SCHED_ASYNC_SCHEDULER_H_
#define VLLM_V1_CORE_SCHED_ASYNC_SCHEDULER_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "vllm/v1/core/sched/scheduler.h"

namespace vllm::v1 {

// AsyncScheduler(Scheduler): the overlap-scheduling scheduler. Constructed with
// the same arguments as the base Scheduler (the engine picks this class instead
// of Scheduler when async scheduling resolves ON — SchedulerConfig::
// ResolveAsyncScheduling, mirroring get_scheduler_cls at scheduler.py:180-189).
class AsyncScheduler : public Scheduler {
 public:
  AsyncScheduler(SchedulerConfig scheduler_config, KVCacheConfig kv_cache_config,
                 int block_size, bool enable_caching = false,
                 StructuredOutputManager* structured_output_manager = nullptr)
      : Scheduler(std::move(scheduler_config), std::move(kv_cache_config),
                  block_size, enable_caching, structured_output_manager) {}

 protected:
  // async_scheduler.py:19-49.
  void update_after_schedule(SchedulerOutput& scheduler_output) override;

  // async_scheduler.py:51-75.
  std::pair<std::vector<int32_t>, bool> update_request_with_output(
      Request& request, std::vector<int32_t> new_token_ids) override;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_SCHED_ASYNC_SCHEDULER_H_
