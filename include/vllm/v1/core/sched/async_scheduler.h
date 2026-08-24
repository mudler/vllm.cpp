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
// SPEC-DFLASH2 W7 (#1824) un-deferred the spec-decode placeholders: with a
// SpeculativeConfig threaded through the ctor, update_after_schedule assigns
// request.spec_token_ids = [-1] * num_spec_tokens_to_schedule after each
// schedule (async_scheduler.py:24-25,43-45) — the NEXT step schedules those
// placeholders as 1+k tokens and the WORKER substitutes the real drafts it
// kept from its own propose ("We will update the actual spec token ids in the
// worker process"). Guarded on num_spec_tokens_to_schedule > 0, so the
// no-speculator path stays byte-identical.
//
// DEFERRED (T0, matches upstream structure so re-adding is mechanical):
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
  // Same parameters as the base Scheduler. speculative_config (W7 #1824) makes
  // the async engine's spec plumbing live: num_lookahead_tokens, the running-
  // loop spec budget, and num_spec_tokens_to_schedule for the placeholder
  // assignment below. std::nullopt (the default) is the production
  // no-speculator path, byte-identical to the pre-W7 ctor.
  AsyncScheduler(SchedulerConfig scheduler_config, KVCacheConfig kv_cache_config,
                 int block_size, bool enable_caching = false,
                 StructuredOutputManager* structured_output_manager = nullptr,
                 std::optional<SpeculativeConfig> speculative_config =
                     std::nullopt)
      : Scheduler(std::move(scheduler_config), std::move(kv_cache_config),
                  block_size, enable_caching, structured_output_manager,
                  std::move(speculative_config)) {}

  // The async-scheduling class answers true (read by EngineCore::post_step to
  // skip the out-of-band draft pull — core.py:617; see the base declaration).
  bool async_scheduling() const override { return true; }

 protected:
  // async_scheduler.py:19-49.
  void update_after_schedule(SchedulerOutput& scheduler_output) override;

  // async_scheduler.py:51-75.
  std::pair<std::vector<int32_t>, bool> update_request_with_output(
      Request& request, std::vector<int32_t> new_token_ids) override;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_SCHED_ASYNC_SCHEDULER_H_
