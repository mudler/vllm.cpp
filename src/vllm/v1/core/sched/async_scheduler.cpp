// Ported from: vllm/v1/core/sched/async_scheduler.py @ e24d1b24
// See include/vllm/v1/core/sched/async_scheduler.h for scope + deferred list.
#include "vllm/v1/core/sched/async_scheduler.h"

#include <cassert>
#include <utility>

namespace vllm::v1 {

void AsyncScheduler::update_after_schedule(SchedulerOutput& scheduler_output) {
  // async_scheduler.py:19-49. Advance the base bookkeeping first (num_computed,
  // is_prefill_chunk with the CURRENT placeholders, has_structured_output_reqs,
  // flush finished/preempted sets), then add this step's placeholders.
  Scheduler::update_after_schedule(scheduler_output);

  // spec_decode_tokens is empty at T0 (no draft tokens), and
  // num_spec_tokens_to_schedule == 0, so the _spec_token_placeholders reset and
  // the per-request spec_token_ids assignment are inert and omitted.
  for (const auto& [req_id, num_scheduled] :
       scheduler_output.num_scheduled_tokens) {
    (void)num_scheduled;
    Request* request = requests.at(req_id).get();
    if (request->is_prefill_chunk) {
      continue;
    }

    // async_scheduler.py:31-33: a structured request whose previous step's
    // tokens are still in flight (placeholders outstanding) must defer sampling
    // this step. Read BEFORE incrementing (upstream checks num_output_
    // placeholders > 0 prior to the += below).
    scheduler_output.pending_structured_output_tokens |=
        (request->use_structured_output() &&
         request->num_output_placeholders > 0);

    // async_scheduler.py:38-41: reserve num_sampled_tokens_per_step placeholders
    // for the token(s) this step samples (+ cur_num_spec_tokens == 0 at T0).
    request->num_output_placeholders += num_sampled_tokens_per_step();

    // async_scheduler.py:46-49 (next_decode_eligible_step, PP microbatching):
    // pp_size == 1 at T0, so next_decode_eligible_step stays current_step + 1
    // which never gates a single-GPU decode; left inert (0) here.
  }
}

std::pair<std::vector<int32_t>, bool> AsyncScheduler::update_request_with_output(
    Request& request, std::vector<int32_t> new_token_ids) {
  // async_scheduler.py:54-59: drop one stale in-flight frame per call while the
  // request is draining a force-preemption (reset_prefix_cache) — the frames
  // that were in flight at preemption time are now invalid.
  if (request.async_tokens_to_discard > 0) {
    request.async_tokens_to_discard -= 1;
    return {std::vector<int32_t>{}, false};
  }

  const RequestStatus status_before_update = request.status;

  // async_scheduler.py:62-64: run the base append/check_stop/trim.
  auto result =
      Scheduler::update_request_with_output(request, std::move(new_token_ids));
  std::vector<int32_t> kept = std::move(result.first);
  const bool stopped = result.second;

  // async_scheduler.py:66-68: drain the placeholder count by the accepted tokens.
  request.num_output_placeholders -= static_cast<int>(kept.size());
  assert(request.num_output_placeholders >= 0);

  // async_scheduler.py:70-74: cache the request's blocks up to the confirmed
  // (non-placeholder) computed tokens. Preempted requests are skipped (their KV
  // was already freed at preemption; caching a preempted request would reference
  // freed blocks).
  if (status_before_update == RequestStatus::kRunning) {
    kv_cache_manager->cache_blocks(
        request, request.num_computed_tokens - request.num_output_placeholders);
  }
  return {std::move(kept), stopped};
}

}  // namespace vllm::v1
