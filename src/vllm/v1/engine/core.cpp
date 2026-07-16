// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (add_request, abort_requests, step — the T0 subset). See core.h for scope,
// deviations and deferrals.
#include "vllm/v1/engine/core.h"

#include <cassert>
#include <optional>
#include <utility>

#include "vllm/v1/core/sched/output.h"          // GrammarOutput
#include "vllm/v1/structured_output/manager.h"  // StructuredOutputManager

namespace vllm::v1 {

void EngineCore::add_request(std::unique_ptr<Request> request) {
  // core.py:870-876 (preprocess_add_request): compile the request's grammar
  // before scheduling it. Upstream runs this in the input-processing thread; at
  // T0 it is synchronous here. No-op for a non-structured request or when no
  // manager is wired.
  if (structured_output_manager_ != nullptr &&
      request->use_structured_output()) {
    structured_output_manager_->grammar_init(*request);
  }
  // core.py:403 self.scheduler.add_request(request). The upstream request_id
  // type / pooling-task / kv_transfer validation and the abort_immediately
  // hook are deferred (see the file header).
  scheduler_.add_request(std::move(request));
}

void EngineCore::abort_requests(const std::vector<std::string>& request_ids) {
  // core.py:415 self.scheduler.finish_requests(request_ids,
  // RequestStatus.FINISHED_ABORTED). Our finish_requests takes one id, so
  // iterate (a no-op for unknown / already-finished ids, matching upstream).
  for (const std::string& request_id : request_ids) {
    scheduler_.finish_requests(request_id, RequestStatus::kFinishedAborted);
  }
}

std::pair<std::map<int, EngineCoreOutputs>, bool> EngineCore::step() {
  // core.py:488 if not self.scheduler.has_requests(): return {}, False
  // Our Scheduler has no has_requests(); inline the interface.py default
  // (connector pending-push term deferred — no connector at T0).
  const bool has_requests = scheduler_.get_num_unfinished_requests() > 0 ||
                            scheduler_.has_finished_requests();
  if (!has_requests) {
    return {{}, false};
  }

  // core.py:490 scheduler_output = self.scheduler.schedule(...)
  // (the _should_throttle_prefills() arg is deferred — DP prefill balancing).
  SchedulerOutput scheduler_output = scheduler_.schedule();

  // core.py:491-499 execute the forward, then sample. The MRV2 runner's
  // execute_model returns None ("forward done"), so we always call sample_tokens.
  std::optional<ModelRunnerOutput> model_output =
      executor_.execute_model(scheduler_output);
  // core.py:492 grammar_output = self.scheduler.get_grammar_bitmask(...). Nullopt
  // when no structured request is scheduled (or no manager is wired); threaded to
  // sample_tokens (Task 3 consumes it).
  const std::optional<GrammarOutput> grammar_output =
      scheduler_.get_grammar_bitmask(scheduler_output);
  if (!model_output.has_value()) {
    model_output = executor_.sample_tokens(grammar_output);
  }

  // core.py:503 self._process_aborts_queue() — deferred (no in-flight aborts at
  // T0; synchronous execution leaves no window).

  // core.py:504-506 engine_core_outputs = scheduler.update_from_output(...).
  // Our update_from_output returns a single flat EngineCoreOutputs (T0 single
  // client); wrap it in the per-client map to keep the upstream return shape.
  // Upstream builds dict[client_index, EngineCoreOutputs] only for clients that
  // produced outputs this step (the dict comprehension over `outputs.items()`),
  // so a 0-output step (e.g. a finished-req flush) yields an empty map. We drop
  // the finished_requests-only entries (that DP-signalling field is deferred),
  // so the entry is present iff there are token outputs.
  EngineCoreOutputs engine_core_outputs =
      scheduler_.update_from_output(scheduler_output, *model_output);
  std::map<int, EngineCoreOutputs> outputs_by_client;
  if (!engine_core_outputs.outputs.empty()) {
    const int client_index = engine_core_outputs.engine_index;
    outputs_by_client.emplace(client_index, std::move(engine_core_outputs));
  }

  // core.py:508 return outputs, scheduler_output.total_num_scheduled_tokens > 0.
  return {std::move(outputs_by_client),
          scheduler_output.total_num_scheduled_tokens > 0};
}

std::pair<std::map<int, EngineCoreOutputs>, bool>
EngineCore::step_with_batch_queue() {
  // core.py:519-632. Fulfilling the batch queue has priority over consuming an
  // output: schedule + execute a new batch (if room + work), then block on the
  // OLDEST queued batch. Our executor resolves eagerly, so "execute" here means
  // the forward + sample ran synchronously and the ModelRunnerOutput is already
  // in hand (the AsyncScheduler placeholder accounting is what lets step N+1 be
  // scheduled before N's output is consumed — see core.h).
  assert(batch_queue_.size() < static_cast<std::size_t>(batch_queue_size_) &&
         "step_with_batch_queue called with a full batch queue");

  bool model_executed = false;
  std::optional<SchedulerOutput> deferred_scheduler_output;

  const bool has_requests = scheduler_.get_num_unfinished_requests() > 0 ||
                            scheduler_.has_finished_requests();
  if (has_requests) {
    // core.py:547 schedule (non-blocking; may return an empty batch).
    SchedulerOutput scheduler_output = scheduler_.schedule();
    // core.py:549-551 execute_model(non_block=True). MRV2: nullopt (forward
    // done). A failed eager forward throws here, through the engine-fatal guard.
    std::optional<ModelRunnerOutput> exec_out =
        executor_.execute_model(scheduler_output);
    // core.py:552-553 model_executed = total_num_scheduled_tokens > 0
    // (is_ec_consumer is always true for us — no EC transfer).
    model_executed = scheduler_output.total_num_scheduled_tokens > 0;

    ModelRunnerOutput model_output;
    bool have_output = false;
    if (!model_executed) {
      // core.py:555-557 no sampling required — carry the (empty) forward result.
      model_output = exec_out.value_or(ModelRunnerOutput{});
      have_output = true;
    } else if (!scheduler_output.pending_structured_output_tokens) {
      // core.py:559-567 not waiting on any tokens — get the grammar bitmask and
      // sample immediately.
      const std::optional<GrammarOutput> grammar_output =
          scheduler_.get_grammar_bitmask(scheduler_output);
      model_output = executor_.sample_tokens(grammar_output);
      have_output = true;
    } else {
      // core.py:568-571 defer sampling until the prior step's output is
      // processed (a structured request is waiting on in-flight tokens).
      deferred_scheduler_output = scheduler_output;
    }

    if (!deferred_scheduler_output.has_value()) {
      // core.py:573-575 add this step's result to the queue (front).
      assert(have_output);
      (void)have_output;
      batch_queue_.push_front(
          BatchQueueItem{std::move(model_output), std::move(scheduler_output)});
      // core.py:576-581 don't block on the next result unless the queue is full
      // or there are no more requests to schedule.
      const bool more_requests =
          scheduler_.get_num_unfinished_requests() > 0 ||
          scheduler_.has_finished_requests();
      if (batch_queue_.size() < static_cast<std::size_t>(batch_queue_size_) &&
          (model_executed || more_requests)) {
        return {{}, model_executed};
      }
    }
  } else if (batch_queue_.empty()) {
    // core.py:583-587 queue empty and no requests — nothing to do.
    return {{}, false};
  }

  // core.py:589-590 block until the next result is available (pop the OLDEST).
  BatchQueueItem item = std::move(batch_queue_.back());
  batch_queue_.pop_back();

  // core.py:604-607 update the scheduler from the popped batch's output.
  EngineCoreOutputs engine_core_outputs =
      scheduler_.update_from_output(item.scheduler_output, item.model_output);

  // core.py:609-630 grammar deferral: now that the prior output is processed,
  // compute the deferred batch's bitmask, sample it, and append it to the queue.
  // (The runner's execute_model stash from this step is still valid — no other
  // execute_model ran between it and here.)
  if (deferred_scheduler_output.has_value()) {
    const std::optional<GrammarOutput> grammar_output =
        scheduler_.get_grammar_bitmask(*deferred_scheduler_output);
    ModelRunnerOutput sampled = executor_.sample_tokens(grammar_output);
    batch_queue_.push_front(
        BatchQueueItem{std::move(sampled), std::move(*deferred_scheduler_output)});
  }

  std::map<int, EngineCoreOutputs> outputs_by_client;
  if (!engine_core_outputs.outputs.empty()) {
    const int client_index = engine_core_outputs.engine_index;
    outputs_by_client.emplace(client_index, std::move(engine_core_outputs));
  }
  // core.py:632 return engine_core_outputs, model_executed.
  return {std::move(outputs_by_client), model_executed};
}

}  // namespace vllm::v1
