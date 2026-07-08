// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (add_request, abort_requests, step — the T0 subset). See core.h for scope,
// deviations and deferrals.
#include "vllm/v1/engine/core.h"

#include <optional>

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
  // core.py:221-222 self.step_fn = self.step if self.batch_queue is None else
  // self.step_with_batch_queue. VT_ASYNC_DECODE selects the depth-2 pipeline; OFF
  // is the exact synchronous path below (byte-identical, no queue).
  if (AsyncDecodeEnabled()) {
    return step_async();
  }
  return step_sync();
}

std::pair<std::map<int, EngineCoreOutputs>, bool> EngineCore::step_sync() {
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

// ─── Async depth-2 pipeline (VT_ASYNC_DECODE) ───────────────────────────────
// Ported from core.py:519-632 (step_with_batch_queue) + has_work (core.py:1247-
// 1253, the `bool(self.batch_queue)` drain term, surfaced via has_pending()).

bool EngineCore::is_overlappable_decode(const SchedulerOutput& scheduler_output) {
  // Only a pure-decode continuation of the SAME request set may be deferred: no
  // admission (scheduled_new_reqs, which also carries resumed-as-new reqs), no
  // finished-request removal (would reshape the InputBatch), no structured output
  // (its FSM must advance before the next schedule), and every scheduled request
  // exactly one token (total == num reqs, and > 0). Otherwise the step reads the
  // host token buffer and must see all prior deferred write-backs applied, so the
  // caller drains + finalizes it in line instead of deferring.
  if (!scheduler_output.scheduled_new_reqs.empty()) return false;
  if (!scheduler_output.finished_req_ids.empty()) return false;
  if (scheduler_output.has_structured_output_requests) return false;
  const int num_reqs =
      static_cast<int>(scheduler_output.num_scheduled_tokens.size());
  if (num_reqs == 0) return false;
  return scheduler_output.total_num_scheduled_tokens == num_reqs;
}

void EngineCore::finalize_and_update(
    PendingStep& step, std::map<int, EngineCoreOutputs>& out) {
  // core.py:595 model_output = future.result() (sync the oldest pending's event),
  // core.py:604-606 engine_core_outputs = update_from_output(...).
  ModelRunnerOutput model_output = executor_.finalize_output(step.pending);
  EngineCoreOutputs eco =
      scheduler_.update_from_output(step.scheduler_output, model_output);
  if (eco.outputs.empty()) {
    return;  // 0-output step (flush / all-speculative-discard): no map entry.
  }
  const int client_index = eco.engine_index;
  auto it = out.find(client_index);
  if (it == out.end()) {
    out.emplace(client_index, std::move(eco));
  } else {
    // Merge (a drain can return more than one step's outputs in one call). The
    // per-request EngineCoreOutputs are appended in FIFO step order, so the
    // OutputProcessor detokenizes each request's tokens in generation order.
    for (EngineCoreOutput& o : eco.outputs) {
      it->second.outputs.push_back(std::move(o));
    }
  }
}

std::pair<std::map<int, EngineCoreOutputs>, bool> EngineCore::step_async() {
  // Invariant (core.py:542): entered only with the queue below capacity.
  std::map<int, EngineCoreOutputs> outputs_by_client;
  const bool has_requests = scheduler_.get_num_unfinished_requests() > 0 ||
                            scheduler_.has_finished_requests();

  bool model_executed = false;
  if (has_requests) {
    // core.py:547 scheduler_output = self.scheduler.schedule(...).
    SchedulerOutput scheduler_output = scheduler_.schedule();
    model_executed = scheduler_output.total_num_scheduled_tokens > 0;
    // core.py:562 grammar_output = self.scheduler.get_grammar_bitmask(...).
    const std::optional<GrammarOutput> grammar_output =
        scheduler_.get_grammar_bitmask(scheduler_output);

    if (!is_overlappable_decode(scheduler_output)) {
      // Not a deferrable pure-decode step (prefill / chunk / reshape / flush). To
      // keep the host token buffer consistent, DRAIN every in-flight pending
      // (applying its deferred write-back) BEFORE running this step, then finalize
      // this step in line (allow_defer=false). This preserves token-exactness at
      // prefill/finish boundaries; only the pure-decode interior overlaps.
      while (!batch_queue_.empty()) {
        PendingStep front = std::move(batch_queue_.front());
        batch_queue_.pop_front();
        finalize_and_update(front, outputs_by_client);
      }
      PendingModelOutput pending = executor_.execute_and_sample_async(
          scheduler_output, grammar_output, /*allow_defer=*/false);
      PendingStep step{std::move(scheduler_output), std::move(pending)};
      finalize_and_update(step, outputs_by_client);
      return {std::move(outputs_by_client), model_executed};
    }

    // core.py:549 exec_future = execute_model(non_block=True); core.py:565
    // future = sample_tokens(non_block=True). Deferred: device tokens + recorded
    // (not synced) readback event, pushed onto the queue.
    PendingModelOutput pending = executor_.execute_and_sample_async(
        scheduler_output, grammar_output, /*allow_defer=*/true);
    // A pending that is already host-ready has NO readback to overlap (a runner
    // that does not defer, e.g. the CPU-only / stub path). When the queue is
    // empty there is also no earlier in-flight step whose FIFO order it must
    // follow, so process it now — restoring synchronous per-step semantics with
    // no loss of overlap (a truly deferred device pending is ready==false and
    // still pipelines below). This keeps a non-deferring runner's step() timing
    // identical to the synchronous engine.
    if (pending.ready && batch_queue_.empty()) {
      PendingStep step{std::move(scheduler_output), std::move(pending)};
      finalize_and_update(step, outputs_by_client);
      return {std::move(outputs_by_client), model_executed};
    }
    // core.py:575 batch_queue.appendleft((future, scheduler_output, exec_future)).
    batch_queue_.push_back(
        PendingStep{std::move(scheduler_output), std::move(pending)});
    // core.py:576-581 don't block on the next result unless the queue is full or
    // there is no more work to schedule.
    const bool more_work =
        model_executed || scheduler_.get_num_unfinished_requests() > 0 ||
        scheduler_.has_finished_requests();
    if (batch_queue_.size() < kBatchQueueSize && more_work) {
      return {std::move(outputs_by_client), model_executed};
    }
  } else if (batch_queue_.empty()) {
    // core.py:583-587 queue empty and nothing scheduled — nothing to do.
    return {std::move(outputs_by_client), false};
  }

  // core.py:590-606 block on the OLDEST pending and update_from_output it.
  PendingStep front = std::move(batch_queue_.front());
  batch_queue_.pop_front();
  finalize_and_update(front, outputs_by_client);
  return {std::move(outputs_by_client), model_executed};
}

}  // namespace vllm::v1
