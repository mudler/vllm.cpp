// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (add_request, abort_requests, step — the T0 subset). See core.h for scope,
// deviations and deferrals.
#include "vllm/v1/engine/core.h"

#include <optional>

namespace vllm::v1 {

void EngineCore::add_request(std::unique_ptr<Request> request) {
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
  // execute_model returns None ("forward done"), so we always call
  // sample_tokens; grammar_output is null at T0 (structured output deferred).
  std::optional<ModelRunnerOutput> model_output =
      executor_.execute_model(scheduler_output);
  if (!model_output.has_value()) {
    model_output = executor_.sample_tokens();
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

}  // namespace vllm::v1
