// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (EngineCoreProc busy loop: run_busy_loop :1259-1266, _process_input_queue
// :1269-1298, _process_engine_step :1300-1318, _handle_shutdown :1324-1370,
// _handle_client_request :1372-1404, _reject_add_in_shutdown :1407-1416,
// _send_finish_outputs_to_client :1714-1722, _send_abort_outputs :1734-1742,
// _send_engine_dead :1470-1480). See core_proc.h for scope, deviations and
// deferrals.
#include "vllm/v1/engine/core_proc.h"

#include <chrono>
#include <stdexcept>
#include <thread>

#include "vllm/v1/core/sched/scheduler.h"

namespace vllm::v1 {

EngineCoreProc::EngineCoreProc(Scheduler& scheduler, Executor& executor,
                               StructuredOutputManager* structured_output_manager,
                               int max_concurrent_batches, int shutdown_timeout_s)
    : EngineCore(scheduler, executor, structured_output_manager),
      shutdown_timeout_s_(shutdown_timeout_s) {
  // core.py:196-223: batch_queue_size = max_concurrent_batches; the batch queue
  // is enabled only when > 1, which flips step_fn to step_with_batch_queue.
  // vllm/config/vllm.py:490-501 returns 2 under async scheduling on a single GPU
  // (pp_size + 1). ENG-ASYNC-SCHED (spec W3) implements step_with_batch_queue,
  // so > 1 is now accepted (was rejected at W1). Depth must be >= 1.
  if (max_concurrent_batches < 1) {
    throw std::invalid_argument(
        "EngineCoreProc: max_concurrent_batches must be >= 1");
  }
  batch_queue_size_ = max_concurrent_batches;
  // core.py:221-223: step_fn = step if batch_queue is None else
  // step_with_batch_queue.
  step_fn_ = (batch_queue_size_ > 1) ? &EngineCore::step_with_batch_queue
                                     : &EngineCore::step;
}

bool EngineCoreProc::has_work() const {
  // core.py:1247-1253: engines_running (DP, deferred) or
  // scheduler.has_requests() or bool(batch_queue). Our Scheduler does not port
  // interface.py has_requests(); inline its default exactly as EngineCore::step
  // does (core.h DEVIATIONS). The batch-queue term keeps the loop stepping to
  // drain queued batches after the scheduler runs dry (ENG-ASYNC-SCHED W3).
  return scheduler_.get_num_unfinished_requests() > 0 ||
         scheduler_.has_finished_requests() || has_batch_queue_work();
}

bool EngineCoreProc::is_running() const {
  // core.py:1255-1257.
  return shutdown_state.load() == EngineShutdownState::kRunning;
}

void EngineCoreProc::run_busy_loop() {
  // core.py:1259-1266. Upstream ends with `raise SystemExit`, caught by
  // run_engine_core; here the loop simply returns and the engine thread ends.
  while (handle_shutdown()) {
    // 1) Poll the input queue until there is work to do.
    process_input_queue();
    // 2) Step the engine core and return the outputs.
    process_engine_step();
  }
}

void EngineCoreProc::process_input_queue() {
  // core.py:1269-1298. "Exits when an engine step needs to be performed."
  while (!has_work() && is_running()) {
    // core.py:1273 _notify_idle_state_callbacks() — deferred (pause/sleep).
    // core.py:1275-1278 aborts_queue drain — deferred with the aborts_queue
    // (aborts arrive via the input queue only at W1).
    const bool block = process_input_queue_block_;
    EngineCoreInputItem item;
    if (block) {
      // core.py:1283-1284 req = self.input_queue.get(block=True).
      item = input_queue.get();
    } else if (!input_queue.try_get(item)) {
      // core.py:1285-1286 queue.Empty -> break.
      break;
    }
    handle_client_request(item);
    if (!block) {
      // core.py:1288-1289.
      break;
    }
  }

  // core.py:1295-1298: handle any more client requests.
  EngineCoreInputItem item;
  while (input_queue.try_get(item)) {
    handle_client_request(item);
  }
}

bool EngineCoreProc::process_engine_step() {
  // core.py:1300-1318. "Called only when there are unfinished local requests."
  // core.py:1303: step the engine core.
  auto [outputs, model_executed] = (this->*step_fn_)();
  // core.py:1305-1306: put EngineCoreOutputs into the output queue.
  for (auto& [client_index, engine_core_outputs] : outputs) {
    EngineCoreOutputItem out;
    out.client_index = client_index;
    out.outputs = std::move(engine_core_outputs);
    output_queue.put_nowait(std::move(out));
  }
  // core.py:1308 post_step(model_executed) — deferred (spec-decode draft
  // tokens; the sync EngineCore has no post_step yet, core.py:510-517).

  // core.py:1313-1317: if no model execution happened but there is still
  // scheduler work, yield briefly (upstream: lets KV-connector background
  // threads take the GIL; here it keeps a 0-token step from hot-spinning).
  if (!model_executed && has_work()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return model_executed;
}

bool EngineCoreProc::handle_shutdown() {
  // core.py:1324-1370: check if shutdown was requested and handle it.
  if (shutdown_state.load() == EngineShutdownState::kRunning) {
    return true;
  }

  if (shutdown_state.load() == EngineShutdownState::kRequested) {
    // core.py:1330-1332: mode = "abort" if shutdown_timeout == 0 else "drain".
    if (shutdown_timeout_s_ == 0) {
      // core.py:1341-1349 abort mode: finish every in-flight request as
      // FINISHED_ABORTED (upstream scheduler.finish_requests(None, ABORTED))
      // and send their abort outputs (_send_abort_outputs :1349).
      std::vector<std::string> request_ids;
      request_ids.reserve(scheduler_.requests.size());
      for (const auto& [request_id, request] : scheduler_.requests) {
        request_ids.push_back(request_id);
      }
      abort_requests(request_ids);
      send_finish_outputs(request_ids, FinishReason::kAbort);
    }
    // core.py:1351-1358 drain mode: in-flight requests keep stepping below
    // until no work remains (the drain-timeout enforcement is the launcher's
    // join, not the loop's).
    // core.py:1360.
    shutdown_state.store(EngineShutdownState::kShuttingDown);
  }

  // core.py:1362-1368: exit when no work remaining.
  if (!has_work()) {
    return false;
  }
  return true;
}

void EngineCoreProc::handle_client_request(EngineCoreInputItem& item) {
  // core.py:1372-1404: dispatch request from client.
  switch (item.type) {
    case EngineCoreRequestType::kWakeup:
      // core.py:1377-1378: the wake-up sentinel is a no-op; the caller's loop
      // re-checks is_running().
      return;
    case EngineCoreRequestType::kAdd: {
      // core.py:1379-1383 (+ _reject_add_in_shutdown :1407-1416): during
      // shutdown a new request is rejected with an abort output instead of
      // being scheduled. The request_wave value is deferred (DP).
      if (shutdown_state.load() != EngineShutdownState::kRunning) {
        send_finish_outputs({item.request->request_id}, FinishReason::kAbort);
        return;
      }
      add_request(std::move(item.request));
      return;
    }
    case EngineCoreRequestType::kAbort:
      // core.py:1384-1385.
      abort_requests(item.request_ids);
      return;
    case EngineCoreRequestType::kExecutorFailed:
      // core.py:1400-1401: raise RuntimeError("Executor failed."), caught by
      // the client's thread guard which posts ENGINE_CORE_DEAD.
      throw std::runtime_error("Executor failed.");
    case EngineCoreRequestType::kStartDpWave:
    case EngineCoreRequestType::kUtility:
    default:
      // core.py:1386-1404: UTILITY (deferred) / unrecognized types are logged
      // and dropped upstream; no logger is wired here yet, so drop silently.
      return;
  }
}

void EngineCoreProc::send_finish_outputs(
    const std::vector<std::string>& request_ids, FinishReason reason) {
  // core.py:1734-1742 groups by request.client_index; single in-proc client 0.
  // core.py:1735: no-op for an empty list.
  if (request_ids.empty()) {
    return;
  }
  // core.py:1714-1722: one empty-token EngineCoreOutput per id carrying the
  // finish reason. (finished_requests on EngineCoreOutputs is deferred —
  // types.h DEFERRED list.)
  EngineCoreOutputItem out;
  out.client_index = 0;
  for (const std::string& request_id : request_ids) {
    EngineCoreOutput output;
    output.request_id = request_id;
    output.finish_reason = reason;
    out.outputs.outputs.push_back(std::move(output));
  }
  output_queue.put_nowait(std::move(out));
}

void EngineCoreProc::send_engine_dead(const std::string& reason) {
  // core.py:1470-1480: put ENGINE_CORE_DEAD in the queue. (The output-thread
  // join/fatal-log half is socket machinery that does not exist in-proc.)
  EngineCoreOutputItem out;
  out.engine_dead = true;
  out.error_message = reason;
  output_queue.put_nowait(std::move(out));
}

}  // namespace vllm::v1
