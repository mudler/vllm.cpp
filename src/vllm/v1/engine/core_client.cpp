// Ported from: vllm/v1/engine/core_client.py @ e24d1b24
// (SyncMPClient :779-893, collapsed to the in-proc queue split) + the
// run_engine_core launch/guard shape (vllm/v1/engine/core.py:1154-1243).
// See core_client.h for scope, deviations and deferrals.
#include "vllm/v1/engine/core_client.h"

#include <iostream>
#include <utility>

namespace vllm::v1 {

InprocClient::InprocClient(Scheduler& scheduler, Executor& executor,
                           StructuredOutputManager* structured_output_manager,
                           int max_concurrent_batches, int shutdown_timeout_s)
    : proc_(scheduler, executor, structured_output_manager,
            max_concurrent_batches, shutdown_timeout_s) {
  // The in-proc analog of launching the background engine process
  // (core_client.py launch_core_engines -> core.py:1154 run_engine_core):
  // one dedicated thread runs the busy loop under the fatal-error guard —
  // on exception, post ENGINE_CORE_DEAD (core.py:1229-1235 -> :1470-1480).
  engine_thread_ = std::thread([this] {
    try {
      proc_.run_busy_loop();
    } catch (const std::exception& e) {
      // Restore the upstream fatal log (vllm/v1/engine/core.py:1233:
      // logger.exception("EngineCore encountered a fatal error.") emitted
      // immediately before _send_engine_dead). This is the ONLY place that
      // holds the true root-cause string: the client-facing EngineDeadError
      // wrapper (core_client.h) deliberately hides it. std::cerr is
      // unit-buffered so the line survives the driver's SIGKILL escalation;
      // std::cout / buffered loggers must never be used here.
      std::cerr << "engine-fatal: EngineCore busy loop threw: " << e.what()
                << "\n";
      proc_.send_engine_dead(e.what());
    } catch (...) {
      std::cerr << "engine-fatal: EngineCore busy loop threw an unknown fatal "
                   "error\n";
      proc_.send_engine_dead("unknown fatal error in the engine busy loop");
    }
  });
}

InprocClient::~InprocClient() { shutdown(); }

EngineCoreOutputs InprocClient::get_output() {
  // core_client.py:849-859: blocking pop; an ENGINE_CORE_DEAD sentinel
  // (validate_alive :454-457) marks the engine dead and raises. The
  // wave_complete branch (:857-858) is deferred (DP).
  EngineCoreOutputItem item = proc_.output_queue.get();
  if (item.engine_dead) {
    engine_dead_.store(true);
    throw EngineDeadError(item.error_message);
  }
  return std::move(item.outputs);
}

void InprocClient::add_request_async(std::unique_ptr<Request> request) {
  // core_client.py:886-889 (ADD; the DP engines_running flip deferred). The
  // socket _send_input (:861-876) collapses to a direct queue put.
  EngineCoreInputItem item;
  item.type = EngineCoreRequestType::kAdd;
  item.request = std::move(request);
  proc_.input_queue.put_nowait(std::move(item));
}

void InprocClient::abort_requests_async(
    const std::vector<std::string>& request_ids) {
  // core_client.py:891-893: "if request_ids and not resources.engine_dead".
  if (request_ids.empty() || engine_dead_.load()) {
    return;
  }
  EngineCoreInputItem item;
  item.type = EngineCoreRequestType::kAbort;
  item.request_ids = request_ids;
  proc_.input_queue.put_nowait(std::move(item));
}

void InprocClient::shutdown() {
  // core.py:1204-1222: the signal handler sets shutdown_state = REQUESTED and
  // wakeup_engine puts the WAKEUP sentinel on the input queue so an idle
  // blocking get() returns; the busy loop then runs its shutdown drain/abort
  // (handle_shutdown) and exits, and we join the thread.
  if (!engine_thread_.joinable()) {
    return;
  }
  proc_.shutdown_state.store(EngineShutdownState::kRequested);
  EngineCoreInputItem wakeup;
  wakeup.type = EngineCoreRequestType::kWakeup;
  proc_.input_queue.put_nowait(std::move(wakeup));
  engine_thread_.join();
}

}  // namespace vllm::v1
