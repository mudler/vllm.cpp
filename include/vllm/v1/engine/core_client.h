// Ported from: vllm/v1/engine/core_client.py @ e24d1b24
// (MPClient contract :467-478 — "EngineCore runs in a background process busy
// loop, getting new EngineCoreRequests and returning EngineCoreOutputs; pushes
// EngineCoreRequests via input_socket, pulls EngineCoreOutputs via
// output_socket" — and SyncMPClient :779-893: get_output blocking pop :849-859,
// add_request -> ADD :886-889, abort_requests -> ABORT guarded on engine_dead
// :891-893) + EngineDeadError (vllm/v1/engine/exceptions.py:9-15) + the engine
// launch/guard shape of run_engine_core (core.py:1154-1243).
//
// Scope (async-serving spec W1, row ENG-CORE-BUSY-LOOP): the client side of
// the queue split. InprocClient owns an EngineCoreProc and the dedicated
// engine thread running its busy loop; frontends submit requests and pull
// outputs without ever touching the scheduler/executor directly.
//
// DEVIATIONS (recorded, spec async-serving.md D2 + Port map W1):
//   - In-process: the engine runs on a std::thread owned by this client, not a
//     forked process. The ZMQ socket<->queue relay threads (SyncMPClient's
//     process_outputs_socket :808-846 and the proc-side input thread) collapse
//     into DIRECT sharing of the proc's queues — get_output() pops the proc's
//     output_queue itself. The API shapes (add_request_async / get_output /
//     abort_requests_async) are kept so a future multi-process client is a
//     drop-in (spec Port map W1 note).
//   - The method names use the *_async shapes the spec pins for W2's AsyncLLM
//     (upstream EngineCoreClient exposes both sync and async spellings,
//     core_client.py:143/218, :175/246); in C++ both collapse to "enqueue and
//     return". get_output keeps the SyncMPClient blocking name/shape.
//   - add_request_async takes an already-built Request: the EngineCoreRequest
//     -> Request build (Request::FromEngineCoreRequest + block_hasher) stays
//     with the frontend exactly as in llm_engine.h (recorded there); upstream
//     does it on the proc's input-socket thread (core.py:855-878).
//   - shutdown() is the SIGTERM-handler analog (core.py:1204-1222): set
//     shutdown_state = REQUESTED, post the WAKEUP sentinel, join the thread.
//     The destructor calls it; shutdown mode (abort vs drain) is fixed by the
//     shutdown_timeout_s the proc was constructed with.
//
// DEFERRED (marked so re-adding is mechanical): AsyncMPClient's asyncio
// surface (:950-1046 — W2 consumes this client directly instead), utility
// calls / call_utility (:878-884), DP (dp_engines_running, wave signalling,
// engine selection), multi-engine indexes, and the launch/handshake machinery
// (core_client.py MPClient __init__, launch_core_engines).
#pragma once

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vllm/v1/engine/core_proc.h"  // EngineCoreProc + queue/message types
#include "vllm/v1/engine/types.h"      // EngineCoreOutputs
#include "vllm/v1/request.h"           // Request

namespace vllm::v1 {

// EngineDeadError (vllm/v1/engine/exceptions.py:9-15): raised when the
// EngineCore dies. Unrecoverable.
class EngineDeadError : public std::runtime_error {
 public:
  explicit EngineDeadError(const std::string& detail)
      : std::runtime_error(
            "EngineCore encountered an issue. See stack trace (above) for the "
            "root cause." +
            (detail.empty() ? std::string() : " [" + detail + "]")) {}
};

// The in-process EngineCore client over the busy-loop queue split.
// Collaborator lifetimes as EngineCore: scheduler/executor (and the optional
// structured-output manager) are caller-owned and must outlive the client.
class InprocClient {
 public:
  // Construct the EngineCoreProc and start the engine thread running its busy
  // loop (the in-proc analog of launching the core engine process +
  // run_engine_core's guard, core.py:1154-1243: on a fatal busy-loop
  // exception the guard posts ENGINE_CORE_DEAD via send_engine_dead so a
  // blocked get_output() learns of the death, core.py:1229-1235).
  InprocClient(Scheduler& scheduler, Executor& executor,
               StructuredOutputManager* structured_output_manager = nullptr,
               int max_concurrent_batches = 1, int shutdown_timeout_s = 0);

  // shutdown() then join (upstream MPClient.shutdown + atexit teardown).
  ~InprocClient();

  InprocClient(const InprocClient&) = delete;
  InprocClient& operator=(const InprocClient&) = delete;

  // get_output (core_client.py:849-859): blocking pop of the next
  // EngineCoreOutputs batch. Throws EngineDeadError when the ENGINE_CORE_DEAD
  // sentinel is popped (validate_alive :454-457 + the exception forward in
  // get_output). The wave_complete branch (:857-858) is deferred (DP).
  EngineCoreOutputs get_output();

  // add_request (core_client.py:886-889): enqueue an ADD. The DP
  // engines_running flip (:887-888) is deferred.
  void add_request_async(std::unique_ptr<Request> request);

  // abort_requests (core_client.py:891-893): enqueue an ABORT; no-op for an
  // empty list or a dead engine.
  void abort_requests_async(const std::vector<std::string>& request_ids);

  // The SIGTERM-handler analog (core.py:1204-1222): request shutdown, wake the
  // blocking input_queue.get() with the WAKEUP sentinel, and join the engine
  // thread. Idempotent. Abort-vs-drain follows the ctor's shutdown_timeout_s.
  void shutdown();

  // resources.engine_dead (core_client.py:441,454-457): set once get_output
  // pops the sentinel; guards abort_requests_async.
  bool engine_dead() const { return engine_dead_.load(); }

  // The wrapped proc — the direct queue-sharing seam (in-proc there is no
  // socket boundary to hide; tests and W2's output handler drive it).
  EngineCoreProc& proc() { return proc_; }

 private:
  EngineCoreProc proc_;
  std::thread engine_thread_;
  std::atomic<bool> engine_dead_{false};
};

}  // namespace vllm::v1
