// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (EngineCoreProc — the busy-loop wrapper over EngineCore. Queues :915-916,
// EngineCoreRequestType vllm/v1/engine/__init__.py:251-264,
// EngineShutdownState :890-893, step_fn/batch-queue selection :196-223,
// has_work :1247-1253, is_running :1255-1257, run_busy_loop :1259-1266,
// _process_input_queue :1269-1298, _process_engine_step :1300-1318,
// _handle_shutdown :1324-1370, _handle_client_request :1372-1404,
// _reject_add_in_shutdown :1407-1416, _send_finish_outputs_to_client
// :1714-1722 / _send_abort_outputs :1734-1742, _send_engine_dead :1470-1480,
// ENGINE_CORE_DEAD sentinel :899.)
//
// Scope (async-serving spec W1, row ENG-CORE-BUSY-LOOP): the engine busy loop
// with the input/output queue split — the boundary that lets a frontend thread
// submit work and pull outputs while the engine steps continuously on its own
// thread. This is the in-process analog of upstream's EngineCoreProc, which
// runs the same loop in a background *process* behind ZMQ sockets.
//
// DEVIATIONS (recorded, spec async-serving.md D2 + Port map W1):
//   - In-process std::thread instead of a forked process: the thread itself is
//     owned by InprocClient (core_client.h); this class only provides the loop.
//   - input_queue/output_queue are mutex+condvar deques (BlockingQueue) with
//     Python queue.Queue's blocking get / put_nowait semantics; no ZMQ frames —
//     payloads move as C++ values. The socket<->queue relay threads
//     (core.py process_input_sockets / process_output_sockets, :980-1001)
//     therefore do not exist; clients share the queues directly.
//   - The ADD payload is an already-built Request (our frontend builds it via
//     Request::FromEngineCoreRequest, see llm_engine.h): upstream's
//     preprocess_add_request (:855-878) runs on the input-socket thread; ours
//     runs grammar_init inside EngineCore::add_request on the engine thread
//     (same recorded deviation as core.h).
//   - Shutdown is triggered by InprocClient::shutdown() setting shutdown_state
//     and posting the WAKEUP sentinel — the in-proc analog of the SIGTERM
//     handler + wakeup_engine (core.py:1204-1222).
//
// step_with_batch_queue + batch_queue are IMPLEMENTED (ENG-ASYNC-SCHED, spec W3):
// max_concurrent_batches > 1 selects step_with_batch_queue (depth-2 overlap under
// async scheduling); the loop's has_work() includes the batch-queue term.
//
// DEFERRED (marked 1:1 so re-adding is mechanical): PP-general batch queue
// (ENG-BATCH-QUEUE, depth > 2), UTILITY calls + _invoke_utility_method (:1434-1452),
// START_DP_WAVE / DP waves / engines_running, aborts_queue +
// _process_aborts_queue (no in-flight-abort window in the sync step yet),
// _notify_idle_state_callbacks (:1320-1322, pause/sleep surface), post_step
// (spec-decode draft tokens, core.py:510-517), mm_receiver_cache, tensor IPC,
// ZMQ handshakes, log_stats, and the per-request client_index fan-out (single
// client 0 — same T0 collapse as EngineCore::step).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/engine/core.h"   // EngineCore
#include "vllm/v1/engine/types.h"  // EngineCoreOutputs
#include "vllm/v1/request.h"       // Request, FinishReason

namespace vllm::v1 {

// EngineCoreRequestType (vllm/v1/engine/__init__.py:251-264): the request
// types a client can put on the input queue. Upstream encodes them as hex byte
// strings for the socket wire; values kept identical.
enum class EngineCoreRequestType : std::uint8_t {
  kAdd = 0x00,
  kAbort = 0x01,
  kStartDpWave = 0x02,  // DEFERRED (DP)
  kUtility = 0x03,      // DEFERRED (utility call surface)
  kExecutorFailed = 0x04,
  // Sentinel to wake up input_queue.get() during shutdown.
  kWakeup = 0x05,
};

// EngineShutdownState (core.py:890-893).
enum class EngineShutdownState : int {
  kRunning = 0,
  kRequested = 1,
  kShuttingDown = 2,
};

// BlockingQueue: in-proc analog of Python's unbounded queue.Queue — the type
// of EngineCoreProc.input_queue / output_queue (core.py:915-916). Blocking
// get(), non-blocking put_nowait()/try_get(), thread-safe.
template <typename T>
class BlockingQueue {
 public:
  void put_nowait(T item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      items_.push_back(std::move(item));
    }
    cv_.notify_one();
  }

  // Blocking pop (queue.Queue.get(block=True)).
  T get() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !items_.empty(); });
    T item = std::move(items_.front());
    items_.pop_front();
    return item;
  }

  // Non-blocking pop (queue.Queue.get_nowait); false <=> queue.Empty.
  bool try_get(T& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (items_.empty()) {
      return false;
    }
    item = std::move(items_.front());
    items_.pop_front();
    return true;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.empty();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<T> items_;
};

// The input-queue message: upstream tuple[EngineCoreRequestType, Any]
// (core.py:915). One payload field per request type; unused fields stay empty.
struct EngineCoreInputItem {
  EngineCoreRequestType type = EngineCoreRequestType::kWakeup;
  std::unique_ptr<Request> request;      // kAdd payload
  std::vector<std::string> request_ids;  // kAbort payload
};

// The output-queue message: upstream tuple[int, EngineCoreOutputs] | bytes
// (core.py:916) — the bytes alternative is the ENGINE_CORE_DEAD sentinel
// (core.py:899, put by _send_engine_dead :1470-1480).
struct EngineCoreOutputItem {
  int client_index = 0;
  EngineCoreOutputs outputs;
  bool engine_dead = false;   // the ENGINE_CORE_DEAD sentinel
  std::string error_message;  // detail carried with the sentinel (ours)
};

// The busy-loop engine wrapper (core.py:896 `class EngineCoreProc(EngineCore)`).
// Collaborator lifetimes as EngineCore: scheduler/executor outlive this object.
// Thread model: run_busy_loop() executes on ONE dedicated engine thread; the
// queues and shutdown_state are the only members other threads may touch.
class EngineCoreProc : public EngineCore {
 public:
  // core.py:905-930 (__init__ queue setup; the T0 in-proc subset) plus the
  // batch-queue/step_fn selection from EngineCore.__init__ (core.py:196-223):
  //   batch_queue_size = max_concurrent_batches (vllm/config/vllm.py:490-501:
  //   pp_size, or pp_size + 1 under async scheduling on MRV2);
  //   step_fn = step if batch_queue is None else step_with_batch_queue.
  // max_concurrent_batches > 1 selects step_with_batch_queue (ENG-ASYNC-SCHED,
  // spec W3); < 1 throws std::invalid_argument.
  // shutdown_timeout_s mirrors VllmConfig.shutdown_timeout
  // (vllm/config/vllm.py:377, default 0): 0 => "abort" shutdown mode,
  // > 0 => "drain" (core.py:1330-1358).
  EngineCoreProc(Scheduler& scheduler, Executor& executor,
                 StructuredOutputManager* structured_output_manager = nullptr,
                 int max_concurrent_batches = 1, int shutdown_timeout_s = 0);

  // The IO queue split (core.py:915-916). Public exactly like the upstream
  // attributes: the in-proc client shares them directly (no socket relay).
  BlockingQueue<EngineCoreInputItem> input_queue;
  BlockingQueue<EngineCoreOutputItem> output_queue;

  // core.py:911 self.shutdown_state = EngineShutdownState.RUNNING; mutated by
  // the client (signal-handler analog, core.py:1216-1222) => atomic.
  std::atomic<EngineShutdownState> shutdown_state{EngineShutdownState::kRunning};

  // has_work (core.py:1247-1253): "Returns true if the engine should be
  // stepped." The engines_running (DP) and batch_queue terms are deferred, so
  // this is scheduler-has-requests (interface.py default, inlined exactly as
  // EngineCore::step does — see core.h DEVIATIONS).
  bool has_work() const;

  // is_running (core.py:1255-1257): "Returns true if shutdown has not been
  // requested."
  bool is_running() const;

  // run_busy_loop (core.py:1259-1266): the core busy loop —
  //   while _handle_shutdown(): _process_input_queue(); _process_engine_step().
  // Runs on the dedicated engine thread; returns (instead of raising
  // SystemExit) when shutdown completes.
  void run_busy_loop();

  // _send_engine_dead (core.py:1470-1480): put the ENGINE_CORE_DEAD sentinel
  // on the output queue so a blocked get_output() learns the engine died.
  // Called by the client's thread guard on a fatal busy-loop exception
  // (run_engine_core's except path, core.py:1229-1235).
  void send_engine_dead(const std::string& reason);

 private:
  // _process_input_queue (core.py:1269-1298): block for work while idle, then
  // drain any further queued client requests.
  void process_input_queue();

  // _process_engine_step (core.py:1300-1318): step_fn() -> put outputs on the
  // output queue; returns model_executed.
  bool process_engine_step();

  // _handle_shutdown (core.py:1324-1370): returns whether the busy loop should
  // keep running; on the first pass after a shutdown request, aborts in-flight
  // requests (timeout 0, "abort" mode) or lets them drain ("drain" mode).
  bool handle_shutdown();

  // _handle_client_request (core.py:1372-1404): dispatch one input item.
  void handle_client_request(EngineCoreInputItem& item);

  // _send_finish_outputs_to_client (core.py:1714-1722) collapsed to the single
  // in-proc client: one EngineCoreOutput(req_id, [], finish_reason) per id.
  // (The upstream finished_requests field on EngineCoreOutputs is deferred —
  // see types.h.) No-op for an empty list (_send_abort_outputs :1735).
  void send_finish_outputs(const std::vector<std::string>& request_ids,
                           FinishReason reason);

  // step_fn (core.py:221-223): step vs step_with_batch_queue. Selected in the
  // ctor from max_concurrent_batches (> 1 => step_with_batch_queue, W3). The
  // batch_queue_size_ + batch_queue_ members live on EngineCore (core.h).
  using StepFn = std::pair<std::map<int, EngineCoreOutputs>, bool> (
      EngineCore::*)();
  StepFn step_fn_ = &EngineCore::step;

  // core.py:958 self.process_input_queue_block = True (the non-blocking
  // variant is a DP-wave path, DPEngineCoreProc core.py:1940).
  bool process_input_queue_block_ = true;

  // VllmConfig.shutdown_timeout (vllm/config/vllm.py:377): selects the
  // abort-vs-drain shutdown mode in handle_shutdown (core.py:1330-1332).
  int shutdown_timeout_s_ = 0;
};

}  // namespace vllm::v1
