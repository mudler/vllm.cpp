// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (EngineCoreProc busy loop: run_busy_loop :1259-1266, _process_input_queue
// :1269-1298, _process_engine_step :1300-1318, _handle_shutdown :1324-1370,
// _handle_client_request :1372-1404, _reject_add_in_shutdown :1407-1416,
// _send_finish_outputs_to_client :1714-1722, _send_abort_outputs :1734-1742,
// _send_engine_dead :1470-1480). See core_proc.h for scope, deviations and
// deferrals.
#include "vllm/v1/engine/core_proc.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "vllm/v1/core/sched/scheduler.h"

namespace vllm::v1 {

EngineCoreProc::EngineCoreProc(Scheduler& scheduler, Executor& executor,
                               StructuredOutputManager* structured_output_manager,
                               int max_concurrent_batches, int shutdown_timeout_s,
                               bool check_for_draft_tokens)
    : EngineCore(scheduler, executor, structured_output_manager,
                 check_for_draft_tokens),
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

  // DIAGNOSTIC (VT_LOOP_TRACE): read the env once on the engine thread. Off =>
  // the loop below is byte-identical to the upstream port (every trace read is
  // guarded by loop_trace_.enabled).
  if (!loop_trace_.initialized) {
    loop_trace_.initialized = true;
    loop_trace_.enabled = std::getenv("VT_LOOP_TRACE") != nullptr;
    if (loop_trace_.enabled) {
      loop_trace_.window_start = MonotonicSeconds();
      loop_trace_.last_iter_start = loop_trace_.window_start;
    }
  }

  while (handle_shutdown()) {
    if (!loop_trace_.enabled) {
      // 1) Poll the input queue until there is work to do.
      process_input_queue();
      // 2) Step the engine core and return the outputs.
      process_engine_step();
      continue;
    }
    // Instrumented path: measure the full-iteration cadence (the admission wait
    // a fresh request pays for the once-per-iteration drain) and the step wall.
    const double iter_top = MonotonicSeconds();
    const double interval = iter_top - loop_trace_.last_iter_start;
    loop_trace_.last_iter_start = iter_top;
    loop_trace_.iters += 1;
    loop_trace_.interval_sum += interval;
    if (interval > loop_trace_.interval_max) loop_trace_.interval_max = interval;

    process_input_queue();

    const double step_t0 = MonotonicSeconds();
    process_engine_step();
    const double step_dt = MonotonicSeconds() - step_t0;
    loop_trace_.step_sum += step_dt;
    if (step_dt > loop_trace_.step_max) loop_trace_.step_max = step_dt;
    loop_trace_maybe_dump(MonotonicSeconds());
  }
}

void EngineCoreProc::loop_trace_record_admit(double enqueue_ts) {
  // VT_LOOP_TRACE: fold one admitted kAdd request's input-queue residence
  // (enqueue -> drain) into the window. Caller guards on loop_trace_.enabled.
  loop_trace_.admits += 1;
  loop_trace_drain_admits_ += 1;
  if (enqueue_ts > 0.0) {
    const double resid = MonotonicSeconds() - enqueue_ts;
    loop_trace_.resid_sum += resid;
    if (resid > loop_trace_.resid_max) loop_trace_.resid_max = resid;
  }
}

void EngineCoreProc::loop_trace_maybe_dump(double now) {
  // VT_LOOP_TRACE: emit + reset the 1 s window aggregate. interval == the
  // admission cadence (one drain per iteration); resid == the measured
  // arrival->QUEUED input-queue wait; depth_max/admits_max expose backlog.
  const double elapsed = now - loop_trace_.window_start;
  if (elapsed < 1.0 || loop_trace_.iters == 0) return;
  const double iters = static_cast<double>(loop_trace_.iters);
  const double amean =
      loop_trace_.admits > 0
          ? loop_trace_.resid_sum / static_cast<double>(loop_trace_.admits)
          : 0.0;
  std::fprintf(
      stderr,
      "LOOPTRACE win=%.3f iters=%ld interval_ms(mean/max)=%.3f/%.3f "
      "step_ms(mean/max)=%.3f/%.3f admits=%ld admits_max/drain=%ld "
      "resid_ms(mean/max)=%.3f/%.3f depth_max=%ld\n",
      elapsed, loop_trace_.iters, 1e3 * loop_trace_.interval_sum / iters,
      1e3 * loop_trace_.interval_max, 1e3 * loop_trace_.step_sum / iters,
      1e3 * loop_trace_.step_max, loop_trace_.admits, loop_trace_.admits_max,
      1e3 * amean, 1e3 * loop_trace_.resid_max, loop_trace_.depth_max);
  // Reset the window (keep last_iter_start / initialized).
  loop_trace_.window_start = now;
  loop_trace_.iters = 0;
  loop_trace_.interval_sum = loop_trace_.interval_max = 0.0;
  loop_trace_.step_sum = loop_trace_.step_max = 0.0;
  loop_trace_.admits = loop_trace_.admits_max = 0;
  loop_trace_.resid_sum = loop_trace_.resid_max = 0.0;
  loop_trace_.depth_max = 0;
}

void EngineCoreProc::process_input_queue() {
  // core.py:1269-1298. "Exits when an engine step needs to be performed."
  if (loop_trace_.enabled) {
    // VT_LOOP_TRACE: snapshot the backlog this drain must clear and reset the
    // per-drain admit counter (folded into admits_max at the tail).
    loop_trace_drain_admits_ = 0;
    const long depth = static_cast<long>(input_queue.size());
    if (depth > loop_trace_.depth_max) loop_trace_.depth_max = depth;
  }
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

  if (loop_trace_.enabled && loop_trace_drain_admits_ > loop_trace_.admits_max) {
    loop_trace_.admits_max = loop_trace_drain_admits_;
  }
}

bool EngineCoreProc::process_engine_step() {
  // core.py:1300-1318. "Called only when there are unfinished local requests."
  // VT_ENGINE_STEP_LOG (external PR #227): a per-step heartbeat. A stalled
  // engine and a busy one look identical from outside — both just produce no
  // client tokens — and this is what separates them: `model_executed=0` with a
  // non-zero unfinished count, step after step, is the engine spinning on work
  // it can never schedule. Diagnostic only; it changes no generation.
  static const bool kStepHeartbeat = [] {
    const char* e = std::getenv("VT_ENGINE_STEP_LOG");
    if (e != nullptr && e[0] == '1') return true;
    const char* v = std::getenv("VT_SERVER_VERBOSE");
    return v != nullptr && v[0] == '1';
  }();
  const double heartbeat_t0 = kStepHeartbeat ? MonotonicSeconds() : 0.0;
  if (kStepHeartbeat) {
    std::fprintf(stderr,
                 "INFO core-step begin unfinished=%d finished_pending=%d\n",
                 scheduler_.get_num_unfinished_requests(),
                 scheduler_.has_finished_requests() ? 1 : 0);
    std::fflush(stderr);
  }
  // core.py:1303: step the engine core.
  auto [outputs, model_executed] = (this->*step_fn_)();
  if (kStepHeartbeat) {
    int n_out = 0;
    for (const auto& [client_index, engine_core_outputs] : outputs) {
      (void)client_index;
      n_out += static_cast<int>(engine_core_outputs.outputs.size());
    }
    std::fprintf(stderr,
                 "INFO core-step end model_executed=%d n_out=%d elapsed_s=%.3f\n",
                 model_executed ? 1 : 0, n_out,
                 MonotonicSeconds() - heartbeat_t0);
    std::fflush(stderr);
  }
  // core.py:1305-1306: put EngineCoreOutputs into the output queue.
  for (auto& [client_index, engine_core_outputs] : outputs) {
    EngineCoreOutputItem out;
    out.client_index = client_index;
    out.outputs = std::move(engine_core_outputs);
    output_queue.put_nowait(std::move(out));
  }
  // core.py:1308 post_step(model_executed). NO LONGER DEFERRED. This is the
  // step the proc loop owes the drafter: it pulls the out-of-band drafts the
  // worker proposed this step and installs them on the scheduler so the NEXT
  // step verifies them. Without it a speculator proposes every step and every
  // proposal is dropped — the engine still produces correct output, just one
  // token per step while paying the draft cost, which is exactly how it was
  // found (SPEC-DSPARK W6: 24 proposals, 0 installs, 5.5x slower).
  //
  // Idempotent by construction: take_draft_token_ids moves pending_drafts_ out,
  // so the `step()` path calling post_step internally and this call cannot
  // double-install, and a step that proposed nothing pulls nullopt.
  post_step(model_executed);

  // core.py:1314: `if not model_executed and self.scheduler.has_requests():`
  // yield briefly (upstream: lets KV-connector background threads take the GIL;
  // here it keeps a 0-token step from hot-spinning). Mirror vLLM EXACTLY — the
  // guard is `scheduler.has_requests()` (unfinished || finished), NOT has_work():
  // it must EXCLUDE the batch-queue term. A pure batch-queue drain (no scheduler
  // requests left, only queued batches to pop) makes real progress every step
  // (each iteration pops one batch), so it must NOT sleep — otherwise the final
  // tokens of the last requests eat a spurious 1 ms/batch that vLLM never pays.
  if (!model_executed && (scheduler_.get_num_unfinished_requests() > 0 ||
                          scheduler_.has_finished_requests())) {
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
      // VT_LOOP_TRACE: measure residence BEFORE add_request stamps the QUEUED
      // event, so resid endpoint == the scheduler's QUEUED timestamp (the same
      // point the TTFTSPLIT intake terminates at).
      if (loop_trace_.enabled) {
        loop_trace_record_admit(item.enqueue_ts);
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
