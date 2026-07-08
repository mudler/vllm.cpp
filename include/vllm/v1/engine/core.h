// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (EngineCore.__init__ T0 subset, add_request, abort_requests, step).
//
// Scope (M1.8 Task 1): the inner engine loop. EngineCore drives the
// schedule -> execute_model -> sample_tokens -> update_from_output cycle over a
// Scheduler (M1.4) and an Executor (the runner pass-through). This is the T0
// synchronous, single-process, single-client slice of upstream EngineCore.
//
// step() mirrors core.py:479-508:
//   if not scheduler.has_requests(): return ({}, false)
//   scheduler_output = scheduler.schedule()
//   model_output = executor.execute_model(scheduler_output)   # nullopt (MRV2)
//   if model_output is None:
//       model_output = executor.sample_tokens()
//   engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)
//   return (engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0)
// The grammar_output / future.result() / async / batch_queue machinery is
// dropped (T0 = synchronous, grammar_output = null).
//
// DEVIATIONS vs the pinned API (recorded, use OUR names):
//   - has_requests(): our M1.4 Scheduler does NOT port the interface.py
//     has_requests() / has_unfinished_requests() convenience wrappers, so
//     step() inlines the interface default:
//       scheduler.get_num_unfinished_requests() > 0 || scheduler.has_finished_requests()
//     (the connector has_pending_push_work() term is deferred — no connector at T0).
//   - abort_requests(list): our M1.4 Scheduler.finish_requests takes a SINGLE
//     request_id (not str | Iterable | None), so abort_requests iterates the
//     list, calling finish_requests(id, kFinishedAborted) per id.
//   - step() return: upstream returns dict[client_index, EngineCoreOutputs];
//     our Scheduler.update_from_output already collapses the per-client fan-out
//     to a single flat EngineCoreOutputs (T0 single client), so step() wraps it
//     in a std::map<int, EngineCoreOutputs> keyed by its engine_index to
//     preserve the upstream return SHAPE.
//   - add_request(unique_ptr<Request>): our Scheduler.add_request takes
//     ownership via unique_ptr; the upstream request_id-type / pooling-task /
//     kv_transfer validation + abort_immediately are deferred/inapplicable.
//
// DEFERRED (marked 1:1 stubs; matches upstream structure so re-adding is
// mechanical): step_with_batch_queue + batch_queue (pipeline parallelism),
// grammar_output / structured_output_manager (M3), async_scheduling,
// DP / EngineCoreProc / ZMQ, mm_receiver_cache (multimodal),
// post_step / take_draft_token_ids (spec-decode), the utility handlers,
// _process_aborts_queue / aborts_queue, log_stats / stats / iteration details,
// _initialize_kv_caches (the runner allocates KV in Task 4), and the whole
// __init__ config/plugin/GC-freeze setup (T0 wires refs directly).
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/core/sched/output.h"     // SchedulerOutput
#include "vllm/v1/core/sched/scheduler.h"  // Scheduler
#include "vllm/v1/engine/types.h"          // EngineCoreOutputs, PendingModelOutput
#include "vllm/v1/executor/executor.h"     // Executor
#include "vllm/v1/request.h"               // Request

namespace vllm::v1 {

class StructuredOutputManager;  // vllm/v1/structured_output/manager.h

// The inner engine loop (T0 subset). Holds non-owning references to the
// Scheduler (M1.4) and the Executor; both outlive the EngineCore.
class EngineCore {
 public:
  // structured_output_manager (upstream core.py:134 the EngineCore constructs it;
  // core.py:153 passes it to the Scheduler; core.py:876 add_request calls
  // grammar_init): the engine's StructuredOutputManager. Optional (null) to keep
  // the M1.8 tests building a bare EngineCore; when null, structured output is a
  // no-op. When provided, it must be the SAME manager the Scheduler was built
  // with (so get_grammar_bitmask/should_advance see the compiled grammars).
  EngineCore(Scheduler& scheduler, Executor& executor,
             StructuredOutputManager* structured_output_manager = nullptr)
      : scheduler_(scheduler),
        executor_(executor),
        structured_output_manager_(structured_output_manager) {}

  // add_request: hand a new request to the scheduler (which takes ownership).
  // Mirrors EngineCore.add_request -> scheduler.add_request (validation +
  // abort_immediately deferred, see the file header).
  void add_request(std::unique_ptr<Request> request);

  // abort_requests: finish the given requests as FINISHED_ABORTED. Mirrors
  // EngineCore.abort_requests -> scheduler.finish_requests(ids,
  // FINISHED_ABORTED); iterates because our finish_requests takes one id.
  void abort_requests(const std::vector<std::string>& request_ids);

  // step: one iteration of the engine loop. Returns the per-client outputs map
  // (single client at T0) and whether the model was executed
  // (total_num_scheduled_tokens > 0). See the file header for the full order.
  // When VT_ASYNC_DECODE is set this dispatches to the depth-2 batch-queue
  // variant (step_with_batch_queue); otherwise the synchronous path (unchanged).
  std::pair<std::map<int, EngineCoreOutputs>, bool> step();

  // has_pending: whether the async batch queue still holds in-flight steps whose
  // outputs have not been drained. The driver loop (LLMEngine) keeps stepping
  // while this is true so the pipeline is fully flushed (and any speculative
  // overflow step is discarded). Always false on the synchronous path.
  bool has_pending() const { return !batch_queue_.empty(); }

 private:
  // The synchronous step (core.py:479-508) — the exact current path.
  std::pair<std::map<int, EngineCoreOutputs>, bool> step_sync();
  // The depth-2 async step (core.py:519-632 step_with_batch_queue): schedule +
  // issue forward/sample non-blocking, push onto batch_queue_; block on and
  // update_from_output the OLDEST pending only when the queue is full (or no more
  // work), so step N's host output-processing overlaps step N+1's GPU forward.
  std::pair<std::map<int, EngineCoreOutputs>, bool> step_async();

  // An in-flight step held in the batch queue: the SchedulerOutput that produced
  // it (needed by update_from_output) + the runner's pending (device tokens +
  // recorded readback event), synced/finalized when popped.
  struct PendingStep {
    SchedulerOutput scheduler_output;
    PendingModelOutput pending;
  };

  // Whether a scheduled step is a pure-decode continuation the pipeline may
  // overlap: no new/resumed requests, no finished-request removals, no structured
  // output, every scheduled req exactly one token. Anything else (prefill, chunk,
  // reshape, flush) is finalized WITHOUT deferral after draining the queue, so a
  // host-token step never reads an InputBatch whose write-back is still pending.
  static bool is_overlappable_decode(const SchedulerOutput& scheduler_output);

  // Finalize + update_from_output one popped pending, merging its per-client
  // outputs into `out`.
  void finalize_and_update(PendingStep& step,
                           std::map<int, EngineCoreOutputs>& out);

  // Depth-2 in-flight queue (empty + unused on the synchronous path).
  static constexpr std::size_t kBatchQueueSize = 2;
  std::deque<PendingStep> batch_queue_;

  Scheduler& scheduler_;
  Executor& executor_;
  // The engine's StructuredOutputManager (null when structured output is not
  // wired). Non-owning; must outlive the EngineCore. See the ctor note.
  StructuredOutputManager* structured_output_manager_ = nullptr;
};

}  // namespace vllm::v1
