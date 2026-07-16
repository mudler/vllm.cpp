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
// step_with_batch_queue + batch_queue (core.py:519-632) are IMPLEMENTED for
// ENG-ASYNC-SCHED (spec async-serving.md W3): the depth-2 overlap step selected
// when max_concurrent_batches > 1 (async scheduling). The synchronous step()
// path is unchanged.
//
// DEFERRED (marked 1:1 stubs; matches upstream structure so re-adding is
// mechanical): PP-general batch queue depth > 2 (ENG-BATCH-QUEUE),
// grammar_output / structured_output_manager (M3),
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
#include "vllm/v1/engine/types.h"          // EngineCoreOutputs, ModelRunnerOutput
#include "vllm/v1/executor/executor.h"     // Executor
#include "vllm/v1/request.h"               // Request

namespace vllm::v1 {

class StructuredOutputManager;  // vllm/v1/structured_output/manager.h

// A scheduled-and-executed batch waiting in the batch queue (core.py:197-198
// deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]]). The
// "future" is an AsyncModelRunnerOutput: under async scheduling
// (ENG-ASYNC-SCHED W3) it may defer the sampled-id D2H, so its get_output() is
// only resolved when this batch is CONSUMED (update_from_output) — off the model's
// critical path, letting the copy overlap the next step's forward. A synchronous
// runner hands back a ReadyModelRunnerOutput (already materialized), so the sync
// path is unchanged. A failed eager forward throws directly through the
// engine-fatal guard rather than surfacing as a None result. See
// EngineCore::step_with_batch_queue.
struct BatchQueueItem {
  std::unique_ptr<AsyncModelRunnerOutput> async_output;
  SchedulerOutput scheduler_output;
};

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
  std::pair<std::map<int, EngineCoreOutputs>, bool> step();

  // step_with_batch_queue (core.py:519-632): the depth-N overlap step selected
  // when max_concurrent_batches > 1 (async scheduling → depth 2). Schedules and
  // executes a NEW batch (if the queue is not full and there is work) BEFORE
  // consuming the OLDEST queued batch's output, so step N+1 is scheduled while
  // step N's output is processed. Returns ({}, model_executed) without blocking
  // while it is still filling the queue; once the queue is full (or no more
  // requests) it pops the oldest batch, runs update_from_output, and returns its
  // per-client outputs. Token-exactness: our executor resolves eagerly, so the
  // sampled tokens are materialized before the next batch is scheduled — the
  // AsyncScheduler placeholder accounting keeps the schedule consistent and
  // update_from_output reconciles one step later, exactly as upstream. The
  // grammar-deferral branch (core.py:559-570,609-630) is ported for structured
  // requests whose bitmask must wait on the prior step's tokens.
  std::pair<std::map<int, EngineCoreOutputs>, bool> step_with_batch_queue();

  // has_batch_queue_work (core.py:1247-1253 `bool(batch_queue)`): whether the
  // batch queue still holds an un-consumed batch. The busy loop must keep
  // stepping to drain it even after the scheduler has no more requests.
  bool has_batch_queue_work() const { return !batch_queue_.empty(); }

 protected:
  // Protected (not private) because EngineCoreProc subclasses EngineCore
  // exactly as upstream (core.py:896 `class EngineCoreProc(EngineCore)`) and
  // its has_work()/shutdown paths read the scheduler directly.
  Scheduler& scheduler_;
  Executor& executor_;
  // The engine's StructuredOutputManager (null when structured output is not
  // wired). Non-owning; must outlive the EngineCore. See the ctor note.
  StructuredOutputManager* structured_output_manager_ = nullptr;

  // Batch queue (core.py:196-202). batch_queue_size_ == max_concurrent_batches
  // (1 = synchronous step(); 2 = depth-2 overlap under async scheduling). The
  // deque holds at most batch_queue_size_ scheduled-and-executed batches;
  // step_with_batch_queue appends to the FRONT and pops the BACK (oldest). Set
  // by EngineCoreProc's ctor. Empty and unused on the synchronous step() path.
  int batch_queue_size_ = 1;
  std::deque<BatchQueueItem> batch_queue_;
};

}  // namespace vllm::v1
