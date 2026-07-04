// Ported from: vllm/v1/executor/abstract.py (Executor.execute_model /
// sample_tokens — the two seam methods) + vllm/v1/executor/uniproc_executor.py
// (UniProcExecutor — the single-process impl) @ e24d1b24.
//
// Scope (M1.8 Task 1): a thin pass-through Executor over a single model runner.
// Upstream's Executor.execute_model / sample_tokens dispatch through
// collective_rpc(...) to a WorkerWrapperBase (UniProcExecutor: one
// driver_worker) and return output[0]; the whole collective_rpc /
// WorkerWrapperBase / Future / non_block / multiproc / Ray / DP machinery
// collapses at T0 (single-process, single-device) to a DIRECT virtual call on
// the ModelRunnerBase (M1.8 Task 4 provides the real runner). DEVIATION
// recorded in porting-inventory §9 (in-process direct call replaces the worker
// RPC seam; in-process queues already replace ZMQ elsewhere).
//
//   execute_model(scheduler_output) -> runner.execute_model  (returns nullopt
//                                      in the MRV2 split — "forward done")
//   sample_tokens()                 -> runner.sample_tokens   -> ModelRunnerOutput
//
// DEFERRED (marked; matches upstream structure so re-adding is mechanical):
//   - collective_rpc + get_class (backend selection: uni/mp/ray/external),
//   - non_block / Future returns (T0 = synchronous), the AsyncOutputFuture path,
//   - initialize_from_config / determine_available_memory / get_kv_cache_specs
//     / supported_tasks / add_lora / sleep / wake_up / check_health / shutdown
//     and the rest of the WorkerBase RPC surface,
//   - grammar_output arg to sample_tokens (T0 = null / no structured output),
//   - take_draft_token_ids (spec-decode), KVOutputAggregator (P/D transfer).
#pragma once

#include <optional>

#include "vllm/v1/core/sched/output.h"          // SchedulerOutput
#include "vllm/v1/engine/types.h"               // ModelRunnerOutput
#include "vllm/v1/worker/gpu/model_runner_base.h"  // ModelRunnerBase

namespace vllm::v1 {

// The single-process pass-through executor (upstream UniProcExecutor collapsed
// to a direct call). Holds a non-owning reference to the model runner; the
// EngineCore owns neither the executor nor the runner (they outlive it),
// mirroring upstream where the executor holds the driver_worker.
class Executor {
 public:
  explicit Executor(ModelRunnerBase& runner) : runner_(runner) {}

  // execute_model: pass the scheduled step to the runner's forward. Mirrors
  // Executor.execute_model -> collective_rpc("execute_model")[0]. Returns
  // std::nullopt in the MRV2 split (forward done, sample separately).
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output);

  // sample_tokens: drive the runner's sampling. Mirrors
  // Executor.sample_tokens -> collective_rpc("sample_tokens")[0].
  ModelRunnerOutput sample_tokens();

 private:
  ModelRunnerBase& runner_;
};

}  // namespace vllm::v1
