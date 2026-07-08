// Ported from: vllm/v1/worker/gpu/model_runner.py @ e24d1b24
// (the execute_model / sample_tokens seam) + vllm/v1/worker/worker_base.py
// (the WorkerBase RPC target the executor drives).
//
// Scope (M1.8 Task 1): a minimal abstract model-runner interface exposing the
// MRV2 forward/sample SPLIT that the Executor pass-through drives. The MRV2
// model runner (vllm/v1/worker/gpu/model_runner.py) splits a step into:
//   * execute_model(scheduler_output) — run the forward, STASH the hidden
//     states, and return "no output yet" (None). It does NOT sample.
//   * sample_tokens() — gather logits from the stashed hidden states, sample,
//     and produce the ModelRunnerOutput.
// EngineCore.step() calls both in order: execute_model first, then (because the
// forward returned None) sample_tokens (mirrors core.py:497-499). The real
// batched runner lands in M1.8 Task 4 as an implementation of this interface;
// Task 1's tests use a runner stub (test double) so the loop is testable first.
//
// Upstream, the Executor reaches the runner via collective_rpc("execute_model")
// / collective_rpc("sample_tokens") against a WorkerWrapperBase around the
// runner; T0 collapses that whole path (collective_rpc / WorkerWrapperBase /
// Future / non_block / multiproc / Ray / DP) into a direct virtual call on this
// ABC (see src/vllm/v1/executor/executor.h). DEVIATION recorded in the
// porting inventory §9 (in-process direct call replaces the worker RPC seam).
//
// grammar_output arg to sample_tokens: THREADED as of M3.4 Task 2 (the scheduler
// produces it via get_grammar_bitmask; nullopt when no structured request is
// scheduled). Task 3 consumes it (apply_grammar_bitmask before sampling).
//
// DEFERRED (marked; slots in without reshaping the interface):
//   - non_block / Future return (T0 = synchronous),
//   - take_draft_token_ids (spec-decode), execute_dummy_batch, the KV-cache
//     init / profiling / LoRA / pooling RPC methods on WorkerBase.
#pragma once

#include <optional>
#include <utility>

#include "vllm/v1/core/sched/output.h"  // SchedulerOutput
#include "vllm/v1/engine/types.h"       // ModelRunnerOutput, PendingModelOutput

namespace vllm::v1 {

// Abstract model runner (the MRV2 execute_model / sample_tokens split). The
// Executor holds a reference to one of these; M1.8 Task 4 provides the real
// batched paged implementation, Task 1's tests provide a stub.
class ModelRunnerBase {
 public:
  virtual ~ModelRunnerBase() = default;

  // execute_model: run the model forward for this scheduled step and stash the
  // resulting hidden states. Returns std::nullopt in the MRV2 split ("forward
  // done, no output yet") — the caller then invokes sample_tokens(). The
  // optional return mirrors upstream execute_model's `ModelRunnerOutput | None`
  // (a non-MRV2 / fused runner could return the output directly here).
  virtual std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) = 0;

  // sample_tokens: gather logits from the stashed hidden states, apply the
  // structured-output grammar bitmask (when present), sample the next token(s),
  // and produce the ModelRunnerOutput the scheduler consumes. grammar_output is
  // the per-step structured-output payload (nullopt when no structured request
  // is scheduled); mirrors upstream sample_tokens(grammar_output)
  // (model_runner.py:1358). M3.4 Task 2 THREADS it here; Task 3 consumes it
  // (apply_grammar_bitmask before _sample). The no-grammar path is unchanged.
  virtual ModelRunnerOutput sample_tokens(
      const std::optional<GrammarOutput>& grammar_output) = 0;

  // ─── Async-decode pipeline seam (VT_ASYNC_DECODE) ──────────────────────────
  // execute_model_async: run the forward AND sample for this step, but return a
  // PendingModelOutput WITHOUT blocking on the sampled-token readback when the
  // step is async-eligible (eager pure-decode all-greedy) and `allow_defer` is
  // set — the tokens stay on device, the side-stream copy event is recorded (not
  // synced), and the deferred host write-back is applied later in finalize_output.
  // Mirrors vLLM's execute_model(non_block=True) returning an
  // AsyncGPUModelRunnerOutput (gpu_model_runner.py:242-296). `allow_defer=false`
  // forces a fully synchronous, host-ready result (used when the engine cannot
  // overlap this step, e.g. a prefill / batch reshape).
  //
  // DEFAULT (this base / test doubles): synchronous — run execute_model then
  // sample_tokens now and hand back a `ready` pending. Overridden by the real
  // GPUModelRunner to implement the true device-token deferral.
  virtual PendingModelOutput execute_model_async(
      const SchedulerOutput& scheduler_output,
      const std::optional<GrammarOutput>& grammar_output,
      bool allow_defer) {
    (void)allow_defer;
    PendingModelOutput pending;
    std::optional<ModelRunnerOutput> forward = execute_model(scheduler_output);
    pending.output = forward.has_value() ? std::move(*forward)
                                         : sample_tokens(grammar_output);
    pending.ready = true;
    return pending;
  }

  // finalize_output: block on the pending step's readback event (if any), apply
  // the deferred write-back, and produce the host ModelRunnerOutput the scheduler
  // consumes. Mirrors AsyncGPUModelRunnerOutput.get_output()
  // (gpu_model_runner.py:290-332). DEFAULT: the pending is already host-ready
  // (produced synchronously above) — hand back its output.
  virtual ModelRunnerOutput finalize_output(PendingModelOutput& pending) {
    return std::move(pending.output);
  }
};

}  // namespace vllm::v1
