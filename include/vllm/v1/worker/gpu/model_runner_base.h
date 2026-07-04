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
// DEFERRED (marked; slots in without reshaping the interface):
//   - grammar_output arg to sample_tokens (T0 = null / no structured output),
//   - non_block / Future return (T0 = synchronous),
//   - take_draft_token_ids (spec-decode), execute_dummy_batch, the KV-cache
//     init / profiling / LoRA / pooling RPC methods on WorkerBase.
#pragma once

#include <optional>

#include "vllm/v1/core/sched/output.h"  // SchedulerOutput
#include "vllm/v1/engine/types.h"       // ModelRunnerOutput

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

  // sample_tokens: gather logits from the stashed hidden states, sample the
  // next token(s), and produce the ModelRunnerOutput the scheduler consumes.
  // (grammar_output is deferred at T0 — see the file header.)
  virtual ModelRunnerOutput sample_tokens() = 0;
};

}  // namespace vllm::v1
