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
//   - take_draft_token_ids: a default-nullopt SEAM is declared for SPEC-MTP (I2);
//     the drafting override lands with the verify/propose runner (I5).
//   - execute_dummy_batch, the KV-cache
//     init / profiling / LoRA / pooling RPC methods on WorkerBase.
#pragma once

#include <memory>
#include <optional>

#include "vllm/v1/core/sched/output.h"           // SchedulerOutput
#include "vllm/v1/engine/types.h"                 // ModelRunnerOutput
#include "vllm/v1/worker/gpu/async_output.h"      // AsyncModelRunnerOutput

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

  // sample_tokens_async: the overlap-scheduling variant of sample_tokens. Under
  // async scheduling the engine's step_with_batch_queue calls this instead of
  // sample_tokens; the returned AsyncModelRunnerOutput may defer the sampled-id
  // D2H (issued on a copy stream with an event) so get_output() — called at batch
  // CONSUME time, off the model's critical path — is the only host block, letting
  // the copy overlap the next step's forward (the ~3.25 ms/step GPU-idle capture,
  // async_utils.py:12-70). The DEFAULT implementation degenerates to the
  // synchronous sample_tokens wrapped in a ReadyModelRunnerOutput, so a runner
  // (or test stub) that has no async path stays correct — exactly the sync case.
  // The real batched runner overrides this to take the device-resident route when
  // async is engaged. runner_supports_async advertises whether a runner offers a
  // non-degenerate path (fed to SchedulerConfig::ResolveAsyncScheduling).
  virtual std::unique_ptr<AsyncModelRunnerOutput> sample_tokens_async(
      const std::optional<GrammarOutput>& grammar_output) {
    return std::make_unique<ReadyModelRunnerOutput>(sample_tokens(grammar_output));
  }

  // runner_supports_async (vllm/config/vllm.py:990-1038 compat gate): whether
  // this runner advertises the placeholder-aware async device path. Default
  // false — a plain runner keeps the synchronous scheduler. The batched runner
  // overrides it to reflect the VT_ASYNC_RUNNER opt-in.
  virtual bool runner_supports_async() const { return false; }

  // take_draft_token_ids (model_runner.py take_draft_token_ids / executor
  // abstract.py): hand the drafter's out-of-band proposal for the NEXT step back
  // to the caller, consuming it (the runner clears its stash). Default nullopt —
  // a runner with no speculator never drafts. The verify/propose runner (SPEC-MTP
  // I5) overrides this to return the drafts its speculator produced; the engine
  // core feeds them to Scheduler::update_draft_token_ids (core.py:511-517). This
  // is the frozen SPEC-MTP seam (I2) — inert until I5 fills it.
  virtual std::optional<DraftTokenIds> take_draft_token_ids() {
    return std::nullopt;
  }
};

}  // namespace vllm::v1
