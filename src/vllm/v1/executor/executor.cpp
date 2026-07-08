// Ported from: vllm/v1/executor/abstract.py + vllm/v1/executor/uniproc_executor.py
// @ e24d1b24. See executor.h for scope / deferrals.
#include "vllm/v1/executor/executor.h"

namespace vllm::v1 {

std::optional<ModelRunnerOutput> Executor::execute_model(
    const SchedulerOutput& scheduler_output) {
  // Upstream: output = collective_rpc("execute_model", args=(scheduler_output,));
  // return output[0]. T0 single-worker: a direct call on the runner.
  return runner_.execute_model(scheduler_output);
}

ModelRunnerOutput Executor::sample_tokens(
    const std::optional<GrammarOutput>& grammar_output) {
  // Upstream: output = collective_rpc("sample_tokens", args=(grammar_output,));
  // return output[0]. Forward the scheduler's structured-output payload to the
  // runner (nullopt when no structured request is scheduled).
  return runner_.sample_tokens(grammar_output);
}

PendingModelOutput Executor::execute_and_sample_async(
    const SchedulerOutput& scheduler_output,
    const std::optional<GrammarOutput>& grammar_output, bool allow_defer) {
  // Upstream: execute_model(non_block=True) then sample_tokens(non_block=True),
  // returning an AsyncGPUModelRunnerOutput future. T0 single-worker: a direct
  // fused call on the runner returning the pending (device tokens + recorded
  // readback event) WITHOUT syncing when allow_defer and the step is eligible.
  return runner_.execute_model_async(scheduler_output, grammar_output,
                                     allow_defer);
}

ModelRunnerOutput Executor::finalize_output(PendingModelOutput& pending) {
  // Upstream: future.result() -> AsyncGPUModelRunnerOutput.get_output() (blocks
  // on the copy event). T0: a direct call on the runner.
  return runner_.finalize_output(pending);
}

}  // namespace vllm::v1
