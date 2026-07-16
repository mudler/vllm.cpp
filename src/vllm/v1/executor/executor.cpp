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

std::unique_ptr<AsyncModelRunnerOutput> Executor::sample_tokens_async(
    const std::optional<GrammarOutput>& grammar_output) {
  // Upstream: collective_rpc("sample_tokens", ..., non_block=True) hands back an
  // AsyncOutputFuture wrapping the runner's AsyncModelRunnerOutput
  // (uniproc_executor.py:97-100). T0 single-worker: a direct call.
  return runner_.sample_tokens_async(grammar_output);
}

}  // namespace vllm::v1
