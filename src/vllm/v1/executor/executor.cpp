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

ModelRunnerOutput Executor::sample_tokens() {
  // Upstream: output = collective_rpc("sample_tokens", args=(grammar_output,));
  // return output[0]. grammar_output is null at T0 (structured output deferred).
  return runner_.sample_tokens();
}

}  // namespace vllm::v1
