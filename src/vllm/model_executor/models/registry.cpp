// vllm.cpp original; see registry.h.
#include "vllm/model_executor/models/registry.h"

#include <stdexcept>

#include "vllm/model_executor/models/qwen3_5.h"

namespace vllm {

ModelForwardFn ResolveModelForward(const std::string& architecture) {
  // The 35B gate checkpoint's architecture. The dense 27B
  // (Qwen3_5ForConditionalGeneration) is a separate arch/forward (deferred, see
  // the M0.9 plan); it is intentionally not registered here.
  if (architecture == "Qwen3_5MoeForConditionalGeneration") {
    // ModelForwardFn is the M0.9 dense single-sequence signature (the dgx
    // M0-exit greedy gate). The M1.8 batched PAGED Qwen3_5Model::Forward is a
    // different signature (attn metadata + KV caches) and is driven directly by
    // the runner (M1.8 Task 4), not through this string registry.
    return &Qwen3_5Model::ForwardDense;
  }
  throw std::runtime_error("vllm: no model registered for architecture '" +
                           architecture + "'");
}

ModelForwardFn ResolveModelForward(const HfConfig& config) {
  if (config.architectures.empty()) {
    throw std::runtime_error(
        "vllm: config has no architectures entry to resolve a model forward");
  }
  return ResolveModelForward(config.architectures.front());
}

}  // namespace vllm
