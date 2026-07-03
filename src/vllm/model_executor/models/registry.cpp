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
    return &Qwen3_5Model::Forward;
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
