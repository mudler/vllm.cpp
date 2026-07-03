// vllm.cpp original (minimal model registry). Mirrors the upstream registry
// pattern (vllm/model_executor/models/registry.py @ e24d1b24) which maps a HF
// `architectures` string to a model class. M0.9 has a single assembled forward
// (the Qwen3.6-35B-A3B MoE gate), so the registry maps the architecture string
// to that forward function pointer; more models slot in as their forwards land.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm {

// The forward signature every registered Qwen3.5-family model implements.
using ModelForwardFn = std::vector<float> (*)(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const Qwen3_5MoeWeights& weights, const HfConfig& config, vt::Queue& queue);

// Resolves a HF architecture string (e.g. "Qwen3_5MoeForConditionalGeneration")
// to its forward. Throws std::runtime_error naming the architecture if it is
// not registered.
ModelForwardFn ResolveModelForward(const std::string& architecture);

// Convenience overload: resolves from config.architectures[0]. Throws if the
// architectures list is empty or the entry is unregistered.
ModelForwardFn ResolveModelForward(const HfConfig& config);

}  // namespace vllm
