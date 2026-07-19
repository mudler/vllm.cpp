// Shared Qwen3.5/3.6 registry-glue helpers used by BOTH per-variant registry
// TUs (qwen3_5_dense.cpp, qwen3_5_moe.cpp). Not installed.
//
// Holds the registry-facing bits the dense and MoE variants share verbatim: the
// per-family _ModelInfo capability record, the config hook, the KV-cache spec
// builder (identical for both variants), the borrowed-weights tag, and the
// host-logits carrier helper. The heavy per-variant forward machinery
// (Qwen3_5Model::/Qwen3_5DenseModel::) stays in qwen3_5.cpp.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"

namespace vllm {

// Tag selecting the borrowing (non-owning) LoadedModel constructor overloads in
// each variant TU (the synthetic in-memory Borrow* adapters).
struct BorrowedWeightsTag {};

// registry.py _ModelInfo for both Qwen3.6 variants (dense + MoE): text
// generation, hybrid attention, multimodal-capable wrappers.
inline constexpr ModelInfo kQwen3_5Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

// Per-family config hook. LoadHfConfig/HfConfigFromGguf already materialize the
// consumed Qwen fields; this explicit hook is where a family adds normalization
// or validation without changing the registry/runner contract.
void ParseQwen3_5Config(const HfConfig& config);

// KV-cache spec builder — identical for the dense and MoE variants (both are the
// same hybrid full-attention + GDN/Mamba layout).
v1::KVCacheConfig MakeQwen3_5KVCache(const HfConfig& config, int block_size,
                                     int num_blocks);

// Wraps a host logits vector into a ForwardLogits (rows inferred from vocab).
ForwardLogits HostLogits(std::vector<float>&& host, int64_t vocab);

}  // namespace vllm
