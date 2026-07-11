// Ported from:
//   vllm/model_executor/layers/attention/attention.py:204-236,387-400
//   vllm/v1/attention/backends/flash_attn.py:255-300
// @ e24d1b24fe96.
//
// Generic attention-layer window plumbing. This is deliberately independent of
// any model family: per-layer configuration wins over the model-level value,
// then the semantic window W is mapped exactly once to the backend-neutral
// FlashAttention pair carried by vt::PagedAttentionArgs.
#pragma once

#include <cstdint>
#include <optional>

#include "vllm/v1/attention/backend.h"
#include "vt/ops.h"

namespace vllm {

// Resolve upstream Attention.__init__ precedence, the model-level disable flag,
// and FlashAttentionImpl's decoder/encoder mapping. An explicit per-layer value
// wins even when the model-level value is disabled. A missing value means full
// attention. W must be positive; HfConfig normalizes checkpoint W==0 to nullopt.
std::optional<vt::AttentionWindow> ResolveAttentionWindow(
    std::optional<int64_t> per_layer_sliding_window,
    std::optional<int64_t> model_sliding_window,
    v1::AttentionType attention_type,
    bool disable_model_sliding_window = false);

// Build the shared vt operator arguments from a generic AttentionLayer. Future
// model ports set layer.window_size with ResolveAttentionWindow; backend code
// does not reinterpret W or clone model-specific window logic.
vt::PagedAttentionArgs MakePagedAttentionArgs(float scale, bool causal,
                                               const v1::AttentionLayer& layer);

}  // namespace vllm
