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

// The MODEL-LEVEL sliding-window kill switch, mirroring vLLM's
// `ModelConfig.disable_sliding_window` (`vllm/config/model.py:248` @ pin
// `5559679229`). One switch for every model, because upstream's is one field on
// `ModelConfig` rather than a per-architecture toggle, and its own docstring
// settles the asymmetry: "If the model does not support sliding window, this
// argument is ignored." A model with no window needs no opt-out, so covering
// every family costs nothing at the sites that have nothing to disable.
//
// WHY A PROCESS-LEVEL SWITCH AND NOT A CONFIG FIELD, which is what upstream
// mutates. `config/model.py:766-769` sets `hf_text_config.sliding_window = None`
// and every layer inherits it. That mechanism CANNOT work here, and the reason is
// a bypass rather than a missing field: `gemma2.cpp:103`, `gemma3.cpp:101`,
// `gemma4.cpp:139` and `laguna_weights.cpp:117` each read
//
//   cfg.sliding_window.value_or(RawInt(cfg.raw, "sliding_window", 0))
//
// and `cfg.raw` is the FULL UNTOUCHED checkpoint document
// (`hf_config.cpp:598`). Nulling the typed field leaves the raw one, so the
// window would survive the switch silently. Tracked as its own defect; this
// switch is consumed where the window is USED instead, through
// `ResolveAttentionWindow`'s `disable_model_sliding_window` parameter, which has
// carried this exact meaning since W1 and has never had a caller that passes
// anything but the default.
void SetDisableSlidingWindow(bool disabled);
bool DisableSlidingWindowActive();
void ResetDisableSlidingWindowForTesting();

// Build the shared vt operator arguments from a generic AttentionLayer. Future
// model ports set layer.window_size with ResolveAttentionWindow; backend code
// does not reinterpret W or clone model-specific window logic.
vt::PagedAttentionArgs MakePagedAttentionArgs(float scale, bool causal,
                                               const v1::AttentionLayer& layer);

}  // namespace vllm
