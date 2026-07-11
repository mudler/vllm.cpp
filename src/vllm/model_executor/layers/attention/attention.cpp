// Ported from:
//   vllm/model_executor/layers/attention/attention.py:204-236,387-400
//   vllm/v1/attention/backends/flash_attn.py:255-300
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/attention/attention.h"

#include <limits>
#include <stdexcept>

namespace vllm {

std::optional<vt::AttentionWindow> ResolveAttentionWindow(
    std::optional<int64_t> per_layer_sliding_window,
    std::optional<int64_t> model_sliding_window,
    v1::AttentionType attention_type, bool disable_model_sliding_window) {
  const std::optional<int64_t> effective_model_window =
      disable_model_sliding_window ? std::nullopt : model_sliding_window;
  const std::optional<int64_t> window = per_layer_sliding_window.has_value()
                                            ? per_layer_sliding_window
                                            : effective_model_window;
  if (!window.has_value()) return std::nullopt;
  if (*window <= 0 || *window > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("attention sliding_window must be in [1, INT32_MAX]");
  }

  const int32_t radius = static_cast<int32_t>(*window - 1);
  if (attention_type == v1::AttentionType::kEncoderOnly) {
    return vt::AttentionWindow{radius, radius};
  }
  return vt::AttentionWindow{radius, 0};
}

vt::PagedAttentionArgs MakePagedAttentionArgs(float scale, bool causal,
                                               const v1::AttentionLayer& layer) {
  vt::PagedAttentionArgs args;
  args.scale = scale;
  args.causal = causal;
  args.window_size = layer.window_size;
  return args;
}

}  // namespace vllm
