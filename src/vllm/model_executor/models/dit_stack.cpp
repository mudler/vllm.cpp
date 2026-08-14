// S2Mel DiT transformer stack. See dit_stack.h for the upstream anchors.
#include "vllm/model_executor/models/dit_stack.h"

#include <cstddef>
#include <vector>

#include "vllm/model_executor/models/dit_skip.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace dit_stack {

std::vector<float> Forward(const Config& cfg, const Weights& w,
                           const std::vector<float>& x, const std::vector<float>& cond,
                           const std::vector<float>& freqs) {
  const int64_t layers = static_cast<int64_t>(w.layers.size());
  VT_CHECK(layers > 0, "dit_stack: no layers");
  VT_CHECK(cfg.dim > 0 && cfg.frames > 0, "dit_stack: dim and frames must be positive");
  VT_CHECK(x.size() == static_cast<size_t>(cfg.frames * cfg.dim),
           "dit_stack: x must be [frames, dim]");

  const dit_skip::Schedule plan = dit_skip::Plan(layers);

  std::vector<float> cur = x;
  // The stack holds the OUTPUTS of emitting layers, in order, and receivers pop
  // the most recent. `plan.source` already says which layer each receiver takes,
  // so the stack here only has to carry the values.
  std::vector<std::vector<float>> stack;

  for (int64_t i = 0; i < layers; ++i) {
    const LayerWeights& layer = w.layers[static_cast<size_t>(i)];

    // Receive BEFORE the layer runs, merging with skip_in_linear.
    if (plan.source[static_cast<size_t>(i)] >= 0) {
      VT_CHECK(!stack.empty(), "dit_stack: a receiving layer found no skip");
      VT_CHECK(!layer.skip_in_w.empty(),
               "dit_stack: a receiving layer has no skip_in_linear");
      const std::vector<float> skip = stack.back();
      stack.pop_back();
      cur = dit_skip::ApplySkip(cur, skip, cfg.frames, cfg.dim, layer.skip_in_w,
                                layer.skip_in_b);
    }

    cur = dit::Block(cur, cond, cfg.frames, cfg.dim, cfg.heads, cfg.head_dim,
                     cfg.intermediate, freqs, layer.block, cfg.eps);

    // Emit AFTER, pushing this layer's own output.
    bool emits = false;
    for (const int64_t e : plan.emit) {
      if (e == i) {
        emits = true;
      }
    }
    if (emits) {
      stack.push_back(cur);
    }
  }

  // transformer.norm: the same AdaptiveLayerNorm shape as the per-block norms.
  return dit::AdaptiveLayerNorm(cur, cfg.frames, cfg.dim, cond, w.norm_proj_w,
                                w.norm_proj_b, w.norm_w, cfg.eps);
}

}  // namespace dit_stack
}  // namespace models
}  // namespace vllm
