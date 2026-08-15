// Adaptive layer norm — how S2Mel's DiT is conditioned (#634).
//
// Every DiT block, and the final layer, condition on the timestep/style vector
// through adaLN rather than through cross-attention.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace adaln {

// modulate(x, shift, scale) = x * (1 + scale) + shift, broadcast over frames
// (diffusion_transformer.py:11-12).
//
// THE `1 +` IS THE DETAIL. Without it the modulation is centred on 0 instead of
// 1, which still trains and still generates -- it is simply a different model,
// and no shape or finiteness check can see it.
std::vector<float> Modulate(const std::vector<float>& x, int64_t frames, int64_t hidden,
                            const std::vector<float>& shift, const std::vector<float>& scale);

// LayerNorm with elementwise_affine = FALSE: normalize only, NO gamma or beta.
// The checkpoint contains no such tensors, so a port that applies them is
// reading parameters that do not exist.
std::vector<float> LayerNormNoAffine(const std::vector<float>& x, int64_t frames, int64_t hidden,
                                     double eps);

struct FinalLayerWeights {
  // adaLN_modulation is SiLU -> Linear(hidden, 2*hidden); the output CHUNKS into
  // [shift, scale] in that order.
  std::vector<float> ada_w, ada_b;
  // The output projection is weight-normed: (g, v) with w = g * v / ||v||.
  std::vector<float> linear_g, linear_v, linear_bias;
};

// FinalLayer::forward — norm (no affine) -> modulate -> weight-normed linear.
// Returns [frames, out_channels].
std::vector<float> FinalLayer(const std::vector<float>& x, int64_t frames, int64_t hidden,
                              int64_t out_channels, const std::vector<float>& cond,
                              const FinalLayerWeights& weights, double eps);

}  // namespace adaln
}  // namespace models
}  // namespace vllm
