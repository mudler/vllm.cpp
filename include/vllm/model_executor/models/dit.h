// S2Mel DiT block primitives (#634), from gpt-fast as the DiT wraps it.
//
// TWO CONVENTIONS THAT DIFFER FROM THEIR NEIGHBOURS IN THIS SAME MODEL:
//
//  * `AdaptiveLayerNorm` here is `weight * norm(x) + bias` -- NO `1 +`, unlike
//    `adaln::Modulate` used by the DiT's FinalLayer. Two adaLN conventions
//    coexist in one model, so porting one over the other is silent and easy.
//  * The rotary embedding pairs ADJACENT components (`reshape(..., -1, 2)`),
//    not halves of the vector. The half-split convention is far more common and
//    yields a rotation that is smooth, norm-preserving and wrong.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace dit {

// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight. NO mean subtraction -- that is
// what separates it from LayerNorm, and subtracting anyway still normalizes.
std::vector<float> RmsNorm(const std::vector<float>& x, int64_t frames, int64_t dim,
                           const std::vector<float>& weight, double eps);

// AdaptiveLayerNorm: split(project(embedding)) -> (weight, bias), then
// `weight * rms_norm(x) + bias`. The split is [weight, bias] in that order.
std::vector<float> AdaptiveLayerNorm(const std::vector<float>& x, int64_t frames, int64_t dim,
                                     const std::vector<float>& embedding,
                                     const std::vector<float>& proj_w,
                                     const std::vector<float>& proj_b,
                                     const std::vector<float>& norm_weight, double eps);

// apply_rotary_emb over [frames, heads, head_dim], with `freqs` laid out
// [frames, head_dim/2, 2] as (cos, sin).
//
//   out[2i]   = x[2i] * cos - x[2i+1] * sin
//   out[2i+1] = x[2i+1] * cos + x[2i] * sin
//
// The pairs are (0,1), (2,3), ... -- ADJACENT. Pairing i with i + head_dim/2 is
// the other common convention and is wrong here.
std::vector<float> ApplyRotary(const std::vector<float>& x, int64_t frames, int64_t heads,
                               int64_t head_dim, const std::vector<float>& freqs);

}  // namespace dit
}  // namespace models
}  // namespace vllm
