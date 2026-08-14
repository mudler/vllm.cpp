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


// SwiGLU feed-forward: w2(silu(w1(x)) * w3(x)).
//
// W1 IS THE GATE (it takes the SiLU), w3 is the up-projection. Swapping them
// yields a network of identical shape that trains and generates, and is a
// different function -- the checkpoint stores them as separate tensors, so
// nothing but the values can tell.
std::vector<float> SwiGlu(const std::vector<float>& x, int64_t frames, int64_t dim,
                          int64_t intermediate, const std::vector<float>& w1,
                          const std::vector<float>& w3, const std::vector<float>& w2);

struct BlockWeights {
  std::vector<float> wqkv;          // [3*heads*head_dim, dim], fused q|k|v
  std::vector<float> wo;            // [dim, dim]
  std::vector<float> w1, w3, w2;    // SwiGLU
  std::vector<float> attn_proj_w, attn_proj_b, attn_norm_w;   // attention_norm
  std::vector<float> ffn_proj_w, ffn_proj_b, ffn_norm_w;      // ffn_norm
};

// TransformerBlock::forward (gpt_fast/model.py:221-239):
//
//   h   = x + attention(attention_norm(x, c))
//   out = h + feed_forward(ffn_norm(h, c))
//
// Both norms are AdaptiveLayerNorm conditioned on `c`, and both residuals are
// FULL (no scaling) -- unlike the macaron halves in the w2v-bert Conformer,
// which is a different block type in the same lane.
std::vector<float> Block(const std::vector<float>& x, const std::vector<float>& cond,
                         int64_t frames, int64_t dim, int64_t heads, int64_t head_dim,
                         int64_t intermediate, const std::vector<float>& freqs,
                         const BlockWeights& weights, double eps);

}  // namespace dit
}  // namespace models
}  // namespace vllm
