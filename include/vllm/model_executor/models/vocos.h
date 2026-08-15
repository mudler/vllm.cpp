// VocosBackbone — EnhancedCodec's ConvNeXt-1D encoder (#634).
//
// `EnhancedCodec.quantize` runs its input through this before the quantizer
// (indextts/codec/models.py:187). Each block is ConvNeXt adapted to 1-D audio:
// depthwise conv, layer norm, two pointwise linears with GELU, and a LEARNED
// per-channel layer scale.
//
// TWO THINGS THAT ARE EASY TO GET WRONG:
//   eps IS 1e-6 here, not the 1e-5 used elsewhere in this codebase.
//   THE OUTPUT IS [T, dim], not [dim, T]: the final layer norm is applied to the
//   transposed tensor and never transposed back, which is why the caller writes
//   `encoder(x.transpose(1,2)).transpose(1,2)`.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace vocos {

struct BlockWeights {
  std::vector<float> dw_weight, dw_bias;      // depthwise conv, k=7, groups=dim
  std::vector<float> ln_gamma, ln_beta;
  std::vector<float> pw1_w, pw1_b;            // [intermediate, dim]
  std::vector<float> pw2_w, pw2_b;            // [dim, intermediate]
  std::vector<float> gamma;                   // layer scale, [dim]
};

struct BackboneWeights {
  std::vector<float> embed_w, embed_b;        // Conv1d(input_channels -> dim), k=7
  std::vector<float> norm_gamma, norm_beta;
  std::vector<BlockWeights> blocks;
  std::vector<float> final_gamma, final_beta;
};

// One ConvNeXt block over a [dim, frames] signal; returns [dim, frames].
std::vector<float> ConvNeXtBlock(const std::vector<float>& x, int64_t dim, int64_t frames,
                                 int64_t intermediate, const BlockWeights& weights, double eps);

// VocosBackbone: embed -> norm -> blocks -> final norm.
// Input is [input_channels, frames]; OUTPUT IS [frames, dim].
std::vector<float> Backbone(const std::vector<float>& x, int64_t input_channels, int64_t frames,
                            int64_t dim, int64_t intermediate, const BackboneWeights& weights,
                            double eps);

}  // namespace vocos
}  // namespace models
}  // namespace vllm
