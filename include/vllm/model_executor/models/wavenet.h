// WaveNet stack — the S2Mel DiT's FINAL LAYER (#634).
//
// Upstream `indextts/s2mel/modules/wavenet.py` class `WN`, index-tts
// @4f8792ff120cd3ea470dd511e997a17c86cddd10. The shipped config sets
// `s2mel.DiT.final_layer_type: wavenet`, so this is the path the model takes,
// not an alternative branch: `net.cfm.estimator.wavenet.*` is in `s2mel.pth`.
//
// Three details are easy to get wrong and are gated:
//   - the convolutions are upstream `SConv1d`, whose default `pad_mode` is
//     REFLECT, not zero. With stride 1 the extra-padding term is always 0 and
//     the padding is symmetric, but it is still a reflection.
//   - the gate is `tanh(first half) * sigmoid(second half)` of `x_in + g_l`,
//     where `g_l` is this layer's slice of ONE conditioning projection.
//   - the mask multiplies the residual update and the final output, but NOT the
//     input to `in_layers`, so a masked run is not a shorter run.
//
// Layout is [channels, frames] (channel-major), matching torch's [B, C, T].
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace wavenet {

// One weight-normed SConv1d. The checkpoint carries the LEGACY (g, v) spelling,
// `...conv.conv.weight_g` / `weight_v`; the fold itself has one home,
// `vocoder1d::MaterializeWeightNorm`.
struct ConvWeights {
  std::vector<float> g;     // [out_channels]
  std::vector<float> v;     // [out_channels, in_channels, kernel]
  std::vector<float> bias;  // [out_channels]
};

struct Config {
  int64_t hidden = 0;
  int64_t kernel = 0;         // must be odd, as upstream asserts
  int64_t dilation_rate = 1;  // layer i dilates by dilation_rate^i
  int64_t layers = 0;
  int64_t gin = 0;  // 0 disables conditioning and `cond` is then unused
};

struct Weights {
  ConvWeights cond;                    // [gin -> 2 * hidden * layers], kernel 1
  std::vector<ConvWeights> in_layers;  // [hidden -> 2 * hidden], kernel
  // [hidden -> 2 * hidden], except the LAST which is [hidden -> hidden]:
  // upstream notes the extra half would never be read.
  std::vector<ConvWeights> res_skip_layers;
};

// x is [hidden, frames]; g is [gin] (upstream passes [B, gin, 1], one frame);
// mask is [frames] and may be empty for an all-ones mask.
// Returns [hidden, frames].
std::vector<float> Forward(const Config& cfg, const Weights& w,
                           const std::vector<float>& x, int64_t frames,
                           const std::vector<float>& g,
                           const std::vector<float>& mask);

}  // namespace wavenet
}  // namespace models
}  // namespace vllm
