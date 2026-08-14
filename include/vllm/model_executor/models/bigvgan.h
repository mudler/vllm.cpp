// BigVGAN generator — the LAST stage of IndexTTS-2.5: mel in, samples out (#634).
//
// Upstream `indextts/s2mel/modules/bigvgan/bigvgan.py` class BigVGAN, index-tts
// @4f8792ff120cd3ea470dd511e997a17c86cddd10:
//
//   x = conv_pre(x)
//   for each upsample stage i:
//       x = ups[i](x)                                  // ConvTranspose1d
//       x = mean over j of resblocks[i * num_kernels + j](x)   // AMPBlock1
//   x = activation_post(x); x = conv_post(x)
//   x = use_tanh_at_final ? tanh(x) : clamp(x, -1, 1)
//
// Composition only: every primitive is `vocoder1d`'s, shared with MiniMax-H3 and
// LTX-2.5 since #681. Nothing here is a fourth copy of a 1-D convolution.
//
// Three details the gate holds:
//   - the AMP blocks over one stage are AVERAGED, not summed. Summing scales the
//     signal by num_kernels and still produces audio, just louder and clipped.
//   - `use_tanh_at_final` is FALSE for this checkpoint, so the bound is a CLAMP.
//     tanh and clamp agree closely in the interior and diverge only near +/-1,
//     which is exactly where a vocoder spends its loudest samples.
//   - `use_bias_at_final` is FALSE, so `conv_post` has NO bias, unlike conv_pre.
//
// The weights here are EFFECTIVE weights: upstream calls `remove_weight_norm()`
// before inference, so the (g, v) fold happens at load through
// `vocoder1d::MaterializeWeightNorm` and never per forward.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace bigvgan {

struct ConvSpec {
  std::vector<float> weight;
  std::vector<float> bias;  // may be empty
};

// One AMPBlock1: pairs of (activation, conv) with dilations, then dilation 1.
struct AmpBlock {
  int64_t kernel = 0;
  std::vector<int64_t> dilations;
  std::vector<ConvSpec> convs1;  // dilated
  std::vector<ConvSpec> convs2;  // dilation 1
  // 2 * dilations.size() activations, interleaved a1, a2, a1, a2, ...
  std::vector<std::vector<float>> alpha;
  std::vector<std::vector<float>> beta;
};

struct Weights {
  ConvSpec conv_pre;
  std::vector<ConvSpec> ups;        // one ConvTranspose1d per stage
  std::vector<AmpBlock> resblocks;  // num_upsamples * num_kernels, stage-major
  std::vector<float> post_alpha, post_beta;
  ConvSpec conv_post;  // NO bias when use_bias_at_final is false
};

struct Config {
  int64_t mels = 0;
  int64_t init_channels = 0;
  std::vector<int64_t> up_rates;
  std::vector<int64_t> up_kernels;
  int64_t num_kernels = 0;
  bool snake_logscale = true;
  bool tanh_at_final = false;
};

// x is [mels, frames]; returns the waveform, [samples], where samples is
// frames * prod(up_rates).
std::vector<float> Forward(const Config& cfg, const Weights& w,
                           const std::vector<float>& x, int64_t frames);

}  // namespace bigvgan
}  // namespace models
}  // namespace vllm
