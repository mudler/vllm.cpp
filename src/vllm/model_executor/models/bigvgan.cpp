// BigVGAN generator. See bigvgan.h for the upstream anchors.
#include "vllm/model_executor/models/bigvgan.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "vllm/model_executor/models/vocoder1d.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace bigvgan {
namespace {

// nn.Conv1d(..., padding=p): zero-pad then convolve, keeping the length.
std::vector<float> ConvSame(const std::vector<float>& x, int64_t in_ch, int64_t len,
                            const ConvSpec& c, int64_t out_ch, int64_t kernel,
                            int64_t dilation, int64_t* out_len) {
  const int64_t pad = dilation * (kernel - 1) / 2;
  int64_t padded_len = 0;
  const std::vector<float> padded =
      vocoder1d::Pad1d(x, in_ch, len, pad, pad, /*replicate=*/false, &padded_len);
  const std::vector<float>* bias = c.bias.empty() ? nullptr : &c.bias;
  return vocoder1d::Conv1d(padded, in_ch, padded_len, c.weight, bias, out_ch, kernel,
                           /*stride=*/1, dilation, /*groups=*/1, out_len);
}

}  // namespace

std::vector<float> Forward(const Config& cfg, const Weights& w,
                           const std::vector<float>& x, int64_t frames) {
  VT_CHECK(cfg.mels > 0 && cfg.init_channels > 0 && frames > 0,
           "bigvgan: mels, init_channels and frames must be positive");
  VT_CHECK(cfg.up_rates.size() == cfg.up_kernels.size() &&
               w.ups.size() == cfg.up_rates.size(),
           "bigvgan: one kernel and one ConvTranspose per upsample rate");
  VT_CHECK(cfg.num_kernels > 0 &&
               w.resblocks.size() ==
                   cfg.up_rates.size() * static_cast<size_t>(cfg.num_kernels),
           "bigvgan: resblocks must be num_upsamples * num_kernels");
  VT_CHECK(x.size() == static_cast<size_t>(cfg.mels * frames),
           "bigvgan: x must be [mels, frames]");

  // conv_pre: kernel 7, padding 3.
  int64_t len = 0;
  std::vector<float> cur =
      ConvSame(x, cfg.mels, frames, w.conv_pre, cfg.init_channels, 7, 1, &len);
  int64_t channels = cfg.init_channels;

  const int64_t stages = static_cast<int64_t>(cfg.up_rates.size());
  for (int64_t i = 0; i < stages; ++i) {
    const int64_t rate = cfg.up_rates[static_cast<size_t>(i)];
    const int64_t kernel = cfg.up_kernels[static_cast<size_t>(i)];
    const int64_t out_ch = channels / 2;
    // ups[i] is ConvTranspose1d(ch, ch/2, k, stride=rate, padding=(k-rate)/2).
    const ConvSpec& up = w.ups[static_cast<size_t>(i)];
    const std::vector<float>* up_bias = up.bias.empty() ? nullptr : &up.bias;
    int64_t up_len = 0;
    cur = vocoder1d::ConvTranspose1d(cur, channels, len, up.weight, up_bias, out_ch,
                                     kernel, rate, (kernel - rate) / 2, 1, &up_len);
    channels = out_ch;
    len = up_len;

    // The AMP blocks over this stage are AVERAGED, not summed.
    std::vector<float> acc(cur.size(), 0.0F);
    for (int64_t j = 0; j < cfg.num_kernels; ++j) {
      const AmpBlock& rb =
          w.resblocks[static_cast<size_t>(i * cfg.num_kernels + j)];
      VT_CHECK(rb.convs1.size() == rb.dilations.size() &&
                   rb.convs2.size() == rb.dilations.size(),
               "bigvgan: an AMP block's conv counts must match its dilations");
      VT_CHECK(rb.alpha.size() == 2 * rb.dilations.size(),
               "bigvgan: an AMP block has two activations per dilation");

      // AMPBlock1.forward (bigvgan.py:132-141): `x` carries forward and the
      // residual is added ONCE PER (conv1, conv2) PAIR, not once at the end.
      std::vector<float> h = cur;
      int64_t h_len = len;
      for (size_t d = 0; d < rb.dilations.size(); ++d) {
        vocoder1d::AliasFreeActivation1d act;
        act.Build();
        int64_t xt_len = 0;
        std::vector<float> xt = act.Apply(h, channels, h_len, rb.alpha[2 * d],
                                          &rb.beta[2 * d], cfg.snake_logscale, &xt_len);
        xt = ConvSame(xt, channels, xt_len, rb.convs1[d], channels, rb.kernel,
                      rb.dilations[d], &xt_len);
        xt = act.Apply(xt, channels, xt_len, rb.alpha[2 * d + 1], &rb.beta[2 * d + 1],
                       cfg.snake_logscale, &xt_len);
        xt = ConvSame(xt, channels, xt_len, rb.convs2[d], channels, rb.kernel, 1,
                      &xt_len);
        VT_CHECK(xt.size() == h.size(), "bigvgan: an AMP pair changed the length");
        for (size_t k = 0; k < h.size(); ++k) {
          h[k] += xt[k];
        }
        h_len = xt_len;
      }
      for (size_t k = 0; k < acc.size(); ++k) {
        acc[k] += h[k];
      }
    }
    for (float& v : acc) {
      v /= static_cast<float>(cfg.num_kernels);
    }
    cur = acc;
  }

  vocoder1d::AliasFreeActivation1d post;
  post.Build();
  int64_t post_len = 0;
  cur = post.Apply(cur, channels, len, w.post_alpha, &w.post_beta, cfg.snake_logscale,
                   &post_len);
  len = post_len;

  int64_t out_len = 0;
  std::vector<float> wave =
      ConvSame(cur, channels, len, w.conv_post, 1, 7, 1, &out_len);

  for (float& v : wave) {
    v = cfg.tanh_at_final ? std::tanh(v) : std::clamp(v, -1.0F, 1.0F);
  }
  return wave;
}

}  // namespace bigvgan
}  // namespace models
}  // namespace vllm
