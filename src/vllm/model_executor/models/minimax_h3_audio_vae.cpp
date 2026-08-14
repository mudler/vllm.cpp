// MiniMax-H3 AUDIO VAE — a DAC-lineage encoder + BigVGAN vocoder, REIMPLEMENTED.
//
// Two halves live here. The DECODER (synthesis) is what generation needs; the
// ENCODER (analysis, at the bottom of this file) is what a REFERENCE AUDIO needs,
// because ref2va conditions on rows produced from a supplied waveform.
//
// ─── WHY REIMPLEMENTED AND NOT ADAPTED ───────────────────────────────────────
// H3's two VAEs are checkpoint REMOTE CODE: their implementations ship inside the
// HF repo (`FL2VA/audio_vae/*.py`) and are loaded through
// `get_class_from_dynamic_module` under `trust_remote_code`; vLLM-Omni's `vae.py`
// only ADAPTS them (vae.py:41-53). A no-Python engine cannot do that, so this is a
// from-scratch port of the checkpoint's own modules, gated against them by
// scripts/gen-minimax-h3-audio-vae-goldens.py (which imports the remote code and
// runs it at reduced dimensions as the oracle). The remote code is NOT vendored
// here — it ships under the MiniMax H3 Community License with the checkpoint.
//
// ─── ARCHITECTURE (checkpoint config.yaml + metadata.json) ───────────────────
//   decode(z[32, T]) = dec_in_proj (Conv1d 32 -> 2048, k=1) -> BigVGAN
//   BigVGAN: conv_pre(2048 -> 1024, k=7)
//            x7 [ ConvTranspose1d upsample (rates 5,5,2,2,2,2,2;
//                                           kernels 9,9,4,4,4,4,4)
//                 then 3 AMPBlock1 residual blocks (kernels 3,7,11,
//                 dilations 1,3,5) whose outputs are AVERAGED ]
//            anti-aliased SnakeBeta -> conv_post(-> 1 ch, k=7, no bias)
//            -> clamp[-1, 1]   (use_tanh_at_final is false for H3)
//   32 kHz, 2 channels; the DiT's audio rows are decoded per channel.
//
// Two details that are easy to get wrong and are gated explicitly:
//   * Every conv is WEIGHT-NORMALIZED. The checkpoint stores (g, v) as
//     `...parametrizations.weight.original0` / `original1`, and the effective
//     weight is `g * v / ||v||` with the norm taken over every dim except dim 0.
//   * The anti-aliased activation upsamples 2x, applies SnakeBeta, then
//     downsamples 2x, both through a KAISER-SINC filter built at load time (never
//     loaded from the checkpoint) with REPLICATE padding.
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/model_executor/models/vocoder1d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numbers>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

const std::vector<float>& MiniMaxH3AudioVaeWeights::Get(const std::string& name) const {
  const auto it = tensors.find(name);
  VT_CHECK(it != tensors.end(), "minimax_h3 audio vae: missing checkpoint tensor");
  return it->second;
}


std::vector<float> MiniMaxH3MaterializeWeightNorm(const std::vector<float>& g,
                                                  const std::vector<float>& v,
                                                  int64_t out_channels) {
  VT_CHECK(out_channels > 0 && v.size() % static_cast<size_t>(out_channels) == 0,
           "minimax_h3 audio vae: weight-norm direction does not divide by out_channels");
  const int64_t per_out = static_cast<int64_t>(v.size()) / out_channels;
  VT_CHECK(static_cast<int64_t>(g.size()) == out_channels,
           "minimax_h3 audio vae: weight-norm magnitude must have one value per output channel");
  std::vector<float> out(v.size());
  for (int64_t c = 0; c < out_channels; ++c) {
    double norm = 0.0;
    for (int64_t i = 0; i < per_out; ++i) {
      const double value = v[static_cast<size_t>(c * per_out + i)];
      norm += value * value;
    }
    norm = std::sqrt(norm);
    const double scale = norm > 0.0 ? static_cast<double>(g[static_cast<size_t>(c)]) / norm : 0.0;
    for (int64_t i = 0; i < per_out; ++i) {
      out[static_cast<size_t>(c * per_out + i)] =
          static_cast<float>(v[static_cast<size_t>(c * per_out + i)] * scale);
    }
  }
  return out;
}

// The anti-aliased activation, `Activation1d`: upsample by `ratio` -> Snake(Beta)
// -> downsample by `ratio` (MiniMax-H3: dac_alias_free_act.py +
// dac_alias_free_resample.py; LTX-2.5: audio_vae/vocoder.py:104-184). Declared in
// minimax_h3.h and shared with ltx2_audio_vae.cpp — the pad/trim geometry below is
// exactly the arithmetic that must not exist twice.

namespace {

int64_t GetPadding(int64_t kernel_size, int64_t dilation) {
  return (kernel_size * dilation - dilation) / 2;
}

}  // namespace

// BigVGAN.forward (dac_bigvgan.py:170-195) preceded by DacAudioVAE.dec_in_proj.
std::vector<float> MiniMaxH3AudioVaeDecode(const MiniMaxH3AudioVaeConfig& config,
                                           const MiniMaxH3AudioVaeWeights& weights,
                                           const std::vector<float>& latent, int64_t frames,
                                           int64_t* out_samples) {
  const int64_t num_upsamples = static_cast<int64_t>(config.upsample_rates.size());
  const int64_t num_kernels = static_cast<int64_t>(config.resblock_kernel_sizes.size());
  VT_CHECK(num_upsamples > 0 && num_kernels > 0, "minimax_h3 audio vae: empty decoder config");
  VT_CHECK(static_cast<int64_t>(config.upsample_kernel_sizes.size()) == num_upsamples,
           "minimax_h3 audio vae: upsample rates/kernels length mismatch");
  VT_CHECK(static_cast<int64_t>(config.resblock_dilation_sizes.size()) == num_kernels,
           "minimax_h3 audio vae: resblock kernels/dilations length mismatch");
  vocoder1d::AliasFreeActivation1d act;
  act.Build();

  auto conv_weight = [&](const std::string& prefix, int64_t out_channels) {
    return MiniMaxH3MaterializeWeightNorm(weights.Get(prefix + ".parametrizations.weight.original0"),
                                          weights.Get(prefix + ".parametrizations.weight.original1"),
                                          out_channels);
  };

  // --- dec_in_proj: vocoder1d::Conv1d(vae_latent_channels -> num_mels, k=1) ---
  // DacAudioVAE.decode applies this BEFORE BigVGAN (dac_audio_vae.py:218-231). It
  // is absent when the caller already supplies a num_mels-wide tensor, which is
  // what the standalone BigVGAN gate does.
  std::vector<float> mels;
  const float* mel_source = latent.data();
  if (weights.Has("dec_in_proj.weight")) {
    const std::vector<float>& w = weights.Get("dec_in_proj.weight");
    VT_CHECK(static_cast<int64_t>(w.size()) % config.num_mels == 0,
             "minimax_h3 audio vae: dec_in_proj weight does not divide by num_mels");
    const int64_t in_channels = static_cast<int64_t>(w.size()) / config.num_mels;
    VT_CHECK(static_cast<int64_t>(latent.size()) == in_channels * frames,
             "minimax_h3 audio vae: latent does not match dec_in_proj input channels");
    int64_t projected_len = 0;
    const std::vector<float>* bias =
        weights.Has("dec_in_proj.bias") ? &weights.Get("dec_in_proj.bias") : nullptr;
    mels = vocoder1d::Conv1d(latent, in_channels, frames, w, bias, config.num_mels, 1, 1, 1, 1, &projected_len);
    VT_CHECK(projected_len == frames, "minimax_h3 audio vae: dec_in_proj changed the length");
    mel_source = mels.data();
  } else {
    VT_CHECK(static_cast<int64_t>(latent.size()) == config.num_mels * frames,
             "minimax_h3 audio vae: latent size does not match [num_mels, frames]");
  }

  // --- conv_pre: vocoder1d::Conv1d(num_mels -> upsample_initial_channel, k=7, padding=3) ---
  int64_t channels = config.upsample_initial_channel;
  int64_t length = 0;
  std::vector<float> x;
  {
    std::vector<float> padded(static_cast<size_t>(config.num_mels * (frames + 6)), 0.0f);
    for (int64_t c = 0; c < config.num_mels; ++c) {
      for (int64_t t = 0; t < frames; ++t) {
        padded[static_cast<size_t>(c * (frames + 6) + 3 + t)] = mel_source[c * frames + t];
      }
    }
    const int64_t padded_len = frames + 6;  // zero padding 3 on each side
    const std::vector<float> w = conv_weight("conv_pre", channels);
    const std::vector<float>& b = weights.Get("conv_pre.bias");
    x = vocoder1d::Conv1d(padded, config.num_mels, padded_len, w, &b, channels, 7, 1, 1, 1, &length);
  }

  // --- upsample stages ---
  for (int64_t i = 0; i < num_upsamples; ++i) {
    const int64_t rate = config.upsample_rates[static_cast<size_t>(i)];
    const int64_t kernel = config.upsample_kernel_sizes[static_cast<size_t>(i)];
    const int64_t out_channels = config.upsample_initial_channel / (int64_t{1} << (i + 1));
    const std::string prefix = "ups." + std::to_string(i) + ".0";
    // ConvTranspose1d weight is [in, out/groups, k] -> weight-norm dim 0 is IN.
    const std::vector<float> w = conv_weight(prefix, channels);
    const std::vector<float>& b = weights.Get(prefix + ".bias");
    int64_t up_len = 0;
    x = vocoder1d::ConvTranspose1d(x, channels, length, w, &b, out_channels, kernel, rate,
                        /*padding=*/(kernel - rate) / 2, /*groups=*/1, &up_len);
    channels = out_channels;
    length = up_len;

    // --- the num_kernels AMPBlock1s, AVERAGED ---
    std::vector<float> sum(static_cast<size_t>(channels * length), 0.0f);
    for (int64_t j = 0; j < num_kernels; ++j) {
      const int64_t kernel_size = config.resblock_kernel_sizes[static_cast<size_t>(j)];
      const std::vector<int64_t>& dilations = config.resblock_dilation_sizes[static_cast<size_t>(j)];
      const std::string block = "resblocks." + std::to_string(i * num_kernels + j);
      std::vector<float> h = x;
      for (size_t d = 0; d < dilations.size(); ++d) {
        const int64_t dilation = dilations[d];
        const std::string c1 = block + ".convs1." + std::to_string(d);
        const std::string c2 = block + ".convs2." + std::to_string(d);
        const std::string a1 = block + ".activations." + std::to_string(2 * d);
        const std::string a2 = block + ".activations." + std::to_string(2 * d + 1);

        int64_t act_len = 0;
        std::vector<float> xt = act.Apply(h, channels, length, weights.Get(a1 + ".act.alpha"),
                                          &weights.Get(a1 + ".act.beta"), config.snake_logscale,
                                          &act_len);
        VT_CHECK(act_len == length, "minimax_h3 audio vae: anti-aliased activation changed length");
        int64_t padded_len = 0;
        const int64_t pad1 = GetPadding(kernel_size, dilation);
        std::vector<float> padded(static_cast<size_t>(channels * (length + 2 * pad1)), 0.0f);
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t t = 0; t < length; ++t) {
            padded[static_cast<size_t>(c * (length + 2 * pad1) + pad1 + t)] =
                xt[static_cast<size_t>(c * length + t)];
          }
        }
        padded_len = length + 2 * pad1;
        int64_t conv_len = 0;
        xt = vocoder1d::Conv1d(padded, channels, padded_len, conv_weight(c1, channels),
                             &weights.Get(c1 + ".bias"), channels, kernel_size, 1, dilation, 1,
                             &conv_len);

        xt = act.Apply(xt, channels, conv_len, weights.Get(a2 + ".act.alpha"),
                       &weights.Get(a2 + ".act.beta"), config.snake_logscale, &act_len);
        const int64_t pad2 = GetPadding(kernel_size, 1);
        std::vector<float> padded2(static_cast<size_t>(channels * (act_len + 2 * pad2)), 0.0f);
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t t = 0; t < act_len; ++t) {
            padded2[static_cast<size_t>(c * (act_len + 2 * pad2) + pad2 + t)] =
                xt[static_cast<size_t>(c * act_len + t)];
          }
        }
        int64_t conv2_len = 0;
        xt = vocoder1d::Conv1d(padded2, channels, act_len + 2 * pad2, conv_weight(c2, channels),
                             &weights.Get(c2 + ".bias"), channels, kernel_size, 1, 1, 1,
                             &conv2_len);
        VT_CHECK(conv2_len == length, "minimax_h3 audio vae: resblock changed the sequence length");
        for (size_t n = 0; n < h.size(); ++n) h[n] += xt[n];
      }
      for (size_t n = 0; n < sum.size(); ++n) sum[n] += h[n];
    }
    for (float& value : sum) value /= static_cast<float>(num_kernels);
    x.swap(sum);
  }

  // --- post activation + conv_post + bound ---
  {
    int64_t act_len = 0;
    x = act.Apply(x, channels, length, weights.Get("activation_post.act.alpha"),
                  &weights.Get("activation_post.act.beta"), config.snake_logscale, &act_len);
    length = act_len;
  }
  {
    const int64_t pad = 3;
    std::vector<float> padded(static_cast<size_t>(channels * (length + 2 * pad)), 0.0f);
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t t = 0; t < length; ++t) {
        padded[static_cast<size_t>(c * (length + 2 * pad) + pad + t)] =
            x[static_cast<size_t>(c * length + t)];
      }
    }
    const std::vector<float> w = conv_weight("conv_post", 1);
    const std::vector<float>* b = nullptr;
    std::vector<float> bias_storage;
    if (config.use_bias_at_final) {
      bias_storage = weights.Get("conv_post.bias");
      b = &bias_storage;
    }
    int64_t final_len = 0;
    x = vocoder1d::Conv1d(padded, channels, length + 2 * pad, w, b, 1, 7, 1, 1, 1, &final_len);
    length = final_len;
  }
  // H3 sets use_tanh_at_final=false, so the output is CLAMPED, not squashed.
  for (float& value : x) {
    value = config.use_tanh_at_final ? std::tanh(value) : std::min(1.0f, std::max(-1.0f, value));
  }
  *out_samples = length;
  return x;
}

// ===========================================================================
// AUDIO VAE ENCODER — the analysis half (dac_audio_vae.py:25-117)
//
// A DAC encoder: one wide input conv, then one strided EncoderBlock per rate
// (each three dilated ResidualUnits followed by a Snake1d and the downsampling
// conv), then a final Snake1d + conv to the latent width. Every conv is
// weight-normalized, exactly like the decoder's.
//
// After it, the reference-audio path runs `pre_block` (an AttnProjection) and
// `mean_proj`, which is how vLLM-Omni composes an encode the shipped module does
// not expose (vae.py:317-325).
// ===========================================================================

namespace {

// Snake1d (dac_audio_vae.py:25-40):  x + (alpha + 1e-9)^-1 * sin(alpha * x)^2.
// NOT the decoder's SnakeBeta: there is one parameter, it is NEVER log-scaled
// (upstream initializes it to ones), and it both scales the sine and forms the
// reciprocal.
void Snake1d(std::vector<float>& x, int64_t channels, int64_t length,
             const std::vector<float>& alpha) {
  VT_CHECK(static_cast<int64_t>(alpha.size()) == channels,
           "minimax_h3 audio encoder: Snake1d alpha must have one value per channel");
  for (int64_t c = 0; c < channels; ++c) {
    const double a = alpha[static_cast<size_t>(c)];
    const double inv = 1.0 / (a + 1e-9);
    for (int64_t t = 0; t < length; ++t) {
      const double v = x[static_cast<size_t>(c * length + t)];
      const double s = std::sin(a * v);
      x[static_cast<size_t>(c * length + t)] = static_cast<float>(v + inv * s * s);
    }
  }
}

// Zero-pad `left`/`right` along time, which is what nn.Conv1d's `padding=` does.
std::vector<float> PadZero(const std::vector<float>& in, int64_t channels, int64_t in_len,
                           int64_t left, int64_t right, int64_t* out_len) {
  const int64_t length = in_len + left + right;
  std::vector<float> out(static_cast<size_t>(channels * length), 0.0f);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < in_len; ++t) {
      out[static_cast<size_t>(c * length + left + t)] = in[static_cast<size_t>(c * in_len + t)];
    }
  }
  *out_len = length;
  return out;
}

// One weight-normalized Conv1d, materializing (g, v) the way the decoder does.
std::vector<float> WnConv1d(const MiniMaxH3AudioVaeWeights& weights, const std::string& prefix,
                            const std::vector<float>& in, int64_t in_channels, int64_t in_len,
                            int64_t out_channels, int64_t kernel, int64_t stride, int64_t dilation,
                            int64_t padding, int64_t* out_len) {
  const std::vector<float> w =
      MiniMaxH3MaterializeWeightNorm(weights.Get(prefix + ".parametrizations.weight.original0"),
                                     weights.Get(prefix + ".parametrizations.weight.original1"),
                                     out_channels);
  VT_CHECK(static_cast<int64_t>(w.size()) == out_channels * in_channels * kernel,
           "minimax_h3 audio encoder: conv weight does not match [out, in, k]");
  const std::vector<float>& bias = weights.Get(prefix + ".bias");
  int64_t padded_len = 0;
  const std::vector<float> padded = PadZero(in, in_channels, in_len, padding, padding, &padded_len);
  return vocoder1d::Conv1d(padded, in_channels, padded_len, w, &bias, out_channels, kernel, stride, dilation,
                /*groups=*/1, out_len);
}

// ResidualUnit (dac_audio_vae.py:50-66). The k=7 conv's `padding` keeps the
// length, so the trim upstream performs is a no-op here — it is still written out,
// because a config where it is NOT a no-op would otherwise silently misalign.
std::vector<float> ResidualUnit(const MiniMaxH3AudioVaeWeights& weights, const std::string& prefix,
                                const std::vector<float>& x, int64_t dim, int64_t length,
                                int64_t dilation, int64_t* out_len) {
  const int64_t pad = ((7 - 1) * dilation) / 2;
  std::vector<float> y = x;
  Snake1d(y, dim, length, weights.Get(prefix + ".block.0.alpha"));
  int64_t len1 = 0;
  y = WnConv1d(weights, prefix + ".block.1", y, dim, length, dim, /*kernel=*/7, /*stride=*/1,
               dilation, pad, &len1);
  Snake1d(y, dim, len1, weights.Get(prefix + ".block.2.alpha"));
  int64_t len2 = 0;
  y = WnConv1d(weights, prefix + ".block.3", y, dim, len1, dim, /*kernel=*/1, /*stride=*/1,
               /*dilation=*/1, /*padding=*/0, &len2);

  const int64_t trim = (length - len2) / 2;
  VT_CHECK(trim >= 0, "minimax_h3 audio encoder: residual unit grew the sequence");
  std::vector<float> out(static_cast<size_t>(dim * len2));
  for (int64_t c = 0; c < dim; ++c) {
    for (int64_t t = 0; t < len2; ++t) {
      out[static_cast<size_t>(c * len2 + t)] =
          x[static_cast<size_t>(c * length + trim + t)] + y[static_cast<size_t>(c * len2 + t)];
    }
  }
  *out_len = len2;
  return out;
}

int64_t CeilDiv2(int64_t stride) {  // math.ceil(stride / 2)
  return (stride + 1) / 2;
}

// EncoderBlock (dac_audio_vae.py:69-87): three dilated residual units at dim/2,
// then Snake1d, then the strided conv that doubles the channel count.
std::vector<float> EncoderBlock(const MiniMaxH3AudioVaeWeights& weights, const std::string& prefix,
                                const std::vector<float>& x, int64_t dim, int64_t length,
                                int64_t stride, int64_t* out_len) {
  VT_CHECK(dim % 2 == 0, "minimax_h3 audio encoder: EncoderBlock dim must be even");
  const int64_t half = dim / 2;
  std::vector<float> h = x;
  int64_t len = length;
  static const int64_t kDilations[3] = {1, 3, 9};
  for (int64_t u = 0; u < 3; ++u) {
    int64_t next = 0;
    h = ResidualUnit(weights, prefix + ".block." + std::to_string(u), h, half, len, kDilations[u],
                     &next);
    len = next;
  }
  Snake1d(h, half, len, weights.Get(prefix + ".block.3.alpha"));
  return WnConv1d(weights, prefix + ".block.4", h, half, len, dim, /*kernel=*/2 * stride, stride,
                  /*dilation=*/1, CeilDiv2(stride), out_len);
}

// nn.LayerNorm over the LAST dimension of [rows, dim].
void LayerNorm(std::vector<float>& x, int64_t rows, int64_t dim, const std::vector<float>& gamma,
               const std::vector<float>& beta, double eps) {
  VT_CHECK(static_cast<int64_t>(gamma.size()) == dim && static_cast<int64_t>(beta.size()) == dim,
           "minimax_h3 audio encoder: LayerNorm weight/bias width mismatch");
  for (int64_t r = 0; r < rows; ++r) {
    float* row = x.data() + r * dim;
    double sum = 0.0;
    for (int64_t i = 0; i < dim; ++i) sum += row[i];
    const double mean = sum / static_cast<double>(dim);
    double var = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
      const double d = row[i] - mean;
      var += d * d;
    }
    // torch normalizes by N (biased), not N-1.
    const double inv = 1.0 / std::sqrt(var / static_cast<double>(dim) + eps);
    for (int64_t i = 0; i < dim; ++i) {
      row[i] = static_cast<float>((row[i] - mean) * inv * gamma[static_cast<size_t>(i)] +
                                  beta[static_cast<size_t>(i)]);
    }
  }
}

// y[r, o] = sum_i x[r, i] * w[o, i] + b[o]  — torch's nn.Linear layout.
std::vector<float> Linear(const std::vector<float>& x, int64_t rows, int64_t in_dim,
                          const std::vector<float>& w, const std::vector<float>* b,
                          int64_t out_dim) {
  VT_CHECK(static_cast<int64_t>(w.size()) == out_dim * in_dim,
           "minimax_h3 audio encoder: linear weight is not [out, in]");
  std::vector<float> out(static_cast<size_t>(rows * out_dim));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_dim; ++o) {
      double acc = b != nullptr ? (*b)[static_cast<size_t>(o)] : 0.0;
      for (int64_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_dim + i)]) *
               static_cast<double>(w[static_cast<size_t>(o * in_dim + i)]);
      }
      out[static_cast<size_t>(r * out_dim + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

// nn.GELU(approximate="tanh").
float GeluTanh(float x) {
  const double v = x;
  const double inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
  return static_cast<float>(0.5 * v * (1.0 + std::tanh(inner)));
}

// F.adaptive_avg_pool1d over the last dimension of [rows, in_dim].
std::vector<float> AdaptiveAvgPool1d(const std::vector<float>& x, int64_t rows, int64_t in_dim,
                                     int64_t out_dim) {
  VT_CHECK(out_dim > 0 && in_dim > 0, "minimax_h3 audio encoder: adaptive pool needs positive dims");
  std::vector<float> out(static_cast<size_t>(rows * out_dim));
  for (int64_t o = 0; o < out_dim; ++o) {
    // torch's bin bounds: start = floor(o * in / out), end = ceil((o+1) * in / out).
    const int64_t start = (o * in_dim) / out_dim;
    const int64_t end = ((o + 1) * in_dim + out_dim - 1) / out_dim;
    const int64_t count = end - start;
    for (int64_t r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (int64_t i = start; i < end; ++i) acc += x[static_cast<size_t>(r * in_dim + i)];
      out[static_cast<size_t>(r * out_dim + o)] = static_cast<float>(acc / static_cast<double>(count));
    }
  }
  return out;
}

}  // namespace

int64_t MiniMaxH3AudioVaeEncoderConfig::hop_length() const {
  int64_t hop = 1;
  for (int64_t rate : encoder_rates) {
    VT_CHECK(rate > 0, "minimax_h3 audio encoder: encoder rates must be positive");
    hop *= rate;
  }
  return hop;
}

int64_t MiniMaxH3AudioVaeEncoderConfig::attn_proj_dim() const {
  VT_CHECK(vae_latent_channels > 0, "minimax_h3 audio encoder: vae_latent_channels must be > 0");
  if (latent_dim % vae_latent_channels == 0) return vae_latent_channels;
  // "smallest power of two >= vae_latent_channels" (dac_audio_vae.py:155).
  int64_t pow2 = 1;
  while (pow2 < vae_latent_channels) pow2 *= 2;
  return pow2;
}

std::vector<float> MiniMaxH3AudioVaeEncoderForward(const MiniMaxH3AudioVaeEncoderConfig& config,
                                                   const MiniMaxH3AudioVaeWeights& weights,
                                                   const std::vector<float>& samples,
                                                   int64_t sample_count, int64_t* out_frames) {
  VT_CHECK(sample_count > 0, "minimax_h3 audio encoder: an empty waveform has nothing to encode");
  VT_CHECK(static_cast<int64_t>(samples.size()) >= sample_count,
           "minimax_h3 audio encoder: sample_count exceeds the supplied waveform");
  VT_CHECK(!config.encoder_rates.empty(), "minimax_h3 audio encoder: encoder_rates is empty");

  // --- preprocess: right-pad to a whole number of hops (dac_audio_vae.py:201-209).
  const int64_t hop = config.hop_length();
  const int64_t padded_samples = ((sample_count + hop - 1) / hop) * hop;
  std::vector<float> x(static_cast<size_t>(padded_samples), 0.0f);
  std::copy(samples.begin(), samples.begin() + sample_count, x.begin());
  int64_t length = padded_samples;

  // --- block.0: WNConv1d(1, d_model, k=7, padding=3) (dac_audio_vae.py:99).
  int64_t channels = config.encoder_dim;
  {
    int64_t next = 0;
    x = WnConv1d(weights, "block.0", x, /*in_channels=*/1, length, channels, /*kernel=*/7,
                 /*stride=*/1, /*dilation=*/1, /*padding=*/3, &next);
    length = next;
  }

  // --- one EncoderBlock per rate, doubling channels as it downsamples
  // (dac_audio_vae.py:102-104).
  for (size_t i = 0; i < config.encoder_rates.size(); ++i) {
    channels *= 2;
    int64_t next = 0;
    x = EncoderBlock(weights, "block." + std::to_string(i + 1), x, channels, length,
                     config.encoder_rates[i], &next);
    length = next;
  }

  // --- Snake1d + WNConv1d(d_model, d_latent, k=3, padding=1)
  // (dac_audio_vae.py:107-110).
  const std::string tail = "block." + std::to_string(config.encoder_rates.size() + 1);
  Snake1d(x, channels, length, weights.Get(tail + ".alpha"));
  {
    int64_t next = 0;
    x = WnConv1d(weights, "block." + std::to_string(config.encoder_rates.size() + 2), x, channels,
                 length, config.latent_dim, /*kernel=*/3, /*stride=*/1, /*dilation=*/1,
                 /*padding=*/1, &next);
    length = next;
  }

  VT_CHECK(length == padded_samples / hop,
           "minimax_h3 audio encoder: the strided stack did not downsample by hop_length");
  if (out_frames != nullptr) *out_frames = length;
  return x;
}

std::vector<float> MiniMaxH3AudioVaeAttnProjection(const MiniMaxH3AudioVaeEncoderConfig& config,
                                                   const MiniMaxH3AudioVaeWeights& weights,
                                                   const std::vector<float>& tokens,
                                                   int64_t token_count) {
  const int64_t in_dim = config.latent_dim;
  const int64_t out_dim = config.attn_proj_dim();
  const int64_t heads = config.attn_proj_heads;
  VT_CHECK(token_count > 0 && heads > 0, "minimax_h3 audio encoder: attn projection needs tokens");
  VT_CHECK(static_cast<int64_t>(tokens.size()) == token_count * in_dim,
           "minimax_h3 audio encoder: attn projection input is not [tokens, latent_dim]");
  // The checkpoint's AttnProjection NARROWS (2048 -> 32). The widening branch
  // (dac_attn_proj.py:38-44, 63-64) is a different qkv shape and a different
  // head_dim, and no shipped H3 config selects it; refusing beats running an
  // untested second path.
  VT_CHECK(in_dim > out_dim,
           "minimax_h3 audio encoder: only the narrowing AttnProjection branch (in_dim > out_dim) "
           "is ported, which is the one the checkpoint ships");
  VT_CHECK(in_dim % heads == 0,
           "minimax_h3 audio encoder: latent_dim must divide by the attn head count");
  const int64_t head_dim = in_dim / heads;

  // --- x = proj(norm3(x)) + attn(norm1(x))  (dac_attn_proj.py:86) ---
  std::vector<float> h1 = tokens;
  LayerNorm(h1, token_count, in_dim, weights.Get("pre_block.norm1.weight"),
            weights.Get("pre_block.norm1.bias"), config.layer_norm_eps);

  // CausalAttention.forward (dac_attn_proj.py:52-66). The qkv bias is ASSEMBLED,
  // not stored: [q_bias | zero_k_bias | v_bias], so keys carry no bias at all.
  const std::vector<float>& q_bias = weights.Get("pre_block.attn.q_bias");
  const std::vector<float>& v_bias = weights.Get("pre_block.attn.v_bias");
  VT_CHECK(static_cast<int64_t>(q_bias.size()) == in_dim &&
               static_cast<int64_t>(v_bias.size()) == in_dim,
           "minimax_h3 audio encoder: q_bias/v_bias must be latent_dim wide");
  std::vector<float> qkv_bias(static_cast<size_t>(3 * in_dim), 0.0f);
  std::copy(q_bias.begin(), q_bias.end(), qkv_bias.begin());
  if (weights.Has("pre_block.attn.zero_k_bias")) {
    // A registered BUFFER that is zero by construction, but it is serialized, so
    // the checkpoint's own bytes are used rather than assumed.
    const std::vector<float>& zk = weights.Get("pre_block.attn.zero_k_bias");
    VT_CHECK(static_cast<int64_t>(zk.size()) == in_dim,
             "minimax_h3 audio encoder: zero_k_bias must be latent_dim wide");
    std::copy(zk.begin(), zk.end(), qkv_bias.begin() + in_dim);
  }
  std::copy(v_bias.begin(), v_bias.end(), qkv_bias.begin() + 2 * in_dim);
  const std::vector<float> qkv = Linear(h1, token_count, in_dim,
                                        weights.Get("pre_block.attn.qkv.weight"), &qkv_bias,
                                        3 * in_dim);

  // qkv.reshape(B, N, 3, heads, head_dim).permute(2, 0, 3, 1, 4): the head axis is
  // the SLOWEST inside each of q/k/v, and q/k/v are the OUTER split of the row.
  auto at = [&](int64_t which, int64_t token, int64_t head, int64_t d) {
    return static_cast<double>(
        qkv[static_cast<size_t>(token * 3 * in_dim + which * in_dim + head * head_dim + d)]);
  };

  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  // scaled_dot_product_attention(..., is_causal=True), then MEAN over heads.
  std::vector<double> pooled(static_cast<size_t>(token_count * head_dim), 0.0);
  std::vector<double> logits(static_cast<size_t>(token_count));
  for (int64_t head = 0; head < heads; ++head) {
    for (int64_t i = 0; i < token_count; ++i) {
      double max_logit = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j <= i; ++j) {  // causal: keys 0..i only
        double dot = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) dot += at(0, i, head, d) * at(1, j, head, d);
        dot *= scale;
        logits[static_cast<size_t>(j)] = dot;
        max_logit = std::max(max_logit, dot);
      }
      double denom = 0.0;
      for (int64_t j = 0; j <= i; ++j) {
        const double e = std::exp(logits[static_cast<size_t>(j)] - max_logit);
        logits[static_cast<size_t>(j)] = e;
        denom += e;
      }
      for (int64_t j = 0; j <= i; ++j) {
        const double p = logits[static_cast<size_t>(j)] / denom;
        for (int64_t d = 0; d < head_dim; ++d) {
          pooled[static_cast<size_t>(i * head_dim + d)] += p * at(2, j, head, d);
        }
      }
    }
  }
  std::vector<float> head_mean(static_cast<size_t>(token_count * head_dim));
  for (size_t n = 0; n < head_mean.size(); ++n) {
    head_mean[n] = static_cast<float>(pooled[n] / static_cast<double>(heads));
  }

  // in_dim // num_heads != out_dim => adaptive average pool down to out_dim
  // (dac_attn_proj.py:61-62).
  std::vector<float> narrowed = head_dim == out_dim
                                    ? head_mean
                                    : AdaptiveAvgPool1d(head_mean, token_count, head_dim, out_dim);
  const std::vector<float> attn_out =
      Linear(narrowed, token_count, out_dim, weights.Get("pre_block.attn.proj.weight"),
             &weights.Get("pre_block.attn.proj.bias"), out_dim);

  std::vector<float> h3 = tokens;
  LayerNorm(h3, token_count, in_dim, weights.Get("pre_block.norm3.weight"),
            weights.Get("pre_block.norm3.bias"), config.layer_norm_eps);
  std::vector<float> x = Linear(h3, token_count, in_dim, weights.Get("pre_block.proj.weight"),
                                &weights.Get("pre_block.proj.bias"), out_dim);
  for (size_t n = 0; n < x.size(); ++n) x[n] += attn_out[n];

  // --- x = x + mlp(norm2(x))  (dac_attn_proj.py:87) ---
  std::vector<float> h2 = x;
  LayerNorm(h2, token_count, out_dim, weights.Get("pre_block.norm2.weight"),
            weights.Get("pre_block.norm2.bias"), config.layer_norm_eps);
  // GeGluMlp (dac_attn_proj.py:8-25): another LayerNorm, then gelu(w0) * w1 -> w2.
  LayerNorm(h2, token_count, out_dim, weights.Get("pre_block.mlp.norm.weight"),
            weights.Get("pre_block.mlp.norm.bias"), config.layer_norm_eps);
  const std::vector<float>& w0 = weights.Get("pre_block.mlp.w0.weight");
  VT_CHECK(static_cast<int64_t>(w0.size()) % out_dim == 0,
           "minimax_h3 audio encoder: mlp.w0 does not divide by the projection width");
  const int64_t hidden = static_cast<int64_t>(w0.size()) / out_dim;
  std::vector<float> gate = Linear(h2, token_count, out_dim, w0,
                                   &weights.Get("pre_block.mlp.w0.bias"), hidden);
  const std::vector<float> up = Linear(h2, token_count, out_dim,
                                       weights.Get("pre_block.mlp.w1.weight"),
                                       &weights.Get("pre_block.mlp.w1.bias"), hidden);
  for (size_t n = 0; n < gate.size(); ++n) gate[n] = GeluTanh(gate[n]) * up[n];
  const std::vector<float> mlp = Linear(gate, token_count, hidden,
                                        weights.Get("pre_block.mlp.w2.weight"),
                                        &weights.Get("pre_block.mlp.w2.bias"), out_dim);
  for (size_t n = 0; n < x.size(); ++n) x[n] += mlp[n];
  return x;
}

std::vector<float> MiniMaxH3AudioVaeEncodeToLatent(const MiniMaxH3AudioVaeEncoderConfig& config,
                                                   const MiniMaxH3AudioVaeWeights& weights,
                                                   const std::vector<float>& samples,
                                                   int64_t sample_count, int64_t* out_frames) {
  int64_t frames = 0;
  std::vector<float> latent =
      MiniMaxH3AudioVaeEncoderForward(config, weights, samples, sample_count, &frames);
  int64_t width = config.latent_dim;

  if (config.attn_proj) {
    // latent.transpose(1, 2) -> [frames, latent_dim] -> AttnProjection -> transpose back.
    std::vector<float> tokens(static_cast<size_t>(frames * width));
    for (int64_t c = 0; c < width; ++c) {
      for (int64_t t = 0; t < frames; ++t) {
        tokens[static_cast<size_t>(t * width + c)] = latent[static_cast<size_t>(c * frames + t)];
      }
    }
    const int64_t projected = config.attn_proj_dim();
    const std::vector<float> rows = MiniMaxH3AudioVaeAttnProjection(config, weights, tokens, frames);
    latent.assign(static_cast<size_t>(projected * frames), 0.0f);
    for (int64_t c = 0; c < projected; ++c) {
      for (int64_t t = 0; t < frames; ++t) {
        latent[static_cast<size_t>(c * frames + t)] = rows[static_cast<size_t>(t * projected + c)];
      }
    }
    width = projected;
  }

  // mean_proj: a PLAIN vocoder1d::Conv1d(attn_proj_dim -> vae_latent_channels, k=1); the VAE's
  // distribution MEAN. `logs_proj` is never evaluated — a reference must be
  // deterministic, so nothing is sampled.
  const std::vector<float>& w = weights.Get("mean_proj.weight");
  const std::vector<float>& b = weights.Get("mean_proj.bias");
  VT_CHECK(static_cast<int64_t>(w.size()) == config.vae_latent_channels * width,
           "minimax_h3 audio encoder: mean_proj is not [vae_latent_channels, attn_proj_dim, 1]");
  int64_t out_len = 0;
  std::vector<float> mean = vocoder1d::Conv1d(latent, width, frames, w, &b, config.vae_latent_channels,
                                   /*kernel=*/1, /*stride=*/1, /*dilation=*/1, /*groups=*/1,
                                   &out_len);
  VT_CHECK(out_len == frames, "minimax_h3 audio encoder: mean_proj changed the frame count");
  if (out_frames != nullptr) *out_frames = frames;
  return mean;
}

std::vector<float> MiniMaxH3AudioVaeEncodeToRows(const MiniMaxH3AudioVaeEncoderConfig& config,
                                                 const MiniMaxH3AudioVaeWeights& weights,
                                                 const std::vector<float>& waveform,
                                                 int64_t channels, int64_t samples_per_channel,
                                                 const std::vector<float>& latents_mean,
                                                 const std::vector<float>& latents_std,
                                                 int64_t* out_audio_t) {
  VT_CHECK(channels > 0 && samples_per_channel > 0,
           "minimax_h3 audio encoder: a reference waveform needs channels and samples");
  VT_CHECK(static_cast<int64_t>(waveform.size()) == channels * samples_per_channel,
           "minimax_h3 audio encoder: waveform is not [channels, samples_per_channel]");
  const int64_t width = config.vae_latent_channels;
  const bool normalize = !latents_mean.empty() || !latents_std.empty();
  if (normalize) {
    VT_CHECK(static_cast<int64_t>(latents_mean.size()) == width &&
                 static_cast<int64_t>(latents_std.size()) == width,
             "minimax_h3 audio encoder: latents_mean/std must have vae_latent_channels values");
  }

  // Channels encode INDEPENDENTLY: upstream hands the VAE a [C, 1, T] batch, so
  // each channel is its own [1, T] clip (vae.py:317-322).
  std::vector<float> rows;
  int64_t frames = 0;
  for (int64_t c = 0; c < channels; ++c) {
    const std::vector<float> channel(waveform.begin() + c * samples_per_channel,
                                     waveform.begin() + (c + 1) * samples_per_channel);
    int64_t f = 0;
    const std::vector<float> latent =
        MiniMaxH3AudioVaeEncodeToLatent(config, weights, channel, samples_per_channel, &f);
    if (c == 0) {
      frames = f;
      rows.assign(static_cast<size_t>(channels * frames * width), 0.0f);
    } else {
      VT_CHECK(f == frames, "minimax_h3 audio encoder: channels produced different frame counts");
    }
    // [vae_latent_channels, frames] -> the packed row layout [frames,
    // vae_latent_channels], normalized per channel-of-the-latent (vae.py:327-341).
    for (int64_t t = 0; t < frames; ++t) {
      for (int64_t d = 0; d < width; ++d) {
        double v = latent[static_cast<size_t>(d * frames + t)];
        if (normalize) {
          v = (v - latents_mean[static_cast<size_t>(d)]) / latents_std[static_cast<size_t>(d)];
        }
        rows[static_cast<size_t>(((c * frames) + t) * width + d)] = static_cast<float>(v);
      }
    }
  }
  if (out_audio_t != nullptr) *out_audio_t = frames;
  return rows;
}

}  // namespace vllm
