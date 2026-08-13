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

namespace {

// Zeroth-order modified Bessel function of the first kind, matching the series
// torch.kaiser_window uses.
double BesselI0(double x) {
  double sum = 1.0, term = 1.0;
  const double half_x_sq = (x / 2.0) * (x / 2.0);
  for (int k = 1; k < 64; ++k) {
    term *= half_x_sq / (static_cast<double>(k) * static_cast<double>(k));
    sum += term;
    if (term < sum * 1e-18) break;
  }
  return sum;
}

double Sinc(double x) {
  if (x == 0.0) return 1.0;
  const double pix = std::numbers::pi_v<double> * x;
  return std::sin(pix) / pix;
}

// torch.kaiser_window(n, periodic=false, beta).
std::vector<double> KaiserWindow(int64_t length, double beta) {
  std::vector<double> window(static_cast<size_t>(length));
  const double denom = BesselI0(beta);
  // periodic=false => the window spans [0, length-1] inclusive.
  const double n_minus_1 = static_cast<double>(length - 1);
  for (int64_t i = 0; i < length; ++i) {
    const double ratio = (2.0 * static_cast<double>(i) - n_minus_1) / n_minus_1;
    window[static_cast<size_t>(i)] = BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
  }
  return window;
}

}  // namespace

// ---------------------------------------------------------------------------
// The shared 1-D BigVGAN primitives (declared in minimax_h3.h). They were private
// to this translation unit until LTX-2.5's audio VAE — the same BigVGAN lineage —
// needed them and copied them instead; see the header for why one implementation
// gated by two suites beats two implementations each with its own green gate.
// ---------------------------------------------------------------------------

// One 1-D convolution over [C_in, T] with dilation/stride/groups.
// Weight is [C_out, C_in/groups, K]; input is assumed ALREADY padded.
std::vector<float> MiniMaxH3Conv1d(const std::vector<float>& in, int64_t in_channels,
                                   int64_t in_len, const std::vector<float>& weight,
                                   const std::vector<float>* bias, int64_t out_channels,
                                   int64_t kernel, int64_t stride, int64_t dilation,
                                   int64_t groups, int64_t* out_len) {
  const int64_t effective = dilation * (kernel - 1) + 1;
  const int64_t length = (in_len - effective) / stride + 1;
  VT_CHECK(length > 0, "minimax_h3 audio vae: conv1d output length is empty");
  const int64_t in_per_group = in_channels / groups;
  const int64_t out_per_group = out_channels / groups;
  std::vector<float> out(static_cast<size_t>(out_channels * length), 0.0f);
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    const int64_t g = oc / out_per_group;
    for (int64_t t = 0; t < length; ++t) {
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0;
      for (int64_t ic = 0; ic < in_per_group; ++ic) {
        const int64_t src_c = g * in_per_group + ic;
        for (int64_t k = 0; k < kernel; ++k) {
          const int64_t pos = t * stride + k * dilation;
          acc += static_cast<double>(in[static_cast<size_t>(src_c * in_len + pos)]) *
                 static_cast<double>(weight[static_cast<size_t>((oc * in_per_group + ic) * kernel + k)]);
        }
      }
      out[static_cast<size_t>(oc * length + t)] = static_cast<float>(acc);
    }
  }
  *out_len = length;
  return out;
}

// torch.nn.functional.conv_transpose1d over [C_in, T].
// Weight is [C_in, C_out/groups, K]; output length = (T-1)*stride - 2*padding + K.
std::vector<float> MiniMaxH3ConvTranspose1d(const std::vector<float>& in, int64_t in_channels,
                                            int64_t in_len, const std::vector<float>& weight,
                                            const std::vector<float>* bias, int64_t out_channels,
                                            int64_t kernel, int64_t stride, int64_t padding,
                                            int64_t groups, int64_t* out_len) {
  const int64_t full = (in_len - 1) * stride + kernel;
  const int64_t length = full - 2 * padding;
  VT_CHECK(length > 0, "minimax_h3 audio vae: conv_transpose1d output length is empty");
  const int64_t in_per_group = in_channels / groups;
  const int64_t out_per_group = out_channels / groups;
  std::vector<double> acc(static_cast<size_t>(out_channels * full), 0.0);
  for (int64_t ic = 0; ic < in_channels; ++ic) {
    const int64_t g = ic / in_per_group;
    for (int64_t t = 0; t < in_len; ++t) {
      const double value = in[static_cast<size_t>(ic * in_len + t)];
      if (value == 0.0) continue;
      for (int64_t oc = 0; oc < out_per_group; ++oc) {
        const int64_t dst_c = g * out_per_group + oc;
        for (int64_t k = 0; k < kernel; ++k) {
          acc[static_cast<size_t>(dst_c * full + t * stride + k)] +=
              value * static_cast<double>(weight[static_cast<size_t>((ic * out_per_group + oc) * kernel + k)]);
        }
      }
    }
  }
  std::vector<float> out(static_cast<size_t>(out_channels * length));
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      double value = acc[static_cast<size_t>(c * full + t + padding)];
      if (bias != nullptr) value += (*bias)[static_cast<size_t>(c)];
      out[static_cast<size_t>(c * length + t)] = static_cast<float>(value);
    }
  }
  *out_len = length;
  return out;
}

// F.pad along the time axis: mode="replicate", or the zero pad an ordinary
// nn.Conv1d `padding=` argument performs.
std::vector<float> MiniMaxH3Pad1d(const std::vector<float>& in, int64_t channels, int64_t in_len,
                                  int64_t left, int64_t right, bool replicate, int64_t* out_len) {
  const int64_t length = in_len + left + right;
  std::vector<float> out(static_cast<size_t>(channels * length), 0.0f);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      int64_t src = t - left;
      if (src < 0 || src >= in_len) {
        if (!replicate) continue;  // already zero
        src = std::max<int64_t>(0, std::min<int64_t>(in_len - 1, src));
      }
      out[static_cast<size_t>(c * length + t)] = in[static_cast<size_t>(c * in_len + src)];
    }
  }
  *out_len = length;
  return out;
}

// Snake / SnakeBeta: x + (b + kMiniMaxH3SnakeEps)^-1 * sin^2(a * x). A null `beta`
// is plain Snake, which reuses ALPHA as the reciprocal scale (LTX-2.5
// vocoder.py:198); a non-null one is SnakeBeta (vocoder.py:221), which is what
// every MiniMax-H3 checkpoint carries. Both are exponentiated when the checkpoint
// stores them in log scale.
void MiniMaxH3SnakeActivation(std::vector<float>& x, int64_t channels, int64_t length,
                              const std::vector<float>& alpha, const std::vector<float>* beta,
                              bool logscale) {
  for (int64_t c = 0; c < channels; ++c) {
    double a = alpha[static_cast<size_t>(c)];
    double b = beta != nullptr ? (*beta)[static_cast<size_t>(c)] : a;
    if (logscale) {
      a = std::exp(a);
      b = std::exp(b);
    }
    const double inv_beta = 1.0 / (b + kMiniMaxH3SnakeEps);
    for (int64_t t = 0; t < length; ++t) {
      const double v = x[static_cast<size_t>(c * length + t)];
      const double s = std::sin(a * v);
      x[static_cast<size_t>(c * length + t)] = static_cast<float>(v + inv_beta * s * s);
    }
  }
}

// kaiser_sinc_filter1d (dac_alias_free_filter.py:26-60). Returns [kernel_size].
std::vector<float> MiniMaxH3KaiserSincFilter1d(double cutoff, double half_width,
                                               int64_t kernel_size) {
  VT_CHECK(kernel_size > 0, "minimax_h3 audio vae: kernel_size must be positive");
  VT_CHECK(cutoff >= 0.0 && cutoff <= 0.5, "minimax_h3 audio vae: cutoff must be in [0, 0.5]");
  const bool even = (kernel_size % 2) == 0;
  const int64_t half_size = kernel_size / 2;

  const double delta_f = 4.0 * half_width;
  const double a = 2.285 * (static_cast<double>(half_size) - 1.0) *
                       std::numbers::pi_v<double> * delta_f +
                   7.95;
  double beta = 0.0;
  if (a > 50.0) {
    beta = 0.1102 * (a - 8.7);
  } else if (a >= 21.0) {
    beta = 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
  }
  const std::vector<double> window = KaiserWindow(kernel_size, beta);

  std::vector<double> time(static_cast<size_t>(kernel_size));
  for (int64_t i = 0; i < kernel_size; ++i) {
    time[static_cast<size_t>(i)] = even ? (static_cast<double>(-half_size + i) + 0.5)
                                        : static_cast<double>(i - half_size);
  }

  std::vector<double> filter(static_cast<size_t>(kernel_size), 0.0);
  if (cutoff == 0.0) {
    return std::vector<float>(static_cast<size_t>(kernel_size), 0.0f);
  }
  double sum = 0.0;
  for (int64_t i = 0; i < kernel_size; ++i) {
    filter[static_cast<size_t>(i)] =
        2.0 * cutoff * window[static_cast<size_t>(i)] * Sinc(2.0 * cutoff * time[static_cast<size_t>(i)]);
    sum += filter[static_cast<size_t>(i)];
  }
  // Normalized to sum 1 so a constant input does not leak.
  std::vector<float> out(static_cast<size_t>(kernel_size));
  for (int64_t i = 0; i < kernel_size; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(filter[static_cast<size_t>(i)] / sum);
  }
  return out;
}

// torch weight_norm: w = g * v / ||v||, norm over every dim except dim 0.
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
void MiniMaxH3AliasFreeActivation1d::Build() {
  // Up and down use the same cutoff/half_width/kernel, so one window serves both.
  filter = MiniMaxH3KaiserSincFilter1d(0.5 / static_cast<double>(ratio),
                                       0.6 / static_cast<double>(ratio), kernel_size);
}

std::vector<float> MiniMaxH3AliasFreeActivation1d::Apply(const std::vector<float>& in,
                                                         int64_t channels, int64_t in_len,
                                                         const std::vector<float>& alpha,
                                                         const std::vector<float>* beta,
                                                         bool logscale, int64_t* out_len) const {
  // --- UpSample1d ---
  const int64_t pad = kernel_size / ratio - 1;
  const int64_t pad_left = pad * ratio + (kernel_size - ratio) / 2;
  const int64_t pad_right = pad * ratio + (kernel_size - ratio + 1) / 2;
  int64_t padded_len = 0;
  const std::vector<float> padded =
      MiniMaxH3Pad1d(in, channels, in_len, pad, pad, /*replicate=*/true, &padded_len);
  // Depthwise transposed conv: filter.expand(C, -1, -1) => weight [C, 1, K].
  std::vector<float> depthwise(static_cast<size_t>(channels * kernel_size));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t k = 0; k < kernel_size; ++k) {
      depthwise[static_cast<size_t>(c * kernel_size + k)] = filter[static_cast<size_t>(k)];
    }
  }
  int64_t up_len = 0;
  std::vector<float> up =
      MiniMaxH3ConvTranspose1d(padded, channels, padded_len, depthwise, nullptr, channels,
                               kernel_size, ratio, /*padding=*/0, /*groups=*/channels, &up_len);
  for (float& value : up) value *= static_cast<float>(ratio);
  // x[..., pad_left : -pad_right]
  const int64_t trimmed_len = up_len - pad_left - pad_right;
  VT_CHECK(trimmed_len > 0, "minimax_h3 audio vae: upsample trim emptied the signal");
  std::vector<float> trimmed(static_cast<size_t>(channels * trimmed_len));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < trimmed_len; ++t) {
      trimmed[static_cast<size_t>(c * trimmed_len + t)] =
          up[static_cast<size_t>(c * up_len + pad_left + t)];
    }
  }

  // --- Snake / SnakeBeta ---
  MiniMaxH3SnakeActivation(trimmed, channels, trimmed_len, alpha, beta, logscale);

  // --- DownSample1d (LowPassFilter1d, stride = ratio, replicate padding) ---
  const bool even = (kernel_size % 2) == 0;
  const int64_t lp_left = kernel_size / 2 - (even ? 1 : 0);
  const int64_t lp_right = kernel_size / 2;
  int64_t lp_padded_len = 0;
  const std::vector<float> lp_padded = MiniMaxH3Pad1d(trimmed, channels, trimmed_len, lp_left,
                                                      lp_right, /*replicate=*/true, &lp_padded_len);
  return MiniMaxH3Conv1d(lp_padded, channels, lp_padded_len, depthwise, nullptr, channels,
                         kernel_size, /*stride=*/ratio, /*dilation=*/1, /*groups=*/channels,
                         out_len);
}

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
  MiniMaxH3AliasFreeActivation1d act;
  act.Build();

  auto conv_weight = [&](const std::string& prefix, int64_t out_channels) {
    return MiniMaxH3MaterializeWeightNorm(weights.Get(prefix + ".parametrizations.weight.original0"),
                                          weights.Get(prefix + ".parametrizations.weight.original1"),
                                          out_channels);
  };

  // --- dec_in_proj: MiniMaxH3Conv1d(vae_latent_channels -> num_mels, k=1) ---
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
    mels = MiniMaxH3Conv1d(latent, in_channels, frames, w, bias, config.num_mels, 1, 1, 1, 1, &projected_len);
    VT_CHECK(projected_len == frames, "minimax_h3 audio vae: dec_in_proj changed the length");
    mel_source = mels.data();
  } else {
    VT_CHECK(static_cast<int64_t>(latent.size()) == config.num_mels * frames,
             "minimax_h3 audio vae: latent size does not match [num_mels, frames]");
  }

  // --- conv_pre: MiniMaxH3Conv1d(num_mels -> upsample_initial_channel, k=7, padding=3) ---
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
    x = MiniMaxH3Conv1d(padded, config.num_mels, padded_len, w, &b, channels, 7, 1, 1, 1, &length);
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
    x = MiniMaxH3ConvTranspose1d(x, channels, length, w, &b, out_channels, kernel, rate,
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
        xt = MiniMaxH3Conv1d(padded, channels, padded_len, conv_weight(c1, channels),
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
        xt = MiniMaxH3Conv1d(padded2, channels, act_len + 2 * pad2, conv_weight(c2, channels),
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
    x = MiniMaxH3Conv1d(padded, channels, length + 2 * pad, w, b, 1, 7, 1, 1, 1, &final_len);
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
  return MiniMaxH3Conv1d(padded, in_channels, padded_len, w, &bias, out_channels, kernel, stride, dilation,
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

  // mean_proj: a PLAIN MiniMaxH3Conv1d(attn_proj_dim -> vae_latent_channels, k=1); the VAE's
  // distribution MEAN. `logs_proj` is never evaluated — a reference must be
  // deterministic, so nothing is sampled.
  const std::vector<float>& w = weights.Get("mean_proj.weight");
  const std::vector<float>& b = weights.Get("mean_proj.bias");
  VT_CHECK(static_cast<int64_t>(w.size()) == config.vae_latent_channels * width,
           "minimax_h3 audio encoder: mean_proj is not [vae_latent_channels, attn_proj_dim, 1]");
  int64_t out_len = 0;
  std::vector<float> mean = MiniMaxH3Conv1d(latent, width, frames, w, &b, config.vae_latent_channels,
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
