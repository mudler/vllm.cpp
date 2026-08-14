// Definitions for the shared 1-D BigVGAN vocoder core. See vocoder1d.h.
#include "vllm/model_executor/models/vocoder1d.h"

#include "vt/dtype.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace vllm {
namespace vocoder1d {

namespace {

// TU-private helpers for KaiserSincFilter1d, moved with their only caller.

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
// The shared 1-D BigVGAN primitives (declared in vocoder1d.h). They were private
// to this translation unit until LTX-2.5's audio VAE — the same BigVGAN lineage —
// needed them and copied them instead; see the header for why one implementation
// gated by two suites beats two implementations each with its own green gate.
// ---------------------------------------------------------------------------

// One 1-D convolution over [C_in, T] with dilation/stride/groups.
// Weight is [C_out, C_in/groups, K]; input is assumed ALREADY padded.
std::vector<float> Conv1d(const std::vector<float>& in, int64_t in_channels,
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
std::vector<float> ConvTranspose1d(const std::vector<float>& in, int64_t in_channels,
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
std::vector<float> Pad1d(const std::vector<float>& in, int64_t channels, int64_t in_len,
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

// Snake / SnakeBeta: x + (b + kSnakeEps)^-1 * sin^2(a * x). A null `beta`
// is plain Snake, which reuses ALPHA as the reciprocal scale (LTX-2.5
// vocoder.py:198); a non-null one is SnakeBeta (vocoder.py:221), which is what
// every MiniMax-H3 checkpoint carries. Both are exponentiated when the checkpoint
// stores them in log scale.
void SnakeActivation(std::vector<float>& x, int64_t channels, int64_t length,
                              const std::vector<float>& alpha, const std::vector<float>* beta,
                              bool logscale) {
  for (int64_t c = 0; c < channels; ++c) {
    double a = alpha[static_cast<size_t>(c)];
    double b = beta != nullptr ? (*beta)[static_cast<size_t>(c)] : a;
    if (logscale) {
      a = std::exp(a);
      b = std::exp(b);
    }
    const double inv_beta = 1.0 / (b + kSnakeEps);
    for (int64_t t = 0; t < length; ++t) {
      const double v = x[static_cast<size_t>(c * length + t)];
      const double s = std::sin(a * v);
      x[static_cast<size_t>(c * length + t)] = static_cast<float>(v + inv_beta * s * s);
    }
  }
}

// kaiser_sinc_filter1d (dac_alias_free_filter.py:26-60). Returns [kernel_size].
std::vector<float> KaiserSincFilter1d(double cutoff, double half_width,
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

void AliasFreeActivation1d::Build() {
  // Up and down use the same cutoff/half_width/kernel, so one window serves both.
  filter = KaiserSincFilter1d(0.5 / static_cast<double>(ratio),
                                       0.6 / static_cast<double>(ratio), kernel_size);
}

std::vector<float> AliasFreeActivation1d::Apply(const std::vector<float>& in,
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
      Pad1d(in, channels, in_len, pad, pad, /*replicate=*/true, &padded_len);
  // Depthwise transposed conv: filter.expand(C, -1, -1) => weight [C, 1, K].
  std::vector<float> depthwise(static_cast<size_t>(channels * kernel_size));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t k = 0; k < kernel_size; ++k) {
      depthwise[static_cast<size_t>(c * kernel_size + k)] = filter[static_cast<size_t>(k)];
    }
  }
  int64_t up_len = 0;
  std::vector<float> up =
      ConvTranspose1d(padded, channels, padded_len, depthwise, nullptr, channels,
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
  SnakeActivation(trimmed, channels, trimmed_len, alpha, beta, logscale);

  // --- DownSample1d (LowPassFilter1d, stride = ratio, replicate padding) ---
  const bool even = (kernel_size % 2) == 0;
  const int64_t lp_left = kernel_size / 2 - (even ? 1 : 0);
  const int64_t lp_right = kernel_size / 2;
  int64_t lp_padded_len = 0;
  const std::vector<float> lp_padded = Pad1d(trimmed, channels, trimmed_len, lp_left,
                                                      lp_right, /*replicate=*/true, &lp_padded_len);
  return Conv1d(lp_padded, channels, lp_padded_len, depthwise, nullptr, channels,
                         kernel_size, /*stride=*/ratio, /*dilation=*/1, /*groups=*/channels,
                         out_len);
}

// torch weight_norm: w = g * v / ||v||, norm over every dim except dim 0.
// Moved here from `minimax_h3_audio_vae.cpp` when MiniMax-Music3's vocoder
// became its second consumer; see the declaration for why the axis is named
// `dim0` and not `out_channels`.
std::vector<float> MaterializeWeightNorm(const std::vector<float>& g,
                                         const std::vector<float>& v, int64_t dim0) {
  VT_CHECK(dim0 > 0 && v.size() % static_cast<size_t>(dim0) == 0,
           "vocoder1d: weight-norm direction does not divide by dim 0");
  const int64_t per_slice = static_cast<int64_t>(v.size()) / dim0;
  VT_CHECK(static_cast<int64_t>(g.size()) == dim0,
           "vocoder1d: weight-norm magnitude must have one value per dim-0 slice");
  std::vector<float> out(v.size());
  for (int64_t c = 0; c < dim0; ++c) {
    double norm = 0.0;
    for (int64_t i = 0; i < per_slice; ++i) {
      const double value = v[static_cast<size_t>(c * per_slice + i)];
      norm += value * value;
    }
    norm = std::sqrt(norm);
    const double scale = norm > 0.0 ? static_cast<double>(g[static_cast<size_t>(c)]) / norm : 0.0;
    for (int64_t i = 0; i < per_slice; ++i) {
      out[static_cast<size_t>(c * per_slice + i)] =
          static_cast<float>(v[static_cast<size_t>(c * per_slice + i)] * scale);
    }
  }
  return out;
}

}  // namespace vocoder1d
}  // namespace vllm
