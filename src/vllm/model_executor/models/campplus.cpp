// CAMPPlus primitives. See campplus.h for the upstream anchors.
#include "vllm/model_executor/models/campplus.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace campplus {

std::vector<float> StatsPool(const std::vector<float>& x, int64_t channels, int64_t frames) {
  VT_CHECK(x.size() == static_cast<size_t>(channels * frames), "campplus: StatsPool shape");
  VT_CHECK(frames > 1, "campplus: unbiased std needs at least two frames");
  std::vector<float> out(static_cast<size_t>(2 * channels));
  for (int64_t c = 0; c < channels; ++c) {
    const float* row = x.data() + static_cast<size_t>(c * frames);
    double mean = 0.0;
    for (int64_t t = 0; t < frames; ++t) mean += static_cast<double>(row[t]);
    mean /= static_cast<double>(frames);
    double sq = 0.0;
    for (int64_t t = 0; t < frames; ++t) {
      const double d = static_cast<double>(row[t]) - mean;
      sq += d * d;
    }
    // UNBIASED: torch's std(unbiased=True) divides by N-1.
    const double std_dev = std::sqrt(sq / static_cast<double>(frames - 1));
    out[static_cast<size_t>(c)] = static_cast<float>(mean);
    out[static_cast<size_t>(channels + c)] = static_cast<float>(std_dev);
  }
  return out;
}

std::vector<float> BatchNorm1dEval(const std::vector<float>& x, int64_t channels, int64_t frames,
                                   const std::vector<float>& gamma, const std::vector<float>& beta,
                                   const std::vector<float>& running_mean,
                                   const std::vector<float>& running_var, double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(channels * frames), "campplus: BatchNorm shape");
  std::vector<float> out(x.size());
  for (int64_t c = 0; c < channels; ++c) {
    const double m = static_cast<double>(running_mean[static_cast<size_t>(c)]);
    const double inv = 1.0 / std::sqrt(static_cast<double>(running_var[static_cast<size_t>(c)]) + eps);
    const double g = gamma.empty() ? 1.0 : static_cast<double>(gamma[static_cast<size_t>(c)]);
    const double b = beta.empty() ? 0.0 : static_cast<double>(beta[static_cast<size_t>(c)]);
    for (int64_t t = 0; t < frames; ++t) {
      const size_t i = static_cast<size_t>(c * frames + t);
      out[i] = static_cast<float>((static_cast<double>(x[i]) - m) * inv * g + b);
    }
  }
  return out;
}

std::vector<float> SegPooling(const std::vector<float>& x, int64_t channels, int64_t frames,
                              int64_t seg_len) {
  VT_CHECK(x.size() == static_cast<size_t>(channels * frames), "campplus: SegPooling shape");
  VT_CHECK(seg_len > 0, "campplus: seg_len must be positive");
  // ceil_mode=true: a partial trailing window still produces a segment, averaged
  // over the frames it actually covers.
  const int64_t segments = (frames + seg_len - 1) / seg_len;
  std::vector<float> out(x.size());
  for (int64_t c = 0; c < channels; ++c) {
    const float* row = x.data() + static_cast<size_t>(c * frames);
    float* dst = out.data() + static_cast<size_t>(c * frames);
    for (int64_t s = 0; s < segments; ++s) {
      const int64_t begin = s * seg_len;
      const int64_t end = std::min(begin + seg_len, frames);
      double acc = 0.0;
      for (int64_t t = begin; t < end; ++t) acc += static_cast<double>(row[t]);
      const float mean = static_cast<float>(acc / static_cast<double>(end - begin));
      // expand back over seg_len frames, then TRUNCATE to the input length.
      for (int64_t t = begin; t < end; ++t) dst[t] = mean;
    }
  }
  return out;
}

// 1-D convolution over [C, T] with `same` padding, as every CAMPPlus conv uses.
static std::vector<float> Conv1dSame(const std::vector<float>& in, int64_t in_ch, int64_t frames,
                              const std::vector<float>& w, const std::vector<float>& bias,
                              int64_t out_ch, int64_t kernel, int64_t dilation) {
  const int64_t pad = (kernel - 1) / 2 * dilation;
  std::vector<float> out(static_cast<size_t>(out_ch * frames), 0.0F);
  for (int64_t o = 0; o < out_ch; ++o) {
    for (int64_t t = 0; t < frames; ++t) {
      double acc = bias.empty() ? 0.0 : static_cast<double>(bias[static_cast<size_t>(o)]);
      for (int64_t i = 0; i < in_ch; ++i) {
        for (int64_t k = 0; k < kernel; ++k) {
          const int64_t src = t + k * dilation - pad;
          if (src < 0 || src >= frames) continue;  // zero padding
          acc += static_cast<double>(w[static_cast<size_t>((o * in_ch + i) * kernel + k)]) *
                 static_cast<double>(in[static_cast<size_t>(i * frames + src)]);
        }
      }
      out[static_cast<size_t>(o * frames + t)] = static_cast<float>(acc);
    }
  }
  return out;
}

std::vector<float> CamLayer(const std::vector<float>& x, int64_t bn_channels, int64_t frames,
                            int64_t out_channels, int64_t kernel, int64_t dilation,
                            int64_t seg_len, const CamLayerWeights& weights) {
  const std::vector<float> y =
      Conv1dSame(x, bn_channels, frames, weights.linear_local, {}, out_channels, kernel, dilation);

  // context = x.mean(-1, keepdim=True) + seg_pooling(x): a [C, T] signal, since
  // the per-channel mean broadcasts across the segment-pooled one.
  const std::vector<float> seg = SegPooling(x, bn_channels, frames, seg_len);
  std::vector<float> context(x.size());
  for (int64_t c = 0; c < bn_channels; ++c) {
    double mean = 0.0;
    for (int64_t t = 0; t < frames; ++t) {
      mean += static_cast<double>(x[static_cast<size_t>(c * frames + t)]);
    }
    mean /= static_cast<double>(frames);
    for (int64_t t = 0; t < frames; ++t) {
      const size_t i = static_cast<size_t>(c * frames + t);
      context[i] = static_cast<float>(mean + static_cast<double>(seg[i]));
    }
  }

  const int64_t reduced = bn_channels / 2;
  std::vector<float> h = Conv1dSame(context, bn_channels, frames, weights.linear1_weight,
                                    weights.linear1_bias, reduced, 1, 1);
  for (float& v : h) v = v > 0.0F ? v : 0.0F;  // ReLU
  std::vector<float> m = Conv1dSame(h, reduced, frames, weights.linear2_weight,
                                    weights.linear2_bias, out_channels, 1, 1);
  for (float& v : m) v = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(v))));

  std::vector<float> out(y.size());
  for (size_t i = 0; i < y.size(); ++i) out[i] = y[i] * m[i];
  return out;
}


std::vector<float> BatchNormRelu(const std::vector<float>& x, int64_t channels, int64_t frames,
                                 const std::vector<float>& gamma, const std::vector<float>& beta,
                                 const std::vector<float>& running_mean,
                                 const std::vector<float>& running_var, double eps) {
  std::vector<float> out =
      BatchNorm1dEval(x, channels, frames, gamma, beta, running_mean, running_var, eps);
  for (float& v : out) v = v > 0.0F ? v : 0.0F;
  return out;
}

std::vector<float> TransitLayer(const std::vector<float>& x, int64_t in_channels, int64_t frames,
                                int64_t out_channels, const std::vector<float>& bn_gamma,
                                const std::vector<float>& bn_beta,
                                const std::vector<float>& bn_mean,
                                const std::vector<float>& bn_var,
                                const std::vector<float>& weight, const std::vector<float>& bias,
                                double eps) {
  // nonlinear FIRST, then the 1x1 projection.
  const std::vector<float> h =
      BatchNormRelu(x, in_channels, frames, bn_gamma, bn_beta, bn_mean, bn_var, eps);
  return Conv1dSame(h, in_channels, frames, weight, bias, out_channels, 1, 1);
}

std::vector<float> DenseLayer(const std::vector<float>& x, int64_t in_channels, int64_t frames,
                              int64_t out_channels, const std::vector<float>& weight,
                              const std::vector<float>& bias, const std::vector<float>& bn_gamma,
                              const std::vector<float>& bn_beta, const std::vector<float>& bn_mean,
                              const std::vector<float>& bn_var, double eps, bool apply_relu) {
  // 1x1 projection FIRST, then the nonlinear -- the opposite order to
  // TransitLayer. A pooled stats vector arrives as frames == 1.
  const std::vector<float> h =
      Conv1dSame(x, in_channels, frames, weight, bias, out_channels, 1, 1);
  std::vector<float> out =
      BatchNorm1dEval(h, out_channels, frames, bn_gamma, bn_beta, bn_mean, bn_var, eps);
  // `batchnorm_` is batchnorm ALONE (affine=false, no relu).
  if (apply_relu) {
    for (float& v : out) v = v > 0.0F ? v : 0.0F;
  }
  return out;
}

std::vector<float> DenseTdnnLayer(const std::vector<float>& x, int64_t in_channels, int64_t frames,
                                  int64_t bn_channels, int64_t out_channels, int64_t kernel,
                                  int64_t dilation, int64_t seg_len,
                                  const DenseTdnnLayerWeights& w, double eps) {
  const std::vector<float> a = BatchNormRelu(x, in_channels, frames, w.bn1_gamma, w.bn1_beta,
                                             w.bn1_mean, w.bn1_var, eps);
  const std::vector<float> b =
      Conv1dSame(a, in_channels, frames, w.linear1, {}, bn_channels, 1, 1);
  const std::vector<float> c = BatchNormRelu(b, bn_channels, frames, w.bn2_gamma, w.bn2_beta,
                                             w.bn2_mean, w.bn2_var, eps);
  return CamLayer(c, bn_channels, frames, out_channels, kernel, dilation, seg_len, w.cam);
}

std::vector<float> DenseTdnnBlock(const std::vector<float>& x, int64_t in_channels, int64_t frames,
                                  int64_t bn_channels, int64_t growth, int64_t kernel,
                                  int64_t dilation, int64_t seg_len,
                                  const std::vector<DenseTdnnLayerWeights>& layers, double eps) {
  std::vector<float> acc = x;
  int64_t channels = in_channels;
  for (const DenseTdnnLayerWeights& w : layers) {
    const std::vector<float> y = DenseTdnnLayer(acc, channels, frames, bn_channels, growth, kernel,
                                                dilation, seg_len, w, eps);
    // cat([x, layer(x)], dim=1): the new channels are APPENDED, so every later
    // layer sees all earlier outputs.
    acc.insert(acc.end(), y.begin(), y.end());
    channels += growth;
  }
  return acc;
}

}  // namespace campplus
}  // namespace models
}  // namespace vllm
