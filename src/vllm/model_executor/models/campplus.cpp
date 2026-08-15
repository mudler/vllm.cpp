// CAMPPlus primitives. See campplus.h for the upstream anchors.
#include "vllm/model_executor/models/campplus.h"

#include <stdexcept>

#include "vllm/model_executor/model_loader/safetensors_reader.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
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


namespace {

// 2-D convolution over [C, H, W], zero-padded, stride (sh, sw).
std::vector<float> Conv2d(const std::vector<float>& in, int64_t in_ch, int64_t h, int64_t w,
                          const std::vector<float>& weight, int64_t out_ch, int64_t kernel,
                          int64_t sh, int64_t sw, int64_t pad, int64_t* oh, int64_t* ow) {
  const int64_t H = (h + 2 * pad - kernel) / sh + 1;
  const int64_t W = (w + 2 * pad - kernel) / sw + 1;
  std::vector<float> out(static_cast<size_t>(out_ch * H * W), 0.0F);
  for (int64_t o = 0; o < out_ch; ++o) {
    for (int64_t y = 0; y < H; ++y) {
      for (int64_t x = 0; x < W; ++x) {
        double acc = 0.0;
        for (int64_t c = 0; c < in_ch; ++c) {
          for (int64_t ky = 0; ky < kernel; ++ky) {
            const int64_t sy = y * sh + ky - pad;
            if (sy < 0 || sy >= h) continue;
            for (int64_t kx = 0; kx < kernel; ++kx) {
              const int64_t sx = x * sw + kx - pad;
              if (sx < 0 || sx >= w) continue;
              acc += static_cast<double>(
                         weight[static_cast<size_t>(((o * in_ch + c) * kernel + ky) * kernel + kx)]) *
                     static_cast<double>(in[static_cast<size_t>((c * h + sy) * w + sx)]);
            }
          }
        }
        out[static_cast<size_t>((o * H + y) * W + x)] = static_cast<float>(acc);
      }
    }
  }
  *oh = H; *ow = W;
  return out;
}

// BatchNorm2d in eval: per-CHANNEL running statistics over the H*W plane.
void BatchNorm2dEvalInPlace(std::vector<float>& x, int64_t channels, int64_t plane,
                            const std::vector<float>& g, const std::vector<float>& b,
                            const std::vector<float>& mean, const std::vector<float>& var,
                            double eps) {
  for (int64_t c = 0; c < channels; ++c) {
    const double m = static_cast<double>(mean[static_cast<size_t>(c)]);
    const double inv = 1.0 / std::sqrt(static_cast<double>(var[static_cast<size_t>(c)]) + eps);
    const double gc = g.empty() ? 1.0 : static_cast<double>(g[static_cast<size_t>(c)]);
    const double bc = b.empty() ? 0.0 : static_cast<double>(b[static_cast<size_t>(c)]);
    for (int64_t i = 0; i < plane; ++i) {
      const size_t k = static_cast<size_t>(c * plane + i);
      x[k] = static_cast<float>((static_cast<double>(x[k]) - m) * inv * gc + bc);
    }
  }
}

}  // namespace

std::vector<float> ResBlock2d(const std::vector<float>& x, int64_t in_planes, int64_t h, int64_t w,
                              int64_t planes, int64_t stride, const ResBlock2dWeights& wt,
                              double eps, int64_t* out_h) {
  int64_t h1 = 0, w1 = 0;
  // stride is (stride, 1): FREQUENCY only.
  std::vector<float> out =
      Conv2d(x, in_planes, h, w, wt.conv1, planes, 3, stride, 1, 1, &h1, &w1);
  BatchNorm2dEvalInPlace(out, planes, h1 * w1, wt.bn1_gamma, wt.bn1_beta, wt.bn1_mean, wt.bn1_var, eps);
  for (float& v : out) v = v > 0.0F ? v : 0.0F;

  int64_t h2 = 0, w2 = 0;
  out = Conv2d(out, planes, h1, w1, wt.conv2, planes, 3, 1, 1, 1, &h2, &w2);
  BatchNorm2dEvalInPlace(out, planes, h2 * w2, wt.bn2_gamma, wt.bn2_beta, wt.bn2_mean, wt.bn2_var, eps);

  std::vector<float> shortcut;
  if (wt.has_shortcut) {
    int64_t sh = 0, sw = 0;
    shortcut = Conv2d(x, in_planes, h, w, wt.short_conv, planes, 1, stride, 1, 0, &sh, &sw);
    BatchNorm2dEvalInPlace(shortcut, planes, sh * sw, wt.short_gamma, wt.short_beta, wt.short_mean,
                           wt.short_var, eps);
  } else {
    shortcut = x;
  }
  VT_CHECK(shortcut.size() == out.size(), "campplus: residual shape mismatch");
  for (size_t i = 0; i < out.size(); ++i) out[i] += shortcut[i];
  for (float& v : out) v = v > 0.0F ? v : 0.0F;
  *out_h = h2;
  return out;
}


const std::vector<float>& CampplusWeights::Get(const std::string& name) const {
  const auto it = t.find(name);
  VT_CHECK(it != t.end(), "campplus: missing checkpoint tensor '" + name + "'");
  return it->second;
}

namespace {

// Conv1d with an EXPLICIT stride and padding (the TDNN head strides TIME by 2,
// unlike every `same` convolution elsewhere in this file).
std::vector<float> Conv1dStrided(const std::vector<float>& in, int64_t in_ch, int64_t len,
                                 const std::vector<float>& w, const std::vector<float>& bias,
                                 int64_t out_ch, int64_t kernel, int64_t stride, int64_t dilation,
                                 int64_t pad, int64_t* out_len) {
  const int64_t L = (len + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
  std::vector<float> out(static_cast<size_t>(out_ch * L));
  for (int64_t o = 0; o < out_ch; ++o) {
    for (int64_t t = 0; t < L; ++t) {
      double acc = bias.empty() ? 0.0 : static_cast<double>(bias[static_cast<size_t>(o)]);
      for (int64_t c = 0; c < in_ch; ++c) {
        for (int64_t k = 0; k < kernel; ++k) {
          const int64_t src = t * stride + k * dilation - pad;
          if (src < 0 || src >= len) continue;
          acc += static_cast<double>(w[static_cast<size_t>((o * in_ch + c) * kernel + k)]) *
                 static_cast<double>(in[static_cast<size_t>(c * len + src)]);
        }
      }
      out[static_cast<size_t>(o * L + t)] = static_cast<float>(acc);
    }
  }
  *out_len = L;
  return out;
}

ResBlock2dWeights ResW(const CampplusWeights& w, const std::string& p, bool shortcut) {
  ResBlock2dWeights r;
  r.conv1 = w.Get(p + ".conv1.weight");
  r.bn1_gamma = w.Get(p + ".bn1.weight"); r.bn1_beta = w.Get(p + ".bn1.bias");
  r.bn1_mean = w.Get(p + ".bn1.running_mean"); r.bn1_var = w.Get(p + ".bn1.running_var");
  r.conv2 = w.Get(p + ".conv2.weight");
  r.bn2_gamma = w.Get(p + ".bn2.weight"); r.bn2_beta = w.Get(p + ".bn2.bias");
  r.bn2_mean = w.Get(p + ".bn2.running_mean"); r.bn2_var = w.Get(p + ".bn2.running_var");
  r.has_shortcut = shortcut;
  if (shortcut) {
    r.short_conv = w.Get(p + ".shortcut.0.weight");
    r.short_gamma = w.Get(p + ".shortcut.1.weight"); r.short_beta = w.Get(p + ".shortcut.1.bias");
    r.short_mean = w.Get(p + ".shortcut.1.running_mean");
    r.short_var = w.Get(p + ".shortcut.1.running_var");
  }
  return r;
}

DenseTdnnLayerWeights DlW(const CampplusWeights& w, const std::string& p) {
  DenseTdnnLayerWeights d;
  d.bn1_gamma = w.Get(p + ".nonlinear1.batchnorm.weight");
  d.bn1_beta = w.Get(p + ".nonlinear1.batchnorm.bias");
  d.bn1_mean = w.Get(p + ".nonlinear1.batchnorm.running_mean");
  d.bn1_var = w.Get(p + ".nonlinear1.batchnorm.running_var");
  d.linear1 = w.Get(p + ".linear1.weight");
  d.bn2_gamma = w.Get(p + ".nonlinear2.batchnorm.weight");
  d.bn2_beta = w.Get(p + ".nonlinear2.batchnorm.bias");
  d.bn2_mean = w.Get(p + ".nonlinear2.batchnorm.running_mean");
  d.bn2_var = w.Get(p + ".nonlinear2.batchnorm.running_var");
  d.cam.linear_local = w.Get(p + ".cam_layer.linear_local.weight");
  d.cam.linear1_weight = w.Get(p + ".cam_layer.linear1.weight");
  d.cam.linear1_bias = w.Get(p + ".cam_layer.linear1.bias");
  d.cam.linear2_weight = w.Get(p + ".cam_layer.linear2.weight");
  d.cam.linear2_bias = w.Get(p + ".cam_layer.linear2.bias");
  return d;
}

}  // namespace

std::vector<float> Forward(const CampplusParams& p, const CampplusWeights& w,
                           const std::vector<float>& feats, int64_t frames, ForwardTrace* trace) {
  VT_CHECK(feats.size() == static_cast<size_t>(frames * p.feat_dim), "campplus: feats shape");

  // forward() permutes (T, F) -> (F, T), then FCM treats it as a 1-channel image
  // of height feat_dim and width T.
  std::vector<float> img(feats.size());
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t f = 0; f < p.feat_dim; ++f) {
      img[static_cast<size_t>(f * frames + t)] = feats[static_cast<size_t>(t * p.feat_dim + f)];
    }
  }

  int64_t h = p.feat_dim, wid = frames, oh = 0, ow = 0;
  std::vector<float> x = Conv2d(img, 1, h, wid, w.Get("head.conv1.weight"), p.m_channels, 3, 1, 1, 1,
                                &oh, &ow);
  BatchNorm2dEvalInPlace(x, p.m_channels, oh * ow, w.Get("head.bn1.weight"), w.Get("head.bn1.bias"),
                         w.Get("head.bn1.running_mean"), w.Get("head.bn1.running_var"), p.eps);
  for (float& v : x) v = v > 0.0F ? v : 0.0F;
  h = oh; wid = ow;

  // layer1 / layer2: two BasicResBlocks each, the FIRST striding frequency by 2.
  for (int layer = 1; layer <= 2; ++layer) {
    for (int b = 0; b < 2; ++b) {
      const std::string pre = "head.layer" + std::to_string(layer) + "." + std::to_string(b);
      const int64_t stride = (b == 0) ? 2 : 1;
      const bool shortcut = w.Has(pre + ".shortcut.0.weight");
      int64_t nh = 0;
      x = ResBlock2d(x, p.m_channels, h, wid, p.m_channels, stride, ResW(w, pre, shortcut), p.eps,
                     &nh);
      h = nh;
    }
  }

  x = Conv2d(x, p.m_channels, h, wid, w.Get("head.conv2.weight"), p.m_channels, 3, 2, 1, 1, &oh, &ow);
  BatchNorm2dEvalInPlace(x, p.m_channels, oh * ow, w.Get("head.bn2.weight"), w.Get("head.bn2.bias"),
                         w.Get("head.bn2.running_mean"), w.Get("head.bn2.running_var"), p.eps);
  for (float& v : x) v = v > 0.0F ? v : 0.0F;
  h = oh; wid = ow;

  // reshape (C, H, W) -> (C*H, W): the channel-major layout already matches.
  int64_t channels = p.m_channels * h;
  int64_t len = wid;

  // xvector.tdnn: Conv1d(k=5, stride=2, padding=(5-1)//2*1) then batchnorm-relu.
  int64_t nl = 0;
  x = Conv1dStrided(x, channels, len, w.Get("xvector.tdnn.linear.weight"), {}, p.init_channels, 5, 2,
                    1, 2, &nl);
  len = nl;
  channels = p.init_channels;
  x = BatchNormRelu(x, channels, len, w.Get("xvector.tdnn.nonlinear.batchnorm.weight"),
                    w.Get("xvector.tdnn.nonlinear.batchnorm.bias"),
                    w.Get("xvector.tdnn.nonlinear.batchnorm.running_mean"),
                    w.Get("xvector.tdnn.nonlinear.batchnorm.running_var"), p.eps);

  if (trace != nullptr) {
    trace->tdnn = x;
    trace->tdnn_channels = channels;
    trace->tdnn_frames = len;
  }

  const int64_t bn_channels = p.bn_size * p.growth_rate;
  const int64_t counts[3] = {12, 24, 16};
  const int64_t dilations[3] = {1, 2, 2};
  for (int i = 0; i < 3; ++i) {
    std::vector<DenseTdnnLayerWeights> layers;
    for (int64_t j = 0; j < counts[i]; ++j) {
      layers.push_back(DlW(w, "xvector.block" + std::to_string(i + 1) + ".tdnnd" +
                                  std::to_string(j + 1)));
    }
    x = DenseTdnnBlock(x, channels, len, bn_channels, p.growth_rate, 3, dilations[i], p.seg_len,
                       layers, p.eps);
    channels += counts[i] * p.growth_rate;

    const std::string tp = "xvector.transit" + std::to_string(i + 1);
    x = TransitLayer(x, channels, len, channels / 2, w.Get(tp + ".nonlinear.batchnorm.weight"),
                     w.Get(tp + ".nonlinear.batchnorm.bias"),
                     w.Get(tp + ".nonlinear.batchnorm.running_mean"),
                     w.Get(tp + ".nonlinear.batchnorm.running_var"), w.Get(tp + ".linear.weight"),
                     {}, p.eps);
    channels /= 2;
  }

  x = BatchNormRelu(x, channels, len, w.Get("xvector.out_nonlinear.batchnorm.weight"),
                    w.Get("xvector.out_nonlinear.batchnorm.bias"),
                    w.Get("xvector.out_nonlinear.batchnorm.running_mean"),
                    w.Get("xvector.out_nonlinear.batchnorm.running_var"), p.eps);

  const std::vector<float> stats = StatsPool(x, channels, len);
  // The final dense is `batchnorm_`: affine=false AND no relu.
  return DenseLayer(stats, 2 * channels, 1, p.embedding_size, w.Get("xvector.dense.linear.weight"),
                    {}, {}, {}, w.Get("xvector.dense.nonlinear.batchnorm.running_mean"),
                    w.Get("xvector.dense.nonlinear.batchnorm.running_var"), p.eps,
                    /*apply_relu=*/false);
}

CampplusWeights LoadCampplus(const std::string& path) {
  const SafetensorsFile file = SafetensorsFile::Open(path);
  CampplusWeights w;
  for (const std::string& name : file.Names()) {
    // `num_batches_tracked` is an I64 TRAINING COUNTER -- BatchNorm's momentum
    // bookkeeping -- and inference never reads it. It is skipped BY NAME rather
    // than by "ignore anything that is not F32", because that broader rule
    // would silently drop a real weight the day one ships in another dtype.
    if (name.size() >= 19 &&
        name.compare(name.size() - 19, 19, "num_batches_tracked") == 0) {
      continue;
    }
    const StTensor& tensor = file.Get(name);
    if (tensor.dtype != "F32") {
      throw std::runtime_error("campplus: tensor '" + name + "' is " + tensor.dtype +
                               ", expected F32");
    }
    std::vector<float> values(tensor.nbytes / sizeof(float));
    if (!values.empty()) {
      std::memcpy(values.data(), tensor.data, tensor.nbytes);
    }
    w.t.emplace(name, std::move(values));
  }
  if (w.t.empty()) {
    throw std::runtime_error("campplus: '" + path + "' held no tensors");
  }
  return w;
}

}  // namespace campplus
}  // namespace models
}  // namespace vllm
