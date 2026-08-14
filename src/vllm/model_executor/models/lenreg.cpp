// S2Mel length regulator primitives. See lenreg.h.
#include "vllm/model_executor/models/lenreg.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace lenreg {

std::vector<float> InterpolateNearest(const std::vector<float>& x, int64_t channels,
                                      int64_t in_frames, int64_t out_frames) {
  VT_CHECK(x.size() == static_cast<size_t>(channels * in_frames), "lenreg: interpolate shape");
  VT_CHECK(in_frames > 0 && out_frames > 0, "lenreg: frame counts must be positive");
  std::vector<float> out(static_cast<size_t>(channels * out_frames));
  const double ratio = static_cast<double>(in_frames) / static_cast<double>(out_frames);
  for (int64_t t = 0; t < out_frames; ++t) {
    // torch: src = floor(i * in / out), NOT a rounded or half-offset index.
    int64_t src = static_cast<int64_t>(std::floor(static_cast<double>(t) * ratio));
    if (src >= in_frames) src = in_frames - 1;
    for (int64_t c = 0; c < channels; ++c) {
      out[static_cast<size_t>(c * out_frames + t)] =
          x[static_cast<size_t>(c * in_frames + src)];
    }
  }
  return out;
}

std::vector<float> GroupNorm(const std::vector<float>& x, int64_t channels, int64_t frames,
                             int64_t groups, const std::vector<float>& gamma,
                             const std::vector<float>& beta, double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(channels * frames), "lenreg: GroupNorm shape");
  VT_CHECK(groups > 0 && channels % groups == 0, "lenreg: channels must divide by groups");
  const int64_t per_group = channels / groups;
  std::vector<float> out(x.size());
  for (int64_t g = 0; g < groups; ++g) {
    // Statistics span the whole GROUP: every channel in it and every frame.
    double mean = 0.0;
    const int64_t count = per_group * frames;
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      for (int64_t t = 0; t < frames; ++t) {
        mean += static_cast<double>(x[static_cast<size_t>(c * frames + t)]);
      }
    }
    mean /= static_cast<double>(count);
    double var = 0.0;
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      for (int64_t t = 0; t < frames; ++t) {
        const double d = static_cast<double>(x[static_cast<size_t>(c * frames + t)]) - mean;
        var += d * d;
      }
    }
    var /= static_cast<double>(count);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      for (int64_t t = 0; t < frames; ++t) {
        const size_t i = static_cast<size_t>(c * frames + t);
        out[i] = static_cast<float>((static_cast<double>(x[i]) - mean) * inv *
                                        static_cast<double>(gamma[static_cast<size_t>(c)]) +
                                    static_cast<double>(beta[static_cast<size_t>(c)]));
      }
    }
  }
  return out;
}

double Mish(double x) {
  // softplus is log1p(exp(x)), guarded the way torch does for large x so the
  // exponential cannot overflow.
  const double sp = (x > 20.0) ? x : std::log1p(std::exp(x));
  return x * std::tanh(sp);
}

namespace {

constexpr const char* kLrPrefix = "net.length_regulator.";

std::vector<float> LrRead(const SafetensorsFile& file, const std::string& suffix) {
  const std::string name = kLrPrefix + suffix;
  const StTensor* t = nullptr;
  try {
    t = &file.Get(name);
  } catch (const std::exception&) {
    throw std::runtime_error("IndexTTS-2.5 length_regulator: missing tensor '" + name + "'");
  }
  if (t->dtype != "F32") {
    throw std::runtime_error("IndexTTS-2.5 length_regulator: '" + name + "' is " + t->dtype);
  }
  std::vector<float> out(t->nbytes / sizeof(float));
  std::memcpy(out.data(), t->data, t->nbytes);
  return out;
}

std::vector<int64_t> LrShape(const SafetensorsFile& file, const std::string& suffix) {
  const std::string name = kLrPrefix + suffix;
  try {
    return file.Get(name).shape;
  } catch (const std::exception&) {
    throw std::runtime_error("IndexTTS-2.5 length_regulator: missing tensor '" + name + "'");
  }
}

// Conv1d over [channels, frames] with zero padding that preserves the length.
std::vector<float> LrConv(const std::vector<float>& x, int64_t channels, int64_t frames,
                          const std::vector<float>& w, const std::vector<float>& b,
                          int64_t kernel) {
  const int64_t pad = (kernel - 1) / 2;
  std::vector<float> out(static_cast<size_t>(channels * frames));
  for (int64_t o = 0; o < channels; ++o) {
    const double bias = b.empty() ? 0.0 : static_cast<double>(b[static_cast<size_t>(o)]);
    for (int64_t t = 0; t < frames; ++t) {
      double acc = bias;
      for (int64_t c = 0; c < channels; ++c) {
        for (int64_t k = 0; k < kernel; ++k) {
          const int64_t src = t + k - pad;
          if (src < 0 || src >= frames) {
            continue;
          }
          acc += static_cast<double>(x[static_cast<size_t>(c * frames + src)]) *
                 static_cast<double>(w[static_cast<size_t>((o * channels + c) * kernel + k)]);
        }
      }
      out[static_cast<size_t>(o * frames + t)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

std::vector<float> RegulateHost(const RegulatorConfig& cfg, const RegulatorWeights& w,
                                const std::vector<float>& x, int64_t frames,
                                int64_t out_frames) {
  VT_CHECK(cfg.channels > 0 && cfg.in_channels > 0 && frames > 0 && out_frames > 0,
           "lenreg: channels, in_channels, frames and out_frames must be positive");
  VT_CHECK(x.size() == static_cast<size_t>(frames * cfg.in_channels),
           "lenreg: x must be [frames, in_channels]");
  VT_CHECK(w.conv_w.size() == w.norm_w.size() && !w.conv_w.empty(),
           "lenreg: one GroupNorm per convolution is required");

  // content_in_proj, frame-major in and out.
  std::vector<float> proj(static_cast<size_t>(frames * cfg.channels));
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t o = 0; o < cfg.channels; ++o) {
      double acc = w.in_proj_b.empty()
                       ? 0.0
                       : static_cast<double>(w.in_proj_b[static_cast<size_t>(o)]);
      for (int64_t i = 0; i < cfg.in_channels; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(f * cfg.in_channels + i)]) *
               static_cast<double>(w.in_proj_w[static_cast<size_t>(o * cfg.in_channels + i)]);
      }
      proj[static_cast<size_t>(f * cfg.channels + o)] = static_cast<float>(acc);
    }
  }
  // Transpose to channel-major, which is what interpolate and the convs want.
  std::vector<float> cm(static_cast<size_t>(cfg.channels * frames));
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t c = 0; c < cfg.channels; ++c) {
      cm[static_cast<size_t>(c * frames + f)] = proj[static_cast<size_t>(f * cfg.channels + c)];
    }
  }

  std::vector<float> cur = InterpolateNearest(cm, cfg.channels, frames, out_frames);

  for (size_t i = 0; i < w.conv_w.size(); ++i) {
    cur = LrConv(cur, cfg.channels, out_frames, w.conv_w[i], w.conv_b[i], 3);
    cur = GroupNorm(cur, cfg.channels, out_frames, cfg.groups, w.norm_w[i], w.norm_b[i],
                    cfg.eps);
    for (float& v : cur) {
      v = static_cast<float>(Mish(static_cast<double>(v)));
    }
  }
  cur = LrConv(cur, cfg.channels, out_frames, w.out_conv_w, w.out_conv_b, 1);

  // Back to frame-major, as upstream's trailing transpose does.
  std::vector<float> out(static_cast<size_t>(out_frames * cfg.channels));
  for (int64_t f = 0; f < out_frames; ++f) {
    for (int64_t c = 0; c < cfg.channels; ++c) {
      out[static_cast<size_t>(f * cfg.channels + c)] =
          cur[static_cast<size_t>(c * out_frames + f)];
    }
  }
  return out;
}

RegulatorWeights LoadRegulator(const SafetensorsFile& file, RegulatorConfig* cfg) {
  VT_CHECK(cfg != nullptr, "lenreg: a config out-parameter is required");
  RegulatorWeights w;
  const std::vector<int64_t> ip = LrShape(file, "content_in_proj.weight");
  VT_CHECK(ip.size() == 2, "lenreg: content_in_proj.weight must be 2-D");
  cfg->channels = ip[0];
  cfg->in_channels = ip[1];
  cfg->groups = 1;
  cfg->eps = 1e-5;
  w.in_proj_w = LrRead(file, "content_in_proj.weight");
  w.in_proj_b = LrRead(file, "content_in_proj.bias");

  // The Sequential's occupied indices: (conv, norm) at 0/1, 3/4, 6/7, 9/10, and
  // the 1x1 convolution at 12. The gaps are the activations, which carry no
  // parameters -- reading them as layers is the obvious misstep.
  const int64_t conv_idx[] = {0, 3, 6, 9};
  for (const int64_t i : conv_idx) {
    const std::string ci = "model." + std::to_string(i) + ".";
    const std::string ni = "model." + std::to_string(i + 1) + ".";
    w.conv_w.push_back(LrRead(file, ci + "weight"));
    w.conv_b.push_back(LrRead(file, ci + "bias"));
    w.norm_w.push_back(LrRead(file, ni + "weight"));
    w.norm_b.push_back(LrRead(file, ni + "bias"));
  }
  w.out_conv_w = LrRead(file, "model.12.weight");
  w.out_conv_b = LrRead(file, "model.12.bias");
  return w;
}

}  // namespace lenreg
}  // namespace models
}  // namespace vllm
