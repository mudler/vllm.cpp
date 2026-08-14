// S2Mel length regulator primitives. See lenreg.h.
#include "vllm/model_executor/models/lenreg.h"

#include <cmath>
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

}  // namespace lenreg
}  // namespace models
}  // namespace vllm
