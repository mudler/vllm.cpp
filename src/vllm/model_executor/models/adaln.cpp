// Adaptive layer norm. See adaln.h for the upstream anchors.
#include "vllm/model_executor/models/adaln.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace adaln {

std::vector<float> Modulate(const std::vector<float>& x, int64_t frames, int64_t hidden,
                            const std::vector<float>& shift, const std::vector<float>& scale) {
  VT_CHECK(x.size() == static_cast<size_t>(frames * hidden), "adaln: modulate shape");
  VT_CHECK(shift.size() == static_cast<size_t>(hidden) &&
               scale.size() == static_cast<size_t>(hidden),
           "adaln: shift/scale must be one per hidden unit");
  std::vector<float> out(x.size());
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t h = 0; h < hidden; ++h) {
      const size_t i = static_cast<size_t>(t * hidden + h);
      // 1 + scale, not scale.
      out[i] = static_cast<float>(static_cast<double>(x[i]) *
                                      (1.0 + static_cast<double>(scale[static_cast<size_t>(h)])) +
                                  static_cast<double>(shift[static_cast<size_t>(h)]));
    }
  }
  return out;
}

std::vector<float> LayerNormNoAffine(const std::vector<float>& x, int64_t frames, int64_t hidden,
                                     double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(frames * hidden), "adaln: layernorm shape");
  std::vector<float> out(x.size());
  for (int64_t t = 0; t < frames; ++t) {
    const float* row = x.data() + static_cast<size_t>(t * hidden);
    double mean = 0.0;
    for (int64_t h = 0; h < hidden; ++h) mean += static_cast<double>(row[h]);
    mean /= static_cast<double>(hidden);
    double var = 0.0;
    for (int64_t h = 0; h < hidden; ++h) {
      const double d = static_cast<double>(row[h]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(hidden);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t h = 0; h < hidden; ++h) {
      // No gamma, no beta: elementwise_affine is false.
      out[static_cast<size_t>(t * hidden + h)] =
          static_cast<float>((static_cast<double>(row[h]) - mean) * inv);
    }
  }
  return out;
}

namespace {

std::vector<float> WeightNorm(const std::vector<float>& g, const std::vector<float>& v,
                              int64_t out_dim) {
  const int64_t per = static_cast<int64_t>(v.size()) / out_dim;
  std::vector<float> w(v.size());
  for (int64_t o = 0; o < out_dim; ++o) {
    double norm = 0.0;
    for (int64_t i = 0; i < per; ++i) {
      const double q = static_cast<double>(v[static_cast<size_t>(o * per + i)]);
      norm += q * q;
    }
    norm = std::sqrt(norm);
    const double s = (norm > 0.0) ? static_cast<double>(g[static_cast<size_t>(o)]) / norm : 0.0;
    for (int64_t i = 0; i < per; ++i) {
      w[static_cast<size_t>(o * per + i)] =
          static_cast<float>(static_cast<double>(v[static_cast<size_t>(o * per + i)]) * s);
    }
  }
  return w;
}

double Silu(double x) { return x / (1.0 + std::exp(-x)); }

}  // namespace

std::vector<float> FinalLayer(const std::vector<float>& x, int64_t frames, int64_t hidden,
                              int64_t out_channels, const std::vector<float>& cond,
                              const FinalLayerWeights& w, double eps) {
  VT_CHECK(cond.size() == static_cast<size_t>(hidden), "adaln: cond shape");

  // SiLU then Linear(hidden -> 2*hidden); the result chunks into [shift, scale].
  std::vector<double> activated(static_cast<size_t>(hidden));
  for (int64_t h = 0; h < hidden; ++h) {
    activated[static_cast<size_t>(h)] = Silu(static_cast<double>(cond[static_cast<size_t>(h)]));
  }
  std::vector<float> shift(static_cast<size_t>(hidden)), scale(static_cast<size_t>(hidden));
  for (int64_t o = 0; o < 2 * hidden; ++o) {
    double acc = static_cast<double>(w.ada_b[static_cast<size_t>(o)]);
    for (int64_t h = 0; h < hidden; ++h) {
      acc += static_cast<double>(w.ada_w[static_cast<size_t>(o * hidden + h)]) *
             activated[static_cast<size_t>(h)];
    }
    // chunk(2): the FIRST half is shift, the second is scale.
    if (o < hidden) {
      shift[static_cast<size_t>(o)] = static_cast<float>(acc);
    } else {
      scale[static_cast<size_t>(o - hidden)] = static_cast<float>(acc);
    }
  }

  const std::vector<float> normed = LayerNormNoAffine(x, frames, hidden, eps);
  const std::vector<float> modulated = Modulate(normed, frames, hidden, shift, scale);

  const std::vector<float> lin = WeightNorm(w.linear_g, w.linear_v, out_channels);
  std::vector<float> out(static_cast<size_t>(frames * out_channels));
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t o = 0; o < out_channels; ++o) {
      double acc = static_cast<double>(w.linear_bias[static_cast<size_t>(o)]);
      for (int64_t h = 0; h < hidden; ++h) {
        acc += static_cast<double>(lin[static_cast<size_t>(o * hidden + h)]) *
               static_cast<double>(modulated[static_cast<size_t>(t * hidden + h)]);
      }
      out[static_cast<size_t>(t * out_channels + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace adaln
}  // namespace models
}  // namespace vllm
