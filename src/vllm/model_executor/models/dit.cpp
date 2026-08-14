// S2Mel DiT block primitives. See dit.h for the conventions this pins.
#include "vllm/model_executor/models/dit.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace dit {

std::vector<float> RmsNorm(const std::vector<float>& x, int64_t frames, int64_t dim,
                           const std::vector<float>& weight, double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(frames * dim), "dit: rmsnorm shape");
  std::vector<float> out(x.size());
  for (int64_t t = 0; t < frames; ++t) {
    const float* row = x.data() + static_cast<size_t>(t * dim);
    double sq = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
      const double v = static_cast<double>(row[i]);
      sq += v * v;   // NO mean subtraction: this is RMS, not variance.
    }
    const double inv = 1.0 / std::sqrt(sq / static_cast<double>(dim) + eps);
    for (int64_t i = 0; i < dim; ++i) {
      out[static_cast<size_t>(t * dim + i)] =
          static_cast<float>(static_cast<double>(row[i]) * inv *
                             static_cast<double>(weight[static_cast<size_t>(i)]));
    }
  }
  return out;
}

std::vector<float> AdaptiveLayerNorm(const std::vector<float>& x, int64_t frames, int64_t dim,
                                     const std::vector<float>& embedding,
                                     const std::vector<float>& proj_w,
                                     const std::vector<float>& proj_b,
                                     const std::vector<float>& norm_weight, double eps) {
  VT_CHECK(embedding.size() == static_cast<size_t>(dim), "dit: adaln embedding shape");
  // project(embedding) -> 2*dim, split into [weight, bias].
  std::vector<double> proj(static_cast<size_t>(2 * dim));
  for (int64_t o = 0; o < 2 * dim; ++o) {
    double acc = static_cast<double>(proj_b[static_cast<size_t>(o)]);
    for (int64_t i = 0; i < dim; ++i) {
      acc += static_cast<double>(proj_w[static_cast<size_t>(o * dim + i)]) *
             static_cast<double>(embedding[static_cast<size_t>(i)]);
    }
    proj[static_cast<size_t>(o)] = acc;
  }

  const std::vector<float> normed = RmsNorm(x, frames, dim, norm_weight, eps);
  std::vector<float> out(normed.size());
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t i = 0; i < dim; ++i) {
      const size_t k = static_cast<size_t>(t * dim + i);
      // weight * norm + bias. NOT (1 + weight) -- that is the FinalLayer's
      // convention, and this is a different one in the same model.
      out[k] = static_cast<float>(proj[static_cast<size_t>(i)] * static_cast<double>(normed[k]) +
                                  proj[static_cast<size_t>(dim + i)]);
    }
  }
  return out;
}

std::vector<float> ApplyRotary(const std::vector<float>& x, int64_t frames, int64_t heads,
                               int64_t head_dim, const std::vector<float>& freqs) {
  VT_CHECK(x.size() == static_cast<size_t>(frames * heads * head_dim), "dit: rotary shape");
  VT_CHECK(head_dim % 2 == 0, "dit: head_dim must be even");
  const int64_t pairs = head_dim / 2;
  VT_CHECK(freqs.size() == static_cast<size_t>(frames * pairs * 2), "dit: freqs shape");

  std::vector<float> out(x.size());
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t p = 0; p < pairs; ++p) {
        // ADJACENT pairing: components 2p and 2p+1.
        const size_t lo = static_cast<size_t>((t * heads + h) * head_dim + 2 * p);
        const size_t hi = lo + 1;
        const double cos_v = static_cast<double>(freqs[static_cast<size_t>((t * pairs + p) * 2)]);
        const double sin_v =
            static_cast<double>(freqs[static_cast<size_t>((t * pairs + p) * 2 + 1)]);
        const double a = static_cast<double>(x[lo]);
        const double b = static_cast<double>(x[hi]);
        out[lo] = static_cast<float>(a * cos_v - b * sin_v);
        out[hi] = static_cast<float>(b * cos_v + a * sin_v);
      }
    }
  }
  return out;
}

}  // namespace dit
}  // namespace models
}  // namespace vllm
