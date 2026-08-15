// VocosBackbone. See vocos.h for the upstream anchors.
#include "vllm/model_executor/models/vocos.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace vocos {
namespace {

// LayerNorm over the CHANNEL axis of a [dim, frames] signal, i.e. per frame.
void LayerNormChannels(std::vector<float>& x, int64_t dim, int64_t frames,
                       const std::vector<float>& gamma, const std::vector<float>& beta,
                       double eps) {
  for (int64_t t = 0; t < frames; ++t) {
    double mean = 0.0;
    for (int64_t c = 0; c < dim; ++c) mean += static_cast<double>(x[static_cast<size_t>(c * frames + t)]);
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (int64_t c = 0; c < dim; ++c) {
      const double d = static_cast<double>(x[static_cast<size_t>(c * frames + t)]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t c = 0; c < dim; ++c) {
      const size_t i = static_cast<size_t>(c * frames + t);
      x[i] = static_cast<float>((static_cast<double>(x[i]) - mean) * inv *
                                    static_cast<double>(gamma[static_cast<size_t>(c)]) +
                                static_cast<double>(beta[static_cast<size_t>(c)]));
    }
  }
}

// nn.GELU default is the EXACT erf form, not the tanh approximation.
double Gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

}  // namespace

std::vector<float> ConvNeXtBlock(const std::vector<float>& x, int64_t dim, int64_t frames,
                                 int64_t intermediate, const BlockWeights& w, double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(dim * frames), "vocos: block shape");
  constexpr int64_t kKernel = 7;
  const int64_t pad = 3;

  // depthwise: groups == dim, so each channel has its own [1, 7] kernel.
  std::vector<float> h(static_cast<size_t>(dim * frames));
  for (int64_t c = 0; c < dim; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      double acc = w.dw_bias.empty() ? 0.0 : static_cast<double>(w.dw_bias[static_cast<size_t>(c)]);
      for (int64_t k = 0; k < kKernel; ++k) {
        const int64_t src = t + k - pad;
        if (src < 0 || src >= frames) continue;
        acc += static_cast<double>(w.dw_weight[static_cast<size_t>(c * kKernel + k)]) *
               static_cast<double>(x[static_cast<size_t>(c * frames + src)]);
      }
      h[static_cast<size_t>(c * frames + t)] = static_cast<float>(acc);
    }
  }

  LayerNormChannels(h, dim, frames, w.ln_gamma, w.ln_beta, eps);

  // pointwise linears act per FRAME over the channel axis.
  std::vector<float> out(static_cast<size_t>(dim * frames));
  std::vector<double> mid(static_cast<size_t>(intermediate));
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t o = 0; o < intermediate; ++o) {
      double acc = static_cast<double>(w.pw1_b[static_cast<size_t>(o)]);
      for (int64_t c = 0; c < dim; ++c) {
        acc += static_cast<double>(w.pw1_w[static_cast<size_t>(o * dim + c)]) *
               static_cast<double>(h[static_cast<size_t>(c * frames + t)]);
      }
      mid[static_cast<size_t>(o)] = Gelu(acc);
    }
    for (int64_t c = 0; c < dim; ++c) {
      double acc = static_cast<double>(w.pw2_b[static_cast<size_t>(c)]);
      for (int64_t o = 0; o < intermediate; ++o) {
        acc += static_cast<double>(w.pw2_w[static_cast<size_t>(c * intermediate + o)]) *
               mid[static_cast<size_t>(o)];
      }
      // LEARNED layer scale, then the residual.
      acc *= static_cast<double>(w.gamma[static_cast<size_t>(c)]);
      out[static_cast<size_t>(c * frames + t)] =
          static_cast<float>(acc + static_cast<double>(x[static_cast<size_t>(c * frames + t)]));
    }
  }
  return out;
}

std::vector<float> Backbone(const std::vector<float>& x, int64_t input_channels, int64_t frames,
                            int64_t dim, int64_t intermediate, const BackboneWeights& w,
                            double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(input_channels * frames), "vocos: backbone shape");
  constexpr int64_t kKernel = 7;
  const int64_t pad = 3;

  std::vector<float> h(static_cast<size_t>(dim * frames));
  for (int64_t o = 0; o < dim; ++o) {
    for (int64_t t = 0; t < frames; ++t) {
      double acc = static_cast<double>(w.embed_b[static_cast<size_t>(o)]);
      for (int64_t c = 0; c < input_channels; ++c) {
        for (int64_t k = 0; k < kKernel; ++k) {
          const int64_t src = t + k - pad;
          if (src < 0 || src >= frames) continue;
          acc += static_cast<double>(
                     w.embed_w[static_cast<size_t>((o * input_channels + c) * kKernel + k)]) *
                 static_cast<double>(x[static_cast<size_t>(c * frames + src)]);
        }
      }
      h[static_cast<size_t>(o * frames + t)] = static_cast<float>(acc);
    }
  }

  LayerNormChannels(h, dim, frames, w.norm_gamma, w.norm_beta, eps);
  for (const BlockWeights& b : w.blocks) {
    h = ConvNeXtBlock(h, dim, frames, intermediate, b, eps);
  }
  LayerNormChannels(h, dim, frames, w.final_gamma, w.final_beta, eps);

  // The final norm is applied to the TRANSPOSED tensor upstream and never
  // transposed back, so the result is [frames, dim].
  std::vector<float> out(static_cast<size_t>(frames * dim));
  for (int64_t c = 0; c < dim; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      out[static_cast<size_t>(t * dim + c)] = h[static_cast<size_t>(c * frames + t)];
    }
  }
  return out;
}

}  // namespace vocos
}  // namespace models
}  // namespace vllm
