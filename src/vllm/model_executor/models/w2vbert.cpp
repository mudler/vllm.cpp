// w2v-bert-2.0 Conformer pieces. See w2vbert.h for the upstream anchors.
#include "vllm/model_executor/models/w2vbert.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace w2vbert {

double Swish(double x) { return x / (1.0 + std::exp(-x)); }

std::vector<float> LayerNorm(const std::vector<float>& x, int64_t frames, int64_t dim,
                             const std::vector<float>& gamma, const std::vector<float>& beta,
                             double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(frames * dim), "w2vbert: LayerNorm shape");
  std::vector<float> out(x.size());
  for (int64_t t = 0; t < frames; ++t) {
    const float* row = x.data() + static_cast<size_t>(t * dim);
    double mean = 0.0;
    for (int64_t i = 0; i < dim; ++i) mean += static_cast<double>(row[i]);
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
      const double d = static_cast<double>(row[i]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t i = 0; i < dim; ++i) {
      out[static_cast<size_t>(t * dim + i)] = static_cast<float>(
          ((static_cast<double>(row[i]) - mean) * inv) * static_cast<double>(gamma[static_cast<size_t>(i)]) +
          static_cast<double>(beta[static_cast<size_t>(i)]));
    }
  }
  return out;
}

namespace {

// y = x W^T + b, with W stored [out, in] as torch.nn.Linear does.
std::vector<float> Linear(const std::vector<float>& x, int64_t frames, int64_t in_dim,
                          int64_t out_dim, const std::vector<float>& w,
                          const std::vector<float>& b) {
  std::vector<float> out(static_cast<size_t>(frames * out_dim));
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t o = 0; o < out_dim; ++o) {
      double acc = b.empty() ? 0.0 : static_cast<double>(b[static_cast<size_t>(o)]);
      for (int64_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(w[static_cast<size_t>(o * in_dim + i)]) *
               static_cast<double>(x[static_cast<size_t>(t * in_dim + i)]);
      }
      out[static_cast<size_t>(t * out_dim + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

std::vector<float> FeedForward(const std::vector<float>& x, int64_t frames, int64_t hidden,
                               int64_t intermediate, const std::vector<float>& in_w,
                               const std::vector<float>& in_b, const std::vector<float>& out_w,
                               const std::vector<float>& out_b) {
  std::vector<float> h = Linear(x, frames, hidden, intermediate, in_w, in_b);
  for (float& v : h) v = static_cast<float>(Swish(static_cast<double>(v)));
  return Linear(h, frames, intermediate, hidden, out_w, out_b);
}

std::vector<float> ConvModule(const std::vector<float>& x, int64_t frames, int64_t hidden,
                              int64_t kernel, const ConvModuleWeights& w, double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(frames * hidden), "w2vbert: ConvModule shape");
  const std::vector<float> normed = LayerNorm(x, frames, hidden, w.ln_gamma, w.ln_beta, eps);

  // torch works [C, T] from here; the weights are Conv1d [out, in/groups, k].
  std::vector<float> ct(static_cast<size_t>(hidden * frames));
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t c = 0; c < hidden; ++c) {
      ct[static_cast<size_t>(c * frames + t)] = normed[static_cast<size_t>(t * hidden + c)];
    }
  }

  // pointwise_conv1: [2H, H, 1] -> 2H channels, then GLU halves them back to H.
  std::vector<float> wide(static_cast<size_t>(2 * hidden * frames));
  for (int64_t o = 0; o < 2 * hidden; ++o) {
    for (int64_t t = 0; t < frames; ++t) {
      double acc = 0.0;
      for (int64_t c = 0; c < hidden; ++c) {
        acc += static_cast<double>(w.pointwise1[static_cast<size_t>(o * hidden + c)]) *
               static_cast<double>(ct[static_cast<size_t>(c * frames + t)]);
      }
      wide[static_cast<size_t>(o * frames + t)] = static_cast<float>(acc);
    }
  }
  // GLU over the CHANNEL axis: first half gated by sigmoid(second half).
  std::vector<float> gated(static_cast<size_t>(hidden * frames));
  for (int64_t c = 0; c < hidden; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      const double a = static_cast<double>(wide[static_cast<size_t>(c * frames + t)]);
      const double b = static_cast<double>(wide[static_cast<size_t>((hidden + c) * frames + t)]);
      gated[static_cast<size_t>(c * frames + t)] =
          static_cast<float>(a * (1.0 / (1.0 + std::exp(-b))));
    }
  }

  // CAUSAL pad: kernel-1 on the LEFT only, so a frame never sees the future.
  const int64_t pad = kernel - 1;
  std::vector<float> dw(static_cast<size_t>(hidden * frames));
  for (int64_t c = 0; c < hidden; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      double acc = 0.0;
      for (int64_t k = 0; k < kernel; ++k) {
        const int64_t src = t + k - pad;
        if (src < 0) continue;  // left pad only
        acc += static_cast<double>(w.depthwise[static_cast<size_t>(c * kernel + k)]) *
               static_cast<double>(gated[static_cast<size_t>(c * frames + src)]);
      }
      dw[static_cast<size_t>(c * frames + t)] = static_cast<float>(acc);
    }
  }

  // depthwise_layer_norm runs over the FEATURE axis, so transpose back first.
  std::vector<float> tc(static_cast<size_t>(frames * hidden));
  for (int64_t c = 0; c < hidden; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      tc[static_cast<size_t>(t * hidden + c)] = dw[static_cast<size_t>(c * frames + t)];
    }
  }
  tc = LayerNorm(tc, frames, hidden, w.dw_ln_gamma, w.dw_ln_beta, eps);
  for (float& v : tc) v = static_cast<float>(Swish(static_cast<double>(v)));

  // pointwise_conv2 back to [T, H].
  std::vector<float> out(static_cast<size_t>(frames * hidden));
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t o = 0; o < hidden; ++o) {
      double acc = 0.0;
      for (int64_t c = 0; c < hidden; ++c) {
        acc += static_cast<double>(w.pointwise2[static_cast<size_t>(o * hidden + c)]) *
               static_cast<double>(tc[static_cast<size_t>(t * hidden + c)]);
      }
      out[static_cast<size_t>(t * hidden + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace w2vbert
}  // namespace models
}  // namespace vllm
