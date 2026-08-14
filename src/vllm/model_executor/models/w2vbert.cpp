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


std::vector<float> SelfAttentionRelativeKey(const std::vector<float>& x, int64_t frames,
                                            int64_t hidden, int64_t heads, int64_t left_max,
                                            int64_t right_max, const SelfAttentionWeights& w) {
  VT_CHECK(hidden % heads == 0, "w2vbert: hidden must divide by heads");
  const int64_t head_dim = hidden / heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  const std::vector<float> q = Linear(x, frames, hidden, hidden, w.q_w, w.q_b);
  const std::vector<float> k = Linear(x, frames, hidden, hidden, w.k_w, w.k_b);
  const std::vector<float> v = Linear(x, frames, hidden, hidden, w.v_w, w.v_b);

  std::vector<float> ctx(static_cast<size_t>(frames * hidden));
  for (int64_t h = 0; h < heads; ++h) {
    for (int64_t i = 0; i < frames; ++i) {
      std::vector<double> scores(static_cast<size_t>(frames));
      for (int64_t j = 0; j < frames; ++j) {
        double dot = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) {
          dot += static_cast<double>(q[static_cast<size_t>(i * hidden + h * head_dim + d)]) *
                 static_cast<double>(k[static_cast<size_t>(j * hidden + h * head_dim + d)]);
        }
        dot *= scale;

        // distance is key MINUS query, clamped asymmetrically, then shifted into
        // the embedding table by +left_max.
        int64_t dist = j - i;
        if (dist < -left_max) dist = -left_max;
        if (dist > right_max) dist = right_max;
        const int64_t row = dist + left_max;
        double rel = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) {
          rel += static_cast<double>(q[static_cast<size_t>(i * hidden + h * head_dim + d)]) *
                 static_cast<double>(
                     w.distance_embedding[static_cast<size_t>(row * head_dim + d)]);
        }
        // divided by sqrt(d) AGAIN, separately from the score above.
        scores[static_cast<size_t>(j)] = dot + rel * scale;
      }

      double best = scores[0];
      for (const double s : scores) best = std::max(best, s);
      double denom = 0.0;
      for (double& s : scores) { s = std::exp(s - best); denom += s; }
      for (int64_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j < frames; ++j) {
          acc += scores[static_cast<size_t>(j)] *
                 static_cast<double>(v[static_cast<size_t>(j * hidden + h * head_dim + d)]);
        }
        ctx[static_cast<size_t>(i * hidden + h * head_dim + d)] =
            static_cast<float>(acc / denom);
      }
    }
  }
  return Linear(ctx, frames, hidden, hidden, w.out_w, w.out_b);
}


std::vector<float> EncoderLayer(const std::vector<float>& x, int64_t frames, int64_t hidden,
                                int64_t heads, int64_t intermediate, int64_t conv_kernel,
                                int64_t left_max, int64_t right_max, const EncoderLayerWeights& w,
                                double eps) {
  std::vector<float> h = x;

  // 1. macaron feed-forward, HALF weighted.
  {
    const std::vector<float> n = LayerNorm(h, frames, hidden, w.ffn1_ln_gamma, w.ffn1_ln_beta, eps);
    const std::vector<float> f = FeedForward(n, frames, hidden, intermediate, w.ffn1_in_w,
                                             w.ffn1_in_b, w.ffn1_out_w, w.ffn1_out_b);
    for (size_t i = 0; i < h.size(); ++i) h[i] = f[i] * 0.5F + h[i];
  }

  // 2. self-attention, FULL residual.
  {
    const std::vector<float> n = LayerNorm(h, frames, hidden, w.attn_ln_gamma, w.attn_ln_beta, eps);
    const std::vector<float> a =
        SelfAttentionRelativeKey(n, frames, hidden, heads, left_max, right_max, w.attn);
    for (size_t i = 0; i < h.size(); ++i) h[i] = a[i] + h[i];
  }

  // 3. convolution module -- it applies its OWN layer_norm internally, so there
  //    is no norm here; adding one would double-normalize.
  {
    const std::vector<float> c = ConvModule(h, frames, hidden, conv_kernel, w.conv, eps);
    for (size_t i = 0; i < h.size(); ++i) h[i] = h[i] + c[i];
  }

  // 4. second macaron feed-forward, HALF weighted.
  {
    const std::vector<float> n = LayerNorm(h, frames, hidden, w.ffn2_ln_gamma, w.ffn2_ln_beta, eps);
    const std::vector<float> f = FeedForward(n, frames, hidden, intermediate, w.ffn2_in_w,
                                             w.ffn2_in_b, w.ffn2_out_w, w.ffn2_out_b);
    for (size_t i = 0; i < h.size(); ++i) h[i] = f[i] * 0.5F + h[i];
  }

  return LayerNorm(h, frames, hidden, w.final_ln_gamma, w.final_ln_beta, eps);
}


std::vector<float> FeatureProjection(const std::vector<float>& x, int64_t frames, int64_t in_dim,
                                     int64_t hidden, const std::vector<float>& ln_gamma,
                                     const std::vector<float>& ln_beta,
                                     const std::vector<float>& proj_w,
                                     const std::vector<float>& proj_b, double eps,
                                     std::vector<float>* norm_out) {
  const std::vector<float> normed = LayerNorm(x, frames, in_dim, ln_gamma, ln_beta, eps);
  if (norm_out != nullptr) *norm_out = normed;
  return Linear(normed, frames, in_dim, hidden, proj_w, proj_b);
}

std::vector<float> EncoderStack(const std::vector<float>& x, int64_t frames, int64_t hidden,
                                int64_t heads, int64_t intermediate, int64_t conv_kernel,
                                int64_t left_max, int64_t right_max,
                                const std::vector<EncoderLayerWeights>& layers, double eps) {
  std::vector<float> h = x;
  for (const EncoderLayerWeights& l : layers) {
    h = EncoderLayer(h, frames, hidden, heads, intermediate, conv_kernel, left_max, right_max, l,
                     eps);
  }
  // No final layer norm: the encoder returns the last layer's output directly.
  return h;
}

}  // namespace w2vbert
}  // namespace models
}  // namespace vllm
