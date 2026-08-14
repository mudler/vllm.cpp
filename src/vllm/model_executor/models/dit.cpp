// S2Mel DiT block primitives. See dit.h for the conventions this pins.
#include "vllm/model_executor/models/dit.h"

#include <cmath>
#include <limits>
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


namespace {

std::vector<float> Linear(const std::vector<float>& x, int64_t frames, int64_t in_dim,
                          int64_t out_dim, const std::vector<float>& w) {
  std::vector<float> out(static_cast<size_t>(frames * out_dim));
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t o = 0; o < out_dim; ++o) {
      double acc = 0.0;
      for (int64_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(w[static_cast<size_t>(o * in_dim + i)]) *
               static_cast<double>(x[static_cast<size_t>(t * in_dim + i)]);
      }
      out[static_cast<size_t>(t * out_dim + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

double Silu(double x) { return x / (1.0 + std::exp(-x)); }

}  // namespace

std::vector<float> SwiGlu(const std::vector<float>& x, int64_t frames, int64_t dim,
                          int64_t intermediate, const std::vector<float>& w1,
                          const std::vector<float>& w3, const std::vector<float>& w2) {
  const std::vector<float> gate = Linear(x, frames, dim, intermediate, w1);
  const std::vector<float> up = Linear(x, frames, dim, intermediate, w3);
  std::vector<float> mid(gate.size());
  for (size_t i = 0; i < gate.size(); ++i) {
    // SiLU applies to W1's output only.
    mid[i] = static_cast<float>(Silu(static_cast<double>(gate[i])) * static_cast<double>(up[i]));
  }
  return Linear(mid, frames, intermediate, dim, w2);
}

std::vector<float> Block(const std::vector<float>& x, const std::vector<float>& cond,
                         int64_t frames, int64_t dim, int64_t heads, int64_t head_dim,
                         int64_t intermediate, const std::vector<float>& freqs,
                         const BlockWeights& w, double eps) {
  VT_CHECK(x.size() == static_cast<size_t>(frames * dim), "dit: block shape");

  // ── attention half ────────────────────────────────────────────────────────
  const std::vector<float> normed =
      AdaptiveLayerNorm(x, frames, dim, cond, w.attn_proj_w, w.attn_proj_b, w.attn_norm_w, eps);

  const int64_t qkv_dim = heads * head_dim;
  const std::vector<float> qkv = Linear(normed, frames, dim, 3 * qkv_dim, w.wqkv);
  std::vector<float> q(static_cast<size_t>(frames * qkv_dim));
  std::vector<float> k(q.size()), v(q.size());
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t i = 0; i < qkv_dim; ++i) {
      const size_t base = static_cast<size_t>(t * 3 * qkv_dim);
      q[static_cast<size_t>(t * qkv_dim + i)] = qkv[base + static_cast<size_t>(i)];
      k[static_cast<size_t>(t * qkv_dim + i)] = qkv[base + static_cast<size_t>(qkv_dim + i)];
      v[static_cast<size_t>(t * qkv_dim + i)] = qkv[base + static_cast<size_t>(2 * qkv_dim + i)];
    }
  }
  // Rotary applies to q and k, never to v.
  const std::vector<float> qr = ApplyRotary(q, frames, heads, head_dim, freqs);
  const std::vector<float> kr = ApplyRotary(k, frames, heads, head_dim, freqs);

  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<float> ctx(static_cast<size_t>(frames * qkv_dim));
  for (int64_t hd = 0; hd < heads; ++hd) {
    for (int64_t i = 0; i < frames; ++i) {
      std::vector<double> scores(static_cast<size_t>(frames));
      double best = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j < frames; ++j) {
        double dot = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) {
          dot += static_cast<double>(qr[static_cast<size_t>(i * qkv_dim + hd * head_dim + d)]) *
                 static_cast<double>(kr[static_cast<size_t>(j * qkv_dim + hd * head_dim + d)]);
        }
        dot *= scale;
        scores[static_cast<size_t>(j)] = dot;
        best = std::max(best, dot);
      }
      double denom = 0.0;
      for (double& sc : scores) { sc = std::exp(sc - best); denom += sc; }
      for (int64_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j < frames; ++j) {
          acc += scores[static_cast<size_t>(j)] *
                 static_cast<double>(v[static_cast<size_t>(j * qkv_dim + hd * head_dim + d)]);
        }
        ctx[static_cast<size_t>(i * qkv_dim + hd * head_dim + d)] =
            static_cast<float>(acc / denom);
      }
    }
  }
  const std::vector<float> attn_out = Linear(ctx, frames, qkv_dim, dim, w.wo);

  std::vector<float> h(x.size());
  for (size_t i = 0; i < h.size(); ++i) h[i] = x[i] + attn_out[i];  // FULL residual

  // ── feed-forward half ─────────────────────────────────────────────────────
  const std::vector<float> ffn_normed =
      AdaptiveLayerNorm(h, frames, dim, cond, w.ffn_proj_w, w.ffn_proj_b, w.ffn_norm_w, eps);
  const std::vector<float> ffn = SwiGlu(ffn_normed, frames, dim, intermediate, w.w1, w.w3, w.w2);

  std::vector<float> out(h.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = h[i] + ffn[i];  // FULL residual
  return out;
}

}  // namespace dit
}  // namespace models
}  // namespace vllm
