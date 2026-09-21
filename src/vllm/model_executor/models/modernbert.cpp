// ModernBERT encoder host reference. See modernbert.h for the upstream anchors.
#include "vllm/model_executor/models/modernbert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace modernbert {

void CheckpointTensors::Set(const std::string& name, std::vector<int64_t> shape,
                             std::vector<float> data) {
  int64_t count = 1;
  for (const int64_t dim : shape) count *= dim;
  VT_CHECK(static_cast<size_t>(count) == data.size(),
           "modernbert checkpoint: element count does not match shape");
  shapes[name] = std::move(shape);
  values[name] = std::move(data);
}

const std::vector<float>& CheckpointTensors::Get(const std::string& name) const {
  const auto it = values.find(name);
  VT_CHECK(it != values.end(), "modernbert checkpoint: missing tensor '" + name + "'");
  return it->second;
}

const std::vector<int64_t>& CheckpointTensors::Shape(const std::string& name) const {
  const auto it = shapes.find(name);
  VT_CHECK(it != shapes.end(), "modernbert checkpoint: missing shape for '" + name + "'");
  return it->second;
}

namespace {

// y = W x + b, with W stored [out, in] (nn.Linear orientation) and x a single
// row of `in` values. When b is empty, no bias is added.
void LinearRow(const float* x, const std::vector<float>& w, const std::vector<float>& b,
               int64_t in_dim, int64_t out_dim, float* out) {
  for (int64_t o = 0; o < out_dim; ++o) {
    double acc = b.empty() ? 0.0 : static_cast<double>(b[static_cast<size_t>(o)]);
    const float* row = w.data() + static_cast<size_t>(o * in_dim);
    for (int64_t i = 0; i < in_dim; ++i) {
      acc += static_cast<double>(row[i]) * static_cast<double>(x[i]);
    }
    out[o] = static_cast<float>(acc);
  }
}

// torch.nn.LayerNorm over the last dimension, with the biased (1/N) variance
// torch uses. When `b` is empty, no bias (affine shift) is applied — matching
// ModernBERT's norm_bias=false.
void LayerNormRow(const float* x, const std::vector<float>& w, const std::vector<float>& b,
                  int64_t n, double eps, float* out) {
  double mean = 0.0;
  for (int64_t i = 0; i < n; ++i) mean += static_cast<double>(x[i]);
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(x[i]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(n);
  const double inv = 1.0 / std::sqrt(var + eps);
  if (b.empty()) {
    for (int64_t i = 0; i < n; ++i) {
      const double normed = (static_cast<double>(x[i]) - mean) * inv;
      out[i] = static_cast<float>(normed * static_cast<double>(w[static_cast<size_t>(i)]));
    }
  } else {
    for (int64_t i = 0; i < n; ++i) {
      const double normed = (static_cast<double>(x[i]) - mean) * inv;
      out[i] = static_cast<float>(normed * static_cast<double>(w[static_cast<size_t>(i)]) +
                                  static_cast<double>(b[static_cast<size_t>(i)]));
    }
  }
}

// Exact GELU (erf-based), matching ACT2FN["gelu"] in HF transformers.
double Gelu(double x) {
  constexpr double kSqrt1Over2 = 0.70710678118654752440;  // 1/sqrt(2)
  return x * 0.5 * (1.0 + std::erf(x * kSqrt1Over2));
}

// Apply RoPE to a single head's Q or K vector using the "rotate_half" style
// (GPT-J convention, matching transformers' RotaryEmbedding).
//
// freq_i = 1 / (theta ^ (2i / dim))  for i in [0, dim/2)
// angle_pos_i = pos * freq_i
// cos_i = cos(angle_pos_i), sin_i = sin(angle_pos_i)
// x'[i]         = x[i] * cos_i - x[i + dim/2] * sin_i
// x'[i + dim/2] = x[i + dim/2] * cos_i + x[i] * sin_i
void ApplyRope(float* x, int64_t dim, int64_t pos, double theta) {
  const int64_t half = dim / 2;
  for (int64_t i = 0; i < half; ++i) {
    const double freq = 1.0 / std::pow(theta, static_cast<double>(2 * i) / static_cast<double>(dim));
    const double angle = static_cast<double>(pos) * freq;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double x1 = static_cast<double>(x[i]);
    const double x2 = static_cast<double>(x[static_cast<size_t>(i + half)]);
    x[i] = static_cast<float>(x1 * c - x2 * s);
    x[static_cast<size_t>(i + half)] = static_cast<float>(x2 * c + x1 * s);
  }
}

}  // namespace

Weights Load(const Params& params, const CheckpointTensors& tensors) {
  VT_CHECK(params.hidden_size > 0 && params.num_attention_heads > 0,
           "modernbert: hidden_size and num_attention_heads must be positive");
  VT_CHECK(params.hidden_size % params.num_attention_heads == 0,
           "modernbert: hidden_size must divide by num_attention_heads");

  Weights w;
  w.tok_embeddings = tensors.Get("encoder.embeddings.tok_embeddings.weight");
  w.emb_norm_weight = tensors.Get("encoder.embeddings.norm.weight");
  w.final_norm_weight = tensors.Get("encoder.final_norm.weight");

  w.layers.resize(static_cast<size_t>(params.num_hidden_layers));
  for (int64_t i = 0; i < params.num_hidden_layers; ++i) {
    const std::string p = "encoder.layers." + std::to_string(i) + ".";
    LayerWeights& l = w.layers[static_cast<size_t>(i)];
    l.qkv_weight = tensors.Get(p + "attn.Wqkv.weight");
    l.attn_out_weight = tensors.Get(p + "attn.Wo.weight");
    // Layer 0 has Identity attn_norm (no weight tensor).
    if (i > 0) {
      l.attn_norm_weight = tensors.Get(p + "attn_norm.weight");
    }
    l.mlp_gate_up_weight = tensors.Get(p + "mlp.Wi.weight");
    l.mlp_down_weight = tensors.Get(p + "mlp.Wo.weight");
    l.mlp_norm_weight = tensors.Get(p + "mlp_norm.weight");
  }
  return w;
}

std::vector<float> ForwardHost(const Params& params, const Weights& weights,
                                const std::vector<int64_t>& input_ids) {
  const int64_t h = params.hidden_size;
  const int64_t seq = static_cast<int64_t>(input_ids.size());
  const int64_t heads = params.num_attention_heads;
  const int64_t hd = params.resolved_head_dim();
  const int64_t inter = params.intermediate_size;
  const int64_t sw = params.sliding_window();
  const double eps = params.layer_norm_eps;

  // ── Embeddings ──────────────────────────────────────────────────────────
  // ModernBertEmbeddings: token lookup → LayerNorm(bias=norm_bias).
  // No position embeddings (RoPE provides positional information).
  std::vector<float> x(static_cast<size_t>(seq * h));
  for (int64_t t = 0; t < seq; ++t) {
    const int64_t id = input_ids[static_cast<size_t>(t)];
    VT_CHECK(id >= 0 && id < params.vocab_size, "modernbert: input id out of range");
    const float* emb = weights.tok_embeddings.data() + static_cast<size_t>(id * h);
    LayerNormRow(emb, weights.emb_norm_weight, {}, h, eps, &x[static_cast<size_t>(t * h)]);
  }

  // ── Per-layer forward ──────────────────────────────────────────────────
  std::vector<float> normed(static_cast<size_t>(h));
  std::vector<float> qkv(static_cast<size_t>(3 * h));
  std::vector<float> q(static_cast<size_t>(seq * h));
  std::vector<float> k(static_cast<size_t>(seq * h));
  std::vector<float> v(static_cast<size_t>(seq * h));
  std::vector<float> attn_out(static_cast<size_t>(h));
  std::vector<float> proj(static_cast<size_t>(h));
  std::vector<float> scores(static_cast<size_t>(seq));
  std::vector<float> mlp_inter(static_cast<size_t>(2 * inter));
  std::vector<float> mlp_act(static_cast<size_t>(inter));
  std::vector<float> mlp_out(static_cast<size_t>(h));

  for (int64_t layer = 0; layer < params.num_hidden_layers; ++layer) {
    const LayerWeights& lw = weights.layers[static_cast<size_t>(layer)];
    const bool is_global = params.is_global_layer(layer);
    const double rope_theta = is_global ? params.global_rope_theta : params.local_rope_theta;

    // ── Attention pre-norm ────────────────────────────────────────────────
    // Layer 0: Identity (skip norm). Layers 1+: LayerNorm.
    for (int64_t t = 0; t < seq; ++t) {
      float* normed_t = normed.data();
      const float* xt = x.data() + static_cast<size_t>(t * h);
      if (layer == 0) {
        std::copy(xt, xt + h, normed_t);
      } else {
        LayerNormRow(xt, lw.attn_norm_weight, {}, h, eps, normed_t);
      }

      // ── QKV projection (combined Wqkv) ───────────────────────────────
      LinearRow(normed_t, lw.qkv_weight, {}, h, 3 * h, qkv.data());
      // Split into Q, K, V
      std::copy(qkv.begin(), qkv.begin() + h, q.begin() + static_cast<size_t>(t * h));
      std::copy(qkv.begin() + h, qkv.begin() + 2 * h, k.begin() + static_cast<size_t>(t * h));
      std::copy(qkv.begin() + 2 * h, qkv.begin() + 3 * h, v.begin() + static_cast<size_t>(t * h));

      // ── Apply RoPE to Q and K (per head) ─────────────────────────────
      for (int64_t hd_idx = 0; hd_idx < heads; ++hd_idx) {
        const int64_t offset = static_cast<size_t>(t * h + hd_idx * hd);
        ApplyRope(q.data() + offset, hd, t, rope_theta);
        ApplyRope(k.data() + offset, hd, t, rope_theta);
      }
    }

    // ── Attention: multi-head, bidirectional, optional sliding window ────
    // Each head computes its own scores, softmax, and V mixing independently.
    const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
    for (int64_t t = 0; t < seq; ++t) {
      for (int64_t hd_idx = 0; hd_idx < heads; ++hd_idx) {
        const float* qh = q.data() + static_cast<size_t>(t * h + hd_idx * hd);

        // Scores for this head, query position t.
        double max_score = -std::numeric_limits<double>::max();
        for (int64_t s = 0; s < seq; ++s) {
          if (!is_global && sw > 0 && std::abs(t - s) > sw) {
            scores[static_cast<size_t>(s)] = -1e30f;
            continue;
          }
          double dot = 0.0;
          const float* kh = k.data() + static_cast<size_t>(s * h + hd_idx * hd);
          for (int64_t d = 0; d < hd; ++d) {
            dot += static_cast<double>(qh[d]) * static_cast<double>(kh[d]);
          }
          const double score = dot * attn_scale;
          scores[static_cast<size_t>(s)] = static_cast<float>(score);
          if (score > max_score) max_score = score;
        }

        // Softmax (bidirectional — no causal mask).
        double sum_exp = 0.0;
        for (int64_t s = 0; s < seq; ++s) {
          const double e = std::exp(static_cast<double>(scores[static_cast<size_t>(s)]) - max_score);
          scores[static_cast<size_t>(s)] = static_cast<float>(e);
          sum_exp += e;
        }
        const double inv_sum = 1.0 / sum_exp;

        // Weighted sum of V for this head.
        float* out_head = attn_out.data() + static_cast<size_t>(hd_idx * hd);
        for (int64_t d = 0; d < hd; ++d) {
          double acc = 0.0;
          for (int64_t s = 0; s < seq; ++s) {
            acc += static_cast<double>(scores[static_cast<size_t>(s)]) * inv_sum *
                   static_cast<double>(v[static_cast<size_t>(s * h + hd_idx * hd + d)]);
          }
          out_head[static_cast<size_t>(d)] = static_cast<float>(acc);
        }
      }

      // Output projection Wo → residual add
      LinearRow(attn_out.data(), lw.attn_out_weight, {}, h, h, proj.data());
      for (int64_t d = 0; d < h; ++d) {
        x[static_cast<size_t>(t * h + d)] += proj[static_cast<size_t>(d)];
      }
    }

    // ── MLP pre-norm + GeGLU ────────────────────────────────────────────
    for (int64_t t = 0; t < seq; ++t) {
      const float* xt = x.data() + static_cast<size_t>(t * h);
      LayerNormRow(xt, lw.mlp_norm_weight, {}, h, eps, normed.data());

      // Gate-up projection: Wi [2*I, H] → input [I] + gate [I]
      LinearRow(normed.data(), lw.mlp_gate_up_weight, {}, h, 2 * inter, mlp_inter.data());
      // GeGLU: GELU(input) * gate
      for (int64_t i = 0; i < inter; ++i) {
        const double inp = static_cast<double>(mlp_inter[static_cast<size_t>(i)]);
        const double gate = static_cast<double>(mlp_inter[static_cast<size_t>(i + inter)]);
        mlp_act[static_cast<size_t>(i)] = static_cast<float>(Gelu(inp) * gate);
      }

      // Down projection Wo [H, I] → residual add
      LinearRow(mlp_act.data(), lw.mlp_down_weight, {}, inter, h, mlp_out.data());
      for (int64_t d = 0; d < h; ++d) {
        x[static_cast<size_t>(t * h + d)] += mlp_out[static_cast<size_t>(d)];
      }
    }
  }

  // ── Final norm ──────────────────────────────────────────────────────────
  for (int64_t t = 0; t < seq; ++t) {
    LayerNormRow(&x[static_cast<size_t>(t * h)], weights.final_norm_weight, {}, h, eps,
                 &x[static_cast<size_t>(t * h)]);
  }

  return x;
}

}  // namespace modernbert
}  // namespace vllm
