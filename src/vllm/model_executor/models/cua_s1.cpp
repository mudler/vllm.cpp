// cua-s1-forms TinyTransformerScorer host reference.
// See cua_s1.h for upstream anchors.
#include "vllm/model_executor/models/cua_s1.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace cua_s1 {

void CheckpointTensors::Set(const std::string& name, std::vector<int64_t> shape,
                             std::vector<float> data) {
  int64_t count = 1;
  for (const int64_t dim : shape) count *= dim;
  VT_CHECK(static_cast<size_t>(count) == data.size(),
           "cua_s1 checkpoint: element count does not match shape");
  shapes[name] = std::move(shape);
  values[name] = std::move(data);
}

const std::vector<float>& CheckpointTensors::Get(const std::string& name) const {
  const auto it = values.find(name);
  VT_CHECK(it != values.end(), "cua_s1 checkpoint: missing tensor '" + name + "'");
  return it->second;
}

const std::vector<int64_t>& CheckpointTensors::Shape(const std::string& name) const {
  const auto it = shapes.find(name);
  VT_CHECK(it != shapes.end(), "cua_s1 checkpoint: missing shape for '" + name + "'");
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
// torch uses. eps default 1e-5.
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
  for (int64_t i = 0; i < n; ++i) {
    const double normed = (static_cast<double>(x[i]) - mean) * inv;
    const double scaled = normed * static_cast<double>(w[static_cast<size_t>(i)]);
    const double shifted = b.empty() ? scaled : scaled + static_cast<double>(b[static_cast<size_t>(i)]);
    out[i] = static_cast<float>(shifted);
  }
}

// ReLU activation — the default for nn.TransformerEncoderLayer.
float Relu(float x) { return x > 0.0f ? x : 0.0f; }

// Multi-head self-attention with packed QKV (nn.MultiheadAttention).
// x: (seq, width), mask: (seq,) True=valid (False=padding).
// Output written to out (seq * width).
void SelfAttention(const float* x, const EncoderLayerWeights& lw,
                   int64_t seq, int64_t width, int64_t heads,
                   const uint8_t* mask, float* out) {
  const int64_t hd = width / heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));

  // QKV projection: in_proj_weight is (3*width, width), packed [Q; K; V].
  // Q = x @ in_proj_weight[:width].T + in_proj_bias[:width]
  // K = x @ in_proj_weight[width:2*width].T + in_proj_bias[width:2*width]
  // V = x @ in_proj_weight[2*width:].T + in_proj_bias[2*width:]
  std::vector<float> q(static_cast<size_t>(seq * width));
  std::vector<float> k(static_cast<size_t>(seq * width));
  std::vector<float> v(static_cast<size_t>(seq * width));

  for (int64_t t = 0; t < seq; ++t) {
    const float* xt = x + static_cast<size_t>(t * width);
    std::vector<float> qkv(static_cast<size_t>(3 * width));
    LinearRow(xt, lw.in_proj_weight, lw.in_proj_bias, width, 3 * width, qkv.data());
    std::copy(qkv.begin(), qkv.begin() + width, q.begin() + static_cast<size_t>(t * width));
    std::copy(qkv.begin() + width, qkv.begin() + 2 * width, k.begin() + static_cast<size_t>(t * width));
    std::copy(qkv.begin() + 2 * width, qkv.begin() + 3 * width, v.begin() + static_cast<size_t>(t * width));
  }

  // Attention: for each query position, head, compute scores over all keys.
  std::vector<float> attn_out(static_cast<size_t>(seq * width));
  std::vector<double> scores(static_cast<size_t>(seq));

  for (int64_t t = 0; t < seq; ++t) {
    for (int64_t hi = 0; hi < heads; ++hi) {
      const float* qh = q.data() + static_cast<size_t>(t * width + hi * hd);

      // Compute scores for this query head against all key positions.
      double max_score = -std::numeric_limits<double>::max();
      for (int64_t s = 0; s < seq; ++s) {
        if (!mask[s]) {
          scores[static_cast<size_t>(s)] = std::numeric_limits<float>::lowest();
          continue;
        }
        double dot = 0.0;
        const float* kh = k.data() + static_cast<size_t>(s * width + hi * hd);
        for (int64_t d = 0; d < hd; ++d) {
          dot += static_cast<double>(qh[d]) * static_cast<double>(kh[d]);
        }
        const double score = dot * scale;
        scores[static_cast<size_t>(s)] = score;
        if (score > max_score) max_score = score;
      }

      // Softmax.
      double sum_exp = 0.0;
      for (int64_t s = 0; s < seq; ++s) {
        const double val = scores[static_cast<size_t>(s)];
        if (val == std::numeric_limits<float>::lowest()) {
          scores[static_cast<size_t>(s)] = 0.0;
          continue;
        }
        const double e = std::exp(val - max_score);
        scores[static_cast<size_t>(s)] = e;
        sum_exp += e;
      }
      const double inv_sum = sum_exp > 0.0 ? 1.0 / sum_exp : 0.0;

      // Weighted sum of V.
      float* out_head = attn_out.data() + static_cast<size_t>(t * width + hi * hd);
      for (int64_t d = 0; d < hd; ++d) {
        double acc = 0.0;
        for (int64_t s = 0; s < seq; ++s) {
          acc += scores[static_cast<size_t>(s)] * inv_sum *
                 static_cast<double>(v[static_cast<size_t>(s * width + hi * hd + d)]);
        }
        out_head[static_cast<size_t>(d)] = static_cast<float>(acc);
      }
    }
  }

  // Output projection.
  for (int64_t t = 0; t < seq; ++t) {
    const float* at = attn_out.data() + static_cast<size_t>(t * width);
    float* out_t = out + static_cast<size_t>(t * width);
    LinearRow(at, lw.out_proj_weight, lw.out_proj_bias, width, width, out_t);
  }
}

// nn.TransformerEncoderLayer with norm_first=True, batch_first=True.
// Pre-norm: x = x + attn(norm1(x)); x = x + ff(norm2(x))
// ff(x) = linear2(relu(linear1(x)))
void EncoderLayerForward(const float* x, const EncoderLayerWeights& lw,
                         int64_t seq, int64_t width, int64_t heads,
                         const uint8_t* mask, float* out) {
  const double eps = 1e-5;
  const int64_t dim_ff = width * 4;

  std::vector<float> normed(static_cast<size_t>(seq * width));
  std::vector<float> attn_out(static_cast<size_t>(seq * width));
  std::vector<float> ff_inter(static_cast<size_t>(dim_ff));
  std::vector<float> ff_out(static_cast<size_t>(width));

  // Pre-norm + self-attention + residual.
  for (int64_t t = 0; t < seq; ++t) {
    LayerNormRow(x + static_cast<size_t>(t * width), lw.norm1_weight, lw.norm1_bias,
                 width, eps, normed.data() + static_cast<size_t>(t * width));
  }
  SelfAttention(normed.data(), lw, seq, width, heads, mask, attn_out.data());
  for (int64_t i = 0; i < seq * width; ++i) {
    out[i] = x[i] + attn_out[static_cast<size_t>(i)];
  }

  // Pre-norm + feed-forward + residual.
  for (int64_t t = 0; t < seq; ++t) {
    float* xt = out + static_cast<size_t>(t * width);
    LayerNormRow(xt, lw.norm2_weight, lw.norm2_bias,
                 width, eps, normed.data() + static_cast<size_t>(t * width));

    // linear1: (dim_ff, width)
    LinearRow(normed.data() + static_cast<size_t>(t * width), lw.linear1_weight,
              lw.linear1_bias, width, dim_ff, ff_inter.data());
    // ReLU
    for (int64_t i = 0; i < dim_ff; ++i) {
      ff_inter[static_cast<size_t>(i)] = Relu(ff_inter[static_cast<size_t>(i)]);
    }
    // linear2: (width, dim_ff)
    LinearRow(ff_inter.data(), lw.linear2_weight, lw.linear2_bias,
              dim_ff, width, ff_out.data());
    // Residual add
    for (int64_t d = 0; d < width; ++d) {
      xt[static_cast<size_t>(d)] += ff_out[static_cast<size_t>(d)];
    }
  }
}

// Run a stack of TransformerEncoderLayer.
void EncoderForward(const float* x, const std::vector<EncoderLayerWeights>& layers,
                    int64_t seq, int64_t width, int64_t heads,
                    const uint8_t* mask, float* out) {
  std::vector<float> cur(x, x + static_cast<size_t>(seq * width));
  std::vector<float> next(static_cast<size_t>(seq * width));
  for (size_t i = 0; i < layers.size(); ++i) {
    EncoderLayerForward(cur.data(), layers[i], seq, width, heads, mask, next.data());
    cur.swap(next);
  }
  std::copy(cur.begin(), cur.end(), out);
}

}  // namespace

Weights Load(const Params& params, const CheckpointTensors& tensors) {
  VT_CHECK(params.width > 0 && params.heads > 0,
           "cua_s1: width and heads must be positive");
  VT_CHECK(params.width % params.heads == 0,
           "cua_s1: width must divide by heads");

  Weights w;
  w.embedding_weight = tensors.Get("embedding.weight");
  w.position_weight = tensors.Get("position.weight");

  auto load_layer = [&](const std::string& p, EncoderLayerWeights& l) {
    l.in_proj_weight = tensors.Get(p + "self_attn.in_proj_weight");
    l.in_proj_bias = tensors.Get(p + "self_attn.in_proj_bias");
    l.out_proj_weight = tensors.Get(p + "self_attn.out_proj.weight");
    l.out_proj_bias = tensors.Get(p + "self_attn.out_proj.bias");
    l.linear1_weight = tensors.Get(p + "linear1.weight");
    l.linear1_bias = tensors.Get(p + "linear1.bias");
    l.linear2_weight = tensors.Get(p + "linear2.weight");
    l.linear2_bias = tensors.Get(p + "linear2.bias");
    l.norm1_weight = tensors.Get(p + "norm1.weight");
    l.norm1_bias = tensors.Get(p + "norm1.bias");
    l.norm2_weight = tensors.Get(p + "norm2.weight");
    l.norm2_bias = tensors.Get(p + "norm2.bias");
  };

  w.encoder_layers.resize(static_cast<size_t>(params.layers));
  for (int64_t i = 0; i < params.layers; ++i) {
    load_layer("encoder.layers." + std::to_string(i) + ".",
               w.encoder_layers[static_cast<size_t>(i)]);
  }

  w.option_encoder_layers.resize(1);
  load_layer("option_encoder.layers.0.", w.option_encoder_layers[0]);

  w.head.context_norm_weight = tensors.Get("head.context_norm.weight");
  w.head.context_norm_bias = tensors.Get("head.context_norm.bias");
  w.head.option_norm_weight = tensors.Get("head.option_norm.weight");
  w.head.option_norm_bias = tensors.Get("head.option_norm.bias");
  w.head.query_weight = tensors.Get("head.query.weight");
  w.head.key_weight = tensors.Get("head.key.weight");
  w.head.value_weight = tensors.Get("head.value.weight");

  return w;
}

std::vector<float> ForwardHost(
    const Params& params, const Weights& weights,
    const std::vector<int64_t>& context_ids,
    const std::vector<uint8_t>& context_mask,
    const std::vector<int64_t>& option_ids,
    const std::vector<uint8_t>& option_tok_mask,
    const std::vector<uint8_t>& option_mask) {
  const int64_t width = params.width;
  const int64_t rank = params.rank;
  const int64_t heads = params.heads;
  const int64_t ctx_len = static_cast<int64_t>(context_ids.size());
  const int64_t n_opt = static_cast<int64_t>(option_mask.size());
  const int64_t opt_len = n_opt > 0 ? static_cast<int64_t>(option_ids.size()) / n_opt : 0;
  const double eps = 1e-5;

  // ── Context encoding ────────────────────────────────────────────────────
  // embed: embedding(context_ids) + position(arange(ctx_len))
  std::vector<float> ctx_embed(static_cast<size_t>(ctx_len * width));
  for (int64_t t = 0; t < ctx_len; ++t) {
    const int64_t id = context_ids[static_cast<size_t>(t)];
    VT_CHECK(id >= 0 && id < 257, "cua_s1: context token id out of range");
    const float* emb = weights.embedding_weight.data() + static_cast<size_t>(id * width);
    const float* pos = weights.position_weight.data() + static_cast<size_t>(t * width);
    for (int64_t d = 0; d < width; ++d) {
      ctx_embed[static_cast<size_t>(t * width + d)] = emb[d] + pos[d];
    }
  }

  // Safe mask: force position 0 to True (avoid all-padding → NaN in softmax).
  std::vector<uint8_t> safe_ctx_mask = context_mask;
  if (!safe_ctx_mask.empty()) safe_ctx_mask[0] = 1;

  // Encode through context transformer encoder.
  // src_key_padding_mask=~safe_ctx_mask in Python: True = padding.
  // Our SelfAttention takes mask where 1 = valid, 0 = padding.
  std::vector<float> context(static_cast<size_t>(ctx_len * width));
  EncoderForward(ctx_embed.data(), weights.encoder_layers,
                 ctx_len, width, heads, safe_ctx_mask.data(), context.data());

  // ── Option encoding ──────────────────────────────────────────────────────
  // Each option is encoded independently (batch dim = n_opt, seq dim = opt_len).
  // Python: flat_ids = option_ids.reshape(n_opt, opt_len);
  //         hidden = option_encoder(embed(flat_ids), src_key_padding_mask=~safe_mask)
  std::vector<float> options(static_cast<size_t>(n_opt * width));
  for (int64_t n = 0; n < n_opt; ++n) {
    // Embed this option's tokens.
    std::vector<float> opt_embed(static_cast<size_t>(opt_len * width));
    for (int64_t t = 0; t < opt_len; ++t) {
      const int64_t id = option_ids[static_cast<size_t>(n * opt_len + t)];
      VT_CHECK(id >= 0 && id < 257, "cua_s1: option token id out of range");
      const float* emb = weights.embedding_weight.data() + static_cast<size_t>(id * width);
      const float* pos = weights.position_weight.data() + static_cast<size_t>(t * width);
      for (int64_t d = 0; d < width; ++d) {
        opt_embed[static_cast<size_t>(t * width + d)] = emb[d] + pos[d];
      }
    }

    // Safe mask: force position 0 to True.
    std::vector<uint8_t> opt_safe_mask(opt_len);
    for (int64_t t = 0; t < opt_len; ++t) {
      opt_safe_mask[static_cast<size_t>(t)] = option_tok_mask[static_cast<size_t>(n * opt_len + t)];
    }
    opt_safe_mask[0] = 1;

    // Encode through option transformer encoder (1 layer).
    std::vector<float> opt_hidden(static_cast<size_t>(opt_len * width));
    EncoderForward(opt_embed.data(), weights.option_encoder_layers,
                   opt_len, width, heads, opt_safe_mask.data(), opt_hidden.data());

    // Mean-pool with mask weights.
    // pooled[d] = sum(hidden[t, d] * mask[t]) / sum(mask[t])
    double weight_sum = 0.0;
    for (int64_t t = 0; t < opt_len; ++t) {
      if (option_tok_mask[static_cast<size_t>(n * opt_len + t)]) {
        weight_sum += 1.0;
      }
    }
    if (weight_sum < 1.0) weight_sum = 1.0;
    for (int64_t d = 0; d < width; ++d) {
      double acc = 0.0;
      for (int64_t t = 0; t < opt_len; ++t) {
        if (option_tok_mask[static_cast<size_t>(n * opt_len + t)]) {
          acc += static_cast<double>(opt_hidden[static_cast<size_t>(t * width + d)]);
        }
      }
      options[static_cast<size_t>(n * width + d)] = static_cast<float>(acc / weight_sum);
    }
  }

  // ── AttentionHead scoring ────────────────────────────────────────────────
  // context_norm(context.float()), option_norm(options.float())
  std::vector<float> ctx_normed(static_cast<size_t>(ctx_len * width));
  for (int64_t t = 0; t < ctx_len; ++t) {
    LayerNormRow(context.data() + static_cast<size_t>(t * width),
                 weights.head.context_norm_weight, weights.head.context_norm_bias,
                 width, eps, ctx_normed.data() + static_cast<size_t>(t * width));
  }
  std::vector<float> opt_normed(static_cast<size_t>(n_opt * width));
  for (int64_t n = 0; n < n_opt; ++n) {
    LayerNormRow(options.data() + static_cast<size_t>(n * width),
                 weights.head.option_norm_weight, weights.head.option_norm_bias,
                 width, eps, opt_normed.data() + static_cast<size_t>(n * width));
  }

  // query = query_linear(options)  (n_opt, rank)
  // key = key_linear(context)      (ctx_len, rank)
  // value = value_linear(context)  (ctx_len, rank)
  std::vector<float> query(static_cast<size_t>(n_opt * rank));
  for (int64_t n = 0; n < n_opt; ++n) {
    LinearRow(opt_normed.data() + static_cast<size_t>(n * width),
              weights.head.query_weight, {}, width, rank,
              query.data() + static_cast<size_t>(n * rank));
  }
  std::vector<float> key(static_cast<size_t>(ctx_len * rank));
  for (int64_t t = 0; t < ctx_len; ++t) {
    LinearRow(ctx_normed.data() + static_cast<size_t>(t * width),
              weights.head.key_weight, {}, width, rank,
              key.data() + static_cast<size_t>(t * rank));
  }
  std::vector<float> value(static_cast<size_t>(ctx_len * rank));
  for (int64_t t = 0; t < ctx_len; ++t) {
    LinearRow(ctx_normed.data() + static_cast<size_t>(t * width),
              weights.head.value_weight, {}, width, rank,
              value.data() + static_cast<size_t>(t * rank));
  }

  // scores = einsum("bnr,blr->bnl", query, key) / sqrt(rank)
  //         (n_opt, ctx_len)
  const double scale = 1.0 / std::sqrt(static_cast<double>(rank));
  std::vector<float> scores(static_cast<size_t>(n_opt * ctx_len));
  for (int64_t n = 0; n < n_opt; ++n) {
    const float* qn = query.data() + static_cast<size_t>(n * rank);
    for (int64_t l = 0; l < ctx_len; ++l) {
      const float* kl = key.data() + static_cast<size_t>(l * rank);
      double dot = 0.0;
      for (int64_t r = 0; r < rank; ++r) {
        dot += static_cast<double>(qn[r]) * static_cast<double>(kl[r]);
      }
      scores[static_cast<size_t>(n * ctx_len + l)] = static_cast<float>(dot * scale);
    }
  }

  // Mask: scores.masked_fill(~context_mask[:, None, :], finfo.min)
  for (int64_t n = 0; n < n_opt; ++n) {
    for (int64_t l = 0; l < ctx_len; ++l) {
      if (!context_mask[static_cast<size_t>(l)]) {
        scores[static_cast<size_t>(n * ctx_len + l)] = std::numeric_limits<float>::lowest();
      }
    }
  }

  // attended = einsum("bnl,blr->bnr", scores.softmax(-1), value)
  //           (n_opt, rank)
  std::vector<float> attended(static_cast<size_t>(n_opt * rank));
  for (int64_t n = 0; n < n_opt; ++n) {
    // Softmax over context dimension.
    double max_score = -std::numeric_limits<double>::max();
    for (int64_t l = 0; l < ctx_len; ++l) {
      const double s = scores[static_cast<size_t>(n * ctx_len + l)];
      if (s > max_score) max_score = s;
    }
    double sum_exp = 0.0;
    std::vector<double> probs(ctx_len);
    for (int64_t l = 0; l < ctx_len; ++l) {
      const double s = scores[static_cast<size_t>(n * ctx_len + l)];
      if (s == std::numeric_limits<float>::lowest()) {
        probs[static_cast<size_t>(l)] = 0.0;
      } else {
        const double e = std::exp(s - max_score);
        probs[static_cast<size_t>(l)] = e;
        sum_exp += e;
      }
    }
    const double inv_sum = sum_exp > 0.0 ? 1.0 / sum_exp : 0.0;

    for (int64_t r = 0; r < rank; ++r) {
      double acc = 0.0;
      for (int64_t l = 0; l < ctx_len; ++l) {
        acc += probs[static_cast<size_t>(l)] * inv_sum *
               static_cast<double>(value[static_cast<size_t>(l * rank + r)]);
      }
      attended[static_cast<size_t>(n * rank + r)] = static_cast<float>(acc);
    }
  }

  // logits = (query * attended).sum(-1) / sqrt(rank)
  //         (n_opt,)
  std::vector<float> logits(n_opt);
  for (int64_t n = 0; n < n_opt; ++n) {
    const float* qn = query.data() + static_cast<size_t>(n * rank);
    const float* an = attended.data() + static_cast<size_t>(n * rank);
    double dot = 0.0;
    for (int64_t r = 0; r < rank; ++r) {
      dot += static_cast<double>(qn[r]) * static_cast<double>(an[r]);
    }
    logits[static_cast<size_t>(n)] = static_cast<float>(dot * scale);
  }

  // Mask: logits.masked_fill(~option_mask, finfo.min)
  for (int64_t n = 0; n < n_opt; ++n) {
    if (!option_mask[static_cast<size_t>(n)]) {
      logits[static_cast<size_t>(n)] = std::numeric_limits<float>::lowest();
    }
  }

  return logits;
}

}  // namespace cua_s1
}  // namespace vllm
