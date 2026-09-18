// DeBERTa v2 encoder host reference. See deberta_v2.h for the upstream anchors.
#include "vllm/model_executor/models/deberta_v2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace deberta_v2 {

void CheckpointTensors::Set(const std::string& name, std::vector<int64_t> shape,
                             std::vector<float> data) {
  int64_t count = 1;
  for (const int64_t dim : shape) count *= dim;
  VT_CHECK(static_cast<size_t>(count) == data.size(),
           "deberta_v2 checkpoint: element count does not match shape");
  shapes[name] = std::move(shape);
  values[name] = std::move(data);
}

const std::vector<float>& CheckpointTensors::Get(const std::string& name) const {
  const auto it = values.find(name);
  VT_CHECK(it != values.end(), "deberta_v2 checkpoint: missing tensor '" + name + "'");
  return it->second;
}

const std::vector<int64_t>& CheckpointTensors::Shape(const std::string& name) const {
  const auto it = shapes.find(name);
  VT_CHECK(it != shapes.end(), "deberta_v2 checkpoint: missing shape for '" + name + "'");
  return it->second;
}

namespace {

// y = W x + b, with W stored [out, in] (nn.Linear orientation) and x a single
// row of `in` values.
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
// torch uses.
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
    out[i] = static_cast<float>(normed * static_cast<double>(w[static_cast<size_t>(i)]) +
                                static_cast<double>(b[static_cast<size_t>(i)]));
  }
}

// Exact GELU (erf-based), matching ACT2FN["gelu"] in HF transformers.
// gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
double Gelu(double x) {
  constexpr double kSqrt1Over2 = 0.70710678118654752440;  // 1/sqrt(2)
  return x * 0.5 * (1.0 + std::erf(x * kSqrt1Over2));
}

// build_relative_position (modeling_deberta_v2.py) combined with
// make_log_bucket_position. Maps relative positions to log-scale buckets.
// Returns the bucketed relative position matrix [seq, seq].
std::vector<int64_t> BuildRelativePosition(int64_t query_size, int64_t key_size,
                                            int64_t bucket_size, int64_t max_position) {
  std::vector<int64_t> rel_pos(static_cast<size_t>(query_size * key_size));
  for (int64_t i = 0; i < query_size; ++i) {
    for (int64_t j = 0; j < key_size; ++j) {
      rel_pos[static_cast<size_t>(i * key_size + j)] = i - j;
    }
  }
  if (bucket_size > 0 && max_position > 0) {
    // Apply log bucketing
    const int64_t mid = bucket_size / 2;
    const double log_mid = std::log(static_cast<double>(max_position - 1) / static_cast<double>(mid));
    for (int64_t i = 0; i < query_size; ++i) {
      for (int64_t j = 0; j < key_size; ++j) {
        int64_t rel = rel_pos[static_cast<size_t>(i * key_size + j)];
        int64_t sign = (rel > 0) - (rel < 0);
        int64_t abs_pos = std::abs(rel);
        if (abs_pos < mid) {
          // keep as-is
        } else {
          double log_pos = std::ceil(
              std::log(static_cast<double>(abs_pos) / static_cast<double>(mid)) /
              log_mid * static_cast<double>(mid - 1));
          rel_pos[static_cast<size_t>(i * key_size + j)] =
              (static_cast<int64_t>(log_pos) + mid) * sign;
        }
      }
    }
  }
  return rel_pos;
}

}  // namespace

Weights Load(const Params& params, const CheckpointTensors& tensors) {
  VT_CHECK(params.hidden_size > 0 && params.num_attention_heads > 0,
           "deberta_v2: hidden_size and num_attention_heads must be positive");
  VT_CHECK(params.hidden_size % params.num_attention_heads == 0,
           "deberta_v2: hidden_size must divide by num_attention_heads");

  Weights w;
  w.word_embeddings = tensors.Get("encoder.embeddings.word_embeddings.weight");
  w.emb_ln_weight = tensors.Get("encoder.embeddings.LayerNorm.weight");
  w.emb_ln_bias = tensors.Get("encoder.embeddings.LayerNorm.bias");
  w.rel_embeddings = tensors.Get("encoder.encoder.rel_embeddings.weight");
  if (params.norm_rel_ebd) {
    w.rel_ln_weight = tensors.Get("encoder.encoder.LayerNorm.weight");
    w.rel_ln_bias = tensors.Get("encoder.encoder.LayerNorm.bias");
  }

  w.layers.resize(static_cast<size_t>(params.num_hidden_layers));
  for (int64_t i = 0; i < params.num_hidden_layers; ++i) {
    const std::string p = "encoder.encoder.layer." + std::to_string(i) + ".";
    LayerWeights& l = w.layers[static_cast<size_t>(i)];
    l.query_weight = tensors.Get(p + "attention.self.query_proj.weight");
    l.query_bias = tensors.Get(p + "attention.self.query_proj.bias");
    l.key_weight = tensors.Get(p + "attention.self.key_proj.weight");
    l.key_bias = tensors.Get(p + "attention.self.key_proj.bias");
    l.value_weight = tensors.Get(p + "attention.self.value_proj.weight");
    l.value_bias = tensors.Get(p + "attention.self.value_proj.bias");
    l.attn_dense_weight = tensors.Get(p + "attention.output.dense.weight");
    l.attn_dense_bias = tensors.Get(p + "attention.output.dense.bias");
    l.attn_ln_weight = tensors.Get(p + "attention.output.LayerNorm.weight");
    l.attn_ln_bias = tensors.Get(p + "attention.output.LayerNorm.bias");
    l.inter_weight = tensors.Get(p + "intermediate.dense.weight");
    l.inter_bias = tensors.Get(p + "intermediate.dense.bias");
    l.out_dense_weight = tensors.Get(p + "output.dense.weight");
    l.out_dense_bias = tensors.Get(p + "output.dense.bias");
    l.out_ln_weight = tensors.Get(p + "output.LayerNorm.weight");
    l.out_ln_bias = tensors.Get(p + "output.LayerNorm.bias");
  }
  return w;
}

std::vector<float> ForwardHost(const Params& params, const Weights& weights,
                               const std::vector<int64_t>& input_ids) {
  const int64_t h = params.hidden_size;
  const int64_t seq = static_cast<int64_t>(input_ids.size());
  const int64_t heads = params.num_attention_heads;
  const int64_t head_dim = params.head_dim();
  const int64_t inter = params.intermediate_size;
  const int64_t span = params.att_span();
  const int64_t pos_ebd = params.pos_ebd_size();
  const int64_t sf = params.scale_factor();
  const double scale = std::sqrt(static_cast<double>(head_dim * sf));

  // ── Embeddings ──────────────────────────────────────────────────────────
  // DebertaV2Embeddings.forward: word_embeddings -> LayerNorm.
  // No position_embeddings (position_biased_input=false),
  // no token_type_embeddings (type_vocab_size=0).
  std::vector<float> x(static_cast<size_t>(seq * h));
  for (int64_t t = 0; t < seq; ++t) {
    const int64_t id = input_ids[static_cast<size_t>(t)];
    VT_CHECK(id >= 0 && id < params.vocab_size, "deberta_v2: input id out of range");
    const float* emb = weights.word_embeddings.data() + static_cast<size_t>(id * h);
    LayerNormRow(emb, weights.emb_ln_weight, weights.emb_ln_bias, h,
                 params.layer_norm_eps, &x[static_cast<size_t>(t * h)]);
  }

  // ── Relative embeddings ──────────────────────────────────────────────────
  // DebertaV2Encoder.get_rel_embedding: rel_embeddings -> LayerNorm (norm_rel_ebd).
  std::vector<float> rel_emb(static_cast<size_t>(pos_ebd * h));
  for (int64_t i = 0; i < pos_ebd; ++i) {
    const float* src = weights.rel_embeddings.data() + static_cast<size_t>(i * h);
    float* dst = rel_emb.data() + static_cast<size_t>(i * h);
    if (params.norm_rel_ebd) {
      LayerNormRow(src, weights.rel_ln_weight, weights.rel_ln_bias, h,
                   params.layer_norm_eps, dst);
    } else {
      std::copy(src, src + h, dst);
    }
  }

  // ── Relative position buckets ───────────────────────────────────────────
  // build_relative_position -> make_log_bucket_position.
  std::vector<int64_t> rel_pos = BuildRelativePosition(seq, seq, params.position_buckets,
                                                        params.max_rel_pos());
  // c2p_pos[i][j] = clamp(rel_pos[i][j] + span, 0, span*2-1)
  // p2c_pos[i][j] = clamp(-rel_pos[i][j] + span, 0, span*2-1)
  std::vector<int64_t> c2p_pos(static_cast<size_t>(seq * seq));
  std::vector<int64_t> p2c_pos(static_cast<size_t>(seq * seq));
  for (int64_t i = 0; i < seq; ++i) {
    for (int64_t j = 0; j < seq; ++j) {
      int64_t rp = rel_pos[static_cast<size_t>(i * seq + j)];
      c2p_pos[static_cast<size_t>(i * seq + j)] = std::clamp<int64_t>(rp + span, 0, span * 2 - 1);
      p2c_pos[static_cast<size_t>(i * seq + j)] = std::clamp<int64_t>(-rp + span, 0, span * 2 - 1);
    }
  }

  // ── Position projection buffers (computed per layer, see below) ──────────
  // share_att_key=true: each layer applies its own query_proj/key_proj to the
  // LayerNorm'd rel_embeddings to get pos_query and pos_key.
  std::vector<float> pos_q(static_cast<size_t>(pos_ebd * h));
  std::vector<float> pos_k(static_cast<size_t>(pos_ebd * h));

  // Buffers reused across layers
  std::vector<float> normed(static_cast<size_t>(seq * h));
  std::vector<float> q_buf(static_cast<size_t>(seq * h));
  std::vector<float> k_buf(static_cast<size_t>(seq * h));
  std::vector<float> v_buf(static_cast<size_t>(seq * h));
  std::vector<float> attn_out(static_cast<size_t>(seq * h));
  std::vector<float> proj(static_cast<size_t>(seq * h));
  std::vector<float> inner_buf(static_cast<size_t>(seq * inter));

  for (const LayerWeights& l : weights.layers) {
    // ── Self-attention ──────────────────────────────────────────────────
    // Q, K, V projections
    for (int64_t t = 0; t < seq; ++t) {
      LinearRow(&x[static_cast<size_t>(t * h)], l.query_weight, l.query_bias, h, h,
                &q_buf[static_cast<size_t>(t * h)]);
      LinearRow(&x[static_cast<size_t>(t * h)], l.key_weight, l.key_bias, h, h,
                &k_buf[static_cast<size_t>(t * h)]);
      LinearRow(&x[static_cast<size_t>(t * h)], l.value_weight, l.value_bias, h, h,
                &v_buf[static_cast<size_t>(t * h)]);
    }

    // Position projections (share_att_key=true: use this layer's query_proj/key_proj)
    for (int64_t i = 0; i < pos_ebd; ++i) {
      LinearRow(rel_emb.data() + static_cast<size_t>(i * h), l.query_weight, l.query_bias, h, h,
                &pos_q[static_cast<size_t>(i * h)]);
      LinearRow(rel_emb.data() + static_cast<size_t>(i * h), l.key_weight, l.key_bias, h, h,
                &pos_k[static_cast<size_t>(i * h)]);
    }

    // Disentangled attention per head
    for (int64_t head = 0; head < heads; ++head) {
      // Scores: [seq, seq]
      std::vector<double> scores(static_cast<size_t>(seq * seq));
      for (int64_t i = 0; i < seq; ++i) {
        const float* q = &q_buf[static_cast<size_t>(i * h + head * head_dim)];
        for (int64_t j = 0; j < seq; ++j) {
          const float* k = &k_buf[static_cast<size_t>(j * h + head * head_dim)];
          double dot = 0.0;
          for (int64_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(q[d]) * static_cast<double>(k[d]);
          }
          scores[static_cast<size_t>(i * seq + j)] = dot / scale;
        }
      }

      // c2p bias: query @ pos_key^T, gathered by c2p_pos
      if (params.use_c2p) {
        for (int64_t i = 0; i < seq; ++i) {
          const float* q = &q_buf[static_cast<size_t>(i * h + head * head_dim)];
          for (int64_t j = 0; j < seq; ++j) {
            int64_t idx = c2p_pos[static_cast<size_t>(i * seq + j)];
            const float* pk = &pos_k[static_cast<size_t>(idx * h + head * head_dim)];
            double dot = 0.0;
            for (int64_t d = 0; d < head_dim; ++d) {
              dot += static_cast<double>(q[d]) * static_cast<double>(pk[d]);
            }
            scores[static_cast<size_t>(i * seq + j)] += dot / scale;
          }
        }
      }

      // p2c bias: key @ pos_query^T, gathered by p2c_pos, then transposed
      // p2c_att_final[i][j] = p2c_att_raw[j][p2c_pos[j][i]]
      // p2c_att_raw[j][k] = dot(k[j], pos_query[k])
      if (params.use_p2c) {
        for (int64_t i = 0; i < seq; ++i) {
          for (int64_t j = 0; j < seq; ++j) {
            // p2c_pos is indexed [i][j] in the HF code, but after the transpose,
            // the gather index for position (i,j) in the transposed output is
            // p2c_pos[j][i] (from the untransposed [j][i] entry).
            // Actually: p2c_att_raw[h][j][k] = dot(k[h][j], pos_query[h][k])
            // gather: p2c_att_gathered[h][j][i] = p2c_att_raw[h][j][p2c_pos[j][i]]
            // transpose: p2c_att_final[h][i][j] = p2c_att_gathered[h][j][i]
            // So: p2c_att_final[h][i][j] = dot(k[h][j], pos_query[h][p2c_pos[j][i]])
            int64_t idx = p2c_pos[static_cast<size_t>(j * seq + i)];
            const float* kj = &k_buf[static_cast<size_t>(j * h + head * head_dim)];
            const float* pq = &pos_q[static_cast<size_t>(idx * h + head * head_dim)];
            double dot = 0.0;
            for (int64_t d = 0; d < head_dim; ++d) {
              dot += static_cast<double>(kj[d]) * static_cast<double>(pq[d]);
            }
            scores[static_cast<size_t>(i * seq + j)] += dot / scale;
          }
        }
      }

      // Softmax over j (bidirectional, no mask for single sequence)
      for (int64_t i = 0; i < seq; ++i) {
        double max_score = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < seq; ++j) {
          max_score = std::max(max_score, scores[static_cast<size_t>(i * seq + j)]);
        }
        double denom = 0.0;
        std::vector<double> probs(static_cast<size_t>(seq));
        for (int64_t j = 0; j < seq; ++j) {
          probs[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(i * seq + j)] - max_score);
          denom += probs[static_cast<size_t>(j)];
        }
        // Weighted sum of V
        for (int64_t d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < seq; ++j) {
            const float* v = &v_buf[static_cast<size_t>(j * h + head * head_dim)];
            acc += probs[static_cast<size_t>(j)] * static_cast<double>(v[d]);
          }
          attn_out[static_cast<size_t>(i * h + head * head_dim + d)] =
              static_cast<float>(acc / denom);
        }
      }
    }

    // ── Attention output (DebertaV2SelfOutput) ──────────────────────────
    // dense -> LayerNorm(residual)
    for (int64_t t = 0; t < seq; ++t) {
      LinearRow(&attn_out[static_cast<size_t>(t * h)], l.attn_dense_weight, l.attn_dense_bias, h, h,
                &proj[static_cast<size_t>(t * h)]);
    }
    for (int64_t t = 0; t < seq; ++t) {
      float* x_row = &x[static_cast<size_t>(t * h)];
      for (int64_t d = 0; d < h; ++d) x_row[d] += proj[static_cast<size_t>(t * h + d)];
      LayerNormRow(x_row, l.attn_ln_weight, l.attn_ln_bias, h,
                   params.layer_norm_eps, &normed[static_cast<size_t>(t * h)]);
    }

    // ── Intermediate (dense -> gelu) ────────────────────────────────────
    for (int64_t t = 0; t < seq; ++t) {
      LinearRow(&normed[static_cast<size_t>(t * h)], l.inter_weight, l.inter_bias, h, inter,
                &inner_buf[static_cast<size_t>(t * inter)]);
    }
    for (float& v : inner_buf) v = static_cast<float>(Gelu(static_cast<double>(v)));

    // ── Output (dense -> LayerNorm + residual) ──────────────────────────
    for (int64_t t = 0; t < seq; ++t) {
      LinearRow(&inner_buf[static_cast<size_t>(t * inter)], l.out_dense_weight, l.out_dense_bias,
                inter, h, &proj[static_cast<size_t>(t * h)]);
    }
    // Residual is attention_output (normed), not the pre-LayerNorm x.
    for (int64_t t = 0; t < seq; ++t) {
      float* x_row = &x[static_cast<size_t>(t * h)];
      for (int64_t d = 0; d < h; ++d)
        x_row[d] = normed[static_cast<size_t>(t * h + d)] + proj[static_cast<size_t>(t * h + d)];
      LayerNormRow(x_row, l.out_ln_weight, l.out_ln_bias, h,
                   params.layer_norm_eps, &x[static_cast<size_t>(t * h)]);
    }
  }

  return x;
}

}  // namespace deberta_v2
}  // namespace vllm
