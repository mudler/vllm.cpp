// Laya decision head host reference. See laya.h for the upstream anchors.
#include "vllm/model_executor/models/laya.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace laya {

void CheckpointTensors::Set(const std::string& name, std::vector<int64_t> shape,
                              std::vector<float> data) {
  int64_t count = 1;
  for (const int64_t dim : shape) count *= dim;
  VT_CHECK(static_cast<size_t>(count) == data.size(),
           "laya checkpoint: element count does not match shape");
  shapes[name] = std::move(shape);
  values[name] = std::move(data);
}

const std::vector<float>& CheckpointTensors::Get(const std::string& name) const {
  const auto it = values.find(name);
  VT_CHECK(it != values.end(), "laya checkpoint: missing tensor '" + name + "'");
  return it->second;
}

const std::vector<int64_t>& CheckpointTensors::Shape(const std::string& name) const {
  const auto it = shapes.find(name);
  VT_CHECK(it != shapes.end(), "laya checkpoint: missing shape for '" + name + "'");
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
// torch uses. When `b` is empty, no bias (affine shift) is applied.
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

// Exact GELU (erf-based), matching nn.GELU() default.
double Gelu(double x) {
  constexpr double kSqrt1Over2 = 0.70710678118654752440;  // 1/sqrt(2)
  return x * 0.5 * (1.0 + std::erf(x * kSqrt1Over2));
}

double Relu(double x) { return x > 0.0 ? x : 0.0; }

// One standard nn.TransformerEncoderLayer forward (norm_first=True, ReLU).
//
//   h = h + self_attn(norm1(h), key_padding_mask=~mask)
//   h = h + linear2(relu(linear1(norm2(h))))
//
// The attention is standard multi-head (no RoPE, no sliding window).
// key_padding_mask: attention_mask[j] == 0 → position j is masked.
void HeadLayerForward(const Params& params, const HeadLayerWeights& lw,
                      const std::vector<int64_t>& attention_mask,
                      std::vector<float>& h, int64_t seq) {
  const int64_t d = params.hidden_size;
  if (seq <= 0 || d <= 0) return;
  const int64_t heads = params.num_heads;
  const int64_t hd = params.resolved_head_dim();
  const double eps = params.layer_norm_eps;

  std::vector<float> normed(static_cast<size_t>(seq * d));
  std::vector<float> qkv(static_cast<size_t>(3 * d));
  std::vector<float> q(static_cast<size_t>(seq * d));
  std::vector<float> k(static_cast<size_t>(seq * d));
  std::vector<float> v(static_cast<size_t>(seq * d));
  std::vector<float> attn_out(static_cast<size_t>(d));
  std::vector<float> proj(static_cast<size_t>(d));
  std::vector<float> scores(static_cast<size_t>(seq));
  std::vector<float> ff_inter(static_cast<size_t>(params.dim_ff));

  // ── Attention pre-norm + QKV ──────────────────────────────────────────
  for (int64_t t = 0; t < seq; ++t) {
    const float* ht = h.data() + static_cast<size_t>(t * d);
    LayerNormRow(ht, lw.norm1_weight, lw.norm1_bias, d, eps,
                 &normed[static_cast<size_t>(t * d)]);
    LinearRow(&normed[static_cast<size_t>(t * d)], lw.in_proj_weight, lw.in_proj_bias,
              d, 3 * d, qkv.data());
    std::copy(qkv.begin(), qkv.begin() + d, q.begin() + static_cast<size_t>(t * d));
    std::copy(qkv.begin() + d, qkv.begin() + 2 * d, k.begin() + static_cast<size_t>(t * d));
    std::copy(qkv.begin() + 2 * d, qkv.begin() + 3 * d, v.begin() + static_cast<size_t>(t * d));
  }

  // ── Multi-head attention with key_padding_mask ─────────────────────────
  const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
  for (int64_t t = 0; t < seq; ++t) {
    for (int64_t hd_idx = 0; hd_idx < heads; ++hd_idx) {
      const float* qh = q.data() + static_cast<size_t>(t * d + hd_idx * hd);

      double max_score = -std::numeric_limits<double>::max();
      for (int64_t s = 0; s < seq; ++s) {
        // key_padding_mask: attention_mask == 0 → masked
        if (attention_mask[static_cast<size_t>(s)] == 0) {
          scores[static_cast<size_t>(s)] = -1e30f;
          continue;
        }
        double dot = 0.0;
        const float* kh = k.data() + static_cast<size_t>(s * d + hd_idx * hd);
        for (int64_t i = 0; i < hd; ++i) {
          dot += static_cast<double>(qh[i]) * static_cast<double>(kh[i]);
        }
        const double score = dot * attn_scale;
        scores[static_cast<size_t>(s)] = static_cast<float>(score);
        if (score > max_score) max_score = score;
      }

      // Softmax over keys.
      double sum_exp = 0.0;
      for (int64_t s = 0; s < seq; ++s) {
        const double e = std::exp(static_cast<double>(scores[static_cast<size_t>(s)]) - max_score);
        scores[static_cast<size_t>(s)] = static_cast<float>(e);
        sum_exp += e;
      }
      const double inv_sum = 1.0 / sum_exp;

      // Weighted sum of V for this head.
      float* out_head = attn_out.data() + static_cast<size_t>(hd_idx * hd);
      for (int64_t i = 0; i < hd; ++i) {
        double acc = 0.0;
        for (int64_t s = 0; s < seq; ++s) {
          acc += static_cast<double>(scores[static_cast<size_t>(s)]) * inv_sum *
                 static_cast<double>(v[static_cast<size_t>(s * d + hd_idx * hd + i)]);
        }
        out_head[static_cast<size_t>(i)] = static_cast<float>(acc);
      }
    }

    // Output projection → residual add
    LinearRow(attn_out.data(), lw.out_proj_weight, lw.out_proj_bias, d, d, proj.data());
    for (int64_t i = 0; i < d; ++i) {
      h[static_cast<size_t>(t * d + i)] += proj[static_cast<size_t>(i)];
    }
  }

  // ── FFN pre-norm + ReLU ────────────────────────────────────────────────
  for (int64_t t = 0; t < seq; ++t) {
    const float* ht = h.data() + static_cast<size_t>(t * d);
    LayerNormRow(ht, lw.norm2_weight, lw.norm2_bias, d, eps,
                 &normed[static_cast<size_t>(t * d)]);

    // linear1 → ReLU → linear2
    LinearRow(&normed[static_cast<size_t>(t * d)], lw.linear1_weight, lw.linear1_bias,
              d, params.dim_ff, ff_inter.data());
    for (int64_t i = 0; i < params.dim_ff; ++i) {
      ff_inter[static_cast<size_t>(i)] = static_cast<float>(Relu(static_cast<double>(ff_inter[static_cast<size_t>(i)])));
    }
    LinearRow(ff_inter.data(), lw.linear2_weight, lw.linear2_bias,
              params.dim_ff, d, proj.data());
    for (int64_t i = 0; i < d; ++i) {
      h[static_cast<size_t>(t * d + i)] += proj[static_cast<size_t>(i)];
    }
  }
}

}  // namespace

Weights Load(const Params& params, const CheckpointTensors& tensors) {
  VT_CHECK(params.hidden_size > 0 && params.num_heads > 0,
           "laya: hidden_size and num_heads must be positive");
  VT_CHECK(params.hidden_size % params.num_heads == 0,
           "laya: hidden_size must divide by num_heads");

  Weights w;
  w.type_emb = tensors.Get("type_emb.weight");

  w.head_layers.resize(static_cast<size_t>(params.head_layers));
  for (int64_t i = 0; i < params.head_layers; ++i) {
    const std::string p = "head.layers." + std::to_string(i) + ".";
    HeadLayerWeights& l = w.head_layers[static_cast<size_t>(i)];
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
  }

  w.scorer.norm_weight = tensors.Get("scorer.0.weight");
  w.scorer.norm_bias = tensors.Get("scorer.0.bias");
  w.scorer.linear1_weight = tensors.Get("scorer.1.weight");
  w.scorer.linear1_bias = tensors.Get("scorer.1.bias");
  w.scorer.linear2_weight = tensors.Get("scorer.3.weight");
  w.scorer.linear2_bias = tensors.Get("scorer.3.bias");

  w.act_head.linear1_weight = tensors.Get("act_head.0.weight");
  w.act_head.linear1_bias = tensors.Get("act_head.0.bias");
  w.act_head.linear2_weight = tensors.Get("act_head.2.weight");
  w.act_head.linear2_bias = tensors.Get("act_head.2.bias");

  w.temperature = tensors.Get("temperature");

  return w;
}

ForwardOutput ForwardHost(
    const Params& params, const Weights& weights,
    const std::vector<float>& hidden_states,
    const std::vector<int64_t>& attention_mask,
    const std::vector<int64_t>& marker_pos,
    const std::vector<int64_t>& marker_mask,
    int64_t qtype) {
  const int64_t d = params.hidden_size;
  const int64_t seq = static_cast<int64_t>(attention_mask.size());
  const int64_t k_max = static_cast<int64_t>(marker_pos.size());
  const double eps = params.layer_norm_eps;

  VT_CHECK(static_cast<int64_t>(hidden_states.size()) == seq * d,
           "laya: hidden_states size mismatch");
  VT_CHECK(static_cast<int64_t>(marker_mask.size()) == k_max,
           "laya: marker_mask size mismatch");
  VT_CHECK(qtype >= 0 && qtype < 3, "laya: qtype out of range");

  // ── 1. type_emb addition ───────────────────────────────────────────────
  // h = h + type_emb(qtype)[:, None, :]  (broadcast over all positions)
  std::vector<float> h = hidden_states;
  const float* type_vec = weights.type_emb.data() + static_cast<size_t>(qtype * d);
  for (int64_t t = 0; t < seq; ++t) {
    for (int64_t i = 0; i < d; ++i) {
      h[static_cast<size_t>(t * d + i)] += type_vec[static_cast<size_t>(i)];
    }
  }

  // ── 2. Head layers (standard pre-norm transformer encoder) ─────────────
  for (int64_t layer = 0; layer < params.head_layers; ++layer) {
    HeadLayerForward(params, weights.head_layers[static_cast<size_t>(layer)],
                     attention_mask, h, seq);
  }

  // ── 3. Gather marker positions ─────────────────────────────────────────
  // idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
  // m = torch.gather(h, 1, idx)  → [k_max, d]
  std::vector<float> m(static_cast<size_t>(k_max * d));
  for (int64_t k = 0; k < k_max; ++k) {
    int64_t pos = marker_pos[static_cast<size_t>(k)];
    if (pos < 0) pos = 0;  // clamp(min=0)
    const float* src = h.data() + static_cast<size_t>(pos * d);
    std::copy(src, src + d, m.begin() + static_cast<size_t>(k * d));
  }

  // ── 4. Scorer: LayerNorm → Linear → GELU → Linear → squeeze ──────────
  // logits = scorer(m).squeeze(-1)  → [k_max]
  std::vector<float> normed(static_cast<size_t>(d));
  std::vector<float> inter(static_cast<size_t>(d));
  std::vector<float> logits(static_cast<size_t>(k_max));
  for (int64_t k = 0; k < k_max; ++k) {
    const float* mk = m.data() + static_cast<size_t>(k * d);
    LayerNormRow(mk, weights.scorer.norm_weight, weights.scorer.norm_bias, d, eps, normed.data());
    LinearRow(normed.data(), weights.scorer.linear1_weight, weights.scorer.linear1_bias,
              d, d, inter.data());
    for (int64_t i = 0; i < d; ++i) {
      inter[static_cast<size_t>(i)] = static_cast<float>(Gelu(static_cast<double>(inter[static_cast<size_t>(i)])));
    }
    float logit;
    LinearRow(inter.data(), weights.scorer.linear2_weight, weights.scorer.linear2_bias,
              d, 1, &logit);
    logits[static_cast<size_t>(k)] = logit;
  }

  // ── 5. Mask invalid markers ────────────────────────────────────────────
  // logits = logits.masked_fill(~marker_mask, -1e4)
  for (int64_t k = 0; k < k_max; ++k) {
    if (marker_mask[static_cast<size_t>(k)] == 0) {
      logits[static_cast<size_t>(k)] = -1e4f;
    }
  }

  // ── 6. Confidence features from softmax(logits) ───────────────────────
  // p = torch.softmax(logits.detach(), -1)  → [k_max]
  double max_logit = -std::numeric_limits<double>::max();
  for (int64_t k = 0; k < k_max; ++k) {
    if (logits[static_cast<size_t>(k)] > max_logit) max_logit = logits[static_cast<size_t>(k)];
  }
  std::vector<double> p(static_cast<size_t>(k_max));
  double sum_exp = 0.0;
  for (int64_t k = 0; k < k_max; ++k) {
    p[static_cast<size_t>(k)] = std::exp(static_cast<double>(logits[static_cast<size_t>(k)]) - max_logit);
    sum_exp += p[static_cast<size_t>(k)];
  }
  for (int64_t k = 0; k < k_max; ++k) {
    p[static_cast<size_t>(k)] /= sum_exp;
  }

  // k = marker_mask.sum(-1).clamp(min=2).float()
  int64_t k_count = 0;
  for (int64_t k = 0; k < k_max; ++k) {
    if (marker_mask[static_cast<size_t>(k)] != 0) ++k_count;
  }
  if (k_count < 2) k_count = 2;
  const double k_count_d = static_cast<double>(k_count);

  // ent = -(p * log(p.clamp_min(1e-9))).sum(-1) / log(k)
  double ent = 0.0;
  for (int64_t k = 0; k < k_max; ++k) {
    const double pk = p[static_cast<size_t>(k)];
    const double log_pk = std::log(std::max(pk, 1e-9));
    ent -= pk * log_pk;
  }
  ent /= std::log(k_count_d);

  // top2 = p.topk(2, -1).values  (descending)
  double top1 = -1.0, top2 = -1.0;
  for (int64_t k = 0; k < k_max; ++k) {
    const double pk = p[static_cast<size_t>(k)];
    if (pk > top1) {
      top2 = top1;
      top1 = pk;
    } else if (pk > top2) {
      top2 = pk;
    }
  }

  // feats = [top1, top1 - top2, ent, k / 255.0]
  double feats[4] = {top1, top1 - top2, ent, k_count_d / 255.0};

  // ── 7. CLS pool + act_head ─────────────────────────────────────────────
  // pooled = h[:, 0]  (CLS token)
  const float* pooled = h.data();  // position 0

  // act_head: Linear(d+4, act_hidden) → GELU → Linear(act_hidden, n_act)
  std::vector<float> act_in(static_cast<size_t>(d + 4));
  std::copy(pooled, pooled + d, act_in.data());
  for (int64_t i = 0; i < 4; ++i) {
    act_in[static_cast<size_t>(d + i)] = static_cast<float>(feats[i]);
  }

  std::vector<float> act_inter(static_cast<size_t>(params.act_hidden));
  LinearRow(act_in.data(), weights.act_head.linear1_weight, weights.act_head.linear1_bias,
            d + 4, params.act_hidden, act_inter.data());
  for (int64_t i = 0; i < params.act_hidden; ++i) {
    act_inter[static_cast<size_t>(i)] = static_cast<float>(Gelu(static_cast<double>(act_inter[static_cast<size_t>(i)])));
  }

  std::vector<float> act_logits(static_cast<size_t>(params.n_act));
  LinearRow(act_inter.data(), weights.act_head.linear2_weight, weights.act_head.linear2_bias,
            params.act_hidden, params.n_act, act_logits.data());

  ForwardOutput out;
  out.logits = std::move(logits);
  out.act_logits = std::move(act_logits);
  return out;
}

float TemperatureFor(
    int qtype, int k,
    const std::vector<float>& temperature,
    const std::map<std::string, float>& temperature_by_options) {
  static const char* kQTypeNames[] = {"choice", "score", "noul"};
  std::string bucket = kQTypeNames[qtype];
  bucket += ":";
  if (k <= 2) bucket += "2";
  else if (k <= 5) bucket += "3-5";
  else if (k <= 10) bucket += "6-10";
  else bucket += "11+";

  float temp = temperature[static_cast<size_t>(qtype)];
  auto it = temperature_by_options.find(bucket);
  if (it != temperature_by_options.end()) temp = it->second;
  return temp;
}

}  // namespace laya
}  // namespace vllm
