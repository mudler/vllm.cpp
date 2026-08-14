// GPT-2 backbone host reference. See gpt2.h for the upstream anchors.
#include "vllm/model_executor/models/gpt2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace gpt2 {

void CheckpointTensors::Set(const std::string& name, std::vector<int64_t> shape,
                            std::vector<float> data) {
  int64_t count = 1;
  for (const int64_t dim : shape) count *= dim;
  VT_CHECK(static_cast<size_t>(count) == data.size(),
           "gpt2 checkpoint: element count does not match shape");
  shapes[name] = std::move(shape);
  values[name] = std::move(data);
}

const std::vector<float>& CheckpointTensors::Get(const std::string& name) const {
  const auto it = values.find(name);
  VT_CHECK(it != values.end(), "gpt2 checkpoint: missing tensor '" + name + "'");
  return it->second;
}

const std::vector<int64_t>& CheckpointTensors::Shape(const std::string& name) const {
  const auto it = shapes.find(name);
  VT_CHECK(it != shapes.end(), "gpt2 checkpoint: missing shape for '" + name + "'");
  return it->second;
}

namespace {

// gpt2.py:242-254 `_transpose_conv1d`. HF's GPT-2 stores c_attn/c_proj/c_fc as
// Conv1D, whose 2D weight is [in, out]; the matmul below wants [out, in]. The
// transpose happens once, at load.
std::vector<float> TransposeConv1d(const std::vector<float>& w, int64_t in_dim, int64_t out_dim) {
  VT_CHECK(w.size() == static_cast<size_t>(in_dim * out_dim),
           "gpt2: Conv1D weight does not match [in, out]");
  std::vector<float> out(w.size());
  for (int64_t i = 0; i < in_dim; ++i) {
    for (int64_t o = 0; o < out_dim; ++o) {
      out[static_cast<size_t>(o * in_dim + i)] = w[static_cast<size_t>(i * out_dim + o)];
    }
  }
  return out;
}

// y = W x + b, with W stored [out, in] and x a single row of `in` values.
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

// GPT-2's `activation_function` is gelu_new: the tanh approximation, NOT the
// exact-erf GELU. They differ by ~1e-3, which is far above this gate's
// tolerance, so picking the wrong one is caught rather than absorbed.
double GeluNew(double x) {
  constexpr double kSqrt2OverPi = 0.7978845608028654;  // sqrt(2/pi)
  const double inner = kSqrt2OverPi * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + std::tanh(inner));
}

}  // namespace

Weights Load(const Params& params, const CheckpointTensors& tensors) {
  VT_CHECK(params.hidden_size > 0 && params.num_attention_heads > 0,
           "gpt2: hidden_size and num_attention_heads must be positive");
  VT_CHECK(params.hidden_size % params.num_attention_heads == 0,
           "gpt2: hidden_size must divide by num_attention_heads");

  Weights w;
  w.wte = tensors.Get("wte.weight");
  w.wpe = tensors.Get("wpe.weight");
  w.ln_f_weight = tensors.Get("ln_f.weight");
  w.ln_f_bias = tensors.Get("ln_f.bias");

  const int64_t h = params.hidden_size;
  const int64_t inner = params.inner_size;
  w.layers.resize(static_cast<size_t>(params.num_hidden_layers));
  for (int64_t i = 0; i < params.num_hidden_layers; ++i) {
    const std::string b = "h." + std::to_string(i) + ".";
    LayerWeights& l = w.layers[static_cast<size_t>(i)];
    l.ln_1_weight = tensors.Get(b + "ln_1.weight");
    l.ln_1_bias = tensors.Get(b + "ln_1.bias");
    l.ln_2_weight = tensors.Get(b + "ln_2.weight");
    l.ln_2_bias = tensors.Get(b + "ln_2.bias");
    l.c_attn_weight = TransposeConv1d(tensors.Get(b + "attn.c_attn.weight"), h, 3 * h);
    l.c_attn_bias = tensors.Get(b + "attn.c_attn.bias");
    l.c_proj_weight = TransposeConv1d(tensors.Get(b + "attn.c_proj.weight"), h, h);
    l.c_proj_bias = tensors.Get(b + "attn.c_proj.bias");
    l.c_fc_weight = TransposeConv1d(tensors.Get(b + "mlp.c_fc.weight"), h, inner);
    l.c_fc_bias = tensors.Get(b + "mlp.c_fc.bias");
    l.mlp_c_proj_weight = TransposeConv1d(tensors.Get(b + "mlp.c_proj.weight"), inner, h);
    l.mlp_c_proj_bias = tensors.Get(b + "mlp.c_proj.bias");
  }
  return w;
}

std::vector<float> ForwardHost(const Params& params, const Weights& weights,
                               const std::vector<int64_t>& input_ids,
                               const std::vector<int64_t>& positions) {
  VT_CHECK(input_ids.size() == positions.size(), "gpt2: ids and positions must be the same length");
  const int64_t seq = static_cast<int64_t>(input_ids.size());
  const int64_t h = params.hidden_size;
  const int64_t heads = params.num_attention_heads;
  const int64_t head_dim = params.head_dim();
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));  // gpt2.py:76

  // gpt2.py:225-229 — inputs_embeds + position_embeds.
  std::vector<float> x(static_cast<size_t>(seq * h));
  for (int64_t t = 0; t < seq; ++t) {
    const int64_t id = input_ids[static_cast<size_t>(t)];
    const int64_t pos = positions[static_cast<size_t>(t)];
    VT_CHECK(id >= 0 && id < params.vocab_size, "gpt2: input id out of range");
    VT_CHECK(pos >= 0 && pos < params.max_position_embeddings, "gpt2: position out of range");
    for (int64_t j = 0; j < h; ++j) {
      x[static_cast<size_t>(t * h + j)] =
          weights.wte[static_cast<size_t>(id * h + j)] + weights.wpe[static_cast<size_t>(pos * h + j)];
    }
  }

  std::vector<float> normed(static_cast<size_t>(seq * h));
  std::vector<float> qkv(static_cast<size_t>(seq * 3 * h));
  std::vector<float> attn_out(static_cast<size_t>(seq * h));
  std::vector<float> proj(static_cast<size_t>(seq * h));
  std::vector<float> inner_buf(static_cast<size_t>(seq * params.inner_size));

  for (const LayerWeights& l : weights.layers) {
    // gpt2.py:170-174 — ln_1 -> attn -> residual.
    for (int64_t t = 0; t < seq; ++t) {
      LayerNormRow(&x[static_cast<size_t>(t * h)], l.ln_1_weight, l.ln_1_bias, h,
                   params.layer_norm_eps, &normed[static_cast<size_t>(t * h)]);
    }
    for (int64_t t = 0; t < seq; ++t) {
      LinearRow(&normed[static_cast<size_t>(t * h)], l.c_attn_weight, l.c_attn_bias, h, 3 * h,
                &qkv[static_cast<size_t>(t * 3 * h)]);
    }
    // gpt2.py:107 — qkv.chunk(3, dim=-1): q | k | v, contiguous thirds.
    for (int64_t head = 0; head < heads; ++head) {
      for (int64_t t = 0; t < seq; ++t) {
        const float* q = &qkv[static_cast<size_t>(t * 3 * h + head * head_dim)];
        // Causal: key positions beyond t are masked out entirely (gpt2 is a
        // decoder), so the softmax runs over [0, t].
        std::vector<double> scores(static_cast<size_t>(t + 1));
        double max_score = -std::numeric_limits<double>::infinity();
        for (int64_t s = 0; s <= t; ++s) {
          const float* k = &qkv[static_cast<size_t>(s * 3 * h + h + head * head_dim)];
          double dot = 0.0;
          for (int64_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(q[d]) * static_cast<double>(k[d]);
          }
          dot *= scale;
          scores[static_cast<size_t>(s)] = dot;
          max_score = std::max(max_score, dot);
        }
        double denom = 0.0;
        for (double& s : scores) {
          s = std::exp(s - max_score);
          denom += s;
        }
        for (int64_t d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int64_t s = 0; s <= t; ++s) {
            const float* v = &qkv[static_cast<size_t>(s * 3 * h + 2 * h + head * head_dim)];
            acc += scores[static_cast<size_t>(s)] * static_cast<double>(v[d]);
          }
          attn_out[static_cast<size_t>(t * h + head * head_dim + d)] =
              static_cast<float>(acc / denom);
        }
      }
    }
    for (int64_t t = 0; t < seq; ++t) {
      LinearRow(&attn_out[static_cast<size_t>(t * h)], l.c_proj_weight, l.c_proj_bias, h, h,
                &proj[static_cast<size_t>(t * h)]);
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

    // gpt2.py:176-180 — ln_2 -> mlp -> residual.
    for (int64_t t = 0; t < seq; ++t) {
      LayerNormRow(&x[static_cast<size_t>(t * h)], l.ln_2_weight, l.ln_2_bias, h,
                   params.layer_norm_eps, &normed[static_cast<size_t>(t * h)]);
    }
    for (int64_t t = 0; t < seq; ++t) {
      LinearRow(&normed[static_cast<size_t>(t * h)], l.c_fc_weight, l.c_fc_bias, h,
                params.inner_size, &inner_buf[static_cast<size_t>(t * params.inner_size)]);
    }
    for (float& v : inner_buf) v = static_cast<float>(GeluNew(static_cast<double>(v)));
    for (int64_t t = 0; t < seq; ++t) {
      LinearRow(&inner_buf[static_cast<size_t>(t * params.inner_size)], l.mlp_c_proj_weight,
                l.mlp_c_proj_bias, params.inner_size, h, &proj[static_cast<size_t>(t * h)]);
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];
  }

  // gpt2.py:239 — ln_f.
  std::vector<float> out(static_cast<size_t>(seq * h));
  for (int64_t t = 0; t < seq; ++t) {
    LayerNormRow(&x[static_cast<size_t>(t * h)], weights.ln_f_weight, weights.ln_f_bias, h,
                 params.layer_norm_eps, &out[static_cast<size_t>(t * h)]);
  }
  return out;
}

std::vector<float> LogitsHost(const Params& params, const Weights& weights,
                              const std::vector<float>& hidden) {
  const int64_t h = params.hidden_size;
  VT_CHECK(hidden.size() % static_cast<size_t>(h) == 0, "gpt2: hidden is not a multiple of H");
  const int64_t seq = static_cast<int64_t>(hidden.size()) / h;
  std::vector<float> out(static_cast<size_t>(seq * params.vocab_size));
  for (int64_t t = 0; t < seq; ++t) {
    // Tied lm_head: the projection IS wte, [vocab, H], so this is already
    // [out, in] and needs no transpose.
    LinearRow(&hidden[static_cast<size_t>(t * h)], weights.wte, {}, h, params.vocab_size,
              &out[static_cast<size_t>(t * params.vocab_size)]);
  }
  return out;
}

}  // namespace gpt2
}  // namespace vllm
