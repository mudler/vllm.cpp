// LTX-2.5 DURATION HEAD — see
// include/vllm/model_executor/models/ltx2_duration_head.h for the upstream
// mapping and the four things that fail silently.
//
// The pooler's attention routes through `vt::AttentionCross`, the shared
// non-causal cross-attention seam phase L2 added, rather than through a local
// softmax loop: the queries are the learnable tokens and the keys/values are the
// token stream, so Tq != S and `vt::Attention` cannot express it. Every
// projection routes through `vt::MatmulBT`.
#include "vllm/model_executor/models/ltx2_duration_head.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& message) { throw std::runtime_error(message); }

void Require(bool condition, const std::string& message) {
  if (!condition) Refuse(message);
}

// `torch.nn.Linear`: out = in @ weight^T + bias, with `weight` [out, in].
std::vector<float> Linear(vt::Queue& q, const float* in, int64_t rows, int64_t in_features,
                          const std::vector<float>& weight, const std::vector<float>& bias,
                          int64_t out_features) {
  Require(weight.size() == static_cast<size_t>(out_features * in_features),
          "ltx2 duration head: linear weight has the wrong element count");
  std::vector<float> out(static_cast<size_t>(rows * out_features));
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(in), vt::DType::kF32, vt::Device{},
                                        {rows, in_features});
  vt::Tensor w = vt::Tensor::Contiguous(const_cast<float*>(weight.data()), vt::DType::kF32,
                                        vt::Device{}, {out_features, in_features});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, vt::Device{},
                                        {rows, out_features});
  vt::MatmulBT(q, o, a, w);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t i = 0; i < out_features; ++i) {
      out[static_cast<size_t>(r * out_features + i)] += bias[static_cast<size_t>(i)];
    }
  }
  return out;
}

// `torch.nn.functional.gelu(x, approximate="tanh")` — the same activation the
// DiT's FeedForward uses (gelu_approx.py:4-10).
//
// POINTWISE f64, WIDER than upstream's f32, and NOT covered by the suite's
// f64-reduction convention: this is not a reduction at all. Upstream rounds to
// f32 at each step of the expression, so computing the whole thing in double and
// rounding once is numerically FINER than the mirror rather than equal to it —
// the too-wide polarity AGENTS.md warns about, which a value gate cannot catch.
// Left as-is here rather than narrowed in a review-repair branch, because
// narrowing moves the duration-head goldens and so owes its own red-first change.
// Same class as `Silu` in ltx2_upsampler.cpp; see that file's header note.
float GeluTanh(float x) {
  const double v = static_cast<double>(x);
  const double inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
  return static_cast<float>(0.5 * v * (1.0 + std::tanh(inner)));
}

}  // namespace

std::vector<Ltx2DurationHeadTensorSpec> EnumerateLtx2DurationHeadTensors(
    const Ltx2DurationHeadConfig& config) {
  // torch's `named_parameters()` yields a module's own bare nn.Parameters BEFORE
  // it descends into submodules, so the two modality embeddings come first even
  // though they are declared after `video_input_proj`.
  const std::string p = config.prefix;
  const int64_t hidden = config.pooler_hidden_dim;
  std::vector<Ltx2DurationHeadTensorSpec> specs;
  specs.push_back({p + "video_modality_emb", {hidden}});
  specs.push_back({p + "audio_modality_emb", {hidden}});
  specs.push_back({p + "video_input_proj.weight", {hidden, config.video_cross_attention_dim}});
  specs.push_back({p + "video_input_proj.bias", {hidden}});
  specs.push_back({p + "audio_input_proj.weight", {hidden, config.audio_cross_attention_dim}});
  specs.push_back({p + "audio_input_proj.bias", {hidden}});
  specs.push_back({p + "attention_pooler.query_tokens", {config.num_queries, hidden}});
  // nn.MultiheadAttention's PACKED projection: [3 * E, E] in Q, K, V order.
  specs.push_back({p + "attention_pooler.cross_attn.in_proj_weight", {3 * hidden, hidden}});
  specs.push_back({p + "attention_pooler.cross_attn.in_proj_bias", {3 * hidden}});
  specs.push_back({p + "attention_pooler.cross_attn.out_proj.weight", {hidden, hidden}});
  specs.push_back({p + "attention_pooler.cross_attn.out_proj.bias", {hidden}});
  specs.push_back({p + "mlp_hidden.weight", {config.mlp_hidden, hidden * config.num_queries}});
  specs.push_back({p + "mlp_hidden.bias", {config.mlp_hidden}});
  specs.push_back({p + "mlp_out.weight", {1, config.mlp_hidden}});
  specs.push_back({p + "mlp_out.bias", {1}});
  return specs;
}

std::vector<float> Ltx2DurationAttentionPool(const Ltx2DurationHeadConfig& config,
                                             const Ltx2VaeWeights& weights, const float* tokens,
                                             int64_t batch, int64_t token_count) {
  Require(tokens != nullptr, "ltx2 duration pooler: `tokens` is required");
  const int64_t hidden = config.pooler_hidden_dim;
  const int64_t heads = config.num_pooler_heads;
  Require(hidden % heads == 0,
          "ltx2 duration pooler: pooler_hidden_dim " + std::to_string(hidden) +
              " must be divisible by num_pooler_heads " + std::to_string(heads));
  const int64_t head_dim = hidden / heads;
  const int64_t queries = config.num_queries;
  const std::string p = config.prefix + "attention_pooler.";

  vt::Queue q{vt::Device{}, nullptr};
  const std::vector<float>& in_proj_weight = weights.Get(p + "cross_attn.in_proj_weight");
  const std::vector<float>& in_proj_bias = weights.Get(p + "cross_attn.in_proj_bias");
  Require(in_proj_weight.size() == static_cast<size_t>(3 * hidden * hidden),
          "ltx2 duration pooler: in_proj_weight must be [3 * E, E]");

  // The packed [3E, E] projection, sliced Q / K / V — and it is CROSS attention,
  // so the three slices are applied to two DIFFERENT inputs. A port that treated
  // it as self-attention would project the queries three times.
  auto slice_weight = [&](int64_t index) {
    return std::vector<float>(
        in_proj_weight.begin() + index * hidden * hidden,
        in_proj_weight.begin() + (index + 1) * hidden * hidden);
  };
  auto slice_bias = [&](int64_t index) {
    return std::vector<float>(in_proj_bias.begin() + index * hidden,
                              in_proj_bias.begin() + (index + 1) * hidden);
  };

  // The learnable queries, broadcast across the batch (duration_head.py:47).
  const std::vector<float>& query_tokens = weights.Get(p + "query_tokens");
  std::vector<float> expanded(static_cast<size_t>(batch * queries * hidden));
  for (int64_t b = 0; b < batch; ++b) {
    std::copy(query_tokens.begin(), query_tokens.end(),
              expanded.begin() + b * queries * hidden);
  }

  const std::vector<float> qw = slice_weight(0);
  const std::vector<float> kw = slice_weight(1);
  const std::vector<float> vw = slice_weight(2);
  const std::vector<float> qb = slice_bias(0);
  const std::vector<float> kb = slice_bias(1);
  const std::vector<float> vb = slice_bias(2);

  std::vector<float> qp =
      Linear(q, expanded.data(), batch * queries, hidden, qw, qb, hidden);
  std::vector<float> kp = Linear(q, tokens, batch * token_count, hidden, kw, kb, hidden);
  std::vector<float> vp = Linear(q, tokens, batch * token_count, hidden, vw, vb, hidden);

  std::vector<float> attn(static_cast<size_t>(batch * queries * hidden));
  // torch SDPA's default scale is `E ** -0.5` with E = head_dim.
  vt::AttentionCrossArgs args;
  args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  for (int64_t b = 0; b < batch; ++b) {
    vt::Tensor tq = vt::Tensor::Contiguous(qp.data() + b * queries * hidden, vt::DType::kF32,
                                           vt::Device{}, {queries, heads, head_dim});
    vt::Tensor tk = vt::Tensor::Contiguous(kp.data() + b * token_count * hidden, vt::DType::kF32,
                                           vt::Device{}, {token_count, heads, head_dim});
    vt::Tensor tv = vt::Tensor::Contiguous(vp.data() + b * token_count * hidden, vt::DType::kF32,
                                           vt::Device{}, {token_count, heads, head_dim});
    vt::Tensor to = vt::Tensor::Contiguous(attn.data() + b * queries * hidden, vt::DType::kF32,
                                           vt::Device{}, {queries, heads, head_dim});
    // No mask, by construction: the connector has already replaced every padded
    // position with a register and zeroed its mask (duration_head.py:15-16).
    vt::AttentionCross(q, to, tq, tk, tv, nullptr, args);
  }

  return Linear(q, attn.data(), batch * queries, hidden,
                weights.Get(p + "cross_attn.out_proj.weight"),
                weights.Get(p + "cross_attn.out_proj.bias"), hidden);
}

std::vector<float> Ltx2DurationPredict(const Ltx2DurationHeadConfig& config,
                                       const Ltx2VaeWeights& weights, const float* video_tokens,
                                       int64_t video_token_count, const float* audio_tokens,
                                       int64_t audio_token_count, int64_t batch) {
  // duration_head.py:104-105 — upstream's own ValueError.
  Require(video_tokens != nullptr || audio_tokens != nullptr,
          "ltx2 duration head: forward requires at least one of video_tokens / audio_tokens");

  const std::string p = config.prefix;
  const int64_t hidden = config.pooler_hidden_dim;
  vt::Queue q{vt::Device{}, nullptr};

  // The modality embedding is added AFTER the projection (:109, :111), which is
  // what lets the pooler tell the two streams apart.
  auto project = [&](const float* tokens, int64_t token_count, const std::string& stream) {
    std::vector<float> out =
        Linear(q, tokens, batch * token_count, stream == "video"
                                                   ? config.video_cross_attention_dim
                                                   : config.audio_cross_attention_dim,
               weights.Get(p + stream + "_input_proj.weight"),
               weights.Get(p + stream + "_input_proj.bias"), hidden);
    const std::vector<float>& emb = weights.Get(p + stream + "_modality_emb");
    for (int64_t row = 0; row < batch * token_count; ++row) {
      for (int64_t i = 0; i < hidden; ++i) {
        out[static_cast<size_t>(row * hidden + i)] += emb[static_cast<size_t>(i)];
      }
    }
    return out;
  };

  std::vector<float> video_projected;
  std::vector<float> audio_projected;
  if (video_tokens != nullptr) video_projected = project(video_tokens, video_token_count, "video");
  if (audio_tokens != nullptr) audio_projected = project(audio_tokens, audio_token_count, "audio");

  const int64_t video_rows = video_tokens != nullptr ? video_token_count : 0;
  const int64_t audio_rows = audio_tokens != nullptr ? audio_token_count : 0;
  const int64_t token_count = video_rows + audio_rows;

  // `torch.cat(token_groups, dim=1)` (:113) — along the TOKEN axis, per batch
  // row. Concatenating along the feature axis also type-checks here.
  std::vector<float> tokens(static_cast<size_t>(batch * token_count * hidden));
  for (int64_t b = 0; b < batch; ++b) {
    float* dst = tokens.data() + b * token_count * hidden;
    if (video_rows > 0) {
      std::copy(video_projected.begin() + b * video_rows * hidden,
                video_projected.begin() + (b + 1) * video_rows * hidden, dst);
    }
    if (audio_rows > 0) {
      std::copy(audio_projected.begin() + b * audio_rows * hidden,
                audio_projected.begin() + (b + 1) * audio_rows * hidden,
                dst + video_rows * hidden);
    }
  }

  const std::vector<float> pooled =
      Ltx2DurationAttentionPool(config, weights, tokens.data(), batch, token_count);
  // `pooled.reshape(pooled.shape[0], -1)` (:115) — every query's vector,
  // concatenated, which is why mlp_hidden's input width is hidden * num_queries.
  const std::vector<float> hidden_out =
      Linear(q, pooled.data(), batch, hidden * config.num_queries,
             weights.Get(p + "mlp_hidden.weight"), weights.Get(p + "mlp_hidden.bias"),
             config.mlp_hidden);
  std::vector<float> activated(hidden_out.size());
  for (size_t i = 0; i < hidden_out.size(); ++i) activated[i] = GeluTanh(hidden_out[i]);

  const std::vector<float> log_duration =
      Linear(q, activated.data(), batch, config.mlp_hidden, weights.Get(p + "mlp_out.weight"),
             weights.Get(p + "mlp_out.bias"), 1);

  // :117-118 — the regression is trained in LOG-seconds and exponentiated here,
  // so callers always get seconds. Returning the raw regression gives a finite,
  // plausible, completely different duration.
  std::vector<float> seconds(static_cast<size_t>(batch));
  for (int64_t b = 0; b < batch; ++b) {
    seconds[static_cast<size_t>(b)] =
        static_cast<float>(std::exp(static_cast<double>(log_duration[static_cast<size_t>(b)])));
  }
  return seconds;
}

}  // namespace vllm
