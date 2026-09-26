// CLM (Contrastive-LM) host reference forward (MODEL-CLM).
//
// MLP head: Linear(H,P) → GELU → LayerNorm(P) → Linear(P,P) → GELU → Linear(P,E)
// then L2-normalize. Scoring: scaled cosine = exp(logit_scale) * dot(state, option).
//
// Ported from Contrastive-LM/CLM-v0.1-8B (spec .agents/specs/clm.md).

#include "vllm/model_executor/models/clm.h"

#include <algorithm>
#include <cmath>

namespace vllm {
namespace clm {

namespace {

float Gelu(float x) {
  // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
  return x * 0.5F * (1.0F + std::erf(x * 0.7071067811865475F));
}

// Linear: y[out] = sum_i(x[i] * W[out*in + i]) + b[out]
// W layout: [out_features, in_features] (PyTorch nn.Linear)
std::vector<float> Linear(
    const std::vector<float>& x, const std::vector<float>& w,
    const std::vector<float>& b, int64_t in, int64_t out) {
  std::vector<float> y(static_cast<size_t>(out), 0.0F);
  for (int64_t j = 0; j < out; ++j) {
    const float* wr = &w[static_cast<size_t>(j * in)];
    double acc = 0.0;
    for (int64_t i = 0; i < in; ++i) {
      acc += static_cast<double>(x[static_cast<size_t>(i)]) *
             static_cast<double>(wr[i]);
    }
    y[static_cast<size_t>(j)] = static_cast<float>(acc + static_cast<double>(b[static_cast<size_t>(j)]));
  }
  return y;
}

// LayerNorm: y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
std::vector<float> LayerNorm(
    const std::vector<float>& x, const std::vector<float>& gamma,
    const std::vector<float>& beta, int64_t n, float eps = 1e-5F) {
  double mean = 0.0;
  for (int64_t i = 0; i < n; ++i) mean += x[static_cast<size_t>(i)];
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = x[static_cast<size_t>(i)] - mean;
    var += d * d;
  }
  var /= static_cast<double>(n);
  const float inv = 1.0F / std::sqrt(static_cast<float>(var) + eps);
  std::vector<float> y(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    y[static_cast<size_t>(i)] =
        (x[static_cast<size_t>(i)] - static_cast<float>(mean)) * inv *
            gamma[static_cast<size_t>(i)] +
        beta[static_cast<size_t>(i)];
  }
  return y;
}

void L2Normalize(float* v, int64_t n) {
  double norm = 0.0;
  for (int64_t i = 0; i < n; ++i) norm += static_cast<double>(v[i]) * v[i];
  norm = std::sqrt(norm);
  if (norm < 1e-12) norm = 1e-12;
  const float inv = static_cast<float>(1.0 / norm);
  for (int64_t i = 0; i < n; ++i) v[i] *= inv;
}

}  // namespace

std::vector<float> ClmMlpHeadForward(
    const MlpHeadWeights& hw, const HeadParams& params,
    const std::vector<float>& hidden) {
  const int64_t H = params.hidden_size;
  const int64_t P = params.proj_dim;
  const int64_t E = params.embed_dim;

  // Linear(H, P) → GELU
  auto h = Linear(hidden, hw.w0, hw.b0, H, P);
  for (float& v : h) v = Gelu(v);

  // LayerNorm(P)
  h = LayerNorm(h, hw.w2, hw.b2, P);

  // Linear(P, P) → GELU
  h = Linear(h, hw.w4, hw.b4, P, P);
  for (float& v : h) v = Gelu(v);

  // Linear(P, E)
  h = Linear(h, hw.w6, hw.b6, P, E);

  // L2-normalize
  L2Normalize(h.data(), E);

  return h;
}

std::vector<float> ClmScoreOptions(
    const std::vector<float>& h_state,
    const std::vector<std::vector<float>>& h_options,
    float logit_scale) {
  const float scale = std::exp(std::min(logit_scale, 100.0F));
  const int64_t E = static_cast<int64_t>(h_state.size());
  const int64_t K = static_cast<int64_t>(h_options.size());

  std::vector<float> scores(static_cast<size_t>(K), 0.0F);
  for (int64_t k = 0; k < K; ++k) {
    double dot = 0.0;
    for (int64_t i = 0; i < E; ++i) {
      dot += static_cast<double>(h_state[static_cast<size_t>(i)]) *
             static_cast<double>(h_options[static_cast<size_t>(k)][static_cast<size_t>(i)]);
    }
    scores[static_cast<size_t>(k)] = static_cast<float>(dot * static_cast<double>(scale));
  }
  return scores;
}

}  // namespace clm
}  // namespace vllm
