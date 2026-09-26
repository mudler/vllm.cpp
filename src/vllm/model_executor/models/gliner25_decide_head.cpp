// GLiNER2.5-Decide classification head — host reference forward (MODEL-GLINER25-DECIDE).
//
// Head: Linear(H, 2H) → ReLU → Linear(2H, 1)
// Activation: softmax for exclusive (single-label) tasks.

#include "vllm/model_executor/models/gliner25_decide.h"

#include <algorithm>
#include <cmath>

namespace vllm {
namespace gliner25_decide {

namespace {

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
    y[static_cast<size_t>(j)] =
        static_cast<float>(acc + static_cast<double>(b[static_cast<size_t>(j)]));
  }
  return y;
}

}  // namespace

std::vector<float> ClassifierForward(
    const DecideHeadWeights& hw, const DecideHeadParams& params,
    const std::vector<float>& label_embs, int64_t num_labels) {
  const int64_t H = params.hidden_size;
  const int64_t H2 = H * 2;

  std::vector<float> logits(static_cast<size_t>(num_labels));

  for (int64_t n = 0; n < num_labels; ++n) {
    // Extract one label embedding [H]
    std::vector<float> emb(static_cast<size_t>(H));
    std::copy_n(&label_embs[static_cast<size_t>(n * H)],
                static_cast<size_t>(H), emb.data());

    // Linear(H, 2H) → ReLU
    auto hidden = Linear(emb, hw.w0, hw.b0, H, H2);
    for (float& v : hidden) v = std::max(0.0F, v);

    // Linear(2H, 1) → scalar logit
    auto out = Linear(hidden, hw.w2, hw.b2, H2, 1);
    logits[static_cast<size_t>(n)] = out[0];
  }

  return logits;
}

std::vector<float> Softmax(const std::vector<float>& logits,
                           float temperature) {
  if (logits.empty()) return {};
  std::vector<float> scaled(logits.size());
  float inv_temp = 1.0F / (temperature > 0 ? temperature : 1.0F);
  for (size_t i = 0; i < logits.size(); ++i) {
    scaled[i] = logits[i] * inv_temp;
  }

  float max_val = *std::max_element(scaled.begin(), scaled.end());
  std::vector<float> exp_vals(scaled.size());
  double sum = 0.0;
  for (size_t i = 0; i < scaled.size(); ++i) {
    exp_vals[i] = std::exp(scaled[i] - max_val);
    sum += static_cast<double>(exp_vals[i]);
  }
  if (sum < 1e-12) sum = 1e-12;
  std::vector<float> probs(scaled.size());
  for (size_t i = 0; i < scaled.size(); ++i) {
    probs[i] = static_cast<float>(exp_vals[i] / sum);
  }
  return probs;
}

std::vector<float> Sigmoid(const std::vector<float>& logits,
                           float temperature) {
  std::vector<float> probs(logits.size());
  float inv_temp = 1.0F / (temperature > 0 ? temperature : 1.0F);
  for (size_t i = 0; i < logits.size(); ++i) {
    float x = logits[i] * inv_temp;
    if (x >= 0) {
      probs[i] = 1.0F / (1.0F + std::exp(-x));
    } else {
      float e = std::exp(x);
      probs[i] = e / (1.0F + e);
    }
  }
  return probs;
}

}  // namespace gliner25_decide
}  // namespace vllm
