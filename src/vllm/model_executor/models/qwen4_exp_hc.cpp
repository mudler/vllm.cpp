// Qwen4-Exp (`Qwen3.8-Flash-Next`) W3 — the 4-branch GATED-RESIDUAL
// hyper-connection stream and its grouped RMSNorm. HOST reference.
//
// Algorithm: transformers v5.16.0 (the lane pin)
//   `models/qwen4_exp/modeling_qwen4_exp.py`
//   ::Qwen4ExpTextRMSNorm (:158-181), ::Qwen4ExpTextGatedResidual (:941-969),
//   ::Qwen4ExpTextDecoderLayer.forward write-back (:825-826, :831-832 in the
//   modular file `modular_qwen4_exp.py`).
// Op form: vLLM @ origin/main 6a5e8f5979
//   `model_executor/layers/layernorm.py`::RMSNormGated (:172), `group_size`
//   (:187), grouped branch in `forward_static` (:243-244, :258-264).
// See qwen4_exp_hc.h for the oracle table, the `1 + w` resolution, and the
// fusion-seam note.
#include "vllm/model_executor/models/qwen4_exp_hc.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace vllm::qwen4_exp {

namespace {

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y[o] = Σ_i w[o*K + i] · x[i]. PyTorch `nn.Linear(bias=False)` weight layout,
// `(out_features, in_features)` row-major.
void LinearNoBias(const float* w, const float* x, int64_t out_dim, int64_t in_dim, float* y) {
  for (int64_t o = 0; o < out_dim; ++o) {
    const float* row = w + o * in_dim;
    float acc = 0.0f;
    for (int64_t i = 0; i < in_dim; ++i) acc += row[i] * x[i];
    y[o] = acc;
  }
}

void RequireSize(size_t got, size_t want, const char* what) {
  if (got != want) {
    throw std::invalid_argument(std::string("qwen4_exp: ") + what + " has " + std::to_string(got) +
                                " elements, expected " + std::to_string(want) + ".");
  }
}

}  // namespace

std::vector<float> HcNormWeightFromHf(const std::vector<float>& w_hf) {
  std::vector<float> w(w_hf.size());
  for (size_t i = 0; i < w_hf.size(); ++i) w[i] = 1.0f + w_hf[i];
  return w;
}

std::vector<float> GroupedRmsNorm(const std::vector<float>& x, const std::vector<float>& weight,
                                  int64_t group_size, float eps) {
  if (group_size <= 0) {
    throw std::invalid_argument("qwen4_exp: group_size must be positive, got " +
                                std::to_string(group_size) + ".");
  }
  const size_t g = static_cast<size_t>(group_size);
  if (x.size() % g != 0) {
    // `Qwen4ExpTextRMSNorm.__init__` (:164-165) raises the same condition.
    throw std::invalid_argument("qwen4_exp: hidden_size (" + std::to_string(x.size()) +
                                ") must be divisible by group_size (" + std::to_string(g) + ").");
  }
  RequireSize(weight.size(), x.size(), "hc_norm weight");

  std::vector<float> out(x.size());
  for (size_t base = 0; base < x.size(); base += g) {
    // The reduction is INDEPENDENT per group: this is what `RMSNormGated`'s
    // `rearrange(x, "... (g d) -> ... g d")` buys and what the plain `RMSNorm`
    // cannot express. Accumulated in double, per the host-reference convention.
    double ss = 0.0;
    for (size_t d = 0; d < g; ++d) {
      const double v = x[base + d];
      ss += v * v;
    }
    // eps is INSIDE the rsqrt, added to the mean square, never to the norm.
    const float r = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(g)) + eps);
    for (size_t d = 0; d < g; ++d) {
      out[base + d] = x[base + d] * r * weight[base + d];
    }
  }
  return out;
}

GatedResidualResult GatedResidualForward(const std::vector<float>& hyper_input,
                                         const GatedResidualWeights& weights, int64_t hc,
                                         int64_t hidden, float eps) {
  if (hc <= 1 || hidden <= 0) {
    // `Qwen4ExpTextConfig.__post_init__` rejects hc_count <= 1 outright.
    throw std::invalid_argument("qwen4_exp: hc_count must be > 1 and hidden_size > 0, got " +
                                std::to_string(hc) + " and " + std::to_string(hidden) + ".");
  }
  const int64_t flat = hc * hidden;
  if (hyper_input.size() != static_cast<size_t>(flat)) {
    // Mirrors `Qwen4ExpTextGatedResidual.forward` (:955-958).
    throw std::invalid_argument("qwen4_exp: expected " + std::to_string(flat) +
                                " hyper-connection features, got " +
                                std::to_string(hyper_input.size()) + ".");
  }
  if (weights.mix_down.empty() || weights.mix_down.size() % static_cast<size_t>(flat) != 0) {
    throw std::invalid_argument("qwen4_exp: input_mix_weight_down is not a multiple of " +
                                std::to_string(flat) + " (got " +
                                std::to_string(weights.mix_down.size()) + ").");
  }
  const int64_t rank = static_cast<int64_t>(weights.mix_down.size() / static_cast<size_t>(flat));
  RequireSize(weights.mix_up.size(), static_cast<size_t>(flat * rank), "input_mix_weight_up");
  if (!weights.block_inject.empty()) {
    RequireSize(weights.block_inject.size(), static_cast<size_t>(hc * flat), "block_inject_weight");
  }

  GatedResidualResult out;
  out.hyper_input_normed =
      GroupedRmsNorm(hyper_input, weights.hc_norm_weight, hidden, eps);
  const float* normed = out.hyper_input_normed.data();

  // DIVISION 1 — inside the SiLU, on the [rank] low-rank intermediate, BEFORE
  // the activation: `F.silu(down(x) / hc_count)`. SiLU is not homogeneous, so
  // `silu(a)/hc` is a different function; the placement is load-bearing.
  std::vector<float> low(static_cast<size_t>(rank));
  LinearNoBias(weights.mix_down.data(), normed, rank, flat, low.data());
  // A true division, not a reciprocal multiply: upstream spells `/ self.hc_count`
  // and 1/hc is inexact for any hc that is not a power of two, so the shortcut
  // would put a 1-ulp wedge between this reference and the oracle at hc_count=3.
  const float hc_f = static_cast<float>(hc);
  for (int64_t r = 0; r < rank; ++r) {
    const float a = low[r] / hc_f;
    low[r] = a * Sigmoid(a);
  }

  // NO division on the up projection: `torch.sigmoid(up(...))`, full stop.
  std::vector<float> gate(static_cast<size_t>(flat));
  LinearNoBias(weights.mix_up.data(), low.data(), flat, rank, gate.data());
  for (int64_t p = 0; p < flat; ++p) gate[p] = Sigmoid(gate[p]);

  // `.unflatten(-1, (hc, H))`, multiply against the NORMED stream — not the raw
  // one — then `.mean(dim=-2)`. A MEAN over hc, never a sum.
  out.mixed_input.assign(static_cast<size_t>(hidden), 0.0f);
  for (int64_t j = 0; j < hc; ++j) {
    const float* g_row = gate.data() + j * hidden;
    const float* s_row = normed + j * hidden;
    for (int64_t h = 0; h < hidden; ++h) out.mixed_input[h] += g_row[h] * s_row[h];
  }
  for (int64_t h = 0; h < hidden; ++h) out.mixed_input[h] /= hc_f;

  // `block_inject_weight is None` returns here, and that early return IS the
  // model's final mixer (`use_combine=False`).
  if (weights.block_inject.empty()) return out;

  // DIVISION 2 — inside the injection sigmoid, whole sigmoid scaled by 2:
  // `2 * sigmoid(inject(x) / hc_count)`. Range (0, 2), exactly 1.0 at a zero
  // logit, so an untrained branch is the identity rather than a half-scale.
  out.injection_weights.assign(static_cast<size_t>(hc), 0.0f);
  LinearNoBias(weights.block_inject.data(), normed, hc, flat, out.injection_weights.data());
  for (int64_t j = 0; j < hc; ++j) {
    out.injection_weights[j] = 2.0f * Sigmoid(out.injection_weights[j] / hc_f);
  }
  return out;
}

void GatedResidualWriteBackInPlace(float* hyper, const float* block_out,
                                   const float* injection_weights, int64_t hc, int64_t hidden) {
  // The rank-1 update, never the materialized broadcast. See the header: this
  // function is the seam a fused device kernel replaces.
  for (int64_t j = 0; j < hc; ++j) {
    const float w = injection_weights[j];
    float* row = hyper + j * hidden;
    for (int64_t h = 0; h < hidden; ++h) row[h] += block_out[h] * w;
  }
}

std::vector<float> GatedResidualWriteBack(const std::vector<float>& hyper_input,
                                          const std::vector<float>& block_out,
                                          const std::vector<float>& injection_weights, int64_t hc,
                                          int64_t hidden) {
  RequireSize(hyper_input.size(), static_cast<size_t>(hc * hidden), "hyper_input");
  RequireSize(block_out.size(), static_cast<size_t>(hidden), "block output");
  RequireSize(injection_weights.size(), static_cast<size_t>(hc), "injection_weights");
  std::vector<float> out = hyper_input;
  GatedResidualWriteBackInPlace(out.data(), block_out.data(), injection_weights.data(), hc, hidden);
  return out;
}

}  // namespace vllm::qwen4_exp
