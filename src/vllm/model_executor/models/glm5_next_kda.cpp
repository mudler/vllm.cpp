// GLM-5.3-Flash — the KDA linear-attention arm's numerics.
// See glm5_next_kda.h for the full port map (file:line on both sides, @ the
// transformers lane pin v5.16.1) and for why the forget gate cannot be a
// parameter of kimi_kda.cpp's.
#include "vllm/model_executor/models/glm5_next_kda.h"

#include <algorithm>  // std::copy
#include <cmath>
#include <stdexcept>
#include <string>    // std::to_string
#include <utility>   // std::move

#include "vt/dtype.h"  // VT_CHECK, F32ToBF16, BF16ToF32

namespace vllm::glm5_next_kda {

namespace {

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// `torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))`
// (modular_glm5_next.py:403). The comparison is on the RAW g and the threshold
// is a literal 20.0, not a beta-scaled one: this is upstream's own overflow
// linearisation written inline, not `F.softplus`.
double Softplus20(double g) { return g > 20.0 ? g : std::log1p(std::exp(g)); }

// out[t, o] = sum_i w[o, i] * x[t, i], accumulated in double.
std::vector<float> MatVecRows(const std::vector<float>& w,
                              const std::vector<float>& x, int64_t out_dim,
                              int64_t in_dim, int64_t num_tokens,
                              const char* what) {
  VT_CHECK(static_cast<int64_t>(w.size()) == out_dim * in_dim,
           std::string("glm5_next kda: ") + what + " weight size mismatch");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * in_dim,
           std::string("glm5_next kda: ") + what + " input size mismatch");
  std::vector<float> y(static_cast<size_t>(num_tokens) * out_dim, 0.0f);
  for (int64_t t = 0; t < num_tokens; ++t) {
    const float* x_t = &x[t * in_dim];
    for (int64_t o = 0; o < out_dim; ++o) {
      const float* w_o = &w[o * in_dim];
      double acc = 0.0;
      for (int64_t i = 0; i < in_dim; ++i) acc += static_cast<double>(w_o[i]) * x_t[i];
      y[static_cast<size_t>(t * out_dim + o)] = static_cast<float>(acc);
    }
  }
  return y;
}

vt::Tensor MakeT(void* data, vt::DType dt, vt::Device dev,
                 const std::vector<int64_t>& shape) {
  vt::Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

}  // namespace

// ── (1) the forget gate ──────────────────────────────────────────────────────

std::vector<float> Glm5NextLowRankProjection(const std::vector<float>& x,
                                             const std::vector<float>& f_a,
                                             const std::vector<float>& f_b,
                                             int64_t num_tokens,
                                             int64_t hidden_size,
                                             int64_t num_heads,
                                             int64_t head_dim) {
  VT_CHECK(hidden_size > 0 && num_heads > 0 && head_dim > 0,
           "glm5_next kda: low-rank projection needs positive dims");
  // f_a: hidden -> head_dim (the RANK); f_b: head_dim -> H*D. No activation
  // between them (modular_glm5_next.py:382-383,:392): a pure low-rank map.
  const std::vector<float> r =
      MatVecRows(f_a, x, head_dim, hidden_size, num_tokens, "f_a_proj");
  return MatVecRows(f_b, r, num_heads * head_dim, head_dim, num_tokens, "f_b_proj");
}

std::vector<float> Glm5NextForgetGate(
    const std::vector<float>& g1, const std::vector<float>& a_log,
    const std::vector<float>& dt_bias, int64_t num_tokens, int64_t num_heads,
    int64_t head_dim, std::optional<double> safe_gate_lower_bound) {
  const int64_t hd = num_heads * head_dim;
  VT_CHECK(num_heads > 0 && head_dim > 0, "glm5_next kda: bad forget-gate dims");
  VT_CHECK(static_cast<int64_t>(g1.size()) == num_tokens * hd,
           "glm5_next kda: forget-gate g1 size mismatch");
  VT_CHECK(static_cast<int64_t>(a_log.size()) == num_heads,
           "glm5_next kda: forget-gate A_log size mismatch");
  // `self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))` (:384) is
  // UNCONDITIONAL and :393 always adds it, so this model has no biasless mode
  // to mirror. An absent or misshaped tensor is refused by name rather than
  // treated as a zero bias: silently ignoring the field and adding nothing
  // computes a DIFFERENT gate that stays finite and plausible, which is the
  // same failure the non-`silu` `hidden_act` refusal below exists to prevent.
  VT_CHECK(static_cast<int64_t>(dt_bias.size()) == hd,
           "glm5_next kda: the forget gate needs a dt_bias of "
           "num_heads*head_dim = " + std::to_string(hd) + ", and got " +
               std::to_string(dt_bias.size()) +
               " (modular_glm5_next.py:384 declares `dt_bias` unconditionally "
               "and :393 always adds it, so there is no biasless mode)");
  // Upstream multiplies by `self.safe_gate_lower_bound` unnegated (:399), so a
  // non-negative bound yields a non-negative LOG-decay and the recurrent state
  // grows without bound. `ParseGlm5NextParams` refuses that at config time; a
  // caller that builds the dims by hand is refused here rather than producing
  // the divergent model.
  VT_CHECK(!safe_gate_lower_bound.has_value() || *safe_gate_lower_bound < 0.0,
           "glm5_next kda: `gate_lower_bound` must be negative; a non-negative "
           "bound makes the KDA log-decay non-negative and the recurrent state "
           "diverge (modular_glm5_next.py:398-399 multiplies by the bound "
           "as-is, it does not negate it)");

  std::vector<float> y(static_cast<size_t>(num_tokens) * hd, 0.0f);
  for (int64_t h = 0; h < num_heads; ++h) {
    // decay_rate = exp(A_log[h]) and it is POSITIVE (:394-395). The softplus
    // branch negates it OUTSIDE the nonlinearity (:405); the sigmoid branch
    // multiplies g by it INSIDE (:399). Reusing one sign for the other mirrors
    // the whole gate about g = 0.
    const double decay_rate = std::exp(static_cast<double>(a_log[h]));
    for (int64_t t = 0; t < num_tokens; ++t) {
      const float* g_t = &g1[t * hd + h * head_dim];
      float* y_t = &y[t * hd + h * head_dim];
      for (int64_t d = 0; d < head_dim; ++d) {
        double g = g_t[d];
        g += dt_bias[h * head_dim + d];
        const double out = safe_gate_lower_bound.has_value()
                               ? *safe_gate_lower_bound * Sigmoid(decay_rate * g)
                               : -decay_rate * Softplus20(g);
        y_t[d] = static_cast<float>(out);
      }
    }
  }
  return y;
}

// ── (2) the strict-fp32 gated output norm ────────────────────────────────────

std::vector<float> Glm5NextRmsNormGated(const std::vector<float>& x,
                                        const std::vector<float>& gate,
                                        const std::vector<float>& weight,
                                        int64_t num_tokens, int64_t num_heads,
                                        int64_t head_dim, double eps,
                                        Glm5NextActivationDType out_dtype) {
  const int64_t hd = num_heads * head_dim;
  VT_CHECK(num_heads > 0 && head_dim > 0, "glm5_next kda: bad gated-norm dims");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * hd,
           "glm5_next kda: gated-norm x size mismatch");
  VT_CHECK(static_cast<int64_t>(gate.size()) == num_tokens * hd,
           "glm5_next kda: gated-norm gate size mismatch");
  VT_CHECK(static_cast<int64_t>(weight.size()) == head_dim,
           "glm5_next kda: gated-norm weight size mismatch");

  std::vector<float> out(static_cast<size_t>(num_tokens) * hd, 0.0f);
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t h = 0; h < num_heads; ++h) {
      const float* x_r = &x[t * hd + h * head_dim];
      const float* g_r = &gate[t * hd + h * head_dim];
      float* o_r = &out[t * hd + h * head_dim];
      // f32 (double here) VARIANCE and WEIGHT, whatever the model dtype is:
      // upstream casts both explicitly and says why — "Strict FP32 norm (do
      // not downcast on the weights)" (:417-421).
      double var = 0.0;
      for (int64_t d = 0; d < head_dim; ++d)
        var += static_cast<double>(x_r[d]) * x_r[d];
      var /= static_cast<double>(head_dim);
      const double rstd = 1.0 / std::sqrt(var + eps);
      for (int64_t d = 0; d < head_dim; ++d) {
        const double normed = static_cast<double>(x_r[d]) * rstd * weight[d];
        // SIGMOID, not silu (:412 sets activation="sigmoid").
        const double res = normed * Sigmoid(static_cast<double>(g_r[d]));
        float v = static_cast<float>(res);
        // `return hidden_states.to(input_dtype)` (:426): the ONE place the
        // model dtype re-enters this norm.
        if (out_dtype == Glm5NextActivationDType::kBFloat16)
          v = vt::BF16ToF32(vt::F32ToBF16(v));
        o_r[d] = v;
      }
    }
  }
  return out;
}

// ── (3) l2norm ───────────────────────────────────────────────────────────────

std::vector<float> Glm5NextL2Norm(const std::vector<float>& x, int64_t num_rows,
                                  int64_t dim, double eps) {
  VT_CHECK(dim > 0, "glm5_next kda: bad l2norm dim");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_rows * dim,
           "glm5_next kda: l2norm x size mismatch");
  std::vector<float> y(static_cast<size_t>(num_rows) * dim, 0.0f);
  for (int64_t r = 0; r < num_rows; ++r) {
    const float* x_r = &x[r * dim];
    double ss = 0.0;
    for (int64_t d = 0; d < dim; ++d) ss += static_cast<double>(x_r[d]) * x_r[d];
    // sqrt(SUM + eps), and the eps is INSIDE the root and ADDED — not
    // `F.normalize`'s max(norm, eps) (:433-436).
    const double inv_norm = std::sqrt(ss + eps);
    for (int64_t d = 0; d < dim; ++d)
      y[static_cast<size_t>(r * dim + d)] =
          static_cast<float>(static_cast<double>(x_r[d]) / inv_norm);
  }
  return y;
}

// ── (4) the q/k/v short convs ────────────────────────────────────────────────

std::vector<float> Glm5NextMixedQkvConvWeight(const std::vector<float>& q_conv,
                                              const std::vector<float>& k_conv,
                                              const std::vector<float>& v_conv,
                                              int64_t qkv_dim,
                                              int64_t kernel_size) {
  VT_CHECK(qkv_dim > 0 && kernel_size > 0, "glm5_next kda: bad conv dims");
  const size_t per = static_cast<size_t>(qkv_dim) * kernel_size;
  VT_CHECK(q_conv.size() == per && k_conv.size() == per && v_conv.size() == per,
           "glm5_next kda: each of q/k/v_conv1d must be [qkv_dim, kernel_size]");
  // q, k, v — the order `torch.cat` builds mixed_qkv in (:655-661). Any other
  // order permutes channels between the three streams silently.
  std::vector<float> w;
  w.reserve(per * 3);
  w.insert(w.end(), q_conv.begin(), q_conv.end());
  w.insert(w.end(), k_conv.begin(), k_conv.end());
  w.insert(w.end(), v_conv.begin(), v_conv.end());
  return w;
}

std::vector<float> Glm5NextMixedQkvConv(const std::vector<float>& x,
                                        const std::vector<float>& weight,
                                        int64_t num_tokens, int64_t channels,
                                        int64_t kernel_size,
                                        std::vector<float>* conv_state,
                                        const std::string& activation) {
  VT_CHECK(channels > 0 && kernel_size > 0, "glm5_next kda: bad conv dims");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * channels,
           "glm5_next kda: conv x size mismatch");
  VT_CHECK(static_cast<int64_t>(weight.size()) == channels * kernel_size,
           "glm5_next kda: conv weight size mismatch");
  // `self.activation = config.hidden_act` (:613) and it is "silu" on this
  // checkpoint. Refuse any other spelling by name: silently ignoring the field
  // and applying silu anyway is how a config-driven activation stops being
  // config-driven without anybody noticing.
  VT_CHECK(activation == "silu",
           "glm5_next kda: only hidden_act=\"silu\" is ported for the KDA short "
           "conv; got \"" + activation + "\" (modular_glm5_next.py:613, and "
           "`zai-org/GLM-5.3-Flash` sets \"silu\")");

  // state_len is whatever the caller's cache carries; the arithmetic needs
  // K-1 columns of history and this model's cache holds K (a slack column
  // upstream keeps at inkling/modeling_inkling.py:452).
  int64_t state_len = 0;
  if (conv_state != nullptr && !conv_state->empty()) {
    VT_CHECK(static_cast<int64_t>(conv_state->size()) % channels == 0,
             "glm5_next kda: conv_state must be [channels, state_len]");
    state_len = static_cast<int64_t>(conv_state->size()) / channels;
    VT_CHECK(state_len >= kernel_size - 1,
             "glm5_next kda: conv_state carries fewer than kernel_size-1 "
             "positions of history");
  }

  // Read position p of the [history ++ x] stream for channel c; history
  // positions below the carried state read 0 (a fresh sequence).
  const auto at = [&](int64_t c, int64_t p) -> double {
    if (p >= 0) return static_cast<double>(x[p * channels + c]);
    const int64_t idx = state_len + p;  // p in [-state_len, -1]
    if (idx < 0) return 0.0;
    return static_cast<double>((*conv_state)[c * state_len + idx]);
  };

  std::vector<float> y(static_cast<size_t>(num_tokens) * channels, 0.0f);
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t c = 0; c < channels; ++c) {
      const float* w_c = &weight[c * kernel_size];
      double acc = 0.0;
      // Causal: tap j aligns to stream position t - (K-1) + j.
      for (int64_t j = 0; j < kernel_size; ++j)
        acc += static_cast<double>(w_c[j]) * at(c, t - (kernel_size - 1) + j);
      y[static_cast<size_t>(t * channels + c)] =
          static_cast<float>(acc * Sigmoid(acc));  // silu
    }
  }

  // `conv_state.copy_(hidden_states_new[:, :, -state_len:])` (inkling:452):
  // the last state_len positions of the PRE-conv stream, newest last.
  if (state_len > 0) {
    std::vector<float> next(static_cast<size_t>(channels) * state_len, 0.0f);
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t s = 0; s < state_len; ++s) {
        const int64_t p = num_tokens - state_len + s;
        next[static_cast<size_t>(c * state_len + s)] =
            static_cast<float>(at(c, p));
      }
    }
    *conv_state = std::move(next);
  }
  return y;
}

// ── (5) the assembled KDA layer ──────────────────────────────────────────────

std::vector<float> Glm5NextKdaLayerForward(
    const Glm5NextKdaLayerWeights& w, const std::vector<float>& hidden_states,
    const Glm5NextKdaDims& d, int64_t num_tokens, Glm5NextKdaCache* cache,
    vt::Queue& queue) {
  const int64_t H = d.hidden_size;
  const int64_t nh = d.num_heads;
  const int64_t hd = d.head_dim;
  const int64_t proj = d.qkv_dim();
  const int64_t conv_dim = d.conv_dim();
  const int64_t K = d.conv_kernel_size;
  const int64_t T = num_tokens;
  VT_CHECK(H > 0 && nh > 0 && hd > 0 && K > 0, "glm5_next kda: bad layer dims");
  VT_CHECK(static_cast<int64_t>(hidden_states.size()) == T * H,
           "glm5_next kda: hidden_states size mismatch");
  // The device arm of this layer belongs to the assembled text forward (W5).
  // Refusing here beats half-wiring it: vt::KdaGatedDeltaRule dispatches on the
  // queue's device, and handing it host pointers on a CUDA queue is a crash,
  // not a fallback.
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU,
           "glm5_next kda: Glm5NextKdaLayerForward is the HOST reference and "
           "needs a CPU queue; the device arm is the assembled text forward's "
           "(W5, .agents/specs/glm5-next-flash.md)");

  // q/k/v projections, then ONE concatenated [q; k; v] stream (:655-661).
  const std::vector<float> q_raw = MatVecRows(w.q_proj, hidden_states, proj, H, T, "q_proj");
  const std::vector<float> k_raw = MatVecRows(w.k_proj, hidden_states, proj, H, T, "k_proj");
  const std::vector<float> v_raw = MatVecRows(w.v_proj, hidden_states, proj, H, T, "v_proj");
  std::vector<float> mixed(static_cast<size_t>(T) * conv_dim);
  for (int64_t t = 0; t < T; ++t) {
    float* row = &mixed[t * conv_dim];
    std::copy(&q_raw[t * proj], &q_raw[t * proj] + proj, row);
    std::copy(&k_raw[t * proj], &k_raw[t * proj] + proj, row + proj);
    std::copy(&v_raw[t * proj], &v_raw[t * proj] + proj, row + 2 * proj);
  }

  // The checkpoint's three separate depthwise convs ARE the reference's one
  // grouped conv over that stream, concatenated in q, k, v order.
  const std::vector<float> conv_w =
      Glm5NextMixedQkvConvWeight(w.q_conv1d, w.k_conv1d, w.v_conv1d, proj, K);
  std::vector<float>* conv_state = cache != nullptr ? &cache->conv_state : nullptr;
  if (cache != nullptr && cache->conv_state.empty())
    cache->conv_state.assign(static_cast<size_t>(conv_dim) * K, 0.0f);
  const std::vector<float> mixed_conv = Glm5NextMixedQkvConv(
      mixed, conv_w, T, conv_dim, K, conv_state, d.hidden_act);

  // Split back to q, k, v and l2-normalize q and k
  // (use_qk_l2norm_in_kernel=True at :722/:734, applied at :458-459 in fp32).
  std::vector<float> q(static_cast<size_t>(T) * proj), k(q.size()), v(q.size());
  for (int64_t t = 0; t < T; ++t) {
    const float* row = &mixed_conv[t * conv_dim];
    std::copy(row, row + proj, &q[t * proj]);
    std::copy(row + proj, row + 2 * proj, &k[t * proj]);
    std::copy(row + 2 * proj, row + 3 * proj, &v[t * proj]);
  }
  const std::vector<float> q_n = Glm5NextL2Norm(q, T * nh, hd);
  const std::vector<float> k_n = Glm5NextL2Norm(k, T * nh, hd);

  // g, beta and the output gate all read the PRE-CONV hidden states
  // (:709, :710, :742). Fusing any of them into the conv path changes them.
  const std::vector<float> g1 = Glm5NextLowRankProjection(
      hidden_states, w.f_a_proj, w.f_b_proj, T, H, nh, hd);
  const std::vector<float> g = Glm5NextForgetGate(g1, w.a_log, w.dt_bias, T, nh,
                                                  hd, d.gate_lower_bound);
  const std::vector<float> beta_raw =
      MatVecRows(w.b_proj, hidden_states, nh, H, T, "b_proj");
  std::vector<float> beta(beta_raw.size());
  for (size_t i = 0; i < beta.size(); ++i)
    beta[i] = static_cast<float>(Sigmoid(static_cast<double>(beta_raw[i])));
  const std::vector<float> gate_a =
      MatVecRows(w.g_a_proj, hidden_states, hd, H, T, "g_a_proj");
  const std::vector<float> gate =
      MatVecRows(w.g_b_proj, gate_a, proj, hd, T, "g_b_proj");

  // The delta recurrence, through the SHARED SEAM. vt::KdaGatedDeltaRule takes
  // the already-computed per-K-channel log-decay, which is exactly what makes
  // it usable by BOTH forget-gate branches; vt::KdaChunkPrefill fuses the
  // softplus branch on device and cannot serve this model (header, O14).
  std::vector<float> state;
  if (cache != nullptr) {
    if (cache->recurrent_state.empty())
      cache->recurrent_state.assign(static_cast<size_t>(nh) * hd * hd, 0.0f);
    state = cache->recurrent_state;
  } else {
    state.assign(static_cast<size_t>(nh) * hd * hd, 0.0f);
  }
  VT_CHECK(static_cast<int64_t>(state.size()) == nh * hd * hd,
           "glm5_next kda: recurrent_state must be [num_heads, head_dim, head_dim]");
  std::vector<float> core(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};
  const vt::Device dev = queue.device;
  vt::Tensor t_out = MakeT(core.data(), vt::DType::kF32, dev, {T, nh, hd});
  vt::Tensor t_q = MakeT(const_cast<float*>(q_n.data()), vt::DType::kF32, dev, {T, nh, hd});
  vt::Tensor t_k = MakeT(const_cast<float*>(k_n.data()), vt::DType::kF32, dev, {T, nh, hd});
  vt::Tensor t_v = MakeT(const_cast<float*>(v.data()), vt::DType::kF32, dev, {T, nh, hd});
  vt::Tensor t_g = MakeT(const_cast<float*>(g.data()), vt::DType::kF32, dev, {T, nh, hd});
  vt::Tensor t_b = MakeT(beta.data(), vt::DType::kF32, dev, {T, nh});
  vt::Tensor t_s = MakeT(state.data(), vt::DType::kF32, dev, {1, nh, hd, hd});
  vt::Tensor t_qsl = MakeT(qsl.data(), vt::DType::kI32, dev, {2});
  vt::GdnArgs args;
  args.scale = static_cast<float>(std::pow(static_cast<double>(hd), -0.5));  // :464
  vt::KdaGatedDeltaRule(queue, t_out, t_q, t_k, t_v, t_g, t_b, t_s, t_qsl, args);

  // `update_recurrent_state(last_recurrent_state.to(torch.float32))` (:739):
  // the state is f32 whatever the model dtype is, because it is a running sum
  // over the whole sequence and a bf16 store has no way to shed the error.
  if (cache != nullptr) cache->recurrent_state = state;

  const std::vector<float> normed =
      Glm5NextRmsNormGated(core, gate, w.o_norm, T, nh, hd, d.rms_norm_eps,
                           d.activation_dtype);
  return MatVecRows(w.o_proj, normed, H, proj, T, "o_proj");
}

}  // namespace vllm::glm5_next_kda
