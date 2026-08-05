// Kimi-Linear CPU REFERENCE forward (W2-W6). This TU replaces the refuse-by-name
// host `KimiLinearModel::Forward` with a REAL, per-op-gated CPU reference that
// composes the whole 27-layer KDA/NoPE-MLA hybrid + 256-expert-MoE decoder from the
// already-landed host primitives, so the ONLY remaining correctness step is the e2e
// SACRED token golden on GB10 (spike §4/§8, the W0/W7 GPU gate). The DEVICE
// (born-on-the-runner) forward stays refuse-by-name in kimi_linear.cpp — the KDA
// device kernel (W3), the absorbed-MLA decode (W4), the grouped-MoE slabs (W5) and
// the het-KV runner wiring (W6) are that lane. Mirrors the DeepSeek-V4 cadence:
// deepseek_v4.cpp's `DeepseekV4ForwardHost` is likewise the host f32 reference the
// device kernels are gated against (deepseek_v4.cpp:2-101).
//
// ─── WHAT THIS COMPOSES (file:line on BOTH sides, @ pin 555967922) ──────────────
//   Decoder layer   <-  kimi_linear.py:353-378 (pre-norm residual stream: attn
//                       branch add + RMSNorm, mlp branch add + RMSNorm)
//   KDA layer       <-  kimi_gdn_linear_attn.py:233-268 (proj/conv/gate/o_norm) +
//                       :390-441 + fused_recurrent.py:122-149 (gated-delta
//                       recurrence: decay per k-channel, delta rule, S@q output);
//                       REUSES vllm::kimi_kda {KdaShortConv, L2NormRows,
//                       KdaLowRankDecay, KdaDecayGate, FusedRMSNormGated}
//   NoPE-MLA layer  <-  kimi_linear.py:180-285 + layers/mla.py forward; the
//                       UNABSORBED materialized-MHA reference (mla_attention.h),
//                       scaling qk_head_dim**-0.5 (kimi_linear.py:212), NO RoPE
//   MoE block       <-  kimi_linear.py:104-177 (sigmoid noaux_tc router +
//                       e_score_correction_bias + shared expert) + router
//                       grouped_topk_router.py:106-161 (num_expert_group=1 trivial)
//   Dense MLP       <-  kimi_linear.py:64-101 (KimiMLP SwiGLU)
#include "vllm/model_executor/models/kimi_linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "vllm/model_executor/models/kimi_kda.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double Silu(double x) { return x * Sigmoid(x); }

// y[T,out] = x[T,in] @ w^T, with w row-major [out,in] (torch Linear layout).
std::vector<float> MatMul(const std::vector<float>& w, const std::vector<float>& x,
                          int64_t out, int64_t in, int64_t T) {
  VT_CHECK(static_cast<int64_t>(w.size()) == out * in, "MatMul: weight size");
  VT_CHECK(static_cast<int64_t>(x.size()) == T * in, "MatMul: input size");
  std::vector<float> y(static_cast<size_t>(T) * out, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    const float* xt = &x[t * in];
    float* yt = &y[t * out];
    for (int64_t o = 0; o < out; ++o) {
      const float* wo = &w[o * in];
      double acc = 0.0;
      for (int64_t i = 0; i < in; ++i) acc += static_cast<double>(wo[i]) * xt[i];
      yt[o] = static_cast<float>(acc);
    }
  }
  return y;
}

// RMSNorm over the last axis (`dim`), affine weight required (all Kimi norms have
// one). variance = mean(x^2); out = x * rsqrt(var+eps) * weight. Per row.
std::vector<float> RmsNorm(const std::vector<float>& x,
                           const std::vector<float>& weight, int64_t rows,
                           int64_t dim, float eps) {
  VT_CHECK(static_cast<int64_t>(x.size()) == rows * dim, "RmsNorm: input size");
  VT_CHECK(static_cast<int64_t>(weight.size()) == dim, "RmsNorm: weight size");
  std::vector<float> y(static_cast<size_t>(rows) * dim, 0.0f);
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = &x[r * dim];
    double var = 0.0;
    for (int64_t d = 0; d < dim; ++d) var += static_cast<double>(xr[d]) * xr[d];
    var /= static_cast<double>(dim);
    const double rstd = 1.0 / std::sqrt(var + eps);
    float* yr = &y[r * dim];
    for (int64_t d = 0; d < dim; ++d)
      yr[d] = static_cast<float>(static_cast<double>(xr[d]) * rstd * weight[d]);
  }
  return y;
}

// silu(gate @ x) * (up @ x) -> down @ (...), row-major weights. Returns [T,hidden].
std::vector<float> SwiGLU(const std::vector<float>& gate,
                          const std::vector<float>& up,
                          const std::vector<float>& down,
                          const std::vector<float>& x, int64_t hidden, int64_t inter,
                          int64_t T) {
  const std::vector<float> g = MatMul(gate, x, inter, hidden, T);
  const std::vector<float> u = MatMul(up, x, inter, hidden, T);
  std::vector<float> h(static_cast<size_t>(T) * inter, 0.0f);
  for (size_t i = 0; i < h.size(); ++i)
    h[i] = static_cast<float>(Silu(g[i]) * static_cast<double>(u[i]));
  return MatMul(down, h, hidden, inter, T);
}

}  // namespace

// ─── (1) KDA linear-attention layer ──────────────────────────────────────────
std::vector<float> KimiKdaLayerForward(const KdaLayerHostWeights& w,
                                       const std::vector<float>& hidden_normed,
                                       const KimiLinearParams& p,
                                       int64_t num_tokens) {
  const int64_t H = p.hidden_size;
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;
  const int64_t K = p.kda_short_conv_kernel_size;
  const int64_t T = num_tokens;
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "KdaLayerForward: hidden size");

  // q/k/v projections -> silu short convs (kimi_gdn_linear_attn.py:240-242,:324-356).
  const std::vector<float> raw_q = MatMul(w.q_proj, hidden_normed, proj, H, T);
  const std::vector<float> raw_k = MatMul(w.k_proj, hidden_normed, proj, H, T);
  const std::vector<float> raw_v = MatMul(w.v_proj, hidden_normed, proj, H, T);
  const std::vector<float> q_c = kimi_kda::KdaShortConv(raw_q, w.q_conv, {}, T, proj, K);
  const std::vector<float> k_c = kimi_kda::KdaShortConv(raw_k, w.k_conv, {}, T, proj, K);
  const std::vector<float> v = kimi_kda::KdaShortConv(raw_v, w.v_conv, {}, T, proj, K);

  // per-head q/k L2-norm (use_qk_l2norm_in_kernel, kda.py:1511-1513): rows (t,h).
  const std::vector<float> q_n = kimi_kda::L2NormRows(q_c, T * nh, hd);
  const std::vector<float> k_n = kimi_kda::L2NormRows(k_c, T * nh, hd);

  // beta = sigmoid(b_proj(x)) per head (kimi_gdn_linear_attn.py:244).
  const std::vector<float> beta_raw = MatMul(w.b_proj, hidden_normed, nh, H, T);
  std::vector<float> beta(static_cast<size_t>(T) * nh);
  for (size_t i = 0; i < beta.size(); ++i)
    beta[i] = static_cast<float>(Sigmoid(beta_raw[i]));

  // g1 = f_b(f_a(x)) low-rank decay -> per-channel log-decay gate g (kda gate).
  const std::vector<float> g1 =
      kimi_kda::KdaLowRankDecay(hidden_normed, w.f_a_proj, w.f_b_proj, T, H, nh, hd);
  const std::vector<float> g =
      kimi_kda::KdaDecayGate(g1, w.a_log, w.dt_bias, T, nh, hd);  // [T,nh,hd]

  // g2 = g_b(g_a(x)) — the sigmoid output-norm gate (kimi_gdn_linear_attn.py:249).
  const std::vector<float> g_a = MatMul(w.g_a_proj, hidden_normed, hd, H, T);
  const std::vector<float> g2 = MatMul(w.g_b_proj, g_a, proj, hd, T);  // [T,nh*hd]

  // The gated-delta recurrence (fused_recurrent.py:122-149). State S[h] is [hd_v,hd_k]
  // (b_h[BV,BK]); scale = head_dim**-0.5 applied to q (kda.py:128-130).
  const double scale = std::pow(static_cast<double>(hd), -0.5);
  std::vector<double> S(static_cast<size_t>(nh) * hd * hd, 0.0);
  std::vector<float> core(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<double> u(static_cast<size_t>(hd));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < nh; ++h) {
      const int64_t base = t * proj + h * hd;
      const float* qn = &q_n[base];
      const float* kn = &k_n[base];
      const float* vv = &v[base];
      const float* gh = &g[base];  // per-k-channel log-decay
      const double b = beta[t * nh + h];
      double* Sp = &S[static_cast<size_t>(h) * hd * hd];  // Sp[v*hd + k]
      // decay: b_h *= exp(g_k) per k-channel.
      for (int64_t vd = 0; vd < hd; ++vd) {
        double* Sr = &Sp[vd * hd];
        for (int64_t k = 0; k < hd; ++k) Sr[k] *= std::exp(static_cast<double>(gh[k]));
      }
      // delta: u_v = (v - S@k) * beta ; then S += outer(u, k).
      for (int64_t vd = 0; vd < hd; ++vd) {
        const double* Sr = &Sp[vd * hd];
        double pred = 0.0;
        for (int64_t k = 0; k < hd; ++k) pred += Sr[k] * kn[k];
        u[static_cast<size_t>(vd)] = (static_cast<double>(vv[vd]) - pred) * b;
      }
      for (int64_t vd = 0; vd < hd; ++vd) {
        double* Sr = &Sp[vd * hd];
        const double uv = u[static_cast<size_t>(vd)];
        for (int64_t k = 0; k < hd; ++k) Sr[k] += uv * kn[k];
      }
      // output: o_v = S @ (q*scale).
      float* cr = &core[base];
      for (int64_t vd = 0; vd < hd; ++vd) {
        const double* Sr = &Sp[vd * hd];
        double o = 0.0;
        for (int64_t k = 0; k < hd; ++k) o += Sr[k] * (static_cast<double>(qn[k]) * scale);
        cr[vd] = static_cast<float>(o);
      }
    }
  }

  // sigmoid-gated output RMSNorm then o_proj (kimi_gdn_linear_attn.py:266-268).
  const std::vector<float> core_normed = kimi_kda::FusedRMSNormGated(
      core, g2, w.o_norm, T, nh, hd, kimi_kda::GatedNormActivation::kSigmoid,
      p.rms_norm_eps);
  return MatMul(w.o_proj, core_normed, H, proj, T);
}

// ─── (2) NoPE-MLA full-attention layer ───────────────────────────────────────
std::vector<float> KimiNoPEMlaLayerForward(const MlaLayerHostWeights& w,
                                           const std::vector<float>& hidden_normed,
                                           const KimiLinearParams& p,
                                           int64_t num_tokens) {
  const int64_t H = p.hidden_size;
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;  // qk_head_dim (192)
  const int64_t vh = p.v_head_dim;
  const int64_t L = p.kv_lora_rank;
  const int64_t T = num_tokens;
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NoPEMlaLayerForward: hidden size");

  // q_proj -> [T, nah*qk]; kv_a_proj_with_mqa -> [T, L+qr] (latent | shared k_pe).
  const std::vector<float> q = MatMul(w.q_proj, hidden_normed, nah * qk, H, T);
  const std::vector<float> latent =
      MatMul(w.kv_a_proj_with_mqa, hidden_normed, L + qr, H, T);
  // split latent -> kv_c[T,L] (normed), k_pe[T,qr] (shared, NOT normed, NOT rotated).
  std::vector<float> kv_c(static_cast<size_t>(T) * L);
  std::vector<float> k_pe(static_cast<size_t>(T) * qr);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t d = 0; d < L; ++d) kv_c[t * L + d] = latent[t * (L + qr) + d];
    for (int64_t d = 0; d < qr; ++d) k_pe[t * qr + d] = latent[t * (L + qr) + L + d];
  }
  const std::vector<float> kv_c_n = RmsNorm(kv_c, w.kv_a_layernorm, T, L, p.rms_norm_eps);
  // kv_b_proj -> [T, nah*(qn+vh)] -> per-head k_nope[qn], v[vh].
  const std::vector<float> kv = MatMul(w.kv_b_proj, kv_c_n, nah * (qn + vh), L, T);

  const double scale = std::pow(static_cast<double>(qk), -0.5);
  std::vector<float> out(static_cast<size_t>(T) * (nah * vh), 0.0f);
  std::vector<double> scores(static_cast<size_t>(T));
  for (int64_t h = 0; h < nah; ++h) {
    for (int64_t t = 0; t < T; ++t) {
      const float* q_nope = &q[t * (nah * qk) + h * qk];       // [qn]
      const float* q_pe = q_nope + qn;                          // [qr]
      // causal scores over s<=t.
      double mx = -INFINITY;
      for (int64_t s = 0; s <= t; ++s) {
        const float* k_nope = &kv[s * (nah * (qn + vh)) + h * (qn + vh)];  // [qn]
        const float* kpe = &k_pe[s * qr];
        double dot = 0.0;
        for (int64_t d = 0; d < qn; ++d) dot += static_cast<double>(q_nope[d]) * k_nope[d];
        for (int64_t d = 0; d < qr; ++d) dot += static_cast<double>(q_pe[d]) * kpe[d];
        dot *= scale;
        scores[static_cast<size_t>(s)] = dot;
        mx = std::max(mx, dot);
      }
      double sum = 0.0;
      for (int64_t s = 0; s <= t; ++s) {
        const double e = std::exp(scores[static_cast<size_t>(s)] - mx);
        scores[static_cast<size_t>(s)] = e;
        sum += e;
      }
      float* ot = &out[t * (nah * vh) + h * vh];
      for (int64_t d = 0; d < vh; ++d) {
        double acc = 0.0;
        for (int64_t s = 0; s <= t; ++s) {
          const float* vs = &kv[s * (nah * (qn + vh)) + h * (qn + vh) + qn];  // [vh]
          acc += (scores[static_cast<size_t>(s)] / sum) * static_cast<double>(vs[d]);
        }
        ot[d] = static_cast<float>(acc);
      }
    }
  }
  return MatMul(w.o_proj, out, H, nah * vh, T);
}

// ─── (3) sigmoid noaux_tc router (num_expert_group=1 trivial group) ──────────
KimiMoeRouting KimiMoeRoute(const MoeHostWeights& w,
                            const std::vector<float>& hidden_normed,
                            const KimiLinearParams& p, int64_t num_tokens) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t k = p.num_experts_per_token;
  const int64_t T = num_tokens;
  const bool has_bias = !w.e_score_correction_bias.empty();
  const std::vector<float> logits = MatMul(w.gate, hidden_normed, E, H, T);

  KimiMoeRouting r;
  r.ids.assign(static_cast<size_t>(T) * k, -1);
  r.weights.assign(static_cast<size_t>(T) * k, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> scores(static_cast<size_t>(E));
    std::vector<double> sel(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
      scores[static_cast<size_t>(e)] = Sigmoid(logits[t * E + e]);
      sel[static_cast<size_t>(e)] =
          scores[static_cast<size_t>(e)] +
          (has_bias ? static_cast<double>(w.e_score_correction_bias[e]) : 0.0);
    }
    // top-k on the biased selection score; tie -> lowest index (our convention).
    std::vector<int64_t> idx(static_cast<size_t>(E));
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
      if (sel[static_cast<size_t>(a)] != sel[static_cast<size_t>(b)])
        return sel[static_cast<size_t>(a)] > sel[static_cast<size_t>(b)];
      return a < b;
    });
    double denom = 0.0;
    for (int64_t j = 0; j < k; ++j) {
      const int64_t e = idx[static_cast<size_t>(j)];
      r.ids[t * k + j] = static_cast<int32_t>(e);
      r.weights[t * k + j] = static_cast<float>(scores[static_cast<size_t>(e)]);  // UNBIASED
      denom += scores[static_cast<size_t>(e)];
    }
    if (p.moe_renormalize) {
      if (!(denom > 0.0)) denom = 1.0;
      for (int64_t j = 0; j < k; ++j)
        r.weights[t * k + j] = static_cast<float>(r.weights[t * k + j] / denom);
    }
    if (p.routed_scaling_factor != 1.0) {
      for (int64_t j = 0; j < k; ++j)
        r.weights[t * k + j] =
            static_cast<float>(r.weights[t * k + j] * p.routed_scaling_factor);
    }
  }
  return r;
}

// ─── (3b) whole MoE block: router -> shared + routed experts ─────────────────
std::vector<float> KimiMoeBlockForward(const MoeHostWeights& w,
                                       const std::vector<float>& hidden_normed,
                                       const KimiLinearParams& p, int64_t num_tokens) {
  const int64_t H = p.hidden_size;
  const int64_t moe_i = p.moe_intermediate_size;
  const int64_t k = p.num_experts_per_token;
  const int64_t T = num_tokens;
  const KimiMoeRouting r = KimiMoeRoute(w, hidden_normed, p, T);

  std::vector<float> out(static_cast<size_t>(T) * H, 0.0f);
  // shared expert (always, no routing weight) — reduce_results=False upstream, so
  // it is simply added to the routed sum (KimiMoE.forward via FusedMoE).
  if (w.has_shared) {
    const int64_t shared_i = moe_i * p.num_shared_experts;
    const std::vector<float> s = SwiGLU(w.shared.gate_proj, w.shared.up_proj,
                                        w.shared.down_proj, hidden_normed, H, shared_i, T);
    for (size_t i = 0; i < out.size(); ++i) out[i] += s[i];
  }
  // routed experts: per token, sum weight * expert_ffn(x).
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> xt(hidden_normed.begin() + static_cast<long>(t * H),
                                hidden_normed.begin() + static_cast<long>((t + 1) * H));
    for (int64_t j = 0; j < k; ++j) {
      const int32_t e = r.ids[t * k + j];
      if (e < 0) continue;
      const double weight = r.weights[t * k + j];
      const MlpHostWeights& ex = w.experts[static_cast<size_t>(e)];
      const std::vector<float> y =
          SwiGLU(ex.gate_proj, ex.up_proj, ex.down_proj, xt, H, moe_i, 1);
      float* ot = &out[t * H];
      for (int64_t d = 0; d < H; ++d)
        ot[d] = static_cast<float>(ot[d] + weight * static_cast<double>(y[d]));
    }
  }
  return out;
}

// ─── (4) dense layer-0 SwiGLU MLP ────────────────────────────────────────────
std::vector<float> KimiDenseMlpForward(const MlpHostWeights& w,
                                       const std::vector<float>& hidden_normed,
                                       const KimiLinearParams& p, int64_t num_tokens) {
  return SwiGLU(w.gate_proj, w.up_proj, w.down_proj, hidden_normed, p.hidden_size,
                p.intermediate_size, num_tokens);
}

namespace {

// The whole 27-layer decoder over a single token sequence, fresh state — the
// pre-norm residual stream (kimi_linear.py:353-378): each branch adds its output to
// the residual, the norm sees the accumulated residual. Returns per-position logits
// [num_out, vocab] for the requested positions (all if `logits_indices` empty).
std::vector<float> HostForwardSeq(const KimiLinearHostWeights& host,
                                  const KimiLinearParams& p,
                                  const std::vector<int32_t>& token_ids,
                                  const std::vector<int32_t>& logits_indices) {
  VT_CHECK(host.materialized,
           "KimiLinear CPU reference forward: host weights are not materialized "
           "(load via LoadKimiLinearForCausalLMWeights, or build the synthetic host "
           "weights for a unit gate). The device forward is the born-on-runner "
           "W6/W7 residual (refuse-by-name).");
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t L = p.num_hidden_layers;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "KimiLinear forward: empty token sequence");
  VT_CHECK(static_cast<int64_t>(host.layers.size()) == L,
           "KimiLinear forward: host layer count != num_hidden_layers");

  // embed lookup -> residual stream [T,H].
  std::vector<float> h(static_cast<size_t>(T) * H, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    const int32_t tok = token_ids[static_cast<size_t>(t)];
    VT_CHECK(tok >= 0 && tok < V, "KimiLinear forward: token id out of range");
    const float* row = &host.embed_tokens[static_cast<size_t>(tok) * H];
    std::copy(row, row + H, &h[t * H]);
  }

  for (int64_t l = 0; l < L; ++l) {
    const KimiLinearLayerHostWeights& lw = host.layers[static_cast<size_t>(l)];
    // attention branch: normed = RMSNorm(h); h += attn(normed).
    const std::vector<float> a_in = RmsNorm(h, lw.input_layernorm, T, H, p.rms_norm_eps);
    const std::vector<float> attn = lw.is_kda
        ? KimiKdaLayerForward(lw.kda, a_in, p, T)
        : KimiNoPEMlaLayerForward(lw.mla, a_in, p, T);
    for (size_t i = 0; i < h.size(); ++i) h[i] += attn[i];
    // mlp branch: normed = RMSNorm(h); h += mlp(normed).
    const std::vector<float> m_in =
        RmsNorm(h, lw.post_attention_layernorm, T, H, p.rms_norm_eps);
    const std::vector<float> mlp = lw.is_moe ? KimiMoeBlockForward(lw.moe, m_in, p, T)
                                             : KimiDenseMlpForward(lw.dense, m_in, p, T);
    for (size_t i = 0; i < h.size(); ++i) h[i] += mlp[i];
  }
  const std::vector<float> hn = RmsNorm(h, host.final_norm, T, H, p.rms_norm_eps);

  // logits = lm_head @ hn for the requested positions.
  std::vector<int64_t> want;
  if (logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    std::iota(want.begin(), want.end(), 0);
  } else {
    for (int32_t idx : logits_indices) {
      VT_CHECK(idx >= 0 && idx < T, "KimiLinear forward: logits index out of range");
      want.push_back(idx);
    }
  }
  std::vector<float> logits(want.size() * static_cast<size_t>(V), 0.0f);
  for (size_t r = 0; r < want.size(); ++r) {
    const float* hr = &hn[want[r] * H];
    float* lr = &logits[r * V];
    for (int64_t o = 0; o < V; ++o) {
      const float* wo = &host.lm_head[static_cast<size_t>(o) * H];
      double acc = 0.0;
      for (int64_t i = 0; i < H; ++i) acc += static_cast<double>(wo[i]) * hr[i];
      lr[o] = static_cast<float>(acc);
    }
  }
  return logits;
}

}  // namespace

std::vector<int32_t> KimiLinearGreedyDecode(const KimiLinearHostWeights& host,
                                            const KimiLinearParams& p,
                                            const std::vector<int32_t>& prompt,
                                            int num_new) {
  std::vector<int32_t> seq = prompt;
  std::vector<int32_t> out;
  const int64_t V = p.vocab_size;
  for (int step = 0; step < num_new; ++step) {
    // Only the last position's logits are needed each step; the KDA recurrent state
    // + MLA causal context are (re)advanced over the growing sequence (the device
    // incremental-cache decode is the born-on-runner W6/W7 residual).
    const std::vector<float> logits = HostForwardSeq(
        host, p, seq, {static_cast<int32_t>(seq.size() - 1)});
    int32_t best = 0;
    float best_v = logits[0];
    for (int64_t o = 1; o < V; ++o)
      if (logits[static_cast<size_t>(o)] > best_v) {
        best_v = logits[static_cast<size_t>(o)];
        best = static_cast<int32_t>(o);
      }
    out.push_back(best);
    seq.push_back(best);
  }
  return out;
}

std::vector<float> KimiLinearModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  // The CPU reference manages its own (fresh, single-sequence) recurrent + latent
  // context, so the runner's paged-KV / positions / queue are unused here — the
  // device runner path (paged KV, het-KV groups, on-GPU sampling) is ForwardDevice
  // (born-on-runner, W6/W7, refuse-by-name).
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)queue;
  return HostForwardSeq(weights.host, weights.params, token_ids, logits_indices);
}

}  // namespace vllm
