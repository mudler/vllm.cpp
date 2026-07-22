// The MLA attention block + weight absorption — MLA campaign W6 gate.
//
// Upstream test modules ported per .agents/test-porting.md:
//   * vllm/tests/kernels/attention/test_mla_decode_cpu.py @ pin e24d1b24 — the
//     device-independent MLA numerical oracle. Its `ref_mla` two-pass softmax is
//     the shape of `RefAttend` below, lifted to the WHOLE BLOCK (projections,
//     norms, decoupled RoPE, attention, o_proj) rather than the kernel alone.
//   * vllm/tests/v1/attention/test_mla_backends.py — "MLA backend correctness vs
//     a reference across batch/seq shapes"; its ragged/multi-request sweep is
//     what the decode-only / prefill-only / MIXED cases below cover.
//   * vllm/tests/v1/attention/test_mla_prefill_quant_output.py:158 — the source
//     of the V2-Lite prefill dims (16 query heads, v_head_dim 128).
//
// Logic under gate (upstream `file:line` on the other side):
//   mla_attention.py:875-962  process_weights_after_loading  -> AbsorbKvBProjBf16
//   mla_attention.py:700-830  forward_impl dispatch+absorbed decode
//   mla_attention.py:2344-2425 forward_mha (materialized prefill)
//   mla.py:119-181            MultiHeadLatentAttentionWrapper.forward
//   deepseek_scaling_rope.py:20-23,76-118  yarn_get_mscale + the cos|sin cache
//   deepseek_v2.py:981-996,1067-1075       the mscale^2 softmax scale
//
// ─── WHY THIS FILE EXISTS, in one sentence ──────────────────────────────────
// W6's whole claim is that the ABSORBED decode form (MQA, QK 576 / V 512, ONE
// KV head, K and V NEVER materialized) computes the same function as the
// UNABSORBED materialized form (MHA, QK 192 / V 128, N KV heads). That claim is
// PROVEN here numerically, three independent ways, not argued:
//   (1) ORACLE-INTERNAL — an independent double-precision reference computes the
//       attention BOTH ways from the same weights and the two agree to ~1e-12,
//       which is the mathematical identity itself;
//   (2) OURS-vs-ORACLE — our absorbed decode path reproduces the UNABSORBED
//       double oracle;
//   (3) OURS-vs-OURS, THROUGH TWO DIFFERENT CODE PATHS — the SAME batch is run
//       once labelled all-DECODE (our absorbed MQA kernel over the 576-wide
//       latent) and once labelled all-PREFILL (our materialized MHA over
//       per-head K/V at QK 192), and the two outputs must agree. Nothing is
//       shared between those two paths except the weights, so an absorption bug
//       cannot cancel out.
//
// Real geometry throughout: DeepSeek-V2-Lite (W0-confirmed against the shipped
// config.json) kv_lora_rank 512, qk_nope 128, qk_rope 64, v_head_dim 128, latent
// 576, 16 heads, block_size 16, `q_lora_rank = null`; plus a DeepSeek-V3 case
// (hidden 7168, 128 heads, `q_lora_rank = 1536`) that is the ONLY coverage the
// lora query branch gets — see the coverage note on that test case.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/mla_attention.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm::dense_attn::Dev;
using vllm::mla::AbsorbedKvBProj;
using vllm::mla::AbsorbKvBProjBf16;
using vllm::mla::BuildDeepseekRopeCosSinCache;
using vllm::mla::DeepseekYarnRopeParams;
using vllm::mla::ForwardMlaAttentionBlock;
using vllm::mla::MlaAttentionScale;
using vllm::mla::MlaBlockDims;
using vllm::mla::MlaBlockMetadata;
using vllm::mla::MlaBlockWeights;
using vllm::mla::MlaChunkDeviceMetadata;
using vllm::mla::YarnGetMscale;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

std::vector<float> RandF32(size_t n, uint32_t seed, float amp = 1.0f) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = ((static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f) * amp;
  }
  return v;
}

// ─── DeepSeek-V2-Lite geometry (W0-confirmed) ───────────────────────────────
constexpr int64_t kBlockSize = 16;

MlaBlockDims LiteDims() {
  MlaBlockDims d;
  d.hidden_size = 2048;
  d.num_heads = 16;
  d.qk_nope_head_dim = 128;
  d.qk_rope_head_dim = 64;
  d.v_head_dim = 128;
  d.kv_lora_rank = 512;
  d.q_lora_rank = 0;  // null — the DIRECT q_proj branch
  d.rms_norm_eps = 1e-6f;
  return d;
}

// DeepSeek-V3 (the config the spike records at §5.1). The MLA geometry is
// IDENTICAL to V2-Lite; only hidden_size, num_heads and q_lora_rank differ.
MlaBlockDims V3Dims() {
  MlaBlockDims d = LiteDims();
  d.hidden_size = 7168;
  d.num_heads = 128;
  d.q_lora_rank = 1536;
  return d;
}

DeepseekYarnRopeParams LiteRope() {
  DeepseekYarnRopeParams p;
  p.base = 10000.0;
  p.scaling_factor = 40.0;
  p.mscale = 0.707;
  p.mscale_all_dim = 0.707;
  p.beta_fast = 32.0;
  p.beta_slow = 1.0;
  p.rotary_dim = 64;
  p.original_max_position_embeddings = 4096;
  p.yarn = true;
  return p;
}

// ─── host-side weight set, kept in f32 so the oracle and the device tensors
//     are built from EXACTLY the same numbers ─────────────────────────────────
struct HostWeights {
  std::vector<float> fused_qkv_a;  // [ql + L + R, H]
  std::vector<float> q_a_ln;       // [ql]
  std::vector<float> q_b;          // [N*Dqk, ql]
  std::vector<float> kv_a;         // [L + R, H]
  std::vector<float> q_proj;       // [N*Dqk, H]
  std::vector<float> kv_a_ln;      // [L]
  std::vector<float> kv_b;         // [N*(P+V), L]
  std::vector<float> o_proj;       // [H, N*V]
  std::vector<float> w_uk_t;       // [N, P, L]
  std::vector<float> w_uv;         // [N, L, V]
  std::vector<float> cos_sin;      // [rows, R]
};

std::vector<float> RoundBf16(const std::vector<float>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::BF16ToF32(vt::F32ToBF16(v[i]));
  return o;
}

// Build the weight set, then ROUND every projection weight to bf16 and keep the
// rounded values — so the oracle sees the exact bits the device does and the
// only remaining difference is accumulation precision.
HostWeights MakeWeights(const MlaBlockDims& d, const DeepseekYarnRopeParams& rp,
                        int64_t rope_rows, uint32_t seed) {
  const int64_t H = d.hidden_size, N = d.num_heads, L = d.kv_lora_rank;
  const int64_t P = d.qk_nope_head_dim, R = d.qk_rope_head_dim, V = d.v_head_dim;
  const int64_t Dqk = d.qk_head_dim(), ql = d.q_lora_rank;
  HostWeights w;
  // Small amplitudes keep the pre-softmax logits in a sane range at these widths.
  const float amp = 0.05f;
  if (d.has_q_lora()) {
    w.fused_qkv_a = RoundBf16(RandF32(static_cast<size_t>((ql + L + R) * H), seed, amp));
    w.q_a_ln = RoundBf16(RandF32(static_cast<size_t>(ql), seed + 1, 0.5f));
    w.q_b = RoundBf16(RandF32(static_cast<size_t>(N * Dqk * ql), seed + 2, amp));
  } else {
    w.kv_a = RoundBf16(RandF32(static_cast<size_t>((L + R) * H), seed + 3, amp));
    w.q_proj = RoundBf16(RandF32(static_cast<size_t>(N * Dqk * H), seed + 4, amp));
  }
  w.kv_a_ln = RoundBf16(RandF32(static_cast<size_t>(L), seed + 5, 0.5f));
  w.kv_b = RoundBf16(RandF32(static_cast<size_t>(N * (P + V) * L), seed + 6, amp));
  w.o_proj = RoundBf16(RandF32(static_cast<size_t>(H * N * V), seed + 7, amp));
  // The ABSORBED forms come from the SAME kv_b weight, through the production
  // transform — the test never builds W_UK/W_UV independently, so a bug in
  // AbsorbKvBProjBf16 shows up in every downstream case too (and is isolated by
  // its own test case below).
  std::vector<uint16_t> kv_b_bf16(w.kv_b.size());
  for (size_t i = 0; i < w.kv_b.size(); ++i) kv_b_bf16[i] = vt::F32ToBF16(w.kv_b[i]);
  AbsorbedKvBProj ab = AbsorbKvBProjBf16(kv_b_bf16.data(), d);
  w.w_uk_t.resize(ab.w_uk_t.size());
  for (size_t i = 0; i < ab.w_uk_t.size(); ++i) w.w_uk_t[i] = vt::BF16ToF32(ab.w_uk_t[i]);
  w.w_uv.resize(ab.w_uv.size());
  for (size_t i = 0; i < ab.w_uv.size(); ++i) w.w_uv[i] = vt::BF16ToF32(ab.w_uv[i]);
  w.cos_sin = BuildDeepseekRopeCosSinCache(rp, rope_rows);
  return w;
}

// ─── the per-step batch layout ──────────────────────────────────────────────
struct Request {
  int64_t ctx_len = 0;  // tokens already in the KV cache
  int64_t q_len = 1;    // NEW tokens this step
};

// ─── the INDEPENDENT double-precision block oracle ──────────────────────────
// Written from the upstream FORMULAS, not from our implementation. `absorbed`
// selects which of the two mathematically-equal attention routes to take:
//   false — UNABSORBED: materialize k_nope = W_UK^T kv_c and v = W_UV^T kv_c per
//           head, then MHA at QK 192 / V 128 (mla_attention.py:66-89, :2371-2375)
//   true  — ABSORBED:   fold W_UK into the query (ql_nope = q_nope W_UK_T), MQA
//           against the raw 576-wide latent, then un-project with W_UV
//           (mla_attention.py:94-117, :775-789, :1024-1034)
// The softmax is TWO-PASS (max, exp-sum, weighted sum) — deliberately a
// different algorithm from the streaming online-softmax both of our kernels use.
struct RefContext {
  // The already-cached context, per request: [ctx_len, 576] latent rows.
  std::vector<std::vector<double>> ctx_rows;
};

std::vector<double> RefBlock(const MlaBlockDims& d, const HostWeights& w,
                             const std::vector<float>& hidden,
                             const std::vector<int32_t>& positions,
                             const std::vector<Request>& reqs, const RefContext& ctx,
                             bool absorbed) {
  const int64_t H = d.hidden_size, N = d.num_heads, L = d.kv_lora_rank;
  const int64_t P = d.qk_nope_head_dim, R = d.qk_rope_head_dim, V = d.v_head_dim;
  const int64_t Dqk = d.qk_head_dim(), ql = d.q_lora_rank;
  int64_t T = 0;
  for (const Request& r : reqs) T += r.q_len;

  // ---- projections (mla.py:126-153) ----
  auto matvec_bt = [](const std::vector<float>& wt, const double* x, int64_t nout,
                      int64_t k, double* out) {
    for (int64_t o = 0; o < nout; ++o) {
      double acc = 0.0;
      const float* row = wt.data() + o * k;
      for (int64_t j = 0; j < k; ++j) acc += static_cast<double>(row[j]) * x[j];
      out[o] = acc;
    }
  };
  auto rmsnorm = [](std::vector<double>& x, const std::vector<float>& weight, double eps) {
    double ss = 0.0;
    for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * inv * static_cast<double>(weight[i]);
  };

  std::vector<std::vector<double>> q_all(static_cast<size_t>(T));       // [T][N*Dqk]
  std::vector<std::vector<double>> kv_c_all(static_cast<size_t>(T));    // [T][L]
  std::vector<std::vector<double>> k_pe_all(static_cast<size_t>(T));    // [T][R]
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> x(static_cast<size_t>(H));
    for (int64_t i = 0; i < H; ++i) x[static_cast<size_t>(i)] = hidden[static_cast<size_t>(t * H + i)];
    std::vector<double> kv_c(static_cast<size_t>(L)), k_pe(static_cast<size_t>(R));
    std::vector<double> q(static_cast<size_t>(N * Dqk));
    if (d.has_q_lora()) {
      // `fused_qkv_a_proj` rows: [q_a | kv_a_latent | kv_a_rope].
      std::vector<double> q_c(static_cast<size_t>(ql));
      std::vector<double> a(static_cast<size_t>(ql + L + R));
      matvec_bt(w.fused_qkv_a, x.data(), ql + L + R, H, a.data());
      for (int64_t i = 0; i < ql; ++i) q_c[static_cast<size_t>(i)] = a[static_cast<size_t>(i)];
      for (int64_t i = 0; i < L; ++i) kv_c[static_cast<size_t>(i)] = a[static_cast<size_t>(ql + i)];
      for (int64_t i = 0; i < R; ++i) k_pe[static_cast<size_t>(i)] = a[static_cast<size_t>(ql + L + i)];
      rmsnorm(q_c, w.q_a_ln, d.rms_norm_eps);  // mla.py:143
      matvec_bt(w.q_b, q_c.data(), N * Dqk, ql, q.data());
    } else {
      std::vector<double> a(static_cast<size_t>(L + R));
      matvec_bt(w.kv_a, x.data(), L + R, H, a.data());
      for (int64_t i = 0; i < L; ++i) kv_c[static_cast<size_t>(i)] = a[static_cast<size_t>(i)];
      for (int64_t i = 0; i < R; ++i) k_pe[static_cast<size_t>(i)] = a[static_cast<size_t>(L + i)];
      matvec_bt(w.q_proj, x.data(), N * Dqk, H, q.data());
    }
    // `kv_a_layernorm` over the LATENT ONLY — the rope part is NOT normed
    // (deepseek_v2.py:516).
    rmsnorm(kv_c, w.kv_a_ln, d.rms_norm_eps);

    // ---- decoupled RoPE, is_neox_style=False (adjacent-pair GPT-J) ----
    const int64_t pos = positions[static_cast<size_t>(t)];
    const int64_t half = R / 2;
    const float* cs = w.cos_sin.data() + pos * R;
    for (int64_t h = 0; h < N; ++h) {
      double* qp = q.data() + h * Dqk + P;  // the TRAILING rope slice
      for (int64_t i = 0; i < half; ++i) {
        const double c = cs[i], s = cs[half + i];
        const double x0 = qp[2 * i], x1 = qp[2 * i + 1];
        qp[2 * i] = x0 * c - x1 * s;
        qp[2 * i + 1] = x0 * s + x1 * c;
      }
    }
    for (int64_t i = 0; i < half; ++i) {
      const double c = cs[i], s = cs[half + i];
      const double x0 = k_pe[static_cast<size_t>(2 * i)], x1 = k_pe[static_cast<size_t>(2 * i + 1)];
      k_pe[static_cast<size_t>(2 * i)] = x0 * c - x1 * s;
      k_pe[static_cast<size_t>(2 * i + 1)] = x0 * s + x1 * c;
    }
    q_all[static_cast<size_t>(t)] = std::move(q);
    kv_c_all[static_cast<size_t>(t)] = std::move(kv_c);
    k_pe_all[static_cast<size_t>(t)] = std::move(k_pe);
  }

  // ---- attention ----
  std::vector<double> attn(static_cast<size_t>(T * N * V), 0.0);
  int64_t tok = 0;
  for (size_t ri = 0; ri < reqs.size(); ++ri) {
    const Request& r = reqs[ri];
    // The full key sequence: cached context rows, then this step's new rows.
    std::vector<std::vector<double>> latent, rope;
    const int64_t C = r.ctx_len;
    latent.reserve(static_cast<size_t>(C + r.q_len));
    rope.reserve(static_cast<size_t>(C + r.q_len));
    for (int64_t s = 0; s < C; ++s) {
      const double* row = ctx.ctx_rows[ri].data() + s * (L + R);
      latent.emplace_back(row, row + L);
      rope.emplace_back(row + L, row + L + R);
    }
    for (int64_t i = 0; i < r.q_len; ++i) {
      latent.push_back(kv_c_all[static_cast<size_t>(tok + i)]);
      rope.push_back(k_pe_all[static_cast<size_t>(tok + i)]);
    }
    const int64_t S = C + r.q_len;
    for (int64_t i = 0; i < r.q_len; ++i) {
      const std::vector<double>& q = q_all[static_cast<size_t>(tok + i)];
      const int64_t last = C + i;  // BOTTOM-RIGHT causal alignment
      for (int64_t h = 0; h < N; ++h) {
        const double* q_nope = q.data() + h * Dqk;
        const double* q_pe = q.data() + h * Dqk + P;
        // The absorbed query lives in latent space: ql_nope[l] = sum_p q_nope[p]
        // * W_UK_T[h,p,l]  (mla_attention.py:789).
        std::vector<double> ql_nope;
        if (absorbed) {
          ql_nope.assign(static_cast<size_t>(L), 0.0);
          for (int64_t p = 0; p < P; ++p) {
            const float* wr = w.w_uk_t.data() + (h * P + p) * L;
            const double qv = q_nope[p];
            for (int64_t l = 0; l < L; ++l) ql_nope[static_cast<size_t>(l)] += qv * wr[l];
          }
        }
        std::vector<double> logits(static_cast<size_t>(last + 1));
        double mx = -std::numeric_limits<double>::infinity();
        for (int64_t s = 0; s <= last; ++s) {
          double dot = 0.0;
          if (absorbed) {
            for (int64_t l = 0; l < L; ++l)
              dot += ql_nope[static_cast<size_t>(l)] * latent[static_cast<size_t>(s)][static_cast<size_t>(l)];
          } else {
            // Materialize k_nope[h,s][p] = sum_l latent[s][l] * W_UK_T[h,p,l].
            for (int64_t p = 0; p < P; ++p) {
              const float* wr = w.w_uk_t.data() + (h * P + p) * L;
              double kn = 0.0;
              for (int64_t l = 0; l < L; ++l)
                kn += latent[static_cast<size_t>(s)][static_cast<size_t>(l)] * wr[l];
              dot += q_nope[p] * kn;
            }
          }
          for (int64_t j = 0; j < R; ++j)
            dot += q_pe[j] * rope[static_cast<size_t>(s)][static_cast<size_t>(j)];
          logits[static_cast<size_t>(s)] = dot * static_cast<double>(d.scale);
          mx = std::max(mx, logits[static_cast<size_t>(s)]);
        }
        double denom = 0.0;
        for (int64_t s = 0; s <= last; ++s) {
          logits[static_cast<size_t>(s)] = std::exp(logits[static_cast<size_t>(s)] - mx);
          denom += logits[static_cast<size_t>(s)];
        }
        double* dst = attn.data() + ((tok + i) * N + h) * V;
        if (absorbed) {
          // out_latent = sum_s p_s latent_s ; then W_UV un-projects it.
          std::vector<double> acc(static_cast<size_t>(L), 0.0);
          for (int64_t s = 0; s <= last; ++s) {
            const double p = logits[static_cast<size_t>(s)] / denom;
            for (int64_t l = 0; l < L; ++l)
              acc[static_cast<size_t>(l)] += p * latent[static_cast<size_t>(s)][static_cast<size_t>(l)];
          }
          for (int64_t vi = 0; vi < V; ++vi) {
            double o = 0.0;
            for (int64_t l = 0; l < L; ++l)
              o += acc[static_cast<size_t>(l)] * w.w_uv[static_cast<size_t>((h * L + l) * V + vi)];
            dst[vi] = o;
          }
        } else {
          // out = sum_s p_s v_s with v_s materialized per head.
          for (int64_t vi = 0; vi < V; ++vi) dst[vi] = 0.0;
          for (int64_t s = 0; s <= last; ++s) {
            const double p = logits[static_cast<size_t>(s)] / denom;
            for (int64_t vi = 0; vi < V; ++vi) {
              double vv = 0.0;
              for (int64_t l = 0; l < L; ++l)
                vv += latent[static_cast<size_t>(s)][static_cast<size_t>(l)] *
                      w.w_uv[static_cast<size_t>((h * L + l) * V + vi)];
              dst[vi] += p * vv;
            }
          }
        }
      }
    }
    tok += r.q_len;
    (void)S;
  }

  // ---- o_proj ----
  std::vector<double> out(static_cast<size_t>(T * H), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    const double* a = attn.data() + t * N * V;
    for (int64_t o = 0; o < H; ++o) {
      double acc = 0.0;
      const float* row = w.o_proj.data() + o * (N * V);
      for (int64_t j = 0; j < N * V; ++j) acc += static_cast<double>(row[j]) * a[j];
      out[static_cast<size_t>(t * H + o)] = acc;
    }
  }
  return out;
}

// ─── device plumbing for the block ──────────────────────────────────────────
// Owns every device buffer so a test case is a few lines.
class BlockHarness {
 public:
  BlockHarness(Backend& b, Queue& q, const MlaBlockDims& d, const HostWeights& hw,
               DType dt, int64_t num_blocks)
      : b_(b), q_(q), d_(d), dt_(dt), dev_{b, q} {
    const int64_t H = d.hidden_size, N = d.num_heads, L = d.kv_lora_rank;
    const int64_t P = d.qk_nope_head_dim, R = d.qk_rope_head_dim, V = d.v_head_dim;
    const int64_t Dqk = d.qk_head_dim(), ql = d.q_lora_rank;
    if (d.has_q_lora()) {
      w_.fused_qkv_a_proj = Up(hw.fused_qkv_a, {ql + L + R, H});
      w_.q_a_layernorm = Up(hw.q_a_ln, {ql});
      w_.q_b_proj = Up(hw.q_b, {N * Dqk, ql});
    } else {
      w_.kv_a_proj_with_mqa = Up(hw.kv_a, {L + R, H});
      w_.q_proj = Up(hw.q_proj, {N * Dqk, H});
    }
    w_.kv_a_layernorm = Up(hw.kv_a_ln, {L});
    w_.kv_b_proj = Up(hw.kv_b, {N * (P + V), L});
    w_.w_uk_t = Up(hw.w_uk_t, {N, P, L});
    w_.w_uv = Up(hw.w_uv, {N, L, V});
    w_.o_proj = Up(hw.o_proj, {H, N * V});
    w_.rope_cos_sin_cache =
        Up(hw.cos_sin, {static_cast<int64_t>(hw.cos_sin.size()) / R, R});
    kv_cache_ = Alloc(dt, {num_blocks, kBlockSize, L + R});
  }

  Dev dev() { return dev_; }
  MlaBlockWeights& weights() { return w_; }
  Tensor& kv_cache() { return kv_cache_; }

  // Upload an f32 host vector in the harness dtype.
  Tensor Up(const std::vector<float>& v, const std::vector<int64_t>& shape) {
    return UpAs(v, shape, dt_);
  }
  Tensor UpAs(const std::vector<float>& v, const std::vector<int64_t>& shape, DType dt) {
    Tensor t = Alloc(dt, shape);
    if (dt == DType::kF32) {
      b_.Copy(q_, t.data, v.data(), v.size() * sizeof(float));
    } else {
      std::vector<uint16_t> bf(v.size());
      for (size_t i = 0; i < v.size(); ++i) bf[i] = vt::F32ToBF16(v[i]);
      b_.Copy(q_, t.data, bf.data(), bf.size() * sizeof(uint16_t));
    }
    return t;
  }
  Tensor UpI32(const std::vector<int32_t>& v, const std::vector<int64_t>& shape) {
    Tensor t = Alloc(DType::kI32, shape);
    b_.Copy(q_, t.data, v.data(), v.size() * sizeof(int32_t));
    return t;
  }
  Tensor UpI64(const std::vector<int64_t>& v, const std::vector<int64_t>& shape) {
    Tensor t = Alloc(DType::kI64, shape);
    b_.Copy(q_, t.data, v.data(), v.size() * sizeof(int64_t));
    return t;
  }
  Tensor Alloc(DType dt, const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    const size_t bytes = static_cast<size_t>(numel) * vt::SizeOf(dt);
    void* p = b_.Alloc(bytes == 0 ? 1 : bytes);
    owned_.push_back(p);
    Tensor t;
    t.data = p;
    t.dtype = dt;
    t.device = q_.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = t.rank - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    return t;
  }
  std::vector<float> Down(const Tensor& t) {
    const int64_t n = t.Numel();
    std::vector<float> out(static_cast<size_t>(n));
    if (t.dtype == DType::kF32) {
      b_.Copy(q_, out.data(), t.data, out.size() * sizeof(float));
      b_.Synchronize(q_);
    } else {
      std::vector<uint16_t> bf(static_cast<size_t>(n));
      b_.Copy(q_, bf.data(), t.data, bf.size() * sizeof(uint16_t));
      b_.Synchronize(q_);
      for (size_t i = 0; i < bf.size(); ++i) out[i] = vt::BF16ToF32(bf[i]);
    }
    return out;
  }
  std::vector<uint8_t> DownRaw(const Tensor& t) {
    std::vector<uint8_t> out(static_cast<size_t>(t.Numel()) * vt::SizeOf(t.dtype));
    b_.Copy(q_, out.data(), t.data, out.size());
    b_.Synchronize(q_);
    return out;
  }
  ~BlockHarness() {
    for (void* p : owned_) b_.Free(p);
  }
  BlockHarness(const BlockHarness&) = delete;
  BlockHarness& operator=(const BlockHarness&) = delete;

 private:
  Backend& b_;
  Queue& q_;
  MlaBlockDims d_;
  DType dt_;
  Dev dev_;
  MlaBlockWeights w_{};
  Tensor kv_cache_{};
  std::vector<void*> owned_;
};

// The paged layout for a batch: an ADVERSARIAL reverse-interleaved block table
// (the W4/W5 hardening — upstream's own tests use `arange` and cannot catch a
// page-stride assumption).
struct Paging {
  int64_t max_blocks = 0;
  std::vector<int32_t> block_table;   // [num_reqs, max_blocks]
  std::vector<int64_t> slot_mapping;  // [T], for the NEW tokens
  std::vector<int32_t> seq_lens;      // [num_reqs] = ctx + q_len
  int64_t num_blocks = 0;
};

Paging MakePaging(const std::vector<Request>& reqs) {
  Paging p;
  int64_t max_seq = 0;
  for (const Request& r : reqs) max_seq = std::max(max_seq, r.ctx_len + r.q_len);
  p.max_blocks = (max_seq + kBlockSize - 1) / kBlockSize;
  const int64_t nreq = static_cast<int64_t>(reqs.size());
  p.num_blocks = nreq * p.max_blocks + 3;  // +3 unused pages, so index 0 is not "the" page
  p.block_table.assign(static_cast<size_t>(nreq * p.max_blocks), 0);
  for (int64_t r = 0; r < nreq; ++r) {
    for (int64_t j = 0; j < p.max_blocks; ++j) {
      // Reverse-interleaved: descending, non-contiguous physical pages.
      p.block_table[static_cast<size_t>(r * p.max_blocks + j)] =
          static_cast<int32_t>(p.num_blocks - 1 - (j * nreq + r));
    }
  }
  for (size_t r = 0; r < reqs.size(); ++r) {
    p.seq_lens.push_back(static_cast<int32_t>(reqs[r].ctx_len + reqs[r].q_len));
    for (int64_t i = 0; i < reqs[r].q_len; ++i) {
      const int64_t s = reqs[r].ctx_len + i;
      const int32_t blk =
          p.block_table[static_cast<size_t>(static_cast<int64_t>(r) * p.max_blocks +
                                            s / kBlockSize)];
      p.slot_mapping.push_back(static_cast<int64_t>(blk) * kBlockSize + s % kBlockSize);
    }
  }
  return p;
}

double RelErr(const std::vector<float>& got, const std::vector<double>& want) {
  double worst = 0.0, scale = 0.0;
  for (double v : want) scale = std::max(scale, std::abs(v));
  scale = std::max(scale, 1e-6);
  for (size_t i = 0; i < got.size(); ++i) {
    REQUIRE(!std::isnan(static_cast<double>(got[i])));
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - want[i]) / scale);
  }
  return worst;
}

// Runs the whole block once. `decode_tokens` is upstream's `num_mqa_tokens`, and
// the requests it names must be the LEADING ones (decode tokens packed first).
std::vector<float> RunBlock(Backend& b, Queue& q, const MlaBlockDims& d,
                            const HostWeights& hw, DType dt,
                            const std::vector<float>& hidden,
                            const std::vector<int32_t>& positions,
                            const std::vector<Request>& reqs, const RefContext& ctx,
                            int64_t decode_reqs, int64_t workspace_tokens,
                            std::vector<uint8_t>* raw_out = nullptr) {
  const int64_t L = d.kv_lora_rank, R = d.qk_rope_head_dim, H = d.hidden_size;
  int64_t T = 0;
  for (const Request& r : reqs) T += r.q_len;
  Paging pg = MakePaging(reqs);
  BlockHarness hh(b, q, d, hw, dt, pg.num_blocks);

  // Seed the cache with the context rows (they were written by earlier steps).
  {
    std::vector<float> cache(static_cast<size_t>(pg.num_blocks * kBlockSize * (L + R)), 0.0f);
    for (size_t r = 0; r < reqs.size(); ++r) {
      for (int64_t s = 0; s < reqs[r].ctx_len; ++s) {
        const int32_t blk = pg.block_table[static_cast<size_t>(
            static_cast<int64_t>(r) * pg.max_blocks + s / kBlockSize)];
        const int64_t slot = static_cast<int64_t>(blk) * kBlockSize + s % kBlockSize;
        for (int64_t j = 0; j < L + R; ++j) {
          cache[static_cast<size_t>(slot * (L + R) + j)] =
              static_cast<float>(ctx.ctx_rows[r][static_cast<size_t>(s * (L + R) + j)]);
        }
      }
    }
    const size_t n = cache.size();
    if (dt == DType::kF32) {
      b.Copy(q, hh.kv_cache().data, cache.data(), n * sizeof(float));
    } else {
      std::vector<uint16_t> bf(n);
      for (size_t i = 0; i < n; ++i) bf[i] = vt::F32ToBF16(cache[i]);
      b.Copy(q, hh.kv_cache().data, bf.data(), n * sizeof(uint16_t));
    }
    b.Synchronize(q);
  }

  Tensor t_hidden = hh.Up(hidden, {T, H});
  Tensor t_pos = hh.UpI32(positions, {T});
  Tensor t_slot = hh.UpI64(pg.slot_mapping, {T});
  Tensor t_out = hh.Alloc(dt, {T, H});
  // NaN-poison the output: a block that fails to write a token FAILS (the W4
  // discipline, ported from test_mla_decode_cpu.py:71-73).
  {
    const size_t n = static_cast<size_t>(T * H);
    if (dt == DType::kF32) {
      std::vector<uint32_t> poison(n, 0x7FC00000u);
      b.Copy(q, t_out.data, poison.data(), n * sizeof(uint32_t));
    } else {
      std::vector<uint16_t> poison(n, 0x7FC0u);
      b.Copy(q, t_out.data, poison.data(), n * sizeof(uint16_t));
    }
    b.Synchronize(q);
  }

  MlaBlockMetadata meta;
  int64_t decode_tokens = 0;
  for (int64_t i = 0; i < decode_reqs; ++i) decode_tokens += reqs[static_cast<size_t>(i)].q_len;
  meta.num_decode_tokens = decode_tokens;

  // --- decode half ---
  std::vector<int32_t> dec_bt, dec_sl;
  int32_t dec_max_seq = 0;
  for (int64_t r = 0; r < decode_reqs; ++r) {
    for (int64_t j = 0; j < pg.max_blocks; ++j)
      dec_bt.push_back(pg.block_table[static_cast<size_t>(r * pg.max_blocks + j)]);
    dec_sl.push_back(pg.seq_lens[static_cast<size_t>(r)]);
    dec_max_seq = std::max(dec_max_seq, pg.seq_lens[static_cast<size_t>(r)]);
  }
  if (decode_reqs > 0) {
    meta.decode.block_table = hh.UpI32(dec_bt, {decode_reqs, pg.max_blocks});
    meta.decode.seq_lens = hh.UpI32(dec_sl, {decode_reqs});
    meta.decode.max_seq_len = dec_max_seq;
  }

  // --- prefill half ---
  const int64_t prefill_reqs = static_cast<int64_t>(reqs.size()) - decode_reqs;
  std::vector<MlaChunkDeviceMetadata> chunks;
  std::vector<Tensor> keepalive;
  if (prefill_reqs > 0) {
    std::vector<int32_t> cu_q{0}, ctx_lens, pre_bt;
    int32_t running = 0, max_q = 0;
    for (int64_t r = decode_reqs; r < static_cast<int64_t>(reqs.size()); ++r) {
      running += static_cast<int32_t>(reqs[static_cast<size_t>(r)].q_len);
      cu_q.push_back(running);
      max_q = std::max(max_q, static_cast<int32_t>(reqs[static_cast<size_t>(r)].q_len));
      ctx_lens.push_back(static_cast<int32_t>(reqs[static_cast<size_t>(r)].ctx_len));
      for (int64_t j = 0; j < pg.max_blocks; ++j)
        pre_bt.push_back(pg.block_table[static_cast<size_t>(r * pg.max_blocks + j)]);
    }
    meta.prefill_cu_seqlens_q = hh.UpI32(cu_q, {prefill_reqs + 1});
    meta.prefill_block_table = hh.UpI32(pre_bt, {prefill_reqs, pg.max_blocks});
    meta.max_query_len = max_q;

    auto cm = vllm::mla::BuildMlaChunkedContext(ctx_lens, cu_q, workspace_tokens, kBlockSize);
    meta.prefill_tokens_with_context = cm.prefill_tokens_with_context;
    meta.chunk_workspace_tokens = workspace_tokens;
    const int64_t np = cm.num_prefills;
    for (int32_t i = 0; i < cm.num_chunks; ++i) {
      MlaChunkDeviceMetadata cd;
      std::vector<int32_t> cu(cm.cu_seq_lens.begin() + static_cast<size_t>(i) * (np + 1),
                              cm.cu_seq_lens.begin() +
                                  static_cast<size_t>(i + 1) * (np + 1));
      std::vector<int32_t> starts(cm.starts.begin() + static_cast<size_t>(i) * np,
                                  cm.starts.begin() + static_cast<size_t>(i + 1) * np);
      const int32_t row = std::max<int32_t>(cm.max_token_num_over_chunk, 1);
      std::vector<int32_t> t2s(cm.token_to_seq.begin() + static_cast<size_t>(i) * row,
                               cm.token_to_seq.begin() + static_cast<size_t>(i + 1) * row);
      cd.cu_seq_lens = hh.UpI32(cu, {np + 1});
      cd.starts = hh.UpI32(starts, {np});
      cd.token_to_seq = hh.UpI32(t2s, {row});
      cd.total_tokens = cm.chunk_total_token[static_cast<size_t>(i)];
      cd.max_seq_len = cm.max_seq_lens[static_cast<size_t>(i)];
      chunks.push_back(cd);
    }
    meta.chunks = chunks;
  }

  vllm::v1::TritonMLAImpl impl;
  Dev dev = hh.dev();
  Tensor kvc = hh.kv_cache();
  ForwardMlaAttentionBlock(dev, d, hh.weights(), t_hidden, t_pos, kvc, t_slot, meta, impl,
                           t_out);
  b.Synchronize(q);
  if (raw_out != nullptr) *raw_out = hh.DownRaw(t_out);
  (void)ctx;
  return hh.Down(t_out);
}

RefContext MakeContext(const MlaBlockDims& d, const std::vector<Request>& reqs,
                       uint32_t seed) {
  const int64_t W = d.kv_lora_rank + d.qk_rope_head_dim;
  RefContext c;
  for (size_t r = 0; r < reqs.size(); ++r) {
    auto v = RandF32(static_cast<size_t>(reqs[r].ctx_len * W), seed + 17u * static_cast<uint32_t>(r),
                     0.6f);
    // The cache is stored in the block's dtype; round to bf16 so the oracle sees
    // the same bits even when the harness dtype is bf16 (rounding f32 values that
    // are already bf16-representable is a no-op on the f32 path).
    std::vector<double> dv(v.size());
    for (size_t i = 0; i < v.size(); ++i) dv[i] = vt::BF16ToF32(vt::F32ToBF16(v[i]));
    c.ctx_rows.push_back(std::move(dv));
  }
  return c;
}

std::vector<int32_t> MakePositions(const std::vector<Request>& reqs) {
  std::vector<int32_t> pos;
  for (const Request& r : reqs)
    for (int64_t i = 0; i < r.q_len; ++i)
      pos.push_back(static_cast<int32_t>(r.ctx_len + i));
  return pos;
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("AbsorbKvBProjBf16 reproduces process_weights_after_loading exactly") {
  // mla_attention.py:880-900 + :959-962. The reference here is an INDEPENDENT
  // transcription that literally performs `.T -> view -> split -> permute` step
  // by step, rather than the folded index arithmetic the production function
  // uses — so agreement is evidence, not a restatement.
  MlaBlockDims d = LiteDims();
  d.scale = 1.0f;  // Validate() only needs it non-zero here
  const int64_t N = d.num_heads, P = d.qk_nope_head_dim, V = d.v_head_dim,
                L = d.kv_lora_rank;
  auto raw = RandF32(static_cast<size_t>(N * (P + V) * L), 4242u);
  std::vector<uint16_t> src(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) src[i] = vt::F32ToBF16(raw[i]);

  AbsorbedKvBProj got = AbsorbKvBProjBf16(src.data(), d);
  REQUIRE(got.w_uk_t.size() == static_cast<size_t>(N * P * L));
  REQUIRE(got.w_uv.size() == static_cast<size_t>(N * L * V));

  // step 1: transpose  [N*(P+V), L] -> [L, N*(P+V)]
  std::vector<uint16_t> tr(src.size());
  const int64_t rows = N * (P + V);
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < L; ++j) tr[static_cast<size_t>(j * rows + i)] = src[static_cast<size_t>(i * L + j)];
  // step 2/3: view [L, N, P+V], split on the last axis
  // step 4: W_UK.permute(1,2,0) -> [N,P,L] ; W_UV.transpose(0,1) -> [N,L,V]
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t p = 0; p < P; ++p)
      for (int64_t l = 0; l < L; ++l)
        REQUIRE(got.w_uk_t[static_cast<size_t>((n * P + p) * L + l)] ==
                tr[static_cast<size_t>(l * rows + n * (P + V) + p)]);
    for (int64_t l = 0; l < L; ++l)
      for (int64_t v = 0; v < V; ++v)
        REQUIRE(got.w_uv[static_cast<size_t>((n * L + l) * V + v)] ==
                tr[static_cast<size_t>(l * rows + n * (P + V) + P + v)]);
  }
}

TEST_CASE("YaRN mscale and the mscale^2 softmax scale match the upstream formulas") {
  // deepseek_scaling_rope.py:20-23.
  CHECK(YarnGetMscale(0.5, 1.0) == doctest::Approx(1.0));
  CHECK(YarnGetMscale(1.0, 1.0) == doctest::Approx(1.0));
  CHECK(YarnGetMscale(40.0, 0.707) ==
        doctest::Approx(0.1 * 0.707 * std::log(40.0) + 1.0).epsilon(1e-12));

  // deepseek_v2.py:995 then :1067-1075 — the SQUARE, applied to qk_head_dim^-0.5.
  MlaBlockDims d = LiteDims();
  DeepseekYarnRopeParams p = LiteRope();
  const double m = 0.1 * 0.707 * std::log(40.0) + 1.0;
  const double want = std::pow(192.0, -0.5) * m * m;
  CHECK(static_cast<double>(MlaAttentionScale(d, p)) == doctest::Approx(want).epsilon(1e-6));
  // A non-YaRN config gets the bare head_dim^-0.5 (the `rope_type == "default"`
  // arm at deepseek_v2.py:1069-1070 never multiplies).
  DeepseekYarnRopeParams plain;
  plain.rotary_dim = 64;
  plain.yarn = false;
  CHECK(static_cast<double>(MlaAttentionScale(d, plain)) ==
        doctest::Approx(std::pow(192.0, -0.5)).epsilon(1e-6));
}

TEST_CASE("The DeepSeek YaRN cos|sin cache matches an independent transcription") {
  // Independent re-derivation of deepseek_scaling_rope.py:76-118 over
  // rotary_embedding/common.py:34-70, written from the formulas.
  DeepseekYarnRopeParams p = LiteRope();
  const int64_t rows = 64, rot = p.rotary_dim, half = rot / 2;
  auto got = BuildDeepseekRopeCosSinCache(p, rows);
  REQUIRE(got.size() == static_cast<size_t>(rows * rot));

  auto corr_dim = [&](double nrot) {
    return (static_cast<double>(rot) *
            std::log(static_cast<double>(p.original_max_position_embeddings) /
                     (nrot * 2.0 * M_PI))) /
           (2.0 * std::log(p.base));
  };
  double low = std::max(std::floor(corr_dim(p.beta_fast)), 0.0);
  double high = std::min(std::ceil(corr_dim(p.beta_slow)), static_cast<double>(rot - 1));
  if (low == high) high += 0.001;
  const double ms = (0.1 * p.mscale * std::log(p.scaling_factor) + 1.0) /
                    (0.1 * p.mscale_all_dim * std::log(p.scaling_factor) + 1.0);
  for (int64_t i = 0; i < half; ++i) {
    const double pf = std::pow(p.base, 2.0 * static_cast<double>(i) / rot);
    const double extrap = 1.0 / pf, interp = 1.0 / (p.scaling_factor * pf);
    const double ramp = std::min(1.0, std::max(0.0, (static_cast<double>(i) - low) / (high - low)));
    const double mask = 1.0 - ramp;
    const double inv = interp * (1.0 - mask) + extrap * mask;
    for (int64_t t = 0; t < rows; ++t) {
      const double ang = static_cast<double>(t) * inv;
      CHECK(static_cast<double>(got[static_cast<size_t>(t * rot + i)]) ==
            doctest::Approx(std::cos(ang) * ms).epsilon(1e-6));
      CHECK(static_cast<double>(got[static_cast<size_t>(t * rot + half + i)]) ==
            doctest::Approx(std::sin(ang) * ms).epsilon(1e-6));
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// (1) THE IDENTITY ITSELF — absorbed and unabsorbed are the same function.
TEST_CASE("ABSORBED and UNABSORBED attention are numerically the SAME function") {
  for (bool lora : {false, true}) {
    CAPTURE(lora);
    MlaBlockDims d = LiteDims();
    if (lora) d.q_lora_rank = 1536;  // V3's real rank
    DeepseekYarnRopeParams rp = LiteRope();
    d.scale = MlaAttentionScale(d, rp);
    HostWeights hw = MakeWeights(d, rp, 512, 909u);

    const std::vector<Request> reqs = {{7, 1}, {23, 1}, {0, 5}, {31, 3}};
    int64_t T = 0;
    for (const Request& r : reqs) T += r.q_len;
    auto hidden = RandF32(static_cast<size_t>(T * d.hidden_size), 606u, 0.8f);
    hidden = RoundBf16(hidden);
    auto pos = MakePositions(reqs);
    RefContext ctx = MakeContext(d, reqs, 303u);

    const auto unabsorbed = RefBlock(d, hw, hidden, pos, reqs, ctx, /*absorbed=*/false);
    const auto absorbed = RefBlock(d, hw, hidden, pos, reqs, ctx, /*absorbed=*/true);
    REQUIRE(unabsorbed.size() == absorbed.size());
    double scale = 0.0;
    for (double v : unabsorbed) scale = std::max(scale, std::abs(v));
    double worst = 0.0;
    for (size_t i = 0; i < unabsorbed.size(); ++i)
      worst = std::max(worst, std::abs(unabsorbed[i] - absorbed[i]));
    // Both routes are double-precision; the only difference is the ORDER of the
    // W_UK / W_UV contractions, so they agree to accumulation noise.
    CHECK(worst / std::max(scale, 1e-9) < 1e-11);
  }
}

// (2) OURS vs the UNABSORBED oracle, in f32 so the comparison is tight.
TEST_CASE("CPU MLA block (f32) reproduces the unabsorbed double oracle — decode, prefill, MIXED") {
  Backend& b = vt::GetBackend(DeviceType::kCPU);
  Queue q{Cpu(), nullptr};
  MlaBlockDims d = LiteDims();
  DeepseekYarnRopeParams rp = LiteRope();
  d.scale = MlaAttentionScale(d, rp);
  HostWeights hw = MakeWeights(d, rp, 512, 1717u);

  struct Case {
    std::vector<Request> reqs;
    int64_t decode_reqs;
    const char* what;
  };
  const std::vector<Case> cases = {
      // Pure decode — every request is one new token over cached context.
      {{{5, 1}, {16, 1}, {17, 1}, {63, 1}}, 4, "decode-only, ragged ctx across page boundaries"},
      // Pure prefill, NO context — the causal new-tokens pass alone
      // (mla_attention.py:2421-2425 skips the whole chunk loop).
      {{{0, 9}, {0, 1}, {0, 16}}, 0, "prefill-only, no context, ragged"},
      // Pure prefill WITH context — the chunked-context loop + LSE merge. NOTE
      // the request ORDER: upstream requires the with-context prefills to come
      // FIRST, because `prefill_tokens_with_context` is
      // `query_start_loc[num_prefills_with_context]` (mla_attention.py:1806-1810)
      // and every query row past it takes the suffix result verbatim.
      {{{40, 5}, {19, 2}, {0, 3}}, 0, "prefill with context (chunked) + a context-free request"},
      // MIXED, decode tokens packed FIRST (mla_attention.py:700-709), and inside
      // the prefill tail the with-context request first.
      {{{12, 1}, {33, 1}, {21, 4}, {0, 7}}, 2, "MIXED batch, decode packed first"},
  };
  for (const Case& c : cases) {
    CAPTURE(c.what);
    int64_t T = 0;
    for (const Request& r : c.reqs) T += r.q_len;
    auto hidden = RoundBf16(RandF32(static_cast<size_t>(T * d.hidden_size), 808u, 0.8f));
    auto pos = MakePositions(c.reqs);
    RefContext ctx = MakeContext(d, c.reqs, 404u);
    const auto want = RefBlock(d, hw, hidden, pos, c.reqs, ctx, /*absorbed=*/false);
    // A workspace of 64 rows over contexts up to 40 forces MULTIPLE chunks and
    // exercises the page-aligned round-down (mla_attention.py:1687-1690).
    const auto got =
        RunBlock(b, q, d, hw, DType::kF32, hidden, pos, c.reqs, ctx, c.decode_reqs, 64);
    CHECK(RelErr(got, want) < 2e-4);
  }
}

// (3) OURS vs OURS THROUGH TWO DIFFERENT CODE PATHS — the strongest form of the
//     absorption proof, because nothing but the weights is shared.
TEST_CASE("The SAME batch through the ABSORBED decode kernel and the UNABSORBED prefill path agree") {
  Backend& b = vt::GetBackend(DeviceType::kCPU);
  Queue q{Cpu(), nullptr};
  MlaBlockDims d = LiteDims();
  DeepseekYarnRopeParams rp = LiteRope();
  d.scale = MlaAttentionScale(d, rp);
  HostWeights hw = MakeWeights(d, rp, 512, 2323u);

  // Every request has exactly ONE new token, so the batch is legally either a
  // decode batch or a single-query-token prefill batch. Labelled decode it runs
  // vt::MlaDecodeAttention (MQA, QK 576 / V 512) with W_UK folded into the query
  // and W_UV un-projecting the output; labelled prefill it runs
  // vt::MlaPrefillAttention (MHA, QK 192 / V 128) over K/V materialized by
  // kv_b_proj plus the chunked-context loop over the cached latent.
  const std::vector<Request> reqs = {{9, 1}, {16, 1}, {35, 1}, {1, 1}};
  int64_t T = 0;
  for (const Request& r : reqs) T += r.q_len;
  auto hidden = RoundBf16(RandF32(static_cast<size_t>(T * d.hidden_size), 5150u, 0.8f));
  auto pos = MakePositions(reqs);
  RefContext ctx = MakeContext(d, reqs, 5151u);

  const auto as_decode =
      RunBlock(b, q, d, hw, DType::kF32, hidden, pos, reqs, ctx, /*decode_reqs=*/4, 64);
  const auto as_prefill =
      RunBlock(b, q, d, hw, DType::kF32, hidden, pos, reqs, ctx, /*decode_reqs=*/0, 64);
  REQUIRE(as_decode.size() == as_prefill.size());
  double scale = 0.0, worst = 0.0;
  for (float v : as_decode) scale = std::max(scale, std::abs(static_cast<double>(v)));
  for (size_t i = 0; i < as_decode.size(); ++i) {
    REQUIRE(!std::isnan(as_decode[i]));
    REQUIRE(!std::isnan(as_prefill[i]));
    worst = std::max(worst, std::abs(static_cast<double>(as_decode[i] - as_prefill[i])));
  }
  CHECK(worst / std::max(scale, 1e-9) < 3e-4);
}

// The lora query branch — DeepSeek-V2-Lite has `q_lora_rank = null` and CANNOT
// exercise it, so this is the ONLY coverage `fused_qkv_a_proj` / `q_a_layernorm`
// / `q_b_proj` gets. Stated plainly in the record: unit-gated only, NO e2e
// coverage on GB10 (GLM-4.7-Flash, 58.2 GiB, has q_lora_rank=768 and could close
// it later; it is not attempted in W6).
TEST_CASE("CPU MLA block, the q_lora branch at DeepSeek-V3 dimensions") {
  Backend& b = vt::GetBackend(DeviceType::kCPU);
  Queue q{Cpu(), nullptr};
  MlaBlockDims d = V3Dims();  // hidden 7168, 128 heads, q_lora_rank 1536
  DeepseekYarnRopeParams rp = LiteRope();
  d.scale = MlaAttentionScale(d, rp);
  REQUIRE(d.has_q_lora());
  HostWeights hw = MakeWeights(d, rp, 128, 3131u);

  const std::vector<Request> reqs = {{6, 1}, {0, 2}};  // one decode + one prefill
  int64_t T = 0;
  for (const Request& r : reqs) T += r.q_len;
  auto hidden = RoundBf16(RandF32(static_cast<size_t>(T * d.hidden_size), 3132u, 0.5f));
  auto pos = MakePositions(reqs);
  RefContext ctx = MakeContext(d, reqs, 3133u);
  const auto want = RefBlock(d, hw, hidden, pos, reqs, ctx, /*absorbed=*/false);
  const auto got =
      RunBlock(b, q, d, hw, DType::kF32, hidden, pos, reqs, ctx, /*decode_reqs=*/1, 64);
  CHECK(RelErr(got, want) < 3e-4);
}

TEST_CASE("CPU MLA block is BIT-exact run to run") {
  Backend& b = vt::GetBackend(DeviceType::kCPU);
  Queue q{Cpu(), nullptr};
  MlaBlockDims d = LiteDims();
  DeepseekYarnRopeParams rp = LiteRope();
  d.scale = MlaAttentionScale(d, rp);
  HostWeights hw = MakeWeights(d, rp, 512, 6161u);
  const std::vector<Request> reqs = {{12, 1}, {40, 1}, {0, 6}, {23, 3}};
  int64_t T = 0;
  for (const Request& r : reqs) T += r.q_len;
  auto hidden = RoundBf16(RandF32(static_cast<size_t>(T * d.hidden_size), 6162u, 0.8f));
  auto pos = MakePositions(reqs);
  RefContext ctx = MakeContext(d, reqs, 6163u);

  std::vector<uint8_t> first;
  RunBlock(b, q, d, hw, DType::kBF16, hidden, pos, reqs, ctx, 2, 64, &first);
  for (int rep = 0; rep < 3; ++rep) {
    std::vector<uint8_t> again;
    RunBlock(b, q, d, hw, DType::kBF16, hidden, pos, reqs, ctx, 2, 64, &again);
    REQUIRE(again == first);
  }
}

// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("CUDA MLA block (bf16) reproduces the unabsorbed oracle and is bit-exact") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  Queue q = b.CreateQueue();
  MlaBlockDims d = LiteDims();
  DeepseekYarnRopeParams rp = LiteRope();
  d.scale = MlaAttentionScale(d, rp);
  HostWeights hw = MakeWeights(d, rp, 512, 7171u);

  struct Case {
    std::vector<Request> reqs;
    int64_t decode_reqs;
    const char* what;
  };
  const std::vector<Case> cases = {
      {{{5, 1}, {16, 1}, {17, 1}, {63, 1}}, 4, "decode-only"},
      {{{0, 9}, {0, 1}, {0, 16}}, 0, "prefill-only, no context"},
      {{{40, 5}, {19, 2}, {0, 3}}, 0, "prefill with chunked context"},
      {{{12, 1}, {33, 1}, {21, 4}, {0, 7}}, 2, "MIXED, decode packed first"},
  };
  for (const Case& c : cases) {
    CAPTURE(c.what);
    int64_t T = 0;
    for (const Request& r : c.reqs) T += r.q_len;
    auto hidden = RoundBf16(RandF32(static_cast<size_t>(T * d.hidden_size), 7272u, 0.8f));
    auto pos = MakePositions(c.reqs);
    RefContext ctx = MakeContext(d, c.reqs, 7373u);
    const auto want = RefBlock(d, hw, hidden, pos, c.reqs, ctx, /*absorbed=*/false);
    std::vector<uint8_t> raw;
    const auto got =
        RunBlock(b, q, d, hw, DType::kBF16, hidden, pos, c.reqs, ctx, c.decode_reqs, 64, &raw);
    // bf16 storage of every intermediate (q, the latent, the attention output)
    // over ~2k-wide reductions: the achievable bound is percent-level, and the
    // f32 CPU case above is what pins the MATH. What this case proves is that
    // the CUDA kernels, the strided views and the dispatch are right.
    CHECK(RelErr(got, want) < 3e-2);
    for (int rep = 0; rep < 2; ++rep) {
      std::vector<uint8_t> again;
      RunBlock(b, q, d, hw, DType::kBF16, hidden, pos, c.reqs, ctx, c.decode_reqs, 64, &again);
      REQUIRE(again == raw);
    }
  }
  b.DestroyQueue(q);
}

TEST_CASE("CUDA: the absorbed decode path and the unabsorbed prefill path agree") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  Queue q = b.CreateQueue();
  MlaBlockDims d = LiteDims();
  DeepseekYarnRopeParams rp = LiteRope();
  d.scale = MlaAttentionScale(d, rp);
  HostWeights hw = MakeWeights(d, rp, 512, 8181u);

  const std::vector<Request> reqs = {{9, 1}, {16, 1}, {35, 1}, {1, 1}};
  int64_t T = 0;
  for (const Request& r : reqs) T += r.q_len;
  auto hidden = RoundBf16(RandF32(static_cast<size_t>(T * d.hidden_size), 8282u, 0.8f));
  auto pos = MakePositions(reqs);
  RefContext ctx = MakeContext(d, reqs, 8383u);

  const auto as_decode = RunBlock(b, q, d, hw, DType::kBF16, hidden, pos, reqs, ctx, 4, 64);
  const auto as_prefill = RunBlock(b, q, d, hw, DType::kBF16, hidden, pos, reqs, ctx, 0, 64);
  double scale = 0.0, worst = 0.0;
  for (float v : as_decode) scale = std::max(scale, std::abs(static_cast<double>(v)));
  for (size_t i = 0; i < as_decode.size(); ++i) {
    REQUIRE(!std::isnan(as_decode[i]));
    REQUIRE(!std::isnan(as_prefill[i]));
    worst = std::max(worst, std::abs(static_cast<double>(as_decode[i] - as_prefill[i])));
  }
  CHECK(worst / std::max(scale, 1e-9) < 4e-2);
  b.DestroyQueue(q);
}
