// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace vt::cpu {
namespace {

float LoadF32(const Tensor& t, int64_t elem_offset) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_offset];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    default: VT_CHECK(false, "LoadF32: unsupported dtype"); return 0.0f;
  }
}

// Mirror of LoadF32 for outputs: f32 stores directly, bf16 rounds via F32ToBF16.
void StoreF32(const Tensor& t, int64_t elem_offset, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[elem_offset] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[elem_offset] = F32ToBF16(v); break;
    default: VT_CHECK(false, "StoreF32: unsupported dtype");
  }
}

void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += LoadF32(a, i * k + p) * LoadF32(b, p * n + j);
      }
      StoreF32(out, i * n + j, acc);
    }
  }
}

void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  for (int64_t i = 0; i < t; ++i) {
    float* res_row = residual ? residual->Ptr<float>() + i * h : nullptr;
    float sumsq = 0.0f;
    for (int64_t j = 0; j < h; ++j) {
      float v = LoadF32(x, i * h + j);
      if (res_row) {
        v += res_row[j];
        res_row[j] = v;  // new residual stream
      }
      sumsq += v * v;
    }
    float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(h) + args.eps);
    for (int64_t j = 0; j < h; ++j) {
      float v = res_row ? res_row[j] : LoadF32(x, i * h + j);
      float wj = LoadF32(w, j);
      if (args.gemma) wj += 1.0f;
      StoreF32(out, i * h + j, v * inv * wj);
    }
  }
}

void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  for (int64_t i = 0; i < t; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      float gate = LoadF32(x, i * 2 * d + j);
      float up = LoadF32(x, i * 2 * d + d + j);
      float silu = gate / (1.0f + std::exp(-gate));
      StoreF32(out, i * d + j, silu * up);
    }
  }
}

void MoeSiluMulKernel(Queue&, Tensor& out, const Tensor& gate, const Tensor& up) {
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) {
    const float g = LoadF32(gate, i);
    const float silu = g / (1.0f + std::exp(-g));
    StoreF32(out, i, silu * LoadF32(up, i));
  }
}

void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  const int64_t t = ids.shape[0], h = table.shape[1], v = table.shape[0];
  for (int64_t i = 0; i < t; ++i) {
    int64_t id = ids.dtype == DType::kI32 ? ids.Ptr<int32_t>()[i] : ids.Ptr<int64_t>()[i];
    VT_CHECK(id >= 0 && id < v, "embedding: id out of range");
    for (int64_t j = 0; j < h; ++j) {
      StoreF32(out, i * h + j, LoadF32(table, id * h + j));
    }
  }
}

// In-place rotation of one head starting at element head_off; f32 math,
// stores round back to the tensor's dtype (f32 or bf16).
void RopeRotateHead(const Tensor& t, int64_t head_off, int rot, double base, int64_t pos) {
  const int half = rot / 2;
  for (int i = 0; i < half; ++i) {
    double freq = std::pow(base, -2.0 * i / rot);
    double angle = static_cast<double>(pos) * freq;
    float c = static_cast<float>(std::cos(angle));
    float s = static_cast<float>(std::sin(angle));
    float x = LoadF32(t, head_off + i);
    float y = LoadF32(t, head_off + i + half);
    StoreF32(t, head_off + i, x * c - y * s);
    StoreF32(t, head_off + i + half, x * s + y * c);
  }
}

void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  for (int64_t i = 0; i < t; ++i) {
    int64_t p = pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[i] : pos.Ptr<int64_t>()[i];
    for (int64_t hh = 0; hh < hq; ++hh) {
      RopeRotateHead(qs, (i * hq + hh) * d, args.rotary_dim, static_cast<double>(args.base), p);
    }
    for (int64_t hh = 0; hh < hk; ++hh) {
      RopeRotateHead(ks, (i * hk + hh) * d, args.rotary_dim, static_cast<double>(args.base), p);
    }
  }
}

float Silu(float x) { return x / (1.0f + std::exp(-x)); }

// GDN CPU reference kernels. Formulas: .agents/gdn-semantics.md (§ cited per
// kernel); scalar f32 math throughout, states f32 in place.

// §2 causal_conv1d_fn. Per sequence s (tokens [qsl[s], qsl[s+1])), channel c,
// token t: window[j] = x token t-(K-1-j), falling back to
// conv_state[c, (K-1)+(t-i)] (init state) or 0 before the sequence start.
// Write-back: last K-1 RAW x tokens, left-padded with zeros / shifted old
// state when T < K-1. Outputs read the OLD state, so the row is buffered
// before overwrite.
void CausalConv1dFwdKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                           const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t total = x.shape[0], c_dim = x.shape[1], k = w.shape[1], width = k - 1;
  const int64_t n = conv_state.shape[0];
  const int32_t* qslp = qsl.Ptr<int32_t>();
  const int32_t* hisp = his.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == total, "causal_conv1d_fwd: bad query_start_loc bounds");
  std::vector<float> old_row(static_cast<size_t>(width));
  for (int64_t s = 0; s < n; ++s) {
    const int64_t begin = qslp[s], end = qslp[s + 1], t_len = end - begin;
    VT_CHECK(t_len >= 0 && begin >= 0, "causal_conv1d_fwd: query_start_loc not monotonic");
    const bool init = hisp[s] != 0;
    float* srow_base = conv_state.Ptr<float>() + s * c_dim * width;
    for (int64_t c = 0; c < c_dim; ++c) {
      float* srow = srow_base + c * width;
      for (int64_t j = 0; j < width; ++j) old_row[static_cast<size_t>(j)] = srow[j];
      const float b = bias != nullptr ? LoadF32(*bias, c) : 0.0f;
      for (int64_t t = 0; t < t_len; ++t) {
        float acc = b;
        for (int64_t j = 0; j < k; ++j) {
          const int64_t ti = t - (k - 1 - j);  // token index of window[j]
          float v = 0.0f;
          if (ti >= 0) {
            v = LoadF32(x, (begin + ti) * c_dim + c);
          } else if (init) {
            v = old_row[static_cast<size_t>(width + ti)];  // state col (K-1)+(t-i)
          }
          acc += LoadF32(w, c * k + j) * v;
        }
        StoreF32(out, (begin + t) * c_dim + c, args.silu_activation ? Silu(acc) : acc);
      }
      for (int64_t j = 0; j < width; ++j) {
        const int64_t tj = t_len - width + j;  // new state col j holds token tj
        float v = 0.0f;
        if (tj >= 0) {
          v = LoadF32(x, (begin + tj) * c_dim + c);
        } else if (init) {
          v = old_row[static_cast<size_t>(width + tj)];  // shifted old state
        }
        srow[j] = v;
      }
    }
  }
}

// §3 causal_conv1d_update (seqlen==1): read-old-then-roll.
void CausalConv1dUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const CausalConv1dArgs& args) {
  const int64_t batch = x.shape[0], c_dim = x.shape[1], k = w.shape[1], width = k - 1;
  for (int64_t bt = 0; bt < batch; ++bt) {
    float* srow_base = conv_state.Ptr<float>() + bt * c_dim * width;
    for (int64_t c = 0; c < c_dim; ++c) {
      float* srow = srow_base + c * width;
      const float xt = LoadF32(x, bt * c_dim + c);
      float acc = bias != nullptr ? LoadF32(*bias, c) : 0.0f;
      for (int64_t j = 0; j < width; ++j) acc += LoadF32(w, c * k + j) * srow[j];
      acc += LoadF32(w, c * k + width) * xt;
      StoreF32(out, bt * c_dim + c, args.silu_activation ? Silu(acc) : acc);
      for (int64_t j = 0; j + 1 < width; ++j) srow[j] = srow[j + 1];  // roll left
      if (width > 0) srow[width - 1] = xt;                            // raw x
    }
  }
}

// §4 l2norm_fwd: y = x * rsqrt(sum(x^2) + eps) over the last dim (plain SUM).
void L2NormKernel(Queue&, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = x.Numel() / d;
  for (int64_t r = 0; r < rows; ++r) {
    float sumsq = 0.0f;
    for (int64_t j = 0; j < d; ++j) {
      const float v = LoadF32(x, r * d + j);
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq + args.eps);
    for (int64_t j = 0; j < d; ++j) StoreF32(out, r * d + j, LoadF32(x, r * d + j) * inv);
  }
}

// §5 RMSNormGated (norm_before_gate=True, group_size=None):
// out = x * rsqrt(mean(x^2) + eps) * w * act(gate); act = silu or sigmoid.
void RmsNormGatedKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                        const Tensor& w, const RmsNormGatedArgs& args) {
  const int64_t t = x.shape[0], d = x.shape[1];
  for (int64_t i = 0; i < t; ++i) {
    float sumsq = 0.0f;
    for (int64_t j = 0; j < d; ++j) {
      const float v = LoadF32(x, i * d + j);
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(d) + args.eps);
    for (int64_t j = 0; j < d; ++j) {
      const float z = LoadF32(gate, i * d + j);
      const float act = args.sigmoid_gate ? 1.0f / (1.0f + std::exp(-z)) : Silu(z);
      StoreF32(out, i * d + j, LoadF32(x, i * d + j) * inv * LoadF32(w, j) * act);
    }
  }
}

// §7 gated-delta-rule token step, shared by prefill and decode. state points
// at this sequence's [Hv,Dv,Dk] f32 block; tok indexes the packed q/k/v/g/beta
// rows. GQA broadcast: v-head hv reads q/k head hv / (Hv/Hk).
//   q' = q * scale;  S *= exp(g[hv]);  v' = (v - S @ k) * beta[hv];
//   S += outer(v', k);  out = S @ q'      (k is NOT scaled)
void GdnTokenStep(Tensor& out, const Tensor& q_in, const Tensor& k_in, const Tensor& v_in,
                  const Tensor& g, const Tensor& beta, float* state, int64_t tok, float scale,
                  std::vector<float>& qbuf, std::vector<float>& kbuf,
                  std::vector<float>& vbuf) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v_in.shape[1], dv = v_in.shape[2];
  const int64_t ratio = hv_n / hk_n;
  for (int64_t hv = 0; hv < hv_n; ++hv) {
    const int64_t hk = hv / ratio;
    float* s_head = state + hv * dv * dk;  // [Dv, Dk]
    const float g_t = g.Ptr<float>()[tok * hv_n + hv];
    const float beta_t = beta.Ptr<float>()[tok * hv_n + hv];
    const float decay = std::exp(g_t);
    for (int64_t i = 0; i < dk; ++i) {
      qbuf[static_cast<size_t>(i)] = LoadF32(q_in, (tok * hk_n + hk) * dk + i) * scale;
      kbuf[static_cast<size_t>(i)] = LoadF32(k_in, (tok * hk_n + hk) * dk + i);
    }
    for (int64_t vi = 0; vi < dv; ++vi) {
      float* s_row = s_head + vi * dk;
      float dot = 0.0f;  // (S * exp(g)) @ k, fused with the decay pass
      for (int64_t ki = 0; ki < dk; ++ki) {
        s_row[ki] *= decay;
        dot += s_row[ki] * kbuf[static_cast<size_t>(ki)];
      }
      vbuf[static_cast<size_t>(vi)] =
          (LoadF32(v_in, (tok * hv_n + hv) * dv + vi) - dot) * beta_t;
    }
    for (int64_t vi = 0; vi < dv; ++vi) {
      float* s_row = s_head + vi * dk;
      float o = 0.0f;  // (S + outer(v',k)) @ q', fused with the rank-1 update
      for (int64_t ki = 0; ki < dk; ++ki) {
        s_row[ki] += vbuf[static_cast<size_t>(vi)] * kbuf[static_cast<size_t>(ki)];
        o += s_row[ki] * qbuf[static_cast<size_t>(ki)];
      }
      StoreF32(out, (tok * hv_n + hv) * dv + vi, o);
    }
  }
}

void GdnPrefillKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                      const Tensor& g, const Tensor& beta, Tensor& state, const Tensor& qsl,
                      const GdnArgs& args) {
  const int64_t n = state.shape[0], hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == q_in.shape[0],
           "gdn_prefill: bad query_start_loc bounds");
  std::vector<float> qbuf(static_cast<size_t>(dk)), kbuf(static_cast<size_t>(dk)),
      vbuf(static_cast<size_t>(dv));
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s], "gdn_prefill: query_start_loc not monotonic");
    float* s_state = state.Ptr<float>() + s * hv_n * dv * dk;
    for (int64_t t = qslp[s]; t < qslp[s + 1]; ++t)
      GdnTokenStep(out, q_in, k, v, g, beta, s_state, t, args.scale, qbuf, kbuf, vbuf);
  }
}

void GdnDecodeKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                     const Tensor& g, const Tensor& beta, Tensor& state, const GdnArgs& args) {
  const int64_t batch = q_in.shape[0], hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  std::vector<float> qbuf(static_cast<size_t>(dk)), kbuf(static_cast<size_t>(dk)),
      vbuf(static_cast<size_t>(dv));
  for (int64_t bt = 0; bt < batch; ++bt) {
    float* s_state = state.Ptr<float>() + bt * hv_n * dv * dk;
    GdnTokenStep(out, q_in, k, v, g, beta, s_state, bt, args.scale, qbuf, kbuf, vbuf);
  }
}

// §3 router: softmax (f32, over all E) -> greedy top-k (lowest-index tie-break)
// -> optional renormalize. weights [T,K] f32, indices [T,K] i32.
void MoeRouterTopKKernel(Queue&, Tensor& weights, Tensor& indices, const Tensor& logits,
                         const MoeRouterTopKArgs& args) {
  const int64_t t = logits.shape[0], e = logits.shape[1];
  const int k = args.top_k;
  std::vector<float> p(static_cast<size_t>(e));
  std::vector<char> chosen(static_cast<size_t>(e));
  for (int64_t row = 0; row < t; ++row) {
    // softmax(logits.float()) with max-subtraction (topk_softmax_kernels.cu).
    float mx = -INFINITY;
    for (int64_t j = 0; j < e; ++j) mx = std::max(mx, LoadF32(logits, row * e + j));
    float sum = 0.0f;
    for (int64_t j = 0; j < e; ++j) {
      const float ex = std::exp(LoadF32(logits, row * e + j) - mx);
      p[static_cast<size_t>(j)] = ex;
      sum += ex;
    }
    for (int64_t j = 0; j < e; ++j) {
      float& pj = p[static_cast<size_t>(j)];
      pj = sum > 0.0f ? pj / sum : 0.0f;
      if (!std::isfinite(pj)) pj = 0.0f;  // NaN/Inf clamp (.cu:136)
      chosen[static_cast<size_t>(j)] = 0;
    }
    // Greedy argmax, k rounds; strict `>` over ascending j -> lowest index wins.
    float denom = 0.0f;
    for (int j = 0; j < k; ++j) {
      int64_t best = -1;
      float best_v = -INFINITY;
      for (int64_t idx = 0; idx < e; ++idx) {
        if (chosen[static_cast<size_t>(idx)]) continue;
        if (p[static_cast<size_t>(idx)] > best_v) {
          best_v = p[static_cast<size_t>(idx)];
          best = idx;
        }
      }
      chosen[static_cast<size_t>(best)] = 1;
      weights.Ptr<float>()[row * k + j] = best_v;
      indices.Ptr<int32_t>()[row * k + j] = static_cast<int32_t>(best);
      denom += best_v;
    }
    if (args.renormalize) {
      if (!(denom > 0.0f)) denom = 1.0f;  // (.cu:245-253) denom<=0 -> 1 guard
      for (int j = 0; j < k; ++j) weights.Ptr<float>()[row * k + j] /= denom;
    }
  }
}

// §4/§6 weighted scatter-combine: out[t,:] = sum_j w[t,j]*expert_out[t,j,:]
// (f32 accumulation) + shared[t,:] (optional). Stored at out's dtype.
void MoeCombineKernel(Queue&, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                      const Tensor* shared) {
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  for (int64_t row = 0; row < t; ++row) {
    for (int64_t col = 0; col < h; ++col) {
      float acc = 0.0f;
      for (int64_t j = 0; j < k; ++j)
        acc += weights.Ptr<float>()[row * k + j] *
               LoadF32(expert_out, (row * k + j) * h + col);
      if (shared != nullptr) acc += LoadF32(*shared, row * h + col);
      StoreF32(out, row * h + col, acc);
    }
  }
}

// Dense causal attention (qwen36-forward-notes.md §5). Causal scaled-dot-product
// with GQA broadcast over a single packed sequence. query [T,Hq,D],
// key/value [T,Hk,D], out [T,Hq,D]. Per q-head h (kv-head g = h/(Hq/Hk)) and
// query i: softmax over keys j<=i of scale*(q·k), then weighted sum of v. f32
// softmax with online max-subtraction for numerical stability.
void AttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& key,
                     const Tensor& value, const AttentionArgs& args) {
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  const int64_t qpk = hq / hk;  // q-heads per kv-head (GQA ratio)
  const float scale = args.scale;
  std::vector<float> probs(static_cast<size_t>(t));
  std::vector<float> acc(static_cast<size_t>(d));
  for (int64_t h = 0; h < hq; ++h) {
    const int64_t g = h / qpk;
    for (int64_t i = 0; i < t; ++i) {
      const int64_t jmax = args.causal ? i : t - 1;  // causal: keys 0..i
      const int64_t qoff = (i * hq + h) * d;
      // Pass 1: scores + running max.
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j <= jmax; ++j) {
        const int64_t koff = (j * hk + g) * d;
        float dot = 0.0f;
        for (int64_t e = 0; e < d; ++e) dot += LoadF32(query, qoff + e) * LoadF32(key, koff + e);
        dot *= scale;
        probs[static_cast<size_t>(j)] = dot;
        if (dot > m) m = dot;
      }
      // Pass 2: exp + normalization denominator.
      float denom = 0.0f;
      for (int64_t j = 0; j <= jmax; ++j) {
        float e = std::exp(probs[static_cast<size_t>(j)] - m);
        probs[static_cast<size_t>(j)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;  // denom >= 1 (j==i term is exp(0)=1)
      // Pass 3: weighted sum of v (f32 accumulation), stored at out's dtype.
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
      for (int64_t j = 0; j <= jmax; ++j) {
        const float p = probs[static_cast<size_t>(j)] * inv;
        const int64_t voff = (j * hk + g) * d;
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += p * LoadF32(value, voff + e);
      }
      for (int64_t e = 0; e < d; ++e) StoreF32(out, qoff + e, acc[static_cast<size_t>(e)]);
    }
  }
}

struct Registrar {
  Registrar() {
    // static_cast against the ops.h aliases ties kernel signatures to the
    // registration contract at compile time.
    RegisterOp(OpId::kMatmul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kMoeSiluMul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MoeSiluMulFn>(&MoeSiluMulKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernel)));
    RegisterOp(
        OpId::kCausalConv1dUpdate, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernel)));
    RegisterOp(OpId::kL2Norm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<L2NormFn>(&L2NormKernel)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernel)));
    RegisterOp(OpId::kGdnPrefill, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernel)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernel)));
    RegisterOp(OpId::kMoeRouterTopK, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MoeRouterTopKFn>(&MoeRouterTopKKernel)));
    RegisterOp(OpId::kMoeCombine, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MoeCombineFn>(&MoeCombineKernel)));
    RegisterOp(OpId::kAttention, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
