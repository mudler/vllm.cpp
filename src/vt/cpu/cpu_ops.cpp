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
    const int64_t rbase = i * h;
    float sumsq = 0.0f;
    for (int64_t j = 0; j < h; ++j) {
      float v = LoadF32(x, i * h + j);
      if (residual) {
        v += LoadF32(*residual, rbase + j);   // add in f32
        StoreF32(*residual, rbase + j, v);     // new residual stream (rounds to its dtype)
        v = LoadF32(*residual, rbase + j);     // re-read rounded value (bf16-faithful)
      }
      sumsq += v * v;
    }
    float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(h) + args.eps);
    for (int64_t j = 0; j < h; ++j) {
      float v = residual ? LoadF32(*residual, rbase + j) : LoadF32(x, i * h + j);
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

// --- TRUE W4A4 (fp4xfp4) helpers + kernels (notes §7). Self-contained fp8/fp4
// codec (vt does not depend on vllm), bit-matching vllm::F8E4M3ToF32 /
// F32ToF8E4M3 / CastToFp4 / kE2M1Lut so the op equals vllm::RunNvfp4Emulation.
constexpr float kFp4Max = 6.0F;    // E2M1 max magnitude
constexpr float kFp8Max = 448.0F;  // fp8-e4m3fn max finite
constexpr float kE2M1[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

inline float ClampF(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline float RecipF(float x) { return x == 0.0F ? 0.0F : 1.0F / x; }

// IEEE fp8-e4m3fn byte -> f32 (bit-matches vllm::F8E4M3ToF32).
float Fp8ToF32(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1U;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFU;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7U;
  const float sm = sign ? -1.0F : 1.0F;
  if (exp == 0xFU && mant == 0x7U) return std::numeric_limits<float>::quiet_NaN();
  if (exp == 0U) return sm * (static_cast<float>(mant) * (1.0F / 512.0F));
  const float mantissa = 1.0F + static_cast<float>(mant) * (1.0F / 8.0F);
  return sm * std::ldexp(mantissa, static_cast<int>(exp) - 7);
}

// f32 -> fp8-e4m3fn byte, round-to-nearest-even saturating (bit-matches
// vllm::F32ToF8E4M3).
uint8_t F32ToFp8(float f) {
  if (std::isnan(f)) return 0x7FU;
  const uint8_t sign = std::signbit(f) ? 0x80U : 0x00U;
  const float a = std::fabs(f);
  if (!std::isfinite(a) || a >= kFp8Max) return static_cast<uint8_t>(sign | 0x7EU);
  if (a == 0.0F) return sign;
  int e2 = 0;
  const float frac = std::frexp(a, &e2);
  int exp_field = (e2 - 1) + 7;
  if (exp_field <= 0) {
    const double qd = static_cast<double>(a) * 512.0;
    const int qi = static_cast<int>(std::nearbyint(qd));
    if (qi <= 0) return sign;
    if (qi < 8) return static_cast<uint8_t>(sign | static_cast<uint8_t>(qi));
    return static_cast<uint8_t>(sign | (1U << 3));
  }
  const double sig = static_cast<double>(frac) * 2.0;
  int mi = static_cast<int>(std::nearbyint(sig * 8.0));
  if (mi == 16) {
    mi = 8;
    exp_field += 1;
  }
  const int mant = mi - 8;
  if (exp_field > 15 || (exp_field == 15 && mant >= 7)) {
    return static_cast<uint8_t>(sign | 0x7EU);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint8_t>(exp_field) << 3) |
                              static_cast<uint8_t>(mant));
}

// f32 -> E2M1 nibble (bit-matches vllm::CastToFp4 + Fp4ToNibble). Input pre-scaled.
uint8_t F32ToFp4Nibble(float x) {
  const float a = std::fabs(x);
  uint8_t idx = 7;  // 6.0
  if (a <= 0.25F) idx = 0;
  else if (a < 0.75F) idx = 1;
  else if (a <= 1.25F) idx = 2;
  else if (a < 1.75F) idx = 3;
  else if (a <= 2.5F) idx = 4;
  else if (a < 3.5F) idx = 5;
  else if (a <= 5.0F) idx = 6;
  if (idx == 0) return 0;
  return static_cast<uint8_t>((x < 0.0F ? 0x8U : 0x0U) | idx);
}

inline float Nibble(uint8_t nib) {
  return kE2M1[nib & 0x7U] * ((nib & 0x8U) ? -1.0F : 1.0F);
}

// ScaledFp4Quant CPU kernel: x [M,K] float -> out_packed [M,K/2] i8 + out_scale
// [M,K/16] i8. Per-token, per-16-group; equals vllm::RefScaledFp4Quant.
void ScaledFp4QuantKernel(Queue&, Tensor& out_packed, Tensor& out_scale, const Tensor& x,
                          float input_global_scale_inv) {
  const int64_t m = x.shape[0], k = x.shape[1];
  constexpr int kBS = 16;
  const int64_t groups = k / kBS;
  const float gs_recip = RecipF(input_global_scale_inv);
  auto* packed = out_packed.Ptr<uint8_t>();
  auto* scale = out_scale.Ptr<uint8_t>();
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t g = 0; g < groups; ++g) {
      const int64_t base = g * kBS;
      float vec_max = 0.0F;
      for (int j = 0; j < kBS; ++j) vec_max = std::fmax(vec_max, std::fabs(LoadF32(x, i * k + base + j)));
      float sc = ClampF(input_global_scale_inv * (vec_max * (1.0F / kFp4Max)), -kFp8Max, kFp8Max);
      const uint8_t sc_f8 = F32ToFp8(sc);
      scale[i * groups + g] = sc_f8;
      const float block_scale = Fp8ToF32(sc_f8) * gs_recip;
      const float out_scale_v = RecipF(block_scale);
      for (int j = 0; j < kBS; j += 2) {
        const float lo = ClampF(LoadF32(x, i * k + base + j) * out_scale_v, -kFp4Max, kFp4Max);
        const float hi = ClampF(LoadF32(x, i * k + base + j + 1) * out_scale_v, -kFp4Max, kFp4Max);
        packed[(i * k + base + j) / 2] =
            static_cast<uint8_t>(F32ToFp4Nibble(lo) | (F32ToFp4Nibble(hi) << 4));
      }
    }
  }
}

// SiluMulFp4Quant CPU fallback = the exact composite (bf16 intermediate then
// quant) — which IS the definition of correctness for the CUDA fused kernel. The
// bf16 scratch reproduces the round-through-bf16 the CUDA kernel folds in.
void SiluMulFp4QuantKernel(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& gate,
                           const Tensor& up, float input_global_scale_inv) {
  const int64_t m = gate.shape[0], i = gate.shape[1];
  std::vector<uint16_t> tmp(static_cast<size_t>(m) * static_cast<size_t>(i));
  Tensor act = Tensor::Contiguous(tmp.data(), DType::kBF16, gate.device, {m, i});
  MoeSiluMulKernel(q, act, gate, up);
  ScaledFp4QuantKernel(q, out_packed, out_scale, act, input_global_scale_inv);
}

// MatmulNvfp4Fp4 CPU kernel: out[m,n] = alpha * Σ_k (a_fp4·f8(a_scale))·(b_fp4·
// f8(b_scale)). Equals vllm::RunNvfp4Emulation up to K-reduction order.
void MatmulNvfp4Fp4Kernel(Queue&, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                          const Tensor& b_packed, const Tensor& b_scale, float alpha) {
  const int64_t m = a_packed.shape[0], k = a_packed.shape[1] * 2, n = b_packed.shape[0];
  constexpr int kBS = 16;
  const int64_t groups = k / kBS;
  const auto* ap = a_packed.Ptr<uint8_t>();
  const auto* as = a_scale.Ptr<uint8_t>();
  const auto* bp = b_packed.Ptr<uint8_t>();
  const auto* bs = b_scale.Ptr<uint8_t>();
  std::vector<float> arow(static_cast<size_t>(k));
  for (int64_t i = 0; i < m; ++i) {
    // Decode a_fp4·a_scale_fp8 for this row once (reused across N columns).
    for (int64_t g = 0; g < groups; ++g) {
      const float asf = Fp8ToF32(as[i * groups + g]);
      for (int j = 0; j < kBS / 2; ++j) {
        const uint8_t byte = ap[(i * k + g * kBS) / 2 + j];
        arow[static_cast<size_t>(g * kBS + 2 * j)] = Nibble(byte & 0x0FU) * asf;
        arow[static_cast<size_t>(g * kBS + 2 * j + 1)] = Nibble(byte >> 4) * asf;
      }
    }
    for (int64_t col = 0; col < n; ++col) {
      float acc = 0.0F;
      for (int64_t g = 0; g < groups; ++g) {
        const float bsf = Fp8ToF32(bs[col * groups + g]);
        for (int j = 0; j < kBS / 2; ++j) {
          const uint8_t byte = bp[(col * k + g * kBS) / 2 + j];
          acc += arow[static_cast<size_t>(g * kBS + 2 * j)] * (Nibble(byte & 0x0FU) * bsf);
          acc += arow[static_cast<size_t>(g * kBS + 2 * j + 1)] * (Nibble(byte >> 4) * bsf);
        }
      }
      StoreF32(out, i * n + col, alpha * acc);
    }
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

// Per-step RoPE cos|sin cache fill (fused-attn-preamble prep). cos_sin[T,rot] f32:
// cols [0,half)=cos, [half,rot)=sin. Angle math in DOUBLE + f32 cast, matching
// RopeRotateHead/RopeNeoxKernel element-for-element so the cache reproduces the
// inline rotation bit-for-bit.
void RopeCosSinCacheKernel(Queue&, Tensor& cos_sin, const Tensor& positions, const RopeArgs& args) {
  const int64_t t = cos_sin.shape[0];
  const int rot = args.rotary_dim;
  const int64_t half = rot / 2;
  const double base = static_cast<double>(args.base);
  for (int64_t i = 0; i < t; ++i) {
    const int64_t p =
        positions.dtype == DType::kI32 ? positions.Ptr<int32_t>()[i] : positions.Ptr<int64_t>()[i];
    for (int64_t pair = 0; pair < half; ++pair) {
      const double freq = std::pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
      const double angle = static_cast<double>(p) * freq;
      StoreF32(cos_sin, i * rot + pair, static_cast<float>(std::cos(angle)));
      StoreF32(cos_sin, i * rot + half + pair, static_cast<float>(std::sin(angle)));
    }
  }
}

// gemma-RMSNorm one element: (v*inv)*(gemma ? w+1 : w) — matches RmsNormKernel's
// `v * inv * wj` (wj = w [+1 if gemma]) grouping and order exactly.
float GemmaNormElem(float v, float inv, float w, bool gemma) {
  float wj = w;
  if (gemma) wj += 1.0f;
  return v * inv * wj;
}

// Fused full-attention preamble: split q|gate + gemma qk-RMSNorm(Dh) + partial
// NeoX RoPE (from the cos_sin cache) + gate passthrough, in one pass. Bit-for-bit
// equal (f32 out) to AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox composed:
// the variance is f32, the weight is applied as (1+w), and the rotation reuses the
// same f32 c/sn the cache holds (x*c - y*sn / x*sn + y*c). Tail dims [rot,Dh) are
// normed but unrotated.
void AttnQkNormRopeGateKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                              const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                              const Tensor& k_norm, const Tensor& cos_sin,
                              const RmsNormArgs& na, const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  const int rot = ra.rotary_dim;
  const int64_t half = rot / 2;
  const bool gemma = na.gemma;
  // Normalize one head row (src..src+Dh) into out.., applying partial NeoX RoPE
  // from cs[0..rot). Recomputes normed[i]/normed[i+half] where paired.
  auto do_head = [&](const Tensor& src, int64_t src_off, const Tensor& w, const Tensor& out,
                     int64_t out_off, const float* cs) {
    float ss = 0.0f;
    for (int64_t j = 0; j < dh; ++j) {
      const float v = LoadF32(src, src_off + j);
      ss += v * v;
    }
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dh) + na.eps);
    for (int64_t j = 0; j < dh; ++j) {
      if (j < half) {
        const float ni = GemmaNormElem(LoadF32(src, src_off + j), inv, LoadF32(w, j), gemma);
        const float nih =
            GemmaNormElem(LoadF32(src, src_off + j + half), inv, LoadF32(w, j + half), gemma);
        StoreF32(out, out_off + j, ni * cs[j] - nih * cs[half + j]);
      } else if (j < rot) {
        const int64_t i = j - half;
        const float ni = GemmaNormElem(LoadF32(src, src_off + i), inv, LoadF32(w, i), gemma);
        const float nih =
            GemmaNormElem(LoadF32(src, src_off + i + half), inv, LoadF32(w, i + half), gemma);
        StoreF32(out, out_off + j, ni * cs[half + i] + nih * cs[i]);
      } else {
        StoreF32(out, out_off + j,
                 GemmaNormElem(LoadF32(src, src_off + j), inv, LoadF32(w, j), gemma));
      }
    }
  };
  for (int64_t tok = 0; tok < t; ++tok) {
    const float* cs = cos_sin.Ptr<float>() + tok * rot;
    for (int64_t h = 0; h < hq; ++h) {
      const int64_t qrow = tok * (hq * 2 * dh) + h * 2 * dh;
      do_head(qgate, qrow, q_norm, q_out, (tok * hq + h) * dh, cs);
      // gate passthrough: the second Dh of each (t,h) q|gate pair (no norm/rope).
      const int64_t gbase = qrow + dh;
      const int64_t gout = (tok * hq + h) * dh;
      for (int64_t j = 0; j < dh; ++j) StoreF32(gate_out, gout + j, LoadF32(qgate, gbase + j));
    }
    for (int64_t h = 0; h < hkv; ++h) {
      do_head(kf, tok * (hkv * dh) + h * dh, k_norm, k_out, (tok * hkv + h) * dh, cs);
    }
  }
}

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

// §3 causal_conv1d_update (seqlen==1): read-old-then-roll. conv_state_indices
// (optional; mirrors mamba conv_state_indices): token bt's row is cache slot
// idx[bt] (idx<0 == NULL block → skip); null => compact row == bt.
void CausalConv1dUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const Tensor* conv_state_indices,
                              const CausalConv1dArgs& args) {
  const int64_t batch = x.shape[0], c_dim = x.shape[1], k = w.shape[1], width = k - 1;
  const int32_t* cache_idx =
      conv_state_indices != nullptr ? conv_state_indices->Ptr<int32_t>() : nullptr;
  for (int64_t bt = 0; bt < batch; ++bt) {
    int64_t srow_row = bt;
    if (cache_idx != nullptr) {
      if (cache_idx[bt] < 0) continue;  // NULL block
      srow_row = cache_idx[bt];
    }
    float* srow_base = conv_state.Ptr<float>() + srow_row * c_dim * width;
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
                     const Tensor& g, const Tensor& beta, Tensor& state,
                     const Tensor* state_idx, const GdnArgs& args) {
  const int64_t batch = q_in.shape[0], hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  const int32_t* sidx = state_idx != nullptr ? state_idx->Ptr<int32_t>() : nullptr;
  std::vector<float> qbuf(static_cast<size_t>(dk)), kbuf(static_cast<size_t>(dk)),
      vbuf(static_cast<size_t>(dv));
  for (int64_t bt = 0; bt < batch; ++bt) {
    // state_idx != null => in-place on the FULL cache at slot sidx[bt] (fla
    // ssm_state_indices); sidx[bt]<0 == NULL block → zero out; null => row == bt.
    int64_t srow = bt;
    if (sidx != nullptr) {
      if (sidx[bt] < 0) {
        for (int64_t hv = 0; hv < hv_n; ++hv)
          for (int64_t d = 0; d < dv; ++d)
            StoreF32(out, (bt * hv_n + hv) * dv + d, 0.0f);
        continue;
      }
      srow = sidx[bt];
    }
    float* s_state = state.Ptr<float>() + srow * hv_n * dv * dk;
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

// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). Elementwise fusions of the
// small host-side loops between the big decode ops; all math f32, dims inferred
// from the tensor shapes.

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// out[i] = F32ToBF16(in[i]); out bf16, in f32, same element count.
void CastBf16Kernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) StoreF32(out, i, LoadF32(in, i));
}

// Split fused [T, Hq*2*Dh] q/gate projection into q_out/gate_out [T,Hq,Dh].
void AttnGateSplitKernel(Queue&, Tensor& q_out, Tensor& gate_out, const Tensor& qgate) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  for (int64_t i = 0; i < t; ++i) {
    for (int64_t h = 0; h < hq; ++h) {
      const int64_t base = i * (hq * 2 * dh) + h * 2 * dh;  // start of (i,h) pair
      const int64_t out_off = (i * hq + h) * dh;
      for (int64_t d = 0; d < dh; ++d) {
        StoreF32(q_out, out_off + d, LoadF32(qgate, base + d));
        StoreF32(gate_out, out_off + d, LoadF32(qgate, base + dh + d));
      }
    }
  }
}

// out[i] = F32ToBF16(attn[i] * sigmoid(gate[i])); out bf16, attn/gate f32.
void SigmoidGateBf16Kernel(Queue&, Tensor& out, const Tensor& attn, const Tensor& gate) {
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) StoreF32(out, i, LoadF32(attn, i) * Sigmoid(LoadF32(gate, i)));
}

// GDN g/beta from raw projections (gdn-semantics.md §6). softplus threshold 20.
void GdnGBetaKernel(Queue&, Tensor& g_out, Tensor& beta_out, const Tensor& araw,
                    const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias) {
  const int64_t t = g_out.shape[0], hv = g_out.shape[1];
  for (int64_t i = 0; i < t; ++i) {
    for (int64_t h = 0; h < hv; ++h) {
      const int64_t idx = i * hv + h;
      const float x = LoadF32(araw, idx) + LoadF32(dt_bias, h);
      const float sp = x > 20.0f ? x : std::log1p(std::exp(x));  // softplus
      StoreF32(g_out, idx, -std::exp(LoadF32(a_log, h)) * sp);
      StoreF32(beta_out, idx, Sigmoid(LoadF32(braw, idx)));
    }
  }
}

// Split GDN conv [T, 2*key_dim+value_dim] into q/k [T,key_dim] and v [T,value_dim].
void GdnConvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& conv) {
  const int64_t t = conv.shape[0], key_dim = q_out.Numel() / t, value_dim = v_out.Numel() / t;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  for (int64_t i = 0; i < t; ++i) {
    const int64_t row = i * conv_dim;
    for (int64_t j = 0; j < key_dim; ++j) {
      StoreF32(q_out, i * key_dim + j, LoadF32(conv, row + j));
      StoreF32(k_out, i * key_dim + j, LoadF32(conv, row + key_dim + j));
    }
    for (int64_t j = 0; j < value_dim; ++j)
      StoreF32(v_out, i * value_dim + j, LoadF32(conv, row + 2 * key_dim + j));
  }
}

// Fused GDN post-conv prep: GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta in
// one pass (mirror of fla fused_gdn_prefill_post_conv). Bit-for-bit equal to
// composing those four ops (same f32 math, same softplus threshold 20).
void GdnPostConvKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                       Tensor& beta_out, const Tensor& conv, const Tensor& araw,
                       const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias,
                       const L2NormArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  for (int64_t i = 0; i < t; ++i) {
    const int64_t row = i * conv_dim;
    // q/k: split then l2norm over Dk, per head (plain SUM of squares, §4).
    for (int64_t h = 0; h < hk; ++h) {
      float qss = 0.0f, kss = 0.0f;
      for (int64_t j = 0; j < dk; ++j) {
        const float qv = LoadF32(conv, row + h * dk + j);
        const float kv = LoadF32(conv, row + key_dim + h * dk + j);
        qss += qv * qv;
        kss += kv * kv;
      }
      const float qinv = 1.0f / std::sqrt(qss + args.eps);
      const float kinv = 1.0f / std::sqrt(kss + args.eps);
      for (int64_t j = 0; j < dk; ++j) {
        const int64_t o = (i * hk + h) * dk + j;
        StoreF32(q_out, o, LoadF32(conv, row + h * dk + j) * qinv);
        StoreF32(k_out, o, LoadF32(conv, row + key_dim + h * dk + j) * kinv);
      }
    }
    // v: plain copy.
    for (int64_t j = 0; j < value_dim; ++j)
      StoreF32(v_out, i * value_dim + j, LoadF32(conv, row + 2 * key_dim + j));
    // g/beta from a/b + A_log/dt_bias (§6).
    for (int64_t h = 0; h < hv; ++h) {
      const int64_t idx = i * hv + h;
      const float x = LoadF32(araw, idx) + LoadF32(dt_bias, h);
      const float sp = x > 20.0f ? x : std::log1p(std::exp(x));  // softplus
      StoreF32(g_out, idx, -std::exp(LoadF32(a_log, h)) * sp);
      StoreF32(beta_out, idx, Sigmoid(LoadF32(braw, idx)));
    }
  }
}

// out[t,c] = F32ToBF16(sigmoid(gl[t]) * sd[t*H+c]); shared-expert sigmoid gate.
void SharedExpertGateKernel(Queue&, Tensor& out, const Tensor& sd, const Tensor& gl) {
  const int64_t t = out.shape[0], h = out.shape[1];
  for (int64_t i = 0; i < t; ++i) {
    const float g = Sigmoid(LoadF32(gl, i));
    for (int64_t c = 0; c < h; ++c) StoreF32(out, i * h + c, g * LoadF32(sd, i * h + c));
  }
}

// --- Fused declarative recipe (TDR Phase 0). Two realizations of ANY
// FusedRecipe over the Phase-0 vocabulary {kAdd, kMul, kRmsNorm}, selected at
// runtime by VT_FUSED_TIER (FusedTier()):
//   Tier 0 (default, safe): walk the steps, calling the ALREADY-REGISTERED
//     vt::RmsNorm primitive for the reduce step (composite over existing ops).
//   Tier 1: a single scalar pass per row that walks the recipe inline.
// Both are bit-identical to vt::RmsNorm(residual) for kFusedAddRmsNorm — same
// f32 variance accumulation, gemma (1+w), and residual-add rounding. Operand
// ROLES resolve to physical tensors: kIn->x, kResidual->residual (in/out),
// kWeight->weight ([H], per-column), kOut->out. Row tensors are [T,H].

// Read one operand element (row, column j). kWeight is per-column ([H]); the
// [T,H] roles index row*h + j.
float FusedLoad(FOperand o, int64_t row, int64_t j, int64_t h, const Tensor& x,
                const Tensor& weight, const Tensor* residual, const Tensor& out) {
  switch (o) {
    case FOperand::kIn: return LoadF32(x, row * h + j);
    case FOperand::kResidual: return LoadF32(*residual, row * h + j);
    case FOperand::kWeight: return LoadF32(weight, j);
    case FOperand::kOut: return LoadF32(out, row * h + j);
  }
  return 0.0f;
}

// Store one element into a WRITABLE operand (kResidual/kOut only), rounding to
// that operand's dtype (bf16 residual mirrors vLLM's model-dtype residual).
void FusedStore(FOperand o, int64_t row, int64_t j, int64_t h, float v, Tensor* residual,
                Tensor& out) {
  switch (o) {
    case FOperand::kResidual: StoreF32(*residual, row * h + j, v); break;
    case FOperand::kOut: StoreF32(out, row * h + j, v); break;
    case FOperand::kIn:
    case FOperand::kWeight: VT_CHECK(false, "fused_chain: step writes a read-only operand");
  }
}

// Tier 1 — scalar interpreter: walk the recipe over EACH ROW in one pass.
void FusedChainInterpKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                            Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  for (int64_t row = 0; row < t; ++row) {
    for (int s = 0; s < r.n; ++s) {
      const FStep& st = r.steps[s];
      switch (st.op) {
        case FOp::kAdd:
        case FOp::kMul:
          for (int64_t j = 0; j < h; ++j) {
            const float a = FusedLoad(st.a, row, j, h, x, weight, residual, out);
            const float b = FusedLoad(st.b, row, j, h, x, weight, residual, out);
            FusedStore(st.out, row, j, h, st.op == FOp::kAdd ? a + b : a * b, residual, out);
          }
          break;
        case FOp::kRmsNorm: {
          VT_CHECK(st.reduce == FReduce::kMeanSquare, "fused_chain: rmsnorm needs kMeanSquare");
          float sumsq = 0.0f;
          for (int64_t j = 0; j < h; ++j) {
            const float v = FusedLoad(st.a, row, j, h, x, weight, residual, out);
            sumsq += v * v;  // f32 variance accumulation
          }
          const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(h) + eps);
          for (int64_t j = 0; j < h; ++j) {
            const float v = FusedLoad(st.a, row, j, h, x, weight, residual, out);
            float wj = FusedLoad(st.b, row, j, h, x, weight, residual, out);
            if (st.gemma) wj += 1.0f;
            FusedStore(st.out, row, j, h, v * inv * wj, residual, out);
          }
          break;
        }
      }
    }
  }
}

// Tier 0 — composite: walk the steps and realize them through the ALREADY-
// REGISTERED vt:: primitives (re-dispatched through the op table). The residual-
// add + RMSNorm idiom collapses onto vt::RmsNorm(residual) — the registered
// fused_add_rms_norm primitive, which does add-then-normalize in f32. Folding the
// add INTO that call (rather than a naive add-then-RmsNorm(no-residual) split)
// both matches the golden bit-for-bit and keeps operand dtypes intact: RmsNorm
// takes x + weight in one dtype and the residual separately, so a bf16 x with a
// f32 residual works — a split would hand RmsNorm an x(=f32 residual)/weight(bf16)
// dtype mismatch that the CUDA kernel rejects. Same realization on both backends.
void FusedChainCompositeKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                               Tensor* residual, const FusedRecipe& r, float eps) {
  bool residual_add_pending = false;
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    switch (st.op) {
      case FOp::kAdd:
        // Residual-add producing the residual stream: fold it into the next
        // kRmsNorm's registered RmsNorm(residual) call.
        VT_CHECK(st.out == FOperand::kResidual && st.a == FOperand::kIn &&
                     st.b == FOperand::kResidual && residual != nullptr,
                 "fused_chain composite: only residual-add (out=res,a=in,b=res) is supported");
        residual_add_pending = true;
        break;
      case FOp::kMul:
        VT_CHECK(false, "fused_chain composite: kMul has no registered primitive yet (Phase 1)");
        break;
      case FOp::kRmsNorm:
        VT_CHECK(st.reduce == FReduce::kMeanSquare && st.out == FOperand::kOut &&
                     st.b == FOperand::kWeight,
                 "fused_chain composite: rmsnorm must write kOut with a kWeight");
        if (residual_add_pending) {
          VT_CHECK(st.a == FOperand::kResidual,
                   "fused_chain composite: rmsnorm after residual-add must read kResidual");
          vt::RmsNorm(q, out, x, weight, RmsNormArgs{eps, st.gemma}, residual);
          residual_add_pending = false;
        } else {
          // Plain RMSNorm (no preceding add): a is kIn or the raw residual.
          const Tensor& a = (st.a == FOperand::kIn) ? x : *residual;
          vt::RmsNorm(q, out, a, weight, RmsNormArgs{eps, st.gemma}, nullptr);
        }
        break;
    }
  }
}

// Registered kernel: pick the tier (default Tier-0 composite for safety).
void FusedChainKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  if (FusedTier() == 1) {
    FusedChainInterpKernel(q, out, x, weight, residual, r, eps);
  } else {
    FusedChainCompositeKernel(q, out, x, weight, residual, r, eps);
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
    RegisterOp(OpId::kScaledFp4Quant, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ScaledFp4QuantFn>(&ScaledFp4QuantKernel)));
    RegisterOp(OpId::kSiluMulFp4Quant, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SiluMulFp4QuantFn>(&SiluMulFp4QuantKernel)));
    RegisterOp(OpId::kMatmulNvfp4Fp4, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulNvfp4Fp4Fn>(&MatmulNvfp4Fp4Kernel)));
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
    RegisterOp(OpId::kCastBf16, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16Kernel)));
    RegisterOp(OpId::kAttnGateSplit, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttnGateSplitFn>(&AttnGateSplitKernel)));
    RegisterOp(OpId::kSigmoidGateBf16, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16Kernel)));
    RegisterOp(OpId::kGdnGBeta, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnGBetaFn>(&GdnGBetaKernel)));
    RegisterOp(OpId::kGdnConvSplit, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnConvSplitFn>(&GdnConvSplitKernel)));
    RegisterOp(OpId::kGdnPostConv, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernel)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernel)));
    RegisterOp(
        OpId::kAttnQkNormRopeGate, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernel)));
    RegisterOp(OpId::kSharedExpertGate, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SharedExpertGateFn>(&SharedExpertGateKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
