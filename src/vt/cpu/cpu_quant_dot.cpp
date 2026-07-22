// Tier-0 generic `vec_dot` kernels — QUANT-GGUF-CIQ-GEMM work row G3.
//
// The six portable-C++ block dot products, ported byte-for-byte from
// llama.cpp @ 237ad9b96 `ggml/src/ggml-cpu/quants.c`:
//   :174 ggml_vec_dot_q4_0_q8_0_generic
//   :400 ggml_vec_dot_q8_0_q8_0_generic
//   :566 ggml_vec_dot_q3_K_q8_K_generic
//   :645 ggml_vec_dot_q4_K_q8_K_generic
//   :720 ggml_vec_dot_q5_K_q8_K_generic
//   :800 ggml_vec_dot_q6_K_q8_K_generic
// `GGML_CPU_FP16_TO_FP32` maps to `vt::F16ToF32` (the same IEEE binary16
// decode) and the `*_generic` names are retained in each comment so an
// upstream diff lands mechanically.
//
// THIS IS THE PORTABLE TIER ONLY. The x86 AVX2/AVX512 variants
// (`arch/x86/quants.c`) are work row G5 and the Arm NEON/dotprod/i8mm variants
// (`arch/arm/quants.c`) are G6; upstream's `nrows == 2` mmla rows stay
// unreachable until G6 brings both the mmla kernels AND the odd-shape boundary
// guards at `ggml-cpu.c:1426-1433`, so every kernel here asserts `nrc == 1`
// exactly as its upstream counterpart does.
//
// Why the odd-looking scalar structure is preserved verbatim (upstream's own
// comment at quants.c:583-590): these bodies are shaped so the compiler
// auto-vectorizes them: the decode-into-`aux8`-then-dot split, the 8-wide
// `aux16`/`aux32` staging, and the deferred `sums[8]` reduction are all
// load-bearing for that. Rewriting them "more naturally" measured 4x slower
// upstream. They also fix the REDUCTION ORDER, which is what makes our GEMM
// bit-reproducible run to run.
//
// The `bs`/`bx`/`by` row strides are part of upstream's signature but unused on
// the nrc==1 tier (they only carry meaning for the 2-row mmla kernels); they
// are kept in the signature so G5/G6 drop in without touching call sites.
#include <cstring>

#include "cpu_quant_blocks.h"
#include "vt/quant.h"

namespace vt::cpu {
namespace {

// quants.c:174 — ggml_vec_dot_q4_0_q8_0_generic
void VecDotQ4_0Q8_0(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  const int qk = kQK8_0;
  const int nb = n / qk;

  VT_CHECK(n % qk == 0, "vec_dot_q4_0_q8_0: n must be a multiple of 32");
  VT_CHECK(nrc == 1, "vec_dot_q4_0_q8_0: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ4_0* x = static_cast<const BlockQ4_0*>(vx);
  const BlockQ8_0* y = static_cast<const BlockQ8_0*>(vy);

  int ib = 0;
  float sumf = 0;

  for (; ib < nb; ++ib) {
    int sumi0 = 0;
    int sumi1 = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[ib].qs[j] & 0x0F) - 8;
      const int v1 = (x[ib].qs[j] >> 4) - 8;

      sumi0 += (v0 * y[ib].qs[j]);
      sumi1 += (v1 * y[ib].qs[j + qk / 2]);
    }

    int sumi = sumi0 + sumi1;
    sumf += sumi * F16ToF32(x[ib].d) * F16ToF32(y[ib].d);
  }

  *s = sumf;
}

// quants.c:400 — ggml_vec_dot_q8_0_q8_0_generic
void VecDotQ8_0Q8_0(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  const int qk = kQK8_0;
  const int nb = n / qk;

  VT_CHECK(n % qk == 0, "vec_dot_q8_0_q8_0: n must be a multiple of 32");
  VT_CHECK(nrc == 1, "vec_dot_q8_0_q8_0: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ8_0* x = static_cast<const BlockQ8_0*>(vx);
  const BlockQ8_0* y = static_cast<const BlockQ8_0*>(vy);

  int ib = 0;
  float sumf = 0;

  for (; ib < nb; ++ib) {
    int sumi = 0;

    for (int j = 0; j < qk; j++) {
      sumi += x[ib].qs[j] * y[ib].qs[j];
    }

    sumf += sumi * (F16ToF32(x[ib].d) * F16ToF32(y[ib].d));
  }

  *s = sumf;
}

// quants.c:566 — ggml_vec_dot_q3_K_q8_K_generic
void VecDotQ3_KQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_q3_K_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q3_K_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  const BlockQ3_K* x = static_cast<const BlockQ3_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);

  const int nb = n / kQK_K;

  int8_t aux8[kQK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  std::memset(sums, 0, 8 * sizeof(float));

  uint32_t auxs[4];
  const int8_t* scales = reinterpret_cast<const int8_t*>(auxs);

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* q3 = x[i].qs;
    const uint8_t* hm = x[i].hmask;
    const int8_t* q8 = y[i].qs;
    std::memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < kQK_K; j += 128) {
      for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
      for (int l = 0; l < 32; ++l)
        a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
      a += 32;
      m = static_cast<uint8_t>(m << 1);
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
      for (int l = 0; l < 32; ++l)
        a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
      a += 32;
      m = static_cast<uint8_t>(m << 1);
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
      for (int l = 0; l < 32; ++l)
        a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
      a += 32;
      m = static_cast<uint8_t>(m << 1);
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
      for (int l = 0; l < 32; ++l)
        a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
      a += 32;
      m = static_cast<uint8_t>(m << 1);
      q3 += 32;
    }
    a = aux8;

    std::memcpy(auxs, x[i].scales, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (int j = 0; j < kQK_K / 16; ++j) {
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = F16ToF32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
}

// quants.c:645 — ggml_vec_dot_q4_K_q8_K_generic
void VecDotQ4_KQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_q4_K_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q4_K_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ4_K* x = static_cast<const BlockQ4_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);

  const int nb = n / kQK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);

  int8_t aux8[kQK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  std::memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* q4 = x[i].qs;
    const int8_t* q8 = y[i].qs;
    std::memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* a = aux8;
    for (int j = 0; j < kQK_K / 64; ++j) {
      for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] & 0xF);
      a += 32;
      for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] >> 4);
      a += 32;
      q4 += 32;
    }
    std::memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    int sumi = 0;
    for (int j = 0; j < kQK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
    a = aux8;
    int is = 0;
    for (int j = 0; j < kQK_K / 32; ++j) {
      int32_t scale = scales[is++];
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = F16ToF32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    const float dmin = F16ToF32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
}

// quants.c:720 — ggml_vec_dot_q5_K_q8_K_generic
void VecDotQ5_KQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_q5_K_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q5_K_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ5_K* x = static_cast<const BlockQ5_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);

  const int nb = n / kQK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);

  int8_t aux8[kQK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  std::memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* q4 = x[i].qs;
    const uint8_t* hm = x[i].qh;
    const int8_t* q8 = y[i].qs;
    std::memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < kQK_K / 64; ++j) {
      for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] & 0xF);
      for (int l = 0; l < 32; ++l)
        a[l] = static_cast<int8_t>(a[l] + ((hm[l] & m) ? 16 : 0));
      a += 32;
      m = static_cast<uint8_t>(m << 1);
      for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] >> 4);
      for (int l = 0; l < 32; ++l)
        a[l] = static_cast<int8_t>(a[l] + ((hm[l] & m) ? 16 : 0));
      a += 32;
      m = static_cast<uint8_t>(m << 1);
      q4 += 32;
    }
    std::memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    int sumi = 0;
    for (int j = 0; j < kQK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
    a = aux8;
    int is = 0;
    for (int j = 0; j < kQK_K / 32; ++j) {
      int32_t scale = scales[is++];
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = F16ToF32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    const float dmin = F16ToF32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
}

// quants.c:800 — ggml_vec_dot_q6_K_q8_K_generic
void VecDotQ6_KQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_q6_K_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q6_K_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ6_K* x = static_cast<const BlockQ6_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);

  const int nb = n / kQK_K;

  int8_t aux8[kQK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  std::memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* q4 = x[i].ql;
    const uint8_t* qh = x[i].qh;
    const int8_t* q8 = y[i].qs;
    std::memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* a = aux8;
    for (int j = 0; j < kQK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        a[l + 0] = static_cast<int8_t>(
            static_cast<int8_t>((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
        a[l + 32] = static_cast<int8_t>(
            static_cast<int8_t>((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
        a[l + 64] = static_cast<int8_t>(
            static_cast<int8_t>((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
        a[l + 96] = static_cast<int8_t>(
            static_cast<int8_t>((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
      }
      a += 128;
      q4 += 64;
      qh += 32;
    }
    a = aux8;
    int is = 0;
    for (int j = 0; j < kQK_K / 16; ++j) {
      int scale = x[i].scales[is++];
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l] * a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = F16ToF32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
}

}  // namespace

VecDotFn BlockVecDot(DType dtype) {
  switch (dtype) {
    case DType::kQ4_0: return &VecDotQ4_0Q8_0;  // quants.c:174
    case DType::kQ8_0: return &VecDotQ8_0Q8_0;  // quants.c:400
    case DType::kQ3_K: return &VecDotQ3_KQ8_K;  // quants.c:566
    case DType::kQ4_K: return &VecDotQ4_KQ8_K;  // quants.c:645
    case DType::kQ5_K: return &VecDotQ5_KQ8_K;  // quants.c:720
    case DType::kQ6_K: return &VecDotQ6_KQ8_K;  // quants.c:800
    default:
      // kQ8_K is the ACTIVATION encoding — upstream gives it no vec_dot row
      // (it is only ever the `y` side of the K-quant kernels above), so a
      // Q8_K "weight" correctly stays on the dequant-composite fallback.
      return nullptr;
  }
}

}  // namespace vt::cpu
