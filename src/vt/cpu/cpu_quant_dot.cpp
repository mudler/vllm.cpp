// Tier-0 generic `vec_dot` kernels — QUANT-GGUF-CIQ-GEMM work row G3.
//
// The portable-C++ block dot products, ported byte-for-byte from
// llama.cpp @ 237ad9b96 `ggml/src/ggml-cpu/quants.c`:
//   :174 ggml_vec_dot_q4_0_q8_0_generic
//   :400 ggml_vec_dot_q8_0_q8_0_generic
//   :514 ggml_vec_dot_q2_K_q8_K_generic     (DeepSeek-V4 W8)
//   :566 ggml_vec_dot_q3_K_q8_K_generic
//   :645 ggml_vec_dot_q4_K_q8_K_generic
//   :720 ggml_vec_dot_q5_K_q8_K_generic
//   :800 ggml_vec_dot_q6_K_q8_K_generic
//   :855 ggml_vec_dot_iq2_xxs_q8_K_generic  (DeepSeek-V4 W8 — keep-quant enabler)
//   :999 ggml_vec_dot_iq3_xxs_q8_K_generic  (DeepSeek-V4 W8 — keep-quant enabler)
// Q2_K/IQ2_XXS/IQ3_XXS landed for the single-Spark DeepSeek-V4 GGUF vehicle
// (CLAIM-DEEPSEEK-V4-W8): keeping the ~2-3-bit routed-expert weights COMPRESSED
// and dotting them directly is what keeps the 158 B model at ~91 GiB instead of
// OOM-expanding to bf16 (~316 GiB) — see .agents/specs/deepseek-v4-flash.md.
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
#include "cpu_quant_iq_tables.h"  // kIq2xxsGrid/kIq3xxsGrid/kKsignsIq2xs/kKmaskIq2xs
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

// quants.c:514 — ggml_vec_dot_q2_K_q8_K_generic. 2-bit weight (qs, shift
// 0/2/4/6) × a 4-bit per-16 sub-scale (low nibble of scales[]); the per-16
// sub-min (high nibble) is applied via the activation's bsums in one pass
// (`summs`), so the block minimum never touches the quant loop.
void VecDotQ2_KQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_q2_K_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q2_K_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ2_K* x = static_cast<const BlockQ2_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* q2 = x[i].qs;
    const int8_t* q8 = y[i].qs;
    const uint8_t* sc = x[i].scales;

    int summs = 0;
    for (int j = 0; j < 16; ++j) summs += y[i].bsums[j] * (sc[j] >> 4);

    const float dall = y[i].d * F16ToF32(x[i].d);
    const float dmin = y[i].d * F16ToF32(x[i].dmin);

    int isum = 0;
    int is = 0;
    int d;
    for (int k = 0; k < kQK_K / 128; ++k) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        d = sc[is++] & 0xF;
        int isuml = 0;
        for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
        isum += d * isuml;
        d = sc[is++] & 0xF;
        isuml = 0;
        for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
        isum += d * isuml;
        shift += 2;
        q8 += 32;
      }
      q2 += 32;
    }
    sumf += dall * isum - dmin * summs;
  }
  *s = sumf;
}

// quants.c:855 — ggml_vec_dot_iq2_xxs_q8_K_generic. Codebook dot: each 32-lane
// sub-block reads two u32 (four 8-bit grid indices + four 7-bit sign selectors
// with a 4-bit scale `ls` in the top nibble); the grid byte × activation ×
// (±1 sign) is accumulated and scaled by `ls`. The final 0.125 folds the grid's
// fixed 8x magnitude. The grid/sign tables live in cpu_quant_iq_tables.h.
void VecDotIQ2_XXSQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                       const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_iq2_xxs_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_iq2_xxs_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockIQ2_XXS* x = static_cast<const BlockIQ2_XXS*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  uint32_t aux32[2];
  const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);

  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const float d = F16ToF32(x[i].d) * y[i].d;
    const uint16_t* q2 = x[i].qs;
    const int8_t* q8 = y[i].qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
      std::memcpy(aux32, q2, 2 * sizeof(uint32_t));
      q2 += 4;
      const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
      int32_t sumi = 0;
      for (int l = 0; l < 4; ++l) {
        const uint8_t* grid =
            reinterpret_cast<const uint8_t*>(kIq2xxsGrid + aux8[l]);
        const uint8_t signs = kKsignsIq2xs[(aux32[1] >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j)
          sumi += grid[j] * q8[j] * ((signs & kKmaskIq2xs[j]) ? -1 : 1);
        q8 += 8;
      }
      bsum += sumi * static_cast<int32_t>(ls);
    }
    sumf += d * bsum;
  }
  *s = 0.125f * sumf;
}

// quants.c:999 — ggml_vec_dot_iq3_xxs_q8_K_generic. Codebook dot: `q3` holds
// QK_K/4 grid-index bytes (two 4-byte grid entries per lane), `gas` the per-32
// scale+sign u32s. The final 0.25 folds the grid's fixed 4x magnitude.
void VecDotIQ3_XXSQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                       const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_iq3_xxs_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_iq3_xxs_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockIQ3_XXS* x = static_cast<const BlockIQ3_XXS*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  uint32_t aux32;
  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const float d = F16ToF32(x[i].d) * y[i].d;
    const uint8_t* q3 = x[i].qs;
    const uint8_t* gas = x[i].qs + kQK_K / 4;
    const int8_t* q8 = y[i].qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
      std::memcpy(&aux32, gas, sizeof(uint32_t));
      gas += sizeof(uint32_t);
      const uint32_t ls = 2 * (aux32 >> 28) + 1;
      int32_t sumi = 0;
      for (int l = 0; l < 4; ++l) {
        const uint8_t* grid1 =
            reinterpret_cast<const uint8_t*>(kIq3xxsGrid + q3[2 * l + 0]);
        const uint8_t* grid2 =
            reinterpret_cast<const uint8_t*>(kIq3xxsGrid + q3[2 * l + 1]);
        const uint8_t signs = kKsignsIq2xs[(aux32 >> (7 * l)) & 127];
        for (int j = 0; j < 4; ++j) {
          sumi += grid1[j] * q8[j + 0] * ((signs & kKmaskIq2xs[j + 0]) ? -1 : 1);
          sumi += grid2[j] * q8[j + 4] * ((signs & kKmaskIq2xs[j + 4]) ? -1 : 1);
        }
        q8 += 8;
      }
      q3 += 8;
      bsum += sumi * static_cast<int32_t>(ls);
    }
    sumf += d * bsum;
  }
  *s = 0.25f * sumf;
}

// quants.c:1099 — ggml_vec_dot_iq1_s_q8_K_generic. Codebook dot: 8 sub-blocks of
// 32, each 4 lane groups of 8. The 11-bit grid index is `qs[l]` widened by 3
// bits from `qh[ib]`; kIq1sGrid entries are packed TERNARY (-1/0/+1) bytes, so
// there is no sign array to apply.
//
// The delta term is why this kernel reads `bsums` and the others do not. IQ1_S
// reconstructs a weight as dl*(grid[j] + delta), so the dot splits into
// sum(dl*grid*q8) plus delta*dl*sum(q8), and that second sum over each group of
// 16 is exactly what Q8_K already caches in `bsums`. Recomputing it from `qs`
// would be arithmetically identical but would read the activation twice.
void VecDotIQ1_SQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                     const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_iq1_s_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_iq1_s_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockIQ1_S* x = static_cast<const BlockIQ1_S*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const int8_t* q8 = y[i].qs;
    const uint8_t* qs = x[i].qs;
    const uint16_t* qh = x[i].qh;

    int sumi = 0;
    int sumi1 = 0;
    for (int ib = 0; ib < kQK_K / 32; ++ib) {
      const int ls = 2 * ((qh[ib] >> 12) & 7) + 1;
      const int delta = (qh[ib] & 0x8000) ? -1 : 1;
      int lsum = 0;
      for (int l = 0; l < 4; ++l) {
        const int8_t* grid = reinterpret_cast<const int8_t*>(
            kIq1sGrid + (qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8)));
        for (int j = 0; j < 8; ++j) lsum += q8[j] * grid[j];
        q8 += 8;
      }
      sumi += ls * lsum;
      sumi1 += ls * delta * (y[i].bsums[2 * ib + 0] + y[i].bsums[2 * ib + 1]);
      qs += 4;
    }

    sumf += F16ToF32(x[i].d) * y[i].d *
            (static_cast<float>(sumi) + kIq1sDelta * static_cast<float>(sumi1));
  }

  *s = sumf;
}

// PINNED FORK oracle `llama-cpp-unsloth` @ 36fe8e1cc, quants.c:1281
// ggml_vec_dot_iq1_xxxs_q8_K_generic. Structurally the IQ1_S dot with the sc
// NIBBLE standing in for qh: the 256-entry grid makes qs[l] a whole index, and
// one nibble carries both the sub-block scale (bits 0-2) and the delta sign
// (bit 3). The bsums split is identical and for the identical reason.
void VecDotIQ1_XXXSQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                        const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_iq1_xxxs_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1,
           "vec_dot_iq1_xxxs_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockIQ1_XXXS* x = static_cast<const BlockIQ1_XXXS*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const int8_t* q8 = y[i].qs;
    const uint8_t* qs = x[i].qs;
    const uint8_t* sc = x[i].sc;

    int sumi = 0;
    int sumi1 = 0;
    for (int ib = 0; ib < kQK_K / 32; ++ib) {
      const int nib = (sc[ib / 2] >> (4 * (ib & 1))) & 0xf;
      const int ls = 2 * (nib & 7) + 1;
      const int delta = (nib & 8) ? -1 : 1;
      int lsum = 0;
      for (int l = 0; l < 4; ++l) {
        const int8_t* grid =
            reinterpret_cast<const int8_t*>(kIq1xxxsGrid + qs[l]);
        for (int j = 0; j < 8; ++j) lsum += q8[j] * grid[j];
        q8 += 8;
      }
      sumi += ls * lsum;
      sumi1 += ls * delta * (y[i].bsums[2 * ib + 0] + y[i].bsums[2 * ib + 1]);
      qs += 4;
    }

    sumf += F16ToF32(x[i].d) * y[i].d *
            (static_cast<float>(sumi) + kIq1sDelta * static_cast<float>(sumi1));
  }

  *s = sumf;
}

// quants.c:947 — ggml_vec_dot_iq2_s_q8_K_generic. Codebook dot: 8 sub-blocks of
// 32. Each lane's 10-bit grid index (`qs[l] | qh high 2 bits`) picks a kIq2sGrid
// entry; the DIRECT sign byte `signs[l]` (= qs + QK_K/8, NO ksigns lookup) flips
// lanes; the per-32 scales fold in as `ls = 1 + 2*ls_nibble` (l=0,1 share the low
// nibble, l=2,3 the high). The final 0.125 folds the grid's fixed 8x magnitude
// (as in IQ2_XXS). Grid/sign tables in cpu_quant_iq_tables.h.
void VecDotIQ2_SQ8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                     const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_iq2_s_q8_K: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_iq2_s_q8_K: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockIQ2_S* x = static_cast<const BlockIQ2_S*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const float d = F16ToF32(x[i].d) * y[i].d;
    const int8_t* q8 = y[i].qs;
    const uint8_t* qs = x[i].qs;
    const uint8_t* qh = x[i].qh;
    const uint8_t* signs = qs + kQK_K / 8;
    int bsum = 0;
    for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
      const int ls1 = 1 + 2 * (x[i].scales[ib32] & 0xf);
      const int ls2 = 1 + 2 * (x[i].scales[ib32] >> 4);
      int sumi1 = 0;
      int sumi2 = 0;
      for (int l = 0; l < 2; ++l) {
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(
            kIq2sGrid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
        for (int j = 0; j < 8; ++j)
          sumi1 += q8[j] * grid[j] * ((signs[l] & kKmaskIq2xs[j]) ? -1 : 1);
        q8 += 8;
      }
      for (int l = 2; l < 4; ++l) {
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(
            kIq2sGrid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
        for (int j = 0; j < 8; ++j)
          sumi2 += q8[j] * grid[j] * ((signs[l] & kKmaskIq2xs[j]) ? -1 : 1);
        q8 += 8;
      }
      bsum += ls1 * sumi1 + ls2 * sumi2;
      qs += 4;
      signs += 4;
    }
    sumf += d * bsum;
  }
  *s = 0.125f * sumf;
}

// quants.c:247 — ggml_vec_dot_mxfp4_q8_0_generic. UNLIKE the K-quants above,
// MXFP4 dots against a Q8_0 activation (QK_MXFP4 == QK8_0 == 32), not Q8_K.
// Integer core: kValuesMxfp4 (int8) x the q8_0 quants (int8), scaled per 32-block
// by F16(q8_0.d) * E8M0ToF32Half(mxfp4.e). Split-half nibble packing (like q4_0).
void VecDotMXFP4Q8_0(int n, float* s, size_t bs, const void* vx, size_t bx,
                     const void* vy, size_t by, int nrc) {
  const int qk = kQK_MXFP4;
  VT_CHECK(n % qk == 0, "vec_dot_mxfp4_q8_0: n must be a multiple of 32");
  VT_CHECK(nrc == 1, "vec_dot_mxfp4_q8_0: generic tier supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockMXFP4* x = static_cast<const BlockMXFP4*>(vx);
  const BlockQ8_0* y = static_cast<const BlockQ8_0*>(vy);
  const int nb = n / qk;

  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const float d = F16ToF32(y[ib].d) * E8M0ToF32Half(x[ib].e);
    int sumi1 = 0;
    int sumi2 = 0;
    for (int j = 0; j < qk / 2; ++j) {
      sumi1 += y[ib].qs[j + 0] * kValuesMxfp4[x[ib].qs[j] & 0xf];
      sumi2 += y[ib].qs[j + qk / 2] * kValuesMxfp4[x[ib].qs[j] >> 4];
    }
    sumf += d * (sumi1 + sumi2);
  }
  *s = sumf;
}

}  // namespace

// KERNEL-CPU-A76-Q8-DOT test/benchmark seam: the TRUE portable Q8_0 x Q8_0
// reference (quants.c:400 accumulation order), independent of the runtime
// SelectQuantQ8VecDot choice. On an A76 the selected QuantTraits vec_dot is
// the assembly tier, so exact-order comparisons must reference THIS symbol.
VecDotFn QuantQ8PortableVecDot() { return &VecDotQ8_0Q8_0; }

VecDotFn BlockVecDot(DType dtype) {
  switch (dtype) {
    case DType::kQ4_0: return &VecDotQ4_0Q8_0;        // quants.c:174
    case DType::kQ8_0:
      return SelectQuantQ8VecDot(&VecDotQ8_0Q8_0);  // quants.c:400 + A76 tier
    case DType::kQ2_K: return &VecDotQ2_KQ8_K;        // quants.c:514
    case DType::kQ3_K: return &VecDotQ3_KQ8_K;        // quants.c:566
    case DType::kQ4_K: return &VecDotQ4_KQ8_K;        // quants.c:645
    case DType::kQ5_K: return &VecDotQ5_KQ8_K;        // quants.c:720
    case DType::kQ6_K: return &VecDotQ6_KQ8_K;        // quants.c:800
    case DType::kIQ2_XXS: return &VecDotIQ2_XXSQ8_K;  // quants.c:855
    case DType::kIQ3_XXS: return &VecDotIQ3_XXSQ8_K;  // quants.c:999
    case DType::kIQ2_S: return &VecDotIQ2_SQ8_K;      // quants.c:947
    case DType::kIQ1_S: return &VecDotIQ1_SQ8_K;      // quants.c:1099
    case DType::kIQ1_XXXS: return &VecDotIQ1_XXXSQ8_K;  // fork quants.c:1281
    case DType::kMXFP4: return &VecDotMXFP4Q8_0;      // quants.c:247
    default:
      // kQ8_K is the ACTIVATION encoding — upstream gives it no vec_dot row
      // (it is only ever the `y` side of the K-quant kernels above), so a
      // Q8_K "weight" correctly stays on the dequant-composite fallback.
      return nullptr;
  }
}

}  // namespace vt::cpu
