// Arm i8mm (mmla) `vec_dot` tier — QUANT-GGUF-CIQ-GEMM work row G6.
//
// The `nrc == 2` int8-matmul (`vmmlaq_s32`) block dot products, ported
// byte-for-byte from llama.cpp @ 237ad9b96 `ggml/src/ggml-cpu/arch/arm/quants.c`
// (the `#if/#elif defined(__ARM_FEATURE_MATMUL_INT8)` branches):
//   :241  ggml_vec_dot_q4_0_q8_0  (nrc==2 mmla block)
//   :1094 ggml_vec_dot_q8_0_q8_0  (nrc==2 mmla block)
//   :2495 ggml_vec_dot_q4_K_q8_K  (#elif MATMUL_INT8, nrc==2 mmla block)
//   :3101 ggml_vec_dot_q6_K_q8_K  (#elif MATMUL_INT8, nrc==2 mmla block)
// Upstream has an mmla path only for these four encodings; q3_K and q5_K have
// none (SVE + NEON only), so on the mmla tier they legitimately stay on the
// portable nrc==1 kernels (cpu_quant_dot.cpp) — faithfully "port the mmla path
// where upstream has it", nothing invented.
//
// A single mmla `vec_dot(nrc=2)` produces a 2x2 output tile: two consecutive
// WEIGHT rows (src0, stride `bx`) against two consecutive ACTIVATION rows
// (src1, stride `by`), written as s[0]=(w0,a0), s[1]=(w1,a0), s[bs]=(w0,a1),
// s[bs+1]=(w1,a1) — the exact convention `ggml_compute_forward_mul_mat_one_chunk`
// (`ggml-cpu.c:1233-1239`) consumes. The GEMM driver (cpu_quant_gemm.cpp) tiles
// on that; here we only mirror the kernels.
//
// NUMERICS. The int8 products accumulate into i32 EXACTLY (integer add is
// associative), so `vmmlaq_s32`'s regrouping cannot move a bit versus the
// scalar integer dot. What can differ is the FLOAT scale-accumulate: upstream
// (and this port) apply it with `vmlaq_f32`, the ACLE non-fused multiply-add
// (round(prod) then round(sum)), block by block in the same order the scalar
// kernel uses — so Q8_0/Q4_0 (whose only float step is that MAC) are expected
// bit-identical to the portable tier, while the K-quants add a `vpaddq`/`vmull`
// bias reduction that DOES reassociate and are therefore gated at the f64
// dequant-dot NMSE bar. Both facts are asserted directly in
// tests/vt/test_ops_quant_dot.cpp (tier cross-check).
//
// COMPILE/RUNTIME GATING. This translation unit is compiled with `+i8mm` for
// aarch64 (CMakeLists per-file COMPILE_OPTIONS) so `arm_neon.h` exposes
// `vmmlaq_s32`; the whole body is additionally `#if defined(__aarch64__) &&
// defined(__ARM_FEATURE_MATMUL_INT8)` so a build without the feature (or any
// non-Arm build) links the empty stubs and the portable tier runs everywhere.
// At RUNTIME the mmla kernels are handed out ONLY when `getauxval(AT_HWCAP2) &
// HWCAP2_I8MM` — selecting an mmla kernel on a CPU that lacks i8mm would be an
// illegal-instruction crash. `VT_CPU_QUANT_MMLA=0|off|false` forces the tier
// off for a same-binary A/B (the portable nrc==1 path then serves every shape).
#include "vt/quant.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)

#include <arm_neon.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>

#include <cstdlib>
#include <cstring>

#include "cpu_quant_blocks.h"

#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif

namespace vt::cpu {
namespace {

// GGML_CPU_FP16_TO_FP32 on this target is an IEEE-exact fp16->fp32 widen; so is
// vt::F16ToF32. Using the latter keeps the scale bit-identical to the portable
// tier's `F16ToF32(x.d)`.
inline float H2F(uint16_t h) { return F16ToF32(h); }

// quants.c:1094 — ggml_vec_dot_q8_0_q8_0, nrc==2 mmla block.
void VecDotMmlaQ8_0(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  const int qk = kQK8_0;
  const int nb = n / qk;
  VT_CHECK(n % qk == 0, "vec_dot_mmla_q8_0: n must be a multiple of 32");
  VT_CHECK(nrc == 2, "vec_dot_mmla_q8_0: mmla tier is nrc == 2 only");

  const BlockQ8_0* vx0 = static_cast<const BlockQ8_0*>(vx);
  const BlockQ8_0* vx1 =
      reinterpret_cast<const BlockQ8_0*>(static_cast<const uint8_t*>(vx) + bx);
  const BlockQ8_0* vy0 = static_cast<const BlockQ8_0*>(vy);
  const BlockQ8_0* vy1 =
      reinterpret_cast<const BlockQ8_0*>(static_cast<const uint8_t*>(vy) + by);

  float32x4_t sumv0 = vdupq_n_f32(0.0f);

  for (int i = 0; i < nb; i++) {
    const BlockQ8_0* b_x0 = &vx0[i];
    const BlockQ8_0* b_y0 = &vy0[i];
    const BlockQ8_0* b_x1 = &vx1[i];
    const BlockQ8_0* b_y1 = &vy1[i];

    const int8x16_t x0_l = vld1q_s8(b_x0->qs);
    const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
    const int8x16_t x1_l = vld1q_s8(b_x1->qs);
    const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

    const int8x16_t y0_l = vld1q_s8(b_y0->qs);
    const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
    const int8x16_t y1_l = vld1q_s8(b_y1->qs);
    const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

    float32_t _scale[4] = {H2F(b_x0->d) * H2F(b_y0->d), H2F(b_x0->d) * H2F(b_y1->d),
                           H2F(b_x1->d) * H2F(b_y0->d), H2F(b_x1->d) * H2F(b_y1->d)};
    float32x4_t scale = vld1q_f32(_scale);

    int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
    int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
    int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
    int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

    int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
    int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
    int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
    int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

    sumv0 = vmlaq_f32(sumv0,
                      vcvtq_f32_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), l0, r0),
                                                                     l1, r1),
                                                          l2, r2),
                                               l3, r3)),
                      scale);
  }

  float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
  float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);
  vst1_f32(s, vget_low_f32(sumv2));
  vst1_f32(s + bs, vget_high_f32(sumv2));
}

// quants.c:241 — ggml_vec_dot_q4_0_q8_0, nrc==2 mmla block.
void VecDotMmlaQ4_0(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  const int qk = kQK8_0;
  const int nb = n / qk;
  VT_CHECK(n % qk == 0, "vec_dot_mmla_q4_0: n must be a multiple of 32");
  VT_CHECK(nrc == 2, "vec_dot_mmla_q4_0: mmla tier is nrc == 2 only");

  const BlockQ4_0* vx0 = static_cast<const BlockQ4_0*>(vx);
  const BlockQ4_0* vx1 =
      reinterpret_cast<const BlockQ4_0*>(static_cast<const uint8_t*>(vx) + bx);
  const BlockQ8_0* vy0 = static_cast<const BlockQ8_0*>(vy);
  const BlockQ8_0* vy1 =
      reinterpret_cast<const BlockQ8_0*>(static_cast<const uint8_t*>(vy) + by);

  float32x4_t sumv0 = vdupq_n_f32(0.0f);

  for (int i = 0; i < nb; i++) {
    const BlockQ4_0* b_x0 = &vx0[i];
    const BlockQ4_0* b_x1 = &vx1[i];
    const BlockQ8_0* b_y0 = &vy0[i];
    const BlockQ8_0* b_y1 = &vy1[i];

    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(0x8);

    const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
    const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

    const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
    const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
    const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
    const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

    const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
    const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
    const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
    const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

    const int8x16_t y0_l = vld1q_s8(b_y0->qs);
    const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
    const int8x16_t y1_l = vld1q_s8(b_y1->qs);
    const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

    float32_t _scale[4] = {H2F(b_x0->d) * H2F(b_y0->d), H2F(b_x0->d) * H2F(b_y1->d),
                           H2F(b_x1->d) * H2F(b_y0->d), H2F(b_x1->d) * H2F(b_y1->d)};
    float32x4_t scale = vld1q_f32(_scale);

    int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
    int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
    int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
    int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

    int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
    int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
    int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
    int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

    sumv0 = vmlaq_f32(sumv0,
                      vcvtq_f32_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), l0, r0),
                                                                     l1, r1),
                                                          l2, r2),
                                               l3, r3)),
                      scale);
  }

  float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
  float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);
  vst1_f32(s, vget_low_f32(sumv2));
  vst1_f32(s + bs, vget_high_f32(sumv2));
}

// quants.c:2495 — ggml_vec_dot_q4_K_q8_K, #elif MATMUL_INT8 nrc==2 mmla block.
void VecDotMmlaQ4_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_mmla_q4_K: n must be a multiple of 256");
  VT_CHECK(nrc == 2, "vec_dot_mmla_q4_K: mmla tier is nrc == 2 only");

  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;

  const int nb = n / kQK_K;

  const BlockQ4_K* x0 = static_cast<const BlockQ4_K*>(vx);
  const BlockQ4_K* x1 =
      reinterpret_cast<const BlockQ4_K*>(static_cast<const uint8_t*>(vx) + bx);
  const BlockQ8_K* y0 = static_cast<const BlockQ8_K*>(vy);
  const BlockQ8_K* y1 =
      reinterpret_cast<const BlockQ8_K*>(static_cast<const uint8_t*>(vy) + by);

  const uint8x16_t m4b = vdupq_n_u8(0x0f);
  float32x4_t vfsum = vdupq_n_f32(0.0f);

  for (int i = 0; i < nb; ++i, ++x0, ++x1, ++y0, ++y1) {
    const uint8_t* qx0 = x0->qs;
    const uint8_t* qx1 = x1->qs;
    const int8_t* qy0 = y0->qs;
    const int8_t* qy1 = y1->qs;

    int8_t x0_scales[8], x1_scales[8];
    int16x8_t x0_mins, x1_mins;
    {
      uint32_t scales_mins[3];
      std::memcpy(scales_mins, x0->scales, 12);
      const uint32_t mins_0_3 = scales_mins[1] & kmask1;
      const uint32_t mins_4_7 = ((scales_mins[2] >> 4) & kmask2) | (((scales_mins[1] >> 6) & kmask3) << 4);
      const uint32x2_t mins = {mins_0_3, mins_4_7};
      x0_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
      uint32_t scales[2];
      scales[0] = scales_mins[0] & kmask1;
      scales[1] = (scales_mins[2] & kmask2) | (((scales_mins[0] >> 6) & kmask3) << 4);
      std::memcpy(x0_scales, scales, 8);
    }
    {
      uint32_t scales_mins[3];
      std::memcpy(scales_mins, x1->scales, 12);
      const uint32_t mins_0_3 = scales_mins[1] & kmask1;
      const uint32_t mins_4_7 = ((scales_mins[2] >> 4) & kmask2) | (((scales_mins[1] >> 6) & kmask3) << 4);
      const uint32x2_t mins = {mins_0_3, mins_4_7};
      x1_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
      uint32_t scales[2];
      scales[0] = scales_mins[0] & kmask1;
      scales[1] = (scales_mins[2] & kmask2) | (((scales_mins[0] >> 6) & kmask3) << 4);
      std::memcpy(x1_scales, scales, 8);
    }

    int32x4_t visum = {0};

    for (int j = 0; j < kQK_K / 64; ++j, qx0 += 32, qx1 += 32, qy0 += 64, qy1 += 64) {
      const int8x16x4_t vy0v = vld1q_s8_x4(qy0);
      const int8x16x4_t vy1v = vld1q_s8_x4(qy1);

      int8x16_t vx0[4], vx1[4];
      {
        const uint8x16x2_t vv = vld1q_u8_x2(qx0);
        vx0[0] = vreinterpretq_s8_u8(vandq_u8(vv.val[0], m4b));
        vx0[1] = vreinterpretq_s8_u8(vandq_u8(vv.val[1], m4b));
        vx0[2] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[0], 4));
        vx0[3] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[1], 4));
      }
      {
        const uint8x16x2_t vv = vld1q_u8_x2(qx1);
        vx1[0] = vreinterpretq_s8_u8(vandq_u8(vv.val[0], m4b));
        vx1[1] = vreinterpretq_s8_u8(vandq_u8(vv.val[1], m4b));
        vx1[2] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[0], 4));
        vx1[3] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[1], 4));
      }

      for (int k = 0; k < 2; ++k) {
        const int blk = j * 2 + k;
        const int32x4_t block_scale = {
            x0_scales[blk],
            x0_scales[blk],
            x1_scales[blk],
            x1_scales[blk],
        };

        int32x4_t vr = {0};
        for (int l = 0; l < 2; ++l) {
          const int idx = k * 2 + l;
          const int64x2_t vx0_s64 = vreinterpretq_s64_s8(vx0[idx]);
          const int64x2_t vx1_s64 = vreinterpretq_s64_s8(vx1[idx]);
          const int64x2_t vy0_s64 = vreinterpretq_s64_s8(vy0v.val[idx]);
          const int64x2_t vy1_s64 = vreinterpretq_s64_s8(vy1v.val[idx]);
          const int8x16_t vx_l = vreinterpretq_s8_s64(vzip1q_s64(vx0_s64, vx1_s64));
          const int8x16_t vx_h = vreinterpretq_s8_s64(vzip2q_s64(vx0_s64, vx1_s64));
          const int8x16_t vy_l = vreinterpretq_s8_s64(vzip1q_s64(vy0_s64, vy1_s64));
          const int8x16_t vy_h = vreinterpretq_s8_s64(vzip2q_s64(vy0_s64, vy1_s64));
          vr = vmmlaq_s32(vr, vx_l, vy_l);
          vr = vmmlaq_s32(vr, vx_h, vy_h);
        }
        visum = vmlaq_s32(visum, vr, block_scale);
      }
    }

    {
      int32_t bias[4];
      const int16x8_t y0_sums = vpaddq_s16(vld1q_s16(y0->bsums), vld1q_s16(y0->bsums + 8));
      const int16x8_t y1_sums = vpaddq_s16(vld1q_s16(y1->bsums), vld1q_s16(y1->bsums + 8));
      bias[0] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y0_sums), vget_low_s16(x0_mins)),
                                     vmull_s16(vget_high_s16(y0_sums), vget_high_s16(x0_mins))));
      bias[1] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y1_sums), vget_low_s16(x0_mins)),
                                     vmull_s16(vget_high_s16(y1_sums), vget_high_s16(x0_mins))));
      bias[2] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y0_sums), vget_low_s16(x1_mins)),
                                     vmull_s16(vget_high_s16(y0_sums), vget_high_s16(x1_mins))));
      bias[3] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y1_sums), vget_low_s16(x1_mins)),
                                     vmull_s16(vget_high_s16(y1_sums), vget_high_s16(x1_mins))));
      const float32x4_t dmins = {
          H2F(x0->dmin) * y0->d,
          H2F(x0->dmin) * y1->d,
          H2F(x1->dmin) * y0->d,
          H2F(x1->dmin) * y1->d,
      };
      vfsum = vmlsq_f32(vfsum, vcvtq_f32_s32(vld1q_s32(bias)), dmins);

      const float32x4_t superblock_scale = {
          H2F(x0->d) * y0->d,
          H2F(x0->d) * y1->d,
          H2F(x1->d) * y0->d,
          H2F(x1->d) * y1->d,
      };
      vfsum = vmlaq_f32(vfsum, vcvtq_f32_s32(visum), superblock_scale);
    }
  }

  vfsum = vzip1q_f32(vfsum, vextq_f32(vfsum, vfsum, 2));
  vst1_f32(s, vget_low_f32(vfsum));
  vst1_f32(s + bs, vget_high_f32(vfsum));
}

// quants.c:3101 — ggml_vec_dot_q6_K_q8_K, #elif MATMUL_INT8 nrc==2 mmla block.
void VecDotMmlaQ6_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                    const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_mmla_q6_K: n must be a multiple of 256");
  VT_CHECK(nrc == 2, "vec_dot_mmla_q6_K: mmla tier is nrc == 2 only");

  const int nb = n / kQK_K;

  const BlockQ6_K* x0 = static_cast<const BlockQ6_K*>(vx);
  const BlockQ6_K* x1 =
      reinterpret_cast<const BlockQ6_K*>(static_cast<const uint8_t*>(vx) + bx);
  const BlockQ8_K* y0 = static_cast<const BlockQ8_K*>(vy);
  const BlockQ8_K* y1 =
      reinterpret_cast<const BlockQ8_K*>(static_cast<const uint8_t*>(vy) + by);

  float32x4_t vfsum = vdupq_n_f32(0.0f);

  for (int i = 0; i < nb; ++i, ++x0, ++x1, ++y0, ++y1) {
    const uint8_t* ql0 = x0->ql;
    const uint8_t* ql1 = x1->ql;
    const uint8_t* qh0 = x0->qh;
    const uint8_t* qh1 = x1->qh;
    const int8_t* qy0 = y0->qs;
    const int8_t* qy1 = y1->qs;

    const uint8x16_t mone = vdupq_n_u8(0x30);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);

    int32x4_t visum = vdupq_n_s32(0);

    for (int j = 0; j < 2; ++j, qh0 += 32, ql0 += 64, qh1 += 32, ql1 += 64) {
      int8x16_t vx0[8], vx1[8];

      {
        const uint8x16x2_t qh_bits = vld1q_u8_x2(qh0);
        const uint8x16x4_t ql_bits = vld1q_u8_x4(ql0);

        uint8x16_t q6h_0 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 4));
        uint8x16_t q6h_1 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 4));
        uint8x16_t q6h_2 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 2));
        uint8x16_t q6h_3 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 2));

        vx0[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[0], m4b), q6h_0));
        vx0[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[1], m4b), q6h_1));
        vx0[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[2], m4b), q6h_2));
        vx0[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[3], m4b), q6h_3));

        q6h_0 = vandq_u8(mone, qh_bits.val[0]);
        q6h_1 = vandq_u8(mone, qh_bits.val[1]);
        q6h_2 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[0], 2));
        q6h_3 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[1], 2));

        vx0[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[0], 4), q6h_0));
        vx0[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[1], 4), q6h_1));
        vx0[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[2], 4), q6h_2));
        vx0[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[3], 4), q6h_3));
      }

      {
        const uint8x16x2_t qh_bits = vld1q_u8_x2(qh1);
        const uint8x16x4_t ql_bits = vld1q_u8_x4(ql1);

        uint8x16_t q6h_0 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 4));
        uint8x16_t q6h_1 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 4));
        uint8x16_t q6h_2 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 2));
        uint8x16_t q6h_3 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 2));

        vx1[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[0], m4b), q6h_0));
        vx1[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[1], m4b), q6h_1));
        vx1[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[2], m4b), q6h_2));
        vx1[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[3], m4b), q6h_3));

        q6h_0 = vandq_u8(mone, qh_bits.val[0]);
        q6h_1 = vandq_u8(mone, qh_bits.val[1]);
        q6h_2 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[0], 2));
        q6h_3 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[1], 2));

        vx1[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[0], 4), q6h_0));
        vx1[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[1], 4), q6h_1));
        vx1[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[2], 4), q6h_2));
        vx1[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[3], 4), q6h_3));
      }

      for (int k = 0; k < 8; ++k) {
        const int blk = j * 8 + k;

        const int8x16_t vy0 = vld1q_s8(qy0);
        const int8x16_t vy1 = vld1q_s8(qy1);
        qy0 += 16;
        qy1 += 16;

        const int32x4_t block_scale = {
            x0->scales[blk],
            x0->scales[blk],
            x1->scales[blk],
            x1->scales[blk],
        };

        const int8x16_t vx_l = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(vx0[k]), vreinterpretq_s64_s8(vx1[k])));
        const int8x16_t vx_h = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(vx0[k]), vreinterpretq_s64_s8(vx1[k])));
        const int8x16_t vy_l = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(vy0), vreinterpretq_s64_s8(vy1)));
        const int8x16_t vy_h = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(vy0), vreinterpretq_s64_s8(vy1)));
        int32x4_t vr = vdupq_n_s32(0);
        vr = vmmlaq_s32(vr, vx_l, vy_l);
        vr = vmmlaq_s32(vr, vx_h, vy_h);

        visum = vmlaq_s32(visum, vr, block_scale);
      }
    }

    {
      int32_t bias[4];
      const int16x8x2_t q8sums0 = vld1q_s16_x2(y0->bsums);
      const int16x8x2_t q8sums1 = vld1q_s16_x2(y1->bsums);

      int8x16_t scales_s8 = vld1q_s8(x0->scales);
      const int16x8x2_t q6scales0 = {{vmovl_s8(vget_low_s8(scales_s8)), vmovl_s8(vget_high_s8(scales_s8))}};
      scales_s8 = vld1q_s8(x1->scales);
      const int16x8x2_t q6scales1 = {{vmovl_s8(vget_low_s8(scales_s8)), vmovl_s8(vget_high_s8(scales_s8))}};

      int32x4_t prod;
      prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums0.val[0]), vget_low_s16(q6scales0.val[0])),
                                 vmull_s16(vget_high_s16(q8sums0.val[0]), vget_high_s16(q6scales0.val[0]))),
                       vaddq_s32(vmull_s16(vget_low_s16(q8sums0.val[1]), vget_low_s16(q6scales0.val[1])),
                                 vmull_s16(vget_high_s16(q8sums0.val[1]), vget_high_s16(q6scales0.val[1]))));
      bias[0] = vaddvq_s32(prod);
      prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums1.val[0]), vget_low_s16(q6scales0.val[0])),
                                 vmull_s16(vget_high_s16(q8sums1.val[0]), vget_high_s16(q6scales0.val[0]))),
                       vaddq_s32(vmull_s16(vget_low_s16(q8sums1.val[1]), vget_low_s16(q6scales0.val[1])),
                                 vmull_s16(vget_high_s16(q8sums1.val[1]), vget_high_s16(q6scales0.val[1]))));
      bias[1] = vaddvq_s32(prod);
      prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums0.val[0]), vget_low_s16(q6scales1.val[0])),
                                 vmull_s16(vget_high_s16(q8sums0.val[0]), vget_high_s16(q6scales1.val[0]))),
                       vaddq_s32(vmull_s16(vget_low_s16(q8sums0.val[1]), vget_low_s16(q6scales1.val[1])),
                                 vmull_s16(vget_high_s16(q8sums0.val[1]), vget_high_s16(q6scales1.val[1]))));
      bias[2] = vaddvq_s32(prod);
      prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums1.val[0]), vget_low_s16(q6scales1.val[0])),
                                 vmull_s16(vget_high_s16(q8sums1.val[0]), vget_high_s16(q6scales1.val[0]))),
                       vaddq_s32(vmull_s16(vget_low_s16(q8sums1.val[1]), vget_low_s16(q6scales1.val[1])),
                                 vmull_s16(vget_high_s16(q8sums1.val[1]), vget_high_s16(q6scales1.val[1]))));
      bias[3] = vaddvq_s32(prod);

      const int32x4_t vibias = vmulq_n_s32(vld1q_s32(bias), 32);

      const float32x4_t superblock_scale = {
          H2F(x0->d) * y0->d,
          H2F(x0->d) * y1->d,
          H2F(x1->d) * y0->d,
          H2F(x1->d) * y1->d,
      };

      visum = vsubq_s32(visum, vibias);
      vfsum = vmlaq_f32(vfsum, vcvtq_f32_s32(visum), superblock_scale);
    }
  }

  vfsum = vzip1q_f32(vfsum, vextq_f32(vfsum, vfsum, 2));
  vst1_f32(s, vget_low_f32(vfsum));
  vst1_f32(s + bs, vget_high_f32(vfsum));
}

}  // namespace

bool QuantMmlaActive() {
  static const bool v = [] {
    const char* e = std::getenv("VT_CPU_QUANT_MMLA");
    if (e != nullptr &&
        (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 || std::strcmp(e, "false") == 0)) {
      return false;
    }
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
  }();
  return v;
}

VecDotFn QuantMmlaVecDot(DType dtype) {
  if (!QuantMmlaActive()) return nullptr;
  switch (dtype) {
    case DType::kQ8_0: return &VecDotMmlaQ8_0;  // quants.c:1094
    case DType::kQ4_0: return &VecDotMmlaQ4_0;  // quants.c:241
    case DType::kQ4_K: return &VecDotMmlaQ4_K;  // quants.c:2495
    case DType::kQ6_K: return &VecDotMmlaQ6_K;  // quants.c:3101
    default:
      // q3_K / q5_K have no upstream mmla path; every other dtype is not a
      // routed block weight. They stay on the portable nrc==1 tier.
      return nullptr;
  }
}

}  // namespace vt::cpu

#else  // no aarch64 i8mm — empty stubs so the portable tier serves every shape.

namespace vt::cpu {
bool QuantMmlaActive() { return false; }
VecDotFn QuantMmlaVecDot(DType) { return nullptr; }
}  // namespace vt::cpu

#endif
