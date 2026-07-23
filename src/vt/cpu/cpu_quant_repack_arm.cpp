// CIQ G7 repack-at-load — Arm i8mm gemm/gemv kernels + dispatch.
//
// Ported from llama.cpp @ 237ad9b96 `ggml/src/ggml-cpu/arch/arm/repack.cpp`:
//   ggml_gemm_q8_0_4x8_q8_0  :5006  (NEON + i8mm 4x4-tile prefill kernel)
//   ggml_gemv_q8_0_4x8_q8_0  :1757  (NEON + dotprod M=1 / leftover-row kernel)
// selected for q8_0 on NEON + i8mm by `ggml_repack_get_optimal_repack_type`
// (repack.cpp:4683 -> q8_0_4x8_q8_0 == tensor_traits<block_q8_0, 8, 4>).
//
// DEVIATION for BIT-IDENTITY (recorded). Upstream folds the per-block scale with
// the FUSED `vfmaq_f32`; this port uses the NON-FUSED `vmlaq_f32` in the SAME
// block order, so — combined with the global -ffp-contract=off — the float
// accumulation is the exact `scale = d_w*d_a; acc = cvt(int)*scale + acc`
// sequence the tier-0 / G6 mmla kernels use (cpu_quant_dot_arm.cpp). The integer
// mmla/dot sums are exact regardless of grouping, and the interleave carries the
// quant values verbatim, so the repacked GEMM is byte-for-byte equal to
// `kMatmulBTQuant`'s non-repacked output (memcmp round-trip test).
//
// COMPILE/RUNTIME GATING mirrors the mmla tier: compiled with +i8mm for one TU
// (CMakeLists), body guarded by `__aarch64__ && __ARM_FEATURE_MATMUL_INT8`, and
// handed out at runtime only when HWCAP2_I8MM is set and VT_CPU_QUANT_REPACK is
// not disabled. Off i8mm aarch64 the stubs make QuantRepackActive() false, so
// the loader never repacks and this code is never reached.
#include "vt/quant.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)

#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "cpu_quant_blocks.h"
#include "cpu_quant_repack.h"
#include "cpu_threadpool.h"
#include "vt/tensor.h"

#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif

namespace vt::cpu {
namespace {

float LoadActF32(const Tensor& t, int64_t elem_offset) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_offset];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    default:
      VT_CHECK(false, "quant_repack: unsupported activation dtype");
      return 0.0f;
  }
}

void StoreOutF32(const Tensor& t, int64_t elem_offset, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[elem_offset] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[elem_offset] = F32ToBF16(v); break;
    default: VT_CHECK(false, "quant_repack: unsupported output dtype");
  }
}

// ggml_gemm_q8_0_4x8_q8_0 (NEON+i8mm branch, repack.cpp:5091-5155), restricted
// to a range of weight COLUMN-groups [xg0, xg1) so the driver can parallelize
// over them. Runs the full set of activation ROW-groups (mgroups). `wgroups` and
// `agroups` are the repacked weight / interleaved activation, laid out as
// group-major (each group = `nblocks` BlockQ8_0x4). `s` is f32 [*, N] row stride
// `bs`. Non-fused vmla — see the file header (bit-identity).
void GemmTileQ8_0(float* s, int64_t bs, const BlockQ8_0x4* wgroups,
                  const BlockQ8_0x4* agroups, int64_t mgroups, int64_t nblocks,
                  int64_t xg0, int64_t xg1) {
  for (int64_t yg = 0; yg < mgroups; ++yg) {
    const BlockQ8_0x4* a_base = agroups + yg * nblocks;
    for (int64_t xg = xg0; xg < xg1; ++xg) {
      const BlockQ8_0x4* b_ptr = wgroups + xg * nblocks;
      const BlockQ8_0x4* a_ptr = a_base;

      float32x4_t acc_f32[4];
      for (int i = 0; i < 4; i++) acc_f32[i] = vdupq_n_f32(0);

      for (int64_t bi = 0; bi < nblocks; ++bi) {
        int32x4_t acc[4];
        for (int i = 0; i < 4; i++) acc[i] = vdupq_n_s32(0);

        for (int chunk = 0; chunk < 4; chunk++) {
          int8x16_t a01 = vld1q_s8(a_ptr->qs + chunk * 32);
          int8x16_t a23 = vld1q_s8(a_ptr->qs + chunk * 32 + 16);
          int8x16_t b01 = vld1q_s8(b_ptr->qs + chunk * 32);
          int8x16_t b23 = vld1q_s8(b_ptr->qs + chunk * 32 + 16);

          acc[0] = vmmlaq_s32(acc[0], a01, b01);
          acc[1] = vmmlaq_s32(acc[1], a01, b23);
          acc[2] = vmmlaq_s32(acc[2], a23, b01);
          acc[3] = vmmlaq_s32(acc[3], a23, b23);
        }

        int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
        int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
        int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
        int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));

        float32x4_t a_d = vcvt_f32_f16(vld1_f16((const __fp16*)a_ptr->d));
        float32x4_t b_d = vcvt_f32_f16(vld1_f16((const __fp16*)b_ptr->d));

        acc_f32[0] = vmlaq_f32(acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
        acc_f32[1] = vmlaq_f32(acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
        acc_f32[2] = vmlaq_f32(acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
        acc_f32[3] = vmlaq_f32(acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));

        a_ptr++;
        b_ptr++;
      }

      const int64_t x = xg * 4;
      for (int row = 0; row < 4; row++) {
        vst1q_f32(s + (yg * 4 + row) * bs + x, acc_f32[row]);
      }
    }
  }
}

// ggml_gemv_q8_0_4x8_q8_0 (NEON+dotprod branch, repack.cpp:1779-1820) for ONE
// activation row (plain BlockQ8_0, NOT interleaved), restricted to weight
// column-groups [xg0, xg1). `s_row` is the f32 output row. Non-fused vmla.
void GemvRowQ8_0(float* s_row, const BlockQ8_0x4* wgroups,
                 const BlockQ8_0* a_row, int64_t nblocks, int64_t xg0,
                 int64_t xg1) {
  for (int64_t xg = xg0; xg < xg1; ++xg) {
    const BlockQ8_0x4* b_ptr = wgroups + xg * nblocks;
    const BlockQ8_0* a_ptr = a_row;
    float32x4_t acc = vdupq_n_f32(0);

    for (int64_t bi = 0; bi < nblocks; ++bi) {
      int8x16x4_t b_low = vld1q_s8_x4((const int8_t*)b_ptr->qs);
      int8x16x4_t b_high = vld1q_s8_x4((const int8_t*)b_ptr->qs + 64);
      float16x4_t bd = vld1_f16((const __fp16*)b_ptr->d);

      int8x8x4_t a_chunks = vld1_s8_x4(a_ptr->qs);
      int8x16_t a0 = vcombine_s8(a_chunks.val[0], a_chunks.val[0]);
      int8x16_t a1 = vcombine_s8(a_chunks.val[1], a_chunks.val[1]);
      int8x16_t a2 = vcombine_s8(a_chunks.val[2], a_chunks.val[2]);
      int8x16_t a3 = vcombine_s8(a_chunks.val[3], a_chunks.val[3]);
      float16x4_t ad = vld1_dup_f16((const __fp16*)&a_ptr->d);

      int32x4_t ret0 = vdupq_n_s32(0);
      int32x4_t ret1 = vdupq_n_s32(0);
      ret0 = vdotq_s32(ret0, b_low.val[0], a0);
      ret1 = vdotq_s32(ret1, b_low.val[1], a0);
      ret0 = vdotq_s32(ret0, b_low.val[2], a1);
      ret1 = vdotq_s32(ret1, b_low.val[3], a1);
      ret0 = vdotq_s32(ret0, b_high.val[0], a2);
      ret1 = vdotq_s32(ret1, b_high.val[1], a2);
      ret0 = vdotq_s32(ret0, b_high.val[2], a3);
      ret1 = vdotq_s32(ret1, b_high.val[3], a3);

      int32x4_t ret = vpaddq_s32(ret0, ret1);
      acc = vmlaq_f32(acc, vcvtq_f32_s32(ret),
                      vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
      a_ptr++;
      b_ptr++;
    }
    vst1q_f32(s_row + xg * 4, acc);
  }
}

}  // namespace

bool QuantRepackActive() {
  static const bool v = [] {
    const char* e = std::getenv("VT_CPU_QUANT_REPACK");
    if (e != nullptr && (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 ||
                         std::strcmp(e, "false") == 0)) {
      return false;
    }
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
  }();
  return v;
}

void QuantRepackMatmul(Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(b.repacked && b.dtype == DType::kQ8_0,
           "quant_repack_matmul: weight must be a repacked q8_0 tensor");
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = b.shape[0];
  const int64_t nblocks = k / kQK8_0;
  const int64_t a_rs = a.stride[0];
  VT_CHECK(n % kQ8_0xNrowsInterleaved == 0 && k % kQK8_0 == 0,
           "quant_repack_matmul: N%4 and K%32 required (repack eligibility)");

  const FromFloatFn from_float = QuantTraits(DType::kQ8_0).from_float;
  const BlockQ8_0x4* wgroups = reinterpret_cast<const BlockQ8_0x4*>(b.data);

  // Output: write f32 directly when possible, else into an f32 scratch that is
  // converted to bf16 at the end. MatmulBTQuant guarantees `out` contiguous, so
  // the f32 row stride is exactly N.
  const bool out_is_f32 = (out.dtype == DType::kF32);
  std::vector<float> out_scratch;
  float* s;
  if (out_is_f32) {
    s = out.Ptr<float>();
  } else {
    out_scratch.resize(static_cast<size_t>(m) * n);
    s = out_scratch.data();
  }
  const int64_t bs = n;
  const int64_t ncol_groups = n / kQ8_0xNrowsInterleaved;

  // Prefill: activation rows in groups of 4 -> the i8mm gemm.
  const int64_t mgroups = m / kQ8_0xNrowsInterleaved;
  if (mgroups > 0) {
    // Interleave the first mgroups*4 activation rows into group-major
    // BlockQ8_0x4, quantizing each row with the SAME from_float as tier-0.
    std::vector<uint8_t> act(static_cast<size_t>(mgroups) * nblocks *
                             sizeof(BlockQ8_0x4));
    BlockQ8_0x4* actb = reinterpret_cast<BlockQ8_0x4*>(act.data());
    std::vector<BlockQ8_0> tmp(static_cast<size_t>(kQ8_0xNrowsInterleaved) * nblocks);
    std::vector<float> row(static_cast<size_t>(k));
    for (int64_t g = 0; g < mgroups; ++g) {
      for (int r = 0; r < kQ8_0xNrowsInterleaved; ++r) {
        const int64_t i = g * kQ8_0xNrowsInterleaved + r;
        for (int64_t p = 0; p < k; ++p) row[p] = LoadActF32(a, i * a_rs + p);
        from_float(row.data(), tmp.data() + r * nblocks, k);
      }
      InterleaveQ8_0Rows4(tmp.data() + 0 * nblocks, tmp.data() + 1 * nblocks,
                          tmp.data() + 2 * nblocks, tmp.data() + 3 * nblocks,
                          actb + g * nblocks, nblocks);
    }
    ParallelForRows(CurrentThreadpool(), ncol_groups,
                    [&](int64_t xg0, int64_t xg1) {
                      GemmTileQ8_0(s, bs, wgroups, actb, mgroups, nblocks, xg0,
                                   xg1);
                    });
  }

  // Leftover rows (m % 4, includes decode m == 1): the i8mm/dotprod gemv, one
  // plain-q8_0 activation row at a time.
  const int64_t mg_rows = mgroups * kQ8_0xNrowsInterleaved;
  if (mg_rows < m) {
    for (int64_t i = mg_rows; i < m; ++i) {
      std::vector<BlockQ8_0> arow(static_cast<size_t>(nblocks));
      std::vector<float> row(static_cast<size_t>(k));
      for (int64_t p = 0; p < k; ++p) row[p] = LoadActF32(a, i * a_rs + p);
      from_float(row.data(), arow.data(), k);
      float* s_row = s + i * bs;
      ParallelForRows(CurrentThreadpool(), ncol_groups,
                      [&](int64_t xg0, int64_t xg1) {
                        GemvRowQ8_0(s_row, wgroups, arow.data(), nblocks, xg0,
                                    xg1);
                      });
    }
  }

  if (!out_is_f32) {
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        StoreOutF32(out, i * n + j, out_scratch[static_cast<size_t>(i) * n + j]);
      }
    }
  }
}

}  // namespace vt::cpu

#else  // no aarch64 i8mm — stubs; the loader never repacks, so these are inert.

#include "vt/tensor.h"

namespace vt::cpu {
bool QuantRepackActive() { return false; }
void QuantRepackMatmul(Tensor&, const Tensor&, const Tensor&) {
  VT_CHECK(false, "quant_repack_matmul: repack tier is not built on this target");
}
}  // namespace vt::cpu

#endif
