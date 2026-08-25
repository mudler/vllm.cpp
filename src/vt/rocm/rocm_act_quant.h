// Shared ROCm device-side Q8_K activation-superblock quantizer (Lever C,
// GFX1100-TG200-NORMQ). One source of truth for the byte-exactness-critical
// numeric path: src/vt/rocm/rocm_grouped_gemm.hip (standalone QuantizeQ8KK +
// MMVQ fused-prologue) AND src/vt/rocm/rocm_rmsnorm.hip (producer-fused
// epilogue behind VT_NORM_QUANT_FUSED=1) both instantiate THIS body, so
// "byte-equal vs standalone" holds by construction rather than by two copies
// drifting. Contract carried over from cuda_quant_dot.cu QuantizeQ8KPreqKernel:
// the amax carries its ORIGINAL element index and ties break by LOWEST index
// (`ax > amax`, never `>=`); tests assert this on tied-amax rows.
//
// The helpers here were moved verbatim out of rocm_grouped_gemm.hip's
// anonymous namespace (clean cutover, no second copy left behind); every
// consumer in that file keeps resolving the same names through this include.
#ifndef VLLM_CPP_SRC_VT_ROCM_ROCM_ACT_QUANT_H_
#define VLLM_CPP_SRC_VT_ROCM_ROCM_ACT_QUANT_H_

#include <cstdint>
#include "vt/dtype.h"
#include "vt/cpu/cpu_quant_blocks.h"

namespace vt::rocm {

enum class ActDT : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

inline ActDT ActDtOf(DType dt) {
  return dt == DType::kF32 ? ActDT::kF32 : dt == DType::kF16 ? ActDT::kF16 : ActDT::kBF16;
}

__device__ inline float DF16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0x1F) return __int_as_float(sign | 0x7F800000 | (mant << 13));
  if (exp == 0) {
    if (mant == 0) return __int_as_float(sign);
    int shift = 0;
    while ((mant & 0x400) == 0) { mant <<= 1; ++shift; }
    mant &= 0x3FF;
    return __int_as_float(sign | ((113 - shift) << 23) | (mant << 13));
  }
  return __int_as_float(sign | ((exp + 112) << 23) | (mant << 13));
}
__device__ inline float DBF16ToF32(uint16_t b) {
  return __int_as_float(static_cast<uint32_t>(b) << 16);
}
__device__ inline uint16_t DF32ToBF16(float f) {
  uint32_t u = __float_as_int(f);
  if ((u & 0x7F800000) == 0x7F800000 && (u & 0x7FFFFF))
    return static_cast<uint16_t>((u >> 16) | 0x0040);
  uint32_t rounding = 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>((u + rounding) >> 16);
}
__device__ inline uint16_t DF32ToF16(float f) {
  uint32_t u = __float_as_uint(f);
  uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000);
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = u & 0x7FFFFF;
  if (((u >> 23) & 0xFF) == 0xFF)
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 | (mant >> 13) : 0));
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
  return static_cast<uint16_t>(sign | half);
}
__device__ inline int DNearestInt(float fval) {
  float val = fval + 12582912.0f;
  int i = __float_as_int(val);
  return (i & 0x007fffff) - 0x00400000;
}
__device__ inline float DLoadAct(const void* base, ActDT dt, int64_t idx) {
  switch (dt) {
    case ActDT::kF32: return static_cast<const float*>(base)[idx];
    case ActDT::kF16: return DF16ToF32(static_cast<const uint16_t*>(base)[idx]);
    default: return DBF16ToF32(static_cast<const uint16_t*>(base)[idx]);
  }
}

// Q8_K (thread-per-256-superblock): cuda_quant_dot.cu QuantizeQ8KKernel.
// The per-super-block body is factored so EVERY arm that produces Q8_K
// activation scratch (standalone grid, MMVQ LDS prologue, norm-fused
// epilogue) produces BYTE-IDENTICAL output: same amax first-occurrence
// tie-break, same scale/iscale arithmetic, same bsums walk. Asserted by
// tests/vt/test_rocm_quant_dot.cpp on random AND tied-amax inputs.
//
// T5 vectorization (2026-08-25): the body used to run two SCALAR passes over
// the superblock with the ActDT switch re-executed per element, which priced
// every consumer at ~50us per launch against a <2us memory floor (see
// docs/bench-evidence/gfx1100-tg200-t5-native-baseline-20260825.md). The
// dtype is now unswitched ONCE into a template parameter, and the loads go
// through 16-byte vectors when the address allows it (callers pass elem0 as a
// multiple of kQK_K=256, so the element offset never breaks alignment; the
// guard covers the base pointer itself). BYTE-EXACTNESS ARGUMENT: the amax
// scan still visits elements in STRICTLY ASCENDING order with the same
// `ax > amax` comparison, so the recorded mx/amax pair is bit-identical to
// the scalar walk including the first-occurrence lowest-index tie-break; the
// quantized values depend only on (x_j, iscale) through the same DNearestInt
// arithmetic and are order-independent; bsums are integer sums, exact under
// any order. No numeric op was added, removed, or reordered.
__device__ inline void QuantQ8KSBlockScalar(vt::cpu::BlockQ8_K& y,
                                            const void* __restrict__ a, ActDT adt,
                                            int64_t elem0) {
  using vt::cpu::kQK_K;
  float mx = 0.0f, amax = 0.0f;
  for (int j = 0; j < kQK_K; ++j) {
    const float x = DLoadAct(a, adt, elem0 + j);
    if (const float ax = fabsf(x); ax > amax) { amax = ax; mx = x; }
  }
  if (amax == 0.0f) {
    y.d = 0.0f;
    for (int j = 0; j < kQK_K; ++j) y.qs[j] = 0;
    for (int g = 0; g < kQK_K / 16; ++g) y.bsums[g] = 0;
    return;
  }
  const float iscale = -127.0f / mx;
  for (int j = 0; j < kQK_K; ++j) {
    const int v = DNearestInt(iscale * DLoadAct(a, adt, elem0 + j));
    y.qs[j] = static_cast<int8_t>(v < 127 ? v : 127);
  }
  for (int g = 0; g < kQK_K / 16; ++g) {
    int sum = 0;
    for (int ii = 0; ii < 16; ++ii) sum += y.qs[g * 16 + ii];
    y.bsums[g] = static_cast<int16_t>(sum);
  }
  y.d = 1.0f / iscale;
}

template <ActDT adt>
__device__ inline void QuantQ8KSBlockBody(vt::cpu::BlockQ8_K& y,
                                          const void* __restrict__ a, int64_t elem0) {
  using vt::cpu::kQK_K;
  constexpr bool kIsF32 = adt == ActDT::kF32;
  constexpr int kW = kIsF32 ? 4 : 8;  // one 16-byte vector = kW elements
  const char* const base = static_cast<const char*>(a);
  const size_t kElemBytes = kIsF32 ? 4 : 2;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(base) & 15u) == 0 &&
      ((static_cast<size_t>(elem0) * kElemBytes) & 15u) == 0;
  if (!aligned) {
    QuantQ8KSBlockScalar(y, a, adt, elem0);
    return;
  }
  float mx = 0.0f, amax = 0.0f;
  for (int j = 0; j < kQK_K; j += kW) {
    float v[kW];
    if constexpr (kIsF32) {
      const float4 raw = *reinterpret_cast<const float4*>(base + (elem0 + j) * 4);
      v[0] = raw.x; v[1] = raw.y; v[2] = raw.z; v[3] = raw.w;
    } else {
      const uint4 raw = *reinterpret_cast<const uint4*>(base + (elem0 + j) * 2);
      const uint16_t h[8] = {static_cast<uint16_t>(raw.x & 0xFFFF),
                             static_cast<uint16_t>(raw.x >> 16),
                             static_cast<uint16_t>(raw.y & 0xFFFF),
                             static_cast<uint16_t>(raw.y >> 16),
                             static_cast<uint16_t>(raw.z & 0xFFFF),
                             static_cast<uint16_t>(raw.z >> 16),
                             static_cast<uint16_t>(raw.w & 0xFFFF),
                             static_cast<uint16_t>(raw.w >> 16)};
      if constexpr (adt == ActDT::kF16) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = DF16ToF32(h[u]);
      } else {
        #pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = DBF16ToF32(h[u]);
      }
    }
    // Ascending element order preserved: this IS the serial tie-break scan.
    #pragma unroll
    for (int u = 0; u < kW; ++u) {
      if (const float ax = fabsf(v[u]); ax > amax) { amax = ax; mx = v[u]; }
    }
  }
  if (amax == 0.0f) {
    y.d = 0.0f;
    for (int j = 0; j < kQK_K; ++j) y.qs[j] = 0;
    for (int g = 0; g < kQK_K / 16; ++g) y.bsums[g] = 0;
    return;
  }
  const float iscale = -127.0f / mx;
  // Second pass re-reads the superblock; at 512 B/row it is L1/L2-resident.
  for (int j = 0; j < kQK_K; j += kW) {
    #pragma unroll
    for (int u = 0; u < kW; ++u) {
      const int idx = j + u;
      const int qv = DNearestInt(iscale * DLoadAct(a, adt, elem0 + idx));
      y.qs[idx] = static_cast<int8_t>(qv < 127 ? qv : 127);
    }
  }
  for (int g = 0; g < kQK_K / 16; ++g) {
    int sum = 0;
    for (int ii = 0; ii < 16; ++ii) sum += y.qs[g * 16 + ii];
    y.bsums[g] = static_cast<int16_t>(sum);
  }
  y.d = 1.0f / iscale;
}

__device__ inline void QuantQ8KSBlock(vt::cpu::BlockQ8_K& y, const void* __restrict__ a,
                                      ActDT adt, int64_t elem0) {
  switch (adt) {
    case ActDT::kF32: QuantQ8KSBlockBody<ActDT::kF32>(y, a, elem0); break;
    case ActDT::kF16: QuantQ8KSBlockBody<ActDT::kF16>(y, a, elem0); break;
    default: QuantQ8KSBlockBody<ActDT::kBF16>(y, a, elem0); break;
  }
}

}  // namespace vt::rocm

#endif  // VLLM_CPP_SRC_VT_ROCM_ROCM_ACT_QUANT_H_
