#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace vt::rocm {

#if defined(__gfx1100__)
// RDNA3 ISA section 7.9: WMMA executes DOT instructions internally. Eight
// consecutive BF16 DOT2 operations reproduce one 16-element WMMA reduction.
// Ordinary FP32 multiply/add expressions have different intermediate rounding.
__device__ __forceinline__ float AttentionDot2(unsigned a, unsigned b, float acc) {
  using Pair = short __attribute__((ext_vector_type(2)));
  Pair ap, bp;
  __builtin_memcpy(&ap, &a, sizeof(ap));
  __builtin_memcpy(&bp, &b, sizeof(bp));
  return __builtin_amdgcn_fdot2_f32_bf16(ap, bp, acc, false);
}

__device__ __forceinline__ unsigned AttentionBf16Pair(float lo, float hi) {
  const __hip_bfloat16 values[2] = {__float2bfloat16(lo), __float2bfloat16(hi)};
  unsigned packed;
  __builtin_memcpy(&packed, values, sizeof(packed));
  return packed;
}
#endif

}  // namespace vt::rocm
