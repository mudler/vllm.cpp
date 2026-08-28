// ROCm grouped quant expert GEMM (BACKEND-ROCM; issue #41, the MoE-path
// blocker). Port of src/vt/cuda/cuda_quant_dot.cu grouped path:
//   QuantizeQ8KKernel  (Q8_K activation for K-quant formats)
//   QuantizeQ8_0Kernel (Q8_0 activation for the Q8_0 format)
//   QuantDotGemmGroupedKernel (:746) + QuantDotGemmGroupedQ8_0Kernel (:1404)
// Dot superblocks DotQ8_0 / DotQ4K / DotQ6K ported 1:1. Bit-exact integer
// cores (__dp4a); float scale products reassociate across lanes as the donor's.
//
//   out[P,N] (f32/bf16) = per (p,j): sum_sb dot(Q8 act[p], keepquant w[e,j]),
//   e = expert_ids[p]; activation quantized once (broadcast when 1 row).
//
// Covers the formats the target GDN-MoE GGUFs use (Q4_K / Q6_K / Q8_0). The
// IQ2/IQ3/Q2_K/Q3_K/Q5_K superblocks port identically against this skeleton.

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"

// Block layouts — the single source of truth (ggml-common.h mirrors).
#include "vt/cpu/cpu_quant_blocks.h"

namespace vt::rocm {
namespace {

using vt::cpu::BlockQ8_0;
using vt::cpu::BlockQ8_K;
using vt::cpu::BlockQ4_K;
using vt::cpu::BlockQ5_K;
using vt::cpu::BlockQ6_K;
using vt::cpu::kQK8_0;
using vt::cpu::kQK_K;

enum class ActDT : int { kF32 = 0, kF16 = 1, kBF16 = 2 };
inline ActDT ActDtOf(DType dt) {
  return dt == DType::kF32 ? ActDT::kF32 : dt == DType::kF16 ? ActDT::kF16 : ActDT::kBF16;
}

// ---- device numeric helpers (bit-exact ports from cuda_quant_dot.cu) ----
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
__device__ __forceinline__ int GetIntB2(const int8_t* qs, int i32) {
  const uint16_t* x16 = reinterpret_cast<const uint16_t*>(qs);
  return static_cast<int>(x16[2 * i32 + 0]) | (static_cast<int>(x16[2 * i32 + 1]) << 16);
}

// Signed 8-bit x4 dot-product-accumulate, bit-identical to __dp4a (integer
// math is exact either way). The HW dot instruction (v_dot4_i32_i8 /
// __ockl_sdot4) is a perf lever, not a correctness requirement.
__device__ __forceinline__ int Dp4a(int a, int b, int acc) {
  const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
  const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
  return acc + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}

// ---- activation quantizers ----
// Q8_0 (thread-per-32-block): cuda_quant_dot.cu:869.
__global__ void QuantizeQ8_0K(BlockQ8_0* __restrict__ scratch, const void* __restrict__ a,
                              ActDT adt, int64_t a_rs, int64_t m, int64_t nb) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * nb) return;
  const int64_t i = t / nb;
  const int64_t b = t % nb;
  const int64_t elem0 = i * a_rs + b * kQK8_0;
  float amax = 0.0f;
  for (int j = 0; j < kQK8_0; ++j) {
    const float av = fabsf(DLoadAct(a, adt, elem0 + j));
    amax = amax > av ? amax : av;
  }
  BlockQ8_0& y = scratch[t];
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  y.d = DF32ToF16(d);
  for (int j = 0; j < kQK8_0; ++j) {
    const float x0 = DLoadAct(a, adt, elem0 + j) * id;
    y.qs[j] = static_cast<int8_t>(roundf(x0));
  }
}

// Q8_K (thread-per-256-superblock): cuda_quant_dot.cu QuantizeQ8KKernel.
__global__ void QuantizeQ8KK(BlockQ8_K* __restrict__ scratch, const void* __restrict__ a,
                             ActDT adt, int64_t a_rs, int64_t m, int64_t nsb) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * nsb) return;
  const int64_t i = t / nsb;
  const int64_t sb = t % nsb;
  const int64_t elem0 = i * a_rs + sb * kQK_K;
  float mx = 0.0f, amax = 0.0f;
  for (int j = 0; j < kQK_K; ++j) {
    const float ax = fabsf(DLoadAct(a, adt, elem0 + j));
    if (ax > amax) { amax = ax; mx = DLoadAct(a, adt, elem0 + j); }
  }
  BlockQ8_K& y = scratch[t];
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

// ---- dot superblocks (1:1 ports) ----
// Q8_0 x Q8_0: cuda_quant_dot.cu QuantDotGemmQ8_0 — dp4a int core.
__device__ inline float DotQ8_0(const BlockQ8_0* wb, const BlockQ8_0* ab) {
  int sumi = 0;
#pragma unroll
  for (int k = 0; k < kQK8_0 / 4; ++k)
    sumi = Dp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
  return sumi * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
}

// Q4_K x Q8_K: cuda_quant_dot.cu DotQ4K. dp4a-vectorized, one scale per 32.
__device__ inline float DotQ4K(const BlockQ4_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
  const uint8_t* q4 = xb->qs;
  const int8_t* q8 = yb->qs;
  uint32_t utmp[4];
  memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
  int sumi = 0;
  for (int j = 0; j < kQK_K / 16; ++j) sumi += yb->bsums[j] * mins[j / 2];
  int isum = 0;
  for (int sb = 0; sb < kQK_K / 32; ++sb) {
    const int scale = scales[sb];
    const uint8_t* q4b = q4 + (sb / 2) * 32;
    const int8_t* q8b = q8 + sb * 32;
    const int shift = (sb & 1) ? 4 : 0;
    int sub = 0;
    for (int l = 0; l < 32; l += 4) {
      const int v = (*reinterpret_cast<const int*>(q4b + l) >> shift) & 0x0F0F0F0F;
      sub = Dp4a(v, *reinterpret_cast<const int*>(q8b + l), sub);
    }
    isum += scale * sub;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  return d * isum - dmin * sumi;
}

// Q5_K x Q8_K: cuda_quant_dot.cu DotQ5K. Q4_K nibble + a high bit from qh.
__device__ inline float DotQ5K(const BlockQ5_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
  const uint8_t* q4 = xb->qs;
  const uint8_t* hm = xb->qh;
  const int8_t* q8 = yb->qs;
  uint32_t utmp[4];
  memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
  int sumi = 0;
  for (int j = 0; j < kQK_K / 16; ++j) sumi += yb->bsums[j] * mins[j / 2];
  int isum = 0;
  for (int sb = 0; sb < kQK_K / 32; ++sb) {
    const int scale = scales[sb];
    const uint8_t* q4b = q4 + (sb / 2) * 32;
    const int8_t* q8b = q8 + sb * 32;
    const int shift = (sb & 1) ? 4 : 0;
    int sub = 0;
    for (int l = 0; l < 32; l += 4) {
      const int lo = (*reinterpret_cast<const int*>(q4b + l) >> shift) & 0x0F0F0F0F;
      const int hi = ((*reinterpret_cast<const int*>(hm + l) >> sb) & 0x01010101) << 4;
      sub = Dp4a(lo | hi, *reinterpret_cast<const int*>(q8b + l), sub);
    }
    isum += scale * sub;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  return d * isum - dmin * sumi;
}

// Q6_K x Q8_K: cuda_quant_dot.cu DotQ6K. Rebuild the 6-bit quants then scalar-MAC.
__device__ inline float DotQ6K(const BlockQ6_K* xb, const BlockQ8_K* yb) {
  const uint8_t* q4 = xb->ql;
  const uint8_t* qh = xb->qh;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  for (int j = 0; j < kQK_K; j += 128) {
    for (int l = 0; l < 32; ++l) {
      a[l + 0] = static_cast<int8_t>(static_cast<int8_t>((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
      a[l + 32] = static_cast<int8_t>(static_cast<int8_t>((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
      a[l + 64] = static_cast<int8_t>(static_cast<int8_t>((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
      a[l + 96] = static_cast<int8_t>(static_cast<int8_t>((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
    }
    a += 128; q4 += 64; qh += 32;
  }
  a = aux8;
  const int8_t* q8p = q8;
  int is = 0;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    const int scale = xb->scales[is++];
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
    q8p += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum;
}

// Q6_K integer core over a SUB-BLOCK RANGE, for the cooperative decode arm
// (`KQuantGemmKCoopQ6K`, issue #1910). A K-quant superblock is kQK_K = 256
// weights in kSubBlocks = 8 groups of 32; this returns the dp4a accumulator for
// half-open group range [s0, s1) so several warps can each take a share of one
// output row's superblock.
//
// It hands back the raw INTEGER accumulator, never a float. That is what makes
// the cooperative arm safe: integer addition is exact and associative, so the
// partial `isum` a shared-memory reduction reassembles across warps equals the
// single-warp arm's accumulator bit for bit, `d * isum` still runs once per
// superblock, and no reassociation of the f32 scale product is introduced.
// Splitting the FLOATS instead would have cost token-exactness for what is only
// a launch-shape change.
//
// This is a SECOND body rather than a range parameter on `DotQ6K`, because
// making `DotQ6K` call it at [0, 8) measured 642.59 us/call against 371.90 for
// the donor loop on the n=4096 k=12288 shape (1.73x): the donor's two fixed
// 128-wide chunks unroll into registers where a loop over a runtime group range
// does not. The two bodies are held equal by the byte-equality case in
// tests/vt/test_backend_cross_device.cpp, which runs the same activation row
// through the cooperative arm (m == 1) and the single-warp arm (m == 2) and
// demands identical bytes — so a drift between them reds a gate rather than
// quietly changing a token.
//
// Index map, read off the donor's 128-wide chunks: group g covers weights
// [32*g, 32*g+32). Chunk c = g / 4 supplies ql[64*c ..] and qh[32*c ..], and
// r = g % 4 selects the donor's four lines: r picks the ql half (r odd => the
// +32 half) and the nibble (r >= 2 => the high one), and shifts qh by 2*r.
constexpr int kSubBlocks = kQK_K / 32;

__device__ inline int DotQ6KIsumRange(const BlockQ6_K* xb, const BlockQ8_K* yb, int s0, int s1) {
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int g = s0; g < s1; ++g) {
    const uint8_t* q4 = xb->ql + (g / 4) * 64;
    const uint8_t* qh = xb->qh + (g / 4) * 32;
    const int r = g & 3;
    const uint8_t* q4b = q4 + ((r & 1) ? 32 : 0);
    const int nib_shift = (r >= 2) ? 4 : 0;
    const int qh_shift = 2 * r;
    int8_t aux8[32];
    for (int l = 0; l < 32; ++l)
      aux8[l] = static_cast<int8_t>(
          static_cast<int8_t>(((q4b[l] >> nib_shift) & 0xF) | (((qh[l] >> qh_shift) & 3) << 4)) -
          32);
    // Two 16-wide scale groups per 32-wide sub-block, the donor's `is` order.
    const int8_t* q8p = yb->qs + g * 32;
    const int8_t* a = aux8;
    for (int t = 0; t < 2; ++t) {
      const int scale = xb->scales[g * 2 + t];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
      q8p += 8; a += 8;
      for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
      q8p += 8; a += 8;
    }
  }
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return isum;
}

// ---- grouped kernels ----
template <typename OutT>
__global__ void GroupedQ8_0K(OutT* __restrict__ out, const uint8_t* __restrict__ weight,
                             const BlockQ8_0* __restrict__ act,
                             const int32_t* __restrict__ expert_ids, int64_t P, int64_t n,
                             int64_t nb, size_t w_row_bytes, bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_0* a_row = act + (bcast ? 0 : p) * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                          sizeof(BlockQ8_0));
    partial += DotQ8_0(wb, a_row + b);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffULL, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[p * n + j] = partial;
    else out[p * n + j] = DF32ToBF16(partial);
  }
}

// K-quant grouped kernel body (shared by Q4_K/Q5_K/Q6_K instantiations).
// Fmt: 0=Q4_K, 1=Q5_K, 2=Q6_K.
template <typename OutT, int Fmt>
__global__ void GroupedKQ8K(OutT* __restrict__ out, const uint8_t* __restrict__ weight,
                            const BlockQ8_K* __restrict__ act,
                            const int32_t* __restrict__ expert_ids, int64_t P, int64_t n,
                            int64_t nsb, size_t w_row_bytes, size_t w_block_bytes, bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_K* a_row = act + (bcast ? 0 : p) * nsb;
  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    const BlockQ8_K* a_sb = a_row + sb;
    if constexpr (Fmt == 2) partial += DotQ6K(static_cast<const BlockQ6_K*>(w_sb), a_sb);
    else if constexpr (Fmt == 1) partial += DotQ5K(static_cast<const BlockQ5_K*>(w_sb), a_sb);
    else partial += DotQ4K(static_cast<const BlockQ4_K*>(w_sb), a_sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffULL, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[p * n + j] = partial;
    else out[p * n + j] = DF32ToBF16(partial);
  }
}


// --- non-grouped keep-quant GEMM (cuda_quant_dot.cu QuantDotGemmKernel :706) ---
// warp per (i,j); no expert indirection (w_row = weight + j*w_row_bytes).
template <typename OutT, int Fmt>
__global__ void KQuantGemmK(OutT* __restrict__ out, const uint8_t* __restrict__ weight,
                            const BlockQ8_K* __restrict__ act, int64_t m, int64_t n,
                            int64_t nsb, size_t w_row_bytes, size_t w_block_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_K* a_row = act + i * nsb;
  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    if constexpr (Fmt == 2) partial += DotQ6K(static_cast<const BlockQ6_K*>(w_sb), a_row + sb);
    else if constexpr (Fmt == 1) partial += DotQ5K(static_cast<const BlockQ5_K*>(w_sb), a_row + sb);
    else partial += DotQ4K(static_cast<const BlockQ4_K*>(w_sb), a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffULL, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

// --- cooperative decode arm of KQuantGemmK, Q6_K only (issue #1910) ---
//
// `nsb` = K / 256, so a 4096-wide projection gives 16 superblocks and the loop
// above hands lanes 16..31 nothing while the 5-round reduction still runs. On
// the issue's model that is 195 of 259 decode dispatches.
//
// llama.cpp answers this on RDNA4 by putting NWarps warps on ONE output row and
// splitting each K block across them: `calc_nwarps` returns 8 for
// Q4_K/Q5_K/Q6_K at ncols_dst == 1 (ggml/src/ggml-cuda/mmvq.cu:385-408) — pin
// b10451, 10bf611e533d81f739128304991c5e133c6aebd8. Upstream's other lever,
// `rows_per_cuda_block` row-packing, is off for every RDNA table (:460-478,
// :898), but per the spec's `## Upstream anchor` correction that exclusion was
// never benchmarked on RDNA hardware, so it is not evidence against packing —
// it is only the reason this arm, which does have an RDNA4 benchmark behind
// it, is the one ported here.
//
// Warp w takes sub-block range [w*8/NWarps, (w+1)*8/NWarps) of whichever
// superblocks its lane holds, and each iteration reduces the INTEGER
// accumulators across warps through shared memory before warp 0 applies the
// superblock's f32 scale. That is why this arm is bit-identical to
// `KQuantGemmK` rather than merely close — see the `DotQ6KIsumRange` note.
//
// Q6_K ONLY, and that is a measured restriction rather than a scoping choice.
// Splitting a superblock across warps re-reads its header on every warp and
// breaks the contiguity of the quant read, which for a bandwidth-bound GEMV
// costs more than the extra warps buy. Q4_K and Q5_K lose 1.5x to 1.8x at every
// nsb <= 32 shape and every width in {2, 4, 8}; only Q6_K, whose per-lane
// reconstruction is heavy enough to hide the extra traffic, comes out ahead.
// The sweep is in .agents/specs/rocm-kquant-nwarps-decode.md `## Outcome`.
template <typename OutT, int NWarps>
__global__ void KQuantGemmKCoopQ6K(OutT* __restrict__ out, const uint8_t* __restrict__ weight,
                                   const BlockQ8_K* __restrict__ act, int64_t m, int64_t n,
                                   int64_t nsb, size_t w_row_bytes, size_t w_block_bytes) {
  static_assert(NWarps > 1 && NWarps <= kSubBlocks && kSubBlocks % NWarps == 0,
                "NWarps must divide the superblock's sub-blocks evenly");
  constexpr int kPerWarp = kSubBlocks / NWarps;
  __shared__ int sh_isum[(NWarps - 1) * 32];

  // One block per output element: `elem` is block-uniform, so every
  // __syncthreads() below is reached by the whole block.
  const int64_t elem = static_cast<int64_t>(blockIdx.x);
  if (elem >= m * n) return;
  const int64_t i = elem / n;
  const int64_t j = elem % n;
  const int lane = threadIdx.x;
  const int wid = threadIdx.y;
  const int s0 = wid * kPerWarp, s1 = s0 + kPerWarp;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_K* a_row = act + i * nsb;

  // Trip count is block-uniform for the same reason: it depends only on nsb.
  const int64_t iters = (nsb + 31) / 32;
  float partial = 0.0f;
  for (int64_t it = 0; it < iters; ++it) {
    // Clamped so the address arithmetic stays inside the row on the tail
    // iteration; every read below is still gated on the unclamped `sb < nsb`.
    const int64_t sb = it * 32 + lane;
    const int64_t sb_safe = sb < nsb ? sb : 0;
    const BlockQ6_K* xb =
        reinterpret_cast<const BlockQ6_K*>(w_row + static_cast<size_t>(sb_safe) * w_block_bytes);
    const BlockQ8_K* a_sb = a_row + sb_safe;
    int isum = sb < nsb ? DotQ6KIsumRange(xb, a_sb, s0, s1) : 0;
    if (wid > 0) sh_isum[(wid - 1) * 32 + lane] = isum;
    __syncthreads();
    if (wid == 0) {
#pragma unroll
      for (int w = 0; w < NWarps - 1; ++w) isum += sh_isum[w * 32 + lane];
      // One f32 scale product per superblock, exactly as the single-warp arm
      // does it, applied to an integer sum that is exact however it was split.
      if (sb < nsb) partial += (DF16ToF32(xb->d) * a_sb->d) * isum;
    }
    // Warps 1.. must not overwrite the shared partials before warp 0 has read
    // them, so a second iteration would wait here rather than race.
    //
    // THAT SECOND ITERATION IS UNREACHABLE TODAY, and the fact is written
    // here rather than left to be re-derived from the dispatch guard:
    // KQuantDecodeCoopWarps returns a cooperative width only for nsb <= 32,
    // so `iters` is always exactly 1 and the loop never comes back round.
    // Deleting this barrier therefore leaves the suite green, which says the
    // race is untested rather than that the barrier is redundant. It becomes
    // load-bearing the moment that nsb bound moves.
    __syncthreads();
  }
  if (wid != 0) return;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffULL, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

template <typename OutT>
__global__ void Q8_0GemmK(OutT* __restrict__ out, const uint8_t* __restrict__ weight,
                          const BlockQ8_0* __restrict__ act, int64_t m, int64_t n,
                          int64_t nb) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const uint8_t* w_row = weight + static_cast<size_t>(j * nb) * sizeof(BlockQ8_0);
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  for (int64_t bb = lane; bb < nb; bb += 32) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(bb) *
                                                                          sizeof(BlockQ8_0));
    partial += DotQ8_0(wb, a_row + bb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffULL, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm grouped_gemm: ") + what + ": " +
                             hipGetErrorString(err));
}

// Grow-only, per-stream activation-quant scratch pool — the mirror of the
// donor's EnsureScratch + RetireGraphScratch discipline
// (src/vt/cuda/cuda_quant_dot.cu:1507, src/vt/cuda/graph_safe_scratch.h):
// hipMalloc/hipFree/hipStreamSynchronize per call are ILLEGAL under hipGraph
// stream capture (the #473/#332 decode-graph lane), and the old free-after-
// launch also needed the sync that serialized every call. The pool allocates
// stream-ordered (hipMallocAsync), never frees during the process (a captured
// graph may have baked the pointer), and needs NO synchronization: reuse is
// stream-ordered and retirement keeps every baked pointer valid. Bounded: the
// buffer grows O(log(max/min)) times over a process.
struct StreamScratch {
  void* buf = nullptr;
  size_t bytes = 0;
};

StreamScratch& ScratchFor(hipStream_t s) {
  static std::mutex mu;
  static std::unordered_map<hipStream_t, StreamScratch> pools;
  std::lock_guard<std::mutex> lk(mu);
  return pools[s];
}

void* EnsureQuantScratch(size_t need, hipStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.bytes) {
    // Retire, never free (see the header note above).
    Check(hipMallocAsync(&sc.buf, need, s), "quant scratch grow");
    sc.bytes = need;
  }
  return sc.buf;
}

// How many dense K-quant dispatches took the cooperative arm. Host-side and
// relaxed: nothing reads it on the decode path, and the ROCm decode gate reads
// it to prove the call site below is REACHED rather than merely compiled
// (AGENTS.md "Nothing lands dead" — the arm is bit-identical to the single-warp
// one by construction, so no output comparison can witness the dispatch).
std::atomic<uint64_t> g_kq_coop_dispatches{0};

}  // namespace

// Warps that cooperate on one output row of the dense K-quant GEMM. 1 selects
// the single-warp `KQuantGemmK` the path has always had.
//
// Every value here is MEASURED on this tree's `Dot*K` bodies rather than copied
// from llama.cpp's RDNA4 table, and the measurement disagreed with that table.
// Upstream's `vec_dot_q*_K_q8_1` already splits a superblock 16 ways where ours
// does all 256 weights in one lane, so the traffic a warp split adds is a
// different quantity here: re-reading each superblock's header per warp and
// breaking the quant read's contiguity costs a bandwidth-bound GEMV more than
// the extra resident warps buy. Q4_K and Q5_K therefore keep the single-warp
// arm at every width tried. `## Outcome` in the spec has the full sweep.
int KQuantDecodeCoopWarps(DType wdt, int64_t m, int64_t nsb) {
  // Decode only. The spec's `## Scope` keeps prefill on the arm it already has:
  // one block per output element costs m*n blocks, and m > 1 is not the regime
  // where a row's lanes run out of superblocks to chew on.
  if (m != 1) return 1;
  // nsb > 32 already fills all 32 lanes of a single warp (the k=12288 shapes in
  // the issue's table), and `## Gate` requires those to keep the kernel and the
  // launch config they had.
  if (nsb > 32) return 1;
  // Q6_K at 8: the one arm the sweep put ahead of the single-warp kernel.
  return wdt == DType::kQ6_K ? 8 : 1;
}

uint64_t KQuantCoopDispatchCount() {
  return g_kq_coop_dispatches.load(std::memory_order_relaxed);
}

void MatmulBTQuantKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  EnsureQueueDevice(q);
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  if (m == 0 || n == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const uint8_t* w = static_cast<const uint8_t*>(b.data);
  if (b.dtype == DType::kQ8_0) {
    if (k % kQK8_0 != 0) throw std::runtime_error("vt rocm: matmul_bt_quant Q8_0: K%32!=0");
    const int64_t nb = k / kQK8_0;
    BlockQ8_0* qact = static_cast<BlockQ8_0*>(EnsureQuantScratch(
        static_cast<size_t>(m) * nb * sizeof(BlockQ8_0), s));
    QuantizeQ8_0K<<<static_cast<unsigned>((m * nb + 127) / 128), 128, 0, s>>>(
        qact, a.data, ActDtOf(a.dtype), a.stride[0], m, nb);
    Check(hipGetLastError(), "q8_0 quant");
    const int64_t grid = (m * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (out.dtype == DType::kF32)
      Q8_0GemmK<float><<<static_cast<unsigned>(grid), block, 0, s>>>(static_cast<float*>(out.data), w, qact, m, n, nb);
    else
      Q8_0GemmK<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(static_cast<uint16_t*>(out.data), w, qact, m, n, nb);
    Check(hipGetLastError(), "q8_0 gemm");
    return;
  }
  if (b.dtype == DType::kQ4_K || b.dtype == DType::kQ5_K || b.dtype == DType::kQ6_K) {
    if (k % kQK_K != 0) throw std::runtime_error("vt rocm: matmul_bt_quant K-quant: K%256!=0");
    const int64_t nsb = k / kQK_K;
    const size_t w_block_bytes = b.dtype == DType::kQ6_K ? sizeof(BlockQ6_K)
                                : b.dtype == DType::kQ5_K ? sizeof(BlockQ5_K)
                                                          : sizeof(BlockQ4_K);
    const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;
    BlockQ8_K* qact = static_cast<BlockQ8_K*>(EnsureQuantScratch(
        static_cast<size_t>(m) * nsb * sizeof(BlockQ8_K), s));
    QuantizeQ8KK<<<static_cast<unsigned>((m * nsb + 127) / 128), 128, 0, s>>>(
        qact, a.data, ActDtOf(a.dtype), a.stride[0], m, nsb);
    Check(hipGetLastError(), "q8_K quant");
    const int64_t grid = (m * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int fmt = b.dtype == DType::kQ6_K ? 2 : b.dtype == DType::kQ5_K ? 1 : 0;
    const int coop = KQuantDecodeCoopWarps(b.dtype, m, nsb);
    auto launch = [&](auto ot) {
      using OutT = decltype(ot);
      auto* o = static_cast<OutT*>(out.data);
      if (coop == 8 && fmt == 2) {
        // One block per output element, eight warps deep. Only the measured
        // width is instantiated; anything else falls through to the single-warp
        // arm rather than silently launching a shape nothing was measured on.
        g_kq_coop_dispatches.fetch_add(1, std::memory_order_relaxed);
        KQuantGemmKCoopQ6K<OutT, 8><<<static_cast<unsigned>(m * n), dim3(32, 8), 0, s>>>(
            o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes);
        return;
      }
      if (fmt == 2) KQuantGemmK<OutT, 2><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes);
      else if (fmt == 1) KQuantGemmK<OutT, 1><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes);
      else KQuantGemmK<OutT, 0><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes);
    };
    if (out.dtype == DType::kF32) launch(float{}); else launch(uint16_t{});
    Check(hipGetLastError(), "K-quant gemm");
    return;
  }
  throw std::runtime_error("vt rocm: matmul_bt_quant: unsupported weight dtype (ported: Q8_0/Q4_K/Q5_K/Q6_K; owed: Q4_0/Q2_K/Q3_K/IQ2_XXS/IQ3_XXS/IQ2_S/MXFP4 -- the loader pre-filters to the ported set, so reaching here is a bug)");
}


// kMatmulBTQuantGrouped for ROCm: Q8_0 / Q4_K / Q6_K natively (the formats the
// target GDN-MoE GGUFs carry); anything else throws loudly (never a silent
// CPU-pointer deref on a discrete card).
void MatmulBTQuantGroupedKernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                    const Tensor& weight, const Tensor& expert_ids) {
  EnsureQueueDevice(q);
  const int64_t P = out.shape[0], n = out.shape[1], k = act.shape[1];
  if (P == 0 || n == 0) return;
  const int64_t Pa = act.shape[0];
  const bool bcast = (Pa == 1 && P > 1);
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const uint8_t* w = static_cast<const uint8_t*>(weight.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);

  if (weight.dtype == DType::kQ8_0) {
    if (k % kQK8_0 != 0)
      throw std::runtime_error("vt rocm: matmul_bt_quant_grouped Q8_0: K must be a multiple of 32");
    const int64_t nb = k / kQK8_0;
    const size_t w_row_bytes = static_cast<size_t>(nb) * sizeof(BlockQ8_0);
    BlockQ8_0* qact = static_cast<BlockQ8_0*>(EnsureQuantScratch(
        static_cast<size_t>(Pa) * nb * sizeof(BlockQ8_0), s));
    constexpr int kQBlock = 128;
    QuantizeQ8_0K<<<static_cast<unsigned>((Pa * nb + kQBlock - 1) / kQBlock), kQBlock, 0, s>>>(
        qact, act.data, ActDtOf(act.dtype), act.stride[0], Pa, nb);
    Check(hipGetLastError(), "q8_0 quant");
    const int64_t grid = (P * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (out.dtype == DType::kF32)
      GroupedQ8_0K<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
          static_cast<float*>(out.data), w, qact, eids, P, n, nb, w_row_bytes, bcast);
    else
      GroupedQ8_0K<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
          static_cast<uint16_t*>(out.data), w, qact, eids, P, n, nb, w_row_bytes, bcast);
    Check(hipGetLastError(), "q8_0 grouped");
    return;
  }

  if (weight.dtype == DType::kQ4_K || weight.dtype == DType::kQ5_K || weight.dtype == DType::kQ6_K) {
    if (k % kQK_K != 0)
      throw std::runtime_error("vt rocm: matmul_bt_quant_grouped K-quant: K must be a multiple of 256");
    const int64_t nsb = k / kQK_K;
    const size_t w_block_bytes = weight.dtype == DType::kQ4_K ? sizeof(BlockQ4_K)
                                : weight.dtype == DType::kQ5_K ? sizeof(BlockQ5_K)
                                                               : sizeof(BlockQ6_K);
    const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;
    BlockQ8_K* qact = static_cast<BlockQ8_K*>(EnsureQuantScratch(
        static_cast<size_t>(Pa) * nsb * sizeof(BlockQ8_K), s));
    QuantizeQ8KK<<<static_cast<unsigned>((Pa * nsb + 127) / 128), 128, 0, s>>>(
        qact, act.data, ActDtOf(act.dtype), act.stride[0], Pa, nsb);
    Check(hipGetLastError(), "q8_K quant");
    const int64_t grid = (P * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int fmt = weight.dtype == DType::kQ6_K ? 2 : weight.dtype == DType::kQ5_K ? 1 : 0;
    auto launch = [&](auto ot) {
      using OutT = decltype(ot);
      auto* o = static_cast<OutT*>(out.data);
      if (fmt == 2) GroupedKQ8K<OutT, 2><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast);
      else if (fmt == 1) GroupedKQ8K<OutT, 1><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast);
      else GroupedKQ8K<OutT, 0><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast);
    };
    if (out.dtype == DType::kF32) launch(float{}); else launch(uint16_t{});
    Check(hipGetLastError(), "K-quant grouped");
    return;
  }

  throw std::runtime_error(
      "vt rocm: matmul_bt_quant_grouped: unsupported weight dtype (ported: Q8_0/Q4_K/Q5_K/Q6_K; "
      "owed: Q4_0/Q2_K/Q3_K/IQ2_XXS/IQ3_XXS/IQ2_S/MXFP4 -- the loader pre-filters, so reaching here is a bug)");
}

}  // namespace vt::rocm
