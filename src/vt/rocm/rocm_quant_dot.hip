// ROCm keep-quant GGUF k-quant GEMM (KERNEL-QUANT-CIQ-GEMM-ROCM W1) — the
// kROCM provider for `OpId::kMatmulBTQuant` and `OpId::kMatmulBTQuantGrouped`.
//
// Port of src/vt/cuda/cuda_quant_dot.cu (the kCUDA provider), which is itself
// a port of the CPU oracle:
//   src/vt/cpu/cpu_quant_gemm.cpp        MatmulBTQuantKernel   (the GEMM wiring)
//   src/vt/cpu/cpu_quant_dot.cpp         VecDot{Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,
//                                        IQ2_XXS,IQ3_XXS,IQ2_S,IQ1_S,IQ1_XXXS}Q8_K
//   src/vt/cpu/cpu_quant_act.cpp         QuantizeRowQ8_K       (the activation quant)
// The device numeric helpers below are the SAME bit-exact ports of
// src/vt/dtype.cpp + cpu_quant_act.cpp, so the Q8_K activation bytes — and
// therefore the whole INTEGER dot — are IDENTICAL to the CPU reference. Only
// the per-super-block float scale sum is reassociated (warp reduction vs the
// CPU's sequential add), so the gate is the CUDA sibling's gate: INTEGER core
// bit-exact, final scale within the NMSE band test_ops_quant_dot uses.
//
//   * Scratch uses hipMallocAsync (present in ROCm 7.14) with the same retire
//     -never-free discipline as the CUDA side (graph_safe_scratch.h).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/cpu/cpu_quant_blocks.h"         // vt::cpu::Block* struct mirror
                                             // (single source; plain C++)
#include "vt/cuda/cuda_quant_iq_tables.cuh"  // d_iq2xxs_grid / d_iq3xxs_grid /
                                             // d_iq2s_grid / d_iq1s_grid /
                                             // d_iq1xxxs_grid / d_ksigns /
                                             // d_kmask (single source; pure
                                             // __device__ syntax, no CUDA)
#include "vt/cuda/graph_safe_scratch.h"      // RetireGraphScratch (portable)
#include "vt/ops.h"
#include "vt/quant.h"

// Forward declarations: the optimized K-quant GEMM kernels in
// rocm_grouped_gemm.hip (renamed *Gdn). F1 delegates Q4_K/Q5_K/Q6_K to these
// to preserve the baseline-optimized path; the new kernels in this file
// handle IQ types and Q2_K/Q3_K that the old code did not support.
namespace vt::rocm {
void MatmulBTQuantKernelRocmGdn(Queue& q, Tensor& out, const Tensor& a,
                                const Tensor& b);
void MatmulBTQuantGroupedKernelRocmGdn(Queue& q, Tensor& out, const Tensor& act,
                                       const Tensor& weight,
                                       const Tensor& expert_ids);
}

namespace vt::rocm {
namespace {

// dtype.cpp F32ToF16 — round-to-nearest-even, subnormals, inf/nan. Used only
// for the Q8_0 activation scale (the CPU Q8_0 vec_dot's f16 round-trip).
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

// Load one activation element (dtype-decoded, exactly like cpu LoadActF32).

using vt::cpu::BlockIQ1_S;
using vt::cpu::BlockIQ1_XXXS;
using vt::cpu::BlockIQ2_S;
using vt::cpu::BlockIQ2_XXS;
using vt::cpu::BlockIQ3_XXS;
using vt::cpu::BlockQ2_K;
using vt::cpu::BlockQ3_K;
using vt::cpu::BlockQ4_K;
using vt::cpu::BlockQ5_K;
using vt::cpu::BlockQ6_K;
using vt::cpu::BlockQ8_K;
using vt::cpu::BlockQ8_0;
using vt::cpu::kQK_K;
using vt::cpu::kQK8_0;

void CheckHip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: matmul_bt_quant: ") + what +
                             ": " + hipGetErrorString(err));
  }
}

// --- device numeric helpers — bit-exact ports of src/vt/dtype.cpp -------------
__device__ inline float DF16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0x1F) return __int_as_float(sign | 0x7F800000 | (mant << 13));
  if (exp == 0) {
    if (mant == 0) return __int_as_float(sign);
    int shift = 0;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      ++shift;
    }
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
  if ((u & 0x7F800000) == 0x7F800000 && (u & 0x7FFFFF)) {
    return static_cast<uint16_t>((u >> 16) | 0x0040);
  }
  uint32_t rounding = 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

// cpu_quant_act.cpp NearestInt (ggml-quants.c:563) — magic-constant round-to-even.
__device__ inline int DNearestInt(float fval) {
  float val = fval + 12582912.0f;
  int i = __float_as_int(val);
  return (i & 0x007fffff) - 0x00400000;
}

enum class ActDT : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

__device__ inline float DLoadAct(const void* base, ActDT dt, int64_t idx) {
  switch (dt) {
    case ActDT::kF32: return static_cast<const float*>(base)[idx];
    case ActDT::kF16: return DF16ToF32(static_cast<const uint16_t*>(base)[idx]);
    default: return DBF16ToF32(static_cast<const uint16_t*>(base)[idx]);
  }
}

// ---------------------------------------------------------------------------
// GPU activation quantizer — ds4-parity grid: ONE BLOCK per (super-block, row),
// 256 threads (one thread per element). Bit-exact port of QuantizeRowQ8_K with
// the argmax tie broken by LOWEST original index (== the sequential first-
// occurrence scan). Byte-identical to both CUDA quantizers by construction.
// ---------------------------------------------------------------------------
__global__ void QuantizeQ8KKernel(BlockQ8_K* __restrict__ scratch,
                                  const void* __restrict__ a, ActDT adt,
                                  int64_t a_rs, int64_t m, int64_t nsb) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);  // super-block within row
  const int64_t i = static_cast<int64_t>(blockIdx.y);  // activation row
  if (b >= nsb || i >= m) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t elem0 = i * a_rs + b * kQK_K;
  const float v = DLoadAct(a, adt, elem0 + tid);

  __shared__ float sabs[kQK_K];
  __shared__ float sval[kQK_K];
  __shared__ int sidx[kQK_K];
  sabs[tid] = fabsf(v);
  sval[tid] = v;
  sidx[tid] = tid;
  __syncthreads();
#pragma unroll
  for (int stride = kQK_K >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float oa = sabs[tid + stride];
      if (oa > sabs[tid] || (oa == sabs[tid] && sidx[tid + stride] < sidx[tid])) {
        sabs[tid] = oa;
        sval[tid] = sval[tid + stride];
        sidx[tid] = sidx[tid + stride];
      }
    }
    __syncthreads();
  }
  const float mx = sval[0];
  const float amax = sabs[0];

  BlockQ8_K& y = scratch[i * nsb + b];
  if (amax == 0.0f) {
    if (tid == 0) y.d = 0.0f;
    y.qs[tid] = 0;
    if (tid < kQK_K / 16) y.bsums[tid] = 0;
    return;
  }
  const float iscale = -127.0f / mx;
  const int qv = DNearestInt(iscale * v);
  y.qs[tid] = static_cast<int8_t>(qv < 127 ? qv : 127);
  __syncthreads();
  if (tid < kQK_K / 16) {
    int sum = 0;
    for (int ii = 0; ii < 16; ++ii) sum += y.qs[tid * 16 + ii];
    y.bsums[tid] = static_cast<int16_t>(sum);
  }
  if (tid == 0) y.d = 1.0f / iscale;
}

// ---------------------------------------------------------------------------
// Per-super-block integer dots. PORTABLE SCALAR forms of the CPU reference
// bodies (cpu_quant_dot.cpp), NOT the CUDA __dp4a forms: gfx1100 has no
// signed byte dot (see the HIP DELTAS header note). Exactness is preserved by
// keeping each dot's accumulation ORDER identical to the CPU body it mirrors:
// Q2_K sums sub-blocks in k/j/l order; Q3_K/Q6_K use the 8-wide aux32 split;
// Q4_K/Q5_K accumulate per-16 bsums then per-32 scale groups in order.
// ---------------------------------------------------------------------------
__device__ inline float DotQ2K(const BlockQ2_K* xb, const BlockQ8_K* yb) {
  const uint8_t* q2 = xb->qs;
  const int8_t* q8 = yb->qs;
  const uint8_t* sc = xb->scales;
  int summs = 0;
  for (int j = 0; j < 16; ++j) summs += yb->bsums[j] * (sc[j] >> 4);
  const float dall = yb->d * DF16ToF32(xb->d);
  const float dmin = yb->d * DF16ToF32(xb->dmin);
  int isum = 0;
  int is = 0;
  for (int k = 0; k < kQK_K / 128; ++k) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      int d = sc[is++] & 0xF;
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
  return dall * isum - dmin * summs;
}

__device__ inline float DotQ3K(const BlockQ3_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint8_t* hm = xb->hmask;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  const uint8_t* q3 = xb->qs;
  uint8_t m = 1;
  for (int jj = 0; jj < kQK_K; jj += 128) {
    for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    q3 += 32;
  }
  uint32_t auxs[4];
  memcpy(auxs, xb->scales, 12);
  const int8_t* scales = reinterpret_cast<const int8_t*>(auxs);
  uint32_t tmp = auxs[2];
  auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  a = aux8;
  const int8_t* q8p = q8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * (q8p[l] * a[l]);
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * (q8p[l] * a[l]);
    q8p += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum;
}

__device__ inline float DotQ4K(const BlockQ4_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;
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
  // Portable nibble walk in the CPU body's order: 64-element groups decode
  // low nibble then high nibble into aux8, then the per-32 scale groups.
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  for (int j = 0; j < kQK_K / 64; ++j) {
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] & 0xF);
    a += 32;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] >> 4);
    a += 32;
    q4 += 32;
  }
  a = aux8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 32; ++j) {
    const int32_t scale = scales[j];
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
    q8 += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
    q8 += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
    q8 += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
    q8 += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum - dmin * sumi;
}

__device__ inline float DotQ5K(const BlockQ5_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;
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
  int8_t aux8[kQK_K];
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
  a = aux8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 32; ++j) {
    const int32_t scale = scales[j];
    for (int r = 0; r < 4; ++r) {
      for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
      q8 += 8; a += 8;
    }
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum - dmin * sumi;
}

__device__ inline float DotQ6K(const BlockQ6_K* xb, const BlockQ8_K* yb) {
  const uint8_t* q4 = xb->ql;
  const uint8_t* qh = xb->qh;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
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
    a += 128; q4 += 64; qh += 32;
  }
  a = aux8;
  const int8_t* q8p = q8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    const int scale = xb->scales[j];
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

__device__ inline float DotIQ2XXS(const BlockIQ2_XXS* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const uint16_t* qs = xb->qs;
  const int8_t* q8 = yb->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    uint32_t aux32[2];
    memcpy(aux32, qs + 4 * ib32, 2 * sizeof(uint32_t));
    const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
    int32_t sumi = 0;
    for (int l = 0; l < 4; ++l) {
      const uint8_t* grid = reinterpret_cast<const uint8_t*>(
          &vt::cuda::d_iq2xxs_grid[(aux32[0] >> (8 * l)) & 0xff]);
      const uint8_t signs =
          vt::cuda::d_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
      for (int j = 0; j < 8; ++j)
        sumi += grid[j] * q8[j] * ((signs & vt::cuda::d_kmask_iq2xs[j]) ? -1 : 1);
      q8 += 8;
    }
    bsum += sumi * static_cast<int32_t>(ls);
  }
  return d * bsum;  // final *0.125 applied after the warp reduction
}

__device__ inline float DotIQ3XXS(const BlockIQ3_XXS* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const uint8_t* q3 = xb->qs;
  const uint8_t* gas = xb->qs + kQK_K / 4;
  const int8_t* q8 = yb->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    uint32_t a32;
    memcpy(&a32, gas, sizeof(uint32_t));
    gas += sizeof(uint32_t);
    const uint32_t ls = 2 * (a32 >> 28) + 1;
    int32_t sumi = 0;
    for (int l = 0; l < 4; ++l) {
      const uint32_t g1 = vt::cuda::d_iq3xxs_grid[q3[2 * l + 0]];
      const uint32_t g2 = vt::cuda::d_iq3xxs_grid[q3[2 * l + 1]];
      const uint8_t signs = vt::cuda::d_ksigns_iq2xs[(a32 >> (7 * l)) & 127];
      for (int j = 0; j < 4; ++j) {
        const int b1 = static_cast<int>((g1 >> (8 * j)) & 0xff);
        const int b2 = static_cast<int>((g2 >> (8 * j)) & 0xff);
        sumi += b1 * q8[j + 0] * ((signs & vt::cuda::d_kmask_iq2xs[j + 0]) ? -1 : 1);
        sumi += b2 * q8[j + 4] * ((signs & vt::cuda::d_kmask_iq2xs[j + 4]) ? -1 : 1);
      }
      q8 += 8;
    }
    q3 += 8;
    bsum += sumi * static_cast<int32_t>(ls);
  }
  return d * bsum;  // final *0.25 applied after the warp reduction
}

__device__ inline float DotIQ2S(const BlockIQ2_S* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const int8_t* q8 = yb->qs;
  const uint8_t* qs = xb->qs;
  const uint8_t* qh = xb->qh;
  const uint8_t* signs = qs + kQK_K / 8;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    const int ls1 = 1 + 2 * (xb->scales[ib32] & 0xf);
    const int ls2 = 1 + 2 * (xb->scales[ib32] >> 4);
    int sumi1 = 0;
    int sumi2 = 0;
    for (int l = 0; l < 2; ++l) {
      const uint8_t* grid = reinterpret_cast<const uint8_t*>(
          &vt::cuda::d_iq2s_grid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)]);
      for (int j = 0; j < 8; ++j)
        sumi1 += q8[j] * grid[j] * ((signs[l] & vt::cuda::d_kmask_iq2xs[j]) ? -1 : 1);
      q8 += 8;
    }
    for (int l = 2; l < 4; ++l) {
      const uint8_t* grid = reinterpret_cast<const uint8_t*>(
          &vt::cuda::d_iq2s_grid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)]);
      for (int j = 0; j < 8; ++j)
        sumi2 += q8[j] * grid[j] * ((signs[l] & vt::cuda::d_kmask_iq2xs[j]) ? -1 : 1);
      q8 += 8;
    }
    bsum += ls1 * sumi1 + ls2 * sumi2;
    qs += 4;
    signs += 4;
  }
  return d * bsum;  // final *0.125 applied after the warp reduction
}

__device__ inline float DotIQ1S(const BlockIQ1_S* xb, const BlockQ8_K* yb) {
  const int8_t* q8 = yb->qs;
  const uint8_t* qs = xb->qs;
  const uint16_t* qh = xb->qh;
  int32_t sumi = 0;
  int32_t sumi1 = 0;
  for (int ib = 0; ib < kQK_K / 32; ++ib) {
    const int ls = 2 * ((qh[ib] >> 12) & 7) + 1;
    const int delta = (qh[ib] & 0x8000) ? -1 : 1;
    int lsum = 0;
    for (int l = 0; l < 4; ++l) {
      const int8_t* grid = reinterpret_cast<const int8_t*>(
          &vt::cuda::d_iq1s_grid[qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8)]);
      for (int j = 0; j < 8; ++j) lsum += q8[j] * grid[j];
      q8 += 8;
    }
    sumi += ls * lsum;
    sumi1 += ls * delta * (yb->bsums[2 * ib + 0] + yb->bsums[2 * ib + 1]);
    qs += 4;
  }
  return DF16ToF32(xb->d) * yb->d *
         (static_cast<float>(sumi) + 0.125f * static_cast<float>(sumi1));
}

__device__ inline float DotIQ1XXXS(const BlockIQ1_XXXS* xb, const BlockQ8_K* yb) {
  const int8_t* q8 = yb->qs;
  const uint8_t* qs = xb->qs;
  const uint8_t* sc = xb->sc;
  int32_t sumi = 0;
  int32_t sumi1 = 0;
  for (int ib = 0; ib < kQK_K / 32; ++ib) {
    const int nib = (sc[ib / 2] >> (4 * (ib & 1))) & 0xf;
    const int ls = 2 * (nib & 7) + 1;
    const int delta = (nib & 8) ? -1 : 1;
    int lsum = 0;
    for (int l = 0; l < 4; ++l) {
      const int8_t* grid =
          reinterpret_cast<const int8_t*>(&vt::cuda::d_iq1xxxs_grid[qs[l]]);
      for (int j = 0; j < 8; ++j) lsum += q8[j] * grid[j];
      q8 += 8;
    }
    sumi += ls * lsum;
    sumi1 += ls * delta * (yb->bsums[2 * ib + 0] + yb->bsums[2 * ib + 1]);
    qs += 4;
  }
  return DF16ToF32(xb->d) * yb->d *
         (static_cast<float>(sumi) + 0.125f * static_cast<float>(sumi1));
}

// ---------------------------------------------------------------------------
// WType tags + DotSuperblock dispatch — mirrors the CUDA file's table.
// ---------------------------------------------------------------------------
enum class WType : int {
  kIQ2_XXS = 0,
  kIQ3_XXS = 1,
  kQ2_K = 2,
  kQ3_K = 3,
  kQ4_K = 4,
  kQ5_K = 5,
  kQ6_K = 6,
  kIQ2_S = 7,
  kIQ1_S = 8,
  kIQ1_XXXS = 9,
};

template <WType W>
__device__ inline float DotSuperblock(const void* w_sb, const BlockQ8_K* a_sb);

template <>
__device__ inline float DotSuperblock<WType::kIQ2_XXS>(const void* w, const BlockQ8_K* a) {
  return DotIQ2XXS(static_cast<const BlockIQ2_XXS*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ3_XXS>(const void* w, const BlockQ8_K* a) {
  return DotIQ3XXS(static_cast<const BlockIQ3_XXS*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ2_K>(const void* w, const BlockQ8_K* a) {
  return DotQ2K(static_cast<const BlockQ2_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ3_K>(const void* w, const BlockQ8_K* a) {
  return DotQ3K(static_cast<const BlockQ3_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ4_K>(const void* w, const BlockQ8_K* a) {
  return DotQ4K(static_cast<const BlockQ4_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ5_K>(const void* w, const BlockQ8_K* a) {
  return DotQ5K(static_cast<const BlockQ5_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ6_K>(const void* w, const BlockQ8_K* a) {
  return DotQ6K(static_cast<const BlockQ6_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ2_S>(const void* w, const BlockQ8_K* a) {
  return DotIQ2S(static_cast<const BlockIQ2_S*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ1_S>(const void* w, const BlockQ8_K* a) {
  return DotIQ1S(static_cast<const BlockIQ1_S*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ1_XXXS>(const void* w, const BlockQ8_K* a) {
  return DotIQ1XXXS(static_cast<const BlockIQ1_XXXS*>(w), a);
}

template <WType W>
__device__ constexpr float FinalFactor() {
  return (W == WType::kIQ2_XXS || W == WType::kIQ2_S)
             ? 0.125f
             : (W == WType::kIQ3_XXS ? 0.25f : 1.0f);
}

// ---------------------------------------------------------------------------
// The MMVQ-style GEMM: one WARP per output element (i,j). Lanes split the K
// super-blocks; the warp reduction sums the partials. HIP delta: the shuffle
// mask is 64-bit on this target. Determinism note unchanged from CUDA: the
// integer core is exact; only the scale sum reassociates (within NMSE).
// ---------------------------------------------------------------------------
template <WType W, typename OutT>
__global__ void QuantDotGemmKernel(OutT* __restrict__ out,
                                   const uint8_t* __restrict__ weight,
                                   const BlockQ8_K* __restrict__ act, int64_t m,
                                   int64_t n, int64_t nsb, size_t w_row_bytes,
                                   size_t w_block_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) +
                       (threadIdx.x >> 5);
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x & 31;

  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_K* a_row = act + i * nsb;

  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    partial += DotSuperblock<W>(w_sb, a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffffffffffull, partial, off);

  if (lane == 0) {
    const float v = FinalFactor<W>() * partial;
    if constexpr (sizeof(OutT) == 4) {
      out[i * n + j] = v;
    } else {
      out[i * n + j] = DF32ToBF16(v);
    }
  }
}

// GROUPED variant: warp per (p, n); weight row selected by expert_ids[p].
template <WType W, typename OutT>
__global__ void QuantDotGemmGroupedKernel(OutT* __restrict__ out,
                                          const uint8_t* __restrict__ weight,
                                          const BlockQ8_K* __restrict__ act,
                                          const int32_t* __restrict__ expert_ids,
                                          int64_t P, int64_t n, int64_t nsb,
                                          size_t w_row_bytes,
                                          size_t w_block_bytes, bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) +
                       (threadIdx.x >> 5);
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x & 31;

  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  // Broadcast activation: the routed gate/up share ONE quantized hidden.
  const BlockQ8_K* a_row = act + (bcast ? 0 : p) * nsb;

  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    partial += DotSuperblock<W>(w_sb, a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffffffffffull, partial, off);

  if (lane == 0) {
    const float v = FinalFactor<W>() * partial;
    if constexpr (sizeof(OutT) == 4) {
      out[p * n + j] = v;
    } else {
      out[p * n + j] = DF32ToBF16(v);
    }
  }
}

// ---------------------------------------------------------------------------
// Host wiring: scratch, launches, providers, registrar.
// ---------------------------------------------------------------------------
struct StreamScratch {
  void* buf = nullptr;
  size_t bytes = 0;
};

std::mutex& ScratchMutex() {
  static std::mutex mu;
  return mu;
}

StreamScratch& ScratchFor(hipStream_t s) {
  static std::unordered_map<hipStream_t, StreamScratch> map;
  return map[s];
}

void CheckHipLaunch(const char* what) { CheckHip(hipGetLastError(), what); }

void* EnsureScratch(size_t need, hipStream_t s) {
  std::lock_guard<std::mutex> lock(ScratchMutex());
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.bytes) {
    // Retire (never free): a captured hipGraph may have baked this pointer.
    vt::cuda::RetireGraphScratch(sc.buf);
    CheckHip(hipMallocAsync(&sc.buf, need, s), "hipMallocAsync q8_K act scratch");
    sc.bytes = need;
  }
  return sc.buf;
}

inline ActDT ActDtOf(DType dt) {
  return dt == DType::kF32 ? ActDT::kF32 : dt == DType::kF16 ? ActDT::kF16 : ActDT::kBF16;
}

void LaunchQuantizeQ8K(BlockQ8_K* qact, const void* data, ActDT adt, int64_t a_rs,
                       int64_t rows, int64_t nsb, hipStream_t s) {
  dim3 qgrid(static_cast<unsigned>(nsb), static_cast<unsigned>(rows), 1);
  QuantizeQ8KKernel<<<qgrid, kQK_K, 0, s>>>(qact, data, adt, a_rs, rows, nsb);
  CheckHipLaunch("quantize_q8_K launch");
}

bool IsRocmKeepQuantSupported(DType dt, WType* out) {
  switch (dt) {
    case DType::kIQ2_XXS: *out = WType::kIQ2_XXS; return true;
    case DType::kIQ3_XXS: *out = WType::kIQ3_XXS; return true;
    case DType::kQ2_K: *out = WType::kQ2_K; return true;
    case DType::kQ3_K: *out = WType::kQ3_K; return true;
    case DType::kQ4_K: *out = WType::kQ4_K; return true;
    case DType::kQ5_K: *out = WType::kQ5_K; return true;
    case DType::kQ6_K: *out = WType::kQ6_K; return true;
    case DType::kIQ2_S: *out = WType::kIQ2_S; return true;
    case DType::kIQ1_S: *out = WType::kIQ1_S; return true;
    case DType::kIQ1_XXXS: *out = WType::kIQ1_XXXS; return true;
    // Q4_0 / Q8_0 / MXFP4 dot a Q8_0 activation and have no native arm here.
    default: return false;
  }
}

template <WType W>
void LaunchGemm(Tensor& out, const uint8_t* weight, const BlockQ8_K* act,
                int64_t m, int64_t n, int64_t nsb, size_t w_row_bytes,
                size_t w_block_bytes, hipStream_t s) {
  // Wave32 geometry: one warp per output; 8 warps (256 threads) per block.
  constexpr int kWarpsPerBlock = 8;
  dim3 block(32 * kWarpsPerBlock, 1, 1);
  const int64_t warps = m * n;
  const unsigned grid =
      static_cast<unsigned>((warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (out.dtype == DType::kF32) {
    QuantDotGemmKernel<W, float><<<grid, block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, m, n, nsb, w_row_bytes,
        w_block_bytes);
  } else {
    QuantDotGemmKernel<W, uint16_t><<<grid, block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, m, n, nsb, w_row_bytes,
        w_block_bytes);
  }
  CheckHipLaunch("matmul_bt_quant launch");
}

template <WType W>
void LaunchGroupedGemm(Tensor& out, const uint8_t* weight, const BlockQ8_K* act,
                       const int32_t* expert_ids, int64_t P, int64_t n,
                       int64_t nsb, size_t w_row_bytes, size_t w_block_bytes,
                       bool bcast, hipStream_t s) {
  constexpr int kWarpsPerBlock = 8;
  dim3 block(32 * kWarpsPerBlock, 1, 1);
  const int64_t warps = P * n;
  const unsigned grid =
      static_cast<unsigned>((warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (out.dtype == DType::kF32) {
    QuantDotGemmGroupedKernel<W, float><<<grid, block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, expert_ids, P, n, nsb,
        w_row_bytes, w_block_bytes, bcast);
  } else {
    QuantDotGemmGroupedKernel<W, uint16_t><<<grid, block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, expert_ids, P, n, nsb,
        w_row_bytes, w_block_bytes, bcast);
  }
  CheckHipLaunch("matmul_bt_quant_grouped launch");
}



// Q8_0 (legacy 32-block, Q8_0-activation) arm. Self-contained: quantize the
// activation to Q8_0 on the device, then the Q8_0xQ8_0 integer dot. The dot is
// the PORTABLE SCALAR form of cpu_quant_dot.cpp VecDotQ8_0Q8_0 (quants.c:400):
// sumi0 over the first 16 elements, sumi1 over the second, ONE int32 add, then
// the f16-scale fold — accumulation order preserved for exactness. The CUDA
// file's __dp4a form is byte-identical to this on a signed-dot target; gfx1100
// has no signed byte dot, so W1 ships the scalar body.
// ---------------------------------------------------------------------------
__global__ void QuantizeQ8_0Kernel(BlockQ8_0* __restrict__ scratch,
                                   const void* __restrict__ a, ActDT adt,
                                   int64_t a_rs, int64_t m, int64_t nb) {
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
    y.qs[j] = static_cast<int8_t>(roundf(DLoadAct(a, adt, elem0 + j) * id));
  }
}

template <typename OutT>
__global__ void QuantDotGemmQ8_0Kernel(OutT* __restrict__ out,
                                       const uint8_t* __restrict__ weight,
                                       const BlockQ8_0* __restrict__ act,
                                       int64_t m, int64_t n, int64_t nb,
                                       size_t w_row_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) +
                       (threadIdx.x >> 5);
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x & 31;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const BlockQ8_0* wb =
        reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                      sizeof(BlockQ8_0));
    const BlockQ8_0* ab = a_row + b;
    // CPU order: first half then second half, one add, then scale.
    int sumi0 = 0;
    for (int j2 = 0; j2 < kQK8_0 / 2; ++j2)
      sumi0 += ab->qs[j2] * wb->qs[j2];
    int sumi1 = 0;
    for (int j2 = kQK8_0 / 2; j2 < kQK8_0; ++j2)
      sumi1 += ab->qs[j2] * wb->qs[j2];
    partial += (sumi0 + sumi1) * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffffffffffull, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4)
      out[i * n + j] = partial;
    else
      out[i * n + j] = DF32ToBF16(partial);
  }
}

void MatmulQ8_0Rocm(Tensor& out, const Tensor& a, const Tensor& b,
                    hipStream_t s) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  if (m == 0 || n == 0) return;
  if (k % kQK8_0 != 0)
    throw std::runtime_error(
        "vt rocm: matmul_bt_quant Q8_0: K must be a multiple of 32");
  const int64_t nb = k / kQK8_0;
  const size_t w_row_bytes = static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  const size_t act_bytes =
      static_cast<size_t>(m) * static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  BlockQ8_0* act = static_cast<BlockQ8_0*>(EnsureScratch(act_bytes, s));
  {
    constexpr int kQBlock = 128;
    const int64_t grid = (m * nb + kQBlock - 1) / kQBlock;
    QuantizeQ8_0Kernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        act, a.data, ActDtOf(a.dtype), a.stride[0], m, nb);
    CheckHipLaunch("quantize_q8_0 launch");
  }
  constexpr int kWarpsPerBlock = 8;
  dim3 block(32 * kWarpsPerBlock, 1, 1);
  const unsigned grid =
      static_cast<unsigned>((m * n + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (out.dtype == DType::kF32)
    QuantDotGemmQ8_0Kernel<float><<<grid, block, 0, s>>>(
        static_cast<float*>(out.data), static_cast<const uint8_t*>(b.data),
        act, m, n, nb, w_row_bytes);
  else
    QuantDotGemmQ8_0Kernel<uint16_t><<<grid, block, 0, s>>>(
        static_cast<uint16_t*>(out.data), static_cast<const uint8_t*>(b.data),
        act, m, n, nb, w_row_bytes);
  CheckHipLaunch("matmul_bt_quant Q8_0 launch");
}

// The kROCM provider for OpId::kMatmulBTQuant. Validation already done by
// vt::MatmulBTQuant (ops.cpp). Contract: b is [N,K] block-quant, a [M,K]
// f32/bf16 row-packed, out [M,N]. On this DISCRETE backend an unsupported
// dtype cannot fall back to the CPU kernel (it would follow device pointers),
// so it throws naming the dtype — VT_GGUF_KEEP_QUANT=0 restores the load-time
// bf16 expansion for such files.
void MatmulBTQuantKernelRocm(Queue& q, Tensor& out, const Tensor& a,
                             const Tensor& b) {
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = b.shape[0];
  if (b.dtype == DType::kQ8_0) {
    MatmulQ8_0Rocm(out, a, b, s);
    return;
  }

  // Delegate Q4_K/Q5_K/Q6_K to the optimized kernels in rocm_grouped_gemm.hip
  // (renamed *Gdn). These have cooperative-warp dispatch and format-specific
  // templates tuned for gfx1100; the new CUDA-ported kernels in this file are
  // slower for these types. The new kernels handle IQ types and Q2_K/Q3_K.
  if (b.dtype == DType::kQ4_K || b.dtype == DType::kQ5_K ||
      b.dtype == DType::kQ6_K) {
    MatmulBTQuantKernelRocmGdn(q, out, a, b);
    return;
  }

  WType w{};
  if (!IsRocmKeepQuantSupported(b.dtype, &w)) {
    throw std::runtime_error(
        std::string("vt rocm: matmul_bt_quant: no keep-quant kernel for dtype ") +
        Name(b.dtype) +
        " (this discrete backend has no host fallback; set VT_GGUF_KEEP_QUANT=0 "
        "to expand the file to bf16 at load)");
  }
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt rocm: matmul_bt_quant: K must be a whole number of 256-element "
        "Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  const size_t w_block_bytes = static_cast<size_t>(vt::BlockBytes(b.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;

  const size_t act_bytes =
      static_cast<size_t>(m) * static_cast<size_t>(nsb) * sizeof(BlockQ8_K);
  BlockQ8_K* act = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));
  LaunchQuantizeQ8K(act, a.data, ActDtOf(a.dtype), a.stride[0], m, nsb, s);

  const uint8_t* weight = static_cast<const uint8_t*>(b.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGemm<WType::kIQ2_XXS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ3_XXS: LaunchGemm<WType::kIQ3_XXS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ2_K: LaunchGemm<WType::kQ2_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ3_K: LaunchGemm<WType::kQ3_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ4_K: LaunchGemm<WType::kQ4_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ5_K: LaunchGemm<WType::kQ5_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ6_K: LaunchGemm<WType::kQ6_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ2_S: LaunchGemm<WType::kIQ2_S>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ1_S: LaunchGemm<WType::kIQ1_S>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ1_XXXS: LaunchGemm<WType::kIQ1_XXXS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    // IsRocmKeepQuantSupported answered yes, so a missing case must be LOUD:
    // launching nothing leaves `out` stale while callers see success (#967).
    default:
      throw std::runtime_error(
          std::string("vt rocm: matmul_bt_quant: no keep-quant kernel for dtype ") +
          Name(b.dtype));
  }
}

void MatmulBTQuantGroupedKernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                    const Tensor& weight,
                                    const Tensor& expert_ids) {
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const int64_t P = out.shape[0];
  const int64_t n = out.shape[1];
  const int64_t k = act.shape[1];
  if (P == 0 || n == 0) return;

  // Delegate Q8_0/Q4_K/Q5_K/Q6_K to the optimized grouped kernels. Q8_0 has no
  // Q8_K-superblock arm in this file (it dots a Q8_0 activation), so leaving it
  // out of this list turns a format the *Gdn kernel serves today into a throw.
  if (weight.dtype == DType::kQ8_0 || weight.dtype == DType::kQ4_K ||
      weight.dtype == DType::kQ5_K || weight.dtype == DType::kQ6_K) {
    MatmulBTQuantGroupedKernelRocmGdn(q, out, act, weight, expert_ids);
    return;
  }

  WType w{};
  if (!IsRocmKeepQuantSupported(weight.dtype, &w)) {
    throw std::runtime_error(
        std::string("vt rocm: matmul_bt_quant_grouped: no keep-quant kernel for "
                    "dtype ") +
        Name(weight.dtype) +
        " (set VT_GGUF_KEEP_QUANT=0 to expand at load)");
  }
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt rocm: matmul_bt_quant_grouped: K must be a whole number of "
        "256-element Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  const size_t w_block_bytes = static_cast<size_t>(vt::BlockBytes(weight.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;

  // Broadcast activation (preq-reuse): ONE quantized hidden feeds every routed
  // expert slot; bit-identical because identical input yields identical Q8_K.
  const int64_t Pa = act.shape[0];
  const bool bcast = (Pa == 1 && P > 1);

  const size_t act_bytes =
      static_cast<size_t>(Pa) * static_cast<size_t>(nsb) * sizeof(BlockQ8_K);
  BlockQ8_K* qact = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));
  LaunchQuantizeQ8K(qact, act.data, ActDtOf(act.dtype), act.stride[0], Pa, nsb,
                    s);

  const uint8_t* wt = static_cast<const uint8_t*>(weight.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGroupedGemm<WType::kIQ2_XXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ3_XXS: LaunchGroupedGemm<WType::kIQ3_XXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ2_K: LaunchGroupedGemm<WType::kQ2_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ3_K: LaunchGroupedGemm<WType::kQ3_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ4_K: LaunchGroupedGemm<WType::kQ4_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ5_K: LaunchGroupedGemm<WType::kQ5_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ6_K: LaunchGroupedGemm<WType::kQ6_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ2_S: LaunchGroupedGemm<WType::kIQ2_S>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ1_S: LaunchGroupedGemm<WType::kIQ1_S>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ1_XXXS: LaunchGroupedGemm<WType::kIQ1_XXXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    default:
      throw std::runtime_error(
          std::string("vt rocm: matmul_bt_quant_grouped: no grouped kernel for "
                      "keep-quant dtype ") +
          Name(weight.dtype));
  }
}

// Registers the ROCm keep-quant GEMM during static init (table fill only, no
// HIP calls — same contract as every other registrar). The registration is what
// makes GgufQuantComputeAvailable() answer yes for kROCM, so the GGUF loader
// keeps the file quantized instead of expanding it to bf16 at load.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MatmulFn>(&MatmulBTQuantKernelRocm)));
    RegisterOp(OpId::kMatmulBTQuantGrouped, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MatmulBTQuantGroupedFn>(
                       &MatmulBTQuantGroupedKernelRocm)));
  }
} registrar;

}  // namespace
}  // namespace vt::rocm
