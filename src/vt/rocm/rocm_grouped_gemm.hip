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
// rocWMMA int8 tile arm (KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4, issue #2109),
// gfx1200/gfx1201 (RDNA4) only — same gate rocm_paged_attn.hip already uses
// for its own WMMA arm, and the same reason: a fat build also emits other
// gfx12 targets (for example gfx1036) that must not see this include.
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
#define VT_ROCM_QUANT_WMMA_OK 1
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vt/grow_only_stream_scratch.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_arch.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_f16_codec.h"
#include "vt/rocm/rocm_runtime.h"

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
using Q8KKernelWitnessCount = unsigned long long;
inline ActDT ActDtOf(DType dt) {
  return dt == DType::kF32 ? ActDT::kF32 : dt == DType::kF16 ? ActDT::kF16 : ActDT::kBF16;
}

// ---- device numeric helpers ----
// The four half/bfloat16 conversions moved to `vt/rocm/rocm_f16_codec.h`
// (included above) when BACKEND-ROCM-EXL3 needed the same two of them for
// the trellis decode. They are unchanged, and they are still the bit-exact
// transcriptions of `vt::F16ToF32` / `vt::F32ToF16` rather than the hardware
// conversion intrinsics; a second copy is what moving them avoided.
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

// Signed 8-bit x4 dot-product-accumulate. Uses the HW v_dot4_i32_i8
// instruction (__ockl_sdot4) on gfx1100 — one instruction instead of 4
// int8 multiplies + 4 adds. Bit-identical: signed int8×int8→int32 dot
// product is exact either way (the HW instruction and the scalar expansion
// compute the same integer result).
__device__ __forceinline__ int Dp4a(int a, int b, int acc) {
  using char4_native = char __attribute__((ext_vector_type(4)));
  char4_native va = *reinterpret_cast<const char4_native*>(&a);
  char4_native vb = *reinterpret_cast<const char4_native*>(&b);
  return __ockl_sdot4(va, vb, acc, false);
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
                             ActDT adt, int64_t a_rs, int64_t m, int64_t nsb,
                             Q8KKernelWitnessCount* kernel_witness) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * nsb) return;
  if (t == 0 && kernel_witness != nullptr) atomicAdd(&kernel_witness[0], 1ULL);
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

// Q8_K (block-per-256-superblock): cuda_quant_dot.cu
// QuantizeQ8KPreqKernel. One thread loads and writes one activation element.
// The reduction retains the signed value and the lowest original index, which
// preserves the legacy first-occurrence rule for opposite-sign absolute ties.
__global__ void QuantizeQ8KCooperativeK(BlockQ8_K* __restrict__ scratch,
                                        const void* __restrict__ a, ActDT adt,
                                        int64_t a_rs, int64_t m, int64_t nsb,
                                        Q8KKernelWitnessCount* kernel_witness) {
  const int64_t sb = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (sb >= nsb || row >= m) return;
  const int tid = static_cast<int>(threadIdx.x);
  if (sb == 0 && row == 0 && tid == 0 && kernel_witness != nullptr)
    atomicAdd(&kernel_witness[1], 1ULL);
  const int64_t elem0 = row * a_rs + sb * kQK_K;
  const float value = DLoadAct(a, adt, elem0 + tid);

  __shared__ float reduction_abs[kQK_K];
  __shared__ float reduction_value[kQK_K];
  __shared__ int reduction_index[kQK_K];
  reduction_abs[tid] = fabsf(value);
  reduction_value[tid] = value;
  reduction_index[tid] = tid;
  __syncthreads();

#pragma unroll
  for (int stride = kQK_K / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_abs = reduction_abs[tid + stride];
      const int other_index = reduction_index[tid + stride];
      if (other_abs > reduction_abs[tid] ||
          (other_abs == reduction_abs[tid] && other_index < reduction_index[tid])) {
        reduction_abs[tid] = other_abs;
        reduction_value[tid] = reduction_value[tid + stride];
        reduction_index[tid] = other_index;
      }
    }
    __syncthreads();
  }

  BlockQ8_K& output = scratch[row * nsb + sb];
  if (reduction_abs[0] == 0.0f) {
    if (tid == 0) output.d = 0.0f;
    output.qs[tid] = 0;
    if (tid < kQK_K / 16) output.bsums[tid] = 0;
    return;
  }

  const float iscale = -127.0f / reduction_value[0];
  const int quant = DNearestInt(iscale * value);
  output.qs[tid] = static_cast<int8_t>(quant < 127 ? quant : 127);
  __syncthreads();
  if (tid < kQK_K / 16) {
    int sum = 0;
#pragma unroll
    for (int ii = 0; ii < 16; ++ii) sum += output.qs[tid * 16 + ii];
    output.bsums[tid] = static_cast<int16_t>(sum);
  }
  if (tid == 0) output.d = 1.0f / iscale;
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

// Reconstructs the 16 RAW (unscaled) int8 weights of Q6_K scale-group `grp`
// (`grp` in [0, kQK_K/16) == [0, 16)).
// Bit-identical to `DotQ6KIsumRange`'s own reconstruction: `grp = g*2 + t`
// selects the same 32-wide sub-block `g` and the same upper/lower 16-wide
// half `t` that function already walks, just returning the 16 values instead
// of folding them into a dot product. Kept beside it rather than expressed as
// a call into it, because that function accumulates over a *range* of groups
// and has no seam to hand back one group's raw values without either
// widening its contract or adding a second exit path to it.
//
// Guarded by VT_ROCM_QUANT_WMMA_OK (only the WMMA kernel body below calls it):
// on the host pass and on every non-gfx1200/gfx1201 device pass that body
// takes its stub branch, which would leave this genuinely unused and trip
// -Werror,-Wunused-function.
#if defined(VT_ROCM_QUANT_WMMA_OK)
__device__ inline void DequantQ6KGroup16(const BlockQ6_K* xb, int grp, int8_t out[16]) {
  const int g = grp >> 1;
  const int t = grp & 1;
  const uint8_t* q4 = xb->ql + (g / 4) * 64;
  const uint8_t* qh = xb->qh + (g / 4) * 32;
  const int r = g & 3;
  const uint8_t* q4b = q4 + ((r & 1) ? 32 : 0);
  const int nib_shift = (r >= 2) ? 4 : 0;
  const int qh_shift = 2 * r;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const int l = t * 16 + k;
    out[k] = static_cast<int8_t>(
        static_cast<int8_t>(((q4b[l] >> nib_shift) & 0xF) | (((qh[l] >> qh_shift) & 3) << 4)) - 32);
  }
}
#endif  // VT_ROCM_QUANT_WMMA_OK

// --- RDNA4 WMMA int8 tile arm of KQuantGemmK (KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4,
// issue #2109). Q6_K only in this first cut: the row's spec lists Q4_K/Q5_K —
// same shape, same 16-wide-group scale layout — as `## Owed` to a follow-up
// commit, and M/N tail handling (m or n not a multiple of 16) as owed to
// whoever generalizes the launch site below.
//
// One warp computes one 16x16 tile of `out[i0:i0+16, j0:j0+16]`. Q6_K's 256-
// element superblock has exactly `kQK_K/16 == 16` independent int8 scales
// (one per 16-wide group), which is exactly one WMMA K-tile (`WK == 16`), so
// each group maps to one `mma_sync` with no ragged remainder — the spec's
// `## Design` calls this out as the reason 16 was the tile width chosen.
//
// The scale is per (weight row, group), so it cannot be folded into the int8
// operands ahead of the tensor-core op without overflowing int8 (a 6-bit
// scale times a 6-bit dequant already exceeds it). It is applied in an
// epilogue instead, exactly like the scalar `DotQ6KIsumRange` arm: the raw
// (unscaled) tile the tensor core produces is staged to shared memory, and
// every lane folds `isum[i][j] += raw[i][j] * scale[j]` in INTEGER
// arithmetic. Integer addition is exact and reorders freely, so `isum` here
// is bit-for-bit the quantity the scalar warp computes; only the
// superblock's one f32 scale product (`d * isum`, applied once below, same as
// every other arm in this file) is shared with the rest of the reduction.
//
// This first cut bounces every group's raw tile through shared memory before
// scaling it, which is a known, NOT-yet-measured performance cost (a
// synchronization and a shared-memory round trip every 16 K-elements) rather
// than a correctness concern — the row's spec gate (c) (rocprofv3 total
// kernel time) is not asserted by this commit, and closing it is follow-up
// work once this mechanism is confirmed correct on-target.
// Plain integers, never gated by VT_ROCM_QUANT_WMMA_OK: the host launch site
// below needs kQuantWmmaWarpsPerBlock on every compilation pass, including
// the host pass, where that macro is never defined; kGroupGemmBatch is
// shared by every WMMA-format kernel in this file (Q4_K's below included),
// so it lives at file scope rather than once per kernel body.
constexpr int kQuantWmmaWarpsPerBlock = 4;
// KERNEL-QUANT-CIQ-GEMM-ROCM-WIDEBLOCK (issue #3032): llama.cpp's `mul_mat_q`
// runs the same 16x16x16 int8 WMMA tile at 8 warps/block (256 threads) and
// finishes each launch 15-45x faster than this kernel's 4-warp/block arm at
// matching shapes, per a same-tool `rocprofv3` trace. Each warp here already
// owns a fully independent output tile with its own shared-memory slice
// (indexed by `wy` throughout both kernel bodies below) and loops the whole
// `nsb` reduction alone, so widening the block was tested as a launch-count
// lever: does packing more independent warps per launch (half as many
// blocks, same total work) close any of the gap?
//
// MEASURED AND REJECTED (`examples/quant-gemm-bench`, RX 9060 XT, gfx1200,
// best-of-4, idle host): geomean -4.3% across the six Q4_K/Q6_K prefill
// shapes (range -9.6% to +2.6%), i.e. a net regression, not a win. Each warp
// here shares NO data with any other warp in its block -- no activation row,
// no dequantized weight tile -- so a wider block only adds independent warps
// to the SAME occupancy budget the LDS footprint already caps (4 warps at
// this kernel's ~6.4 KiB/warp already uses two blocks' worth of the same
// 64 KiB/CU two 4-warp blocks would use resident together), which is why
// this axis alone does not move the needle: llama.cpp's block is not merely
// wider, it very likely shares loaded/dequantized data ACROSS its tiles in a
// way this kernel's per-warp-independent design does not. That is the
// sharper, still-open hypothesis issue #3032 records as the next thing to
// try, not this one. `kQuantWmmaWideWarpsPerBlock` and
// `VT_ROCM_QUANT_WMMA_WIDE` stay in the tree, default OFF, as a ready-made
// A/B for re-checking this specific axis on different hardware (gfx1201) or
// a future ROCm/ toolchain revision, rather than re-deriving it from
// scratch — matching this file's own `Q6KSmallPrivateEnabled` precedent for
// a measured-and-rejected diagnostic arm.
constexpr int kQuantWmmaWideWarpsPerBlock = 8;
constexpr int kGroupGemmBatch = 2;
static_assert(16 % kGroupGemmBatch == 0, "16 scale-groups must divide evenly into batches");
// LDS budget check for the widest arm actually instantiated below (Q4_K is
// the larger of the two: w_stage 4096B + row_scales/row_mins 128B each +
// raw_tile 2048B = 6400B per warp slot). At kQuantWmmaWideWarpsPerBlock == 8
// that is 51200B, comfortably under the 64KiB/block LDS limit this project's
// other WMMA kernels stay inside (see `rocm_paged_attn.hip`'s own note).
static_assert(kQuantWmmaWideWarpsPerBlock * 6400 <= 65536,
             "widened WMMA block must stay under the 64KiB LDS budget");

template <typename OutT, int WarpsPerBlock>
__global__ void KQuantGemmKWmmaQ6K(OutT* __restrict__ out, const uint8_t* __restrict__ weight,
                                   const BlockQ8_K* __restrict__ act, int64_t m, int64_t n,
                                   int64_t nsb, int64_t n_tiles) {
#if !defined(VT_ROCM_QUANT_WMMA_OK)
  // Compiled for a target other than gfx1200/gfx1201, or for the host pass:
  // never launched there (`Gfx12QuantWmmaHostOk` gates every launch site
  // below on the actual device's runtime gcnArchName, not on this macro —
  // the macro is only visible to a device compilation pass for gfx1200/
  // gfx1201 itself, exactly the trap issue #785 already found and fixed for
  // `rocm_paged_attn.hip`'s own WMMA arm).
  (void)out; (void)weight; (void)act; (void)m; (void)n; (void)nsb; (void)n_tiles;
  return;
#else
  using namespace rocwmma;
  const int64_t tile = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock +
                       static_cast<int64_t>(threadIdx.y);
  const int64_t m_tiles = m / 16;
  if (tile >= m_tiles * n_tiles) return;
  const int64_t it = tile / n_tiles;
  const int64_t jt = tile % n_tiles;
  const int64_t i0 = it * 16;
  const int64_t j0 = jt * 16;
  const int lane = static_cast<int>(threadIdx.x);
  const int wy = static_cast<int>(threadIdx.y);

  // w_stage holds ALL 16 scale-groups' dequantized tiles for the current
  // superblock, filled once per superblock rather than once per group: the
  // first cut synced after every single-group fill (16 barriers/superblock
  // for that step alone), and every lane sat idle half the time doing it
  // (only 16 of 32 lanes filled one row each). Spreading the same total work
  // over all 32 lanes across all 16 groups removes both costs in one change.
  __shared__ int8_t w_stage[WarpsPerBlock][16][16][16];
  // raw_tile holds kGroupGemmBatch groups' raw WMMA tiles at once, so a
  // whole batch's compute+store is one barrier and its scale-fold is
  // another, rather than a barrier per single group each way. 16 KiB
  // (w_stage) + 8 KiB (this, at batch 2, WarpsPerBlock==4) = 24 KiB/block,
  // comfortably under the 64 KiB LDS budget this project's other WMMA
  // kernels stay inside (rocm_paged_attn.hip's own BM=16 BN=64 d=256 note);
  // see the file-scope `static_assert` above for the widened-block budget.
  __shared__ int32_t raw_tile[WarpsPerBlock][kGroupGemmBatch][16][16];

  const size_t row_stride = static_cast<size_t>(nsb) * sizeof(BlockQ6_K);
  const BlockQ6_K* w_row_base[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    w_row_base[r] =
        reinterpret_cast<const BlockQ6_K*>(weight + static_cast<size_t>(j0 + r) * row_stride);
  }
  const BlockQ8_K* a_row0 = act + i0 * nsb;

  float acc_f[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) acc_f[c] = 0.0f;

  for (int64_t sb = 0; sb < nsb; ++sb) {
    const BlockQ6_K* xb_cur[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) xb_cur[r] = w_row_base[r] + sb;

    int32_t isum[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) isum[c] = 0;

    // One fill for the whole superblock: 16 groups x 16 rows = 256 (row,
    // group) pairs, 8 per lane, each call still filling that pair's 16
    // elements via DequantQ6KGroup16 unchanged.
    for (int t = lane; t < 16 * 16; t += 32) {
      const int row = t / 16;
      const int grp2 = t % 16;
      DequantQ6KGroup16(xb_cur[row], grp2, w_stage[wy][grp2][row]);
    }
    __syncthreads();

    // A batch of kGroupGemmBatch groups: every WMMA op in the batch computes
    // and stores to its OWN slice of raw_tile before the batch's single
    // barrier, and every group's scale-fold reads back before the batch's
    // other barrier — replacing what was a barrier PER GROUP each way.
    for (int gb = 0; gb < 16; gb += kGroupGemmBatch) {
#pragma unroll
      for (int gi = 0; gi < kGroupGemmBatch; ++gi) {
        const int grp = gb + gi;
        fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, int32_t> acc;
        fill_fragment(acc, 0);
        const int8_t* a_ptr = a_row0[sb].qs + grp * 16;
        const long a_ld = static_cast<long>(nsb) * static_cast<long>(sizeof(BlockQ8_K));
        load_matrix_sync(a_frag, a_ptr, a_ld);
        load_matrix_sync(b_frag, &w_stage[wy][grp][0][0], 16);
        mma_sync(acc, a_frag, b_frag, acc);
        store_matrix_sync(&raw_tile[wy][gi][0][0], acc, 16, mem_row_major);
      }
      __syncthreads();

#pragma unroll
      for (int gi = 0; gi < kGroupGemmBatch; ++gi) {
        const int grp = gb + gi;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const int cell = lane + c * 32;
          const int ii = cell / 16, jj = cell % 16;
          isum[c] += raw_tile[wy][gi][ii][jj] * static_cast<int32_t>(xb_cur[jj]->scales[grp]);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int cell = lane + c * 32;
      const int ii = cell / 16, jj = cell % 16;
      const float d = DF16ToF32(xb_cur[jj]->d) * a_row0[ii * nsb + sb].d;
      acc_f[c] += d * static_cast<float>(isum[c]);
    }
  }

#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const int cell = lane + c * 32;
    const int ii = cell / 16, jj = cell % 16;
    const int64_t oi = i0 + ii, oj = j0 + jj;
    if constexpr (sizeof(OutT) == 4) out[oi * n + oj] = acc_f[c];
    else out[oi * n + oj] = DF32ToBF16(acc_f[c]);
  }
#endif  // VT_ROCM_QUANT_WMMA_OK
}

// --- RDNA4 WMMA int8 tile arm of KQuantGemmK, Q4_K
// (KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4, issue #2109's `## Owed`). Same mechanism
// as `KQuantGemmKWmmaQ6K` above (same tile shape, same fill-once-per-
// superblock staging, same batch-of-`kGroupGemmBatch` store/fold), but Q4_K's
// layout differs from Q6_K's in two ways this kernel has to carry:
//
// 1. Q4_K's nibbles are UNSIGNED (0..15, no -32 recentering — DotQ4K's own
//    scalar reconstruction has none either, because the "DC offset" a
//    superblock's values sit at is corrected by the min term below rather
//    than by centering the nibble itself).
// 2. Q4_K's scale granularity is 32-wide (`kQK_K/32 == 8` sub-blocks), twice
//    the 16-wide WMMA K-tile, so `kGroupGemmBatch == 2` batches two K-tiles
//    per sub-block by CONSTRUCTION here, not by the batch-size tuning
//    `KQuantGemmKWmmaQ6K`'s spec table recorded — `qsb = gb / kGroupGemmBatch`
//    is exact for batch 2 and would silently stop being one sub-block if that
//    constant ever changed independently of this comment.
//
// Q4_K also carries a SECOND per-sub-block correction the scalar arm calls
// `dmin * sumi`: `sumi = sum_g bsums[g] * mins[g/2]` uses the ACTIVATION's
// already-quantized per-16-group sums (`BlockQ8_K::bsums`) against the
// WEIGHT's per-32-sub-block mins, and needs no tensor-core work at all — it
// is 16 integer multiply-adds per output cell, done directly in registers.
// `isum` (the tensor-core term) and `sumi` (the min term) are kept in
// SEPARATE integer accumulators for the same reason `KQuantGemmKWmmaQ6K`
// keeps one: each is exact on its own, and `d * isum - dmin * sumi` is the
// only float expression, applied once per superblock, matching `DotQ4K`
// verbatim.
#if defined(VT_ROCM_QUANT_WMMA_OK)
__device__ inline void DequantQ4KTile16(const BlockQ4_K* xb, int tile16, int8_t out[16]) {
  const int sb = tile16 >> 1;
  const int t = tile16 & 1;
  const uint8_t* q4b = xb->qs + (sb / 2) * 32;
  const int shift = (sb & 1) ? 4 : 0;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const int e = t * 16 + k;
    out[k] = static_cast<int8_t>((q4b[e] >> shift) & 0x0F);
  }
}

// The packed-scales unpack `DotQ4K`/`DotQ5K` both already do inline; factored
// out here only because this kernel needs it once per (row, superblock)
// rather than folded into a single-row dot product.
__device__ inline void UnpackQ4KScalesMins(const BlockQ4_K* xb, uint8_t scales[8],
                                           uint8_t mins[8]) {
  const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
  uint32_t utmp[4];
  memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* s = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mn = reinterpret_cast<const uint8_t*>(&utmp[2]);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    scales[i] = s[i];
    mins[i] = mn[i];
  }
}
#endif  // VT_ROCM_QUANT_WMMA_OK

template <typename OutT, int WarpsPerBlock>
__global__ void KQuantGemmKWmmaQ4K(OutT* __restrict__ out, const uint8_t* __restrict__ weight,
                                   const BlockQ8_K* __restrict__ act, int64_t m, int64_t n,
                                   int64_t nsb, int64_t n_tiles) {
#if !defined(VT_ROCM_QUANT_WMMA_OK)
  (void)out; (void)weight; (void)act; (void)m; (void)n; (void)nsb; (void)n_tiles;
  return;
#else
  using namespace rocwmma;
  const int64_t tile = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock +
                       static_cast<int64_t>(threadIdx.y);
  const int64_t m_tiles = m / 16;
  if (tile >= m_tiles * n_tiles) return;
  const int64_t it = tile / n_tiles;
  const int64_t jt = tile % n_tiles;
  const int64_t i0 = it * 16;
  const int64_t j0 = jt * 16;
  const int lane = static_cast<int>(threadIdx.x);
  const int wy = static_cast<int>(threadIdx.y);

  __shared__ int8_t w_stage[WarpsPerBlock][16][16][16];
  __shared__ int32_t raw_tile[WarpsPerBlock][kGroupGemmBatch][16][16];
  // Unpacked once per (row, superblock): 8 scales + 8 mins, tiny next to the
  // tile buffers above.
  __shared__ uint8_t row_scales[WarpsPerBlock][16][8];
  __shared__ uint8_t row_mins[WarpsPerBlock][16][8];

  const size_t row_stride = static_cast<size_t>(nsb) * sizeof(BlockQ4_K);
  const BlockQ4_K* w_row_base[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    w_row_base[r] =
        reinterpret_cast<const BlockQ4_K*>(weight + static_cast<size_t>(j0 + r) * row_stride);
  }
  const BlockQ8_K* a_row0 = act + i0 * nsb;

  float acc_f[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) acc_f[c] = 0.0f;

  for (int64_t sb = 0; sb < nsb; ++sb) {
    const BlockQ4_K* xb_cur[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) xb_cur[r] = w_row_base[r] + sb;

    if (lane < 16) UnpackQ4KScalesMins(xb_cur[lane], row_scales[wy][lane], row_mins[wy][lane]);

    int32_t isum[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) isum[c] = 0;

    for (int t = lane; t < 16 * 16; t += 32) {
      const int row = t / 16;
      const int tile16 = t % 16;
      DequantQ4KTile16(xb_cur[row], tile16, w_stage[wy][tile16][row]);
    }
    __syncthreads();

    for (int gb = 0; gb < 16; gb += kGroupGemmBatch) {
      const int qsb = gb / kGroupGemmBatch;  // one Q4_K sub-block of 32 (see header note)
#pragma unroll
      for (int gi = 0; gi < kGroupGemmBatch; ++gi) {
        const int grp = gb + gi;
        fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, int32_t> acc;
        fill_fragment(acc, 0);
        const int8_t* a_ptr = a_row0[sb].qs + grp * 16;
        const long a_ld = static_cast<long>(nsb) * static_cast<long>(sizeof(BlockQ8_K));
        load_matrix_sync(a_frag, a_ptr, a_ld);
        load_matrix_sync(b_frag, &w_stage[wy][grp][0][0], 16);
        mma_sync(acc, a_frag, b_frag, acc);
        store_matrix_sync(&raw_tile[wy][gi][0][0], acc, 16, mem_row_major);
      }
      __syncthreads();

#pragma unroll
      for (int gi = 0; gi < kGroupGemmBatch; ++gi) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const int cell = lane + c * 32;
          const int ii = cell / 16, jj = cell % 16;
          isum[c] += raw_tile[wy][gi][ii][jj] * static_cast<int32_t>(row_scales[wy][jj][qsb]);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int cell = lane + c * 32;
      const int ii = cell / 16, jj = cell % 16;
      int32_t sumi = 0;
#pragma unroll
      for (int g = 0; g < 16; ++g) {
        sumi += static_cast<int32_t>(a_row0[ii * nsb + sb].bsums[g]) *
                static_cast<int32_t>(row_mins[wy][jj][g / 2]);
      }
      const float d = DF16ToF32(xb_cur[jj]->d) * a_row0[ii * nsb + sb].d;
      const float dmin = DF16ToF32(xb_cur[jj]->dmin) * a_row0[ii * nsb + sb].d;
      acc_f[c] += d * static_cast<float>(isum[c]) - dmin * static_cast<float>(sumi);
    }
  }

#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const int cell = lane + c * 32;
    const int ii = cell / 16, jj = cell % 16;
    const int64_t oi = i0 + ii, oj = j0 + jj;
    if constexpr (sizeof(OutT) == 4) out[oi * n + oj] = acc_f[c];
    else out[oi * n + oj] = DF32ToBF16(acc_f[c]);
  }
#endif  // VT_ROCM_QUANT_WMMA_OK
}

// Per-device, once: launch the Q6_K WMMA tile only when this device's
// gcnArchName is gfx1200/gfx1201 (prefix match). NEVER gated by
// VT_ROCM_QUANT_WMMA_OK at a host call site — that macro is defined only on
// the device compilation pass for gfx1200/gfx1201 itself, so a host-side
// `#if` on it always compiles away the launch (issue #785's exact trap,
// documented in `.agents/specs/rocm-sharedk-wmma-host-dispatch.md`). Reuses
// the same HIP-free, unit-tested prefix check `rocm_paged_attn.hip`'s own
// WMMA dispatch already established (`vt::rocm::GcnArchNameIsGfx12PrefillWmma`,
// `include/vt/rocm/rocm_arch.h`); the name is attention-flavored but the test
// it names is pure architecture matching, shared rather than duplicated here.
bool Gfx12QuantWmmaHostOk(int device_index) {
  struct Slot {
    std::once_flag once;
    bool ok = false;
  };
  static std::array<Slot, 64> slots;
  if (device_index < 0 || static_cast<unsigned>(device_index) >= slots.size()) return false;
  std::call_once(slots[static_cast<unsigned>(device_index)].once, [device_index] {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_index) != hipSuccess) {
      slots[static_cast<unsigned>(device_index)].ok = false;
      return;
    }
    slots[static_cast<unsigned>(device_index)].ok =
        vt::rocm::GcnArchNameIsGfx12PrefillWmma(prop.gcnArchName);
  });
  return slots[static_cast<unsigned>(device_index)].ok;
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
//
// skip_m/skip_n: the WMMA tile-aligned corner [0,skip_m)x[0,skip_n) a caller
// already filled and this launch must not recompute (KERNEL-QUANT-CIQ-GEMM-
// ROCM-RDNA4's M/N tail arm). Every pre-existing call site passes 0,0 and
// takes the FULL-GRID branch below unchanged -- the tail arm is the only
// caller that passes a nonzero skip_m or skip_n.
//
// The tail-fill call sites used to launch this kernel over the entire m*n
// grid and rely on an `i < skip_m && j < skip_n` guard to no-op every warp
// landing inside the corner. That is correct but not sized to the work: for
// a large aligned corner and a wide N (e.g. N=12288, this row's own
// production benchmark shape), the no-op warps outnumber the real ones by
// roughly the corner's own area, so the launch and its address computation
// cost close to as much as the whole GEMM to fill in what is typically a
// thin remainder strip. This review finding (KERNEL-QUANT-CIQ-GEMM-ROCM,
// independent review round) replaces the skip guard with an index remap so
// the host can size the tail-fill grid to the remainder itself, not m*n:
// a launched index `t` enumerates ONLY the bottom strip [skip_m,m)x[0,n)
// (contiguous at the buffer's own row stride n, so it needs no separate
// stride parameter) followed by the right strip [0,skip_m)x[skip_n,n) --
// together the exact complement of the WMMA corner, and disjoint from each
// other by construction (the bottom strip already claims every row at or
// past skip_m; the right strip only visits rows before it).
template <typename OutT, int Fmt>
__global__ void KQuantGemmK(OutT* __restrict__ out, const uint8_t* __restrict__ weight,
                            const BlockQ8_K* __restrict__ act, int64_t m, int64_t n,
                            int64_t nsb, size_t w_row_bytes, size_t w_block_bytes,
                            int64_t skip_m, int64_t skip_n) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  int64_t i, j;
  if (skip_m == 0 && skip_n == 0) {
    // Full-grid path, byte-for-byte the launch every pre-existing call site
    // still takes: unchanged from before this review finding.
    if (t >= m * n) return;
    i = t / n;
    j = t % n;
  } else {
    // Tail-fill path: t enumerates only the remainder the WMMA corner left
    // untouched, bottom strip first then right strip (see the kernel-level
    // note above for the shapes and why they are disjoint).
    const int64_t bottom_strip = (m - skip_m) * n;
    const int64_t right_strip = skip_m * (n - skip_n);
    if (t >= bottom_strip + right_strip) return;
    if (t < bottom_strip) {
      i = skip_m + t / n;
      j = t % n;
    } else {
      const int64_t t2 = t - bottom_strip;
      i = t2 / (n - skip_n);
      j = skip_n + t2 % (n - skip_n);
    }
  }
  const int lane = threadIdx.x;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_K* a_row = act + i * nsb;
  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    if constexpr (Fmt == 3) {
      // Fmt 3 IS Fmt 2 (Q6_K) with a different PRIVATE-MEMORY footprint, and
      // nothing else. `DotQ6K` materialises the whole superblock into an
      // `int8_t aux8[256]` per thread, which the compiler places in SCRATCH;
      // `DotQ6KIsumRange` reconstructs 32 weights at a time into `int8_t
      // aux8[32]`, which fits in registers. The arithmetic is the same integer
      // accumulator times the same one f32 scale product, which is the
      // bit-equality this file's `DotQ6KIsumRange` note already argues and
      // which tests/vt/test_backend_cross_device.cpp already asserts between
      // the cooperative and single-warp arms.
      const BlockQ6_K* xb = static_cast<const BlockQ6_K*>(w_sb);
      const BlockQ8_K* yb = a_row + sb;
      partial += (DF16ToF32(xb->d) * yb->d) * DotQ6KIsumRange(xb, yb, 0, kSubBlocks);
    } else if constexpr (Fmt == 2) partial += DotQ6K(static_cast<const BlockQ6_K*>(w_sb), a_row + sb);
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
// (src/vt/cuda/cuda_quant_dot.cu:1812 `EnsureScratch`,
// src/vt/cuda/graph_safe_scratch.h; the :1507 this comment used to cite has
// moved): hipMalloc/hipFree/hipStreamSynchronize per call are ILLEGAL under
// hipGraph stream capture (the #473/#332 decode-graph lane), and the old
// free-after-launch also needed the sync that serialized every call. The pool
// allocates stream-ordered (hipMallocAsync), never frees during the process (a
// captured graph may have baked the pointer), and needs NO stream
// synchronization: reuse is stream-ordered and retirement keeps every baked
// pointer valid. Bounded: the buffer grows O(log(max/min)) times over a process.
//
// The bookkeeping is vt::GrowOnlyStreamScratch (src/vt/grow_only_stream_scratch.h),
// which is free of HIP headers so the HOST synchronization it owns is gated on a
// CPU-only runner (#2712). Only the hipMallocAsync stays here.
vt::GrowOnlyStreamScratch<hipStream_t>& QuantScratchPool() {
  static vt::GrowOnlyStreamScratch<hipStream_t> pool;
  return pool;
}

void* EnsureQuantScratch(size_t need, hipStream_t s) {
  return QuantScratchPool().Ensure(s, need, [s](size_t bytes) {
    void* block = nullptr;
    // Retire, never free (see the header note above). Throws on failure, which
    // leaves the pool entry exactly as it was.
    Check(hipMallocAsync(&block, bytes, s), "quant scratch grow");
    return block;
  });
}

enum class Q8KQuantArm : uint8_t { kLegacy = 0, kCandidate = 1 };
enum class Q8KProductionRoute : uint8_t { kDense = 0, kGrouped = 1 };
using Q8KArchResolver = std::string (*)(int) noexcept;

// The accepted real-checkpoint and profiler gates apply only to gfx1100.
// Explicit arms and every other architecture remain governed by the selector.
constexpr bool kQ8KGfx1100DefaultAccepted = true;

struct Q8KArchCacheEntry {
  int device_index;
  Q8KArchResolver resolve;
  std::string arch;
};

class Q8KArchCache {
 public:
  std::string Resolve(int device_index, Q8KArchResolver resolve) {
    for (const Q8KArchCacheEntry& entry : entries_) {
      if (entry.device_index == device_index && entry.resolve == resolve) return entry.arch;
    }
    const std::string arch = resolve == nullptr ? std::string{} : resolve(device_index);
    entries_.push_back(Q8KArchCacheEntry{device_index, resolve, arch});
    return arch;
  }

 private:
  std::vector<Q8KArchCacheEntry> entries_;
};

std::string ResolveQ8KArch(int device_index, Q8KArchResolver resolve) {
  // Match rocm_skinny_gemm_arch.h: cache each device and resolver on the
  // calling thread. The launch hot path takes no process-wide mutex.
  static thread_local Q8KArchCache cache;
  return cache.Resolve(device_index, resolve);
}

bool Q8KArchIsGfx1100(const std::string& arch) {
  constexpr char kStem[] = "gfx1100";
  constexpr size_t kStemLength = sizeof(kStem) - 1;
  if (arch.size() < kStemLength || arch.compare(0, kStemLength, kStem) != 0) return false;
  if (arch.size() == kStemLength) return true;
  const char next = arch[kStemLength];
  return next < '0' || next > '9';
}

Q8KQuantArm SelectQ8KQuantArm(const char* env_value, bool gfx1100_default_accepted,
                              int device_index, Q8KArchResolver resolve) {
  if (env_value != nullptr) {
    if (std::strcmp(env_value, "0") == 0) return Q8KQuantArm::kLegacy;
    if (std::strcmp(env_value, "1") == 0) return Q8KQuantArm::kCandidate;
    throw std::runtime_error(std::string("vt rocm: VT_ROCM_Q8K_BLOCK=") + env_value +
                             " must be 0 or 1");
  }
  const std::string arch = ResolveQ8KArch(device_index, resolve);
  return gfx1100_default_accepted && Q8KArchIsGfx1100(arch)
             ? Q8KQuantArm::kCandidate
             : Q8KQuantArm::kLegacy;
}

std::atomic<uint64_t> g_q8k_route_dispatches[2][2]{};
thread_local Q8KKernelWitnessCount* g_q8k_kernel_execution_witness_for_test = nullptr;

void LaunchQ8KQuantArm(BlockQ8_K* scratch, const void* act, ActDT dtype,
                       int64_t row_stride, int64_t rows, int64_t nsb, hipStream_t stream,
                       Q8KQuantArm arm, const Q8KProductionRoute* route,
                       Q8KKernelWitnessCount* kernel_witness) {
  if (route != nullptr) {
    g_q8k_route_dispatches[static_cast<size_t>(*route)][static_cast<size_t>(arm)].fetch_add(
        1, std::memory_order_relaxed);
  }
  if (arm == Q8KQuantArm::kCandidate) {
    const dim3 grid(static_cast<unsigned>(nsb), static_cast<unsigned>(rows), 1);
    QuantizeQ8KCooperativeK<<<grid, kQK_K, 0, stream>>>(
        scratch, act, dtype, row_stride, rows, nsb, kernel_witness);
    Check(hipGetLastError(), "q8_K cooperative quant");
    return;
  }
  constexpr int kThreads = 128;
  const int64_t blocks = (rows * nsb + kThreads - 1) / kThreads;
  QuantizeQ8KK<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      scratch, act, dtype, row_stride, rows, nsb, kernel_witness);
  Check(hipGetLastError(), "q8_K legacy quant");
}

void LaunchQ8KQuantizer(BlockQ8_K* scratch, const void* act, ActDT dtype,
                        int64_t row_stride, int64_t rows, int64_t nsb,
                        hipStream_t stream, Q8KProductionRoute route, int device_index) {
  const Q8KQuantArm arm = SelectQ8KQuantArm(std::getenv("VT_ROCM_Q8K_BLOCK"),
                                            kQ8KGfx1100DefaultAccepted, device_index,
                                            vt::rocm::DeviceArchName);
  // The ordinary path passes null. Its instrumentation cost is one pointer
  // argument and one kernel-leader null check, with no allocation, host copy,
  // synchronization, or atomic operation.
  LaunchQ8KQuantArm(scratch, act, dtype, row_stride, rows, nsb, stream, arm, &route,
                    g_q8k_kernel_execution_witness_for_test);
}

// How many dense K-quant dispatches took the cooperative arm. Host-side and
// relaxed: nothing reads it on the decode path, and the ROCm decode gate reads
// it to prove the call site below is REACHED rather than merely compiled
// (AGENTS.md "Nothing lands dead" — the arm is bit-identical to the single-warp
// one by construction, so no output comparison can witness the dispatch).
std::atomic<uint64_t> g_kq_coop_dispatches{0};

// How many dense Q6_K prefill dispatches took the RDNA4 WMMA tile arm
// (KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4, issue #2109). Same reason as the coop
// counter above: the arm is designed to be bit-identical to the scalar one,
// so the reachability gate needs a witness that does not depend on comparing
// output values.
std::atomic<uint64_t> g_kq_wmma_dispatches{0};

// Same witness, Q4_K's own call site. Kept as a separate counter rather than
// folded into the one above so a reviewer can prove EACH format's launch
// site is reached independently, not infer Q4_K's from Q6_K's.
std::atomic<uint64_t> g_kq_wmma_q4k_dispatches{0};

// VT_ROCM_QUANT_WMMA=0 forces the scalar arm even on gfx1200/gfx1201 — the
// A/B escape hatch every WMMA arm in this tree ships with
// (VT_ATTN_PREFILL_SHAREDK_WMMA is the paged-attention precedent). Default
// on: the gate below still requires the runtime architecture match and
// `m >= 16 && n >= 16` (not exact alignment — the M/N tail arm fills the
// rest) before this path is ever reached.
bool QuantWmmaEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ROCM_QUANT_WMMA");
    return e == nullptr || !(e[0] == '0' && e[1] == '\0');
  }();
  return on;
}

// VT_ROCM_Q6K_SMALL_PRIVATE=1 — the A/B arm for
// [#2511](https://github.com/mudler/vllm.cpp/issues/2511). Every failing leg on
// `gfx1151` ended inside `KQuantGemmK<OutT, 2>`, the Q6_K arm, with all three of
// its pointer arguments EXACTLY the size the launch needs (measured from
// `AMD_LOG_LEVEL=4`: weight `obj` == n * w_row_bytes to the byte, act `obj` ==
// m * nsb * sizeof(BlockQ8_K) to the byte) and every index provably inside them.
// The one thing that arm has and no other kernel on the path has is a 256-byte
// per-thread private array in `DotQ6K`, which lands in scratch. This selects the
// register-resident body instead, so a leg run with it on and a leg run with it
// off differ in private-memory footprint and in NOTHING ELSE.
//
// Default OFF: this is a diagnostic A/B until a leg count says which arm is
// right, and the range body measured 1.73x slower on the n=4096 k=12288 shape
// (see the `KQuantGemmKCoopQ6K` note above), so flipping the default is a
// separate decision with its own evidence.
bool Q6KSmallPrivateEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ROCM_Q6K_SMALL_PRIVATE");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

// VT_ROCM_QUANT_WMMA_WIDE=1 — the A/B arm for issue #3032, MEASURED AND
// REJECTED as a net regression (geomean -4.3%, `examples/quant-gemm-bench`,
// RX 9060 XT): see `kQuantWmmaWideWarpsPerBlock`'s own comment for the full
// result and why. Default OFF, permanently, and not expected to flip; kept
// as a ready-made A/B for re-checking this specific axis on different
// hardware or a future toolchain revision, the same posture
// `Q6KSmallPrivateEnabled` above ships with.
bool QuantWmmaWideBlockEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ROCM_QUANT_WMMA_WIDE");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

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

uint64_t KQuantWmmaDispatchCount() {
  return g_kq_wmma_dispatches.load(std::memory_order_relaxed);
}

uint64_t KQuantWmmaQ4KDispatchCount() {
  return g_kq_wmma_q4k_dispatches.load(std::memory_order_relaxed);
}

void Q8KQuantizeForTest(Queue& q, void* scratch, const void* act, DType dtype,
                        int64_t row_stride, int64_t rows, int64_t nsb, bool candidate) {
  EnsureQueueDevice(q);
  if (dtype != DType::kF32 && dtype != DType::kF16 && dtype != DType::kBF16) {
    throw std::runtime_error("vt rocm: Q8KQuantizeForTest requires f32, f16, or bf16");
  }
  LaunchQ8KQuantArm(static_cast<BlockQ8_K*>(scratch), act, ActDtOf(dtype), row_stride, rows,
                    nsb, static_cast<hipStream_t>(q.handle),
                    candidate ? Q8KQuantArm::kCandidate : Q8KQuantArm::kLegacy, nullptr,
                    nullptr);
}

bool Q8KCandidateSelectedForTest(const char* env_value, bool gfx1100_default_accepted,
                                 int device_index,
                                 std::string (*resolve)(int) noexcept) {
  return SelectQ8KQuantArm(env_value, gfx1100_default_accepted, device_index, resolve) ==
         Q8KQuantArm::kCandidate;
}

void Q8KResetRouteDispatchCountsForTest() {
  for (auto& route : g_q8k_route_dispatches)
    for (std::atomic<uint64_t>& count : route) count.store(0, std::memory_order_relaxed);
}

uint64_t Q8KRouteDispatchCountForTest(bool grouped, bool candidate) {
  return g_q8k_route_dispatches[grouped ? 1 : 0][candidate ? 1 : 0].load(
      std::memory_order_relaxed);
}

void* Q8KSetKernelExecutionWitnessForTest(void* device_counts) {
  Q8KKernelWitnessCount* previous = g_q8k_kernel_execution_witness_for_test;
  g_q8k_kernel_execution_witness_for_test =
      static_cast<Q8KKernelWitnessCount*>(device_counts);
  return previous;
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
    LaunchQ8KQuantizer(qact, a.data, ActDtOf(a.dtype), a.stride[0], m, nsb, s,
                       Q8KProductionRoute::kDense, q.device.index);
    const int64_t grid = (m * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int fmt = b.dtype == DType::kQ6_K ? 2 : b.dtype == DType::kQ5_K ? 1 : 0;
    const int coop = KQuantDecodeCoopWarps(b.dtype, m, nsb);
    // RDNA4 WMMA tile arm: Q6_K and Q4_K prefill only (m > 1, since
    // KQuantDecodeCoopWarps above already owns m == 1). Q5_K is `## Owed` —
    // same 32-wide scale/min shape as Q4_K plus a high-bit plane, not yet
    // ported. Every other format and every other architecture falls through
    // to the arms below unchanged.
    //
    // M/N TAIL (KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4's `## Owed`, closed here):
    // `m >= 16 && n >= 16` is the gate, not exact alignment — the WMMA
    // kernels already floor internally (`m_tiles = m / 16`, and `n_tiles`
    // is passed in as `n / 16`), so a non-16-multiple shape still gets the
    // WMMA arm over its floor(m/16)*16 x floor(n/16)*16 corner. What was
    // missing was the REST of the output: a real prompt is essentially never
    // 16-aligned (this row's own real-model gate (c) measurement needed a
    // prompt hand-trimmed to exactly 288/512 tokens to engage the WMMA arm
    // at all), so requiring exact alignment meant the common case fell back
    // to scalar for the WHOLE shape, not just the remainder. The scalar
    // kernel's `skip_m`/`skip_n`-indexed tail fills in that remainder --
    // bottom rows [m_aligned, m) x all columns, and the right columns
    // [0, m_aligned) x [n_aligned, n) -- without recomputing the corner
    // WMMA already wrote. The tail-fill grid below is sized to that
    // remainder (`grid_tail`), not to the full m*n grid `grid` above: an
    // independent review found the original full-grid launch (relying on a
    // skip guard to no-op every warp inside the corner) cost close to as
    // much as the whole GEMM to fill a typically thin remainder strip, for
    // a large aligned corner and wide N.
    const bool wmma_ok = (fmt == 2 || fmt == 0) && m >= 16 && n >= 16 &&
                        QuantWmmaEnabled() && Gfx12QuantWmmaHostOk(q.device.index);
    auto launch = [&](auto ot) {
      using OutT = decltype(ot);
      auto* o = static_cast<OutT*>(out.data);
      if (wmma_ok && fmt == 2) {
        g_kq_wmma_dispatches.fetch_add(1, std::memory_order_relaxed);
        const int64_t n_tiles = n / 16;
        const int64_t m_tiles = m / 16;
        const int wpb = QuantWmmaWideBlockEnabled() ? kQuantWmmaWideWarpsPerBlock
                                                    : kQuantWmmaWarpsPerBlock;
        const int64_t grid_wmma = (m_tiles * n_tiles + wpb - 1) / wpb;
        if (wpb == kQuantWmmaWideWarpsPerBlock) {
          KQuantGemmKWmmaQ6K<OutT, kQuantWmmaWideWarpsPerBlock>
              <<<static_cast<unsigned>(grid_wmma), dim3(32, kQuantWmmaWideWarpsPerBlock), 0, s>>>(
                  o, w, qact, m, n, nsb, n_tiles);
        } else {
          KQuantGemmKWmmaQ6K<OutT, kQuantWmmaWarpsPerBlock>
              <<<static_cast<unsigned>(grid_wmma), dim3(32, kQuantWmmaWarpsPerBlock), 0, s>>>(
                  o, w, qact, m, n, nsb, n_tiles);
        }
        const int64_t m_aligned = m_tiles * 16, n_aligned = n_tiles * 16;
        if (m_aligned < m || n_aligned < n) {
          // Remainder cell count: the bottom strip [m_aligned,m)x[0,n) plus
          // the right strip [0,m_aligned)x[n_aligned,n) -- see KQuantGemmK's
          // own comment for why these two are exactly the WMMA corner's
          // complement. Sized to the remainder, not m*n.
          const int64_t remainder = (m - m_aligned) * n + m_aligned * (n - n_aligned);
          const int64_t grid_tail = (remainder + kWarpsPerBlock - 1) / kWarpsPerBlock;
          KQuantGemmK<OutT, 2><<<static_cast<unsigned>(grid_tail), block, 0, s>>>(
              o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes, m_aligned, n_aligned);
        }
        return;
      }
      if (wmma_ok && fmt == 0) {
        g_kq_wmma_q4k_dispatches.fetch_add(1, std::memory_order_relaxed);
        const int64_t n_tiles = n / 16;
        const int64_t m_tiles = m / 16;
        const int wpb = QuantWmmaWideBlockEnabled() ? kQuantWmmaWideWarpsPerBlock
                                                    : kQuantWmmaWarpsPerBlock;
        const int64_t grid_wmma = (m_tiles * n_tiles + wpb - 1) / wpb;
        if (wpb == kQuantWmmaWideWarpsPerBlock) {
          KQuantGemmKWmmaQ4K<OutT, kQuantWmmaWideWarpsPerBlock>
              <<<static_cast<unsigned>(grid_wmma), dim3(32, kQuantWmmaWideWarpsPerBlock), 0, s>>>(
                  o, w, qact, m, n, nsb, n_tiles);
        } else {
          KQuantGemmKWmmaQ4K<OutT, kQuantWmmaWarpsPerBlock>
              <<<static_cast<unsigned>(grid_wmma), dim3(32, kQuantWmmaWarpsPerBlock), 0, s>>>(
                  o, w, qact, m, n, nsb, n_tiles);
        }
        const int64_t m_aligned = m_tiles * 16, n_aligned = n_tiles * 16;
        if (m_aligned < m || n_aligned < n) {
          const int64_t remainder = (m - m_aligned) * n + m_aligned * (n - n_aligned);
          const int64_t grid_tail = (remainder + kWarpsPerBlock - 1) / kWarpsPerBlock;
          KQuantGemmK<OutT, 0><<<static_cast<unsigned>(grid_tail), block, 0, s>>>(
              o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes, m_aligned, n_aligned);
        }
        return;
      }
      if (coop == 8 && fmt == 2) {
        // One block per output element, eight warps deep. Only the measured
        // width is instantiated; anything else falls through to the single-warp
        // arm rather than silently launching a shape nothing was measured on.
        g_kq_coop_dispatches.fetch_add(1, std::memory_order_relaxed);
        KQuantGemmKCoopQ6K<OutT, 8><<<static_cast<unsigned>(m * n), dim3(32, 8), 0, s>>>(
            o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes);
        return;
      }
      if (fmt == 2 && Q6KSmallPrivateEnabled()) KQuantGemmK<OutT, 3><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes, 0, 0);
      else if (fmt == 2) KQuantGemmK<OutT, 2><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes, 0, 0);
      else if (fmt == 1) KQuantGemmK<OutT, 1><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes, 0, 0);
      else KQuantGemmK<OutT, 0><<<static_cast<unsigned>(grid), block, 0, s>>>(o, w, qact, m, n, nsb, w_row_bytes, w_block_bytes, 0, 0);
    };
    if (out.dtype == DType::kF32) launch(float{}); else launch(uint16_t{});
    Check(hipGetLastError(), "K-quant gemm");
    return;
  }
  throw std::runtime_error("vt rocm: matmul_bt_quant: unsupported weight dtype (ported: Q8_0/Q4_K/Q5_K/Q6_K; owed: Q4_0/Q2_K/Q3_K/IQ2_XXS/IQ3_XXS/IQ3_S/IQ2_S/MXFP4 -- the loader pre-filters to the ported set, so reaching here is a bug)");
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
    LaunchQ8KQuantizer(qact, act.data, ActDtOf(act.dtype), act.stride[0], Pa, nsb, s,
                       Q8KProductionRoute::kGrouped, q.device.index);
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
