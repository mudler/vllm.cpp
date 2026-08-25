// EXL3 (exllamav3 trellis) device kernels, CUDA arm — MODEL-DSV4-EXL3 W2a/W2b.
//
// PORTED 1:1 FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT).
// vLLM implements no EXL3 at the parity pin, so exllamav3 is this format's
// secondary oracle (AGENTS.md "When vLLM has no implementation"); see
// `.agents/specs/model-dsv4-exl3.md`, whose `## W2 design` fixes the parity
// contract, the output dtype and the shape policy this file obeys.
//
// The files this is a port of, and what each contributed:
//   exllamav3_ext/ptx.cuh:6-16,52-74,103-139,159-212  fragments, mma, the
//       global barrier, cp.async and ldmatrix
//   exllamav3_ext/quant/codebook.cuh:44-54,92-116     the MCG (cb 1) codebook
//   exllamav3_ext/quant/exl3_dq.cuh:5-13,96-161,254-293  the tail-biting window
//       read and dq_dispatch
//   exllamav3_ext/quant/hadamard_inner.cuh:17-44,93-279  shuffle_had_f4x32 and
//       the hf / ff / fh inners
//   exllamav3_ext/quant/exl3_gemm_inner.cuh:22-733    the pipelined tile loop
//   exllamav3_ext/quant/exl3_gemm_kernel.cuh:8-80     the fused had -> gemm ->
//       had grid-cooperative kernel
//   exllamav3_ext/quant/exl3_gemm.cu:110-309          the launcher
//   exllamav3_ext/quant/exl3_devctx.cu:59-70          the per-device lock buffer
//
// DELIBERATE DIVERGENCES FROM THE UPSTREAM SOURCE, all recorded:
//   * The torch/ATen layer is gone. Tensors arrive as `vt::Tensor`, the stream
//     comes from the `vt::Queue`, and every TORCH_CHECK is a VT_CHECK in
//     src/vt/ops.cpp, so the refusals are shared with the CPU arm rather than
//     duplicated per backend.
//   * `register` is dropped from the fragment declarations. The keyword is
//     REMOVED in C++17 and this tree compiles CUDA at -std=c++20.
//   * The shape table lives in src/vt/exl3_policy.cpp as pure host code, so it
//     is gated on a machine with no GPU. This file calls it; it does not carry
//     a second copy.
//   * Only bits == 3 and codebook == 1 (mcg) are INSTANTIATED. That is the whole
//     of the DeepSeek-V4-Flash 3.0bpw artifact and it keeps 8 template
//     instantiations rather than 64 in a translation unit the fat build compiles
//     for ten architectures. Every other (bits, codebook) refuses BY NAME from
//     the launcher and is recorded owed in the row's spec; the CPU arm stays
//     generic over bits, so the reference is not narrowed with the kernel.
//   * The grid sync is a hand-rolled sense-reversing barrier rather than
//     `cooperative_groups`. NOT a preference: `grid_group::sync()` lowers to a
//     `cudadevrt` call and therefore requires `-rdc=true` device linking, which
//     this tree sets nowhere (no target carries CUDA_SEPARABLE_COMPILATION and
//     nothing else under src/vt/cuda/ includes the cooperative-groups header).
//     Upstream itself replaces the same sync with a hand-rolled barrier on newer
//     architectures (`ptx.cuh:319-348` `group_barrier`, taken by
//     `exl3_gemm_kernel.cuh:166-170`), so this is upstream's own primitive and
//     not an invention. `cudaLaunchCooperativeKernel` is still what makes the
//     barrier legal: it is what guarantees every block is CO-RESIDENT, and a
//     spin barrier over blocks that are not co-resident deadlocks.
//   * The autotuner (`coop_autotune.cuh`), the m<=8 GEMV (`exl3_gemv.cu`, W2c)
//     and the fused MoE mgemm (`exl3_moe.cu`, W2d) are NOT ported in this wave.
//     Shape selection therefore always goes through the table.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <mutex>
#include <set>
#include <utility>
#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/cuda/cuda_device_caps.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda exl3: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

#define EXL3_MIN(a, b) ((a) < (b) ? (a) : (b))
#define EXL3_MAX(a, b) ((a) > (b) ? (a) : (b))
#define EXL3_CEIL_DIVIDE(a, b) (((a) + (b) - 1) / (b))

// exl3_gemm_inner.cuh:6-7. 90 KiB is the largest dynamic shared-memory opt-in
// that every architecture in this tree's fat build accepts (sm_86 caps at
// 99 KiB, sm_89 at 100 KiB, GB10 reports 101376).
constexpr int kBaseThreads = 256;
constexpr int kSmemMax = 90 * 1024;

// exl3_devctx.cuh:7-9. The per-device lock buffer is indexed by output tile
// column, and one int per 16-wide column block covers any n this tree loads. Two
// more ints past the end are the grid barrier's counter and sense, which is
// where upstream's BARRIER_LOCKS_OFFSET puts its own.
constexpr size_t kMaxTilesC = 1024 * 1024;
constexpr size_t kLockInts = kMaxTilesC + 2;

// ── compat types (exllamav3_ext/compat.cuh, util.cuh) ────────────────────────

struct __align__(8) half4 {
  half2 x;
  half2 y;
};

union half2_uint32 {
  uint32_t as_uint32;
  half2 as_half2;
  __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
};

// ── fragments and PTX (ptx.cuh) ──────────────────────────────────────────────

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;

// ptx.cuh:52-74. FP16 @ FP16 + FP32 -> FP32.
__device__ inline void ptx_mma_m16n8k16(const FragA& frag_a, const FragB& frag_b,
                                        FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  const float* d = reinterpret_cast<const float*>(&frag_c);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(d[0]), "f"(d[1]),
        "f"(d[2]), "f"(d[3]));
}

// ptx.cuh:103-139. The split-K column barrier: threadblocks in one output column
// hand the partial sum down in reverse k order, and the LAST one resets the lock.
__device__ inline void barrier_acquire(int* lock, int stage) {
  if (threadIdx.x == 0) {
    volatile int state = -1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    } while (state != stage);
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock, int val, bool reset) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      *lock = 0;
      return;
    }
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

// The whole-grid barrier, in the shape of upstream's `group_barrier`
// (ptx.cuh:319-348) with `cuda::atomic_ref` spelled as the plain atomics +
// `__threadfence()` this tree already uses, and specialised to ONE group (every
// block). `cs[0]` is the arrival counter and `cs[1]` the sense; both live in the
// zero-initialised per-device lock buffer and the barrier restores the counter
// to 0, so successive launches need no re-zeroing.
//
// CORRECTNESS RESTS ON CO-RESIDENCY. Every block spins until the last one
// arrives, which only terminates because `cudaLaunchCooperativeKernel` refuses a
// grid that does not fit on the device at once. No block can lap another:
// round N+1's flip needs every block to have arrived in round N+1, which needs
// each of them to have left round N first.
__device__ inline void grid_barrier(int* cs, int group_size) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile int* sense = cs + 1;
    const int old_sense = *sense;
    __threadfence();  // publish this block's writes before announcing arrival
    const int old = atomicAdd(cs, 1);
    if (old == group_size - 1) {
      atomicExch(cs, 0);
      __threadfence();
      *sense = 1 - old_sense;  // release: everyone may proceed
    } else {
      while (*sense == old_sense) __nanosleep(32);
      __threadfence();  // acquire: see what the other blocks wrote
    }
  }
  __syncthreads();
}

// ptx.cuh:159-212.
__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr) {
  const int bytes = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   cp.async.cg.shared.global [%0], [%1], %2;\n"
               "}\n" ::"r"(smem),
               "l"(glob_ptr), "n"(bytes));
}

__device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// ── the MCG codebook, cb == 1 (codebook.cuh:44-54,109-116) ───────────────────
//
// `lop3.b32 ... 0x6a` is `(a & b) ^ c`, so the three instructions are
// `x *= 0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60;` and the two fp16 halves
// are then summed in fp16. W1a's host `Exl3DecodeMcg` is the same three, and
// tier 1 of the parity contract requires the two to agree bit for bit.
__device__ inline half2 decode_mcg_product_2(uint32_t x0, uint32_t x1) {
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
  half2_uint32 xu0(x0);
  half2_uint32 xu1(x1);
  half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
  half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
  return __hadd2(d0, d1);
}

template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1) {
  static_assert(cb == 1, "MODEL-DSV4-EXL3 W2 instantiates the mcg codebook only");
  x0 *= 0xCBAC1FEDu;
  x1 *= 0xCBAC1FEDu;
  return decode_mcg_product_2(x0, x1);
}

// ── the tail-biting window read (exl3_dq.cuh) ────────────────────────────────

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift) {
  uint64_t merged = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
  return static_cast<uint32_t>(merged >> shift);
}

// exl3_dq.cuh:96-161. `align` is how many of the eight consecutive codewords
// share one funnel shift; for bits == 3 upstream picks 4 (`:267`).
template <int bits, int cb, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0,
                                    FragB& frag1) {
  int b1 = (t_offset + 257) * bits;
  int b0 = b1 - 16;
  int b2 = b1 + bits * 7;
  int i0 = b0 / 32;
  int i2 = (b2 - 1) / 32;
  int s2 = (i2 + 1) * 32 - b2;

  uint32_t a = ptr[i0 % (bits * 256 / 32)];
  uint32_t b = ptr[i2 % (bits * 256 / 32)];
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  if constexpr (align == 4) {
    w7 = fshift(b, a, s2);
    w6 = w7 >> bits;
    w5 = w6 >> bits;
    w4 = w5 >> bits;
    w3 = fshift(b, a, s2 + bits * 4);
    w2 = w3 >> bits;
    w1 = w2 >> bits;
    w0 = w1 >> bits;
  } else {
    w7 = fshift(b, a, s2);
    w6 = fshift(b, a, s2 + bits);
    w5 = fshift(b, a, s2 + bits * 2);
    w4 = fshift(b, a, s2 + bits * 3);
    w3 = fshift(b, a, s2 + bits * 4);
    w2 = fshift(b, a, s2 + bits * 5);
    w1 = fshift(b, a, s2 + bits * 6);
    w0 = fshift(b, a, s2 + bits * 7);
  }
  frag0[0] = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
  frag0[1] = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
  frag1[0] = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
  frag1[1] = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
}

// exl3_dq.cuh:254-293, narrowed to the instantiated arm. bits == 3 takes
// dq8<3, cb, 4>; every other width is refused at the launcher, not here, so the
// refusal names the row and the missing instantiation.
template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0,
                                            FragB& frag1) {
  static_assert(bits == 3, "MODEL-DSV4-EXL3 W2 instantiates bits == 3 only");
  dq8<bits, cb, 4>(ptr, idx, frag0, frag1);
}

// ── the Hadamard inners (hadamard_inner.cuh) ─────────────────────────────────

// hadamard_inner.cuh:17-44. Levels 4..64 of the 128-point transform, carried by
// five xor-partner shuffles. The sign flip is an XOR of the f32 SIGN BIT, which
// is what src/vt/cpu/cpu_exl3_kernels.cpp reproduces so the two arms are byte
// equal (spec `## W2 design` §1 tier 2).
__device__ inline void shuffle_had_f4x32(float& h0, float& h1, float& h2, float& h3,
                                         const int lane_id) {
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    uint32_t i0 = __float_as_uint(h0);
    uint32_t i1 = __float_as_uint(h1);
    uint32_t i2 = __float_as_uint(h2);
    uint32_t i3 = __float_as_uint(h3);
    uint64_t h01 = static_cast<uint64_t>(i0) | (static_cast<uint64_t>(i1) << 32);
    uint64_t h23 = static_cast<uint64_t>(i2) | (static_cast<uint64_t>(i3) << 32);
    uint64_t ph01 = __shfl_xor_sync(0xffffffff, h01, i);
    uint64_t ph23 = __shfl_xor_sync(0xffffffff, h23, i);
    float ph0 = __uint_as_float(static_cast<uint32_t>(ph01 & 0xffffffff));
    float ph1 = __uint_as_float(static_cast<uint32_t>(ph01 >> 32));
    float ph2 = __uint_as_float(static_cast<uint32_t>(ph23 & 0xffffffff));
    float ph3 = __uint_as_float(static_cast<uint32_t>(ph23 >> 32));
    int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
    i0 ^= static_cast<uint32_t>(sfm) & 0x80000000u;
    i1 ^= static_cast<uint32_t>(sfm) & 0x80000000u;
    i2 ^= static_cast<uint32_t>(sfm) & 0x80000000u;
    i3 ^= static_cast<uint32_t>(sfm) & 0x80000000u;
    h0 = __uint_as_float(i0) + ph0;
    h1 = __uint_as_float(i1) + ph1;
    h2 = __uint_as_float(i2) + ph2;
    h3 = __uint_as_float(i3) + ph3;
  }
}

// hadamard_inner.cuh:93-147. Half vector, half scales, half out.
template <bool pre_scale, bool post_scale>
__device__ inline void had_hf_r_128_inner(const half* __restrict__ input_ptr,
                                          half* __restrict__ output_ptr,
                                          const half* __restrict__ scale, const float r_scale,
                                          const int scale_block) {
  (void)scale;
  (void)scale_block;
  int t = threadIdx.x & 31;
  half4 v = reinterpret_cast<const half4*>(input_ptr)[t];
  if constexpr (pre_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);
  }
  float v0 = __half2float(__low2half(v.x));
  float v1 = __half2float(__high2half(v.x));
  float v2 = __half2float(__low2half(v.y));
  float v3 = __half2float(__high2half(v.y));
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  shuffle_had_f4x32(h0, h1, h2, h3, t);
  v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
  v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
  if constexpr (post_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);
  }
  reinterpret_cast<half4*>(output_ptr)[t] = v;
}

// hadamard_inner.cuh:151-212. Float vector, half scales, float out.
template <bool pre_scale, bool post_scale>
__device__ inline void had_ff_r_128_inner(const float* __restrict__ input_ptr,
                                          float* __restrict__ output_ptr,
                                          const half* __restrict__ scale, const float r_scale,
                                          const int scale_block) {
  (void)scale;
  (void)scale_block;
  int t = threadIdx.x & 31;
  float4 v = reinterpret_cast<const float4*>(input_ptr)[t];
  if constexpr (pre_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);
  }
  float v0 = v.x;
  float v1 = v.y;
  float v2 = v.z;
  float v3 = v.w;
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  // DEVIATION, recorded and DELIBERATE: upstream's float inner runs two
  // shuffle_had_f2x32 passes over (x,y) and (z,w) (hadamard_inner.cuh:192-193)
  // while the half inner runs one shuffle_had_f4x32 over all four. The two are
  // the SAME butterfly in the same order — f2x32 packs two values per shuffle,
  // f4x32 four — so using f4x32 here is byte-identical and keeps ONE
  // implementation of the level-4..64 stage for a byte gate to point at.
  shuffle_had_f4x32(h0, h1, h2, h3, t);
  v.x = h0 * r_scale;
  v.y = h1 * r_scale;
  v.z = h2 * r_scale;
  v.w = h3 * r_scale;
  if constexpr (post_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);
  }
  reinterpret_cast<float4*>(output_ptr)[t] = v;
}

// hadamard_inner.cuh:216-279. Float vector, half scales, HALF out — the arm the
// GEMM's fp16 destination takes, where the post-scale applies AFTER the round.
template <bool pre_scale, bool post_scale>
__device__ inline void had_fh_r_128_inner(const float* __restrict__ input_ptr,
                                          half* __restrict__ output_ptr,
                                          const half* __restrict__ scale, const float r_scale,
                                          const int scale_block) {
  (void)scale;
  (void)scale_block;
  int t = threadIdx.x & 31;
  float4 v = reinterpret_cast<const float4*>(input_ptr)[t];
  if constexpr (pre_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);
  }
  float v0 = v.x;
  float v1 = v.y;
  float v2 = v.z;
  float v3 = v.w;
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  shuffle_had_f4x32(h0, h1, h2, h3, t);
  half4 o;
  o.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
  o.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
  if constexpr (post_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    o.x = __hmul2(o.x, scales.x);
    o.y = __hmul2(o.y, scales.y);
  }
  reinterpret_cast<half4*>(output_ptr)[t] = o;
}

// ── the standalone had_r_128 kernels (hadamard.cu:9-37) ──────────────────────

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32) void had_hf_r_128_kernel(const half* __restrict__ input_ptr,
                                                          half* __restrict__ output_ptr,
                                                          const half* __restrict__ scale,
                                                          const float r_scale) {
  const size_t off = static_cast<size_t>(gridDim.y) * 128 * blockIdx.x + blockIdx.y * 128;
  had_hf_r_128_inner<pre_scale, post_scale>(input_ptr + off, output_ptr + off, scale, r_scale,
                                            static_cast<int>(blockIdx.y));
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32) void had_ff_r_128_kernel(const float* __restrict__ input_ptr,
                                                          float* __restrict__ output_ptr,
                                                          const half* __restrict__ scale,
                                                          const float r_scale) {
  const size_t off = static_cast<size_t>(gridDim.y) * 128 * blockIdx.x + blockIdx.y * 128;
  had_ff_r_128_inner<pre_scale, post_scale>(input_ptr + off, output_ptr + off, scale, r_scale,
                                            static_cast<int>(blockIdx.y));
}

// ── the pipelined tile loop (exl3_gemm_inner.cuh:22-733) ─────────────────────

template <int bits, bool c_fp32, int cb, int TILESIZE_M, int TILESIZE_K, int TILESIZE_N,
          int SH_STAGES, int FRAG_STAGES>
__device__ inline void exl3_gemm_kernel_inner(const half* __restrict__ A,
                                              const uint16_t* __restrict__ B,
                                              void* __restrict__ C, const int size_m,
                                              const int size_k, const int size_n,
                                              int* __restrict__ locks,
                                              const half* post_scale) {
  const int TILEBLOCKS_M = TILESIZE_M / 16;
  const int TILEBLOCKS_K = TILESIZE_K / 16;
  const int TILEBLOCKS_N = TILESIZE_N / 16;
  const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (kBaseThreads / 32);

  const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                        // in halfs
  const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;  // in uint16s
  const int sh_c_size = EXL3_MAX(4 * kBaseThreads * FRAGS_N_PER_WARP, TILESIZE_N * TILESIZE_M);

  const int A_COLS = TILESIZE_K / 8;
  const int A_SWIZZLE_MASK = A_COLS - 1;
  const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

  static_assert(kBaseThreads == 256);
  // Upstream carries FRAG_STAGES 1..5; the four shipped shapes use 5 (shape 1)
  // and 3 (shapes 2-4), so only those two ladders are transcribed. Without this
  // assert a shape asking for 2 or 4 would compile to a kernel whose main loop
  // is EMPTY — a silent no-op rather than a build error.
  static_assert(FRAG_STAGES == 1 || FRAG_STAGES == 3 || FRAG_STAGES == 5,
                "MODEL-DSV4-EXL3 W2 transcribes the FRAG_STAGES 1/3/5 ladders only");
  static_assert(TILESIZE_M == 16, "Invalid kernel params");
  static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
  static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
  static_assert(kSmemMax >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) +
                                4 * sh_c_size,
                "Invalid kernel params (insufficient shared memory for shape)");

  extern __shared__ half shared[];
  half* sh_a = shared;
  uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + SH_STAGES * sh_a_stage_size);
  float* sh_c = reinterpret_cast<float*>(sh_b + sh_b_stage_size * SH_STAGES);

  int t = threadIdx.x % kBaseThreads;
  int sub_k = threadIdx.x / kBaseThreads;
  int warp_id = t / 32;
  int lane_id = t % 32;

  int tiles_k = size_k / TILESIZE_K;
  int tiles_n = size_n / TILESIZE_N;
  int blocks_n = tiles_n * TILEBLOCKS_N;

  int num_slices = gridDim.x;
  int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
  int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
  int slice_len = slice_end - slice_beg;
  if (slice_len < 1) return;

  auto index_k = [&](int slice_i) { return (slice_i % tiles_k); };
  auto index_n = [&](int slice_i) { return (slice_i / tiles_k); };

  const int slice_m = 0;

  int slice0_k = index_k(slice_beg);
  int slice0_n = index_n(slice_beg);
  int slice0_iters = slice_len;

  int gl_a_stride_m = TILESIZE_M * size_k;
  const int gl_a_stride_k = TILESIZE_K;
  const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
  const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
  half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

  const int load_a_iters = EXL3_CEIL_DIVIDE(sh0_a_stride_m / 8, kBaseThreads);
  bool pred_a_gl[load_a_iters];
  int load_a_gl[load_a_iters];
  int load_a_sh[load_a_iters];
  for (int i = 0; i < load_a_iters; ++i) {
    int k = (i * kBaseThreads + t) % (gl_a_stride_k / 8);
    int m = (i * kBaseThreads + t) / (gl_a_stride_k / 8);
    load_a_gl[i] = m * size_k / 8 + k;
    load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
    pred_a_gl[i] = m < size_m;
  }

  int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
  const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
  const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
  const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
  uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

  const int load_b_iters = EXL3_CEIL_DIVIDE(sh0_b_stride_k / 8, kBaseThreads);
  bool pred_b_gl[load_b_iters];
  int load_b_gl[load_b_iters];
  for (int i = 0; i < load_b_iters; ++i) {
    int n = (i * kBaseThreads + t) % (gl_b_stride_n / 8);
    int k = (i * kBaseThreads + t) / (gl_b_stride_n / 8);
    load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
    pred_b_gl[i] = i * kBaseThreads + t < sh0_b_stride_k / 8;
  }

  auto advance0 = [&]() {
    slice0_k++;
    slice0_iters--;
    int stage = slice0_iters % SH_STAGES;
    sh0_a_ptr = sh_a + stage * sh_a_stage_size;
    sh0_b_ptr = sh_b + stage * sh_b_stage_size;
    if (slice0_k >= tiles_k) {
      slice0_k = 0;
      slice0_n++;
      gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
      gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    } else {
      gl_a_ptr += gl_a_stride_k;
      gl_b_ptr += gl_b_stride_k;
    }
  };

  int slice1_k = slice0_k;
  int slice1_iters = slice0_iters;
  half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
  uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

  auto advance1 = [&]() {
    slice1_k++;
    slice1_iters--;
    int stage = slice1_iters % SH_STAGES;
    sh1_a_ptr = sh_a + stage * sh_a_stage_size;
    sh1_b_ptr = sh_b + stage * sh_b_stage_size;
    if (slice1_k >= tiles_k) slice1_k = 0;
  };

  int slice2_k = slice0_k;
  int slice2_k0 = slice0_k;
  int slice2_n = slice0_n;
  int slice2_iters = slice0_iters;

  int gl_c_stride_n = TILESIZE_N;
  int gl_c_stride_m = TILESIZE_M * size_n;

  half* gl_c_ptr_16 = static_cast<half*>(C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
  float* gl_c_ptr_32 =
      static_cast<float*>(C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

  FragA frag_a[FRAG_STAGES];
  FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
  FragC frag_c[FRAGS_N_PER_WARP];

  auto advance2 = [&]() {
    slice2_k++;
    slice2_iters--;
    if (slice2_k >= tiles_k) {
      slice2_k = 0;
      slice2_k0 = 0;
      slice2_n++;
      if constexpr (c_fp32)
        gl_c_ptr_32 += gl_c_stride_n;
      else
        gl_c_ptr_16 += gl_c_stride_n;
    }
  };

  auto async_load_gl = [&]() {
    if (sub_k) {
      cp_async_fence();
      return;
    }
    if (slice0_iters) {
      {
        const int4* gl = reinterpret_cast<const int4*>(gl_a_ptr);
        int4* sh = reinterpret_cast<int4*>(sh0_a_ptr);
#pragma unroll
        for (int i = 0; i < load_a_iters; ++i)
          if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
      }
      {
        const int4* gl = reinterpret_cast<const int4*>(gl_b_ptr);
        int4* sh = reinterpret_cast<int4*>(sh0_b_ptr);
#pragma unroll
        for (int i = 0; i < load_b_iters; ++i)
          if (pred_b_gl[i]) cp_async(sh + kBaseThreads * i + t, gl + load_b_gl[i]);
      }
      advance0();
    }
    cp_async_fence();
  };

  auto load_frags = [&](int buf) {
    if (!slice1_iters) return;
    {
      int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
      int base_c = lane_id / 16 + sub_k * 2;
#pragma unroll
      for (int m = 0; m < TILEBLOCKS_M; ++m) {
        int R = r + m * 16;
        int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
        ldsm4(frag_a[buf], reinterpret_cast<int4*>(sh1_a_ptr) + R * A_COLS + c_swizzled);
      }
    }
#pragma unroll
    for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2) {
      int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
      const uint32_t* shb =
          reinterpret_cast<const uint32_t*>(sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);
      dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
    }
    __syncthreads();
    advance1();
  };

  auto clear_frag_c = [&]() {
#pragma unroll
    for (int n = 0; n < FRAGS_N_PER_WARP; ++n) frag_c[n] = {};
  };

  auto threadblock_reduce = [&]() {
    auto store = [&](int i) {
      if (sub_k == i) {
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
#pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
#pragma unroll
          for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[n][j];
        }
      }
      __syncthreads();
    };
    auto add = [&](int i) {
      if (sub_k == i) {
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
#pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
#pragma unroll
          for (int j = 0; j < 4; ++j) frag_c[n][j] += *sh_red++;
        }
      }
    };
    auto store_small = [&](int i) {
      if (sub_k == i && lane_id / 4 < size_m) {
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
#pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
          *sh_red++ = frag_c[n][0];
          *sh_red++ = frag_c[n][1];
        }
      }
      __syncthreads();
    };
    auto add_small = [&](int i) {
      if (sub_k == i && lane_id / 4 < size_m) {
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
#pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
          frag_c[n][0] += *sh_red++;
          frag_c[n][1] += *sh_red++;
        }
      }
    };

    if (size_m <= 8) {
      if constexpr (TILEBLOCKS_K == 2) {
        store_small(1);
        add_small(0);
      }
      if constexpr (TILEBLOCKS_K == 3) {
        store_small(1);
        add_small(0);
        store_small(2);
        add_small(0);
      }
      if constexpr (TILEBLOCKS_K == 4) {
        store_small(3);
        add_small(2);
        store_small(1);
        add_small(0);
        store_small(2);
        add_small(0);
      }
    } else {
      if constexpr (TILEBLOCKS_K == 2) {
        store(1);
        add(0);
      }
      if constexpr (TILEBLOCKS_K == 3) {
        store(1);
        add(0);
        store(2);
        add(0);
      }
      if constexpr (TILEBLOCKS_K == 4) {
        store(3);
        add(2);
        store(1);
        add(0);
        store(2);
        add(0);
      }
    }
  };

  auto write_sum_tile_sh = [&]() {
    const int n0 = warp_id * FRAGS_N_PER_WARP;
    const int r0 = lane_id / 4;
    const int r1 = r0 + 8;
    if (r0 < size_m) {
      const int c = (lane_id % 4) * 2;
#pragma unroll
      for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
        float* c_ptr = sh_c + r0 * TILESIZE_N + (n0 + n) * 8 + c;
        *c_ptr++ = frag_c[n][0];
        *c_ptr++ = frag_c[n][1];
      }
    }
    if (r1 < size_m) {
      const int c = (lane_id % 4) * 2;
#pragma unroll
      for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
        float* c_ptr = sh_c + r1 * TILESIZE_N + (n0 + n) * 8 + c;
        *c_ptr++ = frag_c[n][2];
        *c_ptr++ = frag_c[n][3];
      }
    }
  };

  // exl3_gemm_inner.cuh:456-480. The output transform reads the finished tile
  // out of shared memory and applies svh. `scale_block` is the GLOBAL 128-column
  // index, which upstream reaches through blockIdx.y in the standalone kernel
  // and through the post_scale POINTER here.
  auto output_had_sh_gl = [&]() {
    int sh_warp = warp_id;
    constexpr int active_warps = kBaseThreads / 32;
    for (;; sh_warp += active_warps) {
      int col = sh_warp % (TILESIZE_N / 128);
      int row = sh_warp / (TILESIZE_N / 128);
      if (row >= size_m) break;
      const float* had_in = sh_c + row * TILESIZE_N + col * 128;
      const half* post_scale_c = post_scale + slice2_n * gl_c_stride_n + col * 128;
      if constexpr (c_fp32) {
        float* had_out = gl_c_ptr_32 + row * size_n + col * 128;
        had_ff_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f, 0);
      } else {
        half* had_out = gl_c_ptr_16 + row * size_n + col * 128;
        had_fh_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f, 0);
      }
    }
  };

  auto read_sum_gl = [&]() {
    int n0 = warp_id * FRAGS_N_PER_WARP;
#pragma unroll
    for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
      int r0 = lane_id / 4;
      int r1 = r0 + 8;
      int c = (lane_id % 4) * 2;
      if (r0 < size_m) {
        if constexpr (c_fp32) {
          float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
          frag_c[n][0] += *c_ptr++;
          frag_c[n][1] += *c_ptr++;
        } else {
          half2* c_ptr = reinterpret_cast<half2*>(gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
          float2 interm = __half22float2(*c_ptr);
          frag_c[n][0] += interm.x;
          frag_c[n][1] += interm.y;
        }
      }
      if (r1 < size_m) {
        if constexpr (c_fp32) {
          float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
          frag_c[n][2] += *c_ptr++;
          frag_c[n][3] += *c_ptr++;
        } else {
          half2* c_ptr = reinterpret_cast<half2*>(gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
          float2 interm = __half22float2(*c_ptr);
          frag_c[n][2] += interm.x;
          frag_c[n][3] += interm.y;
        }
      }
    }
  };

  auto write_sum_gl = [&]() {
    int n0 = warp_id * FRAGS_N_PER_WARP;
#pragma unroll
    for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
      int r0 = lane_id / 4;
      int r1 = r0 + 8;
      int c = (lane_id % 4) * 2;
      if (r0 < size_m) {
        if constexpr (c_fp32) {
          float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
          *c_ptr++ = frag_c[n][0];
          *c_ptr++ = frag_c[n][1];
        } else {
          half2* c_ptr = reinterpret_cast<half2*>(gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
          *c_ptr = __floats2half2_rn(frag_c[n][0], frag_c[n][1]);
        }
      }
      if (r1 < size_m) {
        if constexpr (c_fp32) {
          float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
          *c_ptr++ = frag_c[n][2];
          *c_ptr++ = frag_c[n][3];
        } else {
          half2* c_ptr = reinterpret_cast<half2*>(gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
          *c_ptr = __floats2half2_rn(frag_c[n][2], frag_c[n][3]);
        }
      }
    }
  };

  auto reduce = [&]() {
    threadblock_reduce();
    int lock_i = tiles_k - slice2_k - 1;
    int lock_d = slice2_k - slice2_k0 + 1;
    int* lock = &locks[slice_m * blocks_n + slice2_n];
    barrier_acquire(lock, lock_i);
    bool first = lock_i == 0;
    bool last = lock_i + lock_d == tiles_k;
    if (!sub_k && !first) read_sum_gl();
    if (!sub_k && !last) write_sum_gl();
    if (!sub_k && last) write_sum_tile_sh();
    if (last) __syncthreads();
    if (!sub_k && last) output_had_sh_gl();
    barrier_release(lock, lock_d, last);
    clear_frag_c();
  };

  auto wait_stage = [&]() {
    cp_async_wait<SH_STAGES - 2>();
    __syncthreads();
  };

  auto matmul = [&](int buf) {
#pragma unroll
    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
      ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[n]);
  };

#pragma unroll
  for (int i = 0; i < SH_STAGES - 1; ++i) async_load_gl();
  wait_stage();

  clear_frag_c();
  if constexpr (FRAG_STAGES > 1) load_frags(0);

#define EXL3_FSTAGE_OLD(_load, _mul)                                              \
  async_load_gl();                                                                \
  wait_stage();                                                                   \
  load_frags(_load);                                                              \
  matmul(_mul);                                                                   \
  if (slice2_k == tiles_k - 1 || slice2_iters == 1) {                             \
    reduce();                                                                     \
    slice2_k0 = slice2_k + 1;                                                     \
  }                                                                               \
  advance2();                                                                     \
  if (!slice2_iters) break;

#define EXL3_FSTAGE(_load, _mul)                                                  \
  async_load_gl();                                                                \
  wait_stage();                                                                   \
  matmul(_mul);                                                                   \
  if (slice2_k == tiles_k - 1 || slice2_iters == 1) {                             \
    reduce();                                                                     \
    slice2_k0 = slice2_k + 1;                                                     \
  }                                                                               \
  advance2();                                                                     \
  if (!slice2_iters) break;                                                       \
  load_frags(_load);

  if constexpr (FRAG_STAGES == 1) {
    while (true) {
      EXL3_FSTAGE_OLD(0, 0)
    }
  }
  if constexpr (FRAG_STAGES == 3) {
    while (true) {
      EXL3_FSTAGE(1, 0)
      EXL3_FSTAGE(2, 1)
      EXL3_FSTAGE(0, 2)
    }
  }
  if constexpr (FRAG_STAGES == 5) {
    while (true) {
      EXL3_FSTAGE(1, 0)
      EXL3_FSTAGE(2, 1)
      EXL3_FSTAGE(3, 2)
      EXL3_FSTAGE(4, 3)
      EXL3_FSTAGE(0, 4)
    }
  }
#undef EXL3_FSTAGE_OLD
#undef EXL3_FSTAGE
}

// ── the grid-cooperative fused kernel (exl3_gemm_kernel.cuh:8-80) ────────────
//
// One launch does all three steps: the whole grid transforms A into A_had, meets
// at the grid barrier, then walks the output in 16-row bands, and the output
// transform rides the tail of the tile loop. The barrier is why this must be a
// COOPERATIVE launch: it spins, so every block has to be resident.
template <int bits, bool c_fp32, int cb, int TILESIZE_M, int TILESIZE_K, int TILESIZE_N,
          int SH_STAGES, int FRAG_STAGES>
__global__ __launch_bounds__(kBaseThreads* TILESIZE_K / 16) void exl3_gemm_kernel(
    const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
    const int size_m, const int size_k, const int size_n, int* __restrict__ locks,
    const half* __restrict__ suh, half* __restrict__ A_had, const half* __restrict__ svh) {
  // The two ints the grid barrier owns sit past the tile locks, which is exactly
  // where upstream puts its own barrier counters (`exl3_devctx.cuh:9`,
  // BARRIER_LOCKS_OFFSET == MAX_TILES_C).
  int* barrier_cs = locks + kMaxTilesC;
  const int blocks = static_cast<int>(gridDim.x);

  {
    int total_warps = size_m * size_k / 128;
    int warps_grid = gridDim.x * blockDim.x / 32;
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
    for (; this_warp < total_warps; this_warp += warps_grid)
      had_hf_r_128_inner<true, false>(A + this_warp * 128, A_had + this_warp * 128,
                                      suh + (this_warp * 128) % size_k, 0.088388347648f, 0);
    grid_barrier(barrier_cs, blocks);
    A = A_had;
  }

  int size_m_ = size_m;
  const half* A_ = A;
  void* C_ = C;

  while (size_m_ > 0) {
    exl3_gemm_kernel_inner<bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES,
                           FRAG_STAGES>(A_, B, C_, EXL3_MIN(size_m_, 16), size_k, size_n, locks,
                                        svh);
    A_ += 16 * size_k;
    if constexpr (c_fp32)
      C_ = static_cast<void*>(static_cast<float*>(C_) + 16 * size_n);
    else
      C_ = static_cast<void*>(static_cast<half*>(C_) + 16 * size_n);
    size_m_ -= 16;
    if (size_m_ > 0 || svh) grid_barrier(barrier_cs, blocks);
  }
}

// ── the per-device lock buffer (exl3_devctx.cu:59-70) ────────────────────────

int* DeviceLocks(int device) {
  static std::mutex mtx;
  static int* locks[16] = {};
  std::lock_guard<std::mutex> lock(mtx);
  if (device < 0 || device >= 16)
    throw std::runtime_error("vt cuda exl3: device ordinal out of range");
  if (locks[device] == nullptr) {
    void* p = nullptr;
    Check(cudaMalloc(&p, kLockInts * sizeof(int)), "cudaMalloc exl3 locks");
    Check(cudaMemset(p, 0, kLockInts * sizeof(int)), "cudaMemset exl3 locks");
    locks[device] = static_cast<int*>(p);
  }
  return locks[device];
}

// exl3_gemm.cu:38,251-256. Opting a kernel into 90 KiB of dynamic shared memory
// is a per-(device, function) property; doing it per launch costs a driver
// round-trip on the hot path.
void EnsureSmemOptIn(int device, const void* kernel) {
  static std::mutex mtx;
  static std::set<std::pair<int, const void*>> done;
  std::lock_guard<std::mutex> lock(mtx);
  if (done.insert({device, kernel}).second) {
    // Diagnose the shortfall BY NAME rather than letting cudaFuncSetAttribute
    // answer `invalid argument`, which is the lesson cuda_paged_attn.cu:116-140
    // already records for this exact call.
    if (!DynamicSmemFits(static_cast<long long>(kSmemMax))) {
      const DeviceCaps& caps = GetDeviceCaps(device);
      throw std::runtime_error(
          "vt cuda exl3: exl3_gemm needs " + std::to_string(kSmemMax) +
          " B of dynamic shared memory, but sm_" + std::to_string(caps.sm_arch()) +
          " caps opt-in shared memory at " +
          std::to_string(caps.max_shared_memory_per_block_optin) +
          " B. Upstream's SMEM_MAX is 90 KiB (exl3_gemm_inner.cuh:7) and every shape's "
          "static_assert is written against it, so a smaller ceiling needs its own tile "
          "table (MODEL-DSV4-EXL3).");
    }
    Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemMax),
          "cudaFuncSetAttribute exl3_gemm");
  }
}

// ── the launchers ────────────────────────────────────────────────────────────

void Exl3HadR128KernelCuda(Queue& q, Tensor& out, const Tensor& in, const Exl3HadArgs& args) {
  const int rows = static_cast<int>(in.shape[0]);
  const int cols = static_cast<int>(in.shape[1]);
  if (rows == 0 || cols == 0) return;
  const int blocks = cols / 128;
  const float r_scale = args.scale * 0.088388347648f;  // hadamard.cu:107
  const dim3 block_dim(32);
  const dim3 grid_dim(static_cast<unsigned>(rows), static_cast<unsigned>(blocks));
  cudaStream_t stream = AsStream(q);
  const half* sc = args.pre_scale != nullptr  ? args.pre_scale->Ptr<half>()
                   : args.post_scale != nullptr ? args.post_scale->Ptr<half>()
                                                : nullptr;
  const bool pre = args.pre_scale != nullptr;
  const bool post = args.post_scale != nullptr;

  if (in.dtype == DType::kF16) {
    const half* ip = in.Ptr<half>();
    half* op = out.Ptr<half>();
    if (pre)
      had_hf_r_128_kernel<true, false><<<grid_dim, block_dim, 0, stream>>>(ip, op, sc, r_scale);
    else if (post)
      had_hf_r_128_kernel<false, true><<<grid_dim, block_dim, 0, stream>>>(ip, op, sc, r_scale);
    else
      had_hf_r_128_kernel<false, false>
          <<<grid_dim, block_dim, 0, stream>>>(ip, op, nullptr, r_scale);
  } else {
    const float* ip = in.Ptr<float>();
    float* op = out.Ptr<float>();
    if (pre)
      had_ff_r_128_kernel<true, false><<<grid_dim, block_dim, 0, stream>>>(ip, op, sc, r_scale);
    else if (post)
      had_ff_r_128_kernel<false, true><<<grid_dim, block_dim, 0, stream>>>(ip, op, sc, r_scale);
    else
      had_ff_r_128_kernel<false, false>
          <<<grid_dim, block_dim, 0, stream>>>(ip, op, nullptr, r_scale);
  }
  Check(cudaGetLastError(), "exl3_had_r_128 launch");
}

// The instantiated (bits, cb) arm. Every other width refuses BY NAME rather
// than being silently decoded as if it were this one.
constexpr int kInstantiatedBits = 3;
constexpr int kInstantiatedCb = 1;

template <bool c_fp32>
const void* GemmKernelForShape(int shape_idx) {
  switch (shape_idx) {
    case 1:
      return reinterpret_cast<const void*>(
          &exl3_gemm_kernel<kInstantiatedBits, c_fp32, kInstantiatedCb, 16, 16, 128, 6, 5>);
    case 2:
      return reinterpret_cast<const void*>(
          &exl3_gemm_kernel<kInstantiatedBits, c_fp32, kInstantiatedCb, 16, 32, 128, 4, 3>);
    case 3:
      return reinterpret_cast<const void*>(
          &exl3_gemm_kernel<kInstantiatedBits, c_fp32, kInstantiatedCb, 16, 32, 256, 4, 3>);
    case 4:
      return reinterpret_cast<const void*>(
          &exl3_gemm_kernel<kInstantiatedBits, c_fp32, kInstantiatedCb, 16, 16, 512, 4, 3>);
    default:
      return nullptr;
  }
}

void Exl3GemmKernelCuda(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                        const Tensor& suh, const Tensor& svh, Tensor& a_had,
                        const Exl3GemmArgs& args) {
  if (args.bits != kInstantiatedBits || args.codebook != kInstantiatedCb) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_gemm has CUDA instantiations for bits == 3, codebook == 1 (mcg) "
        "only; got bits " +
        std::to_string(args.bits) + " codebook " + std::to_string(args.codebook) +
        ". MODEL-DSV4-EXL3 W2 records the other widths as owed; the CPU arm decodes "
        "every width and can serve them on a CPU queue.");
  }
  // NOT const: cudaLaunchCooperativeKernel takes `void**`, so each argument has
  // to be a modifiable lvalue whose address can be taken as `void*`.
  int size_m = static_cast<int>(a.shape[0]);
  int size_k = static_cast<int>(a.shape[1]);
  int size_n = static_cast<int>(c.shape[1]);
  if (size_m == 0 || size_k == 0 || size_n == 0) return;

  int device = 0;
  Check(cudaGetDevice(&device), "cudaGetDevice");
  const DeviceCaps& caps = GetDeviceCaps(device);
  const Exl3Cc cc = Exl3CcFromSm(caps.sm_major, caps.sm_minor);

  int shape_idx = args.force_shape_idx > 0
                      ? args.force_shape_idx
                      : Exl3SelectGemmShape(cc, size_m, size_k, size_n, args.bits, false);
  if (shape_idx <= 0 || !Exl3GemmShapeCompat(shape_idx, size_k, size_n)) {
    throw std::runtime_error("vt cuda exl3: no compatible exl3_gemm kernel shape for k=" +
                             std::to_string(size_k) + " n=" + std::to_string(size_n) +
                             " (selected shape " + std::to_string(shape_idx) + ")");
  }
  const Exl3GemmShape shape = Exl3GemmShapeParams(shape_idx);
  const bool c_fp32 = c.dtype == DType::kF32;
  const void* kernel = c_fp32 ? GemmKernelForShape<true>(shape_idx)
                              : GemmKernelForShape<false>(shape_idx);
  if (kernel == nullptr)
    throw std::runtime_error("vt cuda exl3: exl3_gemm shape " + std::to_string(shape_idx) +
                             " has no instantiation");

  int num_sms = Exl3GemmNumSms(shape_idx, size_k, size_n,
                               caps.multiprocessor_count > 0 ? caps.multiprocessor_count : 1);
  EnsureSmemOptIn(device, kernel);

  const half* a_ptr = a.Ptr<half>();
  const uint16_t* b_ptr = trellis.Ptr<uint16_t>();
  void* c_ptr = c.data;
  int* locks = DeviceLocks(device);
  const half* suh_ptr = suh.Ptr<half>();
  half* a_had_ptr = a_had.Ptr<half>();
  const half* svh_ptr = svh.Ptr<half>();
  void* kernel_args[] = {
      static_cast<void*>(&a_ptr),  static_cast<void*>(&b_ptr),     static_cast<void*>(&c_ptr),
      static_cast<void*>(&size_m), static_cast<void*>(&size_k),    static_cast<void*>(&size_n),
      static_cast<void*>(&locks),  static_cast<void*>(&suh_ptr),   static_cast<void*>(&a_had_ptr),
      static_cast<void*>(&svh_ptr)};
  Check(cudaLaunchCooperativeKernel(kernel, dim3(static_cast<unsigned>(num_sms)),
                                    dim3(static_cast<unsigned>(shape.block_dim)), kernel_args,
                                    kSmemMax, AsStream(q)),
        "cudaLaunchCooperativeKernel exl3_gemm");
  Check(cudaGetLastError(), "exl3_gemm launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kExl3HadR128, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Exl3HadR128Fn>(&Exl3HadR128KernelCuda)));
    RegisterOp(OpId::kExl3Gemm, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Exl3GemmFn>(&Exl3GemmKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
