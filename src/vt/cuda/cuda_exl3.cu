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
//   exllamav3_ext/quant/exl3_gemv.cu:29-169           the GEMV try-launch (W2c)
//   exllamav3_ext/quant/exl3_gemv_kernel.cuh:1-402    the GEMV kernel (W2c)
//   exllamav3_ext/quant/exl3_moe.cu:99-301            the MoE launcher (W2d)
//   exllamav3_ext/quant/exl3_moe_kernel.cuh:17-283    the fused MoE kernel (W2d)
//   exllamav3_ext/quant/hadamard_inner.cuh:284-473    its guad / d epilogues (W2d)
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
//   * The autotuner (`coop_autotune.cuh`) is NOT ported. Shape selection goes
//     through the table, and the GEMV arm is tried before it exactly as
//     `exl3_gemm.cu:220-236` tries it.
//   * W2c added the m<=8 GEMV (`exl3_gemv.cu` + `exl3_gemv_kernel.cuh`) and W2d
//     the fused MoE mgemm (`exl3_moe.cu` + `exl3_moe_kernel.cuh`). Both are
//     narrowed to bits == 3, codebook == 1 the way the regular kernel is, and
//     the GEMV's fp16 accumulation gives it its OWN bound (tier 3c, spec
//     `## W2cd design` W2c-3) rather than the regular kernel's tier 3.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <map>
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
// exl3_devctx.cuh:8-15. Past the tile locks come `2 * MAX_BARRIERS` ints for the
// group barriers (a counter and a sense per group; the single-group GEMM uses
// group 0, which is where upstream's BARRIER_LOCKS_OFFSET puts its own), and
// past those the MoE scheduler's self-resetting ticket state: [0] next ticket,
// [1] retired groups, [2 + g] the ticket published to group g.
constexpr size_t kMaxBarriers = 1024;
constexpr size_t kBarrierLocksOffset = kMaxTilesC;
constexpr size_t kMoeSchedOffset = kMaxTilesC + 2 * kMaxBarriers;
constexpr size_t kMoeSchedInts = 2 + 64;  // MOE_SCHED_INTS, MOE_MAX_GROUPS == 64
constexpr size_t kLockInts = kMoeSchedOffset + kMoeSchedInts;

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
// ptx.cuh:16. The FP16-accumulate fragment the m<=8 GEMV arm uses (W2c).
using FragC_h = Vec<half2, 2>;
// EXL3_GEMV_MAX_M (exl3_gemv_kernel.cuh:31). The host copy is vt::kExl3GemvMaxM
// in include/vt/ops.h; this one is what the kernel's template arithmetic reads,
// and a static_assert below pins the two together.
constexpr int kExl3GemvMaxMDev = 8;

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
__device__ inline void group_barrier(int* barrier_cs, int group_id, int group_size) {
  __syncthreads();
  if (threadIdx.x == 0) {
    int* cs = barrier_cs + group_id * 2;
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

// The whole-grid case is one group. Kept as a named entry point so the GEMM
// kernel below reads the way it did before the MoE arm needed the general form.
__device__ inline void grid_barrier(int* cs, int group_size) {
  group_barrier(cs, 0, group_size);
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
  int* barrier_cs = locks + kBarrierLocksOffset;
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


// ── the m<=8 GEMV arm (exl3_gemv_kernel.cuh:1-402) ───────────────────────────
//
// A QTIP-style small-m path on the unmodified EXL3 format. Warps split k and
// never synchronise in the main loop; B streams to registers behind a prefetch
// ring; the two-word bit windows are resolved in-warp by lane shuffles; one
// m16n8k16 MMA pair per 16x16 weight tile.
//
// IT IS A DIFFERENT NUMERIC ARM, not only a faster one, and that is why it has
// its own bound. The accumulate is `mma...f16.f16.f16.f16` — FP16 — folded to an
// f32 pair only every FOLD iterations, so an fp16 accumulator absorbs up to
// FOLD*16 k-elements. The regular kernel accumulates in f32 throughout. The
// spec's `## W2cd design` W2c-3 states the resulting bound (tier 3c, 6.0e-3
// relative RMS) rather than reusing tier 3's 1.0e-3, which a correct kernel
// here could not meet.
//
// NARROWED, exactly as the regular kernel is: bits == 3 and cb == 1 (mcg) only.
// Upstream instantiates 2/3/4 bpw over three codebooks; every other width
// DECLINES this arm and falls through to the shape table, which is upstream's
// own failure mode (`exl3_gemv_select_kernel` returns nullptr and
// `exl3_gemv_try_launch` returns false) and not an unimplemented refusal.

// mma.m16n8k16 with the A operand as two FragB halves, FP16 accumulate
// (exl3_gemv_kernel.cuh:37-52).
__device__ inline void ptx_mma_ab_h(const FragB& a01, const FragB& a23, const FragB& b,
                                    FragC_h& c) {
  const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
  const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
  const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
  uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(cc[0]), "+r"(cc[1])
      : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]), "r"(bb[0]), "r"(bb[1]));
}

// The register form of `dq8<3, cb, 4>` with the per-lane funnel alignment
// precomputed (exl3_gemv_kernel.cuh:120-134). Same eight codewords the shared
// -memory form reads, from two already-loaded words.
template <int cb>
__device__ inline void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1) {
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  w7 = fshift(b, a, s2);
  w6 = w7 >> 3;
  w5 = w6 >> 3;
  w4 = w5 >> 3;
  w3 = fshift(b, a, s2 + 12);
  w2 = w3 >> 3;
  w1 = w2 >> 3;
  w0 = w1 >> 3;
  f0[0] = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
  f0[1] = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
  f1[0] = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
  f1[1] = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
}

// exl3_gemv_kernel.cuh:138-402, narrowed to bits == 3. CFG 0 is the "narrow"
// config (512 threads, 2 n-tiles per warp, 16 k-splits) and CFG 1 the "wide" one
// (256 threads, 4 n-tiles, 8 k-splits). MMODE 0 is the m == 1 fast path and
// MMODE 1 covers 2 <= m <= 8 with row-guarded fragment loads.
template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256) void exl3_gemv_kernel(
    const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
    const int size_m, const int size_k, const int size_n, int* __restrict__ locks,
    const half* __restrict__ suh, half* __restrict__ A_had, const half* __restrict__ svh) {
  static_assert(bits == 3, "MODEL-DSV4-EXL3 W2c instantiates the 3 bpw arm only");
  constexpr int WK = CFG == 0 ? 16 : 8;    // k-split (warps per block)
  constexpr int WNT = CFG == 0 ? 2 : 4;    // adjacent n-tiles per warp
  constexpr int PF = CFG == 0 ? 4 : 2;     // prefetch ring depth
  constexpr int FOLD = CFG == 0 ? 4 : 2;   // fp16 -> fp32 fold cadence (divides PF)
  constexpr int THREADS = WK * 32;
  constexpr int ROWS = MMODE == 0 ? 1 : kExl3GemvMaxMDev;
  constexpr int COLS = WNT * 16;
  constexpr int TWORDS = 8 * bits;   // uint32 per 16x16 tile
  constexpr int LOADS = WNT;         // warp loads per k-slice (bits != 2)
  constexpr int LSTRIDE = 24;        // uint32 per load, bits == 3

  int* barrier_cs = locks + kBarrierLocksOffset;
  const int blocks = static_cast<int>(gridDim.x);

  // The input scales and Hadamard, identical to the regular kernel's.
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

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;
  const int chunk = EXL3_CEIL_DIVIDE(kslices, WK);
  const int ks0 = warp * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const size_t slice_stride = static_cast<size_t>(ntiles) * TWORDS;
  const half2* A2 = reinterpret_cast<const half2*>(A);
  const half2 hzero = __half2half2(__ushort_as_half(0));

  const int r0 = lane >> 2;
  const size_t a_row0 = static_cast<size_t>(r0) * (size_k / 2);
  const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

  // Per-lane extraction constants, the bits == 3 arm of :199-209.
  const int t_offset = lane << 3;
  const int b1 = (t_offset + 257) * 3;
  const int b2 = b1 + 21;
  const int i0 = (b1 - 16) / 32;
  const int i2 = (b2 - 1) / 32;
  const int x_s2 = (i2 + 1) * 32 - b2;
  const int x_src_a = i0 % 24;
  const int x_src_b = i2 % 24;

  __shared__ float sh_red[WK][ROWS][COLS];
  // `[[maybe_unused]]` is upstream's own annotation (exl3_gemv_kernel.cuh:214):
  // the staging buffer collapses to one word when SMEM_STAGE is off and nothing
  // reads it, which -Werror would otherwise call a defect.
  [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1]
                                               [SMEM_STAGE ? LOADS * LSTRIDE : 1];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + static_cast<size_t>(ks0) * slice_stride + group * WNT * TWORDS +
                         lane;
    auto ld_b = [&](int i, int l) -> uint32_t {
      return lane < 24 ? __ldcs(bp + static_cast<size_t>(i) * slice_stride + l * LSTRIDE) : 0u;
    };

    uint32_t pf[PF][LOADS];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn)
#pragma unroll
        for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;
        uint32_t bw[LOADS];
#pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];
        if (i + PF < myn) {
#pragma unroll
          for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
        }
        if constexpr (SMEM_STAGE) {
          __syncwarp();
#pragma unroll
          for (int l = 0; l < LOADS; ++l)
            if (lane < 24) sh_stage[warp][l * LSTRIDE + lane] = bw[l];
          __syncwarp();
        }
        const size_t a_col = static_cast<size_t>(ks0 + i) * 8 + (lane & 3);
        FragB a01, a23;
        a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
        a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
        a01[1] = hzero;
        a23[1] = hzero;
#pragma unroll
        for (int t = 0; t < WNT; ++t) {
          FragB f0, f1;
          if constexpr (SMEM_STAGE) {
            const uint32_t* tp = &sh_stage[warp][t * TWORDS];
            dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
          } else {
            const uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
            const uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
            dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
          }
          ptx_mma_ab_h(a01, a23, f0, ch[t][0]);
          ptx_mma_ab_h(a01, a23, f1, ch[t][1]);
        }
        // The FOLD cadence. This is the whole reason tier 3c exists: between
        // folds the accumulator is fp16.
        if ((d + 1) % FOLD == 0 || i + 1 == myn) {
#pragma unroll
          for (int t = 0; t < WNT; ++t)
#pragma unroll
            for (int f = 0; f < 2; ++f) {
              acc0[t][f].x += __low2float(ch[t][f][0]);
              acc0[t][f].y += __high2float(ch[t][f][0]);
              ch[t][f][0] = hzero;
            }
        }
      }
    }

    // Cross-warp reduction over the k splits (:335-360).
    {
      const int c0 = 2 * (lane & 3);
      const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
      const int sr0 = MMODE == 0 ? 0 : r0;
      if (store0) {
#pragma unroll
        for (int t = 0; t < WNT; ++t)
#pragma unroll
          for (int f = 0; f < 2; ++f) {
            const int col = t * 16 + f * 8 + c0;
            sh_red[warp][sr0][col + 0] = acc0[t][f].x;
            sh_red[warp][sr0][col + 1] = acc0[t][f].y;
          }
      }
    }
    __syncthreads();

    const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
    for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS) {
      const int r = idx / COLS;
      const int c = idx % COLS;
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][r][c];
      const int col = group * COLS + c;
      if constexpr (c_fp32)
        static_cast<float*>(C)[static_cast<size_t>(r) * size_n + col] = sum;
      else
        static_cast<half*>(C)[static_cast<size_t>(r) * size_n + col] = __float2half_rn(sum);
    }
    __syncthreads();
  }

  // The output scales and Hadamard, same semantics as the inner GEMM epilogue.
  {
    grid_barrier(barrier_cs, blocks);
    int total_warps = size_m * size_n / 128;
    int warps_grid = gridDim.x * blockDim.x / 32;
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
    for (; this_warp < total_warps; this_warp += warps_grid) {
      if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>(static_cast<const float*>(C) + this_warp * 128,
                                        static_cast<float*>(C) + this_warp * 128,
                                        svh + (this_warp * 128) % size_n, 0.088388347648f, 0);
      else
        had_hf_r_128_inner<false, true>(static_cast<const half*>(C) + this_warp * 128,
                                        static_cast<half*>(C) + this_warp * 128,
                                        svh + (this_warp * 128) % size_n, 0.088388347648f, 0);
    }
  }
}

// ── the fused MoE epilogues (hadamard_inner.cuh:284-473) ─────────────────────
//
// Two inners the standalone GEMM never needs. `guad` is the whole middle of the
// MoE step in one pass — the output Hadamards for gate and up, the activation
// and the gate multiply, the down pre-scale and the down input Hadamard — and
// `d` is the epilogue that turns the down GEMM's fp16 store into the f32
// accumulator with an atomicAdd.
//
// The scale pointers arrive PRE-OFFSET by the caller (`128 * token_off`), which
// is what upstream's `blockIdx.y == 0` reduces to inside the MoE kernel, so both
// index their scale arrays at lane `t`.

// The 128-point transform on the four halves a lane holds, as `had` inside
// `had_hf_r_128_guad_inner` (hadamard_inner.cuh:299-321).
__device__ inline void moe_had4(half4& v, int t, float r_scale) {
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
}

__device__ inline half2 moe_silu_h2(const half2& x) {
  // hadamard_inner.cuh:323-332, in fp16 exactly as upstream computes it.
  half2 one = __float2half2_rn(1.0f);
  half2 e = h2exp(__hneg2(x));
  return __hmul2(x, h2rcp(__hadd2(one, e)));
}

__device__ inline half2 moe_gelu_h2(const half2& x) {
  // hadamard_inner.cuh:334-342.
  float2 xf = __half22float2(x);
  const float c = 0.797884560803f;  // sqrt(2/pi)
  xf.x = 0.5f * xf.x * (1.0f + tanhf(c * (xf.x + 0.044715f * xf.x * xf.x * xf.x)));
  xf.y = 0.5f * xf.y * (1.0f + tanhf(c * (xf.y + 0.044715f * xf.y * xf.y * xf.y)));
  return __float22half2_rn(xf);
}

// `had_hf_r_128_guad_inner` (hadamard_inner.cuh:284-413), plus ONE arm that is
// not upstream's: `kExl3MoeActSiluAndMulClamp` is vLLM's `SiluAndMulWithClamp`
// (activation.py:197-201), which clamps the gate BEFORE the silu and applies the
// limit unconditionally. AGENTS.md makes vLLM the authority wherever it
// implements the behavior; see `.agents/specs/model-dsv4-exl3.md` `## W2cd
// design` W2d-2 for the number the two orders differ by.
constexpr int kExl3MoeActSilu = 0;
constexpr int kExl3MoeActGelu = 1;
constexpr int kExl3MoeActRelu2NoGate = 2;
constexpr int kExl3MoeActSiluAndMulClamp = 3;

__device__ inline void had_hf_r_128_guad_inner(const half* __restrict__ in_g,
                                               const half* __restrict__ in_u,
                                               half* __restrict__ out, const half* __restrict__ sv_g,
                                               const half* __restrict__ sv_u,
                                               const half* __restrict__ su_d, const float r_scale,
                                               const float act_limit, const int act_function) {
  const int t = threadIdx.x & 31;
  const bool gated = act_function != kExl3MoeActRelu2NoGate;

  half4 vg = {};
  half4 vu = reinterpret_cast<const half4*>(in_u)[t];
  moe_had4(vu, t, r_scale);
  half4 su = reinterpret_cast<const half4*>(sv_u)[t];
  vu.x = __hmul2(vu.x, su.x);
  vu.y = __hmul2(vu.y, su.y);

  if (gated) {
    vg = reinterpret_cast<const half4*>(in_g)[t];
    moe_had4(vg, t, r_scale);
    half4 sg = reinterpret_cast<const half4*>(sv_g)[t];
    vg.x = __hmul2(vg.x, sg.x);
    vg.y = __hmul2(vg.y, sg.y);
  }

  if (act_function == kExl3MoeActSiluAndMulClamp) {
    // vLLM's order: clamp gate to the limit (MAX only), clamp up to +/- limit,
    // THEN silu the clamped gate. Unconditional — a zero limit clamps to zero,
    // which is vLLM's own degenerate case and not upstream's "no limit".
    const half2 lo = __float2half2_rn(-act_limit);
    const half2 hi = __float2half2_rn(act_limit);
    vg.x = __hmin2(vg.x, hi);
    vg.y = __hmin2(vg.y, hi);
    vu.x = __hmin2(__hmax2(vu.x, lo), hi);
    vu.y = __hmin2(__hmax2(vu.y, lo), hi);
    vg.x = moe_silu_h2(vg.x);
    vg.y = moe_silu_h2(vg.y);
  } else {
    switch (act_function) {
      case kExl3MoeActSilu:
        vg.x = moe_silu_h2(vg.x);
        vg.y = moe_silu_h2(vg.y);
        break;
      case kExl3MoeActGelu:
        vg.x = moe_gelu_h2(vg.x);
        vg.y = moe_gelu_h2(vg.y);
        break;
      case kExl3MoeActRelu2NoGate:
        vg.x = __hmax2(vu.x, __float2half2_rn(0.0f));
        vg.y = __hmax2(vu.y, __float2half2_rn(0.0f));
        break;
      default:
        break;
    }
    // hadamard_inner.cuh:389-397: upstream's limit is optional and lands AFTER
    // the activation.
    if (act_limit != 0.0f) {
      const half2 lo = __float2half2_rn(-act_limit);
      const half2 hi = __float2half2_rn(act_limit);
      vu.x = __hmin2(__hmax2(vu.x, lo), hi);
      vu.y = __hmin2(__hmax2(vu.y, lo), hi);
      vg.x = __hmin2(vg.x, hi);
      vg.y = __hmin2(vg.y, hi);
    }
  }

  vg.x = __hmul2(vg.x, vu.x);  // the gate multiply
  vg.y = __hmul2(vg.y, vu.y);
  half4 sd = reinterpret_cast<const half4*>(su_d)[t];  // the down PRE scale
  vg.x = __hmul2(vg.x, sd.x);
  vg.y = __hmul2(vg.y, sd.y);
  moe_had4(vg, t, r_scale);
  reinterpret_cast<half4*>(out)[t] = vg;
}

// `had_hf_r_128_d_inner` (hadamard_inner.cuh:418-473). Half in, f32 out, the
// post-scale in f32 and an atomicAdd into the accumulator. The reshuffle through
// shared memory is what makes the four atomicAdds coalesced.
//
// DEVIATION, recorded: upstream declares its own `extern __shared__ float
// temp_shared[]` inside this function and slices it by `threadIdx.x / 32`. Here
// the warp-private 128-float slice arrives as a PARAMETER, so the translation
// unit carries ONE `extern __shared__` declaration instead of two of different
// types. Same memory, same slice, one fewer thing for a reader to reconcile.
__device__ inline void had_hf_r_128_d_inner(const half* __restrict__ in, float* __restrict__ out,
                                            const half* __restrict__ post, const float r_scale,
                                            float* sh) {
  const int t = threadIdx.x & 31;
  half4 v = reinterpret_cast<const half4*>(in)[t];
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
  h0 *= r_scale;
  h1 *= r_scale;
  h2 *= r_scale;
  h3 *= r_scale;
  half4 sc = reinterpret_cast<const half4*>(post)[t];
  h0 *= __low2float(sc.x);
  h1 *= __high2float(sc.x);
  h2 *= __low2float(sc.y);
  h3 *= __high2float(sc.y);
  sh[t * 4 + 0] = h0;
  sh[t * 4 + 1] = h1;
  sh[t * 4 + 2] = h2;
  sh[t * 4 + 3] = h3;
  __syncwarp();
  atomicAdd(out + 0 + t, sh[0 + t]);
  atomicAdd(out + 32 + t, sh[32 + t]);
  atomicAdd(out + 64 + t, sh[64 + t]);
  atomicAdd(out + 96 + t, sh[96 + t]);
}

// ── the fused MoE kernel (exl3_moe_kernel.cuh:17-283) ────────────────────────
//
// A persistent cooperative launch: `num_groups` groups of `group_size` blocks
// each draw expert tickets from a self-resetting scheduler in the lock buffer
// and meet at a group barrier between stages. The group barrier is legal for the
// same reason the GEMM's grid barrier is — `cudaLaunchCooperativeKernel`
// guarantees every block is co-resident, and a spin barrier over blocks that are
// not co-resident deadlocks.
//
// The five stages are the ones `src/vt/cpu/cpu_exl3_kernels.cpp` reproduces on
// the host, in the same order and with the same roundings. Both GEMM calls pass
// a NULL post_scale, so the tile loop stores its f32 accumulator as fp16 without
// an output Hadamard, and stages 3 and 5 own those.
constexpr int kMoeSmsPerExpert = 8;       // MOE_SMS_PER_EXPERT
constexpr int kMoeMaxSmsPerExpert = 32;   // MOE_MAX_SMS_PER_EXPERT
constexpr int kMoeMaxGroups = 64;         // MOE_MAX_GROUPS
constexpr int kMoeTilesizeK = 32;         // MOE_TILESIZE_K
constexpr int kMoeTilesizeM = 16;
constexpr int kMoeShStages = 3;
constexpr int kMoeFragStages = 3;

template <int bits, int MOE_TILESIZE_N, int cb>
__global__ __launch_bounds__(kBaseThreads* kMoeTilesizeK / 16) void exl3_moe_kernel(
    const half* __restrict__ hidden_state, half* __restrict__ temp_state_g,
    half* __restrict__ temp_state_u, half* __restrict__ temp_intermediate_g,
    half* __restrict__ temp_intermediate_u, float* __restrict__ output_state,
    const uint16_t* const* __restrict__ gate_trellis, const half* const* __restrict__ gate_suh,
    const half* const* __restrict__ gate_svh, const uint16_t* const* __restrict__ up_trellis,
    const half* const* __restrict__ up_suh, const half* const* __restrict__ up_svh,
    const uint16_t* const* __restrict__ down_trellis, const half* const* __restrict__ down_suh,
    const half* const* __restrict__ down_svh, const int64_t* __restrict__ expert_count,
    const int64_t* __restrict__ token_sorted, const half* __restrict__ weight_sorted,
    const int hidden_dim, const int intermediate_dim, const int num_experts,
    const int max_tokens_per_expert, const float act_limit, const int act_function,
    const int K_gate, const int K_up, const int K_down, int* __restrict__ locks) {
  const int group_idx = blockIdx.z;
  const int block_idx = blockIdx.x;
  const int group_size = gridDim.x;
  const int num_groups = gridDim.z;
  const int block_threads = kBaseThreads * kMoeTilesizeK / 16;
  const int group_threads = group_size * block_threads;
  const int warp_id = threadIdx.x / 32;
  const int warps_per_group = group_threads / 32;
  const int warps_per_block = block_threads / 32;
  const int warp_idx0 = block_idx * warps_per_block + warp_id;

  temp_state_g += static_cast<size_t>(group_idx) * max_tokens_per_expert * hidden_dim;
  temp_state_u += static_cast<size_t>(group_idx) * max_tokens_per_expert * hidden_dim;
  temp_intermediate_g += static_cast<size_t>(group_idx) * max_tokens_per_expert * intermediate_dim;
  temp_intermediate_u += static_cast<size_t>(group_idx) * max_tokens_per_expert * intermediate_dim;

  int* barrier_cs = locks + kBarrierLocksOffset;
  int* sched = locks + kMoeSchedOffset;
  locks += static_cast<size_t>(group_idx) * EXL3_MAX(hidden_dim, intermediate_dim) / 128;

  extern __shared__ float moe_shared[];

  // exl3_moe_kernel.cuh:47-50. Active experts are numbered in scan order and a
  // group takes the one matching its ticket; after finishing it draws the next
  // unclaimed ticket, so the load balances greedily without assuming a uniform
  // cost per expert.
  int ticket = group_idx;
  int start = 0;
  int end = 0;
  int expert_idx_assign = 0;
  for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    start = end;
    end += static_cast<int>(expert_count[expert_idx]);
    const int token_count = end - start;
    if (token_count == 0) continue;
    if (token_count > max_tokens_per_expert) continue;
    if (expert_idx_assign++ != ticket) continue;

    const uint16_t* e_g_tr = gate_trellis[expert_idx];
    const half* e_g_su = gate_suh[expert_idx];
    const half* e_g_sv = gate_svh[expert_idx];
    const uint16_t* e_u_tr = up_trellis[expert_idx];
    const half* e_u_su = up_suh[expert_idx];
    const half* e_u_sv = up_svh[expert_idx];
    const uint16_t* e_d_tr = down_trellis[expert_idx];
    const half* e_d_su = down_suh[expert_idx];
    const half* e_d_sv = down_svh[expert_idx];

    const bool gated = act_function != kExl3MoeActRelu2NoGate;

    // stage 1: gather + input Hadamard (:85-114).
    {
      const int warps_per_token = hidden_dim / 128;
      const int total_warps = token_count * warps_per_token;
      const int64_t* top_x = token_sorted + start;
      for (int wi = warp_idx0; wi < total_warps; wi += warps_per_group) {
        const int token_idx = static_cast<int>(top_x[wi / warps_per_token]);
        const int token_off = wi % warps_per_token;
        const half* in_ptr = hidden_state + static_cast<size_t>(token_idx) * hidden_dim +
                             static_cast<size_t>(token_off) * 128;
        if (gated)
          had_hf_r_128_inner<true, false>(in_ptr, temp_state_g + 128 * wi,
                                          e_g_su + 128 * token_off, 0.088388347648f, 0);
        had_hf_r_128_inner<true, false>(in_ptr, temp_state_u + 128 * wi,
                                        e_u_su + 128 * token_off, 0.088388347648f, 0);
      }
      group_barrier(barrier_cs, group_idx, group_size);
    }

    // stage 2: the gate and up GEMMs (:117-161). `post_scale` is nullptr, so the
    // tile loop skips the output transform and rounds its f32 accumulator to
    // fp16 at the store.
    //
    // `K` is upstream's per-projection bit width, and upstream switches on it at
    // RUN TIME because its `t_bits == 0` instance leaves all three free
    // (`exl3_moe_kernel.cuh:139-149`). This port instantiates one width, so the
    // switch would be a ladder with one live arm; the parameter is kept because
    // the launcher's refusal is what enforces the equality, and dropping it here
    // would move that contract out of sight.
    auto gemm_band = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, int K,
                         int size_k, int size_n) {
      (void)K;
      int size_m = token_count;
      while (size_m > 0) {
        exl3_gemm_kernel_inner<bits, false, cb, kMoeTilesizeM, kMoeTilesizeK, MOE_TILESIZE_N,
                               kMoeShStages, kMoeFragStages>(
            in_addr, trellis, static_cast<void*>(out_addr), EXL3_MIN(size_m, 16), size_k, size_n,
            locks, nullptr);
        in_addr += 16 * size_k;
        out_addr += 16 * size_n;
        size_m -= 16;
      }
    };

    if (gated)
      gemm_band(temp_state_g, temp_intermediate_g, e_g_tr, K_gate, hidden_dim, intermediate_dim);
    gemm_band(temp_state_u, temp_intermediate_u, e_u_tr, K_up, hidden_dim, intermediate_dim);
    group_barrier(barrier_cs, group_idx, group_size);

    // stage 3: the fused guad pass (:165-188).
    {
      const int warps_per_token = intermediate_dim / 128;
      const int total_warps = token_count * warps_per_token;
      for (int wi = warp_idx0; wi < total_warps; wi += warps_per_group) {
        const int token_off = wi % warps_per_token;
        had_hf_r_128_guad_inner(temp_intermediate_g + 128 * wi, temp_intermediate_u + 128 * wi,
                                temp_intermediate_g + 128 * wi, e_g_sv + 128 * token_off,
                                e_u_sv + 128 * token_off, e_d_su + 128 * token_off,
                                0.088388347648f, act_limit, act_function);
      }
      group_barrier(barrier_cs, group_idx, group_size);
    }

    // stage 4: the down GEMM (:191-233), into `temp_state_g`.
    gemm_band(temp_intermediate_g, temp_state_g, e_d_tr, K_down, intermediate_dim, hidden_dim);
    group_barrier(barrier_cs, group_idx, group_size);

    // stage 5: the output Hadamard for d, times the routing weight, scatter-added
    // into the f32 accumulator (:237-259).
    {
      const int warps_per_token = hidden_dim / 128;
      const int total_warps = token_count * warps_per_token;
      const int64_t* top_x = token_sorted + start;
      const half* weights = weight_sorted + start;
      float* sh = moe_shared + warp_id * 128;
      for (int wi = warp_idx0; wi < total_warps; wi += warps_per_group) {
        const int token_idx = static_cast<int>(top_x[wi / warps_per_token]);
        const half weight = weights[wi / warps_per_token];
        const int token_off = wi % warps_per_token;
        float* out_ptr = output_state + static_cast<size_t>(token_idx) * hidden_dim +
                         static_cast<size_t>(token_off) * 128;
        had_hf_r_128_d_inner(temp_state_g + 128 * wi, out_ptr, e_d_sv + 128 * token_off,
                             0.088388347648f * __half2float(weight), sh);
      }
    }

    // Draw the next ticket and publish it through the end-of-expert barrier,
    // which also protects the temp buffers for reuse (:261-266).
    if (block_idx == 0 && threadIdx.x == 0)
      sched[2 + group_idx] = num_groups + atomicAdd(&sched[0], 1);
    group_barrier(barrier_cs, group_idx, group_size);
    ticket = sched[2 + group_idx];
  }

  // Retire; the last group out resets the scheduler for the next launch
  // (:269-282). The fence is what orders every group's ticket grabs before the
  // reset, so a straggler's in-flight grab cannot leak into the next launch.
  // Upstream spells the ordering with `cuda::atomic_ref` acq_rel; this file
  // already spells every such barrier as the plain atomics plus `__threadfence`
  // this tree uses, for the reason the header records.
  if (block_idx == 0 && threadIdx.x == 0) {
    __threadfence();
    const int retired = atomicAdd(&sched[1], 1);
    if (retired == num_groups - 1) {
      atomicExch(&sched[0], 0);
      atomicExch(&sched[1], 0);
    }
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

// ── the GEMV try-launch (exl3_gemv.cu:92-169) ────────────────────────────────
//
// Returns true when it launched. A false is a DECLINE, not a failure: the caller
// falls through to the shape table unchanged, which is upstream's own
// arrangement at `exl3_gemm.cu:220-236`.
//
// The occupancy query is cached per (device, kernel) because it is a driver
// round-trip and it feeds the shape heuristic on every call
// (`exl3_gemv.cu:125-135`).
const void* GemvKernel(bool c_fp32, int mmode, int cfg, bool smem) {
#define VT_EXL3_GEMV_SEL(fp32_, mm_, cfg_, sm_)                                            \
  if (c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_)                       \
    return reinterpret_cast<const void*>(                                                  \
        &exl3_gemv_kernel<kInstantiatedBits, fp32_, kInstantiatedCb, mm_, cfg_, sm_>);
#define VT_EXL3_GEMV_ROW(sm_)                                                              \
  VT_EXL3_GEMV_SEL(false, 0, 0, sm_) VT_EXL3_GEMV_SEL(false, 0, 1, sm_)                    \
  VT_EXL3_GEMV_SEL(false, 1, 0, sm_) VT_EXL3_GEMV_SEL(false, 1, 1, sm_)                    \
  VT_EXL3_GEMV_SEL(true, 0, 0, sm_) VT_EXL3_GEMV_SEL(true, 0, 1, sm_)                      \
  VT_EXL3_GEMV_SEL(true, 1, 0, sm_) VT_EXL3_GEMV_SEL(true, 1, 1, sm_)
  VT_EXL3_GEMV_ROW(false)
  VT_EXL3_GEMV_ROW(true)
#undef VT_EXL3_GEMV_ROW
#undef VT_EXL3_GEMV_SEL
  return nullptr;
}

int GemvOccupancy(int device, const void* kernel, int block_dim) {
  static std::mutex mtx;
  static std::map<std::pair<int, const void*>, int> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find({device, kernel});
  if (it != cache.end()) return it->second;
  int blocks_per_sm = 0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor exl3_gemv");
  cache[{device, kernel}] = blocks_per_sm;
  return blocks_per_sm;
}

bool Exl3GemvTryLaunch(Queue& q, int device, Exl3Cc cc, int num_sms, void** kernel_args,
                       int size_m, int size_k, int size_n, const Exl3GemmArgs& args,
                       bool c_fp32) {
  static_assert(kExl3GemvMaxMDev == kExl3GemvMaxM,
                "the device and host copies of EXL3_GEMV_MAX_M must agree");
  // exl3_gemv.cu:108-116: the free integer tests first, then the env read.
  if (args.force_gemv == 0) return false;
  if (args.bits != kInstantiatedBits || args.codebook != kInstantiatedCb) return false;
  if (!Exl3GemvHardEligible(size_m, size_k, size_n, args.bits, args.codebook,
                            /*has_su_sv=*/true))
    return false;
  const int mode = args.force_gemv > 0 ? 2 : Exl3GemvMode();
  if (mode == 0) return false;
  const int mmode = size_m == 1 ? 0 : 1;
  const bool smem = Exl3GemvSmemMode() == 1;

  const void* narrow = GemvKernel(c_fp32, mmode, 0, smem);
  if (narrow == nullptr) return false;
  const int narrow_coresident = GemvOccupancy(device, narrow, 512) * num_sms;
  const int cfg = Exl3GemvSelectConfig(cc, size_m, size_k, size_n, args.bits, args.codebook, mode,
                                       narrow_coresident);
  if (cfg < 0) return false;
  const void* kernel = cfg == 0 ? narrow : GemvKernel(c_fp32, mmode, cfg, smem);
  if (kernel == nullptr) return false;

  const int block_dim = cfg == 0 ? 512 : 256;
  const int cols = cfg == 0 ? 32 : 64;
  const int max_blocks = GemvOccupancy(device, kernel, block_dim) * num_sms;
  const int grid = EXL3_MIN(size_n / cols, max_blocks);
  if (grid < 1) return false;

  Check(cudaLaunchCooperativeKernel(kernel, dim3(static_cast<unsigned>(grid)),
                                    dim3(static_cast<unsigned>(block_dim)), kernel_args, 0,
                                    AsStream(q)),
        "cudaLaunchCooperativeKernel exl3_gemv");
  Check(cudaGetLastError(), "exl3_gemv launch");
  return true;
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

  const half* a_ptr0 = a.Ptr<half>();
  const uint16_t* b_ptr0 = trellis.Ptr<uint16_t>();
  void* c_ptr0 = c.data;
  int* locks0 = DeviceLocks(device);
  const half* suh_ptr0 = suh.Ptr<half>();
  half* a_had_ptr0 = a_had.Ptr<half>();
  const half* svh_ptr0 = svh.Ptr<half>();
  void* gemv_args[] = {static_cast<void*>(&a_ptr0),   static_cast<void*>(&b_ptr0),
                       static_cast<void*>(&c_ptr0),   static_cast<void*>(&size_m),
                       static_cast<void*>(&size_k),   static_cast<void*>(&size_n),
                       static_cast<void*>(&locks0),   static_cast<void*>(&suh_ptr0),
                       static_cast<void*>(&a_had_ptr0), static_cast<void*>(&svh_ptr0)};
  // MODEL-DSV4-EXL3 W2c. Upstream tries the GEMV only when the caller forced
  // neither a shape nor an SM count (`exl3_gemm.cu:222`), because forcing either
  // is a request for a specific regular-kernel launch; `force_gemv` is the
  // separate lever that asks for THIS arm.
  if (args.force_shape_idx <= 0 || args.force_gemv > 0) {
    if (Exl3GemvTryLaunch(q, device, cc, caps.multiprocessor_count > 0
                                            ? caps.multiprocessor_count
                                            : 1,
                          gemv_args, size_m, size_k, size_n, args, c.dtype == DType::kF32))
      return;
  }
  if (args.force_gemv > 0) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_gemm was asked to force the m<=8 GEMV arm (force_gemv=1) but the "
        "call is not hard-eligible for it: m=" +
        std::to_string(size_m) + " k=" + std::to_string(size_k) + " n=" +
        std::to_string(size_n) + " bits=" + std::to_string(args.bits) +
        ". Upstream's own direct entry point refuses the same way (exl3_gemv.cu:238).");
  }

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

// ── the fused MoE launcher (exl3_moe.cu:99-301) ──────────────────────────────
//
// Every validation upstream performs lives in `src/vt/ops.cpp`, shared with the
// CPU arm; what is left here is the launch geometry and the instantiation
// choice, which are the only device-specific parts.
void Exl3MoeMlpKernelCuda(Queue& q, Tensor& output_state, const Tensor& hidden_state,
                          const Exl3MoeExpertTables& tables, const Exl3MoeRouting& routing,
                          const Exl3MoeTemps& temps, const Exl3MoeArgs& args) {
  if (args.bits_gate != kInstantiatedBits || args.bits_up != kInstantiatedBits ||
      args.bits_down != kInstantiatedBits || args.codebook != kInstantiatedCb) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_moe has CUDA instantiations for bits == 3, codebook == 1 (mcg) "
        "only; got bits (" +
        std::to_string(args.bits_gate) + ", " + std::to_string(args.bits_up) + ", " +
        std::to_string(args.bits_down) + ") codebook " + std::to_string(args.codebook) +
        ". MODEL-DSV4-EXL3 W2 records the other widths as owed; the CPU arm decodes every "
        "width and can serve them on a CPU queue.");
  }
  // NOT const: cudaLaunchCooperativeKernel takes `void**`, so every argument has
  // to be a modifiable lvalue whose address can be taken as `void*`.
  int bsz = static_cast<int>(hidden_state.shape[0]);
  int hidden_dim = static_cast<int>(hidden_state.shape[1]);
  int intermediate_dim = static_cast<int>(temps.intermediate_g->shape[2]);
  int num_experts = static_cast<int>(routing.expert_count->Numel() - 1);
  int max_tokens_per_expert = static_cast<int>(temps.state_g->shape[1]);
  int concurrency = static_cast<int>(temps.state_g->shape[0]);
  if (bsz == 0 || num_experts == 0) return;

  int device = 0;
  Check(cudaGetDevice(&device), "cudaGetDevice");
  const DeviceCaps& caps = GetDeviceCaps(device);
  const int num_sms = caps.multiprocessor_count > 0 ? caps.multiprocessor_count : 1;

  // exl3_moe.cu:210-222. Every block of the grid must be co-resident for the
  // group barriers, so groups * width <= num_sms. With a known active-expert
  // count, launch only as many groups as there are experts and widen them into
  // the freed SMs.
  if (static_cast<long long>(concurrency) * kMoeSmsPerExpert > num_sms) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_moe was given temp buffers for concurrency " +
        std::to_string(concurrency) + ", which needs " +
        std::to_string(concurrency * kMoeSmsPerExpert) + " co-resident SMs but this device has " +
        std::to_string(num_sms) +
        ". Size the buffers with vt::Exl3MoeMaxConcurrency(device_sms).");
  }
  int num_groups = EXL3_MIN(concurrency, kMoeMaxGroups);
  int group_size = kMoeSmsPerExpert;
  if (args.num_active > 0) {
    num_groups = EXL3_MIN(num_groups, args.num_active);
    group_size = EXL3_MIN(num_sms / num_groups, kMoeMaxSmsPerExpert);
  }
  if (num_groups < 1 || group_size < 1) return;

  // exl3_moe.cu:224-226. The N tile is 256 when both dims allow it.
  const bool n256 = (hidden_dim % 256 == 0) && (intermediate_dim % 256 == 0);
  const void* kernel =
      n256 ? reinterpret_cast<const void*>(
                 &exl3_moe_kernel<kInstantiatedBits, 256, kInstantiatedCb>)
           : reinterpret_cast<const void*>(
                 &exl3_moe_kernel<kInstantiatedBits, 128, kInstantiatedCb>);
  EnsureSmemOptIn(device, kernel);

  const half* hid = hidden_state.Ptr<half>();
  half* st_g = temps.state_g->Ptr<half>();
  half* st_u = temps.state_u->Ptr<half>();
  half* in_g = temps.intermediate_g->Ptr<half>();
  half* in_u = temps.intermediate_u->Ptr<half>();
  float* out = output_state.Ptr<float>();
  auto tbl16 = [](const Tensor* tt) {
    return reinterpret_cast<const uint16_t* const*>(tt->data);
  };
  auto tblh = [](const Tensor* tt) { return reinterpret_cast<const half* const*>(tt->data); };
  const uint16_t* const* g_tr = tbl16(tables.gate_trellis);
  const half* const* g_su = tblh(tables.gate_suh);
  const half* const* g_sv = tblh(tables.gate_svh);
  const uint16_t* const* u_tr = tbl16(tables.up_trellis);
  const half* const* u_su = tblh(tables.up_suh);
  const half* const* u_sv = tblh(tables.up_svh);
  const uint16_t* const* d_tr = tbl16(tables.down_trellis);
  const half* const* d_su = tblh(tables.down_suh);
  const half* const* d_sv = tblh(tables.down_svh);
  const int64_t* cnt = routing.expert_count->Ptr<int64_t>();
  const int64_t* tok = routing.token_sorted->Ptr<int64_t>();
  const half* wgt = routing.weight_sorted->Ptr<half>();
  int* locks = DeviceLocks(device);
  float act_limit = args.act_limit;
  int act_function = static_cast<int>(args.act);
  int k_gate = args.bits_gate, k_up = args.bits_up, k_down = args.bits_down;

  void* kernel_args[] = {
      static_cast<void*>(&hid),        static_cast<void*>(&st_g),
      static_cast<void*>(&st_u),       static_cast<void*>(&in_g),
      static_cast<void*>(&in_u),       static_cast<void*>(&out),
      static_cast<void*>(&g_tr),       static_cast<void*>(&g_su),
      static_cast<void*>(&g_sv),       static_cast<void*>(&u_tr),
      static_cast<void*>(&u_su),       static_cast<void*>(&u_sv),
      static_cast<void*>(&d_tr),       static_cast<void*>(&d_su),
      static_cast<void*>(&d_sv),       static_cast<void*>(&cnt),
      static_cast<void*>(&tok),        static_cast<void*>(&wgt),
      static_cast<void*>(&hidden_dim),  static_cast<void*>(&intermediate_dim),
      static_cast<void*>(&num_experts),
      static_cast<void*>(&max_tokens_per_expert),
      static_cast<void*>(&act_limit),   static_cast<void*>(&act_function),
      static_cast<void*>(&k_gate),      static_cast<void*>(&k_up),
      static_cast<void*>(&k_down),      static_cast<void*>(&locks)};

  const int block_dim = kBaseThreads * kMoeTilesizeK / 16;
  Check(cudaLaunchCooperativeKernel(
            kernel, dim3(static_cast<unsigned>(group_size), 1, static_cast<unsigned>(num_groups)),
            dim3(static_cast<unsigned>(block_dim)), kernel_args, kSmemMax, AsStream(q)),
        "cudaLaunchCooperativeKernel exl3_moe");
  Check(cudaGetLastError(), "exl3_moe launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kExl3HadR128, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Exl3HadR128Fn>(&Exl3HadR128KernelCuda)));
    RegisterOp(OpId::kExl3Gemm, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Exl3GemmFn>(&Exl3GemmKernelCuda)));
    RegisterOp(OpId::kExl3MoeMlp, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Exl3MoeMlpFn>(&Exl3MoeMlpKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
