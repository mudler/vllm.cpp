// EXL3 (exllamav3 trellis) fused linear — ROCm arm. Row BACKEND-ROCM-EXL3,
// issue #2433.
//
// THE DONOR IS THE PORTABLE CPU REFERENCE, NOT THE CUDA KERNEL. This is the one
// decision worth reading before the code:
//
//   src/vt/cpu/cpu_exl3_kernels.cpp   HadBlock128, ShuffleHadWarp, HadRowBlock,
//                                     Exl3GemmKernelCpu
//   src/vt/cpu/cpu_exl3_dequant.cpp   Exl3TileCodeword, Exl3DecodeCodeword,
//                                     Exl3TileRowMajorIndex, Exl3DecodeTile
//
// which are themselves 1:1 ports of exllamav3 @
// 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT) — hadamard_inner.cuh:17-44 and
// :93-279, exl3_gemm_kernel.cuh:8-80, exl3_dq.cuh:15-31, codebook.cuh:56-90,
// quantize.py:22-42. vLLM implements no EXL3 at the parity pin, so exllamav3 is
// this format's registered secondary oracle (.agents/oracles/exllamav3.md).
//
// WHY NOT src/vt/cuda/cuda_exl3.cu. Four structural blockers, of which the first
// alone settles it. The point is not that the PTX would need translating; it is
// that the kernel's geometry does not exist on this hardware.
//
//   cuda_exl3.cu:103  `constexpr int kSmemMax = 90 * 1024;`, static_asserted at
//                     :694 and installed at :1893 through
//                     cudaFuncAttributeMaxDynamicSharedMemorySize. AMD LDS is
//                     64 KiB per workgroup. The tile arithmetic at :694 is sized
//                     against 90 KiB, so no launch bound or flag recovers it.
//   :168, :1248       `mma.sync.aligned.m16n8k16`. m16n8k16 has no AMD shape —
//                     RDNA3.5 WMMA is 16x16x16 — and FragA/FragB/FragC at
//                     :151-155 are NVIDIA's fragment layout, not a width.
//   :257              `ldmatrix.sync.aligned.m8n8.x4.shared.b16`. No equivalent.
//   :241,:247,:251    `cp.async` + commit_group + wait_group, which is what the
//                     SH_STAGES pipeline IS. gfx1151 has no async global->LDS.
//
// src/vt/rocm/ is hand-written HIP ported from vLLM's ROCm csrc/, never a
// hipify of our CUDA, and this tree ships no hipify script. That is preserved.
//
// THE EXEMPLAR MIRRORED is src/vt/rocm/rocm_grouped_gemm.hip, which serves
// kMatmulBTQuant: the same "decode the weight INSIDE the GEMM" shape, the same
// bit-exact scalar codec instead of a vendor intrinsic, and the same
// grow-only per-stream scratch discipline.
//
// WHAT THIS ARM CLAIMS, AND WHY IT IS A BYTE CLAIM. All three steps run the
// same IEEE f32 operations on the same values in the SAME ORDER as
// Exl3GemmKernelCpu: the Hadamard because it is ShuffleHadWarp's five xor-partner
// levels with __syncthreads() where the host has a memcpy, and the GEMM because
// one thread owns one output element and walks `ti` then `rr` ascending, which
// is the host loop nest read down a single accumulator. So the gate on this file
// is byte equality with the CPU arm, NOT the 1.0e-3 bound the CUDA arm carries —
// that bound exists because mma.m16n8k16's internal accumulation order is
// unspecified and a split-K reduction sits on top of it, and this arm has
// neither. `-ffp-contract=off` is already set for HIP in CMakeLists.txt, which
// is what stops a compiler-generated FMA from contracting `acc += xv * w` into
// a different value.
//
// NO matrix cores, NO async copies, NO cooperative launch, and 3 KiB of LDS
// against a 64 KiB budget. The blockers above are not worked around; they are
// not encountered. Speed is a later row and this file offers none.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_f16_codec.h"

namespace vt::rocm {
namespace {

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm exl3: ") + what + ": " +
                             hipGetErrorString(err));
}
inline hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

// 1/sqrt(128) spelled as upstream spells it (hadamard.cu:107) and as
// cpu_exl3_kernels.cpp spells it. Kept as the literal and never recomputed: a
// recomputed 1/sqrt(128) can differ in the last f32 bit, and that bit is the
// difference between this file's byte gate passing and failing.
constexpr float kInvSqrt128 = 0.088388347648f;

// ─── numeric helpers, transcribed from the CPU arm ───────────────────────────

// cpu_exl3_kernels.cpp NegBySignBit. XORing the sign bit, not `0.0f - v`: the
// two differ on a zero and this arm reproduces the host operation.
__device__ inline float NegBySignBit(float v) {
  return __int_as_float(static_cast<int>(__float_as_uint(v) ^ 0x80000000u));
}

// cpu_exl3_kernels.cpp MulF16 — the fp16 multiply __hmul2 performs. The exact
// product of two fp16 significands is 22 bits and fits f32 exactly, so rounding
// once from f32 is the value an fp16 multiply produces.
__device__ inline uint16_t MulF16(uint16_t a, uint16_t b) {
  return DF32ToF16(DF16ToF32(a) * DF16ToF32(b));
}

// cpu_exl3_dequant.cpp RoundHalf.
__device__ inline float RoundHalf(float v) { return DF16ToF32(DF32ToF16(v)); }

// ─── the trellis decode (cpu_exl3_dequant.cpp) ───────────────────────────────

// The tile's uint32 view (exl3_dq.cuh:25-26), assembled by hand so the trellis
// may sit at any alignment inside the loaded buffer.
__device__ inline uint32_t TileWord32(const uint16_t* tile, int index) {
  return static_cast<uint32_t>(tile[2 * index]) |
         (static_cast<uint32_t>(tile[2 * index + 1]) << 16);
}

// Exl3TileCodeword, verbatim. `+ 256*bits` keeps the tail-biting wrap
// non-negative; the `% words32` on the word index is the wrap itself.
__device__ inline uint16_t TileCodeword(const uint16_t* tile, int bits, int t) {
  const int words32 = bits * 256 / 32;
  const int b0 = t * bits + bits - 16 + 256 * bits;
  const int b1 = b0 + 16;
  const int i0 = b0 / 32;
  const int i1 = (b1 - 1) / 32;
  const int s0 = (i1 + 1) * 32 - b1;
  const uint32_t a = TileWord32(tile, i0 % words32);
  const uint32_t b = TileWord32(tile, i1 % words32);
  const uint64_t merged = (static_cast<uint64_t>(a) << 32) | b;
  return static_cast<uint16_t>((merged >> s0) & 0xffffu);
}

// Exl3DecodeCodeword (codebook.cuh:56-90). All THREE codebooks, because an AMD
// box has no reason to see fewer artifacts than an NVIDIA one:
//   cb 0  3INST, the DEFAULT — a checkpoint that ships neither an `mcg` nor a
//         `mul1` marker lands here, which is every stock turboderp/*-exl3
//   cb 1  MCG, which the SparkInfer DeepSeek-V4 artifact marks
//   cb 2  mul1, a DIFFERENT SHAPE and not a third multiplier: the product's four
//         UNSIGNED BYTES are summed into a fixed accumulator, the sum is
//         REINTERPRETED as an fp16 bit pattern, and an fp16 affine map turns it
//         into the codebook value
// `codebook` is validated on the host before launch, so there is no fall-off arm
// here; a device VT_CHECK does not exist and a silent wrong decode is exactly
// the failure this family documents (right distribution, no correlation).
__device__ inline float DecodeCodeword(uint16_t codeword, int codebook) {
  uint32_t x = static_cast<uint32_t>(codeword);
  if (codebook == 2) {
    x *= 0x83DCD12Du;
    // `__dp4a(x, 0x01010101u, acc)` == acc + the four UNSIGNED bytes of x.
    // Scalar here for the same reason rocm_grouped_gemm.hip's Dp4a is scalar:
    // integer arithmetic is exact either way, and v_dot4 is a perf lever, not a
    // correctness requirement.
    const uint32_t byte_sum = (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) +
                              ((x >> 24) & 0xffu);
    // 0x6400 is chosen so the reinterpretation is EXACT: the fp16 binade
    // [1024, 2048) has an ULP of exactly 1.0 and the byte sum is at most 1020,
    // so the pattern never leaves that binade.
    const uint32_t sum = 0x6400u + byte_sum;
    const float h = DF16ToF32(static_cast<uint16_t>(sum));
    // BIT PATTERNS, not the rounded decimals upstream's comments carry: 0x1eee
    // is 887/131072 and 0xc931 is -1329/128.
    const float k_inv = DF16ToF32(static_cast<uint16_t>(0x1eeeu));
    const float k_bias = DF16ToF32(static_cast<uint16_t>(0xc931u));
    // Upstream ends in __hfma, ONE rounding. Evaluating the product and the sum
    // separately in f32 reproduces it exactly (the CPU arm proves why over all
    // 1021 reachable sums), and -ffp-contract=off keeps the compiler from
    // fusing them into a different single rounding.
    return RoundHalf(h * k_inv + k_bias);
  }
  if (codebook == 0) {
    x *= 89226354u;
    x += 64248484u;
  } else {
    x *= 0xCBAC1FEDu;
  }
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  const float lo = DF16ToF32(static_cast<uint16_t>(x & 0xffffu));
  const float hi = DF16ToF32(static_cast<uint16_t>(x >> 16));
  return RoundHalf(lo + hi);  // __hadd: the sum is taken in fp16
}

// Exl3TileRowMajorIndex (quantize.py:28-42). `t / 8` is the tensor-core lane,
// `t % 8` its eight fragment slots. The host's kRowOffset[8] = {0,1,8,9,0,1,8,9}
// is written as arithmetic here so no constant array lands in scratch.
__device__ inline int TileRowMajorIndex(int t) {
  const int lane = t >> 3;
  const int sub = t & 7;
  const int q = sub & 3;
  const int row_offset = q < 2 ? q : q + 6;  // 0,1,8,9
  const int r = (lane % 4) * 2 + row_offset;
  const int c = lane / 4 + (sub < 4 ? 0 : 8);
  return r * 16 + c;
}

// ─── step 1 and step 3: had_r_128 ────────────────────────────────────────────
//
// One 32-lane group per 128-block, four groups per workgroup, `h[4][32]` in LDS.
// Lane t holds the four values of columns 4t..4t+3, so lane bit `i` carries
// Hadamard level 4*i and levels 4..64 are exactly the five xor-partner steps —
// which is why this is ShuffleHadWarp and not a rewritten butterfly.
//
// IN-PLACE IS SAFE, and by construction rather than by luck: every lane loads
// its four inputs BEFORE the first barrier and stores AFTER the last one, so no
// element is written while any lane may still read it. Upstream says the same of
// its own kernel ("Works inplace if y == x", hadamard.cu:86) and Exl3Gemm's
// contract permits `a_had` to alias `a`.
template <bool HalfIn, bool HalfOut>
__global__ void HadK(void* __restrict__ out, const void* __restrict__ in,
                     const uint16_t* __restrict__ pre, const uint16_t* __restrict__ post,
                     float r_scale, int64_t nblocks, int64_t blocks_per_row, int64_t cols,
                     int64_t total_groups) {
  __shared__ float sh[4][4][32];  // [group][j][lane] — 2 KiB
  const int ty = static_cast<int>(threadIdx.y);
  const int t = static_cast<int>(threadIdx.x);

  for (int64_t g = blockIdx.x; g < total_groups; g += gridDim.x) {
    const int64_t gb = g * 4 + ty;
    const bool live = gb < nblocks;
    const int64_t row = live ? gb / blocks_per_row : 0;
    const int64_t blk = live ? gb % blocks_per_row : 0;
    const int64_t base = row * cols + blk * 128;
    const int64_t sbase = blk * 128;  // the scale vectors are indexed by COLUMN

    // load + pre-scale (hadamard_inner.cuh:109-114 for the half arm, :167-174
    // for the float arm: an fp16 multiply BEFORE the widen, versus a widen and
    // an f32 multiply).
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int64_t off = base + 4 * t + j;
      const int64_t soff = sbase + 4 * t + j;
      if (!live) {
        v[j] = 0.0f;
      } else if (HalfIn) {
        const uint16_t* p = static_cast<const uint16_t*>(in);
        const uint16_t w = pre != nullptr ? MulF16(p[off], pre[soff]) : p[off];
        v[j] = DF16ToF32(w);
      } else {
        const float* p = static_cast<const float*>(in);
        v[j] = pre != nullptr ? p[off] * DF16ToF32(pre[soff]) : p[off];
      }
    }

    // levels 1 and 2, on the four values this lane holds
    // (hadamard_inner.cuh:118-129).
    const float s0 = v[0] + v[1];
    const float d0 = v[0] - v[1];
    const float s1 = v[2] + v[3];
    const float d1 = v[2] - v[3];
    __syncthreads();  // every load above is now complete, so an aliased write is safe
    sh[ty][0][t] = s0 + s1;
    sh[ty][1][t] = d0 + d1;
    sh[ty][2][t] = s0 - s1;
    sh[ty][3][t] = d0 - d1;

    // levels 4..64: ShuffleHadWarp, five xor-partner steps.
    for (int i = 1; i < 32; i <<= 1) {
      __syncthreads();
      const int p = t ^ i;
      const bool flip = (t & i) != 0;
      float own[4], par[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        own[j] = flip ? NegBySignBit(sh[ty][j][t]) : sh[ty][j][t];
        par[j] = sh[ty][j][p];
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 4; ++j) sh[ty][j][t] = own[j] + par[j];  // own FIRST, as the host adds
    }
    __syncthreads();

    // r_scale, then the store and the post-scale. Both half-output inners round
    // FIRST and apply the post scale as an fp16 multiply afterwards
    // (hadamard_inner.cuh:137-146); the float arms multiply in f32.
    if (live) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float res = sh[ty][j][t] * r_scale;
        const int64_t off = base + 4 * t + j;
        const int64_t soff = sbase + 4 * t + j;
        if (HalfOut) {
          uint16_t* o = static_cast<uint16_t*>(out);
          const uint16_t r = DF32ToF16(res);
          o[off] = post != nullptr ? MulF16(r, post[soff]) : r;
        } else {
          float* o = static_cast<float*>(out);
          o[off] = post != nullptr ? res * DF16ToF32(post[soff]) : res;
        }
      }
    }
    __syncthreads();  // the next `g` overwrites sh
  }
}

template <bool HalfIn, bool HalfOut>
void LaunchHad(void* out, const void* in, const uint16_t* pre, const uint16_t* post,
               float r_scale, int64_t rows, int64_t cols, hipStream_t st) {
  const int64_t blocks_per_row = cols / 128;
  const int64_t nblocks = rows * blocks_per_row;
  if (nblocks <= 0) return;
  const int64_t total_groups = (nblocks + 3) / 4;
  const unsigned grid =
      static_cast<unsigned>(total_groups > 65535 ? 65535 : total_groups);
  HadK<HalfIn, HalfOut><<<grid, dim3(32, 4), 0, st>>>(out, in, pre, post, r_scale, nblocks,
                                                      blocks_per_row, cols, total_groups);
  Check(hipGetLastError(), "had_r_128");
}

// ─── step 2: the tiled GEMM against the decoded trellis ──────────────────────
//
// Exl3GemmKernelCpu step 2, with its loop nest read down a SINGLE accumulator.
// One workgroup owns one 16-column output tile and up to 16 rows; one thread
// owns one output element. Per k-tile the 256 threads decode the 256 codewords
// of that tile into LDS — which is Exl3DecodeTile with its `t` loop unrolled
// across threads, and the ONLY place a thread reads a codeword it does not own.
//
// THE ACCUMULATION ORDER IS THE HOST'S: `ti` ascending in the outer loop, `rr`
// ascending inside, into one f32 register. The host's `if (xv == 0.0f) continue`
// is kept and is not an optimisation here — with acc == -0.0f, adding a +0.0f
// product flips the sign of the zero, so dropping the guard would make this arm
// differ from its own reference on exactly the values a byte gate exists to
// catch.
__global__ void Exl3GemmK(float* __restrict__ raw, const uint16_t* __restrict__ ah,
                          const uint16_t* __restrict__ trellis, int64_t m, int64_t k,
                          int64_t n, int bits, int codebook) {
  __shared__ float tile[256];  // 1 KiB
  const int64_t tiles_n = n / 16;
  const int64_t tile_words = 16 * static_cast<int64_t>(bits);
  const int64_t tj = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y) * 16 + tid / 16;
  const int cc = tid % 16;
  const bool live = row < m;

  float acc = 0.0f;
  for (int64_t ti = 0; ti < k / 16; ++ti) {
    const uint16_t* tp = trellis + (ti * tiles_n + tj) * tile_words;
    __syncthreads();  // the previous iteration's readers are done with `tile`
    tile[TileRowMajorIndex(tid)] = DecodeCodeword(TileCodeword(tp, bits, tid), codebook);
    __syncthreads();
    if (live) {
      const uint16_t* arow = ah + row * k + ti * 16;
      for (int rr = 0; rr < 16; ++rr) {
        const float xv = DF16ToF32(arow[rr]);
        if (xv == 0.0f) continue;
        acc += xv * tile[rr * 16 + cc];
      }
    }
  }
  if (live) raw[row * n + tj * 16 + cc] = acc;
}

// ─── the f32 staging buffer ──────────────────────────────────────────────────
//
// The CPU arm holds `raw` in a std::vector; a device needs the same [m, n] f32
// between step 2 and step 3. Grow-only, per stream, allocated stream-ordered and
// never freed — the discipline rocm_grouped_gemm.hip's EnsureQuantScratch
// documents, and for its reason: hipMalloc/hipFree/hipStreamSynchronize per call
// are illegal under hipGraph stream capture, and a captured graph may have baked
// the pointer. Bounded: it grows O(log(max/min)) times over a process.
//
// NAMED AS A COST rather than hidden. A fused kernel would not need it. A fused
// kernel is the later speed row, and this one buys the end of the reference tier.
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

float* EnsureRawScratch(size_t need, hipStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.bytes) {
    Check(hipMallocAsync(&sc.buf, need, s), "raw scratch grow");
    sc.bytes = need;
  }
  return static_cast<float*>(sc.buf);
}

}  // namespace

// The fused chain, step for step as Exl3GemmKernelCpu runs it:
//   1. A_had = had_r_128(A, pre_scale = suh)
//   2. C_raw = A_had @ reconstruct(trellis)        [f32 accumulation]
//   3. C     = had_r_128(C_raw, post_scale = svh)
// Argument validation (shapes, dtypes, contiguity, the 16/128 multiples) is done
// once in `vt::Exl3Gemm` for every backend and is not repeated here. What IS
// checked here is what selects device code: `bits` and `codebook` reach the
// kernel as values, and an out-of-range one would decode silently rather than
// throw, because a device has no VT_CHECK.
void Exl3GemmKernelRocm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                        const Tensor& suh, const Tensor& svh, Tensor& a_had,
                        const Exl3GemmArgs& args) {
  EnsureQueueDevice(q);
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = c.shape[1];
  if (m == 0 || k == 0 || n == 0) return;
  VT_CHECK(args.bits >= 1 && args.bits <= 8,
           "vt rocm exl3: bits must be in [1, 8]; got " + std::to_string(args.bits));
  VT_CHECK(args.codebook >= 0 && args.codebook <= 2,
           "vt rocm exl3: codebook " + std::to_string(args.codebook) +
               " is not implemented (0 == 3INST, 1 == MCG, 2 == mul1). Upstream defines "
               "no other value: `decode_3inst<cb>` (codebook.cuh:56-90) has arms for 0, "
               "1 and 2 and falls off the end for anything else.");

  hipStream_t st = AsStream(q);

  // 1. the input transform, into the caller's scratch (which may alias A).
  LaunchHad<true, true>(a_had.data, a.data, suh.Ptr<uint16_t>(), nullptr, kInvSqrt128, m, k,
                        st);

  // 2. the matmul against the decoded trellis, f32 accumulators.
  float* raw = EnsureRawScratch(static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                                st);
  const unsigned gx = static_cast<unsigned>(n / 16);
  const unsigned gy = static_cast<unsigned>((m + 15) / 16);
  Exl3GemmK<<<dim3(gx, gy), 256, 0, st>>>(raw, a_had.Ptr<uint16_t>(),
                                          trellis.Ptr<uint16_t>(), m, k, n, args.bits,
                                          args.codebook);
  Check(hipGetLastError(), "exl3_gemm");

  // 3. the output transform — had_ff for an f32 C, had_fh for an fp16 one, the
  // same two arms the CPU reference picks between off `c.dtype`.
  if (c.dtype == DType::kF32)
    LaunchHad<false, false>(c.data, raw, nullptr, svh.Ptr<uint16_t>(), kInvSqrt128, m, n, st);
  else
    LaunchHad<false, true>(c.data, raw, nullptr, svh.Ptr<uint16_t>(), kInvSqrt128, m, n, st);
}

}  // namespace vt::rocm
