// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA baseline kernels for rmsnorm / silu_and_mul / embedding / rope_neox.
// Correctness-grade (M0.6): plain grid-stride / one-block-per-row kernels, f32
// accumulation, double-precision RoPE angles matching the CPU reference.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cub/cub.cuh>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "vt/cuda/rmsnorm_decode_fast.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

// f32 load/store overloads: bf16 converts on the way in/out, math is f32.
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, same as host F32ToBF16
}

// Round-trip a f32 value through the residual store dtype so the variance below
// squares the SAME rounded value that gets written back to the residual stream —
// mirrors vLLM fused_add_rms_norm, whose bf16 residual add (`z += residual`) rounds
// to the model dtype before the f32 variance (layernorm_kernels.cu). Identity for a
// f32 residual, so the previous f32-residual path stays byte-for-byte unchanged.
template <typename Tres> __device__ inline float ResRound(float v);
template <> __device__ inline float ResRound<float>(float v) { return v; }
template <> __device__ inline float ResRound<__nv_bfloat16>(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

// ---------------------------------------------------------------------------
// rmsnorm: one block per row, shared-memory f32 tree reduction.
// Upstream csrc counterpart: csrc/layernorm_kernels.cu (rms_norm_kernel / fused_add_rms_norm_kernel) — align signatures post-MVP.

template <typename Tin, typename Tout, typename Tres>
__global__ void RmsNormRowKernel(Tout* out, const Tin* x, const Tin* w, Tres* residual,
                                 int64_t h, float eps, bool gemma) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * h;
  Tout* orow = out + row * h;
  Tres* rrow = residual == nullptr ? nullptr : residual + row * h;

  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float v = Load(xrow, j);
    if (rrow != nullptr) {
      v = ResRound<Tres>(v + Load(rrow, j));  // new residual stream: f32 add, round to Tres
      Store(rrow, j, v);                       // updated in place (f32 or bf16)
    }
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v = rrow != nullptr ? Load(rrow, j) : Load(xrow, j);
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(orow, j, v * inv * wj);
  }
}

// ---------------------------------------------------------------------------
// Decode-fast rmsnorm variant (VT_RMSNORM_DECODE_FAST). Re-expressed 2026-07-17
// (KERNEL-EW-NORM-ACT numerics rework, CLAIM-EW-NORM-ACT-2) to be BIT-IDENTICAL
// to the shipped RmsNormRowKernel above (the through-stack 235/235 bit-reference
// that matches vLLM's production greedy stream), not merely ≤1-ulp close.
//
// WHY BIT-IDENTICAL (not ≤1-ulp): the 27B greedy token 6 is a RAZOR near-tie
// (198 "\n" prod vs 271 "\n\n" emu; the logit gap is ~zero). The shipped
// RmsNormRowKernel + GDN cubin = 235/235 (198). The prior fast kernel differed
// from shipped by ≤1 ulp (residual-add rounding + a different variance reduction
// ORDER + rsqrtf); accumulated over 64 layers alongside the GDN cubin's own
// ≤1-ulp perturbation, that tipped the tie to 271 (233/235), forcing the default
// back OFF (a875397). Making the fast output the SAME BITS as shipped removes the
// perturbation by construction: fast+cubin ≡ shipped+cubin ≡ 198 always.
//
// The three divergences vs shipped RmsNormRowKernel (cuda_ops.cu:62-93) and the
// fix that makes each bit-exact:
//   (1) residual add. shipped:62-79 does v = ResRound<bf16>(f32(x)+f32(res)) — a
//       f32 add of the two bf16 operands then a SINGLE round to bf16 (double
//       rounding through f32). The prior fast used __hadd2 (single-round bf16
//       add), which differs from the shipped double-round on rare carries. Fixed:
//       __float2bfloat16(f32(x)+f32(res)) reproduces ResRound exactly.
//   (2) variance sum ORDER. shipped:70-85 uses kBlock=256 threads, per-thread
//       partial acc = Σ_m sq[tid + 256*m] (increasing m), then the shared-memory
//       binary tree `for s=128; s>0; s>>=1`. f32 add is non-associative, so the
//       sum's bits depend on this exact 256-partial + tree structure. The prior
//       fast used cub::BlockReduce<float,1024> over 1024 threads (a different
//       thread count AND a different reduction tree) => a different f32 variance.
//       Fixed: this kernel launches with kBlock(=256) threads and reproduces
//       shipped's scalar-strided Pass 1 and shared-tree byte-for-byte.
//   (3) inv. shipped:86 uses `1.0f / sqrtf(...)` (correctly-rounded sqrt+div);
//       the prior fast used rsqrtf (a ≤2-ulp reciprocal-sqrt approximation).
//       Fixed: `1.0f / sqrtf(partial[0]/h + eps)` verbatim.
//
// The ONLY thing that legitimately differs from shipped — and the whole source of
// any speedup — is Pass 2 (normalize): out = bf16((f32(res)*inv)*(f32(w)+gemma))
// is ELEMENT-INDEPENDENT, so vectorizing it with 16-byte (4×bf162) loads/stores
// changes memory traffic, NOT the arithmetic, and stays bit-identical. Pass 1
// (the variance) is order-sensitive and is kept byte-for-byte shipped.
//
// Scope: bf16-in / bf16-out / bf16-residual, H%8==0, H>=1024 (the 129
// input/post-attn/final residual RMSNorm launches, H=5120 on the 27B / H=2048 on
// the 35B). Other dtype/residual/small-H keep RmsNormRowKernel. The q/k head norms
// take no residual => not this path. 16-byte load = 4 packed bf162.
struct alignas(16) RmsNormBf16x8 {
  __nv_bfloat162 d[4];
};

// Launch geometry. The variance REDUCTION is byte-for-byte the shipped
// RmsNormRowKernel's (kBlock=256 partials + tree), but the memory passes run with
// kFastBlock=1024 threads: at decode the RMSNorm launches only `rows` (= num
// decode tokens, ~16 at c16) blocks, so the GPU is block-starved and thread-level
// parallelism per block — not occupancy — hides the memory latency. That
// thread-count is the ORIGINAL fast kernel's win; this rework keeps it while
// making the arithmetic bit-identical to shipped. kFastMaxH bounds the f32
// square-buffer below (guard rejects larger H); 8192 covers the 27B H=5120 / 35B
// H=2048 decode norms with headroom, at 32 KB static shared (< the 48 KB no-opt-in
// cap, and free here because only ~16 blocks are resident).
constexpr int kFastBlock = 1024;
constexpr int kFastMaxH = 8192;

// One block per row. Pass 1 (vectorized, kFastBlock threads): residual =
// bf16(f32(x)+f32(res)) [== shipped ResRound], stored bf16, with each element's
// f32 square written to the shared buffer ssq. Reduction (kBlock=256 threads):
// p_i = Σ_m ssq[i + 256*m] then the shared-memory binary tree — the EXACT shipped
// summation ORDER (cuda_ops.cu:70-86), reading the same bf16-rounded squares, so
// the f32 variance is bit-identical despite the vectorized loads. inv =
// 1.0f/sqrtf (shipped:86, not rsqrtf). Pass 2 (vectorized): normalize — element-
// independent, so identical bits at higher bandwidth. Every op matches shipped
// bit-for-bit; only the memory access WIDTH and thread COUNT differ.
__global__ void RmsNormRowFastKernel(__nv_bfloat16* __restrict__ out,
                                     const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     __nv_bfloat16* __restrict__ residual, int h, float eps,
                                     bool gemma) {
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t base = static_cast<int64_t>(blockIdx.x) * h;
  const int vh = h / 8;
  const RmsNormBf16x8* xv = reinterpret_cast<const RmsNormBf16x8*>(x + base);
  RmsNormBf16x8* rv = reinterpret_cast<RmsNormBf16x8*>(residual + base);
  const RmsNormBf16x8* wv = reinterpret_cast<const RmsNormBf16x8*>(w);
  RmsNormBf16x8* ov = reinterpret_cast<RmsNormBf16x8*>(out + base);

  __shared__ float ssq[kFastMaxH];   // per-element v^2 (v = the bf16-rounded residual)
  __shared__ float partial[kBlock];  // 256 partials for shipped's exact tree order

  // Pass 1 — vectorized residual add + store + per-element square into ssq. The
  // residual add is bf16(f32(x)+f32(res)) (== ResRound<bf16>, shipped:75) and the
  // square is of that bf16-rounded value (shipped:78), so ssq[j] is byte-for-byte
  // shipped's v*v term for every element.
  for (int vi = tid; vi < vh; vi += kFastBlock) {
    RmsNormBf16x8 t = xv[vi];
    RmsNormBf16x8 r = rv[vi];
    RmsNormBf16x8 nr;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 xf = __bfloat1622float2(t.d[k]);
      float2 rf = __bfloat1622float2(r.d[k]);
      nr.d[k] = __floats2bfloat162_rn(xf.x + rf.x, xf.y + rf.y);  // per-lane == __float2bfloat16
      float2 nf = __bfloat1622float2(nr.d[k]);                    // bf16 value back to f32
      ssq[vi * 8 + 2 * k] = nf.x * nf.x;
      ssq[vi * 8 + 2 * k + 1] = nf.y * nf.y;
    }
    rv[vi] = nr;  // residual store (bf16), bit-identical to shipped
  }
  __syncthreads();  // publish ssq (and residual) before the strided reduction reads them

  // Reduction — BYTE-FOR-BYTE shipped (cuda_ops.cu:80-86). Threads >=256 idle.
  if (tid < kBlock) {
    float acc = 0.0f;
    for (int j = tid; j < h; j += kBlock) acc += ssq[j];  // p_i = Σ_m ssq[i+256m], increasing m
    partial[tid] = acc;
  }
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);

  // Pass 2 — vectorized normalize (bit-identical to shipped:87-92; each output
  // element is (f32(res)*inv)*(f32(w)+gemma) rounded to bf16, independent of the
  // others). Each thread reloads the residual vector it wrote in Pass 1.
  for (int vi = tid; vi < vh; vi += kFastBlock) {
    RmsNormBf16x8 r = rv[vi];
    RmsNormBf16x8 wr = wv[vi];
    RmsNormBf16x8 o;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 rf = __bfloat1622float2(r.d[k]);
      float2 wf = __bfloat1622float2(wr.d[k]);
      float w0 = wf.x, w1 = wf.y;
      if (gemma) {
        w0 += 1.0f;
        w1 += 1.0f;
      }
      o.d[k] = __floats2bfloat162_rn(rf.x * inv * w0, rf.y * inv * w1);
    }
    ov[vi] = o;
  }
}

// Runtime predicate + launch for the decode-fast path. Returns true iff it ran.
// Guard: bf16 in/out/weight/residual, 16-byte-aligned pointers (vectorized loads),
// H%8==0, 1024<=H<=kFastMaxH (H>=1024 scopes to the big residual RMSNorm launches;
// H<=kFastMaxH bounds the f32 square-buffer shared array). The launch uses
// kFastBlock threads for the memory passes, while the reduction reproduces the
// shipped RmsNormRowKernel's kBlock=256 partial + tree order, so the output is
// bit-identical. Out-of-scope shapes keep RmsNormRowKernel.
inline bool TryLaunchRmsNormDecodeFast(cudaStream_t s, Tensor& out, const Tensor& x,
                                       const Tensor& w, const RmsNormArgs& args,
                                       Tensor* residual, unsigned rows, int64_t h) {
  if (!RmsNormDecodeFastFlagIsOn(std::getenv("VT_RMSNORM_DECODE_FAST"))) return false;
  if (out.dtype != DType::kBF16 || x.dtype != DType::kBF16 || w.dtype != DType::kBF16)
    return false;
  if (residual == nullptr || residual->dtype != DType::kBF16) return false;
  if (h % 8 != 0 || h < 1024 || h > kFastMaxH) return false;
  auto aligned16 = [](const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 0xF) == 0;
  };
  if (!aligned16(out.data) || !aligned16(x.data) || !aligned16(w.data) ||
      !aligned16(residual->data))
    return false;
  RmsNormRowFastKernel<<<rows, kFastBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<__nv_bfloat16>(),
                                             w.Ptr<__nv_bfloat16>(),
                                             residual->Ptr<__nv_bfloat16>(),
                                             static_cast<int>(h), args.eps, args.gemma);
  return true;
}

// Dispatch the residual store dtype (f32 or bf16). A bf16 residual mirrors vLLM's
// bf16 model dtype (model_config.dtype=bfloat16): the residual stream is bf16, only
// the variance/normalize accumulation below stays f32. A f32 residual (or none)
// takes the byte-identical previous path.
template <typename Tin, typename Tout>
void LaunchRmsNormRes(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const RmsNormArgs& args, Tensor* residual, unsigned rows, int64_t h) {
  if (residual != nullptr && residual->dtype == DType::kBF16) {
    RmsNormRowKernel<Tin, Tout, __nv_bfloat16><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), residual->Ptr<__nv_bfloat16>(), h,
        args.eps, args.gemma);
  } else {
    float* res = residual == nullptr ? nullptr : residual->Ptr<float>();
    RmsNormRowKernel<Tin, Tout, float><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), res, h, args.eps, args.gemma);
  }
}

template <typename Tin>
void LaunchRmsNorm(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  const unsigned rows = static_cast<unsigned>(t);
  // Decode-fast path (VT_RMSNORM_DECODE_FAST, default ON; '0' = rollback): only
  // engages for the bf16 add+RMSNorm decode launches; every other case keeps
  // RmsNormRowKernel. Output is bit-identical to RmsNormRowKernel by construction.
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    if (TryLaunchRmsNormDecodeFast(s, out, x, w, args, residual, rows, h)) {
      Check(cudaGetLastError(), "rmsnorm fast launch");
      return;
    }
  }
  switch (out.dtype) {
    case DType::kF32:
      LaunchRmsNormRes<Tin, float>(s, out, x, w, args, residual, rows, h);
      break;
    case DType::kBF16:
      LaunchRmsNormRes<Tin, __nv_bfloat16>(s, out, x, w, args, residual, rows, h);
      break;
    default: VT_CHECK(false, "cuda rmsnorm: unsupported out dtype");
  }
  Check(cudaGetLastError(), "rmsnorm launch");
}

void RmsNormKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual) {
  VT_CHECK(w.dtype == x.dtype, "cuda rmsnorm: weight dtype must match x");
  switch (x.dtype) {
    case DType::kF32: LaunchRmsNorm<float>(AsStream(q), out, x, w, args, residual); break;
    case DType::kBF16:
      LaunchRmsNorm<__nv_bfloat16>(AsStream(q), out, x, w, args, residual);
      break;
    default: VT_CHECK(false, "cuda rmsnorm: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// rmsnorm + static fp8 quant, fused: emit the fp8 activation (and optionally the
// bf16 normed activation) directly from the RMSNorm's normalize loop, so the
// standalone QuantFp8Static pass + its bf16 round-trip disappear. Mirror of
// vLLM's Inductor fused_add_rms_norm_static_fp8_quant
// (vllm/compilation/passes/fusion/rms_quant_fusion.py:124). Same reduction as
// RmsNormRowKernel; BIT-IDENTICAL to RmsNorm(bf16)+QuantFp8Static because the
// fp8 is taken from the SAME bf16-rounded value the split path quantizes
// (F32ToFp8Dev(__bfloat162float(bf16(n)) * inv_scale)).
__device__ __forceinline__ uint8_t RmsNormF32ToFp8Dev(float f) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
}

template <typename Tin, typename Tres>
__global__ void RmsNormQuantFp8RowKernel(uint8_t* out_fp8, __nv_bfloat16* out_bf16, const Tin* x,
                                         const Tin* w, Tres* residual, int64_t h, float eps,
                                         bool gemma, float inv_scale) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * h;
  uint8_t* orow = out_fp8 + row * h;
  __nv_bfloat16* brow = out_bf16 == nullptr ? nullptr : out_bf16 + row * h;
  Tres* rrow = residual == nullptr ? nullptr : residual + row * h;

  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float v = Load(xrow, j);
    if (rrow != nullptr) {
      v = ResRound<Tres>(v + Load(rrow, j));  // new residual stream: f32 add, round to Tres
      Store(rrow, j, v);                       // updated in place (f32 or bf16)
    }
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v = rrow != nullptr ? Load(rrow, j) : Load(xrow, j);
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    // bf16-intermediate (matches RmsNorm's bf16 store then QuantFp8Static's bf16 load).
    const __nv_bfloat16 nb = __float2bfloat16(v * inv * wj);
    if (brow != nullptr) brow[j] = nb;
    orow[j] = RmsNormF32ToFp8Dev(__bfloat162float(nb) * inv_scale);
  }
}

template <typename Tin>
void LaunchRmsNormQuantFp8(cudaStream_t s, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                           const Tensor& w, const RmsNormArgs& args, Tensor* residual,
                           float input_scale) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  const unsigned rows = static_cast<unsigned>(t);
  const float inv_scale = 1.0f / input_scale;
  __nv_bfloat16* bf16 = out_bf16 == nullptr ? nullptr : out_bf16->Ptr<__nv_bfloat16>();
  if (residual != nullptr && residual->dtype == DType::kBF16) {
    RmsNormQuantFp8RowKernel<Tin, __nv_bfloat16><<<rows, kBlock, 0, s>>>(
        out_fp8.Ptr<uint8_t>(), bf16, x.Ptr<Tin>(), w.Ptr<Tin>(),
        residual->Ptr<__nv_bfloat16>(), h, args.eps, args.gemma, inv_scale);
  } else {
    float* res = residual == nullptr ? nullptr : residual->Ptr<float>();
    RmsNormQuantFp8RowKernel<Tin, float><<<rows, kBlock, 0, s>>>(
        out_fp8.Ptr<uint8_t>(), bf16, x.Ptr<Tin>(), w.Ptr<Tin>(), res, h, args.eps, args.gemma,
        inv_scale);
  }
  Check(cudaGetLastError(), "rmsnorm_quant_fp8 launch");
}

void RmsNormQuantFp8KernelCuda(Queue& q, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                               const Tensor& w, const RmsNormArgs& args, Tensor* residual,
                               float input_scale) {
  VT_CHECK(w.dtype == x.dtype, "cuda rmsnorm_quant_fp8: weight dtype must match x");
  switch (x.dtype) {
    case DType::kF32:
      LaunchRmsNormQuantFp8<float>(AsStream(q), out_fp8, out_bf16, x, w, args, residual,
                                   input_scale);
      break;
    case DType::kBF16:
      LaunchRmsNormQuantFp8<__nv_bfloat16>(AsStream(q), out_fp8, out_bf16, x, w, args, residual,
                                           input_scale);
      break;
    default: VT_CHECK(false, "cuda rmsnorm_quant_fp8: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// silu_and_mul: grid-stride over the T*D output elements.
// Upstream csrc counterpart: csrc/activation_kernels.cu (act_and_mul_kernel<silu>) — align post-MVP.

template <typename Tin, typename Tout>
__global__ void SiluAndMulKernel(Tout* out, const Tin* x, int64_t n, int64_t d) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t i = idx / d;
    const int64_t j = idx - i * d;
    const float gate = Load(x, i * 2 * d + j);
    const float up = Load(x, i * 2 * d + d + j);
    const float silu = gate / (1.0f + expf(-gate));
    Store(out, idx, silu * up);
  }
}

template <typename Tin>
void LaunchSiluAndMul(cudaStream_t s, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  const int64_t n = t * d;
  if (n == 0) return;
  switch (out.dtype) {
    case DType::kF32:
      SiluAndMulKernel<Tin, float>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<Tin>(), n, d);
      break;
    case DType::kBF16:
      SiluAndMulKernel<Tin, __nv_bfloat16>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<Tin>(), n, d);
      break;
    default: VT_CHECK(false, "cuda silu_and_mul: unsupported out dtype");
  }
  Check(cudaGetLastError(), "silu_and_mul launch");
}

void SiluAndMulKernelCuda(Queue& q, Tensor& out, const Tensor& x) {
  switch (x.dtype) {
    case DType::kF32: LaunchSiluAndMul<float>(AsStream(q), out, x); break;
    case DType::kBF16: LaunchSiluAndMul<__nv_bfloat16>(AsStream(q), out, x); break;
    default: VT_CHECK(false, "cuda silu_and_mul: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// embedding: grid-stride gather. Ids live on the device, so bounds are checked
// in-kernel: bad ids are clamped for the gather (no OOB read) and the first bad
// id is recorded in a device-side flag via atomicCAS. The host wrapper
// synchronizes the stream, reads the flag back, and throws — CUDA Embedding is
// synchronizing for now (M0.6 decision, see ops.h; revisit for full async in
// M0.9/M2).
// No direct csrc counterpart (upstream uses torch embedding); keep vt-native.

struct EmbeddingErr {
  int status;    // 0 = ok, 1 = bad id recorded
  int pad;       // keep `id` naturally aligned
  long long id;  // first out-of-range id seen (valid when status != 0)
};

template <typename Tin, typename Tout, typename Tid>
__global__ void EmbeddingKernel(Tout* out, const Tin* table, const Tid* ids, int64_t n,
                                int64_t h, int64_t v, EmbeddingErr* err) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t i = idx / h;
    const int64_t j = idx - i * h;
    int64_t id = static_cast<int64_t>(ids[i]);
    if (id < 0 || id >= v) {
      if (atomicCAS(&err->status, 0, 1) == 0) err->id = static_cast<long long>(id);
      id = id < 0 ? 0 : v - 1;  // clamp: keep the gather in-bounds
    }
    Store(out, idx, Load(table, id * h + j));
  }
}

template <typename Tin, typename Tout>
cudaError_t LaunchEmbedding(cudaStream_t s, Tensor& out, const Tensor& table,
                            const Tensor& ids, EmbeddingErr* err) {
  const int64_t t = ids.shape[0], h = table.shape[1], v = table.shape[0];
  const int64_t n = t * h;
  if (ids.dtype == DType::kI32) {
    EmbeddingKernel<Tin, Tout, int32_t><<<GridFor(n), kBlock, 0, s>>>(
        out.Ptr<Tout>(), table.Ptr<Tin>(), ids.Ptr<int32_t>(), n, h, v, err);
  } else {
    EmbeddingKernel<Tin, Tout, int64_t><<<GridFor(n), kBlock, 0, s>>>(
        out.Ptr<Tout>(), table.Ptr<Tin>(), ids.Ptr<int64_t>(), n, h, v, err);
  }
  return cudaGetLastError();
}

template <typename Tin>
cudaError_t LaunchEmbeddingIn(cudaStream_t s, Tensor& out, const Tensor& table,
                              const Tensor& ids, EmbeddingErr* err) {
  if (out.dtype == DType::kF32) return LaunchEmbedding<Tin, float>(s, out, table, ids, err);
  return LaunchEmbedding<Tin, __nv_bfloat16>(s, out, table, ids, err);
}

void EmbeddingKernelCuda(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids) {
  // Validate dtypes before allocating the flag buffer so a throw cannot leak it.
  VT_CHECK(table.dtype == DType::kF32 || table.dtype == DType::kBF16,
           "cuda embedding: unsupported table dtype (f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "cuda embedding: unsupported out dtype");
  const int64_t n = ids.shape[0] * table.shape[1];
  if (n == 0) return;
  // v == 0 with nonempty ids can never gather anything valid, and the in-kernel
  // clamp (v - 1) would go out of bounds — throw loudly before launching.
  VT_CHECK(table.shape[0] > 0, "cuda embedding: empty table (vocab 0) with nonempty ids");
  cudaStream_t s = AsStream(q);

  EmbeddingErr* derr = nullptr;
  Check(cudaMalloc(&derr, sizeof(EmbeddingErr)), "cudaMalloc embedding flag");
  EmbeddingErr herr{};
  cudaError_t st = cudaMemsetAsync(derr, 0, sizeof(EmbeddingErr), s);
  if (st == cudaSuccess) {
    st = table.dtype == DType::kF32 ? LaunchEmbeddingIn<float>(s, out, table, ids, derr)
                                    : LaunchEmbeddingIn<__nv_bfloat16>(s, out, table, ids, derr);
  }
  if (st == cudaSuccess) {
    st = cudaMemcpyAsync(&herr, derr, sizeof(EmbeddingErr), cudaMemcpyDeviceToHost, s);
  }
  if (st == cudaSuccess) st = cudaStreamSynchronize(s);
  cudaFree(derr);  // best-effort; the primary error (if any) is reported below
  Check(st, "embedding");
  if (herr.status != 0) {
    throw std::runtime_error("vt cuda: embedding: id " + std::to_string(herr.id) +
                             " out of range [0, " + std::to_string(table.shape[0]) + ")");
  }
}

// ---------------------------------------------------------------------------
// rope_neox: grid-stride over (token, head, rotation pair) across q and k.
// Angle math in double (pow/cos/sin) to match the CPU reference numerics.
// Upstream csrc counterpart: csrc/pos_encoding_kernels.cu (rotary_embedding_kernel) — align post-MVP.

template <typename T, typename Tid>
__global__ void RopeNeoxKernel(T* qs, T* ks, const Tid* pos, int64_t hq, int64_t hk,
                               int64_t d, int64_t half, int rot, double base, int64_t n) {
  const int64_t heads = hq + hk;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t pair = idx % half;
    const int64_t head = (idx / half) % heads;
    const int64_t tok = idx / (half * heads);
    T* ptr;
    int64_t off;
    if (head < hq) {
      ptr = qs;
      off = (tok * hq + head) * d;
    } else {
      ptr = ks;
      off = (tok * hk + (head - hq)) * d;
    }
    const int64_t p = static_cast<int64_t>(pos[tok]);
    const double freq = pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
    const double angle = static_cast<double>(p) * freq;
    const float c = static_cast<float>(cos(angle));
    const float sn = static_cast<float>(sin(angle));
    const float x = Load(ptr, off + pair);
    const float y = Load(ptr, off + pair + half);
    Store(ptr, off + pair, x * c - y * sn);
    Store(ptr, off + pair + half, x * sn + y * c);
  }
}

template <typename T>
void LaunchRope(cudaStream_t s, Tensor& qs, Tensor& ks, const Tensor& pos,
                const RopeArgs& args) {
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  const int64_t half = args.rotary_dim / 2;
  const int64_t n = t * (hq + hk) * half;
  if (n == 0) return;
  const double base = static_cast<double>(args.base);
  if (pos.dtype == DType::kI32) {
    RopeNeoxKernel<T, int32_t><<<GridFor(n), kBlock, 0, s>>>(
        qs.Ptr<T>(), ks.Ptr<T>(), pos.Ptr<int32_t>(), hq, hk, d, half, args.rotary_dim, base, n);
  } else {
    RopeNeoxKernel<T, int64_t><<<GridFor(n), kBlock, 0, s>>>(
        qs.Ptr<T>(), ks.Ptr<T>(), pos.Ptr<int64_t>(), hq, hk, d, half, args.rotary_dim, base, n);
  }
  Check(cudaGetLastError(), "rope_neox launch");
}

void RopeNeoxKernelCuda(Queue& q, Tensor& qs, Tensor& ks, const Tensor& pos,
                        const RopeArgs& args) {
  switch (qs.dtype) {
    case DType::kF32: LaunchRope<float>(AsStream(q), qs, ks, pos, args); break;
    case DType::kBF16: LaunchRope<__nv_bfloat16>(AsStream(q), qs, ks, pos, args); break;
    default: VT_CHECK(false, "cuda rope: unsupported dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// Supplied-cache RoPE. Ported from pinned vLLM base.py:160-252,
// csrc/libtorch_stable/pos_encoding_kernels.cu:8-200, and the 3-axis selection
// in mrope.py:14-187,263-375 @ e24d1b24fe96. This hot kernel performs only
// cache lookup + rotation; YaRN formula construction happens once on the host.

__device__ inline int MropeAxisForPair(int64_t pair, int section_t,
                                       int section_h, int section_w,
                                       bool interleaved) {
  if (interleaved) {
    if (pair % 3 == 1 && pair <= 3LL * section_h) return 1;
    if (pair % 3 == 2 && pair <= 3LL * section_w) return 2;
    return 0;
  }
  if (pair < section_t) return 0;
  if (pair < static_cast<int64_t>(section_t) + section_h) return 1;
  return 2;
}

template <typename T, typename Tid>
__global__ void RopeFromCacheKernel(
    T* qs, T* ks, const Tid* positions, const T* cache, int64_t cache_rows,
    int64_t tokens, int64_t hq, int64_t hk, int64_t head_dim, int rotary_dim,
    int64_t half, bool is_neox_style, bool is_mrope, int section_t,
    int section_h, int section_w, bool mrope_interleaved, int64_t n) {
  const int64_t heads = hq + hk;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
       idx < n; idx += step) {
    const int64_t pair = idx % half;
    const int64_t head = (idx / half) % heads;
    const int64_t token = idx / (half * heads);
    const int axis =
        is_mrope ? MropeAxisForPair(pair, section_t, section_h, section_w,
                                   mrope_interleaved)
                 : 0;
    const int64_t position_offset =
        is_mrope ? static_cast<int64_t>(axis) * tokens + token : token;
    const int64_t position = static_cast<int64_t>(positions[position_offset]);
    // The public contract, like upstream's custom op, requires positions to be
    // valid cache rows. Avoid an out-of-bounds read if a broken caller violates
    // it; CPU validation reports the exact error in reference tests.
    if (position < 0 || position >= cache_rows) continue;
    const int64_t cache_offset = position * rotary_dim;
    const float c = Load(cache, cache_offset + pair);
    const float sn = Load(cache, cache_offset + half + pair);

    T* states = head < hq ? qs : ks;
    const int64_t local_head = head < hq ? head : head - hq;
    const int64_t local_heads = head < hq ? hq : hk;
    const int64_t row = (token * local_heads + local_head) * head_dim;
    const int64_t first = is_neox_style ? pair : pair * 2;
    const int64_t second = is_neox_style ? pair + half : pair * 2 + 1;
    const float x = Load(states, row + first);
    const float y = Load(states, row + second);
    Store(states, row + first, x * c - y * sn);
    Store(states, row + second, x * sn + y * c);
  }
}

template <typename T, typename Tid>
void LaunchRopeFromCacheTyped(cudaStream_t stream, Tensor& qs, Tensor* ks,
                              const Tensor& positions, const Tensor& cache,
                              const RopeArgs& args) {
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  const int64_t half = args.rotary_dim / 2;
  const int64_t n = tokens * (hq + hk) * half;
  if (n == 0) return;
  RopeFromCacheKernel<T, Tid><<<GridFor(n), kBlock, 0, stream>>>(
      qs.Ptr<T>(), ks == nullptr ? nullptr : ks->Ptr<T>(),
      positions.Ptr<Tid>(), cache.Ptr<T>(), cache.shape[0], tokens, hq, hk,
      qs.shape[2], args.rotary_dim, half, args.is_neox_style,
      positions.rank == 2, args.mrope_section[0], args.mrope_section[1],
      args.mrope_section[2], args.mrope_interleaved, n);
}

template <typename T>
void LaunchRopeFromCache(cudaStream_t stream, Tensor& qs, Tensor* ks,
                         const Tensor& positions, const Tensor& cache,
                         const RopeArgs& args) {
  if (positions.dtype == DType::kI32) {
    LaunchRopeFromCacheTyped<T, int32_t>(stream, qs, ks, positions, cache,
                                         args);
  } else {
    LaunchRopeFromCacheTyped<T, int64_t>(stream, qs, ks, positions, cache,
                                         args);
  }
  Check(cudaGetLastError(), "rope_from_cache launch");
}

void RopeFromCacheKernelCuda(Queue& q, Tensor& qs, Tensor* ks,
                             const Tensor& positions, const Tensor& cache,
                             const RopeArgs& args) {
  switch (qs.dtype) {
    case DType::kF32:
      LaunchRopeFromCache<float>(AsStream(q), qs, ks, positions, cache, args);
      break;
    case DType::kBF16:
      LaunchRopeFromCache<__nv_bfloat16>(AsStream(q), qs, ks, positions,
                                         cache, args);
      break;
    default:
      VT_CHECK(false,
               "cuda rope_from_cache: unsupported dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// rope_cos_sin_cache: precompute the batch's cos|sin ONCE per step (grid-stride
// over (token, pair)) so the fused preamble below does zero in-kernel
// transcendentals. Angle math in DOUBLE + f32 cast — bit-for-bit RopeNeoxKernel's
// c/sn. cos_sin[t, i]=cos, cos_sin[t, half+i]=sin. Mirrors vLLM's cos_sin_cache
// (RotaryEmbedding._compute_cos_sin_cache; read by fla fused_qk_norm_rope.py:95).

template <typename Tid>
__global__ void RopeCosSinCacheKernel(float* cos_sin, const Tid* pos, int64_t t, int rot,
                                      int64_t half, double base) {
  const int64_t n = t * half;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t pair = idx % half;
    const int64_t tok = idx / half;
    const int64_t p = static_cast<int64_t>(pos[tok]);
    const double freq = pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
    const double angle = static_cast<double>(p) * freq;
    cos_sin[tok * rot + pair] = static_cast<float>(cos(angle));
    cos_sin[tok * rot + half + pair] = static_cast<float>(sin(angle));
  }
}

void RopeCosSinCacheKernelCuda(Queue& q, Tensor& cos_sin, const Tensor& pos, const RopeArgs& args) {
  const int64_t t = cos_sin.shape[0];
  const int64_t half = args.rotary_dim / 2;
  const int64_t n = t * half;
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  const double base = static_cast<double>(args.base);
  if (pos.dtype == DType::kI32) {
    RopeCosSinCacheKernel<int32_t><<<GridFor(n), kBlock, 0, s>>>(
        cos_sin.Ptr<float>(), pos.Ptr<int32_t>(), t, args.rotary_dim, half, base);
  } else {
    RopeCosSinCacheKernel<int64_t><<<GridFor(n), kBlock, 0, s>>>(
        cos_sin.Ptr<float>(), pos.Ptr<int64_t>(), t, args.rotary_dim, half, base);
  }
  Check(cudaGetLastError(), "rope_cos_sin_cache launch");
}

// ---------------------------------------------------------------------------
// attn_qk_norm_rope_gate: the fused full-attention preamble — one launch replacing
// AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox. Grid (T, Hq+Hkv); one block
// per (token, head) does the shared-mem gemma-RMSNorm tree reduction over Dh (reuses
// RmsNormRowKernel's math), then partial NeoX RoPE reading the precomputed cos_sin
// cache (no per-element pow/cos/sin), then passes the gate half through (q heads).
// Structure mirrors GdnPostConvKernel (cuda_gdn.cu). Bit-for-bit equal to the four
// composed ops for f32 out; templated on Tsrc (qgate/kf) and Tout (q/k/gate).
// Mirrors vLLM's fused_qk_rmsnorm_rope (fla fused_qk_norm_rope.py:95-102).

// gemma-RMSNorm one element: (v*inv)*(gemma ? w+1 : w) — matches RmsNormRowKernel.
__device__ inline float GemmaNormElem(float v, float inv, float w, bool gemma) {
  float wj = w;
  if (gemma) wj += 1.0f;
  return v * inv * wj;
}

// Tqk = q_out/k_out store dtype, Tgate = gate_out store dtype. All math stays
// f32; a bf16 Tqk store is the RN round of the exact f32 value, so (Tqk=bf16,
// Tgate=f32) is bit-identical to the f32 path followed by CastBf16 on q/k — the
// FA-2 prefill combo (bf16 q feeds FA-2, bf16 k feeds the bf16 KV-cache write,
// gate stays f32 because sigmoid(gate) must see the un-rounded f32 value).
template <typename Tsrc, typename Tqk, typename Tgate>
__global__ void AttnQkNormRopeGateKernel(Tqk* q_out, Tqk* k_out, Tgate* gate_out,
                                         const Tsrc* qgate, const Tsrc* kf, const float* q_norm,
                                         const float* k_norm, const float* cos_sin, int64_t hq,
                                         int64_t hkv, int64_t dh,
                                         int64_t qgate_row_stride,
                                         int64_t kf_row_stride, int rot,
                                         int64_t half, float eps, bool gemma) {
  const int64_t tok = blockIdx.x;
  const int64_t head = blockIdx.y;  // [0, hq+hkv)
  const bool is_q = head < hq;

  const Tsrc* src;
  const float* w;
  Tqk* out;
  int64_t gate_base = 0;  // only meaningful for q heads
  if (is_q) {
    const int64_t qrow = tok * qgate_row_stride + head * 2 * dh;
    src = qgate + qrow;       // q half [0,dh)
    gate_base = qrow + dh;    // gate half [dh,2dh)
    w = q_norm;
    out = q_out + (tok * hq + head) * dh;
  } else {
    const int64_t hk = head - hq;
    src = kf + tok * kf_row_stride + hk * dh;
    w = k_norm;
    out = k_out + (tok * hkv + hk) * dh;
  }

  // ---- gemma-RMSNorm reduction over Dh (f32 variance) ----
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < dh; j += kBlock) {
    const float v = Load(src, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(dh) + eps);

  // ---- partial NeoX RoPE + write (recompute the paired normed elems; no shared
  // round-trip). Matches RopeNeoxKernel: out[i]=x*c - y*sn, out[i+half]=x*sn + y*c
  // with x=normed[i], y=normed[i+half]; dims [rot,Dh) normed but unrotated. ----
  const float* cs = cos_sin + tok * rot;
  for (int64_t j = threadIdx.x; j < dh; j += kBlock) {
    if (j < half) {
      const float ni = GemmaNormElem(Load(src, j), inv, w[j], gemma);
      const float nih = GemmaNormElem(Load(src, j + half), inv, w[j + half], gemma);
      Store(out, j, ni * cs[j] - nih * cs[half + j]);
    } else if (j < rot) {
      const int64_t i = j - half;
      const float ni = GemmaNormElem(Load(src, i), inv, w[i], gemma);
      const float nih = GemmaNormElem(Load(src, i + half), inv, w[i + half], gemma);
      Store(out, j, ni * cs[half + i] + nih * cs[i]);
    } else {
      Store(out, j, GemmaNormElem(Load(src, j), inv, w[j], gemma));
    }
  }

  // ---- gate passthrough (q heads only): the raw gate half, no norm/rope ----
  if (is_q) {
    Tgate* go = gate_out + (tok * hq + head) * dh;
    for (int64_t j = threadIdx.x; j < dh; j += kBlock) Store(go, j, Load(qgate, gate_base + j));
  }
}

template <typename Tsrc, typename Tqk, typename Tgate>
void LaunchAttnPreamble(cudaStream_t s, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                        const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                        const Tensor& k_norm, const Tensor& cos_sin, const RmsNormArgs& na,
                        const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  const int64_t half = ra.rotary_dim / 2;
  if (t == 0) return;
  dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>(hq + hkv));
  AttnQkNormRopeGateKernel<Tsrc, Tqk, Tgate><<<grid, kBlock, 0, s>>>(
      q_out.Ptr<Tqk>(), k_out.Ptr<Tqk>(), gate_out.Ptr<Tgate>(), qgate.Ptr<Tsrc>(),
      kf.Ptr<Tsrc>(), q_norm.Ptr<float>(), k_norm.Ptr<float>(), cos_sin.Ptr<float>(), hq, hkv, dh,
      qgate.stride[0], kf.stride[0], ra.rotary_dim, half, na.eps, na.gemma);
  Check(cudaGetLastError(), "attn_qk_norm_rope_gate launch");
}

template <typename Tsrc>
void LaunchAttnPreambleOut(cudaStream_t s, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                           const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                           const Tensor& k_norm, const Tensor& cos_sin, const RmsNormArgs& na,
                           const RopeArgs& ra) {
  // (q/k out, gate out) combos: (f32,f32) — the default token-exact path;
  // (bf16,bf16) — all-bf16; (bf16,f32) — the FA-2 prefill combo (bf16 q/k for
  // FA-2 + the bf16 KV-cache write, f32 gate for the sigmoid). Validation in
  // ops.cpp admits exactly these.
  switch (q_out.dtype) {
    case DType::kF32:
      LaunchAttnPreamble<Tsrc, float, float>(s, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm,
                                             cos_sin, na, ra);
      break;
    case DType::kBF16:
      if (gate_out.dtype == DType::kF32) {
        LaunchAttnPreamble<Tsrc, __nv_bfloat16, float>(s, q_out, k_out, gate_out, qgate, kf,
                                                       q_norm, k_norm, cos_sin, na, ra);
      } else {
        LaunchAttnPreamble<Tsrc, __nv_bfloat16, __nv_bfloat16>(s, q_out, k_out, gate_out, qgate,
                                                               kf, q_norm, k_norm, cos_sin, na, ra);
      }
      break;
    default: VT_CHECK(false, "cuda attn_qk_norm_rope_gate: unsupported out dtype");
  }
}

void AttnQkNormRopeGateKernelCuda(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                                  const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                                  const Tensor& k_norm, const Tensor& cos_sin,
                                  const RmsNormArgs& na, const RopeArgs& ra) {
  cudaStream_t s = AsStream(q);
  switch (qgate.dtype) {
    case DType::kF32:
      LaunchAttnPreambleOut<float>(s, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm, cos_sin,
                                   na, ra);
      break;
    case DType::kBF16:
      LaunchAttnPreambleOut<__nv_bfloat16>(s, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm,
                                           cos_sin, na, ra);
      break;
    default: VT_CHECK(false, "cuda attn_qk_norm_rope_gate: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// attention: one block per (query i, q-head h); block threads cooperate over
// the head_dim and stream the keys with an online (flash-style) softmax. The
// online update is algebraically identical to the CPU two-pass reference
// (qwen36-forward-notes.md §5); f32 accumulation. Correctness-grade (M0.9).

template <typename Tin, typename Tout>
__global__ void AttentionKernel(Tout* out, const Tin* query, const Tin* key, const Tin* value,
                                int64_t hq, int64_t hk, int64_t d, int64_t t, float scale,
                                bool causal) {
  const int64_t i = blockIdx.x;  // query position
  const int64_t h = blockIdx.y;  // q-head
  const int64_t g = h / (hq / hk);
  const int64_t jmax = causal ? i : t - 1;
  const int64_t qoff = (i * hq + h) * d;

  extern __shared__ float smem[];
  float* acc = smem;                    // [d] running output accumulator
  float* red = smem + d;                // [blockDim.x] reduction scratch
  __shared__ float s_score, s_m, s_l;   // block-wide score / running max / denom
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.0f;
  if (threadIdx.x == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int64_t j = 0; j <= jmax; ++j) {
    const int64_t koff = (j * hk + g) * d;
    float part = 0.0f;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      part += Load(query, qoff + e) * Load(key, koff + e);
    red[threadIdx.x] = part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) s_score = red[0] * scale;
    __syncthreads();

    const float s = s_score;
    const float m_new = fmaxf(s_m, s);
    const float corr = expf(s_m - m_new);  // 0 on the first key (s_m == -inf)
    const float p = expf(s - m_new);
    const int64_t voff = (j * hk + g) * d;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      acc[e] = acc[e] * corr + p * Load(value, voff + e);
    __syncthreads();
    if (threadIdx.x == 0) {
      s_l = s_l * corr + p;
      s_m = m_new;
    }
    __syncthreads();
  }

  const float inv = 1.0f / s_l;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) Store(out, qoff + e, acc[e] * inv);
}

template <typename Tin>
void LaunchAttention(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& key,
                     const Tensor& value, const AttentionArgs& args) {
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  if (t == 0 || hq == 0 || d == 0) return;
  const dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>(hq));
  const size_t shmem = (static_cast<size_t>(d) + kBlock) * sizeof(float);
  switch (out.dtype) {
    case DType::kF32:
      AttentionKernel<Tin, float><<<grid, kBlock, shmem, s>>>(
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), hq, hk, d, t,
          args.scale, args.causal);
      break;
    case DType::kBF16:
      AttentionKernel<Tin, __nv_bfloat16><<<grid, kBlock, shmem, s>>>(
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), hq, hk,
          d, t, args.scale, args.causal);
      break;
    default: VT_CHECK(false, "cuda attention: unsupported out dtype");
  }
  Check(cudaGetLastError(), "attention launch");
}

void AttentionKernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                         const Tensor& value, const AttentionArgs& args) {
  switch (query.dtype) {
    case DType::kF32: LaunchAttention<float>(AsStream(q), out, query, key, value, args); break;
    case DType::kBF16:
      LaunchAttention<__nv_bfloat16>(AsStream(q), out, query, key, value, args);
      break;
    default: VT_CHECK(false, "cuda attention: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// fused_chain (TDR Phase 0): realize ANY FusedRecipe over the Phase-0 vocabulary
// {kAdd, kMul, kRmsNorm}, selected at runtime by VT_FUSED_TIER (FusedTier()):
//   Tier 0 (default): walk the steps calling the ALREADY-REGISTERED vt:: ops (the
//     residual-add + RMSNorm idiom folds onto vt::RmsNorm(residual)).
//   Tier 1: one interpreter kernel — one block per row, shared-mem tree reduction
//     (the RmsNormRowKernel / AttnQkNormRopeGateKernel skeleton), walking the
//     recipe in a single HBM pass. Bit-for-bit equal to the CUDA RmsNorm(residual)
//     golden: same f32 tree reduction, gemma (1+w), and Tres-rounded residual add.
// Operand ROLES resolve to the typed pointers: kIn/kWeight->Tin, kResidual->Tres,
// kOut->Tout. The recipe POD is passed BY VALUE into the kernel (small, no heap).

template <typename Tin, typename Tout, typename Tres>
struct FusedCtx {
  const Tin* x;
  const Tin* w;
  Tres* res;
  Tout* out;
  int64_t h;
};

template <typename Tin, typename Tout, typename Tres>
__device__ inline float FusedLoadDev(FOperand o, const FusedCtx<Tin, Tout, Tres>& c, int64_t row,
                                     int64_t j) {
  switch (o) {
    case FOperand::kIn: return Load(c.x, row * c.h + j);
    case FOperand::kResidual: return Load(c.res, row * c.h + j);
    case FOperand::kWeight: return Load(c.w, j);
    case FOperand::kOut: return Load(c.out, row * c.h + j);
  }
  return 0.0f;
}

template <typename Tin, typename Tout, typename Tres>
__device__ inline void FusedStoreDev(FOperand o, const FusedCtx<Tin, Tout, Tres>& c, int64_t row,
                                     int64_t j, float v) {
  switch (o) {
    case FOperand::kResidual: Store(c.res, row * c.h + j, v); break;
    case FOperand::kOut: Store(c.out, row * c.h + j, v); break;
    case FOperand::kIn:
    case FOperand::kWeight: break;  // read-only operands (validated host-side)
  }
}

// Tier 1 — interpreter: one block per row walks the whole recipe.
template <typename Tin, typename Tout, typename Tres>
__global__ void FusedChainInterpKernel(FusedCtx<Tin, Tout, Tres> c, float eps, FusedRecipe r) {
  const int64_t row = blockIdx.x;
  const int64_t h = c.h;
  __shared__ float partial[kBlock];
  for (int s = 0; s < r.n; ++s) {
    const FStep st = r.steps[s];
    if (st.op == FOp::kRmsNorm) {
      float acc = 0.0f;
      for (int64_t j = threadIdx.x; j < h; j += kBlock) {
        const float v = FusedLoadDev(st.a, c, row, j);
        acc += v * v;  // kMeanSquare, f32
      }
      partial[threadIdx.x] = acc;
      __syncthreads();
      for (int stride = kBlock / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride)
          partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
      }
      const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
      for (int64_t j = threadIdx.x; j < h; j += kBlock) {
        const float v = FusedLoadDev(st.a, c, row, j);
        float wj = FusedLoadDev(st.b, c, row, j);
        if (st.gemma) wj += 1.0f;
        FusedStoreDev(st.out, c, row, j, v * inv * wj);
      }
      __syncthreads();
    } else {  // kAdd / kMul — elementwise over the row
      for (int64_t j = threadIdx.x; j < h; j += kBlock) {
        const float a = FusedLoadDev(st.a, c, row, j);
        const float b = FusedLoadDev(st.b, c, row, j);
        FusedStoreDev(st.out, c, row, j, st.op == FOp::kAdd ? a + b : a * b);
      }
      __syncthreads();  // writes (e.g. residual) visible before the next step reads
    }
  }
}

template <typename Tin, typename Tout, typename Tres>
void LaunchFusedInterp(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& weight,
                       Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  FusedCtx<Tin, Tout, Tres> c{x.Ptr<Tin>(), weight.Ptr<Tin>(),
                              residual == nullptr ? nullptr : residual->Ptr<Tres>(),
                              out.Ptr<Tout>(), h};
  FusedChainInterpKernel<Tin, Tout, Tres><<<static_cast<unsigned>(t), kBlock, 0, s>>>(c, eps, r);
  Check(cudaGetLastError(), "fused_chain interp launch");
}

// Resolve the (Tin, Tout, Tres) triple and launch the interpreter. Tres follows
// the residual dtype (f32 or bf16); with no residual it is unused (pass float).
template <typename Tin, typename Tout>
void LaunchFusedInterpRes(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& weight,
                          Tensor* residual, const FusedRecipe& r, float eps) {
  if (residual != nullptr && residual->dtype == DType::kBF16) {
    LaunchFusedInterp<Tin, Tout, __nv_bfloat16>(s, out, x, weight, residual, r, eps);
  } else {
    LaunchFusedInterp<Tin, Tout, float>(s, out, x, weight, residual, r, eps);
  }
}

// Tier 0 — composite: fold the residual-add + RMSNorm idiom onto the REGISTERED
// vt::RmsNorm(residual) fused primitive (keeps x/weight in one dtype, residual
// separate — the CUDA RmsNorm kernel requires w.dtype == x.dtype). Same walker
// shape as the CPU Tier-0.
void FusedChainComposite(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                         Tensor* residual, const FusedRecipe& r, float eps) {
  bool residual_add_pending = false;
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    if (st.op == FOp::kAdd) {
      VT_CHECK(st.out == FOperand::kResidual && st.a == FOperand::kIn &&
                   st.b == FOperand::kResidual && residual != nullptr,
               "fused_chain composite: only residual-add (out=res,a=in,b=res) is supported");
      residual_add_pending = true;
    } else if (st.op == FOp::kMul) {
      VT_CHECK(false, "fused_chain composite: kMul has no registered primitive yet (Phase 1)");
    } else {  // kRmsNorm
      VT_CHECK(st.reduce == FReduce::kMeanSquare && st.out == FOperand::kOut &&
                   st.b == FOperand::kWeight,
               "fused_chain composite: rmsnorm must write kOut with a kWeight");
      if (residual_add_pending) {
        VT_CHECK(st.a == FOperand::kResidual,
                 "fused_chain composite: rmsnorm after residual-add must read kResidual");
        vt::RmsNorm(q, out, x, weight, RmsNormArgs{eps, st.gemma}, residual);
        residual_add_pending = false;
      } else {
        const Tensor& a = (st.a == FOperand::kIn) ? x : *residual;
        vt::RmsNorm(q, out, a, weight, RmsNormArgs{eps, st.gemma}, nullptr);
      }
    }
  }
}

void FusedChainKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                          Tensor* residual, const FusedRecipe& r, float eps) {
  if (FusedTier() != 1) {
    FusedChainComposite(q, out, x, weight, residual, r, eps);
    return;
  }
  VT_CHECK(weight.dtype == x.dtype, "cuda fused_chain: weight dtype must match x");
  cudaStream_t s = AsStream(q);
  switch (x.dtype) {
    case DType::kF32:
      switch (out.dtype) {
        case DType::kF32:
          LaunchFusedInterpRes<float, float>(s, out, x, weight, residual, r, eps);
          break;
        case DType::kBF16:
          LaunchFusedInterpRes<float, __nv_bfloat16>(s, out, x, weight, residual, r, eps);
          break;
        default: VT_CHECK(false, "cuda fused_chain: unsupported out dtype");
      }
      break;
    case DType::kBF16:
      switch (out.dtype) {
        case DType::kF32:
          LaunchFusedInterpRes<__nv_bfloat16, float>(s, out, x, weight, residual, r, eps);
          break;
        case DType::kBF16:
          LaunchFusedInterpRes<__nv_bfloat16, __nv_bfloat16>(s, out, x, weight, residual, r, eps);
          break;
        default: VT_CHECK(false, "cuda fused_chain: unsupported out dtype");
      }
      break;
    default: VT_CHECK(false, "cuda fused_chain: unsupported input dtype (f32/bf16 only)");
  }
}

// Registers the CUDA kernels during static init (pre-main, like the CPU ops).
// Filling the op table is harmless on machines without a GPU: the kCUDA
// backend never registers there, so no CUDA queue can exist to dispatch with.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kRmsNorm, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernelCuda)));
    RegisterOp(OpId::kRmsNormQuantFp8, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<RmsNormQuantFp8Fn>(&RmsNormQuantFp8KernelCuda)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernelCuda)));
    RegisterOp(OpId::kEmbedding, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernelCuda)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernelCuda)));
    RegisterOp(
        OpId::kRopeFromCache, DeviceType::kCUDA,
        reinterpret_cast<void*>(
            static_cast<RopeFromCacheFn>(&RopeFromCacheKernelCuda)));
    RegisterOp(
        OpId::kRopeCosSinCache, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernelCuda)));
    RegisterOp(OpId::kAttnQkNormRopeGate, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernelCuda)));
    RegisterOp(OpId::kAttention, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionKernelCuda)));
    RegisterOp(OpId::kFusedChain, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
