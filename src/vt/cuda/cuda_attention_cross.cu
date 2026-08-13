// CUDA `vt::AttentionCross` — the native kernel `include/vt/ops.h:2174-2182`
// recorded as OWED "alongside the LTX-2.5 device-resident forward, which is the
// first caller that would need it". This is that caller's phase (L8), so this is
// that kernel.
//
// WHY IT HAD TO BE WRITTEN NOW, and not deferred again. Before this TU the op had
// a CPU kernel only. On a DISCRETE CUDA device `GetOp` refuses, which is at least
// loud. On GB10 it is worse and quieter: `Backend::UnifiedMemory()` is true, so
// `RegisterReferenceTier` installs the CPU kernel for the accelerator and every
// cross-attention in the DiT would have executed on the HOST over unified memory
// — running, correct, and making "the forward ran on the GPU" false. LTX-2.5's
// block reaches this op six times per layer (two text cross-attentions, two
// audio<->video cross-attentions, and both self-attentions whenever a mask
// supplies a score bias), so that is most of the attention in the model.
//
// PORTED FROM, in structure: `AttentionDenseFlashKernel`
// (src/vt/cuda/cuda_ops.cu:3229-3318), itself a 1:1 structural port of the
// vendored FlashAttention-2 forward kernel
// (src/vt/cuda/flash_attn/src/flash_fwd_kernel.h `compute_attn_1rowblock`:52 —
// the sK/sV shared tiles :163-165 and the `for (int n_block ...)` K/V-tile stream
// with online rescale). Three things are generalized, each of which is exactly
// what `AttentionCrossArgs` exists to express and `vt::Attention` cannot
// (ops.h:499-517):
//
//   1. The key extent is KEY's own token count S, not query's Tq. Every
//      cross-attention has Tq != S, which is the assertion `vt::Attention`
//      makes and the reason the op was split in the first place.
//   2. There is no causal mode at all. `AttentionCrossArgs` carries no `causal`
//      field: this op is bidirectional by construction, so the key loop has no
//      `jstop` and no per-query cutoff.
//   3. An OPTIONAL additive score bias, rank-2 [Tq, S] or the broadcast [1, S]
//      form a padding mask produces, always f32. It is added to the SCALED
//      score, before the max-subtraction, exactly as the CPU reference does
//      (src/vt/cpu/cpu_ops.cpp:2081-2082) and as torch SDPA's `attn_mask` does.
//
// NUMERICS. This kernel uses the ONLINE-softmax recurrence it inherits from the
// flash kernel; the CPU reference (`AttentionCrossKernel`, cpu_ops.cpp:2055-2104)
// uses an explicit three-pass max / exp / normalize. Both accumulate in f32 and
// both are max-subtracted, so they agree to f32 summation-order slack and are NOT
// bit-identical. That is the same relationship `AttentionDenseFast` already has
// with the CPU `AttentionKernel`, and it is why the LTX-2.5 device forward is
// held to the upstream goldens rather than to the host forward byte-for-byte.
//
// The bias is added INSIDE the recurrence, at the point the score is formed, so a
// fully masked key reaches the softmax as a large negative number rather than as
// -inf — which is what makes an all-masked row degenerate to a uniform average
// exactly as torch's does (ops.h:2168-2172). Substituting -inf here would produce
// NaN on that row, and the masked forward gate is what would catch it.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

// Same tiling shape as AttentionDenseFlashKernel, for the same reason: one warp
// per query row, `kCrossBr` of them per CTA sharing each streamed K/V tile, so the
// K/V global reads are amortized over the whole query block instead of being
// re-read once per (query, head).
constexpr int kCrossBr = 16;      // query-warps per CTA (= K/V global-read reuse factor)
constexpr int kCrossBcMax = 64;   // key/value columns per tile, at most

// ...but unlike that kernel, the tile width is chosen at LAUNCH rather than fixed.
// The flash kernel it is modelled on is used at head_dim 64 (the Whisper encoder),
// where `2 * 64 * 64 * 4` is 32 KiB and fits. LTX-2.5's video stream is head_dim
// 128, and at f32 a fixed 64-column tile would ask for `2 * 64 * 128 * 4` = 64 KiB
// of dynamic shared memory — over the 48 KiB a launch gets without opting in — so
// the kernel would fail to launch on exactly the real geometry while every
// reduced-dimension gate (head_dim 8 and 4) passed. Halving the tile until it fits
// costs tile reuse and nothing else.
constexpr size_t kMaxDynamicSmem = 48u * 1024u;

int ChooseTileCols(int64_t d, size_t elem_bytes) {
  int bc = kCrossBcMax;
  while (bc > 1 && static_cast<size_t>(2 * bc) * static_cast<size_t>(d) * elem_bytes >
                       kMaxDynamicSmem) {
    bc /= 2;
  }
  return bc;
}

template <typename Tin, typename Tout>
__global__ void AttentionCrossFlashKernel(Tout* out, const Tin* query, const Tin* key,
                                          const Tin* value, const float* bias, int64_t bias_rows,
                                          int64_t hq, int64_t hk, int64_t d, int64_t tq, int64_t s,
                                          float scale, int bc) {
  constexpr int kMaxPerLane = 8;  // head_dim up to 256
  extern __shared__ __align__(16) char cross_smem[];
  Tin* sK = reinterpret_cast<Tin*>(cross_smem);
  Tin* sV = sK + static_cast<int64_t>(bc) * d;

  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int nthreads = kCrossBr * 32;
  const int64_t h = blockIdx.y;     // q-head (one per CTA)
  const int64_t g = h / (hq / hk);  // shared kv-head for every warp in the CTA
  const int64_t qi = static_cast<int64_t>(blockIdx.x) * kCrossBr + warp;  // this warp's query
  const bool active = qi < tq;
  const int npl = static_cast<int>((d + 31) / 32);  // head_dim elements this lane owns

  // The bias ROW this query reads: its own for the dense [Tq, S] form, or the
  // single broadcast row for the key-only [1, S] one. Reading row 0 for every
  // query would be indistinguishable from correct under a key-only mask, which
  // is exactly why the LTX-2.5 gate carries a DENSE (B, T, T) mask case.
  // `qi` can run past `tq` in the last CTA, so the row is only formed for an
  // ACTIVE warp — an inactive one would compute an out-of-range pointer it never
  // dereferences, which is still not something to write on purpose.
  const float* brow = (bias == nullptr || !active)
                          ? nullptr
                          : bias + (bias_rows == 1 ? 0 : qi) * s;

  float qreg[kMaxPerLane];
  float acc[kMaxPerLane];
#pragma unroll
  for (int k = 0; k < kMaxPerLane; ++k) {
    qreg[k] = 0.0f;
    acc[k] = 0.0f;
  }
  if (active) {
    const int64_t qoff = (qi * hq + h) * d;
    for (int k = 0; k < npl; ++k) {
      const int e = lane + 32 * k;
      if (e < d) qreg[k] = Load(query, qoff + e);
    }
  }
  float m = -CUDART_INF_F, l = 0.0f;

  // Bidirectional by construction: every warp scans the WHOLE key range [0, S).
  for (int64_t c0 = 0; c0 < s; c0 += bc) {
    const int tile = static_cast<int>(min(static_cast<int64_t>(bc), s - c0));
    __syncthreads();
    for (int idx = static_cast<int>(threadIdx.x); idx < tile * static_cast<int>(d);
         idx += nthreads) {
      const int jj = idx / static_cast<int>(d);
      const int e = idx % static_cast<int>(d);
      const int64_t off = ((c0 + jj) * hk + g) * d + e;
      sK[idx] = key[off];
      sV[idx] = value[off];
    }
    __syncthreads();
    if (!active) continue;
    for (int64_t j = 0; j < tile; ++j) {
      const int64_t base = j * d;
      float part = 0.0f;
#pragma unroll
      for (int k = 0; k < kMaxPerLane; ++k) {
        const int e = lane + 32 * k;
        if (k < npl && e < d) part += qreg[k] * Load(sK, base + e);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
      // The bias joins the SCALED score, before the max-subtraction — cpu_ops.cpp
      // :2081-2082 forms `dot *= scale; dot += brow[j];` in that order.
      float sc = part * scale;
      if (brow != nullptr) sc += brow[c0 + j];
      const float m_new = fmaxf(m, sc);
      const float corr = __expf(m - m_new);
      const float p = __expf(sc - m_new);
#pragma unroll
      for (int k = 0; k < kMaxPerLane; ++k) {
        const int e = lane + 32 * k;
        if (k < npl && e < d) acc[k] = acc[k] * corr + p * Load(sV, base + e);
      }
      l = l * corr + p;
      m = m_new;
    }
  }
  if (!active) return;
  const int64_t qoff = (qi * hq + h) * d;
  const float inv = 1.0f / l;
  for (int k = 0; k < npl; ++k) {
    const int e = lane + 32 * k;
    if (e < d) Store(out, qoff + e, acc[k] * inv);
  }
}

template <typename Tin>
void LaunchAttentionCross(cudaStream_t stream, Tensor& out, const Tensor& query, const Tensor& key,
                          const Tensor& value, const Tensor* bias,
                          const AttentionCrossArgs& args) {
  const int64_t tq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t s = key.shape[0], hk = key.shape[1];
  if (tq == 0 || hq == 0 || d == 0 || s == 0) return;
  VT_CHECK(d <= 256, "cuda attention-cross: head_dim <= 256 only");
  const float* bias_data = bias != nullptr ? bias->Ptr<float>() : nullptr;
  const int64_t bias_rows = bias != nullptr ? bias->shape[0] : 0;
  const unsigned nblk = static_cast<unsigned>((tq + kCrossBr - 1) / kCrossBr);
  const dim3 grid(nblk, static_cast<unsigned>(hq));
  const int bc = ChooseTileCols(d, sizeof(Tin));
  const size_t shmem = static_cast<size_t>(2) * bc * d * sizeof(Tin);  // sK + sV
  VT_CHECK(shmem <= kMaxDynamicSmem,
           "cuda attention-cross: head_dim too large for a one-column shared tile");
  switch (out.dtype) {
    case DType::kF32:
      AttentionCrossFlashKernel<Tin, float><<<grid, kCrossBr * 32, shmem, stream>>>(
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), bias_data,
          bias_rows, hq, hk, d, tq, s, args.scale, bc);
      break;
    case DType::kBF16:
      AttentionCrossFlashKernel<Tin, __nv_bfloat16><<<grid, kCrossBr * 32, shmem, stream>>>(
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), bias_data,
          bias_rows, hq, hk, d, tq, s, args.scale, bc);
      break;
    default: VT_CHECK(false, "cuda attention-cross: unsupported out dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "attention-cross launch");
}

void AttentionCrossKernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                              const Tensor& value, const Tensor* bias,
                              const AttentionCrossArgs& args) {
  switch (query.dtype) {
    case DType::kF32:
      LaunchAttentionCross<float>(AsStream(q), out, query, key, value, bias, args);
      break;
    case DType::kBF16:
      LaunchAttentionCross<__nv_bfloat16>(AsStream(q), out, query, key, value, bias, args);
      break;
    default:
      VT_CHECK(false, "cuda attention-cross: unsupported input dtype (f32/bf16 only)");
  }
}

// Self-registering TU, the established additive pattern (cuda_layernorm.cu:5-9):
// this file is the whole registration surface for the op, so no existing kernel
// TU and no shared op array is edited.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kAttentionCross, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttentionCrossFn>(&AttentionCrossKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
