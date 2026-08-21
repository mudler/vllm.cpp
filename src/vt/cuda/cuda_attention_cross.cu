// CUDA `vt::AttentionCross` — the native kernel `include/vt/ops.h:2174-2182`
// recorded as OWED "alongside the LTX-2.5 device-resident forward, which is the
// first caller that would need it". This is that caller's phase (L8), so this is
// that kernel.
//
// WHY IT HAD TO BE WRITTEN NOW, and not deferred again. Before this TU the op had
// a CPU kernel only, and `GetOp` on a CUDA device would have installed the
// portable CPU tier: the gate read `Backend::UnifiedMemory()`, which GB10
// reports true. Every cross-attention in the DiT would have left the GPU.
// LTX-2.5's block reaches this op six times per layer (two text
// cross-attentions, two audio<->video cross-attentions, and both self-attentions
// whenever a mask supplies a score bias), so that is most of the attention in
// the model.
//
// That gate now reads `Backend::DeviceMemoryIsHostAddressable()`, which CUDA
// answers false because it allocates with `cudaMalloc` (#844, #1435), so the
// same absence would be a named refusal today rather than a silent host run.
// The original note called the host run "correct", and it was not: the host
// kernel dereferences device pointers and the process gets SIGSEGV. Either way
// the conclusion is unchanged — this kernel had to exist — and the refusal makes
// the reason legible instead of leaving it to a crash.
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

// ---------------------------------------------------------------------------
// The BLOCKED provider ([#1555](https://github.com/mudler/vllm.cpp/issues/1555))
//
// WHAT THE KERNEL ABOVE COSTS, MEASURED. At MiniMax-Music3's DiT geometry --
// Tq = S = 690, Hq = Hkv = 32, head_dim 64, f32, which is what
// `.agents/specs/minimax-music3.md` §21.9 measured in situ -- it runs at
// **0.204 TFLOP/s** while the four `vt::MatmulBT` GEMMs beside it in the same
// forward run at 3.98, on a device whose fp32 CUDA-core peak is 5.369.
//
// THE MECHANISM IS MEASURED, NOT ASSUMED, and it is not what the issue guessed.
// `.agents/specs/attention-cross-blocked.md` §3 has the ablation table from an
// `rc` job on `thor:gpu0`. Its three findings:
//
//   * OCCUPANCY IS NOT THE CONSTRAINT. The kernel achieves 2 blocks/SM =
//     32 warps/SM = 0.67 of Thor's 1536-thread limit, and halving the CTA to 8
//     query-warps -- which doubles blocks/SM at the SAME warps/SM -- measures
//     2.3 % SLOWER, not faster.
//   * BANDWIDTH IS NOT THE CONSTRAINT. Every CTA re-reads the whole of K and V,
//     15.9 GB per layer at this geometry; removing that re-read ENTIRELY (an
//     ablation that loads one tile and reuses it) measures 8.4 % SLOWER. The
//     device has 32 MiB of L2 and the whole of K + V is 11.3 MiB, so the
//     re-reads never leave it.
//   * THE SHUFFLE IS ONLY 18 %. Deleting the five-step `__shfl_xor_sync`
//     butterfly outright buys 18.1 %, and deleting the online-softmax
//     recurrence outright buys 19.8 %.
//
// So the cost is not any one instruction: it is the DECOMPOSITION. One warp owns
// one query and walks the keys ONE AT A TIME, so a whole loop body -- the shared
// loads, the butterfly, the exponential, the rescale of the accumulator -- is
// paid per key for `head_dim / 32 = 2` useful multiply-adds per lane. The kernel
// issues roughly one useful FMA per five to eight issued instructions and no
// amount of parallelism fixes an instruction mix.
//
// THE FIX IS THE SHAPE FlashAttention-2 ACTUALLY HAS, and which the kernel above
// claims in its header to be ported from but is not. FA-2's forward gives a CTA
// a QUERY TILE and a KEY TILE and computes their outer product
// (`flash_fwd_kernel.h` `compute_attn_1rowblock`:52, the `gemm` calls at :219
// and :265 over `Br x Bc` accumulator fragments) -- so the head-dim reduction is
// sequential inside a thread's own registers, there is no cross-lane reduction
// at all, and the online-softmax recurrence advances once per KEY TILE instead
// of once per key. FA-2 does that with `mma.sync` on tensor cores; f32 is
// upstream's resolved dtype for this DiT (§21.2, anchored at the diffusers pin)
// and there are no f32 tensor cores, so the same decomposition is realized on
// CUDA cores. The STRUCTURE is ported; the instruction is not, and cannot be.
//
// LAYOUT. 128 threads as 16 query groups x 8 key groups. A thread owns a
// QT x KT tile of the score block S[BR, BC] and a QT x DT tile of the output
// block O[BR, D], with QT = BR/16, KT = BC/8 and DT = D/8. Q and K are staged
// TRANSPOSED ([d][token]) so the innermost loop reads a contiguous run per
// group, which is what makes the shared reads broadcast rather than gather.
//
// TWO TILINGS ARE INSTANTIATED, and both are chosen by the 48 KiB of static
// shared memory a launch gets without opting in:
//
//   head_dim  64 -> BR 64, BC 32: 43 008 B, QT 4, KT 4, DT 8
//   head_dim 128 -> BR 32, BC 16: 35 840 B, QT 2, KT 2, DT 16
//
// Every other shape DECLINES to the kernel above through the provider seam, so
// no caller loses a path and no call site changes.
//
// NUMERICS. Not bit-identical to the kernel above, and it was never going to be:
// the head-dim sum is a sequential f32 accumulation over `head_dim` terms where
// the kernel above uses a 32-lane pairwise butterfly, and the softmax
// denominator sums a tile's keys before the running `l` absorbs it. Both are
// max-subtracted f32 online softmax and both are held to the SAME independent
// f64 reference at the SAME committed tolerances (`tests/vt/
// test_ops_attention_cross.cpp`), which is exactly the relationship the file
// header above already declares between this op's CUDA and CPU kernels.

constexpr int kBlockedThreads = 128;
constexpr int kBlockedQGroups = 16;
constexpr int kBlockedKGroups = 8;
// Above `vt-native`'s 0 so the shape gate runs first, below the Metal MLX
// provider's 100 for no reason other than leaving room; nothing else registers
// for this op.
constexpr int kBlockedPriority = 10;
constexpr const char* kBlockedProvider = "vt-cross-blocked";

template <typename Tin, typename Tout, int D, int BR, int BC>
__global__ __launch_bounds__(kBlockedThreads) void AttentionCrossBlockedKernel(
    Tout* out, const Tin* query, const Tin* key, const Tin* value, const float* bias,
    int64_t bias_rows, int64_t hq, int64_t hk, int64_t tq, int64_t s, float scale) {
  constexpr int QT = BR / kBlockedQGroups;
  constexpr int KT = BC / kBlockedKGroups;
  constexpr int DT = D / kBlockedKGroups;
  // The coverage argument below -- that every element of `sP` the P.V loop reads
  // was written this iteration, and that every `sRed` slot a row reduces over was
  // filled -- rests ENTIRELY on these three divisions being exact. A tiling with,
  // say, BR = 40 would truncate QT to 2, leave `sP` columns 32-39 unwritten, and
  // the P.V loop would read uninitialised shared memory with no diagnostic
  // anywhere. Enforced rather than left as a convention the next tiling has to
  // remember.
  static_assert(BR % kBlockedQGroups == 0, "BR must divide into 16 query groups");
  static_assert(BC % kBlockedKGroups == 0, "BC must divide into 8 key groups");
  static_assert(D % kBlockedKGroups == 0, "D must divide into 8 key groups");
  static_assert(kBlockedQGroups * kBlockedKGroups == kBlockedThreads,
                "the thread grid must be exactly 16 query groups x 8 key groups");

  // The staging tiles are f32 for BOTH streams. A bf16 operand is widened once
  // on the way in rather than on every one of the `D` reads the inner loop makes
  // of it, and the arithmetic is f32 either way -- which is what the kernel above
  // also does, one element at a time, through `Load`.
  __shared__ float sQt[D * BR];              // [d][query]
  __shared__ float sKt[D * BC];              // [d][key]
  __shared__ float sV[BC * D];               // [key][d]
  __shared__ float sP[BC * BR];              // [key][query]
  __shared__ float sRed[kBlockedKGroups * BR];  // [key-group][query]

  const int tid = static_cast<int>(threadIdx.x);
  const int qg = tid / kBlockedKGroups;
  const int kg = tid % kBlockedKGroups;
  const int64_t h = blockIdx.y;
  const int64_t g = h / (hq / hk);  // the shared kv-head, as above
  const int64_t q0 = static_cast<int64_t>(blockIdx.x) * BR;

  // The query tile, transposed, once for the whole key sweep. A row past `tq`
  // is ZEROED rather than skipped: it keeps the tile rectangular, and its output
  // is discarded at the store.
  for (int idx = tid; idx < BR * D; idx += kBlockedThreads) {
    const int qq = idx / D, e = idx % D;
    const int64_t qi = q0 + qq;
    sQt[e * BR + qq] = qi < tq ? Load(query, (qi * hq + h) * D + e) : 0.0f;
  }

  float acc[QT][DT];
#pragma unroll
  for (int i = 0; i < QT; ++i)
#pragma unroll
    for (int x = 0; x < DT; ++x) acc[i][x] = 0.0f;
  float m[QT], l[QT];
#pragma unroll
  for (int i = 0; i < QT; ++i) {
    m[i] = -CUDART_INF_F;
    l[i] = 0.0f;
  }

  // Bidirectional by construction, exactly as above: every CTA scans [0, S).
  for (int64_t c0 = 0; c0 < s; c0 += BC) {
    const int tile = static_cast<int>(min(static_cast<int64_t>(BC), s - c0));
    __syncthreads();
    for (int idx = tid; idx < BC * D; idx += kBlockedThreads) {
      const int kk = idx / D, e = idx % D;
      const bool ok = kk < tile;
      // The out-of-tile lane reads key 0 rather than branching, and throws the
      // value away. Reading past `s` would be an out-of-bounds global access.
      const int64_t off = ((c0 + (ok ? kk : 0)) * hk + g) * D + e;
      const float kv = Load(key, off);
      const float vv = Load(value, off);
      sKt[e * BC + kk] = ok ? kv : 0.0f;
      sV[kk * D + e] = ok ? vv : 0.0f;
    }
    __syncthreads();

    // S = Q . K^T, over the WHOLE head dim, in registers. No shuffle.
    float sfrag[QT][KT];
#pragma unroll
    for (int i = 0; i < QT; ++i)
#pragma unroll
      for (int j = 0; j < KT; ++j) sfrag[i][j] = 0.0f;
#pragma unroll 4
    for (int e = 0; e < D; ++e) {
      float av[QT], bv[KT];
#pragma unroll
      for (int i = 0; i < QT; ++i) av[i] = sQt[e * BR + qg * QT + i];
#pragma unroll
      for (int j = 0; j < KT; ++j) bv[j] = sKt[e * BC + kg * KT + j];
#pragma unroll
      for (int i = 0; i < QT; ++i)
#pragma unroll
        for (int j = 0; j < KT; ++j) sfrag[i][j] += av[i] * bv[j];
    }

    // The bias joins the SCALED score, before the max-subtraction -- the same
    // order `cpu_ops.cpp` and the kernel above form it in. A key past the tile
    // tail is sent to -inf so it cannot enter this tile's max or its sum; a
    // FULLY MASKED key arrives as a large negative FINITE number from the bias
    // and stays finite, which is what keeps an all-masked row a uniform average
    // instead of a NaN (ops.h).
#pragma unroll
    for (int i = 0; i < QT; ++i) {
      const int64_t qi = q0 + qg * QT + i;
      const float* brow =
          (bias == nullptr || qi >= tq) ? nullptr : bias + (bias_rows == 1 ? 0 : qi) * s;
#pragma unroll
      for (int j = 0; j < KT; ++j) {
        const int kk = kg * KT + j;
        float v = sfrag[i][j] * scale;
        if (brow != nullptr && kk < tile) v += brow[c0 + kk];
        sfrag[i][j] = kk < tile ? v : -CUDART_INF_F;
      }
    }

    // The row max and the row sum, reduced ACROSS the 8 key groups. Each thread
    // folds its own KT keys in registers first, so every score is exponentiated
    // exactly ONCE and the only shared traffic is an 8-wide cross-group pass.
    // This is the whole point of the restructure: the recurrence advances once
    // per BC keys, not once per key.
#pragma unroll
    for (int i = 0; i < QT; ++i) {
      float mloc = -CUDART_INF_F;
#pragma unroll
      for (int j = 0; j < KT; ++j) mloc = fmaxf(mloc, sfrag[i][j]);
      sRed[kg * BR + qg * QT + i] = mloc;
    }
    __syncthreads();
    float mt[QT], corr[QT];
#pragma unroll
    for (int i = 0; i < QT; ++i) {
      const int qq = qg * QT + i;
      float v = m[i];
#pragma unroll
      for (int r = 0; r < kBlockedKGroups; ++r) v = fmaxf(v, sRed[r * BR + qq]);
      mt[i] = v;
      corr[i] = __expf(m[i] - v);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < QT; ++i) {
      float lloc = 0.0f;
#pragma unroll
      for (int j = 0; j < KT; ++j) {
        const float pj = __expf(sfrag[i][j] - mt[i]);
        lloc += pj;
        sP[(kg * KT + j) * BR + qg * QT + i] = pj;
      }
      sRed[kg * BR + qg * QT + i] = lloc;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < QT; ++i) {
      const int qq = qg * QT + i;
      float lt = l[i] * corr[i];
#pragma unroll
      for (int r = 0; r < kBlockedKGroups; ++r) lt += sRed[r * BR + qq];
      m[i] = mt[i];
      l[i] = lt;
#pragma unroll
      for (int x = 0; x < DT; ++x) acc[i][x] *= corr[i];
    }
    __syncthreads();

    // O += P . V. A padded key contributes p = 0 against v = 0, so the tail
    // needs no branch here.
#pragma unroll 4
    for (int j = 0; j < BC; ++j) {
      float pv[QT], vv[DT];
#pragma unroll
      for (int i = 0; i < QT; ++i) pv[i] = sP[j * BR + qg * QT + i];
#pragma unroll
      for (int x = 0; x < DT; ++x) vv[x] = sV[j * D + kg * DT + x];
#pragma unroll
      for (int i = 0; i < QT; ++i)
#pragma unroll
        for (int x = 0; x < DT; ++x) acc[i][x] += pv[i] * vv[x];
    }
  }

#pragma unroll
  for (int i = 0; i < QT; ++i) {
    const int64_t qi = q0 + qg * QT + i;
    if (qi >= tq) continue;
    const float inv = 1.0f / l[i];
    const int64_t qoff = (qi * hq + h) * D;
#pragma unroll
    for (int x = 0; x < DT; ++x) Store(out, qoff + kg * DT + x, acc[i][x] * inv);
  }
}

template <typename Tin, typename Tout, int D, int BR, int BC>
void LaunchBlocked(cudaStream_t stream, Tensor& out, const Tensor& query, const Tensor& key,
                   const Tensor& value, const float* bias_data, int64_t bias_rows,
                   const AttentionCrossArgs& args) {
  const int64_t tq = query.shape[0], hq = query.shape[1];
  const int64_t s = key.shape[0], hk = key.shape[1];
  const dim3 grid(static_cast<unsigned>((tq + BR - 1) / BR), static_cast<unsigned>(hq));
  AttentionCrossBlockedKernel<Tin, Tout, D, BR, BC>
      <<<grid, kBlockedThreads, 0, stream>>>(out.Ptr<Tout>(), query.Ptr<Tin>(), key.Ptr<Tin>(),
                                             value.Ptr<Tin>(), bias_data, bias_rows, hq, hk, tq, s,
                                             args.scale);
  Check(cudaGetLastError(), "attention-cross blocked launch");
}

template <typename Tin>
bool TryBlockedOut(cudaStream_t stream, Tensor& out, const Tensor& query, const Tensor& key,
                   const Tensor& value, const float* bias_data, int64_t bias_rows,
                   const AttentionCrossArgs& args) {
  const int64_t d = query.shape[2];
  // The head dim is checked HERE as well as in `BlockedShape`, and the duplication
  // is deliberate. A `d == 64` test read as exhaustive would send every other head
  // dim into the D = 128 instantiation, which indexes query, key, value AND out
  // with a hard-coded stride of 128 -- wrong results plus out-of-bounds global
  // WRITES, from a one-line widening of the gate in a different function. The two
  // are coupled by nothing but this pair of switches, so both state the set.
  switch (out.dtype) {
    case DType::kF32:
      if (d == 64) {
        LaunchBlocked<Tin, float, 64, 64, 32>(stream, out, query, key, value, bias_data, bias_rows,
                                              args);
        return true;
      }
      if (d == 128) {
        LaunchBlocked<Tin, float, 128, 32, 16>(stream, out, query, key, value, bias_data, bias_rows,
                                               args);
        return true;
      }
      return false;
    case DType::kBF16:
      if (d == 64) {
        LaunchBlocked<Tin, __nv_bfloat16, 64, 64, 32>(stream, out, query, key, value, bias_data,
                                                      bias_rows, args);
        return true;
      }
      if (d == 128) {
        LaunchBlocked<Tin, __nv_bfloat16, 128, 32, 16>(stream, out, query, key, value, bias_data,
                                                       bias_rows, args);
        return true;
      }
      return false;
    default: return false;
  }
}

// THE SHAPE GATE, and it is deliberately narrow. Two head dims are instantiated
// because two tilings fit in 48 KiB, and a query tile shorter than `BR` would
// hand most of the CTA nothing to do -- at Tq = 1 (the MiniMax-Music3 RVQ depth
// decoder's step) 63 of 64 query rows are padding, which is the regime the
// warp-per-query kernel above is actually the right shape for. Everything this
// returns false on keeps the kernel it has today, byte for byte.
bool BlockedShape(const Tensor& out, const Tensor& query, const Tensor& key) {
  const int64_t tq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  if (hk == 0 || hq % hk != 0) return false;
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) return false;
  if (d == 64) return tq >= 64;
  if (d == 128) return tq >= 32;
  return false;
}

// The fallback is resolved ONCE and the decline is counted separately, the
// contract `op_provider.h` states and `metal_mlx_provider.mm` measured: a
// shape-gated provider declines on the hot path, and re-walking the provider
// stack per decline is a real cost. The stack is immutable after registration,
// so a function-local static is the right lifetime.
AttentionCrossFn BlockedFallback() {
  static AttentionCrossFn f = reinterpret_cast<AttentionCrossFn>(
      GetOpFallback(OpId::kAttentionCross, DeviceType::kCUDA, kBlockedProvider));
  return f;
}

void AttentionCrossBlockedCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                               const Tensor& value, const Tensor* bias,
                               const AttentionCrossArgs& args) {
  const int64_t tq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t s = key.shape[0];
  if (tq != 0 && hq != 0 && d != 0 && s != 0 && BlockedShape(out, query, key)) {
    const float* bias_data = bias != nullptr ? bias->Ptr<float>() : nullptr;
    const int64_t bias_rows = bias != nullptr ? bias->shape[0] : 0;
    bool ran = false;
    switch (query.dtype) {
      case DType::kF32:
        ran = TryBlockedOut<float>(AsStream(q), out, query, key, value, bias_data, bias_rows, args);
        break;
      case DType::kBF16:
        ran = TryBlockedOut<__nv_bfloat16>(AsStream(q), out, query, key, value, bias_data,
                                           bias_rows, args);
        break;
      default: ran = false;
    }
    if (ran) return;
  }
  NoteOpDecline(OpId::kAttentionCross, DeviceType::kCUDA);
  BlockedFallback()(q, out, query, key, value, bias, args);
}

// Self-registering TU, the established additive pattern (cuda_layernorm.cu:5-9):
// this file is the whole registration surface for the op, so no existing kernel
// TU and no shared op array is edited.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kAttentionCross, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttentionCrossFn>(&AttentionCrossKernelCuda)));
    // The blocked kernel registers ABOVE `vt-native` and declines per call on a
    // shape it has no tiling for, which is how a shape gate is expressed in this
    // seam (op_provider.h `DECLINE-AND-FALL-BACK`). Two consequences are the
    // reason it is done this way rather than with a private env flag: selection
    // becomes OBSERVABLE through `GetOpProviderStats`, so a gate can assert
    // which kernel ran instead of believing a comment; and
    // `VT_OP_PROVIDER_DISABLE=vt-cross-blocked` is a SAME-BINARY A/B lever that
    // already exists, so the measurement needs no second build.
    OpProvider blocked;
    blocked.name = kBlockedProvider;
    blocked.priority = kBlockedPriority;
    blocked.supports = nullptr;  // no device capability beyond CUDA itself
    blocked.fn =
        reinterpret_cast<void*>(static_cast<AttentionCrossFn>(&AttentionCrossBlockedCuda));
    RegisterOpProvider(OpId::kAttentionCross, DeviceType::kCUDA, blocked);
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
