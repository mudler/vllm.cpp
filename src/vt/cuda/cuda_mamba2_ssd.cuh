// CUDA device arm of the Mamba2 / SSD selective-scan core.
// .agents/specs/mamba2-ssd.md W2, issue #496.
//
// Three kernels, each the DEVICE transcription of the CPU host reference that
// landed in W1 (src/vt/cpu/cpu_ops.cpp `Mamba2ChunkScanKernel`,
// `Mamba2StateUpdateKernel`, `RmsNormGatedGroupKernel`), which is itself the 1:1
// port of the upstream path named on it at the pinned oracle `555967922`
// (vLLM 0.26.0.dev0):
//
//   Mamba2ChunkScan   <- ops/ssd_combined.py:27-235 (the 5-stage varlen pipeline)
//                        over ssd_chunk_state.py, ssd_state_passing.py,
//                        ssd_bmm.py and ssd_chunk_scan.py
//   Mamba2StateUpdate <- ops/mamba_ssm.py:497+ `selective_state_update`
//   RmsNormGatedGroup <- mamba_mixer2.py:100-149 `Mixer2RMSNormGated.forward_native`
//
// ─── THE DECLARED EQUIVALENCE CONTRACT (mamba2-ssd.md §8.3) ───────────────────
//
// This arm keeps **f32 accumulation throughout** and does NOT mirror the tile
// downcasts in upstream's Triton kernels — `b = b.to(x_ptr.dtype.element_ty)`
// before `tl.dot` (ssd_chunk_state.py:283-285), and `cb.to(x_ptr.dtype.element_ty)`
// / `prev_states.to(C_ptr.dtype.element_ty)` (ssd_chunk_scan.py:266-269,
// :359-363). Those casts are the INPUT-PRECISION REQUIREMENT OF `tl.dot`, i.e. of
// a tensor-core MMA, not a statement of the algorithm: every one of those tiles is
// loaded with an explicit `.to(tl.float32)` and computed in f32 right up to the
// instant it is fed to the MMA. These are scalar-FMA kernels with no MMA, where
// the downcast would be lossy for nothing.
//
// That is NOT a "wider dtype" deviation, and the distinction matters because a
// token gate cannot catch a dtype that is too wide (.agents/porting.md): the
// MEMORY FORMAT here is byte-for-byte the host arm's. Every load and store goes
// through the operand's own declared dtype; `states` and `CB` are f32 because
// upstream pins them there (`states_in_fp32=True` ssd_combined.py:100-102,
// `output_dtype=torch.float32` :124); the inter-chunk `passed` buffer is allocated
// at `state_dtype` — NOT at the host reference's f32 working width, which W1
// explicitly flagged as a width W2 must not inherit (cpu_ops.cpp, §8.2 F9).
// No extra byte moves; only the register precision of one product differs, and it
// differs in the direction Triton itself takes wherever it is not feeding an MMA.
//
// The consequence for the gate is stated in the test files: the two arms are NOT
// bit-identical. TWO effects are admitted, both structural and both carrying a
// DERIVED forward-error bound; neither is a tuned number. See
// tests/vt/test_ops_mamba2_ssd.cpp `DerivedRtol`.
//
// ─── ADMITTED SOURCE 1: THE ELEMENTARY FUNCTIONS ─────────────────────────────
// The two arms call different libms (`expf`/`log1pf`) — CUDA's `expf` is
// documented to <= 2 ulp and glibc's to <= 0.5.
//
// ─── ADMITTED SOURCE 2: FMA CONTRACTION ──────────────────────────────────────
// Host C++ is pinned `-ffp-contract=off` (CMakeLists.txt:41-56) precisely so
// `a*b + c` keeps two roundings. NOTHING passes `--fmad=false` to nvcc, so this
// header compiles at nvcc's DEFAULT `--fmad=true` and every `acc += a*b` below
// (:290, :335, :364, :434, :442, :501, and the gated norm's `part += v*v` at
// :554) is a SINGLE-rounding `fma` whose host twin is not. That is measured, not
// theoretical: .agents/benchmark-record.md:532 records a pre-rounded `v²`
// differing by <= 1 ulp from this exact nvcc-`fmad` idiom and flipping a
// near-tie. CMakeLists.txt:41-56 carves CUDA out of the contraction policy on
// the grounds that "GPU parity tests compare GPU-vs-GPU"; the device-vs-host
// comparison is exactly the case that carve-out does not cover, so the bound
// carries the term instead of the build removing it.
//
// `-fmad=false` was REJECTED, not overlooked. nvcc takes it per TRANSLATION
// UNIT and this is a header, included by cuda_gdn.cu:48 — so applying it means
// either de-contracting every GDN kernel in that TU (a measured hot decode
// path) or splitting a new `src/vt/` TU, which §8.3 already records as blocked
// on #515. Slowing an unrelated shipped kernel to make a bound's prose true is
// the wrong trade; widening the bound by the term the build actually emits is
// the right one. `DerivedRtol` is `5·(K+2)·u`, not `4·(K+2)·u`, and §8.3 shows
// the arithmetic.
//
// ─── ACCUMULATION ORDER IS PART OF THE PORT ──────────────────────────────────
// Except in the gated norm's group reduction (which is a block reduction, and
// says so), every accumulation below runs in ONE thread, over the SAME index
// range in the SAME direction as the host reference. That is deliberate — order
// is the AMPLIFYING source and pinning it keeps the bound to the two terms
// above. It does not, on its own, make the libm the only one.
//
// ─── WHAT THIS ARM DOES NOT CHECK (named residual, §8.3) ─────────────────────
// The SHARED validator checks metadata SHAPE, DTYPE and DEVICE only
// (`CheckI32Meta`, ops.cpp:1717-1723). Every VALUE check lives in the host
// kernel, which reads the tensors (cpu_ops.cpp:1622-1648): `A < 0`
// (`CheckMamba2ANegative`), `state_indices` distinctness, the `cu_chunk_seqlens`
// tiling, per-chunk length bounds, `seq_idx[c] ∈ [0,S)`, and both halves of
// `0 <= last_chunk_indices[b] < nchunks`. This arm re-checks NONE of them: the
// operands live on the DEVICE, so reading them costs a D2H copy plus a stream
// synchronise per call — the same host tax the GDN prefill path was rebuilt to
// remove (`GdnArgs::query_start_loc_host`, include/vt/ops.h) — and it would make
// the op uncapturable in a CUDA graph. This mirrors the policy cuda_gdn.cu
// already states for exactly this case ("here bad metadata is unchecked --
// correctness-grade; the M0.9 builder owns metadata integrity",
// cuda_gdn.cu:8-13). Closing the gap needs the deferred device error ring
// cuda_ops.cu:790-940 already implements for embedding, and is owed.
//
// MEMORY SAFETY IS A NARROWER CLAIM AND IS MADE SEPARATELY, because dropping a
// check whose consequence is a WRONG NUMBER is correctness-grade while dropping
// one whose consequence is an out-of-bounds ACCESS is not:
//
//   CLAMPED, therefore memory-safe under a violation —
//     * `last_chunk_indices[b] >= nchunks` would make M2Store(passed, ...) at
//       :336 write PAST the cudaMallocAsync allocation; `lci[b-1] < -1` would
//       make `states[...]` at :335 read before it. Clamped at :319-331.
//     * `seq_idx[c] ∉ [0,S)` would read `initial_states` out of bounds at :435,
//       and a `seq_idx[0] < 0` would additionally make `si == si_prev` at c == 0
//       and index `passed` at chunk -1. Clamped at :404-419.
//     * an out-of-range `state_indices` slot writes nothing at all. Clamped at
//       :486, as it always has been.
//   The clamps are the reason the D2H argument above does not apply to these:
//   the values are ALREADY IN REGISTERS at their use sites, so bounding them
//   costs nothing and needs no host round trip. They do NOT restore the checks —
//   out-of-contract metadata still produces a WRONG ANSWER, now with a defined
//   shape. They bound only WHERE it is read from. Both are pinned by device-only
//   cases in tests/vt/test_ops_mamba2_ssd.cpp rather than asserted here.
//
//   NOT CLAMPED, therefore NOT memory-safe under a violation —
//     * the `cu_chunk_seqlens` tiling and per-chunk length checks. `start` and
//       `len` derived from a garbage `ccs` index x/B/C/z/out out of bounds in
//       every stage. Bounding these needs T at each use site and a clamp in the
//       inner loops, which is not free; it is owed with the error ring above.
//   VALUE-ONLY, in-bounds wrong number —
//     * `A < 0` and `state_indices` distinctness.
#ifndef VT_CUDA_MAMBA2_SSD_CUH_
#define VT_CUDA_MAMBA2_SSD_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda::mamba2 {
namespace {

constexpr int kM2Block = 256;

void M2Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda mamba2: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t M2Stream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Scope guard for the prefill path's per-call scratch. `M2Check` throws, so a
// failure on the Nth `cudaMallocAsync` would otherwise leak the N-1 before it.
// The happy path calls `Release()`, which frees on the stream and CHECKS each
// free exactly as the open-coded sequence it replaces did; the destructor is the
// unwinding path only, and cannot throw.
class M2Scratch {
 public:
  explicit M2Scratch(cudaStream_t s) : s_(s) {}
  M2Scratch(const M2Scratch&) = delete;
  M2Scratch& operator=(const M2Scratch&) = delete;
  ~M2Scratch() {
    for (int i = 0; i < n_; ++i) static_cast<void>(cudaFreeAsync(p_[i], s_));
  }
  void* Alloc(size_t bytes, const char* what) {
    // Refuse rather than overrun if a sixth buffer is ever added: a silent
    // overflow here would be the exact defect class the guard exists to remove.
    if (n_ >= kMax) throw std::runtime_error("vt cuda mamba2: scratch slots exhausted");
    void* p = nullptr;
    M2Check(cudaMallocAsync(&p, bytes, s_), what);
    p_[n_++] = p;
    return p;
  }
  void Release() {
    const int n = n_;
    n_ = 0;
    // Free ALL of them before reporting, so a mid-sequence failure does not
    // leak the remainder the way the open-coded `M2Check(cudaFreeAsync(...))`
    // sequence this replaces did; then throw on the first error seen.
    cudaError_t first = cudaSuccess;
    for (int i = 0; i < n; ++i) {
      const cudaError_t e = cudaFreeAsync(p_[i], s_);
      if (first == cudaSuccess) first = e;
    }
    M2Check(first, "scratch free");
  }

 private:
  static constexpr int kMax = 5;
  cudaStream_t s_;
  void* p_[kMax] = {};
  int n_ = 0;
};

// Grid for a grid-stride loop over `n` items at kM2Block threads, capped so a
// launch stays reasonable on any element count.
unsigned M2Grid(int64_t n) {
  const int64_t blocks = (n + kM2Block - 1) / kM2Block;
  if (blocks < 1) return 1u;
  return static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
}

// f32 load/store through the operand's OWN dtype — the memory format is the host
// arm's, only the arithmetic is f32 (`LoadF32`/`StoreF32`, cpu_ops.cpp).
__device__ inline float M2Load(const void* p, DType dt, int64_t i) {
  if (dt == DType::kF32) return static_cast<const float*>(p)[i];
  if (dt == DType::kF16) return __half2float(static_cast<const __half*>(p)[i]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

__device__ inline void M2Store(void* p, DType dt, int64_t i, float v) {
  if (dt == DType::kF32) {
    static_cast<float*>(p)[i] = v;
  } else if (dt == DType::kF16) {
    static_cast<__half*>(p)[i] = __float2half_rn(v);
  } else {
    static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(v);  // RNE, as host F32ToBF16
  }
}

// The value `v` as it reads back after a store/load round trip through `dt` —
// the `state_dtype` / `input_dtype` cast points (`RoundThrough`, cpu_ops.cpp).
__device__ inline float M2RoundThrough(DType dt, float v) {
  if (dt == DType::kF32) return v;
  if (dt == DType::kF16) return __half2float(__float2half_rn(v));
  return __bfloat162float(__float2bfloat16(v));
}

// softplus, guarded exactly as upstream: `tl.where(dt <= 20.0, softplus(dt), dt)`
// (ssd_chunk_state.py:94; the same guard at csrc/cpu/mamba_kernels.hpp:177).
__device__ inline float M2Softplus(float v) { return v <= 20.0f ? log1pf(expf(v)) : v; }

__device__ inline float M2Silu(float z) { return z / (1.0f + expf(-z)); }

// ─────────────────────────────────────────────────────────────────────────────
// stage 1 — `_chunk_cumsum_fwd` (ssd_chunk_state.py:300-346).
// One thread per (h, c): the prefix sum over the chunk is SEQUENTIAL, in the host
// reference's order. Positions past a partial chunk hold dt = 0 (:104-107), so
// dA_cumsum[..., cs-1] is the chunk's TOTAL decay whatever its length.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void M2CumsumKernel(float* dtv, float* dac, const void* dt_in, DType dt_dtype,
                               const float* A, const float* dbp, const int32_t* ccs, int64_t H,
                               int64_t nchunks, int64_t cs, bool softplus, float dt_min,
                               float dt_max) {
  const int64_t total = H * nchunks;
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < total;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t h = r / nchunks, c = r % nchunks;
    const float a = A[h];
    const int64_t start = ccs[c], len = ccs[c + 1] - start;
    const int64_t base = r * cs;
    float acc = 0.0f;
    for (int64_t i = 0; i < cs; ++i) {
      float d = 0.0f;
      if (i < len) {
        d = M2Load(dt_in, dt_dtype, (start + i) * H + h);
        if (dbp != nullptr) d += dbp[h];
        if (softplus) d = M2Softplus(d);
        d = fminf(fmaxf(d, dt_min), dt_max);
      }
      dtv[base + i] = d;
      acc += d * a;
      dac[base + i] = acc;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// stage 2 — `_chunk_state_fwd` (ssd_chunk_state.py:349-407).
// states[c,h,p,n] = sum_i x[i,h,p] * (B[i,g,n] * exp(min(dA_last - dA_i, 0)) * dt_i)
// f32 by upstream's own `states_in_fp32=True` (ssd_combined.py:100-102). One
// thread per (c,h,p,n), accumulating over i in the host reference's order.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void M2ChunkStateKernel(float* states, const void* x, DType xdt, const void* B,
                                   DType Bdt, const float* dtv, const float* dac,
                                   const int32_t* ccs, int64_t nchunks, int64_t H, int64_t P,
                                   int64_t G, int64_t N, int64_t cs, int64_t hpg) {
  const int64_t total = nchunks * H * P * N;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t n = idx % N;
    const int64_t p = (idx / N) % P;
    const int64_t h = (idx / (N * P)) % H;
    const int64_t c = idx / (N * P * H);
    const int64_t g = h / hpg;
    const int64_t start = ccs[c], len = ccs[c + 1] - start;
    const int64_t dbase = (h * nchunks + c) * cs;
    const float da_last = dac[dbase + cs - 1];
    float acc = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
      // The `min(., 0)` is upstream's and is an algebraic no-op inside the
      // enforced contract (`A < 0`, `dt >= 0` make dA_cumsum non-increasing over
      // i); it is kept because upstream keeps it.
      const float scale = expf(fminf(da_last - dac[dbase + i], 0.0f)) * dtv[dbase + i];
      if (scale == 0.0f) continue;
      const float xv = M2Load(x, xdt, ((start + i) * H + h) * P + p);
      const float bv = M2Load(B, Bdt, ((start + i) * G + g) * N + n) * scale;
      acc += xv * bv;
    }
    states[((c * H + h) * P + p) * N + n] = acc;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// stage 3 — `_state_passing_fwd` (ssd_state_passing.py:99-146).
//   S_c = exp(dA_last[c]) * S_{c-1} + states[c],  S_{-1} = initial_states[b]
// out[c] is the state AFTER chunk c (:90-97).
//
// THE RUNNING STATE STAYS F32 AND ONLY THE STORE ROUNDS: upstream carries
// `states` in f32 registers across the chunk loop and stores a `state_dtype` copy
// per chunk (:88-97) — it never reads that store back into the recurrence.
// `passed` is allocated at `state_dtype` here, NOT at the host reference's f32
// working width (cpu_ops.cpp records that width as one W2 must not inherit).
// One thread per (b,h,p,n), sequential over chunks.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void M2StatePassKernel(void* passed, DType sdt, void* final_states, const float* states,
                                  const float* dac, const int32_t* lci, const void* init,
                                  DType initdt, int64_t S, int64_t H, int64_t P, int64_t N,
                                  int64_t nchunks, int64_t cs) {
  const int64_t row = P * N;
  const int64_t total = S * H * row;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t i = idx % row;
    const int64_t h = (idx / row) % H;
    const int64_t b = idx / (row * H);
    // REGISTER-LOCAL MEMORY-SAFETY CLAMP (see the header's "what this arm does
    // not check"). The host arm value-checks `0 <= last_chunk_indices[b] <
    // nchunks` (cpu_ops.cpp); this arm cannot. Unclamped, `lci[b] >= nchunks`
    // makes the `M2Store(passed, ...)` below write PAST the cudaMallocAsync
    // allocation, and `lci[b-1] < -1` makes `states[...]` read before it. Both
    // bounds are already in registers, so the D2H argument for dropping the
    // check does not reach them and this costs nothing. It does NOT restore the
    // check: out-of-contract metadata still yields a wrong number, now with a
    // DEFINED shape — the chunk loop stops at nchunks and starts at 0.
    int64_t chunk_end = lci[b] + 1;
    int64_t chunk_start = b > 0 ? lci[b - 1] + 1 : 0;
    if (chunk_end > nchunks) chunk_end = nchunks;
    if (chunk_start < 0) chunk_start = 0;
    float s = init != nullptr ? M2Load(init, initdt, (b * H + h) * row + i) : 0.0f;
    for (int64_t c = chunk_start; c < chunk_end; ++c) {
      const float decay = expf(dac[(h * nchunks + c) * cs + cs - 1]);
      s = s * decay + states[(c * H + h) * row + i];
      M2Store(passed, sdt, (c * H + h) * row + i, s);
    }
    // `varlen_states = states[last_chunk_indices]` (ssd_combined.py:154).
    M2Store(final_states, sdt, (b * H + h) * row + i, s);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// stage 4 — `_bmm_chunk_fwd` (ssd_bmm.py:148-209).
// CB[c,g,i,j] = sum_n C[i,g,n] * B[j,g,n], f32 REGARDLESS of the activation dtype
// (`output_dtype=torch.float32`, ssd_combined.py:124). Only j <= i is ever read
// (IS_CAUSAL), so only j <= i is written — as in the host reference.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void M2BmmKernel(float* cb, const void* Bp, DType Bdt, const void* Cp, DType Cdt,
                            const int32_t* ccs, int64_t nchunks, int64_t G, int64_t N,
                            int64_t cs) {
  const int64_t total = nchunks * G * cs * cs;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t j = idx % cs;
    const int64_t i = (idx / cs) % cs;
    if (j > i) continue;
    const int64_t g = (idx / (cs * cs)) % G;
    const int64_t c = idx / (cs * cs * G);
    const int64_t start = ccs[c], len = ccs[c + 1] - start;
    if (i >= len) continue;
    float acc = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
      acc += M2Load(Cp, Cdt, ((start + i) * G + g) * N + n) *
             M2Load(Bp, Bdt, ((start + j) * G + g) * N + n);
    }
    cb[((c * G + g) * cs + i) * cs + j] = acc;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// stage 5 — `_chunk_scan_fwd` (ssd_chunk_scan.py:216-525).
//   out_i = exp(dA_i) * (C_i . S_{c-1})                                  inter
//         + sum_{j<=i} CB[i,j] * exp(min(dA_i - dA_j, 0)) * dt_j * x_j    intra
//         + D * x_i                                                      skip
//   then `out *= z * sigmoid(z)` when z is given (:394-406).
// `S_{c-1}` is `initial_states[seq_idx[c]]` when this chunk opens a new sequence
// AND initial states were supplied, ZEROS when they were not (:236-250, :271-289)
// — that is what makes a sequence boundary INSIDE a physical chunk correct.
// One thread per (c,h,i,p); both accumulations run in the host arm's order.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void M2ChunkScanKernel(void* out, DType odt, const void* x, DType xdt, const void* Cp,
                                  DType Cdt, const float* cb, const float* dtv, const float* dac,
                                  const void* passed, DType sdt, const void* init, DType initdt,
                                  const float* D, bool d_has_hdim, const void* z, DType zdt,
                                  const int32_t* ccs, const int32_t* sidx, int64_t nchunks,
                                  int64_t H, int64_t P, int64_t G, int64_t N, int64_t S,
                                  int64_t cs, int64_t hpg) {
  const int64_t total = nchunks * H * cs * P;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t p = idx % P;
    const int64_t i = (idx / P) % cs;
    const int64_t h = (idx / (P * cs)) % H;
    const int64_t c = idx / (P * cs * H);
    const int64_t start = ccs[c], len = ccs[c + 1] - start;
    if (i >= len) continue;
    const int64_t g = h / hpg;
    const int64_t dbase = (h * nchunks + c) * cs;
    const int64_t row = P * N;

    const int32_t si = sidx[c];
    const int32_t si_prev = c >= 1 ? sidx[c - 1] : -1;
    // REGISTER-LOCAL MEMORY-SAFETY CLAMP, as in M2StatePassKernel above. The
    // host arm value-checks `seq_idx[c] in [0,S)` (cpu_ops.cpp); unclamped, an
    // out-of-range `si` reads `initial_states` out of bounds at `prevbase`, and
    // a `sidx[0] < 0` makes `si == si_prev` at c == 0 and indexes `passed` at
    // chunk -1. In-contract `si` is ALWAYS in range, so every in-contract path
    // below is bit-identical to the unclamped form; out of contract the answer
    // is still wrong, now with the defined shape "this chunk opens with a zero
    // previous state".
    const bool si_ok = si >= 0 && static_cast<int64_t>(si) < S;
    bool prev_zero = false;
    const void* prevp = passed;
    DType prevdt = sdt;
    int64_t prevbase = ((c - 1) * H + h) * row;
    if (!si_ok) {
      prev_zero = true;
    } else if (si != si_prev) {
      if (init != nullptr) {
        prevp = init;
        prevdt = initdt;
        prevbase = (static_cast<int64_t>(si) * H + h) * row;
      } else {
        prev_zero = true;
      }
    }

    const float da_i = dac[dbase + i];
    const float scale_m = expf(da_i);
    float acc = 0.0f;
    if (!prev_zero) {
      for (int64_t n = 0; n < N; ++n) {
        acc += M2Load(Cp, Cdt, ((start + i) * G + g) * N + n) *
               M2Load(prevp, prevdt, prevbase + p * N + n);
      }
    }
    acc *= scale_m;
    const float* cbc = cb + (c * G + g) * cs * cs;
    for (int64_t j = 0; j <= i; ++j) {
      const float w = cbc[i * cs + j] * expf(fminf(da_i - dac[dbase + j], 0.0f)) * dtv[dbase + j];
      acc += w * M2Load(x, xdt, ((start + j) * H + h) * P + p);
    }
    const float xi = M2Load(x, xdt, ((start + i) * H + h) * P + p);
    if (D != nullptr) acc += (d_has_hdim ? D[h * P + p] : D[h]) * xi;
    if (z != nullptr) acc *= M2Silu(M2Load(z, zdt, ((start + i) * H + h) * P + p));
    M2Store(out, odt, ((start + i) * H + h) * P + p, acc);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// vt::Mamba2StateUpdate — `selective_state_update` (ops/mamba_ssm.py:497+) at the
// scalar-per-head shape (csrc/cpu/mamba_kernels.hpp:104-250).
// One thread per (b,h,p), sequential over n exactly as the host arm.
//
// The readout uses the F32 value, not the value re-read from the cache: the
// Triton kernel holds `state` in registers and computes `out = sum(state * C)`
// from them, storing the cache-width copy separately (mamba_ssm.py:433,451).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void M2StateUpdateKernel(void* out, DType odt, void* state, DType sdt, const void* x,
                                    DType xdt, const void* dtp, DType dtdt, const float* A,
                                    const void* Bp, DType Bdt, const void* Cp, DType Cdt,
                                    const float* D, const void* z, DType zdt, const float* dbp,
                                    const int32_t* sidx, int64_t Nb, int64_t H, int64_t P,
                                    int64_t G, int64_t N, int64_t S, int64_t hpg,
                                    bool softplus) {
  const int64_t total = Nb * H * P;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t p = idx % P;
    const int64_t h = (idx / P) % H;
    const int64_t b = idx / (P * H);
    int64_t slot = b;
    if (sidx != nullptr) {
      // LOCAL ABI: index < 0 is the NULL row — its cache slot is untouched
      // (`continue`, mamba_kernels.hpp:147) and its output row is zeroed, as
      // GdnDecode already models it.
      if (sidx[b] < 0) {
        M2Store(out, odt, (b * H + h) * P + p, 0.0f);
        continue;
      }
      slot = sidx[b];
      // The host arm REFUSES an out-of-range slot; a device kernel cannot throw,
      // so it writes nothing at all rather than out of bounds (see the header
      // note on unchecked device-side preconditions).
      if (slot >= S) continue;
    }
    const int64_t gg = h / hpg;
    float d = M2Load(dtp, dtdt, b * H + h);
    if (dbp != nullptr) d += dbp[h];
    if (softplus) d = M2Softplus(d);
    const float dA = expf(A[h] * d);
    const float xv = M2Load(x, xdt, (b * H + h) * P + p);
    const int64_t sbase = ((slot * H + h) * P + p) * N;
    float y = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
      const float bv = M2Load(Bp, Bdt, (b * G + gg) * N + n);
      const float cv = M2Load(Cp, Cdt, (b * G + gg) * N + n);
      const float sn = M2Load(state, sdt, sbase + n) * dA + bv * xv * d;
      M2Store(state, sdt, sbase + n, sn);
      y += sn * cv;
    }
    if (D != nullptr) y += D[h] * xv;
    if (z != nullptr) y *= M2Silu(M2Load(z, zdt, (b * H + h) * P + p));
    M2Store(out, odt, (b * H + h) * P + p, y);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// vt::RmsNormGatedGroup — `Mixer2RMSNormGated.forward_native` (mamba_mixer2.py:100-149).
//   v   = x * silu(f32(gate))                                             (:114)
//   out = weight * dtype(x)( v * rsqrt(mean(v^2 over its group) + eps) )   (:136-141, :149)
//
// One BLOCK per (row, group). This is the ONE accumulation in this file whose
// order differs from the host arm's sequential sum: a per-group reduction is the
// whole shape of the op, and a block reduction is how it is done on device. The
// summands are all NON-NEGATIVE (they are squares), so there is no cancellation
// and the reordering carries the plain forward-error bound the tests state.
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline float M2BlockReduceSum(float v) {
  __shared__ float smem[kM2Block / 32];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int nwarps = static_cast<int>(blockDim.x) >> 5;
    for (int i = 0; i < nwarps; ++i) total += smem[i];
    smem[0] = total;
  }
  __syncthreads();
  const float out = smem[0];
  __syncthreads();  // smem is reused by the next (row, group) this block takes
  return out;
}

__global__ void M2GatedNormKernel(void* out, DType odt, const void* x, DType xdt, DType input_dt,
                                  const void* gate, DType gdt, const void* w, DType wdt,
                                  bool has_w, int64_t nblocks, int64_t hidden, int64_t n_groups,
                                  int64_t group_size, float eps) {
  for (int64_t blk = blockIdx.x; blk < nblocks; blk += gridDim.x) {
    const int64_t r = blk / n_groups, gsel = blk % n_groups;
    const int64_t off = r * hidden + gsel * group_size;
    float part = 0.0f;
    for (int64_t j = threadIdx.x; j < group_size; j += blockDim.x) {
      // The gate is promoted to f32 BEFORE the silu (:114).
      const float v = M2Load(x, xdt, off + j) * M2Silu(M2Load(gate, gdt, off + j));
      if (!has_w) {
        // use_rms_norm == False: no parameter, no norm (:94-96, :115-116).
        M2Store(out, odt, off + j, M2RoundThrough(input_dt, v));
      }
      part += v * v;
    }
    if (!has_w) continue;
    // f32 accumulation, NOT double: upstream reduces `x.pow(2).mean(-1)` in f32.
    const float ss = M2BlockReduceSum(part);
    // `rsqrt(variance + eps)` (:130, :141) — eps is INSIDE the square root.
    // Written as 1/sqrt to match the host reference's rounding, not `rsqrtf`.
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(group_size) + eps);
    for (int64_t j = threadIdx.x; j < group_size; j += blockDim.x) {
      const float v = M2Load(x, xdt, off + j) * M2Silu(M2Load(gate, gdt, off + j));
      const float normed = M2RoundThrough(input_dt, v * inv);
      M2Store(out, odt, off + j, M2Load(w, wdt, gsel * group_size + j) * normed);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// host launchers
// ─────────────────────────────────────────────────────────────────────────────

void Mamba2ChunkScanKernelCuda(Queue& q, Tensor& out, Tensor& final_states, const Tensor& x,
                               const Tensor& dt_in, const Tensor& A, const Tensor& B,
                               const Tensor& C, const Tensor* D, const Tensor* z,
                               const Tensor* dt_bias, const Tensor* initial_states,
                               const Tensor& /*cu_seqlens*/, const Tensor& cu_chunk_seqlens,
                               const Tensor& last_chunk_indices, const Tensor& seq_idx,
                               const Mamba2Args& args) {
  const int64_t T = x.shape[0], H = x.shape[1], P = x.shape[2];
  const int64_t G = B.shape[1], N = B.shape[2];
  const int64_t S = final_states.shape[0];
  const int64_t cs = args.chunk_size;
  const int64_t nchunks = cu_chunk_seqlens.shape[0] - 1;
  const int64_t hpg = H / G;  // nheads_ngroups_ratio (ssd_chunk_state.py:238)
  if (T == 0 || nchunks == 0) return;

  cudaStream_t s = M2Stream(q);
  const int32_t* ccs = cu_chunk_seqlens.Ptr<int32_t>();
  const int32_t* lci = last_chunk_indices.Ptr<int32_t>();
  const int32_t* sidx = seq_idx.Ptr<int32_t>();
  const float* Ap = A.Ptr<float>();
  const float* dbp = dt_bias != nullptr ? dt_bias->Ptr<float>() : nullptr;
  const float* Dp = D != nullptr ? D->Ptr<float>() : nullptr;
  const bool d_has_hdim = D != nullptr && D->rank == 2;
  const DType sdt = final_states.dtype;  // `state_dtype` (ssd_combined.py:46,119,176)

  // Per-call scratch on the stream's memory pool, exactly as the sibling varlen
  // prefill KdaChunkPrefill does (cuda_gdn.cu). This is the PREFILL path; the
  // decode kernel below allocates nothing.
  const size_t n_cumsum = static_cast<size_t>(H * nchunks * cs);
  const size_t n_states = static_cast<size_t>(nchunks * H * P * N);
  const size_t n_cb = static_cast<size_t>(nchunks * G * cs * cs);
  const size_t state_elem = sdt == DType::kF32 ? 4u : 2u;
  // `M2Check` THROWS, so the five allocations below are held by a scope guard:
  // without it a failure on the 3rd leaks the 1st and 2nd. The guard is released
  // once the explicit frees at the end of the happy path have run.
  M2Scratch scratch(s);
  float* dtv = static_cast<float*>(scratch.Alloc(n_cumsum * sizeof(float), "dtv alloc"));
  float* dac = static_cast<float*>(scratch.Alloc(n_cumsum * sizeof(float), "dac alloc"));
  float* states = static_cast<float*>(scratch.Alloc(n_states * sizeof(float), "states alloc"));
  float* cb = static_cast<float*>(scratch.Alloc(n_cb * sizeof(float), "cb alloc"));
  void* passed = scratch.Alloc(n_states * state_elem, "passed alloc");
  // cudaMallocAsync hands back DIRTY pool memory. Every element of dtv/dac/states
  // is written before it is read; `cb` and `passed` are written only where they
  // are read (the causal triangle, and the chunks of a scheduled sequence), so
  // they are zeroed rather than left to an initcheck report.
  M2Check(cudaMemsetAsync(cb, 0, n_cb * sizeof(float), s), "cb zero");
  M2Check(cudaMemsetAsync(passed, 0, n_states * state_elem, s), "passed zero");

  M2CumsumKernel<<<M2Grid(H * nchunks), kM2Block, 0, s>>>(
      dtv, dac, dt_in.data, dt_in.dtype, Ap, dbp, ccs, H, nchunks, cs, args.dt_softplus,
      args.dt_min, args.dt_max);
  M2ChunkStateKernel<<<M2Grid(static_cast<int64_t>(n_states)), kM2Block, 0, s>>>(
      states, x.data, x.dtype, B.data, B.dtype, dtv, dac, ccs, nchunks, H, P, G, N, cs, hpg);
  M2StatePassKernel<<<M2Grid(static_cast<int64_t>(S) * H * P * N), kM2Block, 0, s>>>(
      passed, sdt, final_states.data, states, dac, lci,
      initial_states != nullptr ? initial_states->data : nullptr,
      initial_states != nullptr ? initial_states->dtype : DType::kF32, S, H, P, N, nchunks, cs);
  M2BmmKernel<<<M2Grid(static_cast<int64_t>(n_cb)), kM2Block, 0, s>>>(
      cb, B.data, B.dtype, C.data, C.dtype, ccs, nchunks, G, N, cs);
  M2ChunkScanKernel<<<M2Grid(nchunks * H * cs * P), kM2Block, 0, s>>>(
      out.data, out.dtype, x.data, x.dtype, C.data, C.dtype, cb, dtv, dac, passed, sdt,
      initial_states != nullptr ? initial_states->data : nullptr,
      initial_states != nullptr ? initial_states->dtype : DType::kF32, Dp, d_has_hdim,
      z != nullptr ? z->data : nullptr, z != nullptr ? z->dtype : DType::kF32, ccs, sidx,
      nchunks, H, P, G, N, S, cs, hpg);

  const cudaError_t launched = cudaGetLastError();
  scratch.Release();  // frees all five on the stream, reporting the first error
  M2Check(launched, "mamba2_chunk_scan launch");
}

void Mamba2StateUpdateKernelCuda(Queue& q, Tensor& out, Tensor& state, const Tensor& x,
                                 const Tensor& dt_in, const Tensor& A, const Tensor& B,
                                 const Tensor& C, const Tensor* D, const Tensor* z,
                                 const Tensor* dt_bias, const Tensor* state_indices,
                                 const Mamba2Args& args) {
  const int64_t Nb = x.shape[0], H = x.shape[1], P = x.shape[2];
  const int64_t G = B.shape[1], N = B.shape[2];
  const int64_t S = state.shape[0];
  const int64_t hpg = H / G;
  if (Nb == 0) return;
  cudaStream_t s = M2Stream(q);
  M2StateUpdateKernel<<<M2Grid(Nb * H * P), kM2Block, 0, s>>>(
      out.data, out.dtype, state.data, state.dtype, x.data, x.dtype, dt_in.data, dt_in.dtype,
      A.Ptr<float>(), B.data, B.dtype, C.data, C.dtype,
      D != nullptr ? D->Ptr<float>() : nullptr, z != nullptr ? z->data : nullptr,
      z != nullptr ? z->dtype : DType::kF32, dt_bias != nullptr ? dt_bias->Ptr<float>() : nullptr,
      state_indices != nullptr ? state_indices->Ptr<int32_t>() : nullptr, Nb, H, P, G, N, S, hpg,
      args.dt_softplus);
  M2Check(cudaGetLastError(), "mamba2_state_update launch");
}

void RmsNormGatedGroupKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                                 const Tensor* weight, const RmsNormGatedGroupArgs& args) {
  const int64_t hidden = x.shape[x.rank - 1];
  int64_t rows = 1;
  for (int r = 0; r < x.rank - 1; ++r) rows *= x.shape[r];
  if (rows == 0 || hidden == 0) return;
  const int64_t group_size = hidden / args.n_groups;
  const int64_t nblocks = rows * args.n_groups;
  // `input_dtype = x.dtype` (:113) is the width the normalized value is cast back
  // to before the weight multiply (`self.weight * x.to(input_dtype)`, :149).
  cudaStream_t s = M2Stream(q);
  const unsigned grid = static_cast<unsigned>(nblocks < 65535 ? nblocks : 65535);
  M2GatedNormKernel<<<grid, kM2Block, 0, s>>>(
      out.data, out.dtype, x.data, x.dtype, x.dtype, gate.data, gate.dtype,
      weight != nullptr ? weight->data : nullptr,
      weight != nullptr ? weight->dtype : DType::kF32, weight != nullptr, nblocks, hidden,
      args.n_groups, group_size, args.eps);
  M2Check(cudaGetLastError(), "rms_norm_gated_group launch");
}

}  // namespace
}  // namespace vt::cuda::mamba2

#endif  // VT_CUDA_MAMBA2_SSD_CUH_
