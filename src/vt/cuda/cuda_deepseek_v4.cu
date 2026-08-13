// DeepSeek-V4-Flash W7-device — CUDA kernels for the four NEW V4 op families,
// each a 1:1 device port of the landed portable HOST reference and unit-gated
// against it on the DGX GB10 at small shape (tests/vllm/models/
// test_cuda_deepseek_v4.cpp). Registered through the vt OpProvider seam
// (kDeepseekV4{Mhc,Dsa,Compressor,Moe}) so DeepseekV4Model::ForwardDevice can
// dispatch them. See include/.../deepseek_v4_device.h for the seam + honest scope.
//
// ─── PORT MAP (OURS <- host reference <- upstream file:line) ─────────────────
//   MhcSinkhorn/Pre/Post/Head <- deepseek_v4_mhc.cpp   <- kernels/mhc/torch.py:56-106,
//                                                          triton.py:108-140
//   DsaWeightFold/Logits/Topk/SoftmaxSink/GroupedOLora
//                             <- deepseek_v4_dsa.cpp    <- sparse_attn_indexer.py:203-207,
//                                :488-497; triton_fp8_mqa_logits.py:120-156;
//                                flashinfer_sparse.py:777,:896; nvidia/ops/o_proj.py:58-73
//   CompressorSaveScoreApe/PoolNorm/Fp8DsMlaEncode/Decode
//                             <- deepseek_v4_compressor.cpp <- save_partial_states.py:92-101,
//                                fused_compress_quant_cache.py:198-297, compressor.py:307-309
//   SqrtSoftplus/Route/ClampedSwiGLU
//                             <- deepseek_v4_moe.cpp    <- fused_topk_bias_router.py:75-118,
//                                activation.py:197-201
//
// These are correctness-grade STRUCTURAL kernels (tiny-shape gate, per-op host
// round-trip), NOT the fused/perf path — the 512-wide MLA attention + expert
// grouped-GEMM REUSE the existing NVFP4/FP8 kernels and are not re-ported. The
// real paged-engine e2e over a materialized 167B checkpoint stays the W8 residual.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_device.h"
#include "vt/ops.h"

// Defined in cuda_quant_dot.cu (same CUDA library): the fused routed-MoE
// gate+up+SwiGLU epilogue over the keep-quant expert towers.
namespace vt::cuda {
void MoeGateUpSwiGLUGroupedCuda(vt::Queue& q, vt::Tensor& out, const vt::Tensor& act,
                               const vt::Tensor& gate_w, const vt::Tensor& up_w,
                               const vt::Tensor& expert_ids, float limit);
// Brick 12 (ds4-gap launch consolidation): paired + block-diagonal Q8_0 decode GEMVs.
void MatmulQ8_0PairCuda(vt::Tensor& out0, vt::Tensor& out1, const vt::Tensor& a,
                        const vt::Tensor& b0, const vt::Tensor& b1, cudaStream_t s);
void MatmulQ8_0GroupDiagCuda(vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b,
                             int64_t ng, cudaStream_t s);
}  // namespace vt::cuda

namespace vllm::deepseek_v4 {
namespace {

using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;
using vt::Tensor;

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("vt cuda deepseek_v4: ") + what + ": " +
                             cudaGetErrorString(e));
}
cudaStream_t AsStream(Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Minimal owning device buffer (tiny shapes; correctness path).
struct Dev {
  void* p = nullptr;
  explicit Dev(size_t bytes) { Check(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc"); }
  ~Dev() {
    if (p) cudaFree(p);
  }
  Dev(Dev&& o) noexcept : p(o.p) { o.p = nullptr; }
  Dev& operator=(Dev&&) = delete;
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};

template <class T>
Dev Upload(const std::vector<T>& v, cudaStream_t s) {
  Dev d(v.size() * sizeof(T));
  if (!v.empty())
    Check(cudaMemcpyAsync(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s),
          "H2D");
  return d;
}
template <class T>
void Download(std::vector<T>& v, void* dp, cudaStream_t s) {
  if (!v.empty())
    Check(cudaMemcpyAsync(v.data(), dp, v.size() * sizeof(T), cudaMemcpyDeviceToHost, s),
          "D2H");
}

// ── device math helpers (bit-faithful to the host references) ─────────────────
__device__ inline float Sig(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float SqrtSoftplusDev(float x) {
  const float sp = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
  return sqrtf(sp);
}
__device__ inline float Bf16Round(float x) { return __bfloat162float(__float2bfloat16(x)); }

// ============================================================================
// (1) MHC family
// ============================================================================
// Sinkhorn of an hc×hc matrix (torch.py:75-82): row-softmax+eps seed, col-norm,
// then (iters-1)×[row-norm, col-norm]. Single thread (hc small); m in local mem.
__device__ void SinkhornInplace(const float* logits, float* m, int hc, int iters, float eps) {
  for (int j = 0; j < hc; ++j) {
    float rmax = logits[j * hc];
    for (int k = 1; k < hc; ++k) rmax = fmaxf(rmax, logits[j * hc + k]);
    float rsum = 0.0f;
    for (int k = 0; k < hc; ++k) {
      const float e = expf(logits[j * hc + k] - rmax);
      m[j * hc + k] = e;
      rsum += e;
    }
    for (int k = 0; k < hc; ++k) m[j * hc + k] = m[j * hc + k] / rsum + eps;
  }
  for (int k = 0; k < hc; ++k) {
    float c = 0.0f;
    for (int j = 0; j < hc; ++j) c += m[j * hc + k];
    const float den = c + eps;
    for (int j = 0; j < hc; ++j) m[j * hc + k] /= den;
  }
  for (int it = 0; it < iters - 1; ++it) {
    for (int j = 0; j < hc; ++j) {
      float r = 0.0f;
      for (int k = 0; k < hc; ++k) r += m[j * hc + k];
      const float den = r + eps;
      for (int k = 0; k < hc; ++k) m[j * hc + k] /= den;
    }
    for (int k = 0; k < hc; ++k) {
      float c = 0.0f;
      for (int j = 0; j < hc; ++j) c += m[j * hc + k];
      const float den = c + eps;
      for (int j = 0; j < hc; ++j) m[j * hc + k] /= den;
    }
  }
}

__global__ void SinkhornKernel(const float* logits, float* out, int hc, int iters, float eps) {
  float m[256];  // hc <= 16
  SinkhornInplace(logits, m, hc, iters, eps);
  for (int i = 0; i < hc * hc; ++i) out[i] = m[i];
}

// hc==4 specialization of SinkhornInplace (the DeepSeek-V4 hc_mult is structurally 4:
// (2+hc)*hc=24, hc*H=16384 ⇒ hc=4). Hardcoding hc=4 with #pragma unroll lets ptxas keep
// `m[16]` (and the caller's cl[16]) in REGISTERS instead of the 2048-byte LOCAL stack frame
// the generic runtime-`hc` version forces (dynamic index ⇒ off-chip local memory). ncu on
// GB10 pins the finish kernel as ~86% CTA-barrier + local-memory-scoreboard bound: 31 warps
// idle at the __syncthreads while thread 0 grinds the Sinkhorn through 4-byte-of-32 local
// DRAM transactions. Register residency collapses that (2048→0 B stack frame). BYTE-IDENTICAL
// to SinkhornInplace for hc=4: same op sequence, same accumulation order (k asc, j asc; eps
// added last) — verified 0/16 ULP on a locked-clock GB10 A/B. Mirrors ds4's hc4_split_one
// (~/w8run/ds4/ds4_cuda.cu:9618, its `float c[16]` fully-unrolled register split).
__device__ inline void SinkhornInplace4(const float* __restrict__ logits, float* m, int iters,
                                        float eps) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float rmax = logits[j * 4];
#pragma unroll
    for (int k = 1; k < 4; ++k) rmax = fmaxf(rmax, logits[j * 4 + k]);
    float rsum = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float e = expf(logits[j * 4 + k] - rmax);
      m[j * 4 + k] = e;
      rsum += e;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) m[j * 4 + k] = m[j * 4 + k] / rsum + eps;
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float c = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) c += m[j * 4 + k];
    const float den = c + eps;
#pragma unroll
    for (int j = 0; j < 4; ++j) m[j * 4 + k] /= den;
  }
  for (int it = 0; it < iters - 1; ++it) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float r = 0.0f;
#pragma unroll
      for (int k = 0; k < 4; ++k) r += m[j * 4 + k];
      const float den = r + eps;
#pragma unroll
      for (int k = 0; k < 4; ++k) m[j * 4 + k] /= den;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float c = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) c += m[j * 4 + k];
      const float den = c + eps;
#pragma unroll
      for (int j = 0; j < 4; ++j) m[j * 4 + k] /= den;
    }
  }
}

// MhcPre (torch.py:56-91 + folded RMSNorm). Single thread; mixes/scratch global.
__global__ void MhcPreKernel(const float* residual, const float* fn, const float* scale,
                             const float* base, int hc, int hidden, float rms_eps,
                             float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult,
                             int iters, const float* norm_weight, int has_norm, float norm_eps,
                             float* mixes, float* pre_out, float* post_out, float* comb_out,
                             float* layer_out) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  double sqrsum = 0.0;
  for (int i = 0; i < flat; ++i) {
    const double r = residual[i];
    sqrsum += r * r;
  }
  for (int o = 0; o < hc3; ++o) {
    float acc = 0.0f;
    const int frow = o * flat;
    for (int i = 0; i < flat; ++i) acc += residual[i] * fn[frow + i];
    mixes[o] = acc;
  }
  const float rms =
      1.0f / sqrtf(static_cast<float>(sqrsum / static_cast<double>(flat)) + rms_eps);
  for (int o = 0; o < hc3; ++o) mixes[o] *= rms;
  for (int j = 0; j < hc; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
  for (int j = 0; j < hc; ++j)
    post_out[j] = Sig(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult;
  float cl[256];  // hc*hc, hc<=16
  for (int j = 0; j < hc; ++j)
    for (int k = 0; k < hc; ++k) {
      const int idx = j * hc + k;
      cl[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
    }
  float m[256];
  SinkhornInplace(cl, m, hc, iters, hc_sinkhorn_eps);
  for (int i = 0; i < hc * hc; ++i) comb_out[i] = m[i];
  for (int h = 0; h < hidden; ++h) {
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
  }
  if (has_norm) {
    double ss = 0.0;
    for (int h = 0; h < hidden; ++h) {
      const double v = layer_out[h];
      ss += v * v;
    }
    const float r =
        1.0f / sqrtf(static_cast<float>(ss / static_cast<double>(hidden)) + norm_eps);
    for (int h = 0; h < hidden; ++h) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

// Brick B / glue-tune — PARALLEL MhcPre, split into TWO kernels so the hc3 mix
// dot-products (the profiled 25.5%-of-step hot spot, previously 24 SEQUENTIAL block
// reductions inside ONE block = ONE SM) run CONCURRENTLY across hc3 blocks (many
// SMs). Numerics are BIT-IDENTICAL to the prior one-block kernel: each dot is the
// SAME 256-thread block reduction (same partial assignment + tree), just in its own
// block; `acc*rms` is a scalar mult, order-independent. The tiny hc-sized gates +
// 20-iter Sinkhorn stay on thread 0 in HOST ORDER; the per-h layer_out dot (over hc)
// is sequential → order kept. Characterized near-tie vs the single-thread #183 (the
// double block reductions reorder vs single-thread float) — UNCHANGED tolerance.
//
// Kernel A: one block per mix dot `o` → mixes[o] = Σ_i residual[i]*fn[o*flat+i] (RAW).
__global__ void MhcPreDotsKernel(const float* __restrict__ residual,
                                 const float* __restrict__ fn, int flat, float* __restrict__ mixes) {
  const int o = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ double red[];
  const int frow = o * flat;
  double la = 0.0;
  for (int i = tid; i < flat; i += nt) la += static_cast<double>(residual[i]) * fn[frow + i];
  red[tid] = la;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  if (tid == 0) mixes[o] = static_cast<float>(red[0]);  // RAW dot; kernel B applies rms
}
// Kernel B (one block): sqrsum→rms, mixes[o]*=rms, gates+Sinkhorn (thread0),
// layer_out (parallel over hidden), optional folded final RMSNorm.
__global__ void MhcPreFinishKernel(const float* __restrict__ residual,
                                   const float* __restrict__ scale, const float* __restrict__ base,
                                   int hc, int hidden, float rms_eps, float hc_pre_eps,
                                   float hc_sinkhorn_eps, float hc_post_mult, int iters,
                                   const float* __restrict__ norm_weight, int has_norm,
                                   float norm_eps, float* mixes, float* pre_out, float* post_out,
                                   float* comb_out, float* layer_out) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ double red[];
  auto block_reduce = [&](double v) -> double {
    red[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const double r = red[0];
    __syncthreads();
    return r;
  };
  double ls = 0.0;
  for (int i = tid; i < flat; i += nt) { const double r = residual[i]; ls += r * r; }
  const double sqrsum = block_reduce(ls);
  const float rms = 1.0f / sqrtf(static_cast<float>(sqrsum / static_cast<double>(flat)) + rms_eps);
  for (int o = tid; o < hc3; o += nt) mixes[o] = mixes[o] * rms;  // raw dot → *rms (order-indep)
  __syncthreads();
  if (tid == 0) {
    for (int j = 0; j < hc; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
    for (int j = 0; j < hc; ++j)
      post_out[j] = Sig(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult;
    float cl[256], m[256];
    for (int j = 0; j < hc; ++j)
      for (int k = 0; k < hc; ++k) {
        const int idx = j * hc + k;
        cl[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
      }
    SinkhornInplace(cl, m, hc, iters, hc_sinkhorn_eps);
    for (int i = 0; i < hc * hc; ++i) comb_out[i] = m[i];
  }
  __syncthreads();
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
  }
  __syncthreads();
  if (has_norm) {
    double ss = 0.0;
    for (int h = tid; h < hidden; h += nt) { const double v = layer_out[h]; ss += v * v; }
    const double ssr = block_reduce(ss);
    const float r =
        1.0f / sqrtf(static_cast<float>(ssr / static_cast<double>(hidden)) + norm_eps);
    for (int h = tid; h < hidden; h += nt) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

// ── ds4-fold (VT_V4_MHC_FUSED): the FLOAT MHC-pre, a 1:1 structural mirror of ds4's
//    hc4_split_one (~/w8run/ds4/ds4_cuda.cu:9618) + hc_split_weighted_sum_norm_fused_kernel
//    (:9752). The double-accumulating MhcPreDots/Finish kernels above are BIT-FAITHFUL to
//    the host reference, but GB10 (sm_121a) executes FP64 at ~1/32-1/64 of FP32, so the mix
//    dots + the two block-RMS reductions pay a large FP64 penalty on the decode-graph hot
//    path. ds4 runs the SAME algebra in FLOAT (its fused kernel is float throughout). These
//    two kernels keep OUR pre/post/comb output layout (so MhcPost/HcHead are unchanged) and
//    the SAME SinkhornInplace (already float → the Sinkhorn is IDENTICAL to the double path);
//    the ONLY change is the mix dot + the sqrsum/norm reductions moving double→float. That is
//    a CHARACTERIZED near-tie vs the double path (float reduction reorder + float accumulate),
//    the same class as every other MHC glue kernel here — distributional-gated on the IQ2XXS
//    greedy (which is non-deterministic), and token-checked A/B vs the double path.
//
// Kernel A' (float mix dots): one block per output o, mirrors the ds4 mix matvec (float).
// Lever 2 (fold_sqrsum): block o==0 ALSO reduces Σresidual² over flat — it already streams
// the whole residual for its dot, so the sqrsum is nearly free here — and stashes it in
// mixes[hc3]. That removes the DUPLICATE residual pass MhcPreFinishFloatKernel used to do
// (it read the residual once for the sqrsum and again for the weighted sum). BYTE-EXACT: the
// sqrsum is the SAME 256-thread block-tree reduction over the SAME residual/stride the finish
// used, only relocated to the dots kernel (which runs first on the same stream).
__global__ void MhcPreDotsFloatKernel(const float* __restrict__ residual,
                                      const float* __restrict__ fn, int flat,
                                      float* __restrict__ mixes, int fold_sqrsum, int hc3) {
  const int o = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ float redf[];
  const int frow = o * flat;
  float la = 0.0f, sq = 0.0f;
  for (int i = tid; i < flat; i += nt) {
    const float r = residual[i];
    la += r * fn[frow + i];
    if (fold_sqrsum && o == 0) sq += r * r;
  }
  redf[tid] = la;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
  if (tid == 0) mixes[o] = redf[0];  // RAW float dot; kernel B' applies rms
  if (fold_sqrsum && o == 0) {
    __syncthreads();  // reuse redf for the sqrsum reduction
    redf[tid] = sq;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
    if (tid == 0) mixes[hc3] = redf[0];  // Σresidual² → finish reads this instead of re-summing
  }
}
// Kernel B' (float finish): sqrsum→rms, mixes*=rms, gates+Sinkhorn (thread0), weighted-sum
// layer_out (parallel over hidden), optional folded final RMSNorm — all FLOAT, mirroring the
// ds4 fused kernel's rsqrtf + float partial[] reduction.
__global__ void MhcPreFinishFloatKernel(const float* __restrict__ residual,
                                        const float* __restrict__ scale,
                                        const float* __restrict__ base, int hc, int hidden,
                                        float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                                        float hc_post_mult, int iters,
                                        const float* __restrict__ norm_weight, int has_norm,
                                        float norm_eps, float* mixes, float* pre_out,
                                        float* post_out, float* comb_out, float* layer_out,
                                        int fold_sqrsum) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ float redf[];
  auto block_reduce = [&](float v) -> float {
    redf[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
    const float r = redf[0];
    __syncthreads();
    return r;
  };
  // Lever 2: reuse the sqrsum MhcPreDotsFloatKernel already computed (block 0 stashed it in
  // mixes[hc3]) instead of streaming the residual a second time. fold_sqrsum=0 keeps the
  // in-kernel reduction (baseline A/B, VT_V4_MHC_LEAN=0).
  float sqrsum;
  if (fold_sqrsum) {
    sqrsum = mixes[hc3];
  } else {
    float ls = 0.0f;
    for (int i = tid; i < flat; i += nt) { const float r = residual[i]; ls += r * r; }
    sqrsum = block_reduce(ls);
  }
  const float rms = rsqrtf(sqrsum / static_cast<float>(flat) + rms_eps);
  for (int o = tid; o < hc3; o += nt) mixes[o] = mixes[o] * rms;  // raw dot → *rms (order-indep)
  __syncthreads();
  if (tid == 0) {
    for (int j = 0; j < hc; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
    for (int j = 0; j < hc; ++j)
      post_out[j] = Sig(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult;
    float cl[256], m[256];
    for (int j = 0; j < hc; ++j)
      for (int k = 0; k < hc; ++k) {
        const int idx = j * hc + k;
        cl[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
      }
    SinkhornInplace(cl, m, hc, iters, hc_sinkhorn_eps);
    for (int i = 0; i < hc * hc; ++i) comb_out[i] = m[i];
  }
  __syncthreads();
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
  }
  __syncthreads();
  if (has_norm) {
    float ss = 0.0f;
    for (int h = tid; h < hidden; h += nt) { const float v = layer_out[h]; ss += v * v; }
    const float ssr = block_reduce(ss);
    const float r = rsqrtf(ssr / static_cast<float>(hidden) + norm_eps);
    for (int h = tid; h < hidden; h += nt) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

// ── Kernel B'' (VT_V4_MHC_SINK4): the hc==4 register-resident finish. The MEASURED close
//    (ncu, locked-clock GB10) of the last DeepSeek decode lever vs ds4. The generic
//    MhcPreFinishFloatKernel above is barrier + local-memory bound: ptxas hands it a
//    2048-byte LOCAL stack frame for the runtime-`hc` `cl[256]/m[256]` Sinkhorn arrays, and
//    ncu pins ~86% of its warp-cycles STALLED at the __syncthreads while thread 0 grinds that
//    Sinkhorn through scattered 4-byte local-DRAM transactions (Compute 0.1% / Mem 0.3% — it
//    is neither compute- nor bandwidth-bound, it is single-block serial-latency bound). This
//    kernel is byte-identical for hc=4 but (a) runs the split in REGISTERS (SinkhornInplace4,
//    0-byte stack frame) and (b) folds the RMSNorm Σ into the weighted-sum loop (ds4's
//    hc_split_weighted_sum_norm_fused_kernel:9752 does the same `sum += acc*acc` single-pass),
//    removing the duplicate layer_out re-read + one barrier. ncu: ~75→~26 µs/call; event-loop
//    95→57 µs/call — both BYTE-EXACT (comb+layer 0/16 ULP A/B). Same 1024-wide finish block +
//    sqrsum fold as the default LEAN path, so the norm reduction tree is unchanged.
__global__ void MhcPreFinishFloatKernel4(const float* __restrict__ residual,
                                         const float* __restrict__ scale,
                                         const float* __restrict__ base, int hc, int hidden,
                                         float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                                         float hc_post_mult, int iters,
                                         const float* __restrict__ norm_weight, int has_norm,
                                         float norm_eps, float* mixes, float* pre_out,
                                         float* post_out, float* comb_out, float* layer_out,
                                         int fold_sqrsum) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ float redf[];
  auto block_reduce = [&](float v) -> float {
    redf[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
    const float r = redf[0];
    __syncthreads();
    return r;
  };
  float sqrsum;
  if (fold_sqrsum) {
    sqrsum = mixes[hc3];
  } else {
    float ls = 0.0f;
    for (int i = tid; i < flat; i += nt) { const float r = residual[i]; ls += r * r; }
    sqrsum = block_reduce(ls);
  }
  const float rms = rsqrtf(sqrsum / static_cast<float>(flat) + rms_eps);
  for (int o = tid; o < hc3; o += nt) mixes[o] = mixes[o] * rms;
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      post_out[j] = Sig(mixes[4 + j] * scale[1] + base[4 + j]) * hc_post_mult;
    float cl[16], m[16];  // hc==4 ⇒ register-resident (0-byte stack frame)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int idx = j * 4 + k;
        cl[idx] = mixes[8 + idx] * scale[2] + base[8 + idx];
      }
    SinkhornInplace4(cl, m, iters, hc_sinkhorn_eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) comb_out[i] = m[i];
  }
  __syncthreads();
  // ds4-mirror norm fold: one global write of layer_out, no re-read. BYTE-EXACT — ss
  // accumulates acc*acc over the SAME per-thread h-sequence/order the separate pass used.
  float ss = 0.0f;
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
    ss += acc * acc;
  }
  if (has_norm) {
    const float ssr = block_reduce(ss);
    const float r = rsqrtf(ssr / static_cast<float>(hidden) + norm_eps);
    for (int h = tid; h < hidden; h += nt) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

// MhcPost (torch.py:94-106). One thread per (j,h).
__global__ void MhcPostKernel(const float* x, const float* residual, const float* post_mix,
                              const float* comb, int hc, int hidden, float* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hc * hidden) return;
  const int j = idx / hidden, h = idx % hidden;
  float mixed = 0.0f;
  for (int i = 0; i < hc; ++i) mixed += comb[i * hc + j] * residual[i * hidden + h];
  out[idx] = mixed + post_mix[j] * x[h];
}

// HcHeadCollapse (triton.py:108-140). Single thread; pre[hc] in local.
// glue-tune — PARALLEL hc_head collapse (was <<<1,1>>> single-thread = the profiled
// 3.75 ms/instance). One block, blockDim threads over the flat=hc*hidden width: the ss
// reduction + the hc pre-dots are block-tree reductions (double), the per-h collapse is
// parallel over hidden (per-h sequential over hc → order kept). CHARACTERIZED near-tie
// vs the single-thread float accumulation (reduction reorder) — the same class as the
// other MHC glue; BOTH launchers use this kernel, so head_ip == the #183 round-trip
// remain BIT-IDENTICAL to each other. `x[p]*r` stays float (host order) before the dot.
__global__ void HcHeadKernel(const float* __restrict__ x, const float* __restrict__ fn, float scale,
                             const float* __restrict__ base, int hc, int hidden, float rms_eps,
                             float hc_eps, float* __restrict__ out) {
  const int flat = hc * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ double red[];
  __shared__ float pre_sh[256];
  __shared__ float rms_sh;
  auto block_reduce = [&](double v) -> double {
    red[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const double r = red[0];
    __syncthreads();
    return r;
  };
  double ls = 0.0;
  for (int p = tid; p < flat; p += nt) { const double v = x[p]; ls += v * v; }
  const double ss = block_reduce(ls);
  if (tid == 0) rms_sh = 1.0f / sqrtf(static_cast<float>(ss / static_cast<double>(flat)) + rms_eps);
  __syncthreads();
  const float r = rms_sh;
  for (int m = 0; m < hc; ++m) {
    double la = 0.0;
    const int frow = m * flat;
    for (int p = tid; p < flat; p += nt) la += static_cast<double>(x[p] * r) * fn[frow + p];
    const double acc = block_reduce(la);
    if (tid == 0) pre_sh[m] = Sig(static_cast<float>(acc) * scale + base[m]) + hc_eps;
  }
  __syncthreads();
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
    for (int m = 0; m < hc; ++m) acc += pre_sh[m] * x[m * hidden + h];
    out[h] = acc;
  }
}

// ============================================================================
// (2) DSA family
// ============================================================================
__global__ void DsaWeightFoldKernel(const float* wp, float* out, int64_t n, float fold) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = wp[i] * fold;
}

// MQA logit (triton_fp8_mqa_logits.py:120-156). One thread per (t,s).
__global__ void DsaLogitsKernel(const float* q, const float* k, const float* folded,
                                const int64_t* ws, const int64_t* we, int T, int nk, int H,
                                int D, float* out) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(T) * nk) return;
  const int t = static_cast<int>(idx / nk), s = static_cast<int>(idx % nk);
  const int64_t s0 = ws[t] > 0 ? ws[t] : 0;
  const int64_t s1 = we[t] < nk ? we[t] : nk;
  if (s < s0 || s >= s1) {
    out[idx] = -INFINITY;
    return;
  }
  float acc = 0.0f;
  for (int h = 0; h < H; ++h) {
    float dot = 0.0f;
    const float* qp = &q[((static_cast<int64_t>(t) * H) + h) * D];
    const float* kp = &k[static_cast<int64_t>(s) * D];
    for (int d = 0; d < D; ++d) dot += qp[d] * kp[d];
    const float relu = dot > 0.0f ? dot : 0.0f;
    acc += folded[static_cast<int64_t>(t) * H + h] * relu;
  }
  out[idx] = acc;
}

// Causal top-k select (sparse_attn_indexer.py:488-497 + short-context all-select).
// One thread per token row; the same set + ascending emit as the host reference.
__global__ void DsaTopkKernel(const float* logits, const int64_t* ws, const int64_t* we,
                              int T, int nk, int topk, int64_t* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const int64_t s0 = ws[t] > 0 ? ws[t] : 0;
  const int64_t s1 = we[t] < nk ? we[t] : nk;
  const int64_t n = s1 > s0 ? s1 - s0 : 0;
  int64_t* dst = &out[static_cast<int64_t>(t) * topk];
  for (int j = 0; j < topk; ++j) dst[j] = -1;
  if (n <= topk) {
    int w = 0;
    for (int64_t s = s0; s < s1; ++s) dst[w++] = s;
    return;
  }
  // Pick the `topk` best under the SAME total order the host reference sorts by
  // (`DsaTopkSelect`: logit desc, then index asc — a total order because the
  // candidate indices are distinct). Two passes, NO per-thread scratch:
  //
  //   pass 1 walks the order downwards `topk` times to land on the topk-th best
  //          element, which is the selection THRESHOLD;
  //   pass 2 scans the window once in ascending index order and emits every
  //          element better-or-equal to that threshold.
  //
  // Pass 2 emits exactly `topk` entries already in ascending key order, so the
  // ascending sort the previous revision needed is gone with the buffers.
  //
  // This replaces a `bool chosen[512]` + `int64_t picked[64]` pair of literals
  // that could not represent the real `index_topk` (512 on V4-Flash, 1024 on
  // V4-Pro) and overflowed the thread stack on any window wider than `topk`
  // (#505). Cost is unchanged at O(topk*n) for pass 1, and strictly better
  // overall: the O(topk^2) emit sort is eliminated.
  const int64_t row = static_cast<int64_t>(t) * nk;
  // `better(va, a, vb, b)` == "(va, a) outranks (vb, b)".
  auto better = [](float va, int64_t a, float vb, int64_t b) -> bool {
    return va > vb || (va == vb && a < b);
  };
  float th_val = 0.0f;
  int64_t th_idx = -1;
  for (int64_t j = 0; j < topk; ++j) {
    float best_val = 0.0f;
    int64_t best = -1;
    for (int64_t s = s0; s < s1; ++s) {
      const float v = logits[row + s];
      // Skip anything at or above the previous step's element, so each step
      // descends exactly one rank.
      if (th_idx >= 0 && !better(th_val, th_idx, v, s)) continue;
      if (best < 0 || better(v, s, best_val, best)) {
        best_val = v;
        best = s;
      }
    }
    // n > topk holds here, so a strictly worse element always exists under a
    // total order. `best < 0` is therefore unreachable on ordered input; it can
    // only arise if the row carries NaN, which makes every comparison false. Stop
    // rather than reset the threshold, so pass 2 still emits a bounded prefix.
    if (best < 0) break;
    th_val = best_val;
    th_idx = best;
  }
  if (th_idx < 0) return;  // pathological row: leave the -1 padding in place
  // Exactly `topk` elements outrank-or-equal the threshold, so `w` lands on topk.
  // The `w < topk` bound is not load-bearing for ordered input — it is here so a
  // NaN row can never write past this thread's row into the next one, which is
  // the failure class #505 was about.
  int64_t w = 0;
  for (int64_t s = s0; s < s1 && w < topk; ++s) {
    const float v = logits[row + s];
    if (better(v, s, th_val, th_idx) || (v == th_val && s == th_idx)) dst[w++] = s;
  }
}

// Attention-sink softmax (flashinfer_sparse.py:777,:896). Single thread (one row).
__global__ void SoftmaxSinkKernel(const float* scores, int n, float sink, float* out) {
  float m = sink;
  for (int j = 0; j < n; ++j) m = fmaxf(m, scores[j]);
  if (m == -INFINITY) {
    for (int j = 0; j < n; ++j) out[j] = 0.0f;
    return;
  }
  float denom = expf(sink - m);
  for (int j = 0; j < n; ++j) {
    const float e = expf(scores[j] - m);
    out[j] = e;
    denom += e;
  }
  for (int j = 0; j < n; ++j) out[j] /= denom;
}

// Grouped output-LoRA (o_proj.py:58-73). One block per token; global z scratch.
__global__ void GroupedOLoraKernel(const float* o, const float* wo_a, const float* wo_b,
                                   int T, int nh, int hd, int ng, int olr, int H,
                                   int in_per_group, int z_dim, float* z_all, float* out) {
  const int t = blockIdx.x;
  if (t >= T) return;
  float* z = &z_all[static_cast<int64_t>(t) * z_dim];
  const float* o_t = &o[static_cast<int64_t>(t) * nh * hd];
  if (threadIdx.x == 0) {
    for (int g = 0; g < ng; ++g) {
      const float* o_g = o_t + g * in_per_group;
      const float* wa_g = &wo_a[static_cast<int64_t>(g) * olr * in_per_group];
      float* z_g = &z[g * olr];
      for (int d = 0; d < olr; ++d) {
        float acc = 0.0f;
        const float* wa_gd = wa_g + static_cast<int64_t>(d) * in_per_group;
        for (int r = 0; r < in_per_group; ++r) acc += wa_gd[r] * o_g[r];
        z_g[d] = acc;
      }
    }
    float* out_t = &out[static_cast<int64_t>(t) * H];
    for (int h = 0; h < H; ++h) {
      float acc = 0.0f;
      const float* wb_h = &wo_b[static_cast<int64_t>(h) * z_dim];
      for (int c = 0; c < z_dim; ++c) acc += wb_h[c] * z[c];
      out_t[h] = acc;
    }
  }
}

// ============================================================================
// (3) Compressor family
// ============================================================================
__global__ void SaveScoreApeKernel(const float* score, const float* ape,
                                    const int64_t* positions, int T, int width, int cr,
                                    float* out) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(T) * width) return;
  const int t = static_cast<int>(idx / width), d = static_cast<int>(idx % width);
  int64_t ape_row = positions[t] % cr;
  if (ape_row < 0) ape_row += cr;
  out[idx] = score[idx] + ape[ape_row * width + d];
}

// Compressor pool+norm (fused_compress_quant_cache.py:198-218). Single thread; one token.
__global__ void PoolNormKernel(const float* kv, const float* score, const uint8_t* valid,
                               const float* rms_w, float eps, int window, int hd,
                               float* out) {
  extern __shared__ float comp[];  // [hd]
  for (int d = 0; d < hd; ++d) {
    float m = -INFINITY;
    for (int i = 0; i < window; ++i) {
      const float s = valid[i] ? score[i * hd + d] : -INFINITY;
      m = fmaxf(m, s);
    }
    if (m == -INFINITY) {
      comp[d] = 0.0f;
      continue;
    }
    float denom = 0.0f, acc = 0.0f;
    for (int i = 0; i < window; ++i) {
      if (!valid[i]) continue;
      const float e = expf(score[i * hd + d] - m);
      denom += e;
      acc += kv[i * hd + d] * e;
    }
    comp[d] = acc / denom;
  }
  float var = 0.0f;
  for (int d = 0; d < hd; ++d) var += comp[d] * comp[d];
  var /= static_cast<float>(hd);
  const float rrms = 1.0f / sqrtf(var + eps);
  for (int d = 0; d < hd; ++d) out[d] = comp[d] * rrms * rms_w[d];
}

// fp8_ds_mla encode (fused_compress_quant_cache.py:238-297): per 64-wide NoPE
// block bf16-round -> absmax(>=1e-4) -> UE8M0 exponent -> e4m3; rope -> bf16.
__global__ void Fp8EncodeKernel(const float* head, int qblk, int nblk, uint8_t* nope_fp8,
                                uint8_t* scale_ue8m0) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk) return;
  const int base = b * qblk;
  float absmax = 0.0f;
  for (int j = 0; j < qblk; ++j) absmax = fmaxf(absmax, fabsf(Bf16Round(head[base + j])));
  absmax = fmaxf(absmax, 1e-4f);
  const float raw = absmax * (1.0f / 448.0f);
  const float exponent = ceilf(log2f(raw));
  const float inv_scale = exp2f(-exponent);
  for (int j = 0; j < qblk; ++j) {
    float x = Bf16Round(head[base + j]) * inv_scale;
    x = fminf(fmaxf(x, -448.0f), 448.0f);
    nope_fp8[base + j] = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
  }
  float enc = exponent + 127.0f;
  enc = fmaxf(0.0f, fminf(255.0f, enc));
  scale_ue8m0[b] = static_cast<uint8_t>(enc);
}

// The rope part is bf16 verbatim; a separate kernel matches vt::F32ToBF16
// (round-to-nearest-even) via __float2bfloat16.
__global__ void RopeToBf16Kernel(const float* head, int nope, int rope, uint16_t* rope_bf16) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= rope) return;
  const __nv_bfloat16 h = __float2bfloat16(head[nope + j]);
  uint16_t bits;
  memcpy(&bits, &h, sizeof(bits));
  rope_bf16[j] = bits;
}

// fp8_ds_mla decode (SGLang dequant_k_cache.py:122-136).
__global__ void Fp8DecodeKernel(const uint8_t* nope_fp8, const uint8_t* scale_ue8m0,
                                const uint16_t* rope_bf16, int nope, int rope, int qblk,
                                int nblk, float* out) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < nope) {
    const int b = d / qblk;
    const float scale_pow2 = exp2f(static_cast<float>(scale_ue8m0[b]) - 127.0f);
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(nope_fp8[d], __NV_E4M3);
    out[d] = __half2float(hr) * scale_pow2;
  }
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < rope) {
    __nv_bfloat16 h;
    const uint16_t bits = rope_bf16[j];
    memcpy(&h, &bits, sizeof(h));
    out[nope + j] = __bfloat162float(h);
  }
}

// ============================================================================
// (4) MoE family
// ============================================================================
__global__ void SqrtSoftplusKernel(const float* x, float* out, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = SqrtSoftplusDev(x[i]);
}

// sqrtsoftplus + noaux_tc bias router + hash bypass (fused_topk_bias_router.py:75-118).
// One thread per token. E<=256, topk<=32 for the structural gate.
__global__ void RouteKernel(const float* gating, int T, int E, int topk, const float* bias,
                            int has_bias, int is_hash, const int64_t* in_tokens,
                            const int32_t* hashtab, int64_t vocab, int renorm, float scale,
                            int32_t* ids_out, float* w_out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  float scores[256];
  const float* g = &gating[static_cast<int64_t>(t) * E];
  for (int e = 0; e < E; ++e) scores[e] = SqrtSoftplusDev(g[e]);
  int32_t* ids = &ids_out[static_cast<int64_t>(t) * topk];
  float* w = &w_out[static_cast<int64_t>(t) * topk];
  if (is_hash) {
    int64_t tok = in_tokens[t] % vocab;
    if (tok < 0) tok += vocab;
    const int32_t* row = &hashtab[tok * topk];
    for (int j = 0; j < topk; ++j) {
      ids[j] = row[j];
      w[j] = scores[row[j]];
    }
  } else {
    float sfc[256];
    bool used[256];
    for (int e = 0; e < E; ++e) {
      sfc[e] = has_bias ? scores[e] + bias[e] : scores[e];
      used[e] = false;
    }
    for (int j = 0; j < topk; ++j) {
      int best = -1;
      float bestv = -INFINITY;
      for (int e = 0; e < E; ++e) {
        if (used[e]) continue;
        if (best < 0 || sfc[e] > bestv) {  // strict > -> smaller index wins a tie
          bestv = sfc[e];
          best = e;
        }
      }
      used[best] = true;
      ids[j] = best;
      w[j] = scores[best];  // GATHER from the UNBIASED scores
    }
  }
  if (renorm) {
    float sum = 0.0f;
    for (int j = 0; j < topk; ++j) sum += w[j];
    const float denom = fmaxf(sum, 1e-20f);
    for (int j = 0; j < topk; ++j) w[j] /= denom;
  }
  for (int j = 0; j < topk; ++j) w[j] *= scale;
}

// A/B flag for the warp-parallel router top-k (ds4-gap Lever 3 / Brick 10). Default ON
// (parity enabler ships as default); VT_V4_ROUTE_WARP_TOPK=0 forces the legacy
// single-thread RouteKernel for baseline measurement + the bit-exact A/B gate. Read per
// call so in-process CUDA tests can flip it. BIT-IDENTICAL either way (selection is an
// argmax under a strict total order, weights + renorm run the same float ops in j-order).
inline bool RouteWarpTopkOn(const char* v) { return !(v && v[0] == '0' && v[1] == '\0'); }

// strict-> value compare; equal gating resolved by LOWER expert index (the exact tie-break
// the single-thread RouteKernel encodes via `sfc[e] > bestv` with ascending e).
__device__ __forceinline__ bool RouteScoreBetter(float av, unsigned ai, float bv, unsigned bi) {
  return av > bv || (av == bv && ai < bi);
}

// Warp-parallel sqrtsoftplus + noaux_tc bias router — structure-port of ds4's
// router_select_warp_topk_kernel (ds4_cuda.cu:10113). ONE WARP (32 lanes) per token; each
// lane owns experts e = lane + j*32 (j<8 ⇒ E<=256). BIT-IDENTICAL to RouteKernel: the
// selection is an argmax under RouteScoreBetter's strict total order so the tree-reduction
// order is irrelevant to the winner; the unbiased weight is SqrtSoftplusDev(g[e]) (same
// input, same fn ⇒ same bits) and the renorm (fmaxf(sum,1e-20); /denom; *scale) runs in the
// same j-order on lane 0. Dynamic shared sprob[rows*E] mirrors ds4's sprob[4][256] for the
// hash gather. Requires E<=256 && topk<=32 (else the launcher falls back to RouteKernel).
__global__ void RouteWarpKernel(const float* gating, int T, int E, int topk, const float* bias,
                                int has_bias, int is_hash, const int64_t* in_tokens,
                                const int32_t* hashtab, int64_t vocab, int renorm, float scale,
                                int32_t* ids_out, float* w_out) {
  extern __shared__ float sprob[];  // [blockDim.y * E]
  const unsigned lane = threadIdx.x;  // 0..31
  const unsigned row = threadIdx.y;   // token within block (one warp per row)
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.y) +
                static_cast<int>(row);
  if (t >= T) return;
  const float* g = &gating[static_cast<int64_t>(t) * E];
  int32_t* ids = &ids_out[static_cast<int64_t>(t) * topk];
  float* w = &w_out[static_cast<int64_t>(t) * topk];
  float* srow = &sprob[static_cast<int64_t>(row) * E];

  // Per-lane experts (<=8 for E<=256): local_score is the BIASED selection key; local_prob
  // is the UNBIASED gathered weight. Invalid lanes (e>=E) hold -INF so they never win.
  const int per = (E + 31) / 32;  // <= 8
  float local_prob[8];
  float local_score[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local_prob[j] = 0.0f;
    local_score[j] = -INFINITY;
    if (j < per) {
      const unsigned e = lane + static_cast<unsigned>(j) * 32u;
      if (e < static_cast<unsigned>(E)) {
        const float p = SqrtSoftplusDev(g[e]);
        local_prob[j] = p;
        local_score[j] = has_bias ? p + bias[e] : p;
        srow[e] = p;
      }
    }
  }
  __syncwarp();

  if (is_hash) {
    if (lane == 0) {
      int64_t tok = in_tokens[t] % vocab;
      if (tok < 0) tok += vocab;
      const int32_t* hrow = &hashtab[tok * topk];
      for (int j = 0; j < topk; ++j) {
        const int32_t e = hrow[j];
        ids[j] = e;
        w[j] = (e >= 0 && e < E) ? srow[e] : 0.0f;  // GATHER from UNBIASED scores
      }
    }
  } else {
    for (int k = 0; k < topk; ++k) {
      float best_score = -INFINITY, best_prob = 0.0f;
      unsigned best_idx = 0xFFFFFFFFu;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const unsigned e = lane + static_cast<unsigned>(j) * 32u;
        if (RouteScoreBetter(local_score[j], e, best_score, best_idx)) {
          best_score = local_score[j];
          best_prob = local_prob[j];
          best_idx = e;
        }
      }
#pragma unroll
      for (unsigned mask = 16u; mask > 0u; mask >>= 1u) {
        const float os = __shfl_xor_sync(0xffffffffu, best_score, mask);
        const float op = __shfl_xor_sync(0xffffffffu, best_prob, mask);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, best_idx, mask);
        if (RouteScoreBetter(os, oi, best_score, best_idx)) {
          best_score = os;
          best_prob = op;
          best_idx = oi;
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const unsigned e = lane + static_cast<unsigned>(j) * 32u;
        if (e == best_idx) local_score[j] = -INFINITY;  // remove the winner for next k
      }
      if (lane == 0) {
        ids[k] = static_cast<int32_t>(best_idx);
        w[k] = best_prob;  // UNBIASED score of the selected expert
      }
    }
  }
  if (lane == 0) {
    if (renorm) {
      float sum = 0.0f;
      for (int j = 0; j < topk; ++j) sum += w[j];
      const float denom = fmaxf(sum, 1e-20f);
      for (int j = 0; j < topk; ++j) w[j] /= denom;
    }
    for (int j = 0; j < topk; ++j) w[j] *= scale;
  }
}

unsigned Grid(int64_t n, int block);  // fwd-decl (defined below); used by RouteDispatch

// Shared launch helper: warp-topk (default) or legacy single-thread, both writing
// ids_out[T*topk] (i32) + w_out[T*topk] (f32). Read the A/B env once per launch so eager
// AND captured V4Graph::Step select the same kernel (no Brick-7 split-path trap). Falls
// back to the single-thread kernel outside the structural gate (E>256 or topk>32).
inline void RouteDispatch(cudaStream_t s, const float* gating, int T, int E, int topk,
                          const float* bias, int has_bias, int is_hash,
                          const int64_t* in_tokens, const int32_t* hashtab, int64_t vocab,
                          int renorm, float scale, int32_t* ids_out, float* w_out) {
  if (T <= 0) return;
  const bool warp = RouteWarpTopkOn(std::getenv("VT_V4_ROUTE_WARP_TOPK")) && E <= 256 && topk <= 32;
  if (warp) {
    const unsigned rows = 4;  // 4 warps/block (ds4's sprob[4][256] layout)
    const dim3 block(32, rows);
    const unsigned grid = static_cast<unsigned>((static_cast<int64_t>(T) + rows - 1) / rows);
    const unsigned shmem = rows * static_cast<unsigned>(E) * sizeof(float);
    RouteWarpKernel<<<grid, block, shmem, s>>>(gating, T, E, topk, bias, has_bias, is_hash,
                                               in_tokens, hashtab, vocab, renorm, scale, ids_out,
                                               w_out);
  } else {
    const int block = 64;
    RouteKernel<<<Grid(T, block), block, 0, s>>>(gating, T, E, topk, bias, has_bias, is_hash,
                                                 in_tokens, hashtab, vocab, renorm, scale, ids_out,
                                                 w_out);
  }
}

// Clamped SwiGLU (activation.py:197-201). One thread per output channel.
__global__ void ClampedSwiGLUKernel(const float* gate_up, int d, float limit, float alpha,
                                    float beta, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  const float gate = fminf(gate_up[i], limit);
  const float up = fminf(fmaxf(gate_up[d + i], -limit), limit);
  out[i] = gate * Sig(alpha * gate) * (up + beta);
}

unsigned Grid(int64_t n, int block) {
  return static_cast<unsigned>((n + block - 1) / block);
}

// ── launchers (host-vector wrappers; upload -> kernel -> download) ────────────
std::vector<float> MhcSinkhornLaunch(Queue& q, const std::vector<float>& logits, int64_t hc,
                                     int64_t iters, float eps) {
  cudaStream_t s = AsStream(q);
  Dev dl = Upload(logits, s);
  std::vector<float> out(static_cast<size_t>(hc * hc));
  Dev dout(out.size() * sizeof(float));
  SinkhornKernel<<<1, 1, 0, s>>>(static_cast<const float*>(dl.p), static_cast<float*>(dout.p),
                                 static_cast<int>(hc), static_cast<int>(iters), eps);
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync sinkhorn");
  return out;
}

MhcPreResult MhcPreLaunch(Queue& q, const std::vector<float>& residual,
                          const std::vector<float>& fn, const std::vector<float>& scale,
                          const std::vector<float>& base, int64_t hc, int64_t hidden,
                          float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                          float hc_post_mult, int64_t iters,
                          const std::vector<float>& norm_weight, float norm_eps) {
  cudaStream_t s = AsStream(q);
  const int hc3 = static_cast<int>((2 + hc) * hc);
  Dev dr = Upload(residual, s), df = Upload(fn, s), ds = Upload(scale, s), db = Upload(base, s);
  const bool has_norm = !norm_weight.empty();
  std::vector<float> nw = has_norm ? norm_weight : std::vector<float>(1, 0.0f);
  Dev dnw = Upload(nw, s);
  Dev dmix(static_cast<size_t>(hc3) * sizeof(float));
  MhcPreResult out;
  out.pre_mix.resize(static_cast<size_t>(hc));
  out.post_mix.resize(static_cast<size_t>(hc));
  out.comb_mix.resize(static_cast<size_t>(hc * hc));
  out.layer_input.resize(static_cast<size_t>(hidden));
  Dev dpre(out.pre_mix.size() * sizeof(float)), dpost(out.post_mix.size() * sizeof(float));
  Dev dcomb(out.comb_mix.size() * sizeof(float)), dlin(out.layer_input.size() * sizeof(float));
  MhcPreKernel<<<1, 1, 0, s>>>(
      static_cast<const float*>(dr.p), static_cast<const float*>(df.p),
      static_cast<const float*>(ds.p), static_cast<const float*>(db.p), static_cast<int>(hc),
      static_cast<int>(hidden), rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult,
      static_cast<int>(iters), static_cast<const float*>(dnw.p), has_norm ? 1 : 0, norm_eps,
      static_cast<float*>(dmix.p), static_cast<float*>(dpre.p), static_cast<float*>(dpost.p),
      static_cast<float*>(dcomb.p), static_cast<float*>(dlin.p));
  Download(out.pre_mix, dpre.p, s);
  Download(out.post_mix, dpost.p, s);
  Download(out.comb_mix, dcomb.p, s);
  Download(out.layer_input, dlin.p, s);
  Check(cudaStreamSynchronize(s), "sync mhc_pre");
  return out;
}

std::vector<float> MhcPostLaunch(Queue& q, const std::vector<float>& x,
                                 const std::vector<float>& residual,
                                 const std::vector<float>& post_mix,
                                 const std::vector<float>& comb, int64_t hc, int64_t hidden) {
  cudaStream_t s = AsStream(q);
  Dev dx = Upload(x, s), dr = Upload(residual, s), dp = Upload(post_mix, s), dc = Upload(comb, s);
  std::vector<float> out(static_cast<size_t>(hc * hidden));
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  MhcPostKernel<<<Grid(hc * hidden, block), block, 0, s>>>(
      static_cast<const float*>(dx.p), static_cast<const float*>(dr.p),
      static_cast<const float*>(dp.p), static_cast<const float*>(dc.p), static_cast<int>(hc),
      static_cast<int>(hidden), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync mhc_post");
  return out;
}

std::vector<float> HcHeadLaunch(Queue& q, const std::vector<float>& x,
                                const std::vector<float>& fn, float scale,
                                const std::vector<float>& base, int64_t hc, int64_t hidden,
                                float rms_eps, float hc_eps) {
  cudaStream_t s = AsStream(q);
  Dev dx = Upload(x, s), df = Upload(fn, s), db = Upload(base, s);
  std::vector<float> out(static_cast<size_t>(hidden));
  Dev dout(out.size() * sizeof(float));
  HcHeadKernel<<<1, 256, 256 * sizeof(double), s>>>(
      static_cast<const float*>(dx.p), static_cast<const float*>(df.p), scale,
      static_cast<const float*>(db.p), static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
      hc_eps, static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync hc_head");
  return out;
}

std::vector<float> DsaWeightFoldLaunch(Queue& q, const std::vector<float>& wp, int64_t T,
                                       int64_t inh, int64_t ihd) {
  cudaStream_t s = AsStream(q);
  const float fold = (1.0f / sqrtf(static_cast<float>(ihd))) *
                     (1.0f / sqrtf(static_cast<float>(inh)));
  Dev dw = Upload(wp, s);
  std::vector<float> out(wp.size());
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  DsaWeightFoldKernel<<<Grid(static_cast<int64_t>(wp.size()), block), block, 0, s>>>(
      static_cast<const float*>(dw.p), static_cast<float*>(dout.p),
      static_cast<int64_t>(wp.size()), fold);
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync fold");
  return out;
}

std::vector<float> DsaLogitsLaunch(Queue& q, const std::vector<float>& qv,
                                   const std::vector<float>& k, const std::vector<float>& folded,
                                   const std::vector<int64_t>& ws, const std::vector<int64_t>& we,
                                   int64_t T, int64_t nk, int64_t inh, int64_t ihd) {
  cudaStream_t s = AsStream(q);
  Dev dq = Upload(qv, s), dk = Upload(k, s), dfo = Upload(folded, s);
  Dev dws = Upload(ws, s), dwe = Upload(we, s);
  std::vector<float> out(static_cast<size_t>(T * nk));
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  DsaLogitsKernel<<<Grid(T * nk, block), block, 0, s>>>(
      static_cast<const float*>(dq.p), static_cast<const float*>(dk.p),
      static_cast<const float*>(dfo.p), static_cast<const int64_t*>(dws.p),
      static_cast<const int64_t*>(dwe.p), static_cast<int>(T), static_cast<int>(nk),
      static_cast<int>(inh), static_cast<int>(ihd), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync logits");
  return out;
}

std::vector<int64_t> DsaTopkLaunch(Queue& q, const std::vector<float>& logits,
                                   const std::vector<int64_t>& ws,
                                   const std::vector<int64_t>& we, int64_t T, int64_t nk,
                                   int64_t topk) {
  cudaStream_t s = AsStream(q);
  Dev dl = Upload(logits, s), dws = Upload(ws, s), dwe = Upload(we, s);
  std::vector<int64_t> out(static_cast<size_t>(T * topk), -1);
  Dev dout(out.size() * sizeof(int64_t));
  const int block = 64;
  DsaTopkKernel<<<Grid(T, block), block, 0, s>>>(
      static_cast<const float*>(dl.p), static_cast<const int64_t*>(dws.p),
      static_cast<const int64_t*>(dwe.p), static_cast<int>(T), static_cast<int>(nk),
      static_cast<int>(topk), static_cast<int64_t*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync topk");
  return out;
}

std::vector<float> SoftmaxSinkLaunch(Queue& q, const std::vector<float>& scores, float sink) {
  cudaStream_t s = AsStream(q);
  Dev dsc = Upload(scores, s);
  std::vector<float> out(scores.size());
  Dev dout(out.size() * sizeof(float));
  SoftmaxSinkKernel<<<1, 1, 0, s>>>(static_cast<const float*>(dsc.p),
                                    static_cast<int>(scores.size()), sink,
                                    static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync softmax_sink");
  return out;
}

std::vector<float> GroupedOLoraLaunch(Queue& q, const std::vector<float>& o,
                                      const std::vector<float>& wo_a,
                                      const std::vector<float>& wo_b, int64_t T, int64_t nh,
                                      int64_t hd, int64_t ng, int64_t olr, int64_t H) {
  cudaStream_t s = AsStream(q);
  const int in_per_group = static_cast<int>(nh * hd / ng);
  const int z_dim = static_cast<int>(ng * olr);
  Dev doo = Upload(o, s), dwa = Upload(wo_a, s), dwb = Upload(wo_b, s);
  std::vector<float> out(static_cast<size_t>(T * H));
  Dev dout(out.size() * sizeof(float));
  Dev dz(static_cast<size_t>(T) * z_dim * sizeof(float));
  GroupedOLoraKernel<<<static_cast<unsigned>(T), 1, 0, s>>>(
      static_cast<const float*>(doo.p), static_cast<const float*>(dwa.p),
      static_cast<const float*>(dwb.p), static_cast<int>(T), static_cast<int>(nh),
      static_cast<int>(hd), static_cast<int>(ng), static_cast<int>(olr), static_cast<int>(H),
      in_per_group, z_dim, static_cast<float*>(dz.p), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync olora");
  return out;
}

std::vector<float> SaveScoreApeLaunch(Queue& q, const std::vector<float>& score,
                                      const std::vector<float>& ape,
                                      const std::vector<int64_t>& positions, int64_t T,
                                      int64_t width, int64_t cr) {
  cudaStream_t s = AsStream(q);
  Dev dsc = Upload(score, s), dap = Upload(ape, s), dp = Upload(positions, s);
  std::vector<float> out(score.size());
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  SaveScoreApeKernel<<<Grid(T * width, block), block, 0, s>>>(
      static_cast<const float*>(dsc.p), static_cast<const float*>(dap.p),
      static_cast<const int64_t*>(dp.p), static_cast<int>(T), static_cast<int>(width),
      static_cast<int>(cr), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync ape");
  return out;
}

std::vector<float> PoolNormLaunch(Queue& q, const std::vector<float>& kv,
                                  const std::vector<float>& score,
                                  const std::vector<uint8_t>& valid,
                                  const std::vector<float>& rms_w, float eps, int64_t window,
                                  int64_t hd) {
  cudaStream_t s = AsStream(q);
  Dev dkv = Upload(kv, s), dsc = Upload(score, s), dv = Upload(valid, s), dr = Upload(rms_w, s);
  std::vector<float> out(static_cast<size_t>(hd));
  Dev dout(out.size() * sizeof(float));
  PoolNormKernel<<<1, 1, static_cast<unsigned>(hd) * sizeof(float), s>>>(
      static_cast<const float*>(dkv.p), static_cast<const float*>(dsc.p),
      static_cast<const uint8_t*>(dv.p), static_cast<const float*>(dr.p), eps,
      static_cast<int>(window), static_cast<int>(hd), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync poolnorm");
  return out;
}

Fp8DsMlaToken Fp8EncodeLaunch(Queue& q, const std::vector<float>& head,
                              const Fp8DsMlaLayout& L) {
  cudaStream_t s = AsStream(q);
  Dev dh = Upload(head, s);
  Fp8DsMlaToken t;
  t.nope_fp8.assign(static_cast<size_t>(L.nope_head_dim), 0);
  t.scale_ue8m0.assign(static_cast<size_t>(L.n_nope_blocks), 0);
  t.rope_bf16.assign(static_cast<size_t>(L.rope_head_dim), 0);
  Dev dn(t.nope_fp8.size()), dsc(t.scale_ue8m0.size());
  Dev drp(t.rope_bf16.size() * sizeof(uint16_t));
  const int block = 64;
  Fp8EncodeKernel<<<Grid(L.n_nope_blocks, block), block, 0, s>>>(
      static_cast<const float*>(dh.p), static_cast<int>(L.quant_block),
      static_cast<int>(L.n_nope_blocks), static_cast<uint8_t*>(dn.p),
      static_cast<uint8_t*>(dsc.p));
  RopeToBf16Kernel<<<Grid(L.rope_head_dim, block), block, 0, s>>>(
      static_cast<const float*>(dh.p), static_cast<int>(L.nope_head_dim),
      static_cast<int>(L.rope_head_dim), static_cast<uint16_t*>(drp.p));
  Download(t.nope_fp8, dn.p, s);
  Download(t.scale_ue8m0, dsc.p, s);
  Download(t.rope_bf16, drp.p, s);
  Check(cudaStreamSynchronize(s), "sync fp8 encode");
  return t;
}

std::vector<float> Fp8DecodeLaunch(Queue& q, const Fp8DsMlaToken& t, const Fp8DsMlaLayout& L) {
  cudaStream_t s = AsStream(q);
  Dev dn = Upload(t.nope_fp8, s), dsc = Upload(t.scale_ue8m0, s), drp = Upload(t.rope_bf16, s);
  std::vector<float> out(static_cast<size_t>(L.nope_head_dim + L.rope_head_dim), 0.0f);
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  Fp8DecodeKernel<<<Grid(L.nope_head_dim, block), block, 0, s>>>(
      static_cast<const uint8_t*>(dn.p), static_cast<const uint8_t*>(dsc.p),
      static_cast<const uint16_t*>(drp.p), static_cast<int>(L.nope_head_dim),
      static_cast<int>(L.rope_head_dim), static_cast<int>(L.quant_block),
      static_cast<int>(L.n_nope_blocks), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync fp8 decode");
  return out;
}

std::vector<float> SqrtSoftplusLaunch(Queue& q, const std::vector<float>& x) {
  cudaStream_t s = AsStream(q);
  Dev dx = Upload(x, s);
  std::vector<float> out(x.size());
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  SqrtSoftplusKernel<<<Grid(static_cast<int64_t>(x.size()), block), block, 0, s>>>(
      static_cast<const float*>(dx.p), static_cast<float*>(dout.p),
      static_cast<int64_t>(x.size()));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync sqrtsoftplus");
  return out;
}

MoeRouteResult RouteLaunch(Queue& q, const std::vector<float>& gating, int64_t T, int64_t E,
                           int64_t topk, const std::vector<float>& bias, bool renorm,
                           float scale, const std::vector<int64_t>& in_tokens,
                           const std::vector<int32_t>& hashtab, int64_t vocab) {
  cudaStream_t s = AsStream(q);
  const bool has_bias = !bias.empty();
  const bool is_hash = !hashtab.empty() && !in_tokens.empty();
  Dev dg = Upload(gating, s);
  std::vector<float> bpad = has_bias ? bias : std::vector<float>(1, 0.0f);
  std::vector<int64_t> tpad = in_tokens.empty() ? std::vector<int64_t>(1, 0) : in_tokens;
  std::vector<int32_t> hpad = hashtab.empty() ? std::vector<int32_t>(1, 0) : hashtab;
  Dev dbias = Upload(bpad, s), dtok = Upload(tpad, s), dhash = Upload(hpad, s);
  MoeRouteResult out;
  out.topk_ids.assign(static_cast<size_t>(T * topk), 0);
  out.topk_weights.assign(static_cast<size_t>(T * topk), 0.0f);
  Dev did(out.topk_ids.size() * sizeof(int32_t)), dw(out.topk_weights.size() * sizeof(float));
  RouteDispatch(s, static_cast<const float*>(dg.p), static_cast<int>(T), static_cast<int>(E),
                static_cast<int>(topk), static_cast<const float*>(dbias.p), has_bias ? 1 : 0,
                is_hash ? 1 : 0, static_cast<const int64_t*>(dtok.p),
                static_cast<const int32_t*>(dhash.p), vocab, renorm ? 1 : 0, scale,
                static_cast<int32_t*>(did.p), static_cast<float*>(dw.p));
  Download(out.topk_ids, did.p, s);
  Download(out.topk_weights, dw.p, s);
  Check(cudaStreamSynchronize(s), "sync route");
  return out;
}

std::vector<float> ClampedSwiGLULaunch(Queue& q, const std::vector<float>& gate_up, int64_t d,
                                       float limit, float alpha, float beta) {
  cudaStream_t s = AsStream(q);
  Dev dgu = Upload(gate_up, s);
  std::vector<float> out(static_cast<size_t>(d));
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  ClampedSwiGLUKernel<<<Grid(d, block), block, 0, s>>>(
      static_cast<const float*>(dgu.p), static_cast<int>(d), limit, alpha, beta,
      static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync swiglu");
  return out;
}

// Brick B — IN-PLACE clamped-SwiGLU (unified memory, no Upload/Download/Sync). Same
// ClampedSwiGLUKernel as ClampedSwiGLULaunch, run directly on the caller's unified
// gate_up[2*d]/out[d] pointers. Bit-identical (elementwise). Caller drains.
void ClampedSwiGLUInPlaceLaunch(Queue& q, float* out, const float* gate_up, int64_t d,
                                float limit, float alpha, float beta) {
  if (d == 0) return;
  cudaStream_t s = AsStream(q);
  const int block = 128;
  ClampedSwiGLUKernel<<<Grid(d, block), block, 0, s>>>(gate_up, static_cast<int>(d), limit,
                                                       alpha, beta, out);
  Check(cudaGetLastError(), "clamped_swiglu_ip launch");
}

// Brick B — IN-PLACE MHC glue (unified memory, no Upload/Download/Sync). Same
// MhcPostKernel/HcHeadKernel/MhcPreKernel as the #183 launchers, run directly on the
// caller's unified pointers. Caller drains. MHC pre/head are <<<1,1>>> (hc=4 tiny);
// post is parallel. Near-tie vs host (device expf/rsqrt vs host in the RMSNorm/gates).
void MhcPostInPlaceLaunch(Queue& q, float* out, const float* x, const float* residual,
                          const float* post_mix, const float* comb, int64_t hc,
                          int64_t hidden) {
  if (hc == 0 || hidden == 0) return;
  cudaStream_t s = AsStream(q);
  const int block = 128;
  MhcPostKernel<<<Grid(hc * hidden, block), block, 0, s>>>(
      x, residual, post_mix, comb, static_cast<int>(hc), static_cast<int>(hidden), out);
  Check(cudaGetLastError(), "mhc_post_ip launch");
}

void HcHeadInPlaceLaunch(Queue& q, float* out, const float* x, const float* fn, float scale,
                         const float* base, int64_t hc, int64_t hidden, float rms_eps,
                         float hc_eps) {
  if (hidden == 0) return;
  cudaStream_t s = AsStream(q);
  HcHeadKernel<<<1, 256, 256 * sizeof(double), s>>>(x, fn, scale, base, static_cast<int>(hc),
                                                    static_cast<int>(hidden), rms_eps, hc_eps, out);
  Check(cudaGetLastError(), "hc_head_ip launch");
}

// ds4-fold gate: route MHC-pre through the FLOAT (ds4-mirroring) kernels instead of the
// bit-faithful FP64 path. DEFAULT ON (parity-enablers-ship-as-defaults) — a MEASURED
// decode win on GB10 where FP64 is throttled; VT_V4_MHC_FUSED=0 opts back to the double
// path for a same-binary A/B. Read once (process-wide), so the captured decode graph bakes
// a consistent path.
static bool MhcFusedEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_MHC_FUSED");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// Lever 2 — MHC-pre FINISH occupancy. MhcPreFinishFloatKernel runs `<<<1, block>>>`:
// ONE block (= ONE SM) executed ~86× SEQUENTIALLY per decode step (2 sub-blocks × 43
// layers; the layer chain is data-dependent so they cannot overlap). It is
// memory-latency-bound — it streams the residual (flat=hc·H) for the sqrsum reduction
// and again (strided) for the pre-weighted sum — so on the >100-SM GB10 the single
// block's few warps cannot hide HBM latency (it was 9.5% of decode GPU time at block=256
// = 8 warps). Widening the block to 1024 threads (32 warps) keeps more loads in flight
// → closer to bandwidth-bound. Same float algebra; a wider block only reorders the tree
// reduction → CHARACTERIZED near-tie (distributional-gated, token-checked A/B). Read once
// process-wide so the captured decode graph bakes a consistent block. VT_V4_MHC_LEAN
// selects the finish block: unset/1 → 1024 (default ON per parity-enablers); 0 → 256
// (baseline A/B); an explicit power-of-two ≤1024 (e.g. 512) → that size. The mix-dots
// kernel keeps block=256 (already grid=hc3 = well-occupied).
static unsigned MhcFinishBlock() {
  static const unsigned b = [] {
    const char* e = std::getenv("VT_V4_MHC_LEAN");
    if (e == nullptr) return 1024u;  // default lean ON
    const int v = std::atoi(e);
    if (v == 0) return 256u;    // baseline (lean off) — same block as the mix dots
    if (v == 1) return 1024u;   // lean on
    if (v == 512 || v == 1024) return static_cast<unsigned>(v);
    return 1024u;               // reject non-power-of-two; the reduction tree needs pow2
  }();
  return b;
}
// Lever 2 ON iff the finish block was widened past the 256 baseline; it also gates the
// sqrsum fold (MhcPreDots block 0 pre-computes Σresidual² → finish skips its second pass).
static inline bool MhcLeanOn() { return MhcFinishBlock() != 256u; }

// VT_V4_MHC_SINK4 — the hc==4 register-resident finish (MhcPreFinishFloatKernel4). DEFAULT-ON
// (parity-enablers): it is BYTE-EXACT for hc=4 and a MEASURED ~75→~26 µs/call (ncu) close of
// the finish kernel's local-memory+barrier stall vs ds4. =0 opts back to the generic
// runtime-`hc` finish (same-binary A/B). Only applies when hc==4 (the DeepSeek-V4 structural
// hc_mult); any other hc always takes the generic path.
static bool MhcSink4On() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_MHC_SINK4");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// MhcPre writes pre/post/comb mixes + layer_input; it needs an hc3=[(2+hc)*hc] mix
// scratch. `mix_scratch` is a caller-provided unified buffer (>= hc3 floats).
void MhcPreInPlaceLaunch(Queue& q, float* pre_mix, float* post_mix, float* comb_mix,
                         float* layer_input, float* mix_scratch, const float* residual,
                         const float* fn, const float* scale, const float* base, int64_t hc,
                         int64_t hidden, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                         float hc_post_mult, int64_t iters, const float* norm_weight,
                         bool has_norm, float norm_eps) {
  if (hidden == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 256;  // 256 threads/block ⇒ the mix dots stay BIT-IDENTICAL
  const int hc3 = static_cast<int>((2 + hc) * hc);
  const int flat = static_cast<int>(hc * hidden);
  if (MhcFusedEnabled()) {
    // ds4-fold: the identical algebra in FLOAT (GB10 FP64 is ~1/32-1/64 of FP32).
    const unsigned shmemf = block * sizeof(float);
    const int fold = MhcLeanOn() ? 1 : 0;  // Lever 2: fold Σresidual² into the dots kernel
    MhcPreDotsFloatKernel<<<static_cast<unsigned>(hc3), block, shmemf, s>>>(residual, fn, flat,
                                                                            mix_scratch, fold, hc3);
    // Lever 2: widen the single-block finish for HBM-latency hiding on its lone SM + skip its
    // duplicate residual pass when the sqrsum was folded into the dots.
    const unsigned fblock = MhcFinishBlock();
    if (MhcSink4On() && hc == 4) {
      // Register-resident hc=4 finish + ds4 norm-fold (byte-exact); kills the 2048-byte local
      // stack frame + the layer_out re-read that made the generic finish barrier-bound.
      MhcPreFinishFloatKernel4<<<1, fblock, fblock * sizeof(float), s>>>(
          residual, scale, base, static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
          hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, static_cast<int>(iters), norm_weight,
          has_norm ? 1 : 0, norm_eps, mix_scratch, pre_mix, post_mix, comb_mix, layer_input, fold);
    } else {
      MhcPreFinishFloatKernel<<<1, fblock, fblock * sizeof(float), s>>>(
          residual, scale, base, static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
          hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, static_cast<int>(iters), norm_weight,
          has_norm ? 1 : 0, norm_eps, mix_scratch, pre_mix, post_mix, comb_mix, layer_input, fold);
    }
    Check(cudaGetLastError(), "mhc_pre_ip launch (fused)");
    return;
  }
  const unsigned shmem = block * sizeof(double);
  // Kernel A: the hc3 mix dot-products, ONE BLOCK EACH → concurrent across SMs.
  MhcPreDotsKernel<<<static_cast<unsigned>(hc3), block, shmem, s>>>(residual, fn, flat, mix_scratch);
  // Kernel B: sqrsum/rms + gates + Sinkhorn + layer_out + final norm (one block).
  MhcPreFinishKernel<<<1, block, shmem, s>>>(
      residual, scale, base, static_cast<int>(hc), static_cast<int>(hidden), rms_eps, hc_pre_eps,
      hc_sinkhorn_eps, hc_post_mult, static_cast<int>(iters), norm_weight, has_norm ? 1 : 0,
      norm_eps, mix_scratch, pre_mix, post_mix, comb_mix, layer_input);
  Check(cudaGetLastError(), "mhc_pre_ip launch");
}

// IN-PLACE router: same RouteKernel; writes topk_ids[T*topk] (i32) + weights[T*topk].
void RouteInPlaceLaunch(Queue& q, int32_t* topk_ids, float* topk_weights, const float* gating,
                        int64_t T, int64_t E, int64_t topk, const float* bias, bool has_bias,
                        const int64_t* in_tokens, bool is_hash, const int32_t* hashtab,
                        int64_t vocab, bool renorm, float scale) {
  if (T == 0) return;
  cudaStream_t s = AsStream(q);
  RouteDispatch(s, gating, static_cast<int>(T), static_cast<int>(E), static_cast<int>(topk), bias,
                has_bias ? 1 : 0, is_hash ? 1 : 0, in_tokens, hashtab, vocab, renorm ? 1 : 0,
                scale, topk_ids, topk_weights);
  Check(cudaGetLastError(), "route_ip launch");
}

// ── Brick C folded-in glue kernels (device RMSNorm / RoPE / MoE combine) ──────
// Weighted RMSNorm over [n]. One block, parallel block-tree reduction (double
// accumulate) → CHARACTERIZED near-tie vs host double-sequential (the reduction
// reorders); the scale+weight multiply is per-element (order-independent). has_w=0
// → no weight (the per-head q-RMS; DeepseekV4QHeadRmsNormInplace).
__global__ void RmsNormKernel(float* __restrict__ out, const float* __restrict__ x,
                              const float* __restrict__ w, int n, float eps, int has_w) {
  extern __shared__ double red[];
  const int tid = threadIdx.x, nt = blockDim.x;
  double ls = 0.0;
  for (int i = tid; i < n; i += nt) { const double v = x[i]; ls += v * v; }
  red[tid] = ls;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  const float r = 1.0f / sqrtf(static_cast<float>(red[0] / static_cast<double>(n)) + eps);
  for (int i = tid; i < n; i += nt) out[i] = has_w ? x[i] * r * w[i] : x[i] * r;
}
void RmsNormLaunch(Queue& q, float* out, const float* x, const float* w, int64_t n, float eps,
                   bool has_w) {
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 256;
  RmsNormKernel<<<1, block, block * sizeof(double), s>>>(out, x, w, static_cast<int>(n), eps,
                                                         has_w ? 1 : 0);
  Check(cudaGetLastError(), "rms_norm launch");
}

// Brick C part 2 — BATCHED RMSNorm over `rows` independent [n] segments in ONE
// launch (blockIdx.x = row). Used for the 64-head per-head q-RMS (has_w=false) so
// the resident decode does not issue nh=64 separate rms_norm launches per layer.
// Same block-tree reduction as RmsNormKernel ⇒ per-row IDENTICAL to it (the same
// characterized near-tie vs host double-sequential; a shared weight w[n] applies to
// every row when has_w).
__global__ void RmsNormRowsKernel(float* __restrict__ out, const float* __restrict__ x,
                                  const float* __restrict__ w, int rows, int n, float eps,
                                  int has_w) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + static_cast<int64_t>(row) * n;
  float* outr = out + static_cast<int64_t>(row) * n;
  extern __shared__ double red[];
  const int tid = threadIdx.x, nt = blockDim.x;
  double ls = 0.0;
  for (int i = tid; i < n; i += nt) { const double v = xr[i]; ls += v * v; }
  red[tid] = ls;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  const float r = 1.0f / sqrtf(static_cast<float>(red[0] / static_cast<double>(n)) + eps);
  for (int i = tid; i < n; i += nt) outr[i] = has_w ? xr[i] * r * w[i] : xr[i] * r;
}
void RmsNormRowsLaunch(Queue& q, float* out, const float* x, const float* w, int64_t rows,
                       int64_t n, float eps, bool has_w) {
  if (n == 0 || rows == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 256;
  RmsNormRowsKernel<<<static_cast<unsigned>(rows), block, block * sizeof(double), s>>>(
      out, x, w, static_cast<int>(rows), static_cast<int>(n), eps, has_w ? 1 : 0);
  Check(cudaGetLastError(), "rms_norm_rows launch");
}

__device__ double YarnCorrDimDev(int n_dims, int n_ctx_orig, double beta, double base) {
  const double kPi = 3.14159265358979323846;
  return static_cast<double>(n_dims) *
         log(static_cast<double>(n_ctx_orig) / (beta * 2.0 * kPi)) / (2.0 * log(base));
}
// FLOAT YaRN correction-dim (mirrors ds4's inline corr0/corr1: logf, 2*logf(base)).
__device__ float YarnCorrDimDevF(int n_dims, int n_ctx_orig, float beta, float base) {
  const float kPi = 3.14159265358979323846f;
  return static_cast<float>(n_dims) *
         logf(static_cast<float>(n_ctx_orig) / (beta * 2.0f * kPi)) / (2.0f * logf(base));
}
// One thread per row; the sequential-recurrence RoPE (host RopeInplaceLayer) on
// v[row*stride + off .. +r]. Near-tie vs host (device cos/sin vs libm; the double
// recurrence theta_extrap*=theta_scale preserves host order).
__global__ void RopeKernel(float* v, int num_rows, int row_stride, int off, int r,
                           const int* row_pos, double base, double freq_scale, double ext_factor,
                           int n_ctx_orig, double beta_fast, double beta_slow, int inverse) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  float* vv = v + static_cast<int64_t>(row) * row_stride + off;
  const double theta_scale = pow(base, -2.0 / static_cast<double>(r));
  const double sin_sign = inverse ? -1.0 : 1.0;
  double corr_lo = 0.0, corr_hi = 0.0;
  if (ext_factor != 0.0) {
    corr_lo = fmax(0.0, floor(YarnCorrDimDev(r, n_ctx_orig, beta_fast, base)));
    corr_hi = fmin(static_cast<double>(r - 1), ceil(YarnCorrDimDev(r, n_ctx_orig, beta_slow, base)));
  }
  double theta_extrap = static_cast<double>(row_pos[row]);
  for (int i = 0; i < r; i += 2) {
    const double theta_interp = freq_scale * theta_extrap;
    double theta = theta_interp;
    if (ext_factor != 0.0) {
      const double y = (static_cast<double>(i / 2) - corr_lo) / fmax(0.001, corr_hi - corr_lo);
      const double ramp = (1.0 - fmin(1.0, fmax(0.0, y))) * ext_factor;
      theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
    }
    const float c = static_cast<float>(cos(theta));
    const float sn = static_cast<float>(sin_sign * sin(theta));
    const float x0 = vv[i], x1 = vv[i + 1];
    vv[i] = x0 * c - x1 * sn;
    vv[i + 1] = x0 * sn + x1 * c;
    theta_extrap *= theta_scale;
  }
}
void RopeLaunch(Queue& q, float* v, int64_t num_rows, int64_t row_stride, int64_t off, int64_t r,
                const int* row_pos, double base, double freq_scale, double ext_factor,
                int64_t n_ctx_orig, double beta_fast, double beta_slow, bool inverse) {
  if (num_rows == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 128;
  const unsigned grid = static_cast<unsigned>((num_rows + block - 1) / block);
  RopeKernel<<<grid, block, 0, s>>>(v, static_cast<int>(num_rows), static_cast<int>(row_stride),
                                    static_cast<int>(off), static_cast<int>(r), row_pos, base,
                                    freq_scale, ext_factor, static_cast<int>(n_ctx_orig), beta_fast,
                                    beta_slow, inverse ? 1 : 0);
  Check(cudaGetLastError(), "rope launch");
}

// ── Brick 7: FUSED per-head RMSNorm + RoPE (ds4 head_rms_norm_rope_tail_kernel :5873
// + dsv4_qkv_rms_norm_rows_kv_rope_kernel :5779). Collapses the resident-decode
// {rms_norm_rows + rope} launch PAIR (q per-head norm+rope, kv norm+rope) into ONE
// kernel — the normalized values never round-trip through HBM — AND parallelizes the
// RoPE tail (block-per-row, threads split the r/2 pairs) instead of the old
// one-thread-per-row serial recurrence.
//
// BIT-IDENTICAL to the split {RmsNormRowsKernel ; RopeKernel} path:
//  - the RMS reduction is the SAME double block-tree reduce as RmsNormRowsKernel;
//  - the RoPE recurrence `theta_extrap *= theta_scale` is a LEFT-FOLD, so pair p's
//    theta_extrap = pos·theta_scale^p, and a thread reaching pair p by p sequential
//    mults from pos reproduces the recurrence's exact product order (double);
//  - cos/sin stay double (like RopeKernel), NO ds4 attn_factor/mscale (we have none).
// do_norm=false + inverse=true covers the standalone post-attention inverse o-RoPE
// (line-1356 site) with the SAME parallelized bit-exact tail, no norm.
__global__ void NormRopeRowsKernel(float* __restrict__ out, const float* __restrict__ in,
                                   const float* __restrict__ w, int rows, int n, int off, int r,
                                   const int* __restrict__ row_pos, double base, double freq_scale,
                                   double ext_factor, int n_ctx_orig, double beta_fast,
                                   double beta_slow, int inverse, int has_w, int do_norm,
                                   float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x, nt = blockDim.x;
  const float* inr = in + static_cast<int64_t>(row) * n;
  float* outr = out + static_cast<int64_t>(row) * n;
  extern __shared__ double red[];
  float rscale = 1.0f;
  if (do_norm) {
    double ls = 0.0;
    for (int i = tid; i < n; i += nt) { const double v = inr[i]; ls += v * v; }
    red[tid] = ls;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    rscale = 1.0f / sqrtf(static_cast<float>(red[0] / static_cast<double>(n)) + eps);
    // normalized passthrough for the NON-roped dims [0, off) and [off+r, n); the
    // roped tail [off, off+r) is written by the RoPE loop below (from the same nv).
    for (int i = tid; i < off; i += nt) outr[i] = has_w ? inr[i] * rscale * w[i] : inr[i] * rscale;
    for (int i = off + r + tid; i < n; i += nt)
      outr[i] = has_w ? inr[i] * rscale * w[i] : inr[i] * rscale;
  }
  // RoPE tail — one thread per pair p (dims off+2p, off+2p+1); bit-exact left-fold.
  const double theta_scale = pow(base, -2.0 / static_cast<double>(r));
  const double sin_sign = inverse ? -1.0 : 1.0;
  double corr_lo = 0.0, corr_hi = 0.0;
  if (ext_factor != 0.0) {
    corr_lo = fmax(0.0, floor(YarnCorrDimDev(r, n_ctx_orig, beta_fast, base)));
    corr_hi = fmin(static_cast<double>(r - 1), ceil(YarnCorrDimDev(r, n_ctx_orig, beta_slow, base)));
  }
  const double pos = static_cast<double>(row_pos[row]);
  const int pairs = r / 2;
  for (int p = tid; p < pairs; p += nt) {
    const int i = 2 * p;
    double theta_extrap = pos;
    for (int j = 0; j < p; ++j) theta_extrap *= theta_scale;  // == recurrence te at pair p
    const double theta_interp = freq_scale * theta_extrap;
    double theta = theta_interp;
    if (ext_factor != 0.0) {
      const double y = (static_cast<double>(i / 2) - corr_lo) / fmax(0.001, corr_hi - corr_lo);
      const double ramp = (1.0 - fmin(1.0, fmax(0.0, y))) * ext_factor;
      theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
    }
    const float c = static_cast<float>(cos(theta));
    const float sn = static_cast<float>(sin_sign * sin(theta));
    float x0, x1;
    if (do_norm) {
      x0 = has_w ? inr[off + i] * rscale * w[off + i] : inr[off + i] * rscale;
      x1 = has_w ? inr[off + i + 1] * rscale * w[off + i + 1] : inr[off + i + 1] * rscale;
    } else {
      x0 = outr[off + i];
      x1 = outr[off + i + 1];
    }
    outr[off + i] = x0 * c - x1 * sn;
    outr[off + i + 1] = x0 * sn + x1 * c;
  }
}
// ── ds4-fold (VT_V4_ROPE_FLOAT): the FLOAT norm+RoPE, a 1:1 structural mirror of the
// double NormRopeRowsKernel above and of ds4's head_rms_norm_rope_tail_kernel
// (~/w8run/ds4/ds4_cuda.cu:5873) + dsv4_qkv_rms_norm_rows_kv_rope_kernel (:5779). Two
// changes vs the double kernel, both ds4-faithful: (1) the RMS reduction + the RoPE
// pow/cos/sin run in FLOAT (GB10 throttles FP64 transcendentals ~1/32-1/64 of FP32 —
// the SAME failure mode the MHC FP64->FP32 fold fixed); (2) the RoPE theta uses ds4's
// DIRECT powf(base, -i/r) per pair (i=2p) instead of the double O(pairs^2) left-fold
// recurrence (theta_extrap*=theta_scale). powf(base,-2p/r) == theta_scale^p, so the
// algebra is identical — only the accumulation precision + per-pair cost change. NO
// ds4 attn_factor/mscale (we have none, matching the double kernel). Same grid=(rows),
// block=256; do_norm/has_w/inverse branches unchanged.
__global__ void NormRopeRowsFloatKernel(float* __restrict__ out, const float* __restrict__ in,
                                        const float* __restrict__ w, int rows, int n, int off,
                                        int r, const int* __restrict__ row_pos, float base,
                                        float freq_scale, float ext_factor, int n_ctx_orig,
                                        float beta_fast, float beta_slow, int inverse, int has_w,
                                        int do_norm, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x, nt = blockDim.x;
  const float* inr = in + static_cast<int64_t>(row) * n;
  float* outr = out + static_cast<int64_t>(row) * n;
  extern __shared__ float redf[];
  float rscale = 1.0f;
  if (do_norm) {
    float ls = 0.0f;
    for (int i = tid; i < n; i += nt) { const float v = inr[i]; ls += v * v; }
    redf[tid] = ls;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
    rscale = rsqrtf(redf[0] / static_cast<float>(n) + eps);
    for (int i = tid; i < off; i += nt) outr[i] = has_w ? inr[i] * rscale * w[i] : inr[i] * rscale;
    for (int i = off + r + tid; i < n; i += nt)
      outr[i] = has_w ? inr[i] * rscale * w[i] : inr[i] * rscale;
  }
  // RoPE tail — one thread per pair p (dims off+2p, off+2p+1); float, direct powf.
  const float sin_sign = inverse ? -1.0f : 1.0f;
  float corr_lo = 0.0f, corr_hi = 0.0f;
  if (ext_factor != 0.0f) {
    corr_lo = fmaxf(0.0f, floorf(YarnCorrDimDevF(r, n_ctx_orig, beta_fast, base)));
    corr_hi = fminf(static_cast<float>(r - 1), ceilf(YarnCorrDimDevF(r, n_ctx_orig, beta_slow, base)));
  }
  const float pos = static_cast<float>(row_pos[row]);
  const int pairs = r / 2;
  for (int p = tid; p < pairs; p += nt) {
    const int i = 2 * p;
    const float theta_extrap = pos * powf(base, -static_cast<float>(i) / static_cast<float>(r));
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
      const float y = (static_cast<float>(i / 2) - corr_lo) / fmaxf(0.001f, corr_hi - corr_lo);
      const float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
      theta = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
    }
    const float c = cosf(theta);
    const float sn = sin_sign * sinf(theta);
    float x0, x1;
    if (do_norm) {
      x0 = has_w ? inr[off + i] * rscale * w[off + i] : inr[off + i] * rscale;
      x1 = has_w ? inr[off + i + 1] * rscale * w[off + i + 1] : inr[off + i + 1] * rscale;
    } else {
      x0 = outr[off + i];
      x1 = outr[off + i + 1];
    }
    outr[off + i] = x0 * c - x1 * sn;
    outr[off + i + 1] = x0 * sn + x1 * c;
  }
}

// ds4-fold gate: route norm+RoPE through the FLOAT (ds4-mirroring) kernel instead of the
// bit-faithful FP64 path. DEFAULT ON (parity-enablers-ship-as-defaults) — a MEASURED
// decode win on GB10 where FP64 is throttled; VT_V4_ROPE_FLOAT=0 opts back to the double
// path for a same-binary A/B. Read once (process-wide) so the captured decode graph bakes
// a consistent path.
static bool RopeFloatEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_ROPE_FLOAT");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

void NormRopeRowsLaunch(Queue& q, float* out, const float* in, const float* w, int64_t rows,
                        int64_t n, int64_t off, int64_t r, const int* row_pos, double base,
                        double freq_scale, double ext_factor, int64_t n_ctx_orig, double beta_fast,
                        double beta_slow, bool inverse, bool has_w, bool do_norm, float eps) {
  if (rows == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 256;
  if (RopeFloatEnabled()) {
    // ds4-fold: the identical algebra in FLOAT (GB10 FP64 is ~1/32-1/64 of FP32).
    NormRopeRowsFloatKernel<<<static_cast<unsigned>(rows), block, block * sizeof(float), s>>>(
        out, in, w, static_cast<int>(rows), static_cast<int>(n), static_cast<int>(off),
        static_cast<int>(r), row_pos, static_cast<float>(base), static_cast<float>(freq_scale),
        static_cast<float>(ext_factor), static_cast<int>(n_ctx_orig),
        static_cast<float>(beta_fast), static_cast<float>(beta_slow), inverse ? 1 : 0,
        has_w ? 1 : 0, do_norm ? 1 : 0, eps);
    Check(cudaGetLastError(), "norm_rope_rows launch (float)");
    return;
  }
  NormRopeRowsKernel<<<static_cast<unsigned>(rows), block, block * sizeof(double), s>>>(
      out, in, w, static_cast<int>(rows), static_cast<int>(n), static_cast<int>(off),
      static_cast<int>(r), row_pos, base, freq_scale, ext_factor, static_cast<int>(n_ctx_orig),
      beta_fast, beta_slow, inverse ? 1 : 0, has_w ? 1 : 0, do_norm ? 1 : 0, eps);
  Check(cudaGetLastError(), "norm_rope_rows launch");
}

// MoE combine: out[h] = Σ_a weights[a]*eo[a*H+h] (one thread per h; sequential over
// a → host order). Near-tie vs host (the device contracts weights[a]*eo+acc to an
// FMA; the host does separate multiply+add) — ~last-ULP, characterized.
__global__ void MoeCombineKernel(float* out, const float* eo, const float* weights, int A, int H) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float acc = 0.0f;
  for (int a = 0; a < A; ++a) acc += weights[a] * eo[static_cast<int64_t>(a) * H + h];
  out[h] = acc;
}
void MoeCombineLaunch(Queue& q, float* out, const float* eo, const float* weights, int64_t A,
                      int64_t H) {
  if (H == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 128;
  const unsigned grid = static_cast<unsigned>((H + block - 1) / block);
  MoeCombineKernel<<<grid, block, 0, s>>>(out, eo, weights, static_cast<int>(A),
                                          static_cast<int>(H));
  Check(cudaGetLastError(), "moe_combine launch");
}

// Brick D step 1 — DEVICE router gate (the last non-capturable host op of the
// resident decode): gating[e] = Σ_h x[h]·bf16→f32(W[e·H+h]) for the [ne,H] BF16
// `ffn.gate.weight`. One thread per expert; the dot is SEQUENTIAL in f32 with the
// exact `bits<<16` bf16 upcast — BIT-IDENTICAL to the host CPU MatmulBT
// (cpu_ops.cpp MatmulChunked<true>, f32 accumulate, LoadF32=BF16ToF32). Replaces
// the CPU MatmulBT (f32-act×bf16-weight, which the CUDA elementwise MatmulBT lacks)
// so the resident step is 100% device — no host op inside the capture region.
// TUNED (glue-tune): ONE WARP per expert (was one thread per expert looping H=4096
// serially — 7.5% of the graphed step). The 32 lanes stride over H, then a warp-tree
// reduce. NEAR-TIE vs the prior sequential kernel (the strided partials reorder the
// f32 sum vs h=0,1,2,… — the same reassociation class as the K-quant CUDA GEMM); the
// integer/bf16 upcast is unchanged, and the real-model top-k routing stays robust.
__global__ void RouterGateKernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                 float* __restrict__ gating, int ne, int H) {
  const int e = blockIdx.x * blockDim.y + threadIdx.y;  // one warp per expert
  if (e >= ne) return;
  const int lane = threadIdx.x;
  const uint16_t* we = w + static_cast<int64_t>(e) * H;
  float acc = 0.0f;
  for (int h = lane; h < H; h += 32)
    acc += x[h] * __uint_as_float(static_cast<uint32_t>(we[h]) << 16);  // bf16→f32 exact
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) gating[e] = acc;
}
void RouterGateLaunch(Queue& q, float* gating, const float* x, const void* w_bf16, int64_t ne,
                      int64_t H) {
  if (ne == 0) return;
  cudaStream_t s = AsStream(q);
  constexpr int kWarpsPerBlock = 8;
  dim3 block(32, kWarpsPerBlock);
  const unsigned grid = static_cast<unsigned>((ne + kWarpsPerBlock - 1) / kWarpsPerBlock);
  RouterGateKernel<<<grid, block, 0, s>>>(x, static_cast<const uint16_t*>(w_bf16), gating,
                                          static_cast<int>(ne), static_cast<int>(H));
  Check(cudaGetLastError(), "router_gate launch");
}

// ── Brick A: device MLA decode/prefill attention (unified memory, in place) ───
// One block per (query t, head h). num KV heads = 1 (all heads share the cached
// latent kv[s]). BIT-IDENTICAL to the host SoftmaxWithSink path by preserving its
// accumulation ORDER: per-key dot sequential over d, then thread-0 sequential
// max/denom over s (incl. the sink), then per-d output sequential over s. `e[]`
// (dynamic shared, sized kv_base+T) holds scores then exp() weights.
__global__ void DecodeAttnKernel(float* __restrict__ o, const float* __restrict__ q,
                                 const float* __restrict__ kv, const float* __restrict__ sink,
                                 int nh, int hd, int64_t kv_base, int T, float scale,
                                 bool no_sink) {
  const int th = blockIdx.x;      // in [0, T*nh)
  const int t = th / nh;
  const int h = th % nh;
  const int64_t n = kv_base + t + 1;  // causal: query t attends keys [0, kv_base+t]
  const float* qh = q + (static_cast<int64_t>(t) * nh + h) * hd;
  extern __shared__ float e[];    // [n] scores -> exp weights
  __shared__ float denom_sh;

  // Pass A: scores[s] = (qh · kv[s]) * scale — dot sequential over d (host order).
  for (int64_t s = threadIdx.x; s < n; s += blockDim.x) {
    const float* ks = kv + s * hd;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) acc += qh[d] * ks[d];
    e[s] = acc * scale;
  }
  __syncthreads();

  // Pass mid (thread 0, sequential — matches host m/denom order exactly).
  if (threadIdx.x == 0) {
    const float ninf = -INFINITY;
    const float sink_h = no_sink ? ninf : sink[h];
    float m = sink_h;
    for (int64_t s = 0; s < n; ++s) m = fmaxf(m, e[s]);
    if (m == ninf) {  // fully -inf row: 0/0 guard (host returns zeros)
      denom_sh = 0.0f;
    } else {
      float denom = expf(sink_h - m);  // sink -> denominator only
      for (int64_t s = 0; s < n; ++s) {
        const float ee = expf(e[s] - m);
        e[s] = ee;
        denom += ee;
      }
      denom_sh = denom;
    }
  }
  __syncthreads();

  // Pass B: o[d] = Σ_s (e[s]/denom) · kv[s][d] — sequential over s (host order).
  const float denom = denom_sh;
  float* oh = o + (static_cast<int64_t>(t) * nh + h) * hd;
  if (denom == 0.0f) {
    for (int d = threadIdx.x; d < hd; d += blockDim.x) oh[d] = 0.0f;
    return;
  }
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t s = 0; s < n; ++s) acc += (e[s] / denom) * kv[s * hd + d];
    oh[d] = acc;
  }
}

void DecodeAttnLaunch(Queue& q, float* o, const float* query, const float* kv,
                      const float* sink, int64_t nh, int64_t hd, int64_t kv_base,
                      int64_t T, float scale, bool no_sink) {
  if (T == 0 || nh == 0) return;
  const int64_t n_max = kv_base + T;
  // scores/weights live in dynamic shared (sized to the largest query's key count).
  // Long contexts beyond this are a named residual (global-scratch variant, R3).
  if (n_max * static_cast<int64_t>(sizeof(float)) > 40 * 1024)
    throw std::runtime_error(
        "vt cuda deepseek_v4: decode_attn context exceeds the shared-memory KV window "
        "(long-context device attention is a named residual)");
  cudaStream_t s = AsStream(q);
  const dim3 grid(static_cast<unsigned>(T * nh));
  const unsigned block = 256;
  const unsigned shmem = static_cast<unsigned>(n_max) * sizeof(float);
  DecodeAttnKernel<<<grid, block, shmem, s>>>(o, query, kv, sink, static_cast<int>(nh),
                                              static_cast<int>(hd), kv_base,
                                              static_cast<int>(T), scale, no_sink);
  Check(cudaGetLastError(), "decode_attn launch");
  // NO sync here — the caller drains (Brick A) or captures (Brick D).
}

// ── Brick D step 2: GRAPH decode attention (T=1, capturable) ──────────────────
// DecodeAttnKernel bakes `kv_base` into the launch (host arg + dynamic shmem), so a
// captured graph would freeze the context. This variant reads the KV length from a
// DEVICE buffer `len_dev` at runtime, uses FIXED shared memory (max_cap keys), and
// attends the `len` prior keys in `cache[0..len)` PLUS the current token's key
// `deck_new` (as key index `len` — it is not yet appended to `cache`). The key set
// {cache[0..len), deck_new} == the eager kernel's cache[0..kv_base] (which already
// had this token's deck appended) in the SAME order ⇒ BIT-IDENTICAL to eager.
__global__ void DecodeAttnGKernel(float* __restrict__ o, const float* __restrict__ q,
                                  const float* __restrict__ cache,
                                  const float* __restrict__ deck_new,
                                  const float* __restrict__ sink, int nh, int hd,
                                  const int* __restrict__ len_dev, float scale, bool no_sink) {
  const int h = blockIdx.x;   // T=1 → t=0, one block per head
  const int len = *len_dev;   // # prior keys already in `cache` (== kv_base)
  const int n = len + 1;      // + this token's key (deck_new)
  const float* qh = q + static_cast<int64_t>(h) * hd;
  extern __shared__ float e[];  // [max_cap] scores → exp weights
  __shared__ float denom_sh;
  for (int s = threadIdx.x; s < n; s += blockDim.x) {
    const float* ks = (s < len) ? (cache + static_cast<int64_t>(s) * hd) : deck_new;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) acc += qh[d] * ks[d];
    e[s] = acc * scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const float ninf = -INFINITY;
    const float sink_h = no_sink ? ninf : sink[h];
    float m = sink_h;
    for (int s = 0; s < n; ++s) m = fmaxf(m, e[s]);
    if (m == ninf) {
      denom_sh = 0.0f;
    } else {
      float denom = expf(sink_h - m);
      for (int s = 0; s < n; ++s) { const float ee = expf(e[s] - m); e[s] = ee; denom += ee; }
      denom_sh = denom;
    }
  }
  __syncthreads();
  const float denom = denom_sh;
  float* oh = o + static_cast<int64_t>(h) * hd;
  if (denom == 0.0f) {
    for (int d = threadIdx.x; d < hd; d += blockDim.x) oh[d] = 0.0f;
    return;
  }
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < n; ++s) {
      const float* ks = (s < len) ? (cache + static_cast<int64_t>(s) * hd) : deck_new;
      acc += (e[s] / denom) * ks[d];
    }
    oh[d] = acc;
  }
}
void DecodeAttnGLaunch(Queue& q, float* o, const float* query, const float* cache,
                       const float* deck_new, const float* sink, int64_t nh, int64_t hd,
                       const int* len_dev, int64_t max_cap, float scale, bool no_sink) {
  if (nh == 0) return;
  if (max_cap * static_cast<int64_t>(sizeof(float)) > 40 * 1024)
    throw std::runtime_error(
        "vt cuda deepseek_v4: decode_attn_g max_cap exceeds the shared-memory KV window");
  cudaStream_t s = AsStream(q);
  const dim3 grid(static_cast<unsigned>(nh));
  const unsigned block = 256;
  const unsigned shmem = static_cast<unsigned>(max_cap) * sizeof(float);
  DecodeAttnGKernel<<<grid, block, shmem, s>>>(o, query, cache, deck_new, sink,
                                               static_cast<int>(nh), static_cast<int>(hd), len_dev,
                                               scale, no_sink);
  Check(cudaGetLastError(), "decode_attn_g launch");
}

// Brick 12 (ds4-gap launch consolidation): thin Tensor pass-throughs to the
// cuda_quant_dot.cu paired / block-diagonal Q8_0 GEMV kernels (same CUDA library, no
// drain — the resident chain drains/captures once). Bit-identical to the launches they
// replace (see the cuda_quant_dot.cu kernel comments).
void MatmulQ8_0PairLaunch(Queue& q, Tensor& out0, Tensor& out1, const Tensor& act,
                          const Tensor& w0, const Tensor& w1) {
  vt::cuda::MatmulQ8_0PairCuda(out0, out1, act, w0, w1, AsStream(q));
}
void MatmulQ8_0OloraALaunch(Queue& q, Tensor& out, const Tensor& act, const Tensor& w,
                            int64_t n_groups) {
  vt::cuda::MatmulQ8_0GroupDiagCuda(out, act, w, n_groups, AsStream(q));
}

// ── the per-family kernels-structs (registered through the seam) ──────────────
const MhcDeviceKernels kMhc = {&MhcSinkhornLaunch, &MhcPreLaunch, &MhcPostLaunch, &HcHeadLaunch,
                               &MhcPostInPlaceLaunch, &HcHeadInPlaceLaunch, &MhcPreInPlaceLaunch};
const DsaDeviceKernels kDsa = {&DsaWeightFoldLaunch, &DsaLogitsLaunch, &DsaTopkLaunch,
                               &SoftmaxSinkLaunch, &GroupedOLoraLaunch, &DecodeAttnLaunch,
                               &RmsNormLaunch, &RopeLaunch, &RmsNormRowsLaunch, &DecodeAttnGLaunch,
                               &NormRopeRowsLaunch, &MatmulQ8_0PairLaunch, &MatmulQ8_0OloraALaunch};
const CompressorDeviceKernels kComp = {&SaveScoreApeLaunch, &PoolNormLaunch, &Fp8EncodeLaunch,
                                       &Fp8DecodeLaunch};
// FUSED routed-MoE gate+up+SwiGLU — forwards to the cuda_quant_dot.cu kernel that
// holds the keep-quant dequant/dot machinery. Thin Tensor pass-through (no drain).
void MoeGateUpSwiGLULaunch(Queue& q, Tensor& out, const Tensor& act, const Tensor& gate_w,
                           const Tensor& up_w, const Tensor& expert_ids, float limit) {
  vt::cuda::MoeGateUpSwiGLUGroupedCuda(q, out, act, gate_w, up_w, expert_ids, limit);
}

const MoeDeviceKernels kMoe = {&SqrtSoftplusLaunch,        &RouteLaunch,
                               &ClampedSwiGLULaunch,       &ClampedSwiGLUInPlaceLaunch,
                               &RouteInPlaceLaunch,        &MoeCombineLaunch,
                               &RouterGateLaunch,          &MoeGateUpSwiGLULaunch};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kDeepseekV4Mhc, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kMhc)));
    RegisterOp(OpId::kDeepseekV4Dsa, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kDsa)));
    RegisterOp(OpId::kDeepseekV4Compressor, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kComp)));
    RegisterOp(OpId::kDeepseekV4Moe, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kMoe)));
  }
} registrar;

}  // namespace
}  // namespace vllm::deepseek_v4
