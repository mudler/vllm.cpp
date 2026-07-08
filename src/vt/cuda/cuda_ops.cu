// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA baseline kernels for rmsnorm / silu_and_mul / embedding / rope_neox.
// Correctness-grade (M0.6): plain grid-stride / one-block-per-row kernels, f32
// accumulation, double-precision RoPE angles matching the CPU reference.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

#ifdef VLLM_CPP_TRITON
// AOT-generated stable dispatcher (triton.tools.link output; see
// cmake/TritonAOT.cmake). Plain C symbols — wrap in extern "C" so this C++ TU
// links against them unmangled. rmsnorm_fwd_default(stream, out, x, w, n_rows,
// n_cols, eps) launches the EMBEDDED cubin via the CUDA driver API (no Triton /
// Python at runtime). The .h #include <cuda.h> for CUresult/CUstream/CUdeviceptr.
extern "C" {
#include "rmsnorm_fwd.h"
}
#endif

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

#ifdef VLLM_CPP_TRITON
// VT_TRITON_RMSNORM=1 routes CUDA RmsNorm through the AOT Triton cubin, but ONLY
// for the exact configuration the proof kernel implements: f32 x/out/weight, the
// gemma (1 + w) variant, and NO fused residual. Every other call falls through to
// the hand kernel below, so the toggle is a safe drop-in. Read fresh each call so
// a test can flip the path within one process (mirrors FusedTier()).
bool TritonRmsNormEnabled() {
  const char* e = std::getenv("VT_TRITON_RMSNORM");
  return e != nullptr && e[0] == '1';
}

// Returns true iff it handled the op (launched the Triton kernel or there was
// nothing to do); false means "not applicable, use the hand path".
bool TryTritonRmsNorm(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const RmsNormArgs& args, const Tensor* residual) {
  if (!TritonRmsNormEnabled()) return false;
  if (residual != nullptr || !args.gemma) return false;
  if (x.dtype != DType::kF32 || out.dtype != DType::kF32 || w.dtype != DType::kF32) return false;
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return true;  // empty: nothing to launch, treat as handled
  // Device pointers (>=256B aligned by cudaMalloc) satisfy the launcher's *fp32:16
  // alignment specialization. The stable symbol is resolved by triton.tools.link,
  // so no hash is hardcoded here.
  const CUresult r = rmsnorm_fwd_default(
      s, reinterpret_cast<CUdeviceptr>(out.Ptr<float>()),
      reinterpret_cast<CUdeviceptr>(x.Ptr<float>()),
      reinterpret_cast<CUdeviceptr>(w.Ptr<float>()), static_cast<int32_t>(t),
      static_cast<int32_t>(h), args.eps);
  VT_CHECK(r == CUDA_SUCCESS, "cuda rmsnorm(triton): launcher returned non-success");
  return true;
}
#endif  // VLLM_CPP_TRITON

void RmsNormKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual) {
  VT_CHECK(w.dtype == x.dtype, "cuda rmsnorm: weight dtype must match x");
#ifdef VLLM_CPP_TRITON
  if (TryTritonRmsNorm(AsStream(q), out, x, w, args, residual)) return;
#endif
  switch (x.dtype) {
    case DType::kF32: LaunchRmsNorm<float>(AsStream(q), out, x, w, args, residual); break;
    case DType::kBF16:
      LaunchRmsNorm<__nv_bfloat16>(AsStream(q), out, x, w, args, residual);
      break;
    default: VT_CHECK(false, "cuda rmsnorm: unsupported input dtype (f32/bf16 only)");
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

template <typename Tsrc, typename Tout>
__global__ void AttnQkNormRopeGateKernel(Tout* q_out, Tout* k_out, Tout* gate_out,
                                         const Tsrc* qgate, const Tsrc* kf, const float* q_norm,
                                         const float* k_norm, const float* cos_sin, int64_t hq,
                                         int64_t hkv, int64_t dh, int rot, int64_t half, float eps,
                                         bool gemma) {
  const int64_t tok = blockIdx.x;
  const int64_t head = blockIdx.y;  // [0, hq+hkv)
  const bool is_q = head < hq;

  const Tsrc* src;
  const float* w;
  Tout* out;
  int64_t gate_base = 0;  // only meaningful for q heads
  if (is_q) {
    const int64_t qrow = tok * (hq * 2 * dh) + head * 2 * dh;
    src = qgate + qrow;       // q half [0,dh)
    gate_base = qrow + dh;    // gate half [dh,2dh)
    w = q_norm;
    out = q_out + (tok * hq + head) * dh;
  } else {
    const int64_t hk = head - hq;
    src = kf + tok * (hkv * dh) + hk * dh;
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
    Tout* go = gate_out + (tok * hq + head) * dh;
    for (int64_t j = threadIdx.x; j < dh; j += kBlock) Store(go, j, Load(qgate, gate_base + j));
  }
}

template <typename Tsrc, typename Tout>
void LaunchAttnPreamble(cudaStream_t s, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                        const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                        const Tensor& k_norm, const Tensor& cos_sin, const RmsNormArgs& na,
                        const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  const int64_t half = ra.rotary_dim / 2;
  if (t == 0) return;
  dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>(hq + hkv));
  AttnQkNormRopeGateKernel<Tsrc, Tout><<<grid, kBlock, 0, s>>>(
      q_out.Ptr<Tout>(), k_out.Ptr<Tout>(), gate_out.Ptr<Tout>(), qgate.Ptr<Tsrc>(),
      kf.Ptr<Tsrc>(), q_norm.Ptr<float>(), k_norm.Ptr<float>(), cos_sin.Ptr<float>(), hq, hkv, dh,
      ra.rotary_dim, half, na.eps, na.gemma);
  Check(cudaGetLastError(), "attn_qk_norm_rope_gate launch");
}

template <typename Tsrc>
void LaunchAttnPreambleOut(cudaStream_t s, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                           const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                           const Tensor& k_norm, const Tensor& cos_sin, const RmsNormArgs& na,
                           const RopeArgs& ra) {
  switch (q_out.dtype) {
    case DType::kF32:
      LaunchAttnPreamble<Tsrc, float>(s, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm,
                                      cos_sin, na, ra);
      break;
    case DType::kBF16:
      LaunchAttnPreamble<Tsrc, __nv_bfloat16>(s, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm,
                                              cos_sin, na, ra);
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
    RegisterOp(OpId::kSiluAndMul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernelCuda)));
    RegisterOp(OpId::kEmbedding, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernelCuda)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernelCuda)));
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
