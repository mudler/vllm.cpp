// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA baseline kernels for rmsnorm / silu_and_mul / embedding / rope_neox.
// Correctness-grade (M0.6): plain grid-stride / one-block-per-row kernels, f32
// accumulation, double-precision RoPE angles matching the CPU reference.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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

// ---------------------------------------------------------------------------
// rmsnorm: one block per row, shared-memory f32 tree reduction.

template <typename Tin, typename Tout>
__global__ void RmsNormRowKernel(Tout* out, const Tin* x, const Tin* w, float* residual,
                                 int64_t h, float eps, bool gemma) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * h;
  Tout* orow = out + row * h;
  float* rrow = residual == nullptr ? nullptr : residual + row * h;

  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float v = Load(xrow, j);
    if (rrow != nullptr) {
      v += rrow[j];
      rrow[j] = v;  // new residual stream, updated in place (f32)
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
    const float v = rrow != nullptr ? rrow[j] : Load(xrow, j);
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(orow, j, v * inv * wj);
  }
}

template <typename Tin>
void LaunchRmsNorm(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  float* res = residual == nullptr ? nullptr : residual->Ptr<float>();
  const unsigned rows = static_cast<unsigned>(t);
  switch (out.dtype) {
    case DType::kF32:
      RmsNormRowKernel<Tin, float><<<rows, kBlock, 0, s>>>(
          out.Ptr<float>(), x.Ptr<Tin>(), w.Ptr<Tin>(), res, h, args.eps, args.gemma);
      break;
    case DType::kBF16:
      RmsNormRowKernel<Tin, __nv_bfloat16><<<rows, kBlock, 0, s>>>(
          out.Ptr<__nv_bfloat16>(), x.Ptr<Tin>(), w.Ptr<Tin>(), res, h, args.eps, args.gemma);
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
// silu_and_mul: grid-stride over the T*D output elements.

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
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
