// ROCm dense basic kernels — correctness-grade ports for discrete R9700 bring-up.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}
inline unsigned GridFor(int64_t n) {
  if (n <= 0) return 1;
  const int64_t g = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(g > 65535 ? 65535 : g);
}

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

template <typename Tin, typename Tout>
__global__ void MulScalarK(Tout* out, const Tin* x, int64_t n, float s) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x)
    St(out, idx, Ld(x, idx) * s);
}

template <typename Tin, typename Tout>
__global__ void SoftCapK(Tout* out, const Tin* x, int64_t n, float cap) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x)
    St(out, idx, cap * tanhf(Ld(x, idx) / cap));
}

template <typename T>
__global__ void AddK(T* out, const T* a, const T* b, int64_t n) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x)
    St(out, idx, Ld(a, idx) + Ld(b, idx));
}

template <typename T>
__global__ void AddBcastK(T* out, const T* a, const T* b, int64_t rows, int64_t cols) {
  const int64_t n = rows * cols;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x)
    St(out, idx, Ld(a, idx) + Ld(b, idx % cols));
}

template <typename Tin, typename Tout>
__global__ void GeluMulK(Tout* out, const Tin* x, int64_t n, int64_t d) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    const int64_t i = idx / d, j = idx - i * d;
    const float g = Ld(x, i * 2 * d + j);
    const float up = Ld(x, i * 2 * d + d + j);
    const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    St(out, idx, 0.5f * g * (1.0f + tanhf(inner)) * up);
  }
}

// gate[i] and up[i] already separate (no pack) — MoE ExpertGeGLU hot path.
template <typename Tin, typename Tout>
__global__ void GeluMulSepK(Tout* out, const Tin* gate, const Tin* up, int64_t n) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    const float g = Ld(gate, idx);
    const float u = Ld(up, idx);
    const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    St(out, idx, 0.5f * g * (1.0f + tanhf(inner)) * u);
  }
}

template <typename Tin, typename Tout>
__global__ void SiluMulK(Tout* out, const Tin* x, int64_t n, int64_t d) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    const int64_t i = idx / d, j = idx - i * d;
    const float g = Ld(x, i * 2 * d + j);
    const float up = Ld(x, i * 2 * d + d + j);
    St(out, idx, (g / (1.0f + expf(-g))) * up);
  }
}

// Matmul / MatmulBT: see rocm_matmul_hipblaslt.hip (hipBLASLt).

__device__ inline double L3Scale(double freq, double sf, double lo, double hi, double omax) {
  if (!(sf > 0.0)) return freq;
  constexpr double kTwoPi = 6.283185307179586;
  const double low_w = omax / lo, high_w = omax / hi, wl = kTwoPi / freq;
  double smooth = 0.0;
  if (lo != hi) smooth = (omax / wl - lo) / (hi - lo);
  if (wl < high_w) return freq;
  if (wl > low_w) return freq / sf;
  return (1.0 - smooth) * freq / sf + smooth * freq;
}

template <typename Tid>
__global__ void RopeCacheK(float* cs, const Tid* pos, int64_t t, int rot, int64_t half,
                           double base, double sf, double lo, double hi, double omax) {
  const int64_t n = t * half;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    const int64_t pair = idx % half, tok = idx / half;
    const int64_t p = static_cast<int64_t>(pos[tok]);
    double freq = pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
    freq = L3Scale(freq, sf, lo, hi, omax);
    const double ang = static_cast<double>(p) * freq;
    cs[tok * rot + pair] = static_cast<float>(cos(ang));
    cs[tok * rot + half + pair] = static_cast<float>(sin(ang));
  }
}

template <typename Tin>
__global__ void ArgmaxK(int64_t* out, const Tin* logits, int64_t rows, int64_t cols) {
  const int64_t r = blockIdx.x;
  if (r >= rows) return;
  __shared__ float bv[kBlock];
  __shared__ int bi[kBlock];
  float best = -INFINITY;
  int arg = 0x7fffffff;  // unused lanes must lose the tree reduction
  for (int64_t j = threadIdx.x; j < cols; j += kBlock) {
    const float v = Ld(logits, r * cols + j);
    if (v > best || (v == best && (int)j < arg)) {
      best = v;
      arg = (int)j;
    }
  }
  bv[threadIdx.x] = best;
  bi[threadIdx.x] = arg;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if ((int)threadIdx.x < s) {
      if (bv[threadIdx.x + s] > bv[threadIdx.x] ||
          (bv[threadIdx.x + s] == bv[threadIdx.x] && bi[threadIdx.x + s] < bi[threadIdx.x])) {
        bv[threadIdx.x] = bv[threadIdx.x + s];
        bi[threadIdx.x] = bi[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[r] = static_cast<int64_t>(bi[0]);
}

}  // namespace

void MulScalarKernelRocm(Queue& q, Tensor& out, const Tensor& x, double scalar) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  const float s = static_cast<float>(scalar);
  hipStream_t st = AsStream(q);
  if (x.dtype == DType::kBF16 && out.dtype == DType::kBF16)
    MulScalarK<__hip_bfloat16, __hip_bfloat16>
        <<<GridFor(n), kBlock, 0, st>>>(out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), n, s);
  else if (x.dtype == DType::kF32 && out.dtype == DType::kF32)
    MulScalarK<float, float>
        <<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<float>(), n, s);
  else if (x.dtype == DType::kBF16 && out.dtype == DType::kF32)
    MulScalarK<__hip_bfloat16, float>
        <<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<__hip_bfloat16>(), n, s);
  else
    VT_CHECK(false, "rocm mul_scalar dtype");
  Check(hipGetLastError(), "mul_scalar");
}

void SoftCapKernelRocm(Queue& q, Tensor& out, const Tensor& x, double cap) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  const float c = static_cast<float>(cap);
  hipStream_t st = AsStream(q);
  if (x.dtype == DType::kBF16 && out.dtype == DType::kBF16)
    SoftCapK<__hip_bfloat16, __hip_bfloat16>
        <<<GridFor(n), kBlock, 0, st>>>(out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), n, c);
  else if (x.dtype == DType::kF32 && out.dtype == DType::kF32)
    SoftCapK<float, float>
        <<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<float>(), n, c);
  else if (x.dtype == DType::kBF16 && out.dtype == DType::kF32)
    SoftCapK<__hip_bfloat16, float>
        <<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<__hip_bfloat16>(), n, c);
  else
    VT_CHECK(false, "rocm soft_cap dtype");
  Check(hipGetLastError(), "soft_cap");
}

// mixed-dtype elementwise / broadcast add — f32 compute like CPU LoadF32
template <typename Ta, typename Tb, typename To>
__global__ void AddMixK(To* out, const Ta* a, const Tb* b, int64_t n) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x)
    St(out, idx, Ld(a, idx) + Ld(b, idx));
}
template <typename Ta, typename Tb, typename To>
__global__ void AddMixBcastK(To* out, const Ta* a, const Tb* b, int64_t rows,
                             int64_t cols) {
  const int64_t n = rows * cols;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x)
    St(out, idx, Ld(a, idx) + Ld(b, idx % cols));
}

void AddKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  hipStream_t st = AsStream(q);
  const int64_t n_elem = a.Numel();
  if (n_elem == 0) return;

  const bool bcast = (a.rank == 2 && b.rank == 1 && a.shape[1] == b.shape[0]);
  if (!(a.rank == b.rank || bcast)) VT_CHECK(false, "rocm add shapes");

  const int64_t rows = bcast ? a.shape[0] : 0;
  const int64_t cols = bcast ? a.shape[1] : 0;
  const int64_t n = bcast ? rows * cols : n_elem;

  auto launch = [&](auto ta, auto tb, auto to) {
    using Ta = decltype(ta);
    using Tb = decltype(tb);
    using To = decltype(to);
    if (bcast)
      AddMixBcastK<Ta, Tb, To><<<GridFor(n), kBlock, 0, st>>>(
          out.Ptr<To>(), a.Ptr<Ta>(), b.Ptr<Tb>(), rows, cols);
    else
      AddMixK<Ta, Tb, To><<<GridFor(n), kBlock, 0, st>>>(out.Ptr<To>(), a.Ptr<Ta>(),
                                                           b.Ptr<Tb>(), n);
  };

  // Same-dtype fast path + mixed f32/bf16 (lm_head: f32 logits + bf16 bias).
  if (a.dtype == DType::kBF16 && b.dtype == DType::kBF16 && out.dtype == DType::kBF16)
    launch(__hip_bfloat16{}, __hip_bfloat16{}, __hip_bfloat16{});
  else if (a.dtype == DType::kF32 && b.dtype == DType::kF32 && out.dtype == DType::kF32)
    launch(float{}, float{}, float{});
  else if (a.dtype == DType::kF32 && b.dtype == DType::kBF16 && out.dtype == DType::kF32)
    launch(float{}, __hip_bfloat16{}, float{});
  else if (a.dtype == DType::kBF16 && b.dtype == DType::kF32 && out.dtype == DType::kF32)
    launch(__hip_bfloat16{}, float{}, float{});
  else if (a.dtype == DType::kBF16 && b.dtype == DType::kBF16 && out.dtype == DType::kF32)
    launch(__hip_bfloat16{}, __hip_bfloat16{}, float{});
  else if (a.dtype == DType::kF32 && b.dtype == DType::kF32 && out.dtype == DType::kBF16)
    launch(float{}, float{}, __hip_bfloat16{});
  else
    VT_CHECK(false, "rocm add dtype combo");
  Check(hipGetLastError(), "add");
}

void GeluAndMulKernelRocm(Queue& q, Tensor& out, const Tensor& x) {
  const int64_t d = x.shape[1] / 2, n = x.shape[0] * d;
  if (n == 0) return;
  hipStream_t st = AsStream(q);
  if (x.dtype == DType::kBF16)
    GeluMulK<__hip_bfloat16, __hip_bfloat16><<<GridFor(n), kBlock, 0, st>>>(
        out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), n, d);
  else if (x.dtype == DType::kF32)
    GeluMulK<float, float>
        <<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<float>(), n, d);
  else
    VT_CHECK(false, "rocm gelu_and_mul dtype");
  Check(hipGetLastError(), "gelu_and_mul");
}

// Separate gate/up GeGLU (no pack). n = T * I elements.
void GeluMulSeparateRocm(Queue& q, void* out, const void* gate, const void* up, int64_t n,
                         DType dtype) {
  if (n == 0) return;
  hipStream_t st = AsStream(q);
  if (dtype == DType::kBF16)
    GeluMulSepK<__hip_bfloat16, __hip_bfloat16><<<GridFor(n), kBlock, 0, st>>>(
        static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(gate),
        static_cast<const __hip_bfloat16*>(up), n);
  else if (dtype == DType::kF32)
    GeluMulSepK<float, float><<<GridFor(n), kBlock, 0, st>>>(
        static_cast<float*>(out), static_cast<const float*>(gate), static_cast<const float*>(up),
        n);
  else
    VT_CHECK(false, "rocm gelu_mul_sep dtype");
  Check(hipGetLastError(), "gelu_mul_sep");
}

void SiluAndMulKernelRocm(Queue& q, Tensor& out, const Tensor& x) {
  const int64_t d = x.shape[1] / 2, n = x.shape[0] * d;
  if (n == 0) return;
  hipStream_t st = AsStream(q);
  if (x.dtype == DType::kBF16)
    SiluMulK<__hip_bfloat16, __hip_bfloat16><<<GridFor(n), kBlock, 0, st>>>(
        out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), n, d);
  else if (x.dtype == DType::kF32)
    SiluMulK<float, float>
        <<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<float>(), n, d);
  else
    VT_CHECK(false, "rocm silu_and_mul dtype");
  Check(hipGetLastError(), "silu_and_mul");
}

// MatmulKernelRocm / MatmulBTKernelRocm live in rocm_matmul_hipblaslt.hip

void RopeCosSinCacheKernelRocm(Queue& q, Tensor& cos_sin, const Tensor& pos,
                               const RopeArgs& args) {
  const int64_t t = cos_sin.shape[0], half = args.rotary_dim / 2, n = t * half;
  if (n == 0) return;
  hipStream_t st = AsStream(q);
  const double base = args.base, sf = args.llama3_scaling_factor, lo = args.llama3_low_freq_factor,
               hi = args.llama3_high_freq_factor, omax = args.llama3_orig_max_position;
  if (pos.dtype == DType::kI32)
    RopeCacheK<int32_t><<<GridFor(n), kBlock, 0, st>>>(
        cos_sin.Ptr<float>(), pos.Ptr<int32_t>(), t, args.rotary_dim, half, base, sf, lo, hi,
        omax);
  else
    RopeCacheK<int64_t><<<GridFor(n), kBlock, 0, st>>>(
        cos_sin.Ptr<float>(), pos.Ptr<int64_t>(), t, args.rotary_dim, half, base, sf, lo, hi,
        omax);
  Check(hipGetLastError(), "rope_cos_sin_cache");
}

void GreedyArgmaxKernelRocm(Queue& q, Tensor& out, const Tensor& logits) {
  const int64_t rows = logits.shape[0], cols = logits.shape[1];
  if (rows == 0) return;
  VT_CHECK(out.dtype == DType::kI64 || out.dtype == DType::kI32,
           "rocm greedy_argmax: out must be i64 or i32");
  hipStream_t st = AsStream(q);
  // Engine + cross-device tests use i64 token ids.
  if (out.dtype == DType::kI64) {
    if (logits.dtype == DType::kF32)
      ArgmaxK<float><<<static_cast<unsigned>(rows), kBlock, 0, st>>>(
          out.Ptr<int64_t>(), logits.Ptr<float>(), rows, cols);
    else if (logits.dtype == DType::kBF16)
      ArgmaxK<__hip_bfloat16><<<static_cast<unsigned>(rows), kBlock, 0, st>>>(
          out.Ptr<int64_t>(), logits.Ptr<__hip_bfloat16>(), rows, cols);
    else
      VT_CHECK(false, "rocm greedy_argmax logits dtype");
  } else {
    // rare i32 path — write via i64 kernel into temp is overkill; cast pointer
    VT_CHECK(false, "rocm greedy_argmax: i32 out not used; use i64");
  }
  Check(hipGetLastError(), "greedy_argmax");
}

// ---------------- CastBf16 / CastF32 (support strided packed rows) ----------
__global__ void CastBf16K(__hip_bfloat16* out, const float* in, int64_t n, int64_t row_size,
                          int64_t row_stride) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const int64_t row = i / row_size, col = i - row * row_size;
    out[i] = __float2bfloat16(in[row * row_stride + col]);
  }
}
__global__ void CastF32K(float* out, const __hip_bfloat16* in, int64_t n, int64_t row_size,
                         int64_t row_stride) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const int64_t row = i / row_size, col = i - row * row_size;
    out[i] = __bfloat162float(in[row * row_stride + col]);
  }
}

void CastBf16KernelRocm(Queue& q, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  const int64_t rows = in.shape[0];
  const int64_t row_size = n / rows;
  CastBf16K<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      out.Ptr<__hip_bfloat16>(), in.Ptr<float>(), n, row_size, in.stride[0]);
  Check(hipGetLastError(), "cast_bf16");
}

void CastF32KernelRocm(Queue& q, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  const int64_t rows = in.shape[0];
  const int64_t row_size = n / rows;
  CastF32K<<<GridFor(n), kBlock, 0, AsStream(q)>>>(out.Ptr<float>(), in.Ptr<__hip_bfloat16>(),
                                                     n, row_size, in.stride[0]);
  Check(hipGetLastError(), "cast_f32");
}

// ---------------- QkvSplit ----------------
template <typename T>
__global__ void QkvSplitK(T* q_out, T* k_out, T* v_out, const T* qkv, int64_t t, int64_t q_dim,
                          int64_t k_dim, int64_t v_dim) {
  const int64_t total = q_dim + k_dim + v_dim;
  const int64_t nq = t * q_dim, nk = t * k_dim, nv = t * v_dim;
  int64_t n = nq;
  if (nk > n) n = nk;
  if (nv > n) n = nv;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    if (idx < nq) {
      const int64_t i = idx / q_dim, j = idx % q_dim;
      q_out[idx] = qkv[i * total + j];
    }
    if (idx < nk) {
      const int64_t i = idx / k_dim, j = idx % k_dim;
      k_out[idx] = qkv[i * total + q_dim + j];
    }
    if (idx < nv) {
      const int64_t i = idx / v_dim, j = idx % v_dim;
      v_out[idx] = qkv[i * total + q_dim + k_dim + j];
    }
  }
}

void QkvSplitKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                        const Tensor& qkv) {
  const int64_t t = qkv.shape[0];
  if (t == 0) return;
  const int64_t q_dim = q_out.Numel() / t, k_dim = k_out.Numel() / t, v_dim = v_out.Numel() / t;
  int64_t n = t * q_dim;
  if (t * k_dim > n) n = t * k_dim;
  if (t * v_dim > n) n = t * v_dim;
  hipStream_t st = AsStream(q);
  if (qkv.dtype == DType::kBF16)
    QkvSplitK<__hip_bfloat16><<<GridFor(n), kBlock, 0, st>>>(
        q_out.Ptr<__hip_bfloat16>(), k_out.Ptr<__hip_bfloat16>(), v_out.Ptr<__hip_bfloat16>(),
        qkv.Ptr<__hip_bfloat16>(), t, q_dim, k_dim, v_dim);
  else if (qkv.dtype == DType::kF32)
    QkvSplitK<float><<<GridFor(n), kBlock, 0, st>>>(
        q_out.Ptr<float>(), k_out.Ptr<float>(), v_out.Ptr<float>(), qkv.Ptr<float>(), t, q_dim,
        k_dim, v_dim);
  else
    VT_CHECK(false, "rocm qkv_split dtype");
  Check(hipGetLastError(), "qkv_split");
}

// ---------------- LayerNorm (tree reduce — wavefront-agnostic) -------------
template <typename OutT, typename InT, typename WT>
__global__ void LayerNormRowK(OutT* out, const InT* x, const WT* weight, const WT* bias,
                              int64_t d, float eps) {
  const int64_t row = blockIdx.x;
  const InT* xr = x + row * d;
  OutT* orow = out + row * d;
  __shared__ float partial[kBlock];

  float sum = 0.f;
  for (int64_t i = threadIdx.x; i < d; i += kBlock) sum += Ld(xr, i);
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if ((int)threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float mean = partial[0] / static_cast<float>(d);
  __syncthreads();

  float sq = 0.f;
  for (int64_t i = threadIdx.x; i < d; i += kBlock) {
    const float dv = Ld(xr, i) - mean;
    sq += dv * dv;
  }
  partial[threadIdx.x] = sq;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if ((int)threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float rstd = rsqrtf(partial[0] / static_cast<float>(d) + eps);

  for (int64_t i = threadIdx.x; i < d; i += kBlock) {
    float v = (Ld(xr, i) - mean) * rstd;
    if (weight != nullptr) v *= Ld(weight, i);
    if (bias != nullptr) v += Ld(bias, i);
    St(orow, i, v);
  }
}

void LayerNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
                         const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  if (rows == 0 || d == 0) return;
  hipStream_t st = AsStream(q);
  const unsigned grid = static_cast<unsigned>(rows < 65535 ? rows : 65535);
  auto launch = [&](auto out_tag, auto in_tag, auto w_tag) {
    using OutT = decltype(out_tag);
    using InT = decltype(in_tag);
    using WT = decltype(w_tag);
    const WT* wp = weight ? static_cast<const WT*>(weight->data) : nullptr;
    const WT* bp = bias ? static_cast<const WT*>(bias->data) : nullptr;
    LayerNormRowK<OutT, InT, WT><<<grid, kBlock, 0, st>>>(
        static_cast<OutT*>(out.data), static_cast<const InT*>(x.data), wp, bp, d, args.eps);
  };
  const DType wdt =
      weight ? weight->dtype : (bias ? bias->dtype : x.dtype);
  if (out.dtype == DType::kBF16 && x.dtype == DType::kBF16 && wdt == DType::kBF16)
    launch(__hip_bfloat16{}, __hip_bfloat16{}, __hip_bfloat16{});
  else if (out.dtype == DType::kBF16 && x.dtype == DType::kBF16 && wdt == DType::kF32)
    launch(__hip_bfloat16{}, __hip_bfloat16{}, float{});
  else if (out.dtype == DType::kF32 && x.dtype == DType::kF32 && wdt == DType::kF32)
    launch(float{}, float{}, float{});
  else if (out.dtype == DType::kBF16 && x.dtype == DType::kF32 && wdt == DType::kF32)
    launch(__hip_bfloat16{}, float{}, float{});
  else if (out.dtype == DType::kF32 && x.dtype == DType::kBF16 && wdt == DType::kBF16)
    launch(float{}, __hip_bfloat16{}, __hip_bfloat16{});
  else if (out.dtype == DType::kF32 && x.dtype == DType::kBF16 && wdt == DType::kF32)
    launch(float{}, __hip_bfloat16{}, float{});
  else
    VT_CHECK(false, "rocm layer_norm dtype");
  Check(hipGetLastError(), "layer_norm");
}

// ---------------- Relu ----------------
template <typename T>
__global__ void ReluK(T* out, const T* x, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const float v = Ld(x, i);
    St(out, i, v > 0.f ? v : 0.f);
  }
}

void ReluKernelRocm(Queue& q, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  hipStream_t st = AsStream(q);
  if (x.dtype == DType::kBF16)
    ReluK<__hip_bfloat16><<<GridFor(n), kBlock, 0, st>>>(out.Ptr<__hip_bfloat16>(),
                                                           x.Ptr<__hip_bfloat16>(), n);
  else if (x.dtype == DType::kF32)
    ReluK<float><<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<float>(), n);
  else
    VT_CHECK(false, "rocm relu dtype");
  Check(hipGetLastError(), "relu");
}

// ---------------- RopeNeox (in-place on qs/ks) -----------------------------
template <typename T, typename Tid>
__global__ void RopeNeoxK(T* qs, T* ks, const Tid* pos, int64_t hq, int64_t hk, int64_t d,
                          int64_t half, int rot, double base, double l3_sf, double l3_lo,
                          double l3_hi, double l3_omax, int64_t n) {
  const int64_t heads = hq + hk;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
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
    double freq = pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
    freq = L3Scale(freq, l3_sf, l3_lo, l3_hi, l3_omax);
    const double angle = static_cast<double>(p) * freq;
    const float c = static_cast<float>(cos(angle));
    const float sn = static_cast<float>(sin(angle));
    const float x = Ld(ptr, off + pair);
    const float y = Ld(ptr, off + pair + half);
    St(ptr, off + pair, x * c - y * sn);
    St(ptr, off + pair + half, x * sn + y * c);
  }
}

void RopeNeoxKernelRocm(Queue& q, Tensor& qs, Tensor& ks, const Tensor& pos,
                        const RopeArgs& args) {
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  const int64_t half = args.rotary_dim / 2;
  const int64_t n = t * (hq + hk) * half;
  if (n == 0) return;
  hipStream_t st = AsStream(q);
  const double base = args.base, sf = args.llama3_scaling_factor, lo = args.llama3_low_freq_factor,
               hi = args.llama3_high_freq_factor, omax = args.llama3_orig_max_position;
  if (qs.dtype == DType::kBF16) {
    if (pos.dtype == DType::kI32)
      RopeNeoxK<__hip_bfloat16, int32_t><<<GridFor(n), kBlock, 0, st>>>(
          qs.Ptr<__hip_bfloat16>(), ks.Ptr<__hip_bfloat16>(), pos.Ptr<int32_t>(), hq, hk, d,
          half, args.rotary_dim, base, sf, lo, hi, omax, n);
    else
      RopeNeoxK<__hip_bfloat16, int64_t><<<GridFor(n), kBlock, 0, st>>>(
          qs.Ptr<__hip_bfloat16>(), ks.Ptr<__hip_bfloat16>(), pos.Ptr<int64_t>(), hq, hk, d,
          half, args.rotary_dim, base, sf, lo, hi, omax, n);
  } else if (qs.dtype == DType::kF32) {
    if (pos.dtype == DType::kI32)
      RopeNeoxK<float, int32_t><<<GridFor(n), kBlock, 0, st>>>(
          qs.Ptr<float>(), ks.Ptr<float>(), pos.Ptr<int32_t>(), hq, hk, d, half, args.rotary_dim,
          base, sf, lo, hi, omax, n);
    else
      RopeNeoxK<float, int64_t><<<GridFor(n), kBlock, 0, st>>>(
          qs.Ptr<float>(), ks.Ptr<float>(), pos.Ptr<int64_t>(), hq, hk, d, half, args.rotary_dim,
          base, sf, lo, hi, omax, n);
  } else
    VT_CHECK(false, "rocm rope_neox dtype");
  Check(hipGetLastError(), "rope_neox");
}

// ---------------- RopeFromCache --------------------------------------------
template <typename T, typename Tid>
__global__ void RopeFromCacheK(T* qs, T* ks, const Tid* positions, const T* cache,
                               int64_t cache_rows, int64_t tokens, int64_t hq, int64_t hk,
                               int64_t q_tok_stride, int64_t q_head_stride, int64_t k_tok_stride,
                               int64_t k_head_stride, int rotary_dim, int64_t half,
                               bool is_neox_style, int64_t n) {
  const int64_t heads = hq + hk;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    const int64_t pair = idx % half;
    const int64_t head = (idx / half) % heads;
    const int64_t token = idx / (half * heads);
    const int64_t position = static_cast<int64_t>(positions[token]);
    if (position < 0 || position >= cache_rows) continue;
    const int64_t cache_offset = position * rotary_dim;
    const float c = Ld(cache, cache_offset + pair);
    const float sn = Ld(cache, cache_offset + half + pair);
    T* states = head < hq ? qs : ks;
    const int64_t local_head = head < hq ? head : head - hq;
    const int64_t row = head < hq ? token * q_tok_stride + local_head * q_head_stride
                                 : token * k_tok_stride + local_head * k_head_stride;
    const int64_t first = is_neox_style ? pair : pair * 2;
    const int64_t second = is_neox_style ? pair + half : pair * 2 + 1;
    const float x = Ld(states, row + first);
    const float y = Ld(states, row + second);
    St(states, row + first, x * c - y * sn);
    St(states, row + second, x * sn + y * c);
  }
}

void RopeFromCacheKernelRocm(Queue& q, Tensor& qs, Tensor* ks, const Tensor& positions,
                             const Tensor& cache, const RopeArgs& args) {
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  const int64_t half = args.rotary_dim / 2;
  const int64_t n = tokens * (hq + hk) * half;
  if (n == 0) return;
  // mrope (positions.rank==2) not needed for Gemma/Phi dense smoke yet
  VT_CHECK(positions.rank == 1, "rocm rope_from_cache: mrope not implemented yet");
  hipStream_t st = AsStream(q);
  auto launch = [&](auto t_tag, auto id_tag) {
    using T = decltype(t_tag);
    using Tid = decltype(id_tag);
    RopeFromCacheK<T, Tid><<<GridFor(n), kBlock, 0, st>>>(
        qs.Ptr<T>(), ks == nullptr ? nullptr : ks->Ptr<T>(), positions.Ptr<Tid>(),
        cache.Ptr<T>(), cache.shape[0], tokens, hq, hk, qs.stride[0], qs.stride[1],
        ks == nullptr ? 0 : ks->stride[0], ks == nullptr ? 0 : ks->stride[1], args.rotary_dim,
        half, args.is_neox_style, n);
  };
  if (qs.dtype == DType::kBF16) {
    if (positions.dtype == DType::kI32)
      launch(__hip_bfloat16{}, int32_t{});
    else
      launch(__hip_bfloat16{}, int64_t{});
  } else if (qs.dtype == DType::kF32) {
    if (positions.dtype == DType::kI32)
      launch(float{}, int32_t{});
    else
      launch(float{}, int64_t{});
  } else
    VT_CHECK(false, "rocm rope_from_cache dtype");
  Check(hipGetLastError(), "rope_from_cache");
}

// ---------------- ReshapeAndCache (NHD paged KV write) ---------------------
template <typename Word>
__global__ void ReshapeAndCacheK(const Word* key, const Word* value, Word* key_cache,
                                 Word* value_cache, const int64_t* slot_mapping,
                                 int64_t block_size, int64_t n_elems, int64_t k_block_stride,
                                 int64_t k_page_stride, int64_t v_block_stride,
                                 int64_t v_page_stride, int64_t k_tok_stride,
                                 int64_t v_tok_stride) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  const int64_t kdst = block * k_block_stride + offset * k_page_stride;
  const int64_t vdst = block * v_block_stride + offset * v_page_stride;
  const int64_t ksrc = token * k_tok_stride;
  const int64_t vsrc = token * v_tok_stride;
  for (int64_t e = threadIdx.x; e < n_elems; e += blockDim.x) {
    key_cache[kdst + e] = key[ksrc + e];
    value_cache[vdst + e] = value[vsrc + e];
  }
}

void ReshapeAndCacheKernelRocm(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                               Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t n_elems = k_cache.shape[2] * k_cache.shape[3];
  if (num_slots == 0 || n_elems == 0) return;
  hipStream_t st = AsStream(q);
  const unsigned grid = static_cast<unsigned>(num_slots);
  const unsigned block = static_cast<unsigned>(n_elems < 512 ? n_elems : 512);
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  switch (SizeOf(k.dtype)) {
    case 4:
      ReshapeAndCacheK<uint32_t><<<grid, block, 0, st>>>(
          k.Ptr<uint32_t>(), v.Ptr<uint32_t>(), k_cache.Ptr<uint32_t>(),
          v_cache.Ptr<uint32_t>(), slots, block_size, n_elems, k_cache.stride[0],
          k_cache.stride[1], v_cache.stride[0], v_cache.stride[1], k.stride[0], v.stride[0]);
      break;
    case 2:
      ReshapeAndCacheK<uint16_t><<<grid, block, 0, st>>>(
          k.Ptr<uint16_t>(), v.Ptr<uint16_t>(), k_cache.Ptr<uint16_t>(),
          v_cache.Ptr<uint16_t>(), slots, block_size, n_elems, k_cache.stride[0],
          k_cache.stride[1], v_cache.stride[0], v_cache.stride[1], k.stride[0], v.stride[0]);
      break;
    default:
      VT_CHECK(false, "rocm reshape_and_cache: unsupported elem size");
  }
  Check(hipGetLastError(), "reshape_and_cache");
}

// ---------------- GeluTanh / GeluErf (elementwise) -------------------------
template <typename T>
__global__ void GeluTanhK(T* out, const T* x, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const float v = Ld(x, i);
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    St(out, i, 0.5f * v * (1.0f + tanhf(inner)));
  }
}
template <typename T>
__global__ void GeluErfK(T* out, const T* x, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const float v = Ld(x, i);
    St(out, i, 0.5f * v * (1.0f + erff(v * 0.7071067811865476f)));
  }
}

void GeluTanhKernelRocm(Queue& q, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  hipStream_t st = AsStream(q);
  if (x.dtype == DType::kBF16)
    GeluTanhK<__hip_bfloat16><<<GridFor(n), kBlock, 0, st>>>(out.Ptr<__hip_bfloat16>(),
                                                               x.Ptr<__hip_bfloat16>(), n);
  else if (x.dtype == DType::kF32)
    GeluTanhK<float><<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<float>(), n);
  else
    VT_CHECK(false, "rocm gelu_tanh dtype");
  Check(hipGetLastError(), "gelu_tanh");
}

void GeluErfKernelRocm(Queue& q, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  hipStream_t st = AsStream(q);
  if (x.dtype == DType::kBF16)
    GeluErfK<__hip_bfloat16><<<GridFor(n), kBlock, 0, st>>>(out.Ptr<__hip_bfloat16>(),
                                                              x.Ptr<__hip_bfloat16>(), n);
  else if (x.dtype == DType::kF32)
    GeluErfK<float><<<GridFor(n), kBlock, 0, st>>>(out.Ptr<float>(), x.Ptr<float>(), n);
  else
    VT_CHECK(false, "rocm gelu_erf dtype");
  Check(hipGetLastError(), "gelu_erf");
}

// --- sampling logit masks (Hermes allowed_token_ids / bad_words) ------------
// mask TRUE => exclude (set -inf). Mirrors CPU/CUDA ApplyAllowedTokenIds.
__global__ void ApplyAllowedTokenIdsK(float* logits, const int8_t* mask, int64_t total) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step)
    if (mask[idx]) logits[idx] = -INFINITY;
}

void ApplyAllowedTokenIdsKernelRocm(Queue& q, Tensor& logits, const Tensor& mask) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  VT_CHECK(logits.dtype == DType::kF32, "rocm apply_allowed_token_ids: logits f32");
  VT_CHECK(mask.dtype == DType::kI8, "rocm apply_allowed_token_ids: mask i8");
  const int64_t total = n * v;
  ApplyAllowedTokenIdsK<<<GridFor(total), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), mask.Ptr<int8_t>(), total);
  Check(hipGetLastError(), "apply_allowed_token_ids");
}

// Sparse -inf scatter at (rows[k], cols[k]) — bad_words path.
__global__ void ApplyTokenMaskK(float* logits, const int32_t* rows, const int32_t* cols,
                                int64_t v, int64_t m) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < m;
       k += step)
    logits[static_cast<int64_t>(rows[k]) * v + cols[k]] = -INFINITY;
}

void ApplyTokenMaskKernelRocm(Queue& q, Tensor& logits, const Tensor& rows,
                              const Tensor& cols) {
  const int64_t v = logits.shape[1], m = rows.shape[0];
  if (m == 0) return;
  VT_CHECK(logits.dtype == DType::kF32, "rocm apply_token_mask: logits f32");
  ApplyTokenMaskK<<<GridFor(m), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), rows.Ptr<int32_t>(), cols.Ptr<int32_t>(), v, m);
  Check(hipGetLastError(), "apply_token_mask");
}

}  // namespace vt::rocm
