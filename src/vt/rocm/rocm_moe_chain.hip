// ROCm MoE combine/gate ops (BACKEND-ROCM; the model-path MoE chain, #41).
// Hand-translations from src/vt/cuda/cuda_moe.cu (MoeCombineKernel :473,
// MoeCombineGateKernel :555) and the SharedExpertGate CPU oracle
// (cpu_ops.cpp:2387), readable side by side against the donors. All f32 math;
// bf16/f32 dtype arms via the Ld/St boundary conversions.
//
//   SharedExpertGate: out[t,c] = sigmoid(gl[t]) * sd[t,c]
//   MoeCombine:       out[r,c] = sum_j w[r,j]*expert_out[(r*k+j),c] (+ shared[r,c])
//   MoeCombineGate:   MoeCombine + the shared-expert sigmoid gate folded in
//                     (rounded through bf16 exactly as the donor).

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm moe_chain: ") + what + ": " +
                             hipGetErrorString(err));
}
inline hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }
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
__device__ inline float SigmoidF(float x) { return 1.0f / (1.0f + expf(-x)); }

// SharedExpertGate (cpu_ops.cpp:2387): out[t,c] = sigmoid(gl[t]) * sd[t,c].
// out bf16 [T,H]; sd f32 [T,H]; gl f32 [T].
template <typename Tsd, typename Tout>
__global__ void SharedExpertGateK(Tout* out, const Tsd* sd, const float* gl, int64_t t,
                                  int64_t h) {
  const int64_t n = t * h;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += step) {
    const int64_t row = idx / h;
    St(out, idx, SigmoidF(gl[row]) * Ld(sd, idx));
  }
}

// MoeCombine (cuda_moe.cu:473).
template <typename Teo, typename Tsh, typename Tout>
__global__ void MoeCombineK(Tout* out, const Teo* expert_out, const float* weights,
                            const Tsh* shared, int64_t t, int64_t h, int k,
                            float routed_scale) {
  const int64_t n = t * h;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t row = idx / h;
    const int64_t col = idx % h;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j)
      acc += weights[row * k + j] * Ld(expert_out, (row * k + j) * h + col);
    // routed_scale multiplies the ROUTED sum only, BEFORE the shared term is
    // added (upstream apply_routed_scale_to_output: fused_output *= factor,
    // shared untouched) — one standalone f32 multiply on the finished
    // accumulator, bit-identical to the CPU reference under -ffp-contract=off.
    acc *= routed_scale;
    if (shared != nullptr) acc += Ld(shared, idx);
    St(out, idx, acc);
  }
}

// MoeCombineGate (cuda_moe.cu:555): MoeCombine + shared-expert sigmoid gate
// folded in, the shared term rounded through bf16 exactly as the donor.
template <typename Teo, typename Tsd, typename Tout>
__global__ void MoeCombineGateK(Tout* out, const Teo* expert_out, const float* weights,
                                const Tsd* sd, const float* gl, int64_t t, int64_t h,
                                int k) {
  const int64_t n = t * h;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t row = idx / h;
    const int64_t col = idx % h;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j)
      acc += weights[row * k + j] * Ld(expert_out, (row * k + j) * h + col);
    const float sv = SigmoidF(gl[row]) * Ld(sd, idx);
    acc += __bfloat162float(__float2bfloat16(sv));
    St(out, idx, acc);
  }
}

}  // namespace

void SharedExpertGateKernelRocm(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl) {
  // Same f16 hazard as the combine ops (review sweep on #509): the seam admits
  // any float dtype and Tensor::Ptr<T>() is unchecked.
  VT_CHECK(sd.dtype == DType::kF32 || sd.dtype == DType::kBF16,
           "rocm shared_expert_gate: unsupported sd dtype (f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "rocm shared_expert_gate: unsupported out dtype (f32/bf16 only)");
  const int64_t t = out.shape[0], h = out.shape[1];
  if (t == 0 || h == 0) return;
  hipStream_t s = AsStream(q);
  const int64_t n = t * h;
  const bool obf = out.dtype == DType::kBF16, sbf = sd.dtype == DType::kBF16;
  if (obf && sbf)
    SharedExpertGateK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__hip_bfloat16>(),
                                                    sd.Ptr<__hip_bfloat16>(),
                                                    gl.Ptr<float>(), t, h);
  else if (obf)
    SharedExpertGateK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__hip_bfloat16>(), sd.Ptr<float>(),
                                                    gl.Ptr<float>(), t, h);
  else if (sbf)
    SharedExpertGateK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), sd.Ptr<__hip_bfloat16>(),
                                                    gl.Ptr<float>(), t, h);
  else
    SharedExpertGateK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), sd.Ptr<float>(),
                                                    gl.Ptr<float>(), t, h);
  Check(hipGetLastError(), "shared_expert_gate launch");
}

void MoeCombineKernelRocm(Queue& q, Tensor& out, const Tensor& expert_out,
                          const Tensor& weights, const Tensor* shared, float routed_scale) {
  // The donor's dtype refusals (cuda_moe.cu:520-524) — f16 is admitted by the
  // seam's IsFloat gate, and Tensor::Ptr<T>() is an unchecked cast, so without
  // these an f16 expert_out would be read at 4 bytes/element from a 2-byte
  // allocation (review sweep on #509).
  VT_CHECK(expert_out.dtype == DType::kF32 || expert_out.dtype == DType::kBF16,
           "rocm moe_combine: unsupported expert_out dtype (f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "rocm moe_combine: unsupported out dtype (f32/bf16 only)");
  VT_CHECK(shared == nullptr || shared->dtype == DType::kF32 ||
               shared->dtype == DType::kBF16,
           "rocm moe_combine: unsupported shared dtype (f32/bf16 only)");
  const int64_t t = out.shape[0], h = out.shape[1];
  const int k = static_cast<int>(weights.shape[1]);
  const int64_t n = t * h;
  if (n == 0) return;
  hipStream_t s = AsStream(q);
  auto launch = [&](auto eo, auto sh, auto ot) {
    using Teo = decltype(eo); using Tsh = decltype(sh); using Tout = decltype(ot);
    MoeCombineK<Teo, Tsh, Tout><<<GridFor(n), kBlock, 0, s>>>(
        out.Ptr<Tout>(), expert_out.Ptr<Teo>(), weights.Ptr<float>(),
        shared != nullptr ? shared->Ptr<Tsh>() : nullptr, t, h, k, routed_scale);
  };
  auto by_shared = [&](auto eo, auto ot) {
    if (shared != nullptr && shared->dtype == DType::kBF16) launch(eo, __hip_bfloat16{}, ot);
    else launch(eo, float{}, ot);
  };
  auto by_out = [&](auto eo) {
    if (out.dtype == DType::kBF16) by_shared(eo, __hip_bfloat16{});
    else by_shared(eo, float{});
  };
  if (expert_out.dtype == DType::kBF16) by_out(__hip_bfloat16{});
  else by_out(float{});
  Check(hipGetLastError(), "moe_combine launch");
}

void MoeCombineGateKernelRocm(Queue& q, Tensor& out, const Tensor& expert_out,
                              const Tensor& weights, const Tensor& sd, const Tensor& gl) {
  // Same donor refusals as MoeCombine (cuda_moe.cu:597-604).
  VT_CHECK(expert_out.dtype == DType::kF32 || expert_out.dtype == DType::kBF16,
           "rocm moe_combine_gate: unsupported expert_out dtype (f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "rocm moe_combine_gate: unsupported out dtype (f32/bf16 only)");
  VT_CHECK(sd.dtype == DType::kF32 || sd.dtype == DType::kBF16,
           "rocm moe_combine_gate: unsupported sd dtype (f32/bf16 only)");
  const int64_t t = out.shape[0], h = out.shape[1];
  const int k = static_cast<int>(weights.shape[1]);
  const int64_t n = t * h;
  if (n == 0) return;
  hipStream_t s = AsStream(q);
  auto launch = [&](auto eo, auto sd_t, auto ot) {
    using Teo = decltype(eo); using Tsd = decltype(sd_t); using Tout = decltype(ot);
    MoeCombineGateK<Teo, Tsd, Tout><<<GridFor(n), kBlock, 0, s>>>(
        out.Ptr<Tout>(), expert_out.Ptr<Teo>(), weights.Ptr<float>(), sd.Ptr<Tsd>(),
        gl.Ptr<float>(), t, h, k);
  };
  auto by_sd = [&](auto eo, auto ot) {
    if (sd.dtype == DType::kBF16) launch(eo, __hip_bfloat16{}, ot);
    else launch(eo, float{}, ot);
  };
  auto by_out = [&](auto eo) {
    if (out.dtype == DType::kBF16) by_sd(eo, __hip_bfloat16{});
    else by_sd(eo, float{});
  };
  if (expert_out.dtype == DType::kBF16) by_out(__hip_bfloat16{});
  else by_out(float{});
  Check(hipGetLastError(), "moe_combine_gate launch");
}

}  // namespace vt::rocm
