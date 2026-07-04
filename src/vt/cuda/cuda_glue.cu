// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA Qwen3.6 elementwise "glue" ops (M0.9 forward): the small reshape/split/
// activation fusions that sit between the big decode ops, moved on-device so the
// whole decode step is CUDA-graph capturable. Correctness-grade — plain
// grid-stride kernels matching the CPU reference math in src/vt/cpu/cpu_ops.cpp
// element for element. All math is f32; dims are inferred from tensor shapes.
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

// f32 load/store overloads: bf16 converts on the way in/out, math is f32
// (mirrors cuda_moe.cu; __float2bfloat16 is round-to-nearest-even, same as the
// host F32ToBF16).
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

__device__ inline float SigmoidF(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// cast_bf16: out[i] = bf16(in[i]). Thread per element, grid-stride.
__global__ void CastBf16Kernel(__nv_bfloat16* out, const float* in, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step)
    Store(out, i, Load(in, i));
}

void CastBf16KernelCuda(Queue& q, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  CastBf16Kernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(out.Ptr<__nv_bfloat16>(), in.Ptr<float>(),
                                                         n);
  Check(cudaGetLastError(), "cast_bf16 launch");
}

// ---------------------------------------------------------------------------
// attn_gate_split: qgate [T, Hq*2*Dh] -> q_out/gate_out [T,Hq,Dh]. Thread per
// output element (flat index over T*Hq*Dh); (i,h,d) recovered from it.
__global__ void AttnGateSplitKernel(float* q_out, float* gate_out, const float* qgate, int64_t t,
                                    int64_t hq, int64_t dh) {
  const int64_t n = t * hq * dh;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t d = idx % dh;
    const int64_t h = (idx / dh) % hq;
    const int64_t i = idx / (dh * hq);
    const int64_t base = i * (hq * 2 * dh) + h * 2 * dh;  // start of (i,h) pair
    Store(q_out, idx, Load(qgate, base + d));
    Store(gate_out, idx, Load(qgate, base + dh + d));
  }
}

void AttnGateSplitKernelCuda(Queue& q, Tensor& q_out, Tensor& gate_out, const Tensor& qgate) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t n = t * hq * dh;
  if (n == 0) return;
  AttnGateSplitKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      q_out.Ptr<float>(), gate_out.Ptr<float>(), qgate.Ptr<float>(), t, hq, dh);
  Check(cudaGetLastError(), "attn_gate_split launch");
}

// ---------------------------------------------------------------------------
// sigmoid_gate_bf16: out[i] = bf16(attn[i] * sigmoid(gate[i])).
__global__ void SigmoidGateBf16Kernel(__nv_bfloat16* out, const float* attn, const float* gate,
                                      int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step)
    Store(out, i, Load(attn, i) * SigmoidF(Load(gate, i)));
}

void SigmoidGateBf16KernelCuda(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  SigmoidGateBf16Kernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      out.Ptr<__nv_bfloat16>(), attn.Ptr<float>(), gate.Ptr<float>(), n);
  Check(cudaGetLastError(), "sigmoid_gate_bf16 launch");
}

// ---------------------------------------------------------------------------
// gdn_g_beta (gdn-semantics.md §6): g/beta from raw a/b/A_log/dt_bias. Thread
// per output element (flat idx over T*Hv); hv = idx % Hv indexes a_log/dt_bias.
__global__ void GdnGBetaKernel(float* g_out, float* beta_out, const float* araw, const float* braw,
                               const float* a_log, const float* dt_bias, int64_t t, int64_t hv) {
  const int64_t n = t * hv;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t h = idx % hv;
    const float x = araw[idx] + dt_bias[h];
    const float sp = x > 20.0f ? x : log1pf(expf(x));  // softplus, threshold 20
    g_out[idx] = -expf(a_log[h]) * sp;
    beta_out[idx] = SigmoidF(braw[idx]);
  }
}

void GdnGBetaKernelCuda(Queue& q, Tensor& g_out, Tensor& beta_out, const Tensor& araw,
                        const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias) {
  const int64_t t = g_out.shape[0], hv = g_out.shape[1];
  const int64_t n = t * hv;
  if (n == 0) return;
  GdnGBetaKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      g_out.Ptr<float>(), beta_out.Ptr<float>(), araw.Ptr<float>(), braw.Ptr<float>(),
      a_log.Ptr<float>(), dt_bias.Ptr<float>(), t, hv);
  Check(cudaGetLastError(), "gdn_g_beta launch");
}

// ---------------------------------------------------------------------------
// gdn_conv_split: conv [T, 2*key_dim+value_dim] -> q/k [T,key_dim], v
// [T,value_dim]. Thread per q/k output element (flat idx over T*key_dim); the v
// copy is folded in for idx < T*value_dim so both halves ride one launch.
__global__ void GdnConvSplitKernel(float* q_out, float* k_out, float* v_out, const float* conv,
                                   int64_t t, int64_t key_dim, int64_t value_dim) {
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t nq = t * key_dim, nv = t * value_dim;
  const int64_t n = nq > nv ? nq : nv;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    if (idx < nq) {
      const int64_t i = idx / key_dim, j = idx % key_dim;
      const int64_t row = i * conv_dim;
      q_out[idx] = conv[row + j];
      k_out[idx] = conv[row + key_dim + j];
    }
    if (idx < nv) {
      const int64_t i = idx / value_dim, j = idx % value_dim;
      v_out[idx] = conv[i * conv_dim + 2 * key_dim + j];
    }
  }
}

void GdnConvSplitKernelCuda(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                            const Tensor& conv) {
  const int64_t t = conv.shape[0];
  if (t == 0) return;
  const int64_t key_dim = q_out.Numel() / t, value_dim = v_out.Numel() / t;
  const int64_t n = t * (key_dim > value_dim ? key_dim : value_dim);
  if (n == 0) return;
  GdnConvSplitKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      q_out.Ptr<float>(), k_out.Ptr<float>(), v_out.Ptr<float>(), conv.Ptr<float>(), t, key_dim,
      value_dim);
  Check(cudaGetLastError(), "gdn_conv_split launch");
}

// ---------------------------------------------------------------------------
// shared_expert_gate: out[t,c] = bf16(sigmoid(gl[t]) * sd[t*H+c]). Thread per
// output element (flat idx over T*H); the token index t = idx / H picks gl[t].
__global__ void SharedExpertGateKernel(__nv_bfloat16* out, const float* sd, const float* gl,
                                       int64_t t, int64_t h) {
  const int64_t n = t * h;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t row = idx / h;
    Store(out, idx, SigmoidF(gl[row]) * sd[idx]);
  }
}

void SharedExpertGateKernelCuda(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl) {
  const int64_t t = out.shape[0], h = out.shape[1];
  const int64_t n = t * h;
  if (n == 0) return;
  SharedExpertGateKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(out.Ptr<__nv_bfloat16>(),
                                                                 sd.Ptr<float>(), gl.Ptr<float>(),
                                                                 t, h);
  Check(cudaGetLastError(), "shared_expert_gate launch");
}

// Registers the CUDA glue kernels during static init (pre-main, like the other
// vt CUDA ops). Harmless on GPU-less machines: the kCUDA backend never
// registers there, so no CUDA queue can dispatch.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kCastBf16, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16KernelCuda)));
    RegisterOp(OpId::kAttnGateSplit, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttnGateSplitFn>(&AttnGateSplitKernelCuda)));
    RegisterOp(
        OpId::kSigmoidGateBf16, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16KernelCuda)));
    RegisterOp(OpId::kGdnGBeta, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnGBetaFn>(&GdnGBetaKernelCuda)));
    RegisterOp(OpId::kGdnConvSplit, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnConvSplitFn>(&GdnConvSplitKernelCuda)));
    RegisterOp(
        OpId::kSharedExpertGate, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<SharedExpertGateFn>(&SharedExpertGateKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
