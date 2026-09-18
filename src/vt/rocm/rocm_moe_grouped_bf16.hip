// Native grouped BF16 MoE (#3094, BACKEND-ROCM-BF16-MOE).
// Logical layout donor: cuda_matmul_nvfp4.cu:943-1004 at 6db4bef906859e864c82523c01107473f7dcca29.
// Native numeric contract: vLLM e126687a9a828d513c01a07cd69f025f27d63280,
// fused_moe.py:517-610, experts/triton_moe.py:388-527,
// activation_kernels.cu:44,165-177, moe_align_sum_kernels.cu:395-459.
//
// Each output has one deterministic reduction. No atomics, reusable scratch,
// allocation, synchronization, or temporary device pointers are required.
// The legacy and native gate/up modes share this implementation. Only the
// explicit native mode rounds gate/up and SiLU through BF16 before multiplication.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "vt/ops.h"

namespace vt::rocm {
namespace {
constexpr int kBlock = 256;
constexpr int kReductionTile = 32;

void Check(hipError_t status, const char* operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("vt rocm grouped_bf16: ") + operation + ": " +
                             hipGetErrorString(status));
}

// Queue handles, including nullptr, belong to a device. Never infer that device
// from the calling thread's ambient HIP state.
class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    Check(hipGetDevice(&previous_), "get device");
    changed_ = previous_ != device;
    if (changed_) Check(hipSetDevice(device), "select queue device");
  }
  ~DeviceScope() { if (changed_) (void)hipSetDevice(previous_); }
 private:
  int previous_ = 0;
  bool changed_ = false;
};

__device__ float Load(const __hip_bfloat16* data, int64_t index) {
  return __bfloat162float(data[index]);
}
__device__ float Load(const float* data, int64_t index) { return data[index]; }
__device__ void Store(float* data, int64_t index, float value) { data[index] = value; }
__device__ void Store(__hip_bfloat16* data, int64_t index, float value) {
  data[index] = __float2bfloat16(value);
}
__device__ float Bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

// Fused=false serves ordinary and weighted down GEMMs. Fused=true serves both
// numeric gate/up modes. FP32 accumulators are required by the pinned dot
// contract; legacy gate/up retains FP32 intermediates for caller compatibility.
template <typename Output, bool Fused, bool Native>
__global__ void GroupedBf16(Output* output, const __hip_bfloat16* activation,
                            const int32_t* expert_ids, const int32_t* row_map,
                            const int64_t* gate_ptrs, const int64_t* up_ptrs,
                            const float* route_weights, int64_t pairs, int64_t n,
                            int64_t k) {
  const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= n) return;
  for (int64_t pair = blockIdx.y; pair < pairs; pair += gridDim.y) {
    const int64_t expert = expert_ids[pair];
    const int64_t row = row_map == nullptr ? pair : row_map[pair];
    const auto* gate = reinterpret_cast<const __hip_bfloat16*>(
        static_cast<uintptr_t>(gate_ptrs[expert]));
    const __hip_bfloat16* up = nullptr;
    if constexpr (Fused)
      up = reinterpret_cast<const __hip_bfloat16*>(static_cast<uintptr_t>(up_ptrs[expert]));
    float g = 0.0f;
    float u = 0.0f;
    for (int64_t start = 0; start < k; start += kReductionTile) {
      #pragma unroll
      for (int offset = 0; offset < kReductionTile; ++offset) {
        const int64_t inner = start + offset;
        if (inner < k) {
          const float x = Load(activation, row * k + inner);
          g += x * Load(gate, inner * n + column);
          if constexpr (Fused) u += x * Load(up, inner * n + column);
        }
      }
    }
    if constexpr (Fused) {
      if constexpr (Native) {
        g = Bf16(g);
        u = Bf16(u);
      }
      float silu = g / (1.0f + expf(-g));
      if constexpr (Native) silu = Bf16(silu);
      Store(output, pair * n + column, silu * u);
    } else {
      // fused_moe.py:593-610: multiply in FP32 BEFORE narrowing the down output.
      if (route_weights != nullptr) g *= route_weights[pair];
      Store(output, pair * n + column, g);
    }
  }
}

template <bool Fused, bool Native>
void Launch(Queue& q, Tensor& output, const Tensor& activation, const Tensor& expert_ids,
             const Tensor* row_map, const Tensor& gate_ptrs, const Tensor* up_ptrs,
             const Tensor* route_weights) {
  const int64_t pairs = output.shape[0], n = output.shape[1], k = activation.shape[1];
  if (pairs == 0 || n == 0) return;
  DeviceScope device(q.device.index);
  const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock),
                  static_cast<unsigned>(pairs < 65535 ? pairs : 65535));
  auto launch = [&](auto* out) {
    GroupedBf16<std::remove_pointer_t<decltype(out)>, Fused, Native>
        <<<grid, kBlock, 0, static_cast<hipStream_t>(q.handle)>>>(
            out, activation.Ptr<__hip_bfloat16>(), expert_ids.Ptr<int32_t>(),
            row_map == nullptr ? nullptr : row_map->Ptr<int32_t>(), gate_ptrs.Ptr<int64_t>(),
            up_ptrs == nullptr ? nullptr : up_ptrs->Ptr<int64_t>(),
            route_weights == nullptr ? nullptr : route_weights->Ptr<float>(), pairs, n, k);
  };
  if (output.dtype == DType::kBF16) launch(output.Ptr<__hip_bfloat16>());
  else launch(output.Ptr<float>());
  Check(hipGetLastError(), "grouped kernel launch");
}

// moe_align_sum_kernels.cu:395-459: values already include router weights.
template <typename Output, typename Shared>
__global__ void CombinePreweighted(Output* output, const __hip_bfloat16* experts,
                                   const Shared* shared, int64_t tokens, int64_t top_k,
                                   int64_t hidden, float routed_scale) {
  const int64_t total = tokens * hidden;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += step) {
    const int64_t token = index / hidden, column = index % hidden;
    float sum = 0.0f;
    for (int64_t slot = 0; slot < top_k; ++slot)
      sum += Load(experts, (token * top_k + slot) * hidden + column);
    sum *= routed_scale;
    if (shared != nullptr) sum += Load(shared, index);
    Store(output, index, sum);
  }
}
}  // namespace

void MoeGroupedGemmBf16KernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                  const Tensor& expert_ids, const Tensor* row_map,
                                  const Tensor& weight_ptrs) {
  Launch<false, false>(q, out, act, expert_ids, row_map, weight_ptrs, nullptr, nullptr);
}

void MoeGroupedGemmBf16GateUpSiluKernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                           const Tensor& expert_ids, const Tensor* row_map,
                                           const Tensor& gate_ptrs, const Tensor& up_ptrs) {
  Launch<true, false>(q, out, act, expert_ids, row_map, gate_ptrs, &up_ptrs, nullptr);
}

void MoeGroupedGemmBf16GateUpSiluNativeKernelRocm(
    Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids, const Tensor* row_map,
    const Tensor& gate_ptrs, const Tensor& up_ptrs) {
  Launch<true, true>(q, out, act, expert_ids, row_map, gate_ptrs, &up_ptrs, nullptr);
}

void MoeGroupedGemmBf16WeightedKernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                          const Tensor& expert_ids, const Tensor* row_map,
                                          const Tensor& weight_ptrs, const Tensor& weights) {
  Launch<false, true>(q, out, act, expert_ids, row_map, weight_ptrs, nullptr, &weights);
}

void MoeCombinePreweightedKernelRocm(Queue& q, Tensor& out, const Tensor& experts,
                                      const Tensor* shared, float routed_scale) {
  const int64_t tokens = out.shape[0], hidden = out.shape[1], top_k = experts.shape[1];
  const int64_t total = tokens * hidden;
  if (total == 0) return;
  DeviceScope device(q.device.index);
  const auto blocks = (total + kBlock - 1) / kBlock;
  const unsigned grid = static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
  auto launch = [&](auto* output, auto* shared_data) {
    CombinePreweighted<<<grid, kBlock, 0, static_cast<hipStream_t>(q.handle)>>>(
        output, experts.Ptr<__hip_bfloat16>(), shared_data, tokens, top_k, hidden, routed_scale);
  };
  auto by_shared = [&](auto* output) {
    if (shared != nullptr && shared->dtype == DType::kBF16)
      launch(output, shared->Ptr<__hip_bfloat16>());
    else launch(output, shared == nullptr ? nullptr : shared->Ptr<float>());
  };
  if (out.dtype == DType::kBF16) by_shared(out.Ptr<__hip_bfloat16>());
  else by_shared(out.Ptr<float>());
  Check(hipGetLastError(), "preweighted combine launch");
}
}  // namespace vt::rocm
