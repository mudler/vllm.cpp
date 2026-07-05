// vllm.cpp — Marlin NVFP4 W4A16 grouped-MoE GEMM drop-in (vt::Tensor launcher).
//
// This is the torch-free host launcher for the vendored Marlin MoE kernel
// (src/vt/cuda/marlin/, a 1:1 lift of vLLM's moe/marlin_moe_wna16 @ e24d1b24 —
// marlin-dropin-feasibility.md). It mirrors the NVFP4 branch of vLLM's
// moe_wna16_marlin_gemm (ops.cu:543): b_type=kFE2M1f + s_type=kFE4M3fn,
// group_blocks=1 (group size 16), bf16 activation/output. All the act-order /
// zero-point / bias / 8-bit-activation branches of the torch launcher are
// irrelevant to NVFP4 W4A16 and are dropped; the compute call into
// MARLIN_NAMESPACE_NAME::marlin_mm is the verbatim vendored dispatcher.
//
// Weights MUST be pre-repacked into Marlin's interleaved layout (b_q_weight)
// with processed fp8 block scales (b_scales) + per-expert global scale
// (global_scale) — see the load-time repack (mirror of
// marlin_utils_fp4.prepare_nvfp4_moe_layer_for_marlin). The align inputs
// (sorted_token_ids / expert_ids / num_tokens_past_padded / topk_weights) are
// vLLM's moe_align_block_size outputs.
//
// Isolated TU (heavy templated kernel). Gated by VT_MARLIN_NVFP4.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/scalar_type.hpp"
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_mm.h"

#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: moe_marlin: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// vt::MoeGroupedGemmNvfp4Marlin registered kernel.
void MoeGroupedGemmNvfp4MarlinKernelCuda(Queue& q, Tensor& c, const Tensor& a,
                                         const Tensor& b_q_weight, const Tensor& b_scales,
                                         const Tensor& global_scale, Tensor& workspace,
                                         const Tensor& sorted_token_ids, const Tensor& expert_ids,
                                         const Tensor& num_tokens_past_padded,
                                         const Tensor& topk_weights, const MoeMarlinArgs& args) {
  cudaStream_t s = AsStream(q);
  const int dev = q.device.index;

  // NVFP4 W4A16, bf16 activation/output (marlin_moe_wna16 generate_kernels.py:94).
  const vllm::ScalarType a_type = vllm::kBFloat16;
  const vllm::ScalarType b_type = vllm::kFE2M1f;
  const vllm::ScalarType c_type = vllm::kBFloat16;
  const vllm::ScalarType s_type = vllm::kFE4M3fn;

  const int num_experts = static_cast<int>(b_q_weight.shape[0]);
  const int size_m = args.size_m;
  const int size_n = args.size_n;
  const int size_k = args.size_k;
  const int moe_block_size = args.moe_block_size;
  const int top_k = args.top_k;
  const int group_size = 16;                 // group_blocks == 1
  const int num_groups = size_k / group_size;

  int sms = -1;
  Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev),
        "cudaDeviceGetAttribute(sms)");

  // C_tmp for the fp32 global reduce (use_fp32_reduce && !use_atomic_add). Size
  // per vLLM ops.cu:709 upper bound (size_n * sorted_token_ids) — always >= the
  // capped size the kernel indexes.
  const bool use_atomic_add = false;
  const bool use_fp32_reduce = true;
  const int64_t sorted_len = sorted_token_ids.shape[0];
  float* c_tmp = nullptr;
  int64_t c_tmp_elems = 0;
  if (use_fp32_reduce && !use_atomic_add) {
    c_tmp_elems = static_cast<int64_t>(size_n) * sorted_len;
    if (moe_block_size == 8) c_tmp_elems *= 2;
    Check(cudaMallocAsync(&c_tmp, static_cast<size_t>(c_tmp_elems) * sizeof(float), s),
          "cudaMallocAsync c_tmp");
  }

  MARLIN_NAMESPACE_NAME::marlin_mm(
      a.data, b_q_weight.data, c.data, c_tmp, /*b_bias=*/nullptr, /*a_s=*/nullptr,
      b_scales.data, global_scale.data, /*zp=*/nullptr, /*g_idx=*/nullptr, /*perm=*/nullptr,
      /*a_tmp=*/nullptr, sorted_token_ids.data, expert_ids.data, num_tokens_past_padded.data,
      topk_weights.data, moe_block_size, num_experts, top_k, args.mul_topk_weights, size_m,
      size_n, size_k, workspace.data, a_type, b_type, c_type, s_type, /*has_bias=*/false,
      /*has_act_order=*/false, /*is_k_full=*/true, /*has_zp=*/false, num_groups, group_size, dev,
      s, /*thread_k=*/-1, /*thread_n=*/-1, sms, /*blocks_per_sm=*/0, use_atomic_add,
      use_fp32_reduce, /*is_zp_float=*/false);

  if (c_tmp) Check(cudaFreeAsync(c_tmp, s), "cudaFreeAsync c_tmp");
  Check(cudaGetLastError(), "moe_marlin marlin_mm launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMoeGroupedGemmNvfp4Marlin, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MoeGroupedGemmNvfp4MarlinFn>(&MoeGroupedGemmNvfp4MarlinKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda
