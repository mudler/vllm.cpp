// vt lift — declaration of the vendored Marlin MoE dispatcher (marlin_mm),
// defined verbatim in marlin_mm_moe.cu (= vLLM ops.cu:344, torch launcher
// stripped). The vt::Tensor launcher (cuda_moe_marlin.cu) calls this directly.
#pragma once

#include <cuda_runtime.h>

#include "core/scalar_type.hpp"

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

namespace MARLIN_NAMESPACE_NAME {

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, void* sorted_token_ids,
               void* expert_ids, void* num_tokens_past_padded,
               void* topk_weights, int moe_block_size, int num_experts,
               int top_k, bool mul_topk_weights, int prob_m, int prob_n,
               int prob_k, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k,
               int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float);

}  // namespace MARLIN_NAMESPACE_NAME
