// ROCm/HIP port of src/vt/cuda/cuda_combine_tokens.cu — device combine/scatter
// kernels for async-scheduling overlap. Same logic, HIP API. These replace the
// host scatter + its pre-sync so the sampled ids never round-trip the host.
//
// The kernels are main-stream-ordered relative to the forward; on a discrete
// ROCm GPU the pointers are device allocations (the AsyncDeviceInputs mirror).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/rocm/combine_tokens.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm combine_tokens: ") + what +
                             ": " + hipGetErrorString(err));
  }
}

hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}

// _combine_sampled_and_draft_tokens_kernel (input_batch.py:304-360), input_ids
// splice only (our logits_indices come from prepare_inputs). One thread per
// request row. Line-for-line the CUDA CombineKernel (cuda_combine_tokens.cu),
// full contract: num_logits derives from cu_num_logits (null == arange == ONE
// per row — not num_new_sampled_tokens; the two part at 0), and the
// num_logits - num_new_sampled_tokens draft tokens splice from the draft
// buffer. The trap replaces the host's two VT_CHECKs (stride narrower than the
// row count; null draft buffer while drafts are due) exactly as the CUDA arm's
// __trap() does — __builtin_trap() because HIP exposes no __trap() alias in
// every header set this file compiles under.
__global__ void CombineKernel(int32_t* input_ids, const int32_t* idx_mapping,
                              const int32_t* last_sampled_tokens,
                              const int32_t* query_start_loc,
                              const int32_t* seq_lens,
                              const int32_t* prefill_len,
                              const int32_t* draft_tokens,
                              int draft_tokens_stride,
                              const int32_t* cu_num_logits, int num_reqs,
                              int num_new_sampled_tokens) {
  const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= num_reqs) return;

  const int req_state_idx =
      idx_mapping != nullptr ? idx_mapping[batch_idx] : batch_idx;

  const int num_logits =
      cu_num_logits != nullptr
          ? cu_num_logits[batch_idx + 1] - cu_num_logits[batch_idx]
          : 1;
  const int num_draft_tokens = num_logits - num_new_sampled_tokens;

  const int query_end = query_start_loc[batch_idx + 1];
  const int logits_start = query_end - num_logits;

  const int seq_len = seq_lens[batch_idx];
  const int pf = prefill_len[req_state_idx];
  if (seq_len <= pf) return;

  const int first_logit_seq_pos = seq_len - num_logits;
  if (num_new_sampled_tokens > 0 && first_logit_seq_pos >= pf) {
    input_ids[logits_start] = last_sampled_tokens[req_state_idx];
  }

  if (num_draft_tokens > 0) {
    if (draft_tokens == nullptr || draft_tokens_stride < num_draft_tokens) {
      __builtin_trap();
    }
    for (int b = 0; b < num_draft_tokens; ++b) {
      input_ids[query_end - num_draft_tokens + b] =
          draft_tokens[req_state_idx * draft_tokens_stride + b];
    }
  }
}

// post_update last_sampled scatter (input_batch.py:457-543 / states.py): one
// thread per request row writes the freshly sampled id into last_sampled_tokens.
__global__ void ScatterLastSampledKernel(int32_t* last_sampled_tokens,
                                         const int64_t* sampled_ids,
                                         const int32_t* idx_mapping,
                                         int num_reqs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_reqs) return;
  const int req_state_idx = idx_mapping != nullptr ? idx_mapping[i] : i;
  last_sampled_tokens[req_state_idx] = static_cast<int32_t>(sampled_ids[i]);
}

// W4 structural replay (see combine_tokens.h). ONE thread, strictly in order:
// the ops are dependent (a condense move can read a slot an earlier move wrote)
// and there are a handful per step at most.
__global__ void ApplyLastSampledOpsKernel(int32_t* last_sampled_tokens,
                                          const int32_t* ops, int num_ops) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int i = 0; i < num_ops; ++i) {
    const int32_t kind = ops[4 * i + 0];
    const int32_t a = ops[4 * i + 1];
    const int32_t b = ops[4 * i + 2];
    const int32_t value = ops[4 * i + 3];
    if (kind == 0) {
      last_sampled_tokens[a] = value;
    } else if (kind == 1) {
      last_sampled_tokens[a] = last_sampled_tokens[b];
    } else if (kind == 2) {
      const int32_t tmp = last_sampled_tokens[a];
      last_sampled_tokens[a] = last_sampled_tokens[b];
      last_sampled_tokens[b] = tmp;
    }
  }
}

}  // namespace

void LaunchCombineSampledAndDraftTokens(
    Queue& queue, int32_t* input_ids, const int32_t* idx_mapping,
    const int32_t* last_sampled_tokens, const int32_t* query_start_loc,
    const int32_t* seq_lens, const int32_t* prefill_len,
    const int32_t* draft_tokens, int draft_tokens_stride,
    const int32_t* cu_num_logits, int num_reqs, int num_new_sampled_tokens) {
  if (num_reqs <= 0) return;
  const int grid = (num_reqs + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(CombineKernel, dim3(grid), dim3(kBlock), 0, AsStream(queue),
                     input_ids, idx_mapping, last_sampled_tokens,
                     query_start_loc, seq_lens, prefill_len, draft_tokens,
                     draft_tokens_stride, cu_num_logits, num_reqs,
                     num_new_sampled_tokens);
  Check(hipGetLastError(), "CombineKernel launch");
}

void LaunchScatterLastSampled(Queue& queue, int32_t* last_sampled_tokens,
                              const int64_t* sampled_ids,
                              const int32_t* idx_mapping, int num_reqs) {
  if (num_reqs <= 0) return;
  const int grid = (num_reqs + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(ScatterLastSampledKernel, dim3(grid), dim3(kBlock), 0,
                     AsStream(queue), last_sampled_tokens, sampled_ids,
                     idx_mapping, num_reqs);
  Check(hipGetLastError(), "ScatterLastSampledKernel launch");
}

void LaunchApplyLastSampledOps(Queue& queue, int32_t* last_sampled_tokens,
                               const int32_t* ops, int num_ops) {
  if (num_ops <= 0) return;
  hipLaunchKernelGGL(ApplyLastSampledOpsKernel, dim3(1), dim3(1), 0,
                     AsStream(queue), last_sampled_tokens, ops, num_ops);
  Check(hipGetLastError(), "ApplyLastSampledOpsKernel launch");
}

}  // namespace vt::rocm
