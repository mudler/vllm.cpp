// Ported from: vllm/v1/worker/gpu/input_batch.py:304-406,457-543 @ e24d1b24
// (_combine_sampled_and_draft_tokens_kernel + the post_update last_sampled
// scatter). See include/vt/cuda/combine_tokens.h for the contract + how these
// remove GPUModelRunner::sample_tokens_async's pre-scatter Synchronize.
//
// The Triton kernel is a 1-program-per-request grid; these are the CUDA analog
// (one thread per request row). T0 non-spec subset: NUM_NEW_SAMPLED_TOKENS == 1,
// no draft tokens (num_draft_tokens == 0), so the block/mask draft loop
// degenerates to a single last-sampled-token write; the SPEC-MTP draft lane is
// deferred exactly as the host combine (prepare_inputs.cpp) defers it.
//
// NOTE: built + verified on dgx.casa (the CI box is CPU-only). The kernels run
// main-stream-ordered relative to the forward; on GB10 (pageable memory access)
// the pointers are the runner's device-addressable host arrays.
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/cuda/combine_tokens.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda combine_tokens: ") + what +
                             ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) {
  return static_cast<cudaStream_t>(q.handle);
}

// _combine_sampled_and_draft_tokens_kernel (input_batch.py:304-360), input_ids
// splice only (our logits_indices come from prepare_inputs). One thread per
// request row.
__global__ void CombineKernel(int32_t* input_ids, const int32_t* idx_mapping,
                              const int32_t* last_sampled_tokens,
                              const int32_t* query_start_loc,
                              const int32_t* seq_lens,
                              const int32_t* prefill_len, int num_reqs,
                              int num_new_sampled_tokens) {
  const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= num_reqs) return;

  // req_state_idx = idx_mapping[batch_idx] (input_batch.py:322); identity when
  // null (condensed-dense persistent batch: batch row == req_state slot).
  const int req_state_idx =
      idx_mapping != nullptr ? idx_mapping[batch_idx] : batch_idx;

  // num_logits == num_new_sampled_tokens at T0 (num_draft_tokens == 0). The
  // decode input position is query_end - num_logits (input_batch.py:344-346).
  const int num_logits = num_new_sampled_tokens;
  const int query_end = query_start_loc[batch_idx + 1];

  // seq_len <= prefill_len: still consuming known prefill tokens (incl. the
  // chunk that exactly completes prefill) — no sampled/draft token to splice; the
  // prompt token in input_ids stays (input_batch.py:338-341).
  const int seq_len = seq_lens[batch_idx];
  const int pf = prefill_len[req_state_idx];
  if (seq_len <= pf) return;

  // Write the last sampled token id at the decode position
  // (input_batch.py:343-347). num_new_sampled_tokens == 0 (draft-only) writes
  // nothing. Draft tokens (num_draft_tokens > 0) are deferred with SPEC-MTP.
  if (num_new_sampled_tokens > 0) {
    input_ids[query_end - num_logits] = last_sampled_tokens[req_state_idx];
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

}  // namespace

void LaunchCombineSampledAndDraftTokens(Queue& queue, int32_t* input_ids,
                                        const int32_t* idx_mapping,
                                        const int32_t* last_sampled_tokens,
                                        const int32_t* query_start_loc,
                                        const int32_t* seq_lens,
                                        const int32_t* prefill_len, int num_reqs,
                                        int num_new_sampled_tokens) {
  if (num_reqs <= 0) return;
  const int grid = (num_reqs + kBlock - 1) / kBlock;
  CombineKernel<<<grid, kBlock, 0, AsStream(queue)>>>(
      input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
      prefill_len, num_reqs, num_new_sampled_tokens);
  Check(cudaGetLastError(), "CombineKernel launch");
}

void LaunchScatterLastSampled(Queue& queue, int32_t* last_sampled_tokens,
                              const int64_t* sampled_ids,
                              const int32_t* idx_mapping, int num_reqs) {
  if (num_reqs <= 0) return;
  const int grid = (num_reqs + kBlock - 1) / kBlock;
  ScatterLastSampledKernel<<<grid, kBlock, 0, AsStream(queue)>>>(
      last_sampled_tokens, sampled_ids, idx_mapping, num_reqs);
  Check(cudaGetLastError(), "ScatterLastSampledKernel launch");
}

}  // namespace vt::cuda
