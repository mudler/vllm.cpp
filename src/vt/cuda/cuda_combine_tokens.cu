// Ported from: vllm/v1/worker/gpu/input_batch.py:303-406,457-543 @ the parity
// pin 5559679229bc961848b121ccdeaa8fa5d79bec98
// (_combine_sampled_and_draft_tokens_kernel + the post_update last_sampled
// scatter). See include/vt/cuda/combine_tokens.h for the contract + how these
// remove GPUModelRunner::sample_tokens_async's pre-scatter Synchronize.
//
// The Triton kernel is a 1-program-per-request grid; these are the CUDA analog
// (one thread per request row). SPEC-DFLASH2 A2-1 made the combine draft-aware
// on both sides: num_logits comes from cu_num_logits, num_draft_tokens is
// num_logits - NUM_NEW_SAMPLED_TOKENS, and the drafts are scattered over the
// tail of the query. The Triton block/mask over BLOCK_SIZE = next_pow2(
// num_speculative_steps + num_new_sampled_tokens) is a serial loop over
// num_draft_tokens here, which is the same set of stores: k is a handful, one
// thread already owns the row, and the mask is exactly `block < num_draft_tokens`.
//
// NOTE: the W3 body was built + verified on dgx.casa (the CI box is CPU-only).
// The SPEC-DFLASH2 A2-1 edit in CombineKernel below is NOT built: no nvcc on
// the box that wave ran on, and the row's spec records the arm under `## Owed`.
// The kernels run main-stream-ordered relative to the forward; on GB10
// (pageable memory access) the pointers are the runner's device-addressable
// host arrays.
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

// _combine_sampled_and_draft_tokens_kernel (input_batch.py:303-361), the
// input_ids stores only (our logits_indices come from prepare_inputs). One
// thread per request row, mirroring the host loop in
// src/vllm/v1/worker/gpu/prepare_inputs.cpp line for line.
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

  // req_state_idx = idx_mapping[batch_idx] (input_batch.py:319); identity when
  // null (condensed-dense persistent batch: batch row == req_state slot).
  const int req_state_idx =
      idx_mapping != nullptr ? idx_mapping[batch_idx] : batch_idx;

  // num_logits and num_draft_tokens (input_batch.py:321-325). A null
  // cu_num_logits stands for arange(num_reqs + 1), so num_logits is ONE — which
  // is what the host computes from a real arange, and is NOT the same thing as
  // num_new_sampled_tokens. The two coincide at 1 and part at 0: a real arange
  // still yields num_logits == 1 there, hence num_draft_tokens == 1, which the
  // host refuses and this kernel now refuses with it. Reading
  // num_new_sampled_tokens instead yielded 0 and wrote nothing, which is a
  // silent disagreement with the host over the same inputs.
  const int num_logits =
      cu_num_logits != nullptr
          ? cu_num_logits[batch_idx + 1] - cu_num_logits[batch_idx]
          : 1;
  const int num_draft_tokens = num_logits - num_new_sampled_tokens;

  const int query_end = query_start_loc[batch_idx + 1];
  const int logits_start = query_end - num_logits;

  // seq_len <= prefill_len: still consuming known prefill tokens (incl. the
  // chunk that exactly completes prefill) — no sampled/draft token to splice; the
  // prompt token in input_ids stays (input_batch.py:337-341).
  const int seq_len = seq_lens[batch_idx];
  const int pf = prefill_len[req_state_idx];
  if (seq_len <= pf) return;

  // Keep prompt-tail slots intact; only rewrite generated-token slots
  // (input_batch.py:343-348). num_new_sampled_tokens == 0 (draft-only) writes
  // nothing.
  const int first_logit_seq_pos = seq_len - num_logits;
  if (num_new_sampled_tokens > 0 && first_logit_seq_pos >= pf) {
    input_ids[logits_start] = last_sampled_tokens[req_state_idx];
  }

  // Write the draft tokens (if any) to input_ids (input_batch.py:350-361). The
  // count comes from num_draft_tokens, never from draft_tokens_stride, which is
  // the speculator's max draft length and pads every shorter row.
  if (num_draft_tokens > 0) {
    // REFUSE, on the same condition the host refuses on. The host throws two
    // VT_CHECKs here (prepare_inputs.cpp: a stride narrower than this row's
    // count, and a draft buffer that holds no row for this req_state — an empty
    // buffer, which is what a null pointer is on this side). A kernel cannot
    // throw, so it traps: __trap() is the only device-abort primitive this
    // tree has. It is NOT an established precedent for this use, and the
    // claim is worth stating precisely. src/vt/cuda/ held 19 __trap() call
    // sites when A2-1 landed, and the other 18 were all the same thing — the
    // `#else` arm of an `#if __CUDA_ARCH__ >= 800` guard, saying "this path was
    // not compiled for this arch and the host gate never launches it"
    // (cuda_ops.cu:2740 and cuda_paged_attn.cu:1012 are both of that kind).
    // This is the tree's FIRST data-dependent runtime contract trap: it fires
    // on the arguments, not on the compile target, so unlike all 18 it is
    // reachable on hardware the kernel was built for.
    //
    // Continuing silently is the failure this replaces. The guard used to read
    // `num_draft_tokens > 0 && draft_tokens != nullptr`, so a step that wired
    // cu_num_logits before draft_tokens — the exact staging A2-3 passes
    // through, and the exact shape of the `/*draft_tokens=*/nullptr` beside
    // `/*cu_num_logits=*/nullptr` at both present call sites — threw on the
    // host while the device wrote the committed token, skipped the drafts and
    // scored STALE draft slots. No token gate can see that: speculative
    // decoding is lossless, so a wrong draft costs acceptance and nothing else
    // (.agents/specs/dflash2-async-spec-sampler.md, reason A).
    if (draft_tokens == nullptr || draft_tokens_stride < num_draft_tokens) {
      __trap();
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
  CombineKernel<<<grid, kBlock, 0, AsStream(queue)>>>(
      input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
      prefill_len, draft_tokens, draft_tokens_stride, cu_num_logits, num_reqs,
      num_new_sampled_tokens);
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

void LaunchApplyLastSampledOps(Queue& queue, int32_t* last_sampled_tokens,
                               const int32_t* ops, int num_ops) {
  if (num_ops <= 0) return;
  ApplyLastSampledOpsKernel<<<1, 1, 0, AsStream(queue)>>>(last_sampled_tokens, ops,
                                                          num_ops);
  Check(cudaGetLastError(), "ApplyLastSampledOpsKernel launch");
}

}  // namespace vt::cuda
