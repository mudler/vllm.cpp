// Device combine/scatter kernels for async-scheduling overlap — ROCm/HIP port
// of include/vt/cuda/combine_tokens.h. Same contract: these replace the host
// scatter + its pre-sync (GPUModelRunner::sample_tokens_async's Synchronize
// before the host loop). The scatter writes last_sampled on the MAIN queue and
// the combine reads it on the MAIN queue, both stream-ordered relative to the
// forward, so no host round-trip of the sampled ids.
//
// Declarations only — the definitions live in src/vt/rocm/rocm_combine_tokens.hip.
// Signatures use plain pointers + vt::Queue so the header stays host-compilable.
#ifndef VT_ROCM_COMBINE_TOKENS_H_
#define VT_ROCM_COMBINE_TOKENS_H_

#include <cstdint>

#include "vt/backend.h"  // vt::Queue

namespace vt::rocm {

// combine_sampled_and_draft_tokens (input_batch.py:304-406). Same FULL contract
// as the CUDA CombineKernel (vt/cuda/combine_tokens.h): num_logits comes from
// cu_num_logits (null == arange, i.e. ONE per row — NOT num_new_sampled_tokens;
// the two part at 0), and num_logits - num_new_sampled_tokens draft tokens are
// spliced from draft_tokens[req_state * draft_tokens_stride + b]. The current
// runner only ever reaches the subset draft_tokens == nullptr &&
// cu_num_logits == nullptr && num_new_sampled_tokens == 1 (T0 non-spec), but
// the signatures must stay IDENTICAL across backends — the shared dispatcher
// (runner.cpp DispatchCombineSampledAndDraftTokens) forwards the same argument
// list to both, and the CUDA arm asserts the draft-bearing staging (A2-3) with
// a device trap rather than a silent skip. For each request row b, if the row
// is a decode row (seq_lens[b] > prefill_len[req_state]) splice the last
// sampled token into input_ids at the decode position (query_start_loc[b+1] -
// num_logits). Prefill/chunked-prefill rows (seq_len <= prefill_len) keep their
// prompt token. idx_mapping is the batch-row -> req_state indirection (the
// abort/finish reorder); pass nullptr for the identity mapping (our persistent
// batch is condensed dense, so batch row == req_state slot). Our runner builds
// logits_indices in prepare_inputs, so this kernel writes only the input_ids
// splice (the upstream kernel's logits_indices store is not needed here).
// Launched on the MAIN queue BEFORE the forward (outside any decode-graph
// capture — input prep always precedes the graph replay).
void LaunchCombineSampledAndDraftTokens(
    Queue& queue, int32_t* input_ids, const int32_t* idx_mapping,
    const int32_t* last_sampled_tokens, const int32_t* query_start_loc,
    const int32_t* seq_lens, const int32_t* prefill_len,
    const int32_t* draft_tokens, int draft_tokens_stride,
    const int32_t* cu_num_logits, int num_reqs, int num_new_sampled_tokens);

// post_update last_sampled scatter (input_batch.py:457-543 / states.py): record
// each row's freshly sampled id into last_sampled_tokens[req_state] on the MAIN
// queue, so the NEXT step's combine reads it without a sampled-id host
// round-trip. sampled_ids is the device-resident [num_reqs] argmax buffer the
// async sampler wrote (int64). idx_mapping is the batch-row -> req_state
// indirection (nullptr == identity). Replaces the runner's host scatter loop and
// its preceding Synchronize.
void LaunchScatterLastSampled(Queue& queue, int32_t* last_sampled_tokens,
                              const int64_t* sampled_ids,
                              const int32_t* idx_mapping, int num_reqs);

// W4 (discrete GPU): replay InputBatch's STRUCTURAL edits to last_sampled_tokens
// onto the device mirror, in stream order.
//
// `ops` is a flat [4 * num_ops] int32 device array of (kind, a, b, value):
//   kind 0 SEED: last_sampled[a] = value      (add_request)
//   kind 1 MOVE: last_sampled[a] = last_sampled[b]  (condense)
//   kind 2 SWAP: swap(last_sampled[a], last_sampled[b])  (swap_states)
// Applied STRICTLY IN ORDER by a single thread: the ops are not independent (a
// move can read a slot a previous move wrote), and there are at most a handful
// per step, so serial application is both correct and free.
void LaunchApplyLastSampledOps(Queue& queue, int32_t* last_sampled_tokens,
                               const int32_t* ops, int num_ops);

}  // namespace vt::rocm

#endif  // VT_ROCM_COMBINE_TOKENS_H_
