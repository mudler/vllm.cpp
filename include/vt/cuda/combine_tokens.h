// Device combine/scatter kernels for async-scheduling overlap (W3 ENG-ASYNC-SCHED
// runner leaf). Ports vllm/v1/worker/gpu/input_batch.py:304-406 @ e24d1b24
// (_combine_sampled_and_draft_tokens_kernel + the post_update last_sampled
// scatter) to CUDA. These replace the host scatter + its pre-sync
// (GPUModelRunner::sample_tokens_async's Synchronize before the host loop): the
// scatter writes last_sampled on the MAIN queue and the combine reads it on the
// MAIN queue, both stream-ordered relative to the forward, so no host round-trip
// of the sampled ids — that Synchronize is exactly what these kernels remove.
//
// On GB10 (cudaDevAttrPageableMemoryAccess) the runner's host arrays are
// device-addressable, so the kernels operate on them in place (last_sampled /
// input_ids / query_start_loc / seq_lens / prefill_len are the same std::vector
// buffers the host path uses); the writes are visible to the same-stream forward
// that embeds input_ids. The CPU backend keeps the host loop (prepare_inputs.cpp
// combine + the runner host scatter); this header is included and its launchers
// are called ONLY under VLLM_CPP_CUDA.
//
// Declarations only — the definitions live in src/vt/cuda/cuda_combine_tokens.cu
// (built + verified on dgx.casa; the CI box is CPU-only). Signatures use plain
// pointers + vt::Queue so the header stays host-compilable.
#ifndef VT_CUDA_COMBINE_TOKENS_H_
#define VT_CUDA_COMBINE_TOKENS_H_

#include <cstdint>

#include "vt/backend.h"  // vt::Queue

namespace vt::cuda {

// combine_sampled_and_draft_tokens (input_batch.py:304-406, T0 non-spec subset:
// NUM_NEW_SAMPLED_TOKENS == 1, no draft tokens). For each request row b, if the
// row is a decode row (seq_lens[b] > prefill_len[req_state]) splice the last
// sampled token into input_ids at the decode position (query_start_loc[b+1] -
// num_new_sampled_tokens). Prefill/chunked-prefill rows (seq_len <= prefill_len)
// keep their prompt token. idx_mapping is the batch-row -> req_state indirection
// (the abort/finish reorder); pass nullptr for the identity mapping (our
// persistent batch is condensed dense, so batch row == req_state slot). Our
// runner builds logits_indices in prepare_inputs, so this kernel writes only the
// input_ids splice (the upstream kernel's logits_indices store is not needed
// here). Launched on the MAIN queue BEFORE the forward (outside any decode-graph
// capture — input prep always precedes the graph replay).
void LaunchCombineSampledAndDraftTokens(Queue& queue, int32_t* input_ids,
                                        const int32_t* idx_mapping,
                                        const int32_t* last_sampled_tokens,
                                        const int32_t* query_start_loc,
                                        const int32_t* seq_lens,
                                        const int32_t* prefill_len, int num_reqs,
                                        int num_new_sampled_tokens);

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

}  // namespace vt::cuda

#endif  // VT_CUDA_COMBINE_TOKENS_H_
