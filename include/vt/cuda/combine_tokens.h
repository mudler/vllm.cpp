// Device combine/scatter kernels for async-scheduling overlap (W3 ENG-ASYNC-SCHED
// runner leaf). Ports vllm/v1/worker/gpu/input_batch.py:303-406 @ the parity
// pin 5559679229bc961848b121ccdeaa8fa5d79bec98
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
// (the W3 body built + verified on dgx.casa; the CI box is CPU-only, and the
// SPEC-DFLASH2 A2-1 edit on top of it is unbuilt — see that row's `## Owed`).
// Signatures use plain pointers + vt::Queue so the header stays
// host-compilable.
#ifndef VT_CUDA_COMBINE_TOKENS_H_
#define VT_CUDA_COMBINE_TOKENS_H_

#include <cstdint>

#include "vt/backend.h"  // vt::Queue

namespace vt::cuda {

// combine_sampled_and_draft_tokens (input_batch.py:303-406), the CUDA analog of
// the host loop in src/vllm/v1/worker/gpu/prepare_inputs.cpp. Both bodies must
// say the same thing, and that includes what they REFUSE: where the host throws
// a VT_CHECK this kernel calls __trap(), because a kernel cannot throw. A device
// arm that continued where the host refused would score stale draft slots, and
// no token gate in this tree could see it. That source file carries the full
// contract and the upstream line-by-line anchors, and this header records only
// what the pointer signature adds.
//
// ONE host check has no device counterpart, and it is recorded rather than
// silently dropped: the host also bounds `req_state * stride + num_draft_tokens`
// against `draft_tokens.size()`, and no length reaches this side. The caller
// owns that allocation. Everything else refuses alike.
//
// State the consequence, because it is not "the check is merely absent". The
// scatter below reads draft_tokens[req_state_idx * draft_tokens_stride + b]
// with NOTHING bounding it, so an over-long row is an unchecked out-of-bounds
// DEVICE READ, and it has two outcomes. The loud one is an illegal memory
// access. The quiet one is the dangerous one: the read lands inside another
// allocation, garbage arrives in the draft slots, and because speculative
// decoding is lossless a wrong draft costs acceptance and NOTHING ELSE. No
// token gate in this tree can see that — it is reason A's class exactly
// (.agents/specs/dflash2-async-spec-sampler.md).
//
// A2-3 is where this becomes live, and the shape is concrete: A2-3 sizes the
// draft buffer by the ACTIVE REQUEST COUNT, while req_state_idx is a req_state
// POOL SLOT (that is the indirection idx_mapping documents below, the one that
// "matters after an abort/finish reorder"). A pool slot can exceed the active
// count, so a high slot indexes past the allocation with every host check
// satisfied. Whoever wires A2-3 either passes a length and refuses on it here
// as the host does, or sizes the buffer by num_req_states and not by num_reqs.
//
// For each request row b: num_logits comes from cu_num_logits (1 + k_i on a
// verify step), num_draft_tokens == num_logits - num_new_sampled_tokens, the
// committed token is spliced at query_end - num_logits under BOTH the
// seq_len > prefill_len and the first_logit_seq_pos >= prefill_len guards, and
// the drafts are scattered over [query_end - num_draft_tokens, query_end).
// Prefill/chunked-prefill rows (seq_len <= prefill_len) keep their prompt token.
//
// Three pointers accept nullptr, each meaning one specific thing:
//   idx_mapping     nullptr == the identity batch-row -> req_state mapping (our
//                   persistent batch is condensed dense, so batch row ==
//                   req_state slot; the indirection matters after an
//                   abort/finish reorder).
//   cu_num_logits   nullptr == arange(num_reqs + 1), i.e. num_logits == 1 for
//                   every row. That is the non-speculative shape and it is what
//                   every caller passes today. It is NOT the same clause as
//                   "num_logits == num_new_sampled_tokens": arange means one,
//                   whatever num_new_sampled_tokens is, and upstream likewise
//                   derives num_logits from cu_num_logits and never from
//                   NUM_NEW_SAMPLED_TOKENS (input_batch.py:322-325). At
//                   num_new_sampled_tokens == 0 the two part company, arange
//                   leaves num_draft_tokens == 1, and the host refuses; so does
//                   this kernel.
//   draft_tokens    nullptr == NO request has drafts this step. It is not a
//                   licence to skip the scatter: a row whose num_draft_tokens is
//                   greater than zero with a null draft_tokens, or with a stride
//                   narrower than its own count, is a contract violation and the
//                   kernel traps, exactly where the host throws. draft_tokens is
//                   otherwise upstream's 2-D [num_req_states,
//                   num_speculative_steps] tensor, row-major, with
//                   draft_tokens_stride == its stride(0).
//
// Our runner builds logits_indices in prepare_inputs, so this kernel writes only
// the input_ids stores (the upstream kernel's logits_indices store is not needed
// here). Launched on the MAIN queue BEFORE the forward (outside any decode-graph
// capture — input prep always precedes the graph replay).
//
// REACHABILITY: the draft lane is UNREACHED on `main`, exactly as on the host
// side. `async_input_combine_` is vetoed for every speculative engine, at BOTH
// GPUModelRunner constructors (src/vllm/v1/worker/gpu/runner.cpp:480 and :553 —
// the assignment, not the comment above it), so both call sites pass a null
// draft_tokens and a null cu_num_logits and the kernel degenerates to the
// pre-A2-1 single splice. Row `SPEC-DFLASH2` owns the wiring (waves A2-2, A2-3),
// issue #2644 tracks it, and the row's spec lists it under `## Owed`.
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

// W4 (discrete CUDA): replay InputBatch's STRUCTURAL edits to last_sampled_tokens
// onto the device mirror, in stream order.
//
// Upstream never needs this. vllm/v1/worker/gpu/states.py:132 frees a request's
// slot index into a pool and reuses it, so a request's req_state index is stable
// for its lifetime and the GPU tensor is never permuted. Our InputBatch instead
// CONDENSES (moves the last live row into the freed slot) and swaps rows in the
// decode-first reorder. Once the values live on the device the host cannot
// perform those moves without reading them back — which is the synchronize this
// whole row exists to delete — so the host records what it did and the device
// replays it here.
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

}  // namespace vt::cuda

#endif  // VT_CUDA_COMBINE_TOKENS_H_
