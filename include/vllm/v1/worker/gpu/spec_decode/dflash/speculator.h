// Ported from: vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
// (DflashSpeculator.propose :300-413, _generate_draft :242-273, sample_draft
// inherited from autoregressive/speculator.py) @ 555967922 (vLLM 0.26.0.dev0).
//
// Scope (SPEC-DFLASH D4, row DF-ENGINE-INTEGRATION): the NON-autoregressive
// whole-block DFlash propose brick — the DFlash analogue of MtpProposePrefill
// (mtp/speculator.h). Where MTP runs one k=1 autoregressive early-exit forward,
// DFlash proposes the entire k-token block in ONE forward that attends over the
// pre-inserted context K/V. This brick COMPOSES the landed D2/D3 pieces:
//   * D3 Qwen3DFlashModel::ForwardBlockLogitsWithContext — the context-aware
//     (1+k) block forward (non-causal in-block for full-attn layers, causal SWA),
//     which internally PrecomputeContextKV's the accumulated context features and
//     lays out [context; block] per request (the D2 vt::DFlashBlockAttention);
//   * greedy sample_draft — argmax over each mask position's draft logits row
//     (the anchor row at block offset 0 is NOT sampled; sample_from_anchor=false).
//
// The caller (GPUModelRunner::propose_drafts, DFlash branch) owns the two stateful
// inputs: (a) `context_states` = the per-request ACCUMULATED combined features
// fc(cat(aux)) for the sequence so far (D1 multi-tap -> D2 CombineAuxFeatures,
// appended per verify step honoring num_rejected), and (b) `block_*` = this step's
// (1+k) mask block from PrepareDflashInputs (anchor=bonus token, then k mask ids).
// This mirrors vLLM's precompute_and_store_context_kv(hidden_states,...) writing
// the accumulating draft KV cache, then the block forward attending over it — but
// via the D3 inline [context; block] recompute (numerically the same projection;
// the persistent-paged-draft-KV fast path is a D6 perf concern, not correctness).
//
// This is a CALLABLE, tested propose brick. It is UNREACHABLE unless a DFlash
// SpeculativeConfig is set (method=="dflash"): the MTP + non-spec runner paths do
// not construct DFlash draft weights and never call it, so the shipped engine is
// byte-identical (identical inertness discipline to MtpProposePrefill).
#ifndef VLLM_V1_WORKER_GPU_SPEC_DECODE_DFLASH_SPECULATOR_H_
#define VLLM_V1_WORKER_GPU_SPEC_DECODE_DFLASH_SPECULATOR_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"  // Qwen3DFlashModel, Qwen3DFlashWeights
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm::v1 {

// SPEC-DFLASH2 W3 (#1314) RETIRED `RefuseDflash2CandidateSelector`, which used
// to be declared here.
//
// W2's boundary was the candidate selector: the draft loaded, its grouped
// convolution ran, and it was refused because nothing could CHOOSE among its
// logits. W3 implements the choosing up to but not including the walk -- the
// target head's top-K (`vt::TopKValuesIndices`), the two codebooks and the edge
// lattice (`vt::Dflash2SelectorEdges`), through
// `vllm::v1::Dflash2SelectCandidates`
// (include/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h) -- so the
// boundary MOVED one step, to `RefuseDflash2PathWalk` in that same header.
//
// The refusal is not declared in two places any more, and neither is the step
// before it. Both propose paths call ONE `Dflash2SelectCandidates`, so the
// sequence a user arrives through and the sequence a gate drives are the same
// code. W2 had two copies of its refusal and only the test-reachable one was
// gated (spec `## Owed` O7); collapsing the duplicate is how this wave stops
// that shape from recurring.

// Greedy per-request draft pick over the (1+k) block logits — the greedy branch of
// DFlash sample_draft (dflash/speculator.py:_generate_draft :259-273 with
// temperature 0 => argmax). `block_logits` is the ForwardBlockLogitsWithContext
// output [num_reqs*(1+k), draft_vocab] in per-request block-row order: request r
// owns rows [r*(1+k) .. r*(1+k)+k]; row +0 is the ANCHOR (bonus/verified token, NOT
// sampled, sample_from_anchor=false :53-56), rows +1..+k are the k mask positions.
// Returns [num_reqs][k]: draft[r][j] = argmax(block_logits row r*(1+k)+1+j) with
// lowest-index tie-break (matching our sampler's argmax + the D2/D3 gates).
//
// Split out from DflashProposeBlock so the sampling/assembly is unit-testable on a
// deterministic synthetic logits fixture independent of a loaded draft model.
std::vector<std::vector<int32_t>> SampleDflashBlockDrafts(
    const std::vector<float>& block_logits, int num_reqs, int k, int64_t draft_vocab);

// One non-autoregressive DFlash block propose (dflash/speculator.py::propose
// :300-413, greedy). Runs the context-aware draft block forward over the caller-
// accumulated combined-feature context + this step's prepared (1+k) mask block, and
// greedily samples the k proposed tokens per request.
//
//   weights / config     the loaded z-lab DFlash draft (Qwen3DFlashWeights) + the
//                        resolved draft HfConfig (num layers, head dims, taps,
//                        mask_token_id, draft_vocab_size).
//   context_states       [num_ctx, H] f32 — the ACCUMULATED combined features
//                        fc(cat(aux)) for every prior sequence token, in per-request
//                        order (request r occupies [ctx_cu[r], ctx_cu[r+1])).
//   context_positions    [num_ctx] i32 — each context token's absolute position.
//   ctx_cu               [num_reqs+1] i32 — per-request context boundaries.
//   block_input_ids      [num_reqs*(1+k)] i32 — the mask block from
//                        PrepareDflashInputs (anchor + k mask_token_id per request).
//   block_positions      [num_reqs*(1+k)] i32 — the block token positions.
//   block_cu             [num_reqs+1] i32 — per-request block boundaries (stride 1+k).
//   num_reqs / k         batch size / num_speculative_tokens (block-1).
//
// Returns [num_reqs][k] draft token ids (the whole proposed block per request).
struct DflashProposeResult {
  std::vector<std::vector<int32_t>> draft_token_ids;  // [num_reqs][k]
};

DflashProposeResult DflashProposeBlock(
    const Qwen3DFlashWeights& weights, const HfConfig& config,
    const std::vector<float>& context_states,
    const std::vector<int32_t>& context_positions,
    const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids,
    const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& block_cu, int num_reqs, int k, vt::Queue& queue);

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_SPEC_DECODE_DFLASH_SPECULATOR_H_
