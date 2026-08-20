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

// SPEC-DFLASH2 W2 (#1314) — refuse a DFlash2 draft's CANDIDATE SELECTOR BY NAME,
// after the draft block forward and before anything samples from its logits.
//
// This is the boundary W2 leaves the architecture at. A `DFlash2DraftModel` draft
// now LOADS and its block forward RUNS, grouped dynamic convolution and all
// (vt::DFlashGroupedConv, wrapped around every attention and MLP sublayer). What
// it cannot do is CHOOSE: upstream replaces the independent per-slot argmax with
// a candidate selector -- keep the target head's top-K per slot, score adjacent
// transitions `<A[p] * project(h), B[c]> + unary[c]`, and walk the best path from
// the verified anchor (`vllm/model_executor/models/qwen3_dflash2.py`
// `CandidateSelector` + `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` @
// vllm-project/vllm#52816 head `19c9351904df4c63042671bc67a866ca48dc7d6f`), and
// none of that exists here.
//
// Falling through to `SampleDflashBlockDrafts` instead would SUCCEED and be
// silent: the per-slot argmax proposes well-formed tokens, the verify is
// lossless, so the engine still emits the target's tokens and only ACCEPTANCE
// falls -- the one defect class no token gate in this repository can see. That is
// why this is a refusal and not a fallback, and why it is placed AFTER the
// forward: the forward is implemented and gated, the choice is not.
//
// Two call sites turn draft logits into draft tokens and both refuse here:
// `GPUModelRunner::propose_drafts_block` (src/vllm/v1/worker/gpu/runner.cpp) and
// `DflashProposeBlock` below. Only the FIRST is production. `DflashProposeBlock`
// has no caller outside `tests/` at this commit -- grep it -- so the refusal that
// a test can delete-and-redden is the test-reachable one, and the site a user
// actually arrives through is UNGATED. Entering it needs a runner whose
// `dflash_weights_` is set, which only the `LoadedModel` construction path does,
// so a gate on it needs an on-disk target plus draft driven through the loader.
// That harness is W4's. See `## Owed` O7 of
// `.agents/specs/dflash2-spec-decode.md`; the refusal itself is owed by W3.
void RefuseDflash2CandidateSelector(const Qwen3DFlashWeights& weights);

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
