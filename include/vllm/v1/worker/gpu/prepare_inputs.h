// Ported from: vllm/v1/worker/gpu_model_runner.py::_prepare_inputs @ e24d1b24
// (+ _update_states, _get_cumsum_and_arange; the num_scheduled_tokens_np build
//  from execute_model @ e24d1b24 lines 4124-4133.)
//
// Scope (M1.5 Task 3): the step-input construction the model runner does before
// the forward — apply a SchedulerOutput's diffs to the persistent InputBatch
// (update_states), then build the flattened per-step inputs the attention op
// (M1.6) + sampler (M1.7) consume (prepare_inputs -> StepInputs). Behavioral
// only: no CUDA, no model. The "tensors" are plain host arrays.
//
// ─── V1 ALGORITHM / MRV2 CONTRACT (recorded) ────────────────────────────────
// Source of truth is the V1 runner's `_prepare_inputs` / `_update_states`
// (`gpu_model_runner.py`), which drive the persistent V1 `InputBatch`
// (Task 2) + `MultiGroupBlockTable` (Task 1) on host arrays. It is DRIVEN by
// the MRV2 scheduler-output CONTRACT, not the MRV1 runner's admission path:
//   * new requests carry `prefill_token_ids`; add_request seeds their per-slot
//     token_ids_cpu (prompt then output == prefill_token_ids). Resumed reqs are
//     folded in AS NEW under our MRV2 scheduler (they arrive in
//     scheduled_new_reqs), so there is a single admission path: add_request.
//   * cached diffs are `num_computed_tokens` + `new_block_ids`: for a request
//     still in the persistent batch, update_states sets num_computed_tokens_cpu
//     and appends new_block_ids to its block-table row.
//
// ─── DEFERRED upstream behavior (marked; T0 never exercises) ────────────────
//   * The MRV1-shape admission path (per-req `all_token_ids` reconstruction,
//     the resumed-from-preemption REPLACE-and-readd of a stored
//     CachedRequestState) is dead under our MRV2 scheduler and NOT ported.
//     update_states owns only the persistent InputBatch; the runner's separate
//     `self.requests` CachedRequestState store is not modeled here — a cached
//     diff for a req NOT in the batch is skipped (the resumed-as-new contract
//     re-admits it via scheduled_new_reqs). Recorded as a carried assumption.
//   * PP (non-last-rank new_token_ids -> token_ids_cpu writes), spec decode
//     (scheduled_spec_decode_tokens / _calc_spec_decode_metadata /
//     num_accepted_tokens), async scheduling (prev_sampled_token_ids /
//     _prepare_input_ids fast paths / prev_positions), M-RoPE / XD-RoPE,
//     LoRA, prompt_embeds, encoder/cross-attention, and the GPU-side staging
//     (input_ids/positions/query_start_loc/seq_lens copy_to_gpu, the
//     non-decreasing query_start_loc / zero-fill seq_lens CUDA-graph padding)
//     are all DEFERRED. At T0 the sampled decode token is assumed already
//     present in token_ids_cpu (the M1.7 sampler / bookkeeping's job).
//
// ─── STEP-INPUT SEMANTICS matched 1:1 ───────────────────────────────────────
//   req_indices    = repeat(arange(num_reqs), num_scheduled_tokens)
//   cu_num_tokens  = cumsum(num_scheduled_tokens)      (per-req end offsets)
//   query_pos      = batched arange (0..n-1 per req)   (_get_cumsum_and_arange)
//   positions[t]   = num_computed_tokens_cpu[req_indices[t]] + query_pos[t]
//   input_token_ids[t] = token_ids_cpu[req_indices[t], positions[t]]
//   query_start_loc = [0] ++ cu_num_tokens             (len num_reqs+1)
//   seq_lens[i]    = num_computed_tokens_cpu[i] + num_scheduled_tokens[i]
//   slot_mapping   = block_table.compute_slot_mapping(...) per KV cache group
//                    (block_id*block_size + within-block offset)
//   logits_indices = query_start_loc[1:] - 1           (last token per seq)
#ifndef VLLM_V1_WORKER_GPU_PREPARE_INPUTS_H_
#define VLLM_V1_WORKER_GPU_PREPARE_INPUTS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "vllm/v1/core/sched/output.h"
#include <string>
#include <unordered_map>

#include "vllm/v1/worker/gpu/input_batch.h"

namespace vllm::v1 {

// The flattened per-step inputs the attention op (M1.6) + sampler (M1.7)
// consume. Host arrays at T0 (device placement is the runner's concern).
struct StepInputs {
  // The scheduled token id per flattened token slot (batch order). [total]
  std::vector<int32_t> input_token_ids;
  // Absolute position of each flattened token: num_computed + arange. [total]
  std::vector<int64_t> positions;
  // Cumulative per-request token offsets: [0, cu_num_tokens...]. [num_reqs+1]
  std::vector<int32_t> query_start_loc;
  // Per-request sequence length after this step: num_computed + num_scheduled.
  // [num_reqs]
  std::vector<int32_t> seq_lens;
  // Per KV cache group: the flat KV-cache slot id per flattened token
  // (block_id*block_size + offset). slot_mapping[g] is group g's mapping,
  // truncated to [0, total). Upstream computes one per group.
  std::vector<std::vector<int64_t>> slot_mapping;
  // Index (into the flattened token stream) of each row the sampler needs
  // logits for. [total_num_logits]
  //
  // NON-SPECULATIVE (the production default, and every step until a speculator
  // is configured): exactly one row per request — query_start_loc[1:] - 1, the
  // last scheduled token per sequence. [num_reqs], byte-identical to the
  // pre-SPEC-REJECTION array.
  //
  // SPECULATIVE (SPEC-REJECTION I3): when the scheduler scheduled drafts, a
  // request with k_i drafts needs logits at 1 + k_i positions — the k_i draft
  // positions plus the bonus position — so the target's argmax at each draft
  // position can be compared against the draft (spec §2.4). The rows are the
  // TAIL of the request's query: logits_start = query_end - (1 + k_i), then
  // 1 + k_i consecutive rows. Mirrors
  // gpu/input_batch.py::_combine_sampled_and_draft_tokens_kernel:317-327
  // ("logits_start = query_end - num_logits").
  std::vector<int32_t> logits_indices;
  // Per-request cumulative offsets into logits_indices: [0] ++ cumsum(1 + k_i).
  // [num_reqs + 1]. On the non-speculative path this is exactly
  // arange(num_reqs + 1) (upstream model_runner.py:872-875), so
  // cu_num_logits[i] == i and cu_num_logits.back() == num_reqs.
  // Mirrors gpu/model_runner.py:866-898.
  std::vector<int32_t> cu_num_logits;
  // Number of draft tokens scheduled per request. EMPTY when no drafts were
  // scheduled (upstream num_draft_tokens_per_req is None then,
  // model_runner.py:867,881-889). [num_reqs] otherwise.
  std::vector<int32_t> num_draft_tokens_per_req;
  // sum(num_draft_tokens_per_req); 0 on the default path. The runner routes
  // through the rejection sampler IFF this is > 0
  // (model_runner.py:1065 `if input_batch.num_draft_tokens == 0 ...`).
  int num_draft_tokens = 0;
  // num_scheduled_tokens per request in batch order (the array upstream's
  // execute_model builds and passes to _prepare_inputs). [num_reqs]
  std::vector<int32_t> num_scheduled_tokens;

  // ─── prompt logprobs (SAMPLE-PROMPT-LOGPROBS) ─────────────────────────────
  // One entry per request that asked for prompt logprobs AND has rows to
  // produce this step, in batch order. Mirrors the per-request body of
  // gpu_model_runner.py::_get_prompt_logprobs_dict:5626-5686 — the arithmetic
  // that decides WHICH prompt positions get an lm_head row is pure input prep,
  // so it is computed here; the scoring and the cross-step accumulation stay on
  // the runner. EMPTY on every step where no request asked, which is what keeps
  // the production path byte-identical (see specs/prompt-logprobs.md).
  struct PromptLogprobRows {
    std::string req_id;
    // Requested count k; the tensor is [num_prompt_tokens-1, k+1].
    int num_prompt_logprobs = 0;
    // Destination row range in the request's accumulated tensor:
    // [dst_start, dst_start + num_rows) == upstream's
    // slice(start_idx, start_idx + num_logits) at :5698.
    int dst_start = 0;
    int num_rows = 0;
    // Offset into prompt_logprob_indices of this request's first row.
    int src_start = 0;
    // Prompt token ids scored at those rows — prompt[start_tok + i], the NEXT
    // token at each position (:5684-5686). [num_rows]
    std::vector<int64_t> target_token_ids;
    // This step completes the request's prompt: emit the tensor and drop the
    // in-progress state (:5665-5667, :5709-5712).
    bool final_chunk = false;
  };
  std::vector<PromptLogprobRows> prompt_logprob_rows;
  // Flat token-stream indices for every row named above, concatenated in the
  // same request order. Appended to logits_indices for the forward's gather so
  // the model's own lm_head produces those rows; empty on the default path.
  std::vector<int32_t> prompt_logprob_indices;
};

// update_states: apply a SchedulerOutput to the persistent InputBatch — remove
// finished + unscheduled requests, admit the scheduled new requests
// (from_new_request -> add_request), apply the cached diffs (num_computed_tokens
// + new_block_ids append) for requests still in the batch, then condense().
// Matches gpu_model_runner.py::_update_states ordering (finished/unscheduled
// removal -> collect new -> cached diffs -> add_request -> condense). The
// deferred PP / spec / async / resumed-store paths are documented in the header.
//
// `req_states` (ENG-MM-INPUT-PIPELINE P2, #2379) is upstream's
// `self.requests: dict[str, CachedRequestState]` (gpu_model_runner.py), the
// per-REQUEST state that lives BESIDE the per-SLOT input batch. Upstream keeps
// it unconditionally; the runner passes non-null ONLY for a model that declares
// the multimodal seam, because the map holds a full copy of every prompt and a
// text engine has no reader for it. NULL — every existing caller — leaves this
// function byte-identical.
void update_states(
    InputBatch& input_batch, const SchedulerOutput& scheduler_output,
    std::unordered_map<std::string, CachedRequestState>* req_states = nullptr);

// prepare_inputs: build the flattened StepInputs from the (already
// update_states'd) persistent InputBatch + the scheduler output. Mirrors
// gpu_model_runner.py::_prepare_inputs (commit_block_table -> req_indices /
// cumsum+arange -> positions -> token gather -> query_start_loc -> seq_lens ->
// compute_slot_mapping -> logits_indices). Requires num_reqs > 0 and
// total_num_scheduled_tokens > 0 (upstream asserts both).
StepInputs prepare_inputs(InputBatch& input_batch,
                          const SchedulerOutput& scheduler_output);

// combine_sampled_and_draft_tokens: the async-scheduling device-input path
// (ENG-ASYNC-SCHED W3 runner leaf, made draft-aware by SPEC-DFLASH2 A2-1).
// Ported from vllm/v1/worker/gpu/input_batch.py::combine_sampled_and_draft_tokens
// (:364-406) + _combine_sampled_and_draft_tokens_kernel (:303-361) @ the parity
// pin 5559679229bc961848b121ccdeaa8fa5d79bec98.
//
// Under async/overlap scheduling step N+1 is prepared BEFORE step N's sampled
// token has crossed to the host (the blocking D2H the whole lever removes), so
// prepare_inputs' host read of token_ids_cpu at the decode position is stale.
// This rebuilds the generated-token rows' input ids from the two buffers that
// already live on the sampler's side of the boundary:
//
//   * the COMMITTED token, from last_sampled_tokens[req_state] — written at
//     logits_start == query_end - num_logits (:344-348);
//   * the DRAFT tokens, from draft_tokens[req_state] — scattered over
//     [query_end - num_draft_tokens, query_end) (:350-361), where
//     num_draft_tokens == num_logits - num_new_sampled_tokens (:325).
//
// num_logits is per request and comes from cu_num_logits, NOT from
// num_new_sampled_tokens. That distinction is the whole of A2-1. On a verify
// step a request with k drafts has num_logits == 1 + k, so logits_start is
// query_end - (1 + k); reading num_logits as 1 there writes the committed token
// over the LAST DRAFT SLOT and silently destroys draft k-1. That corruption
// costs acceptance and nothing else — speculative decoding is lossless, so the
// emitted tokens never move and no token gate, golden or acceptance ratio on
// this tree can see it (`.agents/specs/dflash2-async-spec-sampler.md`, reason A;
// it already shipped once as #1366). The gate that CAN see it is G2, and it
// asserts on the draft token ids in tests/vllm/v1/worker/test_combine_tokens.cpp.
//
// Prefill / chunked-prefill rows (seq_len <= prefill_len, incl. the exact chunk
// that completes prefill) are LEFT UNTOUCHED — their input still comes from the
// prompt in token_ids_cpu. Past that, the committed-token store carries a SECOND
// guard, `first_logit_seq_pos >= prefill_len` (:344-345): when the logits window
// reaches back over the prompt tail (possible once num_logits > 1), logits_start
// addresses a PROMPT slot, and upstream leaves it alone. The draft scatter is
// NOT under that guard, exactly as upstream.
//
// Returns logits_indices, stored at cu_num_logits[b] + j exactly as the kernel
// stores it (:331-335), so on the non-speculative path (cu_num_logits ==
// arange(num_reqs + 1), num_new_sampled_tokens == 1) it is byte-identical to
// prepare_inputs' logits_indices (query_start_loc[1:] - 1).
//
//   input_token_ids  [total]        mutated in place (generated rows overwritten)
//   idx_mapping      [num_reqs]     batch_idx -> req_state slot (identity for our
//                                   already-condensed persistent batch; the
//                                   indirection matters after an abort/finish
//                                   reorders req_states vs the dense batch)
//   last_sampled_tokens [>= max req_state]  per req_state, the last sampled id
//   query_start_loc  [num_reqs + 1] cumulative token offsets (== StepInputs)
//   seq_lens         [num_reqs]     num_computed + num_scheduled (== StepInputs)
//   prefill_len      [>= max req_state]  per req_state, tokens known at admission
//   draft_tokens     [>= (max req_state + 1) * draft_tokens_stride]  ROW-MAJOR
//                                   per req_state, upstream's 2-D
//                                   [num_req_states, num_speculative_steps]
//                                   tensor flattened; may be empty when no
//                                   request has drafts this step
//   draft_tokens_stride             upstream's draft_tokens.stride(0) (:381,396)
//   cu_num_logits    [num_reqs + 1] StepInputs::cu_num_logits — [0] ++ cumsum
//                                   (num_new_sampled_tokens + k_i). REQUIRED, as
//                                   upstream requires it: it is the only source
//                                   of the per-request num_logits
//   num_new_sampled_tokens          0 or 1 (bonus token; excl. accepted drafts)
//
// HARNESS ADAPTATION, the only one: upstream also takes `num_logits` (the total)
// to size its output tensor (:373,383-387). We derive it from cu_num_logits.back(),
// which upstream requires to be the same value, so the two cannot disagree here.
//
// DEVICE-NEUTRAL (recorded): on CPU these are host std::vectors; the CUDA
// counterpart is vt::cuda::LaunchCombineSampledAndDraftTokens
// (src/vt/cuda/cuda_combine_tokens.cu), which mirrors this body over GPU-resident
// buffers so no sampled id round-trips the host. It runs on the HOST side of
// input prep, BEFORE the forward and OUTSIDE any CUDA-graph capture (exactly as
// upstream keeps input prep ahead of the decode graph replay), so it is
// capture-safe by construction.
//
// REACHABILITY, stated once: the draft lane below is STILL UNREACHED on `main`,
// and A2-3 (#2911) changed WHY rather than fixing it. Its `draft_tokens` argument
// is now real — every call site passes the per-req_state `InputBatch::draft_tokens`
// with its stride, and the host arm passes the true per-request
// `step.cu_num_logits` — so the buffer this scatter reads exists and is
// maintained. What still keeps `num_logits > 1` from arriving is the veto on
// `async_input_combine_` at BOTH GPUModelRunner constructors (the two
// `async_input_combine_ = ...` assignments, not the comment blocks that precede
// them), which wave A2-5 lifts and which this wave was explicitly scoped not to
// touch. A2-3 measured what lifting it now does: the combine ACCEPTS the verify
// step and no combine refusal fires; one unrelated gap remains in
// `sample_tokens_async`'s decode arm. Row `SPEC-DFLASH2` owns the wiring, issue
// #2911 tracks it, and the row's spec lists both under `## Owed`.
std::vector<int32_t> combine_sampled_and_draft_tokens(
    std::vector<int32_t>& input_token_ids,
    const std::vector<int32_t>& idx_mapping,
    const std::vector<int32_t>& last_sampled_tokens,
    const std::vector<int32_t>& query_start_loc,
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& prefill_len,
    const std::vector<int32_t>& draft_tokens, int draft_tokens_stride,
    const std::vector<int32_t>& cu_num_logits, int num_new_sampled_tokens = 1);

// ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710): THE COMBINE'S ROW PREDICATE, extracted so
// that one rule has one expression.
//
// TRUE when batch row `i`'s host input identifier is about to be REPLACED by the
// previous step's sampled token — which is the same thing as saying the host
// vector is stale for that row. A row still consuming known prefill tokens,
// INCLUDING the chunk that exactly completes prefill, keeps its prompt token and
// is left untouched, so the boundary is a strict `>` and not a `>=`.
//
// WHY IT IS A NAMED FUNCTION NOW. `combine_sampled_and_draft_tokens` above wrote
// this condition inline and the CUDA kernel it mirrors
// (`src/vt/cuda/cuda_combine_tokens.cu`) writes it again, which was tolerable
// while the only readers WERE the two combines. A third reader now needs the same
// answer: the guard in `ModelRegistry::Forward` that refuses a forward which
// ignores `ModelForwardInput::device_token_ids`. A refusal whose predicate is a
// re-derivation of the route's predicate is precisely the failure this repository
// has already shipped once, so the guard SHARES this expression instead of
// agreeing with it by inspection.
//
// The device kernel keeps its own copy because device code cannot call a host
// inline. That copy is pre-existing; the spec records as owed the fact that
// nothing pins the two against each other on a device.
inline bool CombineSplicesRow(int32_t seq_len, int32_t prefill_len) {
  return seq_len > prefill_len;
}

// TRUE when ANY row of this step is one `CombineSplicesRow` splices — which is
// exactly "the host token identifiers are stale for this step".
//
// PER-STEP, AND THE GRANULARITY IS THE POINT. Staleness is a per-ROW fact, but
// the decision it feeds is a per-STEP one: a forward either reads
// `device_token_ids` or it does not, and it is handed the whole step at once. A
// per-REQUEST reading of the same rule — "this request is a prefill row, so it is
// fine" — lets a MIXED step through for its prefill rows while its decode row is
// served from identifiers the runner never wrote. That is a refusal whose
// predicate disagrees with its route predicate, which this tree has shipped
// before: a per-request refusal paired with a per-step route veto, where every
// test used `num_reqs == 1` so the two agreed on every test input.
//
// So this reduces with OR over the whole batch and never reports a per-row
// answer. `idx_mapping` maps batch row -> req_state slot for `prefill_len`,
// exactly as the combine does; pass nullptr for the identity mapping that the
// condensed-dense persistent batch uses.
bool AnyRowSplicedByCombine(const std::vector<int32_t>& seq_lens,
                            const std::vector<int32_t>& prefill_len,
                            const int32_t* idx_mapping, int num_reqs);

// SPEC-DFLASH2 A2-2 (#2802): THE VERIFY ROUTE PREDICATE, extracted for the same
// reason `CombineSplicesRow` was — one rule, one expression.
//
// Upstream writes it once, in one sampler, at
// `vllm/v1/worker/gpu/model_runner.py:1129` @ pin 5559679229:
//
//     if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
//
// We have TWO sampling entry points (`sample_tokens` and
// `sample_tokens_async`), so the same rule is now asked in two places, and a
// third place — the async input combine's refusal — asks its NEGATION. All
// three call this.
//
// PER STEP, AND THAT IS THE POINT. `num_draft_tokens` is the batch TOTAL,
// `sum(num_draft_tokens_per_req)`. The forward produced ONE expanded logits
// tensor for the whole step and either the rejection sampler consumes it or the
// plain sampler does; there is no per-request choice to make. A per-REQUEST
// reading of the same rule — "this row drafted nothing, so it is a decode row"
// — answers differently on a MIXED step, where some rows carry drafts and
// others carry none, and it would hand the plain sampler a tensor whose rows
// are not one-per-request. This repository has already shipped a per-request
// refusal paired with a per-step route predicate (#2710), where it survived 27
// mutations because every test used `num_reqs == 1`.
inline bool StepRoutesToVerify(int32_t step_num_draft_tokens) {
  return step_num_draft_tokens > 0;
}

// TRUE when batch row `i` carried drafts this step. This is the PER-ROW fact,
// and it is a different question from `StepRoutesToVerify` — it decides only
// per-row bookkeeping inside the verify arm (the acceptance telemetry), never
// the route. Named so the two readings cannot be confused for each other, and
// so a test can put them side by side on a mixed step.
inline bool RowCarriesDraftTokens(int32_t row_num_draft_tokens) {
  return row_num_draft_tokens > 0;
}

// SPEC-DFLASH2 A2-3 (#2911): THE ASYNC PLACEHOLDER FILL'S PER-REQUEST RULE.
//
// Under async scheduling the scheduler ships `-1` placeholders for a verify
// step's draft positions (async_scheduler.py:43-45), because it schedules step
// N+1 before step N's drafts exist. `GPUModelRunner::execute_model` patches a
// LOCAL copy of `scheduled_spec_decode_tokens` with the drafts this runner
// actually proposed, exactly as upstream's worker-side scatter leaves the
// scheduler's own copy untouched (gpu_input_batch.py:520-523).
//
// This returns request `req_state_idx`'s drafts for that patch, read from the
// SAME per-req_state buffer `combine_sampled_and_draft_tokens` scatters from
// (`InputBatch::draft_tokens`, mirroring `RequestStates.draft_tokens`,
// states.py:71-77). Before A2-3 the fill read `pending_drafts_` — a host, ragged,
// req_id-keyed structure with no relation to the scatter's buffer — so the two
// consumers of "what did this runner draft" had two sources. One residence, one
// producer, two readers is upstream's shape and is what this function exists to
// keep true.
//
// WHY IT IS A NAMED FUNCTION. The row arithmetic (`req_state_idx * stride + b`)
// is written by the host combine, by the CUDA kernel, and now by the fill. One
// rule with three expressions is how the predicate split shipped here before
// (#2710), so the fill's reading is stated once and the call site APPLIES it
// rather than agreeing with it by inspection.
//
// THE REFUSALS ARE REFUSALS AND NOT CLAMPS. A placeholder the worker cannot fill
// would otherwise embed a literal `-1` in the model's input ids.
//
//   * `num_valid < num_placeholders` — the runner proposed fewer drafts than the
//     scheduler reserved positions for, which is a bookkeeping defect on either
//     side.
//   * the row must lie inside the buffer — THE BOUND THE CUDA SCATTER LACKS.
//     `LaunchCombineSampledAndDraftTokens` reads
//     `draft_tokens[req_state_idx * stride + b]` with nothing checking it, so a
//     row past the allocation is an unchecked device read. Its loud outcome is
//     an illegal access and its quiet one is garbage in the draft slots, which
//     costs acceptance and NOTHING ELSE because speculative decoding is
//     lossless. Sizing `InputBatch::draft_tokens` by the req_state pool is what
//     makes that read safe; this refusal is what says so out loud if some future
//     caller sizes it by `num_reqs` instead.
//
// WHAT THIS FUNCTION CANNOT CHECK, stated because a reader will otherwise assume
// it does. It cannot tell a FRESH row from a STALE one. `InputBatch::draft_tokens`
// and `num_valid_draft_tokens` are persistent and survive the out-of-band pull —
// that is why they replaced `pending_drafts_` — and only `clear_draft_tokens`
// zeroes the count, so a step that skipped its propose reaches here with the
// PREVIOUS verify step's row and every check above satisfied. The pre-A2-3 fill
// caught that input by name, from `pending_drafts_`, whose move-out made its
// membership test a freshness test. `ProposedDraftLedger` below carries that half
// now, and `FillDraftsForStep` below applies it before it calls this, on every
// async-scheduling step `GPUModelRunner::execute_model` runs.
// Splicing a stale draft costs acceptance and raises nothing, verify being
// lossless, so the guarantee has to live somewhere and it is not here.
//
// `num_placeholders == 0` returns empty and refuses nothing, because that is the
// non-speculative path and every decode row on a mixed step.
//
//   draft_tokens         [>= (max req_state + 1) * stride] ROW-MAJOR per req_state
//   draft_tokens_stride  the speculator's MAX draft length; a shorter row is
//                        PADDED, so the fill count comes from num_placeholders
//                        and never from the stride (the combine draws the same
//                        distinction with num_draft_tokens)
//   req_state_idx        the request's req_state slot, NOT its batch row
//   num_valid            how many of that row this runner actually proposed
//   num_placeholders     how many positions the scheduler reserved this step
//   req_id               named in the refusal only; a refusal that cannot say
//                        which request is worse than one that can
std::vector<int32_t> FillDraftsForRow(const std::vector<int32_t>& draft_tokens,
                                      int draft_tokens_stride, int req_state_idx,
                                      int num_valid, int num_placeholders,
                                      const std::string& req_id);

// SPEC-DFLASH2 A2-3 REPAIR (#2911): THE PRODUCER'S ROW RULE, the counterpart of
// `FillDraftsForRow` above and the other half of the one residence.
//
// WHY IT IS A NAMED FUNCTION, and it is not symmetry for its own sake. A review
// deleted the payload scatter from `GPUModelRunner::set_draft_tokens` — the
// `for (int c = 0; c < stride; ++c)` loop, leaving the count write in place —
// and ALL EIGHT of this row's targets stayed green. Every draft in the process
// could be zeroed end to end and nothing red, because verify is lossless: the
// emitted tokens do not move and only ACCEPTANCE falls (#1366, and reason A in
// this row's spec). Deleting the COUNT write does red, loudly, at the fill's
// refusal — so the call site is proven REACHED and the payload was proven
// UNOBSERVED at the same time. Stating the write here puts the payload and the
// count in one function that `tests/vllm/v1/worker/test_draft_fill.cpp` drives
// against `FillDraftsForRow` as a ROUND TRIP: what the producer wrote is what
// the consumer reads back, so deleting the payload write now reds.
//
// It writes TWO things and a caller may not have one without the other:
//   * the row, PADDED to the stride with zeros. The pad is never read (both
//     readers take their count from `num_valid_draft_tokens` or from the
//     scheduler's placeholder count), and zeroing it keeps a previous occupant's
//     tail out of a dump or a debugger.
//   * `num_valid_draft_tokens[req_state_idx]`, the count that decides how much
//     of the row is real. Upstream governs the same way: `req_states.draft_tokens`
//     keeps whatever it held and the count says what is live.
//
// The refusals mirror the fill's. A row longer than the stride is the speculator
// contradicting the `k` the buffer was sized from, which would truncate silently
// here and truncate a DIFFERENT number of drafts in the combine. A row outside
// the buffer means the buffer was sized by `num_reqs` rather than by the
// req_state pool, which is the sizing that makes the CUDA scatter's unbounded
// row read safe.
//
//   draft_tokens             [>= (max req_state + 1) * stride] ROW-MAJOR
//   num_valid_draft_tokens   [>= max req_state + 1], one count per req_state
//   draft_tokens_stride      the speculator's MAX draft length
//   req_state_idx            the request's req_state slot, NOT its batch row
//   row                      what this propose produced for that request
//   req_id                   named in the refusals only
void WriteDraftRow(std::vector<int32_t>& draft_tokens,
                   std::vector<int32_t>& num_valid_draft_tokens,
                   int draft_tokens_stride, int req_state_idx,
                   const std::vector<int32_t>& row, const std::string& req_id);

// SPEC-DFLASH2 A2-3 REPAIR (#2911): THE ASYNC FILL'S FRESHNESS RULE.
//
// The other half of the pre-A2-3 fill's single refusal, restored as its own
// thing. That refusal read `pending_drafts_` and asked whether this runner had a
// draft row for the request; because `take_draft_token_ids` MOVES that object
// out on the way to the scheduler, the question it actually answered was "did a
// propose run for this request since the last time the fill looked". A2-3 moved
// the drafts' residence to `InputBatch::draft_tokens` — correctly, because
// `pending_drafts_` cannot be a residence two in-step readers share — and
// rehomed the message onto `req_id_to_index`, which asks something weaker and
// nearly always true. Neither `draft_tokens` nor `num_valid_draft_tokens` can
// answer the freshness question: both are persistent, both survive the pull by
// design, and only `clear_draft_tokens` zeroes the count. A step that skips its
// propose therefore reaches the fill with the PREVIOUS verify step's row intact.
//
// WHY THAT IS NOT A HYPOTHETICAL. Verify is lossless, so splicing a stale draft
// emits exactly the same tokens and only costs ACCEPTANCE — the class of defect
// no token gate in this tree can see (#1366). The async decode arm that never
// proposes (`## Owed`, #2911) is one such step today.
//
// THE RULE IS THE PAIR, not the lookup. `Record` is what a propose that produced
// rows leaves behind, `Clear` is what a propose that produced nothing leaves
// behind, and `Consume` is the fill saying it has used them. Freshness comes
// from `Consume`: without it the ledger would answer for a propose two steps old,
// which is exactly the state `pending_drafts_`' move-out used to make
// unrepresentable. `IsFresh` alone is a lookup; the sequence is the guarantee, and
// that is what `tests/vllm/v1/worker/test_draft_fill.cpp` drives — through
// `FillDraftsForStep`, which is where the sequence is stated so that a call site
// cannot separate the fill from the consume.
//
// Keyed by req_id and not by req_state slot deliberately: it is a one-step fact
// about the last propose, so it neither moves through `condense` nor swaps
// through `swap_states`.
class ProposedDraftLedger {
 public:
  // A propose arm finished and wrote rows for these requests. Replaces whatever
  // an earlier propose left, because only the most recent one is fresh.
  void Record(const std::vector<std::string>& req_ids);
  // A propose arm finished and produced nothing. Nothing is fresh.
  void Clear();
  // Did the most recent propose write a row for this request, and has the fill
  // not already used it?
  bool IsFresh(const std::string& req_id) const;
  // The fill has used this ledger. The next fill needs a new propose.
  //
  // IT IS A PROPERTY OF THE STEP AND NOT OF THE FILL. `FillDraftsForStep` below
  // is the only caller for exactly that reason; see its comment for the state a
  // per-fill consume leaves representable.
  void Consume();
  // For tests and diagnostics; never a route.
  std::size_t size() const { return req_ids_.size(); }

 private:
  std::set<std::string> req_ids_;
};

// SPEC-DFLASH2 A2-3 REPAIR ROUND 3 (#2911): THE FILL IS A STEP, NOT A LOOP.
//
// WHAT WENT WRONG WITH THE PREVIOUS SHAPE. `execute_model` ran the per-request
// loop AND `Consume()` inside `if (use_async_scheduling_ && !sched_spec->empty())`,
// so a step with NO placeholders consumed nothing and the ledger outlived its
// step. The mechanism it replaced does not behave that way: `pending_drafts_`'
// move-out through `take_draft_token_ids` (core_proc.cpp:234) runs on EVERY
// deferred-batch step, gated on `check_for_draft_tokens_` alone and never on
// placeholders existing, and `Record` is likewise unconditional on them.
//
// THE STATE THAT LEAVES REPRESENTABLE. Step N proposes and Records {A}. Step N+1
// has no placeholders — the async decode arm that never proposes is one such
// step today (`## Owed`, #2920) — so nothing consumes and nothing Records or
// Clears. Step N+2 has placeholders, `IsFresh(A)` passes off STEP N's propose,
// and step N's drafts are spliced into step N+2's placeholders. That is verbatim
// the "a propose two steps old" state this header says the design makes
// unrepresentable, and it costs acceptance and raises nothing.
//
// NOTHING DISTINGUISHED THE TWO PLACEMENTS. Hoisting the `Consume()` out to the
// step left `test_draft_fill`, `test_mtp_depth`, `test_dflash2_runner_reach`,
// `test_runner` and `test_engine_core_proc` all green, in both placements. So the
// step is stated HERE, as one function that does the loop and the consume
// together, and the sequence is driven in `test_draft_fill.cpp`: a fill, then a
// step with no placeholders, then a fill that MUST refuse. Put the consume back
// inside the emptiness branch and that third fill splices instead of refusing.
//
// WHAT IT RETURNS. The patched copy of `scheduled_spec_decode_tokens` — the
// scheduler's `-1` placeholders replaced by this runner's real drafts, one entry
// per request that had placeholders. An empty input gives an empty result and
// still consumes, which is the whole point.
//
// The caller supplies the residence (`InputBatch::draft_tokens`, its stride, the
// req_id -> slot map and the per-slot valid counts) rather than the InputBatch
// itself, so this header keeps stating a RULE over arrays and the runner keeps
// owning the batch.
std::map<std::string, std::vector<int32_t>> FillDraftsForStep(
    ProposedDraftLedger& ledger,
    const std::map<std::string, std::vector<int32_t>>& scheduled_spec_tokens,
    const std::vector<int32_t>& draft_tokens, int draft_tokens_stride,
    const std::unordered_map<std::string, int>& req_id_to_index,
    const std::vector<int32_t>& num_valid_draft_tokens);

// ─── SPEC-DFLASH2 A2-4 (#3004): WHERE THE PROPOSE READS THIS STEP'S IDS ──────
//
// WHAT THIS IS. `sample_tokens_async` has three write-back branches and they
// disagree about one thing the propose tail cares about: whether the ids this
// step committed are readable ON THE HOST when the tail runs. The host branch
// Synchronizes and writes them itself. Both CUDA branches scatter them with a
// kernel queued on the main queue that nothing waits — the device-mirror branch
// leaves the host array's VALUES stale on purpose (reading them back is the cost
// it exists to remove), and the UMA branch writes the host array from a kernel
// that has not run yet. #2920 answered that with a `committed_ids_on_host` flag
// and a refusal by name, which is why A2-5 cannot flip the veto: on GB10's
// integrated default the UMA branch runs and hits it.
//
// This function is the answer that makes the refusal unnecessary rather than
// deleting it. The runner materializes the step's committed ids ONCE, on every
// branch, and hands them here; the two pieces of host state every propose arm
// reads are then written from that ONE array:
//
//   * `last_sampled_tokens[slot]` — the anchor `propose_drafts_block` takes;
//   * `token_ids_cpu[slot * max_model_len + num_tokens_no_spec[slot] - 1]` —
//     the column `propose_drafts_ngram` matches over. The async arm advances
//     `num_tokens_no_spec` without writing that column (the token-VALUE append
//     is deleted on purpose: the scheduler's `update_from_output` feeds detok
//     and penalties when `get_output()` materializes), so left unwritten the
//     matcher reads a zero where this step's token belongs and drafts off it.
//
// Upstream writes the same pair from the same place: `req_states.last_sampled`
// (gpu/states.py) and `req_states.token_ids` are both updated from the sampler's
// output in `_update_states` / `post_update`, and its n-gram proposer reads the
// token row (`gpu_model_runner.py:1494-1500`, and the GPU n-gram tensors at
// `:1519-1525` @ pin 5559679229bc961848b121ccdeaa8fa5d79bec98).
//
// WHY IT IS A NAMED FUNCTION AND NOT A LOOP AT THE CALL SITE. A per-request rule
// feeding a per-STEP route is the shape that shipped a silent wrong answer on
// this row: `tests/vllm/v1/worker/test_combine_row_predicate.cpp` (#2710) records
// a version that survived 27 mutations because every case used `num_reqs == 1`.
// Stating the rule once, over arrays, is what lets a MIXED step — some rows
// committing, some discarded, different `num_tokens_no_spec` per row — be an
// input rather than a reading.
//
// WHAT IT REFUSES, and each refusal is a case a test drives. A committed row
// whose `num_tokens_no_spec` is not positive, and a committed row whose write
// column falls outside `token_ids_cpu`. Both were SILENT `continue`s at the call
// site #2920 landed, and a fail-open skip is indistinguishable from a fail-closed
// one until something mutates it — the finding three A2-3 reviews each made.
//
// `discard` is the step's `discard_request_mask` and MAY be shorter than
// `num_reqs` (the runner builds it per step); a row past its end is treated as
// committing, exactly as every other loop over it in `runner.cpp` does. A
// discarded row is left BYTE-IDENTICAL in both arrays: it produced no output
// token this step, its `num_tokens_no_spec` was not advanced, and writing its
// column would corrupt the newest token it actually committed.
void ApplyCommittedIdsForPropose(const int64_t* committed_ids, int num_reqs,
                                 const std::vector<uint8_t>& discard,
                                 const std::vector<int32_t>& num_tokens_no_spec,
                                 int max_model_len,
                                 std::vector<int32_t>* last_sampled_tokens,
                                 std::vector<int32_t>* token_ids_cpu);

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_PREPARE_INPUTS_H_
