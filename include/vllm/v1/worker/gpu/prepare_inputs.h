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

#include <cstdint>
#include <vector>

#include "vllm/v1/core/sched/output.h"
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
  // Index (into the flattened token stream) of each sequence's last scheduled
  // token: query_start_loc[1:] - 1. [num_reqs]
  std::vector<int32_t> logits_indices;
  // num_scheduled_tokens per request in batch order (the array upstream's
  // execute_model builds and passes to _prepare_inputs). [num_reqs]
  std::vector<int32_t> num_scheduled_tokens;
};

// update_states: apply a SchedulerOutput to the persistent InputBatch — remove
// finished + unscheduled requests, admit the scheduled new requests
// (from_new_request -> add_request), apply the cached diffs (num_computed_tokens
// + new_block_ids append) for requests still in the batch, then condense().
// Matches gpu_model_runner.py::_update_states ordering (finished/unscheduled
// removal -> collect new -> cached diffs -> add_request -> condense). The
// deferred PP / spec / async / resumed-store paths are documented in the header.
void update_states(InputBatch& input_batch,
                   const SchedulerOutput& scheduler_output);

// prepare_inputs: build the flattened StepInputs from the (already
// update_states'd) persistent InputBatch + the scheduler output. Mirrors
// gpu_model_runner.py::_prepare_inputs (commit_block_table -> req_indices /
// cumsum+arange -> positions -> token gather -> query_start_loc -> seq_lens ->
// compute_slot_mapping -> logits_indices). Requires num_reqs > 0 and
// total_num_scheduled_tokens > 0 (upstream asserts both).
StepInputs prepare_inputs(InputBatch& input_batch,
                          const SchedulerOutput& scheduler_output);

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_PREPARE_INPUTS_H_
