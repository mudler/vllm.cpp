// Ported from: vllm/v1/worker/gpu_input_batch.py @ e24d1b24
//
// Scope (M1.5 Task 2): the PERSISTENT per-slot input batch the model runner
// keeps alive across steps — `CachedRequestState` (the worker's cached copy of
// a scheduled request) and `InputBatch` (the num_reqs-major arrays mutated in
// place as requests are admitted / finished / condensed). Two ported types:
//   * CachedRequestState: req_id, prompt_token_ids, sampling_params, the
//     per-KV-cache-group block_ids, num_computed_tokens, output_token_ids, and
//     the num_prompt_tokens / num_tokens / get_token_id helpers.
//   * InputBatch: the per-slot arrays (req_ids indexed by slot, token_ids_cpu,
//     num_tokens_no_spec, num_prompt_tokens, num_computed_tokens_cpu, the
//     MultiGroupBlockTable, and the per-slot sampling metadata the M1.7 sampler
//     needs — temperature / top_p / top_k / {frequency,presence,repetition}
//     penalties) + the req_id_to_index map, add_request / remove_request /
//     condense (swap-remove densification), and num_reqs.
// Behavioral only: no CUDA, no model. The "tensors" are plain host arrays.
//
// ─── V1 ALGORITHM / MRV2 CONTRACT (recorded) ────────────────────────────────
// The source of truth is the V1 GPU worker file `gpu_input_batch.py` (NOT the
// MRV2 `gpu/input_batch.py`, which at e24d1b24 is a TRANSIENT per-step object
// that holds NO block table; its persistent state is split into `req_states` +
// staged `BlockTables` on the runner — deferred to M2 with the `vt` device).
// The persistent-InputBatch-holds-the-MultiGroupBlockTable design ported here
// is the identical bookkeeping minus the device wrapper. But it is DRIVEN by
// the MRV2 scheduler-output CONTRACT, not the MRV1 runner's admission path:
//   * new requests carry `prefill_token_ids` (= all_token_ids = prompt+output
//     at schedule time; NewRequestData::prefill_token_ids, M1.4). The token-id
//     seed of `token_ids_cpu` is prompt_token_ids then output_token_ids, which
//     by construction equals prefill_token_ids — see from_new_request.
//   * resumed-from-preemption requests are folded in AS NEW (they arrive in
//     scheduled_new_reqs), so there is a single admission path: add_request.
//   * cached diffs are `num_computed_tokens` + `new_block_ids` (Task 3 applies
//     them; add_request seeds the initial state).
// The V1 runner's MRV1-shape admission (`resumed_from_preemption` /
// `resumed_req_ids` / per-req `all_token_ids` reconstruction) is DEAD under our
// MRV2 scheduler and is NOT ported.
//
// The C++ file location follows the repo's `worker/gpu/` layout (as with Task
// 1's block_table); the `Ported from` ref above cites the ACTUAL upstream file.
//
// ─── HOST-ARRAY-FOR-DEVICE-TENSOR DEVIATION (recorded) ──────────────────────
// Upstream keeps paired cpu/device tensors (token_ids_cpu numpy view over a
// cpu tensor; temperature/top_p/... as device tensors with pinned cpu staging
// buffers copied via copy_slice in _make_sampling_metadata). At T0 there is no
// device: every array is a host std::vector. The per-slot "*_cpu" arrays are
// what add_request / condense write; the device-side sampling tensors + the
// copy_slice staging are the sampler's (M1.7) concern and are not materialized
// here. The boolean predicates (all_greedy / no_top_p / ...) the sampler keys
// on ARE ported so M1.7 can build SamplingMetadata from these arrays.
//
// ─── DEFERRED upstream slot state (marked; T0 never populates) ──────────────
//   - LoRA: request_lora_mapping / lora_id_to_request_ids / lora_id_to_lora_
//     request (+ make_lora_inputs).
//   - Spec decode: spec_token_ids is kept as a per-slot list so the
//     _get_active_token_count / condense swap match upstream, but it is ALWAYS
//     empty at T0 (update_req_spec_token_ids / num_accepted_tokens deferred).
//   - Multimodal: mm_features / req_prompt_embeds / prompt_is_token_ids /
//     is_token_ids.
//   - Structured output / logits processors: batch_update_builder's added/moved
//     tracking, logitsprocs, allowed_token_ids_mask, bad_words_token_ids,
//     thinking-budget. Only the REMOVED-index tracking (which drives add-hole
//     fill + condense) is ported — as a minimal RemovedTracker.
//   - Logprobs: num_logprobs / logprob_token_ids.
//   - Pooling: pooling_params / pooling_states / is_pooling_model branch.
//   - Async scheduling: prev_sampled_token_ids / update_async_* .
//   - SamplingMetadata construction (_make_sampling_metadata / refresh_metadata)
//     is deferred to M1.7 — the SamplingMetadata type is not yet landed; the
//     per-slot arrays + boolean predicates it consumes ARE provided here.
#ifndef VLLM_V1_WORKER_GPU_INPUT_BATCH_H_
#define VLLM_V1_WORKER_GPU_INPUT_BATCH_H_

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/sched/output.h"  // NewRequestData (from_new_request)
#include "vllm/v1/worker/gpu/block_table.h"

namespace vllm::v1 {

// The worker's cached, persistent copy of a scheduled request.
// (Upstream: vllm/v1/worker/gpu_input_batch.py CachedRequestState — T0 field
// subset; mm_features / generator / lora / prompt_embeds / pooling DEFERRED.)
struct CachedRequestState {
  std::string req_id;
  // Upstream list[int] | None; always the token path at T0.
  std::vector<int32_t> prompt_token_ids;
  // Upstream SamplingParams | None; always present for generation (pooling
  // DEFERRED). Stored by value, already PostInit'd by the frontend.
  SamplingParams sampling_params;
  // Per-KV-cache-group block ids (upstream tuple[list[int], ...]).
  std::vector<std::vector<int>> block_ids;
  int num_computed_tokens = 0;
  std::vector<int32_t> output_token_ids;

  // num_prompt_tokens (upstream __post_init__ via
  // length_from_prompt_token_ids_or_embeds); no prompt_embeds at T0.
  int num_prompt_tokens = 0;

  // Build a CachedRequestState from the MRV2 NewRequestData contract. The seed
  // is prefill_token_ids (= all_token_ids): prompt_token_ids is the prompt,
  // output_token_ids is the tail of prefill_token_ids beyond the prompt. This
  // is the MRV2-contract adaptation of the V1 algorithm (the V1 runner instead
  // reconstructed all_token_ids per-req; that path is dead here).
  static CachedRequestState from_new_request(const NewRequestData& new_req);

  // num_tokens: num_prompt_tokens + len(output_token_ids).
  int num_tokens() const {
    return num_prompt_tokens + static_cast<int>(output_token_ids.size());
  }

  // get_token_id(idx): prompt then output; -1 past the end (upstream).
  int get_token_id(int idx) const;

  // Recompute num_prompt_tokens from prompt_token_ids (call after populating
  // fields when not using from_new_request). Mirrors __post_init__.
  void finalize() { num_prompt_tokens = static_cast<int>(prompt_token_ids.size()); }
};

// The persistent per-slot input batch. (Upstream: vllm/v1/worker/
// gpu_input_batch.py InputBatch — T0 subset.)
class InputBatch {
 public:
  // Upstream positional order: max_num_reqs, max_model_len,
  // max_num_batched_tokens, (device dropped), vocab_size, block_sizes,
  // kernel_block_sizes, (trailing knobs DEFERRED). block_sizes /
  // kernel_block_sizes are per KV cache group.
  InputBatch(int max_num_reqs, int max_model_len, int max_num_batched_tokens,
             int vocab_size, std::vector<int> block_sizes,
             std::vector<int> kernel_block_sizes);

  // add_request: place `request` into a slot (a freed hole if one exists, else
  // append at num_reqs), fill the per-slot arrays from it, add the block-table
  // rows, and record the sampling metadata. Returns the assigned slot index.
  int add_request(const CachedRequestState& request);

  // remove_request: mark the slot for `req_id` empty and return its index, or
  // nullopt if unknown. MUST be followed by condense() before the next read of
  // the dense [0, num_reqs) range (upstream contract).
  std::optional<int> remove_request(const std::string& req_id);

  // condense: slide active requests down into the freed holes so [0, num_reqs)
  // is dense again, fixing req_id_to_index and swapping/moving every per-slot
  // array + the block-table rows. (Upstream InputBatch.condense.)
  void condense();

  // num_reqs (property): len(req_id_to_index).
  int num_reqs() const { return static_cast<int>(req_id_to_index.size()); }

  // Sampling predicates the M1.7 sampler keys on (upstream properties).
  bool all_greedy() const { return random_reqs.empty(); }
  bool all_random() const { return greedy_reqs.empty(); }
  bool no_top_p() const { return top_p_reqs.empty(); }
  bool no_top_k() const { return top_k_reqs.empty(); }
  bool no_penalties() const {
    return presence_penalties_reqs.empty() &&
           frequency_penalties_reqs.empty() &&
           repetition_penalties_reqs.empty();
  }

  // Convenience token-id read (flat row-major token_ids_cpu; row stride is
  // max_model_len). Mirrors the numpy token_ids_cpu[req, col] access.
  int32_t token_id(int req_index, int col) const {
    return token_ids_cpu[static_cast<size_t>(req_index) * max_model_len + col];
  }

  // ─── Public state (mirrors upstream's accessible attributes) ──────────────
  int max_num_reqs;
  int max_model_len;
  int max_num_batched_tokens;
  int vocab_size;

  // req_ids indexed by slot; nullopt only transiently during remove/condense.
  std::vector<std::optional<std::string>> req_ids;
  std::unordered_map<std::string, int> req_id_to_index;

  // Per-slot token buffer, flat [max_num_reqs, max_model_len] row-major.
  std::vector<int32_t> token_ids_cpu;
  // Per-slot scalar arrays [max_num_reqs].
  std::vector<int32_t> num_tokens_no_spec;
  std::vector<int32_t> num_prompt_tokens;
  std::vector<int32_t> num_computed_tokens_cpu;

  // The per-request KV-cache block table (one BlockTable per group).
  MultiGroupBlockTable block_table;

  // Per-slot sampling metadata [max_num_reqs] (M1.7 sampler input).
  std::vector<float> temperature_cpu;
  std::vector<float> top_p_cpu;
  std::vector<int32_t> top_k_cpu;
  std::vector<float> frequency_penalties_cpu;
  std::vector<float> presence_penalties_cpu;
  std::vector<float> repetition_penalties_cpu;

  // Membership sets driving the sampling predicates (upstream *_reqs sets).
  std::unordered_map<std::string, char> greedy_reqs;
  std::unordered_map<std::string, char> random_reqs;
  std::unordered_map<std::string, char> top_p_reqs;
  std::unordered_map<std::string, char> top_k_reqs;
  std::unordered_map<std::string, char> frequency_penalties_reqs;
  std::unordered_map<std::string, char> presence_penalties_reqs;
  std::unordered_map<std::string, char> repetition_penalties_reqs;

  // Per-slot output token ids (nullopt on a freed slot). Consumed by the M1.7
  // sampler for penalties; mirrors upstream req_output_token_ids.
  std::vector<std::optional<std::vector<int32_t>>> req_output_token_ids;

  // Per-slot speculative token ids. DEFERRED: always empty at T0, kept so
  // _get_active_token_count / condense swap match upstream exactly.
  std::vector<std::vector<int32_t>> spec_token_ids;

 private:
  // Minimal port of BatchUpdateBuilder's REMOVED-index tracking — the only part
  // add_request (hole fill) + condense (densification) depend on. `removed`
  // returns the freed indices sorted DESCENDING; peek/pop return the LOWEST.
  // (The added/moved logitsprocs tracking is deferred.)
  class RemovedTracker {
   public:
    void removed_append(int index);
    bool has_removed() const { return !removed_.empty(); }
    // Lowest removed index (or nullopt); does not pop.
    std::optional<int> peek_removed();
    // Pop + return the lowest removed index (or nullopt).
    std::optional<int> pop_removed();
    // Freed indices sorted descending (upstream `removed` property).
    const std::vector<int>& removed();

   private:
    void ensure_sorted();
    std::vector<int> removed_;
    bool is_sorted_ = false;
  };

  // _register_add_request: pick the slot (freed hole, else num_reqs).
  int register_add_request();
  // _get_active_token_count: num_tokens_no_spec[i] + len(spec_token_ids[i]).
  int get_active_token_count(int req_index) const;

  RemovedTracker removed_tracker_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_INPUT_BATCH_H_
