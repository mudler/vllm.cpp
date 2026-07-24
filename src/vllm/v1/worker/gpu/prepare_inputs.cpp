// Ported from: vllm/v1/worker/gpu_model_runner.py::_prepare_inputs @ e24d1b24
// (+ _update_states, _get_cumsum_and_arange, and the num_scheduled_tokens_np
//  build from execute_model). See include/vllm/v1/worker/gpu/prepare_inputs.h
// for the scope, the V1-algorithm / MRV2-contract note, and the deferred paths.

#include "vllm/v1/worker/gpu/prepare_inputs.h"

#include <cassert>
#include <cstddef>
#include <map>
#include <string>
#include <unordered_set>

namespace vllm::v1 {

// ─── update_states ──────────────────────────────────────────────────────────
// Ported from gpu_model_runner.py::_update_states @ e24d1b24 (the MRV2-contract
// subset: finished/unscheduled removal, new-request admission, in-batch cached
// diffs, condense; PP / spec / async / resumed-store paths deferred).
void update_states(InputBatch& input_batch,
                   const SchedulerOutput& scheduler_output) {
  // Remove the finished requests from the persistent batch.
  for (const std::string& req_id : scheduler_output.finished_req_ids) {
    input_batch.remove_request(req_id);
  }

  // Remove the unscheduled requests from the persistent batch:
  //   unscheduled = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
  // i.e. a cached req that is either not scheduled this step, or is scheduled
  // as a resumed-from-preemption req (which is re-admitted via add_request).
  const std::set<std::string>& resumed_req_ids =
      scheduler_output.scheduled_cached_reqs.resumed_req_ids;
  std::vector<std::string> to_remove;
  for (const auto& [req_id, index] : input_batch.req_id_to_index) {
    (void)index;
    const bool scheduled =
        scheduler_output.num_scheduled_tokens.count(req_id) != 0;
    const bool resumed = resumed_req_ids.count(req_id) != 0;
    if (!scheduled || resumed) {
      to_remove.push_back(req_id);
    }
  }
  for (const std::string& req_id : to_remove) {
    input_batch.remove_request(req_id);
  }

  // Collect the new requests to admit (from the MRV2 prefill_token_ids seed).
  // Resumed-from-preemption requests arrive here as new under our MRV2
  // scheduler (resumed-as-new), so there is a single admission path.
  std::vector<CachedRequestState> reqs_to_add;
  reqs_to_add.reserve(scheduler_output.scheduled_new_reqs.size());
  for (const NewRequestData& new_req : scheduler_output.scheduled_new_reqs) {
    reqs_to_add.push_back(CachedRequestState::from_new_request(new_req));
  }

  // Apply the cached diffs (num_computed_tokens + new_block_ids) to requests
  // still in the persistent batch. A cached req NOT in the batch is a resumed /
  // preempted req whose stored CachedRequestState the runner would re-admit —
  // deferred here (see the header); under our MRV2 scheduler it re-arrives via
  // scheduled_new_reqs.
  const CachedRequestData& cached = scheduler_output.scheduled_cached_reqs;
  for (int i = 0; i < cached.num_reqs(); ++i) {
    const std::string& req_id = cached.req_ids[static_cast<size_t>(i)];
    const auto it = input_batch.req_id_to_index.find(req_id);
    if (it == input_batch.req_id_to_index.end()) {
      continue;  // deferred resumed-store re-admission
    }
    const int req_index = it->second;
    input_batch.num_computed_tokens_cpu[static_cast<size_t>(req_index)] =
        cached.num_computed_tokens[static_cast<size_t>(i)];
    const std::optional<std::vector<std::vector<int>>>& new_block_ids =
        cached.new_block_ids[static_cast<size_t>(i)];
    if (new_block_ids.has_value()) {
      input_batch.block_table.append_row(*new_block_ids, req_index);
    }
  }

  // Add the new (or resumed-as-new) requests. Smaller empty indices first.
  for (const CachedRequestState& request : reqs_to_add) {
    input_batch.add_request(request);
  }

  // Condense to close any gaps left by removed requests.
  input_batch.condense();
}

// ─── prepare_inputs ─────────────────────────────────────────────────────────
// Ported from gpu_model_runner.py::_prepare_inputs @ e24d1b24 (host-array T0
// subset). The GPU staging, PP/spec/async fast paths, M-RoPE, and the
// CUDA-graph query_start_loc/seq_lens padding are deferred (see the header).
StepInputs prepare_inputs(InputBatch& input_batch,
                          const SchedulerOutput& scheduler_output) {
  const int total_num_scheduled_tokens =
      scheduler_output.total_num_scheduled_tokens;
  assert(total_num_scheduled_tokens > 0);
  const int num_reqs = input_batch.num_reqs();
  assert(num_reqs > 0);

  // OPTIMIZATION (upstream): copy the block table to the "device" buffer first;
  // compute_slot_mapping reads it.
  input_batch.block_table.commit_block_table(num_reqs);

  StepInputs out;

  // num_scheduled_tokens in batch order (execute_model builds this np array from
  // scheduler_output.num_scheduled_tokens[req_id] over input_batch.req_ids).
  out.num_scheduled_tokens.resize(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = *input_batch.req_ids[static_cast<size_t>(i)];
    out.num_scheduled_tokens[static_cast<size_t>(i)] =
        scheduler_output.num_scheduled_tokens.at(req_id);
  }

  const int total = total_num_scheduled_tokens;

  // req_indices = repeat(arange(num_reqs), num_scheduled_tokens);
  // query_pos   = the batched arange (0..n-1 per req) from _get_cumsum_and_arange;
  // query_start_loc = [0] ++ cumsum(num_scheduled_tokens).
  std::vector<int> req_indices(static_cast<size_t>(total));
  std::vector<int64_t> query_pos(static_cast<size_t>(total));
  out.query_start_loc.assign(static_cast<size_t>(num_reqs) + 1, 0);
  int offset = 0;
  int cumulative = 0;
  for (int i = 0; i < num_reqs; ++i) {
    const int n = out.num_scheduled_tokens[static_cast<size_t>(i)];
    for (int j = 0; j < n; ++j) {
      req_indices[static_cast<size_t>(offset)] = i;
      query_pos[static_cast<size_t>(offset)] = j;
      ++offset;
    }
    cumulative += n;
    out.query_start_loc[static_cast<size_t>(i) + 1] = cumulative;
  }

  // positions[t] = num_computed_tokens_cpu[req_indices[t]] + query_pos[t];
  // input_token_ids[t] = token_ids_cpu[req_indices[t], positions[t]].
  out.positions.resize(static_cast<size_t>(total));
  out.input_token_ids.resize(static_cast<size_t>(total));
  for (int t = 0; t < total; ++t) {
    const int r = req_indices[static_cast<size_t>(t)];
    const int64_t pos =
        static_cast<int64_t>(
            input_batch.num_computed_tokens_cpu[static_cast<size_t>(r)]) +
        query_pos[static_cast<size_t>(t)];
    out.positions[static_cast<size_t>(t)] = pos;
    out.input_token_ids[static_cast<size_t>(t)] =
        input_batch.token_id(r, static_cast<int>(pos));
  }

  // seq_lens[i] = num_computed_tokens_cpu[i] + num_scheduled_tokens[i].
  out.seq_lens.resize(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    out.seq_lens[static_cast<size_t>(i)] =
        input_batch.num_computed_tokens_cpu[static_cast<size_t>(i)] +
        out.num_scheduled_tokens[static_cast<size_t>(i)];
  }

  // slot_mapping per KV cache group: block_id*block_size + within-block offset.
  input_batch.block_table.compute_slot_mapping(num_reqs, out.query_start_loc,
                                               out.positions);
  const auto& groups = input_batch.block_table.block_tables;
  out.slot_mapping.resize(groups.size());
  for (size_t g = 0; g < groups.size(); ++g) {
    const std::vector<int64_t>& full = groups[g].slot_mapping();
    out.slot_mapping[g].assign(full.begin(),
                               full.begin() + static_cast<std::ptrdiff_t>(total));
  }

  // ─── logits expansion (SPEC-REJECTION I3) ─────────────────────────────────
  // Mirrors gpu/model_runner.py:866-898 (the cu_num_logits build) + the
  // logits-index formula of _combine_sampled_and_draft_tokens_kernel:317-327
  // (`logits_start = query_end - num_logits`).
  const std::map<std::string, std::vector<int32_t>>& draft_tokens =
      scheduler_output.scheduled_spec_decode_tokens;
  if (draft_tokens.empty()) {
    // NO DRAFT TOKEN SCHEDULED (the common case, and the production default —
    // the scheduler only populates the map behind a configured speculator).
    // cu_num_logits = arange(num_reqs + 1); logits_indices = query_start_loc[1:]
    // - 1 (last scheduled token per seq) — byte-identical to the pre-I3 array.
    out.num_draft_tokens = 0;
    out.cu_num_logits.resize(static_cast<size_t>(num_reqs) + 1);
    for (int i = 0; i <= num_reqs; ++i) {
      out.cu_num_logits[static_cast<size_t>(i)] = i;
    }
    out.logits_indices.resize(static_cast<size_t>(num_reqs));
    for (int i = 0; i < num_reqs; ++i) {
      out.logits_indices[static_cast<size_t>(i)] =
          out.query_start_loc[static_cast<size_t>(i) + 1] - 1;
    }
  } else {
    // num_logits[i] = num_draft_tokens_per_req[i] + num_bonus_tokens; the bonus
    // count is 1 for every model we run (upstream
    // model_state.num_new_sampled_tokens_per_step; > 1 only for multi-bonus
    // heads, deferred).
    constexpr int kNumBonusTokens = 1;
    out.num_draft_tokens_per_req.resize(static_cast<size_t>(num_reqs));
    out.cu_num_logits.resize(static_cast<size_t>(num_reqs) + 1);
    out.cu_num_logits[0] = 0;
    int total_num_logits = 0;
    int total_num_draft_tokens = 0;
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& req_id = *input_batch.req_ids[static_cast<size_t>(i)];
      const auto it = draft_tokens.find(req_id);
      const int k = it == draft_tokens.end() ? 0 : static_cast<int>(it->second.size());
      out.num_draft_tokens_per_req[static_cast<size_t>(i)] = k;
      total_num_draft_tokens += k;
      total_num_logits += k + kNumBonusTokens;
      out.cu_num_logits[static_cast<size_t>(i) + 1] = total_num_logits;
    }
    out.num_draft_tokens = total_num_draft_tokens;
    out.logits_indices.reserve(static_cast<size_t>(total_num_logits));
    for (int i = 0; i < num_reqs; ++i) {
      const int num_logits = out.cu_num_logits[static_cast<size_t>(i) + 1] -
                             out.cu_num_logits[static_cast<size_t>(i)];
      const int32_t query_end = out.query_start_loc[static_cast<size_t>(i) + 1];
      const int32_t logits_start = query_end - num_logits;
      for (int j = 0; j < num_logits; ++j) {
        out.logits_indices.push_back(logits_start + j);
      }
    }
  }

  return out;
}

// ─── combine_sampled_and_draft_tokens ───────────────────────────────────────
// Ported from vllm/v1/worker/gpu/input_batch.py::combine_sampled_and_draft_tokens
// + _combine_sampled_and_draft_tokens_kernel @ e24d1b24 (T0 non-spec subset).
// See prepare_inputs.h for the async-scheduling contract, the device-neutral /
// capture-safety note, and the deferred spec-decode draft lane.
std::vector<int32_t> combine_sampled_and_draft_tokens(
    std::vector<int32_t>& input_token_ids,
    const std::vector<int32_t>& idx_mapping,
    const std::vector<int32_t>& last_sampled_tokens,
    const std::vector<int32_t>& query_start_loc,
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& prefill_len, int num_new_sampled_tokens) {
  // Upstream asserts num_new_sampled_tokens in (0, 1): the bonus token, excl.
  // accepted draft tokens.
  assert(num_new_sampled_tokens == 0 || num_new_sampled_tokens == 1);
  const int num_reqs = static_cast<int>(idx_mapping.size());

  // cu_num_logits is one logit per request at T0 (no draft tokens), so
  // num_logits == num_new_sampled_tokens per row and the total == num_reqs *
  // num_new_sampled_tokens. logits_indices[b] = query_end - num_logits (+block).
  std::vector<int32_t> logits_indices;
  logits_indices.reserve(
      static_cast<size_t>(num_reqs) * static_cast<size_t>(num_new_sampled_tokens));

  for (int batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int req_state_idx = idx_mapping[static_cast<size_t>(batch_idx)];
    // num_logits == num_new_sampled_tokens (num_draft_tokens == 0 at T0).
    const int num_logits = num_new_sampled_tokens;
    const int32_t query_end =
        query_start_loc[static_cast<size_t>(batch_idx) + 1];
    const int32_t logits_start = query_end - num_logits;
    for (int j = 0; j < num_logits; ++j) {
      logits_indices.push_back(logits_start + j);
    }

    // seq_len <= prefill_len: still consuming the known prefill tokens (incl.
    // the chunk that exactly completes prefill) — no sampled or draft token to
    // splice; the prompt token in input_token_ids stays.
    const int32_t seq_len = seq_lens[static_cast<size_t>(batch_idx)];
    const int32_t pf = prefill_len[static_cast<size_t>(req_state_idx)];
    if (seq_len <= pf) {
      continue;
    }
    // Write the last sampled token id at the decode position (query_end -
    // num_logits). num_new_sampled_tokens == 0 (draft-only step) writes nothing.
    if (num_new_sampled_tokens > 0) {
      input_token_ids[static_cast<size_t>(query_end - num_logits)] =
          last_sampled_tokens[static_cast<size_t>(req_state_idx)];
    }
    // Draft tokens (num_draft_tokens > 0) are deferred with SPEC-MTP.
  }
  return logits_indices;
}

}  // namespace vllm::v1
