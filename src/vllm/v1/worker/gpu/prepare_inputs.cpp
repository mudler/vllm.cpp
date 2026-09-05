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

#include "vt/dtype.h"  // VT_CHECK

namespace vllm::v1 {

// ─── update_states ──────────────────────────────────────────────────────────
// Ported from gpu_model_runner.py::_update_states @ e24d1b24 (the MRV2-contract
// subset: finished/unscheduled removal, new-request admission, in-batch cached
// diffs, condense; PP / spec / async / resumed-store paths deferred).
void update_states(InputBatch& input_batch,
                   const SchedulerOutput& scheduler_output,
                   std::unordered_map<std::string, CachedRequestState>*
                       req_states) {
  // Remove the finished requests from the persistent batch.
  for (const std::string& req_id : scheduler_output.finished_req_ids) {
    input_batch.remove_request(req_id);
    // ENG-MM-INPUT-PIPELINE P2 (#2379): upstream's `self.requests.pop(req_id)`
    // (_update_states, the finished-ids loop). Only FINISHED ids leave the map:
    // a request that is merely unscheduled this step keeps its entry, because
    // the next step gathers from it again. Null on every text engine, and the
    // three lines this pointer guards are then not executed at all.
    if (req_states != nullptr) req_states->erase(req_id);
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

  // ENG-MM-INPUT-PIPELINE P2 (#2379): the per-REQUEST map upstream keeps beside
  // the per-SLOT batch (`self.requests[req_id] = CachedRequestState(...)` in
  // _update_states, and the `req_state.num_computed_tokens = ...` line in its
  // cached-diff loop). The runner's encoder step, gather and M-RoPE all address
  // state by REQUEST ID and none of them can use a slot index, because a slot is
  // reused by a different request after a condense. A resumed-from-preemption
  // request re-arrives through scheduled_new_reqs under this scheduler, so the
  // assignment below deliberately OVERWRITES: its mm_features are the same items
  // and its M-RoPE prompt array is recomputed by the runner from the same prompt.
  if (req_states != nullptr) {
    for (const CachedRequestState& request : reqs_to_add) {
      (*req_states)[request.req_id] = request;
    }
    for (int i = 0; i < cached.num_reqs(); ++i) {
      const std::string& req_id = cached.req_ids[static_cast<size_t>(i)];
      const auto it = req_states->find(req_id);
      if (it == req_states->end()) continue;
      it->second.num_computed_tokens =
          cached.num_computed_tokens[static_cast<size_t>(i)];
    }
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

  // ─── prompt logprobs (SAMPLE-PROMPT-LOGPROBS) ──────────────────────────────
  // 1:1 the per-request row selection of _get_prompt_logprobs_dict
  // (gpu_model_runner.py:5626-5686). Skipped entirely — not one branch taken,
  // not one byte appended — unless a request asked for prompt logprobs.
  if (!input_batch.num_prompt_logprobs.empty()) {
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& req_id = *input_batch.req_ids[static_cast<size_t>(i)];
      const auto it = input_batch.num_prompt_logprobs.find(req_id);
      if (it == input_batch.num_prompt_logprobs.end()) continue;

      const int32_t num_tokens = out.num_scheduled_tokens[static_cast<size_t>(i)];
      const int32_t num_prompt_tokens =
          input_batch.num_prompt_tokens[static_cast<size_t>(i)];
      // start_idx = num_computed_tokens; start_tok = start_idx + 1 (:5654-5655).
      // The first prompt position has no logprob (nothing precedes it), so the
      // tensor covers num_prompt_tokens - 1 rows and row r scores prompt[r+1].
      const int32_t start_idx =
          input_batch.num_computed_tokens_cpu[static_cast<size_t>(i)];
      const int32_t start_tok = start_idx + 1;
      const int32_t num_remaining_tokens = num_prompt_tokens - start_tok;

      StepInputs::PromptLogprobRows rows;
      rows.req_id = req_id;
      rows.num_prompt_logprobs = it->second;
      rows.dst_start = start_idx;
      if (num_tokens <= num_remaining_tokens) {
        // A chunk; more prompt remains. In the == case there are no more prompt
        // logprobs to produce, but upstream still defers the emit to the next
        // step, where there is a generated token to return alongside them
        // (:5658-5662).
        rows.num_rows = num_tokens;
      } else {
        rows.num_rows = num_remaining_tokens;
        rows.final_chunk = true;
      }
      // num_rows <= 0 is the exact-prefill edge (:5668-5671): the previous step
      // consumed exactly num_prompt_tokens - 1 tokens, so there is nothing left
      // to score. A final chunk still has to be EMITTED, so the entry is kept
      // with zero rows and no gathered indices.
      if (rows.num_rows < 0) rows.num_rows = 0;
      if (rows.num_rows == 0 && !rows.final_chunk) continue;

      rows.src_start = static_cast<int>(out.prompt_logprob_indices.size());
      const int32_t query_start = out.query_start_loc[static_cast<size_t>(i)];
      rows.target_token_ids.reserve(static_cast<size_t>(rows.num_rows));
      for (int j = 0; j < rows.num_rows; ++j) {
        out.prompt_logprob_indices.push_back(query_start + j);
        rows.target_token_ids.push_back(
            static_cast<int64_t>(input_batch.token_id(i, start_tok + j)));
      }
      out.prompt_logprob_rows.push_back(std::move(rows));
    }
  }

  return out;
}

// ─── combine_sampled_and_draft_tokens ───────────────────────────────────────
// Ported from vllm/v1/worker/gpu/input_batch.py::combine_sampled_and_draft_tokens
// (:364-406) + _combine_sampled_and_draft_tokens_kernel (:303-361) @ the parity
// pin 5559679229bc961848b121ccdeaa8fa5d79bec98. The Triton kernel is one program
// per request; this is the same body as a host loop over batch rows.
// See prepare_inputs.h for the async-scheduling contract, the device-neutral /
// capture-safety note, the one harness adaptation, and the reachability record.
std::vector<int32_t> combine_sampled_and_draft_tokens(
    std::vector<int32_t>& input_token_ids,
    const std::vector<int32_t>& idx_mapping,
    const std::vector<int32_t>& last_sampled_tokens,
    const std::vector<int32_t>& query_start_loc,
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& prefill_len,
    const std::vector<int32_t>& draft_tokens, int draft_tokens_stride,
    const std::vector<int32_t>& cu_num_logits, int num_new_sampled_tokens) {
  // Upstream asserts num_new_sampled_tokens in (0, 1): the bonus token, excl.
  // accepted draft tokens (:376-378).
  assert(num_new_sampled_tokens == 0 || num_new_sampled_tokens == 1);
  const int num_reqs = static_cast<int>(idx_mapping.size());
  // cu_num_logits is the ONLY source of the per-request num_logits (:322-324),
  // so a caller that cannot supply it cannot call this function.
  VT_CHECK(static_cast<int>(cu_num_logits.size()) == num_reqs + 1,
           "combine_sampled_and_draft_tokens: cu_num_logits must hold num_reqs "
           "+ 1 entries");

  // Upstream allocates logits_indices at the total num_logits (:383-387) and the
  // kernel stores each row's block at cu_num_logits[b] (:331-335). Same here:
  // sized from cu_num_logits.back(), written at the same offsets, so the array
  // is index-for-index the upstream one.
  const int total_num_logits = cu_num_logits[static_cast<size_t>(num_reqs)];
  std::vector<int32_t> logits_indices(static_cast<size_t>(total_num_logits));

  for (int batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int req_state_idx = idx_mapping[static_cast<size_t>(batch_idx)];

    // Get the number of logits and draft tokens (:321-325). num_logits is
    // 1 + k_i on a verify step and num_new_sampled_tokens on a decode step; the
    // difference is exactly the draft count.
    const int cu_start = cu_num_logits[static_cast<size_t>(batch_idx)];
    const int cu_end = cu_num_logits[static_cast<size_t>(batch_idx) + 1];
    const int num_logits = cu_end - cu_start;
    const int num_draft_tokens = num_logits - num_new_sampled_tokens;

    // Compute the logits indices (:327-335).
    const int32_t query_end =
        query_start_loc[static_cast<size_t>(batch_idx) + 1];
    const int32_t logits_start = query_end - num_logits;
    for (int j = 0; j < num_logits; ++j) {
      logits_indices[static_cast<size_t>(cu_start + j)] = logits_start + j;
    }

    // seq_len <= prefill_len: still consuming the known prefill tokens (incl.
    // the chunk that exactly completes prefill) — no sampled or draft token to
    // splice; the prompt token in input_token_ids stays (:337-341).
    const int32_t seq_len = seq_lens[static_cast<size_t>(batch_idx)];
    const int32_t pf = prefill_len[static_cast<size_t>(req_state_idx)];
    // ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710): the condition that was written here
    // inline now has a name, because the refusal in `ModelRegistry::Forward` has
    // to apply the SAME rule and not a second derivation of it. The behaviour is
    // unchanged, which `tests/vllm/v1/worker/test_combine_tokens.cpp` is the
    // control for.
    if (!CombineSplicesRow(seq_len, pf)) {
      continue;
    }

    // Keep prompt-tail slots intact; only rewrite generated-token slots
    // (:343-348). Once num_logits > 1 the window can reach back over the prompt,
    // and then logits_start addresses a PROMPT id that upstream leaves alone.
    // With num_logits == 1 this is implied by the check above, so the
    // non-speculative path is unchanged.
    const int32_t first_logit_seq_pos = seq_len - num_logits;
    if (num_new_sampled_tokens > 0 && first_logit_seq_pos >= pf) {
      input_token_ids[static_cast<size_t>(logits_start)] =
          last_sampled_tokens[static_cast<size_t>(req_state_idx)];
    }

    // Write the draft tokens (if any) to input_ids (:350-361). The count comes
    // from num_draft_tokens and NOT from draft_tokens_stride, which is the
    // speculator's max draft length and pads every shorter row.
    if (num_draft_tokens > 0) {
      VT_CHECK(draft_tokens_stride >= num_draft_tokens,
               "combine_sampled_and_draft_tokens: draft_tokens_stride is "
               "smaller than a request's draft count");
      const size_t row = static_cast<size_t>(req_state_idx) *
                         static_cast<size_t>(draft_tokens_stride);
      VT_CHECK(row + static_cast<size_t>(num_draft_tokens) <=
                   draft_tokens.size(),
               "combine_sampled_and_draft_tokens: draft_tokens does not hold a "
               "row for this request's req_state");
      for (int b = 0; b < num_draft_tokens; ++b) {
        input_token_ids[static_cast<size_t>(query_end - num_draft_tokens + b)] =
            draft_tokens[row + static_cast<size_t>(b)];
      }
    }
  }
  return logits_indices;
}

// ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710). The OR over the batch of the row
// predicate the combine applies, and nothing else: see the header for why this
// may not become a per-request answer.
//
// It reads `prefill_len` through `idx_mapping` for the same reason the combine
// does — after an abort or a finish reorders `req_states` against the dense
// batch, batch row `i` is not req_state slot `i`, and indexing `prefill_len` by
// the batch row would compare a row's sequence length against another request's
// prefill. A short `seq_lens` or `prefill_len` cannot silently answer false: the
// bounds are checked and a row the caller cannot substantiate is refused.
bool AnyRowSplicedByCombine(const std::vector<int32_t>& seq_lens,
                            const std::vector<int32_t>& prefill_len,
                            const int32_t* idx_mapping, int num_reqs) {
  VT_CHECK(num_reqs >= 0,
           "AnyRowSplicedByCombine: negative num_reqs");
  VT_CHECK(static_cast<size_t>(num_reqs) <= seq_lens.size(),
           "AnyRowSplicedByCombine: num_reqs exceeds the seq_lens the caller "
           "supplied, so a row's staleness cannot be decided");
  for (int batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int req_state_idx =
        idx_mapping != nullptr ? idx_mapping[batch_idx] : batch_idx;
    VT_CHECK(req_state_idx >= 0 &&
                 static_cast<size_t>(req_state_idx) < prefill_len.size(),
             "AnyRowSplicedByCombine: idx_mapping points outside prefill_len");
    if (CombineSplicesRow(seq_lens[static_cast<size_t>(batch_idx)],
                          prefill_len[static_cast<size_t>(req_state_idx)])) {
      return true;
    }
  }
  return false;
}

}  // namespace vllm::v1
