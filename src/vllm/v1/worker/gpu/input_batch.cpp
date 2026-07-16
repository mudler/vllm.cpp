// Ported from: vllm/v1/worker/gpu_input_batch.py @ e24d1b24
// See include/vllm/v1/worker/gpu/input_batch.h for the scope, the V1-algorithm
// / MRV2-contract note, and the deferred slot state.

#include "vllm/v1/worker/gpu/input_batch.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace vllm::v1 {

// ─── CachedRequestState ─────────────────────────────────────────────────────

CachedRequestState CachedRequestState::from_new_request(
    const NewRequestData& new_req) {
  CachedRequestState state;
  state.req_id = new_req.req_id;
  if (new_req.prompt_token_ids.has_value()) {
    state.prompt_token_ids = *new_req.prompt_token_ids;
  }
  // T0 generation path always carries sampling_params (pooling DEFERRED).
  state.sampling_params = new_req.sampling_params.value();
  state.block_ids = new_req.block_ids;
  state.num_computed_tokens = new_req.num_computed_tokens;
  state.finalize();

  // MRV2 contract: prefill_token_ids == all_token_ids (prompt + output at
  // schedule time). The output tokens are its tail beyond the prompt, so the
  // per-slot token_ids_cpu seed (prompt then output) reproduces exactly the
  // scheduler's prefill_token_ids. (The V1 runner instead reconstructed
  // all_token_ids per-req — that MRV1 path is dead here.)
  if (new_req.prefill_token_ids.has_value()) {
    const std::vector<int32_t>& all_ids = *new_req.prefill_token_ids;
    for (size_t i = static_cast<size_t>(state.num_prompt_tokens);
         i < all_ids.size(); ++i) {
      state.output_token_ids.push_back(all_ids[i]);
    }
  }
  return state;
}

int CachedRequestState::get_token_id(int idx) const {
  if (idx < num_prompt_tokens) {
    return prompt_token_ids[static_cast<size_t>(idx)];
  }
  if (idx - num_prompt_tokens < static_cast<int>(output_token_ids.size())) {
    return output_token_ids[static_cast<size_t>(idx - num_prompt_tokens)];
  }
  return -1;
}

// ─── InputBatch::RemovedTracker ─────────────────────────────────────────────

void InputBatch::RemovedTracker::ensure_sorted() {
  if (!is_sorted_) {
    // Descending, so back() is the lowest index (upstream _ensure_removed_sorted
    // sorts reverse=True and pops from the end).
    std::sort(removed_.begin(), removed_.end(), std::greater<int>());
    is_sorted_ = true;
  }
}

void InputBatch::RemovedTracker::removed_append(int index) {
  removed_.push_back(index);
  is_sorted_ = false;
}

std::optional<int> InputBatch::RemovedTracker::peek_removed() {
  if (removed_.empty()) {
    return std::nullopt;
  }
  ensure_sorted();
  return removed_.back();
}

std::optional<int> InputBatch::RemovedTracker::pop_removed() {
  if (removed_.empty()) {
    return std::nullopt;
  }
  ensure_sorted();
  const int value = removed_.back();
  removed_.pop_back();
  return value;
}

const std::vector<int>& InputBatch::RemovedTracker::removed() {
  ensure_sorted();
  return removed_;
}

// ─── InputBatch ─────────────────────────────────────────────────────────────

namespace {
// Upstream builds MultiGroupBlockTable with per-group max_num_blocks derived
// from max_model_len; block_sizes / kernel_block_sizes are the caller's groups.
MultiGroupBlockTable make_block_table(int max_num_reqs, int max_model_len,
                                      int max_num_batched_tokens,
                                      std::vector<int> block_sizes,
                                      std::vector<int> kernel_block_sizes) {
  return MultiGroupBlockTable(max_num_reqs, max_model_len,
                              max_num_batched_tokens, std::move(block_sizes),
                              std::move(kernel_block_sizes));
}
}  // namespace

InputBatch::InputBatch(int max_num_reqs, int max_model_len,
                       int max_num_batched_tokens, int vocab_size,
                       std::vector<int> block_sizes,
                       std::vector<int> kernel_block_sizes)
    : max_num_reqs(max_num_reqs),
      max_model_len(max_model_len),
      max_num_batched_tokens(max_num_batched_tokens),
      vocab_size(vocab_size),
      block_table(make_block_table(max_num_reqs, max_model_len,
                                   max_num_batched_tokens, std::move(block_sizes),
                                   std::move(kernel_block_sizes))) {
  const size_t n = static_cast<size_t>(max_num_reqs);
  token_ids_cpu.assign(n * static_cast<size_t>(max_model_len), 0);
  num_tokens_no_spec.assign(n, 0);
  num_prompt_tokens.assign(n, 0);
  num_computed_tokens_cpu.assign(n, 0);
  last_sampled_tokens.assign(n, 0);
  prefill_len.assign(n, 0);
  temperature_cpu.assign(n, 0.0f);
  top_p_cpu.assign(n, 0.0f);
  top_k_cpu.assign(n, 0);
  frequency_penalties_cpu.assign(n, 0.0f);
  presence_penalties_cpu.assign(n, 0.0f);
  repetition_penalties_cpu.assign(n, 0.0f);
  seeds.assign(n, std::nullopt);
}

int InputBatch::register_add_request() {
  // Fill the next empty index if there is one; append to the end otherwise.
  if (std::optional<int> hole = removed_tracker_.pop_removed()) {
    return *hole;
  }
  return num_reqs();
}

int InputBatch::get_active_token_count(int req_index) const {
  return num_tokens_no_spec[static_cast<size_t>(req_index)] +
         static_cast<int>(spec_token_ids[static_cast<size_t>(req_index)].size());
}

int InputBatch::add_request(const CachedRequestState& request) {
  const int req_index = register_add_request();
  const std::string& req_id = request.req_id;

  if (req_index == static_cast<int>(req_ids.size())) {
    req_ids.push_back(req_id);
    req_output_token_ids.push_back(request.output_token_ids);
    spec_token_ids.emplace_back();
  } else {
    req_ids[static_cast<size_t>(req_index)] = req_id;
    req_output_token_ids[static_cast<size_t>(req_index)] =
        request.output_token_ids;
    spec_token_ids[static_cast<size_t>(req_index)].clear();
  }
  req_id_to_index[req_id] = req_index;

  // Seed the token buffer: prompt then output (== prefill_token_ids).
  const int num_prompt = request.num_prompt_tokens;
  num_prompt_tokens[static_cast<size_t>(req_index)] = num_prompt;
  const size_t row = static_cast<size_t>(req_index) *
                     static_cast<size_t>(max_model_len);
  for (int i = 0; i < num_prompt; ++i) {
    token_ids_cpu[row + static_cast<size_t>(i)] = request.prompt_token_ids[
        static_cast<size_t>(i)];
  }
  for (size_t i = 0; i < request.output_token_ids.size(); ++i) {
    token_ids_cpu[row + static_cast<size_t>(num_prompt) + i] =
        request.output_token_ids[i];
  }
  // Number of tokens without spec-decode tokens.
  num_tokens_no_spec[static_cast<size_t>(req_index)] = request.num_tokens();
  num_computed_tokens_cpu[static_cast<size_t>(req_index)] =
      request.num_computed_tokens;
  block_table.add_row(request.block_ids, req_index);

  // Async-scheduling state (states.py::add_request:105-122). prefill_len is the
  // token count known at admission (prompt + pre-existing output), fixed for the
  // request's life. Seed last_sampled_tokens ONLY for a resumed / PD-disagg
  // request (0 < num_computed <= prefill_len) so its first decode step reads the
  // correct input id via combine; a fresh prefill (num_computed == 0) never has
  // combine read it, so it stays 0. Both are inert unless the async runner path
  // is engaged.
  const int prefill = request.num_tokens();
  prefill_len[static_cast<size_t>(req_index)] = prefill;
  if (0 < request.num_computed_tokens &&
      request.num_computed_tokens <= prefill) {
    last_sampled_tokens[static_cast<size_t>(req_index)] =
        token_ids_cpu[row + static_cast<size_t>(request.num_computed_tokens - 1)];
  } else {
    last_sampled_tokens[static_cast<size_t>(req_index)] = 0;
  }

  // Sampling metadata (pooling DEFERRED — T0 always has sampling_params).
  const SamplingParams& sp = request.sampling_params;
  if (sp.Type() == SamplingType::kGreedy) {
    // Avoid a later division-by-zero in apply_temperature.
    temperature_cpu[static_cast<size_t>(req_index)] = 0.0f;
    greedy_reqs[req_id] = 1;
  } else {
    temperature_cpu[static_cast<size_t>(req_index)] =
        static_cast<float>(sp.temperature);
    random_reqs[req_id] = 1;
  }

  top_p_cpu[static_cast<size_t>(req_index)] = static_cast<float>(sp.top_p);
  if (sp.top_p < 1.0) {
    top_p_reqs[req_id] = 1;
  }
  int top_k = sp.top_k;
  if (0 < top_k && top_k < vocab_size) {
    top_k_reqs[req_id] = 1;
  } else {
    top_k = vocab_size;
  }
  top_k_cpu[static_cast<size_t>(req_index)] = top_k;

  frequency_penalties_cpu[static_cast<size_t>(req_index)] =
      static_cast<float>(sp.frequency_penalty);
  if (sp.frequency_penalty != 0.0) {
    frequency_penalties_reqs[req_id] = 1;
  }
  presence_penalties_cpu[static_cast<size_t>(req_index)] =
      static_cast<float>(sp.presence_penalty);
  if (sp.presence_penalty != 0.0) {
    presence_penalties_reqs[req_id] = 1;
  }
  repetition_penalties_cpu[static_cast<size_t>(req_index)] =
      static_cast<float>(sp.repetition_penalty);
  if (sp.repetition_penalty != 1.0) {
    repetition_penalties_reqs[req_id] = 1;
  }

  // Per-request RNG seed (upstream request.generator == sampling_params.seed).
  seeds[static_cast<size_t>(req_index)] = sp.seed;

  return req_index;
}

SamplingMetadata InputBatch::make_sampling_metadata() const {
  // Port of gpu_input_batch.py::_make_sampling_metadata (@ e24d1b24). Fills the
  // dense [0, num_reqs) prefix, matching upstream's field-fill order + the
  // "skip the copy when not needed" None/[]-defaults.
  const int n = num_reqs();
  const size_t nn = static_cast<size_t>(n);
  SamplingMetadata md;

  md.all_greedy = all_greedy();
  md.all_random = all_random();
  md.no_penalties = no_penalties();

  // temperature: None when all_greedy, else the [:num_reqs] slice
  // (gpu_input_batch.py:834-839).
  if (!md.all_greedy) {
    md.temperature = std::vector<float>(temperature_cpu.begin(),
                                        temperature_cpu.begin() + nn);
  }
  // top_p / top_k: None when the corresponding predicate is empty
  // (gpu_input_batch.py:919-920).
  if (!no_top_p()) {
    md.top_p =
        std::vector<float>(top_p_cpu.begin(), top_p_cpu.begin() + nn);
  }
  if (!no_top_k()) {
    md.top_k =
        std::vector<int32_t>(top_k_cpu.begin(), top_k_cpu.begin() + nn);
  }

  // Penalties are always sliced [:num_reqs] in the returned metadata
  // (gpu_input_batch.py:925-927); the device-copy is what upstream gates on
  // no_penalties, not the slice itself.
  md.frequency_penalties = std::vector<float>(
      frequency_penalties_cpu.begin(), frequency_penalties_cpu.begin() + nn);
  md.presence_penalties = std::vector<float>(
      presence_penalties_cpu.begin(), presence_penalties_cpu.begin() + nn);
  md.repetition_penalties = std::vector<float>(
      repetition_penalties_cpu.begin(), repetition_penalties_cpu.begin() + nn);

  // prompt_token_ids: only when penalties (or a token-id-consuming proc, always
  // false at T0) need them (gpu_input_batch.py:861-876). Ragged per-req prompt
  // slice of token_ids_cpu[:, :num_prompt_tokens].
  const bool needs_prompt_token_ids = !md.no_penalties;
  if (needs_prompt_token_ids) {
    std::vector<std::vector<int32_t>> prompts(nn);
    for (int i = 0; i < n; ++i) {
      const int np = num_prompt_tokens[static_cast<size_t>(i)];
      const size_t row =
          static_cast<size_t>(i) * static_cast<size_t>(max_model_len);
      prompts[static_cast<size_t>(i)].assign(
          token_ids_cpu.begin() + static_cast<std::ptrdiff_t>(row),
          token_ids_cpu.begin() + static_cast<std::ptrdiff_t>(row) + np);
    }
    md.prompt_token_ids = std::move(prompts);
  }

  // output_token_ids: only when a proc needs them (gpu_input_batch.py:884-894).
  // At T0 that is !no_penalties (bad_words / logitsprocs_need_output_token_ids
  // are always empty/false). Empty [] otherwise, matching upstream.
  const bool needs_output_token_ids = !md.no_penalties;
  if (needs_output_token_ids) {
    md.output_token_ids.resize(nn);
    for (int i = 0; i < n; ++i) {
      const auto& row = req_output_token_ids[static_cast<size_t>(i)];
      if (row.has_value()) {
        md.output_token_ids[static_cast<size_t>(i)] = *row;
      }
    }
  }

  // spec_token_ids: pass the dense prefix (always empty lists at T0). Upstream
  // passes self.spec_token_ids directly (gpu_input_batch.py:929).
  md.spec_token_ids = std::vector<std::vector<int32_t>>(
      spec_token_ids.begin(), spec_token_ids.begin() + nn);

  // generators (gpu_input_batch.py:921, sourced :413-414 from request.generator
  // == sampling_params.seed): req_index -> seed for every seeded request in the
  // dense prefix. WIRED at M1.8 Task 4 from the per-slot `seeds` array; requests
  // without a seed are absent (they use the sampler's batch-default RNG,
  // upstream NOTE :251-252). CLOSES the M1.7 seed carry.
  for (int i = 0; i < n; ++i) {
    if (seeds[static_cast<size_t>(i)].has_value()) {
      md.generators[i] = static_cast<uint64_t>(*seeds[static_cast<size_t>(i)]);
    }
  }

  // ─── Fields whose InputBatch-side tracking is NOT yet landed (marked) ──────
  // Each is a faithful upstream default (empty/None) with its dependency cite;
  // wiring them requires per-slot state this InputBatch does not yet keep.
  //
  //  * max_num_logprobs (gpu_input_batch.py:922 / :1122, from the num_logprobs
  //    dict populated by sampling_params.logprobs): no num_logprobs tracking
  //    here — left None (no logprobs). Logprobs wiring is an M1.8 dependency.
  //  * allowed_token_ids_mask (gpu_input_batch.py:896-904) + bad_words_token_ids
  //    (:932): no has_allowed_token_ids / bad_words slot tracking — left
  //    None / empty. A Task-3 / M1.8 dependency (SamplingParams also defers
  //    allowed_token_ids / bad_words_token_ids).
  //  * min_tokens / logit_bias / min_p (the T0 builtins,
  //    logits_processor/builtin.py): InputBatch keeps no per-slot min_p array /
  //    min_tokens+stop set / logit_bias map yet — left empty. Populating them
  //    (min_p is on SamplingParams; logit_bias / all_stop_token_ids are
  //    deferred on SamplingParams) is a Task-3 dependency.
  // md.generators / max_num_logprobs / allowed_token_ids_mask /
  // bad_words_token_ids / min_tokens / logit_bias / min_p keep their defaults.

  return md;
}

std::optional<int> InputBatch::remove_request(const std::string& req_id) {
  const auto it = req_id_to_index.find(req_id);
  if (it == req_id_to_index.end()) {
    return std::nullopt;
  }
  const int req_index = it->second;
  req_id_to_index.erase(it);

  removed_tracker_.removed_append(req_index);
  req_ids[static_cast<size_t>(req_index)] = std::nullopt;
  req_output_token_ids[static_cast<size_t>(req_index)] = std::nullopt;
  spec_token_ids[static_cast<size_t>(req_index)].clear();
  seeds[static_cast<size_t>(req_index)] = std::nullopt;
  block_table.clear_row(req_index);

  // Discard from the sampling-predicate sets (LoRA / generators / logprobs /
  // pooling / structured-output DEFERRED).
  greedy_reqs.erase(req_id);
  random_reqs.erase(req_id);
  top_p_reqs.erase(req_id);
  top_k_reqs.erase(req_id);
  frequency_penalties_reqs.erase(req_id);
  presence_penalties_reqs.erase(req_id);
  repetition_penalties_reqs.erase(req_id);
  return req_index;
}

void InputBatch::condense() {
  const int num = num_reqs();

  // empty_req_indices is the removed list, sorted DESCENDING; it shrinks as we
  // pop (a live reference, exactly like upstream's batch_update_builder.removed).
  const std::vector<int>& empty_req_indices = removed_tracker_.removed();
  if (empty_req_indices.empty()) {
    // All removed slots were replaced by adds, or nothing was removed.
    return;
  }
  if (num == 0) {
    // The batched state is empty.
    req_ids.clear();
    req_output_token_ids.clear();
    spec_token_ids.clear();
    removed_tracker_ = RemovedTracker();
    return;
  }

  const auto is_empty = [&](int idx) {
    return std::find(empty_req_indices.begin(), empty_req_indices.end(), idx) !=
           empty_req_indices.end();
  };

  // NOTE(woosuk): assumes empty_req_indices is sorted in descending order.
  int last_req_index = num + static_cast<int>(empty_req_indices.size()) - 1;
  while (removed_tracker_.has_removed()) {
    // Find the largest non-empty index.
    while (is_empty(last_req_index)) {
      --last_req_index;
    }

    // Find the smallest empty index.
    const std::optional<int> empty_peek = removed_tracker_.peek_removed();
    const int empty_index = *empty_peek;
    if (empty_index >= last_req_index) {
      break;
    }

    // Move the active request at last_req_index down into empty_index.
    removed_tracker_.pop_removed();
    const std::optional<std::string> req_id =
        req_ids[static_cast<size_t>(last_req_index)];
    std::optional<std::vector<int32_t>> output_token_ids =
        req_output_token_ids[static_cast<size_t>(last_req_index)];
    req_ids[static_cast<size_t>(empty_index)] = req_id;
    req_ids[static_cast<size_t>(last_req_index)] = std::nullopt;
    req_output_token_ids[static_cast<size_t>(empty_index)] =
        std::move(output_token_ids);
    req_output_token_ids[static_cast<size_t>(last_req_index)] = std::nullopt;
    req_id_to_index[*req_id] = empty_index;

    const int num_tokens = get_active_token_count(last_req_index);

    std::swap(spec_token_ids[static_cast<size_t>(last_req_index)],
              spec_token_ids[static_cast<size_t>(empty_index)]);
    spec_token_ids[static_cast<size_t>(last_req_index)].clear();

    const size_t empty_row = static_cast<size_t>(empty_index) *
                             static_cast<size_t>(max_model_len);
    const size_t last_row = static_cast<size_t>(last_req_index) *
                            static_cast<size_t>(max_model_len);
    for (int i = 0; i < num_tokens; ++i) {
      token_ids_cpu[empty_row + static_cast<size_t>(i)] =
          token_ids_cpu[last_row + static_cast<size_t>(i)];
    }
    num_tokens_no_spec[static_cast<size_t>(empty_index)] =
        num_tokens_no_spec[static_cast<size_t>(last_req_index)];
    num_prompt_tokens[static_cast<size_t>(empty_index)] =
        num_prompt_tokens[static_cast<size_t>(last_req_index)];
    num_computed_tokens_cpu[static_cast<size_t>(empty_index)] =
        num_computed_tokens_cpu[static_cast<size_t>(last_req_index)];
    // Async-scheduling per-slot state moves with the request (keeps it aligned
    // to the dense req_state index combine reads).
    last_sampled_tokens[static_cast<size_t>(empty_index)] =
        last_sampled_tokens[static_cast<size_t>(last_req_index)];
    prefill_len[static_cast<size_t>(empty_index)] =
        prefill_len[static_cast<size_t>(last_req_index)];
    block_table.move_row(last_req_index, empty_index);

    // Sampling metadata (LoRA / generators / allowed-token-ids / bad-words
    // DEFERRED).
    temperature_cpu[static_cast<size_t>(empty_index)] =
        temperature_cpu[static_cast<size_t>(last_req_index)];
    top_p_cpu[static_cast<size_t>(empty_index)] =
        top_p_cpu[static_cast<size_t>(last_req_index)];
    top_k_cpu[static_cast<size_t>(empty_index)] =
        top_k_cpu[static_cast<size_t>(last_req_index)];
    frequency_penalties_cpu[static_cast<size_t>(empty_index)] =
        frequency_penalties_cpu[static_cast<size_t>(last_req_index)];
    presence_penalties_cpu[static_cast<size_t>(empty_index)] =
        presence_penalties_cpu[static_cast<size_t>(last_req_index)];
    repetition_penalties_cpu[static_cast<size_t>(empty_index)] =
        repetition_penalties_cpu[static_cast<size_t>(last_req_index)];
    seeds[static_cast<size_t>(empty_index)] =
        seeds[static_cast<size_t>(last_req_index)];
    seeds[static_cast<size_t>(last_req_index)] = std::nullopt;

    // Decrement last_req_index since it is now empty.
    --last_req_index;
  }

  // Trim the dynamic lists to the batch size.
  req_ids.resize(static_cast<size_t>(num));
  req_output_token_ids.resize(static_cast<size_t>(num));
  spec_token_ids.resize(static_cast<size_t>(num));

  // Upstream clears the removed tracking in refresh_metadata (batch_update
  // reset), which is deferred to M1.7; reset it here so any trailing removed
  // indices (trimmed above) do not leak into the next step's bookkeeping.
  removed_tracker_ = RemovedTracker();
}

void InputBatch::swap_states(int i1, int i2) {
  // Port of gpu_input_batch.py::swap_states (@ e24d1b24). See input_batch.h for
  // the deferred (T0-empty) fields skipped here.
  if (i1 == i2) {
    return;
  }
  const std::optional<std::string> old_id_i1 = req_ids[static_cast<size_t>(i1)];
  const std::optional<std::string> old_id_i2 = req_ids[static_cast<size_t>(i2)];

  // Only swap the active token prefix (max active count of the two rows).
  const int max_active = std::max(get_active_token_count(i1),
                                  get_active_token_count(i2));

  std::swap(req_ids[static_cast<size_t>(i1)], req_ids[static_cast<size_t>(i2)]);
  std::swap(req_output_token_ids[static_cast<size_t>(i1)],
            req_output_token_ids[static_cast<size_t>(i2)]);
  std::swap(spec_token_ids[static_cast<size_t>(i1)],
            spec_token_ids[static_cast<size_t>(i2)]);

  if (old_id_i1.has_value()) {
    req_id_to_index[*old_id_i1] = i2;
  }
  if (old_id_i2.has_value()) {
    req_id_to_index[*old_id_i2] = i1;
  }

  std::swap(num_tokens_no_spec[static_cast<size_t>(i1)],
            num_tokens_no_spec[static_cast<size_t>(i2)]);
  std::swap(num_prompt_tokens[static_cast<size_t>(i1)],
            num_prompt_tokens[static_cast<size_t>(i2)]);
  std::swap(num_computed_tokens_cpu[static_cast<size_t>(i1)],
            num_computed_tokens_cpu[static_cast<size_t>(i2)]);
  // Async-scheduling per-slot state (moves with the row in the decode-first
  // reorder, so combine's dense req_state index stays correct).
  std::swap(last_sampled_tokens[static_cast<size_t>(i1)],
            last_sampled_tokens[static_cast<size_t>(i2)]);
  std::swap(prefill_len[static_cast<size_t>(i1)],
            prefill_len[static_cast<size_t>(i2)]);

  // Swap the active token prefix of the two rows (upstream copies only
  // max_active_token_count columns).
  const size_t row1 =
      static_cast<size_t>(i1) * static_cast<size_t>(max_model_len);
  const size_t row2 =
      static_cast<size_t>(i2) * static_cast<size_t>(max_model_len);
  for (int c = 0; c < max_active; ++c) {
    std::swap(token_ids_cpu[row1 + static_cast<size_t>(c)],
              token_ids_cpu[row2 + static_cast<size_t>(c)]);
  }

  block_table.swap_row(i1, i2);

  // Sampling params (autoregressive models; pooling DEFERRED).
  std::swap(temperature_cpu[static_cast<size_t>(i1)],
            temperature_cpu[static_cast<size_t>(i2)]);
  std::swap(top_p_cpu[static_cast<size_t>(i1)],
            top_p_cpu[static_cast<size_t>(i2)]);
  std::swap(top_k_cpu[static_cast<size_t>(i1)],
            top_k_cpu[static_cast<size_t>(i2)]);
  std::swap(frequency_penalties_cpu[static_cast<size_t>(i1)],
            frequency_penalties_cpu[static_cast<size_t>(i2)]);
  std::swap(presence_penalties_cpu[static_cast<size_t>(i1)],
            presence_penalties_cpu[static_cast<size_t>(i2)]);
  std::swap(repetition_penalties_cpu[static_cast<size_t>(i1)],
            repetition_penalties_cpu[static_cast<size_t>(i2)]);
  std::swap(seeds[static_cast<size_t>(i1)], seeds[static_cast<size_t>(i2)]);
}

}  // namespace vllm::v1
