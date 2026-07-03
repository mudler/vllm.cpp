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
  temperature_cpu.assign(n, 0.0f);
  top_p_cpu.assign(n, 0.0f);
  top_k_cpu.assign(n, 0);
  frequency_penalties_cpu.assign(n, 0.0f);
  presence_penalties_cpu.assign(n, 0.0f);
  repetition_penalties_cpu.assign(n, 0.0f);
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

  return req_index;
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

}  // namespace vllm::v1
