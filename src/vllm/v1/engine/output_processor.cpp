// Ported from: vllm/v1/engine/output_processor.py @ e24d1b24
// See include/vllm/v1/engine/output_processor.h for scope, deviations and
// deferrals.
#include "vllm/v1/engine/output_processor.h"

#include <algorithm>
#include <utility>

namespace vllm::v1 {

// ---------------------------------------------------------------------------
// RequestState
// ---------------------------------------------------------------------------

RequestState RequestState::FromNewRequest(const tok::Tokenizer* tokenizer,
                                          const EngineCoreRequest& request,
                                          std::optional<std::string> prompt,
                                          int request_index,
                                          int stream_interval) {
  // output_processor.py:222-234 (the sampling_params branch; pooling deferred).
  const SamplingParams& sp = request.sampling_params;
  // :223 if not sampling_params.detokenize: tokenizer = None
  const tok::Tokenizer* detok_tokenizer = sp.detokenize ? tokenizer : nullptr;

  // IncrementalDetokenizer.from_new_request(tokenizer, request): our factory
  // takes the detokenization-relevant subset of EngineCoreRequest +
  // SamplingParams (DetokenizerRequest).
  DetokenizerRequest dreq;
  dreq.prompt_token_ids = request.prompt_token_ids;
  dreq.skip_special_tokens = sp.skip_special_tokens;
  dreq.spaces_between_special_tokens = sp.spaces_between_special_tokens;
  dreq.stop = sp.stop;
  dreq.include_stop_str_in_output = sp.include_stop_str_in_output;
  dreq.min_tokens = static_cast<size_t>(sp.min_tokens);

  RequestState state;
  state.request_id = request.request_id;
  // external_req_id == request_id at T0 (see header).
  state.external_req_id = request.request_id;
  state.request_index = request_index;
  state.output_kind = sp.output_kind;
  state.prompt = std::move(prompt);
  state.prompt_token_ids = request.prompt_token_ids;
  state.prompt_len = request.prompt_token_ids.size();
  state.detokenizer =
      IncrementalDetokenizer::FromNewRequest(detok_tokenizer, std::move(dreq));
  state.max_tokens_param = sp.max_tokens;
  state.stream_interval = stream_interval;
  return state;
}

std::optional<RequestOutput> RequestState::make_request_output(
    const std::vector<int32_t>& new_token_ids,
    std::optional<FinishReason> finish_reason,
    std::optional<std::string> stop_reason) {
  // output_processor.py:272-331 (text path; pooling / parent_req deferred).
  const bool finished = finish_reason.has_value();
  const bool final_only = output_kind == RequestOutputKind::kFinalOnly;

  // :283 Only the final output is required in FINAL_ONLY mode.
  if (!finished && final_only) {
    return std::nullopt;
  }

  std::vector<int32_t> token_ids = new_token_ids;

  // :287-308 stream_interval throttling (default 1 => inert).
  if (stream_interval > 1) {
    // detokenizer is non-null on the text path.
    const size_t num_out = detokenizer->NumOutputTokens();
    if (!(finished || sent_tokens_offset == 0 ||
          num_out - sent_tokens_offset >=
              static_cast<size_t>(stream_interval))) {
      return std::nullopt;
    }
    if (output_kind == RequestOutputKind::kDelta) {
      // :305 Send tokens from the offset in DELTA mode.
      const std::vector<int32_t> all = detokenizer->OutputTokenIds();
      token_ids.assign(all.begin() + static_cast<std::ptrdiff_t>(
                                         std::min(sent_tokens_offset, all.size())),
                       all.end());
      sent_tokens_offset = detokenizer->NumOutputTokens();
    }
  }

  // pooling_output is None at T0 -> the completion branch (:319).
  CompletionOutput output =
      NewCompletionOutput(std::move(token_ids), finish_reason, std::move(stop_reason));

  // parent_req is None at T0 (:321) -> outputs = [output].
  std::vector<CompletionOutput> outputs;
  outputs.push_back(std::move(output));

  return NewRequestOutput(external_req_id, std::move(outputs), finished);
}

RequestOutput RequestState::NewRequestOutput(
    const std::string& external_req_id_in,
    std::vector<CompletionOutput> outputs, bool finished) {
  // output_processor.py:333-374 (RequestOutput branch; PoolingRequestOutput and
  // logprobs deferred).
  RequestOutput ro;
  ro.request_id = external_req_id_in;
  ro.prompt = prompt;
  ro.prompt_token_ids = prompt_token_ids;
  ro.outputs = std::move(outputs);
  ro.finished = finished;
  return ro;
}

CompletionOutput RequestState::NewCompletionOutput(
    std::vector<int32_t> token_ids, std::optional<FinishReason> finish_reason,
    std::optional<std::string> stop_reason) {
  // output_processor.py:376-411 (logprobs / routed_experts deferred).
  const bool finished = finish_reason.has_value();
  const bool delta = output_kind == RequestOutputKind::kDelta;

  // :388-390 text / token_ids per delta mode.
  const std::string text = detokenizer->GetNextOutputText(finished, delta);
  if (!delta) {
    token_ids = detokenizer->OutputTokenIds();
  }

  CompletionOutput co;
  co.index = request_index;
  co.text = text;
  co.token_ids = std::move(token_ids);
  // :409-410 finish_reason/stop_reason only reported once finished.
  if (finished) {
    co.SetFinishReason(*finish_reason);  // str(finish_reason)
    co.stop_reason = std::move(stop_reason);
  }
  return co;
}

// ---------------------------------------------------------------------------
// OutputProcessor
// ---------------------------------------------------------------------------

OutputProcessor::OutputProcessor(const tok::Tokenizer* tokenizer,
                                 int stream_interval)
    : tokenizer_(tokenizer), stream_interval_(stream_interval) {}

void OutputProcessor::add_request(const EngineCoreRequest& request,
                                  std::optional<std::string> prompt,
                                  int request_index) {
  // output_processor.py:512-541 (T0: no parent_req / queue; the streaming-input
  // re-entry — a request_id already present — is deferred).
  const std::string& request_id = request.request_id;
  if (request_states_.find(request_id) != request_states_.end()) {
    // Upstream _update_streaming_request_state; deferred. A request_id appears
    // once at T0.
    return;
  }

  RequestState state = RequestState::FromNewRequest(
      tokenizer_, request, std::move(prompt), request_index, stream_interval_);
  const std::string external_req_id = state.external_req_id;
  request_states_[request_id] =
      std::make_unique<RequestState>(std::move(state));

  // :541 Track external_req_id -> [internal_req_id, ...].
  external_req_ids_[external_req_id].push_back(request_id);
}

OutputProcessorOutput OutputProcessor::process_outputs(
    const EngineCoreOutputs& engine_core_outputs) {
  // output_processor.py:576-693 (the synchronous LLMEngine path; stats /
  // tracing / logprobs / pooling / streaming-input deferred).
  OutputProcessorOutput result;

  for (const EngineCoreOutput& eco : engine_core_outputs.outputs) {
    const std::string& req_id = eco.request_id;
    auto it = request_states_.find(req_id);
    if (it == request_states_.end()) {
      // :609 Ignore output for already-aborted / unknown request.
      continue;
    }
    RequestState& req_state = *it->second;

    // 1) Compute stats — deferred (no IterationStats at T0).

    const std::vector<int32_t>& new_token_ids = eco.new_token_ids;
    std::optional<FinishReason> finish_reason = eco.finish_reason;
    std::optional<std::string> stop_reason = eco.stop_reason;
    // routed_experts / prefill_stats deferred.

    if (req_state.is_prefilling) {
      // num_cached_tokens from prefill_stats deferred.
      req_state.is_prefilling = false;
    }

    // pooling_output is None at T0 -> the detokenize branch (:635).
    // 2) Detokenize the token ids into text and perform STRING-level stop checks
    //    (the piece the scheduler's token-level check_stop can't do).
    const bool stop_terminated =
        finish_reason.has_value() && *finish_reason == FinishReason::kStop;
    std::optional<std::string> stop_string =
        req_state.detokenizer->Update(new_token_ids, stop_terminated);
    if (stop_string.has_value()) {
      // :642-644 detokenizer detected a stop string.
      finish_reason = FinishReason::kStop;
      stop_reason = *stop_string;
    }

    // 3) Logprobs — deferred.

    // 4) Create and handle the RequestOutput (:650-666).
    std::optional<RequestOutput> request_output = req_state.make_request_output(
        new_token_ids, finish_reason, stop_reason);
    if (request_output.has_value()) {
      // streaming_input deferred (false) -> no finished=false override.
      // queue is null on the sync path -> collect into the returned list.
      result.request_outputs.push_back(std::move(*request_output));
    }

    // Free completed requests (:669-688).
    if (finish_reason.has_value()) {
      // streaming_input deferred (false) -> the finish branch.
      const bool engine_core_finished = eco.Finished();
      FinishRequest(req_state);  // invalidates req_state / it
      if (!engine_core_finished) {
        // :678 If req not finished in EngineCore but the detokenizer detected a
        // stop string, an abort is needed in EngineCore.
        result.reqs_to_abort.push_back(req_id);
      }
      // stats / tracing deferred.
    }
  }

  return result;
}

void OutputProcessor::FinishRequest(RequestState& req_state) {
  // output_processor.py:695-707. Copy the ids before erasing — erasing the map
  // entry destroys `req_state`.
  const std::string req_id = req_state.request_id;
  const std::string external_req_id = req_state.external_req_id;

  auto mit = external_req_ids_.find(external_req_id);
  if (mit != external_req_ids_.end()) {
    std::vector<std::string>& ids = mit->second;
    ids.erase(std::remove(ids.begin(), ids.end(), req_id), ids.end());
    if (ids.empty()) {
      external_req_ids_.erase(mit);
    }
  }

  // parent_req cleanup deferred (no parallel sampling at T0).
  request_states_.erase(req_id);  // destroys req_state — must be last.
}

}  // namespace vllm::v1
