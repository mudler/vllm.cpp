// Ported from: vllm/v1/engine/output_processor.py @ e24d1b24
// See include/vllm/v1/engine/output_processor.h for scope, deviations and
// deferrals.
#include "vllm/v1/engine/output_processor.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace vllm::v1 {

// ---------------------------------------------------------------------------
// RequestOutputCollector
// ---------------------------------------------------------------------------

RequestOutputCollector::RequestOutputCollector(RequestOutputKind output_kind,
                                               std::string request_id)
    : aggregate_(output_kind == RequestOutputKind::kDelta),
      request_id_(std::move(request_id)) {}

void RequestOutputCollector::put(RequestOutput output) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!output_.has_value()) {
      output_ = std::move(output);
    } else {
      Merge(std::move(output));
    }
  }
  ready_.notify_one();
}

void RequestOutputCollector::put_error(std::exception_ptr error) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = std::move(error);
    output_.reset();
  }
  ready_.notify_one();
}

RequestOutput RequestOutputCollector::get() {
  std::unique_lock<std::mutex> lock(mutex_);
  ready_.wait(lock, [&] { return output_.has_value() || error_ != nullptr; });
  if (error_ != nullptr) {
    std::exception_ptr error = std::move(error_);
    error_ = nullptr;
    lock.unlock();
    std::rethrow_exception(error);
  }
  RequestOutput output = std::move(*output_);
  output_.reset();
  return output;
}

std::optional<RequestOutput> RequestOutputCollector::get_for(
    std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  const bool ready = ready_.wait_for(lock, timeout, [&] {
    return output_.has_value() || error_ != nullptr;
  });
  if (!ready) return std::nullopt;
  if (error_ != nullptr) {
    std::exception_ptr error = std::move(error_);
    error_ = nullptr;
    lock.unlock();
    std::rethrow_exception(error);
  }
  std::optional<RequestOutput> output(std::move(*output_));
  output_.reset();
  return output;
}

std::optional<RequestOutput> RequestOutputCollector::get_nowait() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (error_ != nullptr) {
    std::exception_ptr error = std::move(error_);
    error_ = nullptr;
    lock.unlock();
    std::rethrow_exception(error);
  }
  if (!output_.has_value()) return std::nullopt;
  std::optional<RequestOutput> output(std::move(*output_));
  output_.reset();
  return output;
}

bool RequestOutputCollector::has_output() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return output_.has_value() || error_ != nullptr;
}

void RequestOutputCollector::Merge(RequestOutput next) {
  // RequestOutput.add(next, aggregate=...) from outputs.py. Keep completions
  // with distinct indexes independent (parallel sampling is deferred, but the
  // collector contract is complete and the cumulative n>1 test can exercise
  // the value merge directly).
  RequestOutput& current = *output_;
  for (CompletionOutput& next_completion : next.outputs) {
    auto it = std::find_if(
        current.outputs.begin(), current.outputs.end(),
        [&](const CompletionOutput& completion) {
          return completion.index == next_completion.index;
        });
    if (it == current.outputs.end()) {
      current.outputs.push_back(std::move(next_completion));
      continue;
    }
    if (!aggregate_) {
      *it = std::move(next_completion);
      continue;
    }
    it->text += next_completion.text;
    it->token_ids.insert(it->token_ids.end(), next_completion.token_ids.begin(),
                         next_completion.token_ids.end());
    it->cumulative_logprob = next_completion.cumulative_logprob;
    // outputs.py:168-170: extend (concatenate) the DELTA logprobs, don't replace.
    if (next_completion.logprobs.has_value() && !next_completion.logprobs->empty()) {
      if (!it->logprobs.has_value()) it->logprobs = SampleLogprobs{};
      it->logprobs->insert(
          it->logprobs->end(),
          std::make_move_iterator(next_completion.logprobs->begin()),
          std::make_move_iterator(next_completion.logprobs->end()));
    }
    it->finish_reason = std::move(next_completion.finish_reason);
    it->stop_reason = std::move(next_completion.stop_reason);
  }
  // RequestOutput.add uses logical OR so a terminal state can never be
  // cleared by a later merge. Prompt logprobs belong to the first output and
  // are intentionally left untouched.
  current.finished = current.finished || next.finished;
}

// ---------------------------------------------------------------------------
// RequestState
// ---------------------------------------------------------------------------

RequestState RequestState::FromNewRequest(const tok::Tokenizer* tokenizer,
                                          const EngineCoreRequest& request,
                                          std::optional<std::string> prompt,
                                          int request_index,
                                          int stream_interval,
                                          std::shared_ptr<ParentRequest> parent_req) {
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
  // parent_req (output_processor.py:225): null for a single-sequence request; the
  // shared parent for a child of an n>1 parallel-sampling fan-out.
  state.parent_req = std::move(parent_req);
  state.output_kind = sp.output_kind;
  state.prompt = std::move(prompt);
  state.prompt_token_ids = request.prompt_token_ids;
  state.prompt_len = request.prompt_token_ids.size();
  state.detokenizer =
      IncrementalDetokenizer::FromNewRequest(detok_tokenizer, std::move(dreq));
  // LogprobsProcessor (output_processor.py:225-228): engaged only when the
  // request asked for sample and/or prompt logprobs, so the default generate
  // path leaves it nullopt (inert — SACRED greedy path unchanged). Uses the
  // same detokenize-gated tokenizer as the detokenizer (:222-223).
  if (sp.logprobs.has_value() || sp.prompt_logprobs.has_value()) {
    state.logprobs_processor =
        LogprobsProcessor::FromNewRequest(detok_tokenizer, sp);
  }
  state.max_tokens_param = sp.max_tokens;
  // output_processor.py:227-229 @ vllm#49754: a per-request stream_interval only
  // RAISES the engine-level interval (values below it are clamped up to it).
  if (sp.stream_interval.has_value()) {
    stream_interval = std::max(*sp.stream_interval, stream_interval);
  }
  state.stream_interval = stream_interval;
  // RequestStateStats.arrival_time (stats.py:222): the reference the e2e latency
  // and TTFT are measured from. Upstream uses request.arrival_time; at T0 we
  // stamp it here from the same steady clock the engine stamps its step
  // timestamps with, so the intervals are always non-negative.
  state.arrival_time = MonotonicSeconds();
  return state;
}

std::optional<RequestOutput> RequestState::make_request_output(
    const std::vector<int32_t>& new_token_ids,
    std::optional<FinishReason> finish_reason,
    std::optional<std::string> stop_reason,
    std::optional<std::vector<float>> pooling_output) {
  // output_processor.py:272-331 (text + pooling; parent_req deferred for
  // pooling — a pooling request is always n==1).
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

  // output_processor.py:319-331. Without a parent (single sequence): outputs =
  // [output], keyed by this request's external id. With a parent (n>1 child):
  // route the child output through ParentRequest::get_outputs — it either passes
  // the child output through (streaming) or aggregates the n final outputs, and
  // reports whether the whole parallel-sampling request is now finished. An empty
  // aggregation (children still pending under FINAL_ONLY) suppresses this step's
  // RequestOutput (return nullopt).
  std::vector<CompletionOutput> outputs;
  std::string out_external_req_id = external_req_id;
  bool out_finished = finished;
  if (parent_req == nullptr) {
    outputs.push_back(std::move(output));
  } else {
    std::pair<std::vector<CompletionOutput>, bool> aggregated =
        parent_req->get_outputs(request_id, std::move(output));
    if (aggregated.first.empty()) {
      return std::nullopt;
    }
    outputs = std::move(aggregated.first);
    out_finished = aggregated.second;
    out_external_req_id = parent_req->external_req_id();
  }

  RequestOutput ro =
      NewRequestOutput(out_external_req_id, std::move(outputs), out_finished);
  // ARCH-ONE-SURFACE ROW 6 (output_processor.py:319 pooling branch; recorded
  // deviation: upstream returns a separate PoolingRequestOutput class — ours
  // carries the pooled vector as an optional field on the ONE RequestOutput).
  ro.pooling_output = std::move(pooling_output);
  return ro;
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
  // prompt_logprobs (output_processor.py:363-375): DELTA pops (emit-once), else
  // read the accumulated value. nullopt unless prompt_logprobs was requested.
  if (logprobs_processor.has_value()) {
    if (output_kind == RequestOutputKind::kDelta) {
      ro.prompt_logprobs = logprobs_processor->pop_prompt_logprobs();
    } else {
      ro.prompt_logprobs = logprobs_processor->prompt_logprobs();
    }
  }
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
  // logprobs / cumulative_logprob (output_processor.py:401-417). In DELTA mode
  // emit only this step's tail (logprobs[-len(token_ids):]); else the full
  // accumulation. nullopt unless sample logprobs were requested.
  if (logprobs_processor.has_value()) {
    co.cumulative_logprob = logprobs_processor->cumulative_logprob();
    const std::optional<SampleLogprobs>& all = logprobs_processor->logprobs();
    if (all.has_value()) {
      if (delta && !all->empty()) {
        const std::size_t take = std::min(co.token_ids.size(), all->size());
        co.logprobs = SampleLogprobs(all->end() - static_cast<std::ptrdiff_t>(take),
                                     all->end());
      } else {
        co.logprobs = *all;
      }
    }
  }
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
                                  int request_index,
                                  std::shared_ptr<RequestOutputCollector> queue,
                                  std::shared_ptr<ParentRequest> parent_req) {
  // output_processor.py:512-541 (T0: the streaming-input re-entry — a request_id
  // already present — is deferred).
  const std::string& request_id = request.request_id;
  if (request_states_.find(request_id) != request_states_.end()) {
    // Upstream routes this case only through the explicitly resumable
    // streaming-input path. That path is deferred; silently reusing the state
    // would enqueue a second core Request against the first collector.
    throw std::invalid_argument("duplicate live request id: " + request_id);
  }

  // :547-548 Track the parent (n>1 parallel sampling) so it outlives its children
  // and can be cleared once the last child finishes (FinishRequest). Stored once
  // per parent; each child's RequestState below references the same instance.
  if (parent_req != nullptr) {
    parent_requests_[parent_req->request_id()] = parent_req;
  }

  RequestState state = RequestState::FromNewRequest(
      tokenizer_, request, std::move(prompt), request_index, stream_interval_,
      std::move(parent_req));
  state.queue = std::move(queue);
  const std::string external_req_id = state.external_req_id;
  request_states_[request_id] =
      std::make_unique<RequestState>(std::move(state));

  // :541 Track external_req_id -> [internal_req_id, ...].
  external_req_ids_[external_req_id].push_back(request_id);
}

OutputProcessorOutput OutputProcessor::process_outputs(
    const EngineCoreOutputs& engine_core_outputs,
    IterationStats* iteration_stats) {
  // output_processor.py:576-693 (the synchronous LLMEngine path; tracing /
  // pooling / streaming-input deferred). `iteration_stats` is folded per output
  // when non-null (llm_engine.py:308 builds it only when log_stats is on).
  OutputProcessorOutput result;
  const double engine_core_timestamp = engine_core_outputs.timestamp;

  for (const EngineCoreOutput& eco : engine_core_outputs.outputs) {
    const std::string& req_id = eco.request_id;
    auto it = request_states_.find(req_id);
    if (it == request_states_.end()) {
      // :609 Ignore output for already-aborted / unknown request.
      continue;
    }
    RequestState& req_state = *it->second;

    const std::vector<int32_t>& new_token_ids = eco.new_token_ids;
    std::optional<FinishReason> finish_reason = eco.finish_reason;
    std::optional<std::string> stop_reason = eco.stop_reason;
    // routed_experts / prefill_stats deferred.

    // 1) Compute stats for this iteration (IterationStats.update_from_output,
    // stats.py:377-449). Whether this output is a prefill step is read BEFORE
    // is_prefilling is flipped below, mirroring upstream ordering.
    const bool is_prefilling_output = req_state.is_prefilling;
    if (iteration_stats != nullptr) {
      const int64_t num_new = static_cast<int64_t>(new_token_ids.size());
      iteration_stats->num_generation_tokens += num_new;
      iteration_stats->iteration_tokens += num_new;

      // update_from_events (stats.py:428-450): fold this output's engine-core
      // events into the request's timing endpoints. QUEUED / SCHEDULED mark the
      // queued->prefill boundary (the first SCHEDULED wins; later resume
      // SCHEDULED events are ignored); each PREEMPTED bumps the per-step
      // preemption counter. LoRA request_waiting/request_running bookkeeping is
      // deferred with LoRA. nullopt (no events this output) -> no-op.
      if (eco.events.has_value()) {
        for (const EngineCoreEvent& ev : *eco.events) {
          switch (ev.type) {
            case EngineCoreEventType::kQueued:
              req_state.queued_ts = ev.timestamp;
              break;
            case EngineCoreEventType::kScheduled:
              if (req_state.scheduled_ts == 0.0) {  // ignore preemptions
                req_state.scheduled_ts = ev.timestamp;
              }
              break;
            case EngineCoreEventType::kPreempted:
              iteration_stats->num_preempted_reqs += 1;
              break;
          }
        }
      }

      if (is_prefilling_output) {
        // prompt_token_stats.update_from_output (stats.py:328-335). No
        // prefill_stats at T0 → num_prompt_tokens_cached stays 0; the whole
        // prompt is the prefill token count.
        const int64_t num_prompt = static_cast<int64_t>(req_state.prompt_len);
        iteration_stats->num_prompt_tokens += num_prompt;
        iteration_stats->iteration_tokens += num_prompt;
        // TTFT = engine_core_timestamp - arrival_time (stats.py:441-443).
        iteration_stats->time_to_first_tokens_iter.push_back(
            engine_core_timestamp - req_state.arrival_time);
        req_state.first_token_ts = engine_core_timestamp;
      } else {
        // ITL = engine_core_timestamp - last_token_ts (stats.py:471-473).
        iteration_stats->inter_token_latencies_iter.push_back(
            engine_core_timestamp - req_state.last_token_ts);
      }
      req_state.num_generation_tokens += num_new;
      req_state.last_token_ts = engine_core_timestamp;
    }

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

    // 3) Compute sample and prompt logprobs for the request (:660-662). Inert
    //    (nullopt) unless the request asked for logprobs.
    if (req_state.logprobs_processor.has_value()) {
      req_state.logprobs_processor->update_from_output(eco);
    }

    // 4) Create and handle the RequestOutput (:650-666). The pooled vector of
    //    a finished pooling request rides through (ARCH-ONE-SURFACE ROW 6).
    std::optional<RequestOutput> request_output = req_state.make_request_output(
        new_token_ids, finish_reason, stop_reason, eco.pooling_output);
    if (request_output.has_value()) {
      // streaming_input deferred (false) -> no finished=false override.
      if (req_state.queue != nullptr) {
        // AsyncLLM: hand off to the per-request collector.
        req_state.queue->put(std::move(*request_output));
      } else {
        // LLMEngine: collect into the synchronous return value.
        result.request_outputs.push_back(std::move(*request_output));
      }
    }

    // Free completed requests (:669-688).
    if (finish_reason.has_value()) {
      // streaming_input deferred (false) -> the finish branch.
      const bool engine_core_finished = eco.Finished();

      // update_from_finished_request (stats.py:401-475) BEFORE FinishRequest
      // invalidates req_state.
      if (iteration_stats != nullptr) {
        FinishedRequestStats f;
        f.finish_reason = FinishReasonToString(*finish_reason);
        f.e2e_latency = engine_core_timestamp - req_state.arrival_time;
        f.num_prompt_tokens = static_cast<int64_t>(req_state.prompt_len);
        f.num_generation_tokens = req_state.num_generation_tokens;
        if (req_state.max_tokens_param.has_value()) {
          f.max_tokens_param = *req_state.max_tokens_param;
        }
        // decode interval = first NEW_TOKEN -> last NEW_TOKEN (stats.py:434).
        const double decode_time =
            req_state.last_token_ts - req_state.first_token_ts;
        f.decode_time = decode_time;
        // TPOT excludes the prefill token (stats.py:438-442).
        f.mean_time_per_output_token =
            req_state.num_generation_tokens - 1 > 0
                ? decode_time / static_cast<double>(
                                    req_state.num_generation_tokens - 1)
                : 0.0;
        // The event-derived intervals (stats.py:459-476), now that QUEUED /
        // SCHEDULED events populate queued_ts / scheduled_ts:
        //   queued    = first SCHEDULED - QUEUED  (time in the WAITING queue)
        //   prefill   = first NEW_TOKEN - first SCHEDULED (any preemption during
        //               prefill is included)
        //   inference = last NEW_TOKEN - first SCHEDULED (= prefill + decode;
        //               any preemption during prefill or decode is included)
        f.queued_time = req_state.scheduled_ts - req_state.queued_ts;
        f.prefill_time = req_state.first_token_ts - req_state.scheduled_ts;
        f.inference_time = req_state.last_token_ts - req_state.scheduled_ts;
        // num_cached_tokens deferred → 0.
        iteration_stats->finished_requests.push_back(f);
      }

      // DIAGNOSTIC (VT_TTFT_DUMP): the async serving frontend calls
      // process_outputs with iteration_stats == nullptr, so the FinishedRequest
      // block above never runs and the per-request queue/prefill/decode split is
      // invisible on the production /metrics path (deliberately 404, see
      // api_server.cpp). This env-gated stderr line reconstructs that split
      // directly from the event-populated req_state timestamps (identical
      // arithmetic to stats.py:459-476), so a serving TTFT attribution can read
      // vLLM's own request_{queue,prefill,decode}_time_seconds against ours on
      // the SAME workload. Reads only; byte-identical generation when unset.
      static const bool kDumpTtftSplit = std::getenv("VT_TTFT_DUMP") != nullptr;
      if (kDumpTtftSplit) {
        // intake = QUEUED - arrival: the async engine-core intake wait, from
        // OutputProcessor registration (arrival_time, MonotonicSeconds set in
        // add_request AFTER tokenize) to the scheduler's QUEUED event (same
        // steady clock). This isolates the engine-side ingress cost (input-queue
        // drain / step cadence) from tokenize (before arrival) and egress.
        const double intake = req_state.queued_ts - req_state.arrival_time;
        const double queued = req_state.scheduled_ts - req_state.queued_ts;
        const double prefill = req_state.first_token_ts - req_state.scheduled_ts;
        const double decode = req_state.last_token_ts - req_state.first_token_ts;
        const double e2e = engine_core_timestamp - req_state.arrival_time;
        std::fprintf(stderr,
                     "TTFTSPLIT rid=%s intake=%.6f queued=%.6f prefill=%.6f "
                     "decode=%.6f e2e=%.6f gen=%lld\n",
                     req_id.c_str(), intake, queued, prefill, decode, e2e,
                     static_cast<long long>(req_state.num_generation_tokens));
      }

      FinishRequest(req_state);  // invalidates req_state / it
      if (!engine_core_finished) {
        // :678 If req not finished in EngineCore but the detokenizer detected a
        // stop string, an abort is needed in EngineCore.
        result.reqs_to_abort.push_back(req_id);
      }
      // tracing deferred.
    }
  }

  return result;
}

std::vector<std::string> OutputProcessor::abort_requests(
    const std::vector<std::string>& request_ids, bool produce_final_output) {
  // output_processor.py:450-510 (T0 1:1 id subset): remove each request and,
  // for AsyncLLM, enqueue its terminal ABORT RequestOutput before cleanup.
  std::vector<std::string> request_ids_to_abort;
  for (const std::string& req_id : request_ids) {
    auto it = request_states_.find(req_id);
    if (it == request_states_.end()) continue;
    RequestState& req_state = *it->second;
    request_ids_to_abort.push_back(req_id);
    if (produce_final_output && req_state.queue != nullptr) {
      std::optional<RequestOutput> final_output = req_state.make_request_output(
          /*new_token_ids=*/{}, FinishReason::kAbort, std::nullopt);
      if (final_output.has_value()) {
        req_state.queue->put(std::move(*final_output));
      }
    }
    FinishRequest(req_state);  // erases the entry (invalidates it).
  }
  return request_ids_to_abort;
}

void OutputProcessor::rollback_requests(
    const std::vector<std::string>& request_ids) noexcept {
  for (const std::string& request_id : request_ids) {
    auto request_it = request_states_.find(request_id);
    if (request_it == request_states_.end()) continue;

    const RequestState& state = *request_it->second;
    auto external_it = external_req_ids_.find(state.external_req_id);
    if (external_it != external_req_ids_.end()) {
      std::vector<std::string>& internal_ids = external_it->second;
      internal_ids.erase(
          std::remove(internal_ids.begin(), internal_ids.end(), request_id),
          internal_ids.end());
      if (internal_ids.empty()) external_req_ids_.erase(external_it);
    }
    if (state.parent_req != nullptr) {
      parent_requests_.erase(state.parent_req->request_id());
    }
    request_states_.erase(request_it);
  }
}

std::vector<std::string> OutputProcessor::abort_all_requests(
    bool produce_final_output) {
  std::vector<std::string> request_ids;
  request_ids.reserve(request_states_.size());
  for (const auto& [request_id, state] : request_states_) {
    (void)state;
    request_ids.push_back(request_id);
  }
  return abort_requests(request_ids, produce_final_output);
}

void OutputProcessor::propagate_error(std::exception_ptr error) {
  for (const auto& [request_id, state] : request_states_) {
    (void)request_id;
    if (state->queue != nullptr) state->queue->put_error(error);
  }
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

  // output_processor.py:720-722: once a parallel-sampling parent has no child
  // left in flight, drop it from parent_requests_ (its aggregation is done).
  if (req_state.parent_req != nullptr && !req_state.parent_req->has_children()) {
    parent_requests_.erase(req_state.parent_req->request_id());
  }

  request_states_.erase(req_id);  // destroys req_state — must be last.
}

}  // namespace vllm::v1
