// Ported from: vllm/entrypoints/openai/completion/serving.py @ e24d1b24
// See serving_completion.h for scope, the return-type design and deferrals.
#include "vllm/entrypoints/openai/serving_completion.h"

#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/beam_search.h"
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/validation_error.h"

namespace vllm::entrypoints::openai {

namespace {

// completion_stream_generator (serving.py:278-474) as a blocking pull source.
// Each next() consumes at most one RequestOutput from this request's collector;
// other requests are processed independently by AsyncLLM's output handler.
class CompletionSseStream final : public SseStream {
 public:
  CompletionSseStream(v1::AsyncLLM& engine, v1::AsyncRequest request,
                      std::string response_id, int64_t created,
                      std::string model, StreamUsageSelection usage)
      : engine_(engine),
        request_(std::move(request)),
        response_id_(std::move(response_id)),
        created_(created),
        model_(std::move(model)),
        usage_(usage) {}

  ~CompletionSseStream() override { abort(); }

  bool WaitOutput(RequestOutput& out, std::string& ping_chunk) {
    const int ping_s = SsePingIntervalSec();
    if (ping_s <= 0) {
      out = engine_.get_output(request_);
      return true;
    }
    auto ready = engine_.get_output_for(
        request_, std::chrono::milliseconds(ping_s * 1000));
    // Shared framing with chat: ping is a standalone comment frame.
    return AssignSseWaitResult(std::move(ready), out, ping_chunk);
  }

  bool next(std::string& chunk) override {
    if (complete_) return false;
    // completion/serving.py:330-442 yields ONE frame per CompletionOutput, and
    // a parallel-sampling frame carries up to n of them (the collector keeps
    // distinct indices side by side, output_processor.cpp Merge). Drain those
    // buffered frames before any terminal frame, so the usage chunk and [DONE]
    // stay last.
    if (!pending_.empty()) {
      chunk = std::move(pending_.front());
      pending_.pop_front();
      return true;
    }
    if (usage_pending_) {
      CompletionStreamResponse frame;
      frame.id = response_id_;
      frame.created = created_;
      frame.model = model_;
      frame.usage = UsageInfo{prompt_tokens_,
                              prompt_tokens_ + previous_num_tokens_,
                              previous_num_tokens_};
      chunk = "data: " + nlohmann::json(frame).dump() + "\n\n";
      usage_pending_ = false;
      done_pending_ = true;
      return true;
    }
    if (done_pending_) {
      chunk = "data: [DONE]\n\n";
      done_pending_ = false;
      complete_ = true;
      return true;
    }

    for (;;) {
      RequestOutput response;
      if (!WaitOutput(response, chunk)) {
        return true;  // pure SSE ping
      }
      prompt_tokens_ = static_cast<int>(response.prompt_token_ids.size());
      if (response.outputs.empty()) {
        if (response.finished) {
          engine_finished_ = true;
          if (usage_.include_usage) {
            usage_pending_ = true;
          } else {
            done_pending_ = true;
          }
        }
        if (usage_pending_ || done_pending_) return next(chunk);
        continue;
      }

      // :330-331 `for output in res.outputs:` / `i = output.index + prompt_idx
      // * num_choices` — every output in the frame gets its own chunk, not just
      // the first.
      for (const CompletionOutput& output : response.outputs) {
        const std::string delta_text = SanitizeUtf8(output.text);
        // :376-380 chunked-prefill hold-back, read PER CHOICE
        // (previous_num_tokens[i]). Preserve a terminal empty chunk so clients
        // still observe finish_reason.
        int& choice_tokens = tokens_by_index_[output.index];
        if (delta_text.empty() && output.token_ids.empty() &&
            choice_tokens == 0 && !response.finished) {
          continue;
        }
        choice_tokens += static_cast<int>(output.token_ids.size());
        previous_num_tokens_ += static_cast<int>(output.token_ids.size());

        CompletionResponseStreamChoice choice;
        choice.index = output.index;
        choice.text = delta_text;
        choice.finish_reason = output.finish_reason;

        CompletionStreamResponse frame;
        frame.id = response_id_;
        frame.created = created_;
        frame.model = model_;
        frame.choices.push_back(std::move(choice));
        if (usage_.include_continuous_usage) {
          // :433-440 continuous usage counts THIS choice's completions
          // (previous_num_tokens[i]); the trailing usage frame sums them.
          frame.usage = UsageInfo{prompt_tokens_,
                                  prompt_tokens_ + choice_tokens,
                                  choice_tokens};
        }
        pending_.push_back("data: " + nlohmann::json(frame).dump() + "\n\n");
      }
      if (pending_.empty()) continue;  // every output was held back

      if (response.finished) {
        engine_finished_ = true;
        if (usage_.include_usage) {
          usage_pending_ = true;
        } else {
          done_pending_ = true;
        }
      }
      chunk = std::move(pending_.front());
      pending_.pop_front();
      return true;
    }
  }

  void abort() override {
    if (complete_ || engine_finished_ || aborted_) return;
    aborted_ = true;
    engine_.abort(request_.request_id);
  }

 private:
  v1::AsyncLLM& engine_;
  v1::AsyncRequest request_;
  std::string response_id_;
  int64_t created_ = 0;
  std::string model_;
  StreamUsageSelection usage_;
  int prompt_tokens_ = 0;
  // The SUM over choices, for the trailing usage frame (:445
  // `sum(previous_num_tokens)`).
  int previous_num_tokens_ = 0;
  // previous_num_tokens[i] (:296): per-choice counts, which drive the
  // chunked-prefill hold-back and continuous usage. Degenerate (one entry) for
  // the n == 1 default, where it equals previous_num_tokens_.
  std::map<int, int> tokens_by_index_;
  // Frames built from one multi-output RequestOutput, waiting to be handed out
  // one next() call at a time.
  std::deque<std::string> pending_;
  bool usage_pending_ = false;
  bool done_pending_ = false;
  bool engine_finished_ = false;
  bool complete_ = false;
  bool aborted_ = false;
};

}  // namespace

OpenAIServingCompletion::OpenAIServingCompletion(v1::LLMEngine& engine,
                                                 std::string served_model_name,
                                                 bool enable_force_include_usage)
    : sync_engine_(&engine),
      served_model_name_(std::move(served_model_name)),
      enable_force_include_usage_(enable_force_include_usage) {}

OpenAIServingCompletion::OpenAIServingCompletion(
    v1::AsyncLLM& engine, std::string served_model_name,
    bool enable_force_include_usage)
    : async_engine_(&engine),
      served_model_name_(std::move(served_model_name)),
      enable_force_include_usage_(enable_force_include_usage) {}

CompletionResult OpenAIServingCompletion::create_completion(
    const CompletionRequest& request) {
  // request_id = f"cmpl-{...}" (completion/serving.py:143); created_time =
  // int(time.time()) (:144).
  const std::string request_id =
      "cmpl-" + std::to_string(request_counter_.fetch_add(1));
  const auto created_time = static_cast<int64_t>(std::time(nullptr));
  const std::string model_name =
      request.model.has_value() ? *request.model : served_model_name_;
  const StreamUsageSelection usage = ShouldIncludeUsage(
      request.stream_options, enable_force_include_usage_);

  // ── best_of > n + stream (SAMPLE-BEST-OF, local extension) ───────────────
  // best_of asks the engine for MORE candidates than the response returns and
  // the endpoint keeps the top `n` by FINAL cumulative logprob (SelectBestOf,
  // below). A delta stream has no final score until the last token of each
  // candidate, and frames already sent cannot be recalled, so there is no way
  // for the streaming arm to deliver the choices the non-streaming arm delivers
  // for the same body. The alternatives are worse than a refusal: buffering
  // every frame to the end turns a streaming request into a blocking one with
  // no signal to the client, and streaming an arbitrary n of the best_of
  // children returns a choice set that merely LOOKS like the ranked one. A 400
  // is a disagreement the client can see; a wrong choice set is not.
  //
  // Upstream has no `best_of` field to mirror (0.26 dropped it; protocol.h:201).
  // It does carry exactly one other ranked-selection mode, and it refuses that
  // with `stream` for the same reason and at the same point in the handler
  // (completion/serving.py:136-139). This mirrors that refusal rather than
  // inventing a third semantics.
  //
  // NARROW: only best_of > n is refused. best_of unset or == n has nothing to
  // rank, so it streams n children exactly as a plain n>1 request does.
  if (request.stream && request.best_of.has_value() &&
      *request.best_of > request.n) {
    throw vllm::v1::InputValidationError(
        "best_of > n is not currently supported with stream=true: the top-n "
        "ranking needs every candidate's final cumulative logprob, which a "
        "delta stream cannot know before the last token. Send the same request "
        "without `stream`, or set best_of == n.");
  }

  // ── use_beam_search (completion/serving.py:173-205) ──────────────────────
  // Route the request through the merged BeamSearch driver instead of the
  // sampler: beam_width == n, returns the n best beams as choices. The
  // production server holds an AsyncLLM → BeamSearchAsync (online.py); the sync
  // LLMEngine seam (tests / offline) → BeamSearch (offline.py). Both call the
  // SAME model-free scoring/selection, so the choices are identical. Streaming
  // beam search is rejected exactly as upstream (serving.py:136-139).
  if (request.use_beam_search) {
    if (request.stream) {
      throw std::runtime_error(
          "Streaming is not currently supported with beam search");
    }
    if (beam_tokenizer_ == nullptr ||
        (async_engine_ == nullptr && sync_engine_ == nullptr)) {
      throw std::runtime_error(
          "beam search requires an engine and a tokenizer");
    }
    const int max_tok = request.max_tokens.value_or(16);
    const BeamSearchParams params =
        request.to_beam_search_params(max_tok, &default_sampling_params_);
    const std::vector<int32_t> prompt_ids =
        beam_tokenizer_->Encode(request.prompt);
    const BeamSearchOutput beams =
        async_engine_ != nullptr
            ? BeamSearchAsync(*async_engine_, prompt_ids, params,
                              beam_eos_token_id_, beam_tokenizer_)
            : BeamSearch(*sync_engine_, prompt_ids, params, beam_eos_token_id_,
                         beam_tokenizer_);

    CompletionResponse response;
    response.id = request_id;
    response.created = created_time;
    response.model = model_name;
    const int num_prompt_tokens = static_cast<int>(prompt_ids.size());
    int num_generated_tokens = 0;
    for (std::size_t i = 0; i < beams.sequences.size(); ++i) {
      const BeamSearchSequence& beam = beams.sequences[i];
      CompletionResponseChoice choice;
      choice.index = static_cast<int>(i);
      choice.text = SanitizeUtf8(beam.text.value_or(""));
      choice.finish_reason = beam.finish_reason;
      response.choices.push_back(std::move(choice));
      // Generated tokens are the beam's tokens past the shared prompt.
      if (beam.tokens.size() > prompt_ids.size()) {
        num_generated_tokens +=
            static_cast<int>(beam.tokens.size() - prompt_ids.size());
      }
    }
    response.usage.prompt_tokens = num_prompt_tokens;
    response.usage.completion_tokens = num_generated_tokens;
    response.usage.total_tokens = num_prompt_tokens + num_generated_tokens;

    CompletionResult result;
    result.streaming = false;
    result.response = std::move(response);
    return result;
  }

  // request → SamplingParams. to_sampling_params sets output_kind to kDelta
  // when stream, kFinalOnly otherwise (protocol.cpp) — matching upstream's
  // per-request RequestOutputKind (completion/serving.py:174).
  SamplingParams sampling_params =
      request.to_sampling_params(std::nullopt, &default_sampling_params_);

  // T0: single prompt, single choice (n == 1). The engine sub-request id is
  // f"{request_id}-{i}" upstream (:179); here i == 0.
  const std::string engine_request_id = request_id + "-0";

  // W2 production path: enqueue and return immediately with a live pull source.
  // The HTTP provider blocks on this request's collector one chunk at a time.
  if (async_engine_ != nullptr && request.stream) {
    v1::AsyncRequest async_request = async_engine_->add_request(
        engine_request_id, request.prompt, std::move(sampling_params),
        request.priority);
    CompletionResult result;
    result.streaming = true;
    try {
      result.sse_stream = std::make_shared<CompletionSseStream>(
          *async_engine_, async_request, request_id, created_time, model_name,
          usage);
    } catch (...) {
      async_engine_->abort(async_request.request_id);
      throw;
    }
    return result;
  }

  if (request.stream) {
    // ── Streaming (completion_stream_generator, :278) ─────────────────────
    // Drive the engine over DELTA RequestOutputs; format one
    // CompletionStreamResponse per non-empty delta, then `data: [DONE]\n\n`.
    CompletionResult result;
    result.streaming = true;

    int previous_num_tokens = 0;
    int num_prompt_tokens = 0;
    if (sync_engine_ == nullptr) {
      throw std::runtime_error("completion handler has no engine");
    }
    sync_engine_->add_request(engine_request_id, request.prompt,
                              std::move(sampling_params), request.priority);
    while (sync_engine_->has_unfinished_requests()) {
      for (const RequestOutput& res : sync_engine_->step()) {
        if (res.request_id != engine_request_id) continue;
        num_prompt_tokens = static_cast<int>(res.prompt_token_ids.size());
        for (const CompletionOutput& output : res.outputs) {
          const std::string& delta_text = output.text;
          // :368-374 chunked-prefill: skip empty chunks (no text, no tokens,
          // and nothing emitted yet).
          if (delta_text.empty() && output.token_ids.empty() &&
              previous_num_tokens == 0) {
            continue;
          }
          previous_num_tokens += static_cast<int>(output.token_ids.size());

          CompletionResponseStreamChoice choice;
          choice.index = 0;  // output.index + prompt_idx * num_choices; T0 == 0
          // SanitizeUtf8: our detokenizer keeps raw bytes (not upstream's lossy
          // str), so a split/invalid multibyte run here would make the chunk's
          // json dump() below throw → 500. Replace invalid subparts with U+FFFD
          // (matches upstream str semantics). See serving_utils.h.
          choice.text = SanitizeUtf8(delta_text);
          choice.finish_reason = output.finish_reason;

          CompletionStreamResponse chunk;
          chunk.id = request_id;
          chunk.created = created_time;
          chunk.model = model_name;
          chunk.choices.push_back(std::move(choice));
          if (usage.include_continuous_usage) {
            chunk.usage = UsageInfo{num_prompt_tokens,
                                    num_prompt_tokens + previous_num_tokens,
                                    previous_num_tokens};
          }

          result.sse_chunks.push_back(
              "data: " + nlohmann::json(chunk).dump() + "\n\n");
        }
      }
    }
    if (usage.include_usage) {
      CompletionStreamResponse usage_chunk;
      usage_chunk.id = request_id;
      usage_chunk.created = created_time;
      usage_chunk.model = model_name;
      usage_chunk.usage =
          UsageInfo{num_prompt_tokens,
                    num_prompt_tokens + previous_num_tokens,
                    previous_num_tokens};
      result.sse_chunks.push_back(
          "data: " + nlohmann::json(usage_chunk).dump() + "\n\n");
    }
    result.sse_chunks.push_back("data: [DONE]\n\n");
    return result;
  }

  // ── Non-streaming (request_output_to_completion_response, :475) ──────────
  const RequestOutput final_res = async_engine_ != nullptr
      ? async_engine_->generate(request.prompt, std::move(sampling_params),
                                engine_request_id, request.priority)
      : sync_engine_->generate(request.prompt, std::move(sampling_params),
                               engine_request_id, request.priority);

  CompletionResponse response;
  response.id = request_id;
  response.created = created_time;
  response.model = model_name;

  int num_prompt_tokens = static_cast<int>(final_res.prompt_token_ids.size());
  int num_generated_tokens = 0;
  // prompt_logprobs (completion/serving.py:520): clamp ONCE, before any choice
  // reads it. Upstream clamps in place and then hands the same object to every
  // choice at :588, so a single clamp covers the n>1 fan-out too.
  std::optional<vllm::PromptLogprobs> prompt_logprobs = final_res.prompt_logprobs;
  ClampPromptLogprobs(prompt_logprobs);
  // SAMPLE-BEST-OF: when best_of > n the engine produced best_of children; keep
  // the top-n by cumulative logprob. Guarded on request.best_of so the default
  // (and plain n>1) path binds `outs` to final_res.outputs with NO copy/re-rank.
  std::vector<CompletionOutput> selected_outputs;
  const bool trim_best_of =
      request.best_of.has_value() && *request.best_of > request.n;
  const std::vector<CompletionOutput>& outs =
      trim_best_of
          ? (selected_outputs = SelectBestOf(final_res.outputs, request.n))
          : final_res.outputs;
  for (const CompletionOutput& output : outs) {
    CompletionResponseChoice choice;
    choice.index = static_cast<int>(response.choices.size());
    choice.text = SanitizeUtf8(output.text);  // echo deferred; see serving_utils.h
    // logprobs (completion/serving.py:559-567): build the payload when the
    // request asked for logprobs and the engine produced them. echo (prepending
    // prompt tokens/prompt_logprobs) stays deferred — see SAMPLE-PROMPT-LOGPROBS.
    if (request.logprobs.has_value() && output.logprobs.has_value()) {
      choice.logprobs = BuildCompletionLogProbs(output.token_ids, *output.logprobs,
                                                *request.logprobs);
    }
    choice.finish_reason = output.finish_reason;
    // prompt_logprobs (completion/serving.py:588): set on EVERY choice, inside
    // the per-output loop, so an n>1 response repeats the one prompt payload.
    choice.prompt_logprobs = prompt_logprobs;
    response.choices.push_back(std::move(choice));
    num_generated_tokens += static_cast<int>(output.token_ids.size());
  }

  // UsageInfo (:576).
  response.usage.prompt_tokens = num_prompt_tokens;
  response.usage.completion_tokens = num_generated_tokens;
  response.usage.total_tokens = num_prompt_tokens + num_generated_tokens;

  CompletionResult result;
  result.streaming = false;
  result.response = std::move(response);
  return result;
}

}  // namespace vllm::entrypoints::openai
