// Ported from: vllm/entrypoints/openai/completion/serving.py @ e24d1b24
// See serving_completion.h for scope, the return-type design and deferrals.
#include "vllm/entrypoints/openai/serving_completion.h"

#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/serving_utils.h"

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

  bool next(std::string& chunk) override {
    if (complete_) return false;
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
      RequestOutput response = engine_.get_output(request_);
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

      const CompletionOutput& output = response.outputs.front();
      const std::string delta_text = SanitizeUtf8(output.text);
      // completion/serving.py:368-374 chunked-prefill hold-back. Preserve a
      // terminal empty chunk so clients still observe finish_reason.
      if (delta_text.empty() && output.token_ids.empty() &&
          previous_num_tokens_ == 0 && !response.finished) {
        continue;
      }
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
        frame.usage = UsageInfo{prompt_tokens_,
                                prompt_tokens_ + previous_num_tokens_,
                                previous_num_tokens_};
      }
      chunk = "data: " + nlohmann::json(frame).dump() + "\n\n";

      if (response.finished) {
        engine_finished_ = true;
        if (usage_.include_usage) {
          usage_pending_ = true;
        } else {
          done_pending_ = true;
        }
      }
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
  int previous_num_tokens_ = 0;
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

  // request → SamplingParams. to_sampling_params sets output_kind to kDelta
  // when stream, kFinalOnly otherwise (protocol.cpp) — matching upstream's
  // per-request RequestOutputKind (completion/serving.py:174).
  SamplingParams sampling_params = request.to_sampling_params();

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
  for (const CompletionOutput& output : final_res.outputs) {
    CompletionResponseChoice choice;
    choice.index = static_cast<int>(response.choices.size());
    choice.text = SanitizeUtf8(output.text);  // echo deferred; see serving_utils.h
    choice.finish_reason = output.finish_reason;
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
