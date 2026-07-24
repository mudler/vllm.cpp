// Ported from: vllm/entrypoints/openai/chat_completion/serving.py @ e24d1b24
// See serving_chat.h for scope, the chat-prompt seam and deferrals.
#include "vllm/entrypoints/openai/serving_chat.h"

#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/entrypoints/openai/tool_parsers/structural_tags.h"

namespace vllm::entrypoints::openai {

namespace {
// get_chat_request_role (chat_completion/serving.py:399): the response role is
// self.response_role ("assistant") when add_generation_prompt (the T0 default).
constexpr const char* kAssistantRole = "assistant";

// Whether tool_choice selects a single named function (finish_reason stays the
// model's own — "stop" — for named calls; chat_completion/serving.py:688,935).
bool IsNamedToolChoice(const ChatCompletionRequest& request) {
  return request.tool_choice.has_value() &&
         request.tool_choice->mode == "function";
}

}  // namespace

std::string DefaultChatPromptFallback(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt,
    const std::vector<ChatCompletionToolsParam>& /*tools*/) {
  // T0 SEAM (M3.2 swaps in the real chat-template renderer). A simple
  // "<role>: <content>\n" join + an "assistant:" generation prompt. This is NOT
  // a model chat template — it exists only so the chat path is end-to-end
  // exercisable. `tools` are ignored here (rendered by the real template only).
  std::string prompt;
  for (const ChatMessage& m : messages) {
    prompt += m.role;
    prompt += ": ";
    if (m.content.has_value()) prompt += *m.content;
    prompt += "\n";
  }
  if (add_generation_prompt) {
    prompt += "assistant:";
  }
  return prompt;
}

bool ToolsEnabled(const ChatCompletionRequest& request) {
  if (!request.tools.has_value() || request.tools->empty()) return false;
  // tool_choice absent → defaults to "auto" when tools present (enabled).
  if (request.tool_choice.has_value() &&
      request.tool_choice->mode == "none") {
    return false;
  }
  return true;
}

std::optional<nlohmann::json> ToolChoiceStructuralTagSpec(
    const ChatCompletionRequest& request) {
  // Thin wrapper preserved for source compatibility (existing callers/tests).
  // The Hermes builder now lives in the per-family registry (structural_tags.h);
  // this delegates to the "hermes" family exactly.
  return ToolChoiceStructuralTagSpecFor("hermes", request);
}

void ApplyToolChoiceStructuredOutput(const ChatCompletionRequest& request,
                                     SamplingParams& sampling_params) {
  // Thin wrapper preserved for source compatibility — delegates to the Hermes
  // family. create_chat_completion calls the per-family overload with the active
  // tool_parser_name so the constraint matches the model's native syntax.
  ApplyToolChoiceStructuredOutput("hermes", request, sampling_params);
}

ShapedChatMessage ShapeChatMessage(
    const std::string& role, const std::string& model_output,
    std::optional<std::string> output_finish_reason,
    const ChatCompletionRequest& request, ToolParser* parser,
    ReasoningParser* reasoning_parser) {
  ShapedChatMessage shaped;
  shaped.message.role = role;

  // chat_completion/serving.py:858-866. Reasoning runs FIRST: strip the
  // chain-of-thought, then tool detection operates ONLY on the user-visible
  // content. `content_span` is what the tool parser (and a plain message) see;
  // `has_content_span` distinguishes an absent post-reasoning span (pure
  // reasoning, content=None) from an empty one.
  std::string content_span = model_output;
  bool has_content_span = true;
  std::optional<std::string> reasoning;
  if (reasoning_parser != nullptr) {
    const ExtractedReasoning er =
        reasoning_parser->extract_reasoning(model_output, request);
    reasoning = er.reasoning;
    has_content_span = er.content.has_value();
    content_span = er.content.value_or(std::string());
  }
  const auto attach_reasoning = [&]() {
    if (reasoning.has_value() && !reasoning->empty()) {
      // Same contract as every content assignment below: the non-stream path
      // splits the RAW detokenizer output, so the reasoning span needs the
      // same UTF-8 sanitization content gets (think markers are ASCII, so a
      // multi-byte sequence can never straddle the split boundary).
      shaped.message.reasoning = SanitizeUtf8(*reasoning);
    }
  };

  // chat_completion/serving.py:899-923 (the "auto" path). When tools are active
  // and a parser exists, extract; on tools_called, the finish_reason becomes
  // "tool_calls" (:936).
  if (parser != nullptr && ToolsEnabled(request)) {
    const ExtractedToolCallInformation info =
        parser->extract_tool_calls(content_span, request);
    if (info.tools_called && !info.tool_calls.empty()) {
      if (info.content.has_value()) {
        shaped.message.content = SanitizeUtf8(*info.content);
      } else {
        shaped.message.content = std::nullopt;
      }
      shaped.message.tool_calls = info.tool_calls;
      shaped.finish_reason = "tool_calls";
      attach_reasoning();
      return shaped;
    }
    // tools_called=false → fall through to a plain-content message with the
    // parser's content (the post-reasoning span).
    shaped.message.content = SanitizeUtf8(info.content.value_or(content_span));
    shaped.finish_reason = std::move(output_finish_reason);
    if (!shaped.finish_reason.has_value()) shaped.finish_reason = "stop";
    attach_reasoning();
    return shaped;
  }

  // No tools: plain content message (:881). With a reasoning parser, an absent
  // post-reasoning span means content=null (pure reasoning turn).
  if (reasoning_parser != nullptr && !has_content_span) {
    shaped.message.content = std::nullopt;
  } else {
    shaped.message.content = SanitizeUtf8(content_span);
  }
  shaped.finish_reason = std::move(output_finish_reason);
  if (!shaped.finish_reason.has_value()) shaped.finish_reason = "stop";
  attach_reasoning();
  return shaped;
}

std::optional<DeltaMessage> ShapeChatDelta(const std::string& previous_text,
                                           const std::string& current_text,
                                           const std::string& delta_text,
                                           const ChatCompletionRequest& request,
                                           ToolParser* parser,
                                           ReasoningParser* reasoning_parser) {
  // chat_completion/serving.py:587-613. Reasoning runs BEFORE tools. The
  // reasoning parser splits this delta into a reasoning span (rides on
  // DeltaMessage.reasoning) and a post-reasoning CONTENT span; only that content
  // span is routed to the tool parser, so the tool parse never sees the thoughts.
  if (reasoning_parser != nullptr) {
    std::optional<DeltaMessage> rd = reasoning_parser->extract_reasoning_streaming(
        previous_text, current_text, delta_text, request);
    if (!rd.has_value()) return std::nullopt;  // lone marker swallowed

    const bool tools = (parser != nullptr && ToolsEnabled(request));
    if (!tools) return rd;  // reasoning and/or content deltas straight through

    const std::string content_delta = rd->content.value_or(std::string());
    const std::optional<std::string> reasoning_piece = rd->reasoning;
    if (content_delta.empty()) {
      if (reasoning_piece.has_value()) {
        DeltaMessage m;
        m.reasoning = reasoning_piece;
        return m;
      }
      return std::nullopt;
    }

    // Content-space offsets for the (stateful) tool parser, derived from the
    // reasoning split of the accumulated prefix: the content the user has seen
    // so far, without any reasoning.
    const std::string content_prev =
        reasoning_parser->extract_reasoning(previous_text, request)
            .content.value_or(std::string());
    const std::string content_curr = content_prev + content_delta;
    std::optional<DeltaMessage> td = parser->extract_tool_calls_streaming(
        content_prev, content_curr, content_delta, request);

    DeltaMessage out = td.value_or(DeltaMessage{});
    if (reasoning_piece.has_value()) out.reasoning = reasoning_piece;
    if (!td.has_value() && !reasoning_piece.has_value() &&
        !out.content.has_value()) {
      return std::nullopt;
    }
    return out;
  }

  // No reasoning parser: with a tool parser, run the streaming parse; otherwise
  // emit a plain content delta.
  if (parser != nullptr && ToolsEnabled(request)) {
    return parser->extract_tool_calls_streaming(previous_text, current_text,
                                                delta_text, request);
  }
  DeltaMessage msg;
  msg.content = delta_text;
  return msg;
}

namespace {

// chat_completion_stream_generator (serving.py:404-802) as W2's live,
// pull-based SSE source. Continuous usage waits for and buffers the first
// result so the role frame carries a native prompt-token count; subsequent
// calls block only on this request's collector.
class ChatSseStream final : public SseStream {
 public:
  ChatSseStream(v1::AsyncLLM& engine, v1::AsyncRequest async_request,
                std::string response_id, int64_t created, std::string model,
                ChatCompletionRequest request,
                std::unique_ptr<ToolParser> parser,
                std::unique_ptr<ReasoningParser> reasoning_parser,
                bool named_tool_choice, StreamUsageSelection usage)
      : engine_(engine),
        async_request_(std::move(async_request)),
        response_id_(std::move(response_id)),
        created_(created),
        model_(std::move(model)),
        request_(std::move(request)),
        parser_(std::move(parser)),
        reasoning_parser_(std::move(reasoning_parser)),
        named_tool_choice_(named_tool_choice),
        usage_(usage) {}

  ~ChatSseStream() override { abort(); }

  bool next(std::string& chunk) override {
    if (complete_) return false;
    if (role_pending_) {
      // Upstream emits the role frame on the first engine result. We only need
      // to buffer that result when continuous usage requires its native prompt
      // count; the default path retains its immediately available role frame.
      if (usage_.include_continuous_usage) {
        for (;;) {
          RequestOutput response = engine_.get_output(async_request_);
          prompt_tokens_ =
              static_cast<int>(response.prompt_token_ids.size());
          if (!response.outputs.empty() || response.finished) {
            buffered_response_ = std::move(response);
            break;
          }
        }
      }
      role_pending_ = false;
      ChatCompletionResponseStreamChoice choice;
      choice.index = 0;
      choice.delta.role = kAssistantRole;
      choice.delta.content = "";
      choice.finish_reason = std::nullopt;
      ChatCompletionStreamResponse frame;
      frame.id = response_id_;
      frame.created = created_;
      frame.model = model_;
      frame.choices.push_back(std::move(choice));
      if (usage_.include_continuous_usage) {
        frame.usage = UsageInfo{prompt_tokens_, prompt_tokens_, 0};
      }
      chunk = "data: " + nlohmann::json(frame).dump() + "\n\n";
      return true;
    }
    if (usage_pending_) {
      ChatCompletionStreamResponse frame;
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
      if (buffered_response_.has_value()) {
        response = std::move(*buffered_response_);
        buffered_response_.reset();
      } else {
        response = engine_.get_output(async_request_);
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
          return next(chunk);
        }
        continue;
      }

      const CompletionOutput& output = response.outputs.front();
      const std::string delta_text = SanitizeUtf8(output.text);
      if (delta_text.empty() && output.token_ids.empty() &&
          previous_num_tokens_ == 0 && !response.finished) {
        continue;
      }
      previous_num_tokens_ += static_cast<int>(output.token_ids.size());
      const std::string current_text = previous_text_ + delta_text;
      std::optional<DeltaMessage> delta = ShapeChatDelta(
          previous_text_, current_text, delta_text, request_, parser_.get(),
          reasoning_parser_.get());
      previous_text_ = current_text;

      if (delta.has_value() && delta->tool_calls.has_value() &&
          !delta->tool_calls->empty()) {
        tools_streamed_ = true;
      }
      if (!delta.has_value()) {
        if (!response.finished) continue;
        delta = DeltaMessage{};
      }

      ChatCompletionResponseStreamChoice choice;
      choice.index = output.index;
      choice.delta = std::move(*delta);
      if (response.finished) {
        choice.finish_reason = tools_streamed_ && !named_tool_choice_
                                   ? std::optional<std::string>("tool_calls")
                                   : output.finish_reason;
      } else {
        choice.finish_reason = std::nullopt;
      }

      ChatCompletionStreamResponse frame;
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
    engine_.abort(async_request_.request_id);
  }

 private:
  v1::AsyncLLM& engine_;
  v1::AsyncRequest async_request_;
  std::string response_id_;
  int64_t created_ = 0;
  std::string model_;
  ChatCompletionRequest request_;
  std::unique_ptr<ToolParser> parser_;
  std::unique_ptr<ReasoningParser> reasoning_parser_;
  bool named_tool_choice_ = false;
  StreamUsageSelection usage_;
  std::optional<RequestOutput> buffered_response_;
  int prompt_tokens_ = 0;
  bool role_pending_ = true;
  bool usage_pending_ = false;
  bool done_pending_ = false;
  bool engine_finished_ = false;
  bool complete_ = false;
  bool aborted_ = false;
  bool tools_streamed_ = false;
  int previous_num_tokens_ = 0;
  std::string previous_text_;
};

}  // namespace

std::unique_ptr<ToolParser> OpenAIServingChat::MakeToolParser(
    const ChatCompletionRequest& request) const {
  if (tool_parser_name_.empty() || !ToolsEnabled(request)) return nullptr;
  return get_tool_parser(tool_parser_name_);
}

std::unique_ptr<ReasoningParser> OpenAIServingChat::MakeReasoningParser() const {
  if (reasoning_parser_name_.empty()) return nullptr;
  return get_reasoning_parser(reasoning_parser_name_);
}

OpenAIServingChat::OpenAIServingChat(v1::LLMEngine& engine,
                                     std::string served_model_name,
                                     ChatPromptFn prompt_fn,
                                     std::string tool_parser_name,
                                     std::string reasoning_parser_name,
                                     bool enable_force_include_usage)
    : sync_engine_(&engine),
      served_model_name_(std::move(served_model_name)),
      prompt_fn_(std::move(prompt_fn)),
      tool_parser_name_(std::move(tool_parser_name)),
      reasoning_parser_name_(std::move(reasoning_parser_name)),
      enable_force_include_usage_(enable_force_include_usage) {}

OpenAIServingChat::OpenAIServingChat(v1::AsyncLLM& engine,
                                     std::string served_model_name,
                                     ChatPromptFn prompt_fn,
                                     std::string tool_parser_name,
                                     std::string reasoning_parser_name,
                                     bool enable_force_include_usage)
    : async_engine_(&engine),
      served_model_name_(std::move(served_model_name)),
      prompt_fn_(std::move(prompt_fn)),
      tool_parser_name_(std::move(tool_parser_name)),
      reasoning_parser_name_(std::move(reasoning_parser_name)),
      enable_force_include_usage_(enable_force_include_usage) {}

ChatCompletionResult OpenAIServingChat::create_chat_completion(
    const ChatCompletionRequest& request) {
  // request_id = f"chatcmpl-{...}" (chat_completion/serving.py:268); created =
  // int(time.time()) (:416 / :816).
  const std::string request_id =
      "chatcmpl-" + std::to_string(request_counter_.fetch_add(1));
  const auto created_time = static_cast<int64_t>(std::time(nullptr));
  const std::string model_name =
      request.model.has_value() ? *request.model : served_model_name_;
  const StreamUsageSelection usage = ShouldIncludeUsage(
      request.stream_options, enable_force_include_usage_);

  // Build the prompt from messages via the seam (add_generation_prompt is the
  // upstream default True). The tools are passed through so the chat template's
  // `{% if tools %}` branch renders the function schemas (upstream
  // apply_chat_template(..., tools=...)); empty when tools are absent/disabled.
  const std::vector<ChatCompletionToolsParam> tools =
      ToolsEnabled(request) ? *request.tools
                            : std::vector<ChatCompletionToolsParam>{};
  const std::string prompt =
      prompt_fn_(request.messages, /*add_generation_prompt=*/true, tools);

  // One tool parser per request (the streaming parse is stateful); null when the
  // request has no tools (or the parser is disabled).
  std::unique_ptr<ToolParser> parser = MakeToolParser(request);
  // One reasoning parser per request (streaming may be stateful, olmo3); null
  // when disabled (empty reasoning_parser_name_). Independent of tools.
  std::unique_ptr<ReasoningParser> reasoning_parser = MakeReasoningParser();
  const bool named_tool_choice = IsNamedToolChoice(request);

  SamplingParams sampling_params = request.to_sampling_params();

  // tool_choice -> a structural-tag constraint (structured_outputs.structural_tag)
  // before add_request, built for the ACTIVE tool parser family
  // (tool_parser_name_) so required/named force the model's OWN native tool
  // syntax (not always Hermes). auto -> a LAZY tag (free text until the family's
  // begin marker, then the call is constrained — NOT forced); required/named ->
  // a FORCED tag (a valid, parser-extractable native call from token 0); an
  // unmapped family or none / no tools: no-op. Mirrors upstream's per-model
  // xgrammar StructuralTag (chat_completion/serving.py ->
  // tool_parsers/structural_tag_registry.py::get_model_structural_tag).
  ApplyToolChoiceStructuredOutput(tool_parser_name_, request, sampling_params);

  // Single prompt → sub_request_id == request_id (chat_completion/serving.py:293).
  const std::string engine_request_id = request_id;

  if (request.stream) {
    if (async_engine_ != nullptr) {
      v1::AsyncRequest async_request = async_engine_->add_request(
          engine_request_id, prompt, std::move(sampling_params),
          request.priority);
      ChatCompletionResult result;
      result.streaming = true;
      try {
        result.sse_stream = std::make_shared<ChatSseStream>(
            *async_engine_, async_request, request_id, created_time, model_name,
            request, std::move(parser), std::move(reasoning_parser),
            named_tool_choice, usage);
      } catch (...) {
        async_engine_->abort(async_request.request_id);
        throw;
      }
      return result;
    }

    // ── Streaming (chat_completion_stream_generator, :404) ────────────────
    ChatCompletionResult result;
    result.streaming = true;

    // Content / tool-call deltas, then the finish chunk (:559-716). The tool
    // parser (when present) drives per-delta shaping; `previous_text` is the
    // accumulated output the streaming parser re-parses each step (:615).
    int previous_num_tokens = 0;
    int num_prompt_tokens = 0;
    std::string previous_text;
    bool tools_streamed = false;
    bool role_emitted = false;
    if (sync_engine_ == nullptr) {
      throw std::runtime_error("chat handler has no engine");
    }
    sync_engine_->add_request(engine_request_id, prompt,
                              std::move(sampling_params), request.priority);
    while (sync_engine_->has_unfinished_requests()) {
      for (const RequestOutput& res : sync_engine_->step()) {
        if (res.request_id != engine_request_id) continue;
        num_prompt_tokens = static_cast<int>(res.prompt_token_ids.size());
        if (!role_emitted) {
          role_emitted = true;
          ChatCompletionResponseStreamChoice role_choice;
          role_choice.index = 0;
          role_choice.delta.role = kAssistantRole;
          role_choice.delta.content = "";
          role_choice.finish_reason = std::nullopt;
          ChatCompletionStreamResponse role_chunk;
          role_chunk.id = request_id;
          role_chunk.created = created_time;
          role_chunk.model = model_name;
          role_chunk.choices.push_back(std::move(role_choice));
          if (usage.include_continuous_usage) {
            role_chunk.usage = UsageInfo{num_prompt_tokens,
                                         num_prompt_tokens, 0};
          }
          result.sse_chunks.push_back(
              "data: " + nlohmann::json(role_chunk).dump() + "\n\n");
        }
        for (const CompletionOutput& output : res.outputs) {
          // SanitizeUtf8 (see serving_utils.h): raw-byte deltas may carry an
          // invalid/split multibyte run that would make dump() below throw.
          const std::string delta_text = SanitizeUtf8(output.text);
          // :579-585 chunked-prefill: skip empty chunks.
          if (delta_text.empty() && output.token_ids.empty() &&
              previous_num_tokens == 0) {
            continue;
          }
          previous_num_tokens += static_cast<int>(output.token_ids.size());
          const bool finished = output.finish_reason.has_value();

          const std::string current_text = previous_text + delta_text;
          std::optional<DeltaMessage> delta_message = ShapeChatDelta(
              previous_text, current_text, delta_text, request, parser.get(),
              reasoning_parser.get());
          previous_text = current_text;

          // :598-599 — a tool-call delta flips the finish_reason to tool_calls.
          if (delta_message.has_value() && delta_message->tool_calls.has_value() &&
              !delta_message->tool_calls->empty()) {
            tools_streamed = true;
          }

          // :624-633 — a null delta (parser withholding) skips the chunk unless
          // this is the terminal step, where an empty delta carries the finish.
          if (!delta_message.has_value()) {
            if (!finished) continue;
            delta_message = DeltaMessage{};
          }

          ChatCompletionResponseStreamChoice choice;
          choice.index = 0;
          choice.delta = std::move(*delta_message);
          if (finished) {
            // :688-693 — "tool_calls" for auto/required, else the model's own.
            if (tools_streamed && !named_tool_choice) {
              choice.finish_reason = "tool_calls";
            } else {
              choice.finish_reason = output.finish_reason.value_or("stop");
            }
          } else {
            choice.finish_reason = std::nullopt;
          }

          ChatCompletionStreamResponse chunk;
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
      ChatCompletionStreamResponse usage_chunk;
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

  // ── Non-streaming (chat_completion_full_generator, :804) ────────────────
  const RequestOutput final_res = async_engine_ != nullptr
      ? async_engine_->generate(prompt, std::move(sampling_params),
                                engine_request_id, request.priority)
      : sync_engine_->generate(prompt, std::move(sampling_params),
                               engine_request_id, request.priority);

  ChatCompletionResponse response;
  response.id = request_id;
  response.created = created_time;
  response.model = model_name;

  int num_generated_tokens = 0;
  for (const CompletionOutput& output : final_res.outputs) {
    ChatCompletionResponseChoice choice;
    choice.index = output.index;
    // Tool shaping (:857-966): with a parser + active tools, attach tool_calls
    // and set finish_reason="tool_calls"; else a plain-content assistant message
    // (finish_reason = output.finish_reason or "stop", :956-960).
    ShapedChatMessage shaped = ShapeChatMessage(
        kAssistantRole, output.text, output.finish_reason, request, parser.get(),
        reasoning_parser.get());
    choice.message = std::move(shaped.message);
    choice.finish_reason = std::move(shaped.finish_reason);
    response.choices.push_back(std::move(choice));
    num_generated_tokens += static_cast<int>(output.token_ids.size());
  }

  const int num_prompt_tokens =
      static_cast<int>(final_res.prompt_token_ids.size());
  response.usage.prompt_tokens = num_prompt_tokens;
  response.usage.completion_tokens = num_generated_tokens;
  response.usage.total_tokens = num_prompt_tokens + num_generated_tokens;

  ChatCompletionResult result;
  result.streaming = false;
  result.response = std::move(response);
  return result;
}

}  // namespace vllm::entrypoints::openai
