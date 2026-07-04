// Ported from: vllm/entrypoints/openai/chat_completion/serving.py @ e24d1b24
// See serving_chat.h for scope, the chat-prompt seam and deferrals.
#include "vllm/entrypoints/openai/serving_chat.h"

#include <ctime>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/serving_utils.h"

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

ShapedChatMessage ShapeChatMessage(
    const std::string& role, const std::string& model_output,
    std::optional<std::string> output_finish_reason,
    const ChatCompletionRequest& request, ToolParser* parser) {
  ShapedChatMessage shaped;
  shaped.message.role = role;

  // chat_completion/serving.py:899-923 (the "auto" path). When tools are active
  // and a parser exists, extract; on tools_called, the finish_reason becomes
  // "tool_calls" (:936).
  if (parser != nullptr && ToolsEnabled(request)) {
    const ExtractedToolCallInformation info =
        parser->extract_tool_calls(model_output, request);
    if (info.tools_called && !info.tool_calls.empty()) {
      if (info.content.has_value()) {
        shaped.message.content = SanitizeUtf8(*info.content);
      } else {
        shaped.message.content = std::nullopt;
      }
      shaped.message.tool_calls = info.tool_calls;
      shaped.finish_reason = "tool_calls";
      return shaped;
    }
    // tools_called=false → fall through to a plain-content message with the
    // parser's content (the whole output).
    shaped.message.content =
        SanitizeUtf8(info.content.value_or(model_output));
    shaped.finish_reason = std::move(output_finish_reason);
    if (!shaped.finish_reason.has_value()) shaped.finish_reason = "stop";
    return shaped;
  }

  // No tools: plain content message (:881).
  shaped.message.content = SanitizeUtf8(model_output);
  shaped.finish_reason = std::move(output_finish_reason);
  if (!shaped.finish_reason.has_value()) shaped.finish_reason = "stop";
  return shaped;
}

std::optional<DeltaMessage> ShapeChatDelta(const std::string& previous_text,
                                           const std::string& current_text,
                                           const std::string& delta_text,
                                           const ChatCompletionRequest& request,
                                           ToolParser* parser) {
  // chat_completion/serving.py:589-613 — with a parser, run the streaming parse;
  // otherwise emit a plain content delta.
  if (parser != nullptr && ToolsEnabled(request)) {
    return parser->extract_tool_calls_streaming(previous_text, current_text,
                                                delta_text, request);
  }
  DeltaMessage msg;
  msg.content = delta_text;
  return msg;
}

std::unique_ptr<ToolParser> OpenAIServingChat::MakeToolParser(
    const ChatCompletionRequest& request) const {
  if (tool_parser_name_.empty() || !ToolsEnabled(request)) return nullptr;
  return get_tool_parser(tool_parser_name_);
}

OpenAIServingChat::OpenAIServingChat(v1::LLMEngine& engine,
                                     std::string served_model_name,
                                     ChatPromptFn prompt_fn,
                                     std::string tool_parser_name)
    : engine_(engine),
      served_model_name_(std::move(served_model_name)),
      prompt_fn_(std::move(prompt_fn)),
      tool_parser_name_(std::move(tool_parser_name)) {}

ChatCompletionResult OpenAIServingChat::create_chat_completion(
    const ChatCompletionRequest& request) {
  // request_id = f"chatcmpl-{...}" (chat_completion/serving.py:268); created =
  // int(time.time()) (:416 / :816).
  const std::string request_id =
      "chatcmpl-" + std::to_string(request_counter_++);
  const auto created_time = static_cast<int64_t>(std::time(nullptr));
  const std::string model_name =
      request.model.has_value() ? *request.model : served_model_name_;

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
  const bool named_tool_choice = IsNamedToolChoice(request);

  SamplingParams sampling_params = request.to_sampling_params();

  // Single prompt → sub_request_id == request_id (chat_completion/serving.py:293).
  const std::string engine_request_id = request_id;

  if (request.stream) {
    // ── Streaming (chat_completion_stream_generator, :404) ────────────────
    ChatCompletionResult result;
    result.streaming = true;

    // First chunk: the role delta ({role:"assistant", content:""}) — :485-520.
    {
      ChatCompletionResponseStreamChoice choice;
      choice.index = 0;
      choice.delta.role = kAssistantRole;
      choice.delta.content = "";
      choice.finish_reason = std::nullopt;

      ChatCompletionStreamResponse chunk;
      chunk.id = request_id;
      chunk.created = created_time;
      chunk.model = model_name;
      chunk.choices.push_back(std::move(choice));
      result.sse_chunks.push_back(
          "data: " + nlohmann::json(chunk).dump() + "\n\n");
    }

    // Content / tool-call deltas, then the finish chunk (:559-716). The tool
    // parser (when present) drives per-delta shaping; `previous_text` is the
    // accumulated output the streaming parser re-parses each step (:615).
    int previous_num_tokens = 0;
    std::string previous_text;
    bool tools_streamed = false;
    engine_.add_request(engine_request_id, prompt, std::move(sampling_params));
    while (engine_.has_unfinished_requests()) {
      for (const RequestOutput& res : engine_.step()) {
        if (res.request_id != engine_request_id) continue;
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
              previous_text, current_text, delta_text, request, parser.get());
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
          result.sse_chunks.push_back(
              "data: " + nlohmann::json(chunk).dump() + "\n\n");
        }
      }
    }
    result.sse_chunks.push_back("data: [DONE]\n\n");
    return result;
  }

  // ── Non-streaming (chat_completion_full_generator, :804) ────────────────
  const RequestOutput final_res =
      engine_.generate(prompt, std::move(sampling_params), engine_request_id);

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
        kAssistantRole, output.text, output.finish_reason, request, parser.get());
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
