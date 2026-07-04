// Ported from: vllm/entrypoints/openai/chat_completion/serving.py @ e24d1b24
// See serving_chat.h for scope, the chat-prompt seam and deferrals.
#include "vllm/entrypoints/openai/serving_chat.h"

#include <ctime>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {
// get_chat_request_role (chat_completion/serving.py:399): the response role is
// self.response_role ("assistant") when add_generation_prompt (the T0 default).
constexpr const char* kAssistantRole = "assistant";
}  // namespace

std::string DefaultChatPromptFallback(const std::vector<ChatMessage>& messages,
                                      bool add_generation_prompt) {
  // T0 SEAM (Task 3 / M3.2 swaps in the real chat-template renderer). A simple
  // "<role>: <content>\n" join + an "assistant:" generation prompt. This is NOT
  // a model chat template — it exists only so the chat path is end-to-end
  // exercisable before minja-subset templating lands.
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

OpenAIServingChat::OpenAIServingChat(v1::LLMEngine& engine,
                                     std::string served_model_name,
                                     ChatPromptFn prompt_fn)
    : engine_(engine),
      served_model_name_(std::move(served_model_name)),
      prompt_fn_(std::move(prompt_fn)) {}

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
  // upstream default True). Task 3 replaces prompt_fn_ with the real template.
  const std::string prompt =
      prompt_fn_(request.messages, /*add_generation_prompt=*/true);

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

    // Content deltas, then the finish chunk (last delta + finish_reason).
    int previous_num_tokens = 0;
    engine_.add_request(engine_request_id, prompt, std::move(sampling_params));
    while (engine_.has_unfinished_requests()) {
      for (const RequestOutput& res : engine_.step()) {
        if (res.request_id != engine_request_id) continue;
        for (const CompletionOutput& output : res.outputs) {
          const std::string& delta_text = output.text;
          // :579-585 chunked-prefill: skip empty chunks.
          if (delta_text.empty() && output.token_ids.empty() &&
              previous_num_tokens == 0) {
            continue;
          }
          previous_num_tokens += static_cast<int>(output.token_ids.size());

          ChatCompletionResponseStreamChoice choice;
          choice.index = 0;
          // The content delta (:613 DeltaMessage(content=delta_text)).
          choice.delta.content = delta_text;
          // :663-705 finish_reason set only on the terminal chunk (else None).
          if (output.finish_reason.has_value()) {
            choice.finish_reason = *output.finish_reason;  // else "stop" (:692)
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
    choice.message.role = kAssistantRole;
    choice.message.content = output.text;  // reasoning / tool_calls deferred
    // finish_reason = output.finish_reason or "stop" (:956-960).
    choice.finish_reason = output.finish_reason.value_or("stop");
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
