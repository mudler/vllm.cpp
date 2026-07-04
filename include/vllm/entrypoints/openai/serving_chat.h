// Ported from: vllm/entrypoints/openai/chat_completion/serving.py @ e24d1b24
// (OpenAIServingChat.create_chat_completion + chat_completion_stream_generator +
// chat_completion_full_generator).
//
// SCOPE (M3.1 Task 2 / T0): the OpenAI /v1/chat/completions serving logic
// DECOUPLED from HTTP. Same shape as OpenAIServingCompletion, with the chat
// specifics:
//   - the prompt is built from `messages` via a SEAM (see ChatPromptFn). At T0
//     the real chat-template renderer (Task 3 / M3.2) is not wired, so the
//     default is a SIMPLE fallback join ("<role>: <content>\n" per message +
//     an "assistant:" generation prompt). Task 3 swaps a template renderer in
//     by constructing the handler with a custom ChatPromptFn — this is the
//     Task-3 integration point.
//   - streaming cadence: FIRST chunk = role delta ({role:"assistant",
//     content:""}), THEN content deltas ({content: piece}), THEN the finish
//     chunk (last content delta + finish_reason), THEN `data: [DONE]\n\n`
//     (chat_completion/serving.py:485-520, :663-738, :802).
//
// DEFERRED (marked; matches upstream): tools / tool_choice / grammars
// (M3.3/M3.4); reasoning parser; logprobs payload; echo; n > 1; stream_options
// / include_usage trailing usage chunk; beam search; LoRA; multimodal.
#ifndef VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_
#define VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/v1/engine/llm_engine.h"

namespace vllm::entrypoints::openai {

// The result of create_chat_completion. Mirrors upstream's
//   AsyncGenerator[str] | ChatCompletionResponse
// return union (chat_completion/serving.py:233).
struct ChatCompletionResult {
  bool streaming = false;
  std::optional<ChatCompletionResponse> response;   // set when !streaming
  std::vector<std::string> sse_chunks;              // set when streaming
};

// The chat-prompt SEAM: messages + add_generation_prompt → the prompt string
// the completion path tokenizes. Task 3 (M3.2) injects the real chat-template
// renderer here; the default is DefaultChatPromptFallback below.
using ChatPromptFn =
    std::function<std::string(const std::vector<ChatMessage>&, bool)>;

// The T0 fallback template (marked seam). Concatenates "<role>: <content>\n"
// for each message; when add_generation_prompt, appends "assistant:". Exposed
// so it is directly unit-testable and so Task 3 can wrap/replace it.
std::string DefaultChatPromptFallback(const std::vector<ChatMessage>& messages,
                                      bool add_generation_prompt);

class OpenAIServingChat {
 public:
  // `prompt_fn` defaults to DefaultChatPromptFallback (the Task-3 seam).
  OpenAIServingChat(v1::LLMEngine& engine, std::string served_model_name,
                    ChatPromptFn prompt_fn = DefaultChatPromptFallback);

  // create_chat_completion (chat_completion/serving.py:229).
  ChatCompletionResult create_chat_completion(
      const ChatCompletionRequest& request);

 private:
  v1::LLMEngine& engine_;
  std::string served_model_name_;
  ChatPromptFn prompt_fn_;
  // request_id is "chatcmpl-<counter>" (upstream f"chatcmpl-{random_uuid()}").
  int64_t request_counter_ = 0;
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_
