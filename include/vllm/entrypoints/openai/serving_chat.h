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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_completion.h"  // SseStream
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/engine/llm_engine.h"

namespace vllm::entrypoints::openai {

// The result of create_chat_completion. Mirrors upstream's
//   AsyncGenerator[str] | ChatCompletionResponse
// return union (chat_completion/serving.py:233).
struct ChatCompletionResult {
  bool streaming = false;
  std::optional<ChatCompletionResponse> response;   // set when !streaming
  std::vector<std::string> sse_chunks;              // set when streaming
  std::shared_ptr<SseStream> sse_stream;             // live AsyncLLM path
};

// The chat-prompt SEAM: messages + add_generation_prompt + tools → the prompt
// string the completion path tokenizes. The M3.2 chat-template renderer injects
// here (MakeChatTemplatePromptFn); the default is DefaultChatPromptFallback. The
// `tools` arg is what upstream passes to apply_chat_template(..., tools=...) so
// the template's `{% if tools %}` branch renders the function schemas
// (chat_completion/serving.py → chat_utils.apply_hf_chat_template).
using ChatPromptFn = std::function<std::string(
    const std::vector<ChatMessage>&, bool,
    const std::vector<ChatCompletionToolsParam>&)>;

// The T0 fallback template (marked seam). Concatenates "<role>: <content>\n"
// for each message; when add_generation_prompt, appends "assistant:". Ignores
// `tools` (the fallback is not a model template). Exposed for unit testing.
std::string DefaultChatPromptFallback(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt,
    const std::vector<ChatCompletionToolsParam>& tools = {});

// Whether tool extraction is active for `request`: tools present (non-empty) and
// tool_choice is not the explicit "none" (chat_completion/serving.py:896-905 —
// the "auto/required/named" branches). tool_choice defaults to auto when tools
// are present (protocol.py:828-831).
bool ToolsEnabled(const ChatCompletionRequest& request);

// Non-stream tool shaping (chat_completion/serving.py:857-966). Given the model
// output text + its finish_reason, produce the response choice's message + the
// finish_reason. When `parser` is non-null AND ToolsEnabled(request), run
// extract_tool_calls: on tools_called, attach tool_calls (+ the leading content)
// and set finish_reason="tool_calls"; otherwise a plain-content message.
struct ShapedChatMessage {
  ChatMessage message;
  std::optional<std::string> finish_reason;
};
ShapedChatMessage ShapeChatMessage(const std::string& role,
                                   const std::string& model_output,
                                   std::optional<std::string> output_finish_reason,
                                   const ChatCompletionRequest& request,
                                   ToolParser* parser);

// Per-delta stream tool shaping (chat_completion/serving.py:589-613). When
// `parser` is non-null AND ToolsEnabled(request), drive the STATEFUL streaming
// parser (extract_tool_calls_streaming) → a DeltaMessage or nullopt (withhold).
// Otherwise a plain-content DeltaMessage carrying `delta_text`.
std::optional<DeltaMessage> ShapeChatDelta(const std::string& previous_text,
                                           const std::string& current_text,
                                           const std::string& delta_text,
                                           const ChatCompletionRequest& request,
                                           ToolParser* parser);

// Ported from: vllm/tool_parsers/structural_tag_registry.py @ e24d1b24
// (get_hermes_structural_tag:237-269 + _hermes_tool_tags:213-234). Build the
// Hermes STRUCTURAL-TAG spec from a request's tools + tool_choice — the DECODE
// constraint that steers a tool call. Returns nullopt for "none" / no tools.
//
// The spec is our native structural-tag JSON (backend_native.cpp kStructuralTag;
// the SEAM is 1:1 with vLLM's xgrammar StructuralTag, the content backend-private):
//   {"lazy": bool, "triggers": [str], "stop_after_first": bool,
//    "tags": [{"begin": str, "content_schema": <schema|true>, "end": str}]}
// Each tool contributes TWO tags (vLLM's two Hermes surface variants,
// structural_tag_registry.py:219-221): the tool name is baked into `begin`
// (`<tool_call>\n{"name": "<fn>", "arguments": ` and the compact
// `<tool_call>{"name": "<fn>", "arguments": `), `content_schema` is the tool's
// `parameters` (or `true` = any JSON when absent, _get_function_parameters:207),
// and `end` closes the wrapper (`}\n</tool_call>` / `}</tool_call>`).
//
// tool_choice -> spec (get_hermes_structural_tag:248-267):
//   auto (or unset default) -> LAZY: {lazy:true, triggers:["<tool_call>"],
//       tags:[all tools]} — plain text is FREE until the `<tool_call>` trigger,
//       then the tool-call JSON is constrained (TriggeredTagsFormat, :249-254).
//       NOT forced: the model may just reply.
//   required -> FORCED >=1: {lazy:false, stop_after_first:false, tags:[all
//       tools]} (TagsWithSeparatorFormat at_least_one, :262-267).
//   named ("function") -> FORCED exactly one: {lazy:false,
//       stop_after_first:true, tags:[that one tool]} (+stop_after_first, :255-261).
std::optional<nlohmann::json> ToolChoiceStructuralTagSpec(
    const ChatCompletionRequest& request);

// Apply ToolChoiceStructuralTagSpec onto `sampling_params`: sets
// structured_outputs.structural_tag = dump(spec) (the ONE structured-output
// constraint; json/grammar stay unset) and re-runs Verify(). No-op for
// auto/none with no tools. For auto this is a LAZY tag (the model may reply in
// plain text OR emit a `<tool_call>` — NOT forced); required/named force a call.
// This SUBSUMES the old WrapSchemaAsToolCallGbnf forced-json path — the native
// structural-tag compile handles the `<tool_call>` wrapper for ALL cases. Called
// in create_chat_completion before add_request.
void ApplyToolChoiceStructuredOutput(const ChatCompletionRequest& request,
                                     SamplingParams& sampling_params);

class OpenAIServingChat {
 public:
  // `prompt_fn` defaults to DefaultChatPromptFallback (the M3.2 seam).
  // `tool_parser_name` selects the tool-call parser (get_tool_parser) used when
  // a request carries tools; default "hermes" (the gate model's format — Qwen3.6
  // shares it, see qwen3.h). Empty disables tool parsing.
  OpenAIServingChat(v1::LLMEngine& engine, std::string served_model_name,
                    ChatPromptFn prompt_fn = DefaultChatPromptFallback,
                    std::string tool_parser_name = "hermes");
  OpenAIServingChat(v1::AsyncLLM& engine, std::string served_model_name,
                    ChatPromptFn prompt_fn = DefaultChatPromptFallback,
                    std::string tool_parser_name = "hermes");

  // create_chat_completion (chat_completion/serving.py:229).
  ChatCompletionResult create_chat_completion(
      const ChatCompletionRequest& request);

  // See OpenAIServingCompletion::uses_async_engine().
  bool uses_async_engine() const { return async_engine_ != nullptr; }

 private:
  // Build the per-request tool parser (get_tool_parser) when ToolsEnabled and a
  // parser name is configured; else nullptr. ONE instance per request (the
  // streaming parse is stateful).
  std::unique_ptr<ToolParser> MakeToolParser(
      const ChatCompletionRequest& request) const;

  v1::LLMEngine* sync_engine_ = nullptr;
  v1::AsyncLLM* async_engine_ = nullptr;
  std::string served_model_name_;
  ChatPromptFn prompt_fn_;
  std::string tool_parser_name_;
  // request_id is "chatcmpl-<counter>" (upstream f"chatcmpl-{random_uuid()}").
  std::atomic<int64_t> request_counter_{0};
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_
