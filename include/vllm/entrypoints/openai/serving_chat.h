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
// (M3.3/M3.4); reasoning parser; logprobs payload; echo; n > 1; beam search;
// LoRA; multimodal.
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
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
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
//
// REASONING (chat_completion/serving.py:858-866): when `reasoning_parser` is
// non-null it runs FIRST: extract_reasoning splits `model_output` into a
// reasoning span (attached to message.reasoning) and the remaining CONTENT, and
// only that content is then handed to the tool parser. Order: reasoning strips
// the chain-of-thought, then tool extraction runs over what the user actually
// sees, exactly as upstream `parser.parse()` returns (reasoning, content,
// tool_calls) with reasoning removed before tool detection.
struct ShapedChatMessage {
  ChatMessage message;
  std::optional<std::string> finish_reason;
};
ShapedChatMessage ShapeChatMessage(const std::string& role,
                                   const std::string& model_output,
                                   std::optional<std::string> output_finish_reason,
                                   const ChatCompletionRequest& request,
                                   ToolParser* parser,
                                   ReasoningParser* reasoning_parser = nullptr);

// Per-delta stream tool shaping (chat_completion/serving.py:589-613). When
// `parser` is non-null AND ToolsEnabled(request), drive the STATEFUL streaming
// parser (extract_tool_calls_streaming) → a DeltaMessage or nullopt (withhold).
// Otherwise a plain-content DeltaMessage carrying `delta_text`.
//
// REASONING: when `reasoning_parser` is non-null it runs FIRST on the raw delta
// (extract_reasoning_streaming). The reasoning span rides on the returned
// DeltaMessage.reasoning; the post-reasoning CONTENT span (once </think> passes)
// is what feeds the tool parser, with content-space previous/current offsets
// derived from the reasoning split so the tool parse never sees the thoughts.
std::optional<DeltaMessage> ShapeChatDelta(const std::string& previous_text,
                                           const std::string& current_text,
                                           const std::string& delta_text,
                                           const ChatCompletionRequest& request,
                                           ToolParser* parser,
                                           ReasoningParser* reasoning_parser = nullptr);

// Ported from: vllm/tool_parsers/structural_tag_registry.py @ e24d1b24
// (get_hermes_structural_tag:237-269 + _hermes_tool_tags:213-234).
//
// THIN WRAPPERS over the per-family STRUCTURAL-TAG registry
// (tool_parsers/structural_tags.h): both delegate to the "hermes" family and are
// kept only for source compatibility with existing callers/tests. New code
// should call ToolChoiceStructuralTagSpecFor(tool_parser_name, request) /
// ApplyToolChoiceStructuredOutput(tool_parser_name, request, sampling_params) so
// the DECODE constraint matches the ACTIVE model family's native tool syntax
// instead of always Hermes. See structural_tags.h for the native spec shape, the
// per-family coverage table and the tool_choice (auto/required/named) semantics.
std::optional<nlohmann::json> ToolChoiceStructuralTagSpec(
    const ChatCompletionRequest& request);

// Thin wrapper: applies the "hermes" family structural tag onto
// `sampling_params` (structured_outputs.structural_tag). create_chat_completion
// itself calls the per-family overload (structural_tags.h) with the active
// tool_parser_name. Kept for source compatibility.
void ApplyToolChoiceStructuredOutput(const ChatCompletionRequest& request,
                                     SamplingParams& sampling_params);

class OpenAIServingChat {
 public:
  // `prompt_fn` defaults to DefaultChatPromptFallback (the M3.2 seam).
  // `tool_parser_name` selects the tool-call parser (get_tool_parser) used when
  // a request carries tools; default "hermes" (the gate model's format — Qwen3.6
  // shares it, see qwen3.h). Empty disables tool parsing.
  // `reasoning_parser_name` selects the reasoning parser (get_reasoning_parser),
  // mirroring the tool_parser_name pattern; default "" DISABLES reasoning
  // extraction (the C ABI / capi wires the model-specific selection separately).
  OpenAIServingChat(v1::LLMEngine& engine, std::string served_model_name,
                    ChatPromptFn prompt_fn = DefaultChatPromptFallback,
                    std::string tool_parser_name = "hermes",
                    std::string reasoning_parser_name = "",
                    bool enable_force_include_usage = false);
  OpenAIServingChat(v1::AsyncLLM& engine, std::string served_model_name,
                    ChatPromptFn prompt_fn = DefaultChatPromptFallback,
                    std::string tool_parser_name = "hermes",
                    std::string reasoning_parser_name = "",
                    bool enable_force_include_usage = false);

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

  // Build the per-request reasoning parser (get_reasoning_parser) when a parser
  // name is configured; else nullptr. ONE instance per request (the streaming
  // parse may be stateful, olmo3). Unlike tools, reasoning does not gate on
  // ToolsEnabled: a request without tools can still carry a chain-of-thought.
  std::unique_ptr<ReasoningParser> MakeReasoningParser() const;

  v1::LLMEngine* sync_engine_ = nullptr;
  v1::AsyncLLM* async_engine_ = nullptr;
  std::string served_model_name_;
  ChatPromptFn prompt_fn_;
  std::string tool_parser_name_;
  std::string reasoning_parser_name_;
  bool enable_force_include_usage_ = false;
  // request_id is "chatcmpl-<counter>" (upstream f"chatcmpl-{random_uuid()}").
  std::atomic<int64_t> request_counter_{0};
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_
