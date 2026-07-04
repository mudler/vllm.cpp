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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
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

// Ported from: vllm/entrypoints/openai/chat_completion/serving.py @ e24d1b24
// (the tool_choice -> forced structured output; upstream builds an xgrammar
// StructuralTag via tool_parsers/structural_tag_registry.py). The JSON schema an
// emitted tool call MUST match when tool_choice FORCES one — the named
// ("function") tool, or "required" (any listed tool). Returns nullopt for
// "auto"/"none" / no tools (model decides / disabled).
//
// Shape mirrors the upstream Hermes structural tag's INNER tool-call content
// (structural_tag_registry.py:213-234) `{"name": "<fn>", "arguments":
// <fn.parameters>}`:
//   named / one tool         -> {"type":"object",
//                                "properties":{"name":{"const":"<fn>"},
//                                              "arguments":<params|true>},
//                                "required":["name","arguments"],
//                                "additionalProperties":false}
//   required, multiple tools -> {"anyOf":[<per-tool object schema>, ...]}
// DEVIATION (§9): our structured_outputs.json constrains the tool-call JSON
// OBJECT only, NOT the literal `<tool_call>...</tool_call>` wrapper the upstream
// structural tag also forces (our GBNF json path cannot emit surrounding literal
// text). The tool parser (Task 2/3) extracts the wrapped call from the output.
std::optional<nlohmann::json> ToolChoiceForcedSchema(
    const ChatCompletionRequest& request);

// Apply ToolChoiceForcedSchema onto `sampling_params`: sets
// structured_outputs.json = the forced schema (dumped) so the decode is
// CONSTRAINED to a valid tool call. No-op for auto/none. Called in
// create_chat_completion before add_request.
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

  // create_chat_completion (chat_completion/serving.py:229).
  ChatCompletionResult create_chat_completion(
      const ChatCompletionRequest& request);

 private:
  // Build the per-request tool parser (get_tool_parser) when ToolsEnabled and a
  // parser name is configured; else nullptr. ONE instance per request (the
  // streaming parse is stateful).
  std::unique_ptr<ToolParser> MakeToolParser(
      const ChatCompletionRequest& request) const;

  v1::LLMEngine& engine_;
  std::string served_model_name_;
  ChatPromptFn prompt_fn_;
  std::string tool_parser_name_;
  // request_id is "chatcmpl-<counter>" (upstream f"chatcmpl-{random_uuid()}").
  int64_t request_counter_ = 0;
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_
