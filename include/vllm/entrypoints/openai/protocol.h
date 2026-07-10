// Ported from: vllm/entrypoints/openai/protocol.py @ e24d1b24
//
// NOTE ON UPSTREAM LAYOUT: at e24d1b24 the historical monolithic
// `vllm/entrypoints/openai/protocol.py` has been split into per-endpoint
// modules — the shared/base shapes live in
// `vllm/entrypoints/openai/engine/protocol.py` (OpenAIBaseModel, ErrorInfo,
// ErrorResponse, UsageInfo, DeltaMessage, StreamOptions), the completion
// shapes in `.../completion/protocol.py` (CompletionRequest + its
// to_sampling_params, CompletionResponse[Choice], CompletionStreamResponse +
// StreamChoice), and the chat shapes in `.../chat_completion/protocol.py`
// (ChatCompletionRequest + its to_sampling_params, ChatCompletionResponse,
// ChatMessage, ChatCompletionResponse[Stream]Choice, ChatCompletionResponse,
// ChatCompletionStreamResponse). We collapse the T0 subset of all three into a
// single mirrored `protocol.{h,cpp}` (the pre-split upstream path) — the
// Ported-from headers above the individual sections cite the exact current
// files + lines.
//
// SCOPE (T0): text-completion + chat-completion request parsing → our
// SamplingParams, plus the response / stream-chunk / usage / error JSON shapes
// the OpenAI SDK + LocalAI depend on. Field NAMES and the `object` string
// literals are load-bearing for client compat and are mirrored 1:1.
//
// DEFERRED (parsed-and-ignored via OpenAI's extra="allow" — nlohmann simply
// does not read unknown keys — or explicitly marked below):
//   - logit_bias, allowed_token_ids, best_of, suffix, user, prompt_embeds
//   - response_format / structured_outputs / tools / tool_choice / functions
//     (M3.3 tool calling, M3.4 grammars)
//   - logprobs *payload* shapes (CompletionLogProbs / ChatCompletionLogProbs);
//     the request-side `logprobs` counts ARE mapped into SamplingParams.
//   - multimodal message content parts (T0 chat content is a bare string)
//   - array / token-id `prompt` forms (T0 completion prompt is a bare string)
//   - beam search, kv_transfer_params, vllm_xargs, cache_salt, etc.
//     (`priority` IS parsed now — it feeds the priority scheduler; see below.)
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/sampling_params.h"

namespace vllm::entrypoints::openai {

// ---------------------------------------------------------------------------
// Shared shapes — engine/protocol.py
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/engine/protocol.py:111 (UsageInfo).
// `completion_tokens` is Optional[int]=0 upstream; T0 always populates it.
struct UsageInfo {
  int prompt_tokens = 0;
  int total_tokens = 0;
  int completion_tokens = 0;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:60 (ErrorInfo).
struct ErrorInfo {
  std::string message;
  std::string type;
  std::optional<std::string> param;
  // Upstream `code: int` (an HTTP status code). Optional here so callers may
  // omit it; serialized as null when unset.
  std::optional<int> code;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:67 (ErrorResponse).
struct ErrorResponse {
  ErrorInfo error;
};

// ---------------------------------------------------------------------------
// response_format — engine/protocol.py (shared by both request types)
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/engine/protocol.py:123
// (JsonSchemaResponseFormat). The wire `schema` field is aliased to
// `json_schema` (pydantic name-conflict workaround upstream); we keep the schema
// as a raw nlohmann::json (the constraint layer serializes it). `strict` /
// `description` are parsed but not load-bearing at T0.
struct JsonSchemaResponseFormat {
  std::string name;
  std::optional<std::string> description;
  // The JSON Schema object (wire key "schema").
  std::optional<nlohmann::json> json_schema;
  std::optional<bool> strict;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:156 (ResponseFormat).
// `type` is one of "text" | "json_object" | "json_schema". structural_tag is
// deferred (M3.4 STRUCTURAL_TAG stub); a structural_tag response_format parses
// but does not map to a constraint at T0.
struct ResponseFormat {
  std::string type;
  std::optional<JsonSchemaResponseFormat> json_schema;
};

// ---------------------------------------------------------------------------
// Tool / function calling — engine/protocol.py + chat_completion/protocol.py
// (M3.3 Task 1)
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/engine/protocol.py:246
// (FunctionDefinition). `parameters` is the JSON-Schema object for the
// function's arguments (upstream dict[str,Any]|None). `strict` / `defer_loading`
// are parsed-and-ignored at T0 (deferred).
struct FunctionDefinition {
  std::string name;
  std::optional<std::string> description;
  std::optional<nlohmann::json> parameters;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:165
// (ChatCompletionToolsParam). `defer_loading` deferred.
struct ChatCompletionToolsParam {
  std::string type = "function";
  FunctionDefinition function;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:218
// (ChatCompletionRequest.tool_choice) union + :188
// (ChatCompletionNamedToolChoiceParam). The upstream union is
// "none"|"auto"|"required"|{type:"function",function:{name}}. We model both the
// string and named-object forms as one struct: `mode` is one of
// "none"/"auto"/"required"/"function"; `function_name` is set only for the named
// (mode=="function") form.
struct ToolChoice {
  std::string mode;  // "none" | "auto" | "required" | "function"
  std::optional<std::string> function_name;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:310 (FunctionCall).
// The internal `id` field (excluded from serialization upstream) is deferred.
struct FunctionCall {
  std::string name;
  std::string arguments;  // JSON-encoded arguments string (OpenAI spec).
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:319 (ToolCall).
struct ToolCall {
  std::string id;
  std::string type = "function";
  FunctionCall function;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:325
// (DeltaFunctionCall).
struct DeltaFunctionCall {
  std::optional<std::string> name;
  std::optional<std::string> arguments;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:331 (DeltaToolCall).
// `index` is required; id/type/function are per-chunk optional (name-first, then
// arguments deltas).
struct DeltaToolCall {
  int index = 0;
  std::optional<std::string> id;
  std::optional<std::string> type;
  DeltaFunctionCall function;
};

// ---------------------------------------------------------------------------
// Completions — completion/protocol.py
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/completion/protocol.py:45
// (CompletionRequest). T0 subset. `stop` (upstream str|list) is normalized to
// list-form on parse (see from_json). `prompt` T0 = bare string.
struct CompletionRequest {
  std::optional<std::string> model;
  // T0: bare-string prompt. Array / token-id forms deferred (see header note).
  std::string prompt;

  // max_tokens: upstream field default is 16, and its `normalize_null_max_tokens`
  // validator rewrites an explicit `null` back to that default — so this is
  // effectively always populated. Kept optional to model "field absent".
  std::optional<int> max_tokens = 16;
  int n = 1;
  // Sampling knobs are Optional upstream (None => resolve to _DEFAULT_SAMPLING_
  // PARAMS in to_sampling_params); mirrored as std::optional here.
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<int> top_k;
  std::optional<double> min_p;
  std::optional<double> repetition_penalty;
  double presence_penalty = 0.0;
  double frequency_penalty = 0.0;
  std::optional<int64_t> seed;
  std::vector<std::string> stop;           // normalized list-form
  std::vector<int32_t> stop_token_ids;
  bool stream = false;
  std::optional<int> logprobs;
  std::optional<int> prompt_logprobs;
  bool echo = false;                       // parsed; behavior deferred
  int min_tokens = 0;
  bool ignore_eos = false;
  bool include_stop_str_in_output = false;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;

  // priority (completion/protocol.py): the request's scheduling priority for
  // the priority policy (lower = handled first). Default 0. Not a sampling
  // param — the serving layer forwards it to engine add_request/process_inputs.
  int priority = 0;

  // response_format (completion/protocol.py:106): {type: "text"|"json_object"|
  // "json_schema", json_schema?} — normalized into structured_outputs in
  // to_sampling_params (M3.4 Task 5).
  std::optional<ResponseFormat> response_format;

  // to_sampling_params — completion/protocol.py:260. Maps each OpenAI field 1:1
  // onto our SamplingParams (resolving None sampling knobs to _DEFAULT_SAMPLING_
  // PARAMS), then runs PostInit(). `default_max_tokens` is the serving-resolved
  // fallback used only when the request omits max_tokens (Task 2 supplies the
  // model-derived value; unset => our SamplingParams default).
  SamplingParams to_sampling_params(
      std::optional<int> default_max_tokens = std::nullopt) const;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:519
// (CompletionResponseChoice). logprobs / stop_reason / token payloads deferred.
struct CompletionResponseChoice {
  int index = 0;
  std::string text;
  std::optional<std::string> finish_reason;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:547
// (CompletionResponse). object == "text_completion".
struct CompletionResponse {
  std::string id;
  std::string object = "text_completion";
  int64_t created = 0;
  std::string model;
  std::vector<CompletionResponseChoice> choices;
  UsageInfo usage;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:563
// (CompletionResponseStreamChoice).
struct CompletionResponseStreamChoice {
  int index = 0;
  std::string text;
  std::optional<std::string> finish_reason;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:582
// (CompletionStreamResponse). object == "text_completion"; usage optional.
struct CompletionStreamResponse {
  std::string id;
  std::string object = "text_completion";
  int64_t created = 0;
  std::string model;
  std::vector<CompletionResponseStreamChoice> choices;
  std::optional<UsageInfo> usage;
};

// ---------------------------------------------------------------------------
// Chat completions — chat_completion/protocol.py
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:57
// (ChatMessage). T0: bare-string content; refusal / audio / function_call
// (legacy) deferred. A response message carries EITHER content OR tool_calls;
// when tool_calls is empty the key is omitted (upstream _serialize pop).
struct ChatMessage {
  std::string role;
  // Default member initializers so a 2-field aggregate init
  // `ChatMessage{role, content}` (used across the serving/chat-template tests)
  // does not trip -Werror=missing-field-initializers now that tool_calls exists.
  std::optional<std::string> content{};
  std::optional<std::vector<ToolCall>> tool_calls{};
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:350 (DeltaMessage).
// `tool_calls` is a list defaulting to empty upstream (serialized only when
// non-empty); modeled here as optional to mean "absent/empty".
struct DeltaMessage {
  std::optional<std::string> role;
  std::optional<std::string> content;
  std::optional<std::vector<DeltaToolCall>> tool_calls;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:193
// (ChatCompletionRequest). T0 subset. messages content T0 = bare string.
struct ChatCompletionRequest {
  std::vector<ChatMessage> messages;
  std::optional<std::string> model;

  // Upstream: max_tokens (deprecated) + max_completion_tokens, both default
  // None; serving prefers max_completion_tokens (chat_completion/serving.py:299).
  std::optional<int> max_tokens;
  std::optional<int> max_completion_tokens;
  std::optional<int> n = 1;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<int> top_k;
  std::optional<double> min_p;
  std::optional<double> repetition_penalty;
  double presence_penalty = 0.0;
  double frequency_penalty = 0.0;
  std::optional<int64_t> seed;
  std::vector<std::string> stop;           // normalized list-form
  std::vector<int32_t> stop_token_ids;
  bool stream = false;
  // Upstream chat `logprobs` is a BOOL flag; `top_logprobs` is the count.
  bool logprobs = false;
  int top_logprobs = 0;
  bool echo = false;
  std::optional<int> prompt_logprobs;
  int min_tokens = 0;
  bool ignore_eos = false;
  bool include_stop_str_in_output = false;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;

  // priority (chat_completion/protocol.py). See CompletionRequest.priority.
  int priority = 0;

  // response_format (chat_completion/protocol.py:210). See CompletionRequest.
  std::optional<ResponseFormat> response_format;

  // tools / tool_choice (chat_completion/protocol.py:217-224). `tools` is the
  // list of available functions; `tool_choice` selects the calling mode. When
  // absent both stay nullopt (backward compat). The serving-layer default
  // (tool_choice="auto" when tools present, else "none") is applied downstream
  // (chat_completion/protocol.py:828-831), not here.
  // DEFERRED (parsed-and-ignored): parallel_tool_calls (protocol.py:240) and the
  // legacy `functions` / `function_call` fields.
  std::optional<std::vector<ChatCompletionToolsParam>> tools;
  std::optional<ToolChoice> tool_choice;

  // to_sampling_params — chat_completion/protocol.py:585. See CompletionRequest.
  SamplingParams to_sampling_params(
      std::optional<int> default_max_tokens = std::nullopt) const;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:94
// (ChatCompletionResponseChoice). finish_reason default "stop" upstream.
struct ChatCompletionResponseChoice {
  int index = 0;
  ChatMessage message;
  std::optional<std::string> finish_reason = "stop";
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:117
// (ChatCompletionResponse). object == "chat.completion".
struct ChatCompletionResponse {
  std::string id;
  std::string object = "chat.completion";
  int64_t created = 0;
  std::string model;
  std::vector<ChatCompletionResponseChoice> choices;
  UsageInfo usage;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:138
// (ChatCompletionResponseStreamChoice).
struct ChatCompletionResponseStreamChoice {
  int index = 0;
  DeltaMessage delta;
  std::optional<std::string> finish_reason;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:148
// (ChatCompletionStreamResponse). object == "chat.completion.chunk".
struct ChatCompletionStreamResponse {
  std::string id;
  std::string object = "chat.completion.chunk";
  int64_t created = 0;
  std::string model;
  std::vector<ChatCompletionResponseStreamChoice> choices;
  std::optional<UsageInfo> usage;
};

// ---------------------------------------------------------------------------
// nlohmann/json (de)serialization — ADL free functions.
//   Requests: from_json (parse).  Responses/usage/error: to_json (serialize).
// ---------------------------------------------------------------------------
void from_json(const nlohmann::json& j, CompletionRequest& r);
void from_json(const nlohmann::json& j, ChatMessage& m);
void from_json(const nlohmann::json& j, ChatCompletionToolsParam& t);
void from_json(const nlohmann::json& j, ChatCompletionRequest& r);

void to_json(nlohmann::json& j, const UsageInfo& u);
void to_json(nlohmann::json& j, const ErrorInfo& e);
void to_json(nlohmann::json& j, const ErrorResponse& e);
void to_json(nlohmann::json& j, const CompletionResponseChoice& c);
void to_json(nlohmann::json& j, const CompletionResponse& r);
void to_json(nlohmann::json& j, const CompletionResponseStreamChoice& c);
void to_json(nlohmann::json& j, const CompletionStreamResponse& r);
void to_json(nlohmann::json& j, const FunctionCall& f);
void to_json(nlohmann::json& j, const ToolCall& t);
void to_json(nlohmann::json& j, const DeltaFunctionCall& f);
void to_json(nlohmann::json& j, const DeltaToolCall& t);
void to_json(nlohmann::json& j, const ChatMessage& m);
void to_json(nlohmann::json& j, const DeltaMessage& m);
void to_json(nlohmann::json& j, const ChatCompletionResponseChoice& c);
void to_json(nlohmann::json& j, const ChatCompletionResponse& r);
void to_json(nlohmann::json& j, const ChatCompletionResponseStreamChoice& c);
void to_json(nlohmann::json& j, const ChatCompletionStreamResponse& r);

}  // namespace vllm::entrypoints::openai
