// Ported from: vllm/entrypoints/openai/protocol.py @ e24d1b24
// (split upstream into engine/completion/chat_completion/protocol.py — see the
// header for the exact per-shape source citations).

#include "vllm/entrypoints/openai/protocol.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vllm::entrypoints::openai {

namespace {

// completion/protocol.py:231 / chat_completion/protocol.py:559
// (_DEFAULT_SAMPLING_PARAMS). Identical for both request types.
constexpr double kDefaultRepetitionPenalty = 1.0;
constexpr double kDefaultTemperature = 1.0;
constexpr double kDefaultTopP = 1.0;
constexpr int kDefaultTopK = 0;
constexpr double kDefaultMinP = 0.0;

// Read a JSON value that may be absent or explicit null into an optional<T>.
template <typename T>
void GetOpt(const nlohmann::json& j, const char* key, std::optional<T>& out) {
  auto it = j.find(key);
  if (it != j.end() && !it->is_null()) {
    out = it->get<T>();
  }
}

// Read a scalar with a fallback when the key is absent or null.
template <typename T>
void GetOr(const nlohmann::json& j, const char* key, T& out) {
  auto it = j.find(key);
  if (it != j.end() && !it->is_null()) {
    out = it->get<T>();
  }
}

// Normalize the OpenAI `stop` union (str | list[str] | null) to list-form —
// mirrors SamplingParams.__post_init__ stop normalization upstream.
std::vector<std::string> ParseStop(const nlohmann::json& j) {
  auto it = j.find("stop");
  if (it == j.end() || it->is_null()) return {};
  if (it->is_string()) return {it->get<std::string>()};
  if (it->is_array()) return it->get<std::vector<std::string>>();
  return {};
}

// Serialize an optional string as either its value or JSON null.
nlohmann::json OrNull(const std::optional<std::string>& v) {
  if (v.has_value()) return *v;
  return nullptr;
}

// Parse the OpenAI `response_format` object (engine/protocol.py:156 ResponseFormat
// + :123 JsonSchemaResponseFormat). Mirrors validate_response_format
// (completion/protocol.py:387 / chat_completion/protocol.py:700): a present
// response_format needs a string `type`, and `json_schema` type requires the
// json_schema field (its `schema` alias). Throws on a malformed shape (surfaces
// as a 400 bad request).
void ParseResponseFormat(const nlohmann::json& j,
                         std::optional<ResponseFormat>& out) {
  auto it = j.find("response_format");
  if (it == j.end() || it->is_null()) return;
  const nlohmann::json& rf = *it;
  if (!rf.is_object()) {
    throw std::runtime_error("response_format must be an object");
  }
  ResponseFormat r;
  if (auto t = rf.find("type"); t != rf.end() && t->is_string()) {
    r.type = t->get<std::string>();
  } else {
    throw std::runtime_error("response_format requires a string 'type'");
  }
  if (auto js = rf.find("json_schema"); js != rf.end() && js->is_object()) {
    JsonSchemaResponseFormat jsr;
    GetOr(*js, "name", jsr.name);
    GetOpt(*js, "description", jsr.description);
    // The schema is carried on the wire as `schema` (aliased to json_schema).
    if (auto sc = js->find("schema"); sc != js->end() && !sc->is_null()) {
      jsr.json_schema = *sc;
    }
    GetOpt(*js, "strict", jsr.strict);
    r.json_schema = std::move(jsr);
  }
  // validate_response_format: json_schema type must carry the json_schema field.
  if (r.type == "json_schema" && !r.json_schema.has_value()) {
    throw std::runtime_error(
        "When response_format type is 'json_schema', the 'json_schema' field "
        "must be provided.");
  }
  out = std::move(r);
}

// engine/protocol.py:241-243 (StreamOptions) plus completion/protocol.py:
// 471-478 / chat_completion/protocol.py:731-737 (validate_stream_options).
// The upstream before-validator rejects a non-empty options object unless the
// request is streaming. Null booleans use the StreamOptions defaults.
void ParseStreamOptions(const nlohmann::json& j, bool stream,
                        std::optional<StreamOptions>& out) {
  auto it = j.find("stream_options");
  if (it == j.end() || it->is_null()) return;
  if (!it->is_object()) {
    throw std::runtime_error("stream_options must be an object");
  }
  if (!stream && !it->empty()) {
    throw std::runtime_error(
        "Stream options can only be defined when `stream=True`.");
  }
  StreamOptions options;
  GetOr(*it, "include_usage", options.include_usage);
  GetOr(*it, "continuous_usage_stats", options.continuous_usage_stats);
  out = options;
}

// Normalize a parsed response_format into SamplingParams.structured_outputs
// (completion/protocol.py:309-338 / chat_completion/protocol.py:629-658):
//   json_object -> structured_outputs.json_object = true
//   json_schema -> structured_outputs.json = the schema (serialized)
//   text / absent -> no structured-output constraint.
// Merges onto any existing structured_outputs (upstream `replace(...)`), though
// at T0 nothing else populates it.
void ApplyResponseFormat(const std::optional<ResponseFormat>& rf,
                         SamplingParams& sp) {
  if (!rf.has_value()) return;
  StructuredOutputsParams so =
      sp.structured_outputs.value_or(StructuredOutputsParams{});
  bool enabled = false;
  if (rf->type == "json_object") {
    so.json_object = true;
    enabled = true;
  } else if (rf->type == "json_schema") {
    // Parse-time validation guarantees json_schema is present; a json_schema
    // WITHOUT a `schema` payload means "any JSON object" (upstream passes
    // json=None through, which the engine treats as json_object); mirror that by
    // requiring the schema here and falling back only when absent.
    if (rf->json_schema.has_value() &&
        rf->json_schema->json_schema.has_value()) {
      so.json = rf->json_schema->json_schema->dump();
    } else {
      so.json_object = true;
    }
    enabled = true;
  }
  // "text" (or any other type, e.g. deferred structural_tag) -> no constraint.
  if (enabled) sp.structured_outputs = std::move(so);
}

// Parse the OpenAI `tool_choice` union (chat_completion/protocol.py:218-224):
// a string "none"|"auto"|"required", or a named-function object
// {type:"function", function:{name}}. Mirrors the object-form validation
// (chat_completion/protocol.py:864-884): an object tool_choice requires a
// `function` object carrying a string `name`. Throws on a malformed shape
// (surfaces as a 400). An absent key leaves tool_choice unset (the serving layer
// applies the "auto when tools present else none" default upstream).
void ParseToolChoice(const nlohmann::json& j, std::optional<ToolChoice>& out) {
  auto it = j.find("tool_choice");
  if (it == j.end() || it->is_null()) return;
  const nlohmann::json& tc = *it;
  ToolChoice choice;
  if (tc.is_string()) {
    choice.mode = tc.get<std::string>();
  } else if (tc.is_object()) {
    // Named-function choice: {type:"function", function:{name}}.
    auto fn = tc.find("function");
    if (fn == tc.end() || !fn->is_object()) {
      throw std::runtime_error(
          "Expected field `function` in `tool_choice`! Correct usage: "
          "`{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`");
    }
    auto name = fn->find("name");
    if (name == fn->end() || !name->is_string()) {
      throw std::runtime_error(
          "Expected field `name` in `function` in `tool_choice`! Correct usage: "
          "`{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`");
    }
    choice.mode = "function";
    choice.function_name = name->get<std::string>();
  } else {
    throw std::runtime_error("Invalid value for `tool_choice`.");
  }
  out = std::move(choice);
}

}  // namespace

// ---------------------------------------------------------------------------
// Request parsing (from_json)
// ---------------------------------------------------------------------------

void from_json(const nlohmann::json& j, CompletionRequest& r) {
  GetOpt(j, "model", r.model);
  // T0: bare-string prompt (array / token-id forms deferred).
  if (auto it = j.find("prompt"); it != j.end() && it->is_string()) {
    r.prompt = it->get<std::string>();
  }
  // normalize_null_max_tokens (completion/protocol.py:377): an explicit null
  // max_tokens is rewritten to the field default (16); an absent key keeps the
  // default; a value overrides it.
  if (auto it = j.find("max_tokens"); it != j.end()) {
    if (it->is_null()) {
      r.max_tokens = 16;
    } else {
      r.max_tokens = it->get<int>();
    }
  }
  GetOr(j, "n", r.n);
  GetOpt(j, "temperature", r.temperature);
  GetOpt(j, "top_p", r.top_p);
  GetOpt(j, "top_k", r.top_k);
  GetOpt(j, "min_p", r.min_p);
  GetOpt(j, "repetition_penalty", r.repetition_penalty);
  GetOr(j, "presence_penalty", r.presence_penalty);
  GetOr(j, "frequency_penalty", r.frequency_penalty);
  GetOpt(j, "seed", r.seed);
  r.stop = ParseStop(j);
  GetOr(j, "stop_token_ids", r.stop_token_ids);
  GetOr(j, "stream", r.stream);
  ParseStreamOptions(j, r.stream, r.stream_options);
  GetOpt(j, "logprobs", r.logprobs);
  GetOpt(j, "prompt_logprobs", r.prompt_logprobs);
  GetOr(j, "echo", r.echo);
  GetOr(j, "min_tokens", r.min_tokens);
  GetOr(j, "ignore_eos", r.ignore_eos);
  GetOr(j, "include_stop_str_in_output", r.include_stop_str_in_output);
  GetOr(j, "skip_special_tokens", r.skip_special_tokens);
  GetOr(j, "spaces_between_special_tokens", r.spaces_between_special_tokens);
  GetOr(j, "priority", r.priority);
  ParseResponseFormat(j, r.response_format);
}

void from_json(const nlohmann::json& j, ChatMessage& m) {
  GetOr(j, "role", m.role);
  // T0: bare-string content (multimodal content-part arrays deferred).
  if (auto it = j.find("content"); it != j.end() && it->is_string()) {
    m.content = it->get<std::string>();
  }
}

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:165
// (ChatCompletionToolsParam) + engine/protocol.py:246 (FunctionDefinition).
void from_json(const nlohmann::json& j, ChatCompletionToolsParam& t) {
  GetOr(j, "type", t.type);  // defaults to "function".
  if (auto it = j.find("function"); it != j.end() && it->is_object()) {
    const nlohmann::json& fn = *it;
    GetOr(fn, "name", t.function.name);
    GetOpt(fn, "description", t.function.description);
    // `parameters` is the JSON-Schema object (kept as raw json).
    if (auto p = fn.find("parameters"); p != fn.end() && !p->is_null()) {
      t.function.parameters = *p;
    }
  }
}

void from_json(const nlohmann::json& j, ChatCompletionRequest& r) {
  if (auto it = j.find("messages"); it != j.end() && it->is_array()) {
    r.messages = it->get<std::vector<ChatMessage>>();
  }
  GetOpt(j, "model", r.model);
  GetOpt(j, "max_tokens", r.max_tokens);
  GetOpt(j, "max_completion_tokens", r.max_completion_tokens);
  GetOpt(j, "n", r.n);
  GetOpt(j, "temperature", r.temperature);
  GetOpt(j, "top_p", r.top_p);
  GetOpt(j, "top_k", r.top_k);
  GetOpt(j, "min_p", r.min_p);
  GetOpt(j, "repetition_penalty", r.repetition_penalty);
  GetOr(j, "presence_penalty", r.presence_penalty);
  GetOr(j, "frequency_penalty", r.frequency_penalty);
  GetOpt(j, "seed", r.seed);
  r.stop = ParseStop(j);
  GetOr(j, "stop_token_ids", r.stop_token_ids);
  GetOr(j, "stream", r.stream);
  ParseStreamOptions(j, r.stream, r.stream_options);
  GetOr(j, "logprobs", r.logprobs);
  GetOr(j, "top_logprobs", r.top_logprobs);
  GetOr(j, "echo", r.echo);
  GetOpt(j, "prompt_logprobs", r.prompt_logprobs);
  GetOr(j, "min_tokens", r.min_tokens);
  GetOr(j, "ignore_eos", r.ignore_eos);
  GetOr(j, "include_stop_str_in_output", r.include_stop_str_in_output);
  GetOr(j, "skip_special_tokens", r.skip_special_tokens);
  GetOr(j, "spaces_between_special_tokens", r.spaces_between_special_tokens);
  GetOr(j, "priority", r.priority);
  ParseResponseFormat(j, r.response_format);
  // tools / tool_choice (chat_completion/protocol.py:217-224).
  if (auto it = j.find("tools"); it != j.end() && it->is_array()) {
    r.tools = it->get<std::vector<ChatCompletionToolsParam>>();
  }
  ParseToolChoice(j, r.tool_choice);
  // DEFERRED: parallel_tool_calls, legacy functions / function_call — parsed and
  // ignored (nlohmann skips unknown keys; no field mapped here at T0).
}

// ---------------------------------------------------------------------------
// to_sampling_params
// ---------------------------------------------------------------------------

SamplingParams CompletionRequest::to_sampling_params(
    std::optional<int> default_max_tokens) const {
  // completion/protocol.py:260. None sampling knobs resolve to
  // _DEFAULT_SAMPLING_PARAMS (no server-provided default_sampling_params in T0).
  SamplingParams sp;
  sp.n = n;
  sp.presence_penalty = presence_penalty;
  sp.frequency_penalty = frequency_penalty;
  sp.repetition_penalty = repetition_penalty.value_or(kDefaultRepetitionPenalty);
  sp.temperature = temperature.value_or(kDefaultTemperature);
  sp.top_p = top_p.value_or(kDefaultTopP);
  sp.top_k = top_k.value_or(kDefaultTopK);
  sp.min_p = min_p.value_or(kDefaultMinP);
  sp.seed = seed;
  sp.stop = stop;
  sp.stop_token_ids = stop_token_ids;
  sp.logprobs = logprobs;
  // prompt_logprobs: fall back to logprobs when echo is set (protocol.py:303).
  sp.prompt_logprobs =
      prompt_logprobs.has_value() ? prompt_logprobs : (echo ? logprobs : std::nullopt);
  sp.ignore_eos = ignore_eos;
  sp.max_tokens = max_tokens.has_value() ? max_tokens : default_max_tokens;
  sp.min_tokens = min_tokens;
  sp.skip_special_tokens = skip_special_tokens;
  sp.spaces_between_special_tokens = spaces_between_special_tokens;
  sp.include_stop_str_in_output = include_stop_str_in_output;
  sp.output_kind =
      stream ? RequestOutputKind::kDelta : RequestOutputKind::kFinalOnly;
  // response_format -> structured_outputs (completion/protocol.py:309-338).
  ApplyResponseFormat(response_format, sp);
  sp.PostInit();
  return sp;
}

SamplingParams ChatCompletionRequest::to_sampling_params(
    std::optional<int> default_max_tokens) const {
  // chat_completion/protocol.py:585.
  SamplingParams sp;
  sp.n = n.value_or(1);
  sp.presence_penalty = presence_penalty;
  sp.frequency_penalty = frequency_penalty;
  sp.repetition_penalty = repetition_penalty.value_or(kDefaultRepetitionPenalty);
  sp.temperature = temperature.value_or(kDefaultTemperature);
  sp.top_p = top_p.value_or(kDefaultTopP);
  sp.top_k = top_k.value_or(kDefaultTopK);
  sp.min_p = min_p.value_or(kDefaultMinP);
  sp.seed = seed;
  sp.stop = stop;
  sp.stop_token_ids = stop_token_ids;
  // logprobs = top_logprobs if logprobs(bool) else None (protocol.py:677).
  sp.logprobs = logprobs ? std::optional<int>(top_logprobs) : std::nullopt;
  sp.prompt_logprobs =
      prompt_logprobs.has_value() ? prompt_logprobs : (echo ? std::optional<int>(top_logprobs) : std::nullopt);
  sp.ignore_eos = ignore_eos;
  // serving prefers max_completion_tokens over max_tokens
  // (chat_completion/serving.py:299).
  std::optional<int> req_max =
      max_completion_tokens.has_value() ? max_completion_tokens : max_tokens;
  sp.max_tokens = req_max.has_value() ? req_max : default_max_tokens;
  sp.min_tokens = min_tokens;
  sp.skip_special_tokens = skip_special_tokens;
  sp.spaces_between_special_tokens = spaces_between_special_tokens;
  sp.include_stop_str_in_output = include_stop_str_in_output;
  sp.output_kind =
      stream ? RequestOutputKind::kDelta : RequestOutputKind::kFinalOnly;
  // response_format -> structured_outputs (chat_completion/protocol.py:629-658).
  ApplyResponseFormat(response_format, sp);
  sp.PostInit();
  return sp;
}

// ---------------------------------------------------------------------------
// Response serialization (to_json)
// ---------------------------------------------------------------------------

void to_json(nlohmann::json& j, const UsageInfo& u) {
  j = nlohmann::json{
      {"prompt_tokens", u.prompt_tokens},
      {"total_tokens", u.total_tokens},
      {"completion_tokens", u.completion_tokens},
  };
}

void to_json(nlohmann::json& j, const ErrorInfo& e) {
  j = nlohmann::json{
      {"message", e.message},
      {"type", e.type},
      {"param", OrNull(e.param)},
      {"code", e.code.has_value() ? nlohmann::json(*e.code) : nlohmann::json(nullptr)},
  };
}

void to_json(nlohmann::json& j, const ErrorResponse& e) {
  j = nlohmann::json{{"error", e.error}};
}

void to_json(nlohmann::json& j, const CompletionResponseChoice& c) {
  j = nlohmann::json{
      {"index", c.index},
      {"text", c.text},
      {"finish_reason", OrNull(c.finish_reason)},
  };
}

void to_json(nlohmann::json& j, const CompletionResponse& r) {
  j = nlohmann::json{
      {"id", r.id},
      {"object", r.object},
      {"created", r.created},
      {"model", r.model},
      {"choices", r.choices},
      {"usage", r.usage},
  };
}

void to_json(nlohmann::json& j, const CompletionResponseStreamChoice& c) {
  j = nlohmann::json{
      {"index", c.index},
      {"text", c.text},
      {"finish_reason", OrNull(c.finish_reason)},
  };
}

void to_json(nlohmann::json& j, const CompletionStreamResponse& r) {
  j = nlohmann::json{
      {"id", r.id},
      {"object", r.object},
      {"created", r.created},
      {"model", r.model},
      {"choices", r.choices},
  };
  if (r.usage.has_value()) j["usage"] = *r.usage;
}

// engine/protocol.py:310 (FunctionCall). Only name + arguments on the wire (the
// internal id is excluded upstream).
void to_json(nlohmann::json& j, const FunctionCall& f) {
  j = nlohmann::json{{"name", f.name}, {"arguments", f.arguments}};
}

// engine/protocol.py:319 (ToolCall). {id, type:"function", function{...}}.
void to_json(nlohmann::json& j, const ToolCall& t) {
  j = nlohmann::json{
      {"id", t.id},
      {"type", t.type},
      {"function", t.function},
  };
}

// engine/protocol.py:325 (DeltaFunctionCall). name/arguments emitted only when
// present (per-chunk deltas).
void to_json(nlohmann::json& j, const DeltaFunctionCall& f) {
  j = nlohmann::json::object();
  if (f.name.has_value()) j["name"] = *f.name;
  if (f.arguments.has_value()) j["arguments"] = *f.arguments;
}

// engine/protocol.py:331 (DeltaToolCall). index is always present; id/type
// emitted only on the chunk that introduces the call.
void to_json(nlohmann::json& j, const DeltaToolCall& t) {
  j = nlohmann::json{{"index", t.index}, {"function", t.function}};
  if (t.id.has_value()) j["id"] = *t.id;
  if (t.type.has_value()) j["type"] = *t.type;
}

// chat_completion/protocol.py:57 (ChatMessage). content is always serialized
// (null when a tool call carries no text); tool_calls is emitted only when
// non-empty (upstream _serialize pops an empty list).
void to_json(nlohmann::json& j, const ChatMessage& m) {
  j = nlohmann::json{{"role", m.role}, {"content", OrNull(m.content)}};
  if (m.tool_calls.has_value() && !m.tool_calls->empty()) {
    j["tool_calls"] = *m.tool_calls;
  }
}

// engine/protocol.py:350 (DeltaMessage). tool_calls emitted only when non-empty.
void to_json(nlohmann::json& j, const DeltaMessage& m) {
  j = nlohmann::json::object();
  if (m.role.has_value()) j["role"] = *m.role;
  if (m.content.has_value()) j["content"] = *m.content;
  if (m.tool_calls.has_value() && !m.tool_calls->empty()) {
    j["tool_calls"] = *m.tool_calls;
  }
}

void to_json(nlohmann::json& j, const ChatCompletionResponseChoice& c) {
  j = nlohmann::json{
      {"index", c.index},
      {"message", c.message},
      {"finish_reason", OrNull(c.finish_reason)},
  };
}

void to_json(nlohmann::json& j, const ChatCompletionResponse& r) {
  j = nlohmann::json{
      {"id", r.id},
      {"object", r.object},
      {"created", r.created},
      {"model", r.model},
      {"choices", r.choices},
      {"usage", r.usage},
  };
}

void to_json(nlohmann::json& j, const ChatCompletionResponseStreamChoice& c) {
  j = nlohmann::json{
      {"index", c.index},
      {"delta", c.delta},
      {"finish_reason", OrNull(c.finish_reason)},
  };
}

void to_json(nlohmann::json& j, const ChatCompletionStreamResponse& r) {
  j = nlohmann::json{
      {"id", r.id},
      {"object", r.object},
      {"created", r.created},
      {"model", r.model},
      {"choices", r.choices},
  };
  if (r.usage.has_value()) j["usage"] = *r.usage;
}

}  // namespace vllm::entrypoints::openai
