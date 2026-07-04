// Ported from: vllm/entrypoints/openai/protocol.py @ e24d1b24
// (split upstream into engine/completion/chat_completion/protocol.py — see the
// header for the exact per-shape source citations).

#include "vllm/entrypoints/openai/protocol.h"

#include <string>
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
  GetOpt(j, "logprobs", r.logprobs);
  GetOpt(j, "prompt_logprobs", r.prompt_logprobs);
  GetOr(j, "echo", r.echo);
  GetOr(j, "min_tokens", r.min_tokens);
  GetOr(j, "ignore_eos", r.ignore_eos);
  GetOr(j, "include_stop_str_in_output", r.include_stop_str_in_output);
  GetOr(j, "skip_special_tokens", r.skip_special_tokens);
  GetOr(j, "spaces_between_special_tokens", r.spaces_between_special_tokens);
}

void from_json(const nlohmann::json& j, ChatMessage& m) {
  GetOr(j, "role", m.role);
  // T0: bare-string content (multimodal content-part arrays deferred).
  if (auto it = j.find("content"); it != j.end() && it->is_string()) {
    m.content = it->get<std::string>();
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
  GetOr(j, "logprobs", r.logprobs);
  GetOr(j, "top_logprobs", r.top_logprobs);
  GetOr(j, "echo", r.echo);
  GetOpt(j, "prompt_logprobs", r.prompt_logprobs);
  GetOr(j, "min_tokens", r.min_tokens);
  GetOr(j, "ignore_eos", r.ignore_eos);
  GetOr(j, "include_stop_str_in_output", r.include_stop_str_in_output);
  GetOr(j, "skip_special_tokens", r.skip_special_tokens);
  GetOr(j, "spaces_between_special_tokens", r.spaces_between_special_tokens);
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

void to_json(nlohmann::json& j, const ChatMessage& m) {
  j = nlohmann::json{{"role", m.role}, {"content", OrNull(m.content)}};
}

void to_json(nlohmann::json& j, const DeltaMessage& m) {
  j = nlohmann::json::object();
  if (m.role.has_value()) j["role"] = *m.role;
  if (m.content.has_value()) j["content"] = *m.content;
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
