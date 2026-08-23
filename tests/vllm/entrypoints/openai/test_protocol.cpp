// Tests for the OpenAI protocol port
// (vllm/entrypoints/openai/{engine,completion,chat_completion}/protocol.py
// @ e24d1b24).
#include <doctest/doctest.h>

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/logprobs.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/sampling_params.h"

using nlohmann::json;
using namespace vllm::entrypoints::openai;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::SamplingType;

// (a) CompletionRequest parses + maps to SamplingParams correctly.
TEST_CASE("CompletionRequest parse + to_sampling_params") {
  auto j = json::parse(R"({
    "model": "m",
    "prompt": "hi",
    "max_tokens": 16,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["\n"],
    "stream": true
  })");
  auto req = j.get<CompletionRequest>();

  REQUIRE(req.model.has_value());
  CHECK(*req.model == "m");
  CHECK(req.prompt == "hi");
  CHECK(req.max_tokens.value() == 16);
  CHECK(req.temperature.value() == doctest::Approx(0.7));
  CHECK(req.top_p.value() == doctest::Approx(0.9));
  REQUIRE(req.stop.size() == 1);
  CHECK(req.stop[0] == "\n");
  CHECK(req.stream == true);

  SamplingParams sp = req.to_sampling_params();
  CHECK(sp.temperature == doctest::Approx(0.7));
  CHECK(sp.top_p == doctest::Approx(0.9));
  CHECK(sp.max_tokens.value() == 16);
  REQUIRE(sp.stop.size() == 1);
  CHECK(sp.stop[0] == "\n");
  // stream=true => DELTA output kind.
  CHECK(sp.output_kind == RequestOutputKind::kDelta);
  // PostInit ran: temperature 0.7 => random sampling (not greedy).
  CHECK(sp.Type() == SamplingType::kRandom);
}

// (f) max_tokens default + n>1 + stop string-vs-array normalization.
TEST_CASE("Completion to_sampling_params defaults + normalization") {
  SUBCASE("None sampling knobs resolve to _DEFAULT_SAMPLING_PARAMS") {
    auto req = json::parse(R"({"prompt":"x"})").get<CompletionRequest>();
    // max_tokens field default is 16 (normalize_null_max_tokens semantics).
    CHECK(req.max_tokens.value() == 16);
    SamplingParams sp = req.to_sampling_params();
    CHECK(sp.temperature == doctest::Approx(1.0));
    CHECK(sp.top_p == doctest::Approx(1.0));
    CHECK(sp.top_k == 0);
    CHECK(sp.min_p == doctest::Approx(0.0));
    CHECK(sp.repetition_penalty == doctest::Approx(1.0));
    CHECK(sp.max_tokens.value() == 16);
  }
  SUBCASE("n>1 under sampled (non-greedy) is accepted") {
    auto req =
        json::parse(R"({"prompt":"x","n":4,"temperature":0.8})").get<CompletionRequest>();
    CHECK(req.n == 4);
    SamplingParams sp = req.to_sampling_params();
    CHECK(sp.n == 4);
  }
  SUBCASE("stop as bare string normalizes to single-element list") {
    auto req = json::parse(R"({"prompt":"x","stop":"END"})").get<CompletionRequest>();
    REQUIRE(req.stop.size() == 1);
    CHECK(req.stop[0] == "END");
    SamplingParams sp = req.to_sampling_params();
    REQUIRE(sp.stop.size() == 1);
    CHECK(sp.stop[0] == "END");
  }
  SUBCASE("null max_tokens falls back to field default 16") {
    auto req = json::parse(R"({"prompt":"x","max_tokens":null})").get<CompletionRequest>();
    CHECK(req.max_tokens.value() == 16);
  }
  SUBCASE("default_max_tokens used when request omits max_tokens") {
    // Simulate a request whose max_tokens was cleared (serving path).
    CompletionRequest req;
    req.max_tokens = std::nullopt;
    SamplingParams sp = req.to_sampling_params(/*default_max_tokens=*/128);
    CHECK(sp.max_tokens.value() == 128);
  }
}

// (b) ChatCompletionRequest with [system,user] messages parses + maps.
TEST_CASE("ChatCompletionRequest parse messages + to_sampling_params") {
  auto j = json::parse(R"({
    "model": "m",
    "messages": [
      {"role": "system", "content": "be brief"},
      {"role": "user", "content": "hello"}
    ],
    "max_completion_tokens": 32,
    "temperature": 0.5,
    "stop": "STOP",
    "stream": false
  })");
  auto req = j.get<ChatCompletionRequest>();

  REQUIRE(req.messages.size() == 2);
  CHECK(req.messages[0].role == "system");
  CHECK(req.messages[0].content.value() == "be brief");
  CHECK(req.messages[1].role == "user");
  CHECK(req.messages[1].content.value() == "hello");
  CHECK(req.max_completion_tokens.value() == 32);
  REQUIRE(req.stop.size() == 1);
  CHECK(req.stop[0] == "STOP");

  SamplingParams sp = req.to_sampling_params();
  CHECK(sp.temperature == doctest::Approx(0.5));
  // max_completion_tokens preferred over (absent) max_tokens.
  CHECK(sp.max_tokens.value() == 32);
  // stream=false => FINAL_ONLY.
  CHECK(sp.output_kind == RequestOutputKind::kFinalOnly);
}

// Ported from tests/entrypoints/openai/completion/test_completion.py:
// test_completion_stream_options and chat_completion/test_chat.py:
// test_chat_completion_stream_options @ e24d1b24.
TEST_CASE("CompletionRequest stream_options parse defaults and validate stream") {
  SUBCASE("both usage modes parse on a streaming request") {
    auto req = json::parse(R"({
      "prompt":"hi", "stream":true,
      "stream_options":{"include_usage":true,
                        "continuous_usage_stats":true}
    })").get<CompletionRequest>();
    REQUIRE(req.stream_options.has_value());
    CHECK(req.stream_options->include_usage);
    CHECK(req.stream_options->continuous_usage_stats);
  }
  SUBCASE("null booleans resolve to false") {
    auto req = json::parse(R"({
      "prompt":"hi", "stream":true,
      "stream_options":{"include_usage":null,
                        "continuous_usage_stats":null}
    })").get<CompletionRequest>();
    REQUIRE(req.stream_options.has_value());
    CHECK_FALSE(req.stream_options->include_usage);
    CHECK_FALSE(req.stream_options->continuous_usage_stats);
  }
  SUBCASE("empty options remain allowed on a non-stream request") {
    auto req = json::parse(R"({"prompt":"hi","stream_options":{}})")
                   .get<CompletionRequest>();
    REQUIRE(req.stream_options.has_value());
    CHECK_FALSE(req.stream_options->include_usage);
  }
  SUBCASE("non-empty options reject a non-stream request") {
    CHECK_THROWS_WITH_AS(
        json::parse(R"({"prompt":"hi","stream":false,
                         "stream_options":{"include_usage":true}})")
            .get<CompletionRequest>(),
        doctest::Contains("Stream options can only be defined"),
        std::exception);
    CHECK_THROWS(json::parse(R"({"prompt":"hi","stream":false,
                                 "stream_options":{"include_usage":null}})")
                     .get<CompletionRequest>());
    CHECK_THROWS(json::parse(R"({"prompt":"hi","stream":false,
                                 "stream_options":{"continuous_usage_stats":true}})")
                     .get<CompletionRequest>());
  }
}

TEST_CASE("ChatCompletionRequest stream_options mirror completion validation") {
  auto req = json::parse(R"({
    "messages":[{"role":"user","content":"hi"}], "stream":true,
    "stream_options":{"include_usage":true,
                      "continuous_usage_stats":false}
  })").get<ChatCompletionRequest>();
  REQUIRE(req.stream_options.has_value());
  CHECK(req.stream_options->include_usage);
  CHECK_FALSE(req.stream_options->continuous_usage_stats);

  CHECK_THROWS(json::parse(R"({
    "messages":[], "stream":false,
    "stream_options":{"include_usage":true}
  })").get<ChatCompletionRequest>());
}

TEST_CASE("should_include_usage mirrors request and force selection") {
  CHECK_FALSE(ShouldIncludeUsage(std::nullopt, false).include_usage);

  StreamOptions continuous_only;
  continuous_only.continuous_usage_stats = true;
  const auto ignored = ShouldIncludeUsage(continuous_only, false);
  CHECK_FALSE(ignored.include_usage);
  CHECK_FALSE(ignored.include_continuous_usage);

  StreamOptions final_only;
  final_only.include_usage = true;
  const auto final = ShouldIncludeUsage(final_only, false);
  CHECK(final.include_usage);
  CHECK_FALSE(final.include_continuous_usage);

  const auto forced = ShouldIncludeUsage(std::nullopt, true);
  CHECK(forced.include_usage);
  CHECK(forced.include_continuous_usage);
}

// (c) CompletionResponse + CompletionStreamResponse serialize to exact shape.
TEST_CASE("CompletionResponse serialization shape") {
  CompletionResponse resp;
  resp.id = "cmpl-1";
  resp.created = 123;
  resp.model = "m";
  CompletionResponseChoice choice;
  choice.index = 0;
  choice.text = "world";
  choice.finish_reason = "stop";
  resp.choices.push_back(choice);
  resp.usage = UsageInfo{2, 5, 3};

  json j = resp;
  CHECK(j["object"] == "text_completion");
  CHECK(j["id"] == "cmpl-1");
  CHECK(j["created"] == 123);
  CHECK(j["model"] == "m");
  REQUIRE(j["choices"].is_array());
  CHECK(j["choices"][0]["index"] == 0);
  CHECK(j["choices"][0]["text"] == "world");
  CHECK(j["choices"][0]["finish_reason"] == "stop");
  CHECK(j["usage"]["prompt_tokens"] == 2);
  CHECK(j["usage"]["total_tokens"] == 5);
  CHECK(j["usage"]["completion_tokens"] == 3);
}

TEST_CASE("CompletionStreamResponse serialization shape") {
  CompletionStreamResponse resp;
  resp.id = "cmpl-1";
  resp.created = 123;
  resp.model = "m";
  CompletionResponseStreamChoice choice;
  choice.index = 0;
  choice.text = "wor";
  resp.choices.push_back(choice);  // no usage, no finish_reason on mid chunk

  json j = resp;
  CHECK(j["object"] == "text_completion");
  CHECK(j["choices"][0]["text"] == "wor");
  // finish_reason present as null on a mid-stream chunk.
  CHECK(j["choices"][0]["finish_reason"].is_null());
  // usage omitted when unset.
  CHECK(j.contains("usage") == false);
}

// (d) ChatCompletion stream chunk (delta) shape.
TEST_CASE("ChatCompletionStreamResponse delta shape") {
  ChatCompletionStreamResponse resp;
  resp.id = "chatcmpl-1";
  resp.created = 1;
  resp.model = "m";
  ChatCompletionResponseStreamChoice choice;
  choice.index = 0;
  choice.delta.role = "assistant";  // first chunk: role delta
  resp.choices.push_back(choice);

  json j = resp;
  CHECK(j["object"] == "chat.completion.chunk");
  CHECK(j["choices"][0]["delta"]["role"] == "assistant");
  CHECK(j["choices"][0]["finish_reason"].is_null());

  // A content delta chunk.
  ChatCompletionStreamResponse resp2;
  resp2.model = "m";
  ChatCompletionResponseStreamChoice c2;
  c2.delta.content = "hi";
  resp2.choices.push_back(c2);
  json j2 = resp2;
  CHECK(j2["choices"][0]["delta"]["content"] == "hi");
  // role omitted when unset in delta.
  CHECK(j2["choices"][0]["delta"].contains("role") == false);
}

TEST_CASE("ChatCompletionResponse (non-stream) shape") {
  ChatCompletionResponse resp;
  resp.id = "chatcmpl-1";
  resp.created = 1;
  resp.model = "m";
  ChatCompletionResponseChoice choice;
  choice.index = 0;
  choice.message.role = "assistant";
  choice.message.content = "hi there";
  choice.finish_reason = "stop";
  resp.choices.push_back(choice);
  resp.usage = UsageInfo{3, 7, 4};

  json j = resp;
  CHECK(j["object"] == "chat.completion");
  CHECK(j["choices"][0]["message"]["role"] == "assistant");
  CHECK(j["choices"][0]["message"]["content"] == "hi there");
  CHECK(j["choices"][0]["finish_reason"] == "stop");
  CHECK(j["usage"]["total_tokens"] == 7);
}

// (e) ErrorResponse shape.
TEST_CASE("ErrorResponse shape") {
  ErrorResponse err;
  err.error.message = "bad request";
  err.error.type = "BadRequestError";
  err.error.code = 400;

  json j = err;
  CHECK(j["error"]["message"] == "bad request");
  CHECK(j["error"]["type"] == "BadRequestError");
  CHECK(j["error"]["code"] == 400);
}

// ─── response_format -> structured_outputs (M3.4 Task 5) ─────────────────────
// chat_completion/protocol.py:629-658 + completion/protocol.py:309-338.

// json_schema -> structured_outputs.json = the serialized schema.
TEST_CASE("ChatCompletionRequest response_format json_schema -> structured_outputs.json") {
  auto j = json::parse(R"({
    "messages": [{"role":"user","content":"hi"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "schema": {"type":"object","properties":{"age":{"type":"integer"}},
                   "required":["age"]}
      }
    }
  })");
  auto req = j.get<ChatCompletionRequest>();
  REQUIRE(req.response_format.has_value());
  CHECK(req.response_format->type == "json_schema");
  REQUIRE(req.response_format->json_schema.has_value());
  REQUIRE(req.response_format->json_schema->json_schema.has_value());

  SamplingParams sp = req.to_sampling_params();
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->json.has_value());
  CHECK_FALSE(sp.structured_outputs->json_object.value_or(false));
  // The stored json is the serialized schema (round-trips to the same object).
  const json parsed = json::parse(*sp.structured_outputs->json);
  CHECK(parsed["type"] == "object");
  CHECK(parsed["required"][0] == "age");
  // Exactly one constraint set (PostInit / Verify passed).
  CHECK(sp.structured_outputs->all_constraints_none() == false);
}

// json_object -> structured_outputs.json_object = true.
TEST_CASE("ChatCompletionRequest response_format json_object -> json_object=true") {
  auto j = json::parse(R"({
    "messages": [{"role":"user","content":"hi"}],
    "response_format": {"type": "json_object"}
  })");
  auto req = j.get<ChatCompletionRequest>();
  SamplingParams sp = req.to_sampling_params();
  REQUIRE(sp.structured_outputs.has_value());
  CHECK(sp.structured_outputs->json_object.value_or(false) == true);
  CHECK_FALSE(sp.structured_outputs->json.has_value());
}

// type "text" (and absent) -> no structured-output constraint.
TEST_CASE("ChatCompletionRequest response_format text/absent -> no structured_outputs") {
  {
    auto req = json::parse(R"({"messages":[{"role":"user","content":"hi"}],
                               "response_format":{"type":"text"}})")
                   .get<ChatCompletionRequest>();
    SamplingParams sp = req.to_sampling_params();
    CHECK_FALSE(sp.structured_outputs.has_value());
  }
  {
    auto req = json::parse(R"({"messages":[{"role":"user","content":"hi"}]})")
                   .get<ChatCompletionRequest>();
    SamplingParams sp = req.to_sampling_params();
    CHECK_FALSE(sp.structured_outputs.has_value());
  }
}

// The completion endpoint carries the same mapping.
TEST_CASE("CompletionRequest response_format json_schema -> structured_outputs.json") {
  auto j = json::parse(R"({
    "prompt": "hi",
    "response_format": {
      "type": "json_schema",
      "json_schema": {"name":"n","schema":{"type":"string"}}
    }
  })");
  auto req = j.get<CompletionRequest>();
  SamplingParams sp = req.to_sampling_params();
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->json.has_value());
  CHECK(json::parse(*sp.structured_outputs->json)["type"] == "string");
}

// A json_schema response_format WITHOUT the json_schema field is a 400
// (validate_response_format throws at parse).
TEST_CASE("response_format json_schema without json_schema field throws") {
  CHECK_THROWS(json::parse(R"({"messages":[],
                              "response_format":{"type":"json_schema"}})")
                   .get<ChatCompletionRequest>());
}

// ─── M3.3 Task 1: tools / tool_choice / tool_calls ───────────────────────────
// engine/protocol.py:246,310-335 + chat_completion/protocol.py:57,165-224,350.

// (a) tools:[{type,function{name,description,parameters}}] + tool_choice:"auto".
TEST_CASE("ChatCompletionRequest tools + tool_choice=auto parses") {
  auto j = json::parse(R"({
    "messages": [{"role":"user","content":"weather?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type":"object","properties":{"city":{"type":"string"}},
                       "required":["city"]}
      }
    }],
    "tool_choice": "auto"
  })");
  auto req = j.get<ChatCompletionRequest>();

  REQUIRE(req.tools.has_value());
  REQUIRE(req.tools->size() == 1);
  CHECK((*req.tools)[0].type == "function");
  CHECK((*req.tools)[0].function.name == "get_weather");
  REQUIRE((*req.tools)[0].function.description.has_value());
  CHECK(*(*req.tools)[0].function.description == "Get the weather");
  REQUIRE((*req.tools)[0].function.parameters.has_value());
  CHECK((*(*req.tools)[0].function.parameters)["type"] == "object");
  CHECK((*(*req.tools)[0].function.parameters)["required"][0] == "city");

  REQUIRE(req.tool_choice.has_value());
  CHECK(req.tool_choice->mode == "auto");
  CHECK_FALSE(req.tool_choice->function_name.has_value());
}

// (b) named tool_choice object -> mode=function + function_name.
TEST_CASE("ChatCompletionRequest named tool_choice object parses") {
  auto j = json::parse(R"({
    "messages": [{"role":"user","content":"hi"}],
    "tools": [{"type":"function","function":{"name":"get_weather"}}],
    "tool_choice": {"type":"function","function":{"name":"get_weather"}}
  })");
  auto req = j.get<ChatCompletionRequest>();
  REQUIRE(req.tool_choice.has_value());
  CHECK(req.tool_choice->mode == "function");
  REQUIRE(req.tool_choice->function_name.has_value());
  CHECK(*req.tool_choice->function_name == "get_weather");
}

// (c) tool_choice:"required" / "none" string forms parse.
TEST_CASE("ChatCompletionRequest tool_choice required/none string forms") {
  {
    auto req = json::parse(R"({"messages":[],"tool_choice":"required"})")
                   .get<ChatCompletionRequest>();
    REQUIRE(req.tool_choice.has_value());
    CHECK(req.tool_choice->mode == "required");
  }
  {
    auto req = json::parse(R"({"messages":[],"tool_choice":"none"})")
                   .get<ChatCompletionRequest>();
    REQUIRE(req.tool_choice.has_value());
    CHECK(req.tool_choice->mode == "none");
  }
}

// (f) no tools/tool_choice -> fields absent (backward compat).
TEST_CASE("ChatCompletionRequest without tools -> nullopt (backward compat)") {
  auto req = json::parse(R"({"messages":[{"role":"user","content":"hi"}]})")
                 .get<ChatCompletionRequest>();
  CHECK_FALSE(req.tools.has_value());
  CHECK_FALSE(req.tool_choice.has_value());
}

// (d) response message carrying tool_calls serializes to exact OpenAI shape.
TEST_CASE("ChatMessage with tool_calls serializes to OpenAI shape") {
  ChatCompletionResponse resp;
  resp.id = "chatcmpl-1";
  resp.created = 1;
  resp.model = "m";
  ChatCompletionResponseChoice choice;
  choice.index = 0;
  choice.message.role = "assistant";
  choice.message.content = std::nullopt;  // no content when calling a tool
  ToolCall tc;
  tc.id = "call_abc";
  tc.function.name = "get_weather";
  tc.function.arguments = R"({"city":"Paris"})";
  choice.message.tool_calls = std::vector<ToolCall>{tc};
  choice.finish_reason = "tool_calls";
  resp.choices.push_back(choice);
  resp.usage = UsageInfo{3, 7, 4};

  json j = resp;
  const auto& msg = j["choices"][0]["message"];
  CHECK(msg["role"] == "assistant");
  // content is null (tool call, no text).
  CHECK(msg["content"].is_null());
  REQUIRE(msg["tool_calls"].is_array());
  REQUIRE(msg["tool_calls"].size() == 1);
  CHECK(msg["tool_calls"][0]["id"] == "call_abc");
  CHECK(msg["tool_calls"][0]["type"] == "function");
  CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
  CHECK(msg["tool_calls"][0]["function"]["arguments"] == R"({"city":"Paris"})");
  CHECK(j["choices"][0]["finish_reason"] == "tool_calls");
}

// A message WITHOUT tool_calls omits the key entirely (upstream serializer pop).
TEST_CASE("ChatMessage without tool_calls omits the key") {
  ChatMessage m;
  m.role = "assistant";
  m.content = "hello";
  json j = m;
  CHECK(j["content"] == "hello");
  CHECK(j.contains("tool_calls") == false);
}

// (e) stream DeltaMessage tool_calls: name-first chunk then arguments delta.
TEST_CASE("DeltaMessage tool_calls stream shape (index-based)") {
  // First chunk: index + id + function.name.
  {
    DeltaMessage d;
    DeltaToolCall dtc;
    dtc.index = 0;
    dtc.id = "call_abc";
    dtc.type = "function";
    dtc.function.name = "get_weather";
    d.tool_calls = std::vector<DeltaToolCall>{dtc};
    json j = d;
    REQUIRE(j["tool_calls"].is_array());
    CHECK(j["tool_calls"][0]["index"] == 0);
    CHECK(j["tool_calls"][0]["id"] == "call_abc");
    CHECK(j["tool_calls"][0]["type"] == "function");
    CHECK(j["tool_calls"][0]["function"]["name"] == "get_weather");
    // arguments omitted when unset in this chunk.
    CHECK(j["tool_calls"][0]["function"].contains("arguments") == false);
  }
  // Subsequent chunk: index + function.arguments delta only.
  {
    DeltaMessage d;
    DeltaToolCall dtc;
    dtc.index = 0;
    dtc.function.arguments = R"({"city":)";
    d.tool_calls = std::vector<DeltaToolCall>{dtc};
    json j = d;
    CHECK(j["tool_calls"][0]["index"] == 0);
    CHECK(j["tool_calls"][0]["function"]["arguments"] == R"({"city":)");
    CHECK(j["tool_calls"][0]["function"].contains("name") == false);
    CHECK(j["tool_calls"][0].contains("id") == false);
    CHECK(j["tool_calls"][0].contains("type") == false);
  }
}

// A delta WITHOUT tool_calls omits the key (unchanged content-only path).
TEST_CASE("DeltaMessage without tool_calls omits the key") {
  DeltaMessage d;
  d.content = "hi";
  json j = d;
  CHECK(j["content"] == "hi");
  CHECK(j.contains("tool_calls") == false);
}

// ─── ROAD-V1-C7 SAMPLE-LOGIT-FILTERS: logit_bias / allowed_token_ids / bad_words ─
TEST_CASE("CompletionRequest parses logit_bias (string keys) + clamp") {
  auto req = json::parse(R"({
    "prompt":"x",
    "logit_bias":{"100":250.0,"7":-999.0,"42":3.5},
    "allowed_token_ids":[5,9,900],
    "bad_words":["foo","bar"]
  })").get<CompletionRequest>();
  SamplingParams sp = req.to_sampling_params();

  // string keys -> int token ids; bias clamped to [-100, 100].
  REQUIRE(sp.logit_bias.has_value());
  CHECK(sp.logit_bias->at(100) == doctest::Approx(100.0f));   // clamped down
  CHECK(sp.logit_bias->at(7) == doctest::Approx(-100.0f));    // clamped up
  CHECK(sp.logit_bias->at(42) == doctest::Approx(3.5f));

  REQUIRE(sp.allowed_token_ids.has_value());
  CHECK(*sp.allowed_token_ids == std::vector<int32_t>{5, 9, 900});
  CHECK(sp.bad_words == std::vector<std::string>{"foo", "bar"});
}

TEST_CASE("ChatCompletionRequest parses logit_bias + allowed_token_ids + bad_words") {
  auto req = json::parse(R"({
    "messages":[{"role":"user","content":"hi"}],
    "logit_bias":{"11":50.0},
    "allowed_token_ids":[1,2],
    "bad_words":["x"]
  })").get<ChatCompletionRequest>();
  SamplingParams sp = req.to_sampling_params();
  REQUIRE(sp.logit_bias.has_value());
  CHECK(sp.logit_bias->at(11) == doctest::Approx(50.0f));
  REQUIRE(sp.allowed_token_ids.has_value());
  CHECK(*sp.allowed_token_ids == std::vector<int32_t>{1, 2});
  CHECK(sp.bad_words == std::vector<std::string>{"x"});
}

TEST_CASE("logit_bias with a non-integer key is rejected (400)") {
  auto req = json::parse(R"({
    "prompt":"x","logit_bias":{"notanint":1.0}
  })").get<CompletionRequest>();
  CHECK_THROWS_AS(req.to_sampling_params(), std::runtime_error);
}

TEST_CASE("empty allowed_token_ids is rejected via PostInit") {
  auto req = json::parse(R"({
    "prompt":"x","allowed_token_ids":[]
  })").get<CompletionRequest>();
  CHECK_THROWS_AS(req.to_sampling_params(), std::runtime_error);
}

// A non-positive max_tokens means "no client-side limit" (Hermes and some
// OpenAI clients send -1). It must arrive at the engine UNSET so the
// max_model_len - seq_len path runs; substituting a constant would silently
// truncate the request that explicitly asked to be left unlimited.
TEST_CASE("max_tokens <= 0 means UNSET, not a clamped constant") {
  SUBCASE("completions: -1 leaves max_tokens unset") {
    auto req = json::parse(R"({"prompt":"x","max_tokens":-1})")
                   .get<CompletionRequest>();
    CHECK_FALSE(req.to_sampling_params().max_tokens.has_value());
  }
  SUBCASE("completions: 0 leaves max_tokens unset") {
    auto req = json::parse(R"({"prompt":"x","max_tokens":0})")
                   .get<CompletionRequest>();
    CHECK_FALSE(req.to_sampling_params().max_tokens.has_value());
  }
  SUBCASE("completions: a positive value is honoured unchanged") {
    auto req = json::parse(R"({"prompt":"x","max_tokens":7})")
                   .get<CompletionRequest>();
    const auto sp = req.to_sampling_params();
    REQUIRE(sp.max_tokens.has_value());
    CHECK(*sp.max_tokens == 7);
  }
  SUBCASE("completions: -1 yields to the serving default when one is given") {
    auto req = json::parse(R"({"prompt":"x","max_tokens":-1})")
                   .get<CompletionRequest>();
    const auto sp = req.to_sampling_params(/*default_max_tokens=*/128);
    REQUIRE(sp.max_tokens.has_value());
    CHECK(*sp.max_tokens == 128);
  }
  SUBCASE("chat: -1 leaves max_tokens unset") {
    auto req = json::parse(
                   R"({"messages":[{"role":"user","content":"x"}],"max_tokens":-1})")
                   .get<ChatCompletionRequest>();
    CHECK_FALSE(req.to_sampling_params().max_tokens.has_value());
  }
  SUBCASE("chat: max_completion_tokens=-1 leaves max_tokens unset") {
    auto req = json::parse(
                   R"({"messages":[{"role":"user","content":"x"}],"max_completion_tokens":-1})")
                   .get<ChatCompletionRequest>();
    CHECK_FALSE(req.to_sampling_params().max_tokens.has_value());
  }
}

// ─── SAMPLE-PROMPT-LOGPROBS W2: the prompt_logprobs request/response surface ──
//
// Ported from vllm/entrypoints/openai/completion/protocol.py:474-499 and
// chat_completion/protocol.py:763-793 (the `mode="before"` validators) plus
// tests/entrypoints/openai/completion/test_completion_error.py:615-625
// (test_non_numeric_logprobs_rejected) @ 555967922.
//
// Upstream raises VLLMValidationError from a pydantic before-validator, which
// FastAPI turns into a 400. Our parser throws and api_server.cpp maps any
// exception out of the body parse to 400 BadRequestError, so the ported
// assertion is "the parse throws with upstream's message", and the socket
// cases in test_api_server.cpp carry the 400 itself.
TEST_CASE("prompt_logprobs / logprobs request validation mirrors upstream") {
  // test_non_numeric_logprobs_rejected, parametrized over the same two fields.
  SUBCASE("completions: a non-numeric value is refused by name") {
    CHECK_THROWS_WITH_AS(
        json::parse(R"({"prompt":"x","prompt_logprobs":"2"})")
            .get<CompletionRequest>(),
        doctest::Contains("`prompt_logprobs` must be an integer."),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        json::parse(R"({"prompt":"x","logprobs":"2"})").get<CompletionRequest>(),
        doctest::Contains("`logprobs` must be an integer."),
        std::invalid_argument);
  }
  SUBCASE("chat: a non-numeric value is refused by name") {
    CHECK_THROWS_WITH_AS(
        json::parse(
            R"({"messages":[{"role":"user","content":"x"}],"prompt_logprobs":"2"})")
            .get<ChatCompletionRequest>(),
        doctest::Contains("`prompt_logprobs` must be an integer."),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        json::parse(
            R"({"messages":[{"role":"user","content":"x"}],"top_logprobs":"2"})")
            .get<ChatCompletionRequest>(),
        doctest::Contains("`top_logprobs` must be an integer."),
        std::invalid_argument);
  }
  // protocol.py:490-494 — negative is refused, but -1 is the "whole vocabulary"
  // sentinel and is NOT.
  SUBCASE("completions: a negative value that is not -1 is refused") {
    CHECK_THROWS_WITH_AS(
        json::parse(R"({"prompt":"x","prompt_logprobs":-2})")
            .get<CompletionRequest>(),
        doctest::Contains("`prompt_logprobs` must be a positive value or -1."),
        std::invalid_argument);
  }
  SUBCASE("completions: -1 and 0 parse") {
    auto a = json::parse(R"({"prompt":"x","prompt_logprobs":-1})")
                 .get<CompletionRequest>();
    REQUIRE(a.prompt_logprobs.has_value());
    CHECK(*a.prompt_logprobs == -1);
    auto b = json::parse(R"({"prompt":"x","prompt_logprobs":0})")
                 .get<CompletionRequest>();
    REQUIRE(b.prompt_logprobs.has_value());
    CHECK(*b.prompt_logprobs == 0);
  }
  // completion/protocol.py:495-499 — the completion count has NO -1 sentinel;
  // any negative is refused, INCLUDING -1. This is deliberately not the chat
  // rule below, and it closes the divergence logprobs-all-sentinel.md records.
  SUBCASE("completions: a negative logprobs is refused, -1 included") {
    CHECK_THROWS_WITH_AS(
        json::parse(R"({"prompt":"x","logprobs":-1})").get<CompletionRequest>(),
        doctest::Contains("`logprobs` must be a positive value."),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        json::parse(R"({"prompt":"x","logprobs":-4})").get<CompletionRequest>(),
        doctest::Contains("`logprobs` must be a positive value."),
        std::invalid_argument);
  }
  // chat_completion/protocol.py:784-790 — the chat count DOES carry the -1
  // sentinel ("every vocabulary entry"), which this tree serves end to end
  // (test_serving.cpp "serving_chat: top_logprobs=-1 returns every vocab entry
  // per token"). Refusing -1 here would take that capability off the HTTP
  // surface, so the two endpoints get different rules exactly as upstream does.
  SUBCASE("chat: top_logprobs=-1 parses, and -2 is refused with the chat message") {
    auto ok = json::parse(
                  R"({"messages":[{"role":"user","content":"x"}],"logprobs":true,"top_logprobs":-1})")
                  .get<ChatCompletionRequest>();
    CHECK(ok.top_logprobs == -1);
    CHECK_THROWS_WITH_AS(
        json::parse(
            R"({"messages":[{"role":"user","content":"x"}],"logprobs":true,"top_logprobs":-2})")
            .get<ChatCompletionRequest>(),
        doctest::Contains("`top_logprobs` must be a positive value or -1."),
        std::invalid_argument);
  }
  // chat_completion/protocol.py:792-796 — a count that would emit a payload
  // needs the `logprobs` bool. 0 does not, so it parses without the flag.
  SUBCASE("chat: top_logprobs without logprobs=true is refused") {
    CHECK_THROWS_WITH_AS(
        json::parse(
            R"({"messages":[{"role":"user","content":"x"}],"top_logprobs":3})")
            .get<ChatCompletionRequest>(),
        doctest::Contains("when using `top_logprobs`, `logprobs` must be set to true."),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        json::parse(
            R"({"messages":[{"role":"user","content":"x"}],"top_logprobs":-1})")
            .get<ChatCompletionRequest>(),
        doctest::Contains("when using `top_logprobs`, `logprobs` must be set to true."),
        std::invalid_argument);
    auto zero = json::parse(
                    R"({"messages":[{"role":"user","content":"x"}],"top_logprobs":0})")
                    .get<ChatCompletionRequest>();
    CHECK(zero.top_logprobs == 0);
  }
  // protocol.py:483-488 — refused when stream is set AND the value would emit a
  // payload. The upstream condition is `> 0 or == -1`, so 0 with stream PARSES.
  SUBCASE("completions: prompt_logprobs with stream=true is refused") {
    CHECK_THROWS_WITH_AS(
        json::parse(R"({"prompt":"x","prompt_logprobs":1,"stream":true})")
            .get<CompletionRequest>(),
        doctest::Contains("`prompt_logprobs` are not available when `stream=True`."),
        std::invalid_argument);
    CHECK_THROWS_WITH_AS(
        json::parse(R"({"prompt":"x","prompt_logprobs":-1,"stream":true})")
            .get<CompletionRequest>(),
        doctest::Contains("`prompt_logprobs` are not available when `stream=True`."),
        std::invalid_argument);
  }
  SUBCASE("completions: prompt_logprobs=0 with stream=true parses (upstream `> 0`)") {
    auto r = json::parse(R"({"prompt":"x","prompt_logprobs":0,"stream":true})")
                 .get<CompletionRequest>();
    REQUIRE(r.prompt_logprobs.has_value());
    CHECK(*r.prompt_logprobs == 0);
  }
  SUBCASE("chat: prompt_logprobs with stream=true is refused") {
    CHECK_THROWS_WITH_AS(
        json::parse(
            R"({"messages":[{"role":"user","content":"x"}],"prompt_logprobs":2,"stream":true})")
            .get<ChatCompletionRequest>(),
        doctest::Contains("`prompt_logprobs` are not available when `stream=True`."),
        std::invalid_argument);
  }
  SUBCASE("chat: a negative value that is not -1 is refused") {
    CHECK_THROWS_WITH_AS(
        json::parse(
            R"({"messages":[{"role":"user","content":"x"}],"prompt_logprobs":-3})")
            .get<ChatCompletionRequest>(),
        doctest::Contains("`prompt_logprobs` must be a positive value or -1."),
        std::invalid_argument);
  }
}

// Ported from vllm/entrypoints/openai/completion/protocol.py:601
// (CompletionResponseChoice.prompt_logprobs) and chat_completion/protocol.py:126
// (ChatCompletionResponse.prompt_logprobs) @ 555967922. Both are
// `list[dict[int, Logprob] | None] | None`, dumped by `model_dump()` with the
// None fields present, so the wire shape is an explicit `null` when unset and a
// list whose FIRST entry is null (the first prompt token has no predecessor).
TEST_CASE("prompt_logprobs response serialization mirrors upstream") {
  SUBCASE("completions: absent serializes as an explicit null") {
    CompletionResponseChoice c;
    c.index = 0;
    c.text = "hi";
    const json j = c;
    REQUIRE(j.contains("prompt_logprobs"));
    CHECK(j.at("prompt_logprobs").is_null());
  }
  SUBCASE("chat: absent serializes as an explicit null") {
    ChatCompletionResponse r;
    r.id = "cmpl-1";
    const json j = r;
    REQUIRE(j.contains("prompt_logprobs"));
    CHECK(j.at("prompt_logprobs").is_null());
  }
  SUBCASE("present: first position null, later positions {id -> Logprob}") {
    vllm::PromptLogprobs plp;
    plp.push_back(std::nullopt);  // create_prompt_logprobs: first is None
    vllm::LogprobsOnePosition pos;
    pos.put(7, vllm::Logprob{-0.25f, 1, std::string("th")});
    pos.put(9, vllm::Logprob{-1.5f, 2, std::string("er")});
    plp.push_back(std::move(pos));

    CompletionResponseChoice c;
    c.prompt_logprobs = plp;
    const json j = c;
    const json& a = j.at("prompt_logprobs");
    REQUIRE(a.is_array());
    REQUIRE(a.size() == 2);
    CHECK(a.at(0).is_null());
    REQUIRE(a.at(1).is_object());
    // dict[int, Logprob] -> JSON object keyed by the DECIMAL token id.
    REQUIRE(a.at(1).contains("7"));
    CHECK(a.at(1).at("7").at("logprob").get<float>() == doctest::Approx(-0.25f));
    CHECK(a.at(1).at("7").at("rank").get<int>() == 1);
    CHECK(a.at(1).at("7").at("decoded_token").get<std::string>() == "th");
    REQUIRE(a.at(1).contains("9"));
    CHECK(a.at(1).at("9").at("logprob").get<float>() == doctest::Approx(-1.5f));
    CHECK(a.at(1).at("9").at("rank").get<int>() == 2);

    ChatCompletionResponse r;
    r.prompt_logprobs = plp;
    const json rj = r;
    REQUIRE(rj.at("prompt_logprobs").is_array());
    CHECK(rj.at("prompt_logprobs").at(0).is_null());
    CHECK(rj.at("prompt_logprobs").at(1).at("7").at("logprob").get<float>() ==
          doctest::Approx(-0.25f));
  }
  SUBCASE("a rank/decoded_token that upstream leaves None serializes as null") {
    vllm::PromptLogprobs plp;
    vllm::LogprobsOnePosition pos;
    pos.put(3, vllm::Logprob{-2.0f, std::nullopt, std::nullopt});
    plp.push_back(std::move(pos));
    CompletionResponseChoice c;
    c.prompt_logprobs = plp;
    const json j = c;
    CHECK(j.at("prompt_logprobs").at(0).at("3").at("rank").is_null());
    CHECK(j.at("prompt_logprobs").at(0).at("3").at("decoded_token").is_null());
  }
}

// clamp_prompt_logprobs — vllm/entrypoints/generate/base/serving.py:305-317.
// -inf is not representable in JSON, so upstream rewrites it to -9999.0 IN
// PLACE before the response is built. Every other value is untouched.
TEST_CASE("ClampPromptLogprobs rewrites -inf to -9999.0 in place") {
  vllm::PromptLogprobs plp;
  plp.push_back(std::nullopt);
  vllm::LogprobsOnePosition pos;
  pos.put(1, vllm::Logprob{-std::numeric_limits<float>::infinity(), 1, std::nullopt});
  pos.put(2, vllm::Logprob{-3.5f, 2, std::nullopt});
  plp.push_back(std::move(pos));

  std::optional<vllm::PromptLogprobs> opt = plp;
  ClampPromptLogprobs(opt);
  REQUIRE(opt.has_value());
  REQUIRE(opt->size() == 2);
  CHECK_FALSE((*opt)[0].has_value());
  const vllm::Logprob* one = (*opt)[1]->find(1);
  REQUIRE(one != nullptr);
  CHECK(one->logprob == doctest::Approx(-9999.0f));
  const vllm::Logprob* two = (*opt)[1]->find(2);
  REQUIRE(two != nullptr);
  CHECK(two->logprob == doctest::Approx(-3.5f));  // untouched

  std::optional<vllm::PromptLogprobs> none;
  ClampPromptLogprobs(none);  // None in, None out (serving.py:308-309)
  CHECK_FALSE(none.has_value());
}
