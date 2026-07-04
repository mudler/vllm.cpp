// Tests for the OpenAI protocol port
// (vllm/entrypoints/openai/{engine,completion,chat_completion}/protocol.py
// @ e24d1b24).
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
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
