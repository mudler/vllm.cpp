// Multi-turn tool-conversation parsing (review-driven): the assistant-history
// tool_calls and the role="tool" reply's tool_call_id/name/reasoning must
// survive ChatMessage::from_json, or the chat template cannot associate a tool
// result with the call that produced it.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"

using nlohmann::json;
using vllm::entrypoints::openai::ChatMessage;

TEST_CASE("protocol: assistant history tool_calls parse from_json") {
  const json j = json::parse(R"({
    "role": "assistant",
    "content": null,
    "reasoning": "check the weather first",
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": "{\"city\": \"Rome\"}"}}]
  })");
  const ChatMessage m = j.get<ChatMessage>();
  CHECK(m.role == "assistant");
  CHECK(!m.content.has_value());
  REQUIRE(m.tool_calls.has_value());
  REQUIRE(m.tool_calls->size() == 1);
  CHECK((*m.tool_calls)[0].id == "call_1");
  CHECK((*m.tool_calls)[0].function.name == "get_weather");
  CHECK((*m.tool_calls)[0].function.arguments == "{\"city\": \"Rome\"}");
  REQUIRE(m.reasoning.has_value());
  CHECK(*m.reasoning == "check the weather first");
}

TEST_CASE("protocol: role=tool replies keep tool_call_id and name") {
  const json j = json::parse(R"({
    "role": "tool", "tool_call_id": "call_1", "name": "get_weather",
    "content": "{\"temp\": 21}"
  })");
  const ChatMessage m = j.get<ChatMessage>();
  CHECK(m.role == "tool");
  REQUIRE(m.tool_call_id.has_value());
  CHECK(*m.tool_call_id == "call_1");
  REQUIRE(m.name.has_value());
  CHECK(*m.name == "get_weather");
  // Round-trip: serialization carries the identity back out.
  const json out = m;
  CHECK(out.at("tool_call_id") == "call_1");
  CHECK(out.at("name") == "get_weather");
}

TEST_CASE("protocol: plain turns stay free of tool-identity keys") {
  const json j = json::parse(R"({"role": "user", "content": "hi"})");
  const ChatMessage m = j.get<ChatMessage>();
  CHECK(!m.tool_call_id.has_value());
  CHECK(!m.name.has_value());
  CHECK(!m.tool_calls.has_value());
  const json out = m;
  CHECK(!out.contains("tool_call_id"));
  CHECK(!out.contains("name"));
}
