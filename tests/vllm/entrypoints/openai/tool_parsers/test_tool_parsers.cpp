// Tests for the OpenAI tool-call parsers (non-streaming).
// (vllm/tool_parsers/{abstract_tool_parser,hermes_tool_parser,
//  qwen3_engine_tool_parser}.py @ e24d1b24).
//
// The gate model Qwen3.6-35B (arch qwen35moe) emits the Hermes
// `<tool_call>{json}</tool_call>` format (its tokenizer carries <tool_call> /
// </tool_call> tokens and no <function=>/<parameter=> XML tokens — see
// tests/parity/goldens/tokenizer_qwen36/tokenizer.json; docs/features/
// tool_calling.md:317 maps Qwen non-Coder models to the `hermes` parser), so
// Qwen3ToolParser reuses the Hermes extraction verbatim.
#include <doctest/doctest.h>

#include <memory>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"
#include "vllm/entrypoints/openai/tool_parsers/qwen3.h"

using namespace vllm::entrypoints::openai;

namespace {
ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }
}  // namespace

// (a) One <tool_call> block -> tools_called + one ToolCall, no content.
TEST_CASE("Hermes: single tool call, no leading content") {
  HermesToolParser parser;
  const std::string out =
      "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Paris\"}}\n</tool_call>";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].function.arguments == "{\"city\":\"Paris\"}");
  // id scheme: chatcmpl-tool-<uuid> (upstream make_tool_call_id).
  CHECK(info.tool_calls[0].id.rfind("chatcmpl-tool-", 0) == 0);
  // No leading text before the first <tool_call> => content is None.
  CHECK_FALSE(info.content.has_value());
}

// (b) Two <tool_call> blocks -> two ToolCalls.
TEST_CASE("Hermes: two tool calls") {
  HermesToolParser parser;
  const std::string out =
      "<tool_call>{\"name\": \"a\", \"arguments\": {\"x\": 1}}</tool_call>"
      "<tool_call>{\"name\": \"b\", \"arguments\": {\"y\": 2}}</tool_call>";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "a");
  CHECK(info.tool_calls[0].function.arguments == "{\"x\":1}");
  CHECK(info.tool_calls[1].function.name == "b");
  CHECK(info.tool_calls[1].function.arguments == "{\"y\":2}");
  CHECK_FALSE(info.content.has_value());
}

// (c) Leading content + a tool call -> content = the leading text.
TEST_CASE("Hermes: leading content preserved") {
  HermesToolParser parser;
  const std::string out =
      "Let me check.\n<tool_call>{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Paris\"}}</tool_call>";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Let me check.\n");
}

// (d) No tool call -> tools_called=false, content = the whole text.
TEST_CASE("Hermes: no tool call is plain content") {
  HermesToolParser parser;
  const std::string out = "The weather in Paris is sunny.";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// (e) Malformed JSON in a block -> graceful fallback (whole output as content),
//     matching upstream's try/except that returns tools_called=false.
TEST_CASE("Hermes: malformed JSON falls back to content") {
  HermesToolParser parser;
  const std::string out =
      "<tool_call>{\"name\": \"get_weather\", \"arguments\": }</tool_call>";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// (e2) A valid-JSON block missing the "name"/"arguments" keys also falls back.
TEST_CASE("Hermes: JSON missing required keys falls back") {
  HermesToolParser parser;
  const std::string out = "<tool_call>{\"foo\": 1}</tool_call>";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// (f) The Qwen3 parser on the gate-model (Hermes) format parses identically.
TEST_CASE("Qwen3: gate-model Hermes format parses") {
  Qwen3ToolParser parser;
  const std::string out =
      "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"cats\"}}\n"
      "</tool_call>";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "search");
  CHECK(info.tool_calls[0].function.arguments == "{\"q\":\"cats\"}");
}

// (g) The factory returns the right parser by name.
TEST_CASE("Factory: get_tool_parser by name") {
  auto hermes = get_tool_parser("hermes");
  REQUIRE(hermes != nullptr);
  auto qwen = get_tool_parser("qwen3");
  REQUIRE(qwen != nullptr);

  const std::string out =
      "<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>";
  auto req = empty_request();
  CHECK(hermes->extract_tool_calls(out, req).tools_called == true);
  CHECK(qwen->extract_tool_calls(out, req).tools_called == true);

  // Unknown name -> nullptr (upstream KeyError; we return nullptr).
  CHECK(get_tool_parser("does-not-exist") == nullptr);
}
