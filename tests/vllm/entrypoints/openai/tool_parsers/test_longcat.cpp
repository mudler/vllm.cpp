// Tests for the LongCat-Flash tool-call parser.
// (vllm/tool_parsers/longcat_tool_parser.py @ e24d1b24 — a Hermes subclass that
//  swaps the wrapper tag to <longcat_tool_call>.)
//
// Ports the upstream common-suite cases (test_longcat_tool_parser.py) and adds
// coverage that the wrapper swap is the ONLY behavioural change: arguments are
// re-serialized compact exactly like Hermes, streaming is inherited, and the
// bare <tool_call> Hermes tag is NOT recognised (LongCat's template emits only
// the <longcat_tool_call> wrapper).
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/longcat.h"

using namespace vllm::entrypoints::openai;

namespace {
ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

std::vector<DeltaMessage> DriveStream(ToolParser& parser,
                                      const std::vector<std::string>& deltas) {
  std::vector<DeltaMessage> out;
  std::string previous;
  ChatCompletionRequest req;
  for (const std::string& delta : deltas) {
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (dm.has_value()) out.push_back(std::move(*dm));
  }
  return out;
}
}  // namespace

TEST_CASE("Longcat: single tool call, no leading content") {
  LongcatToolParser parser;
  const std::string out =
      R"(<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</longcat_tool_call>)";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  // Hermes re-serializes arguments compact.
  CHECK(info.tool_calls[0].function.arguments == R"({"city":"Tokyo"})");
  CHECK(info.tool_calls[0].id.rfind("chatcmpl-tool-", 0) == 0);
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("Longcat: parallel tool calls") {
  LongcatToolParser parser;
  const std::string out =
      R"(<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</longcat_tool_call>)"
      "\n"
      R"(<longcat_tool_call>{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}</longcat_tool_call>)";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[1].function.name == "get_time");
}

TEST_CASE("Longcat: empty arguments") {
  LongcatToolParser parser;
  const std::string out =
      R"(<longcat_tool_call>{"name": "refresh", "arguments": {}}</longcat_tool_call>)";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.arguments == "{}");
}

TEST_CASE("Longcat: surrounding text -> leading content preserved") {
  LongcatToolParser parser;
  const std::string out =
      "Let me check the weather for you.\n"
      R"(<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</longcat_tool_call>)"
      "\nHere is the result.";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Let me check the weather for you.\n");
}

TEST_CASE("Longcat: escaped strings parse") {
  LongcatToolParser parser;
  const std::string out =
      R"(<longcat_tool_call>{"name": "f", "arguments": {"quoted": "He said \"hi\"", "path": "C:\\Users"}}</longcat_tool_call>)";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  const auto parsed =
      nlohmann::json::parse(info.tool_calls[0].function.arguments);
  CHECK(parsed["quoted"] == "He said \"hi\"");
  CHECK(parsed["path"] == "C:\\Users");
}

TEST_CASE("Longcat: no tool call is plain content") {
  LongcatToolParser parser;
  const std::string out = "This is a regular response without any tool calls.";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("Longcat: malformed input falls back gracefully") {
  LongcatToolParser parser;
  auto req = empty_request();
  // Unterminated / invalid JSON inside the tail capture -> Hermes try/except
  // fallback => tools_called=false, whole output as content.
  const std::string bad = R"(<longcat_tool_call>{"name": "func", "arguments": {)";
  auto info = parser.extract_tool_calls(bad, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == bad);
}

TEST_CASE("Longcat: bare <tool_call> Hermes tag is NOT recognised") {
  // The longcat parser matches ONLY its own <longcat_tool_call> wrapper: a plain
  // Hermes <tool_call> block is treated as content, not a call. (Verified: the
  // LongCat-Flash template emits only <longcat_tool_call>.)
  LongcatToolParser parser;
  const std::string out =
      R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>)";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("Longcat streaming: name-first then argument deltas") {
  LongcatToolParser parser;
  const std::vector<std::string> deltas = {
      "<longcat_tool_call>", R"({"name": "get_)", R"(weather", )",
      R"("arguments": {"ci)", R"(ty": "Par)", R"(is"}})",
      "</longcat_tool_call>"};
  const std::vector<DeltaMessage> msgs = DriveStream(parser, deltas);

  std::string name;
  std::string args;
  int name_deltas = 0;
  for (const DeltaMessage& m : msgs) {
    if (!m.tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *m.tool_calls) {
      if (tc.function.name.has_value()) {
        ++name_deltas;
        name = *tc.function.name;
      }
      if (tc.function.arguments.has_value()) args += *tc.function.arguments;
    }
  }
  CHECK(name_deltas == 1);
  CHECK(name == "get_weather");
  CHECK(args == R"({"city": "Paris"})");
}

TEST_CASE("Factory: longcat by name") {
  auto p = get_tool_parser("longcat");
  REQUIRE(p != nullptr);
  const std::string out =
      R"(<longcat_tool_call>{"name": "f", "arguments": {}}</longcat_tool_call>)";
  auto req = empty_request();
  CHECK(p->extract_tool_calls(out, req).tools_called == true);
}
