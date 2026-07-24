// Tests for the "granite-20b-fc" tool-call parser (Granite20bFCToolParser).
// Ported from: tests/tool_parsers/test_granite_20b_fc_tool_parser.py @ e24d1b24
// (driving common_tests.py::ToolParserTests). As with the granite tests we drive
// streaming character-by-character and reconstruct like StreamingToolReconstructor.
//
// xfail parity: the upstream config marks streaming test_surrounding_text
// xfail(strict) (test_granite_20b_fc_tool_parser.py:70-73) - 20B FC streaming
// requires the `<function_call>` marker at position 0 - so we assert that case
// only for the non-streaming path (which the config does NOT xfail).
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/granite_20b_fc.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

struct ReconTool {
  std::string id;
  std::string name;
  std::string arguments;
};
struct Recon {
  std::vector<ReconTool> tools;
  std::string content;
};

void Append(Recon& r, const DeltaMessage& m) {
  if (m.content.has_value()) r.content += *m.content;
  if (!m.tool_calls.has_value()) return;
  for (const DeltaToolCall& tc : *m.tool_calls) {
    if (tc.index >= 0 && static_cast<std::size_t>(tc.index) < r.tools.size()) {
      if (tc.function.arguments.has_value())
        r.tools[tc.index].arguments += *tc.function.arguments;
    } else {
      ReconTool rt;
      if (tc.id.has_value()) rt.id = *tc.id;
      if (tc.function.name.has_value()) rt.name = *tc.function.name;
      if (tc.function.arguments.has_value()) rt.arguments = *tc.function.arguments;
      r.tools.push_back(std::move(rt));
    }
  }
}

Recon RunStreaming(ToolParser& parser, const std::string& out) {
  Recon r;
  std::string previous;
  ChatCompletionRequest req;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::string delta = out.substr(i, 1);
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (dm.has_value()) Append(r, *dm);
  }
  return r;
}

}  // namespace

// ─── Non-streaming ───────────────────────────────────────────────────────────

TEST_CASE("granite-20b-fc: no tool calls is plain content") {
  Granite20bFCToolParser parser;
  const std::string out = "This is a regular response without any tool calls.";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("granite-20b-fc: single tool call") {
  Granite20bFCToolParser parser;
  const std::string out =
      "<function_call> {\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Tokyo\"}}";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].id.rfind("chatcmpl-tool-", 0) == 0);
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments)["city"] == "Tokyo");
  // No leading content -> None.
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("granite-20b-fc: parallel tool calls") {
  Granite20bFCToolParser parser;
  const std::string out =
      "<function_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Tokyo\"}}\n"
      "<function_call> {\"name\": \"get_time\", \"arguments\": {\"timezone\": "
      "\"Asia/Tokyo\"}}";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[1].function.name == "get_time");
  CHECK(info.tool_calls[0].id != info.tool_calls[1].id);
}

TEST_CASE("granite-20b-fc: various data types") {
  Granite20bFCToolParser parser;
  const std::string out =
      "<function_call> {\"name\": \"test_function\", \"arguments\": {"
      "\"string_field\": \"hello\", \"int_field\": 42, \"float_field\": 3.14, "
      "\"bool_field\": true, \"null_field\": null, \"array_field\": [\"a\"], "
      "\"object_field\": {\"nested\": \"value\"}}}";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = nlohmann::json::parse(info.tool_calls[0].function.arguments);
  CHECK(args["int_field"].is_number_integer());
  CHECK(args["float_field"].is_number_float());
  CHECK(args["null_field"].is_null());
}

TEST_CASE("granite-20b-fc: empty arguments") {
  Granite20bFCToolParser parser;
  const std::string out = "<function_call> {\"name\": \"refresh\", \"arguments\": {}}";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.arguments == "{}");
}

TEST_CASE("granite-20b-fc: surrounding text (non-streaming keeps leading content)") {
  Granite20bFCToolParser parser;
  const std::string out =
      "Let me check the weather for you.\n"
      "<function_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Tokyo\"}}";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Let me check the weather for you.\n");
}

TEST_CASE("granite-20b-fc: malformed inputs never throw (non-streaming)") {
  const std::vector<std::string> malformed = {
      "<function_call> {\"name\": \"func\", \"arguments\": {",  // truncated
      "<function_call> [{\"name\": \"func\", \"arguments\": {}}]",  // array, not dict
      "{\"name\": \"func\", \"arguments\": {}}",  // no marker
      "<function_call> {\"name\": 123}",  // no arguments key
  };
  for (const std::string& in : malformed) {
    Granite20bFCToolParser parser;
    auto req = empty_request();
    CHECK_NOTHROW((void)parser.extract_tool_calls(in, req));
  }
}

TEST_CASE("granite-20b-fc: empty output falls through as content") {
  Granite20bFCToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls("", req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "");
}

// ─── Streaming ───────────────────────────────────────────────────────────────

TEST_CASE("granite-20b-fc streaming: single tool call reconstructs") {
  Granite20bFCToolParser parser;
  const std::string out =
      "<function_call> {\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Tokyo\"}}";
  Recon r = RunStreaming(parser, out);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_weather");
  CHECK(r.tools[0].id.rfind("chatcmpl-tool-", 0) == 0);
  CHECK(nlohmann::json::parse(r.tools[0].arguments)["city"] == "Tokyo");
}

TEST_CASE("granite-20b-fc streaming: parallel tool calls reconstruct") {
  Granite20bFCToolParser parser;
  const std::string out =
      "<function_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Tokyo\"}}\n"
      "<function_call> {\"name\": \"get_time\", \"arguments\": {\"timezone\": "
      "\"Asia/Tokyo\"}}";
  Recon r = RunStreaming(parser, out);
  REQUIRE(r.tools.size() == 2);
  CHECK(r.tools[0].name == "get_weather");
  CHECK(r.tools[1].name == "get_time");
  CHECK(nlohmann::json::parse(r.tools[0].arguments)["city"] == "Tokyo");
  CHECK(nlohmann::json::parse(r.tools[1].arguments)["timezone"] == "Asia/Tokyo");
}

TEST_CASE("granite-20b-fc streaming: empty arguments reconstructs") {
  Granite20bFCToolParser parser;
  const std::string out = "<function_call> {\"name\": \"refresh\", \"arguments\": {}}";
  Recon r = RunStreaming(parser, out);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "refresh");
  CHECK((r.tools[0].arguments.empty() || r.tools[0].arguments == "{}"));
}

TEST_CASE("granite-20b-fc streaming: plain content yields no tool calls") {
  Granite20bFCToolParser parser;
  const std::string out = "This is a regular response without any tool calls.";
  Recon r = RunStreaming(parser, out);
  CHECK(r.tools.empty());
  CHECK(r.content == out);
}

TEST_CASE("granite-20b-fc streaming: malformed input never throws") {
  Granite20bFCToolParser parser;
  CHECK_NOTHROW(
      (void)RunStreaming(parser, "<function_call> {\"name\": \"func\", \"arguments\": {"));
}

// ─── Factory ─────────────────────────────────────────────────────────────────

TEST_CASE("granite-20b-fc: factory registration") {
  auto p = get_tool_parser("granite-20b-fc");
  REQUIRE(p != nullptr);
  auto req = empty_request();
  auto info = p->extract_tool_calls(
      "<function_call> {\"name\": \"f\", \"arguments\": {}}", req);
  CHECK(info.tools_called == true);
}
