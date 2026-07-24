// Tests for the "granite" tool-call parser (GraniteToolParser).
// Ported from: tests/tool_parsers/test_granite_tool_parser.py @ e24d1b24
// (which drives tests/tool_parsers/common_tests.py::ToolParserTests). The Python
// suite runs each case in BOTH streaming and non-streaming via a real tokenizer;
// we have no tokenizer, so - exactly like test_tool_parsers.cpp for Hermes - we
// drive streaming with an explicit finest-grain (character-by-character) cadence
// and reconstruct the tool calls the way StreamingToolReconstructor does.
//
// xfail parity: the upstream config marks these streaming cases xfail(strict)
// (test_granite_tool_parser.py:72-82) and we therefore do NOT assert them:
//   - test_surrounding_text (marker not stripped from streamed content)
//   - test_malformed_input (streaming may fabricate a call from malformed JSON)
//   - test_streaming_reconstruction (content differs: marker leaks into content)
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/granite.h"

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

// Character-by-character streaming (the strictest cadence): the accumulated
// current_text grows one char at a time, mirroring token-by-token serving.
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

TEST_CASE("granite: no tool calls is plain content") {
  GraniteToolParser parser;
  const std::string out = "This is a regular response without any tool calls.";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("granite: single tool call, <|tool_call|> token prefix") {
  GraniteToolParser parser;
  const std::string out =
      "<|tool_call|> [{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Tokyo\"}}]";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].id.rfind("chatcmpl-tool-", 0) == 0);
  auto args = nlohmann::json::parse(info.tool_calls[0].function.arguments);
  CHECK(args["city"] == "Tokyo");
  // Granite strips content when tools are called.
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("granite: single tool call, <tool_call> string prefix") {
  GraniteToolParser parser;
  const std::string out =
      "<tool_call> [{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Tokyo\"}}]";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
}

TEST_CASE("granite: parallel tool calls") {
  GraniteToolParser parser;
  const std::string out =
      "<|tool_call|> [\n"
      "  {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}},\n"
      "  {\"name\": \"get_time\", \"arguments\": {\"timezone\": \"Asia/Tokyo\"}}\n"
      "]";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[1].function.name == "get_time");
  CHECK(info.tool_calls[0].id != info.tool_calls[1].id);
}

TEST_CASE("granite: various data types round-trip through arguments") {
  GraniteToolParser parser;
  const std::string out =
      "<tool_call> [{\"name\": \"test_function\", \"arguments\": {"
      "\"string_field\": \"hello\", \"int_field\": 42, \"float_field\": 3.14, "
      "\"bool_field\": true, \"null_field\": null, "
      "\"array_field\": [\"a\", \"b\"], \"object_field\": {\"nested\": \"value\"}, "
      "\"empty_array\": [], \"empty_object\": {}}}]";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = nlohmann::json::parse(info.tool_calls[0].function.arguments);
  CHECK(args["string_field"].is_string());
  CHECK(args["int_field"].is_number_integer());
  CHECK(args["float_field"].is_number_float());
  CHECK(args["bool_field"].is_boolean());
  CHECK(args["null_field"].is_null());
  CHECK(args["array_field"].is_array());
  CHECK(args["object_field"].is_object());
}

TEST_CASE("granite: empty arguments") {
  GraniteToolParser parser;
  const std::string out = "<|tool_call|> [{\"name\": \"refresh\", \"arguments\": {}}]";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "refresh");
  CHECK(info.tool_calls[0].function.arguments == "{}");
}

TEST_CASE("granite: malformed inputs never throw (non-streaming)") {
  const std::vector<std::string> malformed = {
      "<|tool_call|> [{\"name\": \"func\", \"arguments\": {",
      "<|tool_call|> {\"name\": \"func\", \"arguments\": {}}",  // not an array
      "Some text [{\"name\": \"func\"}]",  // JSON but not tool-call format
  };
  for (const std::string& in : malformed) {
    GraniteToolParser parser;
    auto req = empty_request();
    // Must not throw; graceful fallback.
    CHECK_NOTHROW((void)parser.extract_tool_calls(in, req));
  }
}

TEST_CASE("granite: empty output falls through as content") {
  GraniteToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls("", req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "");
}

// ─── Streaming ───────────────────────────────────────────────────────────────

TEST_CASE("granite streaming: single tool call reconstructs (token prefix)") {
  GraniteToolParser parser;
  const std::string out =
      "<|tool_call|> [{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Tokyo\"}}]";
  Recon r = RunStreaming(parser, out);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_weather");
  CHECK(r.tools[0].id.rfind("chatcmpl-tool-", 0) == 0);
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Tokyo");
}

TEST_CASE("granite streaming: single tool call reconstructs (string prefix)") {
  GraniteToolParser parser;
  const std::string out =
      "<tool_call> [{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Tokyo\"}}]";
  Recon r = RunStreaming(parser, out);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_weather");
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Tokyo");
}

TEST_CASE("granite streaming: parallel tool calls reconstruct") {
  GraniteToolParser parser;
  const std::string out =
      "<|tool_call|> [{\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Tokyo\"}}, {\"name\": \"get_time\", \"arguments\": {\"timezone\": "
      "\"Asia/Tokyo\"}}]";
  Recon r = RunStreaming(parser, out);
  REQUIRE(r.tools.size() == 2);
  CHECK(r.tools[0].name == "get_weather");
  CHECK(r.tools[1].name == "get_time");
  CHECK(nlohmann::json::parse(r.tools[0].arguments)["city"] == "Tokyo");
  CHECK(nlohmann::json::parse(r.tools[1].arguments)["timezone"] == "Asia/Tokyo");
}

TEST_CASE("granite streaming: empty arguments reconstructs") {
  GraniteToolParser parser;
  const std::string out = "<|tool_call|> [{\"name\": \"refresh\", \"arguments\": {}}]";
  Recon r = RunStreaming(parser, out);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "refresh");
  CHECK((r.tools[0].arguments.empty() || r.tools[0].arguments == "{}"));
}

TEST_CASE("granite streaming: various data types reconstruct") {
  GraniteToolParser parser;
  const std::string out =
      "<tool_call> [{\"name\": \"test_function\", \"arguments\": {"
      "\"string_field\": \"hello\", \"int_field\": 42, \"bool_field\": true, "
      "\"array_field\": [1, 2, 3], \"object_field\": {\"nested\": \"value\"}}}]";
  Recon r = RunStreaming(parser, out);
  REQUIRE(r.tools.size() == 1);
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["string_field"] == "hello");
  CHECK(args["int_field"] == 42);
  CHECK(args["bool_field"] == true);
  CHECK(args["array_field"].size() == 3);
}

TEST_CASE("granite streaming: plain content yields no tool calls") {
  GraniteToolParser parser;
  const std::string out = "This is a regular response without any tool calls.";
  Recon r = RunStreaming(parser, out);
  CHECK(r.tools.empty());
  CHECK(r.content == out);
}

TEST_CASE("granite streaming: malformed input never throws") {
  const std::vector<std::string> malformed = {
      "<|tool_call|> [{\"name\": \"func\", \"arguments\": {",
      "<|tool_call|> {\"name\": \"func\"}",
  };
  for (const std::string& in : malformed) {
    GraniteToolParser parser;
    CHECK_NOTHROW((void)RunStreaming(parser, in));
  }
}

// ─── Factory ─────────────────────────────────────────────────────────────────

TEST_CASE("granite: factory registration") {
  auto p = get_tool_parser("granite");
  REQUIRE(p != nullptr);
  auto req = empty_request();
  auto info = p->extract_tool_calls(
      "<|tool_call|> [{\"name\": \"f\", \"arguments\": {}}]", req);
  CHECK(info.tools_called == true);
}
