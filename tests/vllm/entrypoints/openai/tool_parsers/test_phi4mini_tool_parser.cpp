// Tests for the "phi4_mini_json" tool-call parser (Phi4MiniJsonToolParser).
// Ported from: tests/tool_parsers/test_phi4mini_tool_parser.py @ e24d1b24, which
// drives the shared tests/tool_parsers/common_tests.py::ToolParserTests matrix.
// The upstream file has a SINGLE test class; we port its representative data
// (single / parallel / empty-args / no-tool / malformed) and ADD the required
// edge cases (parameters-alias, surrounding text, streaming-reconstruction).
//
// Deviations exercised here:
//   - arguments are re-serialized with nlohmann's COMPACT dump() (upstream
//     json.dumps adds ", "/": " spaces) - semantically identical JSON. Every
//     argument object below has stable key order, so the compact output is
//     deterministic.
//   - a malformed `functools[...]` body whose JSON does not parse still reports
//     tools_called=True with an EMPTY tool list (upstream's inner-except swallow;
//     this is the upstream xfail_nonstreaming "test_malformed_input").
//   - streaming is unimplemented upstream (returns None); we assert nullopt.
#include <doctest/doctest.h>

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/phi4_mini.h"

using namespace vllm::entrypoints::openai;

namespace {
ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }
}  // namespace

// ── no tools ─────────────────────────────────────────────────────────────────

TEST_CASE("phi4_mini: no tool calls is plain content") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      "This is a regular response without any tool calls.", req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "This is a regular response without any tool calls.");
}

// ── single tool (ported) ─────────────────────────────────────────────────────

TEST_CASE("phi4_mini: single tool with simple args") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}])", req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].id.size() > 16);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].function.arguments == R"({"city":"Tokyo"})");
  CHECK_FALSE(info.content.has_value());
}

// ── parallel tools (ported) ──────────────────────────────────────────────────

TEST_CASE("phi4_mini: parallel tool calls") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      "functools[\n"
      "  {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}},\n"
      "  {\"name\": \"get_time\", \"arguments\": {\"timezone\": \"Asia/Tokyo\"}}\n"
      "]",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].function.arguments == R"({"city":"Tokyo"})");
  CHECK(info.tool_calls[1].function.name == "get_time");
  CHECK(info.tool_calls[1].function.arguments == R"({"timezone":"Asia/Tokyo"})");
  CHECK_FALSE(info.content.has_value());
}

// ── EDGE: empty arguments ────────────────────────────────────────────────────

TEST_CASE("phi4_mini: empty arguments object") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(functools[{"name": "refresh", "arguments": {}}])", req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "refresh");
  CHECK(info.tool_calls[0].function.arguments == "{}");
}

// ── EDGE: "parameters" alias for arguments ───────────────────────────────────

TEST_CASE("phi4_mini: parameters alias is accepted") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(functools[{"name": "compute", "parameters": {"x": 1}}])", req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "compute");
  CHECK(info.tool_calls[0].function.arguments == R"({"x":1})");
}

// ── EDGE: surrounding text (content is still stripped to None) ────────────────

TEST_CASE("phi4_mini: surrounding text -> content None, tool extracted") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      "Let me check the weather for you.\n"
      R"(functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}])"
      "\nWould you like to know more?",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  // Phi-4 Mini drops surrounding content entirely (content=None).
  CHECK_FALSE(info.content.has_value());
}

// ── EDGE: malformed JSON body -> tools_called True, empty list (upstream bug) ─

TEST_CASE("phi4_mini: empty functools[] -> empty tool list, tools_called True") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  // `functools[]` matches the regex with an empty body -> json.loads("[]") == [].
  // The inner-except swallow leaves an empty array; tools_called stays True even
  // though nothing was extracted (upstream xfail_nonstreaming "test_malformed").
  auto info = parser.extract_tool_calls("functools[] This is just text", req);
  CHECK(info.tools_called == true);
  CHECK(info.tool_calls.empty());
}

TEST_CASE("phi4_mini: unterminated body (no closing bracket) -> content") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  // No closing `]` -> the `functools\[(.*?)\]` regex never matches -> plain text.
  const std::string out = R"(functools[{"name": "func", "arguments": {)";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// ── EDGE: missing name -> fall back to whole output as content ────────────────

TEST_CASE("phi4_mini: entry missing name -> content fallback") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  const std::string out = R"(functools[{"arguments": {"x": 1}}])";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// ── EDGE: no functools prefix -> plain content ───────────────────────────────

TEST_CASE("phi4_mini: no functools prefix (bare JSON) is content") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  const std::string out = R"([{"name": "x", "arguments": {}}])";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// ── EDGE: streaming reconstruction (unimplemented -> nullopt) ─────────────────

TEST_CASE("phi4_mini: streaming returns nullopt for every delta") {
  Phi4MiniJsonToolParser parser;
  auto req = empty_request();
  const std::string full =
      R"(functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}])";
  std::string previous;
  bool any_delta = false;
  for (std::size_t i = 0; i < full.size(); ++i) {
    const std::string delta = full.substr(i, 1);
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (dm.has_value()) any_delta = true;
  }
  CHECK_FALSE(any_delta);
}

// ── Factory registration ─────────────────────────────────────────────────────

TEST_CASE("Factory: get_tool_parser(\"phi4_mini_json\") works") {
  auto parser = get_tool_parser("phi4_mini_json");
  REQUIRE(parser != nullptr);
  auto req = empty_request();
  auto info = parser->extract_tool_calls(
      R"(functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}])", req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
}
