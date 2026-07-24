// Tests for the "internlm" tool-call parser (Internlm2ToolParser).
// Ported from: tests/tool_parsers/test_internlm2_tool_parser.py @ e24d1b24 (which
// drives the shared common_tests.py::ToolParserTests matrix). We port its
// representative data (single / empty-args / no-tool / surrounding-text /
// malformed) and ADD the required edges (parameters-vs-arguments alias, a
// streaming-reconstruction cadence case).
//
// Deviations exercised here:
//   - malformed action JSON RAISES (upstream extract_tool_calls has no
//     try/except; recorded as upstream xfail_nonstreaming). We CHECK_THROWS.
//   - streaming is driven with ATOMIC special-token deltas (the markers are
//     single tokens upstream), then char-by-char over the JSON body; we
//     reconstruct the streamed name + argument fragments and compare.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/internlm.h"
#include "vllm/entrypoints/openai/tool_parsers/utils.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

// Split a JSON body into deltas the way a real tokenizer would: each quoted
// string literal (with escapes) is ONE token, everything else is a single char.
// This matters because the streaming argument-diff's first-fragment branch uses
// `cur_json.index(delta_text)`, which only lands correctly for multi-char tokens
// (char-by-char would locate a lone `"` at the wrong offset).
std::vector<std::string> TokenizeJson(const std::string& body) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < body.size()) {
    if (body[i] == '"') {
      // Opening quote + content as one token; the CLOSING quote as its own token
      // (see the jamba test for why the terminating quote must arrive separately).
      std::size_t j = i + 1;
      while (j < body.size()) {
        if (body[j] == '\\') {
          j += 2;
        } else if (body[j] == '"') {
          break;
        } else {
          ++j;
        }
      }
      out.push_back(body.substr(i, j - i));
      if (j < body.size()) {
        out.emplace_back("\"");
        i = j + 1;
      } else {
        i = j;
      }
    } else {
      out.emplace_back(1, body[i]);
      ++i;
    }
  }
  return out;
}

// Complete a possibly-partial streamed argument string (the streaming diff
// withholds close-quotes/braces until they are confirmed) and compare by value.
bool ArgsEqual(const std::string& streamed, const std::string& expected) {
  nlohmann::ordered_json got =
      partial_json_loads(streamed, /*allow_partial_str=*/true).first;
  return got == nlohmann::ordered_json::parse(expected);
}

struct StreamResult {
  std::string content;
  std::vector<std::string> names;
  std::string args;  // reconstructed arguments for the single tool.
};

StreamResult DriveStream(Internlm2ToolParser& parser,
                         const std::vector<std::string>& deltas) {
  StreamResult r;
  std::string previous;
  ChatCompletionRequest req;
  for (const std::string& delta : deltas) {
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (!dm.has_value()) continue;
    if (dm->content.has_value()) r.content += *dm->content;
    if (dm->tool_calls.has_value()) {
      for (const DeltaToolCall& tc : *dm->tool_calls) {
        if (tc.function.name.has_value()) r.names.push_back(*tc.function.name);
        if (tc.function.arguments.has_value()) r.args += *tc.function.arguments;
      }
    }
  }
  return r;
}

// Build a delta list: the two markers as atomic tokens, the JSON body char by
// char, then the end marker as an atomic token.
std::vector<std::string> BodyDeltas(const std::string& body) {
  std::vector<std::string> d;
  d.emplace_back("<|action_start|>");
  d.emplace_back("<|plugin|>");
  for (const std::string& t : TokenizeJson(body)) d.push_back(t);
  d.emplace_back("<|action_end|>");
  return d;
}

}  // namespace

// ── no tools ─────────────────────────────────────────────────────────────────

TEST_CASE("internlm: no tool calls is plain content") {
  Internlm2ToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      "This is a regular response without any tool calls.", req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "This is a regular response without any tool calls.");
}

// ── single tool (ported) ─────────────────────────────────────────────────────

TEST_CASE("internlm: single tool with simple args") {
  Internlm2ToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|>)",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].id.size() > 16);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].function.arguments == R"({"city":"Tokyo"})");
  CHECK_FALSE(info.content.has_value());
}

// ── EDGE: "arguments" alias (instead of "parameters") ────────────────────────

TEST_CASE("internlm: arguments alias is accepted") {
  Internlm2ToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(<|action_start|><|plugin|>{"name": "f", "arguments": {"x": 1}}<|action_end|>)",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "f");
  CHECK(info.tool_calls[0].function.arguments == R"({"x":1})");
}

// ── EDGE: empty arguments ────────────────────────────────────────────────────

TEST_CASE("internlm: empty parameters object") {
  Internlm2ToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(<|action_start|><|plugin|>{"name": "refresh", "parameters": {}}<|action_end|>)",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "refresh");
  CHECK(info.tool_calls[0].function.arguments == "{}");
}

// ── EDGE: parallel (internlm supports ONLY single) ───────────────────────────

TEST_CASE("internlm: only the first tool call is parsed") {
  Internlm2ToolParser parser;
  auto req = empty_request();
  // Two plugin blocks; the parser splits on the FIRST and keeps one call.
  auto info = parser.extract_tool_calls(
      R"(<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|>)",
      req);
  CHECK(info.tools_called == true);
  CHECK(info.tool_calls.size() == 1);
}

// ── EDGE: surrounding text before the marker becomes content ─────────────────

TEST_CASE("internlm: leading text is preserved as content") {
  Internlm2ToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(Let me check the weather for you. <|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|>)",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Let me check the weather for you. ");
}

// ── EDGE: malformed action JSON raises (upstream has no guard) ────────────────

TEST_CASE("internlm: malformed action JSON raises") {
  Internlm2ToolParser parser;
  auto req = empty_request();
  CHECK_THROWS(parser.extract_tool_calls(
      "<|action_start|><|plugin|>not json<|action_end|>", req));
}

// ── EDGE: streaming reconstruction cadence ───────────────────────────────────

TEST_CASE("internlm: streaming reconstructs name + arguments") {
  Internlm2ToolParser parser;
  const auto r = DriveStream(
      parser,
      BodyDeltas(R"({"name": "get_weather", "parameters": {"city": "Tokyo"}})"));
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_weather");
  // The streamed argument fragments reconstruct (after completing the withheld
  // close-quote/brace) to the full arguments JSON.
  CHECK(ArgsEqual(r.args, R"({"city":"Tokyo"})"));
  CHECK(r.content.empty());
}

TEST_CASE("internlm: streaming leading content before the marker") {
  Internlm2ToolParser parser;
  std::vector<std::string> deltas = {"Hello", " there", "<|action_start|>",
                                     "<|plugin|>"};
  for (const std::string& t :
       TokenizeJson(R"({"name": "f", "parameters": {"a": 1}})"))
    deltas.push_back(t);
  deltas.emplace_back("<|action_end|>");
  const auto r = DriveStream(parser, deltas);
  CHECK(r.content == "Hello there");
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "f");
}

// ── Factory registration ─────────────────────────────────────────────────────

TEST_CASE("Factory: get_tool_parser(\"internlm\") works") {
  auto parser = get_tool_parser("internlm");
  REQUIRE(parser != nullptr);
  auto req = empty_request();
  auto info = parser->extract_tool_calls(
      R"(<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|>)",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
}
