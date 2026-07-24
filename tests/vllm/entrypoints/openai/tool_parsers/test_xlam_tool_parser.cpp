// Tests for the "xlam" tool-call parser (xLAMToolParser).
// Ported from: tests/tool_parsers/test_xlam_tool_parser.py @ e24d1b24 - the SAME
// fixture inputs and expected names/arguments/content across the five envelope
// forms (bare list / <think> tail / ```json fence / [TOOL_CALLS] / <tool_call>
// tags), the list-structure case, the preprocess_model_output splitter, the
// streaming-incremental cadence, and the non-ASCII (ensure_ascii=False) checks.
//
// Deviations:
//   - streaming is driven CHARACTER-BY-CHARACTER (upstream feeds token-by-token
//     via a real tokenizer; we have none). The cadence assertions are the same:
//     a name-first header chunk (id + name + type, args empty), then incremental
//     argument chunks that reconstruct the full arguments JSON.
//   - the upstream `test_streaming_with_list_structure` unit test pokes the
//     internal `current_tools_sent` attribute (a Python white-box hack that is
//     NOT part of the serving path); it is intentionally NOT ported.
//   - arguments re-serialized with nlohmann; JSON compared by value (whitespace-
//     insensitive) exactly as the upstream test does (json.loads == json.loads).
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/xlam.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

const char* kWeatherArgs = R"({"city":"Dallas","state":"TX","unit":"fahrenheit"})";

struct StreamCapture {
  std::string content;
  std::vector<std::string> names;
  std::vector<std::string> ids_by_index;
  std::vector<std::string> args_by_index;
  bool header_found = false;     // a chunk with id + name (+ type function).
  std::string header_name;
  int header_index = -1;
  int arg_chunk_count_idx0 = 0;  // non-empty arg chunks for index 0.
  int total_chunks = 0;
};

StreamCapture DriveCharByChar(xLAMToolParser& parser, const std::string& out) {
  StreamCapture c;
  std::string previous;
  ChatCompletionRequest req;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::string delta = out.substr(i, 1);
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (!dm.has_value()) continue;
    ++c.total_chunks;
    if (dm->content.has_value()) c.content += *dm->content;
    if (!dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      const int idx = tc.index;
      while (static_cast<int>(c.args_by_index.size()) <= idx) {
        c.args_by_index.emplace_back();
        c.ids_by_index.emplace_back();
      }
      if (tc.id.has_value() && c.ids_by_index[idx].empty())
        c.ids_by_index[idx] = *tc.id;
      if (tc.function.name.has_value()) {
        c.names.push_back(*tc.function.name);
        if (tc.id.has_value() && !c.header_found) {
          c.header_found = true;
          c.header_name = *tc.function.name;
          c.header_index = idx;
        }
      }
      if (tc.function.arguments.has_value()) {
        c.args_by_index[idx] += *tc.function.arguments;
        if (idx == 0 && !tc.function.arguments->empty())
          ++c.arg_chunk_count_idx0;
      }
    }
  }
  return c;
}

}  // namespace

// ═══ NON-STREAMING: no tools ═════════════════════════════════════════════════

TEST_CASE("xlam: no tool calls is plain content") {
  xLAMToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls("This is a test", req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "This is a test");
}

// ═══ NON-STREAMING: the five envelope forms ══════════════════════════════════

TEST_CASE("xlam: parallel bare JSON list") {
  xLAMToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}, {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].id.size() > 16);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments) ==
        nlohmann::json::parse(kWeatherArgs));
  CHECK(info.tool_calls[1].function.name == "get_current_weather");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("xlam: single tool after </think> block") {
  xLAMToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(<think>I'll help you with that.</think>[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments) ==
        nlohmann::json::parse(kWeatherArgs));
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "<think>I'll help you with that.</think>");
}

TEST_CASE("xlam: single tool in ```json fence") {
  xLAMToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      "I'll help you with that.\n```json\n"
      R"([{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}])"
      "\n```",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "I'll help you with that.");
}

TEST_CASE("xlam: single tool with [TOOL_CALLS] tag") {
  xLAMToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(I'll check the weather for you.[TOOL_CALLS][{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "I'll check the weather for you.");
}

TEST_CASE("xlam: single tool with <tool_call> xml tags") {
  xLAMToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(I'll help you check the weather.<tool_call>[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]</tool_call>)",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "I'll help you check the weather.");
}

// ═══ NON-STREAMING: list-structure (content None) ════════════════════════════

TEST_CASE("xlam: list-structured tool call, no content") {
  xLAMToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments) ==
        nlohmann::json::parse(R"({"city":"Seattle","state":"WA","unit":"celsius"})"));
  CHECK_FALSE(info.content.has_value());
}

// ═══ preprocess_model_output ═════════════════════════════════════════════════

TEST_CASE("xlam preprocess: bare list -> (None, whole)") {
  xLAMToolParser parser;
  auto [content, potential] = parser.preprocess_model_output(
      R"([{"name": "get_current_weather", "arguments": {"city": "Seattle"}}])");
  CHECK_FALSE(content.has_value());
  REQUIRE(potential.has_value());
  CHECK(*potential ==
        R"([{"name": "get_current_weather", "arguments": {"city": "Seattle"}}])");
}

TEST_CASE("xlam preprocess: thinking tag") {
  xLAMToolParser parser;
  auto [content, potential] = parser.preprocess_model_output(
      R"(<think>I'll help you with that.</think>[{"name": "get_current_weather", "arguments": {"city": "Seattle"}}])");
  REQUIRE(content.has_value());
  CHECK(*content == "<think>I'll help you with that.</think>");
  REQUIRE(potential.has_value());
  CHECK(*potential ==
        R"([{"name": "get_current_weather", "arguments": {"city": "Seattle"}}])");
}

TEST_CASE("xlam preprocess: json code block") {
  xLAMToolParser parser;
  auto [content, potential] = parser.preprocess_model_output(
      "I'll help you with that.\n```json\n"
      R"([{"name": "get_current_weather", "arguments": {"city": "Seattle"}}])"
      "\n```");
  REQUIRE(content.has_value());
  CHECK(*content == "I'll help you with that.");
  REQUIRE(potential.has_value());
  CHECK(potential->find("get_current_weather") != std::string::npos);
}

TEST_CASE("xlam preprocess: no tool calls -> (whole, None)") {
  xLAMToolParser parser;
  auto [content, potential] =
      parser.preprocess_model_output("I'll help you with that.");
  REQUIRE(content.has_value());
  CHECK(*content == "I'll help you with that.");
  CHECK_FALSE(potential.has_value());
}

// ═══ STREAMING: incremental cadence (the five forms) ═════════════════════════

void CheckStreamingIncremental(const std::string& model_output,
                               const std::string& first_name,
                               const std::string& first_args_json) {
  xLAMToolParser parser;
  const auto c = DriveCharByChar(parser, model_output);
  CHECK(c.total_chunks >= 3);
  REQUIRE(c.header_found);
  CHECK(c.header_name == first_name);
  CHECK(c.header_index == 0);
  REQUIRE(!c.ids_by_index.empty());
  CHECK(!c.ids_by_index[0].empty());
  // Arguments must stream in more than one chunk and reconstruct to valid JSON.
  CHECK(c.arg_chunk_count_idx0 > 1);
  REQUIRE(nlohmann::json::accept(c.args_by_index[0]));
  CHECK(nlohmann::json::parse(c.args_by_index[0]) ==
        nlohmann::json::parse(first_args_json));
}

TEST_CASE("xlam streaming: parallel bare list") {
  CheckStreamingIncremental(
      R"([{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}, {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}])",
      "get_current_weather", kWeatherArgs);
}

TEST_CASE("xlam streaming: think tag") {
  CheckStreamingIncremental(
      R"(<think>I'll help you with that.</think>[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}])",
      "get_current_weather", kWeatherArgs);
}

TEST_CASE("xlam streaming: json code block") {
  CheckStreamingIncremental(
      "```json\n"
      R"([{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}])"
      "\n```",
      "get_current_weather", kWeatherArgs);
}

TEST_CASE("xlam streaming: [TOOL_CALLS] tag") {
  CheckStreamingIncremental(
      R"([TOOL_CALLS][{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}])",
      "get_current_weather", kWeatherArgs);
}

TEST_CASE("xlam streaming: <tool_call> xml tags") {
  CheckStreamingIncremental(
      R"(I can help with that.<tool_call>[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]</tool_call>)",
      "get_current_weather", kWeatherArgs);
}

// ═══ non-ASCII (ensure_ascii=False) ══════════════════════════════════════════

TEST_CASE("xlam non-ascii: non-streaming keeps UTF-8") {
  xLAMToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([{"name": "get_current_weather", "arguments": {"city": "北京"}}, {"name": "get_current_weather", "arguments": {"city": "上海"}}])",
      req);
  REQUIRE(info.tool_calls.size() == 2);
  std::string joined;
  for (const auto& tc : info.tool_calls) joined += tc.function.arguments;
  CHECK(joined.find("北京") != std::string::npos);
  CHECK(joined.find("\\u") == std::string::npos);
}

TEST_CASE("xlam non-ascii: streaming keeps UTF-8") {
  xLAMToolParser parser;
  const auto c = DriveCharByChar(
      parser,
      R"([{"name": "get_current_weather", "arguments": {"city": "北京"}}, {"name": "get_current_weather", "arguments": {"city": "上海"}}])");
  std::string joined;
  for (const auto& a : c.args_by_index) joined += a;
  CHECK(joined.find("北京") != std::string::npos);
  CHECK(joined.find("\\u") == std::string::npos);
}

// ═══ Factory ═════════════════════════════════════════════════════════════════

TEST_CASE("Factory: get_tool_parser(\"xlam\") works") {
  auto parser = get_tool_parser("xlam");
  REQUIRE(parser != nullptr);
  auto req = empty_request();
  auto info = parser->extract_tool_calls(
      R"([{"name": "get_current_weather", "arguments": {"city": "Seattle"}}])", req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
}
