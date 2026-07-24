// Tests for the "ernie45" tool-call parser (Ernie45ToolParser).
// Ported from: tests/tool_parsers/test_ernie45_moe_tool_parser.py @ e24d1b24 -
// the SAME fixture inputs and expected names/arguments/content.
//
// PORTED-VS-UPSTREAM:
//   - Non-streaming: no-tools + all 3 extract cases (single, multiple,
//     content-before-with-</think>).
//   - Streaming: all 3 incremental cases (single, multiple, content-before).
//     Upstream drives the parser with a REAL ERNIE tokenizer + detokenize_
//     incrementally and gates on SPECIAL TOKEN IDS; that is not reproducible on
//     the text-only seam, so we REWORK to text (see ernie45.h) and feed deltas
//     that split the markers atomically the way the tokenizer would. The upstream
//     streaming test only checks the reconstructed TOOL CALLS (never content), so
//     the content-cleanup rework is not load-bearing here.
//   - Plus 4 ADDED edges: nested arguments, missing "arguments" key -> {},
//     whitespace-only content-before -> None, streaming single tool char-split.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/ernie45.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest EmptyRequest() { return ChatCompletionRequest{}; }

struct CollectedCall {
  std::string name;
  std::string arguments;
};

// Reconstruct tool calls from streamed deltas (upstream index-keyed collector).
std::vector<CollectedCall> Drive(Ernie45ToolParser& parser,
                                 const std::vector<std::string>& deltas) {
  std::map<int, CollectedCall> by_index;
  std::vector<int> order;
  std::string prev;
  ChatCompletionRequest req;
  for (const std::string& delta : deltas) {
    const std::string cur = prev + delta;
    auto dm = parser.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (!dm.has_value() || !dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      const int idx = tc.index;
      if (by_index.find(idx) == by_index.end()) {
        by_index[idx] = {tc.function.name.value_or(""),
                         tc.function.arguments.value_or("")};
        order.push_back(idx);
      } else {
        if (tc.function.name.has_value()) by_index[idx].name += *tc.function.name;
        if (tc.function.arguments.has_value())
          by_index[idx].arguments += *tc.function.arguments;
      }
    }
  }
  std::vector<CollectedCall> out;
  for (int idx : order) out.push_back(by_index[idx]);
  return out;
}

// Deltas that split each <tool_call> block into tokenizer-like atoms.
std::vector<std::string> ToolBlockDeltas(const std::string& body_json) {
  return {"<tool_call>", "\n" + body_json + "\n", "</tool_call>", "\n"};
}

}  // namespace

// ─── Non-streaming ──────────────────────────────────────────────────────────

TEST_CASE("ernie45: no tool calls is plain content") {
  Ernie45ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls("This is a test", req);
  CHECK_FALSE(r.tools_called);
  CHECK(r.tool_calls.empty());
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "This is a test");
}

TEST_CASE("ernie45: single tool call") {
  Ernie45ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      "<tool_call>\n{\"name\": \"get_current_temperature\", \"arguments\": "
      "{\"location\": \"Beijing\"}}\n</tool_call>\n",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_current_temperature");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "Beijing"}});
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("ernie45: multiple tool calls") {
  Ernie45ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      "<tool_call>\n{\"name\": \"get_current_temperature\", \"arguments\": "
      "{\"location\": \"Beijing\"}}\n</tool_call>\n<tool_call>\n{\"name\": "
      "\"get_temperature_unit\", \"arguments\": {\"location\": \"Guangzhou\", "
      "\"unit\": \"c\"}}\n</tool_call>\n",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_current_temperature");
  CHECK(r.tool_calls[1].function.name == "get_temperature_unit");
  CHECK(nlohmann::json::parse(r.tool_calls[1].function.arguments) ==
        nlohmann::json{{"location", "Guangzhou"}, {"unit", "c"}});
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("ernie45: tool call with content before (</think>)") {
  Ernie45ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      "I need to call two tools to handle these two issues separately.\n"
      "</think>\n\n<tool_call>\n{\"name\": \"get_current_temperature\", "
      "\"arguments\": {\"location\": \"Beijing\"}}\n</tool_call>\n<tool_call>\n"
      "{\"name\": \"get_temperature_unit\", \"arguments\": {\"location\": "
      "\"Guangzhou\", \"unit\": \"c\"}}\n</tool_call>\n",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 2);
  REQUIRE(r.content.has_value());
  CHECK(*r.content ==
        "I need to call two tools to handle these two issues separately.\n"
        "</think>");
}

// ─── Added edges ────────────────────────────────────────────────────────────

TEST_CASE("ernie45 edge: nested arguments") {
  Ernie45ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<tool_call>{"name": "f", "arguments": {"a": {"b": [1, 2]}}}</tool_call>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json::parse(R"({"a": {"b": [1, 2]}})"));
}

TEST_CASE("ernie45 edge: missing arguments key defaults to {}") {
  Ernie45ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<tool_call>{"name": "ping"}</tool_call>)", req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "ping");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json::object());
}

TEST_CASE("ernie45 edge: whitespace-only content before -> None") {
  Ernie45ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      "\n\n<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>", req);
  CHECK(r.tools_called);
  CHECK_FALSE(r.content.has_value());
}

// ─── Streaming ──────────────────────────────────────────────────────────────

TEST_CASE("ernie45 streaming: single tool call") {
  Ernie45ToolParser parser;
  const auto tc = Drive(parser, ToolBlockDeltas(
      R"({"name": "get_current_temperature", "arguments": {"location": "Beijing"}})"));
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "get_current_temperature");
  CHECK(nlohmann::json::parse(tc[0].arguments) ==
        nlohmann::json{{"location", "Beijing"}});
}

TEST_CASE("ernie45 streaming: multiple tool calls") {
  Ernie45ToolParser parser;
  std::vector<std::string> deltas = ToolBlockDeltas(
      R"({"name": "get_current_temperature", "arguments": {"location": "Beijing"}})");
  auto second = ToolBlockDeltas(
      R"({"name": "get_temperature_unit", "arguments": {"location": "Guangzhou", "unit": "c"}})");
  deltas.insert(deltas.end(), second.begin(), second.end());
  const auto tc = Drive(parser, deltas);
  REQUIRE(tc.size() == 2);
  CHECK(tc[0].name == "get_current_temperature");
  CHECK(tc[1].name == "get_temperature_unit");
  CHECK(nlohmann::json::parse(tc[1].arguments) ==
        nlohmann::json{{"location", "Guangzhou"}, {"unit", "c"}});
}

TEST_CASE("ernie45 streaming: content before then tool") {
  Ernie45ToolParser parser;
  std::vector<std::string> deltas = {
      "I need to call two tools", ".\n", "</think>", "\n\n"};
  auto block = ToolBlockDeltas(
      R"({"name": "get_current_temperature", "arguments": {"location": "Beijing"}})");
  deltas.insert(deltas.end(), block.begin(), block.end());
  const auto tc = Drive(parser, deltas);
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "get_current_temperature");
  CHECK(nlohmann::json::parse(tc[0].arguments) ==
        nlohmann::json{{"location", "Beijing"}});
}

TEST_CASE("ernie45 streaming edge: char-split single tool") {
  Ernie45ToolParser parser;
  const std::string body =
      R"({"name": "f", "arguments": {"a": 1}})";
  std::vector<std::string> deltas;
  deltas.emplace_back("<tool_call>");
  deltas.emplace_back("\n");
  for (char c : body) deltas.emplace_back(1, c);
  deltas.emplace_back("\n");
  deltas.emplace_back("</tool_call>");
  const auto tc = Drive(parser, deltas);
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "f");
  CHECK(nlohmann::json::parse(tc[0].arguments) == nlohmann::json{{"a", 1}});
}

// ─── Factory ────────────────────────────────────────────────────────────────

TEST_CASE("Factory: get_tool_parser(\"ernie45\") works") {
  auto parser = get_tool_parser("ernie45");
  REQUIRE(parser != nullptr);
  auto req = EmptyRequest();
  auto r = parser->extract_tool_calls(
      R"(<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>)", req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "f");
}
