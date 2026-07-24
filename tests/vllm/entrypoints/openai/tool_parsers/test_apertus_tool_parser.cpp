// Tests for the "apertus" tool-call parser (ApertusToolParser).
// Ported from: tests/tool_parsers/test_apertus_tool_parser.py @ e24d1b24 - ALL 19
// upstream cases (8 non-streaming TestExtractToolCalls + 11 streaming
// TestStreamingExtraction, including the MTP / char-by-char / empty-delta edges).
//
// The streaming token-id spans are dropped (text-only seam, see apertus.h); the
// driver mirrors upstream _simulate_streaming + _collect_tool_calls (which
// accumulates name/arguments per index and DOES allow multiple tool-call deltas
// per message) + _collect_content. Arguments compared as PARSED JSON.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/apertus.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest EmptyRequest() { return ChatCompletionRequest{}; }

struct CollectedCall {
  std::string name;
  std::string arguments;
};

struct StreamOut {
  std::map<int, CollectedCall> calls;
  std::vector<int> order;
  std::string content;
};

StreamOut Drive(ApertusToolParser& parser, const std::vector<std::string>& chunks) {
  StreamOut out;
  std::string prev;
  ChatCompletionRequest req;
  for (const std::string& chunk : chunks) {
    const std::string cur = prev + chunk;
    auto dm = parser.extract_tool_calls_streaming(prev, cur, chunk, req);
    prev = cur;
    if (!dm.has_value()) continue;
    if (dm->content.has_value()) out.content += *dm->content;
    if (!dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      const int idx = tc.index;
      if (out.calls.find(idx) == out.calls.end()) {
        out.calls[idx] = {tc.function.name.value_or(""),
                          tc.function.arguments.value_or("")};
        out.order.push_back(idx);
      } else {
        if (tc.function.name.has_value()) out.calls[idx].name += *tc.function.name;
        if (tc.function.arguments.has_value())
          out.calls[idx].arguments += *tc.function.arguments;
      }
    }
  }
  return out;
}

std::string Strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && static_cast<unsigned char>(s[b]) <= ' ') ++b;
  while (e > b && static_cast<unsigned char>(s[e - 1]) <= ' ') --e;
  return s.substr(b, e - b);
}

std::vector<std::string> CharChunks(const std::string& s) {
  std::vector<std::string> out;
  for (char c : s) out.emplace_back(1, c);
  return out;
}

}  // namespace

// ─── Non-streaming ──────────────────────────────────────────────────────────

TEST_CASE("apertus: no tool calls") {
  ApertusToolParser parser;
  auto req = EmptyRequest();
  const std::string out = "Hello, how can I help you today?";
  auto r = parser.extract_tool_calls(out, req);
  CHECK_FALSE(r.tools_called);
  CHECK(r.tool_calls.empty());
  REQUIRE(r.content.has_value());
  CHECK(*r.content == out);
}

TEST_CASE("apertus: single tool call") {
  ApertusToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<|tools_prefix|>[{"get_weather": {"location": "London"}}]<|tools_suffix|>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "London"}});
}

TEST_CASE("apertus: multiple arguments") {
  ApertusToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<|tools_prefix|>[{"get_weather": {"location": "San Francisco", "unit": "celsius"}}]<|tools_suffix|>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "San Francisco"}, {"unit", "celsius"}});
}

TEST_CASE("apertus: text before tool call") {
  ApertusToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(Let me check the weather for you. <|tools_prefix|>[{"get_weather": {"location": "Paris"}}]<|tools_suffix|>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "Let me check the weather for you.");
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
}

TEST_CASE("apertus: multiple tool calls") {
  ApertusToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<|tools_prefix|>[{"get_weather": {"location": "London"}}, {"get_time": {"location": "London"}}]<|tools_suffix|>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(r.tool_calls[1].function.name == "get_time");
}

TEST_CASE("apertus: nested arguments") {
  ApertusToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<|tools_prefix|>[{"complex_function": {"nested": {"inner": "value"}, "list": ["a", "b"]}}]<|tools_suffix|>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "complex_function");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json::parse(
            R"({"nested": {"inner": "value"}, "list": ["a", "b"]})"));
}

TEST_CASE("apertus: incomplete tool call (truncated)") {
  ApertusToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<|tools_prefix|>[{"get_weather": {"location": "London"})", req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "London"}});
}

TEST_CASE("apertus: missing tool suffix") {
  ApertusToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<|tools_prefix|>[{"get_weather": {"location": "San Francisco", "unit": "celsius"}}])",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "San Francisco"}, {"unit", "celsius"}});
}

// ─── Streaming ──────────────────────────────────────────────────────────────

TEST_CASE("apertus streaming: basic single tool") {
  ApertusToolParser parser;
  auto r = Drive(parser, {"<|tools_prefix|>", R"([{"get_weather": )",
                          R"({"location": "Paris, )", R"(France"}}])",
                          "<|tools_suffix|>"});
  REQUIRE(r.order.size() == 1);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) ==
        nlohmann::json{{"location", "Paris, France"}});
}

TEST_CASE("apertus streaming: missing tool suffix") {
  ApertusToolParser parser;
  auto r = Drive(parser, {"<|tools_prefix|>", R"([{"get_weather": )",
                          R"({"location": "Paris, )", R"(France"}}])"});
  REQUIRE(r.order.size() == 1);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) ==
        nlohmann::json{{"location", "Paris, France"}});
}

TEST_CASE("apertus streaming: partial-tag buffering, missing suffix") {
  ApertusToolParser parser;
  auto r = Drive(parser,
                 {"Content", "<|tools_", "prefix|>", R"([{"f": )", R"({"a": 1}}])"});
  CHECK(r.content.find("Content") != std::string::npos);
  CHECK(r.content.find("<|tools_prefix|>") == std::string::npos);
  CHECK(r.content.find("<|tools_suffix|>") == std::string::npos);
  REQUIRE(r.order.size() == 1);
  CHECK(r.calls[0].name == "f");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) == nlohmann::json{{"a", 1}});
}

TEST_CASE("apertus streaming: multi tool") {
  ApertusToolParser parser;
  auto r = Drive(parser, {"<|tools_prefix|>",
                          R"([{"get_weather": {"location": "Tokyo"}})",
                          R"(, {"get_time": {"location": "Tokyo"}}])",
                          "<|tools_suffix|>"});
  REQUIRE(r.order.size() == 2);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) ==
        nlohmann::json{{"location", "Tokyo"}});
  CHECK(r.calls[1].name == "get_time");
  CHECK(nlohmann::json::parse(r.calls[1].arguments) ==
        nlohmann::json{{"location", "Tokyo"}});
}

TEST_CASE("apertus streaming: text before tool call") {
  ApertusToolParser parser;
  auto r = Drive(parser, {"Let me check ", "the weather. ", "<|tools_prefix|>",
                          R"([{"get_weather": {"location": "London"}}])",
                          "<|tools_suffix|>"});
  CHECK(Strip(r.content) == "Let me check the weather.");
  REQUIRE(r.order.size() == 1);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) ==
        nlohmann::json{{"location", "London"}});
}

TEST_CASE("apertus streaming: partial-tag buffering (prefix + suffix)") {
  ApertusToolParser parser;
  auto r = Drive(parser, {"Content", "<|tools_", "prefix|>", R"([{"f": {"a": 1}}])",
                          "<|tools_suf", "fix|>"});
  CHECK(r.content.find("Content") != std::string::npos);
  CHECK(r.content.find("<|tools_prefix|>") == std::string::npos);
  CHECK(r.content.find("<|tools_suffix|>") == std::string::npos);
  REQUIRE(r.order.size() == 1);
  CHECK(r.calls[0].name == "f");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) == nlohmann::json{{"a", 1}});
}

TEST_CASE("apertus streaming: MTP massive single chunk") {
  ApertusToolParser parser;
  auto r = Drive(
      parser,
      {R"(Sure! <|tools_prefix|>[{"get_weather": {"location": "London"}}]<|tools_suffix|>)"});
  CHECK(r.content.find("Sure! ") != std::string::npos);
  REQUIRE(r.order.size() == 1);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) ==
        nlohmann::json{{"location", "London"}});
}

TEST_CASE("apertus streaming: MTP multiple tools burst") {
  ApertusToolParser parser;
  auto r = Drive(
      parser,
      {R"(<|tools_prefix|>[{"get_weather": {"location": "London"}}, {"get_time": {"location": "Paris"}}]<|tools_suffix|>)"});
  REQUIRE(r.order.size() == 2);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) ==
        nlohmann::json{{"location", "London"}});
  CHECK(r.calls[1].name == "get_time");
  CHECK(nlohmann::json::parse(r.calls[1].arguments) ==
        nlohmann::json{{"location", "Paris"}});
}

TEST_CASE("apertus streaming: MTP skip and catch up") {
  ApertusToolParser parser;
  auto r = Drive(parser, {R"(<|tools_prefix|>[{"t1": {"a": 1})",
                          R"(}, {"t2": {"b": 2}}, {"t3": {"c": 3)",
                          R"(}}]<|tools_suffix|>)"});
  REQUIRE(r.order.size() == 3);
  CHECK(r.calls[0].name == "t1");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) == nlohmann::json{{"a", 1}});
  CHECK(r.calls[1].name == "t2");
  CHECK(nlohmann::json::parse(r.calls[1].arguments) == nlohmann::json{{"b", 2}});
  CHECK(r.calls[2].name == "t3");
  CHECK(nlohmann::json::parse(r.calls[2].arguments) == nlohmann::json{{"c", 3}});
}

TEST_CASE("apertus streaming: character by character") {
  ApertusToolParser parser;
  auto r = Drive(parser, CharChunks(
      R"(Hi <|tools_prefix|>[{"get_weather": {"location": "London"}}]<|tools_suffix|> )"));
  CHECK(r.content.find("Hi") != std::string::npos);
  REQUIRE(r.order.size() == 1);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) ==
        nlohmann::json{{"location", "London"}});
}

TEST_CASE("apertus streaming: empty deltas interspersed") {
  ApertusToolParser parser;
  auto r = Drive(parser, {"Wait", "", "<|tools_prefix|>", "", R"([{"get_weather": )",
                          "", R"({"location": "London"}}])", "<|tools_suffix|>"});
  CHECK(r.content == "Wait");
  REQUIRE(r.order.size() == 1);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.calls[0].arguments) ==
        nlohmann::json{{"location", "London"}});
}

// ─── Factory ────────────────────────────────────────────────────────────────

TEST_CASE("Factory: get_tool_parser(\"apertus\") works") {
  auto parser = get_tool_parser("apertus");
  REQUIRE(parser != nullptr);
  auto req = EmptyRequest();
  auto r = parser->extract_tool_calls(
      R"(<|tools_prefix|>[{"f": {"a": 1}}]<|tools_suffix|>)", req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "f");
}
