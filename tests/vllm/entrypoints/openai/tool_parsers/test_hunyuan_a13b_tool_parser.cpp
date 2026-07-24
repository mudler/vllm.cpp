// Tests for the "hunyuan_a13b" tool-call parser (HunyuanA13BToolParser).
// Ported from: tests/tool_parsers/test_hunyuan_a13b_tool_parser.py @ e24d1b24 -
// the SAME fixture inputs and expected names/arguments/content.
//
// PORTED-VS-UPSTREAM:
//   - Non-streaming: all 6 extract cases (no-call, single, parallel, content
//     before, content after stripped, deep-nested) + the non-ASCII case.
//   - Streaming: the 3 NON-xfail cases. Upstream's 4th streaming case is
//     pytest.mark.xfail ("stream parsing not support nested json yet") - the
//     regex-based streaming genuinely cannot emit a 3-level-nested arg object, so
//     we OMIT it exactly as upstream skips it (documented deviation, not a gap).
//   - Plus 4 ADDED edges: <think>-guarded block ignored, invalid-JSON block
//     falls through to content, the a13b "助手：" content prefix strip, empty-args
//     tool call, and an empty-args streaming edge.
//
// The streaming token-id spans are dropped (text-only seam, see hunyuan_a13b.h);
// deltas are fed exactly as upstream lists them and reconstructed with the same
// lenient per-index collector the upstream StreamingToolReconstructor uses when
// assert_one_tool_per_delta=False. Arguments compared as PARSED JSON.
#include <doctest/doctest.h>

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/hunyuan_a13b.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest EmptyRequest() { return ChatCompletionRequest{}; }

struct CollectedCall {
  std::string name;
  std::string arguments;
};

std::vector<CollectedCall> CollectStreaming(HunyuanA13BToolParser& parser,
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
  std::sort(order.begin(), order.end());
  for (int idx : order) out.push_back(by_index[idx]);
  return out;
}

}  // namespace

// ─── Non-streaming ──────────────────────────────────────────────────────────

TEST_CASE("hunyuan_a13b: no tool call is plain content") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls("How can I help you today?", req);
  CHECK_FALSE(r.tools_called);
  CHECK(r.tool_calls.empty());
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "How can I help you today?");
}

TEST_CASE("hunyuan_a13b: single tool, no content") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<tool_calls>[{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}]</tool_calls>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "San Francisco"}, {"metric", "celsius"}});
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("hunyuan_a13b: multiple tool calls") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<tool_calls>[{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}, {"name": "register_user", "arguments": {"name": "John Doe", "age": 37, "address": {"city": "San Francisco", "state": "CA"}, "role": null, "passed_test": true, "aliases": ["John", "Johnny"]}}]</tool_calls>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(r.tool_calls[1].function.name == "register_user");
  CHECK(nlohmann::json::parse(r.tool_calls[1].function.arguments) ==
        nlohmann::json::parse(
            R"({"name": "John Doe", "age": 37, "address": {"city": "San Francisco", "state": "CA"}, "role": null, "passed_test": true, "aliases": ["John", "Johnny"]})"));
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("hunyuan_a13b: content before tool call") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(I will call the tool now. <tool_calls>[{"name": "get_weather", "arguments": {"city": "Boston"}}]</tool_calls>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "I will call the tool now. ");
}

TEST_CASE("hunyuan_a13b: content after tool call is dropped") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      "<tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Seattle\"}}]</tool_calls>\nThank you!",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("hunyuan_a13b: deeply nested arguments") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<tool_calls>[{"name": "complex_tool", "arguments": {"level1": {"level2": {"level3": {"value": 123}}}}}]</tool_calls>)",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json::parse(
            R"({"level1": {"level2": {"level3": {"value": 123}}}})"));
}

TEST_CASE("hunyuan_a13b: non-ascii arguments kept as UTF-8") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      "<tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"\xE5\x8C\x97\xE4\xBA\xAC\"}}]</tool_calls>",
      req);
  REQUIRE(r.tool_calls.size() == 1);
  const std::string& args = r.tool_calls[0].function.arguments;
  CHECK(args.find("\xE5\x8C\x97\xE4\xBA\xAC") != std::string::npos);  // 北京
  CHECK(args.find("\\u") == std::string::npos);
}

// ─── Added edges ────────────────────────────────────────────────────────────

TEST_CASE("hunyuan_a13b edge: block inside <think> is ignored") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  const std::string out =
      R"(<think><tool_calls>[{"name": "f", "arguments": {}}]</tool_calls></think>done)";
  auto r = parser.extract_tool_calls(out, req);
  CHECK_FALSE(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == out);
}

TEST_CASE("hunyuan_a13b edge: invalid JSON block falls through to content") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  const std::string out = "<tool_calls>not valid json</tool_calls>";
  auto r = parser.extract_tool_calls(out, req);
  CHECK_FALSE(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == out);
}

TEST_CASE("hunyuan_a13b edge: chat-template 助手 prefix stripped when no call") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls("\xE5\x8A\xA9\xE6\x89\x8B\xEF\xBC\x9AHello",
                                     req);  // "助手：Hello"
  CHECK_FALSE(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "Hello");
}

TEST_CASE("hunyuan_a13b edge: empty arguments object") {
  HunyuanA13BToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(
      R"(<tool_calls>[{"name": "ping", "arguments": {}}]</tool_calls>)", req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "ping");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json::object());
}

// ─── Streaming ──────────────────────────────────────────────────────────────

TEST_CASE("hunyuan_a13b streaming: single tool (name/args split)") {
  HunyuanA13BToolParser parser;
  const auto tc = CollectStreaming(
      parser, {R"(<tool_calls>[{"name": "get_weather", )",
               R"("arguments": {"city": "San Francisco", )",
               R"("metric": "celsius"}}])", "</tool_calls>"});
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "get_weather");
  CHECK(nlohmann::json::parse(tc[0].arguments) ==
        nlohmann::json{{"city", "San Francisco"}, {"metric", "celsius"}});
}

TEST_CASE("hunyuan_a13b streaming: fine-grained deltas") {
  HunyuanA13BToolParser parser;
  const auto tc = CollectStreaming(
      parser, {R"(<tool_calls>[{"name":)", R"( "get_weather",)",
               R"( "arguments":)", R"( {"city": "Boston"})", "}]",
               "</tool_calls>"});
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "get_weather");
  CHECK(nlohmann::json::parse(tc[0].arguments) ==
        nlohmann::json{{"city", "Boston"}});
}

TEST_CASE("hunyuan_a13b streaming: leading empty delta + trailing answer tag") {
  HunyuanA13BToolParser parser;
  const auto tc = CollectStreaming(
      parser, {"", R"(<tool_calls>[{"name":)", R"( "get_weather",)",
               R"( "arguments":)", R"( {"city": "Boston"})", "}]",
               "</tool_calls>", "\n</answer>"});
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "get_weather");
  CHECK(nlohmann::json::parse(tc[0].arguments) ==
        nlohmann::json{{"city", "Boston"}});
}

TEST_CASE("hunyuan_a13b streaming edge: empty arguments object") {
  HunyuanA13BToolParser parser;
  const auto tc = CollectStreaming(
      parser, {R"(<tool_calls>[{"name": "ping", )", R"("arguments": {})", "}]",
               "</tool_calls>"});
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "ping");
  CHECK(nlohmann::json::parse(tc[0].arguments) == nlohmann::json::object());
}

// ─── Factory ────────────────────────────────────────────────────────────────

TEST_CASE("Factory: get_tool_parser(\"hunyuan_a13b\") works") {
  auto parser = get_tool_parser("hunyuan_a13b");
  REQUIRE(parser != nullptr);
  auto req = EmptyRequest();
  auto r = parser->extract_tool_calls(
      R"(<tool_calls>[{"name": "f", "arguments": {"a": 1}}]</tool_calls>)", req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "f");
}
