// Ported from: vllm/tests/tool_parsers/test_hy_v3_tool_parser.py @ e24d1b24
//
// Ports every upstream case for the `hy_v3` parser: 7 non-streaming
// (TestHYV3ExtractToolCalls) + 6 streaming (TestHYV3ExtractToolCallsStreaming).
//
// Streaming: upstream drives the parser with EXPLICIT per-tag deltas and gates
// on TOKEN IDS built from the tokenizer vocab. Our seam is TEXT-only, so we feed
// the identical explicit deltas and let the parser gate on the tag TEXT (the
// documented deviation - see hy_v3.h). Tool calls are reconstructed with the
// same lenient collector upstream uses (_collect_streaming_tool_calls): it does
// NOT enforce one-tool-per-delta / id-once, so name + first-args deltas may ship
// together. Argument strings are compared as PARSED JSON (compact-vs-json.dumps
// whitespace deviation; json.loads-equivalent).
#include <doctest/doctest.h>

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/hy_v3.h"

using namespace vllm::entrypoints::openai;

namespace {

// mock_request.tools: get_current_date (no params) + get_weather (city/date str).
ChatCompletionRequest MockRequest() {
  ChatCompletionRequest req;
  ChatCompletionToolsParam t1;
  t1.type = "function";
  t1.function.name = "get_current_date";
  t1.function.parameters = nlohmann::json::object();
  ChatCompletionToolsParam t2;
  t2.type = "function";
  t2.function.name = "get_weather";
  t2.function.parameters = nlohmann::json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"},)"
      R"("date":{"type":"string"}}})");
  req.tools = std::vector<ChatCompletionToolsParam>{t1, t2};
  req.tool_choice = ToolChoice{"auto", std::nullopt};
  return req;
}

struct CollectedCall {
  std::string name;
  std::string arguments;
};

// Mirror of upstream _collect_streaming_tool_calls (lenient - accumulates
// name/arguments per index; no id/once invariants).
std::vector<CollectedCall> CollectStreaming(HYV3ToolParser& parser,
                                            const std::vector<std::string>& deltas,
                                            const ChatCompletionRequest& req) {
  std::map<int, CollectedCall> by_index;
  std::vector<int> order;
  std::string prev;
  for (const std::string& delta : deltas) {
    const std::string cur = prev + delta;
    auto dm = parser.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (!dm.has_value() || !dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      const int idx = tc.index;
      if (by_index.find(idx) == by_index.end()) {
        CollectedCall c;
        c.name = tc.function.name.value_or("");
        c.arguments = tc.function.arguments.value_or("");
        by_index[idx] = c;
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

std::string CollectStreamingContent(HYV3ToolParser& parser,
                                    const std::vector<std::string>& deltas,
                                    const ChatCompletionRequest& req) {
  std::string content;
  std::string prev;
  for (const std::string& delta : deltas) {
    const std::string cur = prev + delta;
    auto dm = parser.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (dm.has_value() && dm->content.has_value()) content += *dm->content;
  }
  return content;
}

}  // namespace

// ─── Non-streaming ──────────────────────────────────────────────────────────

TEST_CASE("hy_v3: no tool call") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::string out = "This is a plain response.";
  auto r = parser.extract_tool_calls(out, req);
  CHECK_FALSE(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == out);
}

TEST_CASE("hy_v3: zero arg inline") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::string out =
      "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>";
  auto r = parser.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_current_date");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) == nlohmann::json::object());
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("hy_v3: zero arg newline") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::string out =
      "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>";
  auto r = parser.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_current_date");
}

TEST_CASE("hy_v3: args same line") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::string out =
      "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key>"
      "<arg_value>Beijing</arg_value><arg_key>date</arg_key>"
      "<arg_value>2026-03-30</arg_value></tool_call></tool_calls>";
  auto r = parser.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Beijing"}, {"date", "2026-03-30"}});
}

TEST_CASE("hy_v3: args with newlines") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::string out =
      "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n"
      "<arg_value>Beijing</arg_value>\n<arg_key>date</arg_key>\n"
      "<arg_value>2026-03-30</arg_value>\n</tool_call>\n</tool_calls>";
  auto r = parser.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Beijing"}, {"date", "2026-03-30"}});
}

TEST_CASE("hy_v3: content before") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::string out =
      "Checking.<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n"
      "</tool_calls>";
  auto r = parser.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "Checking.");
}

TEST_CASE("hy_v3: multiple") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::string out =
      "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n"
      "<arg_value>Beijing</arg_value>\n<arg_key>date</arg_key>\n"
      "<arg_value>2026-03-30</arg_value>\n</tool_call>\n"
      "<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n"
      "<arg_value>Hangzhou</arg_value>\n<arg_key>date</arg_key>\n"
      "<arg_value>2026-03-30</arg_value>\n</tool_call>\n</tool_calls>";
  auto r = parser.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 2);
}

TEST_CASE("hy_v3: empty content is none") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::string out =
      "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>";
  auto r = parser.extract_tool_calls(out, req);
  CHECK_FALSE(r.content.has_value());
}

// ─── Streaming ──────────────────────────────────────────────────────────────

TEST_CASE("hy_v3: no tool call streaming") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::vector<std::string> deltas = {"This is ", "a plain ", "response."};
  CHECK(CollectStreamingContent(parser, deltas, req) == "This is a plain response.");
  HYV3ToolParser parser2;
  CHECK(CollectStreaming(parser2, deltas, req).empty());
}

TEST_CASE("hy_v3: zero arg streaming") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::vector<std::string> deltas = {
      "<tool_calls>",   "\n<tool_call>",  "get_current_date",
      "<tool_sep>",     "\n</tool_call>", "\n</tool_calls>"};
  auto tc = CollectStreaming(parser, deltas, req);
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "get_current_date");
  CHECK(nlohmann::json::parse(tc[0].arguments) == nlohmann::json::object());
}

TEST_CASE("hy_v3: args streaming") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::vector<std::string> deltas = {
      "<tool_calls>",
      "\n<tool_call>",
      "get_weather",
      "<tool_sep>",
      "\n<arg_key>city</arg_key>",
      "\n<arg_value>Beijing</arg_value>",
      "\n<arg_key>date</arg_key>",
      "\n<arg_value>2026-03-30</arg_value>",
      "\n</tool_call>",
      "\n</tool_calls>"};
  auto tc = CollectStreaming(parser, deltas, req);
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "get_weather");
  CHECK(nlohmann::json::parse(tc[0].arguments) ==
        nlohmann::json{{"city", "Beijing"}, {"date", "2026-03-30"}});
}

TEST_CASE("hy_v3: content before streaming") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::vector<std::string> deltas = {
      "Checking.",  "<tool_calls>",   "\n<tool_call>", "get_current_date",
      "<tool_sep>", "\n</tool_call>", "\n</tool_calls>"};
  // Collect both content and calls on a single pass.
  std::string content;
  std::map<int, CollectedCall> by_index;
  std::vector<int> order;
  std::string prev;
  for (const std::string& delta : deltas) {
    const std::string cur = prev + delta;
    auto dm = parser.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (!dm.has_value()) continue;
    if (dm->content.has_value()) content += *dm->content;
    if (!dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      if (by_index.find(tc.index) == by_index.end()) {
        by_index[tc.index] = {tc.function.name.value_or(""),
                              tc.function.arguments.value_or("")};
        order.push_back(tc.index);
      } else {
        if (tc.function.name.has_value()) by_index[tc.index].name += *tc.function.name;
        if (tc.function.arguments.has_value())
          by_index[tc.index].arguments += *tc.function.arguments;
      }
    }
  }
  CHECK(content.find("Checking.") != std::string::npos);
  REQUIRE(order.size() == 1);
  CHECK(by_index[order[0]].name == "get_current_date");
}

TEST_CASE("hy_v3: multiple streaming") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::vector<std::string> deltas = {
      "<tool_calls>",
      "\n<tool_call>",
      "get_weather",
      "<tool_sep>",
      "\n<arg_key>city</arg_key>",
      "\n<arg_value>Beijing</arg_value>",
      "\n<arg_key>date</arg_key>",
      "\n<arg_value>2026-03-30</arg_value>",
      "\n</tool_call>",
      "\n<tool_call>",
      "get_weather",
      "<tool_sep>",
      "\n<arg_key>city</arg_key>",
      "\n<arg_value>Hangzhou</arg_value>",
      "\n<arg_key>date</arg_key>",
      "\n<arg_value>2026-03-30</arg_value>",
      "\n</tool_call>",
      "\n</tool_calls>"};
  auto tc = CollectStreaming(parser, deltas, req);
  REQUIRE(tc.size() == 2);
  CHECK(nlohmann::json::parse(tc[0].arguments)["city"] == "Beijing");
  CHECK(nlohmann::json::parse(tc[1].arguments)["city"] == "Hangzhou");
}

TEST_CASE("hy_v3: all in one delta streaming") {
  HYV3ToolParser parser;
  ChatCompletionRequest req = MockRequest();
  const std::vector<std::string> deltas = {
      "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"};
  auto tc = CollectStreaming(parser, deltas, req);
  REQUIRE(tc.size() == 1);
  CHECK(tc[0].name == "get_current_date");
  CHECK(nlohmann::json::parse(tc[0].arguments) == nlohmann::json::object());
}
