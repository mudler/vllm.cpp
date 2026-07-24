// Reimplemented from the WIRE FORMAT of glm47_moe_tool_parser.py @ e24d1b24.
// Ports ALL upstream fidelity cases:
//   tests/tool_parsers/test_glm47_moe_tool_parser.py  (12: 10 extract + 2 stream)
//   tests/tool_parsers/test_glm4_moe_tool_parser.py   (6:  1 registry + 4 extract
//                                                       + 1 stream)
// Argument strings are compared as PARSED JSON (compact-dump deviation inherited
// from the seam); string arg values byte-for-byte since whitespace is load-bearing.
// Streaming asserts the CONCATENATION of emitted argument deltas parses to the
// expected JSON (exactly what the upstream tests assert via json.loads of the
// joined deltas), plus split-edge cases that fragment the tag tokens.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/glm47.h"

using namespace vllm::entrypoints::openai;

namespace {

nlohmann::json ParsedArgs(const std::string& a) { return nlohmann::json::parse(a); }

ChatCompletionToolsParam Tool(const std::string& name, const char* params_json) {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = name;
  if (params_json != nullptr) {
    t.function.parameters = nlohmann::json::parse(params_json);
  } else {
    t.function.parameters = nlohmann::json::parse("{}");
  }
  return t;
}

// glm47 fixture tools: get_current_date (no params) + get_weather (city, date).
ChatCompletionRequest Glm47Req() {
  ChatCompletionRequest req;
  req.tools = std::vector<ChatCompletionToolsParam>{
      Tool("get_current_date", "{}"),
      Tool("get_weather",
           R"({"type":"object","properties":{"city":{"type":"string"},
               "date":{"type":"string"}}})"),
  };
  req.tool_choice = ToolChoice{"auto", std::nullopt};
  return req;
}

// glm4 fixture tools: get_current_weather (strings), calculate (typed), get_time.
ChatCompletionRequest Glm4Req() {
  ChatCompletionRequest req;
  req.tools = std::vector<ChatCompletionToolsParam>{
      Tool("get_current_weather",
           R"({"type":"object","properties":{"city":{"type":"string"},
               "state":{"type":"string"},"unit":{"type":"string"}}})"),
      Tool("calculate",
           R"({"type":"object","properties":{"operation":{"type":"string"},
               "a":{"type":"number"},"b":{"type":"number"},
               "enabled":{"type":"boolean"}}})"),
      Tool("get_time", "{}"),
  };
  req.tool_choice = ToolChoice{"auto", std::nullopt};
  return req;
}

struct Collected {
  std::optional<std::string> id;
  std::string name;
  std::string args;
};

// Feed `chunks` one delta at a time; aggregate tool-call deltas by index and
// join content (mirrors the upstream _collect_tool_deltas helper).
std::pair<std::map<int, Collected>, std::string> FeedChunks(
    Glm47ToolParser& p, const std::vector<std::string>& chunks,
    const ChatCompletionRequest& req) {
  std::map<int, Collected> calls;
  std::string content;
  std::string prev;
  for (const std::string& chunk : chunks) {
    const std::string cur = prev + chunk;
    auto dm = p.extract_tool_calls_streaming(prev, cur, chunk, req);
    prev = cur;
    if (!dm.has_value()) continue;
    if (dm->content.has_value()) content += *dm->content;
    if (!dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      Collected& c = calls[tc.index];
      if (tc.id.has_value() && !tc.id->empty()) c.id = tc.id;
      if (tc.function.name.has_value() && !tc.function.name->empty())
        c.name = *tc.function.name;
      if (tc.function.arguments.has_value()) c.args += *tc.function.arguments;
    }
  }
  return {calls, content};
}

}  // namespace

// ── test_glm47_moe_tool_parser.py (extract) ─────────────────────────────────

TEST_CASE("glm47: no tool call") {
  Glm47ToolParser p;
  const std::string out = "This is a plain response.";
  auto r = p.extract_tool_calls(out, Glm47Req());
  CHECK_FALSE(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == out);
}

TEST_CASE("glm47: zero arg inline") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls("<tool_call>get_current_date</tool_call>", Glm47Req());
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_current_date");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) == nlohmann::json::object());
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("glm47: zero arg newline") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls("<tool_call>get_current_date\n</tool_call>", Glm47Req());
  CHECK(r.tools_called);
  CHECK(r.tool_calls[0].function.name == "get_current_date");
}

TEST_CASE("glm47: args same line") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls(
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>",
      Glm47Req());
  CHECK(r.tools_called);
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Beijing"}});
}

TEST_CASE("glm47: args with newlines") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls(
      "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>",
      Glm47Req());
  CHECK(r.tools_called);
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Beijing"}});
}

TEST_CASE("glm47: whitespace preserved in arg values") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls(
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  </arg_value></tool_call>",
      Glm47Req());
  CHECK(r.tools_called);
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments).at("city") == "  Beijing  ");
}

TEST_CASE("glm47: content before") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls("Checking.<tool_call>get_current_date</tool_call>", Glm47Req());
  CHECK(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "Checking.");
}

TEST_CASE("glm47: multiple") {
  Glm47ToolParser p;
  const std::string out =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>";
  auto r = p.extract_tool_calls(out, Glm47Req());
  CHECK(r.tool_calls.size() == 2);
}

TEST_CASE("glm47: empty content none") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls("<tool_call>get_current_date</tool_call>", Glm47Req());
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("glm47: whitespace content none") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls("  \n  <tool_call>get_current_date</tool_call>", Glm47Req());
  CHECK_FALSE(r.content.has_value());
}

// ── test_glm47_moe_tool_parser.py (streaming) ───────────────────────────────

TEST_CASE("glm47 streaming: no args") {
  Glm47ToolParser p;
  auto [calls, content] = FeedChunks(
      p, {"<tool_call>", "get_current_date", "</tool_call>"}, Glm47Req());
  REQUIRE(calls.count(0) == 1);
  CHECK(calls[0].name == "get_current_date");
  CHECK(calls[0].args == "{}");
}

TEST_CASE("glm47 streaming: with args") {
  Glm47ToolParser p;
  auto [calls, content] = FeedChunks(
      p,
      {"<tool_call>", "get_weather\n", "<arg_key>city</arg_key>", "<arg_value>",
       "Beijing", "</arg_value>", "</tool_call>"},
      Glm47Req());
  REQUIRE(calls.count(0) == 1);
  CHECK(calls[0].name == "get_weather");
  CHECK(ParsedArgs(calls[0].args).at("city") == "Beijing");
}

// ── test_glm4_moe_tool_parser.py ────────────────────────────────────────────

TEST_CASE("glm45/glm47: both names resolve to the GLM parser") {
  // Behavioral check (RTTI-free): both registered names produce a parser that
  // extracts a GLM tool call.
  for (const std::string& name : {std::string("glm45"), std::string("glm47")}) {
    auto parser = get_tool_parser(name);
    REQUIRE(parser != nullptr);
    auto r = parser->extract_tool_calls(
        "<tool_call>get_time\n</tool_call>", Glm4Req());
    CHECK(r.tools_called);
    CHECK(r.tool_calls[0].function.name == "get_time");
  }
}

TEST_CASE("glm45: extract with newline format") {
  Glm47ToolParser p;
  const std::string out =
      "I'll check it. <tool_call>get_current_weather\n"
      "<arg_key>city</arg_key>\n<arg_value>Dallas</arg_value>\n"
      "<arg_key>state</arg_key>\n<arg_value>TX</arg_value>\n"
      "<arg_key>unit</arg_key>\n<arg_value>fahrenheit</arg_value>\n</tool_call>";
  auto r = p.extract_tool_calls(out, Glm4Req());
  CHECK(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "I'll check it.");
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_current_weather");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) ==
        (nlohmann::json{{"city", "Dallas"}, {"state", "TX"}, {"unit", "fahrenheit"}}));
}

TEST_CASE("glm45: extract multiple tool calls with newline format") {
  Glm47ToolParser p;
  const std::string out =
      "<tool_call>get_current_weather\n<arg_key>city</arg_key><arg_value>Dallas</arg_value>\n</tool_call>\n"
      "<tool_call>get_current_weather\n<arg_key>city</arg_key><arg_value>Orlando</arg_value>\n</tool_call>";
  auto r = p.extract_tool_calls(out, Glm4Req());
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_current_weather");
  CHECK(r.tool_calls[1].function.name == "get_current_weather");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments).at("city") == "Dallas");
  CHECK(ParsedArgs(r.tool_calls[1].function.arguments).at("city") == "Orlando");
}

TEST_CASE("glm45: extract coerces schema types") {
  Glm47ToolParser p;
  const std::string out =
      "<tool_call>calculate\n<arg_key>operation</arg_key><arg_value>add</arg_value>\n"
      "<arg_key>a</arg_key><arg_value>42</arg_value>\n"
      "<arg_key>b</arg_key><arg_value>3.14</arg_value>\n"
      "<arg_key>enabled</arg_key><arg_value>true</arg_value>\n</tool_call>";
  auto r = p.extract_tool_calls(out, Glm4Req());
  CHECK(r.tools_called);
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) ==
        (nlohmann::json{{"operation", "add"}, {"a", 42}, {"b", 3.14}, {"enabled", true}}));
}

TEST_CASE("glm45: extract zero-argument tool call") {
  Glm47ToolParser p;
  auto r = p.extract_tool_calls("<tool_call>get_time\n</tool_call>", Glm4Req());
  CHECK(r.tools_called);
  CHECK(r.tool_calls[0].function.name == "get_time");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) == nlohmann::json::object());
}

TEST_CASE("glm45 streaming: newline format") {
  Glm47ToolParser p;
  auto [calls, content] = FeedChunks(
      p,
      {"<tool_call>", "get_current_weather\n", "<arg_key>city</arg_key>",
       "<arg_value>Bei", "jing</arg_value>", "</tool_call>"},
      Glm4Req());
  REQUIRE(calls.count(0) == 1);
  CHECK(calls[0].name == "get_current_weather");
  CHECK(ParsedArgs(calls[0].args) == nlohmann::json{{"city", "Beijing"}});
}

// ── Streaming split edges (added) ───────────────────────────────────────────

TEST_CASE("glm47 streaming: tag tokens split across chunks (char by char)") {
  Glm47ToolParser p;
  ChatCompletionRequest req = Glm47Req();
  const std::string full =
      "pre<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
      "<arg_key>date</arg_key><arg_value>2026-07-24</arg_value></tool_call>";
  std::map<int, Collected> calls;
  std::string content;
  std::string prev;
  for (char ch : full) {
    const std::string delta(1, ch);
    const std::string cur = prev + delta;
    auto dm = p.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (!dm.has_value()) continue;
    if (dm->content.has_value()) content += *dm->content;
    if (!dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      Collected& c = calls[tc.index];
      if (tc.function.name.has_value() && !tc.function.name->empty())
        c.name = *tc.function.name;
      if (tc.function.arguments.has_value()) c.args += *tc.function.arguments;
    }
  }
  CHECK(content == "pre");
  REQUIRE(calls.count(0) == 1);
  CHECK(calls[0].name == "get_weather");
  CHECK(ParsedArgs(calls[0].args) ==
        (nlohmann::json{{"city", "Beijing"}, {"date", "2026-07-24"}}));
}

TEST_CASE("glm47 streaming: coerced number/bool over split value") {
  Glm47ToolParser p;
  auto [calls, content] = FeedChunks(
      p,
      {"<tool_call>calculate<arg_key>a</arg_key><arg_value>4", "2</arg_value>",
       "<arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>"},
      Glm4Req());
  REQUIRE(calls.count(0) == 1);
  CHECK(ParsedArgs(calls[0].args) == (nlohmann::json{{"a", 42}, {"enabled", true}}));
}
