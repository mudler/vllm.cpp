// Reimplemented from the WIRE FORMAT of minimax_m2_tool_parser.py @ e24d1b24.
// Ports ALL 19 upstream cases from tests/tool_parsers/test_minimax_m2_tool_parser
// .py (all STREAMING), plus streaming split-edge cases. Argument strings compared
// as PARSED JSON; streaming asserts the CONCATENATION of emitted deltas parses to
// the expected JSON (exactly the upstream assertion via json.loads of the joined
// argument fragments).
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/minimax_m2.h"

using namespace vllm::entrypoints::openai;

namespace {

nlohmann::json ParsedArgs(const std::string& a) { return nlohmann::json::parse(a); }

struct Collected {
  std::optional<std::string> id;
  std::string name;
  std::string args;
};

struct FeedResult {
  std::map<int, Collected> calls;
  std::string content;
  std::vector<DeltaToolCall> raw_tool_deltas;  // every emitted DeltaToolCall
  int result_count = 0;
};

// Mirrors the upstream _feed helper: delta_text accumulates into current_text;
// collect non-None DeltaMessages.
FeedResult Feed(MinimaxM2ToolParser& p, const std::vector<std::string>& chunks,
                const ChatCompletionRequest* req = nullptr) {
  static ChatCompletionRequest kEmpty;  // no tools
  const ChatCompletionRequest& r = req != nullptr ? *req : kEmpty;
  FeedResult out;
  std::string prev;
  for (const std::string& chunk : chunks) {
    const std::string cur = prev + chunk;
    auto dm = p.extract_tool_calls_streaming(prev, cur, chunk, r);
    prev = cur;
    if (!dm.has_value()) continue;
    out.result_count += 1;
    if (dm->content.has_value()) out.content += *dm->content;
    if (!dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      out.raw_tool_deltas.push_back(tc);
      Collected& c = out.calls[tc.index];
      if (tc.id.has_value() && !tc.id->empty()) c.id = tc.id;
      if (tc.function.name.has_value() && !tc.function.name->empty())
        c.name = *tc.function.name;
      if (tc.function.arguments.has_value()) c.args += *tc.function.arguments;
    }
  }
  return out;
}

ChatCompletionToolsParam Tool(const std::string& name, const char* params_json) {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = name;
  t.function.parameters = nlohmann::json::parse(params_json);
  return t;
}

ChatCompletionRequest ReqWith(const ChatCompletionToolsParam& t) {
  ChatCompletionRequest req;
  req.tools = std::vector<ChatCompletionToolsParam>{t};
  return req;
}

}  // namespace

// ── TestContentStreaming ────────────────────────────────────────────────────

TEST_CASE("minimax_m2: plain content") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"Hello ", "world"});
  CHECK(r.content == "Hello world");
}

TEST_CASE("minimax_m2: content before tool call") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"Let me check. ",
                    "<minimax:tool_call><invoke name=\"get_weather\">"
                    "<parameter name=\"city\">Seattle</parameter>"
                    "</invoke></minimax:tool_call>"});
  CHECK(r.content == "Let me check. ");
}

TEST_CASE("minimax_m2: empty delta no crash") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {""});
  CHECK(r.result_count == 0);
}

// ── TestSingleInvoke ────────────────────────────────────────────────────────

TEST_CASE("minimax_m2: incremental chunks") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call>", "<invoke name=\"get_weather\">",
                    "<parameter name=\"city\">Seattle</parameter>",
                    "</invoke></minimax:tool_call>"});
  REQUIRE(r.calls.size() == 1);
  CHECK(r.calls[0].name == "get_weather");
  CHECK(ParsedArgs(r.calls[0].args) == nlohmann::json{{"city", "Seattle"}});
  CHECK(r.calls[0].id.has_value());
}

TEST_CASE("minimax_m2: single chunk complete") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"get_weather\">"
                    "<parameter name=\"city\">Seattle</parameter>"
                    "</invoke></minimax:tool_call>"});
  REQUIRE(r.calls.size() == 1);
  CHECK(ParsedArgs(r.calls[0].args) == nlohmann::json{{"city", "Seattle"}});
}

TEST_CASE("minimax_m2: multiple params") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call>", "<invoke name=\"get_weather\">",
                    "<parameter name=\"city\">Seattle</parameter>",
                    "<parameter name=\"days\">5</parameter>",
                    "</invoke></minimax:tool_call>"});
  CHECK(ParsedArgs(r.calls[0].args) ==
        (nlohmann::json{{"city", "Seattle"}, {"days", "5"}}));
}

// ── TestMultipleInvokes ─────────────────────────────────────────────────────

TEST_CASE("minimax_m2: two invokes incremental") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call>",
                    "<invoke name=\"search_web\"><parameter name=\"query\">OpenAI</parameter></invoke>",
                    "<invoke name=\"search_web\"><parameter name=\"query\">Gemini</parameter></invoke>",
                    "</minimax:tool_call>"});
  REQUIRE(r.calls.size() == 2);
  CHECK(r.calls[0].name == "search_web");
  CHECK(r.calls[1].name == "search_web");
  CHECK(ParsedArgs(r.calls[0].args) == nlohmann::json{{"query", "OpenAI"}});
  CHECK(ParsedArgs(r.calls[1].args) == nlohmann::json{{"query", "Gemini"}});
}

TEST_CASE("minimax_m2: two invokes in single delta") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call>",
                    "<invoke name=\"fn_a\"><parameter name=\"x\">1</parameter></invoke>"
                    "<invoke name=\"fn_b\"><parameter name=\"y\">2</parameter></invoke>",
                    "</minimax:tool_call>"});
  REQUIRE(r.calls.size() == 2);
  CHECK(r.calls[0].name == "fn_a");
  CHECK(r.calls[1].name == "fn_b");
}

TEST_CASE("minimax_m2: different functions") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call>",
                    "<invoke name=\"get_weather\"><parameter name=\"city\">NYC</parameter></invoke>",
                    "<invoke name=\"get_stock\"><parameter name=\"ticker\">AAPL</parameter></invoke>",
                    "</minimax:tool_call>"});
  CHECK(r.calls[0].name == "get_weather");
  CHECK(r.calls[1].name == "get_stock");
}

// ── TestDeltaMessageFormat ──────────────────────────────────────────────────

TEST_CASE("minimax_m2: tool call fields") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"fn\">"
                    "<parameter name=\"k\">v</parameter>"
                    "</invoke></minimax:tool_call>"});
  REQUIRE(r.raw_tool_deltas.size() == 1);
  const DeltaToolCall& tc = r.raw_tool_deltas[0];
  CHECK(tc.index == 0);
  REQUIRE(tc.type.has_value());
  CHECK(*tc.type == "function");
  CHECK(tc.id.has_value());
  REQUIRE(tc.function.name.has_value());
  CHECK(*tc.function.name == "fn");
  REQUIRE(tc.function.arguments.has_value());
  CHECK(ParsedArgs(*tc.function.arguments) == nlohmann::json{{"k", "v"}});
}

TEST_CASE("minimax_m2: multi invoke indices") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call>",
                    "<invoke name=\"a\"><parameter name=\"x\">1</parameter></invoke>",
                    "<invoke name=\"b\"><parameter name=\"x\">2</parameter></invoke>",
                    "</minimax:tool_call>"});
  std::vector<int> indices;
  for (const DeltaToolCall& tc : r.raw_tool_deltas) indices.push_back(tc.index);
  CHECK(indices == std::vector<int>{0, 1});
}

// ── TestLargeChunks ─────────────────────────────────────────────────────────

TEST_CASE("minimax_m2: header and params in separate chunks") {
  MinimaxM2ToolParser p;
  const std::string chunk1 = "<minimax:tool_call><invoke name=\"get_weather\">";
  const std::string chunk2 =
      "<parameter name=\"city\">Seattle</parameter>"
      "<parameter name=\"days\">5</parameter>"
      "</invoke></minimax:tool_call>";
  auto r = Feed(p, {chunk1, chunk2});
  REQUIRE(r.calls.size() == 1);
  CHECK(ParsedArgs(r.calls[0].args) ==
        (nlohmann::json{{"city", "Seattle"}, {"days", "5"}}));
}

// ── TestAnyOfNullableParam ──────────────────────────────────────────────────

TEST_CASE("minimax_m2: anyOf nullable non-null value") {
  ChatCompletionRequest req = ReqWith(Tool(
      "update_profile",
      R"({"type":"object","properties":{"nickname":{"anyOf":[{"type":"string"},{"type":"null"}]}}})"));
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"update_profile\">"
                    "<parameter name=\"nickname\">Alice</parameter>"
                    "</invoke></minimax:tool_call>"},
                &req);
  REQUIRE(r.calls.size() == 1);
  CHECK(ParsedArgs(r.calls[0].args).at("nickname") == "Alice");
}

TEST_CASE("minimax_m2: anyOf nullable null value") {
  ChatCompletionRequest req = ReqWith(Tool(
      "update_profile",
      R"({"type":"object","properties":{"nickname":{"anyOf":[{"type":"string"},{"type":"null"}]}}})"));
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"update_profile\">"
                    "<parameter name=\"nickname\">null</parameter>"
                    "</invoke></minimax:tool_call>"},
                &req);
  REQUIRE(r.calls.size() == 1);
  CHECK(ParsedArgs(r.calls[0].args).at("nickname").is_null());
}

TEST_CASE("minimax_m2: anyOf nullable object value") {
  ChatCompletionRequest req = ReqWith(Tool(
      "update_settings",
      R"({"type":"object","properties":{"config":{"anyOf":[{"type":"object"},{"type":"null"}]}}})"));
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"update_settings\">"
                    "<parameter name=\"config\">{\"theme\": \"dark\", \"fontSize\": 14}"
                    "</parameter></invoke></minimax:tool_call>"},
                &req);
  REQUIRE(r.calls.size() == 1);
  auto parsed = ParsedArgs(r.calls[0].args);
  CHECK(parsed.at("config") ==
        (nlohmann::json{{"theme", "dark"}, {"fontSize", 14}}));
  CHECK(parsed.at("config").is_object());
}

// ── TestNoneStringPreservation ──────────────────────────────────────────────

TEST_CASE("minimax_m2: 'none' preserved in enum") {
  ChatCompletionRequest req = ReqWith(Tool(
      "set_theme",
      R"({"type":"object","properties":{"theme":{"type":"string","enum":["dark","light","none"]}}})"));
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"set_theme\">"
                    "<parameter name=\"theme\">none</parameter>"
                    "</invoke></minimax:tool_call>"},
                &req);
  auto parsed = ParsedArgs(r.calls[0].args);
  CHECK(parsed.at("theme") == "none");
  CHECK_FALSE(parsed.at("theme").is_null());
}

TEST_CASE("minimax_m2: 'none' preserved as plain string") {
  ChatCompletionRequest req = ReqWith(Tool(
      "echo",
      R"({"type":"object","properties":{"message":{"type":"string"}}})"));
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"echo\">"
                    "<parameter name=\"message\">none</parameter>"
                    "</invoke></minimax:tool_call>"},
                &req);
  CHECK(ParsedArgs(r.calls[0].args).at("message") == "none");
}

TEST_CASE("minimax_m2: 'null' still converts to none") {
  ChatCompletionRequest req = ReqWith(Tool(
      "update_profile",
      R"({"type":"object","properties":{"nickname":{"anyOf":[{"type":"string"},{"type":"null"}]}}})"));
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"update_profile\">"
                    "<parameter name=\"nickname\">null</parameter>"
                    "</invoke></minimax:tool_call>"},
                &req);
  CHECK(ParsedArgs(r.calls[0].args).at("nickname").is_null());
}

TEST_CASE("minimax_m2: 'nil' preserved as string") {
  ChatCompletionRequest req = ReqWith(Tool(
      "echo",
      R"({"type":"object","properties":{"value":{"type":"string"}}})"));
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"echo\">"
                    "<parameter name=\"value\">nil</parameter>"
                    "</invoke></minimax:tool_call>"},
                &req);
  CHECK(ParsedArgs(r.calls[0].args).at("value") == "nil");
}

// ── Streaming split edges (added) ───────────────────────────────────────────

TEST_CASE("minimax_m2 streaming: tag tokens split char by char") {
  MinimaxM2ToolParser p;
  ChatCompletionRequest empty;
  const std::string full =
      "hi <minimax:tool_call><invoke name=\"get_weather\">"
      "<parameter name=\"city\">San Francisco</parameter>"
      "<parameter name=\"days\">5</parameter>"
      "</invoke></minimax:tool_call>";
  std::map<int, Collected> calls;
  std::string content;
  std::string prev;
  for (char ch : full) {
    const std::string delta(1, ch);
    const std::string cur = prev + delta;
    auto dm = p.extract_tool_calls_streaming(prev, cur, delta, empty);
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
  CHECK(content == "hi ");
  REQUIRE(calls.size() == 1);
  CHECK(calls[0].name == "get_weather");
  CHECK(ParsedArgs(calls[0].args) ==
        (nlohmann::json{{"city", "San Francisco"}, {"days", "5"}}));
}

TEST_CASE("minimax_m2 streaming: value split across chunks") {
  MinimaxM2ToolParser p;
  auto r = Feed(p, {"<minimax:tool_call><invoke name=\"echo\">"
                    "<parameter name=\"msg\">Hel",
                    "lo Wor", "ld</parameter></invoke></minimax:tool_call>"});
  REQUIRE(r.calls.size() == 1);
  CHECK(ParsedArgs(r.calls[0].args).at("msg") == "Hello World");
}

// ── Non-streaming extract (added: minimax has no upstream extract test) ──────

TEST_CASE("minimax_m2: extract_tool_calls non-streaming") {
  MinimaxM2ToolParser p;
  ChatCompletionRequest empty;
  auto r = p.extract_tool_calls(
      "note<minimax:tool_call><invoke name=\"get_weather\">"
      "<parameter name=\"city\">Seattle</parameter>"
      "<parameter name=\"days\">5</parameter>"
      "</invoke></minimax:tool_call>",
      empty);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) ==
        (nlohmann::json{{"city", "Seattle"}, {"days", "5"}}));
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "note");
}
