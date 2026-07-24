// Tests for the DeepSeek-V4 DSML tool-call parser.
// (vllm/tool_parsers/deepseekv4_tool_parser.py @ e24d1b24).
//
// Ports the upstream tests/tool_parsers/test_deepseekv4_tool_parser.py cases.
// V4 reuses the V3.2 DSML grammar VERBATIM, wrapping tool calls in
// <｜DSML｜tool_calls> instead of <｜DSML｜function_calls>. Of the 8 upstream
// cases, 7 are ported; ONE is out of scope:
//   - test_get_vllm_registry_structural_tag_returns_structural_tag exercises the
//     xgrammar StructuralTag / get_structural_tag GRAMMAR path, which is not part
//     of the text-only tool-parser seam (there is no get_structural_tag on the
//     C++ ToolParser). Recommended detection markers are reported instead.
// Added: a mid-marker UTF-8-split streaming case (the token-id -> text rework).
#include <doctest/doctest.h>

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v4.h"

using namespace vllm::entrypoints::openai;
using json = nlohmann::json;

namespace {

const std::string TC_START = DeepSeekV4ToolParser::kToolCallStartToken;
const std::string TC_END = DeepSeekV4ToolParser::kToolCallEndToken;
const std::string INV_START = "<｜DSML｜invoke name=\"";
const std::string INV_END = "</｜DSML｜invoke>";
const std::string PARAM_START = "<｜DSML｜parameter name=\"";
const std::string PARAM_END = "</｜DSML｜parameter>";

ChatCompletionToolsParam Tool(const std::string& name, const json& params) {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = name;
  t.function.parameters = params;
  return t;
}
ChatCompletionRequest Req() { return ChatCompletionRequest{}; }
ChatCompletionRequest ReqTools(std::vector<ChatCompletionToolsParam> tools) {
  ChatCompletionRequest r;
  r.tools = std::move(tools);
  return r;
}

std::string Param(const std::string& name, const std::string& attr,
                  const std::string& val) {
  return PARAM_START + name + "\" string=\"" + attr + "\">" + val + PARAM_END;
}
// build_tool_call(func, {k:v,...}) — every param string="true"; note the v4
// upstream helper puts a newline after EACH parameter block.
std::string BuildToolCall(const std::string& fn,
                          const std::vector<std::pair<std::string, std::string>>& params) {
  std::string ps;
  for (const auto& kv : params) ps += Param(kv.first, "true", kv.second) + "\n";
  return TC_START + "\n" + INV_START + fn + "\">\n" + ps + INV_END + "\n" + TC_END;
}

std::vector<DeltaMessage> StreamChunks(ToolParser& p, const std::string& full,
                                       std::size_t n,
                                       const ChatCompletionRequest& req) {
  std::vector<DeltaMessage> out;
  std::string prev;
  for (std::size_t i = 0; i < full.size(); i += n) {
    const std::string c = full.substr(i, n);
    const std::string curr = prev + c;
    auto d = p.extract_tool_calls_streaming(prev, curr, c, req);
    prev = curr;
    if (d.has_value()) out.push_back(std::move(*d));
  }
  return out;
}
std::string ReconArgs(const std::vector<DeltaMessage>& msgs, int tool_index = 0) {
  std::string s;
  for (const auto& m : msgs)
    if (m.tool_calls.has_value())
      for (const auto& tc : *m.tool_calls)
        if (tc.index == tool_index && tc.function.arguments.has_value() &&
            !tc.function.arguments->empty())
          s += *tc.function.arguments;
  return s;
}
std::string ReconContent(const std::vector<DeltaMessage>& msgs) {
  std::string s;
  for (const auto& m : msgs)
    if (m.content.has_value()) s += *m.content;
  return s;
}
std::vector<std::string> StreamedNames(const std::vector<DeltaMessage>& msgs) {
  std::vector<std::string> names;
  for (const auto& m : msgs)
    if (m.tool_calls.has_value())
      for (const auto& tc : *m.tool_calls)
        if (tc.function.name.has_value()) names.push_back(*tc.function.name);
  return names;
}
json ArgsOf(const ExtractedToolCallInformation& info, std::size_t i = 0) {
  return json::parse(info.tool_calls[i].function.arguments);
}

}  // namespace

TEST_CASE("v4: factory registration (test_registered)") {
  auto p = get_tool_parser("deepseek_v4");
  REQUIRE(p != nullptr);
}

TEST_CASE("v4: extract tool calls with leading content") {
  DeepSeekV4ToolParser p;
  auto req = Req();
  const std::string out = "Let me check. " +
                          BuildToolCall("get_weather", {{"location", "Beijing"}, {"unit", "celsius"}});
  auto r = p.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "Let me check. ");
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(ArgsOf(r) == json({{"location", "Beijing"}, {"unit", "celsius"}}));
}

TEST_CASE("v4: function_calls block is NOT accepted") {
  DeepSeekV4ToolParser p;
  auto req = Req();
  // Swap the v4 wrapper for the v3.2 one; v4 must not recognise it.
  std::string out = BuildToolCall("search", {{"query", "vllm"}});
  for (std::size_t pos = out.find("tool_calls"); pos != std::string::npos;
       pos = out.find("tool_calls", pos))
    out.replace(pos, std::string("tool_calls").size(), "function_calls");
  auto r = p.extract_tool_calls(out, req);
  CHECK_FALSE(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == out);
}

TEST_CASE("v4: streaming extracts complete invokes") {
  DeepSeekV4ToolParser p;
  auto req = Req();
  auto msgs = StreamChunks(p, BuildToolCall("search", {{"query", "deepseek v4"}}), 5, req);
  CHECK(StreamedNames(msgs) == std::vector<std::string>{"search"});
  CHECK(json::parse(ReconArgs(msgs)) == json({{"query", "deepseek v4"}}));
}

TEST_CASE("v4: streaming emits incremental argument chunks") {
  auto req = ReqTools({Tool("plan_trip", json::parse(R"({"type":"object","properties":{
      "days":{"type":"integer"},"flexible":{"type":"boolean"},
      "cities":{"type":"array","items":{"type":"string"}},"notes":{"type":"string"}}})"))});
  DeepSeekV4ToolParser p;
  const std::string full = TC_START + "\n" + INV_START + "plan_trip\">\n" +
                           Param("days", "false", "3") + "\n" +
                           Param("flexible", "false", "false") + "\n" +
                           Param("cities", "false", R"(["Beijing","Shanghai","Tokyo","New York"])") +
                           "\n" + Param("notes", "true", "靠窗座位") + "\n" + INV_END +
                           "\n" + TC_END;
  auto msgs = StreamChunks(p, full, 4, req);
  std::vector<std::string> chunks;
  for (const auto& m : msgs)
    if (m.tool_calls.has_value())
      for (const auto& tc : *m.tool_calls)
        if (tc.function.arguments.has_value() && !tc.function.arguments->empty())
          chunks.push_back(*tc.function.arguments);
  CHECK(chunks.size() > 2);
  CHECK(json::parse(ReconArgs(msgs)) == json::parse(R"({"days":3,"flexible":false,
      "cities":["Beijing","Shanghai","Tokyo","New York"],"notes":"靠窗座位"})"));
}

TEST_CASE("v4: extract tool calls with arguments wrapper (no newlines)") {
  auto req = ReqTools({Tool("get_weather", json::parse(
      R"({"type":"object","properties":{"location":{"type":"string"}}})"))});
  DeepSeekV4ToolParser p;
  const std::string out = TC_START + INV_START + "get_weather\">" +
                          Param("arguments", "false", R"({"location":"Beijing"})") +
                          INV_END + TC_END;
  auto r = p.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  CHECK(ArgsOf(r) == json({{"location", "Beijing"}}));
}

TEST_CASE("v4: composed schema converts object and array params") {
  auto req = ReqTools({Tool("set_timer", json::parse(R"({"type":"object","properties":{
      "wait":{"anyOf":[{"type":"object"},{"type":"null"}]},
      "patches":{"allOf":[{"type":"array","items":{"type":"object"}}]}}})"))});
  DeepSeekV4ToolParser p;
  const std::string out = TC_START + "\n" + INV_START + "set_timer\">\n" +
                          Param("wait", "false", R"({"type":"for","minutes":2880})") + "\n" +
                          Param("patches", "false",
                                R"([{"op":"replace","path":"/schedule","value":"quiet"}])") +
                          "\n" + INV_END + "\n" + TC_END;
  auto r = p.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  auto a = ArgsOf(r);
  CHECK(a["wait"] == json::parse(R"({"type":"for","minutes":2880})"));
  CHECK(a["patches"] == json::parse(R"([{"op":"replace","path":"/schedule","value":"quiet"}])"));
}

// ── added coverage: mid-marker UTF-8 split streaming ──────────────────────────
TEST_CASE("v4 streaming: split every marker mid-byte (UTF-8 safety)") {
  DeepSeekV4ToolParser p;
  auto req = Req();
  const std::string full = BuildToolCall("foo", {{"x", "1"}});
  // 3-byte chunks slice the 3-byte ｜ (U+FF5C) glyph mid-character.
  auto msgs = StreamChunks(p, full, 3, req);
  CHECK(json::parse(ReconArgs(msgs)) == json({{"x", "1"}}));
  const std::string content = ReconContent(msgs);
  CHECK(content.empty());
  CHECK(content.find("DSML") == std::string::npos);
}
