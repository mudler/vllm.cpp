// Tests for the DeepSeek-V3.2 "DSML" tool-call parser.
// (vllm/tool_parsers/deepseekv32_tool_parser.py @ e24d1b24).
//
// Ports ALL 49 upstream unit cases from
// tests/tool_parsers/test_deepseekv32_tool_parser.py (the richest suite in the
// registry) plus extra streaming mid-marker UTF-8-split coverage the token-id ->
// text rework must handle.
//
// SEAM ADAPTATIONS (see deepseek_v32.h DEVIATIONS): the upstream tests construct
// the parser WITH tools and pass request=None; our text-only seam reads
// request.tools, so schema-driven cases build a ChatCompletionRequest carrying
// the tool list. Two upstream cases exercise machinery that is not part of the
// text-only tool-parser seam and are adapted, not dropped:
//   - test_responses_function_tool_schema_in_streaming: upstream uses a Responses
//     API FunctionTool; the C++ protocol models tools only as
//     ChatCompletionToolsParam, so we supply the SAME schema through that shape.
//   - test_tool_detection_skip_special_tokens_false: the adjust_request half
//     (flipping skip_special_tokens) has no analogue at the seam — there is no
//     adjust_request — so we port the extraction half (non-streaming + streaming
//     reconstruction) and note the dropped assertion in the case.
//
// The DSML markers use the FULLWIDTH vertical bar ｜ (U+FF5C); the literal bytes
// here are byte-identical to what the parser matches.
#include <doctest/doctest.h>

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v32.h"

using namespace vllm::entrypoints::openai;
using json = nlohmann::json;

namespace {

const std::string FC_START = DeepSeekV32ToolParser::kToolCallStartToken;
const std::string FC_END = DeepSeekV32ToolParser::kToolCallEndToken;
const std::string INV_START = "<｜DSML｜invoke name=\"";
const std::string INV_END = "</｜DSML｜invoke>";
const std::string PARAM_START = "<｜DSML｜parameter name=\"";
const std::string PARAM_END = "</｜DSML｜parameter>";

// ── request / tool builders ───────────────────────────────────────────────────
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

// ── DSML markup builders (mirror the upstream test helpers) ───────────────────
std::string Param(const std::string& name, const std::string& attr,
                  const std::string& val) {
  return PARAM_START + name + "\" string=\"" + attr + "\">" + val + PARAM_END;
}
// build_tool_call(func, {k:v,...}) with every param string="true".
std::string BuildToolCall(const std::string& fn,
                          const std::vector<std::pair<std::string, std::string>>& params) {
  std::string ps;
  for (const auto& kv : params) ps += Param(kv.first, "true", kv.second);
  return FC_START + "\n" + INV_START + fn + "\">\n" + ps + "\n" + INV_END + "\n" +
         FC_END;
}
// One invoke with explicit param lines (each already newline-free).
std::string Invoke(const std::string& fn, const std::vector<std::string>& params) {
  std::string s = FC_START + "\n" + INV_START + fn + "\">\n";
  for (const auto& p : params) s += p + "\n";
  s += INV_END + "\n" + FC_END;
  return s;
}

// ── streaming drivers ─────────────────────────────────────────────────────────
// Line-split (mirrors upstream _stream): each DSML sentinel lands whole in one
// chunk, exercising the clean streaming path.
std::vector<DeltaMessage> StreamLines(ToolParser& p, const std::string& full,
                                      const ChatCompletionRequest& req) {
  std::vector<std::string> chunks;
  std::string rem = full;
  while (!rem.empty()) {
    const std::size_t nl = rem.find('\n');
    if (nl == std::string::npos) {
      chunks.push_back(rem);
      break;
    }
    chunks.push_back(rem.substr(0, nl + 1));
    rem = rem.substr(nl + 1);
  }
  std::vector<DeltaMessage> out;
  std::string prev;
  for (const auto& c : chunks) {
    const std::string curr = prev + c;
    auto d = p.extract_tool_calls_streaming(prev, curr, c, req);
    prev = curr;
    if (d.has_value()) out.push_back(std::move(*d));
  }
  return out;
}
// Fixed-size BYTE chunks (mirrors upstream _stream_chunked): splits sentinels
// (including multi-byte ｜) across delta boundaries.
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
std::map<int, std::string> NamesByIndex(const std::vector<DeltaMessage>& msgs) {
  std::map<int, std::string> names;
  for (const auto& m : msgs)
    if (m.tool_calls.has_value())
      for (const auto& tc : *m.tool_calls)
        if (tc.function.name.has_value()) names[tc.index] = *tc.function.name;
  return names;
}

json ArgsOf(const ExtractedToolCallInformation& info, std::size_t i = 0) {
  return json::parse(info.tool_calls[i].function.arguments);
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// TestExtractToolCalls (non-streaming) — 19 cases
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("v32: no tool call") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto r = p.extract_tool_calls("just some text", req);
  CHECK_FALSE(r.tools_called);
  CHECK(r.tool_calls.empty());
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "just some text");
}

TEST_CASE("v32: single tool, no params") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string out = FC_START + "\n" + INV_START + "get_time\">\n" + INV_END +
                          "\n" + FC_END;
  auto r = p.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_time");
  CHECK(ArgsOf(r) == json::object());
}

TEST_CASE("v32: single tool with params") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto r = p.extract_tool_calls(
      BuildToolCall("get_weather", {{"location", "SF"}, {"date", "2024-01-16"}}),
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(ArgsOf(r) == json({{"location", "SF"}, {"date", "2024-01-16"}}));
}

TEST_CASE("v32: content before tool call") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto r = p.extract_tool_calls(
      "Sure, let me check! " + BuildToolCall("get_weather", {{"location", "NYC"}}),
      req);
  CHECK(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "Sure, let me check! ");
}

TEST_CASE("v32: no content prefix returns none") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto r = p.extract_tool_calls(BuildToolCall("get_weather", {{"location", "NYC"}}),
                                req);
  CHECK(r.tools_called);
  CHECK_FALSE(r.content.has_value());
}

TEST_CASE("v32: multiple tools") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string out = FC_START + "\n" + INV_START + "get_weather\">\n" +
                          Param("location", "true", "SF") + "\n" + INV_END + "\n" +
                          INV_START + "get_weather\">\n" +
                          Param("location", "true", "NYC") + "\n" + INV_END + "\n" +
                          FC_END;
  auto r = p.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(ArgsOf(r, 0) == json({{"location", "SF"}}));
  CHECK(ArgsOf(r, 1) == json({{"location", "NYC"}}));
}

TEST_CASE("v32: type conversion in non-streaming") {
  auto req = ReqTools({Tool("toggle", json::parse(R"({"type":"object","properties":{
      "enabled":{"type":"boolean"},"count":{"type":"integer"}}})"))});
  DeepSeekV32ToolParser p;
  const std::string out = Invoke("toggle", {Param("enabled", "false", "true"),
                                            Param("count", "false", "42")});
  auto r = p.extract_tool_calls(out, req);
  REQUIRE(r.tool_calls.size() == 1);
  auto a = ArgsOf(r);
  CHECK(a == json({{"enabled", true}, {"count", 42}}));
  CHECK(a["enabled"].is_boolean());
  CHECK(a["count"].is_number_integer());
}

TEST_CASE("v32: string=true preserves literal despite schema") {
  auto req = ReqTools({Tool(
      "score", json::parse(R"({"type":"object","properties":{"value":{"type":"integer"}}})"))});
  DeepSeekV32ToolParser p;
  auto r = p.extract_tool_calls(Invoke("score", {Param("value", "true", "42")}), req);
  auto a = ArgsOf(r);
  CHECK(a == json({{"value", "42"}}));
  CHECK(a["value"].is_string());
}

TEST_CASE("v32: string=false allows schema conversion") {
  auto req = ReqTools({Tool(
      "score", json::parse(R"({"type":"object","properties":{"value":{"type":"integer"}}})"))});
  DeepSeekV32ToolParser p;
  auto r = p.extract_tool_calls(Invoke("score", {Param("value", "false", "42")}), req);
  auto a = ArgsOf(r);
  CHECK(a == json({{"value", 42}}));
  CHECK(a["value"].is_number_integer());
}

TEST_CASE("v32: composed schema converts object and array params") {
  auto req = ReqTools({Tool("set_timer", json::parse(R"({"type":"object","properties":{
      "wait":{"anyOf":[{"type":"object"},{"type":"null"}]},
      "patches":{"oneOf":[{"type":"array","items":{"type":"object"}},{"type":"null"}]}}})"))});
  DeepSeekV32ToolParser p;
  const std::string out =
      Invoke("set_timer", {Param("wait", "false", R"({"type":"for","minutes":2880})"),
                           Param("patches", "false",
                                 R"([{"op":"replace","path":"/schedule","value":"quiet"}])")});
  auto r = p.extract_tool_calls(out, req);
  auto a = ArgsOf(r);
  CHECK(a["wait"] == json::parse(R"({"type":"for","minutes":2880})"));
  CHECK(a["patches"] == json::parse(R"([{"op":"replace","path":"/schedule","value":"quiet"}])"));
  CHECK(a["wait"].is_object());
  CHECK(a["patches"].is_array());
}

TEST_CASE("v32: string=true preserves literal for composed schema") {
  auto req = ReqTools({Tool("set_timer", json::parse(R"({"type":"object","properties":{
      "wait":{"anyOf":[{"type":"object"},{"type":"null"}]}}})"))});
  DeepSeekV32ToolParser p;
  auto r = p.extract_tool_calls(
      Invoke("set_timer", {Param("wait", "true", R"({"type":"for","minutes":2880})")}),
      req);
  auto a = ArgsOf(r);
  CHECK(a == json({{"wait", R"({"type":"for","minutes":2880})"}}));
}

TEST_CASE("v32: arguments wrapper repaired") {
  auto req = ReqTools({Tool("get_weather", json::parse(
      R"({"type":"object","properties":{"location":{"type":"string"}}})"))});
  DeepSeekV32ToolParser p;
  auto r = p.extract_tool_calls(
      Invoke("get_weather", {Param("arguments", "false", R"({"location":"Beijing"})")}),
      req);
  CHECK(r.tools_called);
  CHECK(ArgsOf(r) == json({{"location", "Beijing"}}));
}

TEST_CASE("v32: input wrapper repaired") {
  auto req = ReqTools({Tool("get_weather", json::parse(
      R"({"type":"object","properties":{"location":{"type":"string"}}})"))});
  DeepSeekV32ToolParser p;
  auto r = p.extract_tool_calls(
      Invoke("get_weather", {Param("input", "true", R"({"location":"Beijing"})")}), req);
  CHECK(r.tools_called);
  CHECK(ArgsOf(r) == json({{"location", "Beijing"}}));
}

TEST_CASE("v32: object and array params") {
  auto req = ReqTools({Tool("update", json::parse(R"({"type":"object","properties":{
      "tags":{"type":"array"},"meta":{"type":"object"}}})"))});
  DeepSeekV32ToolParser p;
  const std::string out = Invoke("update", {Param("tags", "false", R"(["a", "b"])"),
                                            Param("meta", "false", R"({"k": 1})")});
  auto r = p.extract_tool_calls(out, req);
  auto a = ArgsOf(r);
  CHECK(a["tags"] == json::parse(R"(["a","b"])"));
  CHECK(a["tags"].is_array());
  CHECK(a["meta"] == json::parse(R"({"k":1})"));
  CHECK(a["meta"].is_object());
}

TEST_CASE("v32: number param") {
  auto req = ReqTools({Tool("measure", json::parse(R"({"type":"object","properties":{
      "ratio":{"type":"number"},"whole":{"type":"number"}}})"))});
  DeepSeekV32ToolParser p;
  const std::string out = Invoke("measure", {Param("ratio", "false", "3.14"),
                                             Param("whole", "false", "5.0")});
  auto r = p.extract_tool_calls(out, req);
  auto a = ArgsOf(r);
  CHECK(a["ratio"].get<double>() == doctest::Approx(3.14));
  CHECK(a["whole"] == 5);
  CHECK(a["whole"].is_number_integer());
}

TEST_CASE("v32: multi-typed schema") {
  auto req = ReqTools({Tool("set_val", json::parse(R"({"type":"object","properties":{
      "count":{"type":["integer","null"]},"label":{"type":["string","null"]}}})"))});
  DeepSeekV32ToolParser p;
  const std::string out = Invoke("set_val", {Param("count", "false", "42"),
                                             Param("label", "false", "hello")});
  auto a = ArgsOf(p.extract_tool_calls(out, req));
  CHECK(a["count"] == 42);
  CHECK(a["count"].is_number_integer());
  CHECK(a["label"] == "hello");
}

TEST_CASE("v32: multi-typed null value") {
  auto req = ReqTools({Tool("clear", json::parse(R"({"type":"object","properties":{
      "value":{"type":["integer","null"]}}})"))});
  DeepSeekV32ToolParser p;
  auto a = ArgsOf(p.extract_tool_calls(Invoke("clear", {Param("value", "false", "null")}), req));
  CHECK(a["value"].is_null());
}

TEST_CASE("v32: null not coerced without null in schema") {
  auto req = ReqTools({Tool("echo", json::parse(
      R"({"type":"object","properties":{"text":{"type":"string"}}})"))});
  DeepSeekV32ToolParser p;
  auto a = ArgsOf(p.extract_tool_calls(Invoke("echo", {Param("text", "false", "null")}), req));
  CHECK(a["text"] == "null");
  CHECK(a["text"].is_string());
}

TEST_CASE("v32: no schema keeps strings") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string out = Invoke("unknown_fn", {Param("count", "false", "42"),
                                                Param("flag", "false", "true")});
  auto a = ArgsOf(p.extract_tool_calls(out, req));
  CHECK(a["count"] == "42");
  CHECK(a["flag"] == "true");
}

// ════════════════════════════════════════════════════════════════════════════
// TestExtractToolCallsStreaming — 28 cases
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("v32 streaming: plain content no tool") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamLines(p, "Hello, world!", req);
  CHECK(ReconContent(msgs).find("Hello, world!") != std::string::npos);
  for (const auto& m : msgs) CHECK_FALSE(m.tool_calls.has_value());
}

TEST_CASE("v32 streaming: single tool") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamLines(p, BuildToolCall("get_weather", {{"location", "SF"}}), req);
  CHECK(json::parse(ReconArgs(msgs)) == json({{"location", "SF"}}));
}

TEST_CASE("v32 streaming: tool name emitted") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamLines(p, BuildToolCall("my_func", {{"x", "1"}}), req);
  CHECK(NamesByIndex(msgs)[0] == "my_func");
}

TEST_CASE("v32 streaming: content before tool call") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamLines(p, "Thinking... " + BuildToolCall("fn", {{"a", "b"}}), req);
  CHECK(ReconContent(msgs).find("Thinking") != std::string::npos);
}

TEST_CASE("v32 streaming: type conversion") {
  auto req = ReqTools({Tool("add", json::parse(R"({"type":"object","properties":{
      "x":{"type":"integer"},"y":{"type":"integer"}}})"))});
  DeepSeekV32ToolParser p;
  const std::string full = Invoke("add", {Param("x", "false", "3"), Param("y", "false", "4")});
  CHECK(json::parse(ReconArgs(StreamLines(p, full, req))) == json({{"x", 3}, {"y", 4}}));
}

TEST_CASE("v32 streaming: string=true preserves literal") {
  auto req = ReqTools({Tool("score", json::parse(
      R"({"type":"object","properties":{"value":{"type":"integer"}}})"))});
  DeepSeekV32ToolParser p;
  auto a = json::parse(ReconArgs(StreamLines(p, Invoke("score", {Param("value", "true", "42")}), req)));
  CHECK(a == json({{"value", "42"}}));
  CHECK(a["value"].is_string());
}

TEST_CASE("v32 streaming: composed schema conversion") {
  auto req = ReqTools({Tool("set_timer", json::parse(R"({"type":"object","properties":{
      "wait":{"anyOf":[{"type":"object"},{"type":"null"}]},
      "patches":{"oneOf":[{"type":"array","items":{"type":"object"}},{"type":"null"}]}}})"))});
  DeepSeekV32ToolParser p;
  const std::string full =
      Invoke("set_timer", {Param("wait", "false", R"({"type":"for","minutes":2880})"),
                           Param("patches", "false",
                                 R"([{"op":"replace","path":"/schedule","value":"quiet"}])")});
  auto a = json::parse(ReconArgs(StreamLines(p, full, req)));
  CHECK(a["wait"] == json::parse(R"({"type":"for","minutes":2880})"));
  CHECK(a["patches"] == json::parse(R"([{"op":"replace","path":"/schedule","value":"quiet"}])"));
}

TEST_CASE("v32 streaming: responses-style function tool schema") {
  // Upstream uses a Responses FunctionTool; same schema via ChatCompletionToolsParam.
  auto req = ReqTools({Tool("toggle", json::parse(R"({"type":"object","properties":{
      "enabled":{"type":"boolean"},"count":{"type":"integer"}}})"))});
  DeepSeekV32ToolParser p;
  const std::string full = Invoke("toggle", {Param("enabled", "false", "true"),
                                             Param("count", "false", "42")});
  auto a = json::parse(ReconArgs(StreamLines(p, full, req)));
  CHECK(a == json({{"enabled", true}, {"count", 42}}));
  CHECK(a["enabled"].is_boolean());
  CHECK(a["count"].is_number_integer());
}

TEST_CASE("v32 streaming: matches non-streaming conversion fallbacks") {
  auto req = ReqTools({Tool("coerce", json::parse(R"({"type":"object","properties":{
      "union_value":{"type":["null","string"]},"bad_int":{"type":"integer"},
      "nullable_string":{"type":["null","string"]},"null_string":{"type":"string"},
      "whole_number":{"type":"number"}}})"))});
  const std::string full = Invoke(
      "coerce", {Param("union_value", "false", "hello"), Param("bad_int", "false", "abc"),
                 Param("nullable_string", "false", "null"), Param("null_string", "false", "null"),
                 Param("whole_number", "false", "3.0")});
  DeepSeekV32ToolParser ns;
  auto ns_args = ArgsOf(ns.extract_tool_calls(full, req));
  DeepSeekV32ToolParser st;
  auto s_args = json::parse(ReconArgs(StreamLines(st, full, req)));
  CHECK(s_args == ns_args);
  CHECK(s_args == json::parse(R"({"union_value":"hello","bad_int":"abc",
      "nullable_string":null,"null_string":"null","whole_number":3})"));
}

TEST_CASE("v32 streaming: multiple tools") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string full = FC_START + "\n" + INV_START + "func_a\">\n" +
                           Param("p", "true", "v1") + "\n" + INV_END + "\n" +
                           INV_START + "func_b\">\n" + Param("q", "true", "v2") + "\n" +
                           INV_END + "\n" + FC_END;
  auto msgs = StreamLines(p, full, req);
  auto names = NamesByIndex(msgs);
  CHECK(names[0] == "func_a");
  CHECK(names[1] == "func_b");
  CHECK(json::parse(ReconArgs(msgs, 0)) == json({{"p", "v1"}}));
  CHECK(json::parse(ReconArgs(msgs, 1)) == json({{"q", "v2"}}));
}

TEST_CASE("v32 streaming: state reset on new stream") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string full = BuildToolCall("fn", {{"k", "v"}});
  StreamLines(p, full, req);  // first stream
  auto msgs2 = StreamLines(p, full, req);  // second stream must reset
  CHECK(json::parse(ReconArgs(msgs2)) == json({{"k", "v"}}));
}

TEST_CASE("v32 streaming: empty arguments") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string full = FC_START + "\n" + INV_START + "get_time\">\n" + INV_END +
                           "\n" + FC_END;
  CHECK(json::parse(ReconArgs(StreamLines(p, full, req))) == json::object());
}

TEST_CASE("v32 streaming: unique tool call ids") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string full = FC_START + "\n" + INV_START + "fn_a\">\n" +
                           Param("x", "true", "1") + "\n" + INV_END + "\n" + INV_START +
                           "fn_b\">\n" + Param("y", "true", "2") + "\n" + INV_END + "\n" +
                           FC_END;
  auto msgs = StreamLines(p, full, req);
  std::vector<std::string> ids;
  for (const auto& m : msgs)
    if (m.tool_calls.has_value())
      for (const auto& tc : *m.tool_calls)
        if (tc.id.has_value()) ids.push_back(*tc.id);
  REQUIRE(ids.size() == 2);
  CHECK(ids[0] != ids[1]);
}

TEST_CASE("v32 streaming: EOS after tool calls returns non-null") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string full = BuildToolCall("fn", {{"k", "v"}});
  auto msgs = StreamLines(p, full, req);
  bool saw_tc = false;
  for (const auto& m : msgs)
    if (m.tool_calls.has_value()) saw_tc = true;
  CHECK(saw_tc);
  // EOS: empty delta_text with a started tool array (text-only DEVIATION 4).
  auto eos = p.extract_tool_calls_streaming(full, full, "", req);
  CHECK(eos.has_value());
}

TEST_CASE("v32 streaming: matches non-streaming") {
  const std::string full = BuildToolCall("get_weather", {{"location", "SF"}, {"date", "2024-01-16"}});
  DeepSeekV32ToolParser ns;
  auto req = Req();
  auto info = ns.extract_tool_calls(full, req);
  DeepSeekV32ToolParser st;
  auto msgs = StreamLines(st, full, req);
  CHECK(NamesByIndex(msgs)[0] == info.tool_calls[0].function.name);
  CHECK(json::parse(ReconArgs(msgs)) == ArgsOf(info));
}

TEST_CASE("v32 streaming: single tool, start token split across chunks") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamChunks(p, BuildToolCall("get_weather", {{"location", "SF"}}), 5, req);
  CHECK(json::parse(ReconArgs(msgs)) == json({{"location", "SF"}}));
}

TEST_CASE("v32 streaming: content before tool, chunked") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamChunks(p, "Thinking... " + BuildToolCall("fn", {{"a", "b"}}), 7, req);
  CHECK(ReconContent(msgs).find("Thinking") != std::string::npos);
  CHECK(json::parse(ReconArgs(msgs)) == json({{"a", "b"}}));
}

TEST_CASE("v32 streaming: multiple tools, chunked") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string full = FC_START + "\n" + INV_START + "func_a\">\n" +
                           Param("p", "true", "v1") + "\n" + INV_END + "\n" +
                           INV_START + "func_b\">\n" + Param("q", "true", "v2") + "\n" +
                           INV_END + "\n" + FC_END;
  auto msgs = StreamChunks(p, full, 10, req);
  CHECK(json::parse(ReconArgs(msgs, 0)) == json({{"p", "v1"}}));
  CHECK(json::parse(ReconArgs(msgs, 1)) == json({{"q", "v2"}}));
}

TEST_CASE("v32 streaming: emits arguments before invoke completes") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  // Only a partial invoke (no closing invoke/function_calls tags).
  const std::string partial = FC_START + "\n" + INV_START + "fn\">\n" +
                              Param("k", "true", "val") + "\n";
  auto msgs = StreamLines(p, partial, req);
  std::string args;
  for (const auto& m : msgs)
    if (m.tool_calls.has_value())
      for (const auto& tc : *m.tool_calls)
        if (tc.function.arguments.has_value()) args += *tc.function.arguments;
  CHECK(args == "{\"k\":\"val\"");
}

TEST_CASE("v32 streaming: no marker leak, chunked") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamChunks(p, BuildToolCall("fn", {{"k", "v"}}), 5, req);
  CHECK(ReconContent(msgs).empty());
  CHECK(json::parse(ReconArgs(msgs)) == json({{"k", "v"}}));
}

TEST_CASE("v32 streaming: no marker leak with prefix, chunked") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamChunks(p, "Hello!" + BuildToolCall("fn", {{"a", "b"}}), 5, req);
  const std::string content = ReconContent(msgs);
  CHECK(content == "Hello!");
  CHECK(content.find("DSML") == std::string::npos);
  CHECK(content.find("<｜") == std::string::npos);
  CHECK(json::parse(ReconArgs(msgs)) == json({{"a", "b"}}));
}

TEST_CASE("v32 streaming: no marker leak, char-by-char (1 byte)") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  auto msgs = StreamChunks(p, BuildToolCall("fn", {{"k", "v"}}), 1, req);
  CHECK(ReconContent(msgs).empty());
  CHECK(json::parse(ReconArgs(msgs)) == json({{"k", "v"}}));
}

TEST_CASE("v32 streaming: no marker leak at all split points (mid-byte UTF-8)") {
  const std::string full = BuildToolCall("fn", {{"k", "v"}});
  for (std::size_t cs = 1; cs <= FC_START.size() + 1; ++cs) {
    DeepSeekV32ToolParser p;
    auto req = Req();
    auto msgs = StreamChunks(p, full, cs, req);
    CHECK_MESSAGE(ReconContent(msgs).empty(),
                  "leaked content at chunk_size=" << cs);
  }
}

TEST_CASE("v32 streaming: false partial marker emitted as content") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string full = "<｜DSM some regular text";
  auto msgs = StreamChunks(p, full, 3, req);
  CHECK(ReconContent(msgs) == full);
}

TEST_CASE("v32 streaming: object and array params") {
  auto req = ReqTools({Tool("update", json::parse(R"({"type":"object","properties":{
      "tags":{"type":"array"},"meta":{"type":"object"}}})"))});
  DeepSeekV32ToolParser p;
  const std::string full = Invoke("update", {Param("tags", "false", R"(["a", "b"])"),
                                             Param("meta", "false", R"({"k": 1})")});
  auto a = json::parse(ReconArgs(StreamLines(p, full, req)));
  CHECK(a["tags"] == json::parse(R"(["a","b"])"));
  CHECK(a["meta"] == json::parse(R"({"k":1})"));
}

TEST_CASE("v32 streaming: multi-typed schema") {
  auto req = ReqTools({Tool("set_val", json::parse(R"({"type":"object","properties":{
      "count":{"type":["integer","null"]}}})"))});
  DeepSeekV32ToolParser p;
  auto a = json::parse(ReconArgs(StreamLines(p, Invoke("set_val", {Param("count", "false", "42")}), req)));
  CHECK(a["count"] == 42);
  CHECK(a["count"].is_number_integer());
}

TEST_CASE("v32 streaming: multi-typed null") {
  auto req = ReqTools({Tool("clear", json::parse(R"({"type":"object","properties":{
      "value":{"type":["integer","null"]}}})"))});
  DeepSeekV32ToolParser p;
  auto a = json::parse(ReconArgs(StreamLines(p, Invoke("clear", {Param("value", "false", "null")}), req)));
  CHECK(a["value"].is_null());
}

TEST_CASE("v32 streaming: number param") {
  auto req = ReqTools({Tool("measure", json::parse(R"({"type":"object","properties":{
      "ratio":{"type":"number"}}})"))});
  DeepSeekV32ToolParser p;
  auto a = json::parse(ReconArgs(StreamLines(p, Invoke("measure", {Param("ratio", "false", "3.14")}), req)));
  CHECK(a["ratio"].get<double>() == doctest::Approx(3.14));
}

// ════════════════════════════════════════════════════════════════════════════
// TestDelimiterPreservation — 2 cases
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("v32: delimiter preserved (fast detokenization)") {
  DeepSeekV32ToolParser p;
  auto req = Req();
  const std::string out = Invoke("get_weather", {Param("location", "true", "Tokyo")});
  auto r = p.extract_tool_calls(out, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(ArgsOf(r) == json({{"location", "Tokyo"}}));
  CHECK_FALSE(r.content.has_value());

  auto r2 = p.extract_tool_calls("Here is the weather: " + out, req);
  CHECK(r2.tools_called);
  REQUIRE(r2.content.has_value());
  CHECK(*r2.content == "Here is the weather: ");
}

TEST_CASE("v32: tool detection (extraction half; adjust_request N/A at seam)") {
  // Upstream also asserts adjust_request flips skip_special_tokens=False. The
  // text-only seam has no adjust_request (the markers arrive as literal bytes),
  // so that half is not applicable; the extraction behaviour is what we port.
  auto req = ReqTools({Tool("search", json::parse(
      R"({"type":"object","properties":{"query":{"type":"string"}}})"))});
  DeepSeekV32ToolParser p;
  const std::string full = BuildToolCall("search", {{"query", "vllm documentation"}});
  auto ns = p.extract_tool_calls(full, req);
  CHECK(ns.tools_called);
  REQUIRE(ns.tool_calls.size() == 1);
  CHECK(ns.tool_calls[0].function.name == "search");
  CHECK(ArgsOf(ns) == json({{"query", "vllm documentation"}}));

  DeepSeekV32ToolParser st;
  auto msgs = StreamLines(st, full, req);
  CHECK(json::parse(ReconArgs(msgs)) == ArgsOf(ns));
}

// ── factory ───────────────────────────────────────────────────────────────────
TEST_CASE("v32: factory registration") {
  auto p = get_tool_parser("deepseek_v32");
  REQUIRE(p != nullptr);
  auto req = Req();
  CHECK(p->extract_tool_calls(BuildToolCall("f", {}), req).tools_called);
}
