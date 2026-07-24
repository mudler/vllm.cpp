// Ported from: vllm/tests/tool_parsers/test_gemma4_tool_parser.py @ e24d1b24
//
// Ports ALL 54 upstream cases for the reimplemented-from-wire-format `gemma4`
// parser (the second-richest suite):
//   TestParseGemma4Args   (20) - the _parse_gemma4_args dialect parser (ParseArgs)
//   TestParseGemma4Array   (6) - the _parse_gemma4_array parser (ParseArray)
//   TestExtractToolCalls  (13) - non-streaming extract_tool_calls
//   TestStreamingExtraction (15) - the incremental streaming state machine
// Multi-assert upstream methods are ported as one TEST_CASE with multiple CHECKs.
//
// Helper-parser cases compare RAW (uncoerced, string-valued) dialect output
// exactly as upstream (values are strings; coercion is a later engine step).
// Tool-call cases compare arguments as PARSED JSON (nlohmann) - the schema
// coercion (bare true/42/3.14 -> bool/int/number) is reproduced by the parser
// and asserted through json round-trips, matching upstream.
//
// Plus: get_tool_parser("gemma4") registration + streaming split-edge cases.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/gemma4.h"

using namespace vllm::entrypoints::openai;
using ojson = nlohmann::ordered_json;

namespace {

nlohmann::json Parsed(const std::string& s) { return nlohmann::json::parse(s); }

ChatCompletionToolsParam MakeTool(const std::string& name,
                                  const nlohmann::json& properties) {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = name;
  t.function.parameters =
      nlohmann::json{{"type", "object"}, {"properties", properties}};
  return t;
}

// The upstream _TOOLS fixture (schema-coercion targets).
ChatCompletionRequest MakeRequest() {
  ChatCompletionRequest req;
  req.tools = std::vector<ChatCompletionToolsParam>{
      MakeTool("set_status", {{"is_active", {{"type", "boolean"}}},
                              {"count", {{"type", "integer"}}},
                              {"score", {{"type", "number"}}}}),
      MakeTool("set_config", {{"count", {{"type", "integer"}}},
                              {"active", {{"type", "boolean"}}}}),
      MakeTool("search",
               {{"input",
                 {{"type", "object"},
                  {"properties", {{"all", {{"type", "boolean"}}}}}}}}),
      MakeTool("set", {{"flag", {{"type", "boolean"}}},
                       {"count", {{"type", "integer"}}}}),
      MakeTool("Edit", {{"file_path", {{"type", "string"}}},
                        {"old_string", {{"type", "string"}}},
                        {"new_string", {{"type", "string"}}},
                        {"replace_all", {{"type", "boolean"}}}}),
  };
  return req;
}

// Streaming harness: feed chunks, collect the per-tick deltas.
std::vector<std::optional<DeltaMessage>> Stream(
    Gemma4ToolParser& p, const std::vector<std::string>& chunks,
    const ChatCompletionRequest& req) {
  std::vector<std::optional<DeltaMessage>> out;
  std::string prev;
  for (const std::string& ch : chunks) {
    const std::string cur = prev + ch;
    out.push_back(p.extract_tool_calls_streaming(prev, cur, ch, req));
    prev = cur;
  }
  return out;
}

std::string CollectArgs(const std::vector<std::optional<DeltaMessage>>& r) {
  std::string s;
  for (const auto& d : r)
    if (d && d->tool_calls)
      for (const auto& tc : *d->tool_calls)
        if (tc.function.arguments) s += *tc.function.arguments;
  return s;
}

std::string CollectName(const std::vector<std::optional<DeltaMessage>>& r) {
  for (const auto& d : r)
    if (d && d->tool_calls)
      for (const auto& tc : *d->tool_calls)
        if (tc.function.name) return *tc.function.name;
  return "";
}

std::string CollectContent(const std::vector<std::optional<DeltaMessage>>& r) {
  std::string s;
  for (const auto& d : r)
    if (d && d->content) s += *d->content;
  return s;
}

struct IdxEntry {
  std::string name;
  std::string args;
};
std::map<int, IdxEntry> ByIndex(
    const std::vector<std::optional<DeltaMessage>>& r) {
  std::map<int, IdxEntry> by;
  for (const auto& d : r)
    if (d && d->tool_calls)
      for (const auto& tc : *d->tool_calls) {
        auto& e = by[tc.index];
        if (tc.function.name) e.name = *tc.function.name;
        if (tc.function.arguments) e.args += *tc.function.arguments;
      }
  return by;
}

}  // namespace

// ── TestParseGemma4Args (20) ──────────────────────────────────────────────────

TEST_CASE("gemma4 args: empty string") {
  CHECK(Gemma4ToolParser::ParseArgs("") == ojson::object());
}
TEST_CASE("gemma4 args: whitespace only") {
  CHECK(Gemma4ToolParser::ParseArgs("   ") == ojson::object());
}
TEST_CASE("gemma4 args: single string value") {
  CHECK(Gemma4ToolParser::ParseArgs("location:<|\"|>Paris<|\"|>") ==
        ojson{{"location", "Paris"}});
}
TEST_CASE("gemma4 args: string value with comma") {
  CHECK(Gemma4ToolParser::ParseArgs("location:<|\"|>Paris, France<|\"|>") ==
        ojson{{"location", "Paris, France"}});
}
TEST_CASE("gemma4 args: multiple string values") {
  CHECK(Gemma4ToolParser::ParseArgs(
            "location:<|\"|>San Francisco<|\"|>,unit:<|\"|>celsius<|\"|>") ==
        ojson{{"location", "San Francisco"}, {"unit", "celsius"}});
}
TEST_CASE("gemma4 args: integer value (raw string)") {
  CHECK(Gemma4ToolParser::ParseArgs("count:42") == ojson{{"count", "42"}});
}
TEST_CASE("gemma4 args: float value (raw string)") {
  CHECK(Gemma4ToolParser::ParseArgs("score:3.14") == ojson{{"score", "3.14"}});
}
TEST_CASE("gemma4 args: boolean true (raw string)") {
  CHECK(Gemma4ToolParser::ParseArgs("flag:true") == ojson{{"flag", "true"}});
}
TEST_CASE("gemma4 args: boolean false (raw string)") {
  CHECK(Gemma4ToolParser::ParseArgs("flag:false") == ojson{{"flag", "false"}});
}
TEST_CASE("gemma4 args: null value (raw string)") {
  CHECK(Gemma4ToolParser::ParseArgs("param:null") == ojson{{"param", "null"}});
}
TEST_CASE("gemma4 args: mixed types (raw strings)") {
  CHECK(Gemma4ToolParser::ParseArgs(
            "name:<|\"|>test<|\"|>,count:42,active:true,score:3.14") ==
        ojson{{"name", "test"},
              {"count", "42"},
              {"active", "true"},
              {"score", "3.14"}});
}
TEST_CASE("gemma4 args: nested object") {
  CHECK(Gemma4ToolParser::ParseArgs("nested:{inner:<|\"|>value<|\"|>}") ==
        ojson{{"nested", {{"inner", "value"}}}});
}
TEST_CASE("gemma4 args: array of strings") {
  CHECK(Gemma4ToolParser::ParseArgs("items:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]") ==
        ojson{{"items", {"a", "b"}}});
}
TEST_CASE("gemma4 args: delimited keys stripped") {
  CHECK(Gemma4ToolParser::ParseArgs("<|\"|>location<|\"|>:<|\"|>Paris<|\"|>") ==
        ojson{{"location", "Paris"}});
  CHECK(Gemma4ToolParser::ParseArgs(
            "outer:{<|\"|>inner<|\"|>:<|\"|>val<|\"|>}") ==
        ojson{{"outer", {{"inner", "val"}}}});
  CHECK(Gemma4ToolParser::ParseArgs(
            "<|\"|>name<|\"|>:<|\"|>Alice<|\"|>,count:42") ==
        ojson{{"name", "Alice"}, {"count", "42"}});
}
TEST_CASE("gemma4 args: unterminated string") {
  CHECK(Gemma4ToolParser::ParseArgs("key:<|\"|>unterminated") ==
        ojson{{"key", "unterminated"}});
}
TEST_CASE("gemma4 args: empty value") {
  CHECK(Gemma4ToolParser::ParseArgs("key:") == ojson{{"key", ""}});
}
TEST_CASE("gemma4 args: empty value partial withheld") {
  CHECK(Gemma4ToolParser::ParseArgs("key:", true) == ojson::object());
  CHECK(Gemma4ToolParser::ParseArgs("key: ", true) == ojson::object());
}
TEST_CASE("gemma4 args: empty value after other keys partial withheld") {
  CHECK(Gemma4ToolParser::ParseArgs("name:<|\"|>test<|\"|>,flag:", true) ==
        ojson{{"name", "test"}});
}
TEST_CASE("gemma4 args: trailing dot float partial withheld") {
  CHECK(Gemma4ToolParser::ParseArgs("left:108.,right:22.8", true) ==
        ojson::object());
  CHECK(Gemma4ToolParser::ParseArgs("name:<|\"|>test<|\"|>,score:3.,count:1",
                                    true) == ojson{{"name", "test"}});
  CHECK(Gemma4ToolParser::ParseArgs("left:108.,right:22.8", false) ==
        ojson{{"left", "108."}, {"right", "22.8"}});
}
TEST_CASE("gemma4 args: malformed partial array does not hang") {
  ojson r = Gemma4ToolParser::ParseArgs(":[t:[]");
  CHECK(r.is_object());
}

// ── TestParseGemma4Array (6) ──────────────────────────────────────────────────

TEST_CASE("gemma4 array: string array") {
  CHECK(Gemma4ToolParser::ParseArray("<|\"|>a<|\"|>,<|\"|>b<|\"|>") ==
        ojson::array({"a", "b"}));
}
TEST_CASE("gemma4 array: empty array") {
  CHECK(Gemma4ToolParser::ParseArray("") == ojson::array());
}
TEST_CASE("gemma4 array: bare values (raw strings)") {
  CHECK(Gemma4ToolParser::ParseArray("42,true,3.14") ==
        ojson::array({"42", "true", "3.14"}));
}
TEST_CASE("gemma4 array: string element with closing bracket") {
  CHECK(Gemma4ToolParser::ParseArray(
            "[<|\"|>a]b<|\"|>,<|\"|>c<|\"|>],<|\"|>tail<|\"|>") ==
        ojson::array({ojson::array({"a]b", "c"}), "tail"}));
}
TEST_CASE("gemma4 array: stray closing bracket") {
  CHECK(Gemma4ToolParser::ParseArray("42,]trailing") == ojson::array({"42"}));
}
TEST_CASE("gemma4 array: trailing dot float partial withheld") {
  CHECK(Gemma4ToolParser::ParseArray("108.,22.8", true) == ojson::array());
  CHECK(Gemma4ToolParser::ParseArray("42,108.,3", true) ==
        ojson::array({"42"}));
}

// ── TestExtractToolCalls (13) ─────────────────────────────────────────────────

TEST_CASE("gemma4 extract: no tool calls") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  const std::string out = "Hello, how can I help you today?";
  auto r = p.extract_tool_calls(out, req);
  CHECK(r.tools_called == false);
  CHECK(r.tool_calls.empty());
  REQUIRE(r.content.has_value());
  CHECK(*r.content == out);
}
TEST_CASE("gemma4 extract: single tool call") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|>",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "London"}});
}
TEST_CASE("gemma4 extract: multiple arguments") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:get_weather{location:<|\"|>San Francisco<|\"|>,"
      "unit:<|\"|>celsius<|\"|>}<tool_call|>",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "San Francisco"}, {"unit", "celsius"}});
}
TEST_CASE("gemma4 extract: text before tool call") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "Let me check the weather for you. "
      "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "Let me check the weather for you.");
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
}
TEST_CASE("gemma4 extract: multiple tool calls") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|>"
      "<|tool_call>call:get_time{location:<|\"|>London<|\"|>}<tool_call|>",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(r.tool_calls[1].function.name == "get_time");
}
TEST_CASE("gemma4 extract: nested arguments") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:complex_function{nested:{inner:<|\"|>value<|\"|>},"
      "list:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|>",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "complex_function");
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"nested", {{"inner", "value"}}},
                       {"list", {"a", "b"}}});
}
TEST_CASE("gemma4 extract: number and boolean (schema coerced)") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:set_status{is_active:true,count:42,score:3.14}"
      "<tool_call|>",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "set_status");
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"is_active", true}, {"count", 42}, {"score", 3.14}});
}
TEST_CASE("gemma4 extract: incomplete tool call") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:get_weather{location:<|\"|>London", req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "London"}});
}
TEST_CASE("gemma4 extract: hyphenated function name") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:get-weather{location:<|\"|>London<|\"|>}<tool_call|>",
      req);
  CHECK(r.tools_called);
  CHECK(r.tool_calls[0].function.name == "get-weather");
}
TEST_CASE("gemma4 extract: dotted function name") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:weather.get{location:<|\"|>London<|\"|>}<tool_call|>",
      req);
  CHECK(r.tools_called);
  CHECK(r.tool_calls[0].function.name == "weather.get");
}
TEST_CASE("gemma4 extract: no arguments") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls("<|tool_call>call:get_status{}<tool_call|>", req);
  CHECK(r.tools_called);
  CHECK(r.tool_calls[0].function.name == "get_status");
  CHECK(Parsed(r.tool_calls[0].function.arguments) == nlohmann::json::object());
}
TEST_CASE("gemma4 extract: string with internal comma preserved") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:get_weather{location:<|\"|>Paris, France<|\"|>}"
      "<tool_call|>",
      req);
  CHECK(r.tools_called);
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "Paris, France"}});
}
TEST_CASE("gemma4 extract: array argument") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = p.extract_tool_calls(
      "<|tool_call>call:pick{items:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|>",
      req);
  CHECK(r.tools_called);
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"items", {"a", "b"}}});
}

// ── TestStreamingExtraction (15) ──────────────────────────────────────────────

TEST_CASE("gemma4 stream: basic single tool") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:get_weather{",
                      "location:<|\"|>Paris", ", France", "<|\"|>}",
                      "<tool_call|>"},
                  req);
  CHECK(CollectName(r) == "get_weather");
  CHECK(Parsed(CollectArgs(r)) ==
        nlohmann::json{{"location", "Paris, France"}});
}
TEST_CASE("gemma4 stream: multi arg") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:get_weather{",
                      "location:<|\"|>Tokyo<|\"|>,", "unit:<|\"|>celsius<|\"|>}",
                      "<tool_call|>"},
                  req);
  CHECK(CollectName(r) == "get_weather");
  CHECK(Parsed(CollectArgs(r)) ==
        nlohmann::json{{"location", "Tokyo"}, {"unit", "celsius"}});
}
TEST_CASE("gemma4 stream: no extra brace") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:get_weather{",
                      "location:<|\"|>London<|\"|>}", "<tool_call|>"},
                  req);
  const std::string args = CollectArgs(r);
  CHECK(Parsed(args) == nlohmann::json{{"location", "London"}});
  std::size_t braces = 0;
  for (char c : args)
    if (c == '}') ++braces;
  CHECK(braces <= 1);
}
TEST_CASE("gemma4 stream: no unquoted keys") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:get_weather{",
                      "location:<|\"|>Paris<|\"|>}", "<tool_call|>"},
                  req);
  const std::string args = CollectArgs(r);
  CHECK(args.find('{') != std::string::npos);
  CHECK(args.find("\"location\"") != std::string::npos);
}
TEST_CASE("gemma4 stream: name has no call prefix") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:get_weather{",
                      "location:<|\"|>Paris<|\"|>}", "<tool_call|>"},
                  req);
  const std::string name = CollectName(r);
  CHECK(name == "get_weather");
  CHECK(name.rfind("call:", 0) != 0);
}
TEST_CASE("gemma4 stream: text before tool call") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"Let me check ", "the weather. ", "<|tool_call>",
                      "call:get_weather{", "location:<|\"|>London<|\"|>}",
                      "<tool_call|>"},
                  req);
  std::string content = CollectContent(r);
  // strip leading whitespace for the startswith check
  std::size_t b = content.find_first_not_of(" \t\n");
  if (b != std::string::npos) content = content.substr(b);
  CHECK(content.rfind("Let me check", 0) == 0);
}
TEST_CASE("gemma4 stream: numeric args coerced") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:set_config{", "count:42,",
                      "active:true}", "<tool_call|>"},
                  req);
  nlohmann::json args = Parsed(CollectArgs(r));
  CHECK(args["count"] == 42);
  CHECK(args["active"] == true);
}
TEST_CASE("gemma4 stream: boolean split across chunks (nested)") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", std::string("call:search{input:{all:") +
                                          std::string("true").substr(0, 3),
                      "e}}", "<tool_call|>"},
                  req);
  nlohmann::json args = Parsed(CollectArgs(r));
  CHECK(args["input"]["all"] == true);
}
TEST_CASE("gemma4 stream: false split across chunks") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", std::string("call:set{flag:") +
                                          std::string("false").substr(0, 4),
                      "e}", "<tool_call|>"},
                  req);
  nlohmann::json args = Parsed(CollectArgs(r));
  CHECK(args["flag"] == false);
}
TEST_CASE("gemma4 stream: number split across chunks") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:set{count:4", "2}", "<tool_call|>"},
                  req);
  nlohmann::json args = Parsed(CollectArgs(r));
  CHECK(args["count"] == 42);
}
TEST_CASE("gemma4 stream: empty args") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:get_status{}", "<tool_call|>"}, req);
  CHECK(CollectName(r) == "get_status");
}
TEST_CASE("gemma4 stream: split delimiter no invalid json") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:todowrite{",
                      "content:<|\"|>Buy milk<|", "\"|>}", "<tool_call|>"},
                  req);
  const std::string args = CollectArgs(r);
  CHECK(Parsed(args)["content"] == "Buy milk");
  CHECK(args.find("<|") == std::string::npos);
}
TEST_CASE("gemma4 stream: plain text after tool call not duplicated") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:get_weather{",
                      "location:<|\"|>Paris<|\"|>}", "<tool_call|>", "<", "div>"},
                  req);
  const std::string content = CollectContent(r);
  CHECK(content == "<div>");
}
TEST_CASE("gemma4 stream: html argument not duplicated") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(
      p,
      {"<|tool_call>", "call:write_file{", "path:<|\"|>index.html<|\"|>,",
       "content:<|\"|><!DOCTYPE html>\n<", "html lang=\"zh-CN\">\n<",
       "head>\n    <", "meta charset=\"UTF-8\">\n    <",
       "meta name=\"viewport\" content=\"width=device-width\">\n", "<|\"|>}",
       "<tool_call|>"},
      req);
  nlohmann::json args = Parsed(CollectArgs(r));
  CHECK(args["path"] == "index.html");
  CHECK(args["content"] ==
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n    "
        "<meta charset=\"UTF-8\">\n    "
        "<meta name=\"viewport\" content=\"width=device-width\">\n");
}
TEST_CASE("gemma4 stream: trailing bare bool not duplicated") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool_call>", "call:Edit{",
                      "file_path:<|\"|>src/env.py<|\"|>,",
                      "old_string:<|\"|>old_val<|\"|>,",
                      "new_string:<|\"|>new_val<|\"|>,", "replace_all:",
                      "false}", "<tool_call|>"},
                  req);
  const std::string args = CollectArgs(r);
  CHECK(Parsed(args) == nlohmann::json{{"file_path", "src/env.py"},
                                       {"old_string", "old_val"},
                                       {"new_string", "new_val"},
                                       {"replace_all", false}});
  // "replace_all" must appear exactly once.
  std::size_t count = 0, pos = 0;
  while ((pos = args.find("replace_all", pos)) != std::string::npos) {
    ++count;
    pos += 11;
  }
  CHECK(count == 1);
}

// ── Batched / single-chunk streaming (upstream TestStreamingExtraction) ───────

TEST_CASE("gemma4 stream: single chunk complete tool call") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(
      p,
      {"<|tool_call>call:name_a_color{color_hex:<|\"|>00ff11<|\"|>}<tool_call|>"},
      req);
  int tool_deltas = 0;
  for (const auto& d : r) {
    if (d && d->tool_calls) ++tool_deltas;
    if (d) CHECK_FALSE(d->content.has_value());
  }
  CHECK(tool_deltas == 1);
  auto by = ByIndex(r);
  REQUIRE(by.count(0) == 1);
  CHECK(by[0].name == "name_a_color");
  CHECK(Parsed(by[0].args) == nlohmann::json{{"color_hex", "00ff11"}});
}
TEST_CASE("gemma4 stream: multi-chunk batched tool calls") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(
      p,
      {"<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|>"
       "<|tool_call>call:get_time{timezone:<|\"|>GMT<|\"|>}<tool_call|>"},
      req);
  auto by = ByIndex(r);
  REQUIRE(by.count(0) == 1);
  REQUIRE(by.count(1) == 1);
  CHECK(by[0].name == "get_weather");
  CHECK(Parsed(by[0].args) == nlohmann::json{{"location", "London"}});
  CHECK(by[1].name == "get_time");
  CHECK(Parsed(by[1].args) == nlohmann::json{{"timezone", "GMT"}});
}

// ── Registration + split-edge ─────────────────────────────────────────────────

TEST_CASE("gemma4: get_tool_parser resolves the registered name") {
  auto parser = get_tool_parser("gemma4");
  REQUIRE(parser != nullptr);
  ChatCompletionRequest req = MakeRequest();
  auto r = parser->extract_tool_calls(
      "<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|>",
      req);
  CHECK(r.tools_called);
  CHECK(r.tool_calls[0].function.name == "get_weather");
}
TEST_CASE("gemma4 stream: start tag split across chunks buffers cleanly") {
  Gemma4ToolParser p;
  ChatCompletionRequest req = MakeRequest();
  auto r = Stream(p, {"<|tool", "_call>", "call:get_weather{",
                      "location:<|\"|>London<|\"|>}", "<tool_call|>"},
                  req);
  // The split "<|tool" prefix must not leak into content.
  CHECK(CollectContent(r).empty());
  CHECK(CollectName(r) == "get_weather");
  CHECK(Parsed(CollectArgs(r)) == nlohmann::json{{"location", "London"}});
}
