// Ported from: vllm/tests/parser/engine/test_seed_oss.py (+ the shared Qwen3
// grammar behaviour it defers to, tests/parser/engine/test_qwen3.py) @ e24d1b24
//
// seed_oss is Qwen3 with four `seed:`-prefixed wrapper tokens, so the shared XML
// grammar (arg types-as-strings, multiline values, parallel calls, streaming
// mechanics) is the fidelity reference. Ports the seed_oss-specific text-level
// tool-parser cases (single call, the #46314 malformed-header regression, basic
// streaming, the reasoning-stripped end-to-end tool extraction) and derives 14
// further cases from the Qwen3 grammar (single, parallel, typed-args-as-strings,
// empty args, surrounding text, no-tool, escaped strings, multiline, multiple
// params, nested JSON-array param, consecutive-without-close; + 4 streaming:
// basic, multi-param, split value, parallel).
//
// Values are compared as PARSED JSON; per _qwen3_arg_converter every value is a
// STRING (no schema coercion), matching test_qwen3.py::test_various_data_types.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/seed_oss.h"

using namespace vllm::entrypoints::openai;

namespace {

nlohmann::json Parsed(const std::string& s) { return nlohmann::json::parse(s); }

std::vector<std::optional<DeltaMessage>> Stream(
    SeedOssToolParser& p, const std::vector<std::string>& chunks,
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

std::vector<std::string> CollectNames(
    const std::vector<std::optional<DeltaMessage>>& r) {
  std::vector<std::string> names;
  for (const auto& d : r)
    if (d && d->tool_calls)
      for (const auto& tc : *d->tool_calls)
        if (tc.function.name) names.push_back(*tc.function.name);
  return names;
}

std::string CollectContent(const std::vector<std::optional<DeltaMessage>>& r) {
  std::string s;
  for (const auto& d : r)
    if (d && d->content) s += *d->content;
  return s;
}

}  // namespace

// ── seed_oss-specific (test_seed_oss.py) ──────────────────────────────────────

TEST_CASE("seed_oss: single tool call") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=get_weather>\n"
      "<parameter=city>Tokyo</parameter>\n</function>\n</seed:tool_call>",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Tokyo"}});
}

TEST_CASE("seed_oss: malformed function end does not drop siblings (#46314)") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=broken</function>\n</seed:tool_call>"
      "<seed:tool_call>\n<function=get_weather>\n"
      "<parameter=city>Tokyo</parameter>\n</function>\n</seed:tool_call>",
      req);
  bool found = false;
  for (const auto& tc : r.tool_calls) {
    if (tc.function.name == "get_weather") {
      found = true;
      CHECK(Parsed(tc.function.arguments) == nlohmann::json{{"city", "Tokyo"}});
    }
  }
  CHECK(found);
}

TEST_CASE("seed_oss: basic streaming") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = Stream(p, {"<seed:tool_call>\n", "<function=get_weather>\n",
                      "<parameter=city>Tokyo", "</parameter>\n", "</function>\n",
                      "</seed:tool_call>"},
                  req);
  CHECK(CollectName(r) == "get_weather");
  CHECK(Parsed(CollectArgs(r)) == nlohmann::json{{"city", "Tokyo"}});
}

TEST_CASE("seed_oss: end-to-end tool extraction on reasoning-stripped text") {
  // The reasoning parser strips <seed:think>..</seed:think>; the tool parser
  // then sees only the tool-call region (test_end_to_end_through_adapters).
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=get_weather>\n"
      "<parameter=city>Tokyo</parameter>\n</function>\n</seed:tool_call>",
      req);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Tokyo"}});
}

// ── Derived from the Qwen3 shared grammar ─────────────────────────────────────

TEST_CASE("seed_oss: no tool calls is plain content") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  const std::string out = "This is a regular response without any tool calls.";
  auto r = p.extract_tool_calls(out, req);
  CHECK(r.tools_called == false);
  CHECK(r.tool_calls.empty());
  REQUIRE(r.content.has_value());
  CHECK(*r.content == out);
}

TEST_CASE("seed_oss: parallel tool calls") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=get_weather>\n"
      "<parameter=city>Tokyo</parameter>\n</function>\n</seed:tool_call>"
      "<seed:tool_call>\n<function=get_time>\n"
      "<parameter=timezone>Asia/Tokyo</parameter>\n</function>\n"
      "</seed:tool_call>",
      req);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(r.tool_calls[1].function.name == "get_time");
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Tokyo"}});
  CHECK(Parsed(r.tool_calls[1].function.arguments) ==
        nlohmann::json{{"timezone", "Asia/Tokyo"}});
}

TEST_CASE("seed_oss: typed args stay strings (no schema)") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=test_function>\n"
      "<parameter=string_field>hello</parameter>\n"
      "<parameter=int_field>42</parameter>\n"
      "<parameter=float_field>3.14</parameter>\n"
      "<parameter=bool_field>true</parameter>\n"
      "<parameter=null_field>null</parameter>\n"
      "<parameter=array_field>[\"a\", \"b\", \"c\"]</parameter>\n"
      "<parameter=object_field>{\"nested\": \"value\"}</parameter>\n"
      "</function>\n</seed:tool_call>",
      req);
  nlohmann::json args = Parsed(r.tool_calls[0].function.arguments);
  CHECK(args["string_field"] == "hello");
  CHECK(args["int_field"] == "42");
  CHECK(args["float_field"] == "3.14");
  CHECK(args["bool_field"] == "true");
  CHECK(args["null_field"] == "null");
  CHECK(args["array_field"] == "[\"a\", \"b\", \"c\"]");
  CHECK(args["object_field"] == "{\"nested\": \"value\"}");
}

TEST_CASE("seed_oss: empty arguments") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=refresh>\n</function>\n</seed:tool_call>",
      req);
  CHECK(r.tools_called);
  CHECK(r.tool_calls[0].function.name == "refresh");
  CHECK(Parsed(r.tool_calls[0].function.arguments) == nlohmann::json::object());
}

TEST_CASE("seed_oss: surrounding text preserved as content") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "Let me check the weather for you.\n\n"
      "<seed:tool_call>\n<function=get_weather>\n"
      "<parameter=city>Tokyo</parameter>\n</function>\n</seed:tool_call>\n\n"
      "I will get that information.",
      req);
  CHECK(r.tools_called);
  REQUIRE(r.content.has_value());
  CHECK(r.content->find("Let me check the weather") != std::string::npos);
  CHECK(r.tool_calls[0].function.name == "get_weather");
}

TEST_CASE("seed_oss: escaped strings kept verbatim") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=test_function>\n"
      "<parameter=quoted>He said \"hello\"</parameter>\n"
      "<parameter=path>C:\\Users\\file.txt</parameter>\n"
      "<parameter=newline>line1\nline2</parameter>\n"
      "</function>\n</seed:tool_call>",
      req);
  nlohmann::json args = Parsed(r.tool_calls[0].function.arguments);
  CHECK(args["quoted"] == "He said \"hello\"");
  CHECK(args["path"] == "C:\\Users\\file.txt");
  CHECK(args["newline"] == "line1\nline2");
}

TEST_CASE("seed_oss: multiple parameters") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=search>\n"
      "<parameter=query>vllm parsing</parameter>\n"
      "<parameter=limit>10</parameter>\n"
      "<parameter=exact_match>false</parameter>\n"
      "</function>\n</seed:tool_call>",
      req);
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"query", "vllm parsing"},
                       {"limit", "10"},
                       {"exact_match", "false"}});
}

TEST_CASE("seed_oss: multiline parameter values") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=Bash>\n"
      "<parameter=command>\nls -la /tmp\n</parameter>\n"
      "<parameter=description>\nList files in /tmp directory\n</parameter>\n"
      "</function>\n</seed:tool_call>",
      req);
  nlohmann::json args = Parsed(r.tool_calls[0].function.arguments);
  CHECK(args["command"] == "ls -la /tmp");
  CHECK(args["description"] == "List files in /tmp directory");
}

TEST_CASE("seed_oss: nested JSON array parameter stays string") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=AskUserQuestion>\n"
      "<parameter=questions>[{\"question\": \"Pick a color\", "
      "\"multiSelect\": false, \"answer\": null}]</parameter>\n"
      "</function>\n</seed:tool_call>",
      req);
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"questions",
                        "[{\"question\": \"Pick a color\", "
                        "\"multiSelect\": false, \"answer\": null}]"}});
}

TEST_CASE("seed_oss: consecutive calls without closing tool tag") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = p.extract_tool_calls(
      "<seed:tool_call>\n<function=get_weather>\n"
      "<parameter=city>Tokyo</parameter>\n</function>\n"
      "<seed:tool_call>\n<function=get_weather>\n"
      "<parameter=city>Paris</parameter>\n</function>\n</seed:tool_call>",
      req);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(Parsed(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Tokyo"}});
  CHECK(Parsed(r.tool_calls[1].function.arguments) ==
        nlohmann::json{{"city", "Paris"}});
}

// ── Derived streaming ─────────────────────────────────────────────────────────

TEST_CASE("seed_oss stream: multi-param") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = Stream(p, {"<seed:tool_call>\n", "<function=get_weather>\n",
                      "<parameter=city>Tokyo</parameter>\n",
                      "<parameter=unit>celsius</parameter>\n", "</function>\n",
                      "</seed:tool_call>"},
                  req);
  CHECK(CollectName(r) == "get_weather");
  CHECK(Parsed(CollectArgs(r)) ==
        nlohmann::json{{"city", "Tokyo"}, {"unit", "celsius"}});
}

TEST_CASE("seed_oss stream: value split across chunks") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = Stream(p, {"<seed:tool_call>\n", "<function=search>\n",
                      "<parameter=query>hello ", "world", " test</parameter>\n",
                      "</function>\n", "</seed:tool_call>"},
                  req);
  CHECK(Parsed(CollectArgs(r))["query"] == "hello world test");
}

TEST_CASE("seed_oss stream: parallel calls") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = Stream(p, {"<seed:tool_call>\n", "<function=get_weather>\n",
                      "<parameter=city>Tokyo</parameter>\n", "</function>\n",
                      "</seed:tool_call>", "<seed:tool_call>\n",
                      "<function=get_time>\n", "<parameter=tz>JST</parameter>\n",
                      "</function>\n", "</seed:tool_call>"},
                  req);
  std::vector<std::string> names = CollectNames(r);
  bool has_weather = false, has_time = false;
  for (const auto& n : names) {
    if (n == "get_weather") has_weather = true;
    if (n == "get_time") has_time = true;
  }
  CHECK(has_weather);
  CHECK(has_time);
}

TEST_CASE("seed_oss stream: text before tool call") {
  SeedOssToolParser p;
  ChatCompletionRequest req;
  auto r = Stream(p, {"Let me check ", "the weather. ", "<seed:tool_call>\n",
                      "<function=get_weather>\n",
                      "<parameter=city>Tokyo</parameter>\n", "</function>\n",
                      "</seed:tool_call>"},
                  req);
  std::string content = CollectContent(r);
  std::size_t b = content.find_first_not_of(" \t\n");
  if (b != std::string::npos) content = content.substr(b);
  CHECK(content.rfind("Let me check", 0) == 0);
}

// ── Registration ──────────────────────────────────────────────────────────────

TEST_CASE("seed_oss: get_tool_parser resolves the registered name") {
  auto parser = get_tool_parser("seed_oss");
  REQUIRE(parser != nullptr);
  ChatCompletionRequest req;
  auto r = parser->extract_tool_calls(
      "<seed:tool_call>\n<function=get_weather>\n"
      "<parameter=city>Tokyo</parameter>\n</function>\n</seed:tool_call>",
      req);
  CHECK(r.tools_called);
  CHECK(r.tool_calls[0].function.name == "get_weather");
}
