// Ported from: vllm/tests/tool_parsers/test_llama3_json_tool_parser.py
// @ e24d1b24
//
// Ports all 15 upstream (non-streaming) cases for the llama3_json / llama4_json
// parser. The upstream file ships NO streaming tests, so the streaming section
// below is ADDITIONAL coverage for the ported extract_tool_calls_streaming
// (name-first delta, then argument fragments that reconstruct the compact JSON).
//
// Arguments are re-serialized compact (no space after ':'/','), the documented
// whitespace deviation (see llama.h). Where upstream re-parses arguments with
// json.loads and asserts on VALUES, we do the same (order-independent). Where it
// asserts substring presence, we assert the exact compact form.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/llama.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

nlohmann::json ParseArgs(const std::string& s) { return nlohmann::json::parse(s); }

struct Reconstructor {
  std::vector<ToolCall> tool_calls;
  std::string other_content;

  void Append(const DeltaMessage& d) {
    if (d.content.has_value()) other_content += *d.content;
    if (!d.tool_calls.has_value()) return;
    for (const DeltaToolCall& tc : *d.tool_calls) {
      const std::size_t idx = static_cast<std::size_t>(tc.index);
      if (idx < tool_calls.size()) {
        if (tc.function.arguments.has_value()) {
          tool_calls[idx].function.arguments += *tc.function.arguments;
        }
      } else {
        REQUIRE(idx == tool_calls.size());
        REQUIRE(tc.id.has_value());
        REQUIRE(tc.function.name.has_value());
        ToolCall c;
        c.id = *tc.id;
        c.function.name = *tc.function.name;
        c.function.arguments = tc.function.arguments.value_or("");
        tool_calls.push_back(std::move(c));
      }
    }
  }
};

Reconstructor DriveDeltas(ToolParser& p, const std::vector<std::string>& deltas) {
  Reconstructor r;
  std::string prev;
  ChatCompletionRequest req;
  for (const std::string& delta : deltas) {
    const std::string cur = prev + delta;
    auto dm = p.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (dm.has_value()) r.Append(*dm);
  }
  return r;
}

// The <|python_tag|> bot token is a single special token -> one delta; the JSON
// body is streamed char by char.
std::vector<std::string> BotAwareDeltas(const std::string& full) {
  const std::string bot = LlamaToolParser::kBotToken;
  std::vector<std::string> out;
  std::size_t i = 0;
  if (full.compare(0, bot.size(), bot) == 0) {
    out.push_back(bot);
    i = bot.size();
  }
  for (; i < full.size(); ++i) out.emplace_back(1, full[i]);
  return out;
}

}  // namespace

// (1) upstream test_extract_tool_calls_simple
TEST_CASE("llama: simple with surrounding text") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "Here is the result: {\"name\": \"getOpenIncidentsTool\", "
      "\"parameters\": {}} Would you like to know more?";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "getOpenIncidentsTool");
  CHECK(info.tool_calls[0].function.arguments == "{}");
  CHECK_FALSE(info.content.has_value());
}

// (2) upstream test_extract_tool_calls_with_arguments
TEST_CASE("llama: with arguments (parameters key)") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"searchTool\", \"parameters\": {\"query\": \"test query\", "
      "\"limit\": 10}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "searchTool");
  CHECK(info.tool_calls[0].function.arguments ==
        R"({"query":"test query","limit":10})");
}

// (3) upstream test_extract_tool_calls_no_json
TEST_CASE("llama: no json is plain content") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out = "This is just some text without any tool calls";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// (4) upstream test_extract_tool_calls_invalid_json
TEST_CASE("llama: invalid json falls back to content") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"invalidTool\", \"parameters\": {invalid json}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// (5) upstream test_extract_tool_calls_with_arguments_key
TEST_CASE("llama: with arguments key") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"searchTool\", \"arguments\": {\"query\": \"test\"}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "searchTool");
  CHECK(info.tool_calls[0].function.arguments == R"({"query":"test"})");
}

// (6) upstream test_extract_tool_calls_multiple_json
TEST_CASE("llama: multiple json separated by semicolons") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"searchTool\", \"parameters\": {\"query\": \"test1\"}}; "
      "{\"name\": \"getOpenIncidentsTool\", \"parameters\": {}}; "
      "{\"name\": \"searchTool\", \"parameters\": {\"query\": \"test2\"}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 3);
  CHECK(info.tool_calls[0].function.name == "searchTool");
  CHECK(info.tool_calls[0].function.arguments == R"({"query":"test1"})");
  CHECK(info.tool_calls[1].function.name == "getOpenIncidentsTool");
  CHECK(info.tool_calls[1].function.arguments == "{}");
  CHECK(info.tool_calls[2].function.name == "searchTool");
  CHECK(info.tool_calls[2].function.arguments == R"({"query":"test2"})");
}

// (7) upstream test_extract_tool_calls_multiple_json_with_whitespace
TEST_CASE("llama: multiple json with whitespace before semicolons") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"searchTool\", \"parameters\": {\"query\": \"test1\"}} ; "
      "{\"name\": \"getOpenIncidentsTool\", \"parameters\": {}} ; "
      "{\"name\": \"searchTool\", \"parameters\": {\"query\": \"test2\"}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 3);
  CHECK(info.tool_calls[0].function.name == "searchTool");
  CHECK(info.tool_calls[1].function.name == "getOpenIncidentsTool");
  CHECK(info.tool_calls[2].function.name == "searchTool");
}

// (8) upstream test_extract_tool_calls_multiple_json_with_surrounding_text
TEST_CASE("llama: multiple json with surrounding text") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "Here are the results: "
      "{\"name\": \"searchTool\", \"parameters\": {\"query\": \"test1\"}}; "
      "{\"name\": \"getOpenIncidentsTool\", \"parameters\": {}}; "
      "{\"name\": \"searchTool\", \"parameters\": {\"query\": \"test2\"}} "
      "Would you like to know more?";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 3);
  CHECK(info.tool_calls[0].function.name == "searchTool");
  CHECK(info.tool_calls[1].function.name == "getOpenIncidentsTool");
  CHECK(info.tool_calls[2].function.name == "searchTool");
}

// (9) upstream test_extract_tool_calls_deeply_nested_json
TEST_CASE("llama: deeply nested json parameters") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"complexTool\", \"parameters\": {\"level1\": {\"level2\": "
      "{\"level3\": {\"level4\": {\"value\": \"deep\"}}}}}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "complexTool");
  const nlohmann::json args = ParseArgs(info.tool_calls[0].function.arguments);
  CHECK(args["level1"]["level2"]["level3"]["level4"]["value"] == "deep");
}

// (10) upstream test_extract_tool_calls_multiple_with_deep_nesting
TEST_CASE("llama: multiple tool calls with deep nesting") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"simpleTool\", \"parameters\": {\"value\": \"test\"}}; "
      "{\"name\": \"complexTool\", \"parameters\": {\"config\": {\"database\": "
      "{\"connection\": {\"pool\": {\"size\": 10}}}}}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "simpleTool");
  const nlohmann::json a0 = ParseArgs(info.tool_calls[0].function.arguments);
  CHECK(a0["value"] == "test");
  CHECK(info.tool_calls[1].function.name == "complexTool");
  const nlohmann::json a1 = ParseArgs(info.tool_calls[1].function.arguments);
  CHECK(a1["config"]["database"]["connection"]["pool"]["size"] == 10);
}

// (11) upstream test_extract_tool_calls_with_quotes_and_brackets_in_string
TEST_CASE("llama: quotes and brackets inside string values") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"searchTool\", \"parameters\": {\"query\": \"test {value} "
      "[complex]\",\"nested\": {\"inner\": \"more {brackets}\"}}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "searchTool");
  const nlohmann::json args = ParseArgs(info.tool_calls[0].function.arguments);
  CHECK(args["query"] == "test {value} [complex]");
  CHECK(args["nested"]["inner"] == "more {brackets}");
}

// (12) upstream test_extract_tool_calls_with_escaped_quotes_in_nested_json
TEST_CASE("llama: escaped quotes in nested json") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out =
      "{\"name\": \"parserTool\", \"parameters\": {\"text\": \"He said "
      "\\\"Hello {world}\\\"\"}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "parserTool");
  const nlohmann::json args = ParseArgs(info.tool_calls[0].function.arguments);
  CHECK(args["text"] == "He said \"Hello {world}\"");
}

// (13) upstream test_extract_tool_calls_missing_name_key
TEST_CASE("llama: missing name key falls back to content") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out = "{\"parameters\": {}}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// (14) upstream test_extract_tool_calls_missing_parameters_and_arguments_key
TEST_CASE("llama: missing parameters and arguments keys falls back") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out = "{\"name\": \"toolWithoutParams\"}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// (15) upstream test_regex_timeout_handling - behavioral port (malformed input
// treated as content; no Python `regex` timeout analogue).
TEST_CASE("llama: malformed input falls back to content") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string bad = "{hello world[A(A=\t)A(A=,\t)A(A=,\t";
  auto info = parser.extract_tool_calls(bad, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == bad);
}

// bot-token quick-check: <|python_tag|> with no '{' and no braces still routes
// through the fast path (no tool call, plain content).
TEST_CASE("llama: bot token alone is content") {
  LlamaToolParser parser;
  auto req = empty_request();
  const std::string out = "<|python_tag|>no call here";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// -- Streaming (ADDITIONAL - no upstream streaming tests for this parser) ------

TEST_CASE("llama streaming: single call with bot token") {
  LlamaToolParser parser;
  const std::string full =
      "<|python_tag|>{\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Paris\"}}";
  Reconstructor r = DriveDeltas(parser, BotAwareDeltas(full));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(r.tool_calls[0].function.arguments == R"({"city":"Paris"})");
}

TEST_CASE("llama streaming: single call, bare json (no bot token)") {
  LlamaToolParser parser;
  const std::string full =
      "{\"name\": \"search\", \"arguments\": {\"q\": \"cats\"}}";
  Reconstructor r = DriveDeltas(parser, BotAwareDeltas(full));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "search");
  CHECK(r.tool_calls[0].function.arguments == R"({"q":"cats"})");
}

TEST_CASE("llama streaming: parameters key mapped to arguments") {
  LlamaToolParser parser;
  const std::string full =
      "<|python_tag|>{\"name\": \"f\", \"parameters\": {\"p\": \"v\"}}";
  Reconstructor r = DriveDeltas(parser, BotAwareDeltas(full));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "f");
  CHECK(r.tool_calls[0].function.arguments == R"({"p":"v"})");
}

TEST_CASE("llama streaming: two calls separated by semicolons") {
  LlamaToolParser parser;
  const std::string full =
      "<|python_tag|>{\"name\": \"a\", \"arguments\": {\"x\": \"1\"}}; "
      "{\"name\": \"b\", \"arguments\": {\"y\": \"2\"}}";
  Reconstructor r = DriveDeltas(parser, BotAwareDeltas(full));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "a");
  CHECK(r.tool_calls[0].function.arguments == R"({"x":"1"})");
  CHECK(r.tool_calls[1].function.name == "b");
  CHECK(r.tool_calls[1].function.arguments == R"({"y":"2"})");
}

TEST_CASE("llama streaming: plain content passes through") {
  LlamaToolParser parser;
  const std::string full = "The weather is sunny.";
  std::vector<std::string> deltas;
  for (char c : full) deltas.emplace_back(1, c);
  Reconstructor r = DriveDeltas(parser, deltas);
  CHECK(r.tool_calls.empty());
  CHECK(r.other_content == full);
}
