// Tests for the Mistral OpenAI tool-call parser (mistral_tool_parser.py
// @ e24d1b24). Ports the upstream cases from
// tests/tool_parsers/test_mistral_tool_parser.py - the SAME fixture input
// strings and the SAME expected names/arguments/content.
//
// Two forms, selected by the constructor flag (deviation D1 in mistral.h):
//   - MistralToolParser(/*is_pre_v11=*/true):  [TOOL_CALLS] [{"name":..}]
//   - MistralToolParser(/*is_pre_v11=*/false): [TOOL_CALLS]name{args}   (v11)
//
// Argument-string whitespace (deviation D4): the pre-v11 form re-serializes
// arguments with nlohmann's compact dump() (upstream json.dumps adds ", "/": "
// separators) - semantically identical JSON. The v11 form keeps the raw
// argument substring verbatim, so those expectations retain the input spacing.
// Every argument object in the ported cases has alphabetical/single keys, so
// nlohmann's key-sorted compact output equals the upstream ordering.
#include <doctest/doctest.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/mistral.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

// Mirrors upstream assert_tool_calls' id checks: 9-char alphanumeric.
bool IsValidMistralId(const std::string& id) {
  if (id.size() != 9) return false;
  for (char c : id) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9');
    if (!ok) return false;
  }
  return true;
}

// ── Streaming driver: feed deltas one at a time, accumulate as serving_chat ──
struct StreamResult {
  std::string content;
  std::vector<std::string> names;             // names emitted (each once/tool)
  std::vector<std::string> args_by_index;     // args concatenated per index
  std::vector<std::string> ids_by_index;
  int max_index = -1;
  bool saw_multi_toolcall_in_one_message = false;
};

StreamResult DriveStream(MistralToolParser& parser,
                         const std::vector<std::string>& deltas) {
  StreamResult r;
  std::string previous;
  ChatCompletionRequest req;
  for (const std::string& delta : deltas) {
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (!dm.has_value()) continue;
    if (dm->content.has_value()) r.content += *dm->content;
    if (dm->tool_calls.has_value()) {
      if (dm->tool_calls->size() > 1) r.saw_multi_toolcall_in_one_message = true;
      for (const DeltaToolCall& tc : *dm->tool_calls) {
        const int idx = tc.index;
        r.max_index = std::max(r.max_index, idx);
        while (static_cast<int>(r.args_by_index.size()) <= idx) {
          r.args_by_index.emplace_back();
          r.ids_by_index.emplace_back();
        }
        if (tc.id.has_value() && r.ids_by_index[idx].empty())
          r.ids_by_index[idx] = *tc.id;
        if (tc.function.name.has_value()) r.names.push_back(*tc.function.name);
        if (tc.function.arguments.has_value())
          r.args_by_index[idx] += *tc.function.arguments;
      }
    }
  }
  return r;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// NON-STREAMING - no tools (test_extract_tool_calls_no_tools, both forms)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Mistral no_tools pre_v11: plain text") {
  MistralToolParser parser(/*is_pre_v11=*/true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls("This is a test", req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "This is a test");
}

TEST_CASE("Mistral no_tools v11: plain text") {
  MistralToolParser parser(/*is_pre_v11=*/false);
  auto req = empty_request();
  auto info = parser.extract_tool_calls("This is a test", req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "This is a test");
}

// ═══════════════════════════════════════════════════════════════════════════
// NON-STREAMING pre-v11 (test_extract_tool_calls_pre_v11_tokenizer, 7 cases)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Mistral pre_v11: single_tool_add") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5, "b": 4}}])", req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(IsValidMistralId(info.tool_calls[0].id));
  CHECK(info.tool_calls[0].function.name == "add");
  CHECK(info.tool_calls[0].function.arguments == R"({"a":3.5,"b":4})");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("Mistral pre_v11: single_tool_weather") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS] [{"name": "get_current_weather", "arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(info.tool_calls[0].function.arguments ==
        R"({"city":"San Francisco","state":"CA","unit":"celsius"})");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("Mistral pre_v11: argument_before_name") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS] [{"arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(info.tool_calls[0].function.arguments ==
        R"({"city":"San Francisco","state":"CA","unit":"celsius"})");
}

TEST_CASE("Mistral pre_v11: argument_before_name_and_name_in_argument") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS] [{"arguments":{"name": "John Doe"}, "name": "get_age"}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_age");
  CHECK(info.tool_calls[0].function.arguments == R"({"name":"John Doe"})");
}

TEST_CASE("Mistral pre_v11: multiple_tools") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS] [{"name": "add", "arguments": {"a": 3.5, "b": 4}}, {"name": "get_current_weather", "arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "add");
  CHECK(info.tool_calls[0].function.arguments == R"({"a":3.5,"b":4})");
  CHECK(info.tool_calls[1].function.name == "get_current_weather");
  CHECK(info.tool_calls[1].function.arguments ==
        R"({"city":"San Francisco","state":"CA","unit":"celsius"})");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("Mistral pre_v11: content_before_tool") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"(Hello[TOOL_CALLS] [{"name": "add", "arguments":{"a": 1, "b": 2}}])",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "add");
  CHECK(info.tool_calls[0].function.arguments == R"({"a":1,"b":2})");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Hello");
}

TEST_CASE("Mistral pre_v11: trailing_data_after_json") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      "[TOOL_CALLS] [{\"name\": \"get_current_weather\", \"arguments\":{\"city\": "
      "\"Dallas\", \"state\": \"TX\", \"unit\": \"fahrenheit\"}}]\nextra trailing data",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(info.tool_calls[0].function.arguments ==
        R"({"city":"Dallas","state":"TX","unit":"fahrenheit"})");
  CHECK_FALSE(info.content.has_value());
}

// pre-v11 error / fallback paths
// (test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises,
//  _regex_fallback, _regex_fallback_fails)

TEST_CASE("Mistral pre_v11: multiple BOT tokens raises") {
  MistralToolParser parser(true);
  auto req = empty_request();
  const std::string out =
      R"([TOOL_CALLS] [{"name": "add", "arguments":{"a": 1}}])"
      R"([TOOL_CALLS] [{"name": "sub", "arguments":{"b": 2}}])";
  CHECK_THROWS(parser.extract_tool_calls(out, req));
}

TEST_CASE("Mistral pre_v11: regex fallback recovers from leading junk") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS]  junk [{"name": "add", "arguments":{"a": 1, "b": 2}}] trail)",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "add");
  CHECK(info.tool_calls[0].function.arguments == R"({"a":1,"b":2})");
}

TEST_CASE("Mistral pre_v11: regex fallback fails -> content") {
  MistralToolParser parser(true);
  auto req = empty_request();
  auto info = parser.extract_tool_calls("[TOOL_CALLS] not json at all", req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "not json at all");
}

// ═══════════════════════════════════════════════════════════════════════════
// NON-STREAMING v11 (test_extract_tool_calls, 5 cases) - args kept verbatim
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Mistral v11: single_tool_add") {
  MistralToolParser parser(false);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4})", req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(IsValidMistralId(info.tool_calls[0].id));
  CHECK(info.tool_calls[0].function.name == "add_this_and_that");
  CHECK(info.tool_calls[0].function.arguments == R"({"a": 3.5, "b": 4})");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("Mistral v11: single_tool_weather") {
  MistralToolParser parser(false);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS]get_current_weather{"city": "San Francisco", "state": "CA", "unit": "celsius"})",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(info.tool_calls[0].function.arguments ==
        R"({"city": "San Francisco", "state": "CA", "unit": "celsius"})");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("Mistral v11: multiple_tool_calls") {
  MistralToolParser parser(false);
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      R"([TOOL_CALLS]add{"a": 3.5, "b": 4}[TOOL_CALLS]multiply{"a": 3, "b": 6})",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "add");
  CHECK(info.tool_calls[0].function.arguments == R"({"a": 3.5, "b": 4})");
  CHECK(info.tool_calls[1].function.name == "multiply");
  CHECK(info.tool_calls[1].function.arguments == R"({"a": 3, "b": 6})");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("Mistral v11: complex (unterminated args kept verbatim)") {
  MistralToolParser parser(false);
  auto req = empty_request();
  // hi{hi[TOOL_CALLS]bash{"command": "print(\"hello world!\")\nre.compile(r'{}')
  // (the \" and \n are literal JSON escapes in the model text; no closing "}).
  const std::string out =
      "hi{hi[TOOL_CALLS]bash{\"command\": \"print(\\\"hello world!\\\")\\nre.compile(r'{}')";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "bash");
  CHECK(info.tool_calls[0].function.arguments ==
        "{\"command\": \"print(\\\"hello world!\\\")\\nre.compile(r'{}')");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "hi{hi");
}

TEST_CASE("Mistral v11: wrong_json (kept verbatim as a tool call)") {
  MistralToolParser parser(false);
  auto req = empty_request();
  const std::string out =
      "hi{hi[TOOL_CALLS]bash{\"command\": \"print(\\\"hello world!\\\")\\nre.compile(r'{}')\"}";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "bash");
  CHECK(info.tool_calls[0].function.arguments ==
        "{\"command\": \"print(\\\"hello world!\\\")\\nre.compile(r'{}')\"}");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "hi{hi");
}

// test_extract_tool_calls_v11_without_args_skipped
TEST_CASE("Mistral v11: tool without args is skipped") {
  MistralToolParser parser(false);
  auto req = empty_request();
  auto info = parser.extract_tool_calls("[TOOL_CALLS]toolname_no_args", req);
  CHECK(info.tools_called == true);
  CHECK(info.tool_calls.empty());
  CHECK_FALSE(info.content.has_value());
}

// ═══════════════════════════════════════════════════════════════════════════
// STREAMING - single chunk (test_extract_tool_calls_streaming_one_chunk)
// The whole model_output is fed as ONE delta; each tool arrives as ONE
// DeltaToolCall carrying both name and arguments.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Mistral streaming one_chunk v11: single_tool_add") {
  MistralToolParser parser(false);
  const auto r = DriveStream(
      parser, {R"([TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4})"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add_this_and_that");
  REQUIRE(r.args_by_index.size() == 1);
  CHECK(r.args_by_index[0] == R"({"a": 3.5, "b": 4})");
  CHECK(IsValidMistralId(r.ids_by_index[0]));
  CHECK(r.content.empty());
}

TEST_CASE("Mistral streaming one_chunk v11: single_tool_weather") {
  MistralToolParser parser(false);
  const auto r = DriveStream(
      parser,
      {R"([TOOL_CALLS]get_current_weather{"city": "San Francisco", "state": "CA", "unit": "celsius"})"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_current_weather");
  CHECK(r.args_by_index[0] ==
        R"({"city": "San Francisco", "state": "CA", "unit": "celsius"})");
  CHECK(r.content.empty());
}

TEST_CASE("Mistral streaming one_chunk v11: multiple_tool_calls") {
  MistralToolParser parser(false);
  const auto r = DriveStream(
      parser,
      {R"([TOOL_CALLS]add{"a": 3.5, "b": 4}[TOOL_CALLS]multiply{"a": 3, "b": 6})"});
  REQUIRE(r.names.size() == 2);
  CHECK(r.names[0] == "add");
  CHECK(r.names[1] == "multiply");
  CHECK(r.args_by_index[0] == R"({"a": 3.5, "b": 4})");
  CHECK(r.args_by_index[1] == R"({"a": 3, "b": 6})");
  CHECK(r.max_index == 1);
  CHECK(r.content.empty());
}

TEST_CASE("Mistral streaming one_chunk v11: content_before_tool") {
  MistralToolParser parser(false);
  const auto r = DriveStream(
      parser, {R"(bla[TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4})"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add_this_and_that");
  CHECK(r.args_by_index[0] == R"({"a": 3.5, "b": 4})");
  CHECK(r.content == "bla");
}

TEST_CASE("Mistral streaming one_chunk v11: complex") {
  MistralToolParser parser(false);
  const std::string out =
      "hi{hi[TOOL_CALLS]bash{\"command\": \"print(\\\"hello world!\\\")\\nre.compile(r'{}')\"}";
  const auto r = DriveStream(parser, {out});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "bash");
  CHECK(r.args_by_index[0] ==
        "{\"command\": \"print(\\\"hello world!\\\")\\nre.compile(r'{}')\"}");
  CHECK(r.content == "hi{hi");
}

TEST_CASE("Mistral streaming one_chunk pre_v11: no_tools") {
  MistralToolParser parser(true);
  const auto r = DriveStream(parser, {"This is a test"});
  CHECK(r.names.empty());
  CHECK(r.content == "This is a test");
}

TEST_CASE("Mistral streaming one_chunk pre_v11: single_tool_add (spacey)") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser,
      {R"([TOOL_CALLS]  [ {"name":"add" , "arguments" : {"a": 3, "b": 4} } ])"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add");
  CHECK(r.args_by_index[0] == R"({"a":3,"b":4})");
  CHECK(r.content.empty());
}

TEST_CASE("Mistral streaming one_chunk pre_v11: single_tool_add_strings") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser, {R"([TOOL_CALLS] [{"name": "add", "arguments":{"a": "3", "b": "4"}}])"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add");
  CHECK(r.args_by_index[0] == R"({"a":"3","b":"4"})");
  CHECK(r.content.empty());
}

TEST_CASE("Mistral streaming one_chunk pre_v11: single_tool_weather") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser,
      {R"([TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}}])"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_current_weather");
  CHECK(r.args_by_index[0] ==
        R"({"city":"San Francisco","state":"CA","unit":"celsius"})");
  CHECK(r.content.empty());
}

TEST_CASE("Mistral streaming one_chunk pre_v11: argument_before_name") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser,
      {R"([TOOL_CALLS] [{"arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}])"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_current_weather");
  CHECK(r.args_by_index[0] ==
        R"({"city":"San Francisco","state":"CA","unit":"celsius"})");
}

TEST_CASE("Mistral streaming one_chunk pre_v11: arg_before_name_and_name_in_arg") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser, {R"([TOOL_CALLS] [{"arguments": {"name": "John Doe"}, "name": "get_age"}])"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_age");
  CHECK(r.args_by_index[0] == R"({"name":"John Doe"})");
}

TEST_CASE("Mistral streaming one_chunk pre_v11: multiple_tools") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser,
      {R"([TOOL_CALLS] [{"arguments": {"a": 3.5, "b": 4}, "name": "add"}, {"arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}])"});
  REQUIRE(r.names.size() == 2);
  CHECK(r.names[0] == "add");
  CHECK(r.names[1] == "get_current_weather");
  CHECK(r.args_by_index[0] == R"({"a":3.5,"b":4})");
  CHECK(r.args_by_index[1] ==
        R"({"city":"San Francisco","state":"CA","unit":"celsius"})");
  CHECK(r.max_index == 1);
}

TEST_CASE("Mistral streaming one_chunk pre_v11: content_before_tool") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser, {R"(Some text[TOOL_CALLS] [{"name": "add", "arguments":{"a": 1, "b": 2}}])"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add");
  CHECK(r.args_by_index[0] == R"({"a":1,"b":2})");
  CHECK(r.content == "Some text");
}

// ═══════════════════════════════════════════════════════════════════════════
// STREAMING - fragmented (test_extract_tool_calls_streaming*, incremental)
// Assert the cadence: name emitted in entirety exactly once, argument fragments
// concatenate to the full JSON string, index increments per tool, content
// preserved.  Deltas are fragmented so at most one tool is emitted per message.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Mistral streaming v11 fragmented: single_tool_add") {
  MistralToolParser parser(false);
  const auto r = DriveStream(
      parser, {"[TOOL_CALLS]add", "{\"a\": 3", ", \"b\": 4}"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add");
  CHECK(r.args_by_index[0] == R"({"a": 3, "b": 4})");
  CHECK(IsValidMistralId(r.ids_by_index[0]));
  CHECK(r.content.empty());
  CHECK_FALSE(r.saw_multi_toolcall_in_one_message);
}

TEST_CASE("Mistral streaming v11 fragmented: single_tool_add_strings") {
  MistralToolParser parser(false);
  const auto r = DriveStream(
      parser, {"[TOOL_CALLS]add_two", "_strings", "{\"a\": \"3\"", ", \"b\": \"4\"}"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add_two_strings");
  CHECK(r.args_by_index[0] == R"({"a": "3", "b": "4"})");
  CHECK(r.content.empty());
}

TEST_CASE("Mistral streaming v11 fragmented: multiple_tools index increments") {
  MistralToolParser parser(false);
  const auto r = DriveStream(
      parser, {"[TOOL_CALLS]add", "{\"a\": 3.5, \"b\": 4}",
               "[TOOL_CALLS]get_current_weather",
               "{\"city\": \"San Francisco\", \"state\": \"CA\", \"unit\": \"celsius\"}"});
  REQUIRE(r.names.size() == 2);
  CHECK(r.names[0] == "add");
  CHECK(r.names[1] == "get_current_weather");
  CHECK(r.args_by_index[0] == R"({"a": 3.5, "b": 4})");
  CHECK(r.args_by_index[1] ==
        R"({"city": "San Francisco", "state": "CA", "unit": "celsius"})");
  CHECK(r.max_index == 1);
  CHECK(r.content.empty());
  CHECK_FALSE(r.saw_multi_toolcall_in_one_message);
}

TEST_CASE("Mistral streaming pre_v11 fragmented: single_tool_add") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser, {"[TOOL_CALLS] [{\"name\": \"add\", ", "\"arguments\":{\"a\": 3, \"b\": 4}}", "]"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add");
  CHECK(r.args_by_index[0] == R"({"a":3,"b":4})");
  CHECK(IsValidMistralId(r.ids_by_index[0]));
  CHECK(r.content.empty());
  CHECK_FALSE(r.saw_multi_toolcall_in_one_message);
}

TEST_CASE("Mistral streaming pre_v11 fragmented: multiple_tools") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser,
      {"[TOOL_CALLS] [{\"arguments\": {\"a\": 3.5, \"b\": 4}, \"name\": \"add\"}, ",
       "{\"arguments\":{\"city\": \"San Francisco\", \"state\": \"CA\", \"unit\": \"celsius\"}, \"name\": \"get_current_weather\"}",
       "]"});
  REQUIRE(r.names.size() == 2);
  CHECK(r.names[0] == "add");
  CHECK(r.names[1] == "get_current_weather");
  CHECK(r.args_by_index[0] == R"({"a":3.5,"b":4})");
  CHECK(r.args_by_index[1] ==
        R"({"city":"San Francisco","state":"CA","unit":"celsius"})");
  CHECK(r.max_index == 1);
  CHECK(r.content.empty());
  CHECK_FALSE(r.saw_multi_toolcall_in_one_message);
}

TEST_CASE("Mistral streaming pre_v11 fragmented: trailing_data_after_json") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser,
      {"[TOOL_CALLS] [{\"name\": \"get_current_weather\", \"arguments\":{\"city\": \"Dallas\", \"state\": \"TX\", \"unit\": \"fahrenheit\"}}",
       "]", "\nextra trailing data"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_current_weather");
  CHECK(r.args_by_index[0] ==
        R"({"city":"Dallas","state":"TX","unit":"fahrenheit"})");
  CHECK(r.content == "\nextra trailing data");
}

TEST_CASE("Mistral streaming pre_v11 fragmented: content_before_tool") {
  MistralToolParser parser(true);
  const auto r = DriveStream(
      parser,
      {"Some ", "text", "[TOOL_CALLS] [{\"name\": \"add\", ", "\"arguments\":{\"a\": 1, \"b\": 2}}]"});
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "add");
  CHECK(r.args_by_index[0] == R"({"a":1,"b":2})");
  CHECK(r.content == "Some text");
}

// test_extract_tool_calls_streaming_v11_no_tools / _pre_v11 no_tools fragmented
TEST_CASE("Mistral streaming v11: plain content passes through") {
  MistralToolParser parser(false);
  const auto r = DriveStream(parser, {"This ", "is a ", "test"});
  CHECK(r.names.empty());
  CHECK(r.content == "This is a test");
}

TEST_CASE("Mistral streaming pre_v11: plain content passes through") {
  MistralToolParser parser(true);
  const auto r = DriveStream(parser, {"This ", "is a ", "test"});
  CHECK(r.names.empty());
  CHECK(r.content == "This is a test");
}

// ═══════════════════════════════════════════════════════════════════════════
// Factory registration + malformed-JSON streaming guard
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Factory: get_tool_parser(\"mistral\") returns the v11 parser") {
  auto parser = get_tool_parser("mistral");
  REQUIRE(parser != nullptr);
  auto req = empty_request();
  auto info = parser->extract_tool_calls(R"([TOOL_CALLS]add{"a": 1})", req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "add");
}

// mistral_tool_parser.py:411-413 - a parse error inside the streaming helper is
// swallowed and the delta dropped (returns nullopt). A malformed object never
// gets emitted.
TEST_CASE("Mistral streaming pre_v11: malformed object never emits a tool") {
  MistralToolParser parser(true);
  auto req = empty_request();
  const std::string cur = R"([TOOL_CALLS] [{"name": "a", "arguments": nope}])";
  auto dm = parser.extract_tool_calls_streaming("", cur, cur, req);
  // No valid tool object -> either nullopt or a message without tool calls.
  if (dm.has_value()) {
    CHECK((!dm->tool_calls.has_value() || dm->tool_calls->empty()));
  }
}
