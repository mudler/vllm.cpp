// Ported from: vllm/tests/tool_parsers/test_poolside_v1_tool_parser.py @ e24d1b24
//
// Ports the upstream cases that exercise extract_tool_calls / the streaming state
// machine:
//   test_string_arg_preserves_whitespace          (string arg kept verbatim)
//   test_non_string_arg_still_deserialized         (non-string stripped+parsed)
//   test_responses_extract_tool_calls_with_flat_tools (adapted: same assertion
//       through a ChatCompletion tool - the Responses flat-FunctionTool shape is
//       not part of this seam; the behaviour under test is that _is_string_type
//       does not raise and the string arg survives)
//   test_no_newline_after_name_non_streaming
//   test_newline_after_name_still_parses_non_streaming
//   test_no_newline_after_name_streaming            (char-by-char)
//   test_streaming_responses_request_without_logprobs      (adapted -> logprobs=false)
//   test_streaming_responses_request_with_logprobs_emits_empty_delta
//                                                   (adapted -> logprobs=true)
//
// NOT ported: the 5 adjust_request / structured_outputs cases
// (test_required/named_skips_structured_outputs_{chatcompletion,responses},
// test_auto_still_keeps_special_tokens). adjust_request only toggles
// skip_special_tokens / structured_outputs - serving-layer state the text-only
// abstract.h seam does not model (there is no adjust_request hook, no
// structured_outputs field). See poolside_v1.h DEVIATION note.
//
// Argument strings compared as PARSED JSON (compact-dump deviation inherited from
// the seam); string values compared byte-for-byte since whitespace is load-bearing.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/poolside_v1.h"

using namespace vllm::entrypoints::openai;

namespace {

nlohmann::json ParsedArgs(const std::string& a) { return nlohmann::json::parse(a); }

// write_file: string `content`, integer `mode`.
ChatCompletionRequest WriteFileReq(const std::string& tool_choice = "auto") {
  ChatCompletionRequest req;
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = "write_file";
  t.function.parameters = nlohmann::json::parse(R"({
    "type": "object",
    "properties": {"content": {"type": "string"}, "mode": {"type": "integer"}},
    "required": ["content"]
  })");
  req.tools = std::vector<ChatCompletionToolsParam>{t};
  req.tool_choice = ToolChoice{tool_choice, std::nullopt};
  return req;
}

// get_weather: string `city`.
ChatCompletionRequest WeatherReq(const std::string& tool_choice = "auto") {
  ChatCompletionRequest req;
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = "get_weather";
  t.function.parameters = nlohmann::json::parse(R"({
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  })");
  req.tools = std::vector<ChatCompletionToolsParam>{t};
  req.tool_choice = ToolChoice{tool_choice, std::nullopt};
  return req;
}

struct Reconstructor {
  std::vector<ToolCall> tool_calls;
  std::string other_content;

  void Append(const DeltaMessage& d) {
    if (d.content.has_value()) other_content += *d.content;
    if (!d.tool_calls.has_value()) return;
    for (const DeltaToolCall& tc : *d.tool_calls) {
      const std::size_t idx = static_cast<std::size_t>(tc.index);
      if (idx < tool_calls.size()) {
        if (tc.function.arguments.has_value())
          tool_calls[idx].function.arguments += *tc.function.arguments;
        if (tc.function.name.has_value())
          tool_calls[idx].function.name = *tc.function.name;
      } else {
        ToolCall c;
        c.id = tc.id.value_or("");
        c.function.name = tc.function.name.value_or("");
        c.function.arguments = tc.function.arguments.value_or("");
        tool_calls.push_back(std::move(c));
      }
    }
  }
};

Reconstructor DriveChars(PoolsideV1ToolParser& p, const std::string& full,
                         const ChatCompletionRequest& req) {
  Reconstructor r;
  std::string prev;
  for (char ch : full) {
    const std::string delta(1, ch);
    const std::string cur = prev + delta;
    auto dm = p.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (dm.has_value()) r.Append(*dm);
  }
  return r;
}

}  // namespace

TEST_CASE("poolside_v1: string arg preserves whitespace") {
  PoolsideV1ToolParser parser;
  ChatCompletionRequest req = WriteFileReq();
  const std::string content = "    def f():\n        return 1\n";
  const std::string out =
      "<tool_call>write_file\n<arg_key>content</arg_key>\n<arg_value>" + content +
      "</arg_value>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(ParsedArgs(info.tool_calls[0].function.arguments).at("content") == content);
}

TEST_CASE("poolside_v1: non-string arg still deserialized") {
  PoolsideV1ToolParser parser;
  ChatCompletionRequest req = WriteFileReq();
  const std::string out =
      "<tool_call>write_file\n"
      "<arg_key>content</arg_key>\n<arg_value>hi</arg_value>\n"
      "<arg_key>mode</arg_key>\n<arg_value> 420 </arg_value>\n"
      "</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = ParsedArgs(info.tool_calls[0].function.arguments);
  CHECK(args.at("content") == "hi");
  CHECK(args.at("mode") == 420);
}

TEST_CASE("poolside_v1: extract with (flat-tool equivalent) string arg") {
  // Adapted from test_responses_extract_tool_calls_with_flat_tools: the seam has
  // only the ChatCompletion tool shape; the behaviour under test is that the
  // string arg survives verbatim and _is_string_type does not raise.
  PoolsideV1ToolParser parser;
  ChatCompletionRequest req = WriteFileReq("required");
  const std::string content = "  x = 1\n";
  const std::string out =
      "<tool_call>write_file\n<arg_key>content</arg_key>\n<arg_value>" + content +
      "</arg_value>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(ParsedArgs(info.tool_calls[0].function.arguments).at("content") == content);
}

TEST_CASE("poolside_v1: no newline after name (non-streaming)") {
  PoolsideV1ToolParser parser;
  ChatCompletionRequest req = WeatherReq();
  const std::string out =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
      "</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(ParsedArgs(info.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Paris"}});
}

TEST_CASE("poolside_v1: newline after name still parses (non-streaming)") {
  PoolsideV1ToolParser parser;
  ChatCompletionRequest req = WeatherReq();
  const std::string out =
      "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
      "</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(ParsedArgs(info.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Paris"}});
}

TEST_CASE("poolside_v1: no newline after name (streaming, char-by-char)") {
  PoolsideV1ToolParser parser;
  ChatCompletionRequest req = WeatherReq();
  const std::string out =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
      "</tool_call>";
  Reconstructor r = DriveChars(parser, out, req);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Paris"}});
  CHECK(r.other_content.empty());
}

TEST_CASE("poolside_v1: partial start token without logprobs -> nullopt") {
  PoolsideV1ToolParser parser;
  ChatCompletionRequest req = WriteFileReq();
  req.logprobs = false;
  const std::string first(1, std::string("<tool_call>")[0]);  // "<"
  auto dm = parser.extract_tool_calls_streaming("", first, first, req);
  CHECK_FALSE(dm.has_value());
}

TEST_CASE("poolside_v1: partial start token with logprobs -> empty content delta") {
  PoolsideV1ToolParser parser;
  ChatCompletionRequest req = WriteFileReq();
  req.logprobs = true;
  const std::string first(1, std::string("<tool_call>")[0]);  // "<"
  auto dm = parser.extract_tool_calls_streaming("", first, first, req);
  REQUIRE(dm.has_value());
  REQUIRE(dm->content.has_value());
  CHECK(*dm->content == "");
}
