// Ported from: vllm/tests/tool_parsers/test_functiongemma_tool_parser.py @ e24d1b24
//
// Ports every upstream case for the `functiongemma` parser:
//   TestExtractToolCalls (5)  - non-streaming extract_tool_calls
//   TestParseArguments   (6)  - the _parse_arguments helper (ParseArguments)
//   TestBufferDeltaText  (2)  - the _buffer_delta_text helper (BufferDeltaText)
// The 2 TestAdjustRequest cases are DELIBERATELY NOT ported: adjust_request only
// forces skip_special_tokens=False, a serving/tokenizer concern the text-only
// abstract.h seam does not model (there is no adjust_request hook). See the
// functiongemma.h header DEVIATION note.
//
// Plus streaming split-edge cases (no upstream streaming e2e test exists for this
// parser) driving the incremental state machine across delta boundaries.
//
// Argument strings are compared as PARSED JSON (nlohmann): the compact-dump
// (no space after ':'/',') deviation is inherited from the seam; values and key
// order are otherwise identical to upstream.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/functiongemma.h"

using namespace vllm::entrypoints::openai;

namespace {

nlohmann::json ParsedArgs(const std::string& args) {
  return nlohmann::json::parse(args);
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

Reconstructor DriveDeltas(FunctionGemmaToolParser& p,
                          const std::vector<std::string>& deltas,
                          const ChatCompletionRequest& req) {
  Reconstructor r;
  std::string prev;
  for (const std::string& delta : deltas) {
    const std::string cur = prev + delta;
    auto dm = p.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (dm.has_value()) r.Append(*dm);
  }
  return r;
}

}  // namespace

// --- TestExtractToolCalls -----------------------------------------------------

TEST_CASE("functiongemma: no tool calls is plain content") {
  FunctionGemmaToolParser parser;
  ChatCompletionRequest req;
  const std::string out = "Hello, how can I help you today?";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("functiongemma: single tool call") {
  FunctionGemmaToolParser parser;
  ChatCompletionRequest req;
  const std::string out =
      "<start_function_call>call:get_weather{location:<escape>London<escape>}"
      "<end_function_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(ParsedArgs(info.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "London"}});
}

TEST_CASE("functiongemma: multiple arguments") {
  FunctionGemmaToolParser parser;
  ChatCompletionRequest req;
  const std::string out =
      "<start_function_call>call:get_weather{"
      "location:<escape>San Francisco<escape>,"
      "unit:<escape>celsius<escape>}"
      "<end_function_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(ParsedArgs(info.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "San Francisco"}, {"unit", "celsius"}});
}

TEST_CASE("functiongemma: text before tool call") {
  FunctionGemmaToolParser parser;
  ChatCompletionRequest req;
  const std::string out =
      "Let me check the weather for you. "
      "<start_function_call>call:get_weather{location:<escape>Paris<escape>}"
      "<end_function_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Let me check the weather for you.");
}

TEST_CASE("functiongemma: multiple tool calls") {
  FunctionGemmaToolParser parser;
  ChatCompletionRequest req;
  const std::string out =
      "<start_function_call>call:get_weather{location:<escape>London<escape>}"
      "<end_function_call>"
      "<start_function_call>call:get_time{timezone:<escape>UTC<escape>}"
      "<end_function_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[1].function.name == "get_time");
}

// --- TestParseArguments -------------------------------------------------------

TEST_CASE("functiongemma: parse empty arguments") {
  FunctionGemmaToolParser parser;
  CHECK(parser.ParseArguments("").empty());
}

TEST_CASE("functiongemma: parse single string argument") {
  FunctionGemmaToolParser parser;
  auto r = parser.ParseArguments("city:<escape>Tokyo<escape>");
  CHECK(ParsedArgs(r.dump()) == nlohmann::json{{"city", "Tokyo"}});
}

TEST_CASE("functiongemma: parse multiple arguments") {
  FunctionGemmaToolParser parser;
  auto r =
      parser.ParseArguments("city:<escape>Tokyo<escape>,country:<escape>Japan<escape>");
  CHECK(ParsedArgs(r.dump()) ==
        nlohmann::json{{"city", "Tokyo"}, {"country", "Japan"}});
}

TEST_CASE("functiongemma: parse numeric argument") {
  FunctionGemmaToolParser parser;
  auto r = parser.ParseArguments("count:<escape>42<escape>");
  CHECK(ParsedArgs(r.dump()) == nlohmann::json{{"count", 42}});
}

TEST_CASE("functiongemma: parse boolean argument") {
  FunctionGemmaToolParser parser;
  auto r = parser.ParseArguments("enabled:<escape>true<escape>");
  CHECK(ParsedArgs(r.dump()) == nlohmann::json{{"enabled", true}});
}

TEST_CASE("functiongemma: parse argument with spaces") {
  FunctionGemmaToolParser parser;
  auto r = parser.ParseArguments("message:<escape>Hello World<escape>");
  CHECK(ParsedArgs(r.dump()) == nlohmann::json{{"message", "Hello World"}});
}

// --- TestBufferDeltaText ------------------------------------------------------

TEST_CASE("functiongemma: regular text not buffered") {
  FunctionGemmaToolParser parser;
  const std::string r = parser.BufferDeltaText("hello");
  CHECK(r == "hello");
  CHECK(parser.buffered_delta_text().empty());
}

TEST_CASE("functiongemma: complete tag flushed") {
  FunctionGemmaToolParser parser;
  // Prime the buffer with a partial tag, then complete it.
  parser.BufferDeltaText("<start_function_");
  const std::string r = parser.BufferDeltaText("call>");
  CHECK(r.find("<start_function_call>") != std::string::npos);
}

// --- streaming split-edge cases (new) -----------------------------------------

TEST_CASE("functiongemma: streaming single call across deltas") {
  FunctionGemmaToolParser parser;
  ChatCompletionRequest req;
  const std::vector<std::string> deltas = {
      "<start_function_call>call:get_weather{",
      "location:<escape>London<escape>}",
      "<end_function_call>"};
  Reconstructor r = DriveDeltas(parser, deltas, req);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"location", "London"}});
}

TEST_CASE("functiongemma: streaming plain content before any call") {
  FunctionGemmaToolParser parser;
  ChatCompletionRequest req;
  const std::vector<std::string> deltas = {"Sure, ", "let me check."};
  Reconstructor r = DriveDeltas(parser, deltas, req);
  CHECK(r.tool_calls.empty());
  CHECK(r.other_content == "Sure, let me check.");
}
