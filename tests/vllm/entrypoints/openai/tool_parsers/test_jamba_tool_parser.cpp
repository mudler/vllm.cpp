// Tests for the "jamba" tool-call parser (JambaToolParser).
// Ported from: tests/tool_parsers/test_jamba_tool_parser.py @ e24d1b24 - the SAME
// fixture inputs and expected names/arguments/content for BOTH the non-streaming
// (no_tools / single / single_with_content / parallel) and streaming variants.
//
// MAJOR DEVIATION (recorded in jamba.h): upstream reads the tokenizer vocab +
// delta_token_ids to detect the <tool_calls> control token; this port is fully
// text-based. The streaming below is therefore driven with the <tool_calls> /
// </tool_calls> markers as ATOMIC deltas (they are single special tokens
// upstream) plus char-by-char JSON body, then reconstructed the way the upstream
// test's StreamingToolReconstructor does (per-index name + argument fragments,
// completed with the tolerant partial-JSON completer).
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/jamba.h"
#include "vllm/entrypoints/openai/tool_parsers/utils.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

struct StreamResult {
  std::string content;
  std::vector<std::string> names;
  std::vector<std::string> args_by_index;
  bool saw_multi_in_one_message = false;
};

StreamResult DriveStream(JambaToolParser& parser,
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
      if (dm->tool_calls->size() > 1) r.saw_multi_in_one_message = true;
      for (const DeltaToolCall& tc : *dm->tool_calls) {
        const int idx = tc.index;
        while (static_cast<int>(r.args_by_index.size()) <= idx)
          r.args_by_index.emplace_back();
        if (tc.function.name.has_value()) r.names.push_back(*tc.function.name);
        if (tc.function.arguments.has_value())
          r.args_by_index[idx] += *tc.function.arguments;
      }
    }
  }
  return r;
}

// Split a JSON body into deltas the way a real tokenizer would: each quoted
// string literal (with escapes) is ONE token, everything else a single char.
// The streaming argument-diff's first-fragment branch keys off
// `cur_json.index(delta_text)`, which only lands correctly for multi-char tokens
// (this is the text-only stand-in for upstream's real tokenizer cadence).
std::vector<std::string> TokenizeJson(const std::string& body) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < body.size()) {
    if (body[i] == '"') {
      // Opening quote + content as one token; the CLOSING quote as its own token
      // (matching how BPE tokenizers separate a value's terminating quote). This
      // is load-bearing: the streaming first-fragment branch emits up to and
      // including the delta, so the closing quote must arrive separately or a
      // multi-key object reconstructs with a doubled quote.
      std::size_t j = i + 1;
      while (j < body.size()) {
        if (body[j] == '\\') {
          j += 2;
        } else if (body[j] == '"') {
          break;
        } else {
          ++j;
        }
      }
      out.push_back(body.substr(i, j - i));  // "opening + content (no close)
      if (j < body.size()) {
        out.emplace_back("\"");
        i = j + 1;
      } else {
        i = j;
      }
    } else {
      out.emplace_back(1, body[i]);
      ++i;
    }
  }
  return out;
}

// deltas: `leading` char-by-char, then <tool_calls> atomic, then the tokenized
// `body`, then </tool_calls> atomic.
std::vector<std::string> WrapDeltas(const std::string& leading,
                                    const std::string& body) {
  std::vector<std::string> d;
  for (char c : leading) d.emplace_back(1, c);
  d.emplace_back("<tool_calls>");
  for (const std::string& t : TokenizeJson(body)) d.push_back(t);
  d.emplace_back("</tool_calls>");
  return d;
}

// Complete a possibly-partial streamed argument string and compare to expected
// (mirrors upstream partial_json_parser.ensure_json(Allow.OBJ|Allow.STR)).
bool ArgsEqual(const std::string& streamed, const std::string& expected) {
  nlohmann::ordered_json got = partial_json_loads(streamed, /*allow_partial_str=*/true).first;
  return got == nlohmann::ordered_json::parse(expected);
}

}  // namespace

// ═══ NON-STREAMING ═══════════════════════════════════════════════════════════

TEST_CASE("jamba: no tool calls is plain content") {
  JambaToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls("This is a test", req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "This is a test");
}

TEST_CASE("jamba: single tool (leading space -> content None)") {
  JambaToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      " <tool_calls>[\n    {\"name\": \"get_current_weather\", \"arguments\": "
      "{\"city\": \"Dallas\", \"state\": \"TX\", \"unit\": \"fahrenheit\"}}\n"
      "]</tool_calls>",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].id.size() > 16);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(info.tool_calls[0].function.arguments ==
        R"({"city":"Dallas","state":"TX","unit":"fahrenheit"})");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("jamba: single tool with content") {
  JambaToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      " Sure! let me call the tool for you.<tool_calls>[\n    {\"name\": "
      "\"get_current_weather\", \"arguments\": {\"city\": \"Dallas\", \"state\": "
      "\"TX\", \"unit\": \"fahrenheit\"}}\n]</tool_calls>",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == " Sure! let me call the tool for you.");
}

TEST_CASE("jamba: parallel tools") {
  JambaToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(
      " <tool_calls>[\n    {\"name\": \"get_current_weather\", \"arguments\": "
      "{\"city\": \"Dallas\", \"state\": \"TX\", \"unit\": \"fahrenheit\"}},\n"
      "    {\"name\": \"get_current_weather\", \"arguments\": {\"city\": "
      "\"Orlando\", \"state\": \"FL\", \"unit\": \"fahrenheit\"}}\n]</tool_calls>",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.arguments ==
        R"({"city":"Dallas","state":"TX","unit":"fahrenheit"})");
  CHECK(info.tool_calls[1].function.arguments ==
        R"({"city":"Orlando","state":"FL","unit":"fahrenheit"})");
  CHECK_FALSE(info.content.has_value());
}

// ═══ STREAMING ═══════════════════════════════════════════════════════════════

TEST_CASE("jamba streaming: no tools passes content through") {
  JambaToolParser parser;
  const auto r = DriveStream(parser, {"This ", "is a ", "test"});
  CHECK(r.names.empty());
  CHECK(r.content == "This is a test");
}

TEST_CASE("jamba streaming: single tool") {
  JambaToolParser parser;
  const auto r = DriveStream(
      parser,
      WrapDeltas(
          " ",
          "[\n    {\"name\": \"get_current_weather\", \"arguments\": {\"city\": "
          "\"Dallas\", \"state\": \"TX\", \"unit\": \"fahrenheit\"}}\n]"));
  CHECK(r.content == " ");
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_current_weather");
  REQUIRE(r.args_by_index.size() == 1);
  CHECK(ArgsEqual(r.args_by_index[0],
                  R"({"city":"Dallas","state":"TX","unit":"fahrenheit"})"));
  CHECK_FALSE(r.saw_multi_in_one_message);
}

TEST_CASE("jamba streaming: single tool with content") {
  JambaToolParser parser;
  const auto r = DriveStream(
      parser,
      WrapDeltas(
          " Sure! let me call the tool for you.",
          "[\n    {\"name\": \"get_current_weather\", \"arguments\": {\"city\": "
          "\"Dallas\", \"state\": \"TX\", \"unit\": \"fahrenheit\"}}\n]"));
  CHECK(r.content == " Sure! let me call the tool for you.");
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_current_weather");
  CHECK(ArgsEqual(r.args_by_index[0],
                  R"({"city":"Dallas","state":"TX","unit":"fahrenheit"})"));
}

TEST_CASE("jamba streaming: parallel tools") {
  JambaToolParser parser;
  const auto r = DriveStream(
      parser,
      WrapDeltas(
          " ",
          "[\n    {\"name\": \"get_current_weather\", \"arguments\": {\"city\": "
          "\"Dallas\", \"state\": \"TX\", \"unit\": \"fahrenheit\"}},\n    "
          "{\"name\": \"get_current_weather\", \"arguments\": {\"city\": "
          "\"Orlando\", \"state\": \"FL\", \"unit\": \"fahrenheit\"}}\n]"));
  CHECK(r.content == " ");
  REQUIRE(r.names.size() == 2);
  CHECK(r.names[0] == "get_current_weather");
  CHECK(r.names[1] == "get_current_weather");
  REQUIRE(r.args_by_index.size() == 2);
  CHECK(ArgsEqual(r.args_by_index[0],
                  R"({"city":"Dallas","state":"TX","unit":"fahrenheit"})"));
  CHECK(ArgsEqual(r.args_by_index[1],
                  R"({"city":"Orlando","state":"FL","unit":"fahrenheit"})"));
  // Even for parallel calls, at most one tool-call diff is emitted per message.
  CHECK_FALSE(r.saw_multi_in_one_message);
}

// ═══ Factory ═════════════════════════════════════════════════════════════════

TEST_CASE("Factory: get_tool_parser(\"jamba\") works") {
  auto parser = get_tool_parser("jamba");
  REQUIRE(parser != nullptr);
  auto req = empty_request();
  auto info = parser->extract_tool_calls(
      "<tool_calls>[{\"name\": \"f\", \"arguments\": {\"a\": 1}}]</tool_calls>",
      req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "f");
}
