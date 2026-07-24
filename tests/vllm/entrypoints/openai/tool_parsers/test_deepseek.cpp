// Tests for the DeepSeek-V3 and DeepSeek-V3.1 tool-call parsers.
// (vllm/tool_parsers/deepseekv3_tool_parser.py + deepseekv31_tool_parser.py
//  @ e24d1b24).
//
// Ports ALL upstream cases (deepseekv3: the common-suite config in
// test_deepseekv3_tool_parser.py; deepseekv31: the two cases in
// test_deepseekv31_tool_parser.py) and adds the extra coverage requested for
// the C++ port: single/multiple/plain-text/malformed/content-before, plus a
// streaming test that splits every multi-byte marker across delta boundaries
// (the UTF-8-safety case the token-id -> text rework must handle).
//
// The DeepSeek markers use the FULLWIDTH vertical bar (U+FF5C) and U+2581, NOT
// ASCII '|'/'_'; the literal bytes are pulled from the parser's own constants so
// the test data is byte-identical to what the parser matches.
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v3.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v31.h"

using namespace vllm::entrypoints::openai;

namespace {

// Marker byte strings (copied from the parser constants — byte-exact).
const std::string BEGIN = DeepSeekV3ToolParser::kToolCallsBeginToken;
const std::string END = DeepSeekV3ToolParser::kToolCallsEndToken;
const std::string CBEGIN = DeepSeekV3ToolParser::kToolCallBeginToken;
const std::string CEND = DeepSeekV3ToolParser::kToolCallEndToken;
const std::string SEP = DeepSeekV3ToolParser::kToolSepToken;

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

// One V3 call: <call_begin>function<sep>NAME\n```json\nARGS\n```<call_end>
std::string V3Call(const std::string& name, const std::string& args) {
  return CBEGIN + "function" + SEP + name + "\n```json\n" + args + "\n```" +
         CEND;
}
// One V3.1 call: <call_begin>NAME<sep>ARGS<call_end>
std::string V31Call(const std::string& name, const std::string& args) {
  return CBEGIN + name + SEP + args + CEND;
}

// Drive the stateful streaming parser over a sequence of delta fragments,
// accumulating previous/current text exactly as serving_chat does.
std::vector<DeltaMessage> DriveStream(ToolParser& parser,
                                      const std::vector<std::string>& deltas) {
  std::vector<DeltaMessage> out;
  std::string previous;
  ChatCompletionRequest req;
  for (const std::string& delta : deltas) {
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (dm.has_value()) out.push_back(std::move(*dm));
  }
  return out;
}

// Split `s` into fixed-size BYTE chunks (guaranteed to slice multi-byte markers
// mid-character when chunk < marker byte length).
std::vector<std::string> ByteChunks(const std::string& s, std::size_t n) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < s.size(); i += n) out.push_back(s.substr(i, n));
  return out;
}

// Reconstruct (name, concatenated-args, saw-content) from a stream of deltas.
struct Reconstructed {
  std::vector<std::string> names;
  std::string args;  // concatenation of all argument diffs (index 0)
  std::string content;
};
Reconstructed Reconstruct(const std::vector<DeltaMessage>& msgs) {
  Reconstructed r;
  for (const DeltaMessage& m : msgs) {
    if (m.content.has_value()) r.content += *m.content;
    if (!m.tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *m.tool_calls) {
      if (tc.function.name.has_value()) r.names.push_back(*tc.function.name);
      if (tc.index == 0 && tc.function.arguments.has_value())
        r.args += *tc.function.arguments;
    }
  }
  return r;
}

}  // namespace

// ─── DeepSeek V3 — non-streaming ─────────────────────────────────────────────

TEST_CASE("DeepSeekV3: single tool call, no leading content") {
  DeepSeekV3ToolParser parser;
  const std::string out =
      BEGIN + V3Call("get_weather", R"({"city": "Tokyo", "unit": "celsius"})") +
      END;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].function.arguments ==
        R"({"city": "Tokyo", "unit": "celsius"})");
  CHECK(info.tool_calls[0].id.rfind("chatcmpl-tool-", 0) == 0);
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("DeepSeekV3: parallel tool calls") {
  DeepSeekV3ToolParser parser;
  const std::string out =
      BEGIN + V3Call("get_weather", R"({"city": "Tokyo", "unit": "celsius"})") +
      V3Call("search_hotels", R"({"location": "Tokyo", "check_in": "2025-01-15"})") +
      END;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[1].function.name == "search_hotels");
  // Unique ids.
  CHECK(info.tool_calls[0].id != info.tool_calls[1].id);
}

TEST_CASE("DeepSeekV3: various data types round-trip as valid JSON args") {
  DeepSeekV3ToolParser parser;
  const std::string args =
      R"({"string_field": "hello", "int_field": 42, "float_field": 3.14, )"
      R"("bool_field": true, "null_field": null, "array_field": ["a", "b", "c"], )"
      R"("object_field": {"nested": "value"}, "empty_array": [], "empty_object": {}})";
  const std::string out = BEGIN + V3Call("test_function", args) + END;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "test_function");
  const auto parsed = nlohmann::json::parse(info.tool_calls[0].function.arguments);
  CHECK(parsed["int_field"] == 42);
  CHECK(parsed["bool_field"] == true);
  CHECK(parsed["null_field"].is_null());
  CHECK(parsed["array_field"].size() == 3);
}

TEST_CASE("DeepSeekV3: empty arguments") {
  DeepSeekV3ToolParser parser;
  const std::string out = BEGIN + V3Call("get_current_time", "{}") + END;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.arguments == "{}");
}

TEST_CASE("DeepSeekV3: surrounding (leading) text becomes content") {
  DeepSeekV3ToolParser parser;
  const std::string out = "Let me check the weather for you." + BEGIN +
                          V3Call("get_weather", R"({"city": "Paris"})") + END;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Let me check the weather for you.");
}

TEST_CASE("DeepSeekV3: escaped strings in arguments parse") {
  DeepSeekV3ToolParser parser;
  const std::string args =
      R"({"text": "He said \"hello\"", "path": "C:\\Users\\file", )"
      R"("newline": "line1\nline2"})";
  const std::string out = BEGIN + V3Call("send_message", args) + END;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  REQUIRE(info.tool_calls.size() == 1);
  const auto parsed = nlohmann::json::parse(info.tool_calls[0].function.arguments);
  CHECK(parsed["text"] == "He said \"hello\"");
  CHECK(parsed["path"] == "C:\\Users\\file");
}

TEST_CASE("DeepSeekV3: no tool call is plain content") {
  DeepSeekV3ToolParser parser;
  const std::string out = "How can I help you today? I can check weather.";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("DeepSeekV3: malformed payload -> begin marker seen, no calls") {
  DeepSeekV3ToolParser parser;
  auto req = empty_request();
  // (a) unterminated JSON with a missing closing brace + no closing fence match.
  const std::string bad1 = BEGIN + CBEGIN + "function" + SEP +
                           "get_weather\n```json\n{\"city\": \"Tokyo\"\n```" +
                           CEND + END;
  auto i1 = parser.extract_tool_calls(bad1, req);
  // Upstream xfail: start token present => tools_called stays true even though
  // the regex still matched here (arguments = the unterminated JSON string).
  CHECK(i1.tools_called == true);

  // (b) missing the inner <call_begin> entirely -> no regex match, empty calls.
  const std::string bad2 =
      BEGIN + "function" + SEP + "get_weather\n```json\n{\"city\": \"Tokyo\"}\n```" +
      END;
  auto i2 = parser.extract_tool_calls(bad2, req);
  CHECK(i2.tools_called == true);
  CHECK(i2.tool_calls.empty());
}

// ─── DeepSeek V3 — streaming ─────────────────────────────────────────────────

TEST_CASE("DeepSeekV3 streaming: name-first then argument deltas") {
  DeepSeekV3ToolParser parser;
  const std::string full =
      BEGIN + V3Call("get_weather", R"({"city": "Tokyo", "unit": "celsius"})") +
      END;
  // Fragment across natural-ish boundaries.
  const std::vector<std::string> deltas = {
      BEGIN, CBEGIN, "function", SEP, "get_weather\n```json\n",
      R"({"city": "Tokyo", )", R"("unit": "celsius"})", "\n```", CEND, END};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));

  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_weather");
  CHECK(r.args == R"({"city": "Tokyo", "unit": "celsius"})");
  CHECK(r.content.empty());
}

TEST_CASE("DeepSeekV3 streaming: split every marker mid-byte (UTF-8 safety)") {
  DeepSeekV3ToolParser parser;
  const std::string full =
      BEGIN + V3Call("get_weather", R"({"city": "Tokyo"})") + END;
  // 3-byte chunks slice the 3-byte ｜/▁ marker glyphs mid-character.
  const std::vector<std::string> deltas = ByteChunks(full, 3);
  const std::vector<DeltaMessage> msgs = DriveStream(parser, deltas);
  const Reconstructed r = Reconstruct(msgs);

  // Reconstruction matches the non-streaming extract.
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_weather");
  CHECK(r.args == R"({"city": "Tokyo"})");
  // No content emitted (call starts at byte 0), and crucially NO raw marker
  // bytes leaked into any content chunk.
  CHECK(r.content.empty());
  CHECK(r.content.find(BEGIN) == std::string::npos);
  CHECK(r.content.find(CBEGIN) == std::string::npos);
}

TEST_CASE("DeepSeekV3 streaming: content before tool call") {
  DeepSeekV3ToolParser parser;
  const std::string full = "Let me check." + BEGIN +
                           V3Call("f", R"({"x": 1})") + END;
  const std::vector<std::string> deltas = ByteChunks(full, 4);
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));

  CHECK(r.content == "Let me check.");
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "f");
  CHECK(r.args == R"({"x": 1})");
}

TEST_CASE("DeepSeekV3 streaming: plain content passes through") {
  DeepSeekV3ToolParser parser;
  const std::vector<std::string> deltas = {"The ", "weather ", "is sunny."};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  CHECK(r.content == "The weather is sunny.");
  CHECK(r.names.empty());
}

TEST_CASE("DeepSeekV3 streaming: reconstruction equals non-streaming") {
  const std::string full =
      BEGIN + V3Call("get_weather", R"({"city": "Tokyo", "unit": "celsius"})") +
      END;
  DeepSeekV3ToolParser ns;
  auto req = empty_request();
  auto info = ns.extract_tool_calls(full, req);

  DeepSeekV3ToolParser st;
  const Reconstructed r = Reconstruct(DriveStream(st, ByteChunks(full, 5)));
  REQUIRE(info.tool_calls.size() == 1);
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == info.tool_calls[0].function.name);
  CHECK(r.args == info.tool_calls[0].function.arguments);
}

// ─── DeepSeek V3.1 — non-streaming (ports the 2 upstream cases + extras) ──────

TEST_CASE("DeepSeekV31: single tool call with leading content") {
  DeepSeekV31ToolParser parser;
  const std::string out =
      "normal text" + BEGIN + V31Call("foo", R"({"x":1})") + END;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "foo");
  CHECK(info.tool_calls[0].function.arguments == R"({"x":1})");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "normal text");
}

TEST_CASE("DeepSeekV31: multiple tools, prefix is content, suffix ignored") {
  DeepSeekV31ToolParser parser;
  const std::string out = "some prefix text" + BEGIN +
                          V31Call("foo", R"({"x":1})") +
                          V31Call("bar", R"({"y":2})") + END +
                          " some suffix text";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "foo");
  CHECK(info.tool_calls[0].function.arguments == R"({"x":1})");
  CHECK(info.tool_calls[1].function.name == "bar");
  CHECK(info.tool_calls[1].function.arguments == R"({"y":2})");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "some prefix text");
}

TEST_CASE("DeepSeekV31: empty arguments") {
  DeepSeekV31ToolParser parser;
  const std::string out = BEGIN + V31Call("refresh", "{}") + END;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.arguments == "{}");
}

TEST_CASE("DeepSeekV31: no tool call is plain content") {
  DeepSeekV31ToolParser parser;
  const std::string out = "This is a regular response.";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("DeepSeekV31: malformed payload -> begin marker seen, no calls") {
  DeepSeekV31ToolParser parser;
  auto req = empty_request();
  // missing inner <call_begin> and <sep> -> no match, empty calls, still true.
  const std::string bad = BEGIN + "foo just text" + END;
  auto info = parser.extract_tool_calls(bad, req);
  CHECK(info.tools_called == true);
  CHECK(info.tool_calls.empty());
}

// ─── DeepSeek V3.1 — streaming ───────────────────────────────────────────────

TEST_CASE("DeepSeekV31 streaming: name-first then argument deltas") {
  DeepSeekV31ToolParser parser;
  const std::vector<std::string> deltas = {
      BEGIN, CBEGIN, "get_", "weather", SEP, R"({"city": )",
      R"("Tokyo"})", CEND, END};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "get_weather");
  CHECK(r.args == R"({"city": "Tokyo"})");
}

TEST_CASE("DeepSeekV31 streaming: split every marker mid-byte (UTF-8 safety)") {
  DeepSeekV31ToolParser parser;
  const std::string full = BEGIN + V31Call("foo", R"({"x":1})") + END;
  const std::vector<std::string> deltas = ByteChunks(full, 3);
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));

  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == "foo");
  CHECK(r.args == R"({"x":1})");
  CHECK(r.content.empty());
  CHECK(r.content.find(CEND) == std::string::npos);
}

TEST_CASE("DeepSeekV31 streaming: reconstruction equals non-streaming") {
  const std::string full =
      "hi" + BEGIN + V31Call("bar", R"({"y": [1,2,3]})") + END;
  DeepSeekV31ToolParser ns;
  auto req = empty_request();
  auto info = ns.extract_tool_calls(full, req);

  DeepSeekV31ToolParser st;
  const Reconstructed r = Reconstruct(DriveStream(st, ByteChunks(full, 2)));
  REQUIRE(info.tool_calls.size() == 1);
  REQUIRE(r.names.size() == 1);
  CHECK(r.names[0] == info.tool_calls[0].function.name);
  CHECK(r.args == info.tool_calls[0].function.arguments);
  CHECK(r.content == "hi");
}

// ─── Factory ─────────────────────────────────────────────────────────────────

TEST_CASE("Factory: deepseek parsers by name") {
  auto v3 = get_tool_parser("deepseek_v3");
  REQUIRE(v3 != nullptr);
  auto v31 = get_tool_parser("deepseek_v31");
  REQUIRE(v31 != nullptr);

  auto req = empty_request();
  const std::string v3_out =
      BEGIN + V3Call("f", "{}") + END;
  CHECK(v3->extract_tool_calls(v3_out, req).tools_called == true);
  const std::string v31_out = BEGIN + V31Call("f", "{}") + END;
  CHECK(v31->extract_tool_calls(v31_out, req).tools_called == true);
}
