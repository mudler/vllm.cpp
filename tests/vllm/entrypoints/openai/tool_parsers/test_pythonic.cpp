// Ported from: vllm/tests/tool_parsers/test_pythonic_tool_parser.py @ e24d1b24
//
// Ports every upstream case for the `pythonic` parser: the 7 distinct call
// outputs x {non-streaming, streaming}, plus no_tool_call, the large-steps
// streaming cadence test, and the regex-timeout graceful-fallback test.
//
// Streaming: upstream splits the model output into per-token deltas via the
// model tokenizer (unavailable here). We drive the stateful parser char by char
// (the finest granularity - a strict superset stress of any token split) and
// reconstruct the tool calls exactly as upstream's StreamingToolReconstructor
// does. Argument strings are compared in nlohmann's COMPACT form (no space after
// ':'/','), the documented whitespace deviation (see pythonic.h); values and
// key order are otherwise identical to upstream.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/pythonic.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

// Mirror of upstream StreamingToolReconstructor (one-tool-per-delta relaxed -
// see file header on char-by-char driving).
struct Reconstructor {
  std::vector<ToolCall> tool_calls;
  std::string other_content;

  void Append(const DeltaMessage& d) {
    if (d.content.has_value()) other_content += *d.content;
    if (!d.tool_calls.has_value()) return;
    for (const DeltaToolCall& tc : *d.tool_calls) {
      const std::size_t idx = static_cast<std::size_t>(tc.index);
      if (idx < tool_calls.size()) {
        CHECK_FALSE(tc.id.has_value());               // id only once
        CHECK_FALSE(tc.function.name.has_value());    // name only once
        if (tc.function.arguments.has_value()) {
          tool_calls[idx].function.arguments += *tc.function.arguments;
        }
      } else {
        REQUIRE(idx == tool_calls.size());            // strictly increasing
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

std::vector<std::string> CharDeltas(const std::string& full) {
  std::vector<std::string> out;
  for (char c : full) out.emplace_back(1, c);
  return out;
}

// Non-streaming: one function call, no content.
void CheckSingle(const std::string& output, const std::string& name,
                 const std::string& args) {
  PythonicToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(output, req);
  CHECK(info.tools_called == true);
  CHECK_FALSE(info.content.has_value());
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == name);
  CHECK(info.tool_calls[0].function.arguments == args);
}

// Streaming char-by-char: reconstruct one function call.
void CheckSingleStreaming(const std::string& output, const std::string& name,
                          const std::string& args) {
  PythonicToolParser parser;
  Reconstructor r = DriveDeltas(parser, CharDeltas(output));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == name);
  CHECK(r.tool_calls[0].function.arguments == args);
}

const char* kSimpleOut = "[get_weather(city='San Francisco', metric='celsius')]";
const char* kSimpleArgs = R"({"city":"San Francisco","metric":"celsius"})";

const char* kMoreTypesOut =
    "[register_user(name='John Doe', age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, role=None, "
    "passed_test=True, aliases=['John', 'Johnny'])]";
const char* kMoreTypesArgs =
    R"({"name":"John Doe","age":37,"address":{"city":"San Francisco",)"
    R"("state":"CA"},"role":null,"passed_test":true,"aliases":["John","Johnny"]})";

const char* kParamlessOut = "[get_weather()]";
const char* kParamlessArgs = "{}";

const char* kEmptyDictOut = "[do_something_cool(additional_data={})]";
const char* kEmptyDictArgs = R"({"additional_data":{}})";

const char* kEmptyListOut = "[do_something_cool(steps=[])]";
const char* kEmptyListArgs = R"({"steps":[]})";

// r"[get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')]"
const char* kEscapedOut =
    "[get_weather(city='Martha\\'s Vineyard', metric='\\\"cool units\\\"')]";
const char* kEscapedArgs = R"({"city":"Martha's Vineyard","metric":"\"cool units\""})";

}  // namespace

TEST_CASE("pythonic: no tool call is plain content (non-streaming)") {
  PythonicToolParser parser;
  auto req = empty_request();
  const std::string out = "How can I help you today?";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("pythonic: no tool call is plain content (streaming)") {
  PythonicToolParser parser;
  const std::string out = "How can I help you today?";
  Reconstructor r = DriveDeltas(parser, CharDeltas(out));
  CHECK(r.other_content == out);
  CHECK(r.tool_calls.empty());
}

TEST_CASE("pythonic: simple") {
  CheckSingle(kSimpleOut, "get_weather", kSimpleArgs);
}
TEST_CASE("pythonic: simple streaming") {
  CheckSingleStreaming(kSimpleOut, "get_weather", kSimpleArgs);
}

TEST_CASE("pythonic: more types") {
  CheckSingle(kMoreTypesOut, "register_user", kMoreTypesArgs);
}
TEST_CASE("pythonic: more types streaming") {
  CheckSingleStreaming(kMoreTypesOut, "register_user", kMoreTypesArgs);
}

TEST_CASE("pythonic: parameterless") {
  CheckSingle(kParamlessOut, "get_weather", kParamlessArgs);
}
TEST_CASE("pythonic: parameterless streaming") {
  CheckSingleStreaming(kParamlessOut, "get_weather", kParamlessArgs);
}

TEST_CASE("pythonic: empty dict") {
  CheckSingle(kEmptyDictOut, "do_something_cool", kEmptyDictArgs);
}
TEST_CASE("pythonic: empty dict streaming") {
  CheckSingleStreaming(kEmptyDictOut, "do_something_cool", kEmptyDictArgs);
}

TEST_CASE("pythonic: empty list") {
  CheckSingle(kEmptyListOut, "do_something_cool", kEmptyListArgs);
}
TEST_CASE("pythonic: empty list streaming") {
  CheckSingleStreaming(kEmptyListOut, "do_something_cool", kEmptyListArgs);
}

TEST_CASE("pythonic: escaped string") {
  CheckSingle(kEscapedOut, "get_weather", kEscapedArgs);
}
TEST_CASE("pythonic: escaped string streaming") {
  CheckSingleStreaming(kEscapedOut, "get_weather", kEscapedArgs);
}

TEST_CASE("pythonic: parallel calls") {
  PythonicToolParser parser;
  auto req = empty_request();
  // Build "[SIMPLE, MORE_TYPES]".
  const std::string joined =
      std::string(kSimpleOut).substr(0, std::string(kSimpleOut).size() - 1) +
      ", " + std::string(kMoreTypesOut).substr(1);
  auto info = parser.extract_tool_calls(joined, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].function.arguments == kSimpleArgs);
  CHECK(info.tool_calls[1].function.name == "register_user");
  CHECK(info.tool_calls[1].function.arguments == kMoreTypesArgs);
}

TEST_CASE("pythonic: parallel calls streaming") {
  PythonicToolParser parser;
  const std::string joined =
      std::string(kSimpleOut).substr(0, std::string(kSimpleOut).size() - 1) +
      ", " + std::string(kMoreTypesOut).substr(1);
  Reconstructor r = DriveDeltas(parser, CharDeltas(joined));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(r.tool_calls[0].function.arguments == kSimpleArgs);
  CHECK(r.tool_calls[1].function.name == "register_user");
  CHECK(r.tool_calls[1].function.arguments == kMoreTypesArgs);
}

// Upstream test_streaming_tool_call_with_large_steps - explicit multi-char
// deltas straddling call boundaries (assert_one_tool_per_delta=False).
TEST_CASE("pythonic: streaming with large steps") {
  PythonicToolParser parser;
  const std::vector<std::string> deltas = {
      "[get_weather(city='San",
      " Francisco', metric='celsius'), get_weather(), "
      "do_something_cool(steps=[])]"};
  Reconstructor r = DriveDeltas(parser, deltas);
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 3);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(r.tool_calls[0].function.arguments == kSimpleArgs);
  CHECK(r.tool_calls[1].function.name == "get_weather");
  CHECK(r.tool_calls[1].function.arguments == kParamlessArgs);
  CHECK(r.tool_calls[2].function.name == "do_something_cool");
  CHECK(r.tool_calls[2].function.arguments == kEmptyListArgs);
}

// Upstream test_regex_timeout_handling - the Python `regex` timeout has no C++
// analogue; we port the behavioral assertion (malformed pythonic-ish input is
// treated as plain content, not a tool call).
TEST_CASE("pythonic: malformed input falls back to content") {
  PythonicToolParser parser;
  auto req = empty_request();
  const std::string bad = "hello world[A(A=\t)A(A=,\t)A(A=,\t";
  auto info = parser.extract_tool_calls(bad, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == bad);
}
