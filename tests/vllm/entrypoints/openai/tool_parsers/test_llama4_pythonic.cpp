// Ported from: vllm/tests/tool_parsers/test_llama4_pythonic_tool_parser.py
// @ e24d1b24
//
// Ports every upstream case for the `llama4_pythonic` parser: the pythonic
// core outputs x {non-streaming, streaming}, the <|python_start|>/<|python_end|>
// wrapped variants, the parallel-call variants, no_tool_call, the large-steps
// streaming cadence test, and the regex-timeout graceful-fallback test.
//
// Streaming note: the <|python_start|>/<|python_end|> markers are SINGLE special
// tokens in the model vocabulary, so the tokenizer emits each as one streaming
// delta. We reproduce that (marker => one delta) and drive the enclosed call
// text char by char. Arguments compared in nlohmann COMPACT form (see
// pythonic.h whitespace deviation).
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/llama4_pythonic.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

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

// Split `full` into deltas where each <|python_start|>/<|python_end|> marker is
// one delta (single special token) and everything else is char-by-char.
std::vector<std::string> TagAwareDeltas(const std::string& full) {
  const std::string start = Llama4PythonicToolParser::kPythonStartToken;
  const std::string end = Llama4PythonicToolParser::kPythonEndToken;
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < full.size()) {
    if (full.compare(i, start.size(), start) == 0) {
      out.push_back(start);
      i += start.size();
    } else if (full.compare(i, end.size(), end) == 0) {
      out.push_back(end);
      i += end.size();
    } else {
      out.emplace_back(1, full[i]);
      ++i;
    }
  }
  return out;
}

void CheckNonStreaming(const std::string& output,
                       const std::vector<std::pair<std::string, std::string>>& expected) {
  Llama4PythonicToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(output, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(info.tool_calls[i].type == "function");
    CHECK(info.tool_calls[i].function.name == expected[i].first);
    CHECK(info.tool_calls[i].function.arguments == expected[i].second);
  }
}

void CheckStreaming(const std::string& output,
                    const std::vector<std::pair<std::string, std::string>>& expected) {
  Llama4PythonicToolParser parser;
  Reconstructor r = DriveDeltas(parser, TagAwareDeltas(output));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(r.tool_calls[i].function.name == expected[i].first);
    CHECK(r.tool_calls[i].function.arguments == expected[i].second);
  }
}

const char* kSimpleOut = "[get_weather(city='LA', metric='C')]";
const char* kSimpleArgs = R"({"city":"LA","metric":"C"})";
const char* kMoreTypesOut =
    "[register_user(name='Doe', age=9, address={'city': 'LA', 'state': 'CA'}, "
    "role=None, passed_test=True, aliases=['John', 'Johnny'])]";
const char* kMoreTypesArgs =
    R"({"name":"Doe","age":9,"address":{"city":"LA","state":"CA"},"role":null,)"
    R"("passed_test":true,"aliases":["John","Johnny"]})";
const char* kParamlessOut = "[get_weather()]";
const char* kEmptyDictOut = "[do_something_cool(additional_data={})]";
const char* kEmptyDictArgs = R"({"additional_data":{}})";
const char* kEmptyListOut = "[do_something_cool(steps=[])]";
const char* kEmptyListArgs = R"({"steps":[]})";
const char* kEscapedOut =
    "[get_weather(city='Martha\\'s Vineyard', metric='\\\"cool units\\\"')]";
const char* kEscapedArgs = R"({"city":"Martha's Vineyard","metric":"\"cool units\""})";
const char* kRegisterDoeArgs = R"({"name":"Doe","age":9})";

}  // namespace

TEST_CASE("llama4_pythonic: no tool call is content (non-streaming)") {
  Llama4PythonicToolParser parser;
  auto req = empty_request();
  const std::string out = "How can I help you today?";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("llama4_pythonic: no tool call is content (streaming)") {
  Llama4PythonicToolParser parser;
  const std::string out = "How can I help you today?";
  Reconstructor r = DriveDeltas(parser, TagAwareDeltas(out));
  CHECK(r.other_content == out);
  CHECK(r.tool_calls.empty());
}

TEST_CASE("llama4_pythonic: simple") {
  CheckNonStreaming(kSimpleOut, {{"get_weather", kSimpleArgs}});
}
TEST_CASE("llama4_pythonic: simple streaming (escaped)") {
  CheckStreaming(kEscapedOut, {{"get_weather", kEscapedArgs}});
}
TEST_CASE("llama4_pythonic: more types") {
  CheckNonStreaming(kMoreTypesOut, {{"register_user", kMoreTypesArgs}});
}
TEST_CASE("llama4_pythonic: more types streaming") {
  CheckStreaming(kMoreTypesOut, {{"register_user", kMoreTypesArgs}});
}
TEST_CASE("llama4_pythonic: parameterless") {
  CheckNonStreaming(kParamlessOut, {{"get_weather", "{}"}});
}
TEST_CASE("llama4_pythonic: parameterless streaming") {
  CheckStreaming(kParamlessOut, {{"get_weather", "{}"}});
}
TEST_CASE("llama4_pythonic: empty dict") {
  CheckNonStreaming(kEmptyDictOut, {{"do_something_cool", kEmptyDictArgs}});
}
TEST_CASE("llama4_pythonic: empty dict streaming") {
  CheckStreaming(kEmptyDictOut, {{"do_something_cool", kEmptyDictArgs}});
}
TEST_CASE("llama4_pythonic: empty list") {
  CheckNonStreaming(kEmptyListOut, {{"do_something_cool", kEmptyListArgs}});
}
TEST_CASE("llama4_pythonic: empty list streaming") {
  CheckStreaming(kEmptyListOut, {{"do_something_cool", kEmptyListArgs}});
}
TEST_CASE("llama4_pythonic: escaped string") {
  CheckNonStreaming(kEscapedOut, {{"get_weather", kEscapedArgs}});
}
TEST_CASE("llama4_pythonic: escaped string streaming") {
  CheckStreaming(kEscapedOut, {{"get_weather", kEscapedArgs}});
}

// Parallel calls, no whitespace between args (upstream parallel_calls).
TEST_CASE("llama4_pythonic: parallel calls (no space)") {
  const std::string out =
      "[get_weather(city='LA',metric='C'),register_user(name='Doe',age=9)]";
  CheckNonStreaming(out,
                    {{"get_weather", kSimpleArgs}, {"register_user", kRegisterDoeArgs}});
}
TEST_CASE("llama4_pythonic: parallel calls (no space) streaming") {
  const std::string out =
      "[get_weather(city='LA',metric='C'),register_user(name='Doe',age=9)]";
  CheckStreaming(out,
                 {{"get_weather", kSimpleArgs}, {"register_user", kRegisterDoeArgs}});
}

// <|python_start|>...<|python_end|> wrapped single call.
TEST_CASE("llama4_pythonic: python tag wrapped") {
  const std::string out =
      "<|python_start|>[get_weather(city='LA', metric='C')]<|python_end|>";
  CheckNonStreaming(out, {{"get_weather", kSimpleArgs}});
}
TEST_CASE("llama4_pythonic: python tag wrapped streaming") {
  const std::string out =
      "<|python_start|>[get_weather(city='LA', metric='C')]<|python_end|>";
  CheckStreaming(out, {{"get_weather", kSimpleArgs}});
}

// <|python_start|> prefix, parallel calls, NO end marker (upstream test_str).
TEST_CASE("llama4_pythonic: python start prefix parallel streaming") {
  const std::string out =
      "<|python_start|>[get_weather(city='LA', metric='C'),"
      "register_user(name='Doe', age=9)]";
  CheckStreaming(out,
                 {{"get_weather", kSimpleArgs}, {"register_user", kRegisterDoeArgs}});
}
TEST_CASE("llama4_pythonic: python start prefix parallel non-streaming") {
  const std::string out =
      "<|python_start|>[get_weather(city='LA', metric='C'), "
      "register_user(name='Doe', age=9)]";
  CheckNonStreaming(out,
                    {{"get_weather", kSimpleArgs}, {"register_user", kRegisterDoeArgs}});
}

// Upstream test_streaming_tool_call_with_large_steps - wrapped, three calls,
// one delta (assert_one_tool_per_delta=False).
TEST_CASE("llama4_pythonic: streaming with large steps") {
  Llama4PythonicToolParser parser;
  const std::vector<std::string> deltas = {
      "<|python_start|>[get_weather(city='LA', metric='C'), get_weather(), "
      "do_something_cool(steps=[])]<|python_end|>"};
  Reconstructor r = DriveDeltas(parser, deltas);
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 3);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(r.tool_calls[0].function.arguments == kSimpleArgs);
  CHECK(r.tool_calls[1].function.name == "get_weather");
  CHECK(r.tool_calls[1].function.arguments == "{}");
  CHECK(r.tool_calls[2].function.name == "do_something_cool");
  CHECK(r.tool_calls[2].function.arguments == kEmptyListArgs);
}

// Upstream test_regex_timeout_handling - behavioral port (malformed -> content).
TEST_CASE("llama4_pythonic: malformed input falls back to content") {
  Llama4PythonicToolParser parser;
  auto req = empty_request();
  const std::string bad = "hello world[A(A=\t)A(A=,\t)A(A=,\t";
  auto info = parser.extract_tool_calls(bad, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == bad);
}
