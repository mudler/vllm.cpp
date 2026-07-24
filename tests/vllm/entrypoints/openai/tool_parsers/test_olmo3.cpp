// Ported from: vllm/tests/tool_parsers/test_olmo3_tool_parser.py @ e24d1b24
//
// Ports every upstream case for the `olmo3` parser: no_tool_call, the 8 distinct
// call outputs x {non-streaming, streaming}, the large-steps streaming cadence
// test, and the regex-timeout graceful-fallback test.
//
// Streaming: upstream splits the model output into per-token deltas via the
// model tokenizer (unavailable here). We drive the stateful parser char by char
// (the finest granularity - a strict superset stress of any token split) and
// reconstruct the tool calls exactly as upstream's StreamingToolReconstructor
// does. Argument strings are compared as PARSED JSON (nlohmann), the documented
// whitespace deviation inherited from the pythonic core (compact vs json.dumps
// ", "/": "); values and key order are otherwise identical to upstream.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/olmo3.h"

using namespace vllm::entrypoints::openai;

namespace {

struct Reconstructor {
  std::vector<ToolCall> tool_calls;
  std::string other_content;

  void Append(const DeltaMessage& d) {
    if (d.content.has_value()) other_content += *d.content;
    if (!d.tool_calls.has_value()) return;
    for (const DeltaToolCall& tc : *d.tool_calls) {
      const std::size_t idx = static_cast<std::size_t>(tc.index);
      if (idx < tool_calls.size()) {
        CHECK_FALSE(tc.id.has_value());             // id only once
        CHECK_FALSE(tc.function.name.has_value());  // name only once
        if (tc.function.arguments.has_value()) {
          tool_calls[idx].function.arguments += *tc.function.arguments;
        }
      } else {
        REQUIRE(idx == tool_calls.size());  // strictly increasing
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

std::string Wrap(const std::string& inner) {
  return "<function_calls>" + inner + "</function_calls>";
}

// Non-streaming: one function call, no content; args compared as parsed JSON.
void CheckSingle(const std::string& output, const std::string& name,
                 const nlohmann::json& args) {
  Olmo3ToolParser parser;
  ChatCompletionRequest req;
  auto info = parser.extract_tool_calls(output, req);
  CHECK(info.tools_called == true);
  CHECK_FALSE(info.content.has_value());
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == name);
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments) == args);
}

void CheckSingleStreaming(const std::string& output, const std::string& name,
                          const nlohmann::json& args) {
  Olmo3ToolParser parser;
  Reconstructor r = DriveDeltas(parser, CharDeltas(output));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == name);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) == args);
}

const char* kSimpleOut = "get_weather(city='San Francisco', metric='celsius')";
nlohmann::json SimpleArgs() {
  return nlohmann::json{{"city", "San Francisco"}, {"metric", "celsius"}};
}

const char* kMoreTypesOut =
    "register_user(name='John Doe', age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, role=None, "
    "passed_test=True, aliases=['John', 'Johnny'])";
const char* kMoreTypesJsonLiteralsOut =
    "register_user(name='John Doe', age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, role=null, "
    "passed_test=true, aliases=['John', 'Johnny'])";
nlohmann::json MoreTypesArgs() {
  return nlohmann::json{
      {"name", "John Doe"},
      {"age", 37},
      {"address", {{"city", "San Francisco"}, {"state", "CA"}}},
      {"role", nullptr},
      {"passed_test", true},
      {"aliases", {"John", "Johnny"}}};
}

}  // namespace

TEST_CASE("olmo3: no tool call is plain content (non-streaming)") {
  Olmo3ToolParser parser;
  ChatCompletionRequest req;
  const std::string out = "How can I help you today?";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("olmo3: no tool call is plain content (streaming)") {
  Olmo3ToolParser parser;
  const std::string out = "How can I help you today?";
  Reconstructor r = DriveDeltas(parser, CharDeltas(out));
  CHECK(r.other_content == out);
  CHECK(r.tool_calls.empty());
}

TEST_CASE("olmo3: simple") { CheckSingle(Wrap(kSimpleOut), "get_weather", SimpleArgs()); }
TEST_CASE("olmo3: simple streaming") {
  CheckSingleStreaming(Wrap(kSimpleOut), "get_weather", SimpleArgs());
}

TEST_CASE("olmo3: more types") {
  CheckSingle(Wrap(kMoreTypesOut), "register_user", MoreTypesArgs());
}
TEST_CASE("olmo3: more types streaming") {
  CheckSingleStreaming(Wrap(kMoreTypesOut), "register_user", MoreTypesArgs());
}

TEST_CASE("olmo3: more types json literals") {
  CheckSingle(Wrap(kMoreTypesJsonLiteralsOut), "register_user", MoreTypesArgs());
}
TEST_CASE("olmo3: more types json literals streaming") {
  CheckSingleStreaming(Wrap(kMoreTypesJsonLiteralsOut), "register_user", MoreTypesArgs());
}

TEST_CASE("olmo3: parameterless") {
  CheckSingle(Wrap("get_weather()"), "get_weather", nlohmann::json::object());
}
TEST_CASE("olmo3: parameterless streaming") {
  CheckSingleStreaming(Wrap("get_weather()"), "get_weather", nlohmann::json::object());
}

TEST_CASE("olmo3: empty dict") {
  CheckSingle(Wrap("do_something_cool(additional_data={})"), "do_something_cool",
              nlohmann::json{{"additional_data", nlohmann::json::object()}});
}
TEST_CASE("olmo3: empty dict streaming") {
  CheckSingleStreaming(Wrap("do_something_cool(additional_data={})"), "do_something_cool",
                       nlohmann::json{{"additional_data", nlohmann::json::object()}});
}

TEST_CASE("olmo3: empty list") {
  CheckSingle(Wrap("do_something_cool(steps=[])"), "do_something_cool",
              nlohmann::json{{"steps", nlohmann::json::array()}});
}
TEST_CASE("olmo3: empty list streaming") {
  CheckSingleStreaming(Wrap("do_something_cool(steps=[])"), "do_something_cool",
                       nlohmann::json{{"steps", nlohmann::json::array()}});
}

TEST_CASE("olmo3: escaped string") {
  // r"get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')"
  const std::string out =
      "get_weather(city='Martha\\'s Vineyard', metric='\\\"cool units\\\"')";
  CheckSingle(Wrap(out), "get_weather",
              nlohmann::json{{"city", "Martha's Vineyard"},
                             {"metric", "\"cool units\""}});
}
TEST_CASE("olmo3: escaped string streaming") {
  const std::string out =
      "get_weather(city='Martha\\'s Vineyard', metric='\\\"cool units\\\"')";
  CheckSingleStreaming(Wrap(out), "get_weather",
                       nlohmann::json{{"city", "Martha's Vineyard"},
                                      {"metric", "\"cool units\""}});
}

TEST_CASE("olmo3: parallel calls (newline separated)") {
  Olmo3ToolParser parser;
  ChatCompletionRequest req;
  const std::string out = Wrap(std::string(kSimpleOut) + "\n" + kMoreTypesOut);
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments) == SimpleArgs());
  CHECK(info.tool_calls[1].function.name == "register_user");
  CHECK(nlohmann::json::parse(info.tool_calls[1].function.arguments) == MoreTypesArgs());
}

TEST_CASE("olmo3: parallel calls streaming (newline separated)") {
  Olmo3ToolParser parser;
  const std::string out = Wrap(std::string(kSimpleOut) + "\n" + kMoreTypesOut);
  Reconstructor r = DriveDeltas(parser, CharDeltas(out));
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) == SimpleArgs());
  CHECK(r.tool_calls[1].function.name == "register_user");
  CHECK(nlohmann::json::parse(r.tool_calls[1].function.arguments) == MoreTypesArgs());
}

// Upstream test_streaming_tool_call_with_large_steps - explicit multi-char
// deltas straddling call boundaries (assert_one_tool_per_delta=False).
TEST_CASE("olmo3: streaming with large steps") {
  Olmo3ToolParser parser;
  const std::vector<std::string> deltas = {
      "<function_calls>get_weather(city='San",
      " Francisco', metric='celsius')\n"
      "get_weather()\n"
      "do_something_cool(steps=[])</function_calls>"};
  Reconstructor r = DriveDeltas(parser, deltas);
  CHECK(r.other_content.empty());
  REQUIRE(r.tool_calls.size() == 3);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) == SimpleArgs());
  CHECK(r.tool_calls[1].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[1].function.arguments) == nlohmann::json::object());
  CHECK(r.tool_calls[2].function.name == "do_something_cool");
  CHECK(nlohmann::json::parse(r.tool_calls[2].function.arguments) ==
        nlohmann::json{{"steps", nlohmann::json::array()}});
}

// Upstream test_regex_timeout_handling - the Python `regex` timeout has no C++
// analogue; we port the behavioral assertion (malformed input is treated as
// plain content, not a tool call).
TEST_CASE("olmo3: malformed input falls back to content") {
  Olmo3ToolParser parser;
  ChatCompletionRequest req;
  const std::string bad = "hello world[A(A=\t)A(A=,\t)A(A=,\t";
  auto info = parser.extract_tool_calls(bad, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == bad);
}
