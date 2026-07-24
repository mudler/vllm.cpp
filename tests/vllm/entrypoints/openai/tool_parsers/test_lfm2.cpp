// Ported from: vllm/tests/tool_parsers/test_lfm2_tool_parser.py @ e24d1b24
//
// Ports every upstream case for the `lfm2` parser: no_tool_call, the parametrized
// TEST_CASES battery (simple / more_types / parameterless / empty_dict /
// empty_list / escaped_string / parallel / content_after / dotted_name) over
// {non-streaming, streaming}, plus the streaming-specific cases
// (large_steps, full_block+trailing, leading_content+block, leading+block+
// trailing, echoed-body suppression, char-by-char multi-dict list, dotted-name).
//
// NOT ported: test_adjust_request_disables_skip_special_tokens - adjust_request
// only forces skip_special_tokens=False, a serving/tokenizer concern the
// text-only abstract.h seam does not model (see lfm2.h DEVIATION note).
// test_regex_timeout_handling is ported as its BEHAVIOUR (malformed input falls
// back to plain content); the Python `regex` timeout has no C++ analogue.
//
// Streaming: upstream splits the output into per-TOKEN deltas via the tokenizer
// (the sentinels are single special tokens). We feed the whole block as one delta
// for the parametrized cases (equivalent to the sentinel+body arriving together)
// and drive the sentinel-atomic / char-by-char cadences exactly where upstream
// does. Argument strings are compared as PARSED JSON (compact-dump deviation
// inherited from the pythonic core).
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/lfm2.h"

using namespace vllm::entrypoints::openai;

namespace {

const std::string kStart = "<|tool_call_start|>";
const std::string kEnd = "<|tool_call_end|>";

nlohmann::json ParsedArgs(const std::string& a) { return nlohmann::json::parse(a); }

std::string Wrap(const std::string& tool_text, const std::string& content_after = "") {
  std::string r = kStart + "[" + tool_text + "]" + kEnd;
  if (!content_after.empty()) r += "\n" + content_after;
  return r;
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

Reconstructor DriveDeltas(Lfm2ToolParser& p, const std::vector<std::string>& deltas) {
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

// Non-streaming single-call check; args compared as parsed JSON.
void CheckSingleNS(const std::string& tool_text, const std::string& name,
                   const nlohmann::json& args) {
  Lfm2ToolParser parser;
  ChatCompletionRequest req;
  auto info = parser.extract_tool_calls(Wrap(tool_text), req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == name);
  CHECK(ParsedArgs(info.tool_calls[0].function.arguments) == args);
}

// Streaming single-call check driving the whole block as one delta.
void CheckSingleStream(const std::string& tool_text, const std::string& name,
                       const nlohmann::json& args) {
  Lfm2ToolParser parser;
  Reconstructor r = DriveDeltas(parser, {Wrap(tool_text)});
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == name);
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) == args);
  CHECK(r.other_content.empty());
}

const char* kSimple = "get_candidate_status(candidate_id='12345')";
nlohmann::json SimpleArgs() { return nlohmann::json{{"candidate_id", "12345"}}; }

const char* kMoreTypes =
    "register_user(name='John Doe', age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, role=None, "
    "passed_test=True, aliases=['John', 'Johnny'])";
nlohmann::json MoreTypesArgs() {
  return nlohmann::json{{"name", "John Doe"},
                        {"age", 37},
                        {"address", {{"city", "San Francisco"}, {"state", "CA"}}},
                        {"role", nullptr},
                        {"passed_test", true},
                        {"aliases", {"John", "Johnny"}}};
}

const char* kDotted =
    "grocery.orderIngredients("
    "ingredientList=[{'name': 'Lasagna noodles', 'amount': 250, 'unit': 'g'}], "
    "deliveryAddress='845 Willow Lane, Springfield, IL 62704')";
nlohmann::json DottedArgs() {
  return nlohmann::json{
      {"ingredientList",
       {{{"name", "Lasagna noodles"}, {"amount", 250}, {"unit", "g"}}}},
      {"deliveryAddress", "845 Willow Lane, Springfield, IL 62704"}};
}

}  // namespace

TEST_CASE("lfm2: no tool call (non-streaming)") {
  Lfm2ToolParser parser;
  ChatCompletionRequest req;
  const std::string out = "How can I help you today?";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("lfm2: no tool call (streaming)") {
  Lfm2ToolParser parser;
  Reconstructor r = DriveDeltas(parser, CharDeltas("How can I help you today?"));
  CHECK(r.tool_calls.empty());
  CHECK(r.other_content == "How can I help you today?");
}

TEST_CASE("lfm2: simple") { CheckSingleNS(kSimple, "get_candidate_status", SimpleArgs()); }
TEST_CASE("lfm2: simple streaming") {
  CheckSingleStream(kSimple, "get_candidate_status", SimpleArgs());
}

TEST_CASE("lfm2: more types") { CheckSingleNS(kMoreTypes, "register_user", MoreTypesArgs()); }
TEST_CASE("lfm2: more types streaming") {
  CheckSingleStream(kMoreTypes, "register_user", MoreTypesArgs());
}

TEST_CASE("lfm2: parameterless") {
  CheckSingleNS("get_weather()", "get_weather", nlohmann::json::object());
}
TEST_CASE("lfm2: parameterless streaming") {
  CheckSingleStream("get_weather()", "get_weather", nlohmann::json::object());
}

TEST_CASE("lfm2: empty dict") {
  CheckSingleNS("do_something_cool(additional_data={})", "do_something_cool",
                nlohmann::json{{"additional_data", nlohmann::json::object()}});
}
TEST_CASE("lfm2: empty dict streaming") {
  CheckSingleStream("do_something_cool(additional_data={})", "do_something_cool",
                    nlohmann::json{{"additional_data", nlohmann::json::object()}});
}

TEST_CASE("lfm2: empty list") {
  CheckSingleNS("do_something_cool(steps=[])", "do_something_cool",
                nlohmann::json{{"steps", nlohmann::json::array()}});
}
TEST_CASE("lfm2: empty list streaming") {
  CheckSingleStream("do_something_cool(steps=[])", "do_something_cool",
                    nlohmann::json{{"steps", nlohmann::json::array()}});
}

TEST_CASE("lfm2: escaped string") {
  const std::string out =
      "get_weather(city='Martha\\'s Vineyard', metric='\\\"cool units\\\"')";
  CheckSingleNS(out, "get_weather",
                nlohmann::json{{"city", "Martha's Vineyard"},
                               {"metric", "\"cool units\""}});
}
TEST_CASE("lfm2: escaped string streaming") {
  const std::string out =
      "get_weather(city='Martha\\'s Vineyard', metric='\\\"cool units\\\"')";
  CheckSingleStream(out, "get_weather",
                    nlohmann::json{{"city", "Martha's Vineyard"},
                                   {"metric", "\"cool units\""}});
}

TEST_CASE("lfm2: parallel calls (non-streaming)") {
  Lfm2ToolParser parser;
  ChatCompletionRequest req;
  const std::string out = Wrap(std::string(kSimple) + ", " + kMoreTypes);
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_candidate_status");
  CHECK(ParsedArgs(info.tool_calls[0].function.arguments) == SimpleArgs());
  CHECK(info.tool_calls[1].function.name == "register_user");
  CHECK(ParsedArgs(info.tool_calls[1].function.arguments) == MoreTypesArgs());
}

TEST_CASE("lfm2: parallel calls (streaming)") {
  Lfm2ToolParser parser;
  const std::string out = Wrap(std::string(kSimple) + ", " + kMoreTypes);
  Reconstructor r = DriveDeltas(parser, {out});
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(r.tool_calls[0].function.name == "get_candidate_status");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) == SimpleArgs());
  CHECK(r.tool_calls[1].function.name == "register_user");
  CHECK(ParsedArgs(r.tool_calls[1].function.arguments) == MoreTypesArgs());
}

TEST_CASE("lfm2: dotted name") { CheckSingleNS(kDotted, "grocery.orderIngredients", DottedArgs()); }
TEST_CASE("lfm2: dotted name streaming (single delta)") {
  Lfm2ToolParser parser;
  Reconstructor r = DriveDeltas(parser, {Wrap(kDotted)});
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "grocery.orderIngredients");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) == DottedArgs());
}

TEST_CASE("lfm2: content after tool call (non-streaming)") {
  Lfm2ToolParser parser;
  ChatCompletionRequest req;
  const std::string out =
      Wrap(kSimple, "Checking the current status of candidate ID 12345.");
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Checking the current status of candidate ID 12345.");
}

TEST_CASE("lfm2: streaming full block + trailing in one delta") {
  Lfm2ToolParser parser;
  const std::string full = kStart + "[" + kSimple + "]" + kEnd + "\nDone.";
  Reconstructor r = DriveDeltas(parser, {full});
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_candidate_status");
  CHECK(r.other_content.find("Done.") != std::string::npos);
}

TEST_CASE("lfm2: streaming leading content + full block in one delta") {
  Lfm2ToolParser parser;
  const std::string full = "Let me check. " + kStart + "[" + kSimple + "]" + kEnd;
  Reconstructor r = DriveDeltas(parser, {full});
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_candidate_status");
  CHECK(r.other_content.find("Let me check.") != std::string::npos);
}

TEST_CASE("lfm2: streaming leading + block + trailing in one delta") {
  Lfm2ToolParser parser;
  const std::string full =
      "Let me check. " + kStart + "[" + kSimple + "]" + kEnd + "\nDone.";
  Reconstructor r = DriveDeltas(parser, {full});
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.other_content.find("Let me check.") != std::string::npos);
  CHECK(r.other_content.find("Done.") != std::string::npos);
}

TEST_CASE("lfm2: echoed body not leaked (non-streaming)") {
  Lfm2ToolParser parser;
  ChatCompletionRequest req;
  const std::string body =
      "[grocery.orderIngredients("
      "ingredientList=[{'name': 'apple', 'quantity': '2'}], "
      "deliveryAddress='123 Main St')]";
  const std::string out = kStart + body + kEnd + body + kEnd;
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "grocery.orderIngredients");
  CHECK((!info.content.has_value() || info.content->empty()));
}

TEST_CASE("lfm2: echoed body not leaked (streaming, one delta)") {
  Lfm2ToolParser parser;
  const std::string body =
      "[grocery.orderIngredients("
      "ingredientList=[{'name': 'apple', 'quantity': '2'}], "
      "deliveryAddress='123 Main St')]";
  const std::string out = kStart + body + kEnd + body + kEnd;
  Reconstructor r = DriveDeltas(parser, {out});
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "grocery.orderIngredients");
  CHECK(r.other_content.find("grocery.orderIngredients") == std::string::npos);
  CHECK(r.other_content.find(kEnd) == std::string::npos);
}

TEST_CASE("lfm2: streaming large steps (three calls, one delta)") {
  Lfm2ToolParser parser;
  const std::string full =
      kStart + "[get_candidate_status(candidate_id='12345'), get_weather(), "
               "do_something_cool(steps=[])]" +
      kEnd;
  Reconstructor r = DriveDeltas(parser, {full});
  REQUIRE(r.tool_calls.size() == 3);
  CHECK(r.tool_calls[0].function.name == "get_candidate_status");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) == SimpleArgs());
  CHECK(r.tool_calls[1].function.name == "get_weather");
  CHECK(ParsedArgs(r.tool_calls[1].function.arguments) == nlohmann::json::object());
  CHECK(r.tool_calls[2].function.name == "do_something_cool");
  CHECK(ParsedArgs(r.tool_calls[2].function.arguments) ==
        nlohmann::json{{"steps", nlohmann::json::array()}});
}

TEST_CASE("lfm2: streaming char-by-char multi-dict list") {
  Lfm2ToolParser parser;
  const std::string full =
      kStart + "[grocery.orderIngredients(ingredientList=["
               "{\"name\": \"apple\", \"quantity\": \"2\"}, "
               "{\"name\": \"bread\", \"quantity\": \"1\"}])]" +
      kEnd;
  Reconstructor r = DriveDeltas(parser, CharDeltas(full));
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "grocery.orderIngredients");
  CHECK(ParsedArgs(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"ingredientList",
                        {{{"name", "apple"}, {"quantity", "2"}},
                         {{"name", "bread"}, {"quantity", "1"}}}}});
}

TEST_CASE("lfm2: malformed input falls back to content (regex-timeout behaviour)") {
  Lfm2ToolParser parser;
  ChatCompletionRequest req;
  std::string fake = kStart + "[A(A=";
  for (int i = 0; i < 2; ++i) fake += "\t)A(A=,\t";
  fake += "]" + kEnd;
  auto info = parser.extract_tool_calls(fake, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == fake);
}
