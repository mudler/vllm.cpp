// Ported from: vllm/tests/tool_parsers/test_minicpm5xml_tool_parser.py @ e24d1b24
//
// Ports the upstream cases for the `minicpm5` parser. Non-streaming: all 25
// extract_tool_calls cases (single/surrounding/CDATA/tokenizer-markers/collapsed
// tags/aliases/wrappers/schema-validation/thinking-strip). Streaming: all
// extract_tool_calls_streaming cases.
//
// SKIPPED (justified): the two adjust_request tests (skip_special_tokens /
// tool_choice=="none") - adjust_request is a serving-layer request mutation, not
// part of the TEXT-only ToolParser seam (see minicpm5.h). test_registered_in_
// tool_parser_manager is ported against the get_tool_parser factory.
//
// Streaming: upstream splits via _random_chunks (Python `random`, not
// reproducible here) or explicit chunks. We drive char-by-char - the finest
// granularity, a strict superset stress of any chunking - and reconstruct with
// the StreamingToolReconstructor invariants (id/name once, index increment).
// Argument strings compared as PARSED JSON (compact-vs-json.dumps whitespace
// deviation; json.loads-equivalent).
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/minicpm5.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionToolsParam Tool(const std::string& name, const char* params_json) {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = name;
  t.function.parameters = nlohmann::json::parse(params_json);
  return t;
}

ChatCompletionRequest RequestWith(std::vector<ChatCompletionToolsParam> tools) {
  ChatCompletionRequest req;
  req.tools = std::move(tools);
  req.tool_choice = ToolChoice{"auto", std::nullopt};
  return req;
}

ChatCompletionRequest WeatherRequest() {
  return RequestWith({Tool("get_weather",
                           R"({"type":"object","properties":{"city":{"type":"string"},)"
                           R"("date":{"type":"string"}},"required":["city"]})")});
}
ChatCompletionRequest SumRequest() {
  return RequestWith({Tool("sum_values",
                           R"({"type":"object","properties":{"nums":{"type":"array"},)"
                           R"("exact":{"type":"boolean"}},"required":["nums"]})")});
}
ChatCompletionRequest NoRequiredRequest() {
  return RequestWith({Tool("noop",
                           R"({"type":"object","properties":{"note":{"type":"string"}},)"
                           R"("required":[]})")});
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
        CHECK_FALSE(tc.id.has_value());
        CHECK_FALSE(tc.function.name.has_value());
        if (tc.function.arguments.has_value())
          tool_calls[idx].function.arguments += *tc.function.arguments;
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

Reconstructor DriveChars(const std::string& full, const ChatCompletionRequest& req) {
  MiniCPM5ToolParser parser;
  Reconstructor r;
  std::string prev;
  for (char ch : full) {
    const std::string delta(1, ch);
    const std::string cur = prev + delta;
    auto dm = parser.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (dm.has_value()) r.Append(*dm);
  }
  return r;
}

nlohmann::json Args(const ToolCall& tc) {
  return nlohmann::json::parse(tc.function.arguments);
}

}  // namespace

TEST_CASE("minicpm5: registered in tool parser manager") {
  CHECK(get_tool_parser("minicpm5") != nullptr);
}

TEST_CASE("minicpm5: no tool call") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  auto out = parser.extract_tool_calls("How can I help you?", req);
  CHECK_FALSE(out.tools_called);
  CHECK(out.tool_calls.empty());
  REQUIRE(out.content.has_value());
  CHECK(*out.content == "How can I help you?");
}

TEST_CASE("minicpm5: single call with surrounding text") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "Intro before.\n<function name=\"get_weather\">"
      "<param name=\"city\">\xE4\xB8\x8A\xE6\xB5\xB7</param>"
      "<param name=\"date\">2024-06-27</param></function>\nOutro after.\n";
  auto out = parser.extract_tool_calls(text, req);
  CHECK(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "get_weather");
  CHECK(Args(out.tool_calls[0]) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
  CHECK_FALSE(out.content.has_value());
}

TEST_CASE("minicpm5: cdata multiline") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\">"
      "<param name=\"city\"><![CDATA[\xE5\x8C\x97\n\xE4\xBA\xAC]]></param>"
      "<param name=\"date\">2024-06-27</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  auto args = Args(out.tool_calls[0]);
  CHECK(args["city"] == "\xE5\x8C\x97\n\xE4\xBA\xAC");
  CHECK(args["date"] == "2024-06-27");
}

TEST_CASE("minicpm5: tokenizer space marker") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function\xC4\xA0name=\"get_weather\">"
      "<param\xC4\xA0name=\"city\">\xE4\xB8\x8A\xE6\xB5\xB7</param>"
      "<param\xC4\xA0name=\"date\">2024-06-27</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  CHECK(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(Args(out.tool_calls[0]) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
}

TEST_CASE("minicpm5: collapsed function and param tags") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<functionname=\"get_weather\">"
      "<paramname=\"city\">\xE4\xB8\x8A\xE6\xB5\xB7</param>"
      "<paramname=\"date\">2024-06-27</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  CHECK(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(Args(out.tool_calls[0]) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
}

TEST_CASE("minicpm5: collapsed param tags with tokenizer space function") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function\xC4\xA0name=\"get_weather\">"
      "<paramname=\"city\">\xE4\xB8\x8A\xE6\xB5\xB7</param>"
      "<paramname=\"date\">2024-06-27</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  CHECK(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(Args(out.tool_calls[0]) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
}

TEST_CASE("minicpm5: collapsed tags current weather") {
  MiniCPM5ToolParser parser;
  auto req = RequestWith({Tool(
      "get_current_weather",
      R"({"type":"object","properties":{"city":{"type":"string"},)"
      R"("state":{"type":"string"},"unit":{"type":"string"}},)"
      R"("required":["city","state","unit"]})")});
  const std::string text =
      "<functionname=\"get_current_weather\">"
      "<paramname=\"city\">Dallas</param><paramname=\"state\">TX</param>"
      "<paramname=\"unit\">fahrenheit</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  CHECK(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(Args(out.tool_calls[0]) ==
        nlohmann::json{{"city", "Dallas"}, {"state", "TX"}, {"unit", "fahrenheit"}});
}

TEST_CASE("minicpm5: unknown tool block preserved") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"unknown\"><param name=\"x\">1</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  CHECK_FALSE(out.tools_called);
  REQUIRE(out.content.has_value());
  CHECK(out.content->find("unknown") != std::string::npos);
}

TEST_CASE("minicpm5: non string types") {
  MiniCPM5ToolParser parser;
  auto req = SumRequest();
  const std::string text =
      "<function name=\"sum_values\"><param name=\"nums\">[1, 2, 3]</param>"
      "<param name=\"exact\">true</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  auto args = Args(out.tool_calls[0]);
  CHECK(args["nums"] == nlohmann::json({1, 2, 3}));
  CHECK(args["exact"] == true);
}

TEST_CASE("minicpm5: multiple calls interleaved text") {
  MiniCPM5ToolParser parser;
  std::vector<ChatCompletionToolsParam> tools;
  auto w = WeatherRequest();
  auto s = SumRequest();
  tools.push_back((*w.tools)[0]);
  tools.push_back((*s.tools)[0]);
  auto req = RequestWith(tools);
  const std::string text =
      "Head\n<function name=\"get_weather\"><param name=\"city\">\xE5\x8C\x97\xE4\xBA\xAC</param></function>\n"
      "TXT\n<function name=\"sum_values\"><param name=\"nums\">[7,8,9]</param>"
      "<param name=\"exact\">false</param></function>\nTail\n";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 2);
  CHECK(Args(out.tool_calls[0])["city"] == "\xE5\x8C\x97\xE4\xBA\xAC");
  CHECK(Args(out.tool_calls[1])["nums"] == nlohmann::json({7, 8, 9}));
  CHECK(Args(out.tool_calls[1])["exact"] == false);
  CHECK_FALSE(out.content.has_value());
}

TEST_CASE("minicpm5: incomplete missing function end") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\"><param name=\"city\">\xE5\x8C\x97\xE4\xBA\xAC</param>";
  auto out = parser.extract_tool_calls(text, req);
  CHECK_FALSE(out.tools_called);
  REQUIRE(out.content.has_value());
  CHECK(out.content->find("get_weather") != std::string::npos);
}

TEST_CASE("minicpm5: param missing name invalid") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\"><param>\xE5\x8C\x97\xE4\xBA\xAC</param>"
      "<param name=\"date\">2024-06-27</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  CHECK_FALSE(out.tools_called);
  REQUIRE(out.content.has_value());
  CHECK(out.content->find("<param>\xE5\x8C\x97\xE4\xBA\xAC</param>") != std::string::npos);
}

TEST_CASE("minicpm5: duplicate param names invalid") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\"><param name=\"city\">\xE5\x8C\x97\xE4\xBA\xAC</param>"
      "<param name=\"city\">\xE4\xB8\x8A\xE6\xB5\xB7</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  CHECK_FALSE(out.tools_called);
}

TEST_CASE("minicpm5: case sensitive param name invalid") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\"><param name=\"City\">\xE5\x8C\x97\xE4\xBA\xAC</param></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  CHECK_FALSE(out.tools_called);
}

TEST_CASE("minicpm5: no required and zero param valid") {
  MiniCPM5ToolParser parser;
  auto req = NoRequiredRequest();
  const std::string text = "<function name=\"noop\"></function>\n";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(Args(out.tool_calls[0]) == nlohmann::json::object());
}

TEST_CASE("minicpm5: thinking only sentencepiece normalized in content") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "\xC4\x8A" "First," "\xC4\xA0" "I" "\xC4\xA0" "need" "\xC4\xA0" "to" "\xC4\xA0"
      "check" "\xC4\xA0" "the" "\xC4\xA0" "weather.";
  auto out = parser.extract_tool_calls(text, req);
  CHECK_FALSE(out.tools_called);
  REQUIRE(out.content.has_value());
  CHECK(out.content->find("\xC4\xA0") == std::string::npos);
  CHECK(out.content->find("\xC4\x8A") == std::string::npos);
  CHECK(out.content->find("First, I need to check the weather.") != std::string::npos);
}

TEST_CASE("minicpm5: properties wrapped arguments") {
  MiniCPM5ToolParser parser;
  auto req = RequestWith({Tool(
      "get_customer_by_phone",
      R"({"type":"object","properties":{"phone_number":{"type":"string"}},)"
      R"("required":["phone_number"]})")});
  const std::string text =
      "<function name=\"get_customer_by_phone\">"
      "<param name=\"properties\">{'phone_number': '555-123-2002'}</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "get_customer_by_phone");
  CHECK(Args(out.tool_calls[0]) == nlohmann::json{{"phone_number", "555-123-2002"}});
}

TEST_CASE("minicpm5: arguments wrapped arguments") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\">"
      "<param name=\"arguments\">{\"city\": \"\xE4\xB8\x8A\xE6\xB5\xB7\", \"date\": \"2024-06-27\"}</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(Args(out.tool_calls[0]) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
}

TEST_CASE("minicpm5: wrapped arguments still validate schema") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\">"
      "<param name=\"properties\">{\"unknown\": \"x\"}</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  CHECK_FALSE(out.tools_called);
}

TEST_CASE("minicpm5: extra arguments ignored when required present") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\"><param name=\"city\">\xE4\xB8\x8A\xE6\xB5\xB7</param>"
      "<param name=\"unknown\">ignored</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(Args(out.tool_calls[0]) == nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}});
}

TEST_CASE("minicpm5: extra arguments do not satisfy required") {
  MiniCPM5ToolParser parser;
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\"><param name=\"unknown\">ignored</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  CHECK_FALSE(out.tools_called);
}

TEST_CASE("minicpm5: zero arg tool ignores extra arguments") {
  MiniCPM5ToolParser parser;
  auto req = NoRequiredRequest();
  const std::string text =
      "<function name=\"noop\"><param name=\"note\">ignored</param>"
      "<param name=\"extra\">ignored</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(Args(out.tool_calls[0]) == nlohmann::json{{"note", "ignored"}});
}

TEST_CASE("minicpm5: alias get_details_by_phone") {
  MiniCPM5ToolParser parser;
  auto req = RequestWith({Tool(
      "get_customer_by_phone",
      R"({"type":"object","properties":{"phone_number":{"type":"string"}},)"
      R"("required":["phone_number"]})")});
  const std::string text =
      "<function name=\"get_details_by_phone\">"
      "<param name=\"phone_number\">555-123-2002</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "get_customer_by_phone");
  CHECK(Args(out.tool_calls[0]) == nlohmann::json{{"phone_number", "555-123-2002"}});
}

TEST_CASE("minicpm5: alias get_line_details") {
  MiniCPM5ToolParser parser;
  auto req = RequestWith({Tool(
      "get_details_by_id",
      R"({"type":"object","properties":{"id":{"type":"string"}},"required":["id"]})")});
  const std::string text =
      "<function name=\"get_line_details\"><param name=\"customer_id\">C1001</param>"
      "<param name=\"line_id\">L1001</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "get_details_by_id");
  CHECK(Args(out.tool_calls[0]) == nlohmann::json{{"id", "L1001"}});
}

TEST_CASE("minicpm5: alias enable_roaming") {
  MiniCPM5ToolParser parser;
  auto req = RequestWith({Tool(
      "toggle_roaming",
      R"({"type":"object","properties":{"line_id":{"type":"string"},)"
      R"("enabled":{"type":"boolean"}},"required":["line_id","enabled"]})")});
  const std::string text =
      "<function name=\"enable_roaming\"><param name=\"line_id\">L1001</param></function>";
  auto out = parser.extract_tool_calls(text, req);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "toggle_roaming");
  CHECK(Args(out.tool_calls[0]) == nlohmann::json{{"line_id", "L1001"}, {"enabled", true}});
}

// ─── Streaming ──────────────────────────────────────────────────────────────

TEST_CASE("minicpm5: streaming partial chunks") {
  auto req = WeatherRequest();
  const std::string text =
      "<function name=\"get_weather\"><param name=\"city\">"
      "\xE4\xB8\x8A\xE6\xB5\xB7</param><param name=\"date\">2024-06-27</param></function>\n";
  Reconstructor r = DriveChars(text, req);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
}

TEST_CASE("minicpm5: streaming tokenizer space marker") {
  auto req = WeatherRequest();
  const std::string text =
      "<function\xC4\xA0name=\"get_weather\"><param\xC4\xA0name=\"city\">"
      "\xE4\xB8\x8A\xE6\xB5\xB7</param><param\xC4\xA0name=\"date\">2024-06-27</param></function>\n";
  Reconstructor r = DriveChars(text, req);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
}

TEST_CASE("minicpm5: streaming collapsed tags weather") {
  auto req = WeatherRequest();
  const std::string text =
      "<functionname=\"get_weather\"><paramname=\"city\">\xE4\xB8\x8A\xE6\xB5\xB7</param>"
      "<paramname=\"date\">2024-06-27</param></function>\n";
  Reconstructor r = DriveChars(text, req);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
}

TEST_CASE("minicpm5: streaming incremental arguments") {
  auto req = RequestWith({Tool(
      "get_current_weather",
      R"({"type":"object","properties":{"city":{"type":"string"},)"
      R"("state":{"type":"string"},"unit":{"type":"string"}},)"
      R"("required":["city","state","unit"]})")});
  const std::string text =
      "<function name=\"get_current_weather\"><param name=\"city\">Dallas</param>"
      "<param name=\"state\">TX</param><param name=\"unit\">fahrenheit</param></function>";
  Reconstructor r = DriveChars(text, req);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "Dallas"}, {"state", "TX"}, {"unit", "fahrenheit"}});
}

TEST_CASE("minicpm5: streaming single with surrounding content") {
  auto req = WeatherRequest();
  const std::string text =
      "Intro before.\n<function name=\"get_weather\"><param name=\"city\">"
      "\xE4\xB8\x8A\xE6\xB5\xB7</param><param name=\"date\">2024-06-27</param></function>\n"
      "Outro after.\n";
  Reconstructor r = DriveChars(text, req);
  CHECK(r.other_content.find("Intro before.") != std::string::npos);
  CHECK(r.other_content.find("Outro after.") != std::string::npos);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json{{"city", "\xE4\xB8\x8A\xE6\xB5\xB7"}, {"date", "2024-06-27"}});
}

TEST_CASE("minicpm5: streaming multiple") {
  std::vector<ChatCompletionToolsParam> tools;
  auto w = WeatherRequest();
  auto s = SumRequest();
  tools.push_back((*w.tools)[0]);
  tools.push_back((*s.tools)[0]);
  auto req = RequestWith(tools);
  const std::string text =
      "Head\n<function name=\"get_weather\"><param name=\"city\">\xE5\x8C\x97\xE4\xBA\xAC</param></function>\n"
      "TXT\n<function name=\"sum_values\"><param name=\"nums\">[7,8,9]</param>"
      "<param name=\"exact\">false</param></function>\nTail\n";
  Reconstructor r = DriveChars(text, req);
  CHECK(r.other_content.find("Head") != std::string::npos);
  CHECK(r.other_content.find("TXT") != std::string::npos);
  CHECK(r.other_content.find("Tail") != std::string::npos);
  REQUIRE(r.tool_calls.size() == 2);
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments)["city"] == "\xE5\x8C\x97\xE4\xBA\xAC");
  CHECK(nlohmann::json::parse(r.tool_calls[1].function.arguments) ==
        nlohmann::json{{"nums", {7, 8, 9}}, {"exact", false}});
}
