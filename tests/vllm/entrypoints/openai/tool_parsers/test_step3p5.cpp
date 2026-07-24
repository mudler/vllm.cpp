// Tests for the "step3p5" tool-call parser (Step3p5ToolParser).
// Ported from: tests/tool_parsers/test_step3p5_tool_parser.py @ e24d1b24
// (all 19 upstream cases: the parametrized non-streaming + streaming families,
// the type-conversion / single-quote-object / missing-closing-tag / mixed-
// content / no-content-between / MTP-chunk-boundary / MTP-variable-chunk cases),
// plus streaming-split edges (char-by-char AND chunk cadences) that exercise the
// UTF-8-safe boundary handling the C++ text-only seam adds.
//
// The Python suite drives streaming through a real tokenizer (token-by-token);
// we have no tokenizer, so we drive the strictest cadence (character-by-
// character) as the token-by-token analogue AND the explicit chunk lists the
// upstream MTP regressions use verbatim.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/step3p5.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionToolsParam MakeTool(const std::string& name,
                                  const nlohmann::json& parameters) {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = name;
  t.function.parameters = parameters;
  return t;
}

// tests/tool_parsers/test_step3p5_tool_parser.py:36-69 (sample_tools).
std::vector<ChatCompletionToolsParam> SampleTools() {
  return {
      MakeTool("get_current_weather",
               nlohmann::json::parse(R"({
                 "type": "object",
                 "properties": {
                   "city": {"type": "string"},
                   "state": {"type": "string"},
                   "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]}
                 },
                 "required": ["city", "state"]
               })")),
      MakeTool("calculate_area",
               nlohmann::json::parse(R"({
                 "type": "object",
                 "properties": {
                   "shape": {"type": "string"},
                   "dimensions": {"type": "object"},
                   "precision": {"type": "integer"}
                 }
               })")),
  };
}

ChatCompletionRequest RequestWith(std::vector<ChatCompletionToolsParam> tools) {
  ChatCompletionRequest req;
  req.tools = std::move(tools);
  return req;
}

// Reconstruct tool-call state the way the Python tests accumulate it.
struct ToolState {
  std::optional<std::string> id;
  std::optional<std::string> name;
  std::string arguments;
  std::optional<std::string> type;
};
struct Recon {
  std::map<int, ToolState> tools;
  std::string content;
};

void Apply(Recon& r, const DeltaMessage& m) {
  if (m.content.has_value()) r.content += *m.content;
  if (!m.tool_calls.has_value()) return;
  for (const DeltaToolCall& tc : *m.tool_calls) {
    ToolState& st = r.tools[tc.index];
    if (tc.id.has_value()) st.id = tc.id;
    if (tc.type.has_value()) st.type = tc.type;
    if (tc.function.name.has_value() && !tc.function.name->empty()) {
      st.name = tc.function.name;
    }
    if (tc.function.arguments.has_value()) st.arguments += *tc.function.arguments;
  }
}

// Character-by-character streaming (token-by-token analogue).
Recon RunCharByChar(ToolParser& parser, const std::string& out,
                    const ChatCompletionRequest& req) {
  Recon r;
  std::string previous;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::string delta = out.substr(i, 1);
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (dm.has_value()) Apply(r, *dm);
  }
  return r;
}

// Explicit-chunk streaming (MTP cadence).
Recon RunChunks(ToolParser& parser, const std::vector<std::string>& chunks,
                const ChatCompletionRequest& req) {
  Recon r;
  std::string previous;
  for (const std::string& delta : chunks) {
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (dm.has_value()) Apply(r, *dm);
  }
  return r;
}

nlohmann::json ArgsOf(const ToolCall& tc) {
  return nlohmann::json::parse(tc.function.arguments);
}

}  // namespace

// ─── Non-streaming ───────────────────────────────────────────────────────────

// test_extract_tool_calls_no_tools
TEST_CASE("step3p5: no tool calls is plain content") {
  Step3p5ToolParser parser;
  const std::string out = "This is a test response without any tool calls";
  ChatCompletionRequest req;
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// test_extract_tool_calls[single_tool]
TEST_CASE("step3p5: single tool call") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "</parameter>\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\n"
      "fahrenheit\n</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(info.tool_calls[0].id.rfind("chatcmpl-tool-", 0) == 0);
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["city"] == "Dallas");
  CHECK(args["state"] == "TX");
  CHECK(args["unit"] == "fahrenheit");
  CHECK_FALSE(info.content.has_value());
}

// test_extract_tool_calls[single_tool_with_content]
TEST_CASE("step3p5: single tool call with leading content") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "Sure! Let me check the weather for you.<tool_call>\n"
      "<function=get_current_weather>\n<parameter=city>\nDallas\n</parameter>\n"
      "<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n"
      "</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["city"] == "Dallas");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Sure! Let me check the weather for you.");
}

// test_extract_tool_calls[single_tool_multiline_param]
TEST_CASE("step3p5: multiline object parameter") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=calculate_area>\n<parameter=shape>\nrectangle\n"
      "</parameter>\n<parameter=dimensions>\n{\"width\": 10, \n \"height\": 20}\n"
      "</parameter>\n<parameter=precision>\n2\n</parameter>\n</function>\n"
      "</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "calculate_area");
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["shape"] == "rectangle");
  CHECK(args["dimensions"]["width"] == 10);
  CHECK(args["dimensions"]["height"] == 20);
  CHECK(args["precision"] == 2);
}

// test_extract_tool_calls[parallel_tools]
TEST_CASE("step3p5: parallel tool calls") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "</parameter>\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\n"
      "fahrenheit\n</parameter>\n</function>\n</tool_call>\n"
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nOrlando\n"
      "</parameter>\n<parameter=state>\nFL\n</parameter>\n<parameter=unit>\n"
      "fahrenheit\n</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(ArgsOf(info.tool_calls[0])["city"] == "Dallas");
  CHECK(ArgsOf(info.tool_calls[1])["city"] == "Orlando");
  CHECK(info.tool_calls[0].id != info.tool_calls[1].id);
}

// test_extract_tool_calls[tool_with_typed_params]
TEST_CASE("step3p5: typed params (float object)") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "Let me calculate that area for you.<tool_call>\n"
      "<function=calculate_area>\n<parameter=shape>\ncircle\n</parameter>\n"
      "<parameter=dimensions>\n{\"radius\": 15.5}\n</parameter>\n"
      "<parameter=precision>\n3\n</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["shape"] == "circle");
  CHECK(args["dimensions"]["radius"] == 15.5);
  CHECK(args["precision"] == 3);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Let me calculate that area for you.");
}

// test_extract_tool_calls_fallback_no_tags
TEST_CASE("step3p5: fallback parsing without <tool_call> wrapper") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<function=get_current_weather>\n<parameter=city>\nDallas\n</parameter>\n"
      "<parameter=state>\nTX\n</parameter>\n</function>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
}

// test_extract_tool_calls_type_conversion
TEST_CASE("step3p5: parameter type conversion") {
  const auto tools = std::vector<ChatCompletionToolsParam>{
      MakeTool("test_types", nlohmann::json::parse(R"({
        "type": "object",
        "properties": {
          "int_param": {"type": "integer"},
          "float_param": {"type": "float"},
          "bool_param": {"type": "boolean"},
          "str_param": {"type": "string"},
          "obj_param": {"type": "object"}
        }
      })"))};
  Step3p5ToolParser parser;
  auto req = RequestWith(tools);
  const std::string out =
      "<tool_call>\n<function=test_types>\n<parameter=int_param>\n42\n"
      "</parameter>\n<parameter=float_param>\n3.14\n</parameter>\n"
      "<parameter=bool_param>\ntrue\n</parameter>\n<parameter=str_param>\n"
      "hello world\n</parameter>\n<parameter=obj_param>\n{\"key\": \"value\"}\n"
      "</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["int_param"] == 42);
  CHECK(args["float_param"] == 3.14);
  CHECK(args["bool_param"] == true);
  CHECK(args["str_param"] == "hello world");
  CHECK(args["obj_param"] == nlohmann::json::parse(R"({"key": "value"})"));
}

// test_extract_tool_calls_complex_type_with_single_quote
TEST_CASE("step3p5: object param with python single-quote literal") {
  const auto tools = std::vector<ChatCompletionToolsParam>{
      MakeTool("test_types", nlohmann::json::parse(R"({
        "type": "object",
        "properties": {"obj_param": {"type": "object"}}
      })"))};
  Step3p5ToolParser parser;
  auto req = RequestWith(tools);
  const std::string out =
      "<tool_call>\n<function=test_types>\n<parameter=obj_param>\n"
      "{'key': 'value'}\n</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["obj_param"] == nlohmann::json::parse(R"({"key": "value"})"));
}

// test_extract_tool_calls_missing_closing_parameter_tag
TEST_CASE("step3p5: missing closing </parameter> handled gracefully") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "Let me check the weather for you:\n<tool_call>\n"
      "<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n"
      "</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["city"] == "Dallas");
  CHECK(args["state"] == "TX");
  CHECK(args["unit"] == "fahrenheit");
  REQUIRE(info.content.has_value());
  CHECK(info.content->find("Let me check the weather for you:") !=
        std::string::npos);
}

// test_extract_tool_calls_non_streaming_mixed_content_and_multiple_tool_calls
TEST_CASE("step3p5: non-streaming mixed content + multiple tool calls") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "hello<tool_call>\n<function=get_current_weather>\n<parameter=city>\n"
      "Dallas\n</parameter>\n<parameter=state>\nTX\n</parameter>\n</function>\n"
      "</tool_call>hi<tool_call>\n<function=calculate_area>\n<parameter=shape>\n"
      "rectangle\n</parameter>\n<parameter=dimensions>\n{\"width\": 10, "
      "\"height\": 5}\n</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(ArgsOf(info.tool_calls[0])["city"] == "Dallas");
  CHECK(info.tool_calls[1].function.name == "calculate_area");
  auto args1 = ArgsOf(info.tool_calls[1]);
  CHECK(args1["shape"] == "rectangle");
  CHECK(args1["dimensions"]["width"] == 10);
  CHECK(args1["dimensions"]["height"] == 5);
  REQUIRE(info.content.has_value());
  CHECK(info.content->find("hello") != std::string::npos);
  CHECK(info.content->find("hi") != std::string::npos);
  CHECK(info.content->find("<function=get_current_weather>") ==
        std::string::npos);
}

// test_extract_tool_calls_non_streaming_multiple_tool_calls_no_content_between
TEST_CASE("step3p5: non-streaming multiple tool calls, no content between") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "hello<tool_call>\n<function=get_current_weather>\n<parameter=city>\n"
      "Dallas\n</parameter>\n<parameter=state>\nTX\n</parameter>\n</function>\n"
      "</tool_call><tool_call>\n<function=calculate_area>\n<parameter=shape>\n"
      "rectangle\n</parameter>\n<parameter=dimensions>\n{\"width\": 10, "
      "\"height\": 5}\n</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
  CHECK(info.tool_calls[1].function.name == "calculate_area");
  REQUIRE(info.content.has_value());
  CHECK(info.content->find("hello") != std::string::npos);
}

// ─── Streaming ───────────────────────────────────────────────────────────────

// test_extract_tool_calls_streaming[no_tools]
TEST_CASE("step3p5 streaming: plain content, no tools") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out = "This is a test without tools";
  Recon r = RunCharByChar(parser, out, req);
  CHECK(r.tools.empty());
  CHECK(r.content == out);
}

// test_extract_tool_calls_streaming[single_tool]
TEST_CASE("step3p5 streaming: single tool call reconstructs") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "</parameter>\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\n"
      "fahrenheit\n</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_current_weather");
  CHECK(r.tools[0].id.has_value());
  CHECK(r.tools[0].type == "function");
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Dallas");
  CHECK(args["state"] == "TX");
  CHECK(args["unit"] == "fahrenheit");
  CHECK(r.content.empty());
}

// test_extract_tool_calls_streaming[single_tool_with_content]
TEST_CASE("step3p5 streaming: single tool call with leading content") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "Sure! Let me check the weather for you.<tool_call>\n"
      "<function=get_current_weather>\n<parameter=city>\nDallas\n</parameter>\n"
      "<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n"
      "</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_current_weather");
  CHECK(r.content == "Sure! Let me check the weather for you.");
}

// test_extract_tool_calls_streaming[single_tool_multiline_param]
TEST_CASE("step3p5 streaming: multiline object param reconstructs") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=calculate_area>\n<parameter=shape>\nrectangle\n"
      "</parameter>\n<parameter=dimensions>\n{\"width\": 10, \n \"height\": 20}\n"
      "</parameter>\n<parameter=precision>\n2\n</parameter>\n</function>\n"
      "</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 1);
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["shape"] == "rectangle");
  CHECK(args["dimensions"]["width"] == 10);
  CHECK(args["dimensions"]["height"] == 20);
  CHECK(args["precision"] == 2);
}

// test_extract_tool_calls_streaming[parallel_tools]
TEST_CASE("step3p5 streaming: parallel tool calls reconstruct") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "</parameter>\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\n"
      "fahrenheit\n</parameter>\n</function>\n</tool_call>\n"
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nOrlando\n"
      "</parameter>\n<parameter=state>\nFL\n</parameter>\n<parameter=unit>\n"
      "celsius\n</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 2);
  CHECK(nlohmann::json::parse(r.tools[0].arguments)["city"] == "Dallas");
  CHECK(nlohmann::json::parse(r.tools[1].arguments)["city"] == "Orlando");
  CHECK(nlohmann::json::parse(r.tools[1].arguments)["unit"] == "celsius");
}

// test_extract_tool_calls_streaming[tool_with_typed_params]
TEST_CASE("step3p5 streaming: typed params reconstruct") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "Let me calculate that area for you.<tool_call>\n"
      "<function=calculate_area>\n<parameter=shape>\ncircle\n</parameter>\n"
      "<parameter=dimensions>\n{\"radius\": 15.5}\n</parameter>\n"
      "<parameter=precision>\n3\n</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 1);
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["shape"] == "circle");
  CHECK(args["dimensions"]["radius"] == 15.5);
  CHECK(args["precision"] == 3);
  CHECK(r.content == "Let me calculate that area for you.");
}

// test_extract_tool_calls_streaming_missing_closing_tag
TEST_CASE("step3p5 streaming: missing closing </parameter>") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "Let me check the weather for you:\n<tool_call>\n"
      "<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n"
      "</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  CHECK(r.content.find("Let me check the weather for you:") !=
        std::string::npos);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_current_weather");
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Dallas");
  CHECK(args["state"] == "TX");
  CHECK(args["unit"] == "fahrenheit");
}

// test_extract_tool_calls_streaming_incremental
TEST_CASE("step3p5 streaming: truly incremental (header then arg diffs)") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "I'll check the weather.<tool_call>\n<function=get_current_weather>\n"
      "<parameter=city>\nDallas\n</parameter>\n<parameter=state>\nTX\n"
      "</parameter>\n</function>\n</tool_call>";

  std::vector<DeltaMessage> chunks;
  std::string previous;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::string delta = out.substr(i, 1);
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (dm.has_value()) chunks.push_back(*dm);
  }

  CHECK(chunks.size() > 3);
  // First chunk should be content.
  REQUIRE(!chunks.empty());
  CHECK(chunks[0].content.has_value());
  CHECK((!chunks[0].tool_calls.has_value() || chunks[0].tool_calls->empty()));

  // A header chunk with id + name + empty arguments.
  bool header_found = false;
  for (const DeltaMessage& c : chunks) {
    if (c.tool_calls.has_value() && !c.tool_calls->empty() &&
        (*c.tool_calls)[0].id.has_value()) {
      header_found = true;
      CHECK((*c.tool_calls)[0].function.name == "get_current_weather");
      CHECK((*c.tool_calls)[0].type == "function");
      CHECK((*c.tool_calls)[0].function.arguments == std::string(""));
      break;
    }
  }
  CHECK(header_found);

  // Incremental argument chunks concatenate to valid JSON.
  std::string full_args;
  int arg_chunks = 0;
  for (const DeltaMessage& c : chunks) {
    if (c.tool_calls.has_value() && !c.tool_calls->empty()) {
      const auto& a = (*c.tool_calls)[0].function.arguments;
      if (a.has_value() && !a->empty()) {
        full_args += *a;
        ++arg_chunks;
      }
    }
  }
  CHECK(arg_chunks > 1);
  auto parsed = nlohmann::json::parse(full_args);
  CHECK(parsed["city"] == "Dallas");
  CHECK(parsed["state"] == "TX");
}

// test_extract_tool_calls_streaming_mixed_content_and_multiple_tool_calls
TEST_CASE("step3p5 streaming: mixed content + multiple tool calls") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "hello<tool_call>\n<function=get_current_weather>\n<parameter=city>\n"
      "Dallas\n</parameter>\n<parameter=state>\nTX\n</parameter>\n</function>\n"
      "</tool_call>hi<tool_call>\n<function=calculate_area>\n<parameter=shape>\n"
      "rectangle\n</parameter>\n<parameter=dimensions>\n{\"width\": 10, "
      "\"height\": 5}\n</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 2);
  CHECK(r.tools[0].name == "get_current_weather");
  CHECK(nlohmann::json::parse(r.tools[0].arguments)["city"] == "Dallas");
  CHECK(r.tools[1].name == "calculate_area");
  auto args1 = nlohmann::json::parse(r.tools[1].arguments);
  CHECK(args1["dimensions"]["width"] == 10);
  CHECK(args1["dimensions"]["height"] == 5);
  CHECK(r.content.find("hello") != std::string::npos);
  CHECK(r.content.find("hi") != std::string::npos);
  CHECK(r.content.find("hi") > r.content.find("hello"));
  CHECK(r.content.find("<function=get_current_weather>") == std::string::npos);
}

// test_extract_tool_calls_streaming_full_input_mixed_content_and_multiple_tool_calls
TEST_CASE("step3p5 streaming: whole input as a single delta") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "hello<tool_call>\n<function=get_current_weather>\n<parameter=city>\n"
      "Dallas\n</parameter>\n<parameter=state>\nTX\n</parameter>\n</function>\n"
      "</tool_call>hi<tool_call>\n<function=calculate_area>\n<parameter=shape>\n"
      "rectangle\n</parameter>\n<parameter=dimensions>\n{\"width\": 10, "
      "\"height\": 5}\n</parameter>\n</function>\n</tool_call>";
  Recon r = RunChunks(parser, {out}, req);
  REQUIRE(r.tools.size() == 2);
  CHECK(r.tools[0].name == "get_current_weather");
  CHECK(nlohmann::json::parse(r.tools[0].arguments)["city"] == "Dallas");
  CHECK(r.tools[1].name == "calculate_area");
  CHECK(nlohmann::json::parse(r.tools[1].arguments)["dimensions"]["width"] == 10);
  CHECK(r.content.find("hello") != std::string::npos);
  CHECK(r.content.find("hi") != std::string::npos);
}

// test_extract_tool_calls_streaming_multiple_tool_calls_no_content_between
TEST_CASE("step3p5 streaming: multiple tool calls, no content between") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "hello<tool_call>\n<function=get_current_weather>\n<parameter=city>\n"
      "Dallas\n</parameter>\n<parameter=state>\nTX\n</parameter>\n</function>\n"
      "</tool_call><tool_call>\n<function=calculate_area>\n<parameter=shape>\n"
      "rectangle\n</parameter>\n<parameter=dimensions>\n{\"width\": 10, "
      "\"height\": 5}\n</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 2);
  CHECK(r.tools[0].name == "get_current_weather");
  CHECK(r.tools[1].name == "calculate_area");
  CHECK(nlohmann::json::parse(r.tools[1].arguments)["dimensions"]["height"] == 5);
  CHECK(r.content.find("hello") != std::string::npos);
  CHECK(r.content.find("<function=calculate_area>") == std::string::npos);
}

// test_extract_tool_calls_streaming_multi_token_chunk_boundary
TEST_CASE("step3p5 streaming: chunk boundary does not close a new tool_call") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::vector<std::string> chunks = {
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nSys",
      "\n</parameter>\n</function>\n",
      "</tool_call><tool_call>\n<",
      "function=calculate_area>\n<parameter=shape>\nrectangle",
      "</parameter>\n</function>\n</tool_call>",
  };
  Recon r = RunChunks(parser, chunks, req);
  REQUIRE(r.tools.size() == 2);
  CHECK(r.tools[0].name == "get_current_weather");
  CHECK(r.tools[1].name == "calculate_area");
}

// test_streaming_mtp_variable_chunks
TEST_CASE("step3p5 streaming: MTP variable-size chunks across boundaries") {
  Step3p5ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::vector<std::string> chunks = {
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\n",
      "Dallas\n</parameter>\n<parameter=state>\nTX",
      "\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>",
      "\n</function>\n</tool_call>",
  };
  Recon r = RunChunks(parser, chunks, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_current_weather");
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Dallas");
  CHECK(args["state"] == "TX");
  CHECK(args["unit"] == "fahrenheit");
}

// test_streaming_multi_token_per_step
TEST_CASE("step3p5 streaming: MTP large chunks match char-by-char reference") {
  auto req = RequestWith(SampleTools());
  const std::vector<std::string> mtp_chunks = {
      "<tool_call>\n<function=get_current_weather>\n"
      "<parameter=city>\nDallas\n</parameter>\n"
      "<parameter=state>\nTX",
      "\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n"
      "</function>\n</tool_call>\n"
      "<tool_call>\n<function=get_current_weather>\n"
      "<parameter=city>\nOrlando\n</parameter>\n"
      "<parameter=state>\nFL\n</parameter>\n"
      "<parameter=unit>\ncelsius\n</parameter>\n"
      "</function>\n</tool_call>",
  };
  const std::string model_output =
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "</parameter>\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\n"
      "fahrenheit\n</parameter>\n</function>\n</tool_call>\n"
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nOrlando\n"
      "</parameter>\n<parameter=state>\nFL\n</parameter>\n<parameter=unit>\n"
      "celsius\n</parameter>\n</function>\n</tool_call>";

  Step3p5ToolParser mtp_parser;
  Recon mtp = RunChunks(mtp_parser, mtp_chunks, req);

  Step3p5ToolParser ref_parser;
  Recon ref = RunCharByChar(ref_parser, model_output, req);

  REQUIRE(mtp.tools.size() == 2);
  REQUIRE(ref.tools.size() == 2);
  for (int idx = 0; idx < 2; ++idx) {
    CHECK(mtp.tools[idx].name == ref.tools[idx].name);
    CHECK(nlohmann::json::parse(mtp.tools[idx].arguments) ==
          nlohmann::json::parse(ref.tools[idx].arguments));
  }
}

// ─── Factory ─────────────────────────────────────────────────────────────────

TEST_CASE("step3p5: factory registration") {
  auto p = get_tool_parser("step3p5");
  REQUIRE(p != nullptr);
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "</parameter>\n</function>\n</tool_call>";
  auto info = p->extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_current_weather");
}
