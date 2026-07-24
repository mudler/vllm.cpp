// Tests for the Qwen3-Coder XML tool-call parser (Qwen3CoderToolParser),
// registered as "qwen3_coder" / "qwen3_xml" / "mimo".
//
// REIMPLEMENTED-FROM-WIRE-FORMAT held to the upstream extraction tests. Ports
// EVERY case from tests/tool_parsers/test_qwen3coder_tool_parser.py @ e24d1b24:
// the non-streaming parametrized family + no_tools + fallback_no_tags +
// type_conversion + anyof_type_conversion + missing_closing_parameter_tag +
// malformed_xml_no_gt_delimiter + none_tool_calls_filtered +
// anyof_parameter_not_double_encoded + no_double_serialization_string_args, the
// streaming parametrized family + anyof streaming + missing_closing_tag +
// incremental + missing_opening_tag, and the explicit-chunk streaming edges
// (multi_param_single_chunk, complete_tool_call_single_delta,
// next_tool_call_starts_in_close_delta), plus char-by-char split edges.
//
// The upstream streaming suite drives a real tokenizer token-by-token; the
// text-only C++ seam has no tokenizer, so we drive the strictest cadence
// (character-by-character) as the token-by-token analogue AND the explicit chunk
// lists the upstream regressions use verbatim.
//
// DEVIATION (tools on the request, not the constructor): the upstream fixtures
// construct the parser with tools and pass a tools-less request; the C++ seam
// exposes tools only on the ChatCompletionRequest, so the ported requests carry
// the tools. See qwen3_coder.h DEVIATION 2.
//
// The upstream get_structural_tag / adjust_request tests are engine/xgrammar
// specific (the `qwen_3_coder` xgrammar builtin) and are NOT portable to the
// flat native structural tag; the family's documented nullopt is covered in
// test_structural_tags.cpp instead.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/qwen3_coder.h"

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

// test_qwen3coder_tool_parser.py:44-61 (WEATHER_PARAMS / AREA_PARAMS).
std::vector<ChatCompletionToolsParam> SampleTools() {
  return {
      MakeTool("get_current_weather", nlohmann::json::parse(R"({
                 "type": "object",
                 "properties": {
                   "city": {"type": "string", "description": "The city name"},
                   "state": {"type": "string", "description": "The state code"},
                   "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]}
                 },
                 "required": ["city", "state"]
               })")),
      MakeTool("calculate_area", nlohmann::json::parse(R"({
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

nlohmann::json ArgsOf(const ToolCall& tc) {
  return nlohmann::json::parse(tc.function.arguments);
}

// Reconstruct tool-call state the way the Python tests accumulate it, tracking
// per-tool name-emission counts / id / type for the strict assertions.
struct ToolState {
  std::optional<std::string> id;
  std::optional<std::string> name;
  int name_count = 0;
  std::string arguments;
  std::optional<std::string> type;
};
struct Recon {
  std::map<int, ToolState> tools;
  std::string content;
  bool role_seen = false;
};

void Apply(Recon& r, const DeltaMessage& m) {
  if (m.role.has_value()) r.role_seen = true;
  if (m.content.has_value()) r.content += *m.content;
  if (!m.tool_calls.has_value()) return;
  for (const DeltaToolCall& tc : *m.tool_calls) {
    ToolState& st = r.tools[tc.index];
    if (tc.id.has_value()) st.id = tc.id;
    if (tc.type.has_value()) st.type = tc.type;
    if (tc.function.name.has_value() && !tc.function.name->empty()) {
      st.name = tc.function.name;
      ++st.name_count;
    }
    if (tc.function.arguments.has_value()) st.arguments += *tc.function.arguments;
  }
}

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

std::vector<DeltaMessage> CollectCharByChar(ToolParser& parser,
                                            const std::string& out,
                                            const ChatCompletionRequest& req) {
  std::vector<DeltaMessage> chunks;
  std::string previous;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::string delta = out.substr(i, 1);
    const std::string current = previous + delta;
    auto dm = parser.extract_tool_calls_streaming(previous, current, delta, req);
    previous = current;
    if (dm.has_value()) chunks.push_back(*dm);
  }
  return chunks;
}

}  // namespace

// ─── Non-streaming ───────────────────────────────────────────────────────────

// test_extract_tool_calls_no_tools
TEST_CASE("qwen3_coder: no tool calls is plain content") {
  Qwen3CoderToolParser parser;
  const std::string out = "This is a test response without any tool calls";
  ChatCompletionRequest req;
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// test_extract_tool_calls[single_tool]
TEST_CASE("qwen3_coder: single tool call") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder: single tool call with leading content") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder: multiline object parameter") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder: parallel tool calls") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder: typed params (float object)") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder: fallback parsing without <tool_call> wrapper") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder: parameter type conversion") {
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
  Qwen3CoderToolParser parser;
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
  CHECK(args["int_param"].is_number_integer());
  CHECK(args["float_param"] == 3.14);
  CHECK(args["bool_param"] == true);
  CHECK(args["bool_param"].is_boolean());
  CHECK(args["str_param"] == "hello world");
  CHECK(args["str_param"].is_string());
  CHECK(args["obj_param"] == nlohmann::json::parse(R"({"key": "value"})"));
}

// test_extract_tool_calls_anyof_type_conversion
TEST_CASE("qwen3_coder: anyOf / type-as-array type conversion") {
  const auto tools = std::vector<ChatCompletionToolsParam>{
      MakeTool("test_anyof", nlohmann::json::parse(R"({
        "type": "object",
        "properties": {
          "anyof_int": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 5},
          "anyof_str": {"anyOf": [{"type": "string"}, {"type": "null"}]},
          "anyof_array": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
          "anyof_obj": {"anyOf": [{"type": "object"}, {"type": "null"}]},
          "type_as_array": {"type": ["integer", "null"]},
          "multi_non_null": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
        }
      })"))};
  Qwen3CoderToolParser parser;
  auto req = RequestWith(tools);
  const std::string out =
      "<tool_call>\n<function=test_anyof>\n<parameter=anyof_int>\n5\n"
      "</parameter>\n<parameter=anyof_str>\nhello\n</parameter>\n"
      "<parameter=anyof_array>\n[\"a\", \"b\", \"c\"]\n</parameter>\n"
      "<parameter=anyof_obj>\n{\"key\": \"value\"}\n</parameter>\n"
      "<parameter=type_as_array>\n42\n</parameter>\n"
      "<parameter=multi_non_null>\nsome text\n</parameter>\n</function>\n"
      "</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["anyof_int"] == 5);
  CHECK(args["anyof_int"].is_number_integer());
  CHECK(args["anyof_str"] == "hello");
  CHECK(args["anyof_str"].is_string());
  CHECK(args["anyof_array"] == nlohmann::json::parse(R"(["a", "b", "c"])"));
  CHECK(args["anyof_array"].is_array());
  CHECK(args["anyof_obj"] == nlohmann::json::parse(R"({"key": "value"})"));
  CHECK(args["anyof_obj"].is_object());
  CHECK(args["type_as_array"] == 42);
  CHECK(args["type_as_array"].is_number_integer());
  CHECK(args["multi_non_null"] == "some text");
  CHECK(args["multi_non_null"].is_string());
}

// test_extract_tool_calls_missing_closing_parameter_tag
TEST_CASE("qwen3_coder: missing closing </parameter> handled gracefully") {
  Qwen3CoderToolParser parser;
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

// test_malformed_xml_no_gt_delimiter (PR #36774)
TEST_CASE("qwen3_coder: malformed XML without '>' must not crash") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=get_current_weather\n"
      "<parameter=city>Dallas</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  // A list result with no spurious entries (the malformed function is dropped).
  for (const ToolCall& tc : info.tool_calls) {
    CHECK(!tc.function.name.empty());
  }
}

// test_none_tool_calls_filtered (PR #36774)
TEST_CASE("qwen3_coder: malformed tool calls filtered, valid one kept") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=bad_func_no_gt\n</function>\n</tool_call>\n"
      "<tool_call>\n<function=get_current_weather>\n"
      "<parameter=city>Dallas</parameter>\n<parameter=state>TX</parameter>\n"
      "</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  int valid = 0;
  for (const ToolCall& tc : info.tool_calls) {
    if (tc.function.name == "get_current_weather") {
      ++valid;
      auto args = ArgsOf(tc);
      CHECK(args["city"] == "Dallas");
      CHECK(args["state"] == "TX");
    }
  }
  CHECK(valid == 1);
}

// test_anyof_parameter_not_double_encoded (PR #36032)
TEST_CASE("qwen3_coder: anyOf object parameter not double-encoded") {
  const auto tools = std::vector<ChatCompletionToolsParam>{
      MakeTool("update_record", nlohmann::json::parse(R"({
        "type": "object",
        "properties": {
          "data": {"anyOf": [{"type": "object"}, {"type": "null"}]}
        }
      })"))};
  Qwen3CoderToolParser parser;
  auto req = RequestWith(tools);
  const std::string out =
      "<tool_call>\n<function=update_record>\n"
      "<parameter=data>{\"key\": \"value\", \"count\": 42}</parameter>\n"
      "</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  auto args = ArgsOf(info.tool_calls[0]);
  CHECK(args["data"].is_object());
  CHECK(args["data"] == nlohmann::json::parse(R"({"key": "value", "count": 42})"));
}

// test_no_double_serialization_string_args (PR #35615)
TEST_CASE("qwen3_coder: string arguments not double-serialized") {
  const auto tools = std::vector<ChatCompletionToolsParam>{
      MakeTool("greet", nlohmann::json::parse(R"({
        "type": "object",
        "properties": {"message": {"type": "string"}}
      })"))};
  Qwen3CoderToolParser parser;
  auto req = RequestWith(tools);
  const std::string out =
      "<tool_call>\n<function=greet>\n"
      "<parameter=message>hello world</parameter>\n</function>\n</tool_call>";
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  const std::string raw = info.tool_calls[0].function.arguments;
  auto args = nlohmann::json::parse(raw);
  CHECK(args["message"] == "hello world");
  CHECK(raw.find("\\\"hello world\\\"") == std::string::npos);
}

// ─── Streaming ───────────────────────────────────────────────────────────────

// test_extract_tool_calls_streaming[no_tools]
TEST_CASE("qwen3_coder streaming: plain content, no tools") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out = "This is a test without tools";
  Recon r = RunCharByChar(parser, out, req);
  CHECK(r.tools.empty());
  CHECK(r.content == out);
  CHECK_FALSE(r.role_seen);
}

// test_extract_tool_calls_streaming[single_tool]
TEST_CASE("qwen3_coder streaming: single tool call reconstructs") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
      "</parameter>\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\n"
      "fahrenheit\n</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_current_weather");
  CHECK(r.tools[0].name_count == 1);  // name streamed exactly once.
  CHECK(r.tools[0].id.has_value());
  CHECK(r.tools[0].type == "function");
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Dallas");
  CHECK(args["state"] == "TX");
  CHECK(args["unit"] == "fahrenheit");
  CHECK(r.content.empty());
}

// test_extract_tool_calls_streaming[single_tool_with_content]
TEST_CASE("qwen3_coder streaming: single tool call with leading content") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder streaming: multiline object param reconstructs") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder streaming: parallel tool calls reconstruct") {
  Qwen3CoderToolParser parser;
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
  CHECK(r.tools[0].name_count == 1);
  CHECK(r.tools[1].name_count == 1);
}

// test_extract_tool_calls_streaming[tool_with_typed_params]
TEST_CASE("qwen3_coder streaming: typed params reconstruct") {
  Qwen3CoderToolParser parser;
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

// test_extract_tool_calls_anyof_type_conversion_streaming
TEST_CASE("qwen3_coder streaming: anyOf types reconstruct end-to-end") {
  const auto tools = std::vector<ChatCompletionToolsParam>{
      MakeTool("search_web", nlohmann::json::parse(R"({
        "type": "object",
        "properties": {
          "query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
          "count": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 5},
          "verbose": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
        }
      })"))};
  Qwen3CoderToolParser parser;
  auto req = RequestWith(tools);
  const std::string out =
      "<tool_call>\n<function=search_web>\n<parameter=query>\n"
      "vllm tool parser\n</parameter>\n<parameter=count>\n10\n</parameter>\n"
      "<parameter=verbose>\ntrue\n</parameter>\n</function>\n</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "search_web");
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["query"] == "vllm tool parser");
  CHECK(args["query"].is_string());
  CHECK(args["count"] == 10);
  CHECK(args["count"].is_number_integer());
  CHECK(args["verbose"] == true);
  CHECK(args["verbose"].is_boolean());
}

// test_extract_tool_calls_streaming_missing_closing_tag
TEST_CASE("qwen3_coder streaming: missing closing </parameter>") {
  Qwen3CoderToolParser parser;
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
TEST_CASE("qwen3_coder streaming: truly incremental (header then arg diffs)") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "I'll check the weather.<tool_call>\n<function=get_current_weather>\n"
      "<parameter=city>\nDallas\n</parameter>\n<parameter=state>\nTX\n"
      "</parameter>\n</function>\n</tool_call>";

  std::vector<DeltaMessage> chunks = CollectCharByChar(parser, out, req);

  CHECK(chunks.size() > 3);
  REQUIRE(!chunks.empty());
  // First chunk should be content, no tool calls.
  CHECK(chunks[0].content.has_value());
  CHECK((!chunks[0].tool_calls.has_value() || chunks[0].tool_calls->empty()));

  // A header chunk carrying id + name + type.
  bool header_found = false;
  for (const DeltaMessage& c : chunks) {
    if (c.tool_calls.has_value() && !c.tool_calls->empty() &&
        (*c.tool_calls)[0].id.has_value()) {
      header_found = true;
      CHECK((*c.tool_calls)[0].function.name == "get_current_weather");
      CHECK((*c.tool_calls)[0].type == "function");
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
  CHECK(arg_chunks >= 1);
  auto parsed = nlohmann::json::parse(full_args);
  CHECK(parsed["city"] == "Dallas");
  CHECK(parsed["state"] == "TX");
}

// test_extract_tool_calls_streaming_missing_opening_tag
TEST_CASE("qwen3_coder streaming: missing opening <tool_call> tag") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out =
      "I'll check the weather for you.\n\n<function=get_current_weather>\n"
      "<parameter=city>\nDallas\n</parameter>\n<parameter=state>\nTX\n"
      "</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n"
      "</tool_call>";
  Recon r = RunCharByChar(parser, out, req);
  CHECK(r.content.find("I'll check the weather for you.") != std::string::npos);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].id.has_value());
  CHECK(r.tools[0].type == "function");
  CHECK(r.tools[0].name == "get_current_weather");
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Dallas");
  CHECK(args["state"] == "TX");
  CHECK(args["unit"] == "fahrenheit");
}

// test_streaming_multi_param_single_chunk (PR #35615): one delta delivers all
// three parameters at once (speculative decode).
TEST_CASE("qwen3_coder streaming: multiple params delivered in a single chunk") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::vector<std::string> deltas = {
      "<tool_call>",
      "\n<function=get_current_weather>",
      "\n",
      "<parameter=city>\nDallas\n</parameter>"
      "\n<parameter=state>\nTX\n</parameter>"
      "\n<parameter=unit>\nfahrenheit\n</parameter>",
      "\n</function>",
      "\n</tool_call>",
  };
  Recon r = RunChunks(parser, deltas, req);
  REQUIRE(r.tools.size() == 1);
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Dallas");
  CHECK(args["state"] == "TX");
  CHECK(args["unit"] == "fahrenheit");
}

// test_streaming_complete_tool_call_single_delta
TEST_CASE("qwen3_coder streaming: complete tool call in one delta") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::vector<std::string> deltas = {
      "<tool_call>\n<function=get_current_weather>\n"
      "<parameter=city>\nDallas\n</parameter>\n"
      "<parameter=state>\nTX\n</parameter>\n</function>\n</tool_call>",
  };
  Recon r = RunChunks(parser, deltas, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_current_weather");
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args == nlohmann::json::parse(R"({"city": "Dallas", "state": "TX"})"));
}

// test_streaming_next_tool_call_starts_in_close_delta
TEST_CASE("qwen3_coder streaming: next tool call starts in the close delta") {
  Qwen3CoderToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::vector<std::string> deltas = {
      "<tool_call>\n",
      "<function=get_current_weather>\n",
      "<parameter=city>\nDallas\n</parameter>\n",
      "<parameter=state>\nTX\n</parameter>\n",
      "</function>",
      "\n</tool_call>\n"
      "<tool_call>\n<function=get_current_weather>\n"
      "<parameter=city>\nOrlando\n</parameter>\n"
      "<parameter=state>\nFL\n</parameter>\n</function>\n</tool_call>",
  };
  Recon r = RunChunks(parser, deltas, req);
  REQUIRE(r.tools.size() == 2);
  CHECK(nlohmann::json::parse(r.tools[0].arguments) ==
        nlohmann::json::parse(R"({"city": "Dallas", "state": "TX"})"));
  CHECK(nlohmann::json::parse(r.tools[1].arguments) ==
        nlohmann::json::parse(R"({"city": "Orlando", "state": "FL"})"));
}

// ─── Factory ─────────────────────────────────────────────────────────────────

TEST_CASE("qwen3_coder: factory registration (qwen3_coder / qwen3_xml / mimo)") {
  for (const char* name_cstr : {"qwen3_coder", "qwen3_xml", "mimo"}) {
    const std::string name = name_cstr;
    CAPTURE(name);
    auto p = get_tool_parser(name);
    REQUIRE(p != nullptr);
    auto req = RequestWith(SampleTools());
    const std::string out =
        "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
        "</parameter>\n</function>\n</tool_call>";
    auto info = p->extract_tool_calls(out, req);
    CHECK(info.tools_called == true);
    REQUIRE(info.tool_calls.size() == 1);
    CHECK(info.tool_calls[0].function.name == "get_current_weather");
    CHECK(ArgsOf(info.tool_calls[0])["city"] == "Dallas");
  }
}
