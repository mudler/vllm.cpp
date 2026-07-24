// Tests for the "granite4" tool-call parser (Granite4ToolParser).
// Ported from: tests/tool_parsers/test_granite4_tool_parser.py @ e24d1b24
// (test_tool_call_parser_complex - interleaved text + `<tool_call> {json}
// </tool_call>` blocks, streamed in randomly sized chunks). We have no tokenizer,
// so we build the input from JSON and drive it at two fixed cadences (1 char and
// 4 chars per delta), reconstructing content + tool calls the same way the Python
// test does. The upstream test uses string-typed arguments for `find_bbox` and
// object-typed arguments for `get_stock_price`; both round-trip to the same dicts.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/granite4.h"

using namespace vllm::entrypoints::openai;

namespace {

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

// test_granite4_tool_parser.py:19-38 (create_complex_input). Returns the three
// expected {name, arguments(dict)} tool calls.
std::vector<nlohmann::json> ExpectedToolCalls() {
  const nlohmann::json coord = {
      {"coordinates",
       nlohmann::json::array({nlohmann::json::array({23.54, 43.1}),
                              nlohmann::json::array({-12.2, 54.3}),
                              nlohmann::json::array({4, 5})})},
      {"coordinate_type", "latlong"}};
  const nlohmann::json stock = {{"symbol", "AAPL"},
                                {"start_date", "2021-01-01"},
                                {"end_date", "2021-12-31"}};
  return {{{"name", "find_bbox"}, {"arguments", coord}},
          {{"name", "get_stock_price"}, {"arguments", stock}},
          {{"name", "find_bbox"}, {"arguments", coord}}};
}

// Build the raw model output: text_messages interleaved with `<tool_call> {json}
// </tool_call>` blocks. `find_bbox` uses STRING-typed arguments (the coordinate
// dict json-encoded), exercising granite4's dump_args str passthrough.
std::string BuildComplexOutput(const std::vector<std::string>& text_messages) {
  auto calls = ExpectedToolCalls();
  // Re-encode find_bbox arguments as a JSON *string* (upstream create_string_args).
  const std::string coord_str = calls[0]["arguments"].dump();
  nlohmann::json c0 = {{"name", "find_bbox"}, {"arguments", coord_str}};
  nlohmann::json c1 = calls[1];  // object-typed arguments
  nlohmann::json c2 = {{"name", "find_bbox"}, {"arguments", coord_str}};

  auto block = [](const nlohmann::json& c) {
    return "<tool_call> " + c.dump() + " </tool_call>";
  };
  return text_messages[0] + block(c0) + text_messages[1] + block(c1) +
         text_messages[2] + block(c2) + text_messages[3];
}

struct ReconTool {
  std::string name;
  std::string arguments;
};
struct Recon {
  std::vector<ReconTool> tools;
  std::string content;
};

void Append(Recon& r, const DeltaMessage& m) {
  if (m.content.has_value()) r.content += *m.content;
  if (!m.tool_calls.has_value()) return;
  for (const DeltaToolCall& tc : *m.tool_calls) {
    ReconTool rt;
    if (tc.function.name.has_value()) rt.name = *tc.function.name;
    if (tc.function.arguments.has_value()) rt.arguments = *tc.function.arguments;
    // granite4 emits the whole tool call (name + args) in one delta.
    r.tools.push_back(std::move(rt));
  }
}

// Drive with a fixed chunk size (granite4 streaming ignores previous/current
// text; it only consumes delta_text via its internal look-ahead buffer).
Recon RunStreaming(ToolParser& parser, const std::string& out, std::size_t chunk) {
  Recon r;
  ChatCompletionRequest req;
  for (std::size_t i = 0; i < out.size(); i += chunk) {
    const std::string delta = out.substr(i, chunk);
    auto dm = parser.extract_tool_calls_streaming("", "", delta, req);
    if (dm.has_value()) Append(r, *dm);
  }
  return r;
}

const std::vector<std::string> kTextMessages = {
    "Here goes the bbox call: \n", " Now the stock price call: \n ",
    " Now another bbox call: \n ", " See? I'm a helpful assistant."};

}  // namespace

// ─── Streaming (the ported complex case) ─────────────────────────────────────

TEST_CASE("granite4 streaming: complex interleaved reconstruction") {
  const std::string out = BuildComplexOutput(kTextMessages);
  const auto expected = ExpectedToolCalls();

  for (std::size_t chunk : {std::size_t{1}, std::size_t{4}}) {
    Granite4ToolParser parser;
    Recon r = RunStreaming(parser, out, chunk);

    std::string joined;
    for (const std::string& t : kTextMessages) joined += t;
    CHECK(r.content == joined);

    REQUIRE(r.tools.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      CHECK(r.tools[i].name == expected[i]["name"].get<std::string>());
      auto parsed = nlohmann::json::parse(r.tools[i].arguments);
      CHECK(parsed == expected[i]["arguments"]);
    }
  }
}

// ─── Non-streaming ───────────────────────────────────────────────────────────

TEST_CASE("granite4: complex interleaved (non-streaming)") {
  const std::string out = BuildComplexOutput(kTextMessages);
  const auto expected = ExpectedToolCalls();
  Granite4ToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);

  std::string joined;
  for (const std::string& t : kTextMessages) joined += t;
  CHECK(info.tools_called == true);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == joined);
  REQUIRE(info.tool_calls.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(info.tool_calls[i].function.name == expected[i]["name"].get<std::string>());
    CHECK(nlohmann::json::parse(info.tool_calls[i].function.arguments) ==
          expected[i]["arguments"]);
  }
}

TEST_CASE("granite4: single tool call, no surrounding text") {
  Granite4ToolParser parser;
  const std::string out =
      "<tool_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Tokyo\"}} </tool_call>";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].id.rfind("chatcmpl-tool-", 0) == 0);
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments)["city"] == "Tokyo");
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("granite4: no tool calls is plain content") {
  Granite4ToolParser parser;
  const std::string out = "This is a regular response without any tool calls.";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("granite4: incomplete tool call falls back to content") {
  // An unterminated block (no </tool_call>) -> assert failure -> full content.
  Granite4ToolParser parser;
  const std::string out = "<tool_call> {\"name\": \"f\", \"arguments\": {}}";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("granite4: two start tokens in a row fall back to content") {
  Granite4ToolParser parser;
  const std::string out =
      "<tool_call> <tool_call> {\"name\": \"f\", \"arguments\": {}} </tool_call>";
  auto req = empty_request();
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

TEST_CASE("granite4: empty output yields no tool calls") {
  // granite4_tool_parser.py:145 `content = content or None`: an empty join maps
  // to None (not the empty string) on the success path.
  Granite4ToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls("", req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  CHECK_FALSE(info.content.has_value());
}

TEST_CASE("granite4 streaming: plain content yields no tool calls") {
  Granite4ToolParser parser;
  const std::string out = "This is a regular response without any tool calls.";
  Recon r = RunStreaming(parser, out, 1);
  CHECK(r.tools.empty());
  CHECK(r.content == out);
}

// ─── Factory ─────────────────────────────────────────────────────────────────

TEST_CASE("granite4: factory registration") {
  auto p = get_tool_parser("granite4");
  REQUIRE(p != nullptr);
  auto req = empty_request();
  auto info = p->extract_tool_calls(
      "<tool_call> {\"name\": \"f\", \"arguments\": {}} </tool_call>", req);
  CHECK(info.tools_called == true);
}
