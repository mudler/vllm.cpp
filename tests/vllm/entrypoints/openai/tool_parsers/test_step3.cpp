// Tests for the "step3" tool-call parser (Step3ToolParser).
// Ported from: tests/tool_parsers/test_step3_tool_parser.py @ e24d1b24 (the
// single TestStep3ToolParser config that drives common_tests.py) PLUS six added
// edges: single, parallel, empty params, no-tool, malformed, and streaming
// reconstruction.
//
// FIDELITY NOTE: upstream step3 is knowingly buggy and its Python config xfails
// almost all non-streaming cases and the parallel/reconstruction streaming
// cases. These tests assert the FAITHFUL (bug-for-bug) behaviour of the port,
// with the upstream root cause cited inline, rather than the "ideal" result the
// reference itself fails to produce:
//   - NON-STREAMING requires a `function<｜tool_sep｜>` prefix inside each call
//     that the real step3 format never emits, so it extracts NO tool calls.
//   - STREAMING never consumes the inter-call <｜tool_sep｜>, so a parallel
//     stream wedges after the first call (only the first tool is emitted).
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/step3.h"

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

// A schema so _cast_arguments has typed properties to coerce against.
std::vector<ChatCompletionToolsParam> SampleTools() {
  return {
      MakeTool("get_weather", nlohmann::json::parse(R"({
        "type": "object",
        "properties": {"city": {"type": "string"}}
      })")),
      MakeTool("get_time", nlohmann::json::parse(R"({
        "type": "object",
        "properties": {"timezone": {"type": "string"}}
      })")),
      MakeTool("refresh", nlohmann::json::parse(R"({
        "type": "object", "properties": {}
      })")),
  };
}

ChatCompletionRequest RequestWith(std::vector<ChatCompletionToolsParam> tools) {
  ChatCompletionRequest req;
  req.tools = std::move(tools);
  return req;
}

struct ToolState {
  std::optional<std::string> id;
  std::optional<std::string> name;
  std::string arguments;
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
    if (tc.function.name.has_value() && !tc.function.name->empty()) {
      st.name = tc.function.name;
    }
    if (tc.function.arguments.has_value()) st.arguments += *tc.function.arguments;
  }
}

// Feed the whole output as one delta.
Recon RunWhole(ToolParser& parser, const std::string& out,
               const ChatCompletionRequest& req) {
  Recon r;
  auto dm = parser.extract_tool_calls_streaming("", out, out, req);
  if (dm.has_value()) Apply(r, *dm);
  return r;
}

// Byte-by-byte streaming (markers are multi-byte; the parser matches on the
// accumulated current_text so byte-splitting is safe).
Recon RunByteByByte(ToolParser& parser, const std::string& out,
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

const std::string kSingle =
    "<｜tool_calls_begin｜><｜tool_call_begin｜>"
    "<steptml:invoke name=\"get_weather\">"
    "<steptml:parameter name=\"city\">Tokyo</steptml:parameter>"
    "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>";

const std::string kParallel =
    "<｜tool_calls_begin｜><｜tool_call_begin｜>"
    "<steptml:invoke name=\"get_weather\">"
    "<steptml:parameter name=\"city\">Tokyo</steptml:parameter>"
    "</steptml:invoke><｜tool_call_end｜><｜tool_sep｜>"
    "<｜tool_call_begin｜><steptml:invoke name=\"get_time\">"
    "<steptml:parameter name=\"timezone\">Asia/Tokyo</steptml:parameter>"
    "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>";

const std::string kEmptyParams =
    "<｜tool_calls_begin｜><｜tool_call_begin｜>"
    "<steptml:invoke name=\"refresh\"></steptml:invoke>"
    "<｜tool_call_end｜><｜tool_calls_end｜>";

}  // namespace

// ─── (ported) no tool calls ──────────────────────────────────────────────────

TEST_CASE("step3: no tool calls is plain content (non-streaming)") {
  Step3ToolParser parser;
  const std::string out = "This is a regular response without any tool calls.";
  ChatCompletionRequest req;
  auto info = parser.extract_tool_calls(out, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == out);
}

// ─── (edge) no-tool streaming ────────────────────────────────────────────────

TEST_CASE("step3: no-tool streaming passes content through") {
  Step3ToolParser parser;
  auto req = RequestWith(SampleTools());
  const std::string out = "This is a regular response without any tool calls.";
  Recon r = RunByteByByte(parser, out, req);
  CHECK(r.tools.empty());
  CHECK(r.content == out);
}

// ─── (edge) single tool call — streaming works ───────────────────────────────

TEST_CASE("step3: single tool call streams correctly") {
  Step3ToolParser parser;
  auto req = RequestWith(SampleTools());
  Recon r = RunByteByByte(parser, kSingle, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_weather");
  CHECK(r.tools[0].id.has_value());
  auto args = nlohmann::json::parse(r.tools[0].arguments);
  CHECK(args["city"] == "Tokyo");
}

// ─── (edge) streaming reconstruction ─────────────────────────────────────────

TEST_CASE("step3: single-delta call emits only the first (name) delta") {
  Step3ToolParser parser;
  auto req = RequestWith(SampleTools());
  // The cursor-based streaming returns on the FIRST delta it produces, so a
  // single whole-input call emits the tool NAME only; the arguments delta needs
  // a subsequent call (full incremental reconstruction is covered above).
  Recon r = RunWhole(parser, kSingle, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_weather");
  CHECK(r.tools[0].id.has_value());
  CHECK(r.tools[0].arguments.empty());
}

// ─── (edge) parallel — faithful: parser wedges after the first call ──────────

TEST_CASE("step3: parallel stream emits only the first call (upstream xfail)") {
  Step3ToolParser parser;
  auto req = RequestWith(SampleTools());
  Recon r = RunByteByByte(parser, kParallel, req);
  // Bug-for-bug: the inter-call <｜tool_sep｜> is never consumed, so only the
  // first tool call is emitted (upstream marks parallel streaming xfail).
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_weather");
  CHECK(nlohmann::json::parse(r.tools[0].arguments)["city"] == "Tokyo");
}

// ─── (edge) empty params — name emitted, no arguments delta ──────────────────

TEST_CASE("step3: empty-parameter call emits the name and no arguments") {
  Step3ToolParser parser;
  auto req = RequestWith(SampleTools());
  Recon r = RunByteByByte(parser, kEmptyParams, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "refresh");
  // cast_arguments({}) is empty, so no arguments delta is streamed.
  CHECK(r.tools[0].arguments.empty());
}

// ─── (edge) malformed input never throws ─────────────────────────────────────

TEST_CASE("step3: malformed inputs never throw") {
  const std::vector<std::string> malformed = {
      "<｜tool_calls_begin｜><｜tool_call_begin｜>"
      "<steptml:invoke name=\"func\">",
      "<｜tool_call_begin｜><steptml:invoke name=\"func\">"
      "</steptml:invoke><｜tool_call_end｜>",
  };
  for (const std::string& in : malformed) {
    Step3ToolParser p1;
    Step3ToolParser p2;
    ChatCompletionRequest req;
    CHECK_NOTHROW((void)p1.extract_tool_calls(in, req));
    CHECK_NOTHROW((void)RunByteByByte(p2, in, req));
  }
}

// ─── non-streaming is bug-for-bug empty for real step3 output ────────────────

TEST_CASE("step3: non-streaming extracts nothing (upstream xfail)") {
  Step3ToolParser parser;
  auto req = RequestWith(SampleTools());
  // The non-streaming path demands a `function<｜tool_sep｜>` prefix inside each
  // call that the real format omits, so it yields no tool calls and returns the
  // whole output as content (matches the upstream non-streaming xfail).
  auto info = parser.extract_tool_calls(kSingle, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == kSingle);
}

// ─── Factory ─────────────────────────────────────────────────────────────────

TEST_CASE("step3: factory registration") {
  auto p = get_tool_parser("step3");
  REQUIRE(p != nullptr);
  auto req = RequestWith(SampleTools());
  Recon r = RunByteByByte(*p, kSingle, req);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].name == "get_weather");
}
