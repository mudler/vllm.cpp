// Tests for the Kimi K2 tool-call parser (kimi_k2.h), REIMPLEMENTED from the
// wire format of vllm/parser/kimi_k2.py @ e24d1b24 (a token-id ParserEngine).
//
// FIDELITY ANCHOR: tests/tool_parsers/test_kimi_k2_tool_parser.py (32 cases).
// EVERY upstream case is ported here (the case ids are preserved in the TEST_CASE
// names). Upstream's streaming harness feeds the five markers as separate string
// chunks; the tokenizer-driven TestStreamingIntervals cases (which need the real
// moonshotai/Kimi-K2 tokenizer) are reproduced here as BYTE-chunk streaming at
// the same interval sizes - a STRICTLY STRONGER split: the markers are ASCII
// multi-char, so byte chunks slice them MID-MARKER (the token-id -> text rework
// must hold partial markers back), which token boundaries never do. Extra
// mid-marker split cases are added on top, per the port brief.
#include <doctest/doctest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/kimi_k2.h"

using namespace vllm::entrypoints::openai;

namespace {

// Marker byte strings (copied from the parser constants — byte-exact).
const std::string SECTION_BEGIN = KimiK2ToolParser::kToolCallsSectionBeginToken;
const std::string SECTION_END = KimiK2ToolParser::kToolCallsSectionEndToken;
const std::string TOOL_BEGIN = KimiK2ToolParser::kToolCallBeginToken;
const std::string TOOL_END = KimiK2ToolParser::kToolCallEndToken;
const std::string ARG_BEGIN = KimiK2ToolParser::kToolCallArgumentBeginToken;

ChatCompletionRequest empty_request() { return ChatCompletionRequest{}; }

// Upstream _tool(): <|tool_call_begin|>{id} <|tool_call_argument_begin|>{args}
// <|tool_call_end|>  (note the space after the id).
std::string Tool(const std::string& tool_id, const std::string& args) {
  return TOOL_BEGIN + tool_id + " " + ARG_BEGIN + args + TOOL_END;
}
// Upstream _wrap(): SECTION_BEGIN + calls + SECTION_END.
std::string Wrap(const std::vector<std::string>& tool_strs) {
  std::string s = SECTION_BEGIN;
  for (const std::string& t : tool_strs) s += t;
  return s + SECTION_END;
}
std::string Wrap(const std::string& t) { return Wrap(std::vector<std::string>{t}); }

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

// Upstream _split_tool_output_to_deltas: special tokens as separate chunks, a
// trailing space after the id and after the args (matches the streaming harness).
std::vector<std::string> SplitToDeltas(
    const std::string& content,
    const std::vector<std::pair<std::string, std::string>>& tools) {
  std::vector<std::string> deltas = {content, SECTION_BEGIN};
  for (const auto& [tool_id, args_json] : tools) {
    deltas.push_back(TOOL_BEGIN);
    deltas.push_back(tool_id + " ");
    deltas.push_back(ARG_BEGIN);
    deltas.push_back(args_json + " ");
    deltas.push_back(TOOL_END);
  }
  deltas.push_back(SECTION_END);
  return deltas;
}

// Split `s` into fixed-size BYTE chunks (slices multi-char markers mid-marker).
std::vector<std::string> ByteChunks(const std::string& s, std::size_t n) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < s.size(); i += n) out.push_back(s.substr(i, n));
  return out;
}

// Reconstruct the streamed result: content, and per-index {name,id,args}.
struct ToolRec {
  std::string name;
  std::string id;
  std::string args;
};
struct Reconstructed {
  std::string content;
  std::map<int, ToolRec> tools;  // ordered by index
  std::vector<std::string> name_order;  // names in emission order
  std::vector<int> name_indices;        // indices in emission order

  std::vector<ToolRec> ordered() const {
    std::vector<ToolRec> v;
    for (const auto& [idx, rec] : tools) v.push_back(rec);
    return v;
  }
};
Reconstructed Reconstruct(const std::vector<DeltaMessage>& msgs) {
  Reconstructed r;
  for (const DeltaMessage& m : msgs) {
    if (m.content.has_value()) r.content += *m.content;
    if (!m.tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *m.tool_calls) {
      ToolRec& rec = r.tools[tc.index];
      if (tc.function.name.has_value()) {
        rec.name = *tc.function.name;
        if (tc.id.has_value()) rec.id = *tc.id;
        r.name_order.push_back(*tc.function.name);
        r.name_indices.push_back(tc.index);
      }
      if (tc.function.arguments.has_value()) rec.args += *tc.function.arguments;
    }
  }
  return r;
}

// True when `needle` does not appear anywhere in `hay` (marker-leak assertions).
bool Absent(const std::string& hay, const std::string& needle) {
  return hay.find(needle) == std::string::npos;
}
void CheckNoMarkersLeaked(const std::string& content) {
  for (const std::string& marker :
       {SECTION_BEGIN, SECTION_END, TOOL_BEGIN, TOOL_END, ARG_BEGIN}) {
    CHECK(Absent(content, marker));
  }
}

}  // namespace

// ─── Non-streaming: TestExtractToolCalls ─────────────────────────────────────

TEST_CASE("Kimi non-streaming: no_tools") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  auto info = parser.extract_tool_calls("This is a test", req);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "This is a test");
  CHECK(info.tool_calls.empty());
  CHECK(info.tools_called == false);
}

TEST_CASE("Kimi non-streaming: single_tool_call") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out =
      "I'll check. " + Wrap(Tool("functions.get_weather:0", R"({"city": "Beijing"})"));
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments) ==
        nlohmann::json::parse(R"({"city": "Beijing"})"));
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "I'll check. ");
  // id format "something:digit".
  const std::string id = info.tool_calls[0].id;
  CHECK(id.substr(id.rfind(':') + 1) == "0");
}

TEST_CASE("Kimi non-streaming: parallel_tool_calls") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out =
      "Compare weather. " +
      Wrap({Tool("functions.get_weather:0", R"({"city": "Beijing"})"),
            Tool("functions.get_weather:1", R"({"city": "Shanghai"})")});
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[1].function.name == "get_weather");
  CHECK(nlohmann::json::parse(info.tool_calls[1].function.arguments)["city"] ==
        "Shanghai");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Compare weather. ");
}

TEST_CASE("Kimi non-streaming: three_tool_calls") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out =
      "Multiple tasks. " +
      Wrap({Tool("functions.get_weather:0", R"({"city": "New York"})"),
            Tool("functions.get_news:1", R"({"topic": "technology"})"),
            Tool("functions.send_email:2",
                 R"({"to": "user@example.com", "subject": "Daily Update"})")});
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 3);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[1].function.name == "get_news");
  CHECK(info.tool_calls[2].function.name == "send_email");
  CHECK(nlohmann::json::parse(info.tool_calls[2].function.arguments)["subject"] ==
        "Daily Update");
}

TEST_CASE("Kimi non-streaming: angle_brackets_in_json") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out =
      "Process HTML. " +
      Wrap(Tool("functions.process_html:0", R"({"html": "<div>content</div>"})"));
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "process_html");
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments)["html"] ==
        "<div>content</div>");
}

TEST_CASE("Kimi non-streaming: multiline_json") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out =
      "Formatted. " +
      Wrap(Tool("functions.process_data:0",
                "{\n  \"name\": \"test\",\n  \"value\": 123\n}"));
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "process_data");
  const auto parsed = nlohmann::json::parse(info.tool_calls[0].function.arguments);
  CHECK(parsed["name"] == "test");
  CHECK(parsed["value"] == 123);
}

TEST_CASE("Kimi non-streaming: no_functions_prefix") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out =
      "No prefix. " + Wrap(Tool("get_weather:0", R"({"city": "Tokyo"})"));
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].id == "get_weather:0");
}

TEST_CASE("Kimi non-streaming: empty_arguments") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out = "Empty args. " + Wrap(Tool("functions.test:0", "{}"));
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "test");
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments) ==
        nlohmann::json::object());
}

TEST_CASE("Kimi non-streaming: invalid_json_still_extracted") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out = "Help. " + SECTION_BEGIN +
                          Tool("functions.bad:0", R"({"city": "Beijing")") +
                          Tool("functions.good:1", R"({"city": "Shanghai"})") +
                          SECTION_END;
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "bad");
  CHECK(info.tool_calls[1].function.name == "good");
  // The malformed args are returned as-is (not re-serialized, not dropped).
  CHECK(info.tool_calls[0].function.arguments == R"({"city": "Beijing")");
}

TEST_CASE("Kimi non-streaming: invalid_funcall_id_skipped") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out = "Help. " + SECTION_BEGIN +
                          Tool("functions.invalid.0", R"({"city": "Beijing"})") +
                          Tool("functions.valid:1", R"({"city": "Shanghai"})") +
                          SECTION_END;
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "valid");
}

TEST_CASE("Kimi non-streaming: native_id_extracted (PR #32768)") {
  KimiK2ToolParser parser;
  auto req = empty_request();
  const std::string out =
      "Checking weather. " + Wrap(Tool("functions.get_weather:0", R"({"city": "Tokyo"})"));
  auto info = parser.extract_tool_calls(out, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].id == "functions.get_weather:0");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(nlohmann::json::parse(info.tool_calls[0].function.arguments)["city"] ==
        "Tokyo");
}

TEST_CASE("Kimi non-streaming: multi_turn_native_id_continuity (PR #32768)") {
  auto req = empty_request();
  KimiK2ToolParser turn1;
  auto t1 = turn1.extract_tool_calls(
      "Let me check. " + Wrap(Tool("functions.get_weather:0", R"({"city": "Beijing"})")),
      req);
  REQUIRE(t1.tool_calls.size() == 1);
  CHECK(t1.tool_calls[0].id == "functions.get_weather:0");

  KimiK2ToolParser turn2;
  auto t2 = turn2.extract_tool_calls(
      "Now let me get news. " +
          Wrap(Tool("functions.get_news:0", R"({"topic": "weather in Beijing"})")),
      req);
  REQUIRE(t2.tool_calls.size() == 1);
  CHECK(t2.tool_calls[0].id == "functions.get_news:0");
}

// ─── Streaming: TestStreamingHappyPath ───────────────────────────────────────

TEST_CASE("Kimi streaming: single_tool_call") {
  KimiK2ToolParser parser;
  const auto deltas = SplitToDeltas(
      "I'll help. ", {{"functions.get_weather:0", R"({"city": "Beijing"})"}});
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools.at(0).name == "get_weather");
  CHECK(r.tools.at(0).id == "functions.get_weather:0");
  CHECK(nlohmann::json::parse(r.tools.at(0).args) ==
        nlohmann::json::parse(R"({"city": "Beijing"})"));
}

TEST_CASE("Kimi streaming: multiple_tool_calls") {
  KimiK2ToolParser parser;
  const auto deltas = SplitToDeltas(
      "Compare weather. ", {{"functions.get_weather:0", R"({"city": "Tokyo"})"},
                            {"functions.get_weather:1", R"({"city": "NYC"})"}});
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  REQUIRE(r.tools.size() == 2);
  CHECK(r.tools.at(0).name == "get_weather");
  CHECK(r.tools.at(0).id == "functions.get_weather:0");
  CHECK(nlohmann::json::parse(r.tools.at(0).args)["city"] == "Tokyo");
  CHECK(r.tools.at(1).name == "get_weather");
  CHECK(r.tools.at(1).id == "functions.get_weather:1");
  CHECK(nlohmann::json::parse(r.tools.at(1).args)["city"] == "NYC");
}

TEST_CASE("Kimi streaming: content_before_tools") {
  KimiK2ToolParser parser;
  const auto deltas = SplitToDeltas(
      "I'll check the weather. ", {{"functions.get_weather:0", R"({"city": "Tokyo"})"}});
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  CHECK(r.content.find("check the weather") != std::string::npos);
  CheckNoMarkersLeaked(r.content);
  CHECK(Absent(r.content, "get_weather"));
  CHECK(Absent(r.content, "Tokyo"));
}

TEST_CASE("Kimi streaming: no_tool_calls") {
  KimiK2ToolParser parser;
  const std::vector<std::string> deltas = {"This is just ", "regular text ",
                                           "without tools."};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  CHECK(r.content == "This is just regular text without tools.");
  CHECK(r.tools.empty());
}

TEST_CASE("Kimi streaming: incremental_arguments") {
  KimiK2ToolParser parser;
  const std::vector<std::string> deltas = {
      "Help. ",      SECTION_BEGIN, TOOL_BEGIN, "functions.get_weather:0 ",
      ARG_BEGIN,     R"({"ci)",      R"(ty": "Be)", R"(ijing"})",
      " ",           TOOL_END,      SECTION_END};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools.at(0).name == "get_weather");
  CHECK(nlohmann::json::parse(r.tools.at(0).args) ==
        nlohmann::json::parse(R"({"city": "Beijing"})"));
}

TEST_CASE("Kimi streaming: streaming_matches_nonstreaming (single_tool)") {
  const std::string out =
      "Single. " + Wrap(Tool("functions.get_weather:0", R"({"city": "Beijing"})"));
  KimiK2ToolParser ns;
  auto req = empty_request();
  auto info = ns.extract_tool_calls(out, req);
  KimiK2ToolParser st;
  const Reconstructed r = Reconstruct(DriveStream(st, ByteChunks(out, 5)));
  REQUIRE(info.tool_calls.size() == r.tools.size());
  CHECK(r.tools.at(0).name == info.tool_calls[0].function.name);
  CHECK(nlohmann::json::parse(r.tools.at(0).args) ==
        nlohmann::json::parse(info.tool_calls[0].function.arguments));
}

TEST_CASE("Kimi streaming: streaming_matches_nonstreaming (parallel_tools)") {
  const std::string out =
      "Multi. " + Wrap({Tool("functions.get_weather:0", R"({"city": "Tokyo"})"),
                        Tool("functions.get_news:1", R"({"topic": "tech"})")});
  KimiK2ToolParser ns;
  auto req = empty_request();
  auto info = ns.extract_tool_calls(out, req);
  KimiK2ToolParser st;
  const Reconstructed r = Reconstruct(DriveStream(st, ByteChunks(out, 5)));
  const auto tools = r.ordered();
  REQUIRE(info.tool_calls.size() == tools.size());
  for (std::size_t i = 0; i < tools.size(); ++i) {
    CHECK(tools[i].name == info.tool_calls[i].function.name);
    CHECK(nlohmann::json::parse(tools[i].args) ==
          nlohmann::json::parse(info.tool_calls[i].function.arguments));
  }
}

TEST_CASE("Kimi streaming: streaming_matches_nonstreaming (no_functions_prefix)") {
  const std::string out =
      "No prefix id. " + Wrap(Tool("get_weather:0", R"({"city": "NYC"})"));
  KimiK2ToolParser ns;
  auto req = empty_request();
  auto info = ns.extract_tool_calls(out, req);
  KimiK2ToolParser st;
  const Reconstructed r = Reconstruct(DriveStream(st, ByteChunks(out, 5)));
  REQUIRE(info.tool_calls.size() == r.tools.size());
  CHECK(r.tools.at(0).name == info.tool_calls[0].function.name);
  CHECK(nlohmann::json::parse(r.tools.at(0).args) ==
        nlohmann::json::parse(info.tool_calls[0].function.arguments));
}

// ─── Streaming: TestStreamingEdgeCases ───────────────────────────────────────

TEST_CASE("Kimi streaming edge: marker_suppression") {
  KimiK2ToolParser parser;
  const auto deltas = SplitToDeltas(
      "I'll check. ", {{"functions.get_weather:0", R"({"city": "Tokyo"})"}});
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  CheckNoMarkersLeaked(r.content);
}

TEST_CASE("Kimi streaming edge: noise_between_markers_suppressed") {
  KimiK2ToolParser parser;
  const std::vector<std::string> deltas = {
      "Reasoning. ", SECTION_BEGIN, " spurious noise ", TOOL_BEGIN,
      "functions.test:0 ", ARG_BEGIN, R"({"k": "v"} )", TOOL_END, SECTION_END};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  CHECK(Absent(r.content, "spurious"));
  CHECK(Absent(r.content, "noise"));
}

TEST_CASE("Kimi streaming edge: empty_tool_section") {
  KimiK2ToolParser parser;
  const std::vector<std::string> deltas = {"Reasoning. ", SECTION_BEGIN,
                                           SECTION_END};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  CHECK(r.tools.empty());
}

TEST_CASE("Kimi streaming edge: three_different_tools") {
  KimiK2ToolParser parser;
  const auto deltas = SplitToDeltas(
      "Multiple tasks. ", {{"functions.get_weather:0", R"({"city": "NYC"})"},
                           {"functions.get_news:1", R"({"topic": "tech"})"},
                           {"functions.send_email:2", R"({"to": "a@b.com"})"}});
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  REQUIRE(r.tools.size() == 3);
  const auto tools = r.ordered();
  CHECK(tools[0].name == "get_weather");
  CHECK(tools[1].name == "get_news");
  CHECK(tools[2].name == "send_email");
  // Unique ids.
  CHECK(tools[0].id != tools[1].id);
  CHECK(tools[1].id != tools[2].id);
  CHECK(tools[0].id != tools[2].id);
}

TEST_CASE("Kimi streaming edge: truncated_tool_call_no_end_marker") {
  KimiK2ToolParser parser;
  const std::vector<std::string> deltas = {
      "I'll check. ", SECTION_BEGIN, TOOL_BEGIN, "functions.get_weather:0 ",
      ARG_BEGIN, R"({"city": "Bei)"};  // stream ends here - no TOOL_END/SECTION_END
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools.at(0).name == "get_weather");
  CHECK(r.tools.at(0).id == "functions.get_weather:0");
  CHECK(r.tools.at(0).args == R"({"city": "Bei)");
  CheckNoMarkersLeaked(r.content);
}

TEST_CASE("Kimi streaming edge: content_after_tool_section") {
  KimiK2ToolParser parser;
  const std::vector<std::string> deltas = {
      "Before. ", SECTION_BEGIN, TOOL_BEGIN, "functions.get_weather:0 ",
      ARG_BEGIN, R"({"city": "Tokyo"} )", TOOL_END, SECTION_END, " After tools."};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools.at(0).name == "get_weather");
  CHECK(nlohmann::json::parse(r.tools.at(0).args)["city"] == "Tokyo");
  CHECK(Absent(r.content, "After tools."));
  CheckNoMarkersLeaked(r.content);
}

// ─── TestAdjustRequest ───────────────────────────────────────────────────────

TEST_CASE("Kimi adjust_request: sets_skip_special_tokens_false") {
  KimiK2ToolParser parser;
  ChatCompletionRequest req;
  ChatCompletionToolsParam tool;
  tool.function.name = "test";
  req.tools = std::vector<ChatCompletionToolsParam>{tool};
  req.tool_choice = ToolChoice{"auto", std::nullopt};
  req.skip_special_tokens = true;
  parser.adjust_request(req);
  CHECK(req.skip_special_tokens == false);
}

TEST_CASE("Kimi adjust_request: no_change_when_tool_choice_none") {
  KimiK2ToolParser parser;
  ChatCompletionRequest req;
  ChatCompletionToolsParam tool;
  tool.function.name = "test";
  req.tools = std::vector<ChatCompletionToolsParam>{tool};
  req.tool_choice = ToolChoice{"none", std::nullopt};
  req.skip_special_tokens = true;
  parser.adjust_request(req);
  CHECK(req.skip_special_tokens == true);
}

TEST_CASE("Kimi adjust_request: no_change_when_no_tools") {
  KimiK2ToolParser parser;
  ChatCompletionRequest req;
  req.tools = std::nullopt;
  req.tool_choice = ToolChoice{"auto", std::nullopt};
  req.skip_special_tokens = true;
  parser.adjust_request(req);
  CHECK(req.skip_special_tokens == true);
}

// ─── TestStreamingIntervals (reproduced as BYTE-chunk intervals) ─────────────

TEST_CASE("Kimi streaming intervals: single_tool_call_at_interval") {
  const std::string text =
      "Help. " + Wrap(Tool("functions.get_weather:0", R"({"city": "Beijing"})"));
  for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                              std::size_t{5}, std::size_t{8}}) {
    CAPTURE(n);
    KimiK2ToolParser parser;
    const Reconstructed r = Reconstruct(DriveStream(parser, ByteChunks(text, n)));
    REQUIRE(r.tools.size() == 1);
    CHECK(r.tools.at(0).name == "get_weather");
    CHECK(nlohmann::json::parse(r.tools.at(0).args) ==
          nlohmann::json::parse(R"({"city": "Beijing"})"));
    CheckNoMarkersLeaked(r.content);
  }
}

TEST_CASE("Kimi streaming intervals: content_then_tool_call_at_interval") {
  const std::string text =
      "Sure, let me check. " + Wrap(Tool("functions.get_weather:0", R"({"city": "Tokyo"})"));
  for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                              std::size_t{5}, std::size_t{8}}) {
    CAPTURE(n);
    KimiK2ToolParser parser;
    const Reconstructed r = Reconstruct(DriveStream(parser, ByteChunks(text, n)));
    CHECK(r.content.find("let me check") != std::string::npos);
    CHECK(Absent(r.content, "get_weather"));
    REQUIRE(r.tools.size() == 1);
    CHECK(r.tools.at(0).name == "get_weather");
    CHECK(nlohmann::json::parse(r.tools.at(0).args)["city"] == "Tokyo");
  }
}

TEST_CASE("Kimi streaming intervals: multiple_tool_calls_at_interval") {
  const std::string text =
      "Compare. " + Wrap({Tool("functions.search:0", R"({"q": "cats"})"),
                          Tool("functions.search:1", R"({"q": "dogs"})")});
  for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                              std::size_t{5}, std::size_t{8}}) {
    CAPTURE(n);
    KimiK2ToolParser parser;
    const Reconstructed r = Reconstruct(DriveStream(parser, ByteChunks(text, n)));
    REQUIRE(r.tools.size() == 2);
    const auto tools = r.ordered();
    CHECK(tools[0].name == "search");
    CHECK(nlohmann::json::parse(tools[0].args)["q"] == "cats");
    CHECK(tools[1].name == "search");
    CHECK(nlohmann::json::parse(tools[1].args)["q"] == "dogs");
  }
}

TEST_CASE("Kimi streaming intervals: plain_text_at_interval") {
  const std::string text = "This is plain text with no tool calling involved.";
  for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                              std::size_t{5}, std::size_t{8}}) {
    CAPTURE(n);
    KimiK2ToolParser parser;
    const Reconstructed r = Reconstruct(DriveStream(parser, ByteChunks(text, n)));
    CHECK(r.content == text);
    CHECK(r.tools.empty());
  }
}

TEST_CASE("Kimi streaming intervals: content_and_tool_call_in_single_chunk") {
  const std::string text =
      "Hi! " + Wrap(Tool("functions.get_weather:0", R"({"city": "Beijing"})"));
  KimiK2ToolParser parser;
  const Reconstructed r = Reconstruct(DriveStream(parser, {text}));
  CHECK(r.content.find("Hi!") != std::string::npos);
  CHECK(Absent(r.content, "get_weather"));
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools.at(0).name == "get_weather");
  CHECK(nlohmann::json::parse(r.tools.at(0).args) ==
        nlohmann::json::parse(R"({"city": "Beijing"})"));
}

// ─── Extra: mid-marker split safety (markers are ASCII multi-char) ───────────

TEST_CASE("Kimi streaming: every marker split mid-marker (1-byte chunks)") {
  KimiK2ToolParser parser;
  const std::string full =
      "Prefix. " + Wrap(Tool("functions.get_weather:0", R"({"city": "Tokyo"})"));
  const Reconstructed r = Reconstruct(DriveStream(parser, ByteChunks(full, 1)));
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools.at(0).name == "get_weather");
  CHECK(r.tools.at(0).id == "functions.get_weather:0");
  CHECK(nlohmann::json::parse(r.tools.at(0).args)["city"] == "Tokyo");
  CHECK(r.content == "Prefix. ");
  CheckNoMarkersLeaked(r.content);
}

TEST_CASE("Kimi streaming: closing markers split across the boundary") {
  KimiK2ToolParser parser;
  // Deltas deliberately cut <|tool_call_end|> and <|tool_calls_section_end|> in
  // half so the argument tail must hold back the partial marker.
  const std::vector<std::string> deltas = {
      SECTION_BEGIN, TOOL_BEGIN, "functions.f:0 ", ARG_BEGIN,
      R"({"x": 1})", "<|tool_call", "_end|>", "<|tool_calls_secti", "on_end|>"};
  const Reconstructed r = Reconstruct(DriveStream(parser, deltas));
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools.at(0).name == "f");
  CHECK(nlohmann::json::parse(r.tools.at(0).args)["x"] == 1);
  CheckNoMarkersLeaked(r.content);
  CHECK(Absent(r.tools.at(0).args, "tool_call"));
}

// ─── Factory ─────────────────────────────────────────────────────────────────

TEST_CASE("Kimi factory: get_tool_parser(\"kimi_k2\")") {
  auto p = get_tool_parser("kimi_k2");
  REQUIRE(p != nullptr);
  auto req = empty_request();
  const std::string out = Wrap(Tool("functions.f:0", "{}"));
  auto info = p->extract_tool_calls(out, req);
  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "f");
  CHECK(info.tool_calls[0].id == "functions.f:0");
}
