// Tests for the "inkling" tool-parser registry name (parser_engine_adapter.h):
// the ParserEngineToolAdapter face over the already-ported Inkling ParserEngine.
//
// FIDELITY ANCHOR: tests/parser/engine/test_inkling.py @ 555967922 (vLLM
// 0.26.0.dev0). Upstream has NO tests/tool_parsers/test_inkling_tool_parser.py —
// `InklingEngineToolParser` (inkling_tool_parser.py:7) is a bare
// make_adapters(InklingParser) subclass, so the executable description of the
// dialect is the engine test.
//
// PROVENANCE, case by case: 18 cases here, of which **15 are PORTED** and keep
// their upstream names verbatim, and **3 are AUTHORED** (recorded as
// from-scratch; each says so at its site):
//   - "the registry name resolves and is enumerated" and "EXPLICIT-ONLY — no
//     autodetect row" gate OUR packaging surface. Upstream has no analogue to
//     port: its registry is a lazy dict and it has no chat-template marker
//     table at all, so `DetectToolParser` is an ORIGINAL component (see
//     detect.cpp's header).
//   - "the tool adapter seeds the engine in CONTENT state" gates UPSTREAM
//     behaviour (adapters.py:158/167) that upstream's own suite never
//     exercises, because it drives InklingParser directly and so never
//     constructs the adapter. It was added because a mutation survived without
//     it.
// The upstream TestArgConverter class and the token-id / reasoning cases are
// deliberately NOT here: they belong to the engine layer, which is already
// golden-gated against the pinned oracle in tests/vllm/parser/engine/
// test_parser_engine_assembly.cpp (scenarios inkling_think_tool_text_* and
// inkling_nonobject_args_*). What this file gates is the TOOL-PARSER SEAM:
// that `--tool-call-parser inkling` resolves at all, and that the adapter hands
// the engine the right initial state so the dialect extracts correctly through
// ToolParser.
//
// HARNESS ADAPTATIONS, both forced and both documented:
//   1. Upstream drives InklingParser directly, so its non-streaming cases start
//      in the config's MESSAGE_HEADER state. adapters.py:158 makes the TOOL
//      adapter start in CONTENT (it parses reasoning-stripped content), and we
//      mirror that. Cases whose assertion depends on the header state are
//      prefixed with the "<|message_model|>" token, which is exactly the
//      (CONTENT, MSG_MODEL) -> MESSAGE_HEADER transition (inkling.py:186), so
//      the assertion is preserved rather than weakened.
//   2. Our ToolParser streaming seam is TEXT-only (tool_parsers/abstract.h), so
//      the port uses upstream's _stream_text_only harness (arbitrary character
//      chunks, markers split mid-marker) rather than _stream. Inkling opts into
//      text-lexer terminals upstream (inkling.py:286 token_id_terminals={}), so
//      this is the same grammar path, and character chunks are a STRICTLY
//      STRONGER split than token boundaries.
#include <doctest/doctest.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/detect.h"

using namespace vllm::entrypoints::openai;

namespace {

// inkling.py:48-56 — every marker is a dedicated special token.
const std::string MSG_MODEL = "<|message_model|>";
const std::string TEXT_START = "<|content_text|>";
const std::string TOOL_JSON = "<|content_invoke_tool_json|>";
const std::string TOOL_TEXT = "<|content_invoke_tool_text|>";
const std::string TOOL_ERROR = "<|content_tool_error|>";
const std::string END_MESSAGE = "<|end_message|>";
const std::string END_SAMPLING = "<|content_model_end_sampling|>";

// Upstream _tool_block().
std::string ToolBlock(const std::string& name, const std::string& args) {
  return TOOL_JSON + "{\"name\":\"" + name + "\",\"args\":" + args + "}" +
         END_MESSAGE;
}

std::unique_ptr<ToolParser> MakeParser() {
  std::unique_ptr<ToolParser> p = get_tool_parser("inkling");
  REQUIRE(p != nullptr);
  return p;
}

// Upstream _stream_text_only(): chunk the text at arbitrary character
// boundaries and accumulate previous/current exactly as serving_chat does.
std::vector<DeltaMessage> StreamTextOnly(ToolParser& parser,
                                         const std::string& text,
                                         std::size_t chunk_size) {
  std::vector<DeltaMessage> out;
  std::string previous;
  ChatCompletionRequest req;
  for (std::size_t start = 0; start < text.size(); start += chunk_size) {
    const std::string delta = text.substr(start, chunk_size);
    const std::string current = previous + delta;
    std::optional<DeltaMessage> d =
        parser.extract_tool_calls_streaming(previous, current, delta, req);
    if (d.has_value()) out.push_back(*d);
    previous = current;
  }
  return out;
}

// Upstream streaming_helpers.collect_content.
std::string CollectContent(const std::vector<DeltaMessage>& deltas) {
  std::string s;
  for (const DeltaMessage& d : deltas)
    if (d.content.has_value()) s += *d.content;
  return s;
}

// Upstream streaming_helpers.collect_function_name (the first name emitted).
std::string CollectFunctionName(const std::vector<DeltaMessage>& deltas) {
  for (const DeltaMessage& d : deltas) {
    if (!d.tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *d.tool_calls)
      if (tc.function.name.has_value() && !tc.function.name->empty())
        return *tc.function.name;
  }
  return std::string();
}

// Upstream streaming_helpers.collect_tool_arguments (index 0).
std::string CollectToolArguments(const std::vector<DeltaMessage>& deltas,
                                 int index = 0) {
  std::string s;
  for (const DeltaMessage& d : deltas) {
    if (!d.tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *d.tool_calls)
      if (tc.index == index && tc.function.arguments.has_value())
        s += *tc.function.arguments;
  }
  return s;
}

nlohmann::json J(const std::string& s) { return nlohmann::json::parse(s); }

}  // namespace

// ── The registry surface this wave adds ──────────────────────────────────────

TEST_CASE("inkling: the registry name resolves and is enumerated") {
  // __init__.py:177 registers "inkling" -> InklingEngineToolParser. Before this
  // wave the engine was fully ported (parser/inkling.cpp) but the NAME was not
  // in the tool-parser registry, so --tool-call-parser inkling threw at startup.
  CHECK(get_tool_parser("inkling") != nullptr);
  CHECK(ResolveToolParserName("inkling", "") == "inkling");

  const std::vector<std::string>& names = tool_parser_names();
  bool listed = false;
  for (const std::string& n : names) listed = listed || n == "inkling";
  CHECK(listed);
}

TEST_CASE("inkling: EXPLICIT-ONLY — no autodetect row, on purpose") {
  // Inkling has no jinja chat template at the pin (rendering is
  // vllm/renderers/inkling_encoding.py, not a template), so there is nothing for
  // the marker table to sniff. A template that merely CONTAINS the marker must
  // therefore NOT resolve to inkling; it falls through to the hermes default.
  // See detect.cpp's ORDER MATTERS block.
  CHECK(DetectToolParser(TOOL_JSON) == "hermes");
  CHECK(DetectToolParser("") == "hermes");
}

TEST_CASE("inkling: the tool adapter seeds the engine in CONTENT state") {
  // adapters.py:158/167 — the TOOL adapter (unlike the engine's own entrypoints,
  // which start in the config's initial_state) starts in CONTENT so it parses
  // REASONING-STRIPPED content: what ReasoningParser::extract_reasoning handed
  // back has already lost its leading block markers. Inkling's config
  // initial_state is MESSAGE_HEADER (inkling.py:279), where bare text is
  // metadata and is SWALLOWED (inkling.py:214), so dropping the CONTENT seed
  // silently deletes the assistant's visible text. This case is the only one
  // that separates the adapter from a bare ParserEngine over inkling_config().
  //
  // NOT ported from upstream: upstream drives InklingParser directly in
  // tests/parser/engine/test_inkling.py and so never exercises the adapter's
  // seed. Authored here, and recorded as from-scratch in porting-inventory.md §9.
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info =
      p->extract_tool_calls("Sure, checking that now.", req);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "Sure, checking that now.");

  std::unique_ptr<ToolParser> s = MakeParser();
  const std::vector<DeltaMessage> out =
      StreamTextOnly(*s, "Sure, checking that now." + END_MESSAGE, 5);
  CHECK(CollectContent(out) == "Sure, checking that now.");
}

// ── TestNonStreaming (test_inkling.py) ───────────────────────────────────────

TEST_CASE("inkling test_plain_text") {
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info =
      p->extract_tool_calls(TEXT_START + "hello world" + END_MESSAGE, req);
  CHECK(info.tools_called == false);
  CHECK(info.tool_calls.empty());
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "hello world");
}

TEST_CASE("inkling test_tool_header_name_is_not_visible_content") {
  // Upstream: the optional function name between the model-role and
  // content-kind markers is metadata, not visible assistant content
  // (inkling.py:214). Adaptation 1: MSG_MODEL prefix reproduces upstream's
  // MESSAGE_HEADER initial state on the CONTENT-seeded tool adapter.
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info = p->extract_tool_calls(
      MSG_MODEL + "get_weather" + ToolBlock("get_weather", "{\"city\":\"SF\"}"),
      req);
  CHECK(!info.content.has_value());
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "get_weather");
}

TEST_CASE("inkling test_parallel_tool_calls") {
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info = p->extract_tool_calls(
      ToolBlock("a", "{}") + MSG_MODEL + ToolBlock("b", "{\"x\":[1,2]}"), req);
  CHECK(info.tools_called);
  REQUIRE(info.tool_calls.size() == 2);
  CHECK(info.tool_calls[0].function.name == "a");
  CHECK(info.tool_calls[1].function.name == "b");
  CHECK(J(info.tool_calls[0].function.arguments) == J("{}"));
  CHECK(J(info.tool_calls[1].function.arguments) == J("{\"x\":[1,2]}"));
}

TEST_CASE("inkling test_nested_args") {
  // The hand-rolled span scanner must be string/escape aware: "a}b" carries a
  // closing brace inside a string value.
  const std::string args = "{\"q\":{\"deep\":{\"list\":[{\"k\":\"v\"}]}},\"s\":\"a}b\"}";
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info =
      p->extract_tool_calls(ToolBlock("f", args), req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(J(info.tool_calls[0].function.arguments) == J(args));
}

TEST_CASE("inkling test_invoke_tool_text_is_visible_text") {
  // Raw tool blocks render as visible text, never as a tool call
  // (inkling.py:200).
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info =
      p->extract_tool_calls(TOOL_TEXT + "do something" + END_MESSAGE, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "do something");
}

TEST_CASE("inkling test_tool_error_is_visible_text") {
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info =
      p->extract_tool_calls(TOOL_ERROR + "boom" + END_MESSAGE, req);
  CHECK(info.tools_called == false);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "boom");
}

TEST_CASE("inkling test_end_sampling_closes_blocks") {
  // <|content_model_end_sampling|> ends a block exactly like <|end_message|>
  // (inkling.py:256).
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info =
      p->extract_tool_calls(TEXT_START + "hi" + END_SAMPLING, req);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "hi");
}

TEST_CASE("inkling test_text_after_tool_call") {
  // inkling.py:376 _single_pass_parse — the base engine DEFERS content that
  // follows tool-call events; Inkling allows text blocks after tool blocks, so
  // the trailing text must be flushed. This is the ONLY case that exercises the
  // InklingParser subclass hook through the tool seam.
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info = p->extract_tool_calls(
      ToolBlock("f", "{}") + MSG_MODEL + TEXT_START + "done" + END_MESSAGE, req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "f");
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "done");
}

TEST_CASE("inkling test_incomplete_tool_call_at_eos") {
  // Engine convention: best-effort with what arrived (the Rust parser instead
  // errors with "incomplete Inkling tool call").
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info = p->extract_tool_calls(
      TOOL_JSON + "{\"name\":\"d\",\"args\":{\"k\":\"v\"", req);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].function.name == "d");
}

TEST_CASE("inkling test_prose_marker_without_token_ids_is_structural") {
  // Inkling opts into text-lexer terminal recognition, so a marker that arrives
  // as held-back detokenized TEXT is still parsed structurally.
  std::unique_ptr<ToolParser> p = MakeParser();
  ChatCompletionRequest req;
  ExtractedToolCallInformation info = p->extract_tool_calls(
      TEXT_START + "see " + TEXT_START + " token" + END_MESSAGE, req);
  REQUIRE(info.content.has_value());
  CHECK(*info.content == "see  token");
}

// ── TestStreaming (test_inkling.py) ──────────────────────────────────────────

TEST_CASE("inkling test_chunk_invariance_tool_call_text_only") {
  // Upstream parametrizes chunk_size over [1, 3, 7, 64]; 4096 (one whole
  // delta) is added from the token-id variant's parametrization since character
  // chunking makes it the same shape.
  for (std::size_t chunk : {std::size_t{1}, std::size_t{3}, std::size_t{7},
                            std::size_t{64}, std::size_t{4096}}) {
    CAPTURE(chunk);
    std::unique_ptr<ToolParser> p = MakeParser();
    const std::string text = TEXT_START + "Check this." + END_MESSAGE +
                             MSG_MODEL +
                             ToolBlock("get_weather", "{\"city\":\"San Francisco\"}");
    const std::vector<DeltaMessage> out = StreamTextOnly(*p, text, chunk);
    CHECK(CollectContent(out) == "Check this.");
    CHECK(CollectFunctionName(out) == "get_weather");
    CHECK(J(CollectToolArguments(out)) == J("{\"city\":\"San Francisco\"}"));
  }
}

TEST_CASE("inkling test_split_marker_held_across_chunks") {
  // Mirrors the Rust `inkling_streaming_holds_split_markers`: chunk 9 slices
  // "<|content_text|>" mid-marker, so a lexer that emitted eagerly would leak
  // marker bytes into the content.
  std::unique_ptr<ToolParser> p = MakeParser();
  const std::vector<DeltaMessage> out =
      StreamTextOnly(*p, TEXT_START + "hello" + END_MESSAGE, 9);
  CHECK(CollectContent(out) == "hello");
}

TEST_CASE("inkling test_name_streams_before_args_complete") {
  // Feed only up to the name's closing quote — the name delta must already be
  // emitted before any args arrive.
  std::unique_ptr<ToolParser> p = MakeParser();
  const std::vector<DeltaMessage> out =
      StreamTextOnly(*p, TOOL_JSON + "{\"name\":\"get_weather\",", 4096);
  CHECK(CollectFunctionName(out) == "get_weather");
}

TEST_CASE("inkling test_streamed_args_are_object_only") {
  // inkling.py:146 _inkling_arg_converter — the streamed `arguments` must be
  // the bare args object, NEVER the {"name":...,"args":...} wrapper. Without
  // the converter the engine streams the wrapper verbatim into the OpenAI
  // arguments field.
  std::unique_ptr<ToolParser> p = MakeParser();
  const std::vector<DeltaMessage> out =
      StreamTextOnly(*p, ToolBlock("f", "{\"a\":1}"), 3);
  const std::string args = CollectToolArguments(out);
  CHECK(J(args) == J("{\"a\":1}"));
  CHECK(args.find("name") == std::string::npos);
}

TEST_CASE("inkling test_parallel_calls_streaming") {
  for (std::size_t chunk : {std::size_t{1}, std::size_t{9}}) {
    CAPTURE(chunk);
    std::unique_ptr<ToolParser> p = MakeParser();
    const std::string text = ToolBlock("a", "{\"i\":1}") + MSG_MODEL +
                             ToolBlock("b", "{\"i\":2}");
    const std::vector<DeltaMessage> out = StreamTextOnly(*p, text, chunk);
    std::string name0, name1;
    for (const DeltaMessage& d : out) {
      if (!d.tool_calls.has_value()) continue;
      for (const DeltaToolCall& tc : *d.tool_calls) {
        if (!tc.function.name.has_value() || tc.function.name->empty()) continue;
        if (tc.index == 0) name0 = *tc.function.name;
        if (tc.index == 1) name1 = *tc.function.name;
      }
    }
    CHECK(name0 == "a");
    CHECK(name1 == "b");
    CHECK(J(CollectToolArguments(out, 0)) == J("{\"i\":1}"));
    CHECK(J(CollectToolArguments(out, 1)) == J("{\"i\":2}"));
  }
}
