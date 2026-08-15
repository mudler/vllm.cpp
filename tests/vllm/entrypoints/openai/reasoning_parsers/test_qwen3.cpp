// Ports tests/reasoning/test_qwen3_reasoning_parser.py @ 555967922
// (vLLM 0.26.0.dev0), the gate for `Qwen3ParserReasoningAdapter` — the
// reasoning FACE over the parser engine, registered upstream under BOTH
// "qwen3" (vllm/reasoning/__init__.py:115) and "mimo" (:87).
//
// Every input string, every expected (reasoning, content) pair and every
// streaming cadence below is the upstream fixture verbatim. Harness
// adaptations, all forced by the TEXT-ONLY reasoning seam (see
// reasoning_parsers/abstract.h) and by reasoning_test_utils.h:
//   - upstream parametrizes over three REASONING_MODEL_NAMES tokenizers only to
//     produce the per-token delta split; the parse is a pure text function of
//     that split, so we feed the same strings as text deltas (think markers as
//     their own delta, the cadence the upstream per-token split produces).
//   - `qwen3_tokenizer` is dropped with the tokenizer (no token-ID methods).
//   - test_reasoning_thinking_disabled constructs the parser with
//     chat_template_kwargs={"enable_thinking": False}; the name-only factory
//     cannot carry request kwargs (same deviation as deepseek_v3, threading is
//     W4 in specs/reasoning-parsers.md), so the thinking flag is taken through
//     the public constructor — the same seam test_deepseek_v3.cpp uses.
//
// BEYOND UPSTREAM (written from scratch, no upstream case is dropped — there is
// none to drop). Upstream's reasoning suite has NO qwen3 `is_reasoning_end`
// test at all (only test_base_thinking_reasoning_parser.py:111 for the base
// family), and its thinking-disabled cases run with tool parsing suppressed.
// The last three TEST_CASEs are therefore ours, and cover branches upstream's
// fixtures cannot reach:
//   - "engine thinking-off passthrough survives an unskipped tool" — the
//     qwen3.py:247 override on the ENGINE, not the adapter.
//   - "is_reasoning_end (text form, ...)" — the qwen3.py:256 override,
//     including the reasoning-REOPEN branch (`<tool_call>` then `<think>`).
//   - "seed_oss inherits the qwen3 thinking-off passthrough" — the shared
//     class, which only the thinking-off arm can observe.
// Recorded in .agents/specs/reasoning-parsers.md § Tests to port.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "reasoning_test_utils.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
#include "vllm/entrypoints/openai/reasoning_parsers/parser_engine_adapter.h"
#include "vllm/parser/engine/configs.h"
#include "vllm/parser/parser_manager.h"
#include "vllm/parser/qwen3.h"

using namespace vllm::entrypoints::openai;
using vllm::entrypoints::openai::reasoning_test::Extracted;
using vllm::entrypoints::openai::reasoning_test::RunExtraction;

namespace {
using Opt = std::optional<std::string>;

// test_qwen3_reasoning_parser.py:83 TOOL_CALL_BODY.
const std::string kToolCallBody =
    "<tool_call>\n<function=bash>\n<parameter=command>"
    "\ncat /etc/hosts\n</parameter>\n</function>\n</tool_call>";

// One upstream pytest.param: the delta cadence + the expected split.
struct Case {
  const char* id;
  std::vector<std::string> deltas;
  Opt reasoning;
  Opt content;
};

// The 10 distinct upstream fixtures. Upstream runs each one twice (streaming
// False/True) over the same expectations — TEST_CASES entries pair up exactly,
// including WITHOUT_START_TOKEN/_STREAM, WITH_THINK/_STREAM,
// WITHOUT_THINK/_STREAM, ONLY_OPEN_TAG/_STREAM and
// TRUNCATED_NO_START_TOKEN/_STREAM, whose _STREAM twins are byte-identical
// dicts.
std::vector<Case> UpstreamCases() {
  return {
      // WITHOUT_START_TOKEN (:33) — <think> in the prompt, only </think> out.
      {"without_start_token",
       {"This is a reasoning section", "</think>", "This is the rest"},
       Opt("This is a reasoning section"),
       Opt("This is the rest")},
      // WITHOUT_START_TOKEN_COMPLETE_REASONING (:43).
      {"without_start_token_complete_reasoning",
       {"This is a reasoning section", "</think>"},
       Opt("This is a reasoning section"),
       std::nullopt},
      // WITH_THINK (:51) — <think> present in the output.
      {"with_think",
       {"<think>", "This is a reasoning section", "</think>",
        "This is the rest"},
       Opt("This is a reasoning section"),
       Opt("This is the rest")},
      // WITHOUT_THINK (:66) — thinking enabled, truncated before </think>:
      // ALL output is reasoning. (This is where think_auto disagrees.)
      {"without_think",
       {"This is the rest"},
       Opt("This is the rest"),
       std::nullopt},
      // COMPLETE_REASONING (:102).
      {"complete_reasoning",
       {"<think>", "This is a reasoning section", "</think>"},
       Opt("This is a reasoning section"),
       std::nullopt},
      // MULTILINE_REASONING (:107).
      {"multiline_reasoning",
       {"<think>", "This is a reasoning\nsection", "</think>",
        "This is the rest\nThat"},
       Opt("This is a reasoning\nsection"),
       Opt("This is the rest\nThat")},
      // ONLY_OPEN_TAG (:114) — <think> with no </think>: all reasoning.
      {"only_open_tag",
       {"<think>", "This is a reasoning section"},
       Opt("This is a reasoning section"),
       std::nullopt},
      // TRUNCATED_NO_START_TOKEN (:128).
      {"truncated_no_start_token",
       {"This is a reasoning section"},
       Opt("This is a reasoning section"),
       std::nullopt},
      // TOOL_CALL_NO_THINK_END (:88) — <tool_call> is an IMPLICIT reasoning
      // end; the tool body stays verbatim CONTENT for the reasoning face.
      {"tool_call_no_think_end",
       {"I need to read the file.\n\n", kToolCallBody},
       Opt("I need to read the file.\n\n"),
       Opt(kToolCallBody)},
      // TOOL_CALL_WITH_THINK_NO_END (:94).
      {"tool_call_with_think_no_end",
       {"<think>", "I need to read the file.\n\n", kToolCallBody},
       Opt("I need to read the file.\n\n"),
       Opt(kToolCallBody)},
  };
}

// MULTI_TOKEN_DELTA_CASES (:268) — streaming only, multi-token deltas.
std::vector<Case> MultiTokenDeltaCases() {
  return {
      {"start_token_grouped_with_text",
       {"<think>This is a reasoning section", "</think>", "This is the rest"},
       Opt("This is a reasoning section"),
       Opt("This is the rest")},
      {"end_token_grouped_with_content",
       {"reasoning section", "</think>This is the rest"},
       Opt("reasoning section"),
       Opt("This is the rest")},
      {"start_and_end_in_one_delta_no_content",
       {"<think>reasoning</think>"},
       Opt("reasoning"),
       std::nullopt},
      {"no_start_end_grouped_with_content",
       {"reasoning section", "</think>content"},
       Opt("reasoning section"),
       Opt("content")},
      {"tool_call_implicit_reasoning_end",
       {"I need to read the file.\n\n", "<tool_call>\n<function=bash>"},
       Opt("I need to read the file.\n\n"),
       Opt("<tool_call>\n<function=bash>")},
  };
}
}  // namespace

// __init__.py:115 ("qwen3") and :87 ("mimo") register the SAME class
// (Qwen3ParserReasoningAdapter, registered_adapters.py:48), so both names must
// resolve for one port.
TEST_CASE("qwen3: registered names resolve (qwen3 + mimo alias)") {
  CHECK(get_reasoning_parser("qwen3") != nullptr);
  CHECK(get_reasoning_parser("mimo") != nullptr);
}

// Ports test_reasoning (:244) for BOTH the streaming=False and streaming=True
// parametrizations, over every TEST_CASES fixture.
TEST_CASE("qwen3: upstream test_reasoning fixtures") {
  for (const Case& c : UpstreamCases()) {
    for (bool streaming : {false, true}) {
      CAPTURE(std::string(c.id));
      CAPTURE(streaming);
      auto p = get_reasoning_parser("qwen3");
      REQUIRE(p != nullptr);
      const Extracted got = RunExtraction(*p, c.deltas, streaming);
      CHECK(got.reasoning == c.reasoning);
      CHECK(got.content == c.content);
    }
  }
}

// The "mimo" alias is the same class, so it must produce the same split.
TEST_CASE("qwen3: mimo alias splits identically") {
  for (const Case& c : UpstreamCases()) {
    for (bool streaming : {false, true}) {
      CAPTURE(std::string(c.id));
      CAPTURE(streaming);
      auto p = get_reasoning_parser("mimo");
      REQUIRE(p != nullptr);
      const Extracted got = RunExtraction(*p, c.deltas, streaming);
      CHECK(got.reasoning == c.reasoning);
      CHECK(got.content == c.content);
    }
  }
}

// Ports test_reasoning_streaming_multi_token_deltas (:310) — a single delta may
// carry a marker PLUS surrounding text; the marker must never leak.
TEST_CASE("qwen3: multi-token deltas never leak the markers") {
  for (const Case& c : MultiTokenDeltaCases()) {
    CAPTURE(std::string(c.id));
    auto p = get_reasoning_parser("qwen3");
    REQUIRE(p != nullptr);
    const Extracted got = RunExtraction(*p, c.deltas, /*streaming=*/true);
    CHECK(got.reasoning == c.reasoning);
    CHECK(got.content == c.content);
  }
}

// Ports test_reasoning_thinking_disabled (:357) — THINKING_DISABLED_CASES
// (:332). With enable_thinking=False the WHOLE output is content, including a
// tool-call body that the thinking-on path would have split off as content
// after an implicit reasoning end.
TEST_CASE("qwen3: enable_thinking=False is whole-output passthrough") {
  const std::vector<std::pair<const char*, std::string>> cases = {
      {"thinking_disabled_plain_content", "This is plain content"},
      {"thinking_disabled_no_think_tokens", "Some output without think tokens"},
      {"thinking_disabled_with_tool_call",
       "I need to read the file.\n\n" + kToolCallBody},
  };
  for (const auto& [id, output] : cases) {
    CAPTURE(std::string(id));
    Qwen3ParserReasoningAdapter p(/*thinking=*/false);
    const ExtractedReasoning r =
        p.extract_reasoning(output, ChatCompletionRequest{});
    CHECK(r.reasoning == std::nullopt);
    CHECK(r.content == Opt(output));
  }
}

// qwen3.py:247 lives on the ENGINE (Qwen3Parser), not on the adapter, and the
// adapter always suppresses tool parsing — under which the three
// THINKING_DISABLED_CASES above pass with or without the override, because a
// suppressed <tool_call> already degrades to plain content. Upstream's own test
// has the same blind spot. Exercise the engine DIRECTLY, tool parsing NOT
// suppressed, where the override is the only thing keeping the tool body in the
// content span.
TEST_CASE("qwen3: engine thinking-off passthrough survives an unskipped tool") {
  namespace pe = vllm::parser::engine;
  const std::string output = "I need to read the file.\n\n" + kToolCallBody;

  vllm::parser::Qwen3Parser off(pe::qwen3_config(false, "qwen3"),
                                /*thinking=*/false);
  const auto [reasoning, content] =
      off.extract_reasoning(output, pe::ParserRequest{});
  CHECK(reasoning == std::nullopt);
  CHECK(content == Opt(output));

  // Same engine, thinking ON: the state machine now consumes the tool body, so
  // the content span is NOT the whole output. That difference is exactly what
  // the override exists to suppress.
  vllm::parser::Qwen3Parser on(pe::qwen3_config(true, "qwen3"),
                               /*thinking=*/true);
  const auto [r2, c2] = on.extract_reasoning(output, pe::ParserRequest{});
  CHECK(r2 == Opt("I need to read the file.\n\n"));
  CHECK(c2 != Opt(output));
}

// is_reasoning_end — the TEXT form of parser_engine.py:595 plus the qwen3
// override (qwen3.py:256): an UNPAIRED <tool_call> also ends reasoning.
TEST_CASE("qwen3: is_reasoning_end (text form, incl. unpaired <tool_call>)") {
  auto p = get_reasoning_parser("qwen3");
  REQUIRE(p != nullptr);
  // initial_state is REASONING (thinking on), so an empty / marker-less stream
  // has NOT ended reasoning.
  CHECK(p->is_reasoning_end("") == false);
  CHECK(p->is_reasoning_end("still thinking") == false);
  CHECK(p->is_reasoning_end("<think>still thinking") == false);
  CHECK(p->is_reasoning_end("reasoning</think>answer") == true);
  CHECK(p->is_reasoning_end("<think>reasoning</think>answer") == true);
  // A later <think> re-opens the span (upstream scans backwards and returns
  // False on the start token before it reaches the end token).
  CHECK(p->is_reasoning_end("a</think>b<think>c") == false);
  // qwen3.py:262-275 — an unpaired <tool_call> ends reasoning; a PAIRED one
  // (a </tool_call> after it) does not by itself.
  CHECK(p->is_reasoning_end("thinking<tool_call>\n<function=bash>") == true);
  CHECK(p->is_reasoning_end("thinking<tool_call>x</tool_call>") == false);
  CHECK(p->is_reasoning_end("<think>a<tool_call>b") == true);
  // The three above all put <think> BEFORE the tool-call marker, where the
  // re-open guard cannot fire. This one puts it AFTER: upstream's backwards walk
  // (qwen3.py:263-268) tests the reasoning-start id at EVERY index before it
  // tests the tool-call id, so it meets <think> first and returns False —
  // reasoning has re-opened and the earlier unpaired <tool_call> no longer ends
  // it. It is the only assertion here that fails if `ps > p` (qwen3.cpp:55) is
  // dropped.
  CHECK(p->is_reasoning_end("<tool_call>x<think>y") == false);
}

// Upstream seed_oss is `class SeedOssParser(Qwen3Parser)` (seed_oss.py:24)
// overriding only the four wrapper token strings, so it INHERITS the
// thinking-off passthrough (qwen3.py:247). Since #630 (#605) our
// `get_parser_engine` mirrors that by building the same Qwen3Parser class over
// the `<seed:...>` spelling (parser_manager.cpp:28); before that, seed_oss got
// a bare ParserEngine and the thinking-off arm ran the state machine instead.
//
// Nothing else pins the inheritance. The only production call site,
// `OpenAIServingChat::MakeParserEngine` (serving_chat.cpp:580), takes the
// header default thinking=true, under which a bare engine and Qwen3Parser are
// byte-identical — so ONLY the thinking-off arm can tell the two apart, which
// is why this case exists.
TEST_CASE("seed_oss inherits the qwen3 thinking-off passthrough") {
  namespace pe = vllm::parser::engine;
  // seed_oss.py:26-29 — the qwen3 grammar with `seed:`-prefixed wrappers;
  // `<function=>`/`<parameter=>` are byte-identical.
  const std::string seed_tool_body =
      "<seed:tool_call>\n<function=bash>\n<parameter=command>"
      "\ncat /etc/hosts\n</parameter>\n</function>\n</seed:tool_call>";
  const std::string output = "I need to read the file.\n\n" + seed_tool_body;

  auto off = vllm::parser::get_parser_engine("seed_oss", /*thinking=*/false);
  REQUIRE(off != nullptr);
  const auto [reasoning, content] =
      off->extract_reasoning(output, pe::ParserRequest{});
  CHECK(reasoning == std::nullopt);
  CHECK(content == Opt(output));

  // Same name, thinking ON: the initial state is REASONING and the unpaired
  // <seed:tool_call> is an implicit reasoning end, so the content span is NOT
  // the whole output. That difference is exactly what the override suppresses.
  auto on = vllm::parser::get_parser_engine("seed_oss", /*thinking=*/true);
  REQUIRE(on != nullptr);
  const auto [r2, c2] = on->extract_reasoning(output, pe::ParserRequest{});
  CHECK(r2 == Opt("I need to read the file.\n\n"));
  CHECK(c2 != Opt(output));
}
