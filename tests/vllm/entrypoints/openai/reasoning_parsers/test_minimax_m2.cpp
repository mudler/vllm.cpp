// Ports tests/reasoning/test_minimax_m2_reasoning_parser.py and
// test_minimax_m2_append_reasoning_parser.py @ e24d1b24.
//   "minimax_m2":              end-marker-only split (before </think> = reasoning).
//   "minimax_m2_append_think": pass-through, prepends "<think>", all content.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include "reasoning_test_utils.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

using namespace vllm::entrypoints::openai;
using vllm::entrypoints::openai::reasoning_test::Extracted;
using vllm::entrypoints::openai::reasoning_test::RunExtraction;

namespace {
using Opt = std::optional<std::string>;

struct Case {
  const char* id;
  bool streaming;
  std::vector<std::string> deltas;
  Opt reasoning;
  Opt content;
};

void Run(const char* name, const Case& c) {
  auto parser = get_reasoning_parser(name);
  REQUIRE(parser != nullptr);
  const Extracted r = RunExtraction(*parser, c.deltas, c.streaming);
  INFO("parser=", name, " case=", c.id);
  CHECK(r.reasoning == c.reasoning);
  CHECK(r.content == c.content);
}
}  // namespace

TEST_CASE("minimax_m2: registered") {
  CHECK(get_reasoning_parser("minimax_m2") != nullptr);
  CHECK(get_reasoning_parser("minimax_m2_append_think") != nullptr);
}

TEST_CASE("minimax_m2: end-marker-only reasoning contract") {
  const std::vector<Case> cases = {
      {"simple", false, {"This is a reasoning section</think>This is the rest"},
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"simple_stream", true,
       {"This is a reasoning section", "</think>", "This is the rest"},
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"complete", false, {"This is a reasoning section</think>"},
       Opt("This is a reasoning section"), std::nullopt},
      {"complete_stream", true, {"This is a reasoning section", "</think>"},
       Opt("This is a reasoning section"), std::nullopt},
      {"no_end", false, {"This is reasoning in progress"},
       Opt("This is reasoning in progress"), std::nullopt},
      {"no_end_stream", true, {"This is reasoning in progress"},
       Opt("This is reasoning in progress"), std::nullopt},
      {"multiple_lines", false,
       {"First line\nSecond line</think>Response first line\nResponse second"},
       Opt("First line\nSecond line"),
       Opt("Response first line\nResponse second")},
      {"multiple_lines_stream", true,
       {"First line\nSecond line", "</think>",
        "Response first line\nResponse second"},
       Opt("First line\nSecond line"),
       Opt("Response first line\nResponse second")},
      {"shortest_stream", true, {"</think>", "This is the response"},
       std::nullopt, Opt("This is the response")},
      {"empty_stream", true, {}, std::nullopt, std::nullopt},
      {"special_chars", false,
       {"Let me think... 1+1=2, right?</think>Yes, 1+1=2."},
       Opt("Let me think... 1+1=2, right?"), Opt("Yes, 1+1=2.")},
      {"special_chars_stream", true,
       {"Let me think... 1+1=2, right?", "</think>", "Yes, 1+1=2."},
       Opt("Let me think... 1+1=2, right?"), Opt("Yes, 1+1=2.")},
      {"code", false,
       {"```python\nprint('hello')\n```</think>Here is the code."},
       Opt("```python\nprint('hello')\n```"), Opt("Here is the code.")},
      {"code_stream", true,
       {"```python\nprint('hello')\n```", "</think>", "Here is the code."},
       Opt("```python\nprint('hello')\n```"), Opt("Here is the code.")},
  };
  for (const Case& c : cases) Run("minimax_m2", c);

  auto p = get_reasoning_parser("minimax_m2");
  CHECK(p->is_reasoning_end("reasoning</think>content") == true);
  CHECK(p->is_reasoning_end("This is reasoning in progress") == false);
}

TEST_CASE("minimax_m2_append_think: pass-through contract") {
  const std::vector<Case> cases = {
      {"simple", false, {"This is reasoning</think>This is response"},
       std::nullopt, Opt("<think>This is reasoning</think>This is response")},
      {"simple_stream", true, {"This is reasoning</think>This is response"},
       std::nullopt, Opt("<think>This is reasoning</think>This is response")},
      {"no_end", false, {"This is reasoning in progress"}, std::nullopt,
       Opt("<think>This is reasoning in progress")},
      {"no_end_stream", true, {"This is reasoning in progress"}, std::nullopt,
       Opt("<think>This is reasoning in progress")},
      {"only_end", false, {"</think>This is response"}, std::nullopt,
       Opt("<think></think>This is response")},
      {"only_end_stream", true, {"</think>This is response"}, std::nullopt,
       Opt("<think></think>This is response")},
      {"multiple_lines", false, {"Line 1\nLine 2</think>Response 1\nResponse 2"},
       std::nullopt, Opt("<think>Line 1\nLine 2</think>Response 1\nResponse 2")},
      {"multiple_lines_stream", true,
       {"Line 1\nLine 2</think>Response 1\nResponse 2"}, std::nullopt,
       Opt("<think>Line 1\nLine 2</think>Response 1\nResponse 2")},
      {"empty", false, {""}, std::nullopt, Opt("<think>")},
      {"empty_stream", true, {}, std::nullopt, std::nullopt},
      {"special_chars", false, {"Let me think... 1+1=2</think>Yes!"},
       std::nullopt, Opt("<think>Let me think... 1+1=2</think>Yes!")},
      {"special_chars_stream", true, {"Let me think... 1+1=2</think>Yes!"},
       std::nullopt, Opt("<think>Let me think... 1+1=2</think>Yes!")},
      {"code", false, {"```python\nprint('hi')\n```</think>Here's the code."},
       std::nullopt,
       Opt("<think>```python\nprint('hi')\n```</think>Here's the code.")},
      {"code_stream", true,
       {"```python\nprint('hi')\n```</think>Here's the code."}, std::nullopt,
       Opt("<think>```python\nprint('hi')\n```</think>Here's the code.")},
  };
  for (const Case& c : cases) Run("minimax_m2_append_think", c);

  auto p = get_reasoning_parser("minimax_m2_append_think");
  CHECK(p->is_reasoning_end("This is reasoning</think>This is response") == true);
  CHECK(p->is_reasoning_end("This is reasoning in progress") == false);
  CHECK(p->is_reasoning_end("</think>This is response") == true);
}
