// Ports tests/reasoning/test_mistral_reasoning_parser.py @ e24d1b24
// (name "mistral", [THINK]...[/THINK]). A bare [/THINK] with no preceding
// [THINK] is invalid: the marker is stripped and nothing is reasoning. Mistral
// reuses the BaseThinking streaming engine, with a custom non-streaming split.
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

void Run(const Case& c) {
  auto parser = get_reasoning_parser("mistral");
  REQUIRE(parser != nullptr);
  const Extracted r = RunExtraction(*parser, c.deltas, c.streaming);
  INFO("case=", c.id);
  CHECK(r.reasoning == c.reasoning);
  CHECK(r.content == c.content);
}
}  // namespace

TEST_CASE("mistral: registered") {
  CHECK(get_reasoning_parser("mistral") != nullptr);
}

TEST_CASE("mistral: reasoning contract") {
  const std::vector<Case> cases = {
      {"invalid_simple", false,
       {"This is a reasoning section[/THINK]This is the rest"}, std::nullopt,
       Opt("This is a reasoning sectionThis is the rest")},
      {"invalid_simple_stream", true,
       {"This is a reasoning section", "[/THINK]", "This is the rest"},
       std::nullopt, Opt("This is a reasoning sectionThis is the rest")},
      {"invalid_complete", false, {"This is a reasoning section[/THINK]"},
       std::nullopt, Opt("This is a reasoning section")},
      {"invalid_complete_stream", true,
       {"This is a reasoning section", "[/THINK]"}, std::nullopt,
       Opt("This is a reasoning section")},
      {"no_content", false, {"[THINK]This is reasoning"}, Opt("This is reasoning"),
       std::nullopt},
      {"no_reasoning", false, {"This is content"}, std::nullopt,
       Opt("This is content")},
      {"no_reasoning_stream", true, {"This is a reasoning section"},
       std::nullopt, Opt("This is a reasoning section")},
      {"invalid_multiple_lines", false,
       {"This\nThat[/THINK]This is the rest\nThat"}, std::nullopt,
       Opt("This\nThatThis is the rest\nThat")},
      {"invalid_multiple_lines_stream", true,
       {"This\nThat", "[/THINK]", "This is the rest\nThat"}, std::nullopt,
       Opt("This\nThatThis is the rest\nThat")},
      {"invalid_shortest_stream", true, {"[/THINK]", "This is the rest"},
       std::nullopt, Opt("This is the rest")},
      {"invalid_shortest_nonstream", false, {"[/THINK]This is the rest"},
       std::nullopt, Opt("This is the rest")},
      {"reasoning_with_think", false,
       {"[THINK]This is a reasoning section[/THINK]This is the rest"},
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"reasoning_with_think_stream", true,
       {"[THINK]", "This is a reasoning section", "[/THINK]", "This is the rest"},
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"complete_with_think", false,
       {"[THINK]This is a reasoning section[/THINK]"},
       Opt("This is a reasoning section"), std::nullopt},
      {"complete_with_think_stream", true,
       {"[THINK]", "This is a reasoning section", "[/THINK]"},
       Opt("This is a reasoning section"), std::nullopt},
      {"multiple_lines_with_think", false,
       {"[THINK]This\nThat[/THINK]This is the rest\nThat"}, Opt("This\nThat"),
       Opt("This is the rest\nThat")},
      {"multiple_lines_with_think_stream", true,
       {"[THINK]", "This\nThat", "[/THINK]", "This is the rest\nThat"},
       Opt("This\nThat"), Opt("This is the rest\nThat")},
      {"invalid_shortest_with_think", false, {"[/THINK]This is the rest"},
       std::nullopt, Opt("This is the rest")},
      {"invalid_shortest_with_think_stream", true,
       {"[/THINK]", "This is the rest"}, std::nullopt, Opt("This is the rest")},
      {"think_no_end", false, {"[THINK]This is a reasoning section"},
       Opt("This is a reasoning section"), std::nullopt},
      {"think_no_end_stream", true,
       {"[THINK]", "This is a reasoning section"},
       Opt("This is a reasoning section"), std::nullopt},
      {"empty", false, {""}, std::nullopt, Opt("")},
      {"empty_stream", true, {}, std::nullopt, std::nullopt},
      {"new_line", false,
       {"Before\n[THINK]This is a reasoning section[/THINK]\nThis is the rest"},
       Opt("This is a reasoning section"), Opt("Before\n\nThis is the rest")},
      {"new_line_stream", true,
       {"Before\n", "[THINK]", "This is a reasoning section", "[/THINK]",
        "\nThis is the rest"},
       Opt("This is a reasoning section"), Opt("Before\n\nThis is the rest")},
  };
  for (const Case& c : cases) Run(c);
}

TEST_CASE("mistral: is_reasoning_end requires a start marker") {
  auto p = get_reasoning_parser("mistral");
  CHECK(p->is_reasoning_end("[THINK]reasoning[/THINK]rest") == true);
  CHECK(p->is_reasoning_end("[THINK]reasoning") == false);
  CHECK(p->is_reasoning_end("reasoning[/THINK]rest") == false);  // no [THINK]
  CHECK(p->is_reasoning_end("plain content") == false);
}
