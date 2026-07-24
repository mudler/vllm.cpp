// Ports tests/reasoning/test_deepseekr1_reasoning_parser.py @ e24d1b24
// (name "deepseek_r1", <think>...</think>). Same input strings and expected
// (reasoning, content) as upstream. Streaming cases isolate the markers as their
// own deltas (the upstream per-token split does the same); the leading-newline
// streaming case is where streaming and non-streaming intentionally diverge.
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
  auto parser = get_reasoning_parser("deepseek_r1");
  REQUIRE(parser != nullptr);
  const Extracted r = RunExtraction(*parser, c.deltas, c.streaming);
  INFO("case=", c.id);
  CHECK(r.reasoning == c.reasoning);
  CHECK(r.content == c.content);
}
}  // namespace

TEST_CASE("deepseek_r1: registered") {
  CHECK(get_reasoning_parser("deepseek_r1") != nullptr);
}

TEST_CASE("deepseek_r1: reasoning contract") {
  const std::vector<Case> cases = {
      {"simple", false,
       {"This is a reasoning section</think>This is the rest"},
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"simple_stream", true,
       {"This is a reasoning section", "</think>", "This is the rest"},
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"complete", false, {"This is a reasoning section</think>"},
       Opt("This is a reasoning section"), std::nullopt},
      {"complete_stream", true,
       {"This is a reasoning section", "</think>"},
       Opt("This is a reasoning section"), std::nullopt},
      {"no_content", false, {"This is content"}, Opt("This is content"),
       std::nullopt},
      {"no_reasoning_stream", true, {"This is a reasoning section"},
       Opt("This is a reasoning section"), std::nullopt},
      {"multiple_lines", false, {"This\nThat</think>This is the rest\nThat"},
       Opt("This\nThat"), Opt("This is the rest\nThat")},
      {"multiple_lines_stream", true,
       {"This\nThat", "</think>", "This is the rest\nThat"},
       Opt("This\nThat"), Opt("This is the rest\nThat")},
      {"shortest_stream", true, {"</think>", "This is the rest"}, std::nullopt,
       Opt("This is the rest")},
      {"shortest_nonstream", false, {"</think>This is the rest"}, Opt(""),
       Opt("This is the rest")},
      {"reasoning_with_think", false,
       {"<think>This is a reasoning section</think>This is the rest"},
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"reasoning_with_think_stream", true,
       {"<think>", "This is a reasoning section", "</think>", "This is the rest"},
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"complete_with_think", false,
       {"<think>This is a reasoning section</think>"},
       Opt("This is a reasoning section"), std::nullopt},
      {"complete_with_think_stream", true,
       {"<think>", "This is a reasoning section", "</think>"},
       Opt("This is a reasoning section"), std::nullopt},
      {"multiple_lines_with_think", false,
       {"<think>This\nThat</think>This is the rest\nThat"}, Opt("This\nThat"),
       Opt("This is the rest\nThat")},
      {"multiple_lines_with_think_stream", true,
       {"<think>", "This\nThat", "</think>", "This is the rest\nThat"},
       Opt("This\nThat"), Opt("This is the rest\nThat")},
      {"shortest_with_think", false, {"</think>This is the rest"}, Opt(""),
       Opt("This is the rest")},
      {"shortest_with_think_stream", true, {"</think>", "This is the rest"},
       std::nullopt, Opt("This is the rest")},
      {"think_no_end", false, {"<think>This is a reasoning section"},
       Opt("This is a reasoning section"), std::nullopt},
      {"think_no_end_stream", true,
       {"<think>", "This is a reasoning section"},
       Opt("This is a reasoning section"), std::nullopt},
      {"empty", false, {""}, Opt(""), std::nullopt},
      {"empty_stream", true, {}, std::nullopt, std::nullopt},
      {"new_line", false,
       {"\n<think>This is a reasoning section</think>\nThis is the rest"},
       Opt("This is a reasoning section"), Opt("\nThis is the rest")},
      {"new_line_stream", true,
       {"\n", "<think>", "This is a reasoning section", "</think>",
        "\nThis is the rest"},
       Opt("\nThis is a reasoning section"), Opt("\nThis is the rest")},
  };
  for (const Case& c : cases) Run(c);
}

TEST_CASE("deepseek_r1: is_reasoning_end") {
  auto p = get_reasoning_parser("deepseek_r1");
  CHECK(p->is_reasoning_end("reasoning</think>rest") == true);
  CHECK(p->is_reasoning_end("still thinking") == false);
  CHECK(p->is_reasoning_end("<think>reasoning</think>rest") == true);
  CHECK(p->is_reasoning_end("<think>reasoning") == false);
}
