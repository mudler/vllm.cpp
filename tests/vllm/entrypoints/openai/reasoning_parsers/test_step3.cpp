// Tests for the Step3 reasoning parser (step3_reasoning_parser.py @ e24d1b24,
// name "step3"). Upstream ships no dedicated pytest module for step3, so these
// cases assert the documented behaviour directly: everything before the first
// </think> is reasoning, everything after is content, a leading <think> is
// optional AND remains part of the reasoning span (only </think> is a marker).
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
  auto parser = get_reasoning_parser("step3");
  REQUIRE(parser != nullptr);
  const Extracted r = RunExtraction(*parser, c.deltas, c.streaming);
  INFO("case=", c.id);
  CHECK(r.reasoning == c.reasoning);
  CHECK(r.content == c.content);
}
}  // namespace

TEST_CASE("step3: registered") {
  CHECK(get_reasoning_parser("step3") != nullptr);
}

TEST_CASE("step3: reasoning contract") {
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
      {"no_end", false, {"just reasoning here"}, Opt("just reasoning here"),
       std::nullopt},
      {"no_end_stream", true, {"just reasoning here"}, Opt("just reasoning here"),
       std::nullopt},
      {"shortest_stream", true, {"</think>", "This is the rest"}, std::nullopt,
       Opt("This is the rest")},
      {"shortest_nonstream", false, {"</think>This is the rest"}, Opt(""),
       Opt("This is the rest")},
      {"optional_think_kept", false,
       {"<think>This is a reasoning section</think>This is the rest"},
       Opt("<think>This is a reasoning section"), Opt("This is the rest")},
      {"optional_think_kept_stream", true,
       {"<think>This is a reasoning section", "</think>", "This is the rest"},
       Opt("<think>This is a reasoning section"), Opt("This is the rest")},
      {"multiple_lines", false, {"This\nThat</think>rest\nmore"},
       Opt("This\nThat"), Opt("rest\nmore")},
  };
  for (const Case& c : cases) Run(c);
}

TEST_CASE("step3: is_reasoning_end") {
  auto p = get_reasoning_parser("step3");
  CHECK(p->is_reasoning_end("reasoning</think>rest") == true);
  CHECK(p->is_reasoning_end("still reasoning") == false);
}
