// Reasoning-parser auto-detection (ABI v5): the template marker table and its
// safe empty (disabled) default.
#include "vllm/entrypoints/openai/reasoning_parsers/detect.h"

#include <doctest/doctest.h>

#include <string>

#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

using vllm::entrypoints::openai::DetectReasoningParser;
using vllm::entrypoints::openai::get_reasoning_parser;
using vllm::entrypoints::openai::ReasoningParserMarker;
using vllm::entrypoints::openai::ReasoningParserMarkerTable;

TEST_CASE("reasoning detect: [THINK] selects mistral, <think> selects deepseek_r1") {
  CHECK(DetectReasoningParser("..[THINK]..") == "mistral");
  CHECK(DetectReasoningParser("..<think>..") == "deepseek_r1");
}

TEST_CASE("reasoning detect: [THINK] wins over <think> when both appear") {
  CHECK(DetectReasoningParser("[THINK] and <think>") == "mistral");
}

TEST_CASE("reasoning detect: no marker means DISABLED, not a fallback parser") {
  CHECK(DetectReasoningParser("no reasoning markers here").empty());
}

TEST_CASE("reasoning detect: every built-in row names a registered parser") {
  std::size_t n = 0;
  const ReasoningParserMarker* table = ReasoningParserMarkerTable(&n);
  REQUIRE(table != nullptr);
  REQUIRE(n >= 2);
  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    REQUIRE(table[i].parser != nullptr);
    REQUIRE(table[i].template_marker != nullptr);
    CHECK(get_reasoning_parser(table[i].parser) != nullptr);
  }
}
