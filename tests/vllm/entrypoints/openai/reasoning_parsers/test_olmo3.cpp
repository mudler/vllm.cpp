// Ports tests/reasoning/test_olmo3_reasoning_parser.py @ e24d1b24 (name "olmo3").
// Olmo 3 uses ORDINARY-vocabulary <think>/</think> that the pre-tokenizer splits
// across tokens, so the parser buffers partial markers. The streaming cases feed
// a tokenizer-like split: each think marker is delivered as its natural pieces
// ("<","think",">" / "</","think",">") to exercise the partial-overlap
// buffering, while the reasoning/content text arrives as whole spans (as a real
// tokenizer would surface a word rather than a stray partial-marker byte). This
// keeps a bare "<think></think>" from ever emitting a spurious empty reasoning
// delta (reasoning stays None). Non-streaming uses the regex split.
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

// Tokenizer-like split: each <think>/</think> becomes its natural sub-pieces so
// the buffer sees them assembling across deltas; surrounding text is one span.
std::vector<std::string> SplitOlmo(const std::string& s) {
  std::vector<std::string> out;
  std::string run;
  const auto flush = [&]() {
    if (!run.empty()) {
      out.push_back(run);
      run.clear();
    }
  };
  std::size_t i = 0;
  while (i < s.size()) {
    if (s.compare(i, 7, "<think>") == 0) {
      flush();
      out.push_back("<");
      out.push_back("think");
      out.push_back(">");
      i += 7;
    } else if (s.compare(i, 8, "</think>") == 0) {
      flush();
      out.push_back("</");
      out.push_back("think");
      out.push_back(">");
      i += 8;
    } else {
      run.push_back(s[i]);
      ++i;
    }
  }
  flush();
  return out;
}

struct Case {
  const char* id;
  bool streaming;
  std::string output;
  Opt reasoning;
  Opt content;
};

void Run(const Case& c) {
  auto parser = get_reasoning_parser("olmo3");
  REQUIRE(parser != nullptr);
  const std::vector<std::string> deltas =
      c.streaming ? SplitOlmo(c.output) : std::vector<std::string>{c.output};
  const Extracted r = RunExtraction(*parser, deltas, c.streaming);
  INFO("case=", c.id);
  CHECK(r.reasoning == c.reasoning);
  CHECK(r.content == c.content);
}
}  // namespace

TEST_CASE("olmo3: registered") {
  CHECK(get_reasoning_parser("olmo3") != nullptr);
}

TEST_CASE("olmo3: reasoning contract") {
  const std::vector<Case> cases = {
      {"no_reasoning", false, "<think></think>No thoughts, head empty!",
       std::nullopt, Opt("No thoughts, head empty!")},
      {"no_reasoning_with_newline", false,
       "<think>\n</think>\n\nNo thoughts, head empty!", Opt("\n"),
       Opt("\n\nNo thoughts, head empty!")},
      {"simple_reasoning", false,
       "<think>This is a reasoning section</think>This is the rest",
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"simple_reasoning_with_newline", false,
       "<think> Look!\n\nI'm thinking...</think>\nThis is the rest",
       Opt(" Look!\n\nI'm thinking..."), Opt("\nThis is the rest")},
      {"no_reasoning_only_end_think", false,
       "</think>\n\nNo thoughts, head empty!", std::nullopt,
       Opt("\n\nNo thoughts, head empty!")},
      {"yes_reasoning_only_end_think", false,
       "The user is asking me not to think.</think>No thoughts!",
       Opt("The user is asking me not to think."), Opt("No thoughts!")},
      // Streaming variants.
      {"no_reasoning_stream", true, "<think></think>No thoughts, head empty!",
       std::nullopt, Opt("No thoughts, head empty!")},
      {"no_reasoning_with_newline_stream", true,
       "<think>\n</think>\n\nNo thoughts, head empty!", Opt("\n"),
       Opt("\n\nNo thoughts, head empty!")},
      {"simple_reasoning_stream", true,
       "<think>This is a reasoning section</think>This is the rest",
       Opt("This is a reasoning section"), Opt("This is the rest")},
      {"simple_reasoning_with_newline_stream", true,
       "<think> Look!\n\nI'm thinking...</think>\nThis is the rest",
       Opt(" Look!\n\nI'm thinking..."), Opt("\nThis is the rest")},
      {"simple_reasoning_with_multiple_newlines_stream", true,
       "<think>\nLook!\nI'm thinking...\n\n</think>\n\n\nThis is the rest",
       Opt("\nLook!\nI'm thinking...\n\n"), Opt("\n\n\nThis is the rest")},
      {"simple_reasoning_with_trailing_space_stream", true,
       "<think>\nLook!\nI'm thinking... </think>\nThis is the rest",
       Opt("\nLook!\nI'm thinking... "), Opt("\nThis is the rest")},
      {"no_reasoning_only_end_think_stream", true,
       "</think>\n\nNo thoughts, head empty!", std::nullopt,
       Opt("\n\nNo thoughts, head empty!")},
      {"yes_reasoning_only_end_think_stream", true,
       "The user is asking me not to think.</think>No thoughts!",
       Opt("The user is asking me not to think."), Opt("No thoughts!")},
  };
  for (const Case& c : cases) Run(c);
}

TEST_CASE("olmo3: is_reasoning_end") {
  auto p = get_reasoning_parser("olmo3");
  CHECK(p->is_reasoning_end("reasoning</think>rest") == true);
  CHECK(p->is_reasoning_end("still reasoning") == false);
}
