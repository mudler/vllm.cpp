// Ports the BaseThinkingReasoningParser behavioural contract from
// tests/reasoning/test_base_thinking_reasoning_parser.py @ e24d1b24.
//
// The upstream suite has 25 cases. The TEXT-ONLY seam (reasoning_parsers/
// abstract.h) drops the token-ID surface, so the following upstream cases are
// intentionally NOT ported (they exercise dropped machinery, not extraction
// behaviour): the 4 init/vocab cases (successful_initialization,
// missing_tokenizer, missing_tokens, empty_tokens; tokenizer wiring),
// test_is_reasoning_end_streaming / test_count_reasoning_tokens[_nested] /
// test_extract_content_ids (token-ID methods). test_is_reasoning_end is ported
// in TEXT form. The full EXTRACTION + STREAMING + edge + multi-implementation
// contract (the part the serving layer relies on) is ported 1:1.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include "reasoning_test_utils.h"
#include "vllm/entrypoints/openai/reasoning_parsers/basic.h"

using namespace vllm::entrypoints::openai;
using vllm::entrypoints::openai::reasoning_test::Extracted;
using vllm::entrypoints::openai::reasoning_test::RunExtraction;

namespace {

// Mirrors the upstream TestThinkingReasoningParser (custom <test:think> tokens).
class TestThinkingReasoningParser final : public BaseThinkingReasoningParser {
 public:
  const std::string& start_token() const override { return start_; }
  const std::string& end_token() const override { return end_; }

 private:
  const std::string start_ = "<test:think>";
  const std::string end_ = "</test:think>";
};

class TestThinkingReasoningParserAlt final : public BaseThinkingReasoningParser {
 public:
  const std::string& start_token() const override { return start_; }
  const std::string& end_token() const override { return end_; }

 private:
  const std::string start_ = "<alt:start>";
  const std::string end_ = "<alt:end>";
};

}  // namespace

// ── is_reasoning_end (TEXT form of test_is_reasoning_end) ─────────────────────
TEST_CASE("base_thinking: is_reasoning_end (text form)") {
  TestThinkingReasoningParser p;
  CHECK(p.is_reasoning_end("ab</test:think>cd") == true);
  CHECK(p.is_reasoning_end("abcd") == false);
  CHECK(p.is_reasoning_end("") == false);
  CHECK(p.is_reasoning_end("<test:think>x</test:think>") == true);
  CHECK(p.is_reasoning_end("<test:think>xy") == false);
  CHECK(p.is_reasoning_end("<test:think>x</test:think>yy<test:think>") == false);
}

// ── extraction (TestBaseThinkingReasoningParserExtraction) ────────────────────
TEST_CASE("base_thinking: extract with both tokens") {
  TestThinkingReasoningParser p;
  const Extracted r =
      RunExtraction(p, {"<test:think>This is reasoning</test:think>This is content"},
                    /*streaming=*/false);
  CHECK(r.reasoning == std::optional<std::string>("This is reasoning"));
  CHECK(r.content == std::optional<std::string>("This is content"));
}

TEST_CASE("base_thinking: extract only end token") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(
      p, {"This is reasoning</test:think>This is content"}, false);
  CHECK(r.reasoning == std::optional<std::string>("This is reasoning"));
  CHECK(r.content == std::optional<std::string>("This is content"));
}

TEST_CASE("base_thinking: extract no end token") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(p, {"This is just content"}, false);
  CHECK(r.reasoning == std::optional<std::string>("This is just content"));
  CHECK(r.content == std::nullopt);
}

TEST_CASE("base_thinking: extract empty output") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(p, {""}, false);
  CHECK(r.reasoning == std::optional<std::string>(""));
  CHECK(r.content == std::nullopt);
}

TEST_CASE("base_thinking: extract only tokens") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(p, {"<test:think></test:think>"}, false);
  CHECK(r.reasoning == std::optional<std::string>(""));
  CHECK(r.content == std::nullopt);
}

// ── streaming (TestBaseThinkingReasoningParserStreaming) ──────────────────────
TEST_CASE("base_thinking: simple reasoning extraction (stream + non-stream)") {
  const std::vector<std::string> deltas = {
      "<test:think>", "Some ", "reasoning ", "content",
      "</test:think>", "Final ", "answer"};
  for (bool streaming : {true, false}) {
    TestThinkingReasoningParser p;
    const Extracted r = RunExtraction(p, deltas, streaming);
    CHECK(r.reasoning == std::optional<std::string>("Some reasoning content"));
    CHECK(r.content == std::optional<std::string>("Final answer"));
  }
}

TEST_CASE("base_thinking: streaming with start token") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(
      p, {"<test:think>", "Some ", "reasoning", "</test:think>", "Answer"}, true);
  CHECK(r.reasoning == std::optional<std::string>("Some reasoning"));
  CHECK(r.content == std::optional<std::string>("Answer"));
}

TEST_CASE("base_thinking: streaming no end token") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(
      p, {"<test:think>", "Some ", "reasoning ", "without ", "end"}, true);
  CHECK(r.reasoning == std::optional<std::string>("Some reasoning without end"));
  CHECK(r.content == std::nullopt);
}

TEST_CASE("base_thinking: streaming only end token") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(
      p, {"<test:think>", "Reasoning ", "content", "</test:think>", "Final"}, true);
  CHECK(r.reasoning == std::optional<std::string>("Reasoning content"));
  CHECK(r.content == std::optional<std::string>("Final"));
}

// ── multiple implementations (independent tokens) ────────────────────────────
TEST_CASE("base_thinking: different token implementations are independent") {
  TestThinkingReasoningParser p1;
  TestThinkingReasoningParserAlt p2;

  const Extracted r1 = RunExtraction(p1, {"Reasoning1</test:think>Content1"}, false);
  CHECK(r1.reasoning == std::optional<std::string>("Reasoning1"));
  CHECK(r1.content == std::optional<std::string>("Content1"));

  const Extracted r2 = RunExtraction(p2, {"Reasoning2<alt:end>Content2"}, false);
  CHECK(r2.reasoning == std::optional<std::string>("Reasoning2"));
  CHECK(r2.content == std::optional<std::string>("Content2"));

  CHECK(p1.start_token() != p2.start_token());
  CHECK(p1.end_token() != p2.end_token());
}

// ── edge cases (TestBaseThinkingReasoningParserEdgeCases) ─────────────────────
TEST_CASE("base_thinking: multiple end tokens stop at first") {
  TestThinkingReasoningParser p;
  const Extracted r =
      RunExtraction(p, {"First</test:think>Middle</test:think>Last"}, false);
  CHECK(r.reasoning == std::optional<std::string>("First"));
  CHECK(r.content == std::optional<std::string>("Middle</test:think>Last"));
}

TEST_CASE("base_thinking: nested-like token patterns") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(
      p, {"<test:think>Outer<test:think>Inner</test:think>Content"}, false);
  CHECK(r.reasoning == std::optional<std::string>("Outer<test:think>Inner"));
  CHECK(r.content == std::optional<std::string>("Content"));
}

TEST_CASE("base_thinking: malformed token-like strings are plain content") {
  TestThinkingReasoningParser p;
  const Extracted r = RunExtraction(
      p, {"<test:thinking>Not a real token</test:thinking>Content"}, false);
  CHECK(r.reasoning ==
        std::optional<std::string>(
            "<test:thinking>Not a real token</test:thinking>Content"));
  CHECK(r.content == std::nullopt);
}
