// think_auto (ORIGINAL, auto-detection target for generic <think> templates):
// symmetric semantics - markerless output is CONTENT, marked output splits
// exactly like deepseek_r1.
#include "vllm/entrypoints/openai/reasoning_parsers/think_auto.h"

#include <doctest/doctest.h>

#include <string>

#include "vllm/entrypoints/openai/protocol.h"

using vllm::entrypoints::openai::ChatCompletionRequest;
using vllm::entrypoints::openai::ThinkAutoReasoningParser;

namespace {
const ChatCompletionRequest kReq;
}

TEST_CASE("think_auto: markerless output is pure content (the hybrid no-think case)") {
  ThinkAutoReasoningParser p;
  const auto out = p.extract_reasoning("The capital of France is Paris.", kReq);
  CHECK(!out.reasoning.has_value());
  REQUIRE(out.content.has_value());
  CHECK(*out.content == "The capital of France is Paris.");
}

TEST_CASE("think_auto: a full think block splits like deepseek_r1") {
  ThinkAutoReasoningParser p;
  const auto out =
      p.extract_reasoning("<think>plan it</think>Paris.", kReq);
  REQUIRE(out.reasoning.has_value());
  CHECK(*out.reasoning == "plan it");
  REQUIRE(out.content.has_value());
  CHECK(*out.content == "Paris.");
}

TEST_CASE("think_auto: end-only output keeps the R1 pre-filled semantics") {
  ThinkAutoReasoningParser p;
  const auto out = p.extract_reasoning("plan it</think>Paris.", kReq);
  REQUIRE(out.reasoning.has_value());
  CHECK(*out.reasoning == "plan it");
  REQUIRE(out.content.has_value());
  CHECK(*out.content == "Paris.");
}

TEST_CASE("think_auto: streaming content turn passes deltas through") {
  ThinkAutoReasoningParser p;
  std::string prev;
  std::string text;
  const std::string full = "The capital is Paris.";
  std::string streamed;
  for (std::size_t i = 0; i < full.size(); i += 4) {
    const std::string delta = full.substr(i, 4);
    const std::string cur = full.substr(0, i + delta.size());
    const auto d = p.extract_reasoning_streaming(prev, cur, delta, kReq);
    if (d.has_value() && d->content.has_value()) streamed += *d->content;
    prev = cur;
  }
  CHECK(streamed == full);
}

TEST_CASE("think_auto: streaming think turn splits and suppresses markers") {
  // Marker deltas arrive ATOMIC (the upstream basic_parsers cadence: a real
  // detokenizer emits <think> as one token); text between markers is chunked.
  ThinkAutoReasoningParser p;
  const std::string deltas[] = {"<think>", "a p", "lan", "</think>", "Par",
                                "is."};
  std::string prev;
  std::string reasoning;
  std::string content;
  for (const std::string& delta : deltas) {
    const std::string cur = prev + delta;
    const auto d = p.extract_reasoning_streaming(prev, cur, delta, kReq);
    if (d.has_value()) {
      if (d->reasoning.has_value()) reasoning += *d->reasoning;
      if (d->content.has_value()) content += *d->content;
    }
    prev = cur;
  }
  CHECK(reasoning == "a plan");
  CHECK(content == "Paris.");
}

TEST_CASE("think_auto: the ambiguous '<thi' head is withheld, then flushed") {
  ThinkAutoReasoningParser p;
  // First delta is a strict prefix of "<think>" - must be withheld.
  auto d = p.extract_reasoning_streaming("", "<thi", "<thi", kReq);
  CHECK(!d.has_value());
  // The head then resolves to NOT-a-think-block; the flush carries it all.
  d = p.extract_reasoning_streaming("<thi", "<this is text", "s is text", kReq);
  REQUIRE(d.has_value());
  REQUIRE(d->content.has_value());
  CHECK(*d->content == "<this is text");
}
