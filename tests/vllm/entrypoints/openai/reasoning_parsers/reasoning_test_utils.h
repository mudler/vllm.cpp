// Test harness mirroring tests/reasoning/utils.py @ e24d1b24
// (run_reasoning_extraction / StreamingReasoningReconstructor), adapted to the
// TEXT-ONLY C++ seam. The upstream harness tokenizes each delta and passes
// token-ID spans; the text-only parsers ignore those, so here a "delta" is just
// a text fragment. To mirror how a detokenizer surfaces atomic special tokens,
// think markers are supplied as their own deltas (the same way the upstream
// per-token split isolates them).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai::reasoning_test {

struct Extracted {
  std::optional<std::string> reasoning;
  std::optional<std::string> content;
};

// Mirrors run_reasoning_extraction_nonstreaming: join the deltas and split once.
inline Extracted RunNonStreaming(ReasoningParser& parser,
                                 const std::vector<std::string>& deltas) {
  std::string joined;
  for (const auto& d : deltas) joined += d;
  const ExtractedReasoning er =
      parser.extract_reasoning(joined, ChatCompletionRequest{});
  return Extracted{er.reasoning, er.content};
}

// Mirrors run_reasoning_extraction_streaming + StreamingReasoningReconstructor.
// content "" collapses to nullopt at the end (`other_content or None`);
// reasoning is returned verbatim.
inline Extracted RunStreaming(ReasoningParser& parser,
                              const std::vector<std::string>& deltas) {
  Extracted acc;
  std::string previous;
  for (const auto& delta : deltas) {
    const std::string current = previous + delta;
    const std::optional<DeltaMessage> dm = parser.extract_reasoning_streaming(
        previous, current, delta, ChatCompletionRequest{});
    if (dm.has_value()) {
      if (dm->reasoning.has_value()) {
        if (!acc.reasoning.has_value()) {
          acc.reasoning = *dm->reasoning;
        } else {
          *acc.reasoning += *dm->reasoning;
        }
      }
      if (dm->content.has_value()) {
        if (!acc.content.has_value()) {
          acc.content = *dm->content;
        } else {
          *acc.content += *dm->content;
        }
      }
    }
    previous = current;
  }
  if (acc.content.has_value() && acc.content->empty()) {
    acc.content = std::nullopt;
  }
  return acc;
}

inline Extracted RunExtraction(ReasoningParser& parser,
                               const std::vector<std::string>& deltas,
                               bool streaming) {
  return streaming ? RunStreaming(parser, deltas)
                   : RunNonStreaming(parser, deltas);
}

}  // namespace vllm::entrypoints::openai::reasoning_test
