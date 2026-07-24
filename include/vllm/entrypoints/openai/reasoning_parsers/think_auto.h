// ORIGINAL packaging-layer component (no upstream mirror): the reasoning
// parser the TEMPLATE AUTO-DETECTION selects for generic <think> templates.
//
// Why it exists: deepseek_r1 semantics (basic_parsers.py - text before the
// first </think> is reasoning even when no <think> was emitted) are correct
// ONLY for models whose template PRE-FILLS the opening <think> into the
// generation prompt. Hybrid-thinking families (Qwen3.5) carry <think> in the
// template for history-stripping but may answer WITHOUT any think block; R1
// semantics would then swallow the entire answer as reasoning and leave
// content null. llama.cpp's autoparser handles the same ambiguity by treating
// markerless output as plain content - this parser mirrors that rule:
//   - no marker in the output        => everything is CONTENT;
//   - <think>...</think> present     => split at the markers (R1-identical);
//   - only </think> present          => the prefix is reasoning (the
//     pre-filled-<think> case, R1-identical).
// Registered as "think_auto"; explicitly selectable but primarily the target
// of the "<think>" row in reasoning_parsers/detect.cpp.
#ifndef VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_THINK_AUTO_H_
#define VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_THINK_AUTO_H_

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/reasoning_parsers/basic.h"

namespace vllm::entrypoints::openai {

class ThinkAutoReasoningParser final : public BaseThinkingReasoningParser {
 public:
  const std::string& start_token() const override;
  const std::string& end_token() const override;

  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  // Streaming decides thinking-mode ONCE from the head of the output; the
  // instance is per-request (like every streaming parser), so the decision
  // and the base-parser priming survive across delta calls.
  enum class Mode { kUndecided, kContent, kThink };
  Mode mode_ = Mode::kUndecided;
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_THINK_AUTO_H_
