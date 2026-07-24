// Ported from: vllm/reasoning/abs_reasoning_parsers.py @ e24d1b24
// (ReasoningParser ABC) + vllm/reasoning/basic_parsers.py extraction contract +
// the ReasoningParserManager factory (get_reasoning_parser).
//
// The ReasoningParser splits a model's raw output into a REASONING span
// (chain-of-thought, usually wrapped in <think>...</think>-style markers) and
// the user-visible CONTENT. It covers BOTH the non-streaming extract_reasoning
// (a whole-string split) AND the STREAMING extract_reasoning_streaming (the
// incremental per-delta split that carries reasoning/content on a DeltaMessage).
//
// DEVIATIONS from upstream shape (all TEXT-ONLY-scoped, mirroring the
// tool_parsers/abstract.h seam):
//   - The upstream ReasoningParser.__init__ takes a tokenizer and every
//     token-ID method (is_reasoning_end(input_ids), extract_content_ids,
//     is_reasoning_end_streaming, count_reasoning_tokens) operates on token-ID
//     spans. Those exist ONLY for the structured-decode engines (xgrammar) and
//     are NOT needed by the serving-layer text extraction. We therefore give the
//     ABC a default ctor, drop the tokenizer wiring, and make is_reasoning_end
//     TEXT-based (over the accumulated output string). The other token-ID
//     methods are dropped. Because the think markers are self-delimiting special
//     tokens (they carry their own '<','>','[',']'), a substring match over the
//     detokenized text is equivalent to a token-ID membership test for a
//     well-formed stream (e.g. "<test:think>" is NOT a substring of
//     "<test:thinking>" because the token's own '>' does not match) - the same
//     property the tool-parser seam relies on.
//   - The upstream streaming signature also takes previous/current/delta
//     TOKEN-ID spans. The text-only parse never reads them, so we drop those
//     three params to keep the serving seam small - documented deviation.
//   - ReasoningParserManager's lazy/plugin registry collapses to a small
//     hand-wired get_reasoning_parser() factory over the T0 formats.
#pragma once

#include <memory>
#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"

namespace vllm::entrypoints::openai {

// Ported from: abs_reasoning_parsers.py:146 (extract_reasoning) return shape -
// the (reasoning, content) tuple. Either element is nullopt when that span is
// absent (upstream returns None); an empty-but-present reasoning span (e.g.
// "<think></think>...") is an empty string, distinct from nullopt.
struct ExtractedReasoning {
  std::optional<std::string> reasoning;
  std::optional<std::string> content;
};

// Ported from: abs_reasoning_parsers.py:26 (ReasoningParser). Abstract base;
// derived classes implement the text-only extraction contract.
class ReasoningParser {
 public:
  ReasoningParser() = default;
  virtual ~ReasoningParser() = default;

  ReasoningParser(const ReasoningParser&) = delete;
  ReasoningParser& operator=(const ReasoningParser&) = delete;

  // Ported from: abs_reasoning_parsers.py:146 (extract_reasoning). Split a
  // COMPLETE model-generated string into (reasoning, content). Non-streaming,
  // stateless for the BaseThinking family (olmo3 keeps a streaming buffer but
  // its non-streaming path is a stateless regex).
  virtual ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) = 0;

  // Ported from: abs_reasoning_parsers.py:166 (extract_reasoning_streaming). The
  // INCREMENTAL parse: given the accumulated `previous_text`, the new
  // `current_text` (= previous_text + delta_text) and the `delta_text` fragment,
  // return the DeltaMessage to emit (its `reasoning` and/or `content` slot set),
  // or nullopt when there is nothing new to send yet (upstream returns None -
  // e.g. a lone think-marker delta is swallowed). May be STATEFUL (olmo3 buffers
  // partial markers), so ONE parser must live for the whole stream.
  virtual std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) = 0;

  // Ported from: abs_reasoning_parsers.py:73 (is_reasoning_end) - TEXT form
  // (deviation above). True once the reasoning span has closed in `text` (the
  // accumulated output). Used to decide when downstream content parsing (tool
  // calls) may begin.
  virtual bool is_reasoning_end(const std::string& text) const = 0;
};

// Ported from: abs_reasoning_parsers.py:227
// (ReasoningParserManager.get_reasoning_parser). Returns a fresh parser for
// `name`, or nullptr when the name is not one of the T0-registered formats
// (upstream raises KeyError). Registered names: "deepseek_r1", "mistral",
// "minimax_m2", "minimax_m2_append_think", "step3", "olmo3".
std::unique_ptr<ReasoningParser> get_reasoning_parser(const std::string& name);

}  // namespace vllm::entrypoints::openai
