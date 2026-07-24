// Ported from: vllm/tool_parsers/llama4_pythonic_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/llama4_pythonic.h"

#include <optional>
#include <string>

namespace vllm::entrypoints::openai {

namespace {

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() &&
         s.compare(0, prefix.size(), prefix) == 0;
}
bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Remove all occurrences of `needle` from `s`.
std::string RemoveAll(std::string s, const std::string& needle) {
  if (needle.empty()) return s;
  std::size_t pos = 0;
  while ((pos = s.find(needle, pos)) != std::string::npos) {
    s.erase(pos, needle.size());
  }
  return s;
}

}  // namespace

ExtractedToolCallInformation Llama4PythonicToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // llama4_pythonic_tool_parser.py:74-78 - strip the wrapper markers only when
  // the output actually starts with <|python_start|>.
  std::string cleaned = model_output;
  if (StartsWith(cleaned, kPythonStartToken)) {
    cleaned = cleaned.substr(std::string(kPythonStartToken).size());
    cleaned = RemoveAll(cleaned, kPythonEndToken);
  }
  ExtractedToolCallInformation info = ExtractClean(cleaned);
  // On the plain-content fallback, upstream returns the (possibly stripped)
  // `model_output` it parsed. Mirror that: content is the cleaned text.
  return info;
}

std::optional<DeltaMessage> Llama4PythonicToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // llama4_pythonic_tool_parser.py:132-135 - not a (possibly wrapped) list ->
  // stream delta as plain content.
  if (!StartsWith(current_text, "[") &&
      !StartsWith(current_text, kPythonStartToken)) {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }
  // llama4_pythonic_tool_parser.py:138-142 - strip the leading start marker and
  // a trailing end marker (rfind: everything up to the last <|python_end|>).
  std::string cleaned = current_text;
  if (StartsWith(cleaned, kPythonStartToken)) {
    cleaned = cleaned.substr(std::string(kPythonStartToken).size());
  }
  if (EndsWith(cleaned, kPythonEndToken)) {
    cleaned = cleaned.substr(0, cleaned.rfind(kPythonEndToken));
  }
  return StreamClean(cleaned);
}

}  // namespace vllm::entrypoints::openai
