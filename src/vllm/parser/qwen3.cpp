// Ported from: vllm/parser/qwen3.py:201 (Qwen3Parser) @ 555967922.
#include "vllm/parser/qwen3.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace vllm::parser {

namespace {

// The config terminal literal for `name`, or nullopt when the family does not
// declare it. Keeps the two overrides wrapper-token agnostic (qwen3 vs the
// seed_oss <seed:...> spelling of the same grammar).
std::optional<std::string> terminal_of(const engine::ParserEngineConfig& config,
                                       const std::string& name) {
  auto it = config.terminals.find(name);
  if (it == config.terminals.end()) return std::nullopt;
  return it->second;
}

}  // namespace

// qwen3.py:247:
//     if not self.thinking_enabled:
//         return None, model_output
//     return super().extract_reasoning(model_output, request)
std::pair<std::optional<std::string>, std::optional<std::string>>
Qwen3Parser::extract_reasoning(const std::string& model_output,
                               const engine::ParserRequest& request) {
  if (!thinking_enabled_) return {std::nullopt, model_output};
  return engine::ParserEngine::extract_reasoning(model_output, request);
}

// qwen3.py:256. The base rule first; then scan for a tool-call marker that has
// no closing marker after it. Upstream walks the token ids backwards and
// returns False as soon as it meets the reasoning-start token, so a <think>
// occurring AFTER the tool-call marker wins — the text form compares positions.
bool Qwen3Parser::is_reasoning_end(const std::string& text) const {
  if (engine::ParserEngine::is_reasoning_end(text)) return true;

  const std::optional<std::string> tool_start =
      terminal_of(config_, "TOOL_START");
  if (!tool_start || tool_start->empty()) return false;
  const std::optional<std::string> tool_end = terminal_of(config_, "TOOL_END");
  const std::optional<std::string> think_start = reasoning_start_str();
  const std::size_t ps =
      think_start ? text.rfind(*think_start) : std::string::npos;

  std::size_t p = text.rfind(*tool_start);
  while (p != std::string::npos) {
    // A reasoning-start marker after this tool-call marker is reached first by
    // the backwards scan: reasoning has re-opened (qwen3.py:265-268).
    if (ps != std::string::npos && ps > p) return false;
    const bool paired =
        tool_end && text.find(*tool_end, p + tool_start->size()) !=
                        std::string::npos;
    if (!paired) return true;  // qwen3.py:275
    if (p == 0) break;
    p = text.rfind(*tool_start, p - 1);
  }
  return false;
}

}  // namespace vllm::parser
