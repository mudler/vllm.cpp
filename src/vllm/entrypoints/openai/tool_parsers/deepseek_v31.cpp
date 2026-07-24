// Ported from: vllm/tool_parsers/deepseekv31_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v31.h"

#include <algorithm>
#include <cstddef>
#include <regex>
#include <string>

namespace vllm::entrypoints::openai {

namespace {

// tool_parsers/utils.py:42 (partial_tag_overlap): longest prefix of `tag` that
// is a suffix of `text`. Used to hold back a partial <｜tool▁call▁end｜> on the
// unterminated streaming tail (V3.1 args run straight to the end marker with no
// intervening fence, so a mid-byte end marker would otherwise leak into args).
std::size_t PartialTagOverlap(const std::string& text, const std::string& tag) {
  const std::size_t max_check = std::min(tag.size() - 1, text.size());
  for (std::size_t k = max_check; k >= 1; --k) {
    if (text.compare(text.size() - k, k, tag, 0, k) == 0) return k;
  }
  return 0;
}

}  // namespace

// Out-of-line anchor (key function) for the vtable.
DeepSeekV31ToolParser::~DeepSeekV31ToolParser() = default;

// deepseekv31_tool_parser.py:46-48 — non-greedy name/args, no fence, no type.
const std::regex& DeepSeekV31ToolParser::tool_call_pattern() const {
  static const std::regex re(
      R"(<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>)");
  return re;
}

// deepseekv31_tool_parser.py:98-107 — type is always "function"; group 1 = name,
// group 2 = the raw JSON arguments string (stored verbatim).
ToolCall DeepSeekV31ToolParser::tool_call_from_match(
    const std::smatch& match) const {
  ToolCall tc;
  tc.id = make_tool_call_id();
  tc.type = "function";
  tc.function.name = match[1].str();
  tc.function.arguments = match[2].str();
  return tc;
}

// Streaming per-region parse. `region` = text between <｜tool▁call▁begin｜> and
// its <｜tool▁call▁end｜> (or the tail). Layout: NAME<｜tool▁sep｜>ARGS.
DeepSeekV3ToolParser::ParsedCall DeepSeekV31ToolParser::parse_region(
    const std::string& region) const {
  ParsedCall out;
  const std::string sep = kToolSepToken;
  const std::size_t sep_pos = region.find(sep);
  if (sep_pos == std::string::npos) return out;  // name not yet available

  out.name = region.substr(0, sep_pos);
  std::string args = region.substr(sep_pos + sep.size());
  // Hold back a partial trailing <｜tool▁call▁end｜> (tail regions only; a
  // complete region already excludes the end marker).
  const std::size_t ov = PartialTagOverlap(args, kToolCallEndToken);
  if (ov) args = args.substr(0, args.size() - ov);
  out.arguments = args;
  return out;
}

}  // namespace vllm::entrypoints::openai
