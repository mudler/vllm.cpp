// See detect.h. ORIGINAL packaging-layer component (no upstream mirror).
#include "vllm/entrypoints/openai/tool_parsers/detect.h"

#include <string>

namespace vllm::entrypoints::openai {

namespace {

// The ORDERED marker table. Each row maps a chat-template tell-tale substring to
// the parser whose dialect that template emits, mirroring how llama.cpp's
// common/chat.cpp sniffs a template to select a tool-call format.
//
// ORDER MATTERS: DetectToolParser returns the FIRST matching row, so a more
// SPECIFIC marker MUST precede a more GENERIC one that it is a superset of;
// otherwise the generic row would shadow the specific model. A new parser port
// adds exactly one row here (and, if its marker is generic, places it last).
//
// Marker sources: each row was recommended by the family's port (see the
// parser file headers), with these collision notes:
//   - longcat's wrapper never contains a bare "<tool_call>" substring, so the
//     hermes row cannot shadow it, but the specific marker still goes first.
//   - The DeepSeek begin marker is shared by v3 and v31 templates; detection
//     resolves to deepseek_v3, and a v31 model selects "deepseek_v31"
//     explicitly via the tool_parser option.
//   - granite4 is marker-identical to hermes (<tool_call> blocks) and
//     pythonic's only tell (a bare leading "[") is far too generic for
//     template sniffing: both are EXPLICIT-ONLY families (tool_parser option),
//     deliberately absent from this table.
//   - "<|python_tag|>" (llama3/4 json) is the least discriminating marker that
//     is still template-sniffable, so it sits just above the hermes fallback.
constexpr ToolParserMarker kToolParserMarkers[] = {
    {"longcat", "<longcat_tool_call>"},
    {"deepseek_v3", "<｜tool▁calls▁begin｜>"},
    {"mistral", "[TOOL_CALLS]"},
    {"granite-20b-fc", "<function_call>"},
    {"granite", "<|tool_call|>"},
    {"llama4_pythonic", "<|python_start|>"},
    {"llama3_json", "<|python_tag|>"},
    {"hermes", "<tool_call>"},
};

constexpr std::size_t kToolParserMarkerCount =
    sizeof(kToolParserMarkers) / sizeof(kToolParserMarkers[0]);

}  // namespace

std::string DetectToolParser(const std::string& chat_template,
                             const ToolParserMarker* table, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (table[i].template_marker != nullptr &&
        chat_template.find(table[i].template_marker) != std::string::npos) {
      return table[i].parser;
    }
  }
  // No marker matched: fall back to hermes, the most widely trained dialect.
  // Its parser is a no-op on output that carries no <tool_call> block, so this
  // is safe even when the model does not actually emit tool calls.
  return "hermes";
}

std::string DetectToolParser(const std::string& chat_template) {
  return DetectToolParser(chat_template, kToolParserMarkers,
                          kToolParserMarkerCount);
}

const ToolParserMarker* ToolParserMarkerTable(std::size_t* out_count) {
  if (out_count != nullptr) *out_count = kToolParserMarkerCount;
  return kToolParserMarkers;
}

}  // namespace vllm::entrypoints::openai
