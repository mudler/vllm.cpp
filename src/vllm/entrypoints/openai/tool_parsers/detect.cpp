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
// Today the only registered family is Hermes/Qwen, whose templates wrap each
// call in a <tool_call> block. That same surface is what the structural-tag
// constraint (get_hermes_structural_tag) triggers on, so this row keeps
// detection and constraint in agreement.
constexpr ToolParserMarker kToolParserMarkers[] = {
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
  // No marker matched: fall back to the sole registered family. Its parser is a
  // no-op on output that carries no <tool_call> block, so this is safe even when
  // the model does not actually emit tool calls.
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
