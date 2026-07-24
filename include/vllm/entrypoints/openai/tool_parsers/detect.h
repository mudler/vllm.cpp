// Chat-template tool-parser AUTO-detection for the ABI v4 chat entry points.
//
// ORIGINAL packaging-layer component (no upstream mirror): upstream vLLM takes
// the tool parser from an explicit `--tool-call-parser` CLI flag and never
// auto-detects. The C ABI has no such flag surface, so when the caller does not
// name a parser we sniff the model's chat template for a format-specific marker,
// the same way llama.cpp's common/chat.cpp (common_chat_templates_apply /
// common_chat_format_detect) picks a tool-call format by looking for the
// template's tell-tale substrings. Credit to that llama.cpp approach.
//
// A new parser port adds exactly ONE row to the static marker table (see
// detect.cpp). Because DetectToolParser returns the FIRST row that matches, the
// table is ORDERED: more-specific markers must precede more-generic ones.
#ifndef VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_DETECT_H_
#define VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_DETECT_H_

#include <cstddef>
#include <string>

namespace vllm::entrypoints::openai {

// One row of the marker table: if `template_marker` occurs as a substring in a
// model's chat template, that model speaks `parser`'s tool-call dialect. Both
// pointers are static string literals (the table has static storage duration).
struct ToolParserMarker {
  const char* parser;
  const char* template_marker;
};

// Auto-detect the tool-call parser for a model from its chat template: return
// the `parser` of the FIRST built-in marker row whose `template_marker` occurs
// in `chat_template`, else "hermes" (the only registered family today, and the
// safe default because its <tool_call> surface is what the structural-tag
// constraint already steers on). An empty template yields "hermes".
std::string DetectToolParser(const std::string& chat_template);

// Detection against a caller-supplied table. DetectToolParser above forwards to
// this with the built-in table; tests use it to prove the first-match ordering
// semantics with a synthetic multi-row table (the built-in table has a single
// family today). Returns "hermes" when no row matches.
std::string DetectToolParser(const std::string& chat_template,
                             const ToolParserMarker* table, std::size_t count);

// The built-in marker table, exposed for table-invariant tests (every row names
// a registered parser, every marker is non-empty). `*out_count` receives the row
// count when non-null. The returned pointer has static storage duration.
const ToolParserMarker* ToolParserMarkerTable(std::size_t* out_count);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_DETECT_H_
