// Unit tests for the ABI v4 tool-parser AUTO-detector
// (src/vllm/entrypoints/openai/tool_parsers/detect.*): marker-hit selection, the
// no-marker fallback, first-match ordering semantics (proven with a synthetic
// table since only one family is registered today), and the built-in table's
// invariants. Pure string-level tests — no engine, no disk model.
#include "vllm/entrypoints/openai/tool_parsers/detect.h"

#include <doctest/doctest.h>

#include <cstddef>
#include <string>

#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

using vllm::entrypoints::openai::DetectToolParser;
using vllm::entrypoints::openai::get_tool_parser;
using vllm::entrypoints::openai::ToolParserMarker;
using vllm::entrypoints::openai::ToolParserMarkerTable;

TEST_CASE("detect: the hermes <tool_call> marker selects hermes") {
  // A minimal template that wraps a call in the Hermes/Qwen surface.
  const std::string tmpl =
      "{% for m in messages %}{{ m.role }}{% endfor %}"
      "<tool_call>{\"name\": \"f\"}</tool_call>";
  CHECK(DetectToolParser(tmpl) == "hermes");
}

TEST_CASE("detect: a template with no known marker falls back to hermes") {
  const std::string tmpl =
      "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}";
  CHECK(DetectToolParser(tmpl) == "hermes");
  // The empty template is the "no template resolved" shape.
  CHECK(DetectToolParser("") == "hermes");
}

TEST_CASE("detect: first matching row wins (ordering semantics)") {
  // A synthetic two-row table: the SPECIFIC marker precedes the GENERIC one it
  // is a superset of. DetectToolParser must return the first row that matches,
  // so a template carrying BOTH markers resolves to the specific parser.
  const ToolParserMarker table[] = {
      {"specific", "<specific_call>"},
      {"generic", "<call>"},
  };
  const std::size_t n = sizeof(table) / sizeof(table[0]);

  // Contains both markers -> the earlier (specific) row wins.
  CHECK(DetectToolParser("prefix <specific_call> and <call> suffix", table, n) ==
        "specific");
  // Only the generic marker -> the second row.
  CHECK(DetectToolParser("only <call> here", table, n) == "generic");
  // Neither marker -> the hermes fallback (not a row in this table).
  CHECK(DetectToolParser("nothing to see", table, n) == "hermes");
}

TEST_CASE("detect: the built-in marker table holds its invariants") {
  std::size_t n = 0;
  const ToolParserMarker* table = ToolParserMarkerTable(&n);
  REQUIRE(table != nullptr);
  REQUIRE(n >= 1);  // at least the hermes row.
  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    // Every row names a NON-EMPTY parser + marker...
    REQUIRE(table[i].parser != nullptr);
    REQUIRE(table[i].template_marker != nullptr);
    CHECK(std::string(table[i].parser).size() > 0);
    CHECK(std::string(table[i].template_marker).size() > 0);
    // ...and every named parser is actually REGISTERED, so detection can never
    // hand EnsureChatServing a name get_tool_parser rejects.
    CHECK(get_tool_parser(table[i].parser) != nullptr);
  }
}

TEST_CASE("detect: null count pointer is tolerated") {
  CHECK(ToolParserMarkerTable(nullptr) != nullptr);
}
