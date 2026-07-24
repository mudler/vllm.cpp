// Ported from: vllm/tool_parsers/utils.py @ e24d1b24
//
// The subset of the shared tool-parser utilities the GRANITE family needs:
//   consume_space, find_common_prefix, is_complete_json, partial_json_loads.
// Plus one factored helper (granite_stream_emit) that carries the streaming
// argument-diff state machine that granite_tool_parser.py:148-249 and
// granite_20b_fc_tool_parser.py:159-269 duplicate VERBATIM - the two upstream
// parsers share this body byte-for-byte, so we hoist it here rather than copy it
// into both .cpp files.
//
// DEVIATIONS from upstream shape (all documented per function below):
//   - upstream partial_json_loads defers to the third-party `partial_json_parser`
//     package (a tolerant JSON completer) and falls back to json.JSONDecoder.
//     raw_decode on "Extra data". We have no such dependency, so this file
//     carries a small hand-written tolerant JSON parser (TolerantParser in the
//     .cpp) that reproduces the two behaviours the granite streaming path relies
//     on: (a) completing a truncated JSON document, honouring the Allow.STR flag
//     (whether an unterminated string value may be emitted as a partial value),
//     and (b) returning the consumed length so a complete value followed by
//     extra data yields the raw_decode end index.
//   - upstream's `flags: Allow` bitmask collapses to a single `allow_partial_str`
//     bool: granite only ever toggles Allow.STR (Allow.ALL vs Allow.ALL & ~STR).
//   - we parse into nlohmann::ordered_json (NOT the default nlohmann::json, whose
//     std::map object backing SORTS keys): CPython's json.dumps preserves dict
//     INSERTION order, and the streaming argument-diff (find_common_prefix over
//     successive dumps) only reconstructs correctly when key order is stable.
#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"

namespace vllm::entrypoints::openai {

// utils.py:196 (consume_space). Advance past ASCII whitespace from index `i`.
std::size_t consume_space(std::size_t i, const std::string& s);

// utils.py:52 (find_common_prefix). The shared leading prefix of a and b.
std::string find_common_prefix(const std::string& a, const std::string& b);

// utils.py:135 (is_complete_json). Whether `s` is a complete, valid JSON doc.
bool is_complete_json(const std::string& s);

// Raised when partial_json_loads cannot parse even a partial value yet - the
// text-only analogue of partial_json_parser.core.exceptions.MalformedJSON, which
// the granite streaming loops catch to mean "not enough tokens yet".
struct MalformedPartialJson : std::runtime_error {
  MalformedPartialJson() : std::runtime_error("malformed partial JSON") {}
};

// utils.py:120 (partial_json_loads). Parse the first JSON value out of `s`,
// completing it when the document is truncated. Returns {value, end_idx} where
// end_idx is the number of characters consumed (== s.size() for a pure partial;
// the value's end for a complete value trailed by extra data). `allow_partial_str`
// mirrors the Allow.STR flag: when false an unterminated string value is dropped
// (its enclosing key/value pair vanishes) rather than emitted partially. Throws
// MalformedPartialJson when nothing parseable is present.
std::pair<nlohmann::ordered_json, std::size_t> partial_json_loads(
    const std::string& s, bool allow_partial_str);

// The streaming argument-diff emitter shared by GraniteToolParser and
// Granite20bFCToolParser (granite_tool_parser.py:154-249). Given the parser's
// carried streaming state (by reference) and the freshly parsed `tool_call_arr`
// (+ per-entry `is_complete` flags), return the DeltaMessage to emit (name-first
// then argument diffs) or nullopt. Mutates the state members exactly as upstream:
// advances current_tool_id when a new array element appears, flips
// current_tool_name_sent, grows streamed_args_for_tool, and (except on the
// new-tool early return) sets prev_tool_call_arr = tool_call_arr.
std::optional<DeltaMessage> granite_stream_emit(
    std::vector<nlohmann::ordered_json>& prev_tool_call_arr, int& current_tool_id,
    bool& current_tool_name_sent, std::vector<std::string>& streamed_args_for_tool,
    const std::vector<nlohmann::ordered_json>& tool_call_arr,
    const std::vector<bool>& is_complete);

}  // namespace vllm::entrypoints::openai
