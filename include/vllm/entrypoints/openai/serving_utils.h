// vllm.cpp original (no direct upstream mirror — a serving-boundary helper).
//
// UTF-8 SANITIZATION at the serving boundary (M3.1 Task 4).
//
// Our incremental detokenizer (src/vllm/v1/engine/detokenizer.cpp) keeps the
// RAW decoded bytes, NOT upstream's Python-str (errors="replace") lossy string.
// A genuinely-invalid multibyte run (e.g. a lone 0xFF, or a 4-byte lead split
// across DELTA chunks) can therefore persist verbatim in a RequestOutput's
// text. nlohmann::json::dump() rejects invalid UTF-8 (throws type_error.316),
// which — inside a serving handler that dumps the response/chunk — surfaces as
// an HTTP 500.
//
// SanitizeUtf8 replays the SAME lossy-decode arithmetic as the detokenizer's
// LossyStep (detokenizer.cpp) to reproduce upstream's `str` semantics: every
// maximal invalid/truncated UTF-8 subpart becomes exactly one U+FFFD
// ("\xEF\xBF\xBD", the Unicode REPLACEMENT CHARACTER); valid text (including an
// already-present literal U+FFFD) is left byte-for-byte unchanged. Applying it
// to the text fields before json serialization makes every response dump()-safe
// and matches what the OpenAI SDK / LocalAI see from upstream vLLM.
#ifndef VLLM_ENTRYPOINTS_OPENAI_SERVING_UTILS_H_
#define VLLM_ENTRYPOINTS_OPENAI_SERVING_UTILS_H_

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"

namespace vllm::entrypoints::openai {

// Returns `s` with every maximal invalid/truncated UTF-8 subpart replaced by a
// single U+FFFD. Valid UTF-8 (and literal U+FFFD) passes through unchanged. The
// result is always well-formed UTF-8 and therefore safe for nlohmann json dump.
std::string SanitizeUtf8(const std::string& s);

// Ported from: vllm/entrypoints/serve/utils/api_utils.py:276-289
// (should_include_usage). Force mode enables final and continuous usage;
// request-level continuous stats never take effect without include_usage.
struct StreamUsageSelection {
  bool include_usage = false;
  bool include_continuous_usage = false;
};
StreamUsageSelection ShouldIncludeUsage(
    const std::optional<StreamOptions>& stream_options,
    bool enable_force_include_usage);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_UTILS_H_
