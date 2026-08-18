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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/logprobs.h"  // vllm::SampleLogprobs / Logprob
#include "vllm/outputs.h"    // vllm::CompletionOutput (best_of ranking)

namespace vllm::entrypoints::openai {

// Returns `s` with every maximal invalid/truncated UTF-8 subpart replaced by a
// single U+FFFD. Valid UTF-8 (and literal U+FFFD) passes through unchanged. The
// result is always well-formed UTF-8 and therefore safe for nlohmann json dump.
std::string SanitizeUtf8(const std::string& s);

// SSE comment keepalives while the stream is silent (long prefill / TTFT).
// VT_SERVER_SSE_PING_S: seconds between pings; DEFAULT 0 — OFF (#931); <=0,
// unset and unparsable all disable; a positive value is capped at 600.
// Off is the default because vLLM emits no comment frame from any streaming
// endpoint, and vLLM's own `vllm bench serve` marks a request FAILED when one
// arrives ahead of its first data frame. The full reason, and what it cost, is
// at the definition in serving_utils.cpp.
int SsePingIntervalSec();
inline constexpr const char kSsePingFrame[] = ":\n\n";

// Shared chat + completion framing after a timed collector wait.
// - ready set  -> move into `out`, return true  (caller emits a data frame)
// - ready empty -> set `chunk` to kSsePingFrame ONLY, return false
//   (caller emits that standalone comment; never concatenate with data).
// Used by ChatSseStream / CompletionSseStream WaitOutput.
bool AssignSseWaitResult(std::optional<vllm::RequestOutput> ready,
                         vllm::RequestOutput& out, std::string& chunk);

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

// Ported from: vllm/entrypoints/openai/completion/serving.py:652-741
// (_create_completion_logprobs). Assemble the /v1/completions logprobs payload
// from a run of generated (or echoed) tokens and their per-position SampleLogprobs.
// `num_output_top_logprobs` == request.logprobs (keeps N+1 entries per position).
// `initial_text_offset` seeds the running character offset (nonzero when the
// echoed prompt precedes the completion). Relies on the LogprobsProcessor's
// per-token decoded_token; falls back to a `token_id:N` placeholder when absent.
CompletionLogProbs BuildCompletionLogProbs(const std::vector<int32_t>& token_ids,
                                           const vllm::SampleLogprobs& top_logprobs,
                                           int num_output_top_logprobs,
                                           int initial_text_offset = 0);

// Ported from: vllm/entrypoints/openai/chat_completion/serving.py:1141-1210
// (_create_chat_logprobs + _get_top_logprobs). `num_output_top_logprobs` ==
// request.top_logprobs (keeps N entries per position, 0-based cutoff; -1 => all).
ChatCompletionLogProbs BuildChatLogprobs(const std::vector<int32_t>& token_ids,
                                         const vllm::SampleLogprobs& top_logprobs,
                                         int num_output_top_logprobs);

// SAMPLE-BEST-OF: rank `outputs` by descending cumulative logprob and keep the
// top `return_n`, RE-INDEXING them 0..return_n-1 (classic OpenAI best_of: the
// engine generated best_of children, the endpoint returns the n best). The sort
// is STABLE (ties keep engine order); a child with no cumulative logprob sorts
// last. INERT — returns `outputs` unchanged — when return_n <= 0 or
// outputs.size() <= return_n (i.e. the default best_of == n path never re-ranks
// or re-indexes, preserving the n>1 child indices byte-for-byte). Callers guard
// the trim on request.best_of so the default path does not even copy.
std::vector<vllm::CompletionOutput> SelectBestOf(
    std::vector<vllm::CompletionOutput> outputs, int return_n);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_UTILS_H_
