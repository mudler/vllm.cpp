// Ported from: vllm/v1/engine/detokenizer.py @ e24d1b24
// (free helper functions from vllm/tokenizers/detokenizer_utils.py @
// e24d1b24). Deviations, recorded:
// - FastIncrementalDetokenizer (HF tokenizers' Rust DecodeStream) is not
//   ported — no HF library in C++; SlowIncrementalDetokenizer is the
//   reference algorithm and the only concrete implementation.
// - Text is raw UTF-8 bytes, never lossily re-encoded: where upstream's
//   Python str carries U+FFFD ("�") for invalid byte runs, we emit the raw
//   bytes verbatim. The hold-back check (upstream's `endswith("�")`), the
//   prefix-length arithmetic, and the stop-string hold-back window
//   (stop_buffer_length_) all replicate Python's lossy CHARACTER semantics
//   exactly (see LossyStep in the .cpp). On top of that, GetNextOutputText
//   ALWAYS trims a streamed (not-finished) view to the last complete UTF-8
//   character via ByteOffsetOfLastCompleteUtf8Char, independent of
//   stop_buffer_length_ — so a multi-byte codepoint split across tokens is
//   never streamed half-formed (which would make the server's json::dump
//   reject the delta), even when there are no stop strings
//   (stop_buffer_length_ == 0). Genuinely invalid, non-completable trailing
//   bytes still pass through raw so the stream never stalls. (An earlier
//   byte-based hold-back window/slice was a deviation; now resolved.)
// - DetokenizerRequest carries the detokenization-relevant subset of
//   upstream's EngineCoreRequest + SamplingParams (no prompt_embeds).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vllm/tokenizer/tokenizer.h"

namespace vllm::v1 {

// Detokenization-relevant subset of EngineCoreRequest + SamplingParams
// (upstream constructors receive the whole request object).
struct DetokenizerRequest {
  std::vector<int32_t> prompt_token_ids;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;
  std::vector<std::string> stop;  // SamplingParams.stop (already list-form)
  bool include_stop_str_in_output = false;
  size_t min_tokens = 0;
};

// Byte offset at which a streamed view of `s` must end so it never cuts a
// multi-byte UTF-8 character in half. Returns s.size() when `s` already ends on
// a complete character boundary; otherwise the offset of the START of a
// truncated-but-COMPLETABLE trailing sequence (a lead byte whose continuation
// bytes have not all arrived yet), so those bytes are held back until a later
// token completes them. Genuinely invalid / non-completable trailing bytes (a
// stray continuation byte, an over-full sequence, an invalid lead) are NOT held
// back — they pass through as raw bytes (the recorded raw-bytes deviation) so
// the stream never stalls. This is the deviation-specific guard that keeps
// streamed deltas structurally valid UTF-8 regardless of stop_buffer_length_.
size_t ByteOffsetOfLastCompleteUtf8Char(std::string_view s);

// check_stop_strings: checks if any stop strings are matched and returns
// {stop_string, truncate_to} if so. `truncate_to` is the byte length to
// which output_text should be truncated, or -1 for no truncation.
std::optional<std::pair<std::string, int64_t>> CheckStopStrings(
    std::string_view output_text, size_t new_char_count,
    const std::vector<std::string>& stop, bool include_in_output);

// detokenizer_utils.convert_prompt_ids_to_tokens: converts only the prompt
// tail needed to seed incremental detokenization.
struct PromptTokens {
  std::vector<std::string> tokens;
  size_t prefix_offset = 0;
  size_t read_offset = 0;
};
PromptTokens ConvertPromptIdsToTokens(const tok::Tokenizer& tokenizer,
                                      const std::vector<int32_t>& prompt_ids,
                                      bool skip_special_tokens);

// detokenizer_utils.detokenize_incrementally: detokenizes the last id of
// `all_input_ids` given the previous token strings and offsets; returns the
// new token strings, the newly finalized text ("" while the decoded suffix
// still ends in an incomplete UTF-8 sequence), and the updated offsets.
// `prev_tokens == nullptr` reproduces upstream's first-iteration branch.
// `spaces_between_special_tokens` is accepted for signature parity but is
// inert: upstream only consults it on slow (non-fast) HF tokenizers with
// added vocab, and our byte-level BPE family corresponds to is_fast=True.
struct IncrementalDetokenizeResult {
  std::vector<std::string> new_tokens;
  std::string new_text;
  size_t prefix_offset = 0;
  size_t read_offset = 0;
};
IncrementalDetokenizeResult DetokenizeIncrementally(
    const tok::Tokenizer& tokenizer, const std::vector<int32_t>& all_input_ids,
    const std::vector<std::string>* prev_tokens, size_t prefix_offset,
    size_t read_offset, bool skip_special_tokens,
    bool spaces_between_special_tokens);

// No-op base: accumulates token ids without detokenizing (upstream returns
// this when the engine runs without a tokenizer).
class IncrementalDetokenizer {
 public:
  IncrementalDetokenizer() = default;
  virtual ~IncrementalDetokenizer() = default;
  IncrementalDetokenizer(const IncrementalDetokenizer&) = delete;
  IncrementalDetokenizer& operator=(const IncrementalDetokenizer&) = delete;

  virtual std::vector<int32_t> OutputTokenIds() const { return token_ids_; }
  virtual size_t NumOutputTokens() const { return token_ids_.size(); }

  // Appends the new ids (and, in subclasses, detokenizes them incrementally
  // and evaluates stop criteria). Returns the matched stop string, if any.
  virtual std::optional<std::string> Update(
      const std::vector<int32_t>& new_token_ids, bool stop_terminated);

  // If `delta` is true, only new text since the last call is returned.
  virtual std::string GetNextOutputText(bool finished, bool delta);

  // from_new_request: tokenizer == nullptr => skipping detokenization.
  static std::unique_ptr<IncrementalDetokenizer> FromNewRequest(
      const tok::Tokenizer* tokenizer, DetokenizerRequest request);

 protected:
  std::vector<int32_t> token_ids_;
};

// Stop-string bookkeeping shared by concrete detokenizers.
class BaseIncrementalDetokenizer : public IncrementalDetokenizer {
 public:
  std::optional<std::string> Update(const std::vector<int32_t>& new_token_ids,
                                    bool stop_terminated) override;
  std::string GetNextOutputText(bool finished, bool delta) override;
  const std::string& OutputText() const { return output_text_; }

 protected:
  explicit BaseIncrementalDetokenizer(const DetokenizerRequest& request);
  // Detokenizes one token id; returns the newly finalized text.
  virtual std::string DecodeNext(int32_t next_token_id) = 0;

  std::vector<std::string> stop_;
  size_t min_tokens_ = 0;
  bool include_stop_str_in_output_ = false;
  // Number of chars to hold back when stop strings are to be excluded from
  // streamed output. Counted in lossy Python-str CHARS (upstream's
  // `max(len(s) for s in stop) - 1`), not bytes.
  size_t stop_buffer_length_ = 0;
  // Byte offset into output_text_ of the last streamed delta's end; always
  // lands on a lossy-char boundary.
  size_t last_output_text_offset_ = 0;
  std::string output_text_;
};

// The slow python-based incremental detokenization algorithm
// (prefix_offset/read_offset over convert_ids_to_tokens +
// convert_tokens_to_string) — the reference implementation here.
class SlowIncrementalDetokenizer : public BaseIncrementalDetokenizer {
 public:
  SlowIncrementalDetokenizer(const tok::Tokenizer& tokenizer,
                             DetokenizerRequest request);

  std::vector<int32_t> OutputTokenIds() const override;
  size_t NumOutputTokens() const override;

 protected:
  std::string DecodeNext(int32_t next_token_id) override;

 private:
  const tok::Tokenizer& tokenizer_;
  size_t prompt_len_ = 0;
  // Metadata for incremental detokenization.
  std::vector<std::string> tokens_;
  size_t prefix_offset_ = 0;
  size_t read_offset_ = 0;
  bool skip_special_tokens_ = true;
  bool spaces_between_special_tokens_ = true;
};

}  // namespace vllm::v1
