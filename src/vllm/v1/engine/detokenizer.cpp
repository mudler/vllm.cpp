// Ported from: vllm/v1/engine/detokenizer.py @ e24d1b24
// (free helper functions from vllm/tokenizers/detokenizer_utils.py @
// e24d1b24). See the header for the recorded deviations (no fast path; raw
// bytes instead of lossy U+FFFD substitution).
#include "vllm/v1/engine/detokenizer.h"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "vllm/tokenizer/bpe.h"

namespace vllm::v1 {
namespace {

// detokenizer_utils.INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET: 5 is an
// arbitrary value that should work for all tokenizers (bigger = more
// conservative).
constexpr size_t kInitialIncrementalDetokenizationOffset = 5;

// Decodes one UTF-8 codepoint at `pos`; writes it to `cp` and returns its
// byte length. Invalid bytes decode as a 1-byte U+FFFD (which is outside the
// byte-level bijection image, so the containing token falls back to verbatim
// pass-through in ConvertTokensToString).
size_t DecodeUtf8(std::string_view s, size_t pos, uint32_t& cp) {
  const uint8_t b0 = static_cast<uint8_t>(s[pos]);
  size_t len;
  if (b0 < 0x80) {
    cp = b0;
    return 1;
  } else if ((b0 & 0xE0) == 0xC0) {
    len = 2;
    cp = b0 & 0x1Fu;
  } else if ((b0 & 0xF0) == 0xE0) {
    len = 3;
    cp = b0 & 0x0Fu;
  } else if ((b0 & 0xF8) == 0xF0) {
    len = 4;
    cp = b0 & 0x07u;
  } else {
    cp = 0xFFFD;
    return 1;
  }
  if (pos + len > s.size()) {
    cp = 0xFFFD;
    return 1;
  }
  for (size_t i = 1; i < len; ++i) {
    const uint8_t b = static_cast<uint8_t>(s[pos + i]);
    if ((b & 0xC0) != 0x80) {
      cp = 0xFFFD;
      return 1;
    }
    cp = (cp << 6) | (b & 0x3Fu);
  }
  return len;
}

// TokenizerLike.convert_ids_to_tokens as consumed by detokenizer_utils:
// special added tokens are dropped when skip_special_tokens, and unknown ids
// become "" (upstream's _replace_none_with_empty guard).
void ConvertIdsToTokens(const tok::Tokenizer& tokenizer,
                        const int32_t* ids, size_t count,
                        bool skip_special_tokens,
                        std::vector<std::string>& out) {
  for (size_t i = 0; i < count; ++i) {
    const int32_t id = ids[i];
    if (skip_special_tokens && tokenizer.IsSpecial(id)) continue;
    out.push_back(tokenizer.HasToken(id) ? tokenizer.TokenText(id)
                                         : std::string());
  }
}

// TokenizerLike.convert_tokens_to_string for the byte-level BPE family:
// HF fast tokenizers route this through the ByteLevel decoder, whose byte-map
// reversal is per-TOKEN all-or-nothing: a token whose every codepoint is in
// the bytes_to_unicode image is mapped back to its bytes, while a token
// containing ANY codepoint outside the image (added-token literal content)
// passes through verbatim as its raw UTF-8 text (tokenizers' ByteLevel
// decode_chain: try_fold over CHAR_BYTES, unwrap_or the token's own bytes).
std::string ConvertTokensToString(const std::vector<std::string>& tokens,
                                  size_t begin, size_t end) {
  std::string out;
  std::string unmapped;  // per-token byte-map reversal attempt
  for (size_t t = begin; t < end; ++t) {
    const std::string& token = tokens[t];
    unmapped.clear();
    bool in_image = true;
    size_t pos = 0;
    while (pos < token.size()) {
      uint32_t cp = 0;
      pos += DecodeUtf8(token, pos, cp);
      const int32_t byte = tok::UnicodeToByte(cp);
      if (byte < 0) {
        in_image = false;
        break;
      }
      unmapped.push_back(static_cast<char>(byte));
    }
    out += in_image ? unmapped : token;
  }
  return out;
}

// --- Lossy (errors="replace") string emulation -----------------------------
// Upstream's texts are Python strs: convert_tokens_to_string lossily decodes
// the raw bytes, turning every maximal invalid/truncated UTF-8 subpart into
// one U+FFFD ("�"). The length comparison, the `.endswith("�")` hold-back
// check, and the `new_text[len(prefix_text):]` slice all operate on those
// lossy CHARACTERS. We keep the raw bytes (byte-exact output is required
// here) but replicate the character arithmetic exactly.

// One lossy decoding step at `pos`: consumes one valid UTF-8 character or
// one maximal subpart of an invalid sequence (= one "�"). Returns the bytes
// consumed; `replacement` reports whether this character is a "�" (invalid
// subpart or a literal U+FFFD).
size_t LossyStep(std::string_view s, size_t pos, bool& replacement) {
  const uint8_t b0 = static_cast<uint8_t>(s[pos]);
  if (b0 < 0x80) {
    replacement = false;
    return 1;
  }
  size_t need;         // continuation bytes required by this lead
  uint8_t lo = 0x80;   // valid range for the FIRST continuation byte
  uint8_t hi = 0xBF;   // (encodes the overlong/surrogate/range rules)
  if (b0 >= 0xC2 && b0 <= 0xDF) {
    need = 1;
  } else if (b0 == 0xE0) {
    need = 2;
    lo = 0xA0;
  } else if (b0 >= 0xE1 && b0 <= 0xEC) {
    need = 2;
  } else if (b0 == 0xED) {
    need = 2;
    hi = 0x9F;
  } else if (b0 >= 0xEE && b0 <= 0xEF) {
    need = 2;
  } else if (b0 == 0xF0) {
    need = 3;
    lo = 0x90;
  } else if (b0 >= 0xF1 && b0 <= 0xF3) {
    need = 3;
  } else if (b0 == 0xF4) {
    need = 3;
    hi = 0x8F;
  } else {
    replacement = true;  // stray continuation byte or invalid lead
    return 1;
  }
  size_t len = 1;
  for (size_t i = 0; i < need; ++i) {
    if (pos + len >= s.size()) break;
    const uint8_t b = static_cast<uint8_t>(s[pos + len]);
    if (b < (i == 0 ? lo : uint8_t{0x80}) ||
        b > (i == 0 ? hi : uint8_t{0xBF})) {
      break;
    }
    ++len;
  }
  if (len != need + 1) {
    replacement = true;  // truncated sequence: its maximal subpart is one �
    return len;
  }
  // Valid character; a literal U+FFFD (EF BF BD) still equals "�".
  replacement = b0 == 0xEF && static_cast<uint8_t>(s[pos + 1]) == 0xBF &&
                static_cast<uint8_t>(s[pos + 2]) == 0xBD;
  return len;
}

struct LossyInfo {
  size_t char_count = 0;               // len() of the lossy Python str
  bool ends_with_replacement = false;  // str.endswith("�")
};

LossyInfo AnalyzeLossy(std::string_view s) {
  LossyInfo info;
  size_t pos = 0;
  while (pos < s.size()) {
    bool replacement = false;
    pos += LossyStep(s, pos, replacement);
    ++info.char_count;
    info.ends_with_replacement = replacement;
  }
  return info;
}

// Byte offset just after the first `char_count` lossy characters of `s`
// (the byte-level equivalent of the Python slice s[char_count:]).
size_t ByteOffsetOfLossyChars(std::string_view s, size_t char_count) {
  size_t pos = 0;
  for (size_t n = 0; n < char_count && pos < s.size(); ++n) {
    bool replacement = false;
    pos += LossyStep(s, pos, replacement);
  }
  return pos;
}

// Byte offset of the boundary `char_count` lossy characters before the end
// of `s` (the byte-level equivalent of the Python slices s[:-char_count] /
// s[:len(s) - char_count]). Walks backwards one lossy character at a time.
// This is well-defined because in LossyStep's maximal-subpart segmentation a
// character never contains a non-continuation byte after its first byte, so
// every non-continuation byte starts a character: the character ending at
// `end` is either the nearest non-continuation byte within 4 bytes whose
// maximal subpart reaches exactly `end`, or the single byte at `end - 1`
// (ASCII, or a stray/leftover continuation byte counting as one "�").
size_t ByteOffsetBeforeLossyChars(std::string_view s, size_t char_count) {
  size_t end = s.size();
  for (size_t n = 0; n < char_count && end > 0; ++n) {
    size_t start = end - 1;  // default: one-byte character (ASCII or "�")
    if ((static_cast<uint8_t>(s[end - 1]) & 0xC0) == 0x80) {
      // Continuation byte: scan back (at most a character's reach) for the
      // nearest non-continuation byte and adopt it as the character start iff
      // its maximal subpart ends exactly at `end`.
      const size_t floor = end >= 4 ? end - 4 : 0;
      size_t q = end - 1;
      while (q > floor && (static_cast<uint8_t>(s[q]) & 0xC0) == 0x80) --q;
      if ((static_cast<uint8_t>(s[q]) & 0xC0) != 0x80) {
        bool replacement = false;
        if (q + LossyStep(s.substr(0, end), q, replacement) == end) start = q;
      }
    }
    end = start;
  }
  return end;
}

}  // namespace

std::optional<std::pair<std::string, int64_t>> CheckStopStrings(
    std::string_view output_text, size_t new_char_count,
    const std::vector<std::string>& stop, bool include_in_output) {
  if (new_char_count == 0 || stop.empty()) return std::nullopt;

  for (const std::string& stop_str : stop) {
    const size_t stop_string_len = stop_str.size();
    // Avoid searching already-searched text: upstream starts the find at
    // 1 - new_char_count - stop_string_len (a negative Python index, i.e.
    // from the end and clamped to 0).
    const size_t back = new_char_count + stop_string_len;
    const size_t from =
        output_text.size() + 1 > back ? output_text.size() + 1 - back : 0;
    const size_t found = output_text.find(stop_str, from);
    if (found == std::string_view::npos) continue;
    int64_t stop_index = static_cast<int64_t>(found);

    if (include_in_output) {
      // Truncate to end of stop string.
      stop_index += static_cast<int64_t>(stop_string_len);
      if (stop_index >= static_cast<int64_t>(output_text.size())) {
        // No truncation required.
        return std::make_pair(stop_str, int64_t{-1});
      }
    }

    // Truncate the output text to either the beginning or end of the stop
    // string.
    return std::make_pair(stop_str, stop_index);
  }
  return std::nullopt;
}

PromptTokens ConvertPromptIdsToTokens(const tok::Tokenizer& tokenizer,
                                      const std::vector<int32_t>& prompt_ids,
                                      bool skip_special_tokens) {
  PromptTokens result;
  // We do not need to convert the whole prompt to tokens: offset a little
  // more (+2) in case we have special tokens.
  const size_t tail = kInitialIncrementalDetokenizationOffset + 2;
  const size_t start = prompt_ids.size() > tail ? prompt_ids.size() - tail : 0;
  ConvertIdsToTokens(tokenizer, prompt_ids.data() + start,
                     prompt_ids.size() - start, skip_special_tokens,
                     result.tokens);
  result.read_offset = result.tokens.size();
  result.prefix_offset =
      result.read_offset > kInitialIncrementalDetokenizationOffset
          ? result.read_offset - kInitialIncrementalDetokenizationOffset
          : 0;
  return result;
}

IncrementalDetokenizeResult DetokenizeIncrementally(
    const tok::Tokenizer& tokenizer, const std::vector<int32_t>& all_input_ids,
    const std::vector<std::string>* prev_tokens, size_t prefix_offset,
    size_t read_offset, bool skip_special_tokens,
    bool spaces_between_special_tokens) {
  // Inert for our family — see the header note (upstream only reads it in
  // _convert_tokens_to_string_with_added_encoders, the is_fast=False branch).
  (void)spaces_between_special_tokens;

  const int32_t new_token_id = all_input_ids.back();
  // This is the first iteration for this sequence.
  const bool is_first_iter = prev_tokens == nullptr;
  PromptTokens prompt;
  if (is_first_iter) {
    std::vector<int32_t> prompt_ids(all_input_ids.begin(),
                                    all_input_ids.end() - 1);
    prompt =
        ConvertPromptIdsToTokens(tokenizer, prompt_ids, skip_special_tokens);
    prev_tokens = &prompt.tokens;
    prefix_offset = prompt.prefix_offset;
    read_offset = prompt.read_offset;
  }

  // If the new token id is out of bounds, return an empty string.
  std::vector<std::string> new_tokens;
  if (new_token_id >= 0 && new_token_id < tokenizer.VocabSize()) {
    // Put new_token_id in a list so skip_special_tokens is respected.
    ConvertIdsToTokens(tokenizer, &new_token_id, 1, skip_special_tokens,
                       new_tokens);
  } else {
    new_tokens.emplace_back();
  }
  std::vector<std::string> output_tokens = *prev_tokens;
  output_tokens.insert(output_tokens.end(), new_tokens.begin(),
                       new_tokens.end());

  // If this is the first iteration, return all tokens.
  if (is_first_iter) new_tokens = output_tokens;

  // The prefix text is necessary only to defeat cleanup algorithms in the
  // decode which decide to add a space or not depending on the surrounding
  // ids.
  const std::string prefix_text =
      ConvertTokensToString(output_tokens, prefix_offset, read_offset);
  std::string new_text =
      ConvertTokensToString(output_tokens, prefix_offset, output_tokens.size());

  // Upstream: `if len(new_text) <= len(prefix_text) or new_text.endswith("�")`
  // — a "�" at the end means it's a potential unfinished byte sequence from
  // byte-level tokenization (if it's in the middle, it's probably a real
  // invalid id generated by the model). Lengths and the endswith check are
  // over the lossily decoded str; see AnalyzeLossy.
  const LossyInfo prefix_info = AnalyzeLossy(prefix_text);
  const LossyInfo new_info = AnalyzeLossy(new_text);
  if (new_info.char_count <= prefix_info.char_count ||
      new_info.ends_with_replacement) {
    return {std::move(new_tokens), std::string(), prefix_offset, read_offset};
  }

  // new_text = new_text[len(prefix_text):], sliced at lossy char boundaries.
  new_text.erase(0, ByteOffsetOfLossyChars(new_text, prefix_info.char_count));
  return {std::move(new_tokens), std::move(new_text), read_offset,
          output_tokens.size()};
}

std::optional<std::string> IncrementalDetokenizer::Update(
    const std::vector<int32_t>& new_token_ids, bool stop_terminated) {
  (void)stop_terminated;
  token_ids_.insert(token_ids_.end(), new_token_ids.begin(),
                    new_token_ids.end());
  return std::nullopt;
}

std::string IncrementalDetokenizer::GetNextOutputText(bool finished,
                                                      bool delta) {
  (void)finished;
  (void)delta;
  return std::string();
}

std::unique_ptr<IncrementalDetokenizer> IncrementalDetokenizer::FromNewRequest(
    const tok::Tokenizer* tokenizer, DetokenizerRequest request) {
  if (tokenizer == nullptr) {
    // No tokenizer => skipping detokenization.
    return std::make_unique<IncrementalDetokenizer>();
  }
  // No FastIncrementalDetokenizer in C++ (needs HF tokenizers' DecodeStream);
  // the slow python-based incremental detokenization is the reference.
  return std::make_unique<SlowIncrementalDetokenizer>(*tokenizer,
                                                      std::move(request));
}

BaseIncrementalDetokenizer::BaseIncrementalDetokenizer(
    const DetokenizerRequest& request)
    : stop_(request.stop),
      min_tokens_(request.min_tokens),
      include_stop_str_in_output_(request.include_stop_str_in_output) {
  // Number of chars to hold back when stop strings are to be excluded from
  // streamed output. Upstream's `max(len(s) for s in stop) - 1` counts
  // Python str CHARS, not bytes; measure stop strings the same lossy way.
  if (!stop_.empty() && !include_stop_str_in_output_) {
    size_t max_len = 0;
    for (const std::string& s : stop_) {
      max_len = std::max(max_len, AnalyzeLossy(s).char_count);
    }
    stop_buffer_length_ = max_len > 0 ? max_len - 1 : 0;
  } else {
    stop_buffer_length_ = 0;
  }
}

std::optional<std::string> BaseIncrementalDetokenizer::Update(
    const std::vector<int32_t>& new_token_ids, bool stop_terminated) {
  // 1) Detokenize the new token ids incrementally.
  // 2) Evaluate stop criteria; return the matched stop string, if any.
  if (new_token_ids.empty()) {
    // Skip detokenization if no new token ids.
    return std::nullopt;
  }

  // If stop-terminated, exclude the last token from detokenization based on
  // the include_stop_str_in_output parameter.
  const bool skip_last = stop_terminated && !include_stop_str_in_output_;
  const size_t detok_count =
      skip_last ? new_token_ids.size() - 1 : new_token_ids.size();

  size_t stop_check_offset = output_text_.size();
  for (size_t i = 0; i < detok_count; ++i) {
    token_ids_.push_back(new_token_ids[i]);
    output_text_ += DecodeNext(new_token_ids[i]);
    // Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
    if (min_tokens_ != 0 && NumOutputTokens() <= min_tokens_) {
      stop_check_offset = output_text_.size();
    }
  }

  if (skip_last) {
    // Cleanup after skipping detokenization.
    token_ids_.push_back(new_token_ids.back());
  }

  // 2) Evaluate stop strings.
  std::optional<std::string> stop_string;
  if (!stop_.empty() && NumOutputTokens() > min_tokens_) {
    const auto stop =
        CheckStopStrings(output_text_, output_text_.size() - stop_check_offset,
                         stop_, include_stop_str_in_output_);
    if (stop.has_value()) {
      stop_string = stop->first;
      if (stop->second != -1) {
        output_text_.resize(static_cast<size_t>(stop->second));
      }
    }
  }

  return stop_string;
}

std::string BaseIncrementalDetokenizer::GetNextOutputText(bool finished,
                                                          bool delta) {
  // We return the full output text if the sequence is finished.
  const size_t buffer_length = finished ? 0 : stop_buffer_length_;
  // Upstream slices `buffer_length` CHARS off the end of its Python str;
  // translate that to the byte offset of the same lossy-char boundary so a
  // streamed view never ends mid-UTF-8-character.
  const size_t length =
      buffer_length == 0
          ? output_text_.size()
          : ByteOffsetBeforeLossyChars(output_text_, buffer_length);
  if (!delta) return output_text_.substr(0, length);

  if (last_output_text_offset_ < length) {
    std::string next = output_text_.substr(
        last_output_text_offset_, length - last_output_text_offset_);
    last_output_text_offset_ = length;
    return next;
  }
  return std::string();
}

SlowIncrementalDetokenizer::SlowIncrementalDetokenizer(
    const tok::Tokenizer& tokenizer, DetokenizerRequest request)
    : BaseIncrementalDetokenizer(request),
      tokenizer_(tokenizer),
      prompt_len_(request.prompt_token_ids.size()),
      skip_special_tokens_(request.skip_special_tokens),
      spaces_between_special_tokens_(request.spaces_between_special_tokens) {
  // Metadata for incremental detokenization.
  PromptTokens prompt = ConvertPromptIdsToTokens(
      tokenizer, request.prompt_token_ids, skip_special_tokens_);
  tokens_ = std::move(prompt.tokens);
  prefix_offset_ = prompt.prefix_offset;
  read_offset_ = prompt.read_offset;

  token_ids_ = std::move(request.prompt_token_ids);
}

std::vector<int32_t> SlowIncrementalDetokenizer::OutputTokenIds() const {
  return std::vector<int32_t>(
      token_ids_.begin() + static_cast<std::ptrdiff_t>(prompt_len_),
      token_ids_.end());
}

size_t SlowIncrementalDetokenizer::NumOutputTokens() const {
  return token_ids_.size() - prompt_len_;
}

std::string SlowIncrementalDetokenizer::DecodeNext(int32_t next_token_id) {
  (void)next_token_id;  // already appended to token_ids_ by Update()
  IncrementalDetokenizeResult result = DetokenizeIncrementally(
      tokenizer_, token_ids_, &tokens_, prefix_offset_, read_offset_,
      skip_special_tokens_, spaces_between_special_tokens_);

  tokens_.insert(tokens_.end(),
                 std::make_move_iterator(result.new_tokens.begin()),
                 std::make_move_iterator(result.new_tokens.end()));
  prefix_offset_ = result.prefix_offset;
  read_offset_ = result.read_offset;

  return std::move(result.new_text);
}

}  // namespace vllm::v1
