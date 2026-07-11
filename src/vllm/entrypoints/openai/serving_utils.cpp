// vllm.cpp original — see serving_utils.h.
#include "vllm/entrypoints/openai/serving_utils.h"

#include <cstdint>

namespace vllm::entrypoints::openai {

StreamUsageSelection ShouldIncludeUsage(
    const std::optional<StreamOptions>& stream_options,
    bool enable_force_include_usage) {
  if (enable_force_include_usage) return {true, true};
  if (!stream_options.has_value()) return {};
  const bool include_usage = stream_options->include_usage;
  return {include_usage,
          include_usage && stream_options->continuous_usage_stats};
}

std::string SanitizeUtf8(const std::string& s) {
  // Mirrors detokenizer.cpp::LossyStep: consume one valid UTF-8 character or one
  // maximal invalid subpart per step. Valid characters (and a literal U+FFFD)
  // are copied verbatim; every invalid subpart is emitted as one U+FFFD.
  static constexpr char kReplacement[] = "\xEF\xBF\xBD";
  std::string out;
  out.reserve(s.size());
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    const uint8_t b0 = static_cast<uint8_t>(s[i]);
    if (b0 < 0x80) {  // ASCII
      out.push_back(s[i]);
      ++i;
      continue;
    }
    size_t need;        // continuation bytes required by this lead
    uint8_t lo = 0x80;  // valid range for the FIRST continuation byte
    uint8_t hi = 0xBF;  // (encodes the overlong/surrogate/range rules)
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
    } else {  // stray continuation byte (0x80-0xC1) or invalid lead (0xF5-0xFF)
      out += kReplacement;
      ++i;
      continue;
    }
    size_t len = 1;
    for (size_t k = 0; k < need; ++k) {
      if (i + len >= n) break;
      const uint8_t b = static_cast<uint8_t>(s[i + len]);
      const uint8_t clo = (k == 0) ? lo : uint8_t{0x80};
      const uint8_t chi = (k == 0) ? hi : uint8_t{0xBF};
      if (b < clo || b > chi) break;
      ++len;
    }
    if (len != need + 1) {  // truncated / invalid multibyte: one maximal subpart
      out += kReplacement;
      i += len;
      continue;
    }
    out.append(s, i, len);  // a valid character — copied byte-for-byte
    i += len;
  }
  return out;
}

}  // namespace vllm::entrypoints::openai
