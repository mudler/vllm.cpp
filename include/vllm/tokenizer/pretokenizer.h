// vllm.cpp original (tokenizer); semantics mirror HF tokenizers' Split
// pre-tokenizer (behavior=Isolated) for the byte-level BPE split regexes.
// See src/vllm/tokenizer/pretokenizer.cpp for the verbatim patterns.
#pragma once

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace vllm::tok {

// Which pre-tokenizer Split regex to emulate.
enum class SplitPattern {
  kQwen2,   // Qwen family: single-codepoint \p{N} (regex from Qwen3.6-27B)
  kLlama3,  // Llama-3 family: \p{N}{1,3} digit groups
};

// Splits `text` into pretoken byte spans [first, second), exactly as HF
// tokenizers' Split(Regex(pattern), behavior="isolated") does. Spans are
// contiguous, non-overlapping, and cover the input: concatenating
// text.substr(f, s - f) over all spans reconstructs `text` byte-for-byte.
// Invalid UTF-8 decodes as U+FFFD (category So) for classification purposes,
// but the spans always index the original bytes.
std::vector<std::pair<size_t, size_t>> Pretokenize(std::string_view text,
                                                   SplitPattern pattern);

}  // namespace vllm::tok
