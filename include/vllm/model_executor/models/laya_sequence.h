// Laya sequence construction — mirrors build_sequence from laya/common.py
// (NandhaKishorM/laya).
//
// Format: [CLS] <type> question: <instructions> [SEP]
//          <MASK> opt0 <MASK> opt1 ... [SEP] state [SEP]
//
// Each option starts with a MASK token; MASK positions are the markers the
// decision head gathers. Options are truncated to fit head_max_len (192 for
// the full model). State text fills the remaining space up to max_len (512).
//
// The tokenizer is ModernBERT BPE (GPT-2 style, vocab 50368). Special tokens:
// CLS=50281, SEP/EOS=50282, PAD=50283, MASK=50284. The BPE tokenizer
// infrastructure already in the tree (vllm/tokenizer/tokenizer.h) loads HF
// tokenizer.json and encodes text with no special tokens — exactly what
// build_sequence needs.
//
// THE THINGS THIS FUNCTION GETS WRONG QUIETLY are: (1) wrong question-type
// string in the head text — the model still runs and emits plausible logits;
// (2) missing MASK at option start — markers point to the wrong positions;
// (3) wrong truncation direction — the state is different but plausible;
// (4) not replacing the mask token string in input text — the mask leaks as a
// real token. Each is gated by a perturbation test.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vllm::tok {
class Tokenizer;
}

namespace vllm {
namespace laya {

// Question type: 0=choice, 1=score, 2=noul (mirrors QTYPES in laya/common.py).
enum class QType : int { kChoice = 0, kScore = 1, kNoul = 2 };

// ModernBERT BPE special-token IDs. Populated from the tokenizer's added
// tokens; the defaults are ModernBERT-large's values.
struct SpecialTokens {
  int32_t cls_id = 50281;
  int32_t sep_id = 50282;
  int32_t pad_id = 50283;
  int32_t mask_id = 50284;
  std::string mask_str;  // text form of the MASK token, e.g. "<mask>"
};

// A rendered question ready for sequence construction. The caller (server)
// renders options and serializes state before calling BuildSequence.
struct Question {
  QType type;
  std::string instructions;
  std::vector<std::string> options;
  std::string state;
};

// Result of sequence construction.
struct SequenceOutput {
  std::vector<int32_t> ids;
  std::vector<int32_t> markers;  // positions of MASK tokens, one per option
};

// Build the input sequence for one question.
//
// Mirrors build_sequence(tok, state, q, max_len, head_max_len, ...) from
// laya/common.py:48-85. The option_order and truncate_left parameters are
// supported; option_order defaults to identity (range(len(options))).
//
//   tok: a loaded BPE tokenizer (Tokenizer::FromHfJson)
//   special: CLS/SEP/PAD/MASK ids + mask string
//   q: rendered question (type, instructions, options, state)
//   max_len: total sequence cap (512 for the full model)
//   head_max_len: head budget for type+instructions+options (192)
//   option_order: option permutation, or empty for identity
//   truncate_left: true → keep the END of the state; false → keep the start
SequenceOutput BuildSequence(
    const vllm::tok::Tokenizer& tok,
    const SpecialTokens& special,
    const Question& q,
    int max_len = 512,
    int head_max_len = 192,
    const std::vector<int>& option_order = {},
    bool truncate_left = false);

// Type name string used in the head text ("choice", "score", "noul").
std::string_view QTypeName(QType t);

}  // namespace laya
}  // namespace vllm
