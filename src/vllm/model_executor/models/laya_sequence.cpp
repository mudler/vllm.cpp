// Laya sequence construction — mirrors build_sequence from laya/common.py:48-85.
//
// The algorithm:
//   1. Tokenize "<type> question: <instructions>" → head_ids
//   2. For each option: [MASK_id] + tokenize(" " + option_text)[:48] → opt_ids[i]
//   3. Budget: opt_budget = head_max_len - sum(len(opt_ids))
//      If opt_budget < 16: truncate each option to max(4, (head_max_len-16)//len) tokens
//   4. Truncate head_ids to max(8, opt_budget)
//   5. ids = [CLS] + head_ids + [SEP]
//   6. For each option: markers.append(len(ids)); ids.extend(option)
//   7. ids.append(SEP)
//   8. room = max(0, max_len - len(ids) - 1)
//      state_ids = tokenize(state); truncate to room (left or right)
//   9. ids = ids + state_ids + [SEP]
//   10. return ids[:max_len], [m for m in markers if m < max_len]
#include "vllm/model_executor/models/laya_sequence.h"

#include <algorithm>
#include <string>

#include "vllm/tokenizer/tokenizer.h"

namespace vllm {
namespace laya {

std::string_view QTypeName(QType t) {
  switch (t) {
    case QType::kChoice:
      return "choice";
    case QType::kScore:
      return "score";
    case QType::kNoul:
      return "noul";
  }
  return "noul";
}

namespace {

// Replaces all occurrences of `from` with `to` in `str`. No-op if `from` is
// empty (avoids the infinite-loop / between-every-char insertion of a naive
// replace on an empty needle).
std::string ReplaceAll(std::string str, const std::string& from,
                        const std::string& to) {
  if (from.empty()) return str;
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.length(), to);
    pos += to.length();
  }
  return str;
}

}  // namespace

SequenceOutput BuildSequence(
    const vllm::tok::Tokenizer& tok,
    const SpecialTokens& special,
    const Question& q,
    int max_len,
    int head_max_len,
    const std::vector<int>& option_order,
    bool truncate_left) {
  const std::string& mask_tok = special.mask_str;

  // 1. Head text: "<type> question: <instructions>"
  //    Replace the mask token string in instructions with a space (mirrors
  //    Python: ins = str(q["ins"]).replace(mask_tok, " ")).
  std::string ins = q.instructions;
  if (!mask_tok.empty()) ins = ReplaceAll(ins, mask_tok, " ");
  std::string head_text =
      std::string(QTypeName(q.type)) + " question: " + ins;
  std::vector<int32_t> head_ids = tok.Encode(head_text);

  // 2. Options: [MASK_id] + tokenize(" " + option_text)[:48]
  //    The caller may provide option_order to permute options.
  std::vector<int> order;
  if (option_order.empty()) {
    order.reserve(q.options.size());
    for (int i = 0; i < static_cast<int>(q.options.size()); ++i)
      order.push_back(i);
  } else {
    order = option_order;
  }

  std::vector<std::vector<int32_t>> opt_ids;
  opt_ids.reserve(order.size());
  for (int i : order) {
    if (i < 0 || i >= static_cast<int>(q.options.size())) continue;
    std::string opt_text = " " + ReplaceAll(q.options[i], mask_tok, " ");
    std::vector<int32_t> ids = tok.Encode(opt_text);
    if (ids.size() > 48) ids.resize(48);
    ids.insert(ids.begin(), special.mask_id);
    opt_ids.push_back(std::move(ids));
  }

  // 3. Budget management.
  int opt_total = 0;
  for (const auto& o : opt_ids) opt_total += static_cast<int>(o.size());
  int opt_budget = head_max_len - opt_total;

  if (opt_budget < 16) {
    // Truncate each option to `per` tokens (including the MASK).
    int per = std::max(
        4, (head_max_len - 16) / std::max(1, static_cast<int>(opt_ids.size())));
    opt_total = 0;
    for (auto& o : opt_ids) {
      if (static_cast<int>(o.size()) > per) o.resize(per);
      opt_total += static_cast<int>(o.size());
    }
    opt_budget = head_max_len - opt_total;
  }

  // 4. Truncate head to leave room for options.
  int head_limit = std::max(8, opt_budget);
  if (static_cast<int>(head_ids.size()) > head_limit) head_ids.resize(head_limit);

  // 5-6. Assemble: [CLS] head_ids [SEP] <MASK> opt0 <MASK> opt1 ... [SEP]
  std::vector<int32_t> ids;
  ids.push_back(special.cls_id);
  ids.insert(ids.end(), head_ids.begin(), head_ids.end());
  ids.push_back(special.sep_id);

  std::vector<int32_t> markers;
  for (const auto& o : opt_ids) {
    markers.push_back(static_cast<int32_t>(ids.size()));
    ids.insert(ids.end(), o.begin(), o.end());
  }
  ids.push_back(special.sep_id);

  // 7-9. State: tokenize, truncate to room, append + [SEP].
  int room = std::max(0, max_len - static_cast<int>(ids.size()) - 1);
  std::string state_text = q.state;
  if (!mask_tok.empty()) state_text = ReplaceAll(state_text, mask_tok, " ");
  std::vector<int32_t> st = tok.Encode(state_text);

  if (truncate_left) {
    // st[-room:] — take the LAST `room` tokens.
    // Python gotcha: st[-0:] == st[0:] == st (the whole list). We mirror this
    // by NOT truncating when room == 0.
    if (room > 0 && static_cast<int>(st.size()) > room) {
      st.erase(st.begin(),
               st.begin() + (st.size() - static_cast<size_t>(room)));
    }
  } else {
    // st[:room] — take the FIRST `room` tokens.
    if (static_cast<int>(st.size()) > room) st.resize(room);
  }

  ids.insert(ids.end(), st.begin(), st.end());
  ids.push_back(special.sep_id);

  // 10. Final truncation to max_len.
  if (static_cast<int>(ids.size()) > max_len) ids.resize(max_len);

  // Filter markers beyond max_len (Python: [m for m in markers if m < max_len]).
  std::vector<int32_t> valid_markers;
  for (int32_t m : markers) {
    if (m < max_len) valid_markers.push_back(m);
  }

  return {std::move(ids), std::move(valid_markers)};
}

}  // namespace laya
}  // namespace vllm
