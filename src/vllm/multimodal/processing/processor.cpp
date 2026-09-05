#include "vllm/multimodal/processing/processor.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace vllm::multimodal {

namespace {

// `iter_token_matches` (`processing/processor.py:619-657` @ `9035151d6`): the
// first occurrence of `target` in `ids` at or after `from`, or -1.
//
// Upstream's generator excludes OVERLAPPING matches by advancing to
// `end_idx` after a hit. That exclusion lives in the CALLER here, which
// restarts every search from the previous applied match's end — the same
// property, kept in one place, because this function returns one match rather
// than a stream.
int FindTokenMatch(const std::vector<int32_t>& ids,
                   const std::vector<int32_t>& target, int from) {
  const int n = static_cast<int>(ids.size());
  const int m = static_cast<int>(target.size());
  for (int i = std::max(from, 0); i + m <= n; ++i) {
    bool hit = true;
    for (int j = 0; j < m; ++j) {
      if (ids[static_cast<std::size_t>(i + j)] !=
          target[static_cast<std::size_t>(j)]) {
        hit = false;
        break;
      }
    }
    if (hit) return i;
  }
  return -1;
}

}  // namespace

PromptReplacement MakeTokenTripleReplacement(std::string modality,
                                             int32_t start_id, int32_t pad_id,
                                             int32_t end_id,
                                             const std::vector<int>& counts) {
  PromptReplacement out;
  out.modality = std::move(modality);
  out.target = {start_id, pad_id, end_id};
  out.items.reserve(counts.size());
  for (const int n : counts) {
    if (n < 0) {
      throw std::runtime_error(
          "MakeTokenTripleReplacement: modality '" + out.modality +
          "' asked for a negative placeholder run (" + std::to_string(n) +
          "). The count is `prod(grid) // merge**2` for an image and "
          "`ceil(num_samples / token_stride)` for audio; neither can be "
          "negative, so this is a caller defect and not a request one.");
    }
    PromptUpdateContent c;
    c.full.reserve(static_cast<std::size_t>(n) + 2);
    c.full.push_back(start_id);
    c.full.insert(c.full.end(), static_cast<std::size_t>(n), pad_id);
    c.full.push_back(end_id);
    // `select_token_id(seq, pad_id)`: the embedding rows are the pads, which on
    // this shape are the contiguous run between the two markers.
    c.embed_offset = 1;
    c.embed_length = n;
    out.items.push_back(std::move(c));
  }
  return out;
}

std::vector<int32_t> ApplyPromptReplacements(
    const std::vector<int32_t>& prompt_ids,
    const std::vector<PromptReplacement>& updates,
    std::vector<AppliedPromptUpdate>* applied) {
  for (const PromptReplacement& u : updates) {
    if (u.target.empty()) {
      throw std::runtime_error(
          "ApplyPromptReplacements: modality '" + u.modality +
          "' carries an EMPTY target. Upstream's planner has an empty-target "
          "arm (`processing/processor.py:833-869` @ `9035151d6`) that this "
          "port does not represent, and treating it as a no-op would drop the "
          "item without a word. Refused by name.");
    }
  }

  std::vector<std::size_t> next(updates.size(), 0);
  std::vector<int32_t> out;
  out.reserve(prompt_ids.size());
  if (applied != nullptr) applied->clear();

  int prev_end = 0;
  while (true) {
    // The earliest match across every rule that still has an item, ties going
    // to the rule that appears earlier in `updates`.
    int best_rule = -1;
    int best_start = -1;
    for (std::size_t r = 0; r < updates.size(); ++r) {
      if (next[r] >= updates[r].items.size()) continue;
      const int at = FindTokenMatch(prompt_ids, updates[r].target, prev_end);
      if (at < 0) continue;
      if (best_rule < 0 || at < best_start) {
        best_rule = static_cast<int>(r);
        best_start = at;
      }
    }
    if (best_rule < 0) break;

    const PromptReplacement& u = updates[static_cast<std::size_t>(best_rule)];
    const std::size_t item = next[static_cast<std::size_t>(best_rule)];
    const PromptUpdateContent& content = u.items[item];

    out.insert(out.end(), prompt_ids.begin() + prev_end,
               prompt_ids.begin() + best_start);
    const int base = static_cast<int>(out.size());
    out.insert(out.end(), content.full.begin(), content.full.end());

    if (applied != nullptr) {
      AppliedPromptUpdate a;
      a.modality = u.modality;
      a.item_index = static_cast<int>(item);
      a.offset = base + content.embed_offset;
      a.length = content.embed_length;
      applied->push_back(std::move(a));
    }

    // Matches are EXCLUSIVE: the next search for EVERY rule starts past this
    // match, which is upstream's `prev_end_idx` (`:811`, `:889`).
    prev_end = best_start + static_cast<int>(u.target.size());
    ++next[static_cast<std::size_t>(best_rule)];
  }
  out.insert(out.end(), prompt_ids.begin() + prev_end, prompt_ids.end());

  for (std::size_t r = 0; r < updates.size(); ++r) {
    if (next[r] == updates[r].items.size()) continue;
    throw std::runtime_error(
        "ApplyPromptReplacements: the prompt carries " +
        std::to_string(next[r]) + " '" + updates[r].modality +
        "' placeholder target(s) but the request carries " +
        std::to_string(updates[r].items.size()) +
        " item(s) of that modality. The marker the chat seam injects and the "
        "target this rule matches must be the same token sequence; serving "
        "the request would drop item " + std::to_string(next[r]) +
        " silently, and the model would answer fluently from media it never "
        "saw.");
  }
  return out;
}

}  // namespace vllm::multimodal
