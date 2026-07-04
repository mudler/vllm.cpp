// Ported from: vllm/v1/sample/ops/bad_words.py @ e24d1b24 (+ the inline
// allowed-token-ids masked_fill from vllm/v1/sample/sampler.py:396-397).
#include "vllm/v1/sample/ops/bad_words.h"

#include <cstddef>

#include "vllm/v1/sample/device_scratch.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm::v1 {

void apply_bad_words(vt::Queue& q, vt::Tensor& logits,
                     const std::map<int, std::vector<std::vector<int32_t>>>& bad_words_token_ids,
                     const std::vector<std::vector<int32_t>>& past_tokens_ids) {
  VT_CHECK(logits.rank == 2, "apply_bad_words: logits must be [num_reqs, vocab]");

  // Resolve every (request, blocked-last-token) pair via the n-gram suffix match
  // (_apply_bad_words_single_batch), then scatter -inf in one op.
  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  for (const auto& [i, bad_words_ids] : bad_words_token_ids) {
    VT_CHECK(i >= 0 && static_cast<size_t>(i) < past_tokens_ids.size(),
             "apply_bad_words: request index out of range");
    const std::vector<int32_t>& past = past_tokens_ids[static_cast<size_t>(i)];
    for (const std::vector<int32_t>& bad_word_ids : bad_words_ids) {
      const size_t len = bad_word_ids.size();
      if (len == 0) continue;
      if (len > past.size() + 1) continue;  // longer than possible context

      const size_t prefix_length = len - 1;
      const int32_t last_token_id = bad_word_ids[len - 1];
      // actual_prefix = past[-prefix_length:]; expected_prefix = bad_word_ids[:prefix_length].
      bool match = true;
      for (size_t p = 0; p < prefix_length; ++p) {
        if (past[past.size() - prefix_length + p] != bad_word_ids[p]) {
          match = false;
          break;
        }
      }
      if (match) {
        rows.push_back(i);
        cols.push_back(last_token_id);
      }
    }
  }

  if (rows.empty()) return;
  const int64_t m = static_cast<int64_t>(rows.size());
  DeviceScratch r(logits.device, q, rows.data(), vt::DType::kI32, {m});
  DeviceScratch c(logits.device, q, cols.data(), vt::DType::kI32, {m});
  vt::ApplyTokenMask(q, logits, r.tensor(), c.tensor());
}

void apply_allowed_token_ids(vt::Queue& q, vt::Tensor& logits,
                             const std::vector<std::vector<uint8_t>>& mask) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  VT_CHECK(logits.rank == 2, "apply_allowed_token_ids: logits must be [num_reqs, vocab]");
  VT_CHECK(static_cast<int64_t>(mask.size()) == n,
           "apply_allowed_token_ids: mask must have num_reqs rows");

  // Flatten the [num_reqs][vocab] rows into a contiguous i8 [n, v] buffer.
  std::vector<int8_t> flat(static_cast<size_t>(n) * static_cast<size_t>(v), 0);
  for (int64_t i = 0; i < n; ++i) {
    VT_CHECK(static_cast<int64_t>(mask[static_cast<size_t>(i)].size()) == v,
             "apply_allowed_token_ids: each mask row must be vocab-wide");
    const int64_t base = i * v;
    for (int64_t j = 0; j < v; ++j)
      flat[static_cast<size_t>(base + j)] =
          static_cast<int8_t>(mask[static_cast<size_t>(i)][static_cast<size_t>(j)]);
  }

  DeviceScratch mk(logits.device, q, flat.data(), vt::DType::kI8, {n, v});
  vt::ApplyAllowedTokenIds(q, logits, mk.tensor());
}

}  // namespace vllm::v1
