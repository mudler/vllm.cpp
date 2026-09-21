// cua-s1-forms ByteCollator — UTF-8 byte tokenization + tensor construction.
//
// Ported from trycua/cua libs/cua-s1/python/src/cua_s1/model.py @ 9bbfa7dd:
//   _byte_ids       model.py:48-49
//   ByteCollator    model.py:52-64
//   _tensor_batch   model.py:71-99
//
// Batch is always 1: one context + its options. Padding is dynamic (to the
// max length in the batch), matching the reference. Token 0 is padding;
// valid bytes map to 1-256.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/cua_s1.h"

namespace vllm {
namespace cua_s1 {

// [byte + 1 for byte in text.encode("utf-8")[:length]]
// Truncates to `length` bytes, maps each byte to byte+1 (1-256).
inline std::vector<int64_t> ByteIds(const std::string& text, int64_t max_len) {
  const int64_t n = std::min(static_cast<int64_t>(text.size()), max_len);
  std::vector<int64_t> ids;
  ids.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    ids.push_back(static_cast<int64_t>(
                      static_cast<unsigned char>(text[static_cast<size_t>(i)])) +
                  1);
  }
  return ids;
}

struct CollatedBatch {
  std::vector<int64_t> context_ids;       // (ctx_len,)
  std::vector<uint8_t> context_mask;      // (ctx_len,)
  std::vector<int64_t> option_ids;        // (n_opt * opt_len,) row-major
  std::vector<uint8_t> option_tok_mask;   // (n_opt * opt_len,)
  std::vector<uint8_t> option_mask;       // (n_opt,)
  int64_t ctx_len = 0;
  int64_t n_opt = 0;
  int64_t opt_len = 0;
};

// Collate one context + options into the tensor layout ForwardHost expects.
// Mirrors ByteCollator.__call__ + _tensor_batch for batch=1.
inline CollatedBatch ByteCollate(const Params& params,
                                 const std::string& context,
                                 const std::vector<std::string>& options) {
  CollatedBatch b;

  // Tokenize context (truncated to context_tokens bytes).
  std::vector<int64_t> ctx_ids = ByteIds(context, params.context_tokens);

  // Tokenize each option (truncated to option_tokens bytes).
  std::vector<std::vector<int64_t>> opt_ids;
  opt_ids.reserve(options.size());
  for (const auto& opt : options) {
    opt_ids.push_back(ByteIds(opt, params.option_tokens));
  }

  // Dynamic padding lengths (max in batch, at least 1).
  b.ctx_len = std::max<int64_t>(1, static_cast<int64_t>(ctx_ids.size()));
  b.n_opt = static_cast<int64_t>(opt_ids.size());
  b.opt_len = 1;
  for (const auto& ids : opt_ids) {
    b.opt_len = std::max(b.opt_len, static_cast<int64_t>(ids.size()));
  }

  // Context: pad to ctx_len with 0, mask = ids != 0.
  b.context_ids.resize(static_cast<size_t>(b.ctx_len), 0);
  b.context_mask.resize(static_cast<size_t>(b.ctx_len), 0);
  for (int64_t i = 0; i < static_cast<int64_t>(ctx_ids.size()); ++i) {
    b.context_ids[static_cast<size_t>(i)] = ctx_ids[static_cast<size_t>(i)];
    b.context_mask[static_cast<size_t>(i)] = 1;
  }

  // Options: pad each to opt_len with 0, mask = ids != 0.
  b.option_ids.resize(static_cast<size_t>(b.n_opt * b.opt_len), 0);
  b.option_tok_mask.resize(static_cast<size_t>(b.n_opt * b.opt_len), 0);
  b.option_mask.resize(static_cast<size_t>(b.n_opt), 1);
  for (int64_t n = 0; n < b.n_opt; ++n) {
    for (int64_t t = 0; t < static_cast<int64_t>(opt_ids[static_cast<size_t>(n)].size()); ++t) {
      const int64_t idx = n * b.opt_len + t;
      b.option_ids[static_cast<size_t>(idx)] = opt_ids[static_cast<size_t>(n)][static_cast<size_t>(t)];
      b.option_tok_mask[static_cast<size_t>(idx)] = 1;
    }
  }

  return b;
}

}  // namespace cua_s1
}  // namespace vllm
