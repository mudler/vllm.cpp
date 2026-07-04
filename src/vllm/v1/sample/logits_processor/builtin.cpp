// Ported from: vllm/v1/sample/logits_processor/builtin.py @ e24d1b24.
#include "vllm/v1/sample/logits_processor/builtin.h"

#include <cstddef>

#include "vllm/v1/sample/device_scratch.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm::v1 {

void apply_min_tokens(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, MinTokensState>& min_tokens,
                      const std::vector<std::vector<int32_t>>& output_token_ids) {
  VT_CHECK(logits.rank == 2, "apply_min_tokens: logits must be [num_reqs, vocab]");
  if (min_tokens.empty()) return;

  // Build the (request, stop-token) -inf slice. Upstream pre-filters min_toks so
  // only under-floor requests remain; we re-check output_len < min_tokens so the
  // function is correct on its own (the same set upstream masks).
  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  for (const auto& [i, state] : min_tokens) {
    VT_CHECK(i >= 0 && static_cast<size_t>(i) < output_token_ids.size(),
             "apply_min_tokens: request index out of range");
    const int output_len = static_cast<int>(output_token_ids[static_cast<size_t>(i)].size());
    if (output_len >= state.min_tokens) continue;
    for (int32_t tok : state.stop_token_ids) {
      rows.push_back(i);
      cols.push_back(tok);
    }
  }

  if (rows.empty()) return;
  const int64_t m = static_cast<int64_t>(rows.size());
  DeviceScratch r(logits.device, q, rows.data(), vt::DType::kI32, {m});
  DeviceScratch c(logits.device, q, cols.data(), vt::DType::kI32, {m});
  vt::ApplyTokenMask(q, logits, r.tensor(), c.tensor());
}

void apply_logit_bias(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, std::map<int32_t, float>>& logit_bias) {
  VT_CHECK(logits.rank == 2, "apply_logit_bias: logits must be [num_reqs, vocab]");
  if (logit_bias.empty()) return;

  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  std::vector<float> biases;
  for (const auto& [i, tok_bias] : logit_bias) {
    for (const auto& [tok, bias] : tok_bias) {
      rows.push_back(i);
      cols.push_back(tok);
      biases.push_back(bias);
    }
  }

  if (rows.empty()) return;
  const int64_t m = static_cast<int64_t>(rows.size());
  DeviceScratch r(logits.device, q, rows.data(), vt::DType::kI32, {m});
  DeviceScratch c(logits.device, q, cols.data(), vt::DType::kI32, {m});
  DeviceScratch b(logits.device, q, biases.data(), vt::DType::kF32, {m});
  vt::ApplyLogitBias(q, logits, r.tensor(), c.tensor(), b.tensor());
}

void apply_min_p(vt::Queue& q, vt::Tensor& logits, const std::vector<float>& min_p) {
  const int64_t n = logits.shape[0];
  VT_CHECK(logits.rank == 2, "apply_min_p: logits must be [num_reqs, vocab]");
  VT_CHECK(static_cast<int64_t>(min_p.size()) == n, "apply_min_p: min_p must have num_reqs rows");
  DeviceScratch mp(logits.device, q, min_p.data(), vt::DType::kF32, {n});
  vt::ApplyMinP(q, logits, mp.tensor());
}

}  // namespace vllm::v1
