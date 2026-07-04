// Ported from: vllm/v1/sample/ops/penalties.py + vllm/model_executor/layers/utils.py
// (apply_penalties, get_token_bin_counts_and_mask) + vllm/_custom_ops.py
// (apply_repetition_penalties_torch) @ e24d1b24.
#include "vllm/v1/sample/ops/penalties.h"

#include <cstddef>

#include "vllm/v1/sample/device_scratch.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm::v1 {

void apply_penalties(vt::Queue& q, vt::Tensor& logits,
                     const std::vector<std::vector<int32_t>>& prompt_token_ids,
                     const std::vector<std::vector<int32_t>>& output_token_ids,
                     const std::vector<float>& presence_penalties,
                     const std::vector<float>& frequency_penalties,
                     const std::vector<float>& repetition_penalties) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  VT_CHECK(logits.rank == 2, "apply_penalties: logits must be [num_reqs, vocab]");
  VT_CHECK(static_cast<int64_t>(prompt_token_ids.size()) == n &&
               static_cast<int64_t>(output_token_ids.size()) == n &&
               static_cast<int64_t>(presence_penalties.size()) == n &&
               static_cast<int64_t>(frequency_penalties.size()) == n &&
               static_cast<int64_t>(repetition_penalties.size()) == n,
           "apply_penalties: all per-request inputs must have num_reqs rows");

  // get_token_bin_counts_and_mask over the ragged host lists. A token id outside
  // [0, vocab) (vocab-size pad / -1 placeholder) is skipped — the analogue of
  // upstream's `bin_counts[:, :vocab]` slice dropping the pad column.
  const size_t nv = static_cast<size_t>(n) * static_cast<size_t>(v);
  std::vector<int8_t> prompt_mask(nv, 0);
  std::vector<int8_t> output_mask(nv, 0);
  std::vector<int32_t> output_bin_counts(nv, 0);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t base = i * v;
    for (int32_t t : prompt_token_ids[static_cast<size_t>(i)])
      if (t >= 0 && t < v) prompt_mask[static_cast<size_t>(base + t)] = 1;
    for (int32_t t : output_token_ids[static_cast<size_t>(i)])
      if (t >= 0 && t < v) {
        output_bin_counts[static_cast<size_t>(base + t)] += 1;
        output_mask[static_cast<size_t>(base + t)] = 1;
      }
  }

  DeviceScratch pm(logits.device, q, prompt_mask.data(), vt::DType::kI8, {n, v});
  DeviceScratch om(logits.device, q, output_mask.data(), vt::DType::kI8, {n, v});
  DeviceScratch oc(logits.device, q, output_bin_counts.data(), vt::DType::kI32, {n, v});
  DeviceScratch fr(logits.device, q, frequency_penalties.data(), vt::DType::kF32, {n});
  DeviceScratch pr(logits.device, q, presence_penalties.data(), vt::DType::kF32, {n});
  DeviceScratch rp(logits.device, q, repetition_penalties.data(), vt::DType::kF32, {n});

  vt::ApplyPenalties(q, logits, pm.tensor(), oc.tensor(), om.tensor(), fr.tensor(), pr.tensor(),
                     rp.tensor());
}

void apply_all_penalties(vt::Queue& q, vt::Tensor& logits,
                         const std::vector<std::vector<int32_t>>& prompt_token_ids,
                         const std::vector<float>& presence_penalties,
                         const std::vector<float>& frequency_penalties,
                         const std::vector<float>& repetition_penalties,
                         const std::vector<std::vector<int32_t>>& output_token_ids) {
  apply_penalties(q, logits, prompt_token_ids, output_token_ids, presence_penalties,
                  frequency_penalties, repetition_penalties);
}

}  // namespace vllm::v1
