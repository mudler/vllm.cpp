// Ported from: vllm/v1/sample/ops/penalties.py + vllm/model_executor/layers/utils.py
// (apply_penalties, get_token_bin_counts_and_mask) + vllm/_custom_ops.py
// (apply_repetition_penalties_torch) @ e24d1b24.
//
// Presence / frequency / repetition penalties over the model's final logits
// [num_reqs, vocab] (row-major f32). These wrappers build the per-request bin
// counts + prompt/output presence masks from the ragged host token-id lists
// carried by SamplingMetadata (the analogue of upstream's padded token tensors +
// get_token_bin_counts_and_mask scatter_add), materialize them on the logits'
// device, and apply the exact OpenAI-defined formula via vt::ApplyPenalties.
#ifndef VLLM_V1_SAMPLE_OPS_PENALTIES_H_
#define VLLM_V1_SAMPLE_OPS_PENALTIES_H_

#include <cstdint>
#include <vector>

#include "vt/tensor.h"

namespace vllm::v1 {

// apply_penalties (utils.py::apply_penalties). Builds the prompt presence mask
// and the output bin counts + presence mask, then applies (in place):
//   repetition on the (prompt|output) union mask (divide-if-positive /
//   multiply-if-negative), then `logits -= freq * output_bin_counts` and
//   `logits -= pres * output_mask`. prompt_token_ids / output_token_ids are
//   ragged [num_reqs][...] lists; a token id outside [0, vocab) (the vocab-size
//   pad or the -1 async placeholder) is ignored in the bin-count scatter.
void apply_penalties(vt::Queue& q, vt::Tensor& logits,
                     const std::vector<std::vector<int32_t>>& prompt_token_ids,
                     const std::vector<std::vector<int32_t>>& output_token_ids,
                     const std::vector<float>& presence_penalties,
                     const std::vector<float>& frequency_penalties,
                     const std::vector<float>& repetition_penalties);

// apply_all_penalties (penalties.py::apply_all_penalties). Upstream pads the
// output_token_ids to a tensor with `vocab_size` as the pad and replaces the -1
// async placeholder with `vocab_size` before the scatter; both map to
// out-of-range ids that apply_penalties ignores, so here the padding/-1 handling
// is folded into apply_penalties' range-checked bin-count scatter.
void apply_all_penalties(vt::Queue& q, vt::Tensor& logits,
                         const std::vector<std::vector<int32_t>>& prompt_token_ids,
                         const std::vector<float>& presence_penalties,
                         const std::vector<float>& frequency_penalties,
                         const std::vector<float>& repetition_penalties,
                         const std::vector<std::vector<int32_t>>& output_token_ids);

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_OPS_PENALTIES_H_
