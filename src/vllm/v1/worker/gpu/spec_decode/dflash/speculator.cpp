// Ported from vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
// (DflashSpeculator.propose :300-413, _generate_draft :242-273) @ 555967922.
// See the header for scope + the exact upstream anchors. SPEC-DFLASH D4,
// DF-ENGINE-INTEGRATION.
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"

#include <cstddef>

#include "vt/backend.h"

namespace vllm::v1 {

void RefuseDflash2CandidateSelector(const Qwen3DFlashWeights& weights) {
  if (!weights.IsDflash2()) return;
  VT_CHECK(false,
           "dflash2: this draft is a DFlash2 draft (its dflash_config declares "
           "conv_kernel_size/conv_group_size and its layers carry the "
           "attention_conv/mlp_conv tensors). Its grouped dynamic depthwise "
           "convolution IS implemented and just ran; its CANDIDATE SELECTOR is not. "
           "Upstream replaces the per-slot argmax with a scored path walk over the "
           "target head's top-K -- CandidateSelector "
           "(vllm/model_executor/models/qwen3_dflash2.py) plus the walk kernel and "
           "the realized-q draft-logit cache "
           "(vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py) @ "
           "vllm-project/vllm#52816 head 19c9351904df4c63042671bc67a866ca48dc7d6f. "
           "Sampling this block with the DFlash1 per-slot argmax instead would "
           "succeed and propose worse tokens with NO visible symptom: the verify is "
           "lossless, so the emitted tokens are still the target's and only "
           "acceptance falls. Owed by row SPEC-DFLASH2 wave W3 "
           "(.agents/specs/dflash2-spec-decode.md), issue #1314 "
           "(https://github.com/mudler/vllm.cpp/issues/1314). Use a DFlashDraftModel "
           "checkpoint until that wave lands.");
}

std::vector<std::vector<int32_t>> SampleDflashBlockDrafts(
    const std::vector<float>& block_logits, int num_reqs, int k,
    int64_t draft_vocab) {
  VT_CHECK(num_reqs > 0 && k > 0 && draft_vocab > 0,
           "SampleDflashBlockDrafts: bad shape");
  const int64_t block = static_cast<int64_t>(k) + 1;  // 1 + k rows per request
  const int64_t rows = static_cast<int64_t>(num_reqs) * block;
  VT_CHECK(static_cast<int64_t>(block_logits.size()) == rows * draft_vocab,
           "SampleDflashBlockDrafts: block_logits must be [num_reqs*(1+k), draft_vocab]");

  std::vector<std::vector<int32_t>> drafts(static_cast<size_t>(num_reqs));
  for (int r = 0; r < num_reqs; ++r) {
    std::vector<int32_t>& row = drafts[static_cast<size_t>(r)];
    row.resize(static_cast<size_t>(k));
    // Request r's block occupies rows [r*(1+k) .. r*(1+k)+k]; offset 0 is the
    // ANCHOR (not sampled), offsets 1..k are the k mask positions. Greedy
    // sample_draft = argmax over the draft-vocab logits row, lowest-index tie
    // (the D2/D3 proposed-id gates use the same argmax).
    for (int j = 0; j < k; ++j) {
      const int64_t global_row =
          static_cast<int64_t>(r) * block + 1 + static_cast<int64_t>(j);
      const float* lr =
          block_logits.data() + static_cast<size_t>(global_row) * draft_vocab;
      int32_t best_idx = 0;
      float best_val = lr[0];
      for (int64_t v = 1; v < draft_vocab; ++v) {
        if (lr[static_cast<size_t>(v)] > best_val) {
          best_val = lr[static_cast<size_t>(v)];
          best_idx = static_cast<int32_t>(v);
        }
      }
      row[static_cast<size_t>(j)] = best_idx;
    }
  }
  return drafts;
}

DflashProposeResult DflashProposeBlock(
    const Qwen3DFlashWeights& weights, const HfConfig& config,
    const std::vector<float>& context_states,
    const std::vector<int32_t>& context_positions,
    const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids,
    const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& block_cu, int num_reqs, int k, vt::Queue& queue) {
  VT_CHECK(num_reqs > 0 && k > 0, "DflashProposeBlock: bad batch/k");
  const int64_t block = static_cast<int64_t>(k) + 1;
  VT_CHECK(static_cast<int64_t>(block_input_ids.size()) ==
               static_cast<int64_t>(num_reqs) * block,
           "DflashProposeBlock: block_input_ids must be [num_reqs*(1+k)]");
  VT_CHECK(block_positions.size() == block_input_ids.size(),
           "DflashProposeBlock: block_positions length must match input_ids");
  VT_CHECK(static_cast<int64_t>(block_cu.size()) == num_reqs + 1 &&
               static_cast<int64_t>(ctx_cu.size()) == num_reqs + 1,
           "DflashProposeBlock: cu vectors must be [num_reqs+1]");

  // The context-aware (1+k) block forward (D3): PrecomputeContextKV the accumulated
  // combined features + attend the block over [context; block] per request. Returns
  // [num_reqs*(1+k), draft_vocab] f32 draft logits (lm_head applied).
  const std::vector<float> block_logits =
      Qwen3DFlashModel::ForwardBlockLogitsWithContext(
          context_states, context_positions, ctx_cu, block_input_ids,
          block_positions, block_cu, weights, config, queue);

  // SPEC-DFLASH2 W2 (#1314): the conv has run; the selector has not been ported.
  RefuseDflash2CandidateSelector(weights);

  DflashProposeResult out;
  out.draft_token_ids = SampleDflashBlockDrafts(block_logits, num_reqs, k,
                                                weights.draft_vocab_size);
  return out;
}

}  // namespace vllm::v1
