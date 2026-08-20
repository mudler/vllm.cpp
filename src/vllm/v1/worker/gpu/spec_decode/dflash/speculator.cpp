// Ported from vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
// (DflashSpeculator.propose :300-413, _generate_draft :242-273) @ 555967922.
// See the header for scope + the exact upstream anchors. SPEC-DFLASH D4,
// DF-ENGINE-INTEGRATION.
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"

#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"

#include <cstddef>

#include "vt/backend.h"

namespace vllm::v1 {


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
  //
  // SPEC-DFLASH2 W3 (#1314): a DFlash2 draft ALSO captures `final_out`, the
  // post-final-norm hidden the candidate selector's `hidden_projection` reads.
  // Upstream's `_generate_draft` gets both from one forward for the same reason
  // -- the selector must project the SAME hidden states the logits came from,
  // and a second forward would be a second model state.
  const bool dflash2 = weights.IsDflash2();
  std::vector<float> block_hidden;
  const std::vector<float> block_logits =
      Qwen3DFlashModel::ForwardBlockLogitsWithContext(
          context_states, context_positions, ctx_cu, block_input_ids,
          block_positions, block_cu, weights, config, queue, nullptr,
          dflash2 ? &block_hidden : nullptr);

  // SPEC-DFLASH2 W4 (#1314): the conv (W2), the CANDIDATE SELECTOR (W3) and the
  // PATH WALK (W4) all run for a DFlash2 draft, and the walk -- not the DFlash1
  // per-slot argmax below -- is what produces its tokens. Same sequence as
  // upstream's `DFlash2Speculator._generate_draft`, and the same call as
  // `GPUModelRunner::propose_drafts_block`: one implementation, two entries.
  DflashProposeResult out;
  if (dflash2) {
    std::vector<int32_t> anchors(static_cast<size_t>(num_reqs));
    for (int r = 0; r < num_reqs; ++r)
      anchors[static_cast<size_t>(r)] =
          block_input_ids[static_cast<size_t>(static_cast<int64_t>(r) * block)];
    const Dflash2ProposeState selected = Dflash2SelectCandidates(
        block_logits, block_hidden, anchors, num_reqs, k, weights, config, queue);
    out.draft_token_ids = Dflash2WalkPath(selected, queue).draft_token_ids;
  }

  // The DFlash1 arm. It is entered on emptiness rather than on `!dflash2` so
  // that DELETING the walk above lands here instead of falling past an else --
  // and `RefuseDflash1ArgmaxOnDflash2Block` then makes that loud. A DFlash2
  // block sampled by the per-slot argmax succeeds and loses only acceptance,
  // which no token gate can see.
  if (out.draft_token_ids.empty()) {
    RefuseDflash1ArgmaxOnDflash2Block(weights);
    out.draft_token_ids = SampleDflashBlockDrafts(block_logits, num_reqs, k,
                                                  weights.draft_vocab_size);
  }
  return out;
}

}  // namespace vllm::v1
