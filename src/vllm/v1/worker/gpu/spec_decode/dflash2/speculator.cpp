// DFlash2 speculator — candidate selection (W3) and the PATH WALK (W4), #1314.
// See the header for the port note and the upstream anchors.
#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm::v1 {

void RefuseDflash1ArgmaxOnDflash2Block(const Qwen3DFlashWeights& weights) {
  if (!weights.IsDflash2()) return;
  VT_CHECK(false,
           "dflash2: a DFlash2 block reached the DFlash1 per-slot argmax. That is "
           "not a fallback, it is the one defect this architecture exists to "
           "remove: the argmax proposes well-formed tokens, the verify is "
           "lossless, the engine still emits the TARGET's tokens, and only "
           "ACCEPTANCE falls -- so no token gate in this repository can see it. A "
           "DFlash2 draft must draft through the candidate selector's PATH WALK "
           "(vllm::v1::Dflash2WalkPath -> vt::Dflash2PathWalk), which mirrors "
           "DFlash2Speculator._sample_path / _selector_walk_kernel "
           "(vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py) @ "
           "vllm-project/vllm#52816 head "
           "66e5414c6d75a8529473d977f7458c140bbab8a0. Reaching this message means "
           "the walk's call site is gone from the propose path, not that a "
           "checkpoint is unsupported. Row SPEC-DFLASH2 "
           "(.agents/specs/dflash2-spec-decode.md), issue #1314 "
           "(https://github.com/mudler/vllm.cpp/issues/1314).");
}

Dflash2WalkResult Dflash2WalkPath(const Dflash2ProposeState& scored, vt::Queue& queue) {
  const int64_t B = scored.num_reqs, L = scored.num_steps, K = scored.top_k;
  VT_CHECK(B > 0 && L > 0 && K > 0,
           "dflash2 path-walk: the scored lattice must have requests, steps and "
           "candidates");
  VT_CHECK(scored.candidates.rows == B * L && scored.candidates.top_k == K,
           "dflash2 path-walk: the candidate set must be [num_reqs*num_steps, top_k]");
  VT_CHECK(static_cast<int64_t>(scored.edge_scores.size()) == B * L * K * K,
           "dflash2 path-walk: the lattice must score every (step, predecessor, "
           "child) transition of every request");

  dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};
  dense_attn::DBuf dev_scores(d, vt::DType::kF32, {B, L, K, K}, scored.edge_scores.data());
  dense_attn::DBuf dev_cand(d, vt::DType::kI64, {B, L, K}, scored.candidates.ids.data());
  dense_attn::DBuf dev_tokens(d, vt::DType::kI64, {B, L});
  vt::Dflash2PathWalkArgs args;
  args.top_k = K;
  vt::Dflash2PathWalk(d.q, dev_tokens.t(), dev_scores.t(), dev_cand.t(), args);

  std::vector<int64_t> tokens(static_cast<size_t>(B * L), 0);
  dev_tokens.Download(d, tokens.data());

  Dflash2WalkResult out;
  out.draft_token_ids.assign(static_cast<size_t>(B), {});
  for (int64_t b = 0; b < B; ++b) {
    std::vector<int32_t>& row = out.draft_token_ids[static_cast<size_t>(b)];
    row.reserve(static_cast<size_t>(L));
    for (int64_t l = 0; l < L; ++l) {
      const int64_t id = tokens[static_cast<size_t>(b * L + l)];
      // The verify, the KV rollback and the input batch all carry token ids as
      // i32. A candidate that does not fit is an id-space error and is named
      // here rather than truncated into a different, valid-looking token.
      VT_CHECK(id >= 0 && id <= static_cast<int64_t>(INT32_MAX),
               "dflash2 path-walk: the walk produced a token id outside the i32 "
               "range the verify carries");
      row.push_back(static_cast<int32_t>(id));
    }
  }
  return out;
}

Dflash2ProposeState Dflash2SelectCandidates(const std::vector<float>& block_logits,
                                            const std::vector<float>& block_hidden,
                                            const std::vector<int32_t>& anchors,
                                            int num_reqs, int k,
                                            const Qwen3DFlashWeights& weights,
                                            const HfConfig& config, vt::Queue& queue) {
  VT_CHECK(weights.IsDflash2(),
           "dflash2 select-candidates: called on a draft that is not a DFlash2 draft");
  VT_CHECK(num_reqs > 0 && k > 0,
           "dflash2 select-candidates: num_reqs and k must be > 0");
  const int64_t P = num_reqs, L = k, nq = static_cast<int64_t>(k) + 1;
  const int64_t vocab = weights.draft_vocab_size;
  const int64_t H = config.hidden_size;
  VT_CHECK(vocab > 0 && H > 0, "dflash2 select-candidates: invalid draft vocab/hidden");
  VT_CHECK(static_cast<int64_t>(block_logits.size()) == P * nq * vocab,
           "dflash2 select-candidates: block_logits must be [num_reqs*(1+k), draft_vocab]");
  VT_CHECK(static_cast<int64_t>(block_hidden.size()) == P * nq * H,
           "dflash2 select-candidates: block_hidden must be [num_reqs*(1+k), H] "
           "(the block forward's final_out, captured on the SAME forward)");
  VT_CHECK(static_cast<int64_t>(anchors.size()) == P,
           "dflash2 select-candidates: one anchor token per proposing row");

  // Step 2 — the SAMPLE-ROW gather, upstream's
  // `last_hidden_states[self.sample_indices[:num_sample]]`. Rows +1..+k of each
  // request's block: the k mask positions. Row +0 is the anchor and is skipped,
  // which is `sample_from_anchor=false` (DFlash's layout, unchanged by DFlash2).
  std::vector<float> sample_logits(static_cast<size_t>(P * L * vocab));
  std::vector<float> sample_hidden(static_cast<size_t>(P * L * H));
  for (int64_t r = 0; r < P; ++r) {
    for (int64_t j = 0; j < L; ++j) {
      const int64_t src = r * nq + 1 + j;
      const int64_t dst = r * L + j;
      std::copy(block_logits.begin() + static_cast<std::ptrdiff_t>(src * vocab),
                block_logits.begin() + static_cast<std::ptrdiff_t>((src + 1) * vocab),
                sample_logits.begin() + static_cast<std::ptrdiff_t>(dst * vocab));
      std::copy(block_hidden.begin() + static_cast<std::ptrdiff_t>(src * H),
                block_hidden.begin() + static_cast<std::ptrdiff_t>((src + 1) * H),
                sample_hidden.begin() + static_cast<std::ptrdiff_t>(dst * H));
    }
  }

  Dflash2ProposeState state;
  state.num_reqs = P;
  state.num_steps = L;
  state.top_k = weights.candidate_selector.top_k;
  // Steps 3 and 4. Upstream runs them in this order for a reason the selector
  // depends on: the candidate ids the lattice indexes its codebooks with ARE
  // compute_candidates' output, already rebased and already scaled.
  state.candidates = Qwen3DFlash2Model::ComputeCandidates(sample_logits, P * L, vocab,
                                                          weights, queue);
  state.edge_scores = Qwen3DFlash2Model::SelectorEdgeScores(
      state.candidates, sample_hidden, anchors, P, L, weights, config, queue);
  // The postcondition, asserted here rather than at each call site: every
  // (step, predecessor, child) transition of every request is scored.
  VT_CHECK(static_cast<int64_t>(state.edge_scores.size()) ==
               P * L * state.top_k * state.top_k,
           "dflash2 select-candidates: the selector must score every "
           "(step, predecessor, child) transition of every request");
  return state;
}

}  // namespace vllm::v1
