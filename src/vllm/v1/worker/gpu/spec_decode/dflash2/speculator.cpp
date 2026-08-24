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

Dflash2WalkResult Dflash2WalkPathDevice(const Dflash2ProposeStateDevice& scored,
                                        vt::Queue& queue) {
  const int64_t B = scored.num_reqs, L = scored.num_steps, K = scored.top_k;
  VT_CHECK(B > 0 && L > 0 && K > 0,
           "dflash2 path-walk: the scored lattice must have requests, steps and "
           "candidates");
  VT_CHECK(scored.candidates.rows == B * L && scored.candidates.top_k == K,
           "dflash2 path-walk: the candidate set must be [num_reqs*num_steps, top_k]");
  VT_CHECK(scored.edges.scores.data != nullptr && scored.edges.scores.Numel() == B * L * K * K,
           "dflash2 path-walk: the lattice must score every (step, predecessor, "
           "child) transition of every request");

  dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};
  vt::Tensor cand = scored.candidates.ids;
  cand = dense_attn::Reshape(cand, {B, L, K});
  dense_attn::DBuf dev_tokens(d, vt::DType::kI64, {B, L});
  vt::Dflash2PathWalkArgs args;
  args.top_k = K;
  vt::Dflash2PathWalk(d.q, dev_tokens.t(), scored.edges.scores, cand, args);

  // SPEC-DFLASH2 W8 (#1837): the ONE download of the whole selector+walk — the
  // drafted token ids, which is all upstream's `_generate_draft` ever brings
  // back either.
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
  // The marshaling shell (SPEC-DFLASH2 W8): upload the host lattice and
  // candidates, then run the SAME device walk. This is the "one round trip" the
  // pre-W8 header note named; the production runner no longer makes it.
  dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};
  dense_attn::DBuf dev_scores(d, vt::DType::kF32, {B, L, K, K}, scored.edge_scores.data());
  dense_attn::DBuf dev_cand(d, vt::DType::kI64, {B * L, K}, scored.candidates.ids.data());
  Dflash2ProposeStateDevice dev;
  dev.num_reqs = B;
  dev.num_steps = L;
  dev.top_k = K;
  dev.candidates.rows = B * L;
  dev.candidates.top_k = K;
  dev.candidates.ids = dev_cand.t();
  dev.candidates.keep_ids = dev_cand.ReleaseShared();
  dev.edges.scores = dev_scores.t();
  dev.edges.keep = dev_scores.ReleaseShared();
  return Dflash2WalkPathDevice(dev, queue);
}

Dflash2ProposeStateDevice Dflash2SelectCandidatesDevice(
    const vt::Tensor& block_logits, const vt::Tensor& block_hidden,
    const std::vector<int32_t>& anchors, int num_reqs, int k,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue) {
  VT_CHECK(weights.IsDflash2(),
           "dflash2 select-candidates: called on a draft that is not a DFlash2 draft");
  VT_CHECK(num_reqs > 0 && k > 0,
           "dflash2 select-candidates: num_reqs and k must be > 0");
  const int64_t P = num_reqs, L = k, nq = static_cast<int64_t>(k) + 1;
  const int64_t vocab = weights.draft_vocab_size;
  const int64_t H = config.hidden_size;
  VT_CHECK(vocab > 0 && H > 0, "dflash2 select-candidates: invalid draft vocab/hidden");
  // SPEC-DFLASH2 W5 (#1314): the codebooks must SPAN the vocabulary the
  // candidates are drawn from. `vocab` is the TARGET's head width
  // (`draft_vocab_size`, which `src/vllm/entrypoints/model_loader.cpp` sets from
  // the target's `lm_head.shape[0]`); the codebook extent comes from the DRAFT,
  // checked at load against the DRAFT config's own `vocab_size`
  // (`qwen3_dflash_weights.cpp`, the `[vocab, rank]` assertion).
  //
  // THE TWO HAVE ALWAYS BEEN ABLE TO DIFFER, on BOTH container arms. Through W5
  // this comment said that on safetensors they "trace back to one `config.json`,
  // so they could not differ"; that is wrong, and W5's fresh review found it
  // (#1314 F7). Even there the two numbers are read from two different files --
  // the DRAFT directory's `config.json` and the TARGET's `lm_head` -- so nothing
  // ever forced them equal. What the GGUF arm changes is how LIKELY the mismatch
  // is: a GGUF draft declares no vocabulary at all and is sized from its own
  // `tokenizer.ggml.tokens`, so a drafter paired with the wrong target reaches
  // here with a codebook SHORTER than the ids that index it, which is an
  // out-of-range read of a 127 MB tensor rather than a wrong answer. Refused by
  // name, and the numbers are quoted because the fix is to pair the draft with
  // the target it was trained on.
  //
  // The comparison is STRICT, which is stricter than the out-of-range read
  // requires. W6 SETTLED why, by reading the beyond-pin oracle rather than
  // guessing (`## Owed` O16, #1314): UPSTREAM HAS NO COMPARISON AT ALL. Its
  // `compute_candidates` masks the head's padded tail to -inf
  // (`num_org_vocab_padding`) before the top-k, so no padded column can survive
  // and every id reaching the codebooks lies in the target's ORIGINAL vocab
  // range; the equality between that range and the draft's `vocab_size` is held
  // by checkpoint pairing and enforced by nothing.
  //
  // So the OPERAND is the thing to watch, not the operator. `vocab` here is the
  // MATERIALISED head width, and upstream's reachable span is that width minus
  // its padding. The two are the same number in this engine and only because of
  // what this engine is: single-device, no tensor parallelism, no head padding,
  // so `Dflash2CandidateArgs::num_org_vocab_padding` is structurally 0 on every
  // path that reaches here. THE FIRST TIME THIS ENGINE PADS AN LM HEAD -- a TP
  // shard, or a checkpoint that ships padded rows -- this comparison must move
  // to `vocab - num_org_vocab_padding` in the SAME edit, or a target vLLM would
  // draft for is refused here. It is not written that way today because a
  // subtraction of a provably-zero quantity is an arm no entry point can reach.
  const int64_t codebook_rows =
      weights.candidate_selector.predecessor_codebook.shape[0];
  VT_CHECK(codebook_rows == vocab,
           "dflash2 select-candidates: the draft's candidate-selector codebooks "
           "hold " + std::to_string(codebook_rows) +
               " rows but the target's head is " + std::to_string(vocab) +
               " wide; the codebooks are indexed by TARGET token id, so this "
               "draft was not trained against this target (SPEC-DFLASH2, #1314)");
  VT_CHECK(block_logits.rank == 2 && block_logits.dtype == vt::DType::kF32 &&
               block_logits.shape[0] == P * nq && block_logits.shape[1] == vocab,
           "dflash2 select-candidates: block_logits must be [num_reqs*(1+k), draft_vocab] f32");
  VT_CHECK(block_hidden.rank == 2 && block_hidden.dtype == vt::DType::kBF16 &&
               block_hidden.shape[0] == P * nq && block_hidden.shape[1] == H,
           "dflash2 select-candidates: block_hidden must be [num_reqs*(1+k), H] bf16 "
           "(the block forward's post-final-norm hidden, captured on the SAME forward)");
  VT_CHECK(static_cast<int64_t>(anchors.size()) == P,
           "dflash2 select-candidates: one anchor token per proposing row");

  // Step 2 — the SAMPLE-ROW gather, upstream's
  // `last_hidden_states[self.sample_indices[:num_sample]]`. Rows +1..+k of each
  // request's block: the k mask positions. Row +0 is the anchor and is skipped,
  // which is `sample_from_anchor=false` (DFlash's layout, unchanged by DFlash2).
  // SPEC-DFLASH2 W8 (#1837): a device vt::IndexSelect instead of the host
  // std::copy loops — the same index arithmetic (src = r*(1+k)+1+j), so the
  // gathered rows are the same rows.
  dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};
  std::vector<int32_t> sample_idx(static_cast<size_t>(P * L));
  for (int64_t r = 0; r < P; ++r)
    for (int64_t j = 0; j < L; ++j)
      sample_idx[static_cast<size_t>(r * L + j)] = static_cast<int32_t>(r * nq + 1 + j);
  dense_attn::DBuf idx_d(d, vt::DType::kI32, {P * L}, sample_idx.data());
  dense_attn::DBuf sample_logits(d, vt::DType::kF32, {P * L, vocab});
  dense_attn::DBuf sample_hidden(d, vt::DType::kBF16, {P * L, H});
  vt::IndexSelect(d.q, sample_logits.t(), block_logits, idx_d.t());
  vt::IndexSelect(d.q, sample_hidden.t(), block_hidden, idx_d.t());

  Dflash2ProposeStateDevice state;
  state.num_reqs = P;
  state.num_steps = L;
  state.top_k = weights.candidate_selector.top_k;
  // Steps 3 and 4. Upstream runs them in this order for a reason the selector
  // depends on: the candidate ids the lattice indexes its codebooks with ARE
  // compute_candidates' output, already rebased and already scaled. (On this
  // device lane "rebased" is a rebase of the structurally-zero
  // `org_vocab_start_index` — see Dflash2CandidateSetDevice.)
  state.candidates = Qwen3DFlash2Model::ComputeCandidatesDevice(sample_logits.t(),
                                                                weights, queue);
  state.edges = Qwen3DFlash2Model::SelectorEdgeScoresDevice(
      state.candidates, sample_hidden.t(), anchors, P, L, weights, config, queue);
  // The postcondition, asserted here rather than at each call site: every
  // (step, predecessor, child) transition of every request is scored.
  VT_CHECK(state.edges.scores.Numel() == P * L * state.top_k * state.top_k,
           "dflash2 select-candidates: the selector must score every "
           "(step, predecessor, child) transition of every request");
  return state;
}

Dflash2ProposeState Dflash2SelectCandidates(const std::vector<float>& block_logits,
                                            const std::vector<float>& block_hidden,
                                            const std::vector<int32_t>& anchors,
                                            int num_reqs, int k,
                                            const Qwen3DFlashWeights& weights,
                                            const HfConfig& config, vt::Queue& queue) {
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
  // The marshaling shell (SPEC-DFLASH2 W8): upload the block outputs — the f32
  // hidden cast back to the exact bf16 bits it came from — run the SAME device
  // pipeline, and download the state the host callers and the D9 flip gates
  // read. Bit-identical: every float op is the same op on the same bits.
  dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};
  dense_attn::DBuf logits_d(d, vt::DType::kF32, {P * nq, vocab}, block_logits.data());
  dense_attn::DBuf hidden_f32(d, vt::DType::kF32, {P * nq, H}, block_hidden.data());
  dense_attn::DBuf hidden_bf16(d, vt::DType::kBF16, {P * nq, H});
  vt::CastBf16(d.q, hidden_bf16.t(), hidden_f32.t());
  const Dflash2ProposeStateDevice dev = Dflash2SelectCandidatesDevice(
      logits_d.t(), hidden_bf16.t(), anchors, num_reqs, k, weights, config, queue);

  Dflash2ProposeState state;
  state.num_reqs = dev.num_reqs;
  state.num_steps = dev.num_steps;
  state.top_k = dev.top_k;
  state.candidates.rows = dev.candidates.rows;
  state.candidates.top_k = dev.candidates.top_k;
  state.candidates.ids.assign(static_cast<size_t>(dev.candidates.rows * dev.candidates.top_k),
                              0);
  state.candidates.values.assign(state.candidates.ids.size(), 0.0f);
  state.edge_scores.assign(static_cast<size_t>(P * L * dev.top_k * dev.top_k), 0.0f);
  vt::Backend& b = d.b;
  b.Copy(d.q, state.candidates.ids.data(), dev.candidates.ids.data,
         state.candidates.ids.size() * sizeof(int64_t));
  b.Copy(d.q, state.candidates.values.data(), dev.candidates.values.data,
         state.candidates.values.size() * sizeof(float));
  b.Copy(d.q, state.edge_scores.data(), dev.edges.scores.data,
         state.edge_scores.size() * sizeof(float));
  b.Synchronize(d.q);
  return state;
}

}  // namespace vllm::v1
