// DFlash2 candidate selector (SPEC-DFLASH2 W3, #1314). See qwen3_dflash2.h for
// the port note, the upstream anchors and what the moved PR head changed.
#include "vllm/model_executor/models/qwen3_dflash2.h"

#include <cmath>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight/Reshape
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {

using vt::DType;
using vt::Tensor;
using namespace dense_attn;  // Dev, DBuf, ResidentWeight, Reshape

void RefuseQuantizedDflash2LmHead(const Qwen3DFlashWeights& weights) {
  if (!weights.IsDflash2() || !weights.lm_head_dequantized) return;
  VT_CHECK(false,
           "dflash2: the target's lm_head is QUANTIZED and this draft is a DFlash2 "
           "draft. The candidate selector's whole input is the target head's exact "
           "top-K over the draft's hidden states, so a head read through a "
           "dequantization produces a DIFFERENT candidate set -- and nothing raises: "
           "the verify is lossless, the emitted tokens are still the target's, and "
           "only acceptance falls. Upstream refuses the same case by name "
           "(DFlash2Qwen3ForCausalLM.compute_candidates, "
           "vllm/model_executor/models/qwen3_dflash2.py @ vllm-project/vllm#52816 "
           "head 66e5414c6d75a8529473d977f7458c140bbab8a0: \"DFlash2 requires an "
           "unquantized target LM head for candidate TopK\"). Its guard admits both "
           "UnquantizedEmbeddingMethod and UnquantizedLinearMethod -- the second is "
           "the folded-in vllm#52883 fix, because a ParallelLMHead returns the "
           "LINEAR method whenever a quant config leaves the head itself "
           "unquantized -- and this one mirrors that WIDE form: any head whose "
           "weights are readable as dense floats is admitted, whatever loaded them. "
           "Use a target whose lm_head is stored unquantized (every bf16 "
           "safetensors target is), or a DFlashDraftModel draft. Issue #1314 "
           "(https://github.com/mudler/vllm.cpp/issues/1314).");
}

Dflash2CandidateSet Qwen3DFlash2Model::ComputeCandidates(const std::vector<float>& logits,
                                                        int64_t rows, int64_t vocab,
                                                        const Qwen3DFlashWeights& weights,
                                                        vt::Queue& queue,
                                                        const Dflash2CandidateArgs& args) {
  RefuseQuantizedDflash2LmHead(weights);
  const Dflash2SelectorWeights& sel = weights.candidate_selector;
  VT_CHECK(!sel.Empty(),
           "dflash2 compute_candidates: this draft carries no candidate selector");
  const int64_t K = sel.top_k;
  VT_CHECK(rows > 0 && vocab > 0, "dflash2 compute_candidates: rows and vocab must be > 0");
  VT_CHECK(static_cast<int64_t>(logits.size()) == rows * vocab,
           "dflash2 compute_candidates: logits must be [rows, vocab]");
  VT_CHECK(args.num_org_vocab_padding >= 0 && args.org_vocab_start_index >= 0,
           "dflash2 compute_candidates: the org-vocab shard indices must be >= 0");

  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dev_logits(d, DType::kF32, {rows, vocab}, logits.data());
  DBuf dev_values(d, DType::kF32, {rows, K});
  DBuf dev_ids(d, DType::kI64, {rows, K});
  vt::TopKValuesIndicesArgs tk;
  tk.k = K;
  tk.num_org_vocab_padding = args.num_org_vocab_padding;
  vt::TopKValuesIndices(d.q, dev_values.t(), dev_ids.t(), dev_logits.t(), tk);

  Dflash2CandidateSet out;
  out.rows = rows;
  out.top_k = K;
  out.ids.assign(static_cast<size_t>(rows * K), 0);
  out.values.assign(static_cast<size_t>(rows * K), 0.0f);
  dev_ids.Download(d, out.ids.data());
  dev_values.Download(d, out.values.data());

  // The rebase and the two scalars run on the HOST, on rows*K values — 128 at
  // the published shapes (8 query rows x K 16). Upstream does them on device
  // because its candidates never leave it; ours are already here, and a kernel
  // launch per 128 f32 multiplies would cost more than the multiplies. The
  // arithmetic is plain f32 either way, so the placement is not a numerics
  // decision.
  //
  // ORDER IS UPSTREAM'S ORDER and it is load-bearing: `values.float() *
  // output_multiplier` FIRST, then the softcap on the scaled value. Softcapping
  // first would cap a differently-scaled number and change which candidates
  // survive the selector's ordering one step later.
  for (int64_t& id : out.ids) id += args.org_vocab_start_index;
  const float cap = sel.final_logit_softcapping;
  for (float& v : out.values) {
    v *= sel.output_multiplier;
    if (cap > 0.0f) v = std::tanh(v / cap) * cap;
  }
  return out;
}

std::vector<float> Qwen3DFlash2Model::SelectorEdgeScores(
    const Dflash2CandidateSet& candidates, const std::vector<float>& hidden,
    const std::vector<int32_t>& anchors, int64_t num_reqs, int64_t num_steps,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue) {
  const Dflash2SelectorWeights& sel = weights.candidate_selector;
  VT_CHECK(!sel.Empty(),
           "dflash2 selector: this draft carries no candidate selector");
  const int64_t B = num_reqs, L = num_steps;
  const int64_t K = sel.top_k, R = sel.rank, H = config.hidden_size;
  VT_CHECK(B > 0 && L > 0, "dflash2 selector: num_reqs and num_steps must be > 0");
  VT_CHECK(candidates.rows == B * L && candidates.top_k == K,
           "dflash2 selector: the candidate set must be [num_reqs*num_steps, top_k]");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == B * L * H,
           "dflash2 selector: hidden must be [num_reqs*num_steps, H]");
  VT_CHECK(static_cast<int64_t>(anchors.size()) == B,
           "dflash2 selector: one anchor token per request");

  Dev d{vt::GetBackend(queue.device.type), queue};
  // `hidden_projection(hidden_states)` — a bias-free ReplicatedLinear [rank <- H]
  // over the draft's post-final-norm hidden. The hidden arrives f32 because the
  // block forward downloads it that way, and it came FROM a bf16 tensor, so the
  // cast back is exact and this is upstream's bf16 Linear on the same bits.
  DBuf hidden_f32(d, DType::kF32, {B * L, H}, hidden.data());
  DBuf hidden_bf16(d, DType::kBF16, {B * L, H});
  vt::CastBf16(d.q, hidden_bf16.t(), hidden_f32.t());
  Tensor w_proj = ResidentWeight(d, sel.hidden_projection, {R, H});
  DBuf projected(d, DType::kBF16, {B * L, R});
  vt::MatmulBT(d.q, projected.t(), hidden_bf16.t(), w_proj);
  Tensor projected_3d = Reshape(projected.t(), {B, L, R});

  DBuf dev_ids(d, DType::kI64, {B, L, K}, candidates.ids.data());
  DBuf dev_unary(d, DType::kF32, {B, L, K}, candidates.values.data());
  // The anchor ids widen to i64 here because the codebooks are indexed by token
  // id and every id in this op is i64, as upstream's are after `ids.to(int64)`.
  std::vector<int64_t> anchors64(anchors.begin(), anchors.end());
  DBuf dev_anchors(d, DType::kI64, {B}, anchors64.data());

  const int64_t V = sel.predecessor_codebook.shape[0];
  Tensor pred = ResidentWeight(d, sel.predecessor_codebook, {V, R});
  Tensor succ = ResidentWeight(d, sel.successor_codebook, {V, R});

  DBuf scores(d, DType::kF32, {B, L, K, K});
  vt::Dflash2SelectorEdgesArgs args;
  args.top_k = K;
  vt::Dflash2SelectorEdges(d.q, scores.t(), pred, succ, dev_ids.t(), dev_unary.t(),
                           projected_3d, dev_anchors.t(), args);

  std::vector<float> out(static_cast<size_t>(B * L * K * K), 0.0f);
  scores.Download(d, out.data());
  return out;
}

}  // namespace vllm
