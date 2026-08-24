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
           "dflash2: the target's lm_head was DEQUANTIZED on the way in and this "
           "draft is a DFlash2 draft. The candidate selector's whole input is the "
           "target head's exact top-K over the draft's hidden states, so a head read "
           "through a dequantization produces a DIFFERENT candidate set -- and "
           "nothing raises: the verify is lossless, the emitted tokens are still the "
           "target's, and only acceptance falls. WHAT THIS DOES NOT REFUSE, since "
           "SPEC-DFLASH2-QUANT-LMHEAD (#1628): a head that is quantized and KEPT "
           "PACKED, which the draft computes with through the target's own W4A16 "
           "GEMM and whose top-K is therefore the target's exactly. The two states "
           "are different and the stored dtype cannot tell them apart. Upstream's "
           "guard, which this mirrors, admitted both UnquantizedEmbeddingMethod and "
           "UnquantizedLinearMethod -- the second is the folded-in vllm#52883 fix, "
           "because a ParallelLMHead returns the LINEAR method whenever a quant "
           "config leaves the head itself unquantized -- and it is GONE at the "
           "MERGED vllm-project/vllm#52816 head "
           "b389ac29465b33f9e9c534df221ea3c129e9793f, where compute_candidates "
           "carries no quant-method check and goes through "
           "LogitsProcessor.get_top_k_tokens -> _apply_head -> "
           "lm_head.quant_method.apply (logits_processor.py:241-286,132-142), which "
           "IS the target's own logits path. It survives here for the ONE container "
           "that still widens a head: a GGUF target's q6_K/NVFP4 output.weight "
           "(LoadGgufSharedEmbedAndHeadBf16). Use a safetensors target -- bf16 or "
           "NVFP4 -- or a DFlashDraftModel draft. Issues #1314 and #1628 "
           "(https://github.com/mudler/vllm.cpp/issues/1628).");
}

Dflash2CandidateSetDevice Qwen3DFlash2Model::ComputeCandidatesDevice(
    const vt::Tensor& logits, const Qwen3DFlashWeights& weights, vt::Queue& queue,
    const Dflash2CandidateArgs& args) {
  RefuseQuantizedDflash2LmHead(weights);
  const Dflash2SelectorWeights& sel = weights.candidate_selector;
  VT_CHECK(!sel.Empty(),
           "dflash2 compute_candidates: this draft carries no candidate selector");
  const int64_t K = sel.top_k;
  VT_CHECK(logits.rank == 2 && logits.dtype == DType::kF32,
           "dflash2 compute_candidates: logits must be [rows, vocab] f32");
  const int64_t rows = logits.shape[0];
  const int64_t vocab = logits.shape[1];
  VT_CHECK(rows > 0 && vocab > 0, "dflash2 compute_candidates: rows and vocab must be > 0");
  VT_CHECK(args.num_org_vocab_padding >= 0 && args.org_vocab_start_index >= 0,
           "dflash2 compute_candidates: the org-vocab shard indices must be >= 0");

  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dev_values(d, DType::kF32, {rows, K});
  DBuf dev_ids(d, DType::kI64, {rows, K});
  vt::TopKValuesIndicesArgs tk;
  tk.k = K;
  tk.num_org_vocab_padding = args.num_org_vocab_padding;
  vt::TopKValuesIndices(d.q, dev_values.t(), dev_ids.t(), logits, tk);

  // SPEC-DFLASH2 W8 (#1837): the two output scalars run ON DEVICE, in
  // upstream's order (`get_top_k_tokens`, logits_processor.py:241-286 @ the
  // merged head): `values.float() * output_multiplier` FIRST, then the softcap
  // on the scaled value. Softcapping first would cap a differently-scaled
  // number and change which candidates survive the selector's ordering one
  // step later. The CPU kernels are the exact float arithmetic the pre-W8 host
  // loop ran (`LoadF32 * s`; `c * std::tanh(v / c)`, a commutative multiply),
  // so a host caller downloading these values reads the same bits it always
  // did. The MulScalar runs unconditionally, as the host `v *= mult` did; the
  // softcap is conditioned on `cap > 0`, likewise unchanged.
  vt::MulScalar(d.q, dev_values.t(), dev_values.t(),
                static_cast<double>(sel.output_multiplier));
  const float cap = sel.final_logit_softcapping;
  if (cap > 0.0f)
    vt::SoftCap(d.q, dev_values.t(), dev_values.t(), static_cast<double>(cap));

  // The id rebase (`+= org_vocab_start_index`) stays the HOST caller's step —
  // see Dflash2CandidateSetDevice in the header. The device lane's ids are
  // final because the value is structurally 0 on every shipped path.
  Dflash2CandidateSetDevice out;
  out.rows = rows;
  out.top_k = K;
  out.ids = dev_ids.t();
  out.keep_ids = dev_ids.ReleaseShared();
  out.values = dev_values.t();
  out.keep_values = dev_values.ReleaseShared();
  return out;
}

Dflash2CandidateSet Qwen3DFlash2Model::ComputeCandidates(const std::vector<float>& logits,
                                                        int64_t rows, int64_t vocab,
                                                        const Qwen3DFlashWeights& weights,
                                                        vt::Queue& queue,
                                                        const Dflash2CandidateArgs& args) {
  VT_CHECK(rows > 0 && vocab > 0, "dflash2 compute_candidates: rows and vocab must be > 0");
  VT_CHECK(static_cast<int64_t>(logits.size()) == rows * vocab,
           "dflash2 compute_candidates: logits must be [rows, vocab]");
  // The marshaling shell (SPEC-DFLASH2 W8): upload, run the SAME device core,
  // download. Bit-identical to the pre-W8 body — the top-k op call is
  // unchanged, and the value scalars moved onto device kernels whose float
  // arithmetic equals the old host loop's.
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dev_logits(d, DType::kF32, {rows, vocab}, logits.data());
  const Dflash2CandidateSetDevice dev =
      ComputeCandidatesDevice(dev_logits.t(), weights, queue, args);

  Dflash2CandidateSet out;
  out.rows = rows;
  out.top_k = dev.top_k;
  out.ids.assign(static_cast<size_t>(rows * dev.top_k), 0);
  out.values.assign(static_cast<size_t>(rows * dev.top_k), 0.0f);
  vt::Backend& b = d.b;
  b.Copy(d.q, out.ids.data(), dev.ids.data, out.ids.size() * sizeof(int64_t));
  b.Copy(d.q, out.values.data(), dev.values.data, out.values.size() * sizeof(float));
  b.Synchronize(d.q);
  // The id-space rebase is the HOST caller's step, applied here exactly as
  // before (a rebase of 0 on every shipped path; the synthetic shard-index
  // gates drive it nonzero).
  for (int64_t& id : out.ids) id += args.org_vocab_start_index;
  return out;
}

Qwen3DFlash2Model::Dflash2EdgeScoresDevice Qwen3DFlash2Model::SelectorEdgeScoresDevice(
    const Dflash2CandidateSetDevice& candidates, const vt::Tensor& hidden_bf16,
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
  VT_CHECK(hidden_bf16.rank == 2 && hidden_bf16.dtype == DType::kBF16 &&
               hidden_bf16.shape[0] == B * L && hidden_bf16.shape[1] == H,
           "dflash2 selector: hidden must be [num_reqs*num_steps, H] bf16");
  VT_CHECK(static_cast<int64_t>(anchors.size()) == B,
           "dflash2 selector: one anchor token per request");

  Dev d{vt::GetBackend(queue.device.type), queue};
  // `hidden_projection(hidden_states)` — a bias-free ReplicatedLinear [rank <- H]
  // over the draft's post-final-norm hidden, straight off the DEVICE bf16 bits
  // (SPEC-DFLASH2 W8, #1837): the pre-W8 f32 download + re-upload + CastBf16
  // was an exact bf16->f32->bf16 round trip, so this is the same GEMM input.
  Tensor w_proj = ResidentWeight(d, sel.hidden_projection, {R, H});
  DBuf projected(d, DType::kBF16, {B * L, R});
  vt::MatmulBT(d.q, projected.t(), hidden_bf16, w_proj);
  Tensor projected_3d = Reshape(projected.t(), {B, L, R});

  Tensor ids_3d = candidates.ids;
  ids_3d = Reshape(ids_3d, {B, L, K});
  Tensor unary_3d = candidates.values;
  unary_3d = Reshape(unary_3d, {B, L, K});
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
  vt::Dflash2SelectorEdges(d.q, scores.t(), pred, succ, ids_3d, unary_3d,
                           projected_3d, dev_anchors.t(), args);

  Dflash2EdgeScoresDevice out;
  out.scores = scores.t();
  out.keep = scores.ReleaseShared();
  return out;
}

std::vector<float> Qwen3DFlash2Model::SelectorEdgeScores(
    const Dflash2CandidateSet& candidates, const std::vector<float>& hidden,
    const std::vector<int32_t>& anchors, int64_t num_reqs, int64_t num_steps,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue) {
  const Dflash2SelectorWeights& sel = weights.candidate_selector;
  const int64_t B = num_reqs, L = num_steps;
  const int64_t K = sel.top_k, H = config.hidden_size;
  VT_CHECK(candidates.rows == B * L && candidates.top_k == K,
           "dflash2 selector: the candidate set must be [num_reqs*num_steps, top_k]");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == B * L * H,
           "dflash2 selector: hidden must be [num_reqs*num_steps, H]");
  // The marshaling shell (SPEC-DFLASH2 W8): upload the host candidate set and
  // the f32 hidden (cast back to the exact bf16 bits it came from), run the
  // SAME device core, download the lattice. Bit-identical to the pre-W8 body.
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf hidden_f32(d, DType::kF32, {B * L, H}, hidden.data());
  DBuf hidden_bf16(d, DType::kBF16, {B * L, H});
  vt::CastBf16(d.q, hidden_bf16.t(), hidden_f32.t());
  DBuf dev_ids(d, DType::kI64, {B * L, K}, candidates.ids.data());
  DBuf dev_unary(d, DType::kF32, {B * L, K}, candidates.values.data());
  Dflash2CandidateSetDevice cand_dev;
  cand_dev.rows = candidates.rows;
  cand_dev.top_k = candidates.top_k;
  cand_dev.ids = dev_ids.t();
  cand_dev.keep_ids = dev_ids.ReleaseShared();
  cand_dev.values = dev_unary.t();
  cand_dev.keep_values = dev_unary.ReleaseShared();
  const Dflash2EdgeScoresDevice dev = SelectorEdgeScoresDevice(
      cand_dev, hidden_bf16.t(), anchors, num_reqs, num_steps, weights, config, queue);
  std::vector<float> out(static_cast<size_t>(B * L * K * K), 0.0f);
  vt::Backend& b = d.b;
  b.Copy(d.q, out.data(), dev.scores.data, out.size() * sizeof(float));
  b.Synchronize(d.q);
  return out;
}

}  // namespace vllm
