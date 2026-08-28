// dots3-note (`Dots3NoteForCausalLM`) — W4a: the FULL-attention layer ON THE
// DECODE PATH. Issue #699, spec `.agents/specs/dots3-note.md` (§4.5 is W3's
// evidence, §4.6 is this brick's, `## Owed` is what stays open).
//
// ─── WHAT THIS IS, AND WHAT IT IS NOT ────────────────────────────────────────
// W3 landed `dots3_note_attn.{h,cpp}`: a portable HOST reference of
// `_forward_note_mla`'s full arm, in double, gated against an INDEPENDENT
// double-precision reference. It was not on the decode path, and it named two
// debts. This TU closes both:
//
//   1. `mla::ForwardMlaAttentionBlock` now CARRIES the three deltas that sit
//      inside it — two `double` scales on `MlaBlockDims`, one optional norm
//      weight and one optional gate weight on `MlaBlockWeights`. The extension
//      is in the seam, not here, because the seam is where they belong; every
//      new field's ABSENT state is its default, so the SACRED DeepSeek-V2 path
//      is byte-identical (measured, spec §4.6).
//   2. `Dots3NoteModel::ForwardDevice` stops refusing for a config whose every
//      layer is FULL attention with a DENSE MLP, and reaches the seam through
//      `ModelRegistry::Forward`. Everything else — the released checkpoint
//      included — still refuses BY NAME, naming the brick that owes it.
//
// ─── W4b-2 ADDED THE SLIDING ARM, AND THE PADDED ROW ─────────────────────────
// The 33 `sliding_attention` layers now run, through the SAME
// `mla::ForwardMlaAttentionBlock` over a SECOND `mla::MlaBlockDims`
// (`Dots3NoteSlidingAttnMlaDims`) whose `sliding_window` becomes the
// `AttentionWindow{W - 1, 0}` pair `vt::MlaDecodeAttention` and
// `vt::MlaPrefillAttention` learned at W4b-2. The physical MLA cache row is the
// PADDED 1088 both classes share, and the full layers read their logical 576
// out of the head of it with `Tensor::Slice(2, 0, ...)` — upstream's
// `Dots3NotePaddedSparseImpl._logical_cache` (attention.py:700-702), and no
// `vt` op changed to make it work.
//
// ─── WHAT IS STILL REFUSED, AND BY WHICH BRICK ───────────────────────────────
//   MoE layers                 W5  — the ungrouped noaux_tc router at 256/8
//   seq_len > index_topk AND
//   the request RESUMES        #1925 — LIFTED at W4b-3c for a SINGLE-SHOT
//                                    prefill, which is now served with the DSA
//                                    selection. What is left is a request with
//                                    CACHED CONTEXT: the indexer's key for a
//                                    token comes from that token's own hidden
//                                    state (deepseek_v2.py:808-810), so a
//                                    resumed step has none for its context and
//                                    would need the indexer's own 128-wide key
//                                    cache — a SECOND attention group on the
//                                    same layers, owned by KV-DSV4-MULTICACHE.
//                                    Only asked of a config that HAS a full
//                                    layer: the sliding layers carry no indexer
//                                    (`self.indexer = None` / `is_sparse =
//                                    False`, model.py:432-434)
//   a windowed prefill with
//   chunked CONTEXT            W4b-3 — refused inside the seam; upstream caps a
//                                    sliding layer's gather at the window
//                                    instead of merging context chunks
//                                    (attention.py:206, :594-654), so there is
//                                    no windowed form of that merge to mirror
//   a KV cache row that
//   disagrees with the config  kept — an engine allocates the cache separately
//   the vision / audio towers  W6 / W7 — never part of the language forward
//   the nextn tail             W10
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// BEYOND-PIN. `dots3_note` does NOT exist at our parity pin `555967922`
// (0.26.0.dev0). Every anchor below was first derived at upstream `06ecec7a84`
// and RE-DERIVED at `bc2d63e650`, which is the revision the row's spec and
// W4b-2 read; naming one number on both sides is the point of an anchor.
// MEASURED rather than asserted: `git diff 06ecec7a84 bc2d63e650 --
// vllm/models/dots3_note/` is EMPTY, and the ONLY delta in
// `vllm/model_executor/models/deepseek_v2.py` is two lines inside
// `DeepseekV2Attention` — `q_lora_rank: int | None` at `:460` and
// `self.q_lora_rank` at `:498` — a class dots3-note does not subclass, and an
// equal-length edit, so every line number below is unmoved.
//
// The previous form of this comment said that deepseek_v2.py diff was EMPTY.
// It is not, and it was not silently wrong for free: a claim of emptiness goes
// stale without a symptom, while naming the two lines makes the next reader's
// check a one-command `git diff` instead of a re-derivation.
//
//   OURS                          <-  UPSTREAM
//   Dots3NoteFullAttnMlaDims      <-  `vllm/models/dots3_note/nvidia/model.py`
//                                     ::Dots3NoteFullAttention.__init__
//                                     (:222-308); the rope rebuild at
//                                     :230-238 forces `rope_type="default"`,
//                                     so there is NO YaRN and NO mscale^2 on
//                                     the softmax scale; the two LoRA scalars
//                                     at :303-307
//   ..k_rope_only_layernorm       <-  model.py:299-301 (built), :160 (applied,
//                                     BEFORE the rope at :167-169)
//   ..g_proj                      <-  model.py:286-298 (built, [num_heads,
//                                     hidden] for `headwise`), :190-197
//                                     (applied)
//   MaterializeDots3NoteDevice    <-  `nvidia/multimodal.py`::Dots3NoteFor-
//                                     CausalLM.hf_to_vllm_mapper (:70-78) +
//                                     `deepseek_v2.py`:1565-1568
//                                     (`mla_params_mapping`, which fuses
//                                     `q_a_proj` + `kv_a_proj_with_mqa` into
//                                     ONE `fused_qkv_a_proj`) +
//                                     `layers/attention/mla_attention.py`
//                                     ::MLAAttention.process_weights_after_
//                                     loading (:1066-1196; the two permutes
//                                     that make W_UV / W_UK_T are :1178 and
//                                     :1180)
//
// THREE of those numbers were RE-DERIVED here rather than copied. This tree's
// existing DeepSeek comments cite `deepseek_v2.py:1812-1820` and
// `mla_attention.py:875-962` for the same two facts, and both are correct only
// at the pin `e24d1b24` those files name. At `06ecec7a84` :1812-1820 is
// expert-count bookkeeping and :875-962 is inside `forward_impl`. Spec R2
// again; `check-symbol-anchors.py` cannot see it (#1139).
//   Dots3NoteModel::ForwardDevice <-  `model.py`::Dots3NoteDecoderLayer
//                                     (:481-547) over ::Dots3NoteModel
//                                     (:549-679), which are
//                                     `DeepseekV32DecoderLayer` /
//                                     `DeepseekV32Model` with the attention
//                                     class swapped — i.e. the same residual
//                                     stream `deepseek_v2.cpp` already runs
//
// The DeepSeek shape is reused rather than re-derived on purpose: upstream's
// own class is `Dots3NoteFullAttention(DeepseekV2MLAAttention)`, so `BuildMla-
// Step` (deepseek_v2.h) is the SAME per-step metadata build, not a lookalike.
// A second copy of it here would be the hand-rolled parallel path AGENTS.md
// forbids.
#include "vllm/model_executor/models/dots3_note.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/linear.h"  // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v2.h"  // MlaStep / BuildMlaStep
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/mla_attention.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // kFusedAddRmsNormStd

namespace vllm {

using vt::DType;
using vt::Tensor;

using namespace dense_attn;  // Dev / DBuf / MakeTensor / Reshape / ResidentWeight

namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::MakeOwned;

// The rope dots3-note's FULL layers run. `Dots3NoteFullAttention.__init__`
// rebuilds `rope_parameters` as `{"rope_type": "default", "rope_theta":
// config.rope_parameters["rope_theta"]}` (model.py:230-238), so it is PLAIN
// rope at theta 8e7 — no YaRN ramp, and therefore no mscale^2 on the softmax
// scale. Getting that wrong is silent, which is why it is one function.
mla::DeepseekYarnRopeParams FullAttnRope(const Dots3NoteParams& p) {
  mla::DeepseekYarnRopeParams r;
  r.yarn = false;
  r.scaling_factor = 1.0;
  r.base = p.full.rope_theta;
  r.rotary_dim = p.full.qk_rope_head_dim;
  r.original_max_position_embeddings = p.max_position_embeddings;
  return r;
}

// The rope dots3-note's SLIDING layers run. `Dots3NoteSlidingAttention.__init__`
// builds `get_rope(..., rope_parameters={"rope_type": "default", "rope_theta":
// config.swa_rope_theta}, is_neox_style=False)` (model.py:401-409 @
// `bc2d63e650`) — the SAME plain form as the full layers at a DIFFERENT theta,
// 5e4 against 8e7. Sharing the full layers' cache here is numerically silent
// and is spec §4 trap 6; it is a separate function so a reader sees the two
// side by side (W4b-2, #699).
mla::DeepseekYarnRopeParams SwaRope(const Dots3NoteParams& p) {
  mla::DeepseekYarnRopeParams r;
  r.yarn = false;
  r.scaling_factor = 1.0;
  r.base = p.swa.rope_theta;
  r.rotary_dim = p.swa.qk_rope_head_dim;
  r.original_max_position_embeddings = p.max_position_embeddings;
  return r;
}

// The absorbed decode forms, exactly `MLAAttention.process_weights_after_
// loading` (mla_attention.py:1066-1196 @ 06ecec7a84; the two permutes are
// :1178 and :1180). Both forms are kept: `kv_b_proj` feeds the
// materialized-MHA prefill, `w_uk_t`/`w_uv` the absorbed MQA decode.
void AbsorbInto(Dots3NoteMlaLayerWeights& w, const mla::MlaBlockDims& d) {
  const int64_t N = d.num_heads, P = d.qk_nope_head_dim;
  const int64_t V = d.v_head_dim, L = d.kv_lora_rank;
  VT_CHECK(w.kv_b_proj.rank == 2 && w.kv_b_proj.shape[0] == N * (P + V) &&
               w.kv_b_proj.shape[1] == L,
           "dots3-note: kv_b_proj must be [num_heads*(qk_nope+v), kv_lora_rank]");
  const mla::AbsorbedKvBProj a = mla::AbsorbKvBProjBf16(
      reinterpret_cast<const uint16_t*>(w.kv_b_proj.bytes.data()), d);
  w.w_uk_t = MakeOwned(DType::kBF16, {N, P, L});
  std::memcpy(w.w_uk_t.bytes.data(), a.w_uk_t.data(), a.w_uk_t.size() * sizeof(uint16_t));
  w.w_uv = MakeOwned(DType::kBF16, {N, L, V});
  std::memcpy(w.w_uv.bytes.data(), a.w_uv.data(), a.w_uv.size() * sizeof(uint16_t));
}

// One tensor's shape, checked BY NAME. A silently truncated or transposed
// weight renders plausible text on a model with no oracle (spec §6.4), so the
// load refuses instead.
void RequireShape(const OwnedTensor& t, const std::string& name,
                  const std::vector<int64_t>& want) {
  bool ok = t.rank == static_cast<int>(want.size());
  for (size_t i = 0; ok && i < want.size(); ++i) ok = t.shape[i] == want[i];
  if (ok) return;
  std::string got = "[";
  for (int i = 0; i < t.rank; ++i)
    got += (i ? ", " : "") + std::to_string(t.shape[i]);
  got += "]";
  std::string exp = "[";
  for (size_t i = 0; i < want.size(); ++i)
    exp += (i ? ", " : "") + std::to_string(want[i]);
  exp += "]";
  VT_CHECK(false, "dots3-note: `" + name + "` has shape " + got + ", expected " + exp +
                      " — refusing rather than reading a truncated weight");
}

// The device-resident views the seam consumes for ONE layer. ResidentWeight
// uploads once on first touch and memoizes on the OwnedTensor.
mla::MlaBlockWeights ResidentMla(Dev d, const Dots3NoteMlaLayerWeights& w,
                                 const Tensor& rope_cache) {
  mla::MlaBlockWeights m;
  m.fused_qkv_a_proj = ResidentWeight(d, w.fused_qkv_a_proj);
  m.q_a_layernorm = ResidentWeight(d, w.q_a_layernorm);
  m.q_b_proj = ResidentWeight(d, w.q_b_proj);
  m.kv_a_layernorm = ResidentWeight(d, w.kv_a_layernorm);
  m.kv_b_proj = ResidentWeight(d, w.kv_b_proj);
  m.w_uk_t = ResidentWeight(d, w.w_uk_t);
  m.w_uv = ResidentWeight(d, w.w_uv);
  m.o_proj = ResidentWeight(d, w.o_proj);
  m.rope_cos_sin_cache = rope_cache;
  // The two dots3-only modules. Setting them is what turns the seam's optional
  // branches on; leaving them empty is what keeps DeepSeek byte-identical.
  m.k_rope_only_layernorm = ResidentWeight(d, w.k_rope_only_layernorm);
  m.attn_gate_proj = ResidentWeight(d, w.g_proj);
  // The DSA indexer's five tensors (W4b-3c, #699). EMPTY on a SLIDING layer,
  // which carries no indexer upstream — and `ResidentWeight` REFUSES an empty
  // weight BY NAME (#1953), so the emptiness has to be checked here rather than
  // relied on to pass through.
  if (!w.indexer_wq_b.Empty()) {
    m.indexer_wq_b = ResidentWeight(d, w.indexer_wq_b);
    m.indexer_wk = ResidentWeight(d, w.indexer_wk);
    m.indexer_weights_proj = ResidentWeight(d, w.indexer_weights_proj);
    m.indexer_k_norm_weight = ResidentWeight(d, w.indexer_k_norm_weight);
    m.indexer_k_norm_bias = ResidentWeight(d, w.indexer_k_norm_bias);
  }
  return m;
}

// A per-step host vector uploaded into a DBuf the step owns. The same shape
// `deepseek_v2.cpp`'s own `UploadInto` has; it is file-local there, and a
// second copy here is three lines against exporting a helper whose only
// property is that it owns a buffer.
template <typename T>
Tensor UploadInto(Dev d, std::vector<DBuf>& owned, DType dt,
                  const std::vector<int64_t>& shape, const std::vector<T>& host) {
  owned.emplace_back(d, dt, shape, host.data());
  return owned.back().t();
}

// How many tokens of request `r` were computed BEFORE this step — the
// discriminator both the sparse route and the refusal turn on, because it is
// what says whether the indexer's own keys are in hand.
//
// DERIVED rather than read, and that distinction cost a real defect in review.
// `CommonAttentionMetadata::num_computed_tokens_cpu` carries exactly this
// quantity, but not every caller in this tree populates it — the dots3-note
// benches do not — and a check that reads an EMPTY vector and defaults to 0
// falls OPEN: it reports "nothing was computed before" for a decode step, and
// the refusal it guards silently stops firing. `seq_lens[r] - query_len[r]` is
// the same number and is derived from two fields the block table and the slot
// mapping already depend on, so it cannot be absent while the step is
// well-formed. The recorded value is preferred when it is present, because a
// caller that sets it is the authority on its own batch.
int64_t Dots3NoteComputedTokens(const v1::CommonAttentionMetadata& am, int r) {
  if (r < static_cast<int>(am.num_computed_tokens_cpu.size())) {
    return am.num_computed_tokens_cpu[static_cast<size_t>(r)];
  }
  if (r + 1 < static_cast<int>(am.query_start_loc.size()) &&
      r < static_cast<int>(am.seq_lens.size())) {
    const int64_t query_len = am.query_start_loc[static_cast<size_t>(r + 1)] -
                              am.query_start_loc[static_cast<size_t>(r)];
    return static_cast<int64_t>(am.seq_lens[static_cast<size_t>(r)]) - query_len;
  }
  return 0;
}

// ─── the SPARSE per-token MQA step (W4b-3c, #699) ───────────────────────────
// Upstream promotes a whole step to per-token MQA when the selection actually
// prunes, and leaves it on the dense MHA prefill when it does not:
//
//   use_dense_mha = prefill_max_seq_len <= self.topk_tokens
//                   (sparse_mla_attention.py:296-299 @ `bc2d63e650`)
//   if is_sparse and num_mha_tokens > 0 and not use_mha:
//       num_mqa_tokens = q.size(0)          (mla_attention.py:829-851)
//
// So this builder produces a SECOND metadata object, used by the FULL layers of
// a step that qualifies, while the SLIDING layers keep the ordinary split
// `BuildMlaStep` produced. The routing is per LAYER KIND, which is what having
// two objects rather than a flag on one expresses.
//
// `Dots3NotePaddedSparseImpl.forward_mqa` builds `cu_seqlens_q = arange(
// num_actual_toks + 1)` with `max_seqlen_q = 1` (attention.py:796-808): one
// query per token, each over its own selected key list. The block table and
// `seq_lens` are therefore PER TOKEN here rather than per request — upstream's
// `req_id_per_token` (attention.py:761) expanded on the host, which is the same
// mapping and one fewer device kernel.
//
// WHAT MAKES IT ELIGIBLE, and the discriminator is the INDEX CACHE rather than
// the sequence length: the indexer's `k` for a token is produced by
// `wk_weights_proj` from that token's hidden state, so a step that computes
// every token of the sequence has every index key in hand and a step that
// resumes does not. Carrying the indexer's own KV cache is
// `KV-DSV4-MULTICACHE` ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)),
// not this row, which is why `Dots3NoteModel::ForwardDevice` still refuses a
// resumed request past `index_topk` BY NAME.
struct Dots3NoteSparseStep {
  bool active = false;
  std::vector<DBuf> owned;
  mla::MlaBlockMetadata meta;
};

// WHETHER THIS STEP TAKES THE SPARSE ROUTE, AND WHY NOT — decided ONCE.
//
// The route and the refusal are the two halves of one decision, and the W4b-3c
// review found them written as two different predicates that did not meet. The
// route died for the WHOLE step as soon as ANY request resumed, while the
// refusal asked `seq_len > index_topk AND computed > 0` PER REQUEST. A batch of
// {one resumed request at or under `index_topk`, one FRESH prompt past it}
// satisfied neither — it passed the refusal and took no sparse route — so it was
// served DENSE, in silence, on a model whose selection prunes. That is
// continuous batching's most ordinary step shape, and every dots3-note device
// case was `num_reqs = 1`, so nothing saw it.
//
// One function now answers both questions, so the two cannot drift again: the
// invariant is that a step which `prunes` either takes the sparse route or is
// REFUSED BY NAME, and there is no third outcome.
struct Dots3NoteSparseEligibility {
  // Some request's context exceeds `index_topk`, so the selection really
  // prunes and dense attention is NOT upstream's answer for this step.
  bool prunes = false;
  // Some request resumes from tokens computed on an earlier step, so its index
  // keys are not in hand. `wk_weights_proj` builds a token's index key from
  // that token's hidden state (deepseek_v2.py:808-810), and the indexer's own
  // 128-wide cache is a SECOND attention group — KV-DSV4-MULTICACHE (#1925).
  bool resumes = false;
  // The metadata is shaped the way the builder needs to read it.
  bool well_formed = false;
  int prunes_req = -1;
  int resumes_req = -1;
  int64_t prunes_len = 0;
  int64_t resumes_from = 0;
  // `use_dense_mha = prefill_max_seq_len <= self.topk_tokens`
  // (sparse_mla_attention.py:296-299 @ `bc2d63e650`), and `mla_attention.py`
  // `:829-851` promotes the WHOLE step when that is false. Upstream can promote
  // a resumed request too, because it caches its index keys; we cannot, which
  // is the one place this mirror is narrower than upstream and the reason
  // `resumes` appears here at all.
  bool Active() const { return well_formed && prunes && !resumes; }
};

Dots3NoteSparseEligibility Dots3NoteSparseEligibilityOf(
    const Dots3NoteParams& p, const v1::CommonAttentionMetadata& am) {
  Dots3NoteSparseEligibility e;
  const int64_t topk = p.index_topk;
  const int num_reqs = am.num_reqs;
  e.well_formed = topk > 0 && num_reqs > 0 &&
                  static_cast<int>(am.query_start_loc.size()) == num_reqs + 1 &&
                  static_cast<int>(am.seq_lens.size()) >= num_reqs;
  // `num_reqs` is authoritative for a WELL-FORMED step, which is the same span
  // the builder walked before this repair, so nothing about which steps route
  // sparsely moves. When the metadata is NOT well formed there is no
  // authoritative count, so every published `seq_lens` entry is scanned — the
  // safe direction, because a step that prunes and is not eligible is refused.
  const size_t scan =
      e.well_formed ? static_cast<size_t>(num_reqs) : am.seq_lens.size();
  for (size_t r = 0; r < scan; ++r) {
    const int64_t sl = am.seq_lens[r];
    const int64_t computed = Dots3NoteComputedTokens(am, static_cast<int>(r));
    if (sl > topk && !e.prunes) {
      e.prunes = true;
      e.prunes_req = static_cast<int>(r);
      e.prunes_len = sl;
    }
    if (computed > 0 && !e.resumes) {
      e.resumes = true;
      e.resumes_req = static_cast<int>(r);
      e.resumes_from = computed;
    }
  }
  return e;
}

Dots3NoteSparseStep BuildDots3NoteSparseStep(Dev d, const Dots3NoteParams& p,
                                             const v1::CommonAttentionMetadata& am,
                                             int64_t T) {
  Dots3NoteSparseStep s;
  const int num_reqs = am.num_reqs;
  // `use_dense_mha`: below the threshold the top-k selects every causal
  // candidate and dense attention IS upstream's answer, so the ordinary split
  // stays — byte-identically, which is what keeps W4a's and W4b-2's gates valid.
  // Above it, a step that cannot be selected for was already REFUSED by name in
  // `ForwardDevice`, so an inactive return here is never a silent fallback.
  if (!Dots3NoteSparseEligibilityOf(p, am).Active()) return s;

  const int64_t cols = am.block_table_num_cols;
  std::vector<int32_t> cu(static_cast<size_t>(num_reqs) + 1, 0);
  std::vector<int32_t> tok_seq_lens(static_cast<size_t>(T), 0);
  std::vector<int32_t> tok_block_table(static_cast<size_t>(T * cols), 0);
  for (int r = 0; r <= num_reqs; ++r) cu[static_cast<size_t>(r)] = am.query_start_loc[static_cast<size_t>(r)];
  int64_t max_seq = 1;
  for (int r = 0; r < num_reqs; ++r) {
    const int64_t o = cu[static_cast<size_t>(r)];
    const int64_t len = cu[static_cast<size_t>(r + 1)] - o;
    for (int64_t i = 0; i < len; ++i) {
      // Position `i` within its own request, because nothing was computed
      // before this step. `seq_lens[t]` is therefore the number of causal keys
      // token `t` has, which is what the MQA kernel bounds its walk by.
      tok_seq_lens[static_cast<size_t>(o + i)] = static_cast<int32_t>(i + 1);
      for (int64_t c = 0; c < cols; ++c) {
        tok_block_table[static_cast<size_t>((o + i) * cols + c)] =
            am.block_table_tensor[static_cast<size_t>(r * cols + c)];
      }
    }
    max_seq = std::max(max_seq, len);
  }

  s.active = true;
  s.meta.num_decode_tokens = T;
  s.meta.decode.block_table =
      UploadInto(d, s.owned, DType::kI32, {T, cols}, tok_block_table);
  s.meta.decode.seq_lens = UploadInto(d, s.owned, DType::kI32, {T}, tok_seq_lens);
  s.meta.decode.max_seq_len = static_cast<int>(max_seq);
  s.meta.indexer_cu_seqlens_q = cu;
  return s;
}

// `Dots3NoteMLP.forward` — merged gate_up GEMM -> SiluAndMul -> down GEMM,
// through the shared `layers::MlpGateUpMethodBase` seam.
DBuf DenseMlp(Dev d, const Dots3NoteDenseMlp& w, const Tensor& dh, int64_t T,
              int64_t H, int64_t I) {
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, dh);
  Tensor wdn = ResidentWeight(d, w.down_proj);
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wdn);
  return out;
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

mla::MlaBlockDims Dots3NoteFullAttnMlaDims(const Dots3NoteParams& p) {
  const Dots3NoteAttnParams& f = p.full;
  mla::MlaBlockDims d;
  d.hidden_size = p.hidden_size;
  d.num_heads = f.num_attention_heads;
  d.qk_nope_head_dim = f.qk_nope_head_dim;
  d.qk_rope_head_dim = f.qk_rope_head_dim;
  d.v_head_dim = f.v_head_dim;
  d.kv_lora_rank = f.kv_lora_rank;
  d.q_lora_rank = f.q_lora_rank;
  d.rms_norm_eps = static_cast<float>(p.rms_norm_eps);
  // §4 item 6, corrected at W1 (#1804): BOTH dots3 geometries are GPT-J.
  d.is_neox_style = f.rope_is_neox_style;
  // §4 trap 5. `ParseDots3NoteParams` already resolved these to
  // sqrt(hidden/rank) or to 1.0 when `apply_mla_qkv_lora_rescale` is false, so
  // the seam gets the value upstream computes and never re-derives it.
  d.q_lora_scale = f.q_lora_scale;
  d.kv_lora_scale = f.kv_lora_scale;
  // The DSA "Lightning Indexer" (W4b-3c, #699). Only the FULL layers carry one
  // — `Dots3NoteSlidingAttnMlaDims` below deliberately leaves the group zero,
  // which is upstream's `self.indexer = None` / `is_sparse = False`
  // (model.py:432-434 @ `bc2d63e650`).
  d.index_n_heads = p.index_n_heads;
  d.index_head_dim = p.index_head_dim;
  d.index_topk = p.index_topk;
  // `is_neox_style = not indexer_rope_interleave` (deepseek_v2.py:1159), which
  // is INDEPENDENT of `is_neox_style` above — dots3-note's main MLA rope is
  // GPT-J on both geometries while the indexer follows the config flag.
  d.indexer_rope_is_neox_style = p.indexer_rope_is_neox_style();
  const mla::DeepseekYarnRopeParams rope = FullAttnRope(p);
  d.scale = mla::MlaAttentionScale(d, rope);
  d.Validate();
  return d;
}

mla::MlaBlockDims Dots3NoteSlidingAttnMlaDims(const Dots3NoteParams& p) {
  const Dots3NoteAttnParams& w = p.swa;
  mla::MlaBlockDims d;
  d.hidden_size = p.hidden_size;
  d.num_heads = w.num_attention_heads;
  d.qk_nope_head_dim = w.qk_nope_head_dim;
  d.qk_rope_head_dim = w.qk_rope_head_dim;
  d.v_head_dim = w.v_head_dim;
  d.kv_lora_rank = w.kv_lora_rank;
  d.q_lora_rank = w.q_lora_rank;
  d.rms_norm_eps = static_cast<float>(p.rms_norm_eps);
  d.is_neox_style = w.rope_is_neox_style;
  d.q_lora_scale = w.q_lora_scale;
  d.kv_lora_scale = w.kv_lora_scale;
  // `scale=qk_head_dim**-0.5` (model.py:446) — the SLIDING arm builds
  // `MLAAttention` with the bare inverse square root and NO rope_scaling block
  // at all, so there is no YaRN ramp and no mscale^2. Passing the rope through
  // `MlaAttentionScale` yields the same number because `SwaRope().yarn` is
  // false, and it is routed that way rather than written by hand so the full
  // and sliding arms cannot drift apart on the one factor that is silent.
  d.scale = mla::MlaAttentionScale(d, SwaRope(p));
  // `sliding_window=config.sliding_window_size` (model.py:457) — 513.
  d.sliding_window = w.sliding_window;
  d.Validate();
  return d;
}

int64_t Dots3NoteDenseEquivalentMaxSeqLen(const Dots3NoteParams& params) {
  return params.index_topk;
}

std::string Dots3NoteDeviceRefusal(const Dots3NoteParams& p) {
  for (size_t l = 0; l < p.layer_types.size(); ++l) {
    if (p.is_moe_layer(static_cast<int64_t>(l))) {
      return "layer " + std::to_string(l) +
             " is a MoE layer — the ungrouped noaux_tc router at " +
             std::to_string(p.n_routed_experts) + "/" +
             std::to_string(p.num_experts_per_tok) + " plus the shared expert is W5";
    }
  }
  // ─── LIFTED at W4b-2 (#699): the sliding layer and the PADDED row ─────────
  // W4a refused both here. Both now run.
  //
  // The SLIDING layer runs through the same `mla::ForwardMlaAttentionBlock`
  // over `Dots3NoteSlidingAttnMlaDims`, whose `sliding_window` reaches
  // `vt::MlaDecodeAttention` and `vt::MlaPrefillAttention` as the
  // `AttentionWindow{W - 1, 0}` pair upstream hands FlashAttention
  // (attention.py:300 @ bc2d63e650).
  //
  // The PADDED physical row runs because the MLA cache ops are STRIDE-DRIVEN:
  // `Tensor::Slice(2, 0, logical)` shrinks `shape[2]` and KEEPS both leading
  // strides (tensor.cpp), which IS upstream's `kv_cache[..., : self.head_size]`
  // (`Dots3NotePaddedSparseImpl._logical_cache`, attention.py:700-702). The
  // narrowing is ONE line in the forward and no `vt` op changed. W4b-1 first
  // claimed the ops address the cache contiguously; that was false, and the
  // correction is spec §4.7.
  //
  // WHAT STILL REFUSES here is only what has no upstream form to mirror yet.

  // The nextn tail. `Dots3NoteMTPModel` is deliberately not registered and the
  // backbone forward has no place to put an extra block, so a checkpoint that
  // ships one is refused rather than silently having it enumerated, loaded and
  // never run.
  if (p.num_nextn_predict_layers > 0) {
    return "the checkpoint ships " + std::to_string(p.num_nextn_predict_layers) +
           " nextn layer(s) — `Dots3NoteMTPModel` over the speculator seam is W10";
  }
  return "";
}

Dots3NoteDeviceWeights MaterializeDots3NoteDevice(
    const std::vector<SafetensorsFile>& shards, const Dots3NoteParams& p) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& n : shard.Names()) where.emplace(n, &shard);
  const TensorResolver get = [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "dots3-note: tensor not found: " + name);
    return it->second->Get(name);
  };

  Dots3NoteDeviceWeights w;
  w.mla = Dots3NoteFullAttnMlaDims(p);
  w.swa_mla = Dots3NoteSlidingAttnMlaDims(p);
  const int64_t H = p.hidden_size, V = p.vocab_size, I = p.intermediate_size;
  // Which geometries this config actually uses. A rope cache is 2 * 64 bytes
  // per position over `max_position_embeddings` — 64 MiB each at the released
  // 524288 — so the unused one is never built (W4b-2, #699).
  bool any_full = false, any_sliding = false;
  for (const Dots3NoteLayerKind k : p.layer_types) {
    if (k == Dots3NoteLayerKind::kSlidingAttention) {
      any_sliding = true;
    } else {
      any_full = true;
    }
  }

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  RequireShape(w.embed_tokens, "model.embed_tokens.weight", {V, H});
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  RequireShape(w.final_norm, "model.norm.weight", {H});
  if (!p.tie_word_embeddings) {
    // [vocab, H] on disk -> [H, vocab] Matmul-B.
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  // `_compute_cos_sin_cache` over the whole positional range, once per model —
  // and ONCE PER GEOMETRY, because the two thetas differ (8e7 against
  // `swa_rope_theta` 5e4, model.py:230-238 / :401-409).
  const auto build_rope = [&](const mla::DeepseekYarnRopeParams& rp) {
    const std::vector<float> cache =
        mla::BuildDeepseekRopeCosSinCache(rp, p.max_position_embeddings);
    OwnedTensor t =
        MakeOwned(DType::kBF16, {p.max_position_embeddings, rp.rotary_dim});
    auto* dst = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (size_t i = 0; i < cache.size(); ++i) dst[i] = vt::F32ToBF16(cache[i]);
    return t;
  };
  if (any_full) w.rope_cos_sin_cache = build_rope(FullAttnRope(p));
  if (any_sliding) w.swa_rope_cos_sin_cache = build_rope(SwaRope(p));

  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string pre = "model.layers." + std::to_string(l) + ".";
    const std::string sa = pre + "self_attn.";
    Dots3NoteLayerDeviceWeights& lw = w.layers[static_cast<size_t>(l)];
    // `attention_cls = Dots3NoteSlidingAttention if config.layer_types[
    // layer_idx] == "sliding_attention" else Dots3NoteFullAttention`
    // (model.py:501-505 @ bc2d63e650). EVERY shape below is this choice's,
    // which is why the kind is resolved before the first tensor is read.
    lw.kind = p.kind_of(l);
    const bool sliding = lw.kind == Dots3NoteLayerKind::kSlidingAttention;
    const mla::MlaBlockDims& d = sliding ? w.swa_mla : w.mla;
    const int64_t N = d.num_heads, R = d.qk_rope_head_dim, L = d.kv_lora_rank;
    const int64_t QL = d.q_lora_rank;
    lw.input_layernorm = LoadBf16Direct(get, pre + "input_layernorm.weight");
    RequireShape(lw.input_layernorm, pre + "input_layernorm.weight", {H});
    lw.post_attention_layernorm =
        LoadBf16Direct(get, pre + "post_attention_layernorm.weight");
    RequireShape(lw.post_attention_layernorm, pre + "post_attention_layernorm.weight", {H});

    // `mla_params_mapping = [("fused_qkv_a_proj", "q_a_proj", 0),
    // ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1)]`
    // (deepseek_v2.py:1565-1568 @ 06ecec7a84): ONE merged owner whose row
    // blocks are [q_lora_rank | kv_lora_rank + qk_rope_head_dim].
    lw.attn.fused_qkv_a_proj = LoadMergedBf16RawNK(
        get, {sa + "q_a_proj.weight", sa + "kv_a_proj_with_mqa.weight"});
    RequireShape(lw.attn.fused_qkv_a_proj, sa + "{q_a_proj,kv_a_proj_with_mqa}.weight",
                 {QL + L + R, H});
    lw.attn.q_a_layernorm = LoadBf16Direct(get, sa + "q_a_layernorm.weight");
    RequireShape(lw.attn.q_a_layernorm, sa + "q_a_layernorm.weight", {QL});
    lw.attn.q_b_proj = LoadMergedBf16RawNK(get, {sa + "q_b_proj.weight"});
    RequireShape(lw.attn.q_b_proj, sa + "q_b_proj.weight", {N * d.qk_head_dim(), QL});
    lw.attn.kv_a_layernorm = LoadBf16Direct(get, sa + "kv_a_layernorm.weight");
    RequireShape(lw.attn.kv_a_layernorm, sa + "kv_a_layernorm.weight", {L});
    lw.attn.kv_b_proj = LoadMergedBf16RawNK(get, {sa + "kv_b_proj.weight"});
    RequireShape(lw.attn.kv_b_proj, sa + "kv_b_proj.weight",
                 {N * (d.qk_nope_head_dim + d.v_head_dim), L});
    lw.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
    RequireShape(lw.attn.o_proj, sa + "o_proj.weight", {H, N * d.v_head_dim});
    // The two dots3-only modules (model.py:292-301).
    lw.attn.k_rope_only_layernorm =
        LoadBf16Direct(get, sa + "k_rope_only_layernorm.weight");
    RequireShape(lw.attn.k_rope_only_layernorm, sa + "k_rope_only_layernorm.weight", {R});
    lw.attn.g_proj = LoadMergedBf16RawNK(get, {sa + "g_proj.weight"});
    RequireShape(lw.attn.g_proj, sa + "g_proj.weight", {N, H});
    // The DSA indexer, on FULL-attention layers ONLY (W4b-3c, #699).
    // `Dots3NoteSlidingAttention` sets `self.indexer = None` and
    // `is_sparse = False` (model.py:432-434 @ `bc2d63e650`) and its checkpoint
    // carries no `indexer.*` tensor for those layers, so reading one would fail
    // at load rather than produce a wrong answer. The two `k_norm` tensors are
    // a `LayerNorm(head_dim, eps=1e-6)` (deepseek_v2.py:708) — weight AND bias,
    // not the RmsNorm every other norm on this model is.
    if (!sliding) {
      const int64_t IH = p.index_n_heads, ID = p.index_head_dim;
      lw.attn.indexer_wq_b = LoadMergedBf16RawNK(get, {sa + "indexer.wq_b.weight"});
      RequireShape(lw.attn.indexer_wq_b, sa + "indexer.wq_b.weight", {IH * ID, QL});
      lw.attn.indexer_wk = LoadMergedBf16RawNK(get, {sa + "indexer.wk.weight"});
      RequireShape(lw.attn.indexer_wk, sa + "indexer.wk.weight", {ID, H});
      lw.attn.indexer_weights_proj =
          LoadMergedBf16RawNK(get, {sa + "indexer.weights_proj.weight"});
      RequireShape(lw.attn.indexer_weights_proj, sa + "indexer.weights_proj.weight", {IH, H});
      lw.attn.indexer_k_norm_weight = LoadBf16Direct(get, sa + "indexer.k_norm.weight");
      RequireShape(lw.attn.indexer_k_norm_weight, sa + "indexer.k_norm.weight", {ID});
      lw.attn.indexer_k_norm_bias = LoadBf16Direct(get, sa + "indexer.k_norm.bias");
      RequireShape(lw.attn.indexer_k_norm_bias, sa + "indexer.k_norm.bias", {ID});
    }
    AbsorbInto(lw.attn, d);

    lw.mlp.gate_up_proj = LoadMergedBf16RawNK(
        get, {pre + "mlp.gate_proj.weight", pre + "mlp.up_proj.weight"});
    RequireShape(lw.mlp.gate_up_proj, pre + "mlp.{gate_proj,up_proj}.weight", {2 * I, H});
    lw.mlp.down_proj = LoadMergedBf16RawNK(get, {pre + "mlp.down_proj.weight"});
    RequireShape(lw.mlp.down_proj, pre + "mlp.down_proj.weight", {H, I});
  }
  w.present = true;
  return w;
}

// ─────────────────────────────────────────────────────────────────────────────

ForwardLogits Dots3NoteModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Dots3NoteWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  const Dots3NoteParams& p = weights.params;
  // The scope boundary, stated as a refusal rather than as a comment. Every
  // branch names ONE unrepresentable feature and the brick that owes it; the
  // released `dots-studio/dots3-note-prev` config trips the first one at layer
  // 2 and the second at layer 1, so nothing about its behaviour changed at
  // W4a.
  const std::string why = Dots3NoteDeviceRefusal(p);
  VT_CHECK(why.empty(),
           "Dots3NoteForCausalLM forward: not ported — " + why +
               ". W4a/W4b-2 cover BOTH attention geometries — full and "
               "sliding-window — with a dense MLP; the MoE is W5, the "
               "vision/audio towers are W6/W7, the nextn tail is W10. See "
               ".agents/specs/dots3-note.md and issue #699.");
  VT_CHECK(weights.materialized && weights.device.present,
           "Dots3NoteForCausalLM forward: the language tower was not "
           "materialized — the loader only materializes a config the device "
           "forward can run. See .agents/specs/dots3-note.md and issue #699.");
  // The DSA lightning indexer's SELECTION is ON the device path since W4b-3c,
  // and what is left here is the ONE case it cannot serve.
  //
  // While every request's context fits in `index_topk` the top-k picks every
  // causal candidate and dense attention IS upstream's answer
  // (`use_dense_mha = prefill_max_seq_len <= self.topk_tokens`,
  //  sparse_mla_attention.py:296-299 @ `bc2d63e650`). Past that the selection
  // runs, and W4b-3c wires it: `BuildDots3NoteSparseStep` promotes the step to
  // per-token MQA and `mla::ForwardMlaAttentionBlock` computes the selection
  // inside the seam.
  //
  // WHAT STILL REFUSES is a STEP that needs a selection it cannot compute, and
  // the discriminator is the INDEX KV CACHE rather than the sequence length.
  // The indexer's `k` for a token comes from `wk_weights_proj` over that
  // token's hidden state (deepseek_v2.py:808-810), so a step that computes
  // every token of every sequence has every index key in hand, and a step in
  // which ANY request resumes does not — it would need the indexer's own
  // 128-wide cache, which is a second attention group on the same layers. That
  // is `KV-DSV4-MULTICACHE`
  // ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)), not this row,
  // and duplicating it here is the failure this refusal exists to prevent.
  // `CommonAttentionMetadata::num_computed_tokens_cpu` already carries the
  // discriminator.
  //
  // THE UNIT IS THE STEP, and the W4b-3c review is why that word is here. The
  // sparse route dies for the whole step the moment one request resumes,
  // because the indexer's key space is the step's own tokens
  // (`indexer_cu_seqlens_q`); a refusal asked PER REQUEST therefore left a gap
  // it could not see, and `{resumed request under index_topk, fresh prompt past
  // it}` fell into it and was served DENSE with no message. The refusal is now
  // the exact COMPLEMENT of `Dots3NoteSparseEligibility::Active`, taken from the
  // same function, so a step that prunes is either served sparsely or refused
  // and there is no third outcome.
  //
  // THIS IS THE CONSERVATIVE HALF OF THE REPAIR. Routing PER REQUEST — serving
  // the fresh requests sparsely while the resumed ones ride the dense path they
  // are entitled to — is expressible, but it rescues only the sub-case in which
  // every resumed request is at or under `index_topk`; a resumed request past it
  // still needs #1925's index cache. BOTH sub-cases are ordinary — at the
  // released `index_topk` of 2048 a co-scheduled decode under 2048 tokens is at
  // least as common as one over it — so this refusal turns away a COMMON
  // serving shape rather than a corner. It is recorded under `## Owed` in
  // `.agents/specs/dots3-note.md` rather than written here, because refusing is
  // correct and serving the wrong tokens is not.
  //
  // W4b-2's narrowing of WHO is asked stands, and it is upstream's own
  // statement rather than a convenience: `Dots3NoteSlidingAttention` sets
  // `self.indexer = None` and `is_sparse = False` (model.py:432-434), so a
  // sliding layer has no selection to get wrong and a config with no FULL layer
  // has no DSA anywhere. The released checkpoint has 13 full layers and is
  // unaffected.
  const bool has_full_layer =
      std::any_of(p.layer_types.begin(), p.layer_types.end(), [](Dots3NoteLayerKind k) {
        return k == Dots3NoteLayerKind::kFullAttention;
      });
  const int64_t topk = Dots3NoteDenseEquivalentMaxSeqLen(p);
  if (has_full_layer) {
    const Dots3NoteSparseEligibility elig =
        Dots3NoteSparseEligibilityOf(p, attn_meta);
    VT_CHECK(
        !elig.prunes || elig.Active(),
        "Dots3NoteForCausalLM forward: request " + std::to_string(elig.prunes_req) +
            " needs " + std::to_string(elig.prunes_len) +
            " keys against `index_topk` " + std::to_string(topk) +
            ", so this STEP's DSA selection PRUNES (model.py:171) — and " +
            (elig.resumes
                 ? ("request " + std::to_string(elig.resumes_req) +
                    " in the SAME step resumes from " +
                    std::to_string(elig.resumes_from) +
                    " already-computed tokens, whose index keys are not in hand")
                 : std::string("this step's attention metadata is not shaped the "
                               "way the sparse route reads it")) +
            ". The sparse route is a property of the STEP, not of one request "
            "(`use_dense_mha`, sparse_mla_attention.py:296-299), so it is "
            "unavailable to EVERY request here, and serving the step would "
            "serve DENSE attention on a sparse model in silence. The indexer's "
            "own 128-wide key cache, which a resumed step would have to read, "
            "is a SECOND attention group owned by KV-DSV4-MULTICACHE (#1925). A "
            "step whose requests are all single-shot prefills is served "
            "sparsely, and a step in which nothing exceeds `index_topk` keeps "
            "the dense answer. Refusing rather than serving dense attention on "
            "a sparse model. See issue #699.");
  }

  const Dots3NoteDeviceWeights& dw = weights.device;
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size, vocab = p.vocab_size;
  VT_CHECK(T > 0, "dots3-note forward: empty batch");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "dots3-note forward: positions must have one entry per token");
  VT_CHECK(static_cast<int64_t>(attn_kv.size()) == p.num_hidden_layers,
           "dots3-note forward: one KV cache per backbone layer is required");

  // ── embed ─────────────────────────────────────────────────────────────────
  DBuf hidden_buf(d, DType::kBF16, {T, H});
  {
    DBuf ids(d, DType::kI32, {T}, token_ids.data());
    Tensor tab = ResidentWeight(d, dw.embed_tokens, {vocab, H});
    Tensor h = hidden_buf.t();
    Tensor idt = ids.t();
    vt::Embedding(d.q, h, tab, idt);
  }

  // ── the residual stream (deepseek_v2.py:1262-1345, unchanged by dots3) ────
  Tensor hidden = hidden_buf.t();
  std::shared_ptr<void> hidden_hold;
  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  const int64_t block_size = attn_kv[0].block_size;
  MlaStep step =
      BuildMlaStep(d, positions, attn_meta, block_size, p.max_position_embeddings);
  // The SECOND metadata object (W4b-3c, #699). Inactive unless the step both
  // qualifies for upstream's sparse promotion and has every index key in hand;
  // when it is active the FULL layers use it and the SLIDING layers keep
  // `step.meta`, because the routing is a property of the LAYER KIND.
  const Dots3NoteSparseStep sparse = BuildDots3NoteSparseStep(d, p, attn_meta, T);
  // One resident rope cache per GEOMETRY, and each is made resident ONLY when
  // the config has a layer of that kind. `MaterializeDots3NoteDevice` leaves
  // the other one EMPTY to avoid building a 64 MiB table nothing reads, and
  // since #1953 `ResidentWeight` REFUSES an empty weight BY NAME rather than
  // aliasing a null host pointer — so the guard is the contract, not an
  // optimization. A layer only ever reads its own kind's cache, so the one
  // that stays an empty `Tensor` here is never dereferenced.
  Tensor rope_full{};
  Tensor rope_swa{};
  if (!dw.rope_cos_sin_cache.Empty()) rope_full = ResidentWeight(d, dw.rope_cos_sin_cache);
  if (!dw.swa_rope_cos_sin_cache.Empty()) {
    rope_swa = ResidentWeight(d, dw.swa_rope_cos_sin_cache);
  }
  // `MlaStep::rope_cache` is deliberately LEFT NULL here, and that is a
  // statement rather than an omission. It exists for the one-geometry models —
  // `deepseek_v2.cpp:500`, `minicpm3.cpp:200`, `kimi_linear_device.cpp:2210`
  // each dereference it — because one per-MODEL table serves every layer there.
  // dots3-note has TWO tables, and which one a layer reads is a property of the
  // LAYER (`layer_rope` below), so no single per-model value is correct.
  // Assigning `&rope_full` here would hand a future `*step.rope_cache` reader
  // the FULL arm's rope on a sliding layer — a silently wrong answer — or, on a
  // pure-SWA config, an EMPTY `Tensor` whose failure surfaces somewhere else
  // entirely. A null pointer fails at the first read, loudly, in this function.
  v1::TritonMLAImpl impl;
  const float eps = static_cast<float>(p.rms_norm_eps);
  // The PHYSICAL MLA cache row both attention classes share:
  // `physical_head_size = swa_kv_lora_rank + swa_qk_rope_head_dim`
  // (`Dots3NotePaddedMLAAttention.get_kv_cache_spec`, model.py:204-216, fed at
  // :283). 1088 on the released config, against the full layers' logical 576.
  const int64_t physical_row = p.physical_latent_row();

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const Dots3NoteLayerDeviceWeights& lw = dw.layers[static_cast<size_t>(l)];
    const PagedKvCache& kv = attn_kv[static_cast<size_t>(l)];
    const bool sliding = lw.kind == Dots3NoteLayerKind::kSlidingAttention;
    const mla::MlaBlockDims& ld = sliding ? dw.swa_mla : dw.mla;
    const Tensor& layer_rope = sliding ? rope_swa : rope_full;
    // The PER-STEP cache-row check STAYS, and it is not the config-level one
    // W4b-2 lifted: an engine allocates the KV cache separately from the config
    // it was built from, so a row that disagrees is an input this forward can
    // see and the config cannot. It now compares against the PHYSICAL row,
    // because that is what the allocator is told to give
    // (`MakeDots3NoteKVCache`).
    VT_CHECK(kv.num_kv_heads == 1 && kv.head_size == physical_row,
             "dots3-note forward: the MLA cache row is " +
                 std::to_string(kv.head_size) + " but this model's PHYSICAL row is " +
                 std::to_string(physical_row) +
                 " (`physical_head_size = swa_kv_lora_rank + swa_qk_rope_head_dim`, "
                 "model.py:204-216) — refusing rather than reading a wrong stride");
    Tensor kv_cache = MakeTensor(kv.data, kv.dtype, d.q.device,
                                 {kv.num_blocks, kv.block_size, physical_row});
    // `Dots3NotePaddedSparseImpl._logical_cache` (attention.py:700-702):
    // `kv_cache[..., : self.head_size]`. `Tensor::Slice` shrinks `shape[2]` and
    // KEEPS both leading strides, and every MLA cache op reads those strides
    // from the tensor, so the narrowed view addresses the padded rows correctly
    // with no op change. A SLIDING layer's logical row IS the physical one by
    // construction (the padding exists for the full layers), so the slice is
    // the identity there and is written unconditionally rather than branched.
    //
    // THIS ONE IS UNREACHABLE, and saying so is cheaper than leaving the next
    // reader to discover it with a mutation. Unlike the cache-row check above,
    // both sides come from the SAME parsed config: `physical_latent_row()` IS
    // `swa.latent_row()` (`dots3_note.h:192`), so on a SLIDING layer the
    // comparison is an identity; and on a FULL layer
    // `ParseDots3NoteParams` has already refused
    // `physical_latent_row() < full.latent_row()` at load
    // (`dots3_note.cpp:389`, gated at
    // `tests/vllm/models/test_dots3_note_scaffold.cpp:720-722`). No input the
    // loader accepts can make it fire, which is why the W4b-2 review's
    // mutation of it SURVIVED. It is kept as the executable spelling of
    // upstream's own `assert` and is listed under `## Owed` as an untested
    // assertion rather than presented as a gated refusal.
    VT_CHECK(ld.head_size() <= physical_row,
             "dots3-note forward: a layer reads " + std::to_string(ld.head_size()) +
                 " latent lanes but the physical row is only " +
                 std::to_string(physical_row) +
                 " — upstream asserts `physical_head_size >= self.head_size` "
                 "(model.py:210)");
    kv_cache = kv_cache.Slice(2, 0, ld.head_size());

    // The residual add + RMSNorm goes through the SHARED `vt::FusedChain`
    // catalog (AGENTS.md: route model fusion through vt::FusedChain), with the
    // standalone call as the byte-exact rollback — the same shape
    // `deepseek_v2.cpp`'s decoder layer uses, and the same recipe, because
    // dots3-note's decoder layer IS `DeepseekV32DecoderLayer` upstream.
    DBuf dhn(d, DType::kBF16, {T, H});
    Tensor w_in = ResidentWeight(d, lw.input_layernorm, {H});
    Tensor dhn_t = dhn.t(), res_t = res.t();
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dhn_t, hidden, w_in, &res_t, vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dhn_t, hidden, w_in, vt::RmsNormArgs{eps, false}, &res_t);
    }

    DBuf attn(d, DType::kBF16, {T, H});
    Tensor attn_t = attn.t();
    const mla::MlaBlockWeights mw = ResidentMla(d, lw.attn, layer_rope);
    // A SPARSE step routes the FULL layers through per-token MQA with the DSA
    // selection, and leaves the SLIDING layers on the ordinary split — which is
    // upstream's shape, because only the full layers carry an indexer. When the
    // step is not sparse both kinds take `step.meta` and nothing about W4a's or
    // W4b-2's path moves.
    const mla::MlaBlockMetadata& lmeta =
        (sparse.active && !sliding) ? sparse.meta : step.meta;
    mla::ForwardMlaAttentionBlock(d, ld, mw, dhn.t(), step.positions, kv_cache,
                                  step.slot_mapping, lmeta, impl, attn_t);

    DBuf dh2(d, DType::kBF16, {T, H});
    Tensor w_post = ResidentWeight(d, lw.post_attention_layernorm, {H});
    Tensor dh2_t = dh2.t();
    Tensor attn_ro = attn.t();
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dh2_t, attn_ro, w_post, &res_t, vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dh2_t, attn_ro, w_post, vt::RmsNormArgs{eps, false}, &res_t);
    }

    DBuf mlp = DenseMlp(d, lw.mlp, dh2.t(), T, H, p.intermediate_size);
    auto* held = new DBuf(std::move(mlp));
    hidden = held->t();
    hidden_hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
  }

  // ── final norm + lm_head ──────────────────────────────────────────────────
  Tensor w_fn = ResidentWeight(d, dw.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  Tensor dnorm_t = dnorm.t(), res_t = res.t();
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm_t, hidden, w_fn, &res_t, vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm_t, hidden, w_fn, vt::RmsNormArgs{eps, false}, &res_t);
  }

  const bool tied = p.tie_word_embeddings || dw.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, dw.embed_tokens, {vocab, H})
                   : ResidentWeight(d, dw.lm_head);

  const bool do_gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  const int64_t n_idx = static_cast<int64_t>(logits_indices.size());
  DBuf dgather(d, DType::kBF16, {do_gather ? n_idx : int64_t{0}, H});
  Tensor src = dnorm.t();
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = do_gather ? n_idx : T;
  DBuf logits(d, DType::kF32, {n_out, vocab});
  Tensor logits_t = logits.t();
  if (tied) {
    vt::MatmulBT(d.q, logits_t, src, lm);
  } else {
    vt::Matmul(d.q, logits_t, src, lm);
  }
  return WrapDeviceLogits(std::move(logits), n_out, vocab);
}

}  // namespace vllm
