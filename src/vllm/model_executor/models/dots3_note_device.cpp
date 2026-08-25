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
// ─── WHAT IS STILL REFUSED, AND BY WHICH BRICK ───────────────────────────────
//   sliding_attention layers   W4b — the windowed metadata, the gather, the
//                                    score mask, the padded/heterogeneous KV
//                                    spec (spec §2.3)
//   MoE layers                 W5  — the ungrouped noaux_tc router at 256/8
//   seq_len > index_topk       W4b — the DSA lightning indexer's SELECTION is
//                                    not on the device path, so dense
//                                    attention is only the same answer while
//                                    the top-k selects every causal candidate
//   the vision / audio towers  W6 / W7 — never part of the language forward
//   the nextn tail             W10
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// BEYOND-PIN. `dots3_note` does NOT exist at our parity pin `555967922`
// (0.26.0.dev0). Every anchor below was RE-DERIVED at upstream `origin/main` =
// `06ecec7a84`; `git log 06ecec7a84..origin/main -- vllm/models/dots3_note/`
// and `.. -- vllm/model_executor/models/deepseek_v2.py` are both EMPTY at the
// time of writing, so the same numbers hold at the newer head.
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
  return m;
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
  const mla::DeepseekYarnRopeParams rope = FullAttnRope(p);
  d.scale = mla::MlaAttentionScale(d, rope);
  d.Validate();
  return d;
}

int64_t Dots3NoteDenseEquivalentMaxSeqLen(const Dots3NoteParams& params) {
  return params.index_topk;
}

std::string Dots3NoteDeviceRefusal(const Dots3NoteParams& p) {
  for (size_t l = 0; l < p.layer_types.size(); ++l) {
    if (p.layer_types[l] == Dots3NoteLayerKind::kSlidingAttention) {
      return "layer " + std::to_string(l) +
             " is `sliding_attention` — the sliding-window MLA (the windowed "
             "metadata, the KV gather, the score mask and the padded/"
             "heterogeneous KV spec of `nvidia/attention.py`) is W4b";
    }
    if (p.is_moe_layer(static_cast<int64_t>(l))) {
      return "layer " + std::to_string(l) +
             " is a MoE layer — the ungrouped noaux_tc router at " +
             std::to_string(p.n_routed_experts) + "/" +
             std::to_string(p.num_experts_per_tok) + " plus the shared expert is W5";
    }
  }
  // The PADDED physical latent row. `MakeDots3NoteKVCache` reports the row both
  // attention classes share — `swa_kv_lora_rank + swa_qk_rope_head_dim`
  // (model.py:204-217) — and the full layers read their own logical width out
  // of the head of it. Narrowing on read is
  // `Dots3NotePaddedSparseImpl._logical_cache`, and it is W4b.
  //
  // This is checked HERE, at config level, and not only at the forward — review
  // finding F5. The forward's own cache-row assertion stays (an engine can hand
  // a cache that disagrees with the config it was built from, and a test does
  // exactly that), but leaving the config case to it meant the LOADER
  // materialized a whole tower for a config the very next call refuses.
  if (p.physical_latent_row() != p.full.latent_row()) {
    return "the physical MLA cache row is " +
           std::to_string(p.physical_latent_row()) + " but the full layers read " +
           std::to_string(p.full.latent_row()) +
           " — narrowing a PADDED row back to the logical one "
           "(`Dots3NotePaddedSparseImpl._logical_cache`) is W4b";
  }
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
  const mla::MlaBlockDims& d = w.mla;
  const int64_t H = p.hidden_size, V = p.vocab_size, I = p.intermediate_size;
  const int64_t N = d.num_heads, R = d.qk_rope_head_dim, L = d.kv_lora_rank;
  const int64_t QL = d.q_lora_rank;

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  RequireShape(w.embed_tokens, "model.embed_tokens.weight", {V, H});
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  RequireShape(w.final_norm, "model.norm.weight", {H});
  if (!p.tie_word_embeddings) {
    // [vocab, H] on disk -> [H, vocab] Matmul-B.
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  // `_compute_cos_sin_cache` over the whole positional range, once per model.
  {
    const std::vector<float> cache =
        mla::BuildDeepseekRopeCosSinCache(FullAttnRope(p), p.max_position_embeddings);
    w.rope_cos_sin_cache = MakeOwned(DType::kBF16, {p.max_position_embeddings, R});
    auto* dst = reinterpret_cast<uint16_t*>(w.rope_cos_sin_cache.bytes.data());
    for (size_t i = 0; i < cache.size(); ++i) dst[i] = vt::F32ToBF16(cache[i]);
  }

  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string pre = "model.layers." + std::to_string(l) + ".";
    const std::string sa = pre + "self_attn.";
    Dots3NoteLayerDeviceWeights& lw = w.layers[static_cast<size_t>(l)];
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
               ". W4a covers the full-attention layer with a dense MLP only; the "
               "sliding-window MLA is W4, the MoE is W5, the vision/audio towers "
               "are W6/W7. See .agents/specs/dots3-note.md and issue #699.");
  VT_CHECK(weights.materialized && weights.device.present,
           "Dots3NoteForCausalLM forward: the language tower was not "
           "materialized — the loader only materializes a config the device "
           "forward can run. See .agents/specs/dots3-note.md and issue #699.");
  // The DSA lightning indexer's SELECTION is not on the device path. While
  // every request's context fits in `index_topk` the top-k picks every causal
  // candidate and dense attention IS upstream's answer; past that it is a
  // different answer, and W3 measured that difference at 0.392 on the layer
  // output. Refuse rather than serve dense attention on a sparse model.
  const int64_t topk = Dots3NoteDenseEquivalentMaxSeqLen(p);
  for (int32_t sl : attn_meta.seq_lens) {
    VT_CHECK(static_cast<int64_t>(sl) <= topk,
             "Dots3NoteForCausalLM forward: a request needs " + std::to_string(sl) +
                 " keys but `index_topk` is " + std::to_string(topk) +
                 " — past that the DSA lightning indexer PRUNES "
                 "(model.py:171), and the sparse selection is not on the "
                 "device path yet (W4b). Refusing rather than serving dense "
                 "attention on a sparse model. See issue #699.");
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
  const Tensor rope = ResidentWeight(d, dw.rope_cos_sin_cache);
  step.rope_cache = &rope;
  v1::TritonMLAImpl impl;
  const float eps = static_cast<float>(p.rms_norm_eps);

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const Dots3NoteLayerDeviceWeights& lw = dw.layers[static_cast<size_t>(l)];
    const PagedKvCache& kv = attn_kv[static_cast<size_t>(l)];
    // The PHYSICAL row is the padded 1088 both classes share
    // (`physical_latent_row()`, model.py:204-217); the full layers read their
    // logical 576 out of the head of it. W4a runs a schedule with no sliding
    // layer, so the two coincide unless the config pads deliberately — and a
    // padded row is exactly what `_logical_cache` narrows upstream, which is
    // W4b. Refuse the padded case by name rather than reading a wrong stride.
    VT_CHECK(kv.num_kv_heads == 1 && kv.head_size == dw.mla.head_size(),
             "dots3-note forward: the MLA cache row is " +
                 std::to_string(kv.head_size) + " but this layer reads " +
                 std::to_string(dw.mla.head_size()) +
                 " — narrowing a PADDED physical row back to the logical one "
                 "(`Dots3NotePaddedSparseImpl._logical_cache`) is W4b");
    Tensor kv_cache = MakeTensor(kv.data, kv.dtype, d.q.device,
                                 {kv.num_blocks, kv.block_size, kv.head_size});

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
    const mla::MlaBlockWeights mw = ResidentMla(d, lw.attn, rope);
    mla::ForwardMlaAttentionBlock(d, dw.mla, mw, dhn.t(), step.positions, kv_cache,
                                  step.slot_mapping, step.meta, impl, attn_t);

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
