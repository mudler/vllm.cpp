// DeepSeek-V2 (`DeepseekV2ForCausalLM`) forward — MLA campaign W7. This is the
// first forward in this tree whose attention is MLA: it composes W6's
// `mla::ForwardMlaAttentionBlock` (which itself composes W3's cache write, W4's
// absorbed MQA decode and W5's materialized-MHA chunked prefill) with the
// DeepSeek MoE block and the dense `first_k_dense_replace` layers.
//
// Grounding (@ pin e24d1b24), file:line on both sides:
//   ForwardBody / ForwardLayers  <- vllm/model_executor/models/deepseek_v2.py
//                                   :1347-1520 `DeepseekV2Model.forward`
//   RunLayer                     <- deepseek_v2.py:1262-1345
//                                   `DeepseekV2DecoderLayer.forward`
//                                   (input_layernorm(h, residual) -> self_attn ->
//                                    post_attention_layernorm(h, residual) -> mlp)
//   DenseMlp                     <- deepseek_v2.py:270-274 `DeepseekV2MLP.forward`
//   MoeBlock                     <- deepseek_v2.py:395-424 `DeepseekV2MoE.forward`
//                                   + vllm/model_executor/layers/fused_moe/router/
//                                     grouped_topk_router.py:80-161
//                                   + fused_moe/runner/moe_runner.py:392-407
//                                     (routed_scaling_factor then + shared_output)
//   BuildMlaBatchSplit           <- mla_attention.py:1640-1649 + :1806-1810
//   BuildMlaStep                 <- mla_attention.py:1652-1830
//                                   (`MLACommonMetadataBuilder.build`, non-DCP)
//
// ─── THREE THINGS THAT ARE NEW FOR THIS MODEL FAMILY ────────────────────────
// 1. SHARED EXPERTS. Qwen3-Coder had none, and the only other shared-expert path
//    in this tree (Qwen3.6's) carries a SIGMOID GATE. DeepSeek's does not:
//    `DeepseekV2MoE.shared_experts` is a plain `DeepseekV2MLP` (:344-357) whose
//    output is ADDED to the routed sum. So this block does NOT reuse
//    `RunMoeBlock` / `SharedExpert` (which would apply Qwen's gate and Qwen's
//    ungrouped softmax router); it calls the same underlying vt:: ops directly.
//    That also means ZERO edit to the 27B/35B/Coder MoE paths.
// 2. THE GROUPED (`noaux_tc`-shaped) ROUTER. `use_grouped_topk=True` is
//    UNCONDITIONAL for this family (:370-378) — even V2-Lite, whose `n_group ==
//    topk_group == 1` makes the grouping a no-op. We pass the grouped args
//    through to W3's `vt::MoeRouterTopK` extension rather than special-casing.
// 3. THE BATCH ORDER. MLA's metadata is built from a batch that upstream's
//    scheduler already reordered (decodes first; with-context prefills leading
//    the prefill tail). `BuildMlaBatchSplit` VALIDATES that rather than assuming
//    it — W6's gate measured 0.86 relative error from exactly that violation.
//
// ─── RECORDED DEVIATIONS ────────────────────────────────────────────────────
// (a) `routed_scaling_factor` is applied to the ROUTING WEIGHTS
//     (`MoeRouterTopKArgs::routed_scaling_factor`, i.e.
//     grouped_topk_router.py:159-160) rather than to the combined routed OUTPUT
//     (moe_runner.py:402-406, which vLLM's CUDA path selects via
//     `apply_routed_scale_to_output=True`). The routed combine is linear in the
//     weights, so the two are the same function; they differ only in float
//     rounding order, and the shared-expert term — which must NOT be scaled — is
//     added after either way. DeepSeek-V2-Lite has `routed_scaling_factor: 1.0`,
//     so on the W8 gate vehicle the two are bit-identical.
// (b) The chunked-prefill workspace size mirrors
//     `determine_chunked_prefill_workspace_size` (mla_attention.py:1422-1451)
//     with `max_model_len := config.max_position_embeddings` and
//     `max_num_seqs := this step's num_reqs`, because the model forward has no
//     VllmConfig. With the default (unclamped) `--max-model-len` those are the
//     same numbers upstream uses.
// (c) No decode CUDA-graph sibling and no fused A-projection GEMM: both are W9.
#include "vllm/model_executor/models/deepseek_v2.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/attention/mla_chunked_context.h"
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight glue
#include "vllm/model_executor/models/device_pool.h"
#include "vllm/model_executor/models/mla_attention.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

using namespace dense_attn;  // Dev / DBuf / MakeTensor / Reshape / ResidentWeight

// ─── per-step device inputs the MLA block needs ─────────────────────────────
// Everything that is per-STEP (not per-layer) is uploaded once and shared by
// every layer: positions, slot_mapping, and the whole MlaBlockMetadata (decode
// block table / seq lens, prefill cu_seqlens_q / block table, chunk descriptors).
// The DBufs live here so the Tensor views inside `meta` stay valid for the
// duration of the forward.
struct MlaStep {
  std::vector<DBuf> owned;
  Tensor positions;
  Tensor slot_mapping;
  mla::MlaBlockMetadata meta;
  MlaBatchSplit split;
  // The per-MODEL YaRN [cos|sin] cache (device view), set by the caller once —
  // it is shared by every layer and every step, unlike everything above.
  const Tensor* rope_cache = nullptr;
};

// ─── W8 diagnostic counters (deepseek_v2.h `MlaBatchSplitStats`) ─────────────
// Written once per forward from the runner's single forward thread; read only by
// the paged-engine gate, which uses them to PROVE the engine actually produced
// mixed / with-context MLA batches rather than a stream of trivial ones.
MlaBatchSplitStats& MutableMlaBatchSplitStats() {
  static MlaBatchSplitStats stats;
  return stats;
}

void RecordMlaBatchSplit(const MlaBatchSplit& sp, int num_reqs) {
  MlaBatchSplitStats& st = MutableMlaBatchSplitStats();
  ++st.steps;
  if (sp.num_prefills == 0 && sp.num_decodes > 0) ++st.decode_only_steps;
  if (sp.num_decodes == 0 && sp.num_prefills > 0) ++st.prefill_only_steps;
  if (sp.num_decodes > 0 && sp.num_prefills > 0) ++st.mixed_steps;
  if (sp.num_prefills_with_context > 0) ++st.with_context_prefill_steps;
  st.max_num_decodes = std::max<int64_t>(st.max_num_decodes, sp.num_decodes);
  st.max_num_prefills = std::max<int64_t>(st.max_num_prefills, sp.num_prefills);
  st.max_num_reqs = std::max<int64_t>(st.max_num_reqs, num_reqs);
  st.total_decode_tokens += sp.num_decode_tokens;
  st.total_prefill_tokens += sp.num_prefill_tokens;
}

template <typename T>
Tensor UploadInto(Dev d, std::vector<DBuf>& owned, DType dt,
                  const std::vector<int64_t>& shape, const std::vector<T>& host) {
  owned.emplace_back(d, dt, shape, host.data());
  return owned.back().t();
}

// `MLACommonMetadataBuilder.build` (mla_attention.py:1652-1830), non-DCP branch.
MlaStep BuildMlaStep(Dev d, const std::vector<int32_t>& positions,
                     const CommonAttentionMetadata& am, int64_t block_size,
                     int64_t max_model_len) {
  MlaStep s;
  s.split = BuildMlaBatchSplit(am);
  const MlaBatchSplit& sp = s.split;
  RecordMlaBatchSplit(sp, am.num_reqs);
  const int64_t T = static_cast<int64_t>(positions.size());

  s.positions = UploadInto(d, s.owned, DType::kI32, {T}, positions);
  s.slot_mapping = UploadInto(d, s.owned, DType::kI64, {T}, am.slot_mapping);

  s.meta.num_decode_tokens = sp.num_decode_tokens;
  const int64_t cols = am.block_table_num_cols;

  // --- decode half (triton_mla.py:214-216, 245-246) ---
  if (sp.num_decodes > 0) {
    std::vector<int32_t> bt(am.block_table_tensor.begin(),
                            am.block_table_tensor.begin() +
                                static_cast<std::ptrdiff_t>(sp.num_decodes * cols));
    std::vector<int32_t> sl(am.seq_lens.begin(),
                            am.seq_lens.begin() +
                                static_cast<std::ptrdiff_t>(sp.num_decodes));
    s.meta.decode.block_table =
        UploadInto(d, s.owned, DType::kI32, {sp.num_decodes, cols}, bt);
    s.meta.decode.seq_lens =
        UploadInto(d, s.owned, DType::kI32, {sp.num_decodes}, sl);
    s.meta.decode.max_seq_len = sp.decode_max_seq_len;
  }

  // --- prefill half (mla_attention.py:1652-1682) ---
  if (sp.num_prefills > 0) {
    std::vector<int32_t> bt(
        am.block_table_tensor.begin() +
            static_cast<std::ptrdiff_t>(sp.num_decodes * cols),
        am.block_table_tensor.begin() +
            static_cast<std::ptrdiff_t>((sp.num_decodes + sp.num_prefills) * cols));
    s.meta.prefill_cu_seqlens_q = UploadInto(d, s.owned, DType::kI32,
                                             {sp.num_prefills + 1},
                                             sp.prefill_cu_seqlens_q);
    s.meta.prefill_block_table =
        UploadInto(d, s.owned, DType::kI32, {sp.num_prefills, cols}, bt);
    s.meta.max_query_len = sp.prefill_max_query_len;

    if (sp.num_prefills_with_context > 0) {
      // DEVIATION (b) — see the TU header.
      const int64_t workspace = mla::DetermineChunkedPrefillWorkspaceSize(
          max_model_len, am.num_reqs, block_size);
      const mla::MlaChunkedContextMetadata cm = mla::BuildMlaChunkedContext(
          sp.prefill_context_lens, sp.prefill_cu_seqlens_q, workspace, block_size);
      s.meta.prefill_tokens_with_context = cm.prefill_tokens_with_context;
      s.meta.chunk_workspace_tokens = workspace;
      const int64_t np = cm.num_prefills;
      const int32_t row = std::max<int32_t>(cm.max_token_num_over_chunk, 1);
      for (int32_t i = 0; i < cm.num_chunks; ++i) {
        const std::vector<int32_t> cu(
            cm.cu_seq_lens.begin() + static_cast<size_t>(i) * (np + 1),
            cm.cu_seq_lens.begin() + static_cast<size_t>(i + 1) * (np + 1));
        const std::vector<int32_t> starts(
            cm.starts.begin() + static_cast<size_t>(i) * np,
            cm.starts.begin() + static_cast<size_t>(i + 1) * np);
        const std::vector<int32_t> t2s(
            cm.token_to_seq.begin() + static_cast<size_t>(i) * row,
            cm.token_to_seq.begin() + static_cast<size_t>(i + 1) * row);
        mla::MlaChunkDeviceMetadata cd;
        cd.cu_seq_lens = UploadInto(d, s.owned, DType::kI32, {np + 1}, cu);
        cd.starts = UploadInto(d, s.owned, DType::kI32, {np}, starts);
        cd.token_to_seq = UploadInto(d, s.owned, DType::kI32, {row}, t2s);
        cd.total_tokens = cm.chunk_total_token[static_cast<size_t>(i)];
        cd.max_seq_len = cm.max_seq_lens[static_cast<size_t>(i)];
        s.meta.chunks.push_back(cd);
      }
    }
  }
  return s;
}

// The MLA block's DEVICE-resident weight views for one layer. ResidentWeight
// uploads once on first touch and memoizes on the OwnedTensor, so this is a
// pointer shuffle after the first step.
mla::MlaBlockWeights ResidentMla(Dev d, const DeepseekV2MlaWeights& w,
                                 const DeepseekV2Params& p,
                                 const Tensor& rope_cache) {
  const mla::MlaBlockDims& dm = p.mla;
  mla::MlaBlockWeights m;
  if (dm.has_q_lora()) {
    m.fused_qkv_a_proj = ResidentWeight(d, w.fused_qkv_a_proj);
    m.q_a_layernorm = ResidentWeight(d, w.q_a_layernorm, {dm.q_lora_rank});
    m.q_b_proj = ResidentWeight(d, w.q_b_proj);
  } else {
    m.kv_a_proj_with_mqa = ResidentWeight(d, w.kv_a_proj_with_mqa);
    m.q_proj = ResidentWeight(d, w.q_proj);
  }
  m.kv_a_layernorm = ResidentWeight(d, w.kv_a_layernorm, {dm.kv_lora_rank});
  m.kv_b_proj = ResidentWeight(d, w.kv_b_proj);
  m.w_uk_t = ResidentWeight(d, w.w_uk_t);
  m.w_uv = ResidentWeight(d, w.w_uv);
  m.o_proj = ResidentWeight(d, w.o_proj);
  m.rope_cos_sin_cache = rope_cache;
  return m;
}

// `DeepseekV2MLP.forward` (deepseek_v2.py:270-274): merged gate_up GEMM ->
// SiluAndMul -> down GEMM. Identical op shape to the Qwen3-dense MlpBlock.
DBuf DenseMlp(Dev d, const DeepseekV2DenseMlp& w, const Tensor& dh, int64_t T,
              int64_t H, int64_t I) {
  Tensor wgu = ResidentWeight(d, w.gate_up_proj);  // [2I, H] raw-NK
  DBuf gate_up(d, DType::kBF16, {T, 2 * I});
  vt::MatmulBT(d.q, gate_up.t(), dh, wgu);
  DBuf act(d, DType::kBF16, {T, I});
  vt::SiluAndMul(d.q, act.t(), gate_up.t());
  Tensor wdn = ResidentWeight(d, w.down_proj);  // [H, I] raw-NK
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wdn);
  return out;
}

// Per-layer RESIDENT expert device-pointer arrays for the grouped bf16 MoE GEMM
// (the DeepSeek analog of qwen3_5.cpp's MoeBf16Resident). Uploaded ONCE at first
// touch, keyed on the layer's weight-struct address, and never freed for the
// process lifetime — the same retire-don't-free contract the existing grouped
// MoE paths rely on so nothing can dangle.
struct MoePtrs {
  void* gate = nullptr;
  void* up = nullptr;
  void* down = nullptr;
  std::map<int64_t, void*> tok_map;  // T -> [T*top_k] i32 pair->token row map
  bool ready = false;
};
MoePtrs& MoePtrsFor(const DeepseekV2MoeWeights* key) {
  static std::map<const DeepseekV2MoeWeights*, MoePtrs> table;
  return table[key];
}

bool GroupedMoeEligible(Dev d, const DeepseekV2MoeWeights& w) {
  if (!vllm::platforms::GetPlatform(d.q.device.type).is_cuda()) return false;
  if (w.expert_gate.empty()) return false;
  // vt::MoeGroupedGemmBf16 reads the [K,N] Matmul-B orientation only.
  for (const OwnedTensor* t : {&w.expert_gate[0], &w.expert_up[0], &w.expert_down[0]})
    if (t->nk) return false;
  return true;
}

// `DeepseekV2MoE.forward` (deepseek_v2.py:395-424) + the FusedMoE runner's
// routed/shared composition (moe_runner.py:392-407).
DBuf MoeBlock(Dev d, const DeepseekV2MoeWeights& w, const DeepseekV2Params& p,
              const Tensor& dh, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.n_routed_experts;
  const int64_t top_k = p.num_experts_per_tok;
  const int64_t I = p.moe_intermediate_size;
  const int64_t P = T * top_k;

  // --- router: logits = gate(hidden) then grouped top-k ---------------------
  // `router_logits, _ = self.gate(hidden_states)` (:413). GateLinear is a plain
  // ReplicatedLinear here (router_dtype is None for V2/V2-Lite —
  // `_get_moe_router_dtype` at :120-130 returns fp32 only for glm_moe_dsa or an
  // explicit `moe_router_dtype: "float32"`), so this is the bf16 GEMM.
  Tensor drg = ResidentWeight(d, w.router_gate);  // [H,E] Matmul-B
  DBuf dlog(d, DType::kBF16, {T, E});
  vt::Matmul(d.q, dlog.t(), dh, drg);

  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(top_k);
  args.renormalize = p.norm_topk_prob;
  args.scoring_func = p.scoring_func;
  args.num_expert_group = static_cast<int>(p.n_group);
  args.topk_group = static_cast<int>(p.topk_group);
  args.routed_scaling_factor = p.routed_scaling_factor;  // DEVIATION (a)
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  std::unique_ptr<Tensor> bias;
  if (!w.e_score_correction_bias.Empty()) {
    bias = std::make_unique<Tensor>(ResidentWeight(d, w.e_score_correction_bias, {E}));
  }
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(), args, bias.get());

  // --- routed experts -------------------------------------------------------
  DBuf expert_out(d, DType::kBF16, {T, top_k, H});
  if (GroupedMoeEligible(d, w)) {
    // ~3 grouped bf16 GEMM launches, fully on device (the same op sequence the
    // gated Qwen3-Coder bf16 MoE runs: vt::MoeGroupedGemmBf16 x2 -> MoeSiluMul
    // -> vt::MoeGroupedGemmBf16). The top-k ids [T,top_k] ARE the per-pair
    // expert ids viewed [P].
    MoePtrs& mr = MoePtrsFor(&w);
    if (!mr.ready) {
      std::vector<int64_t> gp(static_cast<size_t>(E)), up(static_cast<size_t>(E)),
          dp(static_cast<size_t>(E));
      for (int64_t e = 0; e < E; ++e) {
        const size_t se = static_cast<size_t>(e);
        gp[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_gate[se]).data);
        up[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_up[se]).data);
        dp[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_down[se]).data);
      }
      const size_t eb = static_cast<size_t>(E) * sizeof(int64_t);
      auto upload = [&](const std::vector<int64_t>& h) {
        void* q = d.b.Alloc(eb);
        d.b.Copy(d.q, q, h.data(), eb);
        return q;
      };
      mr.gate = upload(gp);
      mr.up = upload(up);
      mr.down = upload(dp);
      mr.ready = true;
    }
    auto tok_it = mr.tok_map.find(T);
    if (tok_it == mr.tok_map.end()) {
      std::vector<int32_t> tok_map(static_cast<size_t>(P));
      for (int64_t i = 0; i < P; ++i)
        tok_map[static_cast<size_t>(i)] = static_cast<int32_t>(i / top_k);
      const size_t tb = static_cast<size_t>(P) * sizeof(int32_t);
      void* q = d.b.Alloc(tb);
      d.b.Copy(d.q, q, tok_map.data(), tb);
      tok_it = mr.tok_map.emplace(T, q).first;
    }
    Tensor eids = Reshape(dtid.t(), {P});
    Tensor gate_ptrs = MakeTensor(mr.gate, DType::kI64, d.q.device, {E});
    Tensor up_ptrs = MakeTensor(mr.up, DType::kI64, d.q.device, {E});
    Tensor down_ptrs = MakeTensor(mr.down, DType::kI64, d.q.device, {E});
    Tensor dtok = MakeTensor(tok_it->second, DType::kI32, d.q.device, {P});
    DBuf dgate(d, DType::kF32, {P, I});
    DBuf dup(d, DType::kF32, {P, I});
    vt::MoeGroupedGemmBf16(d.q, dgate.t(), dh, eids, &dtok, gate_ptrs);
    vt::MoeGroupedGemmBf16(d.q, dup.t(), dh, eids, &dtok, up_ptrs);
    DBuf dact(d, DType::kBF16, {P, I});
    vt::MoeSiluMul(d.q, dact.t(), dgate.t(), dup.t());
    Tensor eo = Reshape(expert_out.t(), {P, H});
    vt::MoeGroupedGemmBf16(d.q, eo, dact.t(), eids, nullptr, down_ptrs);
  } else {
    // REFERENCE path (CPU, and any non-Matmul-B expert layout): download the
    // routing decision, gather each activated expert's token rows, run its
    // SwiGLU MLP, scatter back. Same numerics per (token, slot) as the grouped
    // path modulo GEMM accumulation order.
    std::vector<int32_t> ids(static_cast<size_t>(P));
    dtid.Download(d, ids.data());
    std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(
        static_cast<size_t>(E));
    for (int64_t t = 0; t < T; ++t)
      for (int64_t j = 0; j < top_k; ++j)
        lists[static_cast<size_t>(ids[static_cast<size_t>(t * top_k + j)])]
            .push_back({t, j});
    expert_out.Zero(d);
    const size_t row_bytes = static_cast<size_t>(H) * vt::SizeOf(DType::kBF16);
    for (int64_t e = 0; e < E; ++e) {
      const auto& list = lists[static_cast<size_t>(e)];
      if (list.empty()) continue;
      const int64_t n = static_cast<int64_t>(list.size());
      DBuf xg(d, DType::kBF16, {n, H});
      for (int64_t r = 0; r < n; ++r) {
        d.b.Copy(d.q, static_cast<char*>(xg.ptr()) + static_cast<size_t>(r) * row_bytes,
                 static_cast<const char*>(dh.data) +
                     static_cast<size_t>(list[static_cast<size_t>(r)].first) * row_bytes,
                 row_bytes);
      }
      DBuf g(d, DType::kBF16, {n, I}), u(d, DType::kBF16, {n, I});
      vt::Matmul(d.q, g.t(), xg.t(), ResidentWeight(d, w.expert_gate[static_cast<size_t>(e)]));
      vt::Matmul(d.q, u.t(), xg.t(), ResidentWeight(d, w.expert_up[static_cast<size_t>(e)]));
      DBuf a(d, DType::kBF16, {n, I});
      vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
      DBuf o(d, DType::kBF16, {n, H});
      vt::Matmul(d.q, o.t(), a.t(), ResidentWeight(d, w.expert_down[static_cast<size_t>(e)]));
      for (int64_t r = 0; r < n; ++r) {
        const auto& tj = list[static_cast<size_t>(r)];
        d.b.Copy(d.q,
                 static_cast<char*>(expert_out.ptr()) +
                     static_cast<size_t>(tj.first * top_k + tj.second) * row_bytes,
                 static_cast<const char*>(o.ptr()) + static_cast<size_t>(r) * row_bytes,
                 row_bytes);
      }
    }
  }

  // --- shared experts + weighted combine ------------------------------------
  // DeepSeek's shared expert is a PLAIN MLP whose output is added to the routed
  // sum (deepseek_v2.py:344-357; moe_runner.py:407 `shared_output + fused_output`)
  // — NO sigmoid gate, unlike Qwen3.6's. `vt::MoeCombine`'s optional `shared`
  // term is exactly that add.
  const bool has_shared = p.n_shared_experts > 0 && !w.shared.Empty();
  DBuf out(d, DType::kBF16, {T, H});
  if (has_shared) {
    DBuf shared = DenseMlp(d, w.shared, dh, T, H, p.shared_intermediate_size());
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared.t());
  } else {
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), nullptr);
  }
  return out;
}

// One `DeepseekV2DecoderLayer` (deepseek_v2.py:1262-1345). The bf16 residual
// stream is threaded through the two fused add+RMSNorm producers exactly as in
// every other model in this tree; the FP16-overflow rescale at :1298-1308 /
// :1338-1343 is guarded on `hidden_states.dtype == torch.float16` and is
// therefore INACTIVE for a bf16 model (we are bf16), so it is not ported.
void RunLayer(Dev d, const DeepseekV2LayerWeights& layer, const DeepseekV2Params& p,
              Tensor& hidden, std::shared_ptr<void>& hidden_hold, DBuf& res,
              const MlaStep& step, Tensor& kv_cache, v1::TritonMLAImpl& impl,
              int64_t T) {
  const int64_t H = p.hidden_size;
  const float eps = p.rms_norm_eps;

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dhn.t(), hidden, w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dhn.t(), hidden, w_in, vt::RmsNormArgs{eps, false}, &res.t());
  }

  DBuf attn(d, DType::kBF16, {T, H});
  Tensor attn_t = attn.t();
  const mla::MlaBlockWeights mw = ResidentMla(d, layer.attn, p, *step.rope_cache);
  mla::ForwardMlaAttentionBlock(d, p.mla, mw, dhn.t(), step.positions, kv_cache,
                                step.slot_mapping, step.meta, impl, attn_t);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
  }

  // `self.mlp(hidden_states)` — MoE or dense, per first_k_dense_replace (:1214).
  DBuf mlp = layer.is_moe ? MoeBlock(d, layer.moe, p, dh2.t(), T)
                          : DenseMlp(d, layer.dense, dh2.t(), T, H, p.intermediate_size);
  // The MLP output becomes the new residual-stream delta; keep its storage alive
  // until the next producer replaces it.
  auto* held = new DBuf(std::move(mlp));
  hidden = held->t();
  hidden_hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& am,
                 const std::vector<PagedKvCache>& attn_kv,
                 const DeepseekV2Weights& weights,
                 const std::vector<int32_t>& logits_indices) {
  const DeepseekV2Params& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t vocab = p.vocab_size;
  const float eps = p.rms_norm_eps;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "deepseek-v2: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(p.num_hidden_layers),
           "deepseek-v2: one MLA PagedKvCache per layer required");
  VT_CHECK(T > 0, "deepseek-v2: empty batch");

  DBuf hidden_buf(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    Tensor h = hidden_buf.t();
    vt::Embedding(d.q, h, dtab, dids.t());
  }
  Tensor hidden = hidden_buf.t();
  std::shared_ptr<void> hidden_hold;

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  const int64_t block_size = attn_kv[0].block_size;
  MlaStep step = BuildMlaStep(d, positions, am, block_size, p.max_position_embeddings);
  // The rope cache is per-MODEL, not per-step; hand the block a device view.
  const Tensor rope = ResidentWeight(d, weights.rope_cos_sin_cache);
  step.rope_cache = &rope;

  v1::TritonMLAImpl impl;
  const int64_t head_size = p.mla.head_size();
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const PagedKvCache& kv = attn_kv[static_cast<size_t>(l)];
    VT_CHECK(kv.num_kv_heads == 1 && kv.head_size == head_size,
             "deepseek-v2: the MLA cache must be 1-head, "
             "kv_lora_rank + qk_rope_head_dim wide (MLAAttentionSpec)");
    Tensor kv_cache = MakeTensor(kv.data, kv.dtype, d.q.device,
                                 {kv.num_blocks, kv.block_size, head_size});
    RunLayer(d, weights.layers[static_cast<size_t>(l)], p, hidden, hidden_hold, res,
             step, kv_cache, impl, T);
  }

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm.t(), hidden, w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm.t(), hidden, w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  }

  const bool tied = p.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kBF16,
               do_gather ? std::vector<int64_t>{
                               static_cast<int64_t>(logits_indices.size()), H}
                         : std::vector<int64_t>{1, 1});
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);
  return logits;
}

ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  const size_t alloc = dlogits.alloc_bytes();
  void* p = dlogits.Release();
  fl.device_storage =
      std::shared_ptr<void>(p, [alloc](void* q) { Pool().Put(alloc, q); });
  return fl;
}

}  // namespace

// ─── W8: the split-shape counters (deepseek_v2.h) ───────────────────────────
const MlaBatchSplitStats& GetMlaBatchSplitStats() {
  return MutableMlaBatchSplitStats();
}
void ResetMlaBatchSplitStats() { MutableMlaBatchSplitStats() = MlaBatchSplitStats{}; }

// ─── the batch split + its ORDERING invariant ───────────────────────────────
MlaBatchSplit BuildMlaBatchSplit(const CommonAttentionMetadata& am) {
  MlaBatchSplit s;
  const int num_reqs = am.num_reqs;
  VT_CHECK(num_reqs >= 0, "deepseek-v2: negative num_reqs");
  VT_CHECK(static_cast<int>(am.query_start_loc.size()) == num_reqs + 1,
           "deepseek-v2: query_start_loc must have num_reqs + 1 entries");
  VT_CHECK(static_cast<int>(am.seq_lens.size()) >= num_reqs,
           "deepseek-v2: seq_lens must have at least num_reqs entries");

  // `split_decodes_and_prefills(..., decode_threshold=reorder_batch_threshold)`
  // with `reorder_batch_threshold == 1` (mla_attention.py:1420): a request is a
  // DECODE iff its query length is <= 1, and decodes must form a PREFIX.
  bool seen_prefill = false;
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t q_len = am.query_start_loc[static_cast<size_t>(r) + 1] -
                          am.query_start_loc[static_cast<size_t>(r)];
    VT_CHECK(q_len >= 0, "deepseek-v2: non-monotonic query_start_loc");
    const bool is_decode = q_len <= 1;
    if (is_decode) {
      VT_CHECK(!seen_prefill,
               "deepseek-v2 MLA batch ordering: request " + std::to_string(r) +
                   " is a DECODE (query_len <= 1) but follows a PREFILL. MLA "
                   "requires decodes to be packed FIRST "
                   "(reorder_batch_to_split_decodes_and_prefills; the block "
                   "slices q[:num_mqa_tokens] / q[num_mqa_tokens:], "
                   "mla_attention.py:700-737).");
      ++s.num_decodes;
      s.num_decode_tokens += q_len;
      s.decode_max_seq_len =
          std::max(s.decode_max_seq_len, am.seq_lens[static_cast<size_t>(r)]);
    } else {
      seen_prefill = true;
      ++s.num_prefills;
      s.num_prefill_tokens += q_len;
      s.prefill_max_query_len = std::max(s.prefill_max_query_len, q_len);
    }
  }

  // The prefill tail, RELATIVE to its own start (mla_attention.py:1670-1675).
  s.prefill_cu_seqlens_q.reserve(static_cast<size_t>(s.num_prefills) + 1);
  s.prefill_cu_seqlens_q.push_back(0);
  bool seen_without_context = false;
  for (int r = s.num_decodes; r < num_reqs; ++r) {
    const int32_t q_len = am.query_start_loc[static_cast<size_t>(r) + 1] -
                          am.query_start_loc[static_cast<size_t>(r)];
    s.prefill_cu_seqlens_q.push_back(s.prefill_cu_seqlens_q.back() + q_len);
    const int32_t ctx = am.seq_lens[static_cast<size_t>(r)] - q_len;
    VT_CHECK(ctx >= 0,
             "deepseek-v2: request " + std::to_string(r) +
                 " has seq_len < query_len (negative context)");
    s.prefill_context_lens.push_back(ctx);
    // THE W6 INVARIANT: `prefill_tokens_with_context =
    // prefill_query_start_loc_cpu[num_prefills_with_context]`
    // (mla_attention.py:1806-1810) is a PREFIX LENGTH, so every with-context
    // prefill must precede every context-free one. A batch that violates this
    // measured 0.86 relative error in W6's gate — fail loudly instead.
    if (ctx > 0) {
      VT_CHECK(!seen_without_context,
               "deepseek-v2 MLA batch ordering: prefill request " +
                   std::to_string(r) +
                   " HAS context but follows a context-free prefill. MLA "
                   "requires with-context prefills to LEAD the prefill tail "
                   "(prefill_tokens_with_context is a prefix length, "
                   "mla_attention.py:1806-1810).");
      ++s.num_prefills_with_context;
    } else {
      seen_without_context = true;
    }
  }
  VT_CHECK(s.num_decode_tokens + s.num_prefill_tokens == am.num_actual_tokens,
           "deepseek-v2: query_start_loc does not sum to num_actual_tokens");
  return s;
}

std::vector<float> DeepseekV2Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV2Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * weights.params.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits DeepseekV2Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV2Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, weights.params.vocab_size);
}

}  // namespace vllm
