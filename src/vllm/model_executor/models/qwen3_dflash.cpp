// DFlash draft model forward (SPEC-DFLASH D2, DF-DRAFT-MODEL). Ported from
// vllm/model_executor/models/qwen3_dflash.py @ 555967922. See qwen3_dflash.h.
//
// The ONE new brick is the attention: full-attention layers route through
// vt::DFlashBlockAttention with args.causal=false (BIDIRECTIONAL in-block); SWA
// layers use args.causal=true + the window. Every other op (embed, merged-qkv
// GEMM, per-head q/k RMSNorm, NeoX RoPE, SwiGLU, standard add+RMSNorm, lm_head) is
// reused from the landed Qwen3-dense block ops (dense_attn_block.h / vt::).
#include "vllm/model_executor/models/qwen3_dflash.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <vector>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight/Reshape/MakeRopeArgs
#include "vllm/platforms/interface.h"                     // platforms::GetPlatform (static-graph gate)
#include "vt/backend.h"
#include "vt/breakable_graph.h"  // ENG-CUDAGRAPH-BREAK W5: the shared capture seam
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using namespace dense_attn;  // Dev, DBuf, ResidentWeight, Reshape, MakeRopeArgs

constexpr int64_t kPadSlotId = -1;  // vLLM PAD_SLOT_ID (attention/backends/utils.py:45)

// Device-resident per-layer context K/V (D7). The D5 path downloaded each layer's
// projected K/V to host (2 D->H copies/layer) and re-uploaded them in the block
// forward's [context;block] host interleave. This helper keeps the projected K/V
// ON DEVICE as bf16 [num_ctx, kv_size] buffers, so the block forward can build the
// combined sequence with device vt::IndexCopy instead of host round-trips. The
// float ops (cast/RMSNorm/GEMM/k-norm/RoPE) are IDENTICAL to the D5 path and run in
// the same order, so the stored K/V bits are bit-identical to the D5 download.
// Mirrors precompute_and_store_context_kv (qwen3_dflash.py:548-619), minus the
// paged-cache write (our within-step store is these DBufs).
struct ContextKVDev {
  std::vector<DBuf> k;  // per attention layer: bf16 [num_ctx, Hkv*Dh] (normed+RoPE'd)
  std::vector<DBuf> v;  // per attention layer: bf16 [num_ctx, Hkv*Dh] (raw)
  int64_t num_ctx = 0;
};

ContextKVDev PrecomputeContextKVDevice(Dev d, const float* context_states,
                                       const int32_t* context_positions, int64_t C,
                                       const Qwen3DFlashWeights& weights,
                                       const HfConfig& config) {
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const float eps = static_cast<float>(config.rms_norm_eps);

  ContextKVDev out;
  out.num_ctx = C;
  if (C == 0) return out;

  // normed = RMSNorm(context_states, hidden_norm) — the ONE shared hidden_norm
  // over the combined target features (qwen3_dflash.py:505-520).
  DBuf ctx32(d, DType::kF32, {C, H}, context_states);
  DBuf ctxb(d, DType::kBF16, {C, H});
  vt::CastBf16(d.q, ctxb.t(), ctx32.t());
  Tensor w_hn = ResidentWeight(d, weights.hidden_norm, {H});
  DBuf normed(d, DType::kBF16, {C, H});
  vt::RmsNorm(d.q, normed.t(), ctxb.t(), w_hn, vt::RmsNormArgs{eps, false});
  DBuf cpos(d, DType::kI32, {C}, context_positions);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
    Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
    Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
    DBuf k(d, DType::kBF16, {C, kdim});
    DBuf v(d, DType::kBF16, {C, kdim});
    vt::MatmulBT(d.q, k.t(), normed.t(), wk);
    vt::MatmulBT(d.q, v.t(), normed.t(), wv);
    // K-norm over head_dim, then NeoX RoPE on K at the context positions (V raw).
    Tensor k2 = Reshape(k.t(), {C * Hkv, Dh});
    Tensor wkn = ResidentWeight(d, layer.k_norm, {Dh});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
    Tensor k3 = Reshape(k.t(), {C, Hkv, Dh});
    DBuf rope_scratch(d, DType::kBF16, {C, Hkv, Dh});
    rope_scratch.Zero(d);
    Tensor scratch3 = rope_scratch.t();
    vt::RopeNeox(d.q, k3, scratch3, cpos.t(), MakeRopeArgs(config));
    out.k.push_back(std::move(k));  // bf16 [C, kdim] contiguous (RoPE'd view aliases it)
    out.v.push_back(std::move(v));  // bf16 [C, kdim] raw
  }
  return out;
}

}  // namespace

DflashPrepareOutputs PrepareDflashInputs(const DflashPrepareBatch& b) {
  // Pure-integer host port of _prepare_dflash_inputs_kernel (dflash/speculator.py:
  // 472-618). Every store the Triton kernel makes is reproduced here; there is no
  // float math, so this is bit-exact by construction.
  const int32_t num_reqs = static_cast<int32_t>(b.idx_mapping.size());
  VT_CHECK(num_reqs > 0, "prepare_dflash_inputs: num_reqs must be > 0");
  VT_CHECK(static_cast<int32_t>(b.target_query_start_loc.size()) == num_reqs + 1,
           "prepare_dflash_inputs: target_query_start_loc must be [num_reqs+1]");
  const int32_t nqpr = b.num_query_per_req;
  const int32_t nspec = b.num_speculative_steps;
  const int32_t stride = b.block_table_stride;
  const int32_t bs = b.block_size;
  const int64_t num_target_tokens = b.target_query_start_loc.back();

  DflashPrepareOutputs o;
  o.input_ids.assign(static_cast<size_t>(num_reqs) * nqpr, 0);
  o.query_positions.assign(static_cast<size_t>(num_reqs) * nqpr, 0);
  o.query_start_loc.assign(static_cast<size_t>(b.max_num_reqs) + 1, 0);
  o.seq_lens.assign(static_cast<size_t>(b.max_num_reqs), 0);
  o.query_slot_mapping.assign(static_cast<size_t>(b.max_num_tokens), 0);
  o.context_positions.assign(static_cast<size_t>(num_target_tokens), 0);
  o.context_slot_mapping.assign(static_cast<size_t>(num_target_tokens), 0);
  o.sample_indices.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);
  o.sample_pos.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);
  o.sample_idx_mapping.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);

  const int32_t sample_off = b.sample_from_anchor ? 0 : 1;

  for (int32_t r = 0; r < num_reqs; ++r) {
    const int32_t req_state_idx = b.idx_mapping[static_cast<size_t>(r)];
    const int32_t ctx_start = b.target_query_start_loc[static_cast<size_t>(r)];
    const int32_t ctx_end = b.target_query_start_loc[static_cast<size_t>(r) + 1];
    const int32_t num_ctx = ctx_end - ctx_start;
    const int32_t num_rejected = b.num_rejected[static_cast<size_t>(r)];
    const int32_t valid_ctx_end = ctx_end - num_rejected;
    const int32_t num_sampled = b.num_sampled[static_cast<size_t>(r)];
    const int32_t bonus_token =
        num_sampled > 0 ? b.last_sampled[static_cast<size_t>(req_state_idx)]
                        : b.next_prefill_tokens[static_cast<size_t>(req_state_idx)];
    const int64_t last_valid_pos =
        b.target_positions[static_cast<size_t>(valid_ctx_end) - 1];
    const int32_t query_base = r * nqpr;

    // --- Context positions / slots (j in [0, num_ctx)) ---
    for (int32_t j = 0; j < num_ctx; ++j) {
      const int64_t ctx_pos = b.target_positions[static_cast<size_t>(ctx_start + j)];
      int32_t ctx_block_num = static_cast<int32_t>(ctx_pos / bs);
      if (ctx_block_num > stride - 1) ctx_block_num = stride - 1;
      const int64_t ctx_block_id =
          b.block_table[static_cast<size_t>(r) * stride + ctx_block_num];
      const int64_t ctx_slot = ctx_block_id * bs + (ctx_pos % bs);
      o.context_positions[static_cast<size_t>(ctx_start + j)] = ctx_pos;
      o.context_slot_mapping[static_cast<size_t>(ctx_start + j)] = ctx_slot;
    }

    // --- Query positions / input_ids / slots + sample maps (offset in [0,nqpr)) ---
    for (int32_t off = 0; off < nqpr; ++off) {
      const int64_t query_pos = last_valid_pos + 1 + off;
      const int32_t query_idx = query_base + off;
      const int32_t input_id = (off == 0) ? bonus_token : b.parallel_drafting_token_id;
      int32_t q_block_num = static_cast<int32_t>(query_pos / bs);
      if (q_block_num > stride - 1) q_block_num = stride - 1;
      const int64_t q_block_id =
          b.block_table[static_cast<size_t>(r) * stride + q_block_num];
      const int64_t q_slot = q_block_id * bs + (query_pos % bs);
      o.input_ids[static_cast<size_t>(query_idx)] = input_id;
      o.query_positions[static_cast<size_t>(query_idx)] =
          std::min<int64_t>(query_pos, b.max_model_len - 1);
      o.query_slot_mapping[static_cast<size_t>(query_idx)] = q_slot;
      if (off >= sample_off) {
        const int32_t sample_idx = r * nspec + (off - sample_off);
        const int64_t spos = b.sample_from_anchor ? query_pos + 1 : query_pos;
        o.sample_indices[static_cast<size_t>(sample_idx)] = query_idx;
        o.sample_pos[static_cast<size_t>(sample_idx)] = spos;
        o.sample_idx_mapping[static_cast<size_t>(sample_idx)] = req_state_idx;
      }
    }

    o.query_start_loc[static_cast<size_t>(r)] = query_base;
    o.seq_lens[static_cast<size_t>(r)] = static_cast<int32_t>(last_valid_pos) + 1 + nqpr;
  }

  // --- Padding for CUDA-graph replay safety (kernel block_idx==0, req==last) ---
  const int32_t last_query_end = num_reqs * nqpr;
  for (int32_t i = num_reqs; i <= b.max_num_reqs; ++i)
    o.query_start_loc[static_cast<size_t>(i)] = last_query_end;
  // seq_lens[num_reqs, max_num_reqs) already 0 from assign.
  for (int32_t i = num_reqs * nspec; i < b.max_num_reqs * nspec; ++i) {
    o.sample_indices[static_cast<size_t>(i)] = 0;
    o.sample_pos[static_cast<size_t>(i)] = 0;
    o.sample_idx_mapping[static_cast<size_t>(i)] = -1;
  }
  for (int32_t i = num_reqs * nqpr; i < b.max_num_tokens; ++i)
    o.query_slot_mapping[static_cast<size_t>(i)] = kPadSlotId;
  return o;
}

std::vector<float> Qwen3DFlashModel::CombineAuxFeatures(const std::vector<float>& aux_features,
                                                        int64_t T,
                                                        const Qwen3DFlashWeights& weights,
                                                        const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t Fin = H * weights.num_taps;
  VT_CHECK(static_cast<int64_t>(aux_features.size()) == T * Fin,
           "qwen3_dflash fc: aux_features must be [T, H*num_taps]");
  // aux is [T, H*num_taps] f32 -> cast to bf16 -> fc MatmulBT -> [T,H] bf16.
  DBuf aux32(d, DType::kF32, {T, Fin}, aux_features.data());
  DBuf auxb(d, DType::kBF16, {T, Fin});
  vt::CastBf16(d.q, auxb.t(), aux32.t());
  Tensor wfc = ResidentWeight(d, weights.fc);  // [H, H*num_taps] nk
  DBuf comb(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, comb.t(), auxb.t(), wfc);
  DBuf comb32(d, DType::kF32, {T, H});
  vt::CastF32(d.q, comb32.t(), comb.t());
  std::vector<float> out(static_cast<size_t>(T) * H);
  comb32.Download(d, out.data());
  return out;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogits(
    const std::vector<int32_t>& input_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out,
    std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(input_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const int64_t vocab = weights.draft_vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_dflash: positions length must match input_ids");
  VT_CHECK(cu.size() >= 2 && cu.front() == 0 && cu.back() == static_cast<int32_t>(T),
           "qwen3_dflash: cu_seqlens must span [0,T]");
  VT_CHECK(weights.layers.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3_dflash: one layer weight per config.num_hidden_layers");

  // Embed: hidden[T,H] bf16 = embed_tokens[input_ids]; mask slots take
  // embed_tokens[mask_token_id] naturally (in-vocab), or the dedicated mask
  // embedding when present (qwen3_dflash.py:432-438).
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {config.vocab_size, H});
    DBuf dids(d, DType::kI32, {T}, input_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }
  if (!weights.mask_embedding.Empty() && weights.mask_token_id >= 0) {
    // Substitute the dedicated mask embedding for mask_token_id rows.
    Tensor mask_emb = ResidentWeight(d, weights.mask_embedding, {H});
    std::vector<float> mask_host(static_cast<size_t>(H));
    {
      DBuf tmp(d, DType::kF32, {H});
      vt::CastF32(d.q, tmp.t(), mask_emb);
      tmp.Download(d, mask_host.data());
    }
    std::vector<float> hidden_host(static_cast<size_t>(T) * H);
    {
      DBuf tmp(d, DType::kF32, {T, H});
      vt::CastF32(d.q, tmp.t(), hidden.t());
      tmp.Download(d, hidden_host.data());
    }
    for (int64_t r = 0; r < T; ++r)
      if (input_ids[static_cast<size_t>(r)] == weights.mask_token_id)
        for (int64_t j = 0; j < H; ++j)
          hidden_host[static_cast<size_t>(r * H + j)] = mask_host[static_cast<size_t>(j)];
    DBuf hf(d, DType::kF32, {T, H}, hidden_host.data());
    vt::CastBf16(d.q, hidden.t(), hf.t());
  }

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);
  DBuf dpos(d, DType::kI32, {T}, positions.data());

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    // input_layernorm (std add+RMSNorm): dhn = norm(hidden + res); res updated.
    Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {T, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

    // attention over the context-free block (routes through DFlashBlockAttention).
    // Reuse the block helper but feed the real positions to RoPE.
    DBuf attn = [&]() -> DBuf {
      const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
      DBuf q(d, DType::kBF16, {T, qdim});
      DBuf k(d, DType::kBF16, {T, kdim});
      DBuf v(d, DType::kBF16, {T, kdim});
      // Merged QKVParallelLinear: D1 folds the shared-input q/k/v GEMMs to ONE
      // MatmulBT over the merged owner + a contiguous QkvSplit (MergedQkvEnabled(),
      // VT_QWEN3_QKV_MERGE default ON; =0 = byte-identical 3-shard). RoPE handling
      // is UNCHANGED (still RopeNeox below — no RopeFromCache swap).
      Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
      if (MergedQkvEnabled()) {
        DBuf qkv(d, DType::kBF16, {T, qdim + 2 * kdim});
        vt::MatmulBT(d.q, qkv.t(), dhn.t(), wqkv);
        vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
      } else {
        Tensor wq = wqkv.Slice(0, 0, qdim);
        Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
        Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
        vt::MatmulBT(d.q, q.t(), dhn.t(), wq);
        vt::MatmulBT(d.q, k.t(), dhn.t(), wk);
        vt::MatmulBT(d.q, v.t(), dhn.t(), wv);
      }
      Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
      Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
      Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
      Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
      Tensor wqn = ResidentWeight(d, layer.q_norm, {Dh});
      Tensor wkn = ResidentWeight(d, layer.k_norm, {Dh});
      vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});
      vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
      vt::RopeNeox(d.q, q3, k3, dpos.t(), MakeRopeArgs(config));
      Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
      DBuf a(d, DType::kBF16, {T, Hq, Dh});
      vt::DFlashBlockAttentionArgs pa;
      pa.scale = scale;
      pa.causal = layer.attn_mode.causal;
      pa.sliding_window = layer.attn_mode.sliding_window;
      pa.cu_seqlens = cu.data();
      pa.num_reqs = static_cast<int>(cu.size()) - 1;
      vt::DFlashBlockAttention(d.q, a.t(), q3, k3, v3, pa);
      Tensor o_in = Reshape(a.t(), {T, Hq * Dh});
      Tensor wo = ResidentWeight(d, layer.o_proj);
      DBuf o(d, DType::kBF16, {T, H});
      vt::MatmulBT(d.q, o.t(), o_in, wo);
      return o;
    }();

    // post_attention_layernorm (std add+RMSNorm).
    Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {T, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

    // SwiGLU MLP: gate_up GEMM -> SiluAndMul -> down GEMM. gate_up+SiluAndMul run
    // through the SHARED bf16 gate-up MLP seam (layers::UnquantizedMlpGateUpMethod)
    // — byte-for-byte the same op sequence the inline path ran, now on the same
    // exemplar as qwen3.cpp MlpBlock. (Tier-A1 fold, arch-fusion-fold-plan.)
    const int64_t I = config.intermediate_size;
    DBuf act =
        layers::UnquantizedMlpGateUpMethod(&layer.gate_up_proj, I).Apply(d, dh2.t());
    Tensor wdn = ResidentWeight(d, layer.down_proj);
    DBuf down(d, DType::kBF16, {T, H});
    vt::MatmulBT(d.q, down.t(), act.t(), wdn);
    if (per_layer_out != nullptr) {
      DBuf tmp(d, DType::kF32, {T, H});
      vt::CastF32(d.q, tmp.t(), down.t());
      std::vector<float> lh(static_cast<size_t>(T) * H);
      tmp.Download(d, lh.data());
      per_layer_out->push_back(std::move(lh));
    }
    hidden = std::move(down);
  }

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  if (final_out != nullptr) {
    DBuf tmp(d, DType::kF32, {T, H});
    vt::CastF32(d.q, tmp.t(), dnorm.t());
    final_out->assign(static_cast<size_t>(T) * H, 0.0f);
    tmp.Download(d, final_out->data());
  }

  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, H});
  DBuf logits(d, DType::kF32, {T, vocab});
  vt::MatmulBT(d.q, logits.t(), dnorm.t(), lm);
  std::vector<float> out(static_cast<size_t>(T) * vocab);
  logits.Download(d, out.data());
  return out;
}

Qwen3DFlashModel::ContextKV Qwen3DFlashModel::PrecomputeContextKV(
    const std::vector<float>& context_states, const std::vector<int32_t>& context_positions,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t C = static_cast<int64_t>(context_positions.size());
  VT_CHECK(static_cast<int64_t>(context_states.size()) == C * H,
           "PrecomputeContextKV: context_states must be [num_ctx, H]");

  ContextKV ckv;
  ckv.num_ctx = C;
  if (C == 0) {
    ckv.k.assign(static_cast<size_t>(config.num_hidden_layers), {});
    ckv.v.assign(static_cast<size_t>(config.num_hidden_layers), {});
    return ckv;
  }

  // Device-resident projection (D7); download each layer's K/V to the host host
  // ContextKV the CPU/parity gates read. Bit-identical to the old inline path
  // (same ops, same order) — this public host API exists ONLY for the D3 kvprep
  // gates; production reaches the device buffers directly (ForwardBlockLogitsWithContext).
  ContextKVDev dev = PrecomputeContextKVDevice(d, context_states.data(),
                                               context_positions.data(), C, weights, config);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    std::vector<float> kh(static_cast<size_t>(C) * Hkv * Dh);
    std::vector<float> vh(static_cast<size_t>(C) * Hkv * Dh);
    DBuf tk(d, DType::kF32, {C, Hkv, Dh});
    vt::CastF32(d.q, tk.t(), Reshape(dev.k[static_cast<size_t>(l)].t(), {C, Hkv, Dh}));
    tk.Download(d, kh.data());
    DBuf tv(d, DType::kF32, {C, Hkv, Dh});
    vt::CastF32(d.q, tv.t(), Reshape(dev.v[static_cast<size_t>(l)].t(), {C, Hkv, Dh}));
    tv.Download(d, vh.data());
    ckv.k.push_back(std::move(kh));
    ckv.v.push_back(std::move(vh));
  }
  return ckv;
}

// Shared core (D9): the [context; block] block forward GIVEN a device-resident
// per-layer context K/V (ContextKVDev). Both ForwardBlockLogitsWithContext (which
// re-projects the whole context every step) and ForwardBlockLogitsWithPrecomputedKV
// (which uploads the persistent append-only store) build the ckv and delegate here,
// so the two paths are byte-identical downstream of how ckv's bits were obtained.
static std::vector<float> ForwardWithCtxKVDev(
    Dev d, const ContextKVDev& ckv, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    std::vector<std::vector<float>>* per_layer_out, std::vector<float>* final_out) {
  const int64_t Tq = static_cast<int64_t>(block_input_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const int64_t vocab = weights.draft_vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  const int num_reqs = static_cast<int>(cu.size()) - 1;
  VT_CHECK(static_cast<int64_t>(block_positions.size()) == Tq,
           "ForwardWithCtxKVDev: block_positions length must match input_ids");
  VT_CHECK(cu.size() >= 2 && cu.front() == 0 && cu.back() == static_cast<int32_t>(Tq),
           "ForwardWithCtxKVDev: cu must span [0,Tq]");
  VT_CHECK(static_cast<int>(ctx_cu.size()) == num_reqs + 1 && ctx_cu.front() == 0,
           "ForwardWithCtxKVDev: ctx_cu must be [num_reqs+1]");
  const int64_t C = ckv.num_ctx;
  VT_CHECK(ctx_cu.back() == static_cast<int32_t>(C),
           "ForwardWithCtxKVDev: ctx_cu.back() must equal num_ctx");

  // Combined [context; block] per-request layout for the attention (cu_comb), plus
  // the DEVICE index maps (D7) that place context/block rows into the combined
  // buffer with vt::IndexCopy and extract the block-query rows with vt::IndexSelect
  // — replacing the D5 host download + std::vector interleave + re-upload. These are
  // tiny integer maps computed once from the cu vectors and uploaded once.
  const int64_t Ncomb = C + Tq;
  std::vector<int32_t> cu_comb(static_cast<size_t>(num_reqs) + 1, 0);
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t cl = ctx_cu[static_cast<size_t>(r) + 1] - ctx_cu[static_cast<size_t>(r)];
    const int32_t bl = cu[static_cast<size_t>(r) + 1] - cu[static_cast<size_t>(r)];
    cu_comb[static_cast<size_t>(r) + 1] = cu_comb[static_cast<size_t>(r)] + cl + bl;
  }
  // ctx_dest[j] = combined row for context source row j (ctx_cu order).
  // blk_idx[i]  = combined row for block source row i (cu order); used BOTH to
  // scatter block q/k/v in (IndexCopy: comb[blk_idx[i]] = block[i]) AND to gather
  // block outputs back out (IndexSelect: out[i] = comb[blk_idx[i]]).
  std::vector<int32_t> ctx_dest(static_cast<size_t>(C));
  std::vector<int32_t> blk_idx(static_cast<size_t>(Tq));
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t c0 = ctx_cu[static_cast<size_t>(r)], c1 = ctx_cu[static_cast<size_t>(r) + 1];
    const int32_t b0 = cu[static_cast<size_t>(r)], b1 = cu[static_cast<size_t>(r) + 1];
    const int32_t base = cu_comb[static_cast<size_t>(r)];
    for (int32_t j = c0; j < c1; ++j) ctx_dest[static_cast<size_t>(j)] = base + (j - c0);
    const int32_t bbase = base + (c1 - c0);
    for (int32_t i = b0; i < b1; ++i) blk_idx[static_cast<size_t>(i)] = bbase + (i - b0);
  }
  DBuf ctx_dest_d(d, DType::kI32, {C}, ctx_dest.data());
  DBuf blk_idx_d(d, DType::kI32, {Tq}, blk_idx.data());

  // Embed block tokens; substitute the dedicated mask embedding when present.
  DBuf hidden(d, DType::kBF16, {Tq, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {config.vocab_size, H});
    DBuf dids(d, DType::kI32, {Tq}, block_input_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }
  if (!weights.mask_embedding.Empty() && weights.mask_token_id >= 0) {
    Tensor mask_emb = ResidentWeight(d, weights.mask_embedding, {H});
    std::vector<float> mask_host(static_cast<size_t>(H));
    {
      DBuf tmp(d, DType::kF32, {H});
      vt::CastF32(d.q, tmp.t(), mask_emb);
      tmp.Download(d, mask_host.data());
    }
    std::vector<float> hidden_host(static_cast<size_t>(Tq) * H);
    {
      DBuf tmp(d, DType::kF32, {Tq, H});
      vt::CastF32(d.q, tmp.t(), hidden.t());
      tmp.Download(d, hidden_host.data());
    }
    for (int64_t rr = 0; rr < Tq; ++rr)
      if (block_input_ids[static_cast<size_t>(rr)] == weights.mask_token_id)
        for (int64_t j = 0; j < H; ++j)
          hidden_host[static_cast<size_t>(rr * H + j)] = mask_host[static_cast<size_t>(j)];
    DBuf hf(d, DType::kF32, {Tq, H}, hidden_host.data());
    vt::CastBf16(d.q, hidden.t(), hf.t());
  }

  DBuf res(d, DType::kBF16, {Tq, H});
  res.Zero(d);
  DBuf dpos(d, DType::kI32, {Tq}, block_positions.data());

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

    // Block q/k/v: same per-layer path as the context-free forward.
    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    DBuf q(d, DType::kBF16, {Tq, qdim});
    DBuf k(d, DType::kBF16, {Tq, kdim});
    DBuf v(d, DType::kBF16, {Tq, kdim});
    Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
    vt::MatmulBT(d.q, q.t(), dhn.t(), wqkv.Slice(0, 0, qdim));
    vt::MatmulBT(d.q, k.t(), dhn.t(), wqkv.Slice(0, qdim, qdim + kdim));
    vt::MatmulBT(d.q, v.t(), dhn.t(), wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim));
    Tensor q2 = Reshape(q.t(), {Tq * Hq, Dh});
    Tensor k2 = Reshape(k.t(), {Tq * Hkv, Dh});
    Tensor q3 = Reshape(q.t(), {Tq, Hq, Dh});
    Tensor k3 = Reshape(k.t(), {Tq, Hkv, Dh});
    vt::RmsNorm(d.q, q2, q2, ResidentWeight(d, layer.q_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RmsNorm(d.q, k2, k2, ResidentWeight(d, layer.k_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RopeNeox(d.q, q3, k3, dpos.t(), MakeRopeArgs(config));

    // Build the combined [context; block] q/k/v ON DEVICE (D7): scatter the layer's
    // device context K/V and this block's q/k/v into the packed combined buffer via
    // vt::IndexCopy — NO D->H download, NO host std::vector interleave, NO re-upload.
    // The bf16 values are bit-identical to the D5 f32-roundtrip path (bf16->f32->bf16
    // is an identity round-trip), so DFlashBlockAttention sees identical inputs.
    Tensor v3 = Reshape(v.t(), {Tq, Hkv, Dh});
    DBuf qcb(d, DType::kBF16, {Ncomb, Hq, Dh});
    DBuf kcb(d, DType::kBF16, {Ncomb, Hkv, Dh});
    DBuf vcb(d, DType::kBF16, {Ncomb, Hkv, Dh});
    qcb.Zero(d);  // context query rows are unused (their attn output is discarded)
    Tensor qcb3 = qcb.t(), kcb3 = kcb.t(), vcb3 = vcb.t();
    if (C > 0) {  // this layer's device context K/V -> combined at ctx_dest
      Tensor ck2 = Reshape(ckv.k[static_cast<size_t>(l)].t(), {C, Hkv, Dh});
      Tensor cv2 = Reshape(ckv.v[static_cast<size_t>(l)].t(), {C, Hkv, Dh});
      Tensor cdst = ctx_dest_d.t();
      vt::IndexCopy(d.q, kcb3, ck2, cdst);
      vt::IndexCopy(d.q, vcb3, cv2, cdst);
    }
    {  // block q/k/v -> combined at blk_idx
      Tensor bidx = blk_idx_d.t();
      vt::IndexCopy(d.q, qcb3, q3, bidx);
      vt::IndexCopy(d.q, kcb3, k3, bidx);
      vt::IndexCopy(d.q, vcb3, v3, bidx);
    }
    // Attention over the combined sequence via the UNCHANGED D2 primitive.
    DBuf acomb(d, DType::kBF16, {Ncomb, Hq, Dh});
    vt::DFlashBlockAttentionArgs pa;
    pa.scale = scale;
    pa.causal = layer.attn_mode.causal;
    pa.sliding_window = layer.attn_mode.sliding_window;
    pa.cu_seqlens = cu_comb.data();
    pa.num_reqs = num_reqs;
    vt::DFlashBlockAttention(d.q, acomb.t(), qcb.t(), kcb.t(), vcb.t(), pa);
    // Extract the block-query rows out of the combined output ON DEVICE (IndexSelect:
    // a[i] = acomb[blk_idx[i]]), replacing the D5 download + host row-scatter.
    DBuf a(d, DType::kBF16, {Tq, Hq * Dh});
    {
      Tensor acomb2 = Reshape(acomb.t(), {Ncomb, Hq * Dh});
      Tensor a2 = Reshape(a.t(), {Tq, Hq * Dh});
      Tensor bidx = blk_idx_d.t();
      vt::IndexSelect(d.q, a2, acomb2, bidx);
    }
    Tensor wo = ResidentWeight(d, layer.o_proj);
    DBuf attn(d, DType::kBF16, {Tq, H});
    vt::MatmulBT(d.q, attn.t(), a.t(), wo);

    // post_attention_layernorm + SwiGLU MLP (unchanged from ForwardBlockLogits).
    Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
    const int64_t I = config.intermediate_size;
    Tensor wgu = ResidentWeight(d, layer.gate_up_proj);
    DBuf gu(d, DType::kBF16, {Tq, 2 * I});
    vt::MatmulBT(d.q, gu.t(), dh2.t(), wgu);
    DBuf act(d, DType::kBF16, {Tq, I});
    vt::SiluAndMul(d.q, act.t(), gu.t());
    Tensor wdn = ResidentWeight(d, layer.down_proj);
    DBuf down(d, DType::kBF16, {Tq, H});
    vt::MatmulBT(d.q, down.t(), act.t(), wdn);
    if (per_layer_out != nullptr) {
      DBuf tmp(d, DType::kF32, {Tq, H});
      vt::CastF32(d.q, tmp.t(), down.t());
      std::vector<float> lh(static_cast<size_t>(Tq) * H);
      tmp.Download(d, lh.data());
      per_layer_out->push_back(std::move(lh));
    }
    hidden = std::move(down);
  }

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {Tq, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  if (final_out != nullptr) {
    DBuf tmp(d, DType::kF32, {Tq, H});
    vt::CastF32(d.q, tmp.t(), dnorm.t());
    final_out->assign(static_cast<size_t>(Tq) * H, 0.0f);
    tmp.Download(d, final_out->data());
  }
  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, H});
  DBuf logits(d, DType::kF32, {Tq, vocab});
  vt::MatmulBT(d.q, logits.t(), dnorm.t(), lm);
  std::vector<float> out(static_cast<size_t>(Tq) * vocab);
  logits.Download(d, out.data());
  return out;
}

// Upload a persistent host bf16 PrecomputedContextKV into per-layer device buffers,
// producing the SAME ContextKVDev the full recompute (PrecomputeContextKVDevice) would
// build — the stored bf16 bits ARE the projection output (bit-identical, D9).
static ContextKVDev UploadContextKV(Dev d,
                                    const Qwen3DFlashModel::PrecomputedContextKV& store,
                                    const HfConfig& config) {
  const int64_t C = store.num_ctx;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  ContextKVDev out;
  out.num_ctx = C;
  VT_CHECK(store.k.size() == static_cast<size_t>(config.num_hidden_layers) &&
               store.v.size() == static_cast<size_t>(config.num_hidden_layers),
           "UploadContextKV: store must hold one K/V per hidden layer");
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    VT_CHECK(static_cast<int64_t>(store.k[static_cast<size_t>(l)].size()) == C * kdim &&
                 static_cast<int64_t>(store.v[static_cast<size_t>(l)].size()) == C * kdim,
             "UploadContextKV: per-layer K/V size must be num_ctx*kv_dim");
    DBuf k(d, DType::kBF16, {C, kdim},
           C > 0 ? store.k[static_cast<size_t>(l)].data() : nullptr);
    DBuf v(d, DType::kBF16, {C, kdim},
           C > 0 ? store.v[static_cast<size_t>(l)].data() : nullptr);
    out.k.push_back(std::move(k));
    out.v.push_back(std::move(v));
  }
  return out;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogitsWithContext(
    const std::vector<float>& context_states, const std::vector<int32_t>& context_positions,
    const std::vector<int32_t>& ctx_cu, const std::vector<int32_t>& block_input_ids,
    const std::vector<int32_t>& block_positions, const std::vector<int32_t>& cu,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue,
    std::vector<std::vector<float>>* per_layer_out, std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  // Per-step FULL recompute of the context K/V (D5/D7 path): re-projects the ENTIRE
  // growing context every step (O(context^2) total). The D9 persistent path
  // (ForwardBlockLogitsWithPrecomputedKV) replaces this with an append-only store.
  ContextKVDev ckv = PrecomputeContextKVDevice(d, context_states.data(),
                                               context_positions.data(),
                                               static_cast<int64_t>(context_positions.size()),
                                               weights, config);
  return ForwardWithCtxKVDev(d, ckv, ctx_cu, block_input_ids, block_positions, cu, weights,
                             config, per_layer_out, final_out);
}

void Qwen3DFlashModel::AppendContextKVHost(PrecomputedContextKV& store,
                                           const std::vector<float>& new_features,
                                           const std::vector<int32_t>& new_positions,
                                           const Qwen3DFlashWeights& weights,
                                           const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  const int64_t L = config.num_hidden_layers;
  const int64_t count = static_cast<int64_t>(new_positions.size());
  if (store.k.empty()) {
    store.k.assign(static_cast<size_t>(L), {});
    store.v.assign(static_cast<size_t>(L), {});
  }
  VT_CHECK(store.k.size() == static_cast<size_t>(L) && store.v.size() == static_cast<size_t>(L),
           "AppendContextKVHost: store layer count mismatch");
  VT_CHECK(static_cast<int64_t>(new_features.size()) == count * H,
           "AppendContextKVHost: new_features must be [count, H]");
  if (count == 0) return;
  // Project ONLY the `count` new rows (positions == their absolute positions), reusing
  // the EXACT per-row projection the full recompute runs, then download the bf16 K/V
  // and append. Per-row independence (hidden_norm/KV-GEMM/k_norm/RoPE) => these bits
  // equal what a full C-row recompute would produce for these same rows.
  ContextKVDev dev = PrecomputeContextKVDevice(d, new_features.data(), new_positions.data(),
                                               count, weights, config);
  for (int64_t l = 0; l < L; ++l) {
    std::vector<uint16_t> kh(static_cast<size_t>(count) * kdim);
    std::vector<uint16_t> vh(static_cast<size_t>(count) * kdim);
    dev.k[static_cast<size_t>(l)].Download(d, kh.data());  // raw bf16 bits
    dev.v[static_cast<size_t>(l)].Download(d, vh.data());
    std::vector<uint16_t>& sk = store.k[static_cast<size_t>(l)];
    std::vector<uint16_t>& sv = store.v[static_cast<size_t>(l)];
    sk.insert(sk.end(), kh.begin(), kh.end());
    sv.insert(sv.end(), vh.begin(), vh.end());
  }
  store.num_ctx += count;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogitsWithPrecomputedKV(
    const PrecomputedContextKV& ckv_host, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out,
    std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  ContextKVDev ckv = UploadContextKV(d, ckv_host, config);
  return ForwardWithCtxKVDev(d, ckv, ctx_cu, block_input_ids, block_positions, cu, weights,
                             config, per_layer_out, final_out);
}

// ============================ D11 Part A + D13 Part C substrate ============
// DEVICE-RESIDENT append-only draft-KV store, now a FIXED-CAPACITY PAGED cache (D13
// Part C substrate). Per draft layer a persistent pool [max_pages, block_size, Hkv, Dh]
// holds the projected bf16 K/V; a context row at absolute position p lives at flat slot
// p (page p/block_size, offset p%block_size), so the block_table is the IDENTITY and
// appending rows [L,L+count) is an IndexCopy scatter to slots [L,L+count). The store's
// shapes are STATIC (only seq_lens/block_table DATA changes as the context grows), so the
// Part B vt::DFlashPagedBlockAttention kernel reads it capture-safely and the Part C
// draft-step CUDA graph can be captured over these persistent buffers. BIT-IDENTICAL to
// the D9/D11 contiguous store by per-row projection independence (same
// PrecomputeContextKVDevice, same ascending-position append order) — the bf16 bits are
// merely placed at fixed paged slots instead of appended chunks; tokens+acceptance are
// unchanged.
constexpr int64_t kDflashPageSize = 16;       // rows per paged context page (block_size)
constexpr int64_t kDflashMaxCtxSlots = 4096;  // fixed store capacity (== max_pages*page)

struct DflashDeviceKVStore {
  // Per draft layer: a persistent bf16 paged pool [max_pages, block_size, Hkv, Dh].
  std::vector<DBuf> pool_k;
  std::vector<DBuf> pool_v;
  std::unique_ptr<DBuf> block_table;  // [1, max_pages] i32 identity (persistent)
  std::unique_ptr<DBuf> seq_lens;     // [1] i32 = num_ctx (persistent, updated on append)
  int64_t num_layers = 0;
  int64_t num_ctx = 0;
  int64_t max_pages = 0;
  int64_t block_size = 0;
  int64_t kdim = 0;  // Hkv*Dh

  // D13 Part C — per-request CUDA graph over the persistent (1+k) paged draft step.
  // All graph inputs are persistent device buffers: g_hidden (embed target, refreshed
  // OUTSIDE the graph each step), g_dpos (block positions, refreshed in place), g_cu
  // (cu_seqlens {0,Tq}, constant), plus the store's own pools/block_table/seq_lens.
  // The growing context enters purely through the in-place seq_lens VALUE, so the same
  // captured graph replays as the context grows. g_logits holds the graph output.
  std::unique_ptr<DBuf> g_hidden;   // [Tq, H] bf16
  std::unique_ptr<DBuf> g_dpos;     // [Tq] i32
  std::unique_ptr<DBuf> g_cu;       // [2] i32 {0, Tq}
  std::unique_ptr<DBuf> g_logits;   // [Tq, vocab] f32 (persistent graph output)
  // ENG-CUDAGRAPH-BREAK W5 (#1335): the instantiated graph, the ownership of its
  // handle, its release and its `captured()` state live in the SHARED SEAM
  // instead of in a raw `void*` plus a `Backend*` this store kept alive only so
  // its destructor could call `DestroyGraph`. `vt::BreakableGraph` releases every
  // segment it holds through `Backend::DestroyGraph`, which is the routing that
  // lets ENG-CUDAGRAPH-DEDUP (#1162) interpose at the backend later without
  // editing this file.
  vt::BreakableGraph g_graph;
  int64_t g_tq = -1;                // captured (1+k); -1 = not yet
  int g_state = 0;                  // 0 cold, 1 warm (pool warmed, capture next), 2 captured
};

std::shared_ptr<DflashDeviceKVStore> Qwen3DFlashModel::MakeDeviceKVStore(
    const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t L = config.num_hidden_layers;
  auto s = std::make_shared<DflashDeviceKVStore>();
  s->num_layers = L;
  s->block_size = kDflashPageSize;
  s->max_pages = kDflashMaxCtxSlots / kDflashPageSize;
  s->kdim = Hkv * Dh;
  s->pool_k.reserve(static_cast<size_t>(L));
  s->pool_v.reserve(static_cast<size_t>(L));
  for (int64_t l = 0; l < L; ++l) {
    s->pool_k.emplace_back(d, DType::kBF16,
                           std::vector<int64_t>{s->max_pages, s->block_size, Hkv, Dh});
    s->pool_v.emplace_back(d, DType::kBF16,
                           std::vector<int64_t>{s->max_pages, s->block_size, Hkv, Dh});
  }
  // Identity block_table (logical page p -> physical page p) + zero seq_lens, uploaded
  // once; these persistent device buffers never move, so a captured graph reads the
  // growing context purely through the in-place seq_lens value.
  std::vector<int32_t> bt(static_cast<size_t>(s->max_pages));
  for (int64_t p = 0; p < s->max_pages; ++p) bt[static_cast<size_t>(p)] = static_cast<int32_t>(p);
  s->block_table = std::make_unique<DBuf>(d, DType::kI32,
                                          std::vector<int64_t>{1, s->max_pages}, bt.data());
  const int32_t zero = 0;
  s->seq_lens = std::make_unique<DBuf>(d, DType::kI32, std::vector<int64_t>{1}, &zero);
  return s;
}

int64_t Qwen3DFlashModel::DeviceKVNumCtx(const DflashDeviceKVStore& store) {
  return store.num_ctx;
}

void Qwen3DFlashModel::AppendContextKVDevice(DflashDeviceKVStore& store,
                                             const std::vector<float>& new_features,
                                             const std::vector<int32_t>& new_positions,
                                             const Qwen3DFlashWeights& weights,
                                             const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t L = config.num_hidden_layers;
  const int64_t count = static_cast<int64_t>(new_positions.size());
  VT_CHECK(store.pool_k.size() == static_cast<size_t>(L) &&
               store.pool_v.size() == static_cast<size_t>(L),
           "AppendContextKVDevice: store layer count mismatch (call MakeDeviceKVStore)");
  VT_CHECK(static_cast<int64_t>(new_features.size()) == count * config.hidden_size,
           "AppendContextKVDevice: new_features must be [count, H]");
  if (count == 0) return;
  const int64_t L0 = store.num_ctx;
  const int64_t max_slots = store.max_pages * store.block_size;
  VT_CHECK(L0 + count <= max_slots,
           "AppendContextKVDevice: paged store capacity exceeded (raise kDflashMaxCtxSlots)");
  // The runner appends only accepted-prefix rows in ascending order, so the new rows sit
  // at contiguous absolute positions [L0, L0+count) == identity paged slots [L0, L0+count).
  VT_CHECK(new_positions.front() == static_cast<int32_t>(L0) &&
               new_positions.back() == static_cast<int32_t>(L0 + count - 1),
           "AppendContextKVDevice: new_positions must be contiguous [num_ctx, num_ctx+count)");
  // Project the new rows on device (the EXACT op the D9/D11 store ran), then IndexCopy-
  // scatter each layer's [count,kdim] K/V into the fixed pools at slots [L0,L0+count).
  ContextKVDev dev = PrecomputeContextKVDevice(d, new_features.data(), new_positions.data(),
                                               count, weights, config);
  std::vector<int32_t> slot(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) slot[static_cast<size_t>(i)] = static_cast<int32_t>(L0 + i);
  DBuf slot_d(d, DType::kI32, {count}, slot.data());
  for (int64_t l = 0; l < L; ++l) {
    Tensor pk = Reshape(store.pool_k[static_cast<size_t>(l)].t(), {max_slots, store.kdim});
    Tensor pv = Reshape(store.pool_v[static_cast<size_t>(l)].t(), {max_slots, store.kdim});
    vt::IndexCopy(d.q, pk, dev.k[static_cast<size_t>(l)].t(), slot_d.t());
    vt::IndexCopy(d.q, pv, dev.v[static_cast<size_t>(l)].t(), slot_d.t());
  }
  store.num_ctx = L0 + count;
  // Update the persistent seq_lens (the paged kernel's context bound) in place.
  const int32_t nc = static_cast<int32_t>(store.num_ctx);
  d.b.Copy(queue, store.seq_lens->ptr(), &nc, sizeof(int32_t));
}

// Whether the single-request DFlash block forward runs through the capture-safe PAGED
// kernel (Part B) reading the persistent paged store, vs the D11 materialized
// [context;block] path. Default ON (the D13 production path); VT_DFLASH_PAGED=0 selects
// the materialized fallback for a same-binary A/B.
static bool UsePagedDflashForward() {
  const char* e = std::getenv("VT_DFLASH_PAGED");
  return e == nullptr || e[0] != '0';
}

// Whether the single-request paged draft step is CUDA-graph captured + replayed (D13
// Part C). Default ON; VT_DFLASH_GRAPH=0 keeps the eager paged path (a same-binary A/B
// of exactly the capture lever, over the identical paged forward).
static bool UseDflashGraph() {
  const char* e = std::getenv("VT_DFLASH_GRAPH");
  return e == nullptr || e[0] != '0';
}

// Capture/replay counters (proof the graph path RAN; printed when VT_DFLASH_GRAPH_STATS set).
static int64_t g_dflash_captures = 0;
static int64_t g_dflash_replays = 0;
static bool DflashGraphStats() {
  static const bool on = std::getenv("VT_DFLASH_GRAPH_STATS") != nullptr;
  return on;
}

// Capture-safe static-shape single-request draft block forward (D13 Part C). Runs the
// (1+k) block over the PAGED context store via vt::DFlashPagedBlockAttention (no
// [context;block] materialization, no function-local host uploads of ctx/blk index maps).
// The embedding is done OUTSIDE by the caller (device flag + sync stays out of any
// capture region); `hidden_in`/`dpos`/`cu_seqlens` are the caller-owned inputs (eager:
// per-call DBufs; capture: the graph slot's persistent buffers). Returns [Tq, vocab] f32
// logits ON DEVICE (the caller downloads + samples OUTSIDE the graph). Bit-identical to
// ForwardWithCtxKVDev over the same context (Part B == materialized DFlashBlockAttention).
static DBuf ForwardPagedBody(Dev d, const DflashDeviceKVStore& store, const Tensor& hidden_in,
                             const Tensor& dpos, const Tensor& cu_seqlens,
                             const Qwen3DFlashWeights& weights, const HfConfig& config) {
  const int64_t Tq = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const int64_t vocab = weights.draft_vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  Tensor cur = hidden_in;
  std::vector<DBuf> keep;  // keep each layer's post-MLP `down` alive across iterations
  keep.reserve(static_cast<size_t>(config.num_hidden_layers));
  DBuf res(d, DType::kBF16, {Tq, H});
  res.Zero(d);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dhn.t(), cur, w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dhn.t(), cur, w_in, vt::RmsNormArgs{eps, false}, &res.t());

    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    DBuf q(d, DType::kBF16, {Tq, qdim});
    DBuf k(d, DType::kBF16, {Tq, kdim});
    DBuf v(d, DType::kBF16, {Tq, kdim});
    Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
    vt::MatmulBT(d.q, q.t(), dhn.t(), wqkv.Slice(0, 0, qdim));
    vt::MatmulBT(d.q, k.t(), dhn.t(), wqkv.Slice(0, qdim, qdim + kdim));
    vt::MatmulBT(d.q, v.t(), dhn.t(), wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim));
    Tensor q2 = Reshape(q.t(), {Tq * Hq, Dh});
    Tensor k2 = Reshape(k.t(), {Tq * Hkv, Dh});
    Tensor q3 = Reshape(q.t(), {Tq, Hq, Dh});
    Tensor k3 = Reshape(k.t(), {Tq, Hkv, Dh});
    Tensor v3 = Reshape(v.t(), {Tq, Hkv, Dh});
    vt::RmsNorm(d.q, q2, q2, ResidentWeight(d, layer.q_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RmsNorm(d.q, k2, k2, ResidentWeight(d, layer.k_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RopeNeox(d.q, q3, k3, dpos, MakeRopeArgs(config));

    // Paged in-block attention over the persistent paged context store (Part B). The
    // output is the Tq block-query rows directly (no combined buffer, no IndexSelect).
    DBuf a3(d, DType::kBF16, {Tq, Hq, Dh});
    vt::DFlashPagedBlockAttentionArgs pa;
    pa.scale = scale;
    pa.causal = layer.attn_mode.causal;
    pa.sliding_window = layer.attn_mode.sliding_window;
    pa.num_reqs = 1;
    pa.block_size = store.block_size;
    vt::DFlashPagedBlockAttention(d.q, a3.t(), q3, k3, v3, store.pool_k[static_cast<size_t>(l)].t(),
                                  store.pool_v[static_cast<size_t>(l)].t(), cu_seqlens,
                                  store.seq_lens->t(), store.block_table->t(), pa);
    Tensor a = Reshape(a3.t(), {Tq, Hq * Dh});
    Tensor wo = ResidentWeight(d, layer.o_proj);
    DBuf attn(d, DType::kBF16, {Tq, H});
    vt::MatmulBT(d.q, attn.t(), a, wo);

    Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
    const int64_t I = config.intermediate_size;
    Tensor wgu = ResidentWeight(d, layer.gate_up_proj);
    DBuf gu(d, DType::kBF16, {Tq, 2 * I});
    vt::MatmulBT(d.q, gu.t(), dh2.t(), wgu);
    DBuf act(d, DType::kBF16, {Tq, I});
    vt::SiluAndMul(d.q, act.t(), gu.t());
    Tensor wdn = ResidentWeight(d, layer.down_proj);
    DBuf down(d, DType::kBF16, {Tq, H});
    vt::MatmulBT(d.q, down.t(), act.t(), wdn);
    keep.push_back(std::move(down));
    cur = keep.back().t();
  }

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {Tq, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), cur, w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), cur, w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, H});
  DBuf logits(d, DType::kF32, {Tq, vocab});
  vt::MatmulBT(d.q, logits.t(), dnorm.t(), lm);
  return logits;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
    const std::vector<DflashDeviceKVStore*>& stores, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out,
    std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t L = config.num_hidden_layers;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  const int P = static_cast<int>(stores.size());
  VT_CHECK(static_cast<int>(ctx_cu.size()) == P + 1 && ctx_cu.front() == 0,
           "ForwardBlockLogitsWithDeviceKV: ctx_cu must be [num_reqs+1]");
  const int64_t C = ctx_cu.back();

  // D13 Part C — the production single-request path: run the (1+k) block through the
  // capture-safe PAGED kernel reading the persistent paged store directly (no combined
  // buffer, no function-local host index maps). The single-request propose (c1 speed gate
  // + the e2e gate, which drives one request at a time) is exactly this case.
  if (P == 1 && weights.mask_embedding.Empty() && per_layer_out == nullptr &&
      final_out == nullptr && UsePagedDflashForward()) {
    DflashDeviceKVStore& st = *stores[0];
    const int64_t Tq = static_cast<int64_t>(block_input_ids.size());
    const int64_t H = config.hidden_size;
    const int64_t vocab = weights.draft_vocab_size;
    VT_CHECK(static_cast<int64_t>(block_positions.size()) == Tq,
             "ForwardBlockLogitsWithDeviceKV(paged): positions length mismatch");
    VT_CHECK(cu.size() == 2 && cu.front() == 0 && cu.back() == static_cast<int32_t>(Tq),
             "ForwardBlockLogitsWithDeviceKV(paged): single-request cu must be {0,Tq}");
    VT_CHECK(ctx_cu.back() == static_cast<int32_t>(st.num_ctx),
             "ForwardBlockLogitsWithDeviceKV(paged): ctx_cu.back() must equal store num_ctx");

    // `vt::GraphCaptureEnabled()` is the THIRD conjunct and it is not decoration
    // (#1352, found and fixed while landing #1335). Before W5 this driver's
    // capture was its own `BeginCapture` pair, so `VLLM_CPP_CUDAGRAPH` could not
    // reach it and the two conjuncts below were the whole predicate. The capture
    // is now the seam's, and the seam reads that switch itself — so without this
    // conjunct `VLLM_CPP_CUDAGRAPH=0` would still route into the CAPTURE lane,
    // run the eager warm pass, open an INERT scope, and run the whole
    // `ForwardPagedBody` a SECOND time inside it. Two full draft forwards per
    // propose, forever, because the driver would never reach `g_state == 2`.
    // Not wrong, just wasteful, which is exactly the kind of defect that
    // survives a token gate. Asking here makes the switch select this driver's
    // existing single-forward eager path, which is what it means everywhere else.
    const bool graph_ok =
        UseDflashGraph() && vt::GraphCaptureEnabled() && d.b.SupportsGraphCapture() &&
        platforms::GetPlatform(queue.device.type).support_static_graph_mode();

    // --- Eager paged path (VT_DFLASH_GRAPH=0, VLLM_CPP_CUDAGRAPH=0, or capture
    //     unsupported) --------------------------------------------------------
    if (!graph_ok) {
      DBuf hidden(d, DType::kBF16, {Tq, H});
      {
        Tensor dtab = ResidentWeight(d, weights.embed_tokens, {config.vocab_size, H});
        DBuf dids(d, DType::kI32, {Tq}, block_input_ids.data());
        vt::Embedding(d.q, hidden.t(), dtab, dids.t());
      }
      DBuf dpos(d, DType::kI32, {Tq}, block_positions.data());
      const std::vector<int32_t> cus = {0, static_cast<int32_t>(Tq)};
      DBuf cu_d(d, DType::kI32, {2}, cus.data());
      DBuf logits = ForwardPagedBody(d, st, hidden.t(), dpos.t(), cu_d.t(), weights, config);
      std::vector<float> out(static_cast<size_t>(Tq) * vocab);
      logits.Download(d, out.data());
      return out;
    }

    // --- CUDA-graph path (D13 Part C): cold (eager warm) -> warm (capture) -> replay.
    // (Re)allocate the persistent graph inputs when the block width changes (k is fixed
    // per config, so this fires once per request lifetime). A width change invalidates a
    // prior graph.
    if (st.g_tq != Tq) {
      // Reset() releases every segment through Backend::DestroyGraph and returns
      // the container to its as-constructed state, which is also what lets the
      // next capture open a scope on it: the scope REFUSES a container that
      // already holds one, because appending to it would leave
      // `break_count() == segment_count()` and Replay would drop the last break.
      st.g_graph.Reset();
      st.g_hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{Tq, H});
      st.g_dpos = std::make_unique<DBuf>(d, DType::kI32, std::vector<int64_t>{Tq});
      const std::vector<int32_t> cus = {0, static_cast<int32_t>(Tq)};
      st.g_cu = std::make_unique<DBuf>(d, DType::kI32, std::vector<int64_t>{2}, cus.data());
      st.g_logits.reset();
      st.g_tq = Tq;
      st.g_state = 0;
    }
    // Refresh the persistent graph inputs IN PLACE (fixed addresses; only contents move),
    // ALWAYS OUTSIDE the captured region: embed the block tokens into g_hidden and copy
    // this step's block positions into g_dpos. seq_lens/block_table already updated in the
    // store (append), and cu_seqlens is the constant {0,Tq}.
    {
      Tensor dtab = ResidentWeight(d, weights.embed_tokens, {config.vocab_size, H});
      DBuf dids(d, DType::kI32, {Tq}, block_input_ids.data());
      vt::Embedding(d.q, st.g_hidden->t(), dtab, dids.t());
    }
    d.b.Copy(queue, st.g_dpos->ptr(), block_positions.data(),
             static_cast<size_t>(Tq) * sizeof(int32_t));

    std::vector<float> out(static_cast<size_t>(Tq) * vocab);
    if (st.g_state == 2) {
      // Captured: relaunch the graph over the refreshed persistent inputs + grown context
      // (which enters purely via the in-place seq_lens value + paged store), then download.
      // Through the seam's container, never `Backend::ReplayGraph` directly: the
      // container replays its segments in order (one, here, because this capture
      // is kFull) and owns the G3 replay counter the reachability gate reads.
      st.g_graph.Replay(queue);
      st.g_logits->Download(d, out.data());
      if (DflashGraphStats()) {
        ++g_dflash_replays;
        if (g_dflash_replays % 32 == 0)
          std::fprintf(stderr, "[DFLASH-GRAPH] replays=%lld captures=%lld\n",
                       static_cast<long long>(g_dflash_replays),
                       static_cast<long long>(g_dflash_captures));
      }
      return out;
    }

    // First propose step for this request (g_state 0): WARM-then-CAPTURE in this same step.
    // A full 27B target verify forward + KV append runs between draft steps and perturbs
    // the shared DevicePool, so an eager pass from a *previous* step does NOT guarantee the
    // free-list holds the draft's size classes at capture time (a Get miss -> cudaMalloc is
    // forbidden mid-capture). Running one eager ForwardPagedBody HERE, immediately before
    // BeginCapture with no intervening allocation, returns exactly the draft's peak-concurrent
    // blocks to the free-list (retire-don't-free, GB10 pool cap 0), so the capture's identical
    // allocation sequence is a pure pool hit. The eager pass also warms resident draft weights,
    // the RoPE cache and the cuBLASLt/workspace scratch. Its result IS this step's output.
    {
      DBuf warm_lg = ForwardPagedBody(d, st, st.g_hidden->t(), st.g_dpos->t(), st.g_cu->t(),
                                      weights, config);
      warm_lg.Download(d, out.data());
    }  // warm_lg + all ForwardPagedBody scratch freed to the pool free-list here.
    // ENG-CUDAGRAPH-BREAK W5 (#1335): the capture is the SHARED SEAM's, not this
    // driver's hand-rolled `BeginCapture`/`EndCaptureGraph` pair with its own
    // `try`/drain. The scope owns the segment, the handle, its release, the drain
    // a mid-capture throw needs, and the G3 counters.
    //
    // kFULL, INHERITED FROM W2 AND NOT RE-ARGUED. vLLM's v1 default
    // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63` @ pin `5559679229`)
    // is documented at `:630-632` as a FULL graph for DECODE batches and a
    // piecewise one for prefill and mixed batches, and `decode_mode()` (`:65-66`)
    // returns the full half. This is the (1+k) DRAFT step of a speculative
    // decode, which is a decode batch, so its capture is ONE segment with the
    // attention calls INSIDE it — byte-identical in shape to the region this
    // replaces. Opening it kPiecewise would turn every draft layer's attention
    // into an eager call between two graph replays, which is not vLLM's decode
    // behaviour and which nothing in this row's record supports.
    std::optional<DBuf> lg;
    {
      vt::GraphCaptureScope scope(d.b, queue, st.g_graph, vt::GraphCaptureMode::kFull);
      lg = ForwardPagedBody(d, st, st.g_hidden->t(), st.g_dpos->t(), st.g_cu->t(),
                            weights, config);
    }  // ~GraphCaptureScope closes the segment and files it on st.g_graph
    // NOT CAPTURED covers TWO states, and only one of them may continue.
    //
    //   * INERT (`capture_failed() == false`): unreachable here, because
    //     `graph_ok` above already required `SupportsGraphCapture()`; the
    //     remaining inert cause is `VLLM_CPP_CUDAGRAPH=0`, which the seam reads
    //     and this driver no longer does. The region ran EAGERLY, `*lg` is a real
    //     result, and the step falls back to the eager lane for good.
    //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph` threw.
    //     Under stream capture NOTHING between `BeginCapture` and the throw
    //     executed — every kernel was RECORDED — so `*lg` is pool-recycled memory
    //     and downloading it would hand this draft step uncomputed device memory
    //     as its logits. No fault, and a token gate cannot see it, because a
    //     draft the target rejects is indistinguishable from a bad draft.
    //
    // The pre-W5 driver rethrew after draining, and so does this.
    if (!st.g_graph.captured()) {
      if (st.g_graph.capture_failed()) {
        const std::exception_ptr err = st.g_graph.capture_error();
        st.g_graph.Reset();  // clear the failure with the graph it described
        std::fprintf(stderr, "[DFLASH-GRAPH] capture FAILED\n");
        if (err) std::rethrow_exception(err);
        VT_CHECK(false,
                 "DFlash draft graph: the capture was ABANDONED and its logits were "
                 "never computed; refusing to return uncaptured device memory");
      }
      // INERT is now UNREACHABLE from here: `graph_ok` above required both
      // `vt::GraphCaptureEnabled()` and `SupportsGraphCapture()`, which are the
      // only two things that make a scope inert. Kept as a total branch rather
      // than an assertion, because a future inert cause added to the seam must
      // degrade to a correct eager step rather than to undefined behaviour —
      // the region DID run eagerly, so `*lg` holds real values.
      st.g_state = 0;  // stay eager, and re-warm rather than re-capture
      lg->Download(d, out.data());
      return out;
    }
    st.g_logits = std::make_unique<DBuf>(std::move(*lg));
    if (DflashGraphStats()) {
      ++g_dflash_captures;
      std::fprintf(stderr, "[DFLASH-GRAPH] captured #%lld Tq=%lld C=%lld\n",
                   static_cast<long long>(g_dflash_captures), static_cast<long long>(Tq),
                   static_cast<long long>(st.num_ctx));
    }
    st.g_state = 2;  // subsequent steps replay
    return out;      // this step's output is the eager warm pass (bit-identical to the graph)
  }

  // Fallback (P>1, a separate mask embedding, a parity dump, or VT_DFLASH_PAGED=0):
  // materialize one combined per-layer [C, kdim] context by gathering each store's
  // [0,num_ctx) paged slots (ctx_cu order == ascending-position), then run the D11
  // materialized forward. Bit-identical to the paged path; not capture-targeted.
  ContextKVDev ckv;
  ckv.num_ctx = C;
  for (int64_t l = 0; l < L; ++l) {
    ckv.k.emplace_back(d, DType::kBF16, std::vector<int64_t>{C, kdim});
    ckv.v.emplace_back(d, DType::kBF16, std::vector<int64_t>{C, kdim});
  }
  if (C > 0) {
    int64_t off = 0;
    for (int r = 0; r < P; ++r) {
      const DflashDeviceKVStore& st = *stores[static_cast<size_t>(r)];
      VT_CHECK(static_cast<int64_t>(st.pool_k.size()) == L &&
                   static_cast<int64_t>(st.pool_v.size()) == L,
               "ForwardBlockLogitsWithDeviceKV: store layer count mismatch");
      const int64_t cr = st.num_ctx;
      if (cr == 0) continue;
      const int64_t max_slots = st.max_pages * st.block_size;
      std::vector<int32_t> gidx(static_cast<size_t>(cr)), didx(static_cast<size_t>(cr));
      for (int64_t i = 0; i < cr; ++i) {
        gidx[static_cast<size_t>(i)] = static_cast<int32_t>(i);         // paged slot i
        didx[static_cast<size_t>(i)] = static_cast<int32_t>(off + i);   // combined row
      }
      DBuf gidx_d(d, DType::kI32, {cr}, gidx.data());
      DBuf didx_d(d, DType::kI32, {cr}, didx.data());
      for (int64_t l = 0; l < L; ++l) {
        Tensor srck = Reshape(st.pool_k[static_cast<size_t>(l)].t(), {max_slots, kdim});
        Tensor srcv = Reshape(st.pool_v[static_cast<size_t>(l)].t(), {max_slots, kdim});
        DBuf tmpk(d, DType::kBF16, {cr, kdim});
        DBuf tmpv(d, DType::kBF16, {cr, kdim});
        vt::IndexSelect(d.q, tmpk.t(), srck, gidx_d.t());
        vt::IndexSelect(d.q, tmpv.t(), srcv, gidx_d.t());
        vt::IndexCopy(d.q, ckv.k[static_cast<size_t>(l)].t(), tmpk.t(), didx_d.t());
        vt::IndexCopy(d.q, ckv.v[static_cast<size_t>(l)].t(), tmpv.t(), didx_d.t());
      }
      off += cr;
    }
    VT_CHECK(off == C,
             "ForwardBlockLogitsWithDeviceKV: gathered ctx rows != ctx_cu.back()");
  }
  return ForwardWithCtxKVDev(d, ckv, ctx_cu, block_input_ids, block_positions, cu, weights,
                             config, per_layer_out, final_out);
}

}  // namespace vllm
