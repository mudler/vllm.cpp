// Maple (`MapleForCausalLM`) forward — composes the shared dense attention
// block and a maple-specific MoE block, mirroring qwen3_moe.cpp's layer
// structure with THREE maple-specific deltas, each cited to
// deepgrove-ai/llama.cpp src/models/maple.cpp @ 7e30f3a:
//
//   1. PER-LAYER ROPE INVERSION (maple.cpp:96-106): RoPE applies ONLY on the
//      SWA layers; the global layers {3,7,11,15,19,23} skip it entirely
//      (nope_on_global_attention=true). Every standard arch ropes the opposite
//      set. Handled by an ADDITIVE rope-skip arm in dense_attn_block.h
//      (MapleAttnBlock: rotary_dim==0 => no rope) plus the SWA window.
//   2. SLIDING WINDOW on the 18 local layers (window-1 = 511 left context,
//      gemma2.cpp:198 plumbing): pa.window_size = AttentionWindow{511, 0}.
//   3. CLAMPED SWIGLU experts (maple.cpp:21 swiglu_clamp_exp.fill(7.0f)): the
//      activated gate is clamped to ±7 BEFORE the up multiply. RunMoeBlock has
//      no clamp hook, so the expert MLP body is local to this file; router /
//      top-k / combine still run through the shared vt ops.
//
// Numeric contract: bf16 residual stream, per-op bf16 rounding exactly as
// qwen3_moe.cpp; router fp32 softmax+top-k+renorm via vt::MoeRouterTopK;
// combine via vt::MoeCombine with NO shared expert term. Returns [T,vocab]
// f32 logits.
#include "vllm/model_executor/models/maple.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"   // AttnBlock glue + MapleAttnBlock
#include "vllm/v1/attention/backend.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// dense_attn glue types the local matmul bodies use.
using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::ResidentWeight;
using dense_attn::StepInputs;

constexpr float kMapleSwigluClamp = 7.0f;

inline float ClampedSiluMul(float g, float u) {
  // Clamped SwiGLU EXACTLY as llama-maple builds it (llama-graph.cpp:2146-2155,
  // LLM_ARCH_MAPLE arm): UP is clamped symmetrically to +-limit, GATE is
  // clamped ONLY above ((-inf, limit]), then out = silu(gate)*up. Clamping the
  // silu output instead deviates from the reference wherever |up| > limit or
  // the gate sits between its own clamp points.
  const float uc = std::min(std::max(u, -kMapleSwigluClamp), kMapleSwigluClamp);
  const float gc = std::min(g, kMapleSwigluClamp);
  const float s = gc / (1.0F + std::exp(-gc));
  return s * uc;
}

std::vector<float> MapleMatmulF32(Dev d, const std::vector<uint16_t>& x, int64_t M,
                             int64_t K, const OwnedTensor& w) {
  const int64_t N = w.nk ? w.shape[0] : w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kF32, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), dx.t(), dw);
  else
    vt::Matmul(d.q, dout.t(), dx.t(), dw);
  std::vector<float> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

std::vector<uint16_t> MapleMatmulBf16(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                 int64_t K, const OwnedTensor& w) {
  const int64_t N = w.nk ? w.shape[0] : w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kBF16, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), dx.t(), dw);
  else
    vt::Matmul(d.q, dout.t(), dx.t(), dw);
  std::vector<uint16_t> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// Per-expert clamped-SwiGLU MLP over gathered rows x [n,H] bf16 -> [n,H] bf16,
// bf16 (non-keep-quant) arm. The keep-quant arm now runs as 3 grouped GEMMs in
// MapleMoeBlock (vt::MatmulBTQuantGrouped), so the per-expert TQ slice helpers
// (MapleMatmulF32Slice/MapleMatmulBf16Slice/MapleExpertMlpKqClamped) are gone.
std::vector<uint16_t> MapleExpertMlpBf16Clamped(
    Dev d, const OwnedTensor& gate, const OwnedTensor& up,
    const OwnedTensor& down, const std::vector<uint16_t>& x, int64_t n,
    int64_t H, int64_t I) {
  std::vector<float> hg = MapleMatmulF32(d, x, n, H, gate);
  std::vector<float> hu = MapleMatmulF32(d, x, n, H, up);
  std::vector<uint16_t> act(static_cast<size_t>(n) * I);
  for (size_t i = 0; i < act.size(); ++i)
    act[i] = vt::F32ToBF16(ClampedSiluMul(hg[i], hu[i]));
  return MapleMatmulBf16(d, act, n, I, down);
}

}  // namespace

// Names the out-of-anonymous-namespace functions need (the anon-ns block above
// imports them internally; these public functions cannot see those imports).
using dense_attn::BuildStepInputs;
using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::FusedChainAdoptEnabled;
using dense_attn::ResidentWeight;
using dense_attn::StepInputs;

// ─── MoE block (router + clamped experts + combine) ────────────────────────
//
// Phase 2 on-device pipeline: the ENTIRE MoE block runs without a single host
// drain. The router matmul, top-k, fused gate+up+SwiGLU grouped GEMM, down
// grouped GEMM, and combine are all back-to-back device dispatches that batch
// naturally. The only host read in the whole layer is the final logits download
// in MapleForward. This is what closes the ~2 s/token host-overhead gap: the
// ~600 synchronous per-expert dispatches + per-dispatch FlushBatch drains of
// the original path become 4 dispatches/layer with zero drains.
DBuf MapleMoeBlock(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                   const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t P = T * top_k;
  const bool kq = !w.expert_gate_kq.Empty();

  // Router: logits = dh @ router_gate.T — all on device, no download.
  // dh is [T, H] bf16 on device; router_gate is [H, E] (F32 or bf16).
  Tensor w_router = ResidentWeight(d, w.router_gate);
  DBuf dlog(d, DType::kBF16, {T, E});
  {
    Tensor dh_view = dh;
    dh_view.rank = 2;
    dh_view.shape[0] = T;
    dh_view.shape[1] = H;
    vt::MatmulBT(d.q, dlog.t(), dh_view, w_router);
  }

  // Top-k: softmax + top-k + renorm, all on device.
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});

  if (kq) {
    // Keep-quant grouped path — 2 dispatches, 0 drains (decode):
    //   1) fused gate+up+SwiGLU (vt_moe_gate_up_swiglu_grouped_tq2) — reads bf16
    //      dh from device, quantizes Q8_K on-GPU, outputs f32 [P, I].
    //   2) down grouped GEMM (vt_matmul_bt_tq2_grouped_dev) — reads f32 SwiGLU
    //      output from device, quantizes Q8_K on-GPU, outputs bf16 [P, H].
    // The expert_ids for the grouped kernels are dtid reshaped to [P] — the
    // router lays them out [T, top_k] which is exactly p = t*top_k + j.
    //
    // DECODE (T==1): the activation is [1, H] and the fused shader's bcast arm
    // reads row 0 for all P output rows — fully on-device, zero host round-trips.
    //
    // PREFILL (T>1): each output row p = t*top_k + j needs h[t]. The fused
    // shader's gather_k arm reads h[p / top_k] directly from the [T, H]
    // activation — no host gather, no download/upload, zero host round-trips.
    Tensor act_gu = dh;  // [T, H] on device (T==1 => [1, H], bcast arm)
    act_gu.rank = 2;
    act_gu.shape[0] = T;
    act_gu.shape[1] = H;

    Tensor eid_view = dtid.t();
    eid_view.rank = 1;
    eid_view.shape[0] = P;
    eid_view.stride[0] = 1;  // was [top_k, 1] as rank-2; rank-1 needs stride 1

    // 1) Fused gate+up+SwiGLU → f32 [P, I] on device.
    Tensor w_gate = ResidentWeight(d, w.expert_gate_kq);   // [E*I, H] TQ2_0
    Tensor w_up = ResidentWeight(d, w.expert_up_kq);       // [E*I, H] TQ2_0
    DBuf dswiglu(d, DType::kF32, {P, I});
    vt::MoeGateUpSwiGLUGrouped(d.q, dswiglu.t(), act_gu, w_gate, w_up,
                               eid_view, kMapleSwigluClamp);

    // 2) Down grouped GEMM → bf16 [P, H] = [T, top_k, H] on device.
    Tensor w_down = ResidentWeight(d, w.expert_down_kq);   // [E*H, I] TQ2_0
    DBuf dexpert(d, DType::kBF16, {P, H});
    vt::MatmulBTQuantGrouped(d.q, dexpert.t(), dswiglu.t(), w_down, eid_view);

    // 3) Combine: weighted sum over top_k, NO shared expert. The expert output
    //    [P, H] = [T, top_k, H] is already in the layout MoeCombine expects.
    DBuf dout(d, DType::kBF16, {T, H});
    Tensor eo_view = dexpert.t();
    eo_view.rank = 3;
    eo_view.shape[0] = T;
    eo_view.shape[1] = top_k;
    eo_view.shape[2] = H;
    eo_view.stride[0] = top_k * H;  // [P,H] strides are [H,1]; rank-3 needs 3
    eo_view.stride[1] = H;
    eo_view.stride[2] = 1;
    vt::MoeCombine(d.q, dout.t(), eo_view, dtw.t(), nullptr);
    return dout;
  }

  // bf16 per-expert fallback (no keep-quant tower): download h + ids, run the
  // synchronous per-expert loop. This path is NOT the maple hot path (the GGUF
  // ships TQ2_0 experts), but it stays as the reference for bf16-expand loads.
  std::vector<uint16_t> h(static_cast<size_t>(T) * H);
  d.b.Copy(d.q, h.data(), dh.data, h.size() * sizeof(uint16_t));
  d.b.Synchronize(d.q);
  std::vector<float> weights(static_cast<size_t>(T) * top_k);
  std::vector<int32_t> ids(static_cast<size_t>(T) * top_k);
  dtw.Download(d, weights.data());
  dtid.Download(d, ids.data());

  std::vector<uint16_t> expert_out(static_cast<size_t>(T) * top_k * H, 0);
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(
      static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < top_k; ++j)
      lists[static_cast<size_t>(
                ids[static_cast<size_t>(t) * top_k + j])]
          .push_back({t, j});
  for (int64_t e = 0; e < E; ++e) {
    const auto& list = lists[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    std::vector<uint16_t> xg(static_cast<size_t>(n) * H);
    for (int64_t r = 0; r < n; ++r)
      std::memcpy(xg.data() + static_cast<size_t>(r) * H,
                  h.data() + static_cast<size_t>(list[r].first) * H,
                  static_cast<size_t>(H) * sizeof(uint16_t));
    std::vector<uint16_t> y =
        MapleExpertMlpBf16Clamped(d, w.expert_gate[static_cast<size_t>(e)],
                                  w.expert_up[static_cast<size_t>(e)],
                                  w.expert_down[static_cast<size_t>(e)],
                                  xg, n, H, I);
    for (int64_t r = 0; r < n; ++r) {
      const int64_t t = list[r].first, j = list[r].second;
      std::memcpy(expert_out.data() +
                      static_cast<size_t>(t * top_k + j) * H,
                  y.data() + static_cast<size_t>(r) * H,
                  static_cast<size_t>(H) * sizeof(uint16_t));
    }
  }

  DBuf deo(d, DType::kBF16, {T, top_k, H}, expert_out.data());
  DBuf dwt(d, DType::kF32, {T, top_k}, weights.data());
  DBuf dout(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, dout.t(), deo.t(), dwt.t(), nullptr);
  return dout;
}


// Full eager forward: embed -> 24 layers -> final norm -> untied lm_head.
std::vector<float> MapleForward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MapleWeights& weights,
    const HfConfig& config, Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "maple: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "maple: one PagedKvCache per layer required");

  Dev d{vt::GetBackend(queue.device.type), queue};

  Tensor dtab =
      dense_attn::ResidentWeight(d, weights.embed_tokens, {vocab, H});
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, DType::kBF16, {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  Tensor stream = hidden.t();  // the current residual-stream delta view

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);
  StepInputs si = dense_attn::BuildStepInputs(d, positions, attn_meta, config);

  // The MoE output becomes the new residual-stream delta each layer; its
  // owning buffer must outlive the NEXT layer's fused add+RMSNorm read, so it
  // is held in `delta` (the ownership dance qwen3_moe.cpp expresses as
  // hidden/hidden_hold — same contract, one variable).
  std::optional<DBuf> delta;

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const MapleLayerWeights& layer = weights.layers[static_cast<size_t>(l)];

    Tensor w_in = dense_attn::ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {T, H});
    if (dense_attn::FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dhn.t(), stream, w_in, &res.t(),
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dhn.t(), stream, w_in, vt::RmsNormArgs{eps, false},
                  &res.t());
    }

    DBuf attn = dense_attn::AttnBlockMaple(d, layer.attn, layer.is_swa, config,
                                           dhn.t(), si, attn_meta,
                                           attn_kv[static_cast<size_t>(l)], T);

    Tensor w_post =
        dense_attn::ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {T, H});
    if (dense_attn::FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(),
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false},
                  &res.t());
    }
    delta = MapleMoeBlock(d, layer.moe, config, dh2.t(), T);
    // Re-point the stream at the new delta's tensor view. The previous
    // iteration's delta DBuf is released when `delta` is re-assigned; the view
    // stays valid because `delta` holds the block until the next assignment.
    stream = delta->t();
  }

  // Final RMSNorm + untied lm_head.
  Tensor w_fn = dense_attn::ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (dense_attn::FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm.t(), stream, w_fn, &res.t(),
                   vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm.t(), stream, w_fn, vt::RmsNormArgs{eps, false},
                &res.t());
  }
  Tensor lm = dense_attn::ResidentWeight(d, weights.lm_head);
  DBuf logits(d, DType::kF32, {T, vocab});
  if (weights.lm_head.nk)
    vt::MatmulBT(d.q, logits.t(), dnorm.t(), lm);  // [vocab,H] bytes (GGUF head)
  else
    vt::Matmul(d.q, logits.t(), dnorm.t(), lm);    // [H,vocab] bytes (safetensors)
  std::vector<float> out(static_cast<size_t>(T) * vocab);
  logits.Download(d, out.data());

  if (!logits_indices.empty() &&
      static_cast<int64_t>(logits_indices.size()) < T) {
    // Gather selected rows (logits_indices contract from ModelForwardInput).
    std::vector<float> gathered;
    gathered.reserve(logits_indices.size() * vocab);
    for (int32_t idx : logits_indices) {
      const auto* row = out.data() + static_cast<size_t>(idx) * vocab;
      gathered.insert(gathered.end(), row, row + vocab);
    }
    return gathered;
  }
  return out;
}

}  // namespace vllm
