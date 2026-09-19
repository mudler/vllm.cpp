// GLM-5.3-Flash W9c-0/W9c-3 — the k-pool op probe and the device-resident
// compose forward. The probe half is always compiled; the compose half grows
// the TU from the 16-line stub into the full `Glm5NextDeviceForward`.
//
// The implementation follows the `kimi_linear_device.cpp` single-queue pattern:
// one `Dev`, one device queue, and host-fallback islands for the ops that have
// no portable `vt::` device provider.
//
// ─── ON DEVICE (vt:: dispatch) ─────────────────────────────────────────────
//   embedding              vt::Embedding
//   RMSNorm (all sites)    vt::RmsNorm (float acc, vs the host's double acc)
//   KDA recurrence          existing device arm in glm5_next_kda.cpp (dev passed through)
//   MoE routed experts      existing device arm in glm5_next_moe.cpp (dev passed through)
//   MoE combine             vt::MoeCombine (via the queue passed to MoeForward)
//   lm_head                 vt::MatmulBT (float acc, vs the host's chunked double)
//
// ─── DEVICE ARMS (vt:: dispatch or family device kernel) ──────────────────
//   (1) mHC pre/post — ON DEVICE when kDeepseekV4Mhc is registered on the
//       queue's device (CUDA O34, #3199; ROCm O34). Routes through
//       MhcDevice()->pre()/post() with the same host-vector interface as V4.
//       HcHeadCollapseMean stays on host: GLM-5.3 uses an unweighted mean,
//       which has no device kernel (V4's weighted head diverges).
//   (2) DSA/MLA attention — Attention() is a monolithic host function. The
//       k-pool indexer ops (kGlm5NextKpoolCompress/Select) are CUDA-only, so on
//       a CPU queue the indexer stays on host too.
//   (3) Dense+shared MLP — NOW ON DEVICE via vt::ClampedSwiGLU + vt::MatmulBT.
//       The gate/up projections are stacked into one [2I,H] weight so a single
//       MatmulBT produces the fused [T, 2I] gate|up buffer that ClampedSwiGLU
//       consumes. Previously a host island (issue #3201).
//
// On a CPU queue the vt:: kernels use float32 accumulation where the host
// reference uses double, so the output agrees within a float-vs-double envelope
// rather than byte-exact. On a GPU the device kernels match upstream PyTorch's
// float32 numerics.
#include "vllm/model_executor/models/glm5_next_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // ResidentWeight
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/deepseek_v2.h"  // MlaStep / BuildMlaStep
#include "vllm/model_executor/models/deepseek_v4_device.h"  // MhcDeviceKernels / MhcDevice
#include "vllm/model_executor/models/mla_attention.h"
#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_forward.h"
#include "vllm/model_executor/models/glm5_next_layer.h"
#include "vllm/model_executor/models/glm5_next_mhc.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_kda.h"
#include "vllm/model_executor/models/glm5_next_loader.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm::glm5_next {

bool KpoolDeviceOpsAvailable() {
  return vt::OpRegistered(vt::OpId::kGlm5NextKpoolCompress, vt::DeviceType::kCUDA) &&
         vt::OpRegistered(vt::OpId::kGlm5NextKpoolSelect, vt::DeviceType::kCUDA);
}

namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using dense_attn::ResidentWeight;
using v1::CommonAttentionMetadata;
using v1::TritonMLAImpl;
using vt::DType;
using vt::Tensor;

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("glm5_next device forward: " + what);
}

// Device-resident f32 weight view. On CPU this aliases the host bytes (host-
// pointer aliasing is a CPU property); on CUDA it would need a staging upload
// (the production path uses ResidentWeight for that — not wired here yet).
inline Tensor WF32(const Dev& d, const std::vector<float>& v,
                   const std::vector<int64_t>& shape) {
  return MakeTensor(const_cast<float*>(v.data()), DType::kF32, d.q.device, shape);
}

// Standalone RMSNorm on device: upload [T,H], vt::RmsNorm (float acc), download.
// The host reference uses double accumulation; the device kernel uses float.
// This is the primary source of numeric divergence on a CPU queue.
void DeviceRmsNorm(const Dev& d, float* out, const float* in,
                   const std::vector<float>& weight,
                   int64_t T, int64_t H, float eps) {
  DBuf din(d, DType::kF32, {T, H}, in);
  DBuf dout(d, DType::kF32, {T, H});
  Tensor w = WF32(d, weight, {H});
  vt::RmsNorm(d.q, dout.t(), din.t(), w, vt::RmsNormArgs{eps, /*gemma=*/false});
  dout.Download(d, out);
}

// Dense MLP on device: stack gate_proj [I,H] + up_proj [I,H] into [2I,H],
// one MatmulBT produces the fused gate|up [T, 2I], ClampedSwiGLU applies the
// clamped-silu activation, then MatmulBT with down_proj [H,I] produces [T, H].
// The stacking is required because vt::MatmulBT demands a contiguous output
// tensor, so two separate gate/up calls cannot share a [T, 2I] buffer.
std::vector<float> DeviceDenseMlpForward(
    const Dev& d, const DenseMlpWeights& w,
    const std::vector<float>& normed,
    int64_t H, int64_t I, int64_t T, float limit) {
  std::vector<float> stacked(static_cast<size_t>(2 * I * H));
  std::copy(w.gate_proj.begin(), w.gate_proj.end(), stacked.begin());
  std::copy(w.up_proj.begin(), w.up_proj.end(),
            stacked.begin() + static_cast<size_t>(I * H));

  DBuf dh(d, DType::kF32, {T, H}, normed.data());
  Tensor sw = WF32(d, stacked, {2 * I, H});
  DBuf dgu(d, DType::kF32, {T, 2 * I});
  vt::MatmulBT(d.q, dgu.t(), dh.t(), sw);

  DBuf da(d, DType::kF32, {T, I});
  vt::ClampedSwiGLU(d.q, da.t(), dgu.t(), limit);

  Tensor dw = WF32(d, w.down_proj, {H, I});
  DBuf dout(d, DType::kF32, {T, H});
  vt::MatmulBT(d.q, dout.t(), da.t(), dw);

  std::vector<float> out(static_cast<size_t>(T * H));
  dout.Download(d, out.data());
  return out;
}

// ── MLA block dims for GLM-5.3 NoPE ──────────────────────────────────────────
// Mirrors `GlmMoeDsaMlaBlockDims` (glm_moe_dsa.cpp:497) for the geometry this
// model shares with its sibling, minus the indexer schedule: GLM-5.3's k-pool
// indexer is NOT the MLA seam's Lightning Indexer and stays on host, so every
// layer's dims carry no indexer geometry.
mla::MlaBlockDims Glm5NextMlaBlockDims(const Glm5NextParams& p) {
  mla::MlaBlockDims d{};
  d.hidden_size = p.hidden_size;
  d.num_heads = p.num_attention_heads;
  d.q_lora_rank = p.mla.q_lora_rank;
  d.kv_lora_rank = p.mla.kv_lora_rank;
  d.qk_nope_head_dim = p.mla.qk_nope_head_dim;
  d.qk_rope_head_dim = p.mla.qk_rope_head_dim;  // 0 — NoPE
  d.v_head_dim = p.mla.v_head_dim;
  d.rms_norm_eps = static_cast<float>(p.rms_norm_eps);
  // NoPE: no rotation to style. Validate refuses a set flag when R == 0.
  d.is_neox_style = false;
  d.indexer_rope_is_neox_style = false;
  // No Lightning Indexer — GLM-5.3's k-pool is separate and stays on host.
  d.index_n_heads = 0;
  d.index_head_dim = 0;
  d.index_topk = 0;
  d.skip_topk = false;
  d.sliding_window = 0;
  mla::DeepseekYarnRopeParams rope{};
  rope.rotary_dim = 0;  // NoPE
  rope.yarn = false;
  d.scale = mla::MlaAttentionScale(d, rope);
  d.Validate();
  return d;
}

// `MlaBlockWeights` from one layer's `Glm5NextMlaWeights`. The SPLIT
// A-projection arm (q_a_proj + kv_a_proj_with_mqa separately), matching the
// sibling's `GlmResidentMla` (glm_moe_dsa_forward.cpp:158). The absorbed trio
// (kv_b_proj, w_uk_t, w_uv) were produced at load by `AbsorbMla`.
// `rope_cos_sin_cache` stays empty — NoPE has no rotary. Indexer fields stay
// empty — the k-pool indexer is NOT the seam's Lightning Indexer.
mla::MlaBlockWeights Glm5NextResidentMla(Dev d, const Glm5NextMlaWeights& w,
                                         const mla::MlaBlockDims& dm) {
  mla::MlaBlockWeights m;
  m.q_a_proj = ResidentWeight(d, w.q_a_proj);
  m.q_a_layernorm = ResidentWeight(d, w.q_a_layernorm, {dm.q_lora_rank});
  m.q_b_proj = ResidentWeight(d, w.q_b_proj);
  m.kv_a_proj_with_mqa = ResidentWeight(d, w.kv_a_proj_with_mqa);
  m.kv_a_layernorm = ResidentWeight(d, w.kv_a_layernorm, {dm.kv_lora_rank});
  m.kv_b_proj = ResidentWeight(d, w.kv_b_proj);
  m.w_uk_t = ResidentWeight(d, w.w_uk_t);
  m.w_uv = ResidentWeight(d, w.w_uv);
  m.o_proj = ResidentWeight(d, w.o_proj);
  return m;
  }


  }  // namespace

std::vector<float> Glm5NextDeviceForward(
    const Glm5NextWeights& weights, const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& logits_indices, vt::Queue& queue,
    std::vector<LayerCache>* caches, int64_t lm_head_chunk_bytes) {
  const Glm5NextParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t hc = p.mhc.mult;
  const int64_t L = p.num_hidden_layers;
  const float eps = static_cast<float>(p.rms_norm_eps);
  if (lm_head_chunk_bytes <= 0)
    Fail("lm_head_chunk_bytes must be > 0, got " +
         std::to_string(lm_head_chunk_bytes));

  if (T <= 0) Fail("the step carries no tokens");
  if (H <= 0 || V <= 0) Fail("hidden_size or vocab_size is 0");
  if (hc <= 0) Fail("hc_mult must be > 0");

  // Build the device from the caller's queue. On a CPU queue this aliases host
  // memory; on CUDA it uploads/downloads. Reached via VT_GLM5_NEXT_DEVICE=1.
  Dev d{vt::GetBackend(queue.device), queue};

  // A CPU host queue for the host-fallback islands (HcHeadCollapseMean).
  // When the caller's queue is already CPU, reuse it.
  vt::Queue host_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Queue& hq = queue.device.type == vt::DeviceType::kCPU ? queue : host_queue;

  // mHC device kernels: kDeepseekV4Mhc is registered on CUDA (O34, #3199) and
  // ROCm. When available on the queue's device, route pre/post through the
  // device kernels (same host-vector interface as V4's be.device path).
  // HcHeadCollapseMean stays on host: unweighted mean, no device kernel.
  const bool mhc_on_device =
      queue.device.type != vt::DeviceType::kCPU &&
      (vt::OpRegistered(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kCUDA) ||
       vt::OpRegistered(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kROCM));
  const deepseek_v4::MhcDeviceKernels* mhc_kernels =
      mhc_on_device ? deepseek_v4::MhcDevice() : nullptr;

  // ── embedding (ON DEVICE) ──────────────────────────────────────────────────
  // Decode the full table to f32, then vt::Embedding (a row gather). On CPU
  // this is byte-identical to the host forward's per-row decode. The full-table
  // decode works for the test geometry; a production path would chunk it
  // (DecodeOwnedTensorToF32 refuses > 1 GiB by name).
  const std::vector<float> embed_table =
      DecodeOwnedTensorToF32(weights.embed_tokens, "token_embd.weight");
  std::vector<float> embeds(static_cast<size_t>(T * H));
  {
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    Tensor ttab = WF32(d, embed_table, {V, H});
    DBuf demb(d, DType::kF32, {T, H});
    vt::Embedding(d.q, demb.t(), ttab, dids.t());
    demb.Download(d, embeds.data());
  }

  // ── expand to hidden streams (host) ────────────────────────────────────────
  // The [T, H] embedding is broadcast to [T, hc, H] — the manifold every decoder
  // layer operates on. The stream stays on host: mHC pre/post upload/download
  // per token, and HcHeadCollapseMean is host-only.
  std::vector<float> streams = ExpandToHiddenStreams(embeds, 1, T, hc, H);

  // ── the mask (all ones for a single sequence) ───────────────────────────────
  const std::vector<uint8_t> mask(static_cast<size_t>(T), 1);

  // ── the final norm weight ──────────────────────────────────────────────────
  const std::vector<float> norm_w =
      DecodeOwnedTensorToF32(weights.norm, "output_norm.weight");

  // ── the layer source (streaming, one layer at a time) ───────────────────────
  Glm5NextGgufLayerSource layers(weights);

  // ── the layer loop ──────────────────────────────────────────────────────────
  // Mirrors DecoderLayerForward but swaps the host double-acc RmsNorm for the
  // device float-acc vt::RmsNorm. mHC pre/post route through device kernels
  // when available; HcHeadCollapseMean stays on host. DSA/MLA attention is on
  // device via the shared mla::ForwardMlaAttentionBlock.

  // ── MLA step (shared by all DSA/MLA layers) ──────────────────────────────
  // The per-step metadata (positions, slot_mapping, block table) is built ONCE
  // and reused by every MLA layer, matching the sibling's pattern
  // (glm_moe_dsa_forward.cpp:546-547). Each MLA layer gets its own local KV
  // cache, matching the sibling's per-layer `attn_kv[l]`
  // (glm_moe_dsa_forward.cpp:582-587). The persistent `caches` parameter is not
  // wired here; the test path passes caches == nullptr (single-shot prefill).
  const mla::MlaBlockDims mla_dims = Glm5NextMlaBlockDims(p);
  const int64_t mla_head_size = mla_dims.head_size();
  std::vector<int32_t> mla_positions(static_cast<size_t>(T));
  std::iota(mla_positions.begin(), mla_positions.end(), int32_t{0});
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {static_cast<int32_t>(T)};
  am.seq_lens_cpu = am.seq_lens;
  am.num_computed_tokens_cpu = {0};
  am.max_query_len = static_cast<int>(T);
  am.max_seq_len = static_cast<int>(T);
  const int64_t mla_block_size = T;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  am.slot_mapping.resize(static_cast<size_t>(T));
  std::iota(am.slot_mapping.begin(), am.slot_mapping.end(), int64_t{0});
  am.causal = true;
  const MlaStep mla_step =
      BuildMlaStep(d, mla_positions, am, mla_block_size, p.max_position_embeddings);
  TritonMLAImpl mla_impl;

  for (int64_t i = 0; i < L; ++i) {
    const DecoderLayerWeights& w = layers.Layer(i);

    // ── mHC pre (attn site) — DEVICE (when kDeepseekV4Mhc registered) ────
    // One MhcPre per token: collapses [hc, H] → [H] and retains the MhcPreResult
    // the matching MhcPost needs. On a device queue with kDeepseekV4Mhc
    // registered, routes through MhcDevice()->pre() (same interface as V4).
    std::vector<float> collapsed(static_cast<size_t>(T * H));
    std::vector<deepseek_v4::MhcPreResult> pre(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> slab(streams.begin() + t * hc * H,
                               streams.begin() + (t + 1) * hc * H);
      if (mhc_kernels) {
        pre[static_cast<size_t>(t)] = mhc_kernels->pre(
            queue, slab, w.attn_hc.fn, w.attn_hc.scale, w.attn_hc.base,
            hc, H, eps, static_cast<float>(p.mhc.eps),
            static_cast<float>(p.mhc.eps), kHcPostAlpha,
            p.mhc.sinkhorn_iters, /*norm_weight=*/{}, /*norm_eps=*/0.0f);
      } else {
        pre[static_cast<size_t>(t)] =
            MhcPre(slab, w.attn_hc, p.mhc, H, eps);
      }
      std::copy_n(pre[static_cast<size_t>(t)].layer_input.data(),
                   static_cast<size_t>(H), collapsed.data() + t * H);
    }

    // ── RMSNorm (input_layernorm) — DEVICE ──────────────────────────────
    std::vector<float> normed(static_cast<size_t>(T * H));
    DeviceRmsNorm(d, normed.data(), collapsed.data(), w.input_layernorm,
                  T, H, eps);

    // ── attention — HOST ISLAND (KDA recurrence on device via dev) ──────
    std::vector<float> attn_out;
    if (w.attn_kind == Glm5NextLayerKind::kLinearAttention) {
      // Zero padded rows before the recurrence (no-op for a single sequence).
      for (int64_t t = 0; t < T; ++t) {
        if (mask[static_cast<size_t>(t)] != 0) continue;
        std::fill_n(normed.data() + t * H, static_cast<size_t>(H), 0.0F);
      }
      const glm5_next_kda::Glm5NextKdaDims kd = KdaDimsFrom(p);
      LayerCache* lc =
          caches != nullptr ? &(*caches)[static_cast<size_t>(i)] : nullptr;
      if (lc != nullptr && lc->kda.empty()) lc->kda.resize(1);
      attn_out.assign(static_cast<size_t>(T * H), 0.0F);
      const std::vector<float> out = glm5_next_kda::Glm5NextKdaLayerForward(
          w.kda, normed, kd, T,
          lc != nullptr ? &lc->kda[0] : nullptr, hq, &d);
      if (static_cast<int64_t>(out.size()) != T * H)
        Fail("KDA layer output size mismatch");
      std::copy_n(out.data(), static_cast<size_t>(T * H), attn_out.data());
    } else {
      // ── DSA/MLA attention — DEVICE via shared mla::ForwardMlaAttentionBlock ──
      // Replaces the host Attention() island. The k-pool indexer is NOT part of
      // this seam; for the test geometry (index_topk > T) the k-pool mask is a
      // no-op, so dense attention matches the host's sparse attention.
      //
      // The CUDA FA2 prefill kernel requires bf16 query/key/value
      // (cuda_mla_prefill.cu:185), and ConcatAndCacheMla is a raw byte copy
      // (cpu_cache.cpp:88-89) so the kv_cache must match the compute dtype. The
      // sibling (glm_moe_dsa_forward.cpp:444) narrows hidden to bf16 before MLA
      // for the same reason. We cast f32 normed → bf16, run MLA in bf16, then
      // widen the output back to f32 for the host-side mHC post island.
      const mla::MlaBlockWeights mw =
          Glm5NextResidentMla(d, weights.layers[static_cast<size_t>(i)].mla,
                              mla_dims);
      DBuf dhidden_f32(d, DType::kF32, {T, H}, normed.data());
      DBuf dhidden(d, DType::kBF16, {T, H});
      vt::CastBf16(d.q, dhidden.t(), dhidden_f32.t());
      DBuf dattn_bf16(d, DType::kBF16, {T, H});
      Tensor attn_t = dattn_bf16.t();
      DBuf mla_kv_cache(d, DType::kBF16, {1, mla_block_size, mla_head_size});
      Tensor kv_cache_t = mla_kv_cache.t();
      mla::ForwardMlaAttentionBlock(d, mla_dims, mw, dhidden.t(),
                                     mla_step.positions, kv_cache_t,
                                     mla_step.slot_mapping, mla_step.meta,
                                     mla_impl, attn_t,
                                     /*attn_pre_o_proj=*/nullptr,
                                     /*shared=*/nullptr);
      // ForwardMlaAttentionBlock creates internal DBuf temporaries that are
      // destroyed when it returns, returning memory to the pool. Without a
      // sync, CUDA kernels may still be running when that memory is reused,
      // causing an illegal memory access. Sync the queue so all MLA kernels
      // complete before any subsequent allocation can reuse the memory.
      d.b.Synchronize(d.q);
      DBuf dattn_f32(d, DType::kF32, {T, H});
      vt::CastF32(d.q, dattn_f32.t(), dattn_bf16.t());
      attn_out.assign(static_cast<size_t>(T * H), 0.0F);
      dattn_f32.Download(d, attn_out.data());
    }
    if (static_cast<int64_t>(attn_out.size()) != T * H)
      Fail("attention output size mismatch");

    // ── mHC post (attn site) — DEVICE (when kDeepseekV4Mhc registered) ───
    // Folds the sublayer's [H] output back onto the [hc, H] stream.
    for (int64_t t = 0; t < T; ++t) {
      const std::vector<float> out(attn_out.begin() + t * H,
                                    attn_out.begin() + (t + 1) * H);
      const std::vector<float> resid(streams.begin() + t * hc * H,
                                      streams.begin() + (t + 1) * hc * H);
      std::vector<float> mixed;
      if (mhc_kernels) {
        mixed = mhc_kernels->post(queue, out, resid,
                                   pre[static_cast<size_t>(t)].post_mix,
                                   pre[static_cast<size_t>(t)].comb_mix, hc, H);
      } else {
        mixed = MhcPost(out, resid, pre[static_cast<size_t>(t)], hc, H);
      }
      std::copy_n(mixed.data(), static_cast<size_t>(hc * H),
                   streams.data() + t * hc * H);
    }

    // ── mHC pre (ffn site) — DEVICE (when kDeepseekV4Mhc registered) ─────
    // The residual is the stream the attention fold just produced.
    const std::vector<float> ffn_residual = streams;
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> slab(ffn_residual.begin() + t * hc * H,
                               ffn_residual.begin() + (t + 1) * hc * H);
      if (mhc_kernels) {
        pre[static_cast<size_t>(t)] = mhc_kernels->pre(
            queue, slab, w.ffn_hc.fn, w.ffn_hc.scale, w.ffn_hc.base,
            hc, H, eps, static_cast<float>(p.mhc.eps),
            static_cast<float>(p.mhc.eps), kHcPostAlpha,
            p.mhc.sinkhorn_iters, /*norm_weight=*/{}, /*norm_eps=*/0.0f);
      } else {
        pre[static_cast<size_t>(t)] =
            MhcPre(slab, w.ffn_hc, p.mhc, H, eps);
      }
      std::copy_n(pre[static_cast<size_t>(t)].layer_input.data(),
                   static_cast<size_t>(H), collapsed.data() + t * H);
    }

    // ── RMSNorm (post_attention_layernorm) — DEVICE ─────────────────────
    DeviceRmsNorm(d, normed.data(), collapsed.data(),
                  w.post_attention_layernorm, T, H, eps);

    // ── MLP — device arm (dense via ClampedSwiGLU) / device arm (MoE) ───
    std::vector<float> mlp_out;
    if (w.mlp_kind == Glm5NextMlpKind::kDense) {
      mlp_out = DeviceDenseMlpForward(d, w.dense_mlp, normed, H,
                                     p.intermediate_size, T,
                                     static_cast<float>(p.swiglu_limit));
    } else {
      mlp_out = MoeForward(MoeDimsFrom(p), w.moe, normed, T, hq, &d);
    }
    if (static_cast<int64_t>(mlp_out.size()) != T * H)
      Fail("MLP output size mismatch");

    // ── mHC post (ffn site) — DEVICE (when kDeepseekV4Mhc registered) ───
    for (int64_t t = 0; t < T; ++t) {
      const std::vector<float> out(mlp_out.begin() + t * H,
                                    mlp_out.begin() + (t + 1) * H);
      const std::vector<float> resid(ffn_residual.begin() + t * hc * H,
                                      ffn_residual.begin() + (t + 1) * hc * H);
      std::vector<float> mixed;
      if (mhc_kernels) {
        mixed = mhc_kernels->post(queue, out, resid,
                                   pre[static_cast<size_t>(t)].post_mix,
                                   pre[static_cast<size_t>(t)].comb_mix, hc, H);
      } else {
        mixed = MhcPost(out, resid, pre[static_cast<size_t>(t)], hc, H);
      }
      std::copy_n(mixed.data(), static_cast<size_t>(hc * H),
                   streams.data() + t * hc * H);
    }
  }
  // ── HcHeadCollapseMean — HOST ISLAND (unweighted mean, no device kernel) ──
  // GLM-5.3 uses an unweighted mean; V4's weighted head diverges. Stays on host.
  std::vector<float> hidden(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> slab(streams.begin() + t * hc * H,
                                   streams.begin() + (t + 1) * hc * H);
    const std::vector<float> col = HcHeadCollapseMean(slab, hc, H);
    std::copy_n(col.data(), static_cast<size_t>(H), hidden.data() + t * H);
  }

  // ── final RMSNorm — DEVICE ──────────────────────────────────────────────────
  DeviceRmsNorm(d, hidden.data(), hidden.data(), norm_w, T, H, eps);

  // ── logits gather (before lm_head) ─────────────────────────────────────────
  std::vector<int64_t> want;
  if (logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    std::iota(want.begin(), want.end(), int64_t{0});
  } else {
    want.reserve(logits_indices.size());
    for (int32_t idx : logits_indices) {
      if (idx < 0 || static_cast<int64_t>(idx) >= T)
        Fail("logits index " + std::to_string(idx) + " is outside [0, " +
             std::to_string(T) + ")");
      want.push_back(idx);
    }
  }

  // ── lm_head — DEVICE ────────────────────────────────────────────────────────
  // vt::MatmulBT with float acc (vs the host's chunked double acc). The full
  // head is decoded to f32; for the test geometry this is tiny. A production
  // path would chunk it to respect the 1 GiB decode ceiling.
  const OwnedTensor& head =
      weights.tied_word_embeddings ? weights.embed_tokens : weights.lm_head;
  const char* head_name =
      weights.tied_word_embeddings ? "token_embd.weight (tied head)" : "output.weight";
  const std::vector<float> head_f32 = DecodeOwnedTensorToF32(head, head_name);

  const int64_t n_out = static_cast<int64_t>(want.size());
  std::vector<float> logits(static_cast<size_t>(n_out) * static_cast<size_t>(V),
                             0.0f);
  {
    // Gather the rows we need.
    std::vector<float> gathered(static_cast<size_t>(n_out * H));
    for (int64_t r = 0; r < n_out; ++r) {
      std::copy_n(hidden.data() + static_cast<size_t>(want[static_cast<size_t>(r)]) * H,
                   static_cast<size_t>(H),
                   gathered.data() + static_cast<size_t>(r) * H);
    }
    DBuf dhid(d, DType::kF32, {n_out, H}, gathered.data());
    Tensor lm = WF32(d, head_f32, {V, H});
    DBuf dlog(d, DType::kF32, {n_out, V});
    vt::MatmulBT(d.q, dlog.t(), dhid.t(), lm);
    dlog.Download(d, logits.data());
  }

  return logits;
}

}  // namespace vllm::glm5_next
