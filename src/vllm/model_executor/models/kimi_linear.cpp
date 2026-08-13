// Kimi-Linear DEVICE forward — the born-on-the-runner SEAM (W6). `ForwardDevice`
// is the DEFAULT `gather_logits` production/runner path; it no longer refuses. It
// composes the whole 27-layer KDA/NoPE-MLA + 256-expert-MoE hybrid via the landed
// CPU reference (`KimiLinearModel::Forward` -> HostForwardSeq, kimi_linear_forward
// .cpp) and hands the logits back **DEVICE-RESIDENT** (a pooled `DBuf`, wrapped like
// deepseek_v2.cpp:633 `WrapDeviceLogits`), so the runner's on-GPU sampler consumes
// them with NO host logit download on the default path — the third MUST-route seam
// (`ModelRegistry::Forward` -> `ForwardLogits.on_device()==true`). `on_device()` is
// true on BOTH CPU and CUDA (the pooled `DBuf` is a device buffer on either
// backend), so the CPU gate exercises the exact born-on-the-runner contract the GPU
// will (test_kimi_linear_forward "ForwardDevice ... device-resident logits").
//
// WHY THE COMPOSE STAYS ON THE CPU REFERENCE (correctness-first; box is down): the
// device-COMPUTE lane — a `DBuf`-resident bf16 forward that routes each layer
// through the reused device blocks — can only be GATED against the pinned vLLM
// oracle on GB10 (bf16 numerics, the GDN Triton-AOT decode cubins, the paged het-KV
// caches). With no GPU this brick lands the runner SEAM + the full reuse-wiring PLAN
// (below) but NOT the unverifiable device compute, so only the GPU verify + the
// W0/W7 e2e SACRED golden remain. Mirrors the DeepSeek-V4 cadence: deepseek_v4.cpp's
// `ForwardDevice` likewise composes off the host f32 tower + returns device-resident
// logits, with the DBuf-resident paged decode as its own named later residual.
//
// ─── W7 DEVICE-COMPUTE RESIDUAL — the reuse-wiring PLAN (GPU-verify-pending) ────
// Each op named below is CPU+CUDA-registered (so the plan is portable) EXCEPT the
// bf16 grouped-MoE GEMM (CUDA-only; the DeepSeek-V2 per-expert Matmul+MoeSiluMul is
// the CPU lane). Compose over a bf16 `DBuf` residual stream (dense_device_glue.h),
// mirroring `DeepseekV2Model::ForwardBody`/`ForwardLayers`:
//   0. embed -> `DBuf hidden(d, kBF16, {T,H})` (`vt::Embedding`).
//   1. input_layernorm: `vt::FusedChain(kFusedAddRmsNormStd, dhn, hidden, w_in,
//      &res.t(), eps)` — the shared glue seam (fused add+RMSNorm residual,
//      kimi_linear.py:361-365).
//   2. self_attn by is_kda_layer (kimi_linear.py:304-326):
//      - KDA (20 of 27): q/k/v/f_a/f_b/g_a/g_b proj via `vt::MatmulBT`; the 3 short
//        convs via `vt::CausalConv1dFwd`/`Update`; q/k `vt::L2Norm`; the KDA gate g
//        = -exp(A_log)*softplus(f_b(f_a(x))+dt_bias) has NO device exp/softplus op
//        (a genuinely-missing KDA piece) -> HOST-FALLBACK via `vllm::kimi_kda`
//        {KdaLowRankDecay,KdaDecayGate} then upload (a W7-speed residual); then the
//        reused GDN decode/prefill `vt::GdnDecode`/`GdnPrefill`/`GdnPackedDecode`
//        over the KDA MambaSpec state, `vt::RmsNormGated` (sigmoid) output norm,
//        o_proj `vt::MatmulBT` (mirror qwen3_5.cpp:3076/3845).
//      - NoPE-MLA (7 of 27): `mla::ForwardMlaAttentionBlock` (mla_attention.h:278)
//        with NoPE `MlaBlockDims` (rotary_emb=None, q_lora_rank=null, scaling
//        qk_head_dim**-0.5) over the MLA latent-576 paged KV group (mirror
//        deepseek_v2.cpp:496 `ResidentMla`/`BuildMlaStep`).
//   3. post_attention_layernorm: `vt::FusedChain(kFusedAddRmsNormStd, ...)`.
//   4. mlp by is_moe_layer (kimi_linear.py:328-347):
//      - MoE (layers 1..26): `vt::MoeRouterTopK` (MoeScoringFunc::kSigmoid, top-8,
//        num_expert_group=1/topk_group=1, routed_scaling 2.446, e_score_correction
//        _bias) + per-expert grouped GEMM (`vt::MoeGroupedGemmBf16GateUpSilu` +
//        `MoeGroupedGemmBf16`; CPU fallback = per-expert `Matmul`+`MoeSiluMul`) +
//        shared expert + `vt::MoeCombine` (mirror deepseek_v2.cpp:331 `MoeBlock`).
//      - dense (layer 0, first_k_dense_replace=1): gate/up `vt::MatmulBT` +
//        `vt::MoeSiluMul` + down `vt::MatmulBT` (the merged-GEMM seam).
//   5. final norm `vt::RmsNorm` -> untied lm_head `vt::MatmulBT` -> `DBuf` f32
//      logits (gathered per logits_indices) -> `WrapKimiLinearDeviceLogits`.
// Het-KV: MLA latent group + KDA GDN MambaSpec group (MakeKimiLinearKVCache,
// kimi_linear_registry.cpp) advanced in place per step (the paged incremental
// decode — the runner's persistent caches — is part of this W7 residual).
// Grounding: kimi_linear.py:426-458.
#include "vllm/model_executor/models/kimi_linear.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // dense_attn::{Dev,DBuf}
#include "vllm/model_executor/models/device_pool.h"        // Pool()
#include "vt/backend.h"                                     // vt::GetBackend
#include "vt/dtype.h"

namespace vllm {

namespace {

constexpr const char* kHostPending =
    "KimiLinear device forward: host-float weights not materialized — the born-on-"
    "the-runner ForwardDevice composes the [rows,vocab] logits off "
    "KimiLinearWeights::host (populated by LoadKimiLinearForCausalLMWeights); the "
    "DBuf-resident paged device COMPUTE (KDA via the GDN device family, NoPE-MLA via "
    "mla::ForwardMlaAttentionBlock, the DeepSeek-V2 grouped-MoE) is the GPU-verify-"
    "pending W7 residual. See .agents/specs/kimi-linear.md §5.";

// Wrap [rows,vocab] f32 logits as a DEVICE-RESIDENT ForwardLogits — verbatim the
// deepseek_v2.cpp:633 / qwen3_moe.cpp:216 seam: move the pooled block into a
// shared_ptr<void> whose deleter returns it to the shared DevicePool (NO per-step
// cudaMalloc/Free), and expose the [rows,vocab] f32 view. The runner samples
// argmax/temperature/top-k/top-p STRAIGHT off device_tensor with no full-logits D2H;
// only the sampled token ids cross to host. on_device()==true on CPU and CUDA alike.
ForwardLogits WrapKimiLinearDeviceLogits(dense_attn::DBuf&& dlogits, int64_t rows,
                                         int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

}  // namespace

ForwardLogits KimiLinearModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  // §13: the FULL model loads bf16-resident (host f32 would OOM the 119 GiB pool), so
  // the resident path drops the host.materialized precondition and ALWAYS routes to
  // the bf16 device COMPUTE (there are no host f32 weights to compose off).
  const bool resident = weights.resident.resident;
  VT_CHECK(resident || weights.host.materialized, kHostPending);

  // Resident (full-model) path, OR the W7 opt-in (VT_KIMI_DEVICE_COMPUTE=1) on the
  // tiny/host path: route through the REAL DBuf-resident device COMPUTE
  // (kimi_linear_device.cpp). Default-OFF on the host path keeps the CPU-verified
  // W6 host-ref compose below as production until the device compute is GPU-verified
  // against the SACRED oracle.
  if (resident || KimiDeviceComputeEnabled()) {
    return KimiLinearModel::ForwardDeviceCompute(token_ids, positions, attn_meta,
                                                 attn_kv, weights, queue,
                                                 logits_indices);
  }

  // Compose the [rows,vocab] f32 logits via the landed CPU reference (the whole
  // 27-layer hybrid; honors logits_indices gather-before-lm_head). The device-
  // COMPUTE routing of KDA/NoPE-MLA/MoE over the paged het-KV caches is the W7
  // residual documented above (GPU-gated). attn_meta / attn_kv are the runner's
  // paged caches the W7 device compute consumes; the reference manages its own
  // fresh single-sequence context, so they are unused on this seam.
  std::vector<float> flat = KimiLinearModel::Forward(
      token_ids, positions, attn_meta, attn_kv, weights, queue, logits_indices);

  const int64_t vocab = weights.params.vocab_size;
  const int64_t rows = vocab > 0 ? static_cast<int64_t>(flat.size()) / vocab : 0;

  // Hand the logits back DEVICE-RESIDENT so the on-GPU sampler consumes them with no
  // host download on the default gather_logits path (the born-on-the-runner seam).
  dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};
  dense_attn::DBuf dlogits(d, vt::DType::kF32, {rows, vocab}, flat.data());
  return WrapKimiLinearDeviceLogits(std::move(dlogits), rows, vocab);
}

}  // namespace vllm
