// vllm.cpp original (in-memory weight container for the Qwen3.6-35B-A3B MoE
// gate model); no 1:1 upstream mirror — the pinned load path is
// AutoWeightsLoader over nn.Module params (models/qwen3_5.py @ e24d1b24), which
// this replaces with an explicit owned-tensor struct for the M0.9 forward.
//
// Loads the real 35B checkpoint (nvidia/Qwen3.6-35B-A3B-NVFP4) into owned host
// bf16 tensors. Quant schemes per weight class (verified against the real ckpt,
// .agents/specs/qwen36-forward-notes.md §6):
//   - MoE experts + shared_expert + lm_head : NVFP4 W4A16 (DequantNvfp4ToBf16)
//   - attention (q/k/v/o) + GDN (in_proj_qkv/z, out_proj) : per-tensor FP8
//     (DequantFp8ToBf16) — NOT bf16 as the task first assumed
//   - everything else (embeds, norms, router gate, conv1d, A_log/dt_bias,
//     in_proj_a/b) : bf16 (A_log/dt_bias upcast to f32)
//
// All 2-D projection weights are stored TRANSPOSED to vt::Matmul's B layout
// [in, out] (on-disk torch layout is [out, in]). Host checkpoint ownership stays
// per logical projection. On CUDA the production 27B path additionally builds
// resident packed gate_up and full-attention QKV operands, mirroring vLLM's
// MergedColumnParallelLinear/QKVParallelLinear topology at TP=1; diagnostic
// toggles retain the split residents.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {

// One owned, contiguous, host tensor: heap bytes + shape/dtype. View() builds a
// fresh vt::Tensor over the current buffer, so it stays valid across moves/
// reallocations of the owning struct (no cached raw pointer to dangle).
struct OwnedTensor {
  std::vector<uint8_t> bytes;
  vt::DType dtype = vt::DType::kF32;
  int rank = 0;
  int64_t shape[vt::kMaxRank] = {0, 0, 0, 0};

  // Matmul-weight orientation. false (default): Matmul-B [K=in, N=out] (the
  // loader transposed the disk tensor; row-major x row-major vt::Matmul).
  // true: raw torch Linear [N=out, K=in] as on disk (LoadBf16Raw) — consumed
  // via vt::MatmulBT, the cuBLASLt TN fast path (see ops.h MatmulBT).
  bool nk = false;

  bool Empty() const { return bytes.empty(); }
  int64_t Numel() const;
  // Contiguous view over the current buffer (host/CPU device).
  vt::Tensor View() const;

  // Lazily-populated device-resident copies (CUDA forward only; null on host or
  // before first use). Uploaded ONCE and reused across every forward step so the
  // model's bf16/f32 weights (embed table, norms, attention/GDN projections,
  // router) stop re-uploading per op. d_dev holds the raw-dtype bytes; d_dev_f32
  // holds a bf16->f32 upcast (the CUDA norm/conv kernels want f32 when the
  // activation is f32). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_dev;
  mutable std::shared_ptr<void> d_dev_f32;
};

// Device-resident NVFP4 W4A16 weight (M2.2b). The modelopt packed fp4 codes +
// fp8-e4m3 group scales + per-tensor scale, kept RAW in the ORIGINAL torch
// [N=out_features, K=in_features] orientation vt::MatmulNvfp4 expects (NOT
// transposed to Matmul-B [in,out], and NOT dequanted to bf16). Keeping the
// ~22GB fp4 as fp4 avoids the M2.2-profile CPU dequant (~40 min) + the ~70GB
// bf16 host tensors. On the CUDA path the forward uploads packed+scale to the
// GPU ONCE (lazily, on first use — the mutable device handles below) and reads
// them in place across every step; on the host path it dequants for reference.
struct Nvfp4Weight {
  OwnedTensor packed;   // i8 [N, K/2]   two 4-bit E2M1 codes per byte
  OwnedTensor scale;    // i8 [N, K/16]  one fp8-e4m3 scale per 16-elem group
  float scale2 = 0.0F;  // per-tensor weight global scale (1/divisor), multiplied
  int64_t n = 0;        // out_features
  int64_t k = 0;        // in_features (K % 16 == 0)
  bool Empty() const { return packed.Empty(); }

  // TRUE W4A4 fields (27B compressed-tensors NVFP4; notes §7). Populated ONLY on
  // the 27B CT load (LoadCtNvfp4Raw); left 0 for the 35B modelopt W4A16 weights
  // (which have no activation quant) so `IsTrueW4A4()` gates the 27B alone.
  //   input_global_scale_inv = the ON-DISK activation divisor (2688/amax_act),
  //     passed DIRECTLY to vt::ScaledFp4Quant.
  //   alpha = (1/input_divisor)·(1/weight_divisor) = scale2/input_global_scale_inv,
  //     the single scalar vt::MatmulNvfp4Fp4 multiplies the fp4xfp4 accumulator by.
  // Keep the original on-disk weight divisor as well as its reciprocal. Fused
  // logical linears (qkv/gate_up) take max(divisors) BEFORE reciprocating in
  // vLLM; reconstructing the divisor from scale2 can lose one float ULP.
  float weight_global_scale_inv = 0.0F;
  float input_global_scale_inv = 0.0F;
  float alpha = 0.0F;
  // True when the activation-quant globals were loaded (27B true-W4A4 path).
  bool IsTrueW4A4() const { return alpha > 0.0F; }

  // Lazily-populated device-resident copies (CUDA forward only; null on host or
  // before first use). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_packed;
  mutable std::shared_ptr<void> d_scale;
  // Lazily-populated SWIZZLED weight block scale for the cutlass sm120a fp4 GEMM
  // path (VT_NVFP4_CUTLASS): [round_up(n,128), round_up(k/16,4)] in the cutlass
  // atom layout, computed once from d_scale via vt::SwizzleBlockscale.
  mutable std::shared_ptr<void> d_scale_sw;
  // vLLM/FlashInfer-compatible model-owned f32 alpha for the true-W4A4 CUTLASS
  // path. Uploaded once from the persistent `alpha` member; the diagnostic host
  // scalar path leaves this null.
  mutable std::shared_ptr<void> d_alpha;
};

// Device-resident per-tensor FP8 (W8A8) weight — the 35B attn q/k/v/o + GDN
// in_proj_qkv/z/out_proj projections (checkpoint: weight F8_E4M3 + f32
// weight_scale + f32 input_scale, activations W8A8 static per-tensor). Raw IEEE
// fp8-e4m3fn bytes kept in the ORIGINAL torch [N=out_features, K=in_features]
// orientation the cutlass W8A8 GEMM reads directly (NOT transposed to Matmul-B
// [in,out], NOT dequanted to bf16 — halves the projection's device memory vs the
// bf16 field and defers all scaling into the GEMM). On the CUDA path the forward
// uploads the bytes ONCE (lazily; the mutable device handle below) and reads them
// in place across every step. Populated only on the real 35B CUDA load with the
// cutlass W8A8 path enabled (LoadFp8Raw); the bf16 field it replaces is left
// EMPTY. `alpha = input_scale * weight_scale` is precomputed at load (both scales
// are per-tensor scalars) — the single fused scalar vt::MatmulFp8Cutlass applies.
struct Fp8Weight {
  OwnedTensor packed;         // i8 [N, K]  one fp8-e4m3fn byte per element
  float weight_scale = 0.0F;  // per-tensor weight_scale (dequant(w) = f8(w)*this)
  float input_scale = 0.0F;   // per-tensor activation scale (quant a = a/this)
  float alpha = 0.0F;         // input_scale * weight_scale (folded GEMM scalar)
  int64_t n = 0;              // out_features
  int64_t k = 0;              // in_features
  bool Empty() const { return packed.Empty(); }

  // Lazily-populated device-resident copy (CUDA forward only; null on host or
  // before first use). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_packed;
};

// Gated-DeltaNet (linear_attention) layer weights. Projections in Matmul-B
// layout [in, out]; conv1d [conv_dim, K]; a_log/dt_bias f32 [Hv]; norm bf16.
struct GdnLayerWeights {
  OwnedTensor in_proj_qkv;    // bf16 [H, conv_dim]  (FP8 dequant + T)
  OwnedTensor in_proj_z;      // bf16 [H, value_dim] (FP8 dequant + T)
  OwnedTensor in_proj_b;      // bf16 [H, Hv]        (bf16 + T)
  OwnedTensor in_proj_a;      // bf16 [H, Hv]        (bf16 + T)
  OwnedTensor conv1d_weight;  // bf16 [conv_dim, K]  (bf16, NOT transposed)
  OwnedTensor a_log;          // f32  [Hv]
  OwnedTensor dt_bias;        // f32  [Hv]
  OwnedTensor norm_weight;    // bf16 [Dv]           (RMSNormGated)
  OwnedTensor out_proj;       // bf16 [value_dim, H] (FP8 dequant + T)

  // 27B W4A4 fp4-resident variant of out_proj (compressed-tensors NVFP4, notes
  // §3.6). When populated (real 27B CUDA load) the forward calls vt::MatmulNvfp4
  // on it and out_proj above is left EMPTY; the 35B / synthetic loaders populate
  // the bf16 out_proj and leave this empty. Exactly one is filled.
  Nvfp4Weight out_proj_fp4;   // [N=H, K=value_dim]

  // 35B fp8-resident W8A8 variants (per-tensor FP8). Populated BY DEFAULT on the
  // real 35B CUDA+cutlass load (VT_DENSE_NATIVE); the bf16 in_proj_qkv/z/out_proj
  // above are then left EMPTY and the forward calls the native fp8 GEMM (cuBLASLt
  // fp8 by default, or cutlass fp8 under VT_DENSE_CUBLASLT_FP8=0). VT_DENSE_NATIVE
  // =0 flips back to the bf16 fields. The 27B (bf16 in_proj + fp4 out_proj) and
  // GGUF/synthetic (bf16) loaders leave these empty. Forward checks fp8, fp4, bf16.
  Fp8Weight in_proj_qkv_fp8;  // [N=conv_dim, K=H]
  Fp8Weight in_proj_z_fp8;    // [N=value_dim, K=H]
  Fp8Weight out_proj_fp8;     // [N=H, K=value_dim]
};

// Full (dense causal) attention layer weights.
struct FullAttnLayerWeights {
  OwnedTensor q_proj;   // bf16 [H, 2*Hq*Dh]  (FP8 dequant + T; output-gate doubled)
  OwnedTensor k_proj;   // bf16 [H, Hkv*Dh]
  OwnedTensor v_proj;   // bf16 [H, Hkv*Dh]
  OwnedTensor o_proj;   // bf16 [Hq*Dh, H]    (FP8 dequant + T)
  OwnedTensor q_norm;   // bf16 [Dh]
  OwnedTensor k_norm;   // bf16 [Dh]

  // 27B W4A4 fp4-resident variants of q/k/v/o_proj (compressed-tensors NVFP4,
  // notes §3.6). Populated on the real 27B CUDA load; the 35B / synthetic
  // loaders populate the bf16 fields above and leave these empty (exactly one
  // set filled). Each kept in the on-disk [N=out, K=in] orientation MatmulNvfp4
  // reads directly.
  Nvfp4Weight q_proj_fp4;  // [N=2*Hq*Dh, K=H]
  Nvfp4Weight k_proj_fp4;  // [N=Hkv*Dh,  K=H]
  Nvfp4Weight v_proj_fp4;  // [N=Hkv*Dh,  K=H]
  Nvfp4Weight o_proj_fp4;  // [N=H,       K=Hq*Dh]

  // CUDA resident for vLLM's QKVParallelLinear. The checkpoint owns logical
  // Q/K/V shards separately; production concatenates their packed rows and
  // linear block scales once, then keeps the combined packed operand and
  // combined swizzled scale resident. The split weights remain available for
  // VT_FP4_MERGED_QKV=0 and non-CUTLASS diagnostics.
  mutable std::shared_ptr<void> d_qkv_packed;
  mutable std::shared_ptr<void> d_qkv_scale_sw;
  // Merged QKV owns the max-before-reciprocal alpha as one physical device
  // scalar, matching its one physical projection. The host member is persistent
  // storage for the asynchronous H2D copy.
  mutable float qkv_alpha = 0.0F;
  mutable std::shared_ptr<void> d_qkv_alpha;

  // 35B fp8-resident W8A8 variants (per-tensor FP8). Populated BY DEFAULT on the
  // real 35B CUDA+cutlass load (VT_DENSE_NATIVE); the bf16 q/k/v/o_proj above are
  // then left EMPTY and the forward calls the native fp8 GEMM (cuBLASLt fp8 by
  // default, or cutlass fp8 under VT_DENSE_CUBLASLT_FP8=0). VT_DENSE_NATIVE=0
  // flips back to the bf16 fields. The 27B (fp4) and GGUF/synthetic (bf16)
  // loaders leave these empty. Forward checks fp8, then fp4, then bf16.
  Fp8Weight q_proj_fp8;  // [N=2*Hq*Dh, K=H]
  Fp8Weight k_proj_fp8;  // [N=Hkv*Dh,  K=H]
  Fp8Weight v_proj_fp8;  // [N=Hkv*Dh,  K=H]
  Fp8Weight o_proj_fp8;  // [N=H,       K=Hq*Dh]
};

// Exact scalar processing for the three-shard CT NVFP4 QKVParallelLinear.
// Mirrors compressed_tensors_w4a4_nvfp4.py:95-138 and QKVParallelLinear's
// logical-shard loader: take each maximum divisor before reciprocating once.
struct FullAttnQkvGlobals {
  float input_global_scale_inv = 0.0F;  // max on-disk input divisor
  float weight_global_scale = 0.0F;     // reciprocal of max weight divisor
  float alpha = 0.0F;
};

FullAttnQkvGlobals MergeFullAttnQkvGlobals(const Nvfp4Weight& q,
                                           const Nvfp4Weight& k,
                                           const Nvfp4Weight& v);

// Sparse-MoE block (router + per-expert MLP + shared expert). Per-expert and
// shared projections are NVFP4-dequant'd and stored separately (gate/up/down),
// all in Matmul-B layout.
struct MoeBlockWeights {
  OwnedTensor router_gate;   // bf16 [H, E]  (bf16 + T)
  OwnedTensor shared_gate;   // bf16 [H, 1]  (bf16 + T)
  std::vector<OwnedTensor> expert_gate;  // E * bf16 [H, I]
  std::vector<OwnedTensor> expert_up;    // E * bf16 [H, I]
  std::vector<OwnedTensor> expert_down;  // E * bf16 [I, H]
  OwnedTensor shared_gate_proj;  // bf16 [H, Is]
  OwnedTensor shared_up_proj;    // bf16 [H, Is]
  OwnedTensor shared_down_proj;  // bf16 [Is, H]

  // M2.2b fp4-resident variants of the NVFP4 expert/shared projections. When
  // populated (real-checkpoint CUDA load) the forward calls vt::MatmulNvfp4 on
  // these and the bf16 fields above are left EMPTY; the synthetic / GGUF loaders
  // populate the bf16 fields and leave these empty. Exactly one set is filled.
  std::vector<Nvfp4Weight> expert_gate_fp4;  // E * [N=I, K=H]
  std::vector<Nvfp4Weight> expert_up_fp4;    // E * [N=I, K=H]
  std::vector<Nvfp4Weight> expert_down_fp4;  // E * [N=H, K=I]
  Nvfp4Weight shared_gate_proj_fp4;  // [N=Is, K=H]
  Nvfp4Weight shared_up_proj_fp4;    // [N=Is, K=H]
  Nvfp4Weight shared_down_proj_fp4;  // [N=H, K=Is]
};

// One decoder layer: input/post norms + one attention variant + the MoE block.
struct Qwen3_5MoeLayerWeights {
  bool is_linear_attention = false;
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  GdnLayerWeights gdn;                    // valid iff is_linear_attention
  FullAttnLayerWeights attn;             // valid iff !is_linear_attention
  MoeBlockWeights moe;                   // all 35B layers are MoE
};

// Whole-model weights.
struct Qwen3_5MoeWeights {
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (NOT transposed; embed lookup)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab]  (bf16/GGUF path; empty when fp4)
  Nvfp4Weight lm_head_fp4;   // [N=vocab, K=H]   (M2.2b fp4-resident; else empty)
  std::vector<Qwen3_5MoeLayerWeights> layers;
};

// Resolves a tensor name to its StTensor (across shards). Throws if absent.
using TensorResolver = std::function<const StTensor&(const std::string&)>;

// Load one decoder layer's weights from real tensors. `layer_type` is
// "linear_attention" or "full_attention"; `num_experts` drives the expert loop.
// Exercised on real data by the Task 3 unit test (both layer types live in
// shard 1). Prefix is "model.language_model.layers.{layer_idx}.".
Qwen3_5MoeLayerWeights LoadQwen3_5MoeLayer(const TensorResolver& get,
                                           const std::string& layer_type,
                                           int64_t layer_idx,
                                           int64_t num_experts);

// Full-model load: resolves every param across the given shards (name -> shard
// looked up from each file's own header), dequantizes/transposes, and returns
// owned host bf16 tensors. Uses config.num_hidden_layers, config.layer_types,
// config.num_experts. The mmap'd shards may be released after this returns.
Qwen3_5MoeWeights LoadQwen3_5Moe(const std::vector<SafetensorsFile>& shards,
                                 const HfConfig& config);

}  // namespace vllm
