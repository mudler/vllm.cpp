// vllm.cpp original (in-memory weight container for the Qwen3.6-35B-A3B MoE
// gate model); no 1:1 upstream mirror — the pinned load path is
// AutoWeightsLoader over nn.Module params (models/qwen3_5.py @ e24d1b24), which
// this replaces with an explicit owned-tensor struct for the M0.9 forward.
//
// Loads the real 35B checkpoint (nvidia/Qwen3.6-35B-A3B-NVFP4) into owned host
// bf16 tensors. Quant schemes per weight class (verified against the real ckpt,
// .agents/qwen36-forward-notes.md §6):
//   - MoE experts + shared_expert + lm_head : NVFP4 W4A16 (DequantNvfp4ToBf16)
//   - attention (q/k/v/o) + GDN (in_proj_qkv/z, out_proj) : per-tensor FP8
//     (DequantFp8ToBf16) — NOT bf16 as the task first assumed
//   - everything else (embeds, norms, router gate, conv1d, A_log/dt_bias,
//     in_proj_a/b) : bf16 (A_log/dt_bias upcast to f32)
//
// All 2-D projection weights are stored TRANSPOSED to vt::Matmul's B layout
// [in, out] (on-disk torch layout is [out, in]). The four GDN input projections
// and q/k/v/gate/up projections are kept SEPARATE (no qkvz/ba/qkv/gate_up
// fusion) — see §6; fusion is a TP-sharding naming convenience, a no-op at TP=1.
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
  float scale2 = 0.0F;  // per-tensor global scale (amax/2688), multiplied
  int64_t n = 0;        // out_features
  int64_t k = 0;        // in_features (K % 16 == 0)
  bool Empty() const { return packed.Empty(); }

  // Lazily-populated device-resident copies (CUDA forward only; null on host or
  // before first use). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_packed;
  mutable std::shared_ptr<void> d_scale;
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
};

// Full (dense causal) attention layer weights.
struct FullAttnLayerWeights {
  OwnedTensor q_proj;   // bf16 [H, 2*Hq*Dh]  (FP8 dequant + T; output-gate doubled)
  OwnedTensor k_proj;   // bf16 [H, Hkv*Dh]
  OwnedTensor v_proj;   // bf16 [H, Hkv*Dh]
  OwnedTensor o_proj;   // bf16 [Hq*Dh, H]    (FP8 dequant + T)
  OwnedTensor q_norm;   // bf16 [Dh]
  OwnedTensor k_norm;   // bf16 [Dh]
};

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
