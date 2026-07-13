// vllm.cpp original — the DENSE Qwen3.6-27B text gate
// (unsloth/Qwen3.6-27B-NVFP4, arch Qwen3_5ForConditionalGeneration, text_config
// model_type "qwen3_5_text"). See .agents/specs/qwen27b-w4a4-notes.md.
//
// The 27B shares the 35B hybrid backbone WHOLESALE (GDN linear-attention + gated
// full-attention + Gemma (1+w) RMSNorm + mRoPE->NeoX); it REUSES the 35B's
// GdnLayerWeights / FullAttnLayerWeights sub-structs and the GdnBlock /
// FullAttnBlock forward helpers verbatim. The ONLY structural change is the
// per-layer sparse-MoE block being replaced by a DENSE SwiGLU MLP (gate/up/down,
// intermediate 17408): down( silu(gate(x)) * up(x) ). See notes §2.
//
// Quant: compressed-tensors NVFP4 W4A4 (notes §3). For the CPU correctness path
// each quantized Linear is MATERIALIZED to bf16 at load via the CT weight-dequant
// reference (DequantCtNvfp4WeightToF32, multiply by 1/weight_global_scale) and
// the existing bf16 forward carries it — the true GB10 fp4xfp4 GEMM is a later,
// GPU-gated step (notes §5 steps 5-7). The activation-quant round-trip is dropped
// on this correctness path (bf16 activations), matching the notes' §5 step-6a
// FAST PATH; that is a tiny numeric deviation vs true W4A4, validated later vs
// the pip-vLLM oracle golden.
//
// Which Linears are quantized (notes §3.6): QUANTIZED (W4A4) = every dense-MLP
// gate/up/down_proj, every self_attn q/k/v/o_proj, and the GDN linear_attn
// out_proj. NOT quantized (bf16 on disk) = the GDN in_proj_{qkv,z,a,b}, conv1d,
// A_log, dt_bias, all norms, embed_tokens, lm_head, mtp.*, and visual.*.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, GdnStateCache + v1 attention metadata
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, Gdn/FullAttn weights, TensorResolver
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

// Dense SwiGLU MLP (replaces the 35B MoE block). Projections in Matmul-B layout
// [in, out]; W4A4-materialized to bf16 at load. down( silu(gate(x)) * up(x) ).
struct DenseMlpWeights {
  OwnedTensor gate_proj;  // bf16 [H, I]
  OwnedTensor up_proj;    // bf16 [H, I]
  OwnedTensor down_proj;  // bf16 [I, H]

  // W4A4 fp4-resident variants (compressed-tensors NVFP4, notes §5 step-6a). On
  // the real 27B CUDA load these are populated (kept in the on-disk [N=out,K=in]
  // orientation vt::MatmulNvfp4 reads) and the bf16 fields above are left EMPTY;
  // the synthetic CPU tests populate the bf16 fields and leave these empty.
  Nvfp4Weight gate_proj_fp4;  // [N=I, K=H]
  Nvfp4Weight up_proj_fp4;    // [N=I, K=H]
  Nvfp4Weight down_proj_fp4;  // [N=H, K=I]

  // CUDA resident for vLLM's MergedColumnParallelLinear gate_up_proj. The
  // checkpoint stores gate/up separately; production concatenates their packed
  // rows and linear block scales once, then keeps only the combined packed
  // operand and combined swizzled scale resident. Mutable matches Nvfp4Weight's
  // lazy device-resident handles. The split weights remain host-resident for
  // VT_FP4_MERGED_GATE_UP=0 and VT_FP4_FULL_TACTICS=0 diagnostics.
  mutable std::shared_ptr<void> d_gate_up_packed;
  mutable std::shared_ptr<void> d_gate_up_scale_sw;
  // One physical merged projection owns one device alpha. Keep the exact merged
  // host scalar in model-lifetime storage because Backend::Copy is asynchronous.
  mutable float gate_up_alpha = 0.0F;
  mutable std::shared_ptr<void> d_gate_up_alpha;
};

// Exact scalar processing for a two-shard CT NVFP4 MergedColumnParallelLinear.
// Mirrors compressed_tensors_w4a4_nvfp4.py:95-138: max each logical-shard
// divisor first, reciprocate once, then multiply the two reciprocals into alpha.
struct DenseGateUpGlobals {
  float input_global_scale_inv = 0.0F;  // max on-disk input divisor
  float weight_global_scale = 0.0F;     // reciprocal of max on-disk weight divisor
  float alpha = 0.0F;
};

DenseGateUpGlobals MergeDenseGateUpGlobals(const Nvfp4Weight& gate,
                                           const Nvfp4Weight& up);

// One dense decoder layer: input/post norms + one attention variant + dense MLP.
struct Qwen3_5DenseLayerWeights {
  bool is_linear_attention = false;
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  GdnLayerWeights gdn;                    // valid iff is_linear_attention
  FullAttnLayerWeights attn;             // valid iff !is_linear_attention
  DenseMlpWeights mlp;                   // every layer has a dense MLP
};

// Whole dense-model text weights. lm_head is bf16 (unquantized in the 27B).
struct Qwen3_5DenseWeights {
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (NOT transposed; embed lookup)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab]  (unquantized -> Matmul-B layout)
  std::vector<Qwen3_5DenseLayerWeights> layers;
};

// True iff the projection named `name` is a W4A4-quantized Linear in the 27B
// (notes §3.6). `name` is the module path WITHOUT the trailing ".weight*" (e.g.
// "model.language_model.layers.0.mlp.gate_proj"). Encodes the checkpoint's
// config.json `ignore` list: the quantized set is the dense-MLP {gate,up,down}
// proj, the self_attn {q,k,v,o} proj, and the GDN linear_attn out_proj; every
// other Linear (GDN in_proj_*, lm_head, mtp.*, visual.*, ...) is bf16.
bool IsQwen27QuantizedLinear(const std::string& name);

// Materialize one compressed-tensors NVFP4 W4A4 Linear to an owned bf16 tensor
// in Matmul-B layout [in, out]. Reads `<proj>.weight_packed` (U8 [out, in/2]),
// `<proj>.weight_scale` (F8_E4M3 [out, in/16]) and `<proj>.weight_global_scale`
// (F32 scalar divisor); dequants to f32 via DequantCtNvfp4WeightToF32 (which
// reciprocates the global scale), rounds to bf16, and transposes. Exposed for
// unit testing. The `<proj>.input_global_scale` (activation divisor) is ignored
// on this bf16-activation correctness path (notes §3.4 / §5 step-6a).
OwnedTensor MaterializeCtNvfp4Bf16Transposed(const TensorResolver& get,
                                             const std::string& proj);

// Load one dense decoder layer. `layer_type` is "linear_attention" or
// "full_attention". Prefix is "model.language_model.layers.{layer_idx}.". Routes
// each Linear to bf16 vs W4A4-materialized-to-bf16 per IsQwen27QuantizedLinear.
Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(const TensorResolver& get,
                                               const std::string& layer_type,
                                               int64_t layer_idx);

// Full dense-model load across the given shards. Uses config.num_hidden_layers
// and config.layer_types. Text path only — the vision tower (model.visual.*)
// and image/video merger are DEFERRED (notes §0.1). The checkpoint's MTP
// head is intentionally loaded on demand by LoadQwen3_5MTP when speculative
// decoding is enabled; it is not part of the always-resident target weights.
Qwen3_5DenseWeights LoadQwen3_5Dense(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config);

// Dense single-sequence reference forward (text path). Mirrors
// Qwen3_5Model::ForwardDense but runs the dense SwiGLU MLP in place of the MoE
// block; reuses the 35B GDN + gated-attention + norm forward helpers. Returns
// logits [T, vocab] f32 (T = token_ids.size()). CPU or CUDA per `queue`.
class Qwen3_5DenseModel {
 public:
  // Batched PAGED dense forward — the 27B analogue of Qwen3_5Model::Forward.
  // Same signature/structure (paged KV cache for the full-attn layers, batched
  // GDN recurrent state for the GDN layers, the f32 residual thread), reusing the
  // 35B GDN/FullAttn paged machinery VERBATIM with the dense SwiGLU MLP
  // (RunDenseLayerPaged) in place of the MoE block. One PagedKvCache per full-attn
  // layer + one GdnStateCache per GDN layer, in layer order. Returns
  // [num_actual_tokens, vocab] f32 logits (lm_head applied). Runs on `queue`'s
  // device. See qwen3_5.h::Qwen3_5Model::Forward for the metadata contract.
  // `logits_indices` (optional): identical semantics to
  // Qwen3_5Model::Forward — gather the per-request last-token hidden rows
  // on-device before lm_head (prefill/mixed) so the return is [num_reqs, vocab].
  static std::vector<float> Forward(const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& positions,
                                    const v1::CommonAttentionMetadata& attn_meta,
                                    const v1::GDNAttentionMetadata& gdn_meta,
                                    const std::vector<PagedKvCache>& attn_kv,
                                    const std::vector<GdnStateCache>& gdn_state,
                                    const Qwen3_5DenseWeights& weights,
                                    const HfConfig& config, vt::Queue& queue,
                                    const std::vector<int32_t>& logits_indices = {});

  // DEVICE-resident variant of Forward (sampler-on-device hot path): same contract
  // as Forward but returns the lm_head output as a pool-backed DEVICE buffer
  // (ForwardLogits::device_*) with NO full-logits D2H. See
  // Qwen3_5Model::ForwardDevice.
  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const v1::GDNAttentionMetadata& gdn_meta,
      const std::vector<PagedKvCache>& attn_kv,
      const std::vector<GdnStateCache>& gdn_state,
      const Qwen3_5DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // Dense single-sequence reference forward (M0.9 anchor). Runs the whole model
  // for a single non-paged sequence and returns logits [T, vocab] f32 (T =
  // token_ids.size()). Retained as the paged==dense parity reference.
  static std::vector<float> ForwardDense(const std::vector<int32_t>& token_ids,
                                         const std::vector<int32_t>& positions,
                                         const Qwen3_5DenseWeights& weights,
                                         const HfConfig& config,
                                         vt::Queue& queue);
};

// Replay one dense decoder layer from a captured combined residual stream.
// This is the dense sibling of Qwen3_5ReplayLayer and is the focused parity
// seam for the real 27B resident W4A4 layer goldens.
std::vector<float> Qwen3_5ReplayDenseLayer(
    const Qwen3_5DenseLayerWeights& layer, const HfConfig& config,
    const std::vector<float>& hidden_in, const std::vector<int32_t>& positions,
    int64_t seqlen, vt::Queue& queue);

// 27B DENSE decode CUDA-graph driver — the dense sibling of Qwen3_5DecodeGraph
// (qwen3_5.h). Captures the PURE-DECODE dense forward once per PADDED batch size
// (kDecodeGraphSizes {1,2,4,8,16,32,64}, capped at max_num_reqs) and replays it
// per token, removing the per-step host tax (the ~62k cudaMalloc/step + kernel
// launch overhead) that the eager dense decode paid every step. The embedding is
// kept outside the capture region and run per step into a persistent hidden
// buffer; every per-call scratch is pool-backed / resident (DevicePool + the fp4
// GEMM StreamScratch pools + resident weights) so the captured region does zero
// cudaMalloc after a cold pre-warm. Bit-identical to Qwen3_5DenseModel::Forward
// for the same inputs/caches (same op sequence: DenseEmbedInto + DenseForward
// Layers). VLLM_CPP_CUDAGRAPH=0 disables capture (always eager). The 35B MoE
// graph (Qwen3_5DecodeGraph) is a separate driver and is untouched.
class Qwen3_5DenseDecodeGraph {
 public:
  // max_num_reqs == the runner's max_num_seqs (== the GDN state-cache slot count);
  // the padded decode batch is capped at this so it never exceeds the mamba/GDN
  // state cache (mirrors vLLM's decode cudagraph dispatcher, compilation.py).
  Qwen3_5DenseDecodeGraph(const Qwen3_5DenseWeights& weights, const HfConfig& config,
                          vt::Queue queue, int64_t max_num_reqs);
  ~Qwen3_5DenseDecodeGraph();
  Qwen3_5DenseDecodeGraph(const Qwen3_5DenseDecodeGraph&) = delete;
  Qwen3_5DenseDecodeGraph& operator=(const Qwen3_5DenseDecodeGraph&) = delete;

  // One PURE-DECODE step. Returns the [B, vocab] f32 logits as a DEVICE-resident
  // ForwardLogits (the captured graph's output stays on device — a view over the
  // slot's persistent logits buffer; the eager fallback owns a pool block), fed
  // straight to the sampler with NO full-logits D2H. Bit-identical to
  // Qwen3_5DenseModel::Forward for the same inputs/caches. The caller must only
  // route pure-decode batches here (all query_len==1, no prefill).
  ForwardLogits Step(const std::vector<int32_t>& token_ids,
                     const std::vector<int32_t>& positions,
                     const v1::CommonAttentionMetadata& attn_meta,
                     const v1::GDNAttentionMetadata& gdn_meta,
                     const std::vector<PagedKvCache>& attn_kv,
                     const std::vector<GdnStateCache>& gdn_state);

  // Diagnostics (A/B + tests): is a graph currently captured, and how many
  // replays have run since the last (re)capture.
  bool captured() const;
  int64_t replay_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm
