// vllm.cpp original (Qwen3.6-35B-A3B MoE forward). Semantics mirrored 1:1 from
// the pinned upstream forward (vllm/model_executor/models/qwen3_next.py +
// qwen3_5.py @ e24d1b24): embed -> N decoder layers (GDN linear_attention OR
// dense full_attention, each followed by the sparse-MoE block) -> final RMSNorm
// -> lm_head. Single-sequence, non-paged, prefill-only (all T tokens at once,
// causal). Assembly reference: .agents/qwen36-forward-notes.md,
// .agents/gdn-semantics.md, .agents/moe-semantics.md.
//
// The forward is an explicit function over the vt ops (no nn.Module system),
// per the design doc. Activations flow bf16 (matching the model's bf16 hidden
// states); the residual stream is kept f32 (upstream fused_add_rms_norm
// accumulates the residual in f32 before rounding); GDN/attention internals run
// f32. Weights are the owned host bf16 tensors from qwen3_5_weights.h.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm {

class Qwen3_5Model {
 public:
  // Runs the whole model forward for a single sequence and returns the logits
  // as a row-major [T, vocab] f32 buffer (T = token_ids.size()). `positions`
  // is the length-T token position row (0..T-1 for a fresh prompt); the
  // full-attention layers use it for partial NeoX RoPE. Runs on `queue`'s
  // device (CPU or CUDA). Throws (VT_CHECK/runtime_error) on any shape/config
  // mismatch.
  static std::vector<float> Forward(const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& positions,
                                    const Qwen3_5MoeWeights& weights,
                                    const HfConfig& config, vt::Queue& queue);
};

// Per-layer parity replay: runs ONE decoder layer over the combined residual
// stream. `hidden_in` is the [T*H] f32 stream INTO the layer (= residual +
// hidden, as the pinned oracle reconstructs it); `positions` is the length-T
// position row (only used by full_attention layers). Returns the combined
// stream OUT [T*H] f32 (= residual + hidden after the layer), directly
// comparable to the qwen36 layer goldens' `out`. Fresh zero GDN conv/ssm state.
std::vector<float> Qwen3_5ReplayLayer(const Qwen3_5MoeLayerWeights& layer,
                                      const HfConfig& config,
                                      const std::vector<float>& hidden_in,
                                      const std::vector<int32_t>& positions,
                                      int64_t seqlen, vt::Queue& queue);

}  // namespace vllm
