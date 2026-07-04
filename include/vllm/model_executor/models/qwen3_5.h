// vllm.cpp original (Qwen3.6-35B-A3B MoE forward). Semantics mirrored 1:1 from
// the pinned upstream forward (vllm/model_executor/models/qwen3_next.py +
// qwen3_5.py @ e24d1b24): embed -> N decoder layers (GDN linear_attention OR
// dense full_attention, each followed by the sparse-MoE block) -> final RMSNorm
// -> lm_head. Assembly reference: .agents/qwen36-forward-notes.md,
// .agents/gdn-semantics.md, .agents/moe-semantics.md.
//
// M1.8 Task 3 — THE CENTRAL REFACTOR (dense → paged). The primary `Forward` now
// consumes the batched paged KV cache (full-attn: vt::ReshapeAndCache +
// vt::PagedAttention over the (num_blocks,2,block_size,H,D) buffers; GDN: the
// batched GDNAttentionMetadata segmentation over the PERSISTENT mamba
// ssm_state/conv_state). It takes the flattened dense-order step inputs
// (token_ids, positions [both length num_actual_tokens]) + the per-KV-group
// attention metadata + the KV caches, and returns [num_actual_tokens, vocab] f32
// logits (matching the M0.9 return). `ForwardDense` retains the M0.9
// single-sequence dense path as the parity reference (paged==dense anchor) and
// the dgx M0-exit greedy gate (registry entry) — it is unchanged.
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
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

// Per-full-attn-layer paged KV cache: the FlashAttention V1 buffer
// (num_blocks, 2, block_size, num_kv_heads, head_size) referenced by base ptr +
// dims. Rank 5 exceeds vt::kMaxRank (4), so the buffer is carried raw and the
// Forward builds the two unbind(1) rank-4 strided K/V views (block stride
// 2*bs*H*D; see .agents backend.h + cpu_cache.cpp). dtype MUST match the f32
// q/k/v the T0 full-attn path produces (the "auto" ReshapeAndCache copy requires
// cache dtype == k/v dtype). The runner (M1.8 Task 4) owns/allocates it; the
// Forward writes into it in place.
struct PagedKvCache {
  void* data = nullptr;
  vt::DType dtype = vt::DType::kF32;
  int64_t num_blocks = 0;
  int64_t block_size = 0;
  int64_t num_kv_heads = 0;
  int64_t head_size = 0;
};

// Per-GDN-layer PERSISTENT mamba state (device buffers, updated in place). Rows
// are indexed by the GDN metadata's state indices (block_table column 0). The
// Forward gathers the per-request rows, runs the recurrence/conv, and scatters
// the updated rows back — including the GDN-STATE ZEROING obligation
// (qwen_gdn_linear_attn.py:1513-1514: zero the ssm rows whose
// prefill_has_initial_state==0 before vt::GdnPrefill).
struct GdnStateCache {
  vt::Tensor ssm_state;   // [num_state_blocks, Hv, Dv, Dk] f32, in/out
  vt::Tensor conv_state;  // [num_state_blocks, conv_dim, K-1] f32, in/out
};

class Qwen3_5Model {
 public:
  // Batched PAGED forward (M1.8 Task 3, the central refactor). Runs the whole
  // model over a flattened, decode-first-reordered step and returns the logits
  // as a row-major [num_actual_tokens, vocab] f32 buffer.
  //
  //   token_ids / positions  the length-num_actual_tokens flattened step inputs
  //                          (positions[t] = the token's absolute position).
  //   attn_meta              the full-attn KV group's CommonAttentionMetadata
  //                          (query_start_loc / seq_lens / block_table_tensor /
  //                          slot_mapping; M1.6 Task 1).
  //   gdn_meta               the GDN KV group's GDNAttentionMetadata
  //                          (num_prefills/num_decodes segmentation + state
  //                          indices + prefill_has_initial_state; M1.6 Task 4).
  //   attn_kv                one PagedKvCache per FULL-attn layer, in layer order.
  //   gdn_state              one GdnStateCache per GDN layer, in layer order.
  //
  // Runs on `queue`'s device (CPU or CUDA). Throws (VT_CHECK/runtime_error) on
  // any shape/config mismatch.
  static std::vector<float> Forward(const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& positions,
                                    const v1::CommonAttentionMetadata& attn_meta,
                                    const v1::GDNAttentionMetadata& gdn_meta,
                                    const std::vector<PagedKvCache>& attn_kv,
                                    const std::vector<GdnStateCache>& gdn_state,
                                    const Qwen3_5MoeWeights& weights,
                                    const HfConfig& config, vt::Queue& queue);

  // Dense single-sequence reference forward (M0.9). Runs the whole model for a
  // single non-paged sequence and returns logits [T, vocab] f32 (T =
  // token_ids.size()). `positions` is the length-T position row. Retained as the
  // parity reference for the paged==dense anchor + the dgx M0-exit greedy gate.
  static std::vector<float> ForwardDense(const std::vector<int32_t>& token_ids,
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
