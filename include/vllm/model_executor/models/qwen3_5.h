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
#include <memory>
#include <vector>

// (ForwardLogits carries either a device-resident logits buffer or a host copy;
//  see the struct below.)

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
// 2*bs*H*D; see .agents backend.h + cpu_cache.cpp). dtype is bf16 — the full-attn
// path down-casts its f32 K/V to bf16 before the write (the "auto" ReshapeAndCache
// copy requires cache dtype == k/v dtype); this mirrors vLLM's bf16 flash_attn KV
// store and halves KV memory. The query stays f32; the attention kernel converts
// the bf16 cache reads to f32 and accumulates in f32. The runner (M1.8 Task 4)
// owns/allocates it; the Forward writes into it in place.
struct PagedKvCache {
  void* data = nullptr;
  vt::DType dtype = vt::DType::kBF16;
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

// Forward result carrier (M-logits-on-device). The default hot path keeps the
// lm_head output ON DEVICE and hands it straight to the sampler (whose argmax /
// temperature / top-k/top-p kernels already run on-device — sampler.py never
// copies the full [num_reqs, vocab] logits to host, only the sampled token ids).
// This removes the per-step synchronous D2H of the full logits (~num_reqs*vocab*4
// bytes) that drained the stream every prefill AND decode step.
//
//   * DEVICE path (default): `device_storage` owns the pool-backed device buffer
//     (its deleter returns the block to the model's DevicePool, so there is NO
//     per-step cudaMalloc/cudaFree); `device_tensor` is the [rows, vocab] f32 view
//     over it. `rows == num_reqs` on the gather-before-lm_head path (prefill/mixed)
//     or a pure-decode / decode-graph step. The buffer must outlive sampling —
//     the runner holds this whole struct across execute_model -> sample_tokens.
//   * HOST path (VT_LOGITS_GATHER=0 opt-out): `host` holds the row-major
//     [rows, vocab] f32 logits (Downloaded in the forward), `device_storage` is
//     null. The runner re-gathers the per-request rows on host, exactly as before.
struct ForwardLogits {
  std::vector<float> host;                // non-empty on the HOST path
  std::shared_ptr<void> device_storage;   // owns the device buffer on the DEVICE path
  vt::Tensor device_tensor;               // [rows, vocab] view, valid iff on_device()
  int64_t rows = 0;
  int64_t vocab = 0;
  bool on_device() const { return static_cast<bool>(device_storage); }
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
  //
  // `logits_indices` (optional): the per-request last-token row indices
  // (query_start_loc[1:] - 1). When non-empty AND a proper subset of the T
  // rows (prefill/mixed: len < T), the final hidden rows are GATHERED on-device
  // BEFORE lm_head — mirroring vLLM gpu_model_runner.py:4364-4365
  // (sample_hidden_states = hidden_states[logits_indices];
  //  compute_logits(sample_hidden_states)) — so lm_head runs on len(indices)
  // rows and the return is [num_reqs, vocab] in request order. Empty (default)
  // or pure-decode (len == T) keeps the full [num_actual_tokens, vocab] return.
  static std::vector<float> Forward(const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& positions,
                                    const v1::CommonAttentionMetadata& attn_meta,
                                    const v1::GDNAttentionMetadata& gdn_meta,
                                    const std::vector<PagedKvCache>& attn_kv,
                                    const std::vector<GdnStateCache>& gdn_state,
                                    const Qwen3_5MoeWeights& weights,
                                    const HfConfig& config, vt::Queue& queue,
                                    const std::vector<int32_t>& logits_indices = {});

  // DEVICE-resident variant of Forward (the sampler-on-device hot path). Same
  // contract/args as Forward, but returns the lm_head output as a pool-backed
  // DEVICE buffer (ForwardLogits::device_*) WITHOUT the full-logits D2H — the
  // caller feeds it straight into the sampler. `rows == num_reqs` on the
  // gather-before-lm_head path (prefill/mixed) or pure-decode.
  static ForwardLogits ForwardDevice(const std::vector<int32_t>& token_ids,
                                     const std::vector<int32_t>& positions,
                                     const v1::CommonAttentionMetadata& attn_meta,
                                     const v1::GDNAttentionMetadata& gdn_meta,
                                     const std::vector<PagedKvCache>& attn_kv,
                                     const std::vector<GdnStateCache>& gdn_state,
                                     const Qwen3_5MoeWeights& weights,
                                     const HfConfig& config, vt::Queue& queue,
                                     const std::vector<int32_t>& logits_indices = {});

  // Dense single-sequence reference forward (M0.9). Runs the whole model for a
  // single non-paged sequence and returns logits [T, vocab] f32 (T =
  // token_ids.size()). `positions` is the length-T position row. Retained as the
  // parity reference for the paged==dense anchor + the dgx M0-exit greedy gate.
  static std::vector<float> ForwardDense(const std::vector<int32_t>& token_ids,
                                         const std::vector<int32_t>& positions,
                                         const Qwen3_5MoeWeights& weights,
                                         const HfConfig& config, vt::Queue& queue);

  // Eager (load-time) Marlin NVFP4 repack of every layer's routed experts +
  // dense shared-expert/lm_head projections, so the first request pays no
  // first-touch repack (previously a TTFT spike). No-op unless the build has
  // VT_MARLIN_NVFP4 and the runtime gate is on (default ON; VT_NVFP4_MARLIN=0
  // opts out) on a CUDA queue.
  static void PrepareMarlinResident(const Qwen3_5MoeWeights& weights,
                                    const HfConfig& config, vt::Queue& queue);
};

// Decode-step CUDA-graph driver (M2.5 Phase 2, gate-#1 decode-launch unlock).
// Wraps the paged forward COMPUTE body (embed -> layers -> lm_head) in a
// capture-once / replay-per-token CUDA graph for PURE-DECODE batches, collapsing
// the ~thousands of per-step kernel-launch + memcpy host-API calls into a single
// cudaGraphLaunch. Mirrors vLLM's decode CUDAGraph capture: capture keyed on the
// batch SHAPE, per-step-varying inputs threaded through PERSISTENT buffers, and
// decode-only (prefill / mixed batches stay eager, kept off this path by the
// runner).
//
// ── vt-runtime realization (deviations, so upstream ports mechanically) ──────
//   * PERSISTENT INPUTS are the HOST step vectors (token_ids / positions / the
//     attention+GDN metadata), held here and MUTATED IN PLACE each step. On GB10
//     (pageable memory access) the forward's host->device input copies are
//     capturable, so a replay re-reads each new token's inputs from the fixed
//     host addresses — no separate device staging buffers (vLLM keeps torch
//     tensors on-GPU; here the "buffer" is the host vector the copy reads from).
//   * The GDN mamba-state gather offsets and the block-table column count are
//     BAKED at capture, so the SHAPE key includes them; any change re-captures.
//   * A cold shape runs one EAGER step first (pre-warms the DevicePool + the
//     resident weights / fused-MoE constants) so the capture region does zero
//     cudaMalloc; the next same-shape step captures, and subsequent ones replay.
//   * VLLM_CPP_CUDAGRAPH=0 disables capture (always eager) for the A/B and as a
//     safety valve. Non-CUDA devices always run eager.
class Qwen3_5DecodeGraph {
 public:
  // max_num_reqs == the runner's max_num_seqs (== the GDN state-cache slot count).
  // The padded decode batch is capped at this value so it never exceeds the
  // mamba/GDN state cache (mirrors vLLM: the decode cudagraph dispatcher "already
  // caps batch sizes at max_num_seqs", compilation.py:1438-1444 @ e24d1b24).
  Qwen3_5DecodeGraph(const Qwen3_5MoeWeights& weights, const HfConfig& config,
                     vt::Queue queue, int64_t max_num_reqs);
  ~Qwen3_5DecodeGraph();
  Qwen3_5DecodeGraph(const Qwen3_5DecodeGraph&) = delete;
  Qwen3_5DecodeGraph& operator=(const Qwen3_5DecodeGraph&) = delete;

  // One PURE-DECODE step. Returns the [B, vocab] f32 logits as a DEVICE-resident
  // ForwardLogits (the captured graph's output stays on device — a view over the
  // slot's persistent logits buffer; the eager fallback owns a pool block), fed
  // straight to the sampler with NO full-logits D2H. Bit-identical to
  // Qwen3_5Model::Forward for the same inputs/caches. attn_kv / gdn_state are the
  // runner's persistent caches (stable addresses across steps). The caller must
  // only route pure-decode batches here (all query_len==1, no prefill).
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
