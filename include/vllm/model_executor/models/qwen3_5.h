// vllm.cpp original (Qwen3.6-35B-A3B MoE forward). Semantics mirrored 1:1 from
// the pinned upstream forward (vllm/model_executor/models/qwen3_next.py +
// qwen3_5.py @ e24d1b24): embed -> N decoder layers (GDN linear_attention OR
// dense full_attention, each followed by the sparse-MoE block) -> final RMSNorm
// -> lm_head. Assembly reference: .agents/specs/qwen36-forward-notes.md,
// .agents/specs/gdn-semantics.md, .agents/specs/moe-semantics.md.
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

#include <array>
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
#include "vt/fp8_kv.h"  // KV-FP8 W3: the PagedKvCache fp8 interpretation
#include "vt/tensor.h"

namespace vllm {

// Process-wide count of MIXED spec+non-spec GDN batch invocations
// (GdnBlockPagedMixedSpec), incremented per GDN layer per mixed step. A nonzero
// value proves the concurrency split/merge path actually ran; the c>1 spec
// identity gate reads it to prove it exercised the mixed batch, not just the
// pure-spec fast path. Reset lets a test scope the count to its own run.
int64_t Qwen3_5MixedSpecInvocations();
void ResetQwen3_5MixedSpecInvocations();

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
  // KV-FP8 W3 — carried straight from the layer's `AttentionSpec` by the runner
  // and consumed by `dense_attn::WriteKvCache` / `dense_attn::ApplyKvCacheQuant`.
  // ADDITIVE and default-inert: `kAuto` means `dtype` is a float cache and the
  // store/read take the byte-identical float path they always did.
  //
  // `dtype` and `fp8_kind` travel TOGETHER on purpose. `dtype == kI8` sizes the
  // page at one byte per element; `fp8_kind` says what that byte means. A view
  // that carried only the first would read fp8 bytes as int8 and only the second
  // would index a half-sized page at full width — both are wrong tokens rather
  // than a crash, which is why they are one struct and not two.
  vt::Fp8KVCacheDataType fp8_kind = vt::Fp8KVCacheDataType::kAuto;
  float k_scale = 1.0F;
  float v_scale = 1.0F;
};

// Per-GDN-layer PERSISTENT mamba state (device buffers, updated in place). Rows
// are indexed by the GDN metadata's state indices (block_table column 0). The
// Forward gathers the per-request rows, runs the recurrence/conv, and scatters
// the updated rows back — including the GDN-STATE ZEROING obligation
// (qwen_gdn_linear_attn.py:1513-1514: zero the ssm rows whose
// prefill_has_initial_state==0 before vt::GdnPrefill).
struct GdnStateCache {
  // Dtypes are independent, mirroring upstream MambaStateDtypeCalculator.
  // Gate checkpoints use FP32 SSM state and BF16 convolution state on CUDA;
  // every model boundary honors the declared F16/BF16/F32 dtype independently.
  vt::Tensor ssm_state;   // [num_state_blocks, Hv, Dv, Dk], in/out
  vt::Tensor conv_state;  // [num_state_blocks, conv_dim, K-1], in/out
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
  // True when device_storage is a NON-owning view over an externally-held buffer
  // (the decode-graph slot's persistent logits — ViewDeviceLogits). The deleter is
  // a no-op, so releasing this carrier frees nothing: the runner reads it to decide
  // whether resetting exec_state_ would free an in-flight-read pool block (the eager
  // WrapDeviceLogits case, hazard-A) or not (this graphed case). See the
  // VT_ASYNC_EXECUTOR drain-skip in runner.cpp.
  bool non_owning_view = false;
  bool on_device() const { return static_cast<bool>(device_storage); }
};

// The MTP drafter's hidden-state carrier (defined in qwen3_5_mtp.h). Forward-
// declared here so the target forward's hidden-state tap (ForwardDeviceTap,
// SPEC-MTP I5c) can hand back the [T,H] post-final-norm hidden by pointer without
// pulling the MTP header into this one (qwen3_5_mtp.h includes this header).
struct Qwen3_5MTPHiddenStates;

// DFlash DF-AUX-TAPS multi-layer auxiliary hidden-state carrier (SPEC-DFLASH D1).
// Where the SPEC-MTP single tap (Qwen3_5MTPHiddenStates) captures ONE post-final-
// norm [T,H] hidden, the DFlash drafter conditions on the target's residual stream
// at a LIST of intermediate layer boundaries, combined by an `fc` (H×taps -> H;
// qwen3_dflash.py:411-419 + combine_hidden_states :750-770). This carrier holds the
// concatenated [T, H×taps] buffer that mirrors vLLM's eagle3 aux capture:
//   * `layer_ids` are 0-based target-model decoder indices — the DFlash draft's
//     `target_layer_ids` (27B: [1,16,31,46,61]; 35B: [1,6,11,16,22,27,32,37]).
//     For each id L the forward captures (hidden_states + residual), the residual-
//     stream value AFTER decoder layer L, i.e. exactly the value vLLM appends at
//     eagle3 aux key L+1 (interfaces.py:1382 `hidden_states + residual`;
//     eagle3_utils.py:41-56 shifts DFlash's ids by +1 -> keys {2,17,32,47,62}).
//   * `tensor` is bf16 [T, H×layer_ids.size()], the taps concatenated along the
//     last dim in ASCENDING layer_ids order (matching cat(aux, dim=-1) fed to
//     `fc`). Column block k = tap for layer_ids[k].
// Config-gated: only a runner that sets ModelForwardInput::aux_tap routes to the
// multi-tap forward; the default (and MTP-spec) forward never allocates or writes
// it and is byte-identical (identical inertness discipline to the single tap).
struct Qwen3_5AuxTaps {
  std::vector<int32_t> layer_ids;      // ascending 0-based capture-after indices
  std::shared_ptr<void> storage;       // owns the pool-backed device buffer
  vt::Tensor tensor;                   // bf16 [T, H×taps], concat order = layer_ids
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

  // ForwardDevice + the DRAFTER hidden-state tap (SPEC-MTP I5c). Identical to
  // ForwardDevice (same logits, same op sequence — the tap is an inert copy of the
  // full [num_actual_tokens, H] post-final-norm hidden), and additionally moves
  // that hidden into `*hidden_out` (a device-owning carrier) so the MTP drafter's
  // propose() consumes the exact tensor upstream feeds Qwen3_5MTP.forward as
  // `hidden_states` (qwen3_5.py returns self.model(...) output). `hidden_out` may
  // be null (then it is exactly ForwardDevice). Not wired into the runner loop
  // until I5d; default path never calls it, so the engine is byte-identical.
  static ForwardLogits ForwardDeviceTap(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const v1::GDNAttentionMetadata& gdn_meta,
      const std::vector<PagedKvCache>& attn_kv,
      const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
      const HfConfig& config, vt::Queue& queue, Qwen3_5MTPHiddenStates* hidden_out,
      const std::vector<int32_t>& logits_indices = {});

  // ForwardDevice + the DFlash multi-layer aux hidden-state taps (SPEC-DFLASH D1,
  // DF-AUX-TAPS). Byte-identical logits to ForwardDevice, and additionally captures
  // the residual stream (hidden + residual, bf16) at each boundary in
  // `aux_out->layer_ids` into `aux_out->tensor` = [T, H×taps] (concat order =
  // layer_ids), moving the owning device buffer into `aux_out->storage`. The
  // capture mirrors vLLM's eagle3 aux collection (interfaces.py:1382); see
  // Qwen3_5AuxTaps. `aux_out` may be null (then it is exactly ForwardDevice). Not
  // wired into the runner loop until DFlash D4; the default path never calls it, so
  // the shipped engine is byte-identical.
  static ForwardLogits ForwardDeviceMultiTap(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const v1::GDNAttentionMetadata& gdn_meta,
      const std::vector<PagedKvCache>& attn_kv,
      const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
      const HfConfig& config, vt::Queue& queue, Qwen3_5AuxTaps* aux_out,
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
                     const std::vector<GdnStateCache>& gdn_state,
                     // SPEC-DSPARK W8 (#442): non-null captures the DFlash/DSpark
                     // aux hidden taps into this slot's PERSISTENT [S, H*taps]
                     // buffer and points `aux_out->tensor` at it. The view is
                     // valid until this slot's next replay, the same contract the
                     // returned logits already carry. Null keeps the pure-decode
                     // behavior byte-identical.
                     Qwen3_5AuxTaps* aux_out = nullptr);

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

// ── The MoE arm's VL forward (issue #891, .agents/specs/moe-vision-tower.md) ──
//
// Single-image / single-video, single-sequence GREEDY generation on the
// Qwen3.6-35B-A3B GDN-hybrid MoE backbone (`Qwen3_5MoeForConditionalGeneration`).
// The MoE sibling of `Qwen3_5VLGenerateGreedy` / `...Video` (qwen3_5_dense.h),
// sharing ONE implementation with them: upstream composes the SAME
// `Qwen3_VisionTransformer` on both conditional-generation classes over two text
// backbones, so the tower (`LoadQwen3_5MoeVision` -> `Qwen3VLVisionForward`), the
// processor, the MRoPE index math and the greedy driver are shared and only the
// backbone consuming the merger output differs.
//
// The forward is FORKED and GATED ON MM INPUT: these entry points are reached
// only for a request that carries an image or a video. A text-only request goes
// through the registered `ForwardQwen3_5Moe`, which is untouched.
//
// Two mm-only points are active, exactly as on the dense arm: (1) inputs_embeds
// — embed(prompt_ids) then scatter the tower merger output [N,H] into the
// image/video-token rows; (2) MRoPE — the full-attention layers' cos|sin cache is
// built from the `Qwen3VLGetRopeIndex[Video]` positions [3,T] +
// `config.mrope_section` ([11,11,10] interleaved) instead of the 1-D RoPE cache.
// NO DeepStack (`deepstack_visual_indexes: []` on this family). GDN
// (linear_attention) layers carry no rope. Allocates its own paged KV (full-attn)
// + GDN recurrent state and greedy-decodes up to max_new_tokens (stops on
// eos_token_id).
//
// prompt_ids : placeholder-expanded model input ids.
// mm_main    : the tower merger output [N, H] (== out_hidden_size == hidden_size,
//              2048 on the 35B), host f32; scattered (bf16-rounded) into the
//              visual rows. N MUST equal the visual-token count.
// grid_thw   : the LLM-grid source (t,h,w) for get_rope_index.
std::vector<int32_t> Qwen3_5MoeVLGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::array<int64_t, 3>& grid_thw, int32_t image_token_id,
    int32_t eos_token_id, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens);

// Video sibling. The two video differences (as on the dense arm) are (a) the
// merge mask is on video_token_id across all frames and (b) the MRoPE prefill
// positions come from `Qwen3VLGetRopeIndexVideo`, the per-frame,
// timestamp-interleaved scan.
std::vector<int32_t> Qwen3_5MoeVLGenerateGreedyVideo(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::array<int64_t, 3>& grid_thw, int32_t video_token_id,
    int32_t vision_start_token_id, int32_t vision_end_token_id,
    int32_t eos_token_id, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens);

}  // namespace vllm
