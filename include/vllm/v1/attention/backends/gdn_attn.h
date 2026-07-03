// Ported from: vllm/v1/attention/backends/gdn_attn.py @ e24d1b24
//
// Scope (M1.6 Task 4): the GDN (GatedDeltaNet) attention METADATA that segments
// a batched step into GDN prefill (chunked-scan) vs decode (recurrence)
// segments, so the M0.7 GDN ops (vt::GdnPrefill / vt::GdnDecode) can be driven
// from a batched SchedulerOutput. Behavioral only: host arrays, no CUDA, no new
// GDN kernels. This is pure routing metadata — it consumes the same
// CommonAttentionMetadata (M1.5 step-inputs) as every other backend builder and
// emits the per-request segmentation the GDN layer assembly reads.
//
// ─── ORDERING CONTRACT (from upstream split_decodes_and_prefills) ───────────
// The builder ASSUMES an already decode-first-reordered batch. Upstream does
// the reorder in the model runner via reorder_batch_to_split_decodes_and_prefills
// (utils.py:663) keyed on `reorder_batch_threshold = 1`; the builder's
// split_decodes_and_prefills (utils.py:564) then just finds the decode/prefill
// boundary in that reordered batch. We mirror the same split; the reorder is the
// runner's job (M1.5 InputBatch), not the builder's.
//
// ─── DEFERRED upstream fields (T0 gate models never exercise them) ──────────
//   * Spec-decode segmentation (spec_query_start_loc / spec_state_indices_tensor
//     / spec_sequence_masks / spec_token_indx / non_spec_token_indx /
//     num_accepted_tokens / num_spec_decodes / num_spec_decode_tokens). T0 gate
//     models (Qwen3.6) do not speculative-decode. Carried as fields for 1:1
//     fidelity; build() leaves them empty / num_spec_decodes == 0. See the
//     `spec` region below and the marked stub in build().
//   * Triton-kernel launch metadata: chunk_indices / chunk_offsets (FLA chunk
//     kernel) and nums_dict / batch_ptr / token_chunk_offset_ptr (Triton
//     causal_conv1d). OMITTED: our sequential C++ vt::GdnPrefill and
//     vt::CausalConv1dFwd consume `prefill_query_start_loc` + a
//     `has_initial_state` mask DIRECTLY (ops.h — GdnPrefill takes
//     query_start_loc; CausalConv1dFwd takes query_start_loc + has_initial_state),
//     so the chunk/conv-tile precompute has no C++ consumer. Upstream itself
//     tolerates them being None — the FLA ops recompute on the fly
//     (gdn-semantics.md §8).
//   * cudagraph capture (build_for_cudagraph_capture, the full-CG padding of the
//     request-indexed tensors) and the FLA/cutedsl prefill-backend selection.
#ifndef VLLM_V1_ATTENTION_BACKENDS_GDN_ATTN_H_
#define VLLM_V1_ATTENTION_BACKENDS_GDN_ATTN_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "vllm/v1/attention/backend.h"

namespace vllm::v1 {

// Upstream helper vllm/v1/attention/backends/utils.py:46.
inline constexpr int32_t kNullBlockId = 0;  // NULL_BLOCK_ID

// Finds the decode/prefill boundary in an already decode-first-reordered batch
// (utils.py::split_decodes_and_prefills @ e24d1b24, T0 subset:
// require_uniform=False, treat_short_extends_as_decodes=True — the only config
// GDN uses, decode_threshold=1). Sequences with query_len <= decode_threshold at
// the FRONT of the batch are decodes; the first request with query_len >
// threshold marks the start of the prefill region and everything after it is a
// prefill. Returns {num_decodes, num_prefills, num_decode_tokens,
// num_prefill_tokens}.
std::tuple<int, int, int, int> SplitDecodesAndPrefills(
    const CommonAttentionMetadata& m, int decode_threshold = 1);

// The GDN prefill/decode/spec segmentation for one batched step.
// (Upstream @dataclass GDNAttentionMetadata, gdn_attn.py:42-79.) Field names
// mirror upstream 1:1. `std::optional` mirrors upstream's `torch.Tensor | None`
// (None ⇒ nullopt). Masks are 0/1 uint8 (upstream bool tensors); index / offset
// arrays are int32 (upstream int32 tensors).
struct GDNAttentionMetadata : AttentionMetadata {
  int num_prefills = 0;
  int num_prefill_tokens = 0;
  int num_decodes = 0;
  int num_decode_tokens = 0;
  int num_spec_decodes = 0;        // DEFERRED (spec): always 0 at T0.
  int num_spec_decode_tokens = 0;  // DEFERRED (spec): always 0 at T0.
  int num_actual_tokens = 0;

  // has_initial_state = context_lens (num_computed_tokens) > 0, per request, in
  // decode-first order. Upstream ONLY populates it when num_prefills > 0 (the
  // decode path never needs it — a decode request always continues an existing
  // sequence); nullopt otherwise (gdn_attn.py:389-405).
  std::optional<std::vector<uint8_t>> has_initial_state;

  // ── Non-spec (T0) segmentation ──
  // Per-request GDN mamba-state block id = block_table column 0
  // (gdn_attn.py:219). Indexes the mamba-state cache row for each request.
  std::optional<std::vector<int32_t>> non_spec_state_indices_tensor;
  // Non-spec cumulative query offsets = the batch's query_start_loc
  // (gdn_attn.py:221), shape [num_reqs + 1].
  std::optional<std::vector<int32_t>> non_spec_query_start_loc;

  // ── Prefill-kernel inputs (drive vt::GdnPrefill) ──
  // In a MIXED non-spec batch the leading decodes are peeled off to the
  // recurrent kernel, so these are rebased to the prefill-only sub-batch:
  // prefill_query_start_loc = non_spec_query_start_loc[num_decodes:] -
  // num_decode_tokens; prefill_state_indices = state_indices[num_decodes:]
  // (gdn_attn.py:340-354). prefill_has_initial_state is the has_initial_state
  // mask sliced to the prefill region (gdn_attn.py:400-403).
  //
  // ⚠ CALLER OBLIGATION (GDN-state zeroing) — vt::GdnPrefill/GdnDecode read the
  // `state` buffer UNCONDITIONALLY (no has_initial_state gate; see
  // src/vt/cpu/cpu_ops.cpp GdnPrefillKernel). Upstream does NOT pre-zero blocks;
  // it gathers the state rows then zeros the fresh ones in the LAYER forward,
  // keyed by this mask: `initial_state = ssm_state[prefill_state_indices];
  // initial_state[~prefill_has_initial_state] = 0`
  // (qwen_gdn_linear_attn.py:1512-1513 @ e24d1b24). The batched GDN-layer
  // assembly (M0.9/runner) MUST replicate that gather-and-zero before calling
  // vt::GdnPrefill — else a request with prefill_has_initial_state==0 reads a
  // stale mamba block → silent wrong output. This metadata only DELIVERS the
  // mask; it cannot zero (host-only, holds no state buffer). Tracked in
  // .agents/state.md as the M1.5/M1.6 GDN-state-zeroing carry.
  std::optional<std::vector<int32_t>> prefill_query_start_loc;
  std::optional<std::vector<int32_t>> prefill_state_indices;
  std::optional<std::vector<uint8_t>> prefill_has_initial_state;

  // ── Spec-decode segmentation (DEFERRED T0 stub; see header) ──
  std::optional<std::vector<int32_t>> spec_query_start_loc;
  std::optional<std::vector<int32_t>> spec_state_indices_tensor;
  std::optional<std::vector<uint8_t>> spec_sequence_masks;
  std::optional<std::vector<int32_t>> spec_token_indx;
  std::optional<std::vector<int32_t>> non_spec_token_indx;
  std::optional<std::vector<int32_t>> num_accepted_tokens;
};

// Builds GDNAttentionMetadata from a batched step's CommonAttentionMetadata.
// (Upstream GDNAttentionMetadataBuilder, gdn_attn.py:82-536 — T0 non-spec path.)
class GDNAttentionMetadataBuilder final
    : public AttentionMetadataBuilder<GDNAttentionMetadata> {
 public:
  // Upstream class attribute (gdn_attn.py:85): the runner reorders the batch so
  // requests with query_len <= this threshold sit at the front.
  static constexpr int kReorderBatchThreshold = 1;

  GDNAttentionMetadataBuilder() = default;

  // T0 non-spec build. common_prefix_len (cascade attention) and fast_build are
  // accepted for interface fidelity but unused here. Spec-decode kwargs
  // (num_accepted_tokens / num_decode_draft_tokens_cpu) are DEFERRED — T0 has no
  // speculative config so num_spec == 0 and the spec branch is never taken.
  GDNAttentionMetadata build(int common_prefix_len,
                             const CommonAttentionMetadata& common_attn_metadata,
                             bool fast_build = false) override;
};

// Upstream GDNAttentionBackend (gdn_attn.py:27-38). A state-space (mamba) backend
// — no paged KV-cache shape; its cache is the GDN mamba state. get_kv_cache_shape
// is not part of the SSM contract, so it throws (mirrors the fact that the mamba
// state shape comes from MambaSpec, not the attention backend).
class GDNAttentionBackend final : public AttentionBackend {
 public:
  static constexpr const char* kName = "GDN_ATTN";

  std::string get_name() const override { return kName; }
  static bool is_ssm() { return true; }  // upstream is_ssm() -> True

  std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size,
      const std::string& cache_dtype_str = "auto") const override;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ATTENTION_BACKENDS_GDN_ATTN_H_
