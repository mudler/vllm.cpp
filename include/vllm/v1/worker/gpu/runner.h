// Ported from: vllm/v1/worker/gpu/model_runner.py @ e24d1b24
// (GPUModelRunner: initialize_kv_cache / execute_model / sample_tokens / sample /
// postprocess_sampled — the T0 slice) + the decode-first reorder from
// vllm/v1/attention/backends/utils.py::reorder_batch_to_split_decodes_and_prefills.
//
// Scope (M1.8 Task 4): the batched PAGED model runner — the concrete
// ModelRunnerBase (Task 1) the Executor pass-through drives. It owns the
// persistent InputBatch (M1.5), the KV caches (allocated from the M1.3
// KVCacheConfig), and the Sampler (M1.7), and threads a step through the MRV2
// forward/sample SPLIT:
//   * execute_model(scheduler_output): update_states (M1.5, admit new reqs incl.
//     prefill_token_ids + apply cached diffs) -> DECODE-FIRST REORDER -> the
//     paged step build (prepare_inputs M1.5 -> the per-KV-group attention
//     metadata M1.6) -> Qwen3_5Model::Forward (Task 3, over the KV caches) ->
//     STASH the [num_actual_tokens, vocab] logits + the StepInputs. Returns
//     std::nullopt (MRV2 "forward done"), or nullopt-with-empty-stash for a
//     0-token flush step (no forward run).
//   * sample_tokens(): gather the logits_indices rows -> [num_reqs, vocab] ->
//     make_sampling_metadata (M1.7, dense order — CLOSES the M1.7 wiring dep) ->
//     Sampler::forward -> ModelRunnerOutput -> WRITE the sampled tokens BACK into
//     the InputBatch so the next step's prepare_inputs sees them.
//
// ─── THE V1-ALGORITHM / MRV2-CONTRACT COMPOSITION (recorded) ────────────────
// We target the MRV2 runner's forward/sample SPLIT + scheduler-output contract,
// but IMPLEMENT it via the host-array V1 algorithm (persistent InputBatch holds
// the MultiGroupBlockTable; prepare_inputs emits dense-order StepInputs; the
// SamplingMetadata is built fresh each step in the SAME dense order). This
// composes with NO idx_mapping slot-indirection: the gathered logits
// (logits[logits_indices]), the SamplingMetadata rows, the attention metadata,
// and the sampled-token write-back all align on ONE dense order. The MRV2 staged
// per-slot GPU buffers (RequestState / InputBuffers / UvaBackedTensor /
// idx_mapping / apply_staged_writes) are axis-2 STORAGE, deferred to M2 (see
// .agents/vllm-v1-v2.md).
//
// ─── THE DECODE-FIRST REORDER (four-way ordering contract) ──────────────────
// Before building metadata the runner reorders the batch decode-first-then-
// prefill (reorder_batch_to_split_decodes_and_prefills, threshold 1) because the
// GDN split (M1.6 SplitDecodesAndPrefills) assumes decodes lead. ALL consumers
// {attention seq_lens/block_table, GDN segmentation, forward input_ids/positions,
// logits_indices, SamplingMetadata rows, write-back} are built AFTER the reorder,
// in that one order. Inert for a batch-of-1 / pure-decode / pure-prefill batch;
// it only bites a MIXED hybrid batch. Uses InputBatch::swap_states (M1.5).
//
// ─── DEFERRED (marked 1:1 stubs; slot in without reshaping the interface) ────
//   cudagraphs, spec-decode / rejection sampling, LoRA, multimodal / encoder,
//   pipeline parallelism, EPLB, kv-connector, prompt-logprobs,
//   async / AsyncOutput (ModelRunnerOutput returned synchronously), the MRV2
//   staged per-slot buffers + idx_mapping (M2), IntermediateTensors / dummy_run /
//   is_profile. compute_logits: our Qwen3_5Model::Forward already applies lm_head
//   and returns full [num_actual_tokens, vocab] logits, so sample_tokens only
//   GATHERS the logits_indices rows (no separate compute_logits pass at T0).
#ifndef VLLM_V1_WORKER_GPU_RUNNER_H_
#define VLLM_V1_WORKER_GPU_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/sample/sampler.h"
#include "vllm/v1/worker/gpu/input_batch.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm::v1 {

// Decode-first reorder (utils.py::reorder_batch_to_split_decodes_and_prefills @
// e24d1b24, T0 subset). Reorders `input_batch`'s active [0, num_reqs) requests so
// all requests with num_scheduled_tokens <= decode_threshold at the front, using
// InputBatch::swap_states. The short_extend / long_extend chunked-prefill
// sub-regions collapse into the {decode, prefill} split at T0 (the gate models
// full-prefill in one chunk under the budget). Returns true iff the batch was
// modified. The builder's SplitDecodesAndPrefills (M1.6) then finds the boundary.
bool reorder_batch_to_split_decodes_and_prefills(
    InputBatch& input_batch, const SchedulerOutput& scheduler_output,
    int decode_threshold = 1);

// apply_grammar_bitmask (vllm/v1/structured_output/utils.py::apply_grammar_bitmask
// @ e24d1b24, M3.4 Task 3). Applies the per-step structured-output grammar mask
// to the gathered [num_logits, vocab] f32 `logits` (the exact tensor the runner
// feeds Sampler::forward), IN PLACE, BEFORE sampling.
//
// The `grammar_output` bitmask rows are COMPACTED — one row per structured
// request, ordered as `grammar_output.structured_output_request_ids`. `logits`
// rows are in the runner's DENSE batch order (`req_ids`, one per active request).
// This reorders each structured req's compacted bitmask row onto its dense logits
// row (utils.py:112-140), unpacks it (bit (t & 31) of word (t >> 5) SET => token t
// ALLOWED; CLEAR => FORBIDDEN), and sets every forbidden token's logit to -inf
// (reusing the M1.7 apply_allowed_token_ids -inf masking). Non-structured rows are
// untouched (all-allowed).
//
// `scheduled_spec_decode_tokens` supplies each req's spec-token count so the
// per-req logit offset (utils.py:117-118,134-138) matches upstream; spec-decode is
// deferred at T0, so the runner passes an EMPTY map (offset 0 => exactly one
// bitmask row per structured req, aligned to its dense logits row).
void apply_grammar_bitmask(
    const GrammarOutput& grammar_output,
    const std::vector<std::string>& req_ids,
    const std::map<std::string, std::vector<int32_t>>&
        scheduled_spec_decode_tokens,
    vt::Queue& queue, vt::Tensor& logits);

// The batched paged model runner (upstream GPUModelRunner, T0 slice).
class GPUModelRunner final : public ModelRunnerBase {
 public:
  // Generic model-factory path. `model` owns/borrows its concrete weights and
  // must outlive the runner (LoadedEngine declares it before runner_).
  GPUModelRunner(const HfConfig& config, LoadedModel& model,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len,
                 int max_num_batched_tokens);

  // Construct the runner over a model (config + weights) and allocate the KV
  // caches from `kv_cache_config` (initialize_kv_cache). `queue` selects the
  // device (CPU at T0; CUDA dgx-pending). The InputBatch is sized from
  // max_num_reqs / max_model_len / max_num_batched_tokens and the per-KV-group
  // block sizes taken from the KVCacheConfig groups. `config` and `weights` must
  // outlive the runner (held by reference, mirroring upstream self.model).
  GPUModelRunner(const HfConfig& config, const Qwen3_5MoeWeights& weights,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len,
                 int max_num_batched_tokens);

  // DENSE-arch overload (27B, Qwen3_5ForConditionalGeneration / num_experts==0).
  // Identical to the MoE constructor except the model runs through the dense
  // weights + the paged dense forward (Qwen3_5DenseModel::Forward). The KV-cache
  // layout is config-driven (same GDN + full-attn hybrid backbone), so
  // initialize_kv_cache is unchanged. `config` and `weights` must outlive the
  // runner. The MoE-only fp4 decode-graph fast path stays inert on this arch.
  GPUModelRunner(const HfConfig& config, const Qwen3_5DenseWeights& weights,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len,
                 int max_num_batched_tokens);

  // ModelRunnerBase (the MRV2 execute_model / sample_tokens split).
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override;
  ModelRunnerOutput sample_tokens(
      const std::optional<GrammarOutput>& grammar_output) override;

  // ─── Accessors (for tests + the ordering identity gate) ────────────────────
  InputBatch& input_batch() { return input_batch_; }
  const InputBatch& input_batch() const { return input_batch_; }
  const std::vector<PagedKvCache>& attn_kv() const { return attn_kv_; }
  const std::vector<GdnStateCache>& gdn_state() const { return gdn_state_; }
  // The step inputs / metadata built for the most recent execute_model (the four
  // consumers the ordering identity test aligns).
  const StepInputs& last_step() const { return exec_state_.step; }
  const CommonAttentionMetadata& last_attn_meta() const {
    return exec_state_.attn_meta;
  }
  const GDNAttentionMetadata& last_gdn_meta() const {
    return exec_state_.gdn_meta;
  }
  int full_attn_group_id() const { return full_attn_group_id_; }
  int gdn_group_id() const { return gdn_group_id_; }
  int64_t num_blocks() const { return num_blocks_; }
  bool kv_cache_backend_resident() const {
    return kv_cache_backend_resident_;
  }

 private:
  // Owns one persistent cache allocation. CUDA defaults to vt::Alloc-backed
  // device storage; CPU and VT_DEVICE_KV_CACHE=0 retain the host-vector
  // diagnostic fallback. Tensor/PagedKvCache views never own this memory.
  class CacheBuffer {
   public:
    CacheBuffer(vt::Device device, vt::Queue& queue, size_t bytes,
                bool backend_resident);
    ~CacheBuffer();
    CacheBuffer(const CacheBuffer&) = delete;
    CacheBuffer& operator=(const CacheBuffer&) = delete;

    void* data() {
      return backend_resident_ ? backend_data_ : host_data_.data();
    }

   private:
    vt::Device device_;
    bool backend_resident_ = false;
    void* backend_data_ = nullptr;
    std::vector<uint8_t> host_data_;
  };

  // Compatibility path for direct synthetic-weight runner tests. The wrapper
  // is type-erased but borrows the caller-owned concrete weights.
  GPUModelRunner(const HfConfig& config,
                 std::unique_ptr<LoadedModel> owned_model,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len,
                 int max_num_batched_tokens);

  // Allocate the per-full-attn-layer paged KV buffers + the per-GDN-layer
  // persistent mamba ssm/conv buffers from the KVCacheConfig groups.
  void initialize_kv_cache(const KVCacheConfig& kv_cache_config);
  // Build the [num_reqs, num_cols] committed block-table slice for a KV group.
  std::vector<int32_t> gather_block_table(int group_id, int num_reqs,
                                          int* num_cols) const;
  // Rewrite the GDN group's block-table col 0 (the mamba-state pool block-id,
  // scattered over the shared [0, num_blocks) attention pool) into a COMPACT
  // per-sequence state slot in [0, gdn_state_slots_). The GDN mamba state is one
  // recurrent state per SEQUENCE (vLLM MambaSpec "none": block_table is
  // (#reqs, 1); max_memory_usage_bytes == 1 page/seq), so the state cache is
  // sized by max_num_reqs — NOT the attention num_blocks (which grows with
  // concurrency×seq_len and made the f32 ssm_state the dominant memory
  // consumer). The map is keyed on the pool block-id (stable per sequence across
  // steps + condense), so states persist correctly; slots of finished sequences
  // are reclaimed. Only col 0 is read by the GDN builder (state indices).
  void remap_gdn_state_slots(std::vector<int32_t>& gdn_bt, int gdn_cols,
                             int num_reqs);

  const HfConfig& config_;
  // Production: model_ borrows LoadedEngine::model_. Direct runner tests use a
  // small owned adapter that in turn borrows their concrete weights.
  std::unique_ptr<LoadedModel> owned_model_;
  LoadedModel* model_ = nullptr;
  vt::Queue queue_;
  InputBatch input_batch_;
  Sampler sampler_;

  // KV group layout (resolved from the KVCacheConfig).
  int full_attn_group_id_ = -1;
  int gdn_group_id_ = -1;
  int64_t num_blocks_ = 0;
  // Persistent-batch capacity = max concurrent sequences. The GDN mamba-state
  // cache is sized by this (one recurrent state per sequence), decoupled from
  // the attention num_blocks. See remap_gdn_state_slots.
  int max_num_reqs_ = 0;
  int64_t gdn_state_slots_ = 0;
  // Compact GDN state-slot allocator: pool block-id (block_table col 0) -> slot
  // in [0, gdn_state_slots_); free list of unused slots.
  std::unordered_map<int32_t, int32_t> gdn_slot_of_block_;
  std::vector<int32_t> gdn_free_slots_;

  // Owned persistent cache storage plus the non-owning views used by forward.
  // CUDA uses backend allocations by default (VT_DEVICE_KV_CACHE=0 restores the
  // former host-vector storage for same-binary attribution); CPU stays host.
  // Full-attention KV is bf16 by default, or f32 under VT_KV_CACHE_F32.
  bool kv_cache_backend_resident_ = false;
  std::vector<std::unique_ptr<CacheBuffer>> full_attn_buf_;
  // GDN recurrent + conv state caches. On CUDA they are bf16 (gdn_cache_dtype_),
  // mirroring vLLM's default mamba_cache_dtype/mamba_ssm_cache_dtype="auto" →
  // model dtype (bf16 for Qwen3-Next): fla fused_recurrent reads bf16 → f32
  // registers → writes bf16, which the CUDA GDN decode/conv kernels replicate.
  // This halves the decode recurrent-state read/write traffic and the resident
  // state-cache footprint. On CPU (the unit tests) they stay f32 — the CPU GDN
  // ops are f32-only and the exact-value tests assume f32. Storage is raw bytes
  // sized by gdn_cache_dtype_ (0 bytes == +0.0f in both f32 and bf16).
  vt::DType gdn_cache_dtype_ = vt::DType::kF32;
  std::vector<std::unique_ptr<CacheBuffer>> ssm_buf_;
  std::vector<std::unique_ptr<CacheBuffer>> conv_buf_;
  std::vector<PagedKvCache> attn_kv_;
  std::vector<GdnStateCache> gdn_state_;

  // Stashed forward result between execute_model and sample_tokens (upstream
  // ExecuteModelState — hidden_states + input_batch handoff, here the full
  // logits + the dense-order step). num_reqs == 0 marks a 0-token flush step.
  struct ExecuteModelState {
    // The forward result. DEFAULT: a DEVICE-resident [num_reqs, vocab] logits
    // buffer (ForwardLogits::device_*) fed straight to the sampler — NO per-step
    // full-logits D2H. On the VT_LOGITS_GATHER=0 opt-out it carries host logits
    // ([num_actual_tokens, vocab]) and sample_tokens re-gathers on host as before.
    ForwardLogits logits;
    int num_actual_tokens = 0;
    int num_reqs = 0;
    StepInputs step;
    CommonAttentionMetadata attn_meta;
    GDNAttentionMetadata gdn_meta;
    std::vector<std::string> req_ids;  // dense order (== input_batch order)
  } exec_state_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_RUNNER_H_
