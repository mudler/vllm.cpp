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
#include <unordered_set>
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
#include "vllm/v1/worker/gpu/async_output.h"
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

  ~GPUModelRunner();

  // ModelRunnerBase (the MRV2 execute_model / sample_tokens split).
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override;
  ModelRunnerOutput sample_tokens(
      const std::optional<GrammarOutput>& grammar_output) override;

  // sample_tokens_async (ENG-ASYNC-SCHED W3 sampler-OUTPUT half). The overlap
  // variant of sample_tokens: it produces the sampled ids DEVICE-RESIDENT (the
  // Sampler writes them into a fresh device buffer, no host download), does the
  // on-GPU post_update (last_sampled scatter), and returns an
  // AsyncGPUModelRunnerOutput that issued the single non-blocking sampled-id D2H
  // on the runner's COPY queue with a completion event — the MAIN queue is never
  // synchronized. The engine calls get_output() during output processing (off the
  // model's critical path), which blocks only on that copy event, so the copy
  // overlaps the next step's forward: this is the ~3.25 ms/step GPU-idle capture.
  // Token-exact with sample_tokens by construction (same argmax, same ids); on a
  // synchronous/unified backend the copy is a memcpy and events no-op. Only
  // meaningful when runner_supports_async() is true (async engaged). Mirrors
  // gpu_model_runner.py:4673-4693 (AsyncGPUModelRunnerOutput construction).
  std::unique_ptr<AsyncModelRunnerOutput> sample_tokens_async(
      const std::optional<GrammarOutput>& grammar_output) override;

  // runner_supports_async (mirror of the vLLM compat gate feeding
  // SchedulerConfig::ResolveAsyncScheduling — vllm/config/vllm.py:990-1038). TRUE
  // iff the runner advertises the placeholder-aware async device path: it is
  // engaged (async_input_combine(), from VT_ASYNC_RUNNER / set_async_input_combine)
  // AND runs on a backend whose async-output primitives are wired (CUDA, or CPU's
  // synchronous degeneration for the CPU gate). DEFAULT ON since the 2026-07-17
  // flip (VT_ASYNC_RUNNER default ON, async_runner_flag.h): with async on this
  // returns true, so ResolveAsyncScheduling resolves the AsyncScheduler + mcb=2 by
  // default, mirroring vLLM's async-scheduling default. VT_ASYNC_RUNNER=0 rolls the
  // runner back to the synchronous host path (returns false → sync Scheduler);
  // VT_ASYNC_SCHED=0 rolls only the scheduler back to synchronous in the same binary
  // while the runner stays async-capable.
  bool runner_supports_async() const override { return async_input_combine_; }

  // ─── Accessors (for tests + the ordering identity gate) ────────────────────
  InputBatch& input_batch() { return input_batch_; }
  const InputBatch& input_batch() const { return input_batch_; }
  const std::vector<PagedKvCache>& attn_kv() const { return attn_kv_; }
  const std::vector<GdnStateCache>& gdn_state() const { return gdn_state_; }
  // The compact GDN state-slot pool size (== max_num_reqs). Exposed for the
  // state-slot uniqueness regression tests.
  int64_t gdn_state_slots() const { return gdn_state_slots_; }
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

  // Async-scheduling device-input path (ENG-ASYNC-SCHED W3 runner leaf). When
  // ON, execute_model rebuilds each decode row's input token id from the
  // GPU-resident-analog InputBatch::last_sampled_tokens via
  // combine_sampled_and_draft_tokens INSTEAD of the host token_ids_cpu read —
  // the piece that lets step N+1 be prepared without waiting on step N's sampled
  // token to cross to the host (the ~3.25 ms/step idle). DEFAULT ON since the
  // 2026-07-17 flip (VT_ASYNC_RUNNER default ON): the production runner engages the
  // async device path so `runner_supports_async` is TRUE by default; the DGX
  // token-exactness gates + the 6ea7856 discriminator proved greedy tokens are
  // bit-identical to the sync path. VT_ASYNC_RUNNER=0 rolls it back to the
  // synchronous host path at construction; tests toggle it directly. Greedy tokens
  // are bit-identical in both modes (combine writes the same id sample_tokens wrote
  // back to token_ids_cpu).
  void set_async_input_combine(bool enabled) {
    async_input_combine_ = enabled;
  }
  bool async_input_combine() const { return async_input_combine_; }
  bool kv_cache_backend_resident() const {
    return kv_cache_backend_resident_;
  }
  vt::Device device() const { return queue_.device; }

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
  // Rewrite the GDN group's block-table col 0 into a COMPACT per-sequence state
  // slot in [0, gdn_state_slots_). The GDN mamba state is one recurrent state
  // per SEQUENCE, kept in a compact cache sized by max_num_reqs — NOT the
  // attention num_blocks (which grows with concurrency×seq_len and made the f32
  // ssm_state the dominant memory consumer). The pool is keyed on the request's
  // stable IDENTITY (req_id), NOT the mamba pool block-id: under our MambaSpec
  // "none" config the group uses a sub-sequence block_size, so once a sequence
  // exceeds one block MambaManager::remove_skipped_blocks nulls every block but
  // the last and block-table col 0 collapses to the shared null block-id 0 —
  // block-id keying then maps every long sequence to ONE slot (a duplicate live
  // state index / cross-request state corruption). vLLM reaches the same
  // per-sequence state index via mamba_get_block_table_tensor (gathering the
  // CURRENT state block); because our compact per-sequence cache makes the
  // physical block-id irrelevant, the sequence identity is the correct key: each
  // live request owns exactly one slot for its whole lifetime, freed only when
  // it leaves the batch and reused only after. Only col 0 is read by the GDN
  // builder (state indices).
  void remap_gdn_state_slots(std::vector<int32_t>& gdn_bt, int gdn_cols,
                             int num_reqs,
                             const std::vector<std::optional<std::string>>&
                                 req_ids);

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
  // Compact GDN state-slot allocator: request identity (req_id) -> slot in
  // [0, gdn_state_slots_); free list of unused slots. Keyed on the sequence, not
  // the mamba pool block-id (see remap_gdn_state_slots for why block-id keying
  // collapsed long concurrent sequences onto one slot).
  std::unordered_map<std::string, int32_t> gdn_slot_of_req_;
  std::vector<int32_t> gdn_free_slots_;
  // Reused per-step scratch for the live-request set in remap_gdn_state_slots:
  // cleared and refilled each step (buckets stay allocated) so the compact
  // slot remap does no per-step set allocation.
  std::unordered_set<std::string> gdn_alive_scratch_;

  // Owned persistent cache storage plus the non-owning views used by forward.
  // CUDA uses backend allocations by default (VT_DEVICE_KV_CACHE=0 restores the
  // former host-vector storage for same-binary attribution); CPU stays host.
  // Full-attention KV is bf16 by default, or f32 under VT_KV_CACHE_F32.
  bool kv_cache_backend_resident_ = false;
  // Async-scheduling device-input opt-in (see set_async_input_combine). Default
  // from VT_ASYNC_RUNNER at construction; OFF keeps the sync host path.
  bool async_input_combine_ = false;
  // Dedicated COPY queue for the async sampled-id D2H (async_output_copy_stream,
  // gpu_model_runner.py:711-716,1137-1141). Created lazily on the first async
  // sample so the sync path allocates no extra stream; destroyed in the dtor.
  // id == 0 marks "not created". The copy runs OUTSIDE any CUDA-graph capture.
  vt::Queue async_copy_queue_{};
  // Lazily create + return the async-output copy queue on the runner's device.
  vt::Queue& get_or_create_async_copy_queue();
  // Persistent pool of the per-step overlap resources (device sampled-id buffer +
  // pinned host buffer + events), so sample_tokens_async does NO per-step
  // cudaMalloc/cudaHostAlloc/cudaEventCreate (each of which device-syncs and
  // would serialize the depth-2 overlap). Lazily created on the first async
  // sample; freed in the dtor. Mirrors torch's caching device/pinned allocators.
  std::unique_ptr<AsyncOutputPool> async_output_pool_;
  AsyncOutputPool& get_or_create_async_output_pool();
  // Assemble the [num_reqs, vocab] logits the sampler runs on (the three-case
  // device/host gather from the stashed forward result) and apply the grammar
  // bitmask, IN the exact order the sync path uses. Shared by sample_tokens and
  // sample_tokens_async. `sampled_logits` is caller-owned scratch the returned
  // Tensor view may alias (host / VT_GPU_SAMPLE=0 paths); it must outlive the
  // sampler call. Requires exec_state_.num_reqs > 0.
  vt::Tensor assemble_sample_logits(
      const std::optional<GrammarOutput>& grammar_output,
      std::vector<float>& sampled_logits);
  std::vector<std::unique_ptr<CacheBuffer>> full_attn_buf_;
  // GDN convolution and recurrent caches have independent dtypes. This mirrors
  // MambaStateDtypeCalculator::_mamba_state_dtype: mamba_cache_dtype="auto"
  // makes conv state use model dtype (BF16 for the gate models), while Qwen3.5's
  // verify hook copies config.json mamba_ssm_dtype into the temporal/SSM cache
  // dtype (FP32 for both gate checkpoints). Kernels accumulate in FP32 and store
  // each cache through its own declared dtype. Allocation consumes the
  // MambaSpec directly on every backend; raw storage is sized independently and
  // all-zero bytes represent +0.0 in each supported floating type.
  vt::DType gdn_conv_cache_dtype_ = vt::DType::kF32;
  vt::DType gdn_ssm_cache_dtype_ = vt::DType::kF32;
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
    // discard_request_mask (gpu_model_runner.py:2048): per dense batch row, 1 iff
    // the request is still consuming its known prefill tokens this step
    // (optimistic seq_len < num_tokens) and so must NOT sample — its sampled
    // token is cleared to empty before returning.
    std::vector<uint8_t> discard;
  } exec_state_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_RUNNER_H_
