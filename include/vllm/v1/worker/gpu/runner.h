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
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vllm/config/speculative.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d hidden tap + draft
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/model_executor/models/qwen3_dflash.h"  // SPEC-DFLASH D5 draft + aux taps
#include "vllm/model_executor/models/qwen3_dspark.h"  // SPEC-DSPARK W5 draft + Markov head
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/sample/device_scratch.h"  // HostBufferStaging (#1313)
#include "vllm/v1/sample/sampler.h"
#include "vllm/v1/worker/gpu/async_output.h"
#include "vllm/v1/worker/gpu/input_batch.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"
#include "vllm/v1/worker/gpu/pool/pooling_runner.h"  // PoolingRunner (pooling arch)
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm::v1::kv_offload {
class KVConnector;  // KV-EXTERNAL-CACHE: worker-side store/load seam (fwd-decl).
}  // namespace vllm::v1::kv_offload

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
  //
  // SPEC-MTP I5d-pre: the trailing draft parameters are the ENABLING seam for the
  // verify/propose loop (I5d). All three default to "no speculation", so every
  // existing construction site is byte-identical: `spec_config == nullopt` and a
  // null `draft_model` leave the runner on the current single-model decode path.
  // When set (I5d), `draft_model` is the MTP draft the runner owns for its
  // lifetime (built via LoadedModel::BuildMtpDraft) and `draft_kv` is its paged
  // KV layer (the `fa_draft` group). This increment only STORES them; nothing
  // reads them yet.
  GPUModelRunner(const HfConfig& config, LoadedModel& model,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len, int max_num_batched_tokens,
                 std::optional<vllm::SpeculativeConfig> spec_config =
                     std::nullopt,
                 std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model = nullptr,
                 std::vector<PagedKvCache> draft_kv = {});

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

  // take_draft_token_ids (SPEC-MTP I5d; gpu/model_runner.py:1483-1489). Returns
  // the out-of-band drafts the verify/propose loop produced this step, moving
  // them out (a second call returns nullopt). EngineCore::post_step pulls these
  // and installs them on the scheduler for the next verify step. nullopt when no
  // speculator is configured (the production default) — byte-identical.
  std::optional<DraftTokenIds> take_draft_token_ids() override {
    std::optional<DraftTokenIds> out = std::move(pending_drafts_);
    pending_drafts_.reset();
    return out;
  }

  // runner_supports_async (mirror of the vLLM compat gate feeding
  // SchedulerConfig::ResolveAsyncScheduling — vllm/config/vllm.py:990-1038).
  // TRUE iff the runner advertises async SCHEDULING capability: the env opt-in
  // (VT_ASYNC_RUNNER, default ON since the 2026-07-17 flip) on a backend whose
  // async-output primitives are wired (CUDA, or CPU's synchronous
  // degeneration for the CPU gate). With it true, ResolveAsyncScheduling
  // resolves the AsyncScheduler + mcb=2 by default, mirroring vLLM.
  // VT_ASYNC_RUNNER=0 rolls the runner back (false → sync Scheduler);
  // VT_ASYNC_SCHED=0 rolls only the scheduler back in the same binary.
  //
  // SPEC-DFLASH2 W7 (#1824): this is deliberately NOT async_input_combine_ —
  // that lever carries the spec veto (the device combine is not draft-aware,
  // I5e) and stays OFF under a speculator, while async SCHEDULING now stays ON
  // for the Eagle-type family exactly as upstream resolves it
  // (vllm/config/vllm.py:1064-1112): the spec engine keeps the synchronous
  // host input path and the host spec sampler, and the scheduler still
  // overlaps step N+1 with step N's output processing. The two were one flag
  // while spec forced sync scheduling; W7 is the commit that splits them.
  bool runner_supports_async() const override { return async_sched_supported_; }

  // SPEC-DFLASH2 W7 (#1824): whether the ENGINE resolved async scheduling ON.
  // Set once by LoadedEngine after ResolveAsyncEnabled (the runner cannot know
  // — the resolution also reads VT_ASYNC_SCHED and the pooling arm); read by
  // execute_model's draft-placeholder fill and computed-token correction.
  void set_async_scheduling(bool enabled) { use_async_scheduling_ = enabled; }
  bool use_async_scheduling() const { return use_async_scheduling_; }

  // ─── Accessors (for tests + the ordering identity gate) ────────────────────
  InputBatch& input_batch() { return input_batch_; }
  const InputBatch& input_batch() const { return input_batch_; }
  const std::vector<PagedKvCache>& attn_kv() const { return attn_kv_; }
  // The ENGINE-level attention backend selected PER full-attention KV GROUP
  // (resolved inside initialize_kv_cache's full-attn region via
  // vllm::v1::SelectAttentionBackendName — the same walk the registry test
  // covers), parallel to attn_kv(): one name per attention layer, in layer
  // order. Dense groups always resolve (a platform with no registered dense
  // backend throws loudly at init). An MLA group resolves TRITON_MLA where one
  // is registered (CUDA); on a device without a registered MLA backend (CPU,
  // ROCm today) the entry is EMPTY and that group's execution stays op-driven
  // (TritonMLAImpl on the fused cache — not registry-gated), which is why the
  // vector as a whole is empty only when no full-attention group exists at all
  // (a pure-GDN / pooling model caches no paged KV).
  const std::vector<std::string>& attn_backend_names() const {
    return attn_backend_names_;
  }
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
  // SPEC-MTP I5d acceptance telemetry accessors (the gate reads these).
  int64_t spec_drafts_proposed() const { return spec_drafts_proposed_; }
  int64_t spec_drafts_accepted() const { return spec_drafts_accepted_; }
  // SPEC-MTP-K-GT-1 (#81) PER-DEPTH acceptance. Index d counts the drafts
  // verified and accepted at draft depth d+1, so `[0]` is each request's first
  // drafted token and `[k-1]` its deepest. The aggregate counters above cannot
  // answer "how deep did acceptance actually reach", which is what #81's M1 asks
  // for and the only signal an acceptance-driven depth policy could use. The
  // vectors grow to the deepest draft the engine has verified.
  //
  // Their SIZE is NOT a witness that the propose loop ran k times, and an earlier
  // revision of this comment claimed it was. The size is grown from
  // `step.num_draft_tokens_per_req[i]`, which is the LENGTH of the emitted draft
  // list, a pure function of how `propose_drafts` slices whatever the proposer
  // returned. A propose that runs ONE forward and pads all k columns with its
  // step-0 draft satisfies it exactly, and the emitted tokens do not move either,
  // because greedy plus accept-iff-equal makes the sequence independent of k.
  // That mutation was applied and the whole depth suite stayed green on it. The
  // witness that DOES separate them is `spec_mtp_draft_decode_forwards()` below.
  const std::vector<int64_t>& spec_drafts_proposed_by_depth() const {
    return spec_drafts_proposed_by_depth_;
  }
  const std::vector<int64_t>& spec_drafts_accepted_by_depth() const {
    return spec_drafts_accepted_by_depth_;
  }
  // SPEC-MTP-K-GT-1 (#81): the WORK witness, counted where the work happens.
  //
  // `spec_mtp_propose_calls()` counts MTP propose invocations that reached the
  // proposer. `spec_mtp_draft_decode_forwards()` counts draft DECODE forwards
  // those calls actually executed, incremented after each forward returns.
  // Upstream runs `num_speculative_steps - 1` of them per propose
  // (`_multi_step_decode`, autoregressive/speculator.py:374-419 @ 555967922), so
  // the exact relation a caller asserts is
  //
  //     spec_mtp_draft_decode_forwards() == spec_mtp_propose_calls() * (k - 1)
  //
  // with both sides falling to 0 forwards at k=1. Nothing about the shape of the
  // emitted draft list can satisfy that equality, so it is the assertion a
  // propose that SHORT-CIRCUITS or CLAMPS the loop fails.
  //
  // It does NOT see padding, and an earlier revision of this comment said it
  // did. A propose that runs every forward, discards the results and writes the
  // step-0 draft into all k columns increments this counter exactly as the real
  // loop does. That mutation was applied and the whole depth suite stayed green
  // on it. `spec_mtp_proposals_with_varied_drafts()` below is the counter that
  // separates those two.
  int64_t spec_mtp_propose_calls() const { return spec_mtp_propose_calls_; }
  int64_t spec_mtp_draft_decode_forwards() const {
    return spec_mtp_draft_decode_forwards_;
  }
  // SPEC-MTP-K-GT-1 (#81): the RESULT witness, computed on the array the
  // proposer DELIVERED rather than inside the loop that filled it.
  //
  // It counts propose calls whose returned draft row held, for at least one
  // request, a column that differed from column 0. A propose that pads every
  // column with its step-0 draft leaves this at 0 at every k, because a padded
  // row is by construction a pure function of its own first column, and that is
  // the whole reason the counter exists. At k=1 it is 0 by definition, since a
  // one-column row has nothing to differ from.
  //
  // Its bound is stated here because it is easy to over-read. It says the
  // delivered array carries information the prefill draft alone does not
  // determine. It does NOT say that column j came from forward j, so an
  // off-by-one in the column index or a broken carry still satisfies it. It is
  // also a NECESSARY rather than a sufficient condition, and one the fixture
  // participates in: a drafter that resampled the same token on every step of
  // every call would leave it 0 while running the loop correctly. Measured on
  // the synthetic CPU gate model that does happen on individual calls (a
  // `2 2 2` row at k=3) and never on all of them, so the assertion the depth
  // suite makes is `> 0` over a run and never a per-call one. Per-column
  // provenance stays owed to the DGX gate on real weights, and NOT as a
  // non-zero acceptance count at depth, which a padded row also earns whenever
  // the target's greedy continuation repeats a token. It is owed as an
  // acceptance-RATE comparison against a padded control, specified in the
  // row spec under `## Owed`.
  int64_t spec_mtp_proposals_with_varied_drafts() const {
    return spec_mtp_proposals_with_varied_drafts_;
  }
  int full_attn_group_id() const { return full_attn_group_id_; }
  int gdn_group_id() const { return gdn_group_id_; }
  int64_t num_blocks() const { return num_blocks_; }
  // The per-block byte cost the attention-cache allocator ACTUALLY used, taken
  // from the KV SPEC (`spec->page_size_bytes()`, upstream
  // vllm/v1/kv_cache_interface.py:380-398) rather than reconstructed from the
  // HF config. 0 when the model has no full-attention group. This is the
  // positive signal that the spec-driven allocation path RAN — a compiled but
  // unexercised path leaves it 0, and a `page_size_padded` spec produces a
  // value the old HF-config arithmetic could not.
  int64_t fa_page_size_bytes() const { return fa_page_size_bytes_; }

  // #810: the per-layer KV class `initialize_kv_cache` RESOLVED, index == model
  // layer index, one entry per hidden layer. `kNone` is a layer that no KV
  // cache group named and that therefore caches nothing — NemotronH's 23
  // MoE blocks, which upstream's module walk yields no kv_cache_spec entry for
  // (`gpu_model_runner.py:7785-7801`). Recorded because the routing decision is
  // otherwise observable only as buffer COUNTS, and a count cannot see a
  // routing inversion: 3 recurrent + 1 attention has the same counts whichever
  // three layers got which.
  enum class LayerKvClass : uint8_t {
    kNone = 0,
    kFullAttention = 1,
    kRecurrent = 2,
  };
  const std::vector<LayerKvClass>& layer_kv_class() const {
    return layer_kv_class_;
  }

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
  // ENG-ASYNC-SCHED depth-2 lifetime guard, exposed for the regression test.
  // TRUE between a sample_tokens_async that left main-queue work outstanding and
  // the next execute_model that drains it (see async_forward_in_flight_).
  bool async_forward_in_flight() const { return async_forward_in_flight_; }
  bool kv_cache_backend_resident() const {
    return kv_cache_backend_resident_;
  }
  vt::Device device() const { return queue_.device; }

  // KV-EXTERNAL-CACHE (LMCache) worker-side seam. A non-owning pointer to the
  // SAME connector instance the scheduler holds (single-process engine); null =
  // no connector = ZERO behaviour change (the load/store paths are skipped, the
  // KV cache is byte-identical to production). Set by LoadedEngine right after
  // both runner_ and scheduler_ are built. Only the concrete LMCacheConnector
  // carries a worker-side store/load (dynamic_cast in runner.cpp); a base
  // connector with no worker transport is a no-op here.
  void set_kv_connector(kv_offload::KVConnector* connector) {
    kv_connector_ = connector;
  }
  kv_offload::KVConnector* kv_connector() const { return kv_connector_; }

  // SPEC-DFLASH D5 (DF-ENGINE-INTEGRATION): wire the separately-loaded z-lab
  // DFlash draft into the verify/propose loop. `weights`/`config` are borrows
  // owned by LoadedEngine (they outlive the runner); `k` = num_speculative_tokens
  // (block-1's block, the (1+k) query). Sets use_dflash(): on the verify forward
  // the runner captures the D1 multi-tap (target_layer_ids) instead of the MTP
  // single hidden tap, and propose_drafts routes to the DFlash block propose.
  // Idempotent; null weights leaves the runner on the MTP/non-spec path.
  void set_dflash_draft(const vllm::Qwen3DFlashWeights* weights,
                        const vllm::HfConfig* config, int k);

  // SPEC-DSPARK W5: wire the separately-loaded DSpark draft into the SAME
  // verify/propose loop. DSpark inherits DFlash's context accumulation and block
  // forward unchanged (Qwen3DSparkModel(DFlashQwen3Model)); it differs only in
  // the query-block layout (N rows with the anchor itself predicting, or the
  // DFlash 1+N fill-in when `sample_from_anchor` is false) and in sampling
  // (sequential Markov instead of one parallel argmax). Wiring it therefore
  // routes through the same aux-tap capture and the same device KV store —
  // set_dflash_draft is called internally with `&weights->backbone`, so
  // use_dflash() stays the predicate for the shared machinery and use_dspark()
  // only switches the propose tail. Idempotent; null leaves the runner alone.
  void set_dspark_draft(const vllm::Qwen3DSparkWeights* weights,
                        const vllm::HfConfig* config, int k,
                        bool sample_from_anchor);

  // SAMPLE-PROMPT-LOGPROBS route-observation seam. Const, no behaviour, exposed
  // for the gate: the row count the LAST forward actually produced, the token
  // count that step ran on, and its request count. On EVERY step where no
  // request asked for prompt logprobs the forward gathers before lm_head, so
  // last_forward_rows() == step_num_logits(); a step that owes prompt logits
  // instead takes the full-logits route and returns num_actual_tokens rows.
  // test_llm_engine §9(g) asserts that DECISION directly, because an on-vs-off
  // comparison inside ONE build cannot see a change to the shared route — both
  // arms move together (review finding 2 on PR #235).
  int64_t last_forward_rows() const { return exec_state_.logits.rows; }
  int last_forward_num_actual_tokens() const {
    return exec_state_.num_actual_tokens;
  }
  int last_forward_num_reqs() const { return exec_state_.num_reqs; }
  // The expanded logit-row count for the stashed step (StepInputs::cu_num_logits
  // back, == exec_state_.num_reqs on the non-speculative default path). Public
  // so the gate above can name the expected value instead of re-deriving it.
  int step_num_logits() const;

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
  // is type-erased but borrows the caller-owned concrete weights. Widened with
  // the same defaulted SPEC-MTP I5d-pre draft seam so the concrete-weight
  // constructors (which delegate here) stay byte-identical.
  GPUModelRunner(const HfConfig& config,
                 std::unique_ptr<LoadedModel> owned_model,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len, int max_num_batched_tokens,
                 std::optional<vllm::SpeculativeConfig> spec_config =
                     std::nullopt,
                 std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model = nullptr,
                 std::vector<PagedKvCache> draft_kv = {});

  // ARCH-ONE-SURFACE ROW 6: the pooling counterpart of sample_tokens (mirror
  // of gpu/model_runner.py:1586-1607 + pool/pooling_runner.py:29-42). Consumes
  // the stashed forward result — for the pooling arch those are the
  // [rows, hidden] post-final-norm hidden states, NOT vocab logits — applies
  // the model's Pooler via pooling_runner_, and returns a ModelRunnerOutput
  // whose pooler_output carries one pooled vector per fully-prefilled request
  // (nullopt for rows still consuming prefill chunks — the same
  // seq_len == prompt_len validity predicate as is_valid, pooling_runner.py:
  // 40-41, which our discard mask already computes). sampled_token_ids rows
  // stay EMPTY: a pooling step samples nothing.
  ModelRunnerOutput pool_tokens();

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
  // SPEC-MTP I5d-pre draft seam (all inert unless a SpeculativeConfig is set).
  // spec_config_ carries the resolved method/k; draft_model_ is the MTP draft the
  // runner owns for its lifetime (built via LoadedModel::BuildMtpDraft);
  // draft_attn_kv_ is the draft's paged KV layer (the `fa_draft` group). Stored
  // here so the I5d verify/propose loop can reach them; UNREAD in this increment
  // (nullopt / null / empty on every existing construction site).
  std::optional<vllm::SpeculativeConfig> spec_config_;
  std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model_;
  std::vector<PagedKvCache> draft_attn_kv_;
  vt::Queue queue_;
  // #1313 — the [rows, vocab] logits the ON-DEVICE sampler runs on are assembled
  // from a HOST buffer whenever the forward returned ForwardLogits.host
  // (nemotron_h, laguna, qwen3_vl). Those bytes must be reachable by the queue
  // device before any sampling kernel dereferences them: free on a unified
  // backend (GB10 wraps in place, unchanged), a grow-only H2D copy on a discrete
  // one, where the old host address was simply illegal.
  //
  // TWO buffers, not one: collect_prompt_logprobs stages the prompt rows while
  // the assembled sample-logits tensor is still live (sample_tokens calls it
  // between assemble_sample_logits and the sampler), so a shared buffer would
  // invalidate the tensor the sampler is about to read.
  HostBufferStaging sample_logits_staging_;
  HostBufferStaging prompt_logprobs_staging_;
  InputBatch input_batch_;
  Sampler sampler_;
  // ARCH-ONE-SURFACE ROW 6 (mirror of gpu/model_runner.py:368-369
  // `if self.is_pooling_model ...: self.pooling_runner = PoolingRunner(model)`):
  // non-null iff the loaded model's registration declares is_pooling_model and
  // the model owns a Pooler. sample_tokens then routes to pool_tokens() — the
  // POOLED DATA takes the place of sampled tokens (model_runner.py:1586-1607).
  // Null for every text arch: the sampler path below is byte-identical.
  std::unique_ptr<vllm::PoolingRunner> pooling_runner_;

  // KV group layout (resolved from the KVCacheConfig).
  int full_attn_group_id_ = -1;
  int gdn_group_id_ = -1;
  int64_t num_blocks_ = 0;
  // Per-block attention-cache bytes as reported by the KV spec (see the
  // fa_page_size_bytes() accessor).
  int64_t fa_page_size_bytes_ = 0;
  // #810: per-layer KV class, index == model layer index (layer_kv_class()).
  std::vector<LayerKvClass> layer_kv_class_;
  // Persistent-batch capacity = max concurrent sequences. The GDN mamba-state
  // cache is sized by this (one recurrent state per sequence), decoupled from
  // the attention num_blocks. See remap_gdn_state_slots.
  int max_num_reqs_ = 0;
  // Flattened-token bound for one step; sizes the W4 device input_ids mirror.
  int max_num_batched_tokens_ = 0;
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
  // Async SCHEDULING capability (see runner_supports_async): the same
  // env/backend predicate WITHOUT the spec veto. W7 (#1824).
  bool async_sched_supported_ = false;
  // Whether the engine resolved async scheduling ON (set_async_scheduling;
  // LoadedEngine calls it once after ResolveAsyncEnabled). Gates the
  // draft-placeholder fill + computed-token correction in execute_model.
  bool use_async_scheduling_ = false;
  // W7 (#1824): req_id -> the draft count the PREVIOUS step scheduled for it,
  // recorded at splice time under async scheduling. The computed-token
  // correction applies only to requests with an entry here (the scheduler's
  // num_computed_tokens can carry that step's not-yet-rolled-back rejected
  // drafts). Rebuilt every spec step; empty otherwise.
  std::map<std::string, int> prev_sched_draft_counts_;
  // ENG-ASYNC-SCHED depth-2 LIFETIME GUARD. sample_tokens_async DEFERS the main
  // queue's completion to the consuming step's get_output() (one step_with_batch_
  // queue call later), so when it returns the previous step's forward / sample /
  // scatter kernels are STILL IN FLIGHT and still reading exec_state_ (device
  // logits, the StepInputs host arrays) and writing input_batch_.last_sampled_
  // tokens. The next execute_model() would otherwise reset exec_state_ and mutate
  // input_batch_ (update_states condense/swap) WHILE those kernels run — a
  // use-after-free / host-heap corruption that only real GPU overlap exposes
  // (serialized runs — the CPU eager backend, compute-sanitizer — never see it).
  // Set true whenever sample_tokens_async leaves main-queue work outstanding;
  // execute_model() drains it before touching any shared state.
  bool async_forward_in_flight_ = false;
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

  // ─── ENG-ASYNC-SCHED W4: CUDA device-resident async inputs ──────────────────
  // last_sampled_tokens held in a persistent DEVICE buffer, the AUTHORITATIVE copy
  // (the scatter writes it on the main queue and the next step's combine reads it
  // there, so no sampled id crosses to the host on the critical path). Exactly
  // what upstream does on every platform — vllm/v1/worker/gpu/states.py:64 makes
  // last_sampled_tokens a GPU tensor unconditionally.
  //
  // Two reasons to engage it (async_device_mirror, gated VT_ASYNC_DEVICE_MIRROR):
  //  - DISCRETE GPU: REQUIRED. Its host arrays are not device-addressable, so the
  //    W3 host fallback has to synchronize the main stream to read the sampled ids,
  //    which makes the depth-2 async scheduler overlap nothing.
  //  - INTEGRATED GPU (GB10): the W3 kernels CAN operate on the runner's host
  //    arrays in place (pageable host memory is device-addressable), but then
  //    update_states' condense reorders that host array between the scatter that
  //    writes it and the combine that reads it — a read-after-write that pins the
  //    async drain to the TOP of execute_model. Moving last_sampled_tokens onto
  //    this device buffer removes that host RAW (condense records ops replayed
  //    on-queue instead), which lets the drain MOVE past the bulk host prep.
  //
  // Sized once from the batch bound; all zero-initialized. The per-step uploads
  // copy from the caller's pageable host buffers on purpose (see stage_upload).
  //
  // Sized once from the batch bound; all zero-initialized. The per-step uploads
  // copy from the caller's pageable host buffers on purpose (see stage_upload).
  struct AsyncDeviceInputs {
    int32_t* last_sampled = nullptr;     // [max_num_reqs], AUTHORITATIVE
    int32_t* prefill_len = nullptr;      // [max_num_reqs]
    int32_t* query_start_loc = nullptr;  // [max_num_reqs + 1]
    int32_t* seq_lens = nullptr;         // [max_num_reqs]
    int32_t* input_ids = nullptr;        // [max_num_batched_tokens]
    int32_t* ops = nullptr;              // [4 * max_num_reqs] structural replay
    int64_t input_ids_capacity = 0;      // elements in `input_ids`
    int32_t max_reqs = 0;
  };
  std::unique_ptr<AsyncDeviceInputs> async_device_inputs_;
  // Allocate on first use, or return nullptr when the mirror is not engaged
  // (non-CUDA backend, async not engaged, or VT_ASYNC_DEVICE_MIRROR=0). Caller
  // treats nullptr as "take the pre-W4 host-array path".
  AsyncDeviceInputs* get_or_create_async_device_inputs();
  // True when this runner mirrors the async inputs onto the device: CUDA, async
  // engaged, VT_ASYNC_DEVICE_MIRROR not "0", on a real GPU (integrated OR discrete;
  // not the CPU backend). Memoized. Default ON since the 2026-08-06 correctness
  // flip (ROW-SERVE-ASYNC-LLM P0); VT_ASYNC_DEVICE_MIRROR=0 is the rollback.
  bool async_device_mirror() const;
  mutable int async_device_mirror_cached_ = -1;  // -1 unknown, 0 no, 1 yes
  // VT_ASYNC_EXECUTOR (default OFF): the decode-graph slot double-buffer lever.
  // When on AND the previous step's stashed logits are a NON-owning decode-graph
  // slot view (non_owning_view), the depth-2 drain before the forward is skipped —
  // hazard-A (the eager owning-logits reset UAF) is absent for a slot view, and
  // hazard-C (the persistent decode-graph host inputs being overwritten while the
  // previous replay still reads them) is instead guarded inside the model by a
  // 2-slot parity ring + per-slot reuse event. OFF routes through today's single-
  // slot code with the drain intact (byte-identical). Memoized like the mirror.
  bool async_executor() const;
  mutable int async_executor_cached_ = -1;  // -1 unknown, 0 no, 1 yes
  // Push the recorded InputBatch structural edits (seed/move/swap) to the device
  // mirror in stream order, then clear the log. No-op without a mirror.
  void replay_last_sampled_ops(AsyncDeviceInputs& dev);
  // Upload `src` into `dst` through the pinned staging buffer on the main queue.
  void stage_upload(AsyncDeviceInputs& dev, int32_t* dst, const int32_t* src,
                    int64_t count);
  // Assemble the [num_reqs, vocab] logits the sampler runs on (the three-case
  // device/host gather from the stashed forward result) and apply the grammar
  // bitmask, IN the exact order the sync path uses. Shared by sample_tokens and
  // sample_tokens_async. `sampled_logits` is caller-owned scratch the returned
  // Tensor view may alias (host / VT_GPU_SAMPLE=0 paths); it must outlive the
  // sampler call. Requires exec_state_.num_reqs > 0.
  vt::Tensor assemble_sample_logits(
      const std::optional<GrammarOutput>& grammar_output,
      std::vector<float>& sampled_logits);
  // (step_num_logits is declared in the public section above — the
  // SAMPLE-PROMPT-LOGPROBS route gate names it.)
  // SAMPLE-PROMPT-LOGPROBS (gpu_model_runner.py:5612-5719, called at :3841).
  // Score the extra logit rows the forward produced for prompt positions, fold
  // each request's chunk into its accumulated tensor, and move out the tensors
  // whose prompt finished this step. Returns immediately — no branch taken, no
  // allocation — when the stashed step named no prompt rows, which is every
  // step unless a request asked for prompt logprobs.
  void collect_prompt_logprobs(
      std::map<std::string, LogprobsTensors>& prompt_logprobs_dict);
  // A request's prompt-logprob tensor height: num_prompt_tokens - 1.
  int prompt_logprob_positions(const std::string& req_id) const;
  // Forget in-progress prompt logprobs whose request left the batch (abort).
  void drop_stale_prompt_logprobs();
  // The SPEC-DECODE VERIFY half (SPEC-REJECTION I3): route the expanded
  // [Σ(1+k_i), vocab] logits through the greedy rejection sampler, write the
  // accepted tokens back, and record num_accepted_tokens. Called by sample_tokens
  // IFF exec_state_.step.num_draft_tokens > 0, which requires a configured
  // SpeculativeConfig — unreachable on the production default path. Mirrors
  // gpu/model_runner.py:1065-1077.
  ModelRunnerOutput sample_tokens_with_rejection(vt::Tensor& logits);

  // ── SPEC-MTP I5d verify/propose loop helpers ────────────────────────────────
  // Whether a speculator is configured (nullopt on the production default path,
  // so every helper below is unreachable and the runner is byte-identical).
  bool spec_on() const { return spec_config_.has_value(); }
  // k = resolved num_speculative_tokens (0 when spec off).
  int num_spec() const {
    return spec_config_.has_value()
               ? spec_config_->ResolvedNumSpeculativeTokens()
               : 0;
  }
  // Run the k=1 MTP propose after this step's sampling and stash the drafts for
  // take_draft_token_ids (gpu/model_runner.py:1455-1489). Uses the stashed target
  // hidden tap + verify attn metadata; `num_sampled`/`num_rejected` are the
  // per-req accept accounting (1/0 on a plain non-spec step). No-op unless spec_on.
  void propose_drafts(const std::vector<int32_t>& num_sampled,
                      const std::vector<int32_t>& num_rejected);
  // SPEC-DFLASH D5: the DFlash branch of propose_drafts. Combines this step's D1
  // multi-tap into per-token features (CombineAuxFeatures), ACCUMULATES the
  // accepted-prefix features into each request's growing combined-feature context
  // (rolling back the rejected drafts by appending only num_sampled features —
  // the num_rejected rollback, dflash/speculator.py:300-413), then runs the
  // non-autoregressive (1+k) block propose over that context (DflashProposeBlock)
  // and stashes the k drafts/request. Only reachable when use_dflash().
  void propose_drafts_dflash(const std::vector<int32_t>& num_sampled,
                             const std::vector<int32_t>& num_rejected);
  // The shared block-propose body of the DFlash and DSpark branches. Everything
  // through the context accumulation and the block forward is IDENTICAL for the
  // two (DSpark inherits it upstream); the two differ only in `num_query_per_req`
  // / `first_sample_offset` (the anchor-as-first-prediction layout) and in how the
  // resulting block logits become draft ids, which `sample` supplies. `anchors`
  // receives each proposing row's anchor token in the target vocab (DFlash
  // ignores it; DSpark seeds its sequential chain with it).
  void propose_drafts_block(
      const std::vector<int32_t>& num_rejected, const vllm::Qwen3DFlashWeights& backbone,
      const vllm::HfConfig& config, int num_query_per_req,
      const std::function<std::vector<std::vector<int32_t>>(
          const std::vector<float>& block_logits, int num_propose_rows,
          const std::vector<int32_t>& anchors)>& sample);
  // SPEC-DSPARK W5: the DSpark branch of propose_drafts — the shared body above
  // with the anchor-aware layout and the sequential Markov sampler
  // (SampleDsparkBlockDrafts). Only reachable when use_dspark().
  void propose_drafts_dspark(const std::vector<int32_t>& num_sampled,
                             const std::vector<int32_t>& num_rejected);
  // SPEC-NGRAM (ROAD-V1-D3): the draft-FREE branch of propose_drafts. Runs the
  // host-side n-gram matcher (v1/spec_decode/ngram_proposer) over each generating
  // request's own committed context (input_batch_.token_ids_cpu[i,
  // :num_tokens_no_spec[i]]) and stashes the variable-length (0..k) drafts. No
  // draft model, no hidden tap, no draft KV. Only reachable when use_ngram().
  void propose_drafts_ngram(const std::vector<int32_t>& num_sampled,
                            const std::vector<int32_t>& num_rejected);
  // method=="ngram": the draft-free proposer. Distinct from spec_on() (which is
  // any speculator) and use_dflash(); gates the propose routing + suppresses the
  // (unused) hidden-tap capture on the verify forward.
  bool use_ngram() const {
    return spec_config_.has_value() && spec_config_->use_ngram();
  }
  // The drafts produced this step, pending pull by EngineCore::post_step. Empty
  // (nullopt) on the default path.
  std::optional<DraftTokenIds> pending_drafts_;
  // SPEC-MTP I5d acceptance telemetry (spec §5 gate: measured nonzero acceptance).
  // spec_drafts_proposed_ counts draft tokens VERIFIED, spec_drafts_accepted_ the
  // subset the rejection sampler accepted. accepted/proposed is the acceptance
  // rate; total generated / total verify steps is the effective speedup proxy.
  int64_t spec_drafts_proposed_ = 0;
  int64_t spec_drafts_accepted_ = 0;
  // SPEC-MTP-K-GT-1 (#81): the same two counts, split by draft depth. Grown on
  // demand to the deepest draft verified, so they stay EMPTY on the default
  // no-speculation path.
  std::vector<int64_t> spec_drafts_proposed_by_depth_;
  std::vector<int64_t> spec_drafts_accepted_by_depth_;
  // SPEC-MTP-K-GT-1 (#81): the propose-side depth witnesses. All three stay 0
  // unless the MTP proposer runs. The first two carry the RATIO k-1, which the
  // draft list shape cannot produce. The third reads the delivered array, which
  // is the only place padding is visible. See the accessors above for why the
  // per-depth vectors serve neither purpose.
  int64_t spec_mtp_propose_calls_ = 0;
  int64_t spec_mtp_draft_decode_forwards_ = 0;
  int64_t spec_mtp_proposals_with_varied_drafts_ = 0;
  // ── SPEC-DFLASH D5 (DF-ENGINE-INTEGRATION) ──────────────────────────────────
  // The separately-loaded DFlash draft (borrows owned by LoadedEngine; null
  // unless method=="dflash"). use_dflash() gates the aux-tap capture + the DFlash
  // propose branch. dflash_tap_layer_ids_ = the draft's target_layer_ids (the D1
  // multi-tap capture-after indices), resolved once from the draft config.
  const vllm::Qwen3DFlashWeights* dflash_weights_ = nullptr;
  const vllm::HfConfig* dflash_config_ = nullptr;
  int dflash_k_ = 0;
  std::vector<int32_t> dflash_tap_layer_ids_;
  bool use_dflash() const { return dflash_weights_ != nullptr; }
  // ── SPEC-DSPARK W5 ──────────────────────────────────────────────────────────
  // The separately-loaded DSpark draft (borrow owned by LoadedEngine; null unless
  // method=="dspark"). When set, dflash_weights_ points at `&dspark_weights_->
  // backbone`, so every piece of shared machinery — the aux multi-tap capture,
  // the per-request device KV store, the context-aware block forward — runs
  // UNCHANGED, and use_dspark() only redirects the propose tail to the sequential
  // Markov sampler and the anchor-aware block layout.
  const vllm::Qwen3DSparkWeights* dspark_weights_ = nullptr;
  bool dspark_sample_from_anchor_ = true;
  bool use_dspark() const { return dspark_weights_ != nullptr; }
  // Per-request PERSISTENT context KV store (D9 persistent paged draft-KV — the
  // perf form of vLLM's incrementally-written draft KV cache). dflash_kv_store_[i]
  // holds request i's per-layer bf16 context K/V (K normed+RoPE'd, V raw) for its
  // committed positions 0..L_i-1 (L_i = dflash_ctx_len_[i]). Each verify step
  // projects ONLY the newly-accepted rows (AppendContextKVHost) and APPENDS them,
  // instead of re-projecting the whole growing context (the D5/D7 O(context^2)
  // recompute). Bit-identical to the recompute by per-row projection independence.
  // dflash_ctx_reqid_[i] tracks the occupant so a reused batch slot resets its
  // store; rejected drafts' rows are never appended (rollback = don't-append).
  // Indexed by the runner's condensed-dense batch row. Sized on set_dflash_draft.
  // D11 A-wire: the store is now the DEVICE-RESIDENT append-only draft-KV store
  // (DflashDeviceKVStore, opaque, one shared_ptr per condensed-dense batch row).
  // AppendContextKVDevice keeps the projected bf16 K/V on-device (no D<->H round
  // trip) and ForwardBlockLogitsWithDeviceKV runs the block forward straight off
  // the device store — bit-identical to the D9 host path, and the capture-ready
  // substrate for Parts B/C. shared_ptr-to-incomplete is safe: MakeDeviceKVStore
  // constructs the control block (with its deleter) in qwen3_dflash.cpp.
  std::vector<std::shared_ptr<vllm::DflashDeviceKVStore>> dflash_kv_store_;
  std::vector<int32_t> dflash_ctx_len_;
  std::vector<std::string> dflash_ctx_reqid_;
  // Draft KV cache (`fa_draft` group) backing storage, owned by the runner and
  // allocated in initialize_kv_cache when spec is on. draft_attn_kv_ (declared
  // above) views into these buffers. Empty on the default path.
  std::vector<std::unique_ptr<CacheBuffer>> draft_attn_buf_;
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
  // Per-layer attention backend names, parallel to attn_kv_ (see accessor).
  // A dense entry is never empty; an MLA entry may be (op-driven execution).
  std::vector<std::string> attn_backend_names_;

  // ── KV-EXTERNAL-CACHE (LMCache) worker-side store/load ──────────────────────
  // Non-owning; null (default) = inert. See set_kv_connector.
  kv_offload::KVConnector* kv_connector_ = nullptr;
  // Per-request count of prompt blocks already STORED to the external cache, so
  // a multi-step (chunked) prefill stores each full block exactly once and a
  // decode step re-stores nothing. Only touched when kv_connector_ != nullptr.
  std::unordered_map<std::string, int> connector_stored_blocks_;
  // Drain + apply the connector's recorded external-prefix loads into the KV
  // cache blocks BEFORE the forward reads them (load-before-compute). No-op
  // unless kv_connector_ is a worker-capable connector with pending loads.
  void ConnectorLoadExternalKv();
  // After the forward has written this step's KV, STORE every newly-complete
  // prompt block of each running request to the external cache (offload-prompt-
  // only). No-op unless kv_connector_ is a worker-capable connector.
  void ConnectorStorePromptKv(const SchedulerOutput& scheduler_output);

  // SAMPLE-PROMPT-LOGPROBS: the partially-filled prompt-logprob tensor of every
  // request whose prompt is still being consumed, keyed by req_id. A chunked
  // prefill fills it slice by slice and the final chunk moves it out; upstream
  // hangs the same tensor off the per-request state object as
  // `request.in_progress_prompt_logprobs_cpu` (gpu_model_runner.py:5645-5651,
  // cleared at :5712), which we have no equivalent of on the runner. Empty
  // unless a request asked for prompt logprobs.
  std::map<std::string, LogprobsTensors> in_progress_prompt_logprobs_;

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
    // SPEC-MTP I5d: the target's post-final-norm [T,H] hidden tap captured this
    // step (ModelForwardInput::hidden_tap output), consumed by propose_drafts to
    // run the MTP drafter. Empty (null storage) unless spec is on.
    Qwen3_5MTPHiddenStates spec_hidden;
    // SPEC-DFLASH D5: the target's D1 MULTI-tap captured this step
    // (ModelForwardInput::aux_tap output) — the residual stream at the draft's
    // target_layer_ids as [T, H×taps] bf16, consumed by propose_drafts_dflash.
    // Empty (null storage) unless use_dflash(); mutually exclusive with
    // spec_hidden (MTP single tap).
    Qwen3_5AuxTaps spec_aux;
    // discard_request_mask (gpu_model_runner.py:2048): per dense batch row, 1 iff
    // the request is still consuming its known prefill tokens this step
    // (optimistic seq_len < num_tokens) and so must NOT sample — its sampled
    // token is cleared to empty before returning.
    std::vector<uint8_t> discard;
  } exec_state_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_RUNNER_H_
