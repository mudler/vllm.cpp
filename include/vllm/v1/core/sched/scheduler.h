// Ported from: vllm/v1/core/sched/scheduler.py @ e24d1b24
//
// Scope (M1.4 Task 3): the V1 Scheduler's core token-budget algorithm — the
// unified schedule() loop the EngineCore step drives. There is no "prefill" vs
// "decode" distinction: each Request tracks num_computed_tokens vs num_tokens,
// and each step assigns tokens under a per-step budget so num_computed catches
// up to num_tokens. This covers chunked prefill, prefix caching and (deferred)
// speculative / jump decoding uniformly.
//
// Ported T0 behavior (matched line-by-line with upstream schedule()):
//   * running-first loop: num_new_tokens = num_tokens_with_spec +
//     num_output_placeholders - num_computed_tokens, clamped by the
//     long_prefill threshold, the token budget, and (max_model_len -
//     num_computed - num_sampled_tokens_per_step); allocate_slots -> nullopt
//     triggers FCFS-tail preemption (_preempt_request) retried until the
//     request fits or it is itself preempted.
//   * waiting loop (only when no preemption happened this step, and under both
//     the token budget and max_num_running_reqs): get_computed_blocks prefix
//     hit, chunked-prefill split via min(num_new_tokens, token_budget),
//     allocate_slots with full_sequence_must_fit = scheduler_reserve_full_isl,
//     admit as scheduled_new (WAITING) or scheduled_resumed (PREEMPTED).
//   * SchedulerOutput assembly: NewRequestData for first-time reqs +
//     CachedRequestData diff (_make_cached_request_data) for already-running /
//     resumed reqs, the num_scheduled_tokens map + total, num_common_prefix_
//     blocks, finished_req_ids; then _update_after_schedule advances
//     num_computed_tokens (so the next step continues a chunked prefill) and
//     flips is_prefill_chunk.
//   * add_request / finish_requests / get_num_unfinished_requests /
//     get_request_counts / has_finished_requests (T0 subset).
//
// DEFERRED (marked; matches upstream structure so re-adding is mechanical):
//   - Encoder inputs (_try_schedule_encoder_inputs), the encoder cache manager,
//     encoder compute budget.
//   - Speculative-decode tokens (spec_token_ids, num_lookahead_tokens,
//     scheduled_spec_decode_tokens, spec pad, dynamic SD lookup) — T0 keeps
//     num_lookahead_tokens = 0 and num_tokens_with_spec == num_tokens.
//   - num_output_placeholders / async scheduling (the early-continue on
//     max_tokens, next_decode_eligible_step) — treated as 0 / inert.
//   - DP prefill balancing (throttle_prefills / defer_prefills /
//     prefill_capacity_bound), PAUSE state, streaming/resumable sessions,
//     skipped_waiting, priority scheduling (SchedulingPolicy::kPriority throws).
//   - Structured-output grammar, LoRA gating, KV-connector / EC-connector,
//     mamba block-aligned split, the V2 model runner merge (resumed reqs folded
//     into new_reqs + prefill_token_ids — NewRequestData has no prefill_token_ids
//     field at T0), defer_block_free / deferred_frees, kv-cache metrics, MFU
//     perf metrics, log-stats events, take_events / kv_cache_events.
//   The T0 path follows use_v2_model_runner = false (MRV1), the create_scheduler
//   default: resumed reqs stay separate and flow through the cached diff.
#ifndef VLLM_V1_CORE_SCHED_SCHEDULER_H_
#define VLLM_V1_CORE_SCHED_SCHEDULER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/v1/core/kv_cache_manager.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/request_queue.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"

namespace vllm::v1 {

// The V1 Scheduler (T0 subset). Owns the Request lifetime (a map of owning
// unique_ptr); running_ and waiting_ hold non-owning raw pointers into it,
// exactly as upstream where the running list and waiting queue alias the same
// Request instances stored in self.requests.
class Scheduler {
 public:
  // Build the scheduler + its KVCacheManager. Mirrors the fields the upstream
  // ctor reads out of vllm_config for the T0 path: the scheduler_config knobs
  // (max_num_seqs, max_num_batched_tokens/max_num_scheduled_tokens,
  // long_prefill_token_threshold, enable_chunked_prefill, policy, watermark,
  // scheduler_reserve_full_isl, max_model_len) + the kv_cache_config, block
  // size, and the prefix-caching flag (upstream cache_config.enable_prefix_
  // caching). enable_caching is passed straight to the KVCacheManager.
  Scheduler(SchedulerConfig scheduler_config, KVCacheConfig kv_cache_config,
            int block_size, bool enable_caching = false);

  // add_request: enqueue a new request into waiting_ and register it in the
  // requests_ map (the scheduler takes ownership). (Upstream add_request T0
  // branch — no streaming/resumable/connector.)
  void add_request(std::unique_ptr<Request> request);

  // finish_requests: handle an external finish/abort signal for one request.
  // Removes it from running_/waiting_, sets the finished status, records it in
  // finished_req_ids_, frees its KV blocks, and erases it from requests_
  // (freeing the Request). No-op for an unknown or already-finished id.
  // (Upstream finish_requests + _free_request, T0 subset.)
  void finish_requests(const std::string& request_id,
                       RequestStatus finished_status);

  // schedule(): the core token-budget algorithm. See the file header.
  SchedulerOutput schedule();

  // get_num_unfinished_requests: len(waiting) + len(running) (T0 subset).
  int get_num_unfinished_requests() const;
  // get_request_counts: (num_running, num_waiting).
  std::pair<int, int> get_request_counts() const;
  // has_finished_requests: whether any request finished since the last step.
  bool has_finished_requests() const;

  // --- Public state the ported tests inspect directly (mirrors upstream's
  // accessible attributes). --------------------------------------------------

  // req_id -> owning Request. The scheduler owns Request lifetime.
  std::map<std::string, std::unique_ptr<Request>> requests;
  // Running requests, in scheduling order (non-owning; the FCFS tail is
  // running.back(), the first preemption victim).
  std::vector<Request*> running;
  // The waiting (FCFS) queue (non-owning pointers).
  std::unique_ptr<RequestQueue> waiting;
  // Request ids finished between the previous and current step (flushed each
  // step by _update_after_schedule).
  std::set<std::string> finished_req_ids;
  // Ids preempted since the last schedule() (flushed each step).
  std::set<std::string> reset_preempted_req_ids;

  // Resolved scheduling constraints (upstream ctor).
  int max_num_running_reqs;
  int max_num_scheduled_tokens;
  int max_model_len;

  // The KV cache manager the scheduler drives (heap-owned; non-movable).
  std::unique_ptr<KVCacheManager> kv_cache_manager;

 private:
  // _preempt_request: free the request's KV, mark it PREEMPTED, reset its
  // computed tokens, and re-queue it to the FRONT of waiting (FCFS retry). The
  // request must already have been popped from running by the caller.
  void preempt_request(Request* request);

  // _make_cached_request_data: build the diff payload for the already-running
  // (running_reqs) + resumed-from-preemption (resumed_reqs) requests. T0 /
  // MRV1: use_pp is false so new_token_ids stays empty; all_token_ids is
  // populated for reqs not scheduled last step.
  CachedRequestData make_cached_request_data(
      const std::vector<Request*>& running_reqs,
      const std::vector<Request*>& resumed_reqs,
      const std::map<std::string, int>& num_scheduled_tokens,
      const std::map<std::string, KVCacheBlocks>& req_to_new_blocks);

  // _update_after_schedule: advance num_computed_tokens for every scheduled
  // request, refresh is_prefill_chunk, and flush finished/preempted id sets.
  void update_after_schedule(const SchedulerOutput& scheduler_output);

  SchedulerConfig scheduler_config_;
  KVCacheConfig kv_cache_config_;

  int long_prefill_token_threshold_;
  bool enable_chunked_prefill_;
  bool scheduler_reserve_full_isl_;
  // T0: 0 (no speculative decode / eagle).
  int num_lookahead_tokens_ = 0;
  // T0: 1 (not a diffusion model).
  int num_sampled_tokens_per_step_ = 1;
  // Scheduler iteration counter (upstream current_step; inert at T0 since
  // next_decode_eligible_step is deferred/0).
  int current_step_ = 0;
  // MRV1: request ids scheduled in the prior step (gates the all_token_ids
  // payload in _make_cached_request_data).
  std::set<std::string> prev_step_scheduled_req_ids_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_SCHED_SCHEDULER_H_
