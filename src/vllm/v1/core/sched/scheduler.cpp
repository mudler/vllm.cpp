// Ported from: vllm/v1/core/sched/scheduler.py @ e24d1b24
// See include/vllm/v1/core/sched/scheduler.h for the T0 scope + deferred list.
#include "vllm/v1/core/sched/scheduler.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/core/sched/utils.h"           // check_stop
#include "vllm/v1/kv_offload/kv_connector.h"    // KVConnectorScheduler (W4)
#include "vllm/v1/structured_output/manager.h"  // StructuredOutputManager

namespace vllm::v1 {

namespace {

// Map the config-level policy onto the request-queue policy. Priority is
// deferred (T1); create_request_queue(kPriority) throws.
SchedulingPolicy ToQueuePolicy(SchedulerPolicy policy) {
  switch (policy) {
    case SchedulerPolicy::kFCFS:
      return SchedulingPolicy::kFCFS;
    case SchedulerPolicy::kPriority:
      return SchedulingPolicy::kPriority;
  }
  return SchedulingPolicy::kFCFS;
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig scheduler_config,
                     KVCacheConfig kv_cache_config, int block_size,
                     bool enable_caching,
                     StructuredOutputManager* structured_output_manager)
    : max_num_running_reqs(scheduler_config.max_num_seqs),
      max_num_scheduled_tokens(
          scheduler_config.ResolvedMaxNumScheduledTokens()),
      max_model_len(scheduler_config.max_model_len),
      structured_output_manager_(structured_output_manager),
      scheduler_config_(scheduler_config),
      kv_cache_config_(kv_cache_config),
      long_prefill_token_threshold_(
          scheduler_config.long_prefill_token_threshold),
      enable_chunked_prefill_(scheduler_config.enable_chunked_prefill),
      scheduler_reserve_full_isl_(
          scheduler_config.scheduler_reserve_full_isl) {
  // Scheduling policy -> the waiting (FCFS) queue.
  waiting = create_request_queue(ToQueuePolicy(scheduler_config.policy));

  // Build the KV cache manager (upstream scheduler.py ctor). hash_block_size
  // defaults to block_size; use_eagle / kv_cache_events off and dcp/pcp world
  // sizes 1 at T0.
  //
  // log_stats is ON: upstream's `disable_log_stats` defaults False, and the
  // benchmark protocol VOIDS any caching arm that cannot report queries/hits.
  // Cost is three integer adds per admitted request.
  kv_cache_manager = std::make_unique<KVCacheManager>(
      kv_cache_config_, max_model_len, /*scheduler_block_size=*/block_size,
      /*hash_block_size=*/block_size,
      /*max_num_batched_tokens=*/scheduler_config.max_num_batched_tokens,
      enable_caching, /*use_eagle=*/false, /*log_stats=*/true,
      /*enable_kv_cache_events=*/false, /*dcp_world_size=*/1,
      /*pcp_world_size=*/1, scheduler_config.watermark);
}

void Scheduler::add_request(std::unique_ptr<Request> request) {
  const std::string request_id = request->request_id;
  waiting->add_request(request.get());
  requests[request_id] = std::move(request);
}

void Scheduler::finish_requests(const std::string& request_id,
                                RequestStatus finished_status) {
  assert(vllm::v1::IsFinished(finished_status) &&
         "finish_requests requires a finished status");
  auto it = requests.find(request_id);
  if (it == requests.end() || it->second->IsFinished()) {
    // Invalid / already-finished request id.
    return;
  }
  Request* request = it->second.get();

  // Remove from the queue it currently lives in.
  if (request->status == RequestStatus::kRunning) {
    running.erase(std::remove(running.begin(), running.end(), request),
                  running.end());
  } else {
    waiting->remove_request(request);
  }

  // _free_request (T0 subset): mark finished, record the id, free the KV blocks,
  // and erase the owning entry (which destroys the Request).
  request->status = finished_status;
  finished_req_ids.insert(request_id);
  kv_cache_manager->free(*request);
  requests.erase(it);
}

void Scheduler::preempt_request(Request* request) {
  assert(request->status == RequestStatus::kRunning &&
         "Only running requests can be preempted");
  // _free_request_blocks (T0: defer_block_free off -> free immediately).
  kv_cache_manager->free(*request);
  request->status = RequestStatus::kPreempted;
  request->num_computed_tokens = 0;
  // Upstream Request.num_preemptions (read by PrefixCacheStats.record at
  // vllm/v1/core/kv_cache_manager.py:239).
  request->num_preemptions += 1;
  // Put the request back to the FRONT of the waiting queue (FCFS retry).
  waiting->prepend_request(request);
  reset_preempted_req_ids.insert(request->request_id);
}

SchedulerOutput Scheduler::schedule() {
  current_step_ += 1;
  // NOTE(woosuk): there is no "prefill" nor "decode" phase — each request just
  // has num_computed_tokens and num_tokens, and each step assigns tokens so the
  // former catches up to the latter (covers chunked prefill uniformly).

  std::vector<Request*> scheduled_new_reqs;
  std::vector<Request*> scheduled_resumed_reqs;
  std::vector<Request*> scheduled_running_reqs;
  std::vector<Request*> preempted_reqs;

  std::map<std::string, KVCacheBlocks> req_to_new_blocks;
  std::map<std::string, int> num_scheduled_tokens;
  int token_budget = max_num_scheduled_tokens;

  kv_cache_manager->new_step_starts();

  // Whether the priority policy is in effect (else FCFS). Chosen once per step
  // (mirrors scheduler.py reading self.policy inside schedule()).
  const bool priority_policy =
      scheduler_config_.policy == SchedulerPolicy::kPriority;

  // First, schedule the RUNNING requests. req_index is a signed int so the
  // priority-preemption `req_index -= 1` fix-up (upstream scheduler.py:570) is
  // well-defined.
  int req_index = 0;
  while (req_index < static_cast<int>(running.size()) && token_budget > 0) {
    Request* request = running[req_index];

    // Async scheduling (scheduler.py:445-459): avoid scheduling an extra step
    // once we are sure the previous step reached max_tokens. `+2` is
    // (num_computed + 1) - (num_output_placeholders - 1) — placeholders are in
    // the computed count, so we subtract (placeholders - 1) to drop draft
    // tokens. INERT while num_output_placeholders == 0 (synchronous Scheduler).
    if (request->num_output_placeholders > 0 &&
        request->num_computed_tokens + 2 - request->num_output_placeholders >=
            request->num_prompt_tokens + request->MaxTokens()) {
      req_index += 1;
      continue;
    }
    // V2+PP+async decode cadence (scheduler.py:461-465): enforce pp_size steps
    // between same-req decodes. INERT while next_decode_eligible_step == 0.
    if (current_step_ < request->next_decode_eligible_step) {
      req_index += 1;
      continue;
    }

    // num_tokens_with_spec == num_tokens at T0 (no spec tokens). Under async
    // scheduling num_output_placeholders reserves the in-flight sampled token(s)
    // (0 for the synchronous Scheduler, so this is unchanged there).
    int num_new_tokens = request->NumTokens() + request->num_output_placeholders -
                         request->num_computed_tokens;
    if (0 < long_prefill_token_threshold_ &&
        long_prefill_token_threshold_ < num_new_tokens) {
      num_new_tokens = long_prefill_token_threshold_;
    }
    num_new_tokens = std::min(num_new_tokens, token_budget);
    // Do not let the input position exceed the max model len.
    num_new_tokens =
        std::min(num_new_tokens, max_model_len - request->num_computed_tokens -
                                     num_sampled_tokens_per_step_);

    if (num_new_tokens == 0) {
      // Nothing to schedule for this request (e.g. it has reached its cap).
      // NOTE(woosuk): `continue` (not `break`) — do not strictly follow FCFS,
      // let lower-priority requests be scheduled.
      req_index += 1;
      continue;
    }

    // Schedule the KV blocks, preempting on OOM until the request fits.
    std::optional<KVCacheBlocks> new_blocks;
    while (true) {
      new_blocks = kv_cache_manager->allocate_slots(
          *request, num_new_tokens, /*num_new_computed_tokens=*/0,
          /*new_computed_blocks=*/std::nullopt,
          /*num_lookahead_tokens=*/num_lookahead_tokens_);
      if (new_blocks.has_value()) {
        break;  // The request can be scheduled.
      }

      // The request cannot be scheduled — preempt one running request.
      // (scheduler.py:546-572.)
      Request* preempted_req = nullptr;
      if (priority_policy) {
        // Preempt the lowest-priority running request: the max of running by
        // (priority, arrival_time). std::max_element returns the FIRST maximum,
        // matching Python max()'s first-encountered tie behavior.
        auto victim = std::max_element(
            running.begin(), running.end(), [](Request* a, Request* b) {
              return std::make_pair(a->priority, a->arrival_time) <
                     std::make_pair(b->priority, b->arrival_time);
            });
        preempted_req = *victim;
        running.erase(victim);
        // If the victim was already scheduled earlier this step, undo it:
        // restore its token budget, drop its block reservation, and step
        // req_index back one (running shrank in front of the cursor).
        auto sit = std::find(scheduled_running_reqs.begin(),
                             scheduled_running_reqs.end(), preempted_req);
        if (sit != scheduled_running_reqs.end()) {
          const std::string& preempted_id = preempted_req->request_id;
          scheduled_running_reqs.erase(sit);
          token_budget += num_scheduled_tokens[preempted_id];
          num_scheduled_tokens.erase(preempted_id);
          req_to_new_blocks.erase(preempted_id);
          // (spec-decode / encoder budgets are deferred at T0.)
          req_index -= 1;
        }
      } else {
        // FCFS: preempt the tail (lowest scheduling priority = last arrival).
        preempted_req = running.back();
        running.pop_back();
      }

      preempt_request(preempted_req);
      preempted_reqs.push_back(preempted_req);
      if (preempted_req == request) {
        // No more requests to preempt; cannot schedule this request.
        break;
      }
    }

    if (!new_blocks.has_value()) {
      // Cannot schedule this request.
      break;
    }

    // Schedule the request.
    scheduled_running_reqs.push_back(request);
    const std::string request_id = request->request_id;
    req_to_new_blocks[request_id] = *new_blocks;
    num_scheduled_tokens[request_id] = num_new_tokens;
    token_budget -= num_new_tokens;
    req_index += 1;
  }

  // Next, schedule the WAITING requests (skipped entirely if any preemption
  // happened this step, matching upstream).
  // KV-OFFLOAD W4: requests deferred by the connector's "not ready, re-ask"
  // (nullopt) third state. Popped off waiting this step, re-queued to the front
  // after the loop so they are re-asked next step (scheduler.py:744-750,1017).
  std::vector<Request*> connector_skipped_waiting;
  if (preempted_reqs.empty()) {
    while (!waiting->empty() && token_budget > 0) {
      if (static_cast<int>(running.size()) >= max_num_running_reqs) {
        break;
      }

      Request* request = waiting->peek_request();
      const std::string request_id = request->request_id;

      // Get already-cached (prefix) tokens.
      KVCacheBlocks new_computed_blocks = kv_cache_manager->empty_kv_cache_blocks;
      int num_new_local_computed_tokens = 0;
      int num_computed_tokens = 0;
      if (request->num_computed_tokens == 0) {
        auto computed = kv_cache_manager->get_computed_blocks(*request);
        new_computed_blocks = computed.first;
        num_new_local_computed_tokens = computed.second;
        num_computed_tokens = num_new_local_computed_tokens;
      } else {
        // Resumed reqs with num_computed_tokens > 0 come only from the deferred
        // KV-transfer path; at T0 preempted reqs are reset to 0, so this branch
        // is inert (kept for structural fidelity).
        num_computed_tokens = request->num_computed_tokens;
      }

      // KV-OFFLOAD W4: ask the connector how many EXTERNAL (tier-cached) tokens
      // load beyond the local prefix, so a cross-request / restarted-process
      // prefix HIT shortcuts prefill (scheduler.py:737-762). Null connector =
      // zero change. The nullopt THIRD state ("a disk->CPU promotion is still in
      // flight") pops the request and re-asks next step; treating it as 0 would
      // spin or serve a partial prefix (§Risks R5).
      int num_external_computed_tokens = 0;
      if (kv_connector_ != nullptr && request->num_computed_tokens == 0) {
        const auto match = kv_connector_->get_num_new_matched_tokens(
            *request, num_new_local_computed_tokens);
        if (!match.num_matched_tokens.has_value()) {
          waiting->pop_request();
          connector_skipped_waiting.push_back(request);
          continue;
        }
        num_external_computed_tokens = *match.num_matched_tokens;
        // The external tokens are treated as computed (their KV is loaded before
        // compute by the worker), so prefill shrinks by exactly this many.
        num_computed_tokens += num_external_computed_tokens;
      }

      // Number of tokens to schedule (num_tokens covers resumed reqs' output).
      int num_new_tokens = request->NumTokens() - num_computed_tokens;
      if (0 < long_prefill_token_threshold_ &&
          long_prefill_token_threshold_ < num_new_tokens) {
        num_new_tokens = long_prefill_token_threshold_;
      }
      // With chunked prefill disabled, we cannot split -> stop scheduling here.
      if (!enable_chunked_prefill_ && num_new_tokens > token_budget) {
        break;
      }
      num_new_tokens = std::min(num_new_tokens, token_budget);
      assert(num_new_tokens > 0);

      std::optional<KVCacheBlocks> new_blocks = kv_cache_manager->allocate_slots(
          *request, num_new_tokens, num_new_local_computed_tokens,
          new_computed_blocks, /*num_lookahead_tokens=*/num_lookahead_tokens_,
          num_external_computed_tokens, /*delay_cache_blocks=*/false,
          /*num_encoder_tokens=*/0,
          /*full_sequence_must_fit=*/scheduler_reserve_full_isl_,
          /*reserved_blocks=*/0, /*has_scheduled_reqs=*/!running.empty());
      if (!new_blocks.has_value()) {
        // The request cannot be scheduled.
        break;
      }

      request = waiting->pop_request();
      running.push_back(request);
      if (request->status == RequestStatus::kWaiting) {
        scheduled_new_reqs.push_back(request);
      } else if (request->status == RequestStatus::kPreempted) {
        scheduled_resumed_reqs.push_back(request);
      } else {
        throw std::runtime_error("Invalid request status in waiting loop");
      }

      req_to_new_blocks[request_id] = kv_cache_manager->get_blocks(request_id);
      // KV-OFFLOAD W4: register the load of the external prefix into the blocks
      // just allocated (load-before-compute). ext==0 makes this a no-op
      // (scheduler.py:932-937). The worker executes the load before those tokens
      // are read as computed.
      if (kv_connector_ != nullptr && num_external_computed_tokens > 0) {
        kv_connector_->update_state_after_alloc(
            *request, req_to_new_blocks[request_id].get_block_ids(),
            num_external_computed_tokens);
      }
      num_scheduled_tokens[request_id] = num_new_tokens;
      token_budget -= num_new_tokens;
      request->status = RequestStatus::kRunning;
      request->num_computed_tokens = num_computed_tokens;
    }
    // KV-OFFLOAD W4: re-queue the connector-deferred requests to the FRONT of
    // waiting (reverse so FCFS order is preserved) to be re-asked next step.
    for (auto it = connector_skipped_waiting.rbegin();
         it != connector_skipped_waiting.rend(); ++it) {
      waiting->prepend_request(*it);
    }
  }

  // Check the scheduling constraints are satisfied.
  int total_num_scheduled_tokens = 0;
  for (const auto& [id, n] : num_scheduled_tokens) {
    total_num_scheduled_tokens += n;
  }
  assert(total_num_scheduled_tokens <= max_num_scheduled_tokens);
  assert(token_budget >= 0);
  assert(static_cast<int>(running.size()) <= max_num_running_reqs);

  // Longest common prefix among running requests (for cascade attention).
  std::vector<int> num_common_prefix_blocks(kv_cache_config_.kv_cache_groups.size(),
                                            0);
  if (!running.empty()) {
    num_common_prefix_blocks =
        kv_cache_manager->get_num_common_prefix_blocks(running[0]->request_id);
  }

  // Construct the scheduler output (MRV2 / V2 model-runner path — upstream
  // scheduler.py:1047-1080 `if self.use_v2_model_runner:` branch). Resumed
  // (PREEMPTED->RUNNING this step) requests fold into scheduled_new_reqs and
  // are re-sent as FULL new reqs carrying prefill_token_ids, NOT through the
  // cached diff: the V2 gpu runner (M1.5) asserts every new req has
  // prefill_token_ids and re-seeds its per-request all_token_ids from it. The
  // MRV1 path (resumed-as-cached + CachedRequestData.all_token_ids) is NOT
  // ported (the V2 runner never reads CachedRequestData.all_token_ids).
  scheduled_new_reqs.insert(scheduled_new_reqs.end(),
                            scheduled_resumed_reqs.begin(),
                            scheduled_resumed_reqs.end());
  scheduled_resumed_reqs.clear();

  std::vector<NewRequestData> new_reqs_data;
  new_reqs_data.reserve(scheduled_new_reqs.size());
  for (Request* req : scheduled_new_reqs) {
    // prefill_token_ids = the request's full token ids (prompt+output; at
    // prefill this is the prompt). Upstream passes req._all_token_ids.
    new_reqs_data.push_back(NewRequestData::from_request(
        *req, req_to_new_blocks.at(req->request_id).get_block_ids(),
        req->AllTokenIds()));
  }

  // The cached diff now covers only the already-running reqs; scheduled_resumed_
  // reqs was cleared above (kept in the signature for upstream fidelity — the V2
  // branch still passes it, now empty).
  CachedRequestData cached_reqs_data = make_cached_request_data(
      scheduled_running_reqs, scheduled_resumed_reqs, num_scheduled_tokens,
      req_to_new_blocks);

  SchedulerOutput scheduler_output;
  scheduler_output.scheduled_new_reqs = std::move(new_reqs_data);
  scheduler_output.scheduled_cached_reqs = std::move(cached_reqs_data);
  // MOVE the num_scheduled_tokens map (dead after this) and the finished_req_ids
  // set (member, re-assigned fresh in update_after_schedule) instead of COPYING
  // them — vLLM passes the dicts by reference (rescan §6 item e, container
  // plumbing only, zero policy change). The local map is not read after this;
  // update_after_schedule reads scheduler_output.num_scheduled_tokens (the moved
  // destination) and clears the finished_req_ids member (`= {}`) so its
  // moved-from state is unobservable.
  scheduler_output.num_scheduled_tokens = std::move(num_scheduled_tokens);
  scheduler_output.total_num_scheduled_tokens = total_num_scheduled_tokens;
  // scheduled_spec_decode_tokens / scheduled_encoder_inputs stay empty (T0).
  scheduler_output.num_common_prefix_blocks = std::move(num_common_prefix_blocks);
  // finished_req_ids is EXISTING scheduler state (finished between the prior and
  // current step) — moved out here, then flushed in _update_after_schedule.
  scheduler_output.finished_req_ids = std::move(finished_req_ids);
  // free_encoder_mm_hashes stays empty (encoder deferred).

  update_after_schedule(scheduler_output);

  // Fold this step's prefix-cache lookups into the sliding-window hit-rate
  // aggregate. Upstream does this in the FRONTEND (SchedulerStats ->
  // LoggingStatLogger.observe, vllm/v1/metrics/loggers.py); we have no
  // SchedulerStats plumbing, so the aggregation lives on the Scheduler.
  // Behaviourally identical: make_prefix_cache_stats() is a take-and-swap of
  // the SAME per-step delta upstream ships to the logger, and CachingMetrics
  // ignores empty observations, so idle steps are free and cannot slide useful
  // history out of the window.
  if (auto stats = kv_cache_manager->make_prefix_cache_stats()) {
    prefix_cache_metrics_.observe(*stats);
  }

  // KV-OFFLOAD W4: build_connector_meta drains + RESETS the connector's per-step
  // batch state (base.py:516-517); on_schedule_end flushes deferred disk->CPU
  // promotions and polls transfers (tiering manager on_schedule_end). Null
  // connector = no-op. In a live engine the worker consumes the returned load
  // jobs; W4 wires the scheduler seam (default-off, provably inert), and the
  // load execution is exercised end-to-end in the connector correctness harness.
  if (kv_connector_ != nullptr) {
    kv_connector_->build_connector_meta();
    kv_connector_->on_schedule_end();
  }
  return scheduler_output;
}

std::optional<GrammarOutput> Scheduler::get_grammar_bitmask(
    const SchedulerOutput& scheduler_output) {
  // scheduler.py:1477-1499.
  // Collect the scheduled request ids that use structured output. The
  // corresponding rows of the bitmask will be in this order.
  if (!scheduler_output.has_structured_output_requests) {
    return std::nullopt;
  }
  // Without a manager wired, structured output is a no-op (backward-compat with
  // the M1.4/M1.8 tests). Upstream always has a manager.
  if (structured_output_manager_ == nullptr) {
    return std::nullopt;
  }

  std::vector<std::string> structured_output_request_ids;
  for (const auto& [req_id, num_tokens] : scheduler_output.num_scheduled_tokens) {
    (void)num_tokens;
    auto it = requests.find(req_id);
    if (it != requests.end() && it->second->use_structured_output() &&
        !it->second->is_prefill_chunk) {
      structured_output_request_ids.push_back(req_id);
    }
  }
  if (structured_output_request_ids.empty()) {
    return std::nullopt;
  }

  std::optional<TokenBitmask> bitmask = structured_output_manager_->grammar_bitmask(
      requests, structured_output_request_ids,
      scheduler_output.scheduled_spec_decode_tokens);
  if (!bitmask.has_value()) {
    return std::nullopt;
  }

  GrammarOutput out;
  out.structured_output_request_ids = std::move(structured_output_request_ids);
  out.grammar_bitmask = std::move(*bitmask);
  return out;
}

EngineCoreOutputs Scheduler::update_from_output(
    const SchedulerOutput& scheduler_output,
    const ModelRunnerOutput& model_runner_output) {
  const std::vector<std::vector<int32_t>>& sampled_token_ids =
      model_runner_output.sampled_token_ids;
  const std::map<std::string, int>& num_scheduled_tokens =
      scheduler_output.num_scheduled_tokens;

  std::vector<EngineCoreOutput> outputs;
  // Requests that stopped this step, split by the queue they must be removed
  // from (upstream stopped_running_reqs / stopped_preempted_reqs). The KV blocks
  // are freed and finished_req_ids updated inside the loop, but the owning
  // requests-map erase is deferred until after these pointers are used to filter
  // running/waiting (so the Request* stays valid — upstream relies on Python GC).
  std::set<Request*> stopped_running_reqs;
  std::set<Request*> stopped_preempted_reqs;
  std::vector<std::string> finished_ids_to_erase;

  // NOTE(woosuk): upstream iterates num_scheduled_tokens.items() (dict/schedule
  // order); std::map iterates in sorted key order. The set of outputs is the
  // same — only their order in the returned vector differs, which is benign
  // (each EngineCoreOutput is keyed by request_id).
  for (const auto& [req_id, num_tokens_scheduled] : num_scheduled_tokens) {
    assert(num_tokens_scheduled > 0);
    (void)num_tokens_scheduled;

    auto it = requests.find(req_id);
    if (it == requests.end() || it->second->IsFinished()) {
      // Already finished — e.g. aborted while the model was executing it.
      continue;
    }
    Request* request = it->second.get();

    const int req_index = model_runner_output.req_id_to_index.at(req_id);
    // sampled_token_ids[req_index] if sampled_token_ids else []. A request still
    // being prefilled gets an empty list from the runner.
    std::vector<int32_t> new_token_ids =
        sampled_token_ids.empty()
            ? std::vector<int32_t>{}
            : sampled_token_ids[static_cast<std::size_t>(req_index)];

    // DEFERRED: speculative-decode acceptance / num_computed rollback; encoder-
    // input free.

    bool stopped = false;
    const RequestStatus status_before_stop = request->status;

    // _update_request_with_output (scheduler.py:1627-1630): append + check_stop +
    // trim. Called only when the runner produced tokens (upstream `if
    // new_token_ids:`). The virtual hook lets AsyncScheduler wrap it (stale-frame
    // discard + placeholder drain + block caching).
    if (!new_token_ids.empty()) {
      auto result =
          update_request_with_output(*request, std::move(new_token_ids));
      new_token_ids = std::move(result.first);
      stopped = result.second;
    }
    // DEFERRED: pooling stop.

    // scheduler.py:1636-1651: advance the structured-output FSM by the sampled
    // tokens. Only when the request produced tokens and the manager says the FSM
    // should advance (should_advance is true for any structured request at T0).
    // A grammar that rejects the sampled tokens is unexpected: terminate the
    // request with FINISHED_ERROR. No-op when no manager is wired.
    if (!new_token_ids.empty() && structured_output_manager_ != nullptr &&
        structured_output_manager_->should_advance(*request)) {
      auto& struct_output_request = request->structured_output_request;
      assert(struct_output_request.has_value());
      assert(struct_output_request->grammar != nullptr);
      if (!struct_output_request->grammar->accept_tokens(req_id, new_token_ids)) {
        request->status = RequestStatus::kFinishedError;
        // (upstream also sets request.resumable = false; resumable is deferred.)
        stopped = true;
      }
    }

    std::optional<FinishReason> finish_reason;
    if (stopped) {
      // Capture the finish reason before freeing (upstream captures it before
      // _handle_stopped_request, which may reset the status for resumable reqs —
      // resumable/streaming is deferred, so _handle_stopped_request is always
      // "finished" at T0).
      finish_reason = request->GetFinishedReason();
      // _free_request + _free_blocks (T0 subset): free the KV blocks and record
      // the finished id now; defer the requests-map erase (see above).
      kv_cache_manager->free(*request);
      finished_req_ids.insert(request->request_id);
      finished_ids_to_erase.push_back(request->request_id);
      if (status_before_stop == RequestStatus::kRunning) {
        stopped_running_reqs.insert(request);
      } else {
        stopped_preempted_reqs.insert(request);
      }
    }

    // DEFERRED: sample logprobs / prompt logprobs / num_nans_in_logits.

    // Emit an EngineCoreOutput only when the request produced tokens or finished
    // (upstream's `if new_token_ids or ... or stopped`). A partial-prefill
    // request that produced neither is skipped: "EngineCore returns no partial
    // prefill outputs".
    if (!new_token_ids.empty() || stopped) {
      EngineCoreOutput out;
      out.request_id = req_id;
      out.new_token_ids = new_token_ids;
      out.finish_reason = finish_reason;
      // stop_reason is int|str|None upstream; our EngineCoreOutput carries an
      // optional<string> (see engine/types.h). Only a stop_token_ids match sets
      // request.stop_reason at T0 — stringify that token id; otherwise nullopt.
      if (request->stop_reason.has_value()) {
        out.stop_reason = std::to_string(*request->stop_reason);
      }
      outputs.push_back(std::move(out));
    }
  }

  // Remove the stopped requests from the running list and the waiting queue.
  if (!stopped_running_reqs.empty()) {
    running.erase(
        std::remove_if(running.begin(), running.end(),
                       [&](Request* r) {
                         return stopped_running_reqs.count(r) > 0;
                       }),
        running.end());
  }
  if (!stopped_preempted_reqs.empty()) {
    // Rare (a stopped-while-preempted request); remove each from waiting.
    std::vector<Request*> to_remove(stopped_preempted_reqs.begin(),
                                    stopped_preempted_reqs.end());
    waiting->remove_requests(to_remove);
  }
  // Now that no queue references them, drop the owning entries (destroys the
  // finished Request objects — upstream _free_blocks' `del self.requests[...]`).
  for (const std::string& id : finished_ids_to_erase) {
    requests.erase(id);
  }

  EngineCoreOutputs engine_core_outputs;
  engine_core_outputs.outputs = std::move(outputs);
  return engine_core_outputs;
}

CachedRequestData Scheduler::make_cached_request_data(
    const std::vector<Request*>& running_reqs,
    const std::vector<Request*>& resumed_reqs,
    const std::map<std::string, int>& num_scheduled_tokens,
    const std::map<std::string, KVCacheBlocks>& req_to_new_blocks) {
  (void)num_scheduled_tokens;  // Used only by the deferred PP token_ids path.
  CachedRequestData data;
  const std::size_t num_running_reqs = running_reqs.size();

  std::size_t idx = 0;
  auto handle = [&](Request* req) {
    const std::string& req_id = req->request_id;
    data.req_ids.push_back(req_id);
    // NOTE: use_pp is false at T0, so the PP new_token_ids payload is omitted
    // (the model runner caches the sampled tokens itself).
    if (idx >= num_running_reqs) {
      data.resumed_req_ids.insert(req_id);
    }
    // MRV2 (V2 model runner): CachedRequestData.all_token_ids is never read by
    // the runner, so the MRV1 prev-step-gated all_token_ids propagation is
    // dropped (the field stays empty). Resumed reqs no longer flow here at all
    // (folded into scheduled_new_reqs with prefill_token_ids upstream).
    data.new_block_ids.push_back(
        req_to_new_blocks.at(req_id).get_block_ids(/*allow_none=*/true));
    data.num_computed_tokens.push_back(req->num_computed_tokens);
    // num_output_placeholders == 0 at T0.
    data.num_output_tokens.push_back(req->NumOutputTokens());
    idx += 1;
  };

  for (Request* req : running_reqs) {
    handle(req);
  }
  for (Request* req : resumed_reqs) {
    handle(req);
  }
  return data;
}

std::pair<std::vector<int32_t>, bool> Scheduler::update_request_with_output(
    Request& request, std::vector<int32_t> new_token_ids) {
  // scheduler.py:1886-1902: append each generated token, run check_stop after
  // each, and trim any tokens generated past the stop.
  bool stopped = false;
  for (std::size_t num_new = 1; num_new <= new_token_ids.size(); ++num_new) {
    request.AppendOutputToken(new_token_ids[num_new - 1]);
    stopped = check_stop(request, max_model_len);
    if (stopped) {
      new_token_ids.resize(num_new);  // del new_token_ids[num_new:]
      break;
    }
  }
  return {std::move(new_token_ids), stopped};
}

void Scheduler::update_after_schedule(SchedulerOutput& scheduler_output) {
  // Advance num_computed_tokens AFTER building the output so a chunked prefill
  // resumes on the next step; refresh is_prefill_chunk.
  for (const auto& [req_id, num_scheduled_token] :
       scheduler_output.num_scheduled_tokens) {
    Request* request = requests.at(req_id).get();
    request->num_computed_tokens += num_scheduled_token;
    // Under async scheduling num_output_placeholders extends the target token
    // count (the in-flight sampled token(s) not yet appended); 0 for the
    // synchronous Scheduler, so this is unchanged there.
    request->is_prefill_chunk =
        request->num_computed_tokens <
        request->NumTokens() + request->num_output_placeholders;
    // scheduler.py:1186: a structured request that is past its prefill chunk
    // needs a grammar bitmask this step.
    scheduler_output.has_structured_output_requests |=
        request->use_structured_output() && !request->is_prefill_chunk;
  }
  // Flush the finished / preempted id sets (assign fresh sets so the already
  // copied-out scheduler_output is unaffected).
  finished_req_ids = {};
  reset_preempted_req_ids = {};
}

int Scheduler::get_num_unfinished_requests() const {
  return static_cast<int>(waiting->size()) + static_cast<int>(running.size());
}

std::pair<int, int> Scheduler::get_request_counts() const {
  return {static_cast<int>(running.size()),
          static_cast<int>(waiting->size())};
}

bool Scheduler::has_finished_requests() const {
  return !finished_req_ids.empty();
}

}  // namespace vllm::v1
