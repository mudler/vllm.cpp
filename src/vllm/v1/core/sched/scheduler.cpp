// Ported from: vllm/v1/core/sched/scheduler.py @ e24d1b24
// See include/vllm/v1/core/sched/scheduler.h for the T0 scope + deferred list.
#include "vllm/v1/core/sched/scheduler.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
                     bool enable_caching)
    : max_num_running_reqs(scheduler_config.max_num_seqs),
      max_num_scheduled_tokens(
          scheduler_config.ResolvedMaxNumScheduledTokens()),
      max_model_len(scheduler_config.max_model_len),
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
  // defaults to block_size; use_eagle / log_stats / kv_cache_events off and
  // dcp/pcp world sizes 1 at T0.
  kv_cache_manager = std::make_unique<KVCacheManager>(
      kv_cache_config_, max_model_len, /*scheduler_block_size=*/block_size,
      /*hash_block_size=*/block_size,
      /*max_num_batched_tokens=*/scheduler_config.max_num_batched_tokens,
      enable_caching, /*use_eagle=*/false, /*log_stats=*/false,
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

  // First, schedule the RUNNING requests.
  std::size_t req_index = 0;
  while (req_index < running.size() && token_budget > 0) {
    Request* request = running[req_index];

    // num_tokens_with_spec == num_tokens at T0 (no spec tokens);
    // num_output_placeholders == 0 (async scheduling deferred).
    int num_new_tokens = request->NumTokens() - request->num_computed_tokens;
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

    // Schedule the KV blocks, preempting the FCFS tail on OOM until it fits.
    std::optional<KVCacheBlocks> new_blocks;
    while (true) {
      new_blocks = kv_cache_manager->allocate_slots(
          *request, num_new_tokens, /*num_new_computed_tokens=*/0,
          /*new_computed_blocks=*/std::nullopt,
          /*num_lookahead_tokens=*/num_lookahead_tokens_);
      if (new_blocks.has_value()) {
        break;  // The request can be scheduled.
      }
      // Preempt the lowest-priority (FCFS tail) request.
      Request* preempted_req = running.back();
      running.pop_back();
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
          /*num_external_computed_tokens=*/0, /*delay_cache_blocks=*/false,
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
      num_scheduled_tokens[request_id] = num_new_tokens;
      token_budget -= num_new_tokens;
      request->status = RequestStatus::kRunning;
      request->num_computed_tokens = num_computed_tokens;
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
  scheduler_output.num_scheduled_tokens = num_scheduled_tokens;
  scheduler_output.total_num_scheduled_tokens = total_num_scheduled_tokens;
  // scheduled_spec_decode_tokens / scheduled_encoder_inputs stay empty (T0).
  scheduler_output.num_common_prefix_blocks = std::move(num_common_prefix_blocks);
  // finished_req_ids is EXISTING scheduler state (finished between the prior and
  // current step) — copied out here, then flushed in _update_after_schedule.
  scheduler_output.finished_req_ids = finished_req_ids;
  // free_encoder_mm_hashes stays empty (encoder deferred).

  update_after_schedule(scheduler_output);
  return scheduler_output;
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

void Scheduler::update_after_schedule(const SchedulerOutput& scheduler_output) {
  // Advance num_computed_tokens AFTER building the output so a chunked prefill
  // resumes on the next step; refresh is_prefill_chunk.
  for (const auto& [req_id, num_scheduled_token] :
       scheduler_output.num_scheduled_tokens) {
    Request* request = requests.at(req_id).get();
    request->num_computed_tokens += num_scheduled_token;
    // num_output_placeholders == 0 at T0.
    request->is_prefill_chunk =
        request->num_computed_tokens < request->NumTokens();
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
