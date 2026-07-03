// Tests for the V1 Scheduler.schedule() core token-budget algorithm
// (vllm/v1/core/sched/scheduler.py @ e24d1b24).
//
// Ported from tests/v1/core/test_scheduler.py @ e24d1b24 (the create_scheduler /
// create_requests helpers + the schedule() cases), adapted to the T0 subset
// (behavioral only, no model runner — chunked prefill is driven by schedule()'s
// own _update_after_schedule advancing num_computed_tokens; a running decode is
// simulated by appending an output token, which is what update_from_output will
// do once M1.4 Task 4 lands):
//   - test_schedule            : FCFS admission — every waiting request is
//                                scheduled as a new req with min(prompt, budget)
//                                tokens; waiting drains into running.
//   - test_schedule (budget)   : two requests share max_num_batched_tokens
//                                (chunked-prefill split falls out of the budget).
//   - chunked prefill          : a prompt longer than the budget is split across
//                                multiple schedule() calls (num_computed advances
//                                each step) — mirrors the chunked-prefill intent
//                                of test_schedule_partial_requests without the
//                                encoder budget.
//   - running decode           : after prefill completes, the request is
//                                scheduled as a cached (diff) req with 1 token.
//   - test_preempt_during_execution : filling the KV cache makes allocate_slots
//                                return nullopt for a running request, so the
//                                FCFS tail is preempted (status PREEMPTED, KV
//                                freed, re-queued to the waiting front).
//   - max_num_seqs cap         : only max_num_seqs requests run concurrently.
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::Request;
using vllm::v1::RequestStatus;
using vllm::v1::Scheduler;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

// M1.3's KVCacheManager only supports the prefix-caching coordinator
// (enable_caching == false is deferred), so the ported scheduler tests run with
// caching ON. Requests carry the cbor block hasher (block_hashes populate at
// block-size granularity); prompts are distinct per request ([i]*n), so there
// is no cross-request prefix reuse and admission/preemption stay deterministic.

// Mirror of test utils.create_scheduler (T0 subset).
std::unique_ptr<Scheduler> CreateScheduler(int max_num_seqs = 16,
                                           int max_num_batched_tokens = 8192,
                                           bool enable_chunked_prefill = true,
                                           int num_blocks = 10000,
                                           int block_size = 16,
                                           int max_model_len = 8192) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = enable_chunked_prefill;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;  // deterministic admission/preemption mechanics.

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));

  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true);
}

// Mirror of test utils.create_requests (T0 subset): prompts of `num_tokens`
// tokens, req ids "0".."n-1" (or the provided ids). Returns owning requests.
std::vector<std::unique_ptr<Request>> CreateRequests(
    int num_requests, int num_tokens = 10,
    const std::vector<std::string>& req_ids = {}) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  const int block_size = 16;
  auto block_hasher = get_request_block_hasher(block_size, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  std::vector<std::unique_ptr<Request>> requests;
  for (int i = 0; i < num_requests; ++i) {
    std::string id = req_ids.empty() ? std::to_string(i) : req_ids[i];
    std::vector<int32_t> prompt(num_tokens, static_cast<int32_t>(i));
    requests.push_back(std::make_unique<Request>(
        id, prompt, params, /*arrival_time=*/0.0, block_hasher));
  }
  return requests;
}

// Add a request and return the raw (non-owning) pointer for inspection.
Request* AddRequest(Scheduler& sched, std::unique_ptr<Request> req) {
  Request* raw = req.get();
  sched.add_request(std::move(req));
  return raw;
}

}  // namespace

// ---------------------------------------------------------------------------
// FCFS admission: every waiting request is scheduled as a new req with its full
// prompt (under budget); waiting drains into running. (test_schedule)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: FCFS admits every request under budget") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/10, /*num_tokens=*/10);
  std::vector<Request*> raw;
  for (auto& r : requests) raw.push_back(AddRequest(*scheduler, std::move(r)));

  auto output = scheduler->schedule();

  CHECK(output.scheduled_new_reqs.size() == 10);
  CHECK(output.scheduled_cached_reqs.num_reqs() == 0);
  CHECK(output.finished_req_ids.empty());
  // Every request scheduled its full prompt.
  for (const auto& [req_id, num_tokens] : output.num_scheduled_tokens) {
    CHECK(num_tokens == 10);
  }
  CHECK(output.total_num_scheduled_tokens == 100);

  // Waiting drained into running, in order.
  CHECK(scheduler->waiting->empty());
  REQUIRE(scheduler->running.size() == 10);
  for (int i = 0; i < 10; ++i) CHECK(scheduler->running[i] == raw[i]);
}

// ---------------------------------------------------------------------------
// Budget sharing + chunked-prefill split: two requests share the per-step
// budget; the second is chunked to fit. (budget of test_schedule / chunked
// prefill semantics)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: two requests share max_num_batched_tokens") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/30);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/20,
                                 {"a", "b"});
  AddRequest(*scheduler, std::move(requests[0]));
  AddRequest(*scheduler, std::move(requests[1]));

  auto output = scheduler->schedule();

  // "a" fills its 20 tokens; "b" gets the remaining 10 (chunked).
  CHECK(output.num_scheduled_tokens.at("a") == 20);
  CHECK(output.num_scheduled_tokens.at("b") == 10);
  CHECK(output.total_num_scheduled_tokens == 30);
  CHECK(output.scheduled_new_reqs.size() == 2);
}

// ---------------------------------------------------------------------------
// Chunked prefill split across steps: a long prompt (80) with budget 30 is
// scheduled 30 / 30 / 20 over three steps, num_computed advancing each step.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: chunked prefill splits a long prompt") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/30);
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/80, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto out0 = scheduler->schedule();
  CHECK(out0.num_scheduled_tokens.at("0") == 30);
  CHECK(out0.scheduled_new_reqs.size() == 1);  // first chunk => new req
  CHECK(req->num_computed_tokens == 30);

  auto out1 = scheduler->schedule();
  CHECK(out1.num_scheduled_tokens.at("0") == 30);
  CHECK(out1.scheduled_new_reqs.empty());  // subsequent chunks => cached diff
  CHECK(out1.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(req->num_computed_tokens == 60);

  auto out2 = scheduler->schedule();
  CHECK(out2.num_scheduled_tokens.at("0") == 20);
  CHECK(req->num_computed_tokens == 80);
  CHECK_FALSE(req->is_prefill_chunk);  // prefill complete.
}

// ---------------------------------------------------------------------------
// Running decode: after prefill completes, an appended output token (simulating
// the model runner sample) makes the request schedule as a cached (diff) req
// with exactly 1 new token. (SchedulerOutput new-vs-cached shape.)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: running request decodes as a cached req") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto out0 = scheduler->schedule();  // prefill
  CHECK(out0.scheduled_new_reqs.size() == 1);
  CHECK(out0.num_scheduled_tokens.at("0") == 4);
  REQUIRE(req->num_computed_tokens == 4);

  // Simulate the runner sampling one token.
  req->AppendOutputToken(0);

  auto out1 = scheduler->schedule();  // decode
  CHECK(out1.scheduled_new_reqs.empty());
  CHECK(out1.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(out1.scheduled_cached_reqs.req_ids[0] == "0");
  CHECK(out1.num_scheduled_tokens.at("0") == 1);
  CHECK(scheduler->running.size() == 1);
}

// ---------------------------------------------------------------------------
// Preemption on KV exhaustion: with only 10 usable blocks, two 80-token
// requests (5 blocks each) fill the cache; when the first needs a 6th block for
// its decode, allocate_slots returns nullopt and the FCFS tail (the second
// request) is preempted. (test_preempt_during_execution)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: KV exhaustion preempts the FCFS tail") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/100,
                                   /*enable_chunked_prefill=*/true,
                                   /*num_blocks=*/11, /*block_size=*/16);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/80,
                                 {"0", "1"});
  Request* req0 = requests[0].get();
  Request* req1 = requests[1].get();

  // Schedule the first request (5 blocks for 80 tokens).
  AddRequest(*scheduler, std::move(requests[0]));
  auto out0 = scheduler->schedule();
  CHECK(out0.num_scheduled_tokens.size() == 1);
  REQUIRE(out0.scheduled_new_reqs.size() == 1);
  CHECK(out0.scheduled_new_reqs[0].block_ids[0].size() == 5);

  // Schedule the second request while the first still runs (5 more blocks).
  AddRequest(*scheduler, std::move(requests[1]));
  auto out1 = scheduler->schedule();
  CHECK(out1.num_scheduled_tokens.size() == 1);
  REQUIRE(out1.scheduled_new_reqs.size() == 1);
  CHECK(out1.scheduled_new_reqs[0].block_ids[0].size() == 5);

  // The first request samples a token -> it needs a 6th block next step, but
  // the KV cache is full, so the second request is preempted.
  req0->AppendOutputToken(0);
  auto out2 = scheduler->schedule();

  CHECK(scheduler->running.size() == 1);
  CHECK(scheduler->running[0] == req0);
  CHECK(req1->status == RequestStatus::kPreempted);
  // The preempted request is re-queued to the waiting front, KV freed.
  CHECK(scheduler->waiting->size() == 1);
  CHECK(scheduler->waiting->peek_request() == req1);
  CHECK(req1->num_computed_tokens == 0);
  CHECK(out2.num_scheduled_tokens.count("0") == 1);
  CHECK(out2.num_scheduled_tokens.count("1") == 0);
}

// ---------------------------------------------------------------------------
// max_num_seqs cap: only max_num_seqs requests run concurrently. (test_schedule
// admission bound.)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: max_num_seqs caps concurrent requests") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/2);
  auto requests = CreateRequests(/*num_requests=*/3, /*num_tokens=*/4);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto output = scheduler->schedule();

  CHECK(output.scheduled_new_reqs.size() == 2);
  CHECK(scheduler->running.size() == 2);
  CHECK(scheduler->waiting->size() == 1);
}

// ---------------------------------------------------------------------------
// add_request / finish_requests / counts (T0 subset). (test_add_requests,
// test_finish_request, test_get_num_unfinished_requests.)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler: add_request registers into waiting + requests map") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/10);
  for (int i = 0; i < 10; ++i) {
    std::string id = requests[i]->request_id;
    AddRequest(*scheduler, std::move(requests[i]));
    CHECK(scheduler->requests.count(id) == 1);
    CHECK(static_cast<int>(scheduler->waiting->size()) == i + 1);
  }
  CHECK(scheduler->get_num_unfinished_requests() == 10);
}

TEST_CASE("Scheduler: finish_requests removes from waiting + requests map") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/10);
  std::vector<std::string> ids;
  for (auto& r : requests) {
    ids.push_back(r->request_id);
    AddRequest(*scheduler, std::move(r));
  }

  for (int i = 0; i < 10; ++i) {
    scheduler->finish_requests(ids[i], RequestStatus::kFinishedAborted);
    CHECK(scheduler->requests.count(ids[i]) == 0);
    CHECK(static_cast<int>(scheduler->waiting->size()) == 9 - i);
  }
}
