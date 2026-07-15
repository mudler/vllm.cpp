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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::SchedulerPolicy;
using vllm::v1::EngineCoreOutputs;
using vllm::v1::FinishReason;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerOutput;
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

// Build a ModelRunnerOutput from an ordered list of (req_id, sampled_tokens),
// mirroring what the model runner (M1.5) returns: req_ids order + the
// req_id_to_index map + the ragged sampled_token_ids. An empty token list models
// a request still in chunked prefill (no sample this step).
ModelRunnerOutput MakeRunnerOutput(
    const std::vector<std::pair<std::string, std::vector<int32_t>>>& per_req) {
  ModelRunnerOutput mro;
  int idx = 0;
  for (const auto& [id, toks] : per_req) {
    mro.req_ids.push_back(id);
    mro.req_id_to_index[id] = idx++;
    mro.sampled_token_ids.push_back(toks);
  }
  return mro;
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
// Budget-filling co-schedule (SERVE-GATE c2 parity). Two prompts that EXACTLY
// fill the per-step token budget (2 x 1024 == 2048, the online-gate workload's
// in1024/--max-num-batched-tokens 2048) are BOTH scheduled in a SINGLE step:
// each gets its full 1024 tokens, neither is chunked, and the budget lands at 0.
// This mirrors upstream scheduler.py:640-1013 (the waiting loop keeps admitting
// while token_budget > 0 with no per-step / partial-prefill cap — default
// max_num_partial_prefills == 1, long_prefill_token_threshold == 0), so our
// waiting loop (scheduler.cpp:234-298) co-schedules identically. RED if we ever
// serialized budget-filling prefills; GREEN proves we mirror vLLM's co-schedule.
// Ledger: the component's bimodal c2 TTFT (~0.45 s vs ~0.9 s) is therefore pure
// ARRIVAL phasing (whether both reqs are enqueued at the same schedule() call),
// not a scheduler divergence — see the next case for the staggered arrival.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: two budget-filling prefills co-schedule (c2 parity)") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/32,
                                   /*max_num_batched_tokens=*/2048);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/1024,
                                 {"a", "b"});
  Request* a = AddRequest(*scheduler, std::move(requests[0]));
  Request* b = AddRequest(*scheduler, std::move(requests[1]));

  auto output = scheduler->schedule();

  // Both prefills run in ONE step, each at its full 1024 tokens (no chunk).
  CHECK(output.num_scheduled_tokens.at("a") == 1024);
  CHECK(output.num_scheduled_tokens.at("b") == 1024);
  CHECK(output.total_num_scheduled_tokens == 2048);  // budget filled exactly.
  CHECK(output.scheduled_new_reqs.size() == 2);      // neither serialized.
  CHECK(output.scheduled_cached_reqs.num_reqs() == 0);
  CHECK(scheduler->waiting->empty());
  REQUIRE(scheduler->running.size() == 2);
  CHECK(scheduler->running[0] == a);
  CHECK(scheduler->running[1] == b);
  // Both completed prefill this step -> neither is a lingering prefill chunk.
  CHECK_FALSE(a->is_prefill_chunk);
  CHECK_FALSE(b->is_prefill_chunk);
}

// ---------------------------------------------------------------------------
// Staggered arrival (c2 TTFT lottery mechanism). If the second 1024-token
// request is not yet enqueued when the first is scheduled, the first prefills
// ALONE this step (the ~0.45 s TTFT mode). On the next step the first request's
// single decode token co-schedules with the second request's full 1024-token
// prefill inside the same 2048 budget (1 + 1024 <= 2048) — the late prefill is
// NOT refused/serialized, it simply landed one step later (the ~0.9 s mode).
// This is the arrival-phasing "lottery" behind the component's bimodal c2 TTFT;
// it is identical to upstream, so the leg-to-leg 3/3-vs-6/0 mixture is timing,
// not a scheduler policy difference (no scheduler change is warranted).
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: a late prefill co-schedules with a running decode") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/32,
                                   /*max_num_batched_tokens=*/2048);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/1024,
                                 {"a", "b"});
  Request* a = AddRequest(*scheduler, std::move(requests[0]));

  // Step 1: only "a" is enqueued -> it prefills ALONE (the fast TTFT mode).
  auto out0 = scheduler->schedule();
  CHECK(out0.num_scheduled_tokens.at("a") == 1024);
  CHECK(out0.num_scheduled_tokens.size() == 1);
  CHECK(out0.total_num_scheduled_tokens == 1024);  // half the budget unused.
  REQUIRE(a->num_computed_tokens == 1024);

  // "a" samples its first token (runner would return one on the next step).
  a->AppendOutputToken(0);
  // "b" arrives after "a" was already dispatched.
  Request* b = AddRequest(*scheduler, std::move(requests[1]));

  // Step 2: "a" decodes 1 token AND "b" prefills its full 1024 in the SAME step.
  auto out1 = scheduler->schedule();
  CHECK(out1.num_scheduled_tokens.at("a") == 1);      // running decode.
  CHECK(out1.num_scheduled_tokens.at("b") == 1024);   // late prefill co-scheduled.
  CHECK(out1.total_num_scheduled_tokens == 1025);
  CHECK(out1.scheduled_new_reqs.size() == 1);         // only "b" is new.
  CHECK(out1.scheduled_new_reqs[0].req_id == "b");
  CHECK(out1.scheduled_cached_reqs.num_reqs() == 1);  // "a" as a cached diff.
  REQUIRE(scheduler->running.size() == 2);
  CHECK(scheduler->waiting->empty());
  (void)b;
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
// Resumed-as-new (MRV2 output fold): a preempted request that is re-scheduled
// after KV frees appears in scheduled_new_reqs (NOT the cached diff), carrying
// prefill_token_ids == its full token ids. This exercises the V2 output tail
// (upstream scheduler.py `if self.use_v2_model_runner:` branch): resumed reqs
// are folded into scheduled_new_reqs and re-sent as FULL new reqs so the V2 gpu
// runner (M1.5) can re-seed its per-request all_token_ids from prefill_token_ids.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: resumed (preempted) request re-enters as new") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/100,
                                   /*enable_chunked_prefill=*/true,
                                   /*num_blocks=*/11, /*block_size=*/16);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/80,
                                 {"0", "1"});
  Request* req0 = requests[0].get();
  Request* req1 = requests[1].get();

  // Fill the KV cache with two 80-token reqs (5 blocks each).
  AddRequest(*scheduler, std::move(requests[0]));
  scheduler->schedule();
  AddRequest(*scheduler, std::move(requests[1]));
  scheduler->schedule();

  // req0 samples a token -> needs a 6th block; KV is full, so the FCFS tail
  // (req1) is preempted (KV freed, re-queued to the waiting front).
  req0->AppendOutputToken(0);
  scheduler->schedule();
  REQUIRE(req1->status == RequestStatus::kPreempted);
  REQUIRE(scheduler->waiting->peek_request() == req1);

  // Free req0's KV so the preempted req1 has room to be re-scheduled.
  scheduler->finish_requests("0", RequestStatus::kFinishedStopped);

  auto out = scheduler->schedule();

  // The V2 fold: req1 resumes as a NEW req (folded from scheduled_resumed_reqs),
  // NOT through the cached diff.
  REQUIRE(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "1");
  CHECK(out.scheduled_cached_reqs.num_reqs() == 0);
  // resumed_req_ids in the cached diff stays empty (resumed folded into new).
  CHECK(out.scheduled_cached_reqs.resumed_req_ids.empty());
  CHECK(req1->status == RequestStatus::kRunning);

  // prefill_token_ids carries req1's full token ids so the V2 runner re-seeds
  // its state; req1 never decoded, so this is its 80-token prompt.
  const auto& seed = out.scheduled_new_reqs[0].prefill_token_ids;
  REQUIRE(seed.has_value());
  CHECK(*seed == req1->AllTokenIds());
  CHECK(seed->size() == 80);
}

// ---------------------------------------------------------------------------
// A first-time (WAITING->RUNNING) new req also carries prefill_token_ids == its
// full token ids under the MRV2 path (upstream V2 branch seeds every new req).
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: new req carries prefill_token_ids seed") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/10, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto out = scheduler->schedule();

  REQUIRE(out.scheduled_new_reqs.size() == 1);
  const auto& seed = out.scheduled_new_reqs[0].prefill_token_ids;
  REQUIRE(seed.has_value());
  CHECK(*seed == req->AllTokenIds());
  CHECK(seed->size() == 10);
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

// ===========================================================================
// update_from_output — the schedule -> execute -> update loop (M1.4 Task 4).
// Ported from tests/v1/core/test_scheduler.py @ e24d1b24: the
// test_update_from_output family, test_stop_via_update_from_output (eos /
// stop_token_ids / max_tokens / min_tokens), and the partial-prefill
// "no sampled tokens" case (test_schedule_partial_requests).
// ===========================================================================

// A sampled token is appended and the request advances (no stop, still running).
// (test_update_from_output.)
TEST_CASE("Scheduler.update_from_output: appends the sampled token") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto sched_out = scheduler->schedule();  // prefill 4 tokens
  REQUIRE(req->num_computed_tokens == 4);

  auto mro = MakeRunnerOutput({{"0", {42}}});
  EngineCoreOutputs eco = scheduler->update_from_output(sched_out, mro);

  // The token was appended; the request keeps running (max_tokens 16, no eos).
  REQUIRE(req->output_token_ids.size() == 1);
  CHECK(req->output_token_ids[0] == 42);
  CHECK(req->status == RequestStatus::kRunning);
  CHECK_FALSE(req->IsFinished());

  // EngineCoreOutputs shape: one output carrying the new token, no finish.
  REQUIRE(eco.outputs.size() == 1);
  CHECK(eco.outputs[0].request_id == "0");
  CHECK(eco.outputs[0].new_token_ids == std::vector<int32_t>{42});
  CHECK_FALSE(eco.outputs[0].finish_reason.has_value());
  CHECK_FALSE(eco.outputs[0].Finished());
  CHECK(scheduler->running.size() == 1);
  CHECK(scheduler->finished_req_ids.empty());
}

// An EOS token finishes the request: FINISHED_STOPPED, KV freed, id in
// finished_req_ids, erased from the requests map, finish_reason == stop.
// (test_stop_via_update_from_output — eos.)
TEST_CASE("Scheduler.update_from_output: eos token stops the request") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));
  req->sampling_params.eos_token_id = 7;

  auto sched_out = scheduler->schedule();
  auto mro = MakeRunnerOutput({{"0", {7}}});
  EngineCoreOutputs eco = scheduler->update_from_output(sched_out, mro);

  REQUIRE(eco.outputs.size() == 1);
  CHECK(eco.outputs[0].request_id == "0");
  CHECK(eco.outputs[0].new_token_ids == std::vector<int32_t>{7});
  REQUIRE(eco.outputs[0].finish_reason.has_value());
  CHECK(*eco.outputs[0].finish_reason == FinishReason::kStop);
  CHECK(eco.outputs[0].Finished());
  // Freed + removed everywhere.
  CHECK(scheduler->finished_req_ids.count("0") == 1);
  CHECK(scheduler->requests.count("0") == 0);  // erased from the owning map
  CHECK(scheduler->running.empty());
}

// max_tokens reached -> FINISHED_LENGTH_CAPPED / finish_reason == length.
// (test_stop_via_update_from_output — max_tokens.)
TEST_CASE("Scheduler.update_from_output: max_tokens length-caps the request") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));
  req->sampling_params.max_tokens = 1;  // one output token is already the cap

  auto sched_out = scheduler->schedule();
  auto mro = MakeRunnerOutput({{"0", {5}}});  // not eos, not a stop token
  EngineCoreOutputs eco = scheduler->update_from_output(sched_out, mro);

  REQUIRE(eco.outputs.size() == 1);
  REQUIRE(eco.outputs[0].finish_reason.has_value());
  CHECK(*eco.outputs[0].finish_reason == FinishReason::kLength);
  CHECK(scheduler->finished_req_ids.count("0") == 1);
  CHECK(scheduler->requests.count("0") == 0);
}

// min_tokens gates the EOS stop: the same eos token does NOT stop until at least
// min_tokens output tokens exist. (test_stop_via_update_from_output — min_tokens.)
TEST_CASE("Scheduler.update_from_output: min_tokens prevents an early eos stop") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));
  req->sampling_params.eos_token_id = 7;
  req->sampling_params.min_tokens = 2;
  req->sampling_params.max_tokens = 16;

  // First eos: only 1 output token < min_tokens(2) -> not stopped.
  auto out0 = scheduler->schedule();
  auto eco0 = scheduler->update_from_output(out0, MakeRunnerOutput({{"0", {7}}}));
  CHECK_FALSE(req->IsFinished());
  REQUIRE(eco0.outputs.size() == 1);
  CHECK_FALSE(eco0.outputs[0].finish_reason.has_value());

  // Second eos: 2 output tokens >= min_tokens(2) -> stopped.
  auto out1 = scheduler->schedule();  // decode step (num_new == 1)
  auto eco1 = scheduler->update_from_output(out1, MakeRunnerOutput({{"0", {7}}}));
  REQUIRE(eco1.outputs.size() == 1);
  REQUIRE(eco1.outputs[0].finish_reason.has_value());
  CHECK(*eco1.outputs[0].finish_reason == FinishReason::kStop);
  CHECK(scheduler->requests.count("0") == 0);
}

// A stop_token_ids match stops the request and carries the token id as
// stop_reason. (test_stop_via_update_from_output — stop_token_ids.)
TEST_CASE("Scheduler.update_from_output: stop_token_ids match sets stop_reason") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));
  req->sampling_params.stop_token_ids = {99};
  req->sampling_params.max_tokens = 16;

  auto sched_out = scheduler->schedule();
  auto eco = scheduler->update_from_output(sched_out, MakeRunnerOutput({{"0", {99}}}));

  REQUIRE(eco.outputs.size() == 1);
  REQUIRE(eco.outputs[0].finish_reason.has_value());
  CHECK(*eco.outputs[0].finish_reason == FinishReason::kStop);
  REQUIRE(eco.outputs[0].stop_reason.has_value());
  CHECK(*eco.outputs[0].stop_reason == "99");
  CHECK(scheduler->requests.count("0") == 0);
}

// A request still in chunked prefill receives an empty sampled-token list and
// gets no output, staying running with its output unchanged.
// (test_schedule_partial_requests intent.)
TEST_CASE("Scheduler.update_from_output: prefilling request gets no token") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/30);
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/80, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto sched_out = scheduler->schedule();  // first chunk (30 of 80)
  REQUIRE(req->num_computed_tokens == 30);
  REQUIRE(req->is_prefill_chunk);

  // The runner returns an empty token list for a still-prefilling request.
  auto eco = scheduler->update_from_output(sched_out, MakeRunnerOutput({{"0", {}}}));

  CHECK(eco.outputs.empty());  // no partial-prefill output
  CHECK(req->output_token_ids.empty());
  CHECK(req->num_computed_tokens == 30);  // update_from_output does not advance it
  CHECK(req->status == RequestStatus::kRunning);
  CHECK(scheduler->requests.count("0") == 1);
  CHECK(scheduler->finished_req_ids.empty());
}

// EngineCoreOutputs shape with a mixed batch: two decoding requests each yield
// one output; a third still-prefilling request yields none.
TEST_CASE("Scheduler.update_from_output: batched outputs shape") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/40);
  // "a"/"b" are short (fully prefill this step); "c" is long (chunked).
  auto reqs = CreateRequests(/*num_requests=*/2, /*num_tokens=*/4, {"a", "b"});
  AddRequest(*scheduler, std::move(reqs[0]));
  AddRequest(*scheduler, std::move(reqs[1]));
  auto longreq = CreateRequests(/*num_requests=*/1, /*num_tokens=*/60, {"c"});
  AddRequest(*scheduler, std::move(longreq[0]));

  auto sched_out = scheduler->schedule();
  REQUIRE(sched_out.num_scheduled_tokens.at("a") == 4);
  REQUIRE(sched_out.num_scheduled_tokens.at("b") == 4);
  REQUIRE(sched_out.num_scheduled_tokens.at("c") == 32);  // chunked remainder

  auto eco = scheduler->update_from_output(
      sched_out, MakeRunnerOutput({{"a", {1}}, {"b", {2}}, {"c", {}}}));

  // "a" and "b" produced a token; "c" (still prefilling) produced none.
  CHECK(eco.outputs.size() == 2);
  CHECK(eco.engine_index == 0);
}

// ===========================================================================
// Priority scheduling (W4 / ENG-PRIORITY-SCHED).
// Ported from tests/v1/core/test_scheduler.py @ e24d1b24: the 11
// test_priority_scheduling_* cases (:2382-2856) +
// test_priority_scheduling_preemption_and_resumption_when_out_of_kv (:2978),
// with the create_scheduler_with_priority / create_requests_with_priority
// helpers (:2155,:2278).
//
// DEVIATION (recorded): our M1.3 KVCacheManager only supports the
// prefix-caching coordinator (enable_caching == false is deferred), so these
// run with caching ON — exactly like the FCFS scheduler tests above. Upstream's
// priority tests run with caching OFF (create_scheduler_with_priority default
// enable_prefix_caching=False). To keep caching ON behaviorally equivalent to
// caching OFF for the block-math (preemption/resumption) cases, every request is
// given a DISTINCT prompt (via a distinct `starting_idx`) so there is no
// cross-request prefix reuse — the tests exercise priority ORDER / victim
// selection, not the prefix cache. The EC/KV-connector variant (:3769) is not
// ported (no connectors); only the V2-model-runner, no-connector path is
// covered (that is our only output path).
// ===========================================================================

namespace {

// Mirror of test utils.create_scheduler_with_priority (T0 subset): identical to
// CreateScheduler but policy = priority.
std::unique_ptr<Scheduler> CreateSchedulerWithPriority(
    int max_num_seqs = 16, int max_num_batched_tokens = 8192,
    int num_blocks = 10000, int block_size = 16, int max_model_len = 8192) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;
  cfg.policy = SchedulerPolicy::kPriority;  // enable priority scheduling.

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));

  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true);
}

// Mirror of test utils.create_requests_with_priority (T0 subset): each request
// carries a priority + arrival time; prompt = [i + starting_idx] * num_tokens
// (distinct per i+starting_idx). Default req ids "{i+starting_idx}".
std::vector<std::unique_ptr<Request>> CreateRequestsWithPriority(
    int num_requests, const std::vector<int>& priorities,
    const std::vector<double>& arrival_times = {}, int num_tokens = 10,
    int max_tokens = 16, const std::vector<std::string>& req_ids = {},
    int starting_idx = 0) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  const int block_size = 16;
  auto block_hasher = get_request_block_hasher(block_size, sha256_cbor);
  SamplingParams params;
  params.max_tokens = max_tokens;

  std::vector<std::unique_ptr<Request>> requests;
  for (int i = 0; i < num_requests; ++i) {
    std::string id =
        req_ids.empty() ? std::to_string(i + starting_idx) : req_ids[i];
    const double arrival =
        arrival_times.empty() ? static_cast<double>(i) : arrival_times[i];
    std::vector<int32_t> prompt(num_tokens,
                                static_cast<int32_t>(i + starting_idx));
    requests.push_back(std::make_unique<Request>(
        id, prompt, params, arrival, block_hasher, priorities[i]));
  }
  return requests;
}

// Collect the scheduled_new_reqs req ids in schedule order.
std::vector<std::string> ScheduledNewIds(const vllm::v1::SchedulerOutput& out) {
  std::vector<std::string> ids;
  for (const auto& nr : out.scheduled_new_reqs) ids.push_back(nr.req_id);
  return ids;
}

}  // namespace

// test_priority_scheduling_basic_ordering: lower priority value first.
TEST_CASE("Priority scheduling: basic priority ordering") {
  auto scheduler = CreateSchedulerWithPriority();
  auto requests = CreateRequestsWithPriority(
      /*num_requests=*/3, /*priorities=*/{2, 0, 1},
      /*arrival_times=*/{1.0, 2.0, 3.0});
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();

  CHECK(out.scheduled_new_reqs.size() == 3);
  // req_1 (prio 0), req_2 (prio 1), req_0 (prio 2).
  CHECK(ScheduledNewIds(out) == std::vector<std::string>{"1", "2", "0"});
}

// test_priority_scheduling_arrival_time_tiebreaker: equal priority -> arrival.
TEST_CASE("Priority scheduling: arrival time tiebreaker") {
  auto scheduler = CreateSchedulerWithPriority();
  auto requests = CreateRequestsWithPriority(
      /*num_requests=*/3, /*priorities=*/{1, 1, 1},
      /*arrival_times=*/{3.0, 1.0, 2.0});
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();

  CHECK(out.scheduled_new_reqs.size() == 3);
  // req_1 (1.0), req_2 (2.0), req_0 (3.0).
  CHECK(ScheduledNewIds(out) == std::vector<std::string>{"1", "2", "0"});
}

// test_priority_scheduling_mixed_priority_and_arrival.
TEST_CASE("Priority scheduling: mixed priority and arrival") {
  auto scheduler = CreateSchedulerWithPriority();
  auto requests = CreateRequestsWithPriority(
      /*num_requests=*/4, /*priorities=*/{2, 1, 1, 0},
      /*arrival_times=*/{1.0, 3.0, 2.0, 4.0});
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();

  CHECK(out.scheduled_new_reqs.size() == 4);
  // req_3 (prio 0), req_2 (prio 1, arr 2), req_1 (prio 1, arr 3), req_0 (prio 2).
  CHECK(ScheduledNewIds(out) == std::vector<std::string>{"3", "2", "1", "0"});
}

// test_priority_scheduling_preemption: under KV pressure the lowest-PRIORITY
// running request is preempted, not the FCFS tail.
TEST_CASE("Priority scheduling: preempts the lowest-priority running request") {
  const int block_size = 16;
  const int num_blocks = 6;             // 1 null -> 5 usable
  const int num_tokens = block_size * 2;  // 32 tokens = exactly 2 blocks
  auto scheduler = CreateSchedulerWithPriority(
      /*max_num_seqs=*/3, /*max_num_batched_tokens=*/200, num_blocks, block_size);

  // Phase 1: low-priority request starts running (distinct prompt via idx 0).
  auto lo = CreateRequestsWithPriority(1, {5}, {1.0}, num_tokens, /*max_tokens=*/16,
                                       {"lo1"}, /*starting_idx=*/0);
  Request* lo1 = lo[0].get();
  AddRequest(*scheduler, std::move(lo[0]));
  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  scheduler->update_from_output(out, MakeRunnerOutput({{"lo1", {100}}}));

  // Phase 2: high-priority request arrives (distinct prompt via idx 1).
  auto hi = CreateRequestsWithPriority(1, {0}, {2.0}, num_tokens, /*max_tokens=*/16,
                                       {"hi1"}, /*starting_idx=*/1);
  AddRequest(*scheduler, std::move(hi[0]));
  out = scheduler->schedule();
  bool hi_admitted = false;
  for (const auto& nr : out.scheduled_new_reqs)
    if (nr.req_id == "hi1") hi_admitted = true;
  CHECK(hi_admitted);
  CHECK(scheduler->running.size() == 2);
  scheduler->update_from_output(out,
                                MakeRunnerOutput({{"lo1", {101}}, {"hi1", {100}}}));

  // Phase 3: hi1 needs a 3rd block, 0 free -> preempt the lowest-priority
  // running request = lo1 (priority 5 > 0).
  out = scheduler->schedule();
  CHECK(lo1->status == RequestStatus::kPreempted);
  bool hi_running = false;
  for (Request* r : scheduler->running)
    if (r->request_id == "hi1") hi_running = true;
  CHECK(hi_running);
}

// test_priority_scheduling_no_preemption_when_space_available.
TEST_CASE("Priority scheduling: no preemption when space is available") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/3,
                                               /*max_num_batched_tokens=*/200);
  auto lows = CreateRequestsWithPriority(2, {5, 5}, {1.0, 2.0}, /*num_tokens=*/30,
                                         /*max_tokens=*/16, {"lo1", "lo2"},
                                         /*starting_idx=*/0);
  for (auto& r : lows) AddRequest(*scheduler, std::move(r));
  auto out = scheduler->schedule();
  scheduler->update_from_output(
      out, MakeRunnerOutput({{"lo1", {100}}, {"lo2", {100}}}));

  auto hi = CreateRequestsWithPriority(1, {0}, {3.0}, /*num_tokens=*/30,
                                       /*max_tokens=*/16, {"hi1"},
                                       /*starting_idx=*/2);
  AddRequest(*scheduler, std::move(hi[0]));
  out = scheduler->schedule();

  CHECK(out.scheduled_new_reqs.size() == 1);  // hi1 admitted, no preemption
  CHECK(scheduler->running.size() == 3);
  CHECK(scheduler->waiting->empty());
}

// test_priority_scheduling_preemption_victim_selection: waiting queue order.
TEST_CASE("Priority scheduling: victim selection / waiting queue order") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/1);
  auto requests = CreateRequestsWithPriority(3, {3, 2, 0}, {1.0, 2.0, 3.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "2");  // highest priority

  auto waiting = scheduler->waiting->ToList();
  REQUIRE(waiting.size() == 2);
  CHECK(waiting[0]->priority == 2);
  CHECK(waiting[1]->priority == 3);
  CHECK(waiting[0]->request_id == "1");
  CHECK(waiting[1]->request_id == "0");
}

// test_priority_scheduling_equal_priority_preemption: arrival tiebreak in wait.
TEST_CASE("Priority scheduling: equal priority -> arrival tiebreak in waiting") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/1);
  auto requests = CreateRequestsWithPriority(3, {2, 2, 2}, {3.0, 1.0, 2.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "1");  // earliest arrival (1.0)

  auto waiting = scheduler->waiting->ToList();
  REQUIRE(waiting.size() == 2);
  CHECK(waiting[0]->arrival_time == 2.0);
  CHECK(waiting[1]->arrival_time == 3.0);
  CHECK(waiting[0]->request_id == "2");
  CHECK(waiting[1]->request_id == "0");
}

// test_priority_scheduling_waiting_queue_order.
TEST_CASE("Priority scheduling: waiting queue maintains priority order") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/1);
  auto requests = CreateRequestsWithPriority(4, {3, 1, 2, 0},
                                             {1.0, 2.0, 3.0, 4.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "3");  // priority 0

  auto waiting = scheduler->waiting->ToList();
  REQUIRE(waiting.size() == 3);
  std::vector<std::string> ids;
  std::vector<int> prios;
  for (Request* r : waiting) {
    ids.push_back(r->request_id);
    prios.push_back(r->priority);
  }
  CHECK(ids == std::vector<std::string>{"1", "2", "0"});
  CHECK(prios == std::vector<int>{1, 2, 3});
}

// test_priority_scheduling_fcfs_fallback: equal priority -> FCFS by arrival.
TEST_CASE("Priority scheduling: FCFS fallback when priorities equal") {
  auto scheduler = CreateSchedulerWithPriority();
  auto requests = CreateRequestsWithPriority(4, {1, 1, 1, 1},
                                             {4.0, 1.0, 3.0, 2.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 4);
  // Arrival order: req_1 (1.0), req_3 (2.0), req_2 (3.0), req_0 (4.0).
  CHECK(ScheduledNewIds(out) == std::vector<std::string>{"1", "3", "2", "0"});
}

// test_priority_scheduling_with_limited_slots.
TEST_CASE("Priority scheduling: max_num_seqs limits to highest priorities") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/2,
                                               /*max_num_batched_tokens=*/1000);
  auto requests = CreateRequestsWithPriority(4, {3, 1, 2, 0},
                                             {1.0, 2.0, 3.0, 4.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  REQUIRE(out.scheduled_new_reqs.size() == 2);
  auto ids = ScheduledNewIds(out);
  // The 2 highest priorities: req_3 (prio 0), req_1 (prio 1).
  CHECK(std::find(ids.begin(), ids.end(), "3") != ids.end());
  CHECK(std::find(ids.begin(), ids.end(), "1") != ids.end());

  auto waiting = scheduler->waiting->ToList();
  REQUIRE(waiting.size() == 2);
  CHECK(waiting[0]->priority == 2);
  CHECK(waiting[1]->priority == 3);
  CHECK(waiting[0]->request_id == "2");
  CHECK(waiting[1]->request_id == "0");
}

// test_priority_scheduling_heap_property: scheduling one at a time (finishing
// each) drains requests in ascending-priority order.
TEST_CASE("Priority scheduling: schedules in ascending priority order") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/1);
  const std::vector<int> priorities = {5, 1, 8, 3, 2, 7, 4, 6};
  std::vector<double> arrivals;
  for (std::size_t i = 0; i < priorities.size(); ++i)
    arrivals.push_back(static_cast<double>(i));
  auto requests = CreateRequestsWithPriority(
      static_cast<int>(priorities.size()), priorities, arrivals,
      /*num_tokens=*/10);
  // Map req id -> priority for the recording step.
  std::map<std::string, int> id_to_priority;
  for (auto& r : requests) id_to_priority[r->request_id] = r->priority;
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  std::vector<int> scheduled_priorities;
  while (!scheduler->waiting->empty()) {
    auto out = scheduler->schedule();
    if (!out.scheduled_new_reqs.empty()) {
      const std::string& id = out.scheduled_new_reqs[0].req_id;
      scheduled_priorities.push_back(id_to_priority[id]);
      scheduler->update_from_output(out, MakeRunnerOutput({{id, {100}}}));
      scheduler->finish_requests(id, RequestStatus::kFinishedStopped);
    }
  }

  std::vector<int> expected = priorities;
  std::sort(expected.begin(), expected.end());
  CHECK(scheduled_priorities == expected);
}

// test_priority_scheduling_preemption_and_resumption_when_out_of_kv (V2 model
// runner, no connector variant).
TEST_CASE("Priority scheduling: preemption then resumption when out of KV") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/2,
                                               /*max_num_batched_tokens=*/200,
                                               /*num_blocks=*/5,
                                               /*block_size=*/16);

  // request_low: priority 1, 30 tokens (distinct prompt via idx 0).
  auto lo = CreateRequestsWithPriority(1, {1}, {0.0}, /*num_tokens=*/30,
                                       /*max_tokens=*/16, {}, /*starting_idx=*/0);
  const std::string low_id = lo[0]->request_id;
  Request* low = lo[0].get();
  AddRequest(*scheduler, std::move(lo[0]));

  // 1st schedule.
  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(scheduler->waiting->empty());
  CHECK(scheduler->running.size() == 1);
  // 1st decode.
  scheduler->update_from_output(out, MakeRunnerOutput({{low_id, {100}}}));

  // request_high: priority 0, 32 tokens (distinct prompt via idx 1).
  auto hi = CreateRequestsWithPriority(1, {0}, {1.0}, /*num_tokens=*/32,
                                       /*max_tokens=*/16, {}, /*starting_idx=*/1);
  const std::string high_id = hi[0]->request_id;
  AddRequest(*scheduler, std::move(hi[0]));

  // 2nd schedule: KV cache becomes full (0 free blocks).
  out = scheduler->schedule();
  CHECK(scheduler->kv_cache_manager->block_pool.get_num_free_blocks() == 0);
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(scheduler->waiting->empty());
  CHECK(scheduler->running.size() == 2);
  // 2nd decode.
  scheduler->update_from_output(
      out, MakeRunnerOutput({{low_id, {100}}, {high_id, {100}}}));

  // 3rd schedule: triggers priority preemption of request_low.
  out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.empty());
  CHECK(out.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(out.scheduled_cached_reqs.req_ids[0] == high_id);
  CHECK(scheduler->requests.at(low_id)->status == RequestStatus::kPreempted);
  CHECK(scheduler->waiting->size() == 1);
  CHECK(scheduler->running.size() == 1);
  // 3rd decode: low produced nothing (preempted), high produced a token.
  scheduler->update_from_output(
      out, MakeRunnerOutput({{low_id, {}}, {high_id, {100}}}));
  // Finish high so the preempted low can resume.
  scheduler->finish_requests(high_id, RequestStatus::kFinishedStopped);

  // 4th schedule: request_low resumes. V2: folded into scheduled_new_reqs
  // carrying its full token ids in prefill_token_ids (30 prompt + 2 decoded).
  out = scheduler->schedule();
  CHECK(scheduler->waiting->empty());
  CHECK(scheduler->running.size() == 1);
  CHECK(out.scheduled_cached_reqs.num_reqs() == 0);
  REQUIRE(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == low_id);
  const auto& seed = out.scheduled_new_reqs[0].prefill_token_ids;
  REQUIRE(seed.has_value());
  CHECK(seed->size() == 32);
  CHECK((*seed)[31] == 100);
  (void)low;
}
