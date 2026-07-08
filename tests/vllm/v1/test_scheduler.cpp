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
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
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

  // These cases drive the scheduler SYNCHRONOUSLY, hand-simulating the runner
  // (they AppendOutputToken themselves without calling update_from_output), so
  // they exercise the synchronous scheduler contract regardless of the
  // VT_ASYNC_DECODE env: async_scheduling=false disables the num_output_
  // placeholders reservation (which assumes the pipeline defers update_from_output).
  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true,
                                     /*structured_output_manager=*/nullptr,
                                     /*async_scheduling=*/false);
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
