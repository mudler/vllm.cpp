// Tests for the EngineCoreProc busy loop + input/output queue split and the
// in-proc client (vllm/v1/engine/core.py:915-916,1259-1480 +
// vllm/v1/engine/core_client.py:779-893 @ e24d1b24, row ENG-CORE-BUSY-LOOP,
// async-serving spec W1).
//
// Ported from: tests/v1/engine/test_engine_core_client.py @ e24d1b24 —
// test_engine_core_client (:523, the sync/multiprocessing path whose in-proc
// analog this client is): the normal request cycle (all requests finish with
// exactly MAX_TOKENS outputs pulled via get_output), the abort cycle (aborted
// requests produce fewer outputs, un-aborted ones are unaffected), and
// abort-after-finished being a no-op; the pull loop mirrors loop_until_done
// (:266). The upstream module's DP / utility-call / kv-events / startup cases
// are deferred with their features (see core_proc.h DEFERRED). NOTE: the
// async-serving spec's "Tests to port" table binds no dedicated W1 module (the
// AsyncLLM suites are W2); this file carries the applicable upstream
// client-cycle assertions per test-porting.md rule 1, re-expressed T-unit over
// the same RunnerStub seam as test_engine_core.cpp (real M1.4 Scheduler + a
// canned one-token-per-step ModelRunnerBase double; upstream runs a model).
//
// Busy-loop/shutdown semantics tested directly against core.py:
//   - WAKEUP sentinel unblocks an idle input_queue.get() (:1204-1222, :1377).
//   - abort-mode shutdown (shutdown_timeout 0) aborts in-flight requests and
//     sends their abort outputs before exiting (:1330-1349).
//   - drain-mode shutdown (timeout > 0) lets in-flight requests finish
//     (:1351-1358).
//   - ADD during shutdown is rejected with an abort output (:1407-1416).
//   - a fatal busy-loop error posts ENGINE_CORE_DEAD; get_output raises
//     EngineDeadError (:1229-1235, :1470-1480; core_client.py:454-457,849-859).
//   - max_concurrent_batches > 1 is rejected while step_with_batch_queue is
//     deferred (core.py:196-223 selection; ENG-ASYNC-SCHED W3).
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/config/speculative.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/async_scheduler.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vllm/v1/engine/core_client.h"
#include "vllm/v1/engine/core_proc.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::v1::EngineCoreInputItem;
using vllm::v1::EngineCoreOutputItem;
using vllm::v1::EngineCoreOutputs;
using vllm::v1::EngineCoreProc;
using vllm::v1::EngineCoreRequestType;
using vllm::v1::EngineDeadError;
using vllm::v1::EngineShutdownState;
using vllm::v1::Executor;
using vllm::v1::FinishReason;
using vllm::v1::FullAttentionSpec;
using vllm::v1::BlockingQueue;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::InprocClient;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerBase;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::PublishEngineCoreInputWaveAtomically;
using vllm::v1::Request;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

struct MoveGate {
  std::atomic<bool> armed{false};
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
  std::atomic<bool> throw_on_move{false};
};

struct GatedMoveItem {
  GatedMoveItem() = default;
  GatedMoveItem(int value, std::shared_ptr<MoveGate> gate)
      : value(value), gate(std::move(gate)) {}
  GatedMoveItem(const GatedMoveItem&) = delete;
  GatedMoveItem& operator=(const GatedMoveItem&) = delete;

  GatedMoveItem(GatedMoveItem&& other)
      : value(other.value), gate(std::move(other.gate)) {
    if (gate != nullptr && value == 2 && gate->armed.load()) {
      gate->entered.store(true);
      while (!gate->release.load()) std::this_thread::yield();
      if (gate->throw_on_move.load()) {
        throw std::runtime_error("injected batch element move failure");
      }
    }
  }

  GatedMoveItem& operator=(GatedMoveItem&& other) noexcept {
    value = other.value;
    gate = std::move(other.gate);
    return *this;
  }

  int value = 0;
  std::shared_ptr<MoveGate> gate;
};

struct CountingConditionVariable {
  void notify_one() { ++notifications; }

  template <typename Lock, typename Predicate>
  void wait(Lock&, Predicate&&) {}

  static inline int notifications = 0;
};

struct PutManyOnlyQueue {
  int publish_calls = 0;
  std::vector<int> published;

  void put_many_nowait(std::vector<int> items) {
    ++publish_calls;
    published = std::move(items);
  }
};

// The canned token the stub "samples" for every scheduled request (same seam
// as test_engine_core.cpp; upstream drives a real tiny model).
constexpr int32_t kCannedToken = 42;

// MRV2 execute_model/sample_tokens double: one canned token per scheduled
// request per step. Only ever driven from the engine thread.
class RunnerStub : public ModelRunnerBase {
 public:
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    stashed_output_ = scheduler_output;
    return std::nullopt;  // MRV2: forward done, no output yet.
  }

  ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/) override {
    ModelRunnerOutput mro;
    int idx = 0;
    for (const auto& [req_id, n] : stashed_output_.num_scheduled_tokens) {
      mro.req_ids.push_back(req_id);
      mro.req_id_to_index[req_id] = idx++;
      mro.sampled_token_ids.push_back({kCannedToken});
    }
    return mro;
  }

 private:
  SchedulerOutput stashed_output_;
};

// A double whose forward pass dies: drives the ENGINE_CORE_DEAD path
// (core.py:1229-1235 fatal-error guard).
class ThrowingRunnerStub : public ModelRunnerBase {
 public:
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& /*scheduler_output*/) override {
    throw std::runtime_error("injected forward failure");
  }
  ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/) override {
    return {};
  }
};

// Mirror of the engine-core tests' create_scheduler (T0 subset).
std::unique_ptr<Scheduler> CreateScheduler(int max_num_seqs = 16,
                                           int max_num_batched_tokens = 8192) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = 10000;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));

  return std::make_unique<Scheduler>(cfg, kv_cfg, /*block_size=*/16,
                                     /*enable_caching=*/true);
}

// A short-prompt request (prefills in one step); mirrors make_request in the
// upstream client test (SamplingParams(max_tokens=...)).
std::unique_ptr<Request> MakeRequest(const std::string& id, int max_tokens) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  auto block_hasher = get_request_block_hasher(16, sha256_cbor);
  SamplingParams params;
  params.max_tokens = max_tokens;
  std::vector<int32_t> prompt(4, /*value=*/std::stoi(id) + 1);
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   block_hasher);
}

// loop_until_done (tests/v1/engine/test_engine_core_client.py:266), tracked
// per request id: pull get_output() frames, appending each EngineCoreOutput to
// outputs[request_id], until every id in `want` has produced a finished
// output. Returns the collected frames.
void LoopUntilDone(InprocClient& client, const std::set<std::string>& want,
                   std::map<std::string, std::vector<vllm::v1::EngineCoreOutput>>&
                       outputs) {
  std::set<std::string> finished;
  while (finished.size() < want.size()) {
    EngineCoreOutputs eco = client.get_output();
    for (auto& out : eco.outputs) {
      if (out.Finished() && want.count(out.request_id) != 0) {
        finished.insert(out.request_id);
      }
      outputs[out.request_id].push_back(std::move(out));
    }
  }
}

}  // namespace

TEST_CASE("engine core wave route owns one atomic put-many publish") {
  PutManyOnlyQueue queue;
  PublishEngineCoreInputWaveAtomically(queue, std::vector<int>{4, 5, 6});

  CHECK(queue.publish_calls == 1);
  CHECK(queue.published == std::vector<int>{4, 5, 6});
}

TEST_CASE("BlockingQueue batch publish is all-at-once and ordered") {
  BlockingQueue<GatedMoveItem> queue;
  const auto gate = std::make_shared<MoveGate>();
  std::vector<GatedMoveItem> batch;
  batch.emplace_back(1, gate);
  batch.emplace_back(2, gate);
  batch.emplace_back(3, gate);
  gate->armed.store(true);

  auto producer = std::async(std::launch::async, [&] {
    queue.put_many_nowait(std::move(batch));
  });
  for (int i = 0; i < 100 && !gate->entered.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  REQUIRE(gate->entered.load());

  // The first element has already been appended, but a consumer must not be
  // able to acquire the queue mutex until the complete wave is present.
  auto consumer = std::async(std::launch::async, [&] {
    GatedMoveItem item;
    const bool found = queue.try_get(item);
    return std::make_pair(found, item.value);
  });
  CHECK(consumer.wait_for(std::chrono::milliseconds(20)) ==
        std::future_status::timeout);

  gate->release.store(true);
  producer.get();
  const auto first = consumer.get();
  CHECK(first == std::make_pair(true, 1));

  GatedMoveItem item;
  REQUIRE(queue.try_get(item));
  CHECK(item.value == 2);
  REQUIRE(queue.try_get(item));
  CHECK(item.value == 3);
  CHECK_FALSE(queue.try_get(item));
}

TEST_CASE("BlockingQueue batch publish rolls back an appended suffix on failure") {
  BlockingQueue<GatedMoveItem> queue;
  const auto gate = std::make_shared<MoveGate>();
  queue.put_nowait(GatedMoveItem(9, gate));

  std::vector<GatedMoveItem> batch;
  batch.emplace_back(1, gate);
  batch.emplace_back(2, gate);
  batch.emplace_back(3, gate);
  gate->armed.store(true);
  gate->release.store(true);
  gate->throw_on_move.store(true);
  CHECK_THROWS_AS(queue.put_many_nowait(std::move(batch)), std::runtime_error);

  gate->armed.store(false);
  GatedMoveItem item;
  REQUIRE(queue.try_get(item));
  CHECK(item.value == 9);
  CHECK_FALSE(queue.try_get(item));
}

TEST_CASE("BlockingQueue successful batch emits one notification") {
  CountingConditionVariable::notifications = 0;
  BlockingQueue<int, CountingConditionVariable> queue;
  queue.put_many_nowait(std::vector<int>{1, 2, 3});
  CHECK(CountingConditionVariable::notifications == 1);
}

// ---------------------------------------------------------------------------
// Normal request cycle (test_engine_core_client.py:548-562): add N requests,
// pull outputs off the busy loop until all finish; every request produced
// exactly MAX_TOKENS outputs (one canned token per step, finish by length).
// ---------------------------------------------------------------------------
TEST_CASE("engine_core_client in-proc: normal request cycle over the busy loop") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  InprocClient client(*scheduler, executor);

  constexpr int kMaxTokens = 20;
  constexpr int kNumRequests = 10;
  std::set<std::string> ids;
  for (int i = 0; i < kNumRequests; ++i) {
    std::string id = std::to_string(i);
    client.add_request_async(MakeRequest(id, kMaxTokens));
    ids.insert(id);
  }

  std::map<std::string, std::vector<vllm::v1::EngineCoreOutput>> outputs;
  LoopUntilDone(client, ids, outputs);

  for (const std::string& id : ids) {
    // Upstream: assert len(outputs[req_id]) == MAX_TOKENS.
    CHECK(outputs[id].size() == kMaxTokens);
    int total_tokens = 0;
    for (const auto& out : outputs[id]) {
      total_tokens += static_cast<int>(out.new_token_ids.size());
    }
    CHECK(total_tokens == kMaxTokens);
    // The last frame finished by length (max_tokens reached).
    REQUIRE(outputs[id].back().Finished());
    CHECK(*outputs[id].back().finish_reason == FinishReason::kLength);
  }

  client.shutdown();
  CHECK_FALSE(client.engine_dead());
}

// ---------------------------------------------------------------------------
// Abort request cycle (test_engine_core_client.py:563-585): abort half the
// requests right after adding them; aborted requests produce fewer than
// MAX_TOKENS outputs, the others still produce exactly MAX_TOKENS.
// ---------------------------------------------------------------------------
TEST_CASE("engine_core_client in-proc: abort request cycle") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  InprocClient client(*scheduler, executor);

  constexpr int kMaxTokens = 20;
  constexpr int kNumRequests = 10;
  std::set<std::string> surviving;
  std::vector<std::string> aborted;
  for (int i = 0; i < kNumRequests; ++i) {
    std::string id = std::to_string(i);
    client.add_request_async(MakeRequest(id, kMaxTokens));
    if (i % 2 == 0) {
      // Upstream aborts even-indexed requests right after adding them.
      client.abort_requests_async({id});
      aborted.push_back(id);
    } else {
      surviving.insert(id);
    }
  }

  std::map<std::string, std::vector<vllm::v1::EngineCoreOutput>> outputs;
  LoopUntilDone(client, surviving, outputs);

  for (const std::string& id : aborted) {
    // Aborted before finishing: strictly fewer outputs (possibly zero).
    CHECK(outputs[id].size() < kMaxTokens);
  }
  for (const std::string& id : surviving) {
    CHECK(outputs[id].size() == kMaxTokens);
  }

  client.shutdown();
}

// ---------------------------------------------------------------------------
// Abort after a request finished (test_engine_core_client.py:585-597): a
// no-op — and the engine keeps serving new requests afterwards.
// ---------------------------------------------------------------------------
TEST_CASE("engine_core_client in-proc: abort after finished is a no-op") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  InprocClient client(*scheduler, executor);

  std::map<std::string, std::vector<vllm::v1::EngineCoreOutput>> outputs;
  client.add_request_async(MakeRequest("0", /*max_tokens=*/5));
  LoopUntilDone(client, {"0"}, outputs);
  CHECK(outputs["0"].size() == 5);

  // Abort the already-finished request: unknown/finished ids are a no-op
  // (scheduler.finish_requests ignores them).
  client.abort_requests_async({"0"});

  // Engine is still alive and serving.
  client.add_request_async(MakeRequest("1", /*max_tokens=*/3));
  LoopUntilDone(client, {"1"}, outputs);
  CHECK(outputs["1"].size() == 3);
  CHECK(*outputs["1"].back().finish_reason == FinishReason::kLength);

  client.shutdown();
  CHECK_FALSE(client.engine_dead());
}

// ---------------------------------------------------------------------------
// WAKEUP sentinel (core.py:1204-1222 wakeup_engine + :1377-1378): shutdown of
// an IDLE engine — whose busy loop is blocked in input_queue.get() — returns
// promptly instead of hanging.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCoreProc: WAKEUP sentinel unblocks idle shutdown") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  InprocClient client(*scheduler, executor);

  client.shutdown();  // joins; hangs here if the sentinel were dropped.
  CHECK_FALSE(client.engine_dead());
  CHECK_FALSE(client.proc().is_running());
}

// ---------------------------------------------------------------------------
// Abort-mode shutdown (core.py:1330-1349, shutdown_timeout == 0): in-flight
// requests are finished as FINISHED_ABORTED and their abort outputs are sent
// before the loop exits (_send_abort_outputs :1734-1742 -> :1714-1722).
// ---------------------------------------------------------------------------
TEST_CASE("EngineCoreProc: abort-mode shutdown aborts in-flight requests") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  InprocClient client(*scheduler, executor, nullptr,
                      /*max_concurrent_batches=*/1, /*shutdown_timeout_s=*/0);

  client.add_request_async(MakeRequest("0", /*max_tokens=*/100000));
  // Ensure the request was admitted (first token pulled) before requesting
  // shutdown, so the ADD cannot race the shutdown-reject path.
  EngineCoreOutputs first = client.get_output();
  REQUIRE(first.outputs.size() == 1);
  CHECK_FALSE(first.outputs[0].Finished());

  client.shutdown();

  // The abort output for the in-flight request is on the queue (it was
  // enqueued before the busy loop exited). Frames before it are token deltas.
  bool abort_seen = false;
  for (int i = 0; i < 1000 && !abort_seen; ++i) {
    EngineCoreOutputItem item;
    REQUIRE(client.proc().output_queue.try_get(item));
    for (const auto& out : item.outputs.outputs) {
      if (out.request_id == "0" && out.Finished()) {
        CHECK(*out.finish_reason == FinishReason::kAbort);
        CHECK(out.new_token_ids.empty());
        abort_seen = true;
      }
    }
  }
  CHECK(abort_seen);
  // The scheduler was left empty (request removed on abort).
  CHECK(scheduler->get_num_unfinished_requests() == 0);
}

// ---------------------------------------------------------------------------
// Drain-mode shutdown (core.py:1351-1358, shutdown_timeout > 0): in-flight
// requests run to completion before the loop exits.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCoreProc: drain-mode shutdown finishes in-flight requests") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  InprocClient client(*scheduler, executor, nullptr,
                      /*max_concurrent_batches=*/1, /*shutdown_timeout_s=*/60);

  constexpr int kMaxTokens = 8;
  client.add_request_async(MakeRequest("0", kMaxTokens));
  // Admit the request first (same anti-race as above), then request shutdown
  // while it still has 7 tokens to go.
  EngineCoreOutputs first = client.get_output();
  REQUIRE(first.outputs.size() == 1);
  int total_tokens = static_cast<int>(first.outputs[0].new_token_ids.size());

  client.shutdown();

  // Drain the queue: the request finished by LENGTH (not abort) with all its
  // tokens — the drain let it complete.
  bool finished_seen = false;
  EngineCoreOutputItem item;
  while (client.proc().output_queue.try_get(item)) {
    for (const auto& out : item.outputs.outputs) {
      if (out.request_id != "0") {
        continue;
      }
      total_tokens += static_cast<int>(out.new_token_ids.size());
      if (out.Finished()) {
        CHECK(*out.finish_reason == FinishReason::kLength);
        finished_seen = true;
      }
    }
  }
  CHECK(finished_seen);
  CHECK(total_tokens == kMaxTokens);
}

// ---------------------------------------------------------------------------
// ADD during shutdown is rejected with an abort output (core.py:1407-1416).
// Driven single-threaded for determinism: run_busy_loop() on this thread with
// shutdown already REQUESTED, one in-flight request, and one queued ADD.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCoreProc: run_busy_loop rejects ADD during shutdown with an abort output") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCoreProc proc(*scheduler, executor);

  // In-flight request admitted directly (EngineCore::add_request).
  proc.add_request(MakeRequest("0", /*max_tokens=*/100));
  // A second request still sitting in the input queue as an ADD.
  EngineCoreInputItem add;
  add.type = EngineCoreRequestType::kAdd;
  add.request = MakeRequest("1", /*max_tokens=*/100);
  proc.input_queue.put_nowait(std::move(add));

  // Shutdown already requested (abort mode): the loop must abort "0", reject
  // "1" with an abort output, flush, and exit — all on this thread.
  proc.shutdown_state.store(EngineShutdownState::kRequested);
  proc.run_busy_loop();

  CHECK_FALSE(proc.is_running());
  CHECK_FALSE(proc.has_work());

  std::set<std::string> abort_ids;
  EngineCoreOutputItem item;
  while (proc.output_queue.try_get(item)) {
    CHECK_FALSE(item.engine_dead);
    for (const auto& out : item.outputs.outputs) {
      REQUIRE(out.Finished());
      CHECK(*out.finish_reason == FinishReason::kAbort);
      abort_ids.insert(out.request_id);
    }
  }
  CHECK(abort_ids == std::set<std::string>{"0", "1"});
}

// ---------------------------------------------------------------------------
// Fatal busy-loop error: ENGINE_CORE_DEAD is posted and get_output raises
// EngineDeadError (core.py:1229-1235, :1470-1480; core_client.py:454-457,
// :849-859); a later abort_requests_async is a no-op (:891-893).
// ---------------------------------------------------------------------------
TEST_CASE("engine_core_client in-proc: fatal error surfaces as EngineDeadError") {
  auto scheduler = CreateScheduler();
  ThrowingRunnerStub runner;
  Executor executor(runner);
  InprocClient client(*scheduler, executor);

  client.add_request_async(MakeRequest("0", /*max_tokens=*/4));

  CHECK_THROWS_AS((void)client.get_output(), EngineDeadError);
  CHECK(client.engine_dead());

  // Guarded no-op on a dead engine (nothing consumes the queue anymore).
  client.abort_requests_async({"0"});
  CHECK(client.proc().input_queue.empty());

  client.shutdown();  // thread already exited; join must not hang.
}

// ---------------------------------------------------------------------------
// step_fn selection (core.py:196-223): max_concurrent_batches > 1 now selects
// step_with_batch_queue (ENG-ASYNC-SCHED, spec W3) — depth-2 overlap. A depth
// < 1 is rejected; depth 2 is accepted and constructs cleanly.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCoreProc: max_concurrent_batches selection (>=1 accepted, <1 rejected)") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);

  // Depth 2 (async scheduling) is now supported: step_with_batch_queue.
  CHECK_NOTHROW(EngineCoreProc(*scheduler, executor, nullptr,
                               /*max_concurrent_batches=*/2));
  // Depth < 1 is invalid.
  CHECK_THROWS_AS(EngineCoreProc(*scheduler, executor, nullptr,
                                 /*max_concurrent_batches=*/0),
                  std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Depth-2 overlap cycle (core.py:519-632, ENG-ASYNC-SCHED W3): the AsyncScheduler
// + step_with_batch_queue busy loop drains N concurrent requests to completion.
// Each request must still produce exactly MAX_TOKENS outputs (the placeholder
// accounting keeps the depth-2 schedule token-exact vs the synchronous path).
// ---------------------------------------------------------------------------
TEST_CASE("EngineCoreProc: depth-2 async overlap cycle finishes every request exactly") {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  cfg.async_scheduling = true;
  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = 10000;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  vllm::v1::AsyncScheduler scheduler(cfg, kv_cfg, /*block_size=*/16,
                                     /*enable_caching=*/true);

  RunnerStub runner;
  Executor executor(runner);
  InprocClient client(scheduler, executor, /*structured_output_manager=*/nullptr,
                      /*max_concurrent_batches=*/2);

  constexpr int kMaxTokens = 20;
  constexpr int kNumRequests = 10;
  std::set<std::string> ids;
  for (int i = 0; i < kNumRequests; ++i) {
    std::string id = std::to_string(i);
    client.add_request_async(MakeRequest(id, kMaxTokens));
    ids.insert(id);
  }

  std::map<std::string, std::vector<vllm::v1::EngineCoreOutput>> outputs;
  LoopUntilDone(client, ids, outputs);

  for (const std::string& id : ids) {
    int total_tokens = 0;
    for (const auto& out : outputs[id]) {
      total_tokens += static_cast<int>(out.new_token_ids.size());
    }
    CHECK(total_tokens == kMaxTokens);
  }
  client.shutdown();
}

// ═══════════════════════════════════════════════════════════════════════════
// SPEC-DFLASH2 W7 (#1824): the async draft-in-output ENGINE contract, at the
// ModelRunnerBase seam. Under async scheduling the engine must NOT pull drafts
// out-of-band (post_step is guarded, core.py:617 @ 555967922 — "we update
// draft token ids in the worker process"); the scheduler ships -1 placeholders
// and the WORKER substitutes the real values it kept. Under the synchronous
// scheduler the out-of-band take_draft_token_ids -> update_draft_token_ids
// flow stays exactly as SPEC-MTP I2 landed it. Both flows must emit the SAME
// token stream — spec decode is lossless.
//
// RED-first: before W7 the async run FAILED three ways — take_calls read >0
// (core_proc.cpp called post_step unguarded, installing real drafts under
// async), saw_placeholder_values read false (the placeholders were therefore
// never scheduled), and the AsyncScheduler could not even be constructed with
// a SpeculativeConfig (no such ctor parameter).
// ═══════════════════════════════════════════════════════════════════════════
namespace {

// A deterministic drafting model double. Request `id`'s true output at
// position p is Tok(id, p); the drafter proposes the true token except at
// every third position (one wrong draft), so per-step acceptance cycles
// through 0..k and the rejection-rollback arithmetic is exercised at every
// value. Verification is accept-iff-equal, exactly the greedy contract.
class SpecRunnerStub : public ModelRunnerBase {
 public:
  explicit SpecRunnerStub(int k) : k_(k) {}

  static int32_t Tok(const std::string& id, int p) {
    return 100 + (std::stoi(id) * 31 + p) % 23;
  }
  int32_t Draft(const std::string& id, int p) const {
    const int32_t t = Tok(id, p);
    return (p % 3 == 2) ? t + 1 : t;
  }

  // Read by the TEST THREAD after client.shutdown() (doctest assertions are
  // not thread-safe, so the engine thread only records).
  int take_calls = 0;
  bool saw_real_draft_values = false;
  bool saw_placeholder_values = false;
  bool fill_miss = false;        // placeholder row with no drafts of our own
  bool value_mismatch = false;   // sync-installed values != what we proposed
  int drafted_verify_steps = 0;

  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    stashed_ = scheduler_output;
    return std::nullopt;
  }

  ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/) override {
    ModelRunnerOutput mro;
    int idx = 0;
    std::map<std::string, std::vector<int32_t>> proposed_now;
    for (const auto& [req_id, n] : stashed_.num_scheduled_tokens) {
      (void)n;
      mro.req_ids.push_back(req_id);
      mro.req_id_to_index[req_id] = idx++;
      int& pos = out_pos_[req_id];  // output tokens emitted so far

      // The drafts this step verifies: REAL values under the sync contract
      // (post_step installed them), -1 placeholders under async (the worker —
      // this stub — holds the real ones and substitutes, mirroring
      // gpu_input_batch.py:520-523 "placeholders ... overwritten").
      std::vector<int32_t> drafts;
      const auto it = stashed_.scheduled_spec_decode_tokens.find(req_id);
      if (it != stashed_.scheduled_spec_decode_tokens.end()) {
        drafts = it->second;
        bool has_placeholder = false;
        for (const int32_t d : drafts) has_placeholder |= (d == -1);
        const auto own = own_drafts_.find(req_id);
        if (has_placeholder) {
          saw_placeholder_values = true;
          if (own == own_drafts_.end() || own->second.size() < drafts.size()) {
            fill_miss = true;
            drafts.clear();
          } else {
            std::copy(own->second.begin(),
                      own->second.begin() +
                          static_cast<std::ptrdiff_t>(drafts.size()),
                      drafts.begin());
          }
        } else if (!drafts.empty()) {
          saw_real_draft_values = true;
          if (own == own_drafts_.end() || own->second.size() < drafts.size()) {
            value_mismatch = true;
          } else {
            for (std::size_t j = 0; j < drafts.size(); ++j) {
              value_mismatch |= (drafts[j] != own->second[j]);
            }
          }
        }
      }

      std::vector<int32_t> emitted;
      if (drafts.empty()) {
        emitted.push_back(Tok(req_id, pos));  // prefill / no-draft decode
      } else {
        ++drafted_verify_steps;
        int accepted = 0;
        for (std::size_t j = 0; j < drafts.size(); ++j) {
          if (drafts[j] == Tok(req_id, pos + static_cast<int>(j))) {
            ++accepted;
          } else {
            break;
          }
        }
        for (int i = 0; i <= accepted; ++i) {
          emitted.push_back(Tok(req_id, pos + i));
        }
      }
      pos += static_cast<int>(emitted.size());
      mro.sampled_token_ids.push_back(std::move(emitted));

      // Propose the next k drafts (positions pos..pos+k-1).
      std::vector<int32_t> next;
      next.reserve(static_cast<std::size_t>(k_));
      for (int i = 0; i < k_; ++i) {
        next.push_back(Draft(req_id, pos + i));
      }
      proposed_now[req_id] = next;
    }
    // Stash for take_draft_token_ids (sync) AND for our own async fill.
    vllm::v1::DraftTokenIds fresh;
    for (auto& [rid, toks] : proposed_now) {
      fresh.req_ids.push_back(rid);
      fresh.draft_token_ids.push_back(toks);
      own_drafts_[rid] = std::move(toks);
    }
    pending_ = std::move(fresh);
    return mro;
  }

  std::optional<vllm::v1::DraftTokenIds> take_draft_token_ids() override {
    ++take_calls;
    std::optional<vllm::v1::DraftTokenIds> out = std::move(pending_);
    pending_.reset();
    return out;
  }

 private:
  const int k_;
  SchedulerOutput stashed_;
  std::map<std::string, int> out_pos_;
  std::map<std::string, std::vector<int32_t>> own_drafts_;
  std::optional<vllm::v1::DraftTokenIds> pending_;
};

// The sync sibling of the async spec scheduler below.
std::unique_ptr<Scheduler> CreateSyncSpecScheduler(int k) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  cfg.async_scheduling = false;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = 10000;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<Scheduler>(
      cfg, kv_cfg, /*block_size=*/16, /*enable_caching=*/true,
      /*structured_output_manager=*/nullptr,
      vllm::SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/1, k));
}

std::unique_ptr<vllm::v1::AsyncScheduler> CreateAsyncSpecScheduler(int k) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  cfg.async_scheduling = true;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = 10000;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<vllm::v1::AsyncScheduler>(
      cfg, kv_cfg, /*block_size=*/16, /*enable_caching=*/true,
      /*structured_output_manager=*/nullptr,
      vllm::SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/1, k));
}

// One full run over `sched`: kNumReqs requests to completion; returns the
// per-request token streams.
std::map<std::string, std::vector<int32_t>> RunSpecCycle(
    Scheduler& sched, SpecRunnerStub& runner, int max_concurrent_batches,
    int num_reqs, int max_tokens) {
  Executor executor(runner);
  InprocClient client(sched, executor, /*structured_output_manager=*/nullptr,
                      max_concurrent_batches, /*shutdown_timeout_s=*/0,
                      /*check_for_draft_tokens=*/true);
  std::set<std::string> ids;
  for (int i = 0; i < num_reqs; ++i) {
    std::string id = std::to_string(i);
    client.add_request_async(MakeRequest(id, max_tokens));
    ids.insert(id);
  }
  std::map<std::string, std::vector<vllm::v1::EngineCoreOutput>> outputs;
  LoopUntilDone(client, ids, outputs);
  client.shutdown();

  std::map<std::string, std::vector<int32_t>> streams;
  for (const auto& [id, frames] : outputs) {
    for (const auto& f : frames) {
      streams[id].insert(streams[id].end(), f.new_token_ids.begin(),
                         f.new_token_ids.end());
    }
  }
  return streams;
}

}  // namespace

TEST_CASE("W7 (#1824): async scheduling ships -1 placeholders, never pulls "
          "drafts out-of-band, and emits the sync flow's exact tokens") {
  constexpr int kK = 2;
  constexpr int kNumReqs = 3;
  constexpr int kMaxTokens = 13;

  // The expected stream is the stub's own true continuation — spec decode is
  // lossless, so BOTH flows must emit exactly this.
  std::map<std::string, std::vector<int32_t>> expected;
  for (int i = 0; i < kNumReqs; ++i) {
    const std::string id = std::to_string(i);
    for (int p = 0; p < kMaxTokens; ++p) {
      expected[id].push_back(SpecRunnerStub::Tok(id, p));
    }
  }

  // Arm 1 — the synchronous flow (SPEC-MTP I2): post_step pulls the drafts
  // out-of-band and the scheduler carries REAL values.
  SpecRunnerStub sync_runner(kK);
  auto sync_sched = CreateSyncSpecScheduler(kK);
  const auto sync_streams = RunSpecCycle(*sync_sched, sync_runner,
                                         /*max_concurrent_batches=*/1,
                                         kNumReqs, kMaxTokens);
  CHECK(sync_streams == expected);
  CHECK(sync_runner.take_calls > 0);
  CHECK(sync_runner.saw_real_draft_values);
  CHECK_FALSE(sync_runner.saw_placeholder_values);
  CHECK_FALSE(sync_runner.value_mismatch);
  CHECK(sync_runner.drafted_verify_steps > 0);

  // Arm 2 — the async draft-in-output flow (this wave): the scheduler ships
  // placeholders, the worker fills, and the engine NEVER pulls out-of-band
  // (no structured output is scheduled, so the deferred-grammar pull — the
  // one legitimate async take site, core.py:718-731 — never runs either).
  SpecRunnerStub async_runner(kK);
  auto async_sched = CreateAsyncSpecScheduler(kK);
  const auto async_streams = RunSpecCycle(*async_sched, async_runner,
                                          /*max_concurrent_batches=*/2,
                                          kNumReqs, kMaxTokens);
  CHECK(async_streams == expected);
  CHECK(async_streams == sync_streams);
  CHECK(async_runner.take_calls == 0);
  CHECK(async_runner.saw_placeholder_values);
  CHECK_FALSE(async_runner.saw_real_draft_values);
  CHECK_FALSE(async_runner.fill_miss);
  CHECK(async_runner.drafted_verify_steps > 0);
}
