// Async depth-2 PREFILL-ADMISSION TIMING regression tests
// (row ENG-ASYNC-SCHED, async-serving spec W3).
//
// Ported-intent from: the admission contract exercised by
// tests/v1/core/test_scheduler.py (co-schedule + FCFS waiting-queue drain) and
// tests/v1/engine/test_engine_core_client.py (the busy-loop request cycle) @
// e24d1b24 — re-expressed here to pin the ONE property the W3 DGX proof put in
// question: does turning on the depth-2 `step_with_batch_queue` +
// `AsyncScheduler` DELAY when a newly-arrived request's prefill is first
// scheduled, relative to the synchronous depth-1 path (and thus relative to
// vLLM, whose single-GPU `UniProcExecutor.execute_model(non_block=True)` also
// runs the forward synchronously — uniproc_executor.py:91-106)?
//
// WHY THIS FILE EXISTS. The c16 A/B at f086b64 measured W3-on meanTTFT ≈2768 vs
// W3-off ≈2030 (+36%). The hypothesis under investigation was that our depth-2
// loop delays prefill ADMISSION. These tests drive the real EngineCore /
// EngineCoreProc step functions over the M1.4 Scheduler / AsyncScheduler with a
// canned one-token-per-request runner, injecting a request mid-stream and
// recording the exact step its prefill is first scheduled. RESULT (and the
// binding assertion below): the async depth-2 path schedules a newly-arrived
// prefill at the SAME loop iteration as the synchronous path — one step after
// arrival, with NO extra starvation — in both the unsaturated and the
// slot-saturated (max_num_seqs-bound) case. schedule() is called every busy-loop
// iteration (the batch queue is never full at entry: len <= size-1), and the
// input queue is drained before each step, exactly as vLLM's run_busy_loop
// (core.py:1259-1298). So the admission timing is 1:1 with the synchronous path
// and with vLLM; the measured TTFT delta is NOT an admission delay (see
// async-serving.md "W3 TTFT diagnosis"). These tests fail if a future change
// regresses that 1:1 admission timing.
#include <doctest/doctest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/async_scheduler.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/core_proc.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::v1::AsyncScheduler;
using vllm::v1::EngineCore;
using vllm::v1::EngineCoreProc;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::GrammarOutput;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerBase;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::Request;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

constexpr int32_t kCannedToken = 42;

// A canned one-token-per-scheduled-request runner (same seam as
// test_engine_core_proc.cpp) that ALSO records the step index at which each
// request id is first present in a scheduled batch — its "admission step".
class AdmissionRecordingRunner : public ModelRunnerBase {
 public:
  int step = 0;
  std::map<std::string, int> admit_step;

  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    ++step;
    for (const auto& [req_id, n] : scheduler_output.num_scheduled_tokens) {
      (void)n;
      admit_step.emplace(req_id, step);  // first (lowest) step only
    }
    stashed_ = scheduler_output;
    return std::nullopt;
  }

  ModelRunnerOutput sample_tokens(
      const std::optional<GrammarOutput>& /*grammar_output*/) override {
    ModelRunnerOutput mro;
    int idx = 0;
    for (const auto& [req_id, n] : stashed_.num_scheduled_tokens) {
      (void)n;
      mro.req_ids.push_back(req_id);
      mro.req_id_to_index[req_id] = idx++;
      mro.sampled_token_ids.push_back({kCannedToken});
    }
    return mro;
  }

 private:
  SchedulerOutput stashed_;
};

std::unique_ptr<Request> MakeRequest(const std::string& id, int max_tokens,
                                     int prompt_len) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  auto block_hasher = get_request_block_hasher(16, sha256_cbor);
  SamplingParams params;
  params.max_tokens = max_tokens;
  std::vector<int32_t> prompt(static_cast<size_t>(prompt_len),
                              (std::stoi(id) % 89) + 1);
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   block_hasher);
}

KVCacheConfig MakeKV() {
  KVCacheConfig kv;
  kv.num_blocks = 100000;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return kv;
}

SchedulerConfig MakeCfg(int max_num_seqs, bool async) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  cfg.async_scheduling = async;
  return cfg;
}

// Drives the engine one busy-loop iteration at a time (add_request == the
// run_busy_loop input-drain; one step == one _process_engine_step). Returns the
// step at which `probe_id` was first scheduled after it was injected.
struct AdmissionProbe {
  int inject_after_step = 0;
  int admit_step = 0;
  bool finished_probe = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// UNSATURATED: two long-running decoders in steady state, then a fresh request
// arrives. Its prefill must be scheduled at the first step after arrival, and
// the async depth-2 path must not schedule it any later than the sync path.
// ---------------------------------------------------------------------------
TEST_CASE("ENG-ASYNC-SCHED: depth-2 admits a new prefill the same step as sync (unsaturated)") {
  auto run = [](bool async) -> AdmissionProbe {
    KVCacheConfig kv = MakeKV();
    SchedulerConfig cfg = MakeCfg(/*max_num_seqs=*/64, async);
    std::unique_ptr<Scheduler> sched =
        async ? std::unique_ptr<Scheduler>(
                    new AsyncScheduler(cfg, kv, /*block_size=*/16, true))
              : std::unique_ptr<Scheduler>(
                    new Scheduler(cfg, kv, /*block_size=*/16, true));
    AdmissionRecordingRunner runner;
    Executor executor(runner);
    std::unique_ptr<EngineCore> core;
    std::unique_ptr<EngineCoreProc> proc;
    if (async) {
      proc = std::make_unique<EngineCoreProc>(*sched, executor, nullptr,
                                              /*max_concurrent_batches=*/2);
    } else {
      core = std::make_unique<EngineCore>(*sched, executor);
    }
    EngineCore& engine = async ? static_cast<EngineCore&>(*proc) : *core;
    auto step = [&]() {
      if (async) {
        (void)proc->step_with_batch_queue();
      } else {
        (void)core->step();
      }
    };

    engine.add_request(MakeRequest("100", /*max_tokens=*/1000, /*prompt=*/4));
    engine.add_request(MakeRequest("200", /*max_tokens=*/1000, /*prompt=*/4));
    for (int i = 0; i < 10; ++i) step();  // reach depth-2 steady state

    AdmissionProbe p;
    p.inject_after_step = runner.step;
    engine.add_request(MakeRequest("500", /*max_tokens=*/1000, /*prompt=*/8));
    for (int i = 0; i < 6; ++i) step();
    p.admit_step = runner.admit_step["500"];
    return p;
  };

  AdmissionProbe sync = run(false);
  AdmissionProbe async = run(true);

  // Injected after step N, first schedulable at step N+1 in BOTH paths.
  CHECK(sync.admit_step == sync.inject_after_step + 1);
  CHECK(async.admit_step == async.inject_after_step + 1);
  // The binding property: depth-2 admission is never LATER than sync.
  CHECK(async.admit_step - async.inject_after_step <=
        sync.admit_step - sync.inject_after_step);
}

// ---------------------------------------------------------------------------
// SATURATED: max_num_seqs slots are all occupied by decoders; one finishes and
// a fresh request is submitted the moment that finish is observed (closed-loop
// resubmit). Under depth-2 the finish is observed one pipeline step later than
// sync, but — measured from the probe's OWN submit — the prefill is still
// admitted at the very next schedule() in both paths: the full depth-2 queue of
// decode batches does NOT starve the waiting queue by extra steps.
// ---------------------------------------------------------------------------
TEST_CASE("ENG-ASYNC-SCHED: depth-2 does not starve the waiting queue when slots are saturated") {
  auto run = [](bool async) -> AdmissionProbe {
    KVCacheConfig kv = MakeKV();
    SchedulerConfig cfg = MakeCfg(/*max_num_seqs=*/2, async);
    std::unique_ptr<Scheduler> sched =
        async ? std::unique_ptr<Scheduler>(
                    new AsyncScheduler(cfg, kv, /*block_size=*/16, true))
              : std::unique_ptr<Scheduler>(
                    new Scheduler(cfg, kv, /*block_size=*/16, true));
    AdmissionRecordingRunner runner;
    Executor executor(runner);
    std::unique_ptr<EngineCore> core;
    std::unique_ptr<EngineCoreProc> proc;
    if (async) {
      proc = std::make_unique<EngineCoreProc>(*sched, executor, nullptr, 2);
    } else {
      core = std::make_unique<EngineCore>(*sched, executor);
    }
    EngineCore& engine = async ? static_cast<EngineCore&>(*proc) : *core;
    auto step = [&]() -> std::vector<std::string> {
      auto r = async ? proc->step_with_batch_queue() : core->step();
      std::vector<std::string> finished;
      for (auto& [ci, eco] : r.first) {
        (void)ci;
        for (auto& out : eco.outputs) {
          if (out.Finished()) finished.push_back(out.request_id);
        }
      }
      return finished;
    };

    engine.add_request(MakeRequest("100", /*max_tokens=*/1000, /*prompt=*/4));
    engine.add_request(MakeRequest("200", /*max_tokens=*/12, /*prompt=*/4));

    AdmissionProbe p;
    for (int i = 0; i < 60; ++i) {
      auto finished = step();
      for (const auto& id : finished) {
        if (id == "200" && p.inject_after_step == 0) {
          // A slot just freed; the closed-loop client resubmits immediately.
          p.inject_after_step = runner.step;
          engine.add_request(MakeRequest("500", /*max_tokens=*/20, /*prompt=*/8));
        }
      }
      if (runner.admit_step.count("500") && p.admit_step == 0) {
        p.admit_step = runner.admit_step["500"];
      }
    }
    return p;
  };

  AdmissionProbe sync = run(false);
  AdmissionProbe async = run(true);

  REQUIRE(sync.inject_after_step > 0);
  REQUIRE(async.inject_after_step > 0);
  REQUIRE(sync.admit_step > 0);
  REQUIRE(async.admit_step > 0);
  // First schedulable step after the freed slot is observed — in both paths.
  CHECK(sync.admit_step == sync.inject_after_step + 1);
  CHECK(async.admit_step == async.inject_after_step + 1);
  // No extra starvation under the depth-2 queue.
  CHECK(async.admit_step - async.inject_after_step <=
        sync.admit_step - sync.inject_after_step);
}
