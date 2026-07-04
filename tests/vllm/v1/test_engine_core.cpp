// Tests for the V1 EngineCore step loop (vllm/v1/engine/core.py @ e24d1b24)
// and the Executor pass-through seam (vllm/v1/executor/abstract.py +
// uniproc_executor.py @ e24d1b24).
//
// M1.8 Task 1 is a pure control-flow port (no goldens): drive EngineCore.step()
// over the REAL M1.4 Scheduler + a runner STUB (a ModelRunnerBase test double
// that records the SchedulerOutput it received and returns a canned
// ModelRunnerOutput — one sampled token per scheduled request). Cases:
//   (a) empty scheduler        -> step() returns ({}, false); runner not called.
//   (b) one request scheduled  -> execute_model + sample_tokens called (in that
//                                 order) with the scheduled SchedulerOutput;
//                                 update_from_output fed the ModelRunnerOutput
//                                 (the sampled token appended, EngineCoreOutputs
//                                 returned); model_executed == true.
//   (c) abort_requests         -> the request is finished (removed from the
//                                 scheduler; next step early-returns).
//   (d) call ORDER              -> schedule -> execute -> sample -> update.
#include <doctest/doctest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::v1::EngineCore;
using vllm::v1::EngineCoreOutputs;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
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

// The canned token the stub "samples" for every scheduled request.
constexpr int32_t kCannedToken = 42;

// A ModelRunnerBase test double mirroring the MRV2 execute_model / sample_tokens
// split: execute_model STASHES the SchedulerOutput and returns nullopt ("forward
// done, no output yet"); sample_tokens builds a ModelRunnerOutput (one canned
// token per scheduled request) from the stashed step, exactly as the real runner
// samples from the hidden states it stashed. Records call counts + order so the
// loop's schedule -> execute -> sample -> update sequence is checkable.
class RunnerStub : public ModelRunnerBase {
 public:
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    ++execute_calls;
    execute_order = ++global_seq_;
    // Record what the forward saw (proves the Executor forwarded the scheduled
    // output unchanged).
    last_total_scheduled = scheduler_output.total_num_scheduled_tokens;
    last_scheduled_ids.clear();
    for (const auto& [req_id, n] : scheduler_output.num_scheduled_tokens) {
      last_scheduled_ids.push_back(req_id);
    }
    stashed_output_ = scheduler_output;  // stash for sample_tokens.
    return std::nullopt;                 // MRV2: forward done, no output yet.
  }

  ModelRunnerOutput sample_tokens() override {
    ++sample_calls;
    sample_order = ++global_seq_;
    // Build the ModelRunnerOutput from the stashed step: one canned token per
    // scheduled request (the test keeps prefills single-step, so every
    // scheduled request completes and samples this step).
    ModelRunnerOutput mro;
    int idx = 0;
    for (const auto& [req_id, n] : stashed_output_.num_scheduled_tokens) {
      mro.req_ids.push_back(req_id);
      mro.req_id_to_index[req_id] = idx++;
      mro.sampled_token_ids.push_back({kCannedToken});
    }
    return mro;
  }

  int execute_calls = 0;
  int sample_calls = 0;
  int execute_order = 0;
  int sample_order = 0;
  int last_total_scheduled = 0;
  std::vector<std::string> last_scheduled_ids;

 private:
  SchedulerOutput stashed_output_;
  int global_seq_ = 0;
};

// Mirror of the scheduler tests' create_scheduler (T0 subset): caching ON
// (M1.3 only supports the prefix-caching coordinator), large budget so short
// prompts prefill in a single step.
std::unique_ptr<Scheduler> CreateScheduler(int max_num_seqs = 16,
                                           int max_num_batched_tokens = 8192,
                                           int num_blocks = 10000,
                                           int block_size = 16,
                                           int max_model_len = 8192) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));

  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true);
}

// A single request with a short prompt (prefills in one step under the budget).
std::unique_ptr<Request> MakeRequest(const std::string& id, int num_tokens = 4) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  const int block_size = 16;
  auto block_hasher = get_request_block_hasher(block_size, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  std::vector<int32_t> prompt(num_tokens, /*value=*/std::stoi(id) + 1);
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   block_hasher);
}

}  // namespace

// ---------------------------------------------------------------------------
// (a) Empty scheduler: step() early-returns ({}, false) and never touches the
// runner (core.py:488 `if not self.scheduler.has_requests(): return {}, False`).
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.step: empty scheduler returns ({}, false), no runner call") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);

  auto [outputs, model_executed] = engine.step();

  CHECK(outputs.empty());
  CHECK_FALSE(model_executed);
  CHECK(runner.execute_calls == 0);
  CHECK(runner.sample_calls == 0);
}

// ---------------------------------------------------------------------------
// (b) One request: schedule -> execute_model -> sample_tokens -> update_from_
// output. The Executor forwards the scheduled SchedulerOutput unchanged, the
// runner's ModelRunnerOutput is fed to update_from_output (token appended), and
// the EngineCoreOutputs carry the sampled token; model_executed == true.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.step: one request drives the full schedule/exec/sample/update loop") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);

  engine.add_request(MakeRequest("0"));
  Request* req = scheduler->requests.at("0").get();

  auto [outputs, model_executed] = engine.step();

  // Runner driven exactly once through each seam method.
  CHECK(runner.execute_calls == 1);
  CHECK(runner.sample_calls == 1);
  // The Executor forwarded the scheduled step to the forward: the prompt's 4
  // tokens were scheduled for request "0".
  CHECK(runner.last_total_scheduled == 4);
  REQUIRE(runner.last_scheduled_ids.size() == 1);
  CHECK(runner.last_scheduled_ids[0] == "0");

  // Model executed (total_num_scheduled_tokens > 0).
  CHECK(model_executed);

  // update_from_output was fed the runner's ModelRunnerOutput: the canned token
  // was appended to the request...
  REQUIRE(req->output_token_ids.size() == 1);
  CHECK(req->output_token_ids[0] == kCannedToken);
  // ...and surfaced in the per-client EngineCoreOutputs map (single client 0).
  REQUIRE(outputs.size() == 1);
  const EngineCoreOutputs& eco = outputs.at(0);
  REQUIRE(eco.outputs.size() == 1);
  CHECK(eco.outputs[0].request_id == "0");
  REQUIRE(eco.outputs[0].new_token_ids.size() == 1);
  CHECK(eco.outputs[0].new_token_ids[0] == kCannedToken);
  CHECK_FALSE(eco.outputs[0].Finished());  // max_tokens 16, not finished yet.
}

// ---------------------------------------------------------------------------
// (d) Call ORDER: execute_model precedes sample_tokens within step(); and a
// second step continues the request (decode) after the first sampled a token.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.step: execute precedes sample; a second step decodes") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);

  engine.add_request(MakeRequest("0"));
  Request* req = scheduler->requests.at("0").get();

  engine.step();  // prefill + first sample.
  CHECK(runner.execute_order < runner.sample_order);  // execute before sample.
  REQUIRE(req->output_token_ids.size() == 1);

  // Second step: the request is now running; it decodes one token.
  auto [outputs, model_executed] = engine.step();
  CHECK(model_executed);
  CHECK(runner.execute_calls == 2);
  CHECK(runner.sample_calls == 2);
  CHECK(runner.last_total_scheduled == 1);  // one decode token this step.
  REQUIRE(req->output_token_ids.size() == 2);
  CHECK(req->output_token_ids[1] == kCannedToken);
}

// ---------------------------------------------------------------------------
// (c) Abort: abort_requests(FINISHED_ABORTED) finishes the request; it is
// removed from the scheduler and the next step early-returns.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.abort_requests: finishes the request; next step is empty") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);

  engine.add_request(MakeRequest("0"));
  REQUIRE(scheduler->get_num_unfinished_requests() == 1);

  engine.abort_requests({"0"});

  // The request was finished + removed (finish_requests erases it from requests_
  // and records it in finished_req_ids).
  CHECK(scheduler->get_num_unfinished_requests() == 0);
  CHECK(scheduler->requests.find("0") == scheduler->requests.end());
  // finished_req_ids keeps has_requests() true for exactly one more step (so the
  // runner can clear the finished req).
  CHECK(scheduler->has_finished_requests());

  // Flush step: nothing to schedule (total 0), so model_executed is false and no
  // token outputs are produced (empty map). schedule() flushes finished_req_ids.
  auto [flush_outputs, flush_executed] = engine.step();
  CHECK_FALSE(flush_executed);
  CHECK(flush_outputs.empty());
  CHECK_FALSE(scheduler->has_finished_requests());

  // Engine is now quiesced: the next step early-returns without touching the
  // runner again (only the flush step drove the runner's empty batch).
  auto [outputs, model_executed] = engine.step();
  CHECK(outputs.empty());
  CHECK_FALSE(model_executed);
  CHECK(runner.execute_calls == 1);
  CHECK(runner.sample_calls == 1);
}
