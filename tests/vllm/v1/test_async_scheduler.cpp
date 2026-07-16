// Tests for the AsyncScheduler (vllm/v1/core/sched/async_scheduler.py @ e24d1b24,
// row ENG-ASYNC-SCHED, async-serving spec W3).
//
// Ported from tests/v1/core/test_async_scheduler.py @ e24d1b24:
//   - test_stop_by_max_tokens (max_tokens 1,2,3,5): the depth-2 async loop
//     (schedule two batches before processing any output) produces exactly
//     max_tokens outputs per request and schedules exactly
//     prompt + max_tokens - 1 tokens — the output-placeholder accounting keeps
//     step N+1 from over- or under-scheduling before N's tokens return.
//   - test_abort / test_preempt: aborting a scheduled request mid-flight leaves
//     each request with exactly the outputs it had produced before its abort.
//   - test_prefix_caching_for_prefill_dedup / test_prefix_caching_for_multi_turn:
//     async scheduling composes with prefix caching (all requests finish).
//   - test_abort_request_when_structured_output_fsm_cannot_advance: a grammar
//     that rejects the sampled tokens terminates the request FINISHED_ERROR and
//     removes it, composed with the async placeholder drain.
//
// Ownership deviation: our Scheduler OWNS the Request lifetime and DESTROYS a
// request when it finishes/aborts (upstream keeps it alive via the test list),
// so per-request output counts are tracked from the EngineCoreOutputs the loop
// returns rather than read off the destroyed Request. The manual depth-2 loop
// (two SchedulerOutputs in flight) mirrors the upstream deque driver exactly and
// is the scenario the placeholder accounting exists to serve.
#include <doctest/doctest.h>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/async_scheduler.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/manager.h"
#include "vllm/v1/structured_output/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::StructuredOutputsParams;
using vllm::v1::AsyncScheduler;
using vllm::v1::EngineCoreOutput;
using vllm::v1::EngineCoreOutputs;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::Request;
using vllm::v1::RequestStatus;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vllm::v1::StructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputManager;
using vllm::v1::TokenBitmask;
using vt::DType;

namespace {

// create_scheduler(async_scheduling=True) (T0 subset). Prefix caching is always
// on (M1.3 coordinator). The scheduler owns Request lifetime.
std::unique_ptr<AsyncScheduler> CreateAsyncScheduler(
    int max_num_seqs = 16, int max_num_batched_tokens = 8192,
    int num_blocks = 10000, int block_size = 16, int max_model_len = 8192,
    StructuredOutputManager* mgr = nullptr) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;
  cfg.async_scheduling = true;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<AsyncScheduler>(cfg, kv_cfg, block_size,
                                          /*enable_caching=*/true, mgr);
}

void EnsureNoneHash() {
  static bool done = false;
  if (!done) {
    init_none_hash(sha256_cbor);
    done = true;
  }
}

// create_requests: distinct prompts ([i]*num_tokens) unless same_prompt.
std::vector<std::unique_ptr<Request>> CreateRequests(int num_requests,
                                                     int num_tokens,
                                                     int max_tokens,
                                                     bool same_prompt = false,
                                                     int block_size = 16) {
  EnsureNoneHash();
  auto block_hasher = get_request_block_hasher(block_size, sha256_cbor);
  SamplingParams params;
  params.max_tokens = max_tokens;
  std::vector<std::unique_ptr<Request>> requests;
  for (int i = 0; i < num_requests; ++i) {
    const int32_t fill = same_prompt ? 7 : static_cast<int32_t>(i);
    std::vector<int32_t> prompt(static_cast<size_t>(num_tokens), fill);
    requests.push_back(std::make_unique<Request>(
        std::to_string(i), prompt, params, /*arrival_time=*/0.0, block_hasher));
  }
  return requests;
}

// _make_model_runner_output: one canned token per scheduled request, indexed in
// num_scheduled_tokens order (the token value is irrelevant to these tests).
ModelRunnerOutput MakeModelRunnerOutput(const SchedulerOutput& out) {
  ModelRunnerOutput mro;
  int idx = 0;
  for (const auto& [req_id, n] : out.num_scheduled_tokens) {
    (void)n;
    mro.req_ids.push_back(req_id);
    mro.req_id_to_index[req_id] = idx;
    mro.sampled_token_ids.push_back({idx});
    ++idx;
  }
  return mro;
}

// Accumulate produced output-token counts per request id (survives request
// destruction on finish).
void AccumulateOutputs(const EngineCoreOutputs& outs,
                       std::map<std::string, int>& counts) {
  for (const EngineCoreOutput& o : outs.outputs) {
    counts[o.request_id] += static_cast<int>(o.new_token_ids.size());
  }
}

}  // namespace

// test_stop_by_max_tokens (parametrized over max_tokens).
static void RunStopByMaxTokens(int max_tokens) {
  auto scheduler = CreateAsyncScheduler();
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/10,
                                 max_tokens);
  const int prompt_len = 10;

  int expected_total = 0;
  std::deque<SchedulerOutput> sched_outputs;
  std::map<std::string, int> out_counts;

  scheduler->add_request(std::move(requests[0]));
  sched_outputs.push_back(scheduler->schedule());
  expected_total += prompt_len + max_tokens - 1;

  scheduler->add_request(std::move(requests[1]));
  sched_outputs.push_back(scheduler->schedule());
  expected_total += prompt_len + max_tokens - 1;

  int total_scheduled = 0;
  while (!sched_outputs.empty()) {
    SchedulerOutput so = std::move(sched_outputs.front());
    sched_outputs.pop_front();
    total_scheduled += so.total_num_scheduled_tokens;
    ModelRunnerOutput mro = MakeModelRunnerOutput(so);
    AccumulateOutputs(scheduler->update_from_output(so, mro), out_counts);

    SchedulerOutput next = scheduler->schedule();
    if (!next.num_scheduled_tokens.empty()) {
      sched_outputs.push_back(std::move(next));
    }
  }

  CHECK(scheduler->get_num_unfinished_requests() == 0);
  CHECK(out_counts["0"] == max_tokens);
  CHECK(out_counts["1"] == max_tokens);
  // Ensure we are not scheduling more tokens than necessary.
  CHECK(total_scheduled == expected_total);
}

TEST_CASE("AsyncScheduler: test_stop_by_max_tokens (max_tokens 1,2,3,5)") {
  RunStopByMaxTokens(1);
  RunStopByMaxTokens(2);
  RunStopByMaxTokens(3);
  RunStopByMaxTokens(5);
}

// test_abort / test_preempt (identical upstream): abort scheduled requests in a
// fixed order; each request keeps exactly the outputs it had produced before it
// was aborted.
static void RunAbortInterleaved() {
  auto scheduler = CreateAsyncScheduler();
  const int num_requests = 10;
  auto requests = CreateRequests(num_requests, /*num_tokens=*/10,
                                 /*max_tokens=*/20);
  for (auto& req : requests) {
    scheduler->add_request(std::move(req));
  }

  std::deque<SchedulerOutput> sched_outputs;
  sched_outputs.push_back(scheduler->schedule());
  sched_outputs.push_back(scheduler->schedule());

  std::vector<int> abort_order = {0, 8, 3, 1, 6, 4, 2, 5, 7, 9};
  const std::vector<int> abort_order_copy = abort_order;
  size_t abort_pos = 0;
  std::map<std::string, int> out_counts;

  while (!sched_outputs.empty()) {
    // Abort the next scheduled request.
    if (abort_pos < abort_order.size()) {
      const std::string id = std::to_string(abort_order[abort_pos++]);
      scheduler->finish_requests(id, RequestStatus::kFinishedAborted);
    }
    SchedulerOutput so = std::move(sched_outputs.front());
    sched_outputs.pop_front();
    ModelRunnerOutput mro = MakeModelRunnerOutput(so);
    AccumulateOutputs(scheduler->update_from_output(so, mro), out_counts);

    SchedulerOutput next = scheduler->schedule();
    if (!next.num_scheduled_tokens.empty()) {
      sched_outputs.push_back(std::move(next));
    }
  }

  CHECK(scheduler->get_num_unfinished_requests() == 0);
  for (int i = 0; i < num_requests; ++i) {
    // Position of request i in the abort order == the number of outputs it
    // produced before being aborted.
    int expected = 0;
    for (size_t p = 0; p < abort_order_copy.size(); ++p) {
      if (abort_order_copy[p] == i) {
        expected = static_cast<int>(p);
        break;
      }
    }
    CHECK(out_counts[std::to_string(i)] == expected);
  }
}

TEST_CASE("AsyncScheduler: test_abort (interleaved mid-flight abort)") {
  RunAbortInterleaved();
}

TEST_CASE("AsyncScheduler: test_preempt (interleaved mid-flight abort)") {
  // Upstream test_preempt is byte-identical to test_abort.
  RunAbortInterleaved();
}

// test_prefix_caching_for_prefill_dedup: async scheduling composes with prefix
// caching. The async-relevant guarantee ported here is that every request
// finishes exactly once (no lost/duplicated in-flight frames under placeholders).
TEST_CASE("AsyncScheduler: test_prefix_caching_for_prefill_dedup (all finish)") {
  const int chunk = 1000;
  const int block_size = 16;
  const int num_prompt_tokens = 100;
  auto scheduler = CreateAsyncScheduler(/*max_num_seqs=*/16,
                                        /*max_num_batched_tokens=*/chunk,
                                        /*num_blocks=*/10000, block_size);
  auto requests = CreateRequests(/*num_requests=*/5, num_prompt_tokens,
                                 /*max_tokens=*/3, /*same_prompt=*/true,
                                 block_size);

  std::deque<SchedulerOutput> sched_outputs;
  std::map<std::string, int> out_counts;
  size_t next_to_add = 0;

  // Two requests with the same prompt.
  scheduler->add_request(std::move(requests[next_to_add++]));
  scheduler->add_request(std::move(requests[next_to_add++]));

  sched_outputs.push_back(scheduler->schedule());
  sched_outputs.push_back(scheduler->schedule());

  while (!sched_outputs.empty()) {
    if (next_to_add < requests.size()) {
      scheduler->add_request(std::move(requests[next_to_add++]));
    }
    SchedulerOutput so = std::move(sched_outputs.front());
    sched_outputs.pop_front();
    ModelRunnerOutput mro = MakeModelRunnerOutput(so);
    AccumulateOutputs(scheduler->update_from_output(so, mro), out_counts);
    SchedulerOutput next = scheduler->schedule();
    if (!next.num_scheduled_tokens.empty()) {
      sched_outputs.push_back(std::move(next));
    }
  }

  CHECK(scheduler->get_num_unfinished_requests() == 0);
  for (int i = 0; i < 5; ++i) {
    CHECK(out_counts[std::to_string(i)] == 3);  // max_tokens outputs each
  }
}

// test_prefix_caching_for_multi_turn: async scheduling drains a batch of longer
// requests to completion (all finish with their full output).
TEST_CASE("AsyncScheduler: test_prefix_caching_for_multi_turn (all finish)") {
  const int chunk = 1000;
  const int block_size = 16;
  const int num_prompt_tokens = 100;
  const int num_output_tokens = 40;  // shorter than upstream (200) — same shape
  auto scheduler = CreateAsyncScheduler(/*max_num_seqs=*/16,
                                        /*max_num_batched_tokens=*/chunk,
                                        /*num_blocks=*/10000, block_size);
  auto requests = CreateRequests(/*num_requests=*/5, num_prompt_tokens,
                                 num_output_tokens, /*same_prompt=*/false,
                                 block_size);
  for (auto& req : requests) {
    scheduler->add_request(std::move(req));
  }

  std::deque<SchedulerOutput> sched_outputs;
  std::map<std::string, int> out_counts;
  sched_outputs.push_back(scheduler->schedule());
  sched_outputs.push_back(scheduler->schedule());

  while (!sched_outputs.empty()) {
    SchedulerOutput so = std::move(sched_outputs.front());
    sched_outputs.pop_front();
    ModelRunnerOutput mro = MakeModelRunnerOutput(so);
    AccumulateOutputs(scheduler->update_from_output(so, mro), out_counts);
    SchedulerOutput next = scheduler->schedule();
    if (!next.num_scheduled_tokens.empty()) {
      sched_outputs.push_back(std::move(next));
    }
  }

  CHECK(scheduler->get_num_unfinished_requests() == 0);
  for (int i = 0; i < 5; ++i) {
    CHECK(out_counts[std::to_string(i)] == num_output_tokens);
  }
}

// ─── test_abort_request_when_structured_output_fsm_cannot_advance ────────────
namespace {

// A grammar that rejects every accept_tokens call (accept_tokens -> false),
// mirroring the upstream Mock. is_terminated stays false; the reject drives the
// FINISHED_ERROR termination.
class RejectingGrammar : public StructuredOutputGrammar {
 public:
  bool accept_tokens(const std::string&,
                     const std::vector<int32_t>&) override {
    return false;
  }
  std::vector<int32_t> validate_tokens(
      const std::vector<int32_t>& tokens) override {
    return tokens;
  }
  void rollback(int) override {}
  void fill_bitmask(TokenBitmask& bitmask, int batch_index) override {
    const int base = batch_index * bitmask.num_words;
    for (int w = 0; w < bitmask.num_words; ++w) bitmask.data[base + w] = ~0;
  }
  bool is_terminated() override { return false; }
  void reset() override {}
};

class RejectingBackend : public StructuredOutputBackend {
 public:
  explicit RejectingBackend(int vocab) : vocab_(vocab) {}
  std::unique_ptr<StructuredOutputGrammar> compile_grammar(
      vllm::v1::StructuredOutputOptions, const std::string&) override {
    return std::make_unique<RejectingGrammar>();
  }
  TokenBitmask allocate_token_bitmask(int max_num_seqs) override {
    TokenBitmask bm;
    bm.num_seqs = max_num_seqs;
    bm.num_words = vllm::v1::BitmaskWordsForVocab(vocab_);
    bm.data.assign(static_cast<size_t>(bm.num_words) *
                       static_cast<size_t>(max_num_seqs),
                   0);
    return bm;
  }
  void destroy() override {}

 private:
  int vocab_ = 0;
};

}  // namespace

TEST_CASE(
    "AsyncScheduler: test_abort_request_when_structured_output_fsm_cannot_"
    "advance") {
  StructuredOutputManager manager(/*max_num_seqs=*/16, []() {
    return std::make_unique<RejectingBackend>(/*vocab=*/100);
  });
  auto scheduler = CreateAsyncScheduler(/*max_num_seqs=*/16,
                                        /*max_num_batched_tokens=*/8192,
                                        /*num_blocks=*/10000, /*block_size=*/16,
                                        /*max_model_len=*/128, &manager);

  EnsureNoneHash();
  auto block_hasher = get_request_block_hasher(16, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  StructuredOutputsParams so;
  so.grammar = R"(root ::= "a")";
  params.structured_outputs = so;
  auto req = std::make_unique<Request>("0", std::vector<int32_t>{1}, params,
                                       /*arrival_time=*/0.0, block_hasher);
  manager.grammar_init(*req);
  const std::string req_id = req->request_id;
  scheduler->add_request(std::move(req));

  // Schedule the prefill (the request becomes RUNNING with 1 output placeholder
  // reserved for the token this step samples).
  SchedulerOutput out = scheduler->schedule();
  REQUIRE(out.num_scheduled_tokens.count(req_id) == 1);

  // The runner returns a sampled token the grammar will reject.
  ModelRunnerOutput mro;
  mro.req_ids.push_back(req_id);
  mro.req_id_to_index[req_id] = 0;
  mro.sampled_token_ids.push_back({123});
  scheduler->update_from_output(out, mro);

  // The request was terminated FINISHED_ERROR and removed from the scheduler.
  CHECK(scheduler->requests.count(req_id) == 0);
  CHECK(scheduler->running.empty());
  CHECK(scheduler->get_num_unfinished_requests() == 0);
}
