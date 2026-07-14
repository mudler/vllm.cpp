// Ported from: tests/v1/engine/test_async_llm.py @ e24d1b24
// W2 T-unit cases from specs/async-serving.md: test_load (:109), test_abort
// (:157), test_multi_abort (:228), test_finished_flag (:306),
// test_mid_stream_cancellation (:340), and test_abort_final_output (:598).
//
// The upstream suite drives a tiny real model under asyncio. This C++ port
// drives the real Scheduler/EngineCoreProc/OutputProcessor pipeline with the
// same canned one-token-per-step ModelRunnerBase seam used by
// test_engine_core_proc.cpp. It tests the asynchronous queue/collector and
// request lifecycle without GPU/model dependencies; GPU token-exactness and
// live-SSE arrival timing remain W2's explicit G1/G3 gates.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::AsyncLLM;
using vllm::v1::AsyncRequest;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::InputProcessor;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerBase;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::OutputProcessor;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

constexpr int32_t kCannedToken = 17;  // fixture token " world"

class RunnerStub : public ModelRunnerBase {
 public:
  explicit RunnerStub(std::chrono::microseconds delay = {}) : delay_(delay) {}

  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    stashed_output_ = scheduler_output;
    return std::nullopt;
  }

  ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/) override {
    if (delay_.count() != 0) std::this_thread::sleep_for(delay_);
    ModelRunnerOutput output;
    int index = 0;
    for (const auto& [request_id, num_tokens] :
         stashed_output_.num_scheduled_tokens) {
      (void)num_tokens;
      output.req_ids.push_back(request_id);
      output.req_id_to_index[request_id] = index++;
      output.sampled_token_ids.push_back({kCannedToken});
    }
    return output;
  }

 private:
  SchedulerOutput stashed_output_;
  std::chrono::microseconds delay_;
};

// A runner whose engine-thread sampling hook throws, so the busy-loop fatal
// guard (core_client.cpp) is the only place holding the true root-cause string.
class ThrowingRunnerStub : public ModelRunnerBase {
 public:
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    (void)scheduler_output;
    return std::nullopt;
  }

  ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/) override {
    throw std::runtime_error(
        "vt: DIAG_ROOT_CAUSE_SENTINEL at qwen3_5.cpp:0");
  }
};

// Scope-guarded process-wide std::cerr redirect. Installed BEFORE the engine
// starts and restored only when this object is destroyed; sequencing the
// restore after the engine (and its joined threads) guarantees no engine-thread
// write races the rdbuf swap.
class CerrRedirect {
 public:
  explicit CerrRedirect(std::streambuf* target)
      : previous_(std::cerr.rdbuf(target)) {}
  ~CerrRedirect() { std::cerr.rdbuf(previous_); }
  CerrRedirect(const CerrRedirect&) = delete;
  CerrRedirect& operator=(const CerrRedirect&) = delete;

 private:
  std::streambuf* previous_;
};

std::unique_ptr<Scheduler> CreateScheduler() {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 128;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;

  KVCacheConfig kv;
  kv.num_blocks = 10000;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, 1, 1, DType::kF32));
  return std::make_unique<Scheduler>(cfg, kv, 16,
                                     /*enable_caching=*/true);
}

Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_async_llm_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array();
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},    {"l", 2},     {"o", 3},
                {"w", 4},   {"r", 5},    {"d", 6},     {"Ġ", 7},
                {"1", 8},   {"2", 9},    {"ll", 10},   {"he", 11},
                {"llo", 12}, {"hello", 13}, {"Ġw", 14}, {"or", 15},
                {"orld", 16}, {"Ġworld", 17}, {"ld", 18}};
  vocab[MapBytesToUnicode("\xF0\x9F")] = 19;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 20;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  {
    std::ofstream out(path, std::ios::binary);
    out << doc.dump();
  }
  Tokenizer tokenizer = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tokenizer;
}

HfConfig MakeConfig() {
  HfConfig config;
  config.max_position_embeddings = 8192;
  config.raw = json::object();
  return config;
}

SamplingParams Params(int max_tokens, RequestOutputKind output_kind) {
  SamplingParams params;
  params.max_tokens = max_tokens;
  params.output_kind = output_kind;
  params.temperature = 0.0;
  return params;
}

int Drain(AsyncLLM& engine, const AsyncRequest& request,
          std::vector<RequestOutput>* outputs = nullptr) {
  int tokens = 0;
  for (;;) {
    RequestOutput output = engine.get_output(request);
    REQUIRE(output.outputs.size() == 1);
    tokens += static_cast<int>(output.outputs[0].token_ids.size());
    if (outputs != nullptr) outputs->push_back(output);
    if (output.finished) return tokens;
  }
}

void InitHash() {
  static bool initialized = false;
  if (!initialized) {
    init_none_hash(sha256_cbor);
    initialized = true;
  }
}

}  // namespace

TEST_CASE("async_llm test_load: concurrent requests all finish with unique ids") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  constexpr int kNumRequests = 32;
  constexpr int kTokens = 10;
  std::vector<AsyncRequest> requests;
  std::set<std::string> ids;
  for (int i = 0; i < kNumRequests; ++i) {
    const std::string id = "request-" + std::to_string(i);
    requests.push_back(engine.add_request(
        id, "hello", Params(kTokens, RequestOutputKind::kDelta)));
    ids.insert(requests.back().request_id);
  }
  CHECK(ids.size() == kNumRequests);
  for (const AsyncRequest& request : requests) {
    CHECK(Drain(engine, request) == kTokens);
  }
  CHECK_FALSE(engine.has_unfinished_requests());
}

TEST_CASE("async_llm test_abort and test_multi_abort leave other requests healthy") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::microseconds(200));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  AsyncRequest abort_a = engine.add_request(
      "abort-a", "hello", Params(100000, RequestOutputKind::kDelta));
  AsyncRequest survivor = engine.add_request(
      "survivor", "hello", Params(8, RequestOutputKind::kDelta));
  AsyncRequest abort_b = engine.add_request(
      "abort-b", "hello", Params(100000, RequestOutputKind::kFinalOnly));

  // Let abort-a produce at least one partial delta, then multi-abort both ids.
  RequestOutput partial = engine.get_output(abort_a);
  CHECK_FALSE(partial.finished);
  engine.abort(std::vector<std::string>{"abort-a", "abort-b"});

  RequestOutput final_a = engine.get_output(abort_a);
  CHECK(final_a.finished);
  CHECK(*final_a.outputs[0].finish_reason == "abort");
  RequestOutput final_b = engine.get_output(abort_b);
  CHECK(final_b.finished);
  CHECK(*final_b.outputs[0].finish_reason == "abort");
  CHECK(Drain(engine, survivor) == 8);
  CHECK_FALSE(engine.has_unfinished_requests());

  // Reusing an aborted external id is valid once cleanup completes.
  AsyncRequest reused = engine.add_request(
      "abort-a", "hello", Params(3, RequestOutputKind::kDelta));
  CHECK(Drain(engine, reused) == 3);
}

TEST_CASE("async_llm test_finished_flag and mid_stream_cancellation") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::milliseconds(1));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  AsyncRequest request = engine.add_request(
      "cancel", "hello", Params(1000, RequestOutputKind::kDelta));
  std::vector<RequestOutput> seen;
  int tokens = 0;
  while (tokens < 5) {
    RequestOutput output_value = engine.get_output(request);
    CHECK_FALSE(output_value.finished);
    tokens += static_cast<int>(output_value.outputs[0].token_ids.size());
    seen.push_back(std::move(output_value));
  }
  engine.abort("cancel");
  RequestOutput final_output = engine.get_output(request);
  CHECK(final_output.finished);
  CHECK(*final_output.outputs[0].finish_reason == "abort");
  CHECK(tokens >= 5);
  CHECK_FALSE(engine.has_unfinished_requests());
}

TEST_CASE("async_llm test_abort_final_output returns terminal metadata") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::milliseconds(1));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  AsyncRequest request = engine.add_request(
      "abort-final", "hello",
      Params(3000, RequestOutputKind::kFinalOnly));
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  engine.abort(request.request_id);
  RequestOutput final_output = engine.get_output(request);
  CHECK(final_output.finished);
  REQUIRE(final_output.outputs.size() == 1);
  REQUIRE(final_output.outputs[0].finish_reason.has_value());
  CHECK(*final_output.outputs[0].finish_reason == "abort");
  CHECK_FALSE(final_output.outputs[0].stop_reason.has_value());
  CHECK_FALSE(engine.has_unfinished_requests());
}

TEST_CASE("async_llm rejects a duplicate live request id without replacing its collector") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::milliseconds(1));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  AsyncRequest original = engine.add_request(
      "duplicate", "hello", Params(1000, RequestOutputKind::kDelta));
  CHECK_THROWS_AS(
      engine.add_request("duplicate", "hello",
                         Params(1, RequestOutputKind::kFinalOnly)),
      std::invalid_argument);

  engine.abort(original.request_id);
  RequestOutput final_output = engine.get_output(original);
  CHECK(final_output.finished);
  REQUIRE(final_output.outputs.size() == 1);
  REQUIRE(final_output.outputs[0].finish_reason.has_value());
  CHECK(*final_output.outputs[0].finish_reason == "abort");
  CHECK_FALSE(engine.has_unfinished_requests());
}

TEST_CASE("async_llm shutdown wakes an active request with a terminal abort") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::milliseconds(1));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);

  AsyncRequest request;
  {
    auto engine = std::make_unique<AsyncLLM>(
        input, *scheduler, executor, output,
        get_request_block_hasher(16, sha256_cbor));
    request = engine->add_request(
        "shutdown", "hello", Params(1000, RequestOutputKind::kDelta));
  }

  RequestOutput final_output = request.collector->get();
  CHECK(final_output.finished);
  REQUIRE(final_output.outputs.size() == 1);
  REQUIRE(final_output.outputs[0].finish_reason.has_value());
  CHECK(*final_output.outputs[0].finish_reason == "abort");
  CHECK_FALSE(output.has_unfinished_requests());
}

TEST_CASE("async_llm concurrent submission cannot publish after shutdown sweep") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();

  // Exercise both valid outcomes of the admission boundary repeatedly: the
  // request is accepted before shutdown and receives a terminal output, or
  // shutdown wins and add_request raises EngineDeadError. In neither case may
  // a collector be registered after shutdown's abort-all sweep and hang.
  for (int attempt = 0; attempt < 64; ++attempt) {
    auto scheduler = CreateScheduler();
    RunnerStub runner(std::chrono::milliseconds(1));
    Executor executor(runner);
    InputProcessor input(tokenizer, config);
    OutputProcessor output(&tokenizer);
    AsyncLLM engine(input, *scheduler, executor, output,
                    get_request_block_hasher(16, sha256_cbor));

    std::optional<AsyncRequest> accepted;
    std::exception_ptr submit_error;
    std::thread submitter([&] {
      try {
        accepted = engine.add_request(
            "shutdown-race", "hello",
            Params(1000, RequestOutputKind::kDelta));
      } catch (...) {
        submit_error = std::current_exception();
      }
    });

    engine.shutdown();
    submitter.join();

    if (accepted.has_value()) {
      RequestOutput terminal = accepted->collector->get();
      CHECK(terminal.finished);
    } else {
      REQUIRE(submit_error != nullptr);
      CHECK_THROWS_AS(std::rethrow_exception(submit_error),
                      vllm::v1::EngineDeadError);
    }
    CHECK_FALSE(output.has_unfinished_requests());
  }
}

TEST_CASE(
    "async_llm engine-thread guard logs the raw root cause before poisoning") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();

  // Capture everything the engine thread emits, but assert only after the
  // engine (and both its joined threads) are gone so the rdbuf restore cannot
  // race an in-flight write and so no doctest failure text is swallowed.
  std::ostringstream captured;
  bool saw_engine_dead = false;
  std::string second_add_what;
  bool second_add_threw = false;
  {
    CerrRedirect redirect(captured.rdbuf());
    auto scheduler = CreateScheduler();
    ThrowingRunnerStub runner;
    Executor executor(runner);
    InputProcessor input(tokenizer, config);
    OutputProcessor output(&tokenizer);
    AsyncLLM engine(input, *scheduler, executor, output,
                    get_request_block_hasher(16, sha256_cbor));

    AsyncRequest request = engine.add_request(
        "diag", "hello", Params(128, RequestOutputKind::kDelta));

    // Drive the consumer until the poisoned engine surfaces EngineDeadError.
    for (int i = 0; i < 2000 && !saw_engine_dead; ++i) {
      try {
        (void)engine.get_output(request);
      } catch (const vllm::v1::EngineDeadError&) {
        saw_engine_dead = true;
      } catch (...) {
        // Any other terminal exception also ends the drive loop.
        saw_engine_dead = true;
      }
    }

    // A subsequent submission must fast-fail with the generic wrapper, never
    // leaking the raw root cause into the client-facing error.
    try {
      (void)engine.add_request("diag-after", "hello",
                               Params(1, RequestOutputKind::kDelta));
    } catch (const vllm::v1::EngineDeadError& e) {
      second_add_threw = true;
      second_add_what = e.what();
    }
  }

  const std::string log = captured.str();
  CHECK(saw_engine_dead);
  CHECK(log.find("engine-fatal:") != std::string::npos);
  CHECK(log.find("DIAG_ROOT_CAUSE_SENTINEL") != std::string::npos);
  CHECK(second_add_threw);
  CHECK(second_add_what.find("EngineCore encountered an issue") !=
        std::string::npos);
  CHECK(second_add_what.find("DIAG_ROOT_CAUSE_SENTINEL") == std::string::npos);
}
