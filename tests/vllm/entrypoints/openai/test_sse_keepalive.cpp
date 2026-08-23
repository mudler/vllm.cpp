// SSE keepalive contract (VT_SERVER_SSE_PING_S) — chat + completion share
// AssignSseWaitResult / kSsePingFrame. Maint-bot #316: timeout emits a
// standalone comment frame; never concatenated with data; <=0 disables;
// a later real output still streams as a separate data frame.
//
// #931 added the WIRE-LEVEL cases at the bottom. Everything above them reads
// SsePingIntervalSec() or calls AssignSseWaitResult directly, and neither
// observes what a client receives: with the default-off branch deleted from
// both production streams, every case here still passes (measured — 7 cases,
// 34 assertions, SUCCESS, on a relinked binary). A test that measures the
// accessor cannot tell "no keepalive" from "keepalive re-armed", which is the
// only property the #931 default exists to hold.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vt/dtype.h"

using vllm::CompletionOutput;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::entrypoints::openai::AssignSseWaitResult;
using vllm::entrypoints::openai::kSsePingFrame;
using vllm::entrypoints::openai::SsePingIntervalSec;

namespace {

struct EnvRestorer {
  const char* key;
  std::optional<std::string> prev;
  explicit EnvRestorer(const char* k) : key(k) {
    if (const char* v = std::getenv(k)) prev = std::string(v);
  }
  ~EnvRestorer() {
    if (prev.has_value()) {
      ::setenv(key, prev->c_str(), 1);
    } else {
      ::unsetenv(key);
    }
  }
};

// ── The wire-level fixture (#931) ────────────────────────────────────────────
// The production streams are `CompletionSseStream` / `ChatSseStream`, both in
// anonymous namespaces, so the ONLY way to observe their framing is through
// create_completion / create_chat_completion over a real AsyncLLM — which is
// also the reachability the #931 default has to hold.
//
// The runner sleeps before it samples, so the first WaitOutput on each stream
// finds THAT request's collector empty. That is the state the two paths
// disagree about: the blocking get_output() waits and yields a data frame,
// while a get_output_for() with the interval as its timeout returns nullopt and
// emits kSsePingFrame. Without the sleep the collector is usually already
// populated and both paths return the same bytes, which is exactly why the
// default survived a full green suite.

constexpr int32_t kCannedToken = 17;  // fixture token " world"
constexpr int kRunnerDelayMs = 250;

class SlowRunnerStub : public vllm::v1::ModelRunnerBase {
 public:
  explicit SlowRunnerStub(int delay_ms = kRunnerDelayMs) : delay_ms_(delay_ms) {}

  std::optional<vllm::v1::ModelRunnerOutput> execute_model(
      const vllm::v1::SchedulerOutput& scheduler_output) override {
    stashed_ = scheduler_output;
    return std::nullopt;
  }

  vllm::v1::ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/)
      override {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
    vllm::v1::ModelRunnerOutput output;
    int index = 0;
    for (const auto& [request_id, num_tokens] : stashed_.num_scheduled_tokens) {
      (void)num_tokens;
      output.req_ids.push_back(request_id);
      output.req_id_to_index[request_id] = index++;
      output.sampled_token_ids.push_back({kCannedToken});
    }
    return output;
  }

 private:
  vllm::v1::SchedulerOutput stashed_;
  int delay_ms_;
};

vllm::tok::Tokenizer BuildFixtureTokenizer() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_sse_keepalive_tok_" + std::to_string(counter++) + ".json"))
          .string();
  nlohmann::json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = nlohmann::json::array();
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "ByteLevel"},
      {"add_prefix_space", false},
      {"trim_offsets", false},
      {"use_regex", true}};
  nlohmann::json vocab = {{"h", 0},    {"e", 1},      {"l", 2},
                          {"o", 3},    {"w", 4},      {"r", 5},
                          {"d", 6},    {"Ġ", 7},      {"1", 8},
                          {"2", 9},    {"ll", 10},    {"he", 11},
                          {"llo", 12}, {"hello", 13}, {"Ġw", 14},
                          {"or", 15},  {"orld", 16},  {"Ġworld", 17},
                          {"ld", 18}};
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges", nlohmann::json::array(
                     {nlohmann::json::array({"l", "l"}),
                      nlohmann::json::array({"h", "e"}),
                      nlohmann::json::array({"ll", "o"}),
                      nlohmann::json::array({"he", "llo"}),
                      nlohmann::json::array({"Ġ", "w"}),
                      nlohmann::json::array({"o", "r"}),
                      nlohmann::json::array({"l", "d"}),
                      nlohmann::json::array({"or", "ld"}),
                      nlohmann::json::array({"Ġw", "orld"})})}};
  {
    std::ofstream out(path, std::ios::binary);
    out << doc.dump();
  }
  vllm::tok::Tokenizer tokenizer = vllm::tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tokenizer;
}

// The production AsyncLLM frontend the HTTP server holds (server_main.cpp),
// over the delaying runner.
struct SlowAsyncHarness {
  explicit SlowAsyncHarness(int runner_delay_ms = kRunnerDelayMs)
      : tokenizer(BuildFixtureTokenizer()),
        scheduler(MakeSchedulerConfig(), MakeKvConfig(), /*block_size=*/16,
                  /*enable_caching=*/true),
        runner(runner_delay_ms),
        executor(runner),
        input_processor(tokenizer, MakeHfConfig()),
        output_processor(&tokenizer),
        engine(input_processor, scheduler, executor, output_processor,
               Hasher()) {}

  static vllm::SchedulerConfig MakeSchedulerConfig() {
    vllm::SchedulerConfig cfg;
    cfg.max_num_seqs = 8;
    cfg.max_num_batched_tokens = 8192;
    cfg.enable_chunked_prefill = true;
    cfg.max_model_len = 8192;
    cfg.watermark = 0.0;
    return cfg;
  }

  static vllm::v1::KVCacheConfig MakeKvConfig() {
    vllm::v1::KVCacheConfig kv;
    kv.num_blocks = 1024;
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"layer"},
        std::make_shared<vllm::v1::FullAttentionSpec>(16, 1, 1,
                                                      vt::DType::kF32));
    return kv;
  }

  static vllm::HfConfig MakeHfConfig() {
    vllm::HfConfig config;
    config.max_position_embeddings = 8192;
    config.raw = nlohmann::json::object();
    return config;
  }

  static vllm::v1::BlockHasher Hasher() {
    static bool initialized = false;
    if (!initialized) {
      vllm::v1::init_none_hash(vllm::v1::sha256_cbor);
      initialized = true;
    }
    return vllm::v1::get_request_block_hasher(16, vllm::v1::sha256_cbor);
  }

  vllm::tok::Tokenizer tokenizer;
  vllm::v1::Scheduler scheduler;
  SlowRunnerStub runner;
  vllm::v1::Executor executor;
  vllm::v1::InputProcessor input_processor;
  vllm::v1::OutputProcessor output_processor;
  vllm::v1::AsyncLLM engine;
};

// Drain a live production stream and REPORT what came off the wire, so a
// failure names the offending frame instead of a bare false.
struct DrainedStream {
  std::vector<std::string> chunks;
  std::vector<std::string> non_data_frames;
  int ping_frames = 0;
};

DrainedStream Drain(
    const std::shared_ptr<vllm::entrypoints::openai::SseStream>& stream) {
  DrainedStream drained;
  std::string chunk;
  while (stream->next(chunk)) {
    drained.chunks.push_back(chunk);
    if (chunk.rfind("data: ", 0) != 0) {
      drained.non_data_frames.push_back(chunk);
      if (chunk == std::string(kSsePingFrame)) ++drained.ping_frames;
    }
  }
  return drained;
}

}  // namespace

TEST_CASE("SSE keepalive: frame is a pure comment and never a data frame") {
  const std::string ping(kSsePingFrame);
  CHECK(ping == ":\n\n");
  CHECK(ping.find("data:") == std::string::npos);
  const std::string data = "data: {\"id\":\"x\"}\n\n";
  CHECK(ping + data != ping);
  CHECK(data.find(ping) == std::string::npos);
}

TEST_CASE("SSE keepalive: chat+completion timeout -> standalone ping frame") {
  RequestOutput out;
  out.request_id = "stale";
  std::string chunk = "data: SHOULD_BE_OVERWRITTEN\n\n";
  const bool got = AssignSseWaitResult(std::nullopt, out, chunk);
  CHECK_FALSE(got);
  CHECK(chunk == std::string(kSsePingFrame));
  CHECK(chunk == ":\n\n");
  CHECK(out.request_id == "stale");
}

TEST_CASE("SSE keepalive: chat+completion data path fills out with no ping rewrite") {
  RequestOutput ready;
  ready.request_id = "req-1";
  ready.finished = true;
  RequestOutput out;
  std::string chunk = "untouched";
  const bool got = AssignSseWaitResult(std::move(ready), out, chunk);
  CHECK(got);
  CHECK(out.request_id == "req-1");
  CHECK(out.finished);
  CHECK(chunk == "untouched");
  CHECK(chunk != std::string(kSsePingFrame));
}

TEST_CASE("SSE keepalive: later real output is a separate framing step") {
  std::vector<std::string> frames;
  {
    RequestOutput out;
    std::string chunk;
    CHECK_FALSE(AssignSseWaitResult(std::nullopt, out, chunk));
    frames.push_back(chunk);
  }
  {
    RequestOutput ready;
    ready.request_id = "req-2";
    CompletionOutput co;
    co.index = 0;
    co.text = "hi";
    ready.outputs.push_back(std::move(co));
    RequestOutput out;
    std::string chunk;
    CHECK(AssignSseWaitResult(std::move(ready), out, chunk));
    CHECK(out.request_id == "req-2");
    REQUIRE(out.outputs.size() == 1);
    CHECK(out.outputs[0].text == "hi");
    frames.push_back("data: " + out.outputs[0].text + "\n\n");
  }
  REQUIRE(frames.size() == 2);
  CHECK(frames[0] == std::string(kSsePingFrame));
  CHECK(frames[1].rfind("data: ", 0) == 0);
  // Two discrete frames — next() never returns ping+data concatenated.
  CHECK(frames[0] + frames[1] != frames[0]);
  CHECK(frames[1] != frames[0] + frames[1]);
}

// #931. The default was 15 seconds, and that put a byte on the wire that the
// reference never emits.
//
// vLLM serves its streaming completions through uvicorn/starlette and yields
// nothing but `data: ` frames: grepping the pinned oracle
// (0.23.1rc1.dev1511+g555967922) for a yielded comment across ALL of
// vllm/entrypoints/ returns no hit. So a comment frame is not a mirrored
// behaviour, it is an invention — and vLLM's OWN benchmark client cannot
// survive one. `vllm bench serve` strips each network chunk before parsing
// (benchmarks/lib/endpoint_request_func.py:207), which destroys the "\n\n"
// separator at chunk boundaries, and its only resynchronisation path requires a
// `data: ` prefix (:48). A comment arriving before the first data frame
// therefore poisons its buffer permanently: it reports "Never received a valid
// chunk to calculate TTFT" and counts the request FAILED, while the server
// completes it normally and logs nothing.
//
// That is not hypothetical. It voided the 27B c16 leg at 93/96 (#577) and then
// the Qwen3.8-27B c1 and c8 legs at 5/6 and 36/48 (#931, #915) — in both cases
// taking exactly the SLOWEST requests, because those are the ones that reach a
// 15 s silence. In the #931 legs the failed requests' imputed TTFT was 92-94 s
// (c1, all three reps) against a p99 of 4.2 s among the successes: the timeout
// separated the two populations precisely.
//
// So the default mirrors vLLM — no keepalive — and #316's capability stays
// reachable for a deployment behind a proxy that needs it, as an opt-in whose
// cost is now stated.
// The case name carries NO COMMA. `doctest -tc=` splits its filter on commas,
// so a scoped rerun of a comma-bearing name matches nothing, runs 0 cases and
// still prints "Status: SUCCESS!" with exit 0 — a mutation pass over this file
// would read green without executing anything.
TEST_CASE("SSE keepalive: DEFAULT OFF mirroring vLLM's frame set") {
  EnvRestorer rest("VT_SERVER_SSE_PING_S");
  ::unsetenv("VT_SERVER_SSE_PING_S");
  CHECK(SsePingIntervalSec() == 0);
  ::setenv("VT_SERVER_SSE_PING_S", "", 1);
  CHECK(SsePingIntervalSec() == 0);
  // An unparsable spelling falls back to the DEFAULT, and the default is off.
  // It used to fall back to 15, so a typo silently enabled the frame.
  ::setenv("VT_SERVER_SSE_PING_S", "fifteen", 1);
  CHECK(SsePingIntervalSec() == 0);
}

TEST_CASE("SSE keepalive: VT_SERVER_SSE_PING_S <=0 disables") {
  EnvRestorer rest("VT_SERVER_SSE_PING_S");
  ::unsetenv("VT_SERVER_SSE_PING_S");
  CHECK(SsePingIntervalSec() == 0);
  ::setenv("VT_SERVER_SSE_PING_S", "0", 1);
  CHECK(SsePingIntervalSec() == 0);
  ::setenv("VT_SERVER_SSE_PING_S", "-3", 1);
  CHECK(SsePingIntervalSec() == 0);
  ::setenv("VT_SERVER_SSE_PING_S", "2", 1);
  CHECK(SsePingIntervalSec() == 2);
  ::setenv("VT_SERVER_SSE_PING_S", "9999", 1);
  CHECK(SsePingIntervalSec() == 600);
}

TEST_CASE("SSE keepalive: collector get_for timeout then later deliver") {
  using namespace std::chrono_literals;
  vllm::v1::RequestOutputCollector c(RequestOutputKind::kDelta, "sse0");
  CHECK_FALSE(c.get_for(15ms).has_value());
  std::thread th([&] {
    std::this_thread::sleep_for(25ms);
    RequestOutput out;
    out.request_id = "sse0";
    out.finished = true;
    c.put(std::move(out));
  });
  auto hit = c.get_for(500ms);
  th.join();
  REQUIRE(hit.has_value());
  CHECK(hit->request_id == "sse0");
}

// ── WIRE LEVEL (#931) ────────────────────────────────────────────────────────
// What #915's client saw was not an interval, it was a BYTE. These two cases
// assert the byte, on a live production stream, with the request's collector
// held empty across the whole first wait.
//
// They are the cases that fall over when the default-off branch is deleted from
// CompletionSseStream::WaitOutput / ChatSseStream::WaitOutput: the deleted
// branch is what turns an interval of 0 into "block", and without it
// get_output_for(request, 0ms) times out on the empty collector and puts
// kSsePingFrame — the frame `vllm bench serve` cannot resynchronise past — in
// front of the first data frame. That is byte-for-byte the #915 failure.
//
// vLLM's own streams yield `data: ` frames only: grepping the pinned oracle
// (0.23.1rc1.dev1511+g555967922) across vllm/entrypoints/ for a yielded comment
// returns no hit, so "no non-data frame" IS the mirrored wire contract.

TEST_CASE("SSE keepalive: a silent collector yields no comment frame (completion)") {
  EnvRestorer rest("VT_SERVER_SSE_PING_S");
  ::unsetenv("VT_SERVER_SSE_PING_S");
  REQUIRE(SsePingIntervalSec() == 0);

  SlowAsyncHarness h;
  vllm::entrypoints::openai::OpenAIServingCompletion completion(h.engine,
                                                                "test-model");
  vllm::entrypoints::openai::CompletionRequest request;
  request.prompt = "hello";
  request.max_tokens = 3;
  request.temperature = 0.0;
  request.stream = true;

  vllm::entrypoints::openai::CompletionResult result =
      completion.create_completion(request);
  REQUIRE(result.streaming);
  REQUIRE(result.sse_stream != nullptr);

  const DrainedStream drained = Drain(result.sse_stream);
  // A stream that produced nothing would satisfy "no comment frame" vacuously.
  REQUIRE(drained.chunks.size() >= 2);
  CHECK(drained.chunks.back() == "data: [DONE]\n\n");
  CHECK(drained.ping_frames == 0);
  for (const std::string& frame : drained.non_data_frames) {
    FAIL_CHECK("completion stream emitted a non-data frame: " << frame);
  }
  CHECK(drained.non_data_frames.empty());
}

TEST_CASE("SSE keepalive: a silent collector yields no comment frame (chat)") {
  EnvRestorer rest("VT_SERVER_SSE_PING_S");
  ::unsetenv("VT_SERVER_SSE_PING_S");
  REQUIRE(SsePingIntervalSec() == 0);

  SlowAsyncHarness h;
  // The default role-join fallback renders "user: hello\nassistant:", whose
  // characters exceed this minimal BPE fixture's vocabulary. ChatPromptFn is
  // the documented renderer seam (serving_chat.h), so the fixture supplies one
  // that emits a prompt the fixture can encode; the framing under test is
  // downstream of the prompt and unaffected by its text.
  vllm::entrypoints::openai::OpenAIServingChat chat(
      h.engine, "test-model",
      [](const std::vector<vllm::entrypoints::openai::ChatMessage>&, bool,
         const std::vector<vllm::entrypoints::openai::ChatCompletionToolsParam>&,
         const nlohmann::ordered_json&) -> std::string { return "hello"; });
  vllm::entrypoints::openai::ChatCompletionRequest request;
  vllm::entrypoints::openai::ChatMessage message;
  message.role = "user";
  message.content = "hello";
  request.messages.push_back(std::move(message));
  request.max_tokens = 3;
  request.temperature = 0.0;
  request.stream = true;

  vllm::entrypoints::openai::ChatCompletionResult result =
      chat.create_chat_completion(request);
  REQUIRE(result.streaming);
  REQUIRE(result.sse_stream != nullptr);

  const DrainedStream drained = Drain(result.sse_stream);
  REQUIRE(drained.chunks.size() >= 2);
  CHECK(drained.chunks.back() == "data: [DONE]\n\n");
  CHECK(drained.ping_frames == 0);
  for (const std::string& frame : drained.non_data_frames) {
    FAIL_CHECK("chat stream emitted a non-data frame: " << frame);
  }
  CHECK(drained.non_data_frames.empty());
}

// The opt-in half of the same contract: #316's capability must still be
// reachable, or the "default off, still available" claim is untested. With a
// positive interval and the same silent collector, the comment frame DOES
// arrive, ahead of the first data frame.
TEST_CASE("SSE keepalive: a positive interval still pings a silent collector") {
  EnvRestorer rest("VT_SERVER_SSE_PING_S");
  ::setenv("VT_SERVER_SSE_PING_S", "1", 1);
  REQUIRE(SsePingIntervalSec() == 1);

  // 1500 ms per step against a 1 s interval, so the FIRST wait is guaranteed to
  // expire and the ping is not a timing coincidence. Asserting a positive count
  // matters: `ping_frames == 0` would also be reported by a build where the
  // opt-in path silently stopped working, which is the shape this case exists
  // to exclude.
  SlowAsyncHarness h(/*runner_delay_ms=*/1500);
  vllm::entrypoints::openai::OpenAIServingCompletion completion(h.engine,
                                                                "test-model");
  vllm::entrypoints::openai::CompletionRequest request;
  request.prompt = "hello";
  request.max_tokens = 2;
  request.temperature = 0.0;
  request.stream = true;

  vllm::entrypoints::openai::CompletionResult result =
      completion.create_completion(request);
  REQUIRE(result.sse_stream != nullptr);

  const DrainedStream drained = Drain(result.sse_stream);
  REQUIRE(drained.chunks.size() >= 2);
  CHECK(drained.chunks.back() == "data: [DONE]\n\n");
  // #316's capability is still REACHABLE at the opt-in, which is what makes
  // "default off, still available" a claim rather than a hope.
  CHECK(drained.ping_frames >= 1);
  // ...and every non-data frame is exactly the comment, never merged into data.
  CHECK(drained.ping_frames == drained.non_data_frames.size());
  for (const std::string& frame : drained.non_data_frames) {
    CHECK(frame == std::string(kSsePingFrame));
  }
}
