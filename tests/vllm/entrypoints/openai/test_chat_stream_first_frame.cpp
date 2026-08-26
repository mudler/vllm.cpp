// #1982 — WHEN the first `/v1/chat/completions` SSE frame reaches the client.
//
// Upstream builds the role chunk under `if first_iteration:` INSIDE
// `async for res in result_generator:`
// (vllm/entrypoints/openai/chat_completion/serving.py:477,487 @ 555967922) and
// says why at :484-486: "We need to do it here, because if there are exceptions
// in the result_generator, it needs to be sent as the FIRST response (by the
// try...catch)."
//
// Ours wrote that frame before it read the engine at all, and two things follow.
//
//  1. A request that dies before its first token had already been answered 200
//     with a role frame, so the error could not be the first response.
//  2. `vllm/benchmarks/lib/endpoint_request_func.py:404-408` stamps TTFT on the
//     first chunk carrying a `choices` key, REGARDLESS of whether
//     `delta.content` is empty:
//
//         if choices := data.get("choices"):
//             content = choices[0]["delta"].get("content")
//             # First token
//             if ttft == 0.0:
//                 ttft = timestamp - st
//
//     Our role frame carries `delta.content = ""` and no `usage`, so it
//     satisfies that guard. Any TTFT taken against this endpoint with
//     `vllm bench serve --backend openai-chat` was the HTTP round trip to an
//     empty frame — near zero, and independent of load.
//
// WHAT THESE CASES MEASURE, in words, because a role-frame SHAPE assertion
// passes on both sides of the fix and proves nothing: the discriminating
// assertion is the NEGATIVE one, taken while the model runner is held inside
// sample_tokens so that NO token can exist yet. The runner counts its own
// sampled steps, so each case asserts that precondition rather than assuming
// it.
//
// Both cases enter through `ApiServer::handle_chat_completions`, which is what
// the production `/v1/chat/completions` route calls. `ChatSseStream` lives in an
// anonymous namespace and cannot be named from a test, so there is no way to
// reach this code except the way a client reaches it.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/openai/api_server.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/outputs.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/core_client.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vt/dtype.h"

namespace oai = vllm::entrypoints::openai;
using json = nlohmann::json;

namespace {

constexpr int32_t kCannedToken = 17;  // fixture token " world"

// A runner the TEST clocks. sample_tokens blocks until release() is called, and
// sampled_steps() reports how many times it has produced tokens. That counter
// is the instrument's own precondition: while it reads 0, no token exists
// anywhere in the engine, so any SSE frame observed at that moment was produced
// without engine output.
//
// The wait carries a bounded timeout so a regression cannot hang the suite; the
// timeout is a safety net and never the path a passing run takes.
class GatedRunnerStub : public vllm::v1::ModelRunnerBase {
 public:
  std::optional<vllm::v1::ModelRunnerOutput> execute_model(
      const vllm::v1::SchedulerOutput& scheduler_output) override {
    stashed_ = scheduler_output;
    return std::nullopt;
  }

  vllm::v1::ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/)
      override {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      released_cv_.wait_for(lock, std::chrono::seconds(30),
                            [this] { return released_; });
    }
    vllm::v1::ModelRunnerOutput output;
    int index = 0;
    for (const auto& [request_id, num_tokens] : stashed_.num_scheduled_tokens) {
      (void)num_tokens;
      output.req_ids.push_back(request_id);
      output.req_id_to_index[request_id] = index++;
      output.sampled_token_ids.push_back({kCannedToken});
    }
    sampled_steps_.fetch_add(1);
    return output;
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    released_cv_.notify_all();
  }

  int sampled_steps() const { return sampled_steps_.load(); }

 private:
  vllm::v1::SchedulerOutput stashed_;
  std::mutex mutex_;
  std::condition_variable released_cv_;
  bool released_ = false;
  std::atomic<int> sampled_steps_{0};
};

// The poisoned twin: the engine dies before it can produce a token. The engine
// busy-loop guard posts ENGINE_CORE_DEAD, AsyncLLM's output handler calls
// propagate_error, and the request's collector rethrows on the consumer thread —
// which is the SSE stream's own thread.
class ThrowingRunnerStub : public vllm::v1::ModelRunnerBase {
 public:
  std::optional<vllm::v1::ModelRunnerOutput> execute_model(
      const vllm::v1::SchedulerOutput& /*scheduler_output*/) override {
    return std::nullopt;
  }

  vllm::v1::ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/)
      override {
    throw std::runtime_error("vt: CHAT_ROLE_FRAME_ORDER_SENTINEL");
  }
};

vllm::tok::Tokenizer BuildFixtureTokenizer() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_chat_first_frame_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array();
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {{"type", "ByteLevel"},
                          {"add_prefix_space", false},
                          {"trim_offsets", false},
                          {"use_regex", true}};
  const json vocab = {{"h", 0},    {"e", 1},      {"l", 2},      {"o", 3},
                      {"w", 4},    {"r", 5},      {"d", 6},      {"Ġ", 7},
                      {"1", 8},    {"2", 9},      {"ll", 10},    {"he", 11},
                      {"llo", 12}, {"hello", 13}, {"Ġw", 14},    {"or", 15},
                      {"orld", 16}, {"Ġworld", 17}, {"ld", 18}};
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges", json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                              json::array({"ll", "o"}),
                              json::array({"he", "llo"}),
                              json::array({"Ġ", "w"}), json::array({"o", "r"}),
                              json::array({"l", "d"}),
                              json::array({"or", "ld"}),
                              json::array({"Ġw", "orld"})})}};
  {
    std::ofstream out(path, std::ios::binary);
    out << doc.dump();
  }
  vllm::tok::Tokenizer tokenizer = vllm::tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tokenizer;
}

vllm::SchedulerConfig MakeSchedulerConfig() {
  vllm::SchedulerConfig cfg;
  cfg.max_num_seqs = 8;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  return cfg;
}

// #1999 clamps `max_num_seqs` to the seats the KV budget affords. It cannot
// bite here, for three independent reasons, and a future edit to this fixture
// should keep at least one of them true:
//   1. `ComputeHybridKvBudget` returns early on `mamba == nullptr`
//      (hybrid_kv_budget.cpp), and this config carries one FullAttentionSpec
//      and no MambaSpec, so the budget stays `kStateSeqsUnbounded` (-1) and
//      `ClampMaxNumSeqsToStateBudget` passes the configured value through.
//   2. The only production caller is `model_loader.cpp`, and this fixture
//      builds the Scheduler and AsyncLLM directly without the loader.
//   3. Each case issues exactly ONE streaming request, so `max_num_seqs = 8`
//      is headroom rather than a requirement; even a clamp to 1 seat would
//      leave both cases passing.
vllm::v1::KVCacheConfig MakeKvConfig() {
  vllm::v1::KVCacheConfig kv;
  kv.num_blocks = 1024;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<vllm::v1::FullAttentionSpec>(16, 1, 1, vt::DType::kF32));
  return kv;
}

vllm::HfConfig MakeHfConfig() {
  vllm::HfConfig config;
  config.max_position_embeddings = 8192;
  config.raw = json::object();
  return config;
}

vllm::v1::BlockHasher Hasher() {
  static bool initialized = false;
  if (!initialized) {
    vllm::v1::init_none_hash(vllm::v1::sha256_cbor);
    initialized = true;
  }
  return vllm::v1::get_request_block_hasher(16, vllm::v1::sha256_cbor);
}

// The default role-join fallback renders "user: hello\nassistant:", whose
// characters exceed this minimal BPE fixture's vocabulary. ChatPromptFn is the
// documented renderer seam (serving_chat.h), so the fixture supplies one that
// emits a prompt the fixture can encode. The framing under test is downstream
// of the prompt and unaffected by its text.
oai::ChatPromptFn FixturePromptFn() {
  return [](const std::vector<oai::ChatMessage>&, bool,
            const std::vector<oai::ChatCompletionToolsParam>&,
            const nlohmann::ordered_json&) -> std::string { return "hello"; };
}

// The production server stack the HTTP routes hold: AsyncLLM ->
// OpenAIServingChat -> ApiServer. `Runner` is the stub this case clocks.
template <typename Runner>
struct ServerHarness {
  ServerHarness()
      : tokenizer(BuildFixtureTokenizer()),
        scheduler(MakeSchedulerConfig(), MakeKvConfig(), /*block_size=*/16,
                  /*enable_caching=*/true),
        executor(runner),
        input_processor(tokenizer, MakeHfConfig()),
        output_processor(&tokenizer),
        engine(input_processor, scheduler, executor, output_processor, Hasher()),
        models("test-model"),
        completion(engine, "test-model"),
        chat(engine, "test-model", FixturePromptFn()),
        server(completion, chat, models, "9.9.9") {}

  vllm::tok::Tokenizer tokenizer;
  vllm::v1::Scheduler scheduler;
  Runner runner;
  vllm::v1::Executor executor;
  vllm::v1::InputProcessor input_processor;
  vllm::v1::OutputProcessor output_processor;
  vllm::v1::AsyncLLM engine;
  oai::OpenAIServingModels models;
  oai::OpenAIServingCompletion completion;
  oai::OpenAIServingChat chat;
  oai::ApiServer server;
};

constexpr const char* kStreamingChatBody =
    R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],)"
    R"("max_tokens":2,"temperature":0.0,"stream":true})";

// Does this frame carry the key `vllm bench serve` stamps TTFT on?
bool CarriesChoices(const std::string& frame) {
  if (frame.rfind("data: ", 0) != 0) return false;
  const std::string payload = frame.substr(6);
  if (payload.rfind("[DONE]", 0) == 0) return false;
  const json parsed = json::parse(payload, nullptr, false);
  return !parsed.is_discarded() && parsed.contains("choices") &&
         !parsed.at("choices").empty();
}

}  // namespace

// ── 1. Frame ordering ────────────────────────────────────────────────────────
//
// The runner is held inside sample_tokens for the whole first phase, so the
// engine cannot have produced a token. The assertion that discriminates is
// CHECK_FALSE(first_frame_arrived) taken in that state. Before the fix the role
// frame is already on the wire microseconds after the request is admitted, so it
// fails; after the fix it cannot be produced while the gate is closed, whatever
// the load on the box.
//
// The 300 ms is a GRACE for the defective path to show itself, never a deadline
// the correct path has to beat. A slower box makes this case more reliable, not
// less.
TEST_CASE("chat SSE: no choices-bearing frame before the first token exists") {
  ServerHarness<GatedRunnerStub> h;

  oai::ApiServer::DispatchResult result =
      h.server.handle_chat_completions(kStreamingChatBody);
  REQUIRE(result.status == 200);
  REQUIRE(result.streaming);
  REQUIRE(result.sse_stream != nullptr);
  std::shared_ptr<oai::SseStream> stream = result.sse_stream;

  std::atomic<bool> first_frame_arrived{false};
  std::string first_frame;
  std::string puller_error;
  // An exception escaping this thread would std::terminate the whole binary and
  // report nothing, so it is caught and turned into a named failure below.
  std::thread puller([&] {
    try {
      if (stream->next(first_frame)) first_frame_arrived.store(true);
    } catch (const std::exception& e) {
      puller_error = e.what();
    } catch (...) {
      puller_error = "unknown exception";
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // The instrument states its own precondition: 0 sampled steps means no token
  // exists, so a frame observed now was produced with no engine output at all.
  CHECK_MESSAGE(h.runner.sampled_steps() == 0,
                "the gated runner sampled a step while still gated; this case "
                "measured nothing");
  CHECK_MESSAGE(!first_frame_arrived.load(),
                "a chat SSE frame reached the client before any engine output: "
                    << first_frame);

  h.runner.release();
  puller.join();

  // ...and the frame is not merely late, it still arrives, with the shape the
  // wire contract keeps: role "assistant", empty content, choices present.
  CHECK_MESSAGE(puller_error.empty(),
                "the first next() call threw instead of returning a frame: "
                    << puller_error);
  CHECK(first_frame_arrived.load());
  CHECK(h.runner.sampled_steps() >= 1);
  REQUIRE(first_frame.rfind("data: ", 0) == 0);
  CHECK(CarriesChoices(first_frame));
  const json parsed = json::parse(first_frame.substr(6));
  CHECK(parsed.at("choices").at(0).at("delta").at("role") == "assistant");
  CHECK(parsed.at("choices").at(0).at("delta").at("content") == "");
  CHECK_FALSE(parsed.contains("usage"));

  // Drain so the request retires before the harness tears the engine down.
  //
  // This drain also guards the one regression the change could introduce: the
  // buffered first result must be DELIVERED, not swallowed. Both sampled tokens
  // are the fixture's `Ġworld`, so the concatenated content across every frame
  // has to carry "world" exactly twice. Counting the text rather than the frames
  // keeps the assertion independent of collector merging, which can fold two
  // outputs into one frame when the consumer is slow.
  std::string chunk;
  std::vector<std::string> rest;
  std::string streamed;
  while (stream->next(chunk)) {
    rest.push_back(chunk);
    if (!CarriesChoices(chunk)) continue;
    const json frame = json::parse(chunk.substr(6));
    const json& delta = frame.at("choices").at(0).at("delta");
    if (delta.contains("content") && delta.at("content").is_string()) {
      streamed += delta.at("content").get<std::string>();
    }
  }
  REQUIRE_FALSE(rest.empty());
  CHECK(rest.back() == "data: [DONE]\n\n");
  size_t occurrences = 0;
  for (size_t at = streamed.find("world"); at != std::string::npos;
       at = streamed.find("world", at + 1)) {
    ++occurrences;
  }
  CHECK_MESSAGE(occurrences == 2,
                "the buffered first result was dropped or duplicated; streamed "
                "content was: "
                    << streamed);
}

// ── 2. Error ordering ────────────────────────────────────────────────────────
//
// Upstream's stated reason for the ordering, made executable. The engine dies
// before its first token; the FIRST next() call must surface that, and must not
// have written a frame first.
//
// `chunk` carries a sentinel, so the case fails in two independent ways on the
// old behaviour: next() returns instead of throwing, AND the sentinel is
// overwritten by the role frame.
TEST_CASE("chat SSE: an engine that dies before the first token is not "
          "preceded by a role frame") {
  ServerHarness<ThrowingRunnerStub> h;

  oai::ApiServer::DispatchResult result =
      h.server.handle_chat_completions(kStreamingChatBody);
  REQUIRE(result.status == 200);
  REQUIRE(result.streaming);
  REQUIRE(result.sse_stream != nullptr);

  std::string chunk = "SENTINEL_NOT_A_FRAME";
  CHECK_THROWS_MESSAGE(result.sse_stream->next(chunk),
                       "the first next() call returned a frame instead of "
                       "surfacing the engine failure");
  CHECK_MESSAGE(chunk == "SENTINEL_NOT_A_FRAME",
                "a frame was written before the engine failure surfaced: "
                    << chunk);
}
