// M3.4 Task 5 END-TO-END wiring test: an OpenAI `response_format` request flows
//   ChatCompletionRequest.from_json (response_format json_schema)
//     -> to_sampling_params  (structured_outputs.json = the schema)
//       -> Request                     (carries the structured-output constraint)
//         -> StructuredOutputManager.grammar_init  (native backend compiles the
//            JSON-schema -> GBNF grammar)
//           -> Scheduler.get_grammar_bitmask       (a row for the structured req)
//             -> EngineCore.step threads the GrammarOutput to sample_tokens.
//
// The manager is constructed with the NATIVE backend factory
// (MakeNativeBackendFactory(tokenizer, vocab_size)) — the same wiring a
// production engine uses — and shared by the Scheduler and the EngineCore, so a
// response_format request actually constrains decoding end to end. We assert:
//   (1) grammar_init compiled the schema (the request carries a non-null grammar);
//   (2) that grammar is a real JSON grammar (accepts the schema-valid opening
//       byte, rejects a non-JSON byte);
//   (3) the runner receives a NON-NULL GrammarOutput naming the structured req.
//
// A RunnerStub (canned token) stands in for the model: the tiny JSON tokenizer's
// vocab cannot spell a full model generation, and the bar here is the WIRING, not
// a schema-valid synthetic generation (schema acceptance is covered exhaustively
// in test_json_schema_to_gbnf.cpp).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/manager.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::entrypoints::openai::ChatCompletionRequest;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCore;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::GrammarOutput;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::MakeNativeBackendFactory;
using vllm::v1::ModelRunnerBase;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::Request;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vllm::v1::StructuredOutputManager;
using vt::DType;

namespace {

constexpr int32_t kCannedToken = 3;   // a valid in-vocab id the stub "samples".
constexpr int32_t kEos = 200;
// Includes the `<`, `>`, `/`, `_` and newline bytes the WRAPPED tool-call
// grammar (`<tool_call>\n{...}\n</tool_call>`) needs so it is drivable here.
const std::string kChars =
    "{}[]\":,. -0123456789abcdefghijklmnopqrstuvwxyz<>/_\n";

class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_rf_e2e_tok_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

Tokenizer BuildFixture() {
  json vocab = json::object();
  for (std::size_t i = 0; i < kChars.size(); ++i) {
    vocab[MapBytesToUnicode(std::string(1, kChars[i]))] = static_cast<int>(i);
  }
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", kEos}, {"content", "<eos>"}, {"special", true}}});
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
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array()}};
  TempJson f(doc.dump());
  return Tokenizer::FromHfJson(f.path());
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

int VocabSize() { return static_cast<int>(Fixture().VocabSize()); }

// A ModelRunnerBase double (canned token per scheduled req) that records the
// GrammarOutput the EngineCore threads to sample_tokens.
class RunnerStub : public ModelRunnerBase {
 public:
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    stashed_ = scheduler_output;
    return std::nullopt;
  }
  ModelRunnerOutput sample_tokens(
      const std::optional<GrammarOutput>& grammar_output) override {
    last_grammar_present = grammar_output.has_value();
    last_grammar_req_ids.clear();
    if (grammar_output.has_value()) {
      last_grammar_req_ids = grammar_output->structured_output_request_ids;
    }
    ModelRunnerOutput mro;
    int idx = 0;
    for (const auto& [req_id, n] : stashed_.num_scheduled_tokens) {
      mro.req_ids.push_back(req_id);
      mro.req_id_to_index[req_id] = idx++;
      mro.sampled_token_ids.push_back({kCannedToken});
    }
    return mro;
  }
  bool last_grammar_present = false;
  std::vector<std::string> last_grammar_req_ids;

 private:
  SchedulerOutput stashed_;
};

std::unique_ptr<Scheduler> CreateScheduler(StructuredOutputManager* mgr) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 8;
  cfg.max_num_batched_tokens = 4096;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 4096;
  cfg.watermark = 0.0;
  KVCacheConfig kv;
  kv.num_blocks = 1024;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<Scheduler>(cfg, kv, /*block_size=*/16,
                                     /*enable_caching=*/true, mgr);
}

vllm::v1::BlockHasher Hasher() {
  static bool init = false;
  if (!init) {
    init_none_hash(sha256_cbor);
    init = true;
  }
  return get_request_block_hasher(16, sha256_cbor);
}

}  // namespace

// The full response_format -> engine wiring over the NATIVE backend.
TEST_CASE("response_format json_schema constrains decoding end to end (native backend)") {
  // ── The OpenAI request with a json_schema response_format ──────────────────
  auto j = json::parse(R"({
    "messages": [{"role":"user","content":"hi"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "obj",
        "schema": {"type":"object","properties":{"a":{"type":"integer"}},
                   "required":["a"]}
      }
    }
  })");
  auto req = j.get<ChatCompletionRequest>();
  // default_max_tokens mirrors the serving-resolved fallback (the request omits
  // max_tokens); check_stop asserts max_tokens is set.
  SamplingParams sp = req.to_sampling_params(/*default_max_tokens=*/16);
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->json.has_value());  // schema flowed through.

  // ── The engine, with a NATIVE-backed manager shared by scheduler + core ────
  StructuredOutputManager manager(
      /*max_num_seqs=*/8, MakeNativeBackendFactory(Fixture(), VocabSize(),
                                                   std::vector<int32_t>{kEos}));
  auto scheduler = CreateScheduler(&manager);
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor, &manager);

  // A short in-vocab prompt (ids exist in the fixture vocab).
  std::vector<int32_t> prompt = {10, 11, 12, 13};  // 'a','b','c','d'
  auto request = std::make_unique<Request>("req0", prompt, sp,
                                           /*arrival_time=*/0.0, Hasher());
  engine.add_request(std::move(request));

  // (1) grammar_init compiled the JSON-schema constraint into a live grammar.
  Request* r = scheduler->requests.at("req0").get();
  REQUIRE(r->structured_output_request.has_value());
  REQUIRE(r->structured_output_request->grammar != nullptr);

  // (2) it is a real JSON grammar: the opening '{' is allowed, a letter is not.
  auto& grammar = *r->structured_output_request->grammar;
  const int32_t open_brace = static_cast<int32_t>(kChars.find('{'));
  const int32_t letter_a = static_cast<int32_t>(kChars.find('a'));
  // validate_tokens probes acceptance WITHOUT advancing the engine's FSM.
  CHECK(grammar.validate_tokens({open_brace}) == std::vector<int32_t>{open_brace});
  CHECK(grammar.validate_tokens({letter_a}).empty());

  // (3) EngineCore.step threads a NON-NULL GrammarOutput naming the req to the
  // runner's sample path — decoding is constrained.
  auto [outputs, model_executed] = engine.step();
  (void)outputs;
  CHECK(model_executed);
  CHECK(runner.last_grammar_present);
  REQUIRE(runner.last_grammar_req_ids.size() == 1);
  CHECK(runner.last_grammar_req_ids[0] == "req0");
}

// M3.3: a tool_choice=required forced tool-call schema flows the same path
// (ApplyToolChoiceStructuredOutput -> structured_outputs.grammar -> Request ->
// grammar_init -> constrained decode) and yields a live grammar that emits the
// WRAPPED `<tool_call>…</tool_call>` tool call the Hermes/Qwen parser extracts.
TEST_CASE("tool_choice required forces a wrapped tool-call grammar end to end (native backend)") {
  using vllm::entrypoints::openai::ApplyToolChoiceStructuredOutput;
  using vllm::entrypoints::openai::ChatCompletionToolsParam;
  using vllm::entrypoints::openai::ChatMessage;
  using vllm::entrypoints::openai::ToolChoice;

  // A chat request with one tool + tool_choice="required".
  ChatCompletionRequest req;
  req.messages = {ChatMessage{"user", std::string("weather?")}};
  ChatCompletionToolsParam tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.parameters = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},)"
      R"("required":["city"]})");
  req.tools = std::vector<ChatCompletionToolsParam>{tool};
  req.tool_choice = ToolChoice{"required", std::nullopt};

  // The forced tool-call schema flows onto the SamplingParams as the WRAPPED
  // GBNF grammar (structured_outputs.grammar, NOT bare .json).
  SamplingParams sp = req.to_sampling_params(/*default_max_tokens=*/16);
  ApplyToolChoiceStructuredOutput(req, sp);
  REQUIRE(sp.structured_outputs.has_value());
  CHECK_FALSE(sp.structured_outputs->json.has_value());
  REQUIRE(sp.structured_outputs->grammar.has_value());

  // The engine, with a NATIVE-backed manager shared by scheduler + core.
  StructuredOutputManager manager(
      /*max_num_seqs=*/8, MakeNativeBackendFactory(Fixture(), VocabSize(),
                                                   std::vector<int32_t>{kEos}));
  auto scheduler = CreateScheduler(&manager);
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor, &manager);

  std::vector<int32_t> prompt = {10, 11, 12, 13};
  auto request = std::make_unique<Request>("tc0", prompt, sp,
                                           /*arrival_time=*/0.0, Hasher());
  engine.add_request(std::move(request));

  // grammar_init compiled the forced tool-call schema into a live grammar.
  Request* r = scheduler->requests.at("tc0").get();
  REQUIRE(r->structured_output_request.has_value());
  REQUIRE(r->structured_output_request->grammar != nullptr);

  // It is the WRAPPED tool-call grammar: the first byte must be '<' (the
  // `<tool_call>` wrapper), and a BARE '{' is REJECTED at the start — this is
  // what makes the constrained output extractable by the Hermes/Qwen parser
  // (the bare-JSON bug this fix closes).
  auto& grammar = *r->structured_output_request->grammar;
  const int32_t open_angle = static_cast<int32_t>(kChars.find('<'));
  const int32_t open_brace = static_cast<int32_t>(kChars.find('{'));
  CHECK(grammar.validate_tokens({open_angle}) == std::vector<int32_t>{open_angle});
  CHECK(grammar.validate_tokens({open_brace}).empty());

  // EngineCore.step threads a NON-NULL GrammarOutput naming the req.
  auto [outputs, model_executed] = engine.step();
  (void)outputs;
  CHECK(model_executed);
  CHECK(runner.last_grammar_present);
  REQUIRE(runner.last_grammar_req_ids.size() == 1);
  CHECK(runner.last_grammar_req_ids[0] == "tc0");
}
