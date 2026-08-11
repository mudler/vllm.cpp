// Smoke test for the M2.1 benchmark harness (examples/bench/bench_core.h): drive
// the SYNTHETIC CPU engine through the production AsyncLLM measurement loop
// and assert it produces sane metrics. The NUMBERS are meaningless (toy weights)
// — this asserts the HARNESS: all N requests finish, throughput > 0, TTFT > 0,
// and the token accounting is coherent. The real parity numbers come from a GB10
// run with --model (dgx-pending), which this same code path drives.
#include "bench_core.h"

#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/v1/engine/input_processor.h"

using vllm::bench::BenchConfig;
using vllm::bench::BenchResult;
using vllm::bench::DispatchBenchPromptAdmission;
using vllm::bench::DispatchBenchPromptWaveAdmission;
using vllm::bench::PretokenizeBenchPromptsThenStartClock;
using vllm::bench::RunBench;

namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, std::optional<std::string> value) : name_(name) {
    if (const char* previous = std::getenv(name)) previous_ = previous;
    if (value.has_value()) {
      REQUIRE(::setenv(name, value->c_str(), /*overwrite=*/1) == 0);
    } else {
      REQUIRE(::unsetenv(name) == 0);
    }
  }

  ~ScopedEnv() {
    if (previous_.has_value()) {
      (void)::setenv(name_.c_str(), previous_->c_str(), /*overwrite=*/1);
    } else {
      (void)::unsetenv(name_.c_str());
    }
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

struct AdmissionObservation {
  int result = 0;
  int pretokenized_calls = 0;
  int string_calls = 0;
};

AdmissionObservation ObserveAdmission(const char* env_value) {
  AdmissionObservation observed;
  observed.result = DispatchBenchPromptAdmission(
      env_value,
      [&]() {
        ++observed.pretokenized_calls;
        return 17;
      },
      [&]() {
        ++observed.string_calls;
        return 29;
      });
  return observed;
}

std::string RenderReport(const BenchResult& result) {
  std::FILE* const file = std::tmpfile();
  if (file == nullptr) throw std::runtime_error("tmpfile failed");
  vllm::bench::PrintReport(BenchConfig{}, result, file);
  std::fflush(file);
  std::rewind(file);
  std::string report;
  char buffer[1024];
  while (const size_t count = std::fread(buffer, 1, sizeof(buffer), file)) {
    report.append(buffer, count);
  }
  std::fclose(file);
  return report;
}

int ReportedPretokenizedAdmission(const BenchResult& result) {
  const std::string report = RenderReport(result);
  const std::string label = "Pretokenized prompt admission:";
  const size_t begin = report.find(label);
  if (begin == std::string::npos) return -1;
  const size_t end = report.find('\n', begin);
  const std::string line = report.substr(begin, end - begin);
  int value = -1;
  if (std::sscanf(line.c_str(), "Pretokenized prompt admission: %d", &value) !=
      1) {
    return -1;
  }
  return value;
}

struct RecordingTokenizer {
  std::vector<std::string>* events = nullptr;

  std::vector<int32_t> EncodeWithSpecialTokens(std::string_view prompt) const {
    events->push_back("special:" + std::string(prompt));
    return {20, static_cast<int32_t>(prompt.size())};
  }

  std::vector<int32_t> Encode(std::string_view prompt) const {
    events->push_back("raw:" + std::string(prompt));
    return {static_cast<int32_t>(prompt.size())};
  }
};

struct WaveOnlyEngine {
  std::vector<std::string>* events = nullptr;
  int token_publishes = 0;
  int string_publishes = 0;
  std::vector<std::string> published_ids;

  std::vector<std::string> add_request_wave(
      std::vector<vllm::v1::AsyncTokensRequestInput> wave) {
    ++token_publishes;
    events->push_back("token-publish");
    published_ids.clear();
    for (const auto& request : wave) {
      published_ids.push_back(request.request_id);
    }
    return published_ids;
  }

  std::vector<std::string> add_request_wave(
      std::vector<vllm::v1::AsyncStringRequestInput> wave) {
    ++string_publishes;
    events->push_back("string-publish");
    published_ids.clear();
    for (const auto& request : wave) {
      published_ids.push_back(request.request_id);
    }
    return published_ids;
  }
};

}  // namespace

TEST_CASE("bench: pretokenized admission dispatch is default-on with exact rollback") {
  const AdmissionObservation unset = ObserveAdmission(nullptr);
  CHECK(unset.result == 17);
  CHECK(unset.pretokenized_calls == 1);
  CHECK(unset.string_calls == 0);

  const AdmissionObservation enabled = ObserveAdmission("1");
  CHECK(enabled.result == 17);
  CHECK(enabled.pretokenized_calls == 1);
  CHECK(enabled.string_calls == 0);

  const AdmissionObservation rollback = ObserveAdmission("0");
  CHECK(rollback.result == 29);
  CHECK(rollback.pretokenized_calls == 0);
  CHECK(rollback.string_calls == 1);

  // Invalid spellings keep the safe, production-parity default rather than
  // silently restoring timed tokenization.
  const AdmissionObservation invalid = ObserveAdmission("banana");
  CHECK(invalid.result == 17);
  CHECK(invalid.pretokenized_calls == 1);
  CHECK(invalid.string_calls == 0);
}

TEST_CASE("bench: wave admission owns one ordered engine publish") {
  std::vector<std::string> events;
  WaveOnlyEngine engine;
  engine.events = &events;
  const std::vector<std::string> result = DispatchBenchPromptWaveAdmission(
      engine, /*env_value=*/nullptr, /*wave_size=*/3,
      [&](size_t offset) {
        events.push_back("arrival-" + std::to_string(offset));
      },
      [&]() {
        events.push_back("token-build");
        std::vector<vllm::v1::AsyncTokensRequestInput> wave;
        for (int id = 4; id < 7; ++id) {
          wave.push_back({std::to_string(id), {id}, {}, 0});
        }
        return wave;
      },
      [&]() {
        events.push_back("string-build");
        return std::vector<vllm::v1::AsyncStringRequestInput>{
            {"wrong-arm", "prompt", {}, 0}};
      });
  CHECK(result == std::vector<std::string>{"4", "5", "6"});
  CHECK(engine.token_publishes == 1);
  CHECK(engine.string_publishes == 0);
  CHECK(engine.published_ids == std::vector<std::string>{"4", "5", "6"});
  CHECK(events == std::vector<std::string>{"arrival-0", "arrival-1",
                                           "arrival-2", "token-build",
                                           "token-publish"});

  events.clear();
  const std::vector<std::string> rollback = DispatchBenchPromptWaveAdmission(
      engine, /*env_value=*/"0", /*wave_size=*/2,
      [&](size_t offset) {
        events.push_back("arrival-" + std::to_string(offset));
      },
      [&]() {
        events.push_back("token-build");
        return std::vector<vllm::v1::AsyncTokensRequestInput>{
            {"wrong-arm", {99}, {}, 0}};
      },
      [&]() {
        events.push_back("string-build");
        return std::vector<vllm::v1::AsyncStringRequestInput>{
            {"9", "first", {}, 0}, {"10", "second", {}, 0}};
      });
  CHECK(rollback == std::vector<std::string>{"9", "10"});
  CHECK(engine.token_publishes == 1);
  CHECK(engine.string_publishes == 1);
  CHECK(engine.published_ids == std::vector<std::string>{"9", "10"});
  CHECK(events == std::vector<std::string>{"arrival-0", "arrival-1",
                                           "string-build", "string-publish"});
}

TEST_CASE("bench: report exposes the resolved pretokenized admission mode") {
  BenchResult result;
  result.pretokenized_admission = true;
  CHECK(ReportedPretokenizedAdmission(result) == 1);

  result.pretokenized_admission = false;
  CHECK(ReportedPretokenizedAdmission(result) == 0);
}

TEST_CASE("bench: synthetic tokenizer distinguishes prompt special tokens") {
  const vllm::tok::Tokenizer tokenizer =
      vllm::bench::detail::BuildSyntheticTokenizer();
  const std::string prompt = "hello world";

  CHECK(tokenizer.EncodeWithSpecialTokens(prompt) != tokenizer.Encode(prompt));
}

TEST_CASE("bench: default pretokenization completes before the clock") {
  const std::vector<std::string> prompts = {"hello", "world"};
  std::vector<std::string> events;
  const RecordingTokenizer tokenizer{&events};

  auto [prepared, clock_value] = PretokenizeBenchPromptsThenStartClock(
      vllm::bench::ResolveBenchPretokenizedAdmission(nullptr), tokenizer,
      prompts, [&]() {
        events.push_back("clock");
        return 73;
      });

  CHECK(events == std::vector<std::string>{"special:hello", "special:world",
                                           "clock"});
  CHECK(prepared ==
        std::vector<std::vector<int32_t>>{{20, 5}, {20, 5}});
  CHECK(clock_value == 73);

  events.clear();
  auto [rollback, rollback_clock] = PretokenizeBenchPromptsThenStartClock(
      /*pretokenized_admission=*/false, tokenizer, prompts, [&]() {
        events.push_back("clock");
        return 91;
      });
  CHECK(rollback.empty());
  CHECK(events == std::vector<std::string>{"clock"});
  CHECK(rollback_clock == 91);
}

TEST_CASE("bench: pretokenized vectors match timed-string InputProcessor") {
  const vllm::tok::Tokenizer tokenizer =
      vllm::bench::detail::BuildSyntheticTokenizer();
  const std::vector<std::string> prompts = {"hello", "hello world"};
  auto [prepared, clock_value] = PretokenizeBenchPromptsThenStartClock(
      vllm::bench::ResolveBenchPretokenizedAdmission(nullptr), tokenizer,
      prompts, []() { return 17; });

  const vllm::HfConfig config =
      vllm::bench::detail::MakeSyntheticConfig(/*max_model_len=*/128);
  const vllm::v1::InputProcessor input_processor(tokenizer, config);
  REQUIRE(prepared.size() == prompts.size());
  for (size_t i = 0; i < prompts.size(); ++i) {
    const vllm::v1::EngineCoreRequest timed_string =
        input_processor.process_inputs(std::to_string(i), prompts[i],
                                       vllm::SamplingParams{},
                                       /*arrival_time=*/0.0);
    CHECK(prepared[i] == timed_string.prompt_token_ids);
  }
  CHECK(clock_value == 17);
}

TEST_CASE("bench: synthetic engine completes all requests with sane metrics") {
  BenchConfig cfg;
  cfg.num_prompts = 8;
  cfg.input_len = 16;
  cfg.output_len = 16;
  cfg.concurrency = 4;
  cfg.seed = 123;
  cfg.temperature = 0.0;  // greedy => deterministic, exactly output_len tokens.

  const BenchResult r = RunBench(cfg);

  // The comparison harness must exercise the production AsyncLLM frontend.
  // Before the SERVE-CLI-BENCH B1 repair it called synchronous
  // LoadedEngine::engine() even when async scheduling resolved enabled.
  CHECK(r.async_frontend);
  CHECK(r.max_concurrent_batches >= 1);

  // All N requests finished through the engine loop.
  CHECK(r.completed == cfg.num_prompts);
  // Wall time advanced and throughput is positive.
  CHECK(r.duration_s > 0.0);
  CHECK(r.request_throughput > 0.0);
  CHECK(r.output_throughput > 0.0);
  CHECK(r.total_token_throughput > 0.0);
  CHECK(r.input_throughput > 0.0);
  // Token accounting: greedy w/ no eos => exactly output_len tokens per request.
  CHECK(r.total_output == static_cast<int64_t>(cfg.num_prompts) * cfg.output_len);
  REQUIRE(r.output_token_ids.size() == static_cast<size_t>(cfg.num_prompts));
  for (const auto& ids : r.output_token_ids) {
    CHECK(ids.size() == static_cast<size_t>(cfg.output_len));
  }
  CHECK(r.total_input > 0);
  CHECK(r.total_token_throughput ==
        doctest::Approx(r.input_throughput + r.output_throughput));
  // Latency metrics are engaged (first token observed => TTFT > 0; multi-token
  // decode => TPOT/ITL > 0).
  CHECK(r.mean_ttft_ms > 0.0);
  CHECK(r.mean_tpot_ms > 0.0);
  CHECK(r.mean_itl_ms > 0.0);
  CHECK(r.mean_e2el_ms >= r.mean_ttft_ms);
  CHECK(r.mean_per_stream_decode > 0.0);
}

TEST_CASE("bench: output token IDs serialize in submission order") {
  BenchConfig cfg;
  cfg.num_prompts = 3;
  cfg.input_len = 8;
  cfg.output_len = 4;
  cfg.concurrency = 2;

  const BenchResult r = RunBench(cfg);
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "vllm_cpp_bench_token_ids.json";
  vllm::bench::WriteOutputTokenIds(path.string(), r);

  nlohmann::json saved;
  std::ifstream(path) >> saved;
  std::filesystem::remove(path);
  REQUIRE(saved.is_array());
  REQUIRE(saved.size() == 3);
  CHECK(saved == nlohmann::json(r.output_token_ids));
}

TEST_CASE("bench: concurrency=1 (serial) also completes and is coherent") {
  BenchConfig cfg;
  cfg.num_prompts = 4;
  cfg.input_len = 8;
  cfg.output_len = 8;
  cfg.concurrency = 1;
  cfg.seed = 7;

  const BenchResult r = RunBench(cfg);

  CHECK(r.completed == 4);
  CHECK(r.total_output == 4 * 8);
  CHECK(r.request_throughput > 0.0);
  CHECK(r.mean_ttft_ms > 0.0);
}

TEST_CASE("bench: ShareGPT dataset supplies exact prompts") {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "vllm_cpp_bench_sharegpt.json";
  {
    std::ofstream out(path);
    out << R"json([{"conversations":[{"from":"human","value":"hello world"},{"from":"gpt","value":"ok"}]},{"conversations":[{"from":"human","value":"world hello"},{"from":"gpt","value":"ok"}]}])json";
  }

  BenchConfig cfg;
  cfg.dataset_path = path.string();
  cfg.num_prompts = 2;
  cfg.input_len = 8;
  cfg.output_len = 4;
  cfg.concurrency = 2;
  const BenchResult r = RunBench(cfg);
  std::filesystem::remove(path);

  CHECK(r.completed == 2);
  CHECK(r.total_output == 8);
  CHECK(r.total_input > 0);
}

TEST_CASE("bench: pretokenized default preserves prompt and output token IDs") {
  BenchConfig cfg;
  cfg.num_prompts = 5;
  cfg.input_len = 12;
  cfg.output_len = 6;
  cfg.concurrency = 3;
  cfg.seed = 31;
  cfg.temperature = 0.0;

  BenchResult pretokenized;
  {
    ScopedEnv env("VT_BENCH_PRETOKENIZE", std::nullopt);
    pretokenized = RunBench(cfg);
  }

  BenchResult timed_string;
  {
    ScopedEnv env("VT_BENCH_PRETOKENIZE", std::string("0"));
    timed_string = RunBench(cfg);
  }

  CHECK(pretokenized.pretokenized_admission);
  CHECK_FALSE(timed_string.pretokenized_admission);
  REQUIRE(pretokenized.prompt_token_ids.size() ==
          static_cast<size_t>(cfg.num_prompts));
  REQUIRE(timed_string.prompt_token_ids.size() ==
          static_cast<size_t>(cfg.num_prompts));
  CHECK(pretokenized.prompt_token_ids == timed_string.prompt_token_ids);
  CHECK(pretokenized.output_token_ids == timed_string.output_token_ids);
  CHECK(pretokenized.total_input == timed_string.total_input);
  CHECK(pretokenized.total_output == timed_string.total_output);
}
