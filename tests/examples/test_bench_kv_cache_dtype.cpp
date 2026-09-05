// KV-FP8 W7 (#2619) — does `vllm-bench --kv-cache-dtype` reach the KV cache, and
// does the report SAY which cache the number was measured on?
//
// WHY THIS RUNS THE BINARY. `tests/vllm/entrypoints/test_kv_cache_fp8_wiring.cpp`
// already drives `EngineParams::kv_cache_dtype` through `LoadedEngine`, and
// `tests/examples/test_bench.cpp` already drives `RunBench` through a
// hand-built `BenchConfig`. Both stayed green on every day `vllm-bench` had no
// such flag, because neither enters through the harness's own command line.
// .agents/reachability.md's rule is that the smallest failing test enters
// through the production entry point; for a benchmark harness that entry point
// is `argv`. So this file execs the built `vllm-bench` and reads its report.
//
// WHY THE SYNTHETIC ENGINE IS ENOUGH. With no `--model`, `RunBench` builds the
// tiny CPU Qwen3.5-MoE whose attention block W3 routed. That model allocates
// real paged KV and decodes through it, so an fp8 page here is an fp8 page --
// the throughput numbers are meaningless (toy weights) and nothing below reads
// one. No checkpoint, no device, no lease.
//
// THE MUTATIONS THIS FILE EXISTS FOR, and which half of the pair catches each:
//   - delete `params.kv_cache_dtype = cfg.kv_cache_dtype;` from either engine
//     construction in bench_core.h => G1 goes red. The flag parses and is then
//     dropped, the run still succeeds, and the report still prints a resolved
//     line -- naming bf16. That silent-drop is the precise defect #2619 names.
//   - delete the `--kv-cache-dtype` arm from ParseArgs in main.cpp => G1 and G3
//     go red on "unknown argument" and exit 2.
//   - hard-code the resolved line to "fp8_e4m3" => G2 goes red. G2 is the whole
//     reason G1 means anything.
//   - delete the `VT_CHECK` in ParseCacheDType => G3 goes red.
#include <doctest/doctest.h>

#include <sys/wait.h>

#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace {

#ifndef VLLM_BENCH_BINARY
#define VLLM_BENCH_BINARY ""
#endif

constexpr const char* kBenchBinary = VLLM_BENCH_BINARY;

[[noreturn]] void SkipGate() {
  std::fprintf(stderr,
               "\n*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "*** test_bench_kv_cache_dtype: built without "
               "VLLM_CPP_BUILD_EXAMPLES, so there is no vllm-bench binary to "
               "run\n\n");
  std::fflush(stderr);
  std::exit(77);
}

void RequireBenchBinary() {
  if (std::string(kBenchBinary).empty()) SkipGate();
}

struct BenchRun {
  std::string output;
  int status = -1;
};

// The smallest run that still allocates KV and decodes through it. Greedy, one
// request, two output tokens: this file asserts on the REPORT, never on a rate.
constexpr const char* kTinyWorkload =
    "--num-prompts 1 --input-len 8 --output-len 2 --concurrency 1";

BenchRun RunBenchBinary(const std::string& args) {
  const std::string cmd =
      std::string(kBenchBinary) + " " + kTinyWorkload + " " + args + " 2>&1";
  BenchRun run;
  FILE* pipe = ::popen(cmd.c_str(), "r");
  REQUIRE(pipe != nullptr);
  std::array<char, 4096> buf{};
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    run.output += buf.data();
  }
  const int closed = ::pclose(pipe);
  REQUIRE(closed != -1);
  run.status = WIFEXITED(closed) ? WEXITSTATUS(closed) : -1;
  return run;
}

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// ── The REAL-checkpoint arm, without a checkpoint ────────────────────────────
//
// The cases above drive the SYNTHETIC engine, whose `EngineParams` go to a
// direct `LoadedEngine` constructor. That leaves the OTHER pass-through -- the
// one on the `--model` path, which is the arm every published number is
// actually measured on -- unreached: deleting it kept all of the above green.
//
// `LoadedEngine::FromModelDir` resolves `--kv-cache-dtype` against the
// checkpoint's own `kv_cache_quant_algo` in its second statement, before any
// weight operation, and SAYS SO on stderr when the checkpoint is what decided.
// So a directory carrying only `hf_quant_config.json` produces that line and
// then fails on the absent weights. The line is PRESENT when no flag was typed
// and ABSENT when one was, because an explicit value is returned unchanged and
// the checkpoint is never consulted (`vllm/utils/torch_utils.py:380-381`).
//
// That polarity is the gate for the `--model` arm. A run whose flag never
// arrived would carry the default "auto" into the resolver, the checkpoint
// would win, and the line would appear in both cases. This is the harness twin
// of `tests/vllm/entrypoints/openai/test_serve_kv_cache_dtype.cpp`, and the
// `hf_quant_config.json` bytes below are the same ones that file carries,
// transcribed from `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`
// @ `36f717a22990e82c54c1d48ee77c491b87825680` and trimmed to the keys the
// resolver reads.
constexpr const char* kDeclaresFp8 =
    R"({"producer":{"name":"modelopt"},)"
    R"("quantization":{"quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":"FP8"}})";

// The loader's resolution line. It names the CHECKPOINT as the decider, which
// is the only reason it can be read as "no explicit flag arrived".
constexpr const char* kCheckpointDecided =
    "the checkpoint declares kv_cache_quant_algo";

// A throwaway model directory carrying only `hf_quant_config.json`. No weights
// and no tokenizer: the load is MEANT to fail, after the resolution stanza.
class DeclaringCheckpoint {
 public:
  DeclaringCheckpoint() {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllm-cpp-bench-kv-dtype-" + std::to_string(::getpid()) + "-" +
             std::to_string(counter++));
    std::filesystem::create_directories(path_);
    std::ofstream out(path_ / "hf_quant_config.json");
    REQUIRE(out.good());
    out << kDeclaresFp8;
  }
  ~DeclaringCheckpoint() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  DeclaringCheckpoint(const DeclaringCheckpoint&) = delete;
  DeclaringCheckpoint& operator=(const DeclaringCheckpoint&) = delete;

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

// The report line that a published number is read off. Its LABEL is asserted
// separately from its VALUE so that renaming the label reddens every case at
// once instead of quietly making the value assertions vacuous.
constexpr const char* kResolvedLabel = "KV cache dtype (resolved storage):";
constexpr const char* kRequestedLabel = "KV cache dtype (requested):";

// The value printed after `label` on its own report line, whitespace-trimmed, or
// "" when the label is absent. Trimmed rather than matched against a literal
// with the padding baked in, because the report's column width is a formatting
// choice and a gate that asserts on it fails for a reason that is not a defect.
std::string LineValue(const std::string& output, const char* label) {
  const size_t at = output.find(label);
  if (at == std::string::npos) return "";
  const size_t start = at + std::string(label).size();
  const size_t eol = output.find('\n', start);
  std::string rest = output.substr(
      start, eol == std::string::npos ? std::string::npos : eol - start);
  const size_t first = rest.find_first_not_of(" \t");
  if (first == std::string::npos) return "";
  const size_t last = rest.find_last_not_of(" \t\r");
  return rest.substr(first, last - first + 1);
}

std::string ResolvedValue(const std::string& output) {
  return LineValue(output, kResolvedLabel);
}

}  // namespace

// ─── G1. The flag reaches the KV cache ───────────────────────────────────────
TEST_CASE("vllm-bench: --kv-cache-dtype fp8 allocates fp8 KV and the report says so") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("--kv-cache-dtype fp8");
  INFO("vllm-bench output:\n" << run.output);

  // Parsed rather than reported as unknown: `vllm-bench` prints its usage and
  // exits 2 on an unrecognised argument.
  CHECK_FALSE(Contains(run.output, "unknown argument"));
  CHECK(run.status == 0);
  // The report exists and names both dtypes.
  REQUIRE(Contains(run.output, kRequestedLabel));
  REQUIRE(Contains(run.output, kResolvedLabel));
  // THE assertion: the value is read back out of the KV cache config the loader
  // sized, so this is the allocation and not an echo of the flag.
  CHECK(ResolvedValue(run.output) == "fp8_e4m3 (1-byte pages)");
}

TEST_CASE("vllm-bench: --kv-cache-dtype fp8_e4m3 is the same arm as fp8") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("--kv-cache-dtype fp8_e4m3");
  INFO("vllm-bench output:\n" << run.output);
  CHECK(run.status == 0);
  CHECK(ResolvedValue(run.output) == "fp8_e4m3 (1-byte pages)");
}

// ─── G2. The polarity control ────────────────────────────────────────────────
TEST_CASE("vllm-bench: no --kv-cache-dtype is the inert bf16 default") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("");
  INFO("vllm-bench output:\n" << run.output);

  CHECK(run.status == 0);
  REQUIRE(Contains(run.output, kResolvedLabel));
  // Without this case, a resolved line hard-coded to fp8 -- or a report that
  // echoed a flag it never passed on -- would pass G1.
  CHECK(ResolvedValue(run.output) == "bf16");
  CHECK_FALSE(Contains(run.output, "fp8_e4m3"));
  // The requested line still reports the default the operator did not type,
  // because a report with a blank field is a report a reader has to guess at.
  CHECK(LineValue(run.output, kRequestedLabel) == "auto");
}

TEST_CASE("vllm-bench: an explicit --kv-cache-dtype auto matches typing nothing") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("--kv-cache-dtype auto");
  INFO("vllm-bench output:\n" << run.output);
  CHECK(run.status == 0);
  CHECK(ResolvedValue(run.output) == "bf16");
}

// ─── G3. An unknown name is refused BY NAME, by the one owner of that list ───
TEST_CASE("vllm-bench: --kv-cache-dtype nvfp4 is refused and names itself") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("--kv-cache-dtype nvfp4");
  INFO("vllm-bench output:\n" << run.output);

  // The FLAG was accepted -- the refusal must come from ParseCacheDType, not
  // from argument parsing, or the harness would be carrying its own second copy
  // of the legal-name list.
  CHECK_FALSE(Contains(run.output, "unknown argument"));
  CHECK(run.status != 0);
  // Refused by name (kv_cache_dtype.h), so an operator reads back what they
  // asked for rather than a generic dtype rule. nvfp4 is owned by the
  // KV-NVFP4-TURBO row and is issue #2620; it is not implemented here.
  CHECK(Contains(run.output, "nvfp4"));
  CHECK(Contains(run.output, "cache_dtype"));
}

TEST_CASE("vllm-bench: a nonsense --kv-cache-dtype is refused, not defaulted") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("--kv-cache-dtype not_a_dtype");
  INFO("vllm-bench output:\n" << run.output);
  CHECK(run.status != 0);
  CHECK(Contains(run.output, "not_a_dtype"));
}

// ─── G4. A parseable-but-unserved name must never read as a bf16 measurement ─
TEST_CASE("vllm-bench: fp8_e5m2 either refuses or names itself, never bf16") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("--kv-cache-dtype fp8_e5m2");
  INFO("vllm-bench output:\n" << run.output);

  // `ParseCacheDType` ACCEPTS fp8_e5m2 (it is a real CacheDType) while no
  // attention block writes it, so this name is the one that can produce a
  // number nobody can attribute. Both outcomes are correct; silently measuring
  // bf16 under an fp8_e5m2 flag is not.
  CHECK_FALSE(Contains(run.output, "unknown argument"));
  if (run.status == 0) {
    CHECK(ResolvedValue(run.output) == "fp8_e5m2 (1-byte pages)");
  } else {
    CHECK(run.status != 0);
  }
}

// ─── G5. The `--model` arm: the flag reaches FromModelDir's resolver ─────────
//
// These two cases are a PAIR and neither means anything alone. Together they
// kill the mutation that deletes `params.kv_cache_dtype = cfg.kv_cache_dtype;`
// from the real-checkpoint branch, which every case above survives.
TEST_CASE("vllm-bench --model: no flag lets the checkpoint decide the KV dtype") {
  RequireBenchBinary();
  const DeclaringCheckpoint ckpt;
  const BenchRun run = RunBenchBinary("--model " + ckpt.path());
  INFO("vllm-bench output:\n" << run.output);

  // The checkpoint declared fp8 and nothing overrode it, so the loader says so.
  CHECK(Contains(run.output, kCheckpointDecided));
  // And then failed on the deliberately absent weights, which is the point at
  // which this fixture stops being a model.
  CHECK(run.status != 0);
}

TEST_CASE("vllm-bench --model: an explicit --kv-cache-dtype beats the checkpoint") {
  RequireBenchBinary();
  const DeclaringCheckpoint ckpt;
  const BenchRun run =
      RunBenchBinary("--model " + ckpt.path() + " --kv-cache-dtype bfloat16");
  INFO("vllm-bench output:\n" << run.output);

  // THE assertion. An explicit value is returned unchanged and the checkpoint is
  // never consulted, so the line the previous case asserts must be ABSENT. It
  // can only be absent if the flag actually reached `EngineParams` on this arm:
  // a dropped flag leaves "auto" in the resolver and the checkpoint wins again.
  CHECK_FALSE(Contains(run.output, kCheckpointDecided));
  CHECK_FALSE(Contains(run.output, "unknown argument"));
  CHECK(run.status != 0);
}
