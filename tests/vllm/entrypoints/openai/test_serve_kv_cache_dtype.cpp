// KV-FP8 W3 (#1593) — the SERVER end of the chain: does `--kv-cache-dtype`
// actually reach `EngineParams` when a real `vllm serve` line carries it?
//
// This is the row's reachability gate for its own headline flag, in the sense of
// .agents/reachability.md. `tests/vllm/entrypoints/test_kv_cache_fp8_wiring.cpp`
// proves the resolver, the block arithmetic, the store and the read, and it
// enters through `LoadedEngine`; none of it touches argument parsing. Deleting
// BOTH the `--kv-cache-dtype` arm of `ParseArgs` and the
// `engine_params.kv_cache_dtype = args.kv_cache_dtype` assignment left that file
//19/19, 89/89 SUCCESS, and no other test in the tree named the flag. A flag
// nothing reads is exactly the shape AGENTS.md `## Nothing lands dead` is about.
//
// WHAT MAKES IT OBSERVABLE WITHOUT A CHECKPOINT. `LoadedEngine::FromModelDir`
// resolves `--kv-cache-dtype` against the checkpoint's own `kv_cache_quant_algo`
// in its second statement, before any weight operation, and it SAYS SO on stderr
// when the checkpoint is what decided. So a model directory that carries only a
// `hf_quant_config.json` produces that line and then fails on the absent weights.
// The line is present when no flag was typed, and ABSENT when one was — because
// an explicit value is returned unchanged and the checkpoint is never consulted
// (`vllm/utils/torch_utils.py:380-381`).
//
// That polarity is the gate. A run whose flag never arrived would carry the
// default "auto" into the resolver, the checkpoint would win, and CASE 2's line
// would appear. Deleting either half of the chain therefore reddens CASE 2.
//
// WHY A SUBPROCESS. `ParseArgs` reports a bad argument through `Usage()`, which
// calls `std::exit`, so an in-process call would take the test binary with it.
// Each case re-execs this binary into a skip-decorated child that calls
// `VllmServerMain` on argv assembled from `VLLM_TEST_SERVE_ARGS` — the harness
// tests/vllm/entrypoints/openai/test_serve_residency_config.cpp establishes and
// this file mirrors, including its no-spaces-in-arguments limitation.
#include <doctest/doctest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/server_main.h"

namespace {

// The gate checkpoint's own declaration, in the file ModelOpt 0.29.0 and before
// wrote it to. Transcribed from `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`
// @ `36f717a22990e82c54c1d48ee77c491b87825680` and trimmed to the keys the
// resolver reads; `test_kv_cache_fp8_wiring.cpp` carries the same bytes and the
// provenance note.
constexpr const char* kDeclaresFp8 =
    R"({"producer":{"name":"modelopt"},)"
    R"("quantization":{"quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":"FP8"}})";

// Printed by VllmServerMain AFTER ParseArgs returns. Its presence proves
// argument parsing succeeded and control reached engine construction.
constexpr const char* kPostParseBanner = "server: request logging";

// The loader's resolution line. It names the checkpoint as the decider, which is
// the only reason it can be used as a signal for "no explicit flag arrived".
constexpr const char* kCheckpointDecided =
    "the checkpoint declares kv_cache_quant_algo";

constexpr const char* kUnknownArgument = "server: unknown argument";

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// A throwaway model directory carrying only `hf_quant_config.json`. No weights,
// no tokenizer: the load is MEANT to fail, after the resolution stanza.
class DeclaringCheckpoint {
 public:
  DeclaringCheckpoint() {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllm_serve_kvfp8_" + std::to_string(counter++));
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
    std::filesystem::create_directories(path_);
    std::ofstream(path_ / "hf_quant_config.json", std::ios::binary)
        << kDeclaresFp8;
  }
  ~DeclaringCheckpoint() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  DeclaringCheckpoint(const DeclaringCheckpoint&) = delete;
  DeclaringCheckpoint& operator=(const DeclaringCheckpoint&) = delete;

  std::string str() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

struct ChildRun {
  std::string output;  // stdout + stderr, combined
  int status = -1;
};

ChildRun RunServer(const std::string& serve_args) {
  // Resolve our own path in the PARENT: popen runs under /bin/sh, so a literal
  // /proc/self/exe inside the command would resolve to the shell.
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';
  const std::string cmd = "VLLM_TEST_SERVE_ARGS='" + serve_args + "' " +
                          std::string(exe) +
                          " --no-skip --test-case='serve_kv_cache_dtype_child'"
                          " 2>&1";
  FILE* pipe = ::popen(cmd.c_str(), "r");
  REQUIRE(pipe != nullptr);
  ChildRun run;
  std::array<char, 4096> buf{};
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    run.output += buf.data();
  }
  const int closed = ::pclose(pipe);
  REQUIRE(closed != -1);
  run.status = WIFEXITED(closed) ? WEXITSTATUS(closed) : -1;
  return run;
}

std::vector<std::string> SplitOnSpaces(const std::string& text) {
  std::vector<std::string> out;
  std::string current;
  for (const char c : text) {
    if (c == ' ') {
      if (!current.empty()) out.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) out.push_back(current);
  return out;
}

}  // namespace

// The CHILD case, filtered out of a normal run and executed only when a parent
// re-execs it by name.
TEST_CASE("serve_kv_cache_dtype_child" * doctest::skip()) {
  const char* raw = std::getenv("VLLM_TEST_SERVE_ARGS");
  REQUIRE(raw != nullptr);
  std::vector<std::string> args{"vllm-server"};
  for (std::string& token : SplitOnSpaces(raw)) {
    args.push_back(std::move(token));
  }
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) argv.push_back(arg.data());
  const int rc = vllm::entrypoints::openai::VllmServerMain(
      static_cast<int>(argv.size()), argv.data());
  std::cout << "SERVE_RC=" << rc << "\n" << std::flush;
  // Leave immediately: doctest's own summary would otherwise be mistaken for the
  // server's verdict, and the parent reads this process's exit status.
  std::exit(0);
}

// CASE 1 — THE CONTROL. No `--kv-cache-dtype` at all, and the checkpoint's own
// declaration decides. It establishes that the signal CASE 2 waits for is really
// produced by this directory, so CASE 2's absence means "the flag won" rather
// than "the line never prints here".
TEST_CASE("serve: with no --kv-cache-dtype the checkpoint's declaration decides") {
  const DeclaringCheckpoint model;
  const ChildRun run = RunServer("--model " + model.str());
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK(Contains(run.output, kCheckpointDecided));
  // The RESOLVED value, which is what every consumer downstream sees.
  CHECK(Contains(run.output, "fp8_e4m3"));
  // The load then failed on the deliberately absent weights, which is what makes
  // the stanza observable without a checkpoint.
  CHECK(Contains(run.output, "SERVE_RC="));
  CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));
  CHECK(run.status == 0);
}

// CASE 2 — THE REACHABILITY CASE. The same directory, one typed
// `--kv-cache-dtype`, and the checkpoint is not consulted at all. This is the
// case that reddens when the `--kv-cache-dtype` arm of `ParseArgs` is deleted,
// and again when `engine_params.kv_cache_dtype = args.kv_cache_dtype` is
// deleted: either way the default "auto" arrives, the checkpoint wins, and the
// line comes back.
TEST_CASE("serve: --kv-cache-dtype reaches EngineParams and OUTRANKS the checkpoint") {
  const DeclaringCheckpoint model;
  const ChildRun run =
      RunServer("--model " + model.str() + " --kv-cache-dtype bfloat16");
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK_FALSE(Contains(run.output, kCheckpointDecided));
  CHECK(Contains(run.output, "SERVE_RC="));
  CHECK(run.status == 0);
}

// CASE 3 — and `fp8` is a value the flag accepts, not only a value the loader
// resolves to. Without it CASE 2 could be satisfied by a parser that took the
// flag and refused every fp8 spelling, which is the one arm this row ships for.
TEST_CASE("serve: --kv-cache-dtype fp8 parses and reaches the loader") {
  const DeclaringCheckpoint model;
  const ChildRun run =
      RunServer("--model " + model.str() + " --kv-cache-dtype fp8");
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK(Contains(run.output, "server: loading model from"));
  CHECK_FALSE(Contains(run.output, kCheckpointDecided));
  CHECK(run.status == 0);
}
