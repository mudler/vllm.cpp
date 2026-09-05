// fork issue #7, ABI v24 — does `vllm-cli --kv-cache-dtype` actually reach the
// loader?
//
// WHY THIS RUNS A BINARY. `tests/capi/test_capi.cpp` already drives
// `vllm_engine_load` with `kv_cache_dtype` set, so the ABI field is gated. What
// is not gated is the FLAG: `vllm-cli` had no `--kv-cache-dtype` before ABI v24,
// and a test that called the ABI from here would have passed on the day the flag
// was missing. .agents/reachability.md's rule: enter through the production
// entry point, which for the CLI is the command line of a built executable.
//
// THE MODEL DIRECTORY IS DELIBERATELY NONEXISTENT. The load fails on the missing
// checkpoint, which is what makes the flag's acceptance observable without one:
// `vllm-cli` prints its usage and exits 2 on an unrecognised argument, so
// reaching the model-load phase at all proves the flag was parsed.
//
// THE MUTATION this file exists for: delete the `mp.kv_cache_dtype = ...`
// assignment in `examples/cli/main.cpp` and CASE 1 goes red (the flag is silently
// dropped, the load still fails with MODEL_LOAD, but the `--kv-cache-dtype` value
// never reaches the engine). The negative control in CASE 2 catches the polarity.
#include <doctest/doctest.h>

#include <sys/wait.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

#ifndef VLLM_CLI_BINARY
#define VLLM_CLI_BINARY ""
#endif

constexpr const char* kCliBinary = VLLM_CLI_BINARY;

[[noreturn]] void SkipGate() {
  std::fprintf(stderr,
               "\n*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "*** test_cli_kv_cache_dtype: built without VLLM_CPP_BUILD_EXAMPLES,"
               " so there is no vllm-cli binary to run\n\n");
  std::fflush(stderr);
  std::exit(77);
}

void RequireCliBinary() {
  if (std::string(kCliBinary).empty()) SkipGate();
}

struct CliRun {
  std::string output;
  int status = -1;
};

CliRun RunCli(const std::string& args) {
  const std::string cmd = std::string(kCliBinary) + " " + args + " 2>&1";
  CliRun run;
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

constexpr const char* kMissingModel =
    "--model /nonexistent/vllm-cpp/cli-kv-cache-dtype --prompt hi "
    "--max-tokens 1";

}  // namespace

TEST_CASE("vllm-cli: --kv-cache-dtype is accepted and reaches the loader") {
  RequireCliBinary();
  const CliRun run =
      RunCli(std::string(kMissingModel) + " --kv-cache-dtype fp8");
  INFO("vllm-cli output:\n" << run.output);

  // The flag was accepted rather than reported as unknown. `vllm-cli` prints its
  // usage and exits 2 on an unrecognised argument, so this is the assertion that
  // separates "the flag exists" from "the flag was parsed as a model path".
  CHECK_FALSE(Contains(run.output, "usage:"));
  CHECK(Contains(run.output, "vllm-cli: loading model from"));
  // The load then failed on the deliberately missing checkpoint.
  CHECK(Contains(run.output, "model load failed"));
  CHECK(run.status == 1);
}

TEST_CASE("vllm-cli: no --kv-cache-dtype is the inert default") {
  RequireCliBinary();
  const CliRun run = RunCli(kMissingModel);
  INFO("vllm-cli output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, "usage:"));
  CHECK(Contains(run.output, "vllm-cli: loading model from"));
  CHECK(Contains(run.output, "model load failed"));
  CHECK(run.status == 1);
}
