// `ENG-RESIDENCY-CONFIG` W2, issue #1135 — does `vllm-cli --offload-config`
// actually reach the loader?
//
// WHY THIS RUNS A BINARY. `tests/vllm/entrypoints/test_weight_residency_reach.cpp`
// already drives `vllm_engine_load` with the same JSON string and asserts the
// install, so the ABI is gated. What was not gated, and what #1135 is about, is
// the FLAG: `vllm-cli` had no `--offload-config` at all, and a test that called
// the ABI from here would have passed on the day the flag was missing.
// .agents/reachability.md's rule is the same one: enter through the production
// entry point, which for this row's third entry point is the command line of a
// built executable.
//
// THE MODEL DIRECTORY IS DELIBERATELY NONEXISTENT. `LoadedEngine::FromModelDir`
// installs the residency document in its first statement block, ahead of every
// path and weight operation, so a load that fails on a missing checkpoint still
// runs the install and prints its line. That is what makes the chain observable
// without a checkpoint.
//
// THE MUTATION this file exists for: delete the `mp.offload_config = ...`
// assignment in `examples/cli/main.cpp` and CASE 1 goes red while every other
// residency suite stays green.
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

// `cmake -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_BUILD_EXAMPLES=OFF` is a legal
// configuration, and there `VLLM_CLI_BINARY` is the empty string above: there is
// no `vllm-cli` to run and this gate cannot be run at all.
//
// It must not FAIL there, and it must not PASS there either. A `REQUIRE` on the
// define reported four failed cases on a build that simply did not include the
// binary — a verdict about the configuration wearing the shape of a verdict about
// the code — and returning early would have printed doctest's
// `assertions: 0 ... Status: SUCCESS!` banner, which is the same trap issue #463
// files. So the process exits with CTest's SKIP_RETURN_CODE (77, registered on
// every test by `vllm_cpp_add_test` in tests/CMakeLists.txt) and `ctest` reports
// **Skipped**, which is the true result. Exiting rather than skipping one case is
// right because the define is a build-time constant: if it is empty, every case in
// this file is unrunnable, so the first one to ask ends the process for all of them.
[[noreturn]] void SkipGate() {
  std::fprintf(stderr,
               "\n*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "*** test_cli_offload_config: built without VLLM_CPP_BUILD_EXAMPLES,"
               " so there is no vllm-cli binary to run\n\n");
  std::fflush(stderr);
  std::exit(77);
}

// Called first in every case. Named rather than inlined so that a case added later
// cannot get the polarity wrong by writing a bare `REQUIRE`.
void RequireCliBinary() {
  if (std::string(kCliBinary).empty()) SkipGate();
}

// Combined stdout and stderr, plus the child's exit status. `vllm-cli` writes
// its own progress to stderr and the engine writes the install line there too,
// so both streams are read together.
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
    "--model /nonexistent/vllm-cpp/cli-offload-config --prompt hi "
    "--max-tokens 1";

// Printed by `LoadedEngine::FromModelDir` when it installs a residency document.
constexpr const char* kInstallLine = "engine: weight residency";

}  // namespace

TEST_CASE("vllm-cli: --offload-config's vllm_cpp half reaches the loader") {
  RequireCliBinary();
  const CliRun run = RunCli(
      std::string(kMissingModel) +
      R"( --offload-config '{"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},)"
      R"("device_fit":{"weight_budget_bytes":4096}}}')");
  INFO("vllm-cli output:\n" << run.output);

  // The flag was accepted rather than reported as unknown. `vllm-cli` prints its
  // usage and exits 2 on an unrecognised argument, so this is the assertion that
  // separates "the flag exists" from "the flag was parsed as a model path".
  CHECK_FALSE(Contains(run.output, "usage:"));
  CHECK(Contains(run.output, "vllm-cli: loading model from"));

  // ...and the document reached the install, naming the fields the operator set.
  CHECK(Contains(run.output, kInstallLine));
  CHECK(Contains(run.output, "mmap=on"));
  CHECK(Contains(run.output, "prefault=off"));
  CHECK(Contains(run.output, "device_weight_budget_bytes=4096"));

  // The load then failed on the deliberately missing checkpoint, which is what
  // makes the install observable without one.
  CHECK(Contains(run.output, "model load failed"));
  CHECK(run.status == 1);
}

TEST_CASE("vllm-cli: the MIRRORED half of the document reaches the loader too") {
  // #1135 is not specific to the `vllm_cpp` extension: vLLM's own `uva` half had
  // no way to reach this entry point either. The proof that it arrived is the
  // line `CreateWeightOffloader` prints when it CONSTRUCTS a backend
  // (`model_loader.cpp`, the `choice.offloader->moves_weights()` arm), which is
  // printed only for a document that selects one.
  RequireCliBinary();
  const std::string kOffloaderInstalled =
      "engine: offload_config installed UvaWeightOffloader";
  const CliRun run =
      RunCli(std::string("--model /nonexistent/vllm-cpp/cli-offload-uva ") +
             "--prompt hi --max-tokens 1 " +
             R"(--offload-config '{"offload_backend":"uva","uva":{"cpu_offload_gb":10}}')");
  INFO("vllm-cli output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, "usage:"));
  CHECK(Contains(run.output, kOffloaderInstalled));
  // A `vllm_cpp`-free document installs no residency, so the line must NOT
  // print. Without this the case would also pass on an implementation that
  // routed the mirrored half into the extension.
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK(Contains(run.output, "model load failed"));
  CHECK(run.status == 1);

  // The negative control: the same run without the flag prints neither line.
  const CliRun bare = RunCli(kMissingModel);
  INFO("vllm-cli output:\n" << bare.output);
  CHECK_FALSE(Contains(bare.output, kOffloaderInstalled));
}

TEST_CASE("vllm-cli: a mistyped key in the document fails the load, naming the key") {
  // The extension parser closes the whole document, and `vllm-cli` inherits that
  // by parsing nothing itself: the refusal comes out of `vllm_engine_load`. A
  // typo that quietly disabled this tier would be met as an out-of-memory kill
  // minutes later instead of as a message.
  RequireCliBinary();
  const CliRun run =
      RunCli(std::string(kMissingModel) +
             R"( --offload-config '{"vllm_cpp":{"device_fitt":{"weight_budget_bytes":1}}}')");
  INFO("vllm-cli output:\n" << run.output);

  CHECK(Contains(run.output, "unknown key \"vllm_cpp.device_fitt\""));
  CHECK(Contains(run.output, "expected one of: mmap expert_stream device_fit"));
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK(run.status == 1);
}

TEST_CASE("vllm-cli: no --offload-config prints no residency line at all") {
  // The inertness case. Nearly every run passes no such flag, and those runs must
  // print nothing new and install nothing.
  RequireCliBinary();
  const CliRun run = RunCli(kMissingModel);
  INFO("vllm-cli output:\n" << run.output);
  CHECK(Contains(run.output, "vllm-cli: loading model from"));
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK(run.status == 1);
}
