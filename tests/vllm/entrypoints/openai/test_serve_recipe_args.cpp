// SERVE-RECIPE-ARGS (#606) — the accepted-and-inert serve-argument seam.
//
// Ported from / grounded in vllm/entrypoints/openai/cli_args.py:105
// (`enable_auto_tool_choice: bool = False`) and :395
// (`--enable-auto-tool-choice` without `--tool-call-parser` raises a
// `TypeError`) @ 555967922 — the pinned oracle in .agents/upstream-sync.md.
//
// THE GATE, in one sentence: a published `vllm serve` line must reach model
// load, and everything else must keep aborting. 89 of the 157 official
// vllm-project/recipes commands pass `--enable-auto-tool-choice` and 82 pass
// `--trust-remote-code`; both are no-ops here, so both must parse. Nothing
// else changes: an unlisted flag still hits `server_main.cpp`'s
// "unknown argument" path, because silently swallowing
// `--tensor-parallel-size` would let a user believe they got tensor
// parallelism. That second case is the load-bearing one — a mutation turning
// the table into a catch-all must turn it RED.
//
// WHY A SUBPROCESS. `ParseArgs` reports a bad argument through `Usage()`,
// which calls `std::exit`; an in-process call would take the whole test binary
// with it, so the abort cases are unobservable from inside. Each case
// therefore RE-EXECS THIS TEST BINARY — the pattern established by
// tests/vllm/v1/test_none_hash_determinism.cpp — into a skip-decorated child
// case that calls the REAL `VllmServerMain` on argv assembled from
// `VLLM_TEST_SERVE_ARGS`, then asserts on the child's combined output and exit
// status.
//
// The model directory is deliberately nonexistent: it makes "argument parsing
// succeeded" observable (the post-parse banner is printed, and the child
// returns a nonzero rc from the engine-load failure) without the test ever
// needing a checkpoint or binding a port.
#include <doctest/doctest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/server_main.h"

namespace {

// A path that cannot exist, so the child always fails at load rather than
// serving forever.
constexpr const char* kMissingModel =
    "--model /nonexistent/vllm-cpp/serve-recipe-args";

// Printed by VllmServerMain AFTER ParseArgs returns (server_main.cpp's request
// logging banner). Its presence is the proof that argument parsing succeeded
// and control reached engine construction.
constexpr const char* kPostParseBanner = "server: request logging";

// The existing guard's message. It must survive this change verbatim.
constexpr const char* kUnknownArgument = "server: unknown argument";

struct ChildRun {
  std::string output;  // stdout + stderr, combined
  int status = -1;     // the child process's exit status
};

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// Run this same test binary again, in a fresh process, with `serve_args` handed
// to the child through the environment, and return what it printed plus how it
// exited.
ChildRun RunServer(const std::string& serve_args) {
  // Resolve our own path HERE, in the parent: popen runs the command under
  // /bin/sh, so a literal /proc/self/exe inside it would resolve to the shell.
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';
  // --no-skip is required: the child case is skip-decorated so a normal run
  // never executes it.
  const std::string cmd = "VLLM_TEST_SERVE_ARGS='" + serve_args + "' " +
                          std::string(exe) +
                          " --no-skip --test-case='serve_recipe_args_child'"
                          " 2>&1";
  FILE* pipe = ::popen(cmd.c_str(), "r");
  REQUIRE(pipe != nullptr);
  ChildRun run;
  std::array<char, 4096> buf{};
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) !=
         nullptr) {
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

// The CHILD case. It is filtered out of a normal run (doctest skips
// `[.]`-decorated cases unless explicitly named), so it only executes when a
// parent re-execs it by name.
TEST_CASE("serve_recipe_args_child" * doctest::skip()) {
  const char* raw = std::getenv("VLLM_TEST_SERVE_ARGS");
  REQUIRE(raw != nullptr);
  std::vector<std::string> args{"vllm-server"};
  for (std::string& token : SplitOnSpaces(raw)) {
    args.push_back(std::move(token));
  }
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  const int rc = vllm::entrypoints::openai::VllmServerMain(
      static_cast<int>(argv.size()), argv.data());
  std::cout << "SERVE_RC=" << rc << "\n" << std::flush;
  // Leave immediately: doctest's own summary would otherwise be mistaken for
  // the server's verdict, and the parent reads this process's exit status.
  std::exit(0);
}

// CASE 1 — each listed flag starts the server: argument parsing succeeds and
// the engine reaches load.
TEST_CASE("accepted-and-inert serve flags reach model load") {
  for (const char* flag :
       {"--enable-auto-tool-choice", "--trust-remote-code"}) {
    CAPTURE(flag);
    const ChildRun run =
        RunServer(std::string(kMissingModel) + " " + flag);
    INFO("child output:\n" << run.output);
    CHECK_FALSE(Contains(run.output, kUnknownArgument));
    // ParseArgs returned rather than exiting through Usage().
    CHECK(Contains(run.output, kPostParseBanner));
    // ...and control reached MODEL LOAD, which failed on the deliberately
    // missing checkpoint. A nonzero rc printed by the child is exactly
    // "argument parsing succeeded, the engine tried to load".
    CHECK(Contains(run.output, "server: loading model from"));
    CHECK(Contains(run.output, "SERVE_RC="));
    CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));
    // The child ran to completion instead of being killed by Usage()'s exit.
    CHECK(run.status == 0);
  }
}

// CASE 2 — THE LOAD-BEARING ONE. The seam is enumerated, not a catch-all: a
// flag that is not in the table still aborts with the existing message. A
// mutation that widens the table into a catch-all must turn this RED.
TEST_CASE("an unlisted unknown serve flag still aborts") {
  // --tensor-parallel-size is the spec's own example: it is inert here because
  // we LACK the capability, so accepting it would tell a user they got tensor
  // parallelism when they did not.
  for (const char* flag : {"--tensor-parallel-size", "--totally-made-up-flag"}) {
    CAPTURE(flag);
    const ChildRun run = RunServer(std::string(kMissingModel) + " " + flag);
    INFO("child output:\n" << run.output);
    CHECK(Contains(run.output,
                   std::string(kUnknownArgument) + " '" + flag + "'"));
    // Aborted in ParseArgs: the post-parse banner never printed and the child
    // never reached VllmServerMain's return.
    CHECK_FALSE(Contains(run.output, kPostParseBanner));
    CHECK_FALSE(Contains(run.output, "SERVE_RC="));
    // Usage(argv[0], 2).
    CHECK(run.status == 2);
  }
}

// CASE 3 — inert is not unvalidated. Mirrors
// vllm/entrypoints/openai/cli_args.py:395, where `--enable-auto-tool-choice`
// with no tool-call parser is a TypeError. `--tool-call-parser none` is our
// spelling of upstream's unset parser.
TEST_CASE("--enable-auto-tool-choice with --tool-call-parser none fails") {
  const ChildRun run = RunServer(std::string(kMissingModel) +
                                 " --enable-auto-tool-choice"
                                 " --tool-call-parser none");
  INFO("child output:\n" << run.output);
  CHECK(Contains(run.output,
                 "--enable-auto-tool-choice requires --tool-call-parser"));
  CHECK_FALSE(Contains(run.output, kPostParseBanner));
  CHECK_FALSE(Contains(run.output, "SERVE_RC="));
  CHECK(run.status == 2);

  // The same flag WITHOUT the contradiction still reaches load, so the failure
  // above is the validation firing and not the flag being rejected.
  const ChildRun ok = RunServer(std::string(kMissingModel) +
                                " --enable-auto-tool-choice"
                                " --tool-call-parser hermes");
  INFO("child output:\n" << ok.output);
  CHECK_FALSE(Contains(ok.output, kUnknownArgument));
  CHECK(Contains(ok.output, kPostParseBanner));
}

// CASE 4 — accepting is ANNOUNCED. A log reader must learn the flag did
// nothing rather than inferring that it worked, so the notice names both the
// flag and the reason it is inert.
TEST_CASE("each accepted serve flag announces itself and its reason") {
  const ChildRun tool_choice =
      RunServer(std::string(kMissingModel) + " --enable-auto-tool-choice");
  INFO("child output:\n" << tool_choice.output);
  CHECK(Contains(tool_choice.output, "--enable-auto-tool-choice"));
  CHECK(Contains(tool_choice.output, "no effect"));
  CHECK(Contains(tool_choice.output, "--tool-call-parser resolves"));

  const ChildRun trust =
      RunServer(std::string(kMissingModel) + " --trust-remote-code");
  INFO("child output:\n" << trust.output);
  CHECK(Contains(trust.output, "--trust-remote-code"));
  CHECK(Contains(trust.output, "no effect"));
  CHECK(Contains(trust.output, "no remote code to trust"));

  // A flag that was NOT passed is never announced: the notice tracks actual
  // use, so it cannot be read as a list of everything the table tolerates.
  CHECK_FALSE(Contains(tool_choice.output, "--trust-remote-code"));
  CHECK_FALSE(Contains(trust.output, "--enable-auto-tool-choice"));
}
