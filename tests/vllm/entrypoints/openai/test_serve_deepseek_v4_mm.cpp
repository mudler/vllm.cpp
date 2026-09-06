// MODEL-MM-deepseek-v4 W5 (#2411) — DOES `vllm serve` HAND THE SEAM THE TWO
// THINGS IT NEEDS?
//
// `tests/vllm/entrypoints/openai/test_deepseek_v4_mm_chat.cpp` builds the
// install context field by field and proves everything downstream of it. None
// of it touches `server_main.cpp`, and the two fields this wave ADDED to
// `MultiModalChatContext` are assigned there and nowhere else on the server
// path:
//
//   mm_ctx.config      = &loaded->config();
//   mm_ctx.mmproj_path = args.mmproj_path;
//
// Delete either and that suite stays green, because it fills the context in
// itself. That is exactly the UNPASSED PARAMETER shape `.agents/reachability.md`
// names: a function grows an argument and every call site takes the default.
// This file is the gate for those two lines, and it enters through the real
// `VllmServerMain` on a command line a user types.
//
// WHAT MAKES IT OBSERVABLE. `InstallMultiModalChatSeam` announces every outcome
// on stderr and there is no arm that installs nothing on a model that says it is
// multimodal, so the two outcomes are two different lines and each names why.
//
// WHY A SUBPROCESS. `ParseArgs` reports a bad argument through `Usage()`, which
// calls `std::exit`, so an in-process call would take the test binary with it.
// Each case re-execs this binary into a skip-decorated child that calls
// `VllmServerMain` on argv assembled from `VLLM_TEST_SERVE_ARGS` — the harness
// `test_serve_residency_config.cpp` established and `test_serve_kv_cache_dtype
// .cpp` mirrors, including its no-spaces-in-arguments limitation.
#include <doctest/doctest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "deepseek_v4_lang_gguf_fixture.h"
#include "deepseek_v4_mmproj_fixture.h"
#include "vllm/entrypoints/openai/server_main.h"

namespace {

// Printed by VllmServerMain AFTER ParseArgs returns. Its presence proves
// argument parsing succeeded and control reached engine construction.
constexpr const char* kPostParseBanner = "server: request logging";
constexpr const char* kUnknownArgument = "server: unknown argument";

// The two install outcomes, in `mm_chat_registry.cpp`'s own words.
constexpr const char* kWired = "multimodal chat seam wired for architecture";
constexpr const char* kUnavailable = "multimodal chat seam UNAVAILABLE for architecture";

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

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
                          " --no-skip --test-case='serve_deepseek_v4_mm_child'"
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

// The projector geometry, at the PINNED processor's patch size. `output` must be
// the language model's hidden width; `patch` must be 14, because the seam builds
// the pinned processor and the encoder refuses a feature width the projector
// does not want.
dsv4_mmproj_test::Dims ProjDims() {
  dsv4_mmproj_test::Dims d;
  d.output = dsv4_lang_test::kH;
  d.patch = 14;
  return d;
}

// The port every case asks for. Binding it needs privileges this test does not
// have, so the server exits AFTER the install announcement instead of accepting
// connections -- which is the only reason a serve line can be run to completion
// inside a unit test at all.
constexpr const char* kUnbindablePort = "1";

}  // namespace

// The CHILD case, filtered out of a normal run and executed only when a parent
// re-execs it by name.
TEST_CASE("serve_deepseek_v4_mm_child" * doctest::skip()) {
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

// CASE 1 — THE CONTROL. The same checkpoint with NO `--mmproj`. The seam
// REFUSES at install and says why, which is what makes CASE 2's line a
// statement about the second file rather than about the architecture.
TEST_CASE("serve: a DeepSeek-V4 vision checkpoint with no --mmproj refuses the seam by name") {
  gguf_test::TempFile lang(dsv4_lang_test::BuildDeepseek4Gguf(
      /*vision=*/true, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/true));
  const ChildRun run = RunServer("--model " + lang.path() + " --port " +
                                 kUnbindablePort);
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK(Contains(run.output, kUnavailable));
  CHECK(Contains(run.output, "DeepseekV4ForCausalLM"));
  CHECK(Contains(run.output, "--mmproj"));
  CHECK_FALSE(Contains(run.output, kWired));
  CHECK(run.status == 0);
}

// CASE 2 — THE REACHABILITY CASE, and it is what this file exists for.
//
// Delete `mm_ctx.mmproj_path = args.mmproj_path` in `server_main.cpp` and this
// reddens: the factory cannot tell the two-file vehicle from a text checkpoint,
// refuses, and CASE 1's line comes back. Delete
// `mm_ctx.config = &loaded->config()` and it reddens differently: the factory
// refuses an incomplete context, because a `.gguf` has no `config.json` for
// `config_path` to name and the processor is keyed on `vocab_size`.
TEST_CASE("serve: --mmproj reaches the multimodal chat install and wires the DeepSeek seam") {
  gguf_test::TempFile lang(dsv4_lang_test::BuildDeepseek4Gguf(
      /*vision=*/true, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/true));
  gguf_test::TempFile proj(dsv4_mmproj_test::Build(ProjDims()));
  const ChildRun run = RunServer("--model " + lang.path() + " --mmproj " +
                                 proj.path() + " --port " + kUnbindablePort);
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK(Contains(run.output, kWired));
  CHECK(Contains(run.output, "DeepseekV4ForCausalLM"));
  // The seam's own detail line, which names the processor and the second file.
  CHECK(Contains(run.output, "DeepSeek-V4 Flash-Vision processor"));
  CHECK(Contains(run.output, proj.path()));
  CHECK_FALSE(Contains(run.output, kUnavailable));
  CHECK(run.status == 0);
}
