// `ENG-RESIDENCY-CONFIG` (issue #1110) — the SERVER end of the chain: does
// `--offload-config`'s `vllm_cpp` extension actually reach the loader when a real
// `vllm serve` line carries it?
//
// This is the row's reachability gate in the sense of
// .agents/reachability.md. `test_weight_residency_config.cpp` proves the parser,
// `test_weight_residency_reach.cpp` proves EngineParams -> install; neither of
// them touches argument parsing, and a flag that nothing reads would pass both.
// So this file runs the REAL `VllmServerMain` on a real argv and reads what the
// engine printed.
//
// WHY A SUBPROCESS. `ParseArgs` reports a bad argument through `Usage()`, which
// calls `std::exit`, so an in-process call would take the test binary with it.
// Each case re-execs this binary into a skip-decorated child that calls
// `VllmServerMain` on argv assembled from `VLLM_TEST_SERVE_ARGS` — the pattern
// established by tests/vllm/v1/test_none_hash_determinism.cpp and used by
// tests/vllm/entrypoints/openai/test_serve_recipe_args.cpp, whose harness this
// mirrors.
//
// THE MODEL DIRECTORY IS DELIBERATELY NONEXISTENT: it makes "the flag was parsed
// and the engine installed the config" observable without a checkpoint or a bound
// port, because the install sits ahead of every path and weight operation in
// `LoadedEngine::FromModelDir`.
//
// THE ARGUMENT VALUES CARRY NO SPACES. The child splits `VLLM_TEST_SERVE_ARGS` on
// spaces, so every JSON document below is written compactly. A document with a
// space in it is a harness limitation, not a product one.
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

constexpr const char* kMissingModel =
    "--model /nonexistent/vllm-cpp/serve-residency-config";

// Printed by VllmServerMain AFTER ParseArgs returns. Its presence proves argument
// parsing succeeded and control reached engine construction.
constexpr const char* kPostParseBanner = "server: request logging";

// The install line. It names the DOCUMENT that was installed — the fields the
// operator set — and, on a second line, every variable that would win over one of
// them. It does NOT name resolved values: the streaming answer is cached the first
// time it is asked, so resolving it at install would move that decision ahead of the
// weight load. That constraint binds `expert_stream` alone; the line reports the
// document for all six fields so it reports one kind of thing rather than a mixture.
// The pair of lines is what lets a run whose document was overridden say so; see
// CASE 5.
constexpr const char* kInstallLine = "engine: weight residency";

constexpr const char* kUnknownArgument = "server: unknown argument";

struct ChildRun {
  std::string output;  // stdout + stderr, combined
  int status = -1;
};

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// `env_prefix` is prepended to the command as `NAME=VALUE ` assignments. It is a
// SEPARATE parameter and not part of `serve_args` on purpose: the child splits
// VLLM_TEST_SERVE_ARGS on spaces and would turn an assignment into an unknown
// server argument, which is a green-looking failure — the run would abort in
// ParseArgs and every "the override won" assertion would then be checking the
// absence of a line that was never going to print.
ChildRun RunServer(const std::string& serve_args,
                   const std::string& env_prefix = "") {
  // Resolve our own path in the PARENT: popen runs under /bin/sh, so a literal
  // /proc/self/exe inside the command would resolve to the shell.
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';
  const std::string cmd = env_prefix + "VLLM_TEST_SERVE_ARGS='" + serve_args +
                          "' " + std::string(exe) +
                          " --no-skip --test-case='serve_residency_child'"
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
TEST_CASE("serve_residency_child" * doctest::skip()) {
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

// CASE 1 — THE REACHABILITY CASE. `--offload-config` carrying ONLY the vllm_cpp
// extension parses, reaches the loader, and the loader installs it. The
// reachability mutation (delete the install call site in
// `LoadedEngine::FromModelDir`) turns this red.
TEST_CASE("serve: --offload-config's vllm_cpp extension reaches the loader") {
  const ChildRun run = RunServer(
      std::string(kMissingModel) +
      R"( --offload-config {"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},"expert_stream":{"enabled":true,"slots":8000}}})");
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK(Contains(run.output, "server: loading model from"));

  // The install happened, and it named the FIELDS OF THE DOCUMENT rather than
  // merely announcing itself. A line that said only "installed" could not tell an
  // operator which half of a two-tier document had reached this tier, and CASE 5's
  // override note would have nothing to sit beside.
  CHECK(Contains(run.output, kInstallLine));
  CHECK(Contains(run.output, "mmap=on"));
  CHECK(Contains(run.output, "prefault=off"));
  CHECK(Contains(run.output, "expert_stream=on"));
  CHECK(Contains(run.output, "expert_stream_slots=8000"));

  // The load then failed on the deliberately missing checkpoint, which is what
  // makes the install observable without a checkpoint.
  CHECK(Contains(run.output, "SERVE_RC="));
  CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));
  CHECK(run.status == 0);
}

// CASE 2 — the mirrored half is untouched. The same flag, carrying only vLLM's
// own keys, must behave exactly as it did before this row: accepted, and NO
// residency install.
TEST_CASE("serve: an --offload-config with only mirrored keys installs no residency") {
  const ChildRun run =
      RunServer(std::string(kMissingModel) +
                R"( --offload-config {"offload_backend":"uva","uva":{"cpu_offload_gb":10}})");
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK(Contains(run.output, "SERVE_RC="));
  CHECK(run.status == 0);
}

// CASE 3 — and no `--offload-config` at all is silent. This is the inertness case:
// the overwhelming majority of runs pass no such flag, and they must print nothing
// new and install nothing.
TEST_CASE("serve: no --offload-config prints no residency line at all") {
  const ChildRun run = RunServer(kMissingModel);
  INFO("child output:\n" << run.output);
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK(run.status == 0);
}

// CASE 4 — THE OTHER LOAD-BEARING ONE. A mistyped extension key must abort at
// STARTUP, before the multi-GB load, with a message naming the key. The mirrored
// parser ignores a key it does not know — which is what lets this extension share
// the flag — so without an explicit refusal `{"vllm_cpp":{"mmapp":...}}` would
// start a server silently running this tier at its defaults, and the operator would
// discover it as an out-of-memory kill rather than as an error.
TEST_CASE("serve: a mistyped vllm_cpp key aborts at startup, naming the key") {
  const ChildRun run =
      RunServer(std::string(kMissingModel) +
                R"( --offload-config {"vllm_cpp":{"mmapp":{"enabled":true}}})");
  INFO("child output:\n" << run.output);

  // Refused by NAME, and the message says what was expected instead. "unknown
  // key" alone would leave an operator hunting through a nested document.
  CHECK(Contains(run.output, "server: fatal:"));
  CHECK(Contains(run.output, "unknown key \"vllm_cpp.mmapp\""));
  CHECK(Contains(run.output, "expected one of: mmap expert_stream"));
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));

  // ...and it died BEFORE any model I/O. The marker is the loader's OWN first
  // complaint about the missing checkpoint, which the well-formed document
  // reaches and the mistyped one never does. Asserting the ABSENCE of
  // `server: loading model from` would gate the parse's POSITION rather than its
  // effect, and W2 moved that position: the parse now runs ahead of the
  // architecture branch, so the abort happens before that line instead of after
  // it (#1135). This assertion is stable across both positions, which is why it
  // is the one this case makes.
  const std::string kLoaderReached = "model path is not a directory";
  CHECK_FALSE(Contains(run.output, kLoaderReached));

  const ChildRun ok =
      RunServer(std::string(kMissingModel) +
                R"( --offload-config {"vllm_cpp":{"mmap":{"enabled":true}}})");
  INFO("child output:\n" << ok.output);
  CHECK(Contains(ok.output, kInstallLine));
  CHECK(Contains(ok.output, kLoaderReached));
}

// CASE 4b — the TOP-LEVEL misspelling, on the real server. This is the one the
// first pass shipped as an accept: `vllm-cpp` with a hyphen parsed to an empty
// config, the server started, and the tier the operator asked for was off with no
// diagnostic anywhere. The hyphen is the likeliest spelling of all, because every
// flag around it is hyphenated, and the consequence is an out-of-memory kill
// minutes later rather than a message at startup.
TEST_CASE("serve: a misspelled TOP-LEVEL key aborts at startup, naming the key") {
  const ChildRun run =
      RunServer(std::string(kMissingModel) +
                R"( --offload-config {"vllm-cpp":{"mmap":{"enabled":true}}})");
  INFO("child output:\n" << run.output);

  CHECK(Contains(run.output, "server: fatal:"));
  CHECK(Contains(run.output, "unknown key \"vllm-cpp\""));
  // The message lists the four keys the document may carry, so the operator can
  // see the spelling that was meant.
  CHECK(Contains(run.output,
                 "expected one of: offload_backend uva prefetch vllm_cpp"));
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK_FALSE(Contains(run.output, "model path is not a directory"));
  CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));
}

// CASE 5 — the environment WINS over the document, and the server SAYS SO at
// install time. The precedence is a promise this row makes to a benchmark arm in
// flight: these variables exist so an arm is switchable without restarting the
// server with a new document. A unit test pins the resolver; this pins that the
// server honours it AND reports it, which is where the promise is actually kept —
// a config silently shadowed by a variable somebody exported three weeks ago is
// the one way this precedence hurts, and one line at startup is the whole
// mitigation.
//
// The install cannot print the RESOLVED value, and that is not a shortcut: every
// knob resolves lazily, during weight load, through a static that latches on first
// use, so resolving at install would move the latch ahead of the load and change
// the very ordering this row is careful about. What it can do without resolving
// anything is look at whether an overriding variable is present, which is exactly
// the fact the operator is missing.
TEST_CASE("serve: an exported VT_ override beats the document, and is reported") {
  const std::string args =
      std::string(kMissingModel) +
      R"( --offload-config {"vllm_cpp":{"expert_stream":{"enabled":true,"slots":8000}}})";

  const ChildRun overridden = RunServer(args, "VT_MOE_EXPERT_STREAM=0 ");
  INFO("child output:\n" << overridden.output);
  // The document is still recorded and reported...
  CHECK(Contains(overridden.output, kInstallLine));
  CHECK(Contains(overridden.output, "expert_stream=on"));
  // ...and the override is named, by variable and by field.
  CHECK(Contains(overridden.output, "VT_MOE_EXPERT_STREAM"));
  CHECK(Contains(overridden.output, "OVERRIDES"));

  // Without the variable there is no override note, so the note above is the
  // variable being seen and not a line that always prints.
  const ChildRun clean = RunServer(args);
  INFO("child output:\n" << clean.output);
  CHECK(Contains(clean.output, kInstallLine));
  CHECK(Contains(clean.output, "expert_stream=on"));
  CHECK_FALSE(Contains(clean.output, "OVERRIDES"));
}

// ─── W2: the two SERVER entry points the document did not reach (#1135) ──────
//
// The cases above all drive the text-generation path. `--offload-config` was
// parsed inside that path, AFTER the architecture branch, so the pooling and
// transcription-only branches built their engine parameters without it and said
// nothing. That dropped both halves of the document — vLLM's mirrored
// `uva`/`prefetch` weight offload and the `vllm_cpp` residency extension — and had
// done so since before the extension existed.
//
// The parse now happens ONCE, ahead of the branch. These cases drive the two
// branches through the REAL `VllmServerMain`, which is the only thing that can
// tell "the pooling branch takes the document" from "the loader takes a document
// somebody handed it".
//
// A MODEL DIRECTORY IS NEEDED HERE, unlike above: the branch is chosen by reading
// `architectures` out of `config.json`, so a nonexistent path takes neither
// branch. The directory holds that one file and nothing else, so the load still
// fails immediately after the install — which is what keeps the install
// observable without a checkpoint.

namespace {

// A directory holding one `config.json` with the given `architectures`. Created
// in the PARENT and left in place for the re-exec'd child to read. The name has
// no spaces, because the child splits `VLLM_TEST_SERVE_ARGS` on them.
std::string MakeArchDir(const char* arch) {
  std::string tmpl = std::filesystem::temp_directory_path().string() +
                     "/vllm-cpp-serve-arch-XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  const char* made = ::mkdtemp(buf.data());
  REQUIRE(made != nullptr);
  const std::string dir(made);
  std::ofstream cfg(dir + "/config.json");
  REQUIRE(cfg.good());
  cfg << R"({"architectures":[")" << arch << R"("],"model_type":"llama"})";
  cfg.close();
  return dir;
}

// The pooling architecture registered in this tree: `REGISTER_VLLM_MODEL` names
// it in `llama_embedding_registry.cpp`, whose `kLlamaEmbeddingInfo` sets
// `is_pooling_model = true`.
constexpr const char* kPoolingArch = "LlamaModel";
// The transcription-only architecture: `REGISTER_VLLM_MODEL` names it in
// `parakeet_registry.cpp`, whose `kParakeetInfo` sets
// `supports_transcription_only = true`.
constexpr const char* kTranscriptionArch = "ParakeetForCTC";

constexpr const char* kPoolingBanner = "server: pooling (embedding) model";
constexpr const char* kTranscriptionBanner =
    "server: transcription-only model";

}  // namespace

TEST_CASE("serve: the POOLING path installs the residency document") {
  const std::string dir = MakeArchDir(kPoolingArch);
  const ChildRun run = RunServer(
      "--model " + dir +
      R"( --offload-config {"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},"device_fit":{"weight_budget_bytes":4096}}})");
  INFO("child output:\n" << run.output);

  // It really is the pooling branch, and not the text path taking the document
  // as it always did. Without this the case would pass on a build where the
  // architecture peek failed and everything fell through to the text path.
  CHECK(Contains(run.output, kPoolingBanner));

  // ...and that branch's `EngineParams` carried the document to the loader.
  CHECK(Contains(run.output, kInstallLine));
  CHECK(Contains(run.output, "mmap=on"));
  CHECK(Contains(run.output, "prefault=off"));
  CHECK(Contains(run.output, "device_weight_budget_bytes=4096"));
  CHECK(run.status == 0);
}

TEST_CASE("serve: the POOLING path takes the MIRRORED half too, and refuses a typo in it") {
  // #1135 is not specific to the extension: `uva`/`prefetch` was dropped on this
  // branch as well, and for longer. The proof that it arrived is the line
  // `CreateWeightOffloader` prints when it CONSTRUCTS a backend
  // (`model_loader.cpp`, the `choice.offloader->moves_weights()` arm). That line
  // is printed only for a document that selects one, so it cannot appear for an
  // `EngineParams` whose `offload_config` is unset — which is exactly what this
  // branch used to build. It is not the totality REFUSAL
  // (`RefuseUnsupportedWeightOffload`): that one runs after `LoadHfConfig`, which
  // this one-key `config.json` does not get past.
  const std::string kOffloaderInstalled =
      "engine: offload_config installed UvaWeightOffloader";
  const std::string dir = MakeArchDir(kPoolingArch);
  const ChildRun run =
      RunServer("--model " + dir +
                R"( --offload-config {"offload_backend":"uva","uva":{"cpu_offload_gb":10}})");
  INFO("child output:\n" << run.output);
  CHECK(Contains(run.output, kPoolingBanner));
  CHECK(Contains(run.output, kOffloaderInstalled));
  // No `vllm_cpp` key, so no residency install: the two halves stay separate.
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK(run.status == 0);

  // The negative control for the line above: without the flag the same branch on
  // the same directory does not print it, so the assertion is the document being
  // seen rather than a line that always prints.
  const ChildRun bare = RunServer("--model " + dir);
  INFO("child output:\n" << bare.output);
  CHECK(Contains(bare.output, kPoolingBanner));
  CHECK_FALSE(Contains(bare.output, kOffloaderInstalled));

  // A typo in the mirrored half aborts at startup on this branch too, because
  // the parse is now ahead of the branch rather than inside one of them. It
  // aborts BEFORE the branch is even chosen, which is why the pooling banner
  // does not print.
  const ChildRun typo =
      RunServer("--model " + dir +
                R"( --offload-config {"uva":{"cpu_offload_GB":10}})");
  INFO("child output:\n" << typo.output);
  CHECK(Contains(typo.output, "unknown key \"uva.cpu_offload_GB\""));
  CHECK_FALSE(Contains(typo.output, kPoolingBanner));
}

TEST_CASE("serve: the POOLING path is unchanged without the flag") {
  // The inertness control for the two cases above: the branch is chosen the same
  // way and prints nothing new when no document is passed.
  const std::string dir = MakeArchDir(kPoolingArch);
  const ChildRun run = RunServer("--model " + dir);
  INFO("child output:\n" << run.output);
  CHECK(Contains(run.output, kPoolingBanner));
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK(run.status == 0);
}

TEST_CASE("serve: the TRANSCRIPTION-only path REFUSES the document, naming what is missing") {
  // The decided arm of #1135. `ParakeetTranscriber::FromDir` builds no
  // `EngineParams` and calls no `LoadedEngine`, so no field of either half has a
  // reader on this path. AGENTS.md: refuse an unimplemented arm with a message
  // naming the missing part. Accepting and warning would leave a server running
  // while it holds a placement instruction it does not follow, which is the
  // failure this issue was filed about.
  const std::string dir = MakeArchDir(kTranscriptionArch);
  const ChildRun run =
      RunServer("--model " + dir +
                R"( --offload-config {"vllm_cpp":{"mmap":{"enabled":true}}})");
  INFO("child output:\n" << run.output);

  CHECK(Contains(run.output, "server: fatal:"));
  CHECK(Contains(run.output, "--offload-config is not supported on a "
                             "transcription-only model"));
  CHECK(Contains(run.output, "THE MISSING PART"));
  CHECK(Contains(run.output, "ParakeetTranscriber"));
  // It names the issue that owns the wiring, so a reader of the message can find
  // the record rather than concluding the capability was forgotten.
  CHECK(Contains(run.output, "#1195"));
  // Nothing was installed, which is the whole point: the alternative shape
  // accepts the document and ignores it.
  CHECK_FALSE(Contains(run.output, kInstallLine));
  CHECK(run.status == 0);

  // The MIRRORED half is refused on the same terms. It was dropped here for
  // longer than the extension has existed, so a fix that covered only the
  // extension would leave the older half of the same bug in place.
  const ChildRun mirrored =
      RunServer("--model " + dir +
                R"( --offload-config {"uva":{"cpu_offload_gb":10}})");
  INFO("child output:\n" << mirrored.output);
  CHECK(Contains(mirrored.output,
                 "--offload-config is not supported on a transcription-only "
                 "model"));
}

TEST_CASE("serve: the TRANSCRIPTION-only path is unchanged without the flag") {
  // The control that keeps the refusal above a REFUSAL OF THE FLAG rather than a
  // refusal of the path. Without a document the branch is entered exactly as
  // before and fails later, on the checkpoint this directory does not have.
  const std::string dir = MakeArchDir(kTranscriptionArch);
  const ChildRun run = RunServer("--model " + dir);
  INFO("child output:\n" << run.output);
  CHECK(Contains(run.output, kTranscriptionBanner));
  CHECK_FALSE(Contains(run.output, "--offload-config is not supported"));
  CHECK(run.status == 0);
}

TEST_CASE("serve: a placement document reaches the loader through the real argv") {
  // ENG-HYBRID-PLACEMENT W1 (#2018). This is the reachability case for the whole
  // placement family: the REAL `VllmServerMain` on a REAL argv, not a parser
  // called by hand. A resolver test that builds a `PlacementConfig` in-process
  // proves the class works, never that `--offload-config` can carry one.
  const ChildRun run =
      RunServer(std::string(kMissingModel) +
                R"( --offload-config {"vllm_cpp":{"placement":{"n_cpu_moe":40}}})");
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  CHECK(Contains(run.output, kInstallLine));

  // Both halves of the install line: the sugar the operator typed, and the count
  // it desugared into. A `-ncmoe 40` that expanded to 39 is otherwise invisible.
  CHECK(Contains(run.output, "n_cpu_moe=40"));
  CHECK(Contains(run.output, "placement_overrides=40"));

  // The mmap collision warning is a PRODUCTION line, printed by the loader on the
  // same path. This is the assertion the reachability mutation deletes a call site
  // under: remove `DescribePlacementResidencyCollision()` from model_loader.cpp
  // and this goes red while every in-process resolver test stays green.
  CHECK(Contains(run.output, "mmap-resident"));

  CHECK(Contains(run.output, "SERVE_RC="));
  CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));
  CHECK(run.status == 0);
}

TEST_CASE("serve: fit beside a manual placement ABORTS at startup, naming both") {
  // `common/fit.cpp:398-399` refuses the same combination. The refusal has to
  // reach the operator at startup, which means it has to survive the real argv
  // path rather than only the parser's unit test.
  const ChildRun run = RunServer(
      std::string(kMissingModel) +
      R"( --offload-config {"vllm_cpp":{"placement":{"fit":true,"cpu_moe":true}}})");
  INFO("child output:\n" << run.output);

  CHECK(Contains(run.output, "vllm_cpp.placement.fit"));
  CHECK(Contains(run.output, "cannot be combined"));
  // It aborted BEFORE the load, so the loading banner never printed.
  CHECK_FALSE(Contains(run.output, "server: loading model from"));
  CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));
}

TEST_CASE("serve: the resolved DEVICE PLACEMENT is reported on the real argv") {
  // ENG-HYBRID-PLACEMENT W2 (#2023). The reachability case for the seam: the REAL
  // `VllmServerMain`, not a `DevicePlacement` built by hand. A unit test that
  // constructs the class proves the class works, never that a load reaches one.
  //
  // `--device cpu` with `cpu_moe` is the INERT shape — the overrides resolve to
  // the device the engine is already on — and the engine must SAY that rather
  // than print a placement it is not performing or say nothing at all. An
  // operator who pasted a llama.cpp command line at a CPU build lands here.
  const ChildRun run = RunServer(
      std::string(kMissingModel) +
      R"( --device cpu --offload-config {"vllm_cpp":{"placement":{"cpu_moe":true}}})");
  INFO("child output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, "engine: device placement:"));
  CHECK(Contains(run.output, "nothing is placed"));

  CHECK(Contains(run.output, "SERVE_RC="));
  CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));
  CHECK(run.status == 0);
}

TEST_CASE("serve: NO placement prints NO placement line, so an ordinary load is unchanged") {
  // The inertness half, and it is an assertion rather than an inspection: a load
  // that configures no placement must be byte-identical on stderr too. A seam
  // that announced itself on every load would be a behaviour change shipped as a
  // diagnostic.
  const ChildRun run = RunServer(
      std::string(kMissingModel) +
      R"( --offload-config {"vllm_cpp":{"mmap":{"enabled":true}}})");
  INFO("child output:\n" << run.output);
  CHECK_FALSE(Contains(run.output, "engine: device placement:"));
  // The residency half still reported, so the absence above is the placement
  // staying quiet and not the whole install block failing to run.
  CHECK(Contains(run.output, kInstallLine));
}
