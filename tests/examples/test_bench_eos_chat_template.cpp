// #2759 — can `vllm-bench` reproduce a chat-completions benchmark protocol?
//
// Two settings decide that, and until this change neither existed. The harness
// set `sp.ignore_eos = true` with no field and no flag, so no request it has
// ever admitted could terminate on EOS, and it sent the raw prompt string, so no
// chat template was ever applied. The comparator for `BENCH-QWEN38-27B-SOTA`
// posts to `/v1/chat/completions` with a template and never sets `ignore_eos`.
//
// WHY THIS RUNS THE BINARY. `tests/examples/test_bench.cpp` already drives
// `RunBench` from a hand-built `BenchConfig`, and it stayed green on every day
// these flags did not exist, because it never enters through the harness's own
// command line. For a benchmark harness that entry point is `argv`
// (.agents/reachability.md), so this file execs the built `vllm-bench` and reads
// its report. Same shape as `test_bench_kv_cache_dtype.cpp`, for the same
// reason.
//
// WHY THE SYNTHETIC ENGINE IS ENOUGH, AND WHERE IT IS NOT. With no `--model`,
// `RunBench` builds the tiny CPU Qwen3.5-MoE. It tokenizes real prompts through
// the production tokenizer, so a chat template applied to a prompt changes the
// token count the engine actually receives, and G3 reads that. It declares NO
// eos id (`bench_core.h`: `c.raw = nlohmann::json::object();  // no eos =>
// runs to max_tokens`), so `check_stop`'s EOS arm can never fire here whatever
// `ignore_eos` says. G5 asserts that absence rather than hiding it: the flag is
// gated on the SamplingParams the admission path builds, and the end-to-end stop
// is owed to the first checkpoint run
// (.agents/specs/bench-eos-and-chat-template.md §7).
//
// THE MUTATIONS THIS FILE EXISTS FOR:
//   - `sp.ignore_eos = cfg.ignore_eos` back to `sp.ignore_eos = true` in
//     MakeSampling => G1 goes red. The flag parses and is then dropped, the run
//     still succeeds, and the report still prints a resolved line -- saying 1.
//     That silent drop is the precise defect #2759 names.
//   - delete the `--no-ignore-eos` / `--no-skip-chat-template` /
//     `--chat-template` arms from ParseArgs => the owning case goes red on
//     "unknown argument" and exit 2.
//   - delete the render call site in RunBench => G3 goes red: the template is
//     resolved, the report says "applied", and the prompt the engine tokenizes
//     is unchanged.
//   - drop either refusal => G6 or G7 goes red on exit 0.
//
// AND ONE MUTATION NOTHING HERE CATCHES, said plainly because a reader will
// otherwise assume it is covered: replacing the resolved EOS line's read-back
// with `cfg.ignore_eos` leaves every case green, because the echoed value is
// still the correct one. That mutation is harmless BY ITSELF and lethal in
// combination -- it is what would disarm the first mutation above, and the only
// reason deleting the pass-through is detectable at all is that this line does
// not echo the request.
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
               "*** test_bench_eos_chat_template: built without "
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

// The smallest run that still tokenizes a prompt and decodes through it. One
// request, greedy, four output tokens: this file asserts on the REPORT and on
// the token COUNTS, never on a rate.
constexpr const char* kTinyWorkload =
    "--num-prompts 1 --input-len 8 --output-len 4 --concurrency 1";

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

// Report labels, asserted separately from their values so that renaming a label
// reddens every case at once instead of quietly making the value assertions
// vacuous against an absent line.
constexpr const char* kEosRequestedLabel = "Ignore EOS (requested):";
constexpr const char* kEosResolvedLabel = "Ignore EOS (resolved sampling):";
constexpr const char* kTemplateLabel = "Chat template:";
constexpr const char* kTemplateKwargsLabel = "Chat template kwargs:";
constexpr const char* kInputTokensLabel = "Total input tokens:";
constexpr const char* kOutputTokensLabel = "Total generated tokens:";

// What the report prints when no template was applied. Exact, because "the
// default did not change" is the claim G4 makes and a prefix match would pass
// for a value that changed after the prefix.
constexpr const char* kTemplateSkipped = "skipped (raw prompt)";

// The value printed after `label` on its own report line, whitespace-trimmed, or
// "" when the label is absent. Trimmed rather than matched against a literal
// with the padding baked in, because the report's column width is a formatting
// choice and a gate that fails on it fails for a reason that is not a defect.
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

// A chat template written entirely in the synthetic tokenizer's nineteen-piece
// vocabulary, so the rendered prompt is encodable by the fixture. A real Qwen
// template renders `<|im_start|>`, which that vocabulary does not carry. This is
// a fixture constraint and not a behaviour difference: the render seam is
// `MakeChatTemplatePromptFn`, the production one, either way.
constexpr const char* kInVocabTemplate =
    "hello{% for m in messages %}{{ m.content }}{% endfor %}world";

// A throwaway file holding that template, for the `--chat-template <path>` arm.
class TemplateFile {
 public:
  explicit TemplateFile(const std::string& body) {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllm-cpp-bench-chat-template-" + std::to_string(::getpid()) + "-" +
             std::to_string(counter++) + ".jinja");
    std::ofstream out(path_);
    REQUIRE(out.good());
    out << body;
  }
  ~TemplateFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  TemplateFile(const TemplateFile&) = delete;
  TemplateFile& operator=(const TemplateFile&) = delete;

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

}  // namespace

// ─── G1. `--no-ignore-eos` reaches the SamplingParams the engine is handed ───
TEST_CASE("vllm-bench: --no-ignore-eos reaches SamplingParams") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("--no-ignore-eos");
  INFO("vllm-bench output:\n" << run.output);

  CHECK_FALSE(Contains(run.output, "unknown argument"));
  CHECK(run.status == 0);
  REQUIRE(Contains(run.output, kEosRequestedLabel));
  REQUIRE(Contains(run.output, kEosResolvedLabel));
  CHECK(LineValue(run.output, kEosRequestedLabel) == "0");
  // THE assertion: read back out of the SamplingParams object `MakeSampling`
  // builds for an admitted request, so this is the value the scheduler's EOS
  // gate will see and not an echo of the flag.
  CHECK(LineValue(run.output, kEosResolvedLabel) == "0");
}

// ─── G2. The resolved line is not a transcription of the requested one ───────
//
// G1 alone would pass against a report that printed `cfg.ignore_eos` twice. The
// pair below is what makes G1 mean anything: the same machinery has to produce
// two different resolved values on two runs, and the value has to come from
// MakeSampling for the "0" case to be reachable at all.
TEST_CASE("vllm-bench: the resolved EOS line tracks MakeSampling, both ways") {
  RequireBenchBinary();
  const BenchRun on = RunBenchBinary("--ignore-eos");
  const BenchRun off = RunBenchBinary("--no-ignore-eos");
  INFO("--ignore-eos output:\n" << on.output);
  INFO("--no-ignore-eos output:\n" << off.output);

  REQUIRE(on.status == 0);
  REQUIRE(off.status == 0);
  CHECK(LineValue(on.output, kEosResolvedLabel) == "1");
  CHECK(LineValue(off.output, kEosResolvedLabel) == "0");
  CHECK(LineValue(on.output, kEosResolvedLabel) !=
        LineValue(off.output, kEosResolvedLabel));
}

// ─── G3. The chat template reaches the tokens the engine receives ────────────
TEST_CASE("vllm-bench: --no-skip-chat-template renders into the admitted prompt") {
  RequireBenchBinary();
  const TemplateFile tmpl(kInVocabTemplate);
  const BenchRun plain = RunBenchBinary("");
  const BenchRun rendered =
      RunBenchBinary("--no-skip-chat-template --chat-template " + tmpl.path());
  INFO("no template:\n" << plain.output);
  INFO("with template:\n" << rendered.output);

  CHECK_FALSE(Contains(rendered.output, "unknown argument"));
  REQUIRE(plain.status == 0);
  REQUIRE(rendered.status == 0);

  // The report says a template was applied, and names where it came from.
  CHECK(LineValue(plain.output, kTemplateLabel) == kTemplateSkipped);
  CHECK(Contains(LineValue(rendered.output, kTemplateLabel), "applied"));
  CHECK(Contains(LineValue(rendered.output, kTemplateLabel), tmpl.path()));

  // THE assertion, and the one a report line cannot fake: the workload is
  // identical (same seed, same --input-len, same prompt builder), so the ONLY
  // way the engine can see more prompt tokens is that the render reached the
  // admission path. `hello` and `world` are both in the fixture vocabulary.
  const std::string plain_in = LineValue(plain.output, kInputTokensLabel);
  const std::string rendered_in = LineValue(rendered.output, kInputTokensLabel);
  REQUIRE_FALSE(plain_in.empty());
  REQUIRE_FALSE(rendered_in.empty());
  CHECK(std::stoll(rendered_in) > std::stoll(plain_in));
}

// ─── G4. The no-flag run is byte-identical in meaning to the old harness ─────
//
// The entire value of this change is that a `vllm-bench` invocation carrying no
// new flag is the `vllm-bench` every landed figure was measured on. If this case
// ever goes red, a published number changed meaning without being re-measured.
TEST_CASE("vllm-bench: the default still suppresses EOS and applies no template") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("");
  INFO("vllm-bench output:\n" << run.output);

  REQUIRE(run.status == 0);
  CHECK(LineValue(run.output, kEosRequestedLabel) == "1");
  CHECK(LineValue(run.output, kEosResolvedLabel) == "1");
  CHECK(LineValue(run.output, kTemplateLabel) == kTemplateSkipped);
  CHECK(LineValue(run.output, kTemplateKwargsLabel) == "{}");
  // The fixed-length contract the old harness documented: greedy, so exactly
  // --output-len tokens come out.
  CHECK(LineValue(run.output, kOutputTokensLabel) == "4");
}

// ─── G5. What this fixture CANNOT see, asserted rather than assumed ──────────
//
// The synthetic config declares no eos id, so `--no-ignore-eos` cannot shorten a
// generation here even when it arrives. Writing that down as a CHECK keeps the
// spec's §7 honest: a future reader who expects this file to prove an
// EOS-terminated run is told, by a green test, that it does not.
TEST_CASE("vllm-bench: the synthetic fixture has no EOS to stop on") {
  RequireBenchBinary();
  const BenchRun off = RunBenchBinary("--no-ignore-eos");
  INFO("--no-ignore-eos output:\n" << off.output);

  REQUIRE(off.status == 0);
  CHECK(LineValue(off.output, kEosResolvedLabel) == "0");
  // Honouring EOS changes nothing here, because there is no eos_token_id to
  // emit. The end-to-end stop is owed to a checkpoint run.
  CHECK(LineValue(off.output, kOutputTokensLabel) == "4");
}

// ─── G6. A flag that would be silently ignored is refused ────────────────────
TEST_CASE("vllm-bench: --chat-template without --no-skip-chat-template is refused") {
  RequireBenchBinary();
  const TemplateFile tmpl(kInVocabTemplate);
  const BenchRun run = RunBenchBinary("--chat-template " + tmpl.path());
  INFO("vllm-bench output:\n" << run.output);

  CHECK(run.status != 0);
  CHECK(Contains(run.output, "--no-skip-chat-template"));
}

// ─── G7. A template that cannot be found is refused, naming what is missing ──
TEST_CASE("vllm-bench: --no-skip-chat-template with no template source is refused") {
  RequireBenchBinary();
  const BenchRun run = RunBenchBinary("--no-skip-chat-template");
  INFO("vllm-bench output:\n" << run.output);

  CHECK(run.status != 0);
  CHECK(Contains(run.output, "chat template"));
  CHECK(Contains(run.output, "--chat-template"));
}

// ─── G8. A single-line literal and a file resolve to the same prompt ─────────
//
// `--chat-template` takes "the file path to the chat template, or the template
// in single-line form" (vllm/entrypoints/launchers/cli_args.py:80-82 @
// e126687a9). Two spellings of one template must reach the engine as one
// prompt, or the literal arm is a second, untested renderer.
TEST_CASE("vllm-bench: --chat-template takes a path or the template itself") {
  RequireBenchBinary();
  const TemplateFile tmpl(kInVocabTemplate);
  const BenchRun from_file =
      RunBenchBinary("--no-skip-chat-template --chat-template " + tmpl.path());
  const BenchRun from_literal =
      RunBenchBinary(std::string("--no-skip-chat-template --chat-template '") +
                     kInVocabTemplate + "'");
  INFO("from file:\n" << from_file.output);
  INFO("from literal:\n" << from_literal.output);

  REQUIRE(from_file.status == 0);
  REQUIRE(from_literal.status == 0);
  CHECK(Contains(LineValue(from_literal.output, kTemplateLabel), "applied"));
  const std::string file_in = LineValue(from_file.output, kInputTokensLabel);
  const std::string literal_in = LineValue(from_literal.output, kInputTokensLabel);
  REQUIRE_FALSE(file_in.empty());
  CHECK(file_in == literal_in);
}

// ─── G9. --no-enable-thinking reaches the renderer's default kwargs ──────────
//
// The comparator sends `chat_template_kwargs {enable_thinking: false}`, so a
// harness that cannot set it renders a different prompt than the run it is
// compared against. The spelling and the rule are `server_main.cpp`'s, through
// `DefaultChatTemplateKwargs`: neither flag leaves the object EMPTY and
// `enable_thinking` Jinja-undefined, which is not the same as false (#1681).
TEST_CASE("vllm-bench: --no-enable-thinking reaches the chat-template kwargs") {
  RequireBenchBinary();
  const TemplateFile tmpl(kInVocabTemplate);
  const BenchRun unset =
      RunBenchBinary("--no-skip-chat-template --chat-template " + tmpl.path());
  const BenchRun off = RunBenchBinary(
      "--no-skip-chat-template --no-enable-thinking --chat-template " +
      tmpl.path());
  INFO("unset:\n" << unset.output);
  INFO("--no-enable-thinking:\n" << off.output);

  REQUIRE(unset.status == 0);
  REQUIRE(off.status == 0);
  CHECK(LineValue(unset.output, kTemplateKwargsLabel) == "{}");
  CHECK(LineValue(off.output, kTemplateKwargsLabel) ==
        "{\"enable_thinking\":false}");
}
