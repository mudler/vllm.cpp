// ENG-MM-INPUT-PIPELINE wave L2 (#607) — the SERVE FLAGS half.
//
// Ported from / grounded in:
//   - vllm/engine/arg_utils.py:555 (`language_model_only: bool =
//     MultiModalConfig.language_model_only`), :1276 (the argparse registration,
//     which _compute_kwargs gives BooleanOptionalAction, :346-348), :1691 (it
//     reaching the config);
//   - vllm/engine/arg_utils.py:556 (`limit_mm_per_prompt: dict[...] =
//     get_field(MultiModalConfig, "limit_per_prompt")`), :1279, :1692, whose
//     dict type resolves to `type=parse_type(json.loads)` (:379-381, the
//     plain-dict branch — the `union_dict_and_str` branch at :374-378 is a
//     different rule and needs a `str` arm the annotation does not have) — so
//     the value is a JSON OBJECT;
//   - vllm/config/multimodal.py:17-45 (the DummyOptions dataclasses: `count:
//     int = Field(999, ge=0)` on BaseDummyOptions :21; `extra="forbid"` on the
//     video/image/audio subclasses ONLY, :24,33,41; every declared option
//     `Field(None, gt=0)`, :28-30,37-38,45) and :212-236
//     (`_validate_limit_per_prompt`, which rewrites a bare int to
//     {"count": <int>} and routes each modality to its own dataclass, falling to
//     the bare BaseDummyOptions at :233 for anything outside those three);
//   - vllm/config/multimodal.py:321-336 (`get_limit_per_prompt`, the precedence
//     the startup banner prints THROUGH so the flag ordering is observable).
// All at 5559679229bc, the pinned oracle in .agents/upstream-sync.md.
//
// THE GATE: 43 of the 157 official vllm-project/recipes commands pass
// `--language-model-only` and this server used to stop at the unknown-argument
// guard before reading a weight. Both flags must now parse, must REACH THE
// CONFIG (not merely be swallowed — that is the failure mode #606's own spec
// names), and a malformed limit must ABORT rather than silently default to 999,
// because a limit that quietly became 999 is a limit that is not there.
//
// WHY A SUBPROCESS for the flag half. `ParseArgs` lives in an anonymous
// namespace and reports a bad argument through `Usage()`, which calls
// `std::exit`; an in-process call would take the whole test binary with it. Each
// case therefore RE-EXECS THIS TEST BINARY into a skip-decorated child that
// calls the REAL `VllmServerMain`, exactly as
// tests/vllm/entrypoints/openai/test_serve_recipe_args.cpp does (#606), and
// asserts on the child's combined output and exit status. The model directory is
// deliberately nonexistent, so "parsing succeeded" is observable without a
// checkpoint or a bound port.
//
// The PARSER half needs no subprocess: ParseLimitMmPerPromptJson is a library
// function, so its refusals are asserted directly.
#include <doctest/doctest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/config/multimodal.h"
#include "vllm/entrypoints/openai/server_main.h"

namespace {

// A path that cannot exist, so the child always fails at load rather than
// serving forever.
constexpr const char* kMissingModel =
    "--model /nonexistent/vllm-cpp/serve-mm-limits";

// Printed by VllmServerMain AFTER ParseArgs returns. Its presence proves that
// argument parsing succeeded and control reached engine construction.
constexpr const char* kPostParseBanner = "server: request logging";
constexpr const char* kLoadBanner = "server: loading model from";
constexpr const char* kUnknownArgument = "server: unknown argument";
// The RESOLVED-limits banner. It prints through GetLimitPerPrompt, so it is a
// statement about the CONFIG, not about the argv.
constexpr const char* kLimitsBanner = "server: multimodal limits";

struct ChildRun {
  std::string output;
  int status = -1;
};

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

ChildRun RunServer(const std::string& serve_args) {
  // Resolve our own path HERE, in the parent: popen runs the command under
  // /bin/sh, so a literal /proc/self/exe inside it would resolve to the shell.
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';
  // Single quotes around the args would break the JSON values, which carry
  // double quotes; the child reads them from the environment instead, and the
  // shell only ever sees the assignment. Escape any single quote for the shell.
  std::string quoted;
  for (const char c : serve_args) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  const std::string cmd = "VLLM_TEST_MM_SERVE_ARGS='" + quoted + "' " +
                          std::string(exe) +
                          " --no-skip --test-case='serve_mm_limits_child'"
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

// Split on spaces EXCEPT inside single quotes, so a JSON value survives as one
// argv entry the way a shell would hand it over.
std::vector<std::string> SplitArgs(const std::string& text) {
  std::vector<std::string> out;
  std::string current;
  bool in_quote = false;
  bool started = false;
  for (const char c : text) {
    if (c == '\'') {
      in_quote = !in_quote;
      started = true;
      continue;
    }
    if (c == ' ' && !in_quote) {
      if (started || !current.empty()) out.push_back(current);
      current.clear();
      started = false;
      continue;
    }
    current.push_back(c);
  }
  if (started || !current.empty()) out.push_back(current);
  return out;
}

}  // namespace

// The CHILD case, filtered out of a normal run.
TEST_CASE("serve_mm_limits_child" * doctest::skip()) {
  const char* raw = std::getenv("VLLM_TEST_MM_SERVE_ARGS");
  REQUIRE(raw != nullptr);
  std::vector<std::string> args{"vllm-server"};
  for (std::string& token : SplitArgs(raw)) args.push_back(std::move(token));
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) argv.push_back(arg.data());
  const int rc = vllm::entrypoints::openai::VllmServerMain(
      static_cast<int>(argv.size()), argv.data());
  std::cout << "SERVE_RC=" << rc << "\n" << std::flush;
  std::exit(0);
}

// ── The PARSER (vllm/config/multimodal.py:212-236 + the dataclasses :17-43) ──

TEST_CASE("limit-mm-per-prompt: the LEGACY count-only object (multimodal.py:87-88)") {
  std::vector<std::string> ignored;
  // Upstream's own example (:88).
  const std::map<std::string, int> limits =
      vllm::ParseLimitMmPerPromptJson(R"({"image": 16, "video": 2})", &ignored);
  CHECK(limits.at("image") == 16);
  CHECK(limits.at("video") == 2);
  CHECK(ignored.empty());

  // 0 is a LEGAL limit, and it is the one that matters: it is what
  // --language-model-only is defined as being equivalent to (multimodal.py:79).
  const std::map<std::string, int> zeroed =
      vllm::ParseLimitMmPerPromptJson(R"({"image": 0})", nullptr);
  CHECK(zeroed.at("image") == 0);

  vllm::MultiModalConfig cfg;
  cfg.limit_per_prompt = zeroed;
  CHECK(cfg.GetLimitPerPrompt("image") == 0);
  // A modality the map does not mention keeps the 999 default (:331-333) — an
  // explicit zero for ONE modality is not a global off switch.
  CHECK(cfg.GetLimitPerPrompt("video") == vllm::kDefaultLimitPerPrompt);
}

TEST_CASE("limit-mm-per-prompt: the CONFIGURABLE + MIXED objects (multimodal.py:90-96)") {
  std::vector<std::string> ignored;
  // Upstream's own configurable example (:91-92).
  const std::map<std::string, int> limits = vllm::ParseLimitMmPerPromptJson(
      R"({"video": {"count": 1, "num_frames": 32, "width": 512, "height": 512},
          "image": {"count": 5, "width": 512, "height": 512}})",
      &ignored);
  CHECK(limits.at("video") == 1);
  CHECK(limits.at("image") == 5);
  // Only `count` participates in get_limit_per_prompt (:335). The options are
  // validated and dropped, and the drop is REPORTED so the server can announce
  // it rather than let a user infer that num_frames took effect.
  CHECK(ignored.size() == 5);

  // The MIXED spelling (:94-96): one modality as a count, one as an object.
  const std::map<std::string, int> mixed = vllm::ParseLimitMmPerPromptJson(
      R"({"image": 16, "video": {"count": 1, "num_frames": 32}})", nullptr);
  CHECK(mixed.at("image") == 16);
  CHECK(mixed.at("video") == 1);

  // An object with no `count` keeps the dataclass default (:21), not 0 — the
  // object exists to carry OPTIONS, so omitting the count is not a refusal.
  const std::map<std::string, int> options_only =
      vllm::ParseLimitMmPerPromptJson(R"({"image": {"width": 512}})", nullptr);
  CHECK(options_only.at("image") == vllm::kDefaultLimitPerPrompt);
}

TEST_CASE("limit-mm-per-prompt: malformed input is REFUSED, never defaulted") {
  // Each of these is a distinct upstream validation, and every one of them must
  // throw rather than yield a map that silently reads 999.
  const std::vector<std::string> refused = {
      "not json at all",                      // json.loads raises
      "[1, 2, 3]",                            // not an object
      "\"image\"",                            // not an object
      R"({"image": "two"})",                  // count is not an int
      R"({"image": 2.5})",                    // count is not an int
      R"({"image": -1})",                     // count: Field(999, ge=0), :21
      R"({"image": {"count": -1}})",          // same, through the object form
      R"({"image": {"count": "two"}})",       // same, wrong type
      R"({"image": {"num_frames": 32}})",     // image has no num_frames, :33-38
      R"({"video": {"fps": 2}})",             // extra="forbid", :24
      R"({"audio": {"width": 2}})",           // audio takes `length`, :41-45
      R"({"video": {"num_frames": 0}})",      // Field(None, gt=0), :28
      R"({"video": {"num_frames": -4}})",     // same
      R"({"image": {"width": "big"}})",       // option is not an int
  };
  for (const std::string& doc : refused) {
    CAPTURE(doc);
    CHECK_THROWS_AS(vllm::ParseLimitMmPerPromptJson(doc, nullptr),
                    std::invalid_argument);
  }

  // ...and the message NAMES what was wrong, because a dropped `num_frame` typo
  // is invisible otherwise.
  try {
    vllm::ParseLimitMmPerPromptJson(R"({"video": {"num_frame": 32}})", nullptr);
    FAIL("expected a refusal");
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    CHECK(Contains(msg, "num_frame"));
    CHECK(Contains(msg, "video"));
  }

  // An accepted modality name is NOT enumerated: upstream routes an unknown
  // modality to BaseDummyOptions (:233) rather than rejecting it, because
  // the modality set is the model's, not the config's.
  CHECK(vllm::ParseLimitMmPerPromptJson(R"({"tactile": 3})", nullptr)
            .at("tactile") == 3);
}

TEST_CASE("limit-mm-per-prompt: forbidding extras is the BUILTIN modalities' "
          "rule, not a global one") {
  // The divergence the #749 review caught. `extra="forbid"` is declared on
  // VideoDummyOptions / ImageDummyOptions / AudioDummyOptions ONLY
  // (multimodal.py:24,33,41). BaseDummyOptions — what :233's `else` builds for
  // every OTHER modality name — is a plain `@dataclass` (:17-21), so pydantic's
  // default `extra='ignore'` applies and an unlisted key is DROPPED with its
  // value never validated.
  //
  // Re-derived at the pinned oracle 5559679229bc under pydantic 2.12.5, by
  // declaring those two dataclasses verbatim and calling them:
  //     BaseDummyOptions(**{"count": 2, "foo": 3})  -> BaseDummyOptions(count=2)
  //     ImageDummyOptions(**{"count": 2, "foo": 3}) -> ValidationError
  // Refusing the first would refuse a document the reference accepts, which is
  // what this port did before the repair.
  std::vector<std::string> ignored;
  const std::map<std::string, int> limits = vllm::ParseLimitMmPerPromptJson(
      R"({"pointcloud": {"count": 2, "foo": 3}})", &ignored);
  CHECK(limits.at("pointcloud") == 2);
  // Dropped, but ANNOUNCED — upstream discards it silently; we say so, exactly
  // as we do for the builtin options we also drop.
  REQUIRE(ignored.size() == 1);
  CHECK(ignored.at(0) == "pointcloud.foo");

  // The extra's VALUE is not validated either: `Field(None, gt=0)` guards
  // declared fields, and `foo` is not one. `0` and a string both survive on a
  // non-builtin modality...
  CHECK(vllm::ParseLimitMmPerPromptJson(R"({"tactile": {"count": 1, "foo": 0}})",
                                        nullptr)
            .at("tactile") == 1);
  CHECK(vllm::ParseLimitMmPerPromptJson(
            R"({"tactile": {"count": 1, "foo": "big"}})", nullptr)
            .at("tactile") == 1);
  // ...while the same shapes on a BUILTIN modality are still refused, because
  // there the key either is declared (and then gt=0 applies) or extra="forbid"
  // rejects it. This half is what keeps the repair from becoming "accept
  // everything".
  const std::vector<std::string> still_refused = {
      R"({"image": {"count": 1, "foo": 3}})",
      R"({"image": {"count": 1, "width": 0}})",
      R"({"video": {"count": 1, "num_frames": "big"}})"};
  for (const std::string& doc : still_refused) {
    CAPTURE(doc);
    CHECK_THROWS_AS(vllm::ParseLimitMmPerPromptJson(doc, nullptr),
                    std::invalid_argument);
  }

  // `count` is declared on BaseDummyOptions itself (:21), so ITS validation is
  // global — a non-builtin modality does not escape `Field(999, ge=0)`.
  CHECK_THROWS_AS(
      vllm::ParseLimitMmPerPromptJson(R"({"tactile": {"count": -1}})", nullptr),
      std::invalid_argument);
}

// ── The FLAGS (arg_utils.py:1276,1279) reaching the CONFIG ──────────────────

TEST_CASE("serve flags: --language-model-only parses and ZEROES every limit") {
  const ChildRun run =
      RunServer(std::string(kMissingModel) + " --language-model-only");
  INFO("child output:\n" << run.output);
  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kPostParseBanner));
  // It reached MODEL LOAD, which failed on the deliberately missing checkpoint
  // — "parsing succeeded, the engine tried to load".
  CHECK(Contains(run.output, kLoadBanner));
  CHECK_FALSE(Contains(run.output, "SERVE_RC=0"));
  CHECK(run.status == 0);
  // THE LOAD-BEARING ASSERTION: the flag reached the CONFIG. The banner prints
  // through GetLimitPerPrompt, so this is the resolved limit, not the argv.
  CHECK(Contains(run.output, "server: multimodal limits language-model-only=ON"));
  CHECK(Contains(run.output, "image=0"));
  CHECK(Contains(run.output, "video=0"));
  CHECK(Contains(run.output, "audio=0"));
}

TEST_CASE("serve flags: --no-language-model-only is accepted (BooleanOptionalAction)") {
  // arg_utils.py:1276 registers the field with argparse.BooleanOptionalAction
  // (arg_utils.py:346-348), which defines BOTH spellings. A recipe that turns
  // the flag off explicitly must not die on an unknown argument.
  const ChildRun run =
      RunServer(std::string(kMissingModel) + " --no-language-model-only");
  INFO("child output:\n" << run.output);
  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, "server: multimodal limits language-model-only=OFF"));
  CHECK(Contains(run.output, kLoadBanner));
}

TEST_CASE("serve flags: --limit-mm-per-prompt reaches the config") {
  const ChildRun run = RunServer(std::string(kMissingModel) +
                                 " --limit-mm-per-prompt '{\"image\": 2, "
                                 "\"video\": 0}'");
  INFO("child output:\n" << run.output);
  CHECK_FALSE(Contains(run.output, kUnknownArgument));
  CHECK(Contains(run.output, kLimitsBanner));
  CHECK(Contains(run.output, "image=2"));
  CHECK(Contains(run.output, "video=0"));
  CHECK(Contains(run.output, kLoadBanner));
  CHECK(run.status == 0);
}

TEST_CASE("serve flags: --language-model-only WINS over an explicit limit") {
  // get_limit_per_prompt checks the flag BEFORE reading the map
  // (multimodal.py:326-327). Reading the map first would be indistinguishable
  // on every configuration except this one, which is why this case exists.
  const ChildRun run =
      RunServer(std::string(kMissingModel) +
                " --limit-mm-per-prompt '{\"image\": 4}' --language-model-only");
  INFO("child output:\n" << run.output);
  CHECK(Contains(run.output, "language-model-only=ON"));
  CHECK(Contains(run.output, "image=0"));
  CHECK_FALSE(Contains(run.output, "image=4"));
}

TEST_CASE("serve flags: a malformed --limit-mm-per-prompt ABORTS before model load") {
  // THE OTHER LOAD-BEARING ONE. Silently defaulting a mistyped limit to 999 is
  // strictly worse than aborting: the server then runs with no limit at all,
  // and the user believes they set one.
  for (const char* value :
       {"'not json'", "'[1,2]'", "'{\"image\": -1}'",
        "'{\"video\": {\"fps\": 2}}'"}) {
    CAPTURE(value);
    const ChildRun run = RunServer(std::string(kMissingModel) +
                                   " --limit-mm-per-prompt " + value);
    INFO("child output:\n" << run.output);
    CHECK(Contains(run.output, "server: --limit-mm-per-prompt"));
    // Aborted inside ParseArgs: the model load never started.
    CHECK_FALSE(Contains(run.output, kLoadBanner));
    CHECK_FALSE(Contains(run.output, "SERVE_RC="));
  }
}

TEST_CASE("serve flags: the dropped profiling options are ANNOUNCED, not silent") {
  const ChildRun run =
      RunServer(std::string(kMissingModel) +
                " --limit-mm-per-prompt '{\"video\": {\"count\": 1, "
                "\"num_frames\": 32}}'");
  INFO("child output:\n" << run.output);
  CHECK(Contains(run.output, "video=1"));
  CHECK(Contains(run.output, "video.num_frames"));
  CHECK(Contains(run.output, "IGNORED"));
  CHECK(Contains(run.output, kLoadBanner));
}

TEST_CASE("serve flags: neither flag changes anything when neither is passed") {
  // The RED line for the whole wave: a server started without either flag must
  // be byte-identical to the pre-L2 one. 999 per modality is what
  // multimodal.py:331-333 resolves, and no request reaches it.
  const ChildRun run = RunServer(kMissingModel);
  INFO("child output:\n" << run.output);
  CHECK(Contains(run.output, "language-model-only=OFF"));
  CHECK(Contains(run.output, "no per-modality limit set"));
  CHECK(Contains(run.output, "default 999"));
  CHECK(Contains(run.output, kLoadBanner));
}
