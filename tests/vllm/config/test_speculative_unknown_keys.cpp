// #1160 — `--speculative-config` accepted unknown JSON keys and dropped them.
//
// The defect IS the happy path. Before the fix, every case below that expects a
// refusal PARSED CLEANLY and returned a config with the key discarded, so a run
// carrying `"draft_sample_method":"probabilistic"` proceeded GREEDY with no
// warning and exit 0, and a misspelled `"num_speculatve_tokens"` fell back to
// the resolved default. A happy-path-only test cannot see either, which is why
// these cases assert on refusals and on the refusal MESSAGE.
//
// The key set mirrored here is the `SpeculativeConfig` field set at the pinned
// oracle `555967922`, `vllm/config/speculative.py:85-283`. Upstream deserializes
// `--speculative-config` into that dataclass, so a name declared there is a name
// vLLM accepts. Three classes, and the tests below pin all three:
//
//   1. HONOURED — `method`, `num_speculative_tokens`, `model`,
//      `prompt_lookup_min`, `prompt_lookup_max`. Parsed and used.
//   2. ACCEPTED AT THE IMPLEMENTED VALUE — `draft_sample_method` (:283, only
//      "greedy") and `rejection_sample_method` (:216, only "standard"). Those
//      values ARE what this engine does, so a vLLM config that spells the
//      upstream default explicitly still runs. Any other value names
//      `SPEC-ACCEPT-VARIANTS`, per AGENTS.md "Refuse an unimplemented arm with a
//      message that names the missing part".
//   3. REFUSED — every other upstream field, and every name upstream does not
//      declare at all. Both are named in the message; the two messages differ so
//      a real vLLM key does not read as a typo.
//
// Upstream has no unit test for `--speculative-config` key admission: its
// dataclass gets the rejection for free from pydantic's `extra="forbid"`
// (`@config` on `SpeculativeConfig`, speculative.py:81-83), which is exactly the
// guarantee this file re-establishes for the hand-written C++ parser.
#include <doctest/doctest.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "vllm/config/speculative.h"

using vllm::ParseSpeculativeConfigJson;
using vllm::SpeculativeConfig;

namespace {

// The refusal message, or "" when the parse SUCCEEDED. Returning the message
// rather than a bool is what lets a case assert the offending key is NAMED: a
// refusal that says only "invalid config" leaves the user with the same search
// the silent drop did.
std::string RefusalMessage(const std::string& json_text) {
  try {
    ParseSpeculativeConfigJson(json_text);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

bool Mentions(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// A valid DSpark config with one extra key spliced in — the exact reproduction
// shape from the issue.
std::string DsparkWith(const std::string& extra_key_and_value) {
  return R"({"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7",)"
         R"("num_speculative_tokens":7,)" +
         extra_key_and_value + "}";
}

}  // namespace

TEST_CASE("draft_sample_method=probabilistic is refused, not silently dropped") {
  // The issue's reproduction. speculative.py:283 declares the key and
  // speculative.py:77 declares its two values; we implement only the greedy one
  // (`include/vllm/v1/worker/gpu/spec_decode/dspark/speculator.h:36-38` and the
  // deferred stochastic branch at
  // `include/vllm/v1/spec_decode/rejection_sampler.h:53-57`).
  const std::string msg =
      RefusalMessage(DsparkWith(R"("draft_sample_method":"probabilistic")"));
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "draft_sample_method"));
  CHECK(Mentions(msg, "probabilistic"));
  // Names the missing part and where it is owed, so the refusal is actionable.
  CHECK(Mentions(msg, "SPEC-ACCEPT-VARIANTS"));
}

TEST_CASE("draft_sample_method=greedy is ACCEPTED") {
  // "greedy" IS what this engine does, and it is upstream's default
  // (speculative.py:283), so a vLLM config that spells the default explicitly
  // must keep running. Refusing this would be over-strict, not safe.
  const SpeculativeConfig cfg =
      ParseSpeculativeConfigJson(DsparkWith(R"("draft_sample_method":"greedy")"));
  CHECK(cfg.method == "dspark");
  CHECK(cfg.num_speculative_tokens.has_value());
  CHECK(*cfg.num_speculative_tokens == 7);
  CHECK(cfg.draft_model_path.has_value());
}

TEST_CASE("rejection_sample_method: standard accepted, synthetic and block refused") {
  // speculative.py:216 — "standard" is upstream's default and the semantics the
  // landed verify implements (accept-iff-equal under greedy decode,
  // `rejection_sampler.h`). "synthetic" (:224-232) and "block" are the branches
  // at `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:82-91`, and both are
  // listed as DEFERRED in `rejection_sampler.h`.
  const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
      DsparkWith(R"("rejection_sample_method":"standard")"));
  CHECK(cfg.method == "dspark");

  for (const std::string variant : {"synthetic", "block"}) {
    const std::string msg = RefusalMessage(
        DsparkWith(R"("rejection_sample_method":")" + variant + "\""));
    REQUIRE_MESSAGE(msg != "", variant);
    // The OFFENDING value, quoted in place. Asserting the bare word would pass
    // even if the message named the other variant, because the refusal text
    // mentions both.
    CHECK(Mentions(msg, "rejection_sample_method \"" + variant + "\""));
    CHECK(Mentions(msg, "SPEC-ACCEPT-VARIANTS"));
  }
}

TEST_CASE("a misspelled honoured key is refused by name") {
  // The issue's second case: `num_speculatve_tokens` used to fall through to the
  // resolved default, so the user got a k they did not ask for.
  const std::string msg =
      RefusalMessage(R"({"method":"ngram","num_speculatve_tokens":7})");
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "num_speculatve_tokens"));
  // The correct spelling is offered, so the message closes the search.
  CHECK(Mentions(msg, "num_speculative_tokens"));
}

TEST_CASE("an unknown key upstream does not declare is refused by name") {
  const std::string msg = RefusalMessage(DsparkWith(R"("not_a_vllm_key":1)"));
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "not_a_vllm_key"));
  CHECK(Mentions(msg, "unknown key"));
}

TEST_CASE("an upstream key this engine does not implement is refused by name") {
  // These ARE vLLM keys (speculative.py:110, :126, :139, :178), so the message
  // must not call them unknown — that would send the user hunting for a typo
  // that is not there. It names the key and says the engine does not implement
  // it at this pin.
  // `const std::string`, not `const char*`: doctest stringifies a `char*` INFO
  // payload as a bool, so a failure would log "1" instead of the key that failed.
  for (const std::string key : {"quantization", "max_model_len",
                                "disable_padded_drafter_batch",
                                "num_speculative_tokens_per_batch_size"}) {
    const std::string msg = RefusalMessage(DsparkWith("\"" + key + "\":null"));
    REQUIRE_MESSAGE(msg != "", key);
    CHECK(Mentions(msg, key));
    CHECK_FALSE(Mentions(msg, "unknown key"));
  }
}

TEST_CASE("a key is judged even when its value is null") {
  // `null` is not "absent". The honoured optional keys treat null as absent
  // (that is the landed contract, exercised below), but an UNIMPLEMENTED key
  // spelled with a null value still names a capability we do not have, and
  // dropping it silently is the defect this issue is about.
  CHECK(RefusalMessage(DsparkWith(R"("draft_sample_method":null)")) != "");
  CHECK(RefusalMessage(DsparkWith(R"("not_a_vllm_key":null)")) != "");
}

TEST_CASE("REGRESSION: every currently valid config still parses") {
  // The five honoured keys, across all five accepted methods. If the admission
  // check is too strict, this is what goes red.
  SUBCASE("mtp, method only") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(R"({"method":"mtp"})");
    CHECK(cfg.method == "mtp");
    CHECK_FALSE(cfg.num_speculative_tokens.has_value());
  }
  SUBCASE("mtp with an explicit k") {
    const SpeculativeConfig cfg =
        ParseSpeculativeConfigJson(R"({"method":"mtp","num_speculative_tokens":4})");
    CHECK(*cfg.num_speculative_tokens == 4);
  }
  SUBCASE("dflash with a draft checkpoint") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash",)"
        R"("num_speculative_tokens":16})");
    CHECK(cfg.method == "dflash");
    CHECK(*cfg.draft_model_path == "z-lab/Qwen3.6-27B-DFlash");
    CHECK(*cfg.num_speculative_tokens == 16);
  }
  SUBCASE("ngram with the full lookup window") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"ngram","num_speculative_tokens":4,)"
        R"("prompt_lookup_min":3,"prompt_lookup_max":5})");
    CHECK(cfg.method == "ngram");
    CHECK(*cfg.prompt_lookup_min == 3);
    CHECK(*cfg.prompt_lookup_max == 5);
  }
  SUBCASE("dspark with a draft checkpoint") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7",)"
        R"("num_speculative_tokens":7})");
    CHECK(cfg.method == "dspark");
    CHECK(*cfg.num_speculative_tokens == 7);
  }
  SUBCASE("draft_model with a separate checkpoint") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"draft_model","model":"Qwen/Qwen3.6-0.6B",)"
        R"("num_speculative_tokens":3})");
    CHECK(cfg.method == "draft_model");
    CHECK(*cfg.draft_model_path == "Qwen/Qwen3.6-0.6B");
  }
  SUBCASE("an explicit null on an honoured optional key still means absent") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"mtp","num_speculative_tokens":null,)"
        R"("prompt_lookup_min":null,"prompt_lookup_max":null})");
    CHECK_FALSE(cfg.num_speculative_tokens.has_value());
    CHECK_FALSE(cfg.prompt_lookup_min.has_value());
    CHECK_FALSE(cfg.prompt_lookup_max.has_value());
  }
}

TEST_CASE("REGRESSION: the landed refusals are unchanged") {
  // The admission check runs BEFORE these, so a key error must not mask a
  // method error, and the method/required-key refusals must still fire.
  CHECK_THROWS_AS(ParseSpeculativeConfigJson(R"({"method":"eagle3"})"),
                  std::invalid_argument);
  CHECK_THROWS_AS(ParseSpeculativeConfigJson(R"({"num_speculative_tokens":4})"),
                  std::invalid_argument);
  CHECK_THROWS_AS(ParseSpeculativeConfigJson(R"({"method":"dflash"})"),
                  std::invalid_argument);
  CHECK_THROWS_AS(ParseSpeculativeConfigJson(R"({"method":"ngram"})"),
                  std::invalid_argument);
  CHECK_THROWS_AS(ParseSpeculativeConfigJson("not json"), std::invalid_argument);
  CHECK_THROWS_AS(ParseSpeculativeConfigJson("[1,2]"), std::invalid_argument);
  // The method check runs FIRST. `method` selects everything else, so its error
  // is the actionable one, and a key error must not displace it. Both spellings
  // are pinned: a key that PASSES admission, and one that does not.
  for (const std::string extra : {R"("draft_sample_method":"greedy")",
                                  R"("quantization":"fp8")"}) {
    const std::string msg =
        RefusalMessage(R"({"method":"suffix",)" + extra + "}");
    REQUIRE_MESSAGE(msg != "", extra);
    CHECK(Mentions(msg, "suffix"));
    CHECK(Mentions(msg, "only methods"));
  }
}
