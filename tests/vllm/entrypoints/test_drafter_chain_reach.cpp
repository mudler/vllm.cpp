// SPEC-DRAFTER-CHAIN W1 (#1522) — REACHABILITY and ORDERING, not parsing.
// `tests/vllm/config/test_speculative_drafter_chain.cpp` proves the document is
// parsed, validated and refused; this file proves that something a user can
// arrive through actually reads the parsed field, and that it does so BEFORE any
// weight I/O.
//
// WHY THIS FILE EXISTS SEPARATELY. AGENTS.md `## Nothing lands dead`. W1 lands a
// field that nothing resolves yet, so the one thing that could make it dead is
// exactly the thing a parser test cannot see: a stored `drafter_chain` that no
// production caller ever reads. `LoadedEngine::ResolveSpecConfig` is that
// reader. It REFUSES a chain by name rather than reducing it to one drafter or
// to no speculation at all, because a chain document that started a
// single-drafter engine in silence is the #1160 failure with a bigger blast
// radius — the user would then measure the wrong engine.
//
// THE MODEL DIRECTORY IS DELIBERATELY NONEXISTENT, and that is the ORDERING
// assertion rather than a convenience. `FromModelDir` on a missing path throws
// `model path is not a directory`. So a case that asserts the CHAIN message
// comes back from a missing path has proved the refusal ran before the loader
// so much as stat()ed the directory, which is strictly before any weight I/O.
// Deleting the guard in `FromModelDir` does not make the chain load; it makes
// this file report the directory error instead, and that is what goes red.
//
// THE TWO REACHABILITY MUTATIONS for this wave:
//   1. delete the chain refusal in `LoadedEngine::ResolveSpecConfig`;
//   2. delete the early guard at the top of `LoadedEngine::FromModelDir`.
// Both must redden this file. The first also reddens it through the C ABI.
#include <doctest/doctest.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "vllm.h"
#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"

namespace {

constexpr const char* kMissingModel = "/nonexistent/vllm-cpp/drafter-chain-reach";

// A VALID chain: two entries, each carrying its own configuration, in a
// preference order. Every refusal below is about the chain being unresolvable
// at this wave, never about the document being malformed — which is the whole
// distinction W1 has to hold.
constexpr const char* kChainJson =
    R"({"vllm_cpp":{"drafter_chain":[)"
    R"({"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":16},)"
    R"({"method":"ngram","num_speculative_tokens":4}]}})";

// A valid ONE-entry chain — D8's degenerate preference list.
//
// Every other case in this file uses two entries, and that left D8's loader
// half unexecuted: `use_drafter_chain()` is `!drafter_chain.empty()`, and a
// two-entry document cannot tell that predicate apart from a `size() > 1`. The
// difference is not academic. D8 says a single-entry chain "is not equivalent to
// a top-level `method`: the entry lands in the chain, `method` stays empty, and
// the loader's chain refusal still sees it." Under `size() > 1` the parse still
// succeeds, `method` is still empty, and `ResolveSpecConfig` then falls through
// to the bottom of the function and tells the user that "" is not one of
// mtp/dflash/ngram — a one-drafter document answered with a message about a
// method they never wrote. That is precisely the "measured the wrong engine"
// shape this wave exists to refuse, reached by a document D8 declares LEGAL.
constexpr const char* kSingleEntryChainJson =
    R"({"vllm_cpp":{"drafter_chain":[)"
    R"({"method":"mtp"}]}})";

// The refusal text from a call that is expected to throw, or "" when it did not.
template <typename F>
std::string ThrowMessage(F&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool Mentions(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("chain reach: ResolveSpecConfig READS the parsed chain and refuses by name") {
  // Enter where a user enters: a real `--speculative-config` document, parsed by
  // the same function the server and the C ABI call, carried on EngineParams the
  // way both fill it in. Nothing here builds a SpeculativeConfig by hand.
  vllm::entrypoints::EngineParams params;
  params.speculative_config = vllm::ParseSpeculativeConfigJson(kChainJson);
  REQUIRE(params.speculative_config->use_drafter_chain());
  REQUIRE(params.speculative_config->drafter_chain.size() == 2);

  const std::string msg = ThrowMessage([&] {
    (void)vllm::entrypoints::LoadedEngine::ResolveSpecConfig(params,
                                                             vllm::HfConfig{});
  });
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
  // It names the row and the issue that owe the resolution, which is what
  // AGENTS.md requires of a refused arm.
  CHECK(Mentions(msg, "SPEC-DRAFTER-CHAIN"));
  CHECK(Mentions(msg, "1522"));
  // It names the chain the user gave, IN ORDER — so the message is about their
  // document rather than about the feature in general.
  CHECK(Mentions(msg, "dflash"));
  CHECK(Mentions(msg, "ngram"));
  // And it is NOT the fallthrough refusal at the bottom of the same function,
  // which an empty `method` would otherwise reach and which would tell the user
  // that "" is not one of mtp/dflash/ngram. Asserting only that it threw would
  // be satisfied by exactly that wrong guard.
  CHECK_FALSE(Mentions(msg, "supported (got \"\")"));
}

TEST_CASE("chain reach: FromModelDir refuses BEFORE it looks at the path") {
  // G5's ordering half. `kMissingModel` does not exist, so a loader that reached
  // its path resolution would say so. The chain message coming back instead is
  // the proof that the refusal precedes every path, config, tokenizer and weight
  // operation in `FromModelDir`.
  vllm::entrypoints::EngineParams params;
  params.speculative_config = vllm::ParseSpeculativeConfigJson(kChainJson);

  const std::string msg = ThrowMessage([&] {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, params);
  });
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
  CHECK(Mentions(msg, "SPEC-DRAFTER-CHAIN"));
  // THE ORDERING ASSERTION. This is the string the loader produces once it has
  // started resolving the path; seeing it means the guard ran too late, or not
  // at all.
  CHECK_FALSE(Mentions(msg, "model path is not a directory"));
}

TEST_CASE("chain reach: a ONE-entry chain is seen by the loader too (D8)") {
  // #1522 D8, the half no case executed before this one. Both production
  // readers, on a chain of length one.
  vllm::entrypoints::EngineParams params;
  params.speculative_config =
      vllm::ParseSpeculativeConfigJson(kSingleEntryChainJson);
  // Facts of the PARSE only. `use_drafter_chain()` is deliberately NOT required
  // here: a REQUIRE on the predicate aborts the subcase before either loader
  // runs, so narrowing the predicate would be caught by this line and the two
  // subcases below would never execute — proving the predicate moved rather than
  // proving the loader stopped seeing a one-entry chain. Each subcase asserts
  // the loader's own message instead, so each carries its own proof.
  REQUIRE(params.speculative_config->drafter_chain.size() == 1);
  // D8's own words: the entry lands in the CHAIN and `method` stays empty, so
  // the document is NOT silently equivalent to `{"method":"mtp"}`.
  REQUIRE(params.speculative_config->method.empty());
  CHECK(params.speculative_config->drafter_chain[0].method == "mtp");
  CHECK(params.speculative_config->use_drafter_chain());

  SUBCASE("ResolveSpecConfig sees it") {
    const std::string msg = ThrowMessage([&] {
      (void)vllm::entrypoints::LoadedEngine::ResolveSpecConfig(params,
                                                              vllm::HfConfig{});
    });
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
    CHECK(Mentions(msg, "SPEC-DRAFTER-CHAIN"));
    CHECK(Mentions(msg, "mtp"));
    // THE ASSERTION A TWO-ENTRY CASE CANNOT MAKE. This is the fallthrough at the
    // bottom of `ResolveSpecConfig`, which an empty `method` reaches whenever the
    // chain guard declines to fire. A guard narrowed to `size() > 1` lands here.
    CHECK_FALSE(Mentions(msg, "supported (got \"\")"));
  }

  SUBCASE("FromModelDir sees it, still before the path") {
    const std::string msg = ThrowMessage([&] {
      (void)vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, params);
    });
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
    CHECK_FALSE(Mentions(msg, "model path is not a directory"));
  }
}

TEST_CASE("chain reach: the C ABI's speculative_config string carries the field") {
  // include/vllm.h is the public surface and `speculative_config` is ONE string,
  // so the extension needed no new ABI field — the same conclusion the residency
  // extension reached for `offload_config`.
  vllm_model_params p = vllm_model_params_default();
  p.model_path = kMissingModel;
  p.speculative_config = kChainJson;
  vllm_engine* eng = nullptr;
  CHECK(vllm_engine_load(&p, &eng) != VLLM_OK);
  CHECK(eng == nullptr);

  const std::string msg = vllm_last_error() != nullptr ? vllm_last_error() : "";
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
  CHECK(Mentions(msg, "SPEC-DRAFTER-CHAIN"));
  CHECK_FALSE(Mentions(msg, "model path is not a directory"));
}

TEST_CASE("chain reach: a MALFORMED chain is refused through the C ABI too") {
  // The parser's refusals have to survive the trip through the public surface,
  // or the message the user actually sees is not the one the parser tests pin.
  vllm_model_params p = vllm_model_params_default();
  p.model_path = kMissingModel;
  p.speculative_config = R"({"vllm_cpp":{"drafter_chain":[{"method":"eagle3"}]}})";
  vllm_engine* eng = nullptr;
  CHECK(vllm_engine_load(&p, &eng) != VLLM_OK);
  CHECK(eng == nullptr);

  const std::string msg = vllm_last_error() != nullptr ? vllm_last_error() : "";
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "eagle3"));
  CHECK(Mentions(msg, "vllm_cpp.drafter_chain[0]"));
}

TEST_CASE("G1 ADDITIVITY: a document with no chain resolves exactly as before") {
  // THE GATE THAT PROTECTS vLLM's SURFACE, and it runs on REAL documents driven
  // through the production resolver — not on a hand-built struct, which would
  // prove only that the struct still has its fields.
  //
  // Each expectation below is the pre-chain behaviour of this engine, restated
  // here so that a chain-shaped change to any of them goes red: the resolved
  // method, the resolved k, the draft path, the n-gram window defaults, and the
  // fact that `drafter_chain` stays EMPTY on every one of them.
  SUBCASE("no speculative config at all — the production default") {
    vllm::entrypoints::EngineParams params;
    const std::optional<vllm::SpeculativeConfig> resolved =
        vllm::entrypoints::LoadedEngine::ResolveSpecConfig(params,
                                                           vllm::HfConfig{});
    CHECK_FALSE(resolved.has_value());
  }
  SUBCASE("mtp with no explicit k") {
    vllm::entrypoints::EngineParams params;
    params.speculative_config =
        vllm::ParseSpeculativeConfigJson(R"({"method":"mtp"})");
    const std::optional<vllm::SpeculativeConfig> resolved =
        vllm::entrypoints::LoadedEngine::ResolveSpecConfig(params,
                                                           vllm::HfConfig{});
    REQUIRE(resolved.has_value());
    CHECK(resolved->method == "mtp");
    CHECK(resolved->n_predict == 1);
    CHECK(resolved->ResolvedNumSpeculativeTokens() == 1);
    CHECK(resolved->NumLookaheadTokens() == 1);
    CHECK(resolved->drafter_chain.empty());
    CHECK_FALSE(resolved->use_drafter_chain());
  }
  SUBCASE("mtp with an explicit k") {
    vllm::entrypoints::EngineParams params;
    params.speculative_config = vllm::ParseSpeculativeConfigJson(
        R"({"method":"mtp","num_speculative_tokens":4})");
    const std::optional<vllm::SpeculativeConfig> resolved =
        vllm::entrypoints::LoadedEngine::ResolveSpecConfig(params,
                                                           vllm::HfConfig{});
    REQUIRE(resolved.has_value());
    CHECK(resolved->ResolvedNumSpeculativeTokens() == 4);
    CHECK(resolved->drafter_chain.empty());
  }
  SUBCASE("dflash keeps its draft path and its k + 1 lookahead") {
    vllm::entrypoints::EngineParams params;
    params.speculative_config = vllm::ParseSpeculativeConfigJson(
        R"({"method":"dflash","model":"/nonexistent/vllm-cpp/dflash-draft",)"
        R"("num_speculative_tokens":16})");
    const std::optional<vllm::SpeculativeConfig> resolved =
        vllm::entrypoints::LoadedEngine::ResolveSpecConfig(params,
                                                           vllm::HfConfig{});
    REQUIRE(resolved.has_value());
    CHECK(resolved->method == "dflash");
    CHECK(resolved->parallel_drafting);
    CHECK(resolved->ResolvedNumSpeculativeTokens() == 16);
    CHECK(resolved->NumLookaheadTokens() == 17);
    REQUIRE(resolved->draft_model_path.has_value());
    CHECK(*resolved->draft_model_path == "/nonexistent/vllm-cpp/dflash-draft");
    CHECK(resolved->drafter_chain.empty());
  }
  SUBCASE("ngram keeps its 5/5 window default and its zero lookahead") {
    vllm::entrypoints::EngineParams params;
    params.speculative_config = vllm::ParseSpeculativeConfigJson(
        R"({"method":"ngram","num_speculative_tokens":4})");
    const std::optional<vllm::SpeculativeConfig> resolved =
        vllm::entrypoints::LoadedEngine::ResolveSpecConfig(params,
                                                           vllm::HfConfig{});
    REQUIRE(resolved.has_value());
    CHECK(resolved->method == "ngram");
    CHECK(*resolved->prompt_lookup_min == 5);
    CHECK(*resolved->prompt_lookup_max == 5);
    CHECK(resolved->NumLookaheadTokens() == 0);
    CHECK(resolved->drafter_chain.empty());
  }
  SUBCASE("the landed refusals on a chain-free document are untouched") {
    vllm::entrypoints::EngineParams params;
    params.speculative_config = vllm::ParseSpeculativeConfigJson(
        R"({"method":"draft_model","model":"Qwen/Qwen3.6-0.6B",)"
        R"("num_speculative_tokens":3})");
    const std::string msg = ThrowMessage([&] {
      (void)vllm::entrypoints::LoadedEngine::ResolveSpecConfig(params,
                                                               vllm::HfConfig{});
    });
    REQUIRE(msg != "");
    // The pre-existing fallthrough, NOT the new chain refusal. A guard placed
    // where it swallowed this one would pass a "CHECK_THROWS" and change which
    // engine a `draft_model` document reports as missing.
    CHECK(Mentions(msg, "draft_model"));
    CHECK_FALSE(Mentions(msg, "drafter_chain"));
  }
}

TEST_CASE("G1 ADDITIVITY: a chain-free load reaches the SAME loader failure it always did") {
  // The engine half of inertness, through `FromModelDir` on the same missing
  // path the chain cases use. Without the field, the loader must still fail on
  // the checkpoint and not on anything this wave added — which is what makes the
  // chain cases above evidence about the chain rather than about the path.
  for (const char* doc : {R"({"method":"mtp"})",
                          R"({"method":"ngram","num_speculative_tokens":4})"}) {
    CAPTURE(doc);
    vllm::entrypoints::EngineParams params;
    params.speculative_config = vllm::ParseSpeculativeConfigJson(doc);
    const std::string msg = ThrowMessage([&] {
      (void)vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, params);
    });
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "model path is not a directory"));
    CHECK_FALSE(Mentions(msg, "drafter_chain"));
  }

  // And with no speculative config at all.
  vllm::entrypoints::EngineParams bare;
  CHECK_FALSE(bare.speculative_config.has_value());
  const std::string msg = ThrowMessage([&] {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, bare);
  });
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "model path is not a directory"));
  CHECK_FALSE(Mentions(msg, "drafter_chain"));
}
