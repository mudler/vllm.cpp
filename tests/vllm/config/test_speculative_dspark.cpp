// SPEC-DSPARK W1 — the DSpark config slice.
//
// Ported from (test-porting protocol .agents/porting.md), all @ 555967922:
//   * vllm/config/speculative.py:869-887 — method auto-detection: a draft whose
//     name contains "dspark" OR whose architectures contain "Qwen3DSparkModel"
//     / "Gemma4DSparkModel" resolves method == "dspark".
//   * vllm/config/speculative.py:1004-1027 — the HARD error: DSpark is a
//     semi-autoregressive BLOCK drafter, so num_speculative_tokens smaller than
//     the checkpoint's block size feeds the block/Markov machinery an
//     unsupported layout and yields INCORRECT (garbled) output, not merely lower
//     acceptance. Upstream raises ValueError; we throw std::invalid_argument.
//   * vllm/config/speculative.py:963-964 — parallel_drafting is set for
//     ("dflash", "dspark").
//   * vllm/v1/core/sched/scheduler.py:261-265 — DSpark reserves EXACTLY
//     num_spec_tokens lookahead slots, NOT DFlash's num_spec_tokens + 1
//     (:256-260), because the anchor query is itself the first prediction
//     (dspark/speculator.py:5-11,50-52) rather than a separate bonus query.
//   * vllm/config/speculative.py:706-709,875 — the `model` key: the DSpark draft
//     is a SEPARATE checkpoint for the Qwen3/Gemma4 families (e.g.
//     deepseek-ai/dspark_qwen3_8b_block7). The DeepSeek-V4 variant that ships
//     its draft inside the target checkpoint is OUT OF SCOPE for this row
//     (hardware-blocked), so at this pin the key is required, exactly as for
//     DFlash (`speculative.cpp` dflash branch).
//
// Upstream has no unit test for this resolution (it is covered e2e by
// tests/v1/e2e/spec_decode/test_spec_decode.py::test_dspark_correctness_and_
// acceptance_rate, which needs a GPU and two checkpoints). These cases pin the
// pure-config behavior that e2e test depends on.
//
// RED-first: before the W1 fix, ParseSpeculativeConfigJson rejects every method
// outside {mtp, dflash, ngram, draft_model} (src/vllm/config/speculative.cpp:44)
// so the first case throws, and SpeculativeConfig::ResolveDspark does not exist.
#include <doctest/doctest.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "vllm/config/speculative.h"

using vllm::ParseSpeculativeConfigJson;
using vllm::SpeculativeConfig;

TEST_CASE("--speculative-config accepts method dspark with a draft model") {
  const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
      R"({"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7",)"
      R"("num_speculative_tokens":7})");
  CHECK(cfg.method == "dspark");
  CHECK(cfg.draft_model_path.has_value());
  CHECK(*cfg.draft_model_path == "deepseek-ai/dspark_qwen3_4b_block7");
  CHECK(cfg.num_speculative_tokens.has_value());
  CHECK(*cfg.num_speculative_tokens == 7);
}

TEST_CASE("method dspark requires the separate draft checkpoint") {
  // speculative.py:706-709 — the Qwen3/Gemma4 DSpark draft is a separate
  // checkpoint; without `model` there is nothing to load.
  CHECK_THROWS_AS(ParseSpeculativeConfigJson(
                      R"({"method":"dspark","num_speculative_tokens":7})"),
                  std::invalid_argument);
}

TEST_CASE("ResolveDspark defaults k to n_predict when the draft carries one") {
  // speculative.py:973-979 — a Gemma4 DSpark draft gets n_predict = block_size
  // (:957-961) and k defaults to it. parallel_drafting is set at :963-964.
  const SpeculativeConfig cfg = SpeculativeConfig::ResolveDspark(
      /*n_predict=*/7, /*dspark_block_size=*/std::nullopt, /*user_k=*/std::nullopt);
  CHECK(cfg.method == "dspark");
  CHECK(cfg.n_predict == 7);
  CHECK(cfg.num_speculative_tokens.has_value());
  CHECK(*cfg.num_speculative_tokens == 7);
  CHECK(cfg.parallel_drafting);
}

TEST_CASE("ResolveDspark requires k when the draft carries no n_predict") {
  // A native Qwen3DSparkModel config carries `block_size` but NOT `n_predict`
  // (only the Gemma4 branch maps one to the other, speculative.py:957-961), so
  // upstream falls through to :990-994 "A speculative model was provided, but
  // `num_speculative_tokens` was not provided". This is why the upstream test
  // passes num_speculative_tokens=7 explicitly (test_spec_decode.py:1495-1503).
  CHECK_THROWS_AS(SpeculativeConfig::ResolveDspark(std::nullopt, std::nullopt,
                                                   std::nullopt),
                  std::invalid_argument);
  const SpeculativeConfig cfg =
      SpeculativeConfig::ResolveDspark(std::nullopt, std::nullopt, 7);
  CHECK(cfg.n_predict == 0);
  CHECK(*cfg.num_speculative_tokens == 7);
}

TEST_CASE("ResolveDspark keeps the n_predict divisibility rule") {
  // speculative.py:980-988 — k above n_predict must be a multiple of it.
  const SpeculativeConfig multiple =
      SpeculativeConfig::ResolveDspark(7, std::nullopt, 14);
  CHECK(*multiple.num_speculative_tokens == 14);
  CHECK_THROWS_AS(SpeculativeConfig::ResolveDspark(7, std::nullopt, 8),
                  std::invalid_argument);
}

TEST_CASE("ResolveDspark rejects a k below dspark_block_size") {
  // speculative.py:1003-1027 — the DSV4-style `dspark_block_size` floor. A
  // smaller k produces INCORRECT output, so this is a hard error, not a warning.
  CHECK_THROWS_AS(SpeculativeConfig::ResolveDspark(std::nullopt, 7, 6),
                  std::invalid_argument);
  CHECK_THROWS_AS(SpeculativeConfig::ResolveDspark(std::nullopt, 15, 7),
                  std::invalid_argument);
  const SpeculativeConfig at_floor =
      SpeculativeConfig::ResolveDspark(std::nullopt, 7, 7);
  CHECK(*at_floor.num_speculative_tokens == 7);
}

TEST_CASE("DSpark reserves exactly k lookahead slots, DFlash reserves k + 1") {
  // scheduler.py:256-265 — the single scheduler-visible consequence of
  // anchor-as-first-prediction.
  const SpeculativeConfig dspark =
      SpeculativeConfig::ResolveDspark(std::nullopt, std::nullopt, 7);
  CHECK(dspark.NumLookaheadTokens() == 7);
  const SpeculativeConfig dflash = SpeculativeConfig::ResolveDflash(7);
  CHECK(dflash.NumLookaheadTokens() == 8);
}

TEST_CASE("DSpark is a draft-hidden-state method and is not n-gram") {
  const SpeculativeConfig cfg =
      SpeculativeConfig::ResolveDspark(std::nullopt, std::nullopt, 7);
  CHECK(cfg.use_eagle());       // speculative.py:1328
  CHECK(cfg.use_dspark());      // speculative.py:1333-1334
  CHECK_FALSE(cfg.use_dflash());  // the DFlash lookahead branch must NOT fire
  CHECK_FALSE(cfg.use_ngram());
}

TEST_CASE("DSpark method resolution from the draft checkpoint identity") {
  // speculative.py:869-887 — name substring OR architecture string.
  CHECK(SpeculativeConfig::IsDsparkDraft("deepseek-ai/dspark_qwen3_8b_block7",
                                         /*architectures=*/{}));
  CHECK(SpeculativeConfig::IsDsparkDraft("RedHatAI/Qwen3.6-35B-A3B-speculator.dspark",
                                         {}));
  CHECK(SpeculativeConfig::IsDsparkDraft("some/local-snapshot",
                                         {"Qwen3DSparkModel"}));
  CHECK(SpeculativeConfig::IsDsparkDraft("some/local-snapshot",
                                         {"Gemma4DSparkModel"}));
  CHECK_FALSE(SpeculativeConfig::IsDsparkDraft("z-lab/Qwen3.6-27B-DFlash",
                                               {"DFlashDraftModel"}));
}

// ─── SPEC-DSPARK-QWEN3-ROUTING (#1193) ───────────────────────────────────────
//
// BEYOND-PIN. Both cases below mirror vllm-project/vllm#52197, merged
// 2026-08-17 at 7075ddac28c25d4fd2b84bc2a9a6c5ffde0345c8, which is AHEAD of the
// pin 555967922. The PR ships no test of its own — its "Test Plan" is empty and
// its "Test Result" is a pasted gsm8k serve log — so there is nothing to port
// one-for-one and these are written here. .agents/porting.md requires the
// adaptation to be recorded, and this paragraph is that record;
// .agents/specs/dspark-qwen3-routing.md §2 carries the pin decision.
//
// The REACHABILITY of both functions — that the loader calls them at all — is
// tests/vllm/entrypoints/test_dspark_draft_routing.cpp, because a case here
// proves the predicate works and never that anything reaches it.

TEST_CASE("a DSparkDraftModel draft with model_type qwen3 IS a DSpark draft") {
  // vllm#52197 hunk 1. The model id carries no "dspark" substring, so only the
  // architecture arm can answer: with the id spelled the other way the case
  // would pass on speculative.py:882 alone and prove nothing about the
  // architecture list — the mute switch this test set has to avoid.
  CHECK(SpeculativeConfig::IsDsparkDraft("RadixArk/Qwen3.8-27B-Draft",
                                         {"DSparkDraftModel"}, "qwen3"));
  // The PAIR is the condition. The same architecture string names a DeepSeek-V4
  // draft under any other model_type, and upstream's leading branch does not
  // claim it.
  CHECK_FALSE(SpeculativeConfig::IsDsparkDraft("RadixArk/Qwen3.8-27B-Draft",
                                               {"DSparkDraftModel"},
                                               "deepseek_v4"));
  // And the pinned two-argument question keeps the pinned answer.
  CHECK_FALSE(
      SpeculativeConfig::IsDsparkDraft("RadixArk/Qwen3.8-27B-Draft",
                                       {"DSparkDraftModel"}));
}

TEST_CASE("DSpark architecture normalization routes the three implemented names") {
  // vllm#52197 hunk 2 branch 1, and speculative.py:934-944's two survivors. All
  // three answer with the one draft lane this engine implements
  // (LoadQwen3DSpark), which is what upstream's update_arch_() writes back onto
  // the draft config.
  CHECK(SpeculativeConfig::ResolveDsparkArchitecture({"DSparkDraftModel"},
                                                     "qwen3") ==
        "Qwen3DSparkModel");
  CHECK(SpeculativeConfig::ResolveDsparkArchitecture({"Qwen3DSparkModel"},
                                                     "qwen3") ==
        "Qwen3DSparkModel");
  CHECK(SpeculativeConfig::ResolveDsparkArchitecture({"Gemma4DSparkModel"},
                                                     "gemma4") ==
        "Qwen3DSparkModel");
}

TEST_CASE("DSpark architecture normalization REFUSES the DeepSeek-V4 fallback") {
  // The one tracked divergence from speculative.py:940-944 (§7 R2): upstream
  // rewrites the config onto model_type "deepseek_v4" and lets the DeepSeek-V4
  // path take it; we have only a stub for that path, so the arm is refused with
  // the missing part NAMED, as AGENTS.md requires.
  std::string message;
  try {
    SpeculativeConfig::ResolveDsparkArchitecture({"DSparkDraftModel"},
                                                 "deepseek_v4");
  } catch (const std::invalid_argument& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("DeepSeek-V4") != std::string::npos);
  CHECK(message.find("not implemented") != std::string::npos);

  // The catch-all is upstream's, so an architecture list that names nothing
  // DSpark at all lands in the same refusal rather than loading as a Qwen3
  // draft.
  CHECK_THROWS_AS(
      SpeculativeConfig::ResolveDsparkArchitecture({"DeepseekV4ForCausalLM"},
                                                   "deepseek_v4"),
      std::invalid_argument);
}
