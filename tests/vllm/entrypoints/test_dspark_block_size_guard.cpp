// SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225) — the DSpark block floor, from the loader.
//
// The floor itself is `SpeculativeConfig::ResolveDspark`
// (include/vllm/config/speculative.h:179-185), ported under SPEC-DSPARK W1 from
// vllm/config/speculative.py:1003-1027 @ 555967922. It was already correct and
// already unit-tested at tests/vllm/config/test_speculative_dspark.cpp:99-107.
// It was also unreachable: both production call sites
// (src/vllm/entrypoints/model_loader.cpp:881-883 and :1675-1677) passed
// std::nullopt for n_predict AND for dspark_block_size, so no user could arrive
// at the check. .agents/reachability.md calls this the unpassed-parameter shape,
// and its rule is that the smallest failing test enters through the production
// entry point rather than constructing the value by hand.
//
// So these cases enter at LoadedEngine::ResolveSpecConfig, the function the
// LoadedEngine constructor calls (model_loader.cpp:1099) to finalize the
// entrypoint's speculative config against the checkpoint. They write a real
// draft config.json and let the loader read it, exactly as the production path
// does.
//
// RED before the fix: every THROWS case below returns a resolved config instead,
// because the loader hands ResolveDspark two std::nullopt values.
//
// Ported behavior, all @ 555967922:
//   * :1003-1027 — k below the block is a HARD error. Upstream's own comment
//     says a smaller k "produce[s] incorrect output", not merely lower
//     acceptance, because the block/Markov machinery gets an unsupported layout.
//   * :945-961  — the Gemma4 draft's block_size normalizes onto n_predict.
//   * :973-979  — k defaults to n_predict when the draft config carries one.
//   * :990-994  — k with no n_predict anywhere is an error.
//
// ONE DIVERGENCE, argued in .agents/specs/dspark-block-size-guard.md section 2:
// the floor falls back to `block_size` when `dspark_block_size` is absent.
// Upstream reads only `dspark_block_size`, an identifier that appears in no file
// of the pinned checkout except speculative.py, and NEITHER published Qwen3
// draft sets it — deepseek-ai/dspark_qwen3_4b_block7 and
// RadixArk/Qwen3.8-27B-DSpark @ 85ef153b both carry `block_size: 7` and no
// n_predict, and the :945-961 normalization is guarded by Gemma4DSparkModel. A
// literal port would therefore be keyed on a field no checkpoint we support
// sets. The "block_size supplies the floor" case below is that divergence, and
// it is the one that protects the lane we actually ship.
#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

#include "support/process_id.h"

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"

namespace fs = std::filesystem;
using vllm::SpeculativeConfig;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

namespace {

// A scratch draft checkpoint directory holding just a config.json. The loader
// resolves the draft directory with ResolveDflashDraftDir, which accepts any
// directory that contains a config.json, so this is the real production read.
class ScratchDraft {
 public:
  explicit ScratchDraft(const std::string& config_json) {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_dspark_guard_" + std::to_string(vllm_test::ProcessId()) + "_" +
            std::to_string(counter++));
    fs::create_directories(dir_);
    std::ofstream out(dir_ / "config.json");
    out << config_json;
    out.close();
  }
  ~ScratchDraft() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  ScratchDraft(const ScratchDraft&) = delete;
  ScratchDraft& operator=(const ScratchDraft&) = delete;

  std::string path() const { return dir_.string(); }

 private:
  fs::path dir_;
};

// The CLI-side config the entrypoint builds before the checkpoint is read:
// method, the separate draft checkpoint, and the user's k.
EngineParams DsparkParams(const std::string& draft_path, std::optional<int> k) {
  EngineParams params;
  SpeculativeConfig cli;
  cli.method = "dspark";
  cli.draft_model_path = draft_path;
  cli.num_speculative_tokens = k;
  params.speculative_config = cli;
  return params;
}

// The 4B/27B published shape: a self-contained Qwen3 DSpark draft whose block
// depth is spelled `block_size`, with no n_predict and no dspark_block_size.
const char* kQwen3Block7 = R"({
  "architectures": ["Qwen3DSparkModel"],
  "model_type": "qwen3",
  "block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 151669,
  "target_layer_ids": [1, 9, 17, 25, 33]
})";

// The DeepSeek-V4 shape upstream's getattr actually reads.
const char* kDsv4Block7 = R"({
  "architectures": ["Qwen3DSparkModel"],
  "model_type": "qwen3",
  "dspark_block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 151669,
  "target_layer_ids": [1, 9, 17, 25, 33]
})";

// The Gemma4 draft whose block_size upstream normalizes onto n_predict.
const char* kGemma4Block7 = R"({
  "architectures": ["Gemma4DSparkModel"],
  "model_type": "gemma4",
  "block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 151669,
  "target_layer_ids": [1, 9, 17, 25, 33]
})";

}  // namespace

TEST_CASE("the loader refuses k below the draft's dspark_block_size") {
  // speculative.py:1003-1027, reached from the loader rather than by hand.
  const ScratchDraft draft(kDsv4Block7);
  const EngineParams params = DsparkParams(draft.path(), 6);
  CHECK_THROWS_AS(LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
                  std::invalid_argument);
}

TEST_CASE("the loader refuses k below the draft's block_size") {
  // THE DIVERGENCE, and the case that covers every published Qwen3 draft.
  // Upstream accepts this: n_predict is None and dspark_block_size is None, so
  // neither :973-988 nor :1003-1027 fires. Our draft block is sized by k alone
  // (spec_decode/dspark/speculator.h:56) and no weight is block-shaped, so k=6
  // against a block-7 checkpoint drafts a structurally wrong block in silence.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), 6);
  CHECK_THROWS_AS(LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
                  std::invalid_argument);
}

TEST_CASE("the refusal names the block and the k that was asked for") {
  // Upstream's message names both values, and so must ours: the user has to
  // learn which number to raise k to. It also names the KEY the block came
  // from, which upstream can hard-code because it reads one key and we cannot
  // because we read two. On a published Qwen3 draft the 7 comes from
  // `block_size`, so a message saying `dspark_block_size` would send the reader
  // looking through their config for a key that is not in it.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), 6);
  try {
    LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
    FAIL("expected a refusal for k=6 against a block-7 DSpark draft");
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    INFO("what: ", what);
    CHECK(what.find('7') != std::string::npos);
    CHECK(what.find('6') != std::string::npos);
    CHECK(what.find(">= block_size (7)") != std::string::npos);
    CHECK(what.find("dspark_block_size") == std::string::npos);
    // speculative.py:1024-1026 — the sentence that says what to type.
    CHECK(what.find("Use num_speculative_tokens=7 or larger") !=
          std::string::npos);
  }
}

TEST_CASE("the refusal names dspark_block_size when that is the key") {
  // The upstream-keyed shape gets upstream's wording, unchanged. The key name
  // is threaded rather than assumed, so this pins both ends of the choice.
  const ScratchDraft draft(kDsv4Block7);
  const EngineParams params = DsparkParams(draft.path(), 6);
  try {
    LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
    FAIL("expected a refusal for k=6 against a block-7 DSpark draft");
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    INFO("what: ", what);
    CHECK(what.find(">= dspark_block_size (7)") != std::string::npos);
  }
}

TEST_CASE("k at the block is accepted unchanged") {
  // The floor is >=, not >. This is the value both published drafts want.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), 7);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  CHECK(cfg->method == "dspark");
  CHECK(cfg->parallel_drafting);
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 7);
  REQUIRE(cfg->draft_model_path.has_value());
  CHECK(*cfg->draft_model_path == draft.path());
}

TEST_CASE("k above the block is accepted unchanged") {
  // Nothing this row does may narrow a value that works today.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), 14);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 14);
}

TEST_CASE("a Gemma4 draft's block_size defaults k") {
  // speculative.py:945-961 then :973-979. Before this row the loader threw
  // "requires num_speculative_tokens" before ResolveDspark could apply the
  // default, so the threaded n_predict would have been unreachable.
  const ScratchDraft draft(kGemma4Block7);
  const EngineParams params = DsparkParams(draft.path(), std::nullopt);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  CHECK(cfg->n_predict == 7);
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 7);
}

TEST_CASE("a native Qwen3 draft still requires k") {
  // speculative.py:990-994. The native config carries no n_predict, so there is
  // nothing to default from and the existing message must survive.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), std::nullopt);
  CHECK_THROWS_AS(LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
                  std::invalid_argument);
}

TEST_CASE("a draft path with no config.json resolves as it did before") {
  // ResolveSpecConfig had no filesystem dependency before this row. Reading the
  // draft config must not turn a missing checkpoint into a config-time failure:
  // LoadDsparkDraft owns that message and names the path it looked in.
  const EngineParams params =
      DsparkParams("/nonexistent/dspark/draft/for/this/test", 7);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 7);
}

// ─── The SECOND call site: the draft load inside FromModelDir ────────────────
//
// Everything above enters at LoadedEngine::ResolveSpecConfig, which the
// LoadedEngine constructor calls. It is not the first resolution a DSpark run
// meets. `FromModelDir` resolves the draft config itself, before it constructs
// the engine (src/vllm/entrypoints/model_loader.cpp, `maybe_load_dflash`), so
// that site decides what the user sees and this suite did not touch it.
//
// It also could not, until the resolution moved ahead of the model load: the
// site sat behind `LoadShards`, so reaching it needed a real target checkpoint
// and no CPU test could arrive. That is why reverting the site alone left this
// file 8/8 green while half the change was unmeasured. The cases below enter at
// `FromModelDir` with a model directory that does not exist, which is the shape
// `test_loaded_engine_dense.cpp` already uses to pin the device resolution
// "BEFORE any path I/O": the speculative config is refused, or is resolved
// without complaint, before the loader has an opinion about the target at all.
//
// The k the site used to pass was `ResolvedNumSpeculativeTokens()`
// (include/vllm/config/speculative.h, `SpeculativeConfig::ResolvedNumSpeculativeTokens`),
// which is `num_speculative_tokens.value_or(n_predict)` and therefore ZERO for a
// user who named no k — the CLI never fills `n_predict`. With the floor made
// reachable that refused a k of 0 against a key the checkpoint does not carry,
// on a lane that works today.

TEST_CASE("the loader refuses a short k before it touches the model directory") {
  // RED before the repair: FromModelDir never resolved the draft config until
  // after LoadShards, so this reported "model path is not a directory" and the
  // block floor was not consulted on the path a user actually takes.
  const ScratchDraft draft(kQwen3Block7);
  EngineParams params = DsparkParams(draft.path(), 6);
  try {
    (void)LoadedEngine::FromModelDir("/nonexistent/vllm-cpp/dspark/target",
                                     params);
    FAIL("expected a refusal for k=6 against a block-7 DSpark draft");
  } catch (const std::exception& e) {
    const std::string what = e.what();
    INFO("what: ", what);
    CHECK(what.find("block_size (7)") != std::string::npos);
    CHECK(what.find("got 6") != std::string::npos);
    CHECK(what.find("model path is not a directory") == std::string::npos);
  }
}

TEST_CASE("the loader does not refuse an absent k against a k of zero") {
  // The Gemma4 default (speculative.py:945-961 then :973-979) has to reach the
  // loader's own draft-load site, not only ResolveSpecConfig. Passing
  // ResolvedNumSpeculativeTokens() here collapses "the user named no k" to k=0
  // and refuses it against the draft's block, quoting a number nobody typed.
  const ScratchDraft draft(kGemma4Block7);
  EngineParams params = DsparkParams(draft.path(), std::nullopt);
  try {
    (void)LoadedEngine::FromModelDir("/nonexistent/vllm-cpp/dspark/target",
                                     params);
    FAIL("expected the missing target directory to be the failure");
  } catch (const std::exception& e) {
    const std::string what = e.what();
    INFO("what: ", what);
    CHECK(what.find("model path is not a directory") != std::string::npos);
    CHECK(what.find("num_speculative_tokens") == std::string::npos);
    CHECK(what.find("got 0") == std::string::npos);
  }
}

TEST_CASE("the loader keeps the native Qwen3 no-k message at the draft load") {
  // The shipped lane: a native Qwen3 draft carries no n_predict, so k stays
  // required and the message that names WHY must survive. Before the repair the
  // first resolution a user met was the draft-load site, which said
  // "num_speculative_tokens >= dspark_block_size (7); got 0" instead — a key the
  // checkpoint does not carry and a k the user never supplied.
  const ScratchDraft draft(kQwen3Block7);
  EngineParams params = DsparkParams(draft.path(), std::nullopt);
  try {
    (void)LoadedEngine::FromModelDir("/nonexistent/vllm-cpp/dspark/target",
                                     params);
    FAIL("expected a DSpark draft with no k to be refused");
  } catch (const std::exception& e) {
    const std::string what = e.what();
    INFO("what: ", what);
    CHECK(what.find("requires num_speculative_tokens") != std::string::npos);
    CHECK(what.find("a DSpark draft config carries no n_predict") !=
          std::string::npos);
    CHECK(what.find("got 0") == std::string::npos);
  }
}

// ─── The speculators layout ─────────────────────────────────────────────────
//
// Both shipped config layouts must resolve the floor identically. Nothing
// committed covered the speculators one, although it is one of the two this
// engine ships, so these two cases pin the floor on it.
//
// They do NOT prove the `TranslateSpeculatorsDsparkConfig` call in
// `ReadDsparkDraftKeys`. The speculators layout spells the draft BODY under
// `transformer_layer_config`, but `block_size` sits at the TOP LEVEL of the raw
// document — that is where the translation copies it FROM
// (src/vllm/model_executor/models/qwen3_dspark.cpp, the copied-key list), and
// the verbatim RedHatAI/Qwen3.6-35B-A3B-speculator.dspark fixture in
// tests/vllm/models/test_qwen3_dspark_config.cpp carries it there. Deleting the
// translate call leaves all five focused suites green, measured on 2026-08-18
// (.agents/specs/dspark-block-size-guard.md §6c). The call stays because the
// guard must read the same normalized shape `LoadDsparkDraft` builds, and that
// reason is recorded as UNMEASURED rather than as tested here.
const char* kSpeculatorsBlock7 = R"({
  "speculators_model_type": "dspark",
  "block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 151669,
  "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
  "transformer_layer_config": {
    "model_type": "qwen3",
    "hidden_size": 2048,
    "num_hidden_layers": 1
  }
})";

TEST_CASE("the speculators layout supplies the same block floor") {
  const ScratchDraft draft(kSpeculatorsBlock7);
  const EngineParams params = DsparkParams(draft.path(), 6);
  CHECK_THROWS_AS(LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
                  std::invalid_argument);
}

TEST_CASE("the speculators layout accepts k at its block") {
  const ScratchDraft draft(kSpeculatorsBlock7);
  const EngineParams params = DsparkParams(draft.path(), 7);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 7);
}
