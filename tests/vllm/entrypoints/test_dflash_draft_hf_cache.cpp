// ENG-HF-MODEL-DOWNLOAD W2 (#1280): the HuggingFace cache walk, entered through
// the production loader.
//
// `vllm::transformers_utils::ResolveCachedSnapshotDir` has exactly one
// production call site, `ResolveDflashDraftDir` in
// `src/vllm/entrypoints/model_loader.cpp`. `tests/vllm/transformers_utils/
// test_hf_cache.cpp` calls the function directly, which proves the function and
// nothing about the call site, and the DSpark guard suite enters the loader with
// a plain directory, which is the branch that hands `path` back unchanged. Both
// stayed green when a fresh reviewer replaced the body of
// `ResolveDflashDraftDir` with `return path;`.
//
// So this file enters at `LoadedEngine::ResolveSpecConfig`, the function the
// LoadedEngine constructor calls, and gives it a REPOSITORY IDENTIFIER rather
// than a path. The only way that identifier can produce a config.json is the
// cache walk, so the two cases below are an A/B on the walk itself: the same
// call refuses with the cache populated and is accepted with it absent.
//
// RED when the call site goes away: with `return path;` the loader looks for
// `org/repo/config.json`, finds nothing, leaves both DSpark keys unset, and
// accepts k=6 against a block-7 draft.
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "support/process_id.h"
#include "support/test_env.h"
#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"

namespace fs = std::filesystem;
using vllm::SpeculativeConfig;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

namespace {

// The published Qwen3 DSpark draft shape: block depth spelled `block_size`,
// no n_predict and no dspark_block_size. Identical to the fixture in
// tests/vllm/entrypoints/test_dspark_block_size_guard.cpp, which is the suite
// that owns the floor itself; here it is only the payload that proves the
// config.json was found.
const char* kQwen3Block7 = R"({
  "architectures": ["Qwen3DSparkModel"],
  "model_type": "qwen3",
  "block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 151669,
  "target_layer_ids": [1, 9, 17, 25, 33]
})";

// A scratch HOME. `ResolveDflashDraftDir` passes $HOME/.cache/huggingface/hub as
// the cache root, which is the behavior the W2 relocation had to preserve, so
// the fixture writes the tree the production code actually reads.
class ScratchHome {
 public:
  ScratchHome() {
    static int counter = 0;
    const char* previous = std::getenv("HOME");
    saved_ = previous == nullptr ? std::string() : std::string(previous);
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_hf_cache_reach_" + std::to_string(vllm_test::ProcessId()) +
            "_" + std::to_string(counter++));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
    vllm_test::SetEnv("HOME", dir_.string());
  }
  ~ScratchHome() {
    vllm_test::SetEnv("HOME", saved_);
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  ScratchHome(const ScratchHome&) = delete;
  ScratchHome& operator=(const ScratchHome&) = delete;

  // The documented HuggingFace layout: {hub}/models--org--repo/snapshots/{commit}.
  fs::path WriteSnapshot(const std::string& folder, const std::string& commit,
                         const std::string& config_json) const {
    const fs::path snapshot =
        dir_ / ".cache" / "huggingface" / "hub" / folder / "snapshots" / commit;
    fs::create_directories(snapshot);
    std::ofstream out(snapshot / "config.json");
    out << config_json;
    out.close();
    return snapshot;
  }

 private:
  fs::path dir_;
  std::string saved_;
};

EngineParams DsparkParams(const std::string& draft_path, std::optional<int> k) {
  EngineParams params;
  SpeculativeConfig cli;
  cli.method = "dspark";
  cli.draft_model_path = draft_path;
  cli.num_speculative_tokens = k;
  params.speculative_config = cli;
  return params;
}

}  // namespace

TEST_CASE("the loader reads a repository identifier through the HuggingFace cache") {
  const ScratchHome home;
  const std::string repo_id = "z-lab/Qwen3.6-27B-DFlash";
  home.WriteSnapshot("models--z-lab--Qwen3.6-27B-DFlash",
                     "0123456789abcdef0123456789abcdef01234567", kQwen3Block7);

  // k=6 against a block-7 draft. The refusal can only happen if the loader
  // turned "z-lab/Qwen3.6-27B-DFlash" into the snapshot directory and read the
  // config.json inside it.
  const EngineParams params = DsparkParams(repo_id, 6);
  CHECK_THROWS_AS(LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
                  std::invalid_argument);
}

TEST_CASE("the same identifier with an empty cache resolves to nothing") {
  // The other half of the A/B. Without the snapshot the identical call is
  // accepted, so the case above is measuring the cache walk and not some other
  // property of the string.
  const ScratchHome home;
  const EngineParams params = DsparkParams("z-lab/Qwen3.6-27B-DFlash", 6);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 6);
}

TEST_CASE("a cached snapshot with no config.json is not a resolution") {
  // The walk accepts a snapshot only when it holds a config.json, so an
  // interrupted download does not become the answer.
  const ScratchHome home;
  const fs::path hub_repo = fs::path(std::getenv("HOME")) / ".cache" /
                            "huggingface" / "hub" /
                            "models--z-lab--Qwen3.6-27B-DFlash";
  fs::create_directories(hub_repo / "snapshots" / "abc123");
  const EngineParams params = DsparkParams("z-lab/Qwen3.6-27B-DFlash", 6);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 6);
}
