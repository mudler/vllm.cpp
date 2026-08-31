// `ENGINE-HYBRID-PLACEMENT` (issue #2314) — REACHABILITY of the MoE placement
// plan, not its resolution. `test_device_placement.cpp` proves the plan resolves
// and that the seam reads it; this file proves a load a user can actually arrive
// through INSTALLS one.
//
// WHY THIS FILE EXISTS SEPARATELY, and what it would have caught. W1-W3 landed
// the config, the resolver, the per-layer plan, the `RunMoePlaced` seam and five
// architectures on it, all green. `SetActiveMoePlacementPlan` was nevertheless
// called by NOTHING in `src/`: the seam read a process-global that no production
// path ever wrote, so `PlacesAnything()` was false on every load and no expert
// was ever placed on the CPU.
//
// Two things hid it, and both are the reason this test asserts what it asserts:
//
//   * the loader PRINTS `engine: device placement: N layers on cpu` from the
//     RESOLVED plan, so the one signal an operator would check confirmed a
//     feature that was not running;
//   * a token gate cannot see it — with nothing placed, the placed arm is
//     byte-identical to the unplaced arm, so an end-to-end token comparison
//     PASSES for the wrong reason.
//
// So this suite never compares tokens and never constructs a plan. It enters
// through `LoadedEngine::FromModelDir` — what the server and the C ABI both call
// — and asks the global what the loader put there.
//
// WHAT IT ASSERTS, and why it is `resolved_layer_count`. In a CPU-only build the
// engine device IS the placement target, so an installed plan and a
// never-installed one agree on `PlacesAnything()`, `placed_layer_count()` and
// `DeviceForLayer()` — all inert, correctly. The layer count the plan was
// resolved against is the one observable that separates them, and it is also the
// substance of the install: a plan resolved against THIS model's depth.
//
// THE MODEL DIRECTORY HAS A CONFIG AND NO WEIGHTS. Unlike the residency install,
// this one sits after the config parse — it needs `num_hidden_layers`, and it
// must still precede all weight I/O, because `ResidentWeight` aliases host bytes
// on a CPU `Dev` and uploads otherwise, so placing after the upload would pay the
// round trip the placement exists to avoid. A directory with a valid config and
// no tokenizer or shards therefore fails the load AFTER the install, which is
// what makes the install observable without a real checkpoint.
//
// THE REACHABILITY MUTATION for this row deletes the `InstallMoePlacementPlan`
// call sites in `LoadedEngine::FromModelDir` and requires this suite RED.
#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "vllm/config/weight_residency.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/device_placement.h"

namespace fs = std::filesystem;

namespace {

constexpr int64_t kLayers = 8;

// A wired MoE architecture, carrying exactly the keys `LoadHfConfig` requires.
// No tokenizer and no shards, so the load throws once the install is behind it.
std::string WriteConfigOnlyModelDir(const char* stem) {
  const fs::path dir = fs::temp_directory_path() / stem;
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::ofstream(dir / "config.json") << R"({
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "num_hidden_layers": )" << kLayers << R"(,
    "rms_norm_eps": 1e-6
  })";
  return dir.string();
}

struct PlacementFixture {
  PlacementFixture() {
    vllm::ResetWeightResidencyConfigForTesting();
    vllm::ResetActiveMoePlacementPlanForTesting();
  }
  ~PlacementFixture() {
    vllm::ResetWeightResidencyConfigForTesting();
    vllm::ResetActiveMoePlacementPlanForTesting();
  }
};

}  // namespace

TEST_CASE("placement reach: FromModelDir installs a plan resolved against the model") {
  PlacementFixture fx;
  const std::string dir = WriteConfigOnlyModelDir("vllm-cpp-placement-reach");

  // Never installed: the state every load had before #2314.
  REQUIRE(vllm::ActiveMoePlacementPlan().resolved_layer_count() == 0);

  // What `--offload-config '{"vllm_cpp":{"placement":{"cpu_moe":true}}}'`
  // produces. The test does NOT call SetActiveMoePlacementPlan; the loader must.
  vllm::entrypoints::EngineParams params;
  params.weight_residency = vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"placement":{"cpu_moe":true}}})");

  CHECK_THROWS(vllm::entrypoints::LoadedEngine::FromModelDir(dir, params));

  // The load failed on the absent weights, AFTER the install. A plan resolved
  // against THIS model is the whole content of the fix: before it, the seam read
  // a default the loader never wrote.
  CHECK(vllm::ActiveMoePlacementPlan().resolved_layer_count() == kLayers);

  fs::remove_all(dir);
}

TEST_CASE("placement reach: a load with no placement still installs, and stays inert") {
  PlacementFixture fx;
  const std::string dir = WriteConfigOnlyModelDir("vllm-cpp-placement-reach-bare");

  // THE INERTNESS HALF, and it is not merely the absence of the first case. The
  // plan is a process-global, so a second load in the same process must OVERWRITE
  // the first model's plan rather than inherit it — an install that returned
  // early on "no overrides" would leave a stale placement pointed at the wrong
  // model. So a bare load installs too; it installs a plan that places nothing.
  vllm::SetActiveMoePlacementPlan(vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides(
          {vllm::PlacementOverride{"\\.ffn_(up|down|gate)_exps", "cpu"}},
          vt::DeviceType::kCUDA),
      64));
  REQUIRE(vllm::ActiveMoePlacementPlan().resolved_layer_count() == 64);

  vllm::entrypoints::EngineParams bare;
  CHECK_FALSE(bare.weight_residency.has_value());
  CHECK_THROWS(vllm::entrypoints::LoadedEngine::FromModelDir(dir, bare));

  // Re-resolved against the new model, not the 64-layer leftover.
  CHECK(vllm::ActiveMoePlacementPlan().resolved_layer_count() == kLayers);
  CHECK_FALSE(vllm::ActiveMoePlacementPlan().PlacesAnything());

  fs::remove_all(dir);
}
