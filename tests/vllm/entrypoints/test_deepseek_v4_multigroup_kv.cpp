// MODEL-MM-deepseek-v4 (#2411) — CAN A REAL DeepSeek-V4 CHECKPOINT BE SERVED
// WHEN ITS KV CACHE GROUPS DISAGREE ABOUT THEIR BLOCK SIZE?
//
// `test_serve_deepseek_v4_mm` drives the real `VllmServerMain` and passes, and
// it cannot see this defect: its fixture leaves `attention.compress_ratios` all
// zero, so every layer clamps to ratio 1, no layer carries a compressor or an
// indexer, and `MakeDeepseekV4KVCache` publishes exactly ONE group — the SWA
// cache at the hard-coded 64 tokens (`sparse_swa.py:76-83`). One group takes the
// `UnitaryKVCacheCoordinator`, where the scheduler's hash granularity and the
// group's block size are trivially the same number.
//
// A REAL Flash checkpoint SETS those ratios. Then the factory publishes up to
// seven groups at block sizes 256, 64, 4 and 8, no single granularity is every
// group's block size, and the engine takes the `HybridKVCacheCoordinator`
// instead. That is the shape nothing in this tree exercised, which is exactly
// why it survived.
//
// WHAT UPSTREAM DOES, because this is a mirror and not a design decision.
// `resolve_kv_cache_block_sizes` resolves TWO different quantities from the same
// group set: the scheduler's alignment invariant is their LCM, and the
// prefix-hash granularity is their GCD (`vllm/v1/core/kv_cache_utils.py:678-770`
// @ `e126687a9a`). A group whose block size is a MULTIPLE of that granularity
// then reads its hashes through a converting view rather than refusing:
// `BlockHashListWithBlockSize` takes the last fine hash inside each coarse block,
// which is already chained over that block's whole prefix
// (`kv_cache_utils.py:2358-2464`). Upstream's own coordinator asserts only
// DIVISIBILITY (`kv_cache_coordinator.py:608-613`) — it never requires equality.
// So prefix caching stays ON for such a model, and the engine serves.
//
// This suite enters at `LoadedEngine::FromModelDir`, which is the loader entry
// every server and command line takes for a `.gguf` argument, on its DEFAULT
// configuration. A test that built the coordinator by hand would prove the class
// works and say nothing about whether a checkpoint can be served.
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "deepseek_v4_lang_gguf_fixture.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"

namespace {

// The ratios that give the multi-group topology on the fixture's three layers:
// one ratio-4 layer (compressed latent + indexer key cache + the attention and
// indexer compressor states), one ratio-128 layer (latent + one compressor
// state) and one plain layer that has only the SWA cache. Upstream accepts 1, 4
// and 128 and nothing else (`sparse_swa.py:44-55`); a raw 0 is upstream's own
// "no DSA on this layer" and clamps to 1 (`attention.py:205-212`).
const std::vector<int32_t> kFlashRatios = {4, 128, 0};

// DeepSeek-V4's REAL config, in the shape `MakeDeepseekV4KVCache` reads it. The
// same three-layer ratio vector as the checkpoint above, so the group set this
// asserts is the group set the engine case below builds.
vllm::HfConfig FlashLikeConfig() {
  vllm::HfConfig cfg;
  cfg.architectures = {"DeepseekV4ForCausalLM"};
  cfg.hidden_size = 32;
  cfg.num_hidden_layers = 3;
  cfg.vocab_size = 16;
  cfg.num_attention_heads = 2;
  cfg.num_key_value_heads = 1;
  cfg.head_dim = 512;
  cfg.rms_norm_eps = 1e-6;
  cfg.max_position_embeddings = 4096;
  nlohmann::json ratios = nlohmann::json::array();
  for (const int32_t r : kFlashRatios) ratios.push_back(r);
  cfg.raw = {
      {"hidden_size", 32},          {"num_hidden_layers", 3},
      {"vocab_size", 16},           {"num_attention_heads", 2},
      {"num_key_value_heads", 1},   {"head_dim", 512},
      {"qk_rope_head_dim", 64},     {"q_lora_rank", 32},
      {"o_lora_rank", 32},          {"o_groups", 2},
      {"sliding_window", 128},      {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 4096},
      {"n_routed_experts", 4},      {"num_experts_per_tok", 2},
      {"moe_intermediate_size", 32},{"n_shared_experts", 1},
      {"norm_topk_prob", true},     {"routed_scaling_factor", 1.0},
      {"swiglu_limit", 10.0},       {"scoring_func", "sqrtsoftplus"},
      {"topk_method", "noaux_tc"},  {"num_hash_layers", 1},
      {"expert_dtype", "fp4"},      {"hc_mult", 2},
      {"hc_sinkhorn_iters", 3},     {"hc_eps", 1e-6},
      {"index_head_dim", 32},       {"index_n_heads", 2},
      {"index_topk", 3},            {"compress_rope_theta", 160000},
      {"rope_theta", 10000},        {"tie_word_embeddings", false},
      {"compress_ratios", ratios},
  };
  return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) THE TOPOLOGY, AND THE TWO NUMBERS IT RESOLVES TO.
//
// This is the WHY for the case below, taken through the factory pointer the
// loader dereferences rather than by calling `MakeDeepseekV4KVCache` by name.
// It is not the capability gate: it establishes that the group set really does
// disagree about its block size, and that the resolver answers with a hash
// granularity that is SMALLER than most of those groups — which is the input
// the coordinator then has to be able to accept.
// ---------------------------------------------------------------------------
TEST_CASE("a ratio-bearing DeepSeek-V4 publishes groups that no single block size covers") {
  const vllm::HfConfig cfg = FlashLikeConfig();
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(cfg);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->make_kv_cache != nullptr);

  // 256 is the architecture's own floor (`kv_block_size_floor`), which is what a
  // default-configured engine resolves to.
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(cfg, /*block_size=*/256, /*num_blocks=*/4);

  // Seven groups: C4A latent, C128A latent, indexer key, SWA, and the three
  // compressor-state populations.
  REQUIRE(kv.kv_cache_groups.size() == 7);
  std::vector<int> sizes;
  for (const auto& group : kv.kv_cache_groups) {
    REQUIRE(group.kv_cache_spec != nullptr);
    sizes.push_back(group.kv_cache_spec->block_size);
  }
  CHECK(sizes == std::vector<int>{256, 256, 256, 64, 4, 4, 8});

  // The LCM schedules and the GCD hashes (`kv_cache_utils.py:678-770`). The
  // second number is the one that matters here: it is 4, so FIVE of the seven
  // groups page COARSER than the granularity their hashes are computed at.
  const auto [scheduler_block_size, hash_block_size] =
      vllm::v1::resolve_kv_cache_block_sizes(
          kv, /*cache_block_size=*/4, /*prefix_match_unit=*/std::nullopt,
          /*enable_prefix_caching=*/true, /*connector_enabled=*/false,
          /*dcp_world_size=*/1);
  CHECK(scheduler_block_size == 256);
  CHECK(hash_block_size == 4);
}

// ---------------------------------------------------------------------------
// (2) WHAT ACTUALLY HAPPENS TO SUCH A CHECKPOINT TODAY, which is NOT what the
// group set above would lead a reader to predict.
//
// The obvious prediction is that seven groups at four different block sizes
// reach `HybridKVCacheCoordinator` and die on its LOCAL deferral assert that
// every group's block size EQUALS the hash granularity
// (`kv_cache_coordinator.cpp:386`) — an assert inside a server being an abort.
// MEASURED 2026-09-13, THAT IS NOT WHAT HAPPENS, and the difference is the whole
// reason this case exists.
//
// The engine never gets that far. `ApplyCacheDType` runs while `kv_cfg_` is
// being initialized, which PRECEDES `scheduler_block_size_` and `scheduler_` in
// the `LoadedEngine` constructor's initializer list, and `RetypeAttentionSpec`
// refuses any `MLAAttentionSpec` BY NAME
// (`src/vllm/v1/kv_cache_interface.cpp:398`): the fp8_ds_mla page formula landed
// with no store and no read (#2455, owed to KV-DSV4-MULTICACHE W8). The
// compressed-latent groups are exactly the groups a non-zero `compress_ratio`
// adds, so the ratios that create the multi-group topology are also what trips
// that guard.
//
// That is why the all-zero-ratio fixture in `test_serve_deepseek_v4_mm` serves
// happily: `SlidingWindowMLASpec` derives from `SlidingWindowSpec`, NOT from
// `MLAAttentionSpec`, so a SWA-only publication never meets this guard at all.
//
// SO THE COORDINATOR ASSERT IS LATENT, NOT LIVE. DeepSeek-V4 is the only
// architecture in this tree that publishes groups with differing block sizes —
// every other multi-group registry (glm5_next, kimi_linear, nemotron_h,
// qwen4_exp, qwen3_5_common) hands the same `block_size` variable to every group
// — so no production entry point can reach that assert today. WHOEVER LANDS W8
// MAKES IT REACHABLE, and this case is the tripwire: when the refusal below
// stops firing, the port recorded under `## Owed` in
// `.agents/specs/deepseek-v4-flash-vision.md` has to be in place, or this
// becomes the abort the prediction expected.
//
// This case asserts TODAY'S behaviour deliberately. A refusal that names its
// reason is a message rather than a crash, and that is a supported outcome;
// asserting a future pass here would be a test that stays red for a year and
// teaches nobody why.
// ---------------------------------------------------------------------------
TEST_CASE("serve: a multi-group DeepSeek-V4 checkpoint is refused BY NAME, not aborted") {
  gguf_test::TempFile lang(dsv4_lang_test::BuildDeepseek4Gguf(
      /*vision=*/false, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/true, /*vision_bias_scale=*/1.0f,
      /*sliding_window=*/128, /*hash_layers=*/dsv4_lang_test::kHashLayers,
      /*compress_ratios=*/kFlashRatios));

  vllm::entrypoints::EngineParams params;
  std::string message;
  try {
    std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
        vllm::entrypoints::LoadedEngine::FromModelDir(lang.path(), params);
    // Reaching here means the fp8_ds_mla guard stopped firing — see above. That
    // is W8 landing, and it is the moment the hash-granularity port stops being
    // hypothetical.
    FAIL_CHECK(
        "the fp8_ds_mla refusal did not fire: #2455 / KV-DSV4-MULTICACHE W8 has "
        "landed, so re-read this suite's comment -- the coordinator's "
        "block_size == hash_block_size assert is now REACHABLE");
  } catch (const std::exception& e) {
    message = e.what();
  }

  // It is a refusal that names the layout it cannot serve...
  CHECK(message.find("fp8_ds_mla") != std::string::npos);
  // ...and the issue that owes the store and the read, so a reader is not sent
  // hunting for a `--kv-cache-dtype` flag they never passed.
  CHECK(message.find("2455") != std::string::npos);
  // ...and it is the CACHE DTYPE door, not the coordinator's. If this ever reads
  // as a hash-block-size complaint instead, the refusal order moved and every
  // comment above it is stale.
  CHECK(message.find("hash_block_size") == std::string::npos);
}
