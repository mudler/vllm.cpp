// DeepSeek-V4-Flash (`DeepseekV4ForCausalLM`) W1/W2 SCAFFOLDING gate. Proves the
// two things this pass can prove WITHOUT a checkpoint or a GPU:
//   (1) the arch RESOLVES through the registry (the additive TU registered it), and
//   (2) the config DESCENDS: ParseDeepseekV4Params reads the shipped
//       nvidia/DeepSeek-V4-Flash-NVFP4 config.json scalars and validates them.
// The forward + loader materialization + strict gate are NAMED W3-W8 residuals
// (see .agents/specs/deepseek-v4-flash.md §5); nothing here claims the model runs.
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_probe.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/v1/kv_cache_interface.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using vllm::DeepseekV4Params;
using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::ParseDeepseekV4Params;

namespace {
// The shipped nvidia/DeepSeek-V4-Flash-NVFP4 config.json (VERIFIED 2026-07-28),
// reduced to the scalars the parse consumes. Typed HfConfig fields are set from
// the same values; the V4-specific keys live in `raw`.
HfConfig RealConfig() {
  HfConfig c;
  c.architectures = {"DeepseekV4ForCausalLM"};
  c.hidden_size = 4096;
  c.num_hidden_layers = 43;
  c.vocab_size = 129280;
  c.num_attention_heads = 64;
  c.num_key_value_heads = 1;
  c.head_dim = 512;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 1048576;
  nlohmann::json cr = nlohmann::json::array();
  for (int i = 0; i < 44; ++i) {  // 44-entry array (last entry is the MTP layer)
    if (i == 0 || i == 1 || i == 43)
      cr.push_back(0);
    else
      cr.push_back((i % 2 == 0) ? 4 : 128);
  }
  c.raw = {
      {"hidden_size", 4096},        {"num_hidden_layers", 43},
      {"vocab_size", 129280},       {"num_attention_heads", 64},
      {"num_key_value_heads", 1},   {"head_dim", 512},
      {"qk_rope_head_dim", 64},     {"q_lora_rank", 1024},
      {"o_lora_rank", 1024},        {"o_groups", 8},
      {"sliding_window", 128},      {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 1048576},
      {"num_nextn_predict_layers", 1},
      {"n_routed_experts", 256},    {"num_experts_per_tok", 6},
      {"moe_intermediate_size", 2048}, {"n_shared_experts", 1},
      {"norm_topk_prob", true},     {"routed_scaling_factor", 1.5},
      {"swiglu_limit", 10.0},       {"scoring_func", "sqrtsoftplus"},
      {"topk_method", "noaux_tc"},  {"num_hash_layers", 3},
      {"expert_dtype", "fp4"},      {"hc_mult", 4},
      {"hc_sinkhorn_iters", 20},    {"hc_eps", 1e-6},
      {"index_head_dim", 128},      {"index_n_heads", 64},
      {"index_topk", 512},          {"compress_rope_theta", 160000},
      {"rope_theta", 10000},        {"tie_word_embeddings", false},
      {"compress_ratios", cr},
  };
  return c;
}

// THE PRODUCTION SEAM. An engine never calls `vllm::MakeDeepseekV4KVCache`. It
// reaches this topology as `LoadedEngine` -> `MakeKVCacheResolved` ->
// `MakeKVCacheMaybeSpec` (`entrypoints/model_loader.cpp:1394-1404`) ->
// `ModelRegistry::MakeKVCache` (`model_executor/models/model_registry.cpp:381-386`),
// which dereferences `registration().factory->make_kv_cache` — the pointer
// `kDeepseekV4Factory` sets at `deepseek_v4_registry.cpp:123`. Calling the free
// function proves the function builds seven groups; it does NOT prove that
// pointer still names it, and a merge resolution or a W3 refactor that repoints
// it would put the one 576-byte `"mla"` placeholder back with every gate green.
// So the topology gates below enter through the pointer, not around it.
vllm::v1::KVCacheConfig RegistryKVCache(int block_size, int num_blocks) {
  const HfConfig cfg = RealConfig();
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);
  return reg.factory->make_kv_cache(cfg, block_size, num_blocks);
}
}  // namespace

TEST_CASE("deepseek-v4 scaffold: DeepseekV4ForCausalLM RESOLVES through the registry") {
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  const auto has = [&](std::string_view a) {
    for (std::string_view s : supported)
      if (s == a) return true;
    return false;
  };
  CHECK(has("DeepseekV4ForCausalLM"));

  HfConfig cfg;
  cfg.architectures = {"DeepseekV4ForCausalLM"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);
  CHECK(reg.architecture == "DeepseekV4ForCausalLM");
  CHECK(reg.info.is_text_generation_model);
  CHECK_FALSE(reg.info.supports_multimodal);
}

TEST_CASE("deepseek-v4 expert probe input stays in the float domain") {
  for (const float frequency : {0.017f, 0.013f}) {
    const std::vector<float> actual =
        vllm::detail::DeepseekV4ExpertProbeInput(10, frequency);
    REQUIRE(actual.size() == 10);
    for (int64_t i = 0; i < static_cast<int64_t>(actual.size()); ++i) {
      const float expected =
          0.5f * std::sin(frequency * static_cast<float>(i + 1));
      CAPTURE(frequency);
      CAPTURE(i);
      CHECK(actual[static_cast<size_t>(i)] == expected);
    }
  }
}

TEST_CASE("deepseek-v4 scaffold: config DESCENDS (ParseDeepseekV4Params)") {
  const DeepseekV4Params p = ParseDeepseekV4Params(RealConfig());
  // shared geometry
  CHECK(p.hidden_size == 4096);
  CHECK(p.num_hidden_layers == 43);
  CHECK(p.vocab_size == 129280);
  CHECK(p.num_attention_heads == 64);
  CHECK(p.num_key_value_heads == 1);
  CHECK(p.num_nextn_predict_layers == 1);
  // 512-wide MLA (NEW geometry)
  CHECK(p.head_dim == 512);
  CHECK(p.qk_rope_head_dim == 64);
  CHECK(p.q_lora_rank == 1024);
  CHECK(p.o_lora_rank == 1024);
  CHECK(p.o_groups == 8);
  CHECK(p.sliding_window == 128);
  // MoE
  CHECK(p.n_routed_experts == 256);
  CHECK(p.num_experts_per_tok == 6);
  CHECK(p.moe_intermediate_size == 2048);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.num_hash_layers == 3);
  CHECK(p.scoring_func == "sqrtsoftplus");
  CHECK(p.expert_dtype == "fp4");
  CHECK(p.swiglu_limit == doctest::Approx(10.0));
  CHECK(p.routed_scaling_factor == doctest::Approx(1.5));
  // MHC
  CHECK(p.hc_mult == 4);
  CHECK(p.hc_sinkhorn_iters == 20);
  // DSA
  CHECK(p.index_head_dim == 128);
  CHECK(p.index_n_heads == 64);
  CHECK(p.index_topk == 512);
  CHECK(static_cast<int>(p.compress_ratios.size()) == 44);
}

TEST_CASE("deepseek-v4 scaffold: per-layer topology matches the verified schema") {
  const DeepseekV4Params p = ParseDeepseekV4Params(RealConfig());
  // Hash-routed: exactly layers 0,1,2 (num_hash_layers=3).
  CHECK(p.is_hash_layer(0));
  CHECK(p.is_hash_layer(2));
  CHECK_FALSE(p.is_hash_layer(3));
  // Compressor present on layers 2..42 (compress_ratio != 0) == 41 layers;
  // layers 0,1 have ratio 0. Indexer (ratio == 4) on 21 layers.
  int compressor = 0, indexer = 0;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    if (p.has_compressor(l)) ++compressor;
    if (p.has_indexer(l)) ++indexer;
  }
  CHECK(compressor == 41);
  CHECK(indexer == 21);
  CHECK_FALSE(p.has_compressor(0));
  CHECK_FALSE(p.has_compressor(1));
}

TEST_CASE("deepseek-v4 scaffold: parse REJECTS an unrepresentable config") {
  HfConfig bad = RealConfig();
  bad.head_dim = 128;             // not the 512-wide MLA geometry
  bad.raw["head_dim"] = 128;
  CHECK_THROWS_AS(ParseDeepseekV4Params(bad), std::runtime_error);

  HfConfig bad2 = RealConfig();
  bad2.raw["scoring_func"] = "sigmoid";  // V4 is sqrtsoftplus-only
  CHECK_THROWS_AS(ParseDeepseekV4Params(bad2), std::runtime_error);
}

// ─── KV-DSV4-MULTICACHE W2 (#1973) — the published cache topology ────────────
//
// G1, the topology gate of `.agents/specs/kv-dsv4-multicache.md`, on the half
// that needs no checkpoint, no GPU and no forward: upstream's cache topology is
// a pure function of the config, so the whole spec set can be compared exactly.
// Every expectation below was derived from an upstream CONSTRUCTION SITE at the
// pin 5559679229bc961848b121ccdeaa8fa5d79bec98, not from this test's own
// arithmetic, and the page sizes are stated as LITERALS so a change to any of
// them is a red test rather than a silently different pool.
//
// NOTHING CONSUMES THIS. The runner refuses these groups by name (#1973); the
// wiring is owed to W3 and the consumption to W5, both under #1925.
TEST_CASE("deepseek-v4 kv-cache: the published topology is upstream's 167 entries") {
  // 256 is not a guess. `sparse_swa.py:76-83` and `compressor.py:174-178` both
  // derive the geometry from `[256//4, head_dim] = [64, head_dim]`, and the SWA
  // block size of 64 and the compressor block sizes of 4 and 8 only hold there.
  const HfConfig cfg = RealConfig();
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->make_kv_cache != nullptr);
  // Not `&MakeDeepseekV4KVCache` by name: the point is the value the loader
  // will dereference, so the gate dereferences the same one.
  const vllm::v1::KVCacheConfig kv =
      RegistryKVCache(/*block_size=*/256, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 7);
  CHECK(kv.num_blocks == 8);

  size_t entries = 0;
  for (const auto& g : kv.kv_cache_groups) entries += g.layer_names.size();
  CHECK(entries == 167);  // 21 + 20 + 21 + 43 + 21 + 21 + 20

  using vllm::v1::KVCacheSpecKind;
  using vllm::v1::KVQuantMode;
  using vllm::v1::MLAAttentionSpec;
  using vllm::v1::SlidingWindowMLASpec;

  // (a) The compressed MLA latent, C4A. `attention.py:631-645`: block_size from
  // the cache config, num_kv_heads 1, head_size = self.head_dim (512 — NOT
  // head_dim + rope), uint8, compress_ratio 4, cache_dtype_str "fp8_ds_mla",
  // alignment 576, model_version "deepseek_v4", kv_quant_mode FP8_PER_TENSOR.
  {
    const auto& g = kv.kv_cache_groups[0];
    const auto* sp = dynamic_cast<const MLAAttentionSpec*>(g.kv_cache_spec.get());
    REQUIRE(sp != nullptr);
    CHECK(sp->kind() == KVCacheSpecKind::kMlaAttention);
    CHECK(sp->block_size == 256);
    CHECK(sp->storage_block_size() == 64);  // 256 / 4
    CHECK(sp->num_kv_heads == 1);
    CHECK(sp->head_size == 512);
    CHECK(sp->dtype == vt::DType::kI8);
    CHECK(sp->compress_ratio == 4);
    CHECK(sp->alignment == std::optional<int>(576));
    CHECK(sp->cache_dtype_str == std::optional<std::string>("fp8_ds_mla"));
    CHECK(sp->model_version == std::optional<std::string>("deepseek_v4"));
    // The branch order W1 landed, on the real geometry: this spec carries a
    // NON-NONE quant mode AND cache_dtype_str "fp8_ds_mla", so the 584-byte
    // branch must be reached before the quant-mode guard throws.
    CHECK(sp->kv_quant_mode == KVQuantMode::kFp8PerTensor);
    CHECK(sp->real_page_size_bytes() == 37376);  // 64 * 584
    CHECK(sp->page_size_bytes() == 37440);       // round_up(37376, 576)
    CHECK(g.layer_names.size() == 21);
    CHECK(g.layer_names.front() == "model.layers.2.attn");
    CHECK(g.layer_names.back() == "model.layers.42.attn");
  }

  // (a) The compressed MLA latent, C128A. Same site at ratio 128.
  {
    const auto& g = kv.kv_cache_groups[1];
    const auto* sp = dynamic_cast<const MLAAttentionSpec*>(g.kv_cache_spec.get());
    REQUIRE(sp != nullptr);
    CHECK(sp->block_size == 256);
    CHECK(sp->storage_block_size() == 2);  // 256 / 128
    CHECK(sp->head_size == 512);
    CHECK(sp->compress_ratio == 128);
    CHECK(sp->real_page_size_bytes() == 1168);  // 2 * 584
    CHECK(sp->page_size_bytes() == 1728);       // round_up(1168, 576)
    CHECK(g.layer_names.size() == 20);
    CHECK(g.layer_names.front() == "model.layers.3.attn");
    CHECK(g.layer_names.back() == "model.layers.41.attn");
  }

  // (b) The indexer key cache. `attention.py:669-684` passes NO cache_dtype_str
  // and NO model_version, so it takes the ELEMENT formula and not the 584-byte
  // branch, and it defaults kv_quant_mode to NONE. Its width is byte-derived:
  // head_dim + head_dim // quant_block_size * 4 = 128 + 128 // 128 * 4 = 132
  // (`:738`, `:756-759`); the MXFP4 68-byte arm needs
  // `use_fp4_indexer_cache`, whose default is False (`config/attention.py:64`).
  {
    const auto& g = kv.kv_cache_groups[2];
    const auto* sp = dynamic_cast<const MLAAttentionSpec*>(g.kv_cache_spec.get());
    REQUIRE(sp != nullptr);
    CHECK(sp->block_size == 256);
    CHECK(sp->storage_block_size() == 64);  // 256 / 4
    CHECK(sp->head_size == 132);
    CHECK(sp->dtype == vt::DType::kI8);
    CHECK(sp->compress_ratio == 4);
    CHECK(sp->alignment == std::optional<int>(576));
    CHECK_FALSE(sp->cache_dtype_str.has_value());
    CHECK_FALSE(sp->model_version.has_value());
    CHECK(sp->kv_quant_mode == KVQuantMode::kNone);
    CHECK(sp->real_page_size_bytes() == 8448);  // 64 * 1 * 132 * 1
    CHECK(sp->page_size_bytes() == 8640);       // round_up(8448, 576)
    CHECK(g.layer_names.size() == 21);
    CHECK(g.layer_names.front() == "model.layers.2.attn.indexer.k_cache");
  }

  // (c) The sliding-window cache, on ALL 43 attention layers — including
  // layers 0 and 1, which have NO MLA cache at all. `sparse_swa.py:86-101`.
  {
    const auto& g = kv.kv_cache_groups[3];
    const auto* sp =
        dynamic_cast<const SlidingWindowMLASpec*>(g.kv_cache_spec.get());
    REQUIRE(sp != nullptr);
    CHECK(sp->kind() == KVCacheSpecKind::kSlidingWindowMla);
    CHECK(sp->block_size == 64);
    CHECK(sp->storage_block_size() == 64);  // compress_ratio 1
    CHECK(sp->num_kv_heads == 1);
    CHECK(sp->head_size == 512);
    CHECK(sp->dtype == vt::DType::kI8);
    CHECK(sp->sliding_window == std::optional<int>(128));
    CHECK(sp->alignment == std::optional<int>(576));
    CHECK(sp->cache_dtype_str == std::optional<std::string>("fp8_ds_mla"));
    CHECK(sp->model_version == std::optional<std::string>("deepseek_v4"));
    CHECK(sp->kv_quant_mode == KVQuantMode::kFp8PerTensor);
    CHECK(sp->real_page_size_bytes() == 37376);  // 64 * 584
    CHECK(sp->page_size_bytes() == 37440);
    CHECK(g.layer_names.size() == 43);
    CHECK(g.layer_names.front() == "model.layers.0.attn.swa_cache");
    CHECK(g.layer_names.back() == "model.layers.42.attn.swa_cache");
  }

  // (d) The compressor states. `compressor.py:168-200`: f32, state_dim =
  // 2 * coff * head_dim with coff = 1 + (ratio == 4), sliding_window =
  // coff * ratio, block_size 4 at ratio 4 and 8 at ratio 128. NO
  // cache_dtype_str and NO model_version, so the element formula again.
  {
    const auto& g = kv.kv_cache_groups[4];  // attention compressor, ratio 4
    const auto* sp =
        dynamic_cast<const SlidingWindowMLASpec*>(g.kv_cache_spec.get());
    REQUIRE(sp != nullptr);
    CHECK(sp->block_size == 4);
    CHECK(sp->storage_block_size() == 4);
    CHECK(sp->head_size == 2048);  // 2 * 2 * 512
    CHECK(sp->dtype == vt::DType::kF32);
    CHECK(sp->sliding_window == std::optional<int>(8));  // coff * ratio
    CHECK_FALSE(sp->cache_dtype_str.has_value());
    CHECK_FALSE(sp->model_version.has_value());
    CHECK(sp->real_page_size_bytes() == 32768);  // 4 * 1 * 2048 * 4
    CHECK(sp->page_size_bytes() == 32832);       // round_up(32768, 576)
    CHECK(g.layer_names.size() == 21);
    CHECK(g.layer_names.front() ==
          "model.layers.2.attn.compressor.state_cache");
  }
  {
    const auto& g = kv.kv_cache_groups[5];  // the INDEXER's own compressor
    const auto* sp =
        dynamic_cast<const SlidingWindowMLASpec*>(g.kv_cache_spec.get());
    REQUIRE(sp != nullptr);
    CHECK(sp->block_size == 4);
    CHECK(sp->head_size == 512);  // 2 * 2 * 128, built at head_dim=128
    CHECK(sp->dtype == vt::DType::kF32);
    CHECK(sp->sliding_window == std::optional<int>(8));
    CHECK(sp->real_page_size_bytes() == 8192);  // 4 * 1 * 512 * 4
    CHECK(sp->page_size_bytes() == 8640);       // round_up(8192, 576)
    CHECK(g.layer_names.size() == 21);
    CHECK(g.layer_names.front() ==
          "model.layers.2.attn.indexer.compressor.state_cache");
  }
  {
    const auto& g = kv.kv_cache_groups[6];  // attention compressor, ratio 128
    const auto* sp =
        dynamic_cast<const SlidingWindowMLASpec*>(g.kv_cache_spec.get());
    REQUIRE(sp != nullptr);
    CHECK(sp->block_size == 8);
    CHECK(sp->head_size == 1024);  // 2 * 1 * 512
    CHECK(sp->dtype == vt::DType::kF32);
    CHECK(sp->sliding_window == std::optional<int>(128));  // coff * ratio
    CHECK(sp->real_page_size_bytes() == 32768);  // 8 * 1 * 1024 * 4
    CHECK(sp->page_size_bytes() == 32832);
    CHECK(g.layer_names.size() == 20);
    CHECK(g.layer_names.front() ==
          "model.layers.3.attn.compressor.state_cache");
  }

  // THE PORT'S OWN SELF-CHECK. Upstream fixes the SWA block size at 64
  // *because* the SWA and C4A blocks share one physical tensor and must
  // therefore have one page size (`sparse_swa.py:76-83`). Our two classes reach
  // that page by different routes — an MLAAttentionSpec at block_size 256 /
  // compress_ratio 4 and a SlidingWindowMLASpec at block_size 64 /
  // compress_ratio 1 — so the equality holds only if both storage_block_size
  // overrides, both 584-byte branches and the shared 576-byte padding are all
  // right at once. Asserted directly, not just as two literals.
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->page_size_bytes() ==
        kv.kv_cache_groups[3].kv_cache_spec->page_size_bytes());
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->storage_block_size() ==
        kv.kv_cache_groups[3].kv_cache_spec->storage_block_size());
}

// The published names carry `.attn.`, NOT the `.self_attn.` every other
// architecture in this tree uses: `DeepseekV4DecoderLayer` builds its attention
// as `prefix=f"{prefix}.attn"` (`vllm/models/deepseek_v4/nvidia/model.py:808-813`)
// under `prefix=f"{prefix}.layers"` (`:1015`) and `maybe_prefix(prefix, "model")`
// (`:1409`). W3 resolves these back to layer indices, and a name that does not
// parse makes GroupLayerMask fall back wholesale rather than fail, so the
// spelling is gated here where it is still cheap.
TEST_CASE("deepseek-v4 kv-cache: names are the upstream module paths") {
  const vllm::v1::KVCacheConfig kv = RegistryKVCache(256, 8);
  for (const auto& g : kv.kv_cache_groups) {
    for (const std::string& n : g.layer_names) {
      CHECK(n.rfind("model.layers.", 0) == 0);
      CHECK(n.find(".attn") != std::string::npos);
      CHECK(n.find(".self_attn") == std::string::npos);
    }
    // Within one group every name resolves to a DISTINCT layer index, which is
    // GroupLayerMask's all-or-nothing precondition (`runner.cpp`).
    std::vector<std::string> idx;
    for (const std::string& n : g.layer_names) {
      const size_t at = n.find(".layers.") + 8;
      idx.push_back(n.substr(at, n.find('.', at) - at));
    }
    std::vector<std::string> uniq = idx;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    CHECK(uniq.size() == idx.size());
  }
}

// `storage_block_size` is `block_size / compress_ratio`, so a block size below
// the ratio produces a ZERO-byte page rather than a refusal. Upstream has no
// such check because its comments derive the geometry at 256; ours refuses.
TEST_CASE("deepseek-v4 kv-cache: a block_size that cannot hold a C128A row is refused") {
  CHECK_THROWS_AS(RegistryKVCache(16, 8), std::runtime_error);
  CHECK_THROWS_AS(RegistryKVCache(192, 8), std::runtime_error);
  // 32 is `EngineParams::block_size`'s default (`entrypoints/model_loader.h:120`,
  // and the `params.block_size > 0 ? params.block_size : 32` fallback in the
  // `LoadedEngine` ctor). `has_c128` is true for Flash, so `32 % 128 != 0` and
  // THIS refusal is what a default-configured engine hits — inside the
  // `kv_cfg_` member initializer, which precedes `runner_` in the ctor's
  // initializer list. The runner's own by-name group refusal (#1973) is
  // therefore reachable for V4 only at `--block-size` 128 or 256, and the
  // message a default run reads names the block size, not the topology.
  CHECK_THROWS_AS(RegistryKVCache(32, 8), std::runtime_error);
  // 128 divides 128 exactly and gives storage_block_size 1: representable.
  CHECK_NOTHROW(RegistryKVCache(128, 8));
}

// A LoadedModel that carries a registration and nothing else. The engine funnel
// reads only `registration()` on the way to the KV geometry, so this is enough
// to enter it without a checkpoint.
namespace {
class ScaffoldModel final : public vllm::LoadedModel {
 public:
  explicit ScaffoldModel(const vllm::ModelRegistration& r) : LoadedModel(r) {}
};
}  // namespace

// ---------------------------------------------------------------------------
// THE DEFAULT CONFIGURATION. `AGENTS.md` measures reachability on the entry
// point's DEFAULT configuration, and on 2026-08-31 the real 97.68 GiB artifact
// did not load on one: `EngineParams::block_size` is 32, a `compress_ratio`-128
// layer needs 256, and `MakeDeepseekV4KVCache` refused by name. `vllm-server`
// takes `--block-size`; `vllm-cli` does not, so the model was reachable only by
// an operator who guessed a number. Upstream does not ask -- it DERIVES the
// geometry from the model (`sparse_swa.py:76-83`, `compressor.py:174-178`), and
// `kv_block_size_floor` is that derivation.
TEST_CASE("deepseek-v4 pages at the engine's DEFAULT block size") {
  const HfConfig cfg = RealConfig();
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);

  // READ the default, never retype it: what is gated is the number the engine
  // actually starts from, so a change to that field must reach this test.
  const int requested = vllm::entrypoints::EngineParams{}.block_size;

  // The raw default must STAY unbuildable. If this stops throwing the geometry
  // changed underneath, and the resolution below would be gating nothing.
  CHECK_THROWS(reg.factory->make_kv_cache(cfg, requested, 64));

  // THE ENGINE FUNNEL, not the resolver beneath it: `LoadedEngine` ctor ->
  // MakeKVCacheResolved -> MakeKVCacheMaybeSpec. Entering at
  // `ResolveKVBlockSize` would prove the arithmetic and leave the loader free to
  // stop calling it -- measured on 2026-08-31, deleting that call site left this
  // file, test_loaded_engine_dense and test_kv_cache_fp8_wiring all green.
  ScaffoldModel model(reg);
  vllm::v1::KVCacheConfig kv;
  REQUIRE_NOTHROW(kv = vllm::entrypoints::LoadedEngine::MakeKVCacheMaybeSpec(
                      model, cfg, requested, 64, std::nullopt));
  CHECK_FALSE(kv.kv_cache_groups.empty());

  // And the page the ratio-128 group publishes holds at least one token, which
  // is the property the refusal exists to protect.
  const int resolved = ModelRegistry::ResolveKVBlockSize(reg, requested);
  CHECK(resolved > requested);
  CHECK(resolved % 128 == 0);
  CHECK(resolved / 128 >= 1);
}

// An architecture that declares NO floor keeps exactly what the caller asked
// for. Without this the floor could be a blanket rewrite of every model's block
// size and the case above would still pass.
TEST_CASE("a model with no declared floor keeps the requested block size") {
  HfConfig other;
  other.architectures = {"Qwen3ForCausalLM"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(other);
  REQUIRE(reg.factory->kv_block_size_floor == 0);
  for (int bs : {1, 16, 32, 256, 512})
    CHECK(ModelRegistry::ResolveKVBlockSize(reg, bs) == bs);
}
