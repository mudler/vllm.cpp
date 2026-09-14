// KV-DSV4-MULTICACHE W8 slice 4 + slice 6 (#2455) — the fp8_ds_mla BRIDGE.
//
// THE WALL THIS FILE EXISTS TO REMOVE, stated as the disagreement it is.
// `MakeDeepseekV4KVCache` publishes DeepSeek-V4's compressed-latent and SWA
// groups as `MLAAttentionSpec`/`SlidingWindowMLASpec` at `vt::DType::kI8` with
// `cache_dtype_str == "fp8_ds_mla"`, mirroring upstream, where
// `use_fp8_ds_mla_layout` is `ClassVar[bool] = True` (`attention.py:140`) and
// `_resolve_dsv4_kv_cache_dtype` writes `cache_config.cache_dtype =
// "fp8_ds_mla"` BACK onto the cache config and returns `torch.uint8`
// (`attention.py:89-119`). `ApplyCacheDType`'s early-out then needs
// `spec.dtype == resolved.storage`; `--kv-cache-dtype auto` resolves storage to
// the MODEL dtype (bf16), `kI8 != kBF16`, so the early-out misses and
// `RetypeAttentionSpec` refuses every MLA spec — at engine construction, with no
// flag passed (#2455, measured on the real 97.68 GiB artifact).
//
// SLICE 6 IS THE FIX, AND IT IS A CHANGE TO RESOLUTION, NOT A WIDER GUARD.
// Upstream's `auto` for this architecture RESOLVES TO fp8_ds_mla, because the
// model's own `_resolve_dsv4_kv_cache_dtype` overrides the cache config. So
// `auto` here means "the dtype the model's factory published stands", and the
// MLA guard in `RetypeAttentionSpec` keeps firing for every EXPLICIT override.
// The last case below is the over-fire control that pins exactly that.
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using vllm::HfConfig;
using vllm::ModelRegistry;

namespace {

// The shipped nvidia/DeepSeek-V4-Flash config.json scalars, reduced to what the
// KV factory consumes. Same values as `test_deepseek_v4_scaffold.cpp`, which
// pins the resulting topology entry by entry; this file is about what
// `ApplyCacheDType` then does to it.
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
  for (int i = 0; i < 44; ++i) {
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

// THE PRODUCTION SEAM, for the same reason the scaffold gate uses it: an engine
// reaches this topology through `registration().factory->make_kv_cache`, never
// through the free function, so a repoint that put a placeholder back would keep
// every gate here green if this called the free function instead.
vllm::v1::KVCacheConfig RegistryKVCache(int block_size, int num_blocks) {
  const HfConfig cfg = RealConfig();
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);
  return reg.factory->make_kv_cache(cfg, block_size, num_blocks);
}

}  // namespace

// ─── The load-blocking wall (#2455) ──────────────────────────────────────────
TEST_CASE("W8: a DeepSeek-V4 engine CONSTRUCTS on --kv-cache-dtype auto") {
  vllm::v1::KVCacheConfig cfg = RegistryKVCache(/*block_size=*/256,
                                                /*num_blocks=*/8);
  // Seven groups, so this is the real topology and not a degenerate one.
  REQUIRE(cfg.kv_cache_groups.size() == 7);

  // THE DEFAULT PATH. `vllm-cli` has no `--kv-cache-dtype` flag; this is the
  // string an operator gets without asking for anything.
  CHECK_NOTHROW(vllm::v1::ApplyCacheDType(
      cfg, vllm::v1::ParseCacheDType("auto", vt::DType::kBF16), 1.0F, 1.0F));

  // And `auto` left the model's own published layout ALONE. A retype to bf16
  // would be the 3.5x overrun the whole slice exists to stop: 2048 f32/bf16
  // bytes per token against the 584 the spec declares.
  const auto* latent = dynamic_cast<const vllm::v1::MLAAttentionSpec*>(
      cfg.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(latent != nullptr);
  CHECK(latent->dtype == vt::DType::kI8);
  CHECK(latent->cache_dtype_str == std::optional<std::string>("fp8_ds_mla"));
  CHECK(latent->real_page_size_bytes() == 37376);  // 64 * 584, unchanged

  const auto* swa = dynamic_cast<const vllm::v1::SlidingWindowMLASpec*>(
      cfg.kv_cache_groups[3].kv_cache_spec.get());
  REQUIRE(swa != nullptr);
  CHECK(swa->dtype == vt::DType::kI8);
  CHECK(swa->cache_dtype_str == std::optional<std::string>("fp8_ds_mla"));

  // THE TWO SHAPES A `cache_dtype_str` PREDICATE MISSES, asserted because each
  // was a separate silent defect and a gate that checked only the fp8_ds_mla
  // groups stayed green through both.
  //
  // The indexer key cache is `kI8` with NO `cache_dtype_str` — upstream passes
  // none (`attention.py:669-684`) and its width is byte-derived (132 = 128 +
  // 128/128*4). Retyped to bf16 it would double a page sized in bytes.
  const auto* indexer = dynamic_cast<const vllm::v1::MLAAttentionSpec*>(
      cfg.kv_cache_groups[2].kv_cache_spec.get());
  REQUIRE(indexer != nullptr);
  CHECK_FALSE(indexer->cache_dtype_str.has_value());
  CHECK(indexer->dtype == vt::DType::kI8);
  CHECK(indexer->head_size == 132);
  CHECK(indexer->real_page_size_bytes() == 8448);  // 64 * 1 * 132 * 1

  // The three compressor state caches are f32, which upstream ASSERTS
  // (`compressor.py:168-200`). These are the ones the MLA guard never sees:
  // `SlidingWindowMLASpec` derives from `SlidingWindowSpec`, so a retype takes
  // the float branch, where `storage == kBF16` PASSES on a bf16 model and
  // halves the page in silence.
  for (size_t g : {size_t{4}, size_t{5}, size_t{6}}) {
    CAPTURE(g);
    const auto* state = dynamic_cast<const vllm::v1::SlidingWindowMLASpec*>(
        cfg.kv_cache_groups[g].kv_cache_spec.get());
    REQUIRE(state != nullptr);
    CHECK_FALSE(state->cache_dtype_str.has_value());
    CHECK(state->dtype == vt::DType::kF32);
  }
}

// ─── The over-fire control: the guard is NOT widened ─────────────────────────
TEST_CASE("W8: an EXPLICIT --kv-cache-dtype fp8 is still REFUSED by name") {
  // Slice 6 changes what `auto` MEANS, and nothing else. An operator who asks
  // for a different page format on an MLA cache must still be refused, because
  // the fp8_ds_mla page formula is the model's, not a dtype the flag selects.
  // This is the mutation that separates "resolution honours the factory" from
  // "the MLA guard was widened to make the load pass".
  vllm::v1::KVCacheConfig cfg = RegistryKVCache(256, 8);
  try {
    vllm::v1::ApplyCacheDType(
        cfg, vllm::v1::ParseCacheDType("fp8", vt::DType::kBF16), 1.0F, 1.0F);
    FAIL("ApplyCacheDType accepted an explicit fp8 on an MLA topology");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    CHECK(msg.find("MLA") != std::string::npos);
    CHECK(msg.find("fp8_ds_mla") != std::string::npos);
  }
}
