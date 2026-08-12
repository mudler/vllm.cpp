// DeepSeek-V4-Pro SHAPE-GENERALITY gate (issue #504).
//
// `deepseek-ai/DeepSeek-V4-Pro` is `DeepseekV4ForCausalLM` / `model_type:
// deepseek_v4` — the SAME architecture as DeepSeek-V4-Flash, which this project
// already supports. A key-by-key diff of the two shipped `config.json` files has
// ZERO new and ZERO removed keys: every difference is a scaled value. This gate
// proves our config descent is genuinely shape-generic rather than Flash-tuned,
// which is the whole claim behind "no architecture work is owed for Pro".
//
// It proves that WITHOUT a checkpoint, a download, or a GPU — the 805 GiB Pro
// weights do not fit one GB10 (119 GiB), so a load/forward gate is memory-
// infeasible here (see the row spec). Nothing below claims Pro runs.
//
// ─── GROUNDED AGAINST UPSTREAM (pin 555967922, vLLM 0.26.0.dev0) ─────────────
//   OURS                        <-  UPSTREAM (vllm/)
//   ParseDeepseekV4Params       <-  models/deepseek_v4/nvidia/model.py:535-679
//                                   + attention.py:193-211 — every dimension is
//                                   read off `config`; there is NO Pro-specific
//                                   path upstream, the same package serves both.
//   p.has_compressor(l)         <-  attention.py:334 `if self.compress_ratio > 1`
//     (== ratio != 0)               over attention.py:209
//                                   `max(1, config.compress_ratios[layer_id])`.
//                                   Ratio 0 becomes 1 upstream, and 1 > 1 is
//                                   false, so ratio 0 means NO compressor on
//                                   both sides — the polarity matches exactly.
//   p.has_indexer(l)            <-  compressor.py:171 `assert compress_ratio in
//     (== ratio == 4)               [4, 128]` + compressor.py:181 (the ratio-4
//                                   arm is the overlapped Lightning-Indexer one,
//                                   compressor.py:247 `self.overlap = ratio == 4`).
//   index_topk                  <-  attention.py:708 `self.topk_tokens =
//                                   config.index_topk` (512 Flash / 1024 Pro).
//
// ─── GROUNDED AGAINST THE REAL CHECKPOINTS ──────────────────────────────────
// Normalizing layer and expert indices out of both real
// `model.safetensors.index.json` files yields 98 distinct tensor-name patterns on
// each side with ZERO unique to either. Predicting compressor/indexer layers from
// `compress_ratios` alone reproduces both checkpoints exactly:
//     Flash (43L): predicted 41 compressor / 21 indexer  == actual 41 / 21
//     Pro   (61L): predicted 61 compressor / 30 indexer  == actual 61 / 30
// Those two pairs are the load-bearing assertions below. Pro's layers 0-1 carry
// the 4-tensor ratio-128 compressor group where Flash has none, which is the one
// STRUCTURAL difference between the two configs and is asserted directly.
#include "vllm/model_executor/models/deepseek_v4.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

using vllm::DeepseekV4Params;
using vllm::HfConfig;
using vllm::ParseDeepseekV4Params;

namespace {

// The real `compress_ratios` arrays, verbatim from the shipped configs
// (fetched 2026-08-12). Both are `num_hidden_layers + 1` long: the trailing
// entry is the MTP/nextn layer, which carries no compressor.
const std::vector<int64_t>& ProCompressRatios() {
  static const std::vector<int64_t> v = {
      128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
      4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
      128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0};
  return v;
}

const std::vector<int64_t>& FlashCompressRatios() {
  static const std::vector<int64_t> v = {
      0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
      128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
      128, 4, 0};
  return v;
}

// One config builder parameterized by the values that DIFFER between the two
// checkpoints. Everything passed positionally here is a key whose value the
// Flash/Pro diff showed changing; every key hardcoded in the body is one the
// diff showed IDENTICAL on both. Keeping that split explicit is what makes this
// a generality gate rather than two copies of a fixture.
HfConfig V4Config(int64_t hidden_size, int64_t num_hidden_layers,
                  int64_t num_attention_heads, int64_t n_routed_experts,
                  int64_t moe_intermediate_size, int64_t q_lora_rank, int64_t o_groups,
                  int64_t index_topk, double routed_scaling_factor,
                  const std::vector<int64_t>& compress_ratios) {
  HfConfig c;
  c.architectures = {"DeepseekV4ForCausalLM"};
  c.hidden_size = hidden_size;
  c.num_hidden_layers = num_hidden_layers;
  c.vocab_size = 129280;
  c.num_attention_heads = num_attention_heads;
  c.num_key_value_heads = 1;
  c.head_dim = 512;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 1048576;
  c.raw = {
      {"hidden_size", hidden_size},
      {"num_hidden_layers", num_hidden_layers},
      {"num_attention_heads", num_attention_heads},
      {"n_routed_experts", n_routed_experts},
      {"moe_intermediate_size", moe_intermediate_size},
      {"q_lora_rank", q_lora_rank},
      {"o_groups", o_groups},
      {"index_topk", index_topk},
      {"routed_scaling_factor", routed_scaling_factor},
      {"compress_ratios", compress_ratios},
      // --- identical on Flash and Pro ---
      {"vocab_size", 129280},
      {"num_key_value_heads", 1},
      {"head_dim", 512},
      {"qk_rope_head_dim", 64},
      {"o_lora_rank", 1024},
      {"sliding_window", 128},
      {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 1048576},
      {"num_nextn_predict_layers", 1},
      {"num_experts_per_tok", 6},
      {"n_shared_experts", 1},
      {"norm_topk_prob", true},
      {"swiglu_limit", 10.0},
      {"scoring_func", "sqrtsoftplus"},
      {"topk_method", "noaux_tc"},
      {"num_hash_layers", 3},
      {"expert_dtype", "fp4"},
      {"hc_mult", 4},
      {"hc_sinkhorn_iters", 20},
      {"hc_eps", 1e-6},
      {"index_head_dim", 128},
      {"index_n_heads", 64},
      {"compress_rope_theta", 160000},
      {"rope_theta", 10000},
      {"tie_word_embeddings", false},
  };
  return c;
}

HfConfig ProConfig() {
  return V4Config(/*hidden_size=*/7168, /*num_hidden_layers=*/61,
                  /*num_attention_heads=*/128, /*n_routed_experts=*/384,
                  /*moe_intermediate_size=*/3072, /*q_lora_rank=*/1536,
                  /*o_groups=*/16, /*index_topk=*/1024,
                  /*routed_scaling_factor=*/2.5, ProCompressRatios());
}

HfConfig FlashConfig() {
  return V4Config(/*hidden_size=*/4096, /*num_hidden_layers=*/43,
                  /*num_attention_heads=*/64, /*n_routed_experts=*/256,
                  /*moe_intermediate_size=*/2048, /*q_lora_rank=*/1024,
                  /*o_groups=*/8, /*index_topk=*/512,
                  /*routed_scaling_factor=*/1.5, FlashCompressRatios());
}

// Count the compressor/indexer layers the way the forward does — through the
// params' own predicates, over [0, num_hidden_layers) only, so the trailing
// MTP entry is excluded exactly as upstream excludes it (attention.py:208).
struct DsaLayerCounts {
  int64_t compressor = 0;
  int64_t indexer = 0;
};

DsaLayerCounts CountDsaLayers(const DeepseekV4Params& p) {
  DsaLayerCounts n;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    if (p.has_compressor(l)) ++n.compressor;
    if (p.has_indexer(l)) ++n.indexer;
  }
  return n;
}

}  // namespace

TEST_CASE("deepseek-v4-pro: the real Pro config DESCENDS with no new keys") {
  const DeepseekV4Params p = ParseDeepseekV4Params(ProConfig());

  // Every value the Flash/Pro diff showed CHANGING must arrive scaled, not
  // clamped to a Flash default.
  CHECK(p.hidden_size == 7168);
  CHECK(p.num_hidden_layers == 61);
  CHECK(p.num_attention_heads == 128);
  CHECK(p.n_routed_experts == 384);
  CHECK(p.moe_intermediate_size == 3072);
  CHECK(p.q_lora_rank == 1536);
  CHECK(p.o_groups == 16);
  CHECK(p.index_topk == 1024);
  CHECK(p.routed_scaling_factor == doctest::Approx(2.5).scale(0.0));

  // Every value the diff showed IDENTICAL must be unchanged from Flash.
  CHECK(p.head_dim == 512);
  CHECK(p.qk_rope_head_dim == 64);
  CHECK(p.o_lora_rank == 1024);
  CHECK(p.num_experts_per_tok == 6);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.num_hash_layers == 3);
  CHECK(p.sliding_window == 128);
  CHECK(p.hc_mult == 4);
  CHECK(p.hc_sinkhorn_iters == 20);
  CHECK(p.index_head_dim == 128);
  CHECK(p.index_n_heads == 64);
  CHECK(p.scoring_func == "sqrtsoftplus");
  CHECK(p.topk_method == "noaux_tc");
  CHECK(p.expert_dtype == "fp4");
  CHECK(p.vocab_size == 129280);
}

TEST_CASE("deepseek-v4-pro: derived MLA/LoRA geometry scales off the config") {
  const DeepseekV4Params pro = ParseDeepseekV4Params(ProConfig());
  const DeepseekV4Params flash = ParseDeepseekV4Params(FlashConfig());

  // NoPE width is head_dim - rope on both (448 = 512 - 64), the geometry the
  // `head_dim == 512` scope assertion pins.
  CHECK(pro.head_dim - pro.qk_rope_head_dim == 448);
  CHECK(flash.head_dim - flash.qk_rope_head_dim == 448);

  // Grouped OUTPUT LoRA: heads-per-group is 8 on BOTH (64/8 and 128/16), so the
  // wo_a bmm batch shape is invariant across the two checkpoints even though
  // both operands doubled.
  CHECK(pro.num_attention_heads / pro.o_groups == 8);
  CHECK(flash.num_attention_heads / flash.o_groups == 8);

  // The wo_b input width (`zdim` in the forward) scales with o_groups only.
  CHECK(pro.o_groups * pro.o_lora_rank == 16384);
  CHECK(flash.o_groups * flash.o_lora_rank == 8192);
}

TEST_CASE("deepseek-v4-pro: compress_ratios reproduce BOTH real checkpoints") {
  const DsaLayerCounts pro = CountDsaLayers(ParseDeepseekV4Params(ProConfig()));
  const DsaLayerCounts flash = CountDsaLayers(ParseDeepseekV4Params(FlashConfig()));

  // Counted from the real model.safetensors.index.json of each checkpoint by
  // grouping compressor/indexer tensors per layer (see the header note).
  CHECK(pro.compressor == 61);
  CHECK(pro.indexer == 30);
  CHECK(flash.compressor == 41);
  CHECK(flash.indexer == 21);

  // Pro has MORE compressor layers than it has non-MTP layers in Flash, and the
  // two counts must not be equal — a Flash-hardcoded layer count would collapse
  // these into the same number.
  CHECK(pro.compressor != flash.compressor);
  CHECK(pro.indexer != flash.indexer);
}

TEST_CASE("deepseek-v4-pro: the layer-0/1 compressor delta is the ONE structural diff") {
  const DeepseekV4Params pro = ParseDeepseekV4Params(ProConfig());
  const DeepseekV4Params flash = ParseDeepseekV4Params(FlashConfig());

  // Pro's compress_ratios start [128, 128, ...] where Flash starts [0, 0, ...].
  // Pro therefore gains a ratio-128 compressor on layers 0 and 1 (confirmed in
  // the real Pro checkpoint: 4 compressor tensors on each of those layers, where
  // Flash has none). Neither is an indexer layer, since indexer == ratio 4.
  for (int64_t l = 0; l < 2; ++l) {
    CHECK(pro.compress_ratio(l) == 128);
    CHECK(pro.has_compressor(l));
    CHECK_FALSE(pro.has_indexer(l));

    CHECK(flash.compress_ratio(l) == 0);
    CHECK_FALSE(flash.has_compressor(l));
    CHECK_FALSE(flash.has_indexer(l));
  }

  // From layer 2 on, both alternate 4/128 identically, so the compressor and
  // indexer roles agree layer-for-layer over Flash's whole depth.
  for (int64_t l = 2; l < flash.num_hidden_layers; ++l) {
    CHECK(pro.compress_ratio(l) == flash.compress_ratio(l));
    CHECK(pro.has_indexer(l) == flash.has_indexer(l));
  }

  // Only ratios 0, 4 and 128 ever appear — upstream's compressor asserts
  // `compress_ratio in [4, 128]` (compressor.py:171) after mapping 0 to "absent",
  // so any other value would be unrepresentable on both sides.
  for (int64_t l = 0; l < pro.num_hidden_layers; ++l) {
    const int64_t r = pro.compress_ratio(l);
    CHECK((r == 0 || r == 4 || r == 128));
  }
}

TEST_CASE("deepseek-v4-pro: the trailing MTP compress_ratios entry is excluded") {
  const DeepseekV4Params pro = ParseDeepseekV4Params(ProConfig());
  const DeepseekV4Params flash = ParseDeepseekV4Params(FlashConfig());

  // Both arrays are num_hidden_layers + 1 long; the extra entry is the nextn/MTP
  // layer and is 0. Upstream guards the same off-by-one with an explicit
  // `if layer_id < config.num_hidden_layers` (attention.py:208-211).
  REQUIRE(static_cast<int64_t>(pro.compress_ratios.size()) == pro.num_hidden_layers + 1);
  REQUIRE(static_cast<int64_t>(flash.compress_ratios.size()) ==
          flash.num_hidden_layers + 1);
  CHECK(pro.compress_ratios.back() == 0);
  CHECK(flash.compress_ratios.back() == 0);

  // num_nextn_predict_layers == 1 on both, which is what that entry accounts for.
  CHECK(pro.num_nextn_predict_layers == 1);
  CHECK(flash.num_nextn_predict_layers == 1);
}
