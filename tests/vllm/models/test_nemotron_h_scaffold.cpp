// Nemotron-H (`NemotronHForCausalLM`) W3 STRUCTURAL gate — issue #517, spec
// `.agents/specs/nemotron-h-model.md` §4 W3.
//
// Proves the four things W3 can prove on CPU with no GPU and no 20.1 GiB
// checkpoint, plus one live re-verification that runs only where the checkpoint
// is staged:
//   (1) the arch RESOLVES through the registry (the additive TU registered it);
//   (2) the config PARSES off the REAL released config.json (committed as a
//       fixture): the 52-entry schedule with 23 mamba / 23 moe / 6 attention at
//       indices 5,12,19,26,33,42, the mamba/attention/MoE geometry, the coarse
//       quantization surface — and the legacy layouts
//       (`hybrid_override_pattern`, `mamba_n_groups`, `mamba_d_conv`, ...)
//       normalize to the SAME params, while an unrepresentable config REFUSES
//       BY NAME;
//   (3) the on-disk NAME MAP is faithful: every one of the 18487 tensors in the
//       released `model.safetensors.index.json` is CLAIMED by a named consumer,
//       and nothing is enumerated that the checkpoint does not ship;
//   (4) the HETEROGENEOUS KV topology matches `mamba2_state_shape` /
//       `_mamba_state_dtype` and carries the REAL per-layer names.
// The forward REFUSES BY NAME (W4 owns it) and GGUF REFUSES BY NAME (W7).
//
// The fixture is a HEADERS-ONLY projection of the released checkpoint at the
// pinned revision `29f2d174`; see its `_provenance`. The live re-verification
// resolves the checkpoint through `parity::Nemotron35LightningSnapshot()`
// (env `VT_NEMOTRON35_SNAPSHOT`) and SKIPS loudly when it is absent.
#include "vllm/model_executor/models/nemotron_h.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "hf_snapshot.h"
#include "vllm/model_executor/models/model_registry.h"
// The forward-refusal SUBCASE has to CALL the type-erased forward, so it needs
// the concrete definitions of the seam types `model_registry.h` only forward-
// declares. `nemotron_h_registry.cpp:28` reaches for the same header for the
// same reason.
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, *KvCache
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"
#include "vt/dtype.h"

using vllm::EnumerateNemotronHTensors;
using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::MakeNemotronHKVCache;
using vllm::ModelRegistry;
using vllm::NemotronHBlock;
using vllm::NemotronHParams;
using vllm::NemotronHTensor;
using vllm::ParseNemotronHParams;

namespace {

std::string FixtureDir() {
#ifdef NEMOTRON_H_CKPT_FIXTURE_DIR
  return NEMOTRON_H_CKPT_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/nemotron_h_35_lightning";
#endif
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "cannot open " << path);
  nlohmann::json j;
  in >> j;
  return j;
}

// A throwaway config.json on disk, so the test drives the SAME LoadHfConfig the
// engine uses rather than hand-building an HfConfig.
class TempConfig {
 public:
  explicit TempConfig(const nlohmann::json& doc) {
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
           ("nemotron_h_cfg_" + std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "config.json") << doc.dump();
  }
  ~TempConfig() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string path() const { return (dir_ / "config.json").string(); }

 private:
  std::filesystem::path dir_;
};

nlohmann::json FixtureConfigDoc() {
  return ReadJson(FixtureDir() + "/config.json");
}

NemotronHParams FixtureParams() {
  TempConfig cfg(FixtureConfigDoc());
  return ParseNemotronHParams(LoadHfConfig(cfg.path()));
}

// Expand the committed collapsed index into the full 18487 on-disk names.
// `{E}` is the routed-expert index and `count` says how many share the entry.
std::map<std::string, std::pair<std::string, std::vector<int64_t>>>
ExpandIndexFixture(const nlohmann::json& index) {
  std::map<std::string, std::pair<std::string, std::vector<int64_t>>> out;
  for (const auto& [pattern, meta] : index.at("tensors").items()) {
    const auto dtype = meta.at("dtype").get<std::string>();
    const auto shape = meta.at("shape").get<std::vector<int64_t>>();
    const auto count = meta.at("count").get<int64_t>();
    const std::string marker = "{E}";
    const size_t at = pattern.find(marker);
    if (at == std::string::npos) {
      REQUIRE_MESSAGE(count == 1, "non-expert family with count != 1: "
                                      << pattern);
      out.emplace(pattern, std::make_pair(dtype, shape));
      continue;
    }
    for (int64_t e = 0; e < count; ++e) {
      std::string name = pattern;
      name.replace(at, marker.size(), std::to_string(e));
      out.emplace(std::move(name), std::make_pair(dtype, shape));
    }
  }
  return out;
}

// Read the LIVE checkpoint's index + safetensors HEADERS (never a tensor byte;
// the checkpoint is 20.1 GiB).
std::map<std::string, std::pair<std::string, std::vector<int64_t>>>
ReadLiveHeaders(const std::string& dir) {
  const nlohmann::json index = ReadJson(dir + "/model.safetensors.index.json");
  std::set<std::string> shards;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)name;
    shards.insert(shard.get<std::string>());
  }
  std::map<std::string, std::pair<std::string, std::vector<int64_t>>> out;
  for (const std::string& shard : shards) {
    std::ifstream in(dir + "/" + shard, std::ios::binary);
    REQUIRE_MESSAGE(in.good(), "cannot open shard " << shard);
    unsigned char raw[8];
    in.read(reinterpret_cast<char*>(raw), 8);
    uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8) | raw[i];
    std::string header(static_cast<size_t>(n), '\0');
    in.read(header.data(), static_cast<std::streamsize>(n));
    // MATERIALIZE the parsed header: `json::parse(x).items()` binds a range
    // to a TEMPORARY, which is destroyed before the loop body runs. It reads
    // as a clean one-liner and it is undefined behaviour — here it surfaced as
    // `[json.exception.type_error.304] cannot use at() with null` from inside
    // the loop, nowhere near the parse.
    const nlohmann::json hj = nlohmann::json::parse(header);
    REQUIRE_MESSAGE(hj.is_object(), "shard header is not an object: " << shard);
    for (const auto& [name, meta] : hj.items()) {
      if (name == "__metadata__") continue;
      out.emplace(name, std::make_pair(meta.at("dtype").get<std::string>(),
                                       meta.at("shape").get<std::vector<int64_t>>()));
    }
  }
  return out;
}

const vllm::v1::MambaSpec& MambaGroup(const vllm::v1::KVCacheConfig& kv) {
  for (const auto& group : kv.kv_cache_groups) {
    const auto* spec =
        dynamic_cast<const vllm::v1::MambaSpec*>(group.kv_cache_spec.get());
    if (spec != nullptr) return *spec;
  }
  FAIL("no MambaSpec group");
  std::abort();
}

}  // namespace

TEST_CASE("NemotronH: the architecture resolves through the model registry") {
  const std::vector<std::string> archs{"NemotronHForCausalLM"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(archs);
  CHECK(std::string(reg.architecture) == "NemotronHForCausalLM");
  CHECK(reg.info.is_text_generation_model);
  CHECK(reg.info.is_hybrid);
  CHECK_FALSE(reg.info.supports_multimodal);
  CHECK(reg.factory != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
}

TEST_CASE("NemotronH config: the REAL released config.json parses") {
  const NemotronHParams p = FixtureParams();

  // Depth comes from the SCHEDULE, not from the deprecated scalar.
  CHECK(p.num_hidden_layers() == 52);
  CHECK(p.LayerIndices(NemotronHBlock::kMamba).size() == 23);
  CHECK(p.LayerIndices(NemotronHBlock::kMoe).size() == 23);
  CHECK(p.LayerIndices(NemotronHBlock::kMlp).empty());
  CHECK(p.LayerIndices(NemotronHBlock::kAttention) ==
        std::vector<int64_t>{5, 12, 19, 26, 33, 42});

  CHECK(p.hidden_size == 2688);
  CHECK(p.vocab_size == 131072);
  CHECK(p.max_position_embeddings == 1048576);
  CHECK_FALSE(p.tie_word_embeddings);
  CHECK(p.layer_norm_epsilon == 1e-5);

  // Attention: 32 q / 2 kv heads, head_dim 128, full rotary, no window.
  CHECK(p.num_attention_heads == 32);
  CHECK(p.num_key_value_heads == 2);
  CHECK(p.head_dim == 128);
  CHECK(p.rope_theta == 10000.0);
  CHECK(p.partial_rotary_factor == 1.0);
  CHECK_FALSE(p.attention_bias);
  CHECK_FALSE(p.sliding_window.has_value());
  CHECK(p.q_proj_out_features() == 4096);
  CHECK(p.kv_proj_out_features() == 256);

  // Mamba2. conv_dim carries the 2*n_groups*state_size term; dropping it
  // yields 4096 and the released conv1d.weight is [6144, 1, 4].
  CHECK(p.mamba_num_heads == 64);
  CHECK(p.mamba_head_dim == 64);
  CHECK(p.n_groups == 8);
  CHECK(p.ssm_state_size == 128);
  CHECK(p.conv_kernel == 4);
  CHECK(p.chunk_size == 128);
  CHECK(p.expand == 2);
  CHECK(p.mamba_hidden_act == "silu");
  CHECK(p.mamba_ssm_cache_dtype == "float32");
  CHECK(p.use_conv_bias);
  CHECK_FALSE(p.use_bias);
  CHECK_FALSE(p.mamba_proj_bias);
  CHECK(p.mamba_intermediate_size() == 4096);
  CHECK(p.conv_dim() == 6144);
  CHECK(p.in_proj_out_features() == 10304);

  // MoE. `routed_scaling_factor` is applied to the OUTPUT (nemotron_h.py:246).
  CHECK(p.n_routed_experts == 128);
  CHECK(p.num_experts_per_tok == 6);
  CHECK(p.moe_intermediate_size == 1856);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.moe_shared_expert_intermediate_size == 3712);
  CHECK(p.n_group == 1);
  CHECK(p.topk_group == 1);
  CHECK(p.routed_scaling_factor == 2.5);
  CHECK(p.norm_topk_prob);
  CHECK(p.moe_shared_expert_overlap);
  CHECK(p.mlp_hidden_act == "relu2");
  // `null` and ABSENT are the same state upstream (nemotron_h.py:143).
  CHECK_FALSE(p.moe_latent_size.has_value());

  // MTP: 1 predict layer over a 2-block ["attention","moe"] pattern.
  CHECK(p.num_nextn_predict_layers == 1);
  CHECK(p.mtp_layers_block_type ==
        std::vector<NemotronHBlock>{NemotronHBlock::kAttention,
                                    NemotronHBlock::kMoe});

  // The coarse quantization surface (per-module algo resolution is W1).
  CHECK(p.quant.present);
  CHECK(p.quant.quant_method == "modelopt");
  CHECK(p.quant.quant_algo == "MIXED_PRECISION");
  CHECK(p.quant.fp8_kv_cache);
  CHECK(p.quant.mtp_ignored);
}

TEST_CASE("NemotronH config: the LEGACY layout normalizes to the same params") {
  nlohmann::json doc = FixtureConfigDoc();
  // Swap every modern key for the legacy alias transformers still accepts
  // (configuration_nemotron_h.py:142-190). A layout that silently
  // deserializes to all-defaults is a wrong-shaped model with no error.
  const std::vector<NemotronHBlock> modern =
      ParseNemotronHParams(LoadHfConfig(TempConfig(doc).path()))
          .layers_block_type;

  std::string pattern;
  for (NemotronHBlock b : modern) {
    pattern += b == NemotronHBlock::kMamba       ? 'M'
               : b == NemotronHBlock::kMoe       ? 'E'
               : b == NemotronHBlock::kAttention ? '*'
                                                 : '-';
  }
  doc.erase("layers_block_type");
  doc["hybrid_override_pattern"] = pattern;
  doc.erase("mtp_layers_block_type");
  doc["mtp_hybrid_override_pattern"] = "*E";
  doc["mamba_n_groups"] = doc["n_groups"];
  doc.erase("n_groups");
  doc["mamba_d_conv"] = doc["conv_kernel"];
  doc.erase("conv_kernel");
  doc["mamba_expand"] = doc["expand"];
  doc.erase("expand");
  doc["mamba_chunk_size"] = doc["chunk_size"];
  doc.erase("chunk_size");
  doc["mamba_conv_bias"] = doc["use_conv_bias"];
  doc.erase("use_conv_bias");
  doc["mamba_dt_min"] = doc["time_step_min"];
  doc.erase("time_step_min");
  doc["mamba_dt_max"] = doc["time_step_max"];
  doc.erase("time_step_max");
  doc["mamba_dt_init_floor"] = doc["time_step_floor"];
  doc.erase("time_step_floor");

  TempConfig cfg(doc);
  const NemotronHParams p = ParseNemotronHParams(LoadHfConfig(cfg.path()));
  CHECK(p.layers_block_type == modern);
  CHECK(p.LayerIndices(NemotronHBlock::kAttention) ==
        std::vector<int64_t>{5, 12, 19, 26, 33, 42});
  CHECK(p.n_groups == 8);
  CHECK(p.conv_kernel == 4);
  CHECK(p.expand == 2);
  CHECK(p.chunk_size == 128);
  CHECK(p.use_conv_bias);
  CHECK(p.time_step_min == 1e-3);
  CHECK(p.time_step_max == 1e-1);
  CHECK(p.time_step_floor == 1e-4);
  CHECK(p.conv_dim() == 6144);
  CHECK(p.mtp_layers_block_type ==
        std::vector<NemotronHBlock>{NemotronHBlock::kAttention,
                                    NemotronHBlock::kMoe});
}

TEST_CASE("NemotronH config: absent schedules take upstream's defaults") {
  nlohmann::json doc = FixtureConfigDoc();
  doc.erase("layers_block_type");
  doc.erase("mtp_layers_block_type");
  TempConfig cfg(doc);
  const NemotronHParams p = ParseNemotronHParams(LoadHfConfig(cfg.path()));
  // configuration_nemotron_h.py:165 and :180.
  CHECK(p.layers_block_type ==
        std::vector<NemotronHBlock>{NemotronHBlock::kMamba, NemotronHBlock::kMoe,
                                    NemotronHBlock::kAttention,
                                    NemotronHBlock::kMlp});
  CHECK(p.num_hidden_layers() == 4);
  CHECK(p.mtp_layers_block_type ==
        std::vector<NemotronHBlock>{NemotronHBlock::kAttention,
                                    NemotronHBlock::kMoe});
}

TEST_CASE("NemotronH config: unrepresentable configs REFUSE BY NAME") {
  SUBCASE("an unknown block type") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["layers_block_type"][0] = "swa";
    TempConfig cfg(doc);
    CHECK_THROWS_AS(ParseNemotronHParams(LoadHfConfig(cfg.path())),
                    std::runtime_error);
    // The refusal NAMES the offender and enumerates the four block spellings
    // from `NemotronHBlockName`, so it cannot list four kinds after a fifth is
    // added. `doctest::Contains` takes a `const char*` LITERAL here on purpose:
    // doctest 2.5.2 stringifies a `const char*` VARIABLE as `1`.
    CHECK_THROWS_WITH_AS(ParseNemotronHParams(LoadHfConfig(cfg.path())),
                         doctest::Contains("'swa'"), std::runtime_error);
    for (NemotronHBlock block : {NemotronHBlock::kMamba,
                                 NemotronHBlock::kAttention,
                                 NemotronHBlock::kMoe, NemotronHBlock::kMlp}) {
      const std::string name(vllm::NemotronHBlockName(block));
      CHECK_FALSE(name.empty());
      CHECK(name != "unknown");
      // Round-trip: every spelling the refusal offers must actually PARSE.
      nlohmann::json ok = FixtureConfigDoc();
      ok["layers_block_type"][0] = name;
      TempConfig ok_cfg(ok);
      const NemotronHParams p = ParseNemotronHParams(LoadHfConfig(ok_cfg.path()));
      CHECK(p.layers_block_type.at(0) == block);
    }
  }
  SUBCASE("a latent MoE") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["moe_latent_size"] = 512;
    TempConfig cfg(doc);
    CHECK_THROWS_AS(ParseNemotronHParams(LoadHfConfig(cfg.path())),
                    std::runtime_error);
  }
  SUBCASE("a heterogeneous per-layer intermediate_size (NemotronHPuzzle)") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["intermediate_size"] = nlohmann::json::array({1856, 1856});
    TempConfig cfg(doc);
    CHECK_THROWS_AS(ParseNemotronHParams(LoadHfConfig(cfg.path())),
                    std::runtime_error);
  }
  SUBCASE("a quantization method we do not resolve") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["quantization_config"]["quant_method"] = "awq";
    TempConfig cfg(doc);
    CHECK_THROWS_AS(ParseNemotronHParams(LoadHfConfig(cfg.path())),
                    std::runtime_error);
  }
  SUBCASE("an ssm cache dtype we cannot represent") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["mamba_ssm_cache_dtype"] = "float8_e4m3";
    TempConfig cfg(doc);
    CHECK_THROWS_AS(ParseNemotronHParams(LoadHfConfig(cfg.path())),
                    std::runtime_error);
  }
  SUBCASE("an MTP head with no MTP schedule") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["mtp_layers_block_type"] = nlohmann::json::array();
    TempConfig cfg(doc);
    CHECK_THROWS_AS(ParseNemotronHParams(LoadHfConfig(cfg.path())),
                    std::runtime_error);
  }
}

TEST_CASE(
    "NemotronH enumeration: all 18487 released tensors are claimed, none "
    "invented") {
  const NemotronHParams p = FixtureParams();
  const std::vector<NemotronHTensor> enumerated = EnumerateNemotronHTensors(p);

  const nlohmann::json index = ReadJson(FixtureDir() + "/index.json");
  CHECK(index.at("total_tensors").get<int64_t>() == 18487);
  const auto released = ExpandIndexFixture(index);
  REQUIRE(released.size() == 18487);

  // Every enumerated name is on disk, exactly once, with a named consumer.
  std::set<std::string> seen;
  std::vector<std::string> invented;
  for (const NemotronHTensor& t : enumerated) {
    CHECK_MESSAGE(!t.consumer.empty(), "no named consumer for " << t.name);
    CHECK_MESSAGE(seen.insert(t.name).second,
                  "enumerated twice: " << t.name);
    if (released.count(t.name) == 0) invented.push_back(t.name);
  }
  CHECK_MESSAGE(invented.empty(),
                "enumerated tensors the checkpoint does not ship, first: "
                    << (invented.empty() ? std::string("-") : invented.front())
                    << " (" << invented.size() << " total)");

  // ...and every tensor on disk is claimed. Zero unaccounted.
  std::vector<std::string> unaccounted;
  for (const auto& [name, meta] : released) {
    (void)meta;
    if (seen.count(name) == 0) unaccounted.push_back(name);
  }
  CHECK_MESSAGE(unaccounted.empty(),
                "UNCLAIMED checkpoint tensors, first: "
                    << (unaccounted.empty() ? std::string("-")
                                            : unaccounted.front())
                    << " (" << unaccounted.size() << " total)");
  CHECK(enumerated.size() == 18487);
}

TEST_CASE(
    "NemotronH enumeration: an UNQUANTIZED producer claims NO scale "
    "companions") {
  // The released checkpoint is ModelOpt-quantized, so every `quantized` flag is
  // true there and the 18487-tensor gate above cannot see a claimer that IGNORES
  // the flag. Drop `quantization_config` — the shape a released bf16 NemotronH
  // safetensors checkpoint actually ships (spec §5b) — and the scale companions
  // must all disappear. A claimer that hard-codes its FP8/NVFP4 companions
  // enumerates tensors that do not exist, which is the silent mis-enumeration
  // AGENTS.md forbids ("an arm that is not implemented is refused by name").
  nlohmann::json doc = FixtureConfigDoc();
  doc.erase("quantization_config");
  TempConfig cfg(doc);
  const NemotronHParams p = ParseNemotronHParams(LoadHfConfig(cfg.path()));
  REQUIRE_FALSE(p.quant.present);
  REQUIRE_FALSE(p.quant.fp8_kv_cache);

  const std::vector<NemotronHTensor> enumerated = EnumerateNemotronHTensors(p);
  const auto ends_with = [](const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  std::vector<std::string> companions;
  for (const NemotronHTensor& t : enumerated) {
    if (ends_with(t.name, ".weight_scale") ||
        ends_with(t.name, ".weight_scale_2") ||
        ends_with(t.name, ".input_scale") || ends_with(t.name, ".k_scale") ||
        ends_with(t.name, ".v_scale")) {
      companions.push_back(t.name);
    }
  }
  CHECK_MESSAGE(companions.empty(),
                "an unquantized producer still claims scale companions, first: "
                    << (companions.empty() ? std::string("-")
                                           : companions.front())
                    << " (" << companions.size() << " total)");

  // The quantized arm is unchanged: the released config still claims all 18487.
  const std::vector<NemotronHTensor> quantized =
      EnumerateNemotronHTensors(FixtureParams());
  CHECK(quantized.size() == 18487);
  // The bf16 arm is the same MODEL with the companions removed, nothing else:
  // every unquantized name is also claimed by the quantized arm.
  std::set<std::string> quantized_names;
  for (const NemotronHTensor& t : quantized) quantized_names.insert(t.name);
  std::vector<std::string> only_bf16;
  for (const NemotronHTensor& t : enumerated) {
    if (quantized_names.count(t.name) == 0) only_bf16.push_back(t.name);
  }
  CHECK_MESSAGE(only_bf16.empty(),
                "the bf16 arm invented a name the quantized arm never claims, "
                "first: "
                    << (only_bf16.empty() ? std::string("-")
                                          : only_bf16.front()));
}

TEST_CASE(
    "NemotronH config: when BOTH spellings ship, the precedence is upstream's "
    "and it is PER-FAMILY") {
  // Re-derived by RUNNING transformers @ 7d06b1a5 (the pin this file's header
  // names), not by reading it:
  //   NemotronHConfig(n_groups=8, mamba_n_groups=4, conv_kernel=4,
  //                   mamba_d_conv=7)  ->  n_groups=4, conv_kernel=7
  //   NemotronHConfig(layer_types=['mamba','mamba'],
  //                   hybrid_override_pattern='*-')  ->  ['mamba','mamba']
  // The two families genuinely DISAGREE, and each is mirrored on its own terms:
  //
  //   mamba_* SCALARS (configuration_nemotron_h.py:145-155) — LEGACY wins.
  //     `self.n_groups = kwargs.pop("mamba_n_groups") if "mamba_n_groups" in
  //     kwargs else self.n_groups`: the dataclass field already holds the modern
  //     value, and the legacy alias OVERWRITES it unconditionally.
  //
  //   SCHEDULES (configuration_nemotron_h.py:158-165, :176-184) — MODERN wins.
  //     `if "hybrid_override_pattern" in kwargs: ... if self.layer_types is
  //     None: self.layer_types = _pattern_to_list(pattern)`: the legacy pattern
  //     is consulted ONLY when the modern list is absent.
  //
  // No released checkpoint ships both spellings of the same field, so this is a
  // mirroring obligation rather than a live defect — which is exactly why it
  // needs a test: nothing else can catch it drifting.
  nlohmann::json doc = FixtureConfigDoc();
  const std::vector<NemotronHBlock> modern_schedule =
      ParseNemotronHParams(LoadHfConfig(TempConfig(doc).path()))
          .layers_block_type;

  // Every mamba_* scalar gets a legacy alias that DISAGREES with the modern key
  // the fixture already ships.
  doc["mamba_n_groups"] = 4;      // modern `n_groups` is 8
  doc["mamba_d_conv"] = 7;        // modern `conv_kernel` is 4
  doc["mamba_expand"] = 9;        // modern `expand` is 2
  doc["mamba_chunk_size"] = 77;   // modern `chunk_size` is 128
  doc["mamba_conv_bias"] = false; // modern `use_conv_bias` is true
  doc["mamba_dt_min"] = 0.5;      // modern `time_step_min` is 1e-3
  doc["mamba_dt_max"] = 0.6;      // modern `time_step_max` is 1e-1
  doc["mamba_dt_init_floor"] = 0.7;  // modern `time_step_floor` is 1e-4
  // ...and both schedules get a legacy pattern that disagrees too.
  doc["hybrid_override_pattern"] = "*-";
  doc["mtp_hybrid_override_pattern"] = "M-";

  TempConfig cfg(doc);
  const NemotronHParams p = ParseNemotronHParams(LoadHfConfig(cfg.path()));

  // LEGACY wins for the scalars.
  CHECK(p.n_groups == 4);
  CHECK(p.conv_kernel == 7);
  CHECK(p.expand == 9);
  CHECK(p.chunk_size == 77);
  CHECK_FALSE(p.use_conv_bias);
  CHECK(p.time_step_min == 0.5);
  CHECK(p.time_step_max == 0.6);
  CHECK(p.time_step_floor == 0.7);

  // MODERN wins for the schedules — do NOT "unify" these with the scalars.
  CHECK(p.layers_block_type == modern_schedule);
  CHECK(p.layers_block_type.size() == 52);
  CHECK(p.mtp_layers_block_type ==
        std::vector<NemotronHBlock>{NemotronHBlock::kAttention,
                                    NemotronHBlock::kMoe});
}

TEST_CASE("NemotronH enumeration: the shapes it implies are the ones on disk") {
  // The three geometry facts W3 owns, checked against the RELEASED headers
  // rather than against our own arithmetic.
  const NemotronHParams p = FixtureParams();
  const auto released = ExpandIndexFixture(ReadJson(FixtureDir() + "/index.json"));

  const auto shape_of = [&](const std::string& name) {
    const auto it = released.find(name);
    REQUIRE_MESSAGE(it != released.end(), "absent from the index: " << name);
    return it->second.second;
  };

  // conv_dim = intermediate + 2*n_groups*state_size. Without the second term
  // this is 4096.
  CHECK(shape_of("backbone.layers.0.mixer.conv1d.weight") ==
        std::vector<int64_t>{p.conv_dim(), 1, p.conv_kernel});
  CHECK(shape_of("backbone.layers.0.mixer.conv1d.bias") ==
        std::vector<int64_t>{p.conv_dim()});
  // in_proj = z + xBC + dt.
  CHECK(shape_of("backbone.layers.0.mixer.in_proj.weight") ==
        std::vector<int64_t>{p.in_proj_out_features(), p.hidden_size});
  CHECK(shape_of("backbone.layers.0.mixer.out_proj.weight") ==
        std::vector<int64_t>{p.hidden_size, p.mamba_intermediate_size()});
  CHECK(shape_of("backbone.layers.0.mixer.A_log") ==
        std::vector<int64_t>{p.mamba_num_heads});
  // GQA: 32 q heads, 2 kv heads, head_dim 128.
  CHECK(shape_of("backbone.layers.5.mixer.q_proj.weight") ==
        std::vector<int64_t>{p.q_proj_out_features(), p.hidden_size});
  CHECK(shape_of("backbone.layers.5.mixer.k_proj.weight") ==
        std::vector<int64_t>{p.kv_proj_out_features(), p.hidden_size});
  // NVFP4 W4A16 group_size 16: the packed weight halves the input dim and the
  // per-block e4m3 scale divides it by 16.
  const auto up = shape_of("backbone.layers.1.mixer.experts.0.up_proj.weight");
  const auto up_scale =
      shape_of("backbone.layers.1.mixer.experts.0.up_proj.weight_scale");
  CHECK(up == std::vector<int64_t>{p.moe_intermediate_size, p.hidden_size / 2});
  CHECK(up_scale ==
        std::vector<int64_t>{p.moe_intermediate_size, p.hidden_size / 16});
  CHECK(shape_of("backbone.layers.1.mixer.gate.weight") ==
        std::vector<int64_t>{p.n_routed_experts, p.hidden_size});
  // The MTP fusion projection consumes [embed ; hidden].
  CHECK(shape_of("mtp.layers.0.eh_proj.weight") ==
        std::vector<int64_t>{p.hidden_size, 2 * p.hidden_size});
}

TEST_CASE("NemotronH KV: the het groups mirror mamba2_state_shape") {
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const NemotronHParams p = ParseNemotronHParams(config);
  const vllm::v1::KVCacheConfig kv = MakeNemotronHKVCache(config, 16, 4);

  REQUIRE(kv.kv_cache_groups.size() == 2);
  CHECK(kv.num_blocks == 4);
  CHECK(kv.has_mamba_layers());

  // (1) the full-attention group covers exactly the 6 attention layers.
  const auto& attn_group = kv.kv_cache_groups[0];
  const auto* attn = dynamic_cast<const vllm::v1::FullAttentionSpec*>(
      attn_group.kv_cache_spec.get());
  REQUIRE(attn != nullptr);
  CHECK(attn->num_kv_heads == 2);
  CHECK(attn->head_size == 128);
  CHECK(attn_group.layer_names.size() == 6);
  std::vector<std::string> expected_attn;
  for (int64_t i : p.LayerIndices(NemotronHBlock::kAttention)) {
    expected_attn.push_back("backbone.layers." + std::to_string(i) + ".mixer");
  }
  CHECK(attn_group.layer_names == expected_attn);

  // (2) the Mamba2 group covers exactly the 23 mamba layers, at the upstream
  //     state shapes and with the SSM dtype resolved INDEPENDENTLY of the conv
  //     dtype (mamba_ssm_cache_dtype: "float32").
  const auto& mamba_group = kv.kv_cache_groups[1];
  const vllm::v1::MambaSpec& mamba = MambaGroup(kv);
  CHECK(mamba.shapes ==
        std::vector<std::vector<int64_t>>{{6144, 3}, {64, 64, 128}});
  CHECK(mamba.dtypes ==
        std::vector<vt::DType>{vt::DType::kBF16, vt::DType::kF32});
  CHECK(mamba_group.layer_names.size() == 23);
  std::vector<std::string> expected_mamba;
  for (int64_t i : p.LayerIndices(NemotronHBlock::kMamba)) {
    expected_mamba.push_back("backbone.layers." + std::to_string(i) + ".mixer");
  }
  CHECK(mamba_group.layer_names == expected_mamba);

  // The 23x / 6x multipliers are the whole point of carrying real names.
  const int64_t conv_bytes = 6144 * 3 * 2;              // bf16
  const int64_t ssm_bytes = 64 * 64 * 128 * 4;          // f32
  CHECK(mamba.page_size_bytes() == conv_bytes + ssm_bytes);
  CHECK(vllm::v1::KVBytesPerBlock(kv) == attn->page_size_bytes() * 6);
}

TEST_CASE("NemotronH: the unported arms REFUSE BY NAME") {
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);

  SUBCASE("GGUF k-quants are owed (W7), never silently dequantized") {
    vllm::ModelSource source;
    source.kind = vllm::ModelSource::Kind::kGguf;
    CHECK_THROWS_WITH_AS(reg.factory->load_weights(reg, config, source),
                         doctest::Contains("NemotronHForCausalLM"),
                         std::runtime_error);
  }

  SUBCASE("the forward still REFUSES on a checkpoint load, rather than returning zeros") {
    // UPDATED BY W4 (#517). W3 pinned this refusal when
    // `ForwardNemotronHForCausalLM` was an unconditional VT_CHECK reading
    // "forward is not implemented yet". W4 ports the forward MECHANISM
    // (nemotron_h.cpp) and reaches it through this same
    // `ModelRegistry::Forward` seam, so the unconditional refusal is gone — but
    // a checkpoint STILL cannot be run, because there is no NemotronH weight
    // LOADER at all. Every load therefore leaves `NemotronHHostWeights`
    // unmaterialized and the forward refuses THERE instead, naming the piece
    // that is actually missing rather than the whole feature.
    //
    // The guarantee this subcase exists for is UNCHANGED and is the one that
    // matters: reaching the registered forward without a materialized
    // checkpoint THROWS and NAMES the gap. A forward that silently returned
    // `{}` would produce zero logits and a plausible-looking garbage token.
    // What moved is only which piece the message names.
    struct StubModel : vllm::LoadedModel {
      explicit StubModel(const vllm::ModelRegistration& r) : LoadedModel(r) {}
    };
    StubModel model(reg);
    const std::vector<int32_t> token_ids{0};
    const std::vector<int32_t> positions{0};
    const std::vector<int32_t> logits_indices{0};
    const vllm::v1::CommonAttentionMetadata attn_meta{};
    const vllm::v1::GDNAttentionMetadata gdn_meta{};
    std::vector<vllm::PagedKvCache> attn_kv;
    std::vector<vllm::GdnStateCache> gdn_state;
    vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const vllm::ModelForwardInput input{.token_ids = token_ids,
                                        .positions = positions,
                                        .attn_meta = attn_meta,
                                        .gdn_meta = gdn_meta,
                                        .attn_kv = attn_kv,
                                        .gdn_state = gdn_state,
                                        .config = config,
                                        .queue = queue,
                                        .logits_indices = logits_indices,
                                        .num_reqs = 1};
    // The message must NAME the missing piece, not just fail. After W4 the
    // missing piece is the WEIGHT LOAD, not the forward.
    CHECK_THROWS_WITH_AS(reg.factory->forward(model, input),
                         doctest::Contains("host weights are not materialized"),
                         std::runtime_error);
    // ...and it must still say so in NemotronH's own name, so a refusal from
    // some shared helper cannot be mistaken for this one.
    CHECK_THROWS_WITH_AS(reg.factory->forward(model, input),
                         doctest::Contains("NemotronHForCausalLM forward"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(reg.factory->forward(model, input),
                         doctest::Contains("nemotron-h-model.md"),
                         std::runtime_error);
  }
}

TEST_CASE("NemotronH: the committed fixture matches the LIVE checkpoint") {
  const std::string dir = parity::Nemotron35LightningSnapshot();
  if (dir.empty()) {
    MESSAGE(
        "SKIP: set VT_NEMOTRON35_SNAPSHOT to "
        "$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4 (revision "
        "29f2d1746d8f41e316523194b19018707749b1b1) to re-verify the fixture");
    return;
  }

  // config.json: identical key-by-key, and the ELIDED set is EXACTLY the two
  // 5981-entry ModelOpt maps W1 owns.
  const nlohmann::json live = ReadJson(dir + "/config.json");
  const nlohmann::json fixture = FixtureConfigDoc();
  for (const auto& [key, value] : live.items()) {
    if (key == "quantization_config") continue;
    CHECK_MESSAGE(fixture.contains(key), "fixture is missing key " << key);
    if (fixture.contains(key)) {
      CHECK_MESSAGE(fixture.at(key) == value, "fixture key drifted: " << key);
    }
  }
  for (const auto& [key, value] : fixture.items()) {
    (void)value;
    CHECK_MESSAGE(live.contains(key), "fixture invented key " << key);
  }
  std::set<std::string> elided;
  for (const auto& [key, value] : live.at("quantization_config").items()) {
    if (!fixture.at("quantization_config").contains(key)) {
      elided.insert(key);
      continue;
    }
    CHECK_MESSAGE(fixture.at("quantization_config").at(key) == value,
                  "fixture quantization_config key drifted: " << key);
  }
  CHECK(elided == std::set<std::string>{"config_groups", "quantized_layers"});

  // index.json: the expanded fixture is the live header set, both directions.
  const auto expanded = ExpandIndexFixture(ReadJson(FixtureDir() + "/index.json"));
  const auto live_headers = ReadLiveHeaders(dir);
  CHECK(live_headers.size() == 18487);
  CHECK(expanded.size() == live_headers.size());
  std::vector<std::string> drifted;
  for (const auto& [name, meta] : live_headers) {
    const auto it = expanded.find(name);
    if (it == expanded.end() || it->second != meta) drifted.push_back(name);
  }
  CHECK_MESSAGE(drifted.empty(),
                "fixture drifted from the live checkpoint, first: "
                    << (drifted.empty() ? std::string("-") : drifted.front())
                    << " (" << drifted.size() << " total)");
}
