// Kimi-Linear (`KimiLinearForCausalLM`) W1 SCAFFOLDING gate. Proves the four
// things this row's W1 can prove on CPU without a GPU or the 91.5 GiB checkpoint:
//   (1) the arch RESOLVES through the registry (the additive TU registered it),
//   (2) the config PARSES: ParseKimiLinearParams resolves the authoritative
//       48B-A3B schedule (20 KDA + 7 NoPE-MLA), the MLA/MoE/KDA dims, and REJECTS
//       an unrepresentable config (missing linear_attn_config, a positional MLA, a
//       q-LoRA query branch, a non-sigmoid/softmax router),
//   (3) the checkpoint NAME-MAP is faithful: EnumerateKimiLinearTensors branches
//       KDA vs NoPE-MLA per layer and MoE (block_sparse_moe) vs dense per layer,
//       grounded 1:1 in the pinned kimi_linear.py + the real safetensors index,
//   (4) the LOADER round-trips a SYNTHETIC 2-layer checkpoint (layer 0 KDA+dense,
//       layer 1 NoPE-MLA+MoE) — every enumerated tensor accounted, RED-first: a
//       missing tensor AND a mis-shaped tensor each throw BY NAME (never a silent
//       zero).
// The forward REFUSES-by-name (W3-W6), asserted implicitly by the model-matrix row
// staying SPIKE. See .agents/specs/kimi-linear.md §4/§5.
#include "vllm/model_executor/models/kimi_linear.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

using vllm::EnumerateKimiLinearTensors;
using vllm::HfConfig;
using vllm::KimiLinearParams;
using vllm::LoadKimiLinearForCausalLMWeights;
using vllm::ModelRegistry;
using vllm::ParseKimiLinearParams;
using vllm::SafetensorsFile;

namespace {

// ── synthetic safetensors builder (mirror test_laguna_nvfp4_loader.cpp) ───────
struct Fx {
  std::string name;
  std::string dtype;  // "BF16" / "F32"
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}
std::string Fill(size_t n, int seed) {
  std::string s(n, '\0');
  for (size_t i = 0; i < n; ++i)
    s[i] = static_cast<char>((seed * 31 + static_cast<int>(i)) & 0xff);
  return s;
}
std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype},
                   {"shape", t.shape},
                   {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("kimi_linear_scaffold_" + std::to_string(counter++) + ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// ── tiny 2-layer geometry (layer 0 KDA + dense, layer 1 NoPE-MLA + MoE) ────────
constexpr int H = 32, NAH = 4, V = 8, DENSE_I = 64, MOE_I = 16, E = 2;
constexpr int KV_LORA = 16, QK_NOPE = 8, QK_ROPE = 4, V_HEAD = 8;
constexpr int KDA_NH = 4, KDA_HD = 8, CONV = 4;
constexpr int KDA_PROJ = KDA_NH * KDA_HD;                 // 32
constexpr int MLA_QK = QK_NOPE + QK_ROPE;                 // 12

int64_t NumEl(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}
Fx Bf16(const std::string& n, std::vector<int64_t> shape, int seed) {
  return {n, "BF16", shape, Fill(static_cast<size_t>(NumEl(shape)) * 2, seed)};
}
Fx F32(const std::string& n, std::vector<int64_t> shape, int seed) {
  return {n, "F32", shape, Fill(static_cast<size_t>(NumEl(shape)) * 4, seed)};
}

// The synthetic checkpoint: exactly the tensors EnumerateKimiLinearTensors expects
// for the 2-layer tiny config, at the config-implied shapes.
std::vector<Fx> BuildTensors() {
  std::vector<Fx> v;
  int s = 1;
  v.push_back(Bf16("model.embed_tokens.weight", {V, H}, s++));
  v.push_back(Bf16("model.norm.weight", {H}, s++));
  v.push_back(Bf16("lm_head.weight", {V, H}, s++));

  // layer 0 = KDA + dense MLP.
  const std::string a0 = "model.layers.0.";
  v.push_back(Bf16(a0 + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16(a0 + "post_attention_layernorm.weight", {H}, s++));
  const std::string k = a0 + "self_attn.";
  v.push_back(Bf16(k + "q_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "k_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "v_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "f_a_proj.weight", {KDA_HD, H}, s++));
  v.push_back(Bf16(k + "f_b_proj.weight", {KDA_PROJ, KDA_HD}, s++));
  v.push_back(Bf16(k + "b_proj.weight", {KDA_NH, H}, s++));
  v.push_back(Bf16(k + "g_a_proj.weight", {KDA_HD, H}, s++));
  v.push_back(Bf16(k + "g_b_proj.weight", {KDA_PROJ, KDA_HD}, s++));
  v.push_back(Bf16(k + "o_proj.weight", {H, KDA_PROJ}, s++));
  // conv1d weights + A_log are presence-only (rank/layout not config-pinned).
  v.push_back(F32(k + "q_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "k_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "v_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "dt_bias", {KDA_PROJ}, s++));
  v.push_back(F32(k + "A_log", {KDA_NH}, s++));
  v.push_back(Bf16(k + "o_norm.weight", {KDA_HD}, s++));
  v.push_back(Bf16(a0 + "mlp.gate_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16(a0 + "mlp.up_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16(a0 + "mlp.down_proj.weight", {H, DENSE_I}, s++));

  // layer 1 = NoPE-MLA + MoE.
  const std::string a1 = "model.layers.1.";
  v.push_back(Bf16(a1 + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16(a1 + "post_attention_layernorm.weight", {H}, s++));
  const std::string m = a1 + "self_attn.";
  v.push_back(Bf16(m + "q_proj.weight", {NAH * MLA_QK, H}, s++));
  v.push_back(Bf16(m + "kv_a_proj_with_mqa.weight", {KV_LORA + QK_ROPE, H}, s++));
  v.push_back(Bf16(m + "kv_a_layernorm.weight", {KV_LORA}, s++));
  v.push_back(Bf16(m + "kv_b_proj.weight", {NAH * (QK_NOPE + V_HEAD), KV_LORA}, s++));
  v.push_back(Bf16(m + "o_proj.weight", {H, NAH * V_HEAD}, s++));
  const std::string mo = a1 + "block_sparse_moe.";
  v.push_back(Bf16(mo + "gate.weight", {E, H}, s++));
  v.push_back(F32(mo + "gate.e_score_correction_bias", {E}, s++));
  v.push_back(Bf16(mo + "shared_experts.gate_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16(mo + "shared_experts.up_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16(mo + "shared_experts.down_proj.weight", {H, MOE_I}, s++));
  for (int e = 0; e < E; ++e) {
    const std::string ep = mo + "experts." + std::to_string(e) + ".";
    v.push_back(Bf16(ep + "w1.weight", {MOE_I, H}, s++));
    v.push_back(Bf16(ep + "w2.weight", {H, MOE_I}, s++));
    v.push_back(Bf16(ep + "w3.weight", {MOE_I, H}, s++));
  }
  return v;
}

nlohmann::json TinyRaw() {
  return {
      {"hidden_size", H},
      {"num_hidden_layers", 2},
      {"vocab_size", V},
      {"num_attention_heads", NAH},
      {"num_key_value_heads", NAH},
      {"intermediate_size", DENSE_I},
      {"rms_norm_eps", 1e-5},
      {"tie_word_embeddings", false},
      {"num_experts", E},
      {"num_experts_per_token", 1},
      {"num_shared_experts", 1},
      {"moe_intermediate_size", MOE_I},
      {"first_k_dense_replace", 1},
      {"moe_layer_freq", 1},
      {"moe_router_activation_func", "sigmoid"},
      {"routed_scaling_factor", 2.446},
      {"kv_lora_rank", KV_LORA},
      {"qk_nope_head_dim", QK_NOPE},
      {"qk_rope_head_dim", QK_ROPE},
      {"v_head_dim", V_HEAD},
      {"mla_use_nope", true},
      // kda_layers 1-based: layer 0 KDA (1 in set), layer 1 MLA (2 not in set).
      {"linear_attn_config",
       {{"kda_layers", nlohmann::json::array({1})},
        {"full_attn_layers", nlohmann::json::array({2})},
        {"num_heads", KDA_NH},
        {"head_dim", KDA_HD},
        {"short_conv_kernel_size", CONV}}},
  };
}

HfConfig TinyConfig() {
  HfConfig c;
  c.architectures = {"KimiLinearForCausalLM"};
  c.model_type = "kimi_linear";
  c.hidden_size = H;
  c.num_hidden_layers = 2;
  c.vocab_size = V;
  c.num_attention_heads = NAH;
  c.raw = TinyRaw();
  return c;
}

// The authoritative 48B-A3B config (spike §0): 27 layers, 20 KDA + 7 NoPE-MLA,
// full_attn_layers 1-indexed = [4,8,12,16,20,24,27]; 256e/top-8/1-shared sigmoid,
// routed_scaling 2.446, first_k_dense_replace 1; MLA 512/128/64/128, q_lora null.
HfConfig RealConfig() {
  HfConfig c;
  c.architectures = {"KimiLinearForCausalLM"};
  c.model_type = "kimi_linear";
  c.hidden_size = 2304;
  c.num_hidden_layers = 27;
  c.vocab_size = 163840;
  c.num_attention_heads = 32;
  const std::vector<int> full_attn = {4, 8, 12, 16, 20, 24, 27};
  nlohmann::json kda = nlohmann::json::array();
  for (int i = 1; i <= 27; ++i)
    if (std::find(full_attn.begin(), full_attn.end(), i) == full_attn.end())
      kda.push_back(i);
  c.raw = {
      {"hidden_size", 2304},
      {"num_hidden_layers", 27},
      {"vocab_size", 163840},
      {"num_attention_heads", 32},
      {"intermediate_size", 9216},
      {"rms_norm_eps", 1e-5},
      {"tie_word_embeddings", false},
      {"num_experts", 256},
      {"num_experts_per_token", 8},
      {"num_shared_experts", 1},
      {"moe_intermediate_size", 1024},
      {"first_k_dense_replace", 1},
      {"moe_router_activation_func", "sigmoid"},
      {"routed_scaling_factor", 2.446},
      {"kv_lora_rank", 512},
      {"qk_nope_head_dim", 128},
      {"qk_rope_head_dim", 64},
      {"v_head_dim", 128},
      {"mla_use_nope", true},
      {"num_nextn_predict_layers", 0},
      {"linear_attn_config",
       {{"kda_layers", kda},
        {"full_attn_layers",
         nlohmann::json::array({4, 8, 12, 16, 20, 24, 27})},
        {"num_heads", 32},
        {"head_dim", 128},
        {"short_conv_kernel_size", 4}}},
  };
  return c;
}

}  // namespace

TEST_CASE("kimi-linear scaffold: KimiLinearForCausalLM RESOLVES via registry") {
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  CHECK(std::find(supported.begin(), supported.end(),
                  "KimiLinearForCausalLM") != supported.end());

  HfConfig cfg;
  cfg.architectures = {"KimiLinearForCausalLM"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);
  CHECK(reg.architecture == "KimiLinearForCausalLM");
  CHECK(reg.info.is_text_generation_model);
  CHECK(reg.info.is_hybrid);            // 20 KDA linear-attn layers
  CHECK_FALSE(reg.info.supports_multimodal);  // text-only (K3 wrapper is the MM one)
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
}

TEST_CASE("kimi-linear scaffold: config resolves the authoritative 48B schedule") {
  const KimiLinearParams p = ParseKimiLinearParams(RealConfig());
  CHECK(p.hidden_size == 2304);
  CHECK(p.num_hidden_layers == 27);
  CHECK(p.vocab_size == 163840);
  CHECK(p.num_attention_heads == 32);
  // NoPE-MLA geometry.
  CHECK(p.kv_lora_rank == 512);
  CHECK(p.q_lora_rank == 0);  // null (KimiLinear branch)
  CHECK(p.qk_nope_head_dim == 128);
  CHECK(p.qk_rope_head_dim == 64);
  CHECK(p.v_head_dim == 128);
  CHECK(p.mla_use_nope);
  CHECK(p.mla_head_size() == 576);  // kv_lora + qk_rope
  // MoE scalars.
  CHECK(p.num_experts == 256);
  CHECK(p.num_experts_per_token == 8);
  CHECK(p.num_shared_experts == 1);
  CHECK(p.moe_intermediate_size == 1024);
  CHECK(p.first_k_dense_replace == 1);
  CHECK(p.moe_router_activation_func == "sigmoid");
  CHECK(p.routed_scaling_factor == doctest::Approx(2.446));
  CHECK(p.num_nextn_predict_layers == 0);  // no MTP head in this checkpoint
  // KDA dims + conv projection width.
  CHECK(p.kda_num_heads == 32);
  CHECK(p.kda_head_dim == 128);
  CHECK(p.kda_short_conv_kernel_size == 4);
  CHECK(p.kda_conv_dim() == 3 * 32 * 128);  // 12288
}

TEST_CASE("kimi-linear scaffold: per-layer KDA/MLA + MoE/dense split matches upstream") {
  const KimiLinearParams p = ParseKimiLinearParams(RealConfig());
  int kda = 0, mla = 0;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l)
    (p.is_kda_layer(l) ? kda : mla)++;
  CHECK(kda == 20);
  CHECK(mla == 7);
  // is_kda_layer(l) == (l+1) in kda_layers; layer 3 (=full_attn 4) is MLA.
  CHECK(p.is_kda_layer(0));
  CHECK_FALSE(p.is_kda_layer(3));   // full_attn_layers has 4 (1-indexed)
  CHECK_FALSE(p.is_kda_layer(26));  // full_attn_layers has 27
  CHECK(p.is_kda_layer(25));        // 26 in kda set
  // first_k_dense_replace=1 => layer 0 dense, 1..26 MoE.
  CHECK_FALSE(p.is_moe_layer(0));
  CHECK(p.is_moe_layer(1));
  CHECK(p.is_moe_layer(26));
}

TEST_CASE("kimi-linear scaffold: checkpoint name-map is structurally faithful") {
  const KimiLinearParams p = ParseKimiLinearParams(TinyConfig());
  const std::vector<std::string> names = EnumerateKimiLinearTensors(p);
  const auto has = [&](const std::string& n) {
    return std::find(names.begin(), names.end(), n) != names.end();
  };
  CHECK(has("model.embed_tokens.weight"));
  CHECK(has("model.norm.weight"));
  CHECK(has("lm_head.weight"));  // tie_word_embeddings=false

  // layer 0 = KDA + dense: KDA tensors + dense MLP, NO MoE, NO MLA proj.
  CHECK(has("model.layers.0.self_attn.f_a_proj.weight"));  // KDA low-rank decay
  CHECK(has("model.layers.0.self_attn.g_b_proj.weight"));  // KDA gate low-rank
  CHECK(has("model.layers.0.self_attn.o_norm.weight"));    // sigmoid-gated norm
  CHECK(has("model.layers.0.self_attn.q_conv1d.weight"));  // 3 short convs
  CHECK(has("model.layers.0.self_attn.A_log"));
  CHECK(has("model.layers.0.self_attn.dt_bias"));
  CHECK(has("model.layers.0.mlp.gate_proj.weight"));            // dense MLP
  CHECK_FALSE(has("model.layers.0.block_sparse_moe.gate.weight"));  // NOT MoE
  CHECK_FALSE(has("model.layers.0.self_attn.kv_a_proj_with_mqa.weight"));  // NOT MLA

  // layer 1 = NoPE-MLA + MoE. q_lora null => direct q_proj (no q_a/q_b_proj).
  CHECK(has("model.layers.1.self_attn.q_proj.weight"));
  CHECK(has("model.layers.1.self_attn.kv_a_proj_with_mqa.weight"));
  CHECK(has("model.layers.1.self_attn.kv_a_layernorm.weight"));
  CHECK(has("model.layers.1.self_attn.kv_b_proj.weight"));
  CHECK_FALSE(has("model.layers.1.self_attn.q_a_proj.weight"));   // NO q-LoRA branch
  CHECK_FALSE(has("model.layers.1.self_attn.f_a_proj.weight"));   // NOT KDA
  // MoE: block_sparse_moe (NOT mlp) + noaux_tc bias + shared + routed w1/w2/w3.
  CHECK(has("model.layers.1.block_sparse_moe.gate.weight"));
  CHECK(has("model.layers.1.block_sparse_moe.gate.e_score_correction_bias"));
  CHECK(has("model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight"));
  CHECK(has("model.layers.1.block_sparse_moe.experts.0.w1.weight"));
  CHECK(has("model.layers.1.block_sparse_moe.experts.1.w3.weight"));  // E=2 => 0,1
  CHECK_FALSE(has("model.layers.1.block_sparse_moe.experts.2.w1.weight"));
  CHECK_FALSE(has("model.layers.1.mlp.gate_proj.weight"));  // MoE layer, not dense
}

TEST_CASE("kimi-linear scaffold: heterogeneous KV spec = MLA latent + KDA mamba") {
  const HfConfig cfg = RealConfig();
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(cfg, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 2);
  // Group 0 = MLA latent-KV (576-wide, num_kv_heads==1, no separate V).
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kMlaAttention);
  // Group 1 = KDA/GDN recurrent state (conv row + square recurrent row).
  const auto* mamba = dynamic_cast<const vllm::v1::MambaSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(mamba != nullptr);
  REQUIRE(mamba->shapes.size() == 2);
  CHECK(mamba->shapes[0] == std::vector<int64_t>{3 * 32 * 128, 3});  // conv (K-1=3)
  CHECK(mamba->shapes[1] == std::vector<int64_t>{32, 128, 128});      // recurrent
  CHECK(mamba->dtypes[0] == vt::DType::kBF16);
  CHECK(mamba->dtypes[1] == vt::DType::kF32);
}

TEST_CASE("kimi-linear scaffold: loader round-trips a synthetic 2-layer checkpoint") {
  const std::vector<Fx> ts = BuildTensors();
  TempFile f(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));

  const vllm::KimiLinearWeights w =
      LoadKimiLinearForCausalLMWeights(shards, TinyConfig());
  CHECK(w.params.num_hidden_layers == 2);
  CHECK(w.enumerated_tensors == static_cast<int64_t>(ts.size()));
  CHECK(w.accounted_tensors == w.enumerated_tensors);  // every tensor accounted
}

TEST_CASE("kimi-linear scaffold: loader throws BY NAME on a missing tensor (RED)") {
  std::vector<Fx> ts = BuildTensors();
  // Drop one routed-expert tensor -> the loader must throw, not silently zero.
  ts.erase(std::remove_if(ts.begin(), ts.end(),
                          [](const Fx& t) {
                            return t.name ==
                                   "model.layers.1.block_sparse_moe.experts.1.w2.weight";
                          }),
           ts.end());
  TempFile f(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  CHECK_THROWS_AS(LoadKimiLinearForCausalLMWeights(shards, TinyConfig()),
                  std::runtime_error);
}

TEST_CASE("kimi-linear scaffold: loader throws on a mis-shaped tensor (RED)") {
  std::vector<Fx> ts = BuildTensors();
  // Corrupt kv_b_proj's shape (wrong out dim) -> shape-check must catch it.
  for (Fx& t : ts)
    if (t.name == "model.layers.1.self_attn.kv_b_proj.weight") {
      t.shape = {NAH * (QK_NOPE + V_HEAD) + 1, KV_LORA};
      t.bytes = Fill(static_cast<size_t>(NumEl(t.shape)) * 2, 99);
    }
  TempFile f(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  CHECK_THROWS_AS(LoadKimiLinearForCausalLMWeights(shards, TinyConfig()),
                  std::runtime_error);
}

TEST_CASE("kimi-linear scaffold: parse REJECTS unrepresentable configs") {
  // missing linear_attn_config (not a KDA hybrid)
  HfConfig bad1 = TinyConfig();
  bad1.raw.erase("linear_attn_config");
  CHECK_THROWS_AS(ParseKimiLinearParams(bad1), std::runtime_error);
  // positional MLA (mla_use_nope false) — KimiMLAAttention asserts use_nope
  HfConfig bad2 = TinyConfig();
  bad2.raw["mla_use_nope"] = false;
  CHECK_THROWS_AS(ParseKimiLinearParams(bad2), std::runtime_error);
  // q-LoRA query branch (q_lora_rank>0) — the K3 path, not Kimi-Linear
  HfConfig bad3 = TinyConfig();
  bad3.raw["q_lora_rank"] = 256;
  CHECK_THROWS_AS(ParseKimiLinearParams(bad3), std::runtime_error);
  // non-sigmoid/softmax router
  HfConfig bad4 = TinyConfig();
  bad4.raw["moe_router_activation_func"] = "gelu";
  CHECK_THROWS_AS(ParseKimiLinearParams(bad4), std::runtime_error);
}
