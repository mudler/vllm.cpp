// W7 loader gate for DeepSeek-V2 (`DeepseekV2ForCausalLM`, DeepSeek-V2-Lite,
// BF16) — the MLA campaign's first model bring-up. Loads the real 4-shard
// checkpoint from the HF cache and asserts the name map is complete and exact:
// every expected tensor mapped with the right shape (MLA projections + 27 layers
// x [1 dense MLP or 64 routed + 1 shared expert + router] + embed + norm +
// UNTIED lm_head), the LOAD-TIME `kv_b_proj -> W_UK/W_UV` absorption split is
// present and correctly shaped, and NO checkpoint tensor is left
// unmapped/leftover. Mirrors tests/vllm/models/test_qwen3_moe_load.cpp.
//
// On a box without the checkpoint (CI, CPU dev box) the checkpoint cases emit a
// loud SKIP; the config-resolution cases below run everywhere. Pure host load —
// no GPU, no oracle (the token-exact forward gate is W8).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;

using vllm::DeepseekV2Params;
using vllm::DeepseekV2Weights;
using vllm::HfConfig;
using vllm::LoadDeepseekV2ForCausalLMWeights;
using vllm::LoadHfConfig;
using vllm::LoadSafetensorsIndex;
using vllm::OwnedTensor;
using vllm::ParseDeepseekV2Params;
using vllm::SafetensorsFile;
namespace v1 = vllm::v1;

namespace {

std::string FindV2LiteSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec) &&
        fs::exists(e.path() / "model.safetensors.index.json", ec)) {
      return e.path().string();
    }
  }
  return "";
}

void CheckBf16(const OwnedTensor& t, int64_t d0, int64_t d1, bool nk) {
  REQUIRE(t.rank == 2);
  CHECK(t.dtype == vt::DType::kBF16);
  CHECK(t.shape[0] == d0);
  CHECK(t.shape[1] == d1);
  CHECK(t.nk == nk);
}
void CheckBf16_3D(const OwnedTensor& t, int64_t d0, int64_t d1, int64_t d2) {
  REQUIRE(t.rank == 3);
  CHECK(t.dtype == vt::DType::kBF16);
  CHECK(t.shape[0] == d0);
  CHECK(t.shape[1] == d1);
  CHECK(t.shape[2] == d2);
}
void CheckVec(const OwnedTensor& t, int64_t n) {
  REQUIRE(t.rank == 1);
  CHECK(t.dtype == vt::DType::kBF16);
  CHECK(t.shape[0] == n);
}

std::string WriteConfig(const std::string& name, const std::string& body) {
  const char* env = std::getenv("TMPDIR");
  const fs::path base = env != nullptr ? fs::path(env) : fs::temp_directory_path();
  const fs::path dir = base / "vllm_cpp_deepseek_v2_load_test";
  std::error_code ec;
  fs::create_directories(dir, ec);
  const std::string path = (dir / (name + ".json")).string();
  std::ofstream f(path);
  f << body;
  return path;
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// Registry + config resolution (runs everywhere)
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("deepseek-v2 registry: exactly DeepseekV2ForCausalLM is registered") {
  const std::vector<std::string_view> archs = vllm::ModelRegistry::SupportedArchs();
  const auto has = [&](std::string_view a) {
    return std::find(archs.begin(), archs.end(), a) != archs.end();
  };
  CHECK(has("DeepseekV2ForCausalLM"));
  // Deliberately NOT registered — see deepseek_v2.h "WHAT IS DELIBERATELY NOT
  // REGISTERED". In particular `DeepseekForCausalLM` is plain MHA
  // (deepseek_v2.py:1201-1211), not MLA, so claiming it would be false.
  CHECK_FALSE(has("DeepseekForCausalLM"));
  CHECK_FALSE(has("DeepseekV3ForCausalLM"));
  CHECK_FALSE(has("DeepseekV32ForCausalLM"));
}

TEST_CASE("deepseek-v2 config: the use_mha branch is REFUSED, not silently run") {
  // deepseek_v2.py:1201-1211 — model_type "deepseek" OR both qk dims 0 selects
  // DeepseekAttention (plain MHA), a different architecture.
  const std::string mha = R"({
    "architectures": ["DeepseekForCausalLM"], "model_type": "deepseek",
    "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "vocab_size": 100, "intermediate_size": 32, "rms_norm_eps": 1e-6})";
  CHECK_THROWS(ParseDeepseekV2Params(LoadHfConfig(WriteConfig("mha", mha))));

  const std::string zero_qk = R"({
    "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
    "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "vocab_size": 100, "intermediate_size": 32, "rms_norm_eps": 1e-6,
    "qk_nope_head_dim": 0, "qk_rope_head_dim": 0, "v_head_dim": 16,
    "kv_lora_rank": 24})";
  CHECK_THROWS(ParseDeepseekV2Params(LoadHfConfig(WriteConfig("zeroqk", zero_qk))));
}

TEST_CASE("deepseek-v2 KV spec: ONE MLA group — 1 head, 576 wide, NO factor 2") {
  // The cross-cutting cost the spike named (§4): MLA stores ONE latent vector
  // plus the decoupled rope part per token — `num_kv_heads == 1`,
  // `head_size == kv_lora_rank + qk_rope_head_dim`, and NO separate V, so the
  // K+V factor 2 every other attention spec carries simply does not appear
  // (kv_cache_interface.h `MLAAttentionSpec::real_page_size_bytes`).
  const std::string cfg = R"({
    "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
    "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16,
    "vocab_size": 1024, "intermediate_size": 64, "rms_norm_eps": 1e-6,
    "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
    "kv_lora_rank": 512, "q_lora_rank": null, "max_position_embeddings": 128})";
  const HfConfig config = LoadHfConfig(WriteConfig("kvspec", cfg));
  const v1::KVCacheConfig kv = vllm::MakeDeepseekV2KVCache(config, /*block_size=*/16,
                                                           /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);  // ONE group; no Mamba/GDN group
  CHECK(kv.num_blocks == 8);
  const v1::KVCacheSpec* spec = kv.kv_cache_groups[0].kv_cache_spec.get();
  REQUIRE(spec != nullptr);
  CHECK(spec->kind() == v1::KVCacheSpecKind::kMlaAttention);
  const auto* attn = dynamic_cast<const v1::AttentionSpec*>(spec);
  REQUIRE(attn != nullptr);
  CHECK(attn->num_kv_heads == 1);
  CHECK(attn->head_size == 576);  // kv_lora_rank 512 + qk_rope 64
  // block_size * 1 * 576 * sizeof(dtype) — NOT doubled.
  const int64_t es = static_cast<int64_t>(vt::SizeOf(attn->dtype));
  CHECK(spec->page_size_bytes() == 16 * 1 * 576 * es);
}

// ════════════════════════════════════════════════════════════════════════════
// The real-checkpoint loader gate (dgx-only)
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("deepseek-v2 W7 loader: DeepSeek-V2-Lite safetensors -> DeepseekV2Weights") {
  const std::string snap = FindV2LiteSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "SKIP: DeepSeek-V2-Lite checkpoint absent "
        "(~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite); "
        "run on dgx for the W7 load gate.");
    return;
  }

  const HfConfig config = LoadHfConfig(snap + "/config.json");
  const DeepseekV2Params p = ParseDeepseekV2Params(config);

  // Config sanity (DeepSeek-V2-Lite, from the shipped config.json).
  CHECK(p.num_hidden_layers == 27);
  CHECK(p.hidden_size == 2048);
  CHECK(p.vocab_size == 102400);
  CHECK(p.intermediate_size == 10944);
  CHECK(p.mla.num_heads == 16);
  CHECK(p.mla.qk_nope_head_dim == 128);
  CHECK(p.mla.qk_rope_head_dim == 64);
  CHECK(p.mla.v_head_dim == 128);
  CHECK(p.mla.kv_lora_rank == 512);
  // THE QUERY BRANCH: `q_lora_rank: null` -> the DIRECT q_proj path
  // (deepseek_v2.py:1028-1034), NOT the fused_qkv_a_proj path.
  CHECK(p.mla.q_lora_rank == 0);
  CHECK_FALSE(p.mla.has_q_lora());
  CHECK(p.mla.qk_head_dim() == 192);
  CHECK(p.mla.head_size() == 576);  // the MLA cache width
  // The softmax scale carries the YaRN mscale^2 correction
  // (deepseek_v2.py:995 then :1067-1075): 192**-0.5 * (0.1*0.707*ln(40)+1)^2.
  {
    const double mscale = 0.1 * 0.707 * std::log(40.0) + 1.0;
    const double want = std::pow(192.0, -0.5) * mscale * mscale;
    CHECK(p.mla.scale == doctest::Approx(static_cast<float>(want)).epsilon(1e-6));
  }
  CHECK(p.rope.yarn);
  CHECK(p.rope.scaling_factor == doctest::Approx(40.0));
  CHECK(p.rope.mscale == doctest::Approx(0.707));
  CHECK(p.rope.mscale_all_dim == doctest::Approx(0.707));
  CHECK(p.rope.original_max_position_embeddings == 4096);
  // MoE
  CHECK(p.n_routed_experts == 64);
  CHECK(p.num_experts_per_tok == 6);
  CHECK(p.moe_intermediate_size == 1408);
  CHECK(p.n_shared_experts == 2);  // SHARED EXPERTS — new for this family
  CHECK(p.shared_intermediate_size() == 2816);
  CHECK(p.first_k_dense_replace == 1);
  CHECK(p.n_group == 1);
  CHECK(p.topk_group == 1);
  CHECK_FALSE(p.norm_topk_prob);
  CHECK(p.scoring_func == vt::MoeScoringFunc::kSoftmax);
  // topk_method is "greedy", NOT "noaux_tc" -> NO e_score_correction_bias
  // parameter exists in this checkpoint (deepseek_v2.py:313-318).
  CHECK_FALSE(p.has_e_score_correction_bias);
  CHECK(p.routed_scaling_factor == doctest::Approx(1.0f));
  CHECK_FALSE(p.tie_word_embeddings);

  const std::map<std::string, std::string> wmap =
      LoadSafetensorsIndex(snap + "/model.safetensors.index.json");
  std::unordered_set<std::string> shard_files;
  for (const auto& [tensor, file] : wmap) shard_files.insert(file);
  std::vector<SafetensorsFile> shards;
  shards.reserve(shard_files.size());
  for (const std::string& f : shard_files)
    shards.push_back(SafetensorsFile::Open(snap + "/" + f));

  const DeepseekV2Weights w = LoadDeepseekV2ForCausalLMWeights(shards, config);

  const int64_t H = p.hidden_size;            // 2048
  const int64_t V = p.vocab_size;             // 102400
  const int64_t N = p.mla.num_heads;          // 16
  const int64_t P = p.mla.qk_nope_head_dim;   // 128
  const int64_t R = p.mla.qk_rope_head_dim;   // 64
  const int64_t Vd = p.mla.v_head_dim;        // 128
  const int64_t L = p.mla.kv_lora_rank;       // 512
  const int64_t Dqk = p.mla.qk_head_dim();    // 192
  const int64_t E = p.n_routed_experts;       // 64
  const int64_t I = p.moe_intermediate_size;  // 1408
  const int64_t Is = p.intermediate_size;     // 10944
  const int64_t Ish = p.shared_intermediate_size();  // 2816

  CheckBf16(w.embed_tokens, V, H, /*nk=*/false);
  CheckVec(w.final_norm, H);
  REQUIRE_FALSE(w.lm_head.Empty());  // UNTIED
  CheckBf16(w.lm_head, H, V, /*nk=*/false);
  // The shared YaRN [cos|sin] cache, one row per position, qk_rope wide.
  CheckBf16(w.rope_cos_sin_cache, p.max_position_embeddings, R, /*nk=*/false);

  REQUIRE(w.layers.size() == static_cast<size_t>(p.num_hidden_layers));
  int64_t dense_layers = 0, moe_layers = 0;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const auto& layer = w.layers[static_cast<size_t>(l)];
    CheckVec(layer.input_layernorm, H);
    CheckVec(layer.post_attention_layernorm, H);

    // --- MLA projections (the q_lora_rank IS NULL branch) ---
    CHECK(layer.attn.fused_qkv_a_proj.Empty());
    CHECK(layer.attn.q_a_layernorm.Empty());
    CHECK(layer.attn.q_b_proj.Empty());
    CheckBf16(layer.attn.q_proj, N * Dqk, H, /*nk=*/true);            // [3072,2048]
    CheckBf16(layer.attn.kv_a_proj_with_mqa, L + R, H, /*nk=*/true);  // [576,2048]
    CheckVec(layer.attn.kv_a_layernorm, L);
    CheckBf16(layer.attn.kv_b_proj, N * (P + Vd), L, /*nk=*/true);    // [4096,512]
    CheckBf16(layer.attn.o_proj, H, N * Vd, /*nk=*/true);             // [2048,2048]
    // --- the LOAD-TIME absorption split (mla_attention.py:892-900, :959-962) ---
    CheckBf16_3D(layer.attn.w_uk_t, N, P, L);  // [16,128,512]
    CheckBf16_3D(layer.attn.w_uv, N, L, Vd);   // [16,512,128]

    if (layer.is_moe) {
      ++moe_layers;
      CHECK(l >= p.first_k_dense_replace);
      CHECK(layer.dense.Empty());
      CheckBf16(layer.moe.router_gate, H, E, /*nk=*/false);
      CHECK(layer.moe.e_score_correction_bias.Empty());  // topk_method "greedy"
      REQUIRE(layer.moe.expert_gate.size() == static_cast<size_t>(E));
      REQUIRE(layer.moe.expert_up.size() == static_cast<size_t>(E));
      REQUIRE(layer.moe.expert_down.size() == static_cast<size_t>(E));
      for (int64_t e = 0; e < E; ++e) {
        CheckBf16(layer.moe.expert_gate[static_cast<size_t>(e)], H, I, false);
        CheckBf16(layer.moe.expert_up[static_cast<size_t>(e)], H, I, false);
        CheckBf16(layer.moe.expert_down[static_cast<size_t>(e)], I, H, false);
      }
      // SHARED EXPERTS: n_shared_experts == 2 fused into ONE MLP of width
      // moe_intermediate_size * n_shared_experts (deepseek_v2.py:346).
      REQUIRE_FALSE(layer.moe.shared.Empty());
      CheckBf16(layer.moe.shared.gate_up_proj, 2 * Ish, H, /*nk=*/true);
      CheckBf16(layer.moe.shared.down_proj, H, Ish, /*nk=*/true);
    } else {
      ++dense_layers;
      CHECK(l < p.first_k_dense_replace);
      CHECK(layer.moe.expert_gate.empty());
      CHECK(layer.moe.router_gate.Empty());
      REQUIRE_FALSE(layer.dense.Empty());
      CheckBf16(layer.dense.gate_up_proj, 2 * Is, H, /*nk=*/true);
      CheckBf16(layer.dense.down_proj, H, Is, /*nk=*/true);
    }
  }
  CHECK(dense_layers == 1);   // first_k_dense_replace == 1
  CHECK(moe_layers == 26);

  // NO unmapped/leftover tensors: the set of every checkpoint tensor name must
  // equal exactly the set of consumed names.
  std::unordered_set<std::string> expected;
  expected.insert("model.embed_tokens.weight");
  expected.insert("model.norm.weight");
  expected.insert("lm_head.weight");  // untied — consumed
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    expected.insert(b + "input_layernorm.weight");
    expected.insert(b + "post_attention_layernorm.weight");
    expected.insert(b + "self_attn.q_proj.weight");
    expected.insert(b + "self_attn.kv_a_proj_with_mqa.weight");
    expected.insert(b + "self_attn.kv_a_layernorm.weight");
    expected.insert(b + "self_attn.kv_b_proj.weight");
    expected.insert(b + "self_attn.o_proj.weight");
    if (p.is_moe_layer(l)) {
      expected.insert(b + "mlp.gate.weight");
      for (int64_t e = 0; e < E; ++e) {
        const std::string ex = b + "mlp.experts." + std::to_string(e) + ".";
        expected.insert(ex + "gate_proj.weight");
        expected.insert(ex + "up_proj.weight");
        expected.insert(ex + "down_proj.weight");
      }
      expected.insert(b + "mlp.shared_experts.gate_proj.weight");
      expected.insert(b + "mlp.shared_experts.up_proj.weight");
      expected.insert(b + "mlp.shared_experts.down_proj.weight");
    } else {
      expected.insert(b + "mlp.gate_proj.weight");
      expected.insert(b + "mlp.up_proj.weight");
      expected.insert(b + "mlp.down_proj.weight");
    }
  }

  std::unordered_set<std::string> actual;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) actual.insert(name);

  for (const std::string& name : actual)
    CHECK_MESSAGE(expected.count(name) == 1,
                  "unexpected checkpoint tensor: " << name);
  for (const std::string& name : expected)
    CHECK_MESSAGE(actual.count(name) == 1,
                  "expected tensor missing from ckpt: " << name);
  CHECK(actual.size() == expected.size());  // 5291 tensors
}
