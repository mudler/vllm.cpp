// GLM/DSA G1 loader gate for GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`, BF16) — the
// SECOND MLA model bring-up, reusing the DeepSeek-V2 loader over the SAME
// `DeepseekV2Weights` struct (glm4_moe_lite IS deepseek_v2 with GLM config
// values). Asserts the name map is complete and exact at GLM's dims, and that the
// two things new-to-e2e here are present and correctly shaped:
//   * the `q_lora_rank != null` QUERY BRANCH (`fused_qkv_a_proj` merge +
//     `q_a_layernorm` + `q_b_proj`), which DeepSeek-V2-Lite (`q_lora_rank: null`)
//     leaves unit-gated only;
//   * the `noaux_tc` `e_score_correction_bias` router parameter, likewise
//     unit-gated only on DeepSeek-V2-Lite (softmax/greedy, no bias).
// And it proves the MTP-tail SKIP: GLM-4.7-Flash ships `num_nextn_predict_layers:
// 1`, so `model.layers.47.*` is present in the checkpoint but NOT loaded — the
// ONLY unmapped tensors must be exactly that tail (glm4_moe_lite.py:358-360,
// 633-643 get_spec_layer_idx_from_weight_name).
//
// Mirrors tests/vllm/models/test_deepseek_v2_load.cpp. On a box without the
// checkpoint the real-load case SKIPs loudly; the registry/config cases run
// everywhere. Pure host load — no GPU, no oracle (the token-exact gate is the
// paged-engine SACRED test).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
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

std::string FindGlm47Snapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots";
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

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// Registry + config resolution (runs everywhere)
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("glm4-moe-lite registry: Glm4MoeLiteForCausalLM is registered") {
  const std::vector<std::string_view> archs = vllm::ModelRegistry::SupportedArchs();
  const auto has = [&](std::string_view a) {
    return std::find(archs.begin(), archs.end(), a) != archs.end();
  };
  CHECK(has("Glm4MoeLiteForCausalLM"));
  // The DeepSeek-V2 registration is untouched and still exactly one arch.
  CHECK(has("DeepseekV2ForCausalLM"));
}

TEST_CASE("glm4-moe-lite config: num_nextn_predict_layers>0 is TOLERATED (MTP skip)") {
  // GLM-4.7-Flash ships num_nextn_predict_layers: 1. The GLM parse passes
  // allow_mtp_tail=true, so it must NOT throw (the tail is skipped, not refused).
  // The SAME config through the DeepSeek-V2 parse (allow_mtp_tail=false default)
  // MUST throw — proving the two paths diverge only on this flag.
  const std::string cfg = R"({
    "architectures": ["Glm4MoeLiteForCausalLM"], "model_type": "glm4_moe_lite",
    "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 20,
    "vocab_size": 1024, "intermediate_size": 64, "rms_norm_eps": 1e-5,
    "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
    "kv_lora_rank": 512, "q_lora_rank": 768, "max_position_embeddings": 128,
    "n_routed_experts": 64, "num_experts_per_tok": 4, "moe_intermediate_size": 32,
    "n_shared_experts": 1, "first_k_dense_replace": 1, "n_group": 1,
    "topk_group": 1, "norm_topk_prob": true, "scoring_func": "sigmoid",
    "topk_method": "noaux_tc", "routed_scaling_factor": 1.8,
    "num_nextn_predict_layers": 1, "rope_theta": 1000000})";
  const char* env = std::getenv("TMPDIR");
  const fs::path base = env != nullptr ? fs::path(env) : fs::temp_directory_path();
  const fs::path dir = base / "vllm_cpp_glm4_moe_lite_load_test";
  std::error_code ec;
  fs::create_directories(dir, ec);
  const std::string path = (dir / "mtp.json").string();
  { std::ofstream f(path); f << cfg; }
  const HfConfig config = LoadHfConfig(path);

  // GLM parse (allow_mtp_tail=true): accepted, and it resolves the noaux_tc
  // router + q_lora branch.
  const DeepseekV2Params p = ParseDeepseekV2Params(config, /*allow_mtp_tail=*/true);
  CHECK(p.mla.q_lora_rank == 768);
  CHECK(p.mla.has_q_lora());
  CHECK(p.mla.qk_head_dim() == 256);   // 192 + 64
  CHECK(p.mla.head_size() == 576);     // 512 + 64 — SAME latent width as DeepSeek
  CHECK(p.scoring_func == vt::MoeScoringFunc::kSigmoid);
  CHECK(p.has_e_score_correction_bias);  // topk_method noaux_tc
  CHECK(p.norm_topk_prob);
  CHECK(p.routed_scaling_factor == doctest::Approx(1.8f));
  CHECK(p.n_group == 1);
  CHECK(p.topk_group == 1);
  // No yarn (rope_scaling absent) -> scale is the plain qk_head_dim**-0.5.
  CHECK(p.mla.scale == doctest::Approx(static_cast<float>(std::pow(256.0, -0.5))).epsilon(1e-6));
  CHECK_FALSE(p.rope.yarn);

  // DeepSeek-V2 parse (default allow_mtp_tail=false): REFUSES the MTP tail.
  CHECK_THROWS(ParseDeepseekV2Params(config));
}

// (The MLA KV-cache spec is byte-identical to DeepSeek-V2's — same 576-wide
// latent, 1 head, no factor 2 — and is gated by test_deepseek_v2_load.cpp's KV
// spec case plus the `head_size() == 576` assertion above; GLM's
// MakeGlm4MoeLiteKVCache only substitutes allow_mtp_tail=true into the identical
// builder, exercised by the paged-engine SACRED gate's phase 0.)

// ════════════════════════════════════════════════════════════════════════════
// The real-checkpoint loader gate (dgx-only)
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("glm4-moe-lite loader: GLM-4.7-Flash safetensors -> DeepseekV2Weights") {
  const std::string snap = FindGlm47Snapshot();
  if (snap.empty()) {
    MESSAGE(
        "SKIP: GLM-4.7-Flash checkpoint absent "
        "(~/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash); "
        "run on dgx for the loader gate.");
    return;
  }

  const HfConfig config = LoadHfConfig(snap + "/config.json");
  const DeepseekV2Params p = ParseDeepseekV2Params(config, /*allow_mtp_tail=*/true);

  // Config sanity (GLM-4.7-Flash, live config.json).
  CHECK(p.num_hidden_layers == 47);
  CHECK(p.hidden_size == 2048);
  CHECK(p.vocab_size == 154880);
  CHECK(p.intermediate_size == 10240);
  CHECK(p.mla.num_heads == 20);
  CHECK(p.mla.qk_nope_head_dim == 192);
  CHECK(p.mla.qk_rope_head_dim == 64);
  CHECK(p.mla.v_head_dim == 256);
  CHECK(p.mla.kv_lora_rank == 512);
  // THE QUERY BRANCH: q_lora_rank 768 -> the fused_qkv_a_proj path.
  CHECK(p.mla.q_lora_rank == 768);
  CHECK(p.mla.has_q_lora());
  CHECK(p.mla.qk_head_dim() == 256);
  CHECK(p.mla.head_size() == 576);
  // No yarn: plain qk_head_dim**-0.5.
  CHECK(p.mla.scale == doctest::Approx(static_cast<float>(std::pow(256.0, -0.5))).epsilon(1e-6));
  CHECK_FALSE(p.rope.yarn);
  // MoE + the noaux_tc router.
  CHECK(p.n_routed_experts == 64);
  CHECK(p.num_experts_per_tok == 4);
  CHECK(p.moe_intermediate_size == 1536);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.shared_intermediate_size() == 1536);
  CHECK(p.first_k_dense_replace == 1);
  CHECK(p.n_group == 1);
  CHECK(p.topk_group == 1);
  CHECK(p.norm_topk_prob);
  CHECK(p.scoring_func == vt::MoeScoringFunc::kSigmoid);
  CHECK(p.has_e_score_correction_bias);  // topk_method noaux_tc
  CHECK(p.routed_scaling_factor == doctest::Approx(1.8f));
  CHECK_FALSE(p.tie_word_embeddings);

  const std::map<std::string, std::string> wmap =
      LoadSafetensorsIndex(snap + "/model.safetensors.index.json");
  std::unordered_set<std::string> shard_files;
  for (const auto& [tensor, file] : wmap) shard_files.insert(file);
  std::vector<SafetensorsFile> shards;
  shards.reserve(shard_files.size());
  for (const std::string& f : shard_files)
    shards.push_back(SafetensorsFile::Open(snap + "/" + f));

  const DeepseekV2Weights w =
      LoadDeepseekV2ForCausalLMWeights(shards, config, /*allow_mtp_tail=*/true);

  const int64_t H = p.hidden_size;            // 2048
  const int64_t V = p.vocab_size;             // 154880
  const int64_t N = p.mla.num_heads;          // 20
  const int64_t P = p.mla.qk_nope_head_dim;   // 192
  const int64_t R = p.mla.qk_rope_head_dim;   // 64
  const int64_t Vd = p.mla.v_head_dim;        // 256
  const int64_t L = p.mla.kv_lora_rank;       // 512
  const int64_t Ql = p.mla.q_lora_rank;       // 768
  const int64_t Dqk = p.mla.qk_head_dim();    // 256
  const int64_t E = p.n_routed_experts;       // 64
  const int64_t I = p.moe_intermediate_size;  // 1536
  const int64_t Is = p.intermediate_size;     // 10240
  const int64_t Ish = p.shared_intermediate_size();  // 1536

  CheckBf16(w.embed_tokens, V, H, /*nk=*/false);
  CheckVec(w.final_norm, H);
  REQUIRE_FALSE(w.lm_head.Empty());  // UNTIED
  CheckBf16(w.lm_head, H, V, /*nk=*/false);
  CheckBf16(w.rope_cos_sin_cache, p.max_position_embeddings, R, /*nk=*/false);

  REQUIRE(w.layers.size() == static_cast<size_t>(p.num_hidden_layers));
  int64_t dense_layers = 0, moe_layers = 0;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const auto& layer = w.layers[static_cast<size_t>(l)];
    CheckVec(layer.input_layernorm, H);
    CheckVec(layer.post_attention_layernorm, H);

    // --- MLA projections (the q_lora_rank NON-NULL branch) ---
    CHECK(layer.attn.q_proj.Empty());              // fused branch: q_proj unused
    CHECK(layer.attn.kv_a_proj_with_mqa.Empty());  // folded into fused_qkv_a_proj
    // fused_qkv_a_proj = [q_lora | kv_lora | qk_rope, H] = [768+512+64, 2048]
    CheckBf16(layer.attn.fused_qkv_a_proj, Ql + L + R, H, /*nk=*/true);
    CheckVec(layer.attn.q_a_layernorm, Ql);        // [768]
    CheckBf16(layer.attn.q_b_proj, N * Dqk, Ql, /*nk=*/true);  // [5120, 768]
    CheckVec(layer.attn.kv_a_layernorm, L);
    CheckBf16(layer.attn.kv_b_proj, N * (P + Vd), L, /*nk=*/true);  // [8960, 512]
    CheckBf16(layer.attn.o_proj, H, N * Vd, /*nk=*/true);           // [2048, 5120]
    // --- the LOAD-TIME absorption split ---
    CheckBf16_3D(layer.attn.w_uk_t, N, P, L);  // [20,192,512]
    CheckBf16_3D(layer.attn.w_uv, N, L, Vd);   // [20,512,256]

    if (layer.is_moe) {
      ++moe_layers;
      CHECK(l >= p.first_k_dense_replace);
      CHECK(layer.dense.Empty());
      CheckBf16(layer.moe.router_gate, H, E, /*nk=*/false);
      // noaux_tc -> the e_score_correction_bias IS present (new to e2e here). It
      // is an F32 [E] vector (deepseek_v2.py:313-318 creates it in fp32, and the
      // router reads it in f32) — NOT bf16, so it is checked with the f32 helper.
      REQUIRE(layer.moe.e_score_correction_bias.rank == 1);
      CHECK(layer.moe.e_score_correction_bias.dtype == vt::DType::kF32);
      CHECK(layer.moe.e_score_correction_bias.shape[0] == E);
      REQUIRE(layer.moe.expert_gate.size() == static_cast<size_t>(E));
      REQUIRE(layer.moe.expert_up.size() == static_cast<size_t>(E));
      REQUIRE(layer.moe.expert_down.size() == static_cast<size_t>(E));
      for (int64_t e = 0; e < E; ++e) {
        CheckBf16(layer.moe.expert_gate[static_cast<size_t>(e)], H, I, false);
        CheckBf16(layer.moe.expert_up[static_cast<size_t>(e)], H, I, false);
        CheckBf16(layer.moe.expert_down[static_cast<size_t>(e)], I, H, false);
      }
      // ONE shared expert, MLP of width moe_intermediate_size * n_shared_experts.
      REQUIRE_FALSE(layer.moe.shared.Empty());
      CheckBf16(layer.moe.shared.gate_up_proj, 2 * Ish, H, /*nk=*/true);
      CheckBf16(layer.moe.shared.down_proj, H, Ish, /*nk=*/true);
    } else {
      ++dense_layers;
      CHECK(l < p.first_k_dense_replace);
      CHECK(layer.moe.expert_gate.empty());
      REQUIRE_FALSE(layer.dense.Empty());
      CheckBf16(layer.dense.gate_up_proj, 2 * Is, H, /*nk=*/true);
      CheckBf16(layer.dense.down_proj, H, Is, /*nk=*/true);
    }
  }
  CHECK(dense_layers == 1);   // first_k_dense_replace == 1
  CHECK(moe_layers == 46);    // 47 - 1

  // --- e_score_correction_bias f32 sanity: it must be an f32 [E] vector ---
  {
    const auto& moe = w.layers[static_cast<size_t>(p.first_k_dense_replace)].moe;
    REQUIRE_FALSE(moe.e_score_correction_bias.Empty());
    CHECK(moe.e_score_correction_bias.dtype == vt::DType::kF32);
    CHECK(moe.e_score_correction_bias.shape[0] == E);
  }

  // NO unmapped tensors EXCEPT the MTP tail. Build the set of consumed main-model
  // names, then assert every checkpoint tensor is either consumed OR belongs to
  // the MTP tail block `model.layers.{num_hidden_layers + i}.*` (the ONLY skip).
  std::unordered_set<std::string> expected;
  expected.insert("model.embed_tokens.weight");
  expected.insert("model.norm.weight");
  expected.insert("lm_head.weight");
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    expected.insert(b + "input_layernorm.weight");
    expected.insert(b + "post_attention_layernorm.weight");
    expected.insert(b + "self_attn.q_a_proj.weight");
    expected.insert(b + "self_attn.kv_a_proj_with_mqa.weight");
    expected.insert(b + "self_attn.q_a_layernorm.weight");
    expected.insert(b + "self_attn.q_b_proj.weight");
    expected.insert(b + "self_attn.kv_a_layernorm.weight");
    expected.insert(b + "self_attn.kv_b_proj.weight");
    expected.insert(b + "self_attn.o_proj.weight");
    if (p.is_moe_layer(l)) {
      expected.insert(b + "mlp.gate.weight");
      expected.insert(b + "mlp.gate.e_score_correction_bias");
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

  // The MTP tail prefix (num_nextn_predict_layers == 1 -> layer index 47).
  const std::string mtp_prefix =
      "model.layers." + std::to_string(p.num_hidden_layers) + ".";
  int64_t mtp_skipped = 0;
  for (const std::string& name : actual) {
    if (expected.count(name) == 1) continue;
    // The only legitimate unmapped tensors are the MTP draft layer(s).
    const bool is_mtp = name.rfind(mtp_prefix, 0) == 0;
    CHECK_MESSAGE(is_mtp, "unexpected UNMAPPED checkpoint tensor: " << name);
    if (is_mtp) ++mtp_skipped;
  }
  // Zero MISSING: every expected main-model tensor is present.
  for (const std::string& name : expected)
    CHECK_MESSAGE(actual.count(name) == 1,
                  "expected tensor missing from ckpt: " << name);
  // The MTP tail really existed and was really skipped (non-vacuous).
  CHECK(mtp_skipped > 0);
  MESSAGE("glm4-moe-lite loader: " << expected.size()
          << " main-model tensors mapped, " << mtp_skipped
          << " MTP-tail tensors skipped (num_nextn_predict_layers=1), 0 unmapped, "
             "0 missing");
}
