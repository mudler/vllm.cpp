// W2 loader gate for the full-attention MoE Qwen3-Coder text model
// (`Qwen3MoeForCausalLM`, Qwen3-Coder-30B-A3B-Instruct, BF16) — the first
// full-attention MoE bring-up. Loads the real 16-shard checkpoint from the HF
// cache and asserts the name map is complete and exact: every expected tensor
// mapped with the right shape (attention + 128 experts × 48 layers + router +
// embed + norm + UNTIED lm_head), and NO checkpoint tensor left unmapped/leftover.
//
// On a box without the checkpoint (CI, CPU dev box) the body emits a loud SKIP,
// mirroring test_qwen3_load.cpp. On dgx the snapshot is present so the load
// actually runs. Pure host load — no GPU, no oracle (the token-exact forward gate
// is W4).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_moe.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;

using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::LoadQwen3MoeForCausalLMWeights;
using vllm::LoadSafetensorsIndex;
using vllm::OwnedTensor;
using vllm::Qwen3MoeWeights;
using vllm::SafetensorsFile;

namespace {

// Snapshot dir of the Qwen3-Coder-30B-A3B checkpoint (contains config.json +
// model.safetensors.index.json), or "". Same HF cache layout as the parity tests.
std::string FindQwen3CoderSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/"
      "models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots";
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
void CheckVec(const OwnedTensor& t, int64_t n) {
  REQUIRE(t.rank == 1);
  CHECK(t.dtype == vt::DType::kBF16);
  CHECK(t.shape[0] == n);
}

}  // namespace

TEST_CASE("qwen3-moe W2 loader: Qwen3-Coder-30B-A3B safetensors -> Qwen3MoeWeights") {
  const std::string snap = FindQwen3CoderSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "SKIP: Qwen3-Coder-30B-A3B checkpoint absent "
        "(~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct); "
        "run on dgx for the W2 load gate.");
    return;
  }

  const HfConfig config = LoadHfConfig(snap + "/config.json");
  // Config sanity (Qwen3-Coder-30B-A3B): hidden 2048, 48 layers, 32 q / 4 kv
  // heads, head_dim 128, 128 experts top-8, moe_intermediate 768, vocab 151936.
  CHECK(config.num_hidden_layers == 48);
  CHECK(config.hidden_size == 2048);
  CHECK(config.num_attention_heads == 32);
  CHECK(config.num_key_value_heads == 4);
  CHECK(config.head_dim == 128);
  CHECK(config.num_experts == 128);
  CHECK(config.num_experts_per_tok == 8);
  CHECK(config.moe_intermediate_size == 768);
  CHECK(config.shared_expert_intermediate_size == 0);  // NO shared expert
  CHECK(config.vocab_size == 151936);

  // Open all shards named by the index (16 shards).
  const std::map<std::string, std::string> wmap =
      LoadSafetensorsIndex(snap + "/model.safetensors.index.json");
  std::unordered_set<std::string> shard_files;
  for (const auto& [tensor, file] : wmap) shard_files.insert(file);
  std::vector<SafetensorsFile> shards;
  shards.reserve(shard_files.size());
  for (const std::string& f : shard_files)
    shards.push_back(SafetensorsFile::Open(snap + "/" + f));

  const Qwen3MoeWeights w = LoadQwen3MoeForCausalLMWeights(shards, config);

  const int64_t H = config.hidden_size;               // 2048
  const int64_t V = config.vocab_size;                // 151936
  const int64_t Dh = config.head_dim;                 // 128
  const int64_t Hq = config.num_attention_heads;      // 32
  const int64_t Hkv = config.num_key_value_heads;     // 4
  const int64_t E = config.num_experts;               // 128
  const int64_t I = config.moe_intermediate_size;     // 768
  const int64_t q_size = Hq * Dh;                     // 4096
  const int64_t kv_size = Hkv * Dh;                   // 512
  const int64_t qkv_rows = q_size + 2 * kv_size;      // 5120

  // tie_word_embeddings=false (UNTIED), attention_bias=false for Qwen3-Coder.
  CHECK_FALSE(w.tie_word_embeddings);
  CHECK_FALSE(w.attention_bias);

  // UNTIED lm_head: the SEPARATE lm_head.weight is loaded (NOT aliased to
  // embed_tokens), stored as Matmul-B [H, vocab].
  REQUIRE_FALSE(w.lm_head.Empty());
  CheckBf16(w.lm_head, H, V, /*nk=*/false);

  // Model-level tensors.
  CheckBf16(w.embed_tokens, V, H, /*nk=*/false);  // [vocab, H] embed lookup
  CheckVec(w.final_norm, H);

  REQUIRE(w.layers.size() == static_cast<size_t>(config.num_hidden_layers));
  for (const auto& layer : w.layers) {
    CheckVec(layer.input_layernorm, H);
    CheckVec(layer.post_attention_layernorm, H);
    // Merged QKV raw-NK [q+k+v rows, H]; per-head q/k norm [Dh]; o_proj raw-NK.
    CheckBf16(layer.attn.qkv_proj, qkv_rows, H, /*nk=*/true);
    CheckBf16(layer.attn.o_proj, H, q_size, /*nk=*/true);
    CheckVec(layer.attn.q_norm, Dh);
    CheckVec(layer.attn.k_norm, Dh);
    CHECK(layer.attn.qkv_bias.Empty());  // attention_bias=false
    // Router gate [H,E] (Matmul-B); NO shared expert (all shared_* empty).
    CheckBf16(layer.moe.router_gate, H, E, /*nk=*/false);
    CHECK(layer.moe.shared_gate.Empty());
    CHECK(layer.moe.shared_gate_proj.Empty());
    CHECK(layer.moe.shared_up_proj.Empty());
    CHECK(layer.moe.shared_down_proj.Empty());
    // 128 per-expert gate/up [H,I] + down [I,H], all bf16 Matmul-B; no fp4.
    REQUIRE(layer.moe.expert_gate.size() == static_cast<size_t>(E));
    REQUIRE(layer.moe.expert_up.size() == static_cast<size_t>(E));
    REQUIRE(layer.moe.expert_down.size() == static_cast<size_t>(E));
    CHECK(layer.moe.expert_gate_fp4.empty());
    CHECK(layer.moe.expert_up_fp4.empty());
    CHECK(layer.moe.expert_down_fp4.empty());
    for (int64_t e = 0; e < E; ++e) {
      CheckBf16(layer.moe.expert_gate[static_cast<size_t>(e)], H, I, /*nk=*/false);
      CheckBf16(layer.moe.expert_up[static_cast<size_t>(e)], H, I, /*nk=*/false);
      CheckBf16(layer.moe.expert_down[static_cast<size_t>(e)], I, H, /*nk=*/false);
    }
  }

  // NO unmapped/leftover tensors: the set of every checkpoint tensor name must
  // equal exactly the set of consumed names (untied → lm_head.weight is CONSUMED,
  // not skipped).
  std::unordered_set<std::string> expected;
  expected.insert("model.embed_tokens.weight");
  expected.insert("model.norm.weight");
  expected.insert("lm_head.weight");  // untied — consumed
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    expected.insert(b + "input_layernorm.weight");
    expected.insert(b + "post_attention_layernorm.weight");
    expected.insert(b + "self_attn.q_proj.weight");
    expected.insert(b + "self_attn.k_proj.weight");
    expected.insert(b + "self_attn.v_proj.weight");
    expected.insert(b + "self_attn.o_proj.weight");
    expected.insert(b + "self_attn.q_norm.weight");
    expected.insert(b + "self_attn.k_norm.weight");
    expected.insert(b + "mlp.gate.weight");
    for (int64_t e = 0; e < E; ++e) {
      const std::string ex = b + "mlp.experts." + std::to_string(e) + ".";
      expected.insert(ex + "gate_proj.weight");
      expected.insert(ex + "up_proj.weight");
      expected.insert(ex + "down_proj.weight");
    }
  }

  std::unordered_set<std::string> actual;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) actual.insert(name);

  // Every checkpoint tensor is accounted for, and nothing extra is expected.
  for (const std::string& name : actual)
    CHECK_MESSAGE(expected.count(name) == 1,
                  "unexpected checkpoint tensor: " << name);
  for (const std::string& name : expected)
    CHECK_MESSAGE(actual.count(name) == 1,
                  "expected tensor missing from ckpt: " << name);
  CHECK(actual.size() == expected.size());  // 18867 tensors
}
