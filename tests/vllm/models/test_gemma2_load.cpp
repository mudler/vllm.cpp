// W4 loader gate for `Gemma2ForCausalLM` (gemma-2-2b-it, BF16). Loads the real
// checkpoint from the HF cache and asserts the name map is complete and exact:
// every expected tensor mapped with the right shape (four sandwich norms per
// layer, NO q/k norm, no bias), the tied lm_head resolves (aliases embed_tokens;
// the checkpoint has NO lm_head.weight), and NO checkpoint tensor is left
// unmapped/leftover. On a box without the checkpoint the body emits a loud SKIP.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/gemma2.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;

using vllm::Gemma2Weights;
using vllm::HfConfig;
using vllm::LoadGemma2ForCausalLMWeights;
using vllm::LoadHfConfig;
using vllm::OwnedTensor;
using vllm::SafetensorsFile;

namespace {

std::string FindSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const char* repos[] = {"models--unsloth--gemma-2-2b-it", "models--google--gemma-2-2b-it"};
  for (const char* repo : repos) {
    const fs::path snaps = fs::path(home) / ".cache/huggingface/hub" / repo / "snapshots";
    std::error_code ec;
    if (!fs::is_directory(snaps, ec)) continue;
    for (const auto& e : fs::directory_iterator(snaps, ec))
      if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  }
  return "";
}

std::vector<SafetensorsFile> OpenShards(const std::string& snap) {
  std::vector<SafetensorsFile> shards;
  std::error_code ec;
  const fs::path single = fs::path(snap) / "model.safetensors";
  if (fs::exists(single, ec)) {
    shards.push_back(SafetensorsFile::Open(single.string()));
    return shards;
  }
  for (const auto& e : fs::directory_iterator(snap, ec)) {
    const std::string n = e.path().filename().string();
    if (n.rfind("model-", 0) == 0 && n.size() > 12 &&
        n.substr(n.size() - 12) == ".safetensors")
      shards.push_back(SafetensorsFile::Open(e.path().string()));
  }
  return shards;
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

TEST_CASE("gemma2 W4 loader: gemma-2-2b-it safetensors -> Gemma2Weights") {
  const std::string snap = FindSnapshot();
  if (snap.empty()) {
    MESSAGE("SKIP: gemma-2-2b-it checkpoint absent (run on dgx for the W4 load gate).");
    return;
  }

  const HfConfig config = LoadHfConfig(snap + "/config.json");
  // Config sanity (gemma-2-2b-it): hidden 2304, 26L, GQA 8/4, head_dim 256.
  REQUIRE(config.num_hidden_layers == 26);
  CHECK(config.hidden_size == 2304);
  CHECK(config.num_attention_heads == 8);
  CHECK(config.num_key_value_heads == 4);
  CHECK(config.head_dim == 256);
  CHECK(config.intermediate_size == 9216);

  std::vector<SafetensorsFile> shards = OpenShards(snap);
  REQUIRE_FALSE(shards.empty());
  const Gemma2Weights w = LoadGemma2ForCausalLMWeights(shards, config);

  const int64_t H = config.hidden_size;
  const int64_t V = config.vocab_size;
  const int64_t Dh = config.head_dim;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t I = config.intermediate_size;
  const int64_t q_size = Hq * Dh, kv_size = Hkv * Dh;
  const int64_t qkv_rows = q_size + 2 * kv_size;

  CHECK(w.tie_word_embeddings);
  CHECK(w.lm_head.Empty());
  CheckBf16(w.embed_tokens, V, H, /*nk=*/false);
  CheckVec(w.final_norm, H);

  REQUIRE(w.layers.size() == static_cast<size_t>(config.num_hidden_layers));
  for (const auto& layer : w.layers) {
    CheckVec(layer.input_layernorm, H);
    CheckVec(layer.post_attention_layernorm, H);
    CheckVec(layer.pre_feedforward_layernorm, H);
    CheckVec(layer.post_feedforward_layernorm, H);
    CheckBf16(layer.attn.qkv_proj, qkv_rows, H, /*nk=*/true);
    CheckBf16(layer.attn.o_proj, H, q_size, /*nk=*/true);
    CheckBf16(layer.mlp.gate_up_proj, 2 * I, H, /*nk=*/true);
    CheckBf16(layer.mlp.down_proj, H, I, /*nk=*/true);
  }

  // NO unmapped/leftover tensors (Gemma-2 has NO q/k norm; ties + omits lm_head).
  std::unordered_set<std::string> expected;
  expected.insert("model.embed_tokens.weight");
  expected.insert("model.norm.weight");
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    expected.insert(b + "input_layernorm.weight");
    expected.insert(b + "post_attention_layernorm.weight");
    expected.insert(b + "pre_feedforward_layernorm.weight");
    expected.insert(b + "post_feedforward_layernorm.weight");
    expected.insert(b + "self_attn.q_proj.weight");
    expected.insert(b + "self_attn.k_proj.weight");
    expected.insert(b + "self_attn.v_proj.weight");
    expected.insert(b + "self_attn.o_proj.weight");
    expected.insert(b + "mlp.gate_proj.weight");
    expected.insert(b + "mlp.up_proj.weight");
    expected.insert(b + "mlp.down_proj.weight");
  }

  std::unordered_set<std::string> actual;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) actual.insert(name);

  for (const std::string& name : actual)
    CHECK_MESSAGE(expected.count(name) == 1, "unexpected checkpoint tensor: " << name);
  for (const std::string& name : expected)
    CHECK_MESSAGE(actual.count(name) == 1, "expected tensor missing from ckpt: " << name);
  CHECK(actual.size() == expected.size());
}
