// W2 loader gate for `Gemma3ForCausalLM` (gemma-3-1b-it, BF16) — the first
// Gemma-family bring-up. Loads the real checkpoint from the HF cache and asserts
// the name map is complete and exact: every expected tensor mapped with the right
// shape (four sandwich norms per layer, per-head q/k norm [head_dim], no bias),
// the tied lm_head resolves (aliases embed_tokens; the checkpoint has NO
// lm_head.weight), and NO checkpoint tensor is left unmapped/leftover (340).
//
// On a box without the checkpoint the body emits a loud SKIP. Pure host load —
// no GPU, no oracle (the token-exact forward gate is the paged-engine test).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/gemma3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;

using vllm::Gemma3Weights;
using vllm::HfConfig;
using vllm::LoadGemma3ForCausalLMWeights;
using vllm::LoadHfConfig;
using vllm::OwnedTensor;
using vllm::SafetensorsFile;

namespace {

std::string FindGemma3_1BSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
                         ".cache/huggingface/hub/"
                         "models--google--gemma-3-1b-it/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec) &&
        fs::exists(e.path() / "model.safetensors", ec)) {
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

TEST_CASE("gemma3 W2 loader: gemma-3-1b-it safetensors -> Gemma3Weights") {
  const std::string snap = FindGemma3_1BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "SKIP: gemma-3-1b-it checkpoint absent "
        "(~/.cache/huggingface/hub/models--google--gemma-3-1b-it); "
        "run on dgx for the W2 load gate.");
    return;
  }

  const HfConfig config = LoadHfConfig(snap + "/config.json");
  // Config sanity (gemma-3-1b-it): hidden 1152, 26 layers, 4 q / 1 kv heads,
  // head_dim 256, intermediate 6912, vocab 262144.
  REQUIRE(config.num_hidden_layers == 26);
  CHECK(config.hidden_size == 1152);
  CHECK(config.num_attention_heads == 4);
  CHECK(config.num_key_value_heads == 1);
  CHECK(config.head_dim == 256);
  CHECK(config.intermediate_size == 6912);
  CHECK(config.vocab_size == 262144);

  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(snap + "/model.safetensors"));

  const Gemma3Weights w = LoadGemma3ForCausalLMWeights(shards, config);

  const int64_t H = config.hidden_size;             // 1152
  const int64_t V = config.vocab_size;              // 262144
  const int64_t Dh = config.head_dim;               // 256
  const int64_t Hq = config.num_attention_heads;    // 4
  const int64_t Hkv = config.num_key_value_heads;   // 1
  const int64_t I = config.intermediate_size;       // 6912
  const int64_t q_size = Hq * Dh;                    // 1024
  const int64_t kv_size = Hkv * Dh;                  // 256
  const int64_t qkv_rows = q_size + 2 * kv_size;     // 1536

  // Gemma ties embeddings; lm_head is EMPTY (aliases embed_tokens). The
  // checkpoint has NO lm_head.weight at all.
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
    CheckVec(layer.attn.q_norm, Dh);
    CheckVec(layer.attn.k_norm, Dh);
    CheckBf16(layer.mlp.gate_up_proj, 2 * I, H, /*nk=*/true);
    CheckBf16(layer.mlp.down_proj, H, I, /*nk=*/true);
  }

  // NO unmapped/leftover tensors: the set of every checkpoint tensor name must
  // equal exactly the consumed set (Gemma ties + omits lm_head.weight entirely).
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
    expected.insert(b + "self_attn.q_norm.weight");
    expected.insert(b + "self_attn.k_norm.weight");
    expected.insert(b + "mlp.gate_proj.weight");
    expected.insert(b + "mlp.up_proj.weight");
    expected.insert(b + "mlp.down_proj.weight");
  }

  std::unordered_set<std::string> actual;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) actual.insert(name);

  for (const std::string& name : actual)
    CHECK_MESSAGE(expected.count(name) == 1, "unexpected checkpoint tensor: "
                                                 << name);
  for (const std::string& name : expected)
    CHECK_MESSAGE(actual.count(name) == 1, "expected tensor missing from ckpt: "
                                               << name);
  CHECK(actual.size() == expected.size());  // 340 tensors
}
