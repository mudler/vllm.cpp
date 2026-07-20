// W2 loader gate for the DENSE Qwen3 text model (`Qwen3ForCausalLM`,
// Qwen3-0.6B, BF16) — the first additive-model bring-up. Loads the real
// checkpoint from the HF cache and asserts the name map is complete and exact:
// every expected tensor mapped with the right shape, the tied lm_head resolves
// (aliases embed_tokens, checkpoint lm_head.weight intentionally skipped), and
// NO checkpoint tensor is left unmapped/leftover.
//
// On a box without the checkpoint (CI, CPU dev box) the body emits a loud SKIP,
// mirroring the parity engine tests. On dgx the snapshot is present so the load
// actually runs. Pure host load — no GPU, no oracle (the token-exact forward
// gate is W4).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;

using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::LoadQwen3ForCausalLMWeights;
using vllm::OwnedTensor;
using vllm::Qwen3DenseWeights;
using vllm::SafetensorsFile;

namespace {

// Snapshot dir of the Qwen3-0.6B checkpoint (contains config.json), or "".
// Same HF cache layout as the parity engine tests.
std::string FindQwen3_0_6BSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
                         ".cache/huggingface/hub/"
                         "models--Qwen--Qwen3-0.6B/snapshots";
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

TEST_CASE("qwen3 W2 loader: Qwen3-0.6B safetensors -> Qwen3DenseWeights") {
  const std::string snap = FindQwen3_0_6BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "SKIP: Qwen3-0.6B checkpoint absent "
        "(~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B); "
        "run on dgx for the W2 load gate.");
    return;
  }

  const HfConfig config = LoadHfConfig(snap + "/config.json");
  // Config sanity (Qwen3-0.6B): hidden 1024, 28 layers, 16 q / 8 kv heads,
  // head_dim 128, intermediate 3072, vocab 151936.
  CHECK(config.model_type == "qwen3");
  REQUIRE(config.num_hidden_layers == 28);
  CHECK(config.hidden_size == 1024);
  CHECK(config.num_attention_heads == 16);
  CHECK(config.num_key_value_heads == 8);
  CHECK(config.head_dim == 128);
  CHECK(config.intermediate_size == 3072);
  CHECK(config.vocab_size == 151936);

  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(snap + "/model.safetensors"));

  const Qwen3DenseWeights w = LoadQwen3ForCausalLMWeights(shards, config);

  const int64_t H = config.hidden_size;      // 1024
  const int64_t V = config.vocab_size;       // 151936
  const int64_t Dh = config.head_dim;        // 128
  const int64_t Hq = config.num_attention_heads;    // 16
  const int64_t Hkv = config.num_key_value_heads;   // 8
  const int64_t I = config.intermediate_size;       // 3072
  const int64_t q_size = Hq * Dh;                    // 2048
  const int64_t kv_size = Hkv * Dh;                  // 1024
  const int64_t qkv_rows = q_size + 2 * kv_size;     // 4096

  // tie_word_embeddings=true, attention_bias=false for Qwen3-0.6B.
  CHECK(w.tie_word_embeddings);
  CHECK_FALSE(w.attention_bias);

  // Tied lm_head resolves: lm_head is EMPTY (aliases embed_tokens); the
  // checkpoint's redundant lm_head.weight is intentionally skipped.
  CHECK(w.lm_head.Empty());

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
    // Merged gate_up raw-NK [2I, H]; down_proj raw-NK [H, I].
    CheckBf16(layer.mlp.gate_up_proj, 2 * I, H, /*nk=*/true);
    CheckBf16(layer.mlp.down_proj, H, I, /*nk=*/true);
  }

  // NO unmapped/leftover tensors: the set of every checkpoint tensor name must
  // equal exactly {consumed} U {intentionally-skipped lm_head.weight}.
  std::unordered_set<std::string> expected;
  expected.insert("model.embed_tokens.weight");
  expected.insert("model.norm.weight");
  expected.insert("lm_head.weight");  // present in ckpt, skipped (tied)
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
    expected.insert(b + "mlp.gate_proj.weight");
    expected.insert(b + "mlp.up_proj.weight");
    expected.insert(b + "mlp.down_proj.weight");
  }

  std::unordered_set<std::string> actual;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) actual.insert(name);

  // Every checkpoint tensor is accounted for, and nothing extra is expected.
  for (const std::string& name : actual)
    CHECK_MESSAGE(expected.count(name) == 1, "unexpected checkpoint tensor: "
                                                 << name);
  for (const std::string& name : expected)
    CHECK_MESSAGE(actual.count(name) == 1, "expected tensor missing from ckpt: "
                                               << name);
  CHECK(actual.size() == expected.size());  // 311 tensors
}
