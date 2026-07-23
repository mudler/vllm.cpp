// W2 loader gate for Llama (`LlamaForCausalLM`, Llama-3.2-1B, BF16) — the
// cross-family dense additivity bring-up. Loads the real HF snapshot and asserts
// the name map is complete and exact: every expected tensor mapped with the right
// shape (16 layers x {input/post RMSNorm, merged qkv + o_proj, gate_up + down},
// plus embed_tokens + the final RMSNorm), and NO checkpoint tensor left unmapped —
// with `lm_head.weight` the one deliberate skip when present, because Llama-3.2
// ties embeddings (tie_word_embeddings=true; skip_prefixes=["lm_head."],
// llama.py:538).
//
// The Llama-specific loader invariants this pins down vs the Qwen3-dense loader:
//   1. NO per-head q/k norm tensors — Llama has none, so the attn q_norm/k_norm
//      OwnedTensors stay EMPTY (the qk-norm-optional seam).
//   2. TIED lm_head: lm_head is EMPTY, aliasing embed_tokens at forward time.
//   3. GQA 32/8 with head_dim 64 (q dim 2048, kv dim 512) — the merged qkv width.
//
// Checkpoint-GATED + dgx-only: resolves the HF snapshot under
// ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B (ungated mirror of the
// gated meta-llama/Llama-3.2-1B — same weights). On a box without it the body
// emits a loud SKIP. Pure host load — no GPU, no oracle (the token-exact forward
// gate is W4, tests/parity/test_llama_paged_engine.cpp).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/llama.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;

using vllm::HfConfig;
using vllm::LlamaWeights;
using vllm::LoadHfConfig;
using vllm::LoadLlamaForCausalLMWeights;
using vllm::OwnedTensor;
using vllm::SafetensorsFile;

namespace {

std::string FindLlamaSnap() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
      ".cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "model.safetensors", ec)) return e.path().string();
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

TEST_CASE("llama W2 loader: Llama-3.2-1B safetensors -> LlamaWeights (dgx-only)") {
  const std::string dir = FindLlamaSnap();
  if (dir.empty()) {
    MESSAGE("SKIP: Llama-3.2-1B checkpoint absent "
            "(~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B); dgx-only.");
    return;
  }

  const HfConfig config = LoadHfConfig(dir + "/config.json");
  // Config sanity (Llama-3.2-1B): hidden 2048, 16 layers, GQA 32/8, head_dim 64,
  // intermediate 8192, vocab 128256, rope_theta 500000 + llama3 scaling.
  CHECK(config.num_hidden_layers == 16);
  CHECK(config.hidden_size == 2048);
  CHECK(config.num_attention_heads == 32);
  CHECK(config.num_key_value_heads == 8);
  CHECK(config.head_dim == 64);
  CHECK(config.vocab_size == 128256);
  CHECK(config.rope_theta == 500000.0);
  CHECK(config.rotary_dim == 64);  // partial_rotary_factor 1.0 * head_dim
  // The llama3 rope-scaling dictionary parsed + validated by LoadHfConfig.
  CHECK(config.rope_parameters.rope_type == "llama3");
  REQUIRE(config.rope_parameters.factor.has_value());
  CHECK(*config.rope_parameters.factor == 32.0);
  REQUIRE(config.rope_parameters.low_freq_factor.has_value());
  CHECK(*config.rope_parameters.low_freq_factor == 1.0);
  REQUIRE(config.rope_parameters.high_freq_factor.has_value());
  CHECK(*config.rope_parameters.high_freq_factor == 4.0);
  REQUIRE(config.rope_parameters.original_max_position_embeddings.has_value());
  CHECK(*config.rope_parameters.original_max_position_embeddings == 8192);

  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(dir + "/model.safetensors"));

  const LlamaWeights w = LoadLlamaForCausalLMWeights(shards, config);

  const int64_t H = config.hidden_size;                            // 2048
  const int64_t V = config.vocab_size;                             // 128256
  const int64_t I = config.intermediate_size;                      // 8192
  const int64_t qdim = config.num_attention_heads * config.head_dim;   // 2048
  const int64_t kdim = config.num_key_value_heads * config.head_dim;   // 512

  // INVARIANT 2 — TIED: lm_head is EMPTY and aliases embed_tokens at forward time.
  CHECK(w.tie_word_embeddings);
  CHECK(w.lm_head.Empty());
  CHECK_FALSE(w.attention_bias);

  CheckBf16(w.embed_tokens, V, H, /*nk=*/false);
  CheckVec(w.final_norm, H);

  REQUIRE(w.layers.size() == static_cast<size_t>(config.num_hidden_layers));
  for (const auto& layer : w.layers) {
    // Standard (weight-only) RMSNorm, twice per layer — no bias.
    CheckVec(layer.input_layernorm, H);
    CheckVec(layer.post_attention_layernorm, H);
    // Merged QKV raw-NK [qdim + 2*kdim, H] = [3072, 2048]; o_proj [H, qdim].
    CheckBf16(layer.attn.qkv_proj, qdim + 2 * kdim, H, /*nk=*/true);
    CheckBf16(layer.attn.o_proj, H, qdim, /*nk=*/true);
    // INVARIANT 1 — NO per-head q/k norm (Llama has none).
    CHECK(layer.attn.q_norm.Empty());
    CHECK(layer.attn.k_norm.Empty());
    // INVARIANT 3 — no attention bias.
    CHECK(layer.attn.qkv_bias.Empty());
    // Merged gate_up [2I, H] -> SiluAndMul -> down [H, I].
    CheckBf16(layer.mlp.gate_up_proj, 2 * I, H, /*nk=*/true);
    CheckBf16(layer.mlp.down_proj, H, I, /*nk=*/true);
  }

  // NO unmapped/leftover tensors. `lm_head.weight` is the ONE allowed skip (tied);
  // some tied checkpoints omit it entirely, so it is tolerated whether present or
  // not, but no OTHER tensor may be left unaccounted for.
  std::unordered_set<std::string> expected;
  expected.insert("model.embed_tokens.weight");
  expected.insert("model.norm.weight");
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    expected.insert(b + "input_layernorm.weight");
    expected.insert(b + "post_attention_layernorm.weight");
    for (const char* p : {"q_proj", "k_proj", "v_proj", "o_proj"})
      expected.insert(b + "self_attn." + p + ".weight");
    for (const char* p : {"gate_proj", "up_proj", "down_proj"})
      expected.insert(b + "mlp." + p + ".weight");
  }
  const std::unordered_set<std::string> skipped{"lm_head.weight"};

  std::unordered_set<std::string> actual;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) actual.insert(name);

  for (const std::string& name : actual) {
    const bool accounted_for = expected.count(name) == 1 || skipped.count(name) == 1;
    CHECK_MESSAGE(accounted_for, "unexpected checkpoint tensor: " << name);
  }
  for (const std::string& name : expected)
    CHECK_MESSAGE(actual.count(name) == 1, "expected tensor missing: " << name);
  // 2 top-level + 16 layers x 9 = 146 mapped; lm_head.weight optionally present.
  CHECK(actual.size() >= expected.size());
  CHECK(actual.size() <= expected.size() + skipped.size());
}

// DIAGNOSTIC (dgx-only): our tokenizer (Tokenizer::FromHfJson + EncodeWithSpecialTokens,
// exactly what the engine's input_processor uses) must reproduce vLLM's prompt
// tokenization bit-for-bit, INCLUDING the prepended BOS 128000. A tokenization
// mismatch silently forks the whole generation even when the forward is exact.
TEST_CASE("llama tokenizer: EncodeWithSpecialTokens matches vLLM (BOS + splits) (dgx-only)") {
  const std::string dir = FindLlamaSnap();
  if (dir.empty()) { MESSAGE("SKIP: Llama-3.2-1B checkpoint absent"); return; }
  const vllm::tok::Tokenizer tk =
      vllm::tok::Tokenizer::FromHfJson(dir + "/tokenizer.json");
  MESSAGE("tokenizer BosId()=" << tk.BosId());

  struct Case { const char* text; std::vector<int32_t> want; };
  const std::vector<Case> cases = {
      {"The capital of France is", {128000, 791, 6864, 315, 9822, 374}},
      {"The largest planet in our solar system is",
       {128000, 791, 7928, 11841, 304, 1057, 13238, 1887, 374}},
      {"Machine learning is a subfield of",
       {128000, 22333, 6975, 374, 264, 1207, 2630, 315}},
      {"E equals m c", {128000, 36, 17239, 296, 272}},
      {"def fibonacci(n):", {128000, 755, 76798, 1471, 1680}},
  };
  int pass = 0;
  for (const Case& c : cases) {
    const std::vector<int32_t> got = tk.EncodeWithSpecialTokens(c.text);
    std::string gs, ws;
    for (int32_t t : got) gs += std::to_string(t) + " ";
    for (int32_t t : c.want) ws += std::to_string(t) + " ";
    const bool ok = got == c.want;
    if (ok) ++pass;
    MESSAGE("encode \"" << c.text << "\"  got=[" << gs << "] want=[" << ws << "] "
            << (ok ? "OK" : "MISMATCH"));
    CHECK(got == c.want);
  }
  MESSAGE("llama tokenizer: " << pass << "/" << cases.size() << " prompts match vLLM");
}
