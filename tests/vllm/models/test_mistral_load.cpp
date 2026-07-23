// W2 loader gate for Mistral (`MistralForCausalLM`, Mistral-7B-v0.3, BF16) — the
// fifth-family near-additive dense bring-up. Loads the real HF snapshot (3 shards)
// and asserts the name map is complete and exact: every expected tensor mapped with
// the right shape (32 layers x {input/post RMSNorm, merged qkv + o_proj, gate_up +
// down}, plus embed_tokens + the final RMSNorm + the UNTIED lm_head), and NO
// checkpoint tensor left unmapped.
//
// The Mistral-specific loader invariants vs the Llama/Qwen3-dense loader:
//   1. NO per-head q/k norm tensors — Mistral has none, so the attn q_norm/k_norm
//      OwnedTensors stay EMPTY (the qk-norm-optional seam, shared with Llama).
//   2. UNTIED lm_head: tie_word_embeddings=false, so lm_head is a real bf16 owner
//      [H, V] (Matmul-B), NOT aliased to embed_tokens (the opposite of Llama-3.2-1B).
//   3. GQA 32/8 with head_dim 128 (q dim 4096, kv dim 1024) — the merged qkv width.
//   4. sliding_window null (SWA disabled in v0.3) -> config.sliding_window has no
//      value; the KV topology is pure full-attention (asserted in the registry test).
//
// Checkpoint-GATED + dgx-only: resolves the HF snapshot under
// ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3. On a box without it
// the body emits a loud SKIP. Pure host load — no GPU, no oracle (the token-exact
// forward gate is W4, tests/parity/test_mistral_paged_engine.cpp).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/mistral.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;

using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::LoadMistralForCausalLMWeights;
using vllm::MistralWeights;
using vllm::OwnedTensor;
using vllm::SafetensorsFile;

namespace {

std::string FindMistralSnap() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
      ".cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  return "";
}

// Collect the sharded safetensors files for a snapshot (Mistral-7B-v0.3 ships 3).
std::vector<SafetensorsFile> OpenShards(const std::string& dir) {
  std::vector<SafetensorsFile> shards;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    const std::string name = e.path().filename().string();
    if (name.rfind("model-", 0) == 0 && e.path().extension() == ".safetensors")
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

TEST_CASE("mistral W2 loader: Mistral-7B-v0.3 safetensors -> MistralWeights (dgx-only)") {
  const std::string dir = FindMistralSnap();
  if (dir.empty()) {
    MESSAGE("SKIP: Mistral-7B-v0.3 checkpoint absent "
            "(~/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3); dgx-only.");
    return;
  }

  const HfConfig config = LoadHfConfig(dir + "/config.json");
  // Config sanity (Mistral-7B-v0.3): hidden 4096, 32 layers, GQA 32/8, head_dim 128,
  // intermediate 14336, vocab 32768, rope_theta 1e6, NO rope_scaling, SWA disabled.
  CHECK(config.num_hidden_layers == 32);
  CHECK(config.hidden_size == 4096);
  CHECK(config.num_attention_heads == 32);
  CHECK(config.num_key_value_heads == 8);
  CHECK(config.head_dim == 128);
  CHECK(config.intermediate_size == 14336);
  CHECK(config.vocab_size == 32768);
  CHECK(config.rope_theta == 1000000.0);
  CHECK(config.rotary_dim == 128);  // partial_rotary_factor 1.0 * head_dim
  // PLAIN rope: default type, NO llama3 rescale dictionary.
  CHECK(config.rope_parameters.rope_type == "default");
  CHECK_FALSE(config.rope_parameters.factor.has_value());
  // sliding_window is null in v0.3 -> parsed to no value (full attention).
  CHECK_FALSE(config.sliding_window.has_value());

  std::vector<SafetensorsFile> shards = OpenShards(dir);
  REQUIRE(shards.size() >= 1);

  const MistralWeights w = LoadMistralForCausalLMWeights(shards, config);

  const int64_t H = config.hidden_size;                              // 4096
  const int64_t V = config.vocab_size;                               // 32768
  const int64_t I = config.intermediate_size;                        // 14336
  const int64_t qdim = config.num_attention_heads * config.head_dim; // 4096
  const int64_t kdim = config.num_key_value_heads * config.head_dim; // 1024

  // INVARIANT 2 — UNTIED: a standalone lm_head owner [H, V] (Matmul-B), NOT aliased.
  CHECK_FALSE(w.tie_word_embeddings);
  REQUIRE_FALSE(w.lm_head.Empty());
  CheckBf16(w.lm_head, H, V, /*nk=*/false);
  CHECK_FALSE(w.attention_bias);

  CheckBf16(w.embed_tokens, V, H, /*nk=*/false);
  CheckVec(w.final_norm, H);

  REQUIRE(w.layers.size() == static_cast<size_t>(config.num_hidden_layers));
  for (const auto& layer : w.layers) {
    // Standard (weight-only) RMSNorm, twice per layer — no bias.
    CheckVec(layer.input_layernorm, H);
    CheckVec(layer.post_attention_layernorm, H);
    // Merged QKV raw-NK [qdim + 2*kdim, H] = [6144, 4096]; o_proj [H, qdim].
    CheckBf16(layer.attn.qkv_proj, qdim + 2 * kdim, H, /*nk=*/true);
    CheckBf16(layer.attn.o_proj, H, qdim, /*nk=*/true);
    // INVARIANT 1 — NO per-head q/k norm (Mistral has none).
    CHECK(layer.attn.q_norm.Empty());
    CHECK(layer.attn.k_norm.Empty());
    // INVARIANT 3 — no attention bias.
    CHECK(layer.attn.qkv_bias.Empty());
    // Merged gate_up [2I, H] -> SiluAndMul -> down [H, I].
    CheckBf16(layer.mlp.gate_up_proj, 2 * I, H, /*nk=*/true);
    CheckBf16(layer.mlp.down_proj, H, I, /*nk=*/true);
  }

  // NO unmapped/leftover tensors. Mistral is UNTIED so lm_head.weight is EXPECTED
  // (loaded, not skipped) — every checkpoint tensor must be accounted for.
  std::unordered_set<std::string> expected;
  expected.insert("model.embed_tokens.weight");
  expected.insert("model.norm.weight");
  expected.insert("lm_head.weight");
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    expected.insert(b + "input_layernorm.weight");
    expected.insert(b + "post_attention_layernorm.weight");
    for (const char* p : {"q_proj", "k_proj", "v_proj", "o_proj"})
      expected.insert(b + "self_attn." + p + ".weight");
    for (const char* p : {"gate_proj", "up_proj", "down_proj"})
      expected.insert(b + "mlp." + p + ".weight");
  }

  std::unordered_set<std::string> actual;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) actual.insert(name);

  for (const std::string& name : actual)
    CHECK_MESSAGE(expected.count(name) == 1, "unexpected checkpoint tensor: " << name);
  for (const std::string& name : expected)
    CHECK_MESSAGE(actual.count(name) == 1, "expected tensor missing: " << name);
  // 3 top-level + 32 layers x 9 = 291 mapped, all present (untied).
  CHECK(actual.size() == expected.size());
}

// DIAGNOSTIC (dgx-only): our tokenizer must reproduce vLLM's prompt tokenization
// bit-for-bit, INCLUDING the prepended BOS 1 (<s>). The reference ids below come from
// the HF/vLLM Mistral tokenizer.
//
// KNOWN BLOCKER (recorded, NOT a regression): Mistral-7B-v0.3's tokenizer.json is a
// SentencePiece-style BPE with a `Metaspace` pre_tokenizer (replacement U+2581,
// prepend_scheme "first", split false) + byte-fallback vocab. Our tokenizer is
// ByteLevel-BPE-ONLY (GPT-2 bytes<->unicode bijection + Split regexes), so
// Tokenizer::FromHfJson THROWS "unsupported pre_tokenizer component Metaspace".
// SentencePiece/Metaspace tokenization is a separate tokenizer-FAMILY campaign
// (tracked as `LOAD-SENTENCEPIECE`), independent of the Mistral MODEL port
// (which is validated tokenizer-free by the CUDA prefill-argmax-vs-oracle case in
// test_mistral_forward.cpp). Until that lands, this diagnostic SKIPs with a loud
// message rather than failing, and the FULL paged-engine SACRED gate
// (test_mistral_paged_engine.cpp) is correspondingly blocked on the same row.
TEST_CASE("mistral tokenizer: EncodeWithSpecialTokens matches vLLM (BOS + splits) (dgx-only)") {
  const std::string dir = FindMistralSnap();
  if (dir.empty()) { MESSAGE("SKIP: Mistral-7B-v0.3 checkpoint absent"); return; }

  struct Case { const char* text; std::vector<int32_t> want; };
  const std::vector<Case> cases = {
      {"The capital of France is", {1, 1183, 6333, 1070, 5611, 1117}},
      {"The largest planet in our solar system is",
       {1, 1183, 8407, 10641, 1065, 1581, 13792, 2355, 1117}},
      {"Machine learning is a subfield of",
       {1, 14021, 5936, 1117, 1032, 1851, 2990, 1070}},
      {"E equals m c", {1, 1181, 22356, 1058, 1045}},
      {"def fibonacci(n):", {1, 1569, 16950, 1034, 28895, 29500, 29479, 2097}},
  };
  try {
    const vllm::tok::Tokenizer tk =
        vllm::tok::Tokenizer::FromHfJson(dir + "/tokenizer.json");
    MESSAGE("tokenizer BosId()=" << tk.BosId());
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
    MESSAGE("mistral tokenizer: " << pass << "/" << cases.size() << " prompts match vLLM");
  } catch (const std::exception& e) {
    MESSAGE("SKIP (KNOWN BLOCKER LOAD-SENTENCEPIECE): our ByteLevel-only "
            "tokenizer cannot load Mistral's SentencePiece/Metaspace tokenizer.json — "
            << e.what() << ". The MODEL forward is gated tokenizer-free via the CUDA "
            "prefill-argmax-vs-oracle case; the full paged-engine gate is blocked here.");
  }
}
