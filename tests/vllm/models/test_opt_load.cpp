// W2 loader gate for OPT (`OPTForCausalLM`, facebook/opt-125m, BF16) — the
// CROSS-FAMILY additivity canary. Loads the materialized single-shard
// checkpoint and asserts the name map is complete and exact: every expected
// tensor mapped with the right shape (12 layers x {merged qkv + its merged bias,
// out_proj + bias, fc1/fc2 + biases, 2 LayerNorms with weight AND bias}, plus
// embed_tokens, the LEARNED embed_positions table, the decoder final LayerNorm)
// and NO checkpoint tensor left unmapped — with `lm_head.weight` the one
// deliberate skip, because OPT ties embeddings by default (opt.py:352-353,
// skip_prefixes opt.py:390-392).
//
// The three OPT-specific loader invariants this pins down (each a first for
// this tree):
//   1. embed_positions is [max_position_embeddings + 2, H] — the fairseq
//      padding-idx OFFSET OF 2 (OPTLearnedPositionalEmbedding, opt.py:59-68).
//      An off-by-two table would silently mis-position every token.
//   2. Every projection carries a BIAS (`config.enable_bias`), and the merged
//      qkv bias is concatenated in the SAME [q,k,v] order as the merged weight.
//   3. Both LayerNorms carry weight AND bias — a mean-subtracting norm, not the
//      weight-only RMSNorm every Qwen model in the tree uses.
//
// Checkpoint-GATED: resolves the materialized bf16-safetensors dir produced by
// scripts/opt-materialize-checkpoint.py (the HF snapshot ships a torch-pickle
// `pytorch_model.bin`, which our safetensors loader cannot read — see
// .agents/specs/sweep-opt-125m.md decision D2). On a box without it the body
// emits a loud SKIP. Pure host load — no GPU, no oracle (the token-exact
// forward gate is W4, test_opt_paged_engine.cpp).
//
// Upstream test mirrored: `tests/models/registry.py:448` (the opt-125m registry
// fixture) + the load half of `tests/models/language/generation/
// test_common.py:77`.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/opt.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;

using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::LoadOPTForCausalLMWeights;
using vllm::OPTWeights;
using vllm::OwnedTensor;
using vllm::SafetensorsFile;

namespace {

// The materialized bf16-safetensors OPT-125m dir (config.json +
// model.safetensors + tokenizer.json). Overridable for a non-default location.
std::string FindOptModelDir() {
  if (const char* env = std::getenv("VLLM_CPP_OPT_MODEL_DIR"); env != nullptr) {
    std::error_code ec;
    if (fs::exists(fs::path(env) / "config.json", ec)) return env;
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path dir = fs::path(home) / "models/opt-125m-bf16-st";
  std::error_code ec;
  if (fs::exists(dir / "config.json", ec) && fs::exists(dir / "model.safetensors", ec))
    return dir.string();
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

TEST_CASE("opt W2 loader: facebook/opt-125m safetensors -> OPTWeights") {
  const std::string dir = FindOptModelDir();
  if (dir.empty()) {
    MESSAGE(
        "SKIP: materialized OPT-125m checkpoint absent (~/models/opt-125m-bf16-st "
        "or $VLLM_CPP_OPT_MODEL_DIR); build it with "
        "scripts/opt-materialize-checkpoint.py and run on dgx for the W2 load gate.");
    return;
  }

  const HfConfig config = LoadHfConfig(dir + "/config.json");
  // Config sanity (facebook/opt-125m): hidden 768, 12 layers, 12 heads (no GQA),
  // head_dim 64, ffn_dim 3072, vocab 50272, max_position_embeddings 2048.
  CHECK(config.num_hidden_layers == 12);
  CHECK(config.hidden_size == 768);
  CHECK(config.num_attention_heads == 12);
  CHECK(config.num_key_value_heads == 12);  // OPT predates GQA
  CHECK(config.head_dim == 64);
  CHECK(config.vocab_size == 50272);
  CHECK(config.max_position_embeddings == 2048);
  // SEAM NOTE (spec "seams that leaked" L3): OPT has NO rotary embedding at
  // all, yet the shared HfConfig UNCONDITIONALLY synthesizes RoPE fields from
  // defaults — `rotary_dim` comes out as partial_rotary_factor(1.0) * head_dim
  // = 64 and `rope_theta` as 10000, for a model that has neither. That is
  // harmless here only because the OPT forward never reads them (it is the
  // reason OPT cannot reuse dense_attn::BuildStepInputs, which exists to build
  // a RoPE cos|sin cache). Pinned as documentation of the leak, not as an
  // endorsement: a config seam that presumes RoPE is a latent trap for the next
  // no-RoPE family.
  CHECK(config.rotary_dim == 64);
  CHECK(config.rope_theta == 10000.0);

  const vllm::OPTConfigExtras extras = vllm::GetOPTConfigExtras(config);
  CHECK(extras.ffn_dim == 3072);
  CHECK(extras.word_embed_proj_dim == 768);
  CHECK(extras.do_layer_norm_before);  // 125m is PRE-LN
  CHECK(extras.enable_bias);
  CHECK(extras.tie_word_embeddings);
  CHECK(extras.activation_function == "relu");

  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(dir + "/model.safetensors"));

  const OPTWeights w = LoadOPTForCausalLMWeights(shards, config);

  const int64_t H = config.hidden_size;                 // 768
  const int64_t V = config.vocab_size;                  // 50272
  const int64_t P = config.max_position_embeddings;     // 2048
  const int64_t F = extras.ffn_dim;                     // 3072

  CHECK(w.tie_word_embeddings);
  CHECK(w.do_layer_norm_before);
  // TIED: lm_head is EMPTY and aliases embed_tokens at forward time.
  CHECK(w.lm_head.Empty());

  CheckBf16(w.embed_tokens, V, H, /*nk=*/false);
  // INVARIANT 1 — the learned position table is [max_pos + 2, H]: the
  // OPTLearnedPositionalEmbedding offset of 2.
  CheckBf16(w.embed_positions, P + 2, H, /*nk=*/false);
  // Decoder-level final LayerNorm present (do_layer_norm_before && !
  // _remove_final_layer_norm), with BOTH weight and bias.
  CheckVec(w.final_layer_norm, H);
  CheckVec(w.final_layer_norm_bias, H);

  REQUIRE(w.layers.size() == static_cast<size_t>(config.num_hidden_layers));
  for (const auto& layer : w.layers) {
    // INVARIANT 3 — LayerNorm carries weight AND bias, twice per layer.
    CheckVec(layer.self_attn_layer_norm, H);
    CheckVec(layer.self_attn_layer_norm_bias, H);
    CheckVec(layer.final_layer_norm, H);
    CheckVec(layer.final_layer_norm_bias, H);

    // Merged QKV raw-NK [3H, H] (multi-head: q/k/v are all H wide).
    CheckBf16(layer.attn.qkv_proj, 3 * H, H, /*nk=*/true);
    CheckBf16(layer.attn.out_proj, H, H, /*nk=*/true);
    // INVARIANT 2 — every projection is BIASED; the merged qkv bias is [3H].
    CheckVec(layer.attn.qkv_bias, 3 * H);
    CheckVec(layer.attn.out_bias, H);

    // Plain fc1 -> ReLU -> fc2 (no gate/up merge — there is no gating).
    CheckBf16(layer.mlp.fc1, F, H, /*nk=*/true);
    CheckBf16(layer.mlp.fc2, H, F, /*nk=*/true);
    CheckVec(layer.mlp.fc1_bias, F);
    CheckVec(layer.mlp.fc2_bias, H);
  }

  // NO unmapped/leftover tensors. `lm_head.weight` is the ONE deliberate skip
  // (tie_word_embeddings), mirroring vLLM's skip_prefixes (opt.py:390-392).
  std::unordered_set<std::string> expected;
  expected.insert("model.decoder.embed_tokens.weight");
  expected.insert("model.decoder.embed_positions.weight");
  expected.insert("model.decoder.final_layer_norm.weight");
  expected.insert("model.decoder.final_layer_norm.bias");
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string b = "model.decoder.layers." + std::to_string(l) + ".";
    for (const char* ln : {"self_attn_layer_norm", "final_layer_norm"}) {
      expected.insert(b + ln + ".weight");
      expected.insert(b + ln + ".bias");
    }
    for (const char* p : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
      expected.insert(b + "self_attn." + p + ".weight");
      expected.insert(b + "self_attn." + p + ".bias");
    }
    for (const char* p : {"fc1", "fc2"}) {
      expected.insert(b + p + ".weight");
      expected.insert(b + p + ".bias");
    }
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
  // 197 = 5 top-level (incl. the skipped tied lm_head) + 12 layers x 16.
  CHECK(actual.size() == expected.size() + skipped.size());
  CHECK(actual.size() == 197);
}
