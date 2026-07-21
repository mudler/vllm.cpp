// W2 loader gate for Qwen3-32B-NVFP4A16 (`Qwen3ForCausalLM` + compressed-tensors
// **NVFP4A16** / W4A16) — the QUANT-SCHEME additivity proof. Loads the 5-shard
// HF snapshot and asserts the name map is complete and exact under the QUANTIZED
// naming, that every quantized operand has the right dtype/shape, and that NO
// checkpoint tensor is left unmapped.
//
// The loader invariants this pins down (each a first for this tree — the dense
// Qwen3 loader was BF16-only before this row):
//   1. SCHEME PROBE. A compressed-tensors NVFP4 Linear stores `<proj>.weight_packed`
//      (U8 [N,K/2]) + `<proj>.weight_scale` (F8_E4M3 [N,K/16]) +
//      `<proj>.weight_global_scale` (F32 [1]) instead of `<proj>.weight`. Norms
//      and the embed table stay BF16 (they are not Linears, so no config group
//      targets them), and `lm_head` is in the checkpoint's `ignore` list — so it
//      is BF16 too, and `tie_word_embeddings` is false here (unlike 0.6B).
//   2. W4A16, NOT W4A4. There is NO `<proj>.input_global_scale` anywhere. That
//      absence is exactly what selects vLLM's
//      `CompressedTensorsW4A4Fp4(use_a16=True)` (compressed_tensors.py:696-698,
//      the `input_activations is None` branch) and, downstream, the FORCED
//      Marlin kernel (kernels/linear/__init__.py:879-881). Our loader must leave
//      `alpha == 0` so `Nvfp4Weight::IsTrueW4A4()` is false and the weight routes
//      to the W4A16 dispatcher, never the 27B's fp4-activation path.
//   3. DIVISOR CONVENTION + MERGED-SHARD COLLAPSE. compressed-tensors stores the
//      global scale as a DIVISOR, so scale2 is its reciprocal
//      (compressed_tensors_w4a4_nvfp4.py:111-114), and a MERGED linear (qkv,
//      gate_up) takes `max()` across the shards' divisors BEFORE reciprocating.
//      This test asserts the merged qkv's scale2 is the reciprocal of the max of
//      the three source divisors — the arithmetic vLLM performs.
//
// Checkpoint-GATED: on a box without the RedHatAI/Qwen3-32B-NVFP4A16 snapshot the
// body emits a loud SKIP. Pure host load — no GPU, no oracle (the token-exact
// forward gate is W4, test_qwen3_32b_nvfp4a16_paged_engine.cpp).
//
// Upstream test mirrored: `tests/quantization/test_compressed_tensors.py`
// (the scheme-selection + weight-shape assertions for the NVFP4 schemes).
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
using vllm::Nvfp4Weight;
using vllm::OwnedTensor;
using vllm::Qwen3DenseWeights;
using vllm::SafetensorsFile;

namespace {

std::string FindNvfp4A16ModelDir() {
  if (const char* env = std::getenv("VLLM_CPP_QWEN3_32B_NVFP4A16_DIR");
      env != nullptr) {
    std::error_code ec;
    if (fs::exists(fs::path(env) / "config.json", ec)) return env;
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
                         ".cache/huggingface/hub/"
                         "models--RedHatAI--Qwen3-32B-NVFP4A16/snapshots";
  std::error_code ec;
  if (!fs::exists(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec) &&
        fs::exists(e.path() / "model.safetensors.index.json", ec))
      return e.path().string();
  }
  return "";
}

// An NVFP4 operand: packed i8 [N, K/2], scale i8 [N, K/16], scale2 > 0, and
// crucially alpha == 0 (W4A16 — no activation quant).
void CheckNvfp4(const Nvfp4Weight& w, int64_t n, int64_t k) {
  CHECK(w.n == n);
  CHECK(w.k == k);
  REQUIRE(w.packed.rank == 2);
  CHECK(w.packed.dtype == vt::DType::kI8);
  CHECK(w.packed.shape[0] == n);
  CHECK(w.packed.shape[1] == k / 2);
  REQUIRE(w.scale.rank == 2);
  CHECK(w.scale.dtype == vt::DType::kI8);
  CHECK(w.scale.shape[0] == n);
  CHECK(w.scale.shape[1] == k / 16);
  CHECK(w.scale2 > 0.0F);
  CHECK(w.weight_global_scale_inv > 0.0F);
  // W4A16: no activation quant was loaded, so the true-W4A4 predicate is FALSE
  // and the weight routes to the Marlin/naive W4A16 dispatcher.
  CHECK(w.alpha == 0.0F);
  CHECK_FALSE(w.IsTrueW4A4());
}

float ReadF32(const SafetensorsFile& f, const std::string& name) {
  const vllm::StTensor& t = f.Get(name);
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  return v;
}

}  // namespace

TEST_CASE("Qwen3-32B-NVFP4A16 loader: compressed-tensors W4A16 name map is "
          "complete and exact (checkpoint-gated)") {
  const std::string dir = FindNvfp4A16ModelDir();
  if (dir.empty()) {
    MESSAGE(
        "SKIP: RedHatAI/Qwen3-32B-NVFP4A16 snapshot absent "
        "(~/.cache/huggingface/hub/models--RedHatAI--Qwen3-32B-NVFP4A16 or "
        "$VLLM_CPP_QWEN3_32B_NVFP4A16_DIR)");
    return;
  }

  const HfConfig cfg = LoadHfConfig(dir + "/config.json");
  CHECK(cfg.num_hidden_layers == 64);
  CHECK(cfg.hidden_size == 5120);
  CHECK(cfg.num_attention_heads == 64);
  CHECK(cfg.num_key_value_heads == 8);
  CHECK(cfg.head_dim == 128);
  CHECK(cfg.intermediate_size == 25600);

  std::vector<SafetensorsFile> shards;
  std::error_code ec;
  std::vector<std::string> files;
  for (const auto& e : fs::directory_iterator(dir, ec))
    if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  REQUIRE(files.size() == 5);
  shards.reserve(files.size());
  for (const std::string& f : files) shards.push_back(SafetensorsFile::Open(f));

  // Every tensor the checkpoint carries — used for the "nothing left over" check.
  std::unordered_set<std::string> unmapped;
  for (const SafetensorsFile& s : shards)
    for (const std::string& n : s.Names()) unmapped.insert(n);
  const size_t total_tensors = unmapped.size();
  MESSAGE("Qwen3-32B-NVFP4A16: " << files.size() << " shards, " << total_tensors
                                 << " tensors");

  const Qwen3DenseWeights w = LoadQwen3ForCausalLMWeights(shards, cfg);

  // ---- top level ------------------------------------------------------------
  CHECK_FALSE(w.tie_word_embeddings);  // 32B is UNTIED (unlike 0.6B)
  CHECK_FALSE(w.attention_bias);
  REQUIRE(w.embed_tokens.rank == 2);
  CHECK(w.embed_tokens.dtype == vt::DType::kBF16);  // not a Linear -> stays BF16
  CHECK(w.embed_tokens.shape[0] == cfg.vocab_size);
  CHECK(w.embed_tokens.shape[1] == cfg.hidden_size);
  CHECK(w.final_norm.dtype == vt::DType::kBF16);
  // lm_head is in the checkpoint's quantization `ignore` list -> BF16, and the
  // dense loader transposes it to Matmul-B [H, vocab].
  REQUIRE(w.lm_head.rank == 2);
  CHECK(w.lm_head.dtype == vt::DType::kBF16);
  CHECK(w.lm_head.shape[0] == cfg.hidden_size);
  CHECK(w.lm_head.shape[1] == cfg.vocab_size);

  // ---- per layer ------------------------------------------------------------
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const int64_t qdim = cfg.num_attention_heads * cfg.head_dim;
  const int64_t kdim = cfg.num_key_value_heads * cfg.head_dim;
  REQUIRE(static_cast<int64_t>(w.layers.size()) == cfg.num_hidden_layers);
  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    const auto& lw = w.layers[static_cast<size_t>(l)];
    CHECK(lw.input_layernorm.dtype == vt::DType::kBF16);
    CHECK(lw.post_attention_layernorm.dtype == vt::DType::kBF16);
    CHECK(lw.attn.q_norm.dtype == vt::DType::kBF16);
    CHECK(lw.attn.k_norm.dtype == vt::DType::kBF16);

    // The QUANTIZED arm must be the one that got populated, and the BF16 arm
    // must be EMPTY (a populated BF16 field would mean the probe mis-fired and
    // the forward would silently take the unquantized path).
    CHECK(lw.attn.IsNvfp4());
    CHECK(lw.mlp.IsNvfp4());
    CHECK(lw.attn.qkv_proj.Empty());
    CHECK(lw.attn.o_proj.Empty());
    CHECK(lw.mlp.gate_up_proj.Empty());
    CHECK(lw.mlp.down_proj.Empty());

    CheckNvfp4(lw.attn.qkv_proj_fp4, qdim + 2 * kdim, H);
    CheckNvfp4(lw.attn.o_proj_fp4, H, qdim);
    CheckNvfp4(lw.mlp.gate_proj_fp4, I, H);
    CheckNvfp4(lw.mlp.up_proj_fp4, I, H);
    CheckNvfp4(lw.mlp.down_proj_fp4, H, I);
  }

  // ---- merged-shard global-scale collapse (the vLLM `max()` rule) -----------
  // vLLM keeps one global scale per shard of a fused linear and collapses them
  // with `.max()` before reciprocating (compressed_tensors_w4a4_nvfp4.py:111-114).
  // Assert layer 0's merged qkv scale2 is exactly 1/max(divisor_q,k,v).
  {
    const SafetensorsFile* src = nullptr;
    for (const SafetensorsFile& s : shards) {
      const auto& names = s.Names();
      if (std::find(names.begin(), names.end(),
                    "model.layers.0.self_attn.q_proj.weight_global_scale") !=
          names.end()) {
        src = &s;
        break;
      }
    }
    REQUIRE(src != nullptr);
    const float dq =
        ReadF32(*src, "model.layers.0.self_attn.q_proj.weight_global_scale");
    const float dk =
        ReadF32(*src, "model.layers.0.self_attn.k_proj.weight_global_scale");
    const float dv =
        ReadF32(*src, "model.layers.0.self_attn.v_proj.weight_global_scale");
    const float dmax = std::max(dq, std::max(dk, dv));
    CHECK(w.layers[0].attn.qkv_proj_fp4.weight_global_scale_inv == dmax);
    CHECK(w.layers[0].attn.qkv_proj_fp4.scale2 == doctest::Approx(1.0F / dmax));
    MESSAGE("Qwen3-32B-NVFP4A16 L0 qkv global-scale divisors q/k/v = "
            << dq << "/" << dk << "/" << dv << " -> max " << dmax
            << " -> scale2 " << w.layers[0].attn.qkv_proj_fp4.scale2);
  }

  // ---- W4A16, not W4A4: no input_global_scale anywhere ----------------------
  size_t input_global_scales = 0;
  for (const SafetensorsFile& s : shards)
    for (const std::string& n : s.Names())
      if (n.find("input_global_scale") != std::string::npos) ++input_global_scales;
  CHECK_MESSAGE(input_global_scales == 0,
                "checkpoint carries input_global_scale — this is W4A4, not the "
                "W4A16 scheme this loader implements");

  // ---- nothing left unmapped ------------------------------------------------
  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    const std::string sa = b + "self_attn.";
    const std::string mlp = b + "mlp.";
    unmapped.erase(b + "input_layernorm.weight");
    unmapped.erase(b + "post_attention_layernorm.weight");
    unmapped.erase(sa + "q_norm.weight");
    unmapped.erase(sa + "k_norm.weight");
    for (const char* p : {"q_proj", "k_proj", "v_proj", "o_proj"})
      for (const char* suf :
           {".weight_packed", ".weight_scale", ".weight_global_scale"})
        unmapped.erase(sa + p + suf);
    for (const char* p : {"gate_proj", "up_proj", "down_proj"})
      for (const char* suf :
           {".weight_packed", ".weight_scale", ".weight_global_scale"})
        unmapped.erase(mlp + p + suf);
  }
  unmapped.erase("model.embed_tokens.weight");
  unmapped.erase("model.norm.weight");
  unmapped.erase("lm_head.weight");
  for (const std::string& left : unmapped)
    MESSAGE("UNMAPPED checkpoint tensor: " << left);
  CHECK_MESSAGE(unmapped.empty(),
                "checkpoint tensors were left unmapped by the NVFP4 W4A16 loader");
}
