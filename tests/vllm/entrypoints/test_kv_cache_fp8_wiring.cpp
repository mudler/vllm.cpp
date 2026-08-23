// KV-FP8 W3 gate — the RUNNER integration: half-sized KV blocks, the
// `--kv-cache-dtype` thread from the flag to the block sizing, and the
// checkpoint `k_scale`/`v_scale` path.
//
// Upstream anchors, all verified in /home/mudler/_git/vllm at the parity pin
// `555967922`:
//   * `vllm/config/cache.py:19-36` CacheDType, `:76` cache_dtype default,
//     `:111` calculate_kv_scales (deprecated).
//   * `vllm/utils/torch_utils.py:32-52` STR_DTYPE_TO_TORCH_DTYPE (every fp8
//     CacheDType maps to `torch.uint8` — ONE byte), `:64-67`
//     MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP, `:75-80` is_quantized_kv_cache,
//     `:310-362` get_kv_cache_quant_algo_string, `:374-392`
//     resolve_kv_cache_dtype_string, `:394-401` kv_cache_dtype_str_to_dtype.
//   * `vllm/v1/worker/gpu_model_runner.py:484-486` — the runner resolves ONE
//     kv_cache_dtype and every attention spec is built with it.
//   * `vllm/v1/kv_cache_interface.py:204-218` AttentionSpec.real_page_size_bytes
//     — linear in `get_dtype_size(self.dtype)`, which is the whole halving.
//   * `vllm/model_executor/layers/quantization/kv_cache.py:18-30`
//     KVCacheScaleParameter (the -1.0 unloaded sentinel), `:100-102` the
//     is_quantized_kv_cache guard, `:104-127` the three loaded arms, `:150-156`
//     the uncalibrated warning.
//   * `vllm/engine/arg_utils.py:1915-1929` — the resolution happens ONCE, at
//     config construction, and CacheConfig receives the resolved string.
//
// THE CASES ARE ORDERED BY WHAT THEY WOULD LET THROUGH IF THEY WERE MISSING:
//   G1  the checkpoint declaration resolves, and an explicit flag outranks it
//   G2  a declared-but-absent scale is NOT the same state as no declaration
//   G3  the block arithmetic — an fp8 page is EXACTLY half a bf16 page
//   G4  the same halving through the LOADER: the same byte budget buys 2x blocks
//   G5  the fp8 KV path is REACHED from a production entry point (generation)
//   G6  storage dtype and fp8 interpretation cannot disagree
//   G7  an unrouted attention block is refused BY NAME, never silently
//   G8  the refusals: MLA, float16, e5m2, and a Mamba state left alone
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/cache.h"
#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/layers/quantization/kv_cache.h"
#include "vllm/model_executor/models/kv_cache_route.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::KvScaleOrigin;
using vllm::OwnedTensor;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vt::DType;

namespace {

// ─── The gate checkpoint's own declaration, transcribed ──────────────────────
//
// `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b8
// 7825680`, the subject of benchmark campaign #1574. Fetched from the public
// `hf_quant_config.json` on 2026-08-21 and trimmed to the three keys this
// resolver reads; the `quantized_layers` map (1900+ entries) is the WEIGHT half
// and is read elsewhere.
//
// The same revision's `model.safetensors.index.json` lists 2001 tensors and
// ZERO named `k_scale`, `v_scale` or `kv_scale` — measured, not assumed. That
// pair of facts is the whole reason G2 exists.
constexpr const char* kGateCheckpointQuantConfig = R"({
  "producer": {"name": "modelopt", "version": "0.46.0rc1"},
  "quantization": {
    "quant_algo": "MIXED_PRECISION",
    "kv_cache_quant_algo": "FP8",
    "quantized_layers": {
      "model.language_model.layers.0.mlp.gate_proj":
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    }
  }
})";

// A modelopt checkpoint that quantizes WEIGHTS and declares nothing about the
// KV cache — the case that must not reach a default scale.
constexpr const char* kNoKvDeclarationQuantConfig = R"({
  "producer": {"name": "modelopt", "version": "0.46.0rc1"},
  "quantization": {"quant_algo": "FP8", "quantized_layers": {}}
})";

// The SAME declaration in the place ModelOpt 0.31.0 and after writes it: inline
// under `config.json:quantization_config`, flat, with `quant_method` beside the
// algorithm (`vllm/transformers_utils/config.py:751-753`). It quantizes weights
// and says nothing about the KV cache, so a loader that reads it takes the model
// dtype — and a loader that reaches past it to a stale `hf_quant_config.json`
// does not.
constexpr const char* kInlineWeightsOnlyQuantConfig = R"({
  "model_type": "qwen3_5_text",
  "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}
})";

// AND THE ONE THE GATE CHECKPOINT ACTUALLY SHIPS. Fetched from the same
// revision's `config.json` on 2026-08-22 and trimmed to the keys this resolver
// reads (`config_groups`, `quantized_layers` and `ignore` are the WEIGHT half;
// `quantized_layers` holds 401 entries). It carries `quant_method` and
// `producer` — so the modelopt marker is present twice over — and NO
// `kv_cache_*` key anywhere. Because `config.json:quantization_config` is read
// in preference to `hf_quant_config.json`, THIS is the document that decides,
// and it declares nothing. The transcription above is the file that does not
// get read for this checkpoint.
constexpr const char* kGateCheckpointInlineConfig = R"({
  "model_type": "qwen3_5",
  "quantization_config": {
    "quant_algo": "MIXED_PRECISION",
    "producer": {"name": "modelopt", "version": "0.46.0rc1"},
    "quant_method": "modelopt",
    "quantized_layers": {
      "model.language_model.layers.0.mlp.gate_proj":
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    }
  }
})";

// ─── Synthetic dense-hybrid model (the same shape as
// tests/vllm/entrypoints/test_loaded_engine_dense.cpp, which is the file whose
// LOADER path these cases enter through) ─────────────────────────────────────
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

constexpr int kVocab = 24;
constexpr int kMaxModelLen = 32;

HfConfig MakeDenseConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = kVocab;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = kMaxModelLen;
  c.raw = json::object();
  return c;
}

vllm::DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  vllm::DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

vllm::Qwen3_5DenseWeights MakeDenseWeights(const HfConfig& c) {
  vllm::Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// The same model with its full-attention K and V projections in the RAW torch
// Linear layout every real safetensors checkpoint ships: [N=out, K=in] with
// `nk` set. `ProjectFullAttnQkv` serves those through `MatmulBf16D`, so the
// projection emits the MODEL dtype (bf16) instead of the f32 that `MakeDenseWeights`
// happens to produce — which is the difference the fp8 store sees. See G9.
vllm::Qwen3_5DenseWeights MakeDenseWeightsTorchKv(const HfConfig& c) {
  vllm::Qwen3_5DenseWeights w = MakeDenseWeights(c);
  const int64_t H = c.hidden_size;
  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  for (size_t l = 0; l < w.layers.size(); ++l) {
    vllm::Qwen3_5DenseLayerWeights& lw = w.layers[l];
    if (lw.is_linear_attention) continue;
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    const int64_t Hq = c.num_attention_heads;
    // All THREE, because that is the only shape a checkpoint comes in: the fused
    // preamble requires `qgate` and `kf` to share one dtype
    // (`vt::AttnQkNormRopeGate`), so a half-converted layer is a harness defect
    // rather than a case.
    lw.attn.q_proj = MakeOwned(DType::kBF16, {2 * Hq * Dh, H}, s + 10);
    lw.attn.q_proj.nk = true;
    lw.attn.k_proj = MakeOwned(DType::kBF16, {Hkv * Dh, H}, s + 20);
    lw.attn.k_proj.nk = true;
    lw.attn.v_proj = MakeOwned(DType::kBF16, {Hkv * Dh, H}, s + 30);
    lw.attn.v_proj.nk = true;
  }
  return w;
}

// A throwaway model directory carrying only the files the loader's resolution
// stanza opens. `LoadedEngine::FromModelDir` is the only caller of
// `vllm::ReadQuantConfigJson`, so a case that wants to measure the LOADER has to
// hand it a directory rather than a string.
class CheckpointDir {
 public:
  CheckpointDir() {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllm_kvfp8_ckpt_" + std::to_string(counter++));
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
    std::filesystem::create_directories(path_);
  }
  ~CheckpointDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  CheckpointDir(const CheckpointDir&) = delete;
  CheckpointDir& operator=(const CheckpointDir&) = delete;

  void Write(const char* name, const std::string& body) const {
    std::ofstream(path_ / name, std::ios::binary) << body;
  }
  std::string str() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

vllm::tok::Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_kvfp8_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},   {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},   {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11}, {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15}, {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  vocab[vllm::tok::MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[vllm::tok::MapBytesToUnicode("\x8C\x8D")] = 23;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

// The absolute KV budget both arms of G4 are given. Large enough that the fp8
// arm's doubled block count is well inside the pool the tiny model needs, and
// EXACTLY divisible by both per-block sizes so the 2x is an equality rather
// than a rounding coincidence.
constexpr int64_t kKvBudgetBytes = 1 << 20;

EngineParams ParamsWithCacheDType(const std::string& cache_dtype) {
  EngineParams p;
  p.kv_cache_memory_bytes = kKvBudgetBytes;
  p.kv_cache_dtype = cache_dtype;
  return p;
}

// The single full-attention group's spec out of a loaded engine's RESOLVED KV
// config. The synthetic model has exactly one (three GDN layers + one full
// attention layer), so "the first attention spec" is unambiguous.
const vllm::v1::AttentionSpec* SoleAttentionSpec(const LoadedEngine& eng) {
  for (const auto& group : eng.kv_cache_config().kv_cache_groups) {
    const auto* attn =
        dynamic_cast<const vllm::v1::AttentionSpec*>(group.kv_cache_spec.get());
    if (attn != nullptr) return attn;
  }
  return nullptr;
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.output_kind = vllm::RequestOutputKind::kCumulative;
  return sp;
}

class CerrRedirect {
 public:
  explicit CerrRedirect(std::streambuf* target)
      : previous_(std::cerr.rdbuf(target)) {}
  ~CerrRedirect() { std::cerr.rdbuf(previous_); }
  CerrRedirect(const CerrRedirect&) = delete;
  CerrRedirect& operator=(const CerrRedirect&) = delete;

 private:
  std::streambuf* previous_;
};

// A bare NHD KV cache pair for the op-level cases, on the CPU queue.
struct HostKvPair {
  std::vector<uint8_t> storage;
  vt::Tensor k;
  vt::Tensor v;
};

}  // namespace

// ─── G1. The checkpoint's declaration, and who outranks whom ─────────────────
TEST_CASE("kv-fp8 W3 G1: the gate checkpoint's kv_cache_quant_algo resolves") {
  // torch_utils.py:374-392 + :310-362 + :64-67. "FP8" (the modelopt spelling,
  // upper case) maps to vLLM's own `fp8_e4m3`, not to the bare "fp8" alias.
  //
  // This document names NO top-level `quant_method`, and upstream's
  // `get_kv_cache_quant_algo_string` tests exactly that key (`:319`). It still
  // resolves upstream, because `_normalize_quantization_config`
  // (`transformers_utils/model_arch_config_convertor.py:208-247`) INJECTS the
  // marker from `producer.name` into the same dict first — see the long comment
  // in `src/vllm/config/cache.cpp` for the measurement. Reading only the first
  // function makes this case look like a divergence; running both says it is
  // the mirror.
  const vllm::ResolvedCacheDTypeString r =
      vllm::ResolveKvCacheDTypeString("auto", kGateCheckpointQuantConfig);
  CHECK(r.cache_dtype == "fp8_e4m3");
  // The FACT that separates this from an operator who typed the flag.
  CHECK(r.declared_by_checkpoint);
}

TEST_CASE("kv-fp8 W3 G1: the modelopt marker is upstream's THREE, and no more") {
  // `_normalize_quantization_config:216-235` injects `quant_method` on exactly
  // two conditions — `producer["name"] == "modelopt"` (an equality, not a
  // prefix) and a nested `modelopt_quant_config` key — AND ONLY WHEN the same
  // nested document carries a `quant_algo` (`:224`). `torch_utils.py:319` reads
  // the top-level key itself. Nothing upstream can put a marker anywhere else,
  // so nothing else may be accepted here: a resolver that reads a marker
  // upstream cannot see turns on an fp8 KV cache vLLM would not, at half the
  // page, on a checkpoint nobody flagged.

  // (a) The flat 0.31.0-and-after shape: the marker upstream reads directly.
  CHECK(vllm::ResolveKvCacheDTypeString(
            "auto",
            R"({"quant_method":"modelopt","quant_algo":"FP8",)"
            R"("kv_cache_quant_algo":"FP8"})")
            .cache_dtype == "fp8_e4m3");

  // (b) The legacy nested shape, recognised by the key alone (`:218-220`) —
  // WITH the `quant_algo` `:224` requires before it injects anything.
  CHECK(vllm::ResolveKvCacheDTypeString(
            "auto",
            R"({"quantization":{"modelopt_quant_config":{},"quant_algo":"FP8",)"
            R"("kv_cache_quant_algo":"FP8"}})")
            .cache_dtype == "fp8_e4m3");

  // (c) A marker only INSIDE `quantization`. Upstream never looks there — its
  // injector writes the top level and `:319` reads the top level — so neither
  // do we.
  const vllm::ResolvedCacheDTypeString inner = vllm::ResolveKvCacheDTypeString(
      "auto",
      R"({"quantization":{"quant_method":"modelopt",)"
      R"("kv_cache_quant_algo":"FP8"}})");
  CHECK(inner.cache_dtype == "auto");
  CHECK_FALSE(inner.declared_by_checkpoint);

  // (d) A producer that merely STARTS with "modelopt". `:222` is `==`, so
  // `modelopt_fp4` as a PRODUCER name is not the marker (it is a `quant_method`
  // VALUE the injector writes, which arm (a) already covers). The `quant_algo`
  // is present in this document and in (e) so that the PRODUCER test is what
  // refuses them, rather than the `:224` guard arm (g) covers.
  const vllm::ResolvedCacheDTypeString near = vllm::ResolveKvCacheDTypeString(
      "auto",
      R"({"producer":{"name":"modelopt_fp4"},)"
      R"("quantization":{"quant_algo":"FP8","kv_cache_quant_algo":"FP8"}})");
  CHECK(near.cache_dtype == "auto");
  CHECK_FALSE(near.declared_by_checkpoint);

  // (e) A non-modelopt producer with the same key is still nothing.
  CHECK(vllm::ResolveKvCacheDTypeString(
            "auto",
            R"({"producer":{"name":"llm-compressor"},)"
            R"("quantization":{"quant_algo":"FP8","kv_cache_quant_algo":"FP8"}})")
            .cache_dtype == "auto");

  // (f) The two markers are normalised DIFFERENTLY, because upstream normalises
  // them differently: `quant_method` is lower-cased before the prefix test
  // (`:238-246`), the producer name is compared raw against the literal
  // (`:222`). A `MODELOPT` quant_method resolves; a `ModelOpt` producer does
  // not.
  CHECK(vllm::ResolveKvCacheDTypeString(
            "auto",
            R"({"quant_method":"MODELOPT","kv_cache_quant_algo":"FP8"})")
            .cache_dtype == "fp8_e4m3");
  CHECK(vllm::ResolveKvCacheDTypeString(
            "auto",
            R"({"producer":{"name":"ModelOpt"},)"
            R"("quantization":{"quant_algo":"FP8","kv_cache_quant_algo":"FP8"}})")
            .cache_dtype == "auto");

  // (g) THE `:224` GUARD. The injector writes a marker only `if quant_algo is
  // not None`, read out of `quant_cfg.get("quantization", {})` — an EMPTY-object
  // fallback, unlike the reader's `quant_cfg.get("quantization", quant_cfg)` at
  // `torch_utils.py:321`. Three shapes therefore answer `None` upstream, and
  // each of them resolved to `fp8_e4m3` here before the third review:
  //
  //   (g1) a `modelopt` producer whose nested document declares no `quant_algo`
  //   (g2) a legacy `modelopt_quant_config` with the same omission
  //   (g3) a TOP-LEVEL `modelopt_quant_config`, with no `quantization` key at
  //        all — upstream's `{}` fallback means the key is not even looked for
  //
  // A resolver that accepts any of these halves the KV page on a checkpoint vLLM
  // would run at the model dtype.
  const vllm::ResolvedCacheDTypeString g1 = vllm::ResolveKvCacheDTypeString(
      "auto",
      R"({"producer":{"name":"modelopt"},)"
      R"("quantization":{"kv_cache_quant_algo":"FP8"}})");
  CHECK(g1.cache_dtype == "auto");
  CHECK_FALSE(g1.declared_by_checkpoint);

  const vllm::ResolvedCacheDTypeString g2 = vllm::ResolveKvCacheDTypeString(
      "auto",
      R"({"quantization":{"modelopt_quant_config":{},)"
      R"("kv_cache_quant_algo":"FP8"}})");
  CHECK(g2.cache_dtype == "auto");
  CHECK_FALSE(g2.declared_by_checkpoint);

  const vllm::ResolvedCacheDTypeString g3 = vllm::ResolveKvCacheDTypeString(
      "auto",
      R"({"modelopt_quant_config":{},"quant_algo":"FP8",)"
      R"("kv_cache_quant_algo":"FP8"})");
  CHECK(g3.cache_dtype == "auto");
  CHECK_FALSE(g3.declared_by_checkpoint);

  // And the SAME three documents with a `quant_algo` where `:224` reads it do
  // resolve, so what refuses (g1) and (g2) is the guard and not the shape.
  CHECK(vllm::ResolveKvCacheDTypeString(
            "auto",
            R"({"producer":{"name":"modelopt"},)"
            R"("quantization":{"quant_algo":"FP8","kv_cache_quant_algo":"FP8"}})")
            .cache_dtype == "fp8_e4m3");
  CHECK(vllm::ResolveKvCacheDTypeString(
            "auto",
            R"({"quantization":{"modelopt_quant_config":{},"quant_algo":"NVFP4",)"
            R"("kv_cache_quant_algo":"FP8"}})")
            .cache_dtype == "fp8_e4m3");
  // (g3) has no `quantization` object, so no `quant_algo` can rescue it: the
  // legacy key at the top level is not a marker in any spelling.
  CHECK(vllm::ResolveKvCacheDTypeString(
            "auto",
            R"({"modelopt_quant_config":{"quant_algo":"FP8"},)"
            R"("kv_cache_quant_algo":"FP8"})")
            .cache_dtype == "auto");
}

TEST_CASE("kv-fp8 W3 G1: an explicit --kv-cache-dtype outranks the checkpoint") {
  // torch_utils.py:380-381 returns the explicit value UNCHANGED without ever
  // reading the config, and attention.py:279-290 re-applies the same precedence
  // with the comment "an explicit choice (e.g. bfloat16) must win".
  const vllm::ResolvedCacheDTypeString r =
      vllm::ResolveKvCacheDTypeString("bfloat16", kGateCheckpointQuantConfig);
  CHECK(r.cache_dtype == "bfloat16");
  CHECK_FALSE(r.declared_by_checkpoint);
}

TEST_CASE("kv-fp8 W3 G1: a checkpoint that declares no KV algo resolves auto") {
  const vllm::ResolvedCacheDTypeString none =
      vllm::ResolveKvCacheDTypeString("auto", kNoKvDeclarationQuantConfig);
  CHECK(none.cache_dtype == "auto");
  CHECK_FALSE(none.declared_by_checkpoint);

  // No quantization config at all — the ordinary bf16 checkpoint.
  const vllm::ResolvedCacheDTypeString empty =
      vllm::ResolveKvCacheDTypeString("auto", "");
  CHECK(empty.cache_dtype == "auto");
  CHECK_FALSE(empty.declared_by_checkpoint);
}

TEST_CASE("kv-fp8 W3 G1: an unrecognized kv_cache_quant_algo falls back to auto") {
  // torch_utils.py:351-361 — upstream's own safe fallback. It must NOT become
  // "declared", because a KV format we cannot serve is not a declaration we can
  // honour.
  const vllm::ResolvedCacheDTypeString r = vllm::ResolveKvCacheDTypeString(
      "auto",
      R"({"producer":{"name":"modelopt"},
          "quantization":{"quant_algo":"FP8","kv_cache_quant_algo":"INT3"}})");
  CHECK(r.cache_dtype == "auto");
  CHECK_FALSE(r.declared_by_checkpoint);
}

// ─── G2. Declared-but-absent is NOT the same state as never declared ─────────
TEST_CASE("kv-fp8 W3 G2: a DECLARED fp8 cache with no scale tensors takes 1.0") {
  // kv_cache.py:112-116 — this is the arm the gate checkpoint takes: it declares
  // `kv_cache_quant_algo: "FP8"` and ships zero k/v scale tensors, so both
  // sentinels survive and the documented default applies.
  const vllm::ResolvedKvCacheScales r = vllm::ResolveKvCacheScales(
      "fp8_e4m3", /*calculate_kv_scales=*/false, vllm::kKvScaleUnloaded,
      vllm::kKvScaleUnloaded);
  CHECK(r.origin == KvScaleOrigin::kDeclaredButAbsent);
  CHECK(r.k_scale == doctest::Approx(1.0F));
  CHECK(r.v_scale == doctest::Approx(1.0F));
  // kv_cache.py:150-156 — and it says so.
  CHECK(r.uncalibrated);

  // The consumer is happy to be handed this pair: it was DECLARED.
  float k = 0.0F;
  float v = 0.0F;
  vllm::ScalesForFp8Store(r, &k, &v);
  CHECK(k == doctest::Approx(1.0F));
  CHECK(v == doctest::Approx(1.0F));
}

TEST_CASE(
    "kv-fp8 W3 G2: NO declaration is a different state and yields NO scale") {
  // THE CASE THIS WHOLE FILE EXISTS FOR. Identical inputs to the one above
  // except the declaration, and identical NUMBERS out — 1.0/1.0 are the struct's
  // defaults — so a gate that only read k_scale/v_scale could not tell them
  // apart. `origin` can, and the consumer refuses on it.
  const vllm::ResolvedKvCacheScales r = vllm::ResolveKvCacheScales(
      "auto", /*calculate_kv_scales=*/false, vllm::kKvScaleUnloaded,
      vllm::kKvScaleUnloaded);
  CHECK(r.origin == KvScaleOrigin::kNotQuantized);
  // kv_cache.py:100-102: the scale block never ran, so the uncalibrated warning
  // is not owed either.
  CHECK_FALSE(r.uncalibrated);

  float k = 0.0F;
  float v = 0.0F;
  CHECK_THROWS_AS(vllm::ScalesForFp8Store(r, &k, &v), std::runtime_error);
  // And the refusal NAMES what is missing, so the next reader does not have to
  // rediscover the distinction.
  try {
    vllm::ScalesForFp8Store(r, &k, &v);
    FAIL("ScalesForFp8Store accepted a kNotQuantized pair");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    CHECK(msg.find("no fp8 KV cache was declared") != std::string::npos);
    CHECK(msg.find("kv_cache_quant_algo") != std::string::npos);
  }
  // Nothing was written into the outputs.
  CHECK(k == doctest::Approx(0.0F));
  CHECK(v == doctest::Approx(0.0F));
}

TEST_CASE("kv-fp8 W3 G2: the three LOADED arms mirror kv_cache.py:104-127") {
  // Both scales present (:104-111).
  const vllm::ResolvedKvCacheScales both = vllm::ResolveKvCacheScales(
      "fp8_e4m3", /*calculate_kv_scales=*/false, 0.5F, 0.25F);
  CHECK(both.origin == KvScaleOrigin::kCheckpoint);
  CHECK(both.k_scale == doctest::Approx(0.5F));
  CHECK(both.v_scale == doctest::Approx(0.25F));
  CHECK_FALSE(both.uncalibrated);  // not 1.0/1.0

  // A single `kv_scale`, remapped to k_scale at load and duplicated (:117-127).
  const vllm::ResolvedKvCacheScales dup = vllm::ResolveKvCacheScales(
      "fp8_e4m3", /*calculate_kv_scales=*/false, 0.5F, vllm::kKvScaleUnloaded);
  CHECK(dup.origin == KvScaleOrigin::kCheckpointKvScale);
  CHECK(dup.k_scale == doctest::Approx(0.5F));
  CHECK(dup.v_scale == doctest::Approx(0.5F));

  // e5m2 suppresses the uncalibrated warning (:153) — 1.0 is ordinary there.
  const vllm::ResolvedKvCacheScales e5m2 = vllm::ResolveKvCacheScales(
      "fp8_e5m2", /*calculate_kv_scales=*/false, vllm::kKvScaleUnloaded,
      vllm::kKvScaleUnloaded);
  CHECK(e5m2.origin == KvScaleOrigin::kDeclaredButAbsent);
  CHECK_FALSE(e5m2.uncalibrated);

  // The deprecated dynamic path is refused BY NAME rather than silently taking
  // the static arm (cache.py:111).
  CHECK_THROWS_AS(vllm::ResolveKvCacheScales("fp8_e4m3",
                                             /*calculate_kv_scales=*/true,
                                             vllm::kKvScaleUnloaded,
                                             vllm::kKvScaleUnloaded),
                  std::runtime_error);
}

// ─── G3. The block arithmetic ────────────────────────────────────────────────
TEST_CASE("kv-fp8 W3 G3: an fp8 KV page is EXACTLY half a bf16 page") {
  // kv_cache_interface.py:204-218 — `real_page_size_bytes` is
  // `2 * block_size * num_kv_heads * head_dim * get_dtype_size(dtype)`, and
  // torch_utils.py:38-40 makes every fp8 CacheDType one byte. Assert the CLOSED
  // FORM, not just the ratio: a ratio alone is satisfied by any pair of widths
  // in 2:1, including a pair that is wrong on both sides.
  constexpr int kBlock = 16;
  constexpr int kHkv = 4;
  constexpr int kDh = 64;
  constexpr int64_t kElems = 2LL * kBlock * kHkv * kDh;  // K + V

  vllm::v1::KVCacheConfig cfg;
  cfg.num_blocks = 8;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(kBlock, kHkv, kDh,
                                                    DType::kBF16));
  const auto* spec = dynamic_cast<const vllm::v1::AttentionSpec*>(
      cfg.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(spec != nullptr);
  const int64_t bf16_page = spec->page_size_bytes();
  CHECK(bf16_page == kElems * 2);

  vllm::v1::ApplyCacheDType(cfg, vllm::v1::ParseCacheDType("fp8", DType::kBF16),
                            1.0F, 1.0F);
  const int64_t fp8_page = spec->page_size_bytes();
  CHECK(fp8_page == kElems * 1);
  CHECK(fp8_page * 2 == bf16_page);
  // The storage dtype and the interpretation both landed, on the SAME spec.
  CHECK(spec->dtype == DType::kI8);
  CHECK(spec->fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3);
  // KVBytesPerBlock — the divisor the pool sizing actually uses — halves too.
  CHECK(vllm::v1::KVBytesPerBlock(cfg) == kElems);
}

TEST_CASE("kv-fp8 W3 G3: an auto cache_dtype leaves every spec untouched") {
  // The byte-identical default. `ApplyCacheDType` must not rewrite a spec the
  // model's factory already built at the model dtype.
  constexpr int kBlock = 16;
  vllm::v1::KVCacheConfig cfg;
  cfg.num_blocks = 8;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(kBlock, 4, 64,
                                                    DType::kBF16));
  const int64_t before = vllm::v1::KVBytesPerBlock(cfg);
  vllm::v1::ApplyCacheDType(cfg, vllm::v1::ParseCacheDType("auto", DType::kBF16),
                            1.0F, 1.0F);
  CHECK(vllm::v1::KVBytesPerBlock(cfg) == before);
  const auto* spec = dynamic_cast<const vllm::v1::AttentionSpec*>(
      cfg.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(spec != nullptr);
  CHECK(spec->dtype == DType::kBF16);
  CHECK(spec->fp8_kind == vt::Fp8KVCacheDataType::kAuto);
}

TEST_CASE("kv-fp8 W3 G3: the f32 A/B cache and an MLA spec still load on auto") {
  // The regression this early return exists for. `ApplyCacheDType` refuses
  // float16 and refuses MLA — correctly — so it must not REACH those refusals on
  // the default path. `VT_KV_CACHE_F32=1` builds an f32 KV spec and "auto"
  // resolves to f32; an MLA model on "auto" resolves to its own model dtype.
  // Both are asking for nothing to change, and both must survive.
  vllm::v1::KVCacheConfig f32;
  f32.num_blocks = 4;
  f32.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(16, 4, 64, DType::kF32));
  vllm::v1::ApplyCacheDType(
      f32, vllm::v1::ParseCacheDType("auto", DType::kF32), 1.0F, 1.0F);
  const auto* f32_spec = dynamic_cast<const vllm::v1::AttentionSpec*>(
      f32.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(f32_spec != nullptr);
  CHECK(f32_spec->dtype == DType::kF32);

  vllm::v1::KVCacheConfig mla;
  mla.num_blocks = 4;
  mla.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<vllm::v1::MLAAttentionSpec>(16, 576, DType::kBF16));
  // No throw: the MLA refusal is for an fp8 REQUEST, not for every load.
  vllm::v1::ApplyCacheDType(
      mla, vllm::v1::ParseCacheDType("auto", DType::kBF16), 1.0F, 1.0F);
  const auto* mla_spec = dynamic_cast<const vllm::v1::AttentionSpec*>(
      mla.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(mla_spec != nullptr);
  CHECK(mla_spec->dtype == DType::kBF16);
}

// ─── G4. The same halving, through the LOADER ────────────────────────────────
TEST_CASE(
    "kv-fp8 W3 G4: --kv-cache-dtype fp8 buys EXACTLY 2x the blocks at one "
    "--kv-cache-memory") {
  // This case enters through the production entry point — the LoadedEngine
  // constructor -> MakeKVCacheResolved -> ApplyResolvedCacheDType ->
  // ResolveNumBlocks — rather than calling the resolver, because what is under
  // test is that the storage dtype reaches the sizing BEFORE the sizing reads
  // the geometry. Applying it afterwards would leave this equality at 1x and is
  // the ordering mistake the comment in MakeKVCacheResolved names.
  const HfConfig c = MakeDenseConfig();

  LoadedEngine bf16(c, MakeDenseWeights(c), BuildFixture(),
                    ParamsWithCacheDType("auto"));
  LoadedEngine fp8(c, MakeDenseWeights(c), BuildFixture(),
                   ParamsWithCacheDType("fp8"));

  const int bf16_blocks = bf16.kv_cache_config().num_blocks;
  const int fp8_blocks = fp8.kv_cache_config().num_blocks;
  REQUIRE(bf16_blocks > 0);
  CHECK(fp8_blocks == 2 * bf16_blocks);

  // And the specs say why.
  const auto* bf16_spec = SoleAttentionSpec(bf16);
  const auto* fp8_spec = SoleAttentionSpec(fp8);
  REQUIRE(bf16_spec != nullptr);
  REQUIRE(fp8_spec != nullptr);
  CHECK(bf16_spec->dtype == DType::kBF16);
  CHECK(fp8_spec->dtype == DType::kI8);
  CHECK(fp8_spec->page_size_bytes() * 2 == bf16_spec->page_size_bytes());

  // The POOL is the same size in bytes — that is the point of the feature: the
  // same memory now holds twice the context.
  CHECK(static_cast<int64_t>(fp8_blocks) * vllm::v1::KVBytesPerBlock(
            fp8.kv_cache_config()) ==
        static_cast<int64_t>(bf16_blocks) *
            vllm::v1::KVBytesPerBlock(bf16.kv_cache_config()));
}

TEST_CASE("kv-fp8 W3 G4: the GDN/Mamba state is NOT retyped") {
  // config/cache.py:131-138 — recurrent state has its own mamba_cache_dtype
  // knob and `--kv-cache-dtype` never touches it. The synthetic model has three
  // linear-attention layers, so a rewrite that walked every group would corrupt
  // them.
  const HfConfig c = MakeDenseConfig();
  LoadedEngine fp8(c, MakeDenseWeights(c), BuildFixture(),
                   ParamsWithCacheDType("fp8"));
  bool saw_mamba = false;
  for (const auto& group : fp8.kv_cache_config().kv_cache_groups) {
    const auto* mamba =
        dynamic_cast<const vllm::v1::MambaSpec*>(group.kv_cache_spec.get());
    if (mamba == nullptr) continue;
    saw_mamba = true;
    for (const vt::DType dt : mamba->dtypes) {
      CHECK(dt != DType::kI8);
    }
  }
  CHECK(saw_mamba);  // the case would be vacuous without one
}

// ─── G5. Reachability: the fp8 KV path SERVES ────────────────────────────────
TEST_CASE("kv-fp8 W3 G5: an fp8 KV engine generates through the real forward") {
  // The reachability gate. Nothing here constructs a vt op or a PagedKvCache by
  // hand: the engine is built from the loader, the request goes through
  // LLMEngine, and the tokens come out of Qwen3_5DenseModel::Forward, whose
  // full-attention layer writes and reads THIS cache. If the store or the read
  // were not routed, `vt::ReshapeAndCache` refuses the kI8 page by name (G7)
  // and this case throws instead of counting tokens.
  const HfConfig c = MakeDenseConfig();
  EngineParams params = ParamsWithCacheDType("fp8");
  LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), params);

  // The cache really is one byte per element on the layer that serves.
  const auto* spec = SoleAttentionSpec(eng);
  REQUIRE(spec != nullptr);
  REQUIRE(spec->dtype == DType::kI8);
  REQUIRE(spec->fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3);

  constexpr int kMaxTokens = 4;
  const vllm::RequestOutput run1 =
      eng.engine().generate("hello world", Greedy(kMaxTokens), "req");
  REQUIRE(run1.finished);
  REQUIRE(run1.outputs.size() == 1);
  CHECK(static_cast<int>(run1.outputs[0].token_ids.size()) == kMaxTokens);

  // Deterministic across two fresh stacks on the fp8 cache — greedy decode over
  // an fp8 KV store is still a function of the inputs.
  LoadedEngine again(c, MakeDenseWeights(c), BuildFixture(), params);
  const vllm::RequestOutput run2 =
      again.engine().generate("hello world", Greedy(kMaxTokens), "req");
  REQUIRE(run2.outputs.size() == 1);
  CHECK(run2.outputs[0].token_ids == run1.outputs[0].token_ids);
}

TEST_CASE("kv-fp8 W3 G5: the uncalibrated-scale warning fires ONCE per load") {
  // kv_cache.py:150-156. The gate checkpoint's own case: an fp8 cache serving on
  // the default 1.0, said out loud.
  const HfConfig c = MakeDenseConfig();
  std::ostringstream captured;
  {
    CerrRedirect guard(captured.rdbuf());
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(),
                     ParamsWithCacheDType("fp8"));
    std::cerr.flush();
  }
  const std::string logged = captured.str();
  const std::string needle = "KV cache scaling factor 1.0";
  const size_t first = logged.find(needle);
  REQUIRE(first != std::string::npos);
  // Once, not once per ApplyResolvedCacheDType call (there are two per load).
  CHECK(logged.find(needle, first + needle.size()) == std::string::npos);
  // It names the checkpoint as the place to fix it.
  CHECK(logged.find("checkpoint") != std::string::npos);

  // A SECOND engine in the SAME process gets its own line. Upstream's
  // `logger.warning_once` is per-process; ours is per LOAD, because a
  // process-static latch would silence the second engine rather than the second
  // of the two ApplyResolvedCacheDType calls one load makes. Getting that wrong
  // reads as "the warning fired once" on both counts.
  std::ostringstream second;
  {
    CerrRedirect guard(second.rdbuf());
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(),
                     ParamsWithCacheDType("fp8"));
    std::cerr.flush();
  }
  const std::string logged2 = second.str();
  const size_t only = logged2.find(needle);
  REQUIRE(only != std::string::npos);
  CHECK(logged2.find(needle, only + needle.size()) == std::string::npos);

  // And an auto engine says nothing, so the line is a warning rather than noise.
  std::ostringstream quiet;
  {
    CerrRedirect guard(quiet.rdbuf());
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(),
                     ParamsWithCacheDType("auto"));
    std::cerr.flush();
  }
  CHECK(quiet.str().find(needle) == std::string::npos);
}

// ─── G6. Storage dtype and interpretation cannot disagree ────────────────────
TEST_CASE("kv-fp8 W3 G6: a half-sized page with no fp8 kind is refused") {
  // The silent-corruption shape, asserted at the routing seam. A `kI8` page is
  // sized at one byte per element; an fp8 kind of kAuto would send it to the
  // float store, which indexes at the source width. Neither half of the pair is
  // allowed to travel alone.
  vllm::PagedKvCache kv;
  kv.dtype = DType::kI8;
  kv.fp8_kind = vt::Fp8KVCacheDataType::kAuto;
  CHECK_THROWS_AS(vllm::dense_attn::IsFp8KvCache(kv), std::runtime_error);

  vllm::PagedKvCache other;
  other.dtype = DType::kBF16;
  other.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  CHECK_THROWS_AS(vllm::dense_attn::IsFp8KvCache(other), std::runtime_error);

  // The two consistent states answer without throwing.
  vllm::PagedKvCache floatkv;
  CHECK_FALSE(vllm::dense_attn::IsFp8KvCache(floatkv));
  vllm::PagedKvCache fp8kv;
  fp8kv.dtype = DType::kI8;
  fp8kv.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  CHECK(vllm::dense_attn::IsFp8KvCache(fp8kv));
}

TEST_CASE("kv-fp8 W3 G6: ApplyKvCacheQuant is inert on a float cache") {
  // Every existing caller must be byte-identical: the three additive
  // PagedAttentionArgs fields keep their defaults on a float cache, so the op
  // takes exactly the branch it took before W3.
  vllm::PagedKvCache floatkv;
  vt::PagedAttentionArgs args{0.125F, true};
  vllm::dense_attn::ApplyKvCacheQuant(args, floatkv);
  CHECK(args.kv_cache_dtype == vt::Fp8KVCacheDataType::kAuto);
  CHECK(args.k_scale == doctest::Approx(1.0F));
  CHECK(args.v_scale == doctest::Approx(1.0F));

  vllm::PagedKvCache fp8kv;
  fp8kv.dtype = DType::kI8;
  fp8kv.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  fp8kv.k_scale = 0.5F;
  fp8kv.v_scale = 0.25F;
  vt::PagedAttentionArgs fp8_args{0.125F, true};
  vllm::dense_attn::ApplyKvCacheQuant(fp8_args, fp8kv);
  CHECK(fp8_args.kv_cache_dtype == vt::Fp8KVCacheDataType::kFp8E4M3);
  CHECK(fp8_args.k_scale == doctest::Approx(0.5F));
  CHECK(fp8_args.v_scale == doctest::Approx(0.25F));
}

// ─── G7. An unrouted attention block is refused BY NAME ──────────────────────
TEST_CASE("kv-fp8 W3 G7: the float store refuses a 1-byte fp8 cache by name") {
  // W3 routes `dense_attn::AttnBlock` (the shared seam) and `qwen3_5.cpp`. Every
  // other architecture still calls `vt::ReshapeAndCache` directly, and this is
  // what happens when one of them is served `--kv-cache-dtype fp8`: a named
  // refusal at the first store, not a half-width write into a half-sized page.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  constexpr int64_t T = 2, Hkv = 2, Dh = 4, kBlocks = 2, kBlockSize = 4;
  std::vector<uint16_t> k_src(static_cast<size_t>(T * Hkv * Dh), 0);
  std::vector<uint16_t> v_src(k_src.size(), 0);
  // The NHD unbind-slice cache: ONE (num_blocks, 2, block_size, H, D) byte
  // allocation, k/v are the two dim-1 slices.
  std::vector<uint8_t> cache(
      static_cast<size_t>(kBlocks * 2 * kBlockSize * Hkv * Dh), 0);
  std::vector<int64_t> slots{0, 1};

  vt::Tensor k = vt::Tensor::Contiguous(k_src.data(), DType::kBF16, q.device,
                                        {T, Hkv, Dh});
  vt::Tensor v = vt::Tensor::Contiguous(v_src.data(), DType::kBF16, q.device,
                                        {T, Hkv, Dh});
  vt::Tensor slot_mapping = vt::Tensor::Contiguous(
      slots.data(), DType::kI64, q.device, {static_cast<int64_t>(slots.size())});

  const auto kv_slice = [&](int which) {
    vt::Tensor t;
    t.data = cache.data() + static_cast<size_t>(which) *
                                static_cast<size_t>(kBlockSize * Hkv * Dh);
    t.dtype = DType::kI8;
    t.device = q.device;
    t.rank = 4;
    t.shape[0] = kBlocks;
    t.shape[1] = kBlockSize;
    t.shape[2] = Hkv;
    t.shape[3] = Dh;
    t.stride[0] = 2 * kBlockSize * Hkv * Dh;
    t.stride[1] = Hkv * Dh;
    t.stride[2] = Dh;
    t.stride[3] = 1;
    return t;
  };
  vt::Tensor k_cache = kv_slice(0);
  vt::Tensor v_cache = kv_slice(1);

  try {
    vt::ReshapeAndCache(q, k, v, k_cache, v_cache, slot_mapping);
    FAIL("the float store accepted a 1-byte fp8 cache");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    // It names the OP that should have been called...
    CHECK(msg.find("vt::ReshapeAndCacheFp8") != std::string::npos);
    // ...and the missing part, so the reader knows this is an unwired
    // architecture rather than a corrupt tensor.
    CHECK(msg.find("not routed for fp8 KV") != std::string::npos);
  }
}

// ─── G8. The refusals that stop a mis-sized pool ─────────────────────────────
TEST_CASE("kv-fp8 W3 G8: an MLA cache refuses --kv-cache-dtype fp8 by name") {
  // kv_cache_interface.py:398-410 — upstream's MLA fp8 arm is `fp8_ds_mla` with
  // a different page formula (656 B/token on V3.2), not this one. Retyping an
  // MLA spec to kI8 would size the latent page at half and store it at full.
  vllm::v1::KVCacheConfig cfg;
  cfg.num_blocks = 4;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<vllm::v1::MLAAttentionSpec>(16, 576, DType::kBF16));
  try {
    vllm::v1::ApplyCacheDType(
        cfg, vllm::v1::ParseCacheDType("fp8", DType::kBF16), 1.0F, 1.0F);
    FAIL("ApplyCacheDType retyped an MLA spec");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    CHECK(msg.find("MLA") != std::string::npos);
    CHECK(msg.find("fp8_ds_mla") != std::string::npos);
  }
}

TEST_CASE("kv-fp8 W3 G8: float16 and fp8_e5m2 are refused, not mis-stored") {
  const auto fresh = [] {
    vllm::v1::KVCacheConfig cfg;
    cfg.num_blocks = 4;
    cfg.kv_cache_groups.emplace_back(
        std::vector<std::string>{"fa"},
        std::make_shared<vllm::v1::FullAttentionSpec>(16, 4, 64, DType::kBF16));
    return cfg;
  };

  // float16 PARSES — the CacheDType surface is mirrored in full (cache.py:19-36)
  // — but no attention block casts K/V to f16 before the store, so applying it
  // would reach a dtype mismatch deep inside the op instead of a sentence.
  vllm::v1::KVCacheConfig f16 = fresh();
  try {
    vllm::v1::ApplyCacheDType(
        f16, vllm::v1::ParseCacheDType("float16", DType::kBF16), 1.0F, 1.0F);
    FAIL("ApplyCacheDType accepted float16");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("float16") != std::string::npos);
  }

  // fp8_e5m2 likewise: W1/W2 refuse the compute, so the SIZING must refuse too
  // rather than allocate a half-sized pool nothing can write.
  vllm::v1::KVCacheConfig e5m2 = fresh();
  try {
    vllm::v1::ApplyCacheDType(
        e5m2, vllm::v1::ParseCacheDType("fp8_e5m2", DType::kBF16), 1.0F, 1.0F);
    FAIL("ApplyCacheDType accepted fp8_e5m2");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("fp8_e5m2") != std::string::npos);
  }
}

// ─── G9. The store's TWO sources must share one float dtype ──────────────────
TEST_CASE("kv-fp8 W3 G9: the op refuses a mixed f32 K / bf16 V pair by name") {
  // The contract the model has to satisfy, asserted where it lives, so the case
  // below cannot be read as being about a layout. `vt::ReshapeAndCacheFp8` takes
  // the model floats and quantizes them itself; it has ONE source-dtype template
  // instantiation, so a K and a V of different widths is a refusal and never a
  // mode.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  constexpr int64_t T = 2, Hkv = 2, Dh = 4, kBlocks = 2, kBlockSize = 4;
  std::vector<float> k_src(static_cast<size_t>(T * Hkv * Dh), 0.5F);
  std::vector<uint16_t> v_src(static_cast<size_t>(T * Hkv * Dh),
                              vt::F32ToBF16(0.5F));
  std::vector<uint8_t> cache(
      static_cast<size_t>(kBlocks * 2 * kBlockSize * Hkv * Dh), 0);
  std::vector<int64_t> slots{0, 1};

  vt::Tensor k = vt::Tensor::Contiguous(k_src.data(), DType::kF32, q.device,
                                        {T, Hkv, Dh});
  vt::Tensor v = vt::Tensor::Contiguous(v_src.data(), DType::kBF16, q.device,
                                        {T, Hkv, Dh});
  vt::Tensor slot_mapping = vt::Tensor::Contiguous(
      slots.data(), DType::kI64, q.device, {static_cast<int64_t>(slots.size())});

  const auto kv_slice = [&](int which) {
    vt::Tensor t;
    t.data = cache.data() + static_cast<size_t>(which) *
                                static_cast<size_t>(kBlockSize * Hkv * Dh);
    t.dtype = DType::kI8;
    t.device = q.device;
    t.rank = 4;
    t.shape[0] = kBlocks;
    t.shape[1] = kBlockSize;
    t.shape[2] = Hkv;
    t.shape[3] = Dh;
    t.stride[0] = 2 * kBlockSize * Hkv * Dh;
    t.stride[1] = Hkv * Dh;
    t.stride[2] = Dh;
    t.stride[3] = 1;
    return t;
  };
  vt::Tensor k_cache = kv_slice(0);
  vt::Tensor v_cache = kv_slice(1);

  try {
    vt::ReshapeAndCacheFp8(q, k, v, k_cache, v_cache, slot_mapping,
                           vt::Fp8KVCacheDataType::kFp8E4M3, 1.0F, 1.0F);
    FAIL("the fp8 store accepted a mixed-dtype k/v pair");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("must share one float dtype") !=
          std::string::npos);
  }
}

TEST_CASE("kv-fp8 W3 G9: an fp8 engine serves TORCH-LAYOUT K/V projections") {
  // THE PRODUCTION ARM, and the one the CPU gate could not see.
  //
  // `ProjectFullAttnQkv` picks the projection GEMM's OUTPUT dtype from the
  // weight's own layout: a raw torch Linear weight ([N=out, K=in], `nk`) — which
  // is what every real safetensors checkpoint ships — takes `MatmulBf16D` and
  // therefore emits the MODEL dtype, bf16. `MakeDenseWeights` builds [K,N]
  // weights with `nk` unset, so every other case in this file takes `MatmulF32D`
  // and gets an f32 V. That is the ONLY reason they pass.
  //
  // K never has the choice. The attention preamble writes K into `dk3`, whose
  // dtype is `attn_dt`, and `attn_dt` is f32 for EVERY fp8 cache because
  // `kv.dtype == DType::kBF16` is a term of both FA2 eligibility tests
  // (qwen3_5.cpp, `fa2_prefill` and `fa2_decode`). So the fp8 route hands the
  // store an f32 K beside a bf16 V, and the case above is what it then gets.
  //
  // The bf16 cache route never had this problem: it casts BOTH K and V to the
  // cache dtype. The fp8 route now normalises the same way, to the same bf16 —
  // which is also the dtype upstream quantizes from, because upstream's model IS
  // bf16 at `reshape_and_cache_flash` (`cache_kernels.cu:314-401`).
  const HfConfig c = MakeDenseConfig();
  constexpr int kMaxTokens = 4;

  LoadedEngine fp8(c, MakeDenseWeightsTorchKv(c), BuildFixture(),
                   ParamsWithCacheDType("fp8"));
  const auto* spec = SoleAttentionSpec(fp8);
  REQUIRE(spec != nullptr);
  REQUIRE(spec->dtype == DType::kI8);
  const vllm::RequestOutput out =
      fp8.engine().generate("hello world", Greedy(kMaxTokens), "req");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  CHECK(static_cast<int>(out.outputs[0].token_ids.size()) == kMaxTokens);

  // NOT a case about the layout: the same weights serve the default cache too.
  // Without this the case could go green by the torch layout becoming
  // unloadable, which is a different failure wearing the same colour.
  LoadedEngine automatic(c, MakeDenseWeightsTorchKv(c), BuildFixture(),
                         ParamsWithCacheDType("auto"));
  const vllm::RequestOutput ref =
      automatic.engine().generate("hello world", Greedy(kMaxTokens), "req");
  REQUIRE(ref.outputs.size() == 1);
  CHECK(static_cast<int>(ref.outputs[0].token_ids.size()) == kMaxTokens);
}

// ─── G10. The loader stanza, entered through the loader ──────────────────────
TEST_CASE("kv-fp8 W3 G10: FromModelDir READS the checkpoint's declaration") {
  // `LoadedEngine::FromModelDir` is the ONLY caller of
  // `vllm::ReadQuantConfigJson` and of `vllm::ResolveKvCacheDTypeString` on a
  // model directory. G1 calls the resolver with a string; that measures the
  // resolver. This case measures the LOADER, which is the thing an operator
  // reaches: it hands `FromModelDir` a directory and reads what it said.
  //
  // The directory carries no weights, so the load fails — after the stanza,
  // which is what makes the stanza observable without a checkpoint (the pattern
  // tests/vllm/entrypoints/openai/test_serve_residency_config.cpp establishes).
  CheckpointDir dir;
  dir.Write("hf_quant_config.json", kGateCheckpointQuantConfig);

  std::ostringstream captured;
  std::string thrown;
  {
    CerrRedirect guard(captured.rdbuf());
    try {
      (void)LoadedEngine::FromModelDir(dir.str(), EngineParams{});
    } catch (const std::exception& e) {
      thrown = e.what();
    }
    std::cerr.flush();
  }
  CHECK_FALSE(thrown.empty());
  const std::string logged = captured.str();
  CHECK(logged.find("the checkpoint declares kv_cache_quant_algo") !=
        std::string::npos);
  // It names the RESOLVED value, not the modelopt spelling, because that is the
  // string every consumer downstream will see.
  CHECK(logged.find("fp8_e4m3") != std::string::npos);
}

TEST_CASE("kv-fp8 W3 G10: an explicit flag stops the loader consulting the checkpoint") {
  // torch_utils.py:380-381 through the loader rather than through the resolver.
  // The same directory, one explicit `--kv-cache-dtype`, and the line is gone:
  // the operator outranks the checkpoint, and the checkpoint is not even read.
  CheckpointDir dir;
  dir.Write("hf_quant_config.json", kGateCheckpointQuantConfig);

  EngineParams params;
  params.kv_cache_dtype = "bfloat16";

  std::ostringstream captured;
  {
    CerrRedirect guard(captured.rdbuf());
    try {
      (void)LoadedEngine::FromModelDir(dir.str(), params);
    } catch (const std::exception&) {
    }
    std::cerr.flush();
  }
  CHECK(captured.str().find("the checkpoint declares") == std::string::npos);
}

TEST_CASE("kv-fp8 W3 G10: config.json's quantization_config OUTRANKS hf_quant_config.json") {
  // MIRROR, and it used to be inverted. `vllm/transformers_utils/config.py:751-761`:
  //
  //   quantization_config = config_dict.get("quantization_config", None)
  //   if quantization_config is None and file_or_path_exists(hf_quant_config.json):
  //       quantization_config = get_hf_file_to_dict("hf_quant_config.json", ...)
  //
  // with upstream's own comments naming the two producers: "ModelOpt 0.31.0 and
  // after saves the quantization config in the model config file", and the
  // separate file is "ModelOpt 0.29.0 and before". So an inline
  // `quantization_config` means the legacy file is NEVER opened, and a
  // re-quantized checkpoint that still carries a stale `hf_quant_config.json`
  // beside a current `config.json` resolves to what the CURRENT one says.
  //
  // The two files below disagree on purpose, and only on the KV half: the inline
  // one quantizes weights and declares nothing about the KV cache, the legacy one
  // declares FP8. Reading them in the wrong order quantizes a KV cache nobody
  // asked for, at half the page, silently.
  CheckpointDir dir;
  dir.Write("config.json", kInlineWeightsOnlyQuantConfig);
  dir.Write("hf_quant_config.json", kGateCheckpointQuantConfig);

  // The unit half: which bytes come back.
  const std::string read = vllm::ReadQuantConfigJson(dir.str());
  CHECK(read.find("kv_cache_quant_algo") == std::string::npos);
  CHECK(vllm::GetKvCacheQuantAlgoString(read).value_or("auto") == "auto");

  // The loader half: and therefore nothing is declared.
  std::ostringstream captured;
  {
    CerrRedirect guard(captured.rdbuf());
    try {
      (void)LoadedEngine::FromModelDir(dir.str(), EngineParams{});
    } catch (const std::exception&) {
    }
    std::cerr.flush();
  }
  CHECK(captured.str().find("the checkpoint declares") == std::string::npos);

  // And the legacy file IS read when `config.json` carries no inline document —
  // the 0.29.0-and-before arm, which the inversion would have made unreachable.
  CheckpointDir legacy;
  legacy.Write("config.json", R"({"model_type":"qwen3_5_text"})");
  legacy.Write("hf_quant_config.json", kGateCheckpointQuantConfig);
  CHECK(vllm::GetKvCacheQuantAlgoString(vllm::ReadQuantConfigJson(legacy.str()))
            .value_or("auto") == "fp8_e4m3");
}

TEST_CASE("kv-fp8 W3 G10: the #1574 checkpoint declares NOTHING about the KV") {
  // THE CAMPAIGN CONSEQUENCE, stated as a gate instead of as prose. The two
  // documents above are shaped like the gate checkpoint's; these ARE the gate
  // checkpoint's, both transcribed from
  // `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` on 2026-08-22. The
  // legacy file declares `kv_cache_quant_algo: "FP8"`, the inline one declares
  // no `kv_cache_*` key, and the inline one is what gets read — so an fp8 KV
  // run of the #1574 subject needs `--kv-cache-dtype fp8` typed EXPLICITLY, on
  // this engine and on vLLM alike. Running the pinned
  // `get_kv_cache_quant_algo_string` over the same bytes returns `None`, before
  // and after `_normalize_quantization_config`, so the two engines agree.
  CheckpointDir dir;
  dir.Write("config.json", kGateCheckpointInlineConfig);
  dir.Write("hf_quant_config.json", kGateCheckpointQuantConfig);

  const std::string read = vllm::ReadQuantConfigJson(dir.str());
  // It IS the inline document — the modelopt marker is there twice, so this is
  // not "nothing was found".
  CHECK(read.find("MIXED_PRECISION") != std::string::npos);
  CHECK(read.find("kv_cache_quant_algo") == std::string::npos);

  const vllm::ResolvedCacheDTypeString r =
      vllm::ResolveKvCacheDTypeString("auto", read);
  CHECK(r.cache_dtype == "auto");
  CHECK_FALSE(r.declared_by_checkpoint);

  // And the flag still reaches it, which is the path the campaign uses.
  const vllm::ResolvedCacheDTypeString typed =
      vllm::ResolveKvCacheDTypeString("fp8", read);
  CHECK(typed.cache_dtype == "fp8");
  CHECK_FALSE(typed.declared_by_checkpoint);
}

TEST_CASE("kv-fp8 W3 G10: the drafter-chain refusal runs BEFORE the KV resolution") {
  // THE ORDERING F0 had to keep. `main`'s SPEC-DRAFTER-CHAIN W1 refusal (#1522)
  // and this row's resolution stanza both want to be the first statement of
  // `FromModelDir`, and both are load-bearing: G5 of that row requires the chain
  // refusal "before any weight I/O", and `ReadQuantConfigJson` opens a file
  // inside `model_dir`. Taking either side of that conflict drops a guarantee.
  //
  // `tests/vllm/entrypoints/test_drafter_chain_reach.cpp` cannot see an
  // inversion: it points at a NONEXISTENT directory, and `ReadQuantConfigJson`
  // answers "" for one of those without opening anything, so the chain refusal
  // still arrives. This case points at a directory that EXISTS and declares
  // fp8 — so an inverted order announces the declaration first, and that line is
  // the evidence.
  CheckpointDir dir;
  dir.Write("hf_quant_config.json", kGateCheckpointQuantConfig);

  EngineParams params;
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      R"({"vllm_cpp":{"drafter_chain":[)"
      R"({"method":"ngram","num_speculative_tokens":4},)"
      R"({"method":"mtp"}]}})");
  REQUIRE(params.speculative_config.has_value());
  REQUIRE(params.speculative_config->use_drafter_chain());

  std::ostringstream captured;
  std::string thrown;
  {
    CerrRedirect guard(captured.rdbuf());
    try {
      (void)LoadedEngine::FromModelDir(dir.str(), params);
    } catch (const std::exception& e) {
      thrown = e.what();
    }
    std::cerr.flush();
  }
  CHECK(thrown.find("drafter_chain") != std::string::npos);
  CHECK(captured.str().find("the checkpoint declares") == std::string::npos);
}

// ─── G11. The heterogeneous-per-layer seam ───────────────────────────────────
TEST_CASE("kv-fp8 W3 G11: per_layer_attn_specs are retyped, and the pool halves") {
  // `gpu_model_runner.py:484-486` — upstream resolves ONE kv_cache_dtype and
  // EVERY attention spec is built with it. Ours arrives after the specs exist,
  // so `ApplyCacheDType` has to reach every one of them, and a heterogeneous
  // model (Gemma-4 G1b) allocates from `per_layer_attn_specs` instead of from the
  // group spec. A rewrite that reached only the groups would leave
  // `KVBytesPerBlock` at the full width for exactly the models whose pool it
  // sizes — the half-sizing this row calls silent-corruption territory, with the
  // sign reversed.
  const auto build = [] {
    vllm::v1::KVCacheConfig cfg;
    cfg.num_blocks = 4;
    // Two full-attention layers and one GDN layer, which is the null entry the
    // loop has to step over rather than dereference.
    cfg.kv_cache_groups.emplace_back(
        std::vector<std::string>{"fa0", "fa1"},
        std::make_shared<vllm::v1::FullAttentionSpec>(16, 4, 64, DType::kBF16));
    cfg.per_layer_attn_specs = {
        std::make_shared<vllm::v1::FullAttentionSpec>(16, 4, 64, DType::kBF16),
        nullptr,
        std::make_shared<vllm::v1::FullAttentionSpec>(16, 2, 64, DType::kBF16)};
    return cfg;
  };

  vllm::v1::KVCacheConfig bf16 = build();
  const int64_t bf16_bytes = vllm::v1::KVBytesPerBlock(bf16);
  REQUIRE(bf16_bytes > 0);

  vllm::v1::KVCacheConfig fp8 = build();
  vllm::v1::ApplyCacheDType(fp8, vllm::v1::ParseCacheDType("fp8", DType::kBF16),
                            0.5F, 0.25F);

  // Every non-null per-layer spec carries the fp8 storage AND its interpretation
  // AND the scales, because the store and the read both read them off the spec.
  int retyped = 0;
  for (const auto& spec : fp8.per_layer_attn_specs) {
    if (spec == nullptr) continue;
    CHECK(spec->dtype == DType::kI8);
    CHECK(spec->fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3);
    CHECK(spec->k_scale == doctest::Approx(0.5F));
    CHECK(spec->v_scale == doctest::Approx(0.25F));
    ++retyped;
  }
  CHECK(retyped == 2);  // the case would be vacuous without them

  // And the divisor the pool is sized with is EXACTLY half. An equality, not a
  // ratio: `KVBytesPerBlock` reads `per_layer_attn_specs` in preference to the
  // groups, so this number is what the runner's `--kv-cache-memory` buys.
  CHECK(vllm::v1::KVBytesPerBlock(fp8) * 2 == bf16_bytes);
}

// ─── G12. The SHARED SEAM, entered through a model that uses it ──────────────
//
// WHY THIS CASE EXISTS, and what its absence let through. `## Shared seams` in
// AGENTS.md names `dense_attn::AttnBlock` as the decode seam, and `## W3` of the
// spec lists it as ROUTED. It was not. The routing was written
// (`dense_attn_block.h:519,536,543`) behind a preamble guard that admits only
// `kBF16` and `kF32`, and `IsFp8KvCache` is true only for `kI8` — so `fp8_kv`
// was provably false at every call and the fp8 arms of `WriteKvCache` and
// `ApplyKvCacheQuant` could never execute. A fresh review reverted the WHOLE of
// the seam's routing in a scratch copy and this file stayed at 26/26 SUCCESS,
// because every case above enters through `Qwen3_5DenseModel::Forward` and none
// of them enters here.
//
// The seam is the production forward for the Qwen3 dense family
// (`qwen3.cpp:185`), Qwen3-MoE (`qwen3_moe.cpp:84`), Voxtral
// (`voxtral.cpp:102`) and the Llama/Mistral/InternLM2 registries that share
// `Qwen3DenseModel`. `Qwen3DenseModel::Forward` is the function
// `ForwardQwen3ForCausalLM` calls under `ModelRegistry::Forward`
// (`qwen3_dense.cpp:113`), which is the entry `.agents/reachability.md` names;
// the harness adaptation, stated once, is that `LoadedEngine` has no in-memory
// overload for `Qwen3DenseWeights`, so the cache is allocated here rather than
// by the runner. Everything the case measures — the preamble guard, the store
// and the read — is production code entered through the registered forward.
namespace {

vllm::OwnedTensor MakeSeamBf16(const std::vector<int64_t>& shape, bool nk,
                               uint64_t seed) {
  vllm::OwnedTensor t = MakeOwned(DType::kBF16, shape, seed);
  t.nk = nk;
  return t;
}

constexpr int64_t kSeamVocab = 64;

HfConfig MakeSeamConfig() {
  HfConfig c;
  c.model_type = "qwen3";
  c.architectures = {"Qwen3ForCausalLM"};
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 1000000.0;
  c.vocab_size = kSeamVocab;
  c.max_position_embeddings = 128;
  c.raw = json::object();
  return c;
}

vllm::Qwen3DenseWeights MakeSeamWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads;
  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  vllm::Qwen3DenseWeights w;
  w.tie_word_embeddings = true;
  w.attention_bias = false;
  w.embed_tokens = MakeSeamBf16({V, H}, /*nk=*/false, 7001);
  w.final_norm = MakeSeamBf16({H}, false, 7002);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 8000 + static_cast<uint64_t>(l) * 4000;
    vllm::Qwen3DenseLayerWeights lw;
    lw.input_layernorm = MakeSeamBf16({H}, false, s + 1);
    lw.post_attention_layernorm = MakeSeamBf16({H}, false, s + 2);
    lw.attn.qkv_proj = MakeSeamBf16({qdim + 2 * kdim, H}, /*nk=*/true, s + 3);
    lw.attn.o_proj = MakeSeamBf16({H, qdim}, /*nk=*/true, s + 4);
    lw.attn.q_norm = MakeSeamBf16({Dh}, false, s + 5);
    lw.attn.k_norm = MakeSeamBf16({Dh}, false, s + 6);
    lw.mlp.gate_up_proj = MakeSeamBf16({2 * I, H}, /*nk=*/true, s + 7);
    lw.mlp.down_proj = MakeSeamBf16({H, I}, /*nk=*/true, s + 8);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

constexpr int64_t kSeamBlocks = 2;
constexpr int64_t kSeamBlockSize = 8;

// One paged KV buffer per layer, at the STORAGE WIDTH the cache dtype names —
// one byte per element for fp8, two for bf16. Allocating the fp8 arm at the
// float width would hide exactly the mis-sizing this row exists to prevent.
struct SeamCachePool {
  std::vector<std::vector<uint8_t>> buf;
  std::vector<vllm::PagedKvCache> attn_kv;

  SeamCachePool(const HfConfig& c, DType dt, vt::Fp8KVCacheDataType kind,
                float k_scale, float v_scale) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const size_t elems = static_cast<size_t>(kSeamBlocks * 2 * kSeamBlockSize *
                                             Hkv * Dh);
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      buf.emplace_back(elems * vt::SizeOf(dt), 0);
    for (auto& b : buf) {
      vllm::PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = dt;
      kv.num_blocks = kSeamBlocks;
      kv.block_size = kSeamBlockSize;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      kv.fp8_kind = kind;
      kv.k_scale = k_scale;
      kv.v_scale = v_scale;
      attn_kv.push_back(kv);
    }
  }
};

vllm::v1::CommonAttentionMetadata SeamPrefillMeta(int64_t T) {
  vllm::v1::CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t)
    m.slot_mapping.push_back(static_cast<int32_t>(t % kSeamBlockSize));
  m.causal = true;
  return m;
}

const std::vector<int32_t> kSeamTokens = {3, 17, 42, 8, 61};
const std::vector<int32_t> kSeamPositions = {0, 1, 2, 3, 4};

std::vector<float> RunSeamForward(const HfConfig& c,
                                  const vllm::Qwen3DenseWeights& w,
                                  SeamCachePool& pool) {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::v1::CommonAttentionMetadata meta =
      SeamPrefillMeta(static_cast<int64_t>(kSeamTokens.size()));
  return vllm::Qwen3DenseModel::Forward(kSeamTokens, kSeamPositions, meta,
                                        pool.attn_kv, w, c, q);
}

size_t NonZeroBytes(const std::vector<uint8_t>& b) {
  size_t n = 0;
  for (uint8_t x : b)
    if (x != 0) ++n;
  return n;
}

}  // namespace

TEST_CASE("kv-fp8 W3 G12: the SHARED SEAM serves an fp8 KV cache") {
  const HfConfig c = MakeSeamConfig();
  const vllm::Qwen3DenseWeights w = MakeSeamWeights(c);

  SeamCachePool fp8(c, DType::kI8, vt::Fp8KVCacheDataType::kFp8E4M3, 1.0F,
                    1.0F);
  // BEFORE the repair this line threw
  // "qwen3 dense: KV cache must be bf16 or f32" — the preamble guard refused the
  // cache the routing below it was written to serve.
  const std::vector<float> logits = RunSeamForward(c, w, fp8);
  REQUIRE(logits.size() ==
          kSeamTokens.size() * static_cast<size_t>(c.vocab_size));
  for (float v : logits) REQUIRE(std::isfinite(v));

  // The STORE ran through the seam: the half-width pages carry bytes now. Zero
  // here is what a store that silently skipped the fp8 arm would leave.
  for (const auto& page : fp8.buf) CHECK(NonZeroBytes(page) > 0);

  // Deterministic over a fresh cache — an fp8 KV store is still a function of
  // its inputs.
  SeamCachePool again(c, DType::kI8, vt::Fp8KVCacheDataType::kFp8E4M3, 1.0F,
                      1.0F);
  const std::vector<float> repeat = RunSeamForward(c, w, again);
  REQUIRE(repeat.size() == logits.size());
  CHECK(std::memcmp(repeat.data(), logits.data(),
                    logits.size() * sizeof(float)) == 0);
}

TEST_CASE("kv-fp8 W3 G12: the seam's fp8 cache is really QUANTIZED, not float") {
  // The counter-case to the one above, and the reason "it ran and produced
  // finite numbers" is not enough. An fp8 cache that behaved identically to a
  // bf16 one would mean the read never dequantized — and `ApplyKvCacheQuant` is
  // the only thing that tells the paged kernel to. Four mantissa bits against
  // bf16's eight is a difference the logits carry.
  const HfConfig c = MakeSeamConfig();
  const vllm::Qwen3DenseWeights w = MakeSeamWeights(c);

  SeamCachePool bf16(c, DType::kBF16, vt::Fp8KVCacheDataType::kAuto, 1.0F, 1.0F);
  const std::vector<float> float_logits = RunSeamForward(c, w, bf16);

  SeamCachePool fp8(c, DType::kI8, vt::Fp8KVCacheDataType::kFp8E4M3, 1.0F, 1.0F);
  const std::vector<float> fp8_logits = RunSeamForward(c, w, fp8);

  REQUIRE(float_logits.size() == fp8_logits.size());
  REQUIRE(!float_logits.empty());
  // The fp8 page is EXACTLY half the bf16 one, which is the memory the feature
  // buys and the sizing the store has to agree with.
  REQUIRE(fp8.buf[0].size() * 2 == bf16.buf[0].size());

  size_t differing = 0;
  double max_abs = 0.0;
  for (size_t i = 0; i < fp8_logits.size(); ++i) {
    if (fp8_logits[i] != float_logits[i]) ++differing;
    max_abs = std::max(max_abs, static_cast<double>(std::fabs(
                                    fp8_logits[i] - float_logits[i])));
  }
  CHECK(differing > 0);
  // And it is a QUANTIZATION difference rather than a wrong-offset read: an
  // fp8 store indexed at the float width would land the second half of every
  // page outside the tokens it wrote and the logits would not track at all.
  CHECK(max_abs < 1.0);
  MESSAGE("fp8 vs bf16 cache: " << differing << "/" << fp8_logits.size()
                                << " logits differ, max |delta| " << max_abs);
}

namespace {

// The e4m3 ROUND-TRIP ENVELOPE. Every constant below is read off the FORMAT
// (`vt::F32ToF8E4M3`, `include/vt/fp8_kv.h`) rather than fitted to a measured
// delta, because a threshold sized to today's number is the same hole one
// decimal place tighter.
//
// fp8-e4m3fn carries three explicit mantissa bits and rounds to nearest even.
// For a NORMAL magnitude in [2^e, 2^(e+1)) the grid step is 2^(e-3), so the
// rounding error is at most the half step 2^(e-4), which is at most 2^-4 of the
// value. Below the smallest normal (2^-6) the grid is uniform at 2^-9, so the
// error is at most the half step 2^-10 in ABSOLUTE terms. The store divides by
// the scale and the read multiplies by it (`quant_utils.cuh:296-308`), so the
// absolute arm carries a factor of the scale and the relative arm does not.
constexpr double kE4m3RelHalfUlp = 1.0 / 16.0;      // 2^-4
constexpr double kE4m3AbsHalfStep = 1.0 / 1024.0;   // 2^-10
constexpr double kE4m3SmallestNormal = 1.0 / 64.0;  // 2^-6
constexpr double kE4m3Max = 448.0;

// Non-unit, and DIFFERENT per side, so a k/v scale that is swapped, dropped, or
// applied at one end only leaves the envelope instead of staying inside it.
// Both are below one because the store divides by the scale: this toy model's
// layer-0 K and V land around 1e-2, and a scale above one would push most of
// them under e4m3's smallest normal (2^-6), where only the weaker absolute arm
// of the envelope applies. The values below keep every page's largest element a
// normal, which the `pages_with_a_normal` assertion holds them to.
constexpr float kEnvKScale = 0.125F;
constexpr float kEnvVScale = 0.25F;

// Two INVARIANCE scale pairs, for the half the envelope above cannot reach: the
// production READ. Every one of the four is a power of two, and every one keeps
// EVERY layer-0 element a normal — the measured magnitudes are 1.76e-4 to 1.32e-1
// for K and 5.41e-5 to 4.22e-2 for V, so the all-normal window (max/448, min*64]
// is (2.94e-4, 1.13e-2] for K and (9.43e-5, 3.46e-3] for V and all four sit
// inside it. That is what makes the comparison EXACT rather than a tolerance
// for the elements it covers. The `scale_exact` REQUIRE below holds LAYER 0's
// 320 elements to that window, which is 320 of the 640 each run stores; layer 1
// does NOT satisfy it, and the comment on the comparison itself says what that
// costs.
//
// The two pairs differ on BOTH sides, and deliberately: a pair that moved only
// one side would let a read-side defect that depends on the OTHER scale
// reproduce itself identically in both runs and cancel out of the comparison.
constexpr float kInvAKScale = 1.0F / 128.0F;   // 2^-7
constexpr float kInvAVScale = 1.0F / 8192.0F;  // 2^-13
constexpr float kInvBKScale = 1.0F / 2048.0F;  // 2^-11
constexpr float kInvBVScale = 1.0F / 512.0F;   // 2^-9

// Flat element index into a (num_blocks, 2, block_size, Hkv, Dh) contiguous KV
// buffer — the layout `KvSlice` (`src/vllm/model_executor/models/qwen3_5.cpp`)
// views, with `which` 0 = K and 1 = V.
size_t SeamKvIndex(const HfConfig& c, int which, int64_t slot, int64_t h,
                   int64_t d) {
  const int64_t H = c.num_key_value_heads, D = c.head_dim;
  const int64_t block = slot / kSeamBlockSize, off = slot % kSeamBlockSize;
  return static_cast<size_t>(
      ((block * 2 + which) * kSeamBlockSize + off) * H * D + h * D + d);
}

double SeamBf16At(const std::vector<uint8_t>& page, size_t elem) {
  uint16_t raw = 0;
  std::memcpy(&raw, page.data() + elem * sizeof(uint16_t), sizeof(uint16_t));
  return static_cast<double>(vt::BF16ToF32(raw));
}

}  // namespace

TEST_CASE(
    "kv-fp8 W3 G12: the fp8 pages hold THIS layer's K and V, inside the e4m3 "
    "round-trip envelope") {
  // WHAT A LOGIT COMPARISON CANNOT SEE. "The fp8 logits differ from the bf16
  // logits by less than one" is satisfied by a cache that stores V where K
  // belongs, and by a store that divides by a scale the read never multiplies
  // back. Both leave this toy model's logits within a thousandth, so a bound
  // stated on the logits measures the model's insensitivity and not the cache.
  // This case compares the CACHE BYTES, where a mis-routed or mis-scaled store
  // is an O(1) relative error and the correct answer is bounded by the format.
  //
  // LAYER 0 ONLY, and that is the point rather than a limitation: its K and V
  // are functions of the embedding and the input layernorm alone, so the bf16
  // run and the fp8 run hand the store BIT-IDENTICAL floats and the float run's
  // page IS the reference the fp8 page has to round. From layer 1 on, the fp8
  // run's inputs already carry the previous layer's dequantization and no
  // per-element envelope holds.
  const HfConfig c = MakeSeamConfig();
  const vllm::Qwen3DenseWeights w = MakeSeamWeights(c);

  SeamCachePool bf16(c, DType::kBF16, vt::Fp8KVCacheDataType::kAuto, kEnvKScale,
                     kEnvVScale);
  const std::vector<float> float_logits = RunSeamForward(c, w, bf16);
  REQUIRE(!float_logits.empty());

  SeamCachePool fp8(c, DType::kI8, vt::Fp8KVCacheDataType::kFp8E4M3, kEnvKScale,
                    kEnvVScale);
  const std::vector<float> fp8_logits = RunSeamForward(c, w, fp8);
  REQUIRE(fp8_logits.size() == float_logits.size());

  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  size_t outside = 0, normals = 0, elems = 0;
  size_t pages_with_a_normal = 0;
  double worst_ratio = 0.0, worst_delta = 0.0;
  double max_ref[2] = {0.0, 0.0};
  int worst_which = -1;
  int64_t worst_slot = -1, worst_head = -1, worst_dim = -1;

  for (int which = 0; which < 2; ++which) {
    const double scale = which == 0 ? kEnvKScale : kEnvVScale;
    for (size_t t = 0; t < kSeamTokens.size(); ++t) {
      const int64_t slot = static_cast<int64_t>(t) % kSeamBlockSize;
      bool page_has_a_normal = false;
      for (int64_t h = 0; h < Hkv; ++h) {
        for (int64_t d = 0; d < Dh; ++d) {
          const size_t i = SeamKvIndex(c, which, slot, h, d);
          // The float run stored the model-dtype element verbatim, so this IS
          // the value the fp8 store was handed.
          const double ref = SeamBf16At(bf16.buf[0], i);
          const double got = static_cast<double>(
              vt::LoadKvFp8E4M3(fp8.buf[0][i], static_cast<float>(scale)));
          const double bound =
              kE4m3RelHalfUlp * std::fabs(ref) + kE4m3AbsHalfStep * scale;
          const double delta = std::fabs(got - ref);
          ++elems;
          max_ref[which] = std::max(max_ref[which], std::fabs(ref));
          if (std::fabs(ref) >= kE4m3SmallestNormal * scale) {
            ++normals;
            page_has_a_normal = true;
          }
          if (delta > bound) {
            ++outside;
            if (delta / bound > worst_ratio) {
              worst_ratio = delta / bound;
              worst_delta = delta;
              worst_which = which;
              worst_slot = slot;
              worst_head = h;
              worst_dim = d;
            }
          }
        }
      }
      if (page_has_a_normal) ++pages_with_a_normal;
    }
  }

  // Printed BEFORE the assertions, so a red run still says what it measured.
  MESSAGE("layer-0 KV envelope: "
          << outside << "/" << elems
          << " elements outside 2^-4*|ref| + 2^-10*scale; worst ratio "
          << worst_ratio << " (delta " << worst_delta
          << ", which=" << worst_which << " slot=" << worst_slot
          << " head=" << worst_head << " dim=" << worst_dim
          << "), max|ref| k=" << max_ref[0] << " v=" << max_ref[1]
          << ", normals " << normals << "/" << elems << " over "
          << pages_with_a_normal << " pages");

  // ANTI-VACUITY, stated structurally rather than as a fitted fraction. The
  // relative arm only bites on magnitudes the format stores as NORMALS; a token
  // page that held only subnormals — or that the store never wrote at all, and
  // so reads back as zeros — would satisfy the absolute arm alone and prove
  // nothing. Every one of the ten pages (five tokens, K and V) has to carry at
  // least one normal.
  REQUIRE(elems == 2 * kSeamTokens.size() * static_cast<size_t>(Hkv * Dh));
  REQUIRE(pages_with_a_normal == 2 * kSeamTokens.size());
  // Nothing saturated, so the bound above is the pure rounding envelope and not
  // a clamp: e4m3's finite maximum is 448 and the store divides by the scale.
  REQUIRE(max_ref[0] < kE4m3Max * kEnvKScale);
  REQUIRE(max_ref[1] < kE4m3Max * kEnvVScale);

  CHECK(outside == 0);

  // ---- AND NOW THE READ, which nothing above this line gates. ----
  //
  // Everything above measures the STORE. It decodes the cache bytes with the
  // test's own `vt::LoadKvFp8E4M3` and never enters the production dequant
  // (`cpu_paged_attn.cpp:167`), and `fp8_logits` — the one value here that IS
  // downstream of that dequant — carries no assertion at all. Every other case
  // in this file that asserts a number downstream of the read runs at
  // `k_scale == v_scale == 1`, where a k/v scale SWAP on the read is
  // arithmetically inert. So a read that dequantizes V with K's scale, halving
  // every V the softmax sees whenever the two scales differ, walks this whole
  // file green.
  //
  // The gate below is EXACT rather than a tolerance. For the elements that meet
  // its precondition that is a property of the FORMAT and not a lucky
  // measurement; how far the precondition is actually ASSERTED is a separate
  // question, and the block after the identity answers it. e4m3fn's NORMAL
  // grid is relative: for |y| in [2^e, 2^(e+1)) the representable points are
  // m*2^(e-3). Dividing by a power of two is exact in binary floating point and
  // shifts `e` without touching the mantissa, so for any two power-of-two
  // scales s and s' that both leave a value normal and unsaturated,
  //
  //     s * Dequant(Quantize(x / s))  ==  s' * Dequant(Quantize(x / s'))
  //
  // bit for bit. The cache BYTES of the two runs differ — a different exponent
  // field in every element — and the floats the attention kernel is handed do
  // not. There is no constant to fit and none to widen later, which is the same
  // discipline the envelope above is written to.
  //
  // WHAT IS ASSERTED, AND WHAT IS ONLY MEASURED. `scale_exact` below reads
  // `bf16.buf[0]`, so it holds that precondition on LAYER 0 — 320 of the 640
  // elements each run stores, `MakeSeamConfig` setting num_hidden_layers = 2 —
  // while `inv_differing` compares LOGITS, which are a function of BOTH layers'
  // caches. Layer 1 does NOT satisfy the precondition. Decoding each run's own
  // `buf[1]` at its own scales, 3 of its 320 elements disagree, by up to
  // 7.62939e-06: all K-side, an order of magnitude below layer 0's 1.76e-4
  // minimum, so at kInvAKScale = 2^-7 they fall in e4m3's SUBNORMAL region
  // where the grid is the absolute 2^-9 step and the power-of-two covariance
  // does not hold. That layer also carries 2 SATURATED elements, the other
  // escape. So `CHECK(inv_differing == 0)` is exact-by-format for layer 0 and,
  // for layer 1, an EMPIRICAL result for this fixture: a 7.6e-6 cache
  // perturbation absorbed in f32 accumulation before it reaches a logit. Not
  // knife-edge today — a third legal pair (2^-9, 2^-11) against pair A also
  // measures 0/320 logits differing at max |delta| 0 — but a change to
  // MakeSeamWeights, kSeamTokens, num_hidden_layers, the thread count or the
  // accumulation order could push a layer-1 discrepancy into a logit and redden
  // a CORRECT tree with a defect-shaped message. Recorded under `## Owed` in
  // `.agents/specs/fp8-kv-cache.md`.
  //
  // A read that uses the wrong scale breaks this and cannot hide, because the
  // two pairs differ on both sides: dequantizing V with K's scale multiplies V
  // by 64 in one run and by 1/4 in the other, and dropping a read scale
  // altogether multiplies by 2^13 against 2^9.

  // ANTI-VACUITY for that exactness. The identity holds only while every element
  // stays a NORMAL at all four scales and nothing saturates. One subnormal would
  // round on the ABSOLUTE 2^-9 grid, which is not scale invariant, and the
  // comparison below would then be measuring the data instead of the read. Count
  // the elements the identity actually covers and require all of them. (An exact
  // zero is scale invariant on its own and counts; the measured population has
  // none.) LAYER 0 only, for the reason the block above gives.
  size_t scale_exact = 0;
  for (int which = 0; which < 2; ++which) {
    const float sa = which == 0 ? kInvAKScale : kInvAVScale;
    const float sb = which == 0 ? kInvBKScale : kInvBVScale;
    for (size_t t = 0; t < kSeamTokens.size(); ++t) {
      const int64_t slot = static_cast<int64_t>(t) % kSeamBlockSize;
      for (int64_t h = 0; h < Hkv; ++h) {
        for (int64_t d = 0; d < Dh; ++d) {
          const double ref = std::fabs(
              SeamBf16At(bf16.buf[0], SeamKvIndex(c, which, slot, h, d)));
          const bool exact_a =
              ref >= kE4m3SmallestNormal * sa && ref < kE4m3Max * sa;
          const bool exact_b =
              ref >= kE4m3SmallestNormal * sb && ref < kE4m3Max * sb;
          if (ref == 0.0 || (exact_a && exact_b)) ++scale_exact;
        }
      }
    }
  }
  REQUIRE(scale_exact == elems);

  SeamCachePool inv_a(c, DType::kI8, vt::Fp8KVCacheDataType::kFp8E4M3,
                      kInvAKScale, kInvAVScale);
  const std::vector<float> inv_a_logits = RunSeamForward(c, w, inv_a);
  SeamCachePool inv_b(c, DType::kI8, vt::Fp8KVCacheDataType::kFp8E4M3,
                      kInvBKScale, kInvBVScale);
  const std::vector<float> inv_b_logits = RunSeamForward(c, w, inv_b);
  REQUIRE(inv_a_logits.size() == float_logits.size());
  REQUIRE(inv_b_logits.size() == float_logits.size());

  // The two runs really do hold DIFFERENT bytes, so the comparison below is a
  // statement about the read and not about two identical buffers: a store that
  // ignored its scale would write the same page twice and every logit would
  // match while proving nothing.
  REQUIRE(inv_a.buf[0].size() == inv_b.buf[0].size());
  CHECK(std::memcmp(inv_a.buf[0].data(), inv_b.buf[0].data(),
                    inv_a.buf[0].size()) != 0);

  size_t inv_differing = 0;
  double inv_max_abs = 0.0;
  for (size_t i = 0; i < inv_a_logits.size(); ++i) {
    if (inv_a_logits[i] != inv_b_logits[i]) ++inv_differing;
    inv_max_abs = std::max(
        inv_max_abs,
        static_cast<double>(std::fabs(inv_a_logits[i] - inv_b_logits[i])));
  }
  MESSAGE("read-side scale invariance: "
          << inv_differing << "/" << inv_a_logits.size()
          << " logits differ between k/v scales (" << kInvAKScale << ", "
          << kInvAVScale << ") and (" << kInvBKScale << ", " << kInvBVScale
          << "), max |delta| " << inv_max_abs << "; scale-exact elements "
          << scale_exact << "/" << elems);

  CHECK(inv_differing == 0);
}
