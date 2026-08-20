// SPEC-DFLASH2 W3 (#1314) — the PRODUCTION reachability gate for the DFlash2
// candidate selector, and the discharge of spec `## Owed` O5 and O7.
//
// WHAT O5 AND O7 SAID, AND WHY THIS FILE EXISTS. W1 and W2 both landed with
// their production call site UNGATED and mutation-proven so: deleting
// `RefuseDflash2CandidateSelector` from `GPUModelRunner::propose_drafts_block`,
// or `draft->weights.conv_block_size = draft->k + 1;` from `LoadDflashDraft`,
// left every suite in this repository green. The stated reason was that a gate
// on the runner would need an on-disk target plus an on-disk draft driven
// through the loader, plus a populated per-request device KV store, and that
// this was W4's harness.
//
// It was not. What was missing was a way to hand a DFlash draft to a LoadedEngine
// built from IN-MEMORY weights -- the exact seam `mtp_weights` already had, and
// which its own header comment justifies in the same words: "a synthetic spec
// engine could only ever run with a NULL drafter, which is exactly the state a
// depth gate must not mistake for working speculation". W3 adds that overload,
// and everything downstream is the production path unchanged: ResolveSpecConfig,
// the aux-multi-tap refusal, `set_dflash_draft`, `propose_drafts` ->
// `propose_drafts_dflash` -> `propose_drafts_block`.
//
// WHAT THIS GATE ASSERTS, and why the assertion is the refusal's TEXT. A DFlash2
// draft cannot generate: its path walk is W4, and the engine refuses by name at
// `RefuseDflash2PathWalk`. That refusal carries the lattice the selector
// produced -- scored transitions, requests, steps, top_k -- so the message
// proves three things at once that no exit status could:
//
//   1. the runner entered `propose_drafts_block` (nothing else raises this),
//   2. the block forward ran there (the counts are sized from its output),
//   3. the CANDIDATE SELECTOR ran there (the counts ARE its output).
//
// Delete the `Dflash2SelectCandidates` call in `runner.cpp` and pass a
// default-constructed state -- which compiles -- and every count reads zero.
//
// The target is the same tiny synthetic Qwen3.5 DENSE model
// tests/vllm/v1/spec_decode/test_mtp_depth.cpp builds, because
// `supports_aux_multi_tap()` is true only for the Qwen3.5/3.6 dense and MoE
// forwards and the loader refuses any other target by name.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::DenseMlpWeights;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseWeights;
using vllm::StTensor;
using vllm::TensorResolver;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::entrypoints::DflashDraft;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vt::DType;

namespace {

// ─── Synthetic dense weights (mirrors test_loaded_engine_dense.cpp) ──────────
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

constexpr int kVocab = 24;      // == the tiny BPE fixture's ids 0..23, no holes.
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
  // mtp_num_hidden_layers == 1, which is what both gate checkpoints ship and
  // what LoadedEngine::ResolveSpecConfig reads to resolve n_predict.
  c.raw = json::object();
  c.raw["mtp_num_hidden_layers"] = 1;
  return c;
}

DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

Qwen3_5DenseWeights MakeDenseWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
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
    Qwen3_5DenseLayerWeights lw;
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

// The tiny oracle-verified BPE fixture (ids 0..23, no holes).
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_mtp_depth_tok_" + std::to_string(counter++) + ".json"))
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
  vocab[MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 23;
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
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

SamplingParams Greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax) -> deterministic and depth-neutral.
  sp.max_tokens = max_tokens;
  sp.output_kind = RequestOutputKind::kCumulative;
  return sp;
}

// The GDN state dtype the CPU tier can run a spec VERIFY in.
//
// The gate checkpoints are GDN hybrids and their state storage is bf16 (the
// model dtype), but `vt::CausalConv1dSpecUpdate` requires an f32 conv state and
// rejects bf16 off CUDA (src/vt/ops.cpp:1806). That is a property of the GDN
// speculative ROLLBACK path, which this row does not touch: the MTP head is
// itself declared layer_type="full_attention" (qwen3_5_mtp.py:105-112) and every
// draft decode step reads and writes only the draft's own paged full-attention
// KV layer. So the CPU depth gate runs the f32-state arm through the existing
// same-binary escape MakeQwen3_5KVCacheSpec already reads
// (qwen3_5_common.cpp:54-63), and the bf16 arm stays covered by the DGX gate
// this row records as owed. Set once per process, before any engine is built.
struct F32GdnState {
  F32GdnState() { setenv("VT_GDN_STATE_BF16", "0", /*overwrite=*/1); }
};
const F32GdnState kF32GdnState;


// ─── The synthetic DFlash2 draft ────────────────────────────────────────────
// Built through the PRODUCTION weight loader (`vllm::LoadQwen3DFlash`) over a
// resolver holding the exact tensor names the published checkpoint ships, so
// the shape assertions, the conv geometry read and the selector geometry read
// are all the ones a real checkpoint meets. Nothing here fills
// `Qwen3DFlashWeights` by hand.
constexpr int kDraftLayers = 2;
constexpr int kConvTaps = 2;
constexpr int kConvGroup = 8;   // hidden 32 / 8 = 4 groups
constexpr int kSelRank = 4;
constexpr int kSelTopK = 3;
constexpr int kSpecTokens = 3;  // k; the conv's query block is 1 + k = 4

class DraftTensorStore {
 public:
  void Add(const std::string& name, std::vector<int64_t> shape, uint64_t seed) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    Stored& st = tensors_[name];
    st.values.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      st.values[static_cast<size_t>(i)] =
          vt::F32ToBF16(RandV(seed * 977 + static_cast<uint64_t>(i)));
    st.view.dtype = "BF16";
    st.view.shape = std::move(shape);
    st.view.data = reinterpret_cast<const uint8_t*>(st.values.data());
    st.view.nbytes = st.values.size() * sizeof(uint16_t);
  }
  TensorResolver Resolver() const {
    return [this](const std::string& name) -> const StTensor& {
      auto it = tensors_.find(name);
      if (it == tensors_.end())
        throw std::runtime_error("qwen3_dflash: tensor not found: " + name);
      return it->second.view;
    };
  }

 private:
  struct Stored {
    std::vector<uint16_t> values;
    StTensor view;
  };
  std::map<std::string, Stored> tensors_;
};

// The draft's own config.json, as `MakeQwen3FlashDraftConfig` would produce it.
// `target_layer_ids` name TARGET layers, so they must sit inside the target's
// four; the aux multi-tap is captured at exactly these.
HfConfig MakeDraftConfig(const HfConfig& target, bool muse_glimmer_scalars) {
  HfConfig c;
  c.hidden_size = target.hidden_size;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.rotary_dim = 8;
  c.rope_theta = 10000.0;
  c.intermediate_size = 16;
  c.vocab_size = target.vocab_size;
  c.num_hidden_layers = kDraftLayers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 2048;
  c.layer_types =
      std::vector<std::string>(static_cast<size_t>(kDraftLayers), "sliding_attention");
  c.raw = json::object();
  c.raw["dflash_config"] = json::object();
  c.raw["dflash_config"]["mask_token_id"] = target.vocab_size - 1;
  c.raw["dflash_config"]["target_layer_ids"] = json::array({1, 3});
  c.raw["dflash_config"]["conv_kernel_size"] = kConvTaps;
  c.raw["dflash_config"]["conv_group_size"] = kConvGroup;
  c.raw["dflash_config"]["block_size"] = kSpecTokens + 1;
  c.raw["dflash_config"]["selector_rank"] = kSelRank;
  c.raw["dflash_config"]["selector_top_k"] = kSelTopK;
  if (muse_glimmer_scalars) {
    // `z-lab/Muse-Glimmer-30B-DFlash2`'s own values (#1327, D9).
    c.raw["dflash_config"]["output_multiplier"] = 0.19611613513818404;
    c.raw["dflash_config"]["final_logit_softcapping"] = 20.0;
  }
  c.raw["block_size"] = kSpecTokens + 1;
  c.raw["is_causal"] = false;
  return c;
}

std::unique_ptr<DflashDraft> MakeDflash2Draft(const HfConfig& target,
                                              bool muse_glimmer_scalars) {
  static DraftTensorStore store;
  static bool filled = false;
  const HfConfig c = MakeDraftConfig(target, muse_glimmer_scalars);
  const int64_t H = c.hidden_size, V = c.vocab_size, I = c.intermediate_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t taps_fc = 2;  // len(target_layer_ids)
  const int64_t groups = H / kConvGroup;
  if (!filled) {
    store.Add("fc.weight", {H, H * taps_fc}, 1);
    store.Add("hidden_norm.weight", {H}, 2);
    store.Add("norm.weight", {H}, 3);
    for (int64_t l = 0; l < kDraftLayers; ++l) {
      const std::string b = "layers." + std::to_string(l) + ".";
      const uint64_t s = 100 + static_cast<uint64_t>(l) * 50;
      store.Add(b + "input_layernorm.weight", {H}, s + 1);
      store.Add(b + "post_attention_layernorm.weight", {H}, s + 2);
      store.Add(b + "self_attn.q_proj.weight", {Hq * Dh, H}, s + 3);
      store.Add(b + "self_attn.k_proj.weight", {Hkv * Dh, H}, s + 4);
      store.Add(b + "self_attn.v_proj.weight", {Hkv * Dh, H}, s + 5);
      store.Add(b + "self_attn.o_proj.weight", {H, Hq * Dh}, s + 6);
      store.Add(b + "self_attn.q_norm.weight", {Dh}, s + 7);
      store.Add(b + "self_attn.k_norm.weight", {Dh}, s + 8);
      store.Add(b + "mlp.gate_proj.weight", {I, H}, s + 9);
      store.Add(b + "mlp.up_proj.weight", {I, H}, s + 10);
      store.Add(b + "mlp.down_proj.weight", {H, I}, s + 11);
      for (const char* which : {"attention_conv.", "mlp_conv."}) {
        store.Add(b + which + "base_kernel", {2, kConvTaps, H}, s + 12);
        store.Add(b + which + "kernel_projection.weight",
                  {2 * kConvTaps * groups, H}, s + 13);
      }
    }
    store.Add("candidate_selector.hidden_projection.weight", {kSelRank, H}, 900);
    store.Add("candidate_selector.predecessor_codebook", {V, kSelRank}, 901);
    store.Add("candidate_selector.successor_codebook", {V, kSelRank}, 902);
    filled = true;
  }
  auto draft = std::make_unique<DflashDraft>();
  draft->config = c;
  draft->k = kSpecTokens;
  draft->weights = vllm::LoadQwen3DFlash(
      store.Resolver(), c, taps_fc,
      /*mask_token_id=*/static_cast<int32_t>(V - 1));
  // What the loader's SharedHeadSource does: the draft SHARES the target's
  // embed_tokens and lm_head. Built to the same shapes the safetensors arm
  // produces -- [vocab, H] with nk=false for the gather table and the same
  // [vocab, H] with nk=true for the MatmulBT head.
  draft->weights.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 950);
  draft->weights.lm_head = MakeOwned(DType::kBF16, {V, H}, 951);
  draft->weights.lm_head.nk = true;
  draft->weights.draft_vocab_size = V;
  // And what `LoadDflashDraft` does with the resolved k: the conv's block is
  // 1 + k, never the checkpoint key.
  draft->weights.conv_block_size = kSpecTokens + 1;
  return draft;
}

// `--speculative-config` for method "dflash" REQUIRES a `model` key naming the
// draft checkpoint, and `ResolveSpecConfig` classifies that path through
// `CheckDflash2DraftArm` before resolving anything. So this writes a scratch
// directory holding the published DFlash2 draft's `config.json` SHAPE -- which
// is what the classification reads -- and names it. The WEIGHTS still come from
// memory, through the overload this wave adds; the directory is what the
// production classification step consumes, and driving it here means the startup
// NOTICE `CheckDflash2DraftArm` prints is on this path too rather than bypassed.
class ScratchDraftDir {
 public:
  ScratchDraftDir() {
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
           ("vllmcpp_dflash2_reach_" + std::to_string(counter++) + "_" +
            std::to_string(static_cast<long long>(::getpid())));
    std::filesystem::create_directories(dir_);
    json c;
    c["architectures"] = json::array({"DFlash2DraftModel"});
    c["model_type"] = "qwen3";
    std::ofstream((dir_ / "config.json").string(), std::ios::binary) << c.dump();
  }
  ~ScratchDraftDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  ScratchDraftDir(const ScratchDraftDir&) = delete;
  ScratchDraftDir& operator=(const ScratchDraftDir&) = delete;
  std::string path() const { return dir_.string(); }

 private:
  std::filesystem::path dir_;
};

EngineParams DflashSpecParams(const ScratchDraftDir& dir) {
  EngineParams p;
  p.speculative_config = vllm::ParseSpeculativeConfigJson(
      R"({"method":"dflash","num_speculative_tokens":)" +
      std::to_string(kSpecTokens) + R"(,"model":")" + dir.path() + R"("})");
  return p;
}

// Capture whatever the engine writes to std::cerr for the lifetime of the scope.
class CerrCapture {
 public:
  CerrCapture() : old_(std::cerr.rdbuf(buf_.rdbuf())) {}
  ~CerrCapture() { std::cerr.rdbuf(old_); }
  CerrCapture(const CerrCapture&) = delete;
  CerrCapture& operator=(const CerrCapture&) = delete;
  std::string str() const { return buf_.str(); }

 private:
  std::ostringstream buf_;
  std::streambuf* old_;
};

// Generate through the engine and return whatever it threw, or "" on success.
std::string GenerateAndCatch(LoadedEngine& eng, const std::string& prompt) {
  try {
    (void)eng.engine().generate(prompt, Greedy(8), "req");
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

}  // namespace

TEST_CASE("dflash2 runner: the selector RUNS inside propose_drafts_block") {
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                   DflashSpecParams(dir), MakeDflash2Draft(target, false));
  const std::string what = GenerateAndCatch(eng, "hello");
  INFO("what: ", what);
  REQUIRE_FALSE(what.empty());

  // (1) The runner entered the DFlash propose. Nothing else raises this.
  CHECK(what.find("PATH WALK") != std::string::npos);
  // (2) and (3) The block forward and the CANDIDATE SELECTOR ran there, because
  // these counts ARE the selector's own output on this block: one proposing
  // request, k = 3 steps, selector_top_k = 3, so 1*3*3*3 = 27 scored
  // transitions. Delete the `Dflash2SelectCandidates` call in
  // `GPUModelRunner::propose_drafts_block` and pass a default-constructed state
  // -- which compiles -- and all four read zero.
  CHECK(what.find("scored-transitions=27") != std::string::npos);
  CHECK(what.find("requests=1") != std::string::npos);
  CHECK(what.find("steps=3") != std::string::npos);
  CHECK(what.find("top_k=3") != std::string::npos);
}

TEST_CASE("dflash2 runner: the STARTUP notice names the boundary that is real") {
  // `## Risks/decisions` D10 pays for the moved refusal with a startup notice,
  // and W3 moves the boundary again -- so the notice has to move with it. A
  // notice still naming the selector as the missing mechanism would send a user
  // to wait for a wave that has landed.
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  std::string captured;
  {
    CerrCapture cap;
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir), MakeDflash2Draft(target, false));
    captured = cap.str();
  }
  INFO("notice: ", captured);
  CHECK(captured.find("DFlash2DraftModel") != std::string::npos);
  CHECK(captured.find("CANDIDATE SELECTOR are implemented") != std::string::npos);
  CHECK(captured.find("PATH WALK is not implemented") != std::string::npos);
  CHECK(captured.find("wave W4") != std::string::npos);
  CHECK(captured.find("#1314") != std::string::npos);
}

TEST_CASE("dflash2 runner: the DEFAULT scalars and Muse Glimmer's both reach the runner") {
  // `## Risks/decisions` D9 at the PRODUCTION entry. Both drafts run the same
  // block and reach the same refusal; what differs is the candidate values the
  // selector scored, and the shape of the lattice does not change with them --
  // so the counts are the same and this case's value is that the SCALARS do not
  // break the production path, not that they move it (the model suite gates
  // that they move it).
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  LoadedEngine muse(target, MakeDenseWeights(target), BuildFixture(),
                    DflashSpecParams(dir), MakeDflash2Draft(target, true));
  const std::string what = GenerateAndCatch(muse, "hello");
  INFO("what: ", what);
  REQUIRE_FALSE(what.empty());
  CHECK(what.find("PATH WALK") != std::string::npos);
  CHECK(what.find("scored-transitions=27") != std::string::npos);
}
