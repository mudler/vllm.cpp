// The tiny synthetic Muse Glimmer conditional-generation checkpoint, shared by
// every gate that needs one.
//
// EXTRACTED, not written, for #607 L3: `test_muse_glimmer_wiring.cpp` has built
// this checkpoint since W4 and is still its primary consumer. The tower-skip
// gate needs the SAME bytes — a checkpoint that genuinely carries a perception
// encoder, in the real on-disk names — because the whole question it asks is
// whether those particular tensors were read. Two independently written
// checkpoints could drift into asking two different questions, and the one that
// mattered would be the one nobody was looking at.
//
// Geometry: the smallest that still exercises every wiring branch — 2 text
// layers (one RoPE + one NoPE), GQA 2:1, and a 2-layer perception encoder with
// one windowed and one full attention layer, a 4x4 image that patchifies to a
// 2x2 grid and pixel-shuffles down to exactly ONE soft token.
#ifndef VLLM_CPP_TESTS_MUSE_GLIMMER_TINY_FIXTURE_H_
#define VLLM_CPP_TESTS_MUSE_GLIMMER_TINY_FIXTURE_H_

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "support/process_id.h"  // vllm_test::ProcessId, the collision fix below
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace muse_glimmer_tiny {

constexpr int64_t kVocab = 32, kHidden = 8, kInter = 12, kTextLayers = 2;
// `kHeadDim` is 8 rather than the 4 this fixture carried while it lived inside
// the wiring test. A head size must be a multiple of 8 for FLASH_ATTN to accept
// it (`FlashAttentionBackend::supports_head_size`, the port of
// `flash_attn.py:170-178` that landed with #1344), and the tower-skip gate drives
// a real `LoadedEngine::FromModelDir`, so it has to pick an attention backend
// where the wiring gate never did. At 4 the engine refuses with "No valid
// attention backend ... {FLASH_ATTN: [head_size not supported]}" — the rule is
// upstream's and correct; the synthetic geometry was what had to move.
constexpr int64_t kHeads = 2, kKvHeads = 1, kHeadDim = 8;
constexpr int64_t kVHidden = 4, kVHeads = 1, kVLayers = 2, kVInter = 8;
constexpr int64_t kPatch = 2, kPatchT = 2, kMerge = 2, kPosGrid = 4;
constexpr int64_t kOutputDim = kVHidden * kMerge * kMerge;  // 16
constexpr int64_t kAdapter = 6;
constexpr int64_t kPatchDim = kPatchT * 3 * kPatch * kPatch;  // 24
constexpr int32_t kImageToken = 20, kVideoToken = 21;

// A deterministic, tensor-specific value. Reproduced by the caller when it
// checks what the loader put where, so a shard landing in the wrong slot is
// visible.
inline float Val(uint32_t seed, int64_t i) {
  const double x = 0.37 * static_cast<double>(seed) + 0.11 * static_cast<double>(i);
  return static_cast<float>(0.4 * std::sin(x) + 0.05 * std::cos(3.1 * x));
}
inline float Bf16Val(uint32_t seed, int64_t i) {
  return vt::BF16ToF32(vt::F32ToBF16(Val(seed, i)));
}

struct Fx {
  std::string name;
  std::vector<int64_t> shape;
  std::string bytes;
};

inline std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

inline int64_t NumEl(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

inline Fx Bf16(const std::string& name, std::vector<int64_t> shape, uint32_t seed) {
  const int64_t n = NumEl(shape);
  std::string bytes(static_cast<size_t>(n) * 2, '\0');
  for (int64_t i = 0; i < n; ++i) {
    const uint16_t bits = vt::F32ToBF16(Val(seed, i));
    std::memcpy(bytes.data() + static_cast<size_t>(i) * 2, &bits, 2);
  }
  return {name, std::move(shape), std::move(bytes)};
}

inline std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {
        {"dtype", "BF16"}, {"shape", t.shape}, {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    // The counter alone is PER PROCESS, and since this header was extracted TWO
    // binaries build it, so `ctest -j` runs two processes that both start at 0,
    // claim the same path, and let one destructor `std::remove` a file the other
    // still has mmapped. That is a SIGBUS on the next page touch, not a failed
    // assertion, so it reads as a crash in whichever suite lost the race.
    // Measured on 770ed57a6: 100 concurrent test_muse_glimmer_wiring +
    // test_tower_skip pairs produced 2 x rc 135, serial 25/25 clean.
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("muse_glimmer_tiny_" + std::to_string(vllm_test::ProcessId()) + "_" +
              std::to_string(counter++) + ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// The tiny model's config, in the CANONICAL nested layout with the checkpoint's
// real field spellings (`merge_size`, top-level `out_hidden_size` /
// `projector_hidden_size`).
inline vllm::HfConfig TinyConfig() {
  vllm::HfConfig c;
  c.architectures = {"MuseGlimmerForConditionalGeneration"};
  c.hidden_size = kHidden;
  c.num_hidden_layers = kTextLayers;
  c.vocab_size = kVocab;
  c.num_attention_heads = kHeads;
  c.raw = nlohmann::json{
      {"model_type", "muse_glimmer"},
      {"image_token_id", kImageToken},
      {"video_token_id", kVideoToken},
      {"out_hidden_size", kOutputDim},
      {"projector_hidden_size", kAdapter},
      {"text_config",
       {{"model_type", "muse_glimmer_text"},
        {"vocab_size", kVocab},
        {"hidden_size", kHidden},
        {"intermediate_size", kInter},
        {"num_hidden_layers", kTextLayers},
        {"num_attention_heads", kHeads},
        {"num_key_value_heads", kKvHeads},
        {"head_dim", kHeadDim},
        {"max_position_embeddings", 64},
        {"sliding_window", 3},
        {"rms_norm_eps", 1e-5},
        {"post_norm_eps", 1e-8},
        {"hidden_activation", "silu"},
        {"qk_scale_factor", 1.5},
        {"tie_word_embeddings", false},
        {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 500000.0}}}}},
      {"vision_config",
       {{"model_type", "muse_glimmer_vision"},
        {"hidden_size", kVHidden},
        {"num_attention_heads", kVHeads},
        {"num_hidden_layers", kVLayers},
        {"intermediate_size", kVInter},
        {"patch_size", kPatch},
        {"patch_temporal", kPatchT},
        {"merge_size", kMerge},
        {"pos_emb_height", kPosGrid},
        {"pos_emb_width", kPosGrid},
        {"layer_norm_eps", 1e-5},
        {"layer_types", {"window_attention", "full_attention"}}}}};
  return c;
}

// The synthetic checkpoint, written in the REAL on-disk names (canonical
// `model.language_model.*` text + `model.vision_tower.*` vision with SEPARATE
// q/k/v and biases) so the loader's normalization and merge both run for real.
inline std::vector<Fx> TinyTensors() {
  const int64_t qdim = kHeads * kHeadDim, kdim = kKvHeads * kHeadDim;
  std::vector<Fx> t;
  uint32_t s = 1;
  t.push_back(Bf16("model.language_model.embed_tokens.weight", {kVocab, kHidden}, s++));
  t.push_back(Bf16("model.language_model.norm.weight", {kHidden}, s++));
  t.push_back(Bf16("lm_head.weight", {kVocab, kHidden}, s++));
  for (int64_t l = 0; l < kTextLayers; ++l) {
    const std::string b = "model.language_model.layers." + std::to_string(l) + ".";
    t.push_back(Bf16(b + "input_layernorm.weight", {kHidden}, s++));
    t.push_back(Bf16(b + "post_attention_layernorm.weight", {kHidden}, s++));
    t.push_back(Bf16(b + "pre_feedforward_layernorm.weight", {kHidden}, s++));
    t.push_back(Bf16(b + "post_feedforward_layernorm.weight", {kHidden}, s++));
    t.push_back(Bf16(b + "self_attn.q_proj.weight", {qdim, kHidden}, s++));
    t.push_back(Bf16(b + "self_attn.k_proj.weight", {kdim, kHidden}, s++));
    t.push_back(Bf16(b + "self_attn.v_proj.weight", {kdim, kHidden}, s++));
    t.push_back(Bf16(b + "self_attn.o_proj.weight", {kHidden, qdim}, s++));
    t.push_back(Bf16(b + "self_attn.gate_proj.weight", {qdim, kHidden}, s++));
    t.push_back(Bf16(b + "mlp.gate_proj.weight", {kInter, kHidden}, s++));
    t.push_back(Bf16(b + "mlp.up_proj.weight", {kInter, kHidden}, s++));
    t.push_back(Bf16(b + "mlp.down_proj.weight", {kHidden, kInter}, s++));
  }
  t.push_back(Bf16("model.vision_tower.patch_embedder.patch_embedding.weight",
                   {kVHidden, kPatchDim}, s++));
  t.push_back(Bf16("model.vision_tower.patch_embedder.position_embedding_table.weight",
                   {kPosGrid * kPosGrid, kVHidden}, s++));
  t.push_back(Bf16("model.vision_tower.ln_pre.weight", {kVHidden}, s++));
  t.push_back(Bf16("model.vision_tower.ln_pre.bias", {kVHidden}, s++));
  t.push_back(Bf16("model.vision_tower.ln_post.weight", {kVHidden}, s++));
  t.push_back(Bf16("model.vision_tower.ln_post.bias", {kVHidden}, s++));
  for (int64_t l = 0; l < kVLayers; ++l) {
    const std::string b = "model.vision_tower.layers." + std::to_string(l) + ".";
    t.push_back(Bf16(b + "norm1.weight", {kVHidden}, s++));
    t.push_back(Bf16(b + "norm1.bias", {kVHidden}, s++));
    t.push_back(Bf16(b + "norm2.weight", {kVHidden}, s++));
    t.push_back(Bf16(b + "norm2.bias", {kVHidden}, s++));
    // SEPARATE q/k/v, each WITH a bias — what the released checkpoint ships.
    t.push_back(Bf16(b + "attn.q_proj.weight", {kVHidden, kVHidden}, s++));
    t.push_back(Bf16(b + "attn.q_proj.bias", {kVHidden}, s++));
    t.push_back(Bf16(b + "attn.k_proj.weight", {kVHidden, kVHidden}, s++));
    t.push_back(Bf16(b + "attn.k_proj.bias", {kVHidden}, s++));
    t.push_back(Bf16(b + "attn.v_proj.weight", {kVHidden, kVHidden}, s++));
    t.push_back(Bf16(b + "attn.v_proj.bias", {kVHidden}, s++));
    t.push_back(Bf16(b + "attn.proj.weight", {kVHidden, kVHidden}, s++));
    t.push_back(Bf16(b + "attn.proj.bias", {kVHidden}, s++));
    t.push_back(Bf16(b + "mlp.fc1.weight", {kVInter, kVHidden}, s++));
    t.push_back(Bf16(b + "mlp.fc1.bias", {kVInter}, s++));
    t.push_back(Bf16(b + "mlp.fc2.weight", {kVHidden, kVInter}, s++));
    t.push_back(Bf16(b + "mlp.fc2.bias", {kVHidden}, s++));
  }
  t.push_back(Bf16("model.vision_adapter.fc1.weight", {kAdapter, kOutputDim}, s++));
  t.push_back(Bf16("model.vision_adapter.fc2.weight", {kAdapter, kAdapter}, s++));
  t.push_back(Bf16("model.vision_projection.weight", {kHidden, kAdapter}, s++));
  return t;
}

// `TinyConfig()` as it appears ON DISK. `HfConfig::raw` is the config document
// MINUS the fields the struct types separately, and `architectures` is one of
// them — a config.json written from `raw` alone resolves to no architecture at
// all, which is a load failure a long way from its cause.
inline nlohmann::json TinyConfigJson() {
  nlohmann::json j = TinyConfig().raw;
  j["architectures"] = nlohmann::json::array({"MuseGlimmerForConditionalGeneration"});
  return j;
}

// The bytes of every `model.vision_tower.*` / `model.vision_adapter.*` /
// `model.vision_projection.*` tensor in the checkpoint above. This is the
// quantity the #607 L3 tower skip does not read, and it is what the RSS gate's
// threshold is stated against — computed from the fixture rather than asserted,
// so it cannot drift away from the tensors it describes.
inline int64_t TinyVisionTowerBytes() {
  int64_t total = 0;
  for (const Fx& f : TinyTensors()) {
    if (f.name.rfind("model.vision_", 0) == 0)
      total += static_cast<int64_t>(f.bytes.size());
  }
  return total;
}

// A BPE tokenizer over exactly `kVocab` single characters, so a model directory
// built from this fixture is loadable by the production entry point
// (LoadedEngine::FromModelDir) rather than only by the weight loader. Same shape
// as tests/vllm/models/fixtures/llama_embed_e2e/tokenizer.json.
inline std::string TinyTokenizerJson() {
  nlohmann::json vocab = nlohmann::json::object();
  vocab["▁"] = 0;
  // Distinct keys for every id: a JSON object silently keeps only the LAST of a
  // repeated key, so a wrap-around alphabet would quietly hand back a vocabulary
  // smaller than `kVocab` and the ids above the wrap would be unreachable.
  for (int i = 1; i < static_cast<int>(kVocab); ++i) {
    std::string tok(1, static_cast<char>('a' + ((i - 1) % 26)));
    if (i > 26) tok += std::to_string(i);
    vocab[tok] = i;
  }
  const nlohmann::json meta{
      {"type", "Metaspace"},
      {"replacement", "▁"},
      {"prepend_scheme", "always"},
      {"split", true}};
  const nlohmann::json j{
      {"version", "1.0"},
      {"pre_tokenizer", meta},
      {"decoder", meta},
      {"model",
       {{"type", "BPE"}, {"unk_token", nullptr}, {"vocab", vocab}, {"merges", nlohmann::json::array()}}},
      {"added_tokens", nlohmann::json::array()}};
  return j.dump(1);
}

// A self-deleting MODEL DIRECTORY carrying the three files a production load
// needs: config.json, model.safetensors and tokenizer.json.
class TempModelDir {
 public:
  TempModelDir() {
    // Per-process, for the reason TempFile above records: `remove_all` on a
    // directory another process is loading from is the same collision.
    static int counter = 0;
    dir_ = (std::filesystem::temp_directory_path() /
            ("muse_glimmer_tiny_dir_" + std::to_string(vllm_test::ProcessId()) +
             "_" + std::to_string(counter++)))
               .string();
    std::filesystem::create_directories(dir_);
    Write("config.json", TinyConfigJson().dump(1));
    Write("model.safetensors", BuildSt(TinyTensors()));
    Write("tokenizer.json", TinyTokenizerJson());
  }
  ~TempModelDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  TempModelDir(const TempModelDir&) = delete;
  TempModelDir& operator=(const TempModelDir&) = delete;
  const std::string& path() const { return dir_; }

 private:
  void Write(const std::string& name, const std::string& bytes) const {
    std::ofstream out(dir_ + "/" + name, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  std::string dir_;
};

}  // namespace muse_glimmer_tiny

#endif  // VLLM_CPP_TESTS_MUSE_GLIMMER_TINY_FIXTURE_H_
