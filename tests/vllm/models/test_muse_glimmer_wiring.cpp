// Muse Glimmer W4-WIRING gates: the REAL-CHECKPOINT structural contract and the
// perception-encoder -> text-tower wiring (vision load, soft-token projection,
// masked scatter, and the mm forward branch).
//
// ─── OFF-PIN HONESTY (up front) ──────────────────────────────────────────────
// Muse Glimmer does not exist at the parity pin `555967922`; the only upstream
// implementation is the still-open PR vllm#51655 at head `075d645af`, which is
// what every `muse_glimmer.py:NNNN` below cites (porting-inventory §9 deviation
// 16, specs/muse-glimmer.md §0). The pinned oracle cannot load this model, so
// there is NO throughput denominator and NO speed axis is claimable here.
//
// What these gates DO establish:
//   * `EnumerateMuseGlimmerTensors` names EXACTLY the tensors the released
//     `meta-models/Muse-Glimmer-30B` checkpoint ships — every one of its 1436
//     tensors, at the right shape, with nothing enumerated that is absent and
//     nothing present that is unaccounted.
//   * The perception encoder LOADS off a checkpoint into the W3 tower's structs,
//     with the q|k|v merge order and every bias carried across.
//   * The mm forward branch EXISTS and runs: an image prompt produces logits
//     instead of refusing, and the vision soft tokens land on the image/video
//     placeholder rows and nowhere else.
//   * The text-only path through the mm seam is BIT-IDENTICAL to the text path.
//
// What they do NOT establish: image or video END-TO-END correctness. There is no
// reference run for this checkpoint yet (the oracle cannot load it), so nothing
// here says an image produces the right tokens — only that the tower is reachable
// and that the plumbing puts its output in the right rows.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"

#include "muse_glimmer_tiny_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/muse_glimmer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::EnumerateMuseGlimmerTensors;
using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::MuseGlimmerModel;
using vllm::MuseGlimmerParams;
using vllm::MuseGlimmerWeights;
using vllm::NormalizeMuseGlimmerWeightName;
using vllm::PagedKvCache;
using vllm::ParseMuseGlimmerParams;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

namespace {

// ── the committed real-checkpoint fixture ────────────────────────────────────
// tests/vllm/models/fixtures/muse_glimmer_30b/{index.json,config.json} are the
// HEADER-ONLY projection of the released checkpoint (1436 tensors, revision
// f84ecc3a0e): every tensor's dtype and torch-storage shape, with the two
// per-layer families collapsed to a `{N}` pattern because the checkpoint is
// uniform across layers. That uniformity is not assumed — the env-gated live case
// below re-reads the real safetensors headers and requires the expansion to match
// the checkpoint tensor-for-tensor.
std::string FixtureDir() {
#ifdef MUSE_GLIMMER_CKPT_FIXTURE_DIR
  return MUSE_GLIMMER_CKPT_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/muse_glimmer_30b";
#endif
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "cannot open " << path);
  nlohmann::json j;
  in >> j;
  return j;
}

struct TensorMeta {
  std::string dtype;
  std::vector<int64_t> shape;
  bool operator==(const TensorMeta& o) const {
    return dtype == o.dtype && shape == o.shape;
  }
};

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != 0) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

TensorMeta MetaOf(const nlohmann::json& j) {
  TensorMeta m;
  m.dtype = j.at("dtype").get<std::string>();
  for (const auto& d : j.at("shape")) m.shape.push_back(d.get<int64_t>());
  return m;
}

std::string SubstituteLayer(const std::string& pattern, int64_t n) {
  const std::string tag = "{N}";
  const std::string::size_type at = pattern.find(tag);
  REQUIRE(at != std::string::npos);
  return pattern.substr(0, at) + std::to_string(n) + pattern.substr(at + tag.size());
}

// Expand the fixture into the full RAW-name -> meta map the checkpoint ships.
std::map<std::string, TensorMeta> ExpandFixture(const nlohmann::json& fx) {
  std::map<std::string, TensorMeta> out;
  for (const auto& [name, meta] : fx.at("global").items()) out[name] = MetaOf(meta);
  const int64_t nt = fx.at("num_text_layers").get<int64_t>();
  const int64_t nv = fx.at("num_vision_layers").get<int64_t>();
  for (const auto& [pattern, meta] : fx.at("text_layer").items())
    for (int64_t l = 0; l < nt; ++l) out[SubstituteLayer(pattern, l)] = MetaOf(meta);
  for (const auto& [pattern, meta] : fx.at("vision_layer").items())
    for (int64_t l = 0; l < nv; ++l) out[SubstituteLayer(pattern, l)] = MetaOf(meta);
  return out;
}

// The shape our enumeration BELIEVES each canonical name has, derived purely from
// the resolved config. This is the half that makes the structural gate more than a
// name-spelling check: a name can exist in the checkpoint and still be the wrong
// tensor, which is exactly how a merged-vs-separate qkv mistake stays quiet.
std::map<std::string, std::vector<int64_t>> ExpectedShapes(const MuseGlimmerParams& p) {
  const auto& t = p.text;
  const int64_t H = t.hidden_size;
  const int64_t qdim = t.num_attention_heads * t.head_dim;
  const int64_t kdim = t.num_key_value_heads * t.head_dim;
  std::map<std::string, std::vector<int64_t>> s;
  s["model.embed_tokens.weight"] = {t.vocab_size, H};
  s["model.norm.weight"] = {H};
  if (!t.tie_word_embeddings) s["lm_head.weight"] = {t.vocab_size, H};
  for (int64_t l = 0; l < t.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    s[b + "input_layernorm.weight"] = {H};
    s[b + "post_attention_layernorm.weight"] = {H};
    s[b + "pre_feedforward_layernorm.weight"] = {H};
    s[b + "post_feedforward_layernorm.weight"] = {H};
    s[b + "self_attn.q_proj.weight"] = {qdim, H};
    s[b + "self_attn.k_proj.weight"] = {kdim, H};
    s[b + "self_attn.v_proj.weight"] = {kdim, H};
    s[b + "self_attn.o_proj.weight"] = {H, qdim};
    if (t.use_attn_output_gate) s[b + "self_attn.output_gate_proj.weight"] = {qdim, H};
    s[b + "mlp.gate_proj.weight"] = {t.intermediate_size, H};
    s[b + "mlp.up_proj.weight"] = {t.intermediate_size, H};
    s[b + "mlp.down_proj.weight"] = {H, t.intermediate_size};
  }
  if (!p.vision.present) return s;
  const auto& v = p.vision;
  const int64_t VH = v.hidden_size;
  const int64_t VI = v.intermediate_size;
  const int64_t patch_dim = v.patch_temporal * 3 * v.patch_size * v.patch_size;
  s["vision_encoder.conv1_linear.weight"] = {VH, patch_dim};
  s["vision_encoder.positional_embedding_vlm"] = {v.pos_emb_height * v.pos_emb_width, VH};
  s["vision_encoder.ln_pre.weight"] = {VH};
  s["vision_encoder.ln_pre.bias"] = {VH};
  s["vision_encoder.ln_post.weight"] = {VH};
  s["vision_encoder.ln_post.bias"] = {VH};
  for (int64_t l = 0; l < v.num_hidden_layers; ++l) {
    const std::string b = "vision_encoder.transformer." + std::to_string(l) + ".";
    s[b + "ln_1.weight"] = {VH};
    s[b + "ln_1.bias"] = {VH};
    s[b + "ln_2.weight"] = {VH};
    s[b + "ln_2.bias"] = {VH};
    for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
      s[b + "attn." + proj + ".weight"] = {VH, VH};
      s[b + "attn." + proj + ".bias"] = {VH};
    }
    s[b + "mlp.c_fc.weight"] = {VI, VH};
    s[b + "mlp.c_fc.bias"] = {VI};
    s[b + "mlp.c_proj.weight"] = {VH, VI};
    s[b + "mlp.c_proj.bias"] = {VH};
  }
  s["vision_adapter.c_fc.weight"] = {v.adapter_dim, v.output_dim};
  s["vision_adapter.c_proj.weight"] = {v.adapter_dim, v.adapter_dim};
  s["vision_projection.weight"] = {H, v.adapter_dim};
  return s;
}

// Read the real checkpoint's tensor headers: the safetensors 8-byte little-endian
// header length + the JSON header of each shard named by the index. NO tensor
// bytes are read (the checkpoint is ~60 GB and this box has no room for it).
std::map<std::string, TensorMeta> ReadLiveHeaders(const std::string& dir) {
  const nlohmann::json index = ReadJson(dir + "/model.safetensors.index.json");
  std::set<std::string> shards;
  for (const auto& [_, f] : index.at("weight_map").items())
    shards.insert(f.get<std::string>());
  std::map<std::string, TensorMeta> out;
  for (const std::string& shard : shards) {
    std::ifstream in(dir + "/" + shard, std::ios::binary);
    REQUIRE_MESSAGE(in.good(), "cannot open shard " << shard);
    uint64_t n = 0;
    unsigned char raw[8];
    in.read(reinterpret_cast<char*>(raw), 8);
    for (int i = 7; i >= 0; --i) n = (n << 8) | raw[i];
    std::string header(static_cast<size_t>(n), '\0');
    in.read(header.data(), static_cast<std::streamsize>(n));
    const nlohmann::json hj = nlohmann::json::parse(header);
    for (const auto& [name, meta] : hj.items()) {
      if (name == "__metadata__") continue;
      out[name] = MetaOf(meta);
    }
  }
  return out;
}

// ── a tiny synthetic multimodal checkpoint ───────────────────────────────────
// The smallest geometry that still exercises every wiring branch: 2 text layers
// (one RoPE + one NoPE), GQA 2:1, and a 2-layer perception encoder with one
// windowed and one full attention layer, a 4x4 image that patchifies to a 2x2 grid
// and pixel-shuffles down to exactly ONE soft token.
// Geometry, tensor writer, config and checkpoint all live in
// muse_glimmer_tiny_fixture.h, shared with the #607 L3 tower-skip gate so both
// ask their question of the SAME bytes.
using namespace muse_glimmer_tiny;  // NOLINT(build/namespaces) — a test fixture

vt::Queue Qcpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }



// The same tiny model with `perception_emb_norm` ARMED. The released 30B leaves
// `normalize_tok_embeddings` unset, so upstream's `perception_emb_norm` is
// `nn.Identity` there (muse_glimmer.py:1469-1473) and no other config in this file
// turns it on — which is exactly why the norm needs a config of its own to be
// reachable by any assertion at all.
HfConfig TinyConfigNormalizedTokEmbeddings() {
  HfConfig c = TinyConfig();
  c.raw["text_config"]["normalize_tok_embeddings"] = true;
  return c;
}

// The norm OFF arm needs the flag set EXPLICITLY false. Omitting it means TRUE
// (#405) -- the released config omits it, and upstream defaults it on -- so a
// config that merely stays silent is the ON case, not the OFF one.
HfConfig TinyConfigNoNormalizedTokEmbeddings() {
  HfConfig c = TinyConfig();
  c.raw["text_config"]["normalize_tok_embeddings"] = false;
  return c;
}


// The seed each vision tensor was written with, recomputed the same way
// TinyTensors assigns them, so the load-order check is independent of the loader.
std::map<std::string, uint32_t> TinySeeds() {
  std::map<std::string, uint32_t> out;
  uint32_t s = 1;
  for (const Fx& f : TinyTensors()) out[f.name] = s++;
  return out;
}

// One 4x4 RGB still image: grid 2x2 patches -> 4 tokens -> 1 soft token.
vllm::multimodal::MuseGlimmerVisionImage TinyImage(uint32_t seed) {
  vllm::multimodal::MuseGlimmerVisionImage img;
  img.channels = 3;
  img.height = 4;
  img.width = 4;
  img.pixels.resize(static_cast<size_t>(3 * 4 * 4));
  for (size_t i = 0; i < img.pixels.size(); ++i)
    img.pixels[i] = Val(seed, static_cast<int64_t>(i));
  return img;
}

// Two pixel groups — one for the image placeholder, one for the video placeholder
// — so the soft-token count matches the two placeholders in MmPrompt().
std::vector<vllm::multimodal::MuseGlimmerVisionImage> TinyImages() {
  return {TinyImage(901), TinyImage(902)};
}

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(const MuseGlimmerParams& p, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = p.text.num_key_value_heads, Dh = p.text.head_dim;
    for (int64_t l = 0; l < p.text.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }
};

CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  CommonAttentionMetadata m;
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
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

// A prompt with one image placeholder and one video placeholder. Both feed the
// SAME soft-token stream (muse_glimmer.py:1592-1602).
const std::vector<int32_t>& MmPrompt() {
  static const std::vector<int32_t> p = {5, kImageToken, 9, kVideoToken, 2};
  return p;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// The real-checkpoint STRUCTURAL gate.
//
// This is the gate that catches the W0/W1 enumeration bug: it declared a MERGED
// `attn.qkv_proj.weight` for the perception encoder and omitted every vision
// attention bias, neither of which the checkpoint ships. A merged-qkv expectation
// is not a cosmetic mismatch — the loader would look for a tensor that does not
// exist, and the accounting pass would silently count the tower as partially
// present rather than saying so.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MuseGlimmer: enumeration matches the released 30B checkpoint exactly") {
  const nlohmann::json fx = ReadJson(FixtureDir() + "/index.json");
  const std::map<std::string, TensorMeta> real = ExpandFixture(fx);
  CHECK(static_cast<int64_t>(real.size()) == fx.at("total_tensors").get<int64_t>());
  CHECK(real.size() == 1436);

  vllm::HfConfig config;
  config.raw = ReadJson(FixtureDir() + "/config.json");
  const MuseGlimmerParams params = ParseMuseGlimmerParams(config);
  CHECK(params.text.num_hidden_layers == fx.at("num_text_layers").get<int64_t>());
  CHECK(params.vision.present);
  CHECK(params.vision.num_hidden_layers == fx.at("num_vision_layers").get<int64_t>());

  // Normalize every RAW checkpoint name through the mapper the loader uses.
  std::map<std::string, TensorMeta> canonical;
  std::vector<std::string> dropped;
  for (const auto& [raw, meta] : real) {
    std::string name;
    if (!NormalizeMuseGlimmerWeightName(raw, &name)) {
      dropped.push_back(raw);
      continue;
    }
    CHECK_MESSAGE(canonical.count(name) == 0,
                  "two checkpoint tensors normalize to " << name);
    canonical[name] = meta;
  }
  CHECK(dropped.empty());

  const std::vector<std::string> enumerated = EnumerateMuseGlimmerTensors(params);
  const std::map<std::string, std::vector<int64_t>> expect = ExpectedShapes(params);

  // (a) every enumerated name EXISTS in the checkpoint, at the shape we believe.
  std::set<std::string> seen;
  for (const std::string& name : enumerated) {
    CHECK_MESSAGE(seen.insert(name).second, "enumerated twice: " << name);
    const auto it = canonical.find(name);
    REQUIRE_MESSAGE(it != canonical.end(),
                    "enumerated tensor absent from the real checkpoint: " << name);
    CHECK_MESSAGE(it->second.dtype == "BF16", name << " is not BF16");
    const auto ex = expect.find(name);
    REQUIRE_MESSAGE(ex != expect.end(), "no expected shape declared for " << name);
    CHECK_MESSAGE(it->second.shape == ex->second,
                  name << " real " << ShapeStr(it->second.shape) << " != expected "
                       << ShapeStr(ex->second));
  }

  // (b) NOTHING in the checkpoint is unaccounted. A weight the enumeration never
  // names is a weight the loader never reads — silently, with plausible output.
  std::vector<std::string> unaccounted;
  for (const auto& [name, _] : canonical)
    if (seen.count(name) == 0) unaccounted.push_back(name);
  CHECK_MESSAGE(unaccounted.empty(),
                "unaccounted checkpoint tensors, first: "
                    << (unaccounted.empty() ? std::string("-") : unaccounted.front())
                    << " (" << unaccounted.size() << " total)");
  CHECK(enumerated.size() == real.size());
}

// The fixture is only worth what it faithfully records. Env-gated on the real
// checkpoint (`VLLM_MUSE_CKPT=/path/to/muse-glimmer-30b`) so CI never needs the
// NAS; reads the safetensors HEADERS only.
TEST_CASE("MuseGlimmer: the committed index fixture matches the live checkpoint") {
  const char* dir = std::getenv("VLLM_MUSE_CKPT");
  if (dir == nullptr || dir[0] == '\0') {
    MESSAGE("skipped: set VLLM_MUSE_CKPT to the Muse-Glimmer-30B checkpoint dir");
    return;
  }
  const std::map<std::string, TensorMeta> live = ReadLiveHeaders(dir);
  const std::map<std::string, TensorMeta> fixture =
      ExpandFixture(ReadJson(FixtureDir() + "/index.json"));
  CHECK(live.size() == fixture.size());
  for (const auto& [name, meta] : live) {
    const auto it = fixture.find(name);
    REQUIRE_MESSAGE(it != fixture.end(), "live tensor missing from the fixture: " << name);
    CHECK_MESSAGE(it->second == meta, name << " fixture/live metadata disagree");
  }
  for (const auto& [name, _] : fixture)
    CHECK_MESSAGE(live.count(name) == 1, "fixture tensor absent from the checkpoint: " << name);

  // The live config must also be the one the fixture pinned.
  CHECK(ReadJson(std::string(dir) + "/config.json") ==
        ReadJson(FixtureDir() + "/config.json"));
}

// ─────────────────────────────────────────────────────────────────────────────
// The WIRING gates, on a tiny synthetic checkpoint written in the REAL on-disk
// names so the loader's normalization, the q|k|v fold and the biases all run.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MuseGlimmer: the perception encoder loads, q|k|v merged in order") {
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  const HfConfig config = TinyConfig();
  const MuseGlimmerWeights w =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config);

  CHECK(w.text_loaded);
  REQUIRE(w.vision.loaded);
  // Nothing enumerated is missing from this checkpoint, and nothing is extra.
  CHECK(w.accounted_tensors == w.enumerated_tensors);
  CHECK(w.enumerated_tensors == static_cast<int64_t>(TinyTensors().size()));

  const auto& tower = w.vision;
  CHECK(tower.cfg.hidden_size == kVHidden);
  CHECK(tower.cfg.num_hidden_layers == kVLayers);
  CHECK(tower.cfg.output_dim == kOutputDim);
  CHECK(tower.cfg.adapter_dim == kAdapter);
  CHECK(tower.cfg.layer_types.size() == static_cast<size_t>(kVLayers));
  REQUIRE(tower.encoder.blocks.size() == static_cast<size_t>(kVLayers));
  CHECK(tower.encoder.conv1_w.size() == static_cast<size_t>(kVHidden * kPatchDim));
  CHECK(tower.encoder.pos_emb.size() ==
        static_cast<size_t>(kPosGrid * kPosGrid * kVHidden));
  CHECK(tower.projection.size() == static_cast<size_t>(kHidden * kAdapter));
  CHECK(tower.adapter.c_fc_w.size() == static_cast<size_t>(kAdapter * kOutputDim));
  CHECK(tower.adapter.c_proj_w.size() == static_cast<size_t>(kAdapter * kAdapter));

  // The merge ORDER is the load-bearing part: upstream views the merged operand as
  // (tokens, 3, heads, head_dim) and unbinds on dim 1 (muse_glimmer.py:611-618), so
  // rows [0,H) MUST be q, [H,2H) k and [2H,3H) v. Every shard was written with its
  // own seed, so a permutation is visible value-by-value rather than by shape.
  const std::map<std::string, uint32_t> seeds = TinySeeds();
  for (int64_t l = 0; l < kVLayers; ++l) {
    const std::string b = "model.vision_tower.layers." + std::to_string(l) + ".";
    const auto& blk = tower.encoder.blocks[static_cast<size_t>(l)];
    REQUIRE(blk.qkv_w.size() == static_cast<size_t>(3 * kVHidden * kVHidden));
    REQUIRE(blk.qkv_b.size() == static_cast<size_t>(3 * kVHidden));
    int shard = 0;
    for (const char* proj : {"q_proj", "k_proj", "v_proj"}) {
      const uint32_t ws = seeds.at(b + "attn." + proj + ".weight");
      const uint32_t bs = seeds.at(b + "attn." + proj + ".bias");
      for (int64_t i = 0; i < kVHidden * kVHidden; ++i)
        CHECK(blk.qkv_w[static_cast<size_t>(shard * kVHidden * kVHidden + i)] ==
              doctest::Approx(Bf16Val(ws, i)));
      for (int64_t i = 0; i < kVHidden; ++i)
        CHECK(blk.qkv_b[static_cast<size_t>(shard * kVHidden + i)] ==
              doctest::Approx(Bf16Val(bs, i)));
      ++shard;
    }
    // Every vision bias is carried: dropping one is a silent constant shift.
    CHECK(blk.o_b.size() == static_cast<size_t>(kVHidden));
    CHECK(blk.c_fc_b.size() == static_cast<size_t>(kVInter));
    CHECK(blk.c_proj_b.size() == static_cast<size_t>(kVHidden));
    CHECK(blk.ln_1_b.size() == static_cast<size_t>(kVHidden));
    CHECK(blk.ln_2_b.size() == static_cast<size_t>(kVHidden));
    CHECK(blk.o_b[0] == doctest::Approx(Bf16Val(seeds.at(b + "attn.proj.bias"), 0)));
    CHECK(blk.c_fc_b[0] == doctest::Approx(Bf16Val(seeds.at(b + "mlp.fc1.bias"), 0)));
    CHECK(blk.c_proj_b[0] == doctest::Approx(Bf16Val(seeds.at(b + "mlp.fc2.bias"), 0)));
  }
  CHECK(tower.encoder.ln_pre_b.size() == static_cast<size_t>(kVHidden));
  CHECK(tower.encoder.ln_post_b.size() == static_cast<size_t>(kVHidden));
}

TEST_CASE("MuseGlimmer: the placeholder mask covers image AND video tokens") {
  const MuseGlimmerParams p = ParseMuseGlimmerParams(TinyConfig());
  CHECK(p.image_token_id == kImageToken);
  CHECK(p.video_token_id == kVideoToken);
  const std::vector<bool> mask = vllm::MuseGlimmerMultimodalMask(MmPrompt(), p);
  REQUIRE(mask.size() == MmPrompt().size());
  CHECK_FALSE(mask[0]);
  CHECK(mask[1]);  // image
  CHECK_FALSE(mask[2]);
  CHECK(mask[3]);  // video — masking only the image token leaves this row holding
                   // the text embedding of a token with no text meaning
  CHECK_FALSE(mask[4]);
}

TEST_CASE("MuseGlimmer: the mm seam is BIT-IDENTICAL to the text path with no image") {
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  const HfConfig config = TinyConfig();
  const MuseGlimmerWeights w =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config);

  const std::vector<int32_t> ids = {5, 9, 2, 7};
  const int64_t T = static_cast<int64_t>(ids.size());
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  vt::Queue q = Qcpu();

  CachePool pool_text(w.params, 2, 8);
  const std::vector<float> text =
      MuseGlimmerModel::Forward(ids, positions, PrefillMeta(T, 8), pool_text.attn_kv, w, q);

  // Route the SAME prompt through embed_input_ids + ForwardMm. This is the
  // inertness proof: the mm branch changes WHERE the hidden stream comes from and
  // nothing else, so with no placeholder rows the two must agree bit-for-bit.
  const std::vector<uint16_t> embeds =
      vllm::MuseGlimmerMergeMultimodalEmbeds(ids, {}, w, q);
  CHECK(embeds.size() == static_cast<size_t>(T * kHidden));
  CachePool pool_mm(w.params, 2, 8);
  const std::vector<float> mm = MuseGlimmerModel::ForwardMm(
      embeds, positions, PrefillMeta(T, 8), pool_mm.attn_kv, w, q);

  REQUIRE(mm.size() == text.size());
  size_t differing = 0;
  for (size_t i = 0; i < text.size(); ++i)
    if (std::memcmp(&text[i], &mm[i], sizeof(float)) != 0) ++differing;
  CHECK(differing == 0);
}

TEST_CASE("MuseGlimmer: vision soft tokens land on the placeholder rows and nowhere else") {
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  const HfConfig config = TinyConfig();
  const MuseGlimmerWeights w =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config);
  vt::Queue q = Qcpu();

  const std::vector<float> soft =
      vllm::MuseGlimmerEncodePixelGroups(TinyImages(), w, q);
  // Two pixel groups, each pixel-shuffling down to exactly one soft token, each
  // projected into the TEXT hidden width.
  REQUIRE(soft.size() == static_cast<size_t>(2 * kHidden));
  // A stubbed / all-zero projector would satisfy every shape check above.
  double mag = 0.0;
  for (float x : soft) mag += std::abs(static_cast<double>(x));
  CHECK(mag > 0.0);
  // The two groups carry different pixels, so their soft tokens must differ —
  // otherwise the tower is ignoring its input.
  bool groups_differ = false;
  for (int64_t i = 0; i < kHidden; ++i)
    if (soft[static_cast<size_t>(i)] != soft[static_cast<size_t>(kHidden + i)])
      groups_differ = true;
  CHECK(groups_differ);

  const std::vector<int32_t>& ids = MmPrompt();
  const int64_t T = static_cast<int64_t>(ids.size());
  const std::vector<uint16_t> plain = vllm::MuseGlimmerMergeMultimodalEmbeds(ids, {}, w, q);
  const std::vector<uint16_t> merged =
      vllm::MuseGlimmerMergeMultimodalEmbeds(ids, soft, w, q);
  REQUIRE(merged.size() == static_cast<size_t>(T * kHidden));

  const std::vector<bool> mask = vllm::MuseGlimmerMultimodalMask(ids, w.params);
  int64_t slot = 0;
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t i = 0; i < kHidden; ++i) {
      const size_t at = static_cast<size_t>(t * kHidden + i);
      if (mask[static_cast<size_t>(t)]) {
        // The placeholder row IS the soft token, cast to the model dtype.
        CHECK(merged[at] == vt::F32ToBF16(soft[static_cast<size_t>(slot * kHidden + i)]));
      } else {
        // Every text row survives the scatter untouched.
        CHECK(merged[at] == plain[at]);
      }
    }
    if (mask[static_cast<size_t>(t)]) ++slot;
  }
  CHECK(slot == 2);

  // Feature count and placeholder count must agree, BY NAME (muse_glimmer.py:1564).
  const std::vector<float> one_row(soft.begin(), soft.begin() + kHidden);
  CHECK_THROWS(vllm::MuseGlimmerMergeMultimodalEmbeds(ids, one_row, w, q));
}

// ─────────────────────────────────────────────────────────────────────────────
// `perception_emb_norm` — the norm asymmetry between text rows and soft tokens.
//
// COVERAGE HOLE this closes (review of #279). Nothing tested this at all: no
// config in the tree set `normalize_tok_embeddings`, so the branch at
// muse_glimmer_mm.cpp:217 never ran, and the scatter case above compares the
// merged rows against the SAME `soft` vector it just computed — so it is
// structurally blind to what EncodePixelGroups did to that vector. Inverting the
// condition, or deleting the call entirely, left every gate green.
//
// The probe runs the IDENTICAL tower twice, once with the flag and once without,
// which makes the norm the only difference between the two outputs; the soft
// tokens are then required to stand in the exact algebraic relation upstream's
// weightless RMSNorm defines, not merely to differ.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MuseGlimmer: perception_emb_norm runs IFF normalize_tok_embeddings") {
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  const MuseGlimmerWeights w_off =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(
          shards, TinyConfigNoNormalizedTokEmbeddings());
  const MuseGlimmerWeights w_on =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(
          shards, TinyConfigNormalizedTokEmbeddings());
  // The released 30B's case is ON: the key is ABSENT and absent means TRUE
  // (#405). This test previously asserted the opposite and so encoded the bug
  // as an expectation -- worse than not covering it, because it made the wrong
  // default look deliberate. The OFF arm now sets the flag explicitly, and the
  // silent-config case is pinned in test_muse_glimmer_scaffold.
  REQUIRE_FALSE(w_off.params.text.normalize_tok_embeddings);
  REQUIRE(w_on.params.text.normalize_tok_embeddings);
  REQUIRE(ParseMuseGlimmerParams(TinyConfig()).text.normalize_tok_embeddings);
  vt::Queue q = Qcpu();

  const std::vector<float> off = vllm::MuseGlimmerEncodePixelGroups(TinyImages(), w_off, q);
  const std::vector<float> on = vllm::MuseGlimmerEncodePixelGroups(TinyImages(), w_on, q);
  REQUIRE(off.size() == static_cast<size_t>(2 * kHidden));
  REQUIRE(on.size() == off.size());

  const double eps = static_cast<double>(w_off.params.text.rms_norm_eps);
  for (int64_t r = 0; r < 2; ++r) {
    double ms = 0.0;
    for (int64_t i = 0; i < kHidden; ++i) {
      const double v = off[static_cast<size_t>(r * kHidden + i)];
      ms += v * v;
    }
    ms /= static_cast<double>(kHidden);
    const double inv = 1.0 / std::sqrt(ms + eps);
    // The soft tokens are NOT unit-RMS coming out of the projector, so the norm is
    // a real change here — without this the equality below would be satisfied by a
    // forward that never called the norm at all.
    MESSAGE("perception_emb_norm row " << r << ": 1/rms = " << inv);
    CHECK(std::abs(inv - 1.0) > 0.05);
    for (int64_t i = 0; i < kHidden; ++i)
      CHECK(on[static_cast<size_t>(r * kHidden + i)] ==
            doctest::Approx(off[static_cast<size_t>(r * kHidden + i)] * inv).epsilon(1e-5));
  }

  // And the norm reaches the merged rows: a scatter that dropped it would put the
  // UNNORMALIZED token on the placeholder row.
  const std::vector<int32_t>& ids = MmPrompt();
  const std::vector<uint16_t> merged =
      vllm::MuseGlimmerMergeMultimodalEmbeds(ids, on, w_on, q);
  const std::vector<bool> mask = vllm::MuseGlimmerMultimodalMask(ids, w_on.params);
  int64_t slot = 0, differing = 0;
  for (size_t t = 0; t < ids.size(); ++t) {
    if (!mask[t]) continue;
    for (int64_t i = 0; i < kHidden; ++i) {
      const size_t at = static_cast<size_t>(static_cast<int64_t>(t) * kHidden + i);
      const size_t s = static_cast<size_t>(slot * kHidden + i);
      CHECK(merged[at] == vt::F32ToBF16(on[s]));
      if (merged[at] != vt::F32ToBF16(off[s])) ++differing;
    }
    ++slot;
  }
  REQUIRE(slot == 2);
  CHECK(differing > 0);  // the normed and un-normed streams are distinguishable here
}

TEST_CASE("MuseGlimmer: an image prompt runs through the REGISTERED mm forward") {
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  const HfConfig config = TinyConfig();
  std::unique_ptr<vllm::LoadedModel> model =
      ModelRegistry::Load(config, vllm::ModelSource::FromSafetensors(shards));
  REQUIRE(model != nullptr);
  CHECK(model->registration().info.supports_multimodal);
  const MuseGlimmerWeights w =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config);
  vt::Queue q = Qcpu();

  // BEFORE W4 this refused: there was no mm branch and no loaded tower, so an
  // image prompt could not reach the model at all. It now produces tokens.
  const std::vector<int32_t> out = vllm::MuseGlimmerGenerateGreedyViaRegistry(
      *model, MmPrompt(), TinyImages(), /*eos_token_id=*/-1, w, config, q,
      /*max_new_tokens=*/3);
  CHECK(out.size() == 3);
  for (int32_t id : out) {
    CHECK(id >= 0);
    CHECK(id < kVocab);
  }

  // HONESTY: nothing above says those tokens are CORRECT. The pinned oracle cannot
  // load muse_glimmer, so there is no reference decode and no speed denominator
  // (specs/muse-glimmer.md §0). What is established is reachability: the tower
  // runs, its output is projected and scattered, and the registered forward
  // consumes it.
  //
  // A text-only Muse Glimmer checkpoint has no tower, and asking it for an image
  // must say so BY NAME rather than reading empty vectors.
  MuseGlimmerWeights text_only = w;
  text_only.vision = vllm::MuseGlimmerVisionTower{};
  CHECK_THROWS(vllm::MuseGlimmerEncodePixelGroups(TinyImages(), text_only, q));
}

TEST_CASE("MuseGlimmer: ForwardMm consumes the given embeds WITHOUT re-normalizing") {
  // The mm branch must take `inputs_embeds` straight through
  // (muse_glimmer.py:1312-1313). Re-applying `embed_norm` there is invisible to the
  // text-identity gate above — RMSNorm is very nearly idempotent on an already
  // normalized row, so the text arm stays bit-identical — but it would flatten the
  // vision soft tokens, whose magnitude is NOT unit-RMS and carries signal.
  //
  // The probe: scale the whole embedding block by an exact power of two. Both the
  // embed_norm and the first `input_layernorm` are scale-invariant, so a forward
  // that normalized its input would return IDENTICAL logits for both arms. The
  // real forward carries the scale into the residual stream, so they must DIFFER.
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  const HfConfig config = TinyConfig();
  const MuseGlimmerWeights w =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config);
  vt::Queue q = Qcpu();

  const std::vector<int32_t> ids = {5, 9, 2, 7};
  const int64_t T = static_cast<int64_t>(ids.size());
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  const std::vector<uint16_t> base = vllm::MuseGlimmerMergeMultimodalEmbeds(ids, {}, w, q);
  std::vector<uint16_t> scaled(base.size());
  for (size_t i = 0; i < base.size(); ++i)
    scaled[i] = vt::F32ToBF16(4.0f * vt::BF16ToF32(base[i]));  // exact in bf16

  CachePool pa(w.params, 2, 8), pb(w.params, 2, 8);
  const std::vector<float> la =
      MuseGlimmerModel::ForwardMm(base, positions, PrefillMeta(T, 8), pa.attn_kv, w, q);
  const std::vector<float> lb =
      MuseGlimmerModel::ForwardMm(scaled, positions, PrefillMeta(T, 8), pb.attn_kv, w, q);
  REQUIRE(la.size() == lb.size());
  size_t differing = 0;
  for (size_t i = 0; i < la.size(); ++i)
    if (std::memcmp(&la[i], &lb[i], sizeof(float)) != 0) ++differing;
  CHECK(differing > 0);
}
