// ENG-MM-INPUT-PIPELINE wave L3, issue #607 — the TOWER SKIP.
//
// Ported from: vllm/model_executor/models/interfaces.py:288-293 (the
// `no_init_weights(..., StageMissingLayer(stage_name, mod))` context, entered
// under `all(mm_config.get_limit_per_prompt(m) == 0 for m in modalities)`)
// @ 5559679229bc, plus utils.py:687-704,740-778 for the placeholder.
//
// WHAT L2 LEFT OWED, in one sentence, because it is the whole reason this file
// exists: `--language-model-only` was accepted, it correctly zeroed every
// modality limit, the entrypoint correctly refused multimodal requests — and
// nothing gated tower CONSTRUCTION on those limits, so the flag freed no memory
// at all. `.agents/specs/multimodal-track.md` §1.5 L2 says so explicitly and
// forbids describing the flag as freeing VRAM until this lands and is measured.
//
// THE FIVE QUESTIONS, in the order they have to be answered:
//   1. does the DECISION mirror `:293` — all, not any; the flag reaching it only
//      through `get_limit_per_prompt`; no config meaning "load everything"?
//   2. does the LOADER honour it — tower tensors read by default, not read at
//      zero limits, text tower fully loaded either way?
//   3. is the TEXT PATH token-exact with and without the flag? (The gate the
//      spec names. A skip that perturbs the language model is not a skip.)
//   4. does a skipped tower REFUSE BY NAME, the way `StageMissingLayer.__call__`
//      raises rather than returning zeros?
//   5. REACHABILITY: does the flag travel from a production entry point all the
//      way to the loader? The mutation this case exists for is deleting
//      `source.multimodal = &params.multimodal;` from
//      `src/vllm/entrypoints/model_loader.cpp`. Cases 1-4 all stay GREEN under
//      that deletion, because they set `ModelSource::multimodal` themselves;
//      only case 5 goes red, which is exactly the difference between measuring a
//      class and measuring a capability (.agents/reachability.md).
//
// The vehicle is Muse Glimmer because it is one of only two architectures whose
// PRODUCTION loader reads a tower at all (the other is Qwen3-VL, which needs a
// checkpoint this box does not have), and the only one of those two with a
// checkpoint on the NAS for the RSS half of the gate.
#include "vllm/model_executor/models/interfaces.h"

#include "doctest/doctest.h"

#include "muse_glimmer_tiny_fixture.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/config/multimodal.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/muse_glimmer.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_vl.h"  // #607 L3 the SECOND tower loader
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"  // ModelForwardInput::gdn_meta
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using muse_glimmer_tiny::BuildSt;
using muse_glimmer_tiny::TempFile;
using muse_glimmer_tiny::TempModelDir;
using muse_glimmer_tiny::TinyConfig;
using muse_glimmer_tiny::TinyTensors;
using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::MultiModalConfig;
using vllm::MuseGlimmerModel;
using vllm::MuseGlimmerWeights;
using vllm::PagedKvCache;
using vllm::SkipTowerForModalities;
using vllm::v1::CommonAttentionMetadata;

namespace {

vt::Queue Qcpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// `--language-model-only`, as the serve flag and the C ABI set it
// (multimodal.py:78-80).
MultiModalConfig LanguageModelOnly() {
  MultiModalConfig c;
  c.language_model_only = true;
  return c;
}

// The OTHER route to zero, which upstream treats identically because
// `_mark_tower_model` never mentions the flag: explicit per-modality zeros.
MultiModalConfig ZeroImageAndVideo() {
  MultiModalConfig c;
  c.limit_per_prompt = {{"image", 0}, {"video", 0}};
  return c;
}

std::vector<vllm::SafetensorsFile> OpenShards(const TempFile& file) {
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  return shards;
}

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(const vllm::MuseGlimmerParams& p, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = p.text.num_key_value_heads, Dh = p.text.head_dim;
    for (int64_t l = 0; l < p.text.num_hidden_layers; ++l)
      buf.emplace_back(
          static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = vt::DType::kF32;
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

// ── The SECOND architecture that owns a tower: Qwen3-VL ─────────────────────
//
// The tiny Qwen3-VL checkpoint carries the TEXT BACKBONE ONLY, and that absence
// is the instrument. `Qwen3VLWeights::vision_cfg` is not read from the config —
// it is the hard-coded 4B geometry (hidden 1024, depth 24, 2304 position
// embeddings), which is ~300M parameters, so a checkpoint the vision reader
// could actually consume is not something a unit test can synthesise. What a
// unit test CAN do is take away the tensors and watch who asks for them:
//
//   * default limits -> `LoadQwen3VLVisionWeights` runs, reaches for
//     `model.visual.patch_embed.proj.weight`, and throws by that name;
//   * zero limits    -> nothing reaches for it, and the load returns.
//
// A skip that stopped working therefore turns this red by THROWING, which is
// what the previous cut of this row had no case for: destroying the Qwen3-VL
// half outright (`(void)mm_config; vision_skipped = false;`) left all twelve
// declared suites green, because no test anywhere passed a non-null
// `mm_config` to `LoadQwen3VLWeights` at all.
namespace qwen3vl_tiny {

constexpr int64_t kVocab = 32, kHidden = 16, kInter = 24, kLayers = 2;
constexpr int64_t kHeads = 2, kKvHeads = 1, kHeadDim = 8;

using muse_glimmer_tiny::Bf16;
using muse_glimmer_tiny::Fx;

// The text backbone `LoadTextBackbone` reads (qwen3_vl.cpp:110-135), in the
// real on-disk `model.language_model.*` spelling, and NOTHING under
// `model.visual.*`.
inline std::vector<Fx> TextOnlyTensors() {
  const std::string P = "model.language_model.";
  uint32_t seed = 1;
  std::vector<Fx> ts;
  ts.push_back(Bf16(P + "embed_tokens.weight", {kVocab, kHidden}, seed++));
  ts.push_back(Bf16(P + "norm.weight", {kHidden}, seed++));
  for (int64_t l = 0; l < kLayers; ++l) {
    const std::string b = P + "layers." + std::to_string(l) + ".";
    ts.push_back(Bf16(b + "input_layernorm.weight", {kHidden}, seed++));
    ts.push_back(Bf16(b + "post_attention_layernorm.weight", {kHidden}, seed++));
    ts.push_back(Bf16(b + "self_attn.q_proj.weight", {kHeads * kHeadDim, kHidden}, seed++));
    ts.push_back(Bf16(b + "self_attn.k_proj.weight", {kKvHeads * kHeadDim, kHidden}, seed++));
    ts.push_back(Bf16(b + "self_attn.v_proj.weight", {kKvHeads * kHeadDim, kHidden}, seed++));
    ts.push_back(Bf16(b + "self_attn.o_proj.weight", {kHidden, kHeads * kHeadDim}, seed++));
    ts.push_back(Bf16(b + "self_attn.q_norm.weight", {kHeadDim}, seed++));
    ts.push_back(Bf16(b + "self_attn.k_norm.weight", {kHeadDim}, seed++));
    ts.push_back(Bf16(b + "mlp.gate_proj.weight", {kInter, kHidden}, seed++));
    ts.push_back(Bf16(b + "mlp.up_proj.weight", {kInter, kHidden}, seed++));
    ts.push_back(Bf16(b + "mlp.down_proj.weight", {kHidden, kInter}, seed++));
  }
  return ts;
}

inline HfConfig Config() {
  HfConfig c;
  c.architectures = {"Qwen3VLForConditionalGeneration"};
  c.hidden_size = kHidden;
  c.intermediate_size = kInter;
  c.num_hidden_layers = kLayers;
  c.num_attention_heads = kHeads;
  c.num_key_value_heads = kKvHeads;
  c.head_dim = kHeadDim;
  c.vocab_size = kVocab;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 5e6;
  return c;
}

// The first tensor `LoadQwen3VLVisionWeights` reaches for, as its own resolver
// refuses it (qwen3_vl.cpp, `VT_CHECK(it != where.end(), ...)`). Matched as a
// SUBSTRING because VT_CHECK wraps it in a `vt: ` prefix and a ` at file:line`
// suffix, and the line number is not this test's business.
inline const std::string& FirstVisionTensor() {
  static const std::string s =
      "qwen3-vl vision: tensor not found: model.visual.patch_embed.proj.weight";
  return s;
}

}  // namespace qwen3vl_tiny

// A pure TEXT prompt: no image and no video placeholder, so it is servable on
// both arms and its logits are the thing the token-exactness gate compares.
const std::vector<int32_t>& TextPrompt() {
  static const std::vector<int32_t> p = {5, 9, 2, 7};
  return p;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// 1. THE DECISION — interfaces.py:293, transcribed.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("tower skip: the decision mirrors interfaces.py:293") {
  SUBCASE("no multimodal config at all loads everything") {
    // Every caller that is not the engine, and every path that predates L3.
    CHECK_FALSE(SkipTowerForModalities(nullptr, {"image", "video"}));
  }

  SUBCASE("default limits (999 per modality) load everything") {
    const MultiModalConfig def;
    CHECK(def.GetLimitPerPrompt("image") == vllm::kDefaultLimitPerPrompt);
    CHECK_FALSE(SkipTowerForModalities(&def, {"image", "video"}));
  }

  SUBCASE("--language-model-only skips, reaching the decision through the limits") {
    const MultiModalConfig lmo = LanguageModelOnly();
    // The flag is NOT read by the predicate. It is read by GetLimitPerPrompt,
    // which is the only thing the predicate consults, and that is the mirror:
    // `_mark_tower_model` never mentions the flag (interfaces.py:293).
    CHECK(lmo.GetLimitPerPrompt("image") == 0);
    CHECK(lmo.GetLimitPerPrompt("video") == 0);
    CHECK(SkipTowerForModalities(&lmo, {"image", "video"}));
  }

  SUBCASE("explicit per-modality zeros skip too — the flag is not the mechanism") {
    // `--limit-mm-per-prompt '{"image":0,"video":0}'`. Keying the skip on
    // `language_model_only` would leave this route loading a tower nothing can
    // ever reach, and no test of the flag would notice.
    const MultiModalConfig zeros = ZeroImageAndVideo();
    CHECK(SkipTowerForModalities(&zeros, {"image", "video"}));
  }

  SUBCASE("ALL, not ANY: one non-zero modality keeps the tower") {
    // The one-character defect. `_mark_tower_model` is called with the tower's
    // whole modality set (qwen3_5.py:422,634 / qwen3_vl.py:1747), and a tower
    // that still serves video must still be loaded.
    MultiModalConfig image_only;
    image_only.limit_per_prompt = {{"image", 0}};
    CHECK(image_only.GetLimitPerPrompt("image") == 0);
    CHECK(image_only.GetLimitPerPrompt("video") == vllm::kDefaultLimitPerPrompt);
    CHECK_FALSE(SkipTowerForModalities(&image_only, {"image", "video"}));
    // ... and symmetrically for the other one.
    MultiModalConfig video_only;
    video_only.limit_per_prompt = {{"video", 0}};
    CHECK_FALSE(SkipTowerForModalities(&video_only, {"image", "video"}));
  }

  SUBCASE("a single-modality tower is decided on that modality alone") {
    // gemma3n_mm.py:509,515 marks image and audio as SEPARATE towers, so each is
    // skipped independently. `all()` over a one-element set is that element.
    MultiModalConfig image_only;
    image_only.limit_per_prompt = {{"image", 0}};
    CHECK(SkipTowerForModalities(&image_only, {"image"}));
    CHECK_FALSE(SkipTowerForModalities(&image_only, {"audio"}));
  }

  SUBCASE("an EMPTY modality set never skips") {
    // The documented deviation. Python's `all(())` is vacuously TRUE, so a
    // literal transcription would skip a tower marked with no modality at all,
    // for every configuration including the default. Upstream cannot reach that
    // state; we refuse to.
    const MultiModalConfig lmo = LanguageModelOnly();
    CHECK_FALSE(SkipTowerForModalities(&lmo, {}));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. THE LOADER — through ModelRegistry::Load, on a checkpoint that really does
//    carry a perception encoder.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("tower skip: the loader reads the tower by default and not at zero limits") {
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards = OpenShards(file);
  const HfConfig config = TinyConfig();

  SUBCASE("default: the perception encoder is read, as it always was") {
    const MuseGlimmerWeights w =
        vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config);
    CHECK(w.text_loaded);
    CHECK(w.vision.loaded);
    CHECK_FALSE(w.vision_skipped);
    CHECK(w.vision.encoder.blocks.size() == static_cast<size_t>(muse_glimmer_tiny::kVLayers));
  }

  SUBCASE("--language-model-only: the encoder is NOT read, the text tower is") {
    const MultiModalConfig lmo = LanguageModelOnly();
    const MuseGlimmerWeights w =
        vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config, &lmo);
    CHECK(w.vision_skipped);
    CHECK_FALSE(w.vision.loaded);
    CHECK(w.vision.encoder.blocks.empty());
    CHECK(w.vision.projection.empty());
    // The LANGUAGE model is untouched. A skip that also dropped a text tensor
    // would still satisfy every assertion above.
    CHECK(w.text_loaded);
    CHECK(w.layers.size() == static_cast<size_t>(muse_glimmer_tiny::kTextLayers));
  }

  SUBCASE("the tower's GEOMETRY survives the skip — construct, then do not initialise") {
    // upstream constructs on `torch.device("meta")` (utils.py:762): every shape
    // still resolves. Ours parses `vision_config` either way, which is what lets
    // the refusal below say the checkpoint HAS an encoder.
    const MultiModalConfig lmo = LanguageModelOnly();
    const MuseGlimmerWeights w =
        vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config, &lmo);
    CHECK(w.params.vision.present);
    CHECK(w.params.vision.num_hidden_layers == muse_glimmer_tiny::kVLayers);
    CHECK(w.params.vision.hidden_size == muse_glimmer_tiny::kVHidden);
  }

  SUBCASE("the loader COMPLAINS ABOUT NOTHING either way") {
    // `StageMissingLayer` is kept out of the child registry so the weight loader
    // reports no missing keys for a skipped stage (utils.py:693-695). Our
    // analogue is the structural accounting: it is a property of the CHECKPOINT,
    // not of what this load chose to materialise, so it must not move.
    const MultiModalConfig lmo = LanguageModelOnly();
    const MuseGlimmerWeights loaded =
        vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config);
    const MuseGlimmerWeights skipped =
        vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config, &lmo);
    CHECK(skipped.accounted_tensors == loaded.accounted_tensors);
    CHECK(skipped.enumerated_tensors == loaded.enumerated_tensors);
    CHECK(skipped.accounted_tensors == skipped.enumerated_tensors);
  }

  SUBCASE("through ModelRegistry::Load, the skip is reported by stage name") {
    vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    std::unique_ptr<vllm::LoadedModel> plain = ModelRegistry::Load(config, source);
    REQUIRE(plain != nullptr);
    CHECK(plain->skipped_towers().empty());

    const MultiModalConfig lmo = LanguageModelOnly();
    source.multimodal = &lmo;
    std::unique_ptr<vllm::LoadedModel> skipped = ModelRegistry::Load(config, source);
    REQUIRE(skipped != nullptr);
    REQUIRE(skipped->skipped_towers().size() == 1);
    // interfaces.py:279-282 — the stage_name for exactly the {image, video} pair.
    CHECK(skipped->skipped_towers()[0] == "vision_tower");
  }

  SUBCASE("a TEXT-ONLY checkpoint reports no skip, because nothing was skipped") {
    // The distinction the observable has to carry: "there is no encoder here" is
    // not "the encoder was freed". Reporting the second for the first would make
    // every text model look like a memory saving.
    HfConfig text_only = config;
    text_only.raw.erase("vision_config");
    const MultiModalConfig lmo = LanguageModelOnly();
    const MuseGlimmerWeights w =
        vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, text_only, &lmo);
    CHECK_FALSE(w.params.vision.present);
    CHECK_FALSE(w.vision.loaded);
    CHECK_FALSE(w.vision_skipped);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2b. THE OTHER LOADER — Qwen3-VL, whose skip had no test at all until this
//     case. See `namespace qwen3vl_tiny` above for why the absence of the
//     vision tensors is the instrument.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("tower skip: the Qwen3-VL loader reads model.visual.* by default and not at zero limits") {
  const TempFile file(muse_glimmer_tiny::BuildSt(qwen3vl_tiny::TextOnlyTensors()));
  std::vector<vllm::SafetensorsFile> shards = OpenShards(file);
  const HfConfig config = qwen3vl_tiny::Config();

  SUBCASE("default: the vision reader RUNS, and says which tensor it wanted") {
    std::string message;
    try {
      (void)vllm::LoadQwen3VLWeights(shards, config);
      FAIL("the default load must reach for model.visual.*");
    } catch (const std::exception& e) {
      message = e.what();
    }
    CAPTURE(message);
    // By NAME, not merely "it threw": a load that died in the text backbone
    // would also throw, and would prove the opposite of what is claimed here.
    CHECK(message.find(qwen3vl_tiny::FirstVisionTensor()) != std::string::npos);
  }

  SUBCASE("--language-model-only: nothing reaches for them, and the text tower loads") {
    const MultiModalConfig lmo = LanguageModelOnly();
    const vllm::Qwen3VLWeights w =
        vllm::LoadQwen3VLWeights(shards, config, &lmo);
    CHECK(w.vision_skipped);
    CHECK_FALSE(w.vision_loaded);
    CHECK(w.vision.blocks.empty());
    CHECK(w.vision.patch_proj_w.empty());
    // The LANGUAGE model is untouched — a skip that also dropped a text tensor
    // would satisfy every assertion above.
    CHECK(w.text.layers.size() == static_cast<size_t>(qwen3vl_tiny::kLayers));
    CHECK(w.text.embed_tokens.bytes.size() ==
          static_cast<size_t>(qwen3vl_tiny::kVocab * qwen3vl_tiny::kHidden) * 2U);
  }

  SUBCASE("explicit per-modality zeros take the same road") {
    const MultiModalConfig zeros = ZeroImageAndVideo();
    const vllm::Qwen3VLWeights w =
        vllm::LoadQwen3VLWeights(shards, config, &zeros);
    CHECK(w.vision_skipped);
  }

  SUBCASE("ALL, not ANY: image:0 alone still reads the tower") {
    // qwen3_vl.py:1747 marks this tower {"image", "video"}, so a load that can
    // still serve video must still load it — and here that means throwing.
    MultiModalConfig image_only;
    image_only.limit_per_prompt = {{"image", 0}};
    std::string message;
    try {
      (void)vllm::LoadQwen3VLWeights(shards, config, &image_only);
      FAIL("image:0 alone must NOT skip the {image, video} tower");
    } catch (const std::exception& e) {
      message = e.what();
    }
    CHECK(message.find(qwen3vl_tiny::FirstVisionTensor()) != std::string::npos);
  }

  SUBCASE("the tower's GEOMETRY survives the skip") {
    // construct-without-initialise (utils.py:762): every shape still resolves.
    const MultiModalConfig lmo = LanguageModelOnly();
    const vllm::Qwen3VLWeights w =
        vllm::LoadQwen3VLWeights(shards, config, &lmo);
    CHECK(w.vision_cfg.depth == 24);
    CHECK(w.vision_cfg.hidden_size == 1024);
  }

  SUBCASE("through ModelRegistry::Load, the skip is reported by stage name") {
    // The seam a production entry point actually uses, and the only thing that
    // exercises Qwen3VLLoadedModel::skipped_towers (qwen3_vl_registry.cpp:70).
    vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    const MultiModalConfig lmo = LanguageModelOnly();
    source.multimodal = &lmo;
    std::unique_ptr<vllm::LoadedModel> skipped = ModelRegistry::Load(config, source);
    REQUIRE(skipped != nullptr);
    REQUIRE(skipped->skipped_towers().size() == 1);
    CHECK(skipped->skipped_towers()[0] == "vision_tower");

    // ... and with the limits absent the same source throws, which is this
    // checkpoint's way of saying the tensors were wanted.
    source.multimodal = nullptr;
    CHECK_THROWS_AS((void)ModelRegistry::Load(config, source), std::exception);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. TOKEN-EXACTNESS OF THE TEXT PATH, with and without the flag. The gate
//    `.agents/specs/multimodal-track.md` §1.5 L3 names.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("tower skip: the text path is BIT-IDENTICAL with and without the flag") {
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards = OpenShards(file);
  const HfConfig config = TinyConfig();
  const MultiModalConfig lmo = LanguageModelOnly();

  const MuseGlimmerWeights full =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config);
  const MuseGlimmerWeights lean =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config, &lmo);
  REQUIRE(full.vision.loaded);
  REQUIRE(lean.vision_skipped);

  const std::vector<int32_t>& ids = TextPrompt();
  const int64_t T = static_cast<int64_t>(ids.size());
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  vt::Queue q = Qcpu();

  CachePool pool_full(full.params, 2, 8);
  const std::vector<float> a = MuseGlimmerModel::Forward(
      ids, positions, PrefillMeta(T, 8), pool_full.attn_kv, full, q);
  CachePool pool_lean(lean.params, 2, 8);
  const std::vector<float> b = MuseGlimmerModel::Forward(
      ids, positions, PrefillMeta(T, 8), pool_lean.attn_kv, lean, q);

  REQUIRE(a.size() == b.size());
  REQUIRE(!a.empty());
  // BIT-identical, not approximately equal. `doctest::Approx` would pass on a
  // path that had silently changed reduction order, and this gate exists to say
  // the language model did not move at all.
  size_t differing = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++differing;
  CHECK(differing == 0);

  // The same statement at the level a user sees: the greedy id per row.
  const int64_t V = full.params.text.vocab_size;
  REQUIRE(a.size() % static_cast<size_t>(V) == 0);
  for (size_t row = 0; row < a.size() / static_cast<size_t>(V); ++row) {
    int64_t arg_a = 0, arg_b = 0;
    for (int64_t v = 1; v < V; ++v) {
      const size_t base = row * static_cast<size_t>(V);
      if (a[base + static_cast<size_t>(v)] > a[base + static_cast<size_t>(arg_a)]) arg_a = v;
      if (b[base + static_cast<size_t>(v)] > b[base + static_cast<size_t>(arg_b)]) arg_b = v;
    }
    CHECK(arg_a == arg_b);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. THE PLACEHOLDER — `StageMissingLayer.__call__` raises (utils.py:700-701).
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("tower skip: a skipped tower REFUSES BY NAME rather than reading empty buffers") {
  const TempFile file(BuildSt(TinyTensors()));
  std::vector<vllm::SafetensorsFile> shards = OpenShards(file);
  const HfConfig config = TinyConfig();
  const MultiModalConfig lmo = LanguageModelOnly();
  const MuseGlimmerWeights lean =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, config, &lmo);
  vt::Queue q = Qcpu();

  std::string message;
  try {
    (void)vllm::MuseGlimmerEncodePixelGroups({}, lean, q);
    FAIL("a skipped perception encoder must refuse, not return");
  } catch (const std::exception& e) {
    message = e.what();
  }
  // The two absences have DIFFERENT fixes, so they must not share a message: one
  // is repaired by dropping the zero limit, the other cannot be repaired at all.
  CHECK(message.find("SKIPPED") != std::string::npos);
  CHECK(message.find("--language-model-only") != std::string::npos);
  CHECK(message.find("limit 0") != std::string::npos);

  // ... and the text-only checkpoint still gets its own message, unchanged.
  MuseGlimmerWeights text_only = lean;
  text_only.vision_skipped = false;
  std::string other;
  try {
    (void)vllm::MuseGlimmerEncodePixelGroups({}, text_only, q);
    FAIL("a checkpoint with no perception encoder must refuse, not return");
  } catch (const std::exception& e) {
    other = e.what();
  }
  CHECK(other.find("no perception encoder") != std::string::npos);
  CHECK(other != message);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. REACHABILITY — the production entry point, and the ONLY case that reds when
//    `source.multimodal = &params.multimodal;` is deleted from
//    src/vllm/entrypoints/model_loader.cpp.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("tower skip: --language-model-only reaches the loader from LoadedEngine::FromModelDir") {
  const TempModelDir dir;
  vllm::entrypoints::EngineParams params;
  params.device = vllm::Device::kCPU;
  params.max_model_len = 16;
  params.num_blocks = 4;
  params.block_size = 16;  // the KV allocator refuses anything else (multiple of 16)

  SUBCASE("default: nothing is skipped") {
    std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
        vllm::entrypoints::LoadedEngine::FromModelDir(dir.path(), params);
    REQUIRE(engine != nullptr);
    CHECK(engine->mm_config().GetLimitPerPrompt("image") ==
          vllm::kDefaultLimitPerPrompt);
    CHECK(engine->skipped_towers().empty());
  }

  SUBCASE("--language-model-only: the vision tower is skipped") {
    // THE CHAIN, end to end: the serve flag / the C-ABI field ->
    // EngineParams::multimodal -> ModelSource::multimodal -> the Muse Glimmer
    // loader -> the tensors that are never read. Every hop except the last two
    // was already gated by L2; these two are what L3 adds, and deleting the
    // `source.multimodal = ...` assignment in model_loader.cpp turns exactly
    // this subcase red while leaving every other case in this file green.
    vllm::entrypoints::EngineParams lmo = params;
    lmo.multimodal.language_model_only = true;
    std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
        vllm::entrypoints::LoadedEngine::FromModelDir(dir.path(), lmo);
    REQUIRE(engine != nullptr);
    CHECK(engine->mm_config().GetLimitPerPrompt("image") == 0);
    REQUIRE(engine->skipped_towers().size() == 1);
    CHECK(engine->skipped_towers()[0] == "vision_tower");
  }

  SUBCASE("--limit-mm-per-prompt zeros reach it by the same road") {
    // The flag is sugar. The road is the limits, so the other route must arrive
    // at the same place.
    vllm::entrypoints::EngineParams zeros = params;
    zeros.multimodal.limit_per_prompt = {{"image", 0}, {"video", 0}};
    std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
        vllm::entrypoints::LoadedEngine::FromModelDir(dir.path(), zeros);
    REQUIRE(engine != nullptr);
    REQUIRE(engine->skipped_towers().size() == 1);
    CHECK(engine->skipped_towers()[0] == "vision_tower");
  }

  SUBCASE("one zero modality is not enough, through the entry point too") {
    vllm::entrypoints::EngineParams image_only = params;
    image_only.multimodal.limit_per_prompt = {{"image", 0}};
    std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
        vllm::entrypoints::LoadedEngine::FromModelDir(dir.path(), image_only);
    REQUIRE(engine != nullptr);
    CHECK(engine->skipped_towers().empty());
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. ENG-MM-INPUT-PIPELINE P1 (#1358, #2300) — the DeepStack CHANNEL IS TWO
//    FIELDS, and they have to agree.
//
//    P1 turned `MultiModalForwardInput`'s host pointers into borrowed `vt::Tensor`
//    DEVICE views. The old contract required `mm.deepstack_bf16 != nullptr`, and the
//    driver always passed a pointer to a possibly-EMPTY vector, so "I set the level
//    count and forgot the tensor" was a LOUD throw. Under the device contract ABSENT
//    and NOT-APPLICABLE are the SAME value (`data == nullptr`), which is what makes
//    that pair both expressible and silent: with `deepstack_levels > 0` and
//    `deepstack.data == nullptr`, `VLForwardLastLogitsDBuf` (`qwen3_vl.cpp`)
//    computes `has_ds = false` and SKIPS every multiscale-merger add, so a Qwen3-VL
//    prefill returns fluent WRONG tokens with no diagnostic at all. The consumer of
//    this seam is the runner slice (#2300) — the next piece of work — which has to
//    set both fields, and nothing else couples them.
//
//    The REVERSE pairing (tensor set, `levels == 0`) deliberately gets NO second
//    guard, because it already has one: `has_ds` is then true, and the forward
//    core's `VT_CHECK(!has_ds || deepstack.Numel() == L * T * H)` compares a
//    non-zero `Numel()` against `0 * T * H` and refuses. A check with no failing
//    input is not a gate.
//
//    THE VEHICLE, and why it is here. The refusal lives in
//    `ForwardQwen3VLForConditionalGeneration`, so reaching it needs a real
//    `Qwen3VLLoadedModel` — which only `ModelRegistry::Load` produces, and only off
//    a Qwen3-VL checkpoint. This file already synthesises one and already loads it
//    through that seam (§2b), so the case lives beside its fixture rather than
//    behind a third copy of it. Every arm below enters through
//    `reg.factory->forward`, the same function pointer `ModelRegistry::Forward`
//    calls; none constructs the model by hand.
// ─────────────────────────────────────────────────────────────────────────────
namespace {
namespace qwen3vl_mrope_tiny {

// A SECOND tiny Qwen3-VL checkpoint, and the one difference that forces it to
// exist: `MropeArgs` (`qwen3_vl.cpp`) hard-codes `mrope_section = {24, 20, 20}`,
// and `vt::RopeFromCache` requires that section to sum to `rotary_dim / 2`
// (`src/vt/ops.cpp`, "mrope_section must sum to rotary_dim/2"). `rotary_dim` IS
// `config.head_dim`, so any fixture whose forward reaches a decoder layer has to
// carry head_dim == 128. `qwen3vl_tiny` above never forwards, so its head_dim 8 is
// right there and unusable here. Everything else is deliberately minimal.
constexpr int64_t kVocab = 32, kHeads = 2, kKvHeads = 1, kHeadDim = 128;
constexpr int64_t kHidden = kHeads * kHeadDim, kInter = 64, kLayers = 2;

using muse_glimmer_tiny::Bf16;
using muse_glimmer_tiny::Fx;

inline std::vector<Fx> TextOnlyTensors() {
  const std::string P = "model.language_model.";
  uint32_t seed = 1;
  std::vector<Fx> ts;
  ts.push_back(Bf16(P + "embed_tokens.weight", {kVocab, kHidden}, seed++));
  ts.push_back(Bf16(P + "norm.weight", {kHidden}, seed++));
  for (int64_t l = 0; l < kLayers; ++l) {
    const std::string b = P + "layers." + std::to_string(l) + ".";
    ts.push_back(Bf16(b + "input_layernorm.weight", {kHidden}, seed++));
    ts.push_back(Bf16(b + "post_attention_layernorm.weight", {kHidden}, seed++));
    ts.push_back(Bf16(b + "self_attn.q_proj.weight", {kHeads * kHeadDim, kHidden}, seed++));
    ts.push_back(Bf16(b + "self_attn.k_proj.weight", {kKvHeads * kHeadDim, kHidden}, seed++));
    ts.push_back(Bf16(b + "self_attn.v_proj.weight", {kKvHeads * kHeadDim, kHidden}, seed++));
    ts.push_back(Bf16(b + "self_attn.o_proj.weight", {kHidden, kHeads * kHeadDim}, seed++));
    ts.push_back(Bf16(b + "self_attn.q_norm.weight", {kHeadDim}, seed++));
    ts.push_back(Bf16(b + "self_attn.k_norm.weight", {kHeadDim}, seed++));
    ts.push_back(Bf16(b + "mlp.gate_proj.weight", {kInter, kHidden}, seed++));
    ts.push_back(Bf16(b + "mlp.up_proj.weight", {kInter, kHidden}, seed++));
    ts.push_back(Bf16(b + "mlp.down_proj.weight", {kHidden, kInter}, seed++));
  }
  return ts;
}

inline HfConfig Config() {
  HfConfig c;
  c.architectures = {"Qwen3VLForConditionalGeneration"};
  c.hidden_size = kHidden;
  c.intermediate_size = kInter;
  c.num_hidden_layers = kLayers;
  c.num_attention_heads = kHeads;
  c.num_key_value_heads = kKvHeads;
  c.head_dim = kHeadDim;
  c.vocab_size = kVocab;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 5e6;
  return c;
}

}  // namespace qwen3vl_mrope_tiny

// A rank-2 or rank-3 device allocation with a borrowed `vt::Tensor` over it — the
// shape the P1 seam takes. Owns the bytes for its lifetime; `t()` hands out the
// view, exactly as the runner slice will.
class DevBuf {
 public:
  DevBuf(vt::Queue& q, vt::DType dt, int64_t d0, int64_t d1, int64_t d2, const void* src)
      : b_(&vt::GetBackend(q.device.type)), q_(&q), dt_(dt), d0_(d0), d1_(d1), d2_(d2),
        rank_(d2 > 0 ? 3 : 2) {
    const int64_t n = rank_ == 3 ? d0_ * d1_ * d2_ : d0_ * d1_;
    bytes_ = static_cast<size_t>(n) * vt::SizeOf(dt_);
    p_ = b_->Alloc(bytes_);
    if (src != nullptr)
      b_->Copy(*q_, p_, src, bytes_);
    else
      b_->Memset(*q_, p_, 0, bytes_);
  }
  ~DevBuf() { b_->Free(p_); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  vt::Tensor t() const {
    return rank_ == 3
               ? vt::Tensor::Contiguous(p_, dt_, q_->device, {d0_, d1_, d2_})
               : vt::Tensor::Contiguous(p_, dt_, q_->device, {d0_, d1_});
  }

 private:
  vt::Backend* b_ = nullptr;
  vt::Queue* q_ = nullptr;
  vt::DType dt_ = vt::DType::kBF16;
  int64_t d0_ = 0, d1_ = 0, d2_ = 0;
  int rank_ = 2;
  void* p_ = nullptr;
  size_t bytes_ = 0;
};

// One paged KV pool per forward: a forward WRITES its keys and values, so reusing
// one across arms would make the second arm read the first arm's cache.
struct VlKvPool {
  std::vector<std::vector<uint16_t>> buf;
  std::vector<PagedKvCache> attn_kv;
  VlKvPool(int64_t num_blocks, int64_t block_size) {
    using namespace qwen3vl_mrope_tiny;
    for (int64_t l = 0; l < kLayers; ++l)
      buf.emplace_back(
          static_cast<size_t>(num_blocks * 2 * block_size * kKvHeads * kHeadDim), 0);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = vt::DType::kBF16;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = kKvHeads;
      kv.head_size = kHeadDim;
      attn_kv.push_back(kv);
    }
  }
};

}  // namespace

TEST_CASE("mm seam: Qwen3-VL refuses deepstack_levels that no deepstack tensor backs") {
  using namespace qwen3vl_mrope_tiny;
  const TempFile file(BuildSt(TextOnlyTensors()));
  std::vector<vllm::SafetensorsFile> shards = OpenShards(file);
  const HfConfig config = Config();
  // The tiny checkpoint carries no `model.visual.*`, so the load has to skip the
  // tower; that is orthogonal to what this case asks and is §2b's subject.
  const MultiModalConfig lmo = LanguageModelOnly();
  vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
  source.multimodal = &lmo;
  std::unique_ptr<vllm::LoadedModel> model = ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);

  const vllm::ModelRegistration& reg =
      vllm::RegistrationFor("Qwen3VLForConditionalGeneration");
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->forward != nullptr);

  vt::Queue q = Qcpu();
  const int64_t T = 4;
  std::vector<uint16_t> embeds(static_cast<size_t>(T * kHidden));
  for (size_t i = 0; i < embeds.size(); ++i)
    embeds[i] = vt::F32ToBF16(0.125f * static_cast<float>(1 + (i % 5)));
  std::vector<int32_t> pos3(static_cast<size_t>(3 * T));
  for (int64_t s = 0; s < 3; ++s)
    for (int64_t t = 0; t < T; ++t)
      pos3[static_cast<size_t>(s * T + t)] = static_cast<int32_t>(t);

  const DevBuf dembeds(q, vt::DType::kBF16, T, kHidden, 0, embeds.data());
  const DevBuf dpos3(q, vt::DType::kI32, 3, T, 0, pos3.data());

  const std::vector<int32_t> token_ids(static_cast<size_t>(T), 0);
  const std::vector<int32_t> positions(static_cast<size_t>(T), 0);
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  const CommonAttentionMetadata meta = PrefillMeta(T, 8);
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::GdnStateCache> gdn_state;

  // The two channels the seam always requires; only the DeepStack pair varies.
  vllm::MultiModalForwardInput mm;
  mm.inputs_embeds = dembeds.t();
  mm.positions3 = dpos3.t();

  SUBCASE("levels set, tensor absent: REFUSED by name, before anything runs") {
    VlKvPool pool(2, 8);
    vllm::MultiModalForwardInput bad = mm;
    bad.deepstack_levels = 3;  // and `bad.deepstack` stays default-constructed
    vllm::ModelForwardInput in{.token_ids = token_ids,
                               .positions = positions,
                               .attn_meta = meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = pool.attn_kv,
                               .gdn_state = gdn_state,
                               .config = config,
                               .queue = q,
                               .logits_indices = logits_indices,
                               .num_reqs = 1};
    in.gather_logits = false;
    in.mm = bad;

    std::string message;
    try {
      (void)reg.factory->forward(*model, in);
      FAIL("deepstack_levels > 0 with an absent deepstack tensor must be refused");
    } catch (const std::exception& e) {
      message = e.what();
    }
    CAPTURE(message);
    // BY NAME, and by the caller's OWN field names: "it threw" is satisfied by any
    // later failure in the forward, and under the mutation that deletes this guard
    // the forward throws or returns for entirely different reasons. Naming both
    // fields is what tells the two apart.
    CHECK(message.find("deepstack_levels") != std::string::npos);
    CHECK(message.find(".deepstack") != std::string::npos);
  }

  SUBCASE("neither set: the legitimate no-DeepStack step RUNS") {
    // The control, and the reason the guard is a coupling rather than a blanket
    // requirement. A decode step and a DeepStack-less VL checkpoint both arrive
    // here with both fields empty, and both must be served.
    VlKvPool pool(2, 8);
    vllm::ModelForwardInput in{.token_ids = token_ids,
                               .positions = positions,
                               .attn_meta = meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = pool.attn_kv,
                               .gdn_state = gdn_state,
                               .config = config,
                               .queue = q,
                               .logits_indices = logits_indices,
                               .num_reqs = 1};
    in.gather_logits = false;
    in.mm = mm;  // deepstack_levels == 0, deepstack default-constructed

    const vllm::ForwardLogits out = reg.factory->forward(*model, in);
    CHECK_FALSE(out.on_device());
    CHECK(out.host.size() == static_cast<size_t>(kVocab));
  }

  SUBCASE("both set: the coupled input is SERVED, so the guard is satisfiable") {
    // Without this arm the guard could be `deepstack_levels == 0` and still pass
    // the two arms above.
    VlKvPool pool(2, 8);
    const DevBuf ds(q, vt::DType::kBF16, 3, T, kHidden, nullptr);
    vllm::MultiModalForwardInput good = mm;
    good.deepstack = ds.t();
    good.deepstack_levels = 3;
    vllm::ModelForwardInput in{.token_ids = token_ids,
                               .positions = positions,
                               .attn_meta = meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = pool.attn_kv,
                               .gdn_state = gdn_state,
                               .config = config,
                               .queue = q,
                               .logits_indices = logits_indices,
                               .num_reqs = 1};
    in.gather_logits = false;
    in.mm = good;

    const vllm::ForwardLogits out = reg.factory->forward(*model, in);
    CHECK(out.host.size() == static_cast<size_t>(kVocab));
  }
}
