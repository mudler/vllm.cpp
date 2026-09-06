// MODEL-DFLASH2-NVFP4 (#2758) — the ModelOpt NVFP4 arm of the DFlash2 draft.
//
// WHAT THIS SUITE GATES. `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal` @
// `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1` could not be loaded at all. The
// draft loader asks its arm question once, on `fc`, and only about EXL3; that
// artifact EXCLUDES `fc` from quantization, so the probe answers false, the
// BF16 reader is selected, and the load dies 400 lines later inside layer 0
// with `qwen3_dflash: expected BF16 for layers.0.self_attn.q_proj.weight` — a
// sentence that names neither NVFP4 nor ModelOpt nor the missing arm.
//
// EVERY CASE ENTERS THROUGH THE PRODUCTION LOADER, in the order
// `LoadDflashDraft` calls it (`src/vllm/entrypoints/model_loader.cpp`):
// `vllm::MakeQwen3DFlashDraftConfig(config_json)` and then
// `vllm::LoadQwen3DFlash(shards, config, num_taps, mask_id)` over a REAL
// safetensors file. Nothing here builds a `Qwen3DFlashWeights` by hand. A unit
// test that constructed the owners itself would prove the struct has fields,
// never that the loader fills them, and the config carry is the whole reach:
// delete `cfg.raw["quantization_config"] = ...` in
// `MakeQwen3DFlashDraftConfig` and every arm case below reds with the OLD
// message.
//
// THE FIXTURE IS THE PUBLISHED SHAPE AT 1/40 SCALE. Same 7-quantized-per-layer
// split, same 12 exact-name exclusions, same ModelOpt spelling (`.weight` U8 +
// `.weight_scale` F8_E4M3 + `.weight_scale_2` F32 + `.input_scale` F32), same
// absent `lm_head` and `embed_tokens`. Every quantized module's K is a multiple
// of 16, which is what NVFP4 group-16 requires and what the reader checks.
//
// WHY THIS IS A LOAD GATE AND NOT A FORWARD TWIN. `vt::MatmulNvfp4` is
// registered for CUDA only (`src/vt/cuda/cuda_matmul_nvfp4.cu:2703`), so there
// is no host path to decode a packed weight against. The EXL3 sibling suite can
// build a decoded twin because `vt::Exl3DequantLinear` runs on CPU; this arm
// cannot, and the device leg is owed by `BENCH-QWEN38-27B-SOTA` leg F. What a
// load gate CAN falsify is every silent mis-load: a wrong `weight_scale_2`, a
// transposed projection, a module skipped because the declaration excluded it.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/models/dsv4_exl3_fixture.h"

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;
using dsv4_exl3_fixture::Raw;
using dsv4_exl3_fixture::StEntry;
using dsv4_exl3_fixture::TempDir;
using dsv4_exl3_fixture::WriteSafetensors;
using json = nlohmann::json;
using vllm::HfConfig;
using vllm::Nvfp4Weight;
using vllm::Qwen3DFlashWeights;

namespace {

// The published artifact's shape, scaled down. H, the two head products and I
// are all multiples of 16 because NVFP4 packs 16 elements to one fp8 scale, and
// `LoadNvfp4AnyNaming` refuses a K that is not.
struct Dims {
  int64_t H = 128;
  int64_t Hq = 4;
  int64_t Hkv = 2;
  int64_t Dh = 32;   // qdim = 128, kvdim = 64
  int64_t I = 256;
  int64_t vocab = 32;
  int64_t layers = 2;
  int64_t taps_fc = 2;  // fc input = H * taps = 256
  int64_t conv_taps = 2;
  int64_t conv_group = 16;  // 128 / 16 = 8 groups
  int64_t block = 4;
  int64_t sel_rank = 8;
  int64_t sel_top_k = 3;
  int64_t qdim() const { return Hq * Dh; }
  int64_t kvdim() const { return Hkv * Dh; }
  int64_t groups() const { return H / conv_group; }
};

uint32_t Hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

std::vector<uint16_t> Bf16Fill(int64_t n, uint32_t seed, float amp) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t h = Hash32(seed * 2654435761U + static_cast<uint32_t>(i));
    const float u = static_cast<float>(h % 20001) / 10000.0F - 1.0F;
    v[static_cast<size_t>(i)] = vt::F32ToBF16(u * amp);
  }
  return v;
}

StEntry Bf16(const std::string& name, const std::vector<int64_t>& shape, uint32_t seed,
             float amp) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return StEntry{name, "BF16", shape, Raw(Bf16Fill(n, seed, amp))};
}

StEntry F32Scalar(const std::string& name, float value) {
  std::vector<float> v{value};
  return StEntry{name, "F32", {}, Raw(v)};
}

// One ModelOpt NVFP4 module, in the four-tensor spelling the artifact ships.
// The E2M1 nibbles and the fp8-e4m3 group scales are arbitrary bytes: this is a
// load gate, and no arithmetic is performed on them on the host.
void PushNvfp4(std::vector<StEntry>& e, const std::string& proj, int64_t n, int64_t k,
               uint32_t seed, float scale2, bool with_input_scale = true) {
  REQUIRE(k % 16 == 0);
  std::vector<uint8_t> packed(static_cast<size_t>(n) * static_cast<size_t>(k / 2));
  for (size_t i = 0; i < packed.size(); ++i)
    packed[i] = static_cast<uint8_t>(Hash32(seed + static_cast<uint32_t>(i)) & 0xffU);
  std::vector<uint8_t> scale(static_cast<size_t>(n) * static_cast<size_t>(k / 16));
  for (size_t i = 0; i < scale.size(); ++i)
    scale[i] = static_cast<uint8_t>(0x38U + (Hash32(seed ^ static_cast<uint32_t>(i)) & 7U));
  e.push_back(StEntry{proj + ".weight", "U8", {n, k / 2}, packed});
  e.push_back(StEntry{proj + ".weight_scale", "F8_E4M3", {n, k / 16}, scale});
  e.push_back(F32Scalar(proj + ".weight_scale_2", scale2));
  if (with_input_scale) e.push_back(F32Scalar(proj + ".input_scale", 0.125F));
}

// The bf16 spelling of the same module, for the arms and cross-checks that need
// one.
void PushBf16Linear(std::vector<StEntry>& e, const std::string& proj, int64_t n,
                    int64_t k, uint32_t seed) {
  e.push_back(Bf16(proj + ".weight", {n, k}, seed, 0.05F));
}

// Which of the seven per-layer projections a fixture stores packed.
enum class LayerArm { kAllBf16, kAllNvfp4 };

// `scale2` is a per-module value so a swapped or dropped read is visible: every
// module gets a different one and the gate asserts each.
float Scale2For(int64_t layer, int idx) {
  return 0.0078125F * static_cast<float>(1 + idx) + 0.001953125F * static_cast<float>(layer);
}

struct FixtureOptions {
  LayerArm layer_arm = LayerArm::kAllNvfp4;
  bool quantize_fc = false;
  bool quantize_selector = false;
  bool quantize_conv = false;
  // Drop `layers.0.self_attn.q_proj` from the exclude list while storing it
  // bf16 — the "declared quantized, ships nothing" direction.
  bool q_proj_declared_but_bf16 = false;
  // Store `fc` packed while leaving it in the exclude list — the "excluded but
  // shipped" direction, on a module that also has no owner.
  bool exclude_fc_but_ship_packed = false;
  bool with_input_scale = true;
};

std::vector<StEntry> DraftEntries(const Dims& dm, const FixtureOptions& opt) {
  std::vector<StEntry> e;
  const bool packed_layers = opt.layer_arm == LayerArm::kAllNvfp4;

  if (opt.quantize_fc || opt.exclude_fc_but_ship_packed) {
    PushNvfp4(e, "fc", dm.H, dm.H * dm.taps_fc, 11, 0.03125F, opt.with_input_scale);
  } else {
    e.push_back(Bf16("fc.weight", {dm.H, dm.H * dm.taps_fc}, 11, 0.05F));
  }
  e.push_back(Bf16("hidden_norm.weight", {dm.H}, 12, 1.0F));
  e.push_back(Bf16("norm.weight", {dm.H}, 13, 1.0F));

  for (int64_t l = 0; l < dm.layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    const std::string sa = b + "self_attn.";
    const std::string mp = b + "mlp.";
    e.push_back(Bf16(b + "input_layernorm.weight", {dm.H}, 100 + l, 1.0F));
    e.push_back(Bf16(b + "post_attention_layernorm.weight", {dm.H}, 200 + l, 1.0F));
    e.push_back(Bf16(sa + "q_norm.weight", {dm.Dh}, 300 + l, 1.0F));
    e.push_back(Bf16(sa + "k_norm.weight", {dm.Dh}, 400 + l, 1.0F));

    const bool q_packed =
        packed_layers && !(l == 0 && opt.q_proj_declared_but_bf16);
    if (q_packed) {
      PushNvfp4(e, sa + "q_proj", dm.qdim(), dm.H, 500 + static_cast<uint32_t>(l),
                Scale2For(l, 0), opt.with_input_scale);
    } else {
      PushBf16Linear(e, sa + "q_proj", dm.qdim(), dm.H, 500 + static_cast<uint32_t>(l));
    }
    if (packed_layers) {
      PushNvfp4(e, sa + "k_proj", dm.kvdim(), dm.H, 510 + static_cast<uint32_t>(l),
                Scale2For(l, 1), opt.with_input_scale);
      PushNvfp4(e, sa + "v_proj", dm.kvdim(), dm.H, 520 + static_cast<uint32_t>(l),
                Scale2For(l, 2), opt.with_input_scale);
      PushNvfp4(e, sa + "o_proj", dm.H, dm.qdim(), 530 + static_cast<uint32_t>(l),
                Scale2For(l, 3), opt.with_input_scale);
      PushNvfp4(e, mp + "gate_proj", dm.I, dm.H, 540 + static_cast<uint32_t>(l),
                Scale2For(l, 4), opt.with_input_scale);
      PushNvfp4(e, mp + "up_proj", dm.I, dm.H, 550 + static_cast<uint32_t>(l),
                Scale2For(l, 5), opt.with_input_scale);
      PushNvfp4(e, mp + "down_proj", dm.H, dm.I, 560 + static_cast<uint32_t>(l),
                Scale2For(l, 6), opt.with_input_scale);
    } else {
      PushBf16Linear(e, sa + "k_proj", dm.kvdim(), dm.H, 510 + static_cast<uint32_t>(l));
      PushBf16Linear(e, sa + "v_proj", dm.kvdim(), dm.H, 520 + static_cast<uint32_t>(l));
      PushBf16Linear(e, sa + "o_proj", dm.H, dm.qdim(), 530 + static_cast<uint32_t>(l));
      PushBf16Linear(e, mp + "gate_proj", dm.I, dm.H, 540 + static_cast<uint32_t>(l));
      PushBf16Linear(e, mp + "up_proj", dm.I, dm.H, 550 + static_cast<uint32_t>(l));
      PushBf16Linear(e, mp + "down_proj", dm.H, dm.I, 560 + static_cast<uint32_t>(l));
    }

    for (const char* which : {"attention_conv.", "mlp_conv."}) {
      const std::string cp = b + which;
      e.push_back(Bf16(cp + "base_kernel", {2, dm.conv_taps, dm.H},
                       600 + static_cast<uint32_t>(l), 0.1F));
      const int64_t kp_rows = 2 * dm.conv_taps * dm.groups();
      if (opt.quantize_conv) {
        PushNvfp4(e, cp + "kernel_projection", kp_rows, dm.H,
                  700 + static_cast<uint32_t>(l), 0.015625F, opt.with_input_scale);
      } else {
        e.push_back(Bf16(cp + "kernel_projection.weight", {kp_rows, dm.H},
                         700 + static_cast<uint32_t>(l), 0.05F));
      }
    }
  }

  if (opt.quantize_selector) {
    PushNvfp4(e, "candidate_selector.hidden_projection", dm.sel_rank, dm.H, 900,
              0.03125F, opt.with_input_scale);
  } else {
    e.push_back(
        Bf16("candidate_selector.hidden_projection.weight", {dm.sel_rank, dm.H}, 900, 0.1F));
  }
  e.push_back(Bf16("candidate_selector.predecessor_codebook", {dm.vocab, dm.sel_rank}, 901,
                   0.3F));
  e.push_back(Bf16("candidate_selector.successor_codebook", {dm.vocab, dm.sel_rank}, 902,
                   0.3F));
  return e;
}

// The 12 exact-name exclusions, exactly as the published artifact spells them:
// no wildcard, one entry per conv projection per layer.
json ExcludeList(const Dims& dm, const FixtureOptions& opt) {
  json ex = json::array();
  if (!opt.quantize_selector) ex.push_back("candidate_selector.hidden_projection");
  if (!opt.quantize_fc) ex.push_back("fc");
  if (!opt.quantize_conv) {
    for (int64_t l = 0; l < dm.layers; ++l) {
      ex.push_back("layers." + std::to_string(l) + ".attention_conv.kernel_projection");
      ex.push_back("layers." + std::to_string(l) + ".mlp_conv.kernel_projection");
    }
  }
  if (opt.q_proj_declared_but_bf16) {
    // `layers.0.self_attn.q_proj` is stored BF16 and is deliberately NOT
    // excluded, so the declaration and the tensors disagree.
  }
  return ex;
}

json BaseConfigJson(const Dims& dm) {
  json c = json::object();
  c["hidden_size"] = dm.H;
  c["num_attention_heads"] = dm.Hq;
  c["num_key_value_heads"] = dm.Hkv;
  c["head_dim"] = dm.Dh;
  c["intermediate_size"] = dm.I;
  c["vocab_size"] = dm.vocab;
  c["num_hidden_layers"] = dm.layers;
  c["rms_norm_eps"] = 1e-6;
  c["sliding_window"] = 2048;
  c["is_causal"] = false;
  c["rope_parameters"] = {{"rope_theta", 10000000.0}, {"rope_type", "default"}};
  c["layer_types"] = json::array();
  for (int64_t i = 0; i < dm.layers; ++i) c["layer_types"].push_back("sliding_attention");
  json d = json::object();
  d["mask_token_id"] = dm.vocab - 1;
  d["target_layer_ids"] = json::array({1, 3});
  d["conv_kernel_size"] = dm.conv_taps;
  d["conv_group_size"] = dm.conv_group;
  d["block_size"] = dm.block;
  d["selector_rank"] = dm.sel_rank;
  d["selector_top_k"] = dm.sel_top_k;
  c["dflash_config"] = d;
  return c;
}

// The published `quantization_config`: FLAT shape, `quant_method: "modelopt"`,
// and BOTH `exclude_modules` and `ignore` carrying the same list — which is
// what the artifact ships. Upstream's `from_config` reads `ignore` on a flat
// document (modelopt.py:298-311), and so does this loader.
json ModelOptQuantConfig(const json& exclude, const char* algo = "NVFP4") {
  json q = json::object();
  q["quant_method"] = "modelopt";
  q["quant_algo"] = algo;
  q["kv_cache_quant_algo"] = "FP8";
  q["group_size"] = 16;
  q["exclude_modules"] = exclude;
  q["ignore"] = exclude;
  return q;
}

// The PRODUCTION pair, in the production order.
Qwen3DFlashWeights LoadDraft(const fs::path& dir, const Dims& dm, const json& config_json,
                             const std::vector<StEntry>& entries) {
  WriteSafetensors(dir / "model.safetensors", entries);
  const HfConfig c = vllm::MakeQwen3DFlashDraftConfig(config_json);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open((dir / "model.safetensors").string()));
  return vllm::LoadQwen3DFlash(shards, c, dm.taps_fc, static_cast<int32_t>(dm.vocab - 1));
}

std::string ThrowText(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return std::string(e.what());
  }
  return std::string();
}

[[maybe_unused]] bool Contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("dflash2 nvfp4: the published shape loads through the production loader") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt));

  Qwen3DFlashWeights w = LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt));

  CHECK(w.IsDflash2());
  CHECK(w.IsNvfp4());
  CHECK_FALSE(w.IsExl3());
  REQUIRE(w.layers.size() == static_cast<size_t>(dm.layers));

  for (int64_t l = 0; l < dm.layers; ++l) {
    const auto& L = w.layers[static_cast<size_t>(l)];
    // Seven packed owners, and the two MERGED bf16 owners left empty. A layer
    // that filled both would be carrying the weights twice.
    CHECK(L.qkv_proj.Empty());
    CHECK(L.gate_up_proj.Empty());
    CHECK(L.o_proj.Empty());
    CHECK(L.down_proj.Empty());

    // GEOMETRY. A transposed projection loads and returns a confidently wrong
    // answer, so n and k are asserted per owner rather than "non-empty".
    CHECK(L.q_proj_fp4.n == dm.qdim());
    CHECK(L.q_proj_fp4.k == dm.H);
    CHECK(L.k_proj_fp4.n == dm.kvdim());
    CHECK(L.k_proj_fp4.k == dm.H);
    CHECK(L.v_proj_fp4.n == dm.kvdim());
    CHECK(L.v_proj_fp4.k == dm.H);
    CHECK(L.o_proj_fp4.n == dm.H);
    CHECK(L.o_proj_fp4.k == dm.qdim());
    CHECK(L.gate_proj_fp4.n == dm.I);
    CHECK(L.gate_proj_fp4.k == dm.H);
    CHECK(L.up_proj_fp4.n == dm.I);
    CHECK(L.up_proj_fp4.k == dm.H);
    CHECK(L.down_proj_fp4.n == dm.H);
    CHECK(L.down_proj_fp4.k == dm.I);

    // THE GLOBAL SCALE, per module. A dropped or swapped `weight_scale_2` is
    // the invisible defect on this format: the weight stays correctly
    // distributed and is entirely wrong, and because the DFlash verify is
    // lossless the draft would still emit the target's tokens.
    const Nvfp4Weight* owners[7] = {&L.q_proj_fp4,    &L.k_proj_fp4,  &L.v_proj_fp4,
                                    &L.o_proj_fp4,    &L.gate_proj_fp4,
                                    &L.up_proj_fp4,   &L.down_proj_fp4};
    for (int i = 0; i < 7; ++i) {
      CHECK(owners[i]->scale2 == doctest::Approx(Scale2For(l, i)));
      // W4A16 is the executed arm: VT_MODELOPT_W4A4 is unset in the gate, so
      // the activation divisor is NOT consumed and IsTrueW4A4() stays false.
      CHECK_FALSE(owners[i]->IsTrueW4A4());
      CHECK(owners[i]->group_size == 16);
      CHECK_FALSE(owners[i]->is_mxfp4);
    }

    // The 12 EXCLUDED modules stayed dense, at the checkpoint's own dtype.
    CHECK(L.attention_conv.kernel_projection.rank == 2);
    CHECK(L.attention_conv.kernel_projection.dtype == vt::DType::kBF16);
    CHECK(L.mlp_conv.kernel_projection.dtype == vt::DType::kBF16);
  }
  CHECK_FALSE(w.fc.Empty());
  CHECK(w.fc.dtype == vt::DType::kBF16);
  CHECK(w.fc.shape[0] == dm.H);
  CHECK(w.fc.shape[1] == dm.H * dm.taps_fc);
  CHECK(w.candidate_selector.hidden_projection.dtype == vt::DType::kBF16);
}

TEST_CASE("dflash2 nvfp4: a draft with no quantization_config is byte-unchanged") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  opt.layer_arm = LayerArm::kAllBf16;
  const json c = BaseConfigJson(dm);  // no quantization_config at all

  Qwen3DFlashWeights w = LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt));

  CHECK(w.IsDflash2());
  CHECK_FALSE(w.IsNvfp4());
  CHECK_FALSE(w.IsExl3());
  for (const auto& L : w.layers) {
    CHECK_FALSE(L.qkv_proj.Empty());
    CHECK_FALSE(L.gate_up_proj.Empty());
    CHECK(L.q_proj_fp4.Empty());
    CHECK(L.down_proj_fp4.Empty());
  }
}

TEST_CASE("dflash2 nvfp4: a quantized fc is refused BY NAME, not by dtype") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  opt.quantize_fc = true;
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt));

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "\"fc\""));
  CHECK(Contains(msg, "NO packed owner"));
  CHECK(Contains(msg, "Qwen3DFlashWeights::fc"));
  CHECK(Contains(msg, "2758"));
  // The defect this replaces: a BF16 dtype complaint about a tensor that is
  // exactly what its checkpoint says it is.
  CHECK_FALSE(Contains(msg, "expected BF16"));
}

TEST_CASE("dflash2 nvfp4: a quantized selector projection is refused BY NAME") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  opt.quantize_selector = true;
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt));

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "candidate_selector.hidden_projection"));
  CHECK(Contains(msg, "Dflash2SelectorWeights::hidden_projection"));
  CHECK(Contains(msg, "NO packed owner"));
}

TEST_CASE("dflash2 nvfp4: a quantized conv kernel_projection is refused BY NAME") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  opt.quantize_conv = true;
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt));

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "attention_conv.kernel_projection"));
  CHECK(Contains(msg, "Qwen3DFlashConvWeights::kernel_projection"));
  CHECK(Contains(msg, "NO packed owner"));
}

TEST_CASE("dflash2 nvfp4: declared quantized but shipping BF16 is refused") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  opt.q_proj_declared_but_bf16 = true;
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt));

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "layers.0.self_attn.q_proj"));
  CHECK(Contains(msg, "declared QUANTIZED"));
  CHECK(Contains(msg, "ships NO NVFP4 operands"));
}

TEST_CASE("dflash2 nvfp4: excluded but shipping NVFP4 is refused") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  // `fc` stays in the exclude list AND is written packed.
  opt.exclude_fc_but_ship_packed = true;
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt));

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "\"fc\""));
  CHECK(Contains(msg, "stored as an NVFP4 module"));
}

TEST_CASE("dflash2 nvfp4: NVFP4 tensors with no declaration at all are refused") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  const json c = BaseConfigJson(dm);  // packed tensors, no quantization_config

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "declares NO quantization_config"));
  CHECK(Contains(msg, "get_draft_quant_config"));
}

TEST_CASE("dflash2 nvfp4: an unimplemented quant_algo is named, not guessed") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  opt.layer_arm = LayerArm::kAllBf16;
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt), "FP8");

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "quant_algo \"FP8\""));
  CHECK(Contains(msg, "NO arm"));
  CHECK(Contains(msg, "2758"));
}

TEST_CASE("dflash2 nvfp4: a non-ModelOpt quant_method is named, not guessed") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  opt.layer_arm = LayerArm::kAllBf16;
  json c = BaseConfigJson(dm);
  json q = ModelOptQuantConfig(ExcludeList(dm, opt));
  q["quant_method"] = "awq";
  c["quantization_config"] = q;

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "quant_method \"awq\""));
  CHECK(Contains(msg, "2758"));
}

TEST_CASE("dflash2 nvfp4: W4A16_NVFP4 loads and is not reported as a divergence") {
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  opt.with_input_scale = false;  // weight-only checkpoints ship no divisor
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt), "W4A16_NVFP4");

  Qwen3DFlashWeights w = LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt));
  CHECK(w.IsNvfp4());
  CHECK_FALSE(w.layers.front().q_proj_fp4.IsTrueW4A4());
}

TEST_CASE("dflash2 nvfp4: a WILDCARD exclusion is honoured") {
  // The published artifact enumerates every conv projection by name, so the
  // `fnmatch` pass of upstream's `is_layer_excluded` (modelopt.py:171-175) is
  // unexercised by it. It is not unexercised by the rule.
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  json ex = json::array();
  ex.push_back("candidate_selector.hidden_projection");
  ex.push_back("fc");
  ex.push_back("layers.*.attention_conv.kernel_projection");
  ex.push_back("layers.*.mlp_conv.kernel_projection");
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ex);

  Qwen3DFlashWeights w = LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt));
  CHECK(w.IsNvfp4());
  CHECK(w.layers.front().attention_conv.kernel_projection.dtype == vt::DType::kBF16);
}

TEST_CASE("dflash2 nvfp4: the nested hf_quant_config shape resolves the same way") {
  // `from_config` picks the two SHAPES apart (modelopt.py:283-318): the nested
  // `{"quantization": {...}}` of `hf_quant_config.json` reads `exclude_modules`,
  // the flat `config.json` document reads `ignore`. The published drafter ships
  // both files with identical content, so a loader that read the wrong key on
  // the wrong shape would pass on it and fail on the next artifact.
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  json nested = json::object();
  nested["quant_method"] = "modelopt";
  nested["quantization"] = json::object();
  nested["quantization"]["quant_algo"] = "NVFP4";
  nested["quantization"]["group_size"] = 16;
  nested["quantization"]["exclude_modules"] = ExcludeList(dm, opt);
  json c = BaseConfigJson(dm);
  c["quantization_config"] = nested;

  Qwen3DFlashWeights w = LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt));
  CHECK(w.IsNvfp4());
  CHECK(w.fc.dtype == vt::DType::kBF16);
}

TEST_CASE("dflash2 nvfp4: a lane with no presence predicate refuses by lane, not by tensor") {
  // The GGUF draft builder and the DSpark backbone call the four-argument
  // overload, which supplies no `has`. Neither container can carry a packed
  // ModelOpt module, and the refusal must name the LANE rather than report that
  // the tensors are missing.
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  json cj = BaseConfigJson(dm);
  cj["quantization_config"] = ModelOptQuantConfig(ExcludeList(dm, opt));
  WriteSafetensors(dir.path() / "model.safetensors", DraftEntries(dm, opt));
  const HfConfig c = vllm::MakeQwen3DFlashDraftConfig(cj);
  vllm::SafetensorsFile f =
      vllm::SafetensorsFile::Open((dir.path() / "model.safetensors").string());
  std::unordered_map<std::string, const vllm::StTensor*> unused;
  const vllm::TensorResolver get = [&f](const std::string& name) -> const vllm::StTensor& {
    return f.Get(name);
  };
  const std::string msg = ThrowText([&] {
    (void)vllm::LoadQwen3DFlash(get, c, dm.taps_fc, static_cast<int32_t>(dm.vocab - 1));
  });
  REQUIRE_FALSE(msg.empty());
  CHECK(Contains(msg, "supplies no tensor-presence predicate"));
  CHECK(Contains(msg, "DSpark"));
}

TEST_CASE("dflash2 nvfp4: a partially quantized layer is refused rather than half-loaded") {
  // q/k/v share ONE merged bf16 owner and gate/up share another, so a layer
  // that quantizes some and excludes others has no expressible owner here.
  const Dims dm;
  TempDir dir;
  FixtureOptions opt;
  json ex = ExcludeList(dm, opt);
  ex.push_back("layers.1.mlp.down_proj");
  json c = BaseConfigJson(dm);
  c["quantization_config"] = ModelOptQuantConfig(ex);

  const std::string msg =
      ThrowText([&] { (void)LoadDraft(dir.path(), dm, c, DraftEntries(dm, opt)); });
  REQUIRE_FALSE(msg.empty());
  // The cross-check fires first: `layers.1.mlp.down_proj` is excluded by the
  // declaration and packed in the file. Either refusal names the module.
  CHECK(Contains(msg, "layers.1.mlp.down_proj"));
}
