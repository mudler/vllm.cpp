// M2a — Qwen3-VL vision-tower faithfulness gate. Runs the C++
// Qwen3_VisionTransformer forward (src/vllm/model_executor/models/qwen3_vl_vision
// .cpp) on the fixed M1 fixture image and asserts every stage matches the dumped
// vLLM 0.25.0 reference (scripts/mm/m2a_tower_ref_dump.py) within a stated bf16
// tolerance:
//   patch_embed_out, block0_out, merger_out, deepstack_{0,1,2}_out, tower_out
// plus the two NEW host precomputes (pos_embeds bilinear interp, vision-rope
// cos|sin). This is the M2a milestone: the tower proven faithful in ISOLATION
// (image token-exact e2e is M2c, after the M2b MRoPE/DeepStack text backbone).
//
// GPU + weights required: the visual.* weights are dumped (f32, ≈1.2 GiB, NOT
// committed) via scripts/mm/m2a_tower_weight_dump.py; point VLLM_QWEN3VL_WEIGHTS
// at the dir. Without it (or without CUDA) the gate is skipped, not failed.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::multimodal::Qwen3VLVisionCapture;
using vllm::multimodal::Qwen3VLVisionConfig;
using vllm::multimodal::Qwen3VLVisionWeights;
using vllm::multimodal::VisionBlockWeights;
using vllm::multimodal::VisionMergerWeights;

std::string TowerFix() { return std::string(TOWER_FIXTURE_DIR); }
std::string ImgFix() { return std::string(MM_FIXTURE_DIR); }

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

std::vector<int64_t> ReadI64(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<int64_t> v(static_cast<size_t>(n) / sizeof(int64_t));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

// relative L2 error ||got-ref|| / ||ref||, plus max-abs (reported on failure).
struct Err {
  double rel_l2 = 0.0;
  double max_abs = 0.0;
};
Err Compare(const std::vector<float>& got, const std::vector<float>& ref) {
  REQUIRE(got.size() == ref.size());
  double num = 0.0, den = 0.0, mx = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    mx = std::max(mx, std::abs(d));
  }
  return Err{std::sqrt(num / (den + 1e-30)), mx};
}

// Load a tower-relative weight tensor from the (uncommitted) dump dir.
std::vector<float> W(const std::string& dir, const std::string& name) {
  return ReadF32(dir + "/" + name + ".bin");
}

// The same dumped weight, narrowed to the tower's bf16 store (#1359). Not a
// precision change: `PrepareVisionDeviceWeights` put every one of these through
// `vt::F32ToBF16` in `MakeDevBf16` before the first GEMM, so the device bytes are
// the ones this dump always produced. `pos_embed.weight` is the ONE weight that
// stays host f32 and therefore keeps using `W` — the host bilinear interpolation
// reads it before anything narrows.
std::vector<uint16_t> WBits(const std::string& dir, const std::string& name) {
  const std::vector<float> f = W(dir, name);
  std::vector<uint16_t> o(f.size());
  for (size_t i = 0; i < f.size(); ++i) o[i] = vt::F32ToBF16(f[i]);
  return o;
}

VisionMergerWeights LoadMerger(const std::string& dir, const std::string& prefix,
                               bool postshuffle) {
  VisionMergerWeights m;
  m.use_postshuffle_norm = postshuffle;
  m.norm_w = WBits(dir, prefix + ".norm.weight");
  m.norm_b = WBits(dir, prefix + ".norm.bias");
  m.fc1_w = WBits(dir, prefix + ".linear_fc1.weight");
  m.fc1_b = WBits(dir, prefix + ".linear_fc1.bias");
  m.fc2_w = WBits(dir, prefix + ".linear_fc2.weight");
  m.fc2_b = WBits(dir, prefix + ".linear_fc2.bias");
  return m;
}

}  // namespace

TEST_CASE("qwen3vl_vision_tower_faithful_vs_vllm_0_25_0") {
  const char* wdir_c = std::getenv("VLLM_QWEN3VL_WEIGHTS");
  if (wdir_c == nullptr) {
    MESSAGE("SKIP: set VLLM_QWEN3VL_WEIGHTS to the dumped visual.* weight dir "
            "(scripts/mm/m2a_tower_weight_dump.py) to run the tower gate");
    return;
  }
  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  const std::string wdir = wdir_c;

  Qwen3VLVisionConfig cfg;  // Qwen3-VL-4B vision defaults (see header).

  // --- weights --------------------------------------------------------------
  Qwen3VLVisionWeights w;
  w.patch_proj_w = WBits(wdir, "patch_embed.proj.weight");
  w.patch_proj_b = WBits(wdir, "patch_embed.proj.bias");
  w.pos_embed_w = W(wdir, "pos_embed.weight");
  w.blocks.resize(static_cast<size_t>(cfg.depth));
  for (int64_t l = 0; l < cfg.depth; ++l) {
    const std::string p = "blocks." + std::to_string(l);
    VisionBlockWeights& b = w.blocks[static_cast<size_t>(l)];
    b.norm1_w = WBits(wdir, p + ".norm1.weight");
    b.norm1_b = WBits(wdir, p + ".norm1.bias");
    b.norm2_w = WBits(wdir, p + ".norm2.weight");
    b.norm2_b = WBits(wdir, p + ".norm2.bias");
    b.qkv_w = WBits(wdir, p + ".attn.qkv.weight");
    b.qkv_b = WBits(wdir, p + ".attn.qkv.bias");
    b.proj_w = WBits(wdir, p + ".attn.proj.weight");
    b.proj_b = WBits(wdir, p + ".attn.proj.bias");
    b.fc1_w = WBits(wdir, p + ".mlp.linear_fc1.weight");
    b.fc1_b = WBits(wdir, p + ".mlp.linear_fc1.bias");
    b.fc2_w = WBits(wdir, p + ".mlp.linear_fc2.weight");
    b.fc2_b = WBits(wdir, p + ".mlp.linear_fc2.bias");
  }
  w.merger = LoadMerger(wdir, "merger", /*postshuffle=*/false);
  for (size_t i = 0; i < cfg.deepstack_visual_indexes.size(); ++i)
    w.deepstack_mergers.push_back(
        LoadMerger(wdir, "deepstack_merger_list." + std::to_string(i), /*postshuffle=*/true));

  // --- inputs (fixture) -----------------------------------------------------
  std::vector<float> pix_f = ReadF32(ImgFix() + "/pixel_values_f32.bin");  // bf16-widened
  std::vector<uint16_t> pix_bf16(pix_f.size());
  for (size_t i = 0; i < pix_f.size(); ++i) pix_bf16[i] = vt::F32ToBF16(pix_f[i]);
  std::vector<int64_t> g = ReadI64(ImgFix() + "/image_grid_thw_i64.bin");
  std::array<int64_t, 3> grid_thw{g[0], g[1], g[2]};

  // --- run tower with capture -----------------------------------------------
  Qwen3VLVisionCapture cap;
  std::vector<float> tower =
      vllm::multimodal::Qwen3VLVisionForward(pix_bf16, grid_thw, w, cfg, *gpu, &cap);

  // --- references -----------------------------------------------------------
  const std::string T = TowerFix();
  auto ref_pos = ReadF32(T + "/pos_embeds.bin");
  auto ref_cos = ReadF32(T + "/rotary_cos.bin");
  auto ref_sin = ReadF32(T + "/rotary_sin.bin");
  auto ref_patch = ReadF32(T + "/patch_embed_out.bin");
  auto ref_block0 = ReadF32(T + "/block0_out.bin");
  auto ref_merger = ReadF32(T + "/merger_out.bin");
  auto ref_ds0 = ReadF32(T + "/deepstack_0_out.bin");
  auto ref_ds1 = ReadF32(T + "/deepstack_1_out.bin");
  auto ref_ds2 = ReadF32(T + "/deepstack_2_out.bin");
  auto ref_tower = ReadF32(T + "/tower_out.bin");

  // --- Tolerance model (RCA, measured 2026-07-25 on GB10 sm_121a) -----------
  // Every NEW kernel and the first full block match vLLM TIGHTLY, proving the
  // ops are correct: pos-embed interp relL2 2.5e-3, vision-rope maxabs 1.9e-3,
  // patch-embed 2.1e-3, block0 6.8e-3. Deeper stages diverge only by SMOOTH,
  // monotone bf16 accumulation across 24 layers between two independent bf16
  // kernel stacks (our vt ops vs vLLM's cuBLAS/FlashAttention + bf16 rope):
  // hidden relL2 grows ~0.25%/layer (L5 1.2e-2, L11 3.3e-2, L17 4.4e-2,
  // L24 6.5e-2) with NO discontinuity — i.e. numerical, not a logic error. The
  // deep-stage bounds below are that measured bf16-depth envelope (measured x
  // ~1.25 margin); they still FAIL hard on any wrong op (the RED injection
  // drives >>1e-1). Token-exact e2e is the ultimate numeric bar at M2c.

  // NEW host precomputes (pos-embed bilinear interp; vision-rope cos|sin).
  Err e_pos = Compare(cap.pos_embeds, ref_pos);
  INFO("pos_embeds relL2=", e_pos.rel_l2, " maxabs=", e_pos.max_abs);
  CHECK(e_pos.rel_l2 < 5e-3);
  Err e_cos = Compare(cap.rotary_cos, ref_cos);
  Err e_sin = Compare(cap.rotary_sin, ref_sin);
  INFO("rope cos maxabs=", e_cos.max_abs, " sin maxabs=", e_sin.max_abs);
  CHECK(e_cos.max_abs < 3e-3);
  CHECK(e_sin.max_abs < 3e-3);

  // Gate 1 — patch-embed (matmul+bias). TIGHT.
  Err e_patch = Compare(cap.patch_embed_out, ref_patch);
  INFO("patch_embed relL2=", e_patch.rel_l2, " maxabs=", e_patch.max_abs);
  CHECK(e_patch.rel_l2 < 5e-3);

  // Gate 2 — one full ViT block (LN + vision attn + rope + tanh-GELU MLP). TIGHT.
  Err e_block0 = Compare(cap.block0_out, ref_block0);
  INFO("block0 relL2=", e_block0.rel_l2, " maxabs=", e_block0.max_abs);
  CHECK(e_block0.rel_l2 < 1.2e-2);

  // Gate 3 — patch merger (LN + exact-erf-GELU + 2 FCs) on the final hidden.
  Err e_merger = Compare(cap.merger_out, ref_merger);
  INFO("merger relL2=", e_merger.rel_l2, " maxabs=", e_merger.max_abs);
  CHECK(e_merger.rel_l2 < 8e-2);

  // Gate 4 — DeepStack taps (post-shuffle mergers at layers 5/11/17). Envelope
  // scales with tap depth.
  REQUIRE(cap.deepstack_out.size() == 3);
  Err e_ds0 = Compare(cap.deepstack_out[0], ref_ds0);
  Err e_ds1 = Compare(cap.deepstack_out[1], ref_ds1);
  Err e_ds2 = Compare(cap.deepstack_out[2], ref_ds2);
  INFO("deepstack relL2 ds0=", e_ds0.rel_l2, " ds1=", e_ds1.rel_l2, " ds2=", e_ds2.rel_l2);
  CHECK(e_ds0.rel_l2 < 2e-2);   // L5
  CHECK(e_ds1.rel_l2 < 4.5e-2); // L11
  CHECK(e_ds2.rel_l2 < 6e-2);   // L17

  // Full tower output [196, 10240] = concat(merger | ds0 | ds1 | ds2).
  Err e_tower = Compare(tower, ref_tower);
  INFO("tower_out relL2=", e_tower.rel_l2, " maxabs=", e_tower.max_abs);
  CHECK(e_tower.rel_l2 < 7e-2);
}
