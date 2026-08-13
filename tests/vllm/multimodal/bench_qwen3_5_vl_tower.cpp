// Vision-tower SPEED A-B (CLAIM-MULTIMODAL-SPEED-TOWER) on Qwen3.6-27B.
//
// The mm scoping run (.agents/specs/multimodal-speed.md) attributed the ENTIRE
// mm first-token gap to the vision encoder tower (~2114 ms ours vs ~250 ms vLLM
// eager). This driver isolates WHERE that time goes and what the fix recovers:
//
//   BASELINE (the old M2a per-call behavior) = the host-weights overload, which
//   converts every weight host f32->bf16 and re-uploads ~0.5 GiB INSIDE the timed
//   forward every call (prepare) + runs the ViT kernels (forward).
//   FAST (the speed pass) = PrepareVisionDeviceWeights ONCE (amortized model-load,
//   NOT per-image) then the resident-weights forward per image — only the tiny
//   pixel/pos/rope uploads + the ViT GEMMs/attention, which is the fair A-B vs
//   vLLM's ~250 ms encode (whose nn.Linears are likewise pre-loaded).
//
// The numeric correctness of the resident forward (image STRICT 32/32) is proven
// by test_qwen3_5_vl_e2e, which calls the host overload that now delegates to the
// resident forward — BIT-IDENTICAL. This driver is timing only.
//
// dgx-only: needs CUDA + the vision-inclusive bf16 checkpoint Qwen/Qwen3.6-27B
// (VLLM_QWEN36_CKPT or the HF cache path). Skipped (not failed) without.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/support/test_platform_compat.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vt/backend.h"

namespace {
namespace fs = std::filesystem;

std::string ImgFix() { return std::string(MM_FIXTURE_DIR); }

std::vector<uint8_t> ReadU8(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> v(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

std::string FindCkpt() {
  if (const char* e = std::getenv("VLLM_QWEN36_CKPT")) return e;
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) / ".cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& d : fs::directory_iterator(snaps, ec))
    if (fs::exists(d.path() / "config.json", ec)) return d.path().string();
  return "";
}

vllm::multimodal::Qwen3VLVisionConfig VisionConfig27B() {
  vllm::multimodal::Qwen3VLVisionConfig v;
  v.hidden_size = 1152;
  v.num_heads = 16;
  v.depth = 27;
  v.intermediate_size = 4304;
  v.out_hidden_size = 5120;
  v.patch_size = 16;
  v.temporal_patch_size = 2;
  v.spatial_merge_size = 2;
  v.num_position_embeddings = 2304;
  v.in_channels = 3;
  v.deepstack_visual_indexes = {};
  v.norm_eps = 1e-6f;
  return v;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

TEST_CASE("qwen3_5_27b_vision_tower_speed_AB") {
  const std::string ckpt = FindCkpt();
  if (ckpt.empty()) {
    MESSAGE("SKIP: Qwen3.6-27B checkpoint absent (set VLLM_QWEN36_CKPT)");
    return;
  }
  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }

  std::vector<vllm::SafetensorsFile> shards;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(ckpt, ec))
    if (e.path().extension() == ".safetensors")
      shards.push_back(vllm::SafetensorsFile::Open(e.path().string()));
  REQUIRE(!shards.empty());

  const vllm::multimodal::Qwen3VLVisionConfig vcfg = VisionConfig27B();
  const vllm::multimodal::Qwen3VLVisionWeights vw =
      vllm::LoadQwen3VLVisionWeights(shards, vcfg);

  // fixture image -> pixel_values + grid (196 image tokens, 784 patches).
  vllm::multimodal::Qwen3VLProcessorConfig pcfg =
      vllm::multimodal::LoadQwen3VLProcessorConfig(
          ckpt + "/preprocessor_config.json", ckpt + "/config.json", "Qwen/Qwen3.6-27B");
  vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  const std::vector<uint8_t> rgb = ReadU8(ImgFix() + "/image_rgb_uint8_448x448x3.bin");
  REQUIRE(rgb.size() == 448u * 448u * 3u);
  const vllm::multimodal::ImageKwargs img = proc.ProcessImage(rgb.data(), 448, 448);
  const std::array<int64_t, 3> grid = img.image_grid_thw;
  REQUIRE(grid[0] == 1);
  REQUIRE(grid[1] == 28);
  REQUIRE(grid[2] == 28);

  using clock = std::chrono::steady_clock;
  const int kReps = 5;  // rep0 = cold, discarded

  // --- BASELINE: host-weights overload (marshal + compute) per call -----------
  // The wrapper prints the prepare(marshal) vs forward(compute) split under
  // VLLM_MM_TOWER_PROFILE; here we also time the whole call for the band.
  std::vector<double> base_ms;
  std::vector<float> tower_ref;
  for (int r = 0; r < kReps; ++r) {
    const auto t0 = clock::now();
    std::vector<float> tower =
        vllm::multimodal::Qwen3VLVisionForward(img.pixel_values_bf16, grid, vw, vcfg, *gpu);
    const auto t1 = clock::now();
    if (r == 0) tower_ref = tower;
    if (r > 0) base_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  // --- FAST: prepare ONCE (amortized model-load), forward per image -----------
  const auto p0 = clock::now();
  const auto dw = vllm::multimodal::PrepareVisionDeviceWeights(vw, vcfg, *gpu);
  const auto p1 = clock::now();
  const double prepare_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();

  // Same-binary attention A/B: warp AttentionDenseFast (VT_QWEN3VL_ATTN_WARP=1) vs the
  // default flash-tiled AttentionDenseFlash. Both use resident weights. The flash
  // kernel is byte-identical to the warp kernel by construction (spec §16), so the
  // tower output must match to the bit — asserted below.
  auto run_arm = [&](const char* warp_env) {
    if (warp_env != nullptr) {
      vllm::support::test::SetEnvOrThrow("VT_QWEN3VL_ATTN_WARP", warp_env);
    } else {
      vllm::support::test::UnsetEnvOrThrow("VT_QWEN3VL_ATTN_WARP");
    }
    std::vector<double> ms;
    std::vector<float> out;
    for (int r = 0; r < kReps; ++r) {
      const auto t0 = clock::now();
      std::vector<float> tower =
          vllm::multimodal::Qwen3VLVisionForward(img.pixel_values_bf16, grid, *dw, vcfg, *gpu);
      const auto t1 = clock::now();
      if (r == 0) out = tower;
      if (r > 0) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return std::make_pair(Median(ms), out);
  };

  const auto [warp_med, tower_warp] = run_arm("1");     // warp AttentionDenseFast
  const auto [flash_med, tower_flash] = run_arm(nullptr);  // flash AttentionDenseFlash (default)

  // FAST(warp) output BIT-identical to the baseline host path.
  REQUIRE(tower_warp.size() == tower_ref.size());
  size_t mism = 0;
  for (size_t i = 0; i < tower_warp.size(); ++i)
    if (tower_warp[i] != tower_ref[i]) ++mism;
  MESSAGE("warp vs baseline tower mismatches: " << mism << "/" << tower_warp.size());
  CHECK(mism == 0);

  // Flash output BIT-identical to warp (byte-exact lever contract).
  REQUIRE(tower_flash.size() == tower_warp.size());
  size_t mism_fw = 0;
  for (size_t i = 0; i < tower_flash.size(); ++i)
    if (tower_flash[i] != tower_warp[i]) ++mism_fw;
  MESSAGE("flash vs warp tower mismatches: " << mism_fw << "/" << tower_flash.size());
  CHECK(mism_fw == 0);

  const double base_med = Median(base_ms);
  MESSAGE("=== Qwen3.6-27B vision-tower A-B (448x448 image, 784 patches, 196 tokens) ===");
  MESSAGE("BASELINE (host overload: marshal+compute per call) median = " << base_med << " ms");
  MESSAGE("FAST warp  (resident, AttentionDenseFast)          median = " << warp_med << " ms");
  MESSAGE("FAST flash (resident, AttentionDenseFlash, ships)  median = " << flash_med << " ms");
  MESSAGE("one-time PrepareVisionDeviceWeights (marshal)             = " << prepare_ms << " ms");
  MESSAGE("=> per-image tower warp->flash speedup " << (warp_med / flash_med)
          << "x  (denominator vLLM ~250 ms)");
}
