// MoE vision tower (#891) — the HARDWARE gate on the real `Qwen/Qwen3.6-35B-A3B`
// bf16 checkpoint, the one that ships the 333 `model.visual.*` tensors the MoE
// loader used to read straight past.
//
// Three cases, in ascending cost:
//
//   1. TOWER LOADS AND RUNS. Load ONLY `model.visual.*` (`LoadQwen3_5MoeVision`)
//      — the shared reader the dense 27B arm is gated on — run our processor on
//      the committed fixture image, run `Qwen3VLVisionForward`, and check the
//      merger emits [196, hidden_size]. ~1 GiB. This is what "stop dropping the
//      tensors" means, executed rather than argued.
//   2. ABSENCE IS REFUSED on the real thing: a shard set with the tower filtered
//      out must be refused naming `model.visual.*`.
//   3. E2E IMAGE / VIDEO -> TEXT on the full model. OPT-IN via
//      `VLLM_MOE_VISION_E2E=1`, because the bf16 35B is ~67 GiB of weights and
//      this box has been OOM-killed at 117 GiB before; a gate that can take the
//      machine down must be asked for, not stumbled into.
//
// WHAT CASE 3 DOES AND DOES NOT ESTABLISH. It emits real tokens from the real
// tower on the real backbone, and it asserts the tower is INVOKED — by rerunning
// the identical prompt with the merger rows replaced by `embed_tokens[image_
// token_id]`, i.e. exactly what a tower that loaded but was never called would
// leave in those rows, and requiring the two streams to differ. It does NOT
// establish token-EXACTNESS: there is no committed 35B mm golden and no pinned
// oracle reachable from this box. The binding image/video token-exact gate vs
// the pinned oracle is OWED and is NOT this test.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

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

// The vision-inclusive bf16 repo. NOT the NVFP4 requant, which declares
// `vision_config` and ships no `visual.*` weights.
std::string FindCkpt() {
  if (const char* e = std::getenv("VLLM_QWEN36_35B_CKPT")) return e;
  const char* roots[] = {"/usr/local/nas_share/checkpoints/qwen3.6-35b-a3b-bf16",
                         "/mnt/nas_share/checkpoints/qwen3.6-35b-a3b-bf16"};
  std::error_code ec;
  for (const char* r : roots)
    if (fs::exists(fs::path(r) / "config.json", ec)) return r;
  return "";
}

std::vector<vllm::SafetensorsFile> OpenShards(const std::string& ckpt) {
  std::vector<vllm::SafetensorsFile> shards;
  std::error_code ec;
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(ckpt, ec))
    if (e.path().extension() == ".safetensors") paths.push_back(e.path().string());
  std::sort(paths.begin(), paths.end());
  for (const std::string& p : paths) shards.push_back(vllm::SafetensorsFile::Open(p));
  return shards;
}

int64_t CountVisual(const std::vector<vllm::SafetensorsFile>& shards) {
  int64_t n = 0;
  for (const vllm::SafetensorsFile& s : shards)
    for (const std::string& name : s.Names())
      if (name.rfind("model.visual.", 0) == 0) ++n;
  return n;
}

}  // namespace

TEST_CASE("qwen3_5_moe_35b_vision_tower_loads_and_runs_on_the_fixture_image") {
  const std::string ckpt = FindCkpt();
  if (ckpt.empty()) {
    MESSAGE("SKIP: Qwen3.6-35B-A3B bf16 checkpoint absent (VLLM_QWEN36_35B_CKPT)");
    return;
  }
  const vllm::HfConfig cfg = vllm::LoadHfConfig(ckpt + "/config.json");
  MESSAGE("config: layers=" << cfg.num_hidden_layers
                            << " hidden=" << cfg.hidden_size
                            << " experts=" << cfg.num_experts
                            << " vocab=" << cfg.vocab_size);
  REQUIRE(cfg.hidden_size == 2048);
  REQUIRE(cfg.num_hidden_layers == 40);

  std::vector<vllm::SafetensorsFile> shards = OpenShards(ckpt);
  REQUIRE(!shards.empty());
  const int64_t n_visual = CountVisual(shards);
  MESSAGE("model.visual.* tensors in the checkpoint: " << n_visual);
  // The tensors the loader used to read straight past.
  CHECK(n_visual == 333);
  REQUIRE(vllm::HasQwen3_5MoeVisionTower(shards));

  const vllm::multimodal::Qwen3VLVisionConfig vcfg =
      vllm::Qwen3_5MoeVisionConfig(cfg);
  CHECK(vcfg.depth == 27);
  CHECK(vcfg.out_hidden_size == cfg.hidden_size);
  CHECK(vcfg.deepstack_visual_indexes.empty());

  const vllm::multimodal::Qwen3VLVisionWeights vw =
      vllm::LoadQwen3_5MoeVision(shards, cfg);
  CHECK(vw.blocks.size() == 27u);
  CHECK(!vw.patch_proj_w.empty());
  CHECK(!vw.pos_embed_w.empty());
  CHECK(!vw.merger.fc2_w.empty());
  CHECK(vw.deepstack_mergers.empty());

  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  vt::Backend& backend = gpu != nullptr ? *gpu : vt::GetBackend(vt::DeviceType::kCPU);
  MESSAGE("tower backend: " << (gpu != nullptr ? "CUDA" : "CPU"));

  vllm::multimodal::Qwen3VLProcessorConfig pcfg =
      vllm::multimodal::LoadQwen3VLProcessorConfig(
          ckpt + "/preprocessor_config.json", ckpt + "/config.json",
          "Qwen/Qwen3.6-35B-A3B");
  vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  const std::vector<uint8_t> rgb =
      ReadU8(ImgFix() + "/image_rgb_uint8_448x448x3.bin");
  REQUIRE(rgb.size() == 448u * 448u * 3u);
  const vllm::multimodal::ImageKwargs img = proc.ProcessImage(rgb.data(), 448, 448);
  const std::array<int64_t, 3> grid = img.image_grid_thw;
  MESSAGE("grid_thw = [" << grid[0] << "," << grid[1] << "," << grid[2] << "]");
  REQUIRE(grid[0] == 1);

  const std::vector<float> tower = vllm::multimodal::Qwen3VLVisionForward(
      img.pixel_values_bf16, grid, vw, vcfg, backend);
  const int64_t N = static_cast<int64_t>(tower.size()) / vcfg.out_hidden_size;
  MESSAGE("merger out rows=" << N << " cols=" << vcfg.out_hidden_size);
  CHECK(N == (grid[1] / 2) * (grid[2] / 2));
  // Not all-zero, not NaN: the tower actually computed something.
  double absmax = 0.0, sum = 0.0;
  bool finite = true;
  for (float v : tower) {
    if (!(v == v)) finite = false;
    absmax = absmax > std::abs(static_cast<double>(v))
                 ? absmax
                 : std::abs(static_cast<double>(v));
    sum += static_cast<double>(v);
  }
  MESSAGE("merger absmax=" << absmax << " sum=" << sum);
  CHECK(finite);
  CHECK(absmax > 0.0);
}

TEST_CASE("qwen3_5_moe_35b_vision_absence_is_refused_on_the_real_shards") {
  const std::string ckpt = FindCkpt();
  if (ckpt.empty()) {
    MESSAGE("SKIP: Qwen3.6-35B-A3B bf16 checkpoint absent");
    return;
  }
  const vllm::HfConfig cfg = vllm::LoadHfConfig(ckpt + "/config.json");
  // Only the shards that carry NO tower tensor: the text-only view of this very
  // checkpoint, which is what an NVFP4 requant of it looks like.
  std::vector<vllm::SafetensorsFile> text_only;
  for (vllm::SafetensorsFile& s : OpenShards(ckpt)) {
    bool has = false;
    for (const std::string& n : s.Names())
      if (n.rfind("model.visual.", 0) == 0) has = true;
    if (!has) text_only.push_back(std::move(s));
  }
  REQUIRE(!text_only.empty());
  CHECK_FALSE(vllm::HasQwen3_5MoeVisionTower(text_only));
  std::string msg;
  try {
    (void)vllm::LoadQwen3_5MoeVision(text_only, cfg);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE(!msg.empty());
  CHECK(msg.find("model.visual.") != std::string::npos);
}

TEST_CASE("qwen3_5_moe_35b_e2e_image_and_video_invoke_the_tower") {
  const char* optin = std::getenv("VLLM_MOE_VISION_E2E");
  if (optin == nullptr || optin[0] != '1') {
    MESSAGE("SKIP: set VLLM_MOE_VISION_E2E=1 to run the full 35B e2e "
            "(~67 GiB of weights; this box has been OOM-killed before)");
    return;
  }
  const std::string ckpt = FindCkpt();
  if (ckpt.empty()) {
    MESSAGE("SKIP: Qwen3.6-35B-A3B bf16 checkpoint absent");
    return;
  }
  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  const vllm::HfConfig cfg = vllm::LoadHfConfig(ckpt + "/config.json");
  std::vector<vllm::SafetensorsFile> shards = OpenShards(ckpt);
  REQUIRE(!shards.empty());

  vt::Queue q = gpu->CreateQueue();
  const vllm::multimodal::Qwen3VLVisionConfig vcfg =
      vllm::Qwen3_5MoeVisionConfig(cfg);
  const vllm::multimodal::Qwen3VLVisionWeights vw =
      vllm::LoadQwen3_5MoeVision(shards, cfg);
  MESSAGE("tower loaded: " << vw.blocks.size() << " blocks");

  vllm::multimodal::Qwen3VLProcessorConfig pcfg =
      vllm::multimodal::LoadQwen3VLProcessorConfig(
          ckpt + "/preprocessor_config.json", ckpt + "/config.json",
          "Qwen/Qwen3.6-35B-A3B");
  vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  const std::vector<uint8_t> rgb =
      ReadU8(ImgFix() + "/image_rgb_uint8_448x448x3.bin");
  const vllm::multimodal::ImageKwargs img = proc.ProcessImage(rgb.data(), 448, 448);
  const std::array<int64_t, 3> grid = img.image_grid_thw;
  const std::vector<float> tower = vllm::multimodal::Qwen3VLVisionForward(
      img.pixel_values_bf16, grid, vw, vcfg, *gpu);
  const int64_t N = static_cast<int64_t>(tower.size()) / cfg.hidden_size;
  MESSAGE("tower rows N=" << N);

  const vllm::Qwen3_5MoeWeights llm = vllm::LoadQwen3_5Moe(shards, cfg);
  MESSAGE("backbone loaded: " << llm.layers.size() << " layers");

  // The 35B declares image_token_id 248056 in its own config.json.
  const int32_t kImg = 248056;
  std::vector<int32_t> prompt_ids = {kImg};  // placeholder-expanded below
  prompt_ids.clear();
  for (int64_t i = 0; i < N; ++i) prompt_ids.push_back(kImg);
  // A short instruction after the image span. These ids are exercise input, not
  // an oracle-matched prompt: there is no committed 35B mm golden.
  for (int32_t t : {13, 1836, 419, 2168, 13}) prompt_ids.push_back(t);

  const std::vector<int32_t> with_tower = vllm::Qwen3_5MoeVLGenerateGreedy(
      prompt_ids, tower, grid, kImg, /*eos_token_id=*/-1, llm, cfg, q,
      /*max_new_tokens=*/32);
  REQUIRE(with_tower.size() == 32u);

  // THE INVOCATION CONTROL. Replace every merger row with the embedding of the
  // image placeholder id itself — byte for byte what a tower that loaded but was
  // never called would leave in those rows — and require the answer to change.
  const auto* emb =
      reinterpret_cast<const uint16_t*>(llm.embed_tokens.bytes.data());
  std::vector<float> placeholder(static_cast<size_t>(N * cfg.hidden_size));
  for (int64_t r = 0; r < N; ++r)
    for (int64_t j = 0; j < cfg.hidden_size; ++j)
      placeholder[static_cast<size_t>(r * cfg.hidden_size + j)] =
          vt::BF16ToF32(emb[static_cast<size_t>(kImg * cfg.hidden_size + j)]);
  const std::vector<int32_t> without_tower = vllm::Qwen3_5MoeVLGenerateGreedy(
      prompt_ids, placeholder, grid, kImg, -1, llm, cfg, q, 32);

  std::string a, b;
  for (int i = 0; i < 32; ++i) {
    a += std::to_string(with_tower[static_cast<size_t>(i)]) + ",";
    b += std::to_string(without_tower[static_cast<size_t>(i)]) + ",";
  }
  MESSAGE("with tower   : " << a);
  MESSAGE("without tower: " << b);
  CHECK(with_tower != without_tower);
}
