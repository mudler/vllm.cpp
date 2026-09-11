// DeepSeek-V4 Flash Vision W6 parity probe -- OUR side of the real-weight gate.
//
// Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W6, issue
// [#2411](https://github.com/mudler/vllm.cpp/issues/2411). A PROBE, NOT A
// PRODUCT: it is not in the default build, and the W6 job script adds it to a
// scratch copy of `examples/CMakeLists.txt` only.
//
// It drives the SHIPPED path and nothing hand-built:
//   LoadDeepseekV4VisionRuntime  -> the `--mmproj` arm the loader calls
//                                   (refusals, then LoadDeepSeekV4ClipMmprojArm)
//   DeepSeekV4ImageProcessor     -> the W1 processor, entered with raw RGB
//   PrepareDeepSeekV4Inputs      -> the W1 expansion that places the block
//   ModelRegistry::EncodeMm      -> the registered encode_mm hook, i.e.
//                                   EncodeMmDeepseekV4ForCausalLM: tower,
//                                   permutation and sentinel rows
// The loaded model carries NO language weights. `encode_mm` reads only the
// vision runtime and `HfConfig::hidden_size`, so the 82 GB language model is
// not needed to produce the block it would receive.
//
// The leading-pad count is placed, not forced: the placeholder is put after
// `3 - lead_pad` prompt tokens, so the processor derives
// `compress_pad = 3 - offset % 4` exactly as llama.cpp's tokenizer derives
// `lead_pad = 3 - n_past % 4` (`tools/mtmd/mtmd.cpp` at b10766).
//
// Output files, all `[int32 rows][int32 cols][f32 data]`, which is the format
// llama.cpp's `MTMD_DEBUG_EMBEDDINGS=<path>` dump writes:
//   <outdir>/ours-<tag>-block.f32  the encode_mm output, one row per block token
//   <outdir>/ours-<tag>-input.f32  the bf16 patch rows the tower consumed
//   <outdir>/ours-<tag>-vit.f32    the tower after its final RMSNorm
//   <outdir>/ours-<tag>-cells.f32  the aligner output, before the block layout
//
// usage: dsv4v-w6-probe <mmproj.gguf> <image.rgb> <height> <width> <lead_pad>
//                       <outdir> <tag>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/deepseek_v4_mm.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/multimodal/deepseek_v4_processor.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

void WriteF32(const std::string& path, int64_t rows, int64_t cols,
              const std::vector<float>& data) {
  std::ofstream f(path, std::ios::binary);
  const int32_t hdr[2] = {static_cast<int32_t>(rows), static_cast<int32_t>(cols)};
  f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!f) {
    std::fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
    std::exit(3);
  }
}

std::vector<float> Widen(const uint16_t* p, size_t n) {
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = vt::BF16ToF32(p[i]);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr,
                 "usage: %s <mmproj.gguf> <image.rgb> <height> <width> "
                 "<lead_pad> <outdir> <tag>\n",
                 argv[0]);
    return 2;
  }
  const std::string mmproj_path = argv[1];
  const std::string rgb_path = argv[2];
  const int64_t height = std::atoll(argv[3]);
  const int64_t width = std::atoll(argv[4]);
  const int lead_pad = std::atoi(argv[5]);
  const std::string outdir = argv[6];
  const std::string tag = argv[7];
  if (lead_pad < 0 || lead_pad > 3) {
    std::fprintf(stderr, "lead_pad must be 0..3\n");
    return 2;
  }

  std::ifstream in(rgb_path, std::ios::binary);
  std::vector<uint8_t> rgb((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  if (static_cast<int64_t>(rgb.size()) != height * width * 3) {
    std::fprintf(stderr, "FATAL: %s holds %zu bytes, want %lld\n",
                 rgb_path.c_str(), rgb.size(),
                 static_cast<long long>(height * width * 3));
    return 3;
  }

  const vllm::GgufFile gguf = vllm::GgufFile::Open(mmproj_path);
  vllm::HfConfig config;
  config.architectures = {"DeepseekV4ForCausalLM"};
  config.model_type = "deepseek_v4";
  config.hidden_size = 4096;  // the released text width; the arm checks it
  vllm::ModelSource source;
  source.kind = vllm::ModelSource::Kind::kGguf;
  source.mmproj = &gguf;
  source.mmproj_path = mmproj_path;
  std::unique_ptr<vllm::DeepseekV4VisionRuntime> runtime =
      vllm::LoadDeepseekV4VisionRuntime(source, config);
  if (runtime == nullptr) {
    std::fprintf(stderr, "FATAL: the --mmproj arm loaded no tower\n");
    return 3;
  }
  const vllm::multimodal::DeepSeekV4VisionConfig vcfg = runtime->config;
  std::printf(
      "vision config: patch=%lld hidden=%lld heads=%lld depth=%lld inter=%lld "
      "out=%lld r=%lld eps=%g theta=%g\n",
      static_cast<long long>(vcfg.patch_size),
      static_cast<long long>(vcfg.hidden_size),
      static_cast<long long>(vcfg.num_heads),
      static_cast<long long>(vcfg.depth),
      static_cast<long long>(vcfg.intermediate_size),
      static_cast<long long>(vcfg.output_size),
      static_cast<long long>(vcfg.downsample_ratio), vcfg.norm_epsilon,
      vcfg.rope_theta);

  vllm::multimodal::DeepSeekV4ProcessorConfig pc;
  pc.patch_size = vcfg.patch_size;
  pc.downsample_ratio = vcfg.downsample_ratio;
  pc.model_id = "dsv4v-w6-probe";
  const vllm::multimodal::DeepSeekV4ImageProcessor proc(pc);
  const std::span<const uint8_t> pixels(rgb.data(), rgb.size());
  auto kwargs = std::make_shared<vllm::multimodal::ImageKwargs>(
      proc.ProcessImage(pixels, height, width));
  std::printf("processor: grid=[%lld,%lld,%lld] patches=%lld feat=%lld\n",
              static_cast<long long>(kwargs->image_grid_thw[0]),
              static_cast<long long>(kwargs->image_grid_thw[1]),
              static_cast<long long>(kwargs->image_grid_thw[2]),
              static_cast<long long>(kwargs->num_patches),
              static_cast<long long>(kwargs->patch_feature_dim));
  const std::string hash = proc.HashImage(pixels, height, width);

  constexpr int32_t kImageToken = 7;
  std::vector<int32_t> prompt;
  for (int i = 0; i < 3 - lead_pad; ++i) prompt.push_back(100 + i);
  prompt.push_back(kImageToken);
  prompt.push_back(200);
  const vllm::multimodal::MultiModalInputs mm =
      vllm::multimodal::PrepareDeepSeekV4Inputs(prompt, kImageToken,
                                                {{kwargs, hash}}, pc);
  if (mm.mm_features.size() != 1) {
    std::fprintf(stderr, "FATAL: %zu features\n", mm.mm_features.size());
    return 3;
  }
  const vllm::multimodal::MultiModalFeatureSpec& feature = mm.mm_features[0];
  std::printf("feature: offset=%d length=%d (lead_pad asked %d)\n",
              feature.offset, feature.length, lead_pad);

  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  vllm::DeepseekV4LoadedModel model(reg, vllm::DeepseekV4Weights{},
                                    std::move(runtime));
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();

  // THE PRODUCTION HOOK.
  const vllm::MmEncoderOutput enc =
      vllm::ModelRegistry::EncodeMm(model, config, queue, feature);
  backend.Synchronize(queue);
  if (enc.embeds.dtype != vt::DType::kBF16 || enc.embeds.rank != 2) {
    std::fprintf(stderr, "FATAL: encoder output is not a bf16 matrix\n");
    return 3;
  }
  const int64_t rows = enc.embeds.shape[0];
  const int64_t cols = enc.embeds.shape[1];
  WriteF32(outdir + "/ours-" + tag + "-block.f32", rows, cols,
           Widen(enc.embeds.Ptr<uint16_t>(), static_cast<size_t>(rows * cols)));
  std::printf("block: %lld x %lld\n", static_cast<long long>(rows),
              static_cast<long long>(cols));

  // STAGES, for localisation. Same tower instance encode_mm just built, same
  // patch rows, and the capture seam W2 declares for parity gates.
  const int64_t grid_h = kwargs->image_grid_thw[1];
  const int64_t grid_w = kwargs->image_grid_thw[2];
  const int64_t patches = kwargs->num_patches;
  WriteF32(outdir + "/ours-" + tag + "-input.f32", patches,
           kwargs->patch_feature_dim,
           Widen(kwargs->pixel_values_bf16.data(),
                 kwargs->pixel_values_bf16.size()));

  vllm::multimodal::DeepSeekV4Vision& tower = model.vision_tower(backend);
  const size_t bf = vt::SizeOf(vt::DType::kBF16);
  auto alloc = [&](size_t n) { return backend.Alloc(n * bf); };
  void* patch_buf = alloc(kwargs->pixel_values_bf16.size());
  backend.Copy(queue, patch_buf, kwargs->pixel_values_bf16.data(),
               kwargs->pixel_values_bf16.size() * bf);
  const vt::Tensor patch_t = vllm::dense_attn::MakeTensor(
      patch_buf, vt::DType::kBF16, queue.device,
      {patches, kwargs->patch_feature_dim});
  const int64_t cells = vcfg.aligned_rows(grid_h, grid_w);
  void* cell_buf = alloc(static_cast<size_t>(cells * vcfg.output_size));
  vt::Tensor cell_t = vllm::dense_attn::MakeTensor(
      cell_buf, vt::DType::kBF16, queue.device, {cells, vcfg.output_size});
  void* vit_buf = alloc(static_cast<size_t>(patches * vcfg.hidden_size));
  vt::Tensor vit_t = vllm::dense_attn::MakeTensor(
      vit_buf, vt::DType::kBF16, queue.device, {patches, vcfg.hidden_size});
  vllm::multimodal::DeepSeekV4VisionCapture capture;
  capture.final_norm = &vit_t;
  tower.Forward(queue, cell_t, patch_t, grid_h, grid_w, &capture);
  backend.Synchronize(queue);
  std::vector<uint16_t> host(static_cast<size_t>(patches * vcfg.hidden_size));
  backend.Copy(queue, host.data(), vit_buf, host.size() * bf);
  backend.Synchronize(queue);
  WriteF32(outdir + "/ours-" + tag + "-vit.f32", patches, vcfg.hidden_size,
           Widen(host.data(), host.size()));
  host.assign(static_cast<size_t>(cells * vcfg.output_size), 0);
  backend.Copy(queue, host.data(), cell_buf, host.size() * bf);
  backend.Synchronize(queue);
  WriteF32(outdir + "/ours-" + tag + "-cells.f32", cells, vcfg.output_size,
           Widen(host.data(), host.size()));
  backend.Free(patch_buf);
  backend.Free(cell_buf);
  backend.Free(vit_buf);
  backend.DestroyQueue(queue);
  std::printf("PROBE_DONE tag=%s\n", tag.c_str());
  return 0;
}
