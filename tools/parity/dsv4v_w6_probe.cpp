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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
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

// THE F32 ARM (DSV4V_PROBE_F32=1). A MEASUREMENT, NEVER A PRODUCT PATH.
//
// llama.cpp's CPU clip graph keeps its residual stream, norms, RoPE, softmax
// and activations in f32 and rounds to bf16 only at each GEMM input
// (ggml-cpu.c:395-399, vec_dot_type = GGML_TYPE_BF16). Our tower carries every
// intermediate in bf16, which is the model dtype AGENTS.md requires. This arm
// runs the SAME tower with f32 weights (widened exactly from the file's bf16)
// and f32 activations, on the exact f32 pixels, so that if ours-in-f32 lands on
// the oracle the W6 gap is compute dtype and not a defect.
//
// W2 refuses `compute_dtype != bf16`, so this arm runs only in a scratch copy
// whose guard the W6 f32 job script deletes. It is never built from the tree.
int RunF32(const vllm::GgufFile& gguf, const std::vector<uint8_t>& rgb,
           int64_t height, int64_t width, int lead_pad,
           const std::string& outdir, const std::string& tag) {
  vllm::multimodal::DeepSeekV4VisionConfig cfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
  const vllm::DeepSeekV4ClipMmproj proj =
      vllm::LoadDeepSeekV4VisionFromClipMmproj(gguf, cfg);
  std::deque<std::vector<float>> store;
  auto widen = [&](const vt::Tensor& t) -> vt::Tensor {
    if (t.dtype == vt::DType::kF32) return t;
    if (t.dtype != vt::DType::kBF16 || !t.IsContiguous() || t.rank < 1 ||
        t.rank > 2) {
      std::fprintf(stderr, "FATAL: f32 arm cannot widen a weight\n");
      std::exit(3);
    }
    std::vector<float>& s = store.emplace_back(
        Widen(static_cast<const uint16_t*>(t.data),
              static_cast<size_t>(t.Numel())));
    return t.rank == 1
               ? vt::Tensor::Contiguous(s.data(), vt::DType::kF32, t.device,
                                        {t.shape[0]})
               : vt::Tensor::Contiguous(s.data(), vt::DType::kF32, t.device,
                                        {t.shape[0], t.shape[1]});
  };
  const vllm::multimodal::DeepSeekV4VisionWeights& b = proj.weights;
  vllm::multimodal::DeepSeekV4VisionWeights w;
  w.patch_weight = widen(b.patch_weight);
  w.patch_bias = widen(b.patch_bias);
  for (const auto& blk : b.blocks) {
    vllm::multimodal::DeepSeekV4VisionBlockWeights o;
    o.norm1_weight = widen(blk.norm1_weight);
    o.qkv_weight = widen(blk.qkv_weight);
    o.qkv_bias = widen(blk.qkv_bias);
    o.out_weight = widen(blk.out_weight);
    o.out_bias = widen(blk.out_bias);
    o.norm2_weight = widen(blk.norm2_weight);
    o.mlp_w1_weight = widen(blk.mlp_w1_weight);
    o.mlp_w2_weight = widen(blk.mlp_w2_weight);
    w.blocks.push_back(o);
  }
  w.final_norm_weight = widen(b.final_norm_weight);
  w.aligner_w1_weight = widen(b.aligner_w1_weight);
  w.aligner_w1_bias = widen(b.aligner_w1_bias);
  w.aligner_w2_weight = widen(b.aligner_w2_weight);
  w.aligner_w2_bias = widen(b.aligner_w2_bias);
  cfg.compute_dtype = vt::DType::kF32;

  const int64_t P = cfg.patch_size;
  if (height % P != 0 || width % P != 0) {
    std::fprintf(stderr, "FATAL: f32 arm takes an identity-size image only\n");
    return 3;
  }
  const int64_t gh = height / P, gw = width / P, patches = gh * gw;
  const int64_t feat = 3 * P * P;
  // The processor's own formula (deepseek_v4_processor.cpp ProcessImage),
  // without the final bf16 narrowing.
  std::vector<float> px(static_cast<size_t>(patches * feat));
  for (int64_t vh = 0; vh < gh; ++vh)
    for (int64_t vw = 0; vw < gw; ++vw)
      for (int64_t c = 0; c < 3; ++c)
        for (int64_t dy = 0; dy < P; ++dy)
          for (int64_t dx = 0; dx < P; ++dx) {
            const uint8_t raw =
                rgb[static_cast<size_t>(((vh * P + dy) * width + vw * P + dx) * 3 + c)];
            px[static_cast<size_t>((vh * gw + vw) * feat + (c * P + dy) * P + dx)] =
                ((static_cast<float>(raw) / 255.0f) - 0.5f) / 0.5f;
          }
  WriteF32(outdir + "/ours-" + tag + "-input.f32", patches, feat, px);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  vllm::multimodal::DeepSeekV4Vision tower(backend, cfg, w);
  const int64_t cells = cfg.aligned_rows(gh, gw);
  std::vector<float> cell_host(static_cast<size_t>(cells * cfg.output_size));
  std::vector<float> vit_host(static_cast<size_t>(patches * cfg.hidden_size));
  const vt::Tensor patch_t = vt::Tensor::Contiguous(
      px.data(), vt::DType::kF32, queue.device, {patches, feat});
  vt::Tensor cell_t = vt::Tensor::Contiguous(
      cell_host.data(), vt::DType::kF32, queue.device, {cells, cfg.output_size});
  vt::Tensor vit_t = vt::Tensor::Contiguous(
      vit_host.data(), vt::DType::kF32, queue.device, {patches, cfg.hidden_size});
  vllm::multimodal::DeepSeekV4VisionCapture capture;
  capture.final_norm = &vit_t;
  tower.Forward(queue, cell_t, patch_t, gh, gw, &capture);
  backend.Synchronize(queue);
  WriteF32(outdir + "/ours-" + tag + "-vit.f32", patches, cfg.hidden_size,
           vit_host);
  WriteF32(outdir + "/ours-" + tag + "-cells.f32", cells, cfg.output_size,
           cell_host);

  // The block, from the same layout function encode_mm uses, with the f32
  // sentinels un-narrowed.
  const int64_t r = cfg.downsample_ratio;
  const vllm::multimodal::DeepSeekV4ImageBlock block =
      vllm::multimodal::BuildDeepSeekV4ImageBlock((gh + r - 1) / r,
                                                  (gw + r - 1) / r,
                                                  3 - lead_pad);
  const int64_t ow = cfg.output_size;
  std::vector<float> rows(block.types.size() * static_cast<size_t>(ow));
  size_t taken = 0;
  for (size_t i = 0; i < block.types.size(); ++i) {
    const float* src = nullptr;
    switch (block.types[i]) {
      case vllm::multimodal::kImage:
        src = cell_host.data() + block.permutation[taken++] * ow;
        break;
      case vllm::multimodal::kImageStart: src = proj.image_start.data(); break;
      case vllm::multimodal::kImageEnd: src = proj.image_end.data(); break;
      case vllm::multimodal::kImagePad: src = proj.image_pad.data(); break;
      case vllm::multimodal::kImageNewLine: src = proj.image_newline.data(); break;
      default:
        std::fprintf(stderr, "FATAL: unknown block token type\n");
        return 3;
    }
    std::copy(src, src + ow, rows.begin() + static_cast<std::ptrdiff_t>(i * ow));
  }
  WriteF32(outdir + "/ours-" + tag + "-block.f32",
           static_cast<int64_t>(block.types.size()), ow, rows);
  backend.DestroyQueue(queue);
  std::printf("PROBE_F32_DONE tag=%s block=%zu\n", tag.c_str(),
              block.types.size());
  return 0;
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
  if (const char* f32 = std::getenv("DSV4V_PROBE_F32"); f32 && f32[0] == '1') {
    return RunF32(gguf, rgb, height, width, lead_pad, outdir, tag);
  }
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
