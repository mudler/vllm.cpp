// DeepSeek-V4-Flash-Vision — the multimodal MODEL seam (row
// `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W4, issue
// [#2411](https://github.com/mudler/vllm.cpp/issues/2411)).
//
// THIS FILE IS THE PRODUCTION CALL SITE the row's `## Owed` has been naming
// since W2. Every earlier wave landed a capability nothing arrived at:
//
//   W1  `PrepareDeepSeekV4Inputs` / `BuildDeepSeekV4ImageBlock`  -> read here
//   W2  `multimodal::DeepSeekV4Vision`                           -> run here
//   W3A `LoadDeepSeekV4ClipMmprojArm`                            -> called here
//
// It implements the two model hooks `GPUModelRunner` dispatches through, which
// are upstream's `SupportsMultiModal.embed_multimodal` and
// `SupportsMultiModal.embed_input_ids`. The runner is generic over them; a
// registration that leaves either null keeps its whole multimodal arm off.
//
// WHERE THE BLOCK LAYOUT COMES FROM, and why it is not carried. `encode_mm`
// receives one `MultiModalFeatureSpec`, whose `offset` is the expanded prompt
// position the processor passed to `BuildDeepSeekV4ImageBlock` as its
// `start_position` (`deepseek_v4_processor.cpp`, the `PrepareDeepSeekV4Inputs`
// loop). So the hook recomputes the block from the SAME function and the SAME
// two inputs the processor used, and asserts the length it gets back is the
// length the processor recorded. Carrying the type vector on `ImageKwargs`
// instead would put a second copy of the layout in the request, free to
// disagree with the identifiers already written into the prompt.
#include "vllm/model_executor/models/deepseek_v4_mm.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // MakeTensor
#include "vllm/model_executor/models/interfaces.h"        // SkipTowerForModalities
#include "vllm/multimodal/deepseek_v4_processor.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// One owned device allocation, freed when the holder drops.
std::shared_ptr<void> DeviceBuffer(vt::Backend& backend, size_t bytes) {
  void* p = backend.Alloc(bytes);
  return std::shared_ptr<void>(p, [&backend](void* q) { backend.Free(q); });
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace

const DeepseekV4VisionRuntime& DeepseekV4LoadedModel::vision() const {
  VT_CHECK(vision_ != nullptr,
           "DeepseekV4ForCausalLM: this load carries no vision tower. A "
           "DeepSeek-V4 TEXT checkpoint has none, and a vision checkpoint needs "
           "its `deepseek4v` projector named with --mmproj "
           "(.agents/specs/deepseek-v4-flash-vision.md, issue #2411)");
  return *vision_;
}

multimodal::DeepSeekV4Vision& DeepseekV4LoadedModel::vision_tower(
    vt::Backend& backend) {
  VT_CHECK(vision_ != nullptr,
           "DeepseekV4ForCausalLM: asked for the vision tower on a load that "
           "carries no `deepseek4v` projector. Refused by name rather than "
           "answered with an empty encoder output, which a runner would splice "
           "into the prompt as zeros and answer fluently from");
  if (vision_->tower == nullptr) {
    // Built on FIRST USE and against the queue's own backend. The weight loader
    // has no queue in hand, which is the same reason the EXL3 tower is staged
    // from the forward rather than from the loader.
    vision_->tower = std::make_unique<multimodal::DeepSeekV4Vision>(
        backend, vision_->config, vision_->projector.weights);
  }
  return *vision_->tower;
}

std::unique_ptr<DeepseekV4VisionRuntime> LoadDeepseekV4VisionRuntime(
    const ModelSource& source, const HfConfig& config) {
  if (source.mmproj == nullptr) return nullptr;
  // A projector of another family is not this architecture's to open. It is not
  // refused here either: `--mmproj` still means the Qwen3-VL arm in
  // `model_loader.cpp`, and that arm sets no `ModelSource::mmproj`, so reaching
  // this line with a foreign type means a caller built the source by hand.
  VT_CHECK(ClipProjectorType(*source.mmproj) == kClipProjectorDeepSeekV4,
           "--mmproj: '" + source.mmproj_path + "' has clip.projector_type '" +
               ClipProjectorType(*source.mmproj) +
               "', and DeepseekV4ForCausalLM reads '" +
               std::string(kClipProjectorDeepSeekV4) + "' projectors only");
  // #607 L3: the engine's multimodal limits decide whether the tower's tensors
  // are read at all, mirroring `interfaces.py:288-293`. `--language-model-only`
  // sets every limit to zero, and a load that then paid for the projector would
  // be the exact defect that wave closed for the other three towers.
  //
  // ONLY THE READ IS CONDITIONAL. A caller that named a projector this build
  // cannot load is still refused by name below, because the refusal runs inside
  // the arm and the arm is what a zero limit skips.
  if (SkipTowerForModalities(source.multimodal, {"image"})) return nullptr;

  auto runtime = std::make_unique<DeepseekV4VisionRuntime>();
  runtime->projector = LoadDeepSeekV4ClipMmprojArm(
      *source.mmproj, source.mmproj_path, &runtime->config);
  // THE ALIGNER LANDS IN THE TEXT HIDDEN SPACE. A projector whose output width
  // is not the language model's is one whose rows cannot be scattered into the
  // prompt at all, and the failure without this line is a shape error deep
  // inside the merge rather than a message naming the two files.
  VT_CHECK(runtime->config.output_size == config.hidden_size,
           "--mmproj: '" + source.mmproj_path + "' projects to " +
               std::to_string(runtime->config.output_size) +
               "-wide rows and this language model is " +
               std::to_string(config.hidden_size) +
               " wide. The two files are not a pair");
  return runtime;
}

MmEncoderOutput EncodeMmDeepseekV4ForCausalLM(
    LoadedModel& model, const HfConfig& config, vt::Queue& queue,
    const multimodal::MultiModalFeatureSpec& item) {
  auto& ds = ModelAs<DeepseekV4LoadedModel>(model, "DeepseekV4ForCausalLM");
  // IMAGE ONLY, and every other modality is refused by name rather than served
  // from the image path. The pinned encoder emits image content blocks alone
  // (`encoding/encoding_dsv4.py`), so there is no audio or video arm to owe.
  VT_CHECK(item.modality == "image",
           "DeepseekV4ForCausalLM encoder: modality '" + item.modality +
               "' is not part of this architecture. DeepSeek-V4-Flash-Vision "
               "is an image-only model (.agents/specs/"
               "deepseek-v4-flash-vision.md)");
  VT_CHECK(item.data != nullptr && !item.data->empty(),
           "DeepseekV4ForCausalLM encoder: the multimodal item carries no "
           "processed image features (MultiModalFeatureSpec::data)");

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  multimodal::DeepSeekV4Vision& tower = ds.vision_tower(backend);
  const multimodal::DeepSeekV4VisionConfig& cfg = tower.config();

  const multimodal::ImageKwargs& image = *item.data;
  VT_CHECK(image.image_grid_thw[0] == 1,
           "DeepseekV4ForCausalLM encoder: grid_t is " +
               std::to_string(image.image_grid_thw[0]) +
               " and this architecture has no temporal axis");
  const int64_t height = image.image_grid_thw[1];
  const int64_t width = image.image_grid_thw[2];
  VT_CHECK(height > 0 && width > 0 && image.num_patches == height * width,
           "DeepseekV4ForCausalLM encoder: the item declares " +
               std::to_string(image.num_patches) + " patches for a " +
               std::to_string(height) + "x" + std::to_string(width) + " grid");
  VT_CHECK(image.patch_feature_dim == cfg.patch_dim(),
           "DeepseekV4ForCausalLM encoder: the item's patch feature width is " +
               std::to_string(image.patch_feature_dim) +
               " and this projector's patch is " +
               std::to_string(cfg.patch_size) + ", so it wants " +
               std::to_string(cfg.patch_dim()));
  const int64_t values = image.num_patches * image.patch_feature_dim;
  VT_CHECK(static_cast<int64_t>(image.pixel_values_bf16.size()) == values,
           "DeepseekV4ForCausalLM encoder: the item holds " +
               std::to_string(image.pixel_values_bf16.size()) +
               " BF16 values for a shape that needs " + std::to_string(values));
  VT_CHECK(cfg.output_size == config.hidden_size,
           "DeepseekV4ForCausalLM encoder: the aligner emits " +
               std::to_string(cfg.output_size) +
               "-wide rows and the text tower is " +
               std::to_string(config.hidden_size) + " wide");

  const size_t patch_bytes =
      static_cast<size_t>(values) * vt::SizeOf(vt::DType::kBF16);
  std::shared_ptr<void> patch_buf = DeviceBuffer(backend, patch_bytes);
  backend.Copy(queue, patch_buf.get(), image.pixel_values_bf16.data(),
               patch_bytes);
  const vt::Tensor patches = dense_attn::MakeTensor(
      patch_buf.get(), vt::DType::kBF16, queue.device,
      {image.num_patches, image.patch_feature_dim});

  const int64_t aligned_rows = cfg.aligned_rows(height, width);
  const size_t aligned_bytes = static_cast<size_t>(aligned_rows) *
                               static_cast<size_t>(cfg.output_size) *
                               vt::SizeOf(vt::DType::kBF16);
  std::shared_ptr<void> aligned_buf = DeviceBuffer(backend, aligned_bytes);
  vt::Tensor aligned =
      dense_attn::MakeTensor(aligned_buf.get(), vt::DType::kBF16, queue.device,
                             {aligned_rows, cfg.output_size});
  // THE TOWER. Before this call nothing on this row ran on a served request.
  tower.Forward(queue, aligned, patches, height, width);
  backend.Synchronize(queue);

  std::vector<uint16_t> cells(static_cast<size_t>(aligned_rows) *
                              static_cast<size_t>(cfg.output_size));
  backend.Copy(queue, cells.data(), aligned_buf.get(), aligned_bytes);
  backend.Synchronize(queue);

  // The block the processor wrote, recomputed from its own two inputs. See the
  // file header for why it is recomputed rather than carried.
  const int64_t n_llm_h = CeilDiv(height, cfg.downsample_ratio);
  const int64_t n_llm_w = CeilDiv(width, cfg.downsample_ratio);
  const multimodal::DeepSeekV4ImageBlock block =
      multimodal::BuildDeepSeekV4ImageBlock(n_llm_h, n_llm_w, item.offset);
  VT_CHECK(static_cast<int64_t>(block.types.size()) ==
               static_cast<int64_t>(item.length),
           "DeepseekV4ForCausalLM encoder: the image block for a " +
               std::to_string(n_llm_h) + "x" + std::to_string(n_llm_w) +
               " grid at prompt offset " + std::to_string(item.offset) +
               " is " + std::to_string(block.types.size()) +
               " tokens, and the processor recorded a span of " +
               std::to_string(item.length) +
               ". The encoder and the expanded prompt disagree about the "
               "layout, and a masked scatter would splice the wrong rows");
  VT_CHECK(static_cast<int64_t>(block.permutation.size()) == aligned_rows,
           "DeepseekV4ForCausalLM encoder: the block names " +
               std::to_string(block.permutation.size()) +
               " image cells and the aligner produced " +
               std::to_string(aligned_rows));

  // ONE ROW PER SENTINEL TOKEN. A marker takes its learned vector; an image
  // token takes the aligner cell the PERMUTATION names, which is the row-pair
  // reorder `build_image_block` applies.
  const int64_t out_width = cfg.output_size;
  std::vector<uint16_t> rows(static_cast<size_t>(item.length) *
                             static_cast<size_t>(out_width));
  const DeepSeekV4ClipMmproj& proj = ds.vision().projector;
  size_t taken = 0;
  for (size_t i = 0; i < block.types.size(); ++i) {
    uint16_t* dst = rows.data() + i * static_cast<size_t>(out_width);
    if (block.types[i] == multimodal::kImage) {
      const int64_t cell = block.permutation[taken++];
      std::memcpy(dst, cells.data() + static_cast<size_t>(cell * out_width),
                  static_cast<size_t>(out_width) * sizeof(uint16_t));
      continue;
    }
    // The four learned vectors are f32 in the projector, which is the dtype the
    // file holds and the dtype llama.cpp concatenates them at. They narrow to
    // the model dtype HERE, at the one point where they join a bf16 residual
    // stream, so nothing widens the stream to carry them.
    const std::vector<float>* src = nullptr;
    switch (block.types[i]) {
      case multimodal::kImageStart: src = &proj.image_start; break;
      case multimodal::kImageEnd: src = &proj.image_end; break;
      case multimodal::kImagePad: src = &proj.image_pad; break;
      case multimodal::kImageNewLine: src = &proj.image_newline; break;
      default:
        VT_CHECK(false, "DeepseekV4ForCausalLM encoder: image block token type " +
                            std::to_string(block.types[i]) + " is not a "
                            "DeepSeekV4ImageTokenType this build knows");
    }
    VT_CHECK(static_cast<int64_t>(src->size()) == out_width,
             "DeepseekV4ForCausalLM encoder: a learned sentinel vector is " +
                 std::to_string(src->size()) + " wide and the aligner is " +
                 std::to_string(out_width));
    for (int64_t c = 0; c < out_width; ++c) {
      dst[c] = vt::F32ToBF16((*src)[static_cast<size_t>(c)]);
    }
  }

  const size_t out_bytes = rows.size() * vt::SizeOf(vt::DType::kBF16);
  std::shared_ptr<void> out_buf = DeviceBuffer(backend, out_bytes);
  backend.Copy(queue, out_buf.get(), rows.data(), out_bytes);
  backend.Synchronize(queue);
  MmEncoderOutput out;
  out.embeds = dense_attn::MakeTensor(out_buf.get(), vt::DType::kBF16,
                                      queue.device,
                                      {static_cast<int64_t>(item.length),
                                       out_width});
  out.storage = std::move(out_buf);
  return out;
}

MmForwardBuffers EmbedMmDeepseekV4ForCausalLM(LoadedModel& model,
                                              const HfConfig& config,
                                              vt::Queue& queue,
                                              const MmEmbedInputs& inputs) {
  auto& ds = ModelAs<DeepseekV4LoadedModel>(model, "DeepseekV4ForCausalLM");
  const DeepseekV4Weights& weights = ds.weights();
  VT_CHECK(weights.has_host_weights,
           "DeepseekV4ForCausalLM embed: the small f32 host tower carries the "
           "embedding table and this load has none");
  VT_CHECK(inputs.token_ids != nullptr && inputs.is_mm_embed != nullptr &&
               inputs.mm_embeds != nullptr,
           "DeepseekV4ForCausalLM embed: the step is missing one of "
           "`token_ids`, `is_mm_embed` or `mm_embeds`");

  const int64_t hidden = weights.params.hidden_size;
  const int64_t vocab = weights.params.vocab_size;
  VT_CHECK(hidden == config.hidden_size,
           "DeepseekV4ForCausalLM embed: the loaded tower is " +
               std::to_string(hidden) + " wide and the config says " +
               std::to_string(config.hidden_size));
  std::vector<int32_t> ids = *inputs.token_ids;
  const int64_t tokens = static_cast<int64_t>(ids.size());
  VT_CHECK(static_cast<int64_t>(inputs.is_mm_embed->size()) == tokens,
           "DeepseekV4ForCausalLM embed: the mask is " +
               std::to_string(inputs.is_mm_embed->size()) +
               " long and the step has " + std::to_string(tokens) + " tokens");

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  // ENG-MM-EMBED-DEVICE-IDS (#2730): TAKE the device identifiers when the
  // asynchronous runner says the host vector is stale. Its combine splices each
  // decode row's sampled token into the DEVICE buffer and never writes it back,
  // and `token_ids_cpu` is zero-initialised, so a hook that embedded the host
  // vector alone would build every decode row of an image request from token id
  // 0 -- at rc=0, with plausible output. This is a HOST gather, so the resolve
  // is a copy down rather than `detail::ApplyDeviceTokenIds`'s device splice.
  if (inputs.device_token_ids != nullptr) {
    backend.Copy(queue, ids.data(), inputs.device_token_ids,
                 static_cast<size_t>(tokens) * sizeof(int32_t));
    backend.Synchronize(queue);
  }

  std::vector<uint16_t> merged(static_cast<size_t>(tokens) *
                               static_cast<size_t>(hidden), 0);
  int64_t masked = 0;
  for (int64_t t = 0; t < tokens; ++t) {
    if ((*inputs.is_mm_embed)[static_cast<size_t>(t)] != 0) {
      // A masked row EMBEDS TO ZERO and the merge replaces it. It is never
      // looked up: the expanded prompt spells it `vocab_size + type`, which no
      // embedding table has a row for.
      ++masked;
      continue;
    }
    const int64_t id = ids[static_cast<size_t>(t)];
    VT_CHECK(id >= 0 && id < vocab,
             "DeepseekV4ForCausalLM embed: token id " + std::to_string(id) +
                 " at position " + std::to_string(t) +
                 " is outside the vocabulary of " + std::to_string(vocab) +
                 " and is not marked as a multimodal placeholder");
    const float* row = weights.host.embed.data() + id * hidden;
    uint16_t* dst = merged.data() + t * hidden;
    for (int64_t h = 0; h < hidden; ++h) dst[h] = vt::F32ToBF16(row[h]);
  }

  int64_t supplied = 0;
  for (const vt::Tensor& slice : *inputs.mm_embeds) {
    VT_CHECK(slice.rank == 2 && slice.shape[1] == hidden,
             "DeepseekV4ForCausalLM embed: an encoder slice is not "
             "[rows, " + std::to_string(hidden) + "]");
    VT_CHECK(slice.dtype == vt::DType::kBF16,
             "DeepseekV4ForCausalLM embed: an encoder slice is not BF16, which "
             "is the model dtype every row of this stream is stored in");
    supplied += slice.shape[0];
  }
  // THE BALANCE. `gather_mm_embeddings` marks exactly one masked position per
  // gathered row, so a disagreement means the encoder and the mask were built
  // from different layouts and the scatter below would shift every row after
  // the first missing one.
  VT_CHECK(supplied == masked,
           "DeepseekV4ForCausalLM embed: the step gathered " +
               std::to_string(supplied) + " encoder rows for " +
               std::to_string(masked) + " masked positions");

  int64_t next = 0;
  std::vector<int64_t> masked_at;
  masked_at.reserve(static_cast<size_t>(masked));
  for (int64_t t = 0; t < tokens; ++t) {
    if ((*inputs.is_mm_embed)[static_cast<size_t>(t)] != 0) masked_at.push_back(t);
  }
  for (const vt::Tensor& slice : *inputs.mm_embeds) {
    for (int64_t r = 0; r < slice.shape[0]; ++r) {
      const int64_t t = masked_at[static_cast<size_t>(next++)];
      backend.Copy(queue, merged.data() + t * hidden,
                   static_cast<const uint16_t*>(slice.data) + r * hidden,
                   static_cast<size_t>(hidden) * sizeof(uint16_t));
    }
  }
  backend.Synchronize(queue);

  const size_t bytes = merged.size() * vt::SizeOf(vt::DType::kBF16);
  std::shared_ptr<void> buf = DeviceBuffer(backend, bytes);
  backend.Copy(queue, buf.get(), merged.data(), bytes);
  backend.Synchronize(queue);
  MmForwardBuffers out;
  out.mm.inputs_embeds = dense_attn::MakeTensor(
      buf.get(), vt::DType::kBF16, queue.device, {tokens, hidden});
  out.storage.push_back(std::move(buf));
  // `positions3`, `deepstack` and `ple_token_ids` stay unset. DeepSeek-V4 reads
  // the ordinary one-dimensional positions, has no DeepStack and has no
  // per-layer embedding table, so publishing any of them would be inventing a
  // channel the backbone never reads.
  return out;
}

}  // namespace vllm
