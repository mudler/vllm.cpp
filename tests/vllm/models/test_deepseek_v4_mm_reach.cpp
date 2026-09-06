// MODEL-MM-deepseek-v4 W4 (#2411) — DOES AN IMAGE REACH THE MODEL?
//
// Every wave before this one landed a capability that nothing could arrive at.
// W1 encodes the prompt and preprocesses the image, W2 runs the ViT and the
// aligner, W3A reads the `deepseek4v` projector and W3B loads the vision
// routing bias, and the row's spec lists all four under `## Owed` as
// unreachable: no production entry point constructed any of them, and every
// gate that was green reached its subject by building it in the test.
//
// `AGENTS.md` §"Nothing lands dead" says what a gate has to do about that. This
// suite enters through the production seams and nothing else:
//
//   `ModelRegistry::Load`     with a source carrying the second file
//   `ModelRegistry::EncodeMm` the registered `encode_mm` hook
//   `ModelRegistry::EmbedMm`  the registered `embed_mm` hook
//   `ModelRegistry::Forward`  the registered forward
//
// It never constructs `DeepSeekV4Vision`, `DeepSeekV4ClipMmproj` or
// `DeepseekV4LoadedModel` as a DRIVER. It does construct the tower once as an
// ORACLE, to say what the rows should have been, which is a different job: an
// oracle that agrees with the driver proves the production path ran the same
// composition, and an oracle that is the driver proves nothing at all.
//
// THE REACHABILITY MUTATION this suite is written for: delete the `input.mm`
// branch in `ForwardDeepseekV4ForCausalLM` and the last case here goes red,
// because the expanded prompt is out-of-vocabulary sentinel identifiers and a
// forward that embeds them instead of consuming `inputs_embeds` refuses.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "deepseek_v4_lang_gguf_fixture.h"
#include "deepseek_v4_mmproj_fixture.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/clip_mmproj_gguf.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_mm.h"
#include "vllm/model_executor/models/deepseek_v4_vision.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/multimodal/deepseek_v4_processor.h"
#include "vllm/multimodal/inputs.h"
#include "vt/dtype.h"
#include "vt/backend.h"
#include "vt/tensor.h"

namespace {

using dsv4_lang_test::BuildDeepseek4Gguf;
using dsv4_lang_test::kH;
using dsv4_lang_test::kVocab;
using gguf_test::TempFile;
using vllm::multimodal::DeepSeekV4ProcessorConfig;
using vllm::multimodal::DeepSeekV4VisionConfig;
using vllm::multimodal::ImageKwargs;
using vllm::multimodal::MultiModalInputs;

// The projector geometry. `output` MUST be the language model's hidden width,
// because the aligner's output rows go straight into the residual stream; every
// other axis stays at the W3A fixture's deliberately distinct values so a
// transposed or mis-strided read cannot pass by symmetry.
dsv4_mmproj_test::Dims ProjDims() {
  dsv4_mmproj_test::Dims d;
  d.output = kH;
  return d;
}

// This suite RUNS the tower, so it asks for the folded value series. The reader
// gate's unfolded one puts weights at `2^104`, and a two-layer product of those
// is infinity before any comparison can read it.
dsv4_mmproj_test::Options ProjOptions() {
  dsv4_mmproj_test::Options o;
  o.fold_exponents = 7;
  return o;
}

// A 6x9 PATCH grid, which is 2x3 aligner cells at `downsample_ratio` 3.
//
// Neither factor is 1 and the two differ, so a row/column transposition inside
// the aligner, and a row-pair reorder that never reorders, are both visible.
// A 3x3 grid would be one cell and could express neither.
constexpr int64_t kGridH = 6;
constexpr int64_t kGridW = 9;

// One image's processed features, in the shape `PrepareDeepSeekV4Inputs`
// validates: `[num_patches, 3 * patch^2]` at the projector's own patch size.
std::shared_ptr<ImageKwargs> MakeImage(const DeepSeekV4VisionConfig& cfg) {
  auto image = std::make_shared<ImageKwargs>();
  image->num_patches = kGridH * kGridW;
  image->patch_feature_dim = cfg.patch_dim();
  image->image_grid_thw = {1, kGridH, kGridW};
  const int64_t n = image->num_patches * image->patch_feature_dim;
  image->pixel_values_f32.resize(static_cast<size_t>(n));
  image->pixel_values_bf16.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    // bf16-exact and strictly increasing inside each 128-index run, so a
    // permuted patch cannot land on an equal word.
    const float v = std::ldexp(1.0F + static_cast<float>(i % 128) / 128.0F,
                               static_cast<int>(i / 128) - 4);
    image->pixel_values_f32[static_cast<size_t>(i)] = v;
    image->pixel_values_bf16[static_cast<size_t>(i)] = vt::F32ToBF16(v);
  }
  return image;
}

DeepSeekV4ProcessorConfig ProcCfg(const DeepSeekV4VisionConfig& cfg) {
  DeepSeekV4ProcessorConfig p;
  p.patch_size = cfg.patch_size;
  p.downsample_ratio = cfg.downsample_ratio;
  p.vocab_size = static_cast<int32_t>(kVocab);
  return p;
}

// The whole production load, in the order the engine performs it.
struct Loaded {
  std::unique_ptr<TempFile> lang;
  std::unique_ptr<TempFile> proj;
  std::unique_ptr<vllm::GgufFile> lang_gguf;
  std::unique_ptr<vllm::GgufFile> proj_gguf;
  vllm::HfConfig config;
  std::unique_ptr<vllm::LoadedModel> model;
};

std::unique_ptr<Loaded> LoadThroughRegistry(bool vision_checkpoint,
                                            bool with_mmproj) {
  auto out = std::make_unique<Loaded>();
  // `ModelRegistry::Load` runs `ParseDeepseekV4Config`, which refuses every MLA
  // width but 512 by name, so this path takes the released geometry rather than
  // the W3B gate's tiny one.
  out->lang = std::make_unique<TempFile>(BuildDeepseek4Gguf(
      vision_checkpoint, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512));
  out->proj =
      std::make_unique<TempFile>(
      dsv4_mmproj_test::Build(ProjDims(), ProjOptions()));
  out->lang_gguf = std::make_unique<vllm::GgufFile>(
      vllm::GgufFile::Open(out->lang->path()));
  out->proj_gguf = std::make_unique<vllm::GgufFile>(
      vllm::GgufFile::Open(out->proj->path()));
  out->config = vllm::DeepseekV4HfConfigFromGguf(*out->lang_gguf);
  vllm::ModelSource source =
      vllm::ModelSource::FromGguf(*out->lang_gguf, vt::DeviceType::kCPU);
  if (with_mmproj) {
    source.mmproj = out->proj_gguf.get();
    source.mmproj_path = out->proj->path();
  }
  out->model = vllm::ModelRegistry::Load(out->config, source);
  return out;
}

float Bf16RowValue(const vt::Tensor& t, int64_t row, int64_t column) {
  const int64_t index = row * t.shape[1] + column;
  return vt::BF16ToF32(t.Ptr<uint16_t>()[index]);
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
// (1) The registration itself. This is the cheapest of the four questions and
// the one every other case depends on: a runner never calls `encode_mm` or
// `embed_mm` unless `ModelRegistry::SupportsMmInputs` says the registration set
// both, so leaving either null keeps the whole multimodal arm off for this
// architecture no matter what the model can do.
TEST_CASE("REACH: DeepSeek-V4 advertises a multimodal input path to the runner") {
  auto loaded = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                    /*with_mmproj=*/true);
  REQUIRE(loaded->model != nullptr);
  const vllm::ModelRegistration& reg = loaded->model->registration();
  CHECK(reg.info.supports_multimodal);
  CHECK(reg.factory->encode_mm != nullptr);
  CHECK(reg.factory->embed_mm != nullptr);
  CHECK(vllm::ModelRegistry::SupportsMmInputs(*loaded->model));
  // DeepSeek-V4 does not use M-RoPE. Upstream's `uses_mrope == False` is a null
  // `mrope_prompt_positions`, and asserting it is what stops a later wave from
  // adding Qwen3-VL's three-axis positions to a model whose reference uses the
  // ordinary one-dimensional ones.
  CHECK(reg.factory->mrope_prompt_positions == nullptr);
  CHECK_FALSE(vllm::ModelRegistry::UsesMrope(*loaded->model));
}

// ───────────────────────────────────────────────────────────────────────────
// (2) The projector reaches the tower. `encode_mm` is the runner's
// `execute_mm_encoder` hook and the ONLY production caller of the W2 tower and
// the W3A reader.
TEST_CASE("REACH: ModelRegistry::EncodeMm runs the W2 tower on the W3A projector") {
  auto loaded = LoadThroughRegistry(true, true);
  const DeepSeekV4VisionConfig vcfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
  const auto image = MakeImage(vcfg);
  const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
      {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
      static_cast<int32_t>(kVocab) - 1, {image}, ProcCfg(vcfg));
  REQUIRE(mm.mm_features.size() == 1);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const vllm::MmEncoderOutput out = vllm::ModelRegistry::EncodeMm(
      *loaded->model, loaded->config, queue, mm.mm_features[0]);

  // ONE ROW PER SENTINEL TOKEN. `gather_mm_embeddings` indexes the encoder
  // output by the token's offset inside the feature span, so a tower that
  // returned only its aligner rows would put the wrong vector under every
  // marker on a chunked prefill and be invisible on an unchunked one.
  CHECK(out.embeds.rank == 2);
  CHECK(out.embeds.shape[0] == mm.mm_features[0].length);
  CHECK(out.embeds.shape[1] == kH);
  CHECK(out.embeds.dtype == vt::DType::kBF16);

  // THE ORACLE. The same composition, built by hand out of the same projector,
  // says what those rows should be. This is the only hand-built tower in the
  // suite and it is deliberately not the driver.
  DeepSeekV4VisionConfig ocfg = vcfg;
  vllm::DeepSeekV4ClipMmproj oracle_weights =
      vllm::LoadDeepSeekV4VisionFromClipMmproj(*loaded->proj_gguf, ocfg);
  vllm::multimodal::DeepSeekV4Vision oracle(backend, ocfg,
                                            std::move(oracle_weights.weights));
  const int64_t aligned_rows = ocfg.aligned_rows(kGridH, kGridW);
  vt::Tensor patches = vt::Tensor::Contiguous(
      const_cast<uint16_t*>(image->pixel_values_bf16.data()), vt::DType::kBF16,
      queue.device, {image->num_patches, image->patch_feature_dim});
  std::vector<uint16_t> aligner(
      static_cast<size_t>(aligned_rows * ocfg.output_size));
  vt::Tensor aligner_view = vt::Tensor::Contiguous(
      aligner.data(), vt::DType::kBF16, queue.device,
      {aligned_rows, ocfg.output_size});
  oracle.Forward(queue, aligner_view, patches, kGridH, kGridW);

  // The block layout the processor wrote, recomputed from the SAME inputs the
  // hook has: the feature's offset and the image's grid.
  const vllm::multimodal::DeepSeekV4ImageBlock block =
      vllm::multimodal::BuildDeepSeekV4ImageBlock(
          (kGridH + ocfg.downsample_ratio - 1) / ocfg.downsample_ratio,
          (kGridW + ocfg.downsample_ratio - 1) / ocfg.downsample_ratio,
          mm.mm_features[0].offset);
  REQUIRE(static_cast<int64_t>(block.types.size()) ==
          mm.mm_features[0].length);

  // Every marker row is its own learned vector and every image row is the
  // aligner row the PERMUTATION names. Aggregated, so the assertion count does
  // not swamp the signal, and separated by role so a mutation that swapped the
  // two families is not averaged away.
  int64_t image_rows = 0, marker_rows = 0, image_bad = 0, marker_bad = 0;
  size_t taken = 0;
  for (size_t i = 0; i < block.types.size(); ++i) {
    const int64_t type = block.types[i];
    if (type == vllm::multimodal::kImage) {
      const int64_t source = block.permutation[taken++];
      ++image_rows;
      for (int64_t c = 0; c < kH; ++c) {
        const float want = vt::BF16ToF32(
            aligner[static_cast<size_t>(source * kH + c)]);
        if (Bf16RowValue(out.embeds, static_cast<int64_t>(i), c) != want) {
          ++image_bad;
        }
      }
      continue;
    }
    const std::vector<float>* want = nullptr;
    switch (type) {
      case vllm::multimodal::kImageStart: want = &oracle_weights.image_start; break;
      case vllm::multimodal::kImageEnd: want = &oracle_weights.image_end; break;
      case vllm::multimodal::kImagePad: want = &oracle_weights.image_pad; break;
      default: want = &oracle_weights.image_newline; break;
    }
    ++marker_rows;
    for (int64_t c = 0; c < kH; ++c) {
      const float expect = vt::BF16ToF32(
          vt::F32ToBF16((*want)[static_cast<size_t>(c)]));
      if (Bf16RowValue(out.embeds, static_cast<int64_t>(i), c) != expect) {
        ++marker_bad;
      }
    }
  }
  // The fixture must exercise both families, or the comparison above is
  // satisfied by a block that has only one of them.
  CHECK(image_rows == aligned_rows);
  CHECK(marker_rows > 0);
  CHECK(image_bad == 0);
  CHECK(marker_bad == 0);
}

// ───────────────────────────────────────────────────────────────────────────
// (3) The tower's rows reach the residual stream. `embed_mm` is the runner's
// only builder of `MultiModalForwardInput::inputs_embeds`.
TEST_CASE("REACH: ModelRegistry::EmbedMm merges the encoder rows into inputs_embeds") {
  auto loaded = LoadThroughRegistry(true, true);
  const DeepSeekV4VisionConfig vcfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
  const auto image = MakeImage(vcfg);
  const std::vector<int32_t> prompt{1, 2, static_cast<int32_t>(kVocab) - 1, 3};
  const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
      prompt, static_cast<int32_t>(kVocab) - 1, {image}, ProcCfg(vcfg));

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const vllm::MmEncoderOutput enc = vllm::ModelRegistry::EncodeMm(
      *loaded->model, loaded->config, queue, mm.mm_features[0]);

  const int64_t tokens = static_cast<int64_t>(mm.prompt_token_ids.size());
  std::vector<char> is_mm(static_cast<size_t>(tokens), 0);
  for (int i = 0; i < mm.mm_features[0].length; ++i) {
    is_mm[static_cast<size_t>(mm.mm_features[0].offset + i)] = 1;
  }
  const std::vector<vt::Tensor> slices{enc.embeds};
  vllm::MmEmbedInputs in;
  in.token_ids = &mm.prompt_token_ids;
  in.mm_embeds = &slices;
  in.is_mm_embed = &is_mm;

  const vllm::MmForwardBuffers buffers =
      vllm::ModelRegistry::EmbedMm(*loaded->model, loaded->config, queue, in);
  REQUIRE(buffers.mm.inputs_embeds.data != nullptr);
  CHECK(buffers.mm.inputs_embeds.rank == 2);
  CHECK(buffers.mm.inputs_embeds.shape[0] == tokens);
  CHECK(buffers.mm.inputs_embeds.shape[1] == kH);
  CHECK(buffers.mm.inputs_embeds.dtype == vt::DType::kBF16);
  // DeepSeek-V4 takes the ordinary one-dimensional positions, so this stays
  // unset. Asserting it is what stops a copy of the Qwen3-VL hook from
  // publishing three-axis positions the DeepSeek backbone never reads.
  CHECK(buffers.mm.positions3.data == nullptr);
  CHECK(buffers.mm.deepstack.data == nullptr);
  CHECK(buffers.mm.deepstack_levels == 0);

  // (a) Every masked row is the encoder row, byte for byte.
  int64_t merged_bad = 0;
  for (int i = 0; i < mm.mm_features[0].length; ++i) {
    const int64_t row = mm.mm_features[0].offset + i;
    for (int64_t c = 0; c < kH; ++c) {
      if (Bf16RowValue(buffers.mm.inputs_embeds, row, c) !=
          Bf16RowValue(enc.embeds, i, c)) {
        ++merged_bad;
      }
    }
  }
  CHECK(merged_bad == 0);

  // (b) Every UNmasked row is the language model's own embedding of its token,
  // which is the half a hook that merged the whole tensor would destroy.
  const auto& weights =
      vllm::ModelAs<vllm::DeepseekV4LoadedModel>(*loaded->model,
                                                 "DeepseekV4ForCausalLM")
          .weights();
  int64_t text_rows = 0, text_bad = 0;
  for (int64_t t = 0; t < tokens; ++t) {
    if (is_mm[static_cast<size_t>(t)] != 0) continue;
    ++text_rows;
    const int64_t token = mm.prompt_token_ids[static_cast<size_t>(t)];
    for (int64_t c = 0; c < kH; ++c) {
      const float want = vt::BF16ToF32(vt::F32ToBF16(
          weights.host.embed[static_cast<size_t>(token * kH + c)]));
      if (Bf16RowValue(buffers.mm.inputs_embeds, t, c) != want) ++text_bad;
    }
  }
  CHECK(text_rows == static_cast<int64_t>(prompt.size()) - 1);
  CHECK(text_bad == 0);
}

// ───────────────────────────────────────────────────────────────────────────
// (4) THE REACHABILITY CASE. One step through `ModelRegistry::Forward` on the
// expanded prompt.
//
// Its power comes from the sentinel identifiers: `PrepareDeepSeekV4Inputs`
// writes `vocab_size + type`, which is OUT OF VOCABULARY by construction. A
// registered forward that ignored `inputs_embeds` and embedded the identifiers
// would not answer wrongly, it would refuse — so deleting the `input.mm` branch
// in `ForwardDeepseekV4ForCausalLM` turns this case red rather than leaving it
// green on a class it never reached.
TEST_CASE("REACH: an image reaches ModelRegistry::Forward and moves the logits") {
  auto loaded = LoadThroughRegistry(true, true);
  const DeepSeekV4VisionConfig vcfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
  const auto image = MakeImage(vcfg);
  const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
      {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
      static_cast<int32_t>(kVocab) - 1, {image}, ProcCfg(vcfg));

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const vllm::MmEncoderOutput enc = vllm::ModelRegistry::EncodeMm(
      *loaded->model, loaded->config, queue, mm.mm_features[0]);
  const int64_t tokens = static_cast<int64_t>(mm.prompt_token_ids.size());
  std::vector<char> is_mm(static_cast<size_t>(tokens), 0);
  for (int i = 0; i < mm.mm_features[0].length; ++i) {
    is_mm[static_cast<size_t>(mm.mm_features[0].offset + i)] = 1;
  }
  const std::vector<vt::Tensor> slices{enc.embeds};
  vllm::MmEmbedInputs embed_in;
  embed_in.token_ids = &mm.prompt_token_ids;
  embed_in.mm_embeds = &slices;
  embed_in.is_mm_embed = &is_mm;
  vllm::MmForwardBuffers buffers =
      vllm::ModelRegistry::EmbedMm(*loaded->model, loaded->config, queue,
                                   embed_in);

  std::vector<int32_t> positions(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(tokens - 1)};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {0};

  const auto forward = [&](bool with_mm) {
    vllm::ModelForwardInput in{.token_ids = mm.prompt_token_ids,
                               .positions = positions,
                               .attn_meta = attn_meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = attn_kv,
                               .gdn_state = gdn_state,
                               .config = loaded->config,
                               .queue = queue,
                               .logits_indices = logits_indices,
                               .num_reqs = 1};
    in.gather_logits = false;
    if (with_mm) in.mm = buffers.mm;
    return vllm::ModelRegistry::Forward(*loaded->model, in);
  };

  const vllm::ForwardLogits with_image = forward(/*with_mm=*/true);
  CHECK(with_image.rows == 1);
  CHECK(with_image.vocab == kVocab);
  REQUIRE(with_image.host.size() == static_cast<size_t>(kVocab));
  int64_t nonfinite = 0;
  for (const float v : with_image.host) {
    if (!std::isfinite(v)) ++nonfinite;
  }
  CHECK(nonfinite == 0);

  // WITHOUT the merged embeddings, the SAME identifiers refuse. This is the
  // sentence the reachability mutation reads: the production forward has no
  // other way to answer this prompt, so it cannot be green with the branch
  // removed.
  CHECK_THROWS(forward(/*with_mm=*/false));

  // THE TOWER'S OUTPUT IS LOAD-BEARING. Perturbing one image row of the merged
  // tensor moves the logits, so the merge is read rather than carried.
  std::vector<uint16_t> saved(static_cast<size_t>(tokens * kH));
  const size_t bytes = saved.size() * sizeof(uint16_t);
  backend.Copy(queue, saved.data(), buffers.mm.inputs_embeds.data, bytes);
  backend.Synchronize(queue);
  std::vector<uint16_t> nudged = saved;
  const int64_t image_row = mm.mm_features[0].offset;
  for (int64_t c = 0; c < kH; ++c) {
    nudged[static_cast<size_t>(image_row * kH + c)] =
        vt::F32ToBF16(vt::BF16ToF32(saved[static_cast<size_t>(image_row * kH + c)]) + 1.0F);
  }
  backend.Copy(queue, buffers.mm.inputs_embeds.data, nudged.data(), bytes);
  backend.Synchronize(queue);
  const vllm::ForwardLogits perturbed = forward(/*with_mm=*/true);
  REQUIRE(perturbed.host.size() == with_image.host.size());
  int64_t moved = 0;
  for (size_t i = 0; i < perturbed.host.size(); ++i) {
    if (perturbed.host[i] != with_image.host[i]) ++moved;
  }
  CHECK(moved > 0);
  backend.Copy(queue, buffers.mm.inputs_embeds.data, saved.data(), bytes);
  backend.Synchronize(queue);
}

// ───────────────────────────────────────────────────────────────────────────
// (5) TEXT INERTNESS. A DeepSeek-V4 TEXT checkpoint, loaded with no `--mmproj`,
// stays tower-free and answers exactly as it did before this wave.
TEST_CASE("REACH: a text checkpoint loads tower-free and its tokens do not move") {
  auto text = LoadThroughRegistry(/*vision_checkpoint=*/false,
                                  /*with_mmproj=*/false);
  const auto& model = vllm::ModelAs<vllm::DeepseekV4LoadedModel>(
      *text->model, "DeepseekV4ForCausalLM");
  // NO VISION ALLOCATION. `has_vision()` is false, so nothing built the tower
  // and nothing read the projector -- which is also what makes the refusal
  // below the only possible answer rather than an accident of ordering.
  CHECK_FALSE(model.has_vision());

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  const std::vector<int32_t> positions{0, 1, 2, 3};
  const std::vector<int32_t> logits_indices{3};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {0};
  vllm::ModelForwardInput in{.token_ids = prompt,
                             .positions = positions,
                             .attn_meta = attn_meta,
                             .gdn_meta = gdn_meta,
                             .attn_kv = attn_kv,
                             .gdn_state = gdn_state,
                             .config = text->config,
                             .queue = queue,
                             .logits_indices = logits_indices,
                             .num_reqs = 1};
  in.gather_logits = false;
  // `mm` is left unset, which is every text step. The forward must take the
  // embedding path it always took.
  CHECK_FALSE(in.mm.has_value());
  const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*text->model, in);
  REQUIRE(out.host.size() == static_cast<size_t>(kVocab));

  // And asking the tower-free model for an encoder output refuses BY NAME
  // rather than returning an empty tensor a runner would splice as zeros.
  vllm::multimodal::MultiModalFeatureSpec item;
  item.length = 1;
  item.data = std::make_shared<ImageKwargs>();
  CHECK_THROWS(vllm::ModelRegistry::EncodeMm(*text->model, text->config, queue,
                                             item));
}

// ───────────────────────────────────────────────────────────────────────────
// (6) THE LOADER HOP. `LoadedEngine::FromModelDir` is the entry point every
// server and command line takes for a `.gguf` argument, and the projector block
// inside it is where `--mmproj` picks a reader.
//
// Before W4 that block called `RefuseUnsupportedClipMmproj` unconditionally,
// which refuses every projector type but `qwen3vl_merger` BY NAME -- so a
// correct `deepseek4v` file could not get past it. These cases drive the real
// entry point and read the message it produces.
TEST_CASE("REACH: --mmproj sends a deepseek4v projector past the Qwen3-VL reader") {
  TempFile lang(BuildDeepseek4Gguf(/*vision=*/true, dsv4_lang_test::BiasWidths{},
                                   /*vision_from=*/0, /*head_dim=*/512));
  TempFile proj(dsv4_mmproj_test::Build(ProjDims(), ProjOptions()));

  const auto refusal_for = [&](const std::string& projector_path) {
    vllm::entrypoints::EngineParams params;
    params.mmproj_path = projector_path;
    try {
      vllm::entrypoints::LoadedEngine::FromModelDir(lang.path(), params);
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
    return std::string();
  };

  // (a) The DeepSeek projector is NOT refused by the Qwen3-VL discriminator any
  // more, and the claim is POSITIVE rather than an absence: the load now
  // reaches the TOKENIZER, which is the statement immediately after the
  // projector block. This tiny language fixture carries no tokenizer keys, so
  // that is where it stops. An "it no longer says qwen3vl_merger" assertion
  // alone would be satisfied by any earlier failure at all.
  const std::string deepseek = refusal_for(proj.path());
  CHECK(deepseek.find("tokenizer") != std::string::npos);
  CHECK(deepseek.find("qwen3vl_merger") == std::string::npos);

  // (b) THE DISCRIMINATION IS REAL, not a removed refusal. A projector of a
  // type this build does not load still meets the Qwen3-VL message, naming
  // both types, exactly as it did before this wave.
  dsv4_mmproj_test::Options other = ProjOptions();
  other.projector_type = "gemma3";
  TempFile foreign(dsv4_mmproj_test::Build(ProjDims(), other));
  const std::string refused = refusal_for(foreign.path());
  CHECK(refused.find("qwen3vl_merger") != std::string::npos);
  CHECK(refused.find("gemma3") != std::string::npos);

  // (c) THE DEEPSEEK ARM'S OWN ACCOUNTING RUNS, and this is the case the
  // mutation reads. A `deepseek4v` projector carrying a tensor this reader
  // never reads must be refused BY THAT READER, naming the tensor, and BEFORE
  // the tokenizer error above -- which is what makes the message "you have a
  // file this build only half consumes" instead of "this file has no
  // tokenizer". Delete the `RefuseDeepSeekV4ClipMmprojArm` call in
  // `model_loader.cpp` and this case sees the tokenizer error, because nothing
  // looked at the projector at all.
  //
  // A MISSING tensor would not do: the missing direction names itself inside
  // the reader, and the read is deliberately deferred to
  // `LoadDeepseekV4ForCausalLM` so the tower lands on the model.
  dsv4_mmproj_test::Options stray = ProjOptions();
  stray.stray_tensor = "v.blk.0.attn_norm.weight";
  TempFile extra(dsv4_mmproj_test::Build(ProjDims(), stray));
  const std::string unaccounted = refusal_for(extra.path());
  CHECK(unaccounted.find("v.blk.0.attn_norm.weight") != std::string::npos);
  CHECK(unaccounted.find("NEVER reads") != std::string::npos);
  CHECK(unaccounted.find("tokenizer") == std::string::npos);
}

// (7) THE REFUSAL ORDER, at the production call site.
//
// `RefuseUnsupportedDeepSeekV4ClipMmproj` must speak BEFORE
// `RefuseUnaccountedDeepSeekV4ClipMmproj`, and until W4 nothing but a helper
// inside the W3A suite ran the two together, so nothing held the order.
//
// The file that makes the order matter is not hypothetical. The pinned oracle's
// own `convert_hf_to_gguf.py` emits the FUSED `v.blk.{bid}.attn_qkv` -- nothing
// splits it for this family -- and this build does not implement that arm. Its
// names are not in the enumerated set, so the unaccounted refusal fires on it
// too. Reversed, a user with a CORRECTLY converted projector is told it
// "carries tensors we never read" and re-converts a file that was already
// right, which is exactly the outcome the W3B refusal exists to prevent.
//
// Swap the two calls in `RefuseDeepSeekV4ClipMmprojArm` and this case goes red.
TEST_CASE("REACH: a FUSED-qkv projector is told the arm is missing, not that it is unaccounted") {
  TempFile lang(BuildDeepseek4Gguf(/*vision=*/true, dsv4_lang_test::BiasWidths{},
                                   /*vision_from=*/0, /*head_dim=*/512));
  dsv4_mmproj_test::Options fused = ProjOptions();
  fused.fused_qkv = true;
  TempFile proj(dsv4_mmproj_test::Build(ProjDims(), fused));

  vllm::entrypoints::EngineParams params;
  params.mmproj_path = proj.path();
  std::string message;
  try {
    vllm::entrypoints::LoadedEngine::FromModelDir(lang.path(), params);
  } catch (const std::exception& e) {
    message = e.what();
  }
  // It names the LAYOUT this build does not implement...
  CHECK(message.find("attn_qkv") != std::string::npos);
  // ...and the issue that owes the arm, so the user does not re-convert.
  CHECK(message.find("2411") != std::string::npos);
  // ...and it is NOT the unaccounted-tensor refusal, which blames the file.
  CHECK(message.find("NEVER reads") == std::string::npos);
}

// (8) THE PROJECTOR REACHES THE MODEL'S OWN LOADER, and this is the case that
// reads the one line handing it down. `ModelSource::mmproj` is what
// `LoadDeepseekV4ForCausalLM` opens, and it is set in `model_loader.cpp` after
// the tokenizer -- so a fixture that stops AT the tokenizer, as every case
// above does, cannot see it at all.
//
// The observable is a MISMATCHED pair: a projector whose aligner is not the
// language model's width. `LoadDeepseekV4VisionRuntime` refuses that by name,
// and the refusal exists only if the file arrived. Set
// `gguf_source.mmproj = nullptr` in `model_loader.cpp` and this case goes green
// on a load that quietly built no tower, which is the failure the line prevents.
TEST_CASE("REACH: ModelSource::mmproj carries the projector into the model loader") {
  TempFile lang(BuildDeepseek4Gguf(/*vision=*/true, dsv4_lang_test::BiasWidths{},
                                   /*vision_from=*/0, /*head_dim=*/512,
                                   /*with_tokenizer=*/true));
  dsv4_mmproj_test::Dims wrong = ProjDims();
  wrong.output = kH + 4;  // NOT the language model's hidden width
  TempFile proj(dsv4_mmproj_test::Build(wrong, ProjOptions()));

  vllm::entrypoints::EngineParams params;
  params.mmproj_path = proj.path();
  std::string message;
  try {
    vllm::entrypoints::LoadedEngine::FromModelDir(lang.path(), params);
  } catch (const std::exception& e) {
    message = e.what();
  }
  CHECK(message.find("not a pair") != std::string::npos);
  CHECK(message.find(std::to_string(kH + 4)) != std::string::npos);
  CHECK(message.find(std::to_string(kH)) != std::string::npos);
  // And it is NOT the tokenizer error, which is what every case above stops at.
  CHECK(message.find("tokenizer") == std::string::npos);
}
