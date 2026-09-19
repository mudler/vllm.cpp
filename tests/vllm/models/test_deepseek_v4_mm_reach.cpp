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
#include <string_view>
#include <vector>

#include "deepseek_v4_lang_gguf_fixture.h"
#include "deepseek_v4_mmproj_fixture.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
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
//
// THE VALUE IS 13 AND IT IS MEASURED, not chosen for looks. The fold decides
// how far the weights spread along the aligner's 72-wide contraction axis, and
// that spread is the only thing that makes one aligner cell's answer differ
// from another's: bf16 keeps 8 significant bits, so a cell-to-cell difference
// below about 0.4% of the row's own magnitude is not representable at all. At
// this fixture's geometry the six aligner rows come out
//   fold  7 -> 5 of 6 distinct
//   fold 11 -> 6 of 6, largest column spread 0.93% of the column maximum
//   fold 13 -> 6 of 6, largest column spread 2.4%
// so 13 is the first value with a margin over the representable floor rather
// than the first value that happens to pass. The case below ASSERTS the six are
// pairwise distinct, so a later change that collapses them again is red rather
// than vacuously green.
dsv4_mmproj_test::Options ProjOptions() {
  dsv4_mmproj_test::Options o;
  o.fold_exponents = 13;
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
                                            bool with_mmproj,
                                            float vision_bias_scale = 1.0F) {
  auto out = std::make_unique<Loaded>();
  // `ModelRegistry::Load` runs `ParseDeepseekV4Config`, which refuses every MLA
  // width but 512 by name, so this path takes the released geometry rather than
  // the W3B gate's tiny one.
  out->lang = std::make_unique<TempFile>(BuildDeepseek4Gguf(
      vision_checkpoint, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/false, vision_bias_scale));
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
// (1b) A PROJECTOR HANDED TO AN ARCHITECTURE THAT READS NONE IS REFUSED.
//
// `ModelRegistry::Load` is the production call `model_loader.cpp:3126` makes,
// and the projector reaches it from a branch that keys on the PROJECTOR TYPE
// alone (`model_loader.cpp:3020-3023`): `--mmproj <deepseek4v>.gguf` sets
// `gguf_source.mmproj` whatever the language file's architecture is. Nothing
// downstream of that point would notice. A `load_weights` that does not read
// `ModelSource::mmproj` ignores it, the load SUCCEEDS, no tower exists, and the
// first image request is answered as text -- the failure the check's own
// comment names.
//
// Until this case the flag was asserted (case (1) and the scaffold suite) and
// the refusal was not: removing the `VT_CHECK` in `ModelRegistry::Load` left
// all 24 tests of this family green.
//
// The other architecture is DISCOVERED rather than named, because naming one
// pins this case to a model that may be renamed or retired, and what is being
// gated is the property and not the model.
TEST_CASE("REACH: a deepseek4v projector handed to a non-consuming architecture is REFUSED") {
  auto loaded = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                    /*with_mmproj=*/true);
  REQUIRE(loaded->model != nullptr);

  // An architecture registered by THIS build whose loader reads no projector.
  std::string_view other;
  for (const std::string_view name :
       vllm::ModelRegistry::SupportedArchs()) {
    vllm::HfConfig probe;
    probe.architectures = {std::string(name)};
    if (!vllm::ModelRegistry::Resolve(probe).factory->consumes_mmproj) {
      other = name;
      break;
    }
  }
  REQUIRE_FALSE(other.empty());
  INFO("other architecture: ", other);

  // The SAME source the DeepSeek load above took, retargeted by config alone.
  // The refusal sits AFTER `Resolve` and BEFORE `parse_config`/`load_weights`,
  // so the language bytes are never read and the message is about the pairing
  // rather than about the first tensor whose name does not resolve.
  vllm::ModelSource source =
      vllm::ModelSource::FromGguf(*loaded->lang_gguf, vt::DeviceType::kCPU);
  source.mmproj = loaded->proj_gguf.get();
  source.mmproj_path = loaded->proj->path();
  vllm::HfConfig foreign;
  foreign.architectures = {std::string(other)};
  std::string message;
  try {
    (void)vllm::ModelRegistry::Load(foreign, source);
  } catch (const std::exception& e) {
    message = e.what();
  }
  INFO("message: ", message);
  // It names WHICH file, WHICH architecture, and what would have happened.
  CHECK(message.find(loaded->proj->path()) != std::string::npos);
  CHECK(message.find(std::string(other)) != std::string::npos);
  CHECK(message.find("NO vision tower") != std::string::npos);

  // THE DISCRIMINATION IS REAL. The identical source on the architecture that
  // DOES read a projector is not refused -- without this half, a `VT_CHECK`
  // that refused every projector would pass the assertions above.
  auto again = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                   /*with_mmproj=*/true);
  CHECK(again->model != nullptr);
  CHECK(vllm::ModelAs<vllm::DeepseekV4LoadedModel>(*again->model,
                                                   "DeepseekV4ForCausalLM")
            .has_vision());
}

// ───────────────────────────────────────────────────────────────────────────
// (1c) WHICH CHAT ARM THE SERVER'S INSTALL LANDS ON FOR THIS ARCHITECTURE.
//
// `InstallMultiModalChatSeam` reads exactly two inputs:
// `LoadedEngine::is_multimodal_model()`, which is `ModelInfo::supports_multimodal`
// off the loaded registration, and `MultiModalChatRegistry::Find(architecture)`.
// The pair picks one of three arms -- `kTextOnlyModel`, where nothing is
// installed and an image request is answered from the TEXT path; `kRefusing`,
// an HTTP 400 naming the architecture; and `kInstalled`.
//
// W4 flipped the flag while no factory was registered, which moved this
// architecture from `kTextOnlyModel` to `kRefusing`, and an earlier version of
// this case pinned that intermediate state by asserting `Find(arch) == nullptr`
// and reading the `REGISTER_VLLM_MM_CHAT` message off it. W5 then registered
// `mm_chat_deepseek_v4.cpp`, so that assertion described a tree that no longer
// exists: both inputs are positive now and the arm is `kInstalled`. What it
// meant is kept below on an architecture for which it is still true.
//
// This case measures both inputs off the LOADED model rather than off a
// hand-built registration. `test_deepseek_v4_mm_chat` drives the install itself
// and reads the arm it returns.
TEST_CASE("REACH: the multimodal flag and the registered seam put DeepSeek-V4 on the INSTALLED chat arm") {
  auto loaded = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                    /*with_mmproj=*/true);
  const std::string_view arch = loaded->model->registration().architecture;
  CHECK(arch == "DeepseekV4ForCausalLM");

  // (a) The flag `LoadedEngine::is_multimodal_model()` returns. True here is
  // what makes `InstallMultiModalChatSeam` look for a factory at all; false
  // would install nothing and answer image requests from the text path.
  CHECK(loaded->model->registration().info.supports_multimodal);

  // (b) A factory IS registered for this architecture. It is reached through
  // the static library's `--whole-archive`, so a link that dropped
  // `mm_chat_deepseek_v4.cpp` reads as an absent registration HERE rather than
  // as a 400 in front of a user.
  namespace oai = vllm::entrypoints::openai;
  const oai::MultiModalChatRegistration* reg =
      oai::MultiModalChatRegistry::Find(arch);
  REQUIRE(reg != nullptr);
  CHECK(reg->architecture == arch);
  CHECK(reg->make_seam != nullptr);

  // And that factory REFUSES BY NAME an install context it cannot serve rather
  // than half-installing: it needs the tokenizer, the multimodal config, the
  // resolved model config and the image codec, and an empty context carries
  // none of them. The install's catch turns this into the refusing arm, so a
  // misconfigured server still answers 400 and never answers an image from the
  // text path.
  oai::MultiModalChatContext ctx;
  ctx.architecture = arch;
  std::string message;
  try {
    (void)oai::MultiModalChatRegistry::MakeSeam(ctx);
  } catch (const std::exception& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find("DeepSeek-V4") != std::string::npos);
  CHECK(message.find("install context is incomplete") != std::string::npos);

  // THE NEGATIVE CONTROL, and it carries what the replaced assertion meant. An
  // architecture with no registered factory still gets the refusal that names
  // the architecture and names what to register -- which is what this case read
  // off DeepSeek-V4 before W5 gave it a seam.
  oai::MultiModalChatContext none;
  none.architecture = "NotARegisteredArchForCausalLM";
  std::string unregistered;
  try {
    (void)oai::MultiModalChatRegistry::MakeSeam(none);
  } catch (const std::exception& e) {
    unregistered = e.what();
  }
  INFO("unregistered: ", unregistered);
  CHECK(unregistered.find("NotARegisteredArchForCausalLM") != std::string::npos);
  CHECK(unregistered.find("REGISTER_VLLM_MM_CHAT") != std::string::npos);

  // And the refusing chat function the install builds from such a message
  // refuses a multimodal request while leaving a text one alone -- the property
  // that makes the refusing arm an improvement on `kTextOnlyModel`.
  const oai::MultiModalChatFn refuse = oai::MakeRefusingMultiModalChatFn(
      "NotARegisteredArchForCausalLM", unregistered);
  vllm::entrypoints::openai::ChatMessage text;
  text.role = "user";
  text.content = "hello";
  CHECK_FALSE(refuse({text}).has_value());
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
      static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));
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

  // THE ASSERTION THAT MAKES THE PERMUTATION GATED, and it is here because
  // without it the four above are satisfied by a fixture that says nothing.
  //
  // `image_bad == 0` reads "this image row is the aligner row the permutation
  // names". If the aligner rows are all the same vector it degenerates to "this
  // image row is SOME aligner row", and identity, reversal and a constant index
  // all satisfy it -- as does a row/column transposition inside the aligner,
  // which is what the fixture's own grid comment claims to catch. That is
  // exactly what this suite shipped: every one of the six rows was bit
  // identical, non-zero and finite, so nothing looked wrong.
  //
  // Pairwise distinctness is the property the permutation assertion needs, so
  // it is measured rather than assumed. It is a property of the FIXTURE and the
  // tower's arithmetic, not of the code under test, which is why it is a
  // separate assertion and not a stricter comparison.
  int64_t equal_pairs = 0;
  for (int64_t a = 0; a < aligned_rows; ++a) {
    for (int64_t b = a + 1; b < aligned_rows; ++b) {
      bool same = true;
      for (int64_t c = 0; c < ocfg.output_size && same; ++c) {
        same = aligner[static_cast<size_t>(a * ocfg.output_size + c)] ==
               aligner[static_cast<size_t>(b * ocfg.output_size + c)];
      }
      if (same) ++equal_pairs;
    }
  }
  CHECK(equal_pairs == 0);

  // And the four MARKER vectors are pairwise distinct too, so a hook that put
  // the start vector under every marker row would be visible rather than
  // averaged into `marker_bad`.
  const std::vector<const std::vector<float>*> markers{
      &oracle_weights.image_start, &oracle_weights.image_end,
      &oracle_weights.image_pad, &oracle_weights.image_newline};
  int64_t equal_markers = 0;
  for (size_t a = 0; a < markers.size(); ++a) {
    for (size_t b = a + 1; b < markers.size(); ++b) {
      if (*markers[a] == *markers[b]) ++equal_markers;
    }
  }
  CHECK(equal_markers == 0);
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
      prompt, static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));

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
// (3b) THE TWO "IS THIS AN IMAGE ROW" PREDICATES MUST AGREE.
//
// `EmbedMm` reads the runner's `is_mm_embed` MASK. `MoeBlock` and
// `DeepseekV4ImageSpans` read the IDENTIFIER, `id >= vocab_size`. They agree
// only because `PrepareDeepSeekV4Inputs` writes `vocab_size + type` at exactly
// the masked positions, and until this case nothing said so.
//
// One direction was already refused: an UNMASKED row carrying an
// out-of-vocabulary id meets the bounds check, because the embedding table has
// no row for it. The other was silent, and it is the dangerous one -- a MASKED
// row carrying a real token id takes the tower's vector into the residual
// stream while the router reads the TEXT bias for it and no image span opens
// over it. Everything downstream stays in range and the answer stays fluent.
TEST_CASE("REACH: a masked row carrying an in-vocabulary id is refused by EmbedMm") {
  auto loaded = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                    /*with_mmproj=*/true);
  const DeepSeekV4VisionConfig vcfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
  const auto image = MakeImage(vcfg);
  const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
      {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
      static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const vllm::MmEncoderOutput enc = vllm::ModelRegistry::EncodeMm(
      *loaded->model, loaded->config, queue, mm.mm_features[0]);
  const int64_t tokens = static_cast<int64_t>(mm.prompt_token_ids.size());

  // The mask the runner would build, SHIFTED BY ONE. Every row it marks is
  // still marked in the right number, so the encoder/mask balance below still
  // holds and only the predicate disagreement can catch it. Row 1 is a text
  // token of the original prompt.
  std::vector<char> is_mm(static_cast<size_t>(tokens), 0);
  for (int i = 0; i < mm.mm_features[0].length; ++i) {
    is_mm[static_cast<size_t>(mm.mm_features[0].offset + i - 1)] = 1;
  }
  REQUIRE(mm.mm_features[0].offset >= 1);
  REQUIRE(mm.prompt_token_ids[static_cast<size_t>(mm.mm_features[0].offset - 1)] <
          static_cast<int32_t>(kVocab));

  const std::vector<vt::Tensor> slices{enc.embeds};
  vllm::MmEmbedInputs in;
  in.token_ids = &mm.prompt_token_ids;
  in.mm_embeds = &slices;
  in.is_mm_embed = &is_mm;
  std::string message;
  try {
    (void)vllm::ModelRegistry::EmbedMm(*loaded->model, loaded->config, queue, in);
  } catch (const std::exception& e) {
    message = e.what();
  }
  INFO("message: ", message);
  // Phrases unique to THIS refusal. "multimodal placeholder" alone is in the
  // bounds message too, so it would still match with the check removed.
  CHECK(message.find("is marked as a multimodal placeholder but carries token id") !=
        std::string::npos);
  CHECK(message.find("inside the vocabulary") != std::string::npos);
  CHECK(message.find("routed on the text") != std::string::npos);

  // THE CONTROL. The mask the processor's own layout implies is accepted, so
  // the check above is a disagreement test and not a refusal of every mask.
  std::vector<char> right(static_cast<size_t>(tokens), 0);
  for (int i = 0; i < mm.mm_features[0].length; ++i) {
    right[static_cast<size_t>(mm.mm_features[0].offset + i)] = 1;
  }
  in.is_mm_embed = &right;
  const vllm::MmForwardBuffers ok =
      vllm::ModelRegistry::EmbedMm(*loaded->model, loaded->config, queue, in);
  CHECK(ok.mm.inputs_embeds.data != nullptr);
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
      static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));

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

// ───────────────────────────────────────────────────────────────────────────
// (9) THE VISION ROUTING BIAS IS READ, AND ONLY BY THE IMAGE ROWS.
//
// `exp_probs_b_vl` landed with W3B and nothing selected it. This case builds two
// language files differing in NOTHING but the values of that tensor, runs the
// same image through both, and reads two rows of the same forward:
//
//   * a TEXT row BEFORE the image span, whose causal prefix is text only. Under
//     a per-token rule it cannot move. Under llama.cpp's per-UBATCH rule it
//     would, because that rule puts every row of a media batch on the vision
//     bias -- so this assertion is what separates the two, and it is why the
//     choice in `deepseek_v4_moe.h` is a decision rather than a preference.
//   * the LAST row, whose prefix contains the whole span. It must move, because
//     the span's rows routed on a bias that changed.
TEST_CASE("REACH: the vision bias moves the image rows and leaves the text rows alone") {
  const auto run = [&](float scale) {
    auto loaded = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                      /*with_mmproj=*/true, scale);
    const DeepSeekV4VisionConfig vcfg =
        vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
    const auto image = MakeImage(vcfg);
    const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
        {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
        static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));
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
    vllm::MmForwardBuffers buffers = vllm::ModelRegistry::EmbedMm(
        *loaded->model, loaded->config, queue, embed_in);

    std::vector<int32_t> positions(static_cast<size_t>(tokens));
    for (int64_t t = 0; t < tokens; ++t) {
      positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    }
    // ROW 0 is a text row BEFORE the span (`offset` is 2 for this prompt), and
    // the last row is after it.
    REQUIRE(mm.mm_features[0].offset >= 1);
    const std::vector<int32_t> logits_indices{0, static_cast<int32_t>(tokens - 1)};
    std::vector<vllm::PagedKvCache> attn_kv;
    std::vector<vllm::GdnStateCache> gdn_state;
    const vllm::v1::GDNAttentionMetadata gdn_meta{};
    vllm::v1::CommonAttentionMetadata attn_meta{};
    attn_meta.num_reqs = 1;
    attn_meta.num_computed_tokens_cpu = {0};
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
    in.mm = buffers.mm;
    const vllm::ForwardLogits out =
        vllm::ModelRegistry::Forward(*loaded->model, in);
    REQUIRE(out.host.size() == static_cast<size_t>(2 * kVocab));
    return out.host;
  };

  const std::vector<float> a = run(1.0F);
  const std::vector<float> b = run(-4.0F);
  REQUIRE(a.size() == b.size());

  int64_t text_moved = 0, tail_moved = 0;
  for (int64_t v = 0; v < kVocab; ++v) {
    if (a[static_cast<size_t>(v)] != b[static_cast<size_t>(v)]) ++text_moved;
    const size_t tail = static_cast<size_t>(kVocab + v);
    if (a[tail] != b[tail]) ++tail_moved;
  }
  // The pre-span TEXT row is untouched: this is the per-token claim.
  CHECK(text_moved == 0);
  // The row whose prefix contains the span moves: the bias was read.
  CHECK(tail_moved > 0);
}

// ───────────────────────────────────────────────────────────────────────────
// (9b) THE HASH-LAYER SKIP, at the forward.
//
// `deepseek_v4_moe.cpp` routes an image row on the vision bias and the learned
// top-k EVEN ON A HASH LAYER, because that row has no token identifier worth
// hashing. `test_deepseek_v4_moe` gates the condition at the router; nothing
// gated it at a forward, and dropping `!media` left case (9) above GREEN even
// though its language fixture has a hash layer and the forward runs it with
// image rows.
//
// It stays green because case (9) has two GATED layers as well, and an image
// row reads the vision bias on those whichever way the hash layer routes. So
// `tail_moved > 0` survives, and `text_moved == 0` was never about this.
// Meanwhile the failure is silent and plausible rather than loud: the hash route
// is `hash_indices_table[(tok % vocab_size) * topk]`, so an image identifier
// `vocab + type` wraps to `tid2eid[type]` -- in bounds, a real expert, and the
// wrong one.
//
// THE FIXTURE IS THE WHOLE POINT HERE. Every layer of this file is a hash layer,
// so there is no gated layer left to read the vision bias on an image row. Two
// files differing in NOTHING but their `exp_probs_b_vl` values must still move
// the logits of a row whose prefix contains the image span -- and under the
// mutation they cannot, because the bias is then never read at all.
TEST_CASE("REACH: an image row leaves the hash route on a file whose every layer hashes") {
  const auto run = [&](float scale) {
    auto loaded = std::make_unique<Loaded>();
    loaded->lang = std::make_unique<TempFile>(BuildDeepseek4Gguf(
        /*vision=*/true, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
        /*head_dim=*/512, /*with_tokenizer=*/false, scale,
        /*sliding_window=*/0, /*hash_layers=*/dsv4_lang_test::kLayers));
    loaded->proj = std::make_unique<TempFile>(
        dsv4_mmproj_test::Build(ProjDims(), ProjOptions()));
    loaded->lang_gguf = std::make_unique<vllm::GgufFile>(
        vllm::GgufFile::Open(loaded->lang->path()));
    loaded->proj_gguf = std::make_unique<vllm::GgufFile>(
        vllm::GgufFile::Open(loaded->proj->path()));
    loaded->config = vllm::DeepseekV4HfConfigFromGguf(*loaded->lang_gguf);
    // EVERY layer hashes, which is what makes the assertion below possible.
    REQUIRE(vllm::ParseDeepseekV4Params(loaded->config).num_hash_layers ==
            dsv4_lang_test::kLayers);
    vllm::ModelSource source =
        vllm::ModelSource::FromGguf(*loaded->lang_gguf, vt::DeviceType::kCPU);
    source.mmproj = loaded->proj_gguf.get();
    source.mmproj_path = loaded->proj->path();
    loaded->model = vllm::ModelRegistry::Load(loaded->config, source);

    const DeepSeekV4VisionConfig vcfg =
        vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
    const auto image = MakeImage(vcfg);
    const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
        {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
        static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));
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
    vllm::MmForwardBuffers buffers = vllm::ModelRegistry::EmbedMm(
        *loaded->model, loaded->config, queue, embed_in);

    std::vector<int32_t> positions(static_cast<size_t>(tokens));
    for (int64_t t = 0; t < tokens; ++t) {
      positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    }
    const std::vector<int32_t> logits_indices{0,
                                              static_cast<int32_t>(tokens - 1)};
    std::vector<vllm::PagedKvCache> attn_kv;
    std::vector<vllm::GdnStateCache> gdn_state;
    const vllm::v1::GDNAttentionMetadata gdn_meta{};
    vllm::v1::CommonAttentionMetadata attn_meta{};
    attn_meta.num_reqs = 1;
    attn_meta.num_computed_tokens_cpu = {0};
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
    in.mm = buffers.mm;
    const vllm::ForwardLogits out =
        vllm::ModelRegistry::Forward(*loaded->model, in);
    REQUIRE(out.host.size() == static_cast<size_t>(2 * kVocab));
    return out.host;
  };

  const std::vector<float> a = run(1.0F);
  const std::vector<float> b = run(-4.0F);
  REQUIRE(a.size() == b.size());
  int64_t text_moved = 0, tail_moved = 0;
  for (int64_t v = 0; v < kVocab; ++v) {
    if (a[static_cast<size_t>(v)] != b[static_cast<size_t>(v)]) ++text_moved;
    const size_t tail = static_cast<size_t>(kVocab + v);
    if (a[tail] != b[tail]) ++tail_moved;
  }
  // A text row keeps the hash route on every layer, so the vision bias cannot
  // reach it -- the per-token claim, on a file where the hash branch is the
  // only other arm.
  CHECK(text_moved == 0);
  // And the row whose prefix contains the span moves, which on THIS file is
  // possible only if an image row left the hash route.
  CHECK(tail_moved > 0);
}

// ───────────────────────────────────────────────────────────────────────────
// (10) THE IMAGE SPAN IS NON-CAUSAL, at the forward.
//
// The index rule is gated on its indices in `test_deepseek_v4_dsa`. This case
// asks the other question: does `ForwardComposeImpl` READ it? An EARLY row of
// the span is asked for logits while a LATE row of the same span is perturbed.
// Causally the early row cannot see the late one, so under the dense causal
// list this branch built before W4 the logits do not move. Under the span rule
// they must.
TEST_CASE("REACH: a row early in the image span attends a row after it") {
  auto loaded = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                    /*with_mmproj=*/true);
  const DeepSeekV4VisionConfig vcfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
  const auto image = MakeImage(vcfg);
  const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
      {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
      static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const vllm::MmEncoderOutput enc = vllm::ModelRegistry::EncodeMm(
      *loaded->model, loaded->config, queue, mm.mm_features[0]);
  const int64_t tokens = static_cast<int64_t>(mm.prompt_token_ids.size());
  const int64_t span_begin = mm.mm_features[0].offset;
  const int64_t span_len = mm.mm_features[0].length;
  REQUIRE(span_len >= 4);
  std::vector<char> is_mm(static_cast<size_t>(tokens), 0);
  for (int i = 0; i < span_len; ++i) {
    is_mm[static_cast<size_t>(span_begin + i)] = 1;
  }
  const std::vector<vt::Tensor> slices{enc.embeds};
  vllm::MmEmbedInputs embed_in;
  embed_in.token_ids = &mm.prompt_token_ids;
  embed_in.mm_embeds = &slices;
  embed_in.is_mm_embed = &is_mm;
  vllm::MmForwardBuffers buffers =
      vllm::ModelRegistry::EmbedMm(*loaded->model, loaded->config, queue, embed_in);

  std::vector<int32_t> positions(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) {
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }
  // The EARLY row of the span, one past its start marker.
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(span_begin + 1)};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {0};
  const auto forward = [&]() {
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
    in.mm = buffers.mm;
    return vllm::ModelRegistry::Forward(*loaded->model, in).host;
  };

  const std::vector<float> before = forward();
  REQUIRE(before.size() == static_cast<size_t>(kVocab));

  // Perturb a LATE row of the SAME span. It is after the queried row, so only
  // the non-causal half of the rule can carry it.
  const int64_t late = span_begin + span_len - 2;
  REQUIRE(late > span_begin + 1);
  std::vector<uint16_t> saved(static_cast<size_t>(tokens * kH));
  const size_t bytes = saved.size() * sizeof(uint16_t);
  backend.Copy(queue, saved.data(), buffers.mm.inputs_embeds.data, bytes);
  backend.Synchronize(queue);
  std::vector<uint16_t> nudged = saved;
  for (int64_t c = 0; c < kH; ++c) {
    const size_t at = static_cast<size_t>(late * kH + c);
    nudged[at] = vt::F32ToBF16(vt::BF16ToF32(saved[at]) + 2.0F);
  }
  backend.Copy(queue, buffers.mm.inputs_embeds.data, nudged.data(), bytes);
  backend.Synchronize(queue);
  const std::vector<float> after = forward();
  int64_t moved = 0;
  for (size_t i = 0; i < after.size(); ++i) {
    if (after[i] != before[i]) ++moved;
  }
  CHECK(moved > 0);

  // THE CONTROL. Perturbing a row OUTSIDE the span and after the queried row
  // must NOT move it: the exemption is the span's, not a blanket
  // non-causality. Without this the case above is also satisfied by an
  // implementation that made the whole step bidirectional.
  backend.Copy(queue, buffers.mm.inputs_embeds.data, saved.data(), bytes);
  backend.Synchronize(queue);
  std::vector<uint16_t> outside = saved;
  const int64_t after_span = tokens - 1;
  REQUIRE(after_span >= span_begin + span_len);
  for (int64_t c = 0; c < kH; ++c) {
    const size_t at = static_cast<size_t>(after_span * kH + c);
    outside[at] = vt::F32ToBF16(vt::BF16ToF32(saved[at]) + 2.0F);
  }
  backend.Copy(queue, buffers.mm.inputs_embeds.data, outside.data(), bytes);
  backend.Synchronize(queue);
  const std::vector<float> control = forward();
  int64_t control_moved = 0;
  for (size_t i = 0; i < control.size(); ++i) {
    if (control[i] != before[i]) ++control_moved;
  }
  CHECK(control_moved == 0);

  backend.Copy(queue, buffers.mm.inputs_embeds.data, saved.data(), bytes);
  backend.Synchronize(queue);
}

// ───────────────────────────────────────────────────────────────────────────
// (10b) A CHUNK CARRYING THE INTERIOR OF AN IMAGE BLOCK IS REFUSED, at the
// forward.
//
// `DeepseekV4ImageSpans` refused two of the three chunk shapes and returned
// SILENTLY on the third. A chunk holding a START with no END, and one holding
// an END with no START, each threw. A chunk holding NEITHER -- the middle rows
// of a long image block -- opened no span, closed none, and produced an empty
// list for a step whose every row is an image row.
//
// The consequence is not an exception, it is silence: with no span the
// visible-row rule falls back to the ordinary sliding window OVER IMAGE ROWS,
// and the paged refusal keys on a non-empty span list so it does not fire
// either, while `media_rows > 0` still applies the vision routing bias. The
// answer stays fluent and half the image is invisible.
//
// It is reachable from the request path W5 is wiring:
// `SchedulerConfig::disable_chunked_mm_input` defaults to FALSE and
// `gather_mm_embeddings` handles a partial span. This case drives the shape
// through `ModelRegistry::Forward` rather than through the helper, because
// `test_deepseek_v4_dsa` already holds the helper and what is owed here is that
// a served step meets the refusal.
TEST_CASE("REACH: a chunk holding only the INTERIOR of an image block is refused") {
  auto loaded = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                    /*with_mmproj=*/true);
  const DeepSeekV4VisionConfig vcfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
  const auto image = MakeImage(vcfg);
  const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
      {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
      static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const vllm::MmEncoderOutput enc = vllm::ModelRegistry::EncodeMm(
      *loaded->model, loaded->config, queue, mm.mm_features[0]);

  // THE CHUNK: the rows strictly BETWEEN the two markers, which is exactly what
  // a prefill split inside the block hands the model. The markers are found
  // rather than assumed to be first and last, because
  // `BuildDeepSeekV4ImageBlock` writes `compress_pad` PAD rows ahead of the
  // start marker -- so "drop the first row" would leave the marker in and this
  // case would read the OTHER refusal.
  const int64_t block_begin = mm.mm_features[0].offset;
  const int64_t block_len = mm.mm_features[0].length;
  const int32_t start_id = static_cast<int32_t>(
      kVocab + static_cast<int64_t>(vllm::multimodal::kImageStart));
  const int32_t end_id = static_cast<int32_t>(
      kVocab + static_cast<int64_t>(vllm::multimodal::kImageEnd));
  int64_t start_at = -1, end_at = -1;
  for (int64_t i = 0; i < block_len; ++i) {
    const int32_t id = mm.prompt_token_ids[static_cast<size_t>(block_begin + i)];
    if (id == start_id) start_at = block_begin + i;
    if (id == end_id) end_at = block_begin + i;
  }
  REQUIRE(start_at >= 0);
  REQUIRE(end_at > start_at + 1);
  const int64_t chunk_begin = start_at + 1;
  const int64_t chunk = end_at - chunk_begin;
  std::vector<int32_t> ids(
      mm.prompt_token_ids.begin() + static_cast<long>(chunk_begin),
      mm.prompt_token_ids.begin() + static_cast<long>(chunk_begin + chunk));
  for (const int32_t id : ids) {
    REQUIRE(id >= static_cast<int32_t>(kVocab));
    REQUIRE(id != start_id);
    REQUIRE(id != end_id);
  }

  // Its embeddings, taken from the encoder rows that belong to those positions.
  const std::vector<char> is_mm(static_cast<size_t>(chunk), 1);
  const vt::Tensor slice = vt::Tensor::Contiguous(
      enc.embeds.Ptr<uint16_t>() + (chunk_begin - block_begin) * kH,
      vt::DType::kBF16, queue.device, {chunk, kH});
  const std::vector<vt::Tensor> slices{slice};
  vllm::MmEmbedInputs embed_in;
  embed_in.token_ids = &ids;
  embed_in.mm_embeds = &slices;
  embed_in.is_mm_embed = &is_mm;
  vllm::MmForwardBuffers buffers = vllm::ModelRegistry::EmbedMm(
      *loaded->model, loaded->config, queue, embed_in);

  std::vector<int32_t> positions(static_cast<size_t>(chunk));
  for (int64_t t = 0; t < chunk; ++t) {
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(chunk_begin + t);
  }
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(chunk - 1)};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {0};
  vllm::ModelForwardInput in{.token_ids = ids,
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
  in.mm = buffers.mm;

  std::string message;
  try {
    (void)vllm::ModelRegistry::Forward(*loaded->model, in);
  } catch (const std::exception& e) {
    message = e.what();
  }
  INFO("message: ", message);
  // It names the row, why the step cannot be answered, the atomicity
  // requirement and the issue.
  //
  // THE WORDING IS THE IN-LOOP RULE's, and it is asserted rather than left
  // loose because it says WHICH rule refused. This chunk's first row is an
  // `kImage` sentinel, not a pad, so it is refused where the loop reads it --
  // at row 0, before any trailing accounting can run. A message that spoke of
  // the step as a whole would mean the loop had passed the row and something
  // later caught it, which is a different guarantee.
  CHECK(message.find("image span") != std::string::npos);
  CHECK(message.find("the image row at row 0") != std::string::npos);
  CHECK(message.find("outside every complete image block") != std::string::npos);
  CHECK(message.find("scheduled whole") != std::string::npos);
  CHECK(message.find("2411") != std::string::npos);
}

// ───────────────────────────────────────────────────────────────────────────
// (11) THE WINDOW ITSELF, at the forward.
//
// The image-span rule is an exemption FROM the sliding window, so the window
// has to be reachable for the exemption to mean anything. `deepseek4.attention
// .sliding_window` is 128 on the released artifact, and this branch of the
// forward attended the FULL prefix regardless until W4 -- which #2323 already
// recorded as a divergence for the paged arm: "attending the full prefix there
// diverges above the window".
//
// Two models differing in NOTHING but that key answer the same prompt. Under a
// window of 4 a token ten rows back is invisible, so changing it cannot move
// the last row's logits; with the key absent it must.
TEST_CASE("REACH: the sliding window reaches the registered forward") {
  const auto run = [&](int64_t window, int32_t first_token) {
    TempFile lang(BuildDeepseek4Gguf(
        /*vision=*/false, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
        /*head_dim=*/512, /*with_tokenizer=*/false, /*vision_bias_scale=*/1.0F,
        window));
    const vllm::GgufFile gguf = vllm::GgufFile::Open(lang.path());
    const vllm::HfConfig config = vllm::DeepseekV4HfConfigFromGguf(gguf);
    std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(
        config, vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kCPU));
    vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
    vt::Queue queue = backend.CreateQueue();
    std::vector<int32_t> ids(12);
    std::vector<int32_t> positions(12);
    for (int32_t t = 0; t < 12; ++t) {
      ids[static_cast<size_t>(t)] = 1 + (t % 5);
      positions[static_cast<size_t>(t)] = t;
    }
    ids[0] = first_token;
    const std::vector<int32_t> logits_indices{11};
    std::vector<vllm::PagedKvCache> attn_kv;
    std::vector<vllm::GdnStateCache> gdn_state;
    const vllm::v1::GDNAttentionMetadata gdn_meta{};
    vllm::v1::CommonAttentionMetadata attn_meta{};
    attn_meta.num_reqs = 1;
    attn_meta.num_computed_tokens_cpu = {0};
    vllm::ModelForwardInput in{.token_ids = ids,
                               .positions = positions,
                               .attn_meta = attn_meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = attn_kv,
                               .gdn_state = gdn_state,
                               .config = config,
                               .queue = queue,
                               .logits_indices = logits_indices,
                               .num_reqs = 1};
    in.gather_logits = false;
    return vllm::ModelRegistry::Forward(*model, in).host;
  };

  const auto moved = [](const std::vector<float>& a, const std::vector<float>& b) {
    int64_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) ++n;
    }
    return n;
  };

  // With NO window the row eleven positions back is part of the prefix.
  CHECK(moved(run(/*window=*/0, /*first_token=*/1),
              run(/*window=*/0, /*first_token=*/6)) > 0);
  // With a window of four it is not, and the same edit cannot be seen.
  CHECK(moved(run(/*window=*/4, /*first_token=*/1),
              run(/*window=*/4, /*first_token=*/6)) == 0);
}

// ───────────────────────────────────────────────────────────────────────────
// (12) TEXT INERTNESS, measured rather than read.
//
// A DeepSeek-V4 TEXT checkpoint must be exactly what it was before this wave.
// The claim cannot be checked against code that no longer exists, so it is
// checked against the OTHER checkpoint: a vision file differs from a text file
// in nothing but its 43 `exp_probs_b_vl` tensors and its projector, and a
// text-only prompt must get the same logits from both, bit for bit.
//
// THE MUTATION THIS CASE ANSWERS TO. Key the vision bias on the CHECKPOINT --
// `!L.gate_bias_vl.empty()` -- instead of on the row's identifier, and the
// vision model's text answer moves while the text model's does not. That is a
// real shape of this defect: the bias is a property of the file, the rows are
// not, and the two are easy to confuse.
TEST_CASE("REACH: a text prompt is bit-identical on a text and a vision checkpoint") {
  auto text = LoadThroughRegistry(/*vision_checkpoint=*/false,
                                  /*with_mmproj=*/false);
  auto vision = LoadThroughRegistry(/*vision_checkpoint=*/true,
                                    /*with_mmproj=*/true);
  const auto& text_model = vllm::ModelAs<vllm::DeepseekV4LoadedModel>(
      *text->model, "DeepseekV4ForCausalLM");
  const auto& vision_model = vllm::ModelAs<vllm::DeepseekV4LoadedModel>(
      *vision->model, "DeepseekV4ForCausalLM");
  // NO VISION ALLOCATION on the text side, and one on the other -- which is
  // what makes the comparison below a comparison of two different files.
  CHECK_FALSE(text_model.has_vision());
  CHECK(vision_model.has_vision());
  CHECK(text_model.weights().host.layers[0].gate_bias_vl.empty());
  CHECK_FALSE(vision_model.weights().host.layers[0].gate_bias_vl.empty());

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const std::vector<int32_t> prompt{1, 2, 3, 4, 5, 6};
  const std::vector<int32_t> positions{0, 1, 2, 3, 4, 5};
  const std::vector<int32_t> logits_indices{5};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {0};
  const auto answer = [&](Loaded& loaded) {
    vllm::ModelForwardInput in{.token_ids = prompt,
                               .positions = positions,
                               .attn_meta = attn_meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = attn_kv,
                               .gdn_state = gdn_state,
                               .config = loaded.config,
                               .queue = queue,
                               .logits_indices = logits_indices,
                               .num_reqs = 1};
    in.gather_logits = false;
    // `mm` stays unset, which is every text step.
    return vllm::ModelRegistry::Forward(*loaded.model, in).host;
  };
  const std::vector<float> a = answer(*text);
  const std::vector<float> b = answer(*vision);
  REQUIRE(a.size() == static_cast<size_t>(kVocab));
  REQUIRE(b.size() == a.size());
  int64_t differing = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) ++differing;
  }
  CHECK(differing == 0);
}

// ───────────────────────────────────────────────────────────────────────────
// (12b) THE PAGED ARM SERVES A WINDOWED TEXT STEP, AND MUST KEEP SERVING IT.
//
// Case (13) below drives the refusal in the direction that fires. This case
// drives the direction that MUST NOT, and until it existed nothing did: no test
// ran a paged DeepSeek-V4 step at a non-zero `sliding_window` on a text prompt.
//
// The consequence of that hole is not hypothetical. Detach the refusal in
// `deepseek_v4.cpp` from `be.image_spans` -- refuse on the WINDOW alone -- and
// the whole family stays green while every TEXT step of the multi-KV arm is
// refused at the released `sliding_window = 128`. That arm is the one a real
// engine takes, because DeepSeek-V4 publishes a multi-cache topology, so the
// widened predicate would take the served text path down with it.
//
// A TEXT checkpoint and a text prompt, so `image_spans` is empty by
// construction and the refusal has nothing to key on but the window.
TEST_CASE("REACH: the paged arm SERVES a windowed text step") {
  TempFile lang(BuildDeepseek4Gguf(
      /*vision=*/false, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/false, /*vision_bias_scale=*/1.0F,
      /*sliding_window=*/4));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(lang.path());
  const vllm::HfConfig config = vllm::DeepseekV4HfConfigFromGguf(gguf);
  REQUIRE(config.raw.at("sliding_window").get<int64_t>() == 4);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(
      config, vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kCPU));

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();

  // The SAME page publication case (13) uses, so the two differ in the prompt
  // and in nothing else.
  const vllm::DeepseekV4Params params = vllm::ParseDeepseekV4Params(config);
  const int64_t nlayers = params.num_hidden_layers;
  const int64_t nb = 4, bs = 8;
  std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
  std::vector<vllm::PagedKvCache> attn_kv(static_cast<size_t>(nlayers));
  std::vector<std::string> names;
  for (int64_t l = 0; l < nlayers; ++l) {
    const size_t i = static_cast<size_t>(l);
    storage[i].assign(static_cast<size_t>(nb * bs * params.head_dim), 0.0F);
    attn_kv[i].data = storage[i].data();
    attn_kv[i].dtype = vt::DType::kF32;
    attn_kv[i].num_blocks = nb;
    attn_kv[i].block_size = bs;
    attn_kv[i].num_kv_heads = 1;
    attn_kv[i].head_size = static_cast<int>(params.head_dim);
    names.push_back("model.layers." + std::to_string(l) + ".attn.swa_cache");
  }
  vllm::MultiKvCacheIndex mk;
  mk.layer_names = &names;

  const int64_t tokens = 12;
  std::vector<int32_t> ids(static_cast<size_t>(tokens));
  std::vector<int32_t> positions(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) {
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(1 + (t % 5));
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(tokens - 1)};
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {0};
  vllm::ModelForwardInput in{.token_ids = ids,
                             .positions = positions,
                             .attn_meta = attn_meta,
                             .gdn_meta = gdn_meta,
                             .attn_kv = attn_kv,
                             .gdn_state = gdn_state,
                             .config = config,
                             .queue = queue,
                             .logits_indices = logits_indices,
                             .num_reqs = 1};
  in.gather_logits = false;
  in.multi_kv = &mk;
  // `mm` stays unset, which is every text step.
  CHECK_FALSE(in.mm.has_value());

  // It SERVES. Asserting the answer rather than only the absence of a throw:
  // an arm that returned an empty or non-finite row would satisfy a bare
  // `CHECK_NOTHROW` and would not be serving anything.
  std::string thrown;
  vllm::ForwardLogits out;
  try {
    out = vllm::ModelRegistry::Forward(*model, in);
  } catch (const std::exception& e) {
    thrown = e.what();
  }
  INFO("thrown: ", thrown);
  CHECK(thrown.empty());
  REQUIRE(out.host.size() == static_cast<size_t>(kVocab));
  int64_t nonfinite = 0;
  for (const float v : out.host) {
    if (!std::isfinite(v)) ++nonfinite;
  }
  CHECK(nonfinite == 0);
}

// ───────────────────────────────────────────────────────────────────────────
// (13) THE PAGED ARM REFUSES AN IMAGE SPAN IT CANNOT SERVE.
//
// The paged attention op takes ONE `vt::AttentionWindow` for the whole call, so
// it cannot express a per-position exemption. With the released
// `sliding_window = 128` against a 384-token block, clipping the span away
// leaves two thirds of it invisible and the argmax plausible -- the failure no
// token gate can see. It is refused by name instead, and the per-position mask
// is owed with the device path.
//
// The case matters because the paged arm is the one a REAL engine takes:
// DeepSeek-V4 publishes a multi-cache topology, so `ModelRegistry::Forward`
// routes a served step here and not to the branch cases (4) and (10) drive.
TEST_CASE("REACH: the paged arm refuses an image span it would clip to the window") {
  auto loaded = std::make_unique<Loaded>();
  loaded->lang = std::make_unique<TempFile>(BuildDeepseek4Gguf(
      /*vision=*/true, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/false, /*vision_bias_scale=*/1.0F,
      /*sliding_window=*/4));
  loaded->proj = std::make_unique<TempFile>(
      dsv4_mmproj_test::Build(ProjDims(), ProjOptions()));
  loaded->lang_gguf =
      std::make_unique<vllm::GgufFile>(vllm::GgufFile::Open(loaded->lang->path()));
  loaded->proj_gguf =
      std::make_unique<vllm::GgufFile>(vllm::GgufFile::Open(loaded->proj->path()));
  loaded->config = vllm::DeepseekV4HfConfigFromGguf(*loaded->lang_gguf);
  vllm::ModelSource source =
      vllm::ModelSource::FromGguf(*loaded->lang_gguf, vt::DeviceType::kCPU);
  source.mmproj = loaded->proj_gguf.get();
  source.mmproj_path = loaded->proj->path();
  loaded->model = vllm::ModelRegistry::Load(loaded->config, source);
  REQUIRE(loaded->config.raw.at("sliding_window").get<int64_t>() == 4);

  const DeepSeekV4VisionConfig vcfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
  const auto image = MakeImage(vcfg);
  const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
      {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
      static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));

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
      vllm::ModelRegistry::EmbedMm(*loaded->model, loaded->config, queue, embed_in);

  // The pages the runner publishes, under the names `MakeDeepseekV4KVCache`
  // publishes them under. Every layer of this fixture has `compress_ratio 0`,
  // so the SWA group is the whole topology.
  const vllm::DeepseekV4Params params = vllm::ParseDeepseekV4Params(loaded->config);
  const int64_t nlayers = params.num_hidden_layers;
  const int64_t nb = 4, bs = 8;
  std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
  std::vector<vllm::PagedKvCache> attn_kv(static_cast<size_t>(nlayers));
  std::vector<std::string> names;
  for (int64_t l = 0; l < nlayers; ++l) {
    const size_t i = static_cast<size_t>(l);
    storage[i].assign(static_cast<size_t>(nb * bs * params.head_dim), 0.0F);
    attn_kv[i].data = storage[i].data();
    attn_kv[i].dtype = vt::DType::kF32;
    attn_kv[i].num_blocks = nb;
    attn_kv[i].block_size = bs;
    attn_kv[i].num_kv_heads = 1;
    attn_kv[i].head_size = static_cast<int>(params.head_dim);
    names.push_back("model.layers." + std::to_string(l) + ".attn.swa_cache");
  }
  vllm::MultiKvCacheIndex mk;
  mk.layer_names = &names;

  std::vector<int32_t> positions(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) {
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(tokens - 1)};
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {0};
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
  in.multi_kv = &mk;
  in.mm = buffers.mm;

  std::string message;
  try {
    (void)vllm::ModelRegistry::Forward(*loaded->model, in);
  } catch (const std::exception& e) {
    message = e.what();
  }
  CHECK(message.find("image span") != std::string::npos);
  CHECK(message.find("sliding_window 4") != std::string::npos);
  CHECK(message.find("2411") != std::string::npos);
}

// ───────────────────────────────────────────────────────────────────────────
// (14) AN INTERIOR PREFILL CHUNK IS REFUSED AT `ModelRegistry::Forward`.
//
// W4 enforced chunk atomicity for two of the three shapes a cut produces: a
// chunk with START and no END, and one with END and no START. A chunk cut from
// the MIDDLE of one block carries NEITHER, so both checks were silent and the
// step was served with zero spans -- which put the visibility rule back on the
// ordinary sliding window over image rows, AND left the paged arm's
// non-empty-span refusal unarmed, while the routing bias still applied because
// it reads the identifiers. Every signal but the answer looked right.
//
// It is reachable rather than hypothetical: `disable_chunked_mm_input` defaults
// to false, `Scheduler::try_schedule_encoder_inputs` only rolls a step back
// when it is set, and nothing in this tree can set it. W5 wires the request
// path, so a long prompt carrying an image produces exactly this step.
//
// This case enters through `ModelRegistry::Forward` on the SAME merged buffers
// case (4) uses, with the token slice a middle chunk would carry.
TEST_CASE("REACH: an interior prefill chunk of an image block is refused") {
  auto loaded = LoadThroughRegistry(true, true);
  const DeepSeekV4VisionConfig vcfg =
      vllm::DeepSeekV4ClipMmprojVisionConfig(*loaded->proj_gguf);
  const auto image = MakeImage(vcfg);
  const MultiModalInputs mm = vllm::multimodal::PrepareDeepSeekV4Inputs(
      {1, 2, static_cast<int32_t>(kVocab) - 1, 3},
      static_cast<int32_t>(kVocab) - 1, {{image, "reach-image"}}, ProcCfg(vcfg));
  const vllm::multimodal::MultiModalFeatureSpec& f = mm.mm_features[0];
  // The PREMISE: the block is long enough to have an interior at all, so the
  // slice below really does drop both the start and the end identifier.
  REQUIRE(f.length >= 4);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const vllm::MmEncoderOutput enc = vllm::ModelRegistry::EncodeMm(
      *loaded->model, loaded->config, queue, f);

  // The chunk a scheduler hands the runner when the boundary falls inside the
  // block: the rows strictly BETWEEN the start and the end identifier. The two
  // are located rather than assumed, because `build_image_block` writes
  // `3 - offset % 4` leading pad rows before the start, so their indices are a
  // function of the offset.
  const int32_t start_id =
      static_cast<int32_t>(kVocab) +
      static_cast<int32_t>(vllm::multimodal::kImageStart);
  const int32_t end_id = static_cast<int32_t>(kVocab) +
                         static_cast<int32_t>(vllm::multimodal::kImageEnd);
  int64_t start_at = -1, end_at = -1;
  for (int i = 0; i < f.length; ++i) {
    const int32_t id = mm.prompt_token_ids[static_cast<size_t>(f.offset + i)];
    if (id == start_id) start_at = f.offset + i;
    if (id == end_id) end_at = f.offset + i;
  }
  REQUIRE(start_at >= 0);
  REQUIRE(end_at > start_at + 1);
  const int64_t begin = start_at + 1;
  const int64_t end = end_at;
  const std::vector<int32_t> chunk(mm.prompt_token_ids.begin() + begin,
                                   mm.prompt_token_ids.begin() + end);
  const int64_t tokens = static_cast<int64_t>(chunk.size());
  REQUIRE(tokens > 0);
  for (const int32_t id : chunk) {
    REQUIRE(id >= static_cast<int32_t>(kVocab));  // every row is an image row
    REQUIRE(id != start_id);
    REQUIRE(id != end_id);
  }

  // The merged rows for exactly this chunk, produced by the production hook
  // over the encoder-output SLICE the runner would gather for it
  // (`gather_mm_embeddings` narrows the item's rows to the chunk with its own
  // start/end index; this is that narrowing, by hand, on the same tensor).
  vt::Tensor slice = enc.embeds;
  slice.data = static_cast<uint16_t*>(enc.embeds.data) +
               (begin - f.offset) * kH;
  slice.shape[0] = tokens;
  std::vector<char> is_mm(static_cast<size_t>(tokens), 1);
  const std::vector<vt::Tensor> slices{slice};
  vllm::MmEmbedInputs embed_in;
  embed_in.token_ids = &chunk;
  embed_in.mm_embeds = &slices;
  embed_in.is_mm_embed = &is_mm;
  vllm::MmForwardBuffers buffers = vllm::ModelRegistry::EmbedMm(
      *loaded->model, loaded->config, queue, embed_in);

  std::vector<int32_t> positions(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) {
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(begin + t);
  }
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(tokens - 1)};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {static_cast<int32_t>(begin)};
  vllm::ModelForwardInput in{.token_ids = chunk,
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
  in.mm = buffers.mm;

  std::string message;
  try {
    (void)vllm::ModelRegistry::Forward(*loaded->model, in);
  } catch (const std::exception& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find("image row") != std::string::npos);
  CHECK(message.find("outside every complete image block") != std::string::npos);
  CHECK(message.find("2411") != std::string::npos);
}
