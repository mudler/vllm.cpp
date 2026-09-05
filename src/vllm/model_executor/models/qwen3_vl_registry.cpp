// Qwen3-VL (`Qwen3VLForConditionalGeneration`) registry TU — MM-ENGINE-FORWARD.
//
// Self-registers "Qwen3VLForConditionalGeneration" via REGISTER_VLLM_MODEL so the
// ENGINE (ModelRegistry::Resolve/Load/Prepare/Forward) drives the vision-language
// model, instead of the standalone Qwen3VLGenerateGreedy driver running OUTSIDE
// the registry. This is the additive-TU seam (new TU + one REGISTER line → ZERO
// shared-array edit), exactly like olmo2_registry.cpp / qwen3_dense.cpp.
//
// The registered forward (ForwardQwen3VL) folds the M2c forked decode into the
// per-step ModelRegistry::Forward contract: it consumes the ALREADY-MERGED
// embeddings + 3-D MRoPE positions + DeepStack carried on ModelForwardInput.mm
// (the runner mm-path / Qwen3VLGenerateGreedyViaRegistry driver fills them via the
// vision tower + `_merge_multimodal_embeddings`), and calls the SHARED
// Qwen3VLForwardStepLastLogits — the same step the standalone driver runs, so the
// two paths are numerically identical by construction. mm is nullopt for text
// requests, so this registration cannot perturb any text model.
//
// Ported from vllm/model_executor/models/registry.py (the Qwen3-VL entry) +
// qwen3_vl.py forward — see qwen3_vl.h / qwen3_vl.cpp for the port map.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/interfaces.h"  // #607 L3 kVisionTowerStageName
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::ApplyDeviceTokenIds
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight/MakeTensor
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/model_executor/models/qwen3_vl_text.h"  // merge + deepstack + get_rope_index
#include "vllm/multimodal/inputs.h"  // MultiModalFeatureSpec
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Qwen3-VL: text generation + MULTIMODAL (vision
// tower). The 4B text backbone is a PLAIN dense full-attention model → NOT hybrid
// (no GDN linear-attention state). This is the FIRST non-hybrid multimodal
// registration (the two Qwen3.5 ConditionalGeneration wrappers are hybrid+mm).
inline constexpr ModelInfo kQwen3VLInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

class Qwen3VLLoadedModel final : public LoadedModel {
 public:
  Qwen3VLLoadedModel(const ModelRegistration& registration, Qwen3VLWeights weights)
      : LoadedModel(registration),
        owned_weights_(std::move(weights)),
        weights_(&*owned_weights_) {}
  Qwen3VLLoadedModel(const ModelRegistration& registration,
                     const Qwen3VLWeights& weights, BorrowedWeightsTag)
      : LoadedModel(registration), weights_(&weights) {}

  const Qwen3VLWeights& weights() const { return *weights_; }

  // #607 L3: the mirror of `_tower_model_names` + `StageMissingLayer`'s
  // stage_name (interfaces.py:141,279-282,298). Non-empty only when this load
  // deliberately left `model.visual.*` unread.
  std::vector<std::string> skipped_towers() const override {
    if (!weights_->vision_skipped) return {};
    return {std::string(kVisionTowerStageName)};
  }

  // Build-on-first-use persistent MRoPE cos|sin cache. Deterministic + built with
  // the SAME RopeArgs/Pmax as VLGenerateCore, so it is bit-identical to the
  // standalone driver's cache — the registered and standalone paths stay numeric-
  // identical. (The gate driver is single-threaded; the mutex keeps a stray
  // concurrent Prepare/Forward safe.)
  const Qwen3VLCosSinCache& CosSinCache(vt::Queue& queue, const HfConfig& config) {
    std::lock_guard<std::mutex> lock(cos_sin_mu_);
    if (!cos_sin_.storage) {
      cos_sin_ = Qwen3VLMakeCosSinCache(queue, config);
    }
    return cos_sin_;
  }

 private:
  std::optional<Qwen3VLWeights> owned_weights_;
  const Qwen3VLWeights* weights_ = nullptr;
  std::mutex cos_sin_mu_;
  Qwen3VLCosSinCache cos_sin_;
};

std::unique_ptr<LoadedModel> LoadQwen3VLForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Qwen3VLForConditionalGeneration does not support "
        "GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  // #607 L3: `source.multimodal` is the engine's limits, borrowed. Null on every
  // non-engine caller, which loads the vision tower exactly as before.
  return std::make_unique<Qwen3VLLoadedModel>(
      registration,
      LoadQwen3VLWeights(*source.safetensors, config, source.multimodal));
}

void PrepareQwen3VLForConditionalGeneration(LoadedModel& model,
                                            const HfConfig& config,
                                            vt::Queue& queue) {
  // Warm the persistent cos|sin cache so the first forward step does not build it.
  // The call stays INLINE on the checked reference rather than gaining a local:
  // `ModelAs` establishes the dynamic type before the member call either way, so
  // a binding would change this site's shape without changing what it does.
  ModelAs<Qwen3VLLoadedModel>(model, "Qwen3VLForConditionalGeneration")
      .CosSinCache(queue, config);
}

ForwardLogits ForwardQwen3VLForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  auto& vl = ModelAs<Qwen3VLLoadedModel>(model, "Qwen3VLForConditionalGeneration");
  VT_CHECK(input.mm.has_value(),
           "Qwen3VLForConditionalGeneration registered forward requires "
           "multimodal inputs (ModelForwardInput.mm). Text-only Qwen3-VL through "
           "this arch is a named MM-ENGINE-FORWARD residual.");
  const MultiModalForwardInput& mm = *input.mm;
  // ENG-MM-INPUT-PIPELINE P1: both handles are DEVICE views now. `deepstack` is
  // legitimately absent on a decode step and on a DeepStack-less VL checkpoint, so
  // it is not required unconditionally; the forward core checks the shapes it was
  // given.
  VT_CHECK(mm.inputs_embeds.data != nullptr && mm.positions3.data != nullptr,
           "Qwen3-VL mm forward: null merged-embeds / positions3 device handle on "
           "ModelForwardInput.mm");
  // `deepstack` and `deepstack_levels` are ONE channel expressed as two fields, so
  // they are checked together. Under the device contract ABSENT and NOT-APPLICABLE
  // are the SAME value (`data == nullptr`), which is exactly what the host contract
  // this replaced could not express: it required `deepstack_bf16 != nullptr` and the
  // driver always passed a pointer to a possibly-empty vector, so a forgotten
  // DeepStack was a loud throw. Left uncoupled, `VLForwardLastLogitsDBuf` computes
  // `has_ds = false` and silently SKIPS every multiscale-merger add — a prefill that
  // returns fluent WRONG tokens with no diagnostic. The REVERSE pairing (tensor set,
  // levels 0) needs nothing here: `has_ds` is then true and that function's
  // `Numel() == L * T * H` check refuses it against `0 * T * H`.
  VT_CHECK(mm.deepstack_levels == 0 || mm.deepstack.data != nullptr,
           "Qwen3-VL mm forward: MultiModalForwardInput.deepstack_levels is non-zero "
           "but .deepstack carries no device buffer. Set BOTH fields or NEITHER: an "
           "absent tensor is how the seam spells 'this step has no DeepStack', so "
           "leaving it unset here would skip every multiscale-merger add instead.");
  // ENG-MM-INPUT-PIPELINE P2 (#2379): REFUSE a batched step by name rather than
  // answer it wrong. Both entry points below return exactly ONE row — the last
  // token's logits — because that is what the M2c single-sequence driver needed
  // and `input.logits_indices` is not read here at all. That was unobservable
  // while the only caller was a single-sequence driver; the GPU runner is a
  // batching caller, and it hands the sampler a [num_reqs, vocab] tensor. With
  // two requests in one step the sampler would read one row and index past it,
  // which produces tokens rather than an error. The gap is a per-row gather in
  // the VL forward, owed under `specs/multimodal-track.md` `## Owed`.
  VT_CHECK(input.num_reqs <= 1,
           "Qwen3-VL serves one request per step: this registered forward "
           "returns only the LAST token's logits and does not read "
           "logits_indices, so a batched step cannot be answered correctly. "
           "Run the server with --max-num-seqs 1 for this architecture. "
           "ENG-MM-INPUT-PIPELINE P2 (#2379).");
  const int64_t num_tokens = mm.positions3.Numel() / 3;
  const Qwen3VLCosSinCache& cos_sin = vl.CosSinCache(input.queue, input.config);
  // DEVICE-resident logits (sampler-on-device) on the gather path — the mm forward
  // produces exactly the single last-token [1, vocab] row, kept ON DEVICE so the
  // greedy driver / runner samples it straight off device (vt::GreedyArgmax) with
  // no full-vocab D2H. Mirrors the text device path (qwen3_dense.cpp:86). The host
  // path (gather_logits=false) reproduces the old download-then-sample A/B.
  if (input.gather_logits) {
    return Qwen3VLForwardStepLastLogitsDevice(
        input.queue, vl.weights().text, input.config, mm.inputs_embeds,
        mm.positions3, num_tokens, mm.deepstack, mm.deepstack_levels,
        cos_sin.tensor, input.attn_meta, input.attn_kv);
  }
  std::vector<float> logits = Qwen3VLForwardStepLastLogits(
      input.queue, vl.weights().text, input.config, mm.inputs_embeds,
      mm.positions3, num_tokens, mm.deepstack, mm.deepstack_levels,
      cos_sin.tensor, input.attn_meta, input.attn_kv);
  return HostLogits(std::move(logits), input.config.vocab_size);
}

// ── ENG-MM-INPUT-PIPELINE P2 (#2379): the three MODEL hooks the runner calls ──
//
// Upstream's runner does not implement multimodal. It calls model methods
// through two protocols — `SupportsMultiModal.embed_multimodal`,
// `SupportsMultiModal.embed_input_ids` (interfaces.py:383, whose
// `multimodal_embeddings: MultiModalEmbeddings | None = None,` line `grep -c`
// == 1 in that file; `def embed_input_ids` itself is NOT unique, 322 hits
// `-rn vllm/`) and `SupportsMRoPE.get_mrope_input_positions`. On a type-erased
// `LoadedModel` those become three `ModelFactory` function pointers.
//
// WHAT THE RUNNER IS NOT TOLD. DeepStack. `grep -c deepstack` over upstream's
// `gpu_model_runner.py` is 0: the multiscale features ride INSIDE the tower
// output and are unpacked model-side. So `EncodeMmQwen3VL` returns ONE tensor
// [num_placeholder_rows, hidden * (1 + levels)] and the runner slices its ROWS
// without ever reading a column; `EmbedMmQwen3VL` is where the width is split.
// A runner that knew about DeepStack would be a runner that knows about one
// architecture.

// The host round-trip these two hooks pay, named rather than left to be found.
// The tower output is downloaded, the merge and the DeepStack scatter run on the
// host in f32, and the result is uploaded. That is EXACTLY the arithmetic the
// gated M2c driver runs (`Qwen3VLGenerateGreedy`), so the registered runner path
// is numerically identical to the path the golden tokens were measured on, which
// is the property worth having first. A device-resident merge is a measured
// change against that golden, not a cleanup, and it is owed rather than done
// here (specs/multimodal-track.md `## Owed`).

MmEncoderOutput EncodeMmQwen3VLForConditionalGeneration(
    LoadedModel& model, const HfConfig& config, vt::Queue& queue,
    const multimodal::MultiModalFeatureSpec& item) {
  auto& vl = ModelAs<Qwen3VLLoadedModel>(model, "Qwen3VLForConditionalGeneration");
  const Qwen3VLWeights& weights = vl.weights();
  VT_CHECK(weights.vision_loaded && !weights.vision_skipped,
           "Qwen3-VL encoder: this load carries no vision tower. A zero "
           "--limit-mm-per-prompt (or --language-model-only) skips it, and an "
           "image request is refused at the entrypoint long before here; "
           "reaching this point is a defect. ENG-MM-INPUT-PIPELINE P2 (#2379).");
  VT_CHECK(item.modality == "image",
           "Qwen3-VL encoder: modality '" + item.modality +
               "' is not wired through the runner. Only `image` is; video has a "
               "tower and a driver but no runner path yet, and audio has no "
               "tower at all. Recorded as owed under ENG-MM-INPUT-PIPELINE "
               "(#2379) rather than served wrong.");
  VT_CHECK(item.data != nullptr && !item.data->empty(),
           "Qwen3-VL encoder: multimodal item carries no processed image "
           "features (MultiModalFeatureSpec::data). ENG-MM-INPUT-PIPELINE P2.");

  // The tower's geometry is the CHECKPOINT's (`Qwen3VLWeights::vision_cfg`),
  // never the text config's, so `config` is unread here.
  (void)config;
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  const multimodal::Qwen3VLVisionConfig& vcfg = weights.vision_cfg;
  const int64_t levels = static_cast<int64_t>(vcfg.deepstack_visual_indexes.size());
  const int64_t out_hidden = vcfg.out_hidden_size;
  const int64_t width = out_hidden * (1 + levels);

  // THE TOWER. This call is the point of the row: before it, nothing in `src/`
  // ran a vision tower on a served request.
  const std::vector<float> tower = multimodal::Qwen3VLVisionForward(
      item.data->pixel_values_bf16, item.data->image_grid_thw, weights.vision,
      vcfg, backend);
  const int64_t rows = width > 0 ? static_cast<int64_t>(tower.size()) / width : 0;
  VT_CHECK(rows * width == static_cast<int64_t>(tower.size()) && rows > 0,
           "Qwen3-VL encoder: tower produced " + std::to_string(tower.size()) +
               " floats, which is not a whole number of " +
               std::to_string(width) + "-wide rows");
  VT_CHECK(rows == static_cast<int64_t>(item.length),
           "Qwen3-VL encoder: tower produced " + std::to_string(rows) +
               " embedding rows for a placeholder span of " +
               std::to_string(item.length) +
               " tokens. The processor's placeholder expansion and the tower's "
               "spatial merge disagree, and a masked scatter would then splice "
               "the wrong rows into the prompt.");

  // Stored in the MODEL dtype (bf16), which is also the rounding the gated
  // driver applies to the tower output before the merge (`RoundToBf16(main_bf)`
  // in Qwen3VLGenerateGreedy), so the two paths round in the same place.
  std::vector<uint16_t> bits(tower.size());
  for (size_t i = 0; i < tower.size(); ++i) bits[i] = vt::F32ToBF16(tower[i]);

  const size_t bytes = bits.size() * vt::SizeOf(vt::DType::kBF16);
  void* p = backend.Alloc(bytes);
  std::shared_ptr<void> storage(p, [&backend](void* q) { backend.Free(q); });
  backend.Copy(queue, p, bits.data(), bytes);
  MmEncoderOutput out;
  out.storage = std::move(storage);
  out.embeds = dense_attn::MakeTensor(p, vt::DType::kBF16, queue.device,
                                      {rows, width});
  return out;
}

MmForwardBuffers EmbedMmQwen3VLForConditionalGeneration(
    LoadedModel& model, const HfConfig& config, vt::Queue& queue,
    const MmEmbedInputs& inputs) {
  auto& vl = ModelAs<Qwen3VLLoadedModel>(model, "Qwen3VLForConditionalGeneration");
  const Qwen3VLWeights& weights = vl.weights();
  VT_CHECK(inputs.token_ids != nullptr && inputs.is_mm_embed != nullptr &&
               inputs.mm_embeds != nullptr,
           "Qwen3-VL embed: the runner passed a null MmEmbedInputs channel");
  const std::vector<int32_t>& token_ids = *inputs.token_ids;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  VT_CHECK(T > 0, "Qwen3-VL embed: empty step");
  VT_CHECK(static_cast<int64_t>(inputs.is_mm_embed->size()) == T,
           "Qwen3-VL embed: is_mm_embed has " +
               std::to_string(inputs.is_mm_embed->size()) + " entries for " +
               std::to_string(T) + " tokens");

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  dense_attn::Dev d{backend, queue};

  // `embed_input_ids`, the token half: the plain embedding lookup, identical to
  // the one the text path runs.
  std::vector<uint16_t> emb_bits(static_cast<size_t>(T * H));
  {
    dense_attn::DBuf ids(d, vt::DType::kI32, {T}, token_ids.data());
    // ENG-MM-EMBED-DEVICE-IDS (#2730): SPLICE the runner's device identifiers
    // over the host upload just made, before the gather reads it. The host
    // vector is deliberately stale for decode rows -- the combine wrote each
    // sampled token into `MmEmbedInputs::device_token_ids` on the main queue and
    // never wrote it back -- so without this the decode rows of an image request
    // embed token id 0. The `Copy` is enqueued on this same queue, which is why
    // it is ordered AFTER that combine rather than racing it, and it is the
    // SHARED splice the text device arm uses rather than a second copy of it.
    // Null on every non-mirror step, where it writes nothing and this hook is
    // byte-identical to its pre-#2730 self.
    detail::ApplyDeviceTokenIds(
        d.b, d.q, ids.ptr(), T,
        detail::DeviceTokenIds{inputs.device_token_ids, T}, "qwen3-vl mm embed");
    dense_attn::DBuf emb(d, vt::DType::kBF16, {T, H});
    vt::Tensor table = dense_attn::ResidentWeight(
        d, weights.text.embed_tokens, {config.vocab_size, H});
    vt::Embedding(d.q, emb.t(), table, ids.t());
    emb.Download(d, emb_bits.data());
  }

  // The gathered encoder rows, concatenated in mask order. Every slice is a
  // BORROWED device view the runner owns for the duration of this call.
  const int64_t levels = static_cast<int64_t>(
      weights.vision_cfg.deepstack_visual_indexes.size());
  const int64_t width = H * (1 + levels);
  int64_t n_rows = 0;
  for (const vt::Tensor& slice : *inputs.mm_embeds) {
    VT_CHECK(slice.rank == 2 && slice.shape[1] == width,
             "Qwen3-VL embed: a gathered encoder slice is " +
                 std::to_string(slice.shape[1]) + " wide, expected " +
                 std::to_string(width) + " (hidden * (1 + deepstack levels))");
    n_rows += slice.shape[0];
  }
  std::vector<bool> mask(static_cast<size_t>(T), false);
  int64_t n_masked = 0;
  for (int64_t t = 0; t < T; ++t) {
    const bool on = (*inputs.is_mm_embed)[static_cast<size_t>(t)] != 0;
    mask[static_cast<size_t>(t)] = on;
    if (on) ++n_masked;
  }
  VT_CHECK(n_rows == n_masked,
           "Qwen3-VL embed: " + std::to_string(n_rows) +
               " gathered encoder rows for " + std::to_string(n_masked) +
               " masked placeholder positions. A masked scatter that does not "
               "balance splices vision features onto text rows.");

  std::vector<float> embeds(static_cast<size_t>(T * H));
  for (size_t i = 0; i < embeds.size(); ++i) embeds[i] = vt::BF16ToF32(emb_bits[i]);

  std::vector<float> deepstack_bf;
  if (n_rows > 0) {
    // Download the slices and split the width: [:, :H] is the merger output the
    // masked scatter consumes, [:, H:] is the multiscale stream DeepStack needs.
    std::vector<uint16_t> gathered(static_cast<size_t>(n_rows * width));
    size_t offset = 0;
    for (const vt::Tensor& slice : *inputs.mm_embeds) {
      const size_t n = static_cast<size_t>(slice.shape[0] * width);
      backend.Copy(queue, gathered.data() + offset, slice.data,
                   n * vt::SizeOf(vt::DType::kBF16));
      offset += n;
    }
    backend.Synchronize(queue);

    std::vector<float> main(static_cast<size_t>(n_rows * H));
    std::vector<float> multiscale(static_cast<size_t>(n_rows * levels * H));
    for (int64_t r = 0; r < n_rows; ++r) {
      const size_t src = static_cast<size_t>(r * width);
      for (int64_t c = 0; c < H; ++c) {
        main[static_cast<size_t>(r * H + c)] =
            vt::BF16ToF32(gathered[src + static_cast<size_t>(c)]);
      }
      for (int64_t c = 0; c < levels * H; ++c) {
        multiscale[static_cast<size_t>(r * levels * H + c)] =
            vt::BF16ToF32(gathered[src + static_cast<size_t>(H + c)]);
      }
    }
    // `_merge_multimodal_embeddings`, the masked scatter.
    multimodal::Qwen3VLMergeMultimodal(embeds, T, H, main, mask);
    if (levels > 0) {
      deepstack_bf = multimodal::Qwen3VLComputeDeepstack(multiscale, n_rows,
                                                         levels, H, mask, T);
    }
  }

  std::vector<uint16_t> merged_bits(static_cast<size_t>(T * H));
  for (size_t i = 0; i < merged_bits.size(); ++i)
    merged_bits[i] = vt::F32ToBF16(embeds[i]);

  // Positions. M-RoPE is REQUIRED for this architecture: `ForwardQwen3VL` reads
  // `mm.positions3` and refuses a null handle, so an empty array here would be a
  // throw one frame later with a less useful message.
  VT_CHECK(inputs.mrope_positions != nullptr &&
               static_cast<int64_t>(inputs.mrope_positions->size()) == 3 * T,
           "Qwen3-VL embed: expected 3 x " + std::to_string(T) +
               " M-RoPE positions. Qwen3-VL declares `mrope_prompt_positions`, "
               "so the runner must have computed them for this step.");

  vt::Backend& b = backend;
  const auto upload = [&b, &queue](const void* host, size_t bytes,
                                   std::vector<std::shared_ptr<void>>* keep) {
    void* p = b.Alloc(bytes);
    keep->emplace_back(p, [&b](void* q) { b.Free(q); });
    b.Copy(queue, p, host, bytes);
    return p;
  };

  MmForwardBuffers out;
  void* p_embeds = upload(merged_bits.data(),
                          merged_bits.size() * vt::SizeOf(vt::DType::kBF16),
                          &out.storage);
  out.mm.inputs_embeds =
      dense_attn::MakeTensor(p_embeds, vt::DType::kBF16, queue.device, {T, H});
  void* p_pos = upload(inputs.mrope_positions->data(),
                       inputs.mrope_positions->size() * vt::SizeOf(vt::DType::kI32),
                       &out.storage);
  out.mm.positions3 =
      dense_attn::MakeTensor(p_pos, vt::DType::kI32, queue.device, {3, T});
  // DeepStack: BOTH fields or NEITHER. `ForwardQwen3VL` refuses a non-zero level
  // count with no tensor, and silently skips every multiscale add for the
  // reverse, so a decode step (no placeholder rows this window) leaves both
  // unset rather than declaring a level count it cannot back.
  if (!deepstack_bf.empty()) {
    std::vector<uint16_t> ds_bits(deepstack_bf.size());
    for (size_t i = 0; i < ds_bits.size(); ++i)
      ds_bits[i] = vt::F32ToBF16(deepstack_bf[i]);
    void* p_ds = upload(ds_bits.data(),
                        ds_bits.size() * vt::SizeOf(vt::DType::kBF16),
                        &out.storage);
    out.mm.deepstack = dense_attn::MakeTensor(p_ds, vt::DType::kBF16,
                                              queue.device, {levels, T, H});
    out.mm.deepstack_levels = levels;
  }
  b.Synchronize(queue);
  return out;
}

MropePromptPositions MropeQwen3VLForConditionalGeneration(
    LoadedModel& model, const HfConfig& config,
    const std::vector<int32_t>& prompt_token_ids,
    const std::vector<multimodal::MultiModalFeatureSpec>& mm_features) {
  auto& vl = ModelAs<Qwen3VLLoadedModel>(model, "Qwen3VLForConditionalGeneration");
  (void)config;
  std::vector<multimodal::MmImageSpan> images;
  images.reserve(mm_features.size());
  for (const multimodal::MultiModalFeatureSpec& item : mm_features) {
    VT_CHECK(item.modality == "image",
             "Qwen3-VL M-RoPE: modality '" + item.modality +
                 "' has no runner position path (see the encoder hook)");
    VT_CHECK(item.data != nullptr,
             "Qwen3-VL M-RoPE: multimodal item carries no processed features");
    images.push_back(multimodal::MmImageSpan{static_cast<int64_t>(item.offset),
                                             item.data->image_grid_thw});
  }
  MropePromptPositions out;
  int64_t delta = 0;
  out.positions = multimodal::Qwen3VLGetRopeIndex(
      prompt_token_ids, images, vl.weights().vision_cfg.spatial_merge_size,
      &delta);
  out.delta = delta;
  return out;
}

v1::KVCacheConfig MakeQwen3VLForConditionalGenerationKVCache(const HfConfig& config,
                                                            int block_size,
                                                            int num_blocks) {
  // The 4B VL text backbone is pure dense: exactly ONE full-attention KV group,
  // NO MambaSpec/GDN group (identical topology to Qwen3ForCausalLM).
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, num_kv_heads, head_dim,
                                              v1::ResolveKvCacheDType()));
  return kv;
}

void ParseQwen3VLForConditionalGenerationConfig(const HfConfig& config) {
  // LoadHfConfig already resolves the Qwen3-VL text_config onto the top-level
  // HfConfig (hidden 2560, 36 layers, head_dim 128, kv 8, rope_theta 5e6, tied).
  // No extra normalization needed — the seam for future validation.
  (void)config;
}

const ModelFactory kQwen3VLFactory{
    .parse_config = &ParseQwen3VLForConditionalGenerationConfig,
    .load_weights = &LoadQwen3VLForConditionalGeneration,
    .prepare = &PrepareQwen3VLForConditionalGeneration,
    .forward = &ForwardQwen3VLForConditionalGeneration,
    .make_kv_cache = &MakeQwen3VLForConditionalGenerationKVCache,
    // ENG-MM-INPUT-PIPELINE P2 (#2379): the three hooks the GPU runner calls to
    // fill `ModelForwardInput::mm`. Setting the first two is what makes
    // `ModelRegistry::SupportsMmInputs` true for this architecture, and the
    // runner's whole multimodal arm hangs on that.
    .encode_mm = &EncodeMmQwen3VLForConditionalGeneration,
    .embed_mm = &EmbedMmQwen3VLForConditionalGeneration,
    .mrope_prompt_positions = &MropeQwen3VLForConditionalGeneration,
    .is_dense_model = true,
    // ENG-MM-EMBED-DEVICE-IDS (#2730). BOTH bits, and the pair is the design.
    //
    // The HOOK bit is the plain statement: `EmbedMmQwen3VLForConditionalGeneration`
    // splices `MmEmbedInputs::device_token_ids` over the buffer it gathers from.
    //
    // The FORWARD bit is the one that needs its reason written down, because
    // `ForwardQwen3VLForConditionalGeneration` reads no token identifier at all.
    // It refuses a step without `input.mm` BY NAME and then embeds
    // `mm.inputs_embeds`. Through `ModelRegistry::Forward` the only way it is
    // reachable is from the runner branch that has ALREADY called
    // `ModelRegistry::EmbedMm` with the same device pointer on the same step --
    // `forward_input.mm` is assigned inside the same
    // `supports_mm_inputs() && batch_carries_mm()` window that produced it -- so
    // this registration's whole reachable path resolves its identifiers from the
    // device buffer. Without the bit the hook above would compute a correct
    // `inputs_embeds` and the forward's own guard would throw it away on the same
    // step, which is a fix nothing can reach.
    .consumes_device_token_ids = true,
    .embed_mm_consumes_device_token_ids = true,
};

}  // namespace

std::unique_ptr<LoadedModel> MakeQwen3VLLoadedModel(Qwen3VLWeights weights) {
  return std::make_unique<Qwen3VLLoadedModel>(
      RegistrationFor("Qwen3VLForConditionalGeneration"), std::move(weights));
}

std::unique_ptr<LoadedModel> BorrowQwen3VLLoadedModel(
    const Qwen3VLWeights& weights) {
  return std::make_unique<Qwen3VLLoadedModel>(
      RegistrationFor("Qwen3VLForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

REGISTER_VLLM_MODEL(qwen3_vl, "Qwen3VLForConditionalGeneration", kQwen3VLFactory,
                    kQwen3VLInfo)

}  // namespace vllm
