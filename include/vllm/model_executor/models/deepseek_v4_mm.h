// DeepSeek-V4-Flash-Vision — the loaded model and the vision runtime it carries
// (row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W4, issue
// [#2411](https://github.com/mudler/vllm.cpp/issues/2411)).
//
// WHY THIS FILE EXISTS. `DeepseekV4LoadedModel` was a private type inside
// `deepseek_v4_registry.cpp`, which is correct while a concrete model is only
// ever seen through `LoadedModel`. W4 gives it a second thing to own -- the
// projector this load was given and the tower built from it -- and the row's
// gate has to be able to ask whether a text checkpoint stayed tower-free. That
// question cannot be asked through the base class, so the type is published
// here rather than reached by a downcast into an anonymous namespace.
//
// WHAT `## Owed` SAID BEFORE THIS WAVE. W2's tower, W3A's `deepseek4v` reader
// and W1's processor each landed with "no production call site" recorded
// against them in `.agents/specs/deepseek-v4-flash-vision.md`. The runtime
// below is that call site: `LoadDeepseekV4ForCausalLM` fills it from
// `ModelSource::mmproj`, and the registered `encode_mm` hook runs it.
//
// THE TOWER IS BUILT ON FIRST USE, not at load. `multimodal::DeepSeekV4Vision`
// takes a `vt::Backend&`, and the weight loader has no queue in hand -- the
// same reason `StageDeepseekV4Exl3TowerToDevice` runs from the forward rather
// than from the loader. The projector's host weights are read at load time, so
// a file this build cannot read still costs a message before any language
// weight byte.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/clip_mmproj_gguf.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_vision.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vt/backend.h"

namespace vllm {

// The vision group one load was given, and the tower over it.
//
// `projector` OWNS the host storage; `weights` inside it holds non-owning views
// into that storage, and `tower` holds a copy of those views. So the projector
// must outlive the tower, which is what keeping both in one struct guarantees.
//
// ITS NAME COMES FROM THE VEHICLE THAT NEEDED IT FIRST. The same struct now
// holds the OFFICIAL safetensors vision group as well as a `deepseek4v`
// projector, because the two vehicles differ in where the bytes live and in
// nothing the tower can observe. See `LoadDeepseekV4VisionRuntime`.
struct DeepseekV4VisionRuntime {
  multimodal::DeepSeekV4VisionConfig config;
  DeepSeekV4ClipMmproj projector;
  // Null until the first `encode_mm`. See the file header for why the load
  // cannot build it.
  std::unique_ptr<multimodal::DeepSeekV4Vision> tower;

  DeepseekV4VisionRuntime() = default;
  DeepseekV4VisionRuntime(DeepseekV4VisionRuntime&&) = default;
  DeepseekV4VisionRuntime& operator=(DeepseekV4VisionRuntime&&) = default;
  DeepseekV4VisionRuntime(const DeepseekV4VisionRuntime&) = delete;
  DeepseekV4VisionRuntime& operator=(const DeepseekV4VisionRuntime&) = delete;
};

class DeepseekV4LoadedModel final : public LoadedModel {
 public:
  DeepseekV4LoadedModel(const ModelRegistration& registration,
                        DeepseekV4Weights weights,
                        std::unique_ptr<DeepseekV4VisionRuntime> vision = nullptr)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        vision_(std::move(vision)) {}

  const DeepseekV4Weights& weights() const { return weights_; }

  // MODEL-DSV4-PAGED-ENTRY (#2447): the compressor's carried state, which must
  // survive between steps -- it pools a CLOSED window into one row, so a state
  // rebuilt per call has seen nothing and `CompressorLayerStep` refuses on the
  // first decode step. Sized on first use, because the layer count comes from
  // the parsed params rather than from the registration.
  //
  // A STAGED SHORTCUT, DECLARED AS ONE. Upstream keeps this state in the
  // runner's KV-cache pool, and `MakeDeepseekV4KVCache` ALREADY publishes three
  // compressor-state groups (`c4_attn_state`, `c4_indexer_state`,
  // `c128_attn_state`) that nothing reads yet. A model-object member is ONE
  // sequence's state by construction, which is also why the route refuses
  // `num_reqs > 1`. Consuming the published groups is the correct end state and
  // is owed in `.agents/specs/model-dsv4-paged-entry.md` `## Owed`.
  //
  // No `mutable` is needed: the forward hook takes `LoadedModel&` non-const and
  // `ModelAs<T>` returns non-const. Precedent: `Qwen3MoeLoadedModel::decode_graph()`.
  DeepseekV4CompressorState& compressor_state(int64_t num_hidden_layers) {
    if (static_cast<int64_t>(compressor_.state_kv.size()) != num_hidden_layers) {
      compressor_.Resize(num_hidden_layers);
    }
    return compressor_;
  }

  // Did this load attach a vision tower? FALSE for a DeepSeek-V4 TEXT
  // checkpoint and for any load that named no `--mmproj`, and the whole
  // text-inertness claim rests on it: nothing else was read, allocated or
  // built.
  bool has_vision() const { return vision_ != nullptr; }

  const DeepseekV4VisionRuntime& vision() const;

  // The tower, built on first use against `backend`. REFUSES BY NAME when this
  // model carries no projector, because an absent tower is indistinguishable
  // downstream from an encoder that ran and produced nothing -- and that
  // failure splices zeros over the image span and answers fluently.
  multimodal::DeepSeekV4Vision& vision_tower(vt::Backend& backend);

 private:
  DeepseekV4Weights weights_;
  DeepseekV4CompressorState compressor_;
  std::unique_ptr<DeepseekV4VisionRuntime> vision_;
};

// THE PRODUCTION READ of whichever vehicle carries this load's vision group.
// Returns null when neither does: a GGUF load that named no `--mmproj`, a
// safetensors checkpoint with no `vision.*` group (every DeepSeek-V4 TEXT
// checkpoint), a projector of another family, or an engine whose multimodal
// limits put every modality this tower serves at zero (`SkipTowerForModalities`,
// the mirror of `interfaces.py:288-293`).
//
// TWO ARMS, ONE RUNTIME. `--mmproj` reads the `deepseek4v` projector of the
// shipped two-file GGUF vehicle; a safetensors source reads the OFFICIAL BF16
// vision group out of the model's own shards. Both fill the same
// `DeepSeekV4ClipMmproj` owner and the same W2 weight views, so everything
// above this function is indifferent to which vehicle was fed.
//
// It REFUSES BY NAME otherwise, in the one order the refusals may run in --
// `LoadDeepSeekV4ClipMmprojArm` holds that order and this function does not
// restate it.
std::unique_ptr<DeepseekV4VisionRuntime> LoadDeepseekV4VisionRuntime(
    const ModelSource& source, const HfConfig& config);

// ─── The OFFICIAL safetensors vision arm (`deepseek_v4_vision_weights.cpp`) ──
//
// The released `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` checkpoint carries its
// 267-tensor vision group in its own shards rather than in a second file. These
// four functions are that vehicle's counterpart to the `deepseek4v` mmproj
// reader's `DeepSeekV4ClipMmprojVisionConfig` / `...ExpectedTensors` /
// `LoadDeepSeekV4VisionFromClipMmproj` trio.

// The vision geometry the released `config.json` resolves. Refuses an absent or
// absurd value BY THE KEY that carried it. `output_size` is the LANGUAGE
// model's hidden size, because the aligner lands in the text hidden space.
multimodal::DeepSeekV4VisionConfig DeepSeekV4OfficialVisionConfig(
    const HfConfig& config);

// Does this checkpoint carry the official vision group at all? False for every
// DeepSeek-V4 TEXT checkpoint, which is what keeps the text arm tower-free.
// A checkpoint carrying only SOME of the group is not silently text: the loader
// then refuses the missing names one by one.
bool DeepSeekV4ShardsCarryVision(const std::vector<SafetensorsFile>& shards);

// The EXACT set of names the loader below reads for `config`: the patch
// embedding and its bias, `depth` blocks of eight, the final norm, the
// aligner's two weight/bias pairs and the four sentinels. At the released depth
// 32 that is 267, which is the pinned shard-1 header's vision tensor count.
std::vector<std::string> DeepSeekV4OfficialVisionExpectedTensors(
    const multimodal::DeepSeekV4VisionConfig& config);

// Read the official vision group into the W2 types. Every view in the result
// points into storage the result owns, so it survives the shards being closed.
DeepSeekV4ClipMmproj LoadDeepSeekV4VisionFromSafetensors(
    const std::vector<SafetensorsFile>& shards,
    const multimodal::DeepSeekV4VisionConfig& config);

// The registered `encode_mm` hook: `SupportsMultiModal.embed_multimodal`.
//
// It returns ONE ROW PER SENTINEL TOKEN of the image block, not one row per
// aligner cell. The runner indexes an encoder output by a token's offset inside
// its feature span (`gather_mm_embeddings`), so a hook that returned only the
// aligner rows would put the wrong vector under every marker as soon as a
// prefill was chunked, and would be invisible while it was not.
MmEncoderOutput EncodeMmDeepseekV4ForCausalLM(
    LoadedModel& model, const HfConfig& config, vt::Queue& queue,
    const multimodal::MultiModalFeatureSpec& item);

// The registered `embed_mm` hook: `SupportsMultiModal.embed_input_ids`.
//
// Embeds the ordinary identifiers and scatters the encoder rows over the masked
// positions, which is upstream's `_merge_multimodal_embeddings`. A masked row
// is NOT looked up: the expanded prompt spells it `vocab_size + type`, which is
// out of vocabulary, and the spec's data flow says such a row embeds to zero
// before the merge replaces it.
MmForwardBuffers EmbedMmDeepseekV4ForCausalLM(LoadedModel& model,
                                              const HfConfig& config,
                                              vt::Queue& queue,
                                              const MmEmbedInputs& inputs);

}  // namespace vllm
