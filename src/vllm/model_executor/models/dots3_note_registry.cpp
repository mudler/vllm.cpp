// dots3-note registry TU — the ADDITIVE self-registration seam (W1). Follows
// the deepseek_v4_registry.cpp / gemma4_registry.cpp seam exactly: a NEW
// translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array.
//
// Upstream registers ONE architecture string onto this package:
//   registry.py:381 (_MULTIMODAL_MODELS)
//     "Dots3NoteForCausalLM": ("vllm.models.dots3_note", "Dots3NoteForCausalLM")
// and its speculative head separately:
//   registry.py:670 (_SPECULATIVE_DECODING_MODELS)
//     "Dots3NoteMTPModel": ("vllm.models.dots3_note", "Dots3NoteMTP")
// Read at vLLM `origin/main` = `c205726108df54bb6fbf15b19e725a4a3add2b18`.
// `dots3_note` does NOT exist at our parity pin `555967922` — see
// `.agents/porting-inventory.md` §9 and `.agents/specs/dots3-note.md` §6.1.
//
// `Dots3NoteMTPModel` is deliberately NOT registered here. It stays INVENTORIED
// on `MODEL-SPEC-dots3-note-dots3-note-mtp` (W10 owns it): registering a
// speculator that cannot propose would make the engine accept a speculative
// config it then dies on mid-run, which is the failure mode #442 already found
// on another row. W1 does record what the checkpoint says about it — the nextn
// tail is EXACTLY ONE layer (`model.layers.46.*`), which is
// `num_nextn_predict_layers = 1` agreeing with `Dots3NoteConfig.__init__`.
//
// SCOPE HONESTY: registering this arch makes it RESOLVE and parse its config.
// It does NOT make it load and it does NOT make it forward — both refuse BY
// NAME, naming the brick that owes the work. That polarity is not incidental
// here: spec §6.4 records that NO oracle for this model runs on any hardware
// this project owns, so there is no downstream token gate that could catch a
// forward returning plausible garbage. Refusing is the only safe default.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for dots3-note: text generation, multimodal (image,
// video AND audio — `multimodal.py`::Dots3NoteForCausalLM.get_placeholder_str
// :65-72 handles all three), NOT hybrid in this tree's sense (both attention
// classes are attention over a paged MLA cache; the sliding half is a window on
// the same cache, not a recurrent state).
inline constexpr ModelInfo kDots3NoteInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

class Dots3NoteLoadedModel final : public LoadedModel {
 public:
  Dots3NoteLoadedModel(const ModelRegistration& registration,
                       Dots3NoteWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Dots3NoteWeights& weights() const { return weights_; }

 private:
  Dots3NoteWeights weights_;
};

std::unique_ptr<LoadedModel> LoadDots3NoteForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // The GGUF k-quant arm is OWED, not optional (AGENTS.md, porting-a-model.md
    // §2) — and for this row it is the only arm that could ever fit a host we
    // own (spec §6.2: ~576 GB bf16 / ~290 GB fp8 against a 122 GiB ceiling).
    // llama.cpp has no `dots3_note` architecture, so the converter is ours to
    // write. W9. Refusing by name beats a silent dequantize.
    throw std::runtime_error(
        "Dots3NoteForCausalLM: GGUF k-quants are not ported yet (W9 owes both "
        "the converter and the loader — llama.cpp has no dots3_note "
        "architecture, so there is no upstream converter to reuse). See "
        ".agents/specs/dots3-note.md and issue #699.");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Dots3NoteLoadedModel>(
      registration, LoadDots3NoteWeights(*source.safetensors, config));
}

void PrepareDots3NoteForCausalLM(LoadedModel& model, const HfConfig& config,
                                 vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardDots3NoteForCausalLM(LoadedModel& model,
                                          const ModelForwardInput& input) {
  // `ModelAs`, never a bare `static_cast`: opening a type-erased handle by
  // promise is undefined behaviour on any object that is not really this type,
  // and UBSan's vptr check reports it (#775, and the NemotronH repeat #730).
  // It matters MORE on a refusing forward than on a working one, because the
  // type confusion happens on the way to a throw that would have happened
  // anyway and is therefore invisible without a sanitizer.
  auto& d3 = ModelAs<Dots3NoteLoadedModel>(model, "Dots3NoteForCausalLM");
  // [[noreturn]] until W3-W10 land. The delegation is deliberate rather than an
  // inline throw: `check-runner-routing-consistency.py` classifies a registered
  // model from the ForwardDevice impl its hook delegates to, and a model with
  // no recognizable producer lands in the silently-exempt NONE bucket. This
  // shape reports REFUSE, and it is the signature W3 fills in.
  return Dots3NoteModel::ForwardDevice(input.token_ids, input.positions,
                                       input.attn_meta, input.attn_kv,
                                       d3.weights(), input.queue,
                                       input.logits_indices);
}

const ModelFactory kDots3NoteFactory{
    .parse_config = &ParseDots3NoteConfig,
    .load_weights = &LoadDots3NoteForCausalLM,
    .prepare = &PrepareDots3NoteForCausalLM,
    .forward = &ForwardDots3NoteForCausalLM,
    .make_kv_cache = &MakeDots3NoteKVCache,
    .is_dense_model = false,
};

}  // namespace

REGISTER_VLLM_MODEL(dots3_note, "Dots3NoteForCausalLM", kDots3NoteFactory,
                    kDots3NoteInfo)

}  // namespace vllm
