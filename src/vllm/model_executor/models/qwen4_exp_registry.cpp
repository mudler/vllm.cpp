// Qwen4-Exp registry TU — the ADDITIVE self-registration seam (W1 of
// MODEL-MM-QWEN4-EXP, #1981). Follows the dots3_note_registry.cpp /
// gemma4_registry.cpp seam exactly: a NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array.
//
// UPSTREAM. `Qwen4ExpForConditionalGeneration` is registered by NO vLLM
// revision. Read live 2026-08-26 at vLLM `origin/main` = `6a5e8f5979`: no
// `qwen4*` path, no `registry.py` entry, and a repository-wide search for
// `qwen4` returns zero results; `vllm-omni` likewise. That is absence from
// vLLM `main` rather than staleness in our parity pin `555967922`, so this TU
// deliberately carries no pinned upstream module/class anchor, the convention
// `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` follows for a beyond-pin arm.
// The ALGORITHM source is transformers **5.16.0**, the accepted lane pin; see
// `.agents/oracles/transformers.md` and `.agents/specs/qwen4-exp-flash-next.md`.
//
// The MTP head is deliberately NOT registered as a second architecture, and
// unlike dots3-note that is not a scheduling choice: upstream carries it as an
// `mtp` block INSIDE the same text config rather than as a separate registry
// entry, so there is no second architecture string to register. That is why
// this row moves the MODEL row ratchet by ONE and not by two.
//
// SCOPE HONESTY, RESTATED AT W5a (#2031). Registering this arch makes it
// RESOLVE, parse and validate its config, and — since W5a — LOAD a `qwen4exp`
// GGUF on a CPU device. It does NOT make it forward, and it does not make it
// serve: `ModelRegistry::Forward` and `make_kv_cache` both still refuse BY
// NAME, naming the wave that owes the work, so no token has been decoded by
// this architecture. The paragraph this replaces said the load refused too,
// which was true at W5a's parent and is not true here. That polarity matters
// more here than usual, because no oracle for this model runs on any hardware
// this project owns yet (`gateable = no`, blocked on memory rather than
// software), so there is no downstream token gate that would catch a forward
// returning plausible garbage. Refusing is the only safe default.
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform — the load-time device gate

#include "vt/dtype.h"  // VT_CHECK

#include <memory>
#include <stdexcept>

#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// Text generation, multimodal (image AND video: the published config carries
// `image_token_id`, `video_token_id` and a `vision_config`), and HYBRID —
// 36 of 48 layers are Gated DeltaNet carrying recurrent state, so this belongs
// with the hybrids and not with the pure-attention arms.
inline constexpr ModelInfo kQwen4ExpInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    // FALSE by the house convention the blanket assertion in
    // test_model_registry.cpp enforces: our ModelInfo is a consumed subset
    // whose only reader short-circuits on is_hybrid, so the GDN-hybrid
    // wrappers (kQwen3_5Info, kKimiLinearInfo) all leave this false even
    // though upstream's class carries HasInnerState.
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

// `Qwen4ExpLoadedModel` — the concrete model this hook produces — is declared in
// `qwen4_exp_weights.h` rather than here. That header says why: an anonymous
// type is unreachable by `dynamic_cast` from another translation unit, and a
// reachability case that cannot open the handle cannot tell a real load from a
// hook that returns `Qwen4ExpWeights{}`.

std::unique_ptr<LoadedModel> LoadQwen4ExpForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // W5a (#2031) LOADS it. The GGUF k-quant arm is OWED, not optional
    // (AGENTS.md, porting-a-model.md §2), and for this row it is the ONLY arm
    // that fits a host we own: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S is
    // 67.56 GiB of weights against ~119.6 GiB usable on GB10, where every
    // safetensors artifact (bf16 ~360 GB, FP8 ~180 GB, NVFP4 ~128 GB) does not.
    //
    // Both blockers W1 named have since landed. W6a (#1989) added the IQ4_NL
    // and Q5_0 reader arms, so the file opens at all, and made
    // `GgufTensorRole::kEmbeddingTable` keep-quant eligible with a dequantizing
    // gather behind it, so the 51.2 G-parameter n-gram table stays resident as
    // blocks instead of expanding to 102.4 GB of bf16.
    //
    // A null `gguf` reaches here from a caller that set the KIND without the
    // FILE. Refused by name rather than dereferenced: the alternative is a
    // segmentation fault inside a loader the reader is entitled to read as
    // "GGUF is not supported here".
    if (source.gguf == nullptr) {
      throw std::runtime_error(
          "Qwen4ExpForConditionalGeneration: the model source says GGUF but "
          "carries no file. See .agents/specs/qwen4-exp-flash-next.md and "
          "issue #2031.");
    }
    return std::make_unique<Qwen4ExpLoadedModel>(
        registration,
        LoadQwen4ExpFromGguf(*source.gguf, config,
                             platforms::CurrentPlatform().device_type()));
  }
  (void)registration;
  (void)config;
  // The safetensors arm stays refused, and NOT because it is the harder one.
  // Every published safetensors artifact of this model is larger than every
  // device this project owns, so an arm that read them would be code nothing
  // could ever run. The spec's `## Owed` records it with that reason rather
  // than as an unqualified to-do.
  throw std::runtime_error(
      "Qwen4ExpForConditionalGeneration: the safetensors weight loader is not "
      "ported (every published safetensors artifact — bf16 ~360 GB, FP8 ~180 "
      "GB, NVFP4 ~128 GB — exceeds every device this project owns, so the GGUF "
      "arm is the supported one). See .agents/specs/qwen4-exp-flash-next.md and "
      "issue #1978.");
}

void PrepareQwen4ExpForConditionalGeneration(LoadedModel& model,
                                             const HfConfig& config,
                                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardQwen4ExpForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // THE REFUSAL COMES FIRST, AND THERE IS NO DOWNCAST ABOVE IT. That ordering
  // is what makes it reachable at all, and the first draft had it the other way
  // round.
  //
  // The house shape opens the type-erased handle with
  // `ModelAs<Qwen4ExpLoadedModel>` before doing anything else, because a bare
  // `static_cast` down the hierarchy is undefined behaviour on an object that
  // is not really that type (#775, #730). The ordering stays inverted here, and
  // the REASON changed at W5a (#2031): the original one was that nothing could
  // produce a loaded Qwen4-Exp while `load_weights` refused unconditionally, so
  // every handle was a foreign one. `load_weights` LOADS now, so a caller can
  // present a genuine `Qwen4ExpLoadedModel`, and a downcast placed first would
  // simply succeed and then refuse one line later — no worse, but no longer the
  // argument.
  //
  // What still holds is the second half: there is nothing for the opened handle
  // to be used FOR until W5b writes the forward, so a cast in front of an
  // unconditional refusal buys the reader nothing and costs the #775 axis its
  // strictly safer direction — no cast happens, so no type confusion can. W5b
  // restores `ModelAs` in the same change that gives it something to read.
  //
  // `VT_CHECK(false, ...)` IN THE HOOK BODY, and not a bare throw behind a
  // `Class::ForwardDevice` delegate. Two constraints meet here.
  //
  // `scripts/check-runner-routing-consistency.py` recognises a refuse-by-name
  // stub by exactly this token (`_REFUSE`), and it classifies the hook body
  // itself. A model it cannot classify lands in the silently-exempt NONE
  // bucket, which is the hole that checker exists to close — so tripping it
  // would be the defect, not the gate. The delegate hop dots3-note uses does
  // not help a model like this one: it resolves `Class::ForwardDevice` across
  // translation units or through a file-local `ForwardLogits` helper, and this
  // TU has neither.
  //
  // And `[[noreturn]]` on a non-void return type is MSVC C4646, promoted to
  // C2220 under /W4 /WX; `check-windows-portability.py` caught that on the
  // first draft of this function.
  VT_CHECK(false,
           "Qwen4ExpForConditionalGeneration: the forward is not ported yet. W2 "
           "owes the hashed n-gram embedding and the PLE dilated depthwise conv, "
           "W3 the gated-residual hyper-connection stream, W4 Qwen Sparse "
           "Attention and its indexer side cache, and W5 the assembled forward, "
           "vision path and MTP. See .agents/specs/qwen4-exp-flash-next.md and "
           "issue #1978.");
  return ForwardLogits{};  // unreachable; VT_CHECK always throws here
}

v1::KVCacheConfig MakeQwen4ExpKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  (void)config;
  (void)block_size;
  (void)num_blocks;
  // REACHABLE since W5a (#2031), and that is a behaviour change this comment
  // used to deny: it read "unreachable while the loader refuses", which was
  // true at the parent and is not true at this head. `LoadedEngine` now loads
  // the whole text tower and then arrives HERE, so pointing the engine at the
  // shipped 67.56 GiB artifact pays the full load before it is refused, where
  // before W5a it was refused at once. The spec's `## Owed` records that
  // regression and the CUDA n-gram expansion behind it
  // ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)); W5c is what
  // closes it by making this function return a config instead of throwing.
  //
  // Refusing BY NAME rather than returning an empty config: this model needs
  // THREE conv states per linear layer (GDN conv, PLE conv, and an int64
  // n-gram token history) plus a QSA indexer side cache holding one key vector
  // per block of four tokens, and a spec that silently omits them would
  // allocate a wrong-sized cache that nothing downstream checks.
  throw std::runtime_error(
      "Qwen4ExpForConditionalGeneration: the KV-cache spec is not ported yet "
      "(W4 owes the QSA indexer side cache and W2 the third conv state for the "
      "n-gram token history). See .agents/specs/qwen4-exp-flash-next.md and "
      "issue #1978.");
}

const ModelFactory kQwen4ExpFactory{
    .parse_config = &ParseQwen4ExpConfig,
    .load_weights = &LoadQwen4ExpForConditionalGeneration,
    .prepare = &PrepareQwen4ExpForConditionalGeneration,
    .forward = &ForwardQwen4ExpForConditionalGeneration,
    .make_kv_cache = &MakeQwen4ExpKVCache,
    .is_dense_model = false,
};

}  // namespace

REGISTER_VLLM_MODEL(qwen4_exp, "Qwen4ExpForConditionalGeneration",
                    kQwen4ExpFactory, kQwen4ExpInfo)

}  // namespace vllm
