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
// SCOPE HONESTY: registering this arch makes it RESOLVE and parse and validate
// its config. It does NOT make it load and it does NOT make it forward — both
// refuse BY NAME, naming the wave that owes the work. That polarity matters
// more here than usual, because no oracle for this model runs on any hardware
// this project owns yet (`gateable = no`, blocked on memory rather than
// software), so there is no downstream token gate that would catch a forward
// returning plausible garbage. Refusing is the only safe default.
#include "vllm/model_executor/models/model_registry.h"

#include "vt/dtype.h"  // VT_CHECK

#include <memory>
#include <stdexcept>

#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen4_exp.h"
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

class Qwen4ExpLoadedModel final : public LoadedModel {
 public:
  explicit Qwen4ExpLoadedModel(const ModelRegistration& registration)
      : LoadedModel(registration) {}
};

std::unique_ptr<LoadedModel> LoadQwen4ExpForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  (void)registration;
  (void)config;
  if (source.kind == ModelSource::Kind::kGguf) {
    // The GGUF k-quant arm is OWED, not optional (AGENTS.md,
    // porting-a-model.md §2), and for this row it is the arm most likely to
    // fit a host we own: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S is 67.56
    // GiB of weights against ~119.6 GiB usable on GB10, where every
    // safetensors artifact (bf16 ~360 GB, FP8 ~180 GB, NVFP4 ~128 GB) does
    // not. Two things block it and both are ours: our GGUF reader has no
    // `case 20`, so IQ4_NL — which that file uses for `ffn_down_exps` and for
    // the n-gram table — fails at header parse; and the n-gram table is a
    // gather, which `KeepQuantKDim` refuses to keep quantized, expanding 51.2B
    // params to 102.4 GB of bf16. W6 owes both.
    throw std::runtime_error(
        "Qwen4ExpForConditionalGeneration: the GGUF arm is not ported yet (W6 "
        "owes the `qwen4exp` architecture reader, IQ4_NL support, and a "
        "quantized-gather path for the n-gram table). See "
        ".agents/specs/qwen4-exp-flash-next.md and issue #1978.");
  }
  throw std::runtime_error(
      "Qwen4ExpForConditionalGeneration: the weight loader is not ported yet "
      "(W5 owes it; the config resolves and validates, which is all W1 "
      "claims). See .agents/specs/qwen4-exp-flash-next.md and issue #1978.");
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
  // `ModelAs`, never a bare `static_cast`: opening a type-erased handle by
  // promise is undefined behaviour on any object that is not really this type,
  // and it matters MORE on a refusing forward than on a working one, because
  // the type confusion happens on the way to a throw that would have happened
  // anyway and is therefore invisible without a sanitizer (#775, #730).
  (void)ModelAs<Qwen4ExpLoadedModel>(model,
                                     "Qwen4ExpForConditionalGeneration");
  (void)input;
  // `VT_CHECK(false, ...)` IN THE HOOK BODY, and not a bare throw behind a
  // `Class::ForwardDevice` delegate. Three constraints meet here and only this
  // shape satisfies all of them.
  //
  // `scripts/check-runner-routing-consistency.py` recognises a refuse-by-name
  // stub by exactly this token (`_REFUSE`), and it classifies the hook body
  // itself. A model it cannot classify lands in the silently-exempt NONE
  // bucket, which is the hole that checker exists to close — so tripping it
  // would be the defect, not the gate. The delegate hop dots3-note uses does
  // not help a model like this one: it resolves `Class::ForwardDevice` across
  // translation units or through a file-local `ForwardLogits` helper, and a
  // class defined inside this TU's own anonymous namespace is neither.
  //
  // And `[[noreturn]]` on a non-void return type is MSVC C4646, promoted to
  // C2220 under /W4 /WX; `check-windows-portability.py` caught that on the
  // first draft of this function.
  //
  // There is no `Qwen4ExpModel::ForwardDevice` yet because there is no device
  // forward yet. Inventing one to refuse from would assert a routing shape this
  // row has not earned; W5 introduces it when there is something to route.
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
  // Unreachable while the loader refuses, and refusing by name anyway rather
  // than returning an empty config: this model needs THREE conv states per
  // linear layer (GDN conv, PLE conv, and an int64 n-gram token history) plus
  // a QSA indexer side cache holding one key vector per block of four tokens,
  // and a spec that silently omits them would allocate a wrong-sized cache
  // that nothing downstream checks.
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
