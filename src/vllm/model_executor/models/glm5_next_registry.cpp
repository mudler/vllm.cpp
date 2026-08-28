// GLM-5.3-Flash registry TU — the ADDITIVE self-registration seam (W1 of
// MODEL-MM-GLM53-FLASH, #2067). Follows the qwen4_exp_registry.cpp /
// glm4_moe_lite_registry.cpp seam exactly: a NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array.
//
// UPSTREAM. `Glm5NextForConditionalGeneration` is registered by NO vLLM
// revision. `git grep "Glm5\|glm5_next"` returns zero hits at our parity pin
// `555967922` AND at vLLM `origin/main`; `vllm-project/vllm#53906` would
// register it and is OPEN and unmerged, and an unmerged pull request is not a
// revision. SGLang, vllm-omni and llama.cpp implement nothing either. That is
// ABSENCE from vLLM `main` rather than staleness in our pin, so this TU
// deliberately carries no pinned upstream module/class anchor — the convention
// `MODEL-MM-qwen4-exp-*` follows for a beyond-pin arm — and no pin was
// advanced. The ALGORITHM source is transformers **v5.16.1**; see
// `.agents/specs/glm5-next-flash.md` `## Oracles`.
//
// The MTP head is deliberately NOT registered as a second architecture. The
// checkpoint carries a 46th layer directory that is a DeepSeek-V3-style MTP
// block, and the transformers reference DISCARDS it
// (`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", ...]`), so there is
// no second architecture string to register. That is why this row moves the
// architecture count by ONE.
//
// SCOPE HONESTY. Registering this arch makes it RESOLVE and makes its config
// parse and VALIDATE. It does NOT make it load and it does NOT make it forward
// — both refuse BY NAME, naming the wave that owes the work. That polarity
// matters more here than usual: no oracle for this model runs on any hardware
// this project owns, and none can (the smallest published artifact is 181.32
// GiB against ~119.63 GiB on GB10), so there is no downstream token gate that
// would catch a forward returning plausible garbage. Refusing is the only safe
// default and O1 says so.
#include "vllm/model_executor/models/model_registry.h"

#include "vt/dtype.h"  // VT_CHECK

#include <memory>
#include <stdexcept>

#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// Text generation, multimodal (image AND video: the wrapper carries all six
// placeholder ids and a `vision_config`), and HYBRID — 34 of 45 layers are KDA
// linear attention carrying recurrent state, so this belongs with the hybrids
// and not with the pure-attention arms.
inline constexpr ModelInfo kGlm5NextInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    // FALSE by the house convention the blanket assertion in
    // test_model_registry.cpp enforces: our ModelInfo is a consumed subset
    // whose only reader short-circuits on is_hybrid, so every hybrid wrapper
    // here leaves this false even though upstream's class carries inner state.
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

std::unique_ptr<LoadedModel> LoadGlm5NextForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  (void)registration;
  (void)config;
  if (source.kind == ModelSource::Kind::kGguf) {
    // NOT the same refusal as the safetensors arm, because the two are blocked
    // on different things and a reader who lands here has a DIFFERENT next
    // step. The GGUF container is now readable — W1 wired `glm5next` into the
    // architecture dispatch and this config came out of
    // `Glm5NextHfConfigFromGguf` — so what is missing is the weight tower, not
    // the door. And no artifact exists to hand it either: no `.gguf` of this
    // model has ever been produced, by anyone (O7).
    throw std::runtime_error(
        "Glm5NextForConditionalGeneration: the GGUF config is read and "
        "validated, but the weight loader is not ported (W5 owes the KDA, NoPE "
        "MLA, mHC and stacked-expert weight tower). Separately, NO `.gguf` of "
        "this model exists anywhere: `scripts/convert-glm5-next-gguf.py` can "
        "write one but has never been run against the 305.78 GiB checkpoint "
        "(O7). See .agents/specs/glm5-next-flash.md and issue #1998.");
  }
  throw std::runtime_error(
      "Glm5NextForConditionalGeneration: the weight loader is not ported yet "
      "(W5 owes it; the config resolves and validates, which is all W1 "
      "claims). The published safetensors arms do not fit any device this "
      "project reaches either -- FP8 305.78 GiB and BF16 598.53 GiB against "
      "~119.63 GiB on GB10. See .agents/specs/glm5-next-flash.md and issue "
      "#1998.");
}

void PrepareGlm5NextForConditionalGeneration(LoadedModel& model,
                                             const HfConfig& config,
                                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGlm5NextForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // THE REFUSAL COMES FIRST, AND THERE IS NO DOWNCAST ABOVE IT. The house shape
  // opens the type-erased handle with `ModelAs<...>` before anything else,
  // because a bare `static_cast` down the hierarchy is undefined behaviour on
  // an object that is not really that type (#775, #730). But nothing can
  // PRODUCE a loaded GLM-5.3-Flash while `load_weights` refuses
  // unconditionally, so the only handle any caller can present is a foreign
  // one, and a downcast placed first would turn every reach into a
  // type-mismatch report -- leaving the refusal below dead code no test could
  // enter and any later wave could delete without a red. W5 restores `ModelAs`
  // at the moment there is a real forward with a real model to open.
  //
  // `VT_CHECK(false, ...)` IN THE HOOK BODY, not a bare throw behind a
  // `Class::ForwardDevice` delegate: `check-runner-routing-consistency.py`
  // recognises a refuse-by-name stub by exactly this token and classifies the
  // hook body itself, and a model it cannot classify lands in the silently
  // exempt NONE bucket. And `[[noreturn]]` on a non-void return type is MSVC
  // C4646, promoted to C2220 under /W4 /WX.
  //
  // WHAT THIS REFUSAL BUYS, exactly: it prevents a plausible-but-wrong forward,
  // not a wrong number. There is no partial numeric path here to fall back to.
  // Every primitive named below is unimplemented for THIS model, and two of
  // them look implemented and are not -- our KDA is Kimi-Linear's softplus
  // forget gate where this model needs the sigmoid branch, and our
  // `HcHeadCollapse` is DeepSeek-V4's weighted collapse where this model needs
  // an unweighted mean. Reusing either would generate fluent, wrong text that
  // no gate on this fleet could detect.
  VT_CHECK(false,
           "Glm5NextForConditionalGeneration: the forward is not ported yet. W2 "
           "owes the KDA forget gate's SIGMOID branch (`gate_lower_bound` "
           "-5.0; our kimi_kda.cpp implements the softplus branch and is NOT a "
           "substitute), the strict-fp32 gated RMSNorm and `l2norm`; W3 the "
           "NoPE MLA block -- `MlaBlockDims::Validate` still refuses "
           "`qk_rope_head_dim == 0` -- and the DSA k-pool indexer; W4 the "
           "UNWEIGHTED mHC head collapse (`deepseek_v4_mhc.cpp`'s "
           "`HcHeadCollapse` is the weighted DeepSeek-V4 one and is NOT a "
           "substitute); W5 the MoE routing, decoder layer and assembled text "
           "forward; W6 the vision tower, processor and placeholder expansion. "
           "See .agents/specs/glm5-next-flash.md and issue #1998.");
  return ForwardLogits{};  // unreachable; VT_CHECK always throws here
}

v1::KVCacheConfig MakeGlm5NextKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  (void)config;
  (void)block_size;
  (void)num_blocks;
  // Unreachable while the loader refuses, and refusing by name anyway rather
  // than returning an empty config. This model needs THREE distinct cache
  // shapes in one spec -- a KDA recurrent state plus three separate conv states
  // on 34 layers, a 512-wide MLA latent on 11, and a DSA indexer side cache
  // that is 257 floats per token per layer rather than the DeepSeek-V4 parent's
  // 128 because of the k-pool stage -- and a spec that silently omitted any of
  // them would allocate a wrong-sized cache that nothing downstream checks.
  // #1963/#1966 are the standing reason a KV arithmetic here is re-derived
  // against the runner rather than trusted.
  throw std::runtime_error(
      "Glm5NextForConditionalGeneration: the KV-cache spec is not ported yet "
      "(W3 owes the NoPE MLA latent group and the k-pool indexer side cache, "
      "W5 the KDA recurrent and three-conv state group). See "
      ".agents/specs/glm5-next-flash.md and issue #1998.");
}

const ModelFactory kGlm5NextFactory{
    .parse_config = &ParseGlm5NextConfig,
    .load_weights = &LoadGlm5NextForConditionalGeneration,
    .prepare = &PrepareGlm5NextForConditionalGeneration,
    .forward = &ForwardGlm5NextForConditionalGeneration,
    .make_kv_cache = &MakeGlm5NextKVCache,
    .is_dense_model = false,
};

}  // namespace

REGISTER_VLLM_MODEL(glm5_next, "Glm5NextForConditionalGeneration",
                    kGlm5NextFactory, kGlm5NextInfo)

}  // namespace vllm
