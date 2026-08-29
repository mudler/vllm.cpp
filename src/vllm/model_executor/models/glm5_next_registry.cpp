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
#include "vllm/model_executor/models/glm5_next_loader.h"
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
  if (source.kind == ModelSource::Kind::kGguf) {
    // W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)) LOADS it.
    // The GGUF k-quant arm is OWED, not optional (AGENTS.md,
    // porting-a-model.md), and for this row it is the ONLY arm that fits a
    // host we own: `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL is 101.2535 GiB on
    // disk against ~119.63 GiB usable on GB10, where every safetensors artifact
    // (FP8 305.78 GiB, BF16 598.53 GiB, NVFP4 181.32 GiB) does not.
    //
    // THE ARTIFACT EXISTS, and the refusal this replaced said it did not. That
    // sentence — "NO `.gguf` of this model exists anywhere" — was true when W1
    // wrote it and stopped being true when `unsloth/GLM-5.3-Flash-GGUF`
    // revision `d425e572fb9686125831f476129e51cea34bc5b4` was published and
    // staged: 1412 tensors, four shards, `general.architecture = glm5next`,
    // read out of the file's own header. A record correction that leaves the
    // lie in the product is not a correction, so it is removed here and not
    // only in the spec. O7 is W7b's
    // ([#2225](https://github.com/mudler/vllm.cpp/issues/2225)) to discharge;
    // this change does not discharge it and does not contradict it — what W7b
    // still owes is the sha256, the conversion recipe and the peak RSS of a
    // real load, none of which this wave measured.
    //
    // A null `gguf` reaches here from a caller that set the KIND without the
    // FILE. Refused by name rather than dereferenced: the alternative is a
    // segmentation fault inside a loader the reader is entitled to read as
    // "GGUF is not supported here".
    if (source.gguf == nullptr) {
      throw std::runtime_error(
          "Glm5NextForConditionalGeneration: the model source says GGUF but "
          "carries no file. See .agents/specs/glm5-next-flash.md and issue "
          "#2242.");
    }
    return std::make_unique<Glm5NextLoadedModel>(
        registration, LoadGlm5NextFromGguf(*source.gguf, config));
  }
  (void)registration;
  (void)config;
  // The safetensors arm stays refused, and NOT because it is the harder one.
  // Every published safetensors artifact of this model is larger than every
  // device this project owns, so an arm that read them would be code nothing
  // could ever run. The spec's `## Owed` records it with that reason rather
  // than as an unqualified to-do.
  throw std::runtime_error(
      "Glm5NextForConditionalGeneration: the safetensors weight loader is not "
      "ported (every published safetensors artifact -- FP8 305.78 GiB, BF16 "
      "598.53 GiB and NVFP4 181.32 GiB -- exceeds every device this project "
      "owns at ~119.63 GiB on GB10, so the GGUF arm is the supported one). "
      "See .agents/specs/glm5-next-flash.md and issue #1998.");
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
  // an object that is not really that type (#775, #730).
  //
  // W5c CHANGED THE PREMISE HALF-WAY AND THE ORDER STILL STANDS. The earlier
  // version of this comment argued that nothing could PRODUCE a loaded
  // GLM-5.3-Flash while `load_weights` refused unconditionally, so the only
  // handle a caller could present was a foreign one. That is no longer true:
  // the GGUF arm above returns a real `Glm5NextLoadedModel`. What has not
  // changed is that there is no forward to open it FOR, so a downcast placed
  // first would report a type mismatch on a foreign handle and then fall
  // through to this same refusal on our own -- two messages for one missing
  // capability, and the refusal reachable only on the path where it says
  // least. W5b ([#2241](https://github.com/mudler/vllm.cpp/issues/2241))
  // restores `ModelAs` in the same change that gives it something to read.
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
           "substitute); W5b the decoder layer, the DSA attention block and the "
           "assembled text forward; W6 the vision tower, processor and "
           "placeholder expansion. The WEIGHT TOWER is ported and this model "
           "LOADS -- W5c (#2242) -- so a handle reaching here is real and the "
           "missing part is the forward, not the load. "
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
      "W5b the KDA recurrent and three-conv state group). See "
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
