// DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`) registry TU — the ADDITIVE
// self-registration seam for W1 of
// `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`
// (ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP). Follows the
// `deepseek_v4_registry.cpp` / `glm5_next_registry.cpp` seam exactly: a NEW
// translation unit with ONE `REGISTER_VLLM_MODEL` line and ZERO edit to any
// shared registration array.
//
// ONE STRING, AND WHICH ONE. Upstream `registry.py:379` routes
// `DeepseekV41ForCausalLM` into `vllm.models.deepseek_v4_1`, and that class is
// the MULTIMODAL entry (`nvidia/vl_model.py:120`). The text-only
// `DeepseekV41LLMForCausalLM` (`nvidia/model.py:979`) is INTERNAL — it is
// declared by no published artifact and appears in no registry table — so it is
// deliberately not registered. `DSparkV41DraftModel` (`registry.py:647`) is the
// speculator and is a separate wave (W3e). This row therefore moves the
// architecture count by ONE.
//
// SCOPE HONESTY, and this is the whole of it. Registering this architecture
// makes it RESOLVE and makes its config PARSE and VALIDATE. It does NOT make
// the model load and it does NOT make it forward. Every hook below except
// `parse_config` refuses BY NAME and names the wave that owes it. That is the
// shape GLM-5.3-Flash's W1 landed in (`47a2b35a5`), for the same reason: a
// registration that silently returned an empty model would be a silent wrong
// answer, and this architecture has no reachable end-to-end token gate anywhere
// on this fleet to catch one (the smallest published artifact is 98.591 GiB of
// degenerate Q1_0 and the release is 475.27 GiB, against 119 GiB on GB10 — see
// `.agents/specs/deepseek-v4-1-flash.md`).
//
// NOTHING LANDS DEAD, and this is stated at the precision the measurement
// supports. What this wave changes for a user of the PUBLISHED artifact is the
// RESOLVE: `ModelRegistry::Resolve` now returns this registration, so
// `LoadedEngine::FromModelDir` stops answering "Model architectures
// ['DeepseekV41ForCausalLM'] are not supported for now" and instead reaches the
// FP8 `weight_block_size [32, 32]` refusal (#1189), whose missing arm is this
// campaign's own W3b. That is a different outcome from a different entry point,
// which is what reachability asks for.
//
// `parse_config` is NOT reached on the published file, and the reason is in the
// call order: `model_registry.cpp:356-358` runs `RefuseUnsupportedFp8BlockQuant`
// BEFORE `factory.parse_config`, and the published config is stopped by that
// guard. The parser IS reached from production on any V4.1 config whose
// `weight_block_size` this build supports, and through the resolved
// registration's own hooks in the gate. The load, the forward and the KV-cache
// spec are not reached at all, by construction — they refuse — and that is the
// wave's deliberate boundary rather than an omission.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/models/deepseek_v4_1.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

// `supports_multimodal` is FALSE, and that is deliberate rather than an
// oversight about a model whose published config carries a 32-layer
// `vision_config` and an `image_token_id`.
//
// Upstream DOES register this string as the multimodal entry
// (`registry.py:379` -> `vllm.models.deepseek_v4_1`, whose entry class is
// `nvidia/vl_model.py:120`). WHAT UPSTREAM DOES NOT DO is assert the flag for
// this architecture: `tests/models/test_registry.py` gates
// `model_info.supports_multimodal is is_mm` (`:93`) over a parametrize table
// (`:76-88`) of eight architectures, and `DeepseekV41ForCausalLM` is not one of
// them -- `LlavaForConditionalGeneration` is that table's only `True` row. So
// there is no upstream assertion this `false` contradicts; there is a
// registration table, and this comment is about why our flag does not simply
// copy it. This flag is read as a claim about THIS PORT,
// not as a statement about upstream, and `Dots3NoteForCausalLM` already paid for
// the difference: its own W1 set the flag true on upstream's authority, W5
// discovered it was "a claim about THIS port that this port could not honour"
// once the config became loadable, and only W6a set it back -- BACKED by an
// `encode_mm`/`embed_mm` pair on the factory and a registered chat seam, so that
// a served `image_url` reaches the forward. See the comment on that branch in
// `tests/vllm/models/test_model_registry.cpp`.
//
// This factory carries neither hook, registers no chat seam, and refuses its own
// forward, so nothing multimodal is reachable and a true here would be that same
// unhonourable claim. W3f owes the 32-layer DeepSeek-ViT tower and W4 the
// composition; whichever wave first makes an image reach the forward is the one
// that flips this flag and moves this architecture to the multimodal branch of
// that test, in the same change as the capability.
//
// `is_hybrid` is FALSE. The compressors carry state, but the house convention
// that test asserts reserves `is_hybrid` for the linear-attention families whose
// recurrent state the runner must allocate; DeepSeek-V4 sits at false for the
// same reason and V4.1's topology is V4's here. W3d/W4 own revisiting it if the
// `kv_source_layer`-keyed topology changes what the runner must allocate; until
// something reads this field for this model, moving it would be an unmeasured
// guess.
inline constexpr ModelInfo kDeepseekV41Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// The refusals. Each names the architecture, the missing part, the wave that
// owes it, and the row/spec/issue a reader can follow. None of them is a
// placeholder that could be mistaken for a working path.

std::unique_ptr<LoadedModel> LoadDeepseekV41ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  (void)registration;
  // The config still has to be VALID to get a truthful refusal. Without this,
  // a malformed config.json would reach the weight-loader refusal below and be
  // reported as "the loader is not ported", which is true but is not the
  // reader's actual problem. Validating first means the message names the field
  // that is wrong when one is.
  ParseDeepseekV41Config(config);

  if (source.kind == ModelSource::Kind::kGguf) {
    // ONE owner for this text: `DeepseekV41GgufRefusal()`. The entrypoint's
    // architecture dispatch throws the same string, so whichever door a GGUF
    // arrives at, the message is identical.
    throw std::runtime_error(DeepseekV41GgufRefusal());
  }
  throw std::runtime_error(
      "DeepseekV41ForCausalLM: the safetensors weight loader is not ported. W1 "
      "makes this architecture RESOLVE and makes its config.json PARSE and "
      "VALIDATE; it loads no weights in any format. The loader is OWED to W8, "
      "and the op families it would feed -- engram, the MXFP8 32x32 UE8M0 "
      "linear arm, the ratio-1/2 indexer, the kv_source_layer-keyed KV "
      "topology, "
      "DSpark and the vision tower -- are OWED to W3a-W3f. Row "
      "MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm, spec "
      ".agents/specs/deepseek-v4-1-flash.md section `## Work breakdown`, issue "
      "ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP.");
}

void PrepareDeepseekV41ForCausalLM(LoadedModel& model, const HfConfig& config,
                                   vt::Queue& queue) {
  // Unreachable in practice: nothing can construct a `LoadedModel` for this
  // architecture while the loader above refuses. It is still a refusal rather
  // than a no-op, because a silent no-op here is exactly the shape that lets a
  // later wave believe a prepare step ran.
  (void)model;
  (void)config;
  (void)queue;
  throw std::runtime_error(
      "DeepseekV41ForCausalLM: prepare is not ported (no weights can be "
      "loaded yet -- W8 owes the loader). Row "
      "MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm, issue "
      "ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP.");
}

// `VT_CHECK(false, ...)` rather than a bare `throw`, and the difference is
// GATED. `scripts/check-runner-routing-consistency.py` classifies every
// registered forward as DEVICE, HOST, REFUSE or NONE, and NONE is the silently-
// exempt bucket its own regression case exists to keep empty: a model that
// matches no seam drops out of the host/device drift check and the gate goes
// green while exempting it. A bare `throw` matches no seam. `VT_CHECK(false)`
// is the tree's REFUSE-by-name idiom (`KimiK3Model::ForwardDevice` is the live
// precedent) and classifies REFUSE, which is an explicit skip rather than a
// hole. The macro prefixes "vt: " and appends the file and line; the message
// below is otherwise the refusal a reader gets.
ForwardLogits ForwardDeepseekV41ForCausalLM(LoadedModel& model,
                                            const ModelForwardInput& input) {
  (void)model;
  (void)input;
  VT_CHECK(false,
           "DeepseekV41ForCausalLM: the forward is not ported. Six op families "
           "have no counterpart in this tree and each is OWED to its own wave: "
           "engram (W3a), the MXFP8 32x32 UE8M0 block dequant and linear arm "
           "(W3b), the ratio-1/2 indexer and its indexer_k_store/query_quant "
           "ops (W3c), the kv_source_layer-keyed KV topology (W3d), the "
           "DSpark block drafter (W3e) and the "
           "32-layer DeepSeek-ViT tower (W3f); W4 composes them. Row "
           "MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm, spec "
           ".agents/specs/deepseek-v4-1-flash.md section `## Work breakdown`, "
           "issue ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP.");
  return {};
}

v1::KVCacheConfig MakeDeepseekV41KVCache(const HfConfig& config, int block_size,
                                         int num_blocks) {
  (void)block_size;
  (void)num_blocks;
  // Validate first, for the same reason the loader does: a reader whose config
  // is wrong should be told which field, not told that the KV-cache spec is
  // missing.
  ParseDeepseekV41Config(config);
  throw std::runtime_error(
      "DeepseekV41ForCausalLM: the KV-cache spec is not ported. This "
      "architecture needs a cache topology this tree cannot yet describe: the "
      "ratio-1 and ratio-2 compressor state populations (V4's spec covers "
      "ratio-4 and ratio-128 only), the compressed layers' kv_source_layer_ids "
      "/ index_source_layer_ids indirection, and the source resolution "
      "`max(s for s in kv_source_layers if s <= layer_id)` that keys layers 20 "
      "to 39 onto layer 20 (attention.py:284-286). "
      "OWED to W3c/W3d and assembled by W4. Row "
      "MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm, spec "
      ".agents/specs/deepseek-v4-1-flash.md section `## Work breakdown`, issue "
      "ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP.");
}

const ModelFactory kDeepseekV41Factory{
    .parse_config = &ParseDeepseekV41Config,
    .load_weights = &LoadDeepseekV41ForCausalLM,
    .prepare = &PrepareDeepseekV41ForCausalLM,
    .forward = &ForwardDeepseekV41ForCausalLM,
    .make_kv_cache = &MakeDeepseekV41KVCache,
    // Deliberately NOT declaring `kv_block_size_floor`, `consumes_multi_kv` or
    // `is_dense_model`. V4 declares all three because its KV-cache spec is real
    // and its forward consumes a keyed cache set; ours refuses, so any value
    // here would describe a topology nothing builds. W3c/W3d/W4 set them when
    // there is something to describe.
};

}  // namespace

REGISTER_VLLM_MODEL(deepseek_v4_1, "DeepseekV41ForCausalLM",
                    kDeepseekV41Factory, kDeepseekV41Info)

}  // namespace vllm
