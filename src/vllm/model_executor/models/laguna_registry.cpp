// Laguna-S-2.1 (`LagunaForCausalLM` / `model_type=laguna`) registry TU — the
// ADDITIVE self-registration seam (mirrors deepseek_v4_registry.cpp /
// olmo2_registry.cpp exactly: a NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array). It owns the arch
// entry points: the config hook (ParseLagunaConfig), the KV-cache spec, the
// LoadedModel subclass, and the factory.
//
// Registry routing upstream sends `LagunaForCausalLM` into
// `vllm/model_executor/models/laguna.py`. We register exactly that ONE string.
//
// SCOPE HONESTY (ds4 precedent): registering this arch makes it RESOLVE + parse
// config + build the KV-cache spec. It does NOT make it forward — LagunaModel is
// a W3/W4 stub whose forward VT_CHECK(false, ...)s, and the weight loaders
// VT_CHECK(false, ...) on device materialization (name-map + quant-mix scaffolded
// this increment). A load/forward LOUDLY reports the pending brick — never a
// silent wrong answer. The model-matrix row stays SPIKE/ACTIVE until the strict
// dual-oracle gate (W4) passes on a fetched checkpoint. See
// `.agents/specs/laguna-s21-w1w2-2026-07-30.md`.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/laguna.h"
#include "vllm/model_executor/models/host_token_ids.h"  // ResolveHostTokenIds
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Laguna: text generation, NOT hybrid (interleaved
// full + sliding-window attention over a paged full-attention cache), NOT
// multimodal.
inline constexpr ModelInfo kLagunaInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class LagunaLoadedModel final : public LoadedModel {
 public:
  LagunaLoadedModel(const ModelRegistration& registration, LagunaWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const LagunaWeights& weights() const { return weights_; }

 private:
  LagunaWeights weights_;
};

std::unique_ptr<LoadedModel> LoadLagunaForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // The single-GB10 vehicle: unsloth/Laguna-S-2.1-GGUF UD-Q4_K_XL (~73.4 GiB).
    if (source.gguf == nullptr)
      throw std::runtime_error("laguna GGUF model source is empty");
    // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: the residency policy is built from
    // the device the ENGINE resolved for this load, never from
    // `platforms::CurrentPlatform()`. The two disagree on `--device cpu` on a
    // CUDA-capable process, and this hook is where the disagreement reached the
    // loader.
    const GgufLoadPolicy gguf_policy = GgufLoadPolicy::FromEnv(source.device);
    return std::make_unique<LagunaLoadedModel>(
        registration, LoadLagunaFromGguf(*source.gguf, config, &gguf_policy));
  }
  if (source.safetensors == nullptr)
    throw std::runtime_error("safetensors model source is empty");
  return std::make_unique<LagunaLoadedModel>(
      registration, LoadLagunaForCausalLMWeights(*source.safetensors, config));
}

void PrepareLagunaForCausalLM(LoadedModel& model, const HfConfig& config,
                              vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardLagunaForCausalLM(LoadedModel& model,
                                       const ModelForwardInput& input) {
  auto& laguna = ModelAs<LagunaLoadedModel>(model, "LagunaForCausalLM");
  const LagunaWeights& weights = laguna.weights();
  // #2544/#1305 -- TAKE the asynchronous runner's DEVICE identifiers, BEFORE
  // either arm reads them. On the async serving path the runner's combine
  // splices each decode row's sampled token into the DEVICE buffer on the main
  // queue and leaves `token_ids` deliberately stale for decode rows
  // (`v1/worker/gpu/runner.cpp`, the mirror arm, which is the DEFAULT on CUDA).
  // A forward that embeds the host vector generates every step after the first
  // from token id 0 -- silently, at rc=0, with plausible-looking output.
  //
  // Both arms here are HOST gathers -- `ForwardDevice` calls `Forward` -- so this
  // is the host arm of the seam and not `detail::ApplyDeviceTokenIds`, which
  // splices over a device embed buffer this model does not have.
  // `host_token_ids.h` carries the whole argument.
  //
  // WIRED BUT NOT CONVICTED, and the spec says so under O4: #2544 named this
  // registration on a grep and this row has no staged checkpoint for it.
  std::vector<int32_t> device_ids;
  const std::vector<int32_t>& ids =
      ResolveHostTokenIds(input, &device_ids, "LagunaForCausalLM");
  // #2618 -- THE RESIDENT-QUANT ARM, and it is FIRST for the reason
  // `deepseek_v4_registry.cpp` puts its own paged arm first:
  // `ModelForwardInput::gather_logits` defaults to TRUE and the runner leaves it
  // true on every default step, so a branch placed after the `gather_logits`
  // test below is unreachable on a default configuration. Placing this one after
  // it would land it dead.
  //
  // Below this line is `LagunaModel::Forward`, the unit-gated f32 REFERENCE, and
  // it cannot serve either checkpoint this tree's loaders produce.
  // `LoadLagunaForCausalLMWeights` fills `moe.experts_*_fp4` and leaves
  // `moe.experts_*` EMPTY (`laguna_weights.cpp:443-450`), so the reference
  // sliced `exp_g.begin() + id*gu_stride` out of an empty vector and a registry
  // step SIGSEGV'd inside `memcpy`. `LoadLagunaFromGgufShards` DOES fill
  // `moe.experts_*` (`:807-809`), with Q4_K/Q5_K blocks `ReadF32`
  // (`laguna.cpp:172-190`) refuses by name -- a throw, not a crash, and it fires
  // at the first Q8_0 attention GEMM before the MoE block is reached. Two arms,
  // two different failures, one cause: the reference is not the forward either
  // arm needs.
  //
  // The route predicate is `LagunaForwardGguf`'s OWN precondition, quoted from
  // its `VT_CHECK` at `laguna.cpp:1667`, so the route and the refusal are the
  // same predicate rather than two copies that can drift apart.
  //
  // The fallthrough stays the reference, exactly as ds4 falls through to
  // `DeepseekV4Model::Forward`. A `LagunaWeights` with neither flag can only come
  // from a hand-built synthetic tower, which is what the reference exists to
  // serve; every production loader sets one flag, so every production step takes
  // this arm.
  //
  // This arm is a STATELESS whole-sequence recompute and ignores `attn_kv`. So
  // does the reference (`laguna.cpp:1429-1430` voids both `attn_meta` and
  // `attn_kv`), so the cache contract is unchanged by this route. The
  // incremental arm (`LagunaForwardGgufCached`) is owed on this row's spec.
  if (weights.has_gguf_weights || weights.has_nvfp4_weights) {
    return HostLogits(
        LagunaForwardGguf(weights, input.queue, ids, input.positions,
                          input.logits_indices),
        weights.params.vocab_size);
  }
  if (input.gather_logits) {
    return LagunaModel::ForwardDevice(ids, input.positions,
                                      input.attn_meta, input.attn_kv, weights,
                                      input.config, input.queue,
                                      input.logits_indices);
  }
  return HostLogits(
      LagunaModel::Forward(ids, input.positions, input.attn_meta,
                           input.attn_kv, weights, input.config, input.queue,
                           input.logits_indices),
      weights.params.vocab_size);
}

const ModelFactory kLagunaFactory{
    .parse_config = &ParseLagunaConfig,
    .load_weights = &LoadLagunaForCausalLM,
    .prepare = &PrepareLagunaForCausalLM,
    .forward = &ForwardLagunaForCausalLM,
    .make_kv_cache = &MakeLagunaKVCache,
    // MoE model: NOT the dense per-arch scheduler default.
    .is_dense_model = false,
    .consumes_device_token_ids = true,
};

}  // namespace

v1::KVCacheConfig MakeLagunaKVCache(const HfConfig& config, int block_size,
                                    int num_blocks) {
  // One FULL-ATTENTION KV group over all layers. The interleaved sliding-window
  // (512) layers are masked at the attention kernel (per-layer window), NOT by a
  // smaller SlidingWindowSpec cache — the gemma3 topology the shape-agnostic
  // runner already handles (gemma3_registry.cpp:103-121). KV heads 8, head_dim
  // 128 are uniform across layers (only the Q-head COUNT varies per layer, which
  // does not affect the KV cache spec).
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(laguna, "LagunaForCausalLM", kLagunaFactory, kLagunaInfo)

}  // namespace vllm
