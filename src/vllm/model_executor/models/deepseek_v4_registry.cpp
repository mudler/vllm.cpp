// DeepSeek-V4-Flash (`DeepseekV4ForCausalLM`) registry TU — the ADDITIVE self-
// registration seam for the DeepSeek-V4 bring-up (`CLAIM-DEEPSEEK-V4-IMPL`,
// W1/W2). Follows the deepseek_v2_registry.cpp / gemma4_registry.cpp seam exactly:
// a NEW translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array. It owns the arch entry points: the config hook (config-descent
// validation), the KV-cache spec (STUB), the LoadedModel subclass + factory.
//
// Registry routing upstream (`registry.py:94`) sends `DeepseekV4ForCausalLM` into
// `vllm.models.deepseek_v4`. We register exactly that ONE string. The `DSparkDraftModel`
// (registry.py:596) speculator is a separate row (INVENTORIED). The native MTP head
// `DeepSeekV4MTPModel` (registry.py:617 -> vllm.models.deepseek_v4.DeepSeekV4MTP) is
// WIRED at W1: its tiny-config draft forward + the lossless self-spec verify land in
// deepseek_v4.cpp (`DeepseekV4MtpDraftLogitsHost`) + deepseek_v4.h + the loader's
// absence guard (`DeepseekV4GgufHasMtp`). It is NOT registered as an engine
// speculator here yet — the DS4-native propose/verify DECODE-LOOP integration + the
// engine spec-config seam are named residuals (R2/R3), and the real-model gate is
// weight-BLOCKED (both shipped GGUFs dropped the nextn tail). See
// `.agents/specs/deepseek-v4-mtp.md`.
//
// SCOPE HONESTY: registering this arch makes it RESOLVE + parse config + account
// for the checkpoint tensors; it does NOT make it forward. `DeepseekV4Model` is a
// W3-W8 stub that VT_CHECK(false, ...) — so a load succeeds through the loader's
// accounting pass but a FORWARD loudly reports the pending brick. The model-matrix
// row stays SPIKE until the strict gate (W8) passes. HW note: the NVFP4 checkpoint
// is 156.7 GiB and does NOT fit ONE GB10 (119 GiB) — the runnable oracle gate (W1)
// is MEMORY-INFEASIBLE on a single Spark. See .agents/specs/deepseek-v4-flash.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for DeepSeek-V4: text generation, NOT hybrid (MLA is
// full attention over a paged cache), NOT multimodal.
inline constexpr ModelInfo kDeepseekV4Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class DeepseekV4LoadedModel final : public LoadedModel {
 public:
  DeepseekV4LoadedModel(const ModelRegistration& registration,
                        DeepseekV4Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const DeepseekV4Weights& weights() const { return weights_; }

 private:
  DeepseekV4Weights weights_;
};

std::unique_ptr<LoadedModel> LoadDeepseekV4ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // W2b (LANDED): the single-Spark `deepseek4` GGUF vehicle
    // (`unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_XXS`, ~91 GiB — the only build that
    // fits ONE GB10's 119 GiB unified pool). The keep-quant compute for the
    // ~2-3-bit routed experts (IQ2_XXS/IQ3_XXS/Q2_K, vt::MatmulBTQuant) + the
    // blk.N.* -> V4 name map (EXACT 1328/1328 coverage) landed in W8; W2b wires
    // them into the DeepseekV4 weight towers (keep-quant blocks stay COMPRESSED;
    // the small MHC/DSA/norm/embed tensors dequant). See
    // .agents/specs/deepseek-v4-flash.md §W2b. HONEST 3-state: the materialization
    // is gated at tiny synthetic shape (test_deepseek_v4_gguf_load.cpp); the real
    // 91 GiB load + generate is the W8-final operational residual (download + DGX).
    if (source.gguf == nullptr) {
      throw std::runtime_error("deepseek-v4 GGUF model source is empty");
    }
    return std::make_unique<DeepseekV4LoadedModel>(
        registration, LoadDeepseekV4FromGguf(*source.gguf, config));
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<DeepseekV4LoadedModel>(
      registration, LoadDeepseekV4ForCausalLMWeights(*source.safetensors, config));
}

void PrepareDeepseekV4ForCausalLM(LoadedModel& model, const HfConfig& config,
                                  vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardDeepseekV4ForCausalLM(LoadedModel& model,
                                           const ModelForwardInput& input) {
  auto& ds = ModelAs<DeepseekV4LoadedModel>(model, "DeepseekV4ForCausalLM");
  const DeepseekV4Weights& weights = ds.weights();
  if (input.gather_logits) {
    return DeepseekV4Model::ForwardDevice(input.token_ids, input.positions,
                                          input.attn_meta, input.attn_kv, weights,
                                          input.queue, input.logits_indices);
  }
  // KV-DSV4-MULTICACHE W5 (#2323): the runner handed us a name-keyed cache set,
  // so consume it instead of recomputing the prefix every step.
  if (input.multi_kv != nullptr) {
    std::vector<vt::Tensor> pages;
    const std::string refusal = ResolveDeepseekV4SwaPages(
        weights.params, *input.multi_kv, input.attn_kv, input.attn_meta.num_reqs,
        input.queue.device, &pages);
    VT_CHECK(refusal.empty(), refusal);
    const int64_t kv_base =
        input.attn_meta.num_computed_tokens_cpu.empty()
            ? 0
            : static_cast<int64_t>(input.attn_meta.num_computed_tokens_cpu[0]);
    return HostLogits(
        DeepseekV4ForwardGgufPaged(weights, input.queue, pages, kv_base, input.token_ids,
                                   input.positions, input.logits_indices),
        weights.params.vocab_size);
  }
  return HostLogits(
      DeepseekV4Model::Forward(input.token_ids, input.positions, input.attn_meta,
                               input.attn_kv, weights, input.queue,
                               input.logits_indices),
      weights.params.vocab_size);
}

const ModelFactory kDeepseekV4Factory{
    .parse_config = &ParseDeepseekV4Config,
    .load_weights = &LoadDeepseekV4ForCausalLM,
    .prepare = &PrepareDeepseekV4ForCausalLM,
    .forward = &ForwardDeepseekV4ForCausalLM,
    .make_kv_cache = &MakeDeepseekV4KVCache,
    .is_dense_model = false,
    // KV-DSV4-MULTICACHE W5 (#2323): this forward consumes a cache set keyed by
    // layer name. Declaring it is what stops `ModelRegistry::Forward` refusing
    // the topology -- and the adapter above refuses by name every shape it
    // cannot serve, so the guard moves rather than disappearing.
    .consumes_multi_kv = true,
};

}  // namespace

v1::KVCacheConfig MakeDeepseekV4KVCache(const HfConfig& config, int block_size,
                                        int num_blocks) {
  // KV-DSV4-MULTICACHE W2 (#1973): the REAL topology, replacing the one
  // placeholder `"mla"` group this function used to emit.
  //
  // DeepSeek-V4 publishes one cache per (layer x cache role), not one per
  // layer. Four upstream construction sites produce them, and for
  // DeepSeek-V4-Flash's 43 layers they resolve to 167 entries in SEVEN groups
  // (one group per distinct published spec, which is upstream's own grouping
  // rule):
  //
  //   (a) `DeepseekV4Attention.get_kv_cache_spec` — the compressed MLA latent,
  //       `MLAAttentionSpec`, on the 41 layers with compress_ratio > 1 and
  //       `None` on the other two (`vllm/models/deepseek_v4/attention.py:626-645`).
  //       Splits by ratio: 21 at C4A, 20 at C128A.
  //   (b) `DeepseekV4IndexerCache.get_kv_cache_spec` — the indexer key cache,
  //       `MLAAttentionSpec`, on the 21 layers with ratio == 4 (`:669-684`),
  //       built at `k_cache_head_dim` bytes rather than a semantic width
  //       (`:751-760`).
  //   (c) `DeepseekV4SWACache.get_kv_cache_spec` — the sliding-window cache,
  //       `SlidingWindowMLASpec`, on ALL 43 attention layers including the two
  //       that have no MLA cache at all
  //       (`vllm/v1/attention/backends/mla/sparse_swa.py:87-102`, constructed
  //       unconditionally at `attention.py:315-321`).
  //   (d) `CompressorStateCache.get_kv_cache_spec` — the compressor state,
  //       `SlidingWindowMLASpec`, f32, once per compressor
  //       (`vllm/models/deepseek_v4/compressor.py:188-200`). TWO populations:
  //       the attention layer's own compressor (41, `attention.py:333-343`,
  //       head_dim 512) and the indexer's (21, `attention.py:768-777`,
  //       head_dim 128). The attention population splits by ratio into 21 + 20.
  //
  // 21 + 20 + 21 + 43 + 21 + 21 + 20 = 167.
  //
  // NOTHING CONSUMES THIS. No runner allocates any of these groups and
  // `DeepseekV4Model::Forward` still discards `attn_kv`; the runner REFUSES a
  // group it cannot allocate (`src/vllm/v1/worker/gpu/runner.cpp`, #1973)
  // rather than dropping it in silence, so a DeepSeek-V4 engine now refuses to
  // construct instead of allocating a subset of this topology and saying
  // nothing. Carrying the groups is row KV-DSV4-MULTICACHE W3; reading them is
  // W5. Both are tracked under #1925 and listed under `## Owed` in
  // `.agents/specs/kv-dsv4-multicache.md`.
  const DeepseekV4Params p = ParseDeepseekV4Params(config);

  // THE fp8_ds_mla ARM, and why it is published unconditionally.
  // `DeepseekV4Attention.use_fp8_ds_mla_layout` is `ClassVar[bool] = True` on
  // the base (`attention.py:140`); only the FlashInfer-sparse SM120 subclass
  // sets it False (`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:163`).
  // `_resolve_dsv4_kv_cache_dtype` then writes `cache_config.cache_dtype =
  // "fp8_ds_mla"` back onto the cache config and returns `torch.uint8`
  // (`attention.py:89-119`), so the DEFAULT DeepSeek-V4 cache format upstream
  // is fp8_ds_mla and that is the arm mirrored here. Our factory signature
  // carries no cache dtype and `ParseCacheDType` refuses the string by name
  // (`include/vllm/v1/kv_cache_dtype.h:87-90`), so the 512B-aligned plain
  // bf16/fp8 arm is NOT published — owed to W5 with the store path.
  const std::string kCacheDtypeStr = "fp8_ds_mla";
  const std::string kModelVersion = "deepseek_v4";
  constexpr int kAlignment = 576;   // `sparse_swa.py:99`, `attention.py:642`
  constexpr int kSwaBlockSize = 64; // `sparse_swa.py:82`, fixed by tensor sharing
  // 1-byte storage, mirroring upstream's `torch.uint8`. `SizeOf(kI8) == 1` is
  // what the indexer's element formula needs; the 584-byte branch does not read
  // the dtype at all.
  const vt::DType kByteDType = vt::DType::kI8;

  // `get_kv_quant_mode(cache_dtype)` returns FP8_PER_TENSOR for any string
  // starting with "fp8" (`vllm/v1/kv_cache_interface.py:70-71`), and upstream
  // passes it on EXACTLY TWO of the four sites: the SWA cache
  // (`sparse_swa.py:100`) and the compressed latent (`attention.py:644`). The
  // indexer key cache and the compressor state pass nothing and default to
  // NONE. Mirrored exactly, including the asymmetry — which is also what makes
  // this topology exercise W1's branch order, since those two specs carry a
  // non-NONE quant mode AND `cache_dtype_str == "fp8_ds_mla"` and must reach
  // the 584-byte branch before the quant-mode guard throws.
  const v1::KVQuantMode kLatentQuantMode = v1::KVQuantMode::kFp8PerTensor;

  // The published module path, which is what `LayerIndexOfName`
  // (`src/vllm/v1/worker/gpu/runner.cpp`) resolves back to a layer index once
  // W3 reads these names. THE SEGMENT IS `attn`, NOT `self_attn`:
  // `DeepseekV4DecoderLayer` builds its attention as `prefix=f"{prefix}.attn"`
  // (`vllm/models/deepseek_v4/nvidia/model.py:808-813`) under
  // `prefix=f"{prefix}.layers"` (`:1015`) and `maybe_prefix(prefix, "model")`
  // (`:1409`). Every other architecture in this tree spells it `self_attn`, so
  // this is worth stating rather than pattern-matching.
  const auto attn_prefix = [](int64_t l) {
    return "model.layers." + std::to_string(l) + ".attn";
  };

  // Per-group layer-name lists, filled by ONE walk over the layers.
  std::vector<std::string> c4a_latent, c128a_latent, indexer_key, swa;
  std::vector<std::string> c4_attn_state, c4_indexer_state, c128_attn_state;
  bool has_c4 = false, has_c128 = false;

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    // `max(1, config.compress_ratios[layer_id])` — upstream's own guard, and
    // the value every downstream branch tests against (`attention.py:205-212`).
    // Our parser keeps the raw 0 for layers 0 and 1.
    const int64_t raw = p.compress_ratio(l);
    const int64_t ratio = raw < 1 ? 1 : raw;
    VT_CHECK(ratio == 1 || ratio == 4 || ratio == 128,
             std::string("deepseek-v4 kv-cache: unsupported compress_ratio ") +
                 std::to_string(ratio) + " on layer " + std::to_string(l) +
                 "; upstream accepts 1, 4 or 128 only "
                 "(vllm/v1/attention/backends/mla/sparse_swa.py:44-55)");

    // (c) Every attention layer has a SWA cache, unconditionally.
    swa.push_back(attn_prefix(l) + ".swa_cache");

    if (ratio == 1) continue;  // `attention.py:626-630` returns None: no MLA cache.

    // (a) The compressed latent. `head_size = self.head_dim` (512) — NOT
    // head_dim + rope, which is what the placeholder this replaces used.
    (ratio == 4 ? c4a_latent : c128a_latent).push_back(attn_prefix(l));
    // (d) The attention layer's own compressor state.
    (ratio == 4 ? c4_attn_state : c128_attn_state)
        .push_back(attn_prefix(l) + ".compressor.state_cache");
    if (ratio == 4) {
      has_c4 = true;
      // (b) + (d) The indexer's key cache and its own compressor state.
      indexer_key.push_back(attn_prefix(l) + ".indexer.k_cache");
      c4_indexer_state.push_back(attn_prefix(l) +
                                 ".indexer.compressor.state_cache");
    } else {
      has_c128 = true;
    }
  }

  // `storage_block_size` is `block_size / compress_ratio`, so a block size that
  // is not a multiple of the ratio gives a truncated page and one that is
  // smaller than the ratio gives a ZERO-byte page. Upstream has no such check
  // because its own comments derive the whole geometry at 256
  // (`sparse_swa.py:76-83` and `compressor.py:174-178` both spell
  // `[256//4, head_dim] = [64, head_dim]`). Refuse by name rather than publish
  // a pool that cannot hold a token.
  const auto check_ratio_fits = [&](int ratio) {
    VT_CHECK(block_size % ratio == 0 && block_size / ratio >= 1,
             std::string("deepseek-v4 kv-cache: block_size ") +
                 std::to_string(block_size) +
                 " cannot express a compress_ratio-" + std::to_string(ratio) +
                 " page (storage_block_size = block_size / compress_ratio "
                 "would be " + std::to_string(block_size / ratio) +
                 "). Upstream derives this geometry at block_size 256 "
                 "(vllm/v1/attention/backends/mla/sparse_swa.py:76-83, "
                 "vllm/models/deepseek_v4/compressor.py:174-178)");
  };
  if (has_c4) check_ratio_fits(4);
  if (has_c128) check_ratio_fits(128);

  // `head_dim bytes = 128 fp8 + 4 fp32 scale = 132` (`attention.py:756-759`),
  // with `quant_block_size = 128` (`:738`). The MXFP4 68-byte arm is selected by
  // `attention_config.use_fp4_indexer_cache`, whose default is False
  // (`vllm/config/attention.py:64`), so the FP8 width is the default and the
  // other arm is owed to W5.
  constexpr int64_t kIndexerQuantBlock = 128;
  const int indexer_head_size = static_cast<int>(
      p.index_head_dim + p.index_head_dim / kIndexerQuantBlock * 4);

  const int head_size = static_cast<int>(p.head_dim);              // 512
  const int swa_window = static_cast<int>(p.sliding_window);       // 128

  // `state_dim = 2 * coff * head_dim` with `coff = 1 + (compress_ratio == 4)`,
  // `sliding_window = coff * compress_ratio`, and block_size 4 for ratio 4 / 8
  // for ratio 128 (`compressor.py:168-200`). dtype is f32, which upstream
  // asserts (`:170`).
  const int c4_attn_state_dim = static_cast<int>(2 * 2 * p.head_dim);        // 2048
  const int c4_indexer_state_dim = static_cast<int>(2 * 2 * p.index_head_dim);  // 512
  const int c128_state_dim = static_cast<int>(2 * 1 * p.head_dim);           // 1024

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;

  const auto add_mla = [&](std::vector<std::string> names, int hs, int ratio,
                           bool ds_mla_layout) {
    if (names.empty()) return;
    kv.kv_cache_groups.emplace_back(
        std::move(names),
        std::make_shared<v1::MLAAttentionSpec>(
            block_size, hs, kByteDType, /*num_kv_heads=*/1,
            ds_mla_layout ? kLatentQuantMode : v1::KVQuantMode::kNone,
            /*page_size_padded=*/std::nullopt,
            /*indexes_kv_by_block_stride=*/false,
            ds_mla_layout ? std::optional<std::string>(kCacheDtypeStr)
                          : std::nullopt,
            kAlignment, ratio,
            ds_mla_layout ? std::optional<std::string>(kModelVersion)
                          : std::nullopt));
  };
  const auto add_swa_mla = [&](std::vector<std::string> names, int bs, int hs,
                               vt::DType dt, int window, bool ds_mla_layout) {
    if (names.empty()) return;
    kv.kv_cache_groups.emplace_back(
        std::move(names),
        std::make_shared<v1::SlidingWindowMLASpec>(
            bs, /*num_kv_heads=*/1, hs, dt, window,
            ds_mla_layout ? std::optional<std::string>(kCacheDtypeStr)
                          : std::nullopt,
            kAlignment, /*compress_ratio=*/1,
            ds_mla_layout ? std::optional<std::string>(kModelVersion)
                          : std::nullopt,
            ds_mla_layout ? kLatentQuantMode : v1::KVQuantMode::kNone));
  };

  // Group order is ours and is documented rather than incidental: the
  // `MLAAttentionSpec` groups first (latent by ratio, then the indexer key),
  // then the `SlidingWindowMLASpec` groups (SWA, then the three compressor
  // state populations).
  add_mla(std::move(c4a_latent), head_size, /*ratio=*/4, /*ds_mla_layout=*/true);
  add_mla(std::move(c128a_latent), head_size, /*ratio=*/128, true);
  add_mla(std::move(indexer_key), indexer_head_size, /*ratio=*/4,
          /*ds_mla_layout=*/false);
  add_swa_mla(std::move(swa), kSwaBlockSize, head_size, kByteDType, swa_window,
              /*ds_mla_layout=*/true);
  add_swa_mla(std::move(c4_attn_state), /*bs=*/4, c4_attn_state_dim,
              vt::DType::kF32, /*window=*/8, /*ds_mla_layout=*/false);
  add_swa_mla(std::move(c4_indexer_state), /*bs=*/4, c4_indexer_state_dim,
              vt::DType::kF32, /*window=*/8, /*ds_mla_layout=*/false);
  add_swa_mla(std::move(c128_attn_state), /*bs=*/8, c128_state_dim,
              vt::DType::kF32, /*window=*/128, /*ds_mla_layout=*/false);
  return kv;
}

REGISTER_VLLM_MODEL(deepseek_v4, "DeepseekV4ForCausalLM", kDeepseekV4Factory,
                    kDeepseekV4Info)

}  // namespace vllm
