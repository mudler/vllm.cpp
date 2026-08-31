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
// SCOPE HONESTY, and W5b-2b moved it. Registering this arch made it RESOLVE
// (W1); W5c made it LOAD; this TU's `forward` hook now RUNS, on the CPU device,
// through `glm5_next::Glm5NextHostForward`. What has NOT changed is the reason
// the earlier refusal existed: no oracle for this model runs on any hardware
// this project owns (the smallest published artifact is 181.32 GiB against
// ~119.63 GiB on GB10), so there is no downstream token gate that would catch a
// forward returning plausible garbage, and O1 still says so. What replaces the
// blanket refusal is a set of NARROW ones, each naming what is owed rather than
// the whole capability: a non-CPU queue, a multi-request step, and the
// safetensors load are each refused by name in place.
#include "vllm/model_executor/models/model_registry.h"

#include "vt/dtype.h"  // VT_CHECK

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_forward.h"
#include "vllm/model_executor/models/glm5_next_kv.h"
#include "vllm/model_executor/models/glm5_next_loader.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"  // v1::ResolveKvCacheDType
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
  // `ModelAs<...>` FIRST, never a bare `static_cast` down the hierarchy, which
  // is undefined behaviour on an object that is not really this type
  // (#775, #730). W5c made this possible by returning a real
  // `Glm5NextLoadedModel`; W5b-2b is the change that gives it something to
  // open it FOR, which is the condition the previous refusal named.
  auto& g = ModelAs<Glm5NextLoadedModel>(model,
                                         "Glm5NextForConditionalGeneration");
  const Glm5NextWeights& w = g.weights();

  // W5b-2c (#2348): THE PAGED CACHES ARE READ, and the house pattern is
  // deliberately NOT followed here. `NemotronHForCausalLM`
  // (`nemotron_h_registry.cpp`) and `KimiLinearForCausalLM`
  // (`kimi_linear_forward.cpp`) each ignore the paged caches and describe
  // themselves as re-running the whole prefix. O28 measured what the runner
  // actually hands a forward and that description does not survive it:
  // `ModelForwardInput::token_ids` is the step's SCHEDULED tokens, so on the
  // second step of a decode it is ONE id and a forward that treats it as a
  // sequence attends to an empty prefix. That is fluent wrong text, which is
  // the one failure this row refuses to ship.
  //
  // `positions` is still unread, and that is upstream's arithmetic rather than
  // an omission: `qk_rope_head_dim` is ZERO on this architecture -- upstream's
  // own `validate_architecture` requires it -- so there is no rotary embedding
  // to place and the causal order comes from the cache length
  // (`glm5_next_dsa.cpp`, `q_pos = current_length - q_length + s`).
  (void)input.positions;

  // ONE DIVERGENCE FROM THAT PATTERN, in the safe direction. Both precedents
  // take `token_ids` as a single sequence whatever `num_reqs` says; on a
  // two-request step that silently attends ACROSS the boundary. For this model
  // that is a fluent wrong answer no gate on this fleet could detect -- the
  // spec's §Gates records that no end-to-end token gate for it exists or can
  // exist here -- so a multi-request step is refused by name instead of
  // approximated. Ragged batching is `attn_meta.query_start_loc` sliced as at
  // `kimi_linear_device.cpp`, and it is OWED.
  VT_CHECK(input.num_reqs <= 1,
           "Glm5NextForConditionalGeneration: this forward is a SINGLE-SEQUENCE "
           "host reference and the step carries " +
               std::to_string(input.num_reqs) +
               " requests. Concatenating them would attend across the request "
               "boundary and emit fluent, wrong text that no gate on this fleet "
               "could detect. Ragged batching (attn_meta.query_start_loc) is "
               "owed. See .agents/specs/glm5-next-flash.md and issue #2241.");

  // THE BINDING, resolved BY NAME and refusing by name. `glm5_next_kv.h`
  // carries the whole argument: what the engine hands over, why the MLA latent
  // is not a K+V pair, and why the recurrent group's correspondence is a count
  // rather than a name.
  VT_CHECK(input.multi_kv != nullptr,
           "Glm5NextForConditionalGeneration: this step arrived with no "
           "multi-KV channel. `MakeGlm5NextKVCache` publishes THREE groups -- "
           "the MLA latent, the KDA recurrent state and the DSA indexer side "
           "cache -- which is a multi-cache topology, so the runner sets "
           "`ModelForwardInput::multi_kv` on every step of this model. A null "
           "channel means the topology was classified as uniform, and the "
           "positional `attn_kv` convention cannot say which of a DSA layer's "
           "two caches an entry is. Running anyway would attend an empty "
           "prefix on every step after the first. See "
           ".agents/specs/glm5-next-flash.md and issue #2348.");
  const glm5_next::KvBinding binding =
      glm5_next::ResolveKvBinding(w.params, input);
  std::vector<glm5_next::LayerCache> caches;
  glm5_next::LoadCaches(w.params, binding, input, &caches);
  std::vector<float> logits = glm5_next::Glm5NextHostForward(
      w, input.token_ids, input.logits_indices, input.queue, &caches);
  // The new rows go back into the ENGINE's pages, so the next step reads them
  // through the same block table the block manager owns -- rather than onto
  // this `LoadedModel`, which would be per-model state the engine cannot
  // evict, preempt or share.
  glm5_next::StoreCaches(w.params, binding, caches, input);
  return HostLogits(std::move(logits), w.params.vocab_size);
}

// ─── The heterogeneous KV-cache spec (W5, #2223) ─────────────────────────────
//
// THREE published groups, and the shape of them is the decision this function
// exists to record:
//
//   0. the 11 DSA layers' MLA latent        `MLAAttentionSpec`, head 512
//   1. the 34 KDA layers' recurrent state   `MambaSpec`, 2 states
//   2. the 11 DSA layers' indexer cache     `MLAAttentionSpec`, head 257
//
// Following `kimi_linear_registry.cpp:135-166`, which publishes the same MLA +
// KDA pair; the third group is this model's and Kimi-Linear has no analogue.
//
// GROUP 0 IS AN MLA LATENT, NOT A K+V PAIR. `Glm5NextTextAttention` caches the
// compressed `kv_a_proj_with_mqa` output and reconstructs K and V from it
// through `kv_b_proj` (`modeling_glm5_next.py:1136-1153`, `expand_kv`), so one
// row per token of `kv_lora_rank + qk_rope_head_dim` elements, `num_kv_heads`
// 1, and NO separate V. `qk_rope_head_dim` is ZERO on this model -- upstream's
// `validate_architecture` requires it ("Expecting NoPE for the DSA attention
// layers") -- so the row is 512 wide where every DeepSeek variant's and
// Kimi-Linear's is 576. A port that reuses the 576 over-allocates by 12.5% and
// nothing downstream reads the difference.
//
// GROUP 1 IS ONE UNIFORM RECURRENT GROUP AND ITS CONV STATE IS `conv_kernel_dim`
// WIDE, NOT `conv_kernel_dim - 1`. Both halves are upstream's.
//
//   * Uniform, because that is all upstream can express:
//     `get_mamba_state_shape_from_config` is a CLASSMETHOD over the config with
//     no `layer_idx` (`vllm/model_executor/models/interfaces.py:809-812` at the
//     parity pin `5559679229`) and `get_mamba_groups` asserts every `MambaSpec`
//     in the model equal (`vllm/v1/worker/mamba_utils.py:441`). Every one of
//     this model's 34 KDA layers carries the same two states anyway, so
//     uniformity costs nothing here.
//   * `conv_kernel_dim` wide, because the reference ALLOCATES it that wide:
//     `LinearAttentionLayer.lazy_initialization` builds
//     `torch.zeros((*shape[:-1], conv_kernel_size))`
//     (`transformers` v5.16.1 `cache_utils.py:1015-1024`) and
//     `Glm5NextTextLinearAttention.forward` passes
//     `conv_kernel_size=self.conv_kernel_size` (`modeling_glm5_next.py:669-671`),
//     so the state is `[B, conv_dim, 4]` and not the `[B, conv_dim, 3]` the
//     convolution arithmetic alone would need. `causal_conv1d_update` then reads
//     `state_len = conv_state.shape[-1]` (`:382`) and writes back that many
//     columns, so the slack column is part of the contract rather than padding.
//     `glm5_next_kda.h` records the same width for the host reference's
//     `Glm5NextKdaCache::conv_state`, and publishing `K - 1` here -- which is
//     what `kimi_linear_registry.cpp:156` publishes for ITS model -- would give
//     the runner a cache one column short of what the layer reads.
//
// ONE CONV STATE, NOT THREE. The checkpoint stores `self_attn.{q,k,v}_conv1d`
// separately and the reference declares ONE grouped depthwise conv over the
// concatenated `[q; k; v]` channel axis (`modeling_glm5_next.py:620-628`), so
// the CACHE is one `3 * num_heads * head_dim` channel state. An earlier
// revision of this function's refusal said "three separate conv states", which
// would have tripled this group; `glm5_next_kda.h` "THREE LAYOUT FACTS" settles
// it and the case below pins the single width.
//
// GROUP 2 IS AN `MLAAttentionSpec` AND THAT IS LOAD-BEARING, not an MLA claim.
// `MLAAttentionSpec` is the key-only page budget -- one vector per stored state
// instead of a K+V pair. A `FullAttentionSpec` in that position is absorbed by
// the runner's leftover scan as the single `fa_draft` draft-KV slot
// (`src/vllm/v1/worker/gpu/runner.cpp`, the `draft_slot_taken` arm, which
// `continue`s); the leftover count then stays 0, `multi_cache_topology` stays
// false, and the side cache is published and never allocated with nothing
// reported. `MODEL-MM-QWEN4-EXP` W5c-1 (#2206) measured that on its own third
// group and the same arm is live here.
//
// ITS ROW IS 257 WIDE AND `compress_ratio` IS 1. `PackIndexerStates`
// (`glm5_next_dsa.h`) stores `concat[k(head_dim), gate_scores(head_dim),
// valid(1)]` PER TOKEN (`modeling_glm5_next.py:798-801`), so 2 * 128 + 1 = 257
// elements and one row per token. The k-pool stage compresses at READ time
// inside `GetPooledStates`, not at store time, so nothing here divides by
// `index_kpool` -- which is the opposite of `MODEL-MM-QWEN4-EXP`'s QSA side
// cache, where the compression IS in the store and `compress_ratio` is 4. Our
// DeepSeek-V4 parent stores 128 (the key alone); reading that number across
// would under-allocate this cache by half.
//
// REAL PER-LAYER NAMES, NEVER PLACEHOLDERS. `ResolveKVCacheGroupLayerNames`
// (`src/vllm/v1/kv_cache_interface.cpp`) rewrites a placeholder group set into
// per-layer names, but its fallback can name only a TARGET attention group and
// one `fa_draft` slot: a third attention group gets `layer_names.clear()` and
// the runner then refuses the unnamed group. Publishing the real names also
// makes the rewrite a no-op by its own idempotence guard, so what the runner
// allocates is what this function said. #1963/#1966 are the standing reason a
// KV arithmetic here is re-derived against the runner rather than trusted.
v1::KVCacheConfig MakeGlm5NextKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  // The row's own resolve-and-validate, not a second reading of the raw config.
  // It is what rewrites `full_attention` into `deepseek_sparse_attention`, so
  // the classification below is upstream's post-`__post_init__` one.
  const Glm5NextParams p = ParseGlm5NextParams(config);

  VT_CHECK(block_size > 0,
           "glm5_next KV spec: block_size must be positive, got " +
               std::to_string(block_size));

  std::vector<std::string> dsa_layers;
  std::vector<std::string> dsa_indexer_layers;
  std::vector<std::string> kda_layers;
  for (size_t l = 0; l < p.layer_types.size(); ++l) {
    const std::string idx = std::to_string(l);
    if (p.layer_types[l] == Glm5NextLayerKind::kLinearAttention) {
      // The name `ResolveKVCacheGroupLayerNames` builds for a recurrent layer,
      // so the runner's by-name membership sees the same string either way.
      kda_layers.push_back("model.layers." + idx + ".linear_attn");
    } else {
      dsa_layers.push_back("model.layers." + idx + ".self_attn.attn");
      // Upstream addresses a side cache by its own module prefix
      // (`vllm/models/deepseek_v4/attention.py:761-767` registers the indexer
      // key cache under `...indexer.k_cache`); the runner parses the
      // `.layers.<N>.` segment out of it, so the suffix is free to say which
      // cache it is.
      dsa_indexer_layers.push_back("model.layers." + idx +
                                   ".self_attn.indexer.k_cache");
    }
  }

  VT_CHECK(!dsa_layers.empty(),
           "glm5_next KV spec: the config declares no deepseek_sparse_attention "
           "layer, so there is no MLA latent to publish. See "
           ".agents/specs/glm5-next-flash.md and issue #2223.");
  VT_CHECK(!kda_layers.empty(),
           "glm5_next KV spec: the config declares no linear_attention layer, "
           "so there is no KDA recurrent state to publish. See "
           ".agents/specs/glm5-next-flash.md and issue #2223.");

  const int64_t mla_head_size = p.mla.kv_lora_rank + p.mla.qk_rope_head_dim;
  VT_CHECK(mla_head_size > 0,
           "glm5_next KV spec: the MLA latent row is " +
               std::to_string(mla_head_size) +
               " wide (kv_lora_rank + qk_rope_head_dim); a non-positive latent "
               "would publish a zero-byte page the runner allocates and the "
               "attention block then writes past.");

  const int64_t kda_conv_dim = 3 * p.kda.num_heads * p.kda.head_dim;
  VT_CHECK(p.kda.num_heads > 0 && p.kda.head_dim > 0 &&
               p.kda.conv_kernel_dim > 0,
           "glm5_next KV spec: the config declares " +
               std::to_string(kda_layers.size()) +
               " linear_attention layer(s) but no complete `linear_attn_config` "
               "group, so the KDA recurrent state cannot be sized "
               "(linear_num_heads=" +
               std::to_string(p.kda.num_heads) + " linear_head_dim=" +
               std::to_string(p.kda.head_dim) + " linear_conv_kernel_dim=" +
               std::to_string(p.kda.conv_kernel_dim) + ").");

  // `2 * index_head_dim + 1` — see the "257 WIDE" note above.
  const int64_t indexer_row = 2 * p.indexer.head_dim + 1;
  VT_CHECK(p.indexer.head_dim > 0,
           "glm5_next KV spec: the config declares " +
               std::to_string(dsa_layers.size()) +
               " deepseek_sparse_attention layer(s) but `index_head_dim` is " +
               std::to_string(p.indexer.head_dim) +
               ", so the DSA indexer side cache cannot be sized. See "
               ".agents/specs/glm5-next-flash.md and issue #2223.");

  // KDA, NOT GATED DELTA NET, and the two recurrent dtypes follow from that.
  // The mirror is `MambaStateShapeCalculator.kda_state_dtype`
  // (`mamba_utils.py:130-137`), which is the pair
  // `(get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype), torch.float32)`
  // and is exactly what `kimi_linear_registry.cpp:161` publishes for the OTHER
  // KDA model in this tree. So the CONV half follows the paged-KV storage dtype
  // -- model-dtype bf16 by default, f32 under `VT_KV_CACHE_F32`, the
  // fold-identity A/B -- and the RECURRENT half is f32 unconditionally.
  //
  // `detail::ResolveMambaSsmCacheDType`, and with it `HfConfig::mamba_ssm_dtype`,
  // is deliberately NOT called here. That helper mirrors `_mamba_state_dtype`
  // (`mamba_utils.py:96-108`), the Mamba/GDN calculator, and `kda_state_dtype`
  // takes no `mamba_ssm_cache_dtype` parameter at all: honouring the key for a
  // KDA cache would be an invention rather than a port, and a `bfloat16` value
  // in some future `config.json` would then silently halve a state upstream
  // keeps in f32. Qwen3.5 (`qwen3_5_common.cpp:54`) and MODEL-MM-QWEN4-EXP
  // (`qwen4_exp_registry.cpp:433`) call the resolver because their linear layers
  // ARE gated delta net (`gated_delta_net_state_dtype`, `mamba_utils.py:119-128`),
  // which does read it. An earlier revision of this comment claimed this
  // function called that resolver; it never did, and the claim is retired here
  // rather than made true, because the GDN calculator is the wrong one.
  //
  // Why the recurrent f32 is not negotiable: upstream annotates the cast twice.
  // `cache_params.update_recurrent_state(last_recurrent_state.to(torch.float32),
  // ...)` casts explicitly (`modeling_glm5_next.py:739`) and `:452` says
  // "calculations happen in float as states are more susceptible to rounding
  // errors". The state is a running sum over the whole sequence, so a bf16 store
  // accumulates an error with no way out -- and a token gate cannot see it.
  const vt::DType conv_dtype = v1::ResolveKvCacheDType();
  const vt::DType ssm_dtype = vt::DType::kF32;

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::move(dsa_layers),
      std::make_shared<v1::MLAAttentionSpec>(
          block_size, static_cast<int>(mla_head_size), v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::move(kda_layers),
      std::make_shared<v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{
              // ONE grouped [q; k; v] conv state, `conv_kernel_dim` columns.
              {kda_conv_dim, p.kda.conv_kernel_dim},
              // The delta-rule recurrent state, [heads, head_dim, head_dim].
              {p.kda.num_heads, p.kda.head_dim, p.kda.head_dim}},
          std::vector<vt::DType>{conv_dtype, ssm_dtype}));
  kv.kv_cache_groups.emplace_back(
      std::move(dsa_indexer_layers),
      std::make_shared<v1::MLAAttentionSpec>(
          block_size, static_cast<int>(indexer_row), v1::ResolveKvCacheDType(),
          /*num_kv_heads=*/1, v1::KVQuantMode::kNone,
          /*page_size_padded=*/std::nullopt,
          /*indexes_kv_by_block_stride=*/false,
          /*cache_dtype_str=*/std::nullopt, /*alignment=*/std::nullopt,
          // ONE stored row PER TOKEN: the k-pool compresses at read time.
          /*compress_ratio=*/1, /*model_version=*/std::nullopt));
  return kv;
}

const ModelFactory kGlm5NextFactory{
    .parse_config = &ParseGlm5NextConfig,
    .load_weights = &LoadGlm5NextForConditionalGeneration,
    .prepare = &PrepareGlm5NextForConditionalGeneration,
    .forward = &ForwardGlm5NextForConditionalGeneration,
    .make_kv_cache = &MakeGlm5NextKVCache,
    .is_dense_model = false,
    // W5b-2c (#2348): this forward READS the cache set keyed by published layer
    // name, so `ModelRegistry::Forward` stops refusing the step above it. The
    // consuming code is `glm5_next_kv.cpp` and the refusals it can raise are
    // all by name.
    .consumes_multi_kv = true,
};

}  // namespace

REGISTER_VLLM_MODEL(glm5_next, "Glm5NextForConditionalGeneration",
                    kGlm5NextFactory, kGlm5NextInfo)

}  // namespace vllm
