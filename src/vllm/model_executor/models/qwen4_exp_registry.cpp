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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen3_5_internal.h"  // ResolveMambaSsmCacheDType
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vllm/v1/kv_cache_dtype.h"     // ResolveKvCacheDType
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

// ─── The KV-cache spec (W5c, #2031) ──────────────────────────────────────────
//
// THREE published groups, and the shape of them is the decision this function
// exists to record:
//
//   0. the QSA layers' paged K+V              `FullAttentionSpec`
//   1. EVERY linear-attention layer's state   `MambaSpec`, N states
//   2. the QSA layers' indexer side cache     `MLAAttentionSpec`, compress 4
//
// ONE UNIFORM RECURRENT GROUP, NOT ONE PER LAYER, AND THE COST IS DELIBERATE.
// Only ONE linear-attention layer carries the PLE conv and the n-gram token
// history (`ple_layer_ids` selects 0-based layer 1 on the published
// checkpoint), so a per-layer spec would give 35 of the 36 recurrent layers a
// smaller state set. Upstream cannot express that and does not try:
// `get_mamba_state_shape_from_config` is a CLASSMETHOD taking only the config
// (`vllm/model_executor/models/interfaces.py:809-812` at the pin
// `5559679229`), all 18 implementations of it declare ONE shape model-wide and
// not one of them takes a `layer_idx`, and `get_mamba_groups`
// (`vllm/v1/worker/mamba_utils.py:441`) asserts
// `all(mamba_specs[0] == spec for spec in mamba_specs)` — every recurrent spec
// in the model equal, field for field. Upstream pays the uniform cost by
// PADDING rather than by splitting (`vllm/v1/core/kv_cache_utils.py:1101-1109`
// sets `page_size_padded=max_page_size` on the smaller `MambaSpec`).
//
// The cost here, derived from the published shapes rather than measured: the
// PLE conv is `10240 x 9` at bf16 = 184320 B and the n-gram history is 2 int64
// = 16 B, so 184336 B per sequence on each of the 35 linear layers that do not
// use them. At the default `max_num_seqs` of 8 that is 49.2 MiB — 0.09% of the
// GB10 headroom the row's `## Hardware` section accounts. Splitting the group
// to recover it would need a SECOND recurrent group, which
// `.agents/specs/recurrent-multistate.md` records as owed generic engine debt
// and which this topology does not need.
//
// STATE ORDER IS A DELIBERATE DIVERGENCE FROM UPSTREAM'S LIST ORDER, and it is
// the same bytes either way. Upstream keeps the three CONV states adjacent
// (`number_of_conv_states = 3`: GDN conv, PLE conv, n-gram history) with the
// temporal state after them. This tree publishes
// `[gdn_conv, temporal, ple_conv, ngram]` because `GdnStateCache` exposes
// `conv_state = states[0]` and `ssm_state = states[1]` as NAMED fields that
// THREE model families already read (`qwen3_5.cpp`, `kimi_linear_device.cpp`
// and the `nemotron_h` pair `nemotron_h_device.cpp` / `nemotron_h_forward.h`),
// and moving the temporal state off slot 1 would silently re-point every one
// of them. Recorded in the row spec.
//
// THE COUNT IS THREE, NOT FOUR (#2203). `gemma4_mm.cpp` was named here and in
// `.agents/specs/recurrent-multistate.md` as a fourth reader, and it reads
// NEITHER field: its only two mentions of the type are an include comment and
// `std::vector<GdnStateCache> no_gdn_state;` (`gemma4_mm.cpp:221`), passed
// EMPTY, which is the file proving Gemma-4 has no recurrent arm.
// `muse_glimmer_mm.cpp:340` and `qwen3_vl.cpp:621` carry that same empty-vector
// shape. Grepping the FIELD name over-counts in the other direction:
// `glm5_next_kda.cpp` matches `conv_state` 13 times on
// `Glm5NextKdaCache::conv_state`, a `std::vector<float>` KDA sequence state
// (`glm5_next_kda.h:314`), where this one is a `vt::Tensor` (`qwen3_5.h:111`),
// and that file has zero occurrences of `GdnStateCache`. Grep the TYPE.
//
// REAL PER-LAYER NAMES, NEVER PLACEHOLDERS. `ResolveKVCacheGroupLayerNames`
// (`src/vllm/v1/kv_cache_interface.cpp`) rewrites a placeholder group set into
// per-layer names, but its fallback classification can name only a TARGET
// attention group and one `fa_draft` slot: a third attention group gets
// `layer_names.clear()` and an unnamed group is then refused by the runner's
// multi-cache admission check, because its names "do not all resolve to
// distinct in-range layer indices". Publishing the real names also makes the
// rewrite a no-op by its own idempotence guard, so what the runner allocates is
// what this function said.
//
// GROUP 2 IS AN `MLAAttentionSpec` AND THAT IS LOAD-BEARING. `MLAAttentionSpec`
// is not an MLA claim — it is the key-only page budget, one vector per stored
// state instead of a K+V pair — and `compress_ratio` is what makes a state
// cover four tokens (`vllm/v1/kv_cache_interface.py:386` and the
// `storage_block_size = block_size // compress_ratio` property at `:393-395`).
// A `FullAttentionSpec` here would be absorbed by the runner as the single
// `fa_draft` draft-KV slot instead (`gpu/runner.cpp`, the `draft_slot_taken`
// arm of the leftover scan), `multi_cache_topology` would stay false, and the
// side cache would be published and never allocated — in silence.
v1::KVCacheConfig MakeQwen4ExpKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  // The row's own resolve-and-validate, not a second reading of the raw config.
  // It is what rewrites `full_attention` into `qwen_sparse_attention`, so the
  // classification below is upstream's post-`__post_init__` one.
  const Qwen4ExpParams p = ParseQwen4ExpParams(config);

  VT_CHECK(block_size > 0,
           "qwen4_exp KV spec: block_size must be positive, got " +
               std::to_string(block_size));

  std::vector<std::string> qsa_layers;
  std::vector<std::string> qsa_indexer_layers;
  std::vector<std::string> linear_layers;
  for (size_t l = 0; l < p.layer_types.size(); ++l) {
    const std::string idx = std::to_string(l);
    if (p.layer_types[l] == Qwen4ExpLayerKind::kLinearAttention) {
      // The name `ResolveKVCacheGroupLayerNames` builds for a recurrent layer,
      // so the runner's by-name membership sees the same string either way.
      linear_layers.push_back("model.layers." + idx + ".linear_attn");
    } else {
      qsa_layers.push_back("model.layers." + idx + ".self_attn.attn");
      // Upstream addresses a side cache by its own module prefix
      // (`vllm/models/deepseek_v4/attention.py:761-767` registers the indexer
      // key cache under `...indexer.k_cache`); the runner parses the
      // `.layers.<N>.` segment out of it, so the suffix is free to say which
      // cache it is.
      qsa_indexer_layers.push_back("model.layers." + idx +
                                   ".self_attn.indexer.k_cache");
    }
  }

  VT_CHECK(!qsa_layers.empty(),
           "qwen4_exp KV spec: the config declares no qwen_sparse_attention "
           "layer, so there is no attention KV to publish. See "
           ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");
  VT_CHECK(!linear_layers.empty(),
           "qwen4_exp KV spec: the config declares no linear_attention layer, "
           "so there is no recurrent state to publish. See "
           ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");

  // QSA is optional as a WHOLE in the config layer (all five `indexer_*` fields
  // or none), while the `full_attention` -> `qwen_sparse_attention` rewrite is
  // unconditional. So a config CAN declare sparse layers and no indexer, and
  // that combination has no side cache to size. Refuse rather than publish two
  // groups where the model needs three.
  VT_CHECK(p.qsa.compress_ratio > 0 && p.qsa.head_dim > 0 &&
               p.qsa.kv_heads > 0,
           "qwen4_exp KV spec: the config declares " +
               std::to_string(qsa_layers.size()) +
               " qwen_sparse_attention layer(s) but no `indexer_*` group, so "
               "the QSA indexer side cache cannot be sized. See "
               ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");

  // `MLAAttentionSpec::storage_block_size()` is `block_size / compress_ratio`,
  // an INTEGER division that truncates in silence
  // (`vllm/v1/kv_cache_interface.py:393-395`). At a block size the ratio does
  // not divide, the page is sized for `floor(block/ratio)` states while the
  // block still covers `block` tokens, so the last partial state's key has
  // nowhere to go — a short cache, i.e. wrong tokens rather than a crash.
  // Upstream never meets this because its DeepSeek-V4 block sizes are powers of
  // two above the ratio; ours arrives as a caller-supplied parameter.
  VT_CHECK(block_size % p.qsa.compress_ratio == 0,
           "qwen4_exp KV spec: block_size " + std::to_string(block_size) +
               " is not a multiple of `indexer_compress_ratio` " +
               std::to_string(p.qsa.compress_ratio) +
               "; the indexer side cache stores one state per " +
               std::to_string(p.qsa.compress_ratio) +
               " tokens and storage_block_size() would truncate.");

  // The recurrent state set, in the order stated above.
  //
  // The two dtypes come from the SAME resolver every other hybrid in this tree
  // uses, rather than a second reading of `mamba_ssm_dtype`; its refusal
  // message is spelled `qwen3_5:` because that is where the one copy lives.
  const vt::DType conv_dtype = vt::DType::kBF16;
  const vt::DType ssm_dtype =
      detail::ResolveMambaSsmCacheDType(config, conv_dtype);

  std::vector<std::vector<int64_t>> state_shapes{
      // GDN conv: the concatenated q|k|v stream, `conv_kernel - 1` taps.
      {p.linear_conv_dim(), p.linear_conv_kernel_dim - 1},
      // GDN temporal.
      {p.linear_num_value_heads, p.linear_value_head_dim,
       p.linear_key_head_dim},
  };
  std::vector<vt::DType> state_dtypes{conv_dtype, ssm_dtype};

  // `number_of_conv_states` is 3 exactly when the model has a PLE layer, and 1
  // otherwise (`Qwen4ExpParams::number_of_conv_states`, mirroring upstream).
  // The two extra conv states are the PLE conv and the n-gram token history,
  // which upstream keeps in the linear-attention cache beside the GDN conv
  // because the state manipulations are identical (`modular_qwen4_exp.py`
  // :178-180).
  if (p.number_of_conv_states() == 3) {
    // The PLE conv is DILATED by `ngram_size`, so its state is
    // `(kernel - 1) * ngram_size` = 9 columns deep, not `kernel - 1`, and it
    // runs over the FULL hyper-connection stream width.
    state_shapes.push_back(
        {p.stream_width(), p.ple.short_conv_state_len()});
    state_dtypes.push_back(conv_dtype);
    // TOKEN IDS, and `kI64` is not a widening. The history holds
    // `input_ids.long()` and feeds a `uint64_t` hash multiply; storing it in a
    // float dtype rounds a token id, which the row spec records as one of the
    // three silent divergence sites. `ENG-RECURRENT-MULTISTATE` (#2131) is what
    // made an integer recurrent state expressible at all.
    state_shapes.push_back({p.ple.ngram_size - 1});
    state_dtypes.push_back(vt::DType::kI64);
  }

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::move(qsa_layers),
      std::make_shared<v1::FullAttentionSpec>(
          block_size, static_cast<int>(p.num_key_value_heads),
          static_cast<int>(p.head_dim), v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::move(linear_layers),
      std::make_shared<v1::MambaSpec>(block_size, std::move(state_shapes),
                                      std::move(state_dtypes)));
  kv.kv_cache_groups.emplace_back(
      std::move(qsa_indexer_layers),
      std::make_shared<v1::MLAAttentionSpec>(
          block_size, static_cast<int>(p.qsa.head_dim),
          v1::ResolveKvCacheDType(), static_cast<int>(p.qsa.kv_heads),
          v1::KVQuantMode::kNone, /*page_size_padded=*/std::nullopt,
          /*indexes_kv_by_block_stride=*/false,
          /*cache_dtype_str=*/std::nullopt, /*alignment=*/std::nullopt,
          static_cast<int>(p.qsa.compress_ratio),
          /*model_version=*/std::nullopt));
  return kv;
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
