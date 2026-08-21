// Nemotron-H (`NemotronHForCausalLM`) registry TU — the ADDITIVE
// self-registration seam for the W3 structural bring-up (#517,
// .agents/specs/nemotron-h-model.md §4 W3). Follows the
// kimi_linear_registry.cpp / deepseek_v2_registry.cpp seam exactly: a NEW
// translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array. It owns the arch entry points: the config hook, the
// HETEROGENEOUS KV-cache spec (a full-attention group over the 6 GQA layers +
// a Mamba2 recurrent-state group over the 23 mamba layers), the LoadedModel
// subclass and the factory.
//
// W3 registered the arch so it RESOLVES, parses its config and enumerates its
// checkpoint. A2-P (#810,
// .agents/specs/nemotron-h-a2p-paged-forward.md) is where it FORWARDS on the
// runner's own caches: `ForwardNemotronHForCausalLM` now selects
// `NemotronHPagedForward` whenever the runner supplies paged KV and recurrent
// state, and the safety interlock below is NARROWED to `num_reqs <= 1` rather
// than deleted — batching is A2-B's. The host reference stays alive below the
// fold as the operand the numeric gate compares against.
//
// Still refusing BY NAME, and each names its owner: the GGUF arm (spec §5b W7 —
// a silent dequantization to a supported path is exactly what a token gate
// cannot see), the MTP head (W5), and the device `lm_head` (A2-Q2b), which is
// why this forward still returns HOST logits and
// `scripts/runner-routing-allowlist.txt` is narrowed rather than removed.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"  // IsNemotronHGguf
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vllm/model_executor/models/nemotron_h_loader.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py `_ModelInfo` for NemotronH (registry.py:179 ->
// models/nemotron_h.py::NemotronHForCausalLM): text generation, HYBRID (23
// Mamba2 layers ⇒ a recurrent-state KV group), not multimodal. Upstream's class
// carries `HasInnerState` + `IsHybrid` (nemotron_h.py:700-712); our ModelInfo is
// a consumed subset whose only `has_inner_state` reader short-circuits on
// `is_hybrid`, so this follows the established hybrid-recurrent registration
// convention (kQwen3_5Info, kKimiLinearInfo).
inline constexpr ModelInfo kNemotronHInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class NemotronHLoadedModel final : public LoadedModel {
 public:
  NemotronHLoadedModel(const ModelRegistration& registration,
                       NemotronHParams params)
      : LoadedModel(registration), params_(std::move(params)) {}
  const NemotronHParams& params() const { return params_; }
  // W4 ports the forward MECHANISM (nemotron_h.cpp). The weight LOAD that fills
  // this — the 18487 enumerated tensors, NVFP4 W4A16 g16 experts and FP8 W8A8
  // mamba projections included — is still owed, so `materialized` stays false on
  // the checkpoint path and `NemotronHForward` refuses by name. A direct caller
  // (the unit gate) constructs the weights itself and reaches the same forward.
  NemotronHHostWeights& weights() { return weights_; }
  const NemotronHHostWeights& weights() const { return weights_; }
  // What the load did, in numbers. Kept on the model rather than returned by
  // value because the load happens inside the type-erased registry factory and
  // a gate has no other way to reach it; `NemotronHLoadReportOf` is the
  // accessor.
  NemotronHLoadReport& report() { return report_; }
  const NemotronHLoadReport& report() const { return report_; }

 private:
  NemotronHParams params_;
  NemotronHHostWeights weights_;
  NemotronHLoadReport report_;
};

std::unique_ptr<LoadedModel> LoadNemotronHForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    // AGENTS.md makes GGUF k-quants a standing requirement, not a per-model
    // choice; it is OWED here (spec §5b, W7) and refused BY NAME rather than
    // routed to a supported path behind the caller's back. The text lives in
    // `NemotronHGgufRefusal` because the entrypoint's GGUF architecture
    // dispatch throws the SAME refusal before it ever gets here (#809): a real
    // `nemotron_h*` file is refused at the door it actually arrives at, and
    // this guard still covers a direct `Kind::kGguf` caller.
    throw std::runtime_error(NemotronHGgufRefusal());
  }
  // The config descent IS the validation, and it refuses by name on anything
  // this bring-up cannot represent.
  NemotronHParams params = ParseNemotronHParams(config);
  auto model =
      std::make_unique<NemotronHLoadedModel>(registration, params);
  if (source.safetensors == nullptr) {
    throw std::runtime_error(
        "Model architecture NemotronHForCausalLM: the safetensors source "
        "carries no shards");
  }
  // MATERIALIZE. The 18487 enumerated tensors are read into the host weights in
  // the memory format the checkpoint SHIPS them in — NVFP4 W4A16 g16 experts and
  // lm_head, FP8 W8A8 static mamba projections, bf16 everything else — and the
  // MTP tower is deferred by name (W5). See nemotron_h_loader.h.
  model->weights() = LoadNemotronHHostWeights(
      *source.safetensors, params, ResolveNemotronHModelDType(config),
      &model->report());
  return model;
}

void PrepareNemotronHForCausalLM(LoadedModel& model, const HfConfig& config,
                                 vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardNemotronHForCausalLM(LoadedModel& model,
                                          const ModelForwardInput& input) {
  // ★ G-SAFE (#810) — THE SAFETY INTERLOCK, NARROWED BY A2-P
  // (.agents/specs/nemotron-h-a2p-paged-forward.md §1, §6).
  //
  // It began with three clauses. A2-P CONSUMES two of them in this same change:
  // `NemotronHPagedForward` below writes this step's K/V into `input.attn_kv`
  // at the runner's slot mapping and reads attention back out of those pages,
  // and it gathers the conv/SSM rows out of `input.gdn_state` at the step's
  // state indices and scatters the updated rows back. Those two clauses existed
  // because no such forward existed; it exists now, so they go.
  //
  // `input.num_reqs <= 1` STAYS, and A2-B removes it. Nothing in A2-P reorders
  // a batch, splits decodes from prefills across requests, or serves more than
  // one sequence's recurrent state in a step. A multi-request batch reaching
  // the host reference would be decoded as ONE concatenated causal sequence —
  // fluent output, wrong tokens, no error — which is exactly the failure this
  // guard is for and exactly what a token gate cannot see.
  //
  // The MESSAGE is rewritten with the predicate. Leaving the old text beside a
  // two-thirds-smaller check is how a message stops describing what it
  // enforces, which is the drift AGENTS.md §"Changing the rules or a checker"
  // exists to prevent.
  //
  // The guard runs BEFORE `ModelAs` deliberately. It reads only `input` and
  // never touches `model`, so #775's guarantee — no member call before the
  // dynamic type is established — is untouched; and putting it first is what
  // makes it reachable from a test without fabricating a look-alike
  // `NemotronHLoadedModel`, which is exactly the stub #784 removed. Order is
  // not part of the G-SAFE requirement; being gated is.
  VT_CHECK(
      input.num_reqs <= 1,
      "Model architecture NemotronHForCausalLM: BATCHED decode is not ported "
      "(issue #810, .agents/specs/nemotron-h-a2p-paged-forward.md A2-B). The "
      "paged forward carries one request's KV pages and one request's recurrent "
      "state per step; it does not reorder a batch or split decodes from "
      "prefills across requests, so a multi-request step would be decoded as "
      "ONE concatenated causal sequence and would return plausible WRONG tokens "
      "instead of failing. Refusing by name until A2-B lands.");
  // #775: CHECKED, not `static_cast`. A bare downcast down this hierarchy is a
  // promise the compiler is entitled to act on, so on a model that is not
  // really a `NemotronHLoadedModel` every `nh.` member call below is undefined
  // behaviour — and it happens on the way to a refusal that throws anyway,
  // which is what kept it invisible outside the sanitizer lane. `ModelAs`
  // establishes the dynamic type first and refuses by name instead.
  auto& nh = ModelAs<NemotronHLoadedModel>(model, "NemotronHForCausalLM");
  // ── A2-P: THE PAGED FOLD ───────────────────────────────────────────────────
  //
  // Mirrors `ForwardKimiLinearForCausalLM` (kimi_linear_registry.cpp:99-102),
  // which is the only in-tree instance of exactly this shape: the runner's
  // caches SELECT the paged path, the historical seams stay alive below it, and
  // the paged entry point takes `input` WHOLE.
  //
  // THREE CLAUSES, as that idiom has — and the third is the residency one.
  // `nh.weights().materialized` is this model's analogue of Kimi's
  // `weights.resident.resident`: it is what says the tensors this forward is
  // about to upload through `dense_attn::ResidentWeight` actually exist. A
  // non-materialized model falls through to the host reference, which refuses
  // by name on the same condition, so the missing piece is still reported
  // rather than computed on zeros.
  //
  // ONE DELIBERATE DIFFERENCE FROM THE IDIOM, and it is a safety one:
  // `input.gather_logits` is NOT a clause here. Kimi's paged branch needs it
  // because its paged fold returns DEVICE logits; ours returns host logits
  // either way (`lm_head` is NVFP4 and A2-Q2b owns its device arm), and an
  // empty `logits_indices` already means "every row" to
  // `NemotronHPagedForward`. Including the flag would let a step under
  // VT_LOGITS_GATHER=0 arrive with full paged caches and fall through to the
  // host reference — which, with the G-SAFE cache clauses now consumed, would
  // silently return the wrong tokens. That is the precise hazard the interlock
  // was built for, so the flag is left out and this branch serves both settings.
  if (!input.attn_kv.empty() && !input.gdn_state.empty() &&
      nh.weights().materialized) {
    return NemotronHPagedForward(nh.weights(), nh.params(), input);
  }
  // The HOST REFERENCE, unchanged and deliberately kept below the fold exactly
  // as Kimi-Linear keeps its own: it is the operand the numeric gate compares
  // against, and deleting it deletes the gate. It consumes three of
  // `ModelForwardInput`'s fields — `token_ids`, `logits_indices`, `queue` — and
  // is reached only by a direct caller with no paged caches (the CLI and unit
  // vehicles). `NemotronHForward` refuses BY NAME when the host weights are not
  // materialized rather than returning a silent zero forward.
  return HostLogits(NemotronHForward(nh.weights(), nh.params(), input.token_ids,
                                     input.logits_indices, input.queue),
                    nh.params().vocab_size);
}

const ModelFactory kNemotronHFactory{
    .parse_config = &ParseNemotronHConfig,
    .load_weights = &LoadNemotronHForCausalLM,
    .prepare = &PrepareNemotronHForCausalLM,
    .forward = &ForwardNemotronHForCausalLM,
    .make_kv_cache = &MakeNemotronHKVCache,
    .is_dense_model = false,
};

}  // namespace

// The GGUF-side half of the arch entry points. Kept in THIS TU, next to the
// factory whose guard throws the same string, so the refusal has one owner and
// the entrypoint dispatch borrows it instead of restating it.
bool IsNemotronHGguf(const GgufFile& gguf) {
  const GgufValue* arch = gguf.FindKv("general.architecture");
  if (arch == nullptr || arch->TypeId() != kGgufString) return false;
  const std::string& name = std::get<std::string>(arch->v);
  return name == kNemotronHGgufArch || name == kNemotronHMoeGgufArch;
}

std::string NemotronHGgufRefusal() {
  return "Model architecture NemotronHForCausalLM does not support GGUF "
         "weights yet: the GGUF k-quant/i-quant arm is not ported (see "
         ".agents/specs/nemotron-h-model.md §5b W7)";
}

const NemotronHLoadReport& NemotronHLoadReportOf(const LoadedModel& model) {
  const auto* nh = dynamic_cast<const NemotronHLoadedModel*>(&model);
  if (nh == nullptr) {
    throw std::runtime_error(
        "NemotronHLoadReportOf: this LoadedModel is not a NemotronH model");
  }
  return nh->report();
}

v1::KVCacheConfig MakeNemotronHKVCache(const HfConfig& config, int block_size,
                                       int num_blocks) {
  const NemotronHParams p = ParseNemotronHParams(config);

  // Both state tensors' dtypes come from `_mamba_state_dtype`
  // (mamba_utils.py:93-106): the CONVOLUTION state follows the cache dtype
  // (the model dtype, bf16, unless overridden), and the TEMPORAL/SSM state is
  // resolved INDEPENDENTLY from `mamba_ssm_cache_dtype` — "float32" on this
  // checkpoint, so the two differ. Collapsing the SSM state to the activation
  // dtype is a silent precision loss a token gate can absorb. See
  // NemotronHSsmCacheDType for why the shared qwen3_5 resolver is NOT the right
  // reader here (it is keyed on a different config spelling).
  const vt::DType conv_dtype = vt::DType::kBF16;
  const vt::DType ssm_dtype = NemotronHSsmCacheDType(p, conv_dtype);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;

  // (1) the 6 GQA full-attention layers. `FullAttentionSpec` sizes the paged
  //     K+V page; the fp8 KV scheme the checkpoint ships (k_scale/v_scale) is a
  //     W4/W6 storage decision and is deliberately NOT selected here.
  std::vector<std::string> attn_layers;
  for (int64_t i : p.LayerIndices(NemotronHBlock::kAttention)) {
    attn_layers.push_back("backbone.layers." + std::to_string(i) + ".mixer");
  }
  kv.kv_cache_groups.emplace_back(
      std::move(attn_layers),
      std::make_shared<v1::FullAttentionSpec>(
          block_size, static_cast<int>(p.num_key_value_heads),
          static_cast<int>(p.head_dim), v1::ResolveKvCacheDType()));

  // (2) the 23 Mamba2 layers. mamba2_state_shape (mamba_utils.py:173-198) at
  //     tp_world_size 1:
  //       conv  = (conv_dim, conv_kernel - 1 + num_spec)
  //       state = (num_heads, head_dim, state_size)
  //     with conv_dim = mamba_num_heads*mamba_head_dim + 2*n_groups*state_size
  //     = 4096 + 2*8*128 = 6144, confirmed on disk by the released
  //     `mixer.conv1d.weight` BF16 [6144, 1, 4].
  //
  //     LAYOUT: this is upstream's `DS` mode, not a local invention. Upstream
  //     defines `ConvStateLayoutType = Literal["SD", "DS"]`
  //     (mamba_utils.py:23), selects between them with
  //     `VLLM_SSM_CONV_STATE_LAYOUT` (envs.py:227), and orients the shape at
  //     `mamba_utils.py:152-157`: `(dim, state_len)` for DS,
  //     `(state_len, dim)` for SD. `SD` is the DEFAULT (:43) and is transposed
  //     back to dim-major on the way into the kernels
  //     (mamba_mixer2.py:714-721), because the kernels want dim-major either
  //     way. Ours is `(dim, state_len)` = **DS**, i.e.
  //     `VLLM_SSM_CONV_STATE_LAYOUT=DS` — a first-class upstream mode, and the
  //     same orientation qwen3_5_common.cpp:85 and kimi_linear_registry.cpp:156
  //     already use, so the shared runner and manager code sees one. The BYTES
  //     are the same product either way, which is what upstream's own
  //     `test_ds_conv_layout_bias_gt_0_byte_equal_to_sd`
  //     (tests/v1/worker/test_mamba_utils.py:2136, a method of
  //     `TestPostprocessMambaFusedKernel` at :410) asserts and what the ported
  //     twin in tests/vllm/models/test_nemotron_h_paged_forward.cpp gates here.
  //
  //     num_spec is 0: speculative decoding widens the conv row to
  //     (K-1)+k taps, and the MTP head is W5.
  std::vector<std::string> mamba_layers;
  for (int64_t i : p.LayerIndices(NemotronHBlock::kMamba)) {
    mamba_layers.push_back("backbone.layers." + std::to_string(i) + ".mixer");
  }
  kv.kv_cache_groups.emplace_back(
      std::move(mamba_layers),
      std::make_shared<v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{
              {p.conv_dim(), p.conv_kernel - 1},
              {p.mamba_num_heads, p.mamba_head_dim, p.ssm_state_size}},
          std::vector<vt::DType>{conv_dtype, ssm_dtype}));
  return kv;
}

REGISTER_VLLM_MODEL(nemotron_h, "NemotronHForCausalLM", kNemotronHFactory,
                    kNemotronHInfo)

}  // namespace vllm
