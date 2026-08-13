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
// W3 registers the arch so it RESOLVES, parses its config and enumerates its
// checkpoint. It does NOT forward: `ForwardNemotronHForCausalLM` REFUSES BY
// NAME (VT_CHECK(false), exactly like kimi_linear / deepseek_v4 / kimi_k3), so
// the TU builds and the structure is unit-testable while a forward LOUDLY
// reports the pending brick instead of returning a silent wrong answer. The
// GGUF arm refuses by name too — it is OWED (spec §5b W7), and a silent
// dequantization to a supported path is exactly what a token gate cannot see.
// The model-matrix row stays INVENTORIED until W4-W6 land.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits
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

 private:
  NemotronHParams params_;
};

std::unique_ptr<LoadedModel> LoadNemotronHForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    // AGENTS.md makes GGUF k-quants a standing requirement, not a per-model
    // choice; it is OWED here (spec §5b, W7) and refused BY NAME rather than
    // routed to a supported path behind the caller's back.
    throw std::runtime_error(
        "Model architecture NemotronHForCausalLM does not support GGUF weights "
        "yet: the GGUF k-quant/i-quant arm is not ported (see "
        ".agents/specs/nemotron-h-model.md §5b W7)");
  }
  // The config descent IS the validation, and it refuses by name on anything
  // this bring-up cannot represent. W4 owns materializing the tensors
  // EnumerateNemotronHTensors names.
  NemotronHParams params = ParseNemotronHParams(config);
  return std::make_unique<NemotronHLoadedModel>(registration,
                                                std::move(params));
}

void PrepareNemotronHForCausalLM(LoadedModel& model, const HfConfig& config,
                                 vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardNemotronHForCausalLM(LoadedModel& model,
                                          const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // W4 owns the hybrid layer loop, the Mamba2 mixer wiring, the 6 attention
  // layers and the MoE layers (spec §4 W4), and is itself blocked on the
  // Mamba2 SSD CUDA arm (#496 W2). Refuse loudly rather than return zeros.
  VT_CHECK(false,
           "NemotronHForCausalLM forward is not implemented yet (W4 of "
           ".agents/specs/nemotron-h-model.md, issue #517)");
  return {};
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
  //     LAYOUT NOTE: upstream's DEFAULT conv layout is "SD" = (state_len, dim)
  //     (mamba_utils.py:27-48, `VLLM_SSM_CONV_STATE_LAYOUT` unset ⇒ "SD"),
  //     while our local convention across qwen3_5_common.cpp:85 and
  //     kimi_linear_registry.cpp:156 is (dim, state_len). The BYTES are
  //     identical — same product, same page size — and this follows the local
  //     convention so the shared runner/manager code sees one orientation. The
  //     discrepancy is recorded here rather than left for W4 to rediscover.
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
