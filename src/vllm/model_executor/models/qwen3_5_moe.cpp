// Qwen3.6 MoE (35B-A3B) architecture registry TU. Self-registers
// "Qwen3_5MoeForConditionalGeneration" via REGISTER_VLLM_MODEL and owns the MoE
// arch-specific entry points (LoadedModel subclass + load/prepare/forward
// wrappers + factory + synthetic Make/Borrow adapters). The heavy MoE forward
// machinery (Qwen3_5Model::/Qwen3_5DecodeGraph::) lives in qwen3_5.cpp over the
// shared DevicePool/matmul/GDN helpers; this TU only wires it into the registry.
// Extracted verbatim (behavior-preserving) from the former model_registry.cpp
// monolith.
#include <cstdlib>

#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, Qwen3_5Model
#include "vllm/model_executor/models/qwen3_5_common.h"  // kQwen3_5Info, helpers
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "qwen3_5_internal.h"  // W4 DeviceTokenIdsScope
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d-pre draft
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/platforms/interface.h"  // GetPlatform(device.type) memory-model seam

namespace vllm {
namespace {

class Qwen3_5MoeLoadedModel final : public LoadedModel {
 public:
  Qwen3_5MoeLoadedModel(const ModelRegistration& registration,
                        Qwen3_5MoeWeights weights)
      : LoadedModel(registration),
        owned_weights_(std::move(weights)),
        weights_(&*owned_weights_) {}
  Qwen3_5MoeLoadedModel(const ModelRegistration& registration,
                        const Qwen3_5MoeWeights& weights, BorrowedWeightsTag)
      : LoadedModel(registration), weights_(&weights) {}

  const Qwen3_5MoeWeights& weights() const { return *weights_; }
  std::unique_ptr<Qwen3_5DecodeGraph>& decode_graph() { return decode_graph_; }

  // SPEC-MTP I5d-pre: retain the loaded `mtp.*` draft weights + build the draft
  // (the 35B MoE MTP layer) sharing this target's embed_tokens/lm_head. Inert
  // unless FromModelDir attached weights (i.e. unless a SpeculativeConfig is set).
  bool supports_mtp_draft() const override { return true; }
  // SPEC-DFLASH / SPEC-DSPARK: this forward routes to ForwardDeviceMultiTap.
  bool supports_aux_multi_tap() const override { return true; }
  void AttachMtpDraftWeights(Qwen3_5MTPWeights weights) override {
    mtp_draft_weights_ = std::move(weights);
  }
  std::unique_ptr<Qwen3_5MTPModel> BuildMtpDraft(
      const HfConfig& config) const override {
    if (!mtp_draft_weights_.has_value()) return nullptr;
    return std::make_unique<Qwen3_5MTPModel>(*mtp_draft_weights_, *weights_,
                                             config);
  }

 private:
  std::optional<Qwen3_5MoeWeights> owned_weights_;
  const Qwen3_5MoeWeights* weights_ = nullptr;
  std::unique_ptr<Qwen3_5DecodeGraph> decode_graph_;
  // Retained draft weights (SPEC-MTP I5d-pre); empty on the production default.
  std::optional<Qwen3_5MTPWeights> mtp_draft_weights_;
};

std::unique_ptr<LoadedModel> LoadQwen3_5MoeModel(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kSafetensors) {
    if (source.safetensors == nullptr) {
      throw std::runtime_error("safetensors model source is empty");
    }
    // Pass the shared shards owner (when the caller shared it, e.g. disk load):
    // it enables the deferred per-layer routed-expert streaming that bounds the
    // 35B load-phase peak host residency. Null → experts loaded eagerly.
    return std::make_unique<Qwen3_5MoeLoadedModel>(
        registration, LoadQwen3_5Moe(*source.safetensors, config,
                                     source.safetensors_owned));
  }
  if (source.gguf == nullptr) {
    throw std::runtime_error("GGUF model source is empty");
  }
  return std::make_unique<Qwen3_5MoeLoadedModel>(
      registration, LoadQwen3_5MoeFromGguf(*source.gguf, config));
}

void PrepareQwen3_5Moe(LoadedModel& model, const HfConfig& config,
                       vt::Queue& queue) {
  auto& qwen = ModelAs<Qwen3_5MoeLoadedModel>(model, "Qwen3_5MoeForConditionalGeneration");
  Qwen3_5Model::PrepareMarlinResident(qwen.weights(), config, queue);
}

ForwardLogits ForwardQwen3_5Moe(LoadedModel& model,
                                const ModelForwardInput& input) {
  auto& qwen = ModelAs<Qwen3_5MoeLoadedModel>(model, "Qwen3_5MoeForConditionalGeneration");
  const Qwen3_5MoeWeights& weights = qwen.weights();

  // ENG-ASYNC-SCHED W4 (see qwen3_5_dense.cpp): scope the async runner's
  // device-resident input ids to this forward so the embed reads them.
  const detail::DeviceTokenIdsScope device_ids_scope(
      input.device_token_ids, static_cast<int64_t>(input.token_ids.size()));

  // SPEC-MTP I5d-pre hidden-state tap (see qwen3_5_dense.cpp). Non-null routes to
  // the EXISTING ForwardDeviceTap (byte-identical logits + the [T,H] post-norm
  // hidden); null (every spec-off run) is byte-identical to the path below.
  if (input.hidden_tap != nullptr) {
    return Qwen3_5Model::ForwardDeviceTap(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.hidden_tap, input.logits_indices);
  }

  // SPEC-DFLASH D1 (DF-AUX-TAPS): non-null routes to ForwardDeviceMultiTap
  // (byte-identical logits + the [T,H×taps] aux capture); null is byte-identical to
  // the path below. Mutually exclusive with hidden_tap.

  const bool fp4_cuda =
      platforms::GetPlatform(input.queue.device.type).cutlass_fp4_supported() &&
      !weights.layers.empty() &&
      !weights.layers.front().moe.expert_gate_fp4.empty();
  constexpr int kMaxDecodeGraphBatch = 64;

  // SPEC-DSPARK W8 (#442): mirror vLLM's UNIFORM-decode predicate instead of
  // "query_len == 1". Upstream's captured decode length is
  // `1 + num_speculative_tokens` (cudagraph_dispatcher.py:37), so its T=1+k
  // speculative VERIFY is graph-captured; ours fell to the eager path EVERY step,
  // which the paired 35B measurement charged at ~4.8 ms/step (0.870x where
  // acceptance is high). With num_speculative_tokens == 0 this is exactly
  // `pure_decode`, so the non-spec path is unchanged.
  // DEFAULT ON. The staging defect that forced this off is fixed: the captured
  // graph reads gdn_spec_* / num_accepted, which StageSpecStepInputs now refills
  // per step, and the per-request arrays are sized by the REQUEST count rather
  // than the token count. Gated green with capture ON and OFF across the MTP,
  // DFlash, 35B and concurrent e2e suites, and measured +8.5% / +4.7% on the 35B
  // cells (0.870x -> 0.986x of the pinned graphed oracle on the high-acceptance
  // one). VT_SPEC_DECODE_GRAPH=0 restores the eager verify for an A/B.
  static const bool spec_graph = [] {
    const char* v = std::getenv("VT_SPEC_DECODE_GRAPH");
    return v == nullptr || (v[0] != '\0' && v[0] != '0');
  }();
  const bool uniform_decode =
      input.pure_decode ||
      (spec_graph && input.num_speculative_tokens > 0 && input.gdn_meta.num_prefill_tokens == 0 &&
       v1::IsUniformDecodeBatch(input.num_reqs, input.attn_meta.num_actual_tokens,
                                input.attn_meta.max_query_len,
                                input.num_speculative_tokens) &&
       // The captured region emits EVERY row, so only a no-op gather may take it.
       // A spec verify needs all 1+k rows anyway, so this holds there.
       (input.logits_indices.empty() ||
        static_cast<int64_t>(input.logits_indices.size()) ==
            input.attn_meta.num_actual_tokens));
  if (uniform_decode && fp4_cuda &&
      input.num_reqs <= kMaxDecodeGraphBatch) {
    if (!qwen.decode_graph()) {
      qwen.decode_graph() = std::make_unique<Qwen3_5DecodeGraph>(
          weights, input.config, input.queue, input.gdn_state_slots);
    }
    return qwen.decode_graph()->Step(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, input.aux_tap);
  }

  // SPEC-DSPARK W8 (#442): the aux multi-tap forward is the DFlash/DSpark
  // VERIFY path, and it used to return BEFORE the decode-graph gate, which is
  // why the T=1+k verify never captured. The graph branch above now serves it
  // (writing the taps into the slot's persistent buffer); this remains the
  // eager fallback for every batch the graph declines.
  if (input.aux_tap != nullptr) {
    return Qwen3_5Model::ForwardDeviceMultiTap(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.aux_tap, input.logits_indices);
  }
  if (input.gather_logits) {
    return Qwen3_5Model::ForwardDevice(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.logits_indices);
  }
  return HostLogits(
      Qwen3_5Model::Forward(
          input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
          input.attn_kv, input.gdn_state, weights, input.config, input.queue,
          input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kQwen3_5MoeFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadQwen3_5MoeModel,
    .prepare = &PrepareQwen3_5Moe,
    .forward = &ForwardQwen3_5Moe,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = false,
};

}  // namespace

std::unique_ptr<LoadedModel> MakeQwen3_5MoeLoadedModel(
    Qwen3_5MoeWeights weights) {
  return std::make_unique<Qwen3_5MoeLoadedModel>(
      RegistrationFor("Qwen3_5MoeForConditionalGeneration"),
      std::move(weights));
}

std::unique_ptr<LoadedModel> BorrowQwen3_5MoeLoadedModel(
    const Qwen3_5MoeWeights& weights) {
  return std::make_unique<Qwen3_5MoeLoadedModel>(
      RegistrationFor("Qwen3_5MoeForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

REGISTER_VLLM_MODEL(qwen3_5_moe, "Qwen3_5MoeForConditionalGeneration",
                    kQwen3_5MoeFactory, kQwen3_5Info)

// TEXT-ONLY arm of the SAME backbone. Upstream registers it against the same
// `qwen3_5` module (registry.py:202-203 @ `ad5d29db7`, PR #50210) and its class
// is `Qwen3_5ForCausalLMBase` plus `set_moe_parameters()` — not a separate model
// (qwen3_5.py:443-449). So this is the SAME factory, additively: no forward, no
// KV-cache spec and no loader fork. `Qwen/Qwen3.8-2.4T-A95B` is the motivating
// checkpoint; it declares `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text` and is
// the 35B-A3B architecture at larger scale, all of it config-driven.
//
// AHEAD OF THE PIN, DELIBERATELY. `555967922` (.agents/upstream-sync.md) carries
// only the ForConditionalGeneration entries; the text-only arms landed upstream
// after it. This is a forward port of ONE upstream PR and does not advance the
// pin. There is NO run gate for the 2.4T checkpoint on this hardware — see
// .agents/specs/qwen38-text-only.md §Gates, which records that gate as OWED.
REGISTER_VLLM_MODEL(qwen3_5_moe_text, "Qwen3_5MoeForCausalLM",
                    kQwen3_5MoeFactory, kQwen3_5TextInfo)

}  // namespace vllm
