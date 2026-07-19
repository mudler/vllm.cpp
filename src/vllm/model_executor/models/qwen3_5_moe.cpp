// Qwen3.6 MoE (35B-A3B) architecture registry TU. Self-registers
// "Qwen3_5MoeForConditionalGeneration" via REGISTER_VLLM_MODEL and owns the MoE
// arch-specific entry points (LoadedModel subclass + load/prepare/forward
// wrappers + factory + synthetic Make/Borrow adapters). The heavy MoE forward
// machinery (Qwen3_5Model::/Qwen3_5DecodeGraph::) lives in qwen3_5.cpp over the
// shared DevicePool/matmul/GDN helpers; this TU only wires it into the registry.
// Extracted verbatim (behavior-preserving) from the former model_registry.cpp
// monolith.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, Qwen3_5Model
#include "vllm/model_executor/models/qwen3_5_common.h"  // kQwen3_5Info, helpers
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
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

 private:
  std::optional<Qwen3_5MoeWeights> owned_weights_;
  const Qwen3_5MoeWeights* weights_ = nullptr;
  std::unique_ptr<Qwen3_5DecodeGraph> decode_graph_;
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
  auto& qwen = static_cast<Qwen3_5MoeLoadedModel&>(model);
  Qwen3_5Model::PrepareMarlinResident(qwen.weights(), config, queue);
}

ForwardLogits ForwardQwen3_5Moe(LoadedModel& model,
                                const ModelForwardInput& input) {
  auto& qwen = static_cast<Qwen3_5MoeLoadedModel&>(model);
  const Qwen3_5MoeWeights& weights = qwen.weights();
  const bool fp4_cuda =
      platforms::GetPlatform(input.queue.device.type).is_cuda() &&
      !weights.layers.empty() &&
      !weights.layers.front().moe.expert_gate_fp4.empty();
  constexpr int kMaxDecodeGraphBatch = 64;

  if (input.pure_decode && fp4_cuda &&
      input.num_reqs <= kMaxDecodeGraphBatch) {
    if (!qwen.decode_graph()) {
      qwen.decode_graph() = std::make_unique<Qwen3_5DecodeGraph>(
          weights, input.config, input.queue, input.gdn_state_slots);
    }
    return qwen.decode_graph()->Step(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state);
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

}  // namespace vllm
