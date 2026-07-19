// Qwen3.6 DENSE (27B) architecture registry TU. Self-registers
// "Qwen3_5ForConditionalGeneration" via REGISTER_VLLM_MODEL and owns the dense
// arch-specific entry points (LoadedModel subclass + load/prepare/forward
// wrappers + factory + synthetic Make/Borrow adapters). The heavy dense forward
// machinery (Qwen3_5DenseModel::/Qwen3_5DenseDecodeGraph::) lives in qwen3_5.cpp
// over the shared DevicePool/matmul/GDN helpers; this TU only wires it into the
// registry. Extracted verbatim (behavior-preserving) from the former
// model_registry.cpp monolith.
#include "vllm/model_executor/models/model_registry.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"  // kQwen3_5Info, helpers
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/platforms/interface.h"  // GetPlatform(device.type) memory-model seam

namespace vllm {
namespace {

bool DenseDecodeGraphEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_CPP_DENSE_DECODE_GRAPH");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

class Qwen3_5DenseLoadedModel final : public LoadedModel {
 public:
  Qwen3_5DenseLoadedModel(const ModelRegistration& registration,
                          Qwen3_5DenseWeights weights)
      : LoadedModel(registration),
        owned_weights_(std::move(weights)),
        weights_(&*owned_weights_) {}
  Qwen3_5DenseLoadedModel(const ModelRegistration& registration,
                          const Qwen3_5DenseWeights& weights,
                          BorrowedWeightsTag)
      : LoadedModel(registration), weights_(&weights) {}

  const Qwen3_5DenseWeights& weights() const { return *weights_; }
  bool uses_nvfp4_w4a4() const override {
    return !weights_->layers.empty() &&
           weights_->layers.front().mlp.gate_proj_fp4.IsTrueW4A4();
  }
  std::unique_ptr<Qwen3_5DenseDecodeGraph>& decode_graph() {
    return decode_graph_;
  }

 private:
  std::optional<Qwen3_5DenseWeights> owned_weights_;
  const Qwen3_5DenseWeights* weights_ = nullptr;
  std::unique_ptr<Qwen3_5DenseDecodeGraph> decode_graph_;
};

std::unique_ptr<LoadedModel> LoadQwen3_5DenseModel(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Qwen3_5ForConditionalGeneration does not support "
        "GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      registration, LoadQwen3_5Dense(*source.safetensors, config));
}

void PrepareQwen3_5Dense(LoadedModel& model, const HfConfig& config,
                         vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardQwen3_5Dense(LoadedModel& model,
                                  const ModelForwardInput& input) {
  auto& qwen = static_cast<Qwen3_5DenseLoadedModel&>(model);
  const Qwen3_5DenseWeights& weights = qwen.weights();
  const bool fp4_cuda =
      platforms::GetPlatform(input.queue.device.type).is_cuda() &&
      !weights.layers.empty() &&
      !weights.layers.front().mlp.gate_proj_fp4.Empty();
  constexpr int kMaxDecodeGraphBatch = 64;

  if (DenseDecodeGraphEnabled() && input.pure_decode && fp4_cuda &&
      input.num_reqs <= kMaxDecodeGraphBatch) {
    if (!qwen.decode_graph()) {
      qwen.decode_graph() = std::make_unique<Qwen3_5DenseDecodeGraph>(
          weights, input.config, input.queue, input.gdn_state_slots);
    }
    return qwen.decode_graph()->Step(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state);
  }

  if (input.gather_logits) {
    return Qwen3_5DenseModel::ForwardDevice(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.logits_indices);
  }
  return HostLogits(
      Qwen3_5DenseModel::Forward(
          input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
          input.attn_kv, input.gdn_state, weights, input.config, input.queue,
          input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kQwen3_5DenseFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadQwen3_5DenseModel,
    .prepare = &PrepareQwen3_5Dense,
    .forward = &ForwardQwen3_5Dense,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = true,
};

}  // namespace

std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights) {
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      RegistrationFor("Qwen3_5ForConditionalGeneration"), std::move(weights));
}

std::unique_ptr<LoadedModel> BorrowQwen3_5DenseLoadedModel(
    const Qwen3_5DenseWeights& weights) {
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      RegistrationFor("Qwen3_5ForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

REGISTER_VLLM_MODEL(qwen3_5_dense, "Qwen3_5ForConditionalGeneration",
                    kQwen3_5DenseFactory, kQwen3_5Info)

}  // namespace vllm
