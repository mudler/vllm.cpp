// kev System 1 decision model registry (MODEL-KEV).
//
// Self-registers "KevModel" via REGISTER_VLLM_MODEL. kev is a pooling model:
// it forwards the Qwen3.5 dense backbone to post-final-norm hidden states
// (ForwardDenseHidden) and applies a PointerHead readout at inference time
// (Phase 5c). The LoRA adapter is merged into the base weights at CONVERT
// time by scripts/convert-kev.py, so this loader sees a single set of
// safetensors shards: model.safetensors (merged Qwen3.5 BF16) + head.safetensors
// (PointerHead F32). No runtime LoRA merge, no two-directory loading.
//
// Ported from jaredpalmer/kev kev/model.py @ 19dcae9b6e3e1a48200c5825aad9fc200d31e20a.
#include "vllm/model_executor/models/model_registry.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // SafetensorsFile, StTensor
#include "vllm/model_executor/models/kev.h"
#include "vllm/model_executor/models/qwen3_5.h"          // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"   // ParseQwen3_5Config, MakeQwen3_5KVCache, HostLogits
#include "vllm/model_executor/models/qwen3_5_dense.h"     // LoadQwen3_5Dense, ForwardDenseHidden

namespace vllm {
namespace {

// kev is a POOLING model (System 1 decision): no text-generation path.
// is_hybrid mirrors the Qwen3.5 backbone (GDN + full-attention layers).
inline constexpr ModelInfo kKevInfo{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

// Owned model: merged Qwen3.5 dense weights + PointerHead weights + head params.
// The PointerHead readout (KevReadout) runs in the inference pipeline (Phase 5c),
// not in this forward — the factory forward returns hidden states for the
// pooling runner, exactly as the Llama embedding path does.
class KevLoadedModel final : public LoadedModel {
 public:
  KevLoadedModel(const ModelRegistration& registration,
                 Qwen3_5DenseWeights weights,
                 kev::HeadWeights head_weights,
                 kev::HeadParams head_params,
                 float temperature)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        head_weights_(std::move(head_weights)),
        head_params_(head_params),
        temperature_(temperature) {}

  const Qwen3_5DenseWeights& weights() const { return weights_; }
  const kev::HeadWeights& head_weights() const { return head_weights_; }
  const kev::HeadParams& head_params() const { return head_params_; }
  float temperature() const { return temperature_; }

 private:
  Qwen3_5DenseWeights weights_;
  kev::HeadWeights head_weights_;
  kev::HeadParams head_params_;
  float temperature_;
};

// Load PointerHead weights (q.weight, q.bias, k.weight, k.bias) from
// head.safetensors, which co-exists with model.safetensors in the shard list.
// All four tensors are F32, row-major (PyTorch nn.Linear convention).
kev::HeadWeights LoadHeadWeights(const std::vector<SafetensorsFile>& shards) {
  kev::HeadWeights hw;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      const StTensor& t = shard.Get(name);
      if (t.dtype != "F32") {
        throw std::runtime_error(
            "kev: head tensor " + name + " must be F32, got " + t.dtype);
      }
      size_t count = t.nbytes / sizeof(float);
      std::vector<float> dst(count);
      std::memcpy(dst.data(), t.data, t.nbytes);
      if (name == "q.weight") {
        hw.q_weight = std::move(dst);
      } else if (name == "q.bias") {
        hw.q_bias = std::move(dst);
      } else if (name == "k.weight") {
        hw.k_weight = std::move(dst);
      } else if (name == "k.bias") {
        hw.k_bias = std::move(dst);
      }
    }
  }
  if (hw.q_weight.empty() || hw.q_bias.empty() ||
      hw.k_weight.empty() || hw.k_bias.empty()) {
    throw std::runtime_error(
        "kev: head.safetensors not found or incomplete "
        "(q.weight/q.bias/k.weight/k.bias required). "
        "Run scripts/convert-kev.py first.");
  }
  return hw;
}

std::unique_ptr<LoadedModel> LoadKev(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture KevModel does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }

  // Load merged Qwen3.5 dense weights. The LoRA adapter was merged at
  // convert time (convert-kev.py), so the safetensors shards carry the
  // final merged weights with the original Qwen3.5 tensor-name layout.
  Qwen3_5DenseWeights weights =
      LoadQwen3_5Dense(*source.safetensors, config, source.load_queue);

  // Load PointerHead weights from head.safetensors in the same shard list.
  kev::HeadWeights head_weights = LoadHeadWeights(*source.safetensors);

  // Read head params from config (written by convert-kev.py).
  kev::HeadParams head_params;
  head_params.hidden_size = config.hidden_size;
  if (config.raw.contains("kev_head_dim")) {
    head_params.head_dim =
        config.raw["kev_head_dim"].get<int64_t>();
  }
  float temperature = 1.0f;
  if (config.raw.contains("kev_temperature")) {
    temperature = config.raw["kev_temperature"].get<float>();
  }

  return std::make_unique<KevLoadedModel>(
      registration, std::move(weights), std::move(head_weights),
      head_params, temperature);
}

void PrepareKev(LoadedModel& model, const HfConfig& config,
                vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

// The factory forward runs the Qwen3.5 dense backbone to post-final-norm
// hidden states (ForwardDenseHidden) and wraps them as ForwardLogits with
// vocab=hidden_size. The pooling runner consumes these; the PointerHead
// readout (KevReadout) runs in the inference pipeline (Phase 5c), not here.
ForwardLogits ForwardKev(LoadedModel& model, const ModelForwardInput& input) {
  auto& kev = ModelAs<KevLoadedModel>(model, "KevModel");
  std::vector<float> hidden = Qwen3_5DenseModel::ForwardDenseHidden(
      input.token_ids, input.positions, kev.weights(), input.config,
      input.queue);
  return HostLogits(std::move(hidden), input.config.hidden_size);
}

const ModelFactory kKevFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadKev,
    .prepare = &PrepareKev,
    .forward = &ForwardKev,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = true,
};

}  // namespace

REGISTER_VLLM_MODEL(kev_model, "KevModel", kKevFactory, kKevInfo)

}  // namespace vllm
