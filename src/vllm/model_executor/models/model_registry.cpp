// Ported from: vllm/model_executor/models/registry.py
//               @ e24d1b24fe96a56ba8b0d653efa076d03eb95d6c
// (_ModelInfo:746-796, _ModelRegistry:998-1082,
//  resolve_model_cls:1244-1296, global registry:1396-1404).
#include "vllm/model_executor/models/model_registry.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py:701-735, ported verbatim and declaration-ordered.
constexpr std::array<UnsupportedModelInfo, 32> kPreviouslySupportedModels{{
    {"MotifForCausalLM", "0.10.2"},
    {"Phi3SmallForCausalLM", "0.9.2"},
    {"Phi4FlashForCausalLM", "0.10.2"},
    {"Phi4MultimodalForCausalLM", "0.12.0"},
    {"JAISLMHeadModel", "0.22.0"},
    {"ErnieModel", "0.23.0"},
    {"ErnieForSequenceClassification", "0.23.0"},
    {"ErnieForTokenClassification", "0.23.0"},
    {"InternLM2VEForCausalLM", "0.23.0"},
    {"QWenLMHeadModel", "0.23.0"},
    {"QwenVLForConditionalGeneration", "0.23.0"},
    {"InternLMForCausalLM", "0.23.0"},
    {"DonutForConditionalGeneration", "0.10.2"},
    {"MllamaForConditionalGeneration", "0.10.2"},
    {"XverseForCausalLM", "0.23.0"},
    {"Dots1ForCausalLM", "0.23.0"},
    {"BambaForCausalLM", "0.23.0"},
    {"MiniMaxForCausalLM", "0.23.0"},
    {"MiniMaxText01ForCausalLM", "0.23.0"},
    {"MiniMaxM1ForCausalLM", "0.23.0"},
    {"MiniMaxVL01ForConditionalGeneration", "0.23.0"},
    {"BaiChuanForCausalLM", "0.23.0"},
    {"BaichuanForCausalLM", "0.23.0"},
    {"AquilaModel", "0.24.0"},
    {"AquilaForCausalLM", "0.24.0"},
    {"Grok1ModelForCausalLM", "0.24.0"},
    {"Grok1ForCausalLM", "0.24.0"},
    {"TarsierForConditionalGeneration", "0.24.0"},
    {"Tarsier2ForConditionalGeneration", "0.23.0"},
    {"MantisForConditionalGeneration", "0.24.0"},
    {"MusicFlamingoForConditionalGeneration", "0.24.0"},
    {"AyaVisionForConditionalGeneration", "0.24.0"},
}};

// registry.py:737-743, ported verbatim and declaration-ordered.
constexpr std::array<UnsupportedModelInfo, 4> kOutOfTreeSupportedModels{{
    {"BartModel", "https://github.com/vllm-project/bart-plugin"},
    {"BartForConditionalGeneration",
     "https://github.com/vllm-project/bart-plugin"},
    {"Florence2ForConditionalGeneration",
     "https://github.com/vllm-project/bart-plugin"},
    {"MBartForConditionalGeneration",
     "https://github.com/vllm-project/bart-plugin"},
}};

template <typename EntryRange>
const UnsupportedModelInfo* FindUnsupported(const EntryRange& entries,
                                            std::string_view architecture) {
  const auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const UnsupportedModelInfo& entry) {
                                 return entry.architecture == architecture;
                               });
  return it == entries.end() ? nullptr : &*it;
}

std::string PythonStringList(std::span<const std::string> values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << '\'' << values[i] << '\'';
  }
  out << ']';
  return out.str();
}

std::string PythonDictKeys(std::span<const std::string_view> values) {
  std::ostringstream out;
  out << "dict_keys([";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << '\'' << values[i] << '\'';
  }
  out << "])";
  return out.str();
}

ForwardLogits HostLogits(std::vector<float>&& host, int64_t vocab) {
  ForwardLogits logits;
  logits.vocab = vocab;
  logits.rows = vocab > 0 ? static_cast<int64_t>(host.size()) / vocab : 0;
  logits.host = std::move(host);
  return logits;
}

bool DenseDecodeGraphEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_CPP_DENSE_DECODE_GRAPH");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

bool HostWeightReleaseEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_RELEASE_HOST_WEIGHTS");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

struct BorrowedWeightsTag {};

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
  bool owns_weights() const { return owned_weights_.has_value(); }
  Qwen3_5DenseWeights& mutable_weights() { return *owned_weights_; }
  bool host_weights_released() const { return host_weights_released_; }
  void mark_host_weights_released() { host_weights_released_ = true; }
  std::unique_ptr<Qwen3_5DenseDecodeGraph>& decode_graph() {
    return decode_graph_;
  }

 private:
  std::optional<Qwen3_5DenseWeights> owned_weights_;
  const Qwen3_5DenseWeights* weights_ = nullptr;
  std::unique_ptr<Qwen3_5DenseDecodeGraph> decode_graph_;
  bool host_weights_released_ = false;
};

void ParseQwen3_5Config(const HfConfig& config) {
  // LoadHfConfig/HfConfigFromGguf already materialize the consumed Qwen fields.
  // This explicit per-family hook is where a family adds normalization or
  // validation without changing the registry/runner contract.
  (void)config;
}

std::unique_ptr<LoadedModel> LoadQwen3_5MoeModel(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kSafetensors) {
    if (source.safetensors == nullptr) {
      throw std::runtime_error("safetensors model source is empty");
    }
    return std::make_unique<Qwen3_5MoeLoadedModel>(
        registration, LoadQwen3_5Moe(*source.safetensors, config));
  }
  if (source.gguf == nullptr) {
    throw std::runtime_error("GGUF model source is empty");
  }
  return std::make_unique<Qwen3_5MoeLoadedModel>(
      registration, LoadQwen3_5MoeFromGguf(*source.gguf, config));
}

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

void PrepareQwen3_5Moe(LoadedModel& model, const HfConfig& config,
                       vt::Queue& queue) {
  auto& qwen = static_cast<Qwen3_5MoeLoadedModel&>(model);
  Qwen3_5Model::PrepareMarlinResident(qwen.weights(), config, queue);
}

void PrepareQwen3_5Dense(LoadedModel& model, const HfConfig& config,
                         vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardQwen3_5Moe(LoadedModel& model,
                                const ModelForwardInput& input) {
  auto& qwen = static_cast<Qwen3_5MoeLoadedModel&>(model);
  const Qwen3_5MoeWeights& weights = qwen.weights();
  const bool fp4_cuda =
      input.queue.device.type == vt::DeviceType::kCUDA &&
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

ForwardLogits ForwardQwen3_5Dense(LoadedModel& model,
                                  const ModelForwardInput& input) {
  auto& qwen = static_cast<Qwen3_5DenseLoadedModel&>(model);
  const Qwen3_5DenseWeights& weights = qwen.weights();
  const bool fp4_cuda =
      input.queue.device.type == vt::DeviceType::kCUDA &&
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

  ForwardLogits output;
  if (input.gather_logits) {
    output = Qwen3_5DenseModel::ForwardDevice(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.logits_indices);
  } else {
    output = HostLogits(
        Qwen3_5DenseModel::Forward(
            input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
            input.attn_kv, input.gdn_state, weights, input.config, input.queue,
            input.logits_indices),
        input.config.vocab_size);
  }

  vt::Backend& backend = vt::GetBackend(input.queue.device.type);
  if (!input.pure_decode && HostWeightReleaseEnabled() && qwen.owns_weights() &&
      !qwen.host_weights_released() &&
      input.queue.device.type == vt::DeviceType::kCUDA &&
      !backend.UnifiedMemory() && IsPlainBf16Qwen3_5Dense(weights)) {
    backend.Synchronize(input.queue);
    ReleaseResidentQwen3_5DenseHostWeights(qwen.mutable_weights());
    qwen.mark_host_weights_released();
  }
  return output;
}

v1::KVCacheConfig MakeQwen3_5KVCache(const HfConfig& config, int block_size,
                                     int num_blocks) {
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);
  const int num_value_heads = static_cast<int>(config.linear_num_value_heads);
  const int value_head_dim = static_cast<int>(config.linear_value_head_dim);
  const int key_head_dim = static_cast<int>(config.linear_key_head_dim);
  const int conv_kernel = static_cast<int>(config.linear_conv_kernel_dim);
  const int key_dim =
      static_cast<int>(config.linear_num_key_heads) * key_head_dim;
  const int value_dim = num_value_heads * value_head_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, num_kv_heads,
                                               head_dim, vt::DType::kF32));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn"},
      std::make_shared<v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{
              {num_value_heads, value_head_dim, key_head_dim},
              {conv_dim, conv_kernel - 1}},
          std::vector<vt::DType>{vt::DType::kF32, vt::DType::kF32}));
  return kv;
}

constexpr ModelInfo kQwen3_5Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

const ModelFactory kQwen3_5DenseFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadQwen3_5DenseModel,
    .prepare = &PrepareQwen3_5Dense,
    .forward = &ForwardQwen3_5Dense,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = true,
};

const ModelFactory kQwen3_5MoeFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadQwen3_5MoeModel,
    .prepare = &PrepareQwen3_5Moe,
    .forward = &ForwardQwen3_5Moe,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = false,
};

// Mirrors ModelRegistry = _ModelRegistry({... _VLLM_MODELS.items()}) without
// cross-TU static initialization. Only implemented architectures are supported.
#define REGISTER_VLLM_MODEL(architecture_value, factory_value, info_value) \
  ModelRegistration { architecture_value, &factory_value, info_value }

const std::array<ModelRegistration, 2> kRegistrations{{
    REGISTER_VLLM_MODEL("Qwen3_5ForConditionalGeneration",
                        kQwen3_5DenseFactory, kQwen3_5Info),
    REGISTER_VLLM_MODEL("Qwen3_5MoeForConditionalGeneration",
                        kQwen3_5MoeFactory, kQwen3_5Info),
}};

#undef REGISTER_VLLM_MODEL

const ModelRegistration& RegistrationFor(std::string_view architecture) {
  const auto it = std::find_if(
      kRegistrations.begin(), kRegistrations.end(),
      [&](const ModelRegistration& registration) {
        return registration.architecture == architecture;
      });
  if (it == kRegistrations.end()) {
    throw std::logic_error("internal model registration is missing");
  }
  return *it;
}

}  // namespace

LoadedModel::~LoadedModel() = default;

ModelSource ModelSource::FromSafetensors(
    const std::vector<SafetensorsFile>& shards) {
  ModelSource source;
  source.kind = Kind::kSafetensors;
  source.safetensors = &shards;
  return source;
}

ModelSource ModelSource::FromGguf(const GgufFile& gguf) {
  ModelSource source;
  source.kind = Kind::kGguf;
  source.gguf = &gguf;
  return source;
}

std::span<const ModelRegistration> ModelRegistry::Registrations() {
  return kRegistrations;
}

std::vector<std::string_view> ModelRegistry::SupportedArchs() {
  std::vector<std::string_view> supported;
  supported.reserve(kRegistrations.size());
  for (const ModelRegistration& registration : kRegistrations) {
    supported.push_back(registration.architecture);
  }
  return supported;
}

const ModelRegistration& ModelRegistry::Resolve(
    std::span<const std::string> architectures) {
  if (architectures.empty()) {
    throw std::runtime_error("No model architectures are specified");
  }
  for (const std::string& architecture : architectures) {
    const auto it = std::find_if(
        kRegistrations.begin(), kRegistrations.end(),
        [&](const ModelRegistration& registration) {
          return registration.architecture == architecture;
        });
    if (it != kRegistrations.end()) return *it;
  }
  RaiseForUnsupported(architectures);
}

const ModelRegistration& ModelRegistry::Resolve(const HfConfig& config) {
  return Resolve(std::span<const std::string>(config.architectures));
}

void ModelRegistry::RaiseForUnsupported(
    std::span<const std::string> architectures) {
  const std::vector<std::string_view> supported = SupportedArchs();
  RaiseForUnsupported(architectures,
                      std::span<const std::string_view>(supported));
}

void ModelRegistry::RaiseForUnsupported(
    std::span<const std::string> architectures,
    std::span<const std::string_view> supported_architectures) {
  const bool inspection_failed =
      std::any_of(architectures.begin(), architectures.end(),
                  [&](const std::string& architecture) {
                    return std::find(supported_architectures.begin(),
                                     supported_architectures.end(),
                                     architecture) != supported_architectures.end();
                  });
  if (inspection_failed) {
    throw std::runtime_error("Model architectures " +
                             PythonStringList(architectures) +
                             " failed to be inspected. Please check the logs "
                             "for more details.");
  }

  for (const std::string& architecture : architectures) {
    if (const UnsupportedModelInfo* previous =
            FindUnsupported(kPreviouslySupportedModels, architecture)) {
      throw std::runtime_error(
          "Model architecture " + architecture +
          " was supported in vLLM until v" + std::string(previous->detail) +
          ", and is not supported anymore. Please use an older version of "
          "vLLM if you want to use this model architecture.");
    }
    if (const UnsupportedModelInfo* plugin =
            FindUnsupported(kOutOfTreeSupportedModels, architecture)) {
      throw std::runtime_error(
          "Model architecture " + architecture +
          " is not supported in-tree anymore. Please install the plugin at " +
          std::string(plugin->detail) +
          " if you want to use this model architecture.");
    }
  }

  throw std::runtime_error(
      "Model architectures " + PythonStringList(architectures) +
      " are not supported for now. Supported architectures: " +
      PythonDictKeys(supported_architectures));
}

std::span<const UnsupportedModelInfo>
ModelRegistry::PreviouslySupportedModels() {
  return kPreviouslySupportedModels;
}

std::span<const UnsupportedModelInfo>
ModelRegistry::OutOfTreeSupportedModels() {
  return kOutOfTreeSupportedModels;
}

std::unique_ptr<LoadedModel> ModelRegistry::Load(const HfConfig& config,
                                                 const ModelSource& source) {
  const ModelRegistration& registration = Resolve(config);
  const ModelFactory& factory = *registration.factory;
  factory.parse_config(config);
  std::unique_ptr<LoadedModel> model =
      factory.load_weights(registration, config, source);
  if (!model) {
    RaiseForUnsupported(std::span<const std::string>(config.architectures));
  }
  return model;
}

void ModelRegistry::Prepare(LoadedModel& model, const HfConfig& config,
                            vt::Queue& queue) {
  model.registration().factory->prepare(model, config, queue);
}

ForwardLogits ModelRegistry::Forward(LoadedModel& model,
                                     const ModelForwardInput& input) {
  return model.registration().factory->forward(model, input);
}

v1::KVCacheConfig ModelRegistry::MakeKVCache(const LoadedModel& model,
                                              const HfConfig& config,
                                              int block_size, int num_blocks) {
  return model.registration().factory->make_kv_cache(config, block_size,
                                                      num_blocks);
}

bool ModelRegistry::IsDenseModel(const LoadedModel& model) {
  return model.registration().factory->is_dense_model;
}

std::unique_ptr<LoadedModel> MakeQwen3_5MoeLoadedModel(
    Qwen3_5MoeWeights weights) {
  return std::make_unique<Qwen3_5MoeLoadedModel>(
      RegistrationFor("Qwen3_5MoeForConditionalGeneration"),
      std::move(weights));
}

std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights) {
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      RegistrationFor("Qwen3_5ForConditionalGeneration"), std::move(weights));
}

std::unique_ptr<LoadedModel> BorrowQwen3_5MoeLoadedModel(
    const Qwen3_5MoeWeights& weights) {
  return std::make_unique<Qwen3_5MoeLoadedModel>(
      RegistrationFor("Qwen3_5MoeForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

std::unique_ptr<LoadedModel> BorrowQwen3_5DenseLoadedModel(
    const Qwen3_5DenseWeights& weights) {
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      RegistrationFor("Qwen3_5ForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

}  // namespace vllm
