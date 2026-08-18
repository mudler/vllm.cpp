// Ported from: vllm/model_executor/models/registry.py
//               @ e24d1b24fe96a56ba8b0d653efa076d03eb95d6c
// (_ModelInfo:746-796, _ModelRegistry:998-1082,
//  resolve_model_cls:1244-1296, global registry:1396-1404).
//
// This TU is the GENERIC, family-agnostic registry: the ordered lookup,
// capability metadata, unsupported-architecture messages, and the type-erased
// Load/Prepare/Forward/MakeKVCache dispatch. Each architecture's factory + entry
// points live in its OWN TU (e.g. qwen3_5_dense.cpp, qwen3_5_moe.cpp) and
// register themselves here via REGISTER_VLLM_MODEL — mirroring how
// `_VLLM_MODELS` is assembled from per-model registrations (registry.py:682-693)
// rather than a fixed in-file array.
#include "vllm/model_executor/models/model_registry.h"

#include "vllm/model_executor/layers/quantization/fp8_block_quant.h"
#include "vllm/model_executor/weight_offloader.h"

#include <algorithm>
#include <array>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ForwardLogits (the type-erased forward-result carrier) is defined here; its
// complete definition is required for ModelRegistry::Forward's by-value return.
#include "vllm/model_executor/models/qwen3_5.h"
// SPEC-MTP I5d-pre: Qwen3_5MTPWeights / Qwen3_5MTPModel complete types for the
// LoadedModel base defaults of AttachMtpDraftWeights / BuildMtpDraft (the
// non-MTP behavior — throw / return null).
#include "vllm/model_executor/models/qwen3_5_mtp.h"

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

// Process-global registry, populated at static-init by each architecture's
// REGISTER_VLLM_MODEL Registrar (see model_registry.h). Meyers singleton: the
// vector is constructed on the first RegisterModel call, safely before any
// registrar runs.
std::vector<ModelRegistration>& RegistryStorage() {
  static std::vector<ModelRegistration> storage;
  return storage;
}

// Returns the registry with a stable, once-applied canonical order. C++ does not
// order static init across TUs, so registration arrival order is unspecified; we
// sort by architecture name once (on first query, after all static init) to make
// SupportedArchs()/error-message presentation deterministic. Resolution picks the
// first CONFIG-architecture match, which is order-independent, so this sort never
// changes which model resolves — only the cosmetic supported-list order.
const std::vector<ModelRegistration>& OrderedRegistry() {
  [[maybe_unused]] static const bool sorted = [] {
    std::vector<ModelRegistration>& storage = RegistryStorage();
    std::stable_sort(storage.begin(), storage.end(),
                     [](const ModelRegistration& a, const ModelRegistration& b) {
                       return a.architecture < b.architecture;
                     });
    return true;
  }();
  return RegistryStorage();
}

}  // namespace

void RegisterModel(const ModelRegistration& registration) {
  RegistryStorage().push_back(registration);
}

const ModelRegistration& RegistrationFor(std::string_view architecture) {
  const std::vector<ModelRegistration>& registry = OrderedRegistry();
  const auto it = std::find_if(
      registry.begin(), registry.end(),
      [&](const ModelRegistration& registration) {
        return registration.architecture == architecture;
      });
  if (it == registry.end()) {
    throw std::logic_error("internal model registration is missing");
  }
  return *it;
}

LoadedModel::~LoadedModel() = default;

// #775: the refusal behind `ModelAs`. Out-of-line so the header template stays
// a `dynamic_cast` and a branch, and so the message is authored in ONE place
// rather than once per registered architecture.
//
// It names the entry point that refused and the architecture the passed model's
// own registration claims. Those two are usually the SAME string, and that is
// the informative case rather than a redundant one: the realistic defect is a
// caller that resolved a registration and then handed `factory->forward` a
// model some other path produced, so the registration is right and the object
// is not.
void RaiseModelTypeMismatch(std::string_view architecture,
                            const LoadedModel& model) {
  throw std::runtime_error(
      std::string(architecture) +
      ": the LoadedModel handed to this registry entry point was not produced "
      "by " +
      std::string(architecture) +
      "'s own load_weights (the model's registration names architecture '" +
      std::string(model.registration().architecture) +
      "'). Refusing by name rather than downcasting a foreign model, which is "
      "undefined behaviour on every member call that follows (issue #775).");
}

// SPEC-MTP I5d-pre base defaults: a non-MTP architecture cannot retain draft
// weights and builds no draft. The concrete Qwen3.5 dense/MoE models override
// both (qwen3_5_dense.cpp / qwen3_5_moe.cpp).
void LoadedModel::AttachMtpDraftWeights(Qwen3_5MTPWeights /*weights*/) {
  throw std::runtime_error(
      "model does not support MTP draft weights (no speculative-decoding draft "
      "head for this architecture)");
}

std::unique_ptr<Qwen3_5MTPModel> LoadedModel::BuildMtpDraft(
    const HfConfig& /*config*/) const {
  return nullptr;
}

ModelSource ModelSource::FromSafetensors(
    const std::vector<SafetensorsFile>& shards, vt::Queue* load_queue) {
  ModelSource source;
  source.kind = Kind::kSafetensors;
  source.safetensors = &shards;
  source.load_queue = load_queue;
  return source;
}

ModelSource ModelSource::FromSafetensorsOwned(
    std::shared_ptr<const std::vector<SafetensorsFile>> shards,
    vt::Queue* load_queue) {
  ModelSource source;
  source.kind = Kind::kSafetensors;
  source.safetensors = shards.get();
  source.safetensors_owned = std::move(shards);
  source.load_queue = load_queue;
  return source;
}

ModelSource ModelSource::FromGguf(const GgufFile& gguf) {
  ModelSource source;
  source.kind = Kind::kGguf;
  source.gguf = &gguf;
  return source;
}

std::span<const ModelRegistration> ModelRegistry::Registrations() {
  return OrderedRegistry();
}

std::vector<std::string_view> ModelRegistry::SupportedArchs() {
  const std::vector<ModelRegistration>& registry = OrderedRegistry();
  std::vector<std::string_view> supported;
  supported.reserve(registry.size());
  for (const ModelRegistration& registration : registry) {
    supported.push_back(registration.architecture);
  }
  return supported;
}

const ModelRegistration& ModelRegistry::Resolve(
    std::span<const std::string> architectures) {
  if (architectures.empty()) {
    throw std::runtime_error("No model architectures are specified");
  }
  const std::vector<ModelRegistration>& registry = OrderedRegistry();
  for (const std::string& architecture : architectures) {
    const auto it = std::find_if(
        registry.begin(), registry.end(),
        [&](const ModelRegistration& registration) {
          return registration.architecture == architecture;
        });
    if (it != registry.end()) return *it;
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
  // FIX-FP8-BLOCKWISE-REFUSAL (#1166): a block-wise (fine-grained) FP8
  // checkpoint is refused BY NAME here, before any weight loader runs.
  //
  // AFTER `Resolve`, so an unsupported architecture still reports the
  // architecture rather than its quantization. BEFORE `load_weights`, because
  // that is what makes the message about the missing ARM instead of about the
  // first tensor whose name does not resolve: the dense loader branches on the
  // weight dtype alone and pulls a block-wise projection into the per-tensor
  // arm, which then asks for a `weight_scale` a block-wise checkpoint spells
  // `weight_scale_inv` and dies on `tensor not found`.
  //
  // Sited on the registry rather than per model loader on purpose.
  // `weight_block_size` is a property of the checkpoint's quantization config
  // and not of one architecture, so a per-loader refusal would have to be
  // written again for every architecture and would be missing from whichever
  // one is added next. This also covers the GGUF arm, which reaches this
  // function too and where the key is simply absent.
  RefuseUnsupportedFp8BlockQuant(config);
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
  // ENG-WEIGHT-OFFLOAD: the post-load hook, upstream's `post_init`
  // (offloader/base.py:68-76). This is NOT where a weight is offloaded. The
  // offload decision is asked during LOADING, through
  // `WeightOffloader::ConsiderWeight`, because a weight that reached here has
  // already paid the device allocation the feature exists to avoid. The default
  // instance is the no-op, so this line is inert until an engine installs a
  // backend.
  GetWeightOffloader().OnModelPrepared(model);
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

}  // namespace vllm
