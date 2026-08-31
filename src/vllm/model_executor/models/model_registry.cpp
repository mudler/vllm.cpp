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

#include "vt/dtype.h"  // VT_CHECK

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
  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3), narrowing FIX-FP8-BLOCKWISE-REFUSAL
  // (#1166): the block-wise (fine-grained) FP8 config is READ and VALIDATED
  // here, before any weight loader runs, and only what this build cannot
  // execute is refused BY NAME -- a `quant_method` that is not fp8, a
  // `weight_block_size` that is not exactly two dimensions, an
  // `activation_scheme` other than `dynamic`, and a block shape other than
  // 128x128. A supported `[128, 128]` `dynamic` checkpoint now passes and its
  // projections load through the `weight_scale_inv` rung in
  // `qwen3_5_dense_weights.cpp`. The dense forward READS the resulting weight
  // since #1189 M4 (`281b4bc76`). What `ModelRegistry::Prepare` still refuses by
  // name one step later is a device with no block-scaled GEMM, not the weight.
  //
  // AFTER `Resolve`, so an unsupported architecture still reports the
  // architecture rather than its quantization. BEFORE `load_weights`, because
  // that is what makes the message about the unsupported SHAPE instead of about
  // the first tensor whose name does not resolve: the dense loader branches on
  // the weight dtype alone and a block-wise weight really is `F8_E4M3`, so it
  // used to enter the per-tensor arm, ask for a `weight_scale` the checkpoint
  // spells `weight_scale_inv`, and die on `tensor not found`.
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

// KV-DSV4-MULTICACHE W3 (#2068). Upstream's own key: a cache is addressed by the
// prefix its `AttentionLayerBase` registered under
// (`vllm/v1/worker/gpu_model_runner.py:7785-7801`).
size_t MultiKvCacheIndex::size() const {
  return layer_names == nullptr ? 0U : layer_names->size();
}

int MultiKvCacheIndex::num_groups() const {
  if (group_ids == nullptr) return 0;
  std::vector<int32_t> seen;
  for (int32_t g : *group_ids) {
    if (std::find(seen.begin(), seen.end(), g) == seen.end()) seen.push_back(g);
  }
  return static_cast<int>(seen.size());
}

std::string_view MultiKvCacheIndex::first_name() const {
  if (layer_names == nullptr || layer_names->empty()) return {};
  return layer_names->front();
}

// MODEL-MM-QWEN4-EXP W5c-2 (#2249 item 3). A group carries a table only when the
// runner gathered one for it THIS step; an empty row is a group whose table was
// never gathered, which is the state this wave exists to remove.
int MultiKvCacheIndex::num_published_groups() const {
  return group_block_tables == nullptr
             ? 0
             : static_cast<int>(group_block_tables->size());
}

int MultiKvCacheIndex::num_group_block_tables() const {
  if (group_block_tables == nullptr) return 0;
  int n = 0;
  for (const std::vector<int32_t>& bt : *group_block_tables)
    if (!bt.empty()) ++n;
  return n;
}

const std::vector<int32_t>* MultiKvCacheIndex::BlockTableForGroup(
    int group_id, int* num_cols) const {
  if (num_cols != nullptr) *num_cols = 0;
  if (group_block_tables == nullptr || group_block_table_cols == nullptr)
    return nullptr;
  if (group_id < 0 ||
      static_cast<size_t>(group_id) >= group_block_tables->size() ||
      static_cast<size_t>(group_id) >= group_block_table_cols->size())
    return nullptr;
  const std::vector<int32_t>& bt =
      (*group_block_tables)[static_cast<size_t>(group_id)];
  if (bt.empty()) return nullptr;
  if (num_cols != nullptr)
    *num_cols = (*group_block_table_cols)[static_cast<size_t>(group_id)];
  return &bt;
}

int64_t MultiKvCacheIndex::Find(std::string_view layer_name) const {
  if (layer_names == nullptr) return -1;
  for (size_t i = 0; i < layer_names->size(); ++i) {
    if ((*layer_names)[i] == layer_name) return static_cast<int64_t>(i);
  }
  return -1;
}

// ENG-MULTIKV-BYNAME. The counts and the payload locator. Counted rather than
// stored: a stored count is a second derivation of the vectors it describes, and
// a second derivation is the thing that can disagree with them.
int MultiKvCacheIndex::num_paged() const {
  if (payload_kinds == nullptr) return 0;
  int n = 0;
  for (uint8_t k : *payload_kinds)
    if (static_cast<KvCachePayload>(k) == KvCachePayload::kPaged) ++n;
  return n;
}

int MultiKvCacheIndex::num_recurrent() const {
  if (payload_kinds == nullptr) return 0;
  int n = 0;
  for (uint8_t k : *payload_kinds)
    if (static_cast<KvCachePayload>(k) == KvCachePayload::kRecurrent) ++n;
  return n;
}

bool MultiKvCacheIndex::PayloadAt(int64_t index, KvCachePayload* kind,
                                  int32_t* slot) const {
  // Written FIRST and in every path, so the false answer cannot leave a caller
  // holding a slot from a previous call.
  if (kind != nullptr) *kind = KvCachePayload::kPaged;
  if (slot != nullptr) *slot = -1;
  if (payload_kinds == nullptr || payload_slots == nullptr) return false;
  if (index < 0 || static_cast<size_t>(index) >= payload_kinds->size() ||
      static_cast<size_t>(index) >= payload_slots->size())
    return false;
  if (kind != nullptr)
    *kind = static_cast<KvCachePayload>(
        (*payload_kinds)[static_cast<size_t>(index)]);
  if (slot != nullptr) *slot = (*payload_slots)[static_cast<size_t>(index)];
  return true;
}

bool MultiKvCacheIndex::Resolve(std::string_view layer_name,
                                KvCachePayload* kind, int32_t* slot) const {
  return PayloadAt(Find(layer_name), kind, slot);
}

// KV-DSV4-MULTICACHE W5 (#2323). Trivial by construction, and that is the point:
// the rule it encodes ("a name-keyed set that reaches a model which has not
// claimed it is refused") is the one a future edit is most likely to invert or
// widen, and as a free function it is pinned by a test that needs no model.
bool MultiKvRefusalApplies(const MultiKvCacheIndex* mk, bool consumes_multi_kv) {
  return mk != nullptr && !consumes_multi_kv;
}

ForwardLogits ModelRegistry::Forward(LoadedModel& model,
                                     const ModelForwardInput& input) {
  // KV-DSV4-MULTICACHE W3 (#2068): a MULTI-CACHE topology reached the shared
  // decode seam, and the forward registered for this architecture does not
  // consume one. ONE now does — see `consumes_multi_kv` below — so the sentence
  // that read "no registered forward consumes one" is the one thing in this
  // comment that W5j had to change rather than extend.
  //
  // W3 makes the runner allocate every published cache — DeepSeek-V4-Flash's 167
  // across 43 layers, in seven groups at four different page sizes — and hand
  // them here keyed by the name each was published under. What to DO with a cache
  // set keyed that way is known by exactly ONE of the three architectures that
  // reach this guard (#2353, surveyed at `85f65b0e8`; the count was ZERO until
  // W5j). Each of the three, in its own way:
  //
  //   `DeepseekV4Model::Forward` and `::ForwardDevice` open with
  //   `(void)attn_meta; (void)attn_kv;`
  //   (`src/vllm/model_executor/models/deepseek_v4.cpp:3033-3034`, `:3105-3106`
  //   — the anchors this comment carried, `:2886-2887` and `:2959-2960`, were
  //   the values at the SHA W3 was written on and had moved by 147 lines).
  //
  //   `ForwardGlm5NextForConditionalGeneration` opens with
  //   `(void)input.attn_kv; (void)input.gdn_state;`
  //   (`glm5_next_registry.cpp:156-157`) and re-runs the whole prefix each step,
  //   which its own comment says in those words.
  //
  //   `ForwardQwen4ExpForConditionalGeneration` IS THE FIRST CONSUMER, and this
  //   bullet used to be the third refusal in the list. W5j (#2031) makes it
  //   resolve every cache through `MultiKvCacheIndex::Resolve` — group 0's paged
  //   K/V, group 2's indexer side cache and group 1's recurrent states, each by
  //   the name `MakeQwen4ExpKVCache` published it under — and read group 2's own
  //   gathered block table through `BlockTableForGroup`. It sets
  //   `consumes_multi_kv` and is therefore the one architecture this guard lets
  //   past. It still serves a SINGLE-SHOT prefill of one sequence and says so
  //   itself, BY NAME, at its own boundary.
  //
  // Letting the step run would discard a correctly allocated topology in silence
  // and report a decode rate for a full-recompute path, which is the
  // wrong-answer-not-a-crash shape this row exists to remove. So it refuses, and
  // it refuses by READING the channel rather than testing its nullness: the
  // count, the paged/recurrent split, the distinct group count and the first
  // published name all come out of the payload, so a channel that arrived empty
  // says something different.
  //
  // W5 (#2323) turned this from a BLANKET refusal into a DISPATCH. It is gated on
  // `ModelFactory::consumes_multi_kv`, so a model that has wired its forward to
  // read a name-keyed set proceeds, and every model that has not still refuses
  // here by name. Deleting the refusal outright was the one option W5 rejected:
  // it would restore this exact silent discard for every FUTURE model that
  // publishes a topology it cannot consume.
  //
  // The predicate lives in `MultiKvRefusalApplies` rather than inline here, so
  // the rule the refusal applies and the rule a test drives are the SAME
  // expression. An inline copy is a second derivation, and a second derivation
  // is the thing that can disagree with the one the test pins.
  //
  // ENG-MULTIKV-BYNAME added the paged/recurrent split to this message, because
  // the total stopped meaning "attention caches" the moment the channel started
  // carrying a recurrent group's layers. #2343 read the pre-split wording as a
  // contradiction — `22 KV cache(s) from 2 published group(s)` beside
  // `block tables gathered for 3 of 3` — which is what a total that silently
  // omitted 34 recurrent states looked like from the outside.
  //
  // WHAT REPLACES THIS IS NOT ONE WAVE, which is the correction #2353 carries.
  // Lifting the guard is a per-architecture capability the MODEL declares — the
  // polarity `ModelFactory`'s existing `stage_on_load` and offload bits already
  // use — and each of the three rows above owns its own arm of it. KV-DSV4-
  // MULTICACHE W5 is scoped as the DeepSeek-V4 path alone and never spoke for
  // the other two.
  //
  // THAT BIT EXISTS NOW, AND IT LANDED WITH ITS FIRST CONSUMER (W5j, #2031).
  // `ModelFactory::consumes_multi_kv` is the declaration and
  // `ForwardQwen4ExpForConditionalGeneration` is the arm that sets it, which is
  // the condition #2353 named for lifting this at all: "a capability nothing can
  // turn on has no arm a test could drive". The guard did not go away — it
  // NARROWED, from "any multi-cache topology" to "any multi-cache topology
  // reaching a forward that does not ask by name", which is still DeepSeek-V4
  // and GLM-5-Next and still every model ported after them until one of them
  // wires the channel.
  //
  // LETTING A DECLARED CONSUMER PAST IS NOT LETTING ITS INPUT PAST. The channel
  // can be malformed in ways only the model can see — a layer name nothing was
  // published under, a name that resolves to the wrong payload kind, a group
  // whose block table was never gathered — and the consuming forward refuses
  // each of those by name at its own boundary. This guard cannot: it does not
  // know which names the model expects.
  if (MultiKvRefusalApplies(input.multi_kv,
                            model.registration().factory->consumes_multi_kv)) {
    const MultiKvCacheIndex& mk = *input.multi_kv;
    // #2353: NAME THE ARRIVING ARCHITECTURE, and compute it rather than
    // enumerate. When W3 wrote this string DeepSeek-V4 was the only thing that
    // could publish a multi-cache topology, so "row KV-DSV4-MULTICACHE W5 owns
    // the consuming forward" was true by construction. THREE architectures
    // reach it now and the clause is false for two of them:
    //
    //   DeepseekV4ForCausalLM            7 groups, all attention. W5 does own it.
    //   Qwen4ExpForConditionalGeneration 3 groups. Owned by MODEL-MM-QWEN4-EXP.
    //                                    `Qwen4ExpTextModel::Forward` EXISTS on
    //                                    this head, but it prefills once from
    //                                    the two POSITIONAL channels and reads
    //                                    `multi_kv` nowhere, so it consumes
    //                                    nothing this guard holds.
    //   Glm5NextForConditionalGeneration 3 groups. Owned by that model's own row.
    //
    // KV-DSV4-MULTICACHE W5 owns ONE of those three, under either of the two
    // scopings it has carried this week. At `85f65b0e8`, the base this was
    // written on, `## Work breakdown` gave it the DeepSeek-V4 DSA-sparse path
    // that removes `deepseek_v4.cpp`'s `(void)attn_kv`. `44d795d96` (#2352)
    // then NARROWED it to the plumbing — "the caches REACHING the model and
    // each layer routing to its own", plus lifting this guard — and moved the
    // DSA algorithm to `MODEL-DSV4-DSA-COMPOSE`. Both scopings are DeepSeek-V4's
    // model half, so neither reaches `Qwen4ExpTextModel::Forward` or
    // `ForwardGlm5NextForConditionalGeneration`, which is the whole point of
    // naming the architecture instead of a row.
    //
    // THE THREE ARE NOT ENUMERATED IN THE STRING, DELIBERATELY. A hard-coded
    // list of rows in a refusal is precisely the construct #2288 has already
    // driven stale six times over on the sibling row, in both polarities. The
    // registered architecture is read at run time from the handle this function
    // already holds, so it cannot rot, and it is the one value that routes the
    // reader to the row that owes the work.
    //
    // WHAT MADE THE OMISSION EXPENSIVE, measured rather than supposed. #2343
    // drove GLM-5.3-Flash on `dgx:gpu0` on 2026-08-30 and stopped here on the
    // first step. Its index row, `docs/FEATURES.md`, `docs/USAGE.md` and
    // `.agents/claims/CLAIM-GLM53-FLASH-W5B2B.md` each then had to say in prose
    // what this string should have said: that the guard is the ENGINE's, that
    // it fires before dispatch to the model's own hook, and that the consuming
    // forward is owed by the model's row.
    const std::string arch(model.registration().architecture);
    VT_CHECK(false,
             std::string("model forward: architecture '") + arch +
                 "' reached this forward with " + std::to_string(mk.size()) +
                 " KV cache(s) (" + std::to_string(mk.num_paged()) +
                 " paged, " + std::to_string(mk.num_recurrent()) +
                 " recurrent) from " + std::to_string(mk.num_groups()) +
                 " published group(s), first '" +
                 std::string(mk.first_name()) +
                 "', with block tables gathered for " +
                 std::to_string(mk.num_group_block_tables()) + " of " +
                 std::to_string(mk.num_published_groups()) +
                 " published group(s), and the forward registered for '" +
                 arch +
                 "' does not consume a cache set keyed by layer name — its "
                 "ModelFactory leaves `consumes_multi_kv` false. Refusing "
                 "rather than discarding an allocated KV topology in silence. "
                 "THIS GUARD IS THE ENGINE'S (KV-DSV4-MULTICACHE W3, #2068) and "
                 "it fires for ANY architecture that publishes a multi-cache "
                 "topology, BEFORE dispatch to that architecture's own forward "
                 "hook, so the consuming forward is owed by the row that ports '" +
                 arch +
                 "' and not by the engine row that owns this guard. "
                 "#1925, #2068, #2353");
  }
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
