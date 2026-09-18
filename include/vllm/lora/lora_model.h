// LoRAModel + WeightsMapper + parse_fine_tuned_lora_name — the adapter
// checkpoint loader.
//
// UPSTREAM (ported FROM, ground-every-impl rule; ${VLLM_SOURCE} @ 555967922/
// vLLM 0.26.0.dev0):
//   vllm/lora/lora_model.py:60-307  class LoRAModel
//                                   (from_local_checkpoint, from_lora_tensors,
//                                    get_lora, check_lora_name,
//                                    _should_skip_module)
//   vllm/lora/utils.py:155-207      parse_fine_tuned_lora_name
//   vllm/lora/utils.py:210-216      is_base_embedding_weights
//   vllm/lora/utils.py:67-73        get_lora_id
//   vllm/model_executor/models/utils.py:47-135  WeightsMapper
//        (only orig_to_new_prefix and orig_to_new_substr are ported; the
//         regex, stacked, renaming, and suffix maps are not used by any LoRA
//         path and are omitted)
//
// from_local_checkpoint opens adapter_model.safetensors, validates that every
// LoRA module is expected, and builds a LoRAModel whose loras map is keyed by
// module name. Each LoRALayerWeights is created from the PEFTHelper config
// (rank, alpha, scaling) and filled from the safetensors tensor data, with
// BF16→F32 conversion so all weights are portable float.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/lora/lora_weights.h"
#include "vllm/lora/peft_helper.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace vllm {
namespace lora {

// WeightsMapper (models/utils.py:47-135) — minimal port: orig_to_new_prefix
// and orig_to_new_substr only. A None value in upstream means "drop the
// weight"; we represent that as nullopt in the optional<string> value type.
// The LoRA path never drops, so the convenience maps below use plain string
// values for the common case.
struct WeightsMapper {
  std::map<std::string, std::string> orig_to_new_prefix;
  std::map<std::string, std::string> orig_to_new_substr;

  // _map_name (models/utils.py:76-79 → 81-135): apply substr then prefix,
  // return nullopt to drop. Both maps are applied in upstream order.
  std::optional<std::string> MapName(const std::string& key) const;
};

// parse_fine_tuned_lora_name (utils.py:155-207).
// Returns {module_name, is_lora_a}. Throws std::invalid_argument on an
// unsupported weight name.
std::pair<std::string, bool> ParseFineTunedLoraName(
    const std::string& name,
    const WeightsMapper* weights_mapper = nullptr);

// is_base_embedding_weights (utils.py:210-216).
bool IsBaseEmbeddingWeights(const std::string& name);

// get_lora_id (utils.py:67-73): monotonic global counter.
int64_t GetLoraId();

// LoRAModel (lora_model.py:60-307) — one loaded LoRA adapter.
class LoRAModel {
 public:
  int64_t id = 0;
  int rank = 0;
  std::unordered_map<std::string, LoRALayerWeights> loras;
  bool is_3d_lora_weight = false;

  // __init__ (lora_model.py:63-88). Throws if lora_model_id <= 0.
  LoRAModel(int64_t lora_model_id, int rank,
            std::unordered_map<std::string, LoRALayerWeights> loras,
            bool is_3d_lora_weight = false);

  // get_lora (lora_model.py:101-103). Returns nullptr if not found.
  LoRALayerWeights* GetLora(const std::string& module_name);

  // check_lora_name (lora_model.py:105-106).
  bool CheckLoraName(const std::string& lora_name) const;

  // _should_skip_module (lora_model.py:108-114).
  static bool ShouldSkipModule(const std::string& module_name,
                               const std::vector<std::string>& skip_prefixes);

  // from_lora_tensors (lora_model.py:116-164): iterate the safetensors file's
  // tensors, parse each name, create LoRALayerWeights from PEFTHelper config,
  // and fill lora_a/lora_b from the tensor data (BF16→F32 converted).
  static LoRAModel FromLoraTensors(
      int64_t lora_model_id, const SafetensorsFile& file,
      const PEFTHelper& peft_helper,
      std::optional<int> model_vocab_size = std::nullopt,
      const WeightsMapper* weights_mapper = nullptr,
      const std::vector<std::string>& skip_prefixes = {});

  // from_local_checkpoint (lora_model.py:166-307): open
  // adapter_model.safetensors, validate expected modules, delegate to
  // FromLoraTensors. Only the safetensors path is ported (not .bin/.pt).
  static LoRAModel FromLocalCheckpoint(
      const std::string& lora_dir,
      const std::set<std::string>& expected_lora_modules,
      const PEFTHelper& peft_helper, int64_t lora_model_id,
      std::optional<int> model_vocab_size = std::nullopt,
      const WeightsMapper* weights_mapper = nullptr,
      const std::vector<std::string>& skip_prefixes = {});

 private:
  // check_unexpected_modules (lora_model.py:212-242): verify every tensor
  // name's module suffix is in expected_lora_modules. Throws
  // std::invalid_argument on unexpected modules.
  static void CheckUnexpectedModules(
      const SafetensorsFile& file,
      const std::set<std::string>& expected_lora_modules,
      const WeightsMapper* weights_mapper,
      const std::vector<std::string>& skip_prefixes,
      const std::string& lora_dir);
};

}  // namespace lora
}  // namespace vllm
