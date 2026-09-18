// PEFTHelper — adapter config from adapter_config.json.
//
// UPSTREAM (ported FROM, ground-every-impl rule; ${VLLM_SOURCE} @ 555967922/
// vLLM 0.26.0.dev0):
//   vllm/lora/peft_helper.py:19-132  class PEFTHelper
//                                    (r, lora_alpha, target_modules, bias,
//                                     modules_to_save, use_rslora, use_dora,
//                                     vllm_lora_scaling_factor, from_dict,
//                                     from_local_dir, validate_legal)
//
// A PEFTHelper parses an adapter_config.json, computes the LoRA scaling factor
// (alpha/r, or alpha/sqrt(r) for rsLoRA), and validates that the adapter uses
// only features vLLM supports. The scaling factor flows into every
// LoRALayerWeights created by LoRAModel::FromLoraTensors.
//
// Python's dataclass field filtering (from_dict) becomes explicit key-by-key
// extraction: the JSON dict is read for the known fields and unknown keys are
// ignored, mirroring upstream's `filtered_dict`.
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace vllm {
namespace lora {

// PEFTHelper (peft_helper.py:19-132) — adapter config from adapter_config.json.
struct PEFTHelper {
  int r = 0;
  int lora_alpha = 0;
  std::vector<std::string> target_modules;
  std::string bias = "none";
  std::optional<std::vector<std::string>> modules_to_save;
  bool use_rslora = false;
  bool use_dora = false;
  double vllm_lora_scaling_factor = 1.0;
  std::optional<int> vllm_max_position_embeddings;

  // Construct from individual fields. Validates r > 0 and computes scaling.
  // (peft_helper.py:53-60 __post_init__)
  PEFTHelper(int r, int lora_alpha, std::vector<std::string> target_modules,
             std::string bias = "none",
             std::optional<std::vector<std::string>> modules_to_save =
                 std::nullopt,
             bool use_rslora = false, bool use_dora = false,
             std::optional<int> max_pos_emb = std::nullopt);

  // from_dict (peft_helper.py:63-80): filter known fields, construct.
  static PEFTHelper FromDict(const nlohmann::json& config_dict);

  // from_local_dir (peft_helper.py:83-116): read adapter_config.json, call
  // FromDict. max_position_embeddings is injected into the config as
  // vllm_max_position_embeddings.
  static PEFTHelper FromLocalDir(
      const std::string& lora_path,
      std::optional<int> max_position_embeddings = std::nullopt);

  // _validate_features (peft_helper.py:42-51): returns list of error messages.
  std::vector<std::string> ValidateFeatures() const;

  // validate_legal (peft_helper.py:118-132): validates against max_lora_rank.
  // Throws std::invalid_argument with all error messages joined.
  void ValidateLegal(int max_lora_rank) const;
};

}  // namespace lora
}  // namespace vllm
