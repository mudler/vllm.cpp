// PEFTHelper implementation — adapter config from adapter_config.json.
//
// UPSTREAM (ported FROM, ground-every-impl rule; ${VLLM_SOURCE} @ 555967922/
// vLLM 0.26.0.dev0):
//   vllm/lora/peft_helper.py:19-132  class PEFTHelper
//                                    (__post_init__, from_dict, from_local_dir,
//                                     _validate_features, validate_legal)
#include "vllm/lora/peft_helper.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vllm {
namespace lora {

// __post_init__ (peft_helper.py:53-60): validate r > 0, compute scaling.
PEFTHelper::PEFTHelper(int r, int lora_alpha,
                       std::vector<std::string> target_modules,
                       std::string bias,
                       std::optional<std::vector<std::string>> modules_to_save,
                       bool use_rslora, bool use_dora,
                       std::optional<int> max_pos_emb)
    : r(r),
      lora_alpha(lora_alpha),
      target_modules(std::move(target_modules)),
      bias(std::move(bias)),
      modules_to_save(std::move(modules_to_save)),
      use_rslora(use_rslora),
      use_dora(use_dora),
      vllm_max_position_embeddings(max_pos_emb) {
  if (r <= 0) {
    throw std::invalid_argument(
        "LoRA rank `r` must be a positive integer, got " +
        std::to_string(r) + ".");
  }
  if (use_rslora) {
    vllm_lora_scaling_factor = static_cast<double>(lora_alpha) / std::sqrt(r);
  } else {
    vllm_lora_scaling_factor = static_cast<double>(lora_alpha) / r;
  }
}

// from_dict (peft_helper.py:63-80): filter known fields, construct.
PEFTHelper PEFTHelper::FromDict(const nlohmann::json& config_dict) {
  // Required fields: r, lora_alpha, target_modules (peft_helper.py:27-30).
  if (!config_dict.contains("r") || !config_dict.contains("lora_alpha") ||
      !config_dict.contains("target_modules")) {
    throw std::invalid_argument(
        "Missing required configuration fields: r, lora_alpha, or "
        "target_modules.");
  }

  int r_val = config_dict.at("r").get<int>();
  int alpha_val = config_dict.at("lora_alpha").get<int>();

  // target_modules can be a string or a list (peft_helper.py:30).
  std::vector<std::string> modules;
  const auto& tm = config_dict.at("target_modules");
  if (tm.is_string()) {
    modules.push_back(tm.get<std::string>());
  } else {
    modules = tm.get<std::vector<std::string>>();
  }

  // Optional fields with defaults (peft_helper.py:32-40).
  std::string bias_val = "none";
  if (config_dict.contains("bias")) {
    bias_val = config_dict.at("bias").get<std::string>();
  }

  std::optional<std::vector<std::string>> modules_to_save_val;
  if (config_dict.contains("modules_to_save") &&
      !config_dict.at("modules_to_save").is_null()) {
    modules_to_save_val =
        config_dict.at("modules_to_save").get<std::vector<std::string>>();
  }

  bool use_rslora_val = false;
  if (config_dict.contains("use_rslora")) {
    use_rslora_val = config_dict.at("use_rslora").get<bool>();
  }

  bool use_dora_val = false;
  if (config_dict.contains("use_dora")) {
    use_dora_val = config_dict.at("use_dora").get<bool>();
  }

  // vllm_lora_scaling_factor defaults to 1.0 but is recomputed in
  // __post_init__, so we do not read it from the dict.
  std::optional<int> max_pos_emb;
  if (config_dict.contains("vllm_max_position_embeddings") &&
      !config_dict.at("vllm_max_position_embeddings").is_null()) {
    max_pos_emb =
        config_dict.at("vllm_max_position_embeddings").get<int>();
  }

  return PEFTHelper(r_val, alpha_val, std::move(modules), std::move(bias_val),
                    std::move(modules_to_save_val), use_rslora_val,
                    use_dora_val, max_pos_emb);
}

// from_local_dir (peft_helper.py:83-116): read adapter_config.json, call
// FromDict. max_position_embeddings is injected into the config as
// vllm_max_position_embeddings.
PEFTHelper PEFTHelper::FromLocalDir(
    const std::string& lora_path,
    std::optional<int> max_position_embeddings) {
  std::string config_path = lora_path + "/adapter_config.json";

  std::ifstream f(config_path);
  if (!f.is_open()) {
    throw std::invalid_argument("Could not open adapter config: " + config_path);
  }

  std::stringstream ss;
  ss << f.rdbuf();
  nlohmann::json config = nlohmann::json::parse(ss.str());

  // peft_helper.py:115: config["vllm_max_position_embeddings"] = ...
  if (max_position_embeddings.has_value()) {
    config["vllm_max_position_embeddings"] = max_position_embeddings.value();
  } else {
    config["vllm_max_position_embeddings"] = nullptr;
  }

  return FromDict(config);
}

// _validate_features (peft_helper.py:42-51).
std::vector<std::string> PEFTHelper::ValidateFeatures() const {
  std::vector<std::string> error_msg;
  if (modules_to_save.has_value()) {
    error_msg.push_back("vLLM only supports modules_to_save being None.");
  }
  if (use_dora) {
    error_msg.push_back("vLLM does not yet support DoRA.");
  }
  return error_msg;
}

// validate_legal (peft_helper.py:118-132).
void PEFTHelper::ValidateLegal(int max_lora_rank) const {
  std::vector<std::string> error_msg = ValidateFeatures();
  if (r > max_lora_rank) {
    error_msg.push_back(
        "LoRA rank " + std::to_string(r) +
        " is greater than max_lora_rank " + std::to_string(max_lora_rank) + ".");
  }
  if (bias != "none") {
    error_msg.push_back("Adapter bias is not supported.");
  }
  if (!error_msg.empty()) {
    std::string joined;
    for (size_t i = 0; i < error_msg.size(); ++i) {
      if (i > 0) joined += " ";
      joined += error_msg[i];
    }
    throw std::invalid_argument(joined);
  }
}

}  // namespace lora
}  // namespace vllm
