// LoRAModel + WeightsMapper + parse_fine_tuned_lora_name implementation.
//
// UPSTREAM (ported FROM, ${VLLM_SOURCE} @ 555967922):
//   vllm/lora/lora_model.py:60-307  class LoRAModel
//   vllm/lora/utils.py:155-216      parse_fine_tuned_lora_name,
//                                    is_base_embedding_weights
//   vllm/lora/utils.py:67-73        get_lora_id
//   vllm/model_executor/models/utils.py:47-135  WeightsMapper
#include "vllm/lora/lora_model.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace lora {

namespace {

// Split "a.b.c" → ["a", "b", "c"].
std::vector<std::string> SplitByDot(const std::string& s) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == '.') {
      parts.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return parts;
}

// Join parts[start, end) with ".".
std::string JoinByDot(const std::vector<std::string>& parts, size_t start,
                      size_t end) {
  std::string result;
  for (size_t i = start; i < end; ++i) {
    if (i > start) result += ".";
    result += parts[i];
  }
  return result;
}

// Convert a safetensors tensor to a float vector. F32 is a direct copy;
// BF16 widens each element to F32 (top 16 bits → full 32 bits).
std::vector<float> TensorToFloat(const StTensor& tensor) {
  size_t num_elements = 1;
  for (int64_t dim : tensor.shape) {
    num_elements *= static_cast<size_t>(dim);
  }
  std::vector<float> result(num_elements);
  if (tensor.dtype == "F32") {
    std::memcpy(result.data(), tensor.data, num_elements * sizeof(float));
  } else if (tensor.dtype == "BF16") {
    for (size_t i = 0; i < num_elements; ++i) {
      uint16_t bf16;
      std::memcpy(&bf16, tensor.data + i * 2, sizeof(uint16_t));
      uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
      std::memcpy(&result[i], &f32_bits, sizeof(float));
    }
  } else {
    throw std::invalid_argument("Unsupported LoRA tensor dtype: " + tensor.dtype);
  }
  return result;
}

}  // namespace

// --- WeightsMapper ---

// _map_name (models/utils.py:76-79 → 81-135). Applies substr replacements
// first, then prefix replacements, matching upstream order. Returns nullopt
// to signal "drop this weight" (not used by the LoRA path but faithful).
std::optional<std::string> WeightsMapper::MapName(const std::string& key) const {
  std::string result = key;

  // orig_to_new_substr (models/utils.py:108-113).
  for (const auto& [substr, new_key] : orig_to_new_substr) {
    size_t pos = result.find(substr);
    if (pos != std::string::npos) {
      result.replace(pos, substr.size(), new_key);
    }
  }

  // orig_to_new_prefix (models/utils.py:121-126).
  for (const auto& [prefix, new_key] : orig_to_new_prefix) {
    if (result.rfind(prefix, 0) == 0) {
      result.replace(0, prefix.size(), new_key);
    }
  }

  return result;
}

// --- Free functions ---

// parse_fine_tuned_lora_name (utils.py:155-207).
std::pair<std::string, bool> ParseFineTunedLoraName(
    const std::string& name_in,
    const WeightsMapper* weights_mapper) {
  std::string name = name_in;

  // utils.py:174-187: strip "base_model.model." prefix, apply mapper,
  // re-add prefix (or apply mapper directly if no prefix).
  if (name.rfind("base_model.model.", 0) == 0) {
    name = name.substr(17);  // strlen("base_model.model.")
    if (weights_mapper) {
      auto mapped = weights_mapper->MapName(name);
      if (!mapped) {
        throw std::invalid_argument("Mapped LoRA weight name cannot be None.");
      }
      name = "base_model.model." + *mapped;
    }
  } else {
    if (weights_mapper) {
      auto mapped = weights_mapper->MapName(name);
      if (!mapped) {
        throw std::invalid_argument("Mapped LoRA weight name cannot be None.");
      }
      name = *mapped;
    }
  }

  // utils.py:192: start_index = 2 if "base_model.model." prefix present.
  size_t start_index = (name.rfind("base_model.model.", 0) == 0) ? 2 : 0;

  std::vector<std::string> parts = SplitByDot(name);
  size_t n = parts.size();

  // utils.py:195-201: ...lora_A.weight or ...lora_B.weight
  if (n >= 2 && parts[n - 1] == "weight" &&
      (parts[n - 2] == "lora_A" || parts[n - 2] == "lora_B")) {
    std::string new_name = JoinByDot(parts, start_index, n - 2);
    return {new_name, parts[n - 2] == "lora_A"};
  }

  // utils.py:203-205: ...lora_embedding_A or ...lora_embedding_B
  if (n >= 1 &&
      (parts[n - 1] == "lora_embedding_A" ||
       parts[n - 1] == "lora_embedding_B")) {
    std::string new_name = JoinByDot(parts, start_index, n - 1);
    return {new_name, parts[n - 1] == "lora_embedding_A"};
  }

  throw std::invalid_argument(name + " is unsupported LoRA weight");
}

// is_base_embedding_weights (utils.py:210-216).
bool IsBaseEmbeddingWeights(const std::string& name) {
  static const std::string suffix1 = ".embed_tokens.base_layer.weight";
  static const std::string suffix2 = ".lm_head.base_layer.weight";
  return (name.size() >= suffix1.size() &&
          name.compare(name.size() - suffix1.size(), suffix1.size(),
                       suffix1) == 0) ||
         (name.size() >= suffix2.size() &&
          name.compare(name.size() - suffix2.size(), suffix2.size(),
                       suffix2) == 0);
}

// get_lora_id (utils.py:67-73).
int64_t GetLoraId() {
  static int64_t global_lora_id = 0;
  return ++global_lora_id;
}

// --- LoRAModel ---

// __init__ (lora_model.py:63-88).
LoRAModel::LoRAModel(int64_t lora_model_id, int rank_,
                     std::unordered_map<std::string, LoRALayerWeights> loras_,
                     bool is_3d)
    : id(lora_model_id),
      rank(rank_),
      loras(std::move(loras_)),
      is_3d_lora_weight(is_3d) {
  if (lora_model_id <= 0) {
    throw std::invalid_argument(
        "a valid lora id should be greater than 0");
  }
}

// get_lora (lora_model.py:101-103).
LoRALayerWeights* LoRAModel::GetLora(const std::string& module_name) {
  auto it = loras.find(module_name);
  return it == loras.end() ? nullptr : &it->second;
}

// check_lora_name (lora_model.py:105-106).
bool LoRAModel::CheckLoraName(const std::string& lora_name) const {
  return loras.find(lora_name) != loras.end();
}

// _should_skip_module (lora_model.py:108-114).
bool LoRAModel::ShouldSkipModule(const std::string& module_name,
                                 const std::vector<std::string>& skip_prefixes) {
  for (const auto& prefix : skip_prefixes) {
    // Match f".{prefix}" in module_name or module_name.startswith(prefix).
    std::string needle = "." + prefix;
    if (module_name.find(needle) != std::string::npos ||
        module_name.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

// from_lora_tensors (lora_model.py:116-164).
LoRAModel LoRAModel::FromLoraTensors(
    int64_t lora_model_id, const SafetensorsFile& file,
    const PEFTHelper& peft_helper,
    std::optional<int> model_vocab_size,
    const WeightsMapper* weights_mapper,
    const std::vector<std::string>& skip_prefixes) {
  std::unordered_map<std::string, LoRALayerWeights> loras;

  for (const std::string& tensor_name : file.Names()) {
    // lora_model.py:132-133: skip base embedding weights.
    if (IsBaseEmbeddingWeights(tensor_name)) {
      continue;
    }
    // lora_model.py:135-136: skip modules based on model-defined prefixes.
    if (!skip_prefixes.empty() &&
        ShouldSkipModule(tensor_name, skip_prefixes)) {
      continue;
    }

    auto [module_name, is_lora_a] =
        ParseFineTunedLoraName(tensor_name, weights_mapper);

    // lora_model.py:140-143: create from config if not seen.
    if (loras.find(module_name) == loras.end()) {
      LoRALayerWeights layer(module_name, peft_helper.r, peft_helper.lora_alpha,
                             {}, {}, 0, 0);
      // PEFTHelper computes the scaling (alpha/r or alpha/sqrt(r) for rsLoRA).
      // The constructor sets alpha/rank; override with the PEFT value.
      layer.scaling = peft_helper.vllm_lora_scaling_factor;
      loras[module_name] = std::move(layer);
    }

    LoRALayerWeights& layer = loras.at(module_name);
    const StTensor& tensor = file.Get(tensor_name);

    if (is_lora_a) {
      // lora_model.py:146-154: embedding vocab size check.
      if (tensor_name.find("lora_embedding_A") != std::string::npos &&
          model_vocab_size.has_value() &&
          model_vocab_size.value() !=
              static_cast<int>(tensor.shape.back())) {
        throw std::runtime_error(
            "The embedding LoRA size(" +
            std::to_string(tensor.shape.back()) +
            ") must be consistent with the base model's vocabulary size(" +
            std::to_string(model_vocab_size.value()) + ").");
      }
      layer.lora_a = TensorToFloat(tensor);
      // lora_a: [rank, input_dim] → input_dim = shape[1].
      layer.input_dim = static_cast<int>(tensor.shape.back());
    } else {
      layer.lora_b = TensorToFloat(tensor);
      // lora_b: [output_dim, rank] → output_dim = shape[0].
      layer.output_dim = static_cast<int>(tensor.shape.front());
    }
  }

  return LoRAModel(lora_model_id, peft_helper.r, std::move(loras));
}

// check_unexpected_modules (lora_model.py:212-242).
void LoRAModel::CheckUnexpectedModules(
    const SafetensorsFile& file,
    const std::set<std::string>& expected_lora_modules,
    const WeightsMapper* weights_mapper,
    const std::vector<std::string>& skip_prefixes,
    const std::string& lora_dir) {
  std::vector<std::string> unexpected_modules;

  for (const std::string& lora_module : file.Names()) {
    if (IsBaseEmbeddingWeights(lora_module)) {
      continue;
    }
    // lora_model.py:218-219: skip PEFT base_layer keys.
    if (lora_module.find("base_layer") != std::string::npos) {
      continue;
    }
    if (!skip_prefixes.empty() &&
        ShouldSkipModule(lora_module, skip_prefixes)) {
      continue;
    }

    auto [module_name, _] =
        ParseFineTunedLoraName(lora_module, weights_mapper);

    // lora_model.py:227-234: expert vs regular module check.
    size_t expert_pos = module_name.find(".experts");
    if (expert_pos != std::string::npos) {
      std::string expert_suffix = module_name.substr(expert_pos + 1);
      if (expected_lora_modules.find(expert_suffix) ==
          expected_lora_modules.end()) {
        unexpected_modules.push_back(module_name);
      }
    } else {
      size_t last_dot = module_name.rfind('.');
      std::string suffix = (last_dot != std::string::npos)
                               ? module_name.substr(last_dot + 1)
                               : module_name;
      if (expected_lora_modules.find(suffix) == expected_lora_modules.end()) {
        unexpected_modules.push_back(module_name);
      }
    }
  }

  if (!unexpected_modules.empty()) {
    std::string msg = "While loading " + lora_dir + ", expected target modules";
    msg += " but received ";
    for (size_t i = 0; i < unexpected_modules.size(); ++i) {
      if (i > 0) msg += ", ";
      msg += unexpected_modules[i];
    }
    msg += ". Please verify that the loaded LoRA module is correct";
    throw std::invalid_argument(msg);
  }
}

// from_local_checkpoint (lora_model.py:166-307). Only the safetensors path is
// ported; .bin and .pt are not supported.
LoRAModel LoRAModel::FromLocalCheckpoint(
    const std::string& lora_dir,
    const std::set<std::string>& expected_lora_modules,
    const PEFTHelper& peft_helper, int64_t lora_model_id,
    std::optional<int> model_vocab_size,
    const WeightsMapper* weights_mapper,
    const std::vector<std::string>& skip_prefixes) {
  std::string tensor_path = lora_dir + "/adapter_model.safetensors";

  if (!std::filesystem::exists(tensor_path)) {
    throw std::invalid_argument(lora_dir + " doesn't contain tensors");
  }

  SafetensorsFile file = SafetensorsFile::Open(tensor_path);

  // lora_model.py:269-271: validate expected modules before loading data.
  CheckUnexpectedModules(file, expected_lora_modules, weights_mapper,
                         skip_prefixes, lora_dir);

  return FromLoraTensors(lora_model_id, file, peft_helper, model_vocab_size,
                         weights_mapper, skip_prefixes);
}

}  // namespace lora
}  // namespace vllm
