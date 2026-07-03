// vllm.cpp original (container reader); no upstream mirror.
#include "vllm/transformers_utils/hf_config.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace vllm {

namespace {

int64_t GetInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return fallback;
  return it->get<int64_t>();
}

double GetDouble(const nlohmann::json& doc, const char* key, double fallback) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return fallback;
  return it->get<double>();
}

std::string GetString(const nlohmann::json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return {};
  return it->get<std::string>();
}

std::vector<std::string> GetStringArray(const nlohmann::json& doc,
                                        const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return {};
  return it->get<std::vector<std::string>>();
}

void RequireKey(const nlohmann::json& doc, const char* key,
                const std::string& path) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) {
    throw std::runtime_error("hf_config: missing required field \"" +
                             std::string(key) + "\" in " + path);
  }
}

}  // namespace

HfConfig LoadHfConfig(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("hf_config: cannot open " + path);
  }
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(in);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("hf_config: JSON parse error in " + path + ": " +
                             e.what());
  }
  if (!doc.is_object()) {
    throw std::runtime_error("hf_config: top-level JSON is not an object in " +
                             path);
  }

  RequireKey(doc, "model_type", path);
  RequireKey(doc, "hidden_size", path);
  RequireKey(doc, "num_hidden_layers", path);

  HfConfig cfg;
  try {
    cfg.model_type = GetString(doc, "model_type");
    cfg.architectures = GetStringArray(doc, "architectures");
    cfg.hidden_size = GetInt(doc, "hidden_size", 0);
    cfg.num_hidden_layers = GetInt(doc, "num_hidden_layers", 0);
    cfg.vocab_size = GetInt(doc, "vocab_size", 0);
    cfg.num_attention_heads = GetInt(doc, "num_attention_heads", 0);
    // Absent -> MHA, per upstream convention.
    cfg.num_key_value_heads =
        GetInt(doc, "num_key_value_heads", cfg.num_attention_heads);
    int64_t derived_head_dim =
        cfg.num_attention_heads > 0 ? cfg.hidden_size / cfg.num_attention_heads
                                    : 0;
    cfg.head_dim = GetInt(doc, "head_dim", derived_head_dim);
    cfg.layer_types = GetStringArray(doc, "layer_types");
    cfg.intermediate_size = GetInt(doc, "intermediate_size", 0);

    cfg.num_experts = GetInt(doc, "num_experts", 0);
    cfg.num_experts_per_tok = GetInt(doc, "num_experts_per_tok", 0);
    cfg.moe_intermediate_size = GetInt(doc, "moe_intermediate_size", 0);
    cfg.shared_expert_intermediate_size =
        GetInt(doc, "shared_expert_intermediate_size", 0);

    cfg.linear_num_key_heads = GetInt(doc, "linear_num_key_heads", 0);
    cfg.linear_num_value_heads = GetInt(doc, "linear_num_value_heads", 0);
    cfg.linear_key_head_dim = GetInt(doc, "linear_key_head_dim", 0);
    cfg.linear_value_head_dim = GetInt(doc, "linear_value_head_dim", 0);
    cfg.linear_conv_kernel_dim = GetInt(doc, "linear_conv_kernel_dim", 0);

    cfg.rope_theta = GetDouble(doc, "rope_theta", 10000.0);
    // Partial rotary (e.g. Qwen3-Next uses 0.25); absent -> full rotary.
    auto prf = doc.find("partial_rotary_factor");
    if (prf != doc.end() && !prf->is_null()) {
      cfg.rotary_dim = static_cast<int64_t>(
          std::llround(prf->get<double>() * static_cast<double>(cfg.head_dim)));
    } else {
      cfg.rotary_dim = cfg.head_dim;
    }

    cfg.rms_norm_eps = GetDouble(doc, "rms_norm_eps", 0.0);
    cfg.max_position_embeddings = GetInt(doc, "max_position_embeddings", 0);
    cfg.torch_dtype = GetString(doc, "torch_dtype");
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("hf_config: bad field type in " + path + ": " +
                             e.what());
  }

  cfg.raw = std::move(doc);
  return cfg;
}

}  // namespace vllm
