// vllm.cpp original container reader. Sliding-window normalization mirrors
// vllm/config/model.py:542-559,654-660,723-726,1232-1234; typed RoPE
// normalization mirrors vllm/transformers_utils/config.py:439-509 and
// model_executor/layers/rotary_embedding/__init__.py:33-112,243-283
// @ e24d1b24fe96.
#include "vllm/transformers_utils/hf_config.h"

#include <fstream>
#include <limits>
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

bool GetBool(const nlohmann::json& doc, const char* key, bool fallback) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return fallback;
  return it->get<bool>();
}

std::vector<int64_t> GetIntArray(const nlohmann::json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return {};
  return it->get<std::vector<int64_t>>();
}

std::optional<double> GetOptionalDouble(const nlohmann::json& doc,
                                        const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return std::nullopt;
  return it->get<double>();
}

std::optional<int64_t> GetOptionalInt(const nlohmann::json& doc,
                                      const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return std::nullopt;
  return it->get<int64_t>();
}

std::optional<int64_t> GetSlidingWindow(const nlohmann::json& doc) {
  auto it = doc.find("sliding_window");
  if (it == doc.end() || it->is_null()) return std::nullopt;
  const int64_t window = it->get<int64_t>();
  // Pinned ModelConfig normalizes checkpoint sliding_window=0 to None before
  // max-length verification and backend construction (config/model.py:654-660).
  return window == 0 ? std::nullopt : std::optional<int64_t>(window);
}

void RequireKey(const nlohmann::json& doc, const char* key,
                const std::string& path) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) {
    throw std::runtime_error("hf_config: missing required field \"" +
                             std::string(key) + "\" in " + path);
  }
}

// Returns a pointer to the object member `key` of `doc` if it exists and is a
// JSON object, else nullptr.
const nlohmann::json* FindObject(const nlohmann::json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || !it->is_object()) return nullptr;
  return &*it;
}

// Resolves the effective "text config" json object, mirroring upstream
// PretrainedConfig.get_text_config() + the _CONFIG_ATTRS_MAPPING alias
// {"llm_config": "text_config"} (vllm/transformers_utils/config.py:134) and the
// thinker_config.text_config path (thinker_uses_mrope, config.py:529). Composite
// (multimodal wrapper) configs nest the text-model fields under a `text_config`
// (or `llm_config`, or `thinker_config.text_config`) sub-dict; plain dense
// configs have no such nesting and resolve to the top-level doc itself.
const nlohmann::json& ResolveTextConfig(const nlohmann::json& doc) {
  if (const nlohmann::json* text = FindObject(doc, "text_config")) return *text;
  if (const nlohmann::json* llm = FindObject(doc, "llm_config")) return *llm;
  if (const nlohmann::json* thinker = FindObject(doc, "thinker_config")) {
    if (const nlohmann::json* text = FindObject(*thinker, "text_config")) {
      return *text;
    }
  }
  return doc;
}

// True for the Qwen3.5 / Qwen3-Next family, whose upstream config classes
// default partial_rotary_factor to 0.25 (qwen3_next.py:240, qwen3_5_moe.py:92).
// Both the wrapper's top-level model_type ("qwen3_5_moe") and the nested text
// model_type ("qwen3_5_moe_text") carry the signal, so we check either.
bool IsQwen35Family(const std::string& model_type) {
  return model_type == "qwen3_next" || model_type == "qwen3_5" ||
         model_type == "qwen3_5_moe" || model_type == "qwen3_5_text" ||
         model_type == "qwen3_5_moe_text";
}

bool LooksLikeNestedRopeParameters(const nlohmann::json& params) {
  if (params.empty()) return false;
  for (const auto& item : params.items()) {
    if (!item.value().is_object()) return false;
  }
  return true;
}

RopeParameters ParseRopeParameters(const nlohmann::json& text,
                                   double default_partial_rotary_factor,
                                   const std::string& path,
                                   bool* has_parameters) {
  RopeParameters params;
  params.rope_theta = GetDouble(text, "rope_theta", 10000.0);
  params.partial_rotary_factor = GetDouble(
      text, "partial_rotary_factor", default_partial_rotary_factor);

  // Transformers v5 exposes rope_parameters. Older checkpoints use
  // rope_scaling; pinned patch_rope_parameters standardizes either dictionary
  // before get_rope sees it. Prefer the modern spelling when both are present.
  const nlohmann::json* raw = FindObject(text, "rope_parameters");
  if (raw == nullptr) raw = FindObject(text, "rope_scaling");
  *has_parameters = raw != nullptr;
  if (raw == nullptr) return params;

  if (LooksLikeNestedRopeParameters(*raw)) {
    throw std::runtime_error(
        "hf_config: nested per-layer rope parameters are not implemented in " +
        path);
  }

  std::string modern_type = GetString(*raw, "rope_type");
  const std::string legacy_type = GetString(*raw, "type");
  if (!modern_type.empty() && !legacy_type.empty() &&
      modern_type != legacy_type &&
      !(legacy_type == "su" && modern_type == "longrope") &&
      !(legacy_type == "mrope" && modern_type == "default")) {
    throw std::runtime_error(
        "hf_config: conflicting rope_type '" + modern_type + "' and type '" +
        legacy_type + "' in " + path);
  }
  if (modern_type.empty()) modern_type = legacy_type;
  if (modern_type.empty()) modern_type = "default";
  if (modern_type == "su") modern_type = "longrope";
  if (modern_type == "mrope") {
    if (raw->find("mrope_section") == raw->end()) {
      throw std::runtime_error(
          "hf_config: legacy rope type 'mrope' requires mrope_section in " +
          path);
    }
    modern_type = "default";
  }
  params.rope_type = modern_type;

  params.rope_theta = GetDouble(*raw, "rope_theta", params.rope_theta);
  params.partial_rotary_factor = GetDouble(
      *raw, "partial_rotary_factor", params.partial_rotary_factor);
  params.rope_dim = GetOptionalInt(*raw, "rope_dim");
  if (params.rope_dim.has_value() && *params.rope_dim == 0) {
    // Python's `if rotary_dim := ...` treats zero as absent.
    params.rope_dim.reset();
  }
  params.factor = GetOptionalDouble(*raw, "factor");
  params.original_max_position_embeddings =
      GetOptionalInt(*raw, "original_max_position_embeddings");
  params.extrapolation_factor =
      GetDouble(*raw, "extrapolation_factor", 1.0);
  params.attn_factor = GetDouble(*raw, "attn_factor", 1.0);
  params.beta_fast = GetInt(*raw, "beta_fast", 32);
  params.beta_slow = GetInt(*raw, "beta_slow", 1);
  params.apply_yarn_scaling =
      GetBool(*raw, "apply_yarn_scaling", true);
  params.truncate = GetBool(*raw, "truncate", true);
  params.mrope_section = GetIntArray(*raw, "mrope_section");
  params.mrope_interleaved = GetBool(*raw, "mrope_interleaved", false);

  if (params.rope_type == "yarn") {
    if (!params.factor.has_value() ||
        !params.original_max_position_embeddings.has_value()) {
      throw std::runtime_error(
          "hf_config: yarn rope requires factor and "
          "original_max_position_embeddings in " +
          path);
    }
    if (!(*params.factor > 0.0) ||
        *params.original_max_position_embeddings <= 0) {
      throw std::runtime_error(
          "hf_config: yarn factor and original_max_position_embeddings must "
          "be positive in " +
          path);
    }
  } else if (params.rope_type != "default") {
    // W5 deliberately relaxes the old completeness guard only for YaRN.
    // W6-W8 add their own typed fields and factory cases before relaxing this.
    throw std::runtime_error(
        "hf_config: checkpoint declares rope type '" + params.rope_type +
        "' which vllm.cpp does not implement yet (supported: default, yarn) in " +
        path);
  }

  if (!params.rope_dim.has_value() &&
      (!(params.partial_rotary_factor > 0.0) ||
       params.partial_rotary_factor > 1.0)) {
    throw std::runtime_error(
        "hf_config: partial_rotary_factor must be in (0, 1] in " + path);
  }
  return params;
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

  // Resolve the effective text config: for multimodal wrapper configs (e.g.
  // Qwen3_5MoeForConditionalGeneration) the text-model fields are nested under
  // `text_config`; for plain dense configs `text` aliases `doc`. `architectures`
  // and `model_type` are always read from the top-level wrapper doc.
  const nlohmann::json& text = ResolveTextConfig(doc);

  RequireKey(doc, "model_type", path);
  RequireKey(text, "hidden_size", path);
  RequireKey(text, "num_hidden_layers", path);

  HfConfig cfg;
  try {
    cfg.model_type = GetString(doc, "model_type");
    cfg.architectures = GetStringArray(doc, "architectures");
    cfg.hidden_size = GetInt(text, "hidden_size", 0);
    cfg.num_hidden_layers = GetInt(text, "num_hidden_layers", 0);
    cfg.vocab_size = GetInt(text, "vocab_size", 0);
    cfg.num_attention_heads = GetInt(text, "num_attention_heads", 0);
    // Absent -> MHA, per upstream convention.
    cfg.num_key_value_heads =
        GetInt(text, "num_key_value_heads", cfg.num_attention_heads);
    int64_t derived_head_dim =
        cfg.num_attention_heads > 0 ? cfg.hidden_size / cfg.num_attention_heads
                                    : 0;
    // Upstream only honors an explicit head_dim when it is > 0
    // (model_arch_config_convertor.py:61-75); absent or non-positive falls
    // back to hidden_size / num_attention_heads.
    cfg.head_dim = GetInt(text, "head_dim", 0);
    if (cfg.head_dim <= 0) cfg.head_dim = derived_head_dim;
    cfg.sliding_window = GetSlidingWindow(text);
    cfg.layer_types = GetStringArray(text, "layer_types");
    cfg.intermediate_size = GetInt(text, "intermediate_size", 0);

    cfg.num_experts = GetInt(text, "num_experts", 0);
    cfg.num_experts_per_tok = GetInt(text, "num_experts_per_tok", 0);
    cfg.moe_intermediate_size = GetInt(text, "moe_intermediate_size", 0);
    cfg.shared_expert_intermediate_size =
        GetInt(text, "shared_expert_intermediate_size", 0);

    cfg.linear_num_key_heads = GetInt(text, "linear_num_key_heads", 0);
    cfg.linear_num_value_heads = GetInt(text, "linear_num_value_heads", 0);
    cfg.linear_key_head_dim = GetInt(text, "linear_key_head_dim", 0);
    cfg.linear_value_head_dim = GetInt(text, "linear_value_head_dim", 0);
    cfg.linear_conv_kernel_dim = GetInt(text, "linear_conv_kernel_dim", 0);

    // Partial rotary factor. When the key is absent, upstream Qwen-family
    // config classes default it to 0.25 (qwen3_next.py:240, qwen3_5_moe.py:92);
    // all other models default to full rotary (1.0). The wrapper carries the
    // qwen signal on the top-level model_type ("qwen3_5_moe") while the nested
    // text config carries it as "qwen3_5_moe_text" -- check either.
    double default_partial_rotary_factor = 1.0;
    if (IsQwen35Family(cfg.model_type) ||
        IsQwen35Family(GetString(text, "model_type"))) {
      default_partial_rotary_factor = 0.25;
    }
    cfg.rope_parameters = ParseRopeParameters(
        text, default_partial_rotary_factor, path, &cfg.has_rope_parameters);
    cfg.rope_theta = cfg.rope_parameters.rope_theta;
    // Upstream get_rope gives a truthy explicit rope_dim precedence; otherwise
    // it truncates int(head_dim * partial_rotary_factor).
    cfg.rotary_dim = cfg.rope_parameters.rope_dim.has_value()
                         ? *cfg.rope_parameters.rope_dim
                         : static_cast<int64_t>(
                               cfg.rope_parameters.partial_rotary_factor *
                               static_cast<double>(cfg.head_dim));

    cfg.rms_norm_eps = GetDouble(text, "rms_norm_eps", 0.0);
    cfg.max_position_embeddings = GetInt(text, "max_position_embeddings", 0);
    // torch_dtype lives under the text config for nested wrappers, but some
    // wrappers only declare it at the top level -- fall back to the wrapper doc.
    cfg.torch_dtype = GetString(text, "torch_dtype");
    if (cfg.torch_dtype.empty() && &text != &doc) {
      cfg.torch_dtype = GetString(doc, "torch_dtype");
    }
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("hf_config: bad field type in " + path + ": " +
                             e.what());
  }

  cfg.raw = std::move(doc);
  return cfg;
}

}  // namespace vllm
