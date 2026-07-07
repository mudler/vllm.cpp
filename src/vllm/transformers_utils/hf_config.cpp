// vllm.cpp original (container reader); no upstream mirror.
#include "vllm/transformers_utils/hf_config.h"

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

    cfg.rope_theta = GetDouble(text, "rope_theta", 10000.0);
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
    const double partial_rotary_factor = GetDouble(
        text, "partial_rotary_factor", default_partial_rotary_factor);
    // Upstream truncates: int(head_dim * partial_rotary_factor)
    // (rotary_embedding/__init__.py:72).
    cfg.rotary_dim = static_cast<int64_t>(partial_rotary_factor *
                                          static_cast<double>(cfg.head_dim));

    // RoPE completeness guard (dep-audit 2026-07-07): we implement ONLY plain
    // (unscaled) NeoX RoPE (RopeNeox, cuda_ops.cu). vLLM's get_rope bakes
    // yarn/linear/llama3/longrope/mrope/dynamic scaling into a cos_sin_cache
    // (rotary_embedding/__init__.py). A checkpoint declaring any such variant would
    // otherwise be SILENTLY given unscaled embeddings — wrong output, no error signal.
    // HARD-FAIL at load instead; relax per-variant as each is implemented.
    // NOTE: `rope_type: "default"` with an `mrope_section` (the Qwen3.6 gate models)
    // is TEXT-SAFE: for text-only input all mrope position dims equal the text
    // position, so interleaved mrope collapses to plain NeoX — our text path is
    // correct. (mrope only diverges for multimodal image/video positions, which we
    // do not serve yet; guard that when the ViT path lands.) Only a genuine SCALING
    // rope_type (yarn/linear/llama3/longrope/dynamic) would silently miscompute here.
    for (const char* rope_key : {"rope_scaling", "rope_parameters"}) {
      const nlohmann::json* rs = FindObject(text, rope_key);
      if (rs == nullptr) continue;  // absent or null => plain NeoX, fine
      std::string rtype = GetString(*rs, "rope_type");
      if (rtype.empty()) rtype = GetString(*rs, "type");
      const bool is_plain = rtype.empty() || rtype == "default";
      if (!is_plain) {
        throw std::runtime_error(
            "hf_config: checkpoint declares rope " + std::string(rope_key) +
            " type '" + rtype +
            "' which vllm.cpp does not implement yet (only plain unscaled NeoX RoPE). "
            "Refusing to load rather than silently emit unscaled embeddings; implement "
            "the variant (cos_sin_cache in RopeNeox) before using this checkpoint.");
      }
    }
    // rope_theta / partial_rotary_factor may live under the newer nested
    // `rope_parameters` (Qwen3.6) rather than at the text_config top level. Prefer
    // the nested values when present so we don't silently fall back to the 10000
    // default when the checkpoint sets e.g. rope_theta=1e7. (dep-audit 2026-07-07)
    if (const nlohmann::json* rp = FindObject(text, "rope_parameters")) {
      const double rp_theta = GetDouble(*rp, "rope_theta", cfg.rope_theta);
      if (rp_theta != cfg.rope_theta) cfg.rope_theta = rp_theta;
      const double rp_prf = GetDouble(*rp, "partial_rotary_factor", partial_rotary_factor);
      if (rp_prf != partial_rotary_factor) {
        cfg.rotary_dim = static_cast<int64_t>(rp_prf * static_cast<double>(cfg.head_dim));
      }
    }

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
