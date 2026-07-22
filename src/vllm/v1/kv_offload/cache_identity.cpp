// vllm.cpp original. See include/vllm/v1/kv_offload/cache_identity.h for why
// this deliberately exceeds upstream's unread config.json + path-digest scheme.
#include "vllm/v1/kv_offload/cache_identity.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

#include "vllm/v1/core/kv_cache_utils.h"

namespace vllm::v1::kv_offload {
namespace {

std::string ToHex(const std::string& raw) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(raw.size() * 2);
  for (unsigned char c : raw) {
    out.push_back(kDigits[c >> 4]);
    out.push_back(kDigits[c & 0x0f]);
  }
  return out;
}

}  // namespace

std::string CacheIdentity::ToCanonicalJson() const {
  // nlohmann::json's object is an ordered map by default; dump() therefore
  // emits keys in the order inserted. We need SORTED keys for byte stability,
  // so build an std::map-backed ordered_json equivalent by using the standard
  // json type, which sorts keys in its default (std::map) configuration.
  nlohmann::json j;
  j["architectures"] = architectures;
  j["block_size"] = block_size;
  j["checkpoint_fingerprint"] = checkpoint_fingerprint;
  j["dcp_size"] = dcp_size;
  j["format_version"] = format_version;
  j["hash_algo"] = hash_algo;
  j["hash_block_size"] = hash_block_size;
  j["head_size"] = head_size;
  j["head_size_v"] = head_size_v;
  j["hf_config_digest"] = hf_config_digest;
  j["inference_engine"] = inference_engine;
  j["kv_cache_spec_kind"] = kv_cache_spec_kind;
  j["kv_dtype"] = kv_dtype;
  j["kv_quant_mode"] = kv_quant_mode;
  j["max_position_embeddings"] = max_position_embeddings;
  j["model_name"] = model_name;
  j["model_type"] = model_type;
  j["none_hash_hex"] = none_hash_hex;
  // none_hash_source is recorded OUTSIDE the digest (see the header): the same
  // seed value reached by two routes describes the same cache.
  j["num_hidden_layers"] = num_hidden_layers;
  j["num_kv_heads"] = num_kv_heads;
  j["page_size_bytes"] = page_size_bytes;
  j["pcp_size"] = pcp_size;
  j["pp_size"] = pp_size;
  j["rank"] = rank;
  j["rope_config"] = rope_config;
  j["sliding_window"] = sliding_window;
  j["tp_size"] = tp_size;
  j["weight_quantization"] = weight_quantization;
  // nlohmann::json's default object type is std::map, i.e. key-sorted, so
  // dump() is already canonical and separator-free.
  return j.dump();
}

CacheIdentity CacheIdentity::FromCanonicalJson(const std::string& json_text) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(json_text);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("kv_offload: cache identity is not valid JSON: ") +
        e.what());
  }
  if (!j.is_object()) {
    throw std::runtime_error("kv_offload: cache identity is not a JSON object");
  }
  CacheIdentity id;
  const auto str = [&](const char* key, std::string& dst) {
    if (j.contains(key) && j[key].is_string()) {
      dst = j[key].get<std::string>();
    }
  };
  const auto num = [&](const char* key, int64_t& dst) {
    if (j.contains(key) && j[key].is_number_integer()) {
      dst = j[key].get<int64_t>();
    }
  };
  if (j.contains("format_version") && j["format_version"].is_number_integer()) {
    id.format_version = j["format_version"].get<uint32_t>();
  }
  if (j.contains("architectures") && j["architectures"].is_array()) {
    id.architectures = j["architectures"].get<std::vector<std::string>>();
  }
  str("inference_engine", id.inference_engine);
  str("model_name", id.model_name);
  str("model_type", id.model_type);
  str("hf_config_digest", id.hf_config_digest);
  str("weight_quantization", id.weight_quantization);
  str("checkpoint_fingerprint", id.checkpoint_fingerprint);
  str("rope_config", id.rope_config);
  str("kv_cache_spec_kind", id.kv_cache_spec_kind);
  str("kv_dtype", id.kv_dtype);
  str("kv_quant_mode", id.kv_quant_mode);
  str("hash_algo", id.hash_algo);
  str("none_hash_hex", id.none_hash_hex);
  str("none_hash_source", id.none_hash_source);
  num("num_hidden_layers", id.num_hidden_layers);
  num("num_kv_heads", id.num_kv_heads);
  num("head_size", id.head_size);
  num("head_size_v", id.head_size_v);
  num("sliding_window", id.sliding_window);
  num("max_position_embeddings", id.max_position_embeddings);
  num("page_size_bytes", id.page_size_bytes);
  num("block_size", id.block_size);
  num("hash_block_size", id.hash_block_size);
  num("tp_size", id.tp_size);
  num("pp_size", id.pp_size);
  num("pcp_size", id.pcp_size);
  num("dcp_size", id.dcp_size);
  num("rank", id.rank);
  return id;
}

std::string CacheIdentity::Digest() const {
  return sha256_bytes(ToCanonicalJson());
}

std::string CacheIdentity::ShortDigestHex() const {
  // 12 hex chars, exactly upstream's _BASE_PATH_HASH_LEN (file_mapper.py:16).
  return ToHex(Digest()).substr(0, 12);
}

std::optional<std::string> CacheIdentity::Validate() const {
  if (inference_engine.empty()) return "inference_engine";
  if (model_name.empty()) return "model_name";
  if (model_type.empty()) return "model_type";
  if (architectures.empty()) return "architectures";
  if (hf_config_digest.empty()) return "hf_config_digest";
  // Deliberately rejected rather than defaulted: an empty scheme silently
  // reading as "unquantized" is exactly the silent-wrong-output hazard this
  // whole header exists to prevent.
  if (weight_quantization.empty()) return "weight_quantization";
  if (kv_cache_spec_kind.empty()) return "kv_cache_spec_kind";
  if (kv_dtype.empty()) return "kv_dtype";
  if (kv_quant_mode.empty()) return "kv_quant_mode";
  if (hash_algo.empty()) return "hash_algo";
  if (none_hash_hex.empty()) return "none_hash_hex";
  if (num_hidden_layers <= 0) return "num_hidden_layers";
  if (num_kv_heads <= 0) return "num_kv_heads";
  if (head_size <= 0) return "head_size";
  if (page_size_bytes <= 0) return "page_size_bytes";
  if (block_size <= 0) return "block_size";
  if (hash_block_size <= 0) return "hash_block_size";
  return std::nullopt;
}

std::optional<std::string> CacheIdentity::FirstMismatch(
    const CacheIdentity& a, const CacheIdentity& b) {
#define VLLM_KVOFF_CMP(field)  \
  if (!(a.field == b.field)) { \
    return #field;             \
  }
  VLLM_KVOFF_CMP(format_version)
  VLLM_KVOFF_CMP(inference_engine)
  VLLM_KVOFF_CMP(model_name)
  VLLM_KVOFF_CMP(model_type)
  VLLM_KVOFF_CMP(architectures)
  VLLM_KVOFF_CMP(hf_config_digest)
  VLLM_KVOFF_CMP(weight_quantization)
  VLLM_KVOFF_CMP(checkpoint_fingerprint)
  VLLM_KVOFF_CMP(num_hidden_layers)
  VLLM_KVOFF_CMP(num_kv_heads)
  VLLM_KVOFF_CMP(head_size)
  VLLM_KVOFF_CMP(head_size_v)
  VLLM_KVOFF_CMP(sliding_window)
  VLLM_KVOFF_CMP(rope_config)
  VLLM_KVOFF_CMP(max_position_embeddings)
  VLLM_KVOFF_CMP(kv_cache_spec_kind)
  VLLM_KVOFF_CMP(page_size_bytes)
  VLLM_KVOFF_CMP(block_size)
  VLLM_KVOFF_CMP(hash_block_size)
  VLLM_KVOFF_CMP(kv_dtype)
  VLLM_KVOFF_CMP(kv_quant_mode)
  VLLM_KVOFF_CMP(hash_algo)
  VLLM_KVOFF_CMP(none_hash_hex)
  VLLM_KVOFF_CMP(tp_size)
  VLLM_KVOFF_CMP(pp_size)
  VLLM_KVOFF_CMP(pcp_size)
  VLLM_KVOFF_CMP(dcp_size)
  VLLM_KVOFF_CMP(rank)
#undef VLLM_KVOFF_CMP
  return std::nullopt;
}

}  // namespace vllm::v1::kv_offload
