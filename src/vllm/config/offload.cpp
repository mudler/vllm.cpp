// Ported from: vllm/config/offload.py:96-136 @ 555967922, plus the selection
// order in vllm/model_executor/offloader/base.py:139-149 and the segment match
// in vllm/model_executor/offloader/uva.py:91-93.
#include "vllm/config/offload.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace vllm {
namespace {

// uva.py:91-93 — `any(f".{param}." in f".{name}." for param in params)`. The
// dot-wrapping on BOTH sides is the whole mechanism: it anchors the needle to
// segment boundaries, which is what separates "w2_weight" from
// "w2_weight_scale" (offload.py:43).
bool DottedContains(const std::string& name, const std::string& segment) {
  if (segment.empty()) return false;
  const std::string haystack = "." + name + ".";
  const std::string needle = "." + segment + ".";
  return haystack.find(needle) != std::string::npos;
}

bool AnySegmentMatches(const std::set<std::string>& segments,
                       const std::string& name) {
  // An EMPTY set matches everything. Both sub-configs share that polarity, but
  // for DIFFERENT scopes: uva then offloads until its byte budget is reached
  // (offload.py:36-38), prefetch offloads all parameters of each offloaded
  // layer (:72-73). The scope difference lives in the callers, not here.
  if (segments.empty()) return true;
  for (const std::string& s : segments) {
    if (DottedContains(name, s)) return true;
  }
  return false;
}

const nlohmann::json* GetObject(const nlohmann::json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  if (!it->is_object()) {
    throw std::invalid_argument(std::string("offload config: \"") + key +
                                "\" must be a JSON object");
  }
  return &*it;
}

double GetNumber(const nlohmann::json& obj, const char* key, double fallback) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return fallback;
  if (!it->is_number()) {
    throw std::invalid_argument(std::string("offload config: \"") + key +
                                "\" must be a number");
  }
  return it->get<double>();
}

int64_t GetInt(const nlohmann::json& obj, const char* key, int64_t fallback) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return fallback;
  if (!it->is_number_integer()) {
    throw std::invalid_argument(std::string("offload config: \"") + key +
                                "\" must be an integer");
  }
  return it->get<int64_t>();
}

std::set<std::string> GetStringSet(const nlohmann::json& obj, const char* key) {
  std::set<std::string> out;
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return out;
  if (!it->is_array()) {
    throw std::invalid_argument(std::string("offload config: \"") + key +
                                "\" must be an array of strings");
  }
  for (const auto& e : *it) {
    if (!e.is_string()) {
      throw std::invalid_argument(std::string("offload config: \"") + key +
                                  "\" must be an array of strings");
    }
    out.insert(e.get<std::string>());
  }
  return out;
}

}  // namespace

std::optional<OffloadBackend> parse_offload_backend(const std::string& s) {
  // offload.py:12 — exactly these three.
  if (s == "auto") return OffloadBackend::kAuto;
  if (s == "uva") return OffloadBackend::kUva;
  if (s == "prefetch") return OffloadBackend::kPrefetch;
  return std::nullopt;
}

const char* offload_backend_str(OffloadBackend backend) {
  switch (backend) {
    case OffloadBackend::kAuto:
      return "auto";
    case OffloadBackend::kUva:
      return "uva";
    case OffloadBackend::kPrefetch:
      return "prefetch";
  }
  return "";
}

bool UVAOffloadConfig::MatchesSegment(const std::string& param_name,
                                      const std::string& segment) {
  return DottedContains(param_name, segment);
}

bool UVAOffloadConfig::Targets(const std::string& param_name) const {
  return AnySegmentMatches(cpu_offload_params, param_name);
}

int64_t UVAOffloadConfig::cpu_offload_max_bytes() const {
  // offloader/base.py:155 — `int(uva.cpu_offload_gb * 1024**3)`. Python's int()
  // TRUNCATES toward zero; static_cast<int64_t> does the same for a positive
  // double, and Validate() has already refused a negative budget.
  return static_cast<int64_t>(cpu_offload_gb * 1024.0 * 1024.0 * 1024.0);
}

bool PrefetchOffloadConfig::OffloadsLayer(int64_t layer_idx) const {
  // offload.py:55-57 — "Group every N layers together. Offload last
  // `offload_num_in_group` layers of each group", worked example
  // group_size=8, num_in_group=2 -> 6,7,14,15,22,23,...
  if (offload_group_size <= 0) return false;
  if (layer_idx < 0) return false;
  const int64_t pos_in_group = layer_idx % offload_group_size;
  return pos_in_group >= offload_group_size - offload_num_in_group;
}

bool PrefetchOffloadConfig::Targets(const std::string& param_name) const {
  return AnySegmentMatches(offload_params, param_name);
}

void OffloadConfig::Validate() {
  // Idempotent: a second call must not accumulate duplicate warnings.
  warnings.clear();

  // --- the pydantic Field(ge=...) bounds (offload.py:23,54,62,66) -----------
  // Upstream enforces these at construction, so they are separate from the
  // @model_validator's cross-field checks below. Both throw.
  if (!(uva.cpu_offload_gb >= 0.0) || std::isnan(uva.cpu_offload_gb)) {
    throw std::invalid_argument(
        "cpu_offload_gb (" + std::to_string(uva.cpu_offload_gb) +
        ") must be >= 0");
  }
  if (prefetch.offload_group_size < 0) {
    throw std::invalid_argument(
        "offload_group_size (" + std::to_string(prefetch.offload_group_size) +
        ") must be >= 0");
  }
  if (prefetch.offload_num_in_group < 1) {
    throw std::invalid_argument(
        "offload_num_in_group (" +
        std::to_string(prefetch.offload_num_in_group) + ") must be >= 1");
  }
  if (prefetch.offload_prefetch_step < 0) {
    throw std::invalid_argument(
        "offload_prefetch_step (" +
        std::to_string(prefetch.offload_prefetch_step) + ") must be >= 0");
  }

  // --- validate_offload_config (offload.py:98-112) --------------------------
  // The guard is an OR (:99): an explicit `prefetch` backend enables the block
  // even when offload_group_size is still 0.
  if (offload_backend == OffloadBackend::kPrefetch ||
      prefetch.offload_group_size > 0) {
    if (prefetch.offload_num_in_group > prefetch.offload_group_size) {
      throw std::invalid_argument(
          "offload_num_in_group (" +
          std::to_string(prefetch.offload_num_in_group) +
          ") must be <= offload_group_size (" +
          std::to_string(prefetch.offload_group_size) + ")");
    }
    if (prefetch.offload_prefetch_step < 1) {
      throw std::invalid_argument(
          "offload_prefetch_step (" +
          std::to_string(prefetch.offload_prefetch_step) +
          ") must be >= 1 when prefetch offloading is enabled "
          "(offload_group_size > 0)");
    }
  }

  // --- the three mismatch warnings (offload.py:114-135) ---------------------
  // Warnings, never errors: a mismatched pair is HONOURED with the backend
  // winning, not refused. Upstream's chain is if/elif/elif, so at most one
  // fires.
  const bool uva_active = uva.cpu_offload_gb > 0;
  const bool prefetch_active = prefetch.offload_group_size > 0;
  if (offload_backend == OffloadBackend::kUva && prefetch_active) {
    warnings.push_back(
        "Prefetch offload fields are set but offload_backend='uva'. "
        "Prefetch settings will be ignored.");
  } else if (offload_backend == OffloadBackend::kPrefetch && uva_active) {
    warnings.push_back(
        "UVA offload fields are set but offload_backend='prefetch'. "
        "UVA settings will be ignored.");
  } else if (offload_backend == OffloadBackend::kAuto && uva_active &&
             prefetch_active) {
    warnings.push_back(
        "Both UVA and prefetch offload fields are set with "
        "offload_backend='auto'. Prefetch backend will be selected. "
        "Set offload_backend explicitly to suppress this warning.");
  }
}

std::optional<OffloadBackend> OffloadConfig::ResolvedBackend() const {
  // offloader/base.py:139-159. Under "auto" the order is prefetch-then-uva and
  // NoopOffloader otherwise; an explicit backend is returned as-is.
  if (offload_backend != OffloadBackend::kAuto) return offload_backend;
  if (prefetch.offload_group_size > 0) return OffloadBackend::kPrefetch;
  if (uva.cpu_offload_gb > 0) return OffloadBackend::kUva;
  return std::nullopt;
}

bool OffloadConfig::is_offloading_enabled() const {
  const std::optional<OffloadBackend> backend = ResolvedBackend();
  if (!backend.has_value()) return false;
  switch (*backend) {
    case OffloadBackend::kUva:
      return uva.cpu_offload_gb > 0;
    case OffloadBackend::kPrefetch:
      return prefetch.offload_group_size > 0;
    case OffloadBackend::kAuto:
      return false;  // unreachable: ResolvedBackend never returns kAuto
  }
  return false;
}

OffloadConfig parse_offload_config_json(const std::string& json_text) {
  OffloadConfig cfg;

  // Empty/blank == the default, inert config, mirroring how an empty
  // `kv_transfer_config` means the byte-identical default engine.
  const auto first = json_text.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return cfg;

  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    // nlohmann throws its own type; the caller contract is invalid_argument.
    throw std::invalid_argument(std::string("offload config: malformed JSON: ") +
                                e.what());
  }
  if (!doc.is_object()) {
    throw std::invalid_argument("offload config: must be a JSON object");
  }

  auto backend_it = doc.find("offload_backend");
  if (backend_it != doc.end() && !backend_it->is_null()) {
    if (!backend_it->is_string()) {
      throw std::invalid_argument(
          "offload config: \"offload_backend\" must be a string");
    }
    const std::string name = backend_it->get<std::string>();
    const std::optional<OffloadBackend> parsed = parse_offload_backend(name);
    if (!parsed.has_value()) {
      // A typo here would silently disable offloading, so it is refused rather
      // than defaulted. "disk" in particular belongs to ENG-EXPERT-STREAM and
      // must not be quietly accepted by this row.
      throw std::invalid_argument("offload config: unknown offload_backend \"" +
                                  name + "\" (expected auto, uva or prefetch)");
    }
    cfg.offload_backend = *parsed;
  }

  if (const nlohmann::json* u = GetObject(doc, "uva")) {
    cfg.uva.cpu_offload_gb = GetNumber(*u, "cpu_offload_gb", 0.0);
    cfg.uva.cpu_offload_params = GetStringSet(*u, "cpu_offload_params");
  }
  if (const nlohmann::json* p = GetObject(doc, "prefetch")) {
    cfg.prefetch.offload_group_size = GetInt(*p, "offload_group_size", 0);
    cfg.prefetch.offload_num_in_group = GetInt(*p, "offload_num_in_group", 1);
    cfg.prefetch.offload_prefetch_step = GetInt(*p, "offload_prefetch_step", 1);
    cfg.prefetch.offload_params = GetStringSet(*p, "offload_params");
  }

  return cfg;
}

}  // namespace vllm
