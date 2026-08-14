// Ported from: vllm/config/multimodal.py:17-43,212-236 @ 5559679229bc
// (see include/vllm/config/multimodal.h for scope and deviations).
#include "vllm/config/multimodal.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {
namespace {

// The option keys each DummyOptions subclass declares, beyond `count`
// (multimodal.py:23-43). `extra="forbid"` is what makes an unlisted key an
// error rather than a silently-kept extra, so the lookup below is exhaustive by
// construction: a modality with no entry here is BaseDummyOptions, which
// declares NOTHING but `count` (:233-234).
bool IsKnownOption(const std::string& modality, const std::string& key) {
  if (modality == "video") {
    return key == "num_frames" || key == "width" || key == "height";
  }
  if (modality == "image") return key == "width" || key == "height";
  if (modality == "audio") return key == "length";
  return false;
}

[[noreturn]] void Refuse(const std::string& detail) {
  throw std::invalid_argument("limit_mm_per_prompt: " + detail);
}

// multimodal.py:220-222 — the legacy spelling is an int, rewritten to
// {"count": <int>} before the dataclass sees it.
int ParseCount(const std::string& modality, const nlohmann::json& value) {
  if (!value.is_number_integer()) {
    Refuse("\"" + modality + "\".count must be an integer, got " + value.dump());
  }
  const int64_t count = value.get<int64_t>();
  // count: int = Field(999, ge=0) (multimodal.py:20).
  if (count < 0) {
    Refuse("\"" + modality + "\".count must be >= 0, got " +
           std::to_string(count));
  }
  return static_cast<int>(count);
}

}  // namespace

std::map<std::string, int> ParseLimitMmPerPromptJson(
    const std::string& json, std::vector<std::string>* ignored_options) {
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(json);
  } catch (const std::exception& e) {
    Refuse(std::string("value is not valid JSON (") + e.what() + ")");
  }
  if (!doc.is_object()) {
    Refuse("value must be a JSON object mapping a modality to its limit, e.g. "
           "'{\"image\": 2, \"video\": 0}'; got " +
           doc.dump());
  }

  std::map<std::string, int> limits;
  for (const auto& [modality, value] : doc.items()) {
    if (modality.empty()) Refuse("a modality name may not be empty");

    // The LEGACY format: a bare count (multimodal.py:87-88,220-222).
    if (!value.is_object()) {
      limits[modality] = ParseCount(modality, value);
      continue;
    }

    // The CONFIGURABLE format: {"count": N, <profiling options>}
    // (multimodal.py:90-92,224-232). `count` keeps its 999 default when the
    // object omits it — the object exists to carry the OPTIONS, so omitting the
    // count is not "no limit configured", it is the dataclass default.
    int count = kDefaultLimitPerPrompt;
    for (const auto& [key, option] : value.items()) {
      if (key == "count") {
        count = ParseCount(modality, option);
        continue;
      }
      if (!IsKnownOption(modality, key)) {
        // extra="forbid" (multimodal.py:23,32,39). Naming the key is the whole
        // value of the refusal: a dropped `num_frame` typo is invisible.
        Refuse("\"" + modality + "\" has no option \"" + key + "\"");
      }
      // Field(None, gt=0) on every option (multimodal.py:26-28,35-36,42).
      if (!option.is_number_integer() || option.get<int64_t>() <= 0) {
        Refuse("\"" + modality + "\".\"" + key + "\" must be an integer > 0, "
               "got " + option.dump());
      }
      // Validated, then dropped — see the header's recorded deviation.
      if (ignored_options != nullptr) {
        ignored_options->push_back(modality + "." + key);
      }
    }
    limits[modality] = count;
  }
  return limits;
}

}  // namespace vllm
