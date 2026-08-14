// Ported from: vllm/config/multimodal.py:17-45,212-236 @ 5559679229bc
// (see include/vllm/config/multimodal.h for scope and deviations).
#include "vllm/config/multimodal.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {
namespace {

// The modalities `_validate_limit_per_prompt` routes to a dataclass of their own
// (multimodal.py:226-231). Everything else falls to the `else` at :232-233 and
// is built as a bare BaseDummyOptions.
bool IsBuiltinModality(const std::string& modality) {
  return modality == "video" || modality == "image" || modality == "audio";
}

// The option keys each BUILTIN DummyOptions subclass declares, beyond `count`
// (multimodal.py:24-45). Only those three carry
// `@dataclass(config=ConfigDict(extra="forbid"))` (:24,33,41), which is what
// makes an unlisted key an error rather than a silently-kept extra, so this
// lookup is exhaustive for exactly the modalities IsBuiltinModality names.
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
  // count: int = Field(999, ge=0) (multimodal.py:21) — declared on
  // BaseDummyOptions, so it is validated for EVERY modality, builtin or not.
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
      if (!IsBuiltinModality(modality)) {
        // BaseDummyOptions (multimodal.py:17-21) is the ONE dummy-options
        // dataclass declared WITHOUT `config=ConfigDict(extra="forbid")`, unlike
        // the three at :24,33,41 — so pydantic's default `extra='ignore'`
        // applies and an unlisted key on a modality outside image/video/audio is
        // DROPPED, not refused, with its value never validated. Re-derived
        // against the pinned oracle's own declarations under pydantic 2.12.5:
        // `BaseDummyOptions(count=2, foo=3)` returns `BaseDummyOptions(count=2)`
        // while `ImageDummyOptions(count=2, foo=3)` raises ValidationError.
        // Refusing here instead would refuse a document upstream accepts.
        if (ignored_options != nullptr) {
          ignored_options->push_back(modality + "." + key);
        }
        continue;
      }
      if (!IsKnownOption(modality, key)) {
        // extra="forbid" (multimodal.py:24,33,41). Naming the key is the whole
        // value of the refusal: a dropped `num_frame` typo is invisible.
        Refuse("\"" + modality + "\" has no option \"" + key + "\"");
      }
      // Field(None, gt=0) on every option (multimodal.py:28-30,37-38,45).
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
