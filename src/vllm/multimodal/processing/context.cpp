// Ported from: vllm/multimodal/processing/context.py @ 5559679229bc
// (see include/vllm/multimodal/processing/context.h for scope and deviations).
#include "vllm/multimodal/processing/context.h"

#include <algorithm>
#include <string>

#include "vllm/v1/engine/validation_error.h"

namespace vllm::multimodal {
namespace {

// The suffix upstream strips to get the input modality (chat_utils.py:635).
constexpr const char* kEmbedsSuffix = "_embeds";

bool EndsWithEmbeds(const std::string& modality) {
  const std::string suffix(kEmbedsSuffix);
  return modality.size() >= suffix.size() &&
         modality.compare(modality.size() - suffix.size(), suffix.size(),
                          suffix) == 0;
}

}  // namespace

// context.py:392-405.
std::map<std::string, int> BaseProcessingInfo::AllowedMmLimits() const {
  std::map<std::string, int> allowed_limits;
  for (const auto& [modality, supported_limit] : supported_mm_limits_) {
    const int user_limit = mm_config_.GetLimitPerPrompt(modality);
    allowed_limits[modality] = supported_limit.has_value()
                                   ? std::min(user_limit, *supported_limit)
                                   : user_limit;
  }
  return allowed_limits;
}

// context.py:409-428.
void BaseProcessingInfo::ValidateNumItems(const std::string& modality,
                                          int num_items) const {
  // A modality missing from either map defaults to 0 (:414-415) — the model does
  // not support it, so nothing is allowed.
  const auto supported_it = supported_mm_limits_.find(modality);
  const std::map<std::string, int> allowed = AllowedMmLimits();
  const auto allowed_it = allowed.find(modality);
  const int allowed_limit = allowed_it == allowed.end() ? 0 : allowed_it->second;

  // An unlimited model limit collapses to the allowed limit (:417-418), so the
  // user's limit becomes the only constraint.
  int supported_limit = 0;
  if (supported_it != supported_mm_limits_.end())
    supported_limit = supported_it->second.value_or(allowed_limit);

  const int limit = std::min(supported_limit, allowed_limit);
  if (num_items <= limit) return;

  std::string msg = "At most " + std::to_string(limit) + " " + modality +
                    "(s) may be provided in one prompt.";
  // The hint only when raising the user's limit would actually help: the model
  // can take this many, the configuration is what refused them (:425-426).
  if (num_items <= supported_limit)
    msg += " Set `--limit-mm-per-prompt` to increase this limit.";

  throw v1::InputValidationError(msg);
}

// context.py:441-461 — parse_mm_data's validate block.
void BaseProcessingInfo::ValidateParsedMmData(
    const std::vector<ParsedMmItems>& items) const {
  for (const ParsedMmItems& item : items) {
    if (item.is_embedding) {
      if (!mm_config_.enable_mm_embeds) {
        throw v1::InputValidationError(
            "You must set `--enable-mm-embeds` to input `" + item.modality +
            "_embeds`");
      }
      // The escape: embeddings at limit 0 skip the count check entirely, so an
      // encoder-free deployment can still be fed precomputed embeddings
      // (:453-459). A non-zero limit still applies to them.
      if (mm_config_.GetLimitPerPrompt(item.modality) == 0) continue;
    }
    ValidateNumItems(item.modality, item.num_items);
  }
}

// entrypoints/chat_utils.py:630-662 — the chat tracker's per-item check.
std::optional<std::string> BaseProcessingInfo::ValidateTrackedChatItem(
    const std::string& modality, int num_items) const {
  // Not a multimodal modality; it never reaches a limit (:632-633).
  if (modality == "prompt_embeds") return std::nullopt;

  std::string input_modality = modality;
  const std::string suffix(kEmbedsSuffix);
  if (EndsWithEmbeds(input_modality))
    input_modality.erase(input_modality.size() - suffix.size());

  // The same escape as parse_mm_data's, spelled as one conjunction upstream
  // (:653-660): the flag, a zero limit, AND an `*_embeds` content part. All
  // three, or the item is counted.
  const bool escapes = mm_config_.enable_mm_embeds &&
                       mm_config_.GetLimitPerPrompt(input_modality) == 0 &&
                       EndsWithEmbeds(modality);
  if (!escapes) ValidateNumItems(input_modality, num_items);

  return input_modality;
}

}  // namespace vllm::multimodal
