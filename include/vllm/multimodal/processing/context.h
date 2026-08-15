// Ported from: vllm/multimodal/processing/context.py @ 5559679229bc
//
// Scope (ENG-MM-INPUT-PIPELINE wave L1, #607): the ENFORCEMENT half of the
// per-modality input limits — `allowed_mm_limits` (:392-405) and
// `validate_num_items` (:409-428), plus both of upstream's call sites: the
// validate block inside `parse_mm_data` (:441-461) and the chat tracker's
// per-item check (entrypoints/chat_utils.py:648-662). The limits themselves are
// in vllm/config/multimodal.h.
//
// Why this belongs in the same wave as the limits, and not later: a limit
// nothing reads is a number, not a limit. Upstream's `--language-model-only`
// zeroes every modality limit, and its main observable effect is that the server
// then REFUSES every multimodal request — "At most 0 image(s) may be provided in
// one prompt." Shipping the zeroing without the refusal would ACCEPT a request
// upstream rejects, silently, on exactly the axis a user tests first
// (.agents/specs/multimodal-track.md §1.5).
//
// Ported subset: BaseProcessingInfo's limit surface only. The rest of that class
// (get_hf_config / get_hf_processor / get_tokenizer / the dummy-input and
// max-tokens-per-item hooks) and InputProcessingContext are DEFERRED with the M1
// processor work — they are the processor's plumbing, not the limits'.
//
// DEVIATIONS, recorded:
//   * Upstream reads `self.ctx.get_mm_config()` through InputProcessingContext,
//     which we do not have yet, so the MultiModalConfig is held by reference
//     here. It must outlive the BaseProcessingInfo — the model config owns it.
//   * `supported_mm_limits` upstream is a `@cached_property` over the model's own
//     `get_supported_mm_limits()` (:387-390). We have no per-model hook to call
//     yet (the models that would implement it are the M2 towers), so the map is
//     passed in. `std::nullopt` mirrors upstream's `None` = unlimited; a modality
//     ABSENT from the map mirrors "not supported at all" (:380-384).
//   * Upstream raises VLLMValidationError with a `parameter=modality` field for
//     structured error responses; we throw vllm::v1::InputValidationError, the
//     type api_server.cpp already maps to HTTP 400, and carry the modality in the
//     message exactly as upstream's own text does.
//   * chat_utils.py:637-647 remaps image/video to a "vision_chunk" modality when
//     `use_unified_vision_chunk_modality` is set. That is a per-model input
//     feature we have not ported, so `ValidateTrackedChatItem` takes the modality
//     it is given. When the vision-chunk models land, the remap goes in front of
//     this call exactly as upstream has it.
#ifndef VLLM_MULTIMODAL_PROCESSING_CONTEXT_H_
#define VLLM_MULTIMODAL_PROCESSING_CONTEXT_H_

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vllm/config/multimodal.h"

namespace vllm::multimodal {

// One modality's worth of parsed input, as `parse_mm_data` sees it after the
// data parser has run (context.py:441-442: `for modality, items in
// mm_items.items()`). `is_embedding` mirrors upstream's
// `isinstance(items, (EmbeddingItems, DictEmbeddingItems))` test at :443.
struct ParsedMmItems {
  std::string modality;
  int num_items = 0;
  bool is_embedding = false;
};

// BaseProcessingInfo's limit surface (context.py:373-461).
class BaseProcessingInfo {
 public:
  // `supported_mm_limits` mirrors get_supported_mm_limits() (:378-390): the
  // model's OWN ceiling per modality. std::nullopt == unlimited; a modality
  // absent from the map is not supported at all.
  BaseProcessingInfo(
      const MultiModalConfig& mm_config,
      std::map<std::string, std::optional<int>> supported_mm_limits)
      : mm_config_(mm_config),
        supported_mm_limits_(std::move(supported_mm_limits)) {}

  const MultiModalConfig& mm_config() const { return mm_config_; }
  const std::map<std::string, std::optional<int>>& supported_mm_limits() const {
    return supported_mm_limits_;
  }

  // allowed_mm_limits (context.py:392-405): the user's limit folded with the
  // model's by min(). A user limit NEVER raises a model's ceiling — that is the
  // whole point of the fold, and it is why `--limit-mm-per-prompt image=99` on a
  // single-image model still refuses the second image. Keyed by the SUPPORTED
  // modalities only (:396), so an unsupported modality is absent rather than 0.
  std::map<std::string, int> AllowedMmLimits() const;

  // validate_num_items (context.py:409-428). Throws vllm::v1::InputValidationError
  // with upstream's exact message when `num_items` exceeds the limit.
  void ValidateNumItems(const std::string& modality, int num_items) const;

  // The parse_mm_data validate block (context.py:441-461) — call site 1. Applies
  // the enable_mm_embeds escape per modality, then validates the rest.
  void ValidateParsedMmData(const std::vector<ParsedMmItems>& items) const;

  // The chat tracker's per-item check (entrypoints/chat_utils.py:630-662) — call
  // site 2. `modality` is the content-part modality as written by the caller,
  // e.g. "image" or "image_embeds"; the returned value is the resolved INPUT
  // modality the limit was checked against, or std::nullopt for "prompt_embeds",
  // which is not a multimodal modality at all (:632-633). `num_items` is the
  // caller's RUNNING count including this item (:648-652), so the refusal fires
  // across messages, not only within one.
  std::optional<std::string> ValidateTrackedChatItem(
      const std::string& modality, int num_items) const;

 private:
  const MultiModalConfig& mm_config_;
  std::map<std::string, std::optional<int>> supported_mm_limits_;
};

}  // namespace vllm::multimodal

#endif  // VLLM_MULTIMODAL_PROCESSING_CONTEXT_H_
