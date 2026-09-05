// Upstream's PROMPT-UPDATE machinery: a LIST of per-modality replacements, each
// keyed by its own target id sequence, all applied in ONE pass over the prompt.
//
// Mirrors `vllm/multimodal/processing/processor.py` read in `~/_git/vllm` at
// **`9035151d6`** (`git rev-parse 9035151d6` =
// `9035151d6c9fb726181469f9e6aa9ccbf9a5dacb`). The parity pin
// `5559679229bc961848b121ccdeaa8fa5d79bec98` is where the rest of this port
// reads from; it carries no `dots3_note` at all, which is why this file's
// anchors name the later SHA.
//
// WHY THIS EXISTS AND THE TWO EXPANDERS DO NOT SUFFICE (W8a, #2860).
// `multimodal::ExpandImagePlaceholders` (`qwen3vl_processor.cpp:175-206`) and
// `multimodal::ExpandAudioPlaceholders` (`audio_processor.cpp:326-345`) each
// REBUILD the whole id vector and report offsets into the vector THEY built.
// Running them in sequence over one prompt measures the second one's offsets
// against the first one's UN-expanded input, so a request carrying an image AND
// an audio part gets one span that is short by exactly the other's expansion.
// Nothing downstream can see it: the counts still balance, the runner's
// `n_rows == n_masked` still holds, and the model answers fluently from the
// wrong rows. That is why the dots3-note seam REFUSED a mixed request until
// W8a rather than chaining the two.
//
// Upstream never chains. `_get_prompt_updates` builds a list of
// `PromptReplacement`s — image, audio, video, each with its own target — and
// `apply_token_matches` walks the id stream ONCE, taking the earliest match
// across all of them at each step (`processor.py:944-957`, `:906-941`,
// `:799-857`). This header is that shape.
#ifndef VLLM_MULTIMODAL_PROCESSING_PROCESSOR_H_
#define VLLM_MULTIMODAL_PROCESSING_PROCESSOR_H_

#include <cstdint>
#include <string>
#include <vector>

namespace vllm::multimodal {

// `PromptUpdateDetails` (`processor.py:206-256`) narrowed to the
// `select_token_id` shape (`:245-256`), which is the one every dots3-note
// modality uses: the FULL replacement sequence, plus which of its positions are
// EMBEDDING rows.
//
// The embed positions are a contiguous RUN here, where upstream's
// `is_embed` is a boolean mask. Both dots3-note modalities emit
// `[start] + pad * n + [end]`, whose embed positions are exactly `[1, n]`, and
// upstream's own non-contiguous case is VIDEO, which left the row with it: its
// replacement hands the planner a per-item `video_embed_mask` through
// `select_video_embeds` (`common/processor.py:795-802`) instead of one pad id,
// because `preprocess_dots3_note_video` expands one video into INTERLEAVED
// image and audio parts (`common/video.py:383-395`). `use_audio_in_video` is
// NOT that mechanism: the name does not occur anywhere in
// `vllm/models/dots3_note/` at `9035151d6` — it is Qwen-Omni's flag
// (`qwen2_5_omni_thinker.py`) — and this comment took it from
// `scheduler.cpp:505`, which names it for the same idea and is another row's to
// correct. `MultiModalFeatureSpec` carries no mask either
// (`include/vllm/multimodal/inputs.h:118-127`), and `try_schedule_encoder_inputs`
// records at `scheduler.cpp:504-509` that the two counts coincide until an
// interleaved modality lands. Widening this to a mask is that brick's work, in
// the same change that gives the feature spec one.
struct PromptUpdateContent {
  std::vector<int32_t> full;
  int embed_offset = 0;  // into `full`, of the first embedding row
  int embed_length = 0;  // how many embedding rows follow it
};

// `PromptReplacement` (`processor.py:423-519`): ONE modality's rule.
//
// `target` is the id sequence a match must equal — for both dots3-note
// modalities the `[start, pad, end]` triple upstream keys on
// (`common/processor.py:749-756` for image, `:777-783` for audio). `items` is
// one replacement per multimodal item of this modality, consumed IN ORDER as
// the pass finds targets, which is upstream's `item_idx`.
struct PromptReplacement {
  std::string modality;
  std::vector<int32_t> target;
  std::vector<PromptUpdateContent> items;
};

// `PromptUpdateDetails.select_token_id` (`processor.py:245-256`) for a
// start/pad/end triple: the target is `[start, pad, end]` and item `i` replaces
// it with `[start] + pad * counts[i] + [end]`, whose embedding rows are the
// `counts[i]` pads in the middle. This is `image_replacement`
// (`common/processor.py:740-747`) and `audio_replacement` (`:768-776`) with the
// per-item count already resolved by the caller.
PromptReplacement MakeTokenTripleReplacement(std::string modality,
                                             int32_t start_id, int32_t pad_id,
                                             int32_t end_id,
                                             const std::vector<int>& counts);

// One applied replacement, reported in the order it occupies the OUTPUT stream.
//
// THE ORDER IS LOAD-BEARING AND IT IS THE STREAM'S, NOT THE MODALITY'S.
// `GetMmFeaturesInWindow` (`utils.cpp:9-50`) is a pair of BINARY SEARCHES over
// `offset` and `offset + length`, and both the scheduler (`scheduler.cpp:495`)
// and the runner (`runner.cpp:2024`) call it. A feature list that is not sorted
// ascending by `offset` makes both searches skip an item, after which the
// runner either reports an encoder-cache miss or scatters the wrong rows.
struct AppliedPromptUpdate {
  std::string modality;
  int item_index = 0;  // which item of that modality's rule this was
  int offset = 0;      // into the EXPANDED ids, of the first embedding row
  int length = 0;      // how many embedding rows follow it
};

// `apply_token_matches` (`processor.py:944-957`), over the REPLACE-mode,
// non-empty-target subset.
//
// Walks `prompt_ids` ONCE. At each step it asks every rule that still has items
// for the FIRST occurrence of its target at or after the end of the previous
// applied match, takes the earliest of those, and emits the prompt bytes before
// it followed by that item's replacement. A tie goes to the rule that appears
// EARLIER in `updates`, which is upstream's `_next_priority` tie-break — the
// `min(..., key=lambda item: (item[1], _next_priority(item[0])))` at
// `:871-874`, over the queue priority `_next_priority` returns at `:794-797` —
// and its documented "the modality that appears earlier ... takes priority"
// (`:952-954`).
//
// WHAT IS NARROWED, said here rather than implied. Upstream's planner is also
// general over INSERT mode and over an EMPTY target (`:833-869`). Neither is
// representable here and both are REFUSED BY NAME: an empty target throws,
// because a port that treated it as a no-op would silently drop the item, and
// there is no insert mode to select. Every dots3-note rule is a non-empty
// REPLACE, and on that subset upstream's planner reduces exactly to this loop.
//
// THROWS BY NAME when a rule's items are not all consumed. Upstream reaches the
// same conclusion through `_all_items_found` (`:896-903`). Dropping an item
// quietly is the one outcome that produces a fluent WRONG answer: the encoder
// never runs for it, the placeholder run is never written, and the prompt reads
// as if the user had not sent that media at all.
//
// `applied` may be null. When it is not, it is CLEARED first and filled in
// ascending `offset` order.
std::vector<int32_t> ApplyPromptReplacements(
    const std::vector<int32_t>& prompt_ids,
    const std::vector<PromptReplacement>& updates,
    std::vector<AppliedPromptUpdate>* applied);

}  // namespace vllm::multimodal

#endif  // VLLM_MULTIMODAL_PROCESSING_PROCESSOR_H_
