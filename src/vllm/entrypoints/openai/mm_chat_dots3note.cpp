// dots3-note's multimodal CHAT seam, registered on its own architecture
// (W6a, #2512).
//
// This is the SECOND architecture to reach the registry #2481 built, and it is
// what that row was for: adding a multimodal model is a NEW translation unit
// plus one `REGISTER_VLLM_MM_CHAT` line, with ZERO edits to any shared table.
// Before #2481 the server decided whether it could serve images by asking
// whether `<model_dir>/preprocessor_config.json` existed and then built
// Qwen3-VL's processor unconditionally — so this model would have been served
// Qwen3-VL's patch geometry, its merge size and its token ids, against its own
// `vision_config`, and answered 200.
//
// Ported from vLLM read in `~/_git/vllm` at **`9035151d6`**:
//   `Dots3NoteForCausalLM.get_placeholder_str` (`nvidia/multimodal.py:65-72`)
//     image -> f"{IMAGE_START}{IMAGE_PAD}{IMAGE_END}"
//   `IMAGE_START` / `IMAGE_PAD` / `IMAGE_END` (`common/processor.py:41-43`)
//     "<|img|>" / "<|imgpad|>" / "<|endofimg|>"
//   `_process_image_input` (`nvidia/multimodal.py:144-155`)
//     the placeholder run is `grid.prod(-1) // merge_size**2`
//   `MULTIMODAL_REGISTRY.register_processor` (`nvidia/multimodal.py:44-48`)
//     the registration sits ON THE MODEL and edits no shared table — the
//     mechanism this file mirrors.
//
// WHY THE MARKER IS BUILT HERE AND NOT TAKEN FROM `chat_mm.h`. That header's
// `ImagePlaceholderString()` returns Qwen3-VL's
// "<|vision_start|><|image_pad|><|vision_end|>" (`qwen3_vl.py:1716`), and
// `BuildMarkerInjectedContent` dispatches through it. Those are one
// architecture's markers, which is exactly the coupling #2475 removed from the
// install path; reaching for them here would put it back one layer down. What
// IS shared is everything that is not per-architecture: `ValidateChatMmLimits`
// (the per-item limit walk), `DecodeImageUrlPart` (the data-URI decode),
// `BaseProcessingInfo` (the `--limit-mm-per-prompt` fold) and, since W8a
// (#2860), `multimodal::ApplyPromptReplacements` — upstream's own list of
// per-modality `PromptReplacement`s applied in ONE pass over the id stream
// (`vllm/multimodal/processing/processor.py:944-957` @ `9035151d6`), which is
// what lets ONE request carry more than one `mm_feature`.
#include <filesystem>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/multimodal/dots3_note_processor.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/model_executor/models/dots3_note_audio.h"   // the audio refusal
#include "vllm/model_executor/models/dots3_note_vision.h"  // the tower refusal
#include "vllm/multimodal/processing/processor.h"  // the one-pass applier
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/validation_error.h"  // a bad upload is a 400, not a 500

namespace vllm::entrypoints::openai {

namespace {

using vllm::Dots3NoteAudioRefusalFor;
using vllm::Dots3NoteVisionRefusalFor;
using vllm::HfConfig;
using vllm::LoadHfConfig;

namespace fs = std::filesystem;

// TU-local, matching `mm_chat_qwen3vl.cpp:38-56`. A checkpoint path arrives as
// UTF-8 and `fs::path` is `wchar_t`-based on Windows.
fs::path NativeUtf8Path(const std::string& value) {
#if defined(_WIN32)
  const std::u8string utf8(reinterpret_cast<const char8_t*>(value.data()),
                           value.size());
  return fs::path(utf8);
#else
  return fs::path(value);
#endif
}

std::string PathUtf8(const fs::path& path) {
#if defined(_WIN32)
  const std::u8string utf8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
#else
  return path.string();
#endif
}

// `get_placeholder_str` (`nvidia/multimodal.py:65-68` @ `9035151d6`), the image
// branch. The SINGLE `<|imgpad|>` in the middle is what the tokenizer maps to
// ONE `image_token_id`, and the three together are the TARGET the image rule
// matches (`common/processor.py:749-756` @ `9035151d6`).
std::string Dots3NoteImageMarker() {
  return "<|img|><|imgpad|><|endofimg|>";
}

// The AUDIO branch of `get_placeholder_str` (`nvidia/multimodal.py:68-69` @
// `9035151d6`), W7a (#2703): `f"{AUDIO_START}{AUDIO_PAD}{AUDIO_END}"`.
//
// THE STRINGS COME FROM THE CONFIG, not from a literal, because `audio_config`
// carries them (`audio_comp_start` / `audio_comp_span` / `audio_comp_end`,
// `nvidia/audio.py:37-39`) and the marker injected here must be the one whose
// ids the processor resolved from the tokenizer. The image marker above is a
// literal only because `common/processor.py:41-43` hard-codes those three.
std::string Dots3NoteAudioMarker(
    const multimodal::Dots3NoteAudioProcessorConfig& cfg) {
  return cfg.audio_comp_start + cfg.audio_comp_span + cfg.audio_comp_end;
}

// `BuildMarkerInjectedContent`'s dots3 twin: walk the parts IN ORDER, append
// each text part's text and each image or audio part's marker at its position.
// A bare-string message is returned unchanged, so the text path is
// byte-identical.
//
// `audio_marker` is EMPTY when this install has no audio tower, and then an
// audio part contributes nothing — which it cannot reach anyway, because
// `ValidateChatMmLimits` has already refused it against a ceiling that does not
// declare "audio".
std::string BuildDots3NoteMarkerContent(const ChatMessage& message,
                                        const std::string& audio_marker) {
  if (!message.content_parts.has_value()) {
    return message.content.has_value() ? *message.content : std::string();
  }
  std::string out;
  for (const ChatContentPart& part : *message.content_parts) {
    if (part.type == "text") {
      out += part.text;
    } else if (part.type == "image_url") {
      out += Dots3NoteImageMarker();
    } else if (part.type == "input_audio" || part.type == "audio_url") {
      out += audio_marker;
    }
    // Every other part type contributes nothing here and is refused earlier by
    // ValidateChatMmLimits, whose supported-limit map below declares only the
    // modalities this seam can build features for.
  }
  return out;
}

// This seam's OWN ceiling — the other operand of the `min()` fold
// (`context.py:392-405`).
//
// W8a (#2860) RAISED IT TO UPSTREAM'S OWN NUMBERS. Upstream's
// `Dots3NoteProcessingInfo.get_supported_mm_limits` declares
// `{"image": 512, "video": 1, "audio": 128}` (`common/processor.py:527-534` @
// `9035151d6`). This seam declared `{"image": 1, "audio": 1}` until W8a, and
// that was the honest ceiling then: it located exactly one part of each kind
// and the two expanders could not be chained, so a second item could only have
// been dropped. `ApplyPromptReplacements` consumes one item per target
// occurrence in ONE pass, so 512 and 128 are now what this seam can actually
// build.
//
// `video` IS DELIBERATELY ABSENT, although upstream declares `{"video": 1}`
// beside them. `context.py:414-415` reads an absent modality as limit 0, so the
// entrypoint still refuses a video part with upstream's own "At most 0 video(s)
// may be provided in one prompt." — byte for byte the refusal this seam
// produced before W8a. Declaring 1 would promise a capability nothing builds:
// video decode needs a container demuxer, an H.264/VP9/AV1 bitstream decoder
// and a JPEG codec, none of which this tree vendors, and it left the row to
// issue #2814.
//
// `has_audio` IS A PARAMETER AND NOT A CONSTANT, because a checkpoint with no
// `audio_config` has no audio tower and declaring the modality would promise a
// capability that install could not deliver.
//
// A user limit can only LOWER these, so `--limit-mm-per-prompt image=1` is how
// an operator gets the pre-W8a ceiling back.
std::map<std::string, std::optional<int>> Dots3NoteChatSupportedMmLimits(
    bool has_audio) {
  std::map<std::string, std::optional<int>> limits{
      {"image", std::optional<int>(512)}};
  if (has_audio) limits.emplace("audio", std::optional<int>(128));
  return limits;
}

// ── W8a (#2860): PREPARE, then PLACE. ───────────────────────────────────────
//
// Before W8a each of these two functions did both: preprocess ONE item and
// expand its placeholder, each rebuilding the whole id vector and reporting an
// offset into the vector IT built. Chaining them over one prompt therefore
// measured the second one's offsets against the first one's UN-EXPANDED input,
// which is why the seam refused a mixed request rather than serving it half
// (spec §4.18.1).
//
// The split is upstream's own. `_get_prompt_updates` computes each item's
// replacement CONTENT (`common/processor.py:740-747` for image, `:768-776` for
// audio) and hands the whole list to `apply_token_matches`, which is the only
// thing that decides WHERE anything lands
// (`processing/processor.py:944-957`). So each `Prepare*` below produces the
// features, the hash and the placeholder COUNT for one item, and
// `RouteDots3NoteMultiModal` places every item of every modality in ONE pass.

// One image item, preprocessed but NOT yet placed.
struct PreparedImage {
  multimodal::ImageKwargs kwargs;
  std::string mm_hash;
  int num_tokens = 0;
};

// One audio item, the same.
struct PreparedAudio {
  multimodal::AudioKwargs kwargs;
  std::string mm_hash;
  int num_tokens = 0;
};

PreparedImage PrepareDots3NoteImage(
    const multimodal::Dots3NoteImageProcessor& proc, const uint8_t* rgb,
    int64_t height, int64_t width) {
  const multimodal::Dots3NoteProcessorConfig& cfg = proc.config();
  PreparedImage out;
  out.kwargs = proc.ProcessImage(rgb, height, width);
  const std::array<int64_t, 3>& grid = out.kwargs.image_grid_thw;
  const int64_t merge_length =
      static_cast<int64_t>(cfg.merge_size) * cfg.merge_size;
  // `image_replacement` (`common/processor.py:743-745` @ `9035151d6`):
  // `int(grid.prod()) // merge_size**2`. This is the SAME rule
  // `ExpandImagePlaceholders` applied while walking
  // (`qwen3vl_processor.cpp:191-192`); it is computed here because the one-pass
  // applier needs every count BEFORE it can place any span.
  out.num_tokens =
      static_cast<int>((grid[0] * grid[1] * grid[2]) / merge_length);
  out.mm_hash = proc.HashImage(rgb, height, width);
  return out;
}

// `PrepareDots3NoteImage`'s AUDIO twin (W7a, #2703; split out by W8a).
//
// WHY THIS IS NOT `RouteAudioWav` (`chat_mm.cpp:131-160`). That function is
// DEAD in `src/` — nothing outside `tests/` calls it — and its test is the
// `ROAD-V1-MM` parse gate, so it belongs to another row. It is also WRONG for
// this model in a way no shape check would report: it takes the token count
// from `AudioProcessorConfig::max_source_positions`, Whisper's FIXED 1500, and
// dots3's count is `ceil(num_samples / 1280)` and depends on the waveform.
// Editing it would move another row's gate to serve this one.
//
// THE CONTAINER REFUSAL IS HERE AND NOT IN THE PROCESSOR, because the container
// is a REQUEST property while the rate is a CONFIG one.
// `DecodeWavPcm16MeanToMono` already refuses a non-PCM16 or malformed buffer by
// name; what this adds is WHO OWES IT, so an operator learns that rather than
// only what failed.
//
// W7c-1 (#2813) and W7c-2 (#2828) NARROWED it, twice. It used to say a
// multi-channel WAV was owed to W7c and then that a non-16 kHz one was; any
// channel count is now served, mean-reduced as upstream reduces, and any
// sampling rate is served, resampled as upstream's own scipy arm resamples. And
// the container arm left this row entirely: it needs a demuxer this tree does
// not vendor, five surfaces refuse compressed media for that same missing
// brick, and #2814 owns it. The refusal stays at the SEAM and stays the SAME
// predicate as the route, because this throw is HTTP 400 for one request while
// the same throw from inside `encode_mm` would set `AsyncLLM`'s errored latch
// and 500 every later request, TEXT ones included.
PreparedAudio PrepareDots3NoteAudio(
    const multimodal::Dots3NoteAudioProcessor& proc, const DecodedMedia& audio) {
  multimodal::DecodedAudio decoded;
  try {
    // W7c-1 (#2813): the MEAN-reducing sibling, so a multi-channel PCM16 WAV
    // at the target rate is SERVED instead of refused. This is the production
    // call site the reachability mutation deletes.
    decoded = multimodal::DecodeWavPcm16MeanToMono(audio.bytes.data(),
                                                   audio.bytes.size());
  } catch (const std::exception& e) {
    // `InputValidationError`, NOT `std::runtime_error`. The container is a
    // property of the REQUEST, so this is a client error and
    // `ApiServer::handle_chat_completions` maps this type to HTTP 400
    // ("BadRequestError") while a bare `runtime_error` falls through to the
    // generic 500 arm. Upstream reaches the same place: `create_error_response`
    // maps `ValueError` to `BadRequestError`
    // (serve/utils/error_response.py:62-65). Reporting a malformed upload as a
    // SERVER fault is what the 500 arm is for, and it is not this.
    throw vllm::v1::InputValidationError(
        std::string("dots3-note audio chat seam: this request's audio is not a "
                    "PCM16 RIFF/WAVE buffer (") + e.what() +
        "). Only that container is ported. The `mp3`/`flac`/`ogg` an "
        "`input_audio.format` may name needs a demuxer this tree does not "
        "vendor; five surfaces refuse compressed media for that same missing "
        "brick, and it is tracked by issue #2814 — a SHARED brick, not a "
        "dots3-note one. Any CHANNEL count is served since W7c-1 (#2813): the "
        "channels are mean-reduced to mono, as upstream's "
        "`load_audio(..., mono=True)` does "
        "(vllm/multimodal/media/audio.py:207-208, :220 @ 9035151d6). Any "
        "SAMPLING RATE is served since W7c-2 (#2828): the waveform is "
        "resampled to `audio_config.sampling_rate` by upstream's own `scipy` "
        "AudioResampler arm (resample_audio_scipy, "
        "vllm/multimodal/audio.py:232-250 @ 9035151d6), NOT its `pyav` "
        "default, which is libswresample and is not bit-identical to itself "
        "across CPU dispatch. See .agents/specs/dots3-note.md §4.16 and §4.17, "
        "and issues #2813 and #2828.");
  }

  // Since W7c-2 (#2828) the front end RESAMPLES a rate that is not
  // `audio_config.sampling_rate` rather than refusing it, through upstream's
  // own `"scipy"` `AudioResampler` arm (spec §4.17). What it still refuses BY
  // NAME is a non-positive rate; a reduced polyphase ratio past
  // `kMaxPolyphaseRate`, which is a recorded divergence because the rate is
  // named by the request and upstream has no such guard; and — since W7b
  // (#2797) lifted the `chunk_seconds` ceiling — a waveform past ONE chunk on a
  // checkpoint whose `chunk_samples` is not a whole number of `token_stride`s,
  // where upstream's own per-segment row sum and its prompt-side
  // `ceil(total / stride)` disagree (spec §4.15.3). Every message names the
  // reason and the numbers.
  //
  // THIS IS THE FRONT END AND NOT THE ENGINE LOOP, and that is the point of
  // refusing here: `InputValidationError` becomes HTTP 400 for THIS request,
  // where the same throw from inside `encode_mm` would set `AsyncLLM`'s errored
  // latch and 500 every later request, text ones included.
  //
  // ONE RESAMPLE PER REQUEST, NOT TWO (PR #2842 F2). The mm-hash below needs the
  // SAME resampled waveform, and before this it got it by calling
  // `ResampleAudioScipy` a second time over the same input: on the 1 Hz request
  // measured in spec §4.17.10 that was 1220.7 MB twice for a 40 KB upload.
  // `ProcessWaveform` fills `resampled` when it resamples and leaves it empty
  // when it does not, and the hash below is handed the buffer rather than the
  // rate to redo it from.
  std::vector<float> resampled;
  PreparedAudio out;
  out.kwargs = proc.ProcessWaveform(
      decoded.samples.data(), static_cast<int64_t>(decoded.samples.size()),
      decoded.sampling_rate, &resampled);
  // `audio_replacement` (`common/processor.py:770-773` @ `9035151d6`):
  // `ceil(int(length) / stride)`, already resolved by `ProcessWaveform`.
  out.num_tokens = static_cast<int>(out.kwargs.num_tokens);
  // The mm-hash is over the WAVEFORM, before feature extraction, exactly as
  // the image hash is over the raw pixels — so the encoder cache keys on
  // audio rather than on what the front end derived from it.
  //
  // W7c-2 (#2828) PASSES THE REQUEST'S OWN RATE, and that is a correctness
  // requirement rather than a refinement. Two requests whose raw buffers are
  // identical and whose declared rates are not produce DIFFERENT features
  // since W7c-2, and `mm_hash` is a CROSS-REQUEST encoder-cache key, so a key
  // over the raw buffer alone would hand the second one the first one's
  // encoding. The three-argument overload hashes the RESAMPLED waveform,
  // which separates those two and also lets two requests that resample to the
  // same audio share one entry.
  out.mm_hash = proc.HashAudio(decoded.samples.data(),
                               static_cast<int64_t>(decoded.samples.size()),
                               decoded.sampling_rate,
                               resampled.empty() ? nullptr : &resampled);
  return out;
}

// THE ONE PASS (W8a, #2860), and the only place a placeholder gets an offset.
//
// Builds one `PromptReplacement` per modality that has items — image first,
// then audio, which is the order `_get_prompt_updates` appends them in
// (`common/processor.py:734`, `:757`) and therefore the tie-break upstream's
// planner would apply — and hands the whole list to
// `ApplyPromptReplacements`.
//
// THE `mm_features` COME OUT IN STREAM ORDER, NOT MODALITY ORDER, and that is a
// requirement rather than a nicety: `GetMmFeaturesInWindow` (`utils.cpp:9-50`)
// is a pair of BINARY SEARCHES over `offset`, and both the scheduler
// (`scheduler.cpp:495`) and the runner (`runner.cpp:2024`) call it, so a list
// out of order makes both skip an item. `kv_cache_utils.cpp:420` states the
// same precondition for the prefix-cache keys. The applier reports its spans in
// ascending offset by construction, so this loop must NOT be restructured as
// one pass per modality.
multimodal::MultiModalInputs RouteDots3NoteMultiModal(
    const multimodal::Dots3NoteProcessorConfig& image_cfg,
    const multimodal::Dots3NoteAudioProcessorConfig* audio_cfg,
    std::vector<PreparedImage> images, std::vector<PreparedAudio> audios,
    const std::vector<int32_t>& prompt_ids) {
  std::vector<multimodal::PromptReplacement> updates;
  if (!images.empty()) {
    std::vector<int> counts;
    counts.reserve(images.size());
    for (const PreparedImage& item : images) counts.push_back(item.num_tokens);
    updates.push_back(multimodal::MakeTokenTripleReplacement(
        "image", image_cfg.image_start_token_id, image_cfg.image_token_id,
        image_cfg.image_end_token_id, counts));
  }
  if (!audios.empty()) {
    // Unreachable with a null config while the seam's own limit map and its
    // audio processor agree; kept so that reaching it is a named refusal and
    // not a null dereference.
    if (audio_cfg == nullptr) {
      throw std::runtime_error(
          "dots3-note multimodal chat seam: an audio item reached the one-pass "
          "applier on an install with no audio processor config.");
    }
    std::vector<int> counts;
    counts.reserve(audios.size());
    for (const PreparedAudio& item : audios) counts.push_back(item.num_tokens);
    updates.push_back(multimodal::MakeTokenTripleReplacement(
        "audio", audio_cfg->audio_start_token_id, audio_cfg->audio_token_id,
        audio_cfg->audio_end_token_id, counts));
  }

  std::vector<multimodal::AppliedPromptUpdate> applied;
  multimodal::MultiModalInputs out;
  out.prompt_token_ids =
      multimodal::ApplyPromptReplacements(prompt_ids, updates, &applied);
  out.mm_features.reserve(applied.size());
  for (const multimodal::AppliedPromptUpdate& span : applied) {
    multimodal::MultiModalFeatureSpec spec;
    spec.modality = span.modality;
    spec.offset = span.offset;
    spec.length = span.length;
    if (span.modality == "image") {
      PreparedImage& item = images[static_cast<size_t>(span.item_index)];
      spec.mm_hash = std::move(item.mm_hash);
      spec.data =
          std::make_shared<multimodal::ImageKwargs>(std::move(item.kwargs));
    } else {
      PreparedAudio& item = audios[static_cast<size_t>(span.item_index)];
      spec.mm_hash = std::move(item.mm_hash);
      spec.audio_data =
          std::make_shared<multimodal::AudioKwargs>(std::move(item.kwargs));
    }
    out.mm_features.push_back(std::move(spec));
  }
  return out;
}

// THE CHAT FN, over both modalities (W7a, #2703) and over ANY NUMBER of items
// of each (W8a, #2860).
//
// `audio_proc` is NULL when this checkpoint carries no `audio_config`, and then
// this seam behaves exactly as it did before W7a: the supported-limit map does
// not declare "audio", so `ValidateChatMmLimits` refuses an audio part with
// upstream's own "At most 0 audio(s) may be provided in one prompt." and the
// marker builder is handed an empty audio marker it never reaches.
//
// UNTIL W8a THIS FUNCTION LOCATED ONE PART PER MODALITY and refused everything
// else at STEP 0, which was the honest ceiling while the two expanders could
// not be chained. It now collects EVERY part, prepares each one, and places
// them all in one pass, so the declared ceiling is upstream's own 512 / 128.
MultiModalChatFn MakeDots3NoteChatFn(
    std::shared_ptr<const multimodal::Dots3NoteImageProcessor> proc,
    std::shared_ptr<const multimodal::Dots3NoteAudioProcessor> audio_proc,
    const vllm::tok::Tokenizer& tokenizer, ChatPromptRenderFn prompt_fn,
    ImageCodecFn codec,
    std::shared_ptr<const multimodal::BaseProcessingInfo> info) {
  const std::string audio_marker =
      audio_proc != nullptr ? Dots3NoteAudioMarker(audio_proc->config())
                            : std::string();
  return [proc, audio_proc, info, audio_marker, &tokenizer,
          prompt_fn = std::move(prompt_fn),
          codec = std::move(codec)](const std::vector<ChatMessage>& messages)
             -> std::optional<multimodal::MultiModalInputs> {
    // STEP 0: the per-item limit check, BEFORE anything is decoded or dropped
    // (`chat_utils.py:662` validates as it tracks, for the same reason).
    ValidateChatMmLimits(*info, messages);

    // EVERY media part, in message and part order. Grouping by modality here
    // is upstream's own shape — `_get_prompt_updates` builds one rule per
    // modality carrying one item each (`common/processor.py:725-812` @
    // `9035151d6`) — and the STREAM order is recovered by the one-pass applier
    // in `RouteDots3NoteMultiModal`, never by this loop.
    std::vector<const ChatContentPart*> image_parts;
    std::vector<const ChatContentPart*> audio_parts;
    for (const ChatMessage& m : messages) {
      if (!m.content_parts.has_value()) continue;
      for (const ChatContentPart& part : *m.content_parts) {
        if (part.type == "image_url") {
          image_parts.push_back(&part);
        } else if (part.type == "input_audio" || part.type == "audio_url") {
          audio_parts.push_back(&part);
        }
      }
    }
    // The text path, untouched and byte-identical.
    if (image_parts.empty() && audio_parts.empty()) return std::nullopt;

    // NO MIXED-REQUEST REFUSAL ANY MORE (W8a, #2860). It used to throw
    // `InputValidationError` here naming "BOTH an image and an audio part",
    // because the two expanders each rebuilt the whole id vector and running
    // them in sequence would have measured the second one's offsets against
    // the first one's un-expanded input. `ApplyPromptReplacements` below is
    // upstream's own one pass over every modality at once
    // (`processing/processor.py:944-957` @ `9035151d6`), so the request is
    // SERVED and the two spans are disjoint by construction.
    if (!audio_parts.empty() && audio_proc == nullptr) {
      // Unreachable while the limit map and this pointer agree; kept because
      // reaching it would otherwise be a null dereference rather than an
      // answer.
      throw std::runtime_error(
          "dots3-note multimodal chat seam: an audio part reached the route on "
          "an install with no audio processor. The supported-limit map and the "
          "processor must be built from the same `audio_config`.");
    }

    // 1. Inject dots3-note's OWN markers at each part's position and render the
    //    templated prompt.
    std::vector<ChatMessage> rendered = messages;
    for (ChatMessage& m : rendered) {
      if (m.content_parts.has_value()) {
        m.content = BuildDots3NoteMarkerContent(m, audio_marker);
        m.content_parts.reset();
      }
    }
    const std::string prompt =
        prompt_fn(rendered, /*add_generation_prompt=*/true, {},
                  nlohmann::ordered_json::object());

    // 2. Tokenize WITH special tokens: each injected `<|img|><|imgpad|>
    //    <|endofimg|>` or `<|audio_comp_start|><|audio_comp_pad|>
    //    <|audio_comp_end|>` becomes exactly THREE ids (added tokens match
    //    leftmost-longest), which is the TARGET each rule below matches.
    const std::vector<int32_t> prompt_ids =
        tokenizer.EncodeWithSpecialTokens(prompt);

    // 3. Decode and PREPARE every item, then place them all in ONE pass.
    std::vector<PreparedImage> images;
    images.reserve(image_parts.size());
    for (const ChatContentPart* part : image_parts) {
      const DecodedMedia media = DecodeImageUrlPart(*part);
      const DecodedImageRgb img = codec(media);
      images.push_back(PrepareDots3NoteImage(*proc, img.rgb.data(), img.height,
                                             img.width));
    }
    std::vector<PreparedAudio> audios;
    audios.reserve(audio_parts.size());
    for (const ChatContentPart* part : audio_parts) {
      const DecodedMedia media = part->type == "input_audio"
                                     ? DecodeInputAudioPart(*part)
                                     : DecodeDataUri(part->url);
      audios.push_back(PrepareDots3NoteAudio(*audio_proc, media));
    }
    return RouteDots3NoteMultiModal(
        proc->config(), audio_proc != nullptr ? &audio_proc->config() : nullptr,
        std::move(images), std::move(audios), prompt_ids);
  };
}

MultiModalChatSeam MakeDots3NoteChatSeam(const MultiModalChatContext& ctx) {
  if (ctx.tokenizer == nullptr || ctx.mm_config == nullptr || !ctx.prompt_fn ||
      !ctx.codec) {
    // Refuse by name rather than dereference. The install's catch turns this
    // into a REFUSING seam, which is an HTTP 400 naming the architecture — never
    // a silent text answer.
    throw std::runtime_error(
        "dots3-note multimodal chat seam: the install context is incomplete "
        "(tokenizer, multimodal config, chat-prompt renderer and image codec "
        "are all required)");
  }

  const std::string preprocessor_config_path =
      PathUtf8(NativeUtf8Path(ctx.model_dir) / "preprocessor_config.json");
  if (!fs::exists(NativeUtf8Path(preprocessor_config_path))) {
    throw std::runtime_error(
        "dots3-note multimodal chat seam: '" + preprocessor_config_path +
        "' is missing; the image processor's patch/merge geometry, its "
        "per-channel normalization and its pixel bounds are read from it");
  }

  // THE TOWER'S OWN REFUSAL, ASKED HERE AND NOT IN THE ENGINE. W6a shipped the
  // dense blocks and W6b (#2613) the pyramid MoE ones, so the RELEASED
  // `dots-studio/dots3-note-prev` — 17 of whose 42 blocks are routed — is
  // ACCEPTED here now. What still refuses is `use_bias` (#2616), the softmax
  // and top-k-below-2 router arms (#2615), the blockwise-FP8 tower (W9) and
  // video (W7).
  //
  // Asking at INSTALL is not a preference. `EncodeMmDots3NoteForCausalLM`
  // refuses too, but it runs inside the engine's busy loop: throwing there stops
  // `AsyncLLM` and turns every LATER request, TEXT ONES INCLUDED, into a 500.
  // That was measured on this row's served-request gate before this check
  // existed. Refusing here installs a REFUSING seam instead, which is upstream's
  // own shape for "this server does not accept images for this model": HTTP 400
  // naming the architecture and the reason, with the text path untouched. The
  // encoder's check stays as defence in depth, on the same polarity Qwen3-VL's
  // carries ("reaching this point is a defect").
  const HfConfig hf = LoadHfConfig(ctx.config_path);
  {
    const std::string why = Dots3NoteVisionRefusalFor(hf);
    if (!why.empty()) {
      throw std::runtime_error(
          "dots3-note multimodal chat seam: this checkpoint's vision tower is "
          "not ported — " + why +
          ". See .agents/specs/dots3-note.md §4.11 and §4.12, issues #2512 and "
          "#2613.");
    }
  }

  auto proc = std::make_shared<const multimodal::Dots3NoteImageProcessor>(
      multimodal::LoadDots3NoteProcessorConfig(
          preprocessor_config_path, ctx.config_path, ctx.served_model_name));

  // ── THE AUDIO PROCESSOR (W7a, #2703) ────────────────────────────────────
  //
  // A checkpoint with NO `audio_config` gets a null processor and a
  // ceiling that does not declare "audio", which is the state every dots3-note
  // checkpoint was in before this brick. That is not a refusal: upstream builds
  // no audio tower either (`nvidia/multimodal.py:119-126` @ `9035151d6`), so
  // there is nothing owed and nothing to name.
  //
  // A checkpoint WITH one whose arms are not ported REFUSES at INSTALL, for the
  // reason the vision refusal above records: throwing from inside `encode_mm`
  // stops `AsyncLLM` and 500s every LATER request, text ones included.
  std::shared_ptr<const multimodal::Dots3NoteAudioProcessor> audio_proc;
  multimodal::Dots3NoteAudioProcessorConfig audio_cfg =
      multimodal::LoadDots3NoteAudioProcessorConfig(ctx.config_path,
                                                    ctx.served_model_name);
  if (audio_cfg.present) {
    const std::string why = Dots3NoteAudioRefusalFor(hf);
    if (!why.empty()) {
      throw std::runtime_error(
          "dots3-note multimodal chat seam: this checkpoint's audio tower is "
          "not ported — " + why +
          ". See .agents/specs/dots3-note.md §4.14 and issue #2703.");
    }
    // The three marker ids, resolved FROM THE TOKENIZER BY STRING
    // (`common/processor.py:757-760` @ `9035151d6` reads `vocab[AUDIO_START]`
    // and friends). Doing it here rather than from `config.json` is what makes
    // "the marker string this seam injects encodes to this id" true by
    // construction: the object that resolves the id is the object that will
    // encode the prompt. It THROWS BY NAME when one does not resolve — the
    // released checkpoint's three are 151718 / 151719 / 151720 in
    // start / END / PAD order, so a default would be a guess that a shape check
    // could never catch.
    multimodal::Dots3NoteResolveAudioTokenIds(
        &audio_cfg, [&](const std::string& marker) -> int32_t {
          for (const vllm::tok::SpecialToken& t : ctx.tokenizer->AddedTokens()) {
            if (t.text == marker) return t.id;
          }
          return -1;
        });
    audio_proc = std::make_shared<const multimodal::Dots3NoteAudioProcessor>(
        std::move(audio_cfg));
  }

  // The engine's limits (`--limit-mm-per-prompt`, `--language-model-only`)
  // folded by min() against this seam's own ceiling. The MultiModalConfig is
  // held BY REFERENCE (`context.h:105`); the engine owns it and outlives the
  // seam.
  auto info = std::make_shared<const multimodal::BaseProcessingInfo>(
      *ctx.mm_config,
      Dots3NoteChatSupportedMmLimits(/*has_audio=*/audio_proc != nullptr));

  MultiModalChatSeam seam;
  seam.allowed_limits = info->AllowedMmLimits();
  seam.detail = "dots3-note processor from " + preprocessor_config_path +
                " (dense + pyramid MoE vision arm" +
                (audio_proc != nullptr ? ", `dots` audio tower)" : ")");
  seam.chat_fn = MakeDots3NoteChatFn(proc, audio_proc, *ctx.tokenizer,
                                     ctx.prompt_fn, ctx.codec, info);
  return seam;
}

}  // namespace

REGISTER_VLLM_MM_CHAT(dots3_note, "Dots3NoteForCausalLM", &MakeDots3NoteChatSeam)

}  // namespace vllm::entrypoints::openai
