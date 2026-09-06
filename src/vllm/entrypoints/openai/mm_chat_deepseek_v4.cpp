// The `DeepseekV4ForCausalLM` multimodal chat seam (row
// `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue #2411, W5).
//
// Ported from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp at revision
// 86f746b36186f0e567729a5c06a8c918caba82a9:
//   encoding/encoding_dsv4.py::encode_messages  -> EncodeDeepSeekV4Messages
//   inference/image_processor.py::load_image    -> the codec + ProcessImage
//   inference/image_processor.py::prepare_vl_inputs -> PrepareDeepSeekV4Inputs
//
// W4 made an image reach `ModelRegistry::Forward`; this makes a USER able to
// send one. It is one translation unit with one `REGISTER_VLLM_MM_CHAT` line
// and zero edits to a shared array, exactly as `mm_chat_qwen3vl.cpp` and
// `mm_chat_dots3note.cpp` are.
//
// ── WHY THIS SEAM RENDERS ITS OWN PROMPT ────────────────────────────────────
//
// The other two seams inject a marker STRING at each mm part's position and
// hand the messages to `ctx.prompt_fn`, the server's Jinja chat template. This
// one calls `EncodeDeepSeekV4Messages` instead, which is the pinned upstream
// `encode_messages` ported whole.
//
// That is not a preference. `encoding_dsv4.py` is where this model's prompt is
// DEFINED, and the image handling is inseparable from the rest of it: the
// placeholder replaces the content block IN PLACE inside
// `process_image_messages`, a text block that already contains the placeholder
// is REFUSED there, tool results are sorted and merged around it, and the
// thinking-mode elision decides which turns survive to carry it. A Jinja
// template rendering OpenAI content parts expresses none of that, and a marker
// injected around it would put the placeholder at a position the pinned encoder
// does not put it.
//
// WHAT IT COSTS, stated because a user can see it: on this architecture the
// multimodal chat path ignores `--chat-template` and the GGUF
// `tokenizer.chat_template`, while the TEXT path still renders through them, so
// the two can disagree about a conversation carrying both kinds of turn. The
// row's spec records that under `## Owed`.
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/multimodal/deepseek_v4_processor.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/input_processor.h"  // InputValidationError -> HTTP 400

namespace vllm::entrypoints::openai {
namespace {

using json = nlohmann::ordered_json;

// The per-modality ceiling this seam SUPPORTS, which is the other operand of
// `MultiModalConfig`'s `min()` fold (context.py:392-405).
//
// IMAGE IS UNLIMITED, spelled `nullopt`, and that is a measured statement about
// this implementation rather than optimism. The pinned encoder emits one
// placeholder per image content block with no cap
// (`encoding_dsv4.py::process_image_blocks`), and `PrepareDeepSeekV4Inputs`
// walks every placeholder in the expanded prompt and emits one feature for each
// with its own offset, length and key. So the number a user may send comes from
// `--limit-mm-per-prompt` and from nothing this file writes down; the spec's
// "no lower hard-coded one-image ceiling is added" is that sentence.
//
// Every other modality is ABSENT, which context.py:414-415 reads as "not
// supported", limit 0. That is not a policy choice either: DeepSeek-V4-Flash-
// Vision is an image-only model, and `EncodeMmDeepseekV4ForCausalLM` refuses
// any other modality by name.
std::map<std::string, std::optional<int>> DeepSeekV4ChatSupportedMmLimits() {
  return {{"image", std::nullopt}};
}

// The pinned processor geometry for this family
// (`inference/image_processor.py`, and `conversion/deepseek.py`'s asserted
// `vision_max_n_token == 384` / `vision_max_wh_ratio == 8`).
//
// It is NOT read from `mmproj-BF16.gguf` yet, and the four `clip.*` keys that
// artifact carries -- `clip.vision.image_size`, `image_mean`, `image_std` and
// `image_min_pixels` -- are read by nothing in this tree. That is the
// preprocessor-contract residual the row's spec records under `## Owed`; the
// values below are the shipped artifact's own, so the two agree today, and the
// gap is that nothing MAKES them agree.
multimodal::DeepSeekV4ProcessorConfig ProcessorConfigFor(
    const MultiModalChatContext& ctx) {
  multimodal::DeepSeekV4ProcessorConfig cfg;
  cfg.vocab_size = static_cast<int32_t>(ctx.config->vocab_size);
  cfg.model_id = ctx.served_model_name;
  return cfg;
}

// The placeholder's id, resolved FROM THE TOKENIZER BY STRING.
//
// Doing it here rather than from a config number is what makes "the marker the
// encoder emits is the id the expansion counts" true by construction: the
// object that resolves the id is the object that will encode the prompt. It
// THROWS BY NAME when the string does not resolve, because a default would be a
// guess no shape check could ever catch -- `PrepareDeepSeekV4Inputs` would find
// zero placeholders and refuse with an image-count mismatch that names the
// wrong thing.
int32_t ResolveImageTokenId(const vllm::tok::Tokenizer& tokenizer) {
  for (const vllm::tok::SpecialToken& t : tokenizer.AddedTokens()) {
    if (t.text == multimodal::kDeepSeekV4ImagePlaceholder) return t.id;
  }
  throw std::runtime_error(
      std::string("DeepSeek-V4 multimodal chat seam: this checkpoint's "
                  "tokenizer has no added token '") +
      multimodal::kDeepSeekV4ImagePlaceholder +
      "'. The prompt encoder writes that string at every image position and "
      "the expansion counts the id it resolves to, so without it no image "
      "could be placed");
}

// One chat message as the OpenAI-shaped JSON the pinned encoder consumes.
//
// `content_parts` becomes a `content` ARRAY, which is the form
// `process_image_messages` reads: it moves the array to `content_blocks`,
// replaces each image block with the placeholder IN PLACE, and rebuilds the
// joined text. A bare-string message stays a bare string, so a text-only
// conversation reaches the encoder byte-identically to one this seam never
// touched.
json MessageToJson(const ChatMessage& m) {
  json out = json::object();
  out["role"] = m.role;
  if (!m.content_parts.has_value()) {
    out["content"] = m.content.value_or(std::string());
    return out;
  }
  json blocks = json::array();
  for (const ChatContentPart& part : *m.content_parts) {
    if (part.type == "text") {
      blocks.push_back(json{{"type", "text"}, {"text", part.text}});
      continue;
    }
    if (part.type == "image_url") {
      blocks.push_back(json{{"type", "image_url"},
                            {"image_url", json{{"url", part.url}}}});
      continue;
    }
    // Anything else has already been refused by `ValidateChatMmLimits` above
    // (this seam declares no other modality), so reaching here is a defect.
    // Carry the type through rather than dropping it, so the encoder's own
    // "[Unsupported <type>]" says which one.
    blocks.push_back(json{{"type", part.type}});
  }
  out["content"] = std::move(blocks);
  return out;
}

// THE CHAT FN.
MultiModalChatFn MakeDeepSeekV4ChatFn(
    std::shared_ptr<const multimodal::DeepSeekV4ImageProcessor> proc,
    const vllm::tok::Tokenizer& tokenizer, int32_t image_token_id,
    ImageCodecFn codec,
    std::shared_ptr<const multimodal::BaseProcessingInfo> info) {
  return [proc, info, image_token_id, &tokenizer, codec = std::move(codec)](
             const std::vector<ChatMessage>& messages)
             -> std::optional<multimodal::MultiModalInputs> {
    // STEP 0: the per-item limit check, BEFORE anything is decoded or dropped
    // (`chat_utils.py:662` validates as it tracks, for the same reason:
    // refusing costs nothing and truncating is invisible).
    ValidateChatMmLimits(*info, messages);

    // The image parts, in message and part order. This is only a PRESENCE
    // check and an ordered list of bytes to decode -- the POSITION of each
    // placeholder in the prompt is the pinned encoder's answer, recovered
    // below, never this loop's.
    std::vector<const ChatContentPart*> image_parts;
    for (const ChatMessage& m : messages) {
      if (!m.content_parts.has_value()) continue;
      for (const ChatContentPart& part : *m.content_parts) {
        if (part.type == "image_url") image_parts.push_back(&part);
      }
    }
    // The text path, untouched and byte-identical.
    if (image_parts.empty()) return std::nullopt;

    // 1. THE PINNED ENCODER. It replaces each image content block with
    //    `<|deepseek_image|>` in source order and renders the rest of the
    //    DeepSeek chat template around it.
    json encoder_messages = json::array();
    for (const ChatMessage& m : messages) {
      encoder_messages.push_back(MessageToJson(m));
    }
    multimodal::DeepSeekV4EncodedPrompt encoded;
    try {
      encoded = multimodal::EncodeDeepSeekV4Messages(encoder_messages, "chat");
    } catch (const std::invalid_argument& e) {
      // The encoder's refusals are all statements about the REQUEST -- a role
      // it does not know, an image block with no source, a text block that
      // already carries the placeholder. Upstream maps a ValueError to
      // BadRequestError (error_response.py:48-52); without this the generic
      // handler in api_server.cpp answers 500 and blames the server for the
      // client's body.
      throw vllm::v1::InputValidationError(
          std::string("DeepSeek-V4 chat encoding: ") + e.what());
    }
    if (encoded.images.size() != image_parts.size()) {
      // Unreachable while the two walks agree; kept because a disagreement
      // would otherwise pair image N's bytes with image M's placeholder, which
      // is a wrong answer rather than an error.
      throw std::runtime_error(
          "DeepSeek-V4 multimodal chat seam: the prompt encoder placed " +
          std::to_string(encoded.images.size()) +
          " image placeholders and the request carries " +
          std::to_string(image_parts.size()) +
          " image parts. The two walks must see the same blocks in the same "
          "order");
    }

    // 2. Tokenize WITH special tokens: each `<|deepseek_image|>` the encoder
    //    wrote becomes exactly ONE `image_token_id` (added tokens matched
    //    leftmost-longest), which is the target the expansion below replaces.
    const std::vector<int32_t> prompt_ids =
        tokenizer.EncodeWithSpecialTokens(encoded.prompt);

    // 3. Decode and preprocess every image IN SOURCE ORDER. The pinned
    //    encoder's own image records are what is walked, not `image_parts`, so
    //    an image nested inside a `tool_result` block -- which the encoder
    //    reaches and this file's flat loop does not -- is decoded in the
    //    position the encoder gave it.
    std::vector<multimodal::DeepSeekV4ImageItem> images;
    images.reserve(encoded.images.size());
    for (const json& record : encoded.images) {
      const auto url = record.find("url");
      if (url == record.end() || !url->is_string() ||
          url->get_ref<const std::string&>().empty()) {
        throw vllm::v1::InputValidationError(
            "DeepSeek-V4 chat image: only an `image_url` block carrying a "
            "`url` is served; a `source` or `data` block is a named residual "
            "of row MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm (#2411)");
      }
      DecodedImageRgb rgb;
      try {
        DecodedMedia media = DecodeDataUri(url->get_ref<const std::string&>());
        rgb = codec(media);
      } catch (const std::exception& e) {
        // THE CODEC AND THE URI SCHEME ARE NAMED RESIDUALS, and this is where a
        // user meets them. `DefaultImageCodec` decodes raw RGB only and
        // `DecodeDataUri` takes `data:` only, so a `data:image/png;base64,...`
        // or an `https://` image is refused -- but both refuse with
        // `std::runtime_error`, which `api_server.cpp:373` maps to HTTP 500
        // "InternalServerError". A request this server cannot serve is a
        // CLIENT error and has to read like one, so it is re-thrown as the type
        // that maps to 400 with the residual's own message intact.
        //
        // Vendoring a PNG/JPEG decoder is NOT done here: the codec is the
        // LIBRARY's and is shared by three architectures, so implementing it
        // inside a model row would land a cross-model capability under a model
        // row. The arm is recorded as owed instead, which is what
        // AGENTS.md "Shared seams" asks for.
        throw vllm::v1::InputValidationError(
            std::string("DeepSeek-V4 chat image: ") + e.what());
      }
      multimodal::DeepSeekV4ImageItem item;
      item.kwargs = std::make_shared<multimodal::ImageKwargs>(
          proc->ProcessImage(
              std::span<const uint8_t>(rgb.rgb.data(), rgb.rgb.size()),
              rgb.height, rgb.width));
      item.content_hash = proc->HashImage(
          std::span<const uint8_t>(rgb.rgb.data(), rgb.rgb.size()), rgb.height,
          rgb.width);
      images.push_back(std::move(item));
    }

    // 4. EXPAND. Each placeholder becomes its own image block of
    //    `vocab_size + type` sentinel identifiers, and each block gets a
    //    feature carrying its offset, its length and the key the scheduler and
    //    both encoder caches are keyed on.
    try {
      return multimodal::PrepareDeepSeekV4Inputs(prompt_ids, image_token_id,
                                                 images, proc->config());
    } catch (const std::invalid_argument& e) {
      throw vllm::v1::InputValidationError(
          std::string("DeepSeek-V4 chat image: ") + e.what());
    }
  };
}

MultiModalChatSeam MakeDeepSeekV4ChatSeam(const MultiModalChatContext& ctx) {
  if (ctx.tokenizer == nullptr || ctx.mm_config == nullptr ||
      ctx.config == nullptr || !ctx.codec) {
    // Refuse by name rather than dereference. The install's catch turns this
    // into a REFUSING seam, which is an HTTP 400 naming the architecture --
    // never a silent text answer.
    throw std::runtime_error(
        "DeepSeek-V4 multimodal chat seam: the install context is incomplete "
        "(tokenizer, multimodal config, resolved model config and image codec "
        "are all required)");
  }
  // THE SECOND FILE, asked for at INSTALL. `DeepseekV4ForCausalLM` names both
  // the text checkpoint and the Flash-Vision one, so the architecture cannot
  // say whether a tower is present and `--mmproj` is the only thing that can.
  // Without it `DeepseekV4LoadedModel::vision_tower` refuses inside
  // `encode_mm`, which runs in the engine's busy loop: that stops `AsyncLLM`
  // and 500s every later request, text ones included.
  if (ctx.mmproj_path.empty()) {
    throw std::runtime_error(
        "DeepSeek-V4 multimodal chat seam: this engine was loaded without "
        "--mmproj, so it carries no `deepseek4v` vision projector and cannot "
        "answer an image request. Pass the repository's `mmproj-BF16.gguf` "
        "beside the language shards. (A safetensors checkpoint has no --mmproj "
        "arm at all: materialising its own `vision.*` and `aligner.*` tensors "
        "is owed by issue #2411 and row "
        "MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm.)");
  }
  if (ctx.config->vocab_size <= 0) {
    throw std::runtime_error(
        "DeepSeek-V4 multimodal chat seam: the resolved model config reports "
        "vocab_size " + std::to_string(ctx.config->vocab_size) +
        ". Every image position is spelled `vocab_size + type`, so a wrong one "
        "puts the sentinel identifiers inside the vocabulary and the merge "
        "would splice image rows over real tokens");
  }

  const int32_t image_token_id = ResolveImageTokenId(*ctx.tokenizer);
  auto proc = std::make_shared<const multimodal::DeepSeekV4ImageProcessor>(
      ProcessorConfigFor(ctx));

  // The engine's limits (`--limit-mm-per-prompt`, `--language-model-only`)
  // folded by min() against this seam's own ceiling. The MultiModalConfig is
  // held BY REFERENCE (`context.h:105`); the engine owns it and outlives the
  // seam.
  auto info = std::make_shared<const multimodal::BaseProcessingInfo>(
      *ctx.mm_config, DeepSeekV4ChatSupportedMmLimits());

  MultiModalChatSeam seam;
  seam.allowed_limits = info->AllowedMmLimits();
  seam.detail =
      "DeepSeek-V4 Flash-Vision processor (pinned encode_messages prompt, "
      "`deepseek4v` projector from " + ctx.mmproj_path + ")";
  seam.chat_fn = MakeDeepSeekV4ChatFn(proc, *ctx.tokenizer, image_token_id,
                                      ctx.codec, info);
  return seam;
}

}  // namespace

REGISTER_VLLM_MM_CHAT(deepseek_v4, "DeepseekV4ForCausalLM",
                      &MakeDeepSeekV4ChatSeam)

}  // namespace vllm::entrypoints::openai
