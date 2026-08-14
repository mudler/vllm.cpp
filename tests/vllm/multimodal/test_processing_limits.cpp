// ENG-MM-INPUT-PIPELINE wave L1 (#607) — the REFUSAL that makes a limit a limit.
//
// A limit nothing enforces is not a limit: `--language-model-only` zeroes every
// modality limit, and its main user-visible effect upstream is that the server
// then answers an image request with "At most 0 image(s) may be provided in one
// prompt." (multimodal-track.md §1.5). These cases pin that refusal.
//
// Ported from (test-porting protocol .agents/porting.md), all @ 5559679229bc:
//   * tests/multimodal/test_processing.py:902-941
//     `test_limit_mm_per_prompt_apply` — the 7 (num_images, limit, is_valid)
//     triples, refusal matched on "At most". Its model is
//     llava-hf/llava-v1.6-mistral-7b-hf, whose get_supported_mm_limits is
//     {"image": None} (vllm/model_executor/models/llava.py:183-184), i.e.
//     UNLIMITED — mirrored here as a nullopt supported limit. Harness
//     adaptation, unavoidable: upstream drives the check through
//     `processor(...)` on a real checkpoint; we call the ported
//     `ValidateNumItems` directly, which is the function upstream's call reaches
//     at processing/context.py:461. No checkpoint, no image decode.
//   * tests/multimodal/test_processing.py:944-985
//     `test_budget_caps_prevent_dummy_input_validation_failure` — all 10
//     (user_limit, supported_limit) pairs, and all three of its assertions
//     (allowed <= supported, allowed <= user, allowed == min(user, supported)).
//   * tests/entrypoints/multimodal/llm/test_mm_embeds_only.py:41-49
//     `test_raw_image_rejected` — the EXACT message at limit 0, which upstream
//     matches as the regex r"At most 0 image\(s\)". Ported as an exact-string
//     check; upstream's other two cases in that file (:29-38 embeddings still
//     generate, :52-60 text-only still works) need a GPU + llava and are SKIPPED,
//     with the escape they depend on covered below as a unit case instead.
//   * tests/entrypoints/unit_tests/test_chat_utils.py:1498-1530,1534-1560
//     `test_parse_chat_messages_rejects_too_many_images_{in_one_message,
//     across_messages}` — three images against phi3v's limit, matched on
//     "At most". Harness adaptation: upstream drives `parse_chat_messages`; we
//     call the ported chat-path validator, which is what
//     entrypoints/chat_utils.py:662 reaches. The "across messages" half is the
//     same assertion with the count accumulated by the caller's tracker, so it
//     is ported as the running-count case rather than a second message shape.
//
// The behaviour under test, mirrored 1:1:
//   * vllm/multimodal/processing/context.py:392-405 `allowed_mm_limits` — folds
//     the USER limit with the model's own supported limit by min(); a user limit
//     NEVER raises a model's ceiling.
//   * vllm/multimodal/processing/context.py:409-428 `validate_num_items` — the
//     message at :423 and the conditional hint at :425-426.
//   * vllm/multimodal/processing/context.py:441-461 — parse_mm_data's validate
//     block, including the `enable_mm_embeds` escape at :443-460.
//   * vllm/entrypoints/chat_utils.py:648-662 — the per-tracked-item call site
//     and its own spelling of the same escape at :653-660.
//
// AUTHORED, not ported (upstream has no unit case for either): the exact text of
// the " Set `--limit-mm-per-prompt` to increase this limit." hint and the
// conditions under which it does and does not appear (context.py:425-426), and
// the "You must set `--enable-mm-embeds`" refusal text (context.py:445-448).
// Both are message contracts a user reads, so they are pinned rather than left
// to drift.
//
// RED-first note: before wave L1 there is no
// `vllm/multimodal/processing/context.h`, so this file does not compile —
// `BaseProcessingInfo` is undeclared, and every case is red for that reason.
// Once it exists, "a request at the limit passes and one over it is refused" is
// the case that reds if the throw is removed, and "the user limit never raises
// the model ceiling" is the case that reds if the min() fold is dropped (both
// mutations are run in the L1 report).
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vllm/config/multimodal.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/v1/engine/validation_error.h"

using vllm::MultiModalConfig;
using vllm::multimodal::BaseProcessingInfo;
using vllm::multimodal::ParsedMmItems;
using vllm::v1::InputValidationError;

namespace {

// The llava-v1.6-mistral-7b-hf stand-in upstream's test_limit_mm_per_prompt_apply
// uses: get_supported_mm_limits() == {"image": None} (llava.py:183-184).
std::map<std::string, std::optional<int>> UnlimitedImages() {
  return {{"image", std::nullopt}};
}

// Grab the message off the refusal so its exact text can be asserted.
std::string RefusalMessage(const BaseProcessingInfo& info,
                           const std::string& modality, int num_items) {
  try {
    info.ValidateNumItems(modality, num_items);
  } catch (const InputValidationError& e) {
    return e.what();
  }
  return "<no refusal>";
}

}  // namespace

// ---------------------------------------------------------------------------
// test_processing.py:902-941 — test_limit_mm_per_prompt_apply
// ---------------------------------------------------------------------------

TEST_CASE("a request at the limit passes and one over it is refused") {
  // The upstream parametrization, verbatim: (num_images, limit, is_valid).
  struct Case {
    int num_images;
    int limit;
    bool is_valid;
  };
  const Case cases[] = {
      {0, 0, true},  {0, 1, true},  {1, 0, false}, {1, 1, true},
      {1, 2, true},  {2, 1, false}, {2, 2, true},
  };

  for (const Case& c : cases) {
    CAPTURE(c.num_images);
    CAPTURE(c.limit);
    MultiModalConfig mm_config;
    mm_config.limit_per_prompt["image"] = c.limit;
    const BaseProcessingInfo info(mm_config, UnlimitedImages());

    if (c.is_valid)
      CHECK_NOTHROW(info.ValidateNumItems("image", c.num_images));
    else
      CHECK_THROWS_AS(info.ValidateNumItems("image", c.num_images),
                      InputValidationError);
  }
}

// ---------------------------------------------------------------------------
// test_processing.py:944-985 — test_budget_caps_prevent_dummy_input_validation_failure
// ---------------------------------------------------------------------------

TEST_CASE("the user limit never raises the model ceiling") {
  // The upstream parametrization, verbatim: (user_limit, supported_limit).
  struct Case {
    int user_limit;
    int supported_limit;
  };
  const Case cases[] = {
      {0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 2},
      {2, 1}, {2, 2}, {5, 1}, {1, 5}, {10, 0},
  };

  for (const Case& c : cases) {
    CAPTURE(c.user_limit);
    CAPTURE(c.supported_limit);
    MultiModalConfig mm_config;
    mm_config.limit_per_prompt["image"] = c.user_limit;
    const BaseProcessingInfo info(mm_config, {{"image", c.supported_limit}});

    const std::map<std::string, int> allowed = info.AllowedMmLimits();
    REQUIRE(allowed.count("image") == 1);
    // All three upstream assertions.
    CHECK(allowed.at("image") <= c.supported_limit);
    CHECK(allowed.at("image") <= c.user_limit);
    CHECK(allowed.at("image") ==
          (c.user_limit < c.supported_limit ? c.user_limit : c.supported_limit));
  }
}

TEST_CASE("an unlimited model limit leaves the user limit untouched") {
  // context.py:401-404 — supported_limit None means the user limit stands as-is
  // rather than being folded against anything.
  MultiModalConfig mm_config;
  mm_config.limit_per_prompt["image"] = 7;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());
  CHECK(info.AllowedMmLimits().at("image") == 7);
}

TEST_CASE("allowed limits cover exactly the modalities the model supports") {
  // context.py:396-397 iterates supported_mm_limits, so a modality the model
  // does not support is ABSENT from the result — not present with a 0.
  MultiModalConfig mm_config;
  mm_config.limit_per_prompt["audio"] = 4;
  const BaseProcessingInfo info(
      mm_config, {{"image", std::nullopt}, {"video", 2}});
  const std::map<std::string, int> allowed = info.AllowedMmLimits();
  CHECK(allowed.size() == 2);
  CHECK(allowed.count("audio") == 0);
  CHECK(allowed.at("image") == 999);  // unspecified -> the 999 default
  CHECK(allowed.at("video") == 2);
}

// ---------------------------------------------------------------------------
// The message contract (context.py:423,425-426) — and test_mm_embeds_only.py:45
// ---------------------------------------------------------------------------

TEST_CASE("language_model_only refuses with upstream's exact message") {
  // test_mm_embeds_only.py:45 matches r"At most 0 image\(s\)"; this is the whole
  // string. No hint: the model supports images without limit, so
  // supported_limit collapses to the allowed limit (context.py:417-418) and
  // num_items <= supported_limit can never hold at :425.
  MultiModalConfig mm_config;
  mm_config.language_model_only = true;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());
  CHECK(RefusalMessage(info, "image", 1) ==
        "At most 0 image(s) may be provided in one prompt.");
}

TEST_CASE("the hint appears only when raising the user limit would help") {
  // context.py:425-426. The model supports 5 images; the user allowed 1.
  MultiModalConfig mm_config;
  mm_config.limit_per_prompt["image"] = 1;
  const BaseProcessingInfo info(mm_config, {{"image", 5}});
  CHECK(RefusalMessage(info, "image", 2) ==
        "At most 1 image(s) may be provided in one prompt."
        " Set `--limit-mm-per-prompt` to increase this limit.");
}

TEST_CASE("no hint when the MODEL is the binding constraint") {
  // The user asked for 5 but the model supports 1: raising the flag would not
  // help, so upstream withholds the hint (num_items > supported_limit at :425).
  MultiModalConfig mm_config;
  mm_config.limit_per_prompt["image"] = 5;
  const BaseProcessingInfo info(mm_config, {{"image", 1}});
  CHECK(RefusalMessage(info, "image", 2) ==
        "At most 1 image(s) may be provided in one prompt.");
}

TEST_CASE("an unsupported modality is refused at 0 with no hint") {
  // context.py:414-415 — both lookups miss, so both default to 0. The model
  // cannot take audio at all, so the flag would not help either.
  MultiModalConfig mm_config;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());
  CHECK(RefusalMessage(info, "audio", 1) ==
        "At most 0 audio(s) may be provided in one prompt.");
}

TEST_CASE("the modality name is echoed into the message") {
  MultiModalConfig mm_config;
  mm_config.language_model_only = true;
  const BaseProcessingInfo info(
      mm_config, {{"video", std::nullopt}, {"audio", std::nullopt}});
  CHECK(RefusalMessage(info, "video", 3) ==
        "At most 0 video(s) may be provided in one prompt.");
  CHECK(RefusalMessage(info, "audio", 3) ==
        "At most 0 audio(s) may be provided in one prompt.");
}

// ---------------------------------------------------------------------------
// Call site 1 — parse_mm_data's validate block (context.py:441-461)
// ---------------------------------------------------------------------------

TEST_CASE("parse_mm_data validates every parsed modality") {
  MultiModalConfig mm_config;
  mm_config.limit_per_prompt["image"] = 1;
  const BaseProcessingInfo info(
      mm_config, {{"image", std::nullopt}, {"video", std::nullopt}});

  CHECK_NOTHROW(info.ValidateParsedMmData(
      {ParsedMmItems{"image", 1, false}, ParsedMmItems{"video", 2, false}}));
  CHECK_THROWS_AS(info.ValidateParsedMmData({ParsedMmItems{"image", 1, false},
                                             ParsedMmItems{"video", 2, false},
                                             ParsedMmItems{"audio", 1, false}}),
                  InputValidationError);
}

TEST_CASE("embeddings need --enable-mm-embeds") {
  // context.py:443-448.
  MultiModalConfig mm_config;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());
  try {
    info.ValidateParsedMmData({ParsedMmItems{"image", 1, true}});
    FAIL("expected a refusal");
  } catch (const InputValidationError& e) {
    CHECK(std::string(e.what()) ==
          "You must set `--enable-mm-embeds` to input `image_embeds`");
  }
}

TEST_CASE("embeddings at limit 0 skip count validation") {
  // context.py:449-460 — THE escape. This is what makes an embeddings-only
  // deployment possible: the encoder never loads, the raw-image path is refused,
  // and precomputed embeddings still get through.
  MultiModalConfig mm_config;
  mm_config.enable_mm_embeds = true;
  mm_config.limit_per_prompt["image"] = 0;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());

  CHECK_NOTHROW(info.ValidateParsedMmData({ParsedMmItems{"image", 4, true}}));
  // The raw-image arm of the very same config is still refused
  // (test_mm_embeds_only.py:41-49).
  CHECK_THROWS_AS(info.ValidateParsedMmData({ParsedMmItems{"image", 1, false}}),
                  InputValidationError);
}

TEST_CASE("embeddings above a NON-zero limit are still counted") {
  // context.py:453 gates the skip on limit == 0. multimodal.py:103-105:
  // "Limits greater than 0 still apply to embeddings."
  MultiModalConfig mm_config;
  mm_config.enable_mm_embeds = true;
  mm_config.limit_per_prompt["image"] = 1;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());

  CHECK_NOTHROW(info.ValidateParsedMmData({ParsedMmItems{"image", 1, true}}));
  CHECK_THROWS_AS(info.ValidateParsedMmData({ParsedMmItems{"image", 2, true}}),
                  InputValidationError);
}

// ---------------------------------------------------------------------------
// Call site 2 — the chat tracker (chat_utils.py:648-662)
// ---------------------------------------------------------------------------

TEST_CASE("the chat path refuses too many images") {
  // test_chat_utils.py:1498-1530 — three image_url parts against a limit of 2.
  MultiModalConfig mm_config;
  mm_config.limit_per_prompt["image"] = 2;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());

  CHECK(info.ValidateTrackedChatItem("image", 1) == "image");
  CHECK(info.ValidateTrackedChatItem("image", 2) == "image");
  CHECK_THROWS_AS(info.ValidateTrackedChatItem("image", 3),
                  InputValidationError);
}

TEST_CASE("the chat path counts ACROSS messages, not within one") {
  // test_chat_utils.py:1534-1560 — the running count is the caller's; the
  // refusal is the same. Two messages of two images each trips a limit of 3 on
  // the third item.
  MultiModalConfig mm_config;
  mm_config.limit_per_prompt["image"] = 3;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());
  CHECK_NOTHROW(info.ValidateTrackedChatItem("image", 3));
  CHECK_THROWS_AS(info.ValidateTrackedChatItem("image", 4),
                  InputValidationError);
}

TEST_CASE("the chat path strips _embeds before consulting the limit") {
  // chat_utils.py:635 — `input_modality = modality.replace("_embeds", "")`, so
  // the image limit is what an `image_embeds` part is checked against, and it is
  // the STRIPPED name that reaches the message.
  MultiModalConfig mm_config;
  mm_config.limit_per_prompt["image"] = 1;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());

  CHECK(info.ValidateTrackedChatItem("image_embeds", 1) == "image");
  try {
    info.ValidateTrackedChatItem("image_embeds", 2);
    FAIL("expected a refusal");
  } catch (const InputValidationError& e) {
    CHECK(std::string(e.what()) ==
          "At most 1 image(s) may be provided in one prompt.");
  }
}

TEST_CASE("the chat path escape needs the flag, the zero AND the _embeds name") {
  // chat_utils.py:653-660 — all three conjuncts. Dropping any one of them must
  // put the item back under the limit.
  MultiModalConfig escaping;
  escaping.enable_mm_embeds = true;
  escaping.limit_per_prompt["image"] = 0;
  CHECK_NOTHROW(BaseProcessingInfo(escaping, UnlimitedImages())
                    .ValidateTrackedChatItem("image_embeds", 9));

  // ... without enable_mm_embeds.
  MultiModalConfig no_flag = escaping;
  no_flag.enable_mm_embeds = false;
  CHECK_THROWS_AS(BaseProcessingInfo(no_flag, UnlimitedImages())
                      .ValidateTrackedChatItem("image_embeds", 9),
                  InputValidationError);

  // ... at a non-zero limit.
  MultiModalConfig nonzero = escaping;
  nonzero.limit_per_prompt["image"] = 1;
  CHECK_THROWS_AS(BaseProcessingInfo(nonzero, UnlimitedImages())
                      .ValidateTrackedChatItem("image_embeds", 9),
                  InputValidationError);

  // ... for a raw image rather than an `_embeds` part.
  CHECK_THROWS_AS(BaseProcessingInfo(escaping, UnlimitedImages())
                      .ValidateTrackedChatItem("image", 1),
                  InputValidationError);
}

TEST_CASE("prompt_embeds is not a modality") {
  // chat_utils.py:632-633 — it returns before any limit is consulted, so it is
  // never refused however many there are.
  MultiModalConfig mm_config;
  mm_config.language_model_only = true;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());
  CHECK(info.ValidateTrackedChatItem("prompt_embeds", 99) == std::nullopt);
}

TEST_CASE("language_model_only refuses the chat path too") {
  // The flag's whole user-visible point, at the OpenAI call site.
  MultiModalConfig mm_config;
  mm_config.language_model_only = true;
  const BaseProcessingInfo info(mm_config, UnlimitedImages());
  try {
    info.ValidateTrackedChatItem("image", 1);
    FAIL("expected a refusal");
  } catch (const InputValidationError& e) {
    CHECK(std::string(e.what()) ==
          "At most 0 image(s) may be provided in one prompt.");
  }
}
