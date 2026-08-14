// ENG-MM-INPUT-PIPELINE wave L1 (#607) — the per-modality input-limit accessor.
//
// Ported from (test-porting protocol .agents/porting.md), all @ 5559679229bc:
//   * vllm/config/multimodal.py:78 — `language_model_only: bool = False`, whose
//     docstring is the authority for what the flag MEANS: "disables all
//     multimodal inputs by setting all modality limits to 0. Equivalent to
//     setting `--limit-mm-per-prompt` to 0 for every modality."
//   * vllm/config/multimodal.py:81 — `limit_per_prompt`, default empty.
//   * vllm/config/multimodal.py:98 — `enable_mm_embeds: bool = False`.
//   * vllm/config/multimodal.py:321-336 — `get_limit_per_prompt(modality)`, the
//     precedence under test: `language_model_only` returns 0 FIRST (:326-327),
//     BEFORE the map is consulted; then the map entry's count (:335); then the
//     999 default for a modality the map does not mention (:331-333).
//
// Upstream's own `tests/config/test_multimodal_config.py` @ 5559679229bc covers
// the hash/backend/mm-prefix behaviour of MultiModalConfig but has NO case for
// the get_limit_per_prompt precedence — it is exercised only indirectly, through
// e2e suites that need a GPU and a llava checkpoint
// (tests/entrypoints/multimodal/llm/test_mm_embeds_only.py,
// tests/multimodal/test_processing.py::test_limit_mm_per_prompt_apply, both of
// which this port covers in tests/vllm/multimodal/test_processing_limits.cpp).
// These cases are therefore AUTHORED here, not ported, exactly as
// tests/vllm/config/test_speculative_dspark.cpp records for the DSpark
// resolution: they pin the pure-config behaviour those e2e suites depend on.
// SKIPPED from upstream's file, with reasons: the compute_hash cases
// (:17-48,64-75) — we have no config-hash surface; the mm_encoder_attn_backend /
// media_io_kwargs / fp8-scale cases (:17-31,51-62) — those fields are NOT in the
// L1 scope (§1.5 ports the limits mechanism, not the encoder-attention knobs);
// the _supports_multimodal_for_mm_prefix cases (:78-166) — they exercise
// ModelConfig + the multimodal REGISTRY, which L1 does not touch.
//
// RED-first note: before wave L1 there is no `vllm/config/multimodal.h` at all,
// so this file does not compile — `MultiModalConfig` is undeclared. Every case
// below is red for that reason. Once the header exists, the precedence case
// "language_model_only zeroes a modality the map raises" is the one that reds if
// the map is consulted before the flag (the mutation run in the L1 report).
#include <doctest/doctest.h>

#include <string>

#include "vllm/config/multimodal.h"

using vllm::MultiModalConfig;

TEST_CASE("MultiModalConfig defaults mirror upstream") {
  const MultiModalConfig config;
  // multimodal.py:78, :81, :98.
  CHECK(config.language_model_only == false);
  CHECK(config.limit_per_prompt.empty());
  CHECK(config.enable_mm_embeds == false);
}

TEST_CASE("an unspecified modality defaults to 999") {
  // multimodal.py:331-333 — "Unspecified modality is set to 999 by default".
  const MultiModalConfig config;
  CHECK(config.GetLimitPerPrompt("image") == 999);
  CHECK(config.GetLimitPerPrompt("video") == 999);
  CHECK(config.GetLimitPerPrompt("audio") == 999);
  // A modality nobody has heard of is still 999: the map miss is what decides,
  // not a fixed list of known modalities.
  CHECK(config.GetLimitPerPrompt("hologram") == 999);
}

TEST_CASE("an explicit map entry is honoured") {
  // multimodal.py:334-335 — `return limit_data.count`.
  MultiModalConfig config;
  config.limit_per_prompt["image"] = 16;
  config.limit_per_prompt["video"] = 2;
  CHECK(config.GetLimitPerPrompt("image") == 16);
  CHECK(config.GetLimitPerPrompt("video") == 2);
  // Unmentioned modalities keep the default — an entry for one modality never
  // implies a limit for another.
  CHECK(config.GetLimitPerPrompt("audio") == 999);
}

TEST_CASE("an explicit zero is a real limit without the flag") {
  MultiModalConfig config;
  config.limit_per_prompt["image"] = 0;
  CHECK(config.GetLimitPerPrompt("image") == 0);
  CHECK(config.GetLimitPerPrompt("video") == 999);
}

TEST_CASE("language_model_only zeroes EVERY modality, mapped or not") {
  // multimodal.py:326-327 — the docstring's "setting all modality limits to 0"
  // is literally what the accessor does, and it applies to modalities the map
  // never mentions.
  MultiModalConfig config;
  config.language_model_only = true;
  CHECK(config.GetLimitPerPrompt("image") == 0);
  CHECK(config.GetLimitPerPrompt("video") == 0);
  CHECK(config.GetLimitPerPrompt("audio") == 0);
  CHECK(config.GetLimitPerPrompt("hologram") == 0);
}

TEST_CASE("language_model_only zeroes a modality the map raises") {
  // THE PRECEDENCE CASE. multimodal.py:326-327 returns 0 BEFORE :329 reads the
  // map, so an explicit non-zero entry does NOT survive the flag. Consulting the
  // map first would return 16 here and still pass every other case in this file.
  MultiModalConfig config;
  config.language_model_only = true;
  config.limit_per_prompt["image"] = 16;
  config.limit_per_prompt["video"] = 2;
  CHECK(config.GetLimitPerPrompt("image") == 0);
  CHECK(config.GetLimitPerPrompt("video") == 0);
}

TEST_CASE("language_model_only is equivalent to zeroing every modality") {
  // The docstring's own equivalence (multimodal.py:78-80): the flag and an
  // all-zero map must be indistinguishable through the accessor.
  MultiModalConfig by_flag;
  by_flag.language_model_only = true;

  MultiModalConfig by_map;
  for (const char* modality : {"image", "video", "audio"})
    by_map.limit_per_prompt[modality] = 0;

  for (const char* modality : {"image", "video", "audio"})
    CHECK(by_flag.GetLimitPerPrompt(modality) ==
          by_map.GetLimitPerPrompt(modality));
}
