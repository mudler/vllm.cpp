// Ported from: vllm/config/multimodal.py @ 5559679229bc
//
// Scope (ENG-MM-INPUT-PIPELINE wave L1, #607): the per-modality INPUT-COUNT
// limits and the accessor every consumer asks. This is the mechanism the
// `--limit-mm-per-prompt` and `--language-model-only` serve flags are sugar over,
// and it is deliberately ported BEFORE either flag: upstream's
// `--language-model-only` is not a "skip the encoder" boolean, it is
// "set every modality limit to 0" (:78-80), and everything else about it —
// the refusal, the tower skip, the encoder-cudagraph frame budget — falls out of
// the limits being zero. Porting the boolean first would build a bespoke path
// that does not exist upstream. See .agents/specs/multimodal-track.md §1.5.
//
// The limits are only half of it. A limit nothing enforces is not a limit, so
// the enforcement lands in the same wave, mirrored from where upstream keeps it:
// `vllm/multimodal/processing/context.h` (`AllowedMmLimits`,
// `ValidateNumItems`).
//
// Ported T0 fields: language_model_only (:78), limit_per_prompt (:81),
// enable_mm_embeds (:98).
//
// DEFERRED upstream fields, intentionally omitted — each is a separate knob with
// no bearing on the limits mechanism, and none is reachable from a flag we
// accept today: mm_processor_kwargs, mm_processor_cache_gb / _type,
// mm_shm_cache_max_object_size_mb, mm_encoder_tp_mode, mm_encoder_attn_backend /
// _attn_dtype / _fp8_scale_path, interleave_mm_strings, skip_mm_profiling,
// video_pruning_rate, media_io_kwargs, mm_tensor_ipc. The compute_hash() surface
// is omitted with them: we have no config-hash seam to feed.
//
// DEVIATIONS, recorded:
//   * `limit_per_prompt` upstream is `dict[str, BaseDummyOptions]` (:81) — each
//     entry carries a `count` PLUS dummy-profiling options (num_frames, width,
//     height). Only `.count` participates in get_limit_per_prompt (:335), and the
//     option half exists to size DUMMY inputs during memory profiling, which is a
//     surface we do not have. So the map is `modality -> count` here, which is
//     also upstream's own "legacy format (count only)" spelling (:87-88). When
//     the profiling options land they extend the mapped type, not the accessor.
#ifndef VLLM_CONFIG_MULTIMODAL_H_
#define VLLM_CONFIG_MULTIMODAL_H_

#include <map>
#include <string>
#include <vector>

namespace vllm {

// The default limit for a modality nobody named (multimodal.py:331-333).
inline constexpr int kDefaultLimitPerPrompt = 999;

// MultiModalConfig (L1 limits subset). Value type; the model config owns one.
struct MultiModalConfig {
  // language_model_only (multimodal.py:78-80). Upstream's docstring is the
  // authority on what it means, and it is not what the name suggests: "If True,
  // disables all multimodal inputs by setting all modality limits to 0.
  // Equivalent to setting `--limit-mm-per-prompt` to 0 for every modality."
  bool language_model_only = false;

  // limit_per_prompt (multimodal.py:81-97): the maximum number of input items
  // allowed per prompt, per modality. Absent modality == the 999 default, NOT 0
  // — an empty map is "no limits configured", not "nothing allowed".
  std::map<std::string, int> limit_per_prompt;

  // enable_mm_embeds (multimodal.py:98-110). Off by default. It is the ESCAPE
  // from a zero limit for precomputed embeddings only: with it set, a modality at
  // limit 0 accepts `*_embeds` input while still refusing raw media, which is how
  // a deployment skips loading the encoder and still serves embeddings.
  // "Limits greater than 0 still apply to embeddings" (:103-105).
  bool enable_mm_embeds = false;

  // get_limit_per_prompt (multimodal.py:321-336). The single place every consumer
  // asks, and the precedence is load-bearing in this exact order:
  //
  //   1. language_model_only -> 0, for EVERY modality, checked BEFORE the map is
  //      read (:326-327). An explicit non-zero entry does not survive the flag.
  //   2. the map entry's count (:329,335).
  //   3. 999 for a modality the map does not mention (:331-333).
  //
  // Reading the map first would be indistinguishable on every configuration
  // except the one that matters: the flag set alongside explicit limits.
  int GetLimitPerPrompt(const std::string& modality) const {
    if (language_model_only) return 0;

    const auto it = limit_per_prompt.find(modality);
    if (it == limit_per_prompt.end()) {
      // Unspecified modality is set to 999 by default.
      return kDefaultLimitPerPrompt;
    }
    return it->second;
  }
};

// Parse the value of `--limit-mm-per-prompt` (arg_utils.py:556,1279) / the C-ABI
// `vllm_model_params.limit_mm_per_prompt` into `limit_per_prompt`.
//
// Ported from `MultiModalConfig._validate_limit_per_prompt`
// (multimodal.py:212-236) together with the DummyOptions dataclasses it feeds
// (:17-45). Upstream reaches this through argparse `type=parse_type(json.loads)`
// (_compute_kwargs, arg_utils.py:379-381 — the plain-dict branch; the
// `union_dict_and_str` branch immediately above it at :374-378 is a DIFFERENT
// rule and is not the one `limit_per_prompt: dict[str, BaseDummyOptions]` takes,
// because that annotation contains no `str` arm and `dict[...]` reports
// `__module__ == "builtins"`, so `is_not_builtin` is False), so the flag's value
// is a JSON OBJECT and every spelling below is the object's, not the flag's:
//
//   * the LEGACY format, count only — {"image": 16, "video": 2} (:87-88,220-222,
//     which rewrites a bare int to {"count": <int>});
//   * the CONFIGURABLE format — {"video": {"count": 1, "num_frames": 32,
//     "width": 512, "height": 512}} (:90-92,224-232);
//   * the two MIXED in one object (:94-96).
//
// Refusals, all of them upstream's own, none of them invented here:
//   * a non-object document, or a value that is neither an int nor an object —
//     upstream's json.loads/pydantic pair rejects both;
//   * `count` absent-and-not-an-int, or NEGATIVE — `count: int = Field(999,
//     ge=0)` (:21), declared on BaseDummyOptions and therefore checked for EVERY
//     modality;
//   * for a BUILTIN modality only — video, image, audio, the three
//     `_validate_limit_per_prompt` routes to a dataclass of their own
//     (:226-232) — an option key the modality does not define, or an option
//     <= 0: those three are the dataclasses carrying
//     `config=ConfigDict(extra="forbid")` (:24,33,41) over `Field(None, gt=0)`
//     (:28-30,37-38,45). video takes num_frames/width/height, image
//     width/height, audio length.
//
// And the one that is deliberately NOT a refusal: any OTHER modality falls to
// the `else` at :233 and is built as a bare `BaseDummyOptions`, which is the ONE
// dummy-options dataclass declared WITHOUT `extra="forbid"` (:17-21). Pydantic's
// default `extra='ignore'` therefore applies, so an unknown option key on a
// non-builtin modality is DROPPED, not refused, and its value is never
// validated: upstream accepts `{"pointcloud": {"count": 2, "foo": 3}}` and
// yields `BaseDummyOptions(count=2)`. We mirror that — refusing it would refuse
// a document the reference accepts. Only `count` is validated there.
//
// Refusing rather than defaulting is the point wherever upstream refuses: a
// typo'd limit that silently became 999 is a limit that is not there, which is
// exactly the failure this wave exists to remove.
//
// DEVIATION, recorded: the option keys are parsed and VALIDATED, then DROPPED.
// They size DUMMY multimodal inputs during memory profiling, a surface we do not
// have (see the header comment above on `limit_per_prompt`'s mapped type), and
// only `.count` participates in get_limit_per_prompt (:335). `ignored_options`,
// when non-null, collects "<modality>.<key>" for every option dropped — both the
// builtin options validated above and the non-builtin extras upstream itself
// discards — so the caller can ANNOUNCE the drop rather than let a user infer
// that `num_frames` took effect.
//
// Throws std::invalid_argument, carrying the offending key/value, on anything
// above.
std::map<std::string, int> ParseLimitMmPerPromptJson(
    const std::string& json, std::vector<std::string>* ignored_options);

}  // namespace vllm

#endif  // VLLM_CONFIG_MULTIMODAL_H_
