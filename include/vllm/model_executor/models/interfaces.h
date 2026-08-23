// Ported from: vllm/model_executor/models/interfaces.py @ 5559679229bc
//              (_mark_tower_model:257-298, _tower_model_names:141)
//              + vllm/model_executor/models/utils.py
//                (StageMissingLayer:687-704, no_init_weights:740-778).
//
// Scope (ENG-MM-INPUT-PIPELINE wave L3, #607): the TOWER-SKIP decision, and
// nothing else. Upstream's `SupportsMultiModal` mixin carries far more than
// this; the rest of it has no analogue here yet and is deliberately absent
// rather than stubbed.
//
// WHAT UPSTREAM DOES, and why this file is one predicate. A multimodal model
// wraps each tower's construction in `_mark_tower_model`, which enters
// `no_init_weights(...)` — a `torch.device("meta")` context plus a module hook
// that replaces the child with `StageMissingLayer(stage_name, mod)` — exactly
// when
//
//     all(mm_config.get_limit_per_prompt(m) == 0 for m in modalities)   (:293)
//
// So the tower is still CONSTRUCTED (every `__init__` runs, every shape
// resolves) and simply never allocates or loads storage. Three properties of
// that construct are load-bearing and are why this is a mirror rather than a
// boolean:
//
//   * The skip follows from ZERO LIMITS, not from `--language-model-only`. The
//     flag is one route to zero (`multimodal.py:78-80,321-327`);
//     `--limit-mm-per-prompt '{"image":0,"video":0}'` is another, and it must
//     skip the tower too. Keying our skip on the flag would diverge on the
//     second route and would be a bespoke path upstream does not have.
//   * `all`, not `any`, over the tower's OWN modality set. The Qwen3-VL /
//     Qwen3.6 tower is marked `{"image", "video"}` (`qwen3_5.py:422,634`,
//     `qwen3_vl.py:1747`), so a lone `image: 0` does NOT skip it.
//   * `StageMissingLayer` keeps the real module out of the child registry
//     (`utils.py:693-695`) so the weight loader reports no missing keys, and
//     raises from `__call__` (`:700-701`) if anything reaches it. A skipped
//     tower here must likewise be invisible to the loader and refuse BY NAME
//     when called, never read empty buffers.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_INTERFACES_H_
#define VLLM_MODEL_EXECUTOR_MODELS_INTERFACES_H_

#include <initializer_list>
#include <string_view>

#include "vllm/config/multimodal.h"

namespace vllm {

// `stage_name` for the {"image", "video"} pair (interfaces.py:279-282). Named
// once so the loader that sets it and the gate that reads it cannot drift.
inline constexpr std::string_view kVisionTowerStageName = "vision_tower";

// interfaces.py:293 — whether a tower serving exactly `modalities` is
// constructed WITHOUT loading its weights.
//
// `mm_config == nullptr` means "no multimodal limits were configured for this
// load", which is every pre-L3 caller and every path that does not go through
// the engine's EngineParams. It returns false: load everything, byte-identical
// to the behaviour before this predicate existed.
//
// An EMPTY `modalities` also returns false, and that is the one place a literal
// transcription would be wrong. Python's `all(())` is vacuously TRUE, so a
// literal port would skip every tower marked with no modality at all. Upstream
// cannot reach that state — `_mark_tower_model` is only ever called with a
// non-empty set, and a `str` argument is normalised to a one-element set at
// `:276-277` — so the divergence is unobservable against upstream and closes a
// failure mode that is silent on ours.
bool SkipTowerForModalities(const MultiModalConfig* mm_config,
                            std::initializer_list<std::string_view> modalities);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_INTERFACES_H_
