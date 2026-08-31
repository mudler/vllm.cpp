// GLM-5.3-Flash — the env-gated forward diagnostic.
//
// Issue [#2241](https://github.com/mudler/vllm.cpp/issues/2241), campaign issue
// [#1998](https://github.com/mudler/vllm.cpp/issues/1998), spec
// `.agents/specs/glm5-next-flash.md`.
//
// ─── WHY THIS EXISTS ────────────────────────────────────────────────────────
//
// The first end-to-end run of this model on a real checkpoint emitted token id
// 0 eight times. Four different defects produce exactly that byte sequence and
// they need different repairs: the logits are NaN (`NaN > x` is false, so a
// running argmax never leaves index 0), the logits are finite and UNIFORM (a
// dead head or an unloaded tensor), the logits are finite, varied and WRONG, or
// the forward is fine and the sampler is not. A summary statistic that says
// "the tensor is bad" separates none of them, so this prints VALUES: the
// non-finite counts, the range, the moments, and the actual top-k entries.
//
// ─── THE INSTRUMENT MUST SAY IT RAN ─────────────────────────────────────────
//
// A probe that silently does nothing reads as a clean result, which is the
// failure mode this file is most exposed to: every call site is behind
// `Level() > 0`, so an unset variable makes all of them vanish and the run
// looks exactly like a run with no defect found. `Banner()` therefore prints
// the RAW environment string and the parsed level before any measurement, and
// the job script greps for that line. No banner means no instrument, never a
// clean tensor.
//
// ─── AND IT MUST NOT CURE WHAT IT OBSERVES ──────────────────────────────────
//
// Every function here READS a buffer that already exists and writes only to
// stderr. Nothing allocates into the measured buffer, nothing synchronises a
// queue, nothing reorders a computation, and no call site is inside a
// conditional whose branch depends on a measurement. With the variable unset
// the cost is one relaxed read of a function-local static per call site and the
// numerics are byte-identical.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DIAG_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DIAG_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vllm::glm5_next::diag {

// `VT_GLM5_DIAG`, read ONCE per process and parsed with `atoi`:
//   0 / unset / unparsable — off, and every call site below is a no-op
//   1 — the step, the embedding, one line per decoder layer, the final hidden
//       state, the logits and their top-k, and each bridged layer's weights
//   2 — everything at 1 plus the SUBLAYER trace inside each decoder layer
// Unparsable falls back to 0 rather than refusing: a typo must not turn a
// production run into a diagnostic one.
int Level();

// Prints the raw environment string and the parsed level, ONCE. Every other
// function in this file is silent when the level is 0, so this is the only
// evidence that the instrument was compiled in and switched on.
void Banner(const char* where);

// NaN count, Inf count, min, max, mean, stddev, L2 and the first four values.
// The first four are there because the moments of an all-equal buffer and the
// moments of a symmetric one can coincide.
void Stats(const char* what, const float* v, size_t n);
void Stats(const char* what, const std::vector<float>& v);

// `Stats` plus the top-`k` (id, value) pairs, which is what separates "finite
// and uniform" from "finite and wrong": a uniform buffer has a top-1 margin of
// exactly zero and its top-k ids are 0, 1, 2, ... in index order.
void TopK(const char* what, const float* v, size_t n, int k);

// One line naming a step and the ids it carries, so a decode step that arrived
// with the wrong token is visible without reading the engine.
void Ids(const char* what, const std::vector<int32_t>& ids);

}  // namespace vllm::glm5_next::diag

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DIAG_H_
