// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W5b-2b: the ENGINE BINDING. What
// makes `ModelRegistry::Forward` stop refusing by name.
//
// Issue [#2241](https://github.com/mudler/vllm.cpp/issues/2241), campaign issue
// [#1998](https://github.com/mudler/vllm.cpp/issues/1998), spec
// `.agents/specs/glm5-next-flash.md` §W5b-2b.
//
// Model-private, deliberately not under `include/`, the same arrangement every
// other file on this row uses: `include/vllm.h` is the ABI seam, and what this
// row exposes through it is a MODEL that loads and forwards, not a set of C++
// types.
//
// ─── WHAT THIS FILE IS FOR ───────────────────────────────────────────────────
//
// W5b-2a landed `TextModelForward`, which takes a `TextModelWeights` — every
// layer's host f32 weights at once. At the published geometry that object is
// **426.72 GiB** against a **~119.63 GiB** box, so no production forward can
// build one. This file is the two things that stand between a loaded
// `Glm5NextWeights` and a token:
//
//   1. `Glm5NextGgufLayerSource` — a `LayerWeightSource` that bridges ONE
//      decoder layer at a time out of the block-resident tower and overwrites
//      its single slot on the next layer. Peak: one layer.
//   2. `Glm5NextHostForward` — the embed gather, the model forward, the
//      logits gather and the streamed `lm_head`, from a `ModelForwardInput`'s
//      three consumed fields.
//
// ─── THE RESIDENCY LADDER, ALL OF IT THIS ROW'S OWN ARITHMETIC ──────────────
//
//   | what, at the published geometry | GiB |
//   |---|---:|
//   | the tower as loaded, block-resident | 101.14 |
//   | the tower expanded to f32 (what `TextModelWeights` would be) | 426.72 |
//   | usable on `dgx:gpu0`, the largest device this project reaches | ~119.63 |
//   | ONE bridged DSA layer, f32 | 0.4654 |
//   | ONE bridged KDA layer, f32 | 0.5449 |
//   | ONE sparse layer's ROUTER + shared expert, f32 | 0.0996 |
//   | ONE routed expert, f32 (`gate_up` + `down`) | 0.0938 |
//   | one `lm_head` chunk, f32 (`kLmHeadChunkBytes`) | 0.0625 |
//
// So the forward's f32 peak is one layer plus one expert plus one chunk —
// under 0.75 GiB, 0.6% of the box — while the tower it reads stays exactly as
// the loader left it. **The expert banks are never bridged at all**: a sparse
// layer's three banks are 27.0 GiB in f32, and `kBridgeTensorF32ByteCeiling`
// refuses the first one BY NAME at 9.0 GiB before anything is allocated.
//
// ─── FULL-PREFIX RECOMPUTE, AND IT IS A DECISION AND NOT AN OVERSIGHT ────────
//
// This forward RE-RUNS THE WHOLE PREFIX every step and keeps no per-request
// state. That is the house pattern and it was surveyed rather than assumed: two
// registered models already do exactly this inside their `forward` hook —
// `NemotronHForCausalLM` (`nemotron_h_registry.cpp`, whose host arm "consumes
// three of `ModelForwardInput`'s fields — `token_ids`, `logits_indices`,
// `queue`") and `KimiLinearForCausalLM` (`kimi_linear_forward.cpp`, which
// `(void)`s `positions`, `attn_meta`, `attn_kv` and `queue`) — and no
// `LoadedModel` in this tree keeps per-request KV or recurrent state on itself.
// Kimi-Linear is the closest architecture there is to this one (KDA plus MLA),
// so its shape is the precedent.
//
// **The cost is named rather than hidden.** W5b-2a's `LayerCache` binding —
// the DSA latent and the KDA recurrence carried across steps — is gated and
// correct and this path does not call it, so it stays unreached. Carrying
// `std::vector<LayerCache>` on `Glm5NextLoadedModel` has no precedent in this
// tree, and inventing one on a model that cannot be run end to end on this
// fleet is the wrong place to try it. The spec's `## Owed` records it.
//
// ─── ONE DELIBERATE DIVERGENCE FROM THE HOUSE PATTERN, AND WHY ──────────────
//
// Nemotron-H and Kimi-Linear both take `input.token_ids` as ONE sequence and
// ignore `num_reqs`. On a step carrying two requests that silently attends
// across the boundary, which for this model is a fluent wrong answer that no
// gate on this fleet could detect (`.agents/specs/glm5-next-flash.md` §Gates:
// no end-to-end token gate for this model exists or can exist here). So a step
// with more than one request is REFUSED BY NAME here instead. Ragged batching
// is `attn_meta.query_start_loc` sliced as at `kimi_linear_device.cpp`, and it
// is owed rather than approximated.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_FORWARD_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_FORWARD_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_layer.h"
#include "vllm/model_executor/models/glm5_next_loader.h"
#include "vt/ops.h"  // vt::Queue

namespace vllm::glm5_next {

// How many f32 bytes of `lm_head` one chunk of the logits GEMM holds. 64 MiB,
// which is 4096 rows of the published `[154880, 4096]` head — 38 chunks per
// step, each one dropped before the next is decoded. The whole head is 2.36 GiB
// in f32 and `DecodeOwnedTensorToF32` refuses it by name at the 1 GiB ceiling,
// which is the ceiling working and not an obstacle.
inline constexpr int64_t kLmHeadChunkBytes = int64_t{64} << 20;

// A `LayerWeightSource` over a loaded GGUF tower: bridge layer `i` on demand
// and OVERWRITE the single slot.
//
// **There is exactly one slot and no map keyed by layer index**, for the reason
// `glm5_next_bridge.h` gives for having no `BridgeTower`: a map turns "one
// layer" into "every layer visited so far", which is the 426.72 GiB tower again
// with a slower ramp. Re-asking for the layer already in the slot is served
// without a re-bridge; asking for any other one drops it first.
//
// The `ExpertSource` for a sparse layer is owned here too and its lifetime is
// the slot's, which is why `MoeLayerWeights::expert_source` can be a borrowed
// pointer: it never outlives the object that filled it.
class Glm5NextGgufLayerSource final : public LayerWeightSource {
 public:
  explicit Glm5NextGgufLayerSource(
      const Glm5NextWeights& weights,
      int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

  int64_t size() const override;
  const DecoderLayerWeights& Layer(int64_t layer_idx) override;

  // How many times a layer was actually BRIDGED, which is not the same as how
  // many times `Layer` was called. An instrument: a gate reads it to prove the
  // slot is one layer and not a cache that grew.
  int64_t bridged() const { return bridged_; }
  // The f32 bytes the current slot holds, MEASURED from the decoded buffers.
  int64_t slot_f32_bytes() const { return slot_f32_bytes_; }

 private:
  const Glm5NextWeights* weights_;
  int64_t byte_ceiling_;
  DecoderLayerWeights slot_;
  std::unique_ptr<GgufExpertSource> experts_;
  int64_t loaded_ = -1;
  int64_t bridged_ = 0;
  int64_t slot_f32_bytes_ = 0;
};

// The host reference forward, from a loaded tower to logits.
//
//   token_ids      : ONE sequence's ids. The prefix is re-run in full.
//   logits_indices : positions to emit, EMPTY meaning every position. The
//                    gather happens BEFORE `lm_head` so the head never runs on
//                    the full `T`, and every index is bounds-checked.
//   queue          : must be a CPU queue. Every buffer on this path is host
//                    f32 and `vt::MoeRouterTopK` / `vt::MoeCombine` dispatch on
//                    the queue's device, so a CUDA queue here is a crash and
//                    not a fallback. Refused by name.
//   lm_head_chunk_bytes : how much of the head one chunk may hold. Exposed so
//                    the chunk BOUNDARY is a gated fact and not an untested
//                    loop: at the published geometry the default gives 38
//                    chunks, and at any fixture small enough to run in a test
//                    it gives ONE, so a chunking defect would never be reached
//                    by a gate that could only take the default.
//
// Returns [rows, vocab_size] row-major, `rows == logits_indices.size()` or `T`.
std::vector<float> Glm5NextHostForward(const Glm5NextWeights& weights,
                                       const std::vector<int32_t>& token_ids,
                                       const std::vector<int32_t>& logits_indices,
                                       vt::Queue& queue,
                                       int64_t lm_head_chunk_bytes = kLmHeadChunkBytes);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_FORWARD_H_
