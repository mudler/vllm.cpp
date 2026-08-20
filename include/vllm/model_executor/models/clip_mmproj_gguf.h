// A llama.cpp `clip` mmproj GGUF — the SECOND file a GGUF multimodal model
// ships, beside the language file (row `LOAD-GGUF-MMPROJ`, issue
// [#821](https://github.com/mudler/vllm.cpp/issues/821),
// `.agents/specs/qwen38-27b-quant-arms.md` W1).
//
// UPSTREAM. Pinned vLLM `5559679229` has no GGUF loader at all
// (`vllm/model_executor/model_loader/__init__.py:33-49` registers no GGUF
// format), so vLLM defines the vision tower's BEHAVIOR — which this project
// already mirrors in `multimodal::Qwen3VLVisionForward` from
// `vllm/model_executor/models/qwen3_vl.py` — and llama.cpp defines only the
// CONTAINER. The container is read at the pinned secondary oracle
// llama.cpp `b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`. The line
// numbers below were read AT THAT COMMIT (`git show 10bf611e5:<path>`), which
// the tag `b10451` names and which `ggml-org/llama.cpp` `origin/master`
// contains — NOT at the superseded local fork `237ad9b96` whose positions
// `backend-matrix.md` records as owed re-anchoring (#1003):
//
//   tools/mtmd/clip-impl.h:499::PROJECTOR_TYPE_NAMES — `qwen3vl_merger`
//       (:444 enum, :507 name) and `muse-glimmer` (:495 enum, :557 name)
//   tools/mtmd/clip-impl.h:33,40-44,47,58,65 — the `clip.*` metadata key
//       spellings (KEY_PROJ_TYPE :33, KEY_N_EMBD :40, KEY_N_FF :41,
//       KEY_N_BLOCK :42, KEY_PROJ_DIM :43, KEY_N_HEAD :44,
//       KEY_LAYER_NORM_EPS :47, KEY_PATCH_SIZE :58,
//       KEY_SPATIAL_MERGE_SIZE :65)
//   tools/mtmd/clip-impl.h:104,106-108,131-132,153-155 —
//       TN_POS_EMBD :104, TN_PATCH_EMBD :106, TN_PATCH_EMBD_1 :107,
//       TN_PATCH_BIAS :108, TN_LN_POST :131, TN_LLAVA_PROJ :132,
//       TN_DEEPSTACK_NORM / _FC1 / _FC2 :153-155, plus TN_LN_1 / TN_LN_2 /
//       TN_ATTN_QKV / TN_ATTN_OUTPUT / TN_FFN_UP / TN_FFN_DOWN in the same
//       block
//   tools/mtmd/clip.cpp:2021::clip_model_loader::load_tensors — the per-block
//       reads
//   tools/mtmd/models/qwen3vl.cpp:3::clip_graph_qwen3vl::build — which tensor
//       plays which role, including that `v.post_ln` is applied BEFORE the
//       merge reshape (:164-165, so it is our merger's PRE-shuffle norm) and
//       that the projection is `mm.0` -> GELU -> `mm.2` (:172-176, the one
//       `build_ffn` call; `mm_1_w` is `TN_LLAVA_PROJ` index 2 for this
//       projector type — `clip.cpp:2392`, inside the `PROJECTOR_TYPE_QWEN3VL`
//       case that opens at `:2388`, NOT the identical-looking `:2385` in the
//       QWEN2VL/QWEN25VL/EXAONE4_5 case above it — which is why `mm.1` is not
//       a name here)
//   tools/mtmd/models/qwen2vl.cpp:3::clip_graph_qwen2vl::build_inp_with_temporal_merge
//       — the two patch-embedding halves are two `conv2d`s over the two
//       temporal frames, SUMMED by `ggml_add` (:12-26); that is a `conv3d`
//       with `temporal_patch_size = 2` split along its temporal axis, and
//       `n_batch > 2` is refused outright (:28)
//
// The mapping target for the config is the SAME `Qwen3VLVisionConfig` that
// `src/vllm/model_executor/models/minimax_h3_vision_gguf.cpp::MiniMaxH3EncoderVisionConfig`
// builds from a `visual.*` encoder GGUF. This file reads it from the projector's
// own `clip.*` metadata instead of hardcoding a checkpoint's numbers.
//
// SCOPE. `qwen3vl_merger` only. Every other `clip.projector_type` is refused BY
// NAME rather than half-loaded, and `muse-glimmer` is routed to its own recorded
// refusal (`MuseGlimmerRefuseMmproj`), which this row is what finally makes
// reachable from production.
#pragma once

#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"

namespace vllm {

// `general.architecture` of a llama.cpp multimodal projector file
// (clip-impl.h; every mmproj llama.cpp writes carries it).
inline constexpr const char* kClipGgufArch = "clip";
// `general.type`, which distinguishes a projector from a plain vision encoder.
inline constexpr const char* kClipGgufTypeMmproj = "mmproj";
// The ONE `clip.projector_type` this build loads
// (clip-impl.h::PROJECTOR_TYPE_NAMES, PROJECTOR_TYPE_QWEN3VL).
inline constexpr const char* kClipProjectorQwen3VL = "qwen3vl_merger";
// PROJECTOR_TYPE_MUSE_GLIMMER. Named because its file is refused for a
// SPECIFIC, already-recorded reason rather than for being unsupported.
inline constexpr const char* kClipProjectorMuseGlimmer = "muse-glimmer";

// True iff `general.architecture` == "clip". Cheap probe on an already-open
// file; it does NOT say the projector type is one we load.
bool IsClipMmprojGguf(const GgufFile& gguf);

// `clip.projector_type`, or "" when the key is absent or not a string.
std::string ClipProjectorType(const GgufFile& gguf);

// Refuse, BY NAME, a file that is not a `clip` mmproj this build can load.
// `path` is quoted back so a user who passed the language file (or a second
// language file) to --mmproj is told which file was wrong. Returns normally
// only for `clip` + `qwen3vl_merger`.
//
// A `muse-glimmer` projector takes MuseGlimmerRefuseMmproj's own message: the
// file is a valid `clip` mmproj that our perception tower cannot be built from,
// and the reason (its `v.patch_embd.weight` carries only 588 of the 1176 input
// features) is recorded there, not here.
void RefuseUnsupportedClipMmproj(const GgufFile& gguf, const std::string& path);

// The tower geometry, read from the projector's OWN `clip.*` metadata.
//
// `temporal_patch_size` is 2 by construction rather than by metadata: llama.cpp
// writes the temporal axis as exactly two `conv2d` halves
// (qwen2vl.cpp::build_inp_with_temporal_merge asserts n_batch <= 2), and there
// is no `clip.*` key for it. `num_position_embeddings` and `in_channels` are
// read from the TENSOR shapes, which are the only place the file states them.
multimodal::Qwen3VLVisionConfig ClipMmprojVisionConfig(const GgufFile& gguf);

// Load the Qwen3-VL vision tower out of a `qwen3vl_merger` mmproj into the
// SHARED host-f32 weights `multimodal::Qwen3VLVisionForward` consumes — the
// same struct the safetensors reader (`LoadQwen3VLVisionWeights`) and the
// MiniMax-H3 encoder reader (`LoadQwen3VLVisionFromGguf`) fill.
//
// REFUSES BY NAME, and the refusals are the point of the row:
//   * a missing tensor names itself;
//   * a file carrying `v.patch_embd.weight` WITHOUT `v.patch_embd.weight.1`
//     is refused with both shapes in the message. That is the MuseGlimmer
//     condition (`muse_glimmer_gguf_weights.h` §REFUSED-AND-RECORDED) enforced
//     rather than assumed: joining one half would mean inventing the temporal
//     half of a weight, and the result would be a fluent, wrong model rather
//     than an error.
multimodal::Qwen3VLVisionWeights LoadQwen3VLVisionFromClipMmproj(
    const GgufFile& gguf, const multimodal::Qwen3VLVisionConfig& cfg);

}  // namespace vllm
