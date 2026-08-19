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
// llama.cpp `b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`:
//
//   tools/mtmd/clip-impl.h::PROJECTOR_TYPE_NAMES  — `qwen3vl_merger`,
//       `muse-glimmer`, and the `clip.*` metadata key spellings (KEY_N_EMBD,
//       KEY_N_FF, KEY_N_BLOCK, KEY_PROJ_DIM, KEY_N_HEAD, KEY_PATCH_SIZE,
//       KEY_SPATIAL_MERGE_SIZE, KEY_LAYER_NORM_EPS, KEY_PROJ_TYPE)
//   tools/mtmd/clip-impl.h::TN_PATCH_EMBD / TN_PATCH_EMBD_1 / TN_PATCH_BIAS /
//       TN_POS_EMBD / TN_LN_1 / TN_LN_2 / TN_ATTN_QKV / TN_ATTN_OUTPUT /
//       TN_FFN_UP / TN_FFN_DOWN / TN_LN_POST / TN_LLAVA_PROJ /
//       TN_DEEPSTACK_NORM / TN_DEEPSTACK_FC1 / TN_DEEPSTACK_FC2
//   tools/mtmd/clip.cpp::clip_model_loader::load_tensors — the per-block reads
//   tools/mtmd/models/qwen3vl.cpp::clip_graph_qwen3vl::build — which tensor
//       plays which role, including that `v.post_ln` is applied BEFORE the
//       merge reshape (so it is our merger's PRE-shuffle norm) and that the
//       projection is `mm.0` -> GELU -> `mm.2`
//   tools/mtmd/models/qwen2vl.cpp::clip_graph_qwen2vl::build_inp_with_temporal_merge
//       — the two patch-embedding halves are two `conv2d`s over the two
//       temporal frames, SUMMED; that is a `conv3d` with
//       `temporal_patch_size = 2` split along its temporal axis
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
