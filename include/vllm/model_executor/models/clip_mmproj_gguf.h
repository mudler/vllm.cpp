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

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/deepseek_v4_vision.h"
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

// ─── Tensor accounting (QUANT-QWEN38-27B-GGUF-ARM, issue #821) ──────────────

// The EXACT set of tensor names `LoadQwen3VLVisionFromClipMmproj` reads for
// `cfg`: the two patch-embedding halves and their bias, the position table, the
// `cfg.depth` blocks, the merger, and one six-tensor merger per entry in
// `cfg.deepstack_visual_indexes`.
std::vector<std::string> Qwen3VLClipMmprojExpectedTensors(
    const multimodal::Qwen3VLVisionConfig& cfg);

// Refuse a projector that carries tensors the reader NEVER reads, naming them
// and the file.
//
// W1 read `mmproj-BF16.gguf`'s 334 names and closed the map in both directions
// once, behind an env gate over a 931 MB file on a share. CI read nothing, so a
// projector whose extra tensors are silently dropped — a deepstack tap the
// discovery loop missed, a second merger, a `clip` variant this build maps only
// part of — produced a tower that runs and is wrong. The MISSING direction is
// already named tensor by tensor inside the reader's own `load` lambda; this is
// the direction that had no detector.
void RefuseUnaccountedClipMmproj(const GgufFile& gguf,
                                 const multimodal::Qwen3VLVisionConfig& cfg,
                                 const std::string& path);

// ─── DeepSeek-V4 Flash Vision (`deepseek4v`) ────────────────────────────────
//
// The SECOND projector this file reads, and the second one only. Row
// `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W3A, issue
// [#2411](https://github.com/mudler/vllm.cpp/issues/2411).
//
// UPSTREAM. vLLM still has no GGUF loader at the pin, so the CONTAINER is read
// at the secondary oracle `llama-cpp-dsv4vision`
// ([`.agents/oracles/llama-cpp-dsv4vision.md`](../../../../.agents/oracles/llama-cpp-dsv4vision.md)),
// which pins `ggml-org/llama.cpp` release `b10766` =
// `9400c8946e4da5e7694f2c26d6d4e50e14b690fa`, the first release that converts,
// loads and runs this variant. The lines below were read from the diff that
// introduces `tools/mtmd/models/deepseek4v.cpp` as blob `ffe8f59d9997`, which
// is the blob that path holds AT that pin, so the anchors are the pin's own
// bytes rather than a pull-request head that may have moved. The BEHAVIOUR is
// the model author's own runtime, pinned at
// `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9`
// and already mirrored by W2 in `multimodal::DeepSeekV4Vision`. Anchors:
//
//   conversion/deepseek.py::DeepseekV4FlashVisionModel.set_gguf_parameters —
//       the `clip.*` keys this projector writes, including
//       `clip.vision.projector.scale_factor` (the downsample ratio),
//       `clip.use_silu = true` and the 1e-6 eps that is the vision RMSNorm's
//       torch default rather than the language model's 1e-20
//   conversion/deepseek.py::DeepseekV4FlashVisionModel.modify_tensors — the
//       two SPLITS this reader has to undo: `mlp.w1` is chunked into
//       `ffn_gate` + `ffn_up` with `chunk(2, dim=0)`, and
//       `vision.patch_embed.proj.weight` is VIEWED as a conv2d weight with
//       `data_torch.reshape(shape[0], 3, p, p)`
//   gguf-py/gguf/tensor_mapping.py and gguf-py/gguf/constants.py —
//       `vision.blocks.{bid}.attn.wqkv` maps to V_ENC_ATTN_QKV, which
//       `constants.py` spells `v.blk.{bid}.attn_qkv`. NOTHING SPLITS IT.
//       `conversion/base.py` contains no occurrence of `qkv` at all, the only
//       converter that splits a fused vision qkv is the model-specific
//       `conversion/qwenvl.py`, and
//       `conversion/deepseek.py::DeepseekV4FlashVisionModel.modify_tensors`
//       splits `mlp.w1` only. So the pinned `convert_hf_to_gguf.py` emits the
//       FUSED `v.blk.{bid}.attn_qkv.{weight,bias}`, which is 299 tensors at
//       depth 32, while the shipped
//       `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` mmproj-BF16.gguf carries
//       the SPLIT `attn_q` / `attn_k` / `attn_v` form, which is 427. This
//       reader reads the SPLIT form and refuses the FUSED one by name; the
//       fused arm is not implemented and the spec lists it under `## Owed`
//   tools/mtmd/clip-impl.h — `TN_TOK_IMG_START/_END/_PAD` and the `TN_*`
//       spellings of every `v.*` / `mm.*` name
//   tools/mtmd/clip.cpp::clip_model_loader, PROJECTOR_TYPE_DEEPSEEK4V — the
//       hyper-parameter reads, and `hparams.rope_theta = 10000.0f`, which this
//       projector hardcodes because no `clip.*` key carries it
//   tools/mtmd/models/deepseek4v.cpp::clip_graph_deepseek4v::build — which
//       tensor plays which role: `v.post_ln` is the tower's final norm, `mm.1`
//       and `mm.2` are the aligner's two GELU-separated projections, and the
//       four learned vectors are concatenated as extra rows of the token block
//
// SCOPE. This arm reads the projector into the W2 types and stops there. It is
// NOT reached from a production entry point: W4 wires
// `ModelRegistry::Forward`, and the row's spec lists the gap under `## Owed`.
inline constexpr const char* kClipProjectorDeepSeekV4 = "deepseek4v";

// Refuse, BY NAME, a file that is not a `deepseek4v` projector this build can
// load. Separate from `RefuseUnsupportedClipMmproj` ON PURPOSE, and the
// separation is load-bearing rather than stylistic: that function is the
// Qwen3-VL production path's discriminator, and `model_loader.cpp` goes
// straight from it into `LoadQwen3VLVisionFromClipMmproj`. Widening it to admit
// `deepseek4v` would route this file into the Qwen3-VL reader and build a tower
// that runs and is wrong, so it keeps refusing and this one exists beside it.
//
// It also refuses a projector declaring `clip.use_silu = false`, AND one that
// declares nothing. W2's MLP is SwiGLU by construction (it routes through
// `layers::MlpGateUpMethodBase`), the pinned converter writes the key as `true`
// for exactly that reason, and a GELU-MLP variant loaded as SwiGLU is fluent
// and wrong rather than broken. An ABSENT key is not "SwiGLU by omission":
// `tools/mtmd/clip.cpp` defaults to FFN_GELU_QUICK when neither `use_gelu` nor
// `use_silu` is set, so silence means the other activation.
//
// It refuses the FUSED `v.blk.{bid}.attn_qkv` layout by name as well, BEFORE
// `RefuseUnaccountedDeepSeekV4ClipMmproj` can blame the file for carrying
// tensors this reader never reads. That layout is what the pinned converter
// emits, so the file is correct and this build is the one with the gap.
void RefuseUnsupportedDeepSeekV4ClipMmproj(const GgufFile& gguf,
                                           const std::string& path);

// The tower geometry, read from the projector's OWN `clip.*` metadata.
//
// `rope_theta` is the one field no key carries. llama.cpp hardcodes 10000.0 for
// this projector in the same `clip_model_loader` case that reads the keys
// above, and the pinned converter's `get_vision_config` defaults
// `vision_rope_theta` to the same value without writing it, so the W2 default
// stands and is not invented here. It is also a KNOWN GAP shared with the
// oracle: the converter asserts `vision_max_n_token == 384` and
// `vision_max_wh_ratio == 8` but never the theta, so a future variant with a
// different one would be silently mis-read by llama.cpp too. The spec lists it
// under `## Owed`.
//
// Every field this reads is BOUNDED before it is returned, AND SO IS EVERY
// PRODUCT THE LOADER FORMS FROM THEM. Each field becomes a `Require` shape, a
// loop bound or a `resize` argument, and `KvInt` widens any integer spelling a
// converter chose, so an out-of-range value is refused with the key that
// carried it rather than surfacing as `length_error` or `bad_alloc`.
//
// THE SECOND HALF OF THAT SENTENCE WAS ADDED BECAUSE THE FIRST HALF ALONE WAS
// FALSE. Bounding each field left `3 * hidden * hidden` free: at an
// `embedding_length` of 65536, a sixteenth of what the field bound allows, the
// fused qkv buffer was reserved at 12,884,901,888 elements before any
// file-shaped read, and a 1.6 MB projector declaring it threw a bare
// `std::bad_alloc` past every refusal here. `kMaxTensorElements` now bounds the
// element count of each tensor the loader materializes, on the parsed values,
// and `tests/vllm/models/test_deepseek_v4_mmproj.cpp` carries that file.
//
// This arm is NOT reached from production today -- nothing outside the tests
// calls it, and the spec's `## Owed` gives W4 the wiring. The Qwen3-VL
// `ClipMmprojVisionConfig` beside it IS reached, reads `block_count` from a
// user-supplied `--mmproj` into an unbounded `resize`, and
// https://github.com/mudler/vllm.cpp/issues/2995 owns that.
multimodal::DeepSeekV4VisionConfig DeepSeekV4ClipMmprojVisionConfig(
    const GgufFile& gguf);

// One `deepseek4v` projector, read into the W2 types plus the four learned
// sentinel vectors W2 has no field for.
//
// `weights` holds NON-OWNING `vt::Tensor` views into `bf16_storage` and
// `f32_storage` below, so the struct owns its own weights and moving it keeps
// every view valid (moving a `vector<vector<T>>` transfers the outer buffer and
// leaves each inner heap block where it is). Copying would silently duplicate
// the storage and leave the views pointing at the original, so it is deleted.
//
// The tensors are HOST tensors on the default device. W4 owns the upload: this
// wave has no production call site and inventing a device policy here would be
// a decision made by the wrong wave.
struct DeepSeekV4ClipMmproj {
  multimodal::DeepSeekV4VisionWeights weights;

  // `v.token_embd.img_start` / `_end` / `_pad` and `v.image_newline`, each
  // `[output_size]`. They stay f32, which is the dtype the file holds and the
  // dtype llama.cpp concatenates them at: W2 declares no dtype for them because
  // it has no field for them, and W4 owns where they are placed, so narrowing
  // them here would be a dtype decision made by the wrong wave.
  std::vector<float> image_start;
  std::vector<float> image_end;
  std::vector<float> image_pad;
  std::vector<float> image_newline;

  // Host storage behind `weights`. Never read directly.
  std::vector<std::vector<uint16_t>> bf16_storage;
  std::vector<std::vector<float>> f32_storage;

  DeepSeekV4ClipMmproj() = default;
  DeepSeekV4ClipMmproj(DeepSeekV4ClipMmproj&&) = default;
  DeepSeekV4ClipMmproj& operator=(DeepSeekV4ClipMmproj&&) = default;
  DeepSeekV4ClipMmproj(const DeepSeekV4ClipMmproj&) = delete;
  DeepSeekV4ClipMmproj& operator=(const DeepSeekV4ClipMmproj&) = delete;
};

// Load the DeepSeek-V4 vision tower and aligner out of a `deepseek4v` mmproj
// into the W2 `multimodal::DeepSeekV4VisionWeights`.
//
// REFUSES BY NAME, and undoes FOUR layout differences between what the file
// stores and what W2 consumes. Every one of them is a silent wrong answer when
// it is wrong, not a crash:
//
//   * a missing tensor names itself, and a wrong-shaped one names both shapes;
//   * `attn_q` / `attn_k` / `attn_v` are stored SEPARATELY and fuse into
//     `qkv_weight [3*hidden, hidden]` in q, k, v ROW order, which is the order
//     `deepseek_v4_vision.cpp` slices them back out at;
//   * `ffn_gate` and `ffn_up` are stored SEPARATELY and concatenate into
//     `mlp_w1_weight [2*intermediate, hidden]` GATE FIRST, which is the half
//     `vt::SiluAndMul` applies SiLU to;
//   * `v.patch_embd.weight` is a 4-D conv2d weight and flattens back into the
//     2-D torch Linear weight W2 reads, in [channel, dy, dx] column order.
DeepSeekV4ClipMmproj LoadDeepSeekV4VisionFromClipMmproj(
    const GgufFile& gguf, const multimodal::DeepSeekV4VisionConfig& cfg);

// The EXACT set of tensor names `LoadDeepSeekV4VisionFromClipMmproj` reads for
// `cfg`: the patch embedding and its bias, `cfg.depth` blocks of thirteen, the
// final norm, the aligner's two weight/bias pairs, and the four sentinels. On
// the pinned artifact (depth 32) that is 427, which is its tensor count.
std::vector<std::string> DeepSeekV4ClipMmprojExpectedTensors(
    const multimodal::DeepSeekV4VisionConfig& cfg);

// Refuse a `deepseek4v` projector that carries tensors the reader NEVER reads,
// naming them and the file. Same direction, and the same reason, as
// `RefuseUnaccountedClipMmproj`: the MISSING direction names itself tensor by
// tensor inside the reader, and this is the direction that would otherwise drop
// a name silently and produce a tower that runs and is wrong.
void RefuseUnaccountedDeepSeekV4ClipMmproj(
    const GgufFile& gguf, const multimodal::DeepSeekV4VisionConfig& cfg,
    const std::string& path);

}  // namespace vllm
