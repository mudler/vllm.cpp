// MiniMax-Music3 — the QUANTIZED ARMS (W7 of #672).
//
// Row MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation,
// .agents/specs/minimax-music3.md phase W7 and section 9. Issue #672.
//
// ─── WHAT THIS FILE IS ──────────────────────────────────────────────────────
//
// AGENTS.md: "A model port covers the quantized arms, not just bf16. GGUF
// k-quants in particular are a standing requirement, not a per-model choice...
// An arm that is not implemented is refused with a message naming the missing
// piece and recorded as owed -- never left to be discovered later."
//
// This is the refusal. Not one arm is loaded here, and this header says so in
// its first paragraph rather than in a footnote: MiniMax-Music3 today has NO
// quantized arm in this tree, and what W7 landed is the diagnosis that turns a
// quantized checkpoint from a confusing shape error into a message naming the
// format, the evidence, the missing piece and the phase that owes it.
//
// ─── QUANTIZED CHECKPOINTS FOR THIS MODEL EXIST. THAT IS THE FINDING ────────
//
// A survey of HuggingFace on 2026-08-14 (spec section 9 records every query and
// its result count) found that MiniMaxAI ships bf16/fp32 ONLY -- and that the
// community had already published, within two days of the release, at least
// fourteen quantized repositories in five distinct formats:
//
//   GGUF k-quants   Abiray/MiniMax-Music3-GGUF, realrebelai/MiniMax-Music-3_GGUFs,
//                   molbal/Minimax-Music3-GGUF, ChrisColeTech/minimax-music3-GGUF
//                   (the 2.46B DiT alone, Q2_K..Q8_0, 0.9-2.7 GB);
//                   audio-cpp/MiniMax-Music3-GGUF (one GGUF PER COMPONENT with
//                   per-module config JSON, bf16 and Q4_K arms);
//                   scragnog/MiniMax-Music3-GGUF (a 2-file split over 13 tiers,
//                   including MXFP4 and NVFP4 as GGML tensor types)
//   int8 / w4a8     Comfy-Org/MiniMax-Music-3 `..._int8_convrot`,
//                   NidAll/MiniMax-Music3-W4A8,
//                   dummy9996/MiniMax-Music3-w4a8-bf16-comfyui
//   MLX 4/6/8-bit   ddalcu, vanch007, elishabjm
//   proprietary     infosave/MiniMax-Music-3-cmf (Cortiq 4-bit)
//
// NOT found by those queries: AWQ, GPTQ, compressed-tensors, fp8 / fp8_e4m3fn /
// fp8_scaled, bitsandbytes, or NVFP4/MXFP4 as a standalone safetensors
// checkpoint. Those four formats are refused here anyway, because a refusal is
// cheap and "we looked and did not find one on one day" is not a guarantee that
// nobody publishes one tomorrow.
//
// The consequence for the port is exact and it is recorded rather than implied:
// **the bf16/fp32 arm is ~28.5 GB and the Q4_K arm of the same weights is
// ~9 GB**, so the quantized arm is what most users can actually run and what a
// quant-matched llama.cpp comparison would need. It is OWED. What blocks it is
// not knowledge -- the artifacts are enumerated above -- it is that none is
// staged on this box and fetching one needs authority.
//
// ─── WHY A SEPARATE TRANSLATION UNIT ────────────────────────────────────────
//
// .agents/porting-a-model.md: "GGUF is its own translation unit, not an
// afterthought bolted onto the safetensors loader". MiniMax-H3 follows that
// (minimax_h3_gguf.cpp, minimax_h3_nvfp4.cpp) and so does this: when an arm is
// implemented it lands beside this file, and the detector below is what routes
// to it. Putting the detection in the safetensors loader would have made the
// first real arm a rewrite of W1.
//
// ─── THREE DETECTORS, BECAUSE THERE ARE THREE PLACES TO LOOK ────────────────
//
// A quantized checkpoint announces itself in exactly one of three ways, and no
// single detector sees all three:
//
//   TREE      the artifact is not safetensors at all -- `.gguf` files, possibly
//             nested under `diffusion_models/` or `text_encoders/`. There is no
//             component directory to account and no config to parse, so this is
//             the ONLY place it can be caught. Before W7 such a tree got
//             "is not a diffusers-arm MiniMax-Music3 checkpoint; it is missing
//             transformer, condition_encoder, ..." -- seven directories the
//             user does not have and never will, and no mention of GGUF.
//   MANIFEST  the tree IS diffusers-shaped and the tensors are not: an NVFP4
//             triple, an MXFP4 pack, an AWQ qweight, a bitsandbytes absmax, or
//             a weight whose only change is its DTYPE. Before W7 this surfaced
//             as W1's shape or dtype refusal on whichever tensor sorted first,
//             which for a real NVFP4 condition_encoder is `layer_scale` -- a
//             tensor that is not quantized, is not wrong, and has nothing to do
//             with the problem.
//   CONFIG    the checkpoint DECLARES it, in `quantization_config.quant_method`
//             (transformers/compressed-tensors/ModelOpt) or in `quantization`
//             (MLX). This fires before a weight byte is read.
//
// ─── WHAT THE DETECTOR REFUSES TO GUESS ─────────────────────────────────────
//
// A bare `weight_scale` with no `weight_scale_2` and no `weight_packed` is
// consistent with NVFP4-without-the-global-scale, with a compressed-tensors
// block scheme, and with a per-channel int8 scale. It resolves to
// `kUnknownScheme` and the refusal names all three CANDIDATES rather than
// picking one. That is deliberate: ltx2_loader.h:232-268 records what picking
// one costs -- a finite, correctly shaped, correctly scaled, WRONG result that
// no shape gate can see. A refusal that names three candidates is worth more
// than a load that names one wrongly.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_music3_loader.h"

namespace vllm {

// The quantization formats this port can DIAGNOSE. Every one of them is
// refused; none is loaded. The enumeration is closed on purpose -- a format
// nobody named would fall through to `kUnknownScheme`, which refuses too.
enum class MiniMaxMusic3QuantFormat {
  // The gated arm: bf16 AR half, fp32 acoustic half (spec section 2.1). Not a
  // quantization, and the ONLY value that does not refuse.
  kNone,
  // llama.cpp / ComfyUI GGUF, any k-quant or i-quant tier. The standing
  // requirement of AGENTS.md and the arm six repositories already ship.
  kGguf,
  // The compressed-tensors NVFP4 triple: U8 packed [out, in/2], F8_E4M3 group
  // scale [out, in/16], F32 `weight_scale_2` global.
  kNvfp4,
  // compressed-tensors MXFP4: `weight_packed` plus an E8M0 `weight_scale`.
  kMxfp4,
  // An fp8 weight -- F8_E4M3 or F8_E5M2. Detectable only by DTYPE: the tensor
  // keeps its name and its true [out, in] shape.
  kFp8,
  // An int8 weight. Comfy-Org's `int8_convrot` arm and the two w4a8 repos.
  // NOT fp8, and a refusal that said fp8 would be wrong.
  kInt8,
  // AWQ or GPTQ: the `qweight` / `qzeros` / `scales` triple. The two are not
  // separated here because the manifest cannot separate them.
  kAwqGptq,
  // bitsandbytes 4-bit blockwise: `<weight>.absmax` / `.quant_map`.
  kBitsAndBytes,
  // Apple MLX quantized safetensors: `quantization: {group_size, bits}`.
  kMlx,
  // A `quantization_config` naming compressed-tensors or ModelOpt explicitly.
  kCompressedTensors,
  // Quantized, but the evidence does not identify WHICH scheme. The refusal
  // names the candidates; it never picks one.
  kUnknownScheme,
};

// A short, stable, human name. Used in the refusal text, so it is also what a
// test asserts on. `kNone` names the supported arm rather than being empty.
const char* MiniMaxMusic3QuantFormatName(MiniMaxMusic3QuantFormat format);

// What a detector found, and HOW MUCH IT LOOKED AT. `examined` and `matched`
// are returned rather than logged because AGENTS.md's testing discipline is
// that a gate which cannot say how many things it examined has not reported --
// a detector that silently walked an empty directory and a detector that walked
// 1012 tensors and found nothing are the same value otherwise.
struct MiniMaxMusic3QuantFinding {
  MiniMaxMusic3QuantFormat format = MiniMaxMusic3QuantFormat::kNone;
  // The exact file, tensor or config key that PROVES the format. Never a
  // paraphrase: this is what the user greps for in their own checkpoint.
  std::string evidence;
  // The component directory, or empty for a tree-level finding.
  std::string component;
  int64_t examined = 0;
  int64_t matched = 0;
};

// TREE level. Walks `root` to a bounded depth looking for `.gguf` files and,
// failing that, for a `config.json` that declares a quantization. Never throws:
// a caller may ask before committing to a load, and a directory it cannot read
// is reported as `kNone` with `examined == 0` rather than as an accusation.
MiniMaxMusic3QuantFinding MiniMaxMusic3DetectQuantTree(const std::string& root);

// MANIFEST level, from safetensors header entries alone -- no payload is read.
// Rules are applied in a fixed PRIORITY order and the first rule with any match
// wins, so the answer cannot depend on the order the entries arrive in.
MiniMaxMusic3QuantFinding MiniMaxMusic3DetectQuantManifest(
    const std::string& component, const std::vector<MiniMaxMusic3ManifestEntry>& entries);

// CONFIG level, from the raw JSON text of one component's `config.json`.
// Takes text rather than a parsed document so this header stays free of the
// JSON dependency and so a test can state the fixture literally.
//
// Tri-state is honoured: a key that is ABSENT and a key that is `null` both
// mean not-quantized (.agents/porting-a-model.md section 1). Text that is not
// valid JSON is `kNone` -- refusing a config as quantized because it failed to
// parse would be an accusation drawn from an instrument failure.
MiniMaxMusic3QuantFinding MiniMaxMusic3DetectQuantConfig(const std::string& component,
                                                         const std::string& config_json);

// THROW naming the format, the evidence, the missing piece, the supported arm,
// the phase that owes the work and the issue -- or return, when the finding is
// `kNone`.
//
// Every branch of the message is asserted by tests/vllm/models/
// test_minimax_music3_quant.cpp, because a refusal nobody proved firing is not
// a refusal.
void MiniMaxMusic3CheckQuantArm(const MiniMaxMusic3QuantFinding& finding);

// The refusal text, without throwing. Exposed so a caller can report rather
// than abort, and so the message can be gated on its own.
std::string MiniMaxMusic3QuantRefusal(const MiniMaxMusic3QuantFinding& finding);

// ---------------------------------------------------------------------------
// THE ONE ARM THAT IS IMPLEMENTED: the RVQ depth decoder, GGUF Q4_K
// ---------------------------------------------------------------------------
//
// ─── WHY THIS COMPONENT AND ONLY THIS COMPONENT ─────────────────────────────
//
// It is the one arm whose bound can be CALIBRATED rather than asserted. W3
// already gates the depth decoder at full scale against `frame_hiddens[:,4096:]`
// (716 800 values) and its bf16 control is already MEASURED — torch against
// itself with a different attention kernel, 46.34% bit-identical at
// mean|d| 1.659e-03. A Q4_K arm therefore has a known baseline to sit outside
// of, instead of a tolerance somebody chose. The other four components have no
// such measured control, so implementing them here would mean asserting bounds
// rather than deriving them; they stay refused and owed (spec section 9.5).
//
// ─── THE ARTIFACT IS PINNED, BECAUSE AN UNPINNED ORACLE IS NOT REPRODUCIBLE ──
//
//   repo      audio-cpp/MiniMax-Music3-GGUF
//   revision  c36aaeed683f33b05796788e4204f4eeba8fa547
//   file      rvq_depth_decoder_q4_k.gguf, 405 752 480 bytes
//   sha256    4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0
//
// The digest is ASSERTED by the gate, not merely recorded: a re-quantized file
// under the same name is a different oracle, and this project has already been
// bitten by a checkpoint silently re-quantized in place under an unchanged repo
// id.
//
// ─── THREE LINEAGES, AND ONLY ONE IS READ ───────────────────────────────────
//
// Spec section 9.2 measured ten published Music3 GGUFs: they are three mutually
// incompatible lineages and `general.architecture` CANNOT separate them (it
// reads `audiocpp`, `mm3`, `qwen3` and `wan` for the same model, and `wan`
// collides with genuine Wan video GGUFs, so keying on it would bind another
// model's checkpoint).
//
// This arm reads the `audiocpp` lineage ONLY, identified by
// `audiocpp.model_spec.family == "minimax_music3"`, and it is the right one to
// start from for a reason that is measured rather than aesthetic: it declares
// `audiocpp.tensor_name_format = native`, and all 47 tensor names match the
// enumeration `EnumerateMiniMaxMusic3RvqDepthDecoderTensors` derives from
// upstream source EXACTLY. **There is no rename table**, so a mis-binding
// cannot hide in one. The other two lineages are refused BY NAME: `mm3` needs a
// rename table plus fused QKV to split and folded weight-norm to invert, and the
// ComfyUI lineage does not contain this component at all.
//
// ─── DTYPE: bf16 OUT, MIRRORING THE SAFETENSORS ARM ─────────────────────────
//
// The 36 quantized projections dequantize to bf16 and the 11 islands (9 BF16
// norms, 2 F16 embeddings) convert to bf16. That is the component's runtime
// dtype under spec section 2.1 and what the safetensors arm resolves, so the
// quantized arm is substitutable rather than a second numeric regime. Producing
// f32 here would be a WIDENING no measurement asked for, and AGENTS.md is
// explicit that a token gate cannot catch a dtype that is too wide.

class GgufFile;

// What the load ACTUALLY did, per tensor and in totals.
//
// This struct exists for one reason: **a value gate cannot see a dequant
// fallback.** Outputs that land inside tolerance prove nothing about which
// bytes produced them, and this project has already been bitten by exactly
// that. So the load reports the RESIDENT ggml type of every tensor it read and
// counts the dequantizations it performed, and the gate asserts on those
// directly rather than inferring them from the numbers.
struct MiniMaxMusic3GgufLoadReport {
  int64_t tensors = 0;        // tensors read
  int64_t quantized = 0;      // tensors whose resident type was a k-quant
  int64_t unquantized = 0;    // the F16/BF16/F32 islands
  int64_t dequant_calls = 0;  // calls into the shared gguf_dequant seam
  int64_t elements = 0;       // elements materialized
  // name -> the ggml type id the FILE stores. A fallback that read a bf16
  // sibling instead would show BF16(30) here where Q4_K(12) is expected.
  std::map<std::string, uint32_t> resident_type;
};

// True when `file` is the audio-cpp `native` lineage carrying MiniMax-Music3.
// Never throws: a caller may ask before committing.
bool MiniMaxMusic3GgufIsNativeLineage(const GgufFile& file);

// Load the RVQ depth decoder from a `native`-lineage GGUF, to bf16.
//
// Accounts before it materializes: every tensor the config OWES must exist at
// its enumerated SHAPE, and every tensor present must be accounted for — the
// same "enumerated == present, zero unaccounted" contract the safetensors arm
// uses, so a GGUF that lost a tensor cannot read as zeros. The per-tensor DTYPE
// is whatever the quantizer chose and is recorded rather than required.
//
// THROWS, naming the missing piece, when: the lineage is not `audiocpp`; the
// file is a different component; a tensor is missing, extra, or wrongly shaped;
// or a tensor's ggml type is one the shared dequant seam does not read.
MiniMaxMusic3ComponentWeights MiniMaxMusic3LoadRvqDepthDecoderFromGguf(
    const MiniMaxMusic3RvqDepthDecoderConfig& config, const GgufFile& file,
    MiniMaxMusic3GgufLoadReport* report);

// The ggml type ids this arm accepts, so a gate can assert against them without
// hardcoding a magic number.
inline constexpr uint32_t kMusic3GgmlF16 = 1;
inline constexpr uint32_t kMusic3GgmlQ4K = 12;
inline constexpr uint32_t kMusic3GgmlBf16 = 30;

}  // namespace vllm
