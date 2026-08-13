// LTX-2.5 QUANTIZED LOADERS — the FP8 DiT, the torchao-NVFP4 DiT and the
// torchao-NVFP4 Gemma-4 text encoder, materialized from their SHIPPED
// checkpoints onto the contracts phases L2 and L3 already committed.
//
// Row: MODEL-DIFFUSION-LTX25, .agents/specs/ltx-2-5.md phase L6. Issue #435.
//
// ─── NO NEW QUANT SCHEME. ONE DELTA, AND IT IS A SCALE LAYOUT ────────────────
//
// The DiT's FP8 arm is per-tensor E4M3 with an F32 `weight_scale` MULTIPLIER —
// byte for byte what `DequantFp8ToBf16` (nvfp4_dequant.h:76) already consumes.
//
// The text encoder is torchao NVFP4, NOT compressed-tensors. Measured against
// the shipped file rather than assumed: every quantized module carries a
// `torchao_nvfp4` U8[240] marker whose JSON reads
//
//   {"format": "torchao_nvfp4", "block_size": 16, "scope": "full",
//    "config": "NVFP4DynamicActivationNVFP4WeightConfig",
//    "is_swizzled_scales": true, "use_triton_kernel": true,
//    "use_dynamic_activation": true, "use_dynamic_per_tensor_scale": true}
//
// so the encoding is the SAME E2M1 nibble pair + fp8-e4m3 group scale +
// per-tensor f32 global as the modelopt W4A16 path `DequantNvfp4ToBf16`
// (nvfp4_dequant.h:59) implements, with `weight_scale_2` used as a MULTIPLIER
// and not reciprocated — confirmed numerically on the real file, not inferred:
// `model.layers.0.self_attn.q_proj` has weight_scale_2 = 1.3515e-4 and a group
// scale whose maximum is exactly 448.0, so 6 * 448 * 1.3515e-4 = 0.3633 is a
// plausible weight amax. Under the compressed-tensors DIVISOR convention
// (nvfp4_emulation.h:18-23) the same number would imply an amax of 2688/1.35e-4
// = 1.99e7, which no trained weight has. The convention is therefore modelopt's.
//
// The ONE delta is `is_swizzled_scales`: the group-scale tensor is stored in
// the cuBLAS "block scaling factors layout" rather than the linear
// [out, in/16] the two existing paths expect. `Ltx2UnswizzleNvfp4BlockScale`
// inverts exactly the permutation vLLM's own producers apply —
// `swizzle_blockscale` (vllm/model_executor/layers/quantization/utils/
// nvfp4_utils.py:44-49) and `to_blocked`
// (vllm/model_executor/layers/quantization/qutlass_utils.py:165-180), which are
// the same permutation written twice — and hands the result to the UNCHANGED
// `DequantNvfp4ToBf16`. Nothing else about the scheme differs.
//
// ─── AND A SECOND PRODUCER, WITH A SECOND DELTA ──────────────────────────────
//
// Recorded 2026-08-13 (phase L9a, .agents/specs/nvfp4-nibble-order.md). The above
// describes the torchao-quantized TEXT ENCODER. Lightricks' first-party NVFP4
// **DiT** was written by their own `nvfp4-prequant`, and it differs twice:
//
//   * the SAME swizzled bytes are declared in the cuBLAS-PADDED framing
//     [round_up(N,128), round_up(G,4)] rather than torchao's `to_blocked`
//     [32*ceil(N/128), 16*ceil(G/4)]. Same buffer, same permutation, different
//     2-D dress — `Ltx2UnswizzleNvfp4BlockScale` reads both unchanged;
//   * element 2j is in the HIGH nibble, not the low one.
//
// The second is a different byte ENCODING and is why `DequantNvfp4ToBf16` grew an
// `Nvfp4NibbleOrder`. This file used to be REFUSED, with a message diagnosing its
// scale as the LINEAR layout; that diagnosis was wrong — the bytes were swizzled
// all along, and the linear shape is numerically identical to the padded framing
// for every layer here, which is exactly why a shape test could not tell.
// `Ltx2ResolveNvfp4Producer` below is what decides, and what refuses.
//
// ─── WHAT THE STORED SHAPES MEAN, AND THE TRAP IN THEM ───────────────────────
//
// NVFP4 packs TWO values per byte along the LAST dimension, so every U8 width
// in the text encoder's header is HALF the logical one. This campaign already
// shipped that mistake once into a spec (§1.4). The loader therefore NEVER
// reads a packed width as logical: `in_features = stored_cols * 2`, always, and
// the result is cross-checked against the unpacked `model.norm.weight` [3840],
// which is BF16 and so authoritative.
//
// The SWIZZLED scale is the second half of the same trap. Its stored shape is
// [out/4, (in/16)*4] — the same element count as the linear [out, in/16], in a
// different 2-D dress. Reading it as linear type-checks, produces finite
// weights, and permutes every group scale within a 128x64 tile. The loader
// asserts the swizzled shape explicitly and refuses anything else BY NAME.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
//
// The default materialization is **bf16**, which is the checkpoint's own model
// dtype: `DequantFp8ToBf16` / `DequantNvfp4ToBf16` land there natively, the
// biases and norms are stored BF16, and upstream resolves ONE model dtype that
// every layer inherits. A gate cannot catch a dtype that is too WIDE, so f32 is
// never the default here.
//
// The tables (`scale_shift_table` and friends) are the annotated exception:
// they are stored F32 IN THE CHECKPOINT and are kept F32, because narrowing a
// tensor the file itself widened would be this rule applied backwards.
//
// `Ltx2WidenDitToF32` exists for one caller only — `Ltx2DitForward`, which
// phase L2 declared f32-only because f32 is its PARITY dtype against upstream
// run in torch float32 (ltx2.h:33-39). It is opt-in, it is named for what it
// does, and it is not what a production load does.
//
// ─── GB10 RESIDENCY ──────────────────────────────────────────────────────────
//
// `Ltx2StreamDitToDevice` dequantizes and uploads ONE TENSOR AT A TIME, freeing
// each host buffer before the next. Two measured findings force this and it is
// not an option: host/ATS-retagged decode weights run 20-30% slower on GB10, so
// weights are staged at LOAD; and a load-to-host-then-stage of a 21 GB DiT
// holds both copies at once, which is what wedged the box during MiniMax-H3's
// port (minimax_h3.h:1598-1606). Same shape, same reason.
//
// ─── WHAT THE SHIPPED DiT CARRIES THAT PHASE L2 DOES NOT PORT ────────────────
//
// MEASURED 2026-08-12 from the FP8 checkpoint's own header, and reported rather
// than absorbed. The file carries four families outside the L2 contract, and
// they fall into TWO groups that this comment used to conflate — corrected
// 2026-08-13, because the conflation is what made a downstream refusal state
// something untrue about the tree for a whole phase.
//
// UNPORTED. `Ltx2LoadDitFromSafetensors` REFUSES the load by naming these, and
// only an explicit `allow_unported_modules` — which exists so the ported subset
// stays gateable — proceeds, still reporting every one of them in `unported`:
//
//   prompt_adaln_single.*, audio_prompt_adaln_single.*
//       Upstream builds these only when `cross_attention_adaln AND
//       use_prompt_adaln_single` (model.py:222-226, :253-257). Their presence
//       means the shipped LTX-2.5 sets `use_prompt_adaln_single = TRUE`, which
//       contradicts .agents/specs/ltx-2-5.md §1.2 and ltx2.h:115-117 — and with
//       it the prompt-K/V "free win", whose whole premise is that the prompt
//       modulation carries no timestep term.
//   keyframes_abs_pos_embedding  [1, 4096]
//       So `use_keyframes_abs_pos_embedding = TRUE`, contradicting ltx2.h:47-49.
//
// LOADED ELSEWHERE — NOT UNPORTED, and never named in that refusal:
//
//   video_embeddings_connector.*, audio_embeddings_connector.*
//       8 `transformer_1d_blocks` and a `learnable_registers` [128, dim] each.
//       They sit OUTSIDE the DiT contract by design, because upstream loads them
//       into the TEXT ENCODER's `EmbeddingsProcessor` through
//       `EMBEDDINGS_PROCESSOR_KEY_OPS` (encoder_configurator.py:331-346), and
//       phase L9c materializes them here through `Ltx2LoadConnectorWeights`,
//       which the video engine calls on the render path. `LoadedElsewhere`
//       (ltx2_loader.cpp:417-428) is the code that says so.
//
// This paragraph previously listed the connector families among the refused
// ones, citing `ltx2_text_encoder.h`'s "records as owed" note, and phase L10's
// `encoder_path` refusal cited THIS comment as its evidence that the last hop
// could not be taken. By then it was false on both counts: the weights load, and
// the hop is taken (phase L13). Recorded rather than quietly edited, because a
// refusal whose REASON goes stale is the recurring defect of this campaign and
// this is the third instance.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // Nvfp4NibbleOrder
#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_audio_vae.h"
#include "vllm/model_executor/models/ltx2_connector.h"
#include "vllm/model_executor/models/ltx2_text_encoder.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {

class SafetensorsFile;
struct StTensor;

// ---------------------------------------------------------------------------
// torchao NVFP4: the marker, and the one delta
// ---------------------------------------------------------------------------

// The `<module>.torchao_nvfp4` U8 sidecar, parsed. Every field is read rather
// than assumed, and `ParseLtx2TorchaoNvfp4Marker` REFUSES a combination this
// port does not implement — which is what makes "verify the layout before
// reusing anything" a gate and not a comment.
struct Ltx2TorchaoNvfp4Marker {
  std::string format;
  int64_t block_size = 0;
  std::string scope;
  std::string config;
  bool is_swizzled_scales = false;
  bool use_triton_kernel = false;
  bool use_dynamic_activation = false;
  bool use_dynamic_per_tensor_scale = false;
};

// The tensor name this port is keyed on.
inline constexpr const char* kLtx2TorchaoNvfp4MarkerSuffix = ".torchao_nvfp4";

// ---------------------------------------------------------------------------
// WHICH PRODUCER WROTE THIS FILE — .agents/specs/nvfp4-nibble-order.md section 3.2
// ---------------------------------------------------------------------------
//
// The two NVFP4 checkpoints this campaign loads were written by DIFFERENT
// producers that disagree about BOTH the group scale's 2-D framing AND the nibble
// order. Getting either wrong yields finite, correctly shaped, correctly scaled,
// WRONG weights, so the choice is made from evidence and refused when the
// evidence does not fit.
//
//   torchao          `<module>.torchao_nvfp4` marker present, scale stored in the
//                    `to_blocked` framing [32*ceil(N/128), 16*ceil(G/4)],
//                    LOW-nibble-first.  (the Gemma-4 text encoder)
//   nvfp4-prequant   NO marker anywhere, scale stored in the cuBLAS-PADDED framing
//                    [round_up(N,128), round_up(G,4)], HIGH-nibble-first.
//                    (Lightricks' first-party LTX-2.5 DiT)
//
// THE MARKER IS THE DISCRIMINANT, AND ITS ABSENCE IS EVIDENCE, NOT A DEFAULT:
// torchao always emits that sidecar, so a file without one was not written by
// torchao. That is an inference from a producer's signature and it is stated here
// rather than buried, because it is the one step in this path that is not read
// directly off the file. What supports it, measured rather than argued: reading
// the first-party DiT this way correlates 0.9956 (9.46% relative rms, which IS
// NVFP4 quantization error) against the independent vonkaiser FP8 DiT of the same
// base weights, where the alternative readings give 0.0004 to 0.26; and Lightricks'
// own runtime documents exactly this layout and order
// (ltx-kernels/docs/NVFP4.md:27-29, ltx-core/quantization/nvfp4/linear.py:6-7).
//
// THE SHAPE CANNOT DISCRIMINATE ALONE, which is why the marker leads. Every layer
// of the first-party DiT has N % 128 == 0 and G % 4 == 0, so its cuBLAS-padded
// shape is NUMERICALLY IDENTICAL to the linear [N, G] one. A shape test can never
// separate those two. It is used only to CORROBORATE what the marker decided.
//
// WHAT THE REFUSAL DOES AND DOES NOT COVER — state it exactly, because the
// tempting sentence ("any other combination is refused by name") is FALSE and
// would be read as a safety guarantee.
//
// A combination the shape can SEPARATE is refused by name: a marker with the
// to_blocked shape resolves, a marker with any other shape refuses, no marker
// with the to_blocked shape refuses, and no marker with a shape that is neither
// framing refuses. What CANNOT be refused is the case the shape cannot see. A
// marker-less NVFP4 checkpoint whose `weight_scale` is stored LINEAR [N, K/16]
// — never swizzled — presents, for every geometry with N % 128 == 0 and
// G % 4 == 0, a shape numerically identical to the cuBLAS-padded one. It
// therefore resolves kNvfp4Prequant, gets the 128x4 unswizzle applied to scales
// that were never swizzled, and is read high-first. Finite, correctly shaped,
// correctly scaled, WRONG.
//
// That is not an exotic file. LINEAR [N, K/16] is precisely what NVIDIA ModelOpt,
// llm-compressor and compressed-tensors write, and it is what vLLM's own readers
// expect on disk (modelopt.py:1335-1345 and
// compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:73-76 both allocate
// `weight_scale` as [out, in // 16]); none of the three emits a `.torchao_nvfp4`
// sidecar. So the marker's absence excludes torchao and NOTHING ELSE.
//
// It is resolved this way anyway, deliberately, because there is no better
// evidence in the file: the shipped DiT's `__metadata__` carries exactly
// `config`, `gemma_source_checkpoint`, `model_version` and `license` — no
// `quantization_config`, no producer key, and no tensor name mentioning the
// quantizer. Unlike MiniMax-H3, whose community checkpoint DID name its
// converter, there is nothing here to key on. The correlation gate is what
// converts that inference into a measurement, and it is why the gate is the
// result rather than "the forward ran".
//
// The consequence is a tracked condition, recorded in
// .agents/specs/nvfp4-nibble-order.md section 3.1 alongside the H3 one: THIS
// LOADER MUST NOT BE POINTED AT A SECOND MARKER-LESS NVFP4 PRODUCER without
// first re-running the correlation gate against an independent oracle for that
// artifact. It resolves such a file rather than refusing it.
enum class Ltx2Nvfp4Producer {
  kTorchao,        // marker present: to_blocked framing, low-nibble-first
  kNvfp4Prequant,  // marker absent:  cuBLAS-padded framing, high-nibble-first
};

// The two 2-D framings of the SAME swizzled byte buffer. Equal element counts,
// identical layout; `Ltx2UnswizzleNvfp4BlockScale` reads either because it keys
// off the LOGICAL rows/cols and a byte count, not off the declared shape.
std::vector<int64_t> Ltx2Nvfp4ToBlockedScaleShape(int64_t out_features,
                                                  int64_t in_features);
std::vector<int64_t> Ltx2Nvfp4PaddedScaleShape(int64_t out_features,
                                               int64_t in_features);

// Resolve the producer from the marker, corroborated by the stored scale shape.
// `marker` is null when the file declares none for this module.
//
// Throws std::runtime_error naming `module` when the marker and the shape
// disagree, or when the shape matches neither framing — never picking one.
Ltx2Nvfp4Producer Ltx2ResolveNvfp4Producer(const std::string& module,
                                           const Ltx2TorchaoNvfp4Marker* marker,
                                           const std::vector<int64_t>& scale_shape,
                                           int64_t out_features, int64_t in_features);

// The nibble order each producer packs with.
Nvfp4NibbleOrder Ltx2Nvfp4NibbleOrderFor(Ltx2Nvfp4Producer producer);

// Parses the marker's JSON payload. Throws std::runtime_error naming `module`
// on: a non-JSON payload, a `format` other than "torchao_nvfp4", a
// `block_size` other than kNvfp4GroupSize (16), or `is_swizzled_scales` false —
// the last because an UNSWIZZLED torchao checkpoint would need the linear read,
// and silently applying the unswizzle to it permutes every scale.
Ltx2TorchaoNvfp4Marker ParseLtx2TorchaoNvfp4Marker(const std::string& module,
                                                   const StTensor& marker);

// Invert the cuBLAS block-scaling-factors permutation.
//
// Source (vLLM `swizzle_blockscale`, nvfp4_utils.py:44-49): the [rows, cols]
// scale is padded to [round_up(rows,128), round_up(cols,4)], viewed as
// [rows/128, 4, 32, cols/4, 4] and permuted to (0, 3, 2, 1, 4). vLLM's
// `to_blocked` (qutlass_utils.py:165-180) writes the identical permutation a
// second way and is what torchao's own producer matches.
//
// So the element at logical (r, c), with r = 128*rt + 32*a + s and
// c = 4*ct + q, lives at flat offset
//
//   ((((rt * (cols_padded/4)) + ct) * 32 + s) * 4 + a) * 4 + q
//
// `swizzled_bytes` must be exactly round_up(rows,128) * round_up(cols,4);
// `linear` receives rows*cols bytes. Padding rows/cols are never read.
void Ltx2UnswizzleNvfp4BlockScale(const uint8_t* swizzled, size_t swizzled_bytes,
                                  int64_t rows, int64_t cols, uint8_t* linear);

// One NVFP4 quantized module, dequantized to bf16 through the UNCHANGED modelopt
// path: unswizzle the group scale, then `DequantNvfp4ToBf16`.
//
//   packed    U8       [out, in/2]  (in = stored cols * 2), packed per `producer`
//   scale     F8_E4M3  the SWIZZLED group scale, in `producer`'s framing
//   scale_2   F32      scalar       MULTIPLIER
//
// Both scale layouts are the same permutation over the same bytes; only the
// declared 2-D shape and the nibble order differ, and `producer` names which.
// Every one of those three shapes is asserted against `out`/`in` before a byte is
// read, and a mismatch throws naming `module`. `out_bf16` receives out*in bf16 bit
// patterns.
//
// `producer` is NOT defaulted. A default here would let a caller that never
// thought about the question silently get the torchao reading for a
// nvfp4-prequant file, which is precisely the failure this seam exists to refuse.
void Ltx2DequantNvfp4ToBf16(const std::string& module, const StTensor& packed,
                            const StTensor& scale, const StTensor& scale_2,
                            int64_t out_features, int64_t in_features,
                            Ltx2Nvfp4Producer producer, uint16_t* out_bf16);

// ---------------------------------------------------------------------------
// The DiT
// ---------------------------------------------------------------------------

// ComfyUI-format checkpoints — which is what both shipped LTX-2.5 DiTs are —
// prefix every parameter. Stripped before the name meets the L2 contract.
inline constexpr const char* kLtx2DitCheckpointPrefix = "model.diffusion_model.";

// Which quantization the DiT file actually uses, DETECTED from the tensors
// rather than from a filename.
enum class Ltx2DitQuant {
  kFp8,    // F8_E4M3 weight + F32 scalar `<name>_scale`   (vonkaiser 22b-distilled-fp8)
  kNvfp4,  // U8 packed + F8_E4M3 `<name>_scale` + F32 `<name>_scale_2`
};

struct Ltx2DitLoadOptions {
  // Proceed past the unported families named at the top of this header,
  // reporting them in `Ltx2DitCheckpoint::unported` instead of throwing. The
  // ported subset is still bound exactly; nothing is approximated.
  bool allow_unported_modules = false;
  // Widen the bf16 materialization to f32 for `Ltx2DitForward`, whose gate is
  // f32 by declaration. Doubles the footprint; see the DTYPE note above.
  bool widen_to_f32 = false;
};

// A host buffer owned by a loaded checkpoint. Pointer-stable: the views index
// into `bytes`, and the shared_ptr keeps that allocation alive independently of
// how the owning vector grows.
struct Ltx2HostBuffer {
  std::vector<uint8_t> bytes;
  vt::DType dtype = vt::DType::kF32;
};

struct Ltx2DitCheckpoint {
  Ltx2DitQuant quant = Ltx2DitQuant::kFp8;
  // The geometry the FILE describes, including the flags whose modules this
  // port does not carry.
  Ltx2DitParams checkpoint_params;
  // The geometry actually BOUND — `checkpoint_params` with every unported flag
  // cleared. The two differing is exactly what `unported` enumerates.
  Ltx2DitParams params;
  // Module prefixes present in the file and outside the L2 contract, in header
  // order, deduplicated. Empty means the file and the contract agree.
  std::vector<std::string> unported;
  Ltx2DitWeights weights;
  std::map<std::string, vt::Tensor> views;
  // Host-resident buffers (`Ltx2LoadDitFromSafetensors`). Pointer-stable.
  std::vector<std::shared_ptr<Ltx2HostBuffer>> storage;
  // Device allocations (`Ltx2StreamDitToDevice`), each with its backend's Free
  // as the deleter so a staged checkpoint releases exactly like a host one.
  std::vector<std::shared_ptr<void>> device_storage;
};

// Recover `Ltx2DitParams` from a checkpoint header alone, prefix stripped and
// packed widths doubled. Pure manifest work: no payload is touched.
Ltx2DitParams Ltx2ParseDitParamsFromCheckpoint(const SafetensorsFile& file,
                                                Ltx2DitQuant* out_quant = nullptr);

// Materialize the whole DiT onto the L2 contract, HOST-resident.
//
// Every name the contract requires and the file lacks throws BY NAME, so a
// missing tensor can never read as zeros. `EnumerateLtx2DitTensors` supplies
// the required set and its shapes; both are checked before anything is bound.
//
// This is the REFERENCE loader and it materializes the whole model at once
// (~21 GB bf16 for the shipped FP8 DiT). A real GB10 run uses
// `Ltx2StreamDitToDevice`.
Ltx2DitCheckpoint Ltx2LoadDitFromSafetensors(const SafetensorsFile& file,
                                              const Ltx2DitLoadOptions& options = {});

// Widen a bf16 checkpoint in place to f32, rebinding every view. Only for
// `Ltx2DitForward`; see the DTYPE note.
void Ltx2WidenDitToF32(Ltx2DitCheckpoint& checkpoint);

// The GB10 arm: dequantize and upload tensor by tensor, freeing each host
// buffer before the next, so peak residency is the device copy plus ONE tensor.
// Returns the same struct with `views` pointing at DEVICE memory and
// `device_storage` — NOT `storage`, which stays empty on this path — holding the
// device allocations; `widen_to_f32` is refused here because the point of
// staging is not to move twice the bytes.
Ltx2DitCheckpoint Ltx2StreamDitToDevice(vt::Queue& queue, const SafetensorsFile& file,
                                         const Ltx2DitLoadOptions& options = {});

// ---------------------------------------------------------------------------
// The text encoder
// ---------------------------------------------------------------------------

// One caption projection as it comes off the checkpoint: bf16 weight, bf16
// bias. `Ltx2TextFeatureExtractorForward` refuses a config/weights disagreement
// (ltx2_text_encoder.h:260-269), and the bias is loaded here precisely because
// the case it names — reading the U8 weight and missing the BF16 bias — is what
// a two-dtype module invites.
struct Ltx2TextProjection {
  std::vector<uint16_t> weight_bf16;  // [out_features, in_features]
  std::vector<uint16_t> bias_bf16;    // [out_features], empty when bias=False
  int64_t out_features = 0;
  int64_t in_features = 0;
};

struct Ltx2TextEncoderLoadOptions {
  // Mirror upstream's `GemmaAssets.from_single_file` refusal of a file with no
  // `__metadata__`. The shipped checkpoint HAS none, so a caller holding the
  // Gemma config out of band passes false — see ltx2_text_encoder.h:366-373.
  bool require_config = false;
  // Skip the two ~1.5 GB / ~0.8 GB projections and load only the geometry, the
  // assets and the quantized-module inventory. What a manifest gate wants.
  bool skip_projections = false;
};

struct Ltx2TextEncoderCheckpoint {
  // Established from the UNPACKED `model.norm.weight` (BF16, so authoritative),
  // never from a packed U8 width. Cross-checked against both projections.
  int64_t gemma_hidden_size = 0;
  int64_t gemma_num_hidden_layers = 0;
  Ltx2TextProjection video, audio;
  Ltx2GemmaAssets assets;
  // Every module carrying a `torchao_nvfp4` marker, in header order. Includes
  // the full multimodal tower (`vision_model.*`, `multi_modal_projector`,
  // `audio_projector`) the file also ships: text conditioning is the scope, but
  // a loader that CHOKES on their presence cannot read this checkpoint.
  std::vector<std::string> quantized_modules;
};

// Materialize the LTX-specific half of the text encoder — the two caption
// projections, the embedded tokenizer/asset pack and the Gemma geometry — from
// the shipped torchao-NVFP4 file.
//
// Also VALIDATES every quantized module in the file, the Gemma tower included:
// each must carry weight / weight_scale / weight_scale_2 / torchao_nvfp4 with
// shapes consistent under the packed-width and swizzled-scale rules. A module
// that does not throws BY NAME, so an unreadable tower is a load-time refusal
// and not a phase-L7 surprise.
//
// The Gemma TOWER ITSELF is not materialized here: `ltx2_text_encoder.h`
// declares no tower contract (it consumes hidden states through
// `Ltx2TextHiddenStates`, which `Gemma4Model::ForwardHiddenStates` produces).
// Wiring the tower's torchao arm onto `Gemma4Weights` is owed and named as such
// rather than half-done.
Ltx2TextEncoderCheckpoint Ltx2LoadTextEncoderFromSafetensors(
    const SafetensorsFile& file, const Ltx2TextEncoderLoadOptions& options = {});

// Widen the bf16 projections into `Ltx2TextEncoderWeights`, whose f32 is phase
// L3's declared PARITY dtype (ltx2_text_encoder.h:73-83). Opt-in, and ~4.6 GB
// at the shipped widths — which is exactly why it is not what loading does.
Ltx2TextEncoderWeights Ltx2WidenTextProjectionsToF32(
    const Ltx2TextEncoderCheckpoint& checkpoint);

// ---------------------------------------------------------------------------
// The VAEs, the upsampler and the duration head (phase L7)
// ---------------------------------------------------------------------------
//
// Every one of these ships as a plain bf16 safetensors file whose CONFIG rides
// in the file's own `__metadata__["config"]`, exactly as upstream's
// `SafetensorsModelStateDictLoader().metadata(path)` reads it
// (video_vae/model_configurator.py:21-24, audio_vae/model_configurator.py:50,
// :109). So a sibling JSON is not merely unnecessary here — reading one would
// let a caller pair a config with weights it does not describe, which is the
// mis-load this whole seam exists to refuse.

// The parsed `__metadata__["config"]` object. Throws BY NAME when the key is
// absent or is not a JSON object: a VAE whose geometry has to be GUESSED
// decodes to a plausible, finite, wrong picture.
nlohmann::json Ltx2ReadCheckpointConfig(const SafetensorsFile& file);

// Adopt a `{"transformer": {...}}` DiT configuration onto the params the SHAPES
// resolved, or refuse by name.
//
// WHAT THIS DECIDES, and why it is not cosmetic. `Ltx2ParseDitParamsFromManifest`
// reads shapes, which is the only thing a ComfyUI-flavoured checkpoint offers. A
// config states what no shape encodes — `frequencies_precision`,
// `av_ca_timestep_scale_multiplier`, the positional-embedding bounds and theta,
// `norm_eps`, `use_middle_indices_grid`. Each moves every RoPE angle or every
// modulation while leaving the tensor set byte-identical, so the manifest path
// resolves a DIFFERENT MODEL from the same file and nothing downstream can tell.
//
// A config is believed only when `EnumerateLtx2DitTensors` over it reproduces the
// IDENTICAL weight contract `from_shapes` produces. That is what proves it
// describes THIS file rather than another checkpoint's config pasted beside it;
// a disagreement is refused rather than resolved in either direction, because
// taking the shapes renders with the wrong RoPE and taking the config binds the
// wrong tensors.
//
// ONE FUNCTION, because two callers must not answer this differently: the video
// engine (`Ltx2VideoEngine::Load`) and the device gate, which drives
// `Ltx2StreamDitToDevice` directly and therefore owes the same adoption.
//
// `allow_unported_modules` clears `use_keyframes_abs_pos_embedding` IN A COPY of
// the config before parsing, mirroring what the loader does for
// `use_prompt_adaln_single`: the flag is cleared for the CONTRACT, the module
// stays unported, and the checkpoint's `unported` list still names it. Without
// the opt-in `ParseLtx2DitParams` throws, which is the refusal.
//
// `source` names the config in every refusal, so a reader knows whether the
// checkpoint declared it or a caller supplied it.
Ltx2DitParams Ltx2AdoptDeclaredDitParams(const nlohmann::json& config,
                                         const Ltx2DitParams& from_shapes,
                                         bool allow_unported_modules,
                                         const std::string& source);

// ---------------------------------------------------------------------------
// The embeddings connector (phase L9c)
// ---------------------------------------------------------------------------
//
// The two `*_embeddings_connector` families live in the DiT FILE but not in the
// DiT's own weight contract: upstream loads them into the TEXT ENCODER's
// `EmbeddingsProcessor`, through `EMBEDDINGS_PROCESSOR_KEY_OPS`, which rewrites
// `model.diffusion_model.video_embeddings_connector.` to `video_connector.`
// (text_encoders/gemma/encoders/encoder_configurator.py:331-346). They are
// therefore loaded HERE, beside the DiT and not inside it, and they stay outside
// `EnumerateLtx2DitTensors`.

enum class Ltx2ConnectorStream { kVideo, kAudio };

// The tensor-name prefix each stream's family carries in the checkpoint, WITHOUT
// the ComfyUI `model.diffusion_model.` prefix (`PlanDit` strips that).
const char* Ltx2ConnectorCheckpointPrefix(Ltx2ConnectorStream stream);

// Does this DiT file carry a connector at all? Keyed on `learnable_registers`,
// the one tensor the family always has and the only one that is not per-block.
bool Ltx2CheckpointHasConnector(const SafetensorsFile& file, Ltx2ConnectorStream stream);

// `Embeddings1DConnectorConfigurator.from_metadata` and its audio twin
// (embeddings_connector.py:194-256) applied to a `{"transformer": {...}}` object.
//
// FOUR VALUES NO SHAPE ENCODES, and one of them is not close to its default:
// LTX-2.5 declares `connector_positional_embedding_max_pos = [4096]` where the
// class default is `[1]`. `get_fractional_positions` divides the token index by
// it (rope.py:132-141), so the default turns every position into a fractional
// position 4096x too large and every RoPE angle with it. The others are
// `rope_type`, `frequencies_precision` (-> double-precision frequencies) and
// `connector_apply_gated_attention`.
//
// `positional_embedding_theta` is deliberately NOT read from the config even
// though the DiT declares one: neither configurator passes it, so upstream's
// connector always runs at the class default of 10000.0. Reading the DiT's key
// would be a re-invention, not a port — and the shipped file declares 10000.0
// anyway, so the two agree and only the RULE differs.
Ltx2ConnectorConfig Ltx2ParseConnectorConfig(const nlohmann::json& config,
                                             Ltx2ConnectorStream stream);

// Materialize one connector family out of the DiT checkpoint, widened to f32 —
// which is `Ltx2ConnectorForward`'s declared parity dtype.
//
// IT IS NOT CHEAP AND THE CALLER MUST TREAT IT AS EXPENSIVE. 129 tensors is 8
// blocks of four dim x dim projections plus a 4x-wide feed-forward, so at the
// shipped widths the video family is ~1.61G parameters and the audio family
// ~0.40G: **about 8 GB of f32 together**. That is small against the DiT's 21 GB
// on disk and 44 GB staged, and it is NOT small on a 119 GB unified-memory box
// that reboots rather than OOM-killing. The video engine therefore loads these,
// runs the connector once, and drops them inside one scope — the conditioning is
// resolved at load time, so the weights never outlive their single use.
//
// It is also the CONTRACT CHECK on the config. `Ltx2ParseConnectorConfig` reads
// values that mostly cannot be seen in a shape — but `connector_num_layers`,
// `connector_apply_gated_attention`, `connector_ff_bias` and the head geometry
// all CAN, so every enumerated tensor must exist at its enumerated shape AND no
// tensor of the family may be left over. A config that says 2 layers against a
// file carrying 8 is refused by name here rather than binding the first two and
// rendering.
Ltx2VaeWeights Ltx2LoadConnectorWeights(const SafetensorsFile& file,
                                        const Ltx2ConnectorConfig& config);

// `__metadata__["model_version"]` ("2.5.0"), which is what
// `detect_model_version` reads to pick a recipe (ltx-pipelines
// utils/constants.py:161) and what `should_use_ancestral_sampler` keys on
// (distilled.py:84). Empty when the file declares none — never defaulted to a
// generation, because defaulting one picks a sigma schedule that renders.
std::string Ltx2ReadCheckpointModelVersion(const SafetensorsFile& file);

// One `SDOps` key rule: a name is KEPT when it starts with `match_prefix`, and
// that prefix is then rewritten to `replacement`. Mirrors
// `SDOps.with_matching(prefix=...)` + `.with_replacement(from, to)`
// (loader/sd_ops.py), which is how every shipped VAE file's ComfyUI-flavoured
// names are turned into the module's own `state_dict` names. A rule set is
// applied IN ORDER and the first matching rule wins, so the longer prefixes come
// first exactly as upstream's chained filters list them.
//
// A name matched by NO rule is dropped, which is upstream's behaviour and is
// what lets one file hold the encoder, the decoder and the vocoder at once.
struct Ltx2VaeKeyRule {
  std::string match_prefix;
  std::string replacement;
};

// The three shipped filters, verbatim:
//   VAE_DECODER_COMFY_KEYS_FILTER        video_vae/model_configurator.py:255-265
//   AUDIO_VAE_DECODER_COMFY_KEYS_FILTER  audio_vae/model_configurator.py:184-190
//   VOCODER_COMFY_KEYS_FILTER            audio_vae/model_configurator.py:91-103
// The vocoder's rule strips `vocoder.` EXACTLY ONCE (`removeprefix`, :91-97), so
// `vocoder.vocoder.conv_pre` becomes `vocoder.conv_pre` and NOT `conv_pre` —
// which is the difference between the BWE chain finding its two generators and
// binding both of them to one name.
std::vector<Ltx2VaeKeyRule> Ltx2VideoVaeDecoderKeyRules();
std::vector<Ltx2VaeKeyRule> Ltx2AudioVaeDecoderKeyRules();
std::vector<Ltx2VaeKeyRule> Ltx2VocoderKeyRules();

// Every tensor a rule set keeps, widened to f32 and keyed by the REWRITTEN name,
// which is `Ltx2VaeWeights`' whole contract (ltx2_audio_vae.h:60-70). BF16 and
// F32 are the only dtypes these files carry; anything else throws by name rather
// than being reinterpreted. An empty rule set keeps every name unchanged.
//
// f32 here is NOT a widening of a production path: `Ltx2VaeWeights` is declared
// f32 by phases L4/L5 because f32 is their parity dtype, and this materializes
// onto that declared contract. The VAEs together are ~1.8 GB bf16, so the
// widened copy is ~3.7 GB — small next to the DiT, and the reason the DiT does
// NOT take this path.
Ltx2VaeWeights Ltx2LoadVaeWeights(const SafetensorsFile& file,
                                  const std::vector<Ltx2VaeKeyRule>& rules = {});

// `_build_conv_video_decoder` (video_vae/model_configurator.py:81-94) applied to
// `config["vae"]`, plus `_vae_class_name_from_metadata` (:21-24) recovered into
// `*out_kind` so a `CausalDiffusionVAE` checkpoint is REFUSED by
// `Ltx2VideoDecode` rather than decoded by the conv arm.
Ltx2ConvVideoDecoderConfig Ltx2ParseConvVideoDecoderConfig(const nlohmann::json& config,
                                                            Ltx2VideoDecoderKind* out_kind);

// `AudioDecoderConfigurator.from_metadata` (audio_vae/model_configurator.py:108-141).
Ltx2AudioDecoderConfig Ltx2ParseAudioDecoderConfig(const nlohmann::json& config);

// `LatentUpsamplerConfigurator.from_metadata` (upsampler/model_configurator.py:14-21).
// Unlike the two VAEs this config is FLAT — the shipped
// `ltx-2.5-latent-spatial-upscaler-x2` writes its keys at the top level of
// `__metadata__["config"]`, with no `vae` wrapper — so it is read from the
// object itself. `temporal_upsample` is left as the file states it and refused
// downstream by `Ltx2LatentUpsample`, which is where the missing arm lives.
Ltx2UpsamplerConfig Ltx2ParseUpsamplerConfig(const nlohmann::json& config);

// `VocoderConfigurator.from_metadata`, the BWE branch
// (audio_vae/model_configurator.py:49-88). The legacy flat branch (:53-56) is
// REFUSED by name: it is the pre-2.3 `resblock == "1"` vocoder, no LTX-2.5
// checkpoint carries it, and silently building it would emit audio from the
// wrong generator. Every `check_config_value` upstream asserts is asserted here.
Ltx2VocoderBweConfig Ltx2ParseVocoderBweConfig(const nlohmann::json& config);

}  // namespace vllm
