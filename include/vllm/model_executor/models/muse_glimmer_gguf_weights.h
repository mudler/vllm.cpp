// vllm.cpp ORIGINAL GGUF-format Muse Glimmer loader. GGUF is its OWN translation
// unit, never an afterthought bolted onto the safetensors loader
// (`.agents/porting-a-model.md`, "What complete looks like"); this file is the
// Muse sibling of `qwen3_5_gguf_weights.{h,cpp}` and mirrors its shape.
//
// WHY THIS EXISTS. A model port covers the quantized arms — a 30B bf16 checkpoint
// is ~60 GB against a ~17 GB k-quant, and the k-quant is what most users can
// actually run. `LoadMuseGlimmer` used to throw on `ModelSource::Kind::kGguf`,
// which was never a decision: it simply was not on any list.
//
// ─── WHAT THIS IS A PORT OF ──────────────────────────────────────────────────
// There is no upstream vLLM GGUF load format at the pin, so — like `gguf_reader`,
// `gguf_dequant` and the Qwen GGUF loader — this is a recorded ORIGINAL
// (porting-inventory.md §9). The tensor NAMES and metadata KEYS mirror
// llama.cpp's `muse-glimmer` arch (PR ggml-org/llama.cpp#26841, MERGED
// 2026-08-10; `src/llama-arch.cpp` LLM_ARCH_MUSE_GLIMMER + LLM_KV_*), which is
// the secondary C++ reference the Muse spec §0 already declares. llama.cpp is
// NEVER the correctness oracle and never a speed denominator.
//
// ─── THE FOUR CONVERT-TIME TRANSFORMS THIS INVERTS ───────────────────────────
// Each was VERIFIED byte-for-byte against `meta-models/Muse-Glimmer-30B`
// (bf16 safetensors, 1436 tensors) on 2026-08-11, not inferred from the names:
//
//  1. SANDWICH NORMS ARE STORED PRE-OFFSET. The converter bakes Muse's `+1`
//     weight offset into the file, so GGUF `blk.N.attn_norm.weight` equals the
//     safetensors `input_layernorm.weight` PLUS ONE
//     (layer 0 element 0: 1.09619141 == 0.09619141 + 1, and the four sandwich
//     norms agree elementwise). Our forward adds the `+1` itself via
//     `vt::RmsNormArgs{eps, gemma=true}`, and our safetensors loader stores the
//     raw HF value, so this loader SUBTRACTS ONE to land on the same
//     `OwnedTensor`. Skipping the un-shift makes every norm weight ~1.0 larger,
//     which produces fluent-but-wrong text rather than an error. `output_norm`
//     (the FINAL norm) takes NO offset in the model and is therefore stored raw
//     — it must NOT be un-shifted.
//
//  2. THE QUERY PRE-SCALE IS FOLDED INTO `attn_q_norm`. Muse's per-head QK-norm
//     is WEIGHTLESS (muse_glimmer.py:1121) and the `scale_query_by` pre-scale is
//     a separate scalar (:1192). ggml has no weightless RMSNorm, so the
//     converter materializes both as weight vectors: `blk.N.attn_k_norm.weight`
//     is all ONES (the identity) and `blk.N.attn_q_norm.weight` is the CONSTANT
//     `scale_query_by` (3.87 on the released 30B, matching
//     `text_config.qk_scale_factor` in config.json exactly). The GGUF ships no
//     metadata key for the pre-scale, so it is RECOVERED from that tensor. A
//     non-constant `attn_q_norm`, or a `attn_k_norm` that is not ones, would be a
//     genuinely weighted QK-norm this model does not have, and is REFUSED rather
//     than silently averaged away.
//
//  3. THE iRoPE MASK RIDES `attention.sliding_window_pattern`. `true` at layer i
//     means RoPE AND sliding-window; `false` means NoPE AND full attention —
//     the same split the safetensors config encodes twice (`layer_rope_theta[i]
//     == 0` and `layer_types[i] == "full_attention"`), verified to agree on the
//     released checkpoint (NoPE at 3, 7, ... 51 for L = 52).
//
//  4. `attn_q` / `attn_k` ARE STORED IN ggml's INTERLEAVED-RoPE ROW ORDER.
//     llama.cpp's converter applies `LlamaModel.permute`
//     (`conversion/llama.py:163-169`), i.e. per head
//     `w.reshape(heads, 2, Dh/2, K).swapaxes(1, 2)`, to the query and key
//     projections — with `n_head` for q and `n_head_kv` for k. That is the
//     weight-side half of ggml's `rope_norm`, which rotates ADJACENT channel
//     pairs (2i, 2i+1), whereas HF — and our `vt::RopeNeox` — rotates
//     HALF-OFFSET pairs (i, i + Dh/2). The loader therefore UN-permutes both
//     shards on the way into the merged `qkv_proj`; `attn_v`, `attn_output` and
//     the MLP shards are stored verbatim and are taken verbatim.
//
//     VERIFIED against the released bf16 checkpoint on 2026-08-11 at layers 0, 3,
//     25 and 51: read verbatim, GGUF `attn_q`/`attn_k` disagree with
//     `meta-models/Muse-Glimmer-30B` at mean relative error ~1.40 (unrelated
//     numbers); read through the inverse map they agree at ~0.077 — exactly the
//     Q4_K quantization noise every other tensor in the file shows.
//
//     This one hides from every structural check — right names, right shapes,
//     right counts, right dtypes — and from attention itself on the NoPE layers,
//     because a permutation applied to BOTH q and k leaves q·k unchanged. It only
//     bites where RoPE runs. Issue #359: the released 17 GB k-quant loaded, ran a
//     forward, and emitted degenerate text rather than failing.
//
// ─── WHAT IS NOT ESTABLISHED ─────────────────────────────────────────────────
// No e2e, no token-exact and NO SPEED claim of any kind. The pinned oracle
// cannot load `muse_glimmer` at all (specs/muse-glimmer.md §0), so there is
// neither a golden nor a throughput denominator for this model in either weight
// format; every performance axis is an open gap by construction. The evidence
// here is structural (name map, shapes, tensor accounting) plus the byte-level
// value checks above.
#pragma once

#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/muse_glimmer.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// llama.cpp's arch string for the Muse Glimmer TEXT tower. The perception
// encoder ships as a SEPARATE `clip`/`mmproj` file and the drafter as a separate
// `dflash` file, exactly as llama.cpp splits every multimodal family.
inline constexpr const char* kMuseGlimmerGgufArch = "muse-glimmer";

// True when this file's `general.architecture` is the Muse Glimmer text tower.
// Used by the entrypoint's GGUF architecture dispatch.
bool IsMuseGlimmerGguf(const GgufFile& gguf);

// Build the HfConfig from a `muse-glimmer` GGUF's metadata, the GGUF counterpart
// of the entrypoint's `config.json` read. Emits the CANONICAL nested layout
// (`text_config`), so `ParseMuseGlimmerParams` consumes it through exactly the
// same path a safetensors checkpoint takes — there is no second config schema.
//
// `scale_query_by` is recovered from `blk.0.attn_q_norm.weight` (transform 2
// above) and published as an explicit `text_config.scale_query_by`, which wins
// outright in `ResolveMuseGlimmerQueryPreScale`.
//
// NAMED RESIDUAL — `post_norm_eps`. The GGUF carries ONE epsilon
// (`muse-glimmer.attention.layer_norm_rms_epsilon`, 1e-5 on the released file)
// and no key for the post-norm epsilon, which the safetensors config ships
// separately as 1e-8. `ParseMuseGlimmerParams` therefore falls back to
// `rms_norm_eps` for the post-norms. The two differ only inside
// `1/sqrt(mean_square + eps)`; with a mean square of order 1 that is a ~5e-6
// relative change, roughly two orders of magnitude below bf16's ~4e-3 spacing,
// so it is not representable in the activation dtype. Recorded rather than
// hidden: it is a real difference from the safetensors arm, just not an
// observable one.
//
// Throws std::runtime_error naming the key on a missing required kv or a
// non-`muse-glimmer` architecture.
HfConfig MuseGlimmerHfConfigFromGguf(const GgufFile& gguf);

// CANONICAL (post-`NormalizeMuseGlimmerWeightName`) name -> GGUF tensor name.
// Returns false when the canonical name has NO counterpart in a text-tower GGUF
// — which is every `vision_*` name, because the perception encoder lives in the
// separate mmproj file (see `MuseGlimmerMmprojRefusal`).
bool MuseGlimmerGgufTensorName(const std::string& canonical, std::string* out);

// The GGUF tensor names a `muse-glimmer` file must contain for these resolved
// params — the accounting DENOMINATOR, in file order-independent form.
//
// It is NOT simply the image of `EnumerateMuseGlimmerTensors` under the name map:
// the two `attn_{q,k}_norm` vectors per layer exist ONLY in the GGUF, because
// what they encode (a weightless norm, and a scalar pre-scale) carries no tensor
// on our side. Enumerating them here is what lets the structural gate demand
// "every tensor in the file is accounted for, zero unaccounted" in BOTH
// directions instead of quietly tolerating strangers.
std::vector<std::string> EnumerateMuseGlimmerGgufTensors(
    const MuseGlimmerParams& params);

// Recover `scale_query_by` from the folded `attn_q_norm` vectors (transform 2).
// Verifies EVERY layer: each `attn_q_norm` must be a single constant, every
// `attn_k_norm` must be all ones, and all layers must agree on the constant.
// Throws naming the offending layer otherwise.
double MuseGlimmerGgufQueryPreScale(const GgufFile& gguf, int64_t num_layers,
                                    int64_t head_dim);

// Load the Muse Glimmer TEXT tower from a `muse-glimmer` GGUF into the SAME
// `MuseGlimmerWeights` the safetensors loader produces, so the shared forward
// (muse_glimmer.cpp) is unchanged. Only the SOURCE and the tensor NAMES differ.
//
// RESIDENCY — what is kept quantized and what is not, and why:
//
//   KEPT (raw ggml blocks, borrowed or copied per `GgufLoadPolicy`):
//     o_proj, output_gate_proj, down_proj — standalone matmul operands consumed
//     by `vt::MatmulBT` in the file's native [N, K] order.
//     gate_up_proj — the merged [2I, H] SwiGLU operand, kept as a BLOCK CONCAT
//     when `ffn_gate` and `ffn_up` share one ggml type (they do on the released
//     17 GB file: both Q4_K). A k-quant row is a whole number of superblocks
//     (K = 6656 = 26 x 256 for Q4_K), so appending one tensor's rows to the
//     other's is a byte concatenation and nothing is requantized.
//
//   DEQUANTIZED to bf16, each for a stated structural reason — not a preference:
//     qkv_proj — the forward wants ONE merged [Hq*Dh + 2*Hkv*Dh, H] operand, and
//     the file's `attn_{q,k,v}` may carry DIFFERENT ggml types per shard (on the
//     released 17 GB file `attn_v` is Q6_K while `attn_q`/`attn_k` are Q4_K).
//     Block encodings of different types cannot share one tensor, so a
//     heterogeneous trio is expanded. A homogeneous trio is kept as a block
//     concat, same as gate_up.
//     lm_head — the forward consumes an UNTIED head through `vt::Matmul` in
//     Matmul-B [H, vocab] orientation, and a block encoding cannot be transposed
//     without requantizing.
//     embed_tokens — a [vocab, H] gather table read row-wise by the embedding
//     kernel, not a GEMM operand.
//     the norms — [H] and [Dh] vectors, F32 on disk, and the `-1` un-shift is a
//     value transform that a block encoding could not carry anyway.
//
// `policy` null reads the process environment (`GgufLoadPolicy::FromEnv` —
// VT_CPU_REF / VT_GGUF_KEEP_QUANT / VT_GGUF_MMAP), matching every other GGUF
// loader in the tree. Throws naming the tensor on a missing name, a shape
// mismatch, or an unsupported encoding.
MuseGlimmerWeights LoadMuseGlimmerFromGguf(const GgufFile& gguf,
                                           const HfConfig& config,
                                           const GgufLoadPolicy* policy = nullptr);

// ─── REFUSED-AND-RECORDED: the perception encoder ────────────────────────────
// The released `mmproj-kquant.gguf` (arch `clip`, 809 tensors) cannot build our
// perception tower, and this is a property of the FILE, not of our loader.
//
// Our `conv1_linear` is a Linear over patchified input with
// `patch_temporal * 3 * patch_size^2` = 2*3*14*14 = 1176 input features, which is
// exactly the shape the safetensors ships
// (`model.vision_tower.patch_embedder.patch_embedding.weight` = [1536, 1176]).
// The mmproj's `v.patch_embd.weight` is ggml ne [14, 14, 3, 1536], i.e. torch
// [1536, 3, 14, 14] = [1536, 588] — HALF the input features, with the
// `patch_temporal` axis absent. Every other tower tensor maps cleanly
// (`v.blk.N.{ln1,ln2,attn_q,attn_k,attn_v,attn_out,ffn_up,ffn_down}`,
// `v.{pre_ln,post_ln,position_embd}`, `mm.{0,1,2}` -> adapter fc1/fc2 +
// vision_projection), so this is one missing axis and not a naming problem.
//
// Loading it would mean inventing the temporal half of a weight, so the mmproj
// arm REFUSES BY NAME instead. This is an OWED item on the row, not a design
// decision, and the fix is upstream in the llama.cpp converter.
[[noreturn]] void MuseGlimmerRefuseMmproj();

}  // namespace vllm
