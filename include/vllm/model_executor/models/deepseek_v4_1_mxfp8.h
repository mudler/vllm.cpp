// DeepSeek-V4.1-Flash W3b — the MXFP8 32x32 UE8M0 linear family, as a portable
// host (CPU) reference. Five pieces, each ported 1:1 with `file:line`, all @ the
// V4.1 head `e77daef89e` (the revision `.agents/specs/deepseek-v4-1-flash.md`
// `## Dependencies` names; W3b needs no pin advance, per that spec's wave table).
//
// ─── THE HEADLINE: "32x32" IS A CHECKPOINT LAYOUT, NOT A RUNTIME ONE ─────────
// A reader who takes `weight_block_size: [32, 32]` at face value will build a
// two-dimensional block dequant and be wrong about every V4.1 linear. The scale
// the checkpoint ships is `[N/32, K/32]`; the scale the runtime holds is
// `[N, K/32]`, an ORDINARY MXFP8 per-32-column scale. The loader bridges the two
// by expanding the ROWS:
//
//   modelopt.py:2186-2201 (`KMxfp8Static.get_scale_weight_loader`)
//     loaded_weight.view(torch.uint8).repeat_interleave(block_rows, dim=0)
//     ... then hands the result to the ordinary weight_loader.
//
// and the same file states the runtime shape twice over: `create_weights`
// registers `weight_scale` at
// `(output_size_per_partition, input_size_per_partition // 32)`
// (modelopt.py:2222-2234) and then sets `layer.weight_block_size = [1, 32]`
// (modelopt.py:2235). Both `QuantKey`s agree: `kMxfp8StaticScale` and
// `kMxfp8DynamicScale` are each `ScaleDesc(uint8, ..., GroupShape(1, 32))`
// (quant_utils.py:231-235, :246-247). So the 32x32 is only how the FILE stores
// the scale, and 32 is the only legal column block — `block_cols != 32` raises
// `NotImplementedError` (modelopt.py:2189-2192).
//
// **The expansion happens BEFORE tensor-parallel slicing, and the order is
// load-bearing.** `scaled_loader` expands and only then calls `weight_loader`,
// which is what shards. Sharding first is not merely a different spelling: a
// V4.1 shared-expert projection is 96 rows padded to 128, so its checkpoint scale
// has 3 rows, which does not divide by a TP world of 2 at all. `## Tests` below
// gates the order directly.
//
// ─── WHAT LANDS HERE ────────────────────────────────────────────────────────
//   (1) ExpandMxfp8CheckpointScale  — the `repeat_interleave(block_rows, dim=0)`
//                                     row expansion, `[N/32, K/32]` -> `[N, K/32]`,
//                                     refusing `block_cols != 32` and
//                                     `block_rows < 1` BY NAME.
//   (2) DequantMxfp8ToF32 / ToBf16  — the RUNTIME-layout dequant,
//                                     `out[n,k] = f8(w[n,k]) * 2^(s[n, k/32] - 127)`
//                                     (mxfp8_utils.py:230-243,
//                                     `dequant_mxfp8_to_bf16`). Bf16 is the
//                                     model-path emitter and the one the linear
//                                     arm uses; f32 exists for the equivalence
//                                     gate and for a double-precision reference,
//                                     NOT for a model-path buffer.
//   (3) QuantizeMxfp8E4m3           — the DYNAMIC activation quantizer,
//                                     `sb = clamp(ceil(log2(amax / 448)) + 127, 0, 254)`,
//                                     applied as `x * 2^(127 - sb)`
//                                     (mxfp8_utils.py:126-142).
//   (4) Mxfp8LinearEmulationBf16    — the EMULATION linear arm: dequant the
//       (+ Mxfp8LinearEmulation)      weight to bf16 once, then a plain bf16
//                                     linear (emulation.py:16-84). The bf16
//                                     entry point is the MODEL-PATH arm; the f32
//                                     one beside it is the gate's reference and
//                                     is the third named divergence below.
//   (5) Mxfp8LinearScaleParamName   — the checkpoint `weight_scale` /
//                                     `weight_scale_inv` name split
//                                     (nvidia/model.py:871-882).
//
// ─── REUSE, NOT RE-PORT ─────────────────────────────────────────────────────
// The CHECKPOINT-layout reference is already in this tree and is not re-written
// here: `DequantFp8BlockToF32` at `bn = bk = 32`
// (`include/vllm/model_executor/model_loader/nvfp4_dequant.h:160`) computes
// exactly the `weight.reshape(n/32, 32, k/32, 32) * exp2(scale - 127)` the
// upstream test builds its reference from. The E8M0 decode is
// `vllm::E8M0ToF32` (`mxfp4_dequant.cpp:15-22`), which already implements the
// ARITHMETIC `exp2(byte - 127)` form that mxfp8 uses; it is reused unchanged.
// (Upstream carries a SECOND, BITCAST E8M0 decode for Engram at
// `deepseek_v4_1/common/engram.py:613-614`, which differs from this arithmetic
// one at byte 0. That decode belongs to W3a and nothing here touches it or
// `E8M0ToF32`. It is deliberately cited by upstream `file:line` and not by a
// local issue ID: the Engram issue is W3a's to file, and a reference to an ID
// this tree does not carry fails `check-agent-record`.) The e4m3 round is
// `vllm::F32ToF8E4M3`
// (`compressed_tensors/nvfp4_emulation.h:50`), which mirrors torch
// `.to(torch.float8_e4m3fn)` over the range this family can reach, and NOT at
// the very top of it: the local codec saturates every `|a| >= 448` to `0x7E`
// (`nvfp4_emulation.cpp:F32ToF8E4M3`), while c10's `fp8e4m3fn_from_fp32_value`
// takes `fp8_max = 480.0f` and returns `0x7F` — NaN — above that. The two agree
// below 480 (c10 rounds `(448, 480)` down to 448 as well) and differ only for
// `|a| >= 480` and for a non-finite input. `QuantizeMxfp8E4m3` cannot reach
// either from a FINITE activation, because the scale byte puts the block amax in
// `(224, 448]` and every element of the block is at or below the amax. An
// INFINITE activation can: it clamps `sb` to 254, and `inf * 2^-127` is still
// `inf`, which this codec encodes as 448 where torch would encode NaN. That is a
// live difference on a path no finite input reaches, so it is recorded here
// rather than asserted away; W5's device arm sees the real torch codec.
//
// ─── THREE DELIBERATE DIVERGENCES, ALL NAMED ────────────────────────────────
// * **Scale byte 0xFF.** `E8M0ToF32` returns NaN, because 0xFF is the OCP UE8M0
//   NaN encoding. Upstream's `torch.exp2(255 - 127)` returns +inf instead. The
//   quantizer clamps `sb` to 254 so it cannot PRODUCE 0xFF, and only a
//   hand-written checkpoint could carry one; we keep the OCP reading rather than
//   fork a second E8M0 decode. Re-confirm this against the device arm at W5.
// * **Multiply, not divide.** Upstream has two quantizers. The torch reference
//   `_mxfp8_e4m3_quantize_torch` (mxfp8_utils.py:66-67) DIVIDES by
//   `exp2(sb - 127)`; the Triton kernel (mxfp8_utils.py:126-136) MULTIPLIES by
//   `exp2(127 - sb)` and its own comment says why: `sb == 0` makes that divisor
//   `2**-127`, which is subnormal in fp32 and flushes to zero on CDNA, turning a
//   zero block into `0/0 == NaN`. Both are exact powers of two, so they agree
//   bit-for-bit on every host WITHOUT flush-to-zero — which is NOT "for every
//   reachable sb": `sb == 0` is exactly the case upstream's comment separates,
//   and upstream's own wording is the weaker "nothing else changes"
//   (mxfp8_utils.py:133-134). We mirror the MULTIPLY, which is the
//   form that is correct on every backend this project targets. **No host gate
//   can hold this choice**, and that was measured rather than assumed: a true
//   divide leaves the unit suite GREEN with the binary proved changed (M9 in the
//   `## Mutations` table of `ISSUE-LOCAL-01M2C2QSFZWCXNQBJFBWYFNV2P`, which
//   carries the digest beside the build recipe that produced it). It is
//   a source-level decision, and W5's CDNA arm owes the measurement.
// * **An f32 emulation arm beside the bf16 one.** `emulation.py:62-63` is
//   `F.linear(x, weight.to(x.dtype), bias).to(x.dtype)` with `x` the resolved
//   model dtype, which V4.1 makes bf16 (`test_fp8.py:83`). `Mxfp8LinearEmulationBf16`
//   IS that arm and is the model-path one: bf16 in, f32 accumulate, bf16 out.
//   `Mxfp8LinearEmulation` keeps an f32 signature beside it and has no upstream
//   counterpart. Its reason is the one `DequantMxfp8ToF32` already has: it is the
//   double-precision REFERENCE the unit gate compares the bf16 arm against, and
//   it is what a host-float weight tower (`DeepseekV4HostWeights`, all
//   `std::vector<float>`) can call without a widen-narrow round trip. **It is not
//   a model-path buffer**, and W4 wires the bf16 arm, not this one. AGENTS.md is
//   explicit that a token gate cannot see a dtype that is too wide, so the split
//   is stated here and asserted in the gate rather than left to a reader.
//
// ─── WHY A HOST REFERENCE, AND WHAT IS NOT REACHED (honest scope) ───────────
// No published V4.1 artifact fits any device this project reaches (the release
// is 475.27 GiB against `dgx:gpu0`'s 119 GiB), so no end-to-end token gate
// against vLLM exists for this family — see the spec's `## Work breakdown`,
// constraint 1. W3b therefore lands the MATH, unit-gated against the upstream
// test's own bit-exact oracle plus from-first-principles double-precision
// references, exactly as V4's W3-W7 host references were gated.
//
// **Nothing in this file has a production caller yet.** `deepseek_v41` IS
// registered — W1 landed that — but its forward, loader, prepare and KV-cache
// hooks all refuse by name, so no production entry point reaches these
// functions. The wiring is owned by row
// `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`: W8 (the loader arm that
// calls (1) and (5)) and W4 (the host forward assembly that calls (2) and (4)).
// Tracked by `ISSUE-LOCAL-01M2C2QSFZWCXNQBJFBWYFNV2P` and recorded under
// `## Owed` in `.agents/specs/deepseek-v4-1-flash.md`.
//
// ─── WHAT THIS IS A PORT OF (file:line on both sides, @ vLLM e77daef89e) ─────
//   OURS                         <-  UPSTREAM (vllm/)
//   ExpandMxfp8CheckpointScale   <-  model_executor/layers/quantization/
//                                    modelopt.py:2186-2201
//                                    (`KMxfp8Static.get_scale_weight_loader`)
//   DequantMxfp8To{F32,Bf16}     <-  .../utils/mxfp8_utils.py:230-243
//                                    (`dequant_mxfp8_to_bf16`)
//   QuantizeMxfp8E4m3            <-  .../utils/mxfp8_utils.py:126-142
//                                    (`_mxfp8_quant_triton_kernel`), whose
//                                    amax/clamp/bias mirror
//                                    :56-69 (`_mxfp8_e4m3_quantize_torch`)
//   Mxfp8LinearEmulation         <-  model_executor/kernels/linear/mxfp8/
//                                    emulation.py:16-84
//                                    (`EmulationMxfp8LinearKernel`)
//   Mxfp8LinearScaleParamName    <-  models/deepseek_v4_1/nvidia/
//                                    model.py:871-882
//                                    (`_linear_scale_param_name`)
//   the ROUTING that selects all  <- models/deepseek_v4_1/
//   of the above                     quant_config.py:186-201
//                                    (`DeepseekV4FP8Config.get_quant_method`)
//   the ported unit gate          <-  tests/quantization/test_fp8.py:67-238
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vllm::deepseek_v4_1 {

// `MXFP8_BLOCK_SIZE` (mxfp8_utils.py:11). The scale group is 32 ELEMENTS ALONG
// K, in both the checkpoint and the runtime layout.
inline constexpr int64_t kMxfp8BlockSize = 32;

// `torch.finfo(torch.float8_e4m3fn).max` (mxfp8_utils.py:61). The quantizer puts
// the block amax at the TOP of this range rather than at 1.0, or the small
// elements of the block land in the fp8 subnormals (mxfp8_utils.py:123-125).
inline constexpr float kMxfp8Fp8Max = 448.0F;

// (1) `KMxfp8Static.get_scale_weight_loader` (modelopt.py:2186-2201): expand the
// CHECKPOINT scale's rows into the RUNTIME layout, and do it BEFORE any
// tensor-parallel or padding-aware slicing.
//
//   ckpt_scale  [ckpt_rows, ckpt_cols]              UE8M0 bytes, row-major
//   returns     [ckpt_rows * block_rows, ckpt_cols] UE8M0 bytes, row-major
//
// Each checkpoint row is repeated `block_rows` times CONTIGUOUSLY
// (`repeat_interleave`, not `tile`/`repeat`): output row `n` reads checkpoint row
// `n / block_rows`. For V4.1, `block_rows == block_cols == 32`, so a `[N/32, K/32]`
// checkpoint scale becomes the `[N, K/32]` the runtime registers.
//
// REFUSES BY NAME, mirroring modelopt.py:2189-2192: `block_cols` must be exactly
// `kMxfp8BlockSize`, and `block_rows` must be at least 1. `block_rows == 1` is
// legal and is the identity (upstream returns the unwrapped loader in that case,
// modelopt.py:2201).
std::vector<uint8_t> ExpandMxfp8CheckpointScale(const uint8_t* ckpt_scale,
                                                int64_t ckpt_rows, int64_t ckpt_cols,
                                                int64_t block_rows, int64_t block_cols);

// (2) `dequant_mxfp8_to_bf16` (mxfp8_utils.py:230-243) on the RUNTIME layout:
//
//   weight_f8   [N, K]      F8_E4M3 bytes, row-major
//   scale_e8m0  [N, K/32]   UE8M0 bytes, row-major  (K must divide by 32)
//   out         [N, K]      caller-owned
//
//   out[n,k] = F8E4M3ToF32(weight_f8[n,k]) * E8M0ToF32(scale_e8m0[n][k / 32])
//
// Note the scale row index is `n`, NOT `n / 32`: this is the layout AFTER (1).
// Feeding an unexpanded `[N/32, K/32]` checkpoint scale here reads the wrong row
// for 31 of every 32 outputs and is silently plausible, which is why (1) exists
// as its own named step.
void DequantMxfp8ToF32(const uint8_t* weight_f8, const uint8_t* scale_e8m0, int64_t N,
                       int64_t K, float* out_f32);

// Bf16 emitter: the same math, narrowed once at the store. This is the
// MODEL-PATH emitter — upstream's `dequant_mxfp8_to_bf16` returns bf16 and the
// emulation kernel stores bf16 (emulation.py:45-48), so an f32 weight buffer
// here would move twice the bytes for no numerical gain.
void DequantMxfp8ToBf16(const uint8_t* weight_f8, const uint8_t* scale_e8m0, int64_t N,
                        int64_t K, uint16_t* out_bf16);

// (3) The DYNAMIC activation quantizer (`kMxfp8Dynamic` is `KDynamicNoParam`:
// nothing is stored, the kernel quantizes at runtime — modelopt.py:2253-2262,
// :2278). Ported from the Triton kernel `_mxfp8_quant_triton_kernel`
// (mxfp8_utils.py:113-142), whose amax/bias/clamp are identical to the torch
// reference at :56-69:
//
//   amax = max(|x[m, b*32 : b*32+32]|, TINY)      TINY = fp32 smallest normal
//   sb   = clamp(ceil(log2(amax / 448)) + 127, 0, 254)      stored as one byte
//   q    = to_e4m3(x * 2^(127 - sb))              MULTIPLY; see the header note
//
//   x        [M, K]      f32, K divisible by 32
//   out_q    [M, K]      F8_E4M3 bytes, caller-owned
//   out_s    [M, K/32]   UE8M0 bytes, caller-owned
void QuantizeMxfp8E4m3(const float* x, int64_t M, int64_t K, uint8_t* out_q,
                       uint8_t* out_s);

// (4) `EmulationMxfp8LinearKernel` (emulation.py:16-84). The emulation backend
// dequantizes the weight to bf16 ONCE at load and then runs a plain bf16 linear;
// it does NOT quantize the activation, so (3) has no caller on this path and is
// exercised by the native backends instead. The bf16 narrowing of the weight is
// therefore part of the arm's numerics, not an implementation detail:
//
//   out[m,n] = bias[n] + sum_k x[m,k] * bf16(f8(w[n,k]) * 2^(s[n, k/32] - 127))
//
// THE MODEL-PATH ARM. `F.linear(x, weight.to(x.dtype), bias).to(x.dtype)` with
// `x` bf16 (emulation.py:59-63; V4.1 resolves the model dtype to bf16,
// test_fp8.py:83). Accumulation is f32 and the narrowing happens ONCE at the
// store, which is what `F.linear` on bf16 operands does.
//
//   x       [M, K]     bf16 bit patterns
//   weight  [N, K]     F8_E4M3 bytes; scale [N, K/32] UE8M0, the RUNTIME layout
//   bias    [N] bf16, or nullptr
//   out     [M, N]     bf16 bit patterns, caller-owned
void Mxfp8LinearEmulationBf16(const uint16_t* x_bf16, int64_t M, int64_t K,
                              const uint8_t* weight_f8, const uint8_t* scale_e8m0,
                              int64_t N, const uint16_t* bias_bf16,
                              uint16_t* out_bf16);

// The f32 REFERENCE beside it — see the third named divergence above. Same math,
// same bf16 weight narrowing, f32 in and f32 out, so the gate can compare the
// model-path arm against a recompute without a widen-narrow round trip. NOT a
// model-path buffer: W4 wires `Mxfp8LinearEmulationBf16`.
//
//   x       [M, K]     f32
//   weight  [N, K]     F8_E4M3 bytes; scale [N, K/32] UE8M0, the RUNTIME layout
//   bias    [N] or nullptr
//   out     [M, N]     f32, caller-owned; accumulation is f32
void Mxfp8LinearEmulation(const float* x, int64_t M, int64_t K, const uint8_t* weight_f8,
                          const uint8_t* scale_e8m0, int64_t N, const float* bias,
                          float* out_f32);

// (5) `_linear_scale_param_name` (nvidia/model.py:871-882): which parameter name
// the checkpoint's `.scale` keys map onto. NATIVE MXFP8 — `[32, 32]` blocks with
// MXFP4 experts — routes linears through `ModelOptLinearMethod`, which registers
// `weight_scale`; every other combination is block-FP8 and registers
// `weight_scale_inv`. Both halves of the conjunction are load-bearing, and
// upstream's own test parametrizes exactly the three cases that prove it
// (test_fp8.py:202-210).
std::string Mxfp8LinearScaleParamName(const std::vector<int64_t>& weight_block_size,
                                      std::string_view expert_dtype);

}  // namespace vllm::deepseek_v4_1
