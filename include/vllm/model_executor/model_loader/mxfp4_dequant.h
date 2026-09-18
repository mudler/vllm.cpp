// Ported from:
//   vllm/model_executor/layers/quantization/compressed_tensors/schemes/
//     compressed_tensors_w4a4_mxfp4.py:20-97  (CompressedTensorsW4A4Mxfp4:
//     group_size 32, uint8 weight_packed [N,K/2], uint8 E8M0 weight_scale
//     [N,K/32], NO global scale)
//   vllm/model_executor/layers/quantization/utils/mxfp8_utils.py:61-65,222
//     (the E8M0 scale semantics: byte = biased exponent, bias 127;
//      descale = exp2(byte - 127))
//   tests/quantization/reference_mxfp4.py:28-117  (dq_mxfp4_torch — the
//     upstream numerical GOLDEN this file mirrors 1:1: e8m0_to_half +
//     upcast_fp4_to_fp16_or_bf16 + per-32-group multiply)
//   @ pin 555967922 (vLLM 0.26.0.dev0)
//
// CPU reference for the compressed-tensors **MXFP4** (`mxfp4-pack-quantized`)
// WEIGHT dequant. This is the shared unblocker both DeepSeek-V4-Flash
// (W6 MegaMoE MXFP4 experts) and Kimi-K3 (its real checkpoint is MXFP4) need to
// LOAD their weights; this file OWNS the quant path, not those models.
//
// MXFP4 vs our existing NVFP4 (nvfp4_dequant.h / nvfp4_emulation.cpp) — the two
// deviations that make MXFP4 a DISTINCT scheme, both load-bearing:
//
//   1. GROUP SIZE 32 (NVFP4 is 16). One block scale per 32 consecutive input
//      elements, so weight_scale is [out, in/32] (NVFP4: [out, in/16]).
//   2. E8M0 / UE8M0 block scales, NOT fp8-e4m3. The scale byte is a bare biased
//      exponent (8 exp bits, 0 mantissa, 0 sign, bias 127): the real multiplier
//      is 2^(byte - 127) — an exact power of two. NVFP4 stores an fp8-e4m3 byte
//      decoded via F8E4M3ToF32 AND multiplied by a per-tensor global scale.
//      MXFP4 has **NO global scale** (nvfp4 has weight_global_scale). So the
//      MXFP4 dequant is simply e2m1_lut[nibble] * 2^(scale_byte - 127).
//
// The E2M1 4-bit packing is IDENTICAL to NVFP4: two nibbles per byte,
// element 2i = low nibble, element 2i+1 = high nibble; bit 3 = sign, bits 0..2
// index kE2M1Lut {0,.5,1,1.5,2,3,4,6} (nvfp4_dequant.h). Reused verbatim.
//
// The GPU true-W4A4 fp4xfp4 GEMM (FlashInfer CUTLASS on SM100+, else Marlin
// FP4 W4A16 — kernels/linear/mxfp4/{flashinfer,marlin}.py, selected by
// init_mxfp4_linear_kernel) is a NAMED LATER BRICK; this reference is the
// CPU-truth those kernels are validated against, NOT the throughput path.
#pragma once

#include <cstdint>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // kE2M1Lut

namespace vllm {

// Block (group) size for the MXFP4 `mxfp4-pack-quantized` recipe: one E8M0
// scale per 32 consecutive input elements
// (compressed_tensors_w4a4_mxfp4.py:37 `self.group_size = 32`).
inline constexpr int kMxfp4GroupSize = 32;

// Decode one OCP E8M0 (UE8M0) scale byte to f32. E8M0 = 8 biased-exponent bits,
// no mantissa, no sign, bias 127: value = 2^(byte - 127) (an exact power of two).
// Mirrors mxfp8_utils.py:222 `descale = torch.exp2(scales.to(f32) - 127.0)` and
// reference_mxfp4.py:31-34. byte 0xFF is the E8M0 NaN encoding (OCP MX spec);
// QAT scales are always finite, but it is returned as NaN, not silently 2^128.
float E8M0ToF32(uint8_t byte);

// Decode one UE8M0 scale byte to f32 by BITCAST, the second of the two decodes
// upstream uses. `vllm/models/deepseek_v4_1/common/engram.py:613-614` is
// `scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)`, with the
// comment "ue8m0 is a power of two, so its byte *is* the fp32 exponent field".
//
// The two forms agree for every byte in [1, 254] and differ at both ends:
//   byte 0    bitcast -> +0.0            arithmetic -> 2^-127
//   byte 255  bitcast -> +inf            arithmetic -> NaN (the OCP encoding)
// Neither is wrong; they belong to different arms. MXFP8 decodes arithmetically
// (`quantization/utils/mxfp8_utils.py:66`, and `:129-130` comments that it is
// aware `sb == 0` yields 2^-127), and `E8M0ToF32` above serves that arm and its
// two production callers — `mxfp4_dequant.cpp:63` and `nvfp4_dequant.cpp:122`,
// counting invocations of `vllm::E8M0ToF32` under `src/` and `include/` and
// excluding tests, comments and the unrelated `E8M0ToF32Half` /
// `DE8M0ToF32Half` GGUF helpers. An earlier revision of this comment said
// "six", which nothing in the tree supported.
// Engram decodes by bitcast and needs +0.0 at byte 0. Do NOT merge
// the two: nothing upstream pins them against each other, so no ported test
// would catch a wrong choice. Row
// `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`, issue
// `ISSUE-LOCAL-01M2C40RNXB871VW0E560PVBFA`.
float E8M0BitsToF32(uint8_t byte);

// Dequantize a compressed-tensors MXFP4 weight matrix to bf16.
//
//   packed          [out_dim, in_dim/2]   U8, two E2M1 (fp4) per byte,
//                                          element 2i = low nibble, 2i+1 = high
//   weight_scale    U8 (E8M0)  [out, in/32]  one E8M0 byte per 32-elem group
//                                          (LINEAR on-disk layout, no swizzle)
//   out_bf16        [out_dim, in_dim]      bf16 bit patterns (caller-owned)
//
//   out[o, i] = bf16( e2m1_lut[nibble(o, i)] * 2^(weight_scale[o, i/32] - 127) )
//
// The multiply is by an exact power of two, so (barring overflow) the bf16 round
// only ever re-homes the E2M1 exponent and is exact for finite scales. No global
// scale (the MXFP4 vs NVFP4 distinction). Requires in_dim % 32 == 0 (VT_CHECK).
void DequantMxfp4ToBf16(const uint8_t* packed, const uint8_t* weight_scale_e8m0,
                        int64_t out_dim, int64_t in_dim, uint16_t* out_bf16);

// Same MXFP4 dequant, emitted as f32 (the exact pre-bf16-round values). This is
// the high-precision operand the future emulated / GPU W4A4 GEMM consumes and
// the double-precision unit reference compares against. Requires in_dim % 32.
void DequantMxfp4ToF32(const uint8_t* packed, const uint8_t* weight_scale_e8m0,
                       int64_t out_dim, int64_t in_dim, float* out_f32);

}  // namespace vllm
