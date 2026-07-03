// Ported from: vllm/model_executor/layers/quantization/modelopt.py (NVFP4 W4A16 dequant) @ e24d1b24
//
// ModelOpt W4A16_NVFP4 weight-only dequant utility. Materializes a bf16 weight
// matrix from the on-disk modelopt tensors that M0.9 weight loading consumes:
//
//   weight          U8       [out, in/2]   two 4-bit E2M1 (fp4) per byte,
//                                          element 2i = low nibble, 2i+1 = high
//   weight_scale    F8_E4M3  [out, in/16]  one IEEE fp8-e4m3fn per 16-elem group
//                                          (LINEAR layout on disk, NO swizzle),
//                                          LINEAR scale (multiply, not reciprocal)
//   weight_scale_2  F32      scalar        per-tensor global scale = amax/2688,
//                                          used directly (NO reciprocation)
//
// Dequant, matching nvfp4_emulation_utils.dequantize_to_dtype (swizzle=False)
// exactly, including f32 grouping order so the bf16 round is bit-exact:
//
//   scale[o, g] = f32(weight_scale[o, g]) * weight_scale_2      // f32
//   out[o, i]   = bf16( e2m1_lut[nibble(o, i)] * scale[o, i/16] )  // f32 mul, RNE
//
// input_scale is present in W4A4-shaped checkpoints but UNUSED here (W4A16).
//
// NOTE: modelopt scales are standard IEEE fp8-e4m3fn. The GGUF killgate fork's
// UE4M3 x0.5 LUT trap (.agents/gguf-nvfp4-notes.md) does NOT apply here.
#pragma once

#include <cstdint>

namespace vllm {

// Block (group) size for the modelopt NVFP4 recipe: one fp8 scale per 16
// consecutive input elements. Hardcoded per the W4A16_NVFP4 format.
inline constexpr int kNvfp4GroupSize = 16;

// E2M1 (fp4) magnitude LUT: index = 3 low magnitude bits (nibble & 0x7); the
// sign is nibble bit 3 (nibble & 0x8). 1x-scaled floats
// (nvfp4_emulation_utils.py:20-22).
inline constexpr float kE2M1Lut[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                      2.0F, 3.0F, 4.0F, 6.0F};

// Decode one IEEE fp8-e4m3fn byte (1 sign, 4 exp, 3 mantissa; bias 7; no inf;
// NaN = 0x7F/0xFF; 0x00 = +0) to f32. Matches
// torch.Tensor.view(torch.float8_e4m3fn).to(torch.float32).
float F8E4M3ToF32(uint8_t byte);

// Dequantize a modelopt W4A16_NVFP4 weight matrix to bf16 (row-major bit
// patterns in out_bf16).
//
//   packed            [out_dim, in_dim/2]  U8, low-nibble-first packing
//   weight_scale_fp8  [out_dim, in_dim/16] fp8-e4m3fn bytes, linear layout
//   weight_scale_2    per-tensor f32 global scale (multiplied, not reciprocated)
//   out_bf16          [out_dim, in_dim]    bf16 bit patterns (caller-owned)
//
// Requires in_dim % 16 == 0. Aborts (VT_CHECK) otherwise.
void DequantNvfp4ToBf16(const uint8_t* packed, const uint8_t* weight_scale_fp8,
                        float weight_scale_2, int64_t out_dim, int64_t in_dim,
                        uint16_t* out_bf16);

}  // namespace vllm
