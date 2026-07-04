// Ported from: vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
//   and     : vllm/model_executor/layers/quantization/compressed_tensors/schemes/
//             compressed_tensors_w4a4_nvfp4.py
//   and     : vllm/model_executor/kernels/linear/nvfp4/emulation.py @ e24d1b24
//
// CPU reference for the compressed-tensors **NVFP4 W4A4** scheme (the 27B dense
// gate, unsloth/Qwen3.6-27B-NVFP4). This is the software-emulation fallback
// (`EmulationNvFp4LinearKernel.apply_weights`): dequant the fp4 weight to high
// precision, dynamically quantize+dequantize the activation to fp4, then a plain
// f32 matmul. It is the CPU-truth the future GB10 fp4xfp4 GEMM is validated
// against — NOT the throughput path.
//
// W4A4 vs the modelopt W4A16 path (nvfp4_dequant.h) — same E2M1/fp8-group-scale
// weight encoding, but TWO deviations, both load-bearing:
//
//  1. Global-scale storage. compressed-tensors stores BOTH global scales as
//     DIVISORS (~= FP8_MAX*FP4_MAX/amax = 2688/amax), i.e. the reciprocal of the
//     modelopt `weight_scale_2` multiplier. So:
//       - WEIGHT dequant multiplies by 1/weight_global_scale
//         (compressed_tensors_w4a4_nvfp4.py:111-113 pre-reciprocates the param).
//       - ACTIVATION quant uses input_global_scale DIRECTLY (the on-disk divisor;
//         emulation.py:41 passes layer.input_global_scale_inv, the un-reciprocated
//         value). This asymmetry is intentional — see the .cpp.
//  2. Activations are ALSO fp4 (num_bits 4, group_size 16, dynamic "local"
//     per-token, fp8-e4m3 block scales) — a runtime quant step the W4A16 path
//     does not have.
//
// On-disk tensor names differ too: weight_packed / weight_scale /
// weight_global_scale / input_global_scale (CT) vs weight / weight_scale /
// weight_scale_2 / input_scale (modelopt). See .agents/qwen27b-w4a4-notes.md §3.
#pragma once

#include <cstdint>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // kNvfp4GroupSize, F8E4M3ToF32, kE2M1Lut

namespace vllm {

// Max magnitude of E2M1 (fp4): scalar_types.float4_e2m1f.max() == 6.0
// (nvfp4_emulation_utils.py:17). Its reciprocal is the per-block scale factor.
inline constexpr float kFloat4E2M1Max = 6.0F;

// Max finite magnitude of IEEE fp8-e4m3fn (no inf; 2^8 * 1.75). The activation
// block scale is clamped to +/-448 before the fp8 round (ref_nvfp4_quant:434).
inline constexpr float kFloat8E4M3Max = 448.0F;

// Round an f32 to the nearest fp8-e4m3fn byte (round-to-nearest-even, SATURATING
// to +/-448 — "FN" = finite, no inf; NaN only at 0x7F/0xFF). Inverse of
// F8E4M3ToF32 (nvfp4_dequant.h). Mirrors torch `.to(torch.float8_e4m3fn)`.
uint8_t F32ToF8E4M3(float f);

// cast_to_fp4 (nvfp4_emulation_utils.py:413-424): round a scaled f32 to the
// nearest E2M1 value (returned as f32, sign preserved) via the CT reference's
// fixed half-open bucket boundaries. Input is expected pre-clamped to [-6, 6].
float CastToFp4(float x);

// ref_nvfp4_quant + dequant round-trip (nvfp4_emulation_utils.py:427-466) over
// a row-major [m, n] f32 activation (n % block_size == 0). Emulates the dynamic
// per-token, per-16-group activation quantization the W4A4 GEMM applies, then
// dequantizes back to f32 (x_dq ~= x). `input_global_scale` is the ON-DISK
// divisor value (used directly, NOT reciprocated). Writes x_dq [m, n].
void RefNvfp4QuantDequant(const float* x, int64_t m, int64_t n,
                          float input_global_scale, float* x_dq,
                          int block_size = kNvfp4GroupSize);

// compressed-tensors NVFP4 weight dequant to f32 (dequantize_to_dtype with
// swizzle=False — the on-disk weight_scale layout is LINEAR [out, in/16]).
//
//   packed                  [out_dim, in_dim/2]   U8, low-nibble-first E2M1
//   weight_scale_fp8        [out_dim, in_dim/16]  fp8-e4m3fn bytes (linear)
//   weight_global_scale_disk per-tensor f32, the ON-DISK divisor; reciprocated
//                            internally (CT stores 1/scale)
//   out_f32                 [out_dim, in_dim]      caller-owned
//
// Requires in_dim % 16 == 0 (VT_CHECK).
void DequantCtNvfp4WeightToF32(const uint8_t* packed,
                               const uint8_t* weight_scale_fp8,
                               float weight_global_scale_disk, int64_t out_dim,
                               int64_t in_dim, float* out_f32);

// run_nvfp4_emulations (nvfp4_emulation_utils.py:469-495): the full emulated
// W4A4 linear. out[m, out_dim] = x_dq[m, in_dim] @ w_dq[out_dim, in_dim]^T,
// where x_dq is the activation round-trip and w_dq the dequantized weight.
// Both global scales are the ON-DISK values (weight reciprocated internally,
// input used directly). f32 throughout; caller owns out [m, out_dim].
void RunNvfp4Emulation(const float* x, int64_t m, int64_t in_dim,
                       const uint8_t* packed, const uint8_t* weight_scale_fp8,
                       float weight_global_scale_disk,
                       float input_global_scale_disk, int64_t out_dim,
                       float* out, int block_size = kNvfp4GroupSize);

}  // namespace vllm
