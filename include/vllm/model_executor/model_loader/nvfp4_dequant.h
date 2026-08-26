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
// UE4M3 x0.5 LUT trap (.agents/specs/gguf-nvfp4-notes.md) does NOT apply here.
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

// WHICH LOGICAL ELEMENT SITS IN WHICH NIBBLE — a PRODUCER convention, and the two
// producers this project reads DISAGREE. See .agents/specs/nvfp4-nibble-order.md.
//
// Read the wrong way round, every adjacent fp4 pair is transposed. The result is
// finite, correctly shaped and correctly scaled, so nothing downstream notices:
// a decoder emits fluent tokens, a DiT renders a plausible-but-wrong frame.
//
// This parameter IS defaulted, to kLowFirst, and that is the deliberate design:
// every caller predating .agents/specs/nvfp4-nibble-order.md consumes a ModelOpt
// or compressed-tensors checkpoint, which is low-first, so the default makes
// "this change moved nothing else" true BY CONSTRUCTION rather than by
// inspection. Four callers rely on it today — minimax_h3_nvfp4.cpp:112,
// minimax_h3_device.cpp:1311, qwen3_5.cpp:1298 and
// dense_nvfp4_gemm.h DequantNvfp4ToBLayout — and H3 reaches low-first by
// normalizing its bytes at load (MiniMaxH3Nvfp4SwapNibbles) rather than by
// passing an order.
//
// The seam where the order is NEVER defaulted is `Ltx2DequantNvfp4ToBf16`
// (ltx2_loader.h), which takes a resolved `Ltx2Nvfp4Producer` with no default,
// because that is the path where the two conventions actually meet and a default
// would let a caller that never thought about it get the silent wrong answer.
enum class Nvfp4NibbleOrder {
  // Element 2j in the LOW nibble, 2j+1 in the HIGH. torchao's `pack_uint4`
  // (torchao/prototype/mx_formats/kernels.py:160,
  // `uint8_data[::2] | uint8_data[1::2] << 4`), which is what NVIDIA ModelOpt and
  // compressed-tensors checkpoints carry, and what vLLM's own reader assumes
  // (`break_fp4_bytes`, nvfp4_emulation_utils.py:321-324: `low = a_flat & 0x0F`
  // then `torch.stack((low, high))`). THE DEFAULT — every caller predating
  // .agents/specs/nvfp4-nibble-order.md means this one.
  kLowFirst,
  // Element 2j in the HIGH nibble. Lightricks' `nvfp4-prequant`, which wrote the
  // first-party LTX-2.5 NVFP4 DiT (ltx-kernels/docs/NVFP4.md:27-29: "`hi_first=True`
  // (default) puts element `2j` in the **high** nibble of byte `j`"; the same
  // statement at ltx-core/quantization/nvfp4/linear.py:6).
  //
  // MiniMax-H3's community NVFP4 checkpoints are also high-first and are handled a
  // DIFFERENT way — `MiniMaxH3Nvfp4SwapNibbles` normalizes the bytes at load
  // (minimax_h3.h:1500-1517) because H3 also feeds a Marlin fp4-RESIDENT path, and
  // one byte transform fixes both arms where a host-dequant flag fixes only one.
  // Both mechanisms are deliberate; nvfp4-nibble-order.md section 3.1 records which
  // to use when, and the condition under which LTX-2.5 must switch to H3's.
  kHighFirst,
};

// Decode one IEEE fp8-e4m3fn byte (1 sign, 4 exp, 3 mantissa; bias 7; no inf;
// NaN = 0x7F/0xFF; 0x00 = +0) to f32. Matches
// torch.Tensor.view(torch.float8_e4m3fn).to(torch.float32).
float F8E4M3ToF32(uint8_t byte);

// Dequantize a modelopt W4A16_NVFP4 weight matrix to bf16 (row-major bit
// patterns in out_bf16).
//
//   packed            [out_dim, in_dim/2]  U8, packed per `order`
//   weight_scale_fp8  [out_dim, in_dim/16] fp8-e4m3fn bytes, linear layout
//   weight_scale_2    per-tensor f32 global scale (multiplied, not reciprocated)
//   out_bf16          [out_dim, in_dim]    bf16 bit patterns (caller-owned)
//   order             which nibble holds element 2j; DEFAULTS to the torchao /
//                     ModelOpt convention, so every caller written before
//                     .agents/specs/nvfp4-nibble-order.md is unchanged BY
//                     CONSTRUCTION rather than by inspection
//
// Requires in_dim % 16 == 0. Aborts (VT_CHECK) otherwise.
//
// Does NO internal buffer-size bounds checking: it validates only nulls, dim
// signs, and in_dim % 16. The caller must size packed, weight_scale_fp8, and
// out_bf16 to the dims above and should derive those dims from StTensor.shape
// (the safetensors reader validates every span).
void DequantNvfp4ToBf16(const uint8_t* packed, const uint8_t* weight_scale_fp8,
                        float weight_scale_2, int64_t out_dim, int64_t in_dim,
                        uint16_t* out_bf16,
                        Nvfp4NibbleOrder order = Nvfp4NibbleOrder::kLowFirst);

// Dequantize a modelopt per-tensor FP8 (W8A16) weight to bf16. The 35B gate
// checkpoint stores its attention/GDN projections this way (hf_quant_config
// quant_algo "FP8"): a full E4M3 byte per element plus ONE f32 per-tensor
// weight_scale (multiplied, not reciprocated); the sibling input_scale is a
// W8A8 activation-scale placeholder and is UNUSED for the weight-only path.
//
//   weight_f8   [numel]  IEEE fp8-e4m3fn bytes (torch Linear layout [out,in])
//   scale       per-tensor f32 (applied directly)
//   out_bf16    [numel]  bf16 bit patterns (caller-owned, same numel)
//
//   out[i] = bf16( f8_e4m3(weight_f8[i]) * scale )   // f32 mul, RNE
//
// No unpacking, no group scale — distinct from the NVFP4 path above.
void DequantFp8ToBf16(const uint8_t* weight_f8, float weight_scale,
                      int64_t numel, uint16_t* out_bf16);

// Channel-wise FP8 (compressed-tensors / llm-compressor FP8_DYNAMIC weights):
//   weight_f8     [N, K]  F8_E4M3
//   scale_bf16    [N] or [N,1]  bf16 per-output-channel scale
//   out_bf16      [N, K]
//   out[n,k] = bf16( f8(w[n,k]) * bf16_to_f32(scale[n]) )
void DequantFp8ChannelToBf16(const uint8_t* weight_f8, const uint16_t* scale_bf16,
                             int64_t N, int64_t K, uint16_t* out_bf16);

// Block-wise (fine-grained) FP8 weight -> f32. The DeepSeek recipe
// (`weight_block_size: [128, 128]`, `fmt: e4m3`, `scale_fmt: ue8m0`) stores one
// E4M3 byte per element beside a UE8M0 scale byte per BLOCK:
//
//   weight_f8    [N, K]                              F8_E4M3, row-major
//   scale_e8m0   [ceil(N/bn), ceil(K/bk)]            F8_E8M0, row-major
//   out_f32      [N, K]                              caller-owned
//
//   out[n,k] = F8E4M3ToF32(weight_f8[n,k])
//              * E8M0ToF32(scale_e8m0[n / bn][k / bk])
//
// EVERY convention here is taken from the arm that already EXECUTES this format
// in this tree rather than re-derived: `vt::MatmulFp8BlockScaledKernel`
// (`src/vt/cpu/cpu_ops.cpp`), the port of upstream's `native_w8a8_block_matmul`
// (`tests/kernels/quant_utils.py:91-154`). The scale MULTIPLIES and is not a
// reciprocal (`c += matmul(a, b.t()) * s`, `:150-151`); the scale row is indexed
// by OUTPUT ROW divided by the block extent (`offs_bsn = offs_bn // group_n`,
// `fp8_utils.py:823`); and a ragged final block is legal rather than tolerated,
// because upstream's wrapper asserts the CEIL shapes (`fp8_utils.py:935-936`).
//
// f32 out, not bf16, and that is the one deliberate difference from the two
// functions above: the caller is DeepSeek-V4's host-float weight tower
// (`DeepseekV4HostWeights`, all `std::vector<float>`), so a bf16 round here
// would be a narrowing this consumer immediately widens again.
void DequantFp8BlockToF32(const uint8_t* weight_f8, const uint8_t* scale_e8m0,
                          int64_t N, int64_t K, int64_t block_n, int64_t block_k,
                          float* out_f32);

// Nesting guards: expert-level parallel prefetch serializes row-parallel dequant.
void Fp8DequantBeginOuterParallel();
void Fp8DequantEndOuterParallel();

}  // namespace vllm
