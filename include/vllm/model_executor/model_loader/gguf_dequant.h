// vllm.cpp original GGUF-format dequant loader (porting-inventory.md §9
// deviation, like the safetensors reader / gguf_reader.{h,cpp}). No upstream
// vLLM mirror: this is the ggml GGUF quant block format, ported byte-for-byte
// from llama.cpp @ 237ad9b961f009ae19ac29dbce4cd0c1251f94b3:
//
//   ggml/src/ggml-common.h  — block_q4_0, block_q8_0, block_q3_K, block_q4_K,
//                             block_q5_K, block_q6_K struct layouts
//   ggml/src/ggml-quants.c  — dequantize_row_q4_0 / _q8_0 / _q3_K / _q4_K /
//                             _q5_K / _q6_K + get_scale_min_k4 (the 6-bit
//                             super-block scale/min unpack)
//
// The K-quant super-block is 256 elements = 8 sub-blocks of 32 (Q4_K/Q5_K) or
// 16 blocks of 16 (Q3_K/Q6_K); the packed 6-bit scales/mins are the subtle
// part (get_scale_min_k4 for Q4_K/Q5_K, the kmask aux shuffle for Q3_K, plain
// int8 for Q6_K). A wrong bit-unpack yields garbage weights.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {

// Dequantize `numel` elements from the packed GGUF block bytes at `data` to
// f32. `ggml_type` is the ggml type id (see enum ggml_type / GgufValueType).
// `numel` MUST be a multiple of the type's block_elems (GgmlTraits(type)) —
// GGUF rows always are. Supported types: F32(0), Q4_0(2), Q8_0(8), Q3_K(11),
// Q4_K(12), Q5_K(13), Q6_K(14). Any other id (e.g. IQ2_S(22)/IQ4_XS(23))
// throws std::runtime_error("unsupported ggml type N (Task 2/i-quant)").
//
// Reads exactly numel/block_elems * block_bytes bytes from `data`; the caller
// (the GGUF loader) must have validated the tensor span (gguf_reader does).
std::vector<float> DequantGgufRowToF32(uint32_t ggml_type, const uint8_t* data,
                                       int64_t numel);

// Same as above but returns bf16 bit patterns (dequant to f32, then
// vt::F32ToBF16 round-to-nearest-even). The Qwen3.6 loader (Task 2) targets
// bf16 OwnedTensors, matching the safetensors path.
std::vector<uint16_t> DequantGgufRowToBf16(uint32_t ggml_type,
                                           const uint8_t* data, int64_t numel);

}  // namespace vllm
