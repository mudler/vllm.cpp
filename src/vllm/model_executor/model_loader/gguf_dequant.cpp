// vllm.cpp original GGUF-format dequant loader (porting-inventory.md §9
// deviation). Ported byte-for-byte from llama.cpp @
// 237ad9b961f009ae19ac29dbce4cd0c1251f94b3:
//   ggml/src/ggml-common.h  (block_q4_0/q8_0/q3_K/q4_K/q5_K/q6_K layouts)
//   ggml/src/ggml-quants.c  (dequantize_row_* + get_scale_min_k4).
// The BLOCK decoders themselves now live one layer down in vt
// (src/vt/cpu/cpu_quant_dequant.cpp) and are shared with the compute-in-quant
// GEMM; this file keeps the GGUF-facing validation, the unquantized types, and
// the ggml-type -> vt::DType routing. See gguf_dequant.h for the format
// citation.
#include "vllm/model_executor/model_loader/gguf_dequant.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"
#include "vt/quant.h"

namespace vllm {
namespace {

// Read a little-endian ggml_half (f16) at byte pointer `p` and widen to f32.
// (Aligned load is not guaranteed for mmap'd block bytes, so memcpy.)
float ReadF16(const uint8_t* p) {
  uint16_t h = 0;
  std::memcpy(&h, p, sizeof(h));
  return vt::F16ToF32(h);
}

}  // namespace

std::vector<float> DequantGgufRowToF32(uint32_t ggml_type, const uint8_t* data,
                                       int64_t numel) {
  VT_CHECK(data != nullptr, "gguf dequant: data is null");
  VT_CHECK(numel >= 0, "gguf dequant: negative numel");

  const GgmlTypeTraits& traits = GgmlTraits(ggml_type);
  const int64_t block_elems = traits.block_elems;
  VT_CHECK(numel % block_elems == 0,
           std::string("gguf dequant: numel not a multiple of block_elems for ")
               .append(traits.name).c_str());

  std::vector<float> out(static_cast<size_t>(numel));
  float* y = out.data();

  // Guard: our hardcoded per-type block byte stride must equal the reader's
  // traits. A mismatch means a port bug in one place or the other.
  auto check_bytes = [&](int64_t bytes) {
    VT_CHECK(traits.block_bytes == bytes,
             std::string("gguf dequant: block_bytes mismatch vs traits for ")
                 .append(traits.name).c_str());
  };

  switch (ggml_type) {
    case 0:  // F32
      check_bytes(4);
      std::memcpy(y, data, static_cast<size_t>(numel) * sizeof(float));
      break;
    case 1:  // F16 (ggml fp16 = IEEE half; unquantized tensors in mixed files)
      check_bytes(2);
      for (int64_t i = 0; i < numel; ++i) y[i] = ReadF16(data + i * 2);
      break;
    case 30: {  // BF16 (upper 16 bits of f32)
      check_bytes(2);
      for (int64_t i = 0; i < numel; ++i) {
        uint16_t b;
        std::memcpy(&b, data + i * 2, sizeof(b));
        uint32_t u = static_cast<uint32_t>(b) << 16;
        std::memcpy(&y[i], &u, sizeof(u));
      }
      break;
    }
    case 2:    // Q4_0
    case 8:    // Q8_0
    case 11:   // Q3_K
    case 12:   // Q4_K
    case 13:   // Q5_K
    case 14: {  // Q6_K
      // The block decoders moved to vt (src/vt/cpu/cpu_quant_dequant.cpp) so
      // the loader oracle and the compute-in-quant GEMM's generic fallback
      // share ONE implementation. The code is byte-identical to what lived
      // here, so numerics are unchanged; this test suite gates that.
      vt::DType dtype = vt::DType::kF32;
      VT_CHECK(vt::BlockDTypeFromGgmlTypeId(ggml_type, &dtype),
               "gguf dequant: no vt block dtype for this ggml type");
      check_bytes(vt::BlockBytes(dtype));
      VT_CHECK(vt::BlockElems(dtype) == block_elems,
               "gguf dequant: vt block_elems disagrees with reader traits");
      vt::cpu::BlockToFloat(dtype)(data, y, numel);
      break;
    }
    default:
      throw std::runtime_error("gguf dequant: unsupported ggml type " +
                               std::to_string(ggml_type) +
                               " (" + traits.name + ") (Task 2/i-quant)");
  }
  return out;
}

std::vector<uint16_t> DequantGgufRowToBf16(uint32_t ggml_type,
                                           const uint8_t* data, int64_t numel) {
  const std::vector<float> f32 = DequantGgufRowToF32(ggml_type, data, numel);
  std::vector<uint16_t> out(f32.size());
  for (size_t i = 0; i < f32.size(); ++i) out[i] = vt::F32ToBF16(f32[i]);
  return out;
}

}  // namespace vllm
