// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/dtype.h"

#include <cstring>

namespace vt {

namespace {

// Block geometry table for the ggml block-quantized encodings, ported from
// llama.cpp @ 237ad9b96 `ggml/src/ggml-common.h` (the block structs and their
// static_asserts) with the type ids from `ggml/include/ggml.h:390-432`:
//   Q4_0 (id 2)  ggml-common.h:213-218  f16 d + QK4_0/2 qs        = 2 + 16
//   Q8_0 (id 8)  ggml-common.h:242-245  f16 d + QK8_0 qs          = 2 + 32
//   Q3_K (id 11) ggml-common.h:305-310  QK_K/8 hmask + QK_K/4 qs
//                                       + 12 scales + f16 d       = 32+64+12+2
//   Q4_K (id 12) ggml-common.h:317-327  2*f16 dm + 12 scales
//                                       + QK_K/2 qs               = 4+12+128
//   Q5_K (id 13) ggml-common.h:334-345  2*f16 dm + 12 scales
//                                       + QK_K/8 qh + QK_K/2 qs   = 4+12+32+128
//   Q6_K (id 14) ggml-common.h:352-357  QK_K/2 ql + QK_K/4 qh
//                                       + QK_K/16 scales + f16 d  = 128+64+16+2
//   Q8_K (id 15) ggml-common.h:361-365  f32 d + QK_K qs
//                                       + QK_K/16 i16 bsums       = 4+256+32
// This table is INDEPENDENT of the GGUF reader's `GgmlTraits`
// (src/vllm/model_executor/model_loader/gguf_reader.cpp) on purpose: vt:: must
// not depend on the loader. `tests/vt/test_ops_quant_traits.cpp` cross-checks
// the two element-for-element, so a divergence in either port is caught.
struct BlockGeometry {
  int64_t block_elems;
  int64_t block_bytes;
  uint32_t ggml_type;
  const char* name;
};

const BlockGeometry* FindBlockGeometry(DType dtype) {
  switch (dtype) {
    case DType::kQ4_0: {
      static constexpr BlockGeometry g{32, 18, 2, "q4_0"};
      return &g;
    }
    case DType::kQ8_0: {
      static constexpr BlockGeometry g{32, 34, 8, "q8_0"};
      return &g;
    }
    case DType::kQ3_K: {
      static constexpr BlockGeometry g{256, 110, 11, "q3_K"};
      return &g;
    }
    case DType::kQ4_K: {
      static constexpr BlockGeometry g{256, 144, 12, "q4_K"};
      return &g;
    }
    case DType::kQ5_K: {
      static constexpr BlockGeometry g{256, 176, 13, "q5_K"};
      return &g;
    }
    case DType::kQ6_K: {
      static constexpr BlockGeometry g{256, 210, 14, "q6_K"};
      return &g;
    }
    case DType::kQ8_K: {
      static constexpr BlockGeometry g{256, 292, 15, "q8_K"};
      return &g;
    }
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
    case DType::kI8:
    case DType::kI32:
    case DType::kI64:
      return nullptr;
  }
  return nullptr;
}

const BlockGeometry& RequireBlockGeometry(DType dtype) {
  const BlockGeometry* g = FindBlockGeometry(dtype);
  VT_CHECK(g != nullptr, std::string("dtype ") + Name(dtype) +
                             " is not block-quantized");
  return *g;
}

}  // namespace

bool IsBlockQuant(DType dtype) { return FindBlockGeometry(dtype) != nullptr; }

int64_t BlockElems(DType dtype) { return RequireBlockGeometry(dtype).block_elems; }

int64_t BlockBytes(DType dtype) { return RequireBlockGeometry(dtype).block_bytes; }

uint32_t GgmlTypeId(DType dtype) { return RequireBlockGeometry(dtype).ggml_type; }

bool BlockDTypeFromGgmlTypeId(uint32_t ggml_type, DType* out) {
  static constexpr DType kBlockDTypes[] = {
      DType::kQ4_0, DType::kQ8_0, DType::kQ3_K, DType::kQ4_K,
      DType::kQ5_K, DType::kQ6_K, DType::kQ8_K};
  for (DType d : kBlockDTypes) {
    if (FindBlockGeometry(d)->ggml_type == ggml_type) {
      if (out != nullptr) *out = d;
      return true;
    }
  }
  return false;
}

size_t RowSizeBytes(DType dtype, int64_t k) {
  VT_CHECK(k >= 0, "RowSizeBytes: negative element count");
  const BlockGeometry* g = FindBlockGeometry(dtype);
  if (g == nullptr) return static_cast<size_t>(k) * SizeOf(dtype);
  // ggml_row_size asserts ne % blck_size == 0: a row is whole blocks.
  VT_CHECK(k % g->block_elems == 0,
           std::string("RowSizeBytes: ") + std::to_string(k) +
               " elements is not a whole number of " + g->name + " blocks");
  return static_cast<size_t>(k / g->block_elems) *
         static_cast<size_t>(g->block_bytes);
}

size_t SizeOf(DType dtype) {
  switch (dtype) {
    case DType::kF32: return 4;
    case DType::kF16: return 2;
    case DType::kBF16: return 2;
    case DType::kI8: return 1;
    case DType::kI32: return 4;
    case DType::kI64: return 8;
    // Block-quantized dtypes are storage-only: there is no per-element size,
    // so every elementwise path that reaches one fails loudly here rather than
    // silently mis-striding a packed block buffer.
    case DType::kQ4_0:
    case DType::kQ8_0:
    case DType::kQ3_K:
    case DType::kQ4_K:
    case DType::kQ5_K:
    case DType::kQ6_K:
    case DType::kQ8_K:
      VT_CHECK(false, std::string("SizeOf: block-quantized dtype ") +
                          Name(dtype) + " has no per-element size");
      return 0;
  }
  VT_CHECK(false, "unknown dtype");
  return 0;
}

const char* Name(DType dtype) {
  switch (dtype) {
    case DType::kF32: return "f32";
    case DType::kF16: return "f16";
    case DType::kBF16: return "bf16";
    case DType::kI8: return "i8";
    case DType::kI32: return "i32";
    case DType::kI64: return "i64";
    case DType::kQ4_0: return "q4_0";
    case DType::kQ8_0: return "q8_0";
    case DType::kQ3_K: return "q3_K";
    case DType::kQ4_K: return "q4_K";
    case DType::kQ5_K: return "q5_K";
    case DType::kQ6_K: return "q6_K";
    case DType::kQ8_K: return "q8_K";
  }
  return "?";
}

namespace {
uint32_t AsU32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float AsF32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
}  // namespace

float F16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0x1F) {  // inf/nan
    return AsF32(sign | 0x7F800000 | (mant << 13));
  }
  if (exp == 0) {
    if (mant == 0) return AsF32(sign);  // signed zero
    // subnormal: normalize
    int shift = 0;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      ++shift;
    }
    mant &= 0x3FF;
    return AsF32(sign | ((113 - shift) << 23) | (mant << 13));
  }
  return AsF32(sign | ((exp + 112) << 23) | (mant << 13));
}

uint16_t F32ToF16(float f) {
  uint32_t u = AsU32(f);
  uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000);
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = u & 0x7FFFFF;
  if (((u >> 23) & 0xFF) == 0xFF) {  // inf/nan
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 | (mant >> 13) : 0));
  }
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);  // overflow → inf
  if (exp <= 0) {
    if (exp < -10) return sign;  // underflow → zero
    // subnormal
    mant |= 0x800000;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;  // round to nearest even
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;  // may carry into exp: correct
  return static_cast<uint16_t>(sign | half);
}

float BF16ToF32(uint16_t b) { return AsF32(static_cast<uint32_t>(b) << 16); }

uint16_t F32ToBF16(float f) {
  uint32_t u = AsU32(f);
  if ((u & 0x7F800000) == 0x7F800000 && (u & 0x7FFFFF)) {  // nan: keep quiet, truncate
    return static_cast<uint16_t>((u >> 16) | 0x0040);
  }
  uint32_t rounding = 0x7FFF + ((u >> 16) & 1);  // round to nearest even
  return static_cast<uint16_t>((u + rounding) >> 16);
}

}  // namespace vt
