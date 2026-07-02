// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/dtype.h"

#include <cstring>

namespace vt {

size_t SizeOf(DType dtype) {
  switch (dtype) {
    case DType::kF32: return 4;
    case DType::kF16: return 2;
    case DType::kBF16: return 2;
    case DType::kI8: return 1;
    case DType::kI32: return 4;
    case DType::kI64: return 8;
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
