// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#define VT_CHECK(cond, msg)                                                       \
  do {                                                                            \
    if (!(cond)) {                                                                \
      throw std::runtime_error(std::string("vt: ") + (msg) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                   \
    }                                                                             \
  } while (0)

namespace vt {

enum class DType : uint8_t { kF32, kF16, kBF16, kI8, kI32, kI64 };

size_t SizeOf(DType dtype);
const char* Name(DType dtype);

float F16ToF32(uint16_t h);
uint16_t F32ToF16(float f);
float BF16ToF32(uint16_t b);
uint16_t F32ToBF16(float f);

}  // namespace vt
