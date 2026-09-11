#ifndef VLLM_CPP_SRC_VT_ROCM_ROCM_ACT_QUANT_H_
#define VLLM_CPP_SRC_VT_ROCM_ROCM_ACT_QUANT_H_

#include <cstdint>

#include "vt/cpu/cpu_quant_blocks.h"
#include "vt/dtype.h"
#include "vt/rocm/rocm_f16_codec.h"

namespace vt::rocm {

enum class ActDT : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

inline ActDT ActDtOf(DType dtype) {
  return dtype == DType::kF32
             ? ActDT::kF32
             : dtype == DType::kF16 ? ActDT::kF16 : ActDT::kBF16;
}

__device__ inline int DNearestInt(float value) {
  const float rounded = value + 12582912.0F;
  const int bits = __float_as_int(rounded);
  return (bits & 0x007fffff) - 0x00400000;
}

__device__ inline float DLoadAct(const void* base, ActDT dtype,
                                 int64_t index) {
  switch (dtype) {
    case ActDT::kF32:
      return static_cast<const float*>(base)[index];
    case ActDT::kF16:
      return DF16ToF32(static_cast<const uint16_t*>(base)[index]);
    default:
      return DBF16ToF32(static_cast<const uint16_t*>(base)[index]);
  }
}

// All Q8_K producers use this exact serial body. Besides avoiding numeric
// drift, the strict comparison preserves the lowest-index winner for tied
// absolute maxima, matching the CPU quantizer.
__device__ inline void QuantQ8KSBlock(vt::cpu::BlockQ8_K& output,
                                      const void* __restrict__ activation,
                                      ActDT dtype, int64_t first_element) {
  float signed_max = 0.0F;
  float absolute_max = 0.0F;
  for (int element = 0; element < vt::cpu::kQK_K; ++element) {
    const float value =
        DLoadAct(activation, dtype, first_element + element);
    if (const float absolute = fabsf(value); absolute > absolute_max) {
      absolute_max = absolute;
      signed_max = value;
    }
  }
  if (absolute_max == 0.0F) {
    output.d = 0.0F;
    for (int element = 0; element < vt::cpu::kQK_K; ++element) {
      output.qs[element] = 0;
    }
    for (int group = 0; group < vt::cpu::kQK_K / 16; ++group) {
      output.bsums[group] = 0;
    }
    return;
  }

  const float inverse_scale = -127.0F / signed_max;
  for (int element = 0; element < vt::cpu::kQK_K; ++element) {
    const int quantized = DNearestInt(
        inverse_scale * DLoadAct(activation, dtype, first_element + element));
    output.qs[element] =
        static_cast<int8_t>(quantized < 127 ? quantized : 127);
  }
  for (int group = 0; group < vt::cpu::kQK_K / 16; ++group) {
    int sum = 0;
    for (int element = 0; element < 16; ++element) {
      sum += output.qs[group * 16 + element];
    }
    output.bsums[group] = static_cast<int16_t>(sum);
  }
  output.d = 1.0F / inverse_scale;
}

}  // namespace vt::rocm

#endif  // VLLM_CPP_SRC_VT_ROCM_ROCM_ACT_QUANT_H_
