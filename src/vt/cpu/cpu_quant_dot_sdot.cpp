// Cortex-A76 Q8_0 x Q8_0 DotProd tier (KERNEL-CPU-A76-Q8-DOT).
//
// The integer core follows llama.cpp @ 237ad9b96
// ggml/src/ggml-cpu/arch/arm/quants.c:1076-1160, but preserves this project's
// stricter scalar per-block float accumulation order. The compiler arm exists
// to expose what GCC can do from ACLE; the assembly arm is a separately
// measurable AAPCS64 schedule over the exact same arithmetic.
#include "vt/quant.h"

#if defined(VT_CPU_A76_Q8_DOT) && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "cpu_quant_blocks.h"

#if defined(__linux__) && !defined(HWCAP_ASIMDDP)
#define HWCAP_ASIMDDP (1 << 20)
#endif

extern "C" void vt_cpu_q8_dot_a76_asm(int n, float* s, size_t bs, const void* x, size_t bx,
                                      const void* y, size_t by, int nrc);

namespace vt::cpu {
namespace {

bool CpuHasDotProd() {
#if defined(__linux__)
  return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
  return true;
#endif
}

bool CpuIsCortexA76() {
#if defined(__linux__)
  std::ifstream in("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1");
  std::string value;
  if (!(in >> value)) return false;
  char* end = nullptr;
  const unsigned long long midr = std::strtoull(value.c_str(), &end, 0);
  if (end == value.c_str() || *end != '\0') return false;
  const unsigned implementer = static_cast<unsigned>((midr >> 24) & 0xffU);
  const unsigned part = static_cast<unsigned>((midr >> 4) & 0xfffU);
  return implementer == 0x41U && part == 0xd0bU;
#else
  return false;
#endif
}

inline float HalfToFloat(uint16_t bits) {
  _Float16 value;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<float>(value);
}

float Q8DotSdot(const BlockQ8_0* x, const BlockQ8_0* y, int nb) {
  float sumf = 0.0F;
  for (int ib = 0; ib < nb; ++ib) {
    int32x4_t dot = vdupq_n_s32(0);
    dot = vdotq_s32(dot, vld1q_s8(x[ib].qs), vld1q_s8(y[ib].qs));
    dot = vdotq_s32(dot, vld1q_s8(x[ib].qs + 16), vld1q_s8(y[ib].qs + 16));
    const int sumi = vaddvq_s32(dot);
    const float scale = HalfToFloat(x[ib].d) * HalfToFloat(y[ib].d);
    sumf += static_cast<float>(sumi) * scale;
  }
  return sumf;
}

void CheckArgs(int n, int nrc, const char* name) {
  VT_CHECK(n % kQK8_0 == 0, std::string(name) + ": n must be a multiple of 32");
  VT_CHECK(nrc == 1, std::string(name) + ": supports nrc == 1 only");
}

void VecDotQ8Sdot(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by,
                  int nrc) {
  CheckArgs(n, nrc, "vec_dot_q8_0_sdot");
  (void)bs;
  (void)bx;
  (void)by;
  *s = Q8DotSdot(static_cast<const BlockQ8_0*>(vx), static_cast<const BlockQ8_0*>(vy), n / kQK8_0);
}

}  // namespace

extern "C" [[noreturn]] void vt_cpu_q8_dot_a76_bad_args(int n, int nrc) {
  CheckArgs(n, nrc, "vec_dot_q8_0_a76_asm");
  std::abort();
}

VecDotFn QuantQ8SdotVecDot() {
  return CpuHasDotProd() ? &VecDotQ8Sdot : nullptr;
}

VecDotFn QuantQ8A76AsmVecDot() {
  return CpuHasDotProd() ? &vt_cpu_q8_dot_a76_asm : nullptr;
}

VecDotFn SelectQuantQ8VecDot(VecDotFn portable) {
  const char* value = std::getenv("VT_CPU_Q8_DOT");
  if (value == nullptr || std::strcmp(value, "auto") == 0) {
    return QuantQ8A76AsmActive() ? QuantQ8A76AsmVecDot() : portable;
  }
  if (std::strcmp(value, "portable") == 0) {
    return portable;
  }
  if (std::strcmp(value, "sdot") == 0) {
    return QuantQ8SdotVecDot() != nullptr ? QuantQ8SdotVecDot() : portable;
  }
  if (std::strcmp(value, "a76-asm") == 0) {
    return QuantQ8A76AsmVecDot() != nullptr ? QuantQ8A76AsmVecDot() : portable;
  }
  VT_CHECK(false, "VT_CPU_Q8_DOT must be auto, portable, sdot, or a76-asm");
  return portable;
}

bool QuantQ8SdotActive() {
  return QuantQ8SdotVecDot() != nullptr;
}
bool QuantQ8A76AsmActive() {
  return QuantQ8A76AsmVecDot() != nullptr && CpuIsCortexA76();
}

}  // namespace vt::cpu

#else

namespace vt::cpu {
VecDotFn QuantQ8SdotVecDot() {
  return nullptr;
}
VecDotFn QuantQ8A76AsmVecDot() {
  return nullptr;
}
VecDotFn SelectQuantQ8VecDot(VecDotFn portable) {
  return portable;
}
bool QuantQ8SdotActive() {
  return false;
}
bool QuantQ8A76AsmActive() {
  return false;
}
}  // namespace vt::cpu

#endif
