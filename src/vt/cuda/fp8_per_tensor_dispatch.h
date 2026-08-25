// vllm.cpp — the HOST-SIDE tile decision of the PER-TENSOR FP8 CUTLASS GEMM
// (PERF-FP8-SMALL-M-DISPATCH, #1866,
// .agents/specs/perf-fp8-small-m-dispatch.md; owning row `KERNEL-GEMM-FP8`).
//
// This header holds ONLY the pure pieces of `vt::MatmulFp8Cutlass`'s sm120 arm:
// which of upstream's four tile configs a given M selects, the rollback flag's
// parse, the config names the diagnostic prints, and the per-config dispatch
// counters. It has no CUDA dependency of any kind. The heavy CUTLASS templates
// and the registration live in `src/vt/cuda/cuda_matmul_fp8_cutlass.cu`, which
// is compiled only for `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a).
//
// WHY THE SPLIT. It is the arrangement `fp8_block_scaled_dispatch.h` beside it
// already uses for the BLOCKWISE fp8 sibling, and the reason is identical: no
// CI job in this tree has a GPU, and the machines that write this code have
// neither a GPU nor `nvcc`. A tile ladder that can only be exercised on
// hardware is a ladder nothing gates — and this one was already wrong for every
// decode step this tree has run since `7b682cc52` (2026-07-05), seven weeks,
// without anything noticing, because a wrong tile is a SLOW answer, not a WRONG
// one, and no correctness gate can see it.
//
// UPSTREAM, pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`,
// `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
//  scaled_mm_sm120_fp8_dispatch.cuh`:
//   * the four-way M ladder: `cutlass_gemm_sm120_fp8_dispatch`, :155-179.
//   * `sm120_fp8_config_M16`   :127-138 — 16x64x128,  EpilogueTile 16x32.
//   * `sm120_fp8_config_M32`   :112-123 — 32x64x128,  EpilogueTile 32x32.
//   * `sm120_fp8_config_M64`   :94-108  — 64x64x128,  EpilogueTile auto.
//   * `sm120_fp8_config_default` :81-90 — 128x128x128, EpilogueTile auto.
//   * the custom-EpilogueTile wrapper the two small-M configs need:
//     `cutlass_3x_gemm_sm120_custom`, :18-77.
// That ladder is what a GB10 runs upstream: `scaled_mm_entry.cu:222-225`
// routes every `get_sm_version_num() >= 120` — sm_121a reports 121 — into
// `cutlass_scaled_mm_sm120`, and the CUDA FP8 backend order
// (`vllm/model_executor/kernels/linear/__init__.py:325-334`) reaches Cutlass
// before it would ever reach the torch/cuBLASLt fallback.
//
// WHAT THIS TREE SHIPPED INSTEAD, and why it is a decode defect. Our port kept
// only the top two rungs, on the recorded ground that "vLLM's M16/M32
// custom-EpilogueTile refinements are perf-only for tiny M and are covered
// correctly (predicated) by the M64 pingpong tile". Both halves are true. The
// consequence is not: TINY M IS DECODE. Every M from 1 to 256 took the
// 64x64x128 tile, so a batch-1 step computed a 64-row tile for one row, and
// the 9-row spec-decode verify computed one for nine.
#ifndef VT_CUDA_FP8_PER_TENSOR_DISPATCH_H_
#define VT_CUDA_FP8_PER_TENSOR_DISPATCH_H_

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace vt::cuda {

// ---------------------------------------------------------------------------
// The rollback flag
// ---------------------------------------------------------------------------

// `VT_FP8_CUTLASS_SMALL_M` — the small-M rungs are ON by default (they are the
// upstream mirror) and DISABLED only for the exact value "0", the same shape as
// `Fp8PlanCacheFlagIsOn` and `GemmPlanCacheFlagIsOn`: unset, "", "00", "false"
// and every other spelling leave the mirror ON, so a mangled rollback cannot
// silently un-mirror the ladder. "0" selects the two-way ladder this tree
// shipped before, byte for byte, which is what makes the same-binary A/B a
// real A/B rather than two different builds.
inline bool Fp8CutlassSmallMFlagIsOn(const char* env_value) {
  return !(env_value != nullptr && std::string_view(env_value) == "0");
}

// Process-cached gate, read from the environment exactly once. Kept separate
// from the parse above so the parse is unit-testable without latching the
// process-global value — the arrangement every other flag in this directory
// uses.
inline bool Fp8CutlassSmallMEnabled() {
  static const bool enabled =
      Fp8CutlassSmallMFlagIsOn(std::getenv("VT_FP8_CUTLASS_SMALL_M"));
  return enabled;
}

// ---------------------------------------------------------------------------
// Which tile config runs
// ---------------------------------------------------------------------------

// Upstream's four sm120 per-tensor fp8 configs, in its own order of test.
enum class Fp8PerTensorConfig : uint8_t {
  kM16 = 0,   // 16x64x128,   EpilogueTile 16x32, Pingpong
  kM32,       // 32x64x128,   EpilogueTile 32x32, Pingpong
  kM64,       // 64x64x128,   EpilogueTile auto,  Pingpong
  kDefault,   // 128x128x128, EpilogueTile auto,  KernelScheduleAuto
  kCount,
};

// Upstream's boundaries, named rather than spelled inline at the comparison,
// so the ladder reads as the transcription it is.
inline constexpr int64_t kFp8PerTensorM16Max = 16;
inline constexpr int64_t kFp8PerTensorM32Max = 32;
inline constexpr int64_t kFp8PerTensorM64Max = 256;

// The ladder, `cutlass_gemm_sm120_fp8_dispatch:155-179` transcribed.
//
// `small_m` false collapses the two small-M rungs into kM64, which reproduces
// this tree's pre-#1866 dispatch exactly — that is what makes
// `VT_FP8_CUTLASS_SMALL_M=0` a same-binary A/B against the shipped program
// rather than against a third thing.
//
// TOTAL, including at m <= 0. The first comparison is `<=`, so a zero or
// negative extent lands on the smallest rung instead of falling through to a
// tile sized for 4096 rows. The caller returns early on m == 0 and a negative
// extent cannot occur, so this is a property of the function rather than a
// path anything reaches; it is asserted anyway, because the alternative shape
// (`if (m > 0 && m <= 16)`) would silently send m == 0 to `kDefault`.
inline Fp8PerTensorConfig Fp8Sm120ConfigForM(int64_t m, bool small_m) {
  if (small_m) {
    if (m <= kFp8PerTensorM16Max) return Fp8PerTensorConfig::kM16;
    if (m <= kFp8PerTensorM32Max) return Fp8PerTensorConfig::kM32;
  }
  if (m <= kFp8PerTensorM64Max) return Fp8PerTensorConfig::kM64;
  return Fp8PerTensorConfig::kDefault;
}

inline const char* Fp8PerTensorConfigName(Fp8PerTensorConfig c) {
  switch (c) {
    case Fp8PerTensorConfig::kM16:
      return "M16_16x64x128";
    case Fp8PerTensorConfig::kM32:
      return "M32_32x64x128";
    case Fp8PerTensorConfig::kM64:
      return "M64_64x64x128";
    case Fp8PerTensorConfig::kDefault:
      return "default_128x128x128";
    case Fp8PerTensorConfig::kCount:
      break;
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Which rung a real run actually took
// ---------------------------------------------------------------------------
//
// The counters exist because "the ladder selects kM16 at M=9" is a statement
// about this function, and "the model's decode step reaches this function at
// M=9" is a different statement that only a run can make. A gate that asserts
// the first and assumes the second is the shape of defect #1866 is about: the
// tile was wrong for every decode step and every token gate stayed green.
namespace detail {
inline std::atomic<uint64_t>* Fp8PerTensorCounters() {
  static std::atomic<uint64_t> counters[static_cast<size_t>(Fp8PerTensorConfig::kCount)] = {};
  return counters;
}
}  // namespace detail

inline void Fp8PerTensorCountDispatch(Fp8PerTensorConfig c) {
  const auto slot = static_cast<size_t>(c);
  if (slot >= static_cast<size_t>(Fp8PerTensorConfig::kCount)) return;
  detail::Fp8PerTensorCounters()[slot].fetch_add(1, std::memory_order_relaxed);
}

inline uint64_t Fp8PerTensorDispatchCount(Fp8PerTensorConfig c) {
  const auto slot = static_cast<size_t>(c);
  if (slot >= static_cast<size_t>(Fp8PerTensorConfig::kCount)) return 0;
  return detail::Fp8PerTensorCounters()[slot].load(std::memory_order_relaxed);
}

inline void Fp8PerTensorResetDispatchCounts() {
  auto* c = detail::Fp8PerTensorCounters();
  for (size_t i = 0; i < static_cast<size_t>(Fp8PerTensorConfig::kCount); ++i) {
    c[i].store(0, std::memory_order_relaxed);
  }
}

}  // namespace vt::cuda

#endif  // VT_CUDA_FP8_PER_TENSOR_DISPATCH_H_
