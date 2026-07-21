// vllm.cpp original (vt runtime, inventory deviation §9.1) — the RUNTIME SM
// DISPATCH seam (BACKEND-CUDA-ARCH-ADDITIVITY, seam-gap #2 of
// .agents/specs/breadth-sweep-plan.md §A.2).
//
// THE GAP THIS CLOSES. Every first-party host launcher in src/vt/cuda/ took NO
// capability argument and had NO `if (sm == ...)` selector: `LaunchFp4Fp4`
// (cuda_matmul_nvfp4.cu) went straight into an sm_120a-only `mma.sync
// kind::mxf4nvf4` path behind `#if VT_FP4_MMA_SM120A`, the FP8 launcher baked
// `ArchTag = cutlass::arch::Sm120`, and FA-2 compiled every instantiation for
// sm_121a. Adding an architecture therefore meant EDITING those launchers — the
// PR-#4 scatter anti-pattern, one layer down.
//
// THE SEAM. A tiny type-erased registry: each architecture's kernel bodies live
// in their own translation unit and REGISTER a tactic at static init; the
// launcher asks `SelectArchTactic(family, caps)` for the first tactic that
// supports the running device and calls it. Adding an architecture becomes
//   (1) a new TU with that arch's kernel bodies,
//   (2) one `RegisterArchTactic(...)` in it,
//   (3) one FEATURE-TABLE cell widened in cmake/CudaArchFeatures.cmake.
// No launcher edit. Exactly ONE tactic is registered today — the existing
// sm_12x native fp4 path — so behavior on GB10 is unchanged; the value of this
// header is the shape, not the population.
//
// UPSTREAM. This mirrors the tactic/heuristic dispatch every kernel provider in
// vLLM's runtime chain uses instead of compile-time arch pinning:
//   * FlashInfer's per-arch tactic registry + runtime selection —
//     flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:187-220 (the 32 SM12
//     tactics we already mirror in nvfp4_tactic_ids.h) dispatched by the
//     capability-keyed launcher in flashinfer/gemm/;
//   * vLLM's own build-side per-arch intersection, whose runtime counterpart is
//     `vllm/platforms/cuda.py::CudaPlatform.get_device_capability` gating which
//     kernel family is even reachable (cuda.py:154-166, ported in
//     src/vllm/platforms/cuda.cpp:64-71);
//   * cuBLASLt/CUTLASS heuristics, which resolve the kernel per call from the
//     device properties rather than from `__CUDA_ARCH__`.
//
// DISCIPLINE. Registration is TABLE FILL ONLY — no CUDA API calls before main()
// (the pre-main rule stated in cuda_ops.cu). Capability probing happens at
// launch time through GetDeviceCaps().
#pragma once

#include <cstdint>

#include "vt/cuda/cuda_device_caps.h"
#include "vt/dtype.h"

namespace vt::cuda {

// Kernel families that dispatch per architecture. A family exists here as soon
// as its launcher consults the registry; a family with no registered tactic
// simply always falls through to the launcher's portable path.
enum class TacticFamily : int {
  // Native block-scaled fp4xfp4 GEMM (cuda_matmul_nvfp4.cu::LaunchFp4Fp4).
  kNvfp4Fp4Mma = 0,
  // CUTLASS FP8 scaled-mm (cuda_matmul_fp8_cutlass.cu).
  kFp8Cutlass,
  // Vendored FlashAttention-2 prefill/decode (cuda_flash_attn_fa2.cu).
  kFlashAttn,
  kCount
};

// Per-family launch payload for kNvfp4Fp4Mma. Defined here (not in the launcher
// TU) so a future architecture's tactic TU can implement the family without
// touching cuda_matmul_nvfp4.cu.
struct Nvfp4Fp4MmaArgs {
  void* stream = nullptr;  // cudaStream_t, erased to keep this header CUDA-free
  void* out = nullptr;
  DType out_dtype{};
  const uint8_t* a_packed = nullptr;
  const uint8_t* a_scale = nullptr;
  const uint8_t* b_packed = nullptr;
  const uint8_t* b_scale = nullptr;
  float alpha = 1.0f;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

// A tactic answers two questions: can it run on THIS device, and how does it
// launch. `supports` must be side-effect free and cheap (it runs per launch).
// `launch` returns false when it declined the shape, so the launcher falls back
// to its portable path exactly as before.
using ArchTacticSupportsFn = bool (*)(const DeviceCaps&);
using ArchTacticLaunchFn = bool (*)(const DeviceCaps&, void* args);

struct ArchTactic {
  const char* name = nullptr;  // stable identity, e.g. "nvfp4-fp4-mma/sm12x"
  ArchTacticSupportsFn supports = nullptr;
  ArchTacticLaunchFn launch = nullptr;
};

// Registers a tactic for `family`. Call from a static-init registrar in the TU
// that owns the kernel bodies. Registration order is the selection order:
// earlier registrations win, so a more specific arch tactic should register
// before a broader family fallback (today there is exactly one per family).
void RegisterArchTactic(TacticFamily family, const ArchTactic& tactic);

// First registered tactic whose `supports(caps)` is true, or nullptr. Also
// records the selection in the family's stats (see ArchTacticStats).
const ArchTactic* SelectArchTactic(TacticFamily family, const DeviceCaps& caps);

// How many tactics are registered for `family` (the additivity counter: it goes
// up by one per architecture brought up, and by nothing else).
int RegisteredArchTacticCount(TacticFamily family);

const char* ArchTacticFamilyName(TacticFamily family);

// POSITIVE SIGNAL. A passing gate does not prove a new code path ran, so the
// registry counts what it actually did. `selections` increments every time a
// tactic was chosen and `last_selected` names it; `fallbacks` increments when no
// tactic supported the device and the launcher used its portable path. Tests
// assert on these to prove the seam is EXERCISED rather than merely compiled.
// Setting VT_ARCH_TACTIC_STATS=1 additionally prints one line per family the
// first time it selects, so a device run leaves evidence in its own log.
struct ArchTacticStats {
  const char* last_selected = nullptr;
  unsigned long long selections = 0;
  unsigned long long fallbacks = 0;
};
ArchTacticStats GetArchTacticStats(TacticFamily family);

}  // namespace vt::cuda
