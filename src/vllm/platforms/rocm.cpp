// ROCm leg of the Platform seam (BACKEND-ROCM, W0 skeleton). Self-registers
// kROCM via a static Registrar, copying the src/vllm/platforms/cuda.cpp /
// metal.cpp / vulkan.cpp registrar idiom. Compiled only in a HIP build (CMake
// target_sources gate).
//
// COMPILE-VERIFIED, unlike the .hip TUs of this skeleton — this one is plain C++
// with no HIP header anywhere in it, reaching the device only through the vt::
// seam (vt::GetBackend + the two probes in include/vt/rocm/rocm_runtime.h). That
// is the same discipline vulkan.cpp follows, it is what keeps src/vllm/ free of
// vendor headers, and it is what lets a machine with no ROCm compile this file
// at all: CMakeLists.txt builds it as a never-linked OBJECT library in NON-HIP
// builds precisely so a later change to the Platform interface cannot break it
// silently for a contributor mid-bring-up. It has never RUN — that needs a GPU.
//
// UPSTREAM MIRROR: vllm/platforms/rocm.py @ pin 555967922 (`class RocmPlatform`,
// rocm.py:444). Every value below either cites the upstream line it mirrors or
// says outright that it is a W0 placeholder — there is no third category, and no
// value here is a guess dressed as a decision.
#include "vllm/platforms/interface.h"

#include <vector>

#include "vt/backend.h"
#include "vt/rocm/rocm_runtime.h"

namespace vllm::platforms {
namespace {

class RocmPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kROCM; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kROCM); }

  // rocm.py:481-497 get_device_capability -> (major, minor) parsed from
  // gcnArchName by _capability_from_gcn_arch (rocm.py:223). Our backend already
  // did that parse at registration through the SAME ported function
  // (include/vt/rocm/rocm_arch.h), so both layers answer from one source rather
  // than parsing twice and risking two answers — the seam-gap #4 lesson the CUDA
  // leg records at cuda_backend.cu:64-71.
  DeviceCapability get_device_capability() const override {
    Backend& b = backend();
    return DeviceCapability{b.DeviceCapabilityMajor(), b.DeviceCapabilityMinor()};
  }

  // interface.py:181-187 supported_dtypes order (bf16 default first). All three
  // are real on every gfx9/gfx11/gfx12 part vLLM supports; bf16 in particular is
  // native on RDNA3, which is what all three boards on issue #41 are.
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16, DType::kF16, DType::kF32};
  }

  // rocm.py:663 is_integrated_gpu. Answered from the backend's probed
  // UnifiedMemory(), which is the STRICTER conjunction
  // (integrated AND pageable-memory-access, see rocm_backend.hip). So a false
  // here can mean "integrated but no pageable access" — conservative in the
  // direction that matters, since every consumer of this predicate is asking
  // whether it may treat host and device memory as one.
  bool is_integrated_gpu() const override { return backend().UnifiedMemory(); }

  // --- W0 placeholders. Each is the BASE class answer, stated explicitly here
  // --- with what would change it, so nobody has to guess whether the omission
  // --- was considered.
  //
  // supports_fp8() stays false: gfx942/gfx950 have hardware fp8 and rocm.py lists
  //   "fp8" in supported_quantization (rocm.py:457-467), but we have no ROCm fp8
  //   kernel, and this predicate gates a fused path that would then not exist.
  // support_static_graph_mode() stays false: the vt::Backend hipGraph capture
  //   seam is implemented as of BACKEND-ROCM W1 (rocm_backend.hip; see
  //   .agents/specs/rocm-decode-graph.md) and the address-baking concern that
  //   used to justify leaving this false is now an assertion, not a worry —
  //   the mutate-src-then-replay test step fails if replay ever returns a
  //   snapshot. This flag still stays false because flipping it to engage a
  //   real model's decode-graph path is W2, not W1.
  // needs_weight_staging() stays false: this is the memory-model POLICY that
  //   selects the device-resident forward over the host-resident reference path.
  //   HIP's programming model does stage (hipMalloc hands back a distinct
  //   address), so a discrete AMD card will eventually answer true — but in W0
  //   there is one registered op, so the only path that can run at all is the
  //   host-resident one the reference tier serves on a unified part. Answering
  //   true today would route a model into a path with no kernels. Revisit at M2.
  // supports_fa2_attention() / opaque_attention_op() stay false: no ROCm
  //   attention kernel exists yet. See get_attn_backend_priority below.

  // The residency/memory-model policy. DEFAULTS in W0, and the reason is not
  // laziness: on a unified part (780M, Strix Halo) freeing the host copy after
  // "upload" would free the ONLY copy, which is the same answer the CPU, Metal
  // and Vulkan platforms give for the same reason. A DISCRETE card genuinely
  // wants release_host_weights_after_upload=true and a pooled allocator, and
  // that is a per-DEVICE answer this per-DEVICE-TYPE seam cannot express yet
  // (the CUDA leg has the same shape and has not needed to). Flip it when a
  // discrete board actually loads weights, with the measurement in the record.
  ResidencyPolicy residency_policy() const override { return {}; }

  // Attention-backend priority. There is NO ROCm attention kernel in this
  // skeleton — kPagedAttention is not registered for kROCM — so returning a name
  // would be a claim we cannot honour. The EMPTY list is the honest and
  // mechanically correct answer: SelectAttentionBackendName walks the list and
  // takes the first REGISTERED name (include/vllm/v1/attention/registry.h:59-78),
  // so an empty list makes selection throw loudly instead of handing back a
  // backend whose kernels do not exist. Same choice, same reasoning, as the
  // Vulkan leg.
  //
  // WHAT GOES HERE AT M3, read off upstream rather than invented: rocm.py:407
  // _get_backend_priorities returns [ROCM_ATTN, (ROCM_AITER_FA if AITER MHA),
  // (ROCM_AITER_UNIFIED_ATTN if AITER), TRITON_ATTN, TURBOQUANT] for the
  // non-MLA case. AITER is gfx9-only in practice (rocm.py:661-665 gates the FA
  // path on on_gfx9()), so on the RDNA3 boards of issue #41 the reachable entries
  // are ROCM_ATTN and TRITON_ATTN.
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    (void)cfg;
    return {};
  }
};

// Registers kROCM during static init (registration completes before main() per
// the interface.h contract). Probes the DEVICE itself rather than trusting
// another TU's initializer, because static-init order across TUs is unspecified
// — asking "did the backend registrar run?" would intermittently skip platform
// registration. Registering a platform whose backend() would throw is worse than
// not registering: CurrentPlatform() (src/vllm/platforms/platform.cpp:38-40)
// walks the device list and must be able to fall through to CPU on a machine
// that merely happens to have been built with -DVLLM_CPP_HIP=ON.
struct Registrar {
  Registrar() noexcept {
    if (!vt::rocm::DeviceAvailable()) return;
    static RocmPlatform platform;
    RegisterPlatform(DeviceType::kROCM, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
