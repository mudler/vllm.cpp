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

#include <string>
#include <string_view>
#include <vector>

#include "vt/backend.h"
#include "vt/rocm/rocm_runtime.h"

namespace vllm::platforms {
namespace {

class RocmPlatform final : public Platform {
 public:
  // Issue #1934. `device_memory_total_bytes` is `vt::rocm::DeviceMemoryTotalBytes(0)`,
  // probed once by the registrar below at static init, exactly mirroring how
  // `CudaPlatform` threads its own `cudaMemGetInfo` probe through its
  // constructor (`platforms/cuda.cpp`). 0 means the probe failed or no device
  // is present; `residency_policy()` passes it through unexamined, and
  // `gguf_device_fit.h` already reads 0 as UNKNOWN rather than "nothing fits".
  explicit RocmPlatform(size_t device_memory_total_bytes)
      : device_memory_total_bytes_(device_memory_total_bytes) {}

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

  // ENG-EXPERT-STREAM-DEVICE W0b (issue #1124). Answered from the two device
  // attributes `ProbeDevice` already reads, through the HIP-free probe rather
  // than through `UnifiedMemory()`: that one is widened by the managed-alloc
  // branch, which makes the BACKEND's allocations migratable and says nothing
  // about whether a kernel may follow a plain host pointer. So this is NARROWER
  // than `is_integrated_gpu()` above, in the one direction that matters.
  //
  // Device 0, matching this leg's other single-device answers. NEVER RUN — no
  // ROCm hardware is reachable from this project — which is why it reads the
  // probe rather than hardcoding an answer for a board nobody here can check.
  bool host_memory_is_device_addressable() const override {
    return vt::rocm::HostMemoryIsDeviceAddressable(0);
  }

  // --- W0 placeholders. Each is the BASE class answer, stated explicitly here
  // --- with what would change it, so nobody has to guess whether the omission
  // --- was considered.
  //
  // supports_fp8() stays false: gfx942/gfx950 have hardware fp8 and rocm.py lists
  //   "fp8" in supported_quantization (rocm.py:457-467), but we have no ROCm fp8
  //   kernel, and this predicate gates a fused path that would then not exist.
  // GFX1100-TG200 (T2b): support_static_graph_mode() is now TRUE. The W1 note
  //   below recorded the two conditions for this flip: the vt::Backend hipGraph
  //   capture seam is implemented (rocm_backend.hip; BeginCapture/EndCaptureGraph/
  //   ReplayGraph mirror cuda_backend.cu call for call, and the mutate-src-then-
  //   replay test asserts replay never returns a snapshot), and a real model's
  //   decode-graph path had to be exercised. The Qwen3_5 dense decode driver
  //   (Qwen3_5DenseDecodeGraph) gates on this predicate plus SupportsGraphCapture()
  //   plus VLLM_CPP_CUDAGRAPH; with all three true it captures the uniform decode
  //   step per padded batch size and replays it. The keep-quant scratch pool is
  //   already capture-safe (hipMallocAsync, stream-ordered, never freed).
  //   The flip is PER-ARCH OPT-IN: static_graph_requires_opt_in() below returns
  //   false only on gfx1100 (the evidence arch) so the decode-graph gate engages
  //   there and stays eager everywhere else. See
  //   .agents/specs/gfx1100-tg200-t2b-static-graph-opt-in.md.
  bool support_static_graph_mode() const override { return true; }
  // GFX1100-TG200 (T2b): per-arch opt-in, mirroring #2910's Tenstorrent pattern.
  //   The decode-graph gate in qwen3_5_dense.cpp is:
  //     support_static_graph_mode() && !static_graph_requires_opt_in()
  //   so returning false here admits capture and returning true keeps it eager.
  //   gfx1100 (RX 7900 XTX, RDNA3) is the evidence arch — its captured arm has
  //   committed A/B evidence (docs/bench-evidence/gfx1100-tg200-t2b-20260823.md).
  //   Every other ROCm arch (gfx1151, gfx1103, gfx1200, gfx1201) returns true
  //   until it carries its own captured-arm evidence. The arch is read through
  //   the HIP-free vt::rocm::DeviceArchName(0) probe (same prefix-match
  //   discipline GcnArchNameIsGfx12PrefillWmma uses in rocm_arch.h); an empty
  //   string — no device present — does not match and degrades to opt-in (eager),
  //   the conservative answer.
  bool static_graph_requires_opt_in() const override {
    const std::string arch = vt::rocm::DeviceArchName(0);
    return !IsGfx1100(arch);
  }
  // Architecture-aware overload: ROCm's evidence dimension is the GPU arch, not
  //   the model family (unlike Tenstorrent, whose DecodeCaptureDefaultArch
  //   scopes by model architecture). So this delegates to the no-arg overload.
  //   A future arch that needs per-model scoping can override this independently.
  bool static_graph_requires_opt_in(
      const std::vector<std::string>& /*architectures*/) const override {
    return static_graph_requires_opt_in();
  }
  // needs_weight_staging() stays false: this is the memory-model POLICY that
  //   selects the FULLY-OPTIMIZED device-resident forward (indexed GDN state
  //   I/O with no op-registration fallback for a couple of its consumers,
  //   merged/packed GDN projections, fp8/bf16 GDN resident prep) over the
  //   host-resident reference path for THOSE specific kernels. Issue #1934
  //   measured that several of those consumers have no per-op fallback and
  //   would silently assume kernels this device might not register, so
  //   flipping this blindly was rejected — see that issue and
  //   `allocates_bounded_device_memory()` below, which answers the NARROWER
  //   question the load-time device-fit check actually needs without moving
  //   any of this flag's other consumers. `IndexedGdnStateIoEnabled` already
  //   takes ROCm's fast arm today regardless of this flag, by checking op
  //   registration directly rather than trusting this policy bit — the same
  //   move #1934 makes for the device-fit check.
  // supports_fa2_attention() / opaque_attention_op() stay false: no ROCm
  //   attention kernel exists yet. See get_attn_backend_priority below.

  // BACKEND-ROCM, issue #1934. The ONE narrow question the load-time
  // device-fit refusal (`gguf_device_fit.h`, issue #1123) needs answered:
  // does a load here allocate device memory `ResidentWeight` cannot exceed
  // unnoticed? Yes, unconditionally, on any non-CPU platform since issue
  // #125's fix (see the interface doc on this method) — independent of
  // `needs_weight_staging()`, which this method deliberately does not touch.
  bool allocates_bounded_device_memory() const override { return true; }

  // The residency/memory-model policy. `device_memory_total_bytes` is now a
  // REAL probe (issue #1934, mirrors `CudaPlatform`'s own `cudaMemGetInfo`
  // probe) — the ONE field the device-fit check reads. The other two fields
  // stay DEFAULT/false, unlike CUDA's: `release_host_weights_after_upload`
  // and `uses_device_memory_pool` are separate policy questions (a discrete
  // card's host-copy release and DevicePool reuse) this row does not touch,
  // because on a unified part (780M, Strix Halo) freeing the host copy after
  // "upload" would free the ONLY copy — the same answer CPU, Metal and Vulkan
  // give for the same reason, and per-DEVICE (not per-DEVICE-TYPE) besides.
  // Flip those when a discrete board's release/pool behavior is actually
  // measured, not as a side effect of making the budget check reachable.
  ResidencyPolicy residency_policy() const override {
    ResidencyPolicy p;
    p.device_memory_total_bytes = device_memory_total_bytes_;
    return p;
  }

  // Attention-backend priority — M3 (issue #41). Mirrors rocm.py:407-441
  // `_get_backend_priorities` at pin 555967922. The dense branch is
  // [ROCM_ATTN, (ROCM_AITER_FA if `rocm_aiter_ops.is_mha_enabled()`, :434),
  // (ROCM_AITER_UNIFIED_ATTN if `is_aiter_found_and_supported()`, :436),
  // TRITON_ATTN, TURBOQUANT] — mirrored VERBATIM below. Upstream appends
  // ROCM_ATTN only `if not use_kv_connector` (:432-433); our registration uses
  // the shared symmetric NHD layout (see the KV-LAYOUT DEVIATION record in
  // backend.h), which is immune to the asymmetric-view concern that guard
  // protects, so the guard does not apply here — recorded, not implied.
  // SelectAttentionBackendName skips unregistered names, so the AITER entries
  // and TRITON_ATTN/TURBOQUANT are placeholders that cost nothing and need no
  // gfx9 reasoning: a dense request resolves to the first REGISTERED name,
  // ROCM_ATTN. MLA mirrors rocm.py:414-419 ([TRITON_MLA] when AITER MLA is
  // off); TRITON_MLA is unregistered for kROCM (registered for CUDA only), so
  // a use_mla=true request throws loudly rather than silently falling back to
  // a dense backend — the honest answer until a ROCm MLA kernel lands.
  // use_sparse mirrors rocm.py:410 ([ROCM_AITER_MLA_SPARSE], AITER-only) and
  // is likewise unregistered.
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    if (cfg.use_sparse) return {"ROCM_AITER_MLA_SPARSE"};
    if (cfg.use_mla) return {"TRITON_MLA"};
    return {"ROCM_ATTN", "ROCM_AITER_FA", "ROCM_AITER_UNIFIED_ATTN",
            "TRITON_ATTN", "TURBOQUANT"};
  }

 private:
  // True iff `arch` is the gfx1100 gcnArchName prefix (e.g. "gfx1100" or
  // "gfx1100:sramecc+:xnack-"). Prefix, not substring: after the six-char stem
  // the next character must be end-of-string or a non-digit, so "gfx11000" does
  // not match. Same discipline as GcnArchNameIsGfx12PrefillWmma (rocm_arch.h).
  static bool IsGfx1100(std::string_view arch) {
    constexpr std::string_view kStem = "gfx1100";
    if (arch.size() < kStem.size()) return false;
    if (arch.substr(0, kStem.size()) != kStem) return false;
    if (arch.size() == kStem.size()) return true;
    const char c = arch[kStem.size()];
    return c < '0' || c > '9';
  }

  size_t device_memory_total_bytes_ = 0;
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
    // Issue #1934. Device 0, matching this leg's other single-device probes
    // (`host_memory_is_device_addressable()` above states the same choice).
    // HIP-free free function, not `Backend::DeviceMemoryInfo`: the backend's
    // OWN registrar (`rocm_backend.hip`) may not have run yet at this point —
    // static-init order across TUs is unspecified, the same reason this
    // registrar probes the device itself rather than trusting one.
    static RocmPlatform platform(vt::rocm::DeviceMemoryTotalBytes(0));
    RegisterPlatform(DeviceType::kROCM, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
