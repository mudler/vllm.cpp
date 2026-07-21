// Ported from: vllm/v1/attention/backends/registry.py (AttentionBackendEnum +
// @register_backend self-registration) and the platform-driven selection in
// vllm/platforms/cuda.py:361-470 (get_valid_backends / get_attn_backend_cls) @
// pin e24d1b24 — the attention-backend REGISTRY + platform-priority SELECTION
// seam (extensibility item 4). This is the ENGINE-level "which AttentionBackend"
// seam; the concrete attention KERNEL stays selected at the vt:: op-table level
// (vt::PagedAttention -> GetOp(kPagedAttention, device.type)), which is already
// device-additive.
//
// Design (mirrors upstream, realized with our proven static-Registrar idiom):
//   * Backends SELF-REGISTER per DeviceType via a static AttentionBackendRegistrar
//     (copies the vt RegisterOp / RegisterPlatform / REGISTER_VLLM_MODEL static-
//     init idiom), keyed on (DeviceType, name). This is upstream's
//     @register_backend(AttentionBackendEnum.X).
//   * A Platform advertises a capability-ordered priority list of backend NAMES
//     (Platform::get_attn_backend_priority) — upstream's _get_backend_priorities.
//   * SelectAttentionBackendName walks that list and returns the first REGISTERED
//     backend — upstream's get_attn_backend_cls picking the min-priority valid
//     backend (here "registered for this device" IS the validity check: an
//     unregistered name is simply skipped, exactly as an ImportError-ing backend
//     is in upstream get_valid_backends).
// Selection is therefore DATA (a registered backend + a platform priority), not
// an inline code edit: adding a new backend's attention = one self-registering
// TU + its slot in the owning platform's priority, ZERO edit here or in
// model/runner attention code.
#ifndef VLLM_V1_ATTENTION_REGISTRY_H_
#define VLLM_V1_ATTENTION_REGISTRY_H_

#include <memory>
#include <string>

#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/backend.h"
#include "vt/device.h"

namespace vllm::v1 {

// Produces a fresh AttentionBackend instance (mirrors upstream get_class(): the
// registry stores the class/factory, callers construct on demand).
using AttentionBackendFactory = std::unique_ptr<AttentionBackend> (*)();

// Register `name` for `device`. A backend that supports several device types
// registers once per type (e.g. FLASH_ATTN for both kCUDA and kCPU). Registration
// must complete before main() (static initializers); afterwards these are
// lock-free reads. Re-registering the same (device, name) overwrites (idempotent
// with an identical factory).
void RegisterAttentionBackend(vt::DeviceType device, const std::string& name,
                              AttentionBackendFactory factory);

// True when a backend of `name` is registered for `device`.
bool HasAttentionBackend(vt::DeviceType device, const std::string& name);

// Construct the named backend for `device` (throws if unregistered). Mirrors
// AttentionBackendEnum.get_class()().
std::unique_ptr<AttentionBackend> MakeAttentionBackend(vt::DeviceType device,
                                                       const std::string& name);

// The selected backend NAME for `platform`. If `selected` is non-empty it is an
// explicit user override (upstream VLLM_ATTENTION_BACKEND / the selected_backend
// arg of get_attn_backend_cls): it must be registered for the platform's device
// or this throws. Otherwise walk the platform's capability-ordered priority list
// and return the first name registered for the platform's device (upstream's
// min-priority valid backend). Throws if the priority list yields no registered
// backend. Behavior-preserving today: on CUDA (every capability we ship on) and
// CPU this returns "FLASH_ATTN" — the backend whose NHD KV layout the runtime
// already uses.
// `cfg` (W2) carries upstream's `use_mla` plus the `use_sparse` flag. It selects
// which priority list the platform hands back (the MLA vs non-MLA branch of
// _get_backend_priorities) AND is applied as a per-candidate FILTER mirroring
// vllm/v1/attention/backend.py:307-360 validate_configuration: a candidate whose
// `is_mla()` / `is_sparse()` disagrees with the request is skipped exactly as an
// unregistered name is. Defaulted, so pre-W2 dense call sites are unchanged.
//
// THE DSA / SPARSE-MLA SEAM: GB10's MLA list is [TRITON_MLA,
// FLASHINFER_MLA_SPARSE_SM120]. Today the sparse entry is unregistered AND would
// be filtered on is_sparse(); when the sparse-MLA campaign lands its backend, it
// registers with `is_sparse() == true` and is selected for `use_sparse=true`
// requests with NO edit to this selector or to the platform priority table.
std::string SelectAttentionBackendName(
    const platforms::Platform& platform, const std::string& selected = "",
    const platforms::AttnSelectorConfig& cfg = platforms::AttnSelectorConfig{});

// SelectAttentionBackendName + construct the instance.
std::unique_ptr<AttentionBackend> SelectAttentionBackend(
    const platforms::Platform& platform, const std::string& selected = "",
    const platforms::AttnSelectorConfig& cfg = platforms::AttnSelectorConfig{});

// Static-init self-registration helper (copies the vt-op / platform Registrar
// idiom). Declare one file-scope instance per (device, backend) in the backend's
// own TU: `const AttentionBackendRegistrar kReg{DeviceType::kCUDA, "FLASH_ATTN",
// []{ return std::unique_ptr<AttentionBackend>(new FlashAttentionBackend()); }};`.
struct AttentionBackendRegistrar {
  AttentionBackendRegistrar(vt::DeviceType device, const std::string& name,
                            AttentionBackendFactory factory) {
    RegisterAttentionBackend(device, name, factory);
  }
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ATTENTION_REGISTRY_H_
