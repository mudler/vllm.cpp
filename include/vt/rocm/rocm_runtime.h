// HIP-free declarations of the probes the ROCm backend exposes upward
// (BACKEND-ROCM, W0 + the W1 approach-(b) introspection pair). Mirrors the role
// of src/vt/vulkan/vulkan_context.h in the Vulkan skeleton: the engine-side
// platform TU asks "is there a device?" without ever including
// <hip/hip_runtime.h>, which is what keeps src/vllm/ free of vendor headers and
// lets the platform leg be read as plain C++.
//
// All are defined in src/vt/rocm/rocm_backend.hip and all are noexcept: they
// are called from static-init registrars, where throwing would abort the process
// at load time on a machine that merely happens to have HIP installed.
#pragma once

#include <cstddef>
#include <string>

namespace vt::rocm {

// True when the HIP runtime is present AND reports at least one usable device.
// False on any error, which is the conservative answer: a platform whose
// backend() would throw is worse than an unregistered platform, because
// CurrentPlatform() must be able to fall through to CPU.
bool DeviceAvailable() noexcept;

// hipDeviceProp_t::gcnArchName for `index` (e.g. "gfx1100", or
// "gfx942:sramecc+:xnack-"), empty when the device is not present. For error
// messages, test output and bug reports: a ROCm issue that names the arch is
// actionable, one that says "an AMD GPU" is not.
std::string DeviceArchName(int index) noexcept;

// The raw hipDeviceAttributeIntegrated probe for `index`; false when the device
// is absent or the probe fails. HIP-free so tests can branch integrated vs
// discrete without a device header.
bool IntegratedDevice(int index) noexcept;

// ENG-EXPERT-STREAM-DEVICE W0b (issue #1124): may a kernel on device `index`
// DEREFERENCE ordinary host storage? `hipDeviceAttributeIntegrated AND
// hipDeviceAttributePageableMemoryAccess`, both of which ProbeDevice already
// reads; false when the device is absent or either probe fails.
//
// NOT `RocmBackend::UnifiedMemory()`, which the registrar widens with the
// managed-allocation branch: hipMallocManaged makes the BACKEND's own pointers
// migratable and says nothing about a host `std::vector`'s. HIP-free so the
// platform leg can answer without a device header.
bool HostMemoryIsDeviceAddressable(int index) noexcept;

// Which allocation path Backend::Alloc takes for device `index` — the
// approach-(b) introspection seam (issue #41 F6, maintainer-ratified
// 2026-08-08). True: every Backend::Alloc block is
// hipMallocManaged(hipMemAttachGlobal), which is what makes UnifiedMemory()
// true BY CONSTRUCTION on integrated, managed-capable devices (Strix Halo
// gfx1151, Radeon 780M gfx1103). False: plain hipMalloc — every discrete card,
// byte-identical to the W0 behavior — or no device at `index`.
//
// Exists so (a) tests/vt/test_rocm_backend.cpp can assert the discrete branch
// is provably dead without a HIP header, and (b) a board owner can report which
// path their silicon took without reading driver internals.
bool ManagedAllocActive(int index) noexcept;

// BACKEND-ROCM, issue #1934. `hipMemGetInfo`'s `total` for device `index`, in
// bytes; 0 when the device is absent or the probe fails. HIP-free so the
// PLATFORM registrar (static-init time, unspecified cross-TU order — the same
// reasoning `DeviceAvailable()` above states) can read it without depending on
// `RocmBackend`'s own registrar having already run. Mirrors
// `platforms/cuda.cpp`'s own `cudaMemGetInfo` probe at registration, which
// this project's ResidencyPolicy::device_memory_total_bytes doc already
// specifies as "TOTAL rather than FREE, because free at load time carries the
// page cache and whatever else the box is doing" — same reasoning applies to
// HIP's allocator. 0 == UNKNOWN, which `gguf_device_fit.h`'s load-time
// refusal already reads as "do not decide", never as "nothing fits".
size_t DeviceMemoryTotalBytes(int index) noexcept;

}  // namespace vt::rocm
