// Ported from: vllm/config/device.py @ 555967922 (Device Literal:13,
//               DeviceConfig:16-78).
//
// Scope (ARCH-ONE-SURFACE fold ROW 8): the explicit device-selection surface —
// the mirror of vLLM's `DeviceConfig.device` names. Upstream the value set is
// `Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]` (device.py:13);
// "auto" resolves through the platform probe (device.py:49-60) and an explicit
// name is assigned VERBATIM, never silently substituted (device.py:61-66).
//
// PORT NOTES (recorded deviations):
//   - Only the members this build can serve are selectable: auto/cpu/cuda.
//     "tpu" has no backend here; xpu/vulkan/metal/rocm are reachable through
//     the AUTO accelerator-first probe (platform.cpp kCurrentPriority) and
//     gain explicit names ADDITIVELY when a lane needs them.
//   - The integer values are the C-ABI wire contract
//     (vllm_model_params.device, ABI v14): 0 MUST be auto so a zero-initialized
//     struct preserves the pre-v14 accelerator-first behaviour byte for byte;
//     cpu-before-accelerator then follows the shipped v12 precedent
//     (vllm_video_model_params.device: 0 cpu, 1 the accelerator that build
//     resolves) shifted by the auto slot.
#ifndef VLLM_CONFIG_DEVICE_H_
#define VLLM_CONFIG_DEVICE_H_

#include <cstdint>
#include <string>

namespace vllm {

// The device selection an engine is asked to serve on. kAuto is the default
// and the zero value: the accelerator-first platform probe that has always
// selected the queue (CurrentPlatform(), src/vllm/platforms/platform.cpp).
enum class Device : int32_t {
  kAuto = 0,  // platform-probed, upstream's "auto" default (device.py:20).
  kCPU = 1,   // force the CPU queue even when an accelerator is present.
  // ABI v14 slot 2 currently names "cuda". Keep the enum platform-neutral:
  // shared selection resolves DeviceName() through the platform registry.
  kNamedPlatform = 2,
};

// Parse the wire/CLI name ("auto" | "cpu" | "cuda" — the supported subset of
// upstream's Device Literal, device.py:13). Throws std::invalid_argument on
// any other name, mirroring pydantic rejecting a non-Literal value.
Device DeviceFromString(const std::string& value);

// The canonical name for a selection (static storage; never nullptr).
const char* DeviceName(Device device);

}  // namespace vllm

#endif  // VLLM_CONFIG_DEVICE_H_
