// Vulkan backend — allocation registry mapping a raw `void*` (what vt::Tensor
// carries) back to the owning VkBuffer plus a byte offset.
// BACKEND-VULKAN, W0 skeleton.
//
// WHY this exists — the SAME problem the Metal skeleton solved
// (src/vt/metal/metal_buffers.h), and llama.cpp solves on Vulkan too.
// `vt::Tensor::data` is a plain pointer and views/slices (`Tensor::Slice`,
// `Tensor::View`) hand out INTERIOR pointers into an allocation, but Vulkan
// binds RESOURCES (VkBuffer descriptors), not pointers. So the backend keeps an
// interval map of its own allocations and resolves any pointer inside one to
// (buffer, byte offset) at record time. llama.cpp does exactly this walk in
// `ggml_vk_host_get` (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:7416` @ 237ad9b96),
// used by `ggml_vk_tensor_subbuffer` (:7431) to turn a tensor's `data` into a
// buffer+offset pair.
//
// ONE DIFFERENCE FROM llama.cpp, and it is deliberate. `ggml_vk_tensor_subbuffer`
// puts the offset into the DESCRIPTOR and therefore has to assert the offset is
// a multiple of `minStorageBufferOffsetAlignment` (:7448-7451), refusing
// misaligned views unless the shader opts in. We bind every buffer WHOLE and
// pass the byte offset through the push constants instead, so no interior
// pointer is ever rejected for alignment. See
// src/vt/vulkan/shaders/vt_common.glsl § STORAGE MODEL.
//
// Buffers are HOST_VISIBLE|HOST_COHERENT and PERSISTENTLY MAPPED, so
// `Backend::Copy`/`Memset` are plain memcpy/memset once the queue is idle —
// which is why the gate keeps BIT-EXACTNESS for the pure copy/layout paths while
// reductions only owe NMSE.
#ifndef VT_VULKAN_VULKAN_BUFFERS_H_
#define VT_VULKAN_VULKAN_BUFFERS_H_

#include <cstddef>
#include <cstdint>

namespace vt::vulkan {

// Registers an allocation so Resolve() can find it. `base` is the mapped host
// pointer; `buffer` and `memory` are the packed VkBuffer / VkDeviceMemory.
void RegisterAllocation(void* base, size_t bytes, void* buffer, void* memory);

// Unregisters `base` (which must be an allocation BASE, not an interior
// pointer) and hands back its handles. Returns false if `base` is unknown.
bool UnregisterAllocation(void* base, void** out_buffer, void** out_memory);

// Resolves any pointer inside a registered allocation. Throws (VT_CHECK) with an
// actionable message when the pointer was not allocated through the Vulkan
// backend — that is THE bring-up mistake (handing a Vulkan kernel a host
// `std::vector`'s data), and failing loudly beats reading garbage.
struct Resolved {
  void* buffer = nullptr;  // packed VkBuffer
  uint32_t offset = 0;     // BYTE offset from the buffer's start
};
Resolved Resolve(const void* ptr, const char* what);

}  // namespace vt::vulkan

#endif  // VT_VULKAN_VULKAN_BUFFERS_H_
