// Metal backend — allocation registry mapping a raw `void*` (what vt::Tensor
// carries) back to the owning `MTLBuffer` plus a byte offset.
// BACKEND-METAL-MLX, W0 skeleton.
//
// WHY this exists. `vt::Tensor::data` is a plain pointer and views/slices
// (`Tensor::Slice`, `Tensor::View`) hand out INTERIOR pointers into an
// allocation, but Metal binds RESOURCES (`setBuffer:offset:atIndex:`), not
// pointers. So the backend keeps an interval map of its own allocations and
// resolves any pointer inside one to (buffer, offset) at encode time. This is
// the same problem llama.cpp solves with `ggml_metal_get_buffer`
// (`ggml/src/ggml-metal/ggml-metal.m` @ 237ad9b96), which likewise walks the
// registered buffer ranges to turn a tensor's `data` into a buffer+offset pair.
//
// Buffers are `MTLResourceStorageModeShared`: on Apple silicon host and device
// see one copy, so `Backend::Copy`/`Memset` are plain memcpy/memset once the
// queue is idle, and `UnifiedMemory()` is true.
#ifndef VT_METAL_METAL_BUFFERS_H_
#define VT_METAL_METAL_BUFFERS_H_

#include <cstddef>
#include <cstdint>

namespace vt::metal {

// Registers an allocation so Resolve() can find it. `base` is the buffer's
// contents pointer, `buffer` the retained id<MTLBuffer>.
void RegisterAllocation(void* base, size_t bytes, void* buffer);
// Unregisters and returns the id<MTLBuffer> that owned `base` (nullptr if
// unknown). `base` must be an allocation BASE, not an interior pointer.
void* UnregisterAllocation(void* base);

// Resolves any pointer inside a registered allocation. Throws (VT_CHECK) with an
// actionable message when the pointer was not allocated through the Metal
// backend — that is the common bring-up mistake (handing a Metal kernel a host
// `std::vector`'s data), and failing loudly beats reading garbage.
struct Resolved {
  void* buffer = nullptr;  // id<MTLBuffer>
  size_t offset = 0;
};
Resolved Resolve(const void* ptr, const char* what);

}  // namespace vt::metal

#endif  // VT_METAL_METAL_BUFFERS_H_
