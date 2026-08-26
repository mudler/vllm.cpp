// Syntax-check stub for mlx/allocator.h. See ../README.md.
//
// `allocator::Buffer` is the type the zero-copy input path turns one of OUR
// MTLBuffer handles into. The real one is a bare `void*` wrapper
// (mlx/allocator.h:12-29 at MLX 0.32.0, per the provider's own header comment),
// and the provider uses only its explicit `void*` constructor.
#ifndef VT_METAL_STUBS_MLX_ALLOCATOR_H_
#define VT_METAL_STUBS_MLX_ALLOCATOR_H_

namespace mlx::core::allocator {

class Buffer {
 public:
  explicit Buffer(void* ptr) : ptr_(ptr) {}
  void* raw_ptr() const { return ptr_; }

 private:
  void* ptr_;
};

}  // namespace mlx::core::allocator

#endif  // VT_METAL_STUBS_MLX_ALLOCATOR_H_
