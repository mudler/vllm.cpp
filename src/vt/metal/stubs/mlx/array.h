// Syntax-check stub for mlx/array.h. See ../README.md.
//
// `array` is the one MLX type the provider builds by hand, and the surface below
// is precisely what `WrapTensor` and `TryMlxMatmul` name: the four-argument
// constructor, `set_data` with a caller-supplied deleter (the no-op deleter is
// what keeps buffer ownership ours), `set_status` (a freshly `set_data`'d array
// is `unscheduled`, and the provider marks it `available`), and the three
// readers used after the eval boundary.
//
// No member is DEFINED. A syntax gate needs declarations; the object library
// this belongs to is never linked, so an undefined member is never missed.
#ifndef VT_METAL_STUBS_MLX_ARRAY_H_
#define VT_METAL_STUBS_MLX_ARRAY_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/dtype.h"

namespace mlx::core {

using Shape = std::vector<int32_t>;
using Strides = std::vector<int64_t>;

class Primitive;

class array {
 public:
  struct Flags {
    bool contiguous;
    bool row_contiguous;
    bool col_contiguous;
  };
  enum class Status { unscheduled, scheduled, available };
  using Deleter = std::function<void(allocator::Buffer)>;

  array(Shape shape, Dtype dtype, std::shared_ptr<Primitive> primitive,
        std::vector<array> inputs);

  void set_data(allocator::Buffer buffer, size_t data_size, Strides strides, Flags flags,
                Deleter deleter);
  void set_status(Status status) const;

  Dtype dtype() const;
  size_t nbytes() const;
  template <typename T>
  T* data();
};

}  // namespace mlx::core

#endif  // VT_METAL_STUBS_MLX_ARRAY_H_
