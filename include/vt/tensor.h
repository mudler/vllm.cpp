// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cstdint>
#include <initializer_list>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt {

constexpr int kMaxRank = 4;

// Non-owning tensor view. Strides are in ELEMENTS (torch convention).
struct Tensor {
  void* data = nullptr;
  DType dtype = DType::kF32;
  Device device;
  int rank = 0;
  int64_t shape[kMaxRank] = {0, 0, 0, 0};
  int64_t stride[kMaxRank] = {0, 0, 0, 0};

  static Tensor Contiguous(void* data, DType dtype, Device device,
                           std::initializer_list<int64_t> shape);

  int64_t Numel() const;
  bool IsContiguous() const;
  size_t Bytes() const;
  Tensor View(std::initializer_list<int64_t> new_shape) const;
  Tensor Slice(int dim, int64_t start, int64_t stop) const;
  int64_t Offset(std::initializer_list<int64_t> idx) const;

  template <typename T>
  T* Ptr() const {
    return static_cast<T*>(data);
  }
};

}  // namespace vt
