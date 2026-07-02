// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/tensor.h"

namespace vt {

Tensor Tensor::Contiguous(void* data, DType dtype, Device device,
                          std::initializer_list<int64_t> shape) {
  VT_CHECK(shape.size() >= 1 && shape.size() <= static_cast<size_t>(kMaxRank),
           "rank out of range");
  Tensor t;
  t.data = data;
  t.dtype = dtype;
  t.device = device;
  t.rank = static_cast<int>(shape.size());
  int i = 0;
  for (int64_t s : shape) {
    VT_CHECK(s > 0, "shape dims must be positive");
    t.shape[i++] = s;
  }
  int64_t acc = 1;
  for (int d = t.rank - 1; d >= 0; --d) {
    t.stride[d] = acc;
    acc *= t.shape[d];
  }
  return t;
}

int64_t Tensor::Numel() const {
  int64_t n = 1;
  for (int d = 0; d < rank; ++d) n *= shape[d];
  return n;
}

bool Tensor::IsContiguous() const {
  int64_t acc = 1;
  for (int d = rank - 1; d >= 0; --d) {
    if (stride[d] != acc) return false;
    acc *= shape[d];
  }
  return true;
}

size_t Tensor::Bytes() const {
  VT_CHECK(IsContiguous(), "Bytes() requires a contiguous tensor");
  return static_cast<size_t>(Numel()) * SizeOf(dtype);
}

Tensor Tensor::View(std::initializer_list<int64_t> new_shape) const {
  VT_CHECK(IsContiguous(), "View() requires a contiguous tensor");
  Tensor v = Contiguous(data, dtype, device, new_shape);
  VT_CHECK(v.Numel() == Numel(), "View() numel mismatch");
  return v;
}

Tensor Tensor::Slice(int dim, int64_t start, int64_t stop) const {
  VT_CHECK(dim >= 0 && dim < rank, "Slice() dim out of range");
  VT_CHECK(start >= 0 && start < stop && stop <= shape[dim], "Slice() bounds invalid");
  Tensor s = *this;
  s.shape[dim] = stop - start;
  s.data = static_cast<char*>(data) +
           static_cast<size_t>(start * stride[dim]) * SizeOf(dtype);
  return s;
}

int64_t Tensor::Offset(std::initializer_list<int64_t> idx) const {
  VT_CHECK(static_cast<int>(idx.size()) == rank, "Offset() rank mismatch");
  int64_t off = 0;
  int d = 0;
  for (int64_t i : idx) {
    VT_CHECK(i >= 0 && i < shape[d], "Offset() index out of range");
    off += i * stride[d++];
  }
  return off;
}

}  // namespace vt
