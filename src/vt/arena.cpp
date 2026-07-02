// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/arena.h"

namespace vt {

namespace {
constexpr size_t kAlign = 64;
size_t AlignUp(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
}  // namespace

StepArena::StepArena(Device device, size_t capacity_bytes)
    : device_(device), capacity_(capacity_bytes) {
  base_ = static_cast<char*>(GetBackend(device.type).Alloc(capacity_bytes));
}

StepArena::~StepArena() { GetBackend(device_.type).Free(base_); }

Tensor StepArena::Alloc(DType dtype, std::initializer_list<int64_t> shape) {
  Tensor t = Tensor::Contiguous(nullptr, dtype, device_, shape);
  size_t bytes = AlignUp(static_cast<size_t>(t.Numel()) * SizeOf(dtype));
  VT_CHECK(used_ + bytes <= capacity_, "StepArena overflow");
  t.data = base_ + used_;
  used_ += bytes;
  if (used_ > high_water_) high_water_ = used_;
  return t;
}

void StepArena::Reset() { used_ = 0; }

}  // namespace vt
