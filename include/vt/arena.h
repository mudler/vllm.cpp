// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cstddef>
#include <initializer_list>

#include "vt/backend.h"
#include "vt/tensor.h"

namespace vt {

// Per-step bump allocator over one backend slab. Activations live here and
// the whole arena rewinds each engine step. Persistent state (weights, KV
// pages, persistent batch) allocates directly from the Backend instead.
class StepArena {
 public:
  StepArena(Device device, size_t capacity_bytes);
  ~StepArena();
  StepArena(const StepArena&) = delete;
  StepArena& operator=(const StepArena&) = delete;

  Tensor Alloc(DType dtype, std::initializer_list<int64_t> shape);
  void Reset();

  size_t Used() const { return used_; }
  size_t HighWater() const { return high_water_; }
  size_t Capacity() const { return capacity_; }

 private:
  Device device_;
  char* base_ = nullptr;
  size_t capacity_ = 0;
  size_t used_ = 0;
  size_t high_water_ = 0;
};

}  // namespace vt
