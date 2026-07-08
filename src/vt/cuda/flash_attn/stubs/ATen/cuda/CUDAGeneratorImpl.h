// Torch-free stub of <ATen/cuda/CUDAGeneratorImpl.h> for the vendored FA-2 kernel.
// vLLM's flash-attention (csrc/flash_attn/src/flash.h) only needs at::PhiloxCudaState
// as a POD field (params.philox_args) feeding the dropout RNG. Inference runs with
// dropout disabled, so the seed/offset are unused at runtime; this POD keeps the
// kernel torch-free. Source: vllm-project/flash-attention @ 2c839c33.
#pragma once
#include <cstdint>

namespace at {
struct PhiloxCudaState {
  PhiloxCudaState() = default;
  PhiloxCudaState(uint64_t seed, uint64_t offset) : seed_(seed), offset_(offset) {}
  uint64_t seed_ = 0;
  uint64_t offset_ = 0;
};
struct Generator {};
}  // namespace at
