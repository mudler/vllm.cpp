#pragma once
#include <cstdint>
#include <tuple>
#include <ATen/cuda/CUDAGeneratorImpl.h>
namespace at { namespace cuda { namespace philox {
__host__ __device__ inline std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
  return std::make_tuple(arg.seed_, arg.offset_);
}
}}} // namespace at::cuda::philox
