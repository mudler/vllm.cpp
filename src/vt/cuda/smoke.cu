#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/cuda_smoke.h"

namespace vt::cuda {
namespace {
__global__ void AddKernel(int a, int b, int* out) { *out = a + b; }

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}
}  // namespace

int SmokeAdd(int a, int b) {
  int* d_out = nullptr;
  Check(cudaMalloc(&d_out, sizeof(int)), "cudaMalloc");
  AddKernel<<<1, 1>>>(a, b, d_out);
  Check(cudaGetLastError(), "launch");
  int out = 0;
  Check(cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  Check(cudaFree(d_out), "cudaFree");
  return out;
}
}  // namespace vt::cuda
