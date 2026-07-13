// Minimal Nsight Systems cudaProfilerApi capture-range calibration probe.
//
// This is deliberately not part of the production build. Compile it with the
// exact DGX CUDA toolkit and run it under the H1d Nsight command to distinguish
// capture-boundary diagnostics from missing graph activities.
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {

__global__ void AddOne(int* value) { *value += 1; }

void Check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool reset = argc == 2 && std::string_view(argv[1]) == "reset";
  int* value = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  Check(cudaMalloc(&value, sizeof(*value)), "cudaMalloc");
  Check(cudaMemset(value, 0, sizeof(*value)), "cudaMemset");
  Check(cudaStreamCreate(&stream), "cudaStreamCreate");
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture");
  AddOne<<<1, 1, 0, stream>>>(value);
  Check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
  Check(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");

  for (int replay = 0; replay < 4; ++replay) {
    Check(cudaProfilerStart(), "cudaProfilerStart");
    Check(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch");
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    Check(cudaProfilerStop(), "cudaProfilerStop");
  }

  int host_value = 0;
  Check(cudaMemcpy(&host_value, value, sizeof(host_value), cudaMemcpyDeviceToHost),
        "cudaMemcpy");

  Check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy");
  Check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  Check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  Check(cudaFree(value), "cudaFree");
  if (reset) Check(cudaDeviceReset(), "cudaDeviceReset");
  if (host_value != 4) {
    std::fprintf(stderr, "unexpected graph result: %d\n", host_value);
    return 1;
  }
  return 0;
}
