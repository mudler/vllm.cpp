// vllm.cpp trace-only CUDA graph replay profiler control.
//
// This seam is compiled only with VLLM_CPP_BENCH_PROFILE_CONTROL=ON. It is
// intentionally absent from production builds so ordinary ReplayGraph calls
// carry no observer branch. The online parity harness arms it with SIGUSR2
// only after the exact semantic workload has completed outside collection.
#pragma once

#include <cstdint>

namespace vt::cuda {

// Install the trace-only SIGUSR2 arm and capture exactly `replays` eligible
// Qwen dense graph replays. May be configured once/process.
void ConfigureCudaGraphReplayProfiler(uint32_t replays);

// Mark the immediately following ReplayGraph call with its model-owned graph
// identity and warm replay state. Only the exact warmed B=16/S=16 Qwen dense
// path is eligible; all other graph launches remain outside collection.
void MarkCudaGraphReplayProfilerEligible(void* graph, uint32_t real_batch,
                                         uint32_t padded_batch,
                                         uint64_t prior_replays);

}  // namespace vt::cuda
