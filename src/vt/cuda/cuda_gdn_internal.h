// Internal CUDA GDN pool lifecycle and test instrumentation. Not installed.
#pragma once

#include <cstdint>

namespace vt::cuda {

// Called by CudaBackend before destroying a stream. This prevents pooled GDN
// scratch from leaking and prevents a recycled CUDA stream handle from
// inheriting an earlier queue's buffers.
void ReleaseGdnTritonScratch(int device, void* stream);

namespace testing {

// Loads every generated GDN module under the runtime's per-module call_once
// guards. Exposed only to exercise concurrent first use from multiple queues.
void WarmGdnTritonAotModules(int device);

struct GdnTritonDebugStats {
  uint64_t chunk_o_f32_launches = 0;
  uint64_t chunk_o_bf16_launches = 0;
  uint64_t chunk_o_hand_launches = 0;
  uint64_t chunk_pool_allocations = 0;
  uint64_t chunk_pool_growths = 0;
  uint64_t chunk_pool_reuses = 0;
  uint64_t wu_pool_allocations = 0;
  uint64_t wu_pool_growths = 0;
  uint64_t wu_pool_reuses = 0;
};

// Reset also enables counters. Disable after assertions so production
// benchmarks pay only one relaxed flag load at each instrumented event.
void ResetGdnTritonDebugStats();
GdnTritonDebugStats GetGdnTritonDebugStats();
void DisableGdnTritonDebugStats();

}  // namespace testing
}  // namespace vt::cuda
