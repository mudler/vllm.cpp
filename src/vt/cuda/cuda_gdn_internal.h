// Internal CUDA GDN pool lifecycle and test instrumentation. Not installed.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vt/device.h"

namespace vt::cuda {

// Called by CudaBackend before destroying a stream. This prevents pooled GDN
// scratch from leaking and prevents a recycled CUDA stream handle from
// inheriting an earlier queue's buffers.
void ReleaseGdnTritonScratch(int device, void* stream);

namespace testing {

// Loads every generated GDN module under the runtime's per-module call_once
// guards. Exposed only to exercise concurrent first use from multiple queues.
void WarmGdnTritonAotModules(int device);

// Enqueues a byte-pattern fill over every allocated queue-owned GDN scratch
// buffer. The dirty-buffer test uses 0xff (NaNs / invalid metadata) to prove a
// repeated launch initializes every byte it may consume rather than relying on
// zero-filled fresh allocations. Returns the number of poisoned buffers.
size_t PoisonGdnTritonScratch(Queue& queue, unsigned char value);

struct GdnPackedDecodeDebugStats {
  uint64_t launches = 0;
};

// Model-selection instrumentation: counts host dispatches into the packed CUDA
// operator (one per GDN layer). It is independent of Triton AOT and remains
// valid during eager execution or graph capture; graph replay itself has no
// host dispatch, so W1D3 uses nsys node counts for steady-state structure.
void ResetGdnPackedDecodeDebugStats();
GdnPackedDecodeDebugStats GetGdnPackedDecodeDebugStats();
void DisableGdnPackedDecodeDebugStats();

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
