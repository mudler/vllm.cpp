// Shared Tenstorrent mesh-device lifecycle for the backend, op providers, and
// the platform registrar (BACKEND-TENSTORRENT W0, .agents/specs/
// tenstorrent-backend.md). vllm.cpp original; no upstream mirror.
#pragma once

#include <memory>

// ttnn::MeshDevice (ttnn/api/ttnn/device.hpp) is a `using` alias for this real
// type, brought into ttnn:: scope via `using namespace device;` there — the
// alias itself is not forward-declarable, so this header names the concrete
// type it resolves to.
namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace vt::tenstorrent {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// True iff at least one Tenstorrent device is enumerable on this host. Cheap
// (tt::tt_metal::GetNumAvailableDevices(), no device open) — the same
// runtime-probe-before-registering shape vulkan_context.h's Available() uses,
// so a Tenstorrent-enabled build on a host with no Blackhole card registers
// nothing instead of throwing during static init.
bool DeviceAvailable();

// Opens (lazily, once) and returns the single process-wide mesh device this
// W0 skeleton targets — device index 0 only, no multi-device mesh. Throws if
// DeviceAvailable() was not already checked true. DELIBERATELY LEAKED (see
// the .cpp): a plain static shared_ptr's destructor reproducibly segfaults at
// process exit, inside MeshDevice's own teardown chain, a cross-DSO static
// destruction ordering hazard the real device-driver state survives fine
// either way.
MeshDevice& SharedMeshDevice();

// Host-allocation registry for device-resident shadows (BACKEND-TENSTORRENT
// residency). Implemented in tenstorrent_ops.cpp (the only TU that links
// ttnn); called from TenstorrentBackend::Alloc/Free/Copy so Free drops any
// cached ttnn::Tensor that still owns device pages for that host pointer.
// No-ops until the ops registrar has loaded (static init order: backend may
// Free before ops if a test tears down early — Unregister is tolerant).
void RegisterHostBuffer(void* host, size_t bytes);
void UnregisterHostBuffer(void* host);
// Host bytes at `host` (or any interior pointer into that allocation) were
// written by a host-side path (Backend::Copy, Memset, host op). Invalidates
// any device shadow so the next EnsureDevice re-uploads.
void MarkHostWritten(void* host);
// If `host` is inside a registered buffer whose device shadow is the source
// of truth, download to host. Used by Backend::Copy so D2H-style reads see
// device-resident results without every op writing host eagerly.
void EnsureHostBytes(void* host);

// ---- ttnn mesh-trace capture (Backend graph-capture mapping) --------------
// Maps vt::Backend::{BeginCapture,EndCapture,Replay} onto
// ttnn::operations::trace::{begin,end,execute}_trace_capture. Implemented in
// tenstorrent_ops.cpp so the backend TU stays free of ttnn headers.
//
// Contract (same class as CUDA graphs): every op between Begin and End must
// stay async on the mesh CQ with fixed device buffers; host Ensure/Download
// and fresh program compiles during capture are illegal and will throw.
// Warm the program cache with an identical shape before BeginCapture.
void TraceBeginCapture();
void TraceEndCapture();           // stores the default single-slot replay id
void TraceReplay();               // replay the single-slot id
void* TraceEndCaptureGraph();     // returns an opaque MeshTraceId* (caller owns)
void TraceReplayGraph(void* graph);
void TraceDestroyGraph(void* graph);

}  // namespace vt::tenstorrent
