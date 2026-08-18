// Shared Tenstorrent mesh-device lifecycle for the backend, op providers, and
// the platform registrar (BACKEND-TENSTORRENT W0, .agents/specs/
// tenstorrent-backend.md). vllm.cpp original; no upstream mirror.
#pragma once

#include <cstdint>
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
// HOST-FREE-FORWARD R2: when capture is active and BOTH dst and src carry a
// current device shadow, do a device->device copy (ttnn) instead of staging
// through host. Returns true if it performed a device copy, false if the
// caller should fall back to host memcpy.
bool CopyDeviceDeviceIfCapture(void* dst, const void* src);
// HOST-FREE-FORWARD R3: when capture/host-free is active, fill the buffer's
// device shadow on-device (ttnn::zeros, matching the shadow's own
// shape/dtype) instead of host memset. Only value==0 is handled; every other
// value declines so Backend::Memset falls back to host memset. Requires the
// buffer to already carry a current device shadow.
bool MemsetDeviceIfCapture(void* p, int value);

// ITEM 5 (rope): driver-side warm hook — populate the persistent device
// cos/sin tensors for the step's positions BEFORE BeginCapture (the
// SizeSlot::Refresh slot), so the captured rope cache-HITs. No-op unless
// VT_TT_HOST_FREE_DECODE is set. vt::RopeArgs is declared in vt/ops.h
// (included by every TU that needs the args); this header stays ttnn-free.
// (Plain-field args keep this header free of vt/ops.h; llama3 rope scaling
// is NOT supported on the warm path — TT host-free decode is Qwen3/Mistral
// plain-rope only, matching the current allowlist.)
// ITEM 5 (RAC): eagerly create the paged-KV device shadow (ttnn tensor) for
// the given k_cache / v_cache host buffers during warmup, so the captured
// RAC + PA find the shadow without an in-region upload. No-op unless
// VT_TT_HOST_FREE_DECODE. Takes raw host ptrs + geometry (no ttnn types).
#ifdef VLLM_CPP_TENSTORRENT
void WarmPagedKvShadow(void* k_cache_data, void* v_cache_data,
                      int64_t num_blocks, int64_t block_size,
                      int64_t num_kv_heads, int64_t head_size,
                      int64_t used_blocks);
#else
inline void WarmPagedKvShadow(void*, void*, int64_t, int64_t, int64_t, int64_t,
                              int64_t) {}
#endif

// ITEM 5 (RAC): stage the persistent device update-idx / page-table tensors
// for THIS slot mapping, outside capture (driver Refresh slot). No-op unless
// VT_TT_HOST_FREE_DECODE. slot_mapping_owner is the host buffer the captured
// ReshapeAndCache will see as its slot_mapping (keyed identity). page_table is
// the block table (virtual→physical block mapping, int32, [num_reqs, cols]).
// positions are the sequence positions for this step (int32). Both are used to
// build the update_idxs (positions) and page_table tensors paged_update_cache
// reads at replay time.
#ifdef VLLM_CPP_TENSTORRENT
void WarmRacIdx(const void* slot_mapping_owner, const int64_t* slots,
                int64_t num_slots, int64_t block_size,
                const int32_t* page_table, int64_t page_table_cols,
                const int32_t* positions);
#else
inline void WarmRacIdx(const void*, const int64_t*, int64_t, int64_t,
                       const int32_t*, int64_t, const int32_t*) {}
#endif

// ITEM 5 (PA): warm persistent page_table + cur_pos device tensors.
// When `advance_on_device` is true, cur_pos is seeded once and then advanced
// in-trace via CaptureDecodePosAdvance (ttnn::plus_one); WarmPaMeta's cur_pos
// copy_to_device is skipped in steady-state replay (R2 on-device state advance).
#ifdef VLLM_CPP_TENSTORRENT
void WarmPaMeta(const int32_t* block_table, int64_t num_reqs, int64_t max_blocks,
                int64_t bt_row_stride, int64_t bt_col_stride,
                const int32_t* seq_lens);
#else
inline void WarmPaMeta(const int32_t*, int64_t, int64_t, int64_t, int64_t,
                       const int32_t*) {}
#endif

// R2 (on-device state advance): seed the persistent cur_pos device tensor
// (= seq_lens - 1) for this step. Called on the capture/warm step (re-seed),
// NOT every replay step — the captured plus_one advances it on-device.
// Also warms the plus_one program (program cache) so CaptureDecodePosAdvance
// can run inside the trace without a "load new binaries during capture" fatal.
#ifdef VLLM_CPP_TENSTORRENT
void WarmDecodePos(const int32_t* seq_lens, int64_t num_reqs);
#else
inline void WarmDecodePos(const int32_t*, int64_t) {}
#endif

// R2: capture ttnn::plus_one(cur_pos) at the END of the trace body (after all
// reads of cur_pos in sdpa_decode / paged_update_cache), so the NEXT replay
// sees cur_pos+1. Must be called INSIDE BeginCapture/EndCapture.
#ifdef VLLM_CPP_TENSTORRENT
void CaptureDecodePosAdvance(int64_t num_reqs);
#else
inline void CaptureDecodePosAdvance(int64_t) {}
#endif

// HOST-FREE-DECODE: stage the persistent device decode-ids tensor (UINT32 [n],
// ROW_MAJOR) for THIS step's token ids, outside capture. The captured
// embedding reads this stable address; each replay step only refreshes its
// content (copy_to_device — allocation-free). No-op unless
// VT_TT_HOST_FREE_DECODE.
#ifdef VLLM_CPP_TENSTORRENT
void WarmDecodeIds(const int32_t* ids, int64_t n);
#else
inline void WarmDecodeIds(const int32_t*, int64_t) {}
#endif

// HOST-FREE-DECODE: capture-safe embedding over the persistent decode-ids
// tensor (see WarmDecodeIds) into the hidden buffer whose host base is
// `out_host` — the same device->device in-place refresh EmbeddingKernel
// performs on its warm path, but with no host ids upload, so it can run
// INSIDE the captured region. `table_host` is the embedding table's host
// base (device shadow cached by the kEmbedding path). Requires WarmDecodeIds
// to have staged ids for this n first.
#ifdef VLLM_CPP_TENSTORRENT
void EmbedDeviceIdsInto(void* out_host, int64_t rows, int64_t cols,
                        const void* table_host, int64_t vocab, int64_t hidden,
                        int64_t n);
#else
inline void EmbedDeviceIdsInto(void*, int64_t, int64_t, const void*, int64_t,
                               int64_t, int64_t) {}
#endif



#ifdef VLLM_CPP_TENSTORRENT
void WarmRopeCosSin(const int32_t* positions, int64_t tokens, int64_t hq,
                    int64_t hk, int64_t rot, double base);
#else
inline void WarmRopeCosSin(const int32_t*, int64_t, int64_t, int64_t, int64_t,
                           double) {}
#endif

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
