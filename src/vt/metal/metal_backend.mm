// Metal backend — the `vt::Backend` implementation + its static registrar.
// BACKEND-METAL-MLX, W0 skeleton. vllm.cpp original (vt runtime, inventory
// deviation §9.1): vLLM has no Metal platform, so there is no upstream mirror;
// the SHAPE is the CPU reference `src/vt/cpu/cpu_backend.cpp` (the 6 pure
// virtuals, 36 lines) and the Metal resource handling is ported from llama.cpp
// `ggml/src/ggml-metal/ggml-metal.m` @ 237ad9b96 (see metal_buffers.h).
//
// SCOPE / STUBS — stated plainly so nothing here reads as more than it is:
//   * Every op is dispatched SYNCHRONOUSLY (commit + waitUntilCompleted, see
//     metal_ops.mm). `Synchronize` is therefore a no-op and `Copy`/`Memset` are
//     host memcpy/memset over the shared-storage allocation. That is CORRECT but
//     not fast; async command-buffer pipelining is deferred.
//   * `SupportsGraphCapture()` stays FALSE. `MTLIndirectCommandBuffer` is the
//     eventual mapping (include/vt/backend.h:92 already names it) and is NOT
//     implemented here.
//   * The async-output primitives (AllocPinned / events) inherit the
//     `vt::Backend` defaults, which src/vt/backend.cpp documents as already
//     correct for UNIFIED-memory backends — which Apple silicon is.
//   * `DeviceCapabilityMajor/Minor` report the Apple GPU family as {N, 0}.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "metal_buffers.h"
#include "metal_context.h"
#include "vt/backend.h"

namespace vt::metal {
namespace {

struct AllocEntry {
  size_t bytes = 0;
  void* buffer = nullptr;  // id<MTLBuffer>
};

// Interval map keyed by allocation base address. Guarded because allocation can
// legitimately race with encode on different threads.
std::mutex& AllocMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, AllocEntry>& AllocMap() {
  static std::map<uintptr_t, AllocEntry> m;
  return m;
}

class MetalBackend final : public Backend {
 public:
  void* Alloc(size_t bytes) override {
    // A zero-length MTLBuffer is invalid; round up so a 0-byte request still
    // yields a distinct freeable pointer (the CPU backend's contract).
    const size_t len = bytes == 0 ? 1 : bytes;
    id<MTLDevice> dev = static_cast<id<MTLDevice>>(MetalContext::Get().device());
    id<MTLBuffer> buf = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
    VT_CHECK(buf != nil, "metal: newBufferWithLength failed");
    void* base = [buf contents];
    // MTLBuffer contents is page-aligned, so the >= 64-byte alignment
    // vt::StepArena depends on (include/vt/backend.h:26) holds by construction.
    RegisterAllocation(base, len, static_cast<void*>(buf));
    return base;
  }

  void Free(void* p) override {
    if (p == nullptr) return;
    void* buf = UnregisterAllocation(p);
    VT_CHECK(buf != nullptr, "metal: Free() on a pointer this backend did not allocate");
    [static_cast<id<MTLBuffer>>(buf) release];
  }

  // Shared storage + synchronous dispatch => the host may touch the bytes
  // directly. Both are BIT-EXACT byte operations, which is why the gate keeps
  // bit-exactness for pure copy/layout paths while reductions only owe NMSE.
  void Memset(Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }

  // One process-wide MTLCommandQueue is shared by every vt::Queue: the queue
  // handle is the ORDERING domain and, with synchronous dispatch, every op is
  // already ordered. `id` still makes each vt::Queue a distinct identity for the
  // workspace-key machinery (src/vt/ops.cpp MakeWorkspaceKey).
  Queue CreateQueue() override {
    return Queue{Device{DeviceType::kMETAL, 0}, MetalContext::Get().command_queue()};
  }

  bool UnifiedMemory() const override { return MetalContext::Get().unified_memory(); }

  int DeviceCapabilityMajor() const override {
    return MetalContext::Get().gpu_family_apple();
  }
  int DeviceCapabilityMinor() const override { return 0; }
};

struct Registrar {
  Registrar() {
    // A Metal-ENABLED build may still run where no Metal device exists (a
    // headless CI VM). Probing first keeps that a clean "kMETAL not registered"
    // — the state src/vt/ops.cpp:104-111 already treats as supported — instead
    // of a throw during static initialization, which would abort the process.
    if (!MetalContext::Available()) return;
    static MetalBackend backend;
    RegisterBackend(DeviceType::kMETAL, &backend);
  }
} registrar;

}  // namespace

void RegisterAllocation(void* base, size_t bytes, void* buffer) {
  std::lock_guard<std::mutex> g(AllocMutex());
  AllocMap()[reinterpret_cast<uintptr_t>(base)] = AllocEntry{bytes, buffer};
}

void* UnregisterAllocation(void* base) {
  std::lock_guard<std::mutex> g(AllocMutex());
  auto& m = AllocMap();
  auto it = m.find(reinterpret_cast<uintptr_t>(base));
  if (it == m.end()) return nullptr;
  void* buf = it->second.buffer;
  m.erase(it);
  return buf;
}

Resolved Resolve(const void* ptr, const char* what) {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> g(AllocMutex());
  auto& m = AllocMap();
  // upper_bound gives the first allocation starting AFTER addr; the candidate
  // container is its predecessor. Same interior-pointer walk as llama.cpp's
  // ggml_metal_get_buffer.
  auto it = m.upper_bound(addr);
  if (it != m.begin()) {
    --it;
    if (addr >= it->first && addr < it->first + it->second.bytes) {
      return Resolved{it->second.buffer, static_cast<size_t>(addr - it->first)};
    }
  }
  VT_CHECK(false, std::string("metal: ") + what +
                      " points outside every Metal allocation — Metal kernels can only "
                      "bind memory obtained from vt::GetBackend(DeviceType::kMETAL).Alloc()");
  return Resolved{};
}

}  // namespace vt::metal
