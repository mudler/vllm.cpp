// Vulkan backend — the `vt::Backend` implementation + its static registrar, plus
// the allocation registry declared in vulkan_buffers.h. BACKEND-VULKAN, W0
// skeleton. vllm.cpp original (vt runtime, inventory deviation §9.1): vLLM has
// no Vulkan platform, so there is no upstream mirror; the SHAPE is the CPU
// reference `src/vt/cpu/cpu_backend.cpp` (the 6 pure virtuals, 36 lines) and the
// Vulkan resource handling is ported from llama.cpp `ggml/src/ggml-vulkan/
// ggml-vulkan.cpp` @ 237ad9b96 (see vulkan_buffers.h and vulkan_context.h).
//
// SCOPE / STUBS — stated plainly so nothing here reads as more than it is:
//   * Every op is dispatched SYNCHRONOUSLY (record, submit, wait on a fence, see
//     vulkan_context.cpp). `Synchronize` is therefore a no-op and `Copy`/`Memset`
//     are host memcpy/memset over the persistently mapped, host-coherent
//     allocation. That is CORRECT but not fast; async command-buffer pipelining,
//     a transfer queue and a staging path for non-host-visible memory are all
//     deferred.
//   * `SupportsGraphCapture()` stays FALSE. A pre-recorded VkCommandBuffer is
//     the eventual mapping (include/vt/backend.h:92 already names it) and is NOT
//     implemented here.
//   * The async-output primitives (AllocPinned / events) inherit the
//     `vt::Backend` defaults, which src/vt/backend.cpp documents as already
//     correct for UNIFIED-memory backends — which GB10 is (one 89.72 GiB
//     DEVICE_LOCAL|HOST_VISIBLE heap), and which is in any case correct for ANY
//     backend whose transfers are synchronous host memcpy, as these are.
//   * `DeviceCapabilityMajor/Minor` report the VULKAN API VERSION; see
//     vulkan_context.h for why that is the right analogue of CUDA's sm_XY.
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "vulkan_buffers.h"
#include "vulkan_context.h"
#include "vt/backend.h"

namespace vt::vulkan {
namespace {

struct AllocEntry {
  size_t bytes = 0;
  void* buffer = nullptr;  // packed VkBuffer
  void* memory = nullptr;  // packed VkDeviceMemory
};

// Interval map keyed by allocation base address. Guarded because allocation can
// legitimately race with record on different threads.
std::mutex& AllocMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, AllocEntry>& AllocMap() {
  static std::map<uintptr_t, AllocEntry> m;
  return m;
}

class VulkanBackend final : public Backend {
 public:
  void* Alloc(size_t bytes) override {
    void* buffer = nullptr;
    void* memory = nullptr;
    void* base = VulkanContext::Get().AllocBuffer(bytes, &buffer, &memory);
    RegisterAllocation(base, bytes == 0 ? 1 : bytes, buffer, memory);
    return base;
  }

  void Free(void* p) override {
    if (p == nullptr) return;
    void* buffer = nullptr;
    void* memory = nullptr;
    VT_CHECK(UnregisterAllocation(p, &buffer, &memory),
             "vulkan: Free() on a pointer this backend did not allocate");
    VulkanContext::Get().FreeBuffer(buffer, memory);
  }

  // Host-coherent, persistently mapped storage + synchronous dispatch => the
  // host may touch the bytes directly. Both are BIT-EXACT byte operations, which
  // is why the gate keeps bit-exactness for pure copy/layout paths while
  // reductions only owe NMSE.
  void Memset(Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }

  // One process-wide VkQueue is shared by every vt::Queue: the queue handle is
  // the ORDERING domain and, with synchronous dispatch, every op is already
  // ordered. `id` still makes each vt::Queue a distinct identity for the
  // workspace-key machinery (src/vt/ops.cpp MakeWorkspaceKey).
  Queue CreateQueue() override {
    return Queue{Device{DeviceType::kVULKAN, 0}, VulkanContext::Get().queue_handle()};
  }

  bool UnifiedMemory() const override { return VulkanContext::Get().unified_memory(); }

  int DeviceCapabilityMajor() const override { return VulkanContext::Get().api_major(); }
  int DeviceCapabilityMinor() const override { return VulkanContext::Get().api_minor(); }
};

struct Registrar {
  Registrar() {
    // A Vulkan-ENABLED build may still run where there is no loader or no
    // conformant device (a headless CI container). Probing first keeps that a
    // clean "kVULKAN not registered" — the state src/vt/ops.cpp:104-111 already
    // treats as supported — instead of a throw during static initialization,
    // which would abort the process.
    if (!VulkanContext::Available()) return;
    static VulkanBackend backend;
    RegisterBackend(DeviceType::kVULKAN, &backend);
  }
} registrar;

}  // namespace

void RegisterAllocation(void* base, size_t bytes, void* buffer, void* memory) {
  std::lock_guard<std::mutex> g(AllocMutex());
  AllocMap()[reinterpret_cast<uintptr_t>(base)] = AllocEntry{bytes, buffer, memory};
}

bool UnregisterAllocation(void* base, void** out_buffer, void** out_memory) {
  std::lock_guard<std::mutex> g(AllocMutex());
  auto& m = AllocMap();
  auto it = m.find(reinterpret_cast<uintptr_t>(base));
  if (it == m.end()) return false;
  *out_buffer = it->second.buffer;
  *out_memory = it->second.memory;
  m.erase(it);
  return true;
}

Resolved Resolve(const void* ptr, const char* what) {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> g(AllocMutex());
  auto& m = AllocMap();
  // upper_bound gives the first allocation starting AFTER addr; the candidate
  // container is its predecessor. Same interior-pointer walk as llama.cpp's
  // ggml_vk_host_get (ggml-vulkan.cpp:7416).
  auto it = m.upper_bound(addr);
  if (it != m.begin()) {
    --it;
    if (addr >= it->first && addr < it->first + it->second.bytes) {
      return Resolved{it->second.buffer, static_cast<uint32_t>(addr - it->first)};
    }
  }
  VT_CHECK(false, std::string("vulkan: ") + what +
                      " points outside every Vulkan allocation — Vulkan kernels can only "
                      "bind memory obtained from vt::GetBackend(DeviceType::kVULKAN).Alloc()");
  return Resolved{};
}

}  // namespace vt::vulkan
