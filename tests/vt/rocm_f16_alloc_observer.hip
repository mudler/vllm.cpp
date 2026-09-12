#include <hip/hip_runtime.h>
#include <map>
#include <mutex>
#include <vector>
#include "rocm_f16_alloc_observer.h"

namespace {
struct Observer {
  std::mutex mutex;
  std::map<uintptr_t, uint64_t> active;
  struct Allocation { bool freed = false; int device = -1; int free_device = -1; };
  std::vector<Allocation> allocations{Allocation{}};
};
Observer& State() {
  static auto* state = new Observer;
  return *state;
}
}  // namespace

extern "C" hipError_t __real_hipMalloc(void**, size_t);
extern "C" hipError_t __real_hipFree(void*);
extern "C" hipError_t __wrap_hipMalloc(void** pointer, size_t bytes) {
  const auto result = __real_hipMalloc(pointer, bytes);
  if (result == hipSuccess && *pointer) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    hipPointerAttribute_t attributes{};
    const auto provenance = hipPointerGetAttributes(&attributes, *pointer);
    const auto id = state.allocations.size();
    state.allocations.push_back({false, provenance == hipSuccess ? attributes.device : -1, -1});
    state.active[reinterpret_cast<uintptr_t>(*pointer)] = id;
  }
  return result;
}
extern "C" hipError_t __wrap_hipFree(void* pointer) {
  int current = -1;
  if (hipGetDevice(&current) != hipSuccess) current = -1;
  const auto result = __real_hipFree(pointer);
  if (result == hipSuccess && pointer) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto found = state.active.find(reinterpret_cast<uintptr_t>(pointer));
    if (found != state.active.end()) {
      state.allocations[found->second].freed = true;
      state.allocations[found->second].free_device = current;
      state.active.erase(found);
    }
  }
  return result;
}
namespace rocm_f16_test {
uint64_t AllocationId(uintptr_t pointer) {
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  const auto found = state.active.find(pointer);
  return found == state.active.end() ? 0 : found->second;
}
bool WasFreed(uint64_t id) {
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  return id > 0 && id < state.allocations.size() && state.allocations[id].freed;
}
bool FreedOnAllocationDevice(uint64_t id) {
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (id == 0 || id >= state.allocations.size()) return false;
  const auto& allocation = state.allocations[id];
  return allocation.freed && allocation.device >= 0 && allocation.device == allocation.free_device;
}
}  // namespace rocm_f16_test
