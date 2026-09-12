// Test-only observation of successful HIP allocator calls. IDs distinguish
// successive allocations when the driver reuses a device address.
#pragma once
#include <cstdint>
namespace rocm_f16_test {
uint64_t AllocationId(uintptr_t pointer);
bool WasFreed(uint64_t id);
bool FreedOnAllocationDevice(uint64_t id);
}  // namespace rocm_f16_test
