// vllm.cpp original (vt runtime, inventory deviation §9.1) — implementation of
// the CUDA device-capability cache (cuda_device_caps.h, seam-gap #4) and the
// runtime SM-dispatch tactic registry (cuda_arch_tactics.h, seam-gap #2) from
// .agents/specs/cuda-arch-additivity.md.
//
// Both live in ONE translation unit on purpose: the registry's only input is the
// capability, and keeping them together means a new architecture never has to
// touch this file — it adds its own TU and calls RegisterArchTactic().
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "vt/cuda/cuda_arch_tactics.h"
#include "vt/cuda/cuda_device_caps.h"

namespace vt::cuda {
namespace {

// Enough for any single-node CUDA box; a higher ordinal falls back to an
// on-the-spot probe rather than growing a table under a lock on the hot path.
constexpr int kMaxCachedDevices = 16;

DeviceCaps ProbeDeviceCaps(int device) {
  DeviceCaps caps;
  caps.device = device;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count) {
    return caps;  // valid stays false — silent, matching the registrar contract
  }
  auto attr = [&](cudaDeviceAttr which, int* out) {
    return cudaDeviceGetAttribute(out, which, device) == cudaSuccess;
  };
  int pageable = 0;
  int integrated = 0;
  if (!attr(cudaDevAttrComputeCapabilityMajor, &caps.sm_major)) return DeviceCaps{};
  if (!attr(cudaDevAttrComputeCapabilityMinor, &caps.sm_minor)) return DeviceCaps{};
  if (!attr(cudaDevAttrMaxSharedMemoryPerBlockOptin,
            &caps.max_shared_memory_per_block_optin)) {
    return DeviceCaps{};
  }
  if (!attr(cudaDevAttrMultiProcessorCount, &caps.multiprocessor_count)) return DeviceCaps{};
  if (!attr(cudaDevAttrPageableMemoryAccess, &pageable)) return DeviceCaps{};
  if (!attr(cudaDevAttrIntegrated, &integrated)) return DeviceCaps{};
  caps.device = device;
  caps.pageable_memory_access = pageable != 0;
  caps.integrated = integrated != 0;
  caps.valid = true;
  return caps;
}

struct DeviceCapsCache {
  std::once_flag once[kMaxCachedDevices];
  DeviceCaps caps[kMaxCachedDevices];
};

DeviceCapsCache& Cache() {
  static DeviceCapsCache cache;
  return cache;
}

// ---------------------------------------------------------------------------
// Tactic registry. Fixed-capacity, static-storage tables: registration happens
// during static init from arbitrary TUs, so there must be no dynamic allocation
// and no dependence on another TU's constructor having run.
// ---------------------------------------------------------------------------
constexpr int kMaxTacticsPerFamily = 8;
constexpr int kFamilyCount = static_cast<int>(TacticFamily::kCount);

struct FamilyTable {
  ArchTactic tactics[kMaxTacticsPerFamily];
  int count = 0;
  std::atomic<const char*> last_selected{nullptr};
  std::atomic<unsigned long long> selections{0};
  std::atomic<unsigned long long> fallbacks{0};
  std::atomic<bool> announced{false};
};

FamilyTable& Table(TacticFamily family) {
  // Zero-initialized static storage: usable from any static-init order.
  static FamilyTable tables[kFamilyCount];
  const int idx = static_cast<int>(family);
  return tables[(idx >= 0 && idx < kFamilyCount) ? idx : 0];
}

bool StatsEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ARCH_TACTIC_STATS");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

}  // namespace

const DeviceCaps& GetDeviceCaps(int device) {
  if (device < 0 || device >= kMaxCachedDevices) {
    static thread_local DeviceCaps uncached;
    uncached = ProbeDeviceCaps(device);
    return uncached;
  }
  DeviceCapsCache& cache = Cache();
  std::call_once(cache.once[device],
                 [&, device] { cache.caps[device] = ProbeDeviceCaps(device); });
  return cache.caps[device];
}

const DeviceCaps& GetDeviceCaps() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) device = 0;
  return GetDeviceCaps(device);
}

bool DynamicSmemFits(long long bytes) {
  const DeviceCaps& caps = GetDeviceCaps();
  // An unprobed device cannot promise anything beyond the guaranteed 48 KiB
  // every CUDA architecture provides without an opt-in.
  const long long ceiling =
      caps.valid ? static_cast<long long>(caps.max_shared_memory_per_block_optin)
                 : 48LL * 1024LL;
  return bytes <= ceiling;
}

void RegisterArchTactic(TacticFamily family, const ArchTactic& tactic) {
  FamilyTable& table = Table(family);
  if (table.count >= kMaxTacticsPerFamily) return;  // table fill only; never throws
  if (tactic.supports == nullptr || tactic.launch == nullptr) return;
  table.tactics[table.count++] = tactic;
}

int RegisteredArchTacticCount(TacticFamily family) { return Table(family).count; }

const char* ArchTacticFamilyName(TacticFamily family) {
  switch (family) {
    case TacticFamily::kNvfp4Fp4Mma: return "nvfp4-fp4-mma";
    case TacticFamily::kFp8Cutlass: return "fp8-cutlass";
    case TacticFamily::kFlashAttn: return "flash-attn";
    case TacticFamily::kCount: break;
  }
  return "unknown";
}

const ArchTactic* SelectArchTactic(TacticFamily family, const DeviceCaps& caps) {
  FamilyTable& table = Table(family);
  for (int i = 0; i < table.count; ++i) {
    const ArchTactic& t = table.tactics[i];
    if (t.supports(caps)) {
      table.last_selected.store(t.name, std::memory_order_relaxed);
      table.selections.fetch_add(1, std::memory_order_relaxed);
      if (StatsEnabled() && !table.announced.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[vt arch-tactic] family=%s selected=%s sm=%d.%d "
                     "smem_optin=%d registered=%d\n",
                     ArchTacticFamilyName(family), t.name, caps.sm_major, caps.sm_minor,
                     caps.max_shared_memory_per_block_optin, table.count);
      }
      return &t;
    }
  }
  table.fallbacks.fetch_add(1, std::memory_order_relaxed);
  if (StatsEnabled() && !table.announced.exchange(true, std::memory_order_relaxed)) {
    std::fprintf(stderr,
                 "[vt arch-tactic] family=%s selected=<none, portable fallback> "
                 "sm=%d.%d registered=%d\n",
                 ArchTacticFamilyName(family), caps.sm_major, caps.sm_minor, table.count);
  }
  return nullptr;
}

ArchTacticStats GetArchTacticStats(TacticFamily family) {
  FamilyTable& table = Table(family);
  ArchTacticStats s;
  s.last_selected = table.last_selected.load(std::memory_order_relaxed);
  s.selections = table.selections.load(std::memory_order_relaxed);
  s.fallbacks = table.fallbacks.load(std::memory_order_relaxed);
  return s;
}

}  // namespace vt::cuda
