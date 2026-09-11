// Vulkan backend — instance/device/queue/memory/pipeline scaffolding.
// See vulkan_context.h for the port map (llama.cpp `ggml/src/ggml-vulkan/` @
// 237ad9b96). BACKEND-VULKAN, W0 skeleton.
//
// § RELAXED PRECISION — the knobs that had to be pinned.
// The Metal skeleton found that Metal's DEFAULT fast-math would have silently
// voided its CPU comparison and pinned MTLMathModeSafe. Vulkan/SPIR-V has the
// same class of trap in three places, handled as follows:
//   1. `inversesqrt()` — GLSL only requires ~2 ULP and drivers lower it to the
//      hardware reciprocal-sqrt approximation. llama.cpp uses it in both norm
//      shaders (rms_norm.comp:86, norm.comp:39). We use `1.0 / sqrt(x)`, which
//      is literally what the CPU reference computes. Pinned in the SHADER, so it
//      cannot be undone by a driver flag.
//   2. `RelaxedPrecision` decorations — emitted only for mediump/lowp
//      qualifiers, which none of our shaders use, and glslang applies no
//      fast-math relaxation of its own (there is no -ffast-math equivalent). The
//      committed SPIR-V is therefore IEEE-as-written by construction and is
//      regenerated only through scripts/gen-vulkan-spirv.py.
//   3. FLOAT CONTROLS — denormal flush-to-zero and signed-zero/Inf/NaN
//      preservation for fp32 are IMPLEMENTATION-DEFINED in Vulkan
//      (VkPhysicalDeviceFloatControlsProperties). These are the one knob we
//      cannot pin from the shader without SPV_KHR_float_controls execution
//      modes, so instead they are PROBED, recorded on the context, and asserted
//      by the unit gate (tests/vt/test_vulkan_backend.cpp), which reports what
//      the device actually does rather than assuming. They matter only for
//      denormal inputs and NaN/±0 payloads; the bit-exact tier of the
//      cross-device harness covers exactly those cases through the bf16 codec,
//      which is integer arithmetic and therefore unaffected either way.
#include "vulkan_context.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "vulkan_loader.h"
#include "vulkan_spirv.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vt::vulkan {
namespace {

// DEVICE-MEMORY ACCOUNTING, enabled by VT_VULKAN_ALLOC_STATS (BACKEND-VULKAN-
// LOADMEM).
//
// GB10 is a UNIFIED-memory box: the one Vulkan heap and the machine's RAM are
// the SAME 119 GiB, so a Vulkan allocation and a host allocation compete
// directly and an over-allocating load takes the whole machine down rather than
// failing cleanly (`NV_ERR_NO_MEMORY` out of `_memdescAllocInternal`, twice in
// one day). Attributing that needs three numbers that only this layer can
// supply: how many bytes the CALLER asked for, how many the DRIVER actually
// committed (`VkMemoryRequirements::size`, which is rounded up), and how many
// are LIVE at the moment the process peaks. Everything else -- RSS, page cache,
// MemAvailable -- is observable from outside with /proc.
//
// Cost when off: one relaxed atomic add per allocation, which is noise against
// a vkAllocateMemory. Cost when on: additionally a /proc read, but ONLY on a
// new high-water mark, which is O(heap/step) times over a whole run.
const bool kAllocStats = [] {
  const char* v = std::getenv("VT_VULKAN_ALLOC_STATS");
  return v != nullptr && std::strcmp(v, "0") != 0;
}();

struct AllocAccounting {
  std::atomic<uint64_t> live_count{0};
  std::atomic<uint64_t> total_count{0};
  std::atomic<uint64_t> live_requested{0};  // caller bytes, before rounding
  std::atomic<uint64_t> live_allocated{0};  // VkMemoryRequirements::size
  std::atomic<uint64_t> total_requested{0};
  std::atomic<uint64_t> total_allocated{0};
  std::atomic<uint64_t> peak_allocated{0};
  std::atomic<uint64_t> peak_count{0};
  std::atomic<uint64_t> next_report{0};  // next high-water print threshold
};

AllocAccounting& Accounting() {
  static AllocAccounting a;
  return a;
}

// Per-allocation sizes, so FreeBuffer can subtract exactly what AllocBuffer
// added. Keyed by the packed VkDeviceMemory, which is unique while it is live.
// A map plus a mutex is free at this frequency: every entry costs one
// vkAllocateMemory or vkFreeMemory, which is orders of magnitude dearer.
std::mutex& AllocSizeMutex() {
  static std::mutex m;
  return m;
}
std::map<void*, std::pair<uint64_t, uint64_t>>& AllocSizes() {  // {requested, allocated}
  static std::map<void*, std::pair<uint64_t, uint64_t>> m;
  return m;
}

// One /proc key, in KiB as the kernel reports it, or 0 when absent. Read on the
// slow path only.
uint64_t ProcKiB(const char* path, const char* key) {
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr) return 0;
  char line[256];
  const size_t klen = std::strlen(key);
  uint64_t out = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, key, klen) == 0 && line[klen] == ':') {
      out = std::strtoull(line + klen + 1, nullptr, 10);
      break;
    }
  }
  std::fclose(f);
  return out;
}

constexpr double kToGiB = 1.0 / (1024.0 * 1024.0 * 1024.0);

// Prints the full picture at one instant: what Vulkan holds, what the process
// holds, and what the machine has left. The three together are the attribution
// -- a Vulkan-only number cannot tell a driver over-allocation apart from a host
// mirror of the same weights, and that distinction is the whole question.
void ReportAllocState(const char* why) {
  const AllocAccounting& a = Accounting();
  std::fprintf(
      stderr,
      "[vt vulkan] alloc %-9s live=%llu bufs req=%.3f GiB alloc=%.3f GiB "
      "peak=%.3f GiB | VmRSS=%.3f GiB VmHWM=%.3f GiB | MemAvail=%.3f GiB "
      "Cached=%.3f GiB\n",
      why, static_cast<unsigned long long>(a.live_count.load(std::memory_order_relaxed)),
      static_cast<double>(a.live_requested.load(std::memory_order_relaxed)) * kToGiB,
      static_cast<double>(a.live_allocated.load(std::memory_order_relaxed)) * kToGiB,
      static_cast<double>(a.peak_allocated.load(std::memory_order_relaxed)) * kToGiB,
      static_cast<double>(ProcKiB("/proc/self/status", "VmRSS")) * 1024.0 * kToGiB,
      static_cast<double>(ProcKiB("/proc/self/status", "VmHWM")) * 1024.0 * kToGiB,
      static_cast<double>(ProcKiB("/proc/meminfo", "MemAvailable")) * 1024.0 * kToGiB,
      static_cast<double>(ProcKiB("/proc/meminfo", "Cached")) * 1024.0 * kToGiB);
}

// Dispatch accounting, enabled by VT_VULKAN_DISPATCH_STATS (VK-E deep dive).
const bool kDispatchStats = [] {
  const char* v = std::getenv("VT_VULKAN_DISPATCH_STATS");
  return v != nullptr && std::strcmp(v, "0") != 0;
}();

// HOST-SIDE PHASE PROFILE, enabled by VT_VULKAN_HOST_PROFILE (BACKEND-VULKAN-
// HOSTDISPATCH).
//
// The campaign's "host = wall - GPU" subtraction is an INFERENCE across two
// different runs: a wall-clock run with no query pool, minus the sum of the
// per-shader GPU timestamps from a separate stats run. It cannot distinguish
// host recording cost from a GPU bubble between dispatches, and it cannot say
// WHICH host line costs the microseconds. This measures our own side of the
// boundary directly, attributing every nanosecond spent inside Dispatch to one
// of five phases, and separates the submit and the fence wait inside the flush.
//
// Cost when on: 6 clock_gettime calls per dispatch (vDSO, ~20 ns each) against a
// ~5.9 us dispatch, so ~2%. Cost when off: one predictable branch on a cached
// const bool.
const bool kHostProfile = [] {
  const char* v = std::getenv("VT_VULKAN_HOST_PROFILE");
  return v != nullptr && std::strcmp(v, "0") != 0;
}();

// FENCE STALL GUARD (BACKEND-VULKAN-FENCE-TIMEOUT). By default a dispatch that
// never completes blocks in vkWaitForFences(.., UINT64_MAX) forever, holding the
// host at ~100% CPU while a wedged GPU burns a core. VT_VK_FENCE_TIMEOUT_MS sets
// a ceiling (ms); when it fires the process ABORTS with a loud message naming the last
// dispatch instead of spinning until someone power-cycles the box. 0 = block forever
// (stock behaviour, kept for correctness-identical runs).
const int64_t kFenceTimeoutMs = [] {
  const char* v = std::getenv("VT_VK_FENCE_TIMEOUT_MS");
  if (v == nullptr) return int64_t(0);
  char* end = nullptr;
  int64_t ms = std::strtoll(v, &end, 10);
  return (end != v && ms > 0) ? ms : int64_t(0);
}();

// Accumulators for the above. Every one of these is touched only under the
// dispatch mutex, which Dispatch and FlushBatchLocked's caller already hold.
struct HostProfile {
  uint64_t dispatches = 0;
  uint64_t flushes = 0;
  // Dispatch phases, nanoseconds.
  uint64_t ns_pipeline = 0;   // GetPipeline: key build + std::map lookup
  uint64_t ns_descriptor = 0; // the write array + vkUpdateDescriptorSets
  uint64_t ns_record = 0;     // begin / barrier / bind / push / vkCmdDispatch
  uint64_t ns_bookkeep = 0;   // histogram, batch_buffers_ set, counters
  uint64_t ns_total = 0;      // whole Dispatch body, flush time EXCLUDED
  // Flush phases, nanoseconds.
  uint64_t ns_flush_submit = 0;  // End + ResetFences + QueueSubmit
  uint64_t ns_flush_wait = 0;    // vkWaitForFences
  uint64_t ns_flush_other = 0;   // query readback + ring reset + set clear
};
HostProfile g_host_profile;

inline uint64_t NowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Registered on first use rather than unconditionally, so a run with profiling
// off installs no handler at all.
void DumpHostProfileAtExit() {
  static bool registered = false;
  if (registered) return;
  registered = true;
  std::atexit([] {
    const HostProfile& p = g_host_profile;
    const double d = p.dispatches ? double(p.dispatches) : 1.0;
    std::fprintf(stderr,
                 "[vt vulkan] HOST PROFILE  dispatches=%llu flushes=%llu\n"
                 "[vt vulkan]   phase              total_ms     ns/dispatch\n"
                 "[vt vulkan]   pipeline-lookup  %10.2f  %12.1f\n"
                 "[vt vulkan]   descriptor       %10.2f  %12.1f\n"
                 "[vt vulkan]   record           %10.2f  %12.1f\n"
                 "[vt vulkan]   bookkeeping      %10.2f  %12.1f\n"
                 "[vt vulkan]   DISPATCH TOTAL   %10.2f  %12.1f\n"
                 "[vt vulkan]   flush-submit     %10.2f  %12.1f\n"
                 "[vt vulkan]   flush-wait       %10.2f  %12.1f\n"
                 "[vt vulkan]   flush-other      %10.2f  %12.1f\n",
                 static_cast<unsigned long long>(p.dispatches),
                 static_cast<unsigned long long>(p.flushes),
                 p.ns_pipeline / 1.0e6, p.ns_pipeline / d,
                 p.ns_descriptor / 1.0e6, p.ns_descriptor / d,
                 p.ns_record / 1.0e6, p.ns_record / d,
                 p.ns_bookkeep / 1.0e6, p.ns_bookkeep / d,
                 p.ns_total / 1.0e6, p.ns_total / d,
                 p.ns_flush_submit / 1.0e6, p.ns_flush_submit / d,
                 p.ns_flush_wait / 1.0e6, p.ns_flush_wait / d,
                 p.ns_flush_other / 1.0e6, p.ns_flush_other / d);
  });
}
// COMMAND-BUFFER BATCHING (VK-A2), VT_VULKAN_BATCH.
//
// DEFAULT ON. `=0` forces the per-dispatch submit-and-wait path, as a bisect
// lever and for the same-binary A/B that measured this (decode 2.62x, 8 of 8
// interleaved pairs on GB10).
//
// Batching is sound only if EVERY host read of device memory drains the batch
// first, and there are exactly three such paths. Backend::Copy and Memset are
// host memcpy over the persistently mapped allocation, and Synchronize is the
// caller's explicit "make results readable" point -- all three flush. The third
// is not a method on this backend at all: the PORTABLE REFERENCE TIER runs CPU
// kernels DIRECTLY over device memory for any op with no native Vulkan kernel,
// which is sound only because this backend is unified-memory.
//
// That last one is covered by `Backend::FlushPending` (backend.h:44-49), which
// op_provider.cpp calls before dispatching a reference-tier kernel and which the
// Vulkan backend implements. Without it a host kernel would read bytes a pending
// dispatch had not written -- silently, with no error, which is the failure mode
// this campaign has already paid for twice.
const bool kBatchDispatch = [] {
  const char* v = std::getenv("VT_VULKAN_BATCH");
  return v == nullptr || std::strcmp(v, "0") != 0;
}();

// Cap on dispatches per submit. Bounded so a batch cannot pin an unbounded number
// of descriptor sets, and so the fence granularity stays coarse enough to be
// worth batching but fine enough to bound latency.
constexpr uint32_t kMaxBatch = 512;

const std::chrono::steady_clock::time_point g_dispatch_t0 =
    std::chrono::steady_clock::now();

// The void* handle smuggling in vulkan_context.h is only sound while every
// Vulkan handle fits in a pointer. Dispatchable handles are pointers by
// definition; non-dispatchable ones are uint64_t on a 64-bit build.
static_assert(sizeof(VkInstance) <= sizeof(void*), "VkInstance must fit in void*");
static_assert(sizeof(VkBuffer) <= sizeof(void*), "VkBuffer must fit in void*");
static_assert(sizeof(VkDeviceMemory) <= sizeof(void*), "VkDeviceMemory must fit in void*");

template <typename H>
void* Pack(H h) {
  void* p = nullptr;
  std::memcpy(&p, &h, sizeof(H));
  return p;
}
template <typename H>
H Unpack(void* p) {
  H h{};
  std::memcpy(&h, &p, sizeof(H));
  return h;
}

void Check(VkResult r, const char* what) {
  VT_CHECK(r == VK_SUCCESS,
           std::string("vulkan: ") + what + " failed with VkResult " + std::to_string(r));
}

// llama.cpp `ggml_vk_find_memory_properties` (ggml-vulkan.cpp:2957): walk the
// memory types the buffer's requirements allow and take the first that carries
// every required property flag. We keep its ORDERED FALLBACK shape
// (`ggml_vk_create_buffer`, :3065-3090) — first choice DEVICE_LOCAL as well as
// host-visible/coherent (the unified case: GB10 exposes exactly such a type on
// its single 89.72 GiB heap, and so does llvmpipe), falling back to plain
// host-visible/coherent on a discrete GPU.
constexpr VkMemoryPropertyFlags kHostFlags =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

int FindMemoryType(const VkPhysicalDeviceMemoryProperties& props, uint32_t type_bits,
                   VkMemoryPropertyFlags required) {
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) == 0) continue;
    if ((props.memoryTypes[i].propertyFlags & required) == required) return static_cast<int>(i);
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Device selection. Runs during Available(), i.e. possibly on a machine with no
// Vulkan at all, so it must never throw and never leave an instance behind.
// ---------------------------------------------------------------------------
struct Probe {
  bool ok = false;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  uint32_t api_version = 0;
  bool shader_float64 = false;
  bool integer_dot_product_4x8 = false;
  char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};
};

VkInstance CreateInstance() {
  const VulkanApi& vk = Api();
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "vllm.cpp";
  // Ask for 1.1: this backend needs VK_KHR_16bit_storage, which is CORE in 1.1
  // (see the shaders' § STORAGE MODEL). A 1.0-only loader has no
  // vkEnumerateInstanceVersion and would reject this, which is the answer we
  // want — the backend cannot run there.
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;
  VkInstance instance = VK_NULL_HANDLE;
  if (vk.vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) return VK_NULL_HANDLE;
  return instance;
}


bool HasShaderFloat64(VkPhysicalDevice pd) {
  const VulkanApi& vk = Api();
  VkPhysicalDeviceFeatures2 f2{};
  f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  vk.vkGetPhysicalDeviceFeatures2(pd, &f2);
  return f2.features.shaderFloat64 == VK_TRUE;
}

// Probe shaderIntegerDotProduct (VK_KHR_shader_integer_dot_product). The 4x8
// packed signed variant is what the TQ keep-quant shaders use for
// dotPacked4x8EXT. Use the extension-specific struct rather than the 1.2
// core struct so this compiles on older Vulkan headers.
bool HasIntegerDotProduct4x8(VkPhysicalDevice pd) {
  const VulkanApi& vk = Api();
  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR idot{};
  idot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
  VkPhysicalDeviceFeatures2 f2{};
  f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  f2.pNext = &idot;
  vk.vkGetPhysicalDeviceFeatures2(pd, &f2);
  return idot.shaderIntegerDotProduct == VK_TRUE;
}

bool HasStorageBuffer16BitAccess(VkPhysicalDevice pd) {
  const VulkanApi& vk = Api();
  VkPhysicalDevice16BitStorageFeatures f16{};
  f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  VkPhysicalDeviceFeatures2 f2{};
  f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  f2.pNext = &f16;
  vk.vkGetPhysicalDeviceFeatures2(pd, &f2);
  return f16.storageBuffer16BitAccess == VK_TRUE;
}

bool HasDeviceExtension(VkPhysicalDevice pd, const char* want) {
  const VulkanApi& vk = Api();
  uint32_t count = 0;
  if (vk.vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkExtensionProperties> exts(count);
  if (vk.vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, exts.data()) != VK_SUCCESS) {
    return false;
  }
  for (const auto& e : exts) {
    if (std::strcmp(e.extensionName, want) == 0) return true;
  }
  return false;
}

// The ONE configuration the committed coopmat SPIR-V is written to:
// 16x16x16, A/B bf16, C/Result f32, SUBGROUP scope. Vulkan requires an EXACT
// match against a reported configuration -- there is no "nearest" -- so this
// asks for exactly that tuple and nothing else.
bool HasBf16F32CoopMatConfig(VkPhysicalDevice pd) {
  const VulkanApi& vk = Api();
  if (vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR == nullptr) return false;
  uint32_t count = 0;
  if (vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &count, nullptr) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkCooperativeMatrixPropertiesKHR> cfg(count);
  for (auto& c : cfg) c.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  if (vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &count, cfg.data()) != VK_SUCCESS) {
    return false;
  }
  for (const auto& c : cfg) {
    if (c.MSize == 16 && c.NSize == 16 && c.KSize == 16 &&
        c.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
        c.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
        c.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        c.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        c.scope == VK_SCOPE_SUBGROUP_KHR) {
      return true;
    }
  }
  return false;
}

int FindComputeQueueFamily(VkPhysicalDevice pd) {
  const VulkanApi& vk = Api();
  uint32_t count = 0;
  vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());
  for (uint32_t i = 0; i < count; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && families[i].queueCount > 0) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Picks the first physical device that satisfies every requirement. Preference
// order mirrors llama.cpp's device selection intent (`ggml_vk_instance_init`):
// a real GPU before a software rasterizer, so a box that has BOTH — like the dev
// box, which enumerates llvmpipe — still runs on the GPU when there is one.
// VK_VT_DEVICE lets a caller force a specific index, which is how the llvmpipe
// CI path is exercised on a machine that also has a GPU.
Probe ProbeDevice(VkInstance instance) {
  const VulkanApi& vk = Api();
  Probe best;
  uint32_t count = 0;
  if (vk.vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) {
    return best;
  }
  std::vector<VkPhysicalDevice> devices(count);
  if (vk.vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return best;

  const char* forced = std::getenv("VT_VULKAN_DEVICE");
  int forced_index = forced != nullptr ? std::atoi(forced) : -1;

  int best_rank = -1;
  for (uint32_t i = 0; i < count; ++i) {
    if (forced_index >= 0 && static_cast<int>(i) != forced_index) continue;
    VkPhysicalDeviceProperties props{};
    vk.vkGetPhysicalDeviceProperties(devices[i], &props);
    if (VK_API_VERSION_MAJOR(props.apiVersion) < 1) continue;
    if (VK_API_VERSION_MAJOR(props.apiVersion) == 1 && VK_API_VERSION_MINOR(props.apiVersion) < 1) {
      continue;  // needs 1.1 for VK_KHR_16bit_storage in core
    }
    if (!HasStorageBuffer16BitAccess(devices[i])) continue;
    const int qf = FindComputeQueueFamily(devices[i]);
    if (qf < 0) continue;

    VkPhysicalDeviceMemoryProperties mem{};
    vk.vkGetPhysicalDeviceMemoryProperties(devices[i], &mem);
    if (FindMemoryType(mem, ~0u, kHostFlags) < 0) continue;

    // Rank: integrated/discrete GPU (2) > virtual GPU (1) > CPU/other (0).
    int rank = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      rank = 2;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
      rank = 1;
    }
    if (rank <= best_rank) continue;
    best_rank = rank;
    best.ok = true;
    best.shader_float64 = HasShaderFloat64(devices[i]);
    best.integer_dot_product_4x8 = HasIntegerDotProduct4x8(devices[i]);
    best.physical_device = devices[i];
    best.queue_family = static_cast<uint32_t>(qf);
    best.api_version = props.apiVersion;
    std::memcpy(best.name, props.deviceName, sizeof(best.name));
  }
  return best;
}

}  // namespace

// ---------------------------------------------------------------------------

// How many descriptor sets each pipeline owns (VK-A2).
//
// A DESCRIPTOR SET IS READ AT EXECUTION TIME, NOT RECORD TIME. With one set per
// pipeline -- the pre-VK-A2 shape -- recording two dispatches of the same
// pipeline into one command buffer would have the second vkUpdateDescriptorSets
// overwrite the first's operands before the GPU ran either. That was sound only
// because every dispatch waited on a fence before the next one touched the set.
// Batching therefore needs a set PER RECORDED DISPATCH, which is the substantive
// part of this row; deferring the submit alone would silently compute garbage.
//
// The ring is bounded, so a pipeline used more than kDescriptorRing times in one
// batch forces a flush. That is a throughput ceiling, never a correctness
// question: the flush happens before the set is reused.
// MEASURED: 16 was the batch-length limiter, not kMaxBatch. vt_rms_norm runs 112
// times per forward pass (4 per layer x 28 layers), so it exhausted a 16-deep ring
// seven times per pass and forced a flush each time -- observed batches capped at
// 40-46 dispatches against a kMaxBatch of 128. Every forced flush is a
// vkQueueSubmit plus a blocking vkWaitForFences, and a host profile with the idle
// CPU-threadpool spin suppressed puts 62% of on-CPU time in the kernel and the
// NVIDIA driver against only 14% in our own code. So the submits ARE the host
// cost, and the ring depth is what sets how many there are.
// RAISED FROM 128 TO 256 BY BACKEND-VULKAN-HOSTDISPATCH. The ring is now
// PARTITIONED across in-flight slots, so the depth one batch can use is
// kDescriptorRing / kInFlight. Leaving it at 128 would have halved the effective
// depth to 64 the moment pipelining was enabled and doubled the flush count per
// token, trading the win back. 256 keeps a 2-deep pipeline at the same 128 sets
// per batch that the un-pipelined ring had. VT_VULKAN_RING A/Bs the two in ONE
// binary.
constexpr uint32_t kDescriptorRing = 256;

// EFFECTIVE ring depth, clamped to the allocated one. Exists so the ring can be
// A/B'd in ONE binary: it was 16, which capped batches at 40-46 dispatches, and a
// cross-BUILD comparison of two depths is the shape that produced a false 1.2x
// reading for the subgroup tactic earlier in this campaign.
const uint32_t kRingDepth = [] {
  const char* v = std::getenv("VT_VULKAN_RING");
  if (v == nullptr) return kDescriptorRing;
  const int n = std::atoi(v);
  if (n < 1) return 1u;
  return n > static_cast<int>(kDescriptorRing) ? kDescriptorRing : static_cast<uint32_t>(n);
}();

// HOW MANY BATCHES MAY BE IN FLIGHT AT ONCE (BACKEND-VULKAN-HOSTDISPATCH).
//
// 1 restores the pre-row submit-and-wait behaviour EXACTLY, which is what makes
// this a same-binary A/B rather than the cross-build comparison that has already
// produced one false reading in this campaign.
//
// The default is 2: the host only needs ONE batch of look-ahead to overlap its
// recording with the GPU, because a batch is ~57 ms of GPU work against ~0.4 ms
// of host recording. Deeper queues buy nothing and cost descriptor-ring width.
const uint32_t kInFlight = [] {
  const char* v = std::getenv("VT_VULKAN_INFLIGHT");
  uint32_t n = 2;
  if (v != nullptr) {
    const int parsed = std::atoi(v);
    n = parsed < 1 ? 1u : static_cast<uint32_t>(parsed);
  }
  return n > VulkanContext::kMaxInFlight ? VulkanContext::kMaxInFlight : n;
}();

// Upper bound on the descriptor writes ONE dispatch can issue, and therefore the
// size of the stack arrays the dispatch path builds them in. It replaces a pair
// of heap `std::vector`s that were constructed and destroyed on every dispatch.
//
// This is a real bound, not llama.cpp's MAX_PARAMETER_COUNT of 12
// (ggml-vulkan.cpp:6424-6437): the Binder in vulkan_ops.cpp binds a u32 AND a u16
// VIEW of most operands, so the widest kernel here (the fused GDN post-conv
// preamble, 8 operands) already declares 16 bindings. 32 is headroom over that,
// checked at dispatch so an overrun names itself instead of smashing the stack.
constexpr uint32_t kMaxDispatchBindings = 32;

// SMART BARRIERS (BACKEND-VULKAN-BARRIERS), VT_VULKAN_SMART_BARRIERS.
//
// The batched dispatch path records a full COMPUTE->COMPUTE memory barrier before
// EVERY dispatch -- about 900 per 27B decode token -- because the ops in a decode
// step are sequentially dependent and a dispatch that misses its producer's writes
// computes garbage. llama.cpp does not: it calls `ggml_vk_sync_buffers`
// (ggml-vulkan.cpp:3193 @ pin 237ad9b96) at 42 explicit call sites and tracks
// per-scratch-buffer `prealloc_{x,y,split_k}_need_sync` flags (:2108, :8174,
// :8687-8748) so a sync happens only where a real dependency exists.
//
// This is the same idea made GENERAL rather than per-scratch-buffer: the buffers
// each dispatch reads and writes are known exactly (the committed SPIR-V records
// which bindings are NonWritable, see vulkan_spirv.h), so the barrier can be
// driven by an actual hazard test instead of by a call site the author remembered.
//
// DEFAULT OFF. The failure mode of a wrong answer here is not a crash, it is
// silently wrong numbers, and this backend has already shipped one measured-faster
// arm that computed garbage on the real driver while passing on llvmpipe (see the
// FENCE SPIN note above). The default moves only on GB10 evidence.
const bool kSmartBarriersEnv = [] {
  const char* v = std::getenv("VT_VULKAN_SMART_BARRIERS");
  return v != nullptr && std::strcmp(v, "0") != 0;
}();

// Descriptor-ring sets available to ONE slot. The ring is partitioned rather
// than shared, so slot s owns [s*kRingSlice, (s+1)*kRingSlice) and no set can be
// rewritten by a batch while an earlier, still-executing batch reads it.
const uint32_t kRingSlice = [] {
  const uint32_t slice = kRingDepth / kInFlight;
  return slice < 1 ? 1u : slice;
}();

// A FENCE SPIN WAS TRIED AND IS REJECTED. RECORDED HERE BECAUSE THE NEGATIVE IS
// THE EXPENSIVE PART.
//
// llama.cpp spins on `vkGetFenceStatus` instead of blocking, so the host resumes
// the instant the GPU signals rather than when the scheduler gets round to it
// (`ggml_vk_wait_for_fence`, ggml-vulkan.cpp:2306-2326 @ 237ad9b96). Ported here
// it was worth a MEASURED 2.6 ms/token on 27B decode -- and it COMPUTED THE WRONG
// NUMBERS on GB10.
//
// Same binary, same tree, one variable, on the GB10 device:
//
//   VT_VULKAN_FENCE_SPIN=0   test_vulkan_backend 33/33, opt-125m 6/6 token-exact
//   VT_VULKAN_FENCE_SPIN=1   test_vulkan_backend 16/33, opt-125m DIVERGES
//
// It failed identically with pipelining ON and OFF, which is what identifies the
// spin rather than the pipelining as the cause; the failures are host reads of
// device memory returning PRE-DISPATCH contents, so observing the fence signalled
// through `vkGetFenceStatus` did not make the device writes visible to the host on
// this driver the way returning from `vkWaitForFences` does. It passed on
// llvmpipe, which is why only running it on the real device found it.
//
// The retire path below therefore always blocks in `vkWaitForFences`. The
// `vkGetFenceStatus` call it makes first is a pure diagnostic -- it distinguishes
// a retirement that had to block from one that did not -- and is never the
// synchronisation.

struct VulkanContext::Pipeline {
  VkShaderModule module = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorSet sets[kDescriptorRing] = {};
  uint32_t used_this_batch = 0;  // reset by FlushBatchLocked
  uint32_t buffer_count = 0;
  uint32_t push_size = 0;
  // Dispatch accounting moved OFF the hot path. It used to be
  // `++map<string,uint64_t>[name]` on every dispatch: a red-black descent with
  // string comparisons, 900 times per decode token, for a counter that is only
  // ever read by a diagnostic. The pipeline is already in hand at that point, so
  // the counter lives here and DispatchHistogram aggregates on demand.
  uint64_t dispatches = 0;
  const char* module_name = nullptr;  // points into the committed SPIR-V table
  // Bit i set iff binding i is one the SHADER MAY WRITE, copied from the
  // committed SPIR-V table (vulkan_spirv.h § writable_mask). The dispatch path
  // splits its buffer array into a read set and a write set with it.
  uint32_t writable_mask = 0;
};

// The set of buffers bound by any batch that has not yet been drained.
//
// Was `std::set<void*>`, i.e. a red-black node allocation per newly seen buffer
// and a pointer-comparison descent per bound buffer on every dispatch -- about
// 3,600 lookups per decode token. This is an open-addressed table with linear
// probing over a power-of-two array, which is a load and a compare in the common
// case and stops allocating once it has grown to the working set.
//
// It GROWS rather than carrying a fixed capacity. A batch may legitimately bind
// any number of distinct buffers -- a 27B decode token names several hundred
// weight allocations -- so a fixed table would be a model-size-dependent hard
// failure, which is a worse trade than a handful of reallocations during warmup.
class BufferSet {
 public:
  BufferSet() : slots_(1024, nullptr) {}

  void Insert(void* p) {
    // Load factor kept under 1/2, which is what bounds the probe chains and
    // guarantees the scan loops below terminate.
    if ((count_ + 1) * 2 > slots_.size()) Grow();
    const size_t at = Probe(p);
    if (slots_[at] != nullptr) return;  // already present
    slots_[at] = p;
    ++count_;
  }

  bool Contains(void* p) const { return slots_[Probe(p)] != nullptr; }

  void Clear() {
    if (count_ == 0) return;
    std::fill(slots_.begin(), slots_.end(), nullptr);
    count_ = 0;
  }

 private:
  static size_t Hash(void* p) {
    // Fibonacci hashing on the pointer, shifted past the allocator's alignment
    // bits so distinct buffers do not all collide into one probe chain.
    uint64_t x = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)) >> 4;
    x *= 0x9E3779B97F4A7C15ull;
    return static_cast<size_t>(x >> 32);
  }

  // Index of `p` if present, else the index of the free slot it belongs in.
  size_t Probe(void* p) const {
    const size_t mask = slots_.size() - 1;
    size_t i = Hash(p) & mask;
    while (slots_[i] != nullptr && slots_[i] != p) i = (i + 1) & mask;
    return i;
  }

  void Grow() {
    std::vector<void*> old;
    old.swap(slots_);
    slots_.assign(old.size() * 2, nullptr);
    count_ = 0;
    for (void* q : old) {
      if (q == nullptr) continue;
      slots_[Probe(q)] = q;
      ++count_;
    }
  }

  std::vector<void*> slots_;
  size_t count_ = 0;
};

bool VulkanContext::Available() {
  // Cached: probing creates and destroys an instance, and both registrars plus
  // the platform TU ask.
  static const bool available = [] {
    if (!LoadVulkanLibrary()) return false;
    const VulkanApi& vk = Api();
    // A 1.0-only loader cannot give us the 1.1 core features this backend needs.
    if (vk.vkEnumerateInstanceVersion == nullptr) return false;
    uint32_t loader_version = 0;
    if (vk.vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS) return false;
    if (VK_API_VERSION_MAJOR(loader_version) == 1 &&
        VK_API_VERSION_MINOR(loader_version) < 1) {
      return false;
    }
    VkInstance instance = CreateInstance();
    if (instance == VK_NULL_HANDLE) return false;
    LoadInstanceFunctions(instance);
    const Probe probe = ProbeDevice(instance);
    vk.vkDestroyInstance(instance, nullptr);
    return probe.ok;
  }();
  return available;
}

bool VulkanDeviceAvailable() { return VulkanContext::Available(); }

uint32_t FlatGroupCount(int64_t n) {
  if (n <= 0) return 0;
  return static_cast<uint32_t>((n + kWorkgroupSize - 1) / kWorkgroupSize);
}

VulkanContext::VulkanContext() {
  VT_CHECK(LoadVulkanLibrary(), "vulkan: no Vulkan loader (libvulkan.so.1) on this machine");
  const VulkanApi& vk = Api();

  VkInstance instance = CreateInstance();
  VT_CHECK(instance != VK_NULL_HANDLE, "vulkan: vkCreateInstance failed");
  LoadInstanceFunctions(instance);
  instance_ = Pack(instance);

  const Probe probe = ProbeDevice(instance);
  VT_CHECK(probe.ok, "vulkan: no physical device meets the backend's requirements "
                     "(Vulkan >= 1.1, a compute queue, storageBuffer16BitAccess, and a "
                     "HOST_VISIBLE|HOST_COHERENT memory type)");
  physical_device_ = Pack(probe.physical_device);
  queue_family_ = probe.queue_family;
  api_major_ = static_cast<int>(VK_API_VERSION_MAJOR(probe.api_version));
  api_minor_ = static_cast<int>(VK_API_VERSION_MINOR(probe.api_version));
  device_name_ = probe.name;
  shader_float64_ = probe.shader_float64;
  integer_dot_product_4x8_ = probe.integer_dot_product_4x8;

  // Float controls — probed and recorded, not pinned; see § RELAXED PRECISION.
  VkPhysicalDeviceFloatControlsProperties fc{};
  fc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &fc;
  vk.vkGetPhysicalDeviceProperties2(probe.physical_device, &props2);
  denorm_preserve_f32_ = fc.shaderDenormPreserveFloat32 == VK_TRUE;
  sz_inf_nan_preserve_f32_ = fc.shaderSignedZeroInfNanPreserveFloat32 == VK_TRUE;
  max_workgroup_count_x_ = props2.properties.limits.maxComputeWorkGroupCount[0];
  max_workgroup_count_y_ = props2.properties.limits.maxComputeWorkGroupCount[1];
  max_workgroup_invocations_ = props2.properties.limits.maxComputeWorkGroupInvocations;
  max_workgroup_size_x_ = props2.properties.limits.maxComputeWorkGroupSize[0];
  // GPU TIMESTAMP SUPPORT, probed rather than assumed. `timestampPeriod` is
  // nanoseconds per tick; a device reporting 0 cannot timestamp at all, and
  // `timestampComputeAndGraphics == VK_FALSE` means the compute queue family may
  // not support it even when the device nominally does. Either way profiling
  // silently stays off rather than producing nonsense numbers.
  if (props2.properties.limits.timestampComputeAndGraphics == VK_TRUE) {
    timestamp_period_ns_ = static_cast<double>(props2.properties.limits.timestampPeriod);
  }
  VT_CHECK(props2.properties.limits.maxComputeWorkGroupInvocations >= kWorkgroupSize,
           "vulkan: device reports maxComputeWorkGroupInvocations below the Vulkan "
           "guaranteed minimum of 128, which the committed SPIR-V is compiled for");

  // storageBuffer16BitAccess must be ENABLED, not merely supported.
  VkPhysicalDevice16BitStorageFeatures f16{};
  f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  f16.storageBuffer16BitAccess = VK_TRUE;

  // --- COOPERATIVE MATRIX (VK-C), enabled only where the device actually has it.
  //
  // Enablement is CONDITIONAL for a load-bearing reason: naming an unsupported
  // extension in VkDeviceCreateInfo makes vkCreateDevice FAIL OUTRIGHT, so an
  // unconditional request would take the whole backend down on llvmpipe -- which
  // is the only Vulkan device CI can reach, and which exposes the extension not
  // at all (measured 2026-08-07). The scalar tactic stays correct there.
  //
  // The predicate is deliberately narrow: extension present, AND a bf16 x bf16
  // -> f32 16x16x16 SUBGROUP configuration reported, AND the subgroup size known.
  // A device with cooperative matrix but only f16 or int8 configurations gets the
  // scalar path, because the committed coopmat SPIR-V is written to the bf16
  // shape and Vulkan gives no way to "almost" match a configuration.
  std::vector<const char*> device_exts;
  const bool has_coopmat_ext = HasDeviceExtension(probe.physical_device,
                                                  "VK_KHR_cooperative_matrix");
  const bool has_bf16_ext = HasDeviceExtension(probe.physical_device,
                                               "VK_KHR_shader_bfloat16");
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_feat{};
  coop_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16_feat{};
  bf16_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;

  // Subgroup size, which the coopmat shader's workgroup shape depends on.
  VkPhysicalDeviceSubgroupProperties sub{};
  sub.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 sprops{};
  sprops.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  sprops.pNext = &sub;
  Api().vkGetPhysicalDeviceProperties2(probe.physical_device, &sprops);
  subgroup_size_ = sub.subgroupSize;

  // --- SUBGROUP REDUCTION / WIDE REDUCING WORKGROUPS (VK-RMSNORM).
  //
  // Both feature bits are asked for IN THE COMPUTE STAGE. `supportedOperations`
  // alone is not enough: a device may expose arithmetic subgroup operations for
  // fragment shaders and not for compute, and a module built on subgroupAdd is
  // then invalid in exactly the stage we use. There is no VkResult that reports
  // this -- an unsupported capability is undefined behaviour at pipeline
  // creation, not an error code -- so the probe is the only guard there is.
  //
  // No extension has to be REQUESTED for either: subgroup basic and arithmetic
  // are core Vulkan 1.1 features, and 1.1 is already this backend's floor.
  subgroup_arithmetic_compute_ =
      (sub.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
      (sub.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0 &&
      (sub.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
  // Shuffle is core Vulkan 1.1 alongside basic and arithmetic, but a device
  // can expose basic+arithmetic without shuffle. Probe it separately because
  // the TQ shaders use subgroupShuffleXor for the per-block amax reduction.
  subgroup_shuffle_compute_ =
      (sub.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
      (sub.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0;
  wide_reduce_ = subgroup_arithmetic_compute_ && subgroup_size_ > 0 &&
                 max_workgroup_invocations_ >= kWideWorkgroupSize &&
                 max_workgroup_size_x_ >= kWideWorkgroupSize;
  if (const char* v = std::getenv("VT_VULKAN_RMSNORM"); v != nullptr) {
    if (std::strcmp(v, "base") == 0) {
      rms_norm_override_ = -1;
    } else if (std::strcmp(v, "wide") == 0) {
      VT_CHECK(wide_reduce_,
               "vulkan: VT_VULKAN_RMSNORM=wide but this device does not support "
               "1024-invocation workgroups with compute subgroup arithmetic");
      rms_norm_override_ = 1;
    }
  }

  if (has_coopmat_ext && has_bf16_ext &&
      HasBf16F32CoopMatConfig(probe.physical_device) && subgroup_size_ > 0) {
    coopmat_bf16_f32_ = true;
    coop_feat.cooperativeMatrix = VK_TRUE;
    bf16_feat.shaderBFloat16Type = VK_TRUE;
    bf16_feat.shaderBFloat16CooperativeMatrix = VK_TRUE;
    device_exts.push_back("VK_KHR_cooperative_matrix");
    device_exts.push_back("VK_KHR_shader_bfloat16");
    // Chain: f16 -> coopmat -> bf16.
    coop_feat.pNext = &bf16_feat;
    f16.pNext = &coop_feat;
  }

  // --- INTEGER DOT PRODUCT (VK-IDOT). Enable VK_KHR_shader_integer_dot_product
  // when the device supports it, so the TQ keep-quant shaders can use
  // dotPacked4x8EXT. The feature is CORE in Vulkan 1.2, but we request 1.1 so
  // the extension must be named explicitly. On a 1.2+ device the extension is
  // already in core and naming it is a no-op.
  if (integer_dot_product_4x8_) {
    device_exts.push_back("VK_KHR_shader_integer_dot_product");
  }

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = queue_family_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pNext = &f16;
  dci.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
  dci.ppEnabledExtensionNames = device_exts.empty() ? nullptr : device_exts.data();
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice device = VK_NULL_HANDLE;
  Check(vk.vkCreateDevice(probe.physical_device, &dci, nullptr, &device), "vkCreateDevice");
  LoadDeviceFunctions(device);
  device_ = Pack(device);

  VkQueue queue = VK_NULL_HANDLE;
  Api().vkGetDeviceQueue(device, queue_family_, 0, &queue);
  queue_ = Pack(queue);

  // Memory type, chosen once for every allocation this backend makes. Ordered
  // fallback, llama.cpp `ggml_vk_create_buffer`:3065-3090 shape.
  VkPhysicalDeviceMemoryProperties mem{};
  Api().vkGetPhysicalDeviceMemoryProperties(probe.physical_device, &mem);
  int type = FindMemoryType(mem, ~0u, kHostFlags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  unified_memory_ = type >= 0;
  if (type < 0) type = FindMemoryType(mem, ~0u, kHostFlags);
  VT_CHECK(type >= 0, "vulkan: no HOST_VISIBLE|HOST_COHERENT memory type");
  memory_type_index_ = static_cast<uint32_t>(type);

  // ONE COMMAND POOL PER IN-FLIGHT SLOT. vkResetCommandPool resets every buffer
  // allocated from the pool, so a single shared pool cannot be reset while any
  // other slot's buffer is still executing. llama.cpp reaches the same conclusion
  // by a different route -- it keeps a vector of pools and cleans them up only
  // after the graph's fence (`ggml_vk_queue_command_pools_cleanup`,
  // ggml-vulkan.cpp:2740).
  for (uint32_t s = 0; s < kMaxInFlight; ++s) {
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = queue_family_;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    Check(Api().vkCreateCommandPool(device, &cpci, nullptr, &command_pool),
          "vkCreateCommandPool");
    slot_pool_[s] = Pack(command_pool);

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = command_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    Check(Api().vkAllocateCommandBuffers(device, &cbai, &cmd), "vkAllocateCommandBuffers");
    slot_cmd_[s] = Pack(cmd);

    VkFenceCreateInfo sfci{};
    sfci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence sfence = VK_NULL_HANDLE;
    Check(Api().vkCreateFence(device, &sfci, nullptr, &sfence), "vkCreateFence");
    slot_fence_[s] = Pack(sfence);
    slot_names_[s] = new std::vector<std::string>();
  }
  // Slot 0's handles double as the single-slot handles the unbatched path uses.
  command_pool_ = slot_pool_[0];
  command_buffer_ = slot_cmd_[0];

  // Descriptor-pool BUDGET, not a per-set bound: this is the total number of
  // storage-buffer descriptors the pool may hand out, and the headroom factor
  // below over-provisions it by more than an order of magnitude against the
  // pipelines a run actually creates. (llama.cpp grows a vector of pools instead;
  // a fixed pool is enough here.)
  constexpr uint32_t kBudgetBindingsPerSet = 12;
  // ONE DESCRIPTOR SET PER PIPELINE, AND PIPELINES NOW OUTNUMBER MODULES.
  // Since VK-A1 a module can be specialized into several pipelines — vt_cast
  // alone reaches one per (src, dst) dtype pair — and each allocates its own set
  // from this pool. Sizing it by module count would exhaust the pool on the Nth
  // specialization and fail in vkAllocateDescriptorSets, far from the cause. The
  // headroom factor is deliberate slack, not a computed bound; GetPipeline
  // reports pool exhaustion with the kernel name if it is ever hit.
  constexpr uint32_t kSpecializationHeadroom = 16;
  const uint32_t kernels =
      static_cast<uint32_t>(kSpirvModuleCount) * kSpecializationHeadroom;
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = kernels * kDescriptorRing * kBudgetBindingsPerSet;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  // kDescriptorRing sets per pipeline now, not one -- see the ring's comment for
  // why batching cannot share a set across recorded dispatches.
  dpci.maxSets = kernels * kDescriptorRing;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_size;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  Check(Api().vkCreateDescriptorPool(device, &dpci, nullptr, &descriptor_pool),
        "vkCreateDescriptorPool");
  descriptor_pool_ = Pack(descriptor_pool);

  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  Check(Api().vkCreateFence(device, &fci, nullptr, &fence), "vkCreateFence");
  fence_ = Pack(fence);

  scratch_mapped_ = AllocBuffer(kScratchBytes, &scratch_buffer_, &scratch_memory_);

  pipelines_ = new std::map<std::string, Pipeline>();
  dispatch_ms_ = new std::map<std::string, double>();
  batch_buffers_ = new BufferSet();
  hazard_written_ = new BufferSet();
  hazard_read_ = new BufferSet();
  // TWO timestamps per dispatch (before and after), for a whole batch, and ONE
  // RANGE PER SLOT: a slot's queries are read back only when that slot retires,
  // so two in-flight batches must not share query indices. Created only under the
  // stats flag: a query pool is cheap, but writing timestamps adds commands to
  // every dispatch and production must not pay for a diagnostic.
  if (kDispatchStats && timestamp_period_ns_ > 0.0) {
    VkQueryPoolCreateInfo qpci{};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = kMaxBatch * 2 * kMaxInFlight;
    VkQueryPool qp = VK_NULL_HANDLE;
    if (Api().vkCreateQueryPool(Unpack<VkDevice>(device_), &qpci, nullptr, &qp) == VK_SUCCESS) {
      query_pool_ = Pack(qp);
    }
  }

  // VT_VULKAN_ALLOC_STATS=1 dumps the device-memory summary at exit, for the
  // same reason the dispatch histogram does: this context is never destroyed.
  if (kAllocStats) {
    std::atexit([] {
      const AllocAccounting& a = Accounting();
      ReportAllocState("exit");
      std::fprintf(stderr,
                   "[vt vulkan] alloc lifetime allocations=%llu requested=%.3f GiB "
                   "committed=%.3f GiB peak_live=%.3f GiB peak_bufs=%llu\n",
                   static_cast<unsigned long long>(a.total_count.load()),
                   static_cast<double>(a.total_requested.load()) * kToGiB,
                   static_cast<double>(a.total_allocated.load()) * kToGiB,
                   static_cast<double>(a.peak_allocated.load()) * kToGiB,
                   static_cast<unsigned long long>(a.peak_count.load()));
    });
  }

  // VT_VULKAN_DISPATCH_STATS=1 dumps the per-shader histogram at exit. Registered
  // with atexit rather than printed from a destructor because this context is a
  // never-destroyed process singleton, and because a run that is KILLED by a
  // timeout -- which is exactly how the VK-E runs ended -- still unwinds atexit
  // handlers on SIGTERM only if the handler is installed. A run that never
  // reaches a clean exit still leaves the counters readable via the accessors.
  if (const char* v = std::getenv("VT_VULKAN_DISPATCH_STATS");
      v != nullptr && std::strcmp(v, "0") != 0) {
    std::atexit([] {
      if (!Available()) return;
      const VulkanContext& ctx = Get();
      std::fprintf(stderr, "[vt vulkan] TOTAL DISPATCHES: %llu\n",
                   static_cast<unsigned long long>(ctx.dispatch_count()));
      std::map<std::string, uint64_t> counts;
      for (const auto& kv : ctx.DispatchHistogram()) counts[kv.first] = kv.second;
      const auto times = ctx.DispatchTimeMs();
      double total_ms = 0.0;
      for (const auto& kv : times) total_ms += kv.second;
      // Sorted by TIME, not count. The ordering is the point: reading a
      // count-sorted list is how a cheap shader that runs often gets mistaken
      // for the bottleneck.
      // PER-SHADER TIME IS UNAVAILABLE UNDER BATCHING, and printing 0.0 for
      // every row would read as "these kernels are free" rather than "this was
      // not measured". The wait that attributed time to a shader was the
      // per-dispatch fence; batching submits many dispatches under ONE fence, so
      // there is nothing to attribute. Counts stay exact either way.
      //
      // Getting per-kernel time back under batching needs GPU timestamp queries
      // (vkCmdWriteTimestamp around each dispatch), which is the proper Vulkan
      // answer and is not built. Until then, profile with VT_VULKAN_BATCH=0:
      // relative KERNEL cost is still meaningful there, it just also carries the
      // per-dispatch floor that batching removes.
      if (!ctx.batching_enabled() || total_ms > 0.0) {
        std::fprintf(stderr, "[vt vulkan] %-24s %8s %10s %7s %10s\n", "shader",
                     "count", "total ms", "%", "ms/call");
      } else {
        // Reached only when batching is on AND the device could not timestamp
        // (timestampComputeAndGraphics false, or the pool failed to create).
        std::fprintf(stderr,
                     "[vt vulkan] per-shader TIME unavailable: batching is on and this "
                     "device reports no compute timestamps.\n"
                     "[vt vulkan] Re-run with VT_VULKAN_BATCH=0 for per-kernel ms. "
                     "Counts below are exact.\n");
        std::fprintf(stderr, "[vt vulkan] %-24s %8s\n", "shader", "count");
        for (const auto& kv : counts) {
          std::fprintf(stderr, "[vt vulkan] %-24s %8llu\n", kv.first.c_str(),
                       static_cast<unsigned long long>(kv.second));
        }
        std::fprintf(stderr, "[vt vulkan] %-24s %8llu\n", "TOTAL",
                     static_cast<unsigned long long>(ctx.dispatch_count()));
        return;
      }
      for (const auto& kv : times) {
        const uint64_t n = counts[kv.first];
        std::fprintf(stderr, "[vt vulkan] %-24s %8llu %10.1f %6.1f%% %10.4f\n",
                     kv.first.c_str(), static_cast<unsigned long long>(n), kv.second,
                     total_ms > 0 ? 100.0 * kv.second / total_ms : 0.0,
                     n > 0 ? kv.second / double(n) : 0.0);
      }
      std::fprintf(stderr, "[vt vulkan] %-24s %8llu %10.1f\n", "TOTAL",
                   static_cast<unsigned long long>(ctx.dispatch_count()), total_ms);
      // THE BARRIER LINE. The span is the GPU's own wall over each command
      // buffer; total_ms above is the sum of the individual dispatch intervals.
      // Their difference is the GPU time spent BETWEEN dispatches -- the barrier
      // drains and the launch setup -- which is the only direct measurement of
      // what BACKEND-VULKAN-BARRIERS is trying to remove, and it is invisible in
      // every per-shader row above. Read the gap, not the arms' per-shader sums:
      // once barriers are skipped, adjacent dispatches may OVERLAP and their
      // intervals double-count.
      const double span = ctx.gpu_span_ms();
      std::fprintf(stderr,
                   "[vt vulkan] BARRIERS recorded=%llu skipped=%llu smart=%d\n"
                   "[vt vulkan] GPU span=%.1f ms over %llu cmdbufs, "
                   "sum-of-dispatch=%.1f ms, GAP=%.1f ms (%.1f%%)\n",
                   static_cast<unsigned long long>(ctx.barrier_count()),
                   static_cast<unsigned long long>(ctx.barrier_skip_count()),
                   ctx.smart_barriers() ? 1 : 0, span,
                   static_cast<unsigned long long>(ctx.gpu_span_batches()), total_ms,
                   span - total_ms, span > 0 ? 100.0 * (span - total_ms) / span : 0.0);
    });
  }
  mutex_ = new std::mutex();
}

VulkanContext& VulkanContext::Get() {
  // Function-local static: thread-safe initialization, constructed on first use
  // and deliberately never destroyed (process lifetime, matching llama.cpp's
  // `vk_instance` singleton and the Metal skeleton's MetalContext::Get).
  static VulkanContext* ctx = new VulkanContext();
  return *ctx;
}

void* VulkanContext::AllocBuffer(size_t bytes, void** out_buffer, void** out_memory) {
  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);
  // ROUNDED UP TO A WHOLE 32-BIT WORD, and that is a CORRECTNESS requirement of
  // this backend's storage model, not tidiness.
  //
  // Every operand is bound as a `uint32_t[]` view (and, for float operands, also
  // as a `uint16_t[]` one) over the WHOLE buffer — vt_common.glsl § STORAGE
  // MODEL. An array of `uint` over a buffer of N bytes has floor(N/4) elements,
  // so a buffer whose length is not a multiple of 4 has a TRUNCATED 32-bit view
  // and its last partial word is unreachable. Under robustBufferAccess that read
  // returns zero; without it, it is undefined. Either way it is silent.
  //
  // MEASURED (BACKEND-VULKAN-GDN): a 3-byte i8 `has_initial_state[3]` — the GDN
  // per-request "does this row have a prior state" flag, which the gather shader
  // reads byte-wise through the 32-bit view precisely so it need not require
  // VK_KHR_8bit_storage — produced a 0-element view, every flag read back as
  // false, and the gather ZEROED rows it should have copied. The gate caught it
  // as a memcmp mismatch against the CPU oracle.
  //
  // Nothing before that read a non-multiple-of-4 buffer through the 32-bit view
  // (f32/i32/i64 lengths are multiples of 4 by construction, and 16-bit dtypes go
  // through the 16-bit view), which is why the skeleton lived with it. The fix
  // belongs HERE rather than in one shader: any future byte- or word-granular
  // read of a small operand would hit the same edge.
  //
  // A zero-length VkBuffer is also invalid, and the rounding covers that too: a
  // 0-byte request still yields a distinct freeable pointer, which is the CPU
  // backend's contract and what vt::StepArena relies on. Only the BUFFER LENGTH
  // grows; the mapped pointer and every byte the caller wrote are untouched, so
  // Copy/Memset stay bit-exact.
  const VkDeviceSize len = bytes == 0 ? 4 : static_cast<VkDeviceSize>((bytes + 3) & ~size_t{3});

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = len;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buffer = VK_NULL_HANDLE;
  Check(vk.vkCreateBuffer(device, &bci, nullptr, &buffer), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vk.vkGetBufferMemoryRequirements(device, buffer, &req);
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = memory_type_index_;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  Check(vk.vkAllocateMemory(device, &mai, nullptr, &memory), "vkAllocateMemory");
  Check(vk.vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");

  // Account AFTER the allocation succeeded, so a failed allocation never
  // inflates the live total. `req.size` is what the driver committed; `len` is
  // what we asked for. On GB10 they are equal for every size the model loader
  // uses, and the gap -- if a driver ever introduces one -- is exactly the
  // "allocated is bigger than the tensor bytes" term this accounting exists to
  // separate from a host mirror.
  {
    {
      std::lock_guard<std::mutex> g(AllocSizeMutex());
      AllocSizes()[Pack(memory)] = {static_cast<uint64_t>(len), static_cast<uint64_t>(req.size)};
    }
    AllocAccounting& a = Accounting();
    a.live_count.fetch_add(1, std::memory_order_relaxed);
    a.total_count.fetch_add(1, std::memory_order_relaxed);
    a.live_requested.fetch_add(len, std::memory_order_relaxed);
    a.total_requested.fetch_add(len, std::memory_order_relaxed);
    a.total_allocated.fetch_add(req.size, std::memory_order_relaxed);
    const uint64_t live =
        a.live_allocated.fetch_add(req.size, std::memory_order_relaxed) + req.size;
    uint64_t peak = a.peak_allocated.load(std::memory_order_relaxed);
    while (live > peak &&
           !a.peak_allocated.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
    }
    uint64_t pc = a.peak_count.load(std::memory_order_relaxed);
    const uint64_t lc = a.live_count.load(std::memory_order_relaxed);
    while (lc > pc &&
           !a.peak_count.compare_exchange_weak(pc, lc, std::memory_order_relaxed)) {
    }
    if (kAllocStats) {
      // Report on each new 1 GiB high-water mark. A per-allocation line would be
      // hundreds of thousands of lines on a 27B and would itself perturb the
      // load; the high-water crossings are what a memory attribution needs.
      constexpr uint64_t kStep = uint64_t{1} << 30;
      uint64_t mark = a.next_report.load(std::memory_order_relaxed);
      if (live >= mark &&
          a.next_report.compare_exchange_strong(mark, ((live / kStep) + 1) * kStep,
                                                std::memory_order_relaxed)) {
        ReportAllocState("high-water");
      }
    }
  }

  void* mapped = nullptr;
  Check(vk.vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
  // vt::StepArena depends on >= 64-byte alignment (include/vt/backend.h:26).
  // A whole VkDeviceMemory mapping is at least `minMemoryMapAlignment`
  // (>= 64 by spec) aligned, so this holds by construction; assert it anyway
  // because everything downstream silently depends on it.
  VT_CHECK(reinterpret_cast<uintptr_t>(mapped) % 64 == 0,
           "vulkan: mapped allocation is not 64-byte aligned");

  *out_buffer = Pack(buffer);
  *out_memory = Pack(memory);
  return mapped;
}

void VulkanContext::FreeBuffer(void* buffer, void* memory) {
  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);
  {
    std::pair<uint64_t, uint64_t> sizes{0, 0};
    {
      std::lock_guard<std::mutex> g(AllocSizeMutex());
      auto& m = AllocSizes();
      auto it = m.find(memory);
      if (it != m.end()) {
        sizes = it->second;
        m.erase(it);
      }
    }
    AllocAccounting& a = Accounting();
    a.live_count.fetch_sub(1, std::memory_order_relaxed);
    a.live_requested.fetch_sub(sizes.first, std::memory_order_relaxed);
    a.live_allocated.fetch_sub(sizes.second, std::memory_order_relaxed);
  }
  vk.vkUnmapMemory(device, Unpack<VkDeviceMemory>(memory));
  vk.vkDestroyBuffer(device, Unpack<VkBuffer>(buffer), nullptr);
  vk.vkFreeMemory(device, Unpack<VkDeviceMemory>(memory), nullptr);
}

DeviceAllocStats DeviceAllocStatsSnapshot() {
  const AllocAccounting& a = Accounting();
  DeviceAllocStats s;
  s.live_count = a.live_count.load(std::memory_order_relaxed);
  s.total_count = a.total_count.load(std::memory_order_relaxed);
  s.live_requested = a.live_requested.load(std::memory_order_relaxed);
  s.live_allocated = a.live_allocated.load(std::memory_order_relaxed);
  s.total_requested = a.total_requested.load(std::memory_order_relaxed);
  s.total_allocated = a.total_allocated.load(std::memory_order_relaxed);
  s.peak_allocated = a.peak_allocated.load(std::memory_order_relaxed);
  s.peak_count = a.peak_count.load(std::memory_order_relaxed);
  return s;
}

namespace {

// The pipeline cache key: the module name plus its specialization values, which
// together identify one VkPipeline. Decimal so the key reads back in the
// VT_CHECK messages below.
std::string PipelineKey(const std::string& name, const uint32_t* spec_values,
                        uint32_t spec_count) {
  std::string key = name;
  for (uint32_t i = 0; i < spec_count; ++i) {
    key += (i == 0 ? '|' : ',');
    key += std::to_string(spec_values[i]);
  }
  return key;
}

}  // namespace

VulkanContext::Pipeline& VulkanContext::GetPipeline(const std::string& name,
                                                    uint32_t buffer_count, uint32_t push_size,
                                                    const uint32_t* spec_values,
                                                    uint32_t spec_count) {
  const std::string key = PipelineKey(name, spec_values, spec_count);
  auto& cache = *static_cast<std::map<std::string, Pipeline>*>(pipelines_);
  auto it = cache.find(key);
  if (it != cache.end()) {
    // A kernel's binding count and push-constant size are properties of its
    // SPIR-V; a mismatch means the host and the committed shader have drifted,
    // which would corrupt memory rather than fail cleanly.
    VT_CHECK(it->second.buffer_count == buffer_count && it->second.push_size == push_size,
             "vulkan: pipeline " + key + " re-requested with a different binding layout");
    return it->second;
  }

  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);

  const SpirvModule* module = nullptr;
  for (size_t i = 0; i < kSpirvModuleCount; ++i) {
    if (name == kSpirvModules[i].name) { module = &kSpirvModules[i]; break; }
  }
  VT_CHECK(module != nullptr,
           "vulkan: no committed SPIR-V for kernel '" + name +
               "' — regenerate with scripts/gen-vulkan-spirv.py");

  // Specialization values are passed POSITIONALLY against the module's declared
  // SpecIds. Vulkan SILENTLY IGNORES a map entry whose constantID the module does
  // not declare, so an unchecked host/shader drift is wrong numbers rather than an
  // error — the same class as the binding-layout check above, and just as fatal.
  VT_CHECK(spec_count == module->spec_id_count,
           "vulkan: kernel '" + name + "' declares " +
               std::to_string(module->spec_id_count) +
               " specialization constant(s) but " + std::to_string(spec_count) +
               " value(s) were supplied — host and committed SPIR-V have drifted;"
               " regenerate with scripts/gen-vulkan-spirv.py");

  // The binding COUNT must agree too, and for a second reason beyond the layout:
  // the writable mask below is indexed by binding, so a host that bound more
  // buffers than the module declares would get mask bit 0 -- "read-only" -- for
  // the extra ones, and the hazard analysis would then be free to skip a barrier
  // for a buffer the shader can write. Bound LOUDLY rather than conservatively,
  // because a mismatch is a host/shader drift that also breaks the descriptor set
  // layout and should never be tolerated silently.
  VT_CHECK(module->binding_count == buffer_count,
           "vulkan: kernel '" + name + "' declares " +
               std::to_string(module->binding_count) + " descriptor binding(s) but " +
               std::to_string(buffer_count) +
               " buffer(s) were bound — host and committed SPIR-V have drifted;"
               " regenerate with scripts/gen-vulkan-spirv.py");

  Pipeline p;
  p.buffer_count = buffer_count;
  p.push_size = push_size;
  p.writable_mask = module->writable_mask;
  // The committed SPIR-V table has static storage duration, so this pointer
  // outlives the pipeline and the histogram can key on it without copying.
  p.module_name = module->name;

  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = module->word_count * sizeof(uint32_t);
  smci.pCode = module->words;
  Check(vk.vkCreateShaderModule(device, &smci, nullptr, &p.module), "vkCreateShaderModule");

  // One descriptor-set layout per pipeline, with exactly its binding count.
  // llama.cpp instead shares ONE 12-binding layout across every pipeline
  // (ggml-vulkan.cpp:6424-6437); per-pipeline is simpler here because the
  // pipeline LAYOUT already has to be per-pipeline (push-constant sizes differ)
  // and it removes any question about descriptors a shader never declares.
  std::vector<VkDescriptorSetLayoutBinding> bindings(buffer_count);
  for (uint32_t i = 0; i < buffer_count; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = buffer_count;
  dslci.pBindings = bindings.data();
  Check(vk.vkCreateDescriptorSetLayout(device, &dslci, nullptr, &p.set_layout),
        "vkCreateDescriptorSetLayout");

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = push_size;
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &p.set_layout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  Check(vk.vkCreatePipelineLayout(device, &plci, nullptr, &p.layout), "vkCreatePipelineLayout");

  // One uint32 per constant, tightly packed; entry i carries the module's i-th
  // declared SpecId (the generator emits them sorted ascending). These two must
  // OUTLIVE vkCreateComputePipelines below — a temporary whose address is
  // captured and read later is the use-after-free class this project has already
  // hit twice with CUDA-graph capture, so they are named locals in this scope,
  // never a nested block.
  std::vector<VkSpecializationMapEntry> spec_entries(spec_count);
  for (uint32_t i = 0; i < spec_count; ++i) {
    spec_entries[i].constantID = module->spec_ids[i];
    spec_entries[i].offset = i * static_cast<uint32_t>(sizeof(uint32_t));
    spec_entries[i].size = sizeof(uint32_t);
  }
  VkSpecializationInfo spec_info{};
  spec_info.mapEntryCount = spec_count;
  spec_info.pMapEntries = spec_entries.data();
  spec_info.dataSize = spec_count * sizeof(uint32_t);
  spec_info.pData = spec_values;

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = p.module;
  cpci.stage.pName = "main";
  cpci.stage.pSpecializationInfo = spec_count != 0 ? &spec_info : nullptr;
  cpci.layout = p.layout;
  Check(vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline),
        "vkCreateComputePipelines");

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = Unpack<VkDescriptorPool>(descriptor_pool_);
  dsai.descriptorSetCount = kDescriptorRing;
  VkDescriptorSetLayout layouts[kDescriptorRing];
  for (uint32_t i = 0; i < kDescriptorRing; ++i) layouts[i] = p.set_layout;
  dsai.pSetLayouts = layouts;
  // Named, because the plausible cause is pool exhaustion from specialization
  // (one set per PIPELINE, and a module can have many), which is otherwise a bare
  // VkResult a long way from its reason.
  Check(vk.vkAllocateDescriptorSets(device, &dsai, p.sets),
        ("vkAllocateDescriptorSets for pipeline '" + key +
         "' (descriptor pool may be exhausted by specialized pipelines)").c_str());

  return cache.emplace(key, p).first->second;
}

bool VulkanContext::PipelineExistsFor(const std::string& name) const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  const auto& cache = *static_cast<std::map<std::string, Pipeline>*>(pipelines_);
  // Keys are "<module>" or "<module>|<spec values>", so a prefix match up to the
  // separator identifies every specialization of one module.
  for (const auto& kv : cache) {
    if (kv.first == name || kv.first.compare(0, name.size() + 1, name + "|") == 0) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> VulkanContext::PipelineKeys() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  const auto& cache = *static_cast<std::map<std::string, Pipeline>*>(pipelines_);
  std::vector<std::string> out;
  out.reserve(cache.size());
  for (const auto& kv : cache) out.push_back(kv.first);
  return out;
}

// A filter over PipelineKeys() rather than a second walk of the cache. It must
// NOT take the mutex: PipelineKeys() already does, and that mutex is not
// recursive, so locking here would deadlock.
std::vector<std::string> VulkanContext::PipelineKeysFor(const std::string& name) const {
  std::vector<std::string> keys;
  for (auto& key : PipelineKeys()) {
    // Same prefix rule as PipelineExistsFor: "<module>" or "<module>|<spec...>".
    if (key == name || key.compare(0, name.size() + 1, name + "|") == 0) {
      keys.push_back(std::move(key));
    }
  }
  return keys;
}

uint64_t VulkanContext::dispatch_count() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return dispatch_total_;
}

std::vector<std::pair<std::string, double>> VulkanContext::DispatchTimeMs() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  const auto& ms = *static_cast<std::map<std::string, double>*>(dispatch_ms_);
  std::vector<std::pair<std::string, double>> out(ms.begin(), ms.end());
  std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  return out;
}

// Aggregated on demand from the per-pipeline counters rather than maintained by
// the dispatch path. Several PIPELINES can share one MODULE name (one per
// specialization), so the counts are summed by module name, which is exactly what
// the map keyed on `name` used to hold.
std::vector<std::pair<std::string, uint64_t>> VulkanContext::DispatchHistogram() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  std::map<std::string, uint64_t> hist;
  for (const auto& kv : *static_cast<std::map<std::string, Pipeline>*>(pipelines_)) {
    if (kv.second.dispatches == 0) continue;
    hist[kv.second.module_name != nullptr ? kv.second.module_name : kv.first] +=
        kv.second.dispatches;
  }
  std::vector<std::pair<std::string, uint64_t>> out(hist.begin(), hist.end());
  std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  return out;
}

size_t VulkanContext::PipelineCacheSize() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return static_cast<std::map<std::string, Pipeline>*>(pipelines_)->size();
}

bool VulkanContext::batching_enabled() const { return kBatchDispatch; }

uint32_t VulkanContext::pending_batch() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return batch_count_;
}

uint32_t VulkanContext::in_flight_batches() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  uint32_t n = 0;
  for (uint32_t i = 0; i < kMaxInFlight; ++i) n += slot_in_flight_[i] ? 1u : 0u;
  return n;
}

uint32_t VulkanContext::in_flight_limit() const { return kInFlight; }

uint32_t VulkanContext::ring_slice() const { return kRingSlice; }

uint32_t VulkanContext::ring_depth() const { return kRingDepth; }

uint64_t VulkanContext::submit_count() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return submit_count_;
}

uint64_t VulkanContext::fence_wait_count() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return fence_wait_count_;
}

uint32_t VulkanContext::ring_base() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return slot_ring_base_;
}

uint64_t VulkanContext::barrier_count() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return barrier_count_;
}

uint64_t VulkanContext::barrier_skip_count() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return barrier_skipped_;
}

bool VulkanContext::smart_barriers() const {
  if (smart_barriers_override_ != 0) return smart_barriers_override_ > 0;
  return kSmartBarriersEnv;
}

void VulkanContext::set_smart_barriers_override(int v) {
  // Drain FIRST. hazard_written_/hazard_read_ describe commands already recorded
  // under the OLD policy; switching while a batch is open or in flight would let
  // the new policy reason about a history it did not build. After a drain the GPU
  // is idle and the next dispatch starts from a clean slate either way.
  //
  // The sets are deliberately NOT cleared here even so -- see the header for why
  // they are cleared only by an emitted barrier. Keeping them merely costs the
  // next dispatch a barrier it may not have needed.
  FlushBatch("smart-barrier-policy-change");
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  smart_barriers_override_ = v;
  force_barrier_next_ = true;
}

double VulkanContext::gpu_span_ms() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return static_cast<double>(gpu_span_ns_) / 1.0e6;
}

uint64_t VulkanContext::gpu_span_batches() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return gpu_span_batches_;
}

void VulkanContext::FlushIfBatchTouches(void* buffer, const char* why) {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  // A host pointer (nullptr) cannot alias a bound VkBuffer, so it never forces a
  // drain. Otherwise drain only on a genuine intersection.
  if (buffer == nullptr) return;
  // The bound-buffer table now spans every batch that has not been DRAINED, not
  // just the open one. With submission pipelined, "no batch is open" no longer
  // implies "the GPU is finished": a submitted batch can still be writing the
  // buffer the host is about to read, and returning early here would be exactly
  // the silent stale-read this backend's flush contract exists to prevent.
  if (!static_cast<BufferSet*>(batch_buffers_)->Contains(buffer)) return;
  DrainLocked(why);
}

void VulkanContext::FlushBatch(const char* why) {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  DrainLocked(why);
}

// Submits whatever is open and then waits for EVERY submitted batch. This is the
// operation the three host-read paths need, and it is what FlushBatch meant
// before submission was pipelined.
void VulkanContext::DrainLocked(const char* why) {
  FlushBatchLocked(why);
  for (uint32_t i = 0; i < kInFlight; ++i) RetireSlotLocked(i);
  // Safe only now: every batch that could still have been reading these buffers
  // has completed.
  static_cast<BufferSet*>(batch_buffers_)->Clear();
}

// Waits for slot `s`, reads back its timestamps and frees its command buffer and
// its slice of every descriptor ring. A no-op if the slot is idle.
void VulkanContext::RetireSlotLocked(uint32_t s) {
  if (!slot_in_flight_[s]) return;
  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);
  auto fence = Unpack<VkFence>(slot_fence_[s]);
  const uint64_t hp_t0 = kHostProfile ? NowNs() : 0;
  // Counted BEFORE the wait and only when the fence is genuinely unsignalled, so
  // the counter measures blocked retirements rather than retirements.
  if (vk.vkGetFenceStatus(device, fence) == VK_NOT_READY) ++fence_wait_count_;
  if (kFenceTimeoutMs > 0) {
    VkResult wr = vk.vkWaitForFences(device, 1, &fence, VK_TRUE,
                                      static_cast<uint64_t>(kFenceTimeoutMs) * 1000000ull);
    if (wr != VK_SUCCESS) {
      const char* reason = wr == VK_TIMEOUT
          ? "retired dispatch did not signal in time - GPU wedged"
          : "vkWaitForFences returned a non-timeout error - device lost or other fatal error";
      std::fprintf(stderr,
        "[vt vulkan] FATAL: %s (wr=%d, VT_VK_FENCE_TIMEOUT_MS=%lld), aborting\n",
        reason, static_cast<int>(wr), static_cast<long long>(kFenceTimeoutMs));
      std::fflush(stderr); std::abort();
    }
  } else {
    Check(vk.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  }
  const uint64_t hp_t1 = kHostProfile ? NowNs() : 0;

  // Read the timestamps back, now that this slot has certainly completed.
  auto* names = static_cast<std::vector<std::string>*>(slot_names_[s]);
  if (query_pool_ != nullptr && !names->empty()) {
    const uint32_t n = static_cast<uint32_t>(names->size());
    const uint32_t base = s * kMaxBatch * 2;
    std::vector<uint64_t> ticks(n * 2, 0);
    if (vk.vkGetQueryPoolResults(device, Unpack<VkQueryPool>(query_pool_), base, n * 2,
                                 ticks.size() * sizeof(uint64_t), ticks.data(),
                                 sizeof(uint64_t),
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
        VK_SUCCESS) {
      auto& acc = *static_cast<std::map<std::string, double>*>(dispatch_ms_);
      // Earliest top-of-pipe and latest bottom-of-pipe over the whole command
      // buffer. Taken as a MIN and a MAX rather than "first and last", because
      // once barriers are skipped the dispatches may overlap and need not retire
      // in the order they were recorded. Zero ticks mean the query was never
      // written and are excluded, or they would drag the span back to the epoch.
      uint64_t span_lo = 0, span_hi = 0;
      for (uint32_t i = 0; i < n; ++i) {
        const uint64_t t0 = ticks[i * 2], t1 = ticks[i * 2 + 1];
        if (t1 > t0) acc[(*names)[i]] += double(t1 - t0) * timestamp_period_ns_ / 1.0e6;
        if (t0 == 0 || t1 == 0) continue;
        if (span_lo == 0 || t0 < span_lo) span_lo = t0;
        if (t1 > span_hi) span_hi = t1;
      }
      if (span_hi > span_lo && span_lo != 0) {
        gpu_span_ns_ += static_cast<uint64_t>(double(span_hi - span_lo) *
                                              timestamp_period_ns_);
        ++gpu_span_batches_;
      }
    }
    names->clear();
  }

  Check(vk.vkResetFences(device, 1, &fence), "vkResetFences");
  Check(vk.vkResetCommandPool(device, Unpack<VkCommandPool>(slot_pool_[s]), 0),
        "vkResetCommandPool");
  slot_in_flight_[s] = false;
  if (kHostProfile) {
    const uint64_t hp_t2 = NowNs();
    g_host_profile.ns_flush_wait += hp_t1 - hp_t0;
    g_host_profile.ns_flush_other += hp_t2 - hp_t1;
  }
}

// Ends the open command buffer, submits it, and WAITS. The wait is what makes
// every descriptor set in the batch free to reuse and every write visible to the
// host, so it is not an optimisation to drop: without it the reset below would
// race the GPU.
void VulkanContext::FlushBatchLocked(const char* why) {
  if (!batch_open_) return;
  if (kDispatchStats) {
    static std::mutex fw_mu;
    static std::map<std::string, std::pair<uint64_t, uint64_t>> fw;  // reason -> (count, dispatches)
    static bool reg = false;
    std::lock_guard<std::mutex> g(fw_mu);
    auto& e = fw[why];
    e.first += 1;
    e.second += batch_count_;
    if (!reg) {
      reg = true;
      static auto* snap = &fw;
      static auto* snap_mu = &fw_mu;
      std::atexit([] {
        std::lock_guard<std::mutex> g2(*snap_mu);
        std::fprintf(stderr, "[vt vulkan] FLUSH TRIGGERS  %-16s %10s %12s %8s\n",
                     "reason", "flushes", "dispatches", "avg");
        for (const auto& kv : *snap) {
          std::fprintf(stderr, "[vt vulkan]                %-16s %10llu %12llu %8.1f\n",
                       kv.first.c_str(),
                       static_cast<unsigned long long>(kv.second.first),
                       static_cast<unsigned long long>(kv.second.second),
                       kv.second.first ? double(kv.second.second) / double(kv.second.first) : 0.0);
        }
      });
    }
  }
  const VulkanApi& vk = Api();
  auto cmd = Unpack<VkCommandBuffer>(slot_cmd_[slot_]);
  const uint64_t hp_t0 = kHostProfile ? NowNs() : 0;
  Check(vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  auto fence = Unpack<VkFence>(slot_fence_[slot_]);
  Check(vk.vkQueueSubmit(Unpack<VkQueue>(queue_), 1, &si, fence), "vkQueueSubmit");
  slot_in_flight_[slot_] = true;
  ++submit_count_;
  if (kDispatchStats) {
    std::fprintf(stderr, "[vt vulkan] FLUSH %u dispatches in one submit\n", batch_count_);
    std::fflush(stderr);
  }
  if (kHostProfile) {
    g_host_profile.flushes += 1;
    g_host_profile.ns_flush_submit += NowNs() - hp_t0;
  }

  // Every pipeline's ring counter restarts at the NEXT slot's base. It is the
  // slot rotation, not this reset, that keeps the sets safe: the counter only
  // ever indexes within the slot's own slice.
  for (auto& kv : *static_cast<std::map<std::string, Pipeline>*>(pipelines_)) {
    kv.second.used_this_batch = 0;
  }
  batch_open_ = false;
  batch_count_ = 0;

  // Move to the next slot and make sure it is free. With kInFlight == 1 this
  // retires the slot just submitted, which is byte-for-byte the old
  // submit-and-wait behaviour. With kInFlight == 2 the host returns immediately
  // and records the next batch while the GPU runs this one; the wait only
  // happens when the rotation catches up with still-executing work.
  //
  // NOTE the bound-buffer table is deliberately NOT cleared here. It has to keep
  // naming every buffer any UNRETIRED batch touched, because that is what
  // FlushIfBatchTouches consults to decide whether a host read must drain.
  slot_ = (slot_ + 1) % kInFlight;
  slot_ring_base_ = slot_ * kRingSlice;
  RetireSlotLocked(slot_);
}

void VulkanContext::Dispatch(const std::string& name, const void* const* buffers,
                             uint32_t buffer_count, const void* push_constants,
                             uint32_t push_size, uint32_t group_count_x, uint32_t group_count_y,
                             const uint32_t* spec_values, uint32_t spec_count) {
  if (group_count_x == 0 || group_count_y == 0) return;
  VT_CHECK(group_count_y <= max_workgroup_count_y_,
           "vulkan: dispatch needs " + std::to_string(group_count_y) +
               " Y workgroups, above the device limit of " +
               std::to_string(max_workgroup_count_y_));  // nothing to do; an empty dispatch is illegal
  VT_CHECK(group_count_x <= max_workgroup_count_x_,
           "vulkan: dispatch needs " + std::to_string(group_count_x) +
               " workgroups, above the device limit of " +
               std::to_string(max_workgroup_count_x_));

  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);
  // The single command buffer and each pipeline's single descriptor set are
  // re-recorded per dispatch, so the whole record-submit-wait must be
  // serialized. Correct, not fast — the same trade the Metal skeleton makes with
  // one command buffer per op (src/vt/metal/metal_ops.mm § DISPATCH MODEL).
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));

  const uint64_t hp_t0 = kHostProfile ? NowNs() : 0;
  // Flush nanoseconds accumulated by a flush this Dispatch triggers are billed to
  // the flush phases, so they are subtracted from this dispatch's own total.
  const uint64_t hp_flush_before =
      kHostProfile ? g_host_profile.ns_flush_submit + g_host_profile.ns_flush_wait +
                         g_host_profile.ns_flush_other
                   : 0;
  if (kHostProfile) DumpHostProfileAtExit();

  // Counted under the mutex the dispatch already holds -- see the header for why
  // this is measured on OUR side rather than inferred from context switches.
  ++dispatch_total_;
  // PERIODIC dump, not just atexit: `timeout` sends SIGTERM, whose default action
  // terminates WITHOUT running atexit handlers, and every VK-E run so far ended
  // exactly that way. A diagnostic that only reports on a clean exit would have
  // reported nothing on precisely the runs worth diagnosing.
  if (kDispatchStats && dispatch_total_ % 100 == 0) {
    const auto now = std::chrono::steady_clock::now();
    const double secs =
        std::chrono::duration<double>(now - g_dispatch_t0).count();
    std::fprintf(stderr, "[vt vulkan] dispatches=%llu  elapsed=%.1fs  rate=%.0f/s\n",
                 static_cast<unsigned long long>(dispatch_total_), secs,
                 secs > 0 ? dispatch_total_ / secs : 0.0);
  }

  const uint64_t hp_t_book = kHostProfile ? NowNs() : 0;
  Pipeline& p = GetPipeline(name, buffer_count, push_size, spec_values, spec_count);
  const uint64_t hp_t_pipe = kHostProfile ? NowNs() : 0;

  // A pipeline that has consumed this slot's whole ring SLICE must flush before it
  // can reuse the slice's first set, because the GPU has not necessarily read the
  // earlier ones yet. Flushing rotates to the next slot, which owns a DIFFERENT
  // slice, and waits only if that slot is still executing.
  const uint32_t ring_limit = kBatchDispatch ? kRingSlice : kRingDepth;
  if (kBatchDispatch && (p.used_this_batch >= ring_limit || batch_count_ >= kMaxBatch)) {
    FlushBatchLocked(p.used_this_batch >= ring_limit ? "ring-full" : "batch-cap");
  }

  VkDescriptorSet set =
      kBatchDispatch ? p.sets[slot_ring_base_ + p.used_this_batch] : p.sets[0];

  // Stack arrays, not two heap vectors per dispatch. kMaxBindings is the same
  // bound GetPipeline already enforces against the committed SPIR-V.
  VT_CHECK(buffer_count <= kMaxDispatchBindings,
           "vulkan: dispatch binds " + std::to_string(buffer_count) +
               " buffers, above the backend limit of " +
               std::to_string(kMaxDispatchBindings));
  VkDescriptorBufferInfo infos[kMaxDispatchBindings];
  VkWriteDescriptorSet writes[kMaxDispatchBindings];
  for (uint32_t i = 0; i < buffer_count; ++i) {
    infos[i].buffer = Unpack<VkBuffer>(const_cast<void*>(buffers[i]));
    infos[i].offset = 0;  // always WHOLE; the element offset rides push constants
    infos[i].range = VK_WHOLE_SIZE;
    writes[i] = VkWriteDescriptorSet{};
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vk.vkUpdateDescriptorSets(device, buffer_count, writes, 0, nullptr);
  const uint64_t hp_t_desc = kHostProfile ? NowNs() : 0;

  auto cmd = Unpack<VkCommandBuffer>(slot_cmd_[slot_]);
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (!kBatchDispatch) {
    Check(vk.vkResetCommandPool(device, Unpack<VkCommandPool>(slot_pool_[slot_]), 0),
          "vkResetCommandPool");
    Check(vk.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
  } else {
    if (!batch_open_) {
      // The pool was already reset by RetireSlotLocked, which is the only point at
      // which this slot's previous command buffer is provably no longer executing.
      Check(vk.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
      // The query pool must be reset on the DEVICE timeline, inside the command
      // buffer, before any query in it is written -- and only over THIS slot's
      // range, or it would clobber an in-flight slot's unread results.
      if (query_pool_ != nullptr) {
        vk.vkCmdResetQueryPool(cmd, Unpack<VkQueryPool>(query_pool_),
                               slot_ * kMaxBatch * 2, kMaxBatch * 2);
        static_cast<std::vector<std::string>*>(slot_names_[slot_])->clear();
      }
      batch_open_ = true;
    }
    // BEFORE every recorded dispatch, INCLUDING the first in a command buffer: the
    // ops in a decode step are sequentially dependent (norm feeds projection feeds
    // attention), so every dispatch must see the previous one's writes. Without
    // this the batch would run them concurrently and compute garbage.
    //
    // The first-in-buffer case matters only once submission is pipelined. Before
    // this row the previous batch's fence had already been waited on, which is a
    // stronger dependency than any barrier; now the previous batch may still be
    // executing, and this barrier -- which orders against everything earlier in
    // SUBMISSION ORDER on the same queue, not merely earlier in this buffer -- is
    // what carries the dependency across the command-buffer boundary.
    //
    // SMART BARRIERS (BACKEND-VULKAN-BARRIERS) narrow "every dispatch" to "every
    // dispatch that could actually observe or clobber an earlier one". The
    // decision DEFAULTS TO EMITTING: a barrier is skipped only when this
    // dispatch's operands are proven disjoint, in the hazardous direction, from
    // every operand recorded since the previous barrier. The three sets and the
    // invariant they maintain are documented on hazard_written_ in the header,
    // and the cross-command-buffer guarantee above is unchanged -- when the test
    // says hazard, the barrier recorded is exactly the one that was always
    // recorded, with exactly the same submission-order scope.
    auto& written = *static_cast<BufferSet*>(hazard_written_);
    auto& read = *static_cast<BufferSet*>(hazard_read_);
    const bool analyse = smart_barriers_override_ != 0 ? smart_barriers_override_ > 0
                                                       : kSmartBarriersEnv;
    // The always-barrier arm does not maintain the access history -- it does not
    // need it, and paying for it there would make the control arm slower than the
    // tree it is the control FOR. That leaves the history empty on the first
    // dispatch after the policy is switched ON, which would read as "no hazard"
    // when the truth is "no history". force_barrier_next_ makes that transition
    // emit one barrier, after which the invariant is established honestly.
    bool hazard = true;
    if (analyse && !force_barrier_next_) {
      hazard = false;
      for (uint32_t i = 0; i < buffer_count && !hazard; ++i) {
        void* b = const_cast<void*>(buffers[i]);
        if ((p.writable_mask >> i) & 1u) {
          // A WRITE collides with an earlier write (WAW) and with an earlier read
          // (WAR); the latter is why the read set has to exist at all.
          hazard = written.Contains(b) || read.Contains(b);
        } else {
          // A READ collides only with an earlier WRITE (RAW). Two dispatches that
          // merely read the same buffer -- the gate and up projections sharing one
          // normalized activation, say -- need no ordering between them.
          hazard = written.Contains(b);
        }
      }
    }
    if (hazard) {
      VkMemoryBarrier mb{};
      mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr,
                              0, nullptr);
      ++barrier_count_;
      // The barrier orders this dispatch after every command submitted earlier on
      // this queue, so nothing before it can still be a hazard and the history
      // restarts empty. Cleared HERE and nowhere else -- not at a flush, not at a
      // drain -- which is what makes the invariant hold across command buffers.
      written.Clear();
      read.Clear();
      force_barrier_next_ = false;
    } else {
      ++barrier_skipped_;
    }
    // Record this dispatch's accesses AFTER the decision, so a dispatch is never
    // tested against itself. A buffer bound at both a readable and a writable
    // binding is entered in BOTH sets, which is a superset of the truth and can
    // only ever produce an extra barrier.
    if (analyse) {
      for (uint32_t i = 0; i < buffer_count; ++i) {
        void* b = const_cast<void*>(buffers[i]);
        if ((p.writable_mask >> i) & 1u) {
          written.Insert(b);
        } else {
          read.Insert(b);
        }
      }
    }
  }

  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
  vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0,
                             nullptr);
  vk.vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                        push_constants);
  const bool timed = kBatchDispatch && query_pool_ != nullptr && batch_count_ < kMaxBatch;
  const uint32_t query_base = (slot_ * kMaxBatch + batch_count_) * 2;
  if (timed) {
    // TOP_OF_PIPE before / BOTTOM_OF_PIPE after brackets this dispatch's execution
    // on the GPU. Because a barrier separates consecutive dispatches, the interval
    // is this kernel's own time rather than an overlap with its neighbours.
    vk.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           Unpack<VkQueryPool>(query_pool_), query_base);
  }
  vk.vkCmdDispatch(cmd, group_count_x, group_count_y, 1);
  if (timed) {
    vk.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                           Unpack<VkQueryPool>(query_pool_), query_base + 1);
    static_cast<std::vector<std::string>*>(slot_names_[slot_])->push_back(name);
  }

  const uint64_t hp_t_rec = kHostProfile ? NowNs() : 0;

  if (kBatchDispatch) {
    auto& bound = *static_cast<BufferSet*>(batch_buffers_);
    for (uint32_t i = 0; i < buffer_count; ++i) {
      bound.Insert(const_cast<void*>(buffers[i]));
    }
    ++p.dispatches;
    ++p.used_this_batch;
    ++batch_count_;
    if (kHostProfile) {
      const uint64_t hp_t_end = NowNs();
      const uint64_t flush_ns = g_host_profile.ns_flush_submit +
                                g_host_profile.ns_flush_wait +
                                g_host_profile.ns_flush_other - hp_flush_before;
      g_host_profile.dispatches += 1;
      g_host_profile.ns_bookkeep += (hp_t_book - hp_t0) + (hp_t_end - hp_t_rec);
      g_host_profile.ns_pipeline += hp_t_pipe - hp_t_book;
      g_host_profile.ns_descriptor += (hp_t_desc - hp_t_pipe) - flush_ns;
      g_host_profile.ns_record += hp_t_rec - hp_t_desc;
      g_host_profile.ns_total += (hp_t_end - hp_t0) - flush_ns;
    }
    return;  // submitted by FlushBatch, at the next host read or Synchronize
  }
  ++p.dispatches;
  Check(vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  auto fence = Unpack<VkFence>(fence_);
  Check(vk.vkResetFences(device, 1, &fence), "vkResetFences");
  Check(vk.vkQueueSubmit(Unpack<VkQueue>(queue_), 1, &si, fence), "vkQueueSubmit");
  // Blocking wait: the whole backend is synchronous in W0, so by the time an op
  // returns the host may read the mapped memory directly. Host-coherent memory
  // needs no invalidate, and vkQueueSubmit itself makes prior host writes
  // visible to the device (the host-write ordering guarantee), so there is no
  // flush on the way in either.
  // PER-DISPATCH TIMING. The periodic counter showed fewer than 2000 dispatches
  // in 150 s, which rules out "many cheap submits" and points at a few very
  // expensive ones -- so the useful diagnostic is WHICH shader is slow, not how
  // many ran. Anything over the threshold names itself.
  // Printed BEFORE the wait, and flushed. The timing print below runs only if the
  // wait RETURNS -- so if a fence never completes, the post-wait line never
  // appears and the hang is invisible. The last line printed here names the
  // dispatch that hung.
  if (kDispatchStats) {
    std::fprintf(stderr, "[vt vulkan] submit #%llu %-22s groups=%u\n",
                 static_cast<unsigned long long>(dispatch_total_), name.c_str(),
                 group_count_x);
    std::fflush(stderr);
  }
  const auto wait_t0 = std::chrono::steady_clock::now();
  Check(vk.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  if (kDispatchStats) {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - wait_t0).count();
    // Attributed to the shader that just ran. Accumulated OUTSIDE the dispatch
    // mutex would race; this whole function already holds it.
    (*static_cast<std::map<std::string, double>*>(dispatch_ms_))[name] += ms;
    if (ms > 200.0) {
      std::fprintf(stderr, "[vt vulkan] SLOW dispatch %-22s %8.1f ms  groups=%u\n",
                   name.c_str(), ms, group_count_x);
    }
  }
}

}  // namespace vt::vulkan
