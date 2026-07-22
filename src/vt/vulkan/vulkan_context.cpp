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

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "vulkan_loader.h"
#include "vulkan_spirv.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vt::vulkan {
namespace {

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
    best.physical_device = devices[i];
    best.queue_family = static_cast<uint32_t>(qf);
    best.api_version = props.apiVersion;
    std::memcpy(best.name, props.deviceName, sizeof(best.name));
  }
  return best;
}

}  // namespace

// ---------------------------------------------------------------------------

struct VulkanContext::Pipeline {
  VkShaderModule module = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  uint32_t buffer_count = 0;
  uint32_t push_size = 0;
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
  VT_CHECK(props2.properties.limits.maxComputeWorkGroupInvocations >= kWorkgroupSize,
           "vulkan: device reports maxComputeWorkGroupInvocations below the Vulkan "
           "guaranteed minimum of 128, which the committed SPIR-V is compiled for");

  // storageBuffer16BitAccess must be ENABLED, not merely supported.
  VkPhysicalDevice16BitStorageFeatures f16{};
  f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  f16.storageBuffer16BitAccess = VK_TRUE;

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = queue_family_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pNext = &f16;
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

  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  cpci.queueFamilyIndex = queue_family_;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  Check(Api().vkCreateCommandPool(device, &cpci, nullptr, &command_pool), "vkCreateCommandPool");
  command_pool_ = Pack(command_pool);

  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = command_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  Check(Api().vkAllocateCommandBuffers(device, &cbai, &cmd), "vkAllocateCommandBuffers");
  command_buffer_ = Pack(cmd);

  // One descriptor set per kernel, each holding at most kMaxBindings storage
  // buffers. Sized for the whole committed shader table so the pool is never
  // reallocated. (llama.cpp grows a vector of pools instead; a fixed pool is
  // enough here because dispatch is synchronous and sets are re-updated.)
  constexpr uint32_t kMaxBindings = 12;  // llama.cpp's MAX_PARAMETER_COUNT
  const uint32_t kernels =
      static_cast<uint32_t>(sizeof(kSpirvModules) / sizeof(kSpirvModules[0]));
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = kernels * kMaxBindings;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = kernels;
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
  // A zero-length VkBuffer is invalid; round up so a 0-byte request still yields
  // a distinct freeable pointer (the CPU backend's contract, which vt::StepArena
  // relies on). Same treatment as the Metal skeleton.
  const VkDeviceSize len = bytes == 0 ? 1 : static_cast<VkDeviceSize>(bytes);

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
  vk.vkUnmapMemory(device, Unpack<VkDeviceMemory>(memory));
  vk.vkDestroyBuffer(device, Unpack<VkBuffer>(buffer), nullptr);
  vk.vkFreeMemory(device, Unpack<VkDeviceMemory>(memory), nullptr);
}

VulkanContext::Pipeline& VulkanContext::GetPipeline(const std::string& name,
                                                    uint32_t buffer_count, uint32_t push_size) {
  auto& cache = *static_cast<std::map<std::string, Pipeline>*>(pipelines_);
  auto it = cache.find(name);
  if (it != cache.end()) {
    // A kernel's binding count and push-constant size are properties of its
    // SPIR-V; a mismatch means the host and the committed shader have drifted,
    // which would corrupt memory rather than fail cleanly.
    VT_CHECK(it->second.buffer_count == buffer_count && it->second.push_size == push_size,
             "vulkan: pipeline " + name + " re-requested with a different binding layout");
    return it->second;
  }

  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);

  const SpirvModule* module = nullptr;
  for (const SpirvModule& m : kSpirvModules) {
    if (name == m.name) { module = &m; break; }
  }
  VT_CHECK(module != nullptr,
           "vulkan: no committed SPIR-V for kernel '" + name +
               "' — regenerate with scripts/gen-vulkan-spirv.py");

  Pipeline p;
  p.buffer_count = buffer_count;
  p.push_size = push_size;

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

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = p.module;
  cpci.stage.pName = "main";
  cpci.layout = p.layout;
  Check(vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline),
        "vkCreateComputePipelines");

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = Unpack<VkDescriptorPool>(descriptor_pool_);
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &p.set_layout;
  Check(vk.vkAllocateDescriptorSets(device, &dsai, &p.set), "vkAllocateDescriptorSets");

  return cache.emplace(name, p).first->second;
}

void VulkanContext::Dispatch(const std::string& name, const void* const* buffers,
                             uint32_t buffer_count, const void* push_constants,
                             uint32_t push_size, uint32_t group_count_x) {
  if (group_count_x == 0) return;  // nothing to do; an empty dispatch is illegal
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

  Pipeline& p = GetPipeline(name, buffer_count, push_size);

  std::vector<VkDescriptorBufferInfo> infos(buffer_count);
  std::vector<VkWriteDescriptorSet> writes(buffer_count);
  for (uint32_t i = 0; i < buffer_count; ++i) {
    infos[i].buffer = Unpack<VkBuffer>(const_cast<void*>(buffers[i]));
    infos[i].offset = 0;  // always WHOLE; the element offset rides push constants
    infos[i].range = VK_WHOLE_SIZE;
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = p.set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vk.vkUpdateDescriptorSets(device, buffer_count, writes.data(), 0, nullptr);

  auto cmd = Unpack<VkCommandBuffer>(command_buffer_);
  Check(vk.vkResetCommandPool(device, Unpack<VkCommandPool>(command_pool_), 0),
        "vkResetCommandPool");
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  Check(vk.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
  vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0,
                             nullptr);
  vk.vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                        push_constants);
  vk.vkCmdDispatch(cmd, group_count_x, 1, 1);
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
  Check(vk.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

}  // namespace vt::vulkan
