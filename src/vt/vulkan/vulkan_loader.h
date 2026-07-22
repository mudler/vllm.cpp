// Vulkan backend — runtime loader for the Vulkan entry points (BACKEND-VULKAN,
// W0 skeleton). vllm.cpp original: vLLM has no Vulkan path, so the design is
// taken from llama.cpp's Vulkan backend (`ggml/src/ggml-vulkan/` @ 237ad9b96),
// which likewise resolves every entry point through a dynamic dispatcher rather
// than link-time symbols (it uses vulkan-hpp's
// `VULKAN_HPP_DEFAULT_DISPATCHER.init(...)`, ggml-vulkan.cpp
// `ggml_vk_instance_init`).
//
// WHY dlopen INSTEAD OF LINKING. Both of our boxes ship the Vulkan LOADER
// (`libvulkan.so.1` — 1.4.328 on dgx.casa, 1.3.275 on the dev box) but NEITHER
// ships the development package: there is no `libvulkan.so` link name for `-l`
// to find, no `/usr/include/vulkan`, and no sudo to install either (measured
// 2026-07-22). Resolving `libvulkan.so.1` with dlopen therefore:
//   * builds and links with no Vulkan SDK present at all;
//   * leaves the vllm library loadable on a box with no Vulkan runtime — a
//     missing loader becomes "kVULKAN is not registered", the state
//     src/vt/ops.cpp:104-111 already treats as supported, instead of a hard
//     ld.so failure at process start;
//   * keeps the Vulkan API surface out of every other translation unit.
// The API TYPES still come from the vendored Khronos headers
// (third_party/vulkan/, pinned at vulkan-sdk-1.4.328.1) compiled with
// VK_NO_PROTOTYPES, so only the function POINTERS are resolved dynamically —
// hand-declaring the structs would risk a silent ABI mismatch.
#ifndef VT_VULKAN_VULKAN_LOADER_H_
#define VT_VULKAN_VULKAN_LOADER_H_

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

namespace vt::vulkan {

// The entry points this backend uses, in the three resolution TIERS Vulkan
// defines. Tier order matters: global entry points come from the loader itself,
// instance entry points need a VkInstance, device entry points need a VkDevice
// (and going through vkGetDeviceProcAddr skips the loader's dispatch trampoline).
#define VT_VK_GLOBAL_FUNCS(X) \
  X(vkCreateInstance)         \
  X(vkEnumerateInstanceVersion)

#define VT_VK_INSTANCE_FUNCS(X)              \
  X(vkDestroyInstance)                       \
  X(vkEnumeratePhysicalDevices)              \
  X(vkGetPhysicalDeviceProperties)           \
  X(vkGetPhysicalDeviceMemoryProperties)     \
  X(vkGetPhysicalDeviceQueueFamilyProperties)\
  X(vkGetPhysicalDeviceFeatures2)            \
  X(vkGetPhysicalDeviceProperties2)          \
  X(vkEnumerateDeviceExtensionProperties)    \
  X(vkCreateDevice)                          \
  X(vkGetDeviceProcAddr)

#define VT_VK_DEVICE_FUNCS(X)          \
  X(vkDestroyDevice)                   \
  X(vkDeviceWaitIdle)                  \
  X(vkGetDeviceQueue)                  \
  X(vkCreateBuffer)                    \
  X(vkDestroyBuffer)                   \
  X(vkGetBufferMemoryRequirements)     \
  X(vkBindBufferMemory)                \
  X(vkAllocateMemory)                  \
  X(vkFreeMemory)                      \
  X(vkMapMemory)                       \
  X(vkUnmapMemory)                     \
  X(vkCreateShaderModule)              \
  X(vkDestroyShaderModule)             \
  X(vkCreateDescriptorSetLayout)       \
  X(vkDestroyDescriptorSetLayout)      \
  X(vkCreatePipelineLayout)            \
  X(vkDestroyPipelineLayout)           \
  X(vkCreateComputePipelines)          \
  X(vkDestroyPipeline)                 \
  X(vkCreateDescriptorPool)            \
  X(vkDestroyDescriptorPool)           \
  X(vkAllocateDescriptorSets)          \
  X(vkUpdateDescriptorSets)            \
  X(vkCreateCommandPool)               \
  X(vkDestroyCommandPool)              \
  X(vkAllocateCommandBuffers)          \
  X(vkResetCommandPool)                \
  X(vkBeginCommandBuffer)              \
  X(vkEndCommandBuffer)                \
  X(vkCmdBindPipeline)                 \
  X(vkCmdBindDescriptorSets)           \
  X(vkCmdPushConstants)                \
  X(vkCmdDispatch)                     \
  X(vkQueueSubmit)                     \
  X(vkQueueWaitIdle)                   \
  X(vkCreateFence)                     \
  X(vkDestroyFence)                    \
  X(vkWaitForFences)                   \
  X(vkResetFences)

// The resolved pointers. A single process-wide instance; every member is null
// until the matching Load* call succeeds.
struct VulkanApi {
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define VT_VK_DECL(name) PFN_##name name = nullptr;
  VT_VK_GLOBAL_FUNCS(VT_VK_DECL)
  VT_VK_INSTANCE_FUNCS(VT_VK_DECL)
  VT_VK_DEVICE_FUNCS(VT_VK_DECL)
#undef VT_VK_DECL
};

// Opens libvulkan.so.1 (or the platform equivalent) and resolves the global
// entry points. Returns false — without throwing — when there is no loader on
// the machine; the registrars treat that as "do not register kVULKAN".
// Idempotent and thread-safe; the library is never dlclose'd (the process
// outlives it, matching the singleton lifetimes elsewhere in this backend).
bool LoadVulkanLibrary();

// The resolved table. Only meaningful after LoadVulkanLibrary() returned true.
const VulkanApi& Api();

// Resolve the instance-level and then the device-level entry points. Both
// VT_CHECK on a missing symbol: by that point the loader has already told us the
// implementation claims the version that defines them, so a null is a broken
// driver, not an absent one.
void LoadInstanceFunctions(VkInstance instance);
void LoadDeviceFunctions(VkDevice device);

}  // namespace vt::vulkan

#endif  // VT_VULKAN_VULKAN_LOADER_H_
