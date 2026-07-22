// Vulkan backend — shared instance/device/queue/pipeline context
// (BACKEND-VULKAN, W0 skeleton). vllm.cpp original (vt runtime, inventory
// deviation §9.1): vLLM has no Vulkan platform anywhere, so the DESIGN is ported
// from llama.cpp's Vulkan backend (`ggml/src/ggml-vulkan/ggml-vulkan.cpp` @ pin
// 237ad9b96). Specifically:
//
//   * one process-wide VkInstance + VkPhysicalDevice + VkDevice + compute
//     VkQueue, created lazily and kept for the process — llama.cpp
//     `ggml_vk_instance_init` / `ggml_vk_device_init` and its `vk_instance`
//     singleton;
//   * host-visible storage buffers whose memory type is chosen by walking
//     VkPhysicalDeviceMemoryProperties for the required property flags with an
//     ordered fallback list — llama.cpp `ggml_vk_find_memory_properties`
//     (ggml-vulkan.cpp:2957) and `ggml_vk_create_buffer` (:2971-3100);
//   * a NAME -> compute-pipeline cache so each kernel is specialized once —
//     llama.cpp `ggml_vk_create_pipeline_func` (:2460-2560) and its per-device
//     pipeline map;
//   * push constants for the small per-dispatch parameter block — llama.cpp
//     `ggml_vk_dispatch_pipeline` (:7507-7530).
//
// This header is deliberately PLAIN C++ — it does NOT include vulkan_core.h — so
// the op TU, the platform TU and the tests can include it without pulling the
// Vulkan API into their translation units. Handles cross the boundary as void*,
// exactly as the Metal skeleton does for its ObjC types
// (src/vt/metal/metal_context.h). vulkan_context.cpp static_asserts that the
// real handle types fit.
#ifndef VT_VULKAN_VULKAN_CONTEXT_H_
#define VT_VULKAN_VULKAN_CONTEXT_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace vt::vulkan {

// Process-wide Vulkan context. Created on first use, never destroyed (the
// process outlives it; matching llama.cpp's `vk_instance` singleton lifetime and
// the Metal skeleton's MetalContext).
class VulkanContext {
 public:
  // Returns the singleton, creating instance/device/queue and the descriptor and
  // command pools on first call. Throws (VT_CHECK) if anything fails — by the
  // time this is called, Available() has already said a usable device exists, so
  // a failure here is a broken driver, not an absent one.
  static VulkanContext& Get();

  // True iff a Vulkan loader is present AND it enumerates a physical device that
  // satisfies this backend's requirements (Vulkan >= 1.1, a compute queue
  // family, VK_KHR_16bit_storage's storageBuffer16BitAccess, and a
  // HOST_VISIBLE|HOST_COHERENT memory type usable for storage buffers). Safe on
  // a machine with no GPU and no loader: it does NOT throw. This is the
  // predicate both registrars use, so a Vulkan-enabled BUILD on a machine
  // without Vulkan simply does not register kVULKAN rather than aborting during
  // static initialization.
  static bool Available();

  // Dispatch one compute kernel, SYNCHRONOUSLY (record, submit, wait). `name` is
  // a key in the committed SPIR-V table (src/vt/vulkan/vulkan_spirv.h).
  // `buffers` are the VkBuffer handles for descriptor bindings 0..n-1, in order;
  // this backend binds every buffer WHOLE (offset 0, VK_WHOLE_SIZE) and carries
  // the element offset in the push constants, so any interior tensor pointer
  // works regardless of minStorageBufferOffsetAlignment (see
  // src/vt/vulkan/shaders/vt_common.glsl § STORAGE MODEL).
  // Serialized by an internal mutex: the command buffer and each pipeline's
  // descriptor set are single instances that are re-recorded per dispatch.
  void Dispatch(const std::string& name, const void* const* buffers, uint32_t buffer_count,
                const void* push_constants, uint32_t push_size, uint32_t group_count_x);

  // Allocate one host-visible, host-coherent storage buffer of `bytes` and keep
  // it PERSISTENTLY MAPPED. Returns the mapped host pointer — which is what
  // vt::Tensor::data carries — and hands back the VkBuffer / VkDeviceMemory
  // handles for the allocation registry. Persistent mapping is what makes
  // Backend::Copy/Memset plain memcpy/memset and keeps them BIT-EXACT.
  void* AllocBuffer(size_t bytes, void** out_buffer, void** out_memory);
  void FreeBuffer(void* buffer, void* memory);

  // A small device-visible scratch buffer for per-dispatch data that is too big
  // for push constants (the fused-chain recipe step list). Returns the VkBuffer
  // handle; `Data()` is its persistently mapped host pointer. Reused across
  // dispatches, which is safe because dispatch is synchronous.
  void* ScratchBuffer() const { return scratch_buffer_; }
  void* ScratchData() const { return scratch_mapped_; }
  static constexpr size_t kScratchBytes = 1024;

  // --- Capability data mirrored onto the Platform seam (src/vllm/platforms/
  // vulkan.cpp) and onto vt::Backend.
  // The VULKAN API VERSION is what we expose as the DeviceCapability
  // major/minor pair — {1, 4} on GB10 (API 1.4.312). CUDA answers this question
  // with sm_XY and Metal with the Apple GPU family; the Vulkan analogue is the
  // API level, so has_device_capability(1, 1) reads as "Vulkan >= 1.1", the same
  // shape of question the CUDA code already asks.
  // The shared VkQueue, as the opaque handle vt::Queue carries.
  void* queue_handle() const { return queue_; }

  int api_major() const { return api_major_; }
  int api_minor() const { return api_minor_; }
  bool unified_memory() const { return unified_memory_; }
  const std::string& device_name() const { return device_name_; }
  uint32_t max_workgroup_count_x() const { return max_workgroup_count_x_; }
  // The two float-controls properties that decide whether our f32 arithmetic is
  // IEEE as written. Probed, recorded, and asserted by the unit gate; see
  // vulkan_context.cpp § RELAXED PRECISION.
  bool denorm_preserve_f32() const { return denorm_preserve_f32_; }
  bool signed_zero_inf_nan_preserve_f32() const { return sz_inf_nan_preserve_f32_; }

 private:
  VulkanContext();
  struct Pipeline;
  Pipeline& GetPipeline(const std::string& name, uint32_t buffer_count, uint32_t push_size);

  void* instance_ = nullptr;         // VkInstance
  void* physical_device_ = nullptr;  // VkPhysicalDevice
  void* device_ = nullptr;           // VkDevice
  void* queue_ = nullptr;            // VkQueue
  void* command_pool_ = nullptr;     // VkCommandPool
  void* command_buffer_ = nullptr;   // VkCommandBuffer
  void* descriptor_pool_ = nullptr;  // VkDescriptorPool
  void* fence_ = nullptr;            // VkFence
  void* scratch_buffer_ = nullptr;   // VkBuffer
  void* scratch_memory_ = nullptr;   // VkDeviceMemory
  void* scratch_mapped_ = nullptr;   // host pointer
  void* pipelines_ = nullptr;        // std::map<std::string, Pipeline>*
  void* mutex_ = nullptr;            // std::mutex*
  uint32_t queue_family_ = 0;
  uint32_t memory_type_index_ = 0;
  int api_major_ = 0;
  int api_minor_ = 0;
  bool unified_memory_ = false;
  bool denorm_preserve_f32_ = false;
  bool sz_inf_nan_preserve_f32_ = false;
  uint32_t max_workgroup_count_x_ = 0;
  std::string device_name_;

  friend class VulkanAllocator;
};

// Plain-C++ spelling of VulkanContext::Available(), so the engine-side platform
// TU (src/vllm/platforms/vulkan.cpp) can ask "is there a Vulkan device?" without
// depending on static-initialization ORDER — asking "did the backend registrar
// already run?" from another TU's initializer is unspecified-order and would
// intermittently skip platform registration. Same reasoning, same shape, as
// vt::metal::MetalDeviceAvailable().
bool VulkanDeviceAvailable();

// Workgroup size every kernel in this backend is compiled with. Mirrors VT_TG in
// src/vt/vulkan/shaders/vt_common.glsl; the host must agree with the SPIR-V
// because the flat kernels compute their workgroup COUNT from it.
inline constexpr uint32_t kWorkgroupSize = 128;

// Number of workgroups needed to cover `n` elements at kWorkgroupSize threads
// each.
uint32_t FlatGroupCount(int64_t n);

}  // namespace vt::vulkan

#endif  // VT_VULKAN_VULKAN_CONTEXT_H_
