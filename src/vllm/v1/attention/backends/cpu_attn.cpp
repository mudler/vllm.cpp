// Ported from: vllm/v1/attention/backends/cpu_attn.py @ pin 5559679229
// (CPUAttentionBackend). See the header for the three recorded deviations and
// for why registering this name is the fix for issue #1371.
#include "vllm/v1/attention/backends/cpu_attn.h"

#include <memory>

#include "vllm/v1/attention/registry.h"

namespace vllm::v1 {

// DEVIATION 1 (see cpu_attn.h): upstream cpu_attn.py:94-104 returns the HND
// shape (num_blocks, num_kv_heads, block_size, 2 * head_size). Our CPU
// paged-attention kernel reads the NHD layout by tensor strides
// (src/vt/cpu/cpu_paged_attn.cpp:5-8), and that is the layout the engine
// allocates, so this reports what is really there. Identical, dimension for
// dimension, to FlashAttentionBackend::get_kv_cache_shape — which is precisely
// why CPU could fall through to FLASH_ATTN for as long as it did.
std::vector<int64_t> CpuAttentionBackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& /*cache_dtype_str*/) const {
  return {num_blocks, 2, block_size, num_kv_heads, head_size};
}

namespace {
// CPU_ATTN self-registers for kCPU alone. Upstream registers it the same way
// (@register_backend(AttentionBackendEnum.CPU_ATTN), registry.py) and
// cpu.py:75-87 returns it for every CPU request, so with this line
// SelectAttentionBackendName finally gives upstream's answer on kCPU instead of
// walking past an unregistered name. Retained past the linker by the vllm
// --whole-archive INTERFACE option, like every other backend registrar.
const AttentionBackendRegistrar kCpuAttnCpu{
    vt::DeviceType::kCPU, CpuAttentionBackend::kName,
    []() -> std::unique_ptr<AttentionBackend> {
      return std::make_unique<CpuAttentionBackend>();
    }};
}  // namespace

}  // namespace vllm::v1
