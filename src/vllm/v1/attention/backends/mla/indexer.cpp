// Ported from: vllm/v1/attention/backends/mla/indexer.py @ pin e126687a9a.
#include "vllm/v1/attention/backends/mla/indexer.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "vllm/v1/attention/registry.h"

namespace vllm::v1 {

std::vector<int64_t> DeepseekV4IndexerBackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& /*cache_dtype_str*/) const {
  if (!supports_block_size(static_cast<int>(block_size))) {
    throw std::invalid_argument(
        "DEEPSEEK_V4_INDEXER: block size must be 256 (got " +
        std::to_string(block_size) + ").");
  }
  if (num_kv_heads != 1) {
    throw std::invalid_argument(
        "DEEPSEEK_V4_INDEXER: the indexer key cache holds one packed vector "
        "per token, so num_kv_heads must be 1.");
  }
  return {num_blocks, block_size, head_size};
}

namespace {
// Registered for every device type, in no priority list — see the equivalent
// block in sparse_swa.cpp for why both halves of that are deliberate.
AttentionBackendFactory MakeIndexerBackend = []() -> std::unique_ptr<AttentionBackend> {
  return std::make_unique<DeepseekV4IndexerBackend>();
};
const AttentionBackendRegistrar kIndexerCuda{vt::DeviceType::kCUDA,
                                             DeepseekV4IndexerBackend::kName,
                                             MakeIndexerBackend};
const AttentionBackendRegistrar kIndexerCpu{vt::DeviceType::kCPU,
                                            DeepseekV4IndexerBackend::kName,
                                            MakeIndexerBackend};
const AttentionBackendRegistrar kIndexerRocm{vt::DeviceType::kROCM,
                                             DeepseekV4IndexerBackend::kName,
                                             MakeIndexerBackend};
const AttentionBackendRegistrar kIndexerMetal{vt::DeviceType::kMETAL,
                                              DeepseekV4IndexerBackend::kName,
                                              MakeIndexerBackend};
const AttentionBackendRegistrar kIndexerVulkan{vt::DeviceType::kVULKAN,
                                               DeepseekV4IndexerBackend::kName,
                                               MakeIndexerBackend};
const AttentionBackendRegistrar kIndexerTenstorrent{
    vt::DeviceType::kTENSTORRENT, DeepseekV4IndexerBackend::kName,
    MakeIndexerBackend};
}  // namespace

}  // namespace vllm::v1
