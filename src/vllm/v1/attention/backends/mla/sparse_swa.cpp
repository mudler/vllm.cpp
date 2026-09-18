// Ported from: vllm/v1/attention/backends/mla/sparse_swa.py @ pin e126687a9a.
// See the header for what this backend is for and why it has no impl.
#include "vllm/v1/attention/backends/mla/sparse_swa.h"

#include <stdexcept>
#include <string>

#include "vllm/v1/attention/registry.h"

namespace vllm::v1 {

std::vector<int64_t> DeepseekSparseSWABackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& /*cache_dtype_str*/) const {
  // The refusal is against THIS backend's own declared list, which is the whole
  // point of the per-group dispatch: a 4-token compressor group reaching here
  // would be a routing defect and must be loud, and a dense backend's 16-rule
  // must never be what answers for a 64-token SWA cache.
  if (!supports_block_size(static_cast<int>(block_size))) {
    throw std::invalid_argument(
        "DEEPSEEK_SPARSE_SWA: block size must be a multiple of 64 (got " +
        std::to_string(block_size) + ").");
  }
  // sparse_swa.py:101-113 — `[B, H=1, N, C] -> [B, N, C]`. One head, exactly as
  // TritonMLABackend asserts for the latent cache.
  if (num_kv_heads != 1) {
    throw std::invalid_argument(
        "DEEPSEEK_SPARSE_SWA: the sliding-window cache holds one vector per "
        "token, so num_kv_heads must be 1.");
  }
  return {num_blocks, block_size, head_size};
}

namespace {
// Registered for EVERY device type, and that breadth is deliberate rather than
// lazy. Not one line of this backend is device-specific: it declares block
// sizes, head sizes and a page shape, and the bytes it describes are produced
// and consumed by the DeepSeek-V4 DSA ops on whatever device the model runs.
// A missing registration on the device a checkpoint is served on would turn a
// named group into a hard `MakeAttentionBackend` throw at engine construction,
// which is the failure this row exists to remove.
//
// It is in NO platform priority list, so no capability walk can land on it.
// That is upstream's shape exactly: the layer names it (`sparse_swa.py:116-118`)
// and the selector never sees it.
AttentionBackendFactory MakeSparseSwaBackend = []() -> std::unique_ptr<AttentionBackend> {
  return std::make_unique<DeepseekSparseSWABackend>();
};
const AttentionBackendRegistrar kSwaCuda{vt::DeviceType::kCUDA,
                                         DeepseekSparseSWABackend::kName,
                                         MakeSparseSwaBackend};
const AttentionBackendRegistrar kSwaCpu{vt::DeviceType::kCPU,
                                        DeepseekSparseSWABackend::kName,
                                        MakeSparseSwaBackend};
const AttentionBackendRegistrar kSwaRocm{vt::DeviceType::kROCM,
                                         DeepseekSparseSWABackend::kName,
                                         MakeSparseSwaBackend};
const AttentionBackendRegistrar kSwaMetal{vt::DeviceType::kMETAL,
                                          DeepseekSparseSWABackend::kName,
                                          MakeSparseSwaBackend};
const AttentionBackendRegistrar kSwaVulkan{vt::DeviceType::kVULKAN,
                                           DeepseekSparseSWABackend::kName,
                                           MakeSparseSwaBackend};
const AttentionBackendRegistrar kSwaTenstorrent{vt::DeviceType::kTENSTORRENT,
                                                DeepseekSparseSWABackend::kName,
                                                MakeSparseSwaBackend};
}  // namespace

}  // namespace vllm::v1
