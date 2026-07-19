// Shared Qwen3.5/3.6 registry-glue helper definitions (see qwen3_5_common.h).
// Extracted verbatim (behavior-preserving) from the former model_registry.cpp
// monolith so the dense and MoE variant TUs share one copy.
#include "vllm/model_executor/models/qwen3_5_common.h"

#include <cstdlib>

#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_internal.h"  // ResolveMambaSsmCacheDType
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {

void ParseQwen3_5Config(const HfConfig& config) {
  // LoadHfConfig/HfConfigFromGguf already materialize the consumed Qwen fields.
  // This explicit per-family hook is where a family adds normalization or
  // validation without changing the registry/runner contract.
  (void)config;
}

ForwardLogits HostLogits(std::vector<float>&& host, int64_t vocab) {
  ForwardLogits logits;
  logits.vocab = vocab;
  logits.rows = vocab > 0 ? static_cast<int64_t>(host.size()) / vocab : 0;
  logits.host = std::move(host);
  return logits;
}

v1::KVCacheConfig MakeQwen3_5KVCache(const HfConfig& config, int block_size,
                                     int num_blocks) {
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);
  const int num_value_heads = static_cast<int>(config.linear_num_value_heads);
  const int value_head_dim = static_cast<int>(config.linear_value_head_dim);
  const int key_head_dim = static_cast<int>(config.linear_key_head_dim);
  const int conv_kernel = static_cast<int>(config.linear_conv_kernel_dim);
  const int key_dim =
      static_cast<int>(config.linear_num_key_heads) * key_head_dim;
  const int value_dim = num_value_heads * value_head_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  // Diagnostic state-storage overrides belong to planning, not allocation:
  // the MambaSpec must describe the exact bytes the runner will consume.
  vt::DType conv_dtype = vt::DType::kBF16;
  vt::DType ssm_dtype =
      detail::ResolveMambaSsmCacheDType(config, conv_dtype);
  if (const char* state_dtype = std::getenv("VT_GDN_STATE_BF16")) {
    if (state_dtype[0] == '0') {
      conv_dtype = vt::DType::kF32;
      ssm_dtype = vt::DType::kF32;
    } else if (state_dtype[0] == '1') {
      conv_dtype = vt::DType::kBF16;
      ssm_dtype = vt::DType::kBF16;
    }
  }

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, num_kv_heads,
                                               head_dim, vt::DType::kF32));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn"},
      std::make_shared<v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{
              {conv_dim, conv_kernel - 1},
              {num_value_heads, value_head_dim, key_head_dim}},
          std::vector<vt::DType>{conv_dtype, ssm_dtype}));
  return kv;
}

}  // namespace vllm
