// Ported from: vllm/v1/kv_cache_spec_registry.py @ e24d1b24
//
// C++ cannot return Python manager classes, so registry metadata records a
// stable manager kind consumed by get_manager_for_kv_cache_spec(). Exact C++
// dynamic types are registered with std::type_index. An unregistered derived
// spec falls back through its inherited KVCacheSpecKind, mirroring Python's MRO
// lookup for subclasses of a registered built-in spec.
#ifndef VLLM_V1_KV_CACHE_SPEC_REGISTRY_H_
#define VLLM_V1_KV_CACHE_SPEC_REGISTRY_H_

#include <optional>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include "vllm/v1/kv_cache_interface.h"

namespace vllm::v1 {

enum class KVCacheManagerKind {
  kFullAttention,
  kSlidingWindow,
  kChunkedLocalAttention,
  kMamba,
};

struct KVCacheSpecMetadata {
  std::type_index kvcache_spec_type;
  KVCacheManagerKind manager_kind;
  std::type_index uniform_type_base_spec;
};

class KVCacheSpecRegistry {
 public:
  template <typename Spec, typename UniformBase>
  static void register_spec(KVCacheManagerKind manager_kind) {
    static_assert(std::is_base_of_v<KVCacheSpec, Spec>);
    static_assert(std::is_base_of_v<UniformBase, Spec>);
    register_spec_type(typeid(Spec), manager_kind, typeid(UniformBase));
  }

  static std::optional<KVCacheSpecMetadata> get_metadata(
      const KVCacheSpec& kv_cache_spec);

  static std::optional<KVCacheManagerKind> get_manager_kind(
      const KVCacheSpec& kv_cache_spec);

  static std::optional<std::type_index> get_uniform_type_base_spec(
      const KVCacheSpec& kv_cache_spec);

  // Raises std::invalid_argument for an unregistered/unsupported spec.
  static void check_kv_cache_spec_registry(
      const std::vector<std::pair<std::string, const KVCacheSpec*>>& specs);

 private:
  static void ensure_registered();
  static void register_spec_type(std::type_index spec_type,
                                 KVCacheManagerKind manager_kind,
                                 std::type_index uniform_type_base_spec);
};

// Mirrors UniformTypeKVCacheSpecs.is_uniform_type for the currently ported
// built-ins: common block size and registry base; SWA additionally requires a
// common window, chunked-local a common chunk size, and Mamba a common
// speculative-block count.
bool are_uniform_kv_cache_specs(const std::vector<const KVCacheSpec*>& specs);

}  // namespace vllm::v1

#endif  // VLLM_V1_KV_CACHE_SPEC_REGISTRY_H_
