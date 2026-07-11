// Ported from: vllm/v1/kv_cache_spec_registry.py @ e24d1b24
#include "vllm/v1/kv_cache_spec_registry.h"

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace vllm::v1 {
namespace {

using Registry =
    std::unordered_map<std::type_index, KVCacheSpecMetadata>;

Registry& registry() {
  static Registry value;
  return value;
}

std::mutex& registry_mutex() {
  static std::mutex value;
  return value;
}

std::once_flag& registration_once() {
  static std::once_flag value;
  return value;
}

std::type_index built_in_type_for_kind(KVCacheSpecKind kind) {
  switch (kind) {
    case KVCacheSpecKind::kFullAttention:
      return typeid(FullAttentionSpec);
    case KVCacheSpecKind::kSlidingWindow:
      return typeid(SlidingWindowSpec);
    case KVCacheSpecKind::kMamba:
      return typeid(MambaSpec);
    default:
      return typeid(void);
  }
}

}  // namespace

void KVCacheSpecRegistry::register_spec_type(
    std::type_index spec_type, KVCacheManagerKind manager_kind,
    std::type_index uniform_type_base_spec) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto& entries = registry();
  auto it = entries.find(spec_type);
  if (it != entries.end()) {
    if (it->second.manager_kind != manager_kind ||
        it->second.uniform_type_base_spec != uniform_type_base_spec) {
      throw std::invalid_argument(
          "Conflicting registration for KVCacheSpec type");
    }
    return;
  }
  entries.emplace(spec_type,
                  KVCacheSpecMetadata{spec_type, manager_kind,
                                      uniform_type_base_spec});
}

void KVCacheSpecRegistry::ensure_registered() {
  std::call_once(registration_once(), [] {
    register_spec<FullAttentionSpec, FullAttentionSpec>(
        KVCacheManagerKind::kFullAttention);
    register_spec<SlidingWindowSpec, SlidingWindowSpec>(
        KVCacheManagerKind::kSlidingWindow);
    register_spec<MambaSpec, MambaSpec>(KVCacheManagerKind::kMamba);
  });
}

std::optional<KVCacheSpecMetadata> KVCacheSpecRegistry::get_metadata(
    const KVCacheSpec& kv_cache_spec) {
  ensure_registered();
  std::lock_guard<std::mutex> lock(registry_mutex());
  const auto& entries = registry();

  auto exact = entries.find(typeid(kv_cache_spec));
  if (exact != entries.end()) {
    return exact->second;
  }

  // C++ has no runtime MRO. Inherited kind() is the stable built-in-base
  // identity for an unregistered subclass and gives the same lookup result.
  const std::type_index base_type = built_in_type_for_kind(kv_cache_spec.kind());
  auto base = entries.find(base_type);
  if (base != entries.end()) {
    return base->second;
  }
  return std::nullopt;
}

std::optional<KVCacheManagerKind> KVCacheSpecRegistry::get_manager_kind(
    const KVCacheSpec& kv_cache_spec) {
  auto metadata = get_metadata(kv_cache_spec);
  if (!metadata.has_value()) {
    return std::nullopt;
  }
  return metadata->manager_kind;
}

std::optional<std::type_index>
KVCacheSpecRegistry::get_uniform_type_base_spec(
    const KVCacheSpec& kv_cache_spec) {
  auto metadata = get_metadata(kv_cache_spec);
  if (!metadata.has_value()) {
    return std::nullopt;
  }
  return metadata->uniform_type_base_spec;
}

void KVCacheSpecRegistry::check_kv_cache_spec_registry(
    const std::vector<std::pair<std::string, const KVCacheSpec*>>& specs) {
  for (const auto& [layer_name, spec] : specs) {
    if (spec == nullptr || !get_uniform_type_base_spec(*spec).has_value()) {
      throw std::invalid_argument(
          "Unsupported KV cache spec type for layer " + layer_name);
    }
    if (!get_manager_kind(*spec).has_value()) {
      throw std::invalid_argument(
          "No manager found for KV cache spec type for layer " + layer_name);
    }
  }
}

bool are_uniform_kv_cache_specs(const std::vector<const KVCacheSpec*>& specs) {
  if (specs.empty()) {
    return true;
  }
  if (specs.front() == nullptr) {
    return false;
  }
  const int block_size = specs.front()->block_size;
  auto base =
      KVCacheSpecRegistry::get_uniform_type_base_spec(*specs.front());
  if (!base.has_value()) {
    return false;
  }

  for (const KVCacheSpec* spec : specs) {
    if (spec == nullptr || spec->block_size != block_size ||
        KVCacheSpecRegistry::get_uniform_type_base_spec(*spec) != base) {
      return false;
    }
  }

  if (*base == typeid(SlidingWindowSpec)) {
    const auto* first =
        dynamic_cast<const SlidingWindowSpec*>(specs.front());
    if (first == nullptr) {
      return false;
    }
    for (const KVCacheSpec* spec : specs) {
      const auto* sliding = dynamic_cast<const SlidingWindowSpec*>(spec);
      if (sliding == nullptr ||
          sliding->sliding_window != first->sliding_window) {
        return false;
      }
    }
  } else if (*base == typeid(MambaSpec)) {
    const auto* first = dynamic_cast<const MambaSpec*>(specs.front());
    if (first == nullptr) {
      return false;
    }
    for (const KVCacheSpec* spec : specs) {
      const auto* mamba = dynamic_cast<const MambaSpec*>(spec);
      if (mamba == nullptr || mamba->num_speculative_blocks !=
                                   first->num_speculative_blocks) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace vllm::v1
