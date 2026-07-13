// vllm.cpp — persistent NVFP4 tactic-cache document and FlashInfer importer.
//
// Mirrors vLLM v0.25.0
// `vllm/model_executor/warmup/flashinfer_autotune_cache.py` and installed
// FlashInfer 0.6.13 `autotuner.py` load/save/metadata semantics. This layer is
// deliberately CUDA-free: runtime plan publication is adapted separately.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "vt/cuda/nvfp4_plan_cache.h"

namespace vt::cuda::nvfp4 {

inline constexpr std::string_view kPersistentCacheFormat =
    "vllm.cpp_nvfp4_autotune_v1";
inline constexpr std::string_view kPersistentCacheRunner =
    "CutlassFp4GemmRunner";
inline constexpr uint32_t kHybridBucketVersion = 1;
inline constexpr uint32_t kFlashInferWarmups = 3;
inline constexpr uint32_t kFlashInferRepeats = 10;
inline constexpr uint32_t kFlashInferDelayMicroseconds = 5000;

struct PersistentCacheMetadata {
  std::string format = std::string(kPersistentCacheFormat);
  std::string cuda_runtime;
  std::string cuda_driver;
  std::string cutlass_version;
  std::string gpu;
  int32_t device_ordinal = 0;
  int32_t architecture = 0;
  std::string output_dtype = "bf16";
  uint8_t output_dtype_id = 0;
  std::string fp4_layout = "e2m1_packed_k2";
  std::string scale_layout = "cutlass_8x4";
  uint32_t tactic_set_version = kFullTacticSetVersion;
  std::string tactic_descriptor_digest;
  uint32_t warmups = kFlashInferWarmups;
  uint32_t repeats = kFlashInferRepeats;
  uint32_t delay_microseconds = kFlashInferDelayMicroseconds;
  uint32_t hybrid_bucket_version = kHybridBucketVersion;
  std::string build_id;

  bool operator==(const PersistentCacheMetadata&) const = default;
};

struct PersistentPlan {
  PlanKey key;
  std::string fp4_layout = "e2m1_packed_k2";
  std::string scale_layout = "cutlass_8x4";
  std::string runner = std::string(kPersistentCacheRunner);
  int tactic_id = -1;

  bool operator==(const PersistentPlan&) const = default;
};

struct PersistentCacheDocument {
  PersistentCacheMetadata metadata;
  std::vector<PersistentPlan> plans;

  bool operator==(const PersistentCacheDocument&) const = default;
};

struct FlashInferCacheMetadata {
  std::string flashinfer_version;
  std::string cuda_version;
  std::string cublas_version;
  std::string cudnn_version;
  std::string cudnn_frontend_version;
  std::string gpu;
};

struct FlashInferImportTarget {
  FlashInferCacheMetadata expected_metadata;
  int32_t device_ordinal = 0;
  int32_t architecture = 0;
  uint8_t output_dtype = 0;
  uint32_t tactic_set_version = kFullTacticSetVersion;
  std::string fp4_layout = "e2m1_packed_k2";
  std::string scale_layout = "cutlass_8x4";
};

struct FlashInferImportResult {
  std::vector<PersistentPlan> plans;
  size_t ignored_foreign_entries = 0;
};

struct PersistentCacheOptions {
  bool enabled = true;
  bool read_only = false;
  uint32_t delay_microseconds = kFlashInferDelayMicroseconds;
  std::filesystem::path native_path;
  std::filesystem::path flashinfer_path;
  std::string fallback_reason;
};

// Stable digest of the complete 0--31 local tactic ABI.
std::string Nvfp4TacticDescriptorDigest();

// Stable non-secret identity used by startup diagnostics. This fingerprints
// every compatibility field, not the cache path or its plan contents.
std::string PersistentCacheMetadataFingerprint(
    const PersistentCacheMetadata& metadata);

// Resolve the default cache namespace and the W3-C environment overrides.
// Environment values are read on every call so isolated tests can exercise the
// configuration contract in one process.
PersistentCacheOptions ResolvePersistentCacheOptions(
    const PersistentCacheMetadata& metadata);

std::string SerializeNativeCache(const PersistentCacheDocument& document);
PersistentCacheDocument ParseNativeCache(
    std::string_view contents, const PersistentCacheMetadata& expected_metadata);
PersistentCacheDocument LoadNativeCache(
    const std::filesystem::path& path,
    const PersistentCacheMetadata& expected_metadata);

// Compatible prior entries are retained and `current` wins on an identical
// complete plan key. The returned document is deterministically sorted.
PersistentCacheDocument MergeNativeCaches(
    const PersistentCacheDocument& prior,
    const PersistentCacheDocument& current);

// Same-directory temporary + atomic rename. A failed write removes its temp
// file and never truncates the destination.
void WriteNativeCacheAtomically(const std::filesystem::path& path,
                                const PersistentCacheDocument& document);

FlashInferImportResult ParseFlashInferCache(
    std::string_view contents, const FlashInferImportTarget& target);
FlashInferImportResult LoadFlashInferCache(
    const std::filesystem::path& path,
    const FlashInferImportTarget& target);

}  // namespace vt::cuda::nvfp4
