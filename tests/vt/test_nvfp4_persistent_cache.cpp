// Ports the cache/source contracts from vLLM v0.25.0
// `model_executor/warmup/{kernel_warmup,flashinfer_autotune_cache}.py` and
// installed FlashInfer 0.6.13 `autotuner.py`. Upstream has no direct cache unit
// module; its source lifecycle plus the immutable GB10 cache are the spec.
#include <doctest/doctest.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/cuda/nvfp4_persistent_cache.h"

#ifndef TEST_FIXTURES_DIR
#define TEST_FIXTURES_DIR "tests/fixtures"
#endif

namespace {

namespace fs = std::filesystem;
using vt::cuda::nvfp4::FlashInferCacheMetadata;
using vt::cuda::nvfp4::FlashInferImportTarget;
using vt::cuda::nvfp4::PersistentCacheDocument;
using vt::cuda::nvfp4::PersistentCacheMetadata;
using vt::cuda::nvfp4::PersistentPlan;
using vt::cuda::nvfp4::PlanKey;

class TempDir {
 public:
  TempDir() {
    std::string pattern = (fs::temp_directory_path() / "vllm-cpp-nvfp4-cache-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    char* created = ::mkdtemp(buffer.data());
    if (created == nullptr) throw std::runtime_error("mkdtemp failed");
    path_ = created;
  }

  ~TempDir() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }

  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

class ScopedEnv {
 public:
  ScopedEnv(std::string name, std::optional<std::string> value)
      : name_(std::move(name)) {
    if (const char* old = std::getenv(name_.c_str()); old != nullptr) old_ = old;
    Set(value);
  }

  ~ScopedEnv() { Set(old_); }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  void Set(const std::optional<std::string>& value) const {
    if (value.has_value()) {
      if (::setenv(name_.c_str(), value->c_str(), 1) != 0) std::abort();
    } else if (::unsetenv(name_.c_str()) != 0) {
      std::abort();
    }
  }

  std::string name_;
  std::optional<std::string> old_;
};

PersistentCacheMetadata Metadata() {
  PersistentCacheMetadata metadata;
  metadata.cuda_runtime = "13.0";
  metadata.cuda_driver = "580.65.06";
  metadata.cutlass_version = "4.5.0";
  metadata.gpu = "NVIDIA GB10";
  metadata.device_ordinal = 0;
  metadata.architecture = 121;
  metadata.output_dtype_id = 2;
  metadata.tactic_descriptor_digest =
      vt::cuda::nvfp4::Nvfp4TacticDescriptorDigest();
  metadata.build_id = "test-build";
  return metadata;
}

PersistentPlan Plan(uint32_t m, int32_t n, int32_t k, int tactic) {
  PersistentPlan plan;
  plan.key = PlanKey{m, n, k, 0, 121, 2,
                     vt::cuda::nvfp4::kFullTacticSetVersion};
  plan.tactic_id = tactic;
  return plan;
}

std::string ReadFile(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("open fixture failed");
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

fs::path FixturePath() {
  return fs::path(TEST_FIXTURES_DIR) / "nvfp4_flashinfer_v025_gb10" /
         "autotune_configs.json";
}

FlashInferImportTarget ImportTarget() {
  FlashInferImportTarget target;
  target.expected_metadata = FlashInferCacheMetadata{
      "0.6.13", "13.0", "13.1.0", "91900", "1.26.0", "NVIDIA GB10"};
  target.device_ordinal = 0;
  target.architecture = 121;
  target.output_dtype = 2;
  return target;
}

struct ExpectedPlan {
  uint32_t m;
  int32_t n;
  int32_t k;
  int tactic;
};

constexpr std::array<ExpectedPlan, 64> kExpectedOraclePlans{{
    {1, 5120, 6144, 4},      {1, 5120, 17408, 4},
    {1, 14336, 5120, 4},     {1, 34816, 5120, 6},
    {2, 5120, 6144, 4},      {2, 5120, 17408, 14},
    {2, 14336, 5120, 4},     {2, 34816, 5120, 14},
    {4, 5120, 6144, 4},      {4, 5120, 17408, 4},
    {4, 14336, 5120, 4},     {4, 34816, 5120, 4},
    {8, 5120, 6144, 4},      {8, 5120, 17408, 6},
    {8, 14336, 5120, 12},    {8, 34816, 5120, 12},
    {16, 5120, 6144, 0},     {16, 5120, 17408, 6},
    {16, 14336, 5120, 4},    {16, 34816, 5120, 14},
    {32, 5120, 6144, 6},     {32, 5120, 17408, 15},
    {32, 14336, 5120, 4},    {32, 34816, 5120, 14},
    {64, 5120, 6144, 12},    {64, 5120, 17408, 4},
    {64, 14336, 5120, 15},   {64, 34816, 5120, 7},
    {128, 5120, 6144, 17},   {128, 5120, 17408, 13},
    {128, 14336, 5120, 13},  {128, 34816, 5120, 7},
    {256, 5120, 6144, 24},   {256, 5120, 17408, 14},
    {256, 14336, 5120, 23},  {256, 34816, 5120, 15},
    {512, 5120, 6144, 24},   {512, 5120, 17408, 22},
    {512, 14336, 5120, 25},  {512, 34816, 5120, 19},
    {768, 5120, 6144, 19},   {768, 5120, 17408, 19},
    {768, 14336, 5120, 24},  {768, 34816, 5120, 24},
    {1024, 5120, 6144, 19},  {1024, 5120, 17408, 17},
    {1024, 14336, 5120, 16}, {1024, 34816, 5120, 24},
    {1280, 5120, 6144, 18},  {1280, 5120, 17408, 19},
    {1280, 14336, 5120, 16}, {1280, 34816, 5120, 25},
    {1536, 5120, 6144, 24},  {1536, 5120, 17408, 19},
    {1536, 14336, 5120, 24}, {1536, 34816, 5120, 24},
    {1792, 5120, 6144, 16},  {1792, 5120, 17408, 25},
    {1792, 14336, 5120, 16}, {1792, 34816, 5120, 18},
    {2048, 5120, 6144, 18},  {2048, 5120, 17408, 20},
    {2048, 14336, 5120, 22}, {2048, 34816, 5120, 16},
}};

}  // namespace

TEST_CASE("NVFP4 persistent cache resolves deterministic modes and paths") {
  TempDir temp;
  const PersistentCacheMetadata metadata = Metadata();
  ScopedEnv persistent("VT_FP4_PERSISTENT_CACHE", "1");
  ScopedEnv read_only("VT_FP4_AUTOTUNE_CACHE_READONLY", "0");
  ScopedEnv native("VT_FP4_AUTOTUNE_CACHE_PATH", std::nullopt);
  ScopedEnv flashinfer("VT_FP4_FLASHINFER_CACHE_PATH", std::nullopt);
  ScopedEnv delay("VT_FP4_AUTOTUNE_DELAY_US", std::nullopt);
  ScopedEnv xdg("XDG_CACHE_HOME", temp.path().string());

  auto options = vt::cuda::nvfp4::ResolvePersistentCacheOptions(metadata);
  CHECK(options.enabled);
  CHECK_FALSE(options.read_only);
  CHECK(options.delay_microseconds == 5000);
  CHECK(options.native_path.filename() == "autotune_configs.json");
  CHECK(options.native_path.string().find("sm_121") != std::string::npos);
  CHECK(options.native_path.string().find(metadata.tactic_descriptor_digest) !=
        std::string::npos);
  CHECK(options.native_path.string().find("build_test-build") !=
        std::string::npos);

  {
    ScopedEnv override("VT_FP4_AUTOTUNE_CACHE_PATH",
                       (temp.path() / "override.json").string());
    ScopedEnv imported("VT_FP4_FLASHINFER_CACHE_PATH",
                       (temp.path() / "flashinfer.json").string());
    ScopedEnv frozen("VT_FP4_AUTOTUNE_CACHE_READONLY", "true");
    ScopedEnv delay_zero("VT_FP4_AUTOTUNE_DELAY_US", "0");
    options = vt::cuda::nvfp4::ResolvePersistentCacheOptions(metadata);
    CHECK(options.read_only);
    CHECK(options.delay_microseconds == 0);
    CHECK(options.native_path == temp.path() / "override.json");
    CHECK(options.flashinfer_path == temp.path() / "flashinfer.json");
  }
  {
    ScopedEnv disabled("VT_FP4_PERSISTENT_CACHE", "0");
    ScopedEnv ignored_delay("VT_FP4_AUTOTUNE_DELAY_US", "invalid-while-disabled");
    options = vt::cuda::nvfp4::ResolvePersistentCacheOptions(metadata);
    CHECK_FALSE(options.enabled);
    CHECK(options.native_path.empty());
  }
  {
    ScopedEnv invalid("VT_FP4_AUTOTUNE_DELAY_US", "5000000");
    CHECK_THROWS_WITH_AS(
        vt::cuda::nvfp4::ResolvePersistentCacheOptions(metadata),
        "invalid VT_FP4_AUTOTUNE_DELAY_US", std::runtime_error);
  }
  {
    ScopedEnv no_xdg("XDG_CACHE_HOME", std::nullopt);
    ScopedEnv no_home("HOME", std::nullopt);
    options = vt::cuda::nvfp4::ResolvePersistentCacheOptions(metadata);
    CHECK_FALSE(options.enabled);
    CHECK_FALSE(options.fallback_reason.empty());
    ScopedEnv frozen("VT_FP4_AUTOTUNE_CACHE_READONLY", "1");
    CHECK_THROWS_WITH_AS(
        vt::cuda::nvfp4::ResolvePersistentCacheOptions(metadata),
        "read-only NVFP4 cache requested without a cache path/root",
        std::runtime_error);
  }
}

TEST_CASE("NVFP4 native cache round trips deterministically and rejects stale data") {
  PersistentCacheDocument document;
  document.metadata = Metadata();
  document.plans = {Plan(16, 5120, 17408, 6), Plan(1, 14336, 5120, 4)};

  const std::string serialized =
      vt::cuda::nvfp4::SerializeNativeCache(document);
  const PersistentCacheDocument parsed =
      vt::cuda::nvfp4::ParseNativeCache(serialized, document.metadata);
  REQUIRE(parsed.plans.size() == 2);
  CHECK(parsed.plans[0].key.m_bucket == 1);
  CHECK(parsed.plans[1].key.m_bucket == 16);
  CHECK(vt::cuda::nvfp4::SerializeNativeCache(parsed) == serialized);

  nlohmann::json root = nlohmann::json::parse(serialized);
  root["_metadata"]["gpu"] = "foreign GPU";
  CHECK_THROWS_WITH_AS(
      vt::cuda::nvfp4::ParseNativeCache(root.dump(), document.metadata),
      "NVFP4 cache metadata mismatch for gpu", std::runtime_error);

  root = nlohmann::json::parse(serialized);
  root["plans"].push_back(root["plans"].front());
  CHECK_THROWS_WITH_AS(
      vt::cuda::nvfp4::ParseNativeCache(root.dump(), document.metadata),
      "duplicate NVFP4 persistent plan key", std::runtime_error);

  root = nlohmann::json::parse(serialized);
  root["plans"][0]["tactic_id"] = 32;
  CHECK_THROWS_WITH_AS(
      vt::cuda::nvfp4::ParseNativeCache(root.dump(), document.metadata),
      "unknown NVFP4 persistent plan runner/tactic", std::runtime_error);
  CHECK_THROWS_AS(
      vt::cuda::nvfp4::ParseNativeCache("{", document.metadata),
      std::runtime_error);
}

TEST_CASE("NVFP4 native cache merge keeps compatible current plans authoritative") {
  PersistentCacheDocument prior;
  prior.metadata = Metadata();
  prior.plans = {Plan(1, 14336, 5120, 4), Plan(2, 14336, 5120, 4)};
  PersistentCacheDocument current;
  current.metadata = prior.metadata;
  current.plans = {Plan(1, 14336, 5120, 6), Plan(4, 14336, 5120, 4)};

  const PersistentCacheDocument merged =
      vt::cuda::nvfp4::MergeNativeCaches(prior, current);
  REQUIRE(merged.plans.size() == 3);
  CHECK(merged.plans[0].key.m_bucket == 1);
  CHECK(merged.plans[0].tactic_id == 6);
  CHECK(merged.plans[1].key.m_bucket == 2);
  CHECK(merged.plans[2].key.m_bucket == 4);

  current.metadata.cuda_runtime = "13.1";
  CHECK_THROWS_WITH_AS(vt::cuda::nvfp4::MergeNativeCaches(prior, current),
                       "NVFP4 cache metadata mismatch for cuda_runtime",
                       std::runtime_error);
}

TEST_CASE("NVFP4 native cache publication is atomic and leaves no temp files") {
  TempDir temp;
  const fs::path path = temp.path() / "nested" / "cache.json";
  PersistentCacheDocument first;
  first.metadata = Metadata();
  first.plans = {Plan(1, 14336, 5120, 4)};
  PersistentCacheDocument second;
  second.metadata = first.metadata;
  second.plans = {Plan(1, 14336, 5120, 6), Plan(2, 14336, 5120, 4)};

  vt::cuda::nvfp4::WriteNativeCacheAtomically(path, first);
  std::atomic<bool> stop{false};
  std::exception_ptr reader_error;
  std::thread reader([&] {
    try {
      while (!stop.load(std::memory_order_acquire)) {
        const auto loaded =
            vt::cuda::nvfp4::LoadNativeCache(path, first.metadata);
        if (loaded.plans.size() != 1 && loaded.plans.size() != 2) {
          throw std::runtime_error("reader observed partial cache");
        }
      }
    } catch (...) {
      reader_error = std::current_exception();
    }
  });
  for (int i = 0; i < 20; ++i) {
    vt::cuda::nvfp4::WriteNativeCacheAtomically(path,
                                                i % 2 == 0 ? second : first);
  }
  stop.store(true, std::memory_order_release);
  reader.join();
  CHECK(reader_error == nullptr);
  CHECK(vt::cuda::nvfp4::LoadNativeCache(path, first.metadata).plans.size() ==
        1);

  for (const auto& entry : fs::directory_iterator(path.parent_path())) {
    CHECK(entry.path().filename().string().find(".cache.json.") ==
          std::string::npos);
  }
  fs::create_directory(temp.path() / "destination-is-directory");
  CHECK_THROWS_AS(vt::cuda::nvfp4::WriteNativeCacheAtomically(
                      temp.path() / "destination-is-directory", first),
                  std::runtime_error);
}

TEST_CASE("FlashInfer v0.25 GB10 cache imports all exact FP4 plans") {
  const std::string fixture = ReadFile(FixturePath());
  const FlashInferImportTarget target = ImportTarget();
  const auto imported =
      vt::cuda::nvfp4::ParseFlashInferCache(fixture, target);
  CHECK(imported.ignored_foreign_entries == 0);
  REQUIRE(imported.plans.size() == kExpectedOraclePlans.size());

  std::map<std::tuple<uint32_t, int32_t, int32_t>, int> actual;
  for (const PersistentPlan& plan : imported.plans) {
    actual.emplace(std::make_tuple(plan.key.m_bucket, plan.key.n, plan.key.k),
                   plan.tactic_id);
  }
  for (const ExpectedPlan& expected : kExpectedOraclePlans) {
    CAPTURE(expected.m);
    CAPTURE(expected.n);
    CAPTURE(expected.k);
    const auto found =
        actual.find(std::make_tuple(expected.m, expected.n, expected.k));
    REQUIRE(found != actual.end());
    CHECK(found->second == expected.tactic);
  }
  CHECK(vt::cuda::nvfp4::LoadFlashInferCache(FixturePath(), target).plans ==
        imported.plans);
}

TEST_CASE("FlashInfer cache import rejects metadata keys values and duplicates") {
  const FlashInferImportTarget target = ImportTarget();
  nlohmann::json root = nlohmann::json::parse(ReadFile(FixturePath()));

  nlohmann::json wildcard = root;
  wildcard["_metadata"]["gpu"] = "*";
  CHECK(vt::cuda::nvfp4::ParseFlashInferCache(wildcard.dump(), target)
            .plans.size() == 64);

  nlohmann::json stale = root;
  stale["_metadata"]["cuda_version"] = "12.9";
  CHECK_THROWS_WITH_AS(
      vt::cuda::nvfp4::ParseFlashInferCache(stale.dump(), target),
      "FlashInfer cache metadata mismatch for cuda_version",
      std::runtime_error);

  const auto first = std::next(root.begin());
  const std::string first_key = first.key();
  nlohmann::json bad_value = root;
  bad_value[first_key][1] = 32;
  CHECK_THROWS_WITH_AS(
      vt::cuda::nvfp4::ParseFlashInferCache(bad_value.dump(), target),
      "unknown NVFP4 persistent plan runner/tactic", std::runtime_error);

  nlohmann::json malformed = root;
  const nlohmann::json value = malformed[first_key];
  malformed.erase(first_key);
  malformed[first_key + " trailing"] = value;
  CHECK_THROWS_WITH_AS(
      vt::cuda::nvfp4::ParseFlashInferCache(malformed.dump(), target),
      "malformed FlashInfer fp4_gemm cache key: trailing input",
      std::runtime_error);

  nlohmann::json foreign = root;
  foreign["('other_op', 'Runner', (), ())"] = nlohmann::json::array({"Runner", 0});
  const auto with_foreign =
      vt::cuda::nvfp4::ParseFlashInferCache(foreign.dump(), target);
  CHECK(with_foreign.plans.size() == 64);
  CHECK(with_foreign.ignored_foreign_entries == 1);

  nlohmann::json duplicate = root;
  std::string spaced_key = first_key;
  const size_t comma = spaced_key.find(", '");
  REQUIRE(comma != std::string::npos);
  spaced_key.insert(comma + 1, " ");
  duplicate[spaced_key] = value;
  CHECK_THROWS_WITH_AS(
      vt::cuda::nvfp4::ParseFlashInferCache(duplicate.dump(), target),
      "duplicate FlashInfer fp4_gemm cache key", std::runtime_error);
}
