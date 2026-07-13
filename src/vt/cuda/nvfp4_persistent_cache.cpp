#include "vt/cuda/nvfp4_persistent_cache.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <unistd.h>
#include <utility>

#include <nlohmann/json.hpp>

#include "vt/cuda/nvfp4_tactic_ids.h"

namespace vt::cuda::nvfp4 {
namespace {

using Json = nlohmann::json;

struct PlanKeyLess {
  bool operator()(const PlanKey& lhs, const PlanKey& rhs) const {
    return std::tie(lhs.m_bucket, lhs.n, lhs.k, lhs.device_ordinal,
                    lhs.architecture, lhs.output_dtype,
                    lhs.tactic_set_version) <
           std::tie(rhs.m_bucket, rhs.n, rhs.k, rhs.device_ordinal,
                    rhs.architecture, rhs.output_dtype,
                    rhs.tactic_set_version);
  }
};

bool PlanLess(const PersistentPlan& lhs, const PersistentPlan& rhs) {
  const PlanKeyLess less;
  if (less(lhs.key, rhs.key)) return true;
  if (less(rhs.key, lhs.key)) return false;
  return std::tie(lhs.fp4_layout, lhs.scale_layout, lhs.runner,
                  lhs.tactic_id) <
         std::tie(rhs.fp4_layout, rhs.scale_layout, rhs.runner,
                  rhs.tactic_id);
}

std::string ErrnoMessage(std::string_view operation,
                         const std::filesystem::path& path) {
  return std::string(operation) + " " + path.string() + ": " +
         std::strerror(errno);
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    if (error) {
      throw std::runtime_error("query NVFP4 cache file " + path.string() +
                               ": " + error.message());
    }
    throw std::runtime_error("NVFP4 cache path is not a regular file: " +
                             path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("open NVFP4 cache file: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (input.bad()) {
    throw std::runtime_error("read NVFP4 cache file: " + path.string());
  }
  return contents.str();
}

void RequireNonEmpty(std::string_view field, const std::string& value) {
  if (value.empty()) {
    throw std::runtime_error("NVFP4 cache metadata field is empty: " +
                             std::string(field));
  }
}

void ValidateMetadataSelf(const PersistentCacheMetadata& metadata) {
  if (metadata.format != kPersistentCacheFormat) {
    throw std::runtime_error("unsupported NVFP4 cache format: " +
                             metadata.format);
  }
  RequireNonEmpty("cuda_runtime", metadata.cuda_runtime);
  RequireNonEmpty("cuda_driver", metadata.cuda_driver);
  RequireNonEmpty("cutlass_version", metadata.cutlass_version);
  RequireNonEmpty("gpu", metadata.gpu);
  RequireNonEmpty("output_dtype", metadata.output_dtype);
  RequireNonEmpty("fp4_layout", metadata.fp4_layout);
  RequireNonEmpty("scale_layout", metadata.scale_layout);
  RequireNonEmpty("tactic_descriptor_digest",
                  metadata.tactic_descriptor_digest);
  RequireNonEmpty("build_id", metadata.build_id);
  if (metadata.device_ordinal < 0 || metadata.architecture <= 0) {
    throw std::runtime_error("invalid NVFP4 cache device metadata");
  }
  if (metadata.tactic_set_version != kFullTacticSetVersion) {
    throw std::runtime_error("unsupported NVFP4 tactic-set version");
  }
  if (metadata.tactic_descriptor_digest != Nvfp4TacticDescriptorDigest()) {
    throw std::runtime_error("NVFP4 tactic descriptor digest mismatch");
  }
  if (metadata.warmups == 0 || metadata.repeats == 0 ||
      metadata.hybrid_bucket_version != kHybridBucketVersion) {
    throw std::runtime_error("invalid NVFP4 cache timing/bucket metadata");
  }
}

template <typename Value>
void RequireMetadataEqual(std::string_view field, const Value& actual,
                          const Value& expected) {
  if (actual != expected) {
    std::ostringstream message;
    message << "NVFP4 cache metadata mismatch for " << field;
    throw std::runtime_error(message.str());
  }
}

void ValidateMetadataMatch(const PersistentCacheMetadata& actual,
                           const PersistentCacheMetadata& expected) {
  ValidateMetadataSelf(expected);
  ValidateMetadataSelf(actual);
  RequireMetadataEqual("format", actual.format, expected.format);
  RequireMetadataEqual("cuda_runtime", actual.cuda_runtime,
                       expected.cuda_runtime);
  RequireMetadataEqual("cuda_driver", actual.cuda_driver,
                       expected.cuda_driver);
  RequireMetadataEqual("cutlass_version", actual.cutlass_version,
                       expected.cutlass_version);
  RequireMetadataEqual("gpu", actual.gpu, expected.gpu);
  RequireMetadataEqual("device_ordinal", actual.device_ordinal,
                       expected.device_ordinal);
  RequireMetadataEqual("architecture", actual.architecture,
                       expected.architecture);
  RequireMetadataEqual("output_dtype", actual.output_dtype,
                       expected.output_dtype);
  RequireMetadataEqual("output_dtype_id", actual.output_dtype_id,
                       expected.output_dtype_id);
  RequireMetadataEqual("fp4_layout", actual.fp4_layout,
                       expected.fp4_layout);
  RequireMetadataEqual("scale_layout", actual.scale_layout,
                       expected.scale_layout);
  RequireMetadataEqual("tactic_set_version", actual.tactic_set_version,
                       expected.tactic_set_version);
  RequireMetadataEqual("tactic_descriptor_digest",
                       actual.tactic_descriptor_digest,
                       expected.tactic_descriptor_digest);
  RequireMetadataEqual("warmups", actual.warmups, expected.warmups);
  RequireMetadataEqual("repeats", actual.repeats, expected.repeats);
  RequireMetadataEqual("delay_microseconds", actual.delay_microseconds,
                       expected.delay_microseconds);
  RequireMetadataEqual("hybrid_bucket_version",
                       actual.hybrid_bucket_version,
                       expected.hybrid_bucket_version);
  RequireMetadataEqual("build_id", actual.build_id, expected.build_id);
}

Json MetadataToJson(const PersistentCacheMetadata& metadata) {
  return Json{{"architecture", metadata.architecture},
              {"build_id", metadata.build_id},
              {"cuda_driver", metadata.cuda_driver},
              {"cuda_runtime", metadata.cuda_runtime},
              {"cutlass_version", metadata.cutlass_version},
              {"delay_microseconds", metadata.delay_microseconds},
              {"device_ordinal", metadata.device_ordinal},
              {"format", metadata.format},
              {"fp4_layout", metadata.fp4_layout},
              {"gpu", metadata.gpu},
              {"hybrid_bucket_version", metadata.hybrid_bucket_version},
              {"output_dtype", metadata.output_dtype},
              {"output_dtype_id", metadata.output_dtype_id},
              {"repeats", metadata.repeats},
              {"scale_layout", metadata.scale_layout},
              {"tactic_descriptor_digest",
               metadata.tactic_descriptor_digest},
              {"tactic_set_version", metadata.tactic_set_version},
              {"warmups", metadata.warmups}};
}

PersistentCacheMetadata MetadataFromJson(const Json& object) {
  PersistentCacheMetadata metadata;
  metadata.architecture = object.at("architecture").get<int32_t>();
  metadata.build_id = object.at("build_id").get<std::string>();
  metadata.cuda_driver = object.at("cuda_driver").get<std::string>();
  metadata.cuda_runtime = object.at("cuda_runtime").get<std::string>();
  metadata.cutlass_version = object.at("cutlass_version").get<std::string>();
  metadata.delay_microseconds =
      object.at("delay_microseconds").get<uint32_t>();
  metadata.device_ordinal = object.at("device_ordinal").get<int32_t>();
  metadata.format = object.at("format").get<std::string>();
  metadata.fp4_layout = object.at("fp4_layout").get<std::string>();
  metadata.gpu = object.at("gpu").get<std::string>();
  metadata.hybrid_bucket_version =
      object.at("hybrid_bucket_version").get<uint32_t>();
  metadata.output_dtype = object.at("output_dtype").get<std::string>();
  metadata.output_dtype_id = object.at("output_dtype_id").get<uint8_t>();
  metadata.repeats = object.at("repeats").get<uint32_t>();
  metadata.scale_layout = object.at("scale_layout").get<std::string>();
  metadata.tactic_descriptor_digest =
      object.at("tactic_descriptor_digest").get<std::string>();
  metadata.tactic_set_version =
      object.at("tactic_set_version").get<uint32_t>();
  metadata.warmups = object.at("warmups").get<uint32_t>();
  return metadata;
}

void ValidatePlan(const PersistentPlan& plan,
                  const PersistentCacheMetadata& metadata) {
  if (plan.key.m_bucket == 0 ||
      HybridMBucket(plan.key.m_bucket) != plan.key.m_bucket ||
      plan.key.n <= 0 || plan.key.k <= 0 || plan.key.k % 16 != 0) {
    throw std::runtime_error("invalid NVFP4 persistent plan shape/bucket");
  }
  if (plan.key.device_ordinal != metadata.device_ordinal ||
      plan.key.architecture != metadata.architecture ||
      plan.key.output_dtype != metadata.output_dtype_id ||
      plan.key.tactic_set_version != metadata.tactic_set_version) {
    throw std::runtime_error("NVFP4 persistent plan key disagrees with metadata");
  }
  if (plan.fp4_layout != metadata.fp4_layout ||
      plan.scale_layout != metadata.scale_layout) {
    throw std::runtime_error("NVFP4 persistent plan layout disagrees with metadata");
  }
  if (plan.runner != kPersistentCacheRunner ||
      TacticDescriptorForId(plan.tactic_id) == nullptr) {
    throw std::runtime_error("unknown NVFP4 persistent plan runner/tactic");
  }
}

Json PlanToJson(const PersistentPlan& plan) {
  return Json{{"architecture", plan.key.architecture},
              {"device_ordinal", plan.key.device_ordinal},
              {"fp4_layout", plan.fp4_layout},
              {"k", plan.key.k},
              {"m_bucket", plan.key.m_bucket},
              {"n", plan.key.n},
              {"output_dtype", plan.key.output_dtype},
              {"runner", plan.runner},
              {"scale_layout", plan.scale_layout},
              {"tactic_id", plan.tactic_id},
              {"tactic_set_version", plan.key.tactic_set_version}};
}

PersistentPlan PlanFromJson(const Json& object) {
  PersistentPlan plan;
  plan.key.architecture = object.at("architecture").get<int32_t>();
  plan.key.device_ordinal = object.at("device_ordinal").get<int32_t>();
  plan.fp4_layout = object.at("fp4_layout").get<std::string>();
  plan.key.k = object.at("k").get<int32_t>();
  plan.key.m_bucket = object.at("m_bucket").get<uint32_t>();
  plan.key.n = object.at("n").get<int32_t>();
  plan.key.output_dtype = object.at("output_dtype").get<uint8_t>();
  plan.runner = object.at("runner").get<std::string>();
  plan.scale_layout = object.at("scale_layout").get<std::string>();
  plan.tactic_id = object.at("tactic_id").get<int>();
  plan.key.tactic_set_version =
      object.at("tactic_set_version").get<uint32_t>();
  return plan;
}

std::vector<PersistentPlan> SortAndValidatePlans(
    std::vector<PersistentPlan> plans,
    const PersistentCacheMetadata& metadata) {
  std::map<PlanKey, int, PlanKeyLess> unique;
  for (const PersistentPlan& plan : plans) {
    ValidatePlan(plan, metadata);
    if (!unique.emplace(plan.key, plan.tactic_id).second) {
      throw std::runtime_error("duplicate NVFP4 persistent plan key");
    }
  }
  std::sort(plans.begin(), plans.end(), PlanLess);
  return plans;
}

std::string EnvironmentValue(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

bool EnvironmentBool(const char* name, bool default_value) {
  const std::string value = EnvironmentValue(name);
  if (value.empty()) return default_value;
  if (value == "1" || value == "true" || value == "on") return true;
  if (value == "0" || value == "false" || value == "off") return false;
  throw std::runtime_error(std::string("invalid boolean value for ") + name);
}

uint32_t EnvironmentDelay() {
  const std::string value = EnvironmentValue("VT_FP4_AUTOTUNE_DELAY_US");
  if (value.empty()) return kFlashInferDelayMicroseconds;
  uint32_t parsed = 0;
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto [position, error] = std::from_chars(begin, end, parsed);
  if (error != std::errc() || position != end || parsed > 1000000U) {
    throw std::runtime_error("invalid VT_FP4_AUTOTUNE_DELAY_US");
  }
  return parsed;
}

std::string PathComponent(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const unsigned char ch : value) {
    if (std::isalnum(ch) != 0 || ch == '.' || ch == '_' || ch == '-') {
      result.push_back(static_cast<char>(ch));
    } else {
      result.push_back('_');
    }
  }
  return result.empty() ? "unknown" : result;
}

bool FlashInferMetadataValueMatches(const std::string& actual,
                                    const std::string& expected) {
  return actual == "*" || actual == expected;
}

void ValidateFlashInferMetadata(const Json& root,
                                const FlashInferCacheMetadata& expected) {
  const Json& metadata = root.at("_metadata");
  const std::array<std::pair<const char*, const std::string*>, 6> fields{{
      {"flashinfer_version", &expected.flashinfer_version},
      {"cuda_version", &expected.cuda_version},
      {"cublas_version", &expected.cublas_version},
      {"cudnn_version", &expected.cudnn_version},
      {"cudnn_frontend_version", &expected.cudnn_frontend_version},
      {"gpu", &expected.gpu},
  }};
  for (const auto& [name, expected_value] : fields) {
    RequireNonEmpty(name, *expected_value);
    const std::string actual = metadata.at(name).get<std::string>();
    if (!FlashInferMetadataValueMatches(actual, *expected_value)) {
      throw std::runtime_error(std::string("FlashInfer cache metadata mismatch for ") +
                               name);
    }
  }
}

class FlashInferKeyParser {
 public:
  explicit FlashInferKeyParser(std::string_view input) : input_(input) {}

  struct Shape {
    uint32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;
  };

  Shape Parse() {
    Expect('(');
    RequireString("fp4_gemm");
    Comma();
    RequireString(kPersistentCacheRunner);
    Comma();
    Expect('(');
    const auto activation = Pair();
    Comma();
    const auto weight = Pair();
    Comma();
    const auto activation_scale = Pair();
    Comma();
    const auto weight_scale = Pair();
    Comma();
    EmptyTuple();
    Comma();
    Single(0);
    Comma();
    const auto output = Pair();
    Comma();
    Single(0);
    Comma();
    Single(0);
    Comma();
    Single(-1);
    Expect(')');
    Comma();
    EmptyTuple();
    Expect(')');
    SkipSpaces();
    if (position_ != input_.size()) Fail("trailing input");

    const int64_t m = activation.first;
    const int64_t packed_k = activation.second;
    const int64_t n = weight.second;
    if (m <= 0 || packed_k <= 0 || n <= 0 ||
        weight.first != packed_k || activation_scale.first != -1 ||
        activation_scale.second * 8 != packed_k ||
        weight_scale.first != activation_scale.second ||
        weight_scale.second != n || output.first != -1 ||
        output.second != n || packed_k > std::numeric_limits<int32_t>::max() / 2 ||
        m > std::numeric_limits<uint32_t>::max() ||
        n > std::numeric_limits<int32_t>::max()) {
      Fail("inconsistent fp4_gemm shapes");
    }
    const uint32_t m_value = static_cast<uint32_t>(m);
    if (HybridMBucket(m_value) != m_value) {
      Fail("M is not a hybrid optimization bucket");
    }
    return Shape{m_value, static_cast<int32_t>(n),
                 static_cast<int32_t>(packed_k * 2)};
  }

 private:
  [[noreturn]] void Fail(std::string_view reason) const {
    throw std::runtime_error("malformed FlashInfer fp4_gemm cache key: " +
                             std::string(reason));
  }

  void SkipSpaces() {
    while (position_ < input_.size() && input_[position_] == ' ') ++position_;
  }

  void Expect(char expected) {
    SkipSpaces();
    if (position_ >= input_.size() || input_[position_] != expected) {
      Fail(std::string("expected '") + expected + "'");
    }
    ++position_;
  }

  void Comma() { Expect(','); }

  std::string QuotedString() {
    SkipSpaces();
    if (position_ >= input_.size() || input_[position_] != '\'') {
      Fail("expected quoted string");
    }
    ++position_;
    std::string result;
    while (position_ < input_.size() && input_[position_] != '\'') {
      const char ch = input_[position_++];
      if (ch == '\\') Fail("escaped string is unsupported");
      result.push_back(ch);
    }
    if (position_ >= input_.size()) Fail("unterminated quoted string");
    ++position_;
    return result;
  }

  void RequireString(std::string_view expected) {
    if (QuotedString() != expected) Fail("unexpected string field");
  }

  int64_t Integer() {
    SkipSpaces();
    const size_t begin = position_;
    if (position_ < input_.size() && input_[position_] == '-') ++position_;
    const size_t digits = position_;
    while (position_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
      ++position_;
    }
    if (digits == position_) Fail("expected integer");
    int64_t value = 0;
    const auto [end, error] = std::from_chars(
        input_.data() + begin, input_.data() + position_, value);
    if (error != std::errc() || end != input_.data() + position_) {
      Fail("integer overflow");
    }
    return value;
  }

  std::pair<int64_t, int64_t> Pair() {
    Expect('(');
    const int64_t first = Integer();
    Comma();
    const int64_t second = Integer();
    Expect(')');
    return {first, second};
  }

  void EmptyTuple() {
    Expect('(');
    Expect(')');
  }

  void Single(int64_t expected) {
    Expect('(');
    if (Integer() != expected) Fail("unexpected singleton value");
    Comma();
    Expect(')');
  }

  std::string_view input_;
  size_t position_ = 0;
};

Json ParseJson(std::string_view contents, std::string_view kind) {
  try {
    return Json::parse(contents.begin(), contents.end());
  } catch (const Json::exception& error) {
    throw std::runtime_error(std::string("invalid ") + std::string(kind) +
                             " JSON: " + error.what());
  }
}

}  // namespace

std::string Nvfp4TacticDescriptorDigest() {
  std::ostringstream canonical;
  for (const TacticDescriptor& descriptor : kFullTacticDescriptors) {
    canonical << descriptor.id << ':' << descriptor.tile_m << ':'
              << descriptor.tile_n << ':' << descriptor.tile_k << ':'
              << (descriptor.swap_ab ? 1 : 0) << ':'
              << (descriptor.stream_k ? 1 : 0) << ':' << descriptor.name
              << ';';
  }
  uint64_t digest = 14695981039346656037ULL;
  for (const unsigned char ch : canonical.str()) {
    digest ^= ch;
    digest *= 1099511628211ULL;
  }
  std::ostringstream result;
  result << std::hex << std::setfill('0') << std::setw(16) << digest;
  return result.str();
}

PersistentCacheOptions ResolvePersistentCacheOptions(
    const PersistentCacheMetadata& metadata) {
  ValidateMetadataSelf(metadata);
  PersistentCacheOptions options;
  options.enabled = EnvironmentBool("VT_FP4_PERSISTENT_CACHE", true);
  if (!options.enabled) return options;
  options.read_only =
      EnvironmentBool("VT_FP4_AUTOTUNE_CACHE_READONLY", false);
  options.delay_microseconds = EnvironmentDelay();

  const std::string native_override =
      EnvironmentValue("VT_FP4_AUTOTUNE_CACHE_PATH");
  const std::string flashinfer_override =
      EnvironmentValue("VT_FP4_FLASHINFER_CACHE_PATH");
  if (!flashinfer_override.empty()) {
    options.flashinfer_path = flashinfer_override;
  }
  if (!native_override.empty()) {
    options.native_path = native_override;
    return options;
  }

  std::filesystem::path root;
  const std::string xdg = EnvironmentValue("XDG_CACHE_HOME");
  const std::string home = EnvironmentValue("HOME");
  if (!xdg.empty()) {
    root = xdg;
  } else if (!home.empty()) {
    root = std::filesystem::path(home) / ".cache";
  } else {
    options.fallback_reason = "XDG_CACHE_HOME and HOME are unset";
    if (options.flashinfer_path.empty()) {
      if (options.read_only) {
        throw std::runtime_error(
            "read-only NVFP4 cache requested without a cache path/root");
      }
      options.enabled = false;
    }
    return options;
  }

  options.native_path =
      root / "vllm.cpp" / "nvfp4_autotune" /
      PathComponent(metadata.format) /
      ("sm_" + std::to_string(metadata.architecture)) /
      ("device_" + std::to_string(metadata.device_ordinal) + "-gpu_" +
       PathComponent(metadata.gpu)) /
      ("cuda_" + PathComponent(metadata.cuda_runtime) + "-driver_" +
       PathComponent(metadata.cuda_driver)) /
      ("cutlass_" + PathComponent(metadata.cutlass_version)) /
      ("tactics_" + PathComponent(metadata.tactic_descriptor_digest) +
       "-set_" + std::to_string(metadata.tactic_set_version)) /
      ("dtype_" + PathComponent(metadata.output_dtype) + "-id_" +
       std::to_string(metadata.output_dtype_id) + "-fp4_" +
       PathComponent(metadata.fp4_layout) + "-sf_" +
       PathComponent(metadata.scale_layout)) /
      ("timing_w" + std::to_string(metadata.warmups) + "-r" +
       std::to_string(metadata.repeats) + "-d" +
       std::to_string(metadata.delay_microseconds) + "-bucket_" +
       std::to_string(metadata.hybrid_bucket_version)) /
      ("build_" + PathComponent(metadata.build_id)) /
      "autotune_configs.json";
  return options;
}

std::string SerializeNativeCache(const PersistentCacheDocument& document) {
  ValidateMetadataSelf(document.metadata);
  const std::vector<PersistentPlan> plans =
      SortAndValidatePlans(document.plans, document.metadata);
  Json root;
  root["_metadata"] = MetadataToJson(document.metadata);
  root["plans"] = Json::array();
  for (const PersistentPlan& plan : plans) {
    root["plans"].push_back(PlanToJson(plan));
  }
  return root.dump(2) + "\n";
}

PersistentCacheDocument ParseNativeCache(
    std::string_view contents,
    const PersistentCacheMetadata& expected_metadata) {
  try {
    const Json root = ParseJson(contents, "native NVFP4 cache");
    if (!root.is_object() || !root.at("plans").is_array()) {
      throw std::runtime_error("invalid native NVFP4 cache document shape");
    }
    PersistentCacheDocument document;
    document.metadata = MetadataFromJson(root.at("_metadata"));
    ValidateMetadataMatch(document.metadata, expected_metadata);
    for (const Json& plan : root.at("plans")) {
      document.plans.push_back(PlanFromJson(plan));
    }
    document.plans =
        SortAndValidatePlans(std::move(document.plans), document.metadata);
    return document;
  } catch (const Json::exception& error) {
    throw std::runtime_error(std::string("invalid native NVFP4 cache JSON: ") +
                             error.what());
  }
}

PersistentCacheDocument LoadNativeCache(
    const std::filesystem::path& path,
    const PersistentCacheMetadata& expected_metadata) {
  return ParseNativeCache(ReadTextFile(path), expected_metadata);
}

PersistentCacheDocument MergeNativeCaches(
    const PersistentCacheDocument& prior,
    const PersistentCacheDocument& current) {
  ValidateMetadataMatch(prior.metadata, current.metadata);
  const std::vector<PersistentPlan> prior_plans =
      SortAndValidatePlans(prior.plans, prior.metadata);
  const std::vector<PersistentPlan> current_plans =
      SortAndValidatePlans(current.plans, current.metadata);
  std::map<PlanKey, PersistentPlan, PlanKeyLess> merged;
  for (const PersistentPlan& plan : prior_plans) merged.emplace(plan.key, plan);
  for (const PersistentPlan& plan : current_plans) merged.insert_or_assign(plan.key, plan);

  PersistentCacheDocument result;
  result.metadata = current.metadata;
  for (auto& [key, plan] : merged) {
    static_cast<void>(key);
    result.plans.push_back(std::move(plan));
  }
  return result;
}

void WriteNativeCacheAtomically(const std::filesystem::path& path,
                                const PersistentCacheDocument& document) {
  if (path.empty()) throw std::runtime_error("empty NVFP4 cache path");
  const std::string contents = SerializeNativeCache(document);
  std::filesystem::path parent = path.parent_path();
  if (parent.empty()) parent = ".";
  std::filesystem::create_directories(parent);
  if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
    throw std::runtime_error("NVFP4 cache destination is a directory: " +
                             path.string());
  }

  std::string pattern =
      (parent / ("." + path.filename().string() + ".XXXXXX")).string();
  std::vector<char> temporary(pattern.begin(), pattern.end());
  temporary.push_back('\0');
  int descriptor = ::mkstemp(temporary.data());
  if (descriptor < 0) {
    throw std::runtime_error(ErrnoMessage("create NVFP4 cache temp", parent));
  }
  const std::filesystem::path temporary_path(temporary.data());
  try {
    size_t written = 0;
    while (written < contents.size()) {
      const ssize_t count = ::write(descriptor, contents.data() + written,
                                    contents.size() - written);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) {
        throw std::runtime_error(
            ErrnoMessage("write NVFP4 cache temp", temporary_path));
      }
      written += static_cast<size_t>(count);
    }
    if (::fsync(descriptor) != 0) {
      throw std::runtime_error(
          ErrnoMessage("fsync NVFP4 cache temp", temporary_path));
    }
    if (::close(descriptor) != 0) {
      descriptor = -1;
      throw std::runtime_error(
          ErrnoMessage("close NVFP4 cache temp", temporary_path));
    }
    descriptor = -1;
    if (::rename(temporary_path.c_str(), path.c_str()) != 0) {
      throw std::runtime_error(ErrnoMessage("replace NVFP4 cache", path));
    }
  } catch (...) {
    if (descriptor >= 0) ::close(descriptor);
    std::error_code ignored;
    std::filesystem::remove(temporary_path, ignored);
    throw;
  }
}

FlashInferImportResult ParseFlashInferCache(
    std::string_view contents, const FlashInferImportTarget& target) {
  try {
    const Json root = ParseJson(contents, "FlashInfer autotune cache");
    if (!root.is_object()) {
      throw std::runtime_error("invalid FlashInfer autotune cache document shape");
    }
    ValidateFlashInferMetadata(root, target.expected_metadata);

    PersistentCacheMetadata local_metadata;
    local_metadata.cuda_runtime = target.expected_metadata.cuda_version;
    local_metadata.cuda_driver = "import-only";
    local_metadata.cutlass_version = "import-only";
    local_metadata.gpu = target.expected_metadata.gpu;
    local_metadata.device_ordinal = target.device_ordinal;
    local_metadata.architecture = target.architecture;
    local_metadata.output_dtype_id = target.output_dtype;
    local_metadata.fp4_layout = target.fp4_layout;
    local_metadata.scale_layout = target.scale_layout;
    local_metadata.tactic_set_version = target.tactic_set_version;
    local_metadata.tactic_descriptor_digest = Nvfp4TacticDescriptorDigest();
    local_metadata.build_id = "flashinfer-import-validation";
    ValidateMetadataSelf(local_metadata);

    FlashInferImportResult result;
    std::map<PlanKey, int, PlanKeyLess> unique;
    for (const auto& [key, value] : root.items()) {
      if (key == "_metadata") continue;
      if (!key.starts_with("('fp4_gemm'")) {
        ++result.ignored_foreign_entries;
        continue;
      }
      const FlashInferKeyParser::Shape shape = FlashInferKeyParser(key).Parse();
      if (!value.is_array() || value.size() != 2 ||
          value.at(0).get<std::string>() != kPersistentCacheRunner ||
          !value.at(1).is_number_integer()) {
        throw std::runtime_error("invalid FlashInfer fp4_gemm cache value");
      }
      PersistentPlan plan;
      plan.key = PlanKey{shape.m,
                        shape.n,
                        shape.k,
                        target.device_ordinal,
                        target.architecture,
                        target.output_dtype,
                        target.tactic_set_version};
      plan.fp4_layout = target.fp4_layout;
      plan.scale_layout = target.scale_layout;
      plan.tactic_id = value.at(1).get<int>();
      ValidatePlan(plan, local_metadata);
      if (!unique.emplace(plan.key, plan.tactic_id).second) {
        throw std::runtime_error("duplicate FlashInfer fp4_gemm cache key");
      }
      result.plans.push_back(std::move(plan));
    }
    std::sort(result.plans.begin(), result.plans.end(), PlanLess);
    return result;
  } catch (const Json::exception& error) {
    throw std::runtime_error(
        std::string("invalid FlashInfer autotune cache JSON: ") + error.what());
  }
}

FlashInferImportResult LoadFlashInferCache(
    const std::filesystem::path& path,
    const FlashInferImportTarget& target) {
  return ParseFlashInferCache(ReadTextFile(path), target);
}

}  // namespace vt::cuda::nvfp4
