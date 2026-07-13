// vllm.cpp — SM120/SM121 NVFP4 block-scaled FP4xFP4 GEMM adapter.
//
// The raw kernels are split across cuda_nvfp4_tactics_*.cu. The full default
// surface mirrors installed FlashInfer 0.6.12
// `fp4_gemm_cutlass_template_sm120.h:47-220`: 8 CTA tiles x swap-AB x
// StaticPersistent/Stream-K in exact upstream order. The immutable W1 four-
// candidate implementation is retained separately for VT_FP4_FULL_TACTICS=0.
// This TU owns vt::Tensor validation/registration, exact hybrid-bucket tuning,
// high-water workspace reuse, forced-tactic diagnostics and scale swizzling.
// W3 mirrors FlashInfer's eager autotune timing window: after three warmups,
// synchronize the stream and enqueue a one-thread 5 ms GPU delay before the
// start event so host dispatch latency cannot bias near-tied tactic choices.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vt/cuda/nvfp4_autotune.h"
#include "vt/cuda/nvfp4_cutlass_tactics.h"
#include "vt/cuda/nvfp4_persistent_cache.h"
#include "vt/cuda/nvfp4_plan_cache.h"
#include "vt/ops.h"

#ifndef VT_CUTLASS_VERSION_STRING
#define VT_CUTLASS_VERSION_STRING "unknown"
#endif

#ifndef VT_NVFP4_CACHE_BUILD_ID
#define VT_NVFP4_CACHE_BUILD_ID "unknown"
#endif

namespace vt::cuda {
namespace {

thread_local uint32_t g_fp4_warmup_max_tokens = 0;
thread_local bool g_fp4_warmup_observed = false;
std::atomic<int> g_fp4_warmup_active_scopes{0};
std::atomic<bool> g_fp4_warmup_complete{false};
std::atomic<uint64_t> g_fp4_warmup_scopes_started{0};
std::atomic<uint64_t> g_fp4_warmup_scopes_completed{0};
std::atomic<uint64_t> g_fp4_warmup_profiles_requested{0};
std::atomic<uint64_t> g_fp4_warmup_profiles_tuned{0};
std::atomic<uint64_t> g_fp4_lazy_misses{0};
std::atomic<uint64_t> g_fp4_profiles_loaded{0};
std::atomic<uint64_t> g_fp4_cache_documents_rejected{0};
std::atomic<uint64_t> g_fp4_profiles_saved{0};
std::atomic<uint32_t> g_fp4_autotune_delay_microseconds{
    nvfp4::kFlashInferDelayMicroseconds};
std::atomic<bool> g_fp4_persistent_read_only{false};

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul_nvfp4_cutlass: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Queue-scoped grow-only scratch. The unique_ptr keeps returned references
// stable if a later queue insertion rehashes the map. W3 replaces process-exit
// retention with the common backend resource lifecycle.
struct StreamScratch {
  float* alpha = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  void* bf16 = nullptr;
  size_t bf16_bytes = 0;
};

bool CutlassPoolEnabled() {
  static const bool on = [] {
    const char* value = std::getenv("VT_CUTLASS_NOPOOL");
    return !(value != nullptr && value[0] == '1');
  }();
  return on;
}

StreamScratch& ScratchFor(cudaStream_t stream) {
  static std::mutex mutex;
  static std::unordered_map<cudaStream_t, std::unique_ptr<StreamScratch>> scratch_by_stream;
  std::lock_guard<std::mutex> lock(mutex);
  auto& scratch = scratch_by_stream[stream];
  if (!scratch) scratch = std::make_unique<StreamScratch>();
  return *scratch;
}

void* EnsureScratch(void** buffer, size_t* capacity, size_t required, cudaStream_t stream,
                    const char* what) {
  if (required > *capacity) {
    if (*buffer != nullptr) Check(cudaFreeAsync(*buffer, stream), "cudaFreeAsync scratch grow");
    Check(cudaMallocAsync(buffer, required, stream), what);
    *capacity = required;
  }
  return *buffer;
}

float* PersistentAlpha(cudaStream_t stream) {
  StreamScratch& scratch = ScratchFor(stream);
  if (scratch.alpha == nullptr) {
    Check(cudaMallocAsync(&scratch.alpha, sizeof(float), stream), "cudaMallocAsync alpha");
  }
  return scratch.alpha;
}

const std::vector<nvfp4::Candidate>& FullCandidates() {
  static const std::vector<nvfp4::Candidate> candidates = [] {
    std::vector<nvfp4::Candidate> result;
    result.reserve(nvfp4::kFullTacticDescriptors.size());
    const auto append = [&](const nvfp4::CandidateGroup& group) {
      result.insert(result.end(), group.begin(), group.end());
    };
    append(nvfp4::FullTactics128x32x128());
    append(nvfp4::FullTactics128x32x256());
    append(nvfp4::FullTactics128x64x128());
    append(nvfp4::FullTactics128x64x256());
    append(nvfp4::FullTactics128x128x128());
    append(nvfp4::FullTactics128x128x256());
    append(nvfp4::FullTactics256x128x128());
    append(nvfp4::FullTactics128x256x128());
    if (result.size() != nvfp4::kFullTacticDescriptors.size()) {
      throw std::runtime_error("NVFP4 full tactic table has the wrong size");
    }
    for (size_t i = 0; i < result.size(); ++i) {
      if (result[i].descriptor.id != static_cast<int>(i)) {
        throw std::runtime_error("NVFP4 full tactic table violates stable upstream order");
      }
    }
    return result;
  }();
  return candidates;
}

const std::vector<nvfp4::Candidate>& LegacyCandidates() {
  static const std::vector<nvfp4::Candidate> candidates = [] {
    const auto& group = nvfp4::W1Tactics();
    return std::vector<nvfp4::Candidate>(group.begin(), group.end());
  }();
  return candidates;
}

bool Fp4AutotuneEnabled() {
  static const bool on = [] {
    const char* value = std::getenv("VT_FP4_AUTOTUNE");
    return value == nullptr || value[0] != '0';
  }();
  return on;
}

bool Fp4AutotuneVerbose() {
  static const bool on = [] {
    const char* value = std::getenv("VT_FP4_AUTOTUNE_VERBOSE");
    return value != nullptr && value[0] == '1';
  }();
  return on;
}

bool Fp4ExactBucketsEnabled() {
  static const bool on = [] {
    const char* value = std::getenv("VT_FP4_EXACT_BUCKETS");
    return value == nullptr || value[0] != '0';
  }();
  return on;
}

bool Fp4FullTacticsEnabled() {
  static const bool on = [] {
    const char* value = std::getenv("VT_FP4_FULL_TACTICS");
    return value == nullptr || value[0] != '0';
  }();
  return on;
}

bool Fp4PlanCacheEnabled() {
  static const bool on = [] {
    const char* value = std::getenv("VT_FP4_PLAN_CACHE");
    return value == nullptr || value[0] != '0';
  }();
  return on;
}

int Fp4ForcedTactic() {
  static const int tactic = [] {
    const char* value = std::getenv("VT_FP4_FORCE_TACTIC");
    if (value == nullptr || value[0] == '\0') return -1;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 ||
        parsed >= static_cast<long>(nvfp4::kFullTacticDescriptors.size())) {
      throw std::runtime_error("VT_FP4_FORCE_TACTIC must be an integer in [0,31]");
    }
    return static_cast<int>(parsed);
  }();
  return tactic;
}

const std::vector<nvfp4::Candidate>& ActiveCandidates() {
  return Fp4FullTacticsEnabled() ? FullCandidates() : LegacyCandidates();
}

int CandidateIndexForId(const std::vector<nvfp4::Candidate>& candidates, int tactic_id) {
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].descriptor.id == tactic_id) return static_cast<int>(i);
  }
  return -1;
}

int Fp4DeviceArchitecture(int device_ordinal) {
  constexpr size_t kCachedDevices = 64;
  static std::array<std::atomic<int>, kCachedDevices> cached{};
  if (device_ordinal >= 0 && static_cast<size_t>(device_ordinal) < cached.size()) {
    const int value = cached[static_cast<size_t>(device_ordinal)].load(std::memory_order_acquire);
    if (value != 0) return value;
  }

  int major = 0;
  int minor = 0;
  Check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal),
        "query device compute-capability major");
  Check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal),
        "query device compute-capability minor");
  const int architecture = major * 10 + minor;
  if (device_ordinal >= 0 && static_cast<size_t>(device_ordinal) < cached.size()) {
    int empty = 0;
    cached[static_cast<size_t>(device_ordinal)].compare_exchange_strong(
        empty, architecture, std::memory_order_release, std::memory_order_relaxed);
  }
  return architecture;
}

struct WorkspaceLease {
  void* data = nullptr;
  bool ephemeral = false;
};

WorkspaceLease AcquireWorkspace(size_t required, cudaStream_t stream) {
  if (required == 0) return {};
  if (!CutlassPoolEnabled()) {
    void* workspace = nullptr;
    Check(cudaMallocAsync(&workspace, required, stream), "cudaMallocAsync workspace");
    return WorkspaceLease{workspace, true};
  }

  StreamScratch& scratch = ScratchFor(stream);
  if (required > scratch.workspace_bytes) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    Check(cudaStreamIsCapturing(stream, &status), "query stream capture before workspace growth");
    if (status != cudaStreamCaptureStatusNone) {
      throw std::runtime_error("NVFP4 workspace high-water miss during CUDA graph capture");
    }
    EnsureScratch(&scratch.workspace, &scratch.workspace_bytes, required, stream,
                  "cudaMallocAsync workspace high-water");
  }
  return WorkspaceLease{scratch.workspace, false};
}

void ReleaseWorkspace(WorkspaceLease lease, cudaStream_t stream) {
  if (lease.ephemeral && lease.data != nullptr) {
    Check(cudaFreeAsync(lease.data, stream), "cudaFreeAsync workspace");
  }
}

size_t WorkspaceHighWater(const std::vector<nvfp4::Candidate>& candidates,
                          const nvfp4::LaunchParams& params) {
  size_t high_water = 0;
  size_t query_count = 0;
  for (const auto& candidate : candidates) {
    try {
      high_water = std::max(high_water, candidate.workspace_size(params));
      ++query_count;
    } catch (const std::exception&) {
      // Same as FlashInfer getWorkspaceSizeImpl: configurations rejected by
      // CUTLASS during the query do not suppress the remaining tactic family.
    }
  }
  if (query_count == 0) throw std::runtime_error("no NVFP4 tactic workspace query succeeded");
  return high_water;
}

void RunCandidate(const nvfp4::Candidate& candidate, nvfp4::LaunchParams params,
                  WorkspaceLease workspace, size_t workspace_bytes) {
  params.workspace = workspace.data;
  params.workspace_bytes = workspace_bytes;
  candidate.run(params);
}

constexpr float kFp4Inf = 3.4e38F;

// Exact source mirror of installed FlashInfer 0.6.12
// `data/csrc/nv_internal/tensorrt_llm/kernels/delayStream.cu:24-34`, used by
// `autotuner.py:1398-1424`. `__nanosleep` accepts at most one millisecond, so
// the upstream kernel loops in microsecond-sized 1000 ns increments. The
// single thread deliberately occupies negligible SM capacity while putting a
// fixed GPU-side gap between host setup and the measured event window.
__global__ void Fp4AutotuneDelayStreamKernel(long long delay_microseconds) {
  for (long long i = 0; i < delay_microseconds; ++i) __nanosleep(1000);
}

float TimeCandidate(const nvfp4::Candidate& candidate, nvfp4::LaunchParams params,
                    WorkspaceLease workspace, size_t workspace_bytes) {
  constexpr int kWarmupIterations = 3;
  constexpr int kTimingIterations = 10;
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  try {
    for (int i = 0; i < kWarmupIterations; ++i) {
      RunCandidate(candidate, params, workspace, workspace_bytes);
    }
    Check(cudaEventCreate(&begin), "autotune event create begin");
    Check(cudaEventCreate(&end), "autotune event create end");
    const uint32_t delay_microseconds =
        g_fp4_autotune_delay_microseconds.load(std::memory_order_acquire);
    if (delay_microseconds != 0) {
      // FlashInfer synchronizes before its delay launch. Without this boundary,
      // rapid host enqueue can systematically favor a different near-tied
      // tactic family even though CUDA events exclude the warmup kernels.
      Check(cudaStreamSynchronize(params.stream),
            "autotune pre-profile stream synchronize");
      Fp4AutotuneDelayStreamKernel<<<1, 1, 0, params.stream>>>(
          static_cast<long long>(delay_microseconds));
      Check(cudaGetLastError(), "autotune delay kernel launch");
    }
    Check(cudaEventRecord(begin, params.stream), "autotune event record begin");
    for (int i = 0; i < kTimingIterations; ++i) {
      RunCandidate(candidate, params, workspace, workspace_bytes);
    }
    Check(cudaEventRecord(end, params.stream), "autotune event record end");
    Check(cudaEventSynchronize(end), "autotune event synchronize");
    float elapsed_ms = 0.0F;
    Check(cudaEventElapsedTime(&elapsed_ms, begin, end), "autotune event elapsed");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed_ms / kTimingIterations;
  } catch (const std::exception&) {
    if (begin != nullptr) cudaEventDestroy(begin);
    if (end != nullptr) cudaEventDestroy(end);
    cudaGetLastError();
    return kFp4Inf;
  }
}

struct SelectedPlan {
  int candidate_index = -1;
  int tactic_id = -1;
  size_t workspace_bytes = 0;
};

using SelectedPlanCache = nvfp4::SingleFlightPlanCache<SelectedPlan>;

SelectedPlanCache& PlanCache() {
  static SelectedPlanCache plans;
  return plans;
}

struct PersistentRuntimeSession {
  nvfp4::PersistentCacheMetadata metadata;
  nvfp4::PersistentCacheOptions options;
  nvfp4::PersistentCacheDocument prior_native;
  bool enabled = false;
  bool save_allowed = true;
};

struct PersistentRuntimeDiagnostics {
  bool enabled = false;
  bool read_only = false;
  uint32_t delay_microseconds = nvfp4::kFlashInferDelayMicroseconds;
  std::string mode = "not-prepared";
  std::string native_path;
  std::string flashinfer_path;
  std::string metadata_fingerprint;
  std::vector<Nvfp4AutotuneSelectedPlan> selected_plans;
};

std::mutex g_fp4_persistent_mutex;
std::optional<PersistentRuntimeSession> g_fp4_persistent_session;
PersistentRuntimeDiagnostics g_fp4_persistent_diagnostics;

std::string FormatCudaVersion(int version) {
  const int major = version / 1000;
  const int minor = (version % 1000) / 10;
  const int patch = version % 10;
  std::string result = std::to_string(major) + "." + std::to_string(minor);
  if (patch != 0) result += "." + std::to_string(patch);
  return result;
}

nvfp4::PersistentCacheMetadata MakePersistentMetadata(int device_ordinal) {
  int runtime_version = 0;
  int driver_version = 0;
  cudaDeviceProp properties{};
  Check(cudaRuntimeGetVersion(&runtime_version), "query CUDA runtime version");
  Check(cudaDriverGetVersion(&driver_version), "query CUDA driver version");
  Check(cudaGetDeviceProperties(&properties, device_ordinal),
        "query device properties for persistent cache");

  nvfp4::PersistentCacheMetadata metadata;
  metadata.cuda_runtime = FormatCudaVersion(runtime_version);
  metadata.cuda_driver = FormatCudaVersion(driver_version);
  metadata.cutlass_version = VT_CUTLASS_VERSION_STRING;
  metadata.gpu = properties.name;
  metadata.device_ordinal = device_ordinal;
  metadata.architecture = Fp4DeviceArchitecture(device_ordinal);
  metadata.output_dtype = Name(DType::kBF16);
  metadata.output_dtype_id = static_cast<uint8_t>(DType::kBF16);
  metadata.tactic_descriptor_digest = nvfp4::Nvfp4TacticDescriptorDigest();
  metadata.build_id = VT_NVFP4_CACHE_BUILD_ID;
  return metadata;
}

nvfp4::FlashInferImportTarget MakeFlashInferImportTarget(
    const nvfp4::PersistentCacheMetadata& metadata) {
  nvfp4::FlashInferImportTarget target;
  // This seam intentionally accepts the exact dependency environment that
  // produced the binding vLLM v0.25.0 denominator. Native vllm.cpp caches use
  // the live CUDA/CUTLASS/build identity above.
  target.expected_metadata = nvfp4::FlashInferCacheMetadata{
      "0.6.13", metadata.cuda_runtime, "13.1.0", "91900", "1.26.0",
      metadata.gpu};
  target.device_ordinal = metadata.device_ordinal;
  target.architecture = metadata.architecture;
  target.output_dtype = metadata.output_dtype_id;
  target.tactic_set_version = metadata.tactic_set_version;
  target.fp4_layout = metadata.fp4_layout;
  target.scale_layout = metadata.scale_layout;
  return target;
}

std::vector<std::pair<nvfp4::PlanKey, SelectedPlan>> PreparePersistentPlans(
    const std::vector<nvfp4::PersistentPlan>& plans) {
  const auto& candidates = FullCandidates();
  std::vector<std::pair<nvfp4::PlanKey, SelectedPlan>> prepared;
  prepared.reserve(plans.size());
  for (const nvfp4::PersistentPlan& plan : plans) {
    const int index = CandidateIndexForId(candidates, plan.tactic_id);
    if (index < 0) {
      throw std::runtime_error(
          "NVFP4 persistent tactic is absent from the active tactic ABI");
    }
    nvfp4::LaunchParams params;
    params.m = static_cast<int>(plan.key.m_bucket);
    params.n = plan.key.n;
    params.k = plan.key.k;
    const size_t workspace_bytes = WorkspaceHighWater(candidates, params);
    prepared.emplace_back(
        plan.key, SelectedPlan{index, plan.tactic_id, workspace_bytes});
  }
  return prepared;
}

uint64_t InstallPersistentPlans(
    const std::vector<nvfp4::PersistentPlan>& plans) {
  const auto prepared = PreparePersistentPlans(plans);
  uint64_t installed = 0;
  for (const auto& [key, plan] : prepared) {
    if (PlanCache().InsertReadyIfAbsent(key, plan)) ++installed;
  }
  g_fp4_profiles_loaded.fetch_add(installed, std::memory_order_relaxed);
  return installed;
}

std::vector<nvfp4::PersistentPlan> SnapshotPersistentPlans(
    const nvfp4::PersistentCacheMetadata& metadata) {
  std::vector<nvfp4::PersistentPlan> result;
  for (const auto& [key, selected] : PlanCache().SnapshotReady()) {
    if (key.device_ordinal != metadata.device_ordinal ||
        key.architecture != metadata.architecture ||
        key.output_dtype != metadata.output_dtype_id ||
        key.tactic_set_version != metadata.tactic_set_version) {
      continue;
    }
    const int index = CandidateIndexForId(FullCandidates(), selected.tactic_id);
    if (index < 0 || index != selected.candidate_index) {
      throw std::runtime_error(
          "NVFP4 ready plan does not match the persistent tactic ABI");
    }
    nvfp4::PersistentPlan plan;
    plan.key = key;
    plan.fp4_layout = metadata.fp4_layout;
    plan.scale_layout = metadata.scale_layout;
    plan.tactic_id = selected.tactic_id;
    result.push_back(std::move(plan));
  }
  return result;
}

std::vector<Nvfp4AutotuneSelectedPlan> MakeSelectedPlanDiagnostics(
    const std::vector<nvfp4::PersistentPlan>& plans) {
  std::vector<Nvfp4AutotuneSelectedPlan> result;
  result.reserve(plans.size());
  for (const nvfp4::PersistentPlan& plan : plans) {
    result.push_back(Nvfp4AutotuneSelectedPlan{
        plan.key.m_bucket, plan.key.n, plan.key.k, plan.tactic_id});
  }
  return result;
}

bool PathExists(const std::filesystem::path& path) {
  if (path.empty()) return false;
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error) {
    throw std::runtime_error("could not inspect NVFP4 cache path: " +
                             path.string());
  }
  return exists;
}

bool BeginPersistentRuntime(int device_ordinal) {
  std::lock_guard<std::mutex> lock(g_fp4_persistent_mutex);
  if (g_fp4_persistent_session.has_value()) {
    throw std::logic_error("overlapping NVFP4 persistent-cache sessions");
  }

  PersistentRuntimeSession session;
  session.metadata = MakePersistentMetadata(device_ordinal);
  session.options = nvfp4::ResolvePersistentCacheOptions(session.metadata);
  session.metadata.delay_microseconds = session.options.delay_microseconds;
  session.enabled = session.options.enabled;

  g_fp4_autotune_delay_microseconds.store(
      session.options.delay_microseconds, std::memory_order_release);
  g_fp4_persistent_read_only.store(
      session.enabled && session.options.read_only, std::memory_order_release);

  PersistentRuntimeDiagnostics diagnostics;
  diagnostics.enabled = session.enabled;
  diagnostics.read_only = session.enabled && session.options.read_only;
  diagnostics.delay_microseconds = session.options.delay_microseconds;
  diagnostics.mode = !session.enabled
                         ? "disabled"
                         : (session.options.read_only ? "read-only"
                                                      : "read-write");
  diagnostics.native_path = session.options.native_path.string();
  diagnostics.flashinfer_path = session.options.flashinfer_path.string();
  diagnostics.metadata_fingerprint =
      nvfp4::PersistentCacheMetadataFingerprint(session.metadata);

  uint64_t flashinfer_loaded = 0;
  uint64_t native_loaded = 0;
  if (session.enabled && !session.options.flashinfer_path.empty()) {
    try {
      const nvfp4::FlashInferImportResult imported =
          nvfp4::LoadFlashInferCache(session.options.flashinfer_path,
                                     MakeFlashInferImportTarget(session.metadata));
      flashinfer_loaded = InstallPersistentPlans(imported.plans);
    } catch (...) {
      g_fp4_cache_documents_rejected.fetch_add(1, std::memory_order_relaxed);
      g_fp4_persistent_read_only.store(false, std::memory_order_release);
      throw;
    }
  }

  session.prior_native.metadata = session.metadata;
  if (session.enabled && PathExists(session.options.native_path)) {
    try {
      session.prior_native = nvfp4::LoadNativeCache(
          session.options.native_path, session.metadata);
      native_loaded = InstallPersistentPlans(session.prior_native.plans);
    } catch (const std::exception& error) {
      g_fp4_cache_documents_rejected.fetch_add(1, std::memory_order_relaxed);
      if (session.options.read_only &&
          session.options.flashinfer_path.empty()) {
        g_fp4_persistent_read_only.store(false, std::memory_order_release);
        throw;
      }
      session.save_allowed = false;
      diagnostics.mode = session.options.read_only
                             ? "read-only-native-rejected"
                             : "read-write-native-rejected";
      std::fprintf(stderr,
                   "[VT_FP4_CACHE] rejected native cache path=%s error=%s; "
                   "%s\n",
                   session.options.native_path.string().c_str(), error.what(),
                   session.options.read_only
                       ? "using the explicit imported map; frozen misses fail"
                       : "tuning misses without overwriting it");
    }
  }

  const std::vector<nvfp4::PersistentPlan> selected =
      SnapshotPersistentPlans(session.metadata);
  diagnostics.selected_plans = MakeSelectedPlanDiagnostics(selected);
  g_fp4_persistent_diagnostics = std::move(diagnostics);
  g_fp4_persistent_session.emplace(std::move(session));

  std::fprintf(stderr,
               "[VT_FP4_CACHE] prepared mode=%s native=%s flashinfer=%s "
               "loaded=%llu (flashinfer=%llu native=%llu) rejected=%llu "
               "delay_us=%u metadata=%s selected=%zu\n",
               g_fp4_persistent_diagnostics.mode.c_str(),
               g_fp4_persistent_diagnostics.native_path.c_str(),
               g_fp4_persistent_diagnostics.flashinfer_path.c_str(),
               static_cast<unsigned long long>(flashinfer_loaded + native_loaded),
               static_cast<unsigned long long>(flashinfer_loaded),
               static_cast<unsigned long long>(native_loaded),
               static_cast<unsigned long long>(
                   g_fp4_cache_documents_rejected.load(std::memory_order_relaxed)),
               g_fp4_persistent_diagnostics.delay_microseconds,
               g_fp4_persistent_diagnostics.metadata_fingerprint.c_str(),
               g_fp4_persistent_diagnostics.selected_plans.size());
  return true;
}

void CompletePersistentRuntime() {
  std::lock_guard<std::mutex> lock(g_fp4_persistent_mutex);
  if (!g_fp4_persistent_session.has_value()) return;
  PersistentRuntimeSession& session = *g_fp4_persistent_session;
  const std::vector<nvfp4::PersistentPlan> selected =
      SnapshotPersistentPlans(session.metadata);
  g_fp4_persistent_diagnostics.selected_plans =
      MakeSelectedPlanDiagnostics(selected);

  if (session.enabled && !session.options.read_only &&
      session.save_allowed && !session.options.native_path.empty()) {
    nvfp4::PersistentCacheDocument current;
    current.metadata = session.metadata;
    current.plans = selected;
    const nvfp4::PersistentCacheDocument merged =
        nvfp4::MergeNativeCaches(session.prior_native, current);
    try {
      nvfp4::WriteNativeCacheAtomically(session.options.native_path, merged);
      g_fp4_profiles_saved.fetch_add(merged.plans.size(),
                                     std::memory_order_relaxed);
    } catch (const std::exception& error) {
      g_fp4_persistent_diagnostics.mode = "read-write-save-failed";
      std::fprintf(stderr,
                   "[VT_FP4_CACHE] could not publish native cache path=%s "
                   "error=%s; serving with the in-memory ready map\n",
                   session.options.native_path.string().c_str(), error.what());
    }
  }

  std::fprintf(stderr,
               "[VT_FP4_CACHE] complete mode=%s loaded=%llu tuned=%llu "
               "rejected=%llu saved=%llu selected=%zu metadata=%s\n",
               g_fp4_persistent_diagnostics.mode.c_str(),
               static_cast<unsigned long long>(
                   g_fp4_profiles_loaded.load(std::memory_order_relaxed)),
               static_cast<unsigned long long>(
                   g_fp4_warmup_profiles_tuned.load(std::memory_order_relaxed)),
               static_cast<unsigned long long>(
                   g_fp4_cache_documents_rejected.load(std::memory_order_relaxed)),
               static_cast<unsigned long long>(
                   g_fp4_profiles_saved.load(std::memory_order_relaxed)),
               g_fp4_persistent_diagnostics.selected_plans.size(),
               g_fp4_persistent_diagnostics.metadata_fingerprint.c_str());
  for (const Nvfp4AutotuneSelectedPlan& plan :
       g_fp4_persistent_diagnostics.selected_plans) {
    std::fprintf(stderr,
                 "[VT_FP4_CACHE] selected M=%u N=%d K=%d tactic=%d\n",
                 plan.m_bucket, plan.n, plan.k, plan.tactic_id);
  }
}

void EndPersistentRuntime(bool keep_frozen_mode) {
  std::lock_guard<std::mutex> lock(g_fp4_persistent_mutex);
  g_fp4_persistent_session.reset();
  if (!keep_frozen_mode) {
    g_fp4_persistent_read_only.store(false, std::memory_order_release);
  }
}

uint32_t PlanMBucket(const nvfp4::LaunchParams& params) {
  return Fp4ExactBucketsEnabled()
             ? nvfp4::HybridMBucket(static_cast<uint32_t>(params.m))
             : nvfp4::LegacyMBucket(static_cast<uint32_t>(params.m));
}

nvfp4::PlanKey MakePlanKey(const nvfp4::LaunchParams& params,
                           int device_ordinal, bool full_tactics) {
  return nvfp4::PlanKey{PlanMBucket(params),
                        params.n,
                        params.k,
                        device_ordinal,
                        Fp4DeviceArchitecture(device_ordinal),
                        static_cast<uint8_t>(DType::kBF16),
                        full_tactics ? nvfp4::kFullTacticSetVersion
                                     : nvfp4::kW1TacticSetVersion};
}

SelectedPlan ResolveSelectedPlan(nvfp4::LaunchParams params,
                                 int device_ordinal,
                                 bool report_lazy_miss) {
  const auto& candidates = ActiveCandidates();
  const bool full_tactics = Fp4FullTacticsEnabled();
  const nvfp4::PlanKey key = MakePlanKey(params, device_ordinal, full_tactics);
  SelectedPlanCache& plans = PlanCache();

  if (const std::optional<SelectedPlan> ready = plans.FindReady(key);
      ready.has_value()) {
    return *ready;
  }
  if (g_fp4_persistent_read_only.load(std::memory_order_acquire)) {
    std::ostringstream message;
    message << "NVFP4 frozen persistent cache miss before readiness: M="
            << params.m << " bucket=" << key.m_bucket << " N=" << params.n
            << " K=" << params.k << " device=" << device_ordinal
            << " sm=" << key.architecture;
    throw std::runtime_error(message.str());
  }
  if (report_lazy_miss &&
      g_fp4_warmup_active_scopes.load(std::memory_order_acquire) == 0 &&
      g_fp4_warmup_complete.load(std::memory_order_acquire)) {
    g_fp4_lazy_misses.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[VT_FP4_AUTOTUNE] lazy-miss after pre-serve warmup "
                 "M=%d(bucket=%u) N=%d K=%d device=%d sm=%d; tuning eagerly\n",
                 params.m, key.m_bucket, params.n, params.k, device_ordinal,
                 key.architecture);
  }

  const auto can_tune = [stream = params.stream] {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    Check(cudaStreamIsCapturing(stream, &status), "query stream capture state");
    return status == cudaStreamCaptureStatusNone;
  };
  const auto tune = [&] {
    const size_t high_water = WorkspaceHighWater(candidates, params);
    WorkspaceLease workspace = AcquireWorkspace(high_water, params.stream);
    try {
      const int forced = Fp4ForcedTactic();
      if (forced >= 0) {
        const int index = CandidateIndexForId(candidates, forced);
        if (index < 0) {
          throw std::runtime_error("forced NVFP4 tactic is not present in the active tactic set");
        }
        ReleaseWorkspace(workspace, params.stream);
        return SelectedPlan{index, forced, high_water};
      }

      std::vector<float> timings(candidates.size(), kFp4Inf);
      for (size_t i = 0; i < candidates.size(); ++i) {
        timings[i] = TimeCandidate(candidates[i], params, workspace, high_water);
      }
      int best = -1;
      for (size_t i = 0; i < timings.size(); ++i) {
        if (timings[i] < kFp4Inf &&
            (best < 0 || timings[i] < timings[static_cast<size_t>(best)])) {
          best = static_cast<int>(i);
        }
      }
      if (best < 0) throw std::runtime_error("all NVFP4 tactics rejected the requested shape");

      // W1 used a >1% threshold relative to its fixed M baseline. Preserve that
      // only in the fallback arm. FlashInfer's full autotuner chooses the
      // minimum valid event time directly.
      int chosen = best;
      int baseline = (key.m_bucket <= 256) ? 0 : 1;
      if (!full_tactics && timings[static_cast<size_t>(baseline)] < kFp4Inf &&
          !(timings[static_cast<size_t>(best)] <
            timings[static_cast<size_t>(baseline)] * 0.99F)) {
        chosen = baseline;
      }
      if (Fp4AutotuneVerbose()) {
        std::fprintf(stderr,
                     "[VT_FP4_AUTOTUNE] set=%s M=%d(bucket=%u) N=%d K=%d device=%d sm=%d "
                     "delay_us=%u -> id=%d %s (%.1f us), workspace=%zu\n",
                     full_tactics ? "full" : "w1", params.m, key.m_bucket, params.n, params.k,
                     device_ordinal, key.architecture,
                     g_fp4_autotune_delay_microseconds.load(
                         std::memory_order_relaxed),
                     candidates[static_cast<size_t>(chosen)].descriptor.id,
                     candidates[static_cast<size_t>(chosen)].descriptor.name,
                     timings[static_cast<size_t>(chosen)] * 1000.0F, high_water);
      }
      ReleaseWorkspace(workspace, params.stream);
      return SelectedPlan{chosen, candidates[static_cast<size_t>(chosen)].descriptor.id, high_water};
    } catch (...) {
      ReleaseWorkspace(workspace, params.stream);
      throw;
    }
  };
  if (!Fp4PlanCacheEnabled()) {
    if (!can_tune()) {
      throw std::runtime_error(
          "NVFP4 plan-cache bypass cannot select a tactic during CUDA graph capture");
    }
    return tune();
  }
  return nvfp4::ResolvePlan(plans, key, can_tune, tune);
}

SelectedPlan SelectPlan(nvfp4::LaunchParams params, int device_ordinal) {
  // FlashInfer's autotune context expands one maximum-token invocation into
  // every hybrid optimization profile. The input/output buffers were allocated
  // for the maximum M; each profile safely operates on their row prefix, and
  // the final real-M launch overwrites the full output.
  if (Fp4PlanCacheEnabled() && g_fp4_warmup_max_tokens != 0 &&
      params.m == static_cast<int>(g_fp4_warmup_max_tokens)) {
    g_fp4_warmup_observed = true;
    const bool full_tactics = Fp4FullTacticsEnabled();
    const nvfp4::PlanKey current_key =
        MakePlanKey(params, device_ordinal, full_tactics);
    SelectedPlan current_plan;
    for (const uint32_t bucket :
         nvfp4::HybridMTuningBuckets(g_fp4_warmup_max_tokens)) {
      nvfp4::LaunchParams profile = params;
      profile.m = static_cast<int>(bucket);
      const nvfp4::PlanKey profile_key =
          MakePlanKey(profile, device_ordinal, full_tactics);
      std::optional<SelectedPlan> plan = PlanCache().FindReady(profile_key);
      if (!plan.has_value()) {
        g_fp4_warmup_profiles_requested.fetch_add(1,
                                                  std::memory_order_relaxed);
        plan = ResolveSelectedPlan(profile, device_ordinal,
                                   /*report_lazy_miss=*/false);
        g_fp4_warmup_profiles_tuned.fetch_add(1,
                                              std::memory_order_relaxed);
      }
      if (profile_key == current_key) current_plan = *plan;
    }
    if (current_plan.candidate_index < 0) {
      current_plan = ResolveSelectedPlan(params, device_ordinal,
                                         /*report_lazy_miss=*/false);
    }
    return current_plan;
  }
  return ResolveSelectedPlan(params, device_ordinal,
                             /*report_lazy_miss=*/true);
}

void RunFixedW1(nvfp4::LaunchParams params) {
  const auto& candidates = LegacyCandidates();
  const size_t index = params.m <= 256 ? 0 : 1;
  const size_t workspace_bytes = candidates[index].workspace_size(params);
  WorkspaceLease workspace = AcquireWorkspace(workspace_bytes, params.stream);
  try {
    RunCandidate(candidates[index], params, workspace, workspace_bytes);
    ReleaseWorkspace(workspace, params.stream);
  } catch (...) {
    ReleaseWorkspace(workspace, params.stream);
    throw;
  }
}

void Fp4GemmRunBf16(void* d, const void* a, const void* b, const void* a_sf, const void* b_sf,
                    const float* alpha, int m, int n, int k, int device_ordinal,
                    cudaStream_t stream) {
  nvfp4::LaunchParams params{d, a, b, a_sf, b_sf, alpha, m, n, k, nullptr, 0, stream};
  if (!Fp4AutotuneEnabled() && Fp4ForcedTactic() < 0) {
    RunFixedW1(params);
    return;
  }

  const SelectedPlan plan = SelectPlan(params, device_ordinal);
  const auto& candidates = ActiveCandidates();
  if (plan.candidate_index < 0 ||
      static_cast<size_t>(plan.candidate_index) >= candidates.size() ||
      candidates[static_cast<size_t>(plan.candidate_index)].descriptor.id != plan.tactic_id) {
    throw std::runtime_error("NVFP4 selected plan does not match the active tactic ABI");
  }
  WorkspaceLease workspace = AcquireWorkspace(plan.workspace_bytes, stream);
  try {
    RunCandidate(candidates[static_cast<size_t>(plan.candidate_index)], params, workspace,
                 plan.workspace_bytes);
    ReleaseWorkspace(workspace, stream);
  } catch (...) {
    ReleaseWorkspace(workspace, stream);
    throw;
  }
}

// Linear fp8 block scale [rows, cols] -> swizzled [Mp=round_up(rows,128),
// Kp=round_up(cols,4)] in Sm1xxBlkScaledConfig's scale-factor atom layout.
__global__ void SwizzleBlockscaleKernel(uint8_t* dst, const uint8_t* src, int rows, int cols, int mp,
                                        int kp) {
  const int total = mp * kp;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int row = index / kp;
  const int col = index % kp;
  uint8_t value = 0;
  if (row < rows && col < cols) value = src[row * cols + col];
  const int a0 = row / 128;
  const int a1 = (row % 128) / 32;
  const int a2 = (row % 128) % 32;
  const int a3 = col / 4;
  const int a4 = col % 4;
  const int destination = ((((a0 * (kp / 4) + a3) * 32 + a2) * 4 + a1) * 4 + a4);
  dst[destination] = value;
}

void SwizzleBlockscaleKernelCuda(Queue& q, Tensor& out_swizzled, const Tensor& in_linear) {
  const int rows = static_cast<int>(in_linear.shape[0]);
  const int cols = static_cast<int>(in_linear.shape[1]);
  const int mp = static_cast<int>(out_swizzled.shape[0]);
  const int kp = static_cast<int>(out_swizzled.shape[1]);
  cudaStream_t stream = AsStream(q);
  const int total = mp * kp;
  if (total == 0) return;
  constexpr int kBlock = 256;
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
  SwizzleBlockscaleKernel<<<grid, kBlock, 0, stream>>>(out_swizzled.Ptr<uint8_t>(),
                                                       in_linear.Ptr<uint8_t>(), rows, cols, mp, kp);
  Check(cudaGetLastError(), "swizzle_blockscale kernel launch");
}

__global__ void SetScalar(float* value, float scalar) { *value = scalar; }

__global__ void CastBf16ToF32Kernel(float* out, const __nv_bfloat16* in, int64_t size) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += step) {
    out[i] = __bfloat162float(in[i]);
  }
}

void MatmulNvfp4CutlassKernelCuda(Queue& q, Tensor& out, const Tensor& a_packed,
                                  const Tensor& a_sf_sw, const Tensor& b_packed,
                                  const Tensor& b_sf_sw,
                                  const Tensor* alpha_device,
                                  float alpha_host) {
  const int m = static_cast<int>(a_packed.shape[0]);
  const int k = static_cast<int>(a_packed.shape[1] * 2);
  const int n = static_cast<int>(b_packed.shape[0]);
  cudaStream_t stream = AsStream(q);

  const bool pool = CutlassPoolEnabled();
  const float* device_alpha = nullptr;
  float* staged_alpha = nullptr;
  if (alpha_device != nullptr) {
    device_alpha = alpha_device->Ptr<float>();
  } else {
    if (pool) {
      staged_alpha = PersistentAlpha(stream);
    } else {
      Check(cudaMallocAsync(&staged_alpha, sizeof(float), stream),
            "cudaMallocAsync alpha");
    }
    SetScalar<<<1, 1, 0, stream>>>(staged_alpha, alpha_host);
    device_alpha = staged_alpha;
  }

  const bool out_f32 = out.dtype == DType::kF32;
  void* device_out = out.data;
  void* bf16_scratch = nullptr;
  if (out_f32) {
    const size_t required = static_cast<size_t>(m) * n * sizeof(__nv_bfloat16);
    if (pool) {
      StreamScratch& scratch = ScratchFor(stream);
      bf16_scratch = EnsureScratch(&scratch.bf16, &scratch.bf16_bytes, required, stream,
                                   "cudaMallocAsync bf16 scratch");
    } else {
      Check(cudaMallocAsync(&bf16_scratch, required, stream), "cudaMallocAsync bf16 scratch");
    }
    device_out = bf16_scratch;
  }

  Fp4GemmRunBf16(device_out, a_packed.data, b_packed.data, a_sf_sw.data, b_sf_sw.data,
                 device_alpha, m, n, k, q.device.index, stream);

  if (out_f32) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int blocks = static_cast<int>((total + 255) / 256);
    CastBf16ToF32Kernel<<<blocks, 256, 0, stream>>>(
        static_cast<float*>(out.data), static_cast<const __nv_bfloat16*>(bf16_scratch), total);
    if (!pool) Check(cudaFreeAsync(bf16_scratch, stream), "cudaFreeAsync bf16 scratch");
  }

  if (!pool && staged_alpha != nullptr) {
    Check(cudaFreeAsync(staged_alpha, stream), "cudaFreeAsync alpha");
  }
  Check(cudaGetLastError(), "matmul_nvfp4_cutlass launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kSwizzleBlockscale, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<SwizzleBlockscaleFn>(&SwizzleBlockscaleKernelCuda)));
    RegisterOp(OpId::kMatmulNvfp4Cutlass, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MatmulNvfp4CutlassFn>(&MatmulNvfp4CutlassKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace

Nvfp4AutotuneWarmupScope::Nvfp4AutotuneWarmupScope(
    uint32_t max_num_tokens)
    : Nvfp4AutotuneWarmupScope(max_num_tokens, -1) {}

Nvfp4AutotuneWarmupScope::Nvfp4AutotuneWarmupScope(
    uint32_t max_num_tokens, int device_ordinal)
    : max_num_tokens_(max_num_tokens) {
  if (max_num_tokens == 0) {
    throw std::invalid_argument(
        "NVFP4 pre-serve warmup requires a positive maximum token count");
  }
  if (g_fp4_warmup_max_tokens != 0) {
    throw std::logic_error("nested NVFP4 pre-serve warmup scopes are unsupported");
  }
  if (device_ordinal >= 0 && Fp4FullTacticsEnabled()) {
    persistent_session_ = BeginPersistentRuntime(device_ordinal);
  }
  requested_before_ =
      g_fp4_warmup_profiles_requested.load(std::memory_order_relaxed);
  tuned_before_ = g_fp4_warmup_profiles_tuned.load(std::memory_order_relaxed);
  g_fp4_warmup_complete.store(false, std::memory_order_release);
  g_fp4_warmup_active_scopes.fetch_add(1, std::memory_order_acq_rel);
  g_fp4_warmup_scopes_started.fetch_add(1, std::memory_order_relaxed);
  g_fp4_warmup_max_tokens = max_num_tokens;
  g_fp4_warmup_observed = false;
  active_ = true;
}

Nvfp4AutotuneWarmupScope::~Nvfp4AutotuneWarmupScope() {
  if (!active_) return;
  g_fp4_warmup_max_tokens = 0;
  g_fp4_warmup_observed = false;
  g_fp4_warmup_active_scopes.fetch_sub(1, std::memory_order_acq_rel);
  if (!completed_) {
    g_fp4_warmup_complete.store(false, std::memory_order_release);
  }
  if (persistent_session_) EndPersistentRuntime(completed_);
}

void Nvfp4AutotuneWarmupScope::Complete() {
  if (!active_ || completed_) {
    throw std::logic_error("NVFP4 pre-serve warmup scope completed out of order");
  }
  if (!g_fp4_warmup_observed) {
    throw std::runtime_error(
        "NVFP4 pre-serve warmup completed without observing a maximum-token "
        "W4A4 GEMM");
  }
  const uint64_t requested =
      g_fp4_warmup_profiles_requested.load(std::memory_order_relaxed) -
      requested_before_;
  const uint64_t tuned =
      g_fp4_warmup_profiles_tuned.load(std::memory_order_relaxed) - tuned_before_;
  if (persistent_session_) CompletePersistentRuntime();
  completed_ = true;
  g_fp4_warmup_scopes_completed.fetch_add(1, std::memory_order_relaxed);
  g_fp4_warmup_complete.store(true, std::memory_order_release);
  std::fprintf(stderr,
               "[VT_FP4_AUTOTUNE] pre-serve warmup complete max_tokens=%u "
               "profiles_requested=%llu profiles_tuned=%llu cached_plans=%zu\n",
               max_num_tokens_, static_cast<unsigned long long>(requested),
               static_cast<unsigned long long>(tuned),
               PlanCache().SizeForTesting());
}

Nvfp4AutotuneWarmupStats GetNvfp4AutotuneWarmupStats() {
  Nvfp4AutotuneWarmupStats stats;
  stats.scopes_started =
      g_fp4_warmup_scopes_started.load(std::memory_order_relaxed);
  stats.scopes_completed =
      g_fp4_warmup_scopes_completed.load(std::memory_order_relaxed);
  stats.profiles_requested =
      g_fp4_warmup_profiles_requested.load(std::memory_order_relaxed);
  stats.profiles_tuned =
      g_fp4_warmup_profiles_tuned.load(std::memory_order_relaxed);
  stats.lazy_misses = g_fp4_lazy_misses.load(std::memory_order_relaxed);
  stats.profiles_loaded =
      g_fp4_profiles_loaded.load(std::memory_order_relaxed);
  stats.cache_documents_rejected =
      g_fp4_cache_documents_rejected.load(std::memory_order_relaxed);
  stats.profiles_saved =
      g_fp4_profiles_saved.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(g_fp4_persistent_mutex);
  stats.delay_microseconds =
      g_fp4_persistent_diagnostics.delay_microseconds;
  stats.persistent_cache_enabled = g_fp4_persistent_diagnostics.enabled;
  stats.read_only = g_fp4_persistent_diagnostics.read_only;
  stats.mode = g_fp4_persistent_diagnostics.mode;
  stats.native_path = g_fp4_persistent_diagnostics.native_path;
  stats.flashinfer_path = g_fp4_persistent_diagnostics.flashinfer_path;
  stats.metadata_fingerprint =
      g_fp4_persistent_diagnostics.metadata_fingerprint;
  stats.selected_plans = g_fp4_persistent_diagnostics.selected_plans;
  return stats;
}

}  // namespace vt::cuda
