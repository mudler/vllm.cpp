// vllm.cpp — SM120/SM121 NVFP4 block-scaled FP4xFP4 GEMM adapter.
//
// The raw kernels are split across cuda_nvfp4_tactics_*.cu. The full default
// surface mirrors installed FlashInfer 0.6.12
// `fp4_gemm_cutlass_template_sm120.h:47-220`: 8 CTA tiles x swap-AB x
// StaticPersistent/Stream-K in exact upstream order. The immutable W1 four-
// candidate implementation is retained separately for VT_FP4_FULL_TACTICS=0.
// This TU owns vt::Tensor validation/registration, exact hybrid-bucket tuning,
// high-water workspace reuse, forced-tactic diagnostics and scale swizzling.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vt/cuda/nvfp4_cutlass_tactics.h"
#include "vt/cuda/nvfp4_plan_cache.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

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

SelectedPlan SelectPlan(nvfp4::LaunchParams params, int device_ordinal) {
  const auto& candidates = ActiveCandidates();
  const bool full_tactics = Fp4FullTacticsEnabled();
  const uint32_t m_bucket = Fp4ExactBucketsEnabled()
                                ? nvfp4::HybridMBucket(static_cast<uint32_t>(params.m))
                                : nvfp4::LegacyMBucket(static_cast<uint32_t>(params.m));
  const nvfp4::PlanKey key{m_bucket,
                           params.n,
                           params.k,
                           device_ordinal,
                           Fp4DeviceArchitecture(device_ordinal),
                           static_cast<uint8_t>(DType::kBF16),
                           full_tactics ? nvfp4::kFullTacticSetVersion
                                        : nvfp4::kW1TacticSetVersion};
  static nvfp4::SingleFlightPlanCache<SelectedPlan> plans;

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
      int baseline = (m_bucket <= 256) ? 0 : 1;
      if (!full_tactics && timings[static_cast<size_t>(baseline)] < kFp4Inf &&
          !(timings[static_cast<size_t>(best)] <
            timings[static_cast<size_t>(baseline)] * 0.99F)) {
        chosen = baseline;
      }
      if (Fp4AutotuneVerbose()) {
        std::fprintf(stderr,
                     "[VT_FP4_AUTOTUNE] set=%s M=%d(bucket=%u) N=%d K=%d device=%d sm=%d "
                     "-> id=%d %s (%.1f us), workspace=%zu\n",
                     full_tactics ? "full" : "w1", params.m, m_bucket, params.n, params.k,
                     device_ordinal, key.architecture, candidates[static_cast<size_t>(chosen)].descriptor.id,
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
                                  const Tensor& b_sf_sw, float alpha) {
  const int m = static_cast<int>(a_packed.shape[0]);
  const int k = static_cast<int>(a_packed.shape[1] * 2);
  const int n = static_cast<int>(b_packed.shape[0]);
  cudaStream_t stream = AsStream(q);

  const bool pool = CutlassPoolEnabled();
  float* device_alpha = nullptr;
  if (pool) {
    device_alpha = PersistentAlpha(stream);
  } else {
    Check(cudaMallocAsync(&device_alpha, sizeof(float), stream), "cudaMallocAsync alpha");
  }
  SetScalar<<<1, 1, 0, stream>>>(device_alpha, alpha);

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

  if (!pool) Check(cudaFreeAsync(device_alpha, stream), "cudaFreeAsync alpha");
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
}  // namespace vt::cuda
