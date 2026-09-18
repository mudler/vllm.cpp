// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Tier-A4 fold (arch-fusion-fold-plan-2026-07-30.md) — the SHARED fused BF16
// grouped-MoE gate+up+SwiGLU op vt::MoeGroupedGemmBf16GateUpSilu
// (OpId::kMoeGroupedGemmBf16GateUpSilu), the bf16-native arm of the routed-MoE
// gate+up+SwiGLU family. The op REPLACES the {MoeGroupedGemmBf16(gate f32);
// MoeGroupedGemmBf16(up f32); MoeSiluMul(->bf16)} triplet the bf16 grouped-MoE
// archs (Qwen3-Coder, DeepSeek-V2) ran, and the fold plan's Iron Law requires it
// be BIT-IDENTICAL to that composite in every launch regime.
//
// This is an A/B gate: the fused op's bf16 output is compared BYTE-FOR-BYTE (raw
// uint16 bits, not a tolerance) against that exact composite computed with the
// same vt ops. RED-first — the composite is the golden; a fused kernel that (e.g.)
// dropped the sigmoid, reduced the split-K partials in the wrong order, or rounded
// the intermediate through bf16 would diverge in the low mantissa bits and fail
// the memcmp. Covers all three dispatch regimes LaunchGroupedBf16 selects between:
//   * P < kTileMinRows(32), splits==1   -> naive partials + fused reduce+SwiGLU;
//   * P < kTileMinRows,     splits>1    -> split-K partials + fused reduce+SwiGLU;
//   * P >= kTileMinRows                 -> WMMA composite (2x grouped GEMM + silu).
// CUDA-only (skips cleanly with no GPU) — the op is registered for kCUDA only.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// Preserve the CUDA cases and allow the same fixtures on the ROCm provider.
DeviceType TestDevice() {
  const char* device = std::getenv("VT_MOE_TEST_DEVICE");
  return device != nullptr && std::strcmp(device, "rocm") == 0
             ? DeviceType::kROCM : DeviceType::kCUDA;
}
bool HasDevice() {
  try {
    vt::GetBackend(TestDevice());
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, q.device, shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void* ptr() { return p_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

std::vector<uint16_t> RandomBf16(size_t numel, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<uint16_t> v(numel);
  for (auto& x : v) x = vt::F32ToBF16(dist(rng));
  return v;
}

// Runs the fused gate+up+SwiGLU op and the {2x MoeGroupedGemmBf16 (f32 out) +
// MoeSiluMul (bf16 out)} composite it replaces over E synthetic bf16 [K,I]
// gate/up experts, and asserts the fused bf16 output is BYTE-IDENTICAL to the
// composite (raw uint16 bits). `t_rows`/`top_k`/`k_dim`/`i_dim` pick the launch
// regime (P = t_rows*top_k relative to kTileMinRows, and K relative to the
// split-K threshold). Uses the pair->token row_map the production gate/up path
// carries.
void RunFusedGateUpSiluCase(int64_t e_count, int64_t t_rows, int64_t top_k, int64_t k_dim,
                            int64_t i_dim, uint32_t seed) {
  Backend& gpu = vt::GetBackend(TestDevice());
  const int64_t P = t_rows * top_k;

  std::vector<std::vector<uint16_t>> gate_w(static_cast<size_t>(e_count));
  std::vector<std::vector<uint16_t>> up_w(static_cast<size_t>(e_count));
  for (int64_t e = 0; e < e_count; ++e) {
    gate_w[static_cast<size_t>(e)] =
        RandomBf16(static_cast<size_t>(k_dim * i_dim), seed + 17 * static_cast<uint32_t>(e));
    up_w[static_cast<size_t>(e)] =
        RandomBf16(static_cast<size_t>(k_dim * i_dim), seed + 5099 + 17 * static_cast<uint32_t>(e));
  }
  const auto act = RandomBf16(static_cast<size_t>(t_rows * k_dim), seed + 991);

  std::vector<int32_t> expert_ids(static_cast<size_t>(P));
  std::vector<int32_t> row_map(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p) {
    row_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
    expert_ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 7 + 3) % e_count);
  }

  QueueGuard gq(gpu);
  DeviceTensor dact(gpu, gq.q, DType::kBF16, {t_rows, k_dim}, act.data());
  DeviceTensor deids(gpu, gq.q, DType::kI32, {P}, expert_ids.data());
  DeviceTensor drow(gpu, gq.q, DType::kI32, {P}, row_map.data());

  std::vector<std::unique_ptr<DeviceTensor>> gbufs, ubufs;
  std::vector<int64_t> gptrs(static_cast<size_t>(e_count)), uptrs(static_cast<size_t>(e_count));
  for (int64_t e = 0; e < e_count; ++e) {
    gbufs.push_back(std::make_unique<DeviceTensor>(gpu, gq.q, DType::kBF16,
                                                   std::vector<int64_t>{k_dim, i_dim},
                                                   gate_w[static_cast<size_t>(e)].data()));
    ubufs.push_back(std::make_unique<DeviceTensor>(gpu, gq.q, DType::kBF16,
                                                   std::vector<int64_t>{k_dim, i_dim},
                                                   up_w[static_cast<size_t>(e)].data()));
    gptrs[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(gbufs.back()->ptr());
    uptrs[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(ubufs.back()->ptr());
  }
  DeviceTensor dgp(gpu, gq.q, DType::kI64, {e_count}, gptrs.data());
  DeviceTensor dup(gpu, gq.q, DType::kI64, {e_count}, uptrs.data());

  // --- Composite golden: 2x grouped bf16 GEMM (f32) + MoeSiluMul (bf16) --------
  DeviceTensor dgate_f32(gpu, gq.q, DType::kF32, {P, i_dim});
  DeviceTensor dup_f32(gpu, gq.q, DType::kF32, {P, i_dim});
  vt::MoeGroupedGemmBf16(gq.q, dgate_f32.tensor(), dact.tensor(), deids.tensor(), &drow.tensor(),
                         dgp.tensor());
  vt::MoeGroupedGemmBf16(gq.q, dup_f32.tensor(), dact.tensor(), deids.tensor(), &drow.tensor(),
                         dup.tensor());
  DeviceTensor dref(gpu, gq.q, DType::kBF16, {P, i_dim});
  vt::MoeSiluMul(gq.q, dref.tensor(), dgate_f32.tensor(), dup_f32.tensor());
  std::vector<uint16_t> ref(static_cast<size_t>(P * i_dim));
  dref.Download(gq.q, ref.data());

  // --- Fused op under test ----------------------------------------------------
  DeviceTensor dfused(gpu, gq.q, DType::kBF16, {P, i_dim});
  vt::MoeGroupedGemmBf16GateUpSilu(gq.q, dfused.tensor(), dact.tensor(), deids.tensor(),
                                   &drow.tensor(), dgp.tensor(), dup.tensor());
  std::vector<uint16_t> got(static_cast<size_t>(P * i_dim));
  dfused.Download(gq.q, got.data());

  // BYTE-IDENTICAL: the fused op must reproduce the composite's exact bf16 bits.
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i] != ref[i]) {
      if (bad == 0) first = i;
      ++bad;
    }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first);
    CAPTURE(static_cast<int>(got[first]));
    CAPTURE(static_cast<int>(ref[first]));
  }
  CHECK(bad == 0);
}

}  // namespace

// P = 12 < kTileMinRows(32), K=64 -> splits==1 (naive partials) + reduce+SwiGLU.
TEST_CASE("GPU moe_grouped_gemm_bf16_gate_up_silu naive path == composite (byte-exact)") {
  if (!HasDevice()) {
    MESSAGE("requested GPU backend is unavailable; skipping");
    return;
  }
  RunFusedGateUpSiluCase(/*e_count=*/5, /*t_rows=*/6, /*top_k=*/2, /*k_dim=*/64, /*i_dim=*/8,
                         /*seed=*/9000);
}

// P = 6 < kTileMinRows, K=1024 -> MoeSplitKCount picks splits=4: split-K partials
// reduced in fixed ascending order, then fused SwiGLU. The path where the fold's
// launch reduction (5 -> 3) actually lands (Qwen3-Coder / DeepSeek-V2 c1 decode).
TEST_CASE("GPU moe_grouped_gemm_bf16_gate_up_silu split-K decode == composite (byte-exact)") {
  if (!HasDevice()) {
    MESSAGE("requested GPU backend is unavailable; skipping");
    return;
  }
  RunFusedGateUpSiluCase(/*e_count=*/5, /*t_rows=*/3, /*top_k=*/2, /*k_dim=*/1024, /*i_dim=*/8,
                         /*seed=*/9100);
  // Ragged N crossing the reduce block boundary, still split-K.
  RunFusedGateUpSiluCase(/*e_count=*/7, /*t_rows=*/4, /*top_k=*/2, /*k_dim=*/768, /*i_dim=*/130,
                         /*seed=*/9150);
}

// P = 40 (kTileMinRows <= P) -> BM=16 decode WMMA tile; P = 1024 -> BM=64 prefill
// WMMA tile. The WMMA branch reuses the tuned grouped GEMM twice + the identical
// silu-mul, so it is composite-identical by construction — this asserts it.
TEST_CASE("GPU moe_grouped_gemm_bf16_gate_up_silu WMMA tiles == composite (byte-exact)") {
  if (!HasDevice()) {
    MESSAGE("requested GPU backend is unavailable; skipping");
    return;
  }
  RunFusedGateUpSiluCase(/*e_count=*/7, /*t_rows=*/20, /*top_k=*/2, /*k_dim=*/80, /*i_dim=*/130,
                         /*seed=*/9200);
  RunFusedGateUpSiluCase(/*e_count=*/9, /*t_rows=*/128, /*top_k=*/8, /*k_dim=*/96, /*i_dim=*/70,
                         /*seed=*/9300);
  // Aligned pitches -> pipelined WMMA tiles (prefill BM=64 and decode BM=16).
  RunFusedGateUpSiluCase(/*e_count=*/9, /*t_rows=*/128, /*top_k=*/8, /*k_dim=*/264, /*i_dim=*/256,
                         /*seed=*/9400);
}
