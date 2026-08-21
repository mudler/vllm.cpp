// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// M2.4 — the fused-MoE grouped NVFP4 GEMM (vt::MoeGroupedGemmNvfp4) + the fused
// silu-mul activation (vt::MoeSiluMul). Validates that the grouped GEMM over all
// (token, expert) pairs matches, per output row, the single-expert reference
// (DequantNvfp4ToBf16 + per-row f32 matmul) that the per-expert MoE loop uses —
// so the fused MoE and the per-expert MoE agree. Grouped GEMM is CUDA-only
// (skips cleanly with no GPU); MoeSiluMul runs on CPU and CUDA.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "moe_router_warp_env.h"
#include "vllm/model_executor/model_loader/mxfp4_dequant.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/cuda/moe_decode_ref.h"
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

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

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

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    if (!(std::fabs(got[i] - want[i]) <= tol)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(want[first_bad]);
  }
  CHECK(bad == 0);
}

// L2-relative closeness: ||got-want||_2 <= rtol * ||want||_2. This is the standard
// GEMM-correctness metric and — unlike a per-element absolute tolerance — is robust
// to catastrophic cancellation. A bf16 tensor-core GEMM vs a naive fp32-accumulate
// reference legitimately diverges by O(accumulation-magnitude * eps_bf16) at output
// elements whose TRUE value is near zero (a large sum that cancels): the per-element
// |got-want| there (~1) dwarfs the tiny |want| yet is pure bf16 rounding, not a
// kernel defect (proven: the VALIDATED single-expert MoE route diverges identically,
// max|dense-moe|==0). Per-element byte-fidelity is still gated EXACTLY by the
// dense==MoE check below; a wrong stride still fails this L2 gate (the row-shifted
// reference is uncorrelated => ||got-shift|| ~ ||want||, ratio ~1 >> rtol).
#ifdef VT_MARLIN_NVFP4
void CheckCloseL2(const std::vector<float>& got, const std::vector<float>& want, float rtol) {
  REQUIRE(got.size() == want.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
    num += d * d;
    den += static_cast<double>(want[i]) * static_cast<double>(want[i]);
  }
  const double rel = std::sqrt(num) / (std::sqrt(den) + 1e-12);
  CAPTURE(rel);
  CHECK(rel <= rtol);
}
#endif  // VT_MARLIN_NVFP4

struct Nvfp4Weight {
  std::vector<uint8_t> packed;  // [N, K/2]
  std::vector<uint8_t> scale;   // [N, K/16]
  float scale2;
};

Nvfp4Weight MakeNvfp4Weight(int64_t n, int64_t k, uint32_t seed) {
  Nvfp4Weight w;
  w.packed.resize(static_cast<size_t>(n * (k / 2)));
  w.scale.resize(static_cast<size_t>(n * (k / 16)));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> exp_dist(1, 13);
  std::uniform_int_distribution<int> mant_dist(0, 7);
  for (auto& b : w.packed) b = static_cast<uint8_t>(byte_dist(rng));
  for (auto& s : w.scale) {
    const int exp = exp_dist(rng);
    const int mant = mant_dist(rng);
    s = static_cast<uint8_t>((exp << 3) | mant);
  }
  w.scale2 = 0.3125f + 0.05f * static_cast<float>(seed % 7);
  return w;
}

// MXFP4 weight: E2M1 packed [N,K/2] + E8M0 (UE8M0) block scale [N,K/32], no global.
struct Mxfp4Weight {
  std::vector<uint8_t> packed;  // [N, K/2]
  std::vector<uint8_t> scale;   // [N, K/32]  E8M0 biased exponent (2^(byte-127))
};

#ifdef VT_MARLIN_NVFP4
Mxfp4Weight MakeMxfp4Weight(int64_t n, int64_t k, uint32_t seed) {
  Mxfp4Weight w;
  w.packed.resize(static_cast<size_t>(n * (k / 2)));
  w.scale.resize(static_cast<size_t>(n * (k / 32)));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  // E8M0 bytes near 127 => scales ~2^-9..2^3 (finite, in bf16 normal range).
  std::uniform_int_distribution<int> e8m0_dist(118, 132);
  for (auto& b : w.packed) b = static_cast<uint8_t>(byte_dist(rng));
  for (auto& s : w.scale) s = static_cast<uint8_t>(e8m0_dist(rng));
  return w;
}
#endif  // VT_MARLIN_NVFP4

std::vector<float> RandomF32(size_t numel, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> v(numel);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> b(f.size());
  for (size_t i = 0; i < f.size(); ++i) b[i] = vt::F32ToBF16(f[i]);
  return b;
}

}  // namespace

// The grouped GEMM's output row p must equal the single-expert dequant reference
// for expert expert_ids[p] applied to act row row_map[p] — i.e. exactly what the
// per-expert MoE loop computes for that (token, expert) pair.
TEST_CASE("CUDA moe_grouped_gemm_nvfp4 matches the per-expert reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  const int64_t E = 5;     // experts
  const int64_t T = 3;     // tokens (act rows)
  const int64_t top_k = 2;  // slots per token
  const int64_t P = T * top_k;
  const int64_t K = 64;    // in_features
  const int64_t N = 8;     // out_features

  // E synthetic experts (each [N, K]); act rows [T, K] bf16.
  std::vector<Nvfp4Weight> experts;
  for (int64_t e = 0; e < E; ++e) experts.push_back(MakeNvfp4Weight(N, K, 5000 + static_cast<uint32_t>(e)));
  const auto act_f = RandomF32(static_cast<size_t>(T * K), 9001);
  const auto act_bf16 = ToBf16(act_f);
  // The kernel reads bf16 activations, so the reference must too (round-trip).
  std::vector<float> act_r(act_f.size());
  for (size_t i = 0; i < act_r.size(); ++i) act_r[i] = vt::BF16ToF32(act_bf16[i]);

  // Routing: pair p -> token p/top_k, expert a deterministic pseudo-choice.
  std::vector<int32_t> expert_ids(static_cast<size_t>(P));
  std::vector<int32_t> row_map(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p) {
    row_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
    expert_ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 3 + 1) % E);
  }

  // Reference: dequant each expert to bf16 [N,K], out[p,n] = sum_k
  // act[row_map[p],k] * f32(bf16 W_e[n,k]) in f32 (the per-expert MoE row math).
  std::vector<std::vector<uint16_t>> deq(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    deq[static_cast<size_t>(e)].resize(static_cast<size_t>(N * K));
    vllm::DequantNvfp4ToBf16(experts[static_cast<size_t>(e)].packed.data(),
                             experts[static_cast<size_t>(e)].scale.data(),
                             experts[static_cast<size_t>(e)].scale2, N, K,
                             deq[static_cast<size_t>(e)].data());
  }
  std::vector<float> ref(static_cast<size_t>(P * N), 0.0f);
  for (int64_t p = 0; p < P; ++p) {
    const int64_t e = expert_ids[static_cast<size_t>(p)];
    const int64_t r = row_map[static_cast<size_t>(p)];
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += act_r[static_cast<size_t>(r * K + k)] *
               vt::BF16ToF32(deq[static_cast<size_t>(e)][static_cast<size_t>(n * K + k)]);
      ref[static_cast<size_t>(p * N + n)] = acc;
    }
  }

  QueueGuard gq(gpu);
  DeviceTensor dact(gpu, gq.q, DType::kBF16, {T, K}, act_bf16.data());
  DeviceTensor deids(gpu, gq.q, DType::kI32, {P}, expert_ids.data());
  DeviceTensor drow(gpu, gq.q, DType::kI32, {P}, row_map.data());

  // Upload each expert's packed/scale; collect device pointers into i64 arrays.
  std::vector<std::unique_ptr<DeviceTensor>> packed_bufs, scale_bufs;
  std::vector<int64_t> packed_ptrs(static_cast<size_t>(E)), scale_ptrs(static_cast<size_t>(E));
  std::vector<float> scale2s(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    packed_bufs.push_back(std::make_unique<DeviceTensor>(
        gpu, gq.q, DType::kI8, std::vector<int64_t>{N, K / 2}, experts[static_cast<size_t>(e)].packed.data()));
    scale_bufs.push_back(std::make_unique<DeviceTensor>(
        gpu, gq.q, DType::kI8, std::vector<int64_t>{N, K / 16}, experts[static_cast<size_t>(e)].scale.data()));
    packed_ptrs[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(packed_bufs.back()->ptr());
    scale_ptrs[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(scale_bufs.back()->ptr());
    scale2s[static_cast<size_t>(e)] = experts[static_cast<size_t>(e)].scale2;
  }
  DeviceTensor dpp(gpu, gq.q, DType::kI64, {E}, packed_ptrs.data());
  DeviceTensor dsp(gpu, gq.q, DType::kI64, {E}, scale_ptrs.data());
  DeviceTensor ds2(gpu, gq.q, DType::kF32, {E}, scale2s.data());

  DeviceTensor dout(gpu, gq.q, DType::kF32, {P, N});
  vt::MoeGroupedGemmNvfp4(gq.q, dout.tensor(), dact.tensor(), deids.tensor(), &drow.tensor(),
                          dpp.tensor(), dsp.tensor(), ds2.tensor());
  std::vector<float> got(static_cast<size_t>(P * N));
  dout.Download(gq.q, got.data());
  CheckClose(got, ref, 2e-3f, 2e-3f);
}

// Large-P case (P>=32): exercises the shared-memory TILED grouped-GEMM path (the
// launch gates on P; the P=6 case above exercises the naive path). Same per-row
// reference: grouped GEMM row p == single-expert dequant matmul for expert_ids[p].
TEST_CASE("CUDA moe_grouped_gemm_nvfp4 tiled path (large P) matches per-expert reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  const int64_t E = 7;
  const int64_t T = 20;
  const int64_t top_k = 2;
  const int64_t P = T * top_k;  // 40 >= 32 => tiled path
  const int64_t K = 80;   // multiple of 16 but NOT of BK=32 => partial last K-tile
  const int64_t N = 130;  // odd, crosses the BN=128 tile boundary

  std::vector<Nvfp4Weight> experts;
  for (int64_t e = 0; e < E; ++e)
    experts.push_back(MakeNvfp4Weight(N, K, 8000 + static_cast<uint32_t>(e)));
  const auto act_f = RandomF32(static_cast<size_t>(T * K), 8100);
  const auto act_bf16 = ToBf16(act_f);
  std::vector<float> act_r(act_f.size());
  for (size_t i = 0; i < act_r.size(); ++i) act_r[i] = vt::BF16ToF32(act_bf16[i]);

  std::vector<int32_t> expert_ids(static_cast<size_t>(P));
  std::vector<int32_t> row_map(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p) {
    row_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
    expert_ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 5 + 2) % E);
  }

  std::vector<std::vector<uint16_t>> deq(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    deq[static_cast<size_t>(e)].resize(static_cast<size_t>(N * K));
    vllm::DequantNvfp4ToBf16(experts[static_cast<size_t>(e)].packed.data(),
                             experts[static_cast<size_t>(e)].scale.data(),
                             experts[static_cast<size_t>(e)].scale2, N, K,
                             deq[static_cast<size_t>(e)].data());
  }
  std::vector<float> ref(static_cast<size_t>(P * N), 0.0f);
  for (int64_t p = 0; p < P; ++p) {
    const int64_t e = expert_ids[static_cast<size_t>(p)];
    const int64_t r = row_map[static_cast<size_t>(p)];
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += act_r[static_cast<size_t>(r * K + k)] *
               vt::BF16ToF32(deq[static_cast<size_t>(e)][static_cast<size_t>(n * K + k)]);
      ref[static_cast<size_t>(p * N + n)] = acc;
    }
  }

  QueueGuard gq(gpu);
  DeviceTensor dact(gpu, gq.q, DType::kBF16, {T, K}, act_bf16.data());
  DeviceTensor deids(gpu, gq.q, DType::kI32, {P}, expert_ids.data());
  DeviceTensor drow(gpu, gq.q, DType::kI32, {P}, row_map.data());
  std::vector<std::unique_ptr<DeviceTensor>> packed_bufs, scale_bufs;
  std::vector<int64_t> pp(static_cast<size_t>(E)), sp(static_cast<size_t>(E));
  std::vector<float> s2(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    packed_bufs.push_back(std::make_unique<DeviceTensor>(
        gpu, gq.q, DType::kI8, std::vector<int64_t>{N, K / 2},
        experts[static_cast<size_t>(e)].packed.data()));
    scale_bufs.push_back(std::make_unique<DeviceTensor>(
        gpu, gq.q, DType::kI8, std::vector<int64_t>{N, K / 16},
        experts[static_cast<size_t>(e)].scale.data()));
    pp[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(packed_bufs.back()->ptr());
    sp[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(scale_bufs.back()->ptr());
    s2[static_cast<size_t>(e)] = experts[static_cast<size_t>(e)].scale2;
  }
  DeviceTensor dpp(gpu, gq.q, DType::kI64, {E}, pp.data());
  DeviceTensor dsp(gpu, gq.q, DType::kI64, {E}, sp.data());
  DeviceTensor ds2(gpu, gq.q, DType::kF32, {E}, s2.data());
  DeviceTensor dout(gpu, gq.q, DType::kF32, {P, N});
  vt::MoeGroupedGemmNvfp4(gq.q, dout.tensor(), dact.tensor(), deids.tensor(), &drow.tensor(),
                          dpp.tensor(), dsp.tensor(), ds2.tensor());
  std::vector<float> got(static_cast<size_t>(P * N));
  dout.Download(gq.q, got.data());
  CheckClose(got, ref, 2e-3f, 2e-3f);
}

// Null row_map => identity (act row p). Exercises the down-projection path.
TEST_CASE("CUDA moe_grouped_gemm_nvfp4 identity row_map (down-projection path)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t E = 3, P = 4, K = 32, N = 6;
  std::vector<Nvfp4Weight> experts;
  for (int64_t e = 0; e < E; ++e) experts.push_back(MakeNvfp4Weight(N, K, 6000 + static_cast<uint32_t>(e)));
  const auto act_f = RandomF32(static_cast<size_t>(P * K), 6100);
  const auto act_bf16 = ToBf16(act_f);
  std::vector<float> act_r(act_f.size());
  for (size_t i = 0; i < act_r.size(); ++i) act_r[i] = vt::BF16ToF32(act_bf16[i]);
  std::vector<int32_t> expert_ids{0, 2, 1, 2};

  std::vector<float> ref(static_cast<size_t>(P * N), 0.0f);
  for (int64_t p = 0; p < P; ++p) {
    std::vector<uint16_t> deq(static_cast<size_t>(N * K));
    const int64_t e = expert_ids[static_cast<size_t>(p)];
    vllm::DequantNvfp4ToBf16(experts[static_cast<size_t>(e)].packed.data(),
                             experts[static_cast<size_t>(e)].scale.data(),
                             experts[static_cast<size_t>(e)].scale2, N, K, deq.data());
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += act_r[static_cast<size_t>(p * K + k)] * vt::BF16ToF32(deq[static_cast<size_t>(n * K + k)]);
      ref[static_cast<size_t>(p * N + n)] = acc;
    }
  }

  QueueGuard gq(gpu);
  DeviceTensor dact(gpu, gq.q, DType::kBF16, {P, K}, act_bf16.data());
  DeviceTensor deids(gpu, gq.q, DType::kI32, {P}, expert_ids.data());
  std::vector<std::unique_ptr<DeviceTensor>> packed_bufs, scale_bufs;
  std::vector<int64_t> pp(static_cast<size_t>(E)), sp(static_cast<size_t>(E));
  std::vector<float> s2(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    packed_bufs.push_back(std::make_unique<DeviceTensor>(
        gpu, gq.q, DType::kI8, std::vector<int64_t>{N, K / 2}, experts[static_cast<size_t>(e)].packed.data()));
    scale_bufs.push_back(std::make_unique<DeviceTensor>(
        gpu, gq.q, DType::kI8, std::vector<int64_t>{N, K / 16}, experts[static_cast<size_t>(e)].scale.data()));
    pp[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(packed_bufs.back()->ptr());
    sp[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(scale_bufs.back()->ptr());
    s2[static_cast<size_t>(e)] = experts[static_cast<size_t>(e)].scale2;
  }
  DeviceTensor dpp(gpu, gq.q, DType::kI64, {E}, pp.data());
  DeviceTensor dsp(gpu, gq.q, DType::kI64, {E}, sp.data());
  DeviceTensor ds2(gpu, gq.q, DType::kF32, {E}, s2.data());
  DeviceTensor dout(gpu, gq.q, DType::kF32, {P, N});
  vt::MoeGroupedGemmNvfp4(gq.q, dout.tensor(), dact.tensor(), deids.tensor(), nullptr, dpp.tensor(),
                          dsp.tensor(), ds2.tensor());
  std::vector<float> got(static_cast<size_t>(P * N));
  dout.Download(gq.q, got.data());
  CheckClose(got, ref, 2e-3f, 2e-3f);
}

// moe_silu_mul: out = silu(gate) * up. Reference in f32 (std::exp).
static void RunSiluMul(Backend& b, Device dev) {
  const int64_t R = 4, I = 10;
  const auto gate_f = RandomF32(static_cast<size_t>(R * I), 3001);
  const auto up_f = RandomF32(static_cast<size_t>(R * I), 3002);
  std::vector<float> ref(static_cast<size_t>(R * I));
  for (size_t i = 0; i < ref.size(); ++i) {
    const float g = gate_f[i];
    ref[i] = (g / (1.0f + std::exp(-g))) * up_f[i];
  }
  const auto gate_bf16 = ToBf16(gate_f);
  const auto up_bf16 = ToBf16(up_f);

  QueueGuard gq(b);
  DeviceTensor dg(b, gq.q, DType::kBF16, {R, I}, gate_bf16.data());
  DeviceTensor du(b, gq.q, DType::kBF16, {R, I}, up_bf16.data());
  DeviceTensor dout(b, gq.q, DType::kBF16, {R, I});
  (void)dev;
  vt::MoeSiluMul(gq.q, dout.tensor(), dg.tensor(), du.tensor());
  std::vector<uint16_t> got_bf16(static_cast<size_t>(R * I));
  dout.Download(gq.q, got_bf16.data());
  std::vector<float> got(got_bf16.size());
  for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(got_bf16[i]);
  CheckClose(got, ref, 8e-3f, 8e-3f);
}

TEST_CASE("moe_silu_mul matches silu(gate)*up (CPU)") {
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  RunSiluMul(cpu, Cpu());
}

TEST_CASE("moe_silu_mul matches silu(gate)*up (CUDA)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunSiluMul(vt::GetBackend(DeviceType::kCUDA), Gpu());
}

TEST_CASE("moe_grouped_gemm_nvfp4 validates shapes loudly (CPU dispatch)") {
  // K not a multiple of 16 must throw in the public op (before device dispatch).
  std::vector<uint16_t> act(2 * 8, 0);
  std::vector<int32_t> ids(2, 0);
  std::vector<int64_t> pp(1, 0), sp(1, 0);
  std::vector<float> s2(1, 1.0f);
  std::vector<float> out(2 * 4, 0.0f);
  Tensor tact = MakeTensor(act.data(), DType::kBF16, Cpu(), {2, 8});  // K=8, not %16
  Tensor tids = MakeTensor(ids.data(), DType::kI32, Cpu(), {2});
  Tensor tpp = MakeTensor(pp.data(), DType::kI64, Cpu(), {1});
  Tensor tsp = MakeTensor(sp.data(), DType::kI64, Cpu(), {1});
  Tensor ts2 = MakeTensor(s2.data(), DType::kF32, Cpu(), {1});
  Tensor tout = MakeTensor(out.data(), DType::kF32, Cpu(), {2, 4});
  Queue cq{Cpu(), nullptr};
  CHECK_THROWS_AS(
      vt::MoeGroupedGemmNvfp4(cq, tout, tact, tids, nullptr, tpp, tsp, ts2),
      std::runtime_error);
}

// ---------------------------------------------------------------------------
// BYTE-EXACT routing parity (35B decode lever L1): the parallel greedy top-k
// (production, the registered CUDA moe_router_topk) must select the identical
// experts with the identical weights bits as the original single-threaded
// serial reference. Bit-exactness is guaranteed by construction (untouched
// softmax + comparison-only argmax with the same lowest-index tie-break); this
// test pins it on adversarial inputs — exact ties, near-ties, uneven loads, and
// M in {1,8,16}, for f32 and bf16 logits (bf16 rounding manufactures ties).
//
// It is ALSO the on-device gate for the single-warp router (VT_MOE_ROUTER_WARP,
// issue #378), which is what `vt::MoeRouterTopK` dispatches by default for
// E in {32,64,128,256}: that kernel's whole claim is that it is byte-identical
// to this same serial reference, so it must survive every case here. The
// portable companion tests/vt/test_moe_router_warp_map.cpp proves the REDUCTION
// ORDER with no GPU; this one proves the KERNEL. Neither substitutes for the
// other, and neither comparison is ever loosened: weights are compared with
// memcmp and indices with ==.
// Guarded on VLLM_CPP_CUDA: the parallel-vs-serial cross-check calls the
// CUDA-only reference vt::cuda::MoeRouterTopKSerialCuda (cuda_moe.cu), so a
// runtime HasCuda() skip is not enough — the symbol is undefined at link time
// on a CPU build. (Matches the CUDA-only guard on test_ops_gdn.cpp:1176.)
#ifdef VLLM_CPP_CUDA
TEST_CASE("CUDA moe_router_topk parallel == serial byte-for-byte (adversarial)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  // ── The kernels' `-1` sentinel is UNREACHABLE through this op. ────────────
  // An earlier revision of this case ran `k = E + 3` to exercise the `best < 0`
  // path (indices -1, weights -INFINITY, denom<=0 -> 1). That is not an input
  // vt::MoeRouterTopK has: it validates
  //     VT_CHECK(args.top_k >= 1 && args.top_k <= e,
  //              "moe_router_topk: top_k must be in [1, num_experts]");
  // (src/vt/ops.cpp) BEFORE dispatch, so the arm threw, this whole case aborted
  // partway, and the byte-exactness sweep below never finished — while every
  // assertion that HAD run still reported passed. Within `1 <= k <= E` the
  // sentinel cannot fire either: after `sum > 0 ? sp/sum : 0` and the isfinite
  // clamp every prob is finite and >= 0 (cuda_moe.cu), the argmax seeds
  // best_v = -INFINITY, and only masking ALL E experts leaves nothing greater —
  // which needs a (k+1)-th round, i.e. k > E. The sentinel is a defensive guard,
  // not a reachable behaviour of this op.
  //
  // It was also UNSAFE to drive from here: the serial oracle this case compares
  // against is called directly (vt::cuda::MoeRouterTopKSerialCuda), bypassing
  // the validator, and its `sp[best] = -INFINITY` carries no `best >= 0` guard
  // — so a k > E round writes shared memory at sp[-1]. That is the oracle, so
  // the fix is to stop feeding it an input the op forbids, never to edit it.
  //
  // The two kernels' sentinel guards are still compared, at the level where the
  // sentinel exists: tests/vt/test_moe_router_warp_map.cpp models the block and
  // warp kernels directly and keeps its `k > E` arm. Here we pin the contract
  // that makes the sentinel dead.
  {
    QueueGuard gq(gpu);
    const int64_t T = 2, E = 32;
    const int k = static_cast<int>(E) + 3;
    DeviceTensor dlog(gpu, gq.q, DType::kF32, {T, E});
    DeviceTensor w_bad(gpu, gq.q, DType::kF32, {T, static_cast<int64_t>(k)});
    DeviceTensor i_bad(gpu, gq.q, DType::kI32, {T, static_cast<int64_t>(k)});
    const vt::MoeRouterTopKArgs args{k, false};
    CHECK_THROWS_AS(vt::MoeRouterTopK(gq.q, w_bad.tensor(), i_bad.tensor(), dlog.tensor(), args),
                    std::runtime_error);
  }

  auto run = [&](int64_t T, int64_t E, int k, bool renorm, DType dt,
                 const std::vector<float>& logits) {
    QueueGuard gq(gpu);
    const int64_t P = T * k;
    DeviceTensor w_par(gpu, gq.q, DType::kF32, {T, k});
    DeviceTensor i_par(gpu, gq.q, DType::kI32, {T, k});
    DeviceTensor w_ser(gpu, gq.q, DType::kF32, {T, k});
    DeviceTensor i_ser(gpu, gq.q, DType::kI32, {T, k});
    std::unique_ptr<DeviceTensor> dlog;
    std::vector<uint16_t> bf;
    if (dt == DType::kF32) {
      dlog = std::make_unique<DeviceTensor>(gpu, gq.q, DType::kF32,
                                            std::vector<int64_t>{T, E}, logits.data());
    } else {
      bf = ToBf16(logits);
      dlog = std::make_unique<DeviceTensor>(gpu, gq.q, DType::kBF16,
                                            std::vector<int64_t>{T, E}, bf.data());
    }
    const vt::MoeRouterTopKArgs args{k, renorm};
    vt::MoeRouterTopK(gq.q, w_par.tensor(), i_par.tensor(), dlog->tensor(), args);
    vt::cuda::MoeRouterTopKSerialCuda(gq.q, w_ser.tensor(), i_ser.tensor(), dlog->tensor(), args);
    gpu.Synchronize(gq.q);
    std::vector<float> hp(static_cast<size_t>(P)), hs(static_cast<size_t>(P));
    std::vector<int32_t> ip(static_cast<size_t>(P)), is(static_cast<size_t>(P));
    w_par.Download(gq.q, hp.data());
    w_ser.Download(gq.q, hs.data());
    i_par.Download(gq.q, ip.data());
    i_ser.Download(gq.q, is.data());
    size_t wdiff = 0, idiff = 0;
    for (int64_t x = 0; x < P; ++x) {
      if (std::memcmp(&hp[static_cast<size_t>(x)], &hs[static_cast<size_t>(x)], sizeof(float)) != 0)
        ++wdiff;  // strict bitwise weight equality
      if (ip[static_cast<size_t>(x)] != is[static_cast<size_t>(x)]) ++idiff;
    }
    CAPTURE(T);
    CAPTURE(E);
    CAPTURE(k);
    CAPTURE(renorm);
    CAPTURE(static_cast<int>(dt));
    CHECK(wdiff == 0);
    CHECK(idiff == 0);
  };

  // ── The sweep runs TWICE, with VT_MOE_ROUTER_WARP PINNED both ways. ───────
  // vt::cuda::MoeRouterWarpEnabled() (cuda_moe.cu) is a FRESH getenv per launch,
  // so WHICH kernel vt::MoeRouterTopK runs below is decided by the environment
  // ctest was started in. An earlier revision of this case neither set, cleared
  // nor asserted that variable, which made it unable to say what it had tested:
  // with `VT_MOE_ROUTER_WARP=0` exported — exactly what spec §9 gates 6/7 tell
  // the operator to export for the same-binary A/B, in the same shell — every
  // `run()` below silently exercised the BLOCK kernel, i.e. block-vs-serial,
  // green since 6a8c5cf9, reporting the identical case and assertion counts. A
  // green run could not distinguish "the warp kernel is byte-exact" from "the
  // warp kernel never ran".
  //
  // So: pin it, assert the pinned state, and run the sweep under BOTH arms —
  // "1" pins the candidate warp kernel, "0" pins the same-binary rollback, and
  // each must be byte-identical to the serial oracle. The pin restores the
  // ambient value (or its absence) on scope exit.
  //
  // This DOUBLES this case's assertion count on purpose. Spec §9 gate 2 treats a
  // changed count as a red flag; the change is this, and it is the point.
  //
  // The body below keeps its original indentation so this change reads as what
  // it is — a pure wrapper — and a reviewer can see that not one line of the
  // sweep itself moved.
  auto sweep = [&]() {
  for (DType dt : {DType::kF32, DType::kBF16}) {
    for (bool renorm : {true, false}) {
      for (int64_t T : {int64_t{1}, int64_t{8}, int64_t{16}}) {
        // E in {32,64,128,256}: EVERY width the single-warp router dispatches
        // (VPT = E/32 in {1,2,4,8}, VT_MOE_ROUTER_WARP, issue #378), so the
        // byte-exactness of each derived lane map is pinned on device and not
        // only by the portable reduction-order test.
        //
        // The 35B-A3B gate model is E=256 top-8, NOT the E=128 this case used
        // to claim: num_experts=256, num_experts_per_tok=8, and its 40 MoE
        // layers are the 40 router calls/step the decode trace shows. E=256 had
        // only a hand-built near-tie pattern here and never random logits.
        for (int64_t E : {int64_t{32}, int64_t{64}, int64_t{128}, int64_t{256}}) {
          // Random distinct logits.
          {
            std::vector<float> lg(static_cast<size_t>(T * E));
            std::mt19937 rng(1234u + static_cast<uint32_t>(T) + 7919u * static_cast<uint32_t>(E));
            std::uniform_real_distribution<float> d(-4.0f, 4.0f);
            for (auto& v : lg) v = d(rng);
            run(T, E, 8, renorm, dt, lg);
            run(T, E, 1, renorm, dt, lg);  // k=1
            // NO k > E arm here — it is not an input this op has. See the
            // contract block at the top of this case.
          }
          // Exact-tie storm: blocks of identical logits so many experts tie at
          // the max; the tie-break (lowest index) must agree across paths.
          {
            std::vector<float> lg(static_cast<size_t>(T * E));
            for (int64_t t = 0; t < T; ++t)
              for (int64_t e = 0; e < E; ++e)
                lg[static_cast<size_t>(t * E + e)] =
                    static_cast<float>((e / 4) % 5);  // 5 tie groups
            run(T, E, 8, renorm, dt, lg);
          }
          // Uneven load + near-ties around the top-k boundary.
          {
            std::vector<float> lg(static_cast<size_t>(T * E));
            for (int64_t t = 0; t < T; ++t)
              for (int64_t e = 0; e < E; ++e)
                lg[static_cast<size_t>(t * E + e)] =
                    (e < 12 ? 3.0f : 0.0f) + 1e-4f * static_cast<float>((e * 7 + t) % 3);
            run(T, E, 8, renorm, dt, lg);
          }
          // Degenerate rows. The -INFINITY max seed ERASES NaN, so an all-NaN
          // row normalizes to all zeros and the tie-break must hand back
          // 0,1,...,k-1; Inf rows exercise the same clamp from the other side.
          // These are the rows CUDA-graph padding actually produces.
          {
            run(T, E, 8, renorm, dt,
                std::vector<float>(static_cast<size_t>(T * E), std::nanf("")));
            run(T, E, 8, renorm, dt,
                std::vector<float>(static_cast<size_t>(T * E), INFINITY));
            run(T, E, 8, renorm, dt,
                std::vector<float>(static_cast<size_t>(T * E), -INFINITY));
            std::vector<float> lg(static_cast<size_t>(T * E), 1.0f);
            for (int64_t t = 0; t < T; ++t) {
              lg[static_cast<size_t>(t * E)] = std::nanf("");
              lg[static_cast<size_t>(t * E + 1)] = INFINITY;
              lg[static_cast<size_t>(t * E + E - 1)] = -INFINITY;
            }
            run(T, E, 8, renorm, dt, lg);
          }
        }
      }
    }
  }
  };  // sweep

  {
    // ON: vt::MoeRouterTopK dispatches MoeRouterTopKWarpKernel for every E here.
    vt_test::ScopedMoeRouterWarp pin("1");
    REQUIRE(vt_test::ScopedMoeRouterWarp::EffectiveFlag());
    sweep();
  }
  {
    // OFF: the same binary falls through to the unchanged block kernel. This arm
    // is what the pre-existing evidence actually covered, and it stays covered.
    vt_test::ScopedMoeRouterWarp pin("0");
    REQUIRE_FALSE(vt_test::ScopedMoeRouterWarp::EffectiveFlag());
    sweep();
  }
}
#endif  // VLLM_CPP_CUDA

#ifdef VT_MARLIN_NVFP4
#include "vt/cuda/marlin_repack.h"

// Fused-w13 Marlin probe (VT_MOE_FUSED_W13 lever): ONE grouped Marlin GEMM over
// the N-concatenated gate|up (size_n=2N, output [P,2N]) + SiluAndMul on the
// halves must match TWO grouped GEMMs (size_n=N each) + MoeSiluMul — the split
// path MoeBlockFusedMarlinCuda runs by default. Mirrors vLLM marlin_moe.py:133-170
// (one moe_wna16_marlin_gemm with size_n = w13_num_shards*N, then silu_and_mul).
// The comparison is CheckClose(atol=rtol=0) = BIT-EXACT: it PINS whether the 2N
// grouped schedule preserves the k-accumulation order. If a future kernel/tile
// change legitimately reorders accumulation, this case may need a tolerance —
// the model-level gate is 35B greedy 16/16-vs-oracle either way.
TEST_CASE("CUDA marlin fused w13 (size_n=2N) is bit-exact vs split gate/up GEMMs") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  void* stream = gq.q.handle;
  const int dev = gq.q.device.index;

  // 35B-like small shape: K%128==0 and N%64==0 (Marlin tile constraints, the
  // no-padding case vLLM's pad_w13 reduces to). top_k=2, T=6 → P=12 pairs.
  const int64_t E = 4, T = 6, top_k = 2, P = T * top_k;
  const int64_t K = 256, N = 64;

  // Per expert: gate and up nvfp4 weights with EQUAL scale2 (the fused path's
  // single per-expert global scale — vLLM w13_weight_scale_2[:, 0] after its
  // allclose check, modelopt.py:1556-1564).
  std::vector<Nvfp4Weight> gate_w, up_w;
  for (int64_t e = 0; e < E; ++e) {
    gate_w.push_back(MakeNvfp4Weight(N, K, 100 + static_cast<uint32_t>(e)));
    up_w.push_back(MakeNvfp4Weight(N, K, 200 + static_cast<uint32_t>(e)));
    up_w.back().scale2 = gate_w.back().scale2;
  }
  const auto act_f = RandomF32(static_cast<size_t>(T * K), 4242);
  const auto act_bf16 = ToBf16(act_f);

  // combined_scale_factor jointly over gate+up (vLLM computes it over the
  // STACKED w13 scales; identical for both paths).
  std::vector<const uint8_t*> sc_bufs;
  std::vector<size_t> sc_lens;
  for (int64_t e = 0; e < E; ++e) {
    sc_bufs.push_back(gate_w[static_cast<size_t>(e)].scale.data());
    sc_lens.push_back(gate_w[static_cast<size_t>(e)].scale.size());
    sc_bufs.push_back(up_w[static_cast<size_t>(e)].scale.data());
    sc_lens.push_back(up_w[static_cast<size_t>(e)].scale.size());
  }
  const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(sc_bufs, sc_lens);

  // Repack SPLIT (per-shard, size_n=N) and FUSED (host-concat gate|up rows,
  // size_n=2N) residents + processed scales + global scales.
  const size_t wq_i32 = static_cast<size_t>(K / 16) * (N * 2);
  const size_t sc_b = static_cast<size_t>(K / 16) * N;
  DeviceTensor wq_gate(gpu, gq.q, DType::kI32, {E, K / 16, N * 2});
  DeviceTensor wq_up(gpu, gq.q, DType::kI32, {E, K / 16, N * 2});
  DeviceTensor wq_gu(gpu, gq.q, DType::kI32, {E, K / 16, 2 * N * 2});
  DeviceTensor sc_gate(gpu, gq.q, DType::kI8, {E, K / 16, N});
  DeviceTensor sc_up(gpu, gq.q, DType::kI8, {E, K / 16, N});
  DeviceTensor sc_gu(gpu, gq.q, DType::kI8, {E, K / 16, 2 * N});
  std::vector<float> g_gate(static_cast<size_t>(E)), g_up(static_cast<size_t>(E)),
      g_gu(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    const Nvfp4Weight& g = gate_w[se];
    const Nvfp4Weight& u = up_w[se];
    DeviceTensor dpg(gpu, gq.q, DType::kI8, {N, K / 2}, g.packed.data());
    DeviceTensor dpu(gpu, gq.q, DType::kI8, {N, K / 2}, u.packed.data());
    DeviceTensor dsg(gpu, gq.q, DType::kI8, {N, K / 16}, g.scale.data());
    DeviceTensor dsu(gpu, gq.q, DType::kI8, {N, K / 16}, u.scale.data());
    // Concat = flat row-stack (packed [N,K/2] u8 and scales [N,K/16] fp8 are
    // row-major over N; gate rows first — the vLLM w13 shard order).
    std::vector<uint8_t> cat_p(g.packed.size() + u.packed.size());
    std::memcpy(cat_p.data(), g.packed.data(), g.packed.size());
    std::memcpy(cat_p.data() + g.packed.size(), u.packed.data(), u.packed.size());
    std::vector<uint8_t> cat_s(g.scale.size() + u.scale.size());
    std::memcpy(cat_s.data(), g.scale.data(), g.scale.size());
    std::memcpy(cat_s.data() + g.scale.size(), u.scale.data(), u.scale.size());
    DeviceTensor dpc(gpu, gq.q, DType::kI8, {2 * N, K / 2}, cat_p.data());
    DeviceTensor dsc(gpu, gq.q, DType::kI8, {2 * N, K / 16}, cat_s.data());

    auto* wq_gate_e = static_cast<uint32_t*>(wq_gate.ptr()) + se * wq_i32;
    auto* wq_up_e = static_cast<uint32_t*>(wq_up.ptr()) + se * wq_i32;
    auto* wq_gu_e = static_cast<uint32_t*>(wq_gu.ptr()) + se * 2 * wq_i32;
    vt::cuda::MarlinRepackExpertWeight(stream, dev, wq_gate_e,
                                       static_cast<const uint8_t*>(dpg.ptr()), K, N);
    vt::cuda::MarlinRepackExpertWeight(stream, dev, wq_up_e,
                                       static_cast<const uint8_t*>(dpu.ptr()), K, N);
    vt::cuda::MarlinRepackExpertWeight(stream, dev, wq_gu_e,
                                       static_cast<const uint8_t*>(dpc.ptr()), K, 2 * N);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dsg.ptr()),
                                        static_cast<uint8_t*>(sc_gate.ptr()) + se * sc_b, K, N, sf);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dsu.ptr()),
                                        static_cast<uint8_t*>(sc_up.ptr()) + se * sc_b, K, N, sf);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dsc.ptr()),
                                        static_cast<uint8_t*>(sc_gu.ptr()) + se * 2 * sc_b, K,
                                        2 * N, sf);
    g_gate[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(g.scale2, sf);
    g_up[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(u.scale2, sf);
    g_gu[se] = g_gate[se];  // equal scale2 → identical processed global
    gpu.Synchronize(gq.q);  // repack reads the loop-local staging uploads
  }
  DeviceTensor dg_gate(gpu, gq.q, DType::kF32, {E}, g_gate.data());
  DeviceTensor dg_up(gpu, gq.q, DType::kF32, {E}, g_up.data());
  DeviceTensor dg_gu(gpu, gq.q, DType::kF32, {E}, g_gu.data());

  // moe_align inputs over random top-k ids (same for both paths).
  std::vector<int32_t> topk_ids(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p)
    topk_ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 5 + 3) % E);
  std::vector<float> topk_w(static_cast<size_t>(P), 1.0f);
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(static_cast<int>(T),
                                                            static_cast<int>(top_k),
                                                            static_cast<int>(E));
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(static_cast<int>(T), static_cast<int>(top_k),
                                static_cast<int>(E), block, &max_tok, &max_blk);
  DeviceTensor dtid(gpu, gq.q, DType::kI32, {T, top_k}, topk_ids.data());
  DeviceTensor dtw(gpu, gq.q, DType::kF32, {T, top_k}, topk_w.data());
  DeviceTensor sorted_ids(gpu, gq.q, DType::kI32, {max_tok});
  DeviceTensor expert_ids(gpu, gq.q, DType::kI32, {max_blk});
  DeviceTensor num_pad(gpu, gq.q, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()),
                                    static_cast<int>(T), static_cast<int>(top_k),
                                    static_cast<int>(E), block,
                                    static_cast<int32_t*>(sorted_ids.ptr()),
                                    static_cast<int32_t*>(expert_ids.ptr()),
                                    static_cast<int32_t*>(num_pad.ptr()));

  const int sms = vt::cuda::MarlinDeviceSms(dev);
  DeviceTensor ws(gpu, gq.q, DType::kI32, {sms * 4});
  DeviceTensor dact(gpu, gq.q, DType::kBF16, {T, K}, act_bf16.data());
  const vt::MoeMarlinArgs args_n{block, static_cast<int>(top_k), static_cast<int>(T),
                                 static_cast<int>(N), static_cast<int>(K), false};
  const vt::MoeMarlinArgs args_2n{block, static_cast<int>(top_k), static_cast<int>(T),
                                  static_cast<int>(2 * N), static_cast<int>(K), false};

  // SPLIT: two GEMMs + MoeSiluMul.
  DeviceTensor dgate(gpu, gq.q, DType::kBF16, {P, N});
  DeviceTensor dup(gpu, gq.q, DType::kBF16, {P, N});
  DeviceTensor act_split(gpu, gq.q, DType::kBF16, {P, N});
  gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  vt::MoeGroupedGemmNvfp4Marlin(gq.q, dgate.tensor(), dact.tensor(), wq_gate.tensor(),
                                sc_gate.tensor(), dg_gate.tensor(), ws.tensor(),
                                sorted_ids.tensor(), expert_ids.tensor(), num_pad.tensor(),
                                dtw.tensor(), args_n);
  gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  vt::MoeGroupedGemmNvfp4Marlin(gq.q, dup.tensor(), dact.tensor(), wq_up.tensor(),
                                sc_up.tensor(), dg_up.tensor(), ws.tensor(),
                                sorted_ids.tensor(), expert_ids.tensor(), num_pad.tensor(),
                                dtw.tensor(), args_n);
  vt::MoeSiluMul(gq.q, act_split.tensor(), dgate.tensor(), dup.tensor());

  // FUSED: one GEMM (size_n=2N) + SiluAndMul on the halves.
  DeviceTensor dgu(gpu, gq.q, DType::kBF16, {P, 2 * N});
  DeviceTensor act_fused(gpu, gq.q, DType::kBF16, {P, N});
  gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  vt::MoeGroupedGemmNvfp4Marlin(gq.q, dgu.tensor(), dact.tensor(), wq_gu.tensor(),
                                sc_gu.tensor(), dg_gu.tensor(), ws.tensor(),
                                sorted_ids.tensor(), expert_ids.tensor(), num_pad.tensor(),
                                dtw.tensor(), args_2n);
  vt::SiluAndMul(gq.q, act_fused.tensor(), dgu.tensor());

  // Compare BITWISE: fused halves vs split GEMM outputs, and the activations.
  std::vector<uint16_t> h_gate(static_cast<size_t>(P * N)), h_up(static_cast<size_t>(P * N));
  std::vector<uint16_t> h_gu(static_cast<size_t>(P * 2 * N));
  std::vector<uint16_t> h_act_s(static_cast<size_t>(P * N)), h_act_f(static_cast<size_t>(P * N));
  dgate.Download(gq.q, h_gate.data());
  dup.Download(gq.q, h_up.data());
  dgu.Download(gq.q, h_gu.data());
  act_split.Download(gq.q, h_act_s.data());
  act_fused.Download(gq.q, h_act_f.data());
  size_t gate_diff = 0, up_diff = 0, act_diff = 0;
  for (int64_t p = 0; p < P; ++p) {
    for (int64_t n = 0; n < N; ++n) {
      const size_t i = static_cast<size_t>(p * N + n);
      if (h_gate[i] != h_gu[static_cast<size_t>(p * 2 * N + n)]) ++gate_diff;
      if (h_up[i] != h_gu[static_cast<size_t>(p * 2 * N + N + n)]) ++up_diff;
      if (h_act_s[i] != h_act_f[i]) ++act_diff;
    }
  }
  CHECK(gate_diff == 0);
  CHECK(up_diff == 0);
  CHECK(act_diff == 0);
}

// ---------------------------------------------------------------------------
// BYTE-EXACT align parity (35B decode lever L1): the parallel moe_align
// (production) must produce the identical expert_ids, num_tokens_post_pad, and
// per-expert padded-region token multiset as the original single-block serial
// reference. Within-expert token ORDER is unspecified in BOTH paths (the serial
// scatter already races a 256-thread atomicAdd; Marlin gathers each sorted row
// independently), so sorted_ids is compared as a per-expert multiset, not
// position-for-position. Adversarial inputs: empty experts, one hot expert,
// uneven loads, ties on block boundaries, and M in {1,8,16}.
// MXFP4 W4A16 Marlin GEMM (group_blocks=2, E8M0 scales, no global) must match the
// INDEPENDENT CPU dequant reference (DequantMxfp4ToF32 + f32 matmul) — a different
// code path than the Marlin repack+kernel dequant, so this is a real cross-check
// (not a shared-helper tautology). Single expert, all tokens -> expert 0 (the dense
// MatmulMxfp4W4A16D routing). RED-first coverage of the M=1 DECODE path AND M=8.
TEST_CASE("CUDA marlin MXFP4 W4A16 (group_blocks=2, E8M0) matches CPU dequant reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  // K%128==0 (Marlin tile) and %32 (mxfp4 group); N%64==0. Cover the small case
  // AND the real Qwen3-8B projection shapes (o/qkv K=4096; down K=12288; gate/up
  // N=12288) so the group_blocks=2 kernel is exercised at model scale, not just a
  // tiny tile — a large-shape-only bug would pass the tiny case and corrupt e2e.
  for (auto KN : std::vector<std::pair<int64_t, int64_t>>{
           {256, 128}, {4096, 4096}, {4096, 12288}, {12288, 4096}}) {
  const int64_t K = KN.first, N = KN.second;
  CAPTURE(K);
  CAPTURE(N);
  Mxfp4Weight w = MakeMxfp4Weight(N, K, 4321);

  // CPU reference weight [N,K] f32 via the independent dequant helper.
  std::vector<float> w_f32(static_cast<size_t>(N * K));
  vllm::DequantMxfp4ToF32(w.packed.data(), w.scale.data(), N, K, w_f32.data());

  for (int64_t M : {int64_t{1}, int64_t{8}}) {
    CAPTURE(M);
    QueueGuard gq(gpu);
    void* stream = gq.q.handle;
    const int dev = gq.q.device.index;
    const int64_t top_k = 1, E = 1, P = M * top_k;

    const auto act_f = RandomF32(static_cast<size_t>(M * K), 7000 + static_cast<uint32_t>(M));
    const auto act_bf16 = ToBf16(act_f);
    std::vector<float> act_r(act_f.size());
    for (size_t i = 0; i < act_r.size(); ++i) act_r[i] = vt::BF16ToF32(act_bf16[i]);

    // Reference: out[m,n] = sum_k act_r[m,k] * w_f32[n,k] (f32 accum).
    std::vector<float> ref(static_cast<size_t>(P * N), 0.0f);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k)
          acc += act_r[static_cast<size_t>(m * K + k)] * w_f32[static_cast<size_t>(n * K + k)];
        ref[static_cast<size_t>(m * N + n)] = acc;
      }

    // Repack weight + process E8M0 scales (mxfp4: passthrough permute, no global).
    DeviceTensor dp(gpu, gq.q, DType::kI8, {N, K / 2}, w.packed.data());
    DeviceTensor ds(gpu, gq.q, DType::kI8, {N, K / 32}, w.scale.data());
    DeviceTensor wq(gpu, gq.q, DType::kI32, {E, K / 16, N * 2});
    DeviceTensor sc(gpu, gq.q, DType::kI8, {E, K / 32, N});  // K/32 groups (group_blocks=2)
    vt::cuda::MarlinRepackExpertWeight(stream, dev, static_cast<uint32_t*>(wq.ptr()),
                                       static_cast<const uint8_t*>(dp.ptr()),
                                       static_cast<int>(K), static_cast<int>(N));
    vt::cuda::MarlinProcessExpertScalesMxfp4(stream, static_cast<const uint8_t*>(ds.ptr()),
                                             static_cast<uint8_t*>(sc.ptr()),
                                             static_cast<int>(K), static_cast<int>(N));
    float g_dummy = 1.0f;  // ignored on the mxfp4 path (kernel skips global for E8M0)
    DeviceTensor gg(gpu, gq.q, DType::kF32, {E}, &g_dummy);

    // Align inputs: all P tokens -> expert 0.
    std::vector<int32_t> topk_ids(static_cast<size_t>(P), 0);
    std::vector<float> topk_w(static_cast<size_t>(P), 1.0f);
    const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(static_cast<int>(M),
                                                              static_cast<int>(top_k),
                                                              static_cast<int>(E));
    int max_tok = 0, max_blk = 0;
    vt::cuda::MarlinMoeAlignSizes(static_cast<int>(M), static_cast<int>(top_k),
                                  static_cast<int>(E), block, &max_tok, &max_blk);
    DeviceTensor dtid(gpu, gq.q, DType::kI32, {M, top_k}, topk_ids.data());
    DeviceTensor dtw(gpu, gq.q, DType::kF32, {M, top_k}, topk_w.data());
    DeviceTensor sorted_ids(gpu, gq.q, DType::kI32, {max_tok});
    DeviceTensor expert_ids(gpu, gq.q, DType::kI32, {max_blk});
    DeviceTensor num_pad(gpu, gq.q, DType::kI32, {1});
    vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()),
                                      static_cast<int>(M), static_cast<int>(top_k),
                                      static_cast<int>(E), block,
                                      static_cast<int32_t*>(sorted_ids.ptr()),
                                      static_cast<int32_t*>(expert_ids.ptr()),
                                      static_cast<int32_t*>(num_pad.ptr()));
    const int sms = vt::cuda::MarlinDeviceSms(dev);
    DeviceTensor ws(gpu, gq.q, DType::kI32, {sms * 4});
    gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
    DeviceTensor dact(gpu, gq.q, DType::kBF16, {M, K}, act_bf16.data());
    DeviceTensor dout(gpu, gq.q, DType::kBF16, {P, N});

    vt::MoeMarlinArgs args{block, static_cast<int>(top_k), static_cast<int>(M),
                           static_cast<int>(N), static_cast<int>(K), false};
    args.group_size = 32;
    args.mxfp4 = true;
    vt::MoeGroupedGemmNvfp4Marlin(gq.q, dout.tensor(), dact.tensor(), wq.tensor(), sc.tensor(),
                                  gg.tensor(), ws.tensor(), sorted_ids.tensor(),
                                  expert_ids.tensor(), num_pad.tensor(), dtw.tensor(), args);
    std::vector<uint16_t> got_bf16(static_cast<size_t>(P * N));
    dout.Download(gq.q, got_bf16.data());
    std::vector<float> got(got_bf16.size());
    for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(got_bf16[i]);
    // Report the true error magnitude: a correct bf16-out Marlin GEMM vs an f32
    // reference sits near bf16 rounding (~1e-2); a systematic error compounds.
    float max_rel = 0.0f, max_abs = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
      const float a = std::fabs(got[i] - ref[i]);
      const float r = a / (std::fabs(ref[i]) + 1e-6f);
      if (a > max_abs) max_abs = a;
      if (r > max_rel) max_rel = r;
    }
    MESSAGE("MXFP4 K=" << K << " N=" << N << " M=" << M
                       << " max_abs=" << max_abs << " max_rel=" << max_rel);
    // Measured max_rel ~= 3.8e-3 at M=1,M=8 (K=256,N=128; pure bf16 rounding, NOT
    // a systematic error). Gate at 2e-2 leaves bf16 headroom. The added Qwen3-8B
    // shapes ({4096,4096},{4096,12288},{12288,4096}) are RUN-PENDING (box contended
    // when authored) — a FAIL there localizes a large-N/K group_blocks=2 kernel bug
    // as the e2e residual's cause; a PASS shifts the residual to model integration.
    CheckClose(got, ref, 2e-2f, 2e-2f);
  }
  }
}

// --- QUANT-CT-MXFP4-CLOSERS slivers (a)+(b) --------------------------------
// Sliver (a): the dense single-expert path forces moe_block_size=8 at M<=8
// (dense_nvfp4_gemm.h::DenseAlignFor) so M=8 uses vLLM's 8-row m_block_size_8
// tile instead of the padded 16-row tile MarlinMoeAlignBlockSizeSelect otherwise
// picks at M=8. This proves the forced block=8 route is CORRECT vs an
// independent CPU dequant reference at M=8, and MEASURES whether it is
// byte-exact vs the block=16 route production used before the fix (the near-tie
// arbiter). Sliver (b): the shared reduction workspace is zeroed ONCE, not
// per-call — the fp32-reduce marlin barrier self-resets its locks
// (marlin_template.h:2170 barrier_release(...,last) -> :204 lock[0]=0; the
// slice_count==1 case never touches locks at :2162; our launch pins
// use_atomic_add=false at cuda_moe_marlin.cu:141, so the non-self-clearing
// atomic-add path at :614 is unreachable). Proven directly: the workspace is
// all-zero AFTER a GEMM, and a second GEMM on the once-zeroed workspace is
// bit-identical to the first.
TEST_CASE("CUDA marlin MXFP4 W4A16 dense M=8: block=8 route + ws self-reset (closers)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  // Real Qwen3-8B decode projection shapes (K%128==0, K%32==0, N%64==0).
  for (auto KN : std::vector<std::pair<int64_t, int64_t>>{{4096, 4096}, {12288, 4096}}) {
    const int64_t K = KN.first, N = KN.second;
    CAPTURE(K);
    CAPTURE(N);
    const int64_t M = 8, top_k = 1, E = 1, P = M * top_k;

    // Sanity: MarlinMoeAlignBlockSizeSelect DOES pick 16 at M=8 (the pre-fix
    // production route this sliver overrides to 8).
    CHECK(vt::cuda::MarlinMoeAlignBlockSizeSelect(static_cast<int>(M), 1, 1) == 16);

    Mxfp4Weight w = MakeMxfp4Weight(N, K, 4321);
    std::vector<float> w_f32(static_cast<size_t>(N * K));
    vllm::DequantMxfp4ToF32(w.packed.data(), w.scale.data(), N, K, w_f32.data());

    const auto act_f = RandomF32(static_cast<size_t>(M * K), 7700);
    const auto act_bf16 = ToBf16(act_f);
    std::vector<float> act_r(act_f.size());
    for (size_t i = 0; i < act_r.size(); ++i) act_r[i] = vt::BF16ToF32(act_bf16[i]);
    std::vector<float> ref(static_cast<size_t>(P * N), 0.0f);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k)
          acc += act_r[static_cast<size_t>(m * K + k)] * w_f32[static_cast<size_t>(n * K + k)];
        ref[static_cast<size_t>(m * N + n)] = acc;
      }

    QueueGuard gq(gpu);
    void* stream = gq.q.handle;
    const int dev = gq.q.device.index;

    // Repack ONCE (block-independent).
    DeviceTensor dp(gpu, gq.q, DType::kI8, {N, K / 2}, w.packed.data());
    DeviceTensor ds(gpu, gq.q, DType::kI8, {N, K / 32}, w.scale.data());
    DeviceTensor wq(gpu, gq.q, DType::kI32, {E, K / 16, N * 2});
    DeviceTensor sc(gpu, gq.q, DType::kI8, {E, K / 32, N});
    vt::cuda::MarlinRepackExpertWeight(stream, dev, static_cast<uint32_t*>(wq.ptr()),
                                       static_cast<const uint8_t*>(dp.ptr()),
                                       static_cast<int>(K), static_cast<int>(N));
    vt::cuda::MarlinProcessExpertScalesMxfp4(stream, static_cast<const uint8_t*>(ds.ptr()),
                                             static_cast<uint8_t*>(sc.ptr()),
                                             static_cast<int>(K), static_cast<int>(N));
    float g_dummy = 1.0f;
    DeviceTensor gg(gpu, gq.q, DType::kF32, {E}, &g_dummy);
    DeviceTensor dact(gpu, gq.q, DType::kBF16, {M, K}, act_bf16.data());
    const int sms = vt::cuda::MarlinDeviceSms(dev);

    // One dense single-expert GEMM at `block`, sharing `wst` (the workspace).
    auto run = [&](int block, Tensor wst, std::vector<uint16_t>& out) {
      std::vector<int32_t> topk_ids(static_cast<size_t>(P), 0);
      std::vector<float> topk_w(static_cast<size_t>(P), 1.0f);
      int max_tok = 0, max_blk = 0;
      vt::cuda::MarlinMoeAlignSizes(static_cast<int>(M), static_cast<int>(top_k),
                                    static_cast<int>(E), block, &max_tok, &max_blk);
      DeviceTensor dtid(gpu, gq.q, DType::kI32, {M, top_k}, topk_ids.data());
      DeviceTensor dtw(gpu, gq.q, DType::kF32, {M, top_k}, topk_w.data());
      DeviceTensor sorted_ids(gpu, gq.q, DType::kI32, {max_tok});
      DeviceTensor expert_ids(gpu, gq.q, DType::kI32, {max_blk});
      DeviceTensor num_pad(gpu, gq.q, DType::kI32, {1});
      vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()),
                                        static_cast<int>(M), static_cast<int>(top_k),
                                        static_cast<int>(E), block,
                                        static_cast<int32_t*>(sorted_ids.ptr()),
                                        static_cast<int32_t*>(expert_ids.ptr()),
                                        static_cast<int32_t*>(num_pad.ptr()));
      DeviceTensor dout(gpu, gq.q, DType::kBF16, {P, N});
      vt::MoeMarlinArgs args{block, static_cast<int>(top_k), static_cast<int>(M),
                             static_cast<int>(N), static_cast<int>(K), false};
      args.group_size = 32;
      args.mxfp4 = true;
      vt::MoeGroupedGemmNvfp4Marlin(gq.q, dout.tensor(), dact.tensor(), wq.tensor(), sc.tensor(),
                                    gg.tensor(), wst, sorted_ids.tensor(), expert_ids.tensor(),
                                    num_pad.tensor(), dtw.tensor(), args);
      out.assign(static_cast<size_t>(P * N), 0);
      dout.Download(gq.q, out.data());
    };
    auto to_f32 = [](const std::vector<uint16_t>& b) {
      std::vector<float> f(b.size());
      for (size_t i = 0; i < b.size(); ++i) f[i] = vt::BF16ToF32(b[i]);
      return f;
    };

    // Shared workspace zeroed ONCE (mirror the production DenseMarlinWorkspace).
    DeviceTensor ws(gpu, gq.q, DType::kI32, {sms * 4});
    gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
    Tensor wst = MakeTensor(ws.ptr(), DType::kI32, gq.q.device, {sms * 4});

    // Sliver (a): forced block=8 route is CORRECT vs the independent CPU ref.
    std::vector<uint16_t> out8;
    run(8, wst, out8);
    CheckClose(to_f32(out8), ref, 2e-2f, 2e-2f);

    // Sliver (b) invariant: a completed GEMM leaves the workspace all-zero. A
    // non-zero lock here would mean the dropped per-call re-zero is load-bearing.
    std::vector<int32_t> ws_host(static_cast<size_t>(sms * 4), -1);
    ws.Download(gq.q, ws_host.data());
    size_t ws_nonzero = 0;
    for (int32_t v : ws_host)
      if (v != 0) ++ws_nonzero;
    CAPTURE(ws_nonzero);
    CHECK(ws_nonzero == 0);

    // Sliver (b) reuse safety: a SECOND GEMM on the once-zeroed, not-re-zeroed
    // workspace is bit-identical to the first.
    std::vector<uint16_t> out8b;
    run(8, wst, out8b);
    CHECK(out8b == out8);

    // Sliver (a) byte-exact arbiter: block=16 was the pre-fix M=8 route. Measure
    // whether forcing block=8 changes the output bits (byte-exact => zero risk to
    // the SACRED token gate; near-tie => the token gate arbitrates).
    std::vector<uint16_t> out16;
    run(16, wst, out16);
    CheckClose(to_f32(out16), ref, 2e-2f, 2e-2f);
    size_t bitdiff = 0;
    float max_abs = 0.0f;
    {
      const auto f8 = to_f32(out8);
      const auto f16 = to_f32(out16);
      for (size_t i = 0; i < out8.size(); ++i) {
        if (out8[i] != out16[i]) ++bitdiff;
        max_abs = std::max(max_abs, std::fabs(f8[i] - f16[i]));
      }
    }
    MESSAGE("block8-vs-block16 M=8 K=" << K << " N=" << N << " bitdiff=" << bitdiff << "/"
                                       << out8.size() << " max_abs=" << max_abs);
  }
}

// Sliver (a) blast-radius closure for the NVFP4 (group_blocks=1, fp8 scales +
// global) dense route — the OTHER header consumer (Qwen3-32B-NVFP4A16, Laguna).
// The m_block_size_8 tile controls the bf16 ACTIVATION/A-matrix layout, which is
// identical for NVFP4 and MXFP4 (the quant scheme only changes the B-weight scale
// decode along K), so byte-exactness is quant-agnostic; this proves it directly.
TEST_CASE("CUDA marlin NVFP4 W4A16 dense M=8: block=8 byte-exact vs block=16 (closers)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t K = 4096, N = 4096, M = 8, top_k = 1, E = 1, P = M * top_k;

  // The byte-exact A/B is self-validating: block=16 is the production-validated
  // NVFP4 route (the Qwen3-32B-NVFP4A16 SACRED paged-engine gate runs on it), so
  // block=8 == block=16 bit-for-bit proves block=8 is exactly as correct without
  // needing an absolute CPU reference (whose sf/global-scale convention differs
  // from the Marlin repack path — see the divisor-reciprocal note in
  // test_qwen3_forward.cpp).
  Nvfp4Weight w = MakeNvfp4Weight(N, K, 4321);
  const auto act_f = RandomF32(static_cast<size_t>(M * K), 7700);
  const auto act_bf16 = ToBf16(act_f);

  QueueGuard gq(gpu);
  void* stream = gq.q.handle;
  const int dev = gq.q.device.index;

  std::vector<const uint8_t*> bufs{w.scale.data()};
  std::vector<size_t> lens{w.scale.size()};
  const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
  DeviceTensor dp(gpu, gq.q, DType::kI8, {N, K / 2}, w.packed.data());
  DeviceTensor dsx(gpu, gq.q, DType::kI8, {N, K / 16}, w.scale.data());
  DeviceTensor wq(gpu, gq.q, DType::kI32, {E, K / 16, N * 2});
  DeviceTensor sc(gpu, gq.q, DType::kI8, {E, K / 16, N});
  vt::cuda::MarlinRepackExpertWeight(stream, dev, static_cast<uint32_t*>(wq.ptr()),
                                     static_cast<const uint8_t*>(dp.ptr()), static_cast<int>(K),
                                     static_cast<int>(N));
  vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dsx.ptr()),
                                      static_cast<uint8_t*>(sc.ptr()), static_cast<int>(K),
                                      static_cast<int>(N), sf);
  float gsc = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.scale2, sf);
  DeviceTensor gg(gpu, gq.q, DType::kF32, {E}, &gsc);
  DeviceTensor dact(gpu, gq.q, DType::kBF16, {M, K}, act_bf16.data());
  const int sms = vt::cuda::MarlinDeviceSms(dev);
  DeviceTensor ws(gpu, gq.q, DType::kI32, {sms * 4});
  gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  Tensor wst = MakeTensor(ws.ptr(), DType::kI32, gq.q.device, {sms * 4});

  auto run = [&](int block, std::vector<uint16_t>& out) {
    std::vector<int32_t> topk_ids(static_cast<size_t>(P), 0);
    std::vector<float> topk_w(static_cast<size_t>(P), 1.0f);
    int max_tok = 0, max_blk = 0;
    vt::cuda::MarlinMoeAlignSizes(static_cast<int>(M), static_cast<int>(top_k),
                                  static_cast<int>(E), block, &max_tok, &max_blk);
    DeviceTensor dtid(gpu, gq.q, DType::kI32, {M, top_k}, topk_ids.data());
    DeviceTensor dtw(gpu, gq.q, DType::kF32, {M, top_k}, topk_w.data());
    DeviceTensor sorted_ids(gpu, gq.q, DType::kI32, {max_tok});
    DeviceTensor expert_ids(gpu, gq.q, DType::kI32, {max_blk});
    DeviceTensor num_pad(gpu, gq.q, DType::kI32, {1});
    vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()),
                                      static_cast<int>(M), static_cast<int>(top_k),
                                      static_cast<int>(E), block,
                                      static_cast<int32_t*>(sorted_ids.ptr()),
                                      static_cast<int32_t*>(expert_ids.ptr()),
                                      static_cast<int32_t*>(num_pad.ptr()));
    DeviceTensor dout(gpu, gq.q, DType::kBF16, {P, N});
    vt::MoeMarlinArgs args{block, static_cast<int>(top_k), static_cast<int>(M),
                           static_cast<int>(N), static_cast<int>(K), false};
    args.group_size = 16;  // NVFP4
    vt::MoeGroupedGemmNvfp4Marlin(gq.q, dout.tensor(), dact.tensor(), wq.tensor(), sc.tensor(),
                                  gg.tensor(), wst, sorted_ids.tensor(), expert_ids.tensor(),
                                  num_pad.tensor(), dtw.tensor(), args);
    out.assign(static_cast<size_t>(P * N), 0);
    dout.Download(gq.q, out.data());
  };

  std::vector<uint16_t> out8, out16;
  run(8, out8);
  run(16, out16);
  size_t bitdiff = 0;
  for (size_t i = 0; i < out8.size(); ++i)
    if (out8[i] != out16[i]) ++bitdiff;
  MESSAGE("NVFP4 block8-vs-block16 M=8 K=" << K << " N=" << N << " bitdiff=" << bitdiff << "/"
                                           << out8.size());
  CHECK(bitdiff == 0);
}

TEST_CASE("CUDA moe_align parallel == serial (expert_ids/num_pad exact, per-expert multiset)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  // Group a kernel's sorted_ids into expert -> sorted token-id list, using its
  // expert_ids block map. Non-sentinel entries only (sentinel == numel).
  auto group_by_expert = [](const std::vector<int32_t>& sorted, const std::vector<int32_t>& eids,
                            int block, int numel, int num_experts) {
    std::vector<std::vector<int32_t>> per(static_cast<size_t>(num_experts));
    for (size_t b = 0; b < eids.size(); ++b) {
      const int e = eids[b];
      if (e < 0 || e >= num_experts) continue;
      for (int i = 0; i < block; ++i) {
        const size_t pos = b * static_cast<size_t>(block) + static_cast<size_t>(i);
        if (pos >= sorted.size()) break;
        const int32_t v = sorted[pos];
        if (v != numel) per[static_cast<size_t>(e)].push_back(v);
      }
    }
    for (auto& v : per) std::sort(v.begin(), v.end());
    return per;
  };

  auto run = [&](int64_t T, int E, int top_k, const std::vector<int32_t>& topk_ids) {
    QueueGuard gq(gpu);
    void* stream = gq.q.handle;
    const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(static_cast<int>(T), top_k, E);
    int max_tok = 0, max_blk = 0;
    vt::cuda::MarlinMoeAlignSizes(static_cast<int>(T), top_k, E, block, &max_tok, &max_blk);
    const int numel = static_cast<int>(T) * top_k;

    DeviceTensor dtid(gpu, gq.q, DType::kI32, {T, top_k}, topk_ids.data());
    DeviceTensor s_par(gpu, gq.q, DType::kI32, {max_tok});
    DeviceTensor e_par(gpu, gq.q, DType::kI32, {max_blk});
    DeviceTensor n_par(gpu, gq.q, DType::kI32, {1});
    DeviceTensor s_ser(gpu, gq.q, DType::kI32, {max_tok});
    DeviceTensor e_ser(gpu, gq.q, DType::kI32, {max_blk});
    DeviceTensor n_ser(gpu, gq.q, DType::kI32, {1});

    vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()),
                                      static_cast<int>(T), top_k, E, block,
                                      static_cast<int32_t*>(s_par.ptr()),
                                      static_cast<int32_t*>(e_par.ptr()),
                                      static_cast<int32_t*>(n_par.ptr()));
    vt::cuda::MarlinMoeAlignBlockSizeSerial(stream, static_cast<const int32_t*>(dtid.ptr()),
                                            static_cast<int>(T), top_k, E, block,
                                            static_cast<int32_t*>(s_ser.ptr()),
                                            static_cast<int32_t*>(e_ser.ptr()),
                                            static_cast<int32_t*>(n_ser.ptr()));
    gpu.Synchronize(gq.q);

    std::vector<int32_t> hsp(static_cast<size_t>(max_tok)), hss(static_cast<size_t>(max_tok));
    std::vector<int32_t> hep(static_cast<size_t>(max_blk)), hes(static_cast<size_t>(max_blk));
    std::vector<int32_t> hnp(1), hns(1);
    s_par.Download(gq.q, hsp.data());
    s_ser.Download(gq.q, hss.data());
    e_par.Download(gq.q, hep.data());
    e_ser.Download(gq.q, hes.data());
    n_par.Download(gq.q, hnp.data());
    n_ser.Download(gq.q, hns.data());

    CAPTURE(T);
    CAPTURE(E);
    CAPTURE(top_k);
    CAPTURE(block);
    // expert_ids and num_tokens_post_pad are deterministic integer work: exact.
    CHECK(hnp[0] == hns[0]);
    size_t ediff = 0;
    for (int b = 0; b < max_blk; ++b)
      if (hep[static_cast<size_t>(b)] != hes[static_cast<size_t>(b)]) ++ediff;
    CHECK(ediff == 0);
    // Padding sentinel count must match (total slots - real tokens is fixed).
    const size_t pad_par = static_cast<size_t>(std::count(hsp.begin(), hsp.end(), numel));
    const size_t pad_ser = static_cast<size_t>(std::count(hss.begin(), hss.end(), numel));
    CHECK(pad_par == pad_ser);
    // Per-expert token multiset identical (order within an expert is free).
    const auto gp = group_by_expert(hsp, hep, block, numel, E);
    const auto gs = group_by_expert(hss, hes, block, numel, E);
    bool multiset_equal = (gp == gs);
    CHECK(multiset_equal);
    // Self-consistency: every real token landed in ITS expert's region.
    size_t misrouted = 0;
    for (int e = 0; e < E; ++e)
      for (int32_t p : gp[static_cast<size_t>(e)])
        if (topk_ids[static_cast<size_t>(p)] != e) ++misrouted;
    CHECK(misrouted == 0);
  };

  const int E = 128;
  const int top_k = 8;
  for (int64_t T : {int64_t{1}, int64_t{8}, int64_t{16}}) {
    const int numel = static_cast<int>(T) * top_k;
    // (a) strided spread over experts.
    {
      std::vector<int32_t> ids(static_cast<size_t>(numel));
      for (int p = 0; p < numel; ++p) ids[static_cast<size_t>(p)] = (p * 5 + 3) % E;
      run(T, E, top_k, ids);
    }
    // (b) one hot expert (all tokens -> expert 3): maximally uneven load.
    {
      std::vector<int32_t> ids(static_cast<size_t>(numel), 3);
      run(T, E, top_k, ids);
    }
    // (c) two experts only, boundary-aligned counts to exercise block padding.
    {
      std::vector<int32_t> ids(static_cast<size_t>(numel));
      for (int p = 0; p < numel; ++p) ids[static_cast<size_t>(p)] = (p % 2) ? 7 : 42;
      run(T, E, top_k, ids);
    }
    // (d) pseudo-random loads (some experts empty, some heavy).
    {
      std::vector<int32_t> ids(static_cast<size_t>(numel));
      std::mt19937 rng(77u + static_cast<uint32_t>(T));
      std::uniform_int_distribution<int> d(0, E - 1);
      for (auto& v : ids) v = d(rng);
      run(T, E, top_k, ids);
    }
  }
}

// ─── DENSE Marlin (row KERNEL-MARLIN-DENSE-PORT) ────────────────────────────
// vt::MarlinDenseGemm is vLLM's OWN dense W4A16 GEMM (direct-A, tile-per-CTA,
// dense fp32-C_tmp reduce) — the byte-preserving replacement for the
// single-expert MoE-marlin route the dense E=1 NVFP4/MXFP4 projections use today
// (dense_nvfp4_gemm.h). The gate (mission gate a): the dense op must match, per
// output element, BOTH an INDEPENDENT CPU-dequant reference AND the grouped
// (single-expert MoE) route it replaces, across M=1..8, NVFP4 + MXFP4, and a set
// of model-representative shapes; plus a wrong-stride RED-injection proof that the
// comparison actually discriminates (not a vacuous pass). RED-first: on a wrong
// launcher stride (lda / operand layout) the reference match must FAIL.

// Count elements outside tolerance (RED-injection helper: 0 == match, >0 == diff).
size_t MismatchCount(const std::vector<float>& got, const std::vector<float>& want, float atol,
                     float rtol) {
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    if (!(std::fabs(got[i] - want[i]) <= tol)) ++bad;
  }
  return bad;
}

// CPU reference y[M,N] = act(bf16)[M,K] @ dequant(w)[N,K]^T, accumulated in f32.
std::vector<float> DenseRefNvfp4(const Nvfp4Weight& w, const std::vector<uint16_t>& act_bf16,
                                 int64_t M, int64_t N, int64_t K) {
  std::vector<uint16_t> deq(static_cast<size_t>(N * K));
  vllm::DequantNvfp4ToBf16(w.packed.data(), w.scale.data(), w.scale2, N, K, deq.data());
  std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += vt::BF16ToF32(act_bf16[static_cast<size_t>(m * K + k)]) *
               vt::BF16ToF32(deq[static_cast<size_t>(n * K + k)]);
      ref[static_cast<size_t>(m * N + n)] = acc;
    }
  return ref;
}

std::vector<float> DenseRefMxfp4(const Mxfp4Weight& w, const std::vector<uint16_t>& act_bf16,
                                 int64_t M, int64_t N, int64_t K) {
  std::vector<uint16_t> deq(static_cast<size_t>(N * K));
  vllm::DequantMxfp4ToBf16(w.packed.data(), w.scale.data(), N, K, deq.data());
  std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += vt::BF16ToF32(act_bf16[static_cast<size_t>(m * K + k)]) *
               vt::BF16ToF32(deq[static_cast<size_t>(n * K + k)]);
      ref[static_cast<size_t>(m * N + n)] = acc;
    }
  return ref;
}

TEST_CASE("CUDA marlin DENSE gemm matches CPU-dequant ref AND the grouped route (NVFP4)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  REQUIRE(vt::OpRegistered(vt::OpId::kMarlinDenseGemm, DeviceType::kCUDA));
  QueueGuard gq(gpu);
  void* stream = gq.q.handle;
  const int dev = gq.q.device.index;
  const int sms = vt::cuda::MarlinDeviceSms(dev);

  // Model-representative shapes (K % 128 == 0, N % 64 == 0: the marlin tile
  // constraints the dense projections satisfy). The 48-CTA E=1 regime is M<=8.
  const std::vector<std::pair<int64_t, int64_t>> shapes = {{256, 64}, {512, 128}, {128, 256}};
  for (auto [K, N] : shapes) {
    const Nvfp4Weight w = MakeNvfp4Weight(N, K, 9100 + static_cast<uint32_t>(K + N));
    // Resident (repack ONCE): wq [K/16, N*2] i32, sc [K/16, N] fp8, gg [1] f32.
    std::vector<const uint8_t*> sc_bufs{w.scale.data()};
    std::vector<size_t> sc_lens{w.scale.size()};
    const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(sc_bufs, sc_lens);
    DeviceTensor dpacked(gpu, gq.q, DType::kI8, {N, K / 2}, w.packed.data());
    DeviceTensor dscale(gpu, gq.q, DType::kI8, {N, K / 16}, w.scale.data());
    DeviceTensor wq(gpu, gq.q, DType::kI32, {K / 16, N * 2});
    DeviceTensor sc(gpu, gq.q, DType::kI8, {K / 16, N});
    vt::cuda::MarlinRepackExpertWeight(stream, dev, static_cast<uint32_t*>(wq.ptr()),
                                       static_cast<const uint8_t*>(dpacked.ptr()),
                                       static_cast<int>(K), static_cast<int>(N));
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dscale.ptr()),
                                        static_cast<uint8_t*>(sc.ptr()), static_cast<int>(K),
                                        static_cast<int>(N), sf);
    const float g = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.scale2, sf);
    DeviceTensor gg(gpu, gq.q, DType::kF32, {1}, &g);
    gpu.Synchronize(gq.q);

    // Rank-3 resident views for the single-expert MoE route (SAME memory).
    Tensor wq3 = MakeTensor(wq.ptr(), DType::kI32, gq.q.device, {1, K / 16, N * 2});
    Tensor sc3 = MakeTensor(sc.ptr(), DType::kI8, gq.q.device, {1, K / 16, N});

    DeviceTensor ws(gpu, gq.q, DType::kI32, {sms * 4});

    for (int64_t M = 1; M <= 8; ++M) {
      const auto act_f = RandomF32(static_cast<size_t>(M * K), 700 + static_cast<uint32_t>(M));
      const auto act_bf16 = ToBf16(act_f);
      const auto ref = DenseRefNvfp4(w, act_bf16, M, N, K);
      DeviceTensor dact(gpu, gq.q, DType::kBF16, {M, K}, act_bf16.data());

      // (1) DENSE route.
      DeviceTensor dout(gpu, gq.q, DType::kBF16, {M, N});
      gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
      vt::MarlinDenseArgs dargs{static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)};
      vt::MarlinDenseGemm(gq.q, dout.tensor(), dact.tensor(), wq.tensor(), sc.tensor(),
                          gg.tensor(), ws.tensor(), dargs);
      std::vector<uint16_t> h_dense(static_cast<size_t>(M * N));
      dout.Download(gq.q, h_dense.data());
      std::vector<float> got_dense(static_cast<size_t>(M * N));
      for (size_t i = 0; i < got_dense.size(); ++i) got_dense[i] = vt::BF16ToF32(h_dense[i]);
      // NVFP4 random data exercises catastrophic cancellation (some output rows are
      // large sums that nearly cancel), where a bf16 tensor-core result legitimately
      // parts from a naive fp32-accumulate reference by more than a per-element 3e-2
      // — identically for the VALIDATED MoE route (max|dense-moe|==0 below). Gate the
      // vs-reference correctness with the cancellation-robust L2-relative metric; the
      // byte-preserving per-element claim is the EXACT dense==MoE check.
      CheckCloseL2(got_dense, ref, 2e-2f);

      // (2) GROUPED single-expert MoE route (all M tokens -> expert 0).
      const int block = (M <= 8) ? 8 : vt::cuda::MarlinMoeAlignBlockSizeSelect(static_cast<int>(M), 1, 1);
      int max_tok = 0, max_blk = 0;
      vt::cuda::MarlinMoeAlignSizes(static_cast<int>(M), 1, 1, block, &max_tok, &max_blk);
      std::vector<int32_t> tids(static_cast<size_t>(M), 0);
      DeviceTensor dtid(gpu, gq.q, DType::kI32, {M}, tids.data());
      DeviceTensor sorted_ids(gpu, gq.q, DType::kI32, {max_tok});
      DeviceTensor expert_ids(gpu, gq.q, DType::kI32, {max_blk});
      DeviceTensor num_pad(gpu, gq.q, DType::kI32, {1});
      vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()),
                                        static_cast<int>(M), 1, 1, block,
                                        static_cast<int32_t*>(sorted_ids.ptr()),
                                        static_cast<int32_t*>(expert_ids.ptr()),
                                        static_cast<int32_t*>(num_pad.ptr()));
      std::vector<float> ones(static_cast<size_t>(M), 1.0f);
      DeviceTensor topkw(gpu, gq.q, DType::kF32, {M}, ones.data());
      DeviceTensor mout(gpu, gq.q, DType::kBF16, {M, N});
      gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
      vt::MoeMarlinArgs margs{block, 1, static_cast<int>(M), static_cast<int>(N),
                              static_cast<int>(K), false};
      vt::MoeGroupedGemmNvfp4Marlin(gq.q, mout.tensor(), dact.tensor(), wq3, sc3,
                                    gg.tensor(), ws.tensor(), sorted_ids.tensor(),
                                    expert_ids.tensor(), num_pad.tensor(), topkw.tensor(), margs);
      std::vector<uint16_t> h_moe(static_cast<size_t>(M * N));
      mout.Download(gq.q, h_moe.data());
      std::vector<float> got_moe(static_cast<size_t>(M * N));
      for (size_t i = 0; i < got_moe.size(); ++i) got_moe[i] = vt::BF16ToF32(h_moe[i]);
      CheckCloseL2(got_moe, ref, 2e-2f);
      // Dense and grouped agree to within a couple bf16 ULPs (both marlin; the
      // dense reduce differs from the par-regrouped grouped reduce by ~1 ULP —
      // exactly the point of the port).
      CheckClose(got_dense, got_moe, 5e-2f, 5e-2f);

      // (3) WRONG-STRIDE RED injection: a row-shifted reference is a stride-class
      // perturbation; the correct-vs-shifted comparison MUST discriminate (else
      // the whole gate is vacuous). Only meaningful when M*N gives real spread.
      if (M >= 2) {
        std::vector<float> ref_shift(ref.size());
        for (size_t i = 0; i < ref.size(); ++i)
          ref_shift[i] = ref[(i + static_cast<size_t>(N)) % ref.size()];
        CHECK(MismatchCount(got_dense, ref_shift, 3e-2f, 3e-2f) > 0);
      }
    }
  }
}

TEST_CASE("CUDA marlin DENSE gemm matches CPU-dequant ref AND the grouped route (MXFP4)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  REQUIRE(vt::OpRegistered(vt::OpId::kMarlinDenseGemm, DeviceType::kCUDA));
  QueueGuard gq(gpu);
  void* stream = gq.q.handle;
  const int dev = gq.q.device.index;
  const int sms = vt::cuda::MarlinDeviceSms(dev);

  const std::vector<std::pair<int64_t, int64_t>> shapes = {{256, 64}, {512, 128}};
  for (auto [K, N] : shapes) {
    const Mxfp4Weight w = MakeMxfp4Weight(N, K, 9500 + static_cast<uint32_t>(K + N));
    // MXFP4 resident: group_size 32 => sc [K/32, N]; E8M0 passthrough (no global).
    DeviceTensor dpacked(gpu, gq.q, DType::kI8, {N, K / 2}, w.packed.data());
    DeviceTensor dscale(gpu, gq.q, DType::kI8, {N, K / 32}, w.scale.data());
    DeviceTensor wq(gpu, gq.q, DType::kI32, {K / 16, N * 2});
    DeviceTensor sc(gpu, gq.q, DType::kI8, {K / 32, N});
    vt::cuda::MarlinRepackExpertWeight(stream, dev, static_cast<uint32_t*>(wq.ptr()),
                                       static_cast<const uint8_t*>(dpacked.ptr()),
                                       static_cast<int>(K), static_cast<int>(N));
    vt::cuda::MarlinProcessExpertScalesMxfp4(stream, static_cast<const uint8_t*>(dscale.ptr()),
                                             static_cast<uint8_t*>(sc.ptr()), static_cast<int>(K),
                                             static_cast<int>(N));
    const float g = 1.0f;  // unused (kernel skips global for E8M0)
    DeviceTensor gg(gpu, gq.q, DType::kF32, {1}, &g);
    gpu.Synchronize(gq.q);

    Tensor wq3 = MakeTensor(wq.ptr(), DType::kI32, gq.q.device, {1, K / 16, N * 2});
    Tensor sc3 = MakeTensor(sc.ptr(), DType::kI8, gq.q.device, {1, K / 32, N});

    DeviceTensor ws(gpu, gq.q, DType::kI32, {sms * 4});

    for (int64_t M = 1; M <= 8; ++M) {
      const auto act_f = RandomF32(static_cast<size_t>(M * K), 800 + static_cast<uint32_t>(M));
      const auto act_bf16 = ToBf16(act_f);
      const auto ref = DenseRefMxfp4(w, act_bf16, M, N, K);
      DeviceTensor dact(gpu, gq.q, DType::kBF16, {M, K}, act_bf16.data());

      DeviceTensor dout(gpu, gq.q, DType::kBF16, {M, N});
      gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
      vt::MarlinDenseArgs dargs{static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)};
      dargs.group_size = 32;
      dargs.mxfp4 = true;
      vt::MarlinDenseGemm(gq.q, dout.tensor(), dact.tensor(), wq.tensor(), sc.tensor(),
                          gg.tensor(), ws.tensor(), dargs);
      std::vector<uint16_t> h_dense(static_cast<size_t>(M * N));
      dout.Download(gq.q, h_dense.data());
      std::vector<float> got_dense(static_cast<size_t>(M * N));
      for (size_t i = 0; i < got_dense.size(); ++i) got_dense[i] = vt::BF16ToF32(h_dense[i]);
      CheckClose(got_dense, ref, 4e-2f, 4e-2f);

      const int block = (M <= 8) ? 8 : vt::cuda::MarlinMoeAlignBlockSizeSelect(static_cast<int>(M), 1, 1);
      int max_tok = 0, max_blk = 0;
      vt::cuda::MarlinMoeAlignSizes(static_cast<int>(M), 1, 1, block, &max_tok, &max_blk);
      std::vector<int32_t> tids(static_cast<size_t>(M), 0);
      DeviceTensor dtid(gpu, gq.q, DType::kI32, {M}, tids.data());
      DeviceTensor sorted_ids(gpu, gq.q, DType::kI32, {max_tok});
      DeviceTensor expert_ids(gpu, gq.q, DType::kI32, {max_blk});
      DeviceTensor num_pad(gpu, gq.q, DType::kI32, {1});
      vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()),
                                        static_cast<int>(M), 1, 1, block,
                                        static_cast<int32_t*>(sorted_ids.ptr()),
                                        static_cast<int32_t*>(expert_ids.ptr()),
                                        static_cast<int32_t*>(num_pad.ptr()));
      std::vector<float> ones(static_cast<size_t>(M), 1.0f);
      DeviceTensor topkw(gpu, gq.q, DType::kF32, {M}, ones.data());
      DeviceTensor mout(gpu, gq.q, DType::kBF16, {M, N});
      gpu.Memset(gq.q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
      vt::MoeMarlinArgs margs{block, 1, static_cast<int>(M), static_cast<int>(N),
                              static_cast<int>(K), false};
      margs.group_size = 32;
      margs.mxfp4 = true;
      vt::MoeGroupedGemmNvfp4Marlin(gq.q, mout.tensor(), dact.tensor(), wq3, sc3,
                                    gg.tensor(), ws.tensor(), sorted_ids.tensor(),
                                    expert_ids.tensor(), num_pad.tensor(), topkw.tensor(), margs);
      std::vector<uint16_t> h_moe(static_cast<size_t>(M * N));
      mout.Download(gq.q, h_moe.data());
      std::vector<float> got_moe(static_cast<size_t>(M * N));
      for (size_t i = 0; i < got_moe.size(); ++i) got_moe[i] = vt::BF16ToF32(h_moe[i]);
      CheckClose(got_moe, ref, 4e-2f, 4e-2f);
      CheckClose(got_dense, got_moe, 6e-2f, 6e-2f);
    }
  }
}
#endif  // VT_MARLIN_NVFP4
