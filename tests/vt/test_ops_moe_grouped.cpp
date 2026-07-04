// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// M2.4 — the fused-MoE grouped NVFP4 GEMM (vt::MoeGroupedGemmNvfp4) + the fused
// silu-mul activation (vt::MoeSiluMul). Validates that the grouped GEMM over all
// (token, expert) pairs matches, per output row, the single-expert reference
// (DequantNvfp4ToBf16 + per-row f32 matmul) that the per-expert MoE loop uses —
// so the fused MoE and the per-expert MoE agree. Grouped GEMM is CUDA-only
// (skips cleanly with no GPU); MoeSiluMul runs on CPU and CUDA.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
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
        acc += act_f[static_cast<size_t>(r * K + k)] *
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
        acc += act_f[static_cast<size_t>(p * K + k)] * vt::BF16ToF32(deq[static_cast<size_t>(n * K + k)]);
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
