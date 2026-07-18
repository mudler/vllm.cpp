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
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

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
TEST_CASE("CUDA moe_router_topk parallel == serial byte-for-byte (adversarial)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

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

  for (DType dt : {DType::kF32, DType::kBF16}) {
    for (bool renorm : {true, false}) {
      for (int64_t T : {int64_t{1}, int64_t{8}, int64_t{16}}) {
        // 35B routing shape: E=128, top-8. Random distinct logits.
        {
          const int64_t E = 128;
          std::vector<float> lg(static_cast<size_t>(T * E));
          std::mt19937 rng(1234u + static_cast<uint32_t>(T));
          std::uniform_real_distribution<float> d(-4.0f, 4.0f);
          for (auto& v : lg) v = d(rng);
          run(T, E, 8, renorm, dt, lg);
        }
        // Exact-tie storm: blocks of identical logits so many experts tie at
        // the max; the tie-break (lowest index) must agree across paths.
        {
          const int64_t E = 128;
          std::vector<float> lg(static_cast<size_t>(T * E));
          for (int64_t t = 0; t < T; ++t)
            for (int64_t e = 0; e < E; ++e)
              lg[static_cast<size_t>(t * E + e)] = static_cast<float>((e / 4) % 5);  // 5 tie groups
          run(T, E, 8, renorm, dt, lg);
        }
        // Uneven load + near-ties around the top-k boundary (E=256, top-8).
        {
          const int64_t E = 256;
          std::vector<float> lg(static_cast<size_t>(T * E));
          for (int64_t t = 0; t < T; ++t)
            for (int64_t e = 0; e < E; ++e)
              lg[static_cast<size_t>(t * E + e)] =
                  (e < 12 ? 3.0f : 0.0f) + 1e-4f * static_cast<float>((e * 7 + t) % 3);
          run(T, E, 8, renorm, dt, lg);
        }
      }
    }
  }
}

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
#endif  // VT_MARLIN_NVFP4
