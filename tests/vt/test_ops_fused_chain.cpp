// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// TDR Phase 0 parity gate. Asserts the kFusedAddRmsNorm recipe realized three
// ways is BIT-IDENTICAL:
//   Tier-0 composite (VT_FUSED_TIER=0) == Tier-1 interpreter (VT_FUSED_TIER=1)
//   == the existing vt::RmsNorm(residual) golden (the fused_add_rms_norm path at
//   src/vllm/model_executor/models/qwen3_5.cpp:2322).
// The recipe is the single source of truth; every tier must agree to the bit.
// CPU is the primary bit-identical gate; the CUDA section runs on dgx (skips
// cleanly when no GPU backend is registered) and checks the same equalities
// among the GPU realizations.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormArgs;
using vt::Tensor;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

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

std::vector<float> RandF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<uint8_t> Pack(const std::vector<float>& f, DType dt) {
  std::vector<uint8_t> out(f.size() * vt::SizeOf(dt));
  if (dt == DType::kF32) {
    std::memcpy(out.data(), f.data(), out.size());
  } else {
    REQUIRE(dt == DType::kBF16);
    auto* p = reinterpret_cast<uint16_t*>(out.data());
    for (size_t i = 0; i < f.size(); ++i) p[i] = vt::F32ToBF16(f[i]);
  }
  return out;
}

void SetTier(int tier) { setenv("VT_FUSED_TIER", tier == 1 ? "1" : "0", 1); }

// Runs the golden RmsNorm(residual) and both FusedChain tiers on CPU with the
// SAME inputs, and asserts all three outputs (and residual streams) are byte-
// identical. dtypes cover the shapes the real call-site hits (bf16 x/weight with
// an f32 or bf16 residual) plus the all-f32 reference.
void RunCpuCase(int64_t t, int64_t h, DType xdt, DType outdt, DType resdt, uint32_t seed) {
  const auto xf = RandF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandF32(static_cast<size_t>(h), seed + 1);
  const auto rf = RandF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, xdt);
  const auto wb = Pack(wf, xdt);  // weight follows x (the model materializes it so)
  const auto rb = Pack(rf, resdt);
  const float eps = 1e-6f;
  const size_t obytes = static_cast<size_t>(t * h) * vt::SizeOf(outdt);
  const size_t rbytes = rb.size();

  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), xdt, Cpu(), {t, h});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), xdt, Cpu(), {h});
  Queue q{Cpu(), nullptr};

  // Golden: the existing fused_add_rms_norm path (gemma weight = 1+w).
  std::vector<uint8_t> out_g(obytes), res_g = rb;
  Tensor tog = MakeTensor(out_g.data(), outdt, Cpu(), {t, h});
  Tensor trg = MakeTensor(res_g.data(), resdt, Cpu(), {t, h});
  vt::RmsNorm(q, tog, tx, tw, RmsNormArgs{eps, /*gemma=*/true}, &trg);

  // Tier-0 composite.
  SetTier(0);
  std::vector<uint8_t> out_0(obytes), res_0 = rb;
  Tensor to0 = MakeTensor(out_0.data(), outdt, Cpu(), {t, h});
  Tensor tr0 = MakeTensor(res_0.data(), resdt, Cpu(), {t, h});
  vt::FusedChain(q, to0, tx, tw, &tr0, vt::kFusedAddRmsNorm, eps);

  // Tier-1 interpreter.
  SetTier(1);
  std::vector<uint8_t> out_1(obytes), res_1 = rb;
  Tensor to1 = MakeTensor(out_1.data(), outdt, Cpu(), {t, h});
  Tensor tr1 = MakeTensor(res_1.data(), resdt, Cpu(), {t, h});
  vt::FusedChain(q, to1, tx, tw, &tr1, vt::kFusedAddRmsNorm, eps);
  SetTier(0);

  // Bit-identical: composite == golden, interpreter == golden (hence all three).
  CHECK(std::memcmp(out_0.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(out_1.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(res_0.data(), res_g.data(), rbytes) == 0);
  CHECK(std::memcmp(res_1.data(), res_g.data(), rbytes) == 0);
}

}  // namespace

TEST_CASE("fused_chain kFusedAddRmsNorm: Tier-0 == Tier-1 == RmsNorm(residual), bit-identical") {
  const int64_t sizes[] = {1, 7, 8, 127, 128, 129, 512};
  uint32_t seed = 20;
  for (int64_t h : sizes) {
    CAPTURE(h);
    // Primary gate: all-f32 (full-precision bit-identical).
    RunCpuCase(3, h, DType::kF32, DType::kF32, DType::kF32, seed);
    seed += 7;
    // The real call-site: bf16 x/weight, f32 residual, bf16 output.
    RunCpuCase(3, h, DType::kBF16, DType::kBF16, DType::kF32, seed);
    seed += 7;
    // bf16 residual (vLLM model-dtype residual): rounding is identical per tier.
    RunCpuCase(3, h, DType::kBF16, DType::kBF16, DType::kBF16, seed);
    seed += 7;
  }
}

TEST_CASE("fused_chain validates operands at the chokepoint") {
  Queue q{Cpu(), nullptr};
  std::vector<float> x(4, 1.0f), w(2, 1.0f), out(4, 0.0f);
  Tensor tx = MakeTensor(x.data(), DType::kF32, Cpu(), {2, 2});
  Tensor to = MakeTensor(out.data(), DType::kF32, Cpu(), {2, 2});
  // weight size mismatch ([1] != H=2) → wrapper throws before dispatch.
  Tensor tbad = MakeTensor(w.data(), DType::kF32, Cpu(), {1});
  CHECK_THROWS_AS(vt::FusedChain(q, to, tx, tbad, nullptr, vt::kFusedAddRmsNorm, 1e-6f),
                  std::runtime_error);
}

// ---------------------------------------------------------------------------
// CUDA section: the same three realizations must agree bit-for-bit on the GPU
// (all use the identical f32 tree reduction). Skips when no CUDA backend exists.

namespace {

using vt::GetBackend;

bool HasCuda() {
  try {
    GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

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
    t_ = MakeTensor(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
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

void RunCudaCase(int64_t t, int64_t h, DType xdt, DType outdt, DType resdt, uint32_t seed) {
  const auto xf = RandF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandF32(static_cast<size_t>(h), seed + 1);
  const auto rf = RandF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, xdt);
  const auto wb = Pack(wf, xdt);
  const auto rb = Pack(rf, resdt);
  const float eps = 1e-6f;
  const size_t obytes = static_cast<size_t>(t * h) * vt::SizeOf(outdt);

  Backend& gpu = GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);

  DeviceTensor dx(gpu, g.q, xdt, {t, h}, xb.data());
  DeviceTensor dw(gpu, g.q, xdt, {h}, wb.data());

  auto run = [&](int tier, bool golden, std::vector<uint8_t>& out_bytes,
                 std::vector<uint8_t>& res_bytes) {
    SetTier(tier);
    DeviceTensor dout(gpu, g.q, outdt, {t, h});
    DeviceTensor dres(gpu, g.q, resdt, {t, h}, rb.data());
    if (golden) {
      vt::RmsNorm(g.q, dout.tensor(), dx.tensor(), dw.tensor(), RmsNormArgs{eps, true},
                  &dres.tensor());
    } else {
      vt::FusedChain(g.q, dout.tensor(), dx.tensor(), dw.tensor(), &dres.tensor(),
                     vt::kFusedAddRmsNorm, eps);
    }
    out_bytes.assign(obytes, 0);
    res_bytes.assign(rb.size(), 0);
    dout.Download(g.q, out_bytes.data());
    dres.Download(g.q, res_bytes.data());
  };

  std::vector<uint8_t> out_g, res_g, out_0, res_0, out_1, res_1;
  run(0, /*golden=*/true, out_g, res_g);
  run(0, /*golden=*/false, out_0, res_0);
  run(1, /*golden=*/false, out_1, res_1);
  SetTier(0);

  CHECK(std::memcmp(out_0.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(out_1.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(res_0.data(), res_g.data(), rb.size()) == 0);
  CHECK(std::memcmp(res_1.data(), res_g.data(), rb.size()) == 0);
}

}  // namespace

TEST_CASE("CUDA fused_chain: Tier-0 == Tier-1 == RmsNorm(residual), bit-identical") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 4000;
  for (int64_t h : {1, 127, 128, 129, 512}) {
    CAPTURE(h);
    RunCudaCase(3, h, DType::kF32, DType::kF32, DType::kF32, seed);
    seed += 7;
    RunCudaCase(3, h, DType::kBF16, DType::kBF16, DType::kF32, seed);
    seed += 7;
    RunCudaCase(3, h, DType::kBF16, DType::kBF16, DType::kBF16, seed);
    seed += 7;
  }
}
