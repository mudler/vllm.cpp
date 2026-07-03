// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA op kernels vs the CPU reference ops on random inputs (fixed seeds).
// Guarded like test_cuda_backend.cpp: skips cleanly when no GPU is present.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
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

std::vector<float> RandomF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// Packs an f32 master vector into the byte representation of dt (f32 or bf16).
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

std::vector<float> Unpack(const std::vector<uint8_t>& b, DType dt) {
  const size_t n = b.size() / vt::SizeOf(dt);
  std::vector<float> out(n);
  if (dt == DType::kF32) {
    std::memcpy(out.data(), b.data(), b.size());
  } else {
    REQUIRE(dt == DType::kBF16);
    const auto* p = reinterpret_cast<const uint16_t*>(b.data());
    for (size_t i = 0; i < n; ++i) out[i] = vt::BF16ToF32(p[i]);
  }
  return out;
}

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0;
  size_t first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    if (!(std::fabs(got[i] - want[i]) <= tol)) {  // catches NaN too
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

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

// Device buffer + tensor view; uploads on construction when host data given.
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

// Input/output dtype combos per the M0.6 plan, with comparison tolerances:
// f32-in/f32-out 1e-5; bf16-in/f32-out 2e-3. bf16 outputs (compared after
// BF16ToF32) get rtol 8e-3 >= one bf16 ulp (2^-7 ≈ 7.8e-3 relative): the GPU tree
// reduction and the CPU sequential sum legitimately differ by ~1e-6 in f32,
// which can flip the final bf16 rounding by one ulp on large rows.
struct Combo {
  DType in;
  DType out;
  float atol;
  float rtol;
};
constexpr Combo kCombos[] = {
    {DType::kF32, DType::kF32, 1e-5f, 1e-5f},
    {DType::kBF16, DType::kF32, 2e-3f, 2e-3f},
    {DType::kBF16, DType::kBF16, 4e-3f, 8e-3f},
};

void RunRmsNormCase(int64_t t, int64_t h, const Combo& c, bool gemma, bool fused,
                    uint32_t seed) {
  const auto xf = RandomF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandomF32(static_cast<size_t>(h), seed + 1);
  const auto resf = RandomF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, c.in);
  const auto wb = Pack(wf, c.in);
  const vt::RmsNormArgs args{1e-6f, gemma};

  // CPU reference.
  std::vector<uint8_t> out_cpu(static_cast<size_t>(t * h) * vt::SizeOf(c.out));
  std::vector<float> res_cpu = resf;
  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), c.in, Cpu(), {t, h});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), c.in, Cpu(), {h});
  Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {t, h});
  Tensor tr = MakeTensor(res_cpu.data(), DType::kF32, Cpu(), {t, h});
  Queue cq{Cpu(), nullptr};
  vt::RmsNorm(cq, to, tx, tw, args, fused ? &tr : nullptr);

  // CUDA.
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, c.in, {t, h}, xb.data());
  DeviceTensor dw(gpu, gq.q, c.in, {h}, wb.data());
  DeviceTensor dout(gpu, gq.q, c.out, {t, h});
  DeviceTensor dres(gpu, gq.q, DType::kF32, {t, h}, resf.data());
  vt::RmsNorm(gq.q, dout.tensor(), dx.tensor(), dw.tensor(), args,
              fused ? &dres.tensor() : nullptr);
  std::vector<uint8_t> out_gpu(out_cpu.size());
  dout.Download(gq.q, out_gpu.data());

  CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
  if (fused) {
    // The residual stream is f32 on both sides; the add order is elementwise
    // and identical, so it matches tightly regardless of x's dtype.
    std::vector<float> res_gpu(static_cast<size_t>(t * h));
    dres.Download(gq.q, res_gpu.data());
    CheckClose(res_gpu, res_cpu, 1e-6f, 1e-6f);
  }
}

}  // namespace

TEST_CASE("CUDA rmsnorm matches CPU across row sizes and dtypes") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  // Odd sizes (1, 127, 129) exercise the shared-memory reduction tail.
  const int64_t sizes[] = {1, 8, 127, 128, 129, 4096};
  uint32_t seed = 100;
  for (int64_t h : sizes) {
    for (const Combo& c : kCombos) {
      CAPTURE(h);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      RunRmsNormCase(3, h, c, /*gemma=*/false, /*fused=*/false, seed);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA rmsnorm gemma variant matches CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 500;
  for (const Combo& c : kCombos) {
    CAPTURE(static_cast<int>(c.in));
    CAPTURE(static_cast<int>(c.out));
    RunRmsNormCase(4, 128, c, /*gemma=*/true, /*fused=*/false, seed);
    seed += 10;
  }
}

TEST_CASE("CUDA rmsnorm fused residual matches CPU and updates residual") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 700;
  for (int64_t h : {127, 128}) {
    for (const Combo& c : kCombos) {
      CAPTURE(h);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      RunRmsNormCase(3, h, c, /*gemma=*/false, /*fused=*/true, seed);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA silu_and_mul matches CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  uint32_t seed = 900;
  for (int64_t d : {173, 256}) {
    for (const Combo& c : kCombos) {
      CAPTURE(d);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      const int64_t t = 5;
      const auto xf = RandomF32(static_cast<size_t>(t * 2 * d), seed);
      const auto xb = Pack(xf, c.in);

      std::vector<uint8_t> out_cpu(static_cast<size_t>(t * d) * vt::SizeOf(c.out));
      Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), c.in, Cpu(), {t, 2 * d});
      Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {t, d});
      Queue cq{Cpu(), nullptr};
      vt::SiluAndMul(cq, to, tx);

      QueueGuard gq(gpu);
      DeviceTensor dx(gpu, gq.q, c.in, {t, 2 * d}, xb.data());
      DeviceTensor dout(gpu, gq.q, c.out, {t, d});
      vt::SiluAndMul(gq.q, dout.tensor(), dx.tensor());
      std::vector<uint8_t> out_gpu(out_cpu.size());
      dout.Download(gq.q, out_gpu.data());

      CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA embedding matches CPU (i32 and i64 ids)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t v = 64, h = 33, t = 17;
  uint32_t seed = 1200;
  for (DType id_dt : {DType::kI32, DType::kI64}) {
    for (const Combo& c : kCombos) {
      CAPTURE(static_cast<int>(id_dt));
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      const auto tf = RandomF32(static_cast<size_t>(v * h), seed);
      const auto tb = Pack(tf, c.in);
      std::mt19937 rng(seed + 1);
      std::uniform_int_distribution<int64_t> dist(0, v - 1);
      std::vector<int32_t> ids32(static_cast<size_t>(t));
      std::vector<int64_t> ids64(static_cast<size_t>(t));
      for (int64_t i = 0; i < t; ++i) {
        ids64[static_cast<size_t>(i)] = dist(rng);
        ids32[static_cast<size_t>(i)] = static_cast<int32_t>(ids64[static_cast<size_t>(i)]);
      }
      const void* ids_host = id_dt == DType::kI32 ? static_cast<const void*>(ids32.data())
                                                  : static_cast<const void*>(ids64.data());

      std::vector<uint8_t> out_cpu(static_cast<size_t>(t * h) * vt::SizeOf(c.out));
      Tensor ttab = MakeTensor(const_cast<uint8_t*>(tb.data()), c.in, Cpu(), {v, h});
      Tensor tids = MakeTensor(const_cast<void*>(ids_host), id_dt, Cpu(), {t});
      Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {t, h});
      Queue cq{Cpu(), nullptr};
      vt::Embedding(cq, to, ttab, tids);

      QueueGuard gq(gpu);
      DeviceTensor dtab(gpu, gq.q, c.in, {v, h}, tb.data());
      DeviceTensor dids(gpu, gq.q, id_dt, {t}, ids_host);
      DeviceTensor dout(gpu, gq.q, c.out, {t, h});
      vt::Embedding(gq.q, dout.tensor(), dtab.tensor(), dids.tensor());
      std::vector<uint8_t> out_gpu(out_cpu.size());
      dout.Download(gq.q, out_gpu.data());

      CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA embedding: out-of-range device id throws with the id") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t v = 8, h = 4;
  const auto tf = RandomF32(static_cast<size_t>(v * h), 42);

  for (int32_t bad : {int32_t{8}, int32_t{-3}}) {
    CAPTURE(bad);
    std::vector<int32_t> ids = {1, bad, 2};
    QueueGuard gq(gpu);
    DeviceTensor dtab(gpu, gq.q, DType::kF32, {v, h}, tf.data());
    DeviceTensor dids(gpu, gq.q, DType::kI32, {3}, ids.data());
    DeviceTensor dout(gpu, gq.q, DType::kF32, {3, h});
    bool threw = false;
    try {
      vt::Embedding(gq.q, dout.tensor(), dtab.tensor(), dids.tensor());
    } catch (const std::runtime_error& e) {
      threw = true;
      const std::string msg = e.what();
      CAPTURE(msg);
      CHECK(msg.find("embedding") != std::string::npos);
      CHECK(msg.find(std::to_string(bad)) != std::string::npos);
    }
    CHECK(threw);
    // The kernel clamps bad ids, so the stream stays healthy after the throw.
    CHECK_NOTHROW(gpu.Synchronize(gq.q));
  }
}

TEST_CASE("CUDA matmul (cuBLASLt) matches CPU on odd sizes") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  // Matmul-specific tolerances: cuBLASLt may reduce over K in a different
  // order than the CPU triple loop (split-K, tensor-core tiles), so even the
  // all-f32 combo gets 1e-4 instead of the elementwise 1e-5. bf16 inputs are
  // identical bytes on both sides (products are exact in f32), so bf16-in
  // stays at 2e-3 and bf16-out at one output ulp, as in kCombos.
  const Combo combos[] = {
      {DType::kF32, DType::kF32, 1e-4f, 1e-4f},
      {DType::kBF16, DType::kF32, 2e-3f, 2e-3f},
      {DType::kBF16, DType::kBF16, 4e-3f, 8e-3f},
  };
  struct Dims {
    int64_t m, k, n;
  };
  // Odd shapes exercise tile tails; {1,257,1} is a pure K-reduction.
  const Dims dims[] = {{17, 31, 13}, {64, 128, 32}, {1, 257, 1}};
  uint32_t seed = 3000;
  for (const Dims& d : dims) {
    for (const Combo& c : combos) {
      CAPTURE(d.m);
      CAPTURE(d.k);
      CAPTURE(d.n);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      const auto af = RandomF32(static_cast<size_t>(d.m * d.k), seed);
      const auto bf = RandomF32(static_cast<size_t>(d.k * d.n), seed + 1);
      const auto ab = Pack(af, c.in);
      const auto bb = Pack(bf, c.in);

      // CPU reference on the same packed inputs.
      std::vector<uint8_t> out_cpu(static_cast<size_t>(d.m * d.n) * vt::SizeOf(c.out));
      Tensor ta = MakeTensor(const_cast<uint8_t*>(ab.data()), c.in, Cpu(), {d.m, d.k});
      Tensor tb = MakeTensor(const_cast<uint8_t*>(bb.data()), c.in, Cpu(), {d.k, d.n});
      Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {d.m, d.n});
      Queue cq{Cpu(), nullptr};
      vt::Matmul(cq, to, ta, tb);

      // CUDA.
      QueueGuard gq(gpu);
      DeviceTensor da(gpu, gq.q, c.in, {d.m, d.k}, ab.data());
      DeviceTensor db(gpu, gq.q, c.in, {d.k, d.n}, bb.data());
      DeviceTensor dout(gpu, gq.q, c.out, {d.m, d.n});
      vt::Matmul(gq.q, dout.tensor(), da.tensor(), db.tensor());
      std::vector<uint8_t> out_gpu(out_cpu.size());
      dout.Download(gq.q, out_gpu.data());

      CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA matmul: unsupported dtype combo (f16 inputs) throws naming it") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  // The op-level validation admits f16 inputs; the cuBLASLt kernel does not
  // implement them and must throw before touching the data (left unset).
  DeviceTensor da(gpu, gq.q, DType::kF16, {4, 8});
  DeviceTensor db(gpu, gq.q, DType::kF16, {8, 3});
  DeviceTensor dout(gpu, gq.q, DType::kF32, {4, 3});
  bool threw = false;
  try {
    vt::Matmul(gq.q, dout.tensor(), da.tensor(), db.tensor());
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string msg = e.what();
    CAPTURE(msg);
    CHECK(msg.find("matmul") != std::string::npos);
    CHECK(msg.find("f16") != std::string::npos);
  }
  CHECK(threw);
  CHECK_NOTHROW(gpu.Synchronize(gq.q));
}

TEST_CASE("CUDA rope_neox matches CPU (partial rotary, i32/i64 positions)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t t = 7, hq = 4, hk = 2, d = 32;
  uint32_t seed = 2000;
  for (int rotary : {32, 16}) {
    for (DType dt : {DType::kF32, DType::kBF16}) {
      for (DType pos_dt : {DType::kI64, DType::kI32}) {
        CAPTURE(rotary);
        CAPTURE(static_cast<int>(dt));
        CAPTURE(static_cast<int>(pos_dt));
        // bf16 rtol allows one bf16 ulp, as in kCombos.
        const float atol = dt == DType::kF32 ? 1e-5f : 4e-3f;
        const float rtol = dt == DType::kF32 ? 1e-5f : 8e-3f;
        const auto qf = RandomF32(static_cast<size_t>(t * hq * d), seed);
        const auto kf = RandomF32(static_cast<size_t>(t * hk * d), seed + 1);
        const auto qb = Pack(qf, dt);
        const auto kb = Pack(kf, dt);
        std::mt19937 rng(seed + 2);
        std::uniform_int_distribution<int64_t> dist(0, 131072);
        std::vector<int64_t> pos64(static_cast<size_t>(t));
        std::vector<int32_t> pos32(static_cast<size_t>(t));
        for (int64_t i = 0; i < t; ++i) {
          pos64[static_cast<size_t>(i)] = dist(rng);
          pos32[static_cast<size_t>(i)] = static_cast<int32_t>(pos64[static_cast<size_t>(i)]);
        }
        const void* pos_host = pos_dt == DType::kI32 ? static_cast<const void*>(pos32.data())
                                                     : static_cast<const void*>(pos64.data());
        const vt::RopeArgs args{10000.0f, rotary};

        // CPU reference (in place on copies).
        std::vector<uint8_t> q_cpu = qb, k_cpu = kb;
        Tensor tq = MakeTensor(q_cpu.data(), dt, Cpu(), {t, hq, d});
        Tensor tk = MakeTensor(k_cpu.data(), dt, Cpu(), {t, hk, d});
        Tensor tp = MakeTensor(const_cast<void*>(pos_host), pos_dt, Cpu(), {t});
        Queue cq{Cpu(), nullptr};
        vt::RopeNeox(cq, tq, tk, tp, args);

        // CUDA (in place on device copies of the same packed inputs).
        QueueGuard gq(gpu);
        DeviceTensor dq(gpu, gq.q, dt, {t, hq, d}, qb.data());
        DeviceTensor dk(gpu, gq.q, dt, {t, hk, d}, kb.data());
        DeviceTensor dp(gpu, gq.q, pos_dt, {t}, pos_host);
        vt::RopeNeox(gq.q, dq.tensor(), dk.tensor(), dp.tensor(), args);
        std::vector<uint8_t> q_gpu(qb.size()), k_gpu(kb.size());
        dq.Download(gq.q, q_gpu.data());
        dk.Download(gq.q, k_gpu.data());

        CheckClose(Unpack(q_gpu, dt), Unpack(q_cpu, dt), atol, rtol);
        CheckClose(Unpack(k_gpu, dt), Unpack(k_cpu, dt), atol, rtol);
        seed += 10;
      }
    }
  }
}
