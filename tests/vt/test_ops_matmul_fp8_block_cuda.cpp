// vllm.cpp — the block-scaled FP8 GEMM, CUDA arm, against the CPU reference arm
// of the SAME op.
//
// VT-MATMUL-FP8-BLOCK-CUDA (.agents/specs/vt-matmul-fp8-block-cuda.md), issue
// #1189 milestone M5. Pinned oracle: vLLM
// 5559679229bc961848b121ccdeaa8fa5d79bec98, asserted as the local checkout's
// HEAD before every anchor below was read.
//
// THIS FILE HAS NEVER RUN AGAINST A DEVICE. It is registered, it builds, and on
// a host with no CUDA device it SKIPS — and a skip is not a pass. #1189 M5's
// spec records the on-hardware leg as OWED under `## Owed`, in the commit body
// and in the pull request body, and no number produced here appears in any
// document as a measurement. Whoever first runs it on a leased sm_12xa box owns
// filling in that section.
//
// WHAT IT MEASURES WHEN IT DOES RUN. The CUDA kernel against
// `vt::MatmulFp8BlockScaled`'s CPU arm — the reference #1189 M2 landed
// (`770e49486`) precisely to be this comparison's oracle. That arm is upstream's
// `native_w8a8_block_matmul` (tests/kernels/quant_utils.py) ported whole, so
// comparing against it IS comparing against upstream's reference, and the
// criterion below is upstream's own.
//
//   G2  upstream's `test_w8a8_block_fp8_cutlass_matmul`
//       (tests/kernels/quantization/test_block_fp8.py) ported whole:
//       M=32, N=576, K=7168, block [128,128], bf16 out, scales U(0,1)*1e-2,
//       compared at `rel_diff < 0.001` computed by upstream's own formula. N=576
//       is 4*128 + 64 — a RAGGED final block, which is why upstream chose it
//       (DSV3's kv_a_proj_with_mqa).
//   G6  the test's own precondition, and it runs on EVERY host including one
//       with no device: every shape the cases below use must be accepted by
//       `Fp8BlockScaledRefusalFor`. A grid that the arm refuses would make every
//       case throw, and a harness that then reported "skipped" would be lying in
//       the other direction.
//   G7  the M sweep across all three tile configs, with the dispatch counter
//       asserted to advance on the config the heuristic named.
//   G8  the f32 sink equals the bf16 result cast up.
//   G9  the refusals, on a device, by message.
//
// (G1, G3, G4 and G5 are the host-tier cases in
// `tests/vt/test_fp8_block_scaled_dispatch.cpp`, which DOES run here. The
// numbering is the spec's.)
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/cuda/fp8_block_scaled_dispatch.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using vt::cuda::Fp8BlockScaledConfig;
using vt::cuda::Fp8BlockScaledConfigFor;
using vt::cuda::Fp8BlockScaledRefusal;
using vt::cuda::Fp8BlockScaledRefusalFor;
using vt::cuda::Fp8BlockScaledStats;
using vt::cuda::Fp8BlockScaledStatsSnapshot;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
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

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Random fp8-e4m3fn BYTES, not values. Both arms consume the same bytes, so
// nothing here needs an encoder and the comparison cannot be contaminated by one
// — upstream's CUTLASS test quantizes once and feeds both arms too. 0x7F/0xFF
// are the only NaN encodings of "fn" and are excluded.
std::vector<uint8_t> RandomFp8Bytes(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> u(0, 255);
  std::vector<uint8_t> v(n);
  for (auto& b : v) {
    int byte = u(rng);
    if ((byte & 0x7F) == 0x7F) byte &= ~0x7;
    b = static_cast<uint8_t>(byte);
  }
  return v;
}

// upstream: `As = torch.rand(M, k_tiles) * factor_for_scale` with
// `factor_for_scale = 1e-2` (test_block_fp8.py). Uniform on [0, 1e-2).
std::vector<float> RandomScales(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& s : v) s = u(rng) * 1e-2f;
  return v;
}

struct BlockCase {
  int64_t m = 0, n = 0, k = 0;
  int block_n = 128, block_k = 128;
};

// The CPU reference arm of the SAME op, run on a CPU queue in this process.
std::vector<float> ReferenceBf16(const BlockCase& c, const std::vector<uint8_t>& a_fp8,
                                 const std::vector<float>& a_scale,
                                 const std::vector<uint8_t>& b_fp8,
                                 const std::vector<float>& b_scale) {
  const int64_t k_tiles = CDiv(c.k, c.block_k);
  const int64_t n_tiles = CDiv(c.n, c.block_n);
  std::vector<uint16_t> out(static_cast<size_t>(c.m) * c.n, 0);

  Backend& b = vt::GetBackend(DeviceType::kCPU);
  QueueGuard g(b);
  Tensor ta = MakeTensor(const_cast<uint8_t*>(a_fp8.data()), DType::kI8, Cpu(), {c.m, c.k});
  Tensor tas = MakeTensor(const_cast<float*>(a_scale.data()), DType::kF32, Cpu(), {c.m, k_tiles});
  Tensor tb = MakeTensor(const_cast<uint8_t*>(b_fp8.data()), DType::kI8, Cpu(), {c.n, c.k});
  Tensor tbs =
      MakeTensor(const_cast<float*>(b_scale.data()), DType::kF32, Cpu(), {n_tiles, k_tiles});
  Tensor to = MakeTensor(out.data(), DType::kBF16, Cpu(), {c.m, c.n});
  vt::MatmulFp8BlockScaled(g.q, to, ta, tas, tb, tbs, c.block_n, c.block_k);
  b.Synchronize(g.q);

  std::vector<float> f(out.size());
  for (size_t i = 0; i < out.size(); ++i) f[i] = vt::BF16ToF32(out[i]);
  return f;
}

// The CUDA arm, same bytes, same scales, bf16 out.
std::vector<float> DeviceBf16(const BlockCase& c, const std::vector<uint8_t>& a_fp8,
                              const std::vector<float>& a_scale,
                              const std::vector<uint8_t>& b_fp8,
                              const std::vector<float>& b_scale) {
  const int64_t k_tiles = CDiv(c.k, c.block_k);
  const int64_t n_tiles = CDiv(c.n, c.block_n);
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor da(b, g.q, DType::kI8, {c.m, c.k}, a_fp8.data());
  DeviceTensor das(b, g.q, DType::kF32, {c.m, k_tiles}, a_scale.data());
  DeviceTensor db(b, g.q, DType::kI8, {c.n, c.k}, b_fp8.data());
  DeviceTensor dbs(b, g.q, DType::kF32, {n_tiles, k_tiles}, b_scale.data());
  DeviceTensor dout(b, g.q, DType::kBF16, {c.m, c.n});
  vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                           dbs.tensor(), c.block_n, c.block_k);
  std::vector<uint16_t> raw(static_cast<size_t>(c.m) * c.n);
  dout.Download(g.q, raw.data());
  std::vector<float> f(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) f[i] = vt::BF16ToF32(raw[i]);
  return f;
}

// upstream's criterion, its formula, verbatim
// (test_block_fp8.py::test_w8a8_block_fp8_cutlass_matmul):
//
//     rel_diff = mean(|out - ref|) / mean(|ref|);  assert rel_diff < 0.001
//
// It is the right tight criterion here for the reason the spec records: the two
// arms differ only in the K-reduction order inside a block and in associating
// `(part * a_s) * b_s` instead of `part * (a_s * b_s)` — at most one f32 ULP per
// K-block — and this is the bound upstream itself accepts between exactly these
// two implementations.
double RelDiff(const std::vector<float>& got, const std::vector<float>& want) {
  REQUIRE(got.size() == want.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    num += std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    den += std::fabs(static_cast<double>(want[i]));
  }
  REQUIRE(den > 0.0);
  return num / den;
}

// A MEAN CANNOT SEE ONE WRONG ELEMENT, and one wrong element is exactly what a
// ragged final block or an off-by-one scale index produces. This is a coarse
// per-element guard for that failure, not a precision bound: a mis-scaled
// element is wrong by a FACTOR, and this catches it, while the mean-relative
// criterion above is the tight one. The scale term is the run's own mean
// magnitude, because an element whose true value sits near zero after
// cancellation has an unbounded relative error and bounding it would only make
// the test flaky.
void CheckNoElementIsCatastrophic(const std::vector<float>& got, const std::vector<float>& want) {
  REQUIRE(got.size() == want.size());
  double mean_abs = 0.0;
  for (float w : want) mean_abs += std::fabs(static_cast<double>(w));
  mean_abs /= static_cast<double>(want.size());
  size_t bad = 0, first_bad = 0;
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    const double tol = 0.05 * mean_abs + 0.05 * std::fabs(static_cast<double>(want[i]));
    if (d > worst) worst = d;
    if (!(d <= tol)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(want[first_bad]);
    CAPTURE(worst);
    CAPTURE(mean_abs);
  }
  CHECK(bad == 0);
}

// AN ALL-ZERO OUTPUT PASSES EVERY RATIO. Both arms are asked to be non-vacuous.
void CheckNonVacuous(const std::vector<float>& v) {
  size_t nonzero = 0;
  for (float x : v)
    if (x != 0.0f) ++nonzero;
  CAPTURE(nonzero);
  CAPTURE(v.size());
  CHECK(nonzero * 10 > v.size() * 9);
}

// The full grid this file drives, in one place, so G6 can assert the arm accepts
// every shape in it before any of them is run.
const std::vector<BlockCase>& Grid() {
  static const std::vector<BlockCase> grid = {
      // upstream's own CUTLASS case: DSV3 kv_a_proj_with_mqa, ragged N.
      {32, 576, 7168, 128, 128},
      // the M sweep, one per tile config and both clauses of the swap_ab OR.
      {1, 512, 1024, 128, 128},     // swapab: M <= 64, and decode's own shape
      {7, 512, 1024, 128, 128},     // swapab: M <= 64
      {8, 576, 1024, 128, 128},     // swapab: M <= 64, ragged N
      {83, 512, 1024, 128, 128},    // swapab: M > 64 but 83 % 4 != 0
      {200, 512, 1024, 128, 128},   // pingpong: M > 64, 200 % 4 == 0, M <= 256
      {512, 512, 1024, 128, 128},   // default: M > 256
      // a ragged K-block that is still 16-aligned: 1088 = 8*128 + 64.
      {8, 512, 1088, 128, 128},
  };
  return grid;
}

}  // namespace

// ---------------------------------------------------------------------------
// G6 — the instrument's own precondition. RUNS EVERYWHERE, device or not.
// ---------------------------------------------------------------------------
TEST_CASE("G6 every shape this file drives is one the CUDA arm accepts") {
  for (const BlockCase& c : Grid()) {
    CAPTURE(c.m);
    CAPTURE(c.n);
    CAPTURE(c.k);
    CHECK(Fp8BlockScaledRefusalFor(c.n, c.k, c.block_n, c.block_k) ==
          Fp8BlockScaledRefusal::kNone);
    // ... and the activation quant's own divisibility is not assumed: the op
    // tiles K with cdiv, so a ragged K is legal for THIS op even where
    // vt::QuantFp8Group would refuse it.
    CHECK(c.k > 0);
    CHECK(CDiv(c.k, c.block_k) >= 1);
  }
  // All three tile configs are exercised by the grid; a sweep that silently
  // covered one config would prove far less than it looks.
  bool swap = false, ping = false, deflt = false;
  for (const BlockCase& c : Grid()) {
    switch (Fp8BlockScaledConfigFor(c.m)) {
      case Fp8BlockScaledConfig::kSwapAb:
        swap = true;
        break;
      case Fp8BlockScaledConfig::kPingpong:
        ping = true;
        break;
      case Fp8BlockScaledConfig::kDefault:
      case Fp8BlockScaledConfig::kCount:
        deflt = true;
        break;
    }
  }
  CHECK(swap);
  CHECK(ping);
  CHECK(deflt);
}

// ---------------------------------------------------------------------------
// G2 — upstream's CUTLASS case, against the CPU reference arm
// ---------------------------------------------------------------------------
TEST_CASE("G2 the CUDA block-scaled GEMM matches the CPU reference on upstream's case") {
  if (!HasCuda()) {
    MESSAGE("NO CUDA DEVICE: G2 did not run. #1189 M5's on-hardware leg is OWED, not passed.");
    return;
  }
  const BlockCase c{32, 576, 7168, 128, 128};
  const auto a = RandomFp8Bytes(static_cast<size_t>(c.m) * c.k, 1);
  const auto b = RandomFp8Bytes(static_cast<size_t>(c.n) * c.k, 2);
  const auto as = RandomScales(static_cast<size_t>(c.m) * CDiv(c.k, c.block_k), 3);
  const auto bs = RandomScales(static_cast<size_t>(CDiv(c.n, c.block_n)) * CDiv(c.k, c.block_k), 4);

  const std::vector<float> ref = ReferenceBf16(c, a, as, b, bs);
  const std::vector<float> got = DeviceBf16(c, a, as, b, bs);

  CheckNonVacuous(ref);
  CheckNonVacuous(got);
  const double rel = RelDiff(got, ref);
  CAPTURE(rel);
  CHECK(rel < 0.001);  // upstream's own criterion, verbatim
  CheckNoElementIsCatastrophic(got, ref);
}

// ---------------------------------------------------------------------------
// G7 — the M sweep, with the dispatch counter
// ---------------------------------------------------------------------------
TEST_CASE("G7 every tile config matches the CPU reference and reports itself") {
  if (!HasCuda()) {
    MESSAGE("NO CUDA DEVICE: G7 did not run. #1189 M5's on-hardware leg is OWED, not passed.");
    return;
  }
  uint32_t seed = 100;
  for (const BlockCase& c : Grid()) {
    CAPTURE(c.m);
    CAPTURE(c.n);
    CAPTURE(c.k);
    const auto a = RandomFp8Bytes(static_cast<size_t>(c.m) * c.k, seed++);
    const auto b = RandomFp8Bytes(static_cast<size_t>(c.n) * c.k, seed++);
    const auto as = RandomScales(static_cast<size_t>(c.m) * CDiv(c.k, c.block_k), seed++);
    const auto bs =
        RandomScales(static_cast<size_t>(CDiv(c.n, c.block_n)) * CDiv(c.k, c.block_k), seed++);

    const std::vector<float> ref = ReferenceBf16(c, a, as, b, bs);
    const Fp8BlockScaledStats before = Fp8BlockScaledStatsSnapshot();
    const std::vector<float> got = DeviceBf16(c, a, as, b, bs);
    const Fp8BlockScaledStats after = Fp8BlockScaledStatsSnapshot();

    CheckNonVacuous(got);
    const double rel = RelDiff(got, ref);
    CAPTURE(rel);
    CHECK(rel < 0.001);
    CheckNoElementIsCatastrophic(got, ref);

    // EXACTLY ONE dispatch, on the config the heuristic named. A kernel that
    // silently fell through to another config, or a fallback that ran instead
    // of this kernel, is invisible to every value comparison above — the
    // fallback would be numerically BETTER — and visible here.
    CHECK(after.dispatched() - before.dispatched() == 1);
    CHECK(after.refused == before.refused);
    switch (Fp8BlockScaledConfigFor(c.m)) {
      case Fp8BlockScaledConfig::kSwapAb:
        CHECK(after.swap_ab - before.swap_ab == 1);
        break;
      case Fp8BlockScaledConfig::kPingpong:
        CHECK(after.pingpong - before.pingpong == 1);
        break;
      case Fp8BlockScaledConfig::kDefault:
      case Fp8BlockScaledConfig::kCount:
        CHECK(after.deflt - before.deflt == 1);
        break;
    }
  }
}

// ---------------------------------------------------------------------------
// G8 — the f32 sink
// ---------------------------------------------------------------------------
TEST_CASE("G8 an f32 out is the bf16 epilogue value cast up") {
  if (!HasCuda()) {
    MESSAGE("NO CUDA DEVICE: G8 did not run. #1189 M5's on-hardware leg is OWED, not passed.");
    return;
  }
  const BlockCase c{8, 512, 1024, 128, 128};
  const auto a = RandomFp8Bytes(static_cast<size_t>(c.m) * c.k, 11);
  const auto b = RandomFp8Bytes(static_cast<size_t>(c.n) * c.k, 12);
  const auto as = RandomScales(static_cast<size_t>(c.m) * CDiv(c.k, c.block_k), 13);
  const auto bs = RandomScales(static_cast<size_t>(CDiv(c.n, c.block_n)) * CDiv(c.k, c.block_k), 14);

  const std::vector<float> bf16 = DeviceBf16(c, a, as, b, bs);

  Backend& backend = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(backend);
  DeviceTensor da(backend, g.q, DType::kI8, {c.m, c.k}, a.data());
  DeviceTensor das(backend, g.q, DType::kF32, {c.m, CDiv(c.k, c.block_k)}, as.data());
  DeviceTensor db(backend, g.q, DType::kI8, {c.n, c.k}, b.data());
  DeviceTensor dbs(backend, g.q, DType::kF32,
                   {CDiv(c.n, c.block_n), CDiv(c.k, c.block_k)}, bs.data());
  DeviceTensor dout(backend, g.q, DType::kF32, {c.m, c.n});
  vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                           dbs.tensor(), c.block_n, c.block_k);
  std::vector<float> f32(static_cast<size_t>(c.m) * c.n);
  dout.Download(g.q, f32.data());

  CheckNonVacuous(f32);
  // EXACT, elementwise. The collective is instantiated for bf16 only, so an f32
  // out is a widening of the same value and nothing more; anything else would
  // mean a second, wider compute path had appeared.
  size_t mismatched = 0;
  for (size_t i = 0; i < f32.size(); ++i)
    if (f32[i] != bf16[i]) ++mismatched;
  CAPTURE(mismatched);
  CHECK(mismatched == 0);
}

// ---------------------------------------------------------------------------
// G9 — the refusals, on a device
// ---------------------------------------------------------------------------
TEST_CASE("G9 the CUDA arm refuses by name what cutlass cannot implement") {
  if (!HasCuda()) {
    MESSAGE("NO CUDA DEVICE: G9 did not run. #1189 M5's on-hardware leg is OWED, not passed.");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(backend);

  // K = 3884 is upstream's own non-round K and 3884 % 16 == 12, so cutlass
  // cannot take it and upstream reroutes it to triton. The CPU arm of this same
  // op runs it; the message says so.
  {
    const BlockCase c{8, 512, 3884, 128, 128};
    const int64_t kt = CDiv(c.k, c.block_k), nt = CDiv(c.n, c.block_n);
    DeviceTensor da(backend, g.q, DType::kI8, {c.m, c.k});
    DeviceTensor das(backend, g.q, DType::kF32, {c.m, kt});
    DeviceTensor db(backend, g.q, DType::kI8, {c.n, c.k});
    DeviceTensor dbs(backend, g.q, DType::kF32, {nt, kt});
    DeviceTensor dout(backend, g.q, DType::kBF16, {c.m, c.n});
    const Fp8BlockScaledStats before = Fp8BlockScaledStatsSnapshot();
    std::string what;
    try {
      vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                               dbs.tensor(), c.block_n, c.block_k);
      FAIL("a K of 3884 must be refused on the CUDA arm");
    } catch (const std::runtime_error& e) {
      what = e.what();
    }
    CHECK(what.find("3884") != std::string::npos);
    CHECK(what.find("triton") != std::string::npos);
    const Fp8BlockScaledStats after = Fp8BlockScaledStatsSnapshot();
    // A REFUSAL IS NOT A DISPATCH.
    CHECK(after.refused - before.refused == 1);
    CHECK(after.dispatched() == before.dispatched());
  }

  // A block geometry the collective is not instantiated for.
  {
    const BlockCase c{8, 512, 1024, 64, 128};
    const int64_t kt = CDiv(c.k, c.block_k), nt = CDiv(c.n, c.block_n);
    DeviceTensor da(backend, g.q, DType::kI8, {c.m, c.k});
    DeviceTensor das(backend, g.q, DType::kF32, {c.m, kt});
    DeviceTensor db(backend, g.q, DType::kI8, {c.n, c.k});
    DeviceTensor dbs(backend, g.q, DType::kF32, {nt, kt});
    DeviceTensor dout(backend, g.q, DType::kBF16, {c.m, c.n});
    std::string what;
    try {
      vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                               dbs.tensor(), c.block_n, c.block_k);
      FAIL("a block_n of 64 must be refused on the CUDA arm");
    } catch (const std::runtime_error& e) {
      what = e.what();
    }
    CHECK(what.find("block_n is 64") != std::string::npos);
  }
}
