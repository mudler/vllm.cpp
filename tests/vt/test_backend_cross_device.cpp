// Cross-device op-equality harness — the gap
// .agents/specs/backend-fanout-metal-vulkan-xpu.md § Gates calls out explicitly:
// "the seam for a second DeviceType exists but NOTHING exercises CPU-vs-device
// equality". This file is that harness. Newly authored (no upstream vLLM test
// mirrors it: vLLM's device-parameterized kernel tests compare against torch,
// which we do not have).
//
// CONTRACT — read before loosening anything here.
//   * The ORACLE is our own CPU backend, evaluated on the SAME host, from the
//     SAME binary, on the SAME inputs.
//   * The bar for REDUCING / arithmetic ops is NMSE <= 5e-4 — the already-ported
//     llama.cpp threshold (tests/vt/test_ops_quant_dot.cpp, itself ported
//     unwidened from llama.cpp test-quantize-fns:17-28 / test-backend-ops:4277).
//     It is NOT bit-exactness and must not be written as such: the CPU tier's
//     reproducibility comes from a FIXED SEQUENTIAL reduction order
//     (src/vt/cpu/cpu_quant_dot.cpp:22-28, deliberate) and no GPU cross-lane or
//     threadgroup tree reduction preserves it.
//   * The bar for PURE COPY / LAYOUT paths (Backend::Copy, Backend::Memset, a
//     same-dtype cast) IS bit-exactness — nothing is reassociated there, so
//     anything less would be hiding a bug.
//
// The harness runs against EVERY non-CPU backend that is registered in this
// build, so it is one file for Metal, and for CUDA/Vulkan/XPU when they arrive.
// A device that has not registered a given op is SKIPPED rather than failed:
// a partial backend is a supported, tested state (src/vt/ops.cpp:104-111).
#include <doctest/doctest.h>
#include <cstdio>
#include <numeric>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "support/test_env.h"  // SetEnv/UnsetEnv — MSVC has no setenv (#603)
#include "vt/backend.h"
#include "../../src/vt/cpu/cpu_quant_blocks.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/recipes.h"
#include "vt/rocm/rocm_runtime.h"

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// The already-ported bar. See the file header for why this is not memcmp.
constexpr double kNmseTol = 5e-4;

const char* DeviceName(DeviceType t) {
  switch (t) {
    case DeviceType::kCPU: return "CPU";
    case DeviceType::kCUDA: return "CUDA";
    case DeviceType::kMETAL: return "METAL";
    case DeviceType::kVULKAN: return "VULKAN";
    case DeviceType::kXPU: return "XPU";
    case DeviceType::kROCM: return "ROCM";
    case DeviceType::kTENSTORRENT: return "TENSTORRENT";
  }
  return "?";
}

// Normalized mean squared error, the same statistic
// tests/vt/test_ops_quant_dot.cpp gates on: sum((a-b)^2) / sum(a^2).
double Nmse(const std::vector<float>& ref, const std::vector<float>& got) {
  REQUIRE(ref.size() == got.size());
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den == 0.0 ? num : num / den;
}

// Which non-CPU backends does THIS build actually have? GetBackend throws when a
// DeviceType is unregistered, which is the documented probe (no is-registered
// accessor exists on the vt:: seam).
std::vector<DeviceType> RegisteredDevices() {
  std::vector<DeviceType> out;
  for (DeviceType t : {DeviceType::kCUDA, DeviceType::kMETAL, DeviceType::kVULKAN,
                       DeviceType::kXPU, DeviceType::kROCM}) {
    try {
      (void)vt::GetBackend(t);
      out.push_back(t);
    } catch (const std::exception&) {
      // not built / no device present — nothing to compare against
    }
  }
  return out;
}

bool OpAvailable(vt::OpId op, DeviceType t) { return vt::OpRegistered(op, t); }

// A device-resident f32 buffer with host staging, so one body serves a unified
// backend (Metal, GB10) and a discrete one identically: every transfer goes
// through Backend::Copy rather than assuming the host can dereference the
// pointer.
class DevBuf {
 public:
  DevBuf(vt::Backend& b, Queue& q, size_t n) : b_(b), q_(q), n_(n) {
    ptr_ = b_.Alloc(n * sizeof(float));
  }
  ~DevBuf() { b_.Free(ptr_); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;

  void Upload(const std::vector<float>& src) {
    REQUIRE(src.size() == n_);
    b_.Copy(q_, ptr_, src.data(), n_ * sizeof(float));
  }
  std::vector<float> Download() {
    std::vector<float> out(n_);
    b_.Synchronize(q_);
    b_.Copy(q_, out.data(), ptr_, n_ * sizeof(float));
    b_.Synchronize(q_);
    return out;
  }
  void* ptr() const { return ptr_; }

 private:
  vt::Backend& b_;
  Queue& q_;
  size_t n_;
  void* ptr_ = nullptr;
};

std::vector<float> RandomVec(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

Tensor T2(void* p, Device d, int64_t r, int64_t c) {
  return Tensor::Contiguous(p, DType::kF32, d, {r, c});
}
Tensor T1(void* p, Device d, int64_t n) {
  return Tensor::Contiguous(p, DType::kF32, d, {n});
}
// Integer operands: embedding ids (i32 or i64, both accepted by vt::Embedding)
// and sampler token ids (i64 by contract).
Tensor TI32(void* p, Device d, int64_t n) {
  return Tensor::Contiguous(p, DType::kI32, d, {n});
}
Tensor TI64(void* p, Device d, int64_t n) {
  return Tensor::Contiguous(p, DType::kI64, d, {n});
}

}  // namespace

// ---------------------------------------------------------------------------
// Bit-exact tier: the pure byte paths. No arithmetic, so no tolerance.
// ---------------------------------------------------------------------------
TEST_CASE("device Copy/Memset are BIT-EXACT against the host bytes") {
  for (DeviceType dt : RegisteredDevices()) {
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();

    constexpr size_t kN = 977;  // deliberately not a round number
    std::vector<uint8_t> src(kN);
    for (size_t i = 0; i < kN; ++i) src[i] = static_cast<uint8_t>((i * 31u + 7u) & 0xFFu);

    void* p = dev.Alloc(kN);
    dev.Copy(q, p, src.data(), kN);
    dev.Synchronize(q);
    std::vector<uint8_t> back(kN, 0);
    dev.Copy(q, back.data(), p, kN);
    dev.Synchronize(q);
    CHECK(std::memcmp(src.data(), back.data(), kN) == 0);

    dev.Memset(q, p, 0x5A, kN);
    dev.Synchronize(q);
    dev.Copy(q, back.data(), p, kN);
    dev.Synchronize(q);
    std::vector<uint8_t> expect(kN, 0x5A);
    CHECK(std::memcmp(expect.data(), back.data(), kN) == 0);

    dev.Free(p);
    dev.DestroyQueue(q);
  }
}

// The bf16<->f32 casts are a pure ELEMENTWISE CODEC: no reduction, no
// reassociation, one rounding on store. So the bar here is BIT-EXACTNESS against
// the CPU reference, not NMSE — CastF32 (bf16 -> f32) is an exact widening, and
// CastBf16 (f32 -> bf16) must reproduce vt::F32ToBF16's round-to-nearest-EVEN
// (src/vt/dtype.cpp:224-233) exactly. A device that got the rounding "nearly
// right" would sail through an NMSE gate and still corrupt weights, so the
// rounding contract is checked with memcmp over every finite value, +-0, +-inf
// and 16 EXACT halfway ties.
//
// ONE DOCUMENTED CARVE-OUT: the NaN PAYLOAD. Measured on GB10 2026-07-22 with
// this very harness — for input 0x7FC00000 our CPU codec yields bf16 0x7FC0
// (`(u >> 16) | 0x0040`, i.e. truncate-and-quiet) while CUDA's
// `__float2bfloat16` yields 0x7FFF (canonical all-ones payload). Both are valid
// QUIET NaNs and IEEE-754 does not specify payload propagation across a
// narrowing conversion, so this is an architectural representation difference,
// NOT a rounding defect. It is carved out EXPLICITLY and narrowly: the payload
// bits are excluded, the quiet-NaN-ness is still asserted, and nothing about the
// rounding gate is weakened. (Metal, whose MSL codec is a literal transcription
// of vt::F32ToBF16 including its NaN branch, IS bit-exact here too — only CUDA
// differs, which is itself worth knowing.)
// A NORM WEIGHT MAY BE A DIFFERENT DTYPE FROM THE ACTIVATION, and the CPU kernel
// has always allowed it: `RmsNormKernel` widens the weight through
// `WidenRowToF32` and accepts f32/f16/bf16 for x, w, residual and out
// INDEPENDENTLY. Every device must match that, because the CPU backend is this
// project's correctness reference.
//
// IT DID NOT. `RmsNormKernelCuda` demanded `w.dtype == x.dtype` and refused
// otherwise, so a model that ran on the CPU backend died on CUDA by name --
// `vt: cuda rmsnorm: weight dtype must match x`. A gamma loaded bf16 by
// `LoadNormBf16` beside an f32 activation is exactly that shape, and qwen4_exp
// hits it. The existing cross-device rmsnorm case above cannot see this: it
// passes an f32 weight with an f32 x, so the dtypes always agreed and the
// narrower contract was invisible.
//
// A kernel narrower than its own oracle is the defect. This case pins the
// contract rather than the kernel: it asks every registered device for the same
// answer the CPU gives.
TEST_CASE("rmsnorm accepts a bf16 weight beside an f32 activation, on every device") {
  constexpr int64_t rows = 3, cols = 128;
  const size_t n = static_cast<size_t>(rows) * cols;
  const std::vector<float> x = RandomVec(n, 4242);
  const std::vector<float> w_f32 = RandomVec(cols, 2424, -1.0f, 1.0f);

  // The weight as it actually arrives from a GGUF norm: bf16 bits, and the f32
  // values ROUNDED to bf16, so the CPU reference is computed from the same
  // numbers the device sees rather than from the unrounded originals.
  std::vector<uint16_t> w_bf16(cols);
  std::vector<float> w_rounded(cols);
  for (int64_t j = 0; j < cols; ++j) {
    w_bf16[static_cast<size_t>(j)] = vt::F32ToBF16(w_f32[static_cast<size_t>(j)]);
    w_rounded[static_cast<size_t>(j)] =
        vt::BF16ToF32(w_bf16[static_cast<size_t>(j)]);
  }

  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> cx = x, ref(n);
  {
    Tensor tx = T2(cx.data(), cd, rows, cols);
    Tensor tw = Tensor::Contiguous(w_bf16.data(), DType::kBF16, cd, {cols});
    Tensor to = T2(ref.data(), cd, rows, cols);
    vt::RmsNorm(cq, to, tx, tw, vt::RmsNormArgs{1e-6f, false}, nullptr);
  }
  cpu.DestroyQueue(cq);

  // THE ORACLE HALF, AND IT RUNS EVERYWHERE. The device loop below is empty on a
  // CPU-only build, so without these the case would report zero assertions and
  // pass vacuously -- a skip wearing the face of a green test. These pin the
  // contract the CUDA kernel violated: the CPU DOES accept a bf16 weight beside
  // an f32 activation, and produces real numbers from it.
  REQUIRE(ref.size() == n);
  CHECK(std::isfinite(ref[0]));
  CHECK(std::isfinite(ref[n - 1]));
  const double sumsq = std::inner_product(ref.begin(), ref.end(), ref.begin(), 0.0);
  CHECK(sumsq > 0.0);  // not an all-zero row, which any broken widening would give

  // THE CASE ASSERTS ITS OWN PRECONDITION. On a CPU-only build the loop below is
  // empty and this case legitimately proves only the oracle half; on a build with
  // a device it MUST exercise every device that offers rmsnorm. Counting is what
  // makes "it ran" evidence instead of inference -- an external assertion-count
  // threshold is a guess about doctest's arithmetic, and mine was wrong.
  int eligible = 0, exercised = 0;
  for (DeviceType dt : RegisteredDevices())
    if (dt != DeviceType::kCPU && OpAvailable(vt::OpId::kRmsNorm, dt)) ++eligible;

  for (DeviceType dt : RegisteredDevices()) {
    if (dt == DeviceType::kCPU) continue;
    CAPTURE(DeviceName(dt));
    if (!OpAvailable(vt::OpId::kRmsNorm, dt)) continue;
    ++exercised;
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device dd = q.device;

    DevBuf dx(dev, q, n);
    dx.Upload(x);
    DevBuf dout(dev, q, n);
    // The weight is bf16, so it cannot ride `DevBuf` (f32-sized). Raw, and freed
    // on every exit path below.
    void* dw = dev.Alloc(static_cast<size_t>(cols) * sizeof(uint16_t));
    dev.Copy(q, dw, w_bf16.data(), static_cast<size_t>(cols) * sizeof(uint16_t));

    Tensor tx = T2(dx.ptr(), dd, rows, cols);
    Tensor tw = Tensor::Contiguous(dw, DType::kBF16, dd, {cols});
    Tensor to = T2(dout.ptr(), dd, rows, cols);
    vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{1e-6f, false}, nullptr);
    dev.Synchronize(q);
    const std::vector<float> got = dout.Download();
    dev.Free(dw);
    CHECK(Nmse(ref, got) <= kNmseTol);
    dev.DestroyQueue(q);
  }

  // Every eligible device was actually visited. Without this, a filter change, a
  // missing registration or an early `continue` would leave the device half
  // silently unrun and the case would still report PASS.
  CHECK(exercised == eligible);
  std::fprintf(stderr,
               "[rmsnorm widened-weight] devices eligible=%d exercised=%d\n",
               eligible, exercised);
}

TEST_CASE("bf16<->f32 casts are BIT-EXACT against the CPU codec") {
  constexpr int64_t kRows = 8, kCols = 64;
  constexpr size_t kN = kRows * kCols;
  // Deliberately includes values that land ON a bf16 rounding tie, plus a NaN
  // and the infinities, so the tie-break and the NaN path are actually covered.
  std::vector<float> src = RandomVec(kN, 11, -8.0f, 8.0f);
  constexpr size_t kNanIdx = 0;  // the single payload carve-out; see the header
  src[kNanIdx] = std::numeric_limits<float>::quiet_NaN();
  src[1] = std::numeric_limits<float>::infinity();
  src[2] = -std::numeric_limits<float>::infinity();
  src[3] = 0.0f;
  src[4] = -0.0f;
  for (size_t i = 5; i < 21; ++i) {
    // Exact halfway cases for the bf16 mantissa: low 16 bits == 0x8000.
    uint32_t bits = (0x3F800000u + (static_cast<uint32_t>(i - 5) << 16)) | 0x8000u;
    std::memcpy(&src[i], &bits, sizeof(bits));
  }

  // CPU oracle: f32 -> bf16 -> f32.
  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> cs = src;
  std::vector<uint16_t> ref_bf(kN);
  std::vector<float> ref_f32(kN);
  {
    Tensor tin = T2(cs.data(), cd, kRows, kCols);
    Tensor tbf = Tensor::Contiguous(ref_bf.data(), DType::kBF16, cd, {kRows, kCols});
    Tensor tf32 = T2(ref_f32.data(), cd, kRows, kCols);
    vt::CastBf16(cq, tbf, tin);
    vt::CastF32(cq, tf32, tbf);
  }
  cpu.DestroyQueue(cq);

  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kCastBf16, dt) || !OpAvailable(vt::OpId::kCastF32, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    void* pin = dev.Alloc(kN * sizeof(float));
    void* pbf = dev.Alloc(kN * sizeof(uint16_t));
    void* pf32 = dev.Alloc(kN * sizeof(float));
    dev.Copy(q, pin, src.data(), kN * sizeof(float));
    Tensor tin = T2(pin, d, kRows, kCols);
    Tensor tbf = Tensor::Contiguous(pbf, DType::kBF16, d, {kRows, kCols});
    Tensor tf32 = T2(pf32, d, kRows, kCols);
    vt::CastBf16(q, tbf, tin);
    vt::CastF32(q, tf32, tbf);
    dev.Synchronize(q);

    std::vector<uint16_t> got_bf(kN);
    std::vector<float> got_f32(kN);
    dev.Copy(q, got_bf.data(), pbf, kN * sizeof(uint16_t));
    dev.Copy(q, got_f32.data(), pf32, kN * sizeof(float));
    dev.Synchronize(q);

    // Bit-exact everywhere EXCEPT the NaN payload slot. Compared as two
    // memcmp'd spans rather than a loop so a single differing bit anywhere in
    // the rounding-relevant data still fails hard.
    CHECK(std::memcmp(ref_bf.data(), got_bf.data(), kNanIdx * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(ref_bf.data() + kNanIdx + 1, got_bf.data() + kNanIdx + 1,
                      (kN - kNanIdx - 1) * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(ref_f32.data(), got_f32.data(), kNanIdx * sizeof(float)) == 0);
    CHECK(std::memcmp(ref_f32.data() + kNanIdx + 1, got_f32.data() + kNanIdx + 1,
                      (kN - kNanIdx - 1) * sizeof(float)) == 0);
    // The carve-out is on the PAYLOAD only: the value must still be a QUIET NaN
    // (bf16 exponent all ones + mantissa MSB set), and must still widen to a NaN.
    const uint16_t nan_bf = got_bf[kNanIdx];
    CHECK((nan_bf & 0x7F80u) == 0x7F80u);  // exponent all ones
    CHECK((nan_bf & 0x007Fu) != 0u);       // non-zero payload => NaN, not inf
    CHECK((nan_bf & 0x0040u) == 0x0040u);  // mantissa MSB set => QUIET
    CHECK(std::isnan(got_f32[kNanIdx]));

    dev.Free(pin);
    dev.Free(pbf);
    dev.Free(pf32);
    dev.DestroyQueue(q);
  }
}

// ---------------------------------------------------------------------------
// NMSE tier: everything with arithmetic. CPU is the oracle.
// ---------------------------------------------------------------------------
TEST_CASE("elementwise ops match the CPU oracle within NMSE <= 5e-4") {
  constexpr int64_t kRows = 17;
  constexpr int64_t kCols = 128;
  constexpr size_t kN = kRows * kCols;

  const std::vector<float> a = RandomVec(kN, 101);
  const std::vector<float> b = RandomVec(kN, 202);
  const std::vector<float> bias = RandomVec(kCols, 303);

  // --- CPU oracle, computed once through the very same vt:: entry points.
  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ca = a, cb = b, cbias = bias;
  std::vector<float> ref_add(kN), ref_bias(kN), ref_relu(kN), ref_silu(kRows * kCols / 2);
  {
    Tensor ta = T2(ca.data(), cd, kRows, kCols);
    Tensor tb = T2(cb.data(), cd, kRows, kCols);
    Tensor tbias = T1(cbias.data(), cd, kCols);
    Tensor tadd = T2(ref_add.data(), cd, kRows, kCols);
    Tensor tbcast = T2(ref_bias.data(), cd, kRows, kCols);
    Tensor trelu = T2(ref_relu.data(), cd, kRows, kCols);
    Tensor tsilu = T2(ref_silu.data(), cd, kRows, kCols / 2);
    vt::Add(cq, tadd, ta, tb);
    vt::Add(cq, tbcast, ta, tbias);
    vt::Relu(cq, trelu, ta);
    vt::SiluAndMul(cq, tsilu, ta);
  }

  for (DeviceType dt : RegisteredDevices()) {
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    DevBuf da(dev, q, kN), db(dev, q, kN), dbias(dev, q, kCols), dout(dev, q, kN);
    da.Upload(a);
    db.Upload(b);
    dbias.Upload(bias);
    Tensor ta = T2(da.ptr(), d, kRows, kCols);
    Tensor tb = T2(db.ptr(), d, kRows, kCols);
    Tensor tbias = T1(dbias.ptr(), d, kCols);

    if (OpAvailable(vt::OpId::kAdd, dt)) {
      Tensor to = T2(dout.ptr(), d, kRows, kCols);
      vt::Add(q, to, ta, tb);
      CHECK(Nmse(ref_add, dout.Download()) <= kNmseTol);
      // The rank-1 nn.Linear bias broadcast is a DIFFERENT indexing path.
      vt::Add(q, to, ta, tbias);
      CHECK(Nmse(ref_bias, dout.Download()) <= kNmseTol);
    }
    if (OpAvailable(vt::OpId::kRelu, dt)) {
      Tensor to = T2(dout.ptr(), d, kRows, kCols);
      vt::Relu(q, to, ta);
      CHECK(Nmse(ref_relu, dout.Download()) <= kNmseTol);
    }
    if (OpAvailable(vt::OpId::kSiluAndMul, dt)) {
      DevBuf dsilu(dev, q, kRows * kCols / 2);
      Tensor to = T2(dsilu.ptr(), d, kRows, kCols / 2);
      vt::SiluAndMul(q, to, ta);
      CHECK(Nmse(ref_silu, dsilu.Download()) <= kNmseTol);
    }
    dev.DestroyQueue(q);
  }
  cpu.DestroyQueue(cq);
}

TEST_CASE("RopeFromCache matches the CPU oracle within NMSE <= 5e-4, both styles") {
  // The APPLY half of vLLM's rotary split: the cos/sin table is built once (on
  // the portable tier, in double) and this rotates q and k with it.
  constexpr int64_t kTokens = 11, kHq = 4, kHk = 2, kD = 16, kRot = 16;
  constexpr int64_t kMaxPos = 64;

  const std::vector<float> q0 = RandomVec(kTokens * kHq * kD, 801);
  const std::vector<float> k0 = RandomVec(kTokens * kHk * kD, 802);
  const std::vector<float> cache = RandomVec(kMaxPos * kRot, 803, -1.0f, 1.0f);
  // Positions are NOT 0..n-1: a kernel that used the token index instead of the
  // position would pass on the identity mapping and fail here.
  std::vector<int32_t> pos(kTokens);
  for (int64_t i = 0; i < kTokens; ++i) pos[static_cast<size_t>(i)] = int32_t((i * 7 + 3) % kMaxPos);

  // NeoX rotates (pair, pair+half); GPT-J style rotates (2*pair, 2*pair+1). They
  // are different element pairings, so a kernel that hardcoded one passes half
  // the models and silently corrupts the other half.
  for (bool neox : {true, false}) {
    CAPTURE(neox);
    vt::RopeArgs args;
    args.rotary_dim = kRot;
    args.is_neox_style = neox;

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> refq = q0, refk = k0, ccache = cache;
    std::vector<int32_t> cpos = pos;
    {
      Tensor tq = Tensor::Contiguous(refq.data(), DType::kF32, cd, {kTokens, kHq, kD});
      Tensor tk = Tensor::Contiguous(refk.data(), DType::kF32, cd, {kTokens, kHk, kD});
      Tensor tc = Tensor::Contiguous(ccache.data(), DType::kF32, cd, {kMaxPos, kRot});
      Tensor tp = Tensor::Contiguous(cpos.data(), DType::kI32, cd, {kTokens});
      vt::RopeFromCache(cq, tq, &tk, tp, tc, args);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kRopeFromCache, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dq(dev, q, kTokens * kHq * kD), dk(dev, q, kTokens * kHk * kD),
          dc(dev, q, kMaxPos * kRot);
      dq.Upload(q0);   // rotation is IN PLACE, so re-upload the pristine input
      dk.Upload(k0);
      dc.Upload(cache);
      void* dpos = dev.Alloc(kTokens * sizeof(int32_t));
      dev.Copy(q, dpos, pos.data(), kTokens * sizeof(int32_t));
      dev.Synchronize(q);

      Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {kTokens, kHq, kD});
      Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {kTokens, kHk, kD});
      Tensor tc = Tensor::Contiguous(dc.ptr(), DType::kF32, d, {kMaxPos, kRot});
      Tensor tp = Tensor::Contiguous(dpos, DType::kI32, d, {kTokens});
      vt::RopeFromCache(q, tq, &tk, tp, tc, args);
      dev.Synchronize(q);

      CHECK(Nmse(refq, dq.Download()) <= kNmseTol);
      CHECK(Nmse(refk, dk.Download()) <= kNmseTol);

      dev.Free(dpos);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("ReshapeAndCache scatters into the KV cache BIT-EXACTLY") {
  // Byte movement, so the bar is memcmp against the CPU oracle, not NMSE.
  constexpr int64_t kTokens = 9, kHk = 2, kD = 8, kBS = 4, kBlocks = 6;
  constexpr int64_t kElems = kHk * kD;          // one token's page payload
  constexpr int64_t kCacheN = kBlocks * kBS * kHk * kD;

  const std::vector<float> knew = RandomVec(kTokens * kElems, 701);
  const std::vector<float> vnew = RandomVec(kTokens * kElems, 702);
  // Pre-existing cache contents: the padded-token case must leave these INTACT,
  // so they cannot start as zeros or the check would pass vacuously.
  const std::vector<float> kc0 = RandomVec(kCacheN, 703);
  const std::vector<float> vc0 = RandomVec(kCacheN, 704);

  // Slots are deliberately SCATTERED and out of order, and two tokens carry -1.
  // Upstream pads the mapping and marks padded tokens negative (cpu_cache.cpp:60);
  // a kernel that clamped instead of skipping would corrupt a real page, and one
  // that read the i64 slot as unsigned would index astronomically out of range.
  const std::vector<int64_t> slots = {20, -1, 3, 11, -1, 0, 23, 7, 15};

  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ck = knew, cv = vnew, ref_kc = kc0, ref_vc = vc0;
  std::vector<int64_t> cslots = slots;
  {
    Tensor tk = Tensor::Contiguous(ck.data(), DType::kF32, cd, {kTokens, kHk, kD});
    Tensor tv = Tensor::Contiguous(cv.data(), DType::kF32, cd, {kTokens, kHk, kD});
    Tensor tkc = Tensor::Contiguous(ref_kc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
    Tensor tvc = Tensor::Contiguous(ref_vc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
    Tensor tsm = Tensor::Contiguous(cslots.data(), DType::kI64, cd, {kTokens});
    vt::ReshapeAndCache(cq, tk, tv, tkc, tvc, tsm);
  }
  cpu.DestroyQueue(cq);

  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kReshapeAndCache, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    DevBuf dk(dev, q, kTokens * kElems), dv(dev, q, kTokens * kElems),
        dkc(dev, q, kCacheN), dvc(dev, q, kCacheN);
    dk.Upload(knew);
    dv.Upload(vnew);
    dkc.Upload(kc0);   // seeded, so an untouched page must survive
    dvc.Upload(vc0);
    void* dsm = dev.Alloc(kTokens * sizeof(int64_t));
    dev.Copy(q, dsm, slots.data(), kTokens * sizeof(int64_t));
    dev.Synchronize(q);

    Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {kTokens, kHk, kD});
    Tensor tv = Tensor::Contiguous(dv.ptr(), DType::kF32, d, {kTokens, kHk, kD});
    Tensor tkc = Tensor::Contiguous(dkc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
    Tensor tvc = Tensor::Contiguous(dvc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
    Tensor tsm = Tensor::Contiguous(dsm, DType::kI64, d, {kTokens});
    vt::ReshapeAndCache(q, tk, tv, tkc, tvc, tsm);
    dev.Synchronize(q);

    const std::vector<float> got_kc = dkc.Download();
    const std::vector<float> got_vc = dvc.Download();
    CHECK(std::memcmp(ref_kc.data(), got_kc.data(), ref_kc.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(ref_vc.data(), got_vc.data(), ref_vc.size() * sizeof(float)) == 0);

    dev.Free(dsm);
    dev.DestroyQueue(q);
  }

  // --- Unbind flash layout: single (blocks,2,bs,H,D) allocation, K/V strided ---
  // Matches dense_attn::KvSlice — the layout the engine really feeds.
  {
    const int64_t within = kBS * kHk * kD;
    std::vector<float> combined(static_cast<size_t>(kBlocks * 2 * within));
    for (int64_t b = 0; b < kBlocks; ++b)
      for (int64_t e = 0; e < within; ++e) {
        combined[static_cast<size_t>((b * 2 + 0) * within + e)] =
            kc0[static_cast<size_t>(b * within + e)];
        combined[static_cast<size_t>((b * 2 + 1) * within + e)] =
            vc0[static_cast<size_t>(b * within + e)];
      }
    std::vector<float> ref_comb = combined;
    {
      vt::Backend& fcpu = vt::GetBackend(DeviceType::kCPU);
      Queue fq = fcpu.CreateQueue();
      const Device fd{DeviceType::kCPU, 0};
      std::vector<float> fk = knew, fv = vnew, fslots_f;
      std::vector<int64_t> fslots = slots;
      Tensor tk = Tensor::Contiguous(fk.data(), DType::kF32, fd, {kTokens, kHk, kD});
      Tensor tv = Tensor::Contiguous(fv.data(), DType::kF32, fd, {kTokens, kHk, kD});
      Tensor tcomb =
          Tensor::Contiguous(ref_comb.data(), DType::kF32, fd, {kBlocks * 2 * within});
      auto slice = [&](int which) {
        Tensor t = tcomb;
        t.data = static_cast<char*>(t.data) +
                 static_cast<size_t>(which) * static_cast<size_t>(within) * sizeof(float);
        t.rank = 4;
        t.shape[0] = kBlocks;
        t.shape[1] = kBS;
        t.shape[2] = kHk;
        t.shape[3] = kD;
        t.stride[0] = 2 * within;
        t.stride[1] = kHk * kD;
        t.stride[2] = kD;
        t.stride[3] = 1;
        return t;
      };
      Tensor tsm = Tensor::Contiguous(fslots.data(), DType::kI64, fd, {kTokens});
      Tensor tkc = slice(0), tvc = slice(1);
      vt::ReshapeAndCache(fq, tk, tv, tkc, tvc, tsm);
      fcpu.DestroyQueue(fq);
    }

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kReshapeAndCache, dt)) continue;
      CAPTURE(DeviceName(dt));
      CAPTURE("unbind");
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf dk(dev, q, kTokens * kElems), dv(dev, q, kTokens * kElems),
          dcomb(dev, q, kBlocks * 2 * within);
      dk.Upload(knew);
      dv.Upload(vnew);
      dcomb.Upload(combined);
      void* dsm = dev.Alloc(kTokens * sizeof(int64_t));
      dev.Copy(q, dsm, slots.data(), kTokens * sizeof(int64_t));
      dev.Synchronize(q);
      Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {kTokens, kHk, kD});
      Tensor tv = Tensor::Contiguous(dv.ptr(), DType::kF32, d, {kTokens, kHk, kD});
      Tensor tcomb =
          Tensor::Contiguous(dcomb.ptr(), DType::kF32, d, {kBlocks * 2 * within});
      auto slice = [&](int which) {
        Tensor t = tcomb;
        t.data = static_cast<char*>(t.data) +
                 static_cast<size_t>(which) * static_cast<size_t>(within) * sizeof(float);
        t.rank = 4;
        t.shape[0] = kBlocks;
        t.shape[1] = kBS;
        t.shape[2] = kHk;
        t.shape[3] = kD;
        t.stride[0] = 2 * within;
        t.stride[1] = kHk * kD;
        t.stride[2] = kD;
        t.stride[3] = 1;
        return t;
      };
      Tensor tsm = Tensor::Contiguous(dsm, DType::kI64, d, {kTokens});
      Tensor tkc = slice(0), tvc = slice(1);
      vt::ReshapeAndCache(q, tk, tv, tkc, tvc, tsm);
      dev.Synchronize(q);
      const std::vector<float> got = dcomb.Download();
      CHECK(std::memcmp(ref_comb.data(), got.data(), ref_comb.size() * sizeof(float)) == 0);
      dev.Free(dsm);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("paged attention matches the CPU oracle within NMSE <= 5e-4") {
  // GQA prefill: Hq=4 query heads over Hk=2 kv heads, head dim 8, block size 4,
  // one request of 37 tokens spanning 10 pages. Ragged on purpose -- 37 is not a
  // multiple of the block size, so the last page is partly occupied and a kernel
  // that walked whole pages would read past the sequence.
  constexpr int64_t kT = 37, kHq = 4, kHk = 2, kD = 8, kBS = 4;
  constexpr int64_t kBlocks = (kT + kBS - 1) / kBS;  // 10

  const std::vector<float> query = RandomVec(kT * kHq * kD, 601);
  const std::vector<float> kc = RandomVec(kBlocks * kBS * kHk * kD, 602);
  const std::vector<float> vc = RandomVec(kBlocks * kBS * kHk * kD, 603);
  std::vector<int32_t> block_table(kBlocks);
  // A NON-IDENTITY mapping, so a kernel that ignored the block table and indexed
  // the cache linearly would fail. Page j lives at cache block (kBlocks-1-j).
  for (int64_t b = 0; b < kBlocks; ++b) {
    block_table[static_cast<size_t>(b)] = static_cast<int32_t>(kBlocks - 1 - b);
  }
  const std::vector<int32_t> seq_lens = {static_cast<int32_t>(kT)};
  const std::vector<int32_t> qsl = {0, static_cast<int32_t>(kT)};

  // Three configurations, because they are different branches in the kernel and
  // a single causal case would leave two of them unexercised.
  struct Cfg { const char* name; bool causal; bool window; float softcap; };
  const Cfg cfgs[] = {
      {"causal", true, false, 0.0f},
      {"causal+softcap", true, false, 30.0f},   // cap * tanh(s / cap)
      {"sliding-window", true, true, 0.0f},     // window_left bounds jmin
  };

  for (const Cfg& cfg : cfgs) {
    CAPTURE(cfg.name);
    vt::PagedAttentionArgs args;
    args.scale = 0.353553f;
    args.causal = cfg.causal;
    args.logits_soft_cap = cfg.softcap;
    // Both bounds must be >= 0 (ops.cpp:2778). right = 0 is the causal
    // sliding window: no future keys, at most 8 past ones.
    if (cfg.window) args.window_size = vt::AttentionWindow{8, 0};

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cq_v = query, ckc = kc, cvc = vc, ref(kT * kHq * kD);
    std::vector<int32_t> cbt = block_table, csl = seq_lens, cqsl = qsl;
    {
      Tensor tq = Tensor::Contiguous(cq_v.data(), DType::kF32, cd, {kT, kHq, kD});
      Tensor tkc = Tensor::Contiguous(ckc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
      Tensor tvc = Tensor::Contiguous(cvc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
      Tensor tbt = Tensor::Contiguous(cbt.data(), DType::kI32, cd, {1, kBlocks});
      Tensor tsl = Tensor::Contiguous(csl.data(), DType::kI32, cd, {1});
      Tensor tqsl = Tensor::Contiguous(cqsl.data(), DType::kI32, cd, {2});
      Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kT, kHq, kD});
      vt::PagedAttention(cq, to, tq, tkc, tvc, tbt, tsl, tqsl, args);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kPagedAttention, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dq(dev, q, kT * kHq * kD), dkc(dev, q, kBlocks * kBS * kHk * kD),
          dvc(dev, q, kBlocks * kBS * kHk * kD), dout(dev, q, kT * kHq * kD);
      dq.Upload(query);
      dkc.Upload(kc);
      dvc.Upload(vc);
      void* dbt = dev.Alloc(kBlocks * sizeof(int32_t));
      void* dsl = dev.Alloc(sizeof(int32_t));
      void* dqsl = dev.Alloc(2 * sizeof(int32_t));
      dev.Copy(q, dbt, block_table.data(), kBlocks * sizeof(int32_t));
      dev.Copy(q, dsl, seq_lens.data(), sizeof(int32_t));
      dev.Copy(q, dqsl, qsl.data(), 2 * sizeof(int32_t));
      dev.Synchronize(q);

      Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {kT, kHq, kD});
      Tensor tkc = Tensor::Contiguous(dkc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
      Tensor tvc = Tensor::Contiguous(dvc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
      Tensor tbt = Tensor::Contiguous(dbt, DType::kI32, d, {1, kBlocks});
      Tensor tsl = Tensor::Contiguous(dsl, DType::kI32, d, {1});
      Tensor tqsl = Tensor::Contiguous(dqsl, DType::kI32, d, {2});
      Tensor to = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {kT, kHq, kD});
      vt::PagedAttention(q, to, tq, tkc, tvc, tbt, tsl, tqsl, args);
      dev.Synchronize(q);

      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);

      dev.Free(dbt);
      dev.Free(dsl);
      dev.Free(dqsl);
      dev.DestroyQueue(q);
    }
  }

  // --- DECODE shape: one new query token over a filled cache (Tq=1, seq=kT).
  // This is the path multi-token generation hits after prefill; a prefill-only
  // test leaves it unexercised.
  {
    constexpr int64_t kTq = 1;
    const std::vector<float> q_dec = RandomVec(kTq * kHq * kD, 701);
    const std::vector<int32_t> sl_dec = {static_cast<int32_t>(kT)};
    const std::vector<int32_t> qsl_dec = {0, 1};
    vt::PagedAttentionArgs dargs;
    dargs.scale = 0.353553f;
    dargs.causal = true;

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cq_v = q_dec, ckc = kc, cvc = vc, ref(kTq * kHq * kD);
    std::vector<int32_t> cbt = block_table, csl = sl_dec, cqsl = qsl_dec;
    {
      Tensor tq = Tensor::Contiguous(cq_v.data(), DType::kF32, cd, {kTq, kHq, kD});
      Tensor tkc = Tensor::Contiguous(ckc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
      Tensor tvc = Tensor::Contiguous(cvc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
      Tensor tbt = Tensor::Contiguous(cbt.data(), DType::kI32, cd, {1, kBlocks});
      Tensor tsl = Tensor::Contiguous(csl.data(), DType::kI32, cd, {1});
      Tensor tqsl = Tensor::Contiguous(cqsl.data(), DType::kI32, cd, {2});
      Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kTq, kHq, kD});
      vt::PagedAttention(cq, to, tq, tkc, tvc, tbt, tsl, tqsl, dargs);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kPagedAttention, dt)) continue;
      CAPTURE(DeviceName(dt));
      CAPTURE("decode");
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf dq(dev, q, kTq * kHq * kD), dkc(dev, q, kBlocks * kBS * kHk * kD),
          dvc(dev, q, kBlocks * kBS * kHk * kD), dout(dev, q, kTq * kHq * kD);
      dq.Upload(q_dec);
      dkc.Upload(kc);
      dvc.Upload(vc);
      void* dbt = dev.Alloc(kBlocks * sizeof(int32_t));
      void* dsl = dev.Alloc(sizeof(int32_t));
      void* dqsl = dev.Alloc(2 * sizeof(int32_t));
      dev.Copy(q, dbt, block_table.data(), kBlocks * sizeof(int32_t));
      dev.Copy(q, dsl, sl_dec.data(), sizeof(int32_t));
      dev.Copy(q, dqsl, qsl_dec.data(), 2 * sizeof(int32_t));
      dev.Synchronize(q);
      Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {kTq, kHq, kD});
      Tensor tkc = Tensor::Contiguous(dkc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
      Tensor tvc = Tensor::Contiguous(dvc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
      Tensor tbt = Tensor::Contiguous(dbt, DType::kI32, d, {1, kBlocks});
      Tensor tsl = Tensor::Contiguous(dsl, DType::kI32, d, {1});
      Tensor tqsl = Tensor::Contiguous(dqsl, DType::kI32, d, {2});
      Tensor to = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {kTq, kHq, kD});
      vt::PagedAttention(q, to, tq, tkc, tvc, tbt, tsl, tqsl, dargs);
      dev.Synchronize(q);
      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);
      dev.Free(dbt);
      dev.Free(dsl);
      dev.Free(dqsl);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("Embedding gather and greedy argmax match the CPU oracle EXACTLY") {
  // Neither op is arithmetic, so neither gets the NMSE tier: a gather must move
  // the exact bytes, and an argmax must pick the exact index.
  constexpr int64_t kVocab = 61;
  constexpr int64_t kHidden = 40;
  constexpr int64_t kTokens = 7;

  const std::vector<float> table = RandomVec(kVocab * kHidden, 501);
  // Ids chosen to include 0 and the last row, and to REPEAT — a gather that
  // accidentally consumed ids positionally would pass on distinct ids.
  const std::vector<int32_t> ids32 = {0, 60, 13, 13, 1, 59, 0};
  std::vector<int64_t> ids64(ids32.begin(), ids32.end());

  // Logits with a DELIBERATE TIE: row 0 has its maximum twice, at columns 2 and
  // 5. The contract (cpu_sample.cpp:49, strict `>`) is that the FIRST wins, so a
  // kernel that used `>=` or a tie-indifferent tree reduction returns 5 and fails
  // here. That is a different token, not a rounding difference.
  constexpr int64_t kRows = 3;
  std::vector<float> logits(kRows * kVocab, 0.0f);
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t c = 0; c < kVocab; ++c) logits[r * kVocab + c] = -1.0f * float(c + 1);
  }
  logits[0 * kVocab + 2] = 9.0f;
  logits[0 * kVocab + 5] = 9.0f;   // tie with column 2; column 2 must win
  logits[1 * kVocab + 60] = 5.0f;  // last column
  logits[2 * kVocab + 0] = 5.0f;   // first column

  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ctable = table, clogits = logits;
  std::vector<int32_t> cids = ids32;
  std::vector<float> ref_emb(kTokens * kHidden);
  std::vector<int64_t> ref_tok(kRows);
  {
    Tensor tt = T2(ctable.data(), cd, kVocab, kHidden);
    Tensor ti = TI32(cids.data(), cd, kTokens);
    Tensor to = T2(ref_emb.data(), cd, kTokens, kHidden);
    vt::Embedding(cq, to, tt, ti);
    Tensor tl = T2(clogits.data(), cd, kRows, kVocab);
    Tensor ttok = TI64(ref_tok.data(), cd, kRows);
    vt::GreedyArgmax(cq, ttok, tl);
  }
  REQUIRE(ref_tok[0] == 2);  // the oracle itself must honour the tie-break

  for (DeviceType dt : RegisteredDevices()) {
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    if (OpAvailable(vt::OpId::kEmbedding, dt)) {
      DevBuf dtable(dev, q, kVocab * kHidden), demb(dev, q, kTokens * kHidden);
      dtable.Upload(table);
      // i32 and i64 ids are DIFFERENT index paths, so both are exercised.
      for (bool wide : {false, true}) {
        CAPTURE(wide);
        void* dids = dev.Alloc(kTokens * (wide ? sizeof(int64_t) : sizeof(int32_t)));
        if (wide) {
          dev.Copy(q, dids, ids64.data(), kTokens * sizeof(int64_t));
        } else {
          dev.Copy(q, dids, ids32.data(), kTokens * sizeof(int32_t));
        }
        dev.Synchronize(q);
        Tensor tt = T2(dtable.ptr(), d, kVocab, kHidden);
        Tensor ti = wide ? TI64(static_cast<int64_t*>(dids), d, kTokens)
                         : TI32(static_cast<int32_t*>(dids), d, kTokens);
        Tensor to = T2(demb.ptr(), d, kTokens, kHidden);
        vt::Embedding(q, to, tt, ti);
        dev.Synchronize(q);
        const std::vector<float> got = demb.Download();
        // A gather moves bytes; equality is exact, not NMSE.
        CHECK(std::memcmp(ref_emb.data(), got.data(), ref_emb.size() * sizeof(float)) == 0);
        dev.Free(dids);
      }
    }

    if (OpAvailable(vt::OpId::kGreedyArgmax, dt)) {
      DevBuf dlog(dev, q, kRows * kVocab);
      dlog.Upload(logits);
      void* dtok = dev.Alloc(kRows * sizeof(int64_t));
      Tensor tl = T2(dlog.ptr(), d, kRows, kVocab);
      Tensor ttok = TI64(static_cast<int64_t*>(dtok), d, kRows);
      vt::GreedyArgmax(q, ttok, tl);
      dev.Synchronize(q);
      std::vector<int64_t> got(kRows);
      dev.Copy(q, got.data(), dtok, kRows * sizeof(int64_t));
      dev.Synchronize(q);
      for (int64_t r = 0; r < kRows; ++r) {
        CAPTURE(r);
        CHECK(got[r] == ref_tok[r]);
      }
      dev.Free(dtok);
    }
    dev.DestroyQueue(q);
  }
  cpu.DestroyQueue(cq);
}

TEST_CASE("GEMM matches the CPU oracle within NMSE <= 5e-4, both orientations") {
  // Shapes are deliberately RAGGED and not multiples of the workgroup size, so a
  // kernel that silently processed only whole tiles would fail rather than pass
  // on a friendly shape. K is the reduction length and gets the awkward value.
  constexpr int64_t kM = 13;
  constexpr int64_t kK = 37;
  constexpr int64_t kN = 9;

  const std::vector<float> a = RandomVec(kM * kK, 401);
  const std::vector<float> b = RandomVec(kK * kN, 402);   // [K,N] for Matmul
  const std::vector<float> bt = RandomVec(kN * kK, 403);  // [N,K] for MatmulBT

  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ca = a, cb = b, cbt = bt;
  std::vector<float> ref_mm(kM * kN), ref_mmbt(kM * kN);
  {
    Tensor ta = T2(ca.data(), cd, kM, kK);
    Tensor tb = T2(cb.data(), cd, kK, kN);
    Tensor tbt = T2(cbt.data(), cd, kN, kK);
    Tensor tmm = T2(ref_mm.data(), cd, kM, kN);
    Tensor tmmbt = T2(ref_mmbt.data(), cd, kM, kN);
    vt::Matmul(cq, tmm, ta, tb);
    vt::MatmulBT(cq, tmmbt, ta, tbt);
  }

  for (DeviceType dt : RegisteredDevices()) {
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    DevBuf da(dev, q, kM * kK), dout(dev, q, kM * kN);
    da.Upload(a);
    Tensor ta = T2(da.ptr(), d, kM, kK);
    Tensor to = T2(dout.ptr(), d, kM, kN);

    if (OpAvailable(vt::OpId::kMatmul, dt)) {
      DevBuf db(dev, q, kK * kN);
      db.Upload(b);
      Tensor tb = T2(db.ptr(), d, kK, kN);
      vt::Matmul(q, to, ta, tb);
      CHECK(Nmse(ref_mm, dout.Download()) <= kNmseTol);
    }
    // MatmulBT is a DIFFERENT indexing path (the torch Linear [N,K] weight
    // layout), not a transpose of the same code, so it gets its own case.
    if (OpAvailable(vt::OpId::kMatmulBT, dt)) {
      DevBuf dbt(dev, q, kN * kK);
      dbt.Upload(bt);
      Tensor tbt = T2(dbt.ptr(), d, kN, kK);
      vt::MatmulBT(q, to, ta, tbt);
      CHECK(Nmse(ref_mmbt, dout.Download()) <= kNmseTol);
    }
    dev.DestroyQueue(q);
  }
  cpu.DestroyQueue(cq);
}

TEST_CASE("row-reducing ops match the CPU oracle within NMSE <= 5e-4") {
  // Widths chosen to exercise BOTH threadgroup regimes on a GPU: one that is a
  // clean power of two and one that is not (so the strided row loop has a
  // ragged tail), plus one narrower than a single 32-wide simd.
  for (int64_t cols : {128, 100, 17}) {
    CAPTURE(cols);
    const int64_t rows = 9;
    const size_t n = static_cast<size_t>(rows * cols);
    const std::vector<float> x = RandomVec(n, 404 + static_cast<uint32_t>(cols));
    const std::vector<float> w = RandomVec(static_cast<size_t>(cols), 505);
    const std::vector<float> bias = RandomVec(static_cast<size_t>(cols), 606);
    const std::vector<float> res0 = RandomVec(n, 707);

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cx = x, cw = w, cbias = bias;
    std::vector<float> ref_rms(n), ref_ln(n), ref_rms_res(n), ref_res_out = res0;
    {
      Tensor tx = T2(cx.data(), cd, rows, cols);
      Tensor tw = T1(cw.data(), cd, cols);
      Tensor tb = T1(cbias.data(), cd, cols);
      Tensor trms = T2(ref_rms.data(), cd, rows, cols);
      Tensor tln = T2(ref_ln.data(), cd, rows, cols);
      vt::RmsNorm(cq, trms, tx, tw, vt::RmsNormArgs{1e-6f, false}, nullptr);
      vt::LayerNorm(cq, tln, tx, &tw, &tb, vt::LayerNormArgs{1e-5f});
      // The in-place residual-stream form: residual is READ AND WRITTEN.
      Tensor tres = T2(ref_res_out.data(), cd, rows, cols);
      Tensor trr = T2(ref_rms_res.data(), cd, rows, cols);
      vt::RmsNorm(cq, trr, tx, tw, vt::RmsNormArgs{1e-6f, false}, &tres);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dx(dev, q, n), dw(dev, q, static_cast<size_t>(cols)),
          dbias(dev, q, static_cast<size_t>(cols)), dout(dev, q, n), dres(dev, q, n);
      dx.Upload(x);
      dw.Upload(w);
      dbias.Upload(bias);
      Tensor tx = T2(dx.ptr(), d, rows, cols);
      Tensor tw = T1(dw.ptr(), d, cols);
      Tensor tb = T1(dbias.ptr(), d, cols);
      Tensor to = T2(dout.ptr(), d, rows, cols);

      if (OpAvailable(vt::OpId::kRmsNorm, dt)) {
        vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{1e-6f, false}, nullptr);
        CHECK(Nmse(ref_rms, dout.Download()) <= kNmseTol);

        dres.Upload(res0);
        Tensor tres = T2(dres.ptr(), d, rows, cols);
        vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{1e-6f, false}, &tres);
        CHECK(Nmse(ref_rms_res, dout.Download()) <= kNmseTol);
        // The residual stream itself is an OUTPUT and must agree too.
        CHECK(Nmse(ref_res_out, dres.Download()) <= kNmseTol);
      }
      if (OpAvailable(vt::OpId::kLayerNorm, dt)) {
        vt::LayerNorm(q, to, tx, &tw, &tb, vt::LayerNormArgs{1e-5f});
        CHECK(Nmse(ref_ln, dout.Download()) <= kNmseTol);
      }
      dev.DestroyQueue(q);
    }
  }
}

// The single kFusedChain registration is what earns a backend the whole portable
// fusion catalog, so it gets its own cross-device case. BOTH realization tiers
// are exercised on the same recipe (kFusedAddRmsNorm: add into the residual,
// then normalize it), because they are DIFFERENT code paths on a new backend:
//   Tier 0 (default) — the device-agnostic composite in src/vt/ops.cpp walks the
//     recipe dispatching each opcode to the backend's STANDALONE ops. A backend
//     inherits it for free; what is being proven is that its standalone ops
//     compose correctly, including the in-place residual fold.
//   Tier 1 (VT_FUSED_TIER=1) — the backend's OWN single-pass kFusedChain kernel.
// The CPU oracle is recomputed per tier so like is compared with like.
TEST_CASE("FusedChain matches the CPU oracle within NMSE <= 5e-4 (both tiers)") {
  const int64_t rows = 11, cols = 96;
  const size_t n = static_cast<size_t>(rows * cols);
  const std::vector<float> x = RandomVec(n, 808);
  const std::vector<float> w = RandomVec(static_cast<size_t>(cols), 909);
  const std::vector<float> res0 = RandomVec(n, 1010);
  const vt::FusedRecipe& recipe = vt::kFusedAddRmsNorm;

  // vt::FusedTier() re-reads the environment on every call (fused_recipe.h), so
  // the tier can be flipped within this process — the same mechanism the
  // existing tests/vt/test_ops_fused_chain.cpp parity cases rely on.
  const char* prev = std::getenv("VT_FUSED_TIER");
  const std::string saved = prev != nullptr ? std::string(prev) : std::string();
  const bool had_prev = prev != nullptr;

  for (int tier : {0, 1}) {
    CAPTURE(tier);
    vllm_test::SetEnv("VT_FUSED_TIER", tier == 0 ? "0" : "1");
    // ASSERT the tier actually took effect rather than trusting the log: doctest
    // CAPTURE is lazily stringified, so a mis-set environment would silently
    // run the same path twice and still look like two-tier coverage.
    REQUIRE(vt::FusedTier() == tier);

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cx = x, cw = w, cres = res0, ref_out(n);
    {
      Tensor tx = T2(cx.data(), cd, rows, cols);
      Tensor tw = T1(cw.data(), cd, cols);
      Tensor tres = T2(cres.data(), cd, rows, cols);
      Tensor to = T2(ref_out.data(), cd, rows, cols);
      vt::FusedChain(cq, to, tx, tw, &tres, recipe, 1e-6f);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kFusedChain, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dx(dev, q, n), dw(dev, q, static_cast<size_t>(cols)), dres(dev, q, n),
          dout(dev, q, n);
      dx.Upload(x);
      dw.Upload(w);
      dres.Upload(res0);
      Tensor tx = T2(dx.ptr(), d, rows, cols);
      Tensor tw = T1(dw.ptr(), d, cols);
      Tensor tres = T2(dres.ptr(), d, rows, cols);
      Tensor to = T2(dout.ptr(), d, rows, cols);
      vt::FusedChain(q, to, tx, tw, &tres, recipe, 1e-6f);

      CHECK(Nmse(ref_out, dout.Download()) <= kNmseTol);
      CHECK(Nmse(cres, dres.Download()) <= kNmseTol);
      dev.DestroyQueue(q);
    }
  }

  if (had_prev) {
    vllm_test::SetEnv("VT_FUSED_TIER", saved);
  } else {
    vllm_test::UnsetEnv("VT_FUSED_TIER");
  }
}

// ---------------------------------------------------------------------------
// S5 PORTABLE REFERENCE TIER (accelerator-seam-audit.md, work row S5). The proof
// that op count is now a PERFORMANCE budget, not a CORRECTNESS gate: an op a
// UNIFIED-MEMORY device lacks a native kernel for falls back to the CPU reference
// and still returns the right answer, instead of throwing.
//
// Gated on Backend::UnifiedMemory() — THE safety invariant. On a discrete device
// the DevBuf pointer is a real device pointer a CPU kernel must never dereference,
// so the tier is neither installed nor exercised there (test_reference_tier.cpp
// asserts the refusal directly against a fake discrete backend). On this box's
// registered unified devices (Metal M4, GB10 CUDA/Vulkan) the pointer is
// host-accessible, so the fallback runs. On a plain CPU build there is no non-CPU
// device and the case is inert.
// i32 device buffer (state_idx / query_start_loc / has_initial_state /
// conv_state_indices). Same staging discipline as DevBuf.
class DevBufI32 {
 public:
  DevBufI32(vt::Backend& b, Queue& q, size_t n) : b_(b), q_(q), n_(n) {
    ptr_ = b_.Alloc(n * sizeof(int32_t));
  }
  ~DevBufI32() { b_.Free(ptr_); }
  DevBufI32(const DevBufI32&) = delete;
  DevBufI32& operator=(const DevBufI32&) = delete;
  void Upload(const std::vector<int32_t>& src) {
    REQUIRE(src.size() == n_);
    b_.Copy(q_, ptr_, src.data(), n_ * sizeof(int32_t));
  }
  void* ptr() const { return ptr_; }

 private:
  vt::Backend& b_;
  Queue& q_;
  size_t n_;
  void* ptr_ = nullptr;
};

// Byte-addressed device buffer for i8 masks (has_initial_state) and u16 bf16
// cache contents (sized in ELEMENTS of the templated width).
class DevBufBytes {
 public:
  DevBufBytes(vt::Backend& b, Queue& q, size_t nbytes) : b_(b), q_(q), n_(nbytes) {
    ptr_ = b_.Alloc(nbytes);
  }
  ~DevBufBytes() { b_.Free(ptr_); }
  DevBufBytes(const DevBufBytes&) = delete;
  DevBufBytes& operator=(const DevBufBytes&) = delete;
  void Upload(const void* src) { b_.Copy(q_, ptr_, src, n_); }
  void Download(void* dst) {
    b_.Synchronize(q_);
    b_.Copy(q_, dst, ptr_, n_);
    b_.Synchronize(q_);
  }
  void* ptr() const { return ptr_; }

 private:
  vt::Backend& b_;
  Queue& q_;
  size_t n_;
  void* ptr_ = nullptr;
};

// f32 -> bf16 bits through the CPU backend's own cast op, so the test never
// reimplements the codec it is comparing against.
std::vector<uint16_t> Bf16Bits(const std::vector<float>& src) {
  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> in = src;
  std::vector<uint16_t> out(src.size(), 0);
  Tensor tin = T1(in.data(), cd, static_cast<int64_t>(in.size()));
  Tensor tout = Tensor::Contiguous(out.data(), DType::kBF16, cd,
                                   {static_cast<int64_t>(out.size())});
  vt::CastBf16(cq, tout, tin);
  cpu.DestroyQueue(cq);
  return out;
}

TEST_CASE("paged attention at Qwen3 geometry (bf16, GQA 2, head_dim 128) matches the CPU oracle") {
  // #488 / ROCM-DECODE-ATTN-D128: bf16 decode at head_dim==128 (Qwen3/Llama-
  // class GQA) fell all the way to the generic PagedAttnOnline on ROCm --
  // every "fast" decode kernel was gated to d==256/512 only. Mirrors the
  // Metal "Qwen3 geometry" test's shape (nblocks/bsz/hq/hkv/dh, mixed
  // prefill+decode across 2 requests) so a bf16, GQA=2, d=128 case exists
  // for every registered device, not just Metal.
  constexpr int64_t kNBlocks = 24, kBsz = 16, kHq = 16, kHkv = 8, kDh = 128;
  constexpr int64_t kNumReqs = 2;
  const std::vector<int32_t> qsl{0, 40, 45};   // req0: 40 new (prefill); req1: 5 new
  const std::vector<int32_t> slens{40, 71};    // req1 carries 66 context tokens
  const int64_t t_total = qsl.back();
  constexpr int64_t kMaxBlocks = 6;
  std::vector<int32_t> btab(static_cast<size_t>(kNumReqs * kMaxBlocks));
  for (int64_t r = 0; r < kNumReqs; ++r) {
    for (int64_t c = 0; c < kMaxBlocks; ++c) {
      btab[static_cast<size_t>(r * kMaxBlocks + c)] = static_cast<int32_t>(r * kMaxBlocks + c);
    }
  }

  const size_t cache_elems = static_cast<size_t>(kNBlocks * kBsz * kHkv * kDh);
  const std::vector<float> qf = RandomVec(static_cast<size_t>(t_total * kHq * kDh), 811, -1.5f, 1.5f);
  const std::vector<float> kf = RandomVec(cache_elems, 812, -1.5f, 1.5f);
  const std::vector<float> vf = RandomVec(cache_elems, 813, -1.5f, 1.5f);
  const std::vector<uint16_t> qb = Bf16Bits(qf), kb = Bf16Bits(kf), vb = Bf16Bits(vf);

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(kDh));
  args.causal = true;
  args.query_start_loc_host = qsl.data();
  args.max_seq_len = 71;

  std::vector<uint16_t> ref(qb.size(), 0);
  {
    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<uint16_t> cq_v = qb, ckc = kb, cvc = vb;
    std::vector<int32_t> cbt = btab, csl = slens, cqsl = qsl;
    Tensor tq = Tensor::Contiguous(cq_v.data(), DType::kBF16, cd, {t_total, kHq, kDh});
    Tensor tkc = Tensor::Contiguous(ckc.data(), DType::kBF16, cd, {kNBlocks, kBsz, kHkv, kDh});
    Tensor tvc = Tensor::Contiguous(cvc.data(), DType::kBF16, cd, {kNBlocks, kBsz, kHkv, kDh});
    Tensor tbt = Tensor::Contiguous(cbt.data(), DType::kI32, cd, {kNumReqs, kMaxBlocks});
    Tensor tsl = Tensor::Contiguous(csl.data(), DType::kI32, cd, {kNumReqs});
    Tensor tqsl = Tensor::Contiguous(cqsl.data(), DType::kI32, cd, {kNumReqs + 1});
    Tensor to = Tensor::Contiguous(ref.data(), DType::kBF16, cd, {t_total, kHq, kDh});
    vt::PagedAttention(cq, to, tq, tkc, tvc, tbt, tsl, tqsl, args);
    cpu.DestroyQueue(cq);
  }
  std::vector<float> reff(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) reff[i] = vt::BF16ToF32(ref[i]);

  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kPagedAttention, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    DevBufBytes dq(dev, q, qb.size() * 2), dkc(dev, q, kb.size() * 2), dvc(dev, q, vb.size() * 2),
        dout(dev, q, qb.size() * 2);
    dq.Upload(qb.data());
    dkc.Upload(kb.data());
    dvc.Upload(vb.data());
    DevBufI32 dbt(dev, q, btab.size()), dsl(dev, q, slens.size()), dqsl(dev, q, qsl.size());
    dbt.Upload(btab);
    dsl.Upload(slens);
    dqsl.Upload(qsl);
    dev.Synchronize(q);

    Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kBF16, d, {t_total, kHq, kDh});
    Tensor tkc = Tensor::Contiguous(dkc.ptr(), DType::kBF16, d, {kNBlocks, kBsz, kHkv, kDh});
    Tensor tvc = Tensor::Contiguous(dvc.ptr(), DType::kBF16, d, {kNBlocks, kBsz, kHkv, kDh});
    Tensor tbt = Tensor::Contiguous(dbt.ptr(), DType::kI32, d, {kNumReqs, kMaxBlocks});
    Tensor tsl = Tensor::Contiguous(dsl.ptr(), DType::kI32, d, {kNumReqs});
    Tensor tqsl = Tensor::Contiguous(dqsl.ptr(), DType::kI32, d, {kNumReqs + 1});
    Tensor to = Tensor::Contiguous(dout.ptr(), DType::kBF16, d, {t_total, kHq, kDh});

    vt::ResetOpProviderStats(vt::OpId::kPagedAttention, dt);
    vt::PagedAttention(q, to, tq, tkc, tvc, tbt, tsl, tqsl, args);
    dev.Synchronize(q);
    CHECK(vt::GetOpProviderStats(vt::OpId::kPagedAttention, dt).declines == 0);

    std::vector<uint16_t> got(qb.size());
    dout.Download(got.data());
    std::vector<float> gotf(got.size());
    for (size_t i = 0; i < got.size(); ++i) gotf[i] = vt::BF16ToF32(got[i]);
    CHECK(Nmse(reff, gotf) <= kNmseTol);

    dev.DestroyQueue(q);
  }
}

// Rank-3 padded-row view [T, H, D] over a [T, row_stride] f32 buffer — the
// merged-qkvz slice shape the GDN/attention glue ops consume in the model.
Tensor T3PaddedF32(void* p, Device d, int64_t t, int64_t h, int64_t w,
                   int64_t row_stride) {
  Tensor t3 = Tensor::Contiguous(p, DType::kF32, d, {t, h, w});
  t3.stride[0] = row_stride;
  return t3;
}


// --- GDN cases (BACKEND-ROCM-GDN-KERNELS) -------------------------------------

TEST_CASE("GDN state gather/scatter are BIT-EXACT against the CPU oracle") {
  // Indexed data movement between an f32 working set and a persistent cache:
  // no arithmetic anywhere, so the bar is byte equality — including the bf16
  // cache arm, where both sides apply the same RNE round at the boundary.
  // Covers the uniform layout (cache_inner == work_inner), the spec-widened
  // layout (leading work_inner cols per channel at the physical stride), the
  // has_initial_state mask in i8/i32/absent forms, and scatter's untouched-row
  // preservation.
  const int64_t S = 8, R = 4, mid = 4, w_in = 6, c_in = 8;  // c_in>w_in: widened
  for (bool widened : {false, true}) {
    const int64_t cache_inner = widened ? c_in : w_in;
    CAPTURE(widened);
    const size_t cache_n = static_cast<size_t>(S * mid * cache_inner);
    const size_t work_n = static_cast<size_t>(R * mid * w_in);
    const std::vector<float> cache_f = RandomVec(cache_n, 910);
    const std::vector<uint16_t> cache_bf = Bf16Bits(cache_f);
    const std::vector<int32_t> idx = {1, 0, 7, 6};  // unique slots
    const std::vector<int32_t> has32 = {1, 0, 1, 0};
    const std::vector<int8_t> has8 = {1, 0, 1, 0};

    for (int arm = 0; arm < 2; ++arm) {  // 0 = f32 cache, 1 = bf16 cache
      CAPTURE(arm);
      for (int has = 0; has < 3; ++has) {  // 0 = absent, 1 = i8, 2 = i32
        CAPTURE(has);
        // ---- CPU oracle: gather, then scatter the gathered rows back.
        std::vector<float> ref_work(work_n, -1.0f);
        std::vector<float> ref_cache_f = cache_f;
        std::vector<uint16_t> ref_cache_bf = cache_bf;
        std::vector<int32_t> ci = idx, ch32 = has32;
        std::vector<int8_t> ch8 = has8;
        {
          vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
          Queue cq = cpu.CreateQueue();
          const Device cd{DeviceType::kCPU, 0};
          Tensor tidx = TI32(ci.data(), cd, R);
          Tensor th8 = Tensor::Contiguous(ch8.data(), DType::kI8, cd, {R});
          Tensor th32 = TI32(ch32.data(), cd, R);
          Tensor tw = Tensor::Contiguous(ref_work.data(), DType::kF32, cd, {R, mid, w_in});
          const Tensor* ph = has == 1 ? static_cast<const Tensor*>(&th8)
                             : has == 2 ? static_cast<const Tensor*>(&th32)
                                        : nullptr;
          if (arm == 0) {
            Tensor tc = Tensor::Contiguous(ref_cache_f.data(), DType::kF32, cd,
                                           {S, mid, cache_inner});
            vt::GdnStateGather(cq, tw, tc, tidx, ph);
            vt::GdnStateScatter(cq, tc, tw, tidx);
          } else {
            Tensor tc = Tensor::Contiguous(ref_cache_bf.data(), DType::kBF16, cd,
                                           {S, mid, cache_inner});
            vt::GdnStateGather(cq, tw, tc, tidx, ph);
            vt::GdnStateScatter(cq, tc, tw, tidx);
          }
          cpu.DestroyQueue(cq);
        }

        for (DeviceType dt : RegisteredDevices()) {
          if (!OpAvailable(vt::OpId::kGdnStateGather, dt) ||
              !OpAvailable(vt::OpId::kGdnStateScatter, dt))
            continue;
          CAPTURE(DeviceName(dt));
          vt::Backend& dev = vt::GetBackend(dt);
          Queue q = dev.CreateQueue();
          const Device d{dt, 0};
          DevBuf dwork(dev, q, work_n);
          DevBufI32 didx(dev, q, R), dhas32(dev, q, R);
          DevBufBytes dhas8(dev, q, R);
          didx.Upload(idx);
          dhas32.Upload(has32);
          dhas8.Upload(has8.data());
          Tensor tidx = TI32(didx.ptr(), d, R);
          Tensor th8 = Tensor::Contiguous(dhas8.ptr(), DType::kI8, d, {R});
          Tensor th32 = TI32(dhas32.ptr(), d, R);
          Tensor tw = Tensor::Contiguous(dwork.ptr(), DType::kF32, d, {R, mid, w_in});
          const Tensor* ph = has == 1 ? &th8 : has == 2 ? &th32 : nullptr;

          const size_t cache_bytes = cache_n * (arm == 0 ? 4 : 2);
          DevBufBytes dcache(dev, q, cache_bytes);
          dcache.Upload(arm == 0 ? static_cast<const void*>(cache_f.data())
                                 : static_cast<const void*>(cache_bf.data()));
          Tensor tc = Tensor::Contiguous(dcache.ptr(),
                                         arm == 0 ? DType::kF32 : DType::kBF16, d,
                                         {S, mid, cache_inner});
          vt::GdnStateGather(q, tw, tc, tidx, ph);
          CHECK(dwork.Download() == ref_work);  // gather: bit-exact
          vt::GdnStateScatter(q, tc, tw, tidx);
          if (arm == 0) {
            std::vector<float> got(cache_n);
            dcache.Download(got.data());
            CHECK(got == ref_cache_f);  // scatter round-trip: bit-exact
          } else {
            std::vector<uint16_t> got(cache_n);
            dcache.Download(got.data());
            CHECK(got == ref_cache_bf);
          }
          dev.DestroyQueue(q);
        }
      }
    }
  }
}


TEST_CASE("causal conv1d fwd/update match the CPU oracle") {
  // §2/§3 of gdn-semantics.md. Outputs are arithmetic (silu epilogue) -> NMSE;
  // the conv_state write-back/roll moves RAW x values -> bit-exact.
  const int64_t C = 24, K = 4, W = K - 1;
  const std::vector<int32_t> qsl = {0, 5, 6, 15};  // 3 seqs: lens 5, 1, 9
  const std::vector<int32_t> has = {1, 0, 1};
  const int64_t T = 15, N = 3;
  const size_t xn = static_cast<size_t>(T * C), wn = static_cast<size_t>(C * K);
  const std::vector<float> x = RandomVec(xn, 811);
  const std::vector<float> w = RandomVec(wn, 812, -0.5f, 0.5f);
  const std::vector<float> bias = RandomVec(static_cast<size_t>(C), 813, -0.2f, 0.2f);
  const std::vector<float> st0 = RandomVec(static_cast<size_t>(N * C * W), 814, -0.5f, 0.5f);

  // CPU oracle (fwd).
  std::vector<float> ref_out(xn, 0.0f), ref_state = st0;
  {
    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cx = x, cw = w, cb = bias;
    std::vector<int32_t> cqsl = qsl, chas = has;
    Tensor tx = T2(cx.data(), cd, T, C);
    Tensor tw = T2(cw.data(), cd, C, K);
    Tensor tb = T1(cb.data(), cd, C);
    Tensor tst = Tensor::Contiguous(ref_state.data(), DType::kF32, cd, {N, C, W});
    Tensor tqsl = TI32(cqsl.data(), cd, N + 1);
    Tensor this_ = TI32(chas.data(), cd, N);
    Tensor tout = T2(ref_out.data(), cd, T, C);
    vt::CausalConv1dFwd(cq, tout, tx, tw, &tb, tst, tqsl, this_, vt::CausalConv1dArgs{});
    cpu.DestroyQueue(cq);
  }

  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kCausalConv1dFwd, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};
    DevBuf dx(dev, q, xn), dw(dev, q, wn), db(dev, q, static_cast<size_t>(C)),
        dout(dev, q, xn), dst(dev, q, static_cast<size_t>(N * C * W));
    DevBufI32 dqsl(dev, q, N + 1), dhas(dev, q, N);
    dx.Upload(x);
    dw.Upload(w);
    db.Upload(bias);
    dst.Upload(st0);
    dqsl.Upload(qsl);
    dhas.Upload(has);
    Tensor tx = T2(dx.ptr(), d, T, C);
    Tensor tw = T2(dw.ptr(), d, C, K);
    Tensor tb = T1(db.ptr(), d, C);
    Tensor tst = Tensor::Contiguous(dst.ptr(), DType::kF32, d, {N, C, W});
    Tensor tqsl = TI32(dqsl.ptr(), d, N + 1);
    Tensor this_ = TI32(dhas.ptr(), d, N);
    Tensor tout = T2(dout.ptr(), d, T, C);
    vt::CausalConv1dFwd(q, tout, tx, tw, &tb, tst, tqsl, this_, vt::CausalConv1dArgs{});
    CHECK(Nmse(ref_out, dout.Download()) <= kNmseTol);
    CHECK(dst.Download() == ref_state);  // raw-x write-back: bit-exact

    // Exact-chunks descriptor form (VT_CONV_EXACT_CHUNKS — the shape Qwen3.5
    // prefill actually passes on the live path): one program per (sequence,
    // 8-token chunk). lens 5,1,9 -> programs (s0,c0), (s1,c0), (s2,c0),
    // (s2,c1). Must equal the same CPU oracle (the CPU keeps the scalar
    // mapping; the descriptors only re-slice the work).
    const std::vector<int32_t> batch_ptr = {0, 1, 2, 2};
    const std::vector<int32_t> chunk_off = {0, 0, 0, 1};
    DevBufI32 dbp(dev, q, batch_ptr.size()), dtco(dev, q, chunk_off.size());
    dbp.Upload(batch_ptr);
    dtco.Upload(chunk_off);
    dst.Upload(st0);  // reset state for the descriptor run
    Tensor tbp = TI32(dbp.ptr(), d, static_cast<int64_t>(batch_ptr.size()));
    Tensor ttco = TI32(dtco.ptr(), d, static_cast<int64_t>(chunk_off.size()));
    vt::CausalConv1dArgs exact_args;
    exact_args.batch_ptr = &tbp;
    exact_args.token_chunk_offset_ptr = &ttco;
    vt::CausalConv1dFwd(q, tout, tx, tw, &tb, tst, tqsl, this_, exact_args);
    CHECK(Nmse(ref_out, dout.Download()) <= kNmseTol);
    CHECK(dst.Download() == ref_state);  // descriptor write-back: bit-exact
    dev.DestroyQueue(q);
  }

  // Update: B=4 tokens. Compact arm: conv_state [B,C,W] (one row per token,
  // per the ops.cpp contract). Indexed arm: the full [SLOTS,C,W] cache with one
  // NULL slot (-1 -> out row untouched).
  const int64_t B = 4, SLOTS = 6;
  const size_t un = static_cast<size_t>(B * C);
  const std::vector<int32_t> cidx = {3, -1, 0, 5};
  const std::vector<float> ux = RandomVec(un, 821);
  const std::vector<float> ust0 = RandomVec(static_cast<size_t>(SLOTS * C * W), 822, -0.5f, 0.5f);
  for (bool indexed : {false, true}) {
    CAPTURE(indexed);
    const int64_t st_rows = indexed ? SLOTS : B;
    const std::vector<float> ust_arm(ust0.begin(), ust0.begin() + st_rows * C * W);
    // CPU oracle.
    std::vector<float> ref_uout(un, -2.0f), ref_ust = ust_arm;
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> cx = ux, cw = w, cb = bias;
      std::vector<int32_t> cci = cidx;
      Tensor tx = T2(cx.data(), cd, B, C);
      Tensor tw = T2(cw.data(), cd, C, K);
      Tensor tb = T1(cb.data(), cd, C);
      Tensor tst = Tensor::Contiguous(ref_ust.data(), DType::kF32, cd, {st_rows, C, W});
      Tensor tci = TI32(cci.data(), cd, B);
      Tensor tout = T2(ref_uout.data(), cd, B, C);
      vt::CausalConv1dUpdate(cq, tout, tx, tw, &tb, tst, vt::CausalConv1dArgs{},
                             indexed ? &tci : nullptr);
      cpu.DestroyQueue(cq);
    }
    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kCausalConv1dUpdate, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf dx(dev, q, un), dw(dev, q, wn), db(dev, q, static_cast<size_t>(C)),
          dout(dev, q, un), dst(dev, q, static_cast<size_t>(st_rows * C * W));
      DevBufI32 dci(dev, q, B);
      dx.Upload(ux);
      dw.Upload(w);
      db.Upload(bias);
      dst.Upload(ust_arm);
      dci.Upload(cidx);
      // Untouched-row sentinel must match the oracle's initial -2 fill.
      dout.Upload(std::vector<float>(un, -2.0f));
      Tensor tx = T2(dx.ptr(), d, B, C);
      Tensor tw = T2(dw.ptr(), d, C, K);
      Tensor tb = T1(db.ptr(), d, C);
      Tensor tst = Tensor::Contiguous(dst.ptr(), DType::kF32, d, {st_rows, C, W});
      Tensor tci = TI32(dci.ptr(), d, B);
      Tensor tout = T2(dout.ptr(), d, B, C);
      vt::CausalConv1dUpdate(q, tout, tx, tw, &tb, tst, vt::CausalConv1dArgs{},
                             indexed ? &tci : nullptr);
      CHECK(Nmse(ref_uout, dout.Download()) <= kNmseTol);
      CHECK(dst.Download() == ref_ust);  // roll: bit-exact
      dev.DestroyQueue(q);
    }
  }
}


TEST_CASE("GdnPostConv matches the CPU oracle within NMSE <= 5e-4") {
  // The fused post-conv glue (the VT_GLUE_FUSE path the model calls by
  // default): conv-split + q/k l2norm + g/beta in one launch, with padded a/b
  // row strides. All arithmetic: NMSE.
  const int64_t T = 4, HK = 2, DK = 16, HV = 4, DV = 24;
  const int64_t key_dim = HK * DK, value_dim = HV * DV;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t a_outer = HV + 3, b_outer = HV + 5;
  const std::vector<float> conv = RandomVec(static_cast<size_t>(T * conv_dim), 871, -0.5f, 0.5f);
  const std::vector<float> araw = RandomVec(static_cast<size_t>(T * a_outer), 872, -0.4f, 0.4f);
  const std::vector<float> braw = RandomVec(static_cast<size_t>(T * b_outer), 873, -0.4f, 0.4f);
  const std::vector<float> a_log = RandomVec(static_cast<size_t>(HV), 874, -2.0f, -0.5f);
  const std::vector<float> dt_bias = RandomVec(static_cast<size_t>(HV), 875, -0.1f, 0.1f);
  vt::L2NormArgs l2a;
  l2a.eps = 1e-6f;

  std::vector<float> ref_q(static_cast<size_t>(T * key_dim)),
      ref_k(static_cast<size_t>(T * key_dim)), ref_v(static_cast<size_t>(T * value_dim)),
      ref_g(static_cast<size_t>(T * HV)), ref_b(static_cast<size_t>(T * HV));
  {
    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cc = conv, ca = araw, cb = braw, cal = a_log, cdt = dt_bias;
    // araw/braw reach the op as padded [T, HV] rank-2 views (row stride honored).
    auto Make = [](void* p, Device dd, std::initializer_list<int64_t> s) {
      return Tensor::Contiguous(p, DType::kF32, dd, s);
    };
    Tensor tq = Make(ref_q.data(), cd, {T, HK, DK});
    Tensor tk = Make(ref_k.data(), cd, {T, HK, DK});
    Tensor tv = Make(ref_v.data(), cd, {T, HV, DV});
    Tensor tg = Make(ref_g.data(), cd, {T, HV});
    Tensor tb = Make(ref_b.data(), cd, {T, HV});
    Tensor tc = Make(cc.data(), cd, {T, conv_dim});
    Tensor ta = Make(ca.data(), cd, {T, a_outer});  // logical HV cols, padded row
    ta.shape[1] = HV;  // view narrows the row; stride[0] stays a_outer
    Tensor tb2 = Make(cb.data(), cd, {T, b_outer});
    tb2.shape[1] = HV;
    Tensor tal = Make(cal.data(), cd, {HV});
    Tensor tdt = Make(cdt.data(), cd, {HV});
    vt::GdnPostConv(cq, tq, tk, tv, tg, tb, tc, ta, tb2, tal, tdt, l2a);
    cpu.DestroyQueue(cq);
  }
  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kGdnPostConv, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};
    DevBuf dq(dev, q, ref_q.size()), dk(dev, q, ref_k.size()), dv(dev, q, ref_v.size()),
        dg(dev, q, ref_g.size()), db(dev, q, ref_b.size()), dc(dev, q, conv.size()),
        da(dev, q, araw.size()), db2(dev, q, braw.size()), dal(dev, q, HV),
        ddt(dev, q, HV);
    dq.Upload(std::vector<float>(ref_q.size(), 0.0f));
    dc.Upload(conv);
    da.Upload(araw);
    db2.Upload(braw);
    dal.Upload(a_log);
    ddt.Upload(dt_bias);
    auto Make = [&](void* p, std::initializer_list<int64_t> s) {
      return Tensor::Contiguous(p, DType::kF32, d, s);
    };
    Tensor tq = Make(dq.ptr(), {T, HK, DK});
    Tensor tk = Make(dk.ptr(), {T, HK, DK});
    Tensor tv = Make(dv.ptr(), {T, HV, DV});
    Tensor tg = Make(dg.ptr(), {T, HV});
    Tensor tb = Make(db.ptr(), {T, HV});
    Tensor tc = Make(dc.ptr(), {T, conv_dim});
    Tensor ta = Make(da.ptr(), {T, a_outer});
    ta.shape[1] = HV;
    ta.stride[0] = a_outer;
    Tensor tb2 = Make(db2.ptr(), {T, b_outer});
    tb2.shape[1] = HV;
    tb2.stride[0] = b_outer;
    Tensor tal = Make(dal.ptr(), {HV});
    Tensor tdt = Make(ddt.ptr(), {HV});
    vt::GdnPostConv(q, tq, tk, tv, tg, tb, tc, ta, tb2, tal, tdt, l2a);
    CHECK(Nmse(ref_q, dq.Download()) <= kNmseTol);
    CHECK(Nmse(ref_k, dk.Download()) <= kNmseTol);
    CHECK(Nmse(ref_v, dv.Download()) <= kNmseTol);
    CHECK(Nmse(ref_g, dg.Download()) <= kNmseTol);
    CHECK(Nmse(ref_b, db.Download()) <= kNmseTol);
    dev.DestroyQueue(q);
  }
}

TEST_CASE("GDN prefill/decode recurrence matches the CPU oracle within NMSE <= 5e-4") {
  // §7/§8. All f32. NMSE on out AND on the in-place state (the recurrence is
  // arithmetic end to end). Decode covers the compact arm, the indexed arm,
  // and the NULL-slot zero-out.
  const int64_t HK = 2, HV = 4, DK = 16, DV = 24;  // HV = ratio*HK
  const float scale = 0.25f;
  vt::GdnArgs ga;
  ga.scale = scale;

  // ---- prefill: two sequences, lens 4 and 1, fresh zero state.
  const std::vector<int32_t> qsl = {0, 4, 5};
  const int64_t N = 2, T = 5;
  const size_t qkn = static_cast<size_t>(T * HK * DK), vn = static_cast<size_t>(T * HV * DV);
  const size_t gbn = static_cast<size_t>(T * HV), stn = static_cast<size_t>(N * HV * DV * DK);
  const std::vector<float> qin = RandomVec(qkn, 851, -0.5f, 0.5f);
  const std::vector<float> kin = RandomVec(qkn, 852, -0.5f, 0.5f);
  const std::vector<float> vin = RandomVec(vn, 853, -0.5f, 0.5f);
  const std::vector<float> gin = RandomVec(gbn, 854, -0.3f, -0.01f);  // log-decay < 0
  const std::vector<float> bin = RandomVec(gbn, 855, 0.0f, 0.5f);

  std::vector<float> ref_out(vn, 0.0f), ref_st(stn, 0.0f);
  {
    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> hq = qin, hk_ = kin, hv_ = vin, hg = gin, hb = bin;
    std::vector<int32_t> cqsl = qsl;
    Tensor tq = Tensor::Contiguous(hq.data(), DType::kF32, cd, {T, HK, DK});
    Tensor tk = Tensor::Contiguous(hk_.data(), DType::kF32, cd, {T, HK, DK});
    Tensor tv = Tensor::Contiguous(hv_.data(), DType::kF32, cd, {T, HV, DV});
    Tensor tg = T2(hg.data(), cd, T, HV);
    Tensor tb = T2(hb.data(), cd, T, HV);
    Tensor tst = Tensor::Contiguous(ref_st.data(), DType::kF32, cd, {N, HV, DV, DK});
    Tensor tqsl = TI32(cqsl.data(), cd, N + 1);
    Tensor tout = Tensor::Contiguous(ref_out.data(), DType::kF32, cd, {T, HV, DV});
    vt::GdnPrefill(cq, tout, tq, tk, tv, tg, tb, tst, tqsl, ga);
    cpu.DestroyQueue(cq);
  }
  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kGdnPrefill, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};
    DevBuf dq(dev, q, qkn), dk(dev, q, qkn), dv(dev, q, vn), dg(dev, q, gbn),
        db(dev, q, gbn), dout(dev, q, vn), dst(dev, q, stn);
    DevBufI32 dqsl(dev, q, N + 1);
    dq.Upload(qin);
    dk.Upload(kin);
    dv.Upload(vin);
    dg.Upload(gin);
    db.Upload(bin);
    dst.Upload(std::vector<float>(stn, 0.0f));
    dqsl.Upload(qsl);
    Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {T, HK, DK});
    Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {T, HK, DK});
    Tensor tv = Tensor::Contiguous(dv.ptr(), DType::kF32, d, {T, HV, DV});
    Tensor tg = T2(dg.ptr(), d, T, HV);
    Tensor tb = T2(db.ptr(), d, T, HV);
    Tensor tst = Tensor::Contiguous(dst.ptr(), DType::kF32, d, {N, HV, DV, DK});
    Tensor tqsl = TI32(dqsl.ptr(), d, N + 1);
    Tensor tout = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {T, HV, DV});
    vt::GdnPrefill(q, tout, tq, tk, tv, tg, tb, tst, tqsl, ga);
    CHECK(Nmse(ref_out, dout.Download()) <= kNmseTol);
    CHECK(Nmse(ref_st, dst.Download()) <= kNmseTol);
    dev.DestroyQueue(q);
  }

  // ---- decode: B=3 tokens over a 4-slot cache; slot -1 => zero out row,
  // state untouched. Compact arm (no indices) alongside.
  const int64_t B = 3, SLOTS = 4;
  const size_t dqkn = static_cast<size_t>(B * HK * DK), dvn = static_cast<size_t>(B * HV * DV);
  const size_t dgbn = static_cast<size_t>(B * HV);
  const std::vector<int32_t> sidx = {2, -1, 0};
  const std::vector<float> dq_in = RandomVec(dqkn, 861, -0.5f, 0.5f);
  const std::vector<float> dk_in = RandomVec(dqkn, 862, -0.5f, 0.5f);
  const std::vector<float> dv_in = RandomVec(dvn, 863, -0.5f, 0.5f);
  const std::vector<float> dg_in = RandomVec(dgbn, 864, -0.3f, -0.01f);
  const std::vector<float> db_in = RandomVec(dgbn, 865, 0.0f, 0.5f);
  const size_t dstn = static_cast<size_t>(SLOTS * HV * DV * DK);
  const std::vector<float> dst0 = RandomVec(dstn, 866, -0.4f, 0.4f);
  for (bool indexed : {false, true}) {
    CAPTURE(indexed);
    const int64_t st_rows = indexed ? SLOTS : B;
    const std::vector<float> dst_arm(dst0.begin(), dst0.begin() + st_rows * HV * DV * DK);
    std::vector<float> ref_dout(dvn, -7.0f), ref_dst = dst_arm;
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> hq = dq_in, hk_ = dk_in, hv_ = dv_in, hg = dg_in, hb = db_in;
      std::vector<int32_t> csi = sidx;
      Tensor tq = Tensor::Contiguous(hq.data(), DType::kF32, cd, {B, HK, DK});
      Tensor tk = Tensor::Contiguous(hk_.data(), DType::kF32, cd, {B, HK, DK});
      Tensor tv = Tensor::Contiguous(hv_.data(), DType::kF32, cd, {B, HV, DV});
      Tensor tg = T2(hg.data(), cd, B, HV);
      Tensor tb = T2(hb.data(), cd, B, HV);
      Tensor tst = Tensor::Contiguous(ref_dst.data(), DType::kF32, cd, {st_rows, HV, DV, DK});
      Tensor tsi = TI32(csi.data(), cd, B);
      Tensor tout = Tensor::Contiguous(ref_dout.data(), DType::kF32, cd, {B, HV, DV});
      vt::GdnDecode(cq, tout, tq, tk, tv, tg, tb, tst, ga, indexed ? &tsi : nullptr);
      cpu.DestroyQueue(cq);
    }
    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kGdnDecode, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf dq(dev, q, dqkn), dk(dev, q, dqkn), dv(dev, q, dvn), dg(dev, q, dgbn),
          db(dev, q, dgbn), dout(dev, q, dvn), dst(dev, q, static_cast<size_t>(st_rows * HV * DV * DK));
      DevBufI32 dsi(dev, q, B);
      dq.Upload(dq_in);
      dk.Upload(dk_in);
      dv.Upload(dv_in);
      dg.Upload(dg_in);
      db.Upload(db_in);
      dst.Upload(dst_arm);
      dsi.Upload(sidx);
      dout.Upload(std::vector<float>(dvn, -7.0f));  // untouched-row sentinel
      Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {B, HK, DK});
      Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {B, HK, DK});
      Tensor tv = Tensor::Contiguous(dv.ptr(), DType::kF32, d, {B, HV, DV});
      Tensor tg = T2(dg.ptr(), d, B, HV);
      Tensor tb = T2(db.ptr(), d, B, HV);
      Tensor tst = Tensor::Contiguous(dst.ptr(), DType::kF32, d, {st_rows, HV, DV, DK});
      Tensor tsi = TI32(dsi.ptr(), d, B);
      Tensor tout = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {B, HV, DV});
      vt::GdnDecode(q, tout, tq, tk, tv, tg, tb, tst, ga, indexed ? &tsi : nullptr);
      CHECK(Nmse(ref_dout, dout.Download()) <= kNmseTol);
      CHECK(Nmse(ref_dst, dst.Download()) <= kNmseTol);
      dev.DestroyQueue(q);
    }
  }
}


TEST_CASE("RmsNormGated and SigmoidGate match the CPU oracle") {
  // §5. RmsNormGated: NMSE (rms reduction + gate activation), both gate
  // activations, and BOTH gate layouts — contiguous rank-2 and the padded-row
  // rank-3 [T,Hv,D] merged-qkvz view. SigmoidGateBf16 is a single multiply
  // with an RNE store both sides apply: bit-exact.
  const int64_t T = 5, HV = 3, D = 32;
  const int64_t rows = T * HV;
  const int64_t gate_outer = HV * D + 8;  // padded token stride (rank-3 arm)
  const size_t xn = static_cast<size_t>(rows * D);
  const std::vector<float> x = RandomVec(xn, 831);
  const std::vector<float> gate = RandomVec(static_cast<size_t>(T * gate_outer), 832);
  const std::vector<float> w = RandomVec(static_cast<size_t>(D), 833, 0.2f, 1.0f);

  for (bool sig : {false, true}) {
    for (bool rank3 : {false, true}) {
      CAPTURE(sig);
      CAPTURE(rank3);
      vt::RmsNormGatedArgs args;
      args.sigmoid_gate = sig;
      // rank3 arm: x/gate/out are [T,Hv,D] (gate padded-row); rank2 arm: all
      // [rows,D] contiguous (gate buffer's leading rows*D elements).
      std::vector<float> ref(xn, 0.0f);
      {
        vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
        Queue cq = cpu.CreateQueue();
        const Device cd{DeviceType::kCPU, 0};
        std::vector<float> cx = x, cg = gate, cw = w;
        Tensor tx = rank3 ? Tensor::Contiguous(cx.data(), DType::kF32, cd, {T, HV, D})
                          : T2(cx.data(), cd, rows, D);
        Tensor tg = rank3 ? T3PaddedF32(cg.data(), cd, T, HV, D, gate_outer)
                          : T2(cg.data(), cd, rows, D);
        Tensor tw = T1(cw.data(), cd, D);
        Tensor tout = rank3 ? Tensor::Contiguous(ref.data(), DType::kF32, cd, {T, HV, D})
                            : T2(ref.data(), cd, rows, D);
        vt::RmsNormGated(cq, tout, tx, tg, tw, args);
        cpu.DestroyQueue(cq);
      }
      for (DeviceType dt : RegisteredDevices()) {
        if (!OpAvailable(vt::OpId::kRmsNormGated, dt)) continue;
        CAPTURE(DeviceName(dt));
        vt::Backend& dev = vt::GetBackend(dt);
        Queue q = dev.CreateQueue();
        const Device d{dt, 0};
        DevBuf dx(dev, q, xn), dg(dev, q, gate.size()), dw(dev, q, D), dout(dev, q, xn);
        dx.Upload(x);
        dg.Upload(gate);
        dw.Upload(w);
        Tensor tx = rank3 ? Tensor::Contiguous(dx.ptr(), DType::kF32, d, {T, HV, D})
                          : T2(dx.ptr(), d, rows, D);
        Tensor tg = rank3 ? T3PaddedF32(dg.ptr(), d, T, HV, D, gate_outer)
                          : T2(dg.ptr(), d, rows, D);
        Tensor tw = T1(dw.ptr(), d, D);
        Tensor tout = rank3 ? Tensor::Contiguous(dout.ptr(), DType::kF32, d, {T, HV, D})
                            : T2(dout.ptr(), d, rows, D);
        vt::RmsNormGated(q, tout, tx, tg, tw, args);
        CHECK(Nmse(ref, dout.Download()) <= kNmseTol);
        dev.DestroyQueue(q);
      }
    }
  }

  // SigmoidGateBf16: out bf16, gate f32, attn bf16 OR f32 (the FA-2 prefill
  // combo) — single multiply with the same RNE store on both sides: bit-exact.
  const size_t sn = 256;
  const std::vector<float> attn_f = RandomVec(sn, 841);
  const std::vector<float> gate_f = RandomVec(sn, 842);
  const std::vector<uint16_t> attn_bf = Bf16Bits(attn_f);
  for (bool attn_is_bf16 : {true, false}) {
    CAPTURE(attn_is_bf16);
    std::vector<uint16_t> ref_sg(sn, 0);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<uint16_t> ca = attn_bf;
      std::vector<float> caf = attn_f, cg = gate_f;
      Tensor ta = attn_is_bf16
                      ? Tensor::Contiguous(ca.data(), DType::kBF16, cd, {static_cast<int64_t>(sn)})
                      : Tensor::Contiguous(caf.data(), DType::kF32, cd, {static_cast<int64_t>(sn)});
      Tensor tg = T1(cg.data(), cd, static_cast<int64_t>(sn));
      Tensor tout = Tensor::Contiguous(ref_sg.data(), DType::kBF16, cd, {static_cast<int64_t>(sn)});
      vt::SigmoidGateBf16(cq, tout, ta, tg);
      cpu.DestroyQueue(cq);
    }
    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kSigmoidGateBf16, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBufBytes da(dev, q, sn * (attn_is_bf16 ? 2 : 4));
      DevBuf dg(dev, q, sn);
      DevBufBytes dout(dev, q, sn * 2);
      if (attn_is_bf16) {
        da.Upload(attn_bf.data());
      } else {
        da.Upload(attn_f.data());
      }
      dg.Upload(gate_f);
      Tensor ta = Tensor::Contiguous(da.ptr(), attn_is_bf16 ? DType::kBF16 : DType::kF32, d,
                                     {static_cast<int64_t>(sn)});
      Tensor tg = T1(dg.ptr(), d, static_cast<int64_t>(sn));
      Tensor tout = Tensor::Contiguous(dout.ptr(), DType::kBF16, d, {static_cast<int64_t>(sn)});
      vt::SigmoidGateBf16(q, tout, ta, tg);
      std::vector<uint16_t> got(sn);
      dout.Download(got.data());
      CHECK(got == ref_sg);
      dev.DestroyQueue(q);
    }
  }
}


TEST_CASE("FusedNormRope matches the CPU oracle within NMSE <= 5e-4, both styles") {
  // The Tier-A2+A5 MLA norm-rope fold (ROCM-FUSED-NORM-ROPE, #2564). It is the
  // ONE op in this harness whose ABSENCE on a backend is a refusal rather than
  // a slowdown: `mla_attention.cpp` branches on
  // `vt::OpRegistered(kFusedNormRope, device)` BEFORE calling it, and the
  // fallback that branch selects row-slices a weight that a keep-quant MLA
  // checkpoint stores block-quantized. So a device that skips this case cannot
  // serve GLM-5.3 at all, and a SKIP here is a gap, not a pass.
  //
  // Geometry is DeepSeek/GLM-shaped but small: off = kv_lora_rank, rot =
  // qk_rope_head_dim. `off` is deliberately NOT a multiple of the 256-wide
  // block, so the strided reduction loop's tail is exercised rather than
  // divided away.
  constexpr int64_t kTokens = 13, kOff = 130, kRot = 16, kMaxPos = 64;
  constexpr float kEps = 1e-6f;

  const std::vector<float> x0 = RandomVec(kTokens * (kOff + kRot), 9101);
  const std::vector<float> w0 = RandomVec(kOff, 9102, -1.0f, 1.0f);
  const std::vector<float> cache = RandomVec(kMaxPos * kRot, 9103, -1.0f, 1.0f);
  // Positions are NOT 0..n-1: a kernel reading the token index instead of the
  // position passes on the identity mapping and fails here.
  std::vector<int32_t> pos(kTokens);
  for (int64_t i = 0; i < kTokens; ++i) {
    pos[static_cast<size_t>(i)] = int32_t((i * 5 + 2) % kMaxPos);
  }

  // NeoX rotates (pair, pair+half); GPT-J style rotates (2*pair, 2*pair+1).
  // DeepSeek-V2/V3 and GLM-5.3 use the GPT-J form, so `false` is the arm this
  // model actually runs and `true` is the one a hardcoded kernel would break.
  for (bool neox : {true, false}) {
    CAPTURE(neox);
    vt::RmsNormArgs na;
    na.eps = kEps;
    na.gemma = false;  // DeepSeek/GLM: plain RMSNorm, not (1 + w)
    vt::RopeArgs ra;
    ra.rotary_dim = kRot;
    ra.is_neox_style = neox;

    std::vector<float> ref_lat(kTokens * kOff), ref_pe(kTokens * kRot);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> cx = x0, cw = w0, cc = cache;
      std::vector<int32_t> cpos = pos;
      Tensor tl = Tensor::Contiguous(ref_lat.data(), DType::kF32, cd, {kTokens, kOff});
      Tensor tp = Tensor::Contiguous(ref_pe.data(), DType::kF32, cd, {kTokens, kRot});
      Tensor tx = Tensor::Contiguous(cx.data(), DType::kF32, cd, {kTokens, kOff + kRot});
      Tensor tw = Tensor::Contiguous(cw.data(), DType::kF32, cd, {kOff});
      Tensor tpos = Tensor::Contiguous(cpos.data(), DType::kI32, cd, {kTokens});
      Tensor tc = Tensor::Contiguous(cc.data(), DType::kF32, cd, {kMaxPos, kRot});
      vt::FusedNormRope(cq, tl, tp, tx, tw, tpos, tc, na, ra);
      cpu.DestroyQueue(cq);
    }

    // The CPU oracle is itself the composite {RmsNorm ; RopeFromCache}, so an
    // all-zero reference would make the NMSE ratio vacuous. Assert it is not.
    double mag = 0.0;
    for (float v : ref_lat) mag += std::fabs(static_cast<double>(v));
    for (float v : ref_pe) mag += std::fabs(static_cast<double>(v));
    REQUIRE(mag > 1.0);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kFusedNormRope, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dx(dev, q, kTokens * (kOff + kRot)), dw(dev, q, kOff),
          dc(dev, q, kMaxPos * kRot), dlat(dev, q, kTokens * kOff),
          dpe(dev, q, kTokens * kRot);
      dx.Upload(x0);
      dw.Upload(w0);
      dc.Upload(cache);
      // pe_out is written only for an IN-RANGE position, so pre-seed both
      // outputs with a value the kernel must overwrite. Zeros would let a
      // kernel that wrote nothing pass wherever the oracle happened to be small.
      dlat.Upload(std::vector<float>(kTokens * kOff, -7.5f));
      dpe.Upload(std::vector<float>(kTokens * kRot, -7.5f));
      void* dpos = dev.Alloc(kTokens * sizeof(int32_t));
      dev.Copy(q, dpos, pos.data(), kTokens * sizeof(int32_t));
      dev.Synchronize(q);

      Tensor tl = Tensor::Contiguous(dlat.ptr(), DType::kF32, d, {kTokens, kOff});
      Tensor tp = Tensor::Contiguous(dpe.ptr(), DType::kF32, d, {kTokens, kRot});
      Tensor tx = Tensor::Contiguous(dx.ptr(), DType::kF32, d, {kTokens, kOff + kRot});
      Tensor tw = Tensor::Contiguous(dw.ptr(), DType::kF32, d, {kOff});
      Tensor tpos = Tensor::Contiguous(dpos, DType::kI32, d, {kTokens});
      Tensor tc = Tensor::Contiguous(dc.ptr(), DType::kF32, d, {kMaxPos, kRot});
      vt::FusedNormRope(q, tl, tp, tx, tw, tpos, tc, na, ra);
      dev.Synchronize(q);

      CHECK(Nmse(ref_lat, dlat.Download()) <= kNmseTol);
      CHECK(Nmse(ref_pe, dpe.Download()) <= kNmseTol);

      dev.Free(dpos);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("AttnQkNormRopeGate matches the CPU oracle within NMSE <= 5e-4") {
  // Fused full-attention preamble: split q|gate + (gemma) qk-RMSNorm(Dh) +
  // partial NeoX RoPE-from-cache + gate passthrough. Padded qgate/kf token
  // strides; plain + gemma norm variants. All arithmetic: NMSE except the
  // gate passthrough (pure movement).
  const int64_t T = 4;
  // Real Qwen3.5-0.8B attention dims first: Dh=256, rot=64 (partial_rotary
  // 0.25), Hq=8, Hkv=2 — the config the model actually runs; the synthetic
  // 32/16 arm below does not exercise the 192-dim pass-through tail.
  {
    const int64_t HQr = 8, HKVr = 2, DHr = 256, ROTr = 64;
    const int64_t qgo = HQr * 2 * DHr + 7, kfo = HKVr * DHr + 5;
    const std::vector<float> qg = RandomVec(static_cast<size_t>(T * qgo), 981, -0.5f, 0.5f);
    const std::vector<float> kfv = RandomVec(static_cast<size_t>(T * kfo), 982, -0.5f, 0.5f);
    const std::vector<float> qnr = RandomVec(static_cast<size_t>(DHr), 983, 0.2f, 1.0f);
    const std::vector<float> knr = RandomVec(static_cast<size_t>(DHr), 984, 0.2f, 1.0f);
    const std::vector<float> csr = RandomVec(static_cast<size_t>(T * ROTr), 985, -1.0f, 1.0f);
    for (bool gemma : {false, true}) {
      CAPTURE(gemma);
      vt::RmsNormArgs na2; na2.eps = 1e-6f; na2.gemma = gemma;
      vt::RopeArgs ra2; ra2.rotary_dim = static_cast<int>(ROTr);
      std::vector<float> rq(static_cast<size_t>(T * HQr * DHr));
      std::vector<float> rk(static_cast<size_t>(T * HKVr * DHr));
      std::vector<float> rg(static_cast<size_t>(T * HQr * DHr));
      {
        vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
        Queue cq = cpu.CreateQueue();
        const Device cd{DeviceType::kCPU, 0};
        std::vector<float> a = qg, b = kfv, e = qnr, f = knr, g = csr;
        Tensor tqg = Tensor::Contiguous(a.data(), DType::kF32, cd, {T, HQr * 2 * DHr});
        tqg.stride[0] = qgo;
        Tensor tkf = Tensor::Contiguous(b.data(), DType::kF32, cd, {T, HKVr * DHr});
        tkf.stride[0] = kfo;
        Tensor tqn = T1(e.data(), cd, DHr);
        Tensor tkn = T1(f.data(), cd, DHr);
        Tensor tcs = T2(g.data(), cd, T, ROTr);
        Tensor tqo = Tensor::Contiguous(rq.data(), DType::kF32, cd, {T, HQr, DHr});
        Tensor tko = Tensor::Contiguous(rk.data(), DType::kF32, cd, {T, HKVr, DHr});
        Tensor tgo = Tensor::Contiguous(rg.data(), DType::kF32, cd, {T, HQr, DHr});
        vt::AttnQkNormRopeGate(cq, tqo, tko, tgo, tqg, tkf, tqn, tkn, tcs, na2, ra2);
        cpu.DestroyQueue(cq);
      }
      for (DeviceType dt : RegisteredDevices()) {
        if (!OpAvailable(vt::OpId::kAttnQkNormRopeGate, dt)) continue;
        CAPTURE(DeviceName(dt));
        vt::Backend& dev = vt::GetBackend(dt);
        Queue q = dev.CreateQueue();
        const Device d{dt, 0};
        DevBuf dqg(dev, q, qg.size()), dkf(dev, q, kfv.size()), dqn(dev, q, DHr),
            dkn(dev, q, DHr), dcs(dev, q, csr.size()), dqo(dev, q, rq.size()),
            dko(dev, q, rk.size()), dgo(dev, q, rg.size());
        dqg.Upload(qg); dkf.Upload(kfv); dqn.Upload(qnr); dkn.Upload(knr); dcs.Upload(csr);
        Tensor tqg = Tensor::Contiguous(dqg.ptr(), DType::kF32, d, {T, HQr * 2 * DHr});
        tqg.stride[0] = qgo;
        Tensor tkf = Tensor::Contiguous(dkf.ptr(), DType::kF32, d, {T, HKVr * DHr});
        tkf.stride[0] = kfo;
        Tensor tqn = T1(dqn.ptr(), d, DHr);
        Tensor tkn = T1(dkn.ptr(), d, DHr);
        Tensor tcs = T2(dcs.ptr(), d, T, ROTr);
        Tensor tqo = Tensor::Contiguous(dqo.ptr(), DType::kF32, d, {T, HQr, DHr});
        Tensor tko = Tensor::Contiguous(dko.ptr(), DType::kF32, d, {T, HKVr, DHr});
        Tensor tgo = Tensor::Contiguous(dgo.ptr(), DType::kF32, d, {T, HQr, DHr});
        vt::AttnQkNormRopeGate(q, tqo, tko, tgo, tqg, tkf, tqn, tkn, tcs, na2, ra2);
        CHECK(Nmse(rq, dqo.Download()) <= kNmseTol);
        CHECK(Nmse(rk, dko.Download()) <= kNmseTol);
        CHECK(Nmse(rg, dgo.Download()) <= kNmseTol);
        dev.DestroyQueue(q);
      }
    }
  }

  // The in-context production mix (issue #41 M4 W2): the 0.8B bf16 model feeds
  // the preamble a BF16 projection output but wants F32 q/k/gate out (the f32
  // attention path — FA-2 is off on ROCm). The ROCm dispatcher once keyed on
  // the SOURCE dtype and mis-launched all-bf16, writing bf16 bits through the
  // f32 out pointers; this arm pins the (bf16 src -> f32 out) combo at the real
  // 0.8B dims so the bug class cannot return silently.
  {
    const int64_t HQr = 8, HKVr = 2, DHr = 256, ROTr = 64;
    const std::vector<float> qg = RandomVec(static_cast<size_t>(T * HQr * 2 * DHr), 991, -0.5f, 0.5f);
    const std::vector<float> kfv = RandomVec(static_cast<size_t>(T * HKVr * DHr), 992, -0.5f, 0.5f);
    const std::vector<float> qnr = RandomVec(static_cast<size_t>(DHr), 993, 0.2f, 1.0f);
    const std::vector<float> knr = RandomVec(static_cast<size_t>(DHr), 994, 0.2f, 1.0f);
    const std::vector<float> csr = RandomVec(static_cast<size_t>(T * ROTr), 995, -1.0f, 1.0f);
    const std::vector<uint16_t> qg_bf = Bf16Bits(qg), kf_bf = Bf16Bits(kfv);
    vt::RmsNormArgs na3; na3.eps = 1e-6f; na3.gemma = true;
    vt::RopeArgs ra3; ra3.rotary_dim = static_cast<int>(ROTr);
    // CPU reference: bf16 in (exact upcast inside the op) -> f32 out.
    std::vector<float> rq(static_cast<size_t>(T * HQr * DHr));
    std::vector<float> rk(static_cast<size_t>(T * HKVr * DHr));
    std::vector<float> rg(static_cast<size_t>(T * HQr * DHr));
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<uint16_t> a = qg_bf, b = kf_bf; std::vector<float> e = qnr, f = knr, g = csr;
      Tensor tqg = Tensor::Contiguous(a.data(), DType::kBF16, cd, {T, HQr * 2 * DHr});
      Tensor tkf = Tensor::Contiguous(b.data(), DType::kBF16, cd, {T, HKVr * DHr});
      Tensor tqn = T1(e.data(), cd, DHr), tkn = T1(f.data(), cd, DHr);
      Tensor tcs = T2(g.data(), cd, T, ROTr);
      Tensor tqo = Tensor::Contiguous(rq.data(), DType::kF32, cd, {T, HQr, DHr});
      Tensor tko = Tensor::Contiguous(rk.data(), DType::kF32, cd, {T, HKVr, DHr});
      Tensor tgo = Tensor::Contiguous(rg.data(), DType::kF32, cd, {T, HQr, DHr});
      vt::AttnQkNormRopeGate(cq, tqo, tko, tgo, tqg, tkf, tqn, tkn, tcs, na3, ra3);
      cpu.DestroyQueue(cq);
    }
    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kAttnQkNormRopeGate, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBufBytes dqg(dev, q, qg_bf.size() * 2), dkf(dev, q, kf_bf.size() * 2);
      DevBuf dqn(dev, q, DHr), dkn(dev, q, DHr), dcs(dev, q, csr.size());
      DevBuf dqo(dev, q, rq.size()), dko(dev, q, rk.size()), dgo(dev, q, rg.size());
      dqg.Upload(qg_bf.data()); dkf.Upload(kf_bf.data());
      dqn.Upload(qnr); dkn.Upload(knr); dcs.Upload(csr);
      Tensor tqg = Tensor::Contiguous(dqg.ptr(), DType::kBF16, d, {T, HQr * 2 * DHr});
      Tensor tkf = Tensor::Contiguous(dkf.ptr(), DType::kBF16, d, {T, HKVr * DHr});
      Tensor tqn = T1(dqn.ptr(), d, DHr), tkn = T1(dkn.ptr(), d, DHr);
      Tensor tcs = T2(dcs.ptr(), d, T, ROTr);
      Tensor tqo = Tensor::Contiguous(dqo.ptr(), DType::kF32, d, {T, HQr, DHr});
      Tensor tko = Tensor::Contiguous(dko.ptr(), DType::kF32, d, {T, HKVr, DHr});
      Tensor tgo = Tensor::Contiguous(dgo.ptr(), DType::kF32, d, {T, HQr, DHr});
      vt::AttnQkNormRopeGate(q, tqo, tko, tgo, tqg, tkf, tqn, tkn, tcs, na3, ra3);
      CHECK(Nmse(rq, dqo.Download()) <= kNmseTol);
      CHECK(Nmse(rk, dko.Download()) <= kNmseTol);
      CHECK(Nmse(rg, dgo.Download()) <= kNmseTol);
      dev.DestroyQueue(q);
    }
  }

  const int64_t HQ = 3, HKV = 2, DH = 32, ROT = 16;
  const int64_t qg_outer = HQ * 2 * DH + 7, kf_outer = HKV * DH + 5;
  const std::vector<float> qgate = RandomVec(static_cast<size_t>(T * qg_outer), 881, -0.5f, 0.5f);
  const std::vector<float> kf = RandomVec(static_cast<size_t>(T * kf_outer), 882, -0.5f, 0.5f);
  const std::vector<float> qn = RandomVec(static_cast<size_t>(DH), 883, 0.2f, 1.0f);
  const std::vector<float> kn = RandomVec(static_cast<size_t>(DH), 884, 0.2f, 1.0f);
  const std::vector<float> cs = RandomVec(static_cast<size_t>(T * ROT), 885, -1.0f, 1.0f);
  for (bool gemma : {false, true}) {
    CAPTURE(gemma);
    vt::RmsNormArgs na;
    na.eps = 1e-6f;
    na.gemma = gemma;
    vt::RopeArgs ra;
    ra.rotary_dim = ROT;
    std::vector<float> ref_qo(static_cast<size_t>(T * HQ * DH)),
        ref_ko(static_cast<size_t>(T * HKV * DH)), ref_go(static_cast<size_t>(T * HQ * DH));
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> cqg = qgate, ckf = kf, cqn = qn, ckn = kn, ccs = cs;
      Tensor tqg = Tensor::Contiguous(cqg.data(), DType::kF32, cd, {T, HQ * 2 * DH});
      tqg.stride[0] = qg_outer;  // padded token rows (merged-projection view)
      Tensor tkf = Tensor::Contiguous(ckf.data(), DType::kF32, cd, {T, HKV * DH});
      tkf.stride[0] = kf_outer;
      Tensor tqn = T1(cqn.data(), cd, DH);
      Tensor tkn = T1(ckn.data(), cd, DH);
      Tensor tcs = T2(ccs.data(), cd, T, ROT);
      Tensor tqo = Tensor::Contiguous(ref_qo.data(), DType::kF32, cd, {T, HQ, DH});
      Tensor tko = Tensor::Contiguous(ref_ko.data(), DType::kF32, cd, {T, HKV, DH});
      Tensor tgo = Tensor::Contiguous(ref_go.data(), DType::kF32, cd, {T, HQ, DH});
      vt::AttnQkNormRopeGate(cq, tqo, tko, tgo, tqg, tkf, tqn, tkn, tcs, na, ra);
      cpu.DestroyQueue(cq);
    }
    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kAttnQkNormRopeGate, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf dqg(dev, q, qgate.size()), dkf(dev, q, kf.size()), dqn(dev, q, DH),
          dkn(dev, q, DH), dcs(dev, q, cs.size()), dqo(dev, q, ref_qo.size()),
          dko(dev, q, ref_ko.size()), dgo(dev, q, ref_go.size());
      dqg.Upload(qgate);
      dkf.Upload(kf);
      dqn.Upload(qn);
      dkn.Upload(kn);
      dcs.Upload(cs);
      Tensor tqg = Tensor::Contiguous(dqg.ptr(), DType::kF32, d, {T, HQ * 2 * DH});
      tqg.stride[0] = qg_outer;
      Tensor tkf = Tensor::Contiguous(dkf.ptr(), DType::kF32, d, {T, HKV * DH});
      tkf.stride[0] = kf_outer;
      Tensor tqn = T1(dqn.ptr(), d, DH);
      Tensor tkn = T1(dkn.ptr(), d, DH);
      Tensor tcs = T2(dcs.ptr(), d, T, ROT);
      Tensor tqo = Tensor::Contiguous(dqo.ptr(), DType::kF32, d, {T, HQ, DH});
      Tensor tko = Tensor::Contiguous(dko.ptr(), DType::kF32, d, {T, HKV, DH});
      Tensor tgo = Tensor::Contiguous(dgo.ptr(), DType::kF32, d, {T, HQ, DH});
      vt::AttnQkNormRopeGate(q, tqo, tko, tgo, tqg, tkf, tqn, tkn, tcs, na, ra);
      CHECK(Nmse(ref_qo, dqo.Download()) <= kNmseTol);
      CHECK(Nmse(ref_ko, dko.Download()) <= kNmseTol);
      CHECK(Nmse(ref_go, dgo.Download()) <= kNmseTol);  // gate passthrough: exact movement
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("MoeSiluMul matches the CPU oracle within NMSE <= 5e-4") {
  // Elementwise silu(gate)*up (the MoE-path activation). f32 and bf16 arms.
  const size_t n = 1024;
  const std::vector<float> gate = RandomVec(n, 901);
  const std::vector<float> up = RandomVec(n, 902);
  const std::vector<uint16_t> gate_bf = Bf16Bits(gate);
  const std::vector<uint16_t> up_bf = Bf16Bits(up);
  for (bool bf16 : {false, true}) {
    CAPTURE(bf16);
    std::vector<float> ref_f(n, 0.0f);
    std::vector<uint16_t> ref_b(n, 0);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> cg = gate, cu = up;
      std::vector<uint16_t> cgb = gate_bf, cub = up_bf;
      if (bf16) {
        Tensor tg = Tensor::Contiguous(cgb.data(), DType::kBF16, cd, {static_cast<int64_t>(n)});
        Tensor tu = Tensor::Contiguous(cub.data(), DType::kBF16, cd, {static_cast<int64_t>(n)});
        Tensor tout = Tensor::Contiguous(ref_b.data(), DType::kBF16, cd, {static_cast<int64_t>(n)});
        vt::MoeSiluMul(cq, tout, tg, tu);
      } else {
        Tensor tg = T1(cg.data(), cd, static_cast<int64_t>(n));
        Tensor tu = T1(cu.data(), cd, static_cast<int64_t>(n));
        Tensor tout = T1(ref_f.data(), cd, static_cast<int64_t>(n));
        vt::MoeSiluMul(cq, tout, tg, tu);
      }
      cpu.DestroyQueue(cq);
    }
    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kMoeSiluMul, dt)) continue;
      // Wrapped for the same reason as the K-quant case below. This case is
      // the standing red that issue #1954 tracks on gfx1200, and a bare
      // CAPTURE logged its device as `1`, which is the one fact a reader
      // needs from that log.
      CAPTURE(std::string(DeviceName(dt)));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf dg(dev, q, n), du(dev, q, n), dout(dev, q, n);
      DevBufBytes dgb(dev, q, n * 2), dub(dev, q, n * 2), doutb(dev, q, n * 2);
      if (bf16) {
        dgb.Upload(gate_bf.data());
        dub.Upload(up_bf.data());
        Tensor tg = Tensor::Contiguous(dgb.ptr(), DType::kBF16, d, {static_cast<int64_t>(n)});
        Tensor tu = Tensor::Contiguous(dub.ptr(), DType::kBF16, d, {static_cast<int64_t>(n)});
        Tensor tout = Tensor::Contiguous(doutb.ptr(), DType::kBF16, d, {static_cast<int64_t>(n)});
        vt::MoeSiluMul(q, tout, tg, tu);
        std::vector<uint16_t> got(n);
        doutb.Download(got.data());
        CHECK(got == ref_b);  // single multiply + RNE store both sides: exact
      } else {
        dg.Upload(gate);
        du.Upload(up);
        Tensor tg = T1(dg.ptr(), d, static_cast<int64_t>(n));
        Tensor tu = T1(du.ptr(), d, static_cast<int64_t>(n));
        Tensor tout = T1(dout.ptr(), d, static_cast<int64_t>(n));
        vt::MoeSiluMul(q, tout, tg, tu);
        CHECK(Nmse(ref_f, dout.Download()) <= kNmseTol);
      }
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("MoeRouterTopK matches the CPU oracle (f32 and bf16 logits)") {
  // Ungrouped softmax top-k. Weights are arithmetic (softmax + renorm): NMSE.
  // The selected INDICES are discrete outputs: exact match required (the
  // tie-break is lowest-index on both sides). The bf16-logits arm upcasts at
  // the boundary; the softmax stays f32 on both sides.
  const int64_t T = 6, E = 48, K = 5;
  const size_t ln = static_cast<size_t>(T * E);
  const std::vector<float> logits = RandomVec(ln, 891);
  const std::vector<uint16_t> logits_bf = Bf16Bits(logits);
  for (bool bf16 : {false, true}) {
    for (bool renorm : {false, true}) {
      CAPTURE(bf16);
      CAPTURE(renorm);
      vt::MoeRouterTopKArgs args;
      args.top_k = static_cast<int>(K);
      args.renormalize = renorm;
      std::vector<float> ref_w(static_cast<size_t>(T * K), 0.0f);
      std::vector<int32_t> ref_i(static_cast<size_t>(T * K), -1);
      {
        vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
        Queue cq = cpu.CreateQueue();
        const Device cd{DeviceType::kCPU, 0};
        std::vector<float> cl = logits;
        std::vector<uint16_t> clb = logits_bf;
        Tensor tl = bf16 ? Tensor::Contiguous(clb.data(), DType::kBF16, cd, {T, E})
                         : Tensor::Contiguous(cl.data(), DType::kF32, cd, {T, E});
        Tensor tw = Tensor::Contiguous(ref_w.data(), DType::kF32, cd, {T, K});
        Tensor ti = Tensor::Contiguous(ref_i.data(), DType::kI32, cd, {T, K});
        vt::MoeRouterTopK(cq, tw, ti, tl, args, nullptr);
        cpu.DestroyQueue(cq);
      }
      for (DeviceType dt : RegisteredDevices()) {
        if (!OpAvailable(vt::OpId::kMoeRouterTopK, dt)) continue;
        CAPTURE(DeviceName(dt));
        vt::Backend& dev = vt::GetBackend(dt);
        Queue q = dev.CreateQueue();
        const Device d{dt, 0};
        DevBuf dl(dev, q, ln), dw(dev, q, static_cast<size_t>(T * K));
        DevBufBytes dlb(dev, q, ln * 2);
        DevBufI32 di(dev, q, static_cast<size_t>(T * K));
        if (bf16) dlb.Upload(logits_bf.data());
        else dl.Upload(logits);
        Tensor tl = bf16 ? Tensor::Contiguous(dlb.ptr(), DType::kBF16, d, {T, E})
                         : Tensor::Contiguous(dl.ptr(), DType::kF32, d, {T, E});
        Tensor tw = Tensor::Contiguous(dw.ptr(), DType::kF32, d, {T, K});
        Tensor ti = Tensor::Contiguous(di.ptr(), DType::kI32, d, {T, K});
        vt::MoeRouterTopK(q, tw, ti, tl, args, nullptr);
        CHECK(Nmse(ref_w, dw.Download()) <= kNmseTol);
        std::vector<int32_t> got_i(static_cast<size_t>(T * K));
        dev.Synchronize(q);
        dev.Copy(q, got_i.data(), di.ptr(), got_i.size() * sizeof(int32_t));
        dev.Synchronize(q);
        CHECK(got_i == ref_i);  // selected experts: exact
        dev.DestroyQueue(q);
      }
    }
  }
}

TEST_CASE("decode-skinny MatmulBT (wvSplitK path) matches the CPU oracle") {
  // Port of upstream's tests/kernels/quantization/test_rocm_skinny_gemms.py
  // ::test_rocm_wvsplitk_kernel @ pin 55596792 (review sweep on #506: the first
  // version of this case had aggregate-NMSE tolerance that ten completely
  // wrong elements would still pass, every K a multiple of the 512 stride so
  // the K-tail path never ran, and no guard-boundary shapes at all).
  //
  // Preserved from upstream: the NKM factor list (the applicable subset — see
  // below), the xavier on/off scaling, and the ELEMENTWISE tolerance
  // atol = eps_bf16 * sqrt(K), rtol = 1e-2 (torch.testing.assert_close
  // semantics). Deferred with reason: fp16 (our port is bf16-only), bias
  // (the vt::MatmulBT seam has no bias operand), padded strides (our dispatch
  // requires contiguous rows — a documented precondition), and the fp8/rc
  // kernel variants (not ported). The (n,k,m) upstream triple = (tokens, K,
  // features) here.
  struct Shape { int64_t tok, k, feat; const char* why; };
  const Shape shapes[] = {
      // the upstream sweep (token counts 1-4 = our template arms)
      {1, 32, 16, "upstream"}, {1, 64, 64, "upstream"}, {2, 256, 256, "upstream"},
      {3, 1024, 1024, "upstream"}, {4, 4096, 4096, "upstream"},
      // K-tail: K % 512 != 0 exercises the `if (k_ >= K) break` remainder path
      {4, 4096 + 16, 4096, "k-tail"}, {1, 9216, 512, "upstream"},
      // guard boundaries (must stay CORRECT via the BLAS fallback)
      {2, 256, 8, "features<=8 declines (upstream m>8)"},
      {2, 256, 254, "even below bound: takes skinny"},
      {2, 256, 255, "odd features decline (YTILE=2 OOB class)"},
      {2, 254, 256, "K%8!=0 declines"},
  };
  const double kEpsBf16 = 0.0078125;  // 2^-8
  for (const Shape& sh : shapes) {
    for (bool xnorm : {false, true}) {
      CAPTURE(sh.why);
      CAPTURE(sh.tok);
      CAPTURE(sh.k);
      CAPTURE(sh.feat);
      CAPTURE(xnorm);
      const int64_t M = sh.tok, N = sh.feat, K = sh.k;
      const size_t an = static_cast<size_t>(M) * K, bn = static_cast<size_t>(N) * K;
      const double xavier = xnorm ? std::sqrt(2.0 / static_cast<double>(K)) : 1.0;
      std::vector<float> a = RandomVec(an, 991, -1.0f, 1.0f);
      std::vector<float> b = RandomVec(bn, 992, -1.0f, 1.0f);
      for (float& x : a) x = static_cast<float>(x * xavier);
      for (float& x : b) x = static_cast<float>(x * xavier);
      const std::vector<uint16_t> a_bf = Bf16Bits(a), b_bf = Bf16Bits(b);

      std::vector<uint16_t> ref(static_cast<size_t>(M) * N, 0);
      {
        vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
        Queue cq = cpu.CreateQueue();
        const Device cd{DeviceType::kCPU, 0};
        std::vector<uint16_t> ca = a_bf, cb = b_bf;
        Tensor ta = Tensor::Contiguous(ca.data(), DType::kBF16, cd, {M, K});
        Tensor tb = Tensor::Contiguous(cb.data(), DType::kBF16, cd, {N, K});
        Tensor to = Tensor::Contiguous(ref.data(), DType::kBF16, cd, {M, N});
        vt::MatmulBT(cq, to, ta, tb);
        cpu.DestroyQueue(cq);
      }
      for (DeviceType dt : RegisteredDevices()) {
        if (!OpAvailable(vt::OpId::kMatmulBT, dt)) continue;
        CAPTURE(DeviceName(dt));
        vt::Backend& dev = vt::GetBackend(dt);
        Queue q = dev.CreateQueue();
        const Device d{dt, 0};
        // Sentinel-padded output: the dispatch must never write past M*N
        // elements (the odd-features OOB class from the review).
        const size_t out_elems = static_cast<size_t>(M) * N;
        const size_t guard_elems = 128;
        DevBufBytes da(dev, q, an * 2), db(dev, q, bn * 2),
            dout(dev, q, (out_elems + guard_elems) * 2);
        std::vector<uint16_t> fill(out_elems + guard_elems, 0xDEAD);
        dout.Upload(fill.data());
        da.Upload(a_bf.data());
        db.Upload(b_bf.data());
        Tensor ta = Tensor::Contiguous(da.ptr(), DType::kBF16, d, {M, K});
        Tensor tb = Tensor::Contiguous(db.ptr(), DType::kBF16, d, {N, K});
        Tensor to = Tensor::Contiguous(dout.ptr(), DType::kBF16, d, {M, N});
        vt::MatmulBT(q, to, ta, tb);
        std::vector<uint16_t> got(out_elems + guard_elems);
        dout.Download(got.data());
        // Elementwise tolerance (upstream assert_close), never aggregate NMSE.
        const double atol = kEpsBf16 * std::sqrt(static_cast<double>(K));
        for (size_t i = 0; i < out_elems; ++i) {
          uint32_t ug = static_cast<uint32_t>(got[i]) << 16, ur = static_cast<uint32_t>(ref[i]) << 16;
          float gf, rf;
          std::memcpy(&gf, &ug, 4);
          std::memcpy(&rf, &ur, 4);
          CHECK(std::fabs(gf - rf) <= atol + 1e-2 * std::fabs(rf));
        }
        // The guard band must be untouched by ANY path (skinny or BLAS).
        for (size_t i = out_elems; i < out_elems + guard_elems; ++i)
          CHECK(got[i] == 0xDEAD);
        dev.DestroyQueue(q);
      }
    }
  }
}

TEST_CASE("non-grouped keep-quant GEMM (Q8_0/Q4_K/Q5_K/Q6_K) matches the CPU oracle") {
  // kMatmulBTQuant (op 74) on ROCm vs the CPU keep-quant reference. The
  // non-grouped arm carries PR #523's headline mechanism and had NO coverage
  // (review sweep 2026-08-13); the ROCm dispatcher's src-vs-out dtype mix-up
  // in the fused preamble (the 0.8B divergence, row/ROCM-GDN-08B-FIX) is
  // exactly the class an untested-but-registered op hides. REQUIRE (not skip)
  // on ROCm so a dropped RegisterOp can never pass silently.
  constexpr int64_t M = 3, N = 8, K = 512;
  struct Fmt { vt::DType dt; int64_t block_bytes; int d_off; int dmin_off; const char* name; };
  const Fmt fmts[] = {
    {vt::DType::kQ8_0, 34, 0, -1, "q8_0"},
    {vt::DType::kQ4_K, 144, 0, 2, "q4_K"},
    {vt::DType::kQ6_K, 210, 208, -1, "q6_K"},
    {vt::DType::kQ5_K, 176, 0, 2, "q5_K"},
  };
  const bool rocm_present = OpAvailable(vt::OpId::kMatmulBTQuant, DeviceType::kROCM);
  const bool any_rocm = [&] {
    for (DeviceType dt : RegisteredDevices()) if (dt == DeviceType::kROCM) return true;
    return false;
  }();
  if (any_rocm) {
    REQUIRE_MESSAGE(rocm_present,
                    "kMatmulBTQuant must be registered on ROCm (the keep-quant "
                    "loader flips on it) — a missing registration is a failure, "
                    "never a skip");
  }
  for (const Fmt& f : fmts) {
    CAPTURE(f.name);
    const int64_t elems_per_block = (f.dt == vt::DType::kQ8_0) ? 32 : 256;
    const int64_t blocks_per_row = K / elems_per_block;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * f.block_bytes;
    const size_t wn = static_cast<size_t>(N) * row_bytes;
    std::mt19937 rng(779);
    std::vector<uint8_t> wt(wn);
    for (uint8_t& b : wt) b = static_cast<uint8_t>(rng() & 0xFF);
    for (int64_t r = 0; r < N; ++r)
      for (int64_t bIdx = 0; bIdx < blocks_per_row; ++bIdx) {
        uint8_t* blk = wt.data() + r * row_bytes + bIdx * f.block_bytes;
        const float jitter = 1.0f + 0.05f * static_cast<float>((r + bIdx) % 7);
        auto put16 = [&](int off, float v) { uint16_t h = vt::F32ToF16(v); std::memcpy(blk + off, &h, 2); };
        if (f.d_off >= 0) put16(f.d_off, 0.0125f * jitter);
        if (f.dmin_off >= 0) put16(f.dmin_off, 0.0075f * jitter);
      }
    const size_t an = static_cast<size_t>(M) * K, on = static_cast<size_t>(M) * N;
    const std::vector<float> act = RandomVec(an, 780, -0.5f, 0.5f);
    std::vector<float> ref(on, 0.0f);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> ca = act;
      std::vector<uint8_t> cw = wt;
      Tensor tout = T2(ref.data(), cd, M, N);
      Tensor tact = T2(ca.data(), cd, M, K);
      Tensor twt = Tensor::Contiguous(cw.data(), f.dt, cd, {N, K});
      vt::MatmulBTQuant(cq, tout, tact, twt);
      cpu.DestroyQueue(cq);
    }
    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kMatmulBTQuant, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf da(dev, q, an);
      DevBufBytes dwt(dev, q, wn);
      DevBuf dout(dev, q, on);
      da.Upload(act);
      dwt.Upload(wt.data());
      Tensor tact = T2(da.ptr(), d, M, K);
      Tensor twt = Tensor::Contiguous(dwt.ptr(), f.dt, d, {N, K});
      Tensor tout = T2(dout.ptr(), d, M, N);
      vt::MatmulBTQuant(q, tout, tact, twt);
      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);
      dev.DestroyQueue(q);
    }
  }
}

// Issue #2511: the keep-quant Q6_K arm at the launch geometry PRODUCTION uses.
//
// The gate above runs kMatmulBTQuant at M=3, N=8, K=512 -- nsb = 2 and a grid
// of ceil(3*8/4) = 6 blocks. Every failing leg on `strix:gpu0` (gfx1151, ROCm
// 7.2.4) died inside `KQuantGemmK<unsigned short, 2>` launched at m = 5,
// n = 5120, nsb = 68: 6400 blocks and 34x the superblocks, with a bf16 output
// (`OutT == uint16_t`), which is a DIFFERENT template instantiation from the f32
// one the case above exercises. That gate therefore tests the kernel's
// ARITHMETIC and has never tested its LAUNCH, which is why a green ROCm suite
// coexisted with a board that page-faults on one prefill forward pass.
//
// This case closes exactly that gap. It is deliberately NOT a second arithmetic
// check: the oracle runs on a 64-column SLICE of the same weight buffer (output
// column j reads only weight row j, so the slice is exact, not an
// approximation), and the launch itself runs at the full production width,
// repeatedly, because the defect it guards is intermittent.
TEST_CASE("keep-quant Q6_K GEMM runs at the production launch geometry") {
  // m/n/nsb read off the AMD_LOG_LEVEL=4 dump in #2511: grid {6400,1,1},
  // block {32,4,1}, nsb = 68, w_row_bytes = 14280, weight obj = 73,113,600 B.
  constexpr int64_t M = 5, N = 5120, K = 17408;
  constexpr int64_t kRefCols = 64;
  constexpr int64_t kBlockBytes = 210;  // sizeof(BlockQ6_K)
  constexpr int kDOff = 208;            // the superblock scale's byte offset
  // The fault is intermittent (6 of 6 legs at one shape, 6 of 8 at another), so
  // one launch is not a measurement. VT_TEST_Q6K_LAUNCHES raises it for a
  // deliberate soak without editing this file.
  const int64_t launches = [] {
    const char* e = std::getenv("VT_TEST_Q6K_LAUNCHES");
    const long v = e != nullptr ? std::strtol(e, nullptr, 10) : 0;
    return v > 0 ? static_cast<int64_t>(v) : static_cast<int64_t>(8);
  }();
  CAPTURE(launches);

  const int64_t blocks_per_row = K / 256;
  const size_t row_bytes = static_cast<size_t>(blocks_per_row) * kBlockBytes;
  const size_t wn = static_cast<size_t>(N) * row_bytes;
  REQUIRE(blocks_per_row == 68);
  REQUIRE(row_bytes == 14280u);
  REQUIRE(wn == 73113600u);

  std::mt19937 rng(2511);
  std::vector<uint8_t> wt(wn);
  for (uint8_t& b : wt) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t r = 0; r < N; ++r)
    for (int64_t bIdx = 0; bIdx < blocks_per_row; ++bIdx) {
      uint8_t* blk = wt.data() + r * row_bytes + bIdx * kBlockBytes;
      const float jitter = 1.0f + 0.05f * static_cast<float>((r + bIdx) % 7);
      const uint16_t h = vt::F32ToF16(0.0125f * jitter);
      std::memcpy(blk + kDOff, &h, 2);
    }

  const size_t an = static_cast<size_t>(M) * K;
  const std::vector<float> act = RandomVec(an, 2512, -0.5f, 0.5f);

  // Oracle on the first kRefCols columns only. Output column j depends solely on
  // weight row j, so this is the exact answer for those columns rather than a
  // sampled one -- and it costs 1/80th of the full-width CPU reference.
  std::vector<float> ref(static_cast<size_t>(M) * kRefCols, 0.0f);
  {
    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> ca = act;
    std::vector<uint8_t> cw(wt.begin(), wt.begin() + static_cast<size_t>(kRefCols) * row_bytes);
    Tensor tout = T2(ref.data(), cd, M, kRefCols);
    Tensor tact = T2(ca.data(), cd, M, K);
    Tensor twt = Tensor::Contiguous(cw.data(), vt::DType::kQ6_K, cd, {kRefCols, K});
    vt::MatmulBTQuant(cq, tout, tact, twt);
    cpu.DestroyQueue(cq);
  }

  const bool any_rocm = [&] {
    for (DeviceType dt : RegisteredDevices()) if (dt == DeviceType::kROCM) return true;
    return false;
  }();
  if (any_rocm) {
    REQUIRE_MESSAGE(OpAvailable(vt::OpId::kMatmulBTQuant, DeviceType::kROCM),
                    "kMatmulBTQuant must be registered on ROCm");
  }

  for (DeviceType dt : RegisteredDevices()) {
    // The CPU arm is skipped on purpose: comparing the reference against itself
    // proves nothing about a LAUNCH, and a full-width CPU pass is 445M scalar
    // MACs on every CI run for that non-result.
    if (dt == DeviceType::kCPU) continue;
    if (!OpAvailable(vt::OpId::kMatmulBTQuant, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};
    DevBuf da(dev, q, an);
    DevBufBytes dwt(dev, q, wn);
    // bf16 out, because that is the OutT == uint16_t instantiation the model
    // path launches and the one every #2511 leg died inside.
    DevBufBytes dout(dev, q, static_cast<size_t>(M) * N * sizeof(uint16_t));
    da.Upload(act);
    dwt.Upload(wt.data());
    Tensor tact = T2(da.ptr(), d, M, K);
    Tensor twt = Tensor::Contiguous(dwt.ptr(), vt::DType::kQ6_K, d, {N, K});
    Tensor tout = Tensor::Contiguous(dout.ptr(), DType::kBF16, d, {M, N});

    std::vector<uint16_t> got(static_cast<size_t>(M) * N, 0);
    std::vector<uint16_t> firstrun;
    for (int64_t it = 0; it < launches; ++it) {
      CAPTURE(it);
      vt::MatmulBTQuant(q, tout, tact, twt);
      dev.Synchronize(q);
      dout.Download(got.data());
      if (it == 0) {
        firstrun = got;
        // A launch that wrote nothing would otherwise pass every check below.
        int64_t nonzero = 0;
        for (uint16_t v : got) nonzero += (v != 0) ? 1 : 0;
        CHECK(nonzero > 0);
        std::vector<float> gotf(static_cast<size_t>(M) * kRefCols, 0.0f);
        for (int64_t i = 0; i < M; ++i)
          for (int64_t j = 0; j < kRefCols; ++j)
            gotf[i * kRefCols + j] = vt::BF16ToF32(got[i * N + j]);
        CHECK(Nmse(ref, gotf) <= kNmseTol);
      } else {
        // Same inputs, same kernel: the bytes must repeat exactly. A drift here
        // is the silent half of this defect -- a launch that did not fault but
        // read state it did not own.
        CHECK(std::memcmp(firstrun.data(), got.data(),
                          got.size() * sizeof(uint16_t)) == 0);
      }
    }
    dev.DestroyQueue(q);
  }
}

#if defined(VLLM_CPP_HIP)
// Declared here rather than included: the ROCm kernels have no public header,
// and src/vt/rocm/rocm_ops.hip:65 already reaches MatmulBTQuantKernelRocm by a
// file-local extern declaration. This row mirrors that convention instead of
// inventing a header it would be the only user of.
namespace vt::rocm {
int KQuantDecodeCoopWarps(vt::DType wdt, int64_t m, int64_t nsb);
uint64_t KQuantCoopDispatchCount();
void Q8KQuantizeForTest(vt::Queue& q, void* scratch, const void* act, vt::DType dtype,
                        int64_t row_stride, int64_t rows, int64_t nsb, bool candidate);
bool Q8KCandidateSelectedForTest(const char* env_value, bool gfx1100_default_accepted,
                                 int device_index,
                                 std::string (*resolve)(int) noexcept);
void Q8KResetRouteDispatchCountsForTest();
uint64_t Q8KRouteDispatchCountForTest(bool grouped, bool candidate);
void* Q8KSetKernelExecutionWitnessForTest(void* device_counts);
}  // namespace vt::rocm

namespace {

constexpr size_t kQ8KBlockBytes = sizeof(vt::cpu::BlockQ8_K);
static_assert(kQ8KBlockBytes == 292);

std::string Q8KMatrixResolver(int device_index) noexcept {
  switch (device_index) {
    case 0: return "gfx1100";
    case 1: return "gfx1200";
    case 2: return "gfx1201";
    case 3: return "unknown";
    default: return {};
  }
}

std::string Q8KAlternateResolver(int device_index) noexcept {
  if (device_index == 0) return "gfx1200";
  if (device_index == 1) return "gfx1100";
  return {};
}

class ScopedQ8KEnv {
 public:
  explicit ScopedQ8KEnv(const char* value) {
    if (const char* old = std::getenv("VT_ROCM_Q8K_BLOCK")) {
      had_value_ = true;
      old_value_ = old;
    }
    if (value == nullptr) vllm_test::UnsetEnv("VT_ROCM_Q8K_BLOCK");
    else vllm_test::SetEnv("VT_ROCM_Q8K_BLOCK", value);
  }

  ~ScopedQ8KEnv() {
    if (had_value_) vllm_test::SetEnv("VT_ROCM_Q8K_BLOCK", old_value_);
    else vllm_test::UnsetEnv("VT_ROCM_Q8K_BLOCK");
  }

  ScopedQ8KEnv(const ScopedQ8KEnv&) = delete;
  ScopedQ8KEnv& operator=(const ScopedQ8KEnv&) = delete;

 private:
  bool had_value_ = false;
  std::string old_value_;
};

size_t Q8KSourceElementBytes(DType dtype) {
  return dtype == DType::kF32 ? sizeof(float) : sizeof(uint16_t);
}

const char* Q8KDTypeName(DType dtype) {
  if (dtype == DType::kF32) return "f32";
  if (dtype == DType::kF16) return "f16";
  return "bf16";
}

void StoreQ8KSource(std::vector<uint8_t>& bytes, DType dtype, size_t index, float value) {
  const size_t offset = index * Q8KSourceElementBytes(dtype);
  if (dtype == DType::kF32) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
  } else {
    const uint16_t bits = dtype == DType::kF16 ? vt::F32ToF16(value) : vt::F32ToBF16(value);
    std::memcpy(bytes.data() + offset, &bits, sizeof(bits));
  }
}

float LoadQ8KSource(const std::vector<uint8_t>& bytes, DType dtype, size_t index) {
  const size_t offset = index * Q8KSourceElementBytes(dtype);
  if (dtype == DType::kF32) {
    float value = 0.0f;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
  }
  uint16_t bits = 0;
  std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
  return dtype == DType::kF16 ? vt::F16ToF32(bits) : vt::BF16ToF32(bits);
}

enum class Q8KInputKind {
  kRandom,
  kZero,
  kPositiveNegativeWaveTie,
  kNegativePositiveWaveTie,
  kPositiveNegativeReductionTie,
  kNegativePositiveReductionTie,
};

struct Q8KInputCase {
  Q8KInputKind kind;
  const char* name;
};

constexpr std::array<Q8KInputCase, 6> kQ8KInputCases{{
    {Q8KInputKind::kRandom, "random"},
    {Q8KInputKind::kZero, "all-zero-sentinel"},
    {Q8KInputKind::kPositiveNegativeWaveTie, "positive-negative-wave-tie"},
    {Q8KInputKind::kNegativePositiveWaveTie, "negative-positive-wave-tie"},
    {Q8KInputKind::kPositiveNegativeReductionTie, "positive-negative-reduction-tie"},
    {Q8KInputKind::kNegativePositiveReductionTie, "negative-positive-reduction-tie"},
}};

void FillQ8KInput(std::vector<uint8_t>& source, DType dtype, int64_t row_stride,
                  int64_t rows, int64_t nsb, Q8KInputKind kind) {
  std::mt19937 rng(1876u + static_cast<uint32_t>(nsb) * 17u +
                   static_cast<uint32_t>(kind) * 101u);
  std::uniform_real_distribution<float> random_value(-3.0f, 3.0f);
  const int64_t k = nsb * vt::cpu::kQK_K;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      float value = 0.0f;
      if (kind == Q8KInputKind::kRandom) {
        value = random_value(rng);
      } else if (kind != Q8KInputKind::kZero) {
        const int bounded = static_cast<int>((col * 29 + row * 11) % 101) - 50;
        value = static_cast<float>(bounded) / 64.0f;
      }
      StoreQ8KSource(source, dtype, static_cast<size_t>(row * row_stride + col), value);
    }
    if (kind == Q8KInputKind::kRandom || kind == Q8KInputKind::kZero) continue;
    for (int64_t sb = 0; sb < nsb; ++sb) {
      int first = 31;
      int second = 32;
      float first_value = 7.0f;
      float second_value = -7.0f;
      if (kind == Q8KInputKind::kNegativePositiveWaveTie) {
        first_value = -7.0f;
        second_value = 7.0f;
      } else if (kind == Q8KInputKind::kPositiveNegativeReductionTie) {
        first = 127;
        second = 128;
      } else if (kind == Q8KInputKind::kNegativePositiveReductionTie) {
        first = 127;
        second = 128;
        first_value = -7.0f;
        second_value = 7.0f;
      }
      const int64_t block_start = row * row_stride + sb * vt::cpu::kQK_K;
      StoreQ8KSource(source, dtype, static_cast<size_t>(block_start + first), first_value);
      StoreQ8KSource(source, dtype, static_cast<size_t>(block_start + second), second_value);
    }
  }
}

void CheckQ8KBlockBytes(const std::vector<uint8_t>& got,
                        const std::vector<uint8_t>& expected, const char* reference,
                        DType dtype, const char* input_class, int64_t row, int64_t sb,
                        int64_t nsb) {
  const size_t block = static_cast<size_t>(row * nsb + sb);
  const size_t offset = block * kQ8KBlockBytes;
  size_t mismatch = kQ8KBlockBytes;
  for (size_t byte = 0; byte < kQ8KBlockBytes; ++byte) {
    if (got[offset + byte] != expected[offset + byte]) {
      mismatch = byte;
      break;
    }
  }
  CAPTURE(std::string(Q8KDTypeName(dtype)));
  CAPTURE(std::string(input_class));
  CAPTURE(std::string(reference));
  CAPTURE(row);
  CAPTURE(sb);
  CAPTURE(mismatch);
  CHECK_MESSAGE(mismatch == kQ8KBlockBytes,
                "ROCm Q8_K byte mismatch: dtype/class/reference/row/sb/byte are captured");
}

void CheckOnlyQ8KRouteCounter(bool grouped, bool candidate) {
  for (bool observed_grouped : {false, true}) {
    for (bool observed_candidate : {false, true}) {
      const uint64_t count =
          vt::rocm::Q8KRouteDispatchCountForTest(observed_grouped, observed_candidate);
      CAPTURE(grouped);
      CAPTURE(candidate);
      CAPTURE(observed_grouped);
      CAPTURE(observed_candidate);
      CHECK(count == ((grouped == observed_grouped && candidate == observed_candidate) ? 1u
                                                                                       : 0u));
    }
  }
}

void CheckNoQ8KRouteCounters() {
  for (bool grouped : {false, true})
    for (bool candidate : {false, true})
      CHECK(vt::rocm::Q8KRouteDispatchCountForTest(grouped, candidate) == 0);
}

constexpr size_t kQ8KKernelWitnessArmCount = 2;
using Q8KKernelWitnessCounts = std::array<uint64_t, kQ8KKernelWitnessArmCount>;

class ScopedQ8KKernelExecutionWitness {
 public:
  explicit ScopedQ8KKernelExecutionWitness(void* device_counts)
      : previous_(vt::rocm::Q8KSetKernelExecutionWitnessForTest(device_counts)) {}

  ~ScopedQ8KKernelExecutionWitness() {
    vt::rocm::Q8KSetKernelExecutionWitnessForTest(previous_);
  }

  ScopedQ8KKernelExecutionWitness(const ScopedQ8KKernelExecutionWitness&) = delete;
  ScopedQ8KKernelExecutionWitness& operator=(const ScopedQ8KKernelExecutionWitness&) =
      delete;

 private:
  void* previous_ = nullptr;
};

void ResetQ8KKernelExecutionWitness(DevBufBytes& device_counts) {
  const Q8KKernelWitnessCounts zero{};
  device_counts.Upload(zero.data());
}

Q8KKernelWitnessCounts ReadQ8KKernelExecutionWitness(DevBufBytes& device_counts) {
  Q8KKernelWitnessCounts counts{};
  device_counts.Download(counts.data());
  return counts;
}

void CheckOnlyQ8KKernelExecutionWitness(DevBufBytes& device_counts, bool candidate) {
  const Q8KKernelWitnessCounts counts = ReadQ8KKernelExecutionWitness(device_counts);
  for (bool observed_candidate : {false, true}) {
    CAPTURE(candidate);
    CAPTURE(observed_candidate);
    CHECK(counts[observed_candidate ? 1 : 0] ==
          (candidate == observed_candidate ? 1u : 0u));
  }
}

void CheckNoQ8KKernelExecutionWitness(DevBufBytes& device_counts) {
  const Q8KKernelWitnessCounts counts = ReadQ8KKernelExecutionWitness(device_counts);
  for (uint64_t count : counts) CHECK(count == 0);
}

}  // namespace

TEST_CASE("ROCm Q8_K policy is resolver-keyed and architecture-scoped") {
  for (bool default_accepted : {false, true}) {
    for (int device = 0; device < 5; ++device) {
      CAPTURE(default_accepted);
      CAPTURE(device);
      const bool unset_expected = default_accepted && device == 0;
      CHECK(vt::rocm::Q8KCandidateSelectedForTest(
                nullptr, default_accepted, device, Q8KMatrixResolver) == unset_expected);
      CHECK_FALSE(vt::rocm::Q8KCandidateSelectedForTest(
          "0", default_accepted, device, Q8KMatrixResolver));
      CHECK(vt::rocm::Q8KCandidateSelectedForTest(
          "1", default_accepted, device, Q8KMatrixResolver));
      CHECK_THROWS_WITH_AS(
          vt::rocm::Q8KCandidateSelectedForTest(
              "invalid", default_accepted, device, Q8KMatrixResolver),
          doctest::Contains("VT_ROCM_Q8K_BLOCK=invalid"), std::runtime_error);
    }
  }

  // One resolver says device 0 is gfx1100 and device 1 is gfx1200. The next
  // resolver reverses those answers. A cache keyed only by device fails here.
  CHECK(vt::rocm::Q8KCandidateSelectedForTest(nullptr, true, 0, Q8KMatrixResolver));
  CHECK_FALSE(vt::rocm::Q8KCandidateSelectedForTest(nullptr, true, 1, Q8KMatrixResolver));
  CHECK_FALSE(vt::rocm::Q8KCandidateSelectedForTest(nullptr, true, 0,
                                                    Q8KAlternateResolver));
  CHECK(vt::rocm::Q8KCandidateSelectedForTest(nullptr, true, 1,
                                              Q8KAlternateResolver));
}

TEST_CASE("ROCm Q8_K cooperative quantizer matches both 292-byte oracles") {
  const bool rocm_registered = [] {
    for (DeviceType dtype : RegisteredDevices())
      if (dtype == DeviceType::kROCM) return true;
    return false;
  }();
  if (!rocm_registered) return;

  vt::Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();
  constexpr int64_t kRows = 3;
  constexpr int64_t kPadding = 37;
  constexpr std::array<int64_t, 5> kSuperblocks{{1, 2, 3, 10, 16}};
  constexpr std::array<DType, 3> kDTypes{{DType::kF32, DType::kF16, DType::kBF16}};
  const vt::cpu::FromFloatFn cpu_oracle = vt::cpu::BlockFromFloat(DType::kQ8_K);
  REQUIRE(cpu_oracle != nullptr);
  vt::rocm::Q8KResetRouteDispatchCountsForTest();

  for (DType dtype : kDTypes) {
    for (const Q8KInputCase& input_case : kQ8KInputCases) {
      for (int64_t nsb : kSuperblocks) {
        CAPTURE(std::string(Q8KDTypeName(dtype)));
        CAPTURE(std::string(input_case.name));
        CAPTURE(nsb);
        const int64_t k = nsb * vt::cpu::kQK_K;
        const int64_t row_stride = k + kPadding;
        const size_t source_bytes = static_cast<size_t>(kRows * row_stride) *
                                    Q8KSourceElementBytes(dtype);
        const size_t scratch_bytes =
            static_cast<size_t>(kRows * nsb) * kQ8KBlockBytes;
        std::vector<uint8_t> source(source_bytes, 0xD3);
        FillQ8KInput(source, dtype, row_stride, kRows, nsb, input_case.kind);

        std::vector<uint8_t> cpu_bytes(scratch_bytes, 0xA5);
        std::vector<float> cpu_row(static_cast<size_t>(k));
        for (int64_t row = 0; row < kRows; ++row) {
          for (int64_t col = 0; col < k; ++col) {
            cpu_row[static_cast<size_t>(col)] = LoadQ8KSource(
                source, dtype, static_cast<size_t>(row * row_stride + col));
          }
          cpu_oracle(cpu_row.data(),
                     cpu_bytes.data() + static_cast<size_t>(row * nsb) * kQ8KBlockBytes, k);
        }

        DevBufBytes device_source(rocm, q, source_bytes);
        DevBufBytes device_legacy(rocm, q, scratch_bytes);
        DevBufBytes device_candidate(rocm, q, scratch_bytes);
        std::vector<uint8_t> sentinel(scratch_bytes, 0xA5);
        device_source.Upload(source.data());
        device_legacy.Upload(sentinel.data());
        device_candidate.Upload(sentinel.data());
        vt::rocm::Q8KQuantizeForTest(q, device_legacy.ptr(), device_source.ptr(), dtype,
                                    row_stride, kRows, nsb, false);
        vt::rocm::Q8KQuantizeForTest(q, device_candidate.ptr(), device_source.ptr(), dtype,
                                    row_stride, kRows, nsb, true);
        std::vector<uint8_t> legacy(scratch_bytes);
        std::vector<uint8_t> candidate(scratch_bytes);
        device_legacy.Download(legacy.data());
        device_candidate.Download(candidate.data());

        for (int64_t row = 0; row < kRows; ++row) {
          for (int64_t sb = 0; sb < nsb; ++sb) {
            CheckQ8KBlockBytes(candidate, legacy, "legacy-gpu", dtype, input_case.name,
                               row, sb, nsb);
            CheckQ8KBlockBytes(candidate, cpu_bytes, "cpu-block-from-float", dtype,
                               input_case.name, row, sb, nsb);
          }
        }
      }
    }
  }
  CheckNoQ8KRouteCounters();
  rocm.DestroyQueue(q);
}

TEST_CASE("ROCm Q8_K dense and grouped production routes select one actual arm") {
  const bool rocm_registered = [] {
    for (DeviceType dtype : RegisteredDevices())
      if (dtype == DeviceType::kROCM) return true;
    return false;
  }();
  if (!rocm_registered) return;
  REQUIRE(OpAvailable(vt::OpId::kMatmulBTQuant, DeviceType::kROCM));
  REQUIRE(OpAvailable(vt::OpId::kMatmulBTQuantGrouped, DeviceType::kROCM));

  const std::string actual_arch = vt::rocm::DeviceArchName(0);
  if (actual_arch.rfind("gfx1100", 0) != 0) return;
  REQUIRE(actual_arch.rfind("gfx1100", 0) == 0);

  constexpr int64_t kM = 2;
  constexpr int64_t kP = 2;
  constexpr int64_t kN = 4;
  constexpr int64_t kE = 3;
  constexpr int64_t kK = vt::cpu::kQK_K;
  constexpr size_t kQ4BlockBytes = sizeof(vt::cpu::BlockQ4_K);
  std::vector<float> dense_act = RandomVec(static_cast<size_t>(kM * kK), 18760, -0.5f, 0.5f);
  std::vector<float> grouped_act = RandomVec(static_cast<size_t>(kP * kK), 18761, -0.5f, 0.5f);
  std::vector<uint8_t> dense_weight(static_cast<size_t>(kN) * kQ4BlockBytes);
  std::vector<uint8_t> grouped_weight(static_cast<size_t>(kE * kN) * kQ4BlockBytes);
  std::mt19937 rng(18762);
  for (uint8_t& byte : dense_weight) byte = static_cast<uint8_t>(rng() & 0xff);
  for (uint8_t& byte : grouped_weight) byte = static_cast<uint8_t>(rng() & 0xff);
  auto set_q4_deltas = [](std::vector<uint8_t>& bytes) {
    for (size_t offset = 0; offset < bytes.size(); offset += kQ4BlockBytes) {
      const uint16_t d = vt::F32ToF16(0.0125f);
      const uint16_t dmin = vt::F32ToF16(0.0075f);
      std::memcpy(bytes.data() + offset, &d, sizeof(d));
      std::memcpy(bytes.data() + offset + sizeof(d), &dmin, sizeof(dmin));
    }
  };
  set_q4_deltas(dense_weight);
  set_q4_deltas(grouped_weight);
  const std::vector<int32_t> expert_ids{2, 0};

  vt::Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();
  const Device device{DeviceType::kROCM, 0};
  DevBuf dense_act_device(rocm, q, dense_act.size());
  DevBuf grouped_act_device(rocm, q, grouped_act.size());
  DevBufBytes dense_weight_device(rocm, q, dense_weight.size());
  DevBufBytes grouped_weight_device(rocm, q, grouped_weight.size());
  DevBufI32 expert_ids_device(rocm, q, expert_ids.size());
  DevBuf dense_out(rocm, q, static_cast<size_t>(kM * kN));
  DevBuf grouped_out(rocm, q, static_cast<size_t>(kP * kN));
  DevBufBytes kernel_witness_device(rocm, q, sizeof(Q8KKernelWitnessCounts));
  dense_act_device.Upload(dense_act);
  grouped_act_device.Upload(grouped_act);
  dense_weight_device.Upload(dense_weight.data());
  grouped_weight_device.Upload(grouped_weight.data());
  expert_ids_device.Upload(expert_ids);

  Tensor dense_act_tensor = T2(dense_act_device.ptr(), device, kM, kK);
  Tensor dense_weight_tensor = Tensor::Contiguous(
      dense_weight_device.ptr(), DType::kQ4_K, device, {kN, kK});
  Tensor dense_out_tensor = T2(dense_out.ptr(), device, kM, kN);
  Tensor grouped_act_tensor = T2(grouped_act_device.ptr(), device, kP, kK);
  Tensor grouped_weight_tensor = Tensor::Contiguous(
      grouped_weight_device.ptr(), DType::kQ4_K, device, {kE * kN, kK});
  Tensor expert_ids_tensor = TI32(expert_ids_device.ptr(), device, kP);
  Tensor grouped_out_tensor = T2(grouped_out.ptr(), device, kP, kN);
  ScopedQ8KKernelExecutionWitness kernel_witness(kernel_witness_device.ptr());

  auto run_dense = [&] {
    vt::MatmulBTQuant(q, dense_out_tensor, dense_act_tensor, dense_weight_tensor);
    rocm.Synchronize(q);
  };
  auto run_grouped = [&] {
    vt::MatmulBTQuantGrouped(q, grouped_out_tensor, grouped_act_tensor,
                            grouped_weight_tensor, expert_ids_tensor);
    rocm.Synchronize(q);
  };

  struct ArmCase {
    const char* value;
    bool candidate;
  };
  const ArmCase arms[] = {{"0", false}, {"1", true}, {nullptr, true}};
  for (const ArmCase& arm : arms) {
    CAPTURE(arm.value == nullptr ? std::string("unset") : std::string(arm.value));
    {
      ScopedQ8KEnv env(arm.value);
      vt::rocm::Q8KResetRouteDispatchCountsForTest();
      ResetQ8KKernelExecutionWitness(kernel_witness_device);
      run_dense();
      CheckOnlyQ8KRouteCounter(false, arm.candidate);
      CheckOnlyQ8KKernelExecutionWitness(kernel_witness_device, arm.candidate);
    }
    {
      ScopedQ8KEnv env(arm.value);
      vt::rocm::Q8KResetRouteDispatchCountsForTest();
      ResetQ8KKernelExecutionWitness(kernel_witness_device);
      run_grouped();
      CheckOnlyQ8KRouteCounter(true, arm.candidate);
      CheckOnlyQ8KKernelExecutionWitness(kernel_witness_device, arm.candidate);
    }
  }

  {
    ScopedQ8KEnv env("invalid-production");
    vt::rocm::Q8KResetRouteDispatchCountsForTest();
    ResetQ8KKernelExecutionWitness(kernel_witness_device);
    CHECK_THROWS_WITH_AS(run_dense(),
                         doctest::Contains("VT_ROCM_Q8K_BLOCK=invalid-production"),
                         std::runtime_error);
    CheckNoQ8KRouteCounters();
    CheckNoQ8KKernelExecutionWitness(kernel_witness_device);
  }
  {
    ScopedQ8KEnv env("invalid-production");
    vt::rocm::Q8KResetRouteDispatchCountsForTest();
    ResetQ8KKernelExecutionWitness(kernel_witness_device);
    CHECK_THROWS_WITH_AS(run_grouped(),
                         doctest::Contains("VT_ROCM_Q8K_BLOCK=invalid-production"),
                         std::runtime_error);
    CheckNoQ8KRouteCounters();
    CheckNoQ8KKernelExecutionWitness(kernel_witness_device);
  }
  rocm.DestroyQueue(q);
}

TEST_CASE("ROCm Q6_K decode spreads one row's superblocks over several warps") {
  // Issue #1910: KQuantGemmK strides ONE warp's 32 lanes over nsb = K/256
  // superblocks, so a 4096-wide projection (nsb = 16) leaves lanes 16..31 with
  // nothing to do while the full 5-round reduction still runs — 195 of the 259
  // decode dispatches the issue measured. The answer is llama.cpp's: put
  // several warps on the same output row and split each superblock's eight
  // sub-blocks between them (mmvq.cu calc_nwarps, the RDNA4 branch, pin
  // b10451).
  //
  // It is gated for Q6_K ONLY, which is a measured restriction and not a
  // scoping choice — Q4_K and Q5_K lose 1.5x to 1.8x at every width tried,
  // because splitting a superblock re-reads its header once per warp and
  // breaks the quant read's contiguity. The sweep is in the spec's
  // `## Outcome`. Pinning the losing formats to the single-warp arm here is
  // what stops the width being widened back to them without a measurement.
  //
  // Two things need gating and they fail differently. The dispatch POLICY is
  // pinned below on any host, GPU or not. Whether the launcher actually reaches
  // the cooperative arm cannot be read off the output, because the arm reduces
  // the INTEGER dp4a accumulators across warps and so is bit-identical to the
  // single-warp one by construction — a dispatch counter answers that, and the
  // bit-identity itself is gated against the single-warp arm on real hardware.

  // --- policy: pinned everywhere, including a HIP build with no board ---
  // The issue's k=4096 decode shapes, and the boundary the trigger names.
  const int w16 = vt::rocm::KQuantDecodeCoopWarps(DType::kQ6_K, 1, 16);
  CHECK(w16 > 1);
  CHECK(vt::rocm::KQuantDecodeCoopWarps(DType::kQ6_K, 1, 32) > 1);
  // The launcher instantiates exactly one width; any other value falls through
  // to the single-warp arm, so a policy that returns one would be a silent
  // no-op rather than the dispatch this row measured.
  CHECK(w16 == 8);
  // The defect restated as the quantity that fixes it: every one of the w16
  // warps has to receive a whole share of the superblock's eight sub-blocks,
  // or a warp is dispatched with nothing to do — which is the failure the
  // single-warp arm already had, moved one level up.
  CHECK(8 % w16 == 0);
  // k=12288 (nsb = 48) already packs all 32 lanes: the spec's gate requires
  // that shape to keep the kernel and launch config it had.
  CHECK(vt::rocm::KQuantDecodeCoopWarps(DType::kQ6_K, 1, 48) == 1);
  // Prefill is out of this row's scope; m > 1 keeps the arm it had.
  CHECK(vt::rocm::KQuantDecodeCoopWarps(DType::kQ6_K, 64, 16) == 1);
  // Measured losers and a format that is a different kernel entirely.
  CHECK(vt::rocm::KQuantDecodeCoopWarps(DType::kQ4_K, 1, 16) == 1);
  CHECK(vt::rocm::KQuantDecodeCoopWarps(DType::kQ5_K, 1, 16) == 1);
  CHECK(vt::rocm::KQuantDecodeCoopWarps(DType::kQ8_0, 1, 16) == 1);

  const bool rocm_registered = [] {
    for (DeviceType dt : RegisteredDevices())
      if (dt == DeviceType::kROCM) return true;
    return false;
  }();
  if (!rocm_registered) return;
  REQUIRE(OpAvailable(vt::OpId::kMatmulBTQuant, DeviceType::kROCM));

  // --- numeric: the cooperative arm is BIT-identical to the single-warp one ---
  // m == 1 takes the cooperative arm and m == 2 does not, so running the same
  // activation row under both and demanding byte equality is a direct read of
  // the reassociation this row could have introduced and did not. Anything
  // less than byte equality here is a token-exactness regression on a change
  // that only moves work between lanes.
  struct Fmt { DType dt; int64_t block_bytes; int d_off; int dmin_off; const char* name; };
  const Fmt fmts[] = {
    {DType::kQ4_K, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 210, 208, -1, "q6_K"},
  };
  // nsb = 16 is the issue's decode shape (cooperative); nsb = 48 is the
  // k=12288 shape that must stay on the arm it had.
  const int64_t ks[] = {4096, 12288};
  constexpr int64_t N = 64;
  vt::Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  for (const Fmt& f : fmts) {
    for (int64_t K : ks) {
      // Wrapped because doctest sends a `const char*` through
      // `filldata<T*>` to `const volatile void*`, which no `operator<<`
      // accepts and which therefore decays to `bool`. A bare CAPTURE of this
      // field logs `1` instead of the format name.
      CAPTURE(std::string(f.name));
      CAPTURE(K);
      const int64_t nsb = K / 256;
      const size_t row_bytes = static_cast<size_t>(nsb) * f.block_bytes;
      const size_t wn = static_cast<size_t>(N) * row_bytes;
      std::mt19937 rng(1910);
      std::vector<uint8_t> wt(wn);
      for (uint8_t& b : wt) b = static_cast<uint8_t>(rng() & 0xFF);
      for (int64_t r = 0; r < N; ++r)
        for (int64_t bIdx = 0; bIdx < nsb; ++bIdx) {
          uint8_t* blk = wt.data() + r * row_bytes + bIdx * f.block_bytes;
          const float jitter = 1.0f + 0.05f * static_cast<float>((r + bIdx) % 7);
          auto put16 = [&](int off, float v) {
            uint16_t h = vt::F32ToF16(v);
            std::memcpy(blk + off, &h, 2);
          };
          if (f.d_off >= 0) put16(f.d_off, 0.0125f * jitter);
          if (f.dmin_off >= 0) put16(f.dmin_off, 0.0075f * jitter);
        }
      const std::vector<float> row = RandomVec(static_cast<size_t>(K), 1911, -0.5f, 0.5f);
      std::vector<float> act2(row);                       // two IDENTICAL rows
      act2.insert(act2.end(), row.begin(), row.end());

      std::vector<float> ref(static_cast<size_t>(N), 0.0f);
      {
        vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
        Queue cq = cpu.CreateQueue();
        const Device cd{DeviceType::kCPU, 0};
        std::vector<float> ca = row;
        std::vector<uint8_t> cw = wt;
        Tensor tout = T2(ref.data(), cd, 1, N);
        Tensor tact = T2(ca.data(), cd, 1, K);
        Tensor twt = Tensor::Contiguous(cw.data(), f.dt, cd, {N, K});
        vt::MatmulBTQuant(cq, tout, tact, twt);
        cpu.DestroyQueue(cq);
      }

      Queue q = rocm.CreateQueue();
      const Device d{DeviceType::kROCM, 0};
      DevBufBytes dwt(rocm, q, wn);
      dwt.Upload(wt.data());
      Tensor twt = Tensor::Contiguous(dwt.ptr(), f.dt, d, {N, K});

      const uint64_t coop_before = vt::rocm::KQuantCoopDispatchCount();
      std::vector<float> got1;
      {
        DevBuf da(rocm, q, static_cast<size_t>(K));
        DevBuf dout(rocm, q, static_cast<size_t>(N));
        da.Upload(row);
        Tensor tact = T2(da.ptr(), d, 1, K);
        Tensor tout = T2(dout.ptr(), d, 1, N);
        vt::MatmulBTQuant(q, tout, tact, twt);
        got1 = dout.Download();
      }
      const uint64_t coop_after = vt::rocm::KQuantCoopDispatchCount();
      // Reachability: deleting the launcher's cooperative call site leaves this
      // counter flat and reds the case, which no output comparison can do.
      const bool coop_expected = f.dt == DType::kQ6_K && nsb <= 32;
      if (coop_expected) CHECK(coop_after > coop_before);
      else CHECK(coop_after == coop_before);

      std::vector<float> got2;
      {
        DevBuf da(rocm, q, static_cast<size_t>(2 * K));
        DevBuf dout(rocm, q, static_cast<size_t>(2 * N));
        da.Upload(act2);
        Tensor tact = T2(da.ptr(), d, 2, K);
        Tensor tout = T2(dout.ptr(), d, 2, N);
        vt::MatmulBTQuant(q, tout, tact, twt);
        got2 = dout.Download();
      }
      rocm.DestroyQueue(q);

      REQUIRE(got1.size() == static_cast<size_t>(N));
      REQUIRE(got2.size() == static_cast<size_t>(2 * N));
      // Byte equality, not a tolerance: same weights, same activation bytes.
      CHECK(std::memcmp(got1.data(), got2.data(), static_cast<size_t>(N) * sizeof(float)) == 0);
      CHECK(Nmse(ref, got1) <= kNmseTol);
    }
  }
}
#endif  // VLLM_CPP_HIP

TEST_CASE("grouped quant expert GEMM (Q8_0/Q4_K/Q6_K) matches the CPU oracle") {
  // kMatmulBTQuantGrouped on ROCm vs the CPU keep-quant reference
  // (cpu_quant_gemm.cpp:305). Valid random blocks (valid f16 deltas, random
  // quants) at a real expert-MLP shape. Integer cores are bit-exact ports;
  // the f16/f32 scale sum reassociates across lanes, so NMSE <= 5e-4.
  constexpr int64_t P = 3, N = 8, K = 512;         // K%256==0 (K-quant superblocks)
  constexpr int64_t E = 4;                          // experts
  const std::vector<int32_t> eids = {2, 0, 3};      // routed experts (non-sorted)

  struct Fmt { vt::DType dt; int64_t block_bytes; int d_off; int dmin_off; const char* name; };
  // offsets from ggml-common.h (restated in cpu_quant_blocks.h)
  const Fmt fmts[] = {
    {vt::DType::kQ8_0, 34, 0, -1, "q8_0"},   // {d; qs[32]}        K blocks of 32
    {vt::DType::kQ4_K, 144, 0, 2, "q4_K"},   // {d,dmin,sc,qs}     superblocks of 256
    {vt::DType::kQ6_K, 210, 208, -1, "q6_K"},// {ql,qh,scales,d}   superblocks of 256
    {vt::DType::kQ5_K, 176, 0, 2, "q5_K"},   // {d,dmin,sc,qh,qs}  superblocks of 256
  };

  // REQUIRE-proven registration on ROCm (never a silent skip — review sweep
  // on #523: an OpAvailable-guarded case passes green with the registration
  // deleted).
  const bool any_rocm = [&] {
    for (DeviceType dt : RegisteredDevices()) if (dt == DeviceType::kROCM) return true;
    return false;
  }();
  if (any_rocm) {
    REQUIRE_MESSAGE(OpAvailable(vt::OpId::kMatmulBTQuantGrouped, DeviceType::kROCM),
                    "kMatmulBTQuantGrouped must be registered on ROCm — a missing "
                    "registration is a failure, never a skip");
  }
  for (const Fmt& f : fmts) {
    CAPTURE(f.name);
    const int64_t elems_per_block = (f.dt == vt::DType::kQ8_0) ? 32 : 256;
    const int64_t blocks_per_row = K / elems_per_block;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * f.block_bytes;
    const size_t wn = static_cast<size_t>(E) * N * row_bytes;
    // Build valid random blocks: random quant bytes, small positive f16 deltas.
    std::mt19937 rng(777);
    std::vector<uint8_t> wt(wn);
    for (uint8_t& b : wt) b = static_cast<uint8_t>(rng() & 0xFF);
    for (int64_t r = 0; r < E * N; ++r)
      for (int64_t bIdx = 0; bIdx < blocks_per_row; ++bIdx) {
        uint8_t* blk = wt.data() + r * row_bytes + bIdx * f.block_bytes;
        const float jitter = 1.0f + 0.05f * static_cast<float>((r + bIdx) % 7);
        auto put16 = [&](int off, float v) { uint16_t h = vt::F32ToF16(v); std::memcpy(blk + off, &h, 2); };
        if (f.d_off >= 0) put16(f.d_off, 0.0125f * jitter);
        if (f.dmin_off >= 0) put16(f.dmin_off, 0.0075f * jitter);
      }
    const size_t an = static_cast<size_t>(P) * K, on = static_cast<size_t>(P) * N;
    const std::vector<float> act = RandomVec(an, 778, -0.5f, 0.5f);

    std::vector<float> ref(on, 0.0f);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> ca = act;
      std::vector<uint8_t> cw = wt;
      std::vector<int32_t> ce = eids;
      Tensor tout = T2(ref.data(), cd, P, N);
      Tensor tact = T2(ca.data(), cd, P, K);
      Tensor twt = Tensor::Contiguous(cw.data(), f.dt, cd, {E * N, K});
      Tensor te = TI32(ce.data(), cd, P);
      vt::MatmulBTQuantGrouped(cq, tout, tact, twt, te);
      cpu.DestroyQueue(cq);
    }
    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kMatmulBTQuantGrouped, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf da(dev, q, an);
      DevBufBytes dwt(dev, q, wn);
      DevBufI32 de(dev, q, P);
      DevBuf dout(dev, q, on);
      da.Upload(act);
      dwt.Upload(wt.data());
      de.Upload(eids);
      Tensor tact = T2(da.ptr(), d, P, K);
      Tensor twt = Tensor::Contiguous(dwt.ptr(), f.dt, d, {E * N, K});
      Tensor te = TI32(de.ptr(), d, P);
      Tensor tout = T2(dout.ptr(), d, P, N);
      vt::MatmulBTQuantGrouped(q, tout, tact, twt, te);
      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("ReshapeAndCache->PagedAttention composition matches CPU (real dims, shuffled blocks)") {
  // The "paged attention" case above hand-builds a contiguous KV cache; the
  // real model path writes it with ReshapeAndCache and reads it back. This
  // case is that composition, at real model dims (Dh=256, Hq=8, Hkv=2,
  // block_size 16), a shuffled block table, and a non-sequential slot mapping
  // — the layout a stride/scatter bug would live in and the contiguous case
  // cannot see.
  constexpr int64_t T = 20, Hq = 8, Hkv = 2, Dh = 256, BS = 16;
  constexpr int64_t kBlocks = 4;                 // 4 blocks x 16 slots = 64 >= 20
  const size_t qn = static_cast<size_t>(T) * Hq * Dh;
  const size_t kvn = static_cast<size_t>(T) * Hkv * Dh;
  const size_t cachen = static_cast<size_t>(kBlocks) * BS * Hkv * Dh;
  const std::vector<float> q = RandomVec(qn, 711);
  const std::vector<float> k = RandomVec(kvn, 712);
  const std::vector<float> v = RandomVec(kvn, 713);
  // The slot mapping must DERIVE from the logical position through the
  // (shuffled) block table — exactly what the engine produces — otherwise the
  // attention read of logical position p lands on a slot nothing wrote and
  // both backends compare zeros (review on #497: the first version's
  // (i*7+3)%64 scatter was disjoint from the block table, so the composition
  // exercised mostly-unwritten cache).
  std::vector<int32_t> block_table = {3, 1, 2, 0};  // shuffled physical blocks
  std::vector<int64_t> slots(T);
  for (int64_t i = 0; i < T; ++i)
    slots[i] = static_cast<int64_t>(block_table[static_cast<size_t>(i / BS)]) * BS + (i % BS);
  std::vector<int32_t> seq_lens = {T};
  std::vector<int32_t> qsl = {0, T};
  vt::PagedAttentionArgs pa;
  pa.scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  pa.causal = true;

  std::vector<float> ref_out(static_cast<size_t>(T) * Hq * Dh, 0.0f);
  {
    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> q_host = q, ck = k, cv = v;
    std::vector<float> ckc(cachen, 0.0f), cvc(cachen, 0.0f);
    std::vector<int64_t> cslots = slots;
    std::vector<int32_t> cbt = block_table, csl = seq_lens, cqsl = qsl;
    Tensor tq = Tensor::Contiguous(q_host.data(), DType::kF32, cd, {T, Hq, Dh});  // contiguous (op contract)
    Tensor tk = Tensor::Contiguous(ck.data(), DType::kF32, cd, {T, Hkv, Dh});
    Tensor tv = Tensor::Contiguous(cv.data(), DType::kF32, cd, {T, Hkv, Dh});
    Tensor tkc = Tensor::Contiguous(ckc.data(), DType::kF32, cd, {kBlocks, BS, Hkv, Dh});
    Tensor tvc = Tensor::Contiguous(cvc.data(), DType::kF32, cd, {kBlocks, BS, Hkv, Dh});
    Tensor tsm = Tensor::Contiguous(cslots.data(), DType::kI64, cd, {T});
    vt::ReshapeAndCache(cq, tk, tv, tkc, tvc, tsm);
    Tensor tbt = Tensor::Contiguous(cbt.data(), DType::kI32, cd, {1, kBlocks});
    Tensor tsl = Tensor::Contiguous(csl.data(), DType::kI32, cd, {1});
    Tensor tqsl = Tensor::Contiguous(cqsl.data(), DType::kI32, cd, {2});
    Tensor to = Tensor::Contiguous(ref_out.data(), DType::kF32, cd, {T, Hq, Dh});
    vt::PagedAttention(cq, to, tq, tkc, tvc, tbt, tsl, tqsl, pa);
    cpu.DestroyQueue(cq);
  }
  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kPagedAttention, dt) || !OpAvailable(vt::OpId::kReshapeAndCache, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q_ = dev.CreateQueue();
    const Device d{dt, 0};
    DevBuf dq(dev, q_, qn), dk(dev, q_, kvn), dv(dev, q_, kvn),
        dkc(dev, q_, cachen), dvc(dev, q_, cachen), dout(dev, q_, static_cast<size_t>(T) * Hq * Dh);
    DevBufBytes dsm(dev, q_, T * 8), dbt(dev, q_, kBlocks * 4), dsl_(dev, q_, 4), dqsl(dev, q_, 8);
    dq.Upload(q); dk.Upload(k); dv.Upload(v);
    dkc.Upload(std::vector<float>(cachen, 0.0f)); dvc.Upload(std::vector<float>(cachen, 0.0f));
    dsm.Upload(slots.data()); dbt.Upload(block_table.data());
    dsl_.Upload(seq_lens.data()); dqsl.Upload(qsl.data());
    Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {T, Hq, Dh});  // contiguous (op contract)
    Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {T, Hkv, Dh});
    Tensor tv = Tensor::Contiguous(dv.ptr(), DType::kF32, d, {T, Hkv, Dh});
    Tensor tkc = Tensor::Contiguous(dkc.ptr(), DType::kF32, d, {kBlocks, BS, Hkv, Dh});
    Tensor tvc = Tensor::Contiguous(dvc.ptr(), DType::kF32, d, {kBlocks, BS, Hkv, Dh});
    Tensor tsm = Tensor::Contiguous(dsm.ptr(), DType::kI64, d, {T});
    vt::ReshapeAndCache(q_, tk, tv, tkc, tvc, tsm);
    Tensor tbt = Tensor::Contiguous(dbt.ptr(), DType::kI32, d, {1, kBlocks});
    Tensor tsl = Tensor::Contiguous(dsl_.ptr(), DType::kI32, d, {1});
    Tensor tqsl = Tensor::Contiguous(dqsl.ptr(), DType::kI32, d, {2});
    Tensor to = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {T, Hq, Dh});
    vt::PagedAttention(q_, to, tq, tkc, tvc, tbt, tsl, tqsl, pa);
    CHECK(Nmse(ref_out, dout.Download()) <= kNmseTol);

    // Anti-vacuity guard (review on #497): a WRONG physical mapping must NOT
    // reproduce the reference — if the composition were vacuous (reads never
    // hitting writes), a corrupted table would compare equal. Blocks 0 and 2
    // both carry real tokens under the true table, so swapping them must
    // change the output.
    // Swap the mapping of the first two LOGICAL blocks — both hold real
    // tokens (0-15 and 16-19), so the read path changes. (The first version
    // of this guard swapped two blocks OUTSIDE the logical range and was
    // itself vacuous — the guard proved the guard.)
    std::vector<int32_t> bad_table = {1, 3, 2, 0};
    DevBufBytes dbt_bad(dev, q_, kBlocks * 4);
    dbt_bad.Upload(bad_table.data());
    DevBuf dout2(dev, q_, static_cast<size_t>(T) * Hq * Dh);
    Tensor tbt2 = Tensor::Contiguous(dbt_bad.ptr(), DType::kI32, d, {1, kBlocks});
    Tensor to2 = Tensor::Contiguous(dout2.ptr(), DType::kF32, d, {T, Hq, Dh});
    vt::PagedAttention(q_, to2, tq, tkc, tvc, tbt2, tsl, tqsl, pa);
    const std::vector<float> bad_out = dout2.Download();
    bool any_diff = false;
    for (size_t i = 0; i < ref_out.size(); ++i)
      if (std::fabs(bad_out[i] - ref_out[i]) > 1e-3f) { any_diff = true; break; }
    CHECK_MESSAGE(any_diff,
                  "a corrupted block table must change the attention output — "
                  "otherwise the composition test is vacuous");
    dev.DestroyQueue(q_);
  }
}


// Scalar bf16 RNE round-trip helpers for host-side oracles (the MoE combine
// gate reference rounds the shared term through bf16 exactly like the kernel).
static uint16_t F32ToBf16Rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static float Bf16ToF32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

TEST_CASE("MoE combine/gate ops match the CPU oracle") {
  constexpr int64_t T = 5, H = 64, K = 3;
  const size_t en = static_cast<size_t>(T) * K * H, on = static_cast<size_t>(T) * H;
  const std::vector<float> eo = RandomVec(en, 911);
  const std::vector<float> w = RandomVec(static_cast<size_t>(T) * K, 912, 0.0f, 1.0f);
  const std::vector<float> sd = RandomVec(on, 913);
  const std::vector<uint16_t> eo_bf = Bf16Bits(eo), sd_bf = Bf16Bits(sd);
  // SharedExpertGate (sigmoid*mul), MoeCombine (weighted expert sum +/-
  // shared), MoeCombineGate (combine + folded shared gate). f32 and bf16 arms,
  // PLUS the production dtype mix the model actually runs (review sweep on
  // #509): expert_out bf16 (qwen3_5.cpp DBuf ddown), shared bf16, out bf16.
  // MoeCombine/MoeCombineGate are thread-per-element with a single store
  // rounding and NO cross-lane reduction (cuda_moe.cu:465-468), so both arms
  // are asserted BIT-EXACT — the NMSE aggregate would tolerate a few wrong
  // elements, which is exactly how a 2x OOB read hides.
  const std::vector<float> gl = RandomVec(static_cast<size_t>(T), 914);

  for (DeviceType dt : RegisteredDevices()) {
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};
    // CPU oracle for all three, f32.
    std::vector<uint16_t> ref_sg_b(on, 0);
  std::vector<float> ref_c(on), ref_cg(on);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> csd = sd, cgl = gl, ceo = eo, cw = w;
      Tensor tout = Tensor::Contiguous(ref_sg_b.data(), DType::kBF16, cd, {T, H});
      Tensor tsd = T2(csd.data(), cd, T, H);
      Tensor tgl = T1(cgl.data(), cd, T);
      if (OpAvailable(vt::OpId::kSharedExpertGate, DeviceType::kCPU))
        vt::SharedExpertGate(cq, tout, tsd, tgl);
      Tensor teo = Tensor::Contiguous(ceo.data(), DType::kF32, cd, {T, K, H});
      Tensor tw = T2(cw.data(), cd, T, K);
      Tensor to2 = T2(ref_c.data(), cd, T, H);
      if (OpAvailable(vt::OpId::kMoeCombine, DeviceType::kCPU))
        vt::MoeCombine(cq, to2, teo, tw, &tsd, 1.0f);
      cpu.DestroyQueue(cq);
      // MoeCombineGate has no CPU op registration; the oracle is the composite
      // computed on host: MoeCombine (no shared) + bf16-round(sigmoid(gl)*sd).
      for (int64_t r = 0; r < T; ++r) {
        const float g = 1.0f / (1.0f + std::exp(-gl[static_cast<size_t>(r)]));
        for (int64_t c2 = 0; c2 < H; ++c2) {
          float acc = 0.0f;
          for (int64_t j = 0; j < K; ++j)
            acc += w[static_cast<size_t>(r * K + j)] * eo[static_cast<size_t>((r * K + j) * H + c2)];
          const float sv = g * sd[static_cast<size_t>(r * H + c2)];
          const uint16_t svb = F32ToBf16Rne(sv);
          acc += Bf16ToF32(svb);
          ref_cg[static_cast<size_t>(r * H + c2)] = acc;
        }
      }
    }
    // device
    DevBuf deo(dev, q, en), dw(dev, q, T * K), dsd(dev, q, on), dgl(dev, q, T), dout(dev, q, on);
    DevBufBytes doutb(dev, q, on * 2);
    deo.Upload(eo); dw.Upload(w); dsd.Upload(sd); dgl.Upload(gl);
    Tensor teo = Tensor::Contiguous(deo.ptr(), DType::kF32, d, {T, K, H});
    Tensor tw = T2(dw.ptr(), d, T, K);
    Tensor tsd = T2(dsd.ptr(), d, T, H);
    Tensor tgl = T1(dgl.ptr(), d, T);
    Tensor tout = T2(dout.ptr(), d, T, H);
    if (OpAvailable(vt::OpId::kSharedExpertGate, dt)) {
      Tensor toutb = Tensor::Contiguous(doutb.ptr(), DType::kBF16, d, {T, H});
      vt::SharedExpertGate(q, toutb, tsd, tgl);
      std::vector<uint16_t> gotb(on);
      doutb.Download(gotb.data());
      CHECK(gotb == ref_sg_b);  // both sides store bf16: bit-exact
    }
    if (OpAvailable(vt::OpId::kMoeCombine, dt)) {
      vt::MoeCombine(q, tout, teo, tw, &tsd, 1.0f);
      // Thread-per-element, single store rounding, no cross-lane reduction:
      // bit-exact is the achievable and asserted bar (review sweep on #509).
      CHECK(dout.Download() == ref_c);
    }
    if (OpAvailable(vt::OpId::kMoeCombineGate, dt)) {
      vt::MoeCombineGate(q, tout, teo, tw, tsd, tgl);
      CHECK(Nmse(ref_cg, dout.Download()) <= kNmseTol);
    }

    // The production bf16 arm: expert_out bf16 + shared bf16 + out bf16
    // (qwen3_5.cpp:5463 ddown / :5326 shared). The CPU oracle runs the same
    // ops on the same bf16 tensors; both sides thread-per-element with the
    // same sequential K order, so the assertion is BIT-EXACT.
    DevBufBytes deo_bf(dev, q, en * 2), dsd_bf(dev, q, on * 2), dout_bf(dev, q, on * 2);
    deo_bf.Upload(eo_bf.data());
    dsd_bf.Upload(sd_bf.data());
    Tensor teo_b = Tensor::Contiguous(deo_bf.ptr(), DType::kBF16, d, {T, K, H});
    Tensor tsd_b = Tensor::Contiguous(dsd_bf.ptr(), DType::kBF16, d, {T, H});
    Tensor tout_b = Tensor::Contiguous(dout_bf.ptr(), DType::kBF16, d, {T, H});
    if (OpAvailable(vt::OpId::kMoeCombine, dt) &&
        OpAvailable(vt::OpId::kMoeCombine, DeviceType::kCPU)) {
      // CPU reference on bf16 tensors.
      std::vector<uint16_t> ref_b(on, 0);
      {
        vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
        Queue cq = cpu.CreateQueue();
        const Device cd{DeviceType::kCPU, 0};
        std::vector<uint16_t> ceo = eo_bf, csd = sd_bf;
        std::vector<float> cw = w;
        Tensor r = Tensor::Contiguous(ref_b.data(), DType::kBF16, cd, {T, H});
        Tensor teo_c = Tensor::Contiguous(ceo.data(), DType::kBF16, cd, {T, K, H});
        Tensor tw_c = T2(cw.data(), cd, T, K);
        Tensor tsd_c = Tensor::Contiguous(csd.data(), DType::kBF16, cd, {T, H});
        vt::MoeCombine(cq, r, teo_c, tw_c, &tsd_c, 0.7f);
        cpu.DestroyQueue(cq);
      }
      vt::MoeCombine(q, tout_b, teo_b, tw, &tsd_b, 0.7f);
      std::vector<uint16_t> got_b(on);
      dout_bf.Download(got_b.data());
      CHECK(got_b == ref_b);
    }
    dev.DestroyQueue(q);
  }
}

TEST_CASE("reference tier: an op with no native kernel matches the CPU oracle (unified only)") {
  constexpr int64_t kRows = 7, kCols = 48;
  constexpr size_t kN = kRows * kCols;
  const std::vector<float> in = RandomVec(kN, 1313, -4.0f, 4.0f);

  // CPU oracle through the same vt::Relu entry point (Relu is exact elementwise).
  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ci = in, ref(kN);
  {
    Tensor ti = T2(ci.data(), cd, kRows, kCols);
    Tensor to = T2(ref.data(), cd, kRows, kCols);
    vt::Relu(cq, to, ti);
  }
  cpu.DestroyQueue(cq);

  for (DeviceType dt : RegisteredDevices()) {
    if (!vt::GetBackend(dt).UnifiedMemory()) continue;  // safety: unified only
    // Only meaningful where the device LACKS a native kernel for the op; where it
    // has one, the native path is already covered by the NMSE cases above.
    if (vt::OpRegistered(vt::OpId::kRelu, dt)) continue;
    CAPTURE(DeviceName(dt));

    const unsigned long long hits_before = vt::GetReferenceTierHits();
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};
    DevBuf din(dev, q, kN), dout(dev, q, kN);
    din.Upload(in);
    Tensor ti = T2(din.ptr(), d, kRows, kCols);
    Tensor to = T2(dout.ptr(), d, kRows, kCols);
    vt::Relu(q, to, ti);  // no native kernel -> portable CPU fallback

    // Same host kernel, so bit-identical to the CPU oracle, not just close.
    const std::vector<float> got = dout.Download();
    CHECK(std::memcmp(ref.data(), got.data(), kN * sizeof(float)) == 0);
    // The fallback fired and it was not silent.
    CHECK(vt::GetReferenceTierHits() > hits_before);
    CHECK(std::string(vt::OpProviderNameAt(vt::OpId::kRelu, dt, 0)) ==
          vt::kReferenceProviderName);
    dev.DestroyQueue(q);
  }
}

// ---------------------------------------------------------------------------
// MLA/DSA campaign W1 (BACKEND-ROCM, #2715). Four ops that served from the
// portable CPU reference tier on gfx1151, so GLM-5.3 ran them on the host.
//
// READ THIS BEFORE LOOSENING ANY ASSERTION HERE. The reference tier computes the
// SAME ANSWER as a native kernel — it IS the CPU oracle, running on the host
// against device memory the backend reports host-addressable. So an
// oracle-equality assertion is GREEN on a backend with no kernel at all, and it
// was green here before any of these four existed. The assertion that can tell
// the two apart is `vt::OpRegistered`, which is a NATIVE-ONLY probe by design
// (src/vt/op_provider.cpp:788-806) and deliberately excludes the tier. That is
// why the registration case below is separate, unconditional on ROCm, and is the
// one that goes red when a RegisterOp line is deleted.
// ---------------------------------------------------------------------------

TEST_CASE("ROCm registers the W1 MLA ops natively rather than serving them from the tier") {
  // #2715: eight MLA/DSA ops had no ROCm registration, so every one of them
  // installed a host kernel on gfx1151 and docs/ROCM.md:60-61 disqualified any
  // performance result from the run. These four are the W1 slice.
  //
  // This case is NOT `if (!OpAvailable) continue`. The harness header says a
  // device that has not registered an op is skipped rather than failed, and that
  // is right for a partial backend in general — but it is exactly the polarity
  // that let these eight sit unregistered while every numeric arm stayed green.
  // Here the missing registration IS the defect under test.
  bool rocm_built = false;
  for (DeviceType dt : RegisteredDevices()) {
    if (dt == DeviceType::kROCM) rocm_built = true;
  }
  if (!rocm_built) return;  // not a ROCm build — nothing to assert about ROCm

  CHECK(vt::OpRegistered(vt::OpId::kBatchedMatmul, DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kConcatAndCacheMla, DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kConcatMlaNopeRope, DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kGatherMlaCache, DeviceType::kROCM));

  // The three that are STILL owed after this wave (spec § Owed). Asserting their
  // absence would be a lock on the next wave, so this only records them: the
  // GLM-5.3 speed axis stays VOID while any of them is false, and the campaign's
  // W2/W3 flip them.
  const std::string owed =
      std::string("W2/W3 still owed on ROCm (#2715) — kDsaIndexerLogits=") +
      (vt::OpRegistered(vt::OpId::kDsaIndexerLogits, DeviceType::kROCM) ? "1" : "0") +
      " kDsaTopkSelect=" +
      (vt::OpRegistered(vt::OpId::kDsaTopkSelect, DeviceType::kROCM) ? "1" : "0") +
      " — the GLM-5.3 speed axis stays VOID while either is 0. "
      "kMlaDecodeAttention left this list when #2926 landed it.";
  MESSAGE(owed);
}

TEST_CASE("ConcatAndCacheMla writes the concatenated MLA entry BIT-EXACTLY") {
  // Upstream cache_kernels.cu:401-442. Geometry is DeepSeek/GLM-shaped but
  // small; `kRank` is deliberately NOT a multiple of the 512-thread launch block
  // so the strided copy's tail runs rather than dividing away.
  constexpr int64_t kTokens = 11, kRank = 37, kPe = 8, kBlocks = 5, kBlockSize = 4;
  constexpr int64_t kWidth = kRank + kPe;
  constexpr int64_t kCacheN = kBlocks * kBlockSize * kWidth;

  // GUARD BANDS, and they are not decoration. The padded-token skip
  // (`slot < 0`, upstream cache_kernels.cu:419-422) is the one guarantee in this
  // op whose removal writes OUTSIDE the cache rather than inside it: `slot == -1`
  // gives `block = -1/4 = 0` and `offset = -1%4 = -1`, so the entry address is
  // NEGATIVE and a kernel that dropped the skip scribbles BEFORE the buffer. A
  // test that allocated exactly the cache would compare only in-range words,
  // find them all correct, and report that mutation as killed when it was not —
  // measured: the first version of this case passed 46/46 with the skip removed.
  // The tensor therefore points at the MIDDLE of a wider allocation and both
  // bands are asserted untouched.
  constexpr int64_t kGuard = 64;
  const std::vector<float> kv_c = RandomVec(kTokens * kRank, 27101);
  const std::vector<float> k_pe = RandomVec(kTokens * kPe, 27102);
  // The pre-seed is what a kernel that writes nothing has to overwrite, and it
  // is also what a PADDED slot must still be holding at the end.
  const std::vector<float> seed(kCacheN, -13.25f);
  std::vector<float> seed_padded(static_cast<size_t>(kCacheN + 2 * kGuard), 88.125f);
  // A plain loop rather than std::copy: this file does not include <algorithm>
  // and picking one up transitively from a libstdc++ header is not portable to
  // the ROCm toolchain's libc++.
  for (int64_t i = 0; i < kCacheN; ++i) {
    seed_padded[static_cast<size_t>(kGuard + i)] = seed[static_cast<size_t>(i)];
  }

  // Slots are shuffled across blocks, not ascending: a kernel that walked the
  // cache linearly instead of through the slot map passes on an identity map.
  // Two slots are -1 — upstream's padded token (`:419-422`), which must be
  // SKIPPED, leaving the seed value in place. Nothing else may be touched.
  std::vector<int64_t> slots(kTokens);
  const int64_t pattern[kTokens] = {7, 0, -1, 13, 3, 18, 11, -1, 5, 16, 9};
  for (int64_t i = 0; i < kTokens; ++i) slots[static_cast<size_t>(i)] = pattern[i];

  std::vector<float> ref(seed);
  {
    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> a = kv_c, b = k_pe;
    std::vector<int64_t> sm = slots;
    Tensor tkv = T2(a.data(), cd, kTokens, kRank);
    Tensor tpe = T2(b.data(), cd, kTokens, kPe);
    Tensor tc = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kBlocks, kBlockSize, kWidth});
    Tensor ts = Tensor::Contiguous(sm.data(), DType::kI64, cd, {kTokens});
    vt::ConcatAndCacheMla(cq, tkv, tpe, tc, ts);
    cpu.DestroyQueue(cq);
  }
  // The oracle must actually have moved the buffer, or "equal" means nothing.
  REQUIRE(ref != seed);

  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kConcatAndCacheMla, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    DevBuf dkv(dev, q, kTokens * kRank), dpe(dev, q, kTokens * kPe),
        dc(dev, q, static_cast<size_t>(kCacheN + 2 * kGuard));
    dkv.Upload(kv_c);
    dpe.Upload(k_pe);
    dc.Upload(seed_padded);
    // The cache the op is handed starts kGuard floats into the allocation.
    void* cache_ptr = static_cast<void*>(static_cast<float*>(dc.ptr()) + kGuard);
    void* ds = dev.Alloc(kTokens * sizeof(int64_t));
    dev.Copy(q, ds, slots.data(), kTokens * sizeof(int64_t));
    dev.Synchronize(q);

    Tensor tkv = T2(dkv.ptr(), d, kTokens, kRank);
    Tensor tpe = T2(dpe.ptr(), d, kTokens, kPe);
    Tensor tc = Tensor::Contiguous(cache_ptr, DType::kF32, d, {kBlocks, kBlockSize, kWidth});
    Tensor ts = Tensor::Contiguous(ds, DType::kI64, d, {kTokens});

    const unsigned long long hits_before = vt::GetReferenceTierHits();
    vt::ConcatAndCacheMla(q, tkv, tpe, tc, ts);
    dev.Synchronize(q);
    const std::vector<float> got = dc.Download();
    // A pure copy has no reassociation, so the bar is EQUALITY, not NMSE.
    CHECK(std::vector<float>(got.begin() + kGuard, got.begin() + kGuard + kCacheN) == ref);
    // Both guard bands, which is what makes the padded-token skip checkable.
    CHECK(std::vector<float>(got.begin(), got.begin() + kGuard) ==
          std::vector<float>(static_cast<size_t>(kGuard), 88.125f));
    CHECK(std::vector<float>(got.begin() + kGuard + kCacheN, got.end()) ==
          std::vector<float>(static_cast<size_t>(kGuard), 88.125f));
    // If this moved, the call above ran on the host, not on the device — the
    // same quantity docs/ROCM.md:60-61 disqualifies a speed result on.
    CHECK(vt::GetReferenceTierHits() == hits_before);

    dev.Free(ds);
    dev.DestroyQueue(q);
  }
}

TEST_CASE("ConcatMlaNopeRope concatenates BIT-EXACTLY, broadcast rope head and per-head") {
  // Upstream cache_kernels.cu:1572-1584 (decode q, per-head rope) and
  // mla_attention.py:2063-2092 (prefill k, ONE shared rope head broadcast over
  // every q head). Both arms run: a kernel that ignores the broadcast flag is
  // green on the per-head arm alone.
  constexpr int64_t kTokens = 6, kHeads = 5, kDn = 19, kDr = 7;

  for (bool broadcast : {false, true}) {
    CAPTURE(broadcast);
    const int64_t rope_heads = broadcast ? 1 : kHeads;
    const std::vector<float> nope = RandomVec(kTokens * kHeads * kDn, 27201);
    const std::vector<float> rope = RandomVec(kTokens * rope_heads * kDr, 27202);
    const std::vector<float> seed(kTokens * kHeads * (kDn + kDr), 41.5f);

    std::vector<float> ref(seed);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> a = nope, b = rope;
      Tensor to =
          Tensor::Contiguous(ref.data(), DType::kF32, cd, {kTokens, kHeads, kDn + kDr});
      Tensor tn = Tensor::Contiguous(a.data(), DType::kF32, cd, {kTokens, kHeads, kDn});
      Tensor tr = Tensor::Contiguous(b.data(), DType::kF32, cd, {kTokens, rope_heads, kDr});
      vt::ConcatMlaNopeRope(cq, to, tn, tr);
      cpu.DestroyQueue(cq);
    }
    REQUIRE(ref != seed);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kConcatMlaNopeRope, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dn(dev, q, static_cast<size_t>(kTokens * kHeads * kDn)),
          dr(dev, q, static_cast<size_t>(kTokens * rope_heads * kDr)),
          dout(dev, q, static_cast<size_t>(kTokens * kHeads * (kDn + kDr)));
      dn.Upload(nope);
      dr.Upload(rope);
      dout.Upload(seed);

      Tensor to =
          Tensor::Contiguous(dout.ptr(), DType::kF32, d, {kTokens, kHeads, kDn + kDr});
      Tensor tn = Tensor::Contiguous(dn.ptr(), DType::kF32, d, {kTokens, kHeads, kDn});
      Tensor tr = Tensor::Contiguous(dr.ptr(), DType::kF32, d, {kTokens, rope_heads, kDr});

      const unsigned long long hits_before = vt::GetReferenceTierHits();
      vt::ConcatMlaNopeRope(q, to, tn, tr);
      dev.Synchronize(q);
      CHECK(dout.Download() == ref);
      CHECK(vt::GetReferenceTierHits() == hits_before);

      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("GatherMlaCache gathers through the block table BIT-EXACTLY, with and without seq_starts") {
  // Upstream cache_kernels.cu:992-1064. Two requests of unequal length, a block
  // table whose page ids are shuffled, and a `seq_starts` arm — the chunked
  // prefill offset (`:1027-1029`) a kernel that ignores it is green without.
  constexpr int64_t kBatch = 2, kBlocks = 8, kBlockSize = 4, kHeadDim = 13;
  constexpr int64_t kMaxBlocks = 4, kTotal = 9;  // 5 + 4

  const std::vector<float> cache =
      RandomVec(static_cast<size_t>(kBlocks * kBlockSize * kHeadDim), 27301);
  const std::vector<float> seed(static_cast<size_t>(kTotal * kHeadDim), -3.75f);

  // Shuffled page ids: a kernel that read block `i` for logical block `i`
  // passes on an identity table.
  const std::vector<int32_t> block_table = {6, 1, 4, 0,   //
                                            3, 7, 2, 5};
  const std::vector<int32_t> cu_seq_lens = {0, 5, 9};
  const std::vector<int32_t> token_to_seq = {0, 0, 0, 0, 0, 1, 1, 1, 1};

  for (bool with_starts : {false, true}) {
    CAPTURE(with_starts);
    const std::vector<int32_t> starts = {2, 1};

    std::vector<float> ref(seed);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> c = cache;
      std::vector<int32_t> bt = block_table, cs = cu_seq_lens, t2s = token_to_seq,
                           st = starts;
      Tensor td = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kTotal, kHeadDim});
      Tensor tc =
          Tensor::Contiguous(c.data(), DType::kF32, cd, {kBlocks, kBlockSize, kHeadDim});
      Tensor tb = Tensor::Contiguous(bt.data(), DType::kI32, cd, {kBatch, kMaxBlocks});
      Tensor tq = Tensor::Contiguous(cs.data(), DType::kI32, cd, {kBatch + 1});
      Tensor tt = Tensor::Contiguous(t2s.data(), DType::kI32, cd, {kTotal});
      Tensor ts = Tensor::Contiguous(st.data(), DType::kI32, cd, {kBatch});
      vt::GatherMlaCache(cq, td, tc, tb, tq, tt, with_starts ? &ts : nullptr, kTotal);
      cpu.DestroyQueue(cq);
    }
    REQUIRE(ref != seed);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kGatherMlaCache, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dc(dev, q, static_cast<size_t>(kBlocks * kBlockSize * kHeadDim)),
          dd(dev, q, static_cast<size_t>(kTotal * kHeadDim));
      dc.Upload(cache);
      dd.Upload(seed);
      void* dbt = dev.Alloc(block_table.size() * sizeof(int32_t));
      void* dcs = dev.Alloc(cu_seq_lens.size() * sizeof(int32_t));
      void* dt2 = dev.Alloc(token_to_seq.size() * sizeof(int32_t));
      void* dst = dev.Alloc(starts.size() * sizeof(int32_t));
      dev.Copy(q, dbt, block_table.data(), block_table.size() * sizeof(int32_t));
      dev.Copy(q, dcs, cu_seq_lens.data(), cu_seq_lens.size() * sizeof(int32_t));
      dev.Copy(q, dt2, token_to_seq.data(), token_to_seq.size() * sizeof(int32_t));
      dev.Copy(q, dst, starts.data(), starts.size() * sizeof(int32_t));
      dev.Synchronize(q);

      Tensor td = Tensor::Contiguous(dd.ptr(), DType::kF32, d, {kTotal, kHeadDim});
      Tensor tc =
          Tensor::Contiguous(dc.ptr(), DType::kF32, d, {kBlocks, kBlockSize, kHeadDim});
      Tensor tb = Tensor::Contiguous(dbt, DType::kI32, d, {kBatch, kMaxBlocks});
      Tensor tq = Tensor::Contiguous(dcs, DType::kI32, d, {kBatch + 1});
      Tensor tt = Tensor::Contiguous(dt2, DType::kI32, d, {kTotal});
      Tensor ts = Tensor::Contiguous(dst, DType::kI32, d, {kBatch});

      const unsigned long long hits_before = vt::GetReferenceTierHits();
      vt::GatherMlaCache(q, td, tc, tb, tq, tt, with_starts ? &ts : nullptr, kTotal);
      dev.Synchronize(q);
      CHECK(dd.Download() == ref);
      CHECK(vt::GetReferenceTierHits() == hits_before);

      dev.Free(dbt);
      dev.Free(dcs);
      dev.Free(dt2);
      dev.Free(dst);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("BatchedMatmul matches the CPU oracle within NMSE <= 5e-4, dense and strided") {
  // vt::BatchedMatmul is `torch.bmm` at mla_attention.py:789 (W_UK absorption)
  // and :1034 (W_UV up-projection). BOTH upstream call sites pass TRANSPOSED
  // views, which is why the wrapper constrains only the innermost stride — so a
  // strided `a` runs here too, and a kernel that assumed a dense batch stride
  // is green on the dense arm alone.
  constexpr int64_t kG = 3, kM = 5, kK = 9, kN = 7;

  const std::vector<float> a0 = RandomVec(static_cast<size_t>(kG * kM * kK), 27401);
  const std::vector<float> b0 = RandomVec(static_cast<size_t>(kG * kK * kN), 27402);
  // `a` is viewed with a PADDED row stride: kK + 3 elements between rows, so the
  // batch stride is kM * (kK + 3) and neither is the shape.
  constexpr int64_t kARowStride = kK + 3;
  const std::vector<float> a_pad = RandomVec(static_cast<size_t>(kG * kM * kARowStride), 27403);

  for (bool strided : {false, true}) {
    CAPTURE(strided);
    std::vector<float> ref(static_cast<size_t>(kG * kM * kN), 0.0f);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> a = strided ? a_pad : a0, b = b0;
      Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kG, kM, kN});
      Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, cd, {kG, kK, kN});
      // The padded-row view. `Contiguous` on [G, M, kARowStride] already carries
      // strides {M*kARowStride, kARowStride, 1}; narrowing the last extent to kK
      // leaves those strides in place, which is exactly the transposed-view
      // shape upstream passes. There is no Tensor::Strided factory.
      Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, cd,
                                     {kG, kM, strided ? kARowStride : kK});
      ta.shape[2] = kK;
      vt::BatchedMatmul(cq, to, ta, tb);
      cpu.DestroyQueue(cq);
    }
    double mag = 0.0;
    for (float v : ref) mag += std::fabs(static_cast<double>(v));
    REQUIRE(mag > 1.0);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kBatchedMatmul, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      const size_t an = strided ? static_cast<size_t>(kG * kM * kARowStride)
                                : static_cast<size_t>(kG * kM * kK);
      DevBuf da(dev, q, an), db(dev, q, static_cast<size_t>(kG * kK * kN)),
          dout(dev, q, static_cast<size_t>(kG * kM * kN));
      da.Upload(strided ? a_pad : a0);
      db.Upload(b0);
      dout.Upload(std::vector<float>(static_cast<size_t>(kG * kM * kN), -9.5f));

      Tensor to = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {kG, kM, kN});
      Tensor tb = Tensor::Contiguous(db.ptr(), DType::kF32, d, {kG, kK, kN});
      Tensor ta = Tensor::Contiguous(da.ptr(), DType::kF32, d,
                                     {kG, kM, strided ? kARowStride : kK});
      ta.shape[2] = kK;

      const unsigned long long hits_before = vt::GetReferenceTierHits();
      vt::BatchedMatmul(q, to, ta, tb);
      dev.Synchronize(q);
      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);
      CHECK(vt::GetReferenceTierHits() == hits_before);

      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("what an unregistered ROCm op actually does on this board: tier, or refusal") {
  // #2715 says the eight MLA/DSA ops "do not refuse; they install CPU host
  // kernels and run" on gfx1151, so the damage is speed. That was TRUE of the
  // tree it was written against and this case exists because it may no longer
  // be. `6b97a6800` (#2511, "stop giving migratable memory to a part that
  // cannot fault and recover") narrowed the managed-allocation branch to
  // `PageableMemoryAccess == 1`, and made the unified claim FOLLOW the
  // allocator (include/vt/rocm/rocm_arch.h:180-195). gfx1151 reports
  // `PageableMemoryAccess = 0`. On a tree carrying that commit the tier is
  // therefore WITHDRAWN on this board, and a missing op is a REFUSAL, not a
  // slow path — which would make W2 and W3 blocking for GENERATION, not only
  // for measurement.
  //
  // This case does not decide that by reading the source. It asks the seam, on
  // whatever board it runs on, and asserts the consequence EITHER WAY, so it is
  // a measurement and not a restatement. `ReferenceTierEligible` is the public
  // safety gate (include/vt/op_provider.h:248-252) and it is side-effect free.
  bool rocm_built = false;
  for (DeviceType dt : RegisteredDevices()) {
    if (dt == DeviceType::kROCM) rocm_built = true;
  }
  if (!rocm_built) return;

  vt::Backend& dev = vt::GetBackend(DeviceType::kROCM);
  const bool eligible = vt::ReferenceTierEligible(DeviceType::kROCM);
  // Built as one std::string first: MESSAGE expands to `mb * __VA_ARGS__`, so a
  // multi-term `+` chain binds against the builder rather than the string.
  const std::string tier =
      std::string("ROCm reference tier: UnifiedMemory=") +
      (dev.UnifiedMemory() ? "1" : "0") + " DeviceMemoryIsHostAddressable=" +
      (dev.DeviceMemoryIsHostAddressable() ? "1" : "0") + " ReferenceTierEligible=" +
      (eligible ? "1" : "0");
  MESSAGE(tier);

  // The probe op must be one ROCm does not register. It USED to be
  // `kMlaDecodeAttention`; #2926 registered that one, so the probe moved to
  // `kDsaIndexerLogits`, which is still owed. When that one lands too there is
  // nothing left to probe here and this case has no business failing for it.
  //
  // MOVING THE PROBE IS NOT A WEAKENING. The quantity under test is what the
  // BOARD does with an unregistered op, not which op happens to be
  // unregistered, so any op with no ROCm kernel answers the same question.
  if (vt::OpRegistered(vt::OpId::kDsaIndexerLogits, DeviceType::kROCM)) {
    MESSAGE("kDsaIndexerLogits is now registered on ROCm — nothing left to probe");
    return;
  }

  const unsigned long long before = vt::GetReferenceTierHits();
  if (eligible) {
    // The tier is live: the miss installs a host kernel and counts itself.
    CHECK_NOTHROW((void)vt::GetOp(vt::OpId::kDsaIndexerLogits, DeviceType::kROCM));
    CHECK(vt::GetReferenceTierHits() > before);
    MESSAGE("VERDICT: the tier is LIVE here — the missing MLA ops are a SPEED cost");
  } else {
    // No tier: `GetOp` refuses by name rather than handing a host kernel a
    // pointer the host may not dereference.
    CHECK_THROWS((void)vt::GetOp(vt::OpId::kDsaIndexerLogits, DeviceType::kROCM));
    CHECK(vt::GetReferenceTierHits() == before);
    MESSAGE("VERDICT: the tier is WITHDRAWN here — the missing MLA ops are a REFUSAL, "
            "so W2/W3 block GENERATION and not only measurement");
  }
}

// ─── #2926: the two attention ops GLM-5.3 non-flash reaches ─────────────────
//
// These are the two ops that, before `src/vt/rocm/rocm_mla_attn.hip`, had no
// ROCm registration and therefore REFUSED on `gfx1151` — the reference tier is
// withdrawn on a part reporting `PageableMemoryAccess = 0`
// (`docs/ROCM.md:83-85`), so an unregistered op is not a slow path. The
// oracle-equality assertions below are the numerics; the `OpRegistered` case
// further down is what can tell a native kernel from the tier, because on a
// board where the tier IS live it computes the same answer on the host.

TEST_CASE("MlaPrefillAttention matches the CPU oracle: causal, non-causal, windowed") {
  // UNEQUAL query and key lengths on the FIRST request, which is the whole
  // point. FlashAttention's causal mask is BOTTOM-RIGHT aligned — query `i`
  // sees keys `j <= i + (len_k - len_q)` (`flash_attn.py:223`,
  // `cpu_mla_prefill.cpp`'s `causal_shift`) — so a TOP-LEFT implementation is
  // green on a square request and wrong here. Request 0 is 3 queries over 5
  // keys (shift 2); request 1 is 2 over 2 (shift 0), so both alignments are in
  // the same call.
  constexpr int64_t kHeads = 3, kQkDim = 9, kVDim = 6;
  constexpr int64_t kTotalQ = 5, kTotalK = 7, kReqs = 2;
  const std::vector<int32_t> cu_q = {0, 3, 5};
  const std::vector<int32_t> cu_k = {0, 5, 7};

  const std::vector<float> query = RandomVec(static_cast<size_t>(kTotalQ * kHeads * kQkDim), 51001);
  const std::vector<float> key = RandomVec(static_cast<size_t>(kTotalK * kHeads * kQkDim), 51002);
  const std::vector<float> value = RandomVec(static_cast<size_t>(kTotalK * kHeads * kVDim), 51003);
  const std::vector<float> out_seed(static_cast<size_t>(kTotalQ * kHeads * kVDim), -7.25f);
  const std::vector<float> lse_seed(static_cast<size_t>(kHeads * kTotalQ), -7.25f);

  // arm 0 causal, arm 1 NON-causal (the context-chunk call, "Context is
  // unmasked", `flash_attn.py:246`), arm 2 causal + sliding window. A window
  // with `causal=false` is refused by ops.cpp, so the pair is not spelled.
  for (int arm = 0; arm < 3; ++arm) {
    CAPTURE(arm);
    vt::MlaPrefillAttentionArgs args;
    args.scale = 0.37f;
    args.causal = arm != 1;
    args.max_seqlen_q = 3;
    args.max_seqlen_k = 5;
    if (arm == 2) args.window_size = vt::AttentionWindow{1, 0};

    std::vector<float> ref(out_seed), ref_lse(lse_seed);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> qh = query, kh = key, vh = value;
      std::vector<int32_t> cqh = cu_q, ckh = cu_k;
      Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kTotalQ, kHeads, kVDim});
      Tensor tl = Tensor::Contiguous(ref_lse.data(), DType::kF32, cd, {kHeads, kTotalQ});
      Tensor tq = Tensor::Contiguous(qh.data(), DType::kF32, cd, {kTotalQ, kHeads, kQkDim});
      Tensor tk = Tensor::Contiguous(kh.data(), DType::kF32, cd, {kTotalK, kHeads, kQkDim});
      Tensor tv = Tensor::Contiguous(vh.data(), DType::kF32, cd, {kTotalK, kHeads, kVDim});
      Tensor tcq = Tensor::Contiguous(cqh.data(), DType::kI32, cd, {kReqs + 1});
      Tensor tck = Tensor::Contiguous(ckh.data(), DType::kI32, cd, {kReqs + 1});
      vt::MlaPrefillAttention(cq, to, &tl, tq, tk, tv, tcq, tck, args);
      cpu.DestroyQueue(cq);
    }
    // The oracle actually wrote: a seeded buffer that came back unchanged would
    // make every comparison below vacuous.
    REQUIRE(ref != out_seed);
    REQUIRE(ref_lse != lse_seed);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kMlaPrefillAttention, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dq(dev, q, query.size()), dk(dev, q, key.size()), dv(dev, q, value.size()),
          dout(dev, q, out_seed.size()), dlse(dev, q, lse_seed.size());
      dq.Upload(query);
      dk.Upload(key);
      dv.Upload(value);
      dout.Upload(out_seed);
      dlse.Upload(lse_seed);
      void* dcq = dev.Alloc(cu_q.size() * sizeof(int32_t));
      void* dck = dev.Alloc(cu_k.size() * sizeof(int32_t));
      dev.Copy(q, dcq, cu_q.data(), cu_q.size() * sizeof(int32_t));
      dev.Copy(q, dck, cu_k.data(), cu_k.size() * sizeof(int32_t));
      dev.Synchronize(q);

      Tensor to = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {kTotalQ, kHeads, kVDim});
      Tensor tl = Tensor::Contiguous(dlse.ptr(), DType::kF32, d, {kHeads, kTotalQ});
      Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {kTotalQ, kHeads, kQkDim});
      Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {kTotalK, kHeads, kQkDim});
      Tensor tv = Tensor::Contiguous(dv.ptr(), DType::kF32, d, {kTotalK, kHeads, kVDim});
      Tensor tcq = Tensor::Contiguous(dcq, DType::kI32, d, {kReqs + 1});
      Tensor tck = Tensor::Contiguous(dck, DType::kI32, d, {kReqs + 1});

      const unsigned long long hits_before = vt::GetReferenceTierHits();
      vt::MlaPrefillAttention(q, to, &tl, tq, tk, tv, tcq, tck, args);
      dev.Synchronize(q);
      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);
      CHECK(Nmse(ref_lse, dlse.Download()) <= kNmseTol);
      CHECK(vt::GetReferenceTierHits() == hits_before);

      dev.Free(dcq);
      dev.Free(dck);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("MlaDecodeAttention matches the CPU oracle: dense, windowed, selected, sink") {
  // A SHUFFLED block table, so a kernel that read page `i` for logical block `i`
  // is green on an identity table and wrong here. `kVDim < kHeadSize` is the MLA
  // shape: the QK dot spans the FULL row and V is its LEADING slice
  // (`triton_mla.py:236`).
  constexpr int64_t kBatch = 2, kHeads = 3, kHeadSize = 9, kVDim = 6;
  constexpr int64_t kBlocks = 8, kBlockSize = 4, kMaxBlocks = 2;
  const std::vector<int32_t> block_table = {6, 1,   //
                                            3, 7};
  const std::vector<int32_t> seq_lens = {7, 3};
  constexpr int64_t kTopk = 4;

  const std::vector<float> query =
      RandomVec(static_cast<size_t>(kBatch * kHeads * kHeadSize), 52001);
  const std::vector<float> cache =
      RandomVec(static_cast<size_t>(kBlocks * kBlockSize * kHeadSize), 52002);
  const std::vector<float> sink = RandomVec(static_cast<size_t>(kHeads), 52003);
  const std::vector<float> out_seed(static_cast<size_t>(kBatch * kHeads * kVDim), -6.5f);
  const std::vector<float> lse_seed(static_cast<size_t>(kBatch * kHeads), -6.5f);

  // A `-1` INSIDE the live count (upstream's "no token" sentinel,
  // sparse_attn_indexer.py:431-432) and a count SHORTER than the row, so both
  // the sentinel skip and the count clamp are exercised.
  const std::vector<int32_t> sel = {5, 1, -1, 3,   //
                                    0, 2, -1, -1};
  const std::vector<int32_t> sel_cnt = {4, 2};

  // arm 0 dense, 1 windowed, 2 selected, 3 sink, 4 selected + sink.
  for (int arm = 0; arm < 5; ++arm) {
    CAPTURE(arm);
    const bool selected = arm == 2 || arm == 4;
    const bool sinked = arm == 3 || arm == 4;

    // Shared, POINTER-FREE settings. The two Tensor* members are set per side
    // (host oracle / device) against tensors that live on that side, so no
    // pointer here ever outlives its tensor.
    vt::MlaDecodeAttentionArgs base;
    base.scale = 0.41f;
    base.num_kv_splits = 0;
    base.max_seq_len = 7;
    if (arm == 1) base.window_size = vt::AttentionWindow{2, 0};

    std::vector<float> ref(out_seed), ref_lse(lse_seed);
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> qh = query, ch = cache, sh = sink;
      std::vector<int32_t> bth = block_table, slh = seq_lens, selh = sel, cnth = sel_cnt;
      Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kBatch, kHeads, kVDim});
      Tensor tl = Tensor::Contiguous(ref_lse.data(), DType::kF32, cd, {kBatch, kHeads});
      Tensor tq =
          Tensor::Contiguous(qh.data(), DType::kF32, cd, {kBatch, kHeads, kHeadSize});
      Tensor tc =
          Tensor::Contiguous(ch.data(), DType::kF32, cd, {kBlocks, kBlockSize, kHeadSize});
      Tensor tb = Tensor::Contiguous(bth.data(), DType::kI32, cd, {kBatch, kMaxBlocks});
      Tensor ts = Tensor::Contiguous(slh.data(), DType::kI32, cd, {kBatch});
      Tensor ti = Tensor::Contiguous(selh.data(), DType::kI32, cd, {kBatch, kTopk});
      Tensor tn = Tensor::Contiguous(cnth.data(), DType::kI32, cd, {kBatch});
      Tensor tk = Tensor::Contiguous(sh.data(), DType::kF32, cd, {kHeads});
      vt::MlaDecodeAttentionArgs cargs = base;
      if (selected) {
        cargs.topk_indices = &ti;
        cargs.valid_counts = &tn;
      }
      if (sinked) cargs.attn_sink = &tk;
      vt::MlaDecodeAttention(cq, to, &tl, tq, tc, tb, ts, cargs);
      cpu.DestroyQueue(cq);
    }
    // The oracle actually wrote: a seeded buffer that came back unchanged would
    // make every comparison below vacuous.
    REQUIRE(ref != out_seed);
    REQUIRE(ref_lse != lse_seed);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kMlaDecodeAttention, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dq(dev, q, query.size()), dc(dev, q, cache.size()),
          dout(dev, q, out_seed.size()), dlse(dev, q, lse_seed.size()),
          dsink(dev, q, sink.size());
      dq.Upload(query);
      dc.Upload(cache);
      dout.Upload(out_seed);
      dlse.Upload(lse_seed);
      dsink.Upload(sink);
      void* dbt = dev.Alloc(block_table.size() * sizeof(int32_t));
      void* dsl = dev.Alloc(seq_lens.size() * sizeof(int32_t));
      void* dsel = dev.Alloc(sel.size() * sizeof(int32_t));
      void* dcnt = dev.Alloc(sel_cnt.size() * sizeof(int32_t));
      dev.Copy(q, dbt, block_table.data(), block_table.size() * sizeof(int32_t));
      dev.Copy(q, dsl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
      dev.Copy(q, dsel, sel.data(), sel.size() * sizeof(int32_t));
      dev.Copy(q, dcnt, sel_cnt.data(), sel_cnt.size() * sizeof(int32_t));
      dev.Synchronize(q);

      Tensor to = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {kBatch, kHeads, kVDim});
      Tensor tl = Tensor::Contiguous(dlse.ptr(), DType::kF32, d, {kBatch, kHeads});
      Tensor tq =
          Tensor::Contiguous(dq.ptr(), DType::kF32, d, {kBatch, kHeads, kHeadSize});
      Tensor tc =
          Tensor::Contiguous(dc.ptr(), DType::kF32, d, {kBlocks, kBlockSize, kHeadSize});
      Tensor tb = Tensor::Contiguous(dbt, DType::kI32, d, {kBatch, kMaxBlocks});
      Tensor ts = Tensor::Contiguous(dsl, DType::kI32, d, {kBatch});
      Tensor ti = Tensor::Contiguous(dsel, DType::kI32, d, {kBatch, kTopk});
      Tensor tn = Tensor::Contiguous(dcnt, DType::kI32, d, {kBatch});
      Tensor tk = Tensor::Contiguous(dsink.ptr(), DType::kF32, d, {kHeads});
      vt::MlaDecodeAttentionArgs dargs = base;
      if (selected) {
        dargs.topk_indices = &ti;
        dargs.valid_counts = &tn;
      }
      if (sinked) dargs.attn_sink = &tk;

      const unsigned long long hits_before = vt::GetReferenceTierHits();
      vt::MlaDecodeAttention(q, to, &tl, tq, tc, tb, ts, dargs);
      dev.Synchronize(q);
      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);
      CHECK(Nmse(ref_lse, dlse.Download()) <= kNmseTol);
      CHECK(vt::GetReferenceTierHits() == hits_before);

      // THE FULL-SELECTION IDENTITY, on the DENSE arm's own device and asserted
      // BIT FOR BIT. `ops.h` states it as a contract: "A SELECTION LISTING EVERY
      // CAUSAL KEY IS BIT-FOR-BIT the unselected call", because the list is
      // walked in ascending order and the f32 online softmax therefore sees the
      // identical summation order. An NMSE bound against the CPU oracle cannot
      // see a reordered reduction, which is the whole thing this identity is
      // about — so it is a device-against-itself equality and not an oracle
      // comparison.
      //
      // Only REQUEST 1 is compared: its 3 keys fit inside `topk == 4`, so its
      // list can name every causal key. Request 0 has 7 keys and cannot, and its
      // rows are EXCLUDED rather than fudged into the bound.
      if (arm == 0) {
        const std::vector<float> dense = dout.Download();
        const std::vector<int32_t> full = {0, 1, 2, 3,   // request 0: truncated, unused
                                           0, 1, 2, -1};
        const std::vector<int32_t> full_cnt = {4, 3};
        dev.Copy(q, dsel, full.data(), full.size() * sizeof(int32_t));
        dev.Copy(q, dcnt, full_cnt.data(), full_cnt.size() * sizeof(int32_t));
        dout.Upload(out_seed);
        dev.Synchronize(q);
        vt::MlaDecodeAttentionArgs fargs = base;
        fargs.topk_indices = &ti;
        fargs.valid_counts = &tn;
        vt::MlaDecodeAttention(q, to, &tl, tq, tc, tb, ts, fargs);
        dev.Synchronize(q);
        const std::vector<float> got_full = dout.Download();
        for (size_t i = static_cast<size_t>(kHeads * kVDim); i < got_full.size(); ++i) {
          CHECK(got_full[i] == dense[i]);
        }
      }

      dev.Free(dbt);
      dev.Free(dsl);
      dev.Free(dsel);
      dev.Free(dcnt);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("ROCm registers the two attention ops GLM-5.3 non-flash reaches (#2926)") {
  // The load-bearing assertion of this wave, and the one the numeric cases
  // above CANNOT make. `OpRegistered` is a native-only probe
  // (`src/vt/op_provider.cpp:788-806`): on a board where the portable reference
  // tier is live it is FALSE while the op still computes the right answer on
  // the host, so an oracle-equality assertion is green with no kernel at all.
  // That is measured, not argued — four of #2715's RED cases passed on their
  // `REQUIRE` guards alone.
  //
  // On `gfx1151` the tier is withdrawn (`docs/ROCM.md:83-85`), so a false here
  // is not a slow path: it is GLM-5.3 non-flash unable to emit a token.
  bool rocm_built = false;
  for (DeviceType dt : RegisteredDevices()) {
    if (dt == DeviceType::kROCM) rocm_built = true;
  }
  if (!rocm_built) return;  // not a ROCm build — nothing to assert about ROCm

  CHECK(vt::OpRegistered(vt::OpId::kMlaPrefillAttention, DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kMlaDecodeAttention, DeviceType::kROCM));

  // Still owed after this wave (spec § Owed). RECORDED, not asserted: asserting
  // their absence would be a lock on the next wave. A SPARSE step — a prompt
  // longer than `index_topk` — still refuses while either is 0.
  const std::string owed =
      std::string("still owed on ROCm (#2926) — kDsaIndexerLogits=") +
      (vt::OpRegistered(vt::OpId::kDsaIndexerLogits, DeviceType::kROCM) ? "1" : "0") +
      " kDsaTopkSelect=" +
      (vt::OpRegistered(vt::OpId::kDsaTopkSelect, DeviceType::kROCM) ? "1" : "0") +
      " kMergeAttnStates=" +
      (vt::OpRegistered(vt::OpId::kMergeAttnStates, DeviceType::kROCM) ? "1" : "0") +
      " — a SPARSE GLM-5.3 step still refuses while the indexer pair is 0, and "
      "the speed axis stays VOID";
  MESSAGE(owed);
}
