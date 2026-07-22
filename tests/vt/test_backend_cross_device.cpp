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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"

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
                       DeviceType::kXPU}) {
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
    setenv("VT_FUSED_TIER", tier == 0 ? "0" : "1", 1);
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
    setenv("VT_FUSED_TIER", saved.c_str(), 1);
  } else {
    unsetenv("VT_FUSED_TIER");
  }
}
