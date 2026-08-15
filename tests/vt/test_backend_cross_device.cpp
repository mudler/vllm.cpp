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

#include "support/test_env.h"  // SetEnv/UnsetEnv — MSVC has no setenv (#603)
#include "vt/backend.h"
#include "vt/op_provider.h"
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
      CAPTURE(DeviceName(dt));
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
