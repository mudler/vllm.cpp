// M-BLOCKING BYTE-EQUALITY GATE for the specialized elementwise CPU GEMM
// (row LTX25-CONNECTOR-REPAIR, .agents/specs/ltx25-connector-repair.md).
//
// WHAT IS BEING GATED. `MatmulOneChunk` (src/vt/cpu/cpu_ops.cpp) widens a
// 16-row activation tile and walks it in blocks of `mr` rows, where `mr` is the
// tier's M-blocking factor (8 on AVX-512, 6 on NEON, 2 on SSE2, 4 on the
// portable [K,N] path). Every row of the tile now goes through that M-blocked
// micro-kernel: the tile is PADDED up to a whole number of `mr` blocks and the
// pad rows are zeroed, so a row whose index falls in the ragged remainder is
// computed by `btm`/`nkm` beside its neighbours instead of by the one-row
// `bt`/`nk` kernel.
//
// The claim that makes that admissible is that it moves NO output's
// accumulation order: `ElemBtMFn` accumulates lane `l` of row `r` over `p` in
// strict increasing order whatever `mr` is, exactly as `ElemBt16Fn` does for a
// single row. So the gate is `memcmp` against an independent scalar reference,
// never a tolerance.
//
// WHY THIS FILE EXISTS BESIDE test_ops_matmul_elem.cpp. That file gates the
// tier against the reference over the dtype matrix at ONE worker count, and its
// thread-count case runs one shape in one dtype. The change here decides WHICH
// ROWS SHARE ONE ACCUMULATOR SET, and which rows a chunk covers is set by the
// threadpool's partition. A single worker count exercises a single partition,
// so a partition-dependent defect survives it. VT-CPU-ELEM-SURVEY's M9 is what
// that rule was paid for: a deliberately broken kernel passed 753,300
// assertions at VLLM_CPP_CPU_THREADS=1 and only the worker-count case caught it.
//
// THE OPERANDS ARE DELIBERATELY ILL-SCALED. The same row's M2 mutation survived
// its first gate because two operands shared one scale, so their sums fit in 8
// significant bits -- which bf16 holds exactly, making the assertion unable to
// fail. A and B here are filled from independent streams at magnitudes three
// decades apart, so a reordered or dropped addend changes the stored bytes.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/cpu/cpu_matmul_elem.h"  // via -I src: kElemLanes, ElemGemmTier
#include "vt/cpu/cpu_threadpool.h"   // Threadpool::SwapForTesting

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  // `scale` puts the two operands three decades apart. See the header note.
  float Uniform(float scale) {
    return (static_cast<float>(Next() % 20001) / 10000.0f - 1.0f) * scale;
  }
};

struct Buf {
  DType dtype;
  std::vector<uint8_t> bytes;
  std::vector<float> ref;  // the EXACT f32 value of every stored element

  Buf(DType dt, int64_t n, Rng& rng, float scale) : dtype(dt) {
    ref.resize(static_cast<size_t>(n));
    bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
    for (int64_t i = 0; i < n; ++i) {
      const float v = rng.Uniform(scale);
      switch (dt) {
        case DType::kF32:
          reinterpret_cast<float*>(bytes.data())[i] = v;
          ref[i] = v;
          break;
        case DType::kF16: {
          const uint16_t h = vt::F32ToF16(v);
          reinterpret_cast<uint16_t*>(bytes.data())[i] = h;
          ref[i] = vt::F16ToF32(h);
          break;
        }
        case DType::kBF16: {
          const uint16_t h = vt::F32ToBF16(v);
          reinterpret_cast<uint16_t*>(bytes.data())[i] = h;
          ref[i] = vt::BF16ToF32(h);
          break;
        }
        default:
          break;
      }
    }
  }
  void* Data() { return bytes.data(); }
};

// Independent scalar oracle. One accumulator, strictly sequential over p, the
// product rounded before the add (-ffp-contract=off, CMakeLists.txt). It shares
// no code with src/vt/cpu: a gate that compared the kernel against a helper the
// kernel also uses would prove consistency, not correctness.
void RefGemm(bool bt, int64_t m, int64_t n, int64_t k, const std::vector<float>& a,
             const std::vector<float>& b, DType out_dt, std::vector<uint8_t>* out) {
  out->assign(static_cast<size_t>(m * n) * vt::SizeOf(out_dt), 0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += a[static_cast<size_t>(i * k + p)] *
               b[static_cast<size_t>(bt ? j * k + p : p * n + j)];
      }
      const int64_t o = i * n + j;
      if (out_dt == DType::kF32) {
        reinterpret_cast<float*>(out->data())[o] = acc;
      } else {
        reinterpret_cast<uint16_t*>(out->data())[o] = vt::F32ToBF16(acc);
      }
    }
  }
}

struct Shape {
  int64_t m, n, k;
};

// M straddles every `mr` a shipped tier uses (2, 4, 6, 8) AND the 16-row tile
// boundary, so each case leaves a different-sized ragged remainder for the
// padding to absorb: 16 % 6 = 4 and 16 % 8 = 0 on the two x86 tiers, 16 % 4 = 0
// and 16 % 6 = 4 on the two Arm ones. N covers a full 16-lane column block, two
// of them, and a ragged tail that must stay on the scalar column path. K covers
// the SIMD step boundary and the remainders that fall off it.
const std::vector<Shape>& Shapes() {
  static const std::vector<Shape> s = {
      {1, 16, 64},   {2, 16, 64},   {5, 16, 64},   {6, 16, 64},   {7, 16, 64},
      {8, 16, 64},   {9, 16, 64},   {11, 16, 64},  {12, 16, 64},  {13, 16, 64},
      {15, 16, 64},  {16, 16, 64},  {17, 16, 64},  {18, 16, 64},  {23, 32, 65},
      {24, 32, 65},  {25, 32, 65},  {31, 48, 17},  {33, 48, 17},  {16, 23, 64},
      {17, 7, 66},   {34, 33, 129}, {64, 32, 16},  {19, 16, 1},
  };
  return s;
}

bool RunOne(bool bt, DType adt, DType bdt, DType odt, const Shape& s, uint64_t seed) {
  Queue q = CpuQueue();
  Rng rng_a(seed);
  Rng rng_b(seed ^ 0x9e3779b97f4a7c15ULL);
  Buf a(adt, s.m * s.k, rng_a, 1.0f);
  Buf b(bdt, s.n * s.k, rng_b, 0.001f);
  std::vector<uint8_t> got(static_cast<size_t>(s.m * s.n) * vt::SizeOf(odt), 0xAB);
  Tensor ta = Tensor::Contiguous(a.Data(), adt, Cpu(), {s.m, s.k});
  Tensor tb = bt ? Tensor::Contiguous(b.Data(), bdt, Cpu(), {s.n, s.k})
                 : Tensor::Contiguous(b.Data(), bdt, Cpu(), {s.k, s.n});
  Tensor to = Tensor::Contiguous(got.data(), odt, Cpu(), {s.m, s.n});
  if (bt) {
    vt::MatmulBT(q, to, ta, tb);
  } else {
    vt::Matmul(q, to, ta, tb);
  }
  std::vector<uint8_t> want;
  RefGemm(bt, s.m, s.n, s.k, a.ref, b.ref, odt, &want);
  return std::memcmp(got.data(), want.data(), want.size()) == 0;
}

}  // namespace

// T1 + T2: the whole operand dtype matrix, both orientations, at M values that
// are not multiples of any tier's `mr`, with ragged N and ragged K beside them.
TEST_CASE("elementwise CPU GEMM M blocking: byte-identical over the dtype matrix at ragged M") {
  const DType kElem[3] = {DType::kF32, DType::kF16, DType::kBF16};
  const DType kOut[2] = {DType::kF32, DType::kBF16};
  uint64_t seed = 101;
  for (bool bt : {false, true}) {
    for (DType adt : kElem) {
      for (DType bdt : kElem) {
        for (DType odt : kOut) {
          for (const Shape& s : Shapes()) {
            CHECK(RunOne(bt, adt, bdt, odt, s, seed++));
          }
        }
      }
    }
  }
}

// T3: the same claim at SEVEN worker counts. `ForRows`/`MatmulChunked` chunk by
// output rows, so a different worker count is a different partition and
// therefore a different grouping of rows into `mr` blocks. One worker count
// tests one grouping; this tests seven. The M values are chosen so that no two
// of these counts produce the same partition.
TEST_CASE("elementwise CPU GEMM M blocking: byte-identical at seven worker counts") {
  const DType kElem[3] = {DType::kF32, DType::kF16, DType::kBF16};
  const std::vector<Shape> shapes = {
      {64, 32, 65}, {67, 48, 64}, {129, 16, 33}, {17, 32, 128},
  };
  for (bool bt : {false, true}) {
    for (DType adt : kElem) {
      for (DType bdt : kElem) {
        for (const Shape& s : shapes) {
          std::vector<uint8_t> base;
          bool first = true;
          for (int nth : {1, 2, 3, 4, 5, 8, 20}) {
            vt::cpu::Threadpool tp(nth);
            vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
            Queue q = CpuQueue();
            Rng rng_a(9000 + s.m);
            Rng rng_b(4242 + s.n);
            Buf a(adt, s.m * s.k, rng_a, 1.0f);
            Buf b(bdt, s.n * s.k, rng_b, 0.001f);
            std::vector<uint8_t> got(static_cast<size_t>(s.m * s.n) * sizeof(float), 0xAB);
            Tensor ta = Tensor::Contiguous(a.Data(), adt, Cpu(), {s.m, s.k});
            Tensor tb = bt ? Tensor::Contiguous(b.Data(), bdt, Cpu(), {s.n, s.k})
                           : Tensor::Contiguous(b.Data(), bdt, Cpu(), {s.k, s.n});
            Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {s.m, s.n});
            if (bt) {
              vt::MatmulBT(q, to, ta, tb);
            } else {
              vt::Matmul(q, to, ta, tb);
            }
            vt::cpu::Threadpool::SwapForTesting(prev);
            // The oracle is recomputed OUTSIDE the swapped pool so the check is
            // against the scalar reference and not merely against worker-count
            // self-consistency, which would pass a kernel that is wrong the
            // same way at every count.
            std::vector<uint8_t> want;
            RefGemm(bt, s.m, s.n, s.k, a.ref, b.ref, DType::kF32, &want);
            CHECK(std::memcmp(got.data(), want.data(), want.size()) == 0);
            if (first) {
              base = got;
              first = false;
            } else {
              CHECK(std::memcmp(base.data(), got.data(), base.size()) == 0);
            }
          }
        }
      }
    }
  }
}

// The tier this process resolved is printed rather than asserted, because a
// forced tier is how the same binary is swept (VT_CPU_MATMUL_TIER=ref /
// portable / avx2 / avx512). A number quoted without the tier it ran on is not
// a measurement, and neither is a PASS.
TEST_CASE("elementwise CPU GEMM M blocking: the tier and mr that actually ran are reported") {
  const vt::cpu::ElemGemmTierTable& t = vt::cpu::ElemGemmTier();
  // std::string, not the bare pointer: doctest stringifies a `const char*` as
  // a BOOL, so the bare form prints "tier=1" and hides the tier this ran on.
  MESSAGE("tier=" << std::string(t.name) << " mr=" << t.mr
                  << " lanes=" << vt::cpu::kElemLanes);
  CHECK(t.mr >= 1);
  CHECK(t.name != nullptr);
}
