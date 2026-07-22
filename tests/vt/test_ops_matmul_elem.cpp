// Bit-exactness + determinism gate for the specialized/vectorized elementwise
// CPU GEMM (src/vt/cpu/cpu_matmul_elem.{h,cpp}, row CPU-ELEM-GEMM,
// .agents/specs/cpu-elementwise-gemm.md).
//
// The whole correctness claim of that change is that vectorizing ACROSS OUTPUT
// COLUMNS (rather than along K, as llama.cpp's ggml_vec_dot_bf16 /
// ggml_vec_dot_f16 do — vec.cpp:139,264) leaves every output element's f32
// reduction in exactly the order the historical one-accumulator scalar kernel
// used. So the gate is `memcmp`, not `Approx`: the op's output must be
// BYTE-IDENTICAL to an independent in-test scalar reference that mirrors
// MatmulOneChunkRef, across every dtype combination, both weight orientations,
// ragged shapes (K not a multiple of the 4-wide SIMD step, N not a multiple of
// the 16 output lanes) and thread counts.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/cpu/cpu_matmul_elem.h"    // via -I src
#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

// Deterministic LCG so every case is reproducible without a seed corpus.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float Uniform() { return static_cast<float>(Next() % 20001) / 10000.0f - 1.0f; }
};

// Storage for an operand of an arbitrary elementwise dtype.
struct Buf {
  DType dtype;
  std::vector<uint8_t> bytes;
  std::vector<float> ref;  // the exact f32 value of every element

  Buf(DType dt, int64_t n, Rng& rng) : dtype(dt) {
    ref.resize(static_cast<size_t>(n));
    bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
    for (int64_t i = 0; i < n; ++i) {
      const float v = rng.Uniform();
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

// Independent scalar reference: one accumulator, strictly sequential over p,
// product rounded before the add (the -ffp-contract=off contract).
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

// M covers the 16-row activation tile boundary; N covers the 16 output lanes
// (exact, under, over, ragged remainder); K covers the 4-wide SIMD step
// boundary and its remainders 1/2/3.
const std::vector<Shape>& Shapes() {
  static const std::vector<Shape> s = {
      {1, 16, 64},  {1, 16, 67},  {1, 17, 64},  {1, 7, 5},    {1, 48, 2048},
      {3, 16, 4},   {3, 33, 1},   {16, 16, 16}, {16, 32, 63},  {17, 16, 3},
      {33, 48, 65}, {2, 1, 129},  {4, 64, 256}, {128, 96, 128},
  };
  return s;
}

}  // namespace

TEST_CASE("elementwise CPU GEMM: bit-identical to the scalar reference, every dtype x orientation") {
  const DType kElem[3] = {DType::kF32, DType::kF16, DType::kBF16};
  const DType kOut[2] = {DType::kF32, DType::kBF16};
  Queue q = CpuQueue();
  uint64_t seed = 1;
  for (bool bt : {false, true}) {
    for (DType adt : kElem) {
      for (DType bdt : kElem) {
        for (DType odt : kOut) {
          for (const Shape& s : Shapes()) {
            Rng rng(seed++);
            Buf a(adt, s.m * s.k, rng);
            Buf b(bdt, s.n * s.k, rng);
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
            CHECK(std::memcmp(got.data(), want.data(), want.size()) == 0);
          }
        }
      }
    }
  }
}

TEST_CASE("elementwise CPU GEMM: row-strided activation stays bit-exact") {
  // vt::MatmulBT accepts a column slice of a wider buffer (MLA campaign W6).
  Queue q = CpuQueue();
  const int64_t m = 5, n = 32, k = 40, wide = 96;
  Rng rng(7777);
  Buf a(DType::kBF16, m * wide, rng);
  Buf b(DType::kBF16, n * k, rng);
  std::vector<float> a_slice(static_cast<size_t>(m * k));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t p = 0; p < k; ++p) a_slice[static_cast<size_t>(i * k + p)] = a.ref[i * wide + p];
  }
  Tensor ta = Tensor::Contiguous(a.Data(), DType::kBF16, Cpu(), {m, wide});
  ta.shape[1] = k;  // stride[0] stays `wide` — the strided-activation contract
  Tensor tb = Tensor::Contiguous(b.Data(), DType::kBF16, Cpu(), {n, k});
  std::vector<float> got(static_cast<size_t>(m * n), -1.0f);
  Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {m, n});
  vt::MatmulBT(q, to, ta, tb);
  std::vector<uint8_t> want;
  RefGemm(true, m, n, k, a_slice, b.ref, DType::kF32, &want);
  CHECK(std::memcmp(got.data(), want.data(), want.size()) == 0);
}

TEST_CASE("elementwise CPU GEMM: bit-identical across thread counts") {
  // The determinism contract (cpu_threadpool.h): parallelism partitions output
  // elements only, so the result must not depend on the worker count.
  Queue q = CpuQueue();
  const int64_t m = 33, n = 80, k = 129;
  Rng rng(4242);
  Buf a(DType::kBF16, m * k, rng);
  Buf b(DType::kBF16, n * k, rng);
  std::vector<float> base;
  for (int nth : {1, 2, 4, 8}) {
    vt::cpu::Threadpool tp(nth);
    vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
    std::vector<float> got(static_cast<size_t>(m * n), -1.0f);
    Tensor ta = Tensor::Contiguous(a.Data(), DType::kBF16, Cpu(), {m, k});
    Tensor tb = Tensor::Contiguous(b.Data(), DType::kBF16, Cpu(), {n, k});
    Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {m, n});
    vt::MatmulBT(q, to, ta, tb);
    if (base.empty()) {
      base = got;
    } else {
      CHECK(std::memcmp(base.data(), got.data(), base.size() * sizeof(float)) == 0);
    }
    vt::cpu::Threadpool::SwapForTesting(prev);
  }
}

TEST_CASE("elementwise CPU GEMM: SIMD widening matches vt::F16ToF32/BF16ToF32 exhaustively") {
  // The arch tiers widen f16 with a hardware convert and bf16 with a
  // shift-left-16 (llama.cpp vec.cpp:172 / simd-mappings.h). Both must agree
  // with vt's scalar converters over the ENTIRE 16-bit domain, or the GEMM is
  // not bit-exact. Driven through the op so the SIMD path is what runs: a row
  // of 1.0f activations turns each dot into an ordered sum of converted
  // weights, and any single differing conversion moves the sum.
  // Inf/NaN patterns (exponent 0x1F) are covered separately below because a
  // NaN would swallow the rest of the sum.
  Queue q = CpuQueue();
  const int64_t n = vt::cpu::kElemLanes;
  const int64_t k = 4096;  // n * k == 65536 == the full f16/bf16 domain
  for (DType dt : {DType::kF16, DType::kBF16}) {
    std::vector<uint16_t> w(static_cast<size_t>(n * k));
    std::vector<float> wref(static_cast<size_t>(n * k));
    for (size_t i = 0; i < w.size(); ++i) {
      uint16_t pat = static_cast<uint16_t>(i);
      const bool special = dt == DType::kF16 ? ((pat >> 10) & 0x1F) == 0x1F
                                            : ((pat >> 7) & 0xFF) == 0xFF;
      if (special) pat = 0;
      w[i] = pat;
      wref[i] = dt == DType::kF16 ? vt::F16ToF32(pat) : vt::BF16ToF32(pat);
    }
    std::vector<float> a(static_cast<size_t>(k), 1.0f);
    std::vector<float> got(static_cast<size_t>(n), -1.0f);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {1, k});
    Tensor tb = Tensor::Contiguous(w.data(), dt, Cpu(), {n, k});
    Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {1, n});
    vt::MatmulBT(q, to, ta, tb);
    std::vector<uint8_t> want;
    RefGemm(true, 1, n, k, a, wref, DType::kF32, &want);
    CHECK(std::memcmp(got.data(), want.data(), want.size()) == 0);
  }
}

TEST_CASE("elementwise CPU GEMM: inf/nan weight patterns match the scalar converters") {
  // Every exponent-all-ones pattern, one per dot so nothing is swallowed:
  // K = 4 (one full SIMD step) with the pattern in lane 0 and zeros after.
  Queue q = CpuQueue();
  const int64_t n = vt::cpu::kElemLanes, k = 4;
  for (DType dt : {DType::kF16, DType::kBF16}) {
    const int mant_bits = dt == DType::kF16 ? 10 : 7;
    std::vector<uint16_t> pats;
    for (uint32_t p = 0; p < 65536u; ++p) {
      const uint16_t v = static_cast<uint16_t>(p);
      const uint32_t exp_mask = dt == DType::kF16 ? 0x1Fu : 0xFFu;
      if (((v >> mant_bits) & exp_mask) == exp_mask) pats.push_back(v);
    }
    for (size_t base = 0; base < pats.size(); base += static_cast<size_t>(n)) {
      std::vector<uint16_t> w(static_cast<size_t>(n * k), 0);
      std::vector<float> wref(static_cast<size_t>(n * k), 0.0f);
      for (int64_t l = 0; l < n; ++l) {
        const size_t idx = base + static_cast<size_t>(l);
        const uint16_t v = idx < pats.size() ? pats[idx] : 0;
        w[static_cast<size_t>(l * k)] = v;
        wref[static_cast<size_t>(l * k)] = dt == DType::kF16 ? vt::F16ToF32(v) : vt::BF16ToF32(v);
      }
      std::vector<float> a(static_cast<size_t>(k), 1.0f);
      std::vector<float> got(static_cast<size_t>(n), -1.0f);
      Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {1, k});
      Tensor tb = Tensor::Contiguous(w.data(), dt, Cpu(), {n, k});
      Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {1, n});
      vt::MatmulBT(q, to, ta, tb);
      std::vector<uint8_t> want;
      RefGemm(true, 1, n, k, a, wref, DType::kF32, &want);
      CHECK(std::memcmp(got.data(), want.data(), want.size()) == 0);
    }
  }
}
