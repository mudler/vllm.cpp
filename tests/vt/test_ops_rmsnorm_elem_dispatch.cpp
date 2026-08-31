// vllm.cpp original (vt runtime). The BYTE-EQUALITY gate for the hoisted
// per-element dtype dispatch in `vt::RmsNorm` (row VT-CPU-ELEM-SURVEY,
// .agents/specs/vt-cpu-elem-survey.md).
//
// WHAT THIS FILE EXISTS TO HOLD. `RmsNormKernel` used to read every operand
// element through `LoadF32`, which switches on `t.dtype` and multiplies by
// `SizeOf(t.dtype)` once per element -- twice over `x` for the variance pass
// and the scale pass, and once over `w` PER ROW even though `w` is one [H]
// vector reused by every token. The survey measured the consequence with
// `perf record -e cpu-clock` over `tools/bench/vt_cpu_elem_survey_probe.cpp`
// at the Qwen3.6-27B input_layernorm shape (x[1024,5120] f32): `LoadF32`
// 49.06% of the process, `StoreF32` 12.05%, the kernel body 36.48%. The
// dispatch is now resolved ONCE per call into typed micro-kernels and `w` is
// widened to f32 ONCE per call.
//
// The transformation's entire claim is that it is BIT-EXACT, not close: the
// same indices are summed in the same order into the same f32 accumulators.
// A tolerance would not test that claim, so every assertion here is `memcmp`
// over the raw output bytes against a reference that reproduces the ORIGINAL
// per-element loop verbatim, written from the layout contract in
// include/vt/ops.h (the `vt::RmsNorm` block) and using nothing from src/vt/cpu.
//
// WHY THE DTYPE MATRIX IS THE POINT. `vt::RmsNorm` accepts x and weight in
// {f32, f16, bf16} (`IsFloat`) and out and residual in {f32, bf16}
// (`IsOutFloat`, src/vt/ops.cpp:25). FOUR tensors carry INDEPENDENT dtypes, so
// a typed table indexed by the wrong one is invisible whenever they agree --
// and before this row every CPU RmsNorm test ran x, w and out all f32 or all
// bf16. Every case below therefore sweeps the four dtypes independently.
//
// THE RESIDUAL STREAM IS THE SUBTLE ONE. The kernel adds x into the residual in
// f32, STORES it (rounding to the residual's dtype), and RE-READS the rounded
// value before squaring it. Skipping the re-read is bit-exact for an f32
// residual and wrong for a bf16 one, which is why every residual case runs
// both dtypes and checks the residual bytes as well as the output bytes.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormArgs;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

// ── An independent per-element reference ────────────────────────────────────
//
// This is the ORIGINAL kernel body, re-derived from the `vt::RmsNorm` contract
// in include/vt/ops.h. It deliberately shares NOTHING with src/vt/cpu: it keeps
// its own dtype switch, its own conversions and its own untyped byte buffers,
// so a defect that the kernel and a shared helper would both carry cannot hide
// here. A gate that compared the kernel against a helper the kernel also uses
// would prove consistency, not correctness.
size_t Width(DType d) { return d == DType::kF32 ? 4u : 2u; }

float RefLoad(const std::vector<uint8_t>& buf, DType d, size_t i) {
  const uint8_t* p = buf.data() + i * Width(d);
  if (d == DType::kF32) {
    float v;
    std::memcpy(&v, p, 4);
    return v;
  }
  uint16_t bits;
  std::memcpy(&bits, p, 2);
  if (d == DType::kBF16) {
    const uint32_t w = static_cast<uint32_t>(bits) << 16;
    float v;
    std::memcpy(&v, &w, 4);
    return v;
  }
  // IEEE binary16 -> binary32, written out rather than called, so this file
  // does not import the conversion the kernel uses.
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exp = (bits >> 10) & 0x1Fu;
  const uint32_t man = bits & 0x3FFu;
  uint32_t w;
  if (exp == 0) {
    if (man == 0) {
      w = sign;
    } else {
      int e = -1;
      uint32_t m = man;
      do {
        m <<= 1;
        ++e;
      } while ((m & 0x400u) == 0);
      w = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 0x1Fu) {
    w = sign | 0x7F800000u | (man << 13);
  } else {
    w = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float v;
  std::memcpy(&v, &w, 4);
  return v;
}

void RefStore(std::vector<uint8_t>& buf, DType d, size_t i, float v) {
  uint8_t* p = buf.data() + i * Width(d);
  if (d == DType::kF32) {
    std::memcpy(p, &v, 4);
    return;
  }
  uint32_t w;
  std::memcpy(&w, &v, 4);
  // round-to-nearest-even to bf16; the only narrowing store this op reaches
  // (out and residual are restricted to {f32, bf16} by IsOutFloat).
  const uint32_t rounded = w + 0x7FFFu + ((w >> 16) & 1u);
  const uint16_t bits = static_cast<uint16_t>(rounded >> 16);
  std::memcpy(p, &bits, 2);
}

// The original loop, verbatim in structure.
void RefRmsNorm(std::vector<uint8_t>& out, DType dout, const std::vector<uint8_t>& x,
                DType dx, const std::vector<uint8_t>& w, DType dw, int64_t t, int64_t h,
                float eps, bool gemma, std::vector<uint8_t>* residual, DType dres) {
  for (int64_t i = 0; i < t; ++i) {
    const int64_t rbase = i * h;
    float sumsq = 0.0f;
    for (int64_t j = 0; j < h; ++j) {
      float v = RefLoad(x, dx, static_cast<size_t>(i * h + j));
      if (residual != nullptr) {
        v += RefLoad(*residual, dres, static_cast<size_t>(rbase + j));
        RefStore(*residual, dres, static_cast<size_t>(rbase + j), v);
        v = RefLoad(*residual, dres, static_cast<size_t>(rbase + j));
      }
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(h) + eps);
    for (int64_t j = 0; j < h; ++j) {
      const float v = residual != nullptr
                          ? RefLoad(*residual, dres, static_cast<size_t>(rbase + j))
                          : RefLoad(x, dx, static_cast<size_t>(i * h + j));
      float wj = RefLoad(w, dw, static_cast<size_t>(j));
      if (gemma) wj += 1.0f;
      RefStore(out, dout, static_cast<size_t>(i * h + j), v * inv * wj);
    }
  }
}

// ── Deterministic operands, exactly representable in every dtype under test ──
//
// The values are built from a small integer LCG and scaled by a power of two,
// so f32, f16 and bf16 all hold them EXACTLY. That is deliberate: it removes
// "the two arms rounded the input differently" as an explanation for any
// difference, leaving only the arithmetic the gate is about. The rounding of
// RESULTS is still exercised, because a sum of exact values is not exact.
//
// `scale` exists for ONE reason and it is load-bearing. The residual stream's
// contract is add-in-f32, ROUND on store, RE-READ the rounded value. A test
// whose x and residual share a scale cannot see that: k1*2^-6 + k2*2^-6 with
// |k| < 128 needs at most 8 significant bits, which bf16 holds EXACTLY, so
// dropping the re-read is byte-identical and the mutation survives -- it did,
// on the first version of this file. Giving the residual a scale 2^12 above x
// puts the two operands twelve binades apart, so their sum needs ~19 bits and
// the bf16 store is genuinely lossy. The values stay exactly representable in
// f32, f16 and bf16 INDIVIDUALLY; it is only the SUM that rounds, which is
// exactly the thing under test.
std::vector<float> Values(size_t n, uint64_t seed, float scale = 0.015625f) {
  std::vector<float> v(n);
  uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (size_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const int32_t q = static_cast<int32_t>((s >> 40) & 0xFFu) - 128;  // [-128, 128)
    v[i] = static_cast<float>(q) * scale;
  }
  return v;
}

uint16_t ToF16(float f) {
  uint32_t w;
  std::memcpy(&w, &f, 4);
  const uint32_t sign = (w >> 16) & 0x8000u;
  if ((w & 0x7FFFFFFFu) == 0u) return static_cast<uint16_t>(sign);  // +/-0
  const int32_t exp = static_cast<int32_t>((w >> 23) & 0xFFu) - 127 + 15;
  const uint32_t man = w & 0x7FFFFFu;
  REQUIRE(exp > 0);
  REQUIRE(exp < 31);
  REQUIRE((man & 0x1FFFu) == 0u);  // exactly representable, by construction
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

std::vector<uint8_t> Encode(const std::vector<float>& v, DType d) {
  std::vector<uint8_t> b(v.size() * Width(d));
  for (size_t i = 0; i < v.size(); ++i) {
    if (d == DType::kF32) {
      std::memcpy(b.data() + i * 4, &v[i], 4);
    } else if (d == DType::kBF16) {
      uint32_t w;
      std::memcpy(&w, &v[i], 4);
      const uint16_t bits = static_cast<uint16_t>(w >> 16);
      std::memcpy(b.data() + i * 2, &bits, 2);
    } else {
      const uint16_t bits = ToF16(v[i]);
      std::memcpy(b.data() + i * 2, &bits, 2);
    }
  }
  return b;
}

Tensor Make(std::vector<uint8_t>& b, DType d, std::initializer_list<int64_t> shape) {
  return Tensor::Contiguous(b.data(), d, Cpu(), shape);
}

const char* N(DType d) {
  return d == DType::kF32 ? "f32" : (d == DType::kF16 ? "f16" : "bf16");
}

// Runs one (dx, dw, dout, residual) combination and returns the number of
// BYTES that differ between the kernel's output and the reference's.
struct Diff {
  size_t out_bytes = 0;
  size_t residual_bytes = 0;
};

Diff RunOne(int64_t t, int64_t h, DType dx, DType dw, DType dout, bool gemma,
            bool use_residual, DType dres) {
  const std::vector<float> xv = Values(static_cast<size_t>(t * h), 0x5EED01u);
  const std::vector<float> wv = Values(static_cast<size_t>(h), 0x5EED02u);
  const std::vector<float> rv = Values(static_cast<size_t>(t * h), 0x5EED03u, 64.0f);

  std::vector<uint8_t> xb = Encode(xv, dx), wb = Encode(wv, dw);
  std::vector<uint8_t> got(static_cast<size_t>(t * h) * Width(dout), 0xA5u);
  std::vector<uint8_t> want(got.size(), 0xA5u);
  std::vector<uint8_t> rgot = Encode(rv, dres), rwant = Encode(rv, dres);

  Tensor tx = Make(xb, dx, {t, h});
  Tensor tw = Make(wb, dw, {h});
  Tensor to = Make(got, dout, {t, h});
  Tensor tr = Make(rgot, dres, {t, h});
  Queue q{Cpu(), nullptr};
  const RmsNormArgs args{1e-6f, gemma};
  vt::RmsNorm(q, to, tx, tw, args, use_residual ? &tr : nullptr);

  RefRmsNorm(want, dout, xb, dx, wb, dw, t, h, args.eps, gemma,
             use_residual ? &rwant : nullptr, dres);

  Diff d;
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i] != want[i]) ++d.out_bytes;
  if (use_residual)
    for (size_t i = 0; i < rgot.size(); ++i)
      if (rgot[i] != rwant[i]) ++d.residual_bytes;
  return d;
}

const DType kIn[3] = {DType::kF32, DType::kF16, DType::kBF16};
const DType kOut[2] = {DType::kF32, DType::kBF16};

// Swap the pool the CPU kernels dispatch through, and put it back. The hook is
// there for exactly this ("determinism A/B tests instantiate pools of different
// sizes in one process", cpu_threadpool.h).
struct ScopedPool {
  explicit ScopedPool(vt::cpu::Threadpool* tp) { prev = vt::cpu::Threadpool::SwapForTesting(tp); }
  ~ScopedPool() { vt::cpu::Threadpool::SwapForTesting(prev); }
  vt::cpu::Threadpool* prev = nullptr;
};

}  // namespace

TEST_CASE("rmsnorm elem-dispatch: BYTE-identical over the whole dtype matrix") {
  // Ragged extents on purpose: H=5120 is the Qwen3.6-27B hidden size the survey
  // profiled, H=129 is odd and prime so no vector width divides it, and H=1 is
  // the degenerate row. T=3 keeps every case cheap while still crossing rows.
  const int64_t hs[3] = {5120, 129, 1};
  for (int64_t h : hs) {
    for (DType dx : kIn) {
      for (DType dw : kIn) {
        for (DType dout : kOut) {
          for (bool gemma : {false, true}) {
            const Diff d = RunOne(3, h, dx, dw, dout, gemma, false, DType::kF32);
            INFO("h=" << h << " x=" << N(dx) << " w=" << N(dw) << " out=" << N(dout)
                      << " gemma=" << (gemma ? 1 : 0));
            CHECK(d.out_bytes == 0u);
          }
        }
      }
    }
  }
}

TEST_CASE("rmsnorm elem-dispatch: the residual stream is BYTE-identical too") {
  for (DType dres : kOut) {
    for (DType dx : kIn) {
      for (DType dout : kOut) {
        for (bool gemma : {false, true}) {
          const Diff d = RunOne(3, 5120, dx, DType::kBF16, dout, gemma, true, dres);
          INFO("res=" << N(dres) << " x=" << N(dx) << " out=" << N(dout)
                      << " gemma=" << (gemma ? 1 : 0));
          CHECK(d.out_bytes == 0u);
          CHECK(d.residual_bytes == 0u);
        }
      }
    }
  }
}

// WHY THIS CASE EXISTS AND WHY ONE WORKER COUNT WOULD NOT DO. The hoist's claim
// is about SUMMATION ORDER: each row's variance stays a serial f32 reduction on
// one thread, and rows are independent, so the threadpool's row partition cannot
// move a single addend. A run at one worker count cannot test that claim -- it
// exercises ONE partition, and the defect being excluded is a partition-dependent
// one. `ForRows` chunks by rows, so a different worker count is a different
// partition, and T=64 with worker counts up to 20 gives every count a partition
// no other count produces.
//
// The reference is single-threaded by construction, so this is also a
// parallel-vs-serial identity and not merely a parallel-vs-parallel one.
TEST_CASE("rmsnorm elem-dispatch: BYTE-identical at every worker count") {
  const int workers[] = {1, 2, 3, 4, 5, 8, 20};
  for (int nth : workers) {
    vt::cpu::Threadpool pool(nth);
    ScopedPool guard(&pool);
    REQUIRE(vt::cpu::Threadpool::Global().NThreads() >= 1);
    for (DType dx : kIn) {
      for (DType dout : kOut) {
        // 64 rows so every worker count partitions them differently, and a
        // non-multiple of most counts so the tail chunk is never uniform.
        const Diff plain = RunOne(64, 129, dx, DType::kBF16, dout, false, false, DType::kF32);
        INFO("workers=" << nth << " x=" << N(dx) << " out=" << N(dout) << " no-residual");
        CHECK(plain.out_bytes == 0u);
        const Diff res = RunOne(64, 129, dx, DType::kBF16, dout, true, true, DType::kBF16);
        INFO("workers=" << nth << " x=" << N(dx) << " out=" << N(dout) << " bf16 residual");
        CHECK(res.out_bytes == 0u);
        CHECK(res.residual_bytes == 0u);
      }
    }
  }
}

TEST_CASE("rmsnorm elem-dispatch: a non-float operand is still refused") {
  std::vector<uint8_t> xb(4 * 8, 0), wb(4 * 8, 0), ob(4 * 8, 0);
  Queue q{Cpu(), nullptr};
  Tensor tw = Make(wb, DType::kF32, {8});
  Tensor to = Make(ob, DType::kF32, {1, 8});
  SUBCASE("i32 x") {
    Tensor tx = Make(xb, DType::kI32, {1, 8});
    CHECK_THROWS_AS(vt::RmsNorm(q, to, tx, tw, RmsNormArgs{}), std::runtime_error);
  }
  SUBCASE("block-quantized x") {
    Tensor tx = Make(xb, DType::kQ4_K, {1, 8});
    CHECK_THROWS_AS(vt::RmsNorm(q, to, tx, tw, RmsNormArgs{}), std::runtime_error);
  }
}
