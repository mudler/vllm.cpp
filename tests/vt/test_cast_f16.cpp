// vt::CastF16 — the f32/bf16 -> f16 narrowing cast, third sibling of CastBf16
// (f32 -> bf16) and CastF32 (bf16 -> f32).
//
// WHY IT EXISTS (QUANT-EXL3 W1a, #2181). `vt::Exl3Gemm` reads its activation as
// f16 and nothing else: the CPU arm calls `HadRows(HadIo::kHalfHalf, ...)` on
// `a` (`src/vt/cpu/cpu_exl3_kernels.cpp:205`) and the device arm's `a_had`
// staging is fp16 throughout, because exllamav3 runs the whole linear in fp16.
// Our dense forward keeps its residual in the model dtype, so an EXL3 linear
// needs one narrowing cast on the way in. Two of the three casts existed; this
// is the missing one, and it is a general op rather than an EXL3-private helper
// because the direction is not EXL3-specific.
//
// The gate is EXACTNESS, not a tolerance: f32 -> f16 is round-to-nearest-even
// through `vt::F32ToF16`, which is the same function every other f16 store in
// the tree uses, so the expectation is the value that function returns and any
// disagreement is a defect rather than drift.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

vt::Queue CpuQueue() { return vt::GetBackend(vt::DeviceType::kCPU).CreateQueue(); }

}  // namespace

TEST_CASE("cast_f16: f32 -> f16 is exactly F32ToF16, elementwise") {
  vt::Queue q = CpuQueue();
  const vt::Device dev = q.device;

  // Values chosen to exercise the parts of the encoding a mean-error check
  // cannot see: a subnormal, a value that rounds to even, the max finite f16,
  // and one ABOVE it, which saturates to inf rather than wrapping.
  const std::vector<float> in = {0.0f,      -0.0f,     1.0f,        -2.5f,
                                 6.1e-5f,   65504.0f,  70000.0f,    1.0009765625f,
                                 -1.0e-8f,  3.14159265f};
  std::vector<uint16_t> out(in.size(), 0xDEAD);

  vt::Tensor ti = vt::Tensor::Contiguous(const_cast<float*>(in.data()), vt::DType::kF32, dev,
                                         {static_cast<int64_t>(in.size())});
  vt::Tensor to = vt::Tensor::Contiguous(out.data(), vt::DType::kF16, dev,
                                         {static_cast<int64_t>(out.size())});
  vt::CastF16(q, to, ti);

  for (size_t i = 0; i < in.size(); ++i) {
    CHECK(out[i] == vt::F32ToF16(in[i]));
  }
  // The saturating case, spelled out rather than left to the loop: 70000 is
  // above f16's 65504 max finite value.
  CHECK(std::isinf(vt::F16ToF32(out[6])));
  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}

TEST_CASE("cast_f16: bf16 -> f16 goes through the f32 value both encodings name") {
  vt::Queue q = CpuQueue();
  const vt::Device dev = q.device;

  const std::vector<float> src = {1.5f, -0.75f, 1024.0f, 6.0e-8f};
  std::vector<uint16_t> in(src.size());
  for (size_t i = 0; i < src.size(); ++i) in[i] = vt::F32ToBF16(src[i]);
  std::vector<uint16_t> out(src.size(), 0xDEAD);

  vt::Tensor ti = vt::Tensor::Contiguous(in.data(), vt::DType::kBF16, dev,
                                         {static_cast<int64_t>(in.size())});
  vt::Tensor to = vt::Tensor::Contiguous(out.data(), vt::DType::kF16, dev,
                                         {static_cast<int64_t>(out.size())});
  vt::CastF16(q, to, ti);

  for (size_t i = 0; i < src.size(); ++i) {
    // NOT F32ToF16(src[i]): the bf16 store already rounded, and the cast reads
    // back THAT value. Asserting against the original would pass only because
    // these particular values survive bf16, which is the tautology to avoid.
    CHECK(out[i] == vt::F32ToF16(vt::BF16ToF32(in[i])));
  }
  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}

TEST_CASE("cast_f16: it REFUSES what it cannot express") {
  vt::Queue q = CpuQueue();
  const vt::Device dev = q.device;
  std::vector<float> f32(4, 1.0f);
  std::vector<uint16_t> f16(4, 0);

  vt::Tensor tf32 = vt::Tensor::Contiguous(f32.data(), vt::DType::kF32, dev, {4});
  vt::Tensor tf16 = vt::Tensor::Contiguous(f16.data(), vt::DType::kF16, dev, {4});
  vt::Tensor tf16_short = vt::Tensor::Contiguous(f16.data(), vt::DType::kF16, dev, {2});

  // A wrong destination dtype is the defect this refusal exists for: writing
  // f16 bytes into an f32 buffer is silent and produces garbage of the right
  // shape, which no shape check catches.
  CHECK_THROWS(vt::CastF16(q, tf32, tf32));
  // Element-count mismatch.
  CHECK_THROWS(vt::CastF16(q, tf16_short, tf32));
  // An f16 SOURCE is refused rather than silently copied: the two existing
  // casts each name one source dtype, and a third that accepted anything would
  // make a wrong-dtype activation invisible.
  CHECK_THROWS(vt::CastF16(q, tf16, tf16));
  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}
