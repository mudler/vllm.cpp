// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RopeArgs;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
}  // namespace

TEST_CASE("rope neox rotates pairs at pos=1, full rotary") {
  // 1 token, 1 head, head_dim=4, rotary_dim=4, base=10000
  // inv_freq = [1, 10000^(-0.5)=0.01]
  // pairs: (0,2) angle=1.0; (1,3) angle=0.01
  // q = [1, 0, 0, 1]:
  //   q0' = 1*cos(1) - 0*sin(1) = 0.540302
  //   q2' = 1*sin(1) + 0*cos(1) = 0.841471
  //   q1' = 0*cos(.01) - 1*sin(.01) = -0.00999983
  //   q3' = 0*sin(.01) + 1*cos(.01) = 0.99995
  std::vector<float> qs = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> ks = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<int32_t> pos = {1};
  Tensor tq = Tensor::Contiguous(qs.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(ks.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 4});
  CHECK(qs[0] == doctest::Approx(0.540302f));
  CHECK(qs[1] == doctest::Approx(-0.00999983f));
  CHECK(qs[2] == doctest::Approx(0.841471f));
  CHECK(qs[3] == doctest::Approx(0.99995f));
  CHECK(ks[0] == doctest::Approx(0.540302f));  // k rotated identically
}

TEST_CASE("rope bf16 in-place: same pos=1 golden within bf16 eps") {
  // Same golden as the f32 case; q/k stored as bf16 and rotated in place.
  // bf16(1.0)=0x3F80, bf16(0.0)=0x0000 are exact.
  std::vector<uint16_t> qs = {0x3F80, 0x0000, 0x0000, 0x3F80};
  std::vector<uint16_t> ks = {0x3F80, 0x0000, 0x0000, 0x3F80};
  std::vector<int32_t> pos = {1};
  Tensor tq = Tensor::Contiguous(qs.data(), DType::kBF16, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(ks.data(), DType::kBF16, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 4});
  CHECK(vt::BF16ToF32(qs[0]) == doctest::Approx(0.540302f).epsilon(0.01));
  CHECK(vt::BF16ToF32(qs[1]) == doctest::Approx(-0.00999983f).epsilon(0.01));
  CHECK(vt::BF16ToF32(qs[2]) == doctest::Approx(0.841471f).epsilon(0.01));
  CHECK(vt::BF16ToF32(qs[3]) == doctest::Approx(0.99995f).epsilon(0.01));
  CHECK(vt::BF16ToF32(ks[0]) == doctest::Approx(0.540302f).epsilon(0.01));
}

TEST_CASE("rope rejects mismatched q/k dtypes") {
  std::vector<float> qbuf(4, 0.0f);
  std::vector<uint16_t> kbuf(4, 0);
  std::vector<int32_t> pos = {0};
  Tensor tq = Tensor::Contiguous(qbuf.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(kbuf.data(), DType::kBF16, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 4}), std::runtime_error);
}

TEST_CASE("rope pos=0 is identity") {
  std::vector<float> qs = {0.3f, -0.7f, 1.1f, 2.2f};
  std::vector<float> ks = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<int64_t> pos = {0};
  Tensor tq = Tensor::Contiguous(qs.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(ks.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI64, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 4});
  CHECK(qs[0] == doctest::Approx(0.3f));
  CHECK(qs[1] == doctest::Approx(-0.7f));
  CHECK(qs[2] == doctest::Approx(1.1f));
  CHECK(qs[3] == doctest::Approx(2.2f));
}

TEST_CASE("partial rope leaves tail dims untouched") {
  // head_dim=4, rotary_dim=2: only pair (0,1) rotates; dims 2,3 untouched
  std::vector<float> qs = {1.0f, 0.0f, 5.0f, 6.0f};
  std::vector<float> ks = {0.0f, 0.0f, 7.0f, 8.0f};
  std::vector<int32_t> pos = {1};
  Tensor tq = Tensor::Contiguous(qs.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(ks.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 2});
  CHECK(qs[0] == doctest::Approx(0.540302f));   // cos(1)
  CHECK(qs[1] == doctest::Approx(0.841471f));   // sin(1)
  CHECK(qs[2] == 5.0f);
  CHECK(qs[3] == 6.0f);
  CHECK(ks[2] == 7.0f);
}

TEST_CASE("rope validates rotary_dim") {
  std::vector<float> buf(4, 0.0f);
  std::vector<int32_t> pos = {0};
  Tensor tq = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 3}), std::runtime_error);  // odd
  CHECK_THROWS_AS(vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 8}), std::runtime_error);  // > D
}
