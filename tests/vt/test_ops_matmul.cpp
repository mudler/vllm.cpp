// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }
}  // namespace

TEST_CASE("matmul f32: 2x3 @ 3x2 golden") {
  // a = [[1,2,3],[4,5,6]], b = [[7,8],[9,10],[11,12]]
  // out = [[58,64],[139,154]]
  std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<float> b = {7, 8, 9, 10, 11, 12};
  std::vector<float> out(4, -1.0f);
  Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {2, 3});
  Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {3, 2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2, 2});
  Queue q = CpuQueue();
  vt::Matmul(q, to, ta, tb);
  CHECK(out[0] == 58.0f);
  CHECK(out[1] == 64.0f);
  CHECK(out[2] == 139.0f);
  CHECK(out[3] == 154.0f);
}

TEST_CASE("matmul bf16 weights: converts through f32") {
  // identity-ish: a=[[2]], b=[[3]] in bf16 → out f32 [[6]]
  uint16_t a16 = vt::F32ToBF16(2.0f);
  uint16_t b16 = vt::F32ToBF16(3.0f);
  float out = 0.0f;
  Tensor ta = Tensor::Contiguous(&a16, DType::kBF16, Cpu(), {1, 1});
  Tensor tb = Tensor::Contiguous(&b16, DType::kBF16, Cpu(), {1, 1});
  Tensor to = Tensor::Contiguous(&out, DType::kF32, Cpu(), {1, 1});
  Queue q = CpuQueue();
  vt::Matmul(q, to, ta, tb);
  CHECK(out == 6.0f);
}

TEST_CASE("matmul bf16 output: same golden within bf16 eps") {
  // Same golden as the f32 case: out = [[58,64],[139,154]], stored as bf16.
  std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<float> b = {7, 8, 9, 10, 11, 12};
  std::vector<uint16_t> out(4, 0);
  Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {2, 3});
  Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {3, 2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {2, 2});
  Queue q = CpuQueue();
  vt::Matmul(q, to, ta, tb);
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(58.0f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[1]) == doctest::Approx(64.0f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[2]) == doctest::Approx(139.0f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[3]) == doctest::Approx(154.0f).epsilon(0.01));
}

TEST_CASE("matmul_bt f32: b [N,K] row-major matches Matmul on the transposed weight") {
  // Same golden as the Matmul case: a = [[1,2,3],[4,5,6]] [M=2,K=3],
  // b_nk = b^T = [[7,9,11],[8,10,12]] [N=2,K=3] -> out = [[58,64],[139,154]].
  std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<float> b_nk = {7, 9, 11, 8, 10, 12};
  std::vector<float> out(4, -1.0f);
  Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {2, 3});
  Tensor tb = Tensor::Contiguous(b_nk.data(), DType::kF32, Cpu(), {2, 3});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2, 2});
  Queue q = CpuQueue();
  vt::MatmulBT(q, to, ta, tb);
  CHECK(out[0] == 58.0f);
  CHECK(out[1] == 64.0f);
  CHECK(out[2] == 139.0f);
  CHECK(out[3] == 154.0f);
}

TEST_CASE("matmul_bt validates shapes loudly") {
  std::vector<float> buf(6, 0.0f);
  Tensor a = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {2, 3});
  Tensor b = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {3, 2});  // K mismatch ([N,K] wants K=3)
  Tensor o = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {2, 3});
  Queue q = CpuQueue();
  CHECK_THROWS_AS(vt::MatmulBT(q, o, a, b), std::runtime_error);
}

TEST_CASE("matmul validates shapes loudly") {
  std::vector<float> buf(6, 0.0f);
  Tensor a = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {2, 3});
  Tensor b = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {2, 3});  // K mismatch
  Tensor o = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {2, 3});
  Queue q = CpuQueue();
  CHECK_THROWS_AS(vt::Matmul(q, o, a, b), std::runtime_error);
}

TEST_CASE("missing kernel for unregistered device throws") {
  float x = 0;
  Tensor t = Tensor::Contiguous(&x, DType::kF32, Device{DeviceType::kVULKAN, 0}, {1, 1});
  Queue q{Device{DeviceType::kVULKAN, 0}, nullptr};
  CHECK_THROWS_AS(vt::Matmul(q, t, t, t), std::runtime_error);
}
