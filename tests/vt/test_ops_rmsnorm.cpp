// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormArgs;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
}  // namespace

TEST_CASE("rmsnorm golden row") {
  // x = [3,4]; mean(x^2) = 12.5; rms = sqrt(12.5); w = [2, 0.5]
  // out = [3/3.53553*2, 4/3.53553*0.5] = [1.697056, 0.565685]
  std::vector<float> x = {3.0f, 4.0f};
  std::vector<float> w = {2.0f, 0.5f};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false});
  CHECK(out[0] == doctest::Approx(1.697056f));
  CHECK(out[1] == doctest::Approx(0.565685f));
}

TEST_CASE("rmsnorm gemma variant uses (1+w)") {
  std::vector<float> x = {3.0f, 4.0f};
  std::vector<float> w = {1.0f, -0.5f};  // effective weights [2, 0.5]
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, true});
  CHECK(out[0] == doctest::Approx(1.697056f));
  CHECK(out[1] == doctest::Approx(0.565685f));
}

TEST_CASE("rmsnorm eps matters for zero rows") {
  std::vector<float> x = {0.0f, 0.0f};
  std::vector<float> w = {1.0f, 1.0f};
  std::vector<float> out(2, -1.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{1e-6f, false});
  CHECK(out[0] == 0.0f);  // no NaN
  CHECK(out[1] == 0.0f);
}

TEST_CASE("fused residual add updates residual stream then normalizes") {
  // x = [1,2], residual = [2,2] → sum = [3,4] (residual becomes [3,4])
  // norm of [3,4] with w=[1,1]: [0.848528, 1.131371]
  std::vector<float> x = {1.0f, 2.0f};
  std::vector<float> res = {2.0f, 2.0f};
  std::vector<float> w = {1.0f, 1.0f};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tr = Tensor::Contiguous(res.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false}, &tr);
  CHECK(res[0] == 3.0f);
  CHECK(res[1] == 4.0f);
  CHECK(out[0] == doctest::Approx(0.848528f));
  CHECK(out[1] == doctest::Approx(1.131371f));
}

TEST_CASE("rmsnorm multi-row normalizes independently") {
  std::vector<float> x = {3.0f, 4.0f, 6.0f, 8.0f};
  std::vector<float> w = {1.0f, 1.0f};
  std::vector<float> out(4, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {2, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false});
  CHECK(out[0] == doctest::Approx(0.848528f));  // row 0
  CHECK(out[2] == doctest::Approx(0.848528f));  // row 1 same direction
}

TEST_CASE("rmsnorm accepts bf16 inputs via f32 conversion") {
  // bf16(3.0)=0x4040, bf16(4.0)=0x4080 are exact; w bf16(1.0)=0x3F80
  std::vector<uint16_t> x = {0x4040, 0x4080};
  std::vector<uint16_t> w = {0x3F80, 0x3F80};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kBF16, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kBF16, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false});
  CHECK(out[0] == doctest::Approx(0.848528f));
  CHECK(out[1] == doctest::Approx(1.131371f));
}
