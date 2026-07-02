// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <vector>

#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
}  // namespace

TEST_CASE("silu_and_mul golden") {
  // x = [1, 2, 3, 4] with D=2: gate=[1,2], up=[3,4]
  // silu(1)=0.731059, silu(2)=1.761594
  // out = [2.193176, 7.046377]
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 4});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::SiluAndMul(q, to, tx);
  CHECK(out[0] == doctest::Approx(2.193176f));
  CHECK(out[1] == doctest::Approx(7.046377f));
}

TEST_CASE("silu_and_mul rejects odd inner dim") {
  std::vector<float> x(3, 0.0f);
  std::vector<float> out(1, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 3});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 1});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::SiluAndMul(q, to, tx), std::runtime_error);
}

TEST_CASE("embedding gathers rows by id") {
  // table 3x2: [[0,1],[10,11],[20,21]]; ids [2,0] → [[20,21],[0,1]]
  std::vector<float> table = {0, 1, 10, 11, 20, 21};
  std::vector<int32_t> ids = {2, 0};
  std::vector<float> out(4, -1.0f);
  Tensor tt = Tensor::Contiguous(table.data(), DType::kF32, Cpu(), {3, 2});
  Tensor ti = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2, 2});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, to, tt, ti);
  CHECK(out[0] == 20.0f);
  CHECK(out[1] == 21.0f);
  CHECK(out[2] == 0.0f);
  CHECK(out[3] == 1.0f);
}

TEST_CASE("embedding bounds-checks ids") {
  std::vector<float> table = {0, 1};
  std::vector<int64_t> ids = {5};
  std::vector<float> out(2, 0.0f);
  Tensor tt = Tensor::Contiguous(table.data(), DType::kF32, Cpu(), {1, 2});
  Tensor ti = Tensor::Contiguous(ids.data(), DType::kI64, Cpu(), {1});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::Embedding(q, to, tt, ti), std::runtime_error);
}
