// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <vector>

#include "vt/tensor.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
}  // namespace

TEST_CASE("contiguous tensor: shape, strides, numel, bytes") {
  float buf[24];
  Tensor t = Tensor::Contiguous(buf, DType::kF32, Cpu(), {2, 3, 4});
  CHECK(t.rank == 3);
  CHECK(t.Numel() == 24);
  CHECK(t.Bytes() == 96);
  CHECK(t.IsContiguous());
  CHECK(t.stride[0] == 12);
  CHECK(t.stride[1] == 4);
  CHECK(t.stride[2] == 1);
  CHECK(t.Offset({1, 2, 3}) == 23);
  CHECK(t.Ptr<float>() == buf);
}

TEST_CASE("view reshapes contiguous tensors") {
  float buf[24];
  Tensor t = Tensor::Contiguous(buf, DType::kF32, Cpu(), {2, 12});
  Tensor v = t.View({4, 6});
  CHECK(v.rank == 2);
  CHECK(v.shape[0] == 4);
  CHECK(v.stride[0] == 6);
  CHECK(v.data == t.data);
  CHECK_THROWS_AS(t.View({5, 5}), std::runtime_error);  // numel mismatch
}

TEST_CASE("slice advances pointer and keeps strides") {
  std::vector<float> buf(24);
  for (int i = 0; i < 24; ++i) buf[static_cast<size_t>(i)] = static_cast<float>(i);
  Tensor t = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {6, 4});
  Tensor s = t.Slice(0, 2, 5);  // rows 2..4
  CHECK(s.shape[0] == 3);
  CHECK(s.shape[1] == 4);
  CHECK(s.Ptr<float>()[0] == 8.0f);  // row 2, col 0
  CHECK(s.IsContiguous());

  Tensor c = t.Slice(1, 1, 3);  // cols 1..2 — non-contiguous
  CHECK(c.shape[0] == 6);
  CHECK(c.shape[1] == 2);
  CHECK_FALSE(c.IsContiguous());
  CHECK(c.Ptr<float>()[0] == 1.0f);
  CHECK(c.Offset({1, 0}) == 4);  // strides preserved
}

TEST_CASE("huge shapes overflow loudly instead of wrapping") {
  float buf[1];
  // (1 << 21)^3 * 2 == 2^65 overflows int64 during stride accumulation.
  CHECK_THROWS_AS(
      Tensor::Contiguous(buf, DType::kF32, Cpu(), {1 << 21, 1 << 21, 1 << 21, 2}),
      std::runtime_error);
  // 2^62 * 4 == 2^64 overflows even at rank 2.
  CHECK_THROWS_AS(
      Tensor::Contiguous(buf, DType::kF32, Cpu(), {int64_t{1} << 62, 4}),
      std::runtime_error);
  // Near-limit but non-overflowing shape stays fine ((1 << 20)^3 * 2 == 2^61).
  Tensor big = Tensor::Contiguous(buf, DType::kF32, Cpu(), {1 << 20, 1 << 20, 1 << 20, 2});
  CHECK(big.Numel() == int64_t{1} << 61);
}

TEST_CASE("Numel guards hand-built tensors") {
  Tensor t;
  t.rank = 2;

  t.shape[0] = int64_t{1} << 62;
  t.shape[1] = 4;
  CHECK_THROWS_AS(t.Numel(), std::runtime_error);  // 2^64 overflows

  t.shape[0] = 3;
  t.shape[1] = -2;
  CHECK_THROWS_AS(t.Numel(), std::runtime_error);  // negative dim is loud, not UB

  t.shape[0] = 0;
  t.shape[1] = int64_t{1} << 62;
  CHECK(t.Numel() == 0);  // zero dim short-circuits safely

  t.shape[0] = 6;
  t.shape[1] = 7;
  CHECK(t.Numel() == 42);
}

TEST_CASE("bounds are checked loudly") {
  float buf[4];
  Tensor t = Tensor::Contiguous(buf, DType::kF32, Cpu(), {2, 2});
  CHECK_THROWS_AS(t.Slice(2, 0, 1), std::runtime_error);   // bad dim
  CHECK_THROWS_AS(t.Slice(0, 1, 3), std::runtime_error);   // stop > shape
  CHECK_THROWS_AS(Tensor::Contiguous(buf, DType::kF32, Cpu(), {1, 2, 3, 4, 5}),
                  std::runtime_error);                      // rank > kMaxRank
}
