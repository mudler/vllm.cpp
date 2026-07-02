// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cstdint>

#include "vt/arena.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::StepArena;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
}  // namespace

TEST_CASE("arena bump-allocates aligned tensors and resets O(1)") {
  StepArena arena(Cpu(), 4096);
  CHECK(arena.Capacity() == 4096);
  CHECK(arena.Used() == 0);

  Tensor a = arena.Alloc(DType::kF32, {10});  // 40 bytes → padded to 64
  CHECK(a.Numel() == 10);
  CHECK(reinterpret_cast<uintptr_t>(a.data) % 64 == 0);
  CHECK(arena.Used() == 64);

  Tensor b = arena.Alloc(DType::kI8, {1});  // 1 byte → 64
  CHECK(reinterpret_cast<uintptr_t>(b.data) % 64 == 0);
  CHECK(arena.Used() == 128);
  CHECK(b.Ptr<int8_t>() != static_cast<int8_t*>(a.data));

  size_t high = arena.Used();
  arena.Reset();
  CHECK(arena.Used() == 0);
  CHECK(arena.HighWater() == high);

  // memory is reused after reset
  Tensor c = arena.Alloc(DType::kF32, {10});
  CHECK(c.data == a.data);
}

TEST_CASE("arena overflows loudly") {
  StepArena arena(Cpu(), 128);
  (void)arena.Alloc(DType::kF32, {16});  // 64 bytes
  (void)arena.Alloc(DType::kF32, {16});  // 128 total
  CHECK_THROWS_AS(arena.Alloc(DType::kF32, {1}), std::runtime_error);
}

TEST_CASE("arena rejects overflowing allocation sizes loudly") {
  StepArena arena(Cpu(), 256);
  CHECK_THROWS_AS(arena.Alloc(DType::kI64, {int64_t{1} << 61}), std::runtime_error);
  (void)arena.Alloc(DType::kF32, {16});  // 64 bytes used
  CHECK_THROWS_AS(arena.Alloc(DType::kI64, {(int64_t{1} << 61) - 8}), std::runtime_error);
  CHECK(arena.Used() == 64);  // accounting not corrupted by rejected allocs
}

TEST_CASE("arena tensors carry device and dtype") {
  StepArena arena(Cpu(), 256);
  Tensor t = arena.Alloc(DType::kBF16, {2, 3});
  CHECK(t.dtype == DType::kBF16);
  CHECK(t.device == Cpu());
  CHECK(t.IsContiguous());
}
