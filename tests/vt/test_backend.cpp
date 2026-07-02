// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cstring>

#include "vt/backend.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;

TEST_CASE("CPU backend is registered and allocates usable memory") {
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  CHECK(cpu.UnifiedMemory());
  CHECK_FALSE(cpu.SupportsGraphCapture());

  Queue q = cpu.CreateQueue();
  CHECK(q.device.type == DeviceType::kCPU);

  void* p = cpu.Alloc(64);
  REQUIRE(p != nullptr);
  cpu.Memset(q, p, 0xAB, 64);
  CHECK(reinterpret_cast<unsigned char*>(p)[63] == 0xAB);

  unsigned char dst[64];
  cpu.Copy(q, dst, p, 64);
  CHECK(dst[0] == 0xAB);
  cpu.Free(p);
}

TEST_CASE("unregistered backend throws") {
  CHECK_THROWS_AS(vt::GetBackend(DeviceType::kMETAL), std::runtime_error);
}

TEST_CASE("graph capture unsupported on CPU throws loud") {
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue q = cpu.CreateQueue();
  CHECK_THROWS_AS(cpu.BeginCapture(q), std::runtime_error);
}

TEST_CASE("Device equality") {
  CHECK(Device{DeviceType::kCPU, 0} == Device{DeviceType::kCPU, 0});
  CHECK_FALSE(Device{DeviceType::kCPU, 0} == Device{DeviceType::kCUDA, 0});
}
