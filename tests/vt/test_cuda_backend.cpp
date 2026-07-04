// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"

namespace {

using vt::Backend;
using vt::DeviceType;
using vt::Queue;

// The CUDA registrar only registers kCUDA when a device is actually present,
// so a failed lookup means "skip" — both on CPU-only builds and on CUDA
// builds running on a box without a GPU.
bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

}  // namespace

TEST_CASE("CUDA backend: alloc/memset/copy round-trip") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& cuda = vt::GetBackend(DeviceType::kCUDA);
  Queue q = cuda.CreateQueue();
  constexpr size_t kBytes = 4096;
  void* d = cuda.Alloc(kBytes);
  REQUIRE(d != nullptr);
  // 64B alignment contract (cudaMalloc actually guarantees 256B).
  CHECK(reinterpret_cast<uintptr_t>(d) % 64 == 0);

  cuda.Memset(q, d, 0xAB, kBytes);
  std::vector<unsigned char> host(kBytes, 0);
  cuda.Copy(q, host.data(), d, kBytes);  // d2h
  cuda.Synchronize(q);
  std::vector<unsigned char> expected(kBytes, 0xAB);
  CHECK(std::memcmp(host.data(), expected.data(), kBytes) == 0);

  std::vector<unsigned char> src(kBytes);
  for (size_t i = 0; i < kBytes; ++i) src[i] = static_cast<unsigned char>(i % 251);
  cuda.Copy(q, d, src.data(), kBytes);  // h2d
  std::vector<unsigned char> back(kBytes, 0);
  cuda.Copy(q, back.data(), d, kBytes);  // d2h
  cuda.Synchronize(q);
  CHECK(std::memcmp(back.data(), src.data(), kBytes) == 0);

  cuda.Free(d);
  cuda.DestroyQueue(q);
}

TEST_CASE("CUDA backend: queue create/destroy") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& cuda = vt::GetBackend(DeviceType::kCUDA);
  Queue q = cuda.CreateQueue();
  CHECK(q.device.type == DeviceType::kCUDA);
  CHECK(q.device.index == 0);
  CHECK(q.handle != nullptr);  // a real cudaStream_t, unlike CPU's nullptr
  cuda.DestroyQueue(q);
  CHECK(q.handle == nullptr);
  cuda.DestroyQueue(q);  // second destroy is a safe no-op
}

TEST_CASE("CUDA backend: Synchronize drains pending work") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& cuda = vt::GetBackend(DeviceType::kCUDA);
  Queue q = cuda.CreateQueue();
  constexpr size_t kBytes = 1 << 20;
  void* d = cuda.Alloc(kBytes);
  cuda.Memset(q, d, 0x5A, kBytes);
  CHECK_NOTHROW(cuda.Synchronize(q));
  std::vector<unsigned char> host(kBytes, 0);
  cuda.Copy(q, host.data(), d, kBytes);
  cuda.Synchronize(q);
  CHECK(host.front() == 0x5A);
  CHECK(host.back() == 0x5A);
  cuda.Free(d);
  cuda.DestroyQueue(q);
}

TEST_CASE("CUDA backend: graph capture/replay re-executes captured ops") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& cuda = vt::GetBackend(DeviceType::kCUDA);
  CHECK(cuda.SupportsGraphCapture());
  Queue q = cuda.CreateQueue();
  constexpr size_t kBytes = 64 * 1024;

  // Persistent device buffers (allocated ONCE, fixed pointers across replays —
  // the M2.5 decode-capture contract). src feeds a captured d2d copy into dst.
  void* src = cuda.Alloc(kBytes);
  void* dst = cuda.Alloc(kBytes);

  std::vector<unsigned char> p1(kBytes, 0x11);
  std::vector<unsigned char> p2(kBytes, 0x22);
  cuda.Copy(q, src, p1.data(), kBytes);  // load pattern 1 (pre-capture)
  cuda.Memset(q, dst, 0, kBytes);
  cuda.Synchronize(q);

  // Capture a single d2d copy (recorded, NOT executed during capture).
  cuda.BeginCapture(q);
  cuda.Copy(q, dst, src, kBytes);
  cuda.EndCapture(q);

  // Replay #1 -> dst should become pattern 1 (proves the graph ran at all).
  cuda.Replay(q);
  cuda.Synchronize(q);
  std::vector<unsigned char> back(kBytes, 0);
  cuda.Copy(q, back.data(), dst, kBytes);
  cuda.Synchronize(q);
  CHECK(back.front() == 0x11);
  CHECK(back.back() == 0x11);

  // Mutate the (same-pointer) src, replay again -> dst must reflect the NEW src,
  // proving replay RE-EXECUTES the captured copy over the persistent buffers
  // (exactly how the decode graph picks up each new token's inputs).
  cuda.Copy(q, src, p2.data(), kBytes);
  cuda.Replay(q);
  cuda.Synchronize(q);
  cuda.Copy(q, back.data(), dst, kBytes);
  cuda.Synchronize(q);
  CHECK(back.front() == 0x22);
  CHECK(back.back() == 0x22);

  cuda.Free(src);
  cuda.Free(dst);
  cuda.DestroyQueue(q);
}

TEST_CASE("CUDA backend: unified-memory flag report") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& cuda = vt::GetBackend(DeviceType::kCUDA);
  // Informational: GB10 is expected to report true (pageable memory access);
  // discrete GPUs report false. Either value is valid.
  MESSAGE("CUDA UnifiedMemory() = " << (cuda.UnifiedMemory() ? "true" : "false"));
  CHECK(true);
}
