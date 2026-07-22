// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cstring>

#include "vt/backend.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Event;
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

  cpu.Synchronize(q);  // no-op on CPU: all prior work already complete
  CHECK(dst[63] == 0xAB);

  cpu.Free(p);
}

// A DeviceType slot with no backend behind it must throw rather than hand back a
// null/garbage backend. kMETAL was the stand-in for "reserved but unimplemented"
// — but a VLLM_CPP_METAL build on a Metal-capable host now genuinely registers
// it, so the case picks a slot that is still empty there. kXPU is the right
// stand-in: it is HW-BLOCKED with no local target and no implementation
// (.agents/specs/backend-fanout-metal-vulkan-xpu.md § Scope), so the property
// under test — reserved-but-unregistered throws — keeps a live subject on both
// platforms rather than being compiled away on macOS.
TEST_CASE("unregistered backend throws") {
#ifdef VLLM_CPP_METAL
  CHECK_THROWS_AS(vt::GetBackend(DeviceType::kXPU), std::runtime_error);
#else
  CHECK_THROWS_AS(vt::GetBackend(DeviceType::kMETAL), std::runtime_error);
  CHECK_THROWS_AS(vt::GetBackend(DeviceType::kXPU), std::runtime_error);
#endif
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

// ─── ENG-ASYNC-SCHED W3: async-output primitives (async_utils.py:12-70) ───────
// The pinned-host + cross-stream-event seam. On CPU it degenerates to
// synchronous host ops (unified memory), which is the exact contract the CUDA
// override must honour: the value read after SynchronizeEvent equals the value
// written before the copy, and the main queue is never synchronized.
TEST_CASE("pinned host allocation is usable host memory") {
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);

  void* pinned = cpu.AllocPinned(64);
  REQUIRE(pinned != nullptr);
  auto* bytes = reinterpret_cast<unsigned char*>(pinned);
  bytes[0] = 0x11;
  bytes[63] = 0x22;
  CHECK(bytes[0] == 0x11);
  CHECK(bytes[63] == 0x22);
  cpu.FreePinned(pinned);

  // A zero-byte request still returns a valid, freeable block.
  void* z = cpu.AllocPinned(0);
  CHECK(z != nullptr);
  cpu.FreePinned(z);
  // A null free is a no-op (matches the release path when no buffer was taken).
  cpu.FreePinned(nullptr);
}

TEST_CASE("event record/wait/synchronize model the async D2H handoff") {
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue main_q = cpu.CreateQueue();
  Queue copy_q = cpu.CreateQueue();

  // Mirror AsyncOutput.__init__: produce a device-resident value on the main
  // queue, then hand it to a copy queue that first waits the main queue, does a
  // non-blocking D2H into pinned host memory, and records a completion event.
  int32_t device_sampled[2] = {701, 902};  // "device" tensor (unified == host)
  void* pinned = cpu.AllocPinned(sizeof(device_sampled));

  Event fork_ev = cpu.CreateEvent();
  cpu.RecordEvent(fork_ev, main_q);      // event on the main queue's work
  cpu.QueueWaitEvent(copy_q, fork_ev);   // copy queue waits the main queue
  cpu.DestroyEvent(fork_ev);

  cpu.Copy(copy_q, pinned, device_sampled, sizeof(device_sampled));  // non-blocking D2H

  Event ready = cpu.CreateEvent();
  cpu.RecordEvent(ready, copy_q);

  // get_output(): the HOST waits ONLY the copy event, never the main queue.
  cpu.SynchronizeEvent(ready);
  auto* host = reinterpret_cast<int32_t*>(pinned);
  CHECK(host[0] == 701);
  CHECK(host[1] == 902);

  cpu.DestroyEvent(ready);
  cpu.FreePinned(pinned);
}

TEST_CASE("CPU events carry a null handle (synchronous degeneration)") {
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Event e = cpu.CreateEvent();
  CHECK(e.handle == nullptr);
  // Destroy is idempotent on a null-handle event.
  cpu.DestroyEvent(e);
  cpu.DestroyEvent(e);
}
