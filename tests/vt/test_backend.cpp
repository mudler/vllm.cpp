// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cstring>
#include <string>

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
  // S7: a single-stream backend declines the MoE shared-expert aux-stream
  // overlap — base false, exactly what the old `device==kCUDA` gate returned on
  // CPU (the shared path runs serially, byte-identical, no overlap).
  CHECK_FALSE(cpu.SupportsAuxStream());

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

// `DeviceTypeFromName` is the inverse of `DeviceTypeName` (#672). It exists so
// the device-agnostic `vllm` layer can honour a device NAME an operator typed
// without spelling a device enumerator or casting an integer into one — both of
// which `scripts/check-device-leakage.py` counts as leakage, the second because
// a cast hardcodes a device by ENUM VALUE and silently re-points if the enum is
// ever reordered.
//
// The gate is a ROUND TRIP over every DeviceType rather than a spot check of two
// spellings, because the failure this function can have is a MISSING or
// TRANSPOSED entry, and a spot check of the entries that are present cannot see
// one. The header's static_assert catches a list of the wrong LENGTH; only the
// round trip catches a list of the right length with a name repeated.
TEST_CASE("DeviceTypeFromName round-trips every DeviceType") {
  size_t resolved = 0;
  for (size_t i = 0; i < vt::kNumDeviceTypes; ++i) {
    const DeviceType type = static_cast<DeviceType>(i);
    const char* name = vt::DeviceTypeName(type);
    CAPTURE(std::string(name));
    REQUIRE(std::string(name) != "unknown");
    // Seeded with something OTHER than kCPU: kCPU is what a failed resolve
    // leaves behind in the caller, so seeding it here would let a function that
    // never writes `out` pass the first iteration.
    DeviceType back = DeviceType::kTENSTORRENT;
    REQUIRE(vt::DeviceTypeFromName(name, &back));
    CHECK(back == type);
    ++resolved;
  }
  // Say HOW MANY were examined. A loop that silently ran zero times reports the
  // same green as one that checked every platform.
  CHECK(resolved == vt::kNumDeviceTypes);

  DeviceType out = DeviceType::kCPU;
  CHECK_FALSE(vt::DeviceTypeFromName("gpu", &out));
  CHECK_FALSE(vt::DeviceTypeFromName("", &out));
  CHECK_FALSE(vt::DeviceTypeFromName(nullptr, &out));
  // Neither a prefix nor an extension may match: the comparison walks to BOTH
  // NULs, so a truncating `strncmp` spelling of it would accept these.
  CHECK_FALSE(vt::DeviceTypeFromName("cud", &out));
  CHECK_FALSE(vt::DeviceTypeFromName("cudax", &out));
  CHECK_FALSE(vt::DeviceTypeFromName("CUDA", &out));  // the names are lowercase
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

// ISSUE #1635. `Backend::DeviceMemoryIsHostAddressable()` decides whether a HOST
// kernel may dereference a DEVICE allocation. Three landed levers read it: the
// portable CPU reference tier (`src/vt/op_provider.cpp`), the weight loader's
// `VT_ADOPT_DEVICE_BYTES` adoption (both `AdoptDeviceBytesAsHost` branches in
// `src/vllm/model_executor/models/qwen3_5_weights.cpp`) and the logits-processor
// bounce (`src/vllm/v1/sample/logits_processor/builtin.cpp`). Being wrong here
// hands a device pointer to a host memcpy, which is what #844 and #1435 measured
// as a SIGSEGV, so the polarity `include/vt/backend.h` chose is that a backend
// must OPT IN and everything else inherits `false`.
//
// Nothing read that inherited value. Every fake in this tree overrides the method
// -- `tests/vt/test_reference_tier.cpp`, `tests/vllm/test_load_direct_upload.cpp`,
// `tests/vllm/test_qwen36_weights.cpp` and the rest all take the answer as a
// constructor argument -- so each of them measures its own override and none of
// them measures the default. This subclass deliberately declares no override,
// which is the only way to read the default itself.
//
// It is also the half of the CUDA answer that a host lane can hold. `CudaBackend`
// (`src/vt/cuda/cuda_backend.cu`) declares no override either, and a
// `static_assert` beside that class fails the build if one ever appears -- so
// CUDA's answer IS this value, and flipping this default would flip CUDA's answer
// with no CUDA source changed and no CUDA lane able to notice.
namespace {
class DefaultsOnlyBackend final : public Backend {
 public:
  void* Alloc(size_t) override { return nullptr; }
  void Free(void*) override {}
  void Memset(Queue&, void*, int, size_t) override {}
  void Copy(Queue&, void*, const void*, size_t) override {}
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  // The WIDE predicate answers true on purpose, so the case below cannot pass by
  // the two questions happening to agree: this is the GB10 CUDA shape, where host
  // and device address one physical RAM and a `cudaMalloc` pointer is still not
  // host-dereferenceable.
  bool UnifiedMemory() const override { return true; }
  // NO DeviceMemoryIsHostAddressable override. That absence is the subject.
};
}  // namespace

TEST_CASE("Backend::DeviceMemoryIsHostAddressable defaults to false") {
  DefaultsOnlyBackend b;
  REQUIRE(b.UnifiedMemory());
  CHECK_FALSE(b.DeviceMemoryIsHostAddressable());
  // Not one bit read twice: the narrow predicate disagrees with the wide one on
  // exactly the device shape that motivated splitting them.
  CHECK(b.UnifiedMemory() != b.DeviceMemoryIsHostAddressable());
}
