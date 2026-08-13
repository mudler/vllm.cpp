// ROCm backend skeleton gates (BACKEND-ROCM, W0). Newly authored — vLLM has no
// C++ ROCm backend tests to port. Mirrors tests/vt/test_metal_backend.cpp, which
// mirrors tests/vt/test_backend.cpp, so all three read side by side.
//
// RUN STATE, per issue #41: the W0 cases in this file ran green on community
// boards — gfx1151, gfx1103, gfx1100 and gfx1201 (5 cases, 1044 assertions in
// the posted tables). The two approach-(b) cases (alloc path / host-readable)
// have NEVER RUN: no AMD GPU exists on the authoring machine. The file is
// LINKED into a test binary only in a HIP build (tests/CMakeLists.txt gates it
// on VLLM_CPP_HIP) but COMPILED everywhere: a non-HIP build object-compiles it
// as a bit-rot guard (see the CMake block next to the ROCm sources), so its
// types are checked on CI even with no ROCm installed. Compiled is not run. If
// a new case fails on your board, that is far more likely a bug in the blind
// change than in your setup — paste the output into
// https://github.com/mudler/vllm.cpp/issues/41 with the arch it printed.
//
// Deliberately plain C++ with no HIP header: every assertion goes through the
// public vt:: / vllm::platforms:: seams. If the skeleton needed HIP in a test to
// be checkable, the seam would be leaking.
//
// NOT HERE: cross-device numeric equality against the CPU oracle. That lives in
// tests/vt/test_backend_cross_device.cpp, which discovers every registered
// non-CPU backend and so covers ROCm automatically — including the RmsNorm this
// skeleton registers, at NMSE <= 5e-4. Run both.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_arch.h"
#include "vt/rocm/rocm_runtime.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
// Every test is a no-op when the build has HIP but the box has no AMD GPU (a
// contributor cross-compiling, or CI). Skipping is correct; failing would be a
// lie about the hardware.
bool NoDevice() { return !vt::rocm::DeviceAvailable(); }
}  // namespace

TEST_CASE("ROCm backend registers when a device is present") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);

  // The arch string is the single most useful thing a first bug report can
  // carry, so print it unconditionally rather than only on failure.
  const std::string arch = vt::rocm::DeviceArchName(0);
  MESSAGE("ROCm device 0 gcnArchName: ", arch);
  CHECK_FALSE(arch.empty());

  // The capability must agree with the parse the CPU-tier test already gates
  // (tests/vt/test_rocm_arch.cpp). A disagreement means the backend took its
  // fallback path (props.major/.minor) because the string did not parse, which
  // is worth knowing loudly.
  const auto parsed = vt::rocm::CapabilityFromGcnArch(arch);
  CHECK_MESSAGE(parsed.has_value(), "gcnArchName did not parse: ", arch);
  if (parsed) {
    CHECK(rocm.DeviceCapabilityMajor() == parsed->first);
    CHECK(rocm.DeviceCapabilityMinor() == parsed->second);
  }
}

TEST_CASE("alloc / copy / memset round-trip through the ROCm backend") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();
  REQUIRE(q.device.type == DeviceType::kROCM);

  constexpr size_t kN = 1024;
  std::vector<float> host(kN);
  for (size_t i = 0; i < kN; ++i) host[i] = static_cast<float>(i) * 0.5f;
  std::vector<float> back(kN, -1.0f);

  void* dev = rocm.Alloc(kN * sizeof(float));
  REQUIRE(dev != nullptr);
  // Backend::Alloc owes >= 64B alignment (StepArena depends on it).
  CHECK((reinterpret_cast<uintptr_t>(dev) % 64) == 0);

  rocm.Copy(q, dev, host.data(), kN * sizeof(float));
  rocm.Copy(q, back.data(), dev, kN * sizeof(float));
  rocm.Synchronize(q);
  // A pure copy path is BIT-exact — nothing is reassociated, so anything less
  // would be hiding a bug (the contract in test_backend_cross_device.cpp).
  CHECK(std::memcmp(host.data(), back.data(), kN * sizeof(float)) == 0);

  rocm.Memset(q, dev, 0, kN * sizeof(float));
  rocm.Copy(q, back.data(), dev, kN * sizeof(float));
  rocm.Synchronize(q);
  for (size_t i = 0; i < kN; ++i) REQUIRE(back[i] == 0.0f);

  rocm.Free(dev);
  rocm.DestroyQueue(q);
  CHECK(q.handle == nullptr);
}

TEST_CASE("the reference tier follows UnifiedMemory, which is the memory-safety gate") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  const bool unified = rocm.UnifiedMemory();
  MESSAGE("ROCm device 0 UnifiedMemory(): ", unified);

  // THE property this whole skeleton turns on. The portable CPU reference tier
  // serves unimplemented ops by running CPU kernels on the SAME pointers, which
  // is valid only where host and device memory alias. On an APU that is what
  // makes a model run end to end with one registered kernel; on a discrete card
  // it would be memory corruption, so eligibility must track UnifiedMemory()
  // exactly — not the device type, not the arch name.
  CHECK(vt::ReferenceTierEligible(DeviceType::kROCM) == unified);

  if (unified) {
    // Installing is idempotent and must not displace the native RmsNorm: the
    // tier registers strictly below any native kernel.
    const int installed = vt::RegisterReferenceTier(DeviceType::kROCM);
    MESSAGE("reference-tier ops installed for kROCM: ", installed);
    CHECK(installed > 0);
  }
}

TEST_CASE("approach (b): the alloc path and UnifiedMemory() move together") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  const bool integrated = vt::rocm::IntegratedDevice(0);
  const bool managed = vt::rocm::ManagedAllocActive(0);
  const bool unified = rocm.UnifiedMemory();
  // Printed unconditionally: this triple is the first thing a bring-up report
  // on issue #41 should carry.
  MESSAGE("ROCm device 0 integrated: ", integrated, " managed-alloc: ", managed,
          " UnifiedMemory(): ", unified);

  if (!integrated) {
    // DISCRETE (7900 XTX, R9700): the managed branch must be provably dead and
    // the unified claim false — the byte-identical-to-W0 half of the (b)
    // decision. A CPU fallback here would be memory corruption, not a slow
    // path, so these two CHECKs are the memory-safety gate itself.
    CHECK_FALSE(managed);
    CHECK_FALSE(unified);
    return;
  }
  // INTEGRATED. Every board measured on issue #41 (gfx1151 F6 attribute table,
  // gfx1103 confirmation) reports ManagedMemory=1 + ConcurrentManagedAccess=1,
  // so the managed branch is active and UnifiedMemory() is true by
  // construction. An integrated device that probes NOT managed-capable would
  // fail here: that is a hardware class the (b) fix does not cover, and a loud
  // failure carrying the triple above is more useful than a silent skip —
  // please post it on https://github.com/mudler/vllm.cpp/issues/41.
  CHECK_MESSAGE(managed,
                "integrated device without the managed-alloc branch: "
                "ManagedMemory or ConcurrentManagedAccess probed 0 — post the "
                "triple above on issue #41");
  CHECK_MESSAGE(unified == managed,
                "UnifiedMemory() must be true EXACTLY when the managed branch "
                "is active on an XNACK-less integrated part");
}

TEST_CASE("unified path: a kernel-written value is host-readable with no copy") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  // On a discrete card a host dereference of Backend::Alloc memory is
  // undefined behavior, so this gate only exists where UnifiedMemory() claims
  // it is safe — which is exactly the claim under test.
  if (!rocm.UnifiedMemory()) return;

  // Issue #41 F6's decisive experiment ("a kernel writes ...; the host reads
  // ... back directly, no hipMemcpy"), turned into the standing gate. The
  // kernel is the one op this backend registers (RmsNorm), so the file stays
  // free of HIP: host WRITES the inputs directly (what a reference-tier CPU
  // kernel does), the native device kernel reads them, and the host READS the
  // device-written output directly. Same golden row as the native-RmsNorm case
  // below, so a numeric mismatch here isolates COHERENCE, not arithmetic.
  Queue q = rocm.CreateQueue();
  float* dx = static_cast<float*>(rocm.Alloc(2 * sizeof(float)));
  float* dw = static_cast<float*>(rocm.Alloc(2 * sizeof(float)));
  float* dout = static_cast<float*>(rocm.Alloc(2 * sizeof(float)));
  REQUIRE(dx != nullptr);
  REQUIRE(dw != nullptr);
  REQUIRE(dout != nullptr);

  // Host writes, no Copy staging.
  dx[0] = 3.0f;
  dx[1] = 4.0f;
  dw[0] = 2.0f;
  dw[1] = 0.5f;
  dout[0] = -1.0f;
  dout[1] = -1.0f;

  const Device dev{DeviceType::kROCM, 0};
  Tensor tx = Tensor::Contiguous(dx, DType::kF32, dev, {1, 2});
  Tensor tw = Tensor::Contiguous(dw, DType::kF32, dev, {2});
  Tensor to = Tensor::Contiguous(dout, DType::kF32, dev, {1, 2});
  vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{0.0f, false});
  rocm.Synchronize(q);

  // Host reads the device-written output directly, no Copy back. If this
  // faults or reads the -1.0f sentinels, UnifiedMemory() lied — the exact
  // failure mode approach (b) exists to make impossible.
  CHECK(dout[0] == doctest::Approx(1.697056f));
  CHECK(dout[1] == doctest::Approx(0.565685f));

  rocm.Free(dx);
  rocm.Free(dw);
  rocm.Free(dout);
  rocm.DestroyQueue(q);
}

TEST_CASE("RmsNorm is registered natively, and the tier does not displace it") {
  if (NoDevice()) return;
  // Seam 3: the op table. One op today (src/vt/rocm/rocm_ops.hip).
  CHECK(vt::OpRegistered(vt::OpId::kRmsNorm, DeviceType::kROCM));

  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();

  // x = [3,4]; mean(x^2) = 12.5; rms = sqrt(12.5); w = [2, 0.5]
  // out = [3/3.53553*2, 4/3.53553*0.5] = [1.697056, 0.565685]
  // Same golden row as tests/vt/test_ops_rmsnorm.cpp, so the two backends are
  // pinned to one arithmetic statement rather than to each other.
  const std::vector<float> x = {3.0f, 4.0f};
  const std::vector<float> w = {2.0f, 0.5f};
  std::vector<float> out(2, 0.0f);

  void* dx = rocm.Alloc(x.size() * sizeof(float));
  void* dw = rocm.Alloc(w.size() * sizeof(float));
  void* dout = rocm.Alloc(out.size() * sizeof(float));
  rocm.Copy(q, dx, x.data(), x.size() * sizeof(float));
  rocm.Copy(q, dw, w.data(), w.size() * sizeof(float));

  const Device dev{DeviceType::kROCM, 0};
  Tensor tx = Tensor::Contiguous(dx, DType::kF32, dev, {1, 2});
  Tensor tw = Tensor::Contiguous(dw, DType::kF32, dev, {2});
  Tensor to = Tensor::Contiguous(dout, DType::kF32, dev, {1, 2});
  vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{0.0f, false});

  rocm.Copy(q, out.data(), dout, out.size() * sizeof(float));
  rocm.Synchronize(q);
  CHECK(out[0] == doctest::Approx(1.697056f));
  CHECK(out[1] == doctest::Approx(0.565685f));

  // The native kernel RAN — it was not silently served by the CPU fallback. A
  // reference-tier hit here would mean the registration in rocm_ops.hip did not
  // take effect, and the numbers above would still pass, which is exactly the
  // kind of pass that teaches nothing.
  CHECK(vt::GetReferenceTierHits() == 0);

  rocm.Free(dx);
  rocm.Free(dw);
  rocm.Free(dout);
  rocm.DestroyQueue(q);
}

TEST_CASE("the ROCm platform self-registers and is selected over CPU") {
  if (NoDevice()) return;
  using vllm::platforms::CurrentPlatform;
  using vllm::platforms::GetPlatform;
  using vllm::platforms::HasPlatform;

  REQUIRE(HasPlatform(DeviceType::kROCM));
  const auto& rocm = GetPlatform(DeviceType::kROCM);
  CHECK(rocm.device_type() == DeviceType::kROCM);
  CHECK(rocm.get_device_capability().present());
  CHECK(rocm.supported_dtypes().size() == 3);

  // Accelerator-first: on an AMD box with no CUDA, the process platform is ROCm.
  // This is what the kCurrentPriority walk in platform.cpp decides, and the CPU
  // tier gates its membership (tests/vllm/platforms/test_platform.cpp).
  if (!HasPlatform(DeviceType::kCUDA)) {
    CHECK(&CurrentPlatform() == &rocm);
  }

  // W0 registers no attention backend, so the priority list is EMPTY and
  // selection throws loudly rather than naming a backend whose kernels are
  // absent. When M3 lands ROCM_ATTN/TRITON_ATTN this flips, and this assertion
  // is the reminder to update it deliberately.
  CHECK(rocm.get_attn_backend_priority({}).empty());
}

// Mirrors "CUDA backend: graph capture/replay re-executes captured ops" in
// tests/vt/test_cuda_backend.cpp assertion for assertion. Same shape, same
// persistent-buffer contract, hipGraph underneath.
//
// Step 5 is the load-bearing one and the reason this test exists. Replaying
// must RE-EXECUTE the captured copy over the persistent buffers, not replay a
// snapshot of their contents — that is precisely how a decode graph picks up
// each new token's inputs. A capture that bakes values instead of addresses
// passes step 4 and fails step 5, which is the silent-correctness-bug shape
// rocm_backend.hip's scope note warns about.
TEST_CASE("ROCm backend: graph capture/replay re-executes captured ops") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  CHECK(rocm.SupportsGraphCapture());

  Queue q = rocm.CreateQueue();
  constexpr size_t kBytes = 64 * 1024;

  // Allocated ONCE; the pointers stay fixed across every replay below. Only
  // their CONTENTS change — the capture contract.
  void* src = rocm.Alloc(kBytes);
  void* dst = rocm.Alloc(kBytes);

  std::vector<unsigned char> pattern_a(kBytes, 0x11);
  std::vector<unsigned char> pattern_b(kBytes, 0x22);
  std::vector<unsigned char> back(kBytes, 0);

  rocm.Copy(q, src, pattern_a.data(), kBytes);
  rocm.Memset(q, dst, 0, kBytes);
  rocm.Synchronize(q);

  // Recorded, NOT executed: dst must still be zero after EndCapture.
  rocm.BeginCapture(q);
  rocm.Copy(q, dst, src, kBytes);
  rocm.EndCapture(q);
  rocm.Copy(q, back.data(), dst, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x00);

  // Replay #1 -> pattern A. Proves the graph ran at all.
  rocm.Replay(q);
  rocm.Synchronize(q);
  rocm.Copy(q, back.data(), dst, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x11);
  CHECK(back.back() == 0x11);

  // Mutate src in place (SAME address) -> replay must observe the new contents.
  rocm.Copy(q, src, pattern_b.data(), kBytes);
  rocm.Replay(q);
  rocm.Synchronize(q);
  rocm.Copy(q, back.data(), dst, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x22);
  CHECK(back.back() == 0x22);

  // Handle variant — the path decode graphs actually take, since they keep one
  // exec per padded batch size rather than a single stored graph. Captures
  // into a DIFFERENT destination (dst2) than the stored-graph path above
  // (dst), not the same op replayed twice: a mutation that returns the stale
  // member exec_ (still the stored-graph path's Copy(dst, src)) instead of the
  // freshly captured local exec must be distinguishable from the correct
  // behaviour, and only fails here because the two graphs write different
  // buffers.
  void* dst2 = rocm.Alloc(kBytes);
  rocm.Copy(q, src, pattern_a.data(), kBytes);
  rocm.Memset(q, dst2, 0, kBytes);
  rocm.Synchronize(q);

  rocm.BeginCapture(q);
  rocm.Copy(q, dst2, src, kBytes);
  void* graph = rocm.EndCaptureGraph(q);
  REQUIRE(graph != nullptr);

  rocm.ReplayGraph(q, graph);
  rocm.Synchronize(q);
  rocm.Copy(q, back.data(), dst2, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x11);

  rocm.Copy(q, src, pattern_b.data(), kBytes);
  rocm.ReplayGraph(q, graph);
  rocm.Synchronize(q);
  rocm.Copy(q, back.data(), dst2, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x22);
  CHECK(back.back() == 0x22);

  rocm.DestroyGraph(graph);
  rocm.Free(src);
  rocm.Free(dst);
  rocm.Free(dst2);
  rocm.DestroyQueue(q);
}

// The capture contract's allocation clause, asserted rather than assumed
// (.agents/specs/rocm-decode-graph.md D1). hipBLASLt sizes its workspace lazily
// inside the GEMM path — LtWorkspace() in rocm_matmul_hipblaslt.hip does
// hipFree+hipMalloc when a shape needs more than the current high-water mark —
// and hipblasCreate() likewise initialises on first use. Both are illegal
// mid-capture.
//
// MEASURED on gfx1200 during W1: capturing a cold GEMM fails loudly, with
// `hipMalloc: operation not permitted when stream is capturing` (and hipFree,
// and hipblasCreate INTERNAL_ERROR) — never silent corruption. Running the
// identical GEMM once beforehand grows the workspace and creates the handle, so
// the in-capture call is a pure pool hit.
//
// This case pins the MITIGATION, not the hazard: it asserts that a pre-warmed
// GEMM captures and replays correctly. Deliberately not asserting that the cold
// path throws — a future capture-safe allocator would be an improvement, and a
// test that forbade it would be a ratchet in the wrong direction.
TEST_CASE("ROCm backend: a pre-warmed GEMM captures and replays") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  if (!rocm.SupportsGraphCapture()) return;

  Queue q = rocm.CreateQueue();
  const Device dev{DeviceType::kROCM, 0};
  constexpr int kM = 1, kN = 2048, kK = 2048;  // a decode-shaped GEMM

  const std::vector<float> ha(kM * kK, 0.01f);
  const std::vector<float> hb(kN * kK, 0.02f);
  const float expect = 0.01f * 0.02f * static_cast<float>(kK);

  void* da = rocm.Alloc(ha.size() * sizeof(float));
  void* db = rocm.Alloc(hb.size() * sizeof(float));
  void* dc = rocm.Alloc(kM * kN * sizeof(float));
  rocm.Copy(q, da, ha.data(), ha.size() * sizeof(float));
  rocm.Copy(q, db, hb.data(), hb.size() * sizeof(float));
  rocm.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, DType::kF32, dev, {kM, kK});
  Tensor tb = Tensor::Contiguous(db, DType::kF32, dev, {kN, kK});
  Tensor tc = Tensor::Contiguous(dc, DType::kF32, dev, {kM, kN});

  // Pre-warm: grows LtWorkspace's cap and creates the hipBLAS handle.
  vt::MatmulBT(q, tc, ta, tb);
  rocm.Synchronize(q);
  rocm.Memset(q, dc, 0, kM * kN * sizeof(float));
  rocm.Synchronize(q);

  rocm.BeginCapture(q);
  vt::MatmulBT(q, tc, ta, tb);
  rocm.EndCapture(q);

  rocm.Replay(q);
  rocm.Synchronize(q);
  std::vector<float> back(kM * kN, 0.0f);
  rocm.Copy(q, back.data(), dc, back.size() * sizeof(float));
  rocm.Synchronize(q);
  CHECK(back.front() == doctest::Approx(expect).epsilon(0.01));
  CHECK(back.back() == doctest::Approx(expect).epsilon(0.01));

  rocm.Free(da);
  rocm.Free(db);
  rocm.Free(dc);
  rocm.DestroyQueue(q);
}
