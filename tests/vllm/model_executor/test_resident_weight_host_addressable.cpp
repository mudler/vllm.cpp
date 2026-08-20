// ENG-EXPERT-STREAM-DEVICE W0f (issue #1299): on a platform whose kernels can
// DEREFERENCE host storage, does `ResidentWeight` hand back the host bytes
// instead of allocating a second copy of them?
//
// WHAT THE DEFECT IS, AND WHY A TOKEN GATE CANNOT SEE IT. `ResidentWeight`'s
// staging branch is a VERBATIM byte copy: `d.b.Alloc(w.bytes.size())`,
// `d.b.Copy(...)`, then a tensor over the copy with the same dtype, the same
// shape and the same (dropped) marker set. Nothing about the bytes changes. On a
// discrete device that copy is the whole point — the kernel cannot follow a host
// pointer. On a part where device memory IS host memory it buys nothing and
// costs a second full resident copy of every dense weight, out of the same RAM
// the first one came from. The tokens are identical either way, which is exactly
// why this is asserted as an ALLOCATION and never as an output.
//
// MEASURED (issue #1299, `dgx:gpu0`, GB10, seven runs). With W0's lane on,
// `Qwen3.8-2.4T-A95B UD-Q1_0` loads on `--device cuda` — 61.20 GiB resident,
// ~265 s — and then exhausts a 119.631 GiB box inside the FIRST forward, zero
// decode steps, every time. A 0.15 GiB slot arena died exactly where an
// 18.55 GiB one did, so the arena is not the cost; the growth was anonymous
// (`RssAnon` 8.1 -> 61.4 GB) while file-backed stayed flat, so the mapping is not
// pinned. About 39 GiB of that 61.20 is `attn_qkv` (21.56) and `ssm_out` (17.25),
// which the GDN V-head reorder makes `kTransformedWeight` and therefore expands
// to bf16 in OWNED host buffers — the split is measured, not derived, in
// `.agents/specs/expert-streaming.md`. The CPU arm pays that once and serves. The
// CUDA arm paid it twice and could not.
//
// THE PLATFORM THIS NEEDS DOES NOT EXIST ON A CPU TIER. The branch under test is
// selected by `needs_weight_staging() && host_memory_is_device_addressable()`,
// and exactly one machine this project can reach answers that pair (a GB10). A
// fake platform in the otherwise-unused kXPU slot supplies the bit, over a fake
// backend whose `UnifiedMemory() == true` and `DeviceMemoryIsHostAddressable()
// == false` — which is not an arbitrary pair but the GB10 CUDA backend's own
// answers (`src/vt/cuda/cuda_backend.cu:113` and the base default in
// `include/vt/backend.h:76`; a `cudaMalloc` pointer there is still not
// host-dereferenceable even though host and device address the same RAM).
//
// WHAT THIS FILE DOES NOT CLAIM. It does not claim to prove that a forward
// reaches `ResidentWeight`. `test_expert_stream_wiring` proves that through
// `Qwen3_5Model::Forward`, and the reachability mutation for this change is
// stated against that binary, not this one. What is proved here is which BRANCH
// the function takes, and that the predicate — not the device name, not the
// staging flag — is what selects it.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/owned_bytes.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::OwnedTensor;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// A backend over ordinary host memory, standing in for a device allocator. The
// point of malloc is that a STAGED weight is then a real, inspectable allocation
// at an address that differs from the weight's own bytes — which is how a case
// tells "staged" from "aliased" without a GPU.
class HostBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    ++allocs;
    alloc_bytes += bytes;
    return std::malloc(bytes == 0 ? 1 : bytes);
  }
  void Free(void* p) override { std::free(p); }
  void Memset(Queue&, void* p, int v, size_t bytes) override {
    std::memset(p, v, bytes);
  }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  void DestroyQueue(Queue&) override {}
  // TRUE, matching GB10's CUDA backend (cuda_backend.cu:113): host and device
  // address the same physical RAM.
  bool UnifiedMemory() const override { return true; }
  // FALSE, also matching GB10: a `cudaMalloc` pointer is NOT host-dereferenceable
  // there, which is the asymmetry this whole change turns on. It is also what
  // keeps `AdoptDeviceBytesAsHost` inert, so a staged weight stays visibly
  // staged instead of being folded back onto its host buffer by a second
  // mechanism and confusing what this file measures.
  bool DeviceMemoryIsHostAddressable() const override { return false; }

  // Counted, so a case can say HOW MANY allocations it observed rather than only
  // that a pointer was or was not null.
  int allocs = 0;
  size_t alloc_bytes = 0;
};

HostBackend& Fake() {
  static HostBackend b;
  return b;
}

// The GB10 shape: a platform that STAGES its ordinary weights and whose kernels
// can nevertheless dereference host storage. `host_addressable` is a settable
// field rather than a second registered platform because the platform registry
// is process-global — two registrations would fight, and one flag lets a case
// prove the predicate SELECTS by moving only the bit under test.
class FakeDevicePlatform final : public vllm::platforms::Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kXPU; }
  vt::Backend& backend() const override { return Fake(); }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return {12, 1};
  }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }
  bool needs_weight_staging() const override { return true; }
  bool host_memory_is_device_addressable() const override {
    return host_addressable;
  }

  bool host_addressable = true;
};

FakeDevicePlatform& Platform_() {
  static FakeDevicePlatform p;
  return p;
}

struct Registrar {
  Registrar() {
    vt::RegisterBackend(Device{DeviceType::kXPU, 0}, &Fake());
    vllm::platforms::RegisterPlatform(DeviceType::kXPU, &Platform_());
  }
};
const Registrar kRegistrar;

Queue XpuQueue() { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }

constexpr int64_t kN = 6;
constexpr int64_t kK = 8;

// A plain bf16 [N,K] projection — the shape every dense weight in the Qwen3.5
// forward arrives as, and the shape the ~39 GiB of transformed `attn_qkv` /
// `ssm_out` arrives as on the target checkpoint.
OwnedTensor MakeWeight(uint8_t tag) {
  OwnedTensor t;
  t.dtype = DType::kBF16;
  t.rank = 2;
  t.shape[0] = kN;
  t.shape[1] = kK;
  t.nk = true;
  std::vector<uint8_t> b(static_cast<size_t>(kN * kK) * 2);
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<uint8_t>((i * 7 + tag) & 0xFF);
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

// Restore `host_addressable` on EVERY exit path, including a REQUIRE that
// aborts the case body. Three cases below move the bit; a case that fails
// halfway used to leak `false` into every later case in this binary and turn one
// finding into a cascade of confusing ones.
struct PlatformArm {
  explicit PlatformArm(bool on) : prev(Platform_().host_addressable) {
    Platform_().host_addressable = on;
  }
  ~PlatformArm() { Platform_().host_addressable = prev; }
  bool prev;
};

}  // namespace

// THE CONSTANT ITSELF, PINNED TO ITS LITERAL.
//
// Every OTHER assertion about `kDeviceAliasAlignment` in this tree is written
// `% vllm::kDeviceAliasAlignment == 0` -- three times in the cases below, and
// twice more in `test_load_direct_upload.cpp`. Those check the BUFFERS, which is
// a real and separate property, and they stay. What none of them checks is the
// CONSTANT: they are tautologies in it. A fresh review measured that directly --
// lower the constant from 256 to 16 and every one of them still holds, all
// twelve cases in this file pass, the whole suite reports SUCCESS and exits 0.
// The promise the alias branch rests on would be gone with no gate saying so.
//
// WHY 256, AND WHY IT MAY NOT BE QUIETLY LOWERED. The header's argument
// (`qwen3_5_weights.h`, above this constant) is that the enumeration of what a
// kernel may dereference does not close, while cuBLASLt is PROMISED 256:
// `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES` defaults to 256 and this tree
// never overrides it -- `grep -rn MIN_ALIGNMENT src/vt/` returns nothing, and a
// fresh review re-confirmed that with a positive control. Matching the
// allocator's promise rather than the consumers is what makes handing a host
// pointer to a device kernel CORRECT. The tempting change is to lower the
// constant so the re-home memcpy is skipped; a 16-aligned arm even came back
// bit-exact at twelve probe shapes. A promise kept by luck at twelve shapes is
// not a promise, and this case is what makes that argument executable.
TEST_CASE("the alias alignment constant IS 256, the promise cuBLASLt is given") {
  CHECK(vllm::kDeviceAliasAlignment == 256u);
}

TEST_CASE("a host-addressable staging device ALIASES the weight and allocates nothing") {
  const PlatformArm arm(true);
  const OwnedTensor w = MakeWeight(/*tag=*/1);
  // A copy of the bytes taken BEFORE the call, because the call may move them.
  const std::vector<uint8_t> expect(w.bytes.data(), w.bytes.data() + w.bytes.size());
  Queue q = XpuQueue();
  const int allocs_before = Fake().allocs;

  const Tensor t = vllm::detail::StageWeightForTest(q, w);

  // THE #1299 ASSERTION, and it is an allocation rather than an output. `d_dev`
  // is set by, and only by, the staging branch; null means the second resident
  // copy of this weight does not exist.
  CHECK(w.d_dev == nullptr);
  CHECK(Fake().allocs == allocs_before);

  // ...and the tensor the kernel binds points at the weight's OWN host bytes.
  // Both halves matter: a null `d_dev` with a pointer somewhere else would be a
  // different defect wearing the same green.
  REQUIRE(t.data != nullptr);
  CHECK(t.data == static_cast<const void*>(w.bytes.data()));

  // THE SUBSTITUTION IS INDISTINGUISHABLE, which is the whole safety argument.
  // The pointer this branch hands a kernel is at least as aligned as the
  // `cudaMalloc` pointer it replaces, so no consumer — not the 16-byte
  // `cp.async` granule, not cuBLASLt's 256-byte minimum-alignment promise — can
  // tell that the staging copy is gone. A plain `std::vector<uint8_t>` does NOT
  // satisfy this on its own (glibc gives 16), so a green here is also the
  // statement that the re-homing ran.
  CHECK(reinterpret_cast<uintptr_t>(t.data) % vllm::kDeviceAliasAlignment == 0);

  // ...and the bytes did not change while being re-homed. Aliasing the WRONG
  // bytes is the one defect that would otherwise pass every assertion above.
  CHECK(std::memcmp(t.data, expect.data(), expect.size()) == 0);
  CHECK(w.bytes.size() == expect.size());

  // The tensor is otherwise EXACTLY what the staging branch produced: same
  // dtype, same device, same shape, same strides. This branch is that branch
  // minus the copy, and that is the whole claim.
  CHECK(t.dtype == w.dtype);
  CHECK(t.device.type == DeviceType::kXPU);
  CHECK(t.rank == 2);
  CHECK(t.shape[0] == kN);
  CHECK(t.shape[1] == kK);
  CHECK(t.stride[0] == kK);
  CHECK(t.stride[1] == 1);
}

TEST_CASE("the alias is stable across calls and still allocates nothing") {
  // `d_dev` is the staging branch's memo, so a branch that does not set it must
  // not become a per-call allocation instead. Two calls, one address, zero
  // allocations.
  const PlatformArm arm(true);
  const OwnedTensor w = MakeWeight(/*tag=*/2);
  Queue q = XpuQueue();
  const int allocs_before = Fake().allocs;

  const Tensor a = vllm::detail::StageWeightForTest(q, w);
  const Tensor b = vllm::detail::StageWeightForTest(q, w);

  CHECK(Fake().allocs == allocs_before);
  CHECK(w.d_dev == nullptr);
  CHECK(a.data == b.data);
  CHECK(a.data == static_cast<const void*>(w.bytes.data()));
}

TEST_CASE("a device that CANNOT read host memory stages exactly as before") {
  // THE DISCRETE ANSWER, and it must be byte-identical to today. A GPU whose
  // kernels cannot follow a host pointer gets the allocation, the copy, the
  // memo, and a tensor over the copy — the pre-W0f behaviour, unchanged. This is
  // the case that proves the predicate SELECTS rather than that the branch was
  // taken unconditionally.
  const PlatformArm arm(false);
  const OwnedTensor w = MakeWeight(/*tag=*/3);
  Queue q = XpuQueue();
  const int allocs_before = Fake().allocs;
  const size_t bytes_before = Fake().alloc_bytes;

  const Tensor t = vllm::detail::StageWeightForTest(q, w);

  REQUIRE(w.d_dev != nullptr);
  CHECK(Fake().allocs == allocs_before + 1);
  CHECK(Fake().alloc_bytes == bytes_before + w.bytes.size());
  CHECK(t.data == w.d_dev.get());
  CHECK(t.data != static_cast<const void*>(w.bytes.data()));
  // The staged copy holds the same bytes, which is what makes the aliasing arm
  // above a legitimate substitution rather than a different weight.
  CHECK(std::memcmp(t.data, w.bytes.data(), w.bytes.size()) == 0);
}

TEST_CASE("the aliasing branch keeps the elem_kn_repacked refusal") {
  // VT_CPU_ELEM_KN_REPACK transposes the buffer to [K,N] and ONLY the CPU
  // MatmulBTKernel honours the marker. Where the bytes live has nothing to do
  // with which kernel reads them: on a host-addressable device the reader is
  // still the DEVICE kernel, which would read transposed bytes as [N,K] and
  // produce garbage silently. The refusal therefore has to survive the new
  // branch, and this is the case that says so.
  const PlatformArm arm(true);
  OwnedTensor w = MakeWeight(/*tag=*/4);
  w.elem_kn_repacked = true;
  Queue q = XpuQueue();

  CHECK_THROWS_WITH_AS(
      vllm::detail::StageWeightForTest(q, w),
      doctest::Contains("an elem_kn_repacked ([K,N]) weight reached device staging"),
      std::runtime_error);
}

TEST_CASE("an i8mm-repacked weight reaching device residency is refused BY NAME") {
  // A DEFECT FOUND WHILE WRITING W0f (issue #1320), fixed in the same flow, and
  // it predates this change on both branches.
  //
  // `VT_CPU_QUANT_REPACK` rewrites a Q8_0 weight into the `block_q8_0x4` i8mm
  // interleave at load. Only the CPU `MatmulBTKernel` understands that layout;
  // the CUDA quant dot reads plain `block_q8_0`. Its sibling transform
  // `elem_kn_repack` has BOTH a CPU-platform gate in the loader policy and the
  // refusal above — `quant_repack` had NEITHER. It rides `QuantRepackActive()`,
  // which probes the HOST CPU for Arm i8mm, so an aarch64 box doing
  // `--device cuda` (which is precisely the target box) can repack a weight and
  // then hand it to a kernel that misreads it. The result is wrong tokens, not a
  // crash, and no gate in this tree could see it.
  //
  // Harmless on the target checkpoint as measured — one Q8_0 tensor, 0.01% of
  // parameters, and the instrumented load recorded `quant_repack = 0` — which is
  // why this is a tripwire beside its sibling rather than a campaign.
  const PlatformArm arm(true);
  OwnedTensor w = MakeWeight(/*tag=*/9);
  w.repacked = true;
  Queue q = XpuQueue();

  CHECK_THROWS_WITH_AS(
      vllm::detail::StageWeightForTest(q, w),
      doctest::Contains("an i8mm-repacked (block_q8_0x4) weight reached device residency"),
      std::runtime_error);

  // ...and on the DISCRETE arm too, because the kernel that misreads it is the
  // same kernel either way. Where the bytes live was never the question.
  {
    const PlatformArm discrete(false);
    OwnedTensor d = MakeWeight(/*tag=*/10);
    d.repacked = true;
    CHECK_THROWS_WITH_AS(
        vllm::detail::StageWeightForTest(q, d),
        doctest::Contains("an i8mm-repacked (block_q8_0x4) weight reached device residency"),
        std::runtime_error);
  }
}

TEST_CASE("the aliasing branch keeps the streamed-tower refusal") {
  // Same reasoning as the case above, for the W0c tripwire. A tower the expert
  // lane claimed must never be consumed WHOLE, and "whole" is a statement about
  // the tower, not about the allocator: reaching here at all means the lane was
  // defeated. Cheap on this platform and catastrophic on the other, so it fails
  // by name on both.
  const PlatformArm arm(true);
  OwnedTensor w = MakeWeight(/*tag=*/5);
  w.expert_streamed = true;
  Queue q = XpuQueue();

  CHECK_THROWS_WITH_AS(
      vllm::detail::StageWeightForTest(q, w),
      doctest::Contains("a STREAMED expert tower reached device staging"),
      std::runtime_error);
}

TEST_CASE("an ALREADY-ALIGNED buffer is aliased in place, with no second copy") {
  // The free case, and it needs its own assertion because the re-homing above
  // would satisfy every other check in this file while quietly copying a weight
  // that did not need copying. A GGUF mmap borrow whose tensor offset happens to
  // be a multiple of 256 lands here, and so does the SECOND call for any weight
  // the first call re-homed.
  const PlatformArm arm(true);
  OwnedTensor w = MakeWeight(/*tag=*/7);
  const size_t nb = w.bytes.size();
  // An aligned block, borrowed, standing exactly where a lucky mmap offset
  // would. `Borrow` needs a keep-alive, and the block IS the keep-alive.
  void* aligned = ::operator new(nb, std::align_val_t{vllm::kDeviceAliasAlignment});
  std::memcpy(aligned, w.bytes.data(), nb);
  std::shared_ptr<const void> keep(
      static_cast<const void*>(aligned), [](const void* p) {
        ::operator delete(const_cast<void*>(p),
                          std::align_val_t{vllm::kDeviceAliasAlignment});
      });
  w.bytes = vllm::OwnedBytes::Borrow(static_cast<const uint8_t*>(aligned), nb, keep);
  REQUIRE(reinterpret_cast<uintptr_t>(w.bytes.data()) %
              vllm::kDeviceAliasAlignment == 0);
  Queue q = XpuQueue();
  const int allocs_before = Fake().allocs;

  CHECK(vllm::MakeHostBytesDeviceAliasable(w));
  // The SAME address: nothing was moved and nothing was allocated.
  CHECK(w.bytes.data() == static_cast<const uint8_t*>(aligned));

  const Tensor t = vllm::detail::StageWeightForTest(q, w);
  CHECK(w.d_dev == nullptr);
  CHECK(Fake().allocs == allocs_before);
  CHECK(t.data == static_cast<const void*>(aligned));
}

TEST_CASE("a MISALIGNED BORROW is not re-homed, and stages instead") {
  // THE CASE THAT KEEPS THIS CHANGE FROM BACKFIRING. A borrow owns no anonymous
  // pages — it is a clean, file-backed GGUF mapping, or a tied
  // token_embd/lm_head pair's single shared expansion. Copying it into an
  // aligned anonymous block to satisfy the alias would CREATE the residency this
  // row exists to remove, and would break the tie. GGUF guarantees only 32-byte
  // tensor alignment, so this is a real population and not a hypothetical.
  //
  // The correct answer is to decline, and let the (unchanged) staging branch
  // copy it into device memory, where its file pages stay reclaimable.
  const PlatformArm arm(true);
  OwnedTensor w = MakeWeight(/*tag=*/8);
  const size_t nb = w.bytes.size();
  auto backing = std::make_shared<std::vector<uint8_t>>(nb + vllm::kDeviceAliasAlignment);
  // Deliberately off by 8: aligned enough for the element type, nowhere near 256.
  uint8_t* base = backing->data();
  uint8_t* off = base + (vllm::kDeviceAliasAlignment -
                         (reinterpret_cast<uintptr_t>(base) %
                          vllm::kDeviceAliasAlignment)) + 8;
  std::memcpy(off, w.bytes.data(), nb);
  w.bytes = vllm::OwnedBytes::Borrow(
      off, nb, std::static_pointer_cast<const void>(backing));
  REQUIRE(w.bytes.borrowed());
  REQUIRE(reinterpret_cast<uintptr_t>(w.bytes.data()) %
              vllm::kDeviceAliasAlignment != 0);
  Queue q = XpuQueue();
  const int allocs_before = Fake().allocs;

  CHECK_FALSE(vllm::MakeHostBytesDeviceAliasable(w));
  // The borrow is UNTOUCHED — same address, still borrowed.
  CHECK(w.bytes.data() == off);
  CHECK(w.bytes.borrowed());

  const Tensor t = vllm::detail::StageWeightForTest(q, w);
  REQUIRE(w.d_dev != nullptr);
  CHECK(Fake().allocs == allocs_before + 1);
  CHECK(t.data == w.d_dev.get());
  CHECK(std::memcmp(t.data, off, nb) == 0);
}

TEST_CASE("an ALIASED weight's host mirror is NOT redundant, so nothing may free it") {
  // THE USE-AFTER-FREE A FRESH REVIEW CAUGHT (#1299). `MoeBlockBf16Cuda`
  // captures `ResidentWeight(...).data` for every expert into a DEVICE-resident
  // pointer table, uploads the table once, and then releases the host mirrors.
  // Its own comment justified that with "once the device copy exists it is
  // authoritative and nothing reads the host bytes again", which held while
  // `ResidentWeight` had two behaviours. It has three: this branch ALIASES, so
  // the captured pointers ARE `w.bytes.data()`, and the release frees memory the
  // resident table still points at for the model's lifetime, from inside
  // captured graphs. The reviewer demonstrated it with a scratch case that takes
  // SIGSEGV.
  //
  // THIS CASE DOES NOT DEREFERENCE FREED MEMORY, deliberately: a segfault is a
  // red that also destroys the rest of the binary's report, and a gate should
  // fail by assertion. It asserts the DECISION instead, on both arms, which is
  // the thing the production site now asks.
  const PlatformArm arm(true);
  const OwnedTensor aliased = MakeWeight(/*tag=*/11);
  Queue q = XpuQueue();
  const Tensor t = vllm::detail::StageWeightForTest(q, aliased);

  REQUIRE(aliased.d_dev == nullptr);
  REQUIRE(t.data == static_cast<const void*>(aliased.bytes.data()));
  // There is no device copy, so the host bytes are the ONLY copy and releasing
  // them would free what the kernel reads.
  CHECK_FALSE(vllm::HostMirrorIsRedundant(aliased));

  // The discrete arm is the other half: a staged weight DOES have an
  // authoritative device copy, and the release that predates W0f stays correct
  // for it. Without this half the invariant could be satisfied by refusing every
  // release, which would silently undo a measured host-memory lever.
  {
    const PlatformArm discrete(false);
    const OwnedTensor staged = MakeWeight(/*tag=*/12);
    const Tensor dt = vllm::detail::StageWeightForTest(q, staged);
    REQUIRE(staged.d_dev != nullptr);
    CHECK(dt.data == staged.d_dev.get());
    CHECK(vllm::HostMirrorIsRedundant(staged));
  }
}

TEST_CASE("a weight with NOTHING to serve is refused by name, not aliased to null") {
  // THE LIFETIME PRECONDITION, stated in code. The aliasing branch hands out
  // `w.bytes.data()` and keeps no reference of its own, so it is correct only
  // while the weight owns those bytes. `ReleaseHost()` is the one operation that
  // takes them away — it is reachable for the routed-expert fp4/Marlin mirrors
  // (`ShouldReleaseHostWeights`, qwen3_5.cpp) though not for the dense weights
  // this branch serves. If that ever changes, the failure without this check is
  // a null weight pointer inside a kernel, which is a segfault at best and wrong
  // tokens at worst. With it, it is one legible sentence.
  const PlatformArm arm(true);
  OwnedTensor w = MakeWeight(/*tag=*/6);
  w.ReleaseHost();
  REQUIRE(w.bytes.empty());
  REQUIRE(w.d_dev == nullptr);
  Queue q = XpuQueue();

  CHECK_THROWS_WITH_AS(
      vllm::detail::StageWeightForTest(q, w),
      doctest::Contains("no host bytes and no device copy"), std::runtime_error);
}

TEST_CASE("a released host mirror with a DEVICE copy is still served, not refused") {
  // THE OTHER HALF OF THAT PRECONDITION, and W0f is what created the population
  // it protects (a fresh review of #1299 found the regression). A weight whose
  // host mirror is gone but whose `d_dev` is populated has always been served —
  // by the `if (!w.d_dev)` memo, which returns the device tensor. W0f put the
  // "no host bytes" refusal ABOVE that memo, so the same weight began to throw,
  // and the justification written beside it ("`ReleaseHost()` is not reachable
  // for the dense weights this branch serves") is true of the dense weights and
  // false of the expert weights the same function serves: a misaligned GGUF
  // expert borrow declines the alias, stages, gets a `d_dev`, and is then
  // released by the guarded loop beside the `MoeBlockBf16Cuda` pointer capture.
  //
  // Built on the arm that STAGES, then released, then asked again on the ALIASING
  // arm — which is the ordering that reproduces it, because the refusal only
  // lives on the aliasing side.
  OwnedTensor w = MakeWeight(/*tag=*/13);
  Queue q = XpuQueue();
  {
    const PlatformArm discrete(false);
    const Tensor staged = vllm::detail::StageWeightForTest(q, w);
    REQUIRE(w.d_dev != nullptr);
    REQUIRE(staged.data == w.d_dev.get());
  }
  w.ReleaseHost();
  REQUIRE(w.bytes.empty());
  REQUIRE(w.d_dev != nullptr);

  const PlatformArm arm(true);
  const int allocs_before = Fake().allocs;
  const Tensor t = vllm::detail::StageWeightForTest(q, w);

  // Served from the copy that exists, with nothing allocated and nothing thrown.
  CHECK(t.data == w.d_dev.get());
  CHECK(Fake().allocs == allocs_before);
  CHECK(t.dtype == w.dtype);
  CHECK(t.shape[0] == kN);
  CHECK(t.shape[1] == kK);
}

TEST_CASE("an F32 UPCAST does not make an aliased weight's host mirror redundant") {
  // THE SECOND INSTANCE OF THE USE-AFTER-FREE (a fresh review of #1299 found it
  // after the first was fixed). `ReleaseResidentQwen3_5DenseHostWeights` guarded
  // on `d_dev || d_dev_f32`, and `d_dev_f32` is a bf16->f32 UPCAST into a
  // separate device allocation — not a copy of these bytes, and never able to
  // stand in for them. The disjunction was harmless only while every weight with
  // an upcast also had a `d_dev`.
  //
  // `PrepareBf16Resident` passes exactly four weights to BOTH `raw()` and
  // `f32()`: `gdn.conv1d_weight`, `gdn.norm_weight`, `attn.q_norm` and
  // `attn.k_norm`. On the aliasing arm `raw()` leaves `d_dev` null while `f32()`
  // sets `d_dev_f32`, so the guard passed and freed the bytes the aliased raw
  // tensor points at.
  //
  // `d_dev_f32` is set directly here, which is the state that pairing produces:
  // `ResidentWeightF32` is private to qwen3_5.cpp and has no gate seam, and
  // inventing one would add test-only surface to observe a field the production
  // release site reads directly.
  const PlatformArm arm(true);
  OwnedTensor w = MakeWeight(/*tag=*/14);
  Queue q = XpuQueue();
  const Tensor t = vllm::detail::StageWeightForTest(q, w);
  REQUIRE(w.d_dev == nullptr);
  REQUIRE(t.data == static_cast<const void*>(w.bytes.data()));
  w.d_dev_f32 = std::shared_ptr<void>(reinterpret_cast<void*>(1), [](void*) {});

  // The DECISION, which is what the release site now asks and what a mutation of
  // the invariant moves.
  CHECK_FALSE(vllm::HostMirrorIsRedundant(w));

  // ...and the PRODUCTION release, driven end to end over the four-weight
  // pairing that creates the state. Nothing may be freed, and the aliased
  // pointer the kernel holds must still be the weight's own bytes afterwards.
  vllm::Qwen3_5DenseWeights weights;
  weights.layers.resize(1);
  vllm::Qwen3_5DenseLayerWeights& layer = weights.layers[0];
  layer.is_linear_attention = true;
  layer.gdn.conv1d_weight = std::move(w);
  const void* aliased_at = static_cast<const void*>(layer.gdn.conv1d_weight.bytes.data());
  REQUIRE(t.data == aliased_at);

  CHECK(vllm::ReleaseResidentQwen3_5DenseHostWeights(weights) == 0);
  CHECK(layer.gdn.conv1d_weight.HasHostBytes());
  CHECK(static_cast<const void*>(layer.gdn.conv1d_weight.bytes.data()) == aliased_at);

  // THE DISCRETE HALF, without which the invariant could be satisfied by
  // refusing every release — silently undoing a measured host-memory lever. A
  // STAGED weight has an authoritative device copy and is still freed.
  vllm::Qwen3_5DenseWeights staged_weights;
  staged_weights.layers.resize(1);
  vllm::Qwen3_5DenseLayerWeights& staged_layer = staged_weights.layers[0];
  staged_layer.is_linear_attention = true;
  {
    const PlatformArm discrete(false);
    staged_layer.gdn.conv1d_weight = MakeWeight(/*tag=*/15);
    (void)vllm::detail::StageWeightForTest(q, staged_layer.gdn.conv1d_weight);
  }
  REQUIRE(staged_layer.gdn.conv1d_weight.d_dev != nullptr);
  staged_layer.gdn.conv1d_weight.d_dev_f32 =
      std::shared_ptr<void>(reinterpret_cast<void*>(1), [](void*) {});
  const size_t expect_freed = staged_layer.gdn.conv1d_weight.bytes.size();
  CHECK(vllm::ReleaseResidentQwen3_5DenseHostWeights(staged_weights) == expect_freed);
  CHECK_FALSE(staged_layer.gdn.conv1d_weight.HasHostBytes());
}
