// Resident MoE/Marlin state must not outlive the weights it describes
// (issue #237).
//
// WHAT BROKE. Every resident constant these models build — per-expert device
// pointer arrays, Marlin repacks, token row maps — used to live in a `static
// std::unordered_map` keyed on the ADDRESS of the weight, built on first touch
// and never erased. That is correct for exactly one engine per process. Build a
// second `LoadedEngine` and the allocator can hand the new weights the address
// the old ones had; the new weights then inherit an entry already marked ready,
// whose device pointers were freed with the old engine.
//
// It did not crash. Nothing in this tree destroys the CUDA context, so the freed
// pointers stayed mapped and simply pointed at whatever the new engine had put
// there — reported as corrupted and zeroed output token ids from a later
// synthetic Qwen3.5 engine, only under a test ordering that builds more than one,
// and only intermittently.
//
// WHAT IS PINNED HERE. The invariant the fix establishes: residency is a member
// of the weights, so a freshly constructed weights block is unbuilt no matter
// what previously occupied its storage. These cases run on the CPU and need no
// device, because the defect was never in the kernels — it was in who owned the
// cache. The end-to-end proof that a second CUDA engine now decodes correctly is
// the repeated-engine gate on the GPU box; this is the guard that keeps the
// invariant from silently regressing on every build.
//
// The address-reuse case below is the one that matters: it reconstructs a weights
// block in the SAME storage a built-up block just vacated, which is precisely the
// aliasing the old map could not distinguish.
#include <doctest/doctest.h>

#include <memory>
#include <new>

#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/model_executor/models/laguna.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"

using vllm::Dots3NoteMoeWeights;
using vllm::LagunaMoeWeights;
using vllm::MoeBlockWeights;
using vllm::Nvfp4Weight;
using vllm::ResidentSlot;

namespace {

// Stand-in for a resident type. The real ones are implementation details of the
// model .cpp files; what this pins is the SLOT's ownership, which is generic.
struct FakeResident {
  int built = 0;
};

// Mirrors ResidentIn<> in qwen3_5.cpp: build on first use, reuse thereafter.
FakeResident& ResidentOf(const ResidentSlot& slot) {
  if (!slot.state) slot.state = std::make_shared<FakeResident>();
  return *static_cast<FakeResident*>(slot.state.get());
}

}  // namespace

TEST_CASE("a fresh MoE block owns no resident state") {
  MoeBlockWeights w;
  CHECK(w.resident_fused.state == nullptr);
  CHECK(w.resident_bf16.state == nullptr);
  CHECK(w.resident_marlin.state == nullptr);
}

TEST_CASE("resident state is built once per weights block") {
  MoeBlockWeights w;
  ResidentOf(w.resident_bf16).built = 7;
  CHECK(ResidentOf(w.resident_bf16).built == 7);  // same object, not rebuilt
  CHECK(w.resident_bf16.state != nullptr);
  // The slots are independent: building one must not mark the others ready.
  CHECK(w.resident_fused.state == nullptr);
  CHECK(w.resident_marlin.state == nullptr);
}

TEST_CASE("a new block at a REUSED address does not inherit residency") {
  // The exact aliasing the old address-keyed map could not see. Placement-new
  // guarantees the second block occupies the first one's storage, which the
  // allocator only did intermittently in the reported repro.
  alignas(MoeBlockWeights) static unsigned char storage[sizeof(MoeBlockWeights)];

  auto* first = new (storage) MoeBlockWeights();
  ResidentOf(first->resident_bf16).built = 1;
  REQUIRE(first->resident_bf16.state != nullptr);
  const void* first_addr = static_cast<const void*>(first);
  first->~MoeBlockWeights();

  auto* second = new (storage) MoeBlockWeights();
  REQUIRE(static_cast<const void*>(second) == first_addr);  // same address
  CHECK(second->resident_bf16.state == nullptr);            // but NOT ready
  CHECK(ResidentOf(second->resident_bf16).built == 0);       // rebuilt fresh
  second->~MoeBlockWeights();
}

TEST_CASE("residency dies with the weights block") {
  std::weak_ptr<void> observer;
  {
    MoeBlockWeights w;
    ResidentOf(w.resident_marlin).built = 3;
    observer = w.resident_marlin.state;
    REQUIRE_FALSE(observer.expired());
  }
  // Destroying the weights must release the resident state, not leave it for a
  // later block to find.
  CHECK(observer.expired());
}

TEST_CASE("the dense Marlin slots are per-weight and independent") {
  Nvfp4Weight gate;
  Nvfp4Weight up;
  CHECK(gate.resident_marlin.state == nullptr);
  CHECK(gate.resident_marlin_pair.state == nullptr);

  ResidentOf(gate.resident_marlin_pair).built = 2;  // fused gate+up repack
  CHECK(gate.resident_marlin.state == nullptr);     // its own repack, separate
  CHECK(up.resident_marlin_pair.state == nullptr);  // the other half, separate
}

TEST_CASE("Laguna MoE weights own their Marlin residency too") {
  // Same defect, same fix, a different model: the reported repro only exercised
  // the Qwen3.5 bf16 path because VT_MOE_BF16_FAST was the A/B lever available.
  LagunaMoeWeights w;
  CHECK(w.resident_marlin.state == nullptr);
  ResidentOf(w.resident_marlin).built = 5;
  CHECK(ResidentOf(w.resident_marlin).built == 5);
}

// ─── dots3-note (#699, W5) ──────────────────────────────────────────────────
// The grouped bf16 MoE arm shipped its per-expert device-pointer arrays in a
// `static std::map<const Dots3NoteMoeWeights*, ...>` — the pre-#237 shape,
// eighteen days after #237 removed it from qwen3_5.cpp. The failure is the one
// this file was written for, and it is quieter here than it was there: the
// device buffers are deliberately never freed, so a second engine that inherits
// a `ready` entry does not crash. It answers from the FIRST model's experts.
//
// WHAT THESE CASES CAN AND CANNOT SEE. `Dots3NoteMoePtrsFor` is file-local to
// `dots3_note_device.cpp` and is only ever called from inside
// `Dots3NoteGroupedMoeEligible`, which requires a NATIVE
// `vt::OpId::kMoeGroupedGemmBf16` and is therefore CUDA-only — there is no CPU
// reference tier for that op. No CPU gate can call the accessor, so what is
// pinned below is the invariant the accessor now rests on: the residency is a
// member of the weights block, so it is per-OBJECT and cannot be inherited
// through a reused address. Reverting the accessor's body alone, while leaving
// the member in place, is NOT observable from the CPU; that reverted arm is
// covered by the same device run the row already owes for the grouped GEMM
// (spec `## Owed`).
TEST_CASE("a fresh dots3-note MoE block owns no resident state") {
  Dots3NoteMoeWeights w;
  CHECK(w.resident_moe.state == nullptr);
}

TEST_CASE("dots3-note MoE residency is built once per weights block") {
  Dots3NoteMoeWeights w;
  ResidentOf(w.resident_moe).built = 4;
  CHECK(ResidentOf(w.resident_moe).built == 4);  // same object, not rebuilt
  CHECK(w.resident_moe.state != nullptr);
}

TEST_CASE("a new dots3-note MoE block at a REUSED address does not inherit residency") {
  alignas(Dots3NoteMoeWeights) static unsigned char storage[sizeof(Dots3NoteMoeWeights)];

  auto* first = new (storage) Dots3NoteMoeWeights();
  ResidentOf(first->resident_moe).built = 1;
  REQUIRE(first->resident_moe.state != nullptr);
  const void* first_addr = static_cast<const void*>(first);
  first->~Dots3NoteMoeWeights();

  auto* second = new (storage) Dots3NoteMoeWeights();
  REQUIRE(static_cast<const void*>(second) == first_addr);  // same address
  CHECK(second->resident_moe.state == nullptr);             // but NOT ready
  CHECK(ResidentOf(second->resident_moe).built == 0);       // rebuilt fresh
  // Deliberately NOT an assertion that the two state objects have different
  // ADDRESSES: the first one is freed by the destructor above and the allocator
  // will usually hand the second the same block back. That check would be a
  // coin flip, not a gate.
  second->~Dots3NoteMoeWeights();
}

TEST_CASE("dots3-note MoE residency dies with the weights block") {
  std::weak_ptr<void> observer;
  {
    Dots3NoteMoeWeights w;
    ResidentOf(w.resident_moe).built = 6;
    observer = w.resident_moe.state;
    REQUIRE_FALSE(observer.expired());
  }
  CHECK(observer.expired());
}
