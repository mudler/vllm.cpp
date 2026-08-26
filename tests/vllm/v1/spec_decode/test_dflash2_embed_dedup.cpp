// SPEC-DFLASH2 (#1946) — the DEVICE bytes of the shared embedding table,
// counted.
//
// WHAT THE DEFECT WAS. A DFlash/DFlash2 draft owns no embedding table; it runs
// the TARGET's over its own hidden states. Our loader expressed that by READING
// the target's tensor a second time into the draft's own `OwnedTensor`.
// `ResidentWeight` caches the host->device upload ON THE OWNED TENSOR — the
// `if (!w.d_dev)` guard in
// `include/vllm/model_executor/models/dense_attn_block.h::ResidentWeight` — so two `OwnedTensor`s over
// identical bytes are two device allocations, however the HOST bytes are
// shared. On `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` that table is BF16
// [248320, 5120] = 2,542,796,800 B, held twice, out of a unified GB10 pool.
//
// WHY W9 (#1849) DID NOT ALREADY CLOSE IT, AND WHY THIS FILE COUNTS BYTES
// RATHER THAN POINTERS. W9 made both shared reads borrow-first, so the two
// tensors already pointed at the same safetensors mapping. A case asserting
// that the HOST bytes are shared would therefore have passed before this change
// and proved nothing. The quantity that moved is the DEVICE allocation, and it
// is the only quantity this file asserts.
//
// THE PLATFORM THIS NEEDS DOES NOT EXIST ON A CPU TIER, and the machinery to
// stand in for it is not invented here: `ResidentWeight` aliases the host bytes
// outright when `is_cpu()`, so a CPU run allocates nothing and counts nothing.
// `tests/vllm/model_executor/test_resident_weight_host_addressable.cpp`
// established the answer for #1299 — a fake platform in the otherwise-unused
// kXPU slot over a fake `vt::Backend` whose Alloc is a counted malloc — and this
// file reuses that shape. What is measured is a real allocation at a real
// address; only the allocator is a stand-in.
//
// WHAT THIS FILE DOES NOT CLAIM. It does not claim that anything in production
// reaches the rebind. `test_dflash2_embed_dedup_reach.cpp` proves that through
// the LoadedEngine constructor, and the reachability mutation
// `.agents/reachability.md` requires is stated against THAT binary. This file
// owns the byte count and the refusal boundary.
//
// The two inputs are built by hand here on purpose: the function under test is
// `BindDflashDraftSharedEmbed` itself, which is the exact function production
// calls (exported for that reason, as W9 exported
// `LoadDflashSharedEmbedBf16` for the same reason).
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/owned_bytes.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/model_executor/models/qwen3_dspark.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace {

using vllm::OwnedTensor;
using vllm::Qwen3_5DenseWeights;
using vllm::entrypoints::DflashDraft;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// The 27B's real geometry is 248320 x 5120; this is the same SHAPE at a size a
// CPU test can allocate twice without noticing. Nothing here depends on the
// magnitude — the assertion is a ratio between one copy and two.
constexpr int64_t kVocab = 64;
constexpr int64_t kHidden = 16;
constexpr size_t kTableBytes = static_cast<size_t>(kVocab * kHidden) * 2;

// A backend over ordinary host memory, standing in for a device allocator, with
// counted allocations. Same shape and same reason as the #1299 fake.
class CountingBackend final : public vt::Backend {
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
  // TRUE, matching GB10's CUDA backend: host and device address the same
  // physical RAM, which is exactly why a second 2.5 GB device copy of the table
  // is 2.5 GB of the box's own memory and not a separate pool's.
  bool UnifiedMemory() const override { return true; }
  // FALSE, also matching GB10 (a `cudaMalloc` pointer there is not
  // host-dereferenceable), so `AdoptDeviceBytesAsHost` stays inert and a staged
  // weight stays visibly staged rather than being folded back onto its host
  // buffer by a second mechanism, which would confuse what this file measures.
  bool DeviceMemoryIsHostAddressable() const override { return false; }

  int allocs = 0;
  size_t alloc_bytes = 0;
};

CountingBackend& Fake() {
  static CountingBackend b;
  return b;
}

// A non-CPU platform, which is the whole requirement: `ResidentWeight` takes the
// alias branch on `is_cpu()` and the staging branch on everything else.
// `host_memory_is_device_addressable()` stays FALSE so the staging branch is the
// one taken.
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
  bool host_memory_is_device_addressable() const override { return false; }
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

// One [vocab, H] gather table. `tag` seeds the bytes so two tables built with
// different tags are distinguishable, which is what lets a case say WHICH table
// a gather would have read.
OwnedTensor MakeTable(uint8_t tag, DType dt = DType::kBF16,
                      int64_t vocab = kVocab) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = 2;
  t.shape[0] = vocab;
  t.shape[1] = kHidden;
  t.nk = false;
  const size_t width = dt == DType::kBF16 ? 2u : 4u;
  std::vector<uint8_t> b(static_cast<size_t>(vocab * kHidden) * width);
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<uint8_t>((i * 7 + tag) & 0xFF);
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

// A draft carrying only what the rebind reads. Everything else on
// `Qwen3DFlashWeights` is irrelevant to this function and stays default.
std::unique_ptr<DflashDraft> MakeDraft(DType dt = DType::kBF16,
                                       int64_t vocab = kVocab) {
  auto d = std::make_unique<DflashDraft>();
  // Tag 2, not tag 1: the draft's own copy holds DIFFERENT bytes from the
  // target's here, so a case can tell "the draft gathered from the target's
  // table" from "the two happened to agree". In production they are the same
  // tensor of the same file.
  d->weights.embed_tokens = MakeTable(/*tag=*/2, dt, vocab);
  return d;
}

// The ADDRESS of a tensor, as a printable value. doctest stringifies a
// comparison of two `const OwnedTensor*` as a bool, so a failing identity check
// reports `CHECK( 1 == 1 )` and says nothing about which tensor was bound. Going
// through `const void*` makes the two addresses appear in the failure.
const void* Addr(const OwnedTensor& w) { return static_cast<const void*>(&w); }

// Upload a table exactly as the two forwards do, and report what the allocator
// saw. Every draft site (`src/vllm/model_executor/models/qwen3_dflash.cpp::EmbedTable`)
// and every target site (`src/vllm/model_executor/models/qwen3_5.cpp::EmbedInto`)
// reduces to this one call.
void Upload(const OwnedTensor& w, int64_t vocab = kVocab) {
  Queue q = XpuQueue();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  const vt::Tensor t = vllm::dense_attn::ResidentWeight(d, w, {vocab, kHidden});
  REQUIRE(t.data != nullptr);
}

}  // namespace

TEST_CASE("#1946: the draft and the target upload ONE device copy of the table") {
  Qwen3_5DenseWeights target;
  target.embed_tokens = MakeTable(/*tag=*/1);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(target);

  // The target LENDS its table. A null here would mean the override is missing
  // and the case below would pass for the wrong reason.
  REQUIRE(model->shared_embed_tokens() == &target.embed_tokens);

  std::unique_ptr<DflashDraft> draft = MakeDraft();
  // BEFORE: two tensors, two tables, and `EmbedTable()` is the draft's own.
  REQUIRE_FALSE(draft->weights.embed_tokens.Empty());
  REQUIRE(Addr(draft->weights.EmbedTable()) != Addr(target.embed_tokens));

  CHECK(vllm::entrypoints::BindDflashDraftSharedEmbed(*draft, *model));

  // AFTER: `del draft_inner.embed_tokens; draft_inner.embed_tokens = target_embed`
  // (dflash/utils.py:73-74 @ b389ac29465b33f9e9c534df221ea3c129e9793f).
  CHECK(draft->weights.embed_tokens.Empty());
  CHECK(Addr(draft->weights.EmbedTable()) == Addr(target.embed_tokens));

  // THE ASSERTION THIS FILE EXISTS FOR, and it is a device allocation rather
  // than an output: gather the way BOTH forwards gather, then count.
  const int allocs_before = Fake().allocs;
  const size_t bytes_before = Fake().alloc_bytes;
  Upload(target.embed_tokens);
  Upload(draft->weights.EmbedTable());
  const int allocs = Fake().allocs - allocs_before;
  const size_t bytes = Fake().alloc_bytes - bytes_before;

  // BOUNDED FROM BELOW, because a zero here would be a mute switch rather than
  // a dedup: something really was uploaded, and it was the whole table.
  INFO("allocs=", allocs, " bytes=", bytes, " table=", kTableBytes);
  CHECK(allocs >= 1);
  CHECK(bytes >= kTableBytes);
  // ...AND FROM ABOVE, which is the fix. Before #1946 this read
  // 2 allocations and 2 * kTableBytes.
  CHECK(allocs == 1);
  CHECK(bytes == kTableBytes);

  // And the surviving allocation is the TARGET's, reachable from the target's
  // own tensor — not a third buffer that both happen to agree about.
  CHECK(target.embed_tokens.d_dev != nullptr);
}

TEST_CASE("#1946: an UNBOUND draft still uploads its own table, and pays twice") {
  // The control. Without the rebind the two tables are two allocations, which
  // is what the whole issue measured — so the case above is reading a
  // difference the harness can actually produce, not a constant.
  Qwen3_5DenseWeights target;
  target.embed_tokens = MakeTable(/*tag=*/1);
  std::unique_ptr<DflashDraft> draft = MakeDraft();

  const int allocs_before = Fake().allocs;
  const size_t bytes_before = Fake().alloc_bytes;
  Upload(target.embed_tokens);
  Upload(draft->weights.EmbedTable());
  CHECK(Fake().allocs - allocs_before == 2);
  CHECK(Fake().alloc_bytes - bytes_before == 2 * kTableBytes);
}

TEST_CASE("#1946: a DTYPE disagreement refuses the rebind and keeps both tables") {
  // NOT hypothetical. A GGUF target may keep `token_embd` F16 in place
  // (`LoadEmbedAndHead`'s kKeepF16 arm, qwen3_5_gguf_weights.cpp) while
  // `LoadGgufSharedEmbedAndHeadBf16` always hands the draft BF16. Aliasing an
  // f16 table through a bf16 view would make the draft gather garbage rows —
  // silently wrong tokens that no memory gate could ever see.
  Qwen3_5DenseWeights target;
  target.embed_tokens = MakeTable(/*tag=*/1, DType::kF32);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(target);
  std::unique_ptr<DflashDraft> draft = MakeDraft(DType::kBF16);

  CHECK_FALSE(vllm::entrypoints::BindDflashDraftSharedEmbed(*draft, *model));
  CHECK(draft->weights.shared_embed_tokens == nullptr);
  CHECK_FALSE(draft->weights.embed_tokens.Empty());
  CHECK(Addr(draft->weights.EmbedTable()) == Addr(draft->weights.embed_tokens));
}

TEST_CASE("#1946: a SHAPE disagreement refuses the rebind") {
  // Same argument one axis over: a target whose vocabulary is not the draft's
  // would have the draft gathering rows past the end of the table.
  Qwen3_5DenseWeights target;
  target.embed_tokens = MakeTable(/*tag=*/1, DType::kBF16, /*vocab=*/kVocab / 2);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(target);
  std::unique_ptr<DflashDraft> draft = MakeDraft(DType::kBF16, kVocab);

  CHECK_FALSE(vllm::entrypoints::BindDflashDraftSharedEmbed(*draft, *model));
  CHECK(Addr(draft->weights.EmbedTable()) == Addr(draft->weights.embed_tokens));
}

TEST_CASE("#1946: a target with NO table lends nothing") {
  // Every model that is not a Qwen3.5/3.6 dense or MoE forward answers null,
  // and so does one whose loader has not filled the table yet. The draft is
  // left exactly as it was rather than rebound onto a null.
  Qwen3_5DenseWeights target;  // embed_tokens default-constructed, Empty()
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(target);
  CHECK(model->shared_embed_tokens() == nullptr);

  std::unique_ptr<DflashDraft> draft = MakeDraft();
  CHECK_FALSE(vllm::entrypoints::BindDflashDraftSharedEmbed(*draft, *model));
  CHECK(Addr(draft->weights.EmbedTable()) == Addr(draft->weights.embed_tokens));
}

TEST_CASE("#1946: a DSPARK draft is skipped, because its table is not this one") {
  // A DSpark checkpoint usually SHIPS its own table and keeps it by value inside
  // Qwen3DSparkWeights; the loader's shared fallback fills THAT, not
  // `weights.embed_tokens`. Rebinding here would touch a field the DSpark
  // forward never reads and leave the copy that costs the memory in place. It is
  // owed as its own row, and skipped by name rather than by luck.
  Qwen3_5DenseWeights target;
  target.embed_tokens = MakeTable(/*tag=*/1);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(target);

  std::unique_ptr<DflashDraft> draft = MakeDraft();
  draft->dspark = std::make_unique<vllm::Qwen3DSparkWeights>();
  CHECK_FALSE(vllm::entrypoints::BindDflashDraftSharedEmbed(*draft, *model));
  CHECK(draft->weights.shared_embed_tokens == nullptr);
  CHECK_FALSE(draft->weights.embed_tokens.Empty());
}

TEST_CASE("#1946: a post-rebind read of the draft's OWN tensor refuses BY NAME") {
  // THE CLAIM THIS CASE EXISTS TO MAKE TRUE, and it was false when it was only a
  // comment. `## Design` said a future call site that read `weights.embed_tokens`
  // instead of `EmbedTable()` after a rebind would get an empty table "which
  // `vt::Embedding` refuses by name rather than silently re-uploading 2.5 GB".
  // It would not have. `vt::Embedding` (src/vt/ops.cpp `void Embedding(`) checks
  // ranks, shapes, dtypes, contiguity and device and NEVER the data pointer or
  // the byte length, and `ResidentWeight` takes the shape from the CALLER — so an
  // emptied tensor passes every one of those checks. The real outcomes were a
  // null host alias on the CPU arm (SIGSEGV) and, on a device arm, `Alloc(0)`
  // followed by a [vocab, H] view over a zero-byte allocation: out-of-bounds
  // device reads, which IS the silently-wrong-tokens failure the clear is
  // supposed to prevent.
  //
  // `ResidentWeight` now refuses an empty weight on both arms, and this case
  // drives the exact hypothetical: rebind, then read the raw field.
  Qwen3_5DenseWeights target;
  target.embed_tokens = MakeTable(/*tag=*/1);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(target);
  std::unique_ptr<DflashDraft> draft = MakeDraft();
  REQUIRE(vllm::entrypoints::BindDflashDraftSharedEmbed(*draft, *model));
  REQUIRE(draft->weights.embed_tokens.Empty());

  // The accessor is unaffected — it is the supported read and it still works.
  Upload(draft->weights.EmbedTable());

  // The unsupported read now THROWS, and the message names the seam and the
  // condition rather than dying in a kernel. CHECK_THROWS_WITH_AS, not
  // CHECK_THROWS: a bare throw check would stay green if some unrelated
  // precondition started firing instead.
  CHECK_THROWS_WITH_AS(Upload(draft->weights.embed_tokens),
                       doctest::Contains("resident weight: EMPTY tensor"),
                       std::runtime_error);
}
