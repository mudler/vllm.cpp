// The pure half of RocmBackend::Copy's bounded pinned bounce ring:
// .agents/specs/rocm-chunked-pinned-h2d.md §4.
//
// UNCONDITIONAL, no HIP required, for the reason test_rocm_arch.cpp is
// unconditional: include/vt/rocm/rocm_pinned_h2d.h is deliberately free of HIP
// headers, and it holds the only parts of this change that a wrong answer breaks
// SILENTLY. A ring that never engages produces identical bytes and loads the
// same model; only the host residency differs, and no gate in this tree reads
// that by accident. So the chunk arithmetic, the cross-call reuse order and the
// five-term predicate are gated here, on a runner with no AMD GPU.
//
// The OTHER half — that rocm_backend.hip actually calls this — cannot be gated
// here and is not pretended to be. It is gated by the device case
// "large pageable H2D takes the pinned bounce ring" in
// tests/vt/test_backend_cross_device.cpp, whose mutation is deleting the staged
// branch from the backend.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/rocm/rocm_pinned_h2d.h"

namespace {

using vt::rocm::kPinnedH2DBuffers;
using vt::rocm::kPinnedH2DChunkBytesDefault;
using vt::rocm::ParsePinnedH2DChunkBytes;
using vt::rocm::PtrKind;
using vt::rocm::RunStagedH2D;
using vt::rocm::ShouldStageH2D;
using vt::rocm::StagedH2DInputs;
using vt::rocm::StagedH2DRing;
using vt::rocm::StagingTermsExceptRing;

// A fake device: N pinned slots, an ordered event log, and a reassembled
// destination. Everything the real .hip does, minus HIP.
struct FakeRing {
  explicit FakeRing(size_t n_buffers, size_t chunk_bytes)
      : slots(n_buffers, std::vector<uint8_t>(chunk_bytes, 0)) {
    ring.n_buffers = n_buffers;
  }

  StagedH2DRing ring;
  std::vector<std::vector<uint8_t>> slots;
  std::vector<std::string> log;
  // What the "device" ended up holding, and how big each upload was.
  std::vector<uint8_t> dst;
  std::vector<size_t> chunk_sizes;
  std::vector<size_t> chunk_offsets;

  size_t Run(const std::vector<uint8_t>& src, size_t chunk_bytes) {
    return RunStagedH2D(
        ring, src.size(), chunk_bytes,
        [&](size_t slot) { log.push_back("wait " + std::to_string(slot)); },
        [&](size_t slot, size_t off, size_t n) {
          log.push_back("stage " + std::to_string(slot));
          std::memcpy(slots[slot].data(), src.data() + off, n);
        },
        [&](size_t slot, size_t off, size_t n) {
          log.push_back("enqueue " + std::to_string(slot));
          if (dst.size() < off + n) dst.resize(off + n, 0);
          std::memcpy(dst.data() + off, slots[slot].data(), n);
          chunk_sizes.push_back(n);
          chunk_offsets.push_back(off);
        });
  }
};

std::vector<uint8_t> Pattern(size_t n, uint32_t seed) {
  std::vector<uint8_t> v(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<uint8_t>(s >> 24);
  }
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The plan: the chunk count, the sizes, and the bound.
// ---------------------------------------------------------------------------
TEST_CASE("the pinned ring splits a large copy into bounded chunks") {
  // 200 MiB against a 64 MiB chunk: 64 + 64 + 64 + 8.
  constexpr size_t kChunk = static_cast<size_t>(64) << 20;
  constexpr size_t kBytes = static_cast<size_t>(200) << 20;

  StagedH2DRing ring;
  std::vector<size_t> sizes;
  std::vector<size_t> offsets;
  const size_t chunks = RunStagedH2D(
      ring, kBytes, kChunk, [](size_t) {}, [](size_t, size_t, size_t) {},
      [&](size_t, size_t off, size_t n) {
        offsets.push_back(off);
        sizes.push_back(n);
      });

  CHECK(chunks == 4);
  REQUIRE(sizes.size() == 4);
  CHECK(sizes[0] == kChunk);
  CHECK(sizes[1] == kChunk);
  CHECK(sizes[2] == kChunk);
  CHECK(sizes[3] == (static_cast<size_t>(8) << 20));

  // THE BOUND, which is the whole point: no single upload reads more than one
  // chunk out of the pageable source, whatever the copy's total size. Mutating
  // the chunk size to the whole buffer makes this one 200 MiB upload.
  size_t total = 0;
  size_t expect_off = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    CHECK(sizes[i] <= kChunk);
    CHECK(offsets[i] == expect_off);
    expect_off += sizes[i];
    total += sizes[i];
  }
  CHECK(total == kBytes);
}

TEST_CASE("the pinned ring leaves a copy smaller than one chunk as one chunk") {
  StagedH2DRing ring;
  size_t chunks = 0;
  size_t only_size = 0;
  chunks = RunStagedH2D(
      ring, 1000, 4096, [](size_t) {}, [](size_t, size_t, size_t) {},
      [&](size_t, size_t, size_t n) { only_size = n; });
  CHECK(chunks == 1);
  CHECK(only_size == 1000);
}

TEST_CASE("the pinned ring enqueues nothing when the chunk size is zero") {
  // VT_ROCM_PINNED_H2D_MIB=0 restores the pre-change single call in the SAME
  // binary, which is what makes the load-time A/B one build instead of two.
  StagedH2DRing ring;
  size_t enqueued = 0;
  const size_t chunks = RunStagedH2D(
      ring, 1u << 30, 0, [](size_t) {}, [](size_t, size_t, size_t) {},
      [&](size_t, size_t, size_t) { ++enqueued; });
  CHECK(chunks == 0);
  CHECK(enqueued == 0);
}

// ---------------------------------------------------------------------------
// 2. The reuse order — WITHIN a call and, the part that matters, ACROSS calls.
// ---------------------------------------------------------------------------
TEST_CASE("the pinned ring waits on a slot before it refills it, not before first use") {
  FakeRing fake(kPinnedH2DBuffers, 16);
  const std::vector<uint8_t> src = Pattern(16 * 9, 7);  // 9 chunks over 4 slots
  const size_t chunks = fake.Run(src, 16);
  CHECK(chunks == 9);

  // The first four chunks touch four distinct slots and wait on none of them.
  for (size_t i = 0; i < 4; ++i) {
    CHECK(fake.log[i * 2] == "stage " + std::to_string(i));
    CHECK(fake.log[i * 2 + 1] == "enqueue " + std::to_string(i));
  }
  // Chunk 4 is the first reuse of slot 0, and it MUST wait first. Deleting the
  // wait removes exactly this line and overwrites bytes a DMA is still reading.
  CHECK(fake.log[8] == "wait 0");
  CHECK(fake.log[9] == "stage 0");
  // Chunk 8 is the second reuse of slot 0.
  CHECK(fake.log[8 + 4 * 3] == "wait 0");

  size_t waits = 0;
  for (const auto& line : fake.log) {
    if (line.rfind("wait ", 0) == 0) ++waits;
  }
  CHECK(waits == 5);  // chunks 4..8 each reuse a slot; chunks 0..3 do not
}

TEST_CASE("the pinned ring waits across calls, not only within one") {
  // THIS is the case a ring that tracked reuse only within one Copy would fail.
  // Two 2-chunk copies over a 4-slot ring: the second call starts at slot 2, so
  // it waits on nothing; a THIRD 2-chunk copy comes back round to slot 0, which
  // the FIRST call left in flight, and must wait.
  FakeRing fake(kPinnedH2DBuffers, 16);
  const std::vector<uint8_t> src = Pattern(32, 11);
  CHECK(fake.Run(src, 16) == 2);
  CHECK(fake.Run(src, 16) == 2);
  size_t waits_after_two = 0;
  for (const auto& line : fake.log) {
    if (line.rfind("wait ", 0) == 0) ++waits_after_two;
  }
  CHECK(waits_after_two == 0);

  fake.log.clear();
  CHECK(fake.Run(src, 16) == 2);
  REQUIRE(fake.log.size() >= 2);
  CHECK(fake.log[0] == "wait 0");
  CHECK(fake.log[3] == "wait 1");
}

// ---------------------------------------------------------------------------
// 3. The bytes survive the bounce, at a size that is not a chunk multiple.
// ---------------------------------------------------------------------------
TEST_CASE("the pinned ring reassembles the source byte for byte") {
  constexpr size_t kChunk = 4096;
  constexpr size_t kBytes = kChunk * 9 + 137;  // deliberately ragged
  FakeRing fake(kPinnedH2DBuffers, kChunk);
  const std::vector<uint8_t> src = Pattern(kBytes, 4242);
  const size_t chunks = fake.Run(src, kChunk);
  CHECK(chunks == 10);
  REQUIRE(fake.dst.size() == kBytes);
  CHECK(std::memcmp(fake.dst.data(), src.data(), kBytes) == 0);
  CHECK(fake.chunk_sizes.back() == 137);
  for (size_t n : fake.chunk_sizes) CHECK(n <= kChunk);
}

// ---------------------------------------------------------------------------
// 4. The predicate: the five terms, one row per term.
// ---------------------------------------------------------------------------
TEST_CASE("the staged path needs all five terms, and refuses each missing one") {
  StagedH2DInputs ok;
  ok.src = PtrKind::kUnregisteredHost;
  ok.dst = PtrKind::kDevice;
  ok.bytes = kPinnedH2DChunkBytesDefault;
  ok.chunk_bytes = kPinnedH2DChunkBytesDefault;
  ok.stream_capturing = false;
  ok.ring_available = true;
  CHECK(ShouldStageH2D(ok));

  SUBCASE("a copy smaller than one chunk stays on the single call") {
    StagedH2DInputs in = ok;
    in.bytes = kPinnedH2DChunkBytesDefault - 1;
    CHECK_FALSE(ShouldStageH2D(in));
    // A 4 KiB norm weight is the case this term exists for.
    in.bytes = 4096;
    CHECK_FALSE(ShouldStageH2D(in));
  }
  SUBCASE("a device source is never bounced (D2D)") {
    StagedH2DInputs in = ok;
    in.src = PtrKind::kDevice;
    CHECK_FALSE(ShouldStageH2D(in));
  }
  SUBCASE("an already-pinned source is never bounced") {
    StagedH2DInputs in = ok;
    in.src = PtrKind::kPinnedHost;
    CHECK_FALSE(ShouldStageH2D(in));
  }
  SUBCASE("a managed or unified source is never bounced") {
    // VT_ROCM_MANAGED_ALLOC=1 is exactly this configuration, and it must be
    // byte-identical to before: managed memory is already device addressable.
    StagedH2DInputs in = ok;
    in.src = PtrKind::kOther;
    CHECK_FALSE(ShouldStageH2D(in));
  }
  SUBCASE("a host destination is never bounced (D2H and H2H)") {
    StagedH2DInputs in = ok;
    in.dst = PtrKind::kPinnedHost;
    CHECK_FALSE(ShouldStageH2D(in));
    in.dst = PtrKind::kUnregisteredHost;
    CHECK_FALSE(ShouldStageH2D(in));
    in.dst = PtrKind::kOther;
    CHECK_FALSE(ShouldStageH2D(in));
  }
  SUBCASE("a capturing stream is never bounced") {
    StagedH2DInputs in = ok;
    in.stream_capturing = true;
    CHECK_FALSE(ShouldStageH2D(in));
  }
  SUBCASE("a ring that could not be allocated falls back instead of failing") {
    StagedH2DInputs in = ok;
    in.ring_available = false;
    CHECK_FALSE(ShouldStageH2D(in));
  }
  SUBCASE("chunk size zero disables the path") {
    StagedH2DInputs in = ok;
    in.chunk_bytes = 0;
    CHECK_FALSE(ShouldStageH2D(in));
  }
}

// ---------------------------------------------------------------------------
// 5. The knob parse. A typo must not silently disable a residency bound.
// ---------------------------------------------------------------------------
TEST_CASE("VT_ROCM_PINNED_H2D_MIB parses to bytes, and a typo takes the default") {
  CHECK(ParsePinnedH2DChunkBytes("") == kPinnedH2DChunkBytesDefault);
  CHECK(ParsePinnedH2DChunkBytes("64") == kPinnedH2DChunkBytesDefault);
  CHECK(ParsePinnedH2DChunkBytes("0") == 0);
  CHECK(ParsePinnedH2DChunkBytes("8") == (static_cast<size_t>(8) << 20));
  CHECK(ParsePinnedH2DChunkBytes("128") == (static_cast<size_t>(128) << 20));
  // Not a number: the DEFAULT, never 0.
  CHECK(ParsePinnedH2DChunkBytes("off") == kPinnedH2DChunkBytesDefault);
  CHECK(ParsePinnedH2DChunkBytes("-1") == kPinnedH2DChunkBytesDefault);
  CHECK(ParsePinnedH2DChunkBytes("64MiB") == kPinnedH2DChunkBytesDefault);
  // Absurdly large: the default, so nobody pins a terabyte by fat-fingering.
  CHECK(ParsePinnedH2DChunkBytes("99999999") == kPinnedH2DChunkBytesDefault);
}

// ---------------------------------------------------------------------------
// 6. The numbers are the ORACLE'S numbers, not invented ones.
// ---------------------------------------------------------------------------
TEST_CASE("the ring size and chunk size are llama.cpp's, and bound the residency") {
  // llama-cpp 10bf611e5 (b10451), src/llama-model-loader.cpp:1440 and :1449.
  CHECK(kPinnedH2DBuffers == 4);
  CHECK(kPinnedH2DChunkBytesDefault == (static_cast<size_t>(64) << 20));
  // The number the whole change exists to bound, for a model of ANY size.
  CHECK(vt::rocm::kPinnedH2DRingBytes == (static_cast<size_t>(256) << 20));
}

// ---------------------------------------------------------------------------
// 7. The split between the cheap terms and the ALLOCATING one.
//
// Production cannot evaluate `ring_available` without allocating 256 MiB of
// pinned host memory, so it asks StagingTermsExceptRing first and calls
// EnsureRing only when that passes. That is two expressions where the spec
// describes one decision, and two expressions drift.
//
// This case walks 480 inputs against an expectation computed HERE, from the
// four inputs, as a sequence of refusals -- never by calling either expression
// under test. An earlier version of this case asserted
// `StagingTermsExceptRing(in) == ShouldStageH2D(in)` with `ring_available`
// held true, which is `X == (X && true)`: true for ANY definition of the
// helper, including a deleted term. It convicted nothing. Measured: deleting
// `dst == kDevice`, and separately `bytes >= chunk_bytes`, from the helper left
// that version 1443/1443 SUCCESS. The independent expectation below is what
// makes a deleted term fail HERE and not only in the truth table above.
// ---------------------------------------------------------------------------
namespace {

// The four cheap terms, written out independently of the header. A refusal
// sequence rather than a conjunction, so this is not a copy of the expression
// it checks.
bool ExpectedCheapTerms(PtrKind src, PtrKind dst, size_t bytes,
                        size_t chunk_bytes, bool capturing) {
  if (chunk_bytes == 0) return false;
  if (capturing) return false;
  if (dst != PtrKind::kDevice) return false;
  if (src != PtrKind::kUnregisteredHost) return false;
  if (bytes < chunk_bytes) return false;
  return true;
}

}  // namespace

TEST_CASE("the cheap terms are exactly the decision minus the allocating term") {
  const PtrKind kinds[] = {PtrKind::kUnregisteredHost, PtrKind::kPinnedHost,
                           PtrKind::kDevice, PtrKind::kOther};
  const size_t chunks[] = {0, 1024, kPinnedH2DChunkBytesDefault};
  const size_t sizes[] = {0, 1023, 1024, kPinnedH2DChunkBytesDefault,
                          kPinnedH2DChunkBytesDefault * 3};
  size_t staged = 0;
  size_t total = 0;
  for (PtrKind s : kinds) {
    for (PtrKind d : kinds) {
      for (size_t c : chunks) {
        for (size_t b : sizes) {
          for (bool cap : {false, true}) {
            StagedH2DInputs in;
            in.src = s;
            in.dst = d;
            in.chunk_bytes = c;
            in.bytes = b;
            in.stream_capturing = cap;

            const bool want = ExpectedCheapTerms(s, d, b, c, cap);

            // The cheap terms are the four, independently of the ring.
            in.ring_available = true;
            REQUIRE(StagingTermsExceptRing(in) == want);
            // And the decision is those four AND the ring, so with a ring it
            // is the same answer -- proved against `want`, not against the
            // helper, so a term deleted from EITHER expression fails here.
            REQUIRE(ShouldStageH2D(in) == want);
            if (want) ++staged;

            // With no ring, the decision is always no, while the cheap terms
            // are unmoved -- which is the whole point: production learns the
            // answer is no WITHOUT paying for the ring to find out.
            in.ring_available = false;
            CHECK_FALSE(ShouldStageH2D(in));
            CHECK(StagingTermsExceptRing(in) == want);
            ++total;
          }
        }
      }
    }
  }
  // The table is not degenerate: some rows stage and most do not.
  CHECK(total == 480);
  CHECK(staged > 0);
  CHECK(staged < total);
}
