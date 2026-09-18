// The DECISION and the LOOP behind RocmBackend::Copy's bounded pinned bounce
// ring, deliberately free of HIP headers.
//
// Same split, and the same reason, as include/vt/rocm/rocm_arch.h and
// `ResolveMemoryPolicy`: the piece a wrong answer breaks SILENTLY is the piece
// that has to be gated on an ordinary CPU runner with no AMD GPU. A ring that
// quietly never engages looks exactly like one that works — the bytes are still
// correct, the model still loads, and only the host residency differs — so the
// chunk arithmetic, the reuse order and the predicate are compiled and
// table-tested in tests/vt/test_rocm_pinned_h2d.cpp on a machine with no HIP at
// all. src/vt/rocm/rocm_backend.hip reads two pointer attributes and one
// capture status, and calls into here.
//
// Oracle: llama-cpp pin 10bf611e5 (b10451), src/llama-model-loader.cpp:1440
// (`n_buffers = 4`), :1449 (64 MiB per buffer), :1591-1642 (the loop: wait on
// the slot's event, fill the pinned buffer, async-upload, record the event,
// advance modulo n_buffers). See .agents/specs/rocm-chunked-pinned-h2d.md §2
// for the one place the oracle's ring and ours are NOT the same thing.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace vt::rocm {

// Oracle: llama-model-loader.cpp:1440. Four slots keep one buffer filling while
// up to three drain; taken rather than re-derived.
inline constexpr size_t kPinnedH2DBuffers = 4;

// Oracle: llama-model-loader.cpp:1449, "64MB works well for NVMe drives". The
// oracle's `+ 2 * alignment` term is deliberately dropped: it absorbs the
// read-alignment padding of a file read (:1604-1631 computes it and then trims
// it back off), and our source is an in-memory mapping whose chunk boundaries
// are exact.
inline constexpr size_t kPinnedH2DChunkBytesDefault = static_cast<size_t>(64) << 20;

// Total pinned host residency, for a model of ANY size. This is the number the
// whole change exists to bound, so it is written down rather than left to be
// multiplied out by a reader.
inline constexpr size_t kPinnedH2DRingBytes =
    kPinnedH2DBuffers * kPinnedH2DChunkBytesDefault;

// What hipPointerGetAttributes said about a pointer, reduced to the four
// answers this decision needs. `kUnregisteredHost` is ordinary host storage the
// HIP runtime knows nothing about — a std::vector, a heap block, or the
// PROT_READ MAP_PRIVATE view of a GGUF shard that this change exists for.
// `kOther` covers managed and unified pointers, which are ALREADY device
// addressable and must never be bounced.
enum class PtrKind { kUnregisteredHost, kPinnedHost, kDevice, kOther };

// Everything RocmBackend::Copy knows at the moment it has to choose a path.
struct StagedH2DInputs {
  PtrKind src = PtrKind::kOther;
  PtrKind dst = PtrKind::kOther;
  size_t bytes = 0;
  size_t chunk_bytes = 0;   // 0 disables the ring entirely
  bool stream_capturing = false;
  bool ring_available = false;
};

// THE FOUR TERMS THAT COST NOTHING TO ANSWER, split out so the caller can ask
// them BEFORE it allocates anything.
//
// `ring_available` is the one term a caller cannot answer by looking: on the
// first qualifying copy, answering it MEANS allocating 256 MiB of pinned host
// memory and four events. Evaluating it eagerly -- as this file's first
// implementation did -- charges that allocation to every ROCm process that ever
// hands `Copy` one buffer of 64 MiB or more, including the processes that then
// take the direct path because the source is already pinned, the destination is
// managed, or the copy is D2D. Measured: `VT_ROCM_MANAGED_ALLOC=1` on gfx1151
// printed `staged=0 direct=3 chunks=0 ring_bytes=268435456`, a ring built and
// never touched. `.agents/environment.md:95-100` measures that board's managed
// ceiling as bounded by HOST RAM, so those bytes come out of the binding
// resource on the arm that never stages.
//
// So the production path asks THIS first and calls the allocator only when it
// passes. `ShouldStageH2D` stays the single authority on the decision and the
// thing the truth table gates; this is that expression with `ring_available`
// held true. A case in tests/vt/test_rocm_pinned_h2d.cpp walks 480 inputs and
// checks BOTH expressions against an expectation spelled out in that file from
// the four inputs, so a term deleted from either one fails there. Checking the
// two against EACH OTHER would not: that is `X == (X && true)`, true for any
// definition of this helper, and it is what an earlier version of that case
// did.
constexpr bool StagingTermsExceptRing(const StagedH2DInputs& in) {
  return in.chunk_bytes != 0 && !in.stream_capturing &&
         in.dst == PtrKind::kDevice && in.src == PtrKind::kUnregisteredHost &&
         in.bytes >= in.chunk_bytes;
}

// FIVE terms, all required. Spelled as one expression so the truth table in
// tests/vt/test_rocm_pinned_h2d.cpp can flip exactly one at a time.
//
//  * `chunk_bytes != 0`   — VT_ROCM_PINNED_H2D_MIB=0 restores the pre-change
//                           single call in the same binary, which is what makes
//                           the load-time A/B one build instead of two.
//  * `ring_available`     — a failed hipHostMalloc falls back rather than
//                           refusing a model over 256 MiB of pinned memory.
//  * `!stream_capturing`  — hipEventSynchronize aborts a graph capture. The
//                           capture contract already forbids blocking copies in
//                           the region, so this guard changes no legal program;
//                           it keeps an ILLEGAL one failing the way it used to.
//  * `dst == kDevice`     — D2H and H2H never bounce, so every readback and the
//                           sampler's download are untouched.
//  * `src == kUnregisteredHost` — D2D never bounces; an already-pinned source is
//                           already DMA-able; a managed source is already device
//                           addressable, so VT_ROCM_MANAGED_ALLOC=1 is unchanged.
//  * `bytes >= chunk_bytes` — below one chunk the ring is strictly one extra
//                           copy of the bytes with no overlap and no second slot
//                           ever used. A 4 KiB norm weight must not pay a bounce.
//
// The first four are `StagingTermsExceptRing` above, which is what production
// evaluates before it lets `EnsureRing` allocate anything.
constexpr bool ShouldStageH2D(const StagedH2DInputs& in) {
  return StagingTermsExceptRing(in) && in.ring_available;
}

// VT_ROCM_PINNED_H2D_MIB, in MiB. Absent or empty takes the default; an
// explicit `0` disables the ring; anything that does not parse as a
// non-negative integer takes the DEFAULT rather than 0, because a typo must not
// silently turn the feature off — a disabled ring and a working one differ only
// in host residency, which no gate in this tree reads by accident.
inline size_t ParsePinnedH2DChunkBytes(std::string_view v) {
  if (v.empty()) return kPinnedH2DChunkBytesDefault;
  size_t mib = 0;
  for (char c : v) {
    if (c < '0' || c > '9') return kPinnedH2DChunkBytesDefault;
    mib = mib * 10 + static_cast<size_t>(c - '0');
    if (mib > (static_cast<size_t>(1) << 20)) return kPinnedH2DChunkBytesDefault;
  }
  return mib << 20;
}

inline size_t PinnedH2DChunkBytes() {
  static const size_t bytes = [] {
    const char* e = std::getenv("VT_ROCM_PINNED_H2D_MIB");
    return ParsePinnedH2DChunkBytes(e != nullptr ? std::string_view(e) : std::string_view());
  }();
  return bytes;
}

// Ring state, owned by the backend and LIVING ACROSS CALLS. That is the part
// worth testing: a second Copy must wait on the previous Copy's still-in-flight
// chunk before it refills that slot, and a ring that only tracked reuse within
// one call would overwrite bytes a DMA was still reading.
struct StagedH2DRing {
  size_t n_buffers = kPinnedH2DBuffers;
  size_t next_slot = 0;
  bool in_flight[kPinnedH2DBuffers] = {};
};

// The loop, generic over the three device actions so it is drivable with fakes.
// Returns the number of chunks enqueued (0 when the ring is disabled).
//
//   wait(slot)                     — block until that slot's last upload is done
//   stage(slot, offset, bytes)     — copy source bytes into the slot's buffer
//   enqueue(slot, offset, bytes)   — async upload from the slot, then record
//
// The call returns with the last chunks still in flight on the stream, so
// Backend::Copy's asynchronous contract is unchanged. What IS stronger than
// before is that the SOURCE has been fully consumed by the time it returns.
template <class Wait, class Stage, class Enqueue>
inline size_t RunStagedH2D(StagedH2DRing& ring, size_t bytes, size_t chunk_bytes,
                           Wait wait, Stage stage, Enqueue enqueue) {
  if (chunk_bytes == 0 || ring.n_buffers == 0 || ring.n_buffers > kPinnedH2DBuffers) {
    return 0;
  }
  size_t chunks = 0;
  for (size_t off = 0; off < bytes; off += chunk_bytes) {
    const size_t n = (bytes - off) < chunk_bytes ? (bytes - off) : chunk_bytes;
    const size_t slot = ring.next_slot;
    if (ring.in_flight[slot]) {
      wait(slot);
      ring.in_flight[slot] = false;
    }
    stage(slot, off, n);
    enqueue(slot, off, n);
    ring.in_flight[slot] = true;
    ring.next_slot = (slot + 1) % ring.n_buffers;
    ++chunks;
  }
  return chunks;
}

// The instrument. A bounded-residency guarantee changes no byte, so a
// byte-equality case cannot see whether the ring engaged at all — the same
// reason NoteGgufPrefaultedSpan exists for the prefault. These counters are what
// tests/vt/test_backend_cross_device.cpp asserts on to prove the production call
// site in rocm_backend.hip is reached.
struct PinnedH2DStats {
  uint64_t staged_copies = 0;   // Copy() calls that took the ring
  uint64_t direct_copies = 0;   // Copy() calls that stayed on the single call
  uint64_t chunks = 0;          // pinned chunks enqueued, in total
  size_t max_chunk_bytes = 0;   // largest single pinned -> device transfer
  size_t ring_bytes = 0;        // pinned bytes actually allocated, 0 until used
};

namespace detail {
struct PinnedH2DCounters {
  std::atomic<uint64_t> staged_copies{0};
  std::atomic<uint64_t> direct_copies{0};
  std::atomic<uint64_t> chunks{0};
  std::atomic<size_t> max_chunk_bytes{0};
  std::atomic<size_t> ring_bytes{0};
};
// Relaxed atomics, not a plain counter: direct_copies is incremented on EVERY
// Backend::Copy, which is a hot path (about 1,361 ResidentWeight calls per
// forward step on this row's checkpoint), and Copy is reachable from more than
// one thread. A relaxed fetch_add is nanoseconds beside a hipMemcpyAsync.
inline PinnedH2DCounters& Counters() {
  static PinnedH2DCounters c;
  return c;
}
}  // namespace detail

inline void NotePinnedH2DStaged(size_t chunks, size_t max_chunk_bytes) {
  auto& c = detail::Counters();
  c.staged_copies.fetch_add(1, std::memory_order_relaxed);
  c.chunks.fetch_add(chunks, std::memory_order_relaxed);
  size_t prev = c.max_chunk_bytes.load(std::memory_order_relaxed);
  while (max_chunk_bytes > prev &&
         !c.max_chunk_bytes.compare_exchange_weak(prev, max_chunk_bytes,
                                                  std::memory_order_relaxed)) {
  }
}

inline void NotePinnedH2DDirect() {
  detail::Counters().direct_copies.fetch_add(1, std::memory_order_relaxed);
}

inline void NotePinnedH2DRingBytes(size_t bytes) {
  detail::Counters().ring_bytes.store(bytes, std::memory_order_relaxed);
}

inline PinnedH2DStats PinnedH2DSnapshot() {
  auto& c = detail::Counters();
  PinnedH2DStats s;
  s.staged_copies = c.staged_copies.load(std::memory_order_relaxed);
  s.direct_copies = c.direct_copies.load(std::memory_order_relaxed);
  s.chunks = c.chunks.load(std::memory_order_relaxed);
  s.max_chunk_bytes = c.max_chunk_bytes.load(std::memory_order_relaxed);
  s.ring_bytes = c.ring_bytes.load(std::memory_order_relaxed);
  return s;
}

}  // namespace vt::rocm
