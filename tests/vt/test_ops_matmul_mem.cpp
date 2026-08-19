// Allocation bound for the shared CPU GEMM seam (`vt::MatmulBT`), row
// LTX25-TEXT-LINEAR-MEM, .agents/specs/ltx25-text-linear-mem.md, issue #1286.
//
// WHY THIS FILE EXISTS. #1286 reported that routing the LTX-2.5 caption
// projection through `vt::MatmulBT` (#1252) raised peak host memory by ~26 GiB
// and aborted a full-model render, and named the seam's per-worker buffers as
// the first suspect: "~19 live tiles of a [rows, 188160]-shaped f32
// intermediate". That attribution is refuted in the spec — the +26 GiB was the
// box's starting occupancy, and the seam's measured cost at 20 workers is
// 232 MiB — but the refutation does not repair what looking for it exposed:
// NOTHING IN THIS TREE BOUNDS THE SEAM'S MEMORY. Every GEMM gate here asserts
// values or byte equality, and a kernel that allocated a whole intermediate per
// worker would pass all of them, on every model, silently, until a box ran out.
//
// So the bound is written now, while the correct number is known, rather than
// after the next report. The property is the one `cpu_ops.cpp:122-125` already
// claims in a comment and nothing checked: the seam's ONLY per-call allocation
// is the widened-activation buffer, sized by ggml's 16-row `blck_1` tile
// (ggml-cpu.c:1192-1194) and NOT by the chunk's row span or by the whole
// activation. Bytes requested across a call are therefore
// `nthreads x 16 x K x 4` and INDEPENDENT of the row count.
//
// WHY THE INSTRUMENT IS AN ALLOCATION COUNTER AND NOT PEAK RSS. Peak RSS is the
// more physical quantity and it was tried first. It cannot carry this gate: the
// operands are freed between measurements, glibc retains the arena, and the
// seam's next tile is then served from pages that are already resident. The
// first draft of this file measured `VmHWM` around `/proc/self/clear_refs` and
// read `growth_bytes=0` for EVERY thread count and EVERY row count while
// passing every bound — a mute switch that would have reported a green over a
// kernel doing anything at all. A replaced global `operator new` counts what
// the seam ASKS FOR, which is the quantity the bound is about, and no allocator
// policy can silence it.
//
// The bound is upper-only, and an explicit LIVENESS case sits beside it: an
// upper bound whose measurement can read zero is not a bound. That case
// requires the counter to see at least six of eight fresh workers take a tile,
// so the numbers the gate compares are known to be real ones.
//
// Linux-only registration (tests/CMakeLists.txt) is not about the counter,
// which is portable, but about `Threadpool`'s worker model and the
// process-lifetime `thread_local` this measures; a case that skipped instead
// would print `Status: SUCCESS!` over `assertions: 0`, which reads as a pass.
#include <doctest/doctest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

#include "vt/ops.h"
#include "vt/tensor.h"
#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting, via -I src

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

// Live bytes handed out by the replaced global allocator, and the high-water
// mark of that figure. `relaxed` is enough: the mark is read only after the
// measured call has joined every worker through the pool's final barrier.
std::atomic<int64_t> g_live{0};
std::atomic<int64_t> g_high{0};

void Account(int64_t delta) {
  const int64_t live = g_live.fetch_add(delta, std::memory_order_relaxed) + delta;
  int64_t high = g_high.load(std::memory_order_relaxed);
  while (live > high &&
         !g_high.compare_exchange_weak(high, live, std::memory_order_relaxed)) {
  }
}

// Every allocation carries its size in a header, because the sized-delete
// overloads are not guaranteed to be the ones the library calls.
constexpr size_t kHeader = 32;  // keeps max_align_t alignment for the payload

void* Alloc(size_t bytes) {
  void* raw = std::malloc(bytes + kHeader);
  if (raw == nullptr) throw std::bad_alloc();
  *static_cast<size_t*>(raw) = bytes;
  Account(static_cast<int64_t>(bytes));
  return static_cast<char*>(raw) + kHeader;
}

void Free(void* p) noexcept {
  if (p == nullptr) return;
  void* raw = static_cast<char*>(p) - kHeader;
  Account(-static_cast<int64_t>(*static_cast<size_t*>(raw)));
  std::free(raw);
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

constexpr int64_t kMiB = 1024 * 1024;

// Bytes the allocator was asked for across one `vt::MatmulBT` on `nthreads`
// workers at [rows, k] x [n, k]. The operands are allocated BEFORE the mark is
// armed, so the figure is the call's own.
//
// The pool is constructed fresh every time. That is deliberate: `af` is a
// process-lifetime `thread_local`, so a reused worker would allocate nothing
// the second time and the measurement would read zero for a reason that has
// nothing to do with the kernel.
int64_t AllocBytesOfMatmulBT(int nthreads, int64_t rows, int64_t k, int64_t n) {
  std::vector<float> a(static_cast<size_t>(rows * k), 0.5f);
  std::vector<float> b(static_cast<size_t>(n * k), 0.25f);
  std::vector<float> out(static_cast<size_t>(rows * n), 0.0f);

  auto pool = std::make_unique<vt::cpu::Threadpool>(nthreads);
  vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(pool.get());

  g_high.store(g_live.load(std::memory_order_relaxed), std::memory_order_relaxed);
  const int64_t before = g_high.load(std::memory_order_relaxed);

  Queue q{Cpu(), nullptr};
  Tensor a_t = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {rows, k});
  Tensor b_t = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {n, k});
  Tensor o_t = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {rows, n});
  vt::MatmulBT(q, o_t, a_t, b_t);

  const int64_t growth = g_high.load(std::memory_order_relaxed) - before;
  vt::cpu::Threadpool::SwapForTesting(previous);
  return growth;
}

// K is large so one tile (16 x K x 4) is far above the noise of any incidental
// allocation the dispatch makes; N is small so a case stays well under a
// second. K = 65536 puts one tile at exactly 4 MiB.
constexpr int64_t kK = 65536;
constexpr int64_t kN = 32;
constexpr int64_t kTileBytes = 16 * kK * 4;

// The fixed term covers the pool's own per-worker state and the `std::function`
// the dispatch wraps; the per-thread term is doubled so an allocator or a
// future tile that rounds up cannot red the gate. Neither is wide enough to
// admit a second tile-sized buffer per worker, which is what keeps it a bound
// rather than a formality.
int64_t Bound(int nthreads) {
  return 8 * kMiB + static_cast<int64_t>(nthreads) * 2 * kTileBytes;
}

}  // namespace

// Replaced global allocation functions. Standard replacements, so they serve
// the whole program including the seam's `thread_local` buffers.
void* operator new(size_t bytes) { return Alloc(bytes); }
void* operator new[](size_t bytes) { return Alloc(bytes); }
void* operator new(size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return Alloc(bytes);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return Alloc(bytes);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* p) noexcept { Free(p); }
void operator delete[](void* p) noexcept { Free(p); }
void operator delete(void* p, size_t) noexcept { Free(p); }
void operator delete[](void* p, size_t) noexcept { Free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { Free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { Free(p); }

TEST_CASE("the allocation counter SEES the seam's per-worker tiles (#1286)") {
  REQUIRE(kTileBytes == 4 * kMiB);
  // Eight fresh workers, every one of which is handed a starting chunk by
  // `MatmulChunked` (`current_chunk = ith`), so every one of them widens a
  // tile. Six is the floor rather than eight so the case does not depend on
  // the work-stealing cursor handing out a ninth chunk in any particular
  // order — but it is far enough above zero that a counter reading nothing
  // cannot pass, which is the whole point of this case.
  const int64_t growth = AllocBytesOfMatmulBT(8, 256, kK, kN);
  INFO("growth_bytes=" << growth << " tile_bytes=" << kTileBytes);
  CHECK(growth >= 6 * kTileBytes);
}

TEST_CASE("vt::MatmulBT allocates the per-worker TILE, not the whole "
          "intermediate (#1286)") {
  SUBCASE("bytes requested stay inside the tile bound at every worker count") {
    for (const int nthreads : {1, 2, 4, 8}) {
      const int64_t growth = AllocBytesOfMatmulBT(nthreads, 256, kK, kN);
      INFO("nthreads=" << nthreads << " growth_bytes=" << growth
                       << " bound_bytes=" << Bound(nthreads));
      CHECK(growth >= 0);
      CHECK(growth <= Bound(nthreads));
    }
  }

  // The half of the claim a thread sweep cannot see, and the one that fails
  // when a kernel starts widening the whole activation — the shape #1286
  // hypothesised. 16x the rows must not buy a single extra tile.
  SUBCASE("bytes requested do not scale with the ROW count") {
    constexpr int kThreads = 4;
    const int64_t few = AllocBytesOfMatmulBT(kThreads, 64, kK, kN);
    const int64_t many = AllocBytesOfMatmulBT(kThreads, 1024, kK, kN);
    INFO("rows=64 growth_bytes=" << few << "  rows=1024 growth_bytes=" << many
                                 << "  bound_bytes=" << Bound(kThreads));
    CHECK(few <= Bound(kThreads));
    CHECK(many <= Bound(kThreads));
    CHECK(many - few <= kTileBytes);
  }
}
