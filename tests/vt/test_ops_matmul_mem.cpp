// Allocation bound for the shared CPU GEMM seam (`vt::Matmul` and
// `vt::MatmulBT`), row LTX25-TEXT-LINEAR-MEM,
// .agents/specs/ltx25-text-linear-mem.md, issue #1286.
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
// after the next report. The property is the one `cpu_ops.cpp:127-129` already
// claims in a comment and nothing checked: the seam's ONLY per-call allocation
// is the widened-activation buffer at `cpu_ops.cpp:130`, sized by ggml's 16-row
// `blck_1` tile (ggml-cpu.c:1192-1194) and NOT by the chunk's row span or by
// the whole activation. Bytes requested across a call are therefore
// `nthreads x 16 x K x 4` and INDEPENDENT of the row count.
//
// BOTH TEMPLATE INSTANTIATIONS ARE COVERED, because there are two buffers and
// not one. `MatmulOneChunk<false>` and `MatmulOneChunk<true>` are separate
// instantiations, each with its OWN `static thread_local af` (confirmed with
// `nm -C`), so the process-lifetime retention figure is per-instantiation and a
// process using both orientations at a large `K` retains up to twice it. The
// `<false>` arm is not a corner: `MatmulKernel` (`cpu_ops.cpp:292-294`) routes
// `vt::Matmul` there, which is what #1259's `Ltx2FuseLoraIntoTensor` calls, and
// `MatmulBTKernel`'s `b.elem_kn_repacked` lever (`cpu_ops.cpp:306-314`) routes
// there too.
//
// WHY THE INSTRUMENT IS AN ALLOCATION COUNTER AND NOT PEAK RSS. Peak RSS is the
// more physical quantity and it was tried first. It cannot carry this gate: the
// operands are freed between measurements, glibc retains the arena, and the
// seam's next tile is then served from pages that are already resident. The
// first draft of this file measured `VmHWM` around `/proc/self/clear_refs` and
// read `growth_bytes=0` for EVERY thread count and EVERY row count while
// passing every bound — a mute switch that would have reported a green over a
// kernel doing anything at all. Replaced global allocation functions count what
// the seam ASKS FOR, which is the quantity the bound is about, and that figure
// is immune to the allocator's RETENTION policy.
//
// EXACTLY WHAT THE COUNTER SEES, because a broad claim over a narrow instrument
// is how the first draft went wrong. Two mechanisms, and the second exists only
// because a review proved the first was not enough.
//
// 1. It replaces the complete set of global `operator new`/`operator delete`
//    overloads: plain, array, `nothrow`, and the C++17 `std::align_val_t`
//    family. That covers every C++ allocation in the program, including every
//    `std::vector` and every `std::function`, whatever alignment it asks for.
//    The aligned family was MISSING from the first version, and a
//    whole-activation defect delivered through a 64-byte-aligned
//    `operator new` read as byte-identical growth and passed every bound.
// 2. It counts the C allocator through the LINKER -- `-Wl,--wrap=malloc` and
//    friends in tests/CMakeLists.txt -- rather than by defining `malloc`. A
//    definition would be a second strong symbol beside AddressSanitizer's own
//    interceptor and would break the `sanitize-cpu` lane; `--wrap` redirects
//    the CALLS made by the objects in this link, which includes `libvllm.a` and
//    so `cpu_ops.cpp`, and leaves every symbol where it was.
//
// What is still NOT covered, stated rather than left to be found: `mmap` and
// `sbrk` called directly, any allocator reached inside a shared library's own
// internal calls (`--wrap` binds at this link, not inside `libstdc++.so`), and
// `malloc` called from within THIS translation unit, which the compiler folds
// before the linker ever sees a symbol reference. The last one is measured, not
// supposed, and it is why the coverage case below calls through a `volatile`
// function pointer.
//
// The bound is upper-only, and two cases sit beside it because an upper bound
// whose measurement can read zero is not a bound. The COVERAGE case allocates
// through each replaced route and requires the counter to move by the amount
// asked for, so a lost overload reds instead of muting the gate. The LIVENESS
// case requires the counter to see at least six of eight fresh workers take a
// tile on each orientation, so the numbers the gate compares are known to be
// real ones.
//
// Registration is Linux-only (tests/CMakeLists.txt) and the reason is neither
// `/proc` nor the threadpool: the counter and the pool are both portable. It is
// that this gate replaces the global allocation functions FOR THE WHOLE PROGRAM,
// so it depends on every `delete` in the binary reaching the matching replaced
// `delete` — and on the fixed slack term below being calibrated against the
// library that does the allocating. Both were established against
// glibc/libstdc++ and against nothing else. A standard library linked as a
// separate runtime keeps its own `operator new`, and a pointer it allocated but
// this file frees would have the header offset applied to memory it never
// handed out, which is a crash rather than a failed assertion. Extending the
// registration is owed work, not a one-line change; a case that skipped instead
// would print `Status: SUCCESS!` over `assertions: 0`, which reads as a pass.
#include <doctest/doctest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
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

// Bytes requested through the C allocator inside the armed window. This one is
// CUMULATIVE rather than a high-water mark, because the wrappers below do not
// intercept `free` and so cannot know a released block's size. Cumulative
// bytes-requested is the quantity this file's contract already names, and for a
// buffer allocated once per worker the two agree exactly. It errs toward RED
// and never toward a silent green, and it reads zero on a clean run.
std::atomic<int64_t> g_raw{0};

// The linker's `--wrap` originals (tests/CMakeLists.txt). Declared here so the
// counter's own bookkeeping allocations go STRAIGHT to libc and are not counted
// twice -- and referencing them is what makes a dropped `--wrap` flag a LINK
// FAILURE rather than a silent loss of coverage.
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_calloc(size_t, size_t);
extern "C" void* __real_realloc(void*, size_t);
extern "C" void* __real_aligned_alloc(size_t, size_t);
extern "C" int __real_posix_memalign(void**, size_t, size_t);

// Every allocation carries its size in a header, because the sized-delete
// overloads are not guaranteed to be the ones the library calls.
constexpr size_t kHeader = 32;  // keeps max_align_t alignment for the payload

void* Alloc(size_t bytes) {
  void* raw = __real_malloc(bytes + kHeader);
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

// The C++17 over-aligned family. Kept separate from `Alloc` because the payload
// has to land on `align`, so the header cannot be a fixed 32 bytes: the padding
// is whichever of `align` and `kHeader` is larger, and it is recorded in the two
// words immediately below the payload (both padding choices are >= 16 bytes, so
// those two words are always inside the padding). The standard pairs an
// over-aligned `new` with an over-aligned `delete`, so `FreeAligned` never sees
// a pointer `Alloc` produced and vice versa.
void* AllocAligned(size_t bytes, size_t align) {
  if (align < alignof(std::max_align_t)) align = alignof(std::max_align_t);
  const size_t pad = align > kHeader ? align : kHeader;
  const size_t total = ((bytes + pad + align - 1) / align) * align;
  void* raw = __real_aligned_alloc(align, total);
  if (raw == nullptr) throw std::bad_alloc();
  char* payload = static_cast<char*>(raw) + pad;
  reinterpret_cast<size_t*>(payload)[-1] = pad;
  reinterpret_cast<size_t*>(payload)[-2] = bytes;
  Account(static_cast<int64_t>(bytes));
  return payload;
}

void FreeAligned(void* p) noexcept {
  if (p == nullptr) return;
  char* payload = static_cast<char*>(p);
  const size_t pad = reinterpret_cast<size_t*>(payload)[-1];
  const size_t bytes = reinterpret_cast<size_t*>(payload)[-2];
  Account(-static_cast<int64_t>(bytes));
  std::free(payload - pad);
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

constexpr int64_t kMiB = 1024 * 1024;

// Which member of the seam to measure. They are DIFFERENT template
// instantiations of `MatmulOneChunk` with different `thread_local` buffers, not
// two spellings of one path, so each needs its own measurement.
enum class Seam {
  kBT,  // vt::MatmulBT on a non-repacked [N,K] weight -> MatmulOneChunk<true>
  kNK,  // vt::Matmul on a [K,N] weight               -> MatmulOneChunk<false>
};

// Bytes the allocator was asked for across one seam call on `nthreads` workers
// at [rows, k] x (kBT ? [n, k] : [k, n]). The operands and the pool are
// allocated BEFORE the mark is armed, so the figure is the call's own.
//
// The pool is constructed fresh every time. That is deliberate: `af` is a
// process-lifetime `thread_local`, so a reused worker would allocate nothing
// the second time and the measurement would read zero for a reason that has
// nothing to do with the kernel.
int64_t AllocBytesOfSeam(Seam seam, int nthreads, int64_t rows, int64_t k, int64_t n) {
  std::vector<float> a(static_cast<size_t>(rows * k), 0.5f);
  std::vector<float> b(static_cast<size_t>(n * k), 0.25f);
  std::vector<float> out(static_cast<size_t>(rows * n), 0.0f);

  auto pool = std::make_unique<vt::cpu::Threadpool>(nthreads);
  vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(pool.get());

  g_high.store(g_live.load(std::memory_order_relaxed), std::memory_order_relaxed);
  const int64_t before = g_high.load(std::memory_order_relaxed);
  g_raw.store(0, std::memory_order_relaxed);

  Queue q{Cpu(), nullptr};
  Tensor a_t = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {rows, k});
  Tensor o_t = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {rows, n});
  if (seam == Seam::kBT) {
    Tensor b_t = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {n, k});
    vt::MatmulBT(q, o_t, a_t, b_t);
  } else {
    Tensor b_t = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {k, n});
    vt::Matmul(q, o_t, a_t, b_t);
  }

  // Both routes, added: the replaced `operator new` high-water mark and the
  // C allocator's cumulative request total. A defect that takes either one is
  // inside the figure the bound is compared against.
  const int64_t growth = (g_high.load(std::memory_order_relaxed) - before) +
                         g_raw.load(std::memory_order_relaxed);
  vt::cpu::Threadpool::SwapForTesting(previous);
  return growth;
}

// K is large so one tile (16 x K x 4) is far above the noise of any incidental
// allocation the dispatch makes; N is small so a case stays well under a
// second. K = 65536 puts one tile at exactly 4 MiB.
constexpr int64_t kK = 65536;
constexpr int64_t kN = 32;
constexpr int64_t kTileBytes = 16 * kK * 4;

// The bound is `nthreads` tiles plus a small fixed term, which is exactly the
// property this file claims: the seam's only per-call allocation is ONE 16-row
// tile per worker. The fixed term covers the `std::function` the dispatch wraps
// and anything else incidental inside the armed window — measured at 64 bytes
// in total, against 1 MiB of allowance. So there is four orders of magnitude of
// slack on the term that is not supposed to grow, and none on the term that is.
//
// What that detects, as a number rather than as a claim. A second tile-sized
// buffer per worker costs `2 x (nthreads - 1)` tiles once the calling thread's
// buffer is already sized, and that exceeds the bound from `nthreads = 4`
// upward — measured at 24 MiB against 17 MiB on 4 workers and 56 MiB against
// 33 MiB on 8. At 1 and 2 workers it can fit inside `nthreads` tiles and is not
// reliably caught, which is why the sweeps below run to 8 rather than stopping
// at 2. The earlier `2 * kTileBytes` per-thread term caught it at NO worker
// count while its comment claimed it could not be fitted at all.
int64_t Bound(int nthreads) {
  return kMiB + static_cast<int64_t>(nthreads) * kTileBytes;
}

// A geometry that forces the chunk grid to COLLAPSE, so one chunk spans more
// than 16 rows. `MatmulChunked` rewrites the grid to one chunk per thread when
// `nchunk0 * nchunk1 < nth * 4` (`cpu_ops.cpp:257-260`, ggml-cpu.c:1404-1408).
// That branch is LIVE and is the ggml-mirrored default, because
// `VT_CPU_MATMUL_STEAL` is off. After it, `dr1` is `ceil(rows / nth)` when the
// activation is the longer axis and `rows` outright when the weight is — and it
// is 16 or less at EVERY geometry the sweeps above happen to run, which is the
// only reason those sweeps cannot see a chunk-sized buffer.
//
// That distinction is the whole content of the "not by the chunk's row span"
// half of the claim, and without this case nothing here can see it: a kernel
// that sized `af` by `ir1_end - iir1` instead of by `min(16, ...)` is
// byte-identical in growth at every other geometry in this file. With
// `n = 16 <= chunk_size` the column grid is one chunk wide, so `nchunk0 = 1`
// and `nchunk1 = ceil(128/16) = 8 < 4 * 4`; the collapse then gives
// `nchunk1 = 4` and `dr1 = 32`, exactly two tiles per chunk.
//
// The same condition has a much larger form that is NOT reachable today and is
// recorded rather than tested. `IsNuma()` (`cpu_threadpool.h:74`) is
// `constexpr false` in this tree — NUMA is an unported part of the threadpool
// (`cpu_threadpool.h:25`) — and it forces the collapse UNCONDITIONALLY where
// upstream implements it. At the shipped LTX-2.5 caption projection the weight
// is the longer axis (`n = 4096 > rows = 1024`), so that collapse would give
// `nchunk1 = 1` and `dr1 = 1024`: a chunk-sized buffer would be 770 MB per
// worker, which is #1286's hypothesis exactly. Porting NUMA therefore needs
// this case, not the sweeps.
constexpr int64_t kCollapseRows = 128;
constexpr int64_t kCollapseN = 16;
constexpr int kCollapseThreads = 4;

}  // namespace

// Replaced global allocation functions. Standard replacements, so they serve
// the whole program including the seam's `thread_local` buffers. The complete
// set is here on purpose: an over-aligned `operator new` that fell through to
// the library's own would be a silent hole in the counter, and it is the
// natural allocation shape for a SIMD scratch buffer in this kernel.
// The C allocator, reached through the linker rather than through a symbol
// definition. `-Wl,--wrap=malloc` redirects the calls made BY THE OBJECTS IN
// THIS LINK -- which includes `libvllm.a`, and so `cpu_ops.cpp` -- without
// defining `malloc`, so AddressSanitizer's own interceptor keeps its symbol and
// `__real_malloc` resolves to whichever allocator is actually installed. A
// plain `extern "C" void* malloc(...)` here would instead be a second strong
// definition, which is the thing that breaks the `sanitize-cpu` lane.
//
// `free` is deliberately NOT wrapped. Releasing a block gives no size without a
// header, and a header would mean applying an offset to pointers that libc and
// libstdc++ allocated before this file was reached -- a heap corruption rather
// than a failed assertion.
extern "C" void* __wrap_malloc(size_t n) {
  g_raw.fetch_add(static_cast<int64_t>(n), std::memory_order_relaxed);
  return __real_malloc(n);
}
extern "C" void* __wrap_calloc(size_t count, size_t size) {
  g_raw.fetch_add(static_cast<int64_t>(count) * static_cast<int64_t>(size),
                  std::memory_order_relaxed);
  return __real_calloc(count, size);
}
extern "C" void* __wrap_realloc(void* p, size_t n) {
  g_raw.fetch_add(static_cast<int64_t>(n), std::memory_order_relaxed);
  return __real_realloc(p, n);
}
extern "C" void* __wrap_aligned_alloc(size_t align, size_t n) {
  g_raw.fetch_add(static_cast<int64_t>(n), std::memory_order_relaxed);
  return __real_aligned_alloc(align, n);
}
extern "C" int __wrap_posix_memalign(void** out, size_t align, size_t n) {
  g_raw.fetch_add(static_cast<int64_t>(n), std::memory_order_relaxed);
  return __real_posix_memalign(out, align, n);
}

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
void* operator new(size_t bytes, std::align_val_t a) {
  return AllocAligned(bytes, static_cast<size_t>(a));
}
void* operator new[](size_t bytes, std::align_val_t a) {
  return AllocAligned(bytes, static_cast<size_t>(a));
}
void* operator new(size_t bytes, std::align_val_t a, const std::nothrow_t&) noexcept {
  try {
    return AllocAligned(bytes, static_cast<size_t>(a));
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](size_t bytes, std::align_val_t a, const std::nothrow_t&) noexcept {
  try {
    return AllocAligned(bytes, static_cast<size_t>(a));
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
void operator delete(void* p, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
  FreeAligned(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept {
  FreeAligned(p);
}

TEST_CASE("the counter SEES every allocation route it claims to cover (#1286)") {
  // The gate's claim is about which allocation APIs are counted, and that claim
  // is executable rather than prose. Each route is exercised at a size far above
  // any incidental traffic and the counter is required to move by AT LEAST it —
  // at least, because a `new` inside doctest's own INFO machinery could add to
  // it, never subtract. A lost overload reds here instead of quietly reducing
  // every bound below to a formality.
  constexpr size_t kProbe = 8u << 20;  // 8 MiB, 2x the tile

  const int64_t base_plain = g_live.load(std::memory_order_relaxed);
  auto* plain = static_cast<char*>(::operator new(kProbe));
  const int64_t moved_plain = g_live.load(std::memory_order_relaxed) - base_plain;
  ::operator delete(plain);
  INFO("plain operator new moved " << moved_plain);
  CHECK(moved_plain >= static_cast<int64_t>(kProbe));

  const int64_t base_array = g_live.load(std::memory_order_relaxed);
  auto* arr = new char[kProbe];
  const int64_t moved_array = g_live.load(std::memory_order_relaxed) - base_array;
  delete[] arr;
  INFO("array operator new moved " << moved_array);
  CHECK(moved_array >= static_cast<int64_t>(kProbe));

  // The over-aligned route, which is the one review mutation M-D took and the
  // one the first version of this file did not replace at all: it fell through
  // to the library and the counter read a whole-activation defect as zero.
  // 64 bytes is the cache-line request an AVX-512 kernel makes. (M-E's
  // `std::malloc` route has no case here because it is NOT covered — see the
  // header, and `## Owed` in the spec.)
  for (const size_t align : {size_t{64}, size_t{256}}) {
    const int64_t base = g_live.load(std::memory_order_relaxed);
    auto* over = static_cast<char*>(
        ::operator new(kProbe, static_cast<std::align_val_t>(align)));
    const int64_t moved = g_live.load(std::memory_order_relaxed) - base;
    const bool aligned = (reinterpret_cast<uintptr_t>(over) % align) == 0;
    // Touch both ends: a header scheme that mis-sizes the block corrupts the
    // heap rather than failing an assertion, and this is where it would show.
    over[0] = 1;
    over[kProbe - 1] = 2;
    ::operator delete(over, static_cast<std::align_val_t>(align));
    INFO("aligned operator new align=" << align << " moved " << moved
                                       << " aligned=" << (aligned ? 1 : 0));
    CHECK(moved >= static_cast<int64_t>(kProbe));
    CHECK(aligned);
  }

  // The C-allocator route, which is a LINKER redirection and not a symbol
  // definition, and is therefore the one that can go inert without any source
  // change. The call goes through a `volatile` function pointer because a
  // direct `std::malloc(...)` in THIS translation unit is folded by the
  // compiler and never becomes a reference to the `malloc` symbol, so it is
  // never wrapped -- measured in a standalone probe, and the reason this is
  // written the awkward way. Product code is always the separate-TU case.
  {
    const int64_t base_raw = g_raw.load(std::memory_order_relaxed);
    void* (*volatile c_alloc)(size_t) = &std::malloc;
    void (*volatile c_free)(void*) = &std::free;
    void* raw = c_alloc(kProbe);
    const int64_t moved_raw = g_raw.load(std::memory_order_relaxed) - base_raw;
    REQUIRE(raw != nullptr);
    c_free(raw);
    INFO("std::malloc moved the raw counter by " << moved_raw);
    CHECK(moved_raw >= static_cast<int64_t>(kProbe));
  }

  // And the counter must come back down, or every "growth" below is a running
  // total of everything the process ever did rather than the call's own cost.
  const int64_t leaked = g_live.load(std::memory_order_relaxed) - base_plain;
  INFO("live bytes still held after the probes: " << leaked);
  CHECK(leaked < static_cast<int64_t>(kProbe));
}

TEST_CASE("the allocation counter SEES the seam's per-worker tiles (#1286)") {
  REQUIRE(kTileBytes == 4 * kMiB);
  // Eight fresh workers, every one of which is handed a starting chunk by
  // `MatmulChunked` (`current_chunk = ith`), so every one of them widens a
  // tile. Six is the floor rather than eight so the case does not depend on
  // the work-stealing cursor handing out a ninth chunk in any particular
  // order — but it is far enough above zero that a counter reading nothing
  // cannot pass, which is the whole point of this case.
  //
  // Run on BOTH orientations: they are separate instantiations with separate
  // buffers, so a counter that saw one and not the other would read zero on
  // half the seam.
  const int64_t bt = AllocBytesOfSeam(Seam::kBT, 8, 256, kK, kN);
  INFO("MatmulBT growth_bytes=" << bt << " tile_bytes=" << kTileBytes);
  CHECK(bt >= 6 * kTileBytes);

  const int64_t nk = AllocBytesOfSeam(Seam::kNK, 8, 256, kK, kN);
  INFO("Matmul growth_bytes=" << nk << " tile_bytes=" << kTileBytes);
  CHECK(nk >= 6 * kTileBytes);
}

TEST_CASE("vt::MatmulBT allocates the per-worker TILE, not the whole "
          "intermediate (#1286)") {
  SUBCASE("bytes requested stay inside the tile bound at every worker count") {
    for (const int nthreads : {1, 2, 4, 8}) {
      const int64_t growth = AllocBytesOfSeam(Seam::kBT, nthreads, 256, kK, kN);
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
    const int64_t few = AllocBytesOfSeam(Seam::kBT, kThreads, 64, kK, kN);
    const int64_t many = AllocBytesOfSeam(Seam::kBT, kThreads, 1024, kK, kN);
    INFO("rows=64 growth_bytes=" << few << "  rows=1024 growth_bytes=" << many
                                 << "  bound_bytes=" << Bound(kThreads));
    CHECK(few <= Bound(kThreads));
    CHECK(many <= Bound(kThreads));
    CHECK(many - few <= kTileBytes);
  }
}

TEST_CASE("vt::Matmul — the OTHER instantiation, with its own buffer — is "
          "bounded the same way (#1286, #1259)") {
  // `MatmulOneChunk<false>`. Nothing above reaches it: `vt::MatmulBT` on a
  // non-repacked weight only instantiates `<true>`. It is the arm #1259's
  // `Ltx2FuseLoraIntoTensor` runs on the full model, and the arm the loader's
  // `elem_kn_repacked` lever selects, so leaving it unbounded would leave the
  // gate covering one of the two buffers the process actually holds.
  SUBCASE("bytes requested stay inside the tile bound at every worker count") {
    for (const int nthreads : {1, 2, 4, 8}) {
      const int64_t growth = AllocBytesOfSeam(Seam::kNK, nthreads, 256, kK, kN);
      INFO("nthreads=" << nthreads << " growth_bytes=" << growth
                       << " bound_bytes=" << Bound(nthreads));
      CHECK(growth >= 0);
      CHECK(growth <= Bound(nthreads));
    }
  }

  SUBCASE("bytes requested do not scale with the ROW count") {
    constexpr int kThreads = 4;
    const int64_t few = AllocBytesOfSeam(Seam::kNK, kThreads, 64, kK, kN);
    const int64_t many = AllocBytesOfSeam(Seam::kNK, kThreads, 1024, kK, kN);
    INFO("rows=64 growth_bytes=" << few << "  rows=1024 growth_bytes=" << many
                                 << "  bound_bytes=" << Bound(kThreads));
    CHECK(few <= Bound(kThreads));
    CHECK(many <= Bound(kThreads));
    CHECK(many - few <= kTileBytes);
  }
}

TEST_CASE("the tile is the 16-row blck_1 tile, not the CHUNK'S ROW SPAN "
          "(#1286)") {
  // At a collapsed grid one chunk spans `dr1 = 32` rows, so a kernel sized by
  // the chunk instead of by `blck_1` asks for two tiles per worker here and one
  // tile per worker everywhere else in this file. Without this case that defect
  // is invisible, and this shape is a live one rather than a contrivance: the
  // collapse it uses is the shipped default path, not the NUMA branch.
  //
  // The bound is the same one: two tiles per worker on 4 workers is 24 MiB
  // against 17 MiB, so it reds on the bound rather than needing its own
  // threshold.
  for (const Seam seam : {Seam::kBT, Seam::kNK}) {
    const int64_t growth =
        AllocBytesOfSeam(seam, kCollapseThreads, kCollapseRows, kK, kCollapseN);
    // `std::string`, not a ternary over two `const char*`: doctest's INFO
    // stringifies a `char*` operand as a BOOL, and the first draft of this line
    // printed `seam=1` for both arms.
    INFO("seam=" << std::string(seam == Seam::kBT ? "MatmulBT" : "Matmul")
                 << " rows=" << kCollapseRows << " n=" << kCollapseN
                 << " nthreads=" << kCollapseThreads
                 << " growth_bytes=" << growth
                 << " bound_bytes=" << Bound(kCollapseThreads));
    CHECK(growth >= 0);
    CHECK(growth <= Bound(kCollapseThreads));
  }
}
