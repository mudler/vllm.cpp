// Shared process-wide caching device allocator (DevicePool) — extracted VERBATIM
// from the Qwen3.6 forward (qwen3_5.cpp) so the dense Qwen3 forward (qwen3.cpp)
// reuses the SAME pooled-scratch machinery instead of raw per-op Backend
// Alloc/Free. The relocation was byte-for-byte the qwen3_5.cpp definitions; what
// has changed since is that a pool is now bound to ONE DEVICE (see below), and
// the accessors take the backend that names it.
//
// Rationale: both cudaMalloc AND cudaFree SYNCHRONIZE the whole device, so the
// per-op DBuf alloc/free churn in a forward (thousands of tiny scratch buffers per
// step) is itself a sync storm. This pool reuses freed blocks (size-class keyed)
// instead of hitting cudaMalloc/cudaFree, so after a brief warm-up almost every
// DBuf lifetime is sync-free. Reuse is safe under the forward's single-queue
// (single-stream) ordering: a block returned to the pool is only handed back out
// on the same queue, and CUDA stream ordering guarantees the op that last touched
// the block has completed before any reused op runs — no host sync needed. Blocks
// are never returned to the driver (leak at process exit, like the cublasLt
// workspace); the pool is bounded by the forward's peak concurrent scratch.
//
// ONE POOL PER DEVICE (#516, .agents/specs/pool-device-key.md). Until this was
// fixed there was ONE pool for the whole process and its free list was keyed by
// byte size class alone, so a block allocated through one backend was handed to
// the next caller of that size class whatever device it was running on. One
// fault, two symptoms, chosen by direction: a cudaMalloc block reaching a
// CPU-backend forward SIGSEGVs host-side (and compute-sanitizer is CLEAN,
// because the fault is not on the device), while a host block reaching a CUDA
// forward returned a UNIFORM 0x7fff0000 quiet NaN — computed and propagated, not
// garbage read. vLLM never had the bug because it never had the design: its
// allocation handle carries the device as field 0
// (vllm/device_allocator/__init__.py:12-14 @ pin 555967922) and torch's cache is
// per-device by construction (c10/cuda/CUDACachingAllocator.h:118-172).
//
// So a `DevicePool` is BOUND to one backend, `Pool(b)` resolves the pool for a
// device, and there is deliberately NO way to spell "the pool" without naming a
// device. The `vt::Backend*` IS the device identity: the registry hands out
// exactly one Backend* per Device{type,index} (vt/backend.h, kMaxDevicesPerType),
// and GetBackend(type) and GetBackend(Device{type,0}) return the identical
// pointer — so the device enters the key with no new virtual on vt::Backend and
// therefore no edit to any backend implementation.
#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vt/backend.h"

namespace vllm {

// SIZE-CLASS BUCKETING (prefill sync-cudaMalloc kill). Keying the pool on the
// EXACT byte size defeats reuse during prefill: under continuous batching each
// engine step processes a different token count T (prefill-chunk tokens + decode
// tokens), so every [T,*] scratch is a byte size that step-1 never saw — an
// exact-key MISS -> a SYNCHRONOUS cudaMalloc (device-serializing, does NOT
// overlap compute) on the forward's host thread, thousands per prefill. Rounding
// the request UP to a size class (keep the top `kClassBits` significant bits;
// <=1/2^kClassBits ~ 6.25% over-allocation) makes nearby-T scratch of the same
// op share ONE block, so after warm-up almost every prefill DBuf is a pool hit.
// The returned block is >= the requested bytes (the Tensor view uses only the
// logical prefix), and Put/Get round identically so a block always returns to
// its own class bucket. VT_POOL_EXACT=1 restores exact keying (A/B measurement).
class DevicePool {
 public:
  // A pool serves exactly ONE device, named at construction. Resolve one with
  // Pool(b) / AuxPool(b) rather than building your own; a directly-constructed
  // pool is for tests that want an isolated free list.
  explicit DevicePool(vt::Backend& b) : backend_(&b) {}
  DevicePool(const DevicePool&) = delete;
  DevicePool& operator=(const DevicePool&) = delete;

  // Size-class rounding is `private static`, and `tests/vt/test_cpu_isa_x86.cpp`
  // exercises it directly (including the overflow throw) without a backend to
  // build a pool on — this is that seam. It is deliberately `static`, so binding
  // a pool to a device did not change it.
  static size_t SizeClassForTest(size_t bytes) { return ClassOf(bytes); }

  void* Get(vt::Backend& b, size_t bytes) {
    RequireOwnDevice(b, "Get");
    // BYPASS lane (VT_POOL_BYPASS=1) — the pool is a DETECTOR BLIND SPOT and
    // this is how you see through it. Two ways it hides a real defect from
    // compute-sanitizer:
    //   1. size-class rounding hands back a block up to 6.25% LARGER than the
    //      logical tensor, so a write past the last row lands inside the same
    //      driver allocation and memcheck reports nothing;
    //   2. blocks are never returned to the driver, so a use-after-free of a
    //      released DBuf reads memory that is still legally mapped — and, worse,
    //      may already have been handed to an unrelated op.
    // Under bypass every Get is an EXACT-size driver allocation and every Put is
    // a real Free, which restores both boundaries for the detector. It is a
    // debugging lane only: it reinstates the per-op cudaMalloc/cudaFree sync
    // storm this pool exists to remove, so it is never a timing configuration.
    if (Bypass()) return b.Alloc(bytes);
    const size_t key = ClassOf(bytes);
    {
      std::lock_guard<std::mutex> lk(mu_);
      ClassState& cs = classes_[key];
      ++cs.live;
      if (cs.live - cs.base > cs.peak) cs.peak = cs.live - cs.base;
      if (!cs.free.empty()) {
        void* p = cs.free.back();
        cs.free.pop_back();
        retained_ -= key;
        ++hits_;
        return p;
      }
      ++misses_;
    }
    try {
      return b.Alloc(key);
    } catch (...) {
      // The block was never handed out, so it must not count as live: a Get that
      // threw has no matching Put, and a leaked `live` would make every later
      // demand profile read one block too high for this class forever.
      std::lock_guard<std::mutex> lk(mu_);
      ClassState& cs = classes_[key];
      if (cs.live > 0) --cs.live;
      throw;
    }
  }
  // Uncapped retention (deliberately-retained cross-step buffers: the device
  // logits / MTP hidden handed off via a shared_ptr deleter). Bytes are always
  // returned to the free list — the cross-step buffers are not cap-evicted.
  //
  // This used to take no backend at all, which is how ~28 copy-pasted
  // `shared_ptr` deleters came to name neither the device nor the pool: they
  // closed over a byte count and called `Pool().Put(alloc, q)`, so a block from
  // ANY device was returned to the one global pool. The aux-stream half of that
  // was LATENT rather than live: the deleter would have returned an AuxPool
  // block to the main pool, but no `Release()` site sat under an
  // `ActivePoolScope` — the four scope regions are leaf-ward of all nine of
  // them — so the wrong-pool return was reachable only by adding a site, which
  // is precisely the mistake that then costs a debugging campaign.
  // `DBuf::ReleaseShared()` is now the only way to build that carrier and it
  // captures the buffer's own pool and backend, so neither half can come back
  // (#516).
  void Put(vt::Backend& b, size_t bytes, void* p) {
    RequireOwnDevice(b, "Put");
    // Bypass: free for real so a later use-after-free traps.
    if (Bypass()) {
      b.Free(p);
      return;
    }
    const size_t key = ClassOf(bytes);
    std::lock_guard<std::mutex> lk(mu_);
    ClassState& cs = classes_[key];
    if (cs.live > 0) --cs.live;
    retained_ += key;
    cs.free.push_back(p);
  }
  // Cap-aware retention for the high-frequency forward scratch (DBuf). The soft
  // cap comes from the platform's residency_policy().device_pool_cap_bytes
  // (BACKEND-PLATFORM item 2). cap == 0 is UNCAPPED — the identical fast path as
  // the uncapped Put above and thus behavior-preserving on GB10 (cap 0 today).
  // When a discrete GPU sets a bound, scratch over the cap is freed to the driver
  // rather than pooled, so the reuse pool self-limits without a model edit.
  void Put(vt::Backend& b, size_t bytes, void* p, size_t cap) {
    RequireOwnDevice(b, "Put");
    if (Bypass()) {
      b.Free(p);
      return;
    }
    const size_t key = ClassOf(bytes);
    {
      std::lock_guard<std::mutex> lk(mu_);
      ClassState& cs = classes_[key];
      if (cs.live > 0) --cs.live;
      if (cap == 0 || retained_ + key <= cap) {
        retained_ += key;
        cs.free.push_back(p);
        return;
      }
    }
    b.Free(p);  // over the soft cap: to the driver, outside the lock
  }

  // Release every RETAINED block back to the driver, and report the bytes freed.
  //
  // The pool earns its keep WITHIN a phase, where the same size classes recur
  // every step and cudaMalloc/cudaFree would be a sync storm. Across a PHASE
  // CHANGE the retained classes are the wrong shapes for what comes next, so on
  // an UNCAPPED pool (`device_pool_cap_bytes == 0`, which is GB10/Thor today)
  // they are pure headroom loss at exactly the moment the next phase wants its
  // own working set. Draining at that boundary costs one cudaFree per retained
  // block, once, and is what keeps a big-canvas MiniMax-H3 VAE decode from
  // meeting the driver's OOM on top of 50 steps of denoise scratch.
  //
  // SAFETY: a class's free list only ever holds blocks a DBuf already returned, so nothing
  // live is touched. Under VT_POOL_BYPASS the free list is always empty (Put
  // frees straight through) and this is a no-op. And because a pool now holds
  // ONE device's blocks, `b.Free` is guaranteed to be the allocator that made
  // them — before #516 a drain could hand one device's block to another's Free.
  size_t Drain(vt::Backend& b) {
    RequireOwnDevice(b, "Drain");
    std::lock_guard<std::mutex> lk(mu_);
    size_t freed = 0;
    for (auto& entry : classes_) {
      for (void* p : entry.second.free) {
        b.Free(p);
        freed += entry.first;
      }
      entry.second.free.clear();
    }
    retained_ = 0;
    return freed;
  }

  // ── CAPTURE PRE-GROW (#1380) ───────────────────────────────────────────────
  // A `cudaMalloc` inside a captured region ABORTS the capture, so every
  // allocation the captured forward makes has to be a pool HIT. Warming the pool
  // by running the same forward eagerly first is necessary and NOT sufficient,
  // and the missing half is what #1380 measured: the capture RETAINS its
  // `[S, vocab]` logits, so one block never comes back to the free list, and the
  // NEXT capture at that shape is one block short. Since the pool is keyed by
  // SIZE CLASS rather than by tensor, the block the next capture then misses on
  // need not be the logits at all -- on `thor:gpu0` it was `dconv`, the GDN
  // causal-conv output (`qwen3_5.cpp:4665`), which at the gate's shape lands in
  // the SAME class as the retained logits.
  //
  // So the driver cannot pre-grow by naming one tensor. It asks the pool what
  // the step it just ran actually demanded, and asks for that to be servable.
  // `StepDemand` is the per-class PEAK number of blocks live at one time above
  // the baseline the last `MarkStepBoundary` recorded -- the transient demand,
  // not the retention -- so the profile is a property of the FORWARD and not of
  // whatever happened to be resident when it ran.
  using StepDemand = std::vector<std::pair<size_t, int64_t>>;

  // Close the current step for demand accounting. The next step's peak is
  // measured from the blocks that are live right now.
  void MarkStepBoundary() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& e : classes_) {
      e.second.peak = 0;
      e.second.base = e.second.live;
    }
  }

  // The per-class transient demand measured since the last `MarkStepBoundary`.
  // A driver takes this at the end of the EAGER step it runs at a shape and
  // hands it back at that shape's capture, which is why it is a value rather
  // than pool state: two shapes interleave through one pool, and last-step state
  // would answer for whichever step ran most recently instead of for this one.
  StepDemand StepDemandProfile() const {
    std::lock_guard<std::mutex> lk(mu_);
    StepDemand out;
    out.reserve(classes_.size());
    for (const auto& e : classes_)
      if (e.second.peak > 0) out.emplace_back(e.first, e.second.peak);
    return out;
  }

  // Make the free list able to serve `demand` without touching the driver, and
  // report how many blocks that cost. Call it OUTSIDE the captured region; a
  // block created here is a `cudaMalloc` and would abort a capture in progress.
  // Idempotent, and a no-op once the free list is deep enough — which is the
  // steady state, so a warm server pays nothing for it.
  //
  // IT IGNORES THE SOFT CAP, AND UNDER A NON-ZERO ONE THAT REOPENS #1380 BY THE
  // OTHER API. This function adds to `retained_` without consulting `cap`,
  // while `Put` above frees a returned block STRAIGHT TO THE DRIVER once
  // `retained_ + key > cap`. A `cudaFree` inside a captured region aborts the
  // capture exactly as a `cudaMalloc` does — and `Put` is called from INSIDE the
  // captured forward, every time a scratch `DBuf` there goes out of scope. So a
  // pre-grow that pushed `retained_` over the cap would arm the abort it exists
  // to prevent, at the first block the capture returned.
  //
  // Nothing can reach that today and this is a hazard note, not a live defect:
  // every platform resolves `residency_policy().device_pool_cap_bytes` to 0,
  // and `cap == 0` is the uncapped fast path in `Put`. WHOEVER FIRST SETS A
  // NON-ZERO CAP OWES THE FIX HERE — either pass `cap` in and refuse to grow
  // past it, or make the captured region's `Put` cap-exempt. Recorded under
  // `## Owed` in `.agents/specs/eng-cudagraph-break.md`; owner row
  // `ENG-CUDAGRAPH-BREAK`.
  size_t PreGrowForCapture(vt::Backend& b, const StepDemand& demand) {
    RequireOwnDevice(b, "PreGrowForCapture");
    if (Bypass()) return 0;  // no free list to grow; every Get is a driver call
    std::vector<size_t> missing;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (const auto& want : demand) {
        ClassState& cs = classes_[want.first];
        for (int64_t have = static_cast<int64_t>(cs.free.size()); have < want.second; ++have)
          missing.push_back(want.first);
      }
    }
    for (size_t key : missing) {
      void* p = b.Alloc(key);
      std::lock_guard<std::mutex> lk(mu_);
      classes_[key].free.push_back(p);
      retained_ += key;
    }
    return missing.size();
  }

  ~DevicePool() {
    if (std::getenv("VT_POOL_STATS") != nullptr) {
      const uint64_t h = hits_.load(), m = misses_.load();
      const double rate = (h + m) ? 100.0 * static_cast<double>(h) / static_cast<double>(h + m) : 0.0;
      // The backend pointer identifies WHICH device's pool this line is about:
      // a mixed-backend process now prints one line per device, and two lines
      // with no way to tell them apart would be worse than one wrong line.
      std::fprintf(stderr,
                   "[DevicePool backend=%p] hits=%llu misses(cudaMalloc)=%llu hit-rate=%.2f%% "
                   "distinct-classes=%zu\n",
                   static_cast<const void*>(backend_),
                   static_cast<unsigned long long>(h), static_cast<unsigned long long>(m),
                   rate, classes_.size());
    }
  }

 private:
  // The device check, on EVERY pool operation. A hard runtime throw and NOT an
  // `assert`: the SACRED gate builds are Release/NDEBUG, where an assert is
  // compiled out and the pre-fix behavior — a block silently crossing devices —
  // would come straight back. It is one predictable compare against a member,
  // against a `cudaMalloc` this pool exists to avoid.
  //
  // The only way to reach it is an `ActivePoolScope` pointing at another
  // device's pool, which is precisely the mistake this row exists to make
  // impossible to make quietly.
  void RequireOwnDevice(vt::Backend& b, const char* op) const {
    if (&b == backend_) return;
    char msg[192];
    std::snprintf(msg, sizeof(msg),
                  "DevicePool::%s called with backend %p on a pool bound to backend %p: a scratch "
                  "block must never cross devices (see .agents/specs/pool-device-key.md, #516)",
                  op, static_cast<const void*>(&b), static_cast<const void*>(backend_));
    throw std::logic_error(msg);
  }

  // VT_POOL_BYPASS=1 turns every Get/Put into a raw driver Alloc/Free (see Get).
  // Read once: it must not change between an allocation and its matching free,
  // or a pooled block would be handed to Backend::Free (or a driver block leaked
  // into the free list).
  static bool Bypass() {
    static const bool on = [] {
      const char* e = std::getenv("VT_POOL_BYPASS");
      return e != nullptr && e[0] == '1';
    }();
    return on;
  }

  // Round `bytes` up so it keeps at most kClassBits leading significant bits.
  // Exact keying when VT_POOL_EXACT=1 (A/B). Small sizes (< 2^kClassBits) key
  // exactly — there are few of them and the waste would be proportionally large.
  static size_t ClassOf(size_t bytes) {
    static const bool exact = [] {
      const char* e = std::getenv("VT_POOL_EXACT");
      return e != nullptr && e[0] == '1';
    }();
    if (exact || bytes == 0) return bytes == 0 ? 1 : bytes;
    constexpr int kClassBits = 4;  // <=6.25% over-allocation per class
    const int msb = static_cast<int>(std::bit_width(bytes)) - 1;
    if (msb < kClassBits) return bytes;
    const int shift = msb - kClassBits;
    const size_t mask = (static_cast<size_t>(1) << shift) - 1;
    if (bytes > std::numeric_limits<size_t>::max() - mask) {
      throw std::overflow_error("DevicePool size class rounding overflow");
    }
    return (bytes + mask) & ~mask;  // round up to a multiple of 2^shift
  }

  // One size class: its free blocks, and the demand accounting `StepDemand`
  // reads. `live` counts blocks handed out and not yet returned; `base` is what
  // `live` was at the last step boundary; `peak` is the largest `live - base`
  // since then, which is the number of blocks of this class a repeat of that
  // step needs on the free list.
  struct ClassState {
    std::vector<void*> free;
    int64_t live = 0;
    int64_t base = 0;
    int64_t peak = 0;
  };

  mutable std::mutex mu_;
  // THE DEVICE, and the reason this class exists in this shape. Every block in
  // a class's free list was allocated by this backend and will be freed by it;
  // nothing else may draw from or return to this pool.
  vt::Backend* backend_;
  std::unordered_map<size_t, ClassState> classes_;
  size_t retained_ = 0;  // bytes (class-rounded) held free, for the soft cap
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
};

namespace detail {

// Process-wide table of per-device pools. Tiny by construction: one entry per
// `vt::Backend*` the process ever allocates through, i.e. one per
// Device{type,index}. Entries are never erased, which is what lets Pool()'s
// memo below hold a raw pointer.
class PoolTable {
 public:
  DevicePool& For(vt::Backend& b) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& e : pools_)
      if (e.first == &b) return *e.second;
    pools_.emplace_back(&b, std::unique_ptr<DevicePool>(new DevicePool(b)));
    return *pools_.back().second;
  }

 private:
  std::mutex mu_;
  std::vector<std::pair<vt::Backend*, std::unique_ptr<DevicePool>>> pools_;
};

inline PoolTable& MainPoolTable() {
  static PoolTable t;
  return t;
}

// The (backend -> pool) resolution, memoized per thread. A DBuf resolves its
// pool on EVERY construction — thousands per forward step — and this pool's
// whole purpose is to avoid a synchronizing cudaMalloc, so paying a lock and a
// scan for it would be self-defeating. A process drives one device per host
// thread at a time, so the steady state here is a single pointer compare.
inline DevicePool& MemoizedPool(PoolTable& table, vt::Backend& b,
                                vt::Backend*& last_backend, DevicePool*& last_pool) {
  if (last_backend == &b) return *last_pool;
  DevicePool& p = table.For(b);
  last_backend = &b;
  last_pool = &p;
  return p;
}

}  // namespace detail

// THE scratch pool for a device. There is deliberately no no-argument spelling:
// "the pool" without a device is the defect (#516), not an ergonomic shortcut.
inline DevicePool& Pool(vt::Backend& b) {
  thread_local vt::Backend* last_backend = nullptr;
  thread_local DevicePool* last_pool = nullptr;
  return detail::MemoizedPool(detail::MainPoolTable(), b, last_backend, last_pool);
}

// --- Aux-stream scratch pool (ENG-MOE-SHARED-AUX) ----------------------------
// The MoE shared-expert overlap (MoeBlockFusedMarlinCuda) issues the shared MLP
// on a SECOND CUDA stream concurrent with the routed experts on the main stream.
// The main `Pool()` above is only reuse-safe under SINGLE-stream ordering: a
// block returned to the free list is handed back out on the same stream, so CUDA
// stream ordering guarantees the block's last op has completed before any reuse.
// Two streams sharing one pool BREAKS that invariant — a scratch block the aux
// stream freed (e.g. the shared gate/up transient) could be handed to a routed
// GEMM on the main stream and written WHILE the aux kernel still reads it (a
// cross-stream RAW/WAR race compute-sanitizer flags). vLLM sidesteps this with
// its STREAM-AWARE caching allocator (record_stream); we keep the simple pool and
// instead give the aux stream its OWN pool. Each pool then only ever serves ONE
// stream, so single-stream ordering holds within it and the two streams never
// share a live block. Blocks are handed back to the pool they came from (DBuf
// stores its owning pool), so a buffer allocated in the aux region and destroyed
// after the join still returns to the aux pool.
// The aux pool is per-device too: the stream distinction and the device
// distinction are independent, and a process with two devices running the MoE
// overlap needs one aux pool per device, not one shared between them.
#ifdef VT_MARLIN_NVFP4  // only the Marlin MoE overlap path draws from AuxPool
namespace detail {
inline PoolTable& AuxPoolTable() {
  static PoolTable t;
  return t;
}
}  // namespace detail
inline DevicePool& AuxPool(vt::Backend& b) {
  thread_local vt::Backend* last_backend = nullptr;
  thread_local DevicePool* last_pool = nullptr;
  return detail::MemoizedPool(detail::AuxPoolTable(), b, last_backend, last_pool);
}
#endif

// Thread-local OVERRIDE of the scratch pool a DBuf constructs from. Null means
// "this device's own Pool(b)" — the default, and the only correct default, since
// a thread-local cannot know which device the next DBuf will be built on. The
// aux-stream overlap region swaps it to AuxPool(b) for the duration of the
// shared-expert issue via ActivePoolScope. Single host thread drives the
// forward, and the aux ops are issued in one contiguous block, so the swap is a
// simple RAII stack.
//
// An override pointing at ANOTHER device's pool is no longer a silent
// corruption: the pool checks its backend on every operation and throws.
inline DevicePool*& ActivePoolOverride() {
  thread_local DevicePool* p = nullptr;
  return p;
}
inline DevicePool& ActivePool(vt::Backend& b) {
  DevicePool* const override_pool = ActivePoolOverride();
  return override_pool != nullptr ? *override_pool : Pool(b);
}
struct ActivePoolScope {
  DevicePool* prev;
  explicit ActivePoolScope(DevicePool* p) : prev(ActivePoolOverride()) {
    ActivePoolOverride() = p;
  }
  ~ActivePoolScope() { ActivePoolOverride() = prev; }
  ActivePoolScope(const ActivePoolScope&) = delete;
  ActivePoolScope& operator=(const ActivePoolScope&) = delete;
};

}  // namespace vllm
