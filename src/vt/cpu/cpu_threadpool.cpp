// Ported from llama.cpp (local fork) ggml/src/ggml-cpu/ggml-cpu.c @ 237ad9b96
// (b9892). See cpu_threadpool.h for the full upstream anchor map and the
// recorded deviations. Line references below are into that file unless noted.
#include "cpu_threadpool.h"

#include <algorithm>
#include <cstdlib>

#include "vt/dtype.h"  // VT_CHECK

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// Match ggml's GGML_TSAN_ENABLED fence workaround. GCC and Clang both warn
// that a standalone atomic_thread_fence is not modeled by ThreadSanitizer;
// the upstream TSAN build uses a no-op seq-cst RMW on the same synchronization
// atomic instead (ggml-cpu.c:594-600,3122-3128).
#if defined(__SANITIZE_THREAD__)
#define VT_CPU_THREAD_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define VT_CPU_THREAD_SANITIZER 1
#endif
#endif
#ifndef VT_CPU_THREAD_SANITIZER
#define VT_CPU_THREAD_SANITIZER 0
#endif

namespace vt::cpu {
namespace {

// ggml_thread_cpu_relax, ggml-cpu.c:510-529 (arm yield / x86 _mm_pause /
// riscv pause / no-op fallback).
inline void CpuRelax() {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
  __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#else
  ;
#endif
}

// Thread count selection (spec § env contract): VLLM_CPP_CPU_THREADS, default
// std::thread::hardware_concurrency(); clamped to [1, kMaxThreads]
// (GGML_MAX_N_THREADS analogue).
int ThreadsFromEnv() {
  int n = 0;
  if (const char* e = std::getenv("VLLM_CPP_CPU_THREADS")) {
    n = std::atoi(e);
  }
  if (n <= 0) {
    n = static_cast<int>(std::thread::hardware_concurrency());
  }
  return std::clamp(n, 1, kMaxThreads);
}

std::atomic<Threadpool*> g_test_pool{nullptr};

// Loud-failure guard: a kernel body must never dispatch another parallel op
// from inside a parallel region (the non-recursive run mutex / parked-worker
// protocol would deadlock, as in ggml where a node body never calls
// ggml_graph_compute). Tracked per thread; checked at Run() entry.
thread_local bool t_in_parallel_region = false;

struct ParallelRegionScope {
  ParallelRegionScope() { t_in_parallel_region = true; }
  ~ParallelRegionScope() { t_in_parallel_region = false; }
};

}  // namespace

// ggml_threadpool_new_impl, ggml-cpu.c:3237-3308. Workers are allocated and
// initialized (:3261-3271), then threads 1..n-1 are spawned on the secondary
// loop (:3289-3294); worker 0 is the caller. Affinity/priority not ported
// (upstream defaults are inherit-affinity / normal priority, ggml.c:8136-8143).
Threadpool::Threadpool(int n_threads, uint32_t poll)
    : n_threads_(std::clamp(n_threads, 1, kMaxThreads)), poll_(poll) {
  workers_ = new ComputeState[static_cast<size_t>(n_threads_)];
  for (int j = 0; j < n_threads_; ++j) {
    workers_[j].threadpool = this;
    workers_[j].ith = j;
  }
  for (int j = 1; j < n_threads_; ++j) {
    workers_[j].thrd = std::thread([this, j] { SecondaryThread(workers_[j]); });
  }
}

// ggml_threadpool_free, ggml-cpu.c:2682-2711: set stop under the mutex,
// broadcast, join workers 1..n-1.
Threadpool::~Threadpool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_.store(true, std::memory_order_relaxed);
    pause_.store(false, std::memory_order_relaxed);
    cond_.notify_all();
  }
  for (int j = 1; j < n_threads_; ++j) {
    workers_[j].thrd.join();
  }
  delete[] workers_;
}

// ggml_barrier, ggml-cpu.c:566-602 (non-OpenMP branch): relaxed read of the
// passed-count, seq-cst fetch-add entry; the last thread resets n_barrier and
// bumps n_barrier_passed (seq-cst); spinners relax-wait then issue a full
// seq-cst fence on exit.
void Threadpool::Barrier() {
  const int n_threads = static_cast<int>(
      n_graph_.load(std::memory_order_relaxed) & kNThreadsMask);
  if (n_threads == 1) {
    return;
  }

  const int n_passed = n_barrier_passed_.load(std::memory_order_relaxed);

  // enter barrier (full seq-cst fence)
  const int n_barrier = n_barrier_.fetch_add(1, std::memory_order_seq_cst);

  if (n_barrier == n_threads - 1) {
    // last thread
    n_barrier_.store(0, std::memory_order_relaxed);
    // exit barrier (full seq-cst fence)
    n_barrier_passed_.fetch_add(1, std::memory_order_seq_cst);
    return;
  }

  // wait for other threads
  while (n_barrier_passed_.load(std::memory_order_relaxed) == n_passed) {
    CpuRelax();
  }

  // exit barrier (full seq-cst fence). TSAN does not model a standalone
  // fence, so mirror ggml's dummy seq-cst RMW in sanitizer builds.
#if VT_CPU_THREAD_SANITIZER
  n_barrier_passed_.fetch_add(0, std::memory_order_seq_cst);
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// ggml_graph_compute_thread, ggml-cpu.c:3024-3097, per-op: nth comes from the
// n_graph mask (:3039), the stored work fn replaces the per-node loop
// (:3052-3082), and the final barrier (:3090) synchronizes completion.
// Deviation: exceptions from the fn are captured (first wins) for rethrow on
// the Run() caller.
void Threadpool::ComputeThread(ComputeState& state) {
  const int nth = static_cast<int>(
      n_graph_.load(std::memory_order_relaxed) & kNThreadsMask);
  const std::function<void(int, int)>* fn = work_;
  if (fn != nullptr && state.ith < nth) {
    ParallelRegionScope scope;
    try {
      (*fn)(state.ith, nth);
    } catch (...) {
      if (!has_exc_.exchange(true)) {
        exc_ = std::current_exception();
      }
    }
  }
  Barrier();
}

// ggml_graph_compute_thread_ready, ggml-cpu.c:3103-3118: exit polling/sleep on
// pending work, stop, or pause; a new n_graph epoch marks this thread pending
// iff its ith is below the epoch's active-thread count.
bool Threadpool::ThreadReady(ComputeState& state) {
  if (state.pending || stop_.load(std::memory_order_relaxed) ||
      pause_.load(std::memory_order_relaxed)) {
    return true;
  }

  // check for new graph/work
  const uint64_t n_graph = n_graph_.load(std::memory_order_relaxed);
  const int n_threads = static_cast<int>(n_graph & kNThreadsMask);
  if (n_graph != state.last_graph) {
    state.pending = state.ith < n_threads;
    state.last_graph = n_graph;
    return true;
  }

  return false;
}

// ggml_graph_compute_thread_sync, ggml-cpu.c:3121-3129: full seq-cst fence
// after a polling exit (the relaxed epoch read needs it before touching work).
void Threadpool::ThreadSync() {
#if VT_CPU_THREAD_SANITIZER
  n_graph_.fetch_add(0, std::memory_order_seq_cst);
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// ggml_graph_compute_poll_for_work, ggml-cpu.c:3131-3144: 1024*128*poll relax
// rounds before falling back to the cond-var sleep.
bool Threadpool::PollForWork(ComputeState& state) {
  const uint64_t n_rounds = 1024UL * 128 * poll_;
  for (uint64_t i = 0; !ThreadReady(state) && i < n_rounds; ++i) {
    CpuRelax();
  }
  return state.pending;
}

// ggml_graph_compute_check_for_work, ggml-cpu.c:3146-3163: hybrid poll then
// cond-wait under the mutex.
bool Threadpool::CheckForWork(ComputeState& state) {
  if (PollForWork(state)) {
    ThreadSync();
    return state.pending;
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!ThreadReady(state)) {
      // No new work. Wait for the signal.
      cond_.wait(lock);
    }
  }
  return state.pending;
}

// ggml_graph_compute_secondary_thread, ggml-cpu.c:3165-3200 (priority/affinity
// application dropped — not ported).
void Threadpool::SecondaryThread(ComputeState& state) {
  while (true) {
    // Check if we need to sleep
    while (pause_.load(std::memory_order_relaxed)) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (pause_.load(std::memory_order_relaxed)) {
        cond_.wait(lock);
      }
    }

    // This needs to be checked for after the cond_wait
    if (stop_.load(std::memory_order_relaxed)) {
      break;
    }

    // Check if there is new work; the main thread is the only dispatcher.
    CheckForWork(state);
    if (state.pending) {
      state.pending = false;
      ComputeThread(state);
    }
  }
}

// ggml_graph_compute_kickoff, ggml-cpu.c:3202-3233: always take the mutex
// (workers do hybrid poll/wait), bump the epoch in the high bits and store the
// active-thread count in the mask (seq-cst store — the polling threads pair it
// with ThreadSync), then broadcast.
void Threadpool::Kickoff(int n_threads) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t n_graph = n_graph_.load(std::memory_order_relaxed) >> kNThreadsBits;
  n_graph = ((n_graph + 1) << kNThreadsBits) |
            (static_cast<uint64_t>(n_threads) & kNThreadsMask);

  n_graph_.store(n_graph, std::memory_order_seq_cst);

  if (pause_.load(std::memory_order_relaxed)) {
    // resume does cond broadcast (ggml_threadpool_resume_locked, :2721-2725)
    pause_.store(false, std::memory_order_relaxed);
  }
  cond_.notify_all();
}

// ggml_graph_compute, ggml-cpu.c:3314-3389 (persistent-pool branch), per-op:
// reset the chunk cursor (:3337), kick (:3373), caller computes as worker 0
// (:3376). The final barrier inside ComputeThread makes the return of this
// function the completion point for every output element.
void Threadpool::Run(const std::function<void(int, int)>& fn) {
  VT_CHECK(!t_in_parallel_region,
           "cpu_threadpool: nested parallel dispatch from inside a parallel region");
  // Deviation: serialize concurrent submitters (ggml assumes one dispatcher).
  std::lock_guard<std::mutex> run_lock(run_mutex_);

  work_ = &fn;
  current_chunk_.store(0, std::memory_order_relaxed);
  has_exc_.store(false, std::memory_order_relaxed);
  exc_ = nullptr;

  if (n_threads_ == 1) {
    // spec § Dispatch behavior: n_threads==1 short-circuits to the current
    // (inline) code path; keep the epoch mask coherent for Barrier().
    const uint64_t epoch =
        n_graph_.load(std::memory_order_relaxed) >> kNThreadsBits;
    n_graph_.store(((epoch + 1) << kNThreadsBits) | 1u,
                   std::memory_order_relaxed);
    {
      ParallelRegionScope scope;
      fn(0, 1);
    }
    work_ = nullptr;
    return;
  }

  Kickoff(n_threads_);

  // This is a work thread too (worker 0 = caller).
  ComputeThread(workers_[0]);

  work_ = nullptr;
  if (has_exc_.load(std::memory_order_relaxed)) {
    has_exc_.store(false, std::memory_order_relaxed);
    std::exception_ptr e = exc_;
    exc_ = nullptr;
    std::rethrow_exception(e);
  }
}

Threadpool& Threadpool::Global() {
  static Threadpool pool(ThreadsFromEnv());
  return pool;
}

Threadpool* Threadpool::SwapForTesting(Threadpool* tp) {
  return g_test_pool.exchange(tp);
}

Threadpool& CurrentThreadpool() {
  Threadpool* tp = g_test_pool.load(std::memory_order_acquire);
  return tp != nullptr ? *tp : Threadpool::Global();
}

// Flash-attn row chunking, ggml-cpu/ops.cpp:9070-9126: 4x chunks per thread
// (:9078-9081), per-thread re-chunk when the grid is smaller than nth or on
// NUMA (:9083-9085, IsNuma() stubbed), thread 0 seeds the steal cursor at nth
// then a barrier publishes it (:9087-9091), and each thread walks chunk ith
// first then steals via the atomic cursor (:9109-9122).
void ParallelForRows(Threadpool& tp, int64_t nr,
                     const std::function<void(int64_t, int64_t)>& body) {
  if (nr <= 0) {
    return;
  }
  // Min-work fallback (spec § Risks/decisions: "n_chunks==1 → run inline"):
  // a single output row has no partitionable work — the 4x-oversubscribed
  // grid degenerates to one chunk — so run it inline on the caller (the same
  // body over the same [0,1) range: bit-identical by construction, no kick).
  // n_threads==1 likewise short-circuits to the current single-thread code.
  if (nr == 1 || tp.NThreads() == 1) {
    body(0, nr);
    return;
  }
  tp.Run([&tp, nr, &body](int ith, int nth) {
    // 4x chunks per thread
    const int nth_scaled = nth * 4;
    const int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk = (nr + chunk_size - 1) / chunk_size;

    if (nth == 1 || nchunk < nth || IsNuma()) {
      nchunk = nth;
    }

    if (ith == 0) {
      tp.ChunkSet(nth);
    }

    tp.Barrier();

    const int64_t dr = (nr + nchunk - 1) / nchunk;

    int64_t current_chunk = ith;

    while (current_chunk < nchunk) {
      const int64_t ir0 = dr * current_chunk;
      const int64_t ir1 = std::min(ir0 + dr, nr);

      if (ir0 < ir1) {
        body(ir0, ir1);
      }

      current_chunk = tp.ChunkAdd(1);
    }
  });
}

}  // namespace vt::cpu
