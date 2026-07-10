// Ported from llama.cpp (local fork @ 237ad9b96) tests/test-barrier.cpp
// (barrier stress over repeated parallel ops — the reach-all-threads assertion
// shape, plus test_active's mixed-thread-count rounds re-expressed as mixed
// chunk counts) and tests/test-thread-safety.cpp (REDUCED: concurrent op
// submission from one process serialized by the pool; the full multi-context
// server case stays with SERVE-E2E-NIGHTLY — see the tracked SKIP below).
// The determinism A/B battery has no upstream analogue: it is spec gate 1 of
// .agents/specs/gguf-cpu-threadpool.md (byte-identical outputs across
// VLLM_CPP_CPU_THREADS 1/3/20; 3 = non-divisor thread count for boundary bugs).
#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::DType;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;
using vt::cpu::ParallelForRows;
using vt::cpu::Threadpool;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

// Deterministic LCG fill (same values every run/thread count).
void FillF32(std::vector<float>& v, uint32_t seed) {
  uint32_t s = seed * 2654435761u + 1u;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;  // [-1, 1)
  }
}

std::vector<uint8_t> Bytes(const void* p, size_t n) {
  std::vector<uint8_t> b(n);
  std::memcpy(b.data(), p, n);
  return b;
}

// Swap the kernels' dispatch pool for the scope of one battery run.
struct ScopedPool {
  explicit ScopedPool(Threadpool* tp) { prev = Threadpool::SwapForTesting(tp); }
  ~ScopedPool() { Threadpool::SwapForTesting(prev); }
  Threadpool* prev;
};

}  // namespace

// test-barrier.cpp test_barrier: lots of small parallel ops where the barriers
// (kick/park/chunk-steal) in between dominate; every thread must reach every
// round.
TEST_CASE("barrier stress: 1000 rounds of chunked parallel ops reach all threads") {
  constexpr int kThreads = 4;
  constexpr int kRounds = 1000;
  Threadpool pool(kThreads);
  REQUIRE(pool.NThreads() == kThreads);

  // One cache-padded slot per thread: no shared writes (determinism contract).
  std::vector<int64_t> rounds_seen(kThreads * 16, 0);
  std::vector<int64_t> chunks_done(kThreads * 16, 0);

  for (int r = 0; r < kRounds; ++r) {
    pool.Run([&](int ith, int nth) {
      REQUIRE(nth == kThreads);
      rounds_seen[static_cast<size_t>(ith) * 16] += 1;
      // Chunk-steal exercise with a mixed (per-round) chunk count.
      const int nchunk = nth + (r % 5);  // nth .. nth+4 chunks
      if (ith == 0) pool.ChunkSet(nth);
      pool.Barrier();
      int current = ith;
      while (current < nchunk) {
        chunks_done[static_cast<size_t>(ith) * 16] += 1;
        current = pool.ChunkAdd(1);
      }
    });
  }

  int64_t total_chunks = 0;
  for (int t = 0; t < kThreads; ++t) {
    CHECK(rounds_seen[static_cast<size_t>(t) * 16] == kRounds);  // reach-all-threads
    total_chunks += chunks_done[static_cast<size_t>(t) * 16];
  }
  // Every chunk of every round executed exactly once across the pool.
  int64_t expect = 0;
  for (int r = 0; r < kRounds; ++r) expect += kThreads + (r % 5);
  CHECK(total_chunks == expect);
}

TEST_CASE("ParallelForRows visits every row exactly once at mixed sizes") {
  for (int nth : {1, 3, 20}) {
    Threadpool pool(nth);
    for (int64_t nr : {int64_t{0}, int64_t{1}, int64_t{2}, int64_t{3}, int64_t{17},
                       int64_t{64}, int64_t{1000}, int64_t{4096}}) {
      std::vector<int> visits(static_cast<size_t>(nr), 0);
      ParallelForRows(pool, nr, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) visits[static_cast<size_t>(r)] += 1;
      });
      for (int64_t r = 0; r < nr; ++r) CHECK(visits[static_cast<size_t>(r)] == 1);
    }
  }
}

TEST_CASE("per-op epoch remains valid beyond the upstream signed 16-bit epoch") {
  // Upstream advances the packed epoch once per graph. Our adaptation advances
  // it once per op, so a signed 32-bit (epoch << 16) reaches UB after 32767
  // dispatches. Exercise that boundary explicitly; poll=0 keeps the stress
  // test bounded while retaining the real park/wake + barrier path.
  constexpr int kDispatches = 33000;
  Threadpool pool(2, 0);
  std::atomic<int64_t> visits{0};
  for (int i = 0; i < kDispatches; ++i) {
    pool.Run([&](int, int) { visits.fetch_add(1, std::memory_order_relaxed); });
  }
  CHECK(visits.load(std::memory_order_relaxed) == int64_t{kDispatches} * 2);
}

// test-thread-safety.cpp REDUCED: several host threads submit ops through the
// same pool concurrently; the pool serializes dispatch and every result is
// correct. (The full multi-model/multi-context server case is intentionally
// NOT ported here: it needs real model contexts and belongs to the
// SERVE-E2E-NIGHTLY suite — tracked skip per test-porting.md rule 6.)
TEST_CASE("concurrent op submission from multiple host threads is serialized") {
  constexpr int kSubmitters = 4;
  constexpr int kIters = 25;
  constexpr int64_t kM = 8, kK = 16, kN = 8;

  // Reference result (single-threaded submitter, whatever pool is global).
  std::vector<float> a(kM * kK), b(kK * kN), ref(kM * kN);
  FillF32(a, 7);
  FillF32(b, 11);
  {
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {kM, kK});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {kK, kN});
    Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, Cpu(), {kM, kN});
    Queue q = Q();
    vt::Matmul(q, to, ta, tb);
  }

  std::atomic<int> mismatches{0};
  std::vector<std::thread> submitters;
  submitters.reserve(kSubmitters);
  for (int s = 0; s < kSubmitters; ++s) {
    submitters.emplace_back([&]() {
      std::vector<float> out(kM * kN);
      Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {kM, kK});
      Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {kK, kN});
      Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {kM, kN});
      Queue q = Q();
      for (int i = 0; i < kIters; ++i) {
        std::fill(out.begin(), out.end(), -1.0f);
        vt::Matmul(q, to, ta, tb);
        if (std::memcmp(out.data(), ref.data(), out.size() * sizeof(float)) != 0) {
          mismatches.fetch_add(1);
        }
      }
    });
  }
  for (auto& t : submitters) t.join();
  CHECK(mismatches.load() == 0);
}

TEST_CASE("thread safety full multi-context server case is tracked nightly" *
          doctest::skip(true) *
          doctest::description("requires SERVE-E2E-NIGHTLY checkpoint provisioning")) {
  // Porting debt from llama.cpp tests/test-thread-safety.cpp:16-165. This leaf
  // ports the shared-pool concurrent-submit semantics above; the upstream
  // multi-model/multi-context real-server case needs checkpoint provisioning
  // and remains assigned to SERVE-E2E-NIGHTLY (test-porting.md rule 6).
}

TEST_CASE("thread count: constructor clamps; global pool respects the env contract") {
  Threadpool zero(0);
  CHECK(zero.NThreads() == 1);
  Threadpool big(1 << 20);
  CHECK(big.NThreads() == vt::cpu::kMaxThreads);
  // Global(): VLLM_CPP_CPU_THREADS if set, else hardware_concurrency, >= 1.
  CHECK(Threadpool::Global().NThreads() >= 1);
  if (const char* e = std::getenv("VLLM_CPP_CPU_THREADS")) {
    const int want = std::max(1, std::min(atoi(e), vt::cpu::kMaxThreads));
    CHECK(Threadpool::Global().NThreads() == want);
  }
}

TEST_CASE("worker exceptions rethrow on the submitting thread") {
  Threadpool pool(3);
  ScopedPool swap(&pool);
  // embedding id out of range fires VT_CHECK inside the parallel body.
  std::vector<float> table(4 * 2);
  FillF32(table, 3);
  std::vector<int32_t> ids = {0, 1, 99, 2, 3, 1, 0, 2};  // 99 out of range
  std::vector<float> out(ids.size() * 2);
  Tensor tt = Tensor::Contiguous(table.data(), DType::kF32, Cpu(), {4, 2});
  Tensor ti = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(),
                                 {static_cast<int64_t>(ids.size())});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(),
                                 {static_cast<int64_t>(ids.size()), 2});
  Queue q = Q();
  CHECK_THROWS_AS(vt::Embedding(q, to, tt, ti), std::runtime_error);
  // The pool stays usable after the exception.
  ids[2] = 3;
  vt::Embedding(q, to, tt, ti);
  CHECK(out[2 * 2] == table[3 * 2]);
}

TEST_CASE("nested parallel dispatch from a worker body fails loudly") {
  Threadpool pool(2);
  CHECK_THROWS_AS(
      pool.Run([&](int, int) {
        pool.Run([](int, int) {});  // would deadlock without the guard
      }),
      std::runtime_error);
}

// ---------------------------------------------------------------------------
// Determinism battery (spec gate 1): representative kernels produce
// byte-identical outputs at n_threads 1 / 3 / 20. Odd, non-multiple-of-16
// shapes stress chunk boundaries; 3 is a non-divisor thread count.
// ---------------------------------------------------------------------------

namespace {

// Runs the battery through the given pool and returns every output buffer
// (including in/out state tensors) as raw bytes.
std::vector<std::vector<uint8_t>> RunBattery(Threadpool& pool) {
  ScopedPool swap(&pool);
  Queue q = Q();
  std::vector<std::vector<uint8_t>> outs;

  // Matmul f32 [37,129]@[129,53] — small grid → per-thread re-chunk path.
  {
    std::vector<float> a(37 * 129), b(129 * 53), out(37 * 53);
    FillF32(a, 1);
    FillF32(b, 2);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {37, 129});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {129, 53});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {37, 53});
    vt::Matmul(q, to, ta, tb);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // Matmul [48,64]@[64,512] — nchunk0*nchunk1 = 32*3 = 96 >= 20*4 → chunk grid
  // + atomic stealing path; bf16 output exercises the store rounding.
  {
    std::vector<float> a(48 * 64), b(64 * 512);
    std::vector<uint16_t> out(48 * 512);
    FillF32(a, 3);
    FillF32(b, 4);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {48, 64});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {64, 512});
    Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {48, 512});
    vt::Matmul(q, to, ta, tb);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(uint16_t)));
  }
  // Decode-shaped Matmul [1,129]@[129,517] — nr1==1 → chunk_size 64 branch.
  {
    std::vector<float> a(1 * 129), b(129 * 517), out(1 * 517);
    FillF32(a, 5);
    FillF32(b, 6);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {1, 129});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {129, 517});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 517});
    vt::Matmul(q, to, ta, tb);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // MatmulBT [37,129] @ [53,129]^T.
  {
    std::vector<float> a(37 * 129), b(53 * 129), out(37 * 53);
    FillF32(a, 7);
    FillF32(b, 8);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {37, 129});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {53, 129});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {37, 53});
    vt::MatmulBT(q, to, ta, tb);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // RmsNorm with residual (gemma) [37,129] — residual stream is also an output.
  {
    std::vector<float> x(37 * 129), w(129), res(37 * 129), out(37 * 129);
    FillF32(x, 9);
    FillF32(w, 10);
    FillF32(res, 11);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {37, 129});
    Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {129});
    Tensor tr = Tensor::Contiguous(res.data(), DType::kF32, Cpu(), {37, 129});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {37, 129});
    vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{1e-6f, true}, &tr);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
    outs.push_back(Bytes(res.data(), res.size() * sizeof(float)));
  }
  // SiluAndMul [37,258].
  {
    std::vector<float> x(37 * 258), out(37 * 129);
    FillF32(x, 12);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {37, 258});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {37, 129});
    vt::SiluAndMul(q, to, tx);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // Attention causal GQA t=33, Hq=4, Hk=2, D=16.
  {
    std::vector<float> qq(33 * 4 * 16), kk(33 * 2 * 16), vv(33 * 2 * 16),
        out(33 * 4 * 16);
    FillF32(qq, 13);
    FillF32(kk, 14);
    FillF32(vv, 15);
    Tensor tq = Tensor::Contiguous(qq.data(), DType::kF32, Cpu(), {33, 4, 16});
    Tensor tk = Tensor::Contiguous(kk.data(), DType::kF32, Cpu(), {33, 2, 16});
    Tensor tv = Tensor::Contiguous(vv.data(), DType::kF32, Cpu(), {33, 2, 16});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {33, 4, 16});
    vt::Attention(q, to, tq, tk, tv, vt::AttentionArgs{0.25f, true});
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // GdnPrefill: 3 sequences (one empty), Hk=2, Hv=4, Dk=8, Dv=8 — state is an
  // in/out output too.
  {
    const int64_t total = 17;
    std::vector<float> qq(total * 2 * 8), kk(total * 2 * 8), vv(total * 4 * 8),
        g(total * 4), beta(total * 4), state(3 * 4 * 8 * 8, 0.0f),
        out(total * 4 * 8);
    FillF32(qq, 16);
    FillF32(kk, 17);
    FillF32(vv, 18);
    FillF32(g, 19);
    FillF32(beta, 20);
    std::vector<int32_t> qsl = {0, 5, 5, 17};
    Tensor tq = Tensor::Contiguous(qq.data(), DType::kF32, Cpu(), {total, 2, 8});
    Tensor tk = Tensor::Contiguous(kk.data(), DType::kF32, Cpu(), {total, 2, 8});
    Tensor tv = Tensor::Contiguous(vv.data(), DType::kF32, Cpu(), {total, 4, 8});
    Tensor tg = Tensor::Contiguous(g.data(), DType::kF32, Cpu(), {total, 4});
    Tensor tb = Tensor::Contiguous(beta.data(), DType::kF32, Cpu(), {total, 4});
    Tensor ts = Tensor::Contiguous(state.data(), DType::kF32, Cpu(), {3, 4, 8, 8});
    Tensor tqsl = Tensor::Contiguous(qsl.data(), DType::kI32, Cpu(), {4});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {total, 4, 8});
    vt::GdnArgs args;
    args.scale = 0.353553f;
    vt::GdnPrefill(q, to, tq, tk, tv, tg, tb, ts, tqsl, args);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
    outs.push_back(Bytes(state.data(), state.size() * sizeof(float)));
  }
  // MoeRouterTopK t=29, E=16, k=4 (renormalize).
  {
    std::vector<float> logits(29 * 16), weights(29 * 4);
    std::vector<int32_t> indices(29 * 4);
    FillF32(logits, 21);
    Tensor tl = Tensor::Contiguous(logits.data(), DType::kF32, Cpu(), {29, 16});
    Tensor tw = Tensor::Contiguous(weights.data(), DType::kF32, Cpu(), {29, 4});
    Tensor ti = Tensor::Contiguous(indices.data(), DType::kI32, Cpu(), {29, 4});
    vt::MoeRouterTopK(q, tw, ti, tl, vt::MoeRouterTopKArgs{4, true});
    outs.push_back(Bytes(weights.data(), weights.size() * sizeof(float)));
    outs.push_back(Bytes(indices.data(), indices.size() * sizeof(int32_t)));
  }
  // L2Norm [29,31].
  {
    std::vector<float> x(29 * 31), out(29 * 31);
    FillF32(x, 22);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {29, 31});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {29, 31});
    vt::L2Norm(q, to, tx, vt::L2NormArgs{1e-6f});
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // CausalConv1dFwd: 2 sequences, C=13, K=4 — conv_state is an output too.
  {
    const int64_t total = 11, c_dim = 13, k = 4;
    std::vector<float> x(total * c_dim), w(c_dim * k), bias(c_dim),
        state(2 * c_dim * (k - 1)), out(total * c_dim);
    FillF32(x, 23);
    FillF32(w, 24);
    FillF32(bias, 25);
    FillF32(state, 26);
    std::vector<int32_t> qsl = {0, 4, 11};
    std::vector<int32_t> his = {1, 0};
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {total, c_dim});
    Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {c_dim, k});
    Tensor tbias = Tensor::Contiguous(bias.data(), DType::kF32, Cpu(), {c_dim});
    Tensor ts = Tensor::Contiguous(state.data(), DType::kF32, Cpu(), {2, c_dim, k - 1});
    Tensor tqsl = Tensor::Contiguous(qsl.data(), DType::kI32, Cpu(), {3});
    Tensor this_ = Tensor::Contiguous(his.data(), DType::kI32, Cpu(), {2});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {total, c_dim});
    vt::CausalConv1dFwd(q, to, tx, tw, &tbias, ts, tqsl, this_,
                        vt::CausalConv1dArgs{true});
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
    outs.push_back(Bytes(state.data(), state.size() * sizeof(float)));
  }
  return outs;
}

}  // namespace

TEST_CASE("determinism: byte-identical outputs at n_threads 1 / 3 / 20") {
  Threadpool p1(1), p3(3), p20(20);
  auto ref = RunBattery(p1);
  auto got3 = RunBattery(p3);
  auto got20 = RunBattery(p20);
  REQUIRE(ref.size() == got3.size());
  REQUIRE(ref.size() == got20.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK_MESSAGE(ref[i] == got3[i], "buffer " << i << " differs at n_threads=3");
    CHECK_MESSAGE(ref[i] == got20[i], "buffer " << i << " differs at n_threads=20");
  }
}
