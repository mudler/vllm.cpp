// vllm.cpp original (vt runtime). Unit tests for vt::DFlashBlockAttention — the
// DFlash draft's in-block attention (SPEC-DFLASH D2, DF-DRAFT-MODEL), the
// project's FIRST non-causal / bidirectional attention primitive. Semantics ref:
// DFlashQwen3Attention + _resolve_layer_attention (qwen3_dflash.py:86-146,
// 149-263 @ 555967922). These pin hand-computed values for the load-bearing
// corners: BIDIRECTIONAL (non-causal) full attention, causal-within-window SWA,
// per-request BLOCK isolation (cu_seqlens), GQA mapping, and — the RED proof —
// that causal != non-causal so a wrong mask is CAUGHT.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DFlashBlockAttentionArgs;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}
Tensor F32(std::vector<float>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kF32, Cpu(), shape);
}
DFlashBlockAttentionArgs Args(const int32_t* cu, int num_reqs, bool causal, int64_t window) {
  DFlashBlockAttentionArgs a;
  a.scale = 1.0f;
  a.causal = causal;
  a.sliding_window = window;
  a.cu_seqlens = cu;
  a.num_reqs = num_reqs;
  return a;
}
}  // namespace

TEST_CASE("dflash-block-attn NON-CAUSAL: query 0 attends to the FUTURE key (bidirectional)") {
  // T=2, one head, head_dim=2, scale=1, ONE block [0,2), NON-causal (full layer).
  //   q = [[1,0],[0,1]], k = [[1,0],[0,1]], v = [[1,2],[3,4]]
  // query 0 (NON-causal → BOTH keys): scores {q0·k0, q0·k1} = {1,0};
  //   softmax({1,0}) = {e/(1+e), 1/(1+e)} = {0.73106, 0.26894};
  //   out0 = 0.73106*[1,2] + 0.26894*[3,4] = [1.53789, 2.53789].
  // (The causal op would give out0 = v0 = [1,2] — this is the RED-separating value.)
  std::vector<float> q = {1, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {1, 2, 3, 4};
  std::vector<float> out(4, 0.0f);
  Tensor tq = F32(q, {2, 1, 2}), tk = F32(k, {2, 1, 2}), tv = F32(v, {2, 1, 2});
  Tensor to = F32(out, {2, 1, 2});
  Queue qq = Q();
  const int32_t cu[] = {0, 2};
  vt::DFlashBlockAttention(qq, to, tq, tk, tv, Args(cu, 1, /*causal=*/false, 0));
  CHECK(out[0] == doctest::Approx(1.53789f).epsilon(1e-4));
  CHECK(out[1] == doctest::Approx(2.53789f).epsilon(1e-4));
}

TEST_CASE("dflash-block-attn RED: causal vs non-causal DIFFER for query 0 (mask is load-bearing)") {
  // The load-bearing invariant: a full-attention layer wrongly run CAUSAL diverges.
  std::vector<float> q = {1, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {1, 2, 3, 4};
  std::vector<float> out_nc(4, 0.0f), out_c(4, 0.0f);
  Tensor tq = F32(q, {2, 1, 2}), tk = F32(k, {2, 1, 2}), tv = F32(v, {2, 1, 2});
  Queue qq = Q();
  const int32_t cu[] = {0, 2};
  Tensor tnc = F32(out_nc, {2, 1, 2});
  Tensor tc = F32(out_c, {2, 1, 2});
  vt::DFlashBlockAttention(qq, tnc, tq, tk, tv, Args(cu, 1, /*causal=*/false, 0));
  vt::DFlashBlockAttention(qq, tc, tq, tk, tv, Args(cu, 1, /*causal=*/true, 0));
  // Non-causal query 0 sees the future key; causal query 0 sees only key 0 (=v0).
  CHECK(out_c[0] == doctest::Approx(1.0f));
  CHECK(out_c[1] == doctest::Approx(2.0f));
  CHECK(std::fabs(out_nc[0] - out_c[0]) > 0.4f);  // 1.538 vs 1.0 — caught
}

TEST_CASE("dflash-block-attn per-request BLOCK isolation (cu_seqlens)") {
  // Two blocks of 1 token each: [0,1) and [1,2). Even non-causal, block 0's query
  // must NOT see block 1's key (and vice versa) — each is its own softmax of 1 key,
  // so out == v of that row regardless of the other block.
  std::vector<float> q = {1, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {7, 8, 100, 200};
  std::vector<float> out(4, 0.0f);
  Tensor tq = F32(q, {2, 1, 2}), tk = F32(k, {2, 1, 2}), tv = F32(v, {2, 1, 2});
  Tensor to = F32(out, {2, 1, 2});
  Queue qq = Q();
  const int32_t cu[] = {0, 1, 2};  // two singleton blocks
  vt::DFlashBlockAttention(qq, to, tq, tk, tv, Args(cu, 2, /*causal=*/false, 0));
  CHECK(out[0] == doctest::Approx(7.0f));    // block 0 == v[0], not pulled by v[1]
  CHECK(out[1] == doctest::Approx(8.0f));
  CHECK(out[2] == doctest::Approx(100.0f));  // block 1 == v[1]
  CHECK(out[3] == doctest::Approx(200.0f));
}

TEST_CASE("dflash-block-attn SWA window bounds the causal key range") {
  // One block [0,3), causal, window=2: query 2 sees keys {1,2} only (not key 0).
  // q2=[0,0,1] picks key with e2 component. Make v distinctive per key.
  //   q = rows e0,e1,e2 (D=3); k = e0,e1,e2; scores are the identity → query i
  //   attends most to key i. window=2 for query 2 → keys {1,2}; key 0 excluded.
  std::vector<float> q = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<float> v = {10, 0, 0, 0, 20, 0, 0, 0, 30};  // v0,v1,v2 distinct axes
  std::vector<float> out_w(9, 0.0f), out_full(9, 0.0f);
  Tensor tq = F32(q, {3, 1, 3}), tk = F32(k, {3, 1, 3}), tv = F32(v, {3, 1, 3});
  Queue qq = Q();
  const int32_t cu[] = {0, 3};
  Tensor tw = F32(out_w, {3, 1, 3});
  Tensor tf = F32(out_full, {3, 1, 3});
  vt::DFlashBlockAttention(qq, tw, tq, tk, tv, Args(cu, 1, /*causal=*/true, /*window=*/2));
  vt::DFlashBlockAttention(qq, tf, tq, tk, tv, Args(cu, 1, /*causal=*/true, /*window=*/0));
  // query 2 row (out[6..8]): window=2 excludes key 0 (v0 on axis 0), so out_w[6]
  // (axis-0 component) must be strictly smaller than the full-causal case which
  // DOES mix in key 0's v0=10.
  CHECK(out_w[6] < out_full[6] - 1e-3f);
}

TEST_CASE("dflash-block-attn GQA: 2 q-heads share 1 kv-head") {
  // Hq=2, Hk=1, D=1, one block [0,1) (single token). Each q-head reads kv-head 0.
  std::vector<float> q = {1, 1};      // [T=1, Hq=2, D=1]
  std::vector<float> k = {2};         // [1,1,1]
  std::vector<float> v = {5};         // [1,1,1]
  std::vector<float> out(2, 0.0f);
  Tensor tq = F32(q, {1, 2, 1}), tk = F32(k, {1, 1, 1}), tv = F32(v, {1, 1, 1});
  Tensor to = F32(out, {1, 2, 1});
  Queue qq = Q();
  const int32_t cu[] = {0, 1};
  vt::DFlashBlockAttention(qq, to, tq, tk, tv, Args(cu, 1, false, 0));
  CHECK(out[0] == doctest::Approx(5.0f));  // single key → out == v
  CHECK(out[1] == doctest::Approx(5.0f));
}

// ===========================================================================
// CUDA parity (SPEC-DFLASH D2 GPU promotion gate 2): the CUDA
// DFlashBlockAttentionKernel must match the authoritative CPU reference within
// the f32-online-softmax envelope on random inputs, across ALL 5 semantic
// corners the CPU cases above pin — non-causal (full/bidirectional), plain
// causal, per-request BLOCK isolation (multi-block cu_seqlens), SWA window, and
// GQA. The CPU kernel is a two-pass max-subtracted softmax; the CUDA kernel is a
// flash-style online-softmax recurrence, so they agree to f32 rounding (1e-4
// relative), not bit-for-bit — the stated envelope (mirrors test_ops_attention's
// CUDA==CPU gate). Guarded by HasCuda so CPU-only builds skip cleanly.
namespace {

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = Contig(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

std::vector<float> RandF32(size_t n, uint32_t seed) {
  // Deterministic LCG in [-2,2); avoids <random> divergence across libstdc++.
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

// Run one config on BOTH CPU and CUDA over random f32 inputs and assert parity.
void RunCudaParity(int64_t T, int64_t Hq, int64_t Hk, int64_t D, float scale, bool causal,
                   int64_t window, const std::vector<int32_t>& cu, uint32_t seed) {
  auto q = RandF32(static_cast<size_t>(T * Hq * D), seed);
  auto k = RandF32(static_cast<size_t>(T * Hk * D), seed + 1);
  auto v = RandF32(static_cast<size_t>(T * Hk * D), seed + 2);
  const int num_reqs = static_cast<int>(cu.size()) - 1;
  auto mkargs = [&]() {
    DFlashBlockAttentionArgs a = Args(cu.data(), num_reqs, causal, window);
    a.scale = scale;
    return a;
  };

  // CPU reference.
  std::vector<float> cpu(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor cq = Contig(q.data(), DType::kF32, Cpu(), {T, Hq, D});
  Tensor ck = Contig(k.data(), DType::kF32, Cpu(), {T, Hk, D});
  Tensor cv = Contig(v.data(), DType::kF32, Cpu(), {T, Hk, D});
  Tensor co = Contig(cpu.data(), DType::kF32, Cpu(), {T, Hq, D});
  Queue cpuq = Q();
  vt::DFlashBlockAttention(cpuq, co, cq, ck, cv, mkargs());

  // CUDA.
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {T, Hq, D}, q.data());
  DeviceTensor dk(gpu, g.q, DType::kF32, {T, Hk, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kF32, {T, Hk, D}, v.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {T, Hq, D});
  // cu_seqlens is a HOST pointer (the launcher uploads it stream-ordered).
  vt::DFlashBlockAttention(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), mkargs());
  std::vector<float> got(static_cast<size_t>(T * Hq * D), 0.0f);
  dout.Download(g.q, got.data());

  for (size_t i = 0; i < cpu.size(); ++i)
    CHECK(got[i] == doctest::Approx(cpu[i]).epsilon(1e-4));
}

}  // namespace

TEST_CASE("dflash-block-attn CUDA matches CPU across the 5 semantic corners") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping CUDA dflash-block-attn parity");
    return;
  }
  const float sc = std::pow(128.0f, -0.5f);
  // (1) NON-CAUSAL full attention, GQA, real head_dim, one 17-token block (1+k).
  RunCudaParity(/*T=*/17, /*Hq=*/32, /*Hk=*/8, /*D=*/128, sc, /*causal=*/false,
                /*window=*/0, /*cu=*/{0, 17}, /*seed=*/1234);
  // (2) plain CAUSAL (SWA layer, window >> block so it degenerates to causal).
  RunCudaParity(17, 32, 8, 128, sc, /*causal=*/true, /*window=*/2048, {0, 17}, 2222);
  // (3) per-request BLOCK isolation: 3 blocks of 17 (uniform DFlash batch).
  RunCudaParity(51, 16, 4, 64, 0.25f, /*causal=*/false, 0, {0, 17, 34, 51}, 3333);
  // (4) SWA window strictly bounds the causal key range (window=4 < block).
  RunCudaParity(17, 8, 2, 32, 0.3f, /*causal=*/true, /*window=*/4, {0, 17}, 4444);
  // (5) GQA extreme (8 q-heads share 1 kv-head) + ragged multi-block causal.
  RunCudaParity(20, 8, 1, 16, 0.35f, /*causal=*/true, /*window=*/2048, {0, 6, 20}, 5555);
  // (6) head_dim 96 -- MiniMax-H3's PRODUCTION shape (hidden 5376 / 56 heads), and
  // the only head_dim that is a whole number of warp widths but NOT a power of two.
  // Nothing in this file covered it before, so the CUDA fast path's head_dim/32 == 3
  // instantiation shipped UNEXERCISED while the suite reported green -- and it is
  // precisely the instantiation whose per-lane element partition differs from the
  // 64/128 ones (strided rather than contiguous-vector, to keep loads coalesced).
  RunCudaParity(17, 8, 2, 96, std::pow(96.0f, -0.5f), /*causal=*/false, 0, {0, 17}, 6666);
  RunCudaParity(20, 8, 2, 96, std::pow(96.0f, -0.5f), /*causal=*/true, /*window=*/6,
                {0, 6, 20}, 7777);
}

// The LONG non-causal single-document case, which is the only shape that reaches
// the shared-memory tiled CUDA kernel (guarded to !causal, num_reqs == 1, no
// window, seq >= 2048). Every other case in this file is far shorter, so without
// this the tiled path ships UNEXERCISED while the suite reports green -- the exact
// failure mode where a gate proves something other than what it appears to.
//
// Gated against the CPU reference over identical inputs. The tiled kernel keeps
// the same key order and the same online-softmax recurrence as the untiled one, so
// the bar is tight rather than merely "close".
// H3's REAL packed shape: cu_seqlens {0, used, seq_len} -- content plus a padding
// tail, i.e. TWO documents, not one. The first version of the tiled kernel was
// guarded to num_reqs == 1 and therefore never ran on the very workload it was
// written for, while the suite stayed green. This covers both branches: blocks
// wholly inside one request take the shared-tile path, and the block straddling
// the boundary takes the per-warp fallback.
TEST_CASE("dflash-block-attn LONG multi-request matches the reference (H3 packed shape)") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  const int64_t T = 3000, H = 2, D = 64;
  const int32_t used = 2317;  // deliberately NOT a multiple of the 8-warp block
  std::vector<float> q(static_cast<size_t>(T * H * D));
  std::vector<float> k(q.size()), v(q.size());
  uint64_t x = 0xD1B54A32D192ED03ULL;
  auto rnd = [&]() {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return static_cast<float>((x >> 40) / 16777216.0 - 0.5);
  };
  for (size_t i = 0; i < q.size(); ++i) { q[i] = rnd(); k[i] = rnd(); v[i] = rnd(); }
  const int32_t cu[3] = {0, used, static_cast<int32_t>(T)};

  std::vector<float> want(q.size(), 0.0f);
  {
    Queue cq = Q();
    Tensor qt = F32(q, {T, H, D}), kt = F32(k, {T, H, D}), vt_ = F32(v, {T, H, D});
    Tensor ot = F32(want, {T, H, D});
    vt::DFlashBlockAttention(cq, ot, qt, kt, vt_, Args(cu, 2, /*causal=*/false, 0));
  }

  Queue gq = cuda->CreateQueue();
  auto up = [&](const std::vector<float>& hv) {
    void* p = cuda->Alloc(hv.size() * sizeof(float));
    cuda->Copy(gq, p, hv.data(), hv.size() * sizeof(float));
    return p;
  };
  void* dq = up(q); void* dk = up(k); void* dv = up(v);
  void* dout = cuda->Alloc(q.size() * sizeof(float));
  Device gd = gq.device;
  Tensor gqt = Contig(dq, DType::kF32, gd, {T, H, D});
  Tensor gkt = Contig(dk, DType::kF32, gd, {T, H, D});
  Tensor gvt = Contig(dv, DType::kF32, gd, {T, H, D});
  Tensor got = Contig(dout, DType::kF32, gd, {T, H, D});
  vt::DFlashBlockAttention(gq, got, gqt, gkt, gvt, Args(cu, 2, /*causal=*/false, 0));
  cuda->Synchronize(gq);
  std::vector<float> got_host(q.size(), 0.0f);
  cuda->Copy(gq, got_host.data(), dout, got_host.size() * sizeof(float));
  cuda->Synchronize(gq);

  double worst = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(std::isfinite(got_host[i]));
    worst = std::max(worst, std::abs(static_cast<double>(got_host[i]) - want[i]));
  }
  INFO("tiled CUDA multi-request vs CPU reference, max|diff| = " << worst);
  CHECK(worst <= 2e-5);
  cuda->Free(dq); cuda->Free(dk); cuda->Free(dv); cuda->Free(dout);
}

TEST_CASE("dflash-block-attn LONG non-causal matches the reference (tiled CUDA path)") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }

  const int64_t T = 2560, H = 2, D = 64;  // T >= 2048 crosses the tiled threshold
  std::vector<float> q(static_cast<size_t>(T * H * D));
  std::vector<float> k(q.size()), v(q.size());
  uint64_t x = 0x9E3779B97F4A7C15ULL;
  auto rnd = [&]() {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return static_cast<float>((x >> 40) / 16777216.0 - 0.5);
  };
  for (size_t i = 0; i < q.size(); ++i) { q[i] = rnd(); k[i] = rnd(); v[i] = rnd(); }

  const int32_t cu[2] = {0, static_cast<int32_t>(T)};
  std::vector<float> want(q.size(), 0.0f);
  {
    Queue cq = Q();
    Tensor qt = F32(q, {T, H, D}), kt = F32(k, {T, H, D}), vt_ = F32(v, {T, H, D});
    Tensor ot = F32(want, {T, H, D});
    vt::DFlashBlockAttention(cq, ot, qt, kt, vt_, Args(cu, 1, /*causal=*/false, 0));
  }

  Queue gq = cuda->CreateQueue();
  auto up = [&](const std::vector<float>& hostv) {
    void* p = cuda->Alloc(hostv.size() * sizeof(float));
    cuda->Copy(gq, p, hostv.data(), hostv.size() * sizeof(float));
    return p;
  };
  void* dq = up(q);
  void* dk = up(k);
  void* dv = up(v);
  void* dout = cuda->Alloc(q.size() * sizeof(float));
  Device gdev = gq.device;
  Tensor gqt = Contig(dq, DType::kF32, gdev, {T, H, D});
  Tensor gkt = Contig(dk, DType::kF32, gdev, {T, H, D});
  Tensor gvt = Contig(dv, DType::kF32, gdev, {T, H, D});
  Tensor got = Contig(dout, DType::kF32, gdev, {T, H, D});
  vt::DFlashBlockAttention(gq, got, gqt, gkt, gvt, Args(cu, 1, /*causal=*/false, 0));
  cuda->Synchronize(gq);

  std::vector<float> got_host(q.size(), 0.0f);
  cuda->Copy(gq, got_host.data(), dout, got_host.size() * sizeof(float));
  cuda->Synchronize(gq);

  double worst = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(std::isfinite(got_host[i]));
    worst = std::max(worst, std::abs(static_cast<double>(got_host[i]) - want[i]));
  }
  INFO("tiled CUDA vs CPU reference over " << T << " keys, max|diff| = " << worst);
  CHECK(worst <= 2e-5);

  cuda->Free(dq); cuda->Free(dk); cuda->Free(dv); cuda->Free(dout);
}

namespace {

// LONG CUDA-vs-CPU parity over an arbitrary mask.
//
// Both LONG cases above are NON-CAUSAL SINGLE-DOCUMENT, which is the easiest mask
// there is: every query sees exactly the same key range, so any kernel that gets
// the range right ONCE is right for all of them. Nothing in this file previously
// combined a long sequence with a mask that VARIES per query -- causal (jhi moves),
// sliding-window (jlo moves too) or ragged multi-request (queries in one warp
// belong to different documents). Those are where per-query bookkeeping and warp
// scheduling can disagree, and where a long-sequence kernel is most likely to be
// wrong in a way the short cases cannot see.
//
// These were added while evaluating the Q-blocked kernel (DFlashAttnQBlockKernel,
// which walks the UNION of a warp's key ranges and skips per query -- exactly the
// logic a non-causal single document cannot exercise). That kernel measured
// NEGATIVE and is not dispatched, but the gaps these cases close are properties of
// the OP, not of that experiment, so they stay.
//
// cu_seqlens must span [0,T] (vt::DFlashBlockAttention's precondition), so the
// documents always tile the whole tensor; raggedness comes from the boundaries
// sitting off any warp multiple.
void RunLongParity(const char* what, int64_t T, int64_t H, int64_t D, float scale, bool causal,
                   int64_t window, const std::vector<int32_t>& cu, uint64_t seed) {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  const int num_reqs = static_cast<int>(cu.size()) - 1;
  std::vector<float> q(static_cast<size_t>(T * H * D));
  std::vector<float> k(q.size()), v(q.size());
  uint64_t x = seed;
  auto rnd = [&]() {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return static_cast<float>((x >> 40) / 16777216.0 - 0.5);
  };
  for (size_t i = 0; i < q.size(); ++i) { q[i] = rnd(); k[i] = rnd(); v[i] = rnd(); }

  auto mk = [&]() {
    DFlashBlockAttentionArgs a = Args(cu.data(), num_reqs, causal, window);
    a.scale = scale;
    return a;
  };

  std::vector<float> want(q.size(), 0.0f);
  {
    Queue cq = Q();
    Tensor qt = F32(q, {T, H, D}), kt = F32(k, {T, H, D}), vt_ = F32(v, {T, H, D});
    Tensor ot = F32(want, {T, H, D});
    vt::DFlashBlockAttention(cq, ot, qt, kt, vt_, mk());
  }

  Queue gq = cuda->CreateQueue();
  auto up = [&](const std::vector<float>& hv) {
    void* p = cuda->Alloc(hv.size() * sizeof(float));
    cuda->Copy(gq, p, hv.data(), hv.size() * sizeof(float));
    return p;
  };
  void* dq = up(q); void* dk = up(k); void* dv = up(v);
  void* dout = cuda->Alloc(q.size() * sizeof(float));
  Device gd = gq.device;
  Tensor gqt = Contig(dq, DType::kF32, gd, {T, H, D});
  Tensor gkt = Contig(dk, DType::kF32, gd, {T, H, D});
  Tensor gvt = Contig(dv, DType::kF32, gd, {T, H, D});
  Tensor got = Contig(dout, DType::kF32, gd, {T, H, D});
  vt::DFlashBlockAttention(gq, got, gqt, gkt, gvt, mk());
  cuda->Synchronize(gq);
  std::vector<float> got_host(q.size(), 0.0f);
  cuda->Copy(gq, got_host.data(), dout, got_host.size() * sizeof(float));
  cuda->Synchronize(gq);

  double worst = 0.0;
  size_t worst_at = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(std::isfinite(got_host[i]));
    const double dif = std::abs(static_cast<double>(got_host[i]) - want[i]);
    if (dif > worst) { worst = dif; worst_at = i; }
  }
  INFO(what << ": long CUDA vs CPU, T=" << T << " D=" << D << " causal=" << causal
            << " window=" << window << " max|diff|=" << worst << " at " << worst_at);
  CHECK(worst <= 2e-5);
  cuda->Free(dq); cuda->Free(dk); cuda->Free(dv); cuda->Free(dout);
}

}  // namespace

TEST_CASE("dflash-block-attn LONG CAUSAL matches the reference") {
  // Plain causal over one long document: adjacent queries have DIFFERENT jhi, the
  // simplest mask that varies per query, at a length nothing else here reaches.
  RunLongParity("long causal", /*T=*/2560, /*H=*/2, /*D=*/64, 0.125f, /*causal=*/true,
                /*window=*/0, /*cu=*/{0, 2560}, 0x243F6A8885A308D3ULL);
}

TEST_CASE("dflash-block-attn LONG causal SLIDING WINDOW matches the reference") {
  // window=48 moves jlo as well as jhi, so the visible range slides rather than
  // grows. D=128 also exercises the kPerLane=4 instantiation.
  RunLongParity("long SWA", /*T=*/2048, /*H=*/2, /*D=*/128, 0.088388f, /*causal=*/true,
                /*window=*/48, /*cu=*/{0, 2048}, 0x13198A2E03707344ULL);
}

TEST_CASE("dflash-block-attn LONG head_dim 96 matches the reference (H3 production shape)") {
  // The real canvas runs head_dim 96 over thousands of keys. The short case above
  // proves the mask bookkeeping; this one proves the long-sequence CHUNKING at the
  // head_dim the model actually uses, in both masks.
  RunLongParity("long d96 non-causal", /*T=*/2560, /*H=*/2, /*D=*/96, 0.102062f,
                /*causal=*/false, /*window=*/0, /*cu=*/{0, 2560}, 0xBE5466CF34E90C6CULL);
  RunLongParity("long d96 ragged causal", /*T=*/2185, /*H=*/2, /*D=*/96, 0.102062f,
                /*causal=*/true, /*window=*/0, /*cu=*/{0, 501, 1503, 2185},
                0xC0AC29B7C97C50DDULL);
}

TEST_CASE("dflash-block-attn LONG ragged multi-request CAUSAL matches the reference") {
  // Three documents whose boundaries (501, 1503) land off every warp multiple, so
  // warps straddle document boundaries under both masks.
  RunLongParity("long ragged causal", /*T=*/2185, /*H=*/2, /*D=*/64, 0.125f, /*causal=*/true,
                /*window=*/0, /*cu=*/{0, 501, 1503, 2185}, 0xA4093822299F31D0ULL);
  RunLongParity("long ragged non-causal", /*T=*/2185, /*H=*/2, /*D=*/64, 0.125f,
                /*causal=*/false, /*window=*/0, /*cu=*/{0, 501, 1503, 2185},
                0x082EFA98EC4E6C89ULL);
}

// ---------------------------------------------------------------------------
// bf16 CUDA parity — the TENSOR-CORE (mma.sync) path.
//
// Every CUDA case above uploads f32, so all of them land on the CUDA-core kernels
// and NOT ONE of them would touch the bf16 tensor-core kernel. That is exactly the
// failure mode this file has been bitten by before (a long-sequence kernel guarded
// to num_reqs == 1 shipped never-executed while the suite stayed green), so the
// bf16 path gets its own coverage of the same semantic corners.
//
// THE BAR. Q, K and V are bf16 on both sides here — the reference runs on the very
// same rounded values — so the inputs contribute NOTHING. What differs is that the
// tensor-core kernel rounds the softmax probabilities P to bf16 before the P·V
// GEMM (as FlashAttention does), which is a bounded ~2^-9 relative perturbation of
// a convex combination of the V rows. With |v| < 0.5 here that is ~1e-3 absolute,
// so 5e-3 is the honest bf16 bound — the same one test_minimax_h3 uses for its
// bf16 stream — and NOT a loosened f32 bound: the f32 cases above still gate at
// 2e-5 and they still run the CUDA-core kernel.
//
// A tolerance alone cannot prove the kernel is right, because a wrong mask also
// produces "small" numbers when the rows are similar. So each case ALSO checks the
// RMS error against a much tighter statistical bound (random sign errors cancel;
// a structurally wrong kernel does not), and the mask cases are separated from
// their opposite mask by a margin far larger than 5e-3.
namespace {

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> b(f.size());
  for (size_t i = 0; i < f.size(); ++i) b[i] = vt::F32ToBF16(f[i]);
  return b;
}
std::vector<float> FromBf16(const std::vector<uint16_t>& b) {
  std::vector<float> f(b.size());
  for (size_t i = 0; i < b.size(); ++i) f[i] = vt::BF16ToF32(b[i]);
  return f;
}

// Runs one config with bf16 Q/K/V on CUDA against the f32 CPU reference over the
// SAME (bf16-rounded) values. `bf16_out` also rounds the result, which is the
// production stream's shape.
void RunBf16Parity(const char* what, int64_t T, int64_t Hq, int64_t Hk, int64_t D, float scale,
                   bool causal, int64_t window, const std::vector<int32_t>& cu, uint64_t seed,
                   bool bf16_out, double tol, double rms_tol) {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  const int num_reqs = static_cast<int>(cu.size()) - 1;
  std::vector<float> q(static_cast<size_t>(T * Hq * D));
  std::vector<float> k(static_cast<size_t>(T * Hk * D)), v(k.size());
  uint64_t x = seed;
  auto rnd = [&]() {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return static_cast<float>((x >> 40) / 16777216.0 - 0.5);
  };
  for (size_t i = 0; i < q.size(); ++i) q[i] = rnd();
  for (size_t i = 0; i < k.size(); ++i) { k[i] = rnd(); v[i] = rnd(); }

  // Round ONCE; both sides then see identical numbers.
  const std::vector<uint16_t> qb = ToBf16(q), kb = ToBf16(k), vb = ToBf16(v);
  std::vector<float> qr = FromBf16(qb), kr = FromBf16(kb), vr = FromBf16(vb);

  auto mk = [&]() {
    DFlashBlockAttentionArgs a = Args(cu.data(), num_reqs, causal, window);
    a.scale = scale;
    return a;
  };

  std::vector<float> want(q.size(), 0.0f);
  {
    Queue cq = Q();
    Tensor qt = F32(qr, {T, Hq, D}), kt = F32(kr, {T, Hk, D}), vt_ = F32(vr, {T, Hk, D});
    Tensor ot = F32(want, {T, Hq, D});
    vt::DFlashBlockAttention(cq, ot, qt, kt, vt_, mk());
  }

  Queue gq = cuda->CreateQueue();
  auto up16 = [&](const std::vector<uint16_t>& hv) {
    void* p = cuda->Alloc(hv.size() * sizeof(uint16_t));
    cuda->Copy(gq, p, hv.data(), hv.size() * sizeof(uint16_t));
    return p;
  };
  void* dq = up16(qb); void* dk = up16(kb); void* dv = up16(vb);
  const size_t obytes = q.size() * (bf16_out ? sizeof(uint16_t) : sizeof(float));
  void* dout = cuda->Alloc(obytes);
  Device gd = gq.device;
  Tensor gqt = Contig(dq, DType::kBF16, gd, {T, Hq, D});
  Tensor gkt = Contig(dk, DType::kBF16, gd, {T, Hk, D});
  Tensor gvt = Contig(dv, DType::kBF16, gd, {T, Hk, D});
  Tensor got = Contig(dout, bf16_out ? DType::kBF16 : DType::kF32, gd, {T, Hq, D});
  vt::DFlashBlockAttention(gq, got, gqt, gkt, gvt, mk());
  cuda->Synchronize(gq);

  std::vector<float> host(q.size(), 0.0f);
  if (bf16_out) {
    std::vector<uint16_t> h16(q.size(), 0);
    cuda->Copy(gq, h16.data(), dout, obytes);
    cuda->Synchronize(gq);
    host = FromBf16(h16);
  } else {
    cuda->Copy(gq, host.data(), dout, obytes);
    cuda->Synchronize(gq);
  }

  double worst = 0.0, sq = 0.0;
  size_t worst_at = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(std::isfinite(host[i]));
    const double dif = std::abs(static_cast<double>(host[i]) - want[i]);
    sq += dif * dif;
    if (dif > worst) { worst = dif; worst_at = i; }
  }
  const double rms = std::sqrt(sq / static_cast<double>(want.size()));
  INFO(what << ": bf16 CUDA vs bf16-rounded CPU reference, T=" << T << " D=" << D
            << " causal=" << causal << " window=" << window << " bf16_out=" << bf16_out
            << " max|diff|=" << worst << " at " << worst_at << " rms=" << rms);
  CHECK(worst <= tol);
  CHECK(rms <= rms_tol);
  cuda->Free(dq); cuda->Free(dk); cuda->Free(dv); cuda->Free(dout);
}

}  // namespace

TEST_CASE("dflash-block-attn bf16 TENSOR-CORE path matches the reference (semantic corners)") {
  // The same five corners the f32 CUDA case pins, plus head_dim 16 (the smallest
  // shape the MMA serves, and the one test_minimax_h3's golden DiT uses) and 96.
  RunBf16Parity("nc d128 gqa", 17, 32, 8, 128, std::pow(128.0f, -0.5f), false, 0, {0, 17},
                0x243F6A8885A308D3ULL, false, 5e-3, 1e-3);
  RunBf16Parity("causal d128", 17, 32, 8, 128, std::pow(128.0f, -0.5f), true, 2048, {0, 17},
                0x13198A2E03707344ULL, false, 5e-3, 1e-3);
  RunBf16Parity("blocks d64", 51, 16, 4, 64, 0.25f, false, 0, {0, 17, 34, 51},
                0xA4093822299F31D0ULL, false, 5e-3, 1e-3);
  RunBf16Parity("swa d32", 17, 8, 2, 32, 0.3f, true, 4, {0, 17}, 0x082EFA98EC4E6C89ULL, false,
                5e-3, 1e-3);
  RunBf16Parity("gqa-extreme ragged d16", 20, 8, 1, 16, 0.35f, true, 2048, {0, 6, 20},
                0xBE5466CF34E90C6CULL, false, 5e-3, 1e-3);
  RunBf16Parity("nc d96", 17, 8, 2, 96, std::pow(96.0f, -0.5f), false, 0, {0, 17},
                0xC0AC29B7C97C50DDULL, false, 5e-3, 1e-3);
  // bf16 OUTPUT too — the production stream's dtype, which adds one more rounding.
  RunBf16Parity("nc d128 bf16-out", 17, 32, 8, 128, std::pow(128.0f, -0.5f), false, 0, {0, 17},
                0x9E3779B97F4A7C15ULL, true, 8e-3, 2e-3);
}

TEST_CASE("dflash-block-attn bf16 TENSOR-CORE path is right at LENGTH (H3 packed shape)") {
  // H3's real layout: cu_seqlens {0, used, seq_len} — TWO documents, the boundary
  // deliberately off every tile multiple, at head_dim 128 (the production shape,
  // 7168 attention inner / 56 heads) and at 96. Long sequences are where the
  // online-softmax rescaling across many tiles can drift and where blocks straddle
  // a document boundary.
  RunBf16Parity("long two-doc d128", 3000, 2, 2, 128, std::pow(128.0f, -0.5f), false, 0,
                {0, 2317, 3000}, 0xD1B54A32D192ED03ULL, false, 5e-3, 1e-3);
  RunBf16Parity("long nc d96", 2560, 2, 2, 96, 0.102062f, false, 0, {0, 2560},
                0x452821E638D01377ULL, false, 5e-3, 1e-3);
  RunBf16Parity("long causal d64", 2560, 2, 2, 64, 0.125f, true, 0, {0, 2560},
                0x3F84D5B5B5470917ULL, false, 5e-3, 1e-3);
  RunBf16Parity("long SWA d128", 2048, 2, 2, 128, 0.088388f, true, 48, {0, 2048},
                0x9216D5D98979FB1BULL, false, 5e-3, 1e-3);
  RunBf16Parity("long ragged causal d64", 2185, 2, 2, 64, 0.125f, true, 0, {0, 501, 1503, 2185},
                0xBA7C9045F12C7F99ULL, false, 5e-3, 1e-3);
  RunBf16Parity("long ragged nc d96", 2185, 2, 2, 96, 0.102062f, false, 0, {0, 501, 1503, 2185},
                0x24A19947B3916CF7ULL, false, 5e-3, 1e-3);
}

TEST_CASE("dflash-block-attn bf16 RED: the MASK is load-bearing on the tensor-core path") {
  // A tolerance gate alone cannot tell "right kernel" from "wrong mask" — so pin
  // the SEPARATION: running the same inputs causal must move the answer by orders
  // of magnitude more than the 5e-3 bf16 bound the cases above allow.
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  const int64_t T = 64, H = 2, D = 64;
  std::vector<float> q(static_cast<size_t>(T * H * D)), k(q.size()), v(q.size());
  uint64_t x = 0x2FFD72DBD01ADFB7ULL;
  auto rnd = [&]() {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return static_cast<float>((x >> 40) / 16777216.0 - 0.5);
  };
  for (size_t i = 0; i < q.size(); ++i) { q[i] = rnd(); k[i] = rnd(); v[i] = rnd() * 8.0f; }
  const std::vector<uint16_t> qb = ToBf16(q), kb = ToBf16(k), vb = ToBf16(v);

  Queue gq = cuda->CreateQueue();
  auto up16 = [&](const std::vector<uint16_t>& hv) {
    void* p = cuda->Alloc(hv.size() * sizeof(uint16_t));
    cuda->Copy(gq, p, hv.data(), hv.size() * sizeof(uint16_t));
    return p;
  };
  void* dq = up16(qb); void* dk = up16(kb); void* dv = up16(vb);
  void* o_nc = cuda->Alloc(q.size() * sizeof(float));
  void* o_c = cuda->Alloc(q.size() * sizeof(float));
  Device gd = gq.device;
  Tensor gqt = Contig(dq, DType::kBF16, gd, {T, H, D});
  Tensor gkt = Contig(dk, DType::kBF16, gd, {T, H, D});
  Tensor gvt = Contig(dv, DType::kBF16, gd, {T, H, D});
  Tensor tnc = Contig(o_nc, DType::kF32, gd, {T, H, D});
  Tensor tc = Contig(o_c, DType::kF32, gd, {T, H, D});
  const int32_t cu[2] = {0, static_cast<int32_t>(T)};
  vt::DFlashBlockAttention(gq, tnc, gqt, gkt, gvt, Args(cu, 1, /*causal=*/false, 0));
  vt::DFlashBlockAttention(gq, tc, gqt, gkt, gvt, Args(cu, 1, /*causal=*/true, 0));
  cuda->Synchronize(gq);
  std::vector<float> hnc(q.size()), hc(q.size());
  cuda->Copy(gq, hnc.data(), o_nc, hnc.size() * sizeof(float));
  cuda->Copy(gq, hc.data(), o_c, hc.size() * sizeof(float));
  cuda->Synchronize(gq);

  double sep = 0.0;
  for (size_t i = 0; i < hnc.size(); ++i)
    sep = std::max(sep, std::abs(static_cast<double>(hnc[i]) - hc[i]));
  INFO("causal vs non-causal separation on the bf16 tensor-core path = " << sep);
  CHECK(sep > 0.5);  // ~100x the bf16 tolerance the parity cases allow
  cuda->Free(dq); cuda->Free(dk); cuda->Free(dv); cuda->Free(o_nc); cuda->Free(o_c);
}

// ─── SPEC-DFLASH2 W12 D1 (#2087): the SEPARATE QUERY cu ─────────────────────
//
// WHAT THIS GATES, and why it is an EQUALITY rather than a tolerance. Before D1
// the only way to attend a (1+k) block over a materialized [context ; block]
// sequence was to hand this op a query buffer spanning ALL `Ncomb = C + Tq`
// rows, compute an output for every context row, and `IndexSelect` the block
// rows back out — `qwen3_dflash.cpp` did exactly that, and its own comment said
// the context rows' outputs were discarded. `cu_seqlens_q` lets the query stay
// `[Tq, ...]`. The claim D1 rests on is not "close enough": it is that the
// SURVIVING rows see the same keys in the same order under the same mask bound,
// so the two forms are BIT-identical on the CPU reference. A tolerance here
// would pass an off-by-one in the bottom-right anchor, which is the one mistake
// this argument can make.
//
// The two arms are genuinely different code, not one helper called twice: arm A
// takes the `cu_seqlens_q == nullptr` path with a square block, arm B takes the
// offset path. Each case asserts `Ncomb > Tq` so arm A really does compute the
// rows D1 removes, and asserts the compared values are not all zero, because
// two all-zero buffers compare equal and prove nothing.
namespace {

// Returns {ok, message}. `ctx_len[r]` context rows then `blk_len[r]` block rows,
// per request, in the [context ; block] layout the draft materializes.
void RunD1Equivalence(const char* name, const std::vector<int32_t>& ctx_len,
                      const std::vector<int32_t>& blk_len, int64_t Hq, int64_t Hk, int64_t D,
                      float scale, bool causal, int64_t window, uint32_t seed) {
  // doctest stringifies a `const char*` as a BOOL, so a bare `<< name` here
  // would print "1" and the failing case would be unidentifiable.
  INFO("D1 equivalence case: " << std::string(name));
  const int num_reqs = static_cast<int>(ctx_len.size());
  REQUIRE(blk_len.size() == ctx_len.size());
  std::vector<int32_t> cu(static_cast<size_t>(num_reqs) + 1, 0);
  std::vector<int32_t> cu_q(static_cast<size_t>(num_reqs) + 1, 0);
  std::vector<int32_t> blk_rows;  // combined row of each block query, in cu_q order
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t c = ctx_len[static_cast<size_t>(r)], b = blk_len[static_cast<size_t>(r)];
    cu[static_cast<size_t>(r) + 1] = cu[static_cast<size_t>(r)] + c + b;
    cu_q[static_cast<size_t>(r) + 1] = cu_q[static_cast<size_t>(r)] + b;
    for (int32_t i = 0; i < b; ++i) blk_rows.push_back(cu[static_cast<size_t>(r)] + c + i);
  }
  const int64_t Ncomb = cu.back(), Tq = cu_q.back();
  REQUIRE(Tq > 0);

  const auto k = RandF32(static_cast<size_t>(Ncomb * Hk * D), seed + 1);
  const auto v = RandF32(static_cast<size_t>(Ncomb * Hk * D), seed + 2);
  const auto qblk = RandF32(static_cast<size_t>(Tq * Hq * D), seed + 3);
  // Arm A's query buffer: the block queries at their combined rows, ZERO
  // elsewhere — byte-for-byte what `qcb.Zero(d)` + the query IndexCopy built.
  std::vector<float> qfull(static_cast<size_t>(Ncomb * Hq * D), 0.0f);
  for (int64_t i = 0; i < Tq; ++i) {
    const int64_t dst = blk_rows[static_cast<size_t>(i)];
    for (int64_t e = 0; e < Hq * D; ++e)
      qfull[static_cast<size_t>(dst * Hq * D + e)] = qblk[static_cast<size_t>(i * Hq * D + e)];
  }

  auto base = [&]() {
    DFlashBlockAttentionArgs a = Args(cu.data(), num_reqs, causal, window);
    a.scale = scale;
    return a;
  };
  Queue qq = Q();

  // Arm A — the pre-D1 shape: Q over Ncomb rows, then gather the block rows.
  std::vector<float> outA(static_cast<size_t>(Ncomb * Hq * D), 0.0f);
  {
    std::vector<float> kk = k, vv = v, qq2 = qfull;
    Tensor tq = F32(qq2, {Ncomb, Hq, D}), tk = F32(kk, {Ncomb, Hk, D});
    Tensor tv = F32(vv, {Ncomb, Hk, D}), to = F32(outA, {Ncomb, Hq, D});
    vt::DFlashBlockAttention(qq, to, tq, tk, tv, base());
  }
  std::vector<float> refA(static_cast<size_t>(Tq * Hq * D), 0.0f);
  for (int64_t i = 0; i < Tq; ++i)
    for (int64_t e = 0; e < Hq * D; ++e)
      refA[static_cast<size_t>(i * Hq * D + e)] =
          outA[static_cast<size_t>(blk_rows[static_cast<size_t>(i)] * Hq * D + e)];

  // Arm B — D1: Q over the Tq block rows, keys still over Ncomb.
  std::vector<float> outB(static_cast<size_t>(Tq * Hq * D), 0.0f);
  {
    std::vector<float> kk = k, vv = v, qb = qblk;
    Tensor tq = F32(qb, {Tq, Hq, D}), tk = F32(kk, {Ncomb, Hk, D});
    Tensor tv = F32(vv, {Ncomb, Hk, D}), to = F32(outB, {Tq, Hq, D});
    DFlashBlockAttentionArgs a = base();
    a.cu_seqlens_q = cu_q.data();
    vt::DFlashBlockAttention(qq, to, tq, tk, tv, a);
  }

  // The compared values must carry information: all-zero buffers compare equal.
  bool any_nonzero = false;
  for (float x : refA)
    if (x != 0.0f) { any_nonzero = true; break; }
  REQUIRE(any_nonzero);
  size_t mismatches = 0;
  double worst = 0.0;
  for (size_t i = 0; i < refA.size(); ++i) {
    if (refA[i] != outB[i]) {
      ++mismatches;
      worst = std::max(worst, std::abs(static_cast<double>(refA[i]) - outB[i]));
    }
  }
  INFO("case=" << std::string(name) << " Ncomb=" << Ncomb << " Tq=" << Tq
                << " mismatches=" << mismatches << " worst=" << worst);
  CHECK(mismatches == 0);
}

// The CUDA leg of the same claim. `RunD1Equivalence` above runs on the CPU
// reference only, and the CPU reference is ONE function — it cannot say anything
// about the five CUDA kernels that each carry their own mask arithmetic and each
// had to learn the query cu separately (spec R2: a query cu honoured in four of
// them and forgotten in the fifth is an acceptance-only defect a token gate
// cannot see). This runs the D1 shape on CUDA against the CPU reference on the
// SAME inputs, at head_dims that select different kernels.
//
// The bar is a tolerance rather than an equality here, and deliberately so: the
// CUDA forms use a flash-style online recurrence and chunked rescaling, so they
// agree with the two-pass CPU reference to f32 rounding, exactly as the
// pre-existing `RunCudaParity` cases in this file do. What the CUDA leg gates is
// that the mask and the row mapping are right, which a 1e-4 bound catches loudly
// — a dropped offset moves an output by whole units, as the CPU mutation
// evidence for this change measured.
void RunD1CudaParity(const char* name, const std::vector<int32_t>& ctx_len,
                     const std::vector<int32_t>& blk_len, int64_t Hq, int64_t Hk, int64_t D,
                     float scale, bool causal, int64_t window, uint32_t seed) {
  INFO("D1 CUDA case: " << std::string(name));
  const int num_reqs = static_cast<int>(ctx_len.size());
  std::vector<int32_t> cu(static_cast<size_t>(num_reqs) + 1, 0);
  std::vector<int32_t> cu_q(static_cast<size_t>(num_reqs) + 1, 0);
  for (int r = 0; r < num_reqs; ++r) {
    cu[static_cast<size_t>(r) + 1] = cu[static_cast<size_t>(r)] +
                                     ctx_len[static_cast<size_t>(r)] +
                                     blk_len[static_cast<size_t>(r)];
    cu_q[static_cast<size_t>(r) + 1] =
        cu_q[static_cast<size_t>(r)] + blk_len[static_cast<size_t>(r)];
  }
  const int64_t Ncomb = cu.back(), Tq = cu_q.back();
  REQUIRE(Ncomb > Tq);

  auto q = RandF32(static_cast<size_t>(Tq * Hq * D), seed);
  auto k = RandF32(static_cast<size_t>(Ncomb * Hk * D), seed + 1);
  auto v = RandF32(static_cast<size_t>(Ncomb * Hk * D), seed + 2);
  auto mkargs = [&]() {
    DFlashBlockAttentionArgs a = Args(cu.data(), num_reqs, causal, window);
    a.scale = scale;
    a.cu_seqlens_q = cu_q.data();
    return a;
  };

  std::vector<float> cpu(static_cast<size_t>(Tq * Hq * D), 0.0f);
  Tensor cq = Contig(q.data(), DType::kF32, Cpu(), {Tq, Hq, D});
  Tensor ck = Contig(k.data(), DType::kF32, Cpu(), {Ncomb, Hk, D});
  Tensor cv = Contig(v.data(), DType::kF32, Cpu(), {Ncomb, Hk, D});
  Tensor co = Contig(cpu.data(), DType::kF32, Cpu(), {Tq, Hq, D});
  Queue cpuq = Q();
  vt::DFlashBlockAttention(cpuq, co, cq, ck, cv, mkargs());

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {Tq, Hq, D}, q.data());
  DeviceTensor dk(gpu, g.q, DType::kF32, {Ncomb, Hk, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kF32, {Ncomb, Hk, D}, v.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {Tq, Hq, D});
  vt::DFlashBlockAttention(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), mkargs());
  std::vector<float> got(static_cast<size_t>(Tq * Hq * D), 0.0f);
  dout.Download(g.q, got.data());

  bool any_nonzero = false;
  for (float x : cpu)
    if (x != 0.0f) { any_nonzero = true; break; }
  REQUIRE(any_nonzero);
  double worst = 0.0;
  size_t bad = 0;
  for (size_t i = 0; i < cpu.size(); ++i) {
    const double e = std::abs(static_cast<double>(got[i]) - cpu[i]);
    if (e > worst) worst = e;
    if (!(got[i] == doctest::Approx(cpu[i]).epsilon(1e-4))) ++bad;
  }
  INFO("case=" << std::string(name) << " Ncomb=" << Ncomb << " Tq=" << Tq << " bad=" << bad
               << " worst=" << worst);
  CHECK(bad == 0);
}

// The bf16 TENSOR-CORE leg. `RunD1CudaParity` feeds f32, which by dispatch can
// never reach `DFlashAttnMmaKernel` — and that kernel took the largest share of
// D1's change, because its single `rows` bounded the query grid AND the key
// staging clamp and had to become two. Without this the kernel that needed the
// most care would ship with no D1 coverage while the suite reported green, which
// is the exact shape of failure this file's own comments keep recording.
//
// Q/K/V are rounded to bf16 ONCE so both sides see identical numbers; the bar is
// the same bf16 envelope the file's other tensor-core cases use.
void RunD1Bf16Parity(const char* name, const std::vector<int32_t>& ctx_len,
                     const std::vector<int32_t>& blk_len, int64_t Hq, int64_t Hk, int64_t D,
                     float scale, bool causal, int64_t window, uint64_t seed, double tol) {
  INFO("D1 bf16 case: " << std::string(name));
  const int num_reqs = static_cast<int>(ctx_len.size());
  std::vector<int32_t> cu(static_cast<size_t>(num_reqs) + 1, 0);
  std::vector<int32_t> cu_q(static_cast<size_t>(num_reqs) + 1, 0);
  for (int r = 0; r < num_reqs; ++r) {
    cu[static_cast<size_t>(r) + 1] = cu[static_cast<size_t>(r)] +
                                     ctx_len[static_cast<size_t>(r)] +
                                     blk_len[static_cast<size_t>(r)];
    cu_q[static_cast<size_t>(r) + 1] =
        cu_q[static_cast<size_t>(r)] + blk_len[static_cast<size_t>(r)];
  }
  const int64_t Ncomb = cu.back(), Tq = cu_q.back();
  REQUIRE(Ncomb > Tq);

  uint64_t x = seed;
  auto rnd = [&]() {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return static_cast<float>((x >> 40) / 16777216.0 - 0.5);
  };
  std::vector<float> q(static_cast<size_t>(Tq * Hq * D));
  std::vector<float> k(static_cast<size_t>(Ncomb * Hk * D)), v(k.size());
  for (auto& e : q) e = rnd();
  for (size_t i = 0; i < k.size(); ++i) { k[i] = rnd(); v[i] = rnd(); }
  const std::vector<uint16_t> qb = ToBf16(q), kb = ToBf16(k), vb = ToBf16(v);
  std::vector<float> qr = FromBf16(qb), kr = FromBf16(kb), vr = FromBf16(vb);

  auto mk = [&]() {
    DFlashBlockAttentionArgs a = Args(cu.data(), num_reqs, causal, window);
    a.scale = scale;
    a.cu_seqlens_q = cu_q.data();
    return a;
  };

  std::vector<float> want(q.size(), 0.0f);
  {
    Queue cq = Q();
    Tensor qt = F32(qr, {Tq, Hq, D}), kt = F32(kr, {Ncomb, Hk, D});
    Tensor vt_ = F32(vr, {Ncomb, Hk, D}), ot = F32(want, {Tq, Hq, D});
    vt::DFlashBlockAttention(cq, ot, qt, kt, vt_, mk());
  }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kBF16, {Tq, Hq, D}, qb.data());
  DeviceTensor dk(gpu, g.q, DType::kBF16, {Ncomb, Hk, D}, kb.data());
  DeviceTensor dv(gpu, g.q, DType::kBF16, {Ncomb, Hk, D}, vb.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {Tq, Hq, D});
  vt::DFlashBlockAttention(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), mk());
  std::vector<float> host(q.size(), 0.0f);
  dout.Download(g.q, host.data());

  bool any_nonzero = false;
  for (float e : want)
    if (e != 0.0f) { any_nonzero = true; break; }
  REQUIRE(any_nonzero);
  double worst = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(std::isfinite(host[i]));
    worst = std::max(worst, std::abs(static_cast<double>(host[i]) - want[i]));
  }
  INFO("case=" << std::string(name) << " Ncomb=" << Ncomb << " Tq=" << Tq
               << " max|diff|=" << worst);
  CHECK(worst <= tol);
}

}  // namespace

TEST_CASE("dflash-block-attn D1: a separate QUERY cu is BIT-IDENTICAL to the full-Q form") {
  const float sc = std::pow(64.0f, -0.5f);
  // (1) One request, non-causal — the campaign draft's every layer.
  RunD1Equivalence("nc single", {5}, {3}, 2, 1, 4, 0.5f, false, 0, 11);
  // (2) Ragged multi-request, including a request with NO context (Ncomb == Tq
  //     for that block, so the bottom-right offset is 0 for it and positive for
  //     its neighbours in the SAME call).
  RunD1Equivalence("nc ragged 3 reqs", {5, 0, 11}, {3, 3, 3}, 4, 2, 8, 0.35f, false, 0, 22);
  // (3) Plain causal: the mask bound must read the COMBINED offset, so a block
  //     query sees its own context. Reading the query offset instead would make
  //     query 0 see only key 0 and this case reds.
  RunD1Equivalence("causal 2 reqs", {7, 4}, {3, 3}, 4, 2, 8, 0.35f, true, 0, 33);
  // (4) SWA with a window SHORTER than the context, which is the only case that
  //     bounds jlo from below — an offset applied to jhi but not to jlo passes
  //     (3) and reds here.
  RunD1Equivalence("SWA w=4", {7, 4}, {3, 3}, 2, 1, 8, 0.35f, true, 4, 44);
  // (5) GQA extreme: 8 q-heads share 1 kv-head.
  RunD1Equivalence("GQA 8:1", {6, 2}, {3, 3}, 8, 1, 16, 0.25f, false, 0, 55);
  // (6) Production geometry: 9 = 1+k query rows per request over a long context.
  RunD1Equivalence("prod 1+k=9 over 130 ctx", {130, 97}, {9, 9}, 4, 2, 64, sc, false, 0, 66);
  // (7) DEGENERATE: no context at all, so cu_seqlens_q == cu_seqlens and D1 must
  //     be the identity. This is the null case stated as a test rather than as a
  //     comment.
  RunD1Equivalence("no context", {0, 0}, {3, 3}, 2, 1, 4, 0.5f, false, 0, 77);
}

TEST_CASE("dflash-block-attn D1 RED: the full-Q form really computes the discarded rows") {
  // The anti-tautology for the case above. If arm A's context-row outputs were
  // already absent — if the op somehow skipped them — the equality would be
  // vacuous. Run the pre-D1 shape and show that a CONTEXT row carries a computed
  // attention output: the very work D1 deletes. The context query rows are ZERO
  // vectors, so every score is 0, softmax is uniform, and the output is the MEAN
  // of the request's value rows — which is not zero for random V and is exactly
  // what "computed and thrown away" looks like.
  const int64_t C = 5, B = 3, Ncomb = C + B, Hq = 2, Hk = 1, D = 4;
  const auto k = RandF32(static_cast<size_t>(Ncomb * Hk * D), 909);
  auto v = RandF32(static_cast<size_t>(Ncomb * Hk * D), 910);
  std::vector<float> qfull(static_cast<size_t>(Ncomb * Hq * D), 0.0f);
  const auto qblk = RandF32(static_cast<size_t>(B * Hq * D), 911);
  for (int64_t i = 0; i < B; ++i)
    for (int64_t e = 0; e < Hq * D; ++e)
      qfull[static_cast<size_t>((C + i) * Hq * D + e)] = qblk[static_cast<size_t>(i * Hq * D + e)];
  std::vector<float> kk = k, out(static_cast<size_t>(Ncomb * Hq * D), 0.0f);
  Tensor tq = F32(qfull, {Ncomb, Hq, D}), tk = F32(kk, {Ncomb, Hk, D});
  Tensor tv = F32(v, {Ncomb, Hk, D}), to = F32(out, {Ncomb, Hq, D});
  const int32_t cu[] = {0, static_cast<int32_t>(Ncomb)};
  Queue qq = Q();
  DFlashBlockAttentionArgs a = Args(cu, 1, /*causal=*/false, 0);
  a.scale = 0.5f;
  vt::DFlashBlockAttention(qq, to, tq, tk, tv, a);
  // Row 0 is a CONTEXT row: D1 never computes it. Here it holds the mean of V.
  for (int64_t e = 0; e < D; ++e) {
    double mean = 0.0;
    for (int64_t j = 0; j < Ncomb; ++j) mean += v[static_cast<size_t>(j * Hk * D + e)];
    mean /= static_cast<double>(Ncomb);
    CHECK(out[static_cast<size_t>(e)] == doctest::Approx(mean).epsilon(1e-5));
  }
}

TEST_CASE("dflash-block-attn D1 CUDA: the query cu reaches every dispatched kernel") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping the D1 CUDA parity");
    return;
  }
  // Head dims chosen to select DIFFERENT kernels, because "the CUDA arm agrees"
  // is worth nothing if one kernel answered every case. `LaunchDFlashBlockAttention`
  // dispatches on dtype and head_dim: f32 with d % 32 == 0 and d/32 <= 4 takes the
  // chunked reduce-scatter form (or the key-lane / per-key warp form under their
  // env switches, which are process-latched statics and so cannot both be driven
  // from one binary); anything else falls to the general block-reduction kernel.
  // The bf16 tensor-core kernel takes bf16 only and has its own case below.
  const float sc64 = std::pow(64.0f, -0.5f);
  // Chunked form, d = 64, non-causal — the campaign draft's own layer shape.
  RunD1CudaParity("chunk d64 nc", {130, 97}, {9, 9}, 4, 2, 64, sc64, false, 0, 1201);
  // Chunked form, causal, so the mask reads the combined offset on the device.
  RunD1CudaParity("chunk d64 causal", {130, 97}, {9, 9}, 4, 2, 64, sc64, true, 0, 1202);
  // Chunked form with a window SHORTER than the context: the only case that
  // bounds jlo from below.
  RunD1CudaParity("chunk d64 SWA w=40", {130, 97}, {9, 9}, 2, 1, 64, sc64, true, 40, 1203);
  // d = 96, the head_dim that is a whole number of warp widths but not a power of
  // two, and whose per-lane partition differs from the 64/128 instantiations.
  RunD1CudaParity("chunk d96 nc", {77, 0, 51}, {9, 9, 9}, 4, 2, 96,
                  std::pow(96.0f, -0.5f), false, 0, 1204);
  // d = 48: NOT a multiple of 32, so this one falls to the general
  // block-reduction kernel, which nothing else in this case reaches.
  RunD1CudaParity("general d48 causal", {60, 33}, {9, 9}, 2, 1, 48,
                  std::pow(48.0f, -0.5f), true, 0, 1205);
  // GQA extreme through the device path.
  RunD1CudaParity("chunk d64 GQA 8:1", {40, 12}, {9, 9}, 8, 1, 64, sc64, false, 0, 1206);
}

TEST_CASE("dflash-block-attn D1 bf16 TENSOR-CORE: the query cu reaches the MMA kernel") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping the D1 tensor-core parity");
    return;
  }
  // head_dim 128 and 64 both satisfy the MMA dispatch (bf16, d % 16 == 0,
  // 16 <= d <= 128). Contexts are long enough that a 64-query block walks
  // several kMmaKeys tiles, which is where the block-wide key-range UNION and
  // the two row counts actually have to be right.
  RunD1Bf16Parity("mma d128 nc", {301, 208}, {9, 9}, 2, 2, 128, std::pow(128.0f, -0.5f),
                  false, 0, 0xD1B54A32D192ED03ULL, 5e-3);
  RunD1Bf16Parity("mma d64 causal", {301, 208}, {9, 9}, 2, 2, 64, 0.125f, true, 0,
                  0x452821E638D01377ULL, 5e-3);
  RunD1Bf16Parity("mma d64 SWA w=48", {301, 208}, {9, 9}, 2, 2, 64, 0.125f, true, 48,
                  0x9216D5D98979FB1BULL, 5e-3);
  // Ragged, and one request with NO context, so the bottom-right anchor is zero
  // for one block and positive for its neighbours inside the same launch.
  RunD1Bf16Parity("mma d128 ragged", {177, 0, 253}, {9, 9, 9}, 2, 2, 128,
                  std::pow(128.0f, -0.5f), false, 0, 0xBA7C9045F12C7F99ULL, 5e-3);
  // MULTI-TILE. Every case above has `Tq <= 27`, and the MMA grid is
  // `ceil(Tq / (kMmaWarps * kMmaQ)) == ceil(Tq / 64) == 1`, so D1's query-row
  // split had only ever been exercised INSIDE ONE query block. The comment
  // above reasons about walking several `kMmaKeys` tiles, which is the KEY
  // axis; the QUERY axis had no coverage past its first block at all.
  //
  // Production crosses it on every step: 8 concurrent requests at k=8 is
  // `Tq = 8 * 9 = 72`, which is TWO query blocks, and the second holds only the
  // last request's nine rows.
  //
  // These PASS. They are here because nothing was executing the shape, not
  // because the shape was broken -- which is the only reason a reader would
  // otherwise think to add them (#2154).
  RunD1Bf16Parity("mma MULTI-TILE 8 reqs Tq=72 d128 nc",
                  {301, 208, 177, 253, 190, 145, 233, 168},
                  {9, 9, 9, 9, 9, 9, 9, 9}, 2, 2, 128, std::pow(128.0f, -0.5f), false, 0,
                  0x2154D1A5C0FFEE01ULL, 5e-3);
  RunD1Bf16Parity("mma MULTI-TILE 8 reqs Tq=72 d64 causal",
                  {301, 208, 177, 253, 190, 145, 233, 168},
                  {9, 9, 9, 9, 9, 9, 9, 9}, 2, 2, 64, 0.125f, true, 0,
                  0x2154D1A5C0FFEE02ULL, 5e-3);
  // The single-block control at seven requests, `Tq = 63`. It is what makes the
  // pair above diagnostic rather than merely present: if a future change reds
  // the 8-request cases and leaves this green, the query-block boundary is the
  // discriminator and nothing else in the shape is.
  RunD1Bf16Parity("mma SINGLE-TILE control 7 reqs Tq=63 d128 nc",
                  {301, 208, 177, 253, 190, 145, 233}, {9, 9, 9, 9, 9, 9, 9}, 2, 2, 128,
                  std::pow(128.0f, -0.5f), false, 0, 0x2154D1A5C0FFEE03ULL, 5e-3);
  // PRODUCTION SCALE. The cases above cross the query-block boundary but keep
  // contexts small, so the block-wide key union stays ~1.7k combined rows. A
  // c=8 step at ctx 2048 carries ~1.2k context rows PER REQUEST, so the union
  // the kernel walks is ~9.7k -- about 5.5x more `kMmaKeys` tiles per query
  // block, with the online softmax's running max and sum carried across every
  // one of them.
  RunD1Bf16Parity("mma PRODUCTION SCALE 8 reqs Tq=72 ctx~1200 d128 nc",
                  {1203, 1187, 1211, 1195, 1219, 1178, 1206, 1192},
                  {9, 9, 9, 9, 9, 9, 9, 9}, 2, 2, 128, std::pow(128.0f, -0.5f), false, 0,
                  0x2154D1A5C0FFEE04ULL, 5e-3);
  RunD1Bf16Parity("mma PRODUCTION SCALE 8 reqs Tq=72 ctx~1200 d64 causal",
                  {1203, 1187, 1211, 1195, 1219, 1178, 1206, 1192},
                  {9, 9, 9, 9, 9, 9, 9, 9}, 2, 2, 64, 0.125f, true, 0,
                  0x2154D1A5C0FFEE05ULL, 5e-3);
  // The single-REQUEST control at the same key extent, so a future failure can
  // be attributed to the many-request union rather than to context length.
  RunD1Bf16Parity("mma PRODUCTION SCALE 1 req Tq=9 ctx~1200 d128 nc", {1203}, {9}, 2, 2,
                  128, std::pow(128.0f, -0.5f), false, 0, 0x2154D1A5C0FFEE06ULL, 5e-3);
}

// ===========================================================================
// SPEC-DFLASH2 (#2784) — THE RIGHT-HAND HALF OF THE SYMMETRIC WINDOW.
//
// `_maybe_symmetrize_window` (vllm/v1/attention/backends/flash_attn.py:319-330
// @ the parity pin 5559679229) gives a non-causal layer `(w-1, w-1)`, not
// `(w-1, 0)`: "so bidirectional queries attend in both directions". The LEFT
// half is what #2784 measured, because the DFlash draft block's query rows sit
// at the END of the combined sequence and their right bound therefore falls off
// the axis. This case is here because that makes the right half UNGATED by the
// paged battery — the #2784 mutation run proved it: deleting the upper clamp
// left `test_ops_dflash_paged_block_attn` fully green.
//
// The dense op is where it binds. `ForwardWithCtxKVDev`'s materialized
// [context; block] form runs EVERY combined row as a query, so a row in the
// middle has keys on both sides of it, and that is also the reference the
// paged battery compares against. A query row must not see a key more than
// `w-1` positions AFTER it.
TEST_CASE("dflash-block-attn: a NON-CAUSAL window bounds the key range on the RIGHT (#2784)") {
  const int64_t blen = 40, hq = 2, hk = 1, d = 8, window = 6;
  const int64_t kProbe = 10;         // the query row under test
  const int64_t kFirstHidden = 16;   // kProbe + (window - 1) + 1
  std::vector<int32_t> cu = {0, static_cast<int32_t>(blen)};

  auto rnd = [](size_t n, uint32_t seed) {
    std::vector<float> v(n);
    uint32_t s = seed;
    for (auto& x : v) {
      s = s * 1664525u + 1013904223u;
      x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
    }
    return v;
  };
  auto q = rnd(static_cast<size_t>(blen * hq * d), 31);
  auto k_a = rnd(static_cast<size_t>(blen * hk * d), 32);
  auto v_a = rnd(static_cast<size_t>(blen * hk * d), 33);
  auto k_b = k_a;
  auto v_b = v_a;
  // B differs from A ONLY on key rows [kFirstHidden, blen) — strictly to the
  // RIGHT of the probe row's symmetric window.
  auto tail_k = rnd(static_cast<size_t>((blen - kFirstHidden) * hk * d), 91);
  auto tail_v = rnd(static_cast<size_t>((blen - kFirstHidden) * hk * d), 92);
  for (size_t i = 0; i < tail_k.size(); ++i) {
    k_b[static_cast<size_t>(kFirstHidden * hk * d) + i] = tail_k[i];
    v_b[static_cast<size_t>(kFirstHidden * hk * d) + i] = tail_v[i];
  }
  int differing = 0;
  for (size_t i = 0; i < k_a.size(); ++i) differing += (k_a[i] != k_b[i]) ? 1 : 0;
  REQUIRE(differing > 0);  // the fixture's own precondition

  auto run = [&](std::vector<float>& kk, std::vector<float>& vv, int64_t win) {
    std::vector<float> out(static_cast<size_t>(blen * hq * d), 0.0f);
    Queue cq = Q();
    Tensor tq = F32(q, {blen, hq, d});
    Tensor tk = F32(kk, {blen, hk, d});
    Tensor tv = F32(vv, {blen, hk, d});
    Tensor to = F32(out, {blen, hq, d});
    vt::DFlashBlockAttention(cq, to, tq, tk, tv, Args(cu.data(), 1, /*causal=*/false, win));
    return out;
  };
  auto probe = [&](const std::vector<float>& o, int64_t h) {
    return std::vector<float>(o.begin() + static_cast<long>((kProbe * hq + h) * d),
                              o.begin() + static_cast<long>((kProbe * hq + h) * d + d));
  };

  SUBCASE("the CONTROL: with NO window the same pair MOVES the probe row") {
    const auto a = run(k_a, v_a, 0), b = run(k_b, v_b, 0);
    int moved = 0;
    for (int64_t h = 0; h < hq; ++h) {
      const auto pa = probe(a, h), pb = probe(b, h);
      for (size_t i = 0; i < pa.size(); ++i) moved += (pa[i] != pb[i]) ? 1 : 0;
    }
    REQUIRE(moved > 0);
  }

  SUBCASE("with the window, keys to the RIGHT of it cannot reach the probe row") {
    const auto a = run(k_a, v_a, window), b = run(k_b, v_b, window);
    for (int64_t h = 0; h < hq; ++h) {
      const auto pa = probe(a, h), pb = probe(b, h);
      for (size_t i = 0; i < pa.size(); ++i) REQUIRE(pa[i] == pb[i]);
    }
  }
}
