// vt::BatchedMatmul + vt::ConcatMlaNopeRope unit tests (MLA campaign W6) — the
// two primitives MLA WEIGHT ABSORPTION is built out of.
//
// Upstream test modules ported per .agents/test-porting.md:
//   * vllm/tests/kernels/test_concat_mla_q.py @ pin e24d1b24 — the q-concat
//     helper the campaign spec §7 assigns to this campaign. BOTH of its arms are
//     ported: `test_concat_mla_q_contiguous` (:22-34) and, more importantly,
//     `test_concat_mla_q_transposed_nope` (:37-63), whose docstring records
//     exactly why the op must be stride-driven — "In the real code path,
//     mqa_ql_nope is the result of: torch.bmm(q_nope, W_UK_T) # [N, B, L] then
//     .transpose(0, 1) # [B, N, L] - non-contiguous!". Its `ref = torch.cat(...)`
//     with `atol=0, rtol=0` is ported as an EXACT (bit-for-bit) comparison,
//     because a concat is a pure copy.
//   * vllm/tests/v1/attention/test_mla_backends.py — the absorption shapes
//     (V2-Lite 16 heads and V3 128 heads) come from its MLA geometry sweep.
//
// Kernels/logic under port:
//   * vt::BatchedMatmul <- `torch.bmm` at
//     vllm/model_executor/layers/attention/mla_attention.py:789 (the q-side
//     W_UK fold) and :1034 (`_v_up_proj`'s W_UV un-projection). On CUDA torch
//     resolves those to cuBLAS gemmStridedBatchedEx; our CUDA impl is the
//     cuBLASLt strided-batched form of the same GEMM.
//   * vt::ConcatMlaNopeRope <- `ConcatMLAQKernel`
//     (csrc/libtorch_stable/concat_mla_q.cuh) + host wrapper
//     csrc/libtorch_stable/cache_kernels.cu:1555-1600, generalized to serve
//     `_concat_k_nope_k_pe` (mla_attention.py:2063-2092) as well.
//
// THE ORACLES ARE INDEPENDENT. The bmm oracle is a DOUBLE-precision triple loop
// written directly from the mathematical definition, not from either kernel; the
// concat oracle is a pure host-side gather. Every output buffer is pre-poisoned
// with NaN (or a sentinel) so a kernel that fails to write an element FAILS
// rather than silently passing on stale zeros.
//
// THE SHAPES ARE THE REAL ONES. DeepSeek-V2-Lite (W0-confirmed): kv_lora_rank
// 512, qk_nope_head_dim 128, qk_rope_head_dim 64, v_head_dim 128, 16 heads —
// so W_UK_T is [16,128,512] and W_UV is [16,512,128]. DeepSeek-V3's 128-head
// forms ([128,128,512] / [128,512,128]) are covered too, at reduced batch so the
// test stays fast.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

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
    t_.data = p_;
    t_.dtype = dt;
    t_.device = Gpu();
    t_.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = t_.rank - 1; i >= 0; --i) {
      t_.shape[i] = shape[static_cast<size_t>(i)];
      t_.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
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

// A rank-3 strided view over an existing allocation, in ELEMENTS.
Tensor View3(void* base, DType dt, Device dev, int64_t off, int64_t d0, int64_t d1,
             int64_t d2, int64_t s0, int64_t s1, int64_t s2) {
  Tensor t;
  t.data = static_cast<char*>(base) + static_cast<size_t>(off) * vt::SizeOf(dt);
  t.dtype = dt;
  t.device = dev;
  t.rank = 3;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.stride[0] = s0;
  t.stride[1] = s1;
  t.stride[2] = s2;
  return t;
}

// The same deterministic LCG the W3/W4/W5 MLA tests use.
std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
  }
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::F32ToBF16(v[i]);
  return o;
}

std::vector<float> FromBf16(const std::vector<uint16_t>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::BF16ToF32(v[i]);
  return o;
}

// ─── the INDEPENDENT bmm oracle ─────────────────────────────────────────────
// out[g,m,n] = sum_k a[g,m,k] * b[g,k,n], in DOUBLE, written straight from the
// definition of `torch.bmm` — not from either implementation. Strides are given
// so the transposed-view cases can be checked in exactly the layout the kernel
// sees.
std::vector<double> RefBmm(const std::vector<float>& a, const std::vector<float>& b,
                           int64_t g, int64_t m, int64_t k, int64_t n, int64_t a_s0,
                           int64_t a_s1, int64_t b_s0, int64_t b_s1) {
  std::vector<double> out(static_cast<size_t>(g * m * n), 0.0);
  for (int64_t bi = 0; bi < g; ++bi) {
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int64_t p = 0; p < k; ++p) {
          acc += static_cast<double>(a[static_cast<size_t>(bi * a_s0 + i * a_s1 + p)]) *
                 static_cast<double>(b[static_cast<size_t>(bi * b_s0 + p * b_s1 + j)]);
        }
        out[static_cast<size_t>((bi * m + i) * n + j)] = acc;
      }
    }
  }
  return out;
}

constexpr int kKvLoraRank = 512;
constexpr int kQkNope = 128;
constexpr int kQkRope = 64;
constexpr int kVHeadDim = 128;

}  // namespace

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("BatchedMatmul CPU reproduces the double-precision bmm oracle at MLA shapes") {
  // The q-side absorption shape, DeepSeek-V2-Lite: (N,B,P) x (N,P,L) -> (N,B,L)
  // — mla_attention.py:789.
  const int64_t n_heads = 16, batch = 7;
  auto a = RandF32(static_cast<size_t>(n_heads * batch * kQkNope), 11u);
  auto b = RandF32(static_cast<size_t>(n_heads * kQkNope * kKvLoraRank), 22u);
  auto ab = ToBf16(a), bb = ToBf16(b);
  const auto af = FromBf16(ab), bf = FromBf16(bb);

  std::vector<uint16_t> out(static_cast<size_t>(n_heads * batch * kKvLoraRank), 0x7FC0);
  Queue q = CpuQ();
  Tensor ta = Contig(ab.data(), DType::kBF16, Cpu(), {n_heads, batch, kQkNope});
  Tensor tb = Contig(bb.data(), DType::kBF16, Cpu(), {n_heads, kQkNope, kKvLoraRank});
  Tensor to = Contig(out.data(), DType::kBF16, Cpu(), {n_heads, batch, kKvLoraRank});
  vt::BatchedMatmul(q, to, ta, tb);

  const auto ref = RefBmm(af, bf, n_heads, batch, kQkNope, kKvLoraRank,
                          batch * kQkNope, kQkNope, kQkNope * kKvLoraRank, kKvLoraRank);
  double worst = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    const double got = vt::BF16ToF32(out[i]);
    REQUIRE(!std::isnan(got));  // NaN-poisoned: every element must be WRITTEN
    worst = std::max(worst, std::abs(got - ref[i]) /
                                std::max(1e-3, std::abs(ref[i])));
  }
  // bf16 output rounding over a K=128 f32 reduction.
  CHECK(worst < 6e-3);
}

TEST_CASE("BatchedMatmul CPU handles the TRANSPOSED views both upstream call sites pass") {
  // Upstream never passes contiguous operands here:
  //   :745-748  mqa_q_nope = mqa_q_nope.transpose(0, 1)     (B,N,P) -> (N,B,P)
  //   :1026     x = x.view(-1, N, L).transpose(0, 1)        (B,N,L) -> (N,B,L)
  //   :1034     out=out.transpose(0, 1)                     writes into (B,N,V)
  const int64_t n_heads = 5, batch = 6, kdim = 9, ndim = 4;
  auto src = RandF32(static_cast<size_t>(batch * n_heads * kdim), 33u);   // [B,N,K]
  auto w = RandF32(static_cast<size_t>(n_heads * kdim * ndim), 44u);      // [N,K,V]
  auto srcb = ToBf16(src), wb = ToBf16(w);
  const auto srcf = FromBf16(srcb), wf = FromBf16(wb);

  // OUT is a transposed view of a [B,N,V] buffer, exactly like `_v_up_proj`.
  std::vector<uint16_t> outbuf(static_cast<size_t>(batch * n_heads * ndim), 0x7FC0);
  Queue q = CpuQ();
  Tensor ta = View3(srcb.data(), DType::kBF16, Cpu(), 0, n_heads, batch, kdim,
                    /*s0=*/kdim, /*s1=*/n_heads * kdim, /*s2=*/1);
  Tensor tb = Contig(wb.data(), DType::kBF16, Cpu(), {n_heads, kdim, ndim});
  Tensor to = View3(outbuf.data(), DType::kBF16, Cpu(), 0, n_heads, batch, ndim,
                    /*s0=*/ndim, /*s1=*/n_heads * ndim, /*s2=*/1);
  vt::BatchedMatmul(q, to, ta, tb);

  const auto ref = RefBmm(srcf, wf, n_heads, batch, kdim, ndim, kdim, n_heads * kdim,
                          kdim * ndim, ndim);
  for (int64_t h = 0; h < n_heads; ++h) {
    for (int64_t bi = 0; bi < batch; ++bi) {
      for (int64_t j = 0; j < ndim; ++j) {
        const double got =
            vt::BF16ToF32(outbuf[static_cast<size_t>((bi * n_heads + h) * ndim + j)]);
        REQUIRE(!std::isnan(got));
        const double want = ref[static_cast<size_t>((h * batch + bi) * ndim + j)];
        CHECK(std::abs(got - want) <= 5e-3 * std::max(1.0, std::abs(want)));
      }
    }
  }
}

TEST_CASE("BatchedMatmul rejects malformed shapes/strides rather than mis-computing") {
  Queue q = CpuQ();
  std::vector<uint16_t> buf(4096, 0);
  Tensor a = Contig(buf.data(), DType::kBF16, Cpu(), {2, 3, 4});
  Tensor b = Contig(buf.data(), DType::kBF16, Cpu(), {2, 5, 4});  // inner mismatch
  Tensor o = Contig(buf.data(), DType::kBF16, Cpu(), {2, 3, 4});
  CHECK_THROWS(vt::BatchedMatmul(q, o, a, b));
  Tensor a2 = Contig(buf.data(), DType::kBF16, Cpu(), {2, 3, 4});
  Tensor b2 = Contig(buf.data(), DType::kBF16, Cpu(), {3, 4, 4});  // batch mismatch
  CHECK_THROWS(vt::BatchedMatmul(q, o, a2, b2));
  // Innermost stride must be 1 (the one layout constraint upstream also asserts).
  Tensor a3 = View3(buf.data(), DType::kBF16, Cpu(), 0, 2, 3, 4, 24, 8, 2);
  Tensor b3 = Contig(buf.data(), DType::kBF16, Cpu(), {2, 4, 4});
  CHECK_THROWS(vt::BatchedMatmul(q, o, a3, b3));
}

// ────────────────────────────────────────────────────────────────────────────
// tests/kernels/test_concat_mla_q.py:22-34 `test_concat_mla_q_contiguous`, with
// its exact (atol=0, rtol=0) comparison.
TEST_CASE("ConcatMlaNopeRope reproduces torch.cat EXACTLY (contiguous, V3 q shape)") {
  const int64_t tokens = 16, heads = 128;
  auto nope = RandF32(static_cast<size_t>(tokens * heads * kKvLoraRank), 55u);
  auto rope = RandF32(static_cast<size_t>(tokens * heads * kQkRope), 66u);
  auto nb = ToBf16(nope), rb = ToBf16(rope);
  std::vector<uint16_t> out(
      static_cast<size_t>(tokens * heads * (kKvLoraRank + kQkRope)), 0x7FC0);

  Queue q = CpuQ();
  Tensor tn = Contig(nb.data(), DType::kBF16, Cpu(), {tokens, heads, kKvLoraRank});
  Tensor tr = Contig(rb.data(), DType::kBF16, Cpu(), {tokens, heads, kQkRope});
  Tensor to =
      Contig(out.data(), DType::kBF16, Cpu(), {tokens, heads, kKvLoraRank + kQkRope});
  vt::ConcatMlaNopeRope(q, to, tn, tr);

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      const size_t o = static_cast<size_t>((t * heads + h) * (kKvLoraRank + kQkRope));
      for (int64_t d = 0; d < kKvLoraRank; ++d) {
        REQUIRE(out[o + static_cast<size_t>(d)] ==
                nb[static_cast<size_t>((t * heads + h) * kKvLoraRank + d)]);
      }
      for (int64_t d = 0; d < kQkRope; ++d) {
        REQUIRE(out[o + static_cast<size_t>(kKvLoraRank + d)] ==
                rb[static_cast<size_t>((t * heads + h) * kQkRope + d)]);
      }
    }
  }
}

// tests/kernels/test_concat_mla_q.py:37-63 `test_concat_mla_q_transposed_nope` —
// the case that exists BECAUSE the real nope operand is the transposed output of
// the W_UK bmm and is non-contiguous.
TEST_CASE("ConcatMlaNopeRope handles the NON-CONTIGUOUS transposed bmm nope operand") {
  const int64_t tokens = 9, heads = 16, nope_dim = kKvLoraRank;
  // [N, B, L] storage; the operand is its [B, N, L] transpose.
  auto raw = RandF32(static_cast<size_t>(heads * tokens * nope_dim), 77u);
  auto rope = RandF32(static_cast<size_t>(tokens * heads * kQkRope), 88u);
  auto rawb = ToBf16(raw), rb = ToBf16(rope);
  std::vector<uint16_t> out(
      static_cast<size_t>(tokens * heads * (nope_dim + kQkRope)), 0x7FC0);

  Queue q = CpuQ();
  Tensor tn = View3(rawb.data(), DType::kBF16, Cpu(), 0, tokens, heads, nope_dim,
                    /*s0=*/nope_dim, /*s1=*/tokens * nope_dim, /*s2=*/1);
  Tensor tr = Contig(rb.data(), DType::kBF16, Cpu(), {tokens, heads, kQkRope});
  Tensor to = Contig(out.data(), DType::kBF16, Cpu(), {tokens, heads, nope_dim + kQkRope});
  vt::ConcatMlaNopeRope(q, to, tn, tr);

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      const size_t o = static_cast<size_t>((t * heads + h) * (nope_dim + kQkRope));
      for (int64_t d = 0; d < nope_dim; ++d) {
        REQUIRE(out[o + static_cast<size_t>(d)] ==
                rawb[static_cast<size_t>((h * tokens + t) * nope_dim + d)]);
      }
      for (int64_t d = 0; d < kQkRope; ++d) {
        REQUIRE(out[o + static_cast<size_t>(nope_dim + d)] ==
                rb[static_cast<size_t>((t * heads + h) * kQkRope + d)]);
      }
    }
  }
}

// The OTHER upstream site: `_concat_k_nope_k_pe` (mla_attention.py:2063-2092),
// where k_pe carries ONE head and is BROADCAST across all of them.
TEST_CASE("ConcatMlaNopeRope broadcasts the single k_pe head across every head") {
  const int64_t tokens = 11, heads = 16;
  auto nope = RandF32(static_cast<size_t>(tokens * heads * kQkNope), 99u);
  auto rope = RandF32(static_cast<size_t>(tokens * 1 * kQkRope), 101u);
  auto nb = ToBf16(nope), rb = ToBf16(rope);
  std::vector<uint16_t> out(static_cast<size_t>(tokens * heads * (kQkNope + kQkRope)),
                            0x7FC0);

  Queue q = CpuQ();
  Tensor tn = Contig(nb.data(), DType::kBF16, Cpu(), {tokens, heads, kQkNope});
  Tensor tr = Contig(rb.data(), DType::kBF16, Cpu(), {tokens, 1, kQkRope});
  Tensor to = Contig(out.data(), DType::kBF16, Cpu(), {tokens, heads, kQkNope + kQkRope});
  vt::ConcatMlaNopeRope(q, to, tn, tr);

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      const size_t o = static_cast<size_t>((t * heads + h) * (kQkNope + kQkRope));
      for (int64_t d = 0; d < kQkNope; ++d) {
        REQUIRE(out[o + static_cast<size_t>(d)] ==
                nb[static_cast<size_t>((t * heads + h) * kQkNope + d)]);
      }
      for (int64_t d = 0; d < kQkRope; ++d) {
        // EVERY head must see the SAME single rope row.
        REQUIRE(out[o + static_cast<size_t>(kQkNope + d)] ==
                rb[static_cast<size_t>(t * kQkRope + d)]);
      }
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("CUDA BatchedMatmul matches the oracle at BOTH absorption shapes and is bit-exact") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);

  struct Case {
    int64_t heads, batch, k, n;
    const char* what;
  };
  // (1) V2-Lite q-side absorption, (2) V2-Lite v-up-projection,
  // (3) V3 q-side absorption (128 heads), (4) a single-token decode step.
  const Case cases[] = {
      {16, 8, kQkNope, kKvLoraRank, "V2-Lite W_UK fold"},
      {16, 8, kKvLoraRank, kVHeadDim, "V2-Lite W_UV un-projection"},
      {128, 3, kQkNope, kKvLoraRank, "V3 W_UK fold"},
      {16, 1, kKvLoraRank, kVHeadDim, "single-token decode v-up"},
  };
  for (const Case& c : cases) {
    CAPTURE(c.what);
    auto a = RandF32(static_cast<size_t>(c.heads * c.batch * c.k), 1234u);
    auto w = RandF32(static_cast<size_t>(c.heads * c.k * c.n), 4321u);
    auto ab = ToBf16(a), wb = ToBf16(w);
    const auto af = FromBf16(ab), wf = FromBf16(wb);

    DeviceTensor da(b, qg.q, DType::kBF16, {c.heads, c.batch, c.k}, ab.data());
    DeviceTensor dw(b, qg.q, DType::kBF16, {c.heads, c.k, c.n}, wb.data());
    // NaN-poison the output so an unwritten element is caught.
    std::vector<uint16_t> poison(static_cast<size_t>(c.heads * c.batch * c.n), 0x7FC0);
    DeviceTensor dout(b, qg.q, DType::kBF16, {c.heads, c.batch, c.n}, poison.data());
    vt::BatchedMatmul(qg.q, dout.tensor(), da.tensor(), dw.tensor());
    std::vector<uint16_t> got(poison.size());
    dout.Download(qg.q, got.data());

    const auto ref = RefBmm(af, wf, c.heads, c.batch, c.k, c.n, c.batch * c.k, c.k,
                            c.k * c.n, c.n);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double v = vt::BF16ToF32(got[i]);
      REQUIRE(!std::isnan(v));
      worst = std::max(worst, std::abs(v - ref[i]) / std::max(1e-2, std::abs(ref[i])));
    }
    CHECK(worst < 1e-2);

    // Run-to-run BIT-exactness (the house determinism rule).
    for (int rep = 0; rep < 3; ++rep) {
      DeviceTensor again(b, qg.q, DType::kBF16, {c.heads, c.batch, c.n}, poison.data());
      vt::BatchedMatmul(qg.q, again.tensor(), da.tensor(), dw.tensor());
      std::vector<uint16_t> got2(poison.size());
      again.Download(qg.q, got2.data());
      REQUIRE(got2 == got);
    }
  }
}

TEST_CASE("CUDA BatchedMatmul agrees with the CPU reference on the TRANSPOSED views") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  const int64_t heads = 16, batch = 5, kdim = kKvLoraRank, ndim = kVHeadDim;

  auto raw = RandF32(static_cast<size_t>(batch * heads * kdim), 2468u);  // [B,N,L]
  auto w = RandF32(static_cast<size_t>(heads * kdim * ndim), 8642u);
  auto rawb = ToBf16(raw), wb = ToBf16(w);

  // CPU: the same transposed-in / transposed-out layout the block uses.
  std::vector<uint16_t> cpu_out(static_cast<size_t>(batch * heads * ndim), 0x7FC0);
  Queue cq = CpuQ();
  Tensor ca = View3(rawb.data(), DType::kBF16, Cpu(), 0, heads, batch, kdim, kdim,
                    heads * kdim, 1);
  Tensor cw = Contig(wb.data(), DType::kBF16, Cpu(), {heads, kdim, ndim});
  Tensor co = View3(cpu_out.data(), DType::kBF16, Cpu(), 0, heads, batch, ndim, ndim,
                    heads * ndim, 1);
  vt::BatchedMatmul(cq, co, ca, cw);

  DeviceTensor draw(b, qg.q, DType::kBF16, {batch, heads, kdim}, rawb.data());
  DeviceTensor dw(b, qg.q, DType::kBF16, {heads, kdim, ndim}, wb.data());
  std::vector<uint16_t> poison(cpu_out.size(), 0x7FC0);
  DeviceTensor dout(b, qg.q, DType::kBF16, {batch, heads, ndim}, poison.data());
  Tensor ga = View3(draw.tensor().data, DType::kBF16, Gpu(), 0, heads, batch, kdim, kdim,
                    heads * kdim, 1);
  Tensor go = View3(dout.tensor().data, DType::kBF16, Gpu(), 0, heads, batch, ndim, ndim,
                    heads * ndim, 1);
  vt::BatchedMatmul(qg.q, go, ga, dw.tensor());
  std::vector<uint16_t> gpu_out(poison.size());
  dout.Download(qg.q, gpu_out.data());

  double worst = 0.0;
  for (size_t i = 0; i < gpu_out.size(); ++i) {
    const double g = vt::BF16ToF32(gpu_out[i]), c = vt::BF16ToF32(cpu_out[i]);
    REQUIRE(!std::isnan(g));
    worst = std::max(worst, std::abs(g - c) / std::max(1e-2, std::abs(c)));
  }
  // CPU accumulates sequentially over K; cuBLASLt splits it. Both are f32
  // accumulations of the same products, so they agree to bf16 rounding.
  CHECK(worst < 2e-2);
}

TEST_CASE("CUDA ConcatMlaNopeRope is byte-identical to the CPU reference, both operand forms") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);

  SUBCASE("decode q: transposed nope + per-head rope") {
    const int64_t tokens = 9, heads = 16;
    auto raw = RandF32(static_cast<size_t>(heads * tokens * kKvLoraRank), 1357u);
    auto rope = RandF32(static_cast<size_t>(tokens * heads * kQkRope), 2468u);
    auto rawb = ToBf16(raw), rb = ToBf16(rope);
    const size_t n_out = static_cast<size_t>(tokens * heads * (kKvLoraRank + kQkRope));
    std::vector<uint16_t> cpu_out(n_out, 0x7FC0), poison(n_out, 0x7FC0);

    Queue cq = CpuQ();
    Tensor cn = View3(rawb.data(), DType::kBF16, Cpu(), 0, tokens, heads, kKvLoraRank,
                      kKvLoraRank, tokens * kKvLoraRank, 1);
    Tensor cr = Contig(rb.data(), DType::kBF16, Cpu(), {tokens, heads, kQkRope});
    Tensor co = Contig(cpu_out.data(), DType::kBF16, Cpu(),
                       {tokens, heads, kKvLoraRank + kQkRope});
    vt::ConcatMlaNopeRope(cq, co, cn, cr);

    DeviceTensor draw(b, qg.q, DType::kBF16, {heads, tokens, kKvLoraRank}, rawb.data());
    DeviceTensor drope(b, qg.q, DType::kBF16, {tokens, heads, kQkRope}, rb.data());
    DeviceTensor dout(b, qg.q, DType::kBF16, {tokens, heads, kKvLoraRank + kQkRope},
                      poison.data());
    Tensor gn = View3(draw.tensor().data, DType::kBF16, Gpu(), 0, tokens, heads,
                      kKvLoraRank, kKvLoraRank, tokens * kKvLoraRank, 1);
    vt::ConcatMlaNopeRope(qg.q, dout.tensor(), gn, drope.tensor());
    std::vector<uint16_t> gpu_out(n_out);
    dout.Download(qg.q, gpu_out.data());
    REQUIRE(gpu_out == cpu_out);  // a concat is a pure copy: EXACT, atol=rtol=0
  }

  SUBCASE("prefill k: contiguous nope + BROADCAST single-head rope") {
    const int64_t tokens = 13, heads = 16;
    auto nope = RandF32(static_cast<size_t>(tokens * heads * kQkNope), 3690u);
    auto rope = RandF32(static_cast<size_t>(tokens * kQkRope), 963u);
    auto nb = ToBf16(nope), rb = ToBf16(rope);
    const size_t n_out = static_cast<size_t>(tokens * heads * (kQkNope + kQkRope));
    std::vector<uint16_t> cpu_out(n_out, 0x7FC0), poison(n_out, 0x7FC0);

    Queue cq = CpuQ();
    Tensor cn = Contig(nb.data(), DType::kBF16, Cpu(), {tokens, heads, kQkNope});
    Tensor cr = Contig(rb.data(), DType::kBF16, Cpu(), {tokens, 1, kQkRope});
    Tensor co =
        Contig(cpu_out.data(), DType::kBF16, Cpu(), {tokens, heads, kQkNope + kQkRope});
    vt::ConcatMlaNopeRope(cq, co, cn, cr);

    DeviceTensor dn(b, qg.q, DType::kBF16, {tokens, heads, kQkNope}, nb.data());
    DeviceTensor dr(b, qg.q, DType::kBF16, {tokens, 1, kQkRope}, rb.data());
    DeviceTensor dout(b, qg.q, DType::kBF16, {tokens, heads, kQkNope + kQkRope},
                      poison.data());
    vt::ConcatMlaNopeRope(qg.q, dout.tensor(), dn.tensor(), dr.tensor());
    std::vector<uint16_t> gpu_out(n_out);
    dout.Download(qg.q, gpu_out.data());
    REQUIRE(gpu_out == cpu_out);
  }
}
