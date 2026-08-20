// vllm.cpp original (vt runtime). Unit tests for vt::Dflash2SelectorEdges — the
// DFlash2 CANDIDATE SELECTOR's edge lattice (SPEC-DFLASH2 W3, #1314).
//
// BEYOND-PIN. The reference is `_score_edges`
// (vllm/model_executor/models/qwen3_dflash2.py:208-228 @ vllm-project/vllm#52816
// head `66e5414c6d75a8529473d977f7458c140bbab8a0`), and the sequential
// expectation below is upstream's OWN reference loop from
// `tests/v1/spec_decode/test_dflash2.py::test_selector_edges_match_sequential_reference`
// at that head, transcribed rather than reinvented:
//
//   for step in range(steps):
//       pred = (anchors[:, None].expand(-1, top_k) if step == 0
//               else candidate_ids[:, step - 1])
//       expected[:, step] = unary[:, step, None] + torch.einsum(
//           "bpr,bcr->bpc",
//           predecessors[pred] * hidden[:, step, None],
//           successors[candidate_ids[:, step]])
//
// The PR head MOVED under this row (#1404); `_score_edges` and its test are
// BYTE-IDENTICAL at `19c93519` and at `66e5414c`, so the port is unaffected and
// this file cites the new one.
//
// WHAT IS GATED, and why each case exists.
//
//  * UPSTREAM'S OWN SEQUENTIAL REFERENCE, at upstream's own parameters (batch 2,
//    steps 4, top_k 3, rank 5, vocab 17). That reference is independent in
//    PROVENANCE -- it is upstream's per-step loop, not a rearrangement of our
//    kernel -- and it shares the ascending-rank summation order, which is why
//    the f32 assertion is bit-exact rather than an envelope.
//  * THE ANCHOR ARM. At step 0 every one of the K predecessor slots is the
//    request's verified anchor token, so all K predecessor ROWS of step 0 must
//    be identical to each other; at every later step they must not be. Getting
//    this wrong is invisible to a token gate: the draft still emits the target's
//    tokens and only acceptance falls.
//  * THE PREDECESSOR INDEXING. Swapping two of step l-1's candidate ids must
//    swap exactly those two predecessor rows of step l's block. An
//    implementation that indexed the predecessor codebook with THIS step's ids
//    (a plausible off-by-one, since the two tensors have the same shape) passes
//    every shape check and produces finite, plausible scores.
//  * THE UNARY BROADCAST. `unary_logits[:, :, None]` is a per-CHILD bias, not a
//    per-edge or per-predecessor one. Adding a constant to one child's unary
//    must move that child's COLUMN of the block by exactly that constant and
//    nothing else.
//  * THE bf16 ROUNDING PLACEMENT, with hand-written literals. Upstream
//    materializes TWO bf16 tensors in this chain -- `predecessors * hidden` and
//    the einsum's own output -- and this op rounds at those two points and
//    nowhere else. On f32 both roundings are the IDENTITY by construction, so no
//    f32 case above can see the policy, which is exactly the hole W2's second
//    fresh review found in the convolution's evidence (spec `## Owed` O6). The
//    literals below are chosen so that the three candidate placements answer
//    DIFFERENTLY: ours 7.71875, round-once-at-the-end 7.6875, and no rounding at
//    all 7.699830055236816.
//  * CUDA vs CPU, WITHIN AN ENVELOPE and not bit-for-bit. The rank contraction
//    is a REDUCTION, so the CUDA warp-shuffle tree sums in a different order than
//    the CPU reference's serial loop. This differs from
//    vt::DFlashGroupedConv, which is elementwise and IS specified bit-identical;
//    the distinction is stated here rather than inherited by analogy. That case
//    is written and has NEVER RUN on this host (no `nvcc`, so it reports `no CUDA
//    backend; skipping`); it is `## Owed` O10 of the row's spec and is not
//    counted as coverage here.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Dflash2SelectorEdgesArgs;
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
Tensor I64(std::vector<int64_t>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kI64, Cpu(), shape);
}
Tensor Bf16(std::vector<uint16_t>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kBF16, Cpu(), shape);
}
std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::F32ToBF16(v[i]);
  return o;
}

Dflash2SelectorEdgesArgs Args(int64_t k) {
  Dflash2SelectorEdgesArgs a;
  a.top_k = k;
  return a;
}

// Deterministic LCG in [-2,2); avoids <random> divergence across libstdc++.
std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}
std::vector<int64_t> RandIds(size_t n, int64_t vocab, uint32_t seed) {
  std::vector<int64_t> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<int64_t>((s >> 8) % static_cast<uint32_t>(vocab));
  }
  return v;
}

struct Fixture {
  int64_t B = 2, L = 4, K = 3, R = 5, V = 17;
  std::vector<float> pred, succ, hidden, unary;
  std::vector<int64_t> cand, anchors;

  Fixture() {
    pred = RandF32(static_cast<size_t>(V * R), 0x11u);
    succ = RandF32(static_cast<size_t>(V * R), 0x22u);
    hidden = RandF32(static_cast<size_t>(B * L * R), 0x33u);
    unary = RandF32(static_cast<size_t>(B * L * K), 0x44u);
    cand = RandIds(static_cast<size_t>(B * L * K), V, 0x55u);
    anchors = RandIds(static_cast<size_t>(B), V, 0x66u);
  }

  std::vector<float> Run() {
    std::vector<float> out(static_cast<size_t>(B * L * K * K), 0.0f);
    Tensor tp = F32(pred, {V, R});
    Tensor ts = F32(succ, {V, R});
    Tensor tc = I64(cand, {B, L, K});
    Tensor tu = F32(unary, {B, L, K});
    Tensor th = F32(hidden, {B, L, R});
    Tensor ta = I64(anchors, {B});
    Tensor to = F32(out, {B, L, K, K});
    Queue q = Q();
    vt::Dflash2SelectorEdges(q, to, tp, ts, tc, tu, th, ta, Args(K));
    return out;
  }

  // UPSTREAM's own per-step reference loop, transcribed from its test.
  std::vector<float> Reference() const {
    std::vector<float> out(static_cast<size_t>(B * L * K * K), 0.0f);
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t l = 0; l < L; ++l) {
        for (int64_t p = 0; p < K; ++p) {
          const int64_t pid = (l == 0) ? anchors[static_cast<size_t>(b)]
                                       : cand[static_cast<size_t>((b * L + l - 1) * K + p)];
          for (int64_t c = 0; c < K; ++c) {
            const int64_t cid = cand[static_cast<size_t>((b * L + l) * K + c)];
            float acc = 0.0f;
            for (int64_t r = 0; r < R; ++r)
              acc += (pred[static_cast<size_t>(pid * R + r)] *
                      hidden[static_cast<size_t>((b * L + l) * R + r)]) *
                     succ[static_cast<size_t>(cid * R + r)];
            out[static_cast<size_t>(((b * L + l) * K + p) * K + c)] =
                unary[static_cast<size_t>((b * L + l) * K + c)] + acc;
          }
        }
      }
    }
    return out;
  }
};

}  // namespace

TEST_CASE("dflash2-selector-edges: upstream's sequential reference, at its own parameters") {
  Fixture f;  // batch 2, steps 4, top_k 3, rank 5, vocab 17 — upstream's test.
  const std::vector<float> got = f.Run();
  const std::vector<float> want = f.Reference();
  REQUIRE(got.size() == want.size());
  for (size_t i = 0; i < want.size(); ++i) {
    INFO("edge ", i);
    CHECK(got[i] == want[i]);
  }
}

TEST_CASE("dflash2-selector-edges: the ANCHOR is every predecessor slot at step 0") {
  Fixture f;
  const std::vector<float> got = f.Run();
  const int64_t KK = f.K * f.K;
  for (int64_t b = 0; b < f.B; ++b) {
    // Step 0: all K predecessor rows come from the SAME anchor token, so they
    // must be identical to each other.
    for (int64_t p = 1; p < f.K; ++p) {
      for (int64_t c = 0; c < f.K; ++c) {
        INFO("b ", b, " p ", p, " c ", c);
        CHECK(got[static_cast<size_t>((b * f.L) * KK + p * f.K + c)] ==
              got[static_cast<size_t>((b * f.L) * KK + 0 * f.K + c)]);
      }
    }
  }
  // The instrument's own precondition: at a LATER step the predecessor rows come
  // from distinct candidate ids, so they must NOT all be identical -- otherwise
  // the assertion above would pass on an implementation that ignored the
  // predecessor entirely.
  bool later_rows_differ = false;
  for (int64_t b = 0; b < f.B && !later_rows_differ; ++b)
    for (int64_t p = 1; p < f.K && !later_rows_differ; ++p)
      for (int64_t c = 0; c < f.K && !later_rows_differ; ++c)
        later_rows_differ =
            got[static_cast<size_t>((b * f.L + 1) * KK + p * f.K + c)] !=
            got[static_cast<size_t>((b * f.L + 1) * KK + c)];
  CHECK(later_rows_differ);

  // And the anchor must REACH the scores: changing it moves step 0 and leaves
  // every later step byte-for-byte alone.
  Fixture g;
  g.anchors[0] = (g.anchors[0] + 1) % g.V;
  const std::vector<float> moved = g.Run();
  bool step0_moved = false;
  for (int64_t i = 0; i < KK; ++i)
    step0_moved = step0_moved || moved[static_cast<size_t>(i)] != got[static_cast<size_t>(i)];
  CHECK(step0_moved);
  for (int64_t l = 1; l < f.L; ++l)
    for (int64_t i = 0; i < KK; ++i) {
      INFO("later step ", l, " edge ", i);
      CHECK(moved[static_cast<size_t>(l * KK + i)] == got[static_cast<size_t>(l * KK + i)]);
    }
}

TEST_CASE("dflash2-selector-edges: step l reads step l-1's candidates as predecessors") {
  // Swapping two of step 0's candidate ids must swap exactly those two
  // PREDECESSOR ROWS of step 1's block. An implementation that indexed the
  // predecessor codebook with THIS step's ids -- the plausible off-by-one, since
  // the two tensors have identical shape -- would leave step 1 unchanged here.
  Fixture f;
  const std::vector<float> base = f.Run();
  Fixture g;
  std::swap(g.cand[0], g.cand[1]);  // b=0, l=0, slots 0 and 1
  const std::vector<float> got = g.Run();
  const int64_t KK = f.K * f.K;
  for (int64_t c = 0; c < f.K; ++c) {
    INFO("child ", c);
    CHECK(got[static_cast<size_t>(1 * KK + 0 * f.K + c)] ==
          base[static_cast<size_t>(1 * KK + 1 * f.K + c)]);
    CHECK(got[static_cast<size_t>(1 * KK + 1 * f.K + c)] ==
          base[static_cast<size_t>(1 * KK + 0 * f.K + c)]);
  }
  // Precondition: the two rows were not equal to begin with, so the swap is a
  // real observation rather than a tautology.
  bool rows_differed = false;
  for (int64_t c = 0; c < f.K; ++c)
    rows_differed = rows_differed || base[static_cast<size_t>(1 * KK + c)] !=
                                         base[static_cast<size_t>(1 * KK + f.K + c)];
  CHECK(rows_differed);
}

TEST_CASE("dflash2-selector-edges: unary is a per-CHILD bias, broadcast over predecessors") {
  Fixture f;
  const std::vector<float> base = f.Run();
  Fixture g;
  const float delta = 0.25f;  // exact in f32, so the assertion is exact
  g.unary[1] += delta;        // b=0, l=0, child 1
  const std::vector<float> got = g.Run();
  const int64_t KK = f.K * f.K;
  for (int64_t p = 0; p < f.K; ++p) {
    for (int64_t c = 0; c < f.K; ++c) {
      const size_t i = static_cast<size_t>(p * f.K + c);
      INFO("p ", p, " c ", c);
      CHECK(got[i] == (c == 1 ? base[i] + delta : base[i]));
    }
  }
  // Nothing outside that block moved.
  for (size_t i = static_cast<size_t>(KK); i < base.size(); ++i) CHECK(got[i] == base[i]);
}

TEST_CASE("dflash2-selector-edges: the bf16 rounding placement, hand-computed") {
  // ONE request, ONE step, ONE candidate, rank 4. The predecessor is the anchor
  // (step 0), so this case also exercises the anchor path in bf16.
  //
  // Every literal below is bf16-EXACT, so the only rounding in play is the two
  // this op performs. Upstream materializes `predecessors * hidden` as a bf16
  // tensor and the einsum's output as a bf16 tensor, and nothing else in the
  // chain, which gives three candidate placements three different answers:
  //
  //   pred    = [1.2265625, 1.1015625, 1.078125,  1.2578125]
  //   hidden  = [1.265625,  1.0390625, 1.1796875, 1.265625 ]
  //   succ    = [1.125,     1.421875,  1.2578125, 1.3984375]
  //
  //   bf16(pred*hidden)      = [1.5546875, 1.1484375, 1.2734375, 1.59375]
  //   sum(that * succ)  f32  = 7.21246337890625   -> bf16 7.21875
  //   sum(pred*hidden*succ)  = 7.199830055236816  -> bf16 7.1875
  //
  //   OURS               unary + bf16(sum of ROUNDED gated products) = 7.71875
  //   round once at end  unary + bf16(sum of EXACT   gated products) = 7.6875
  //   no rounding at all unary +      sum of EXACT   gated products  = 7.699830055236816
  //
  // A file that ran only in f32 would assert none of this: there the two
  // roundings are the identity, and every placement answers the same.
  const int64_t V = 2, R = 4, B = 1, L = 1, K = 1;
  std::vector<float> pred_f(static_cast<size_t>(V * R), 0.0f);
  std::vector<float> succ_f(static_cast<size_t>(V * R), 0.0f);
  const float a[4] = {1.2265625f, 1.1015625f, 1.078125f, 1.2578125f};
  const float h[4] = {1.265625f, 1.0390625f, 1.1796875f, 1.265625f};
  const float sv[4] = {1.125f, 1.421875f, 1.2578125f, 1.3984375f};
  for (int r = 0; r < 4; ++r) {
    pred_f[static_cast<size_t>(r)] = a[r];               // codebook row 0 == the anchor
    succ_f[static_cast<size_t>(R + r)] = sv[r];          // codebook row 1 == the candidate
  }
  std::vector<uint16_t> pred = ToBf16(pred_f);
  std::vector<uint16_t> succ = ToBf16(succ_f);
  std::vector<uint16_t> hidden = ToBf16({h[0], h[1], h[2], h[3]});
  std::vector<float> unary = {0.5f};
  std::vector<int64_t> cand = {1};
  std::vector<int64_t> anchors = {0};
  std::vector<float> out(1, 0.0f);

  Tensor tp = Bf16(pred, {V, R});
  Tensor ts = Bf16(succ, {V, R});
  Tensor th = Bf16(hidden, {B, L, R});
  Tensor tc = I64(cand, {B, L, K});
  Tensor tu = F32(unary, {B, L, K});
  Tensor ta = I64(anchors, {B});
  Tensor to = F32(out, {B, L, K, K});
  Queue q = Q();
  vt::Dflash2SelectorEdges(q, to, tp, ts, tc, tu, th, ta, Args(K));

  CHECK(out[0] == 7.71875f);        // the placement this op implements
  CHECK(out[0] != 7.6875f);         // NOT rounded once at the end
  CHECK(out[0] != 7.699830055236816f);  // NOT unrounded
}

TEST_CASE("dflash2-selector-edges: the wrapper refuses what it cannot answer") {
  Fixture f;
  std::vector<float> out(static_cast<size_t>(f.B * f.L * f.K * f.K), 0.0f);
  Tensor tp = F32(f.pred, {f.V, f.R});
  Tensor ts = F32(f.succ, {f.V, f.R});
  Tensor tc = I64(f.cand, {f.B, f.L, f.K});
  Tensor tu = F32(f.unary, {f.B, f.L, f.K});
  Tensor th = F32(f.hidden, {f.B, f.L, f.R});
  Tensor ta = I64(f.anchors, {f.B});
  Tensor to = F32(out, {f.B, f.L, f.K, f.K});
  Queue q = Q();
  CHECK_NOTHROW(vt::Dflash2SelectorEdges(q, to, tp, ts, tc, tu, th, ta, Args(f.K)));
  // A candidate id outside the codebook is an id-space error (a missing
  // org-vocab rebase is how it happens), and upstream would index out of bounds.
  Fixture g;
  g.cand[0] = g.V;
  std::vector<float> gout(static_cast<size_t>(g.B * g.L * g.K * g.K), 0.0f);
  Tensor gp = F32(g.pred, {g.V, g.R});
  Tensor gs = F32(g.succ, {g.V, g.R});
  Tensor gc = I64(g.cand, {g.B, g.L, g.K});
  Tensor gu = F32(g.unary, {g.B, g.L, g.K});
  Tensor gh = F32(g.hidden, {g.B, g.L, g.R});
  Tensor ga = I64(g.anchors, {g.B});
  Tensor go = F32(gout, {g.B, g.L, g.K, g.K});
  CHECK_THROWS_WITH_AS(
      vt::Dflash2SelectorEdges(q, go, gp, gs, gc, gu, gh, ga, Args(g.K)),
      doctest::Contains("candidate token id outside the codebook"), std::runtime_error);
  // And a top_k that does not match the candidate tensor's last dim.
  //
  // MATCHED ON THE MESSAGE, because these refusals overlap. A `top_k` that does
  // not match `candidate_ids` also fails the `scores` lattice check two lines
  // below it, so a bare `CHECK_THROWS` here is answered by whichever guard is
  // left standing and cannot tell the two apart -- the shape W4's fresh review
  // measured on the sibling path-walk suite.
  CHECK_THROWS_WITH_AS(
      vt::Dflash2SelectorEdges(q, to, tp, ts, tc, tu, th, ta, Args(f.K + 1)),
      doctest::Contains("candidate_ids last dim must be top_k"), std::runtime_error);

  // The OUTPUT LATTICE's own two trailing axes, each on its own. They are the
  // PREDECESSOR axis and the CHILD axis, they have the same extent, and nothing
  // else in this wrapper checks them: with `candidate_ids` correct, a lattice
  // that is [B,L,K,K-1] or [B,L,K-1,K] passes every other guard here.
  //
  // Both views are built at the wrong extent WITH MATCHING STRIDES rather than
  // by mutating a `shape` field of a tensor `Contig` produced. A mutated field
  // desynchronises the strides, `Tensor::IsContiguous()` turns false, and
  // `dflash2-selector-edges: contiguous tensors required` throws first -- so the
  // case would pass with the shape guard deleted entirely.
  {
    Tensor child = F32(out, {f.B, f.L, f.K, f.K - 1});
    REQUIRE(child.IsContiguous());
    CHECK_THROWS_WITH_AS(
        vt::Dflash2SelectorEdges(q, child, tp, ts, tc, tu, th, ta, Args(f.K)),
        doctest::Contains("scores must be [B,L,K,K]"), std::runtime_error);
    Tensor pred = F32(out, {f.B, f.L, f.K - 1, f.K});
    REQUIRE(pred.IsContiguous());
    CHECK_THROWS_WITH_AS(
        vt::Dflash2SelectorEdges(q, pred, tp, ts, tc, tu, th, ta, Args(f.K)),
        doctest::Contains("scores must be [B,L,K,K]"), std::runtime_error);
  }
}

// ===========================================================================
// CUDA parity, WITHIN AN ENVELOPE. See the file header for why this op is not
// specified bit-identical across backends and vt::DFlashGroupedConv is.
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

}  // namespace

TEST_CASE("dflash2-selector-edges: CUDA vs CPU within the f32 envelope") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping CUDA dflash2-selector-edges parity");
    return;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  // The PUBLISHED shape (rank 256, K 16) plus upstream's own small one, so the
  // warp-shuffle contraction is exercised at both a multi-iteration rank and a
  // sub-warp one.
  struct Case {
    int64_t B, L, K, R, V;
  };
  const Case cases[] = {{2, 4, 3, 5, 17}, {2, 7, 16, 256, 512}};
  for (const Case& cs : cases) {
    Fixture f;
    f.B = cs.B;
    f.L = cs.L;
    f.K = cs.K;
    f.R = cs.R;
    f.V = cs.V;
    f.pred = RandF32(static_cast<size_t>(cs.V * cs.R), 0x11u);
    f.succ = RandF32(static_cast<size_t>(cs.V * cs.R), 0x22u);
    f.hidden = RandF32(static_cast<size_t>(cs.B * cs.L * cs.R), 0x33u);
    f.unary = RandF32(static_cast<size_t>(cs.B * cs.L * cs.K), 0x44u);
    f.cand = RandIds(static_cast<size_t>(cs.B * cs.L * cs.K), cs.V, 0x55u);
    f.anchors = RandIds(static_cast<size_t>(cs.B), cs.V, 0x66u);
    const std::vector<float> cpu = f.Run();

    DeviceTensor dp(b, g.q, DType::kF32, {cs.V, cs.R}, f.pred.data());
    DeviceTensor ds(b, g.q, DType::kF32, {cs.V, cs.R}, f.succ.data());
    DeviceTensor dc(b, g.q, DType::kI64, {cs.B, cs.L, cs.K}, f.cand.data());
    DeviceTensor du(b, g.q, DType::kF32, {cs.B, cs.L, cs.K}, f.unary.data());
    DeviceTensor dh(b, g.q, DType::kF32, {cs.B, cs.L, cs.R}, f.hidden.data());
    DeviceTensor da(b, g.q, DType::kI64, {cs.B}, f.anchors.data());
    DeviceTensor dout(b, g.q, DType::kF32, {cs.B, cs.L, cs.K, cs.K});
    vt::Dflash2SelectorEdges(g.q, dout.tensor(), dp.tensor(), ds.tensor(), dc.tensor(),
                             du.tensor(), dh.tensor(), da.tensor(), Args(cs.K));
    std::vector<float> got(cpu.size(), 0.0f);
    dout.Download(g.q, got.data());
    for (size_t i = 0; i < cpu.size(); ++i) {
      INFO("K ", cs.K, " R ", cs.R, " edge ", i);
      CHECK(std::fabs(got[i] - cpu[i]) <= 1e-4f * (1.0f + std::fabs(cpu[i])));
    }
  }
}
