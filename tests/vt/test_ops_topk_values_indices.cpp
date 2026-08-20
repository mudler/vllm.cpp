// vllm.cpp original (vt runtime). Unit tests for vt::TopKValuesIndices — the
// vocabulary top-k that EMITS the surviving (id, value) pairs (SPEC-DFLASH2 W3 /
// `## Risks/decisions` D2, #1314).
//
// BEYOND-PIN. The reference is `_topk`
// (vllm/model_executor/models/qwen3_dflash2.py:60-64 @ vllm-project/vllm#52816
// head `66e5414c6d75a8529473d977f7458c140bbab8a0`), whose off-CUDA arm is
// `torch.topk(scores, k, dim=-1)` and whose CUDA arm is FlashInfer's radix
// `top_k(..., sorted=True, deterministic=True)`.
//
// WHAT IS GATED, and why each case exists.
//
//  * THE TIE-BREAK, which is the contract and not an incidental property. The
//    sort-free threshold search this op's CUDA arm extends converges to an exact
//    array VALUE, so `{x >= thr}` keeps whole tie groups atomically and can hold
//    MORE than k elements; something has to choose among equals, and choosing
//    differently reorders the DFlash2 selector's candidate slots and moves
//    acceptance without raising anything. Three cases pin it with HAND-WRITTEN
//    literals, which is the only kind of expectation that can: ties inside the
//    kept set, a tie group STRADDLING the k-th boundary (the one the threshold
//    search actually has to resolve), and a tie group larger than k.
//  * THE ORG-VOCAB PADDING MASK. `num_org_vocab_padding` trailing columns can
//    never contribute a candidate, mirroring upstream's
//    `logits[..., -num_pad:] = -float("inf")`. The case puts the row's two
//    LARGEST values inside the padded tail, so an unmasked implementation
//    returns them and this case names the difference.
//  * THAT THE INPUT IS NOT MUTATED. Unlike `vt::ApplyTopKTopP`, which masks its
//    logits in place, this op is read-only: the DFlash2 caller keeps the same
//    block logits for the DFlash1 comparison arm and for the trace.
//  * A -inf-SATURATED ROW. Fewer than k finite entries is the shape a masked
//    vocabulary produces, and the op still returns exactly k pairs.
//  * BULK SHAPES against a full sort with the same comparator. That reference is
//    INDEPENDENT IN ALGORITHM (a full `std::sort` of every column, against the
//    kernel's `partial_sort` of k) but NOT in the tie rule, which it restates
//    rather than derives -- so it is a consistency check on the ordering and the
//    hand-written literals above are the correctness anchor for the ties.
//  * CUDA == CPU. That case is written and has NEVER RUN on this host (no
//    `nvcc`, so it reports `no CUDA backend; skipping`); it is `## Owed` O10 of
//    the row's spec and is not counted as coverage here.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
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
using vt::TopKValuesIndicesArgs;

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

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

TopKValuesIndicesArgs Args(int64_t k, int64_t pad = 0) {
  TopKValuesIndicesArgs a;
  a.k = k;
  a.num_org_vocab_padding = pad;
  return a;
}

struct Result {
  std::vector<float> values;
  std::vector<int64_t> indices;
};

Result Run(std::vector<float> logits, int64_t rows, int64_t v, int64_t k, int64_t pad = 0) {
  Result r;
  r.values.assign(static_cast<size_t>(rows * k), 0.0f);
  r.indices.assign(static_cast<size_t>(rows * k), -1);
  Tensor tl = F32(logits, {rows, v});
  Tensor tv = F32(r.values, {rows, k});
  Tensor ti = I64(r.indices, {rows, k});
  Queue q = Q();
  vt::TopKValuesIndices(q, tv, ti, tl, Args(k, pad));
  return r;
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

}  // namespace

TEST_CASE("topk-values-indices: descending order over distinct values") {
  // Hand-written. row = [0.5, -1.0, 3.0, 2.0, 7.0, 0.0], k = 3.
  const Result r = Run({0.5f, -1.0f, 3.0f, 2.0f, 7.0f, 0.0f}, 1, 6, 3);
  CHECK(r.values[0] == 7.0f);
  CHECK(r.values[1] == 3.0f);
  CHECK(r.values[2] == 2.0f);
  CHECK(r.indices[0] == 4);
  CHECK(r.indices[1] == 2);
  CHECK(r.indices[2] == 3);
}

TEST_CASE("topk-values-indices: ties INSIDE the kept set break by ASCENDING index") {
  // row = [5, 5, 1, 5], k = 2. Every 5 is equal, so only the index rule decides;
  // a descending-index rule would answer {3, 1}.
  const Result r = Run({5.0f, 5.0f, 1.0f, 5.0f}, 1, 4, 2);
  CHECK(r.values[0] == 5.0f);
  CHECK(r.values[1] == 5.0f);
  CHECK(r.indices[0] == 0);
  CHECK(r.indices[1] == 1);
}

TEST_CASE("topk-values-indices: a tie group STRADDLING the k-th boundary") {
  // THE case the threshold search has to resolve. row = [9, 4, 4, 4, 1], k = 2.
  // The k-th largest value is 4 and THREE columns attain it, so `{x >= thr}` has
  // four members for two slots. One slot goes to the 9; the other must go to the
  // LOWEST-indexed 4, which is column 1. Column 3 is the answer a
  // descending-index or an arbitrary-compaction rule gives.
  const Result r = Run({9.0f, 4.0f, 4.0f, 4.0f, 1.0f}, 1, 5, 2);
  CHECK(r.values[0] == 9.0f);
  CHECK(r.values[1] == 4.0f);
  CHECK(r.indices[0] == 0);
  CHECK(r.indices[1] == 1);
}

TEST_CASE("topk-values-indices: a tie group LARGER than k") {
  // row = [4, 4, 4, 4], k = 3: every column ties, so the whole answer is the
  // index rule and nothing else.
  const Result r = Run({4.0f, 4.0f, 4.0f, 4.0f}, 1, 4, 3);
  CHECK(r.indices[0] == 0);
  CHECK(r.indices[1] == 1);
  CHECK(r.indices[2] == 2);
  for (int j = 0; j < 3; ++j) CHECK(r.values[static_cast<size_t>(j)] == 4.0f);
}

TEST_CASE("topk-values-indices: the org-vocab padding tail can never be a candidate") {
  // row = [1, 2, 99, 98] with the LAST TWO columns padded. Upstream forces them
  // to -inf before the search (`logits[..., -num_pad:] = -inf`,
  // compute_candidates @ the PR head), so the answer is drawn from {1, 2} only.
  // Without the mask it is {99, 98} at columns {2, 3}, which is what makes this
  // case separate the two rather than merely exercise the parameter.
  const Result masked = Run({1.0f, 2.0f, 99.0f, 98.0f}, 1, 4, 2, /*pad=*/2);
  CHECK(masked.values[0] == 2.0f);
  CHECK(masked.values[1] == 1.0f);
  CHECK(masked.indices[0] == 1);
  CHECK(masked.indices[1] == 0);

  const Result unmasked = Run({1.0f, 2.0f, 99.0f, 98.0f}, 1, 4, 2, /*pad=*/0);
  CHECK(unmasked.indices[0] == 2);
  CHECK(unmasked.indices[1] == 3);
}

TEST_CASE("topk-values-indices: the logits are NOT mutated") {
  // vt::ApplyTopKTopP masks its input in place; this op must not, because the
  // DFlash2 caller keeps the same block logits afterwards.
  std::vector<float> logits = {0.5f, -1.0f, 3.0f, 2.0f, 7.0f, 0.0f};
  const std::vector<float> before = logits;
  std::vector<float> values(3, 0.0f);
  std::vector<int64_t> indices(3, -1);
  Tensor tl = F32(logits, {1, 6});
  Tensor tv = F32(values, {1, 3});
  Tensor ti = I64(indices, {1, 3});
  Queue q = Q();
  vt::TopKValuesIndices(q, tv, ti, tl, Args(3));
  CHECK(logits == before);
}

TEST_CASE("topk-values-indices: a -inf-saturated row still returns exactly k pairs") {
  // Fewer FINITE entries than k is what a masked vocabulary looks like. The op
  // returns k pairs regardless, and the -inf columns are ordered by ascending
  // index like any other tie group.
  const Result r = Run({kNegInf, 5.0f, kNegInf, kNegInf}, 1, 4, 3);
  CHECK(r.values[0] == 5.0f);
  CHECK(r.indices[0] == 1);
  CHECK(std::isinf(r.values[1]));
  CHECK(std::isinf(r.values[2]));
  CHECK(r.indices[1] == 0);
  CHECK(r.indices[2] == 2);
}

TEST_CASE("topk-values-indices: bulk rows against a full sort") {
  // CONSISTENCY, not correctness of the tie rule: this reference restates the
  // comparator rather than deriving it. What it independently checks is the
  // SELECTION -- a full std::sort of every column against the kernel's partial
  // selection -- over shapes with repeated values (the LCG's 24-bit quantisation
  // produces them at this width).
  constexpr int64_t rows = 5, v = 257, k = 16;
  const std::vector<float> logits = RandF32(static_cast<size_t>(rows * v), 0xC0FFEEu);
  const Result got = Run(logits, rows, v, k);
  for (int64_t row = 0; row < rows; ++row) {
    std::vector<int64_t> order(static_cast<size_t>(v));
    for (int64_t j = 0; j < v; ++j) order[static_cast<size_t>(j)] = j;
    const float* src = logits.data() + row * v;
    std::sort(order.begin(), order.end(), [src](int64_t a, int64_t b) {
      if (src[a] != src[b]) return src[a] > src[b];
      return a < b;
    });
    for (int64_t j = 0; j < k; ++j) {
      INFO("row ", row, " slot ", j);
      CHECK(got.indices[static_cast<size_t>(row * k + j)] == order[static_cast<size_t>(j)]);
      CHECK(got.values[static_cast<size_t>(row * k + j)] ==
            src[order[static_cast<size_t>(j)]]);
    }
  }
}

TEST_CASE("topk-values-indices: the wrapper refuses shapes it cannot answer") {
  std::vector<float> logits(8, 0.0f);
  std::vector<float> values(4, 0.0f);
  std::vector<int64_t> indices(4, 0);
  Tensor tl = F32(logits, {2, 4});
  Tensor tv = F32(values, {2, 2});
  Tensor ti = I64(indices, {2, 2});
  Queue q = Q();
  // k above the USABLE width, not merely above the width: the padded tail is not
  // eligible, so a k that only fits by counting it is refused.
  CHECK_THROWS(vt::TopKValuesIndices(q, tv, ti, tl, Args(2, /*pad=*/3)));
  CHECK_THROWS(vt::TopKValuesIndices(q, tv, ti, tl, Args(0)));
  CHECK_NOTHROW(vt::TopKValuesIndices(q, tv, ti, tl, Args(2, /*pad=*/2)));
}

// ===========================================================================
// CUDA parity. The CUDA arm extends the sort-free pivot-bracket threshold search
// (src/vt/cuda/cuda_sample.cu) to compact and order the survivors rather than
// mask below the k-th largest. Both arms return an exact array value and an
// exact column index, so the assertion is EQUALITY and not an envelope.
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

TEST_CASE("topk-values-indices: CUDA == CPU, values and indices") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping CUDA topk-values-indices parity");
    return;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  struct Case {
    int64_t rows, v, k, pad;
  };
  // The last two carry the tie shapes the hand-written cases pin on CPU: the LCG
  // repeats values at this width, and a padded tail exercises the search bound.
  const Case cases[] = {{4, 513, 16, 0}, {2, 128, 8, 0}, {3, 200, 16, 24}, {1, 64, 16, 0}};
  for (const Case& cs : cases) {
    const std::vector<float> logits =
        RandF32(static_cast<size_t>(cs.rows * cs.v), 0x5EEDu + static_cast<uint32_t>(cs.v));
    const Result cpu = Run(logits, cs.rows, cs.v, cs.k, cs.pad);

    DeviceTensor dl(b, g.q, DType::kF32, {cs.rows, cs.v}, logits.data());
    DeviceTensor dv(b, g.q, DType::kF32, {cs.rows, cs.k});
    DeviceTensor di(b, g.q, DType::kI64, {cs.rows, cs.k});
    vt::TopKValuesIndices(g.q, dv.tensor(), di.tensor(), dl.tensor(), Args(cs.k, cs.pad));
    std::vector<float> gv(static_cast<size_t>(cs.rows * cs.k), 0.0f);
    std::vector<int64_t> gi(static_cast<size_t>(cs.rows * cs.k), -1);
    dv.Download(g.q, gv.data());
    di.Download(g.q, gi.data());
    for (size_t i = 0; i < gv.size(); ++i) {
      INFO("rows ", cs.rows, " v ", cs.v, " k ", cs.k, " pad ", cs.pad, " slot ", i);
      CHECK(gv[i] == cpu.values[i]);
      CHECK(gi[i] == cpu.indices[i]);
    }
  }
}
