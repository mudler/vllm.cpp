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
//    acceptance without raising anything. `LiteralRows()` below is the table of
//    HAND-WRITTEN rows that pins it -- ties inside the kept set, a tie group
//    STRADDLING the k-th boundary (the one the threshold search actually has to
//    resolve), a tie group larger than k, a -inf-saturated row, and NaN -- and
//    BOTH ARMS iterate it, save for the one row named below that the CUDA arm
//    does not implement. That table is a table rather than a case each because
//    it is the only data in this file that can separate the two algorithms, and
//    the CUDA arm has to see the same rows the CPU arm does.
//  * THE ORG-VOCAB PADDING MASK. `num_org_vocab_padding` trailing columns can
//    never contribute a candidate, mirroring upstream's
//    `logits[..., -num_pad:] = -float("inf")`. The table's padded row puts the
//    row's two LARGEST values inside the padded tail, so an unmasked
//    implementation returns them, and a separate case asserts what the unmasked
//    answer is.
//  * THAT THE INPUT IS NOT MUTATED. Unlike `vt::ApplyTopKTopP`, which masks its
//    logits in place, this op is read-only: the DFlash2 caller keeps the same
//    block logits for the DFlash1 comparison arm and for the trace.
//  * BULK SHAPES against a full sort with the same comparator. That reference is
//    INDEPENDENT IN ALGORITHM (a full `std::sort` of every column, against the
//    kernel's `partial_sort` of k) but NOT in the tie rule, which it restates
//    rather than derives. It also contains NO TIE: the LCG's rows are all
//    distinct at these widths, measured rather than assumed. So it is a
//    consistency check on the SELECTION, and the literal table above is the
//    correctness anchor for the ties.
//  * CUDA == CPU. Two cases. They report `no CUDA backend; skipping` on the
//    authoring host, which has no `nvcc`, and they HAVE now run elsewhere: a GB10
//    (sm_121a, nvcc 13.0) ran this file on 2026-08-20 at W3 head `b29b6f886`,
//    562 device assertions against 202 CPU-only, recorded in
//    [#1489](https://github.com/mudler/vllm.cpp/issues/1489). One case runs the
//    LITERAL table on the device and asserts the literals; the other asserts the
//    two arms against each other over the literals and over four bulk shapes.
//  * WHAT THE DEVICE CASES DO NOT RUN, and why that is a narrowing rather than a
//    hole. Both SKIP the NaN row BY NAME (`kNanRowName`). The CUDA arm does not
//    implement NaN-first ordering and cannot: its bracket uses `fmaxf`/`fminf`,
//    which return the non-NaN operand, and its survivor pass tests `r[j] > thr`,
//    which is FALSE for a NaN, so `TopKValuesIndicesRowKernel` can never select
//    one. #1489 measured the disagreement rather than predicting it -- 12 failed
//    assertions, every one on that row, including the direct cross-arm pair
//    `CHECK(gpu.indices[i] == cpu.indices[i])` reading `2 == 1`. The row stays in
//    the table because the CPU arm's ordering IS the guarantee (and is
//    mutation-proven); it is the DEVICE cases that are narrowed to the arm which
//    implements it. Reconciling the kernel to NaN-first is owed to #1489, and
//    `include/vt/ops.h` states the asymmetry where it states the contract.
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

// THE HAND-WRITTEN LITERAL ROWS, and the reason they are a TABLE rather than a
// case each. These are the rows that actually CONTAIN a tie, so they are the
// only data in this file that can separate the CPU arm's explicit comparator
// from the CUDA arm's threshold-compact-and-fill — which spec `## Owed` O10
// calls the real risk of the port. Until this table existed the CUDA parity
// case ran only LCG rows, and the comment claiming those rows repeat values was
// false: reproducing the generator in exact float32 for all four of its
// parameter sets gives 513/513, 128/128, 200/200 and 64/64 DISTINCT values per
// row, with the k-th largest at multiplicity 1 in every one. The device arm was
// gated on data holding no tie at all.
//
// Both arms iterate this table -- the CUDA arm skipping the NaN row alone, for
// the reason `RunsOnCuda` below states -- and both assert the LITERALS rather
// than each other, so neither arm can be right merely by agreeing with the
// other.
// The NaN row's name, spelled ONCE. Both the table below and the two CUDA cases
// that exclude it read this constant, so a rename cannot silently turn the
// exclusion into a no-op -- and `CpuOnlyRowCount()` asserts it still matches
// exactly one row, because a filter matching zero rows would put the row back on
// a device that cannot answer it and a filter matching all of them would leave
// the device cases asserting nothing at all.
constexpr const char* kNanRowName = "NaN sorts first, as torch.topk does";

struct LiteralRow {
  const char* name;
  std::vector<float> logits;
  int64_t rows, v, k, pad;
  std::vector<int64_t> want_idx;
  std::vector<float> want_val;
};

const std::vector<LiteralRow>& LiteralRows() {
  static const std::vector<LiteralRow> rows = {
      // Plain descending order over DISTINCT values: the baseline that makes the
      // tie rows below separable from "the op returns something sorted".
      {"distinct values", {0.5f, -1.0f, 3.0f, 2.0f, 7.0f, 0.0f}, 1, 6, 3, 0,
       {4, 2, 3}, {7.0f, 3.0f, 2.0f}},
      // Ties INSIDE the kept set. Every 5 is equal, so only the index rule
      // decides; a descending-index rule answers {3, 1}.
      {"ties inside the kept set", {5.0f, 5.0f, 1.0f, 5.0f}, 1, 4, 2, 0,
       {0, 1}, {5.0f, 5.0f}},
      // THE case the threshold search has to resolve: a tie group STRADDLING the
      // k-th boundary. The k-th largest value is 4 and THREE columns attain it,
      // so `{x >= thr}` has four members for two slots. One slot goes to the 9;
      // the other must go to the LOWEST-indexed 4, column 1. Column 3 is the
      // answer a descending-index or an arbitrary-compaction rule gives — and
      // arbitrary compaction is exactly what the CUDA arm's `atomicAdd` slot
      // assignment would produce if the fill step were wrong.
      {"tie group straddling the k-th boundary", {9.0f, 4.0f, 4.0f, 4.0f, 1.0f}, 1, 5, 2, 0,
       {0, 1}, {9.0f, 4.0f}},
      // A tie group LARGER than k: every column ties, so the whole answer is the
      // index rule and nothing else.
      {"tie group larger than k", {4.0f, 4.0f, 4.0f, 4.0f}, 1, 4, 3, 0,
       {0, 1, 2}, {4.0f, 4.0f, 4.0f}},
      // The org-vocab padding tail can never be a candidate. Upstream forces the
      // last `pad` columns to -inf before the search
      // (`logits[..., -num_pad:] = -inf`, compute_candidates @ the PR head), so
      // the answer is drawn from {1, 2} only. WITHOUT the mask it is {99, 98} at
      // columns {2, 3}, which the case below asserts separately.
      {"org-vocab padding tail masked", {1.0f, 2.0f, 99.0f, 98.0f}, 1, 4, 2, 2,
       {1, 0}, {2.0f, 1.0f}},
      // A -inf-SATURATED row. Fewer FINITE entries than k is what a masked
      // vocabulary looks like; the op returns k pairs regardless and the -inf
      // columns are ordered by ascending index like any other tie group. This is
      // also the row where the CUDA arm's `cur` has to DROP below the threshold
      // because the tie group is exhausted.
      {"-inf-saturated row", {kNegInf, 5.0f, kNegInf, kNegInf}, 1, 4, 3, 0,
       {1, 0, 2}, {5.0f, kNegInf, kNegInf}},
      // NaN, and THE ONE ROW THIS TABLE DOES NOT RUN ON BOTH ARMS. `torch.topk`
      // — this op's off-CUDA reference — orders NaN FIRST for `largest=True`,
      // and the CPU comparator has to say so explicitly: a comparator that only
      // writes `if (src[a] != src[b]) return src[a] > src[b]` makes NaN compare
      // EQUIVALENT to every value while those values are not equivalent to each
      // other, which is not a strict weak ordering and is undefined behaviour in
      // `std::partial_sort`. No shipped path produces a NaN logit today, so this
      // row is synthetic in the same sense `num_org_vocab_padding` is, and it is
      // here because the ordering has to be DEFINED rather than left to whichever
      // comparison the sort happens to make.
      //
      // The CUDA arm does NOT implement this ordering, and #1489 measured it: the
      // threshold search's `fmaxf`/`fminf` return the non-NaN operand and its
      // survivor test `r[j] > thr` is false for a NaN, so the kernel cannot
      // select one. So the two CUDA cases below SKIP this row by name. The row
      // stays here because the CPU order is the guarantee — reverting the
      // comparator reds it — and because deleting it would take the definition
      // away instead of scoping it. See `RunsOnCuda` below and
      // `include/vt/ops.h`, which states the asymmetry beside the contract.
      {kNanRowName,
       {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 2.0f}, 1, 4, 3, 0,
       {1, 2, 3},
       {std::numeric_limits<float>::quiet_NaN(), 3.0f, 2.0f}},
  };
  return rows;
}

// WHICH ARM IMPLEMENTS WHICH ROW. Every row runs on the CPU arm. All but the NaN
// row run on the CUDA arm too. This is a real backend asymmetry, measured on a
// GB10 and recorded in #1489 and in `include/vt/ops.h`, not a convenience: the
// CUDA arm cannot select a NaN at all, so asserting the CPU contract against it
// ships a permanently red suite on every CUDA build. Reconciling the kernel is
// owed to #1489; scoping the assertion to the arm that implements the contract
// is what keeps the record TRUE in the meantime.
bool RunsOnCuda(const LiteralRow& r) {
  return std::string(r.name) != std::string(kNanRowName);
}

// How many rows the CUDA cases skip. Asserted to be exactly one by the CPU case,
// which runs on every host -- the device cases cannot report a filter that
// matched nothing, because on this host they do not run at all.
size_t CpuOnlyRowCount() {
  size_t n = 0;
  for (const LiteralRow& r : LiteralRows()) {
    if (!RunsOnCuda(r)) ++n;
  }
  return n;
}

// Assert one row's literal answer. A NaN in `want_val` is asserted AS a NaN,
// because `NaN == NaN` is false and a bare equality would silently pass on any
// value at all -- the shape of a check that measures nothing.
void CheckLiteral(const LiteralRow& r, const std::vector<float>& values,
                  const std::vector<int64_t>& indices) {
  for (int64_t j = 0; j < r.rows * r.k; ++j) {
    const size_t s = static_cast<size_t>(j);
    INFO("row \"", std::string(r.name), "\" slot ", j);
    CHECK(indices[s] == r.want_idx[s]);
    if (std::isnan(r.want_val[s])) CHECK(std::isnan(values[s]));
    else CHECK(values[s] == r.want_val[s]);
  }
}

}  // namespace

TEST_CASE("topk-values-indices: the hand-written LITERAL rows") {
  // The tie-break contract, on the CPU arm. Every row here is also run on the
  // device by the CUDA parity case at the bottom of this file, against these
  // same literals -- which is what `## Owed` O10 asks for and what the file did
  // not have: before this table the device arm saw only LCG rows, and those rows
  // hold no duplicate value at all.
  for (const LiteralRow& r : LiteralRows()) {
    const Result got = Run(r.logits, r.rows, r.v, r.k, r.pad);
    CheckLiteral(r, got.values, got.indices);
  }
  // THE EXCLUSION'S OWN MATCH COUNT, asserted where it can run. The two CUDA
  // cases skip exactly one row by name; a filter that matched no row would put
  // the NaN row back on a device that cannot answer it, and a filter that matched
  // every row would leave both device cases asserting nothing. Neither shape
  // fails on a host without `nvcc`, so it is checked here rather than there.
  CHECK(CpuOnlyRowCount() == 1);
  // And WHICH row it is, tied to the PROPERTY rather than to the name. The CUDA
  // arm's only gap is that it cannot select a NaN, so the row the device cases
  // skip has to be the row that EXPECTS one -- and every row that does not
  // expect one has to run there. Asserting this against `want_val` instead of
  // against `kNanRowName` is what keeps it from being the name string compared
  // with itself, and it is what fails if a later row is narrowed out of the
  // device arm for a reason nobody wrote down.
  for (const LiteralRow& r : LiteralRows()) {
    bool expects_nan = false;
    for (const float w : r.want_val) expects_nan = expects_nan || std::isnan(w);
    INFO("row \"", std::string(r.name), "\"");
    CHECK(RunsOnCuda(r) == !expects_nan);
  }
}

TEST_CASE("topk-values-indices: WITHOUT the padding mask the tail wins") {
  // The other half of the padded row above, and what makes that row separate the
  // two rather than merely exercise the parameter: unmasked, the same logits
  // answer {99, 98} at columns {2, 3}.
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

TEST_CASE("topk-values-indices: bulk rows against a full sort") {
  // CONSISTENCY, not correctness of the tie rule: this reference restates the
  // comparator rather than deriving it. What it independently checks is the
  // SELECTION -- a full std::sort of every column against the kernel's partial
  // selection.
  //
  // IT CONTAINS NO TIE, and an earlier revision of this comment claimed it did
  // ("the LCG's 24-bit quantisation produces them at this width"). Reproducing
  // the generator in exact float32 gives 257/257 DISTINCT values in every one of
  // these five rows, with the k-th largest at multiplicity 1. So this case
  // measures the SELECTION and the ordering of distinct values; the tie rule is
  // pinned by the literal table above and by nothing here.
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

// Run one shape on the device and hand back what it wrote.
Result RunCuda(Backend& b, Queue& q, const std::vector<float>& logits, int64_t rows,
               int64_t v, int64_t k, int64_t pad) {
  DeviceTensor dl(b, q, DType::kF32, {rows, v}, logits.data());
  DeviceTensor dv(b, q, DType::kF32, {rows, k});
  DeviceTensor di(b, q, DType::kI64, {rows, k});
  vt::TopKValuesIndices(q, dv.tensor(), di.tensor(), dl.tensor(),
                        Args(k, pad));
  Result r;
  r.values.assign(static_cast<size_t>(rows * k), 0.0f);
  r.indices.assign(static_cast<size_t>(rows * k), -1);
  dv.Download(q, r.values.data());
  di.Download(q, r.indices.data());
  return r;
}

}  // namespace

TEST_CASE("topk-values-indices: CUDA runs the LITERAL tie rows") {
  // SPEC-DFLASH2 `## Owed` O10, and the case that file did not have. O10 names
  // the TIE handling as the real risk of this op: the CPU arm sorts under an
  // explicit comparator, while the CUDA arm finds a threshold, compacts
  // everything strictly above it through an `atomicAdd` slot counter, then fills
  // the remaining slots with the lowest-indexed elements EQUAL to it. Those are
  // two different algorithms reaching for one answer, and only a row that
  // CONTAINS a tie can tell them apart.
  //
  // O10 used to say the hand-written tie cases would exercise that path when the
  // CUDA arm could finally run them. They could not: every one of them called
  // `Run()`, which builds a CPU queue, and the only case that touched the device
  // ran four LCG shapes whose rows hold no duplicate value at all. This case
  // puts the literals on the device.
  //
  // The assertion is the LITERAL, not the CPU arm's answer: two implementations
  // agreeing on a wrong tie rule is the failure this table exists to catch.
  //
  // A GB10 ran it on 2026-08-20 (#1489) and the tie rows AGREED -- the straddling
  // group, the group larger than k, the ties inside the kept set and the
  // -inf-saturated row all matched their literals. The NaN row did not, and it is
  // the one row this case skips; see `RunsOnCuda`.
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping CUDA topk-values-indices literal rows");
    return;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  size_t ran = 0;
  for (const LiteralRow& r : LiteralRows()) {
    if (!RunsOnCuda(r)) continue;  // the NaN row; see `RunsOnCuda` and #1489.
    const Result got = RunCuda(b, g.q, r.logits, r.rows, r.v, r.k, r.pad);
    CheckLiteral(r, got.values, got.indices);
    ++ran;
  }
  // Exactly one row was skipped, stated against the literal 1 rather than
  // against the filter's own answer, which would be a tautology.
  CHECK(ran + 1 == LiteralRows().size());
}

TEST_CASE("topk-values-indices: CUDA == CPU, values and indices") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping CUDA topk-values-indices parity");
    return;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  // The literal rows the CUDA arm implements, both arms, EQUALITY. The case
  // above pins each arm to the literal; this pins them to each other, which is
  // the assertion that still fires if a future row is added to the table without
  // an expectation.
  size_t ran = 0;
  for (const LiteralRow& r : LiteralRows()) {
    // The NaN row is excluded here for the reason `RunsOnCuda` gives: the two
    // arms genuinely DISAGREE on it (#1489), so a cross-arm equality over it
    // asserts a contract no shipped backend delivers. Every other row is
    // compared, including all four tie rows and the -inf-saturated one, which
    // are the rows `## Owed` O10 called the real risk -- and which #1489
    // measured AGREEING.
    if (!RunsOnCuda(r)) continue;
    const Result cpu = Run(r.logits, r.rows, r.v, r.k, r.pad);
    const Result gpu = RunCuda(b, g.q, r.logits, r.rows, r.v, r.k, r.pad);
    for (size_t i = 0; i < cpu.values.size(); ++i) {
      INFO("literal row \"", std::string(r.name), "\" slot ", i);
      CHECK(gpu.indices[i] == cpu.indices[i]);
      CHECK(gpu.values[i] == cpu.values[i]);
    }
    ++ran;
  }
  CHECK(ran + 1 == LiteralRows().size());

  struct Case {
    int64_t rows, v, k, pad;
  };
  // BULK SHAPES, and what they do and do not cover. These four rows are LCG
  // data, and the LCG produces NO duplicate value in any row of any of them:
  // reproducing the generator in exact float32 gives 513/513, 128/128, 200/200
  // and 64/64 distinct values, with the k-th largest at multiplicity 1
  // everywhere. So they exercise the threshold search's BRACKET at widths the
  // literal rows do not reach -- including one wider than the block and one with
  // a padded tail -- and they say nothing about the tie rule. An earlier comment
  // here claimed the opposite ("the LCG repeats values at this width"), which is
  // what left the device arm ungated for ties.
  const Case cases[] = {{4, 513, 16, 0}, {2, 128, 8, 0}, {3, 200, 16, 24}, {1, 64, 16, 0}};
  for (const Case& cs : cases) {
    const std::vector<float> logits =
        RandF32(static_cast<size_t>(cs.rows * cs.v), 0x5EEDu + static_cast<uint32_t>(cs.v));
    const Result cpu = Run(logits, cs.rows, cs.v, cs.k, cs.pad);
    const Result gpu = RunCuda(b, g.q, logits, cs.rows, cs.v, cs.k, cs.pad);
    for (size_t i = 0; i < gpu.values.size(); ++i) {
      INFO("rows ", cs.rows, " v ", cs.v, " k ", cs.k, " pad ", cs.pad, " slot ", i);
      CHECK(gpu.values[i] == cpu.values[i]);
      CHECK(gpu.indices[i] == cpu.indices[i]);
    }
  }
}
