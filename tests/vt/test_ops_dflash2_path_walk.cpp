// vllm.cpp original (vt runtime). Unit tests for vt::Dflash2PathWalk — the
// DFlash2 candidate selector's PATH WALK (SPEC-DFLASH2 W4, #1314).
//
// BEYOND-PIN. The reference is `_selector_walk_kernel`
// (vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:16-79 @
// vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`), one
// Triton program per request with the slot-to-slot dependency as a LOOP INSIDE
// the program. The sequential expectation below is that kernel's own control
// flow at `SAMPLE_PROBABILISTIC=False`, transcribed rather than reinvented:
//
//     previous = 0
//     for step in range(num_steps):
//         scores = scores_ptr[(row*num_steps + step)*top_k + previous][:top_k]
//         _, index = gumbel_noised_argmax(scores, candidates, mask, seed,
//                                         position, 0.0, USE_FP64)
//         token = candidate_ptr[(row*num_steps + step)*top_k + index]
//         tokens_ptr[row*num_steps + step] = token
//         previous = index
//
// At `temp == 0.0` `gumbel_noised_argmax` divides by nothing, adds no noise and
// returns `tl.max(logits, axis=0, return_indices=True)`, whose tie-break is the
// LOWEST index. The fp64 cast is order-preserving on fp32 inputs, so USE_FP64
// cannot move a greedy answer either. That is why this op has ONE arm and it is
// upstream's greedy arm exactly — see `Dflash2PathWalkArgs` (include/vt/ops.h)
// for why the probabilistic arm is not here.
//
// THE HEAD MOVED under this row (#1404) and it moved HERE. At `19c93519` the
// walk carried a hand-written `temperature == 0.0` branch and a hand-written
// Gumbel branch, wrote into a private `_selector_tokens` buffer that was copied
// into `draft_tokens` afterwards, and always allocated a proposal distribution.
// At `66e5414c` the two branches are one `gumbel_noised_argmax` call, the
// private buffer and its `copy_` are DELETED (the walk writes straight into
// `draft_tokens`), and the proposal distribution is optional and `None` for
// greedy. The greedy ANSWER is identical at both heads; the shape this file
// ports is the new one.
//
// WHAT IS GATED, and why each case exists.
//
//  * UPSTREAM'S OWN WALK, at a small shape and at the published one (top_k 16,
//    7 steps). Bit-exact: the walk performs no arithmetic at all, only
//    comparisons and a gather, so there is nothing for a reduction order to
//    move.
//  * THE PREDECESSOR CARRY, with hand-written literals. Step l reads the
//    predecessor ROW `previous` that step l-1 chose. A walk that always read row
//    0 -- which is what step 0 does, so the mistake is invisible at L == 1 --
//    passes every shape check and emits well-formed tokens. The literals below
//    are built so the carrying answer and the row-0 answer DIFFER at step 1.
//  * THE TIE-BREAK. `tl.max(..., return_indices=True)` breaks ties to the LOWEST
//    index; a row with two equal maxima pins it. Getting this wrong reorders the
//    walk's own predecessor for the next step, so it is not a cosmetic choice.
//  * THE ALL -inf ROW. Upstream loads masked lanes as -inf, so a row that is
//    entirely -inf must resolve to index 0 rather than to "no index". The
//    natural parallel formulation (seed the reduction at -inf, keep nothing
//    that does not strictly exceed it) resolves to "no index" instead, and this
//    case is what forces the seed to be named.
//  * THE GATHER. What is stored is `candidate_ptr[base + index]` and NOT the
//    index; a fixture whose ids are the identity permutation cannot tell those
//    apart, so every id here differs from its slot.
//  * A NaN NEVER WINS. Strictness is what the tie rows and the -inf row do not
//    measure: a `>=` reduction answers both of those correctly. This case is
//    the one that fails under `>=`, and it is also the row on which the CUDA
//    arm disagreed with this one until W4's repair (see the CUDA case below).
//  * CUDA vs CPU, BIT-EXACT. Unlike vt::Dflash2SelectorEdges -- a reduction
//    within an f32 envelope -- this op compares and gathers, so the two arms
//    must agree exactly, including the tie rows, the -inf row and the NaN row.
//    That case is written and has NEVER RUN on this host (no `nvcc`, so it
//    reports `no CUDA backend; skipping`); it is owed to the operator's GPU
//    lease and is not counted as coverage here.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Dflash2PathWalkArgs;
using vt::DType;
using vt::Queue;
using vt::Tensor;

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

Dflash2PathWalkArgs Args(int64_t k) {
  Dflash2PathWalkArgs a;
  a.top_k = k;
  return a;
}

// Deterministic LCG in [-2, 2); avoids <random> divergence across libstdc++.
std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<float>(static_cast<double>(s >> 8) / 8388608.0 - 2.0);
  }
  return v;
}
std::vector<int64_t> RandIds(size_t n, int64_t vocab, uint32_t seed) {
  std::vector<int64_t> v(n);
  uint32_t s = seed * 2246822519u + 7u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<int64_t>((s >> 9) % static_cast<uint32_t>(vocab));
  }
  return v;
}

// Upstream's `_selector_walk_kernel` control flow at SAMPLE_PROBABILISTIC=False,
// as a host loop. Independent in PROVENANCE from the op: it is the Triton
// kernel's own sequence, not a rearrangement of ours.
std::vector<int64_t> UpstreamWalk(const std::vector<float>& scores,
                                  const std::vector<int64_t>& cand, int64_t B,
                                  int64_t L, int64_t K) {
  std::vector<int64_t> tokens(static_cast<size_t>(B * L), 0);
  for (int64_t b = 0; b < B; ++b) {
    int64_t previous = 0;
    for (int64_t l = 0; l < L; ++l) {
      const int64_t flat = b * L + l;
      const float* row = scores.data() + (flat * K + previous) * K;
      // tl.max(..., return_indices=True): the largest value, ties to the lowest
      // index. Masked lanes arrive as -inf, so an all -inf row answers 0.
      float best = kNegInf;
      int64_t index = K;
      for (int64_t j = 0; j < K; ++j) {
        if (row[j] > best) {
          best = row[j];
          index = j;
        }
      }
      if (index == K) index = 0;
      tokens[static_cast<size_t>(flat)] = cand[static_cast<size_t>(flat * K + index)];
      previous = index;
    }
  }
  return tokens;
}

std::vector<int64_t> RunCpu(std::vector<float>& scores, std::vector<int64_t>& cand,
                            int64_t B, int64_t L, int64_t K) {
  std::vector<int64_t> tokens(static_cast<size_t>(B * L), -1);
  Queue q = Q();
  Tensor s = Contig(scores.data(), DType::kF32, Cpu(), {B, L, K, K});
  Tensor c = Contig(cand.data(), DType::kI64, Cpu(), {B, L, K});
  Tensor t = Contig(tokens.data(), DType::kI64, Cpu(), {B, L});
  vt::Dflash2PathWalk(q, t, s, c, Args(K));
  return tokens;
}

}  // namespace

TEST_CASE("dflash2-path-walk: matches upstream's sequential walk") {
  struct Case {
    int64_t B, L, K, V;
  };
  // Upstream's own small selector shape, then the PUBLISHED one: top_k 16 on
  // both released DFlash2 drafts, 7 speculative steps.
  const Case cases[] = {{2, 4, 3, 17}, {3, 1, 5, 29}, {2, 7, 16, 512}};
  for (const Case& cs : cases) {
    std::vector<float> scores =
        RandF32(static_cast<size_t>(cs.B * cs.L * cs.K * cs.K), 0x91u);
    std::vector<int64_t> cand =
        RandIds(static_cast<size_t>(cs.B * cs.L * cs.K), cs.V, 0x92u);
    const std::vector<int64_t> want = UpstreamWalk(scores, cand, cs.B, cs.L, cs.K);
    const std::vector<int64_t> got = RunCpu(scores, cand, cs.B, cs.L, cs.K);
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < want.size(); ++i) {
      INFO("B ", cs.B, " L ", cs.L, " K ", cs.K, " slot ", i);
      CHECK(got[i] == want[i]);
    }
  }
}

TEST_CASE("dflash2-path-walk: step l reads the predecessor row step l-1 CHOSE") {
  // One request, two steps, three candidates. Step 0's argmax is slot 2, so
  // step 1 must read its predecessor ROW 2 -- whose argmax is slot 0 -- and not
  // row 0, whose argmax is slot 1. A walk that dropped the carry emits the same
  // number of well-formed tokens and differs only here.
  const int64_t B = 1, L = 2, K = 3;
  std::vector<float> scores(static_cast<size_t>(B * L * K * K), 0.0f);
  auto at = [&](int64_t l, int64_t p, int64_t c) -> float& {
    return scores[static_cast<size_t>(((0 * L + l) * K + p) * K + c)];
  };
  // Step 0: only row 0 is read (previous == 0). Slot 2 wins.
  at(0, 0, 0) = 1.0f;  at(0, 0, 1) = 2.0f;  at(0, 0, 2) = 3.0f;
  at(0, 1, 0) = 9.0f;  at(0, 1, 1) = 9.0f;  at(0, 1, 2) = 9.0f;
  at(0, 2, 0) = 9.0f;  at(0, 2, 1) = 9.0f;  at(0, 2, 2) = 9.0f;
  // Step 1: row 0 would pick slot 1; row 2 -- the correct predecessor -- picks
  // slot 0. Row 1 picks slot 2, so a carry that were off by one is also caught.
  at(1, 0, 0) = 0.0f;  at(1, 0, 1) = 5.0f;  at(1, 0, 2) = 1.0f;
  at(1, 1, 0) = 0.0f;  at(1, 1, 1) = 1.0f;  at(1, 1, 2) = 5.0f;
  at(1, 2, 0) = 5.0f;  at(1, 2, 1) = 1.0f;  at(1, 2, 2) = 0.0f;
  // Ids are NOT the identity permutation: the op must gather the candidate at
  // the winning slot, not report the slot.
  std::vector<int64_t> cand = {41, 42, 43,   // step 0 slots 0,1,2
                               71, 72, 73};  // step 1 slots 0,1,2
  const std::vector<int64_t> got = RunCpu(scores, cand, B, L, K);
  REQUIRE(got.size() == 2);
  CHECK(got[0] == 43);  // step 0 slot 2
  CHECK(got[1] == 71);  // step 1 slot 0, via predecessor row 2
  // What the two live mistakes would produce, asserted as NOT the answer so the
  // case cannot pass by coincidence if the literals are ever retuned.
  CHECK(got[1] != 72);  // row 0 (carry dropped)
  CHECK(got[1] != 73);  // row 1 (carry off by one)
}

TEST_CASE("dflash2-path-walk: a tie resolves to the LOWEST slot") {
  // `tl.max(..., return_indices=True)` breaks ties left. Slots 1 and 3 hold the
  // same maximum; the walk must take 1, and it must then read predecessor ROW 1
  // at the next step, so the tie decides two tokens rather than one.
  const int64_t B = 1, L = 2, K = 4;
  std::vector<float> scores(static_cast<size_t>(B * L * K * K), -1.0f);
  auto at = [&](int64_t l, int64_t p, int64_t c) -> float& {
    return scores[static_cast<size_t>((l * K + p) * K + c)];
  };
  at(0, 0, 0) = 0.5f; at(0, 0, 1) = 2.5f; at(0, 0, 2) = 0.5f; at(0, 0, 3) = 2.5f;
  // Row 1 and row 3 of step 1 answer differently, so the tie is observable.
  at(1, 1, 0) = 7.0f;
  at(1, 3, 2) = 7.0f;
  std::vector<int64_t> cand = {10, 11, 12, 13, 20, 21, 22, 23};
  const std::vector<int64_t> got = RunCpu(scores, cand, B, L, K);
  REQUIRE(got.size() == 2);
  CHECK(got[0] == 11);  // slot 1, not slot 3
  CHECK(got[1] == 20);  // predecessor row 1 -> slot 0
}

TEST_CASE("dflash2-path-walk: an all -inf row resolves to slot 0") {
  // Upstream loads masked lanes with `other=float("-inf")`, so a request whose
  // slots are all masked still reduces, and `tl.max` returns index 0. A parallel
  // formulation seeded at -inf that keeps only what STRICTLY exceeds the seed
  // returns "no index" instead; this case is what pins the answer.
  const int64_t B = 1, L = 2, K = 3;
  std::vector<float> scores(static_cast<size_t>(B * L * K * K), kNegInf);
  // Step 1 row 0 has a real maximum, so the case also proves the walk carried
  // slot 0 forward rather than stopping.
  scores[static_cast<size_t>(((1 * K) + 0) * K + 2)] = 4.0f;
  std::vector<int64_t> cand = {31, 32, 33, 61, 62, 63};
  const std::vector<int64_t> got = RunCpu(scores, cand, B, L, K);
  REQUIRE(got.size() == 2);
  CHECK(got[0] == 31);  // slot 0
  CHECK(got[1] == 63);  // predecessor row 0 -> slot 2
}

TEST_CASE("dflash2-path-walk: a NaN never wins a slot") {
  // STRICTNESS, on its own. The tie rows and the all -inf row above are both
  // answered correctly by a `>=` reduction too, so nothing there measures the
  // word "strictly" in the contract; this case does. A NaN compares false
  // against everything, so a strict `>` can never let one claim a slot, and a
  // row whose only non -inf lane is a NaN collapses to slot 0 exactly as a
  // fully masked row does.
  const int64_t B = 1, L = 2, K = 3;
  const float kNan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> scores(static_cast<size_t>(B * L * K * K), kNegInf);
  auto at = [&](int64_t l, int64_t p, int64_t c) -> float& {
    return scores[static_cast<size_t>((l * K + p) * K + c)];
  };
  // Step 0, row 0: a NaN beside two masked lanes. Under `>=` the LAST -inf lane
  // would claim the row instead and the walk would emit slot 2.
  at(0, 0, 0) = kNan;
  // Step 1 reads predecessor row 0, because step 0 answered slot 0. A NaN sits
  // between two real values: a rule that let a NaN win answers slot 1, and the
  // contract answers slot 2.
  at(1, 0, 0) = 1.0f;
  at(1, 0, 1) = kNan;
  at(1, 0, 2) = 2.0f;
  std::vector<int64_t> cand = {81, 82, 83, 91, 92, 93};
  const std::vector<int64_t> got = RunCpu(scores, cand, B, L, K);
  REQUIRE(got.size() == 2);
  CHECK(got[0] == 81);  // slot 0: neither the NaN nor a masked lane claimed it
  CHECK(got[0] != 83);  // what a `>=` reduction answers
  CHECK(got[1] == 93);  // slot 2
  CHECK(got[1] != 92);  // what a NaN-wins reduction answers
}

TEST_CASE("dflash2-path-walk: refuses a lattice that is not [B,L,K,K]") {
  // BOTH trailing axes, each on its own, each over a GENUINELY CONTIGUOUS
  // tensor, and each matched on the message.
  //
  // Mutating one `shape` field of a tensor built by `Contig` does NOT test this
  // guard. It desynchronises the strides, `Tensor::IsContiguous()` turns false,
  // and `dflash2-path-walk: contiguous tensors required` throws FIRST -- so a
  // bare `CHECK_THROWS` is satisfied by a guard it was not written for, and it
  // still passes with the shape check deleted outright. That is what W4's fresh
  // review measured on the previous version of this case. The lattices below
  // are built at the wrong extent WITH matching strides, so the only guard that
  // can answer is the one named.
  const int64_t B = 1, L = 2, K = 3;
  // Sized for the FULL [B,L,K,K] lattice, so a DELETED guard reads in bounds
  // and simply fails to throw rather than running off the end of the buffer.
  std::vector<float> scores(static_cast<size_t>(B * L * K * K), 0.0f);
  std::vector<int64_t> cand(static_cast<size_t>(B * L * K), 0);
  std::vector<int64_t> tokens(static_cast<size_t>(B * L), 0);
  Queue q = Q();
  Tensor c = Contig(cand.data(), DType::kI64, Cpu(), {B, L, K});
  Tensor t = Contig(tokens.data(), DType::kI64, Cpu(), {B, L});

  // The CHILD axis, [B,L,K,K-1]: a row that is short of one candidate.
  Tensor child = Contig(scores.data(), DType::kF32, Cpu(), {B, L, K, K - 1});
  REQUIRE(child.IsContiguous());
  CHECK_THROWS_WITH_AS(vt::Dflash2PathWalk(q, t, child, c, Args(K)),
                       doctest::Contains("scores must be [B,L,K,K]"),
                       std::runtime_error);
  // The PREDECESSOR axis, [B,L,K-1,K]: the same extent mistake one axis to the
  // left, which is the lattice indexed the wrong way round. It reads plausible
  // scores from the wrong rows and moves acceptance without raising, which is
  // why the wrapper checks both axes instead of checking one and inferring.
  Tensor pred = Contig(scores.data(), DType::kF32, Cpu(), {B, L, K - 1, K});
  REQUIRE(pred.IsContiguous());
  CHECK_THROWS_WITH_AS(vt::Dflash2PathWalk(q, t, pred, c, Args(K)),
                       doctest::Contains("scores must be [B,L,K,K]"),
                       std::runtime_error);

  Tensor good = Contig(scores.data(), DType::kF32, Cpu(), {B, L, K, K});
  CHECK_NOTHROW(vt::Dflash2PathWalk(q, t, good, c, Args(K)));
}

// ===========================================================================
// CUDA parity, BIT-EXACT. See the file header for why this op is specified
// exact across backends where vt::Dflash2SelectorEdges is not.
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

TEST_CASE("dflash2-path-walk: CUDA equals CPU bit-for-bit, ties and -inf included") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping CUDA dflash2-path-walk parity");
    return;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  struct Case {
    int64_t B, L, K, V;
  };
  // A sub-warp K, the published K, and a K wider than one warp so the strided
  // load and the shuffle reduction are both exercised past 32 lanes.
  const Case cases[] = {{2, 4, 3, 17}, {2, 7, 16, 512}, {1, 3, 40, 97}};
  for (const Case& cs : cases) {
    std::vector<float> scores =
        RandF32(static_cast<size_t>(cs.B * cs.L * cs.K * cs.K), 0x91u);
    std::vector<int64_t> cand =
        RandIds(static_cast<size_t>(cs.B * cs.L * cs.K), cs.V, 0x92u);
    // Force the three rows a random fixture cannot produce, CHAINED so each one
    // is certain to be READ: request 0 step 0 is an exact tie group, which
    // answers slot 0; step 1 predecessor row 0 is therefore read and is all
    // -inf, which answers slot 0; step 2 predecessor row 0 is therefore read
    // and is a NaN beside masked lanes. Each is asserted against the CPU arm,
    // which the hand-written CPU cases above pin against literals.
    //
    // The NaN row is the one the two arms DISAGREED on before W4's repair. The
    // CUDA lane scan carried an `|| (v == best && j < slot)` clause whose only
    // reachable effect was at the seed, where a lane holding -inf compared
    // equal to the -inf seed and claimed a slot the CPU arm's strict scan
    // refuses: `[NaN,-inf]` read cpu 0 / cuda 1. Every NaN-free row agreed,
    // including this fixture's tie group and its all -inf row, so nothing that
    // ran here could see it.
    const int64_t K = cs.K;
    for (int64_t c = 0; c < K; ++c) scores[static_cast<size_t>(c)] = 1.25f;
    const size_t inf_row = static_cast<size_t>(((0 * cs.L + 1) * K + 0) * K);
    for (int64_t c = 0; c < K; ++c) scores[inf_row + static_cast<size_t>(c)] = kNegInf;
    const size_t nan_row = static_cast<size_t>(((0 * cs.L + 2) * K + 0) * K);
    for (int64_t c = 0; c < K; ++c) scores[nan_row + static_cast<size_t>(c)] = kNegInf;
    scores[nan_row] = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> scores_cpu = scores;
    std::vector<int64_t> cand_cpu = cand;
    const std::vector<int64_t> cpu = RunCpu(scores_cpu, cand_cpu, cs.B, cs.L, K);

    DeviceTensor ds(b, g.q, DType::kF32, {cs.B, cs.L, K, K}, scores.data());
    DeviceTensor dc(b, g.q, DType::kI64, {cs.B, cs.L, K}, cand.data());
    DeviceTensor dt(b, g.q, DType::kI64, {cs.B, cs.L});
    vt::Dflash2PathWalk(g.q, dt.tensor(), ds.tensor(), dc.tensor(), Args(K));
    std::vector<int64_t> got(cpu.size(), -1);
    dt.Download(g.q, got.data());
    for (size_t i = 0; i < cpu.size(); ++i) {
      INFO("K ", K, " slot ", i);
      CHECK(got[i] == cpu[i]);
    }
    // The forced rows, asserted directly so the parity above cannot pass by
    // two arms agreeing on nothing: slot 0 of the tie group, slot 0 of the -inf
    // row, and slot 0 of the NaN row.
    CHECK(cpu[0] == cand[0]);
    CHECK(cpu[1] == cand[static_cast<size_t>(K)]);
    CHECK(cpu[2] == cand[static_cast<size_t>(2 * K)]);
  }
}
