// vt::DsaIndexerLogits / vt::DsaTopkSelect unit tests — dots3-note W4b-3c, #699.
//
// ─── WHAT IS UNDER TEST, AND AGAINST WHAT ───────────────────────────────────
// The ops are the `vt::Tensor` form of the host reference
// `vllm::deepseek_v4::DsaIndexerLogits` / `DsaIndexerWeightFold` /
// `DsaTopkSelect` (include/vllm/model_executor/models/deepseek_v4_dsa.h), which
// W3 landed as `std::vector<float>` round-trips and which is the ORACLE here.
// Both are transcriptions of the same upstream, @ `bc2d63e650`:
//
//   vllm/v1/attention/ops/triton_fp8_mqa_logits.py:120-156  (the MQA logit)
//   vllm/model_executor/models/deepseek_v2.py:840           (the weight fold)
//   vllm/model_executor/layers/sparse_attn_indexer.py:509   (top_k_per_row)
//
// A shared-helper comparison proves the two arms agree, never that either is
// right — the trap this repository keeps naming. So three assertions here are
// about the MECHANISM rather than about the files:
//
//   * one logit is asserted BY VALUE against a from-first-principles double
//     computation written in this file, which covers the whole fold including
//     `softmax_scale` and `n_head_scale`;
//   * the ReLU is shown to be load-bearing by a key whose every head dots
//     NEGATIVE scoring exactly 0.0, which a sum-without-ReLU cannot produce;
//   * the selection is re-run with BF16 operands and required to select the
//     IDENTICAL set, which measures the fixture's margin adequacy instead of
//     assuming it.
//
// ─── WHY A SINGLE TOLERANCE WOULD SAY ALMOST NOTHING ────────────────────────
// The selection is DISCRETE, so its error is bimodal: either a slot flips and
// the residue jumps to mechanism scale, or it does not and the residue is the
// float floor. `.agents/specs/dots3-note.md` §4.5 measured this row's strict
// selection margins at 1.29e-3 with float logits while a bf16 logit of order 1
// carries ~4e-3 — LARGER than that margin. A fixture inherited from a
// continuous gate would therefore be a coin flip. Every case below prints the
// minimum decision margin, the ulp it is being judged against, and how many
// rows actually prune, so the fixture's adequacy is measured in the output
// rather than asserted in a comment.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DsaIndexerLogitsArgs;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

// ─── the released dots3-note indexer geometry, scaled down ──────────────────
// `index_n_heads` 64, `index_head_dim` 128, `index_topk` 2048 on the shipped
// config. The shapes below keep the RELATIONS that matter — many more keys than
// `topk`, several heads, a head dim wider than the rope slice — at a size whose
// margins can be enumerated and printed.
constexpr int kTokens = 8;
constexpr int kHeads = 4;
constexpr int kHeadDim = 16;
constexpr int kKeys = 8;
constexpr int kTopk = 3;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

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

// One fixture, DESIGNED rather than sampled — because a sampled one is a coin
// flip here, and this file measured that rather than assuming it. The first
// version of this fixture used uniform noise for `k`; its minimum strict
// decision margin came out at 7.43e-4 against a bf16 ulp of 1.28e-3 at its own
// logit scale, i.e. a fixture whose selection could have been flipped by the
// operand narrowing the model path actually performs. It was replaced, not
// given a looser threshold. That is §4.5's 1.29e-3-against-~4e-3 warning
// reproduced in this file at a smaller scale.
//
// THE DESIGN. Every key is `alpha_s * u + small noise` for one FIXED POSITIVE
// direction `u`, with every `q` entry positive. The dot is linear, so
// `logit[t,s] ~ alpha_s * C_t` with `C_t > 0`: the ORDER of the keys is the
// order of the alphas on every row, and the decision margins are
// `(alpha_i - alpha_j) * C_t` — large, analysable, and printed.
//
//   key 0  alpha 0.9
//   key 1  alpha 0.5   \  BYTE-IDENTICAL rows, so their logits are EXACTLY
//   key 4  alpha 0.5   /  equal in float and in bf16 alike
//   key 2  alpha 1.0
//   key 3  alpha 0.2
//   key 6  alpha 0.7
//   key 5  ALL-NEGATIVE  \  every head dots negative, so the ReLU clamps every
//   key 7  ALL-NEGATIVE  /  one and the logit is EXACTLY 0.0
//
// With `topk = 3` and a causal range, rows 4 and 5 have the 0.5/0.5 tie sitting
// exactly ON the k-th boundary — decided by the smaller-index rule alone, in
// both precisions — and rows 3, 6 and 7 are decided by a STRICT margin. Both
// kinds are therefore present, counted and printed.
struct Fixture {
  std::vector<float> q;        // [T, H, D]
  std::vector<float> k;        // [S, D]
  std::vector<float> weights;  // [T, H]
  std::vector<int32_t> win_start;
  std::vector<int32_t> win_end;
  // Keys whose every head dots NEGATIVE, so their logit is exactly 0.0.
  std::vector<int> zero_keys;
  // Keys built from byte-identical rows, so their logits are exactly equal.
  std::vector<int> tie_keys;
};

Fixture MakeFixture(uint32_t seed) {
  Fixture f;
  const std::vector<float> qn = RandF32(static_cast<size_t>(kTokens) * kHeads * kHeadDim, seed);
  // Every q entry POSITIVE, so an all-negative key row dots negative on every
  // head and the ReLU's fixed point is reachable by construction.
  f.q.resize(qn.size());
  for (size_t i = 0; i < qn.size(); ++i) f.q[i] = 0.25f + 0.5f * (qn[i] + 1.0f);

  // The fixed positive direction every key is built from.
  const std::vector<float> un = RandF32(static_cast<size_t>(kHeadDim), seed + 31u);
  std::vector<float> u(static_cast<size_t>(kHeadDim));
  for (int d = 0; d < kHeadDim; ++d) u[static_cast<size_t>(d)] = 0.5f + 0.5f * (un[static_cast<size_t>(d)] + 1.0f);

  const std::vector<float> noise = RandF32(static_cast<size_t>(kKeys) * kHeadDim, seed + 7919u);
  const float alpha[kKeys] = {0.9f, 0.5f, 1.0f, 0.2f, 0.5f, -1.0f, 0.7f, -1.0f};
  f.zero_keys = {5, 7};
  f.tie_keys = {1, 4};
  f.k.assign(static_cast<size_t>(kKeys) * kHeadDim, 0.0f);
  for (int s = 0; s < kKeys; ++s) {
    for (int d = 0; d < kHeadDim; ++d) {
      const float n = 0.02f * noise[static_cast<size_t>(s) * kHeadDim + d];
      f.k[static_cast<size_t>(s) * kHeadDim + d] = alpha[s] * u[static_cast<size_t>(d)] + n;
    }
  }
  // Make the tie pair and the zero pair BYTE-IDENTICAL, which is what makes
  // their logits exactly equal rather than merely close.
  for (int d = 0; d < kHeadDim; ++d) {
    f.k[static_cast<size_t>(f.tie_keys[1]) * kHeadDim + d] =
        f.k[static_cast<size_t>(f.tie_keys[0]) * kHeadDim + d];
    // An all-negative row: `-u` with the noise driven negative too, so no
    // element can cross zero and no head's dot can come out positive.
    const float neg = -(u[static_cast<size_t>(d)] +
                        0.02f * std::abs(noise[static_cast<size_t>(f.zero_keys[0]) * kHeadDim + d]));
    f.k[static_cast<size_t>(f.zero_keys[0]) * kHeadDim + d] = neg;
    f.k[static_cast<size_t>(f.zero_keys[1]) * kHeadDim + d] = neg;
  }

  // Positive gate weights. `weights_proj` is an unconstrained linear output
  // upstream, but a NEGATIVE gate would let a head subtract and would hide the
  // ReLU's effect behind a sign, so this case wants them positive.
  const std::vector<float> wn = RandF32(static_cast<size_t>(kTokens) * kHeads, seed + 104729u);
  f.weights.resize(wn.size());
  for (size_t i = 0; i < wn.size(); ++i) f.weights[i] = 0.25f + 0.5f * (wn[i] + 1.0f);

  // Causal: row t sees keys [0, t]. Rows 0..kTopk-1 therefore have at most
  // `kTopk` candidates and take the short-context all-select branch; rows
  // kTopk..kTokens-1 really prune.
  f.win_start.assign(kTokens, 0);
  f.win_end.resize(kTokens);
  for (int t = 0; t < kTokens; ++t) f.win_end[static_cast<size_t>(t)] = t + 1;
  return f;
}

DsaIndexerLogitsArgs ScaleArgs() {
  DsaIndexerLogitsArgs a;
  a.softmax_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  a.n_head_scale = 1.0f / std::sqrt(static_cast<float>(kHeads));
  return a;
}

// The ORACLE: the W3 host reference, driven with the same inputs.
std::vector<float> OracleLogits(const Fixture& f) {
  const std::vector<float> folded =
      vllm::deepseek_v4::DsaIndexerWeightFold(f.weights, kTokens, kHeads, kHeadDim);
  std::vector<int64_t> ws(f.win_start.begin(), f.win_start.end());
  std::vector<int64_t> we(f.win_end.begin(), f.win_end.end());
  return vllm::deepseek_v4::DsaIndexerLogits(f.q, f.k, folded, ws, we, kTokens, kKeys, kHeads,
                                             kHeadDim);
}

std::vector<int64_t> OracleTopk(const Fixture& f, const std::vector<float>& logits) {
  std::vector<int64_t> ws(f.win_start.begin(), f.win_start.end());
  std::vector<int64_t> we(f.win_end.begin(), f.win_end.end());
  return vllm::deepseek_v4::DsaTopkSelect(logits, ws, we, kTokens, kKeys, kTopk);
}

// Run both ops on the CPU backend. `bf16` narrows q/k/weights to bf16 first,
// which is the memory format the model path actually carries.
void RunCpu(const Fixture& f, std::vector<float>& logits, std::vector<int32_t>& idx,
            std::vector<int32_t>& cnt, bool bf16) {
  Queue q = CpuQ();
  auto ws = f.win_start;
  auto we = f.win_end;
  logits.assign(static_cast<size_t>(kTokens) * kKeys, 0.0f);
  idx.assign(static_cast<size_t>(kTokens) * kTopk, 0);
  cnt.assign(static_cast<size_t>(kTokens), 0);
  Tensor t_lg = Contig(logits.data(), DType::kF32, Cpu(), {kTokens, kKeys});
  Tensor t_ws = Contig(ws.data(), DType::kI32, Cpu(), {kTokens});
  Tensor t_we = Contig(we.data(), DType::kI32, Cpu(), {kTokens});
  const DsaIndexerLogitsArgs args = ScaleArgs();
  if (bf16) {
    std::vector<uint16_t> qb = ToBf16(f.q), kb = ToBf16(f.k), wb = ToBf16(f.weights);
    Tensor t_q = Contig(qb.data(), DType::kBF16, Cpu(), {kTokens, kHeads, kHeadDim});
    Tensor t_k = Contig(kb.data(), DType::kBF16, Cpu(), {kKeys, kHeadDim});
    Tensor t_w = Contig(wb.data(), DType::kBF16, Cpu(), {kTokens, kHeads});
    vt::DsaIndexerLogits(q, t_lg, t_q, t_k, t_w, t_ws, t_we, args);
  } else {
    auto qq = f.q, kk = f.k, ww = f.weights;
    Tensor t_q = Contig(qq.data(), DType::kF32, Cpu(), {kTokens, kHeads, kHeadDim});
    Tensor t_k = Contig(kk.data(), DType::kF32, Cpu(), {kKeys, kHeadDim});
    Tensor t_w = Contig(ww.data(), DType::kF32, Cpu(), {kTokens, kHeads});
    vt::DsaIndexerLogits(q, t_lg, t_q, t_k, t_w, t_ws, t_we, args);
  }
  Tensor t_idx = Contig(idx.data(), DType::kI32, Cpu(), {kTokens, kTopk});
  Tensor t_cnt = Contig(cnt.data(), DType::kI32, Cpu(), {kTokens});
  vt::DsaTopkSelect(q, t_idx, t_cnt, t_lg, t_ws, t_we);
}

// The selected SET of one row, as a sorted list of live entries.
std::vector<int> RowSet(const std::vector<int32_t>& idx, int t) {
  std::vector<int> v;
  for (int i = 0; i < kTopk; ++i) {
    const int32_t e = idx[static_cast<size_t>(t) * kTopk + i];
    if (e >= 0) v.push_back(static_cast<int>(e));
  }
  std::sort(v.begin(), v.end());
  return v;
}

std::vector<int> OracleRowSet(const std::vector<int64_t>& idx, int t) {
  std::vector<int> v;
  for (int i = 0; i < kTopk; ++i) {
    const int64_t e = idx[static_cast<size_t>(t) * kTopk + i];
    if (e >= 0) v.push_back(static_cast<int>(e));
  }
  std::sort(v.begin(), v.end());
  return v;
}

// The DECISION MARGIN of one row: the gap between the WORST logit that was
// selected and the BEST logit that was not. Zero means an exact tie broken by
// the index rule; anything strictly positive is what a rounding error has to
// cross to flip a slot.
struct Margin {
  bool pruned = false;
  bool exact_tie = false;
  double gap = 0.0;
};

Margin RowMargin(const std::vector<float>& logits, const std::vector<int>& chosen, int t,
                 int win_end) {
  Margin m;
  if (static_cast<int>(chosen.size()) >= win_end) return m;  // nothing was pruned
  m.pruned = true;
  double worst_in = std::numeric_limits<double>::infinity();
  double best_out = -std::numeric_limits<double>::infinity();
  for (int s = 0; s < win_end; ++s) {
    const double v = logits[static_cast<size_t>(t) * kKeys + s];
    if (std::find(chosen.begin(), chosen.end(), s) != chosen.end()) {
      worst_in = std::min(worst_in, v);
    } else {
      best_out = std::max(best_out, v);
    }
  }
  m.gap = worst_in - best_out;
  m.exact_tie = (m.gap == 0.0);
  return m;
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("dsa_indexer_logits CPU reproduces the W3 host reference exactly") {
  const Fixture f = MakeFixture(1301u);
  std::vector<float> got;
  std::vector<int32_t> idx, cnt;
  RunCpu(f, got, idx, cnt, /*bf16=*/false);
  const std::vector<float> want = OracleLogits(f);
  REQUIRE(got.size() == want.size());
  // The op and the host reference perform the SAME f32 operations in the SAME
  // order — per-head dot, ReLU, folded multiply, ascending head sum — so this
  // is an exact-equality assertion rather than a tolerance. A tolerance here
  // would silently absorb a reordered head sum, which is precisely the kind of
  // difference a later tensor-core tiling would introduce and must declare.
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::isinf(want[i])) {
      CHECK(std::isinf(got[i]));
      CHECK(got[i] < 0.0f);
    } else {
      CHECK(got[i] == want[i]);
    }
  }
}

TEST_CASE("dsa_indexer_logits: one logit BY VALUE, from first principles") {
  // The fold is `weights * q_scale * softmax_scale * n_head_scale`
  // (deepseek_v2.py:840). `softmax_scale` and `n_head_scale` are GLOBAL
  // POSITIVE scalars, so dropping either cannot move an argmax and their
  // mutations read green DEFINITIONALLY — not because the gate has a hole.
  // This case is what covers them anyway: it recomputes one logit in double,
  // here, from the upstream formula, and compares the VALUE.
  const Fixture f = MakeFixture(1301u);
  std::vector<float> got;
  std::vector<int32_t> idx, cnt;
  RunCpu(f, got, idx, cnt, /*bf16=*/false);

  const double softmax_scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  const double n_head_scale = 1.0 / std::sqrt(static_cast<double>(kHeads));
  const int t = kTokens - 1;  // the row with the most candidates
  const int s = 3;
  double acc = 0.0;
  for (int h = 0; h < kHeads; ++h) {
    double dot = 0.0;
    for (int d = 0; d < kHeadDim; ++d) {
      dot += static_cast<double>(f.q[(static_cast<size_t>(t) * kHeads + h) * kHeadDim + d]) *
             f.k[static_cast<size_t>(s) * kHeadDim + d];
    }
    const double relu = dot > 0.0 ? dot : 0.0;
    acc += static_cast<double>(f.weights[static_cast<size_t>(t) * kHeads + h]) * softmax_scale *
           n_head_scale * relu;
  }
  const double seen = got[static_cast<size_t>(t) * kKeys + s];
  MESSAGE("logit[" << t << "," << s << "] = " << seen << " against " << acc
                   << " recomputed in double");
  CHECK(std::abs(seen - acc) < 1e-6 * std::max(1.0, std::abs(acc)));
  // And the scale is not 1: a fold dropped entirely would land on the raw
  // ReLU'd sum, which this asserts is a DIFFERENT number.
  CHECK(std::abs(acc / (softmax_scale * n_head_scale) - acc) > 1e-3);
}

TEST_CASE("dsa_indexer_logits: the ReLU is load-bearing, and it produces an EXACT 0.0") {
  // `tl.maximum(scores, 0.0)` (triton_fp8_mqa_logits.py:129,:150). A key whose
  // every head dots NEGATIVE contributes nothing at all, so its logit is
  // exactly 0.0 — representable in float and bf16 alike. Without the ReLU it
  // would be a strictly negative number, so this is a statement about the
  // mechanism and needs no reference.
  const Fixture f = MakeFixture(1301u);
  std::vector<float> got;
  std::vector<int32_t> idx, cnt;
  RunCpu(f, got, idx, cnt, /*bf16=*/false);
  int zeros = 0;
  for (int zk : f.zero_keys) {
    for (int t = zk; t < kTokens; ++t) {  // rows that can see this key
      const float v = got[static_cast<size_t>(t) * kKeys + zk];
      CHECK(v == 0.0f);
      ++zeros;
    }
  }
  MESSAGE("ReLU fixed point: " << zeros << " (row, key) logits are EXACTLY 0.0");
  REQUIRE(zeros > 0);
  // The control: a key that is NOT anti-correlated scores strictly positive on
  // at least one row, so the zeros above are the ReLU rather than a dead op.
  double best = 0.0;
  for (int t = 0; t < kTokens; ++t) {
    for (int s = 0; s <= t; ++s) {
      if (std::find(f.zero_keys.begin(), f.zero_keys.end(), s) != f.zero_keys.end()) continue;
      best = std::max(best, static_cast<double>(got[static_cast<size_t>(t) * kKeys + s]));
    }
  }
  REQUIRE(best > 0.0);
}

TEST_CASE("dsa_topk_select CPU: the SET matches the reference, and the margins are measured") {
  const Fixture f = MakeFixture(1301u);
  std::vector<float> logits;
  std::vector<int32_t> idx, cnt;
  RunCpu(f, logits, idx, cnt, /*bf16=*/false);
  const std::vector<int64_t> want = OracleTopk(f, OracleLogits(f));

  int rows_pruned = 0, keys_dropped = 0, exact_ties = 0;
  double min_strict_margin = std::numeric_limits<double>::infinity();
  double max_abs_logit = 0.0;
  for (int t = 0; t < kTokens; ++t) {
    // (1) SELECTION-SET EQUALITY as its own DISCRETE assertion. A value
    //     tolerance says nothing about a slot flip; this does.
    const std::vector<int> mine = RowSet(idx, t);
    const std::vector<int> theirs = OracleRowSet(want, t);
    CHECK(mine == theirs);
    const int n = f.win_end[static_cast<size_t>(t)];
    CHECK(cnt[static_cast<size_t>(t)] == std::min(n, kTopk));
    // The `-1` tail is the "no token" sentinel, and it must be there.
    for (int i = static_cast<int>(mine.size()); i < kTopk; ++i) {
      CHECK(idx[static_cast<size_t>(t) * kTopk + i] == -1);
    }
    // The emission order is ASCENDING, which is what makes a full selection
    // reproduce the dense reduction order in vt::MlaDecodeAttention.
    for (int i = 1; i < static_cast<int>(mine.size()); ++i) {
      CHECK(idx[static_cast<size_t>(t) * kTopk + i] >
            idx[static_cast<size_t>(t) * kTopk + i - 1]);
    }

    for (int s = 0; s < n; ++s) {
      max_abs_logit =
          std::max(max_abs_logit, std::abs(static_cast<double>(logits[static_cast<size_t>(t) * kKeys + s])));
    }
    const Margin m = RowMargin(logits, mine, t, n);
    if (!m.pruned) continue;
    ++rows_pruned;
    keys_dropped += n - static_cast<int>(mine.size());
    if (m.exact_tie) {
      ++exact_ties;
    } else {
      min_strict_margin = std::min(min_strict_margin, m.gap);
    }
  }

  // (3) THE INSTRUMENT SAYS WHAT IT MEASURED. Below the pruning threshold every
  //     assertion above passes on an implementation that performs no selection
  //     at all, so the counts are printed and then required.
  MESSAGE("dsa_topk_select: " << rows_pruned << " of " << kTokens << " rows prune, "
                              << keys_dropped << " keys dropped, " << exact_ties
                              << " boundary decisions are EXACT ties");
  REQUIRE(rows_pruned == kTokens - kTopk);
  REQUIRE(keys_dropped == 1 + 2 + 3 + 4 + 5);  // rows 3..7 see 4..8 keys, keep 3
  // (4) BOTH kinds of boundary decision are present. Rows 4 and 5 have the
  //     0.5/0.5 alpha pair sitting exactly ON the k-th boundary, so they are
  //     decided by the smaller-index rule alone; rows 3, 6 and 7 are decided by
  //     a strict margin. A fixture with only one kind would leave the other
  //     rule untested.
  REQUIRE(exact_ties == 2);
  REQUIRE(rows_pruned - exact_ties == 3);

  // (2) THE MARGIN, MEASURED AGAINST THE WORKING PRECISION'S ULP. The logits
  //     are f32 and the model path narrows the OPERANDS to bf16, so the margin
  //     has to clear the bf16 half-ulp at this fixture's own logit scale — not
  //     an ulp quoted from somewhere else.
  const double f32_ulp = std::nextafter(max_abs_logit, 1e30) - max_abs_logit;
  const double bf16_ulp = max_abs_logit * std::pow(2.0, -8);  // bf16 carries 8 mantissa bits
  MESSAGE("dsa_topk_select margins: min strict margin "
          << min_strict_margin << " against |logit|max " << max_abs_logit << " (f32 ulp "
          << f32_ulp << ", bf16 ulp " << bf16_ulp << ")");
  REQUIRE(std::isfinite(min_strict_margin));
  REQUIRE(bf16_ulp > 0.0);
  // A stated multiple, so the criterion is committed rather than fitted: the
  // strict margins must clear FOUR bf16 ulps at this scale. Four rather than
  // one because a logit is a sum of `kHeads` rounded products, so its own error
  // is a small multiple of one ulp.
  CHECK(min_strict_margin > 4.0 * bf16_ulp);
}

TEST_CASE("dsa_topk_select: BF16 operands select the IDENTICAL set") {
  // This is what MEASURES the fixture's adequacy instead of assuming it. The
  // model path carries bf16 q/k/weights; if any decision margin were inside the
  // bf16 floor, this case would flip a slot and fail. It is the executable form
  // of §4.5's warning that a 1.29e-3 margin is SMALLER than a bf16 logit's
  // ~4e-3 ulp at order 1.
  const Fixture f = MakeFixture(1301u);
  std::vector<float> lg_f32, lg_bf16;
  std::vector<int32_t> idx_f32, cnt_f32, idx_bf16, cnt_bf16;
  RunCpu(f, lg_f32, idx_f32, cnt_f32, /*bf16=*/false);
  RunCpu(f, lg_bf16, idx_bf16, cnt_bf16, /*bf16=*/true);
  CHECK(idx_f32 == idx_bf16);
  CHECK(cnt_f32 == cnt_bf16);
  // The two logit VECTORS are NOT expected to be equal — narrowing the operands
  // is a real numerical change — and that is asserted, so the case above cannot
  // be passing because the bf16 arm silently ran in f32.
  double worst = 0.0;
  for (size_t i = 0; i < lg_f32.size(); ++i) {
    if (std::isinf(lg_f32[i])) continue;
    worst = std::max(worst, std::abs(static_cast<double>(lg_f32[i]) - lg_bf16[i]));
  }
  MESSAGE("bf16 operands move the logits by at most " << worst
                                                      << " and change ZERO selections");
  REQUIRE(worst > 0.0);

  // The zero-logit keys stay EXACTLY 0.0 under bf16 too — a sign cannot be
  // rounded away — which is what makes the ReLU's fixed point precision-free.
  for (int zk : f.zero_keys) {
    for (int t = zk; t < kTokens; ++t) {
      CHECK(lg_bf16[static_cast<size_t>(t) * kKeys + zk] == 0.0f);
    }
  }
}

TEST_CASE("dsa_topk_select: an exact tie is broken by the SMALLER key index") {
  // Keys 1 and 4 are BYTE-IDENTICAL rows, so their logits are exactly equal in
  // float and in bf16 alike, and on rows 4 and 5 that pair sits exactly ON the
  // k-th boundary: whichever of them is kept is decided by the index rule and
  // by nothing else. This is the case that pins the rule, and it cannot be
  // passed by a selection whose tie-break is the sort's whim.
  const Fixture f = MakeFixture(1301u);
  std::vector<float> logits;
  std::vector<int32_t> idx, cnt;
  RunCpu(f, logits, idx, cnt, /*bf16=*/false);
  const int lo = f.tie_keys[0], hi = f.tie_keys[1];
  for (int t : {4, 5}) {
    const std::vector<int> chosen = RowSet(idx, t);
    const bool has_lo = std::find(chosen.begin(), chosen.end(), lo) != chosen.end();
    const bool has_hi = std::find(chosen.begin(), chosen.end(), hi) != chosen.end();
    MESSAGE("tie row " << t << ": logits[" << lo << "]="
                       << logits[static_cast<size_t>(t) * kKeys + lo] << " logits[" << hi
                       << "]=" << logits[static_cast<size_t>(t) * kKeys + hi] << " keeps key"
                       << lo << "=" << has_lo << " key" << hi << "=" << has_hi);
    REQUIRE(logits[static_cast<size_t>(t) * kKeys + lo] ==
            logits[static_cast<size_t>(t) * kKeys + hi]);
    // The SMALLER index wins the tie: the pair competes for the last slot, so
    // exactly one of them is kept and it is the smaller one.
    CHECK(has_lo);
    CHECK(!has_hi);
  }
}

TEST_CASE("dsa_topk_select: a SHORT context selects every candidate, ascending") {
  // The branch that makes a sparse layer identical to dense attention while the
  // whole context fits in `index_topk` — the regime upstream keeps its dense
  // MHA prefill for (`use_dense_mha = prefill_max_seq_len <= self.topk_tokens`,
  // sparse_mla_attention.py:296-299 @ bc2d63e650).
  const Fixture f = MakeFixture(1301u);
  std::vector<float> logits;
  std::vector<int32_t> idx, cnt;
  RunCpu(f, logits, idx, cnt, /*bf16=*/false);
  for (int t = 0; t < kTopk; ++t) {
    const int n = f.win_end[static_cast<size_t>(t)];
    REQUIRE(n <= kTopk);
    CHECK(cnt[static_cast<size_t>(t)] == n);
    for (int i = 0; i < n; ++i) CHECK(idx[static_cast<size_t>(t) * kTopk + i] == i);
    for (int i = n; i < kTopk; ++i) CHECK(idx[static_cast<size_t>(t) * kTopk + i] == -1);
  }
}

TEST_CASE("dsa_indexer ops reject malformed operands") {
  const Fixture f = MakeFixture(1301u);
  Queue q = CpuQ();
  auto qq = f.q, kk = f.k, ww = f.weights;
  auto ws = f.win_start, we = f.win_end;
  std::vector<float> logits(static_cast<size_t>(kTokens) * kKeys, 0.0f);
  Tensor t_q = Contig(qq.data(), DType::kF32, Cpu(), {kTokens, kHeads, kHeadDim});
  Tensor t_k = Contig(kk.data(), DType::kF32, Cpu(), {kKeys, kHeadDim});
  Tensor t_w = Contig(ww.data(), DType::kF32, Cpu(), {kTokens, kHeads});
  Tensor t_ws = Contig(ws.data(), DType::kI32, Cpu(), {kTokens});
  Tensor t_we = Contig(we.data(), DType::kI32, Cpu(), {kTokens});
  Tensor t_lg = Contig(logits.data(), DType::kF32, Cpu(), {kTokens, kKeys});
  const DsaIndexerLogitsArgs args = ScaleArgs();

  SUBCASE("k must be rank-2 — the indexer is MQA") {
    Tensor bad = Contig(kk.data(), DType::kF32, Cpu(), {kKeys, 1, kHeadDim});
    CHECK_THROWS_WITH_AS(vt::DsaIndexerLogits(q, t_lg, t_q, bad, t_w, t_ws, t_we, args),
                         doctest::Contains("rank-2"), std::runtime_error);
  }
  SUBCASE("k's width must equal q's index_head_dim") {
    Tensor bad = Contig(kk.data(), DType::kF32, Cpu(), {kKeys * 2, kHeadDim / 2});
    CHECK_THROWS_WITH_AS(vt::DsaIndexerLogits(q, t_lg, t_q, bad, t_w, t_ws, t_we, args),
                         doctest::Contains("index_head_dim"), std::runtime_error);
  }
  SUBCASE("logits must be f32") {
    std::vector<uint16_t> h(static_cast<size_t>(kTokens) * kKeys, 0);
    Tensor bad = Contig(h.data(), DType::kBF16, Cpu(), {kTokens, kKeys});
    CHECK_THROWS_WITH_AS(vt::DsaIndexerLogits(q, bad, t_q, t_k, t_w, t_ws, t_we, args),
                         doctest::Contains("must be f32"), std::runtime_error);
  }
  SUBCASE("q/k/weights must share one float dtype") {
    std::vector<uint16_t> h(ww.size(), 0);
    Tensor bad = Contig(h.data(), DType::kBF16, Cpu(), {kTokens, kHeads});
    CHECK_THROWS_WITH_AS(vt::DsaIndexerLogits(q, t_lg, t_q, t_k, bad, t_ws, t_we, args),
                         doctest::Contains("one float dtype"), std::runtime_error);
  }
  SUBCASE("both scales must be positive") {
    DsaIndexerLogitsArgs bad = args;
    bad.n_head_scale = 0.0f;
    CHECK_THROWS_WITH_AS(vt::DsaIndexerLogits(q, t_lg, t_q, t_k, t_w, t_ws, t_we, bad),
                         doctest::Contains("n_head_scale"), std::runtime_error);
  }
  SUBCASE("q_scale must be [num_tokens, index_n_heads] f32") {
    std::vector<float> bad_scale(static_cast<size_t>(kTokens) * kHeads, 1.0f);
    Tensor bad = Contig(bad_scale.data(), DType::kF32, Cpu(), {kTokens, kHeads - 1});
    DsaIndexerLogitsArgs a = args;
    a.q_scale = &bad;
    CHECK_THROWS_WITH_AS(vt::DsaIndexerLogits(q, t_lg, t_q, t_k, t_w, t_ws, t_we, a),
                         doctest::Contains("q_scale"), std::runtime_error);
  }
  SUBCASE("topk must be > 0") {
    std::vector<int32_t> idx(1, 0), cnt(kTokens, 0);
    Tensor t_idx = Contig(idx.data(), DType::kI32, Cpu(), {kTokens, 0});
    Tensor t_cnt = Contig(cnt.data(), DType::kI32, Cpu(), {kTokens});
    CHECK_THROWS_WITH_AS(vt::DsaTopkSelect(q, t_idx, t_cnt, t_lg, t_ws, t_we),
                         doctest::Contains("topk must be > 0"), std::runtime_error);
  }
  SUBCASE("indices/counts must be i32") {
    std::vector<float> bad_idx(static_cast<size_t>(kTokens) * kTopk, 0.0f);
    std::vector<int32_t> cnt(kTokens, 0);
    Tensor t_idx = Contig(bad_idx.data(), DType::kF32, Cpu(), {kTokens, kTopk});
    Tensor t_cnt = Contig(cnt.data(), DType::kI32, Cpu(), {kTokens});
    CHECK_THROWS_WITH_AS(vt::DsaTopkSelect(q, t_idx, t_cnt, t_lg, t_ws, t_we),
                         doctest::Contains("must be i32"), std::runtime_error);
  }
}

// ───────────────────────────────────────────────────────────────────────────
// CUDA. The device arm exists so a model forward can reach the selection
// without a host round-trip; it is gated against the CPU arm, which is itself
// gated against the W3 host reference above.
// ───────────────────────────────────────────────────────────────────────────
namespace {

void RunCuda(const Fixture& f, std::vector<float>& logits, std::vector<int32_t>& idx,
             std::vector<int32_t>& cnt, bool bf16) {
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  const std::vector<uint16_t> qb = ToBf16(f.q), kb = ToBf16(f.k), wb = ToBf16(f.weights);
  const DType dt = bf16 ? DType::kBF16 : DType::kF32;
  DeviceTensor d_q(b, g.q, dt, {kTokens, kHeads, kHeadDim},
                   bf16 ? static_cast<const void*>(qb.data())
                        : static_cast<const void*>(f.q.data()));
  DeviceTensor d_k(b, g.q, dt, {kKeys, kHeadDim},
                   bf16 ? static_cast<const void*>(kb.data())
                        : static_cast<const void*>(f.k.data()));
  DeviceTensor d_w(b, g.q, dt, {kTokens, kHeads},
                   bf16 ? static_cast<const void*>(wb.data())
                        : static_cast<const void*>(f.weights.data()));
  DeviceTensor d_ws(b, g.q, DType::kI32, {kTokens}, f.win_start.data());
  DeviceTensor d_we(b, g.q, DType::kI32, {kTokens}, f.win_end.data());
  DeviceTensor d_lg(b, g.q, DType::kF32, {kTokens, kKeys});
  DeviceTensor d_idx(b, g.q, DType::kI32, {kTokens, kTopk});
  DeviceTensor d_cnt(b, g.q, DType::kI32, {kTokens});

  vt::DsaIndexerLogits(g.q, d_lg.tensor(), d_q.tensor(), d_k.tensor(), d_w.tensor(),
                       d_ws.tensor(), d_we.tensor(), ScaleArgs());
  vt::DsaTopkSelect(g.q, d_idx.tensor(), d_cnt.tensor(), d_lg.tensor(), d_ws.tensor(),
                    d_we.tensor());
  b.Synchronize(g.q);
  logits.assign(static_cast<size_t>(kTokens) * kKeys, 0.0f);
  idx.assign(static_cast<size_t>(kTokens) * kTopk, 0);
  cnt.assign(static_cast<size_t>(kTokens), 0);
  d_lg.Download(g.q, logits.data());
  d_idx.Download(g.q, idx.data());
  d_cnt.Download(g.q, cnt.data());
}

}  // namespace

TEST_CASE("CUDA dsa_indexer: the SELECTION SET is identical to the CPU arm") {
  if (!HasCuda()) return;
  const Fixture f = MakeFixture(1301u);
  std::vector<float> cpu_lg, gpu_lg;
  std::vector<int32_t> cpu_idx, cpu_cnt, gpu_idx, gpu_cnt;
  RunCpu(f, cpu_lg, cpu_idx, cpu_cnt, /*bf16=*/false);
  RunCuda(f, gpu_lg, gpu_idx, gpu_cnt, /*bf16=*/false);

  // The DISCRETE assertion first. The device sums the heads in a different
  // order (warp-strided, then a fixed tree over the per-warp partials), so the
  // logits are NOT expected bit-identical to the CPU arm; the SELECTION is,
  // and that is the property the model depends on.
  CHECK(gpu_idx == cpu_idx);
  CHECK(gpu_cnt == cpu_cnt);

  double worst = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < cpu_lg.size(); ++i) {
    if (std::isinf(cpu_lg[i])) {
      CHECK(std::isinf(gpu_lg[i]));
      CHECK(gpu_lg[i] < 0.0f);
      continue;
    }
    worst = std::max(worst, std::abs(static_cast<double>(cpu_lg[i]) - gpu_lg[i]));
    max_abs = std::max(max_abs, std::abs(static_cast<double>(cpu_lg[i])));
  }
  MESSAGE("CUDA dsa_indexer_logits: max |gpu - cpu| = " << worst << " at |logit|max "
                                                        << max_abs);
  CHECK(worst < 1e-5 * std::max(1.0, max_abs));
  // The ReLU fixed point survives the different reduction order exactly.
  for (int zk : f.zero_keys) {
    for (int t = zk; t < kTokens; ++t) {
      CHECK(gpu_lg[static_cast<size_t>(t) * kKeys + zk] == 0.0f);
    }
  }
}

TEST_CASE("CUDA dsa_indexer: BF16 operands select the IDENTICAL set on the device too") {
  if (!HasCuda()) return;
  const Fixture f = MakeFixture(1301u);
  std::vector<float> lg32, lg16;
  std::vector<int32_t> idx32, cnt32, idx16, cnt16;
  RunCuda(f, lg32, idx32, cnt32, /*bf16=*/false);
  RunCuda(f, lg16, idx16, cnt16, /*bf16=*/true);
  CHECK(idx32 == idx16);
  CHECK(cnt32 == cnt16);
  double worst = 0.0;
  for (size_t i = 0; i < lg32.size(); ++i) {
    if (std::isinf(lg32[i])) continue;
    worst = std::max(worst, std::abs(static_cast<double>(lg32[i]) - lg16[i]));
  }
  MESSAGE("CUDA bf16 operands move the logits by at most " << worst
                                                           << " and change ZERO selections");
  REQUIRE(worst > 0.0);
}
