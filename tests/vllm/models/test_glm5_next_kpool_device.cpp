// GLM-5.3-Flash W9c-0 gate — the k-pool DSA indexer's two DEVICE ops, against
// the transformers v5.16.1 RUN goldens and the landed host reference.
//
// Row MODEL-MM-glm5-next-glm5-next-for-conditional-generation, issue #2415,
// `.agents/specs/glm5-next-flash.md` section W9c-0.
//
// ─── WHAT IS UNDER TEST, AND AGAINST WHAT ───────────────────────────────────
//
// `vt::Glm5NextKpoolCompress` and `vt::Glm5NextKpoolSelect` are the CUDA arm of
// the k-pool indexer. There is no CPU provider and there deliberately is not
// one: the CPU answer already exists as `glm5_next_dsa.cpp`, gated by
// `test_glm5_next_dsa` against the same goldens, and registering it a second
// time under an `OpId` would make the seam its own oracle.
//
// So this file compares the device arm against TWO independent things:
//   * `glm5_next_dsa_goldens.inc` — the RUN output of the unmodified
//     `Glm5NextTextIndexer` at transformers v5.16.1, captured by
//     `fixtures/gen_glm5_next_dsa_goldens.py`. Nothing in it is transcribed
//     from our C++, and it carries the INTERMEDIATES (`kPoolKeys`,
//     `kPoolIndices`, `kPoolValid`, `kIndexScores`) as well as the result.
//   * `vllm::glm5_next::GetPooledStates` / `SelectIndexerTopkFromPacked` — the
//     host reference, which accumulates in `double` where the device (and
//     upstream, `modular_glm5_next.py:823,960-964`) accumulate in fp32.
//
// ─── THREE PROPERTIES, EACH LOAD-BEARING ────────────────────────────────────
//
//  1. `seq_len` 21 against `index_topk` 8. At or below `index_topk` the
//     selection is the identity and the pooling is unobservable. Row 1 is
//     LEFT-PADDED by three, so its pool grid starts at token 3 and a kernel
//     that pools from slot 0 passes row 0 and fails row 1. And
//     `np = ceil(21/4) = 6` against `kNumPools = 5`: the fixture EXERCISES the
//     `keep` compaction (`modular_glm5_next.py:968-970`) rather than assuming
//     it, because a kernel that skipped it would put the ragged tail at column
//     12 instead of 8 and miss the `[..., :output_width]` truncation entirely.
//
//  2. Selection error is BIMODAL, not continuous. A wrong-but-adjacent pool's
//     scores can be arbitrarily close, so a tolerance on `index_scores` bounds
//     nothing about which pools were chosen. Every selection assertion here is
//     SET equality of the real token indices PLUS the positionwise comparison,
//     and the minimum decision margin between the `select_k`-th and
//     `select_k + 1`-th masked score is COMPUTED AND PRINTED rather than
//     assumed adequate.
//
//  3. Every float comparison is `isfinite`-guarded on BOTH operands before it
//     is made. An all-NaN forward on this row once read as a perfect match,
//     because every comparison against NaN is false, and the model then emitted
//     token id 0 eight times. A NaN in either arm reds this file.
//
// The device cases SKIP on a CPU-only build (no CUDA backend registered). A
// doctest `assertions: 0` line is a skip wearing a pass; the device job reads
// that line rather than the exit code.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "glm5_next_dsa_goldens.inc"
#include "vllm/model_executor/models/glm5_next_device.h"
#include "vllm/model_executor/models/glm5_next_dsa.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace g = glm5_next_dsa_goldens;

using vllm::glm5_next::GetPooledStates;
using vllm::glm5_next::IndexerDims;
using vllm::glm5_next::IndexerSelection;
using vllm::glm5_next::IndexerWeights;
using vllm::glm5_next::PackIndexerStates;
using vllm::glm5_next::PooledStates;
using vllm::glm5_next::SelectIndexerTopkFromPacked;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

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
    return vllm::glm5_next::KpoolDeviceOpsAvailable();
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

// ─── the fixture, read off the goldens rather than restated ─────────────────

IndexerDims FixtureDims() {
  IndexerDims d;
  d.hidden_size = g::kHidden;
  d.q_lora_rank = g::kQLora;
  d.n_heads = g::kNHeads;
  d.head_dim = g::kHeadDim;
  d.index_topk = g::kIndexTopk;
  d.index_kpool = g::kIndexKpool;
  d.always_select_tail = true;
  return d;
}

IndexerWeights FixtureWeights() {
  IndexerWeights w;
  w.wq_b = g::kWqB;
  w.wk = g::kWk;
  w.k_norm_weight = g::kKNormWeight;
  w.k_norm_bias = g::kKNormBias;
  w.weights_proj = g::kWeightsProj;
  w.kpool_ape = g::kKpoolApe;
  w.kpool_gate = g::kKpoolGate;
  return w;
}

std::vector<uint8_t> MaskBytes(const int32_t* a, size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = a[i] != 0 ? 1 : 0;
  return v;
}

// `nn.Linear` on ONE row, fp32 — the same arithmetic W9c-3 will get from
// `vt::MatmulBT`. It builds the two operands the select op takes and the host
// reference builds internally, so both arms see byte-identical inputs.
std::vector<float> LinearRows(const float* w, const std::vector<float>& x, int64_t rows,
                              int64_t out_features, int64_t in_features) {
  std::vector<float> out(static_cast<size_t>(rows * out_features), 0.0f);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_features; ++o) {
      const float* wr = w + o * in_features;
      const float* xr = x.data() + r * in_features;
      double acc = 0.0;
      for (int64_t i = 0; i < in_features; ++i)
        acc += static_cast<double>(wr[i]) * static_cast<double>(xr[i]);
      out[static_cast<size_t>(r * out_features + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

// ─── the guarded comparisons ────────────────────────────────────────────────
//
// `Finite` is checked on BOTH operands FIRST and asserted, never folded into the
// tolerance: `std::abs(nan - nan) <= tol` is false, so an unguarded CHECK on two
// NaNs reports a failure, but an unguarded `CHECK(a == b || close)` on one NaN
// silently reports nothing at all. The pattern this file uses reds on a NaN.
void RequireFinite(const std::vector<float>& v, const char* what) {
  size_t bad = 0;
  for (float x : v)
    if (!std::isfinite(x)) ++bad;
  INFO("non-finite entries in " << what << ": " << bad << " of " << v.size());
  REQUIRE(bad == 0);
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(std::isfinite(a[i]));
    REQUIRE(std::isfinite(b[i]));
    worst = std::max(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return worst;
}

// The set of REAL token indices a row selected. `-1` is the invalid sentinel and
// duplicates are possible — upstream absorbs them with `scatter_add_` + `ne(0)`
// (`modular_glm5_next.py:1119-1129`) — so a set is the model's own semantics.
// The positionwise comparison is asserted separately and is strictly stronger.
std::set<int32_t> SelectedSet(const int32_t* row, int64_t width) {
  std::set<int32_t> s;
  for (int64_t i = 0; i < width; ++i)
    if (row[i] >= 0) s.insert(row[i]);
  return s;
}

// The decision margin of one query row: the gap between the score of the
// `select_k`-th pool and the best pool that did NOT make the cut. A top-k error
// either flips this gap or it does not, so the gate reports the smallest gap the
// fixture offers instead of asserting a tolerance that bounds nothing.
double RowMargin(const std::vector<float>& scores, const std::vector<uint8_t>& candidate,
                 int64_t num_pools, int64_t select_k) {
  std::vector<double> masked;
  masked.reserve(static_cast<size_t>(num_pools));
  for (int64_t p = 0; p < num_pools; ++p) {
    if (candidate[static_cast<size_t>(p)] == 0) continue;
    masked.push_back(static_cast<double>(scores[static_cast<size_t>(p)]));
  }
  if (static_cast<int64_t>(masked.size()) <= select_k) return -1.0;  // no boundary to cross
  std::sort(masked.begin(), masked.end(), std::greater<double>());
  return masked[static_cast<size_t>(select_k) - 1] - masked[static_cast<size_t>(select_k)];
}

}  // namespace

TEST_CASE("glm5_next k-pool: the device ops are registered on CUDA and absent on CPU") {
  // The registration is the seam. `KpoolDeviceOpsAvailable` is the probe
  // `deepseek_v4_device.cpp:30-35` sets the shape of, and W9c-3's forward is
  // meant to consult it BEFORE it builds operands rather than after it throws.
  //
  // The first CHECK compares the probe against the two lookups it is made of,
  // so it measures CONSISTENCY and not correctness — said plainly rather than
  // dressed up. What it does buy is the `&&`: a probe that lost one of its two
  // clauses would report the family available with half of it registered, and
  // the reviewer's mutation of this file deletes a clause to show that.
  const bool cuda_present = vt::OpRegistered(vt::OpId::kGlm5NextKpoolCompress,
                                             DeviceType::kCUDA) &&
                            vt::OpRegistered(vt::OpId::kGlm5NextKpoolSelect,
                                             DeviceType::kCUDA);
  CHECK(vllm::glm5_next::KpoolDeviceOpsAvailable() == cuda_present);

  // No CPU provider, on purpose: the CPU answer is `glm5_next_dsa.cpp` and it is
  // this file's oracle, not a second registration of itself.
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kGlm5NextKpoolCompress, DeviceType::kCPU));
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kGlm5NextKpoolSelect, DeviceType::kCPU));

  // So a CPU queue is refused by name rather than served a wrong answer.
  const IndexerDims d = FixtureDims();
  std::vector<float> packed_dummy(
      static_cast<size_t>(g::kBatch * g::kSeqLen * (2 * d.head_dim + 1)), 0.0f);
  std::vector<float> ape_dummy(static_cast<size_t>(d.index_kpool * d.head_dim), 0.0f);
  const int64_t np = (g::kSeqLen + d.index_kpool - 1) / d.index_kpool;
  std::vector<float> keys_dummy(static_cast<size_t>(g::kBatch * np * d.head_dim), 0.0f);
  std::vector<int32_t> idx_dummy(static_cast<size_t>(g::kBatch * np * d.index_kpool), -1);
  std::vector<int32_t> valid_dummy(static_cast<size_t>(g::kBatch * np), 0);
  std::vector<int32_t> np_dummy(1, 0);

  Queue cpu_q{Cpu(), nullptr};
  Tensor t_packed = Contig(packed_dummy.data(), DType::kF32, Cpu(),
                           {g::kBatch, g::kSeqLen, 2 * d.head_dim + 1});
  Tensor t_ape = Contig(ape_dummy.data(), DType::kF32, Cpu(), {d.index_kpool, d.head_dim});
  Tensor t_keys = Contig(keys_dummy.data(), DType::kF32, Cpu(), {g::kBatch, np, d.head_dim});
  Tensor t_idx =
      Contig(idx_dummy.data(), DType::kI32, Cpu(), {g::kBatch, np, d.index_kpool});
  Tensor t_valid = Contig(valid_dummy.data(), DType::kI32, Cpu(), {g::kBatch, np});
  Tensor t_np = Contig(np_dummy.data(), DType::kI32, Cpu(), {1});
  CHECK_THROWS(
      vt::Glm5NextKpoolCompress(cpu_q, t_keys, t_idx, t_valid, t_np, t_packed, t_ape));
}

TEST_CASE("glm5_next k-pool device: compress agrees with the transformers goldens") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend: the k-pool compress device gate is SKIPPED");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(backend);

  const IndexerDims d = FixtureDims();
  const IndexerWeights w = FixtureWeights();
  const std::vector<float> hidden(g::kHiddenStates,
                                  g::kHiddenStates + g::kBatch * g::kSeqLen * g::kHidden);
  const std::vector<uint8_t> mask =
      MaskBytes(g::kMask, static_cast<size_t>(g::kBatch * g::kSeqLen));
  const std::vector<float> packed =
      PackIndexerStates(d, w, hidden, mask, g::kBatch, g::kSeqLen);
  RequireFinite(packed, "packed indexer states");

  const int64_t np = (g::kSeqLen + d.index_kpool - 1) / d.index_kpool;
  REQUIRE(np == 6);  // and kNumPools is 5, so the `keep` compaction is exercised

  const std::vector<float> ape(g::kKpoolApe,
                               g::kKpoolApe + d.index_kpool * d.head_dim);
  DeviceTensor dev_packed(backend, qg.q, DType::kF32,
                          {g::kBatch, g::kSeqLen, 2 * d.head_dim + 1}, packed.data());
  DeviceTensor dev_ape(backend, qg.q, DType::kF32, {d.index_kpool, d.head_dim}, ape.data());
  DeviceTensor dev_keys(backend, qg.q, DType::kF32, {g::kBatch, np, d.head_dim});
  DeviceTensor dev_idx(backend, qg.q, DType::kI32, {g::kBatch, np, d.index_kpool});
  DeviceTensor dev_valid(backend, qg.q, DType::kI32, {g::kBatch, np});
  DeviceTensor dev_np(backend, qg.q, DType::kI32, {1});

  vt::Glm5NextKpoolCompress(qg.q, dev_keys.tensor(), dev_idx.tensor(), dev_valid.tensor(),
                            dev_np.tensor(), dev_packed.tensor(), dev_ape.tensor());
  backend.Synchronize(qg.q);

  std::vector<int32_t> got_np(1, -1);
  dev_np.Download(qg.q, got_np.data());
  INFO("device P = " << got_np[0] << ", golden kNumPools = " << g::kNumPools);
  REQUIRE(got_np[0] == static_cast<int32_t>(g::kNumPools));

  const int64_t P = g::kNumPools;
  std::vector<float> got_keys(static_cast<size_t>(g::kBatch * np * d.head_dim));
  std::vector<int32_t> got_idx(static_cast<size_t>(g::kBatch * np * d.index_kpool));
  std::vector<int32_t> got_valid(static_cast<size_t>(g::kBatch * np));
  dev_keys.Download(qg.q, got_keys.data());
  dev_idx.Download(qg.q, got_idx.data());
  dev_valid.Download(qg.q, got_valid.data());

  // Only the COMPACTED prefix [0, P) carries meaning; the slack up to `np` is
  // allocation, and the ops contract says nothing about it.
  std::vector<float> keys_pref, golden_keys(g::kPoolKeys, g::kPoolKeys + g::kBatch * P * d.head_dim);
  for (int64_t b = 0; b < g::kBatch; ++b)
    for (int64_t p = 0; p < P; ++p)
      for (int64_t c = 0; c < d.head_dim; ++c)
        keys_pref.push_back(got_keys[static_cast<size_t>((b * np + p) * d.head_dim + c)]);
  RequireFinite(keys_pref, "device pool_keys");
  const double keys_delta = MaxAbsDiff(keys_pref, golden_keys);
  MESSAGE("pool_keys max|device - transformers| = " << keys_delta);
  CHECK(keys_delta < 2e-5);

  for (int64_t b = 0; b < g::kBatch; ++b) {
    for (int64_t p = 0; p < P; ++p) {
      CHECK(got_valid[static_cast<size_t>(b * np + p)] ==
            g::kPoolValid[static_cast<size_t>(b * P + p)]);
      for (int64_t m = 0; m < d.index_kpool; ++m) {
        CHECK(got_idx[static_cast<size_t>((b * np + p) * d.index_kpool + m)] ==
              g::kPoolIndices[static_cast<size_t>((b * P + p) * d.index_kpool + m)]);
      }
    }
  }

  // And against the host reference, which is the arm W9c-3 replaces. Its
  // `double` accumulation is a host-reference widening the device deliberately
  // does not copy, so this bound is fp32-vs-fp64 reduction and nothing else.
  const PooledStates host = GetPooledStates(d, w, packed, g::kBatch, g::kSeqLen);
  REQUIRE(host.num_pools == P);
  const double host_delta = MaxAbsDiff(keys_pref, host.pool_keys);
  MESSAGE("pool_keys max|device - host reference| = " << host_delta);
  CHECK(host_delta < 2e-5);
}

TEST_CASE("glm5_next k-pool device: the selection is SET-equal and positionwise equal") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend: the k-pool select device gate is SKIPPED");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(backend);

  const IndexerDims d = FixtureDims();
  const IndexerWeights w = FixtureWeights();
  const std::vector<float> hidden(g::kHiddenStates,
                                  g::kHiddenStates + g::kBatch * g::kSeqLen * g::kHidden);
  const std::vector<float> q_resid(g::kQResid,
                                   g::kQResid + g::kBatch * g::kSeqLen * g::kQLora);
  const std::vector<uint8_t> mask =
      MaskBytes(g::kMask, static_cast<size_t>(g::kBatch * g::kSeqLen));
  const std::vector<float> packed =
      PackIndexerStates(d, w, hidden, mask, g::kBatch, g::kSeqLen);
  const int64_t rows = g::kBatch * g::kSeqLen;
  const int64_t np = (g::kSeqLen + d.index_kpool - 1) / d.index_kpool;
  const int64_t P = g::kNumPools;
  const int64_t out_w = d.OutputWidth();
  REQUIRE(out_w == g::kOutputWidth);

  // The two operands the op takes rather than computes: `wq_b(q_resid)` and
  // `weights_proj(hidden)` (`modular_glm5_next.py:795,827`). W9c-3 gets both
  // from `vt::MatmulBT`; here they are built once so both arms see the same
  // bytes.
  const std::vector<float> q_states =
      LinearRows(w.wq_b, q_resid, rows, d.n_heads * d.head_dim, d.q_lora_rank);
  const std::vector<float> head_w = LinearRows(w.weights_proj, hidden, rows, d.n_heads,
                                               d.hidden_size);
  RequireFinite(q_states, "wq_b(q_resid)");
  RequireFinite(head_w, "weights_proj(hidden)");

  std::vector<int32_t> valid_keys(static_cast<size_t>(g::kBatch * g::kSeqLen), 0);
  for (int64_t i = 0; i < g::kBatch * g::kSeqLen; ++i)
    valid_keys[static_cast<size_t>(i)] =
        packed[static_cast<size_t>(i * (2 * d.head_dim + 1) + 2 * d.head_dim)] != 0.0f ? 1 : 0;
  std::vector<int32_t> q_mask(g::kMask, g::kMask + g::kBatch * g::kSeqLen);
  const std::vector<float> ape(g::kKpoolApe, g::kKpoolApe + d.index_kpool * d.head_dim);

  DeviceTensor dev_packed(backend, qg.q, DType::kF32,
                          {g::kBatch, g::kSeqLen, 2 * d.head_dim + 1}, packed.data());
  DeviceTensor dev_ape(backend, qg.q, DType::kF32, {d.index_kpool, d.head_dim}, ape.data());
  DeviceTensor dev_keys(backend, qg.q, DType::kF32, {g::kBatch, np, d.head_dim});
  DeviceTensor dev_idx(backend, qg.q, DType::kI32, {g::kBatch, np, d.index_kpool});
  DeviceTensor dev_valid(backend, qg.q, DType::kI32, {g::kBatch, np});
  DeviceTensor dev_np(backend, qg.q, DType::kI32, {1});
  vt::Glm5NextKpoolCompress(qg.q, dev_keys.tensor(), dev_idx.tensor(), dev_valid.tensor(),
                            dev_np.tensor(), dev_packed.tensor(), dev_ape.tensor());

  DeviceTensor dev_q(backend, qg.q, DType::kF32,
                     {g::kBatch, g::kSeqLen, d.n_heads, d.head_dim}, q_states.data());
  DeviceTensor dev_hw(backend, qg.q, DType::kF32, {g::kBatch, g::kSeqLen, d.n_heads},
                      head_w.data());
  DeviceTensor dev_vk(backend, qg.q, DType::kI32, {g::kBatch, g::kSeqLen}, valid_keys.data());
  DeviceTensor dev_qm(backend, qg.q, DType::kI32, {g::kBatch, g::kSeqLen}, q_mask.data());
  DeviceTensor dev_topk(backend, qg.q, DType::kI32, {g::kBatch, g::kSeqLen, out_w});
  DeviceTensor dev_scores(backend, qg.q, DType::kF32, {g::kBatch, g::kSeqLen, np});

  vt::Glm5NextKpoolSelectArgs args;
  args.index_topk = d.index_topk;
  args.current_length = g::kSeqLen;
  args.softmax_scale = d.softmax_scale();
  args.always_select_tail = true;
  vt::Glm5NextKpoolSelect(qg.q, dev_topk.tensor(), dev_scores.tensor(), dev_q.tensor(),
                          dev_hw.tensor(), dev_keys.tensor(), dev_idx.tensor(),
                          dev_valid.tensor(), dev_np.tensor(), dev_vk.tensor(),
                          dev_qm.tensor(), args);
  backend.Synchronize(qg.q);

  std::vector<int32_t> got_topk(static_cast<size_t>(rows * out_w));
  std::vector<float> got_scores(static_cast<size_t>(rows * np));
  dev_topk.Download(qg.q, got_topk.data());
  dev_scores.Download(qg.q, got_scores.data());

  // The scores over the LIVE pools only. `[P, np)` is allocation slack.
  std::vector<float> scores_pref, golden_scores(g::kIndexScores,
                                                g::kIndexScores + rows * P);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t p = 0; p < P; ++p)
      scores_pref.push_back(got_scores[static_cast<size_t>(r * np + p)]);
  RequireFinite(scores_pref, "device index_scores");
  const double score_delta = MaxAbsDiff(scores_pref, golden_scores);

  // POSITIONWISE against the transformers run, which is strictly stronger than
  // the set comparison and pins the emission ORDER as well as the membership.
  int64_t mismatched = 0;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t i = 0; i < out_w; ++i)
      if (got_topk[static_cast<size_t>(r * out_w + i)] !=
          g::kTopkIndices[static_cast<size_t>(r * out_w + i)])
        ++mismatched;
  INFO("positionwise mismatches against transformers: " << mismatched << " of "
                                                        << rows * out_w);
  CHECK(mismatched == 0);

  // SET equality, which is the model's own semantics (duplicates and -1 both
  // absorbed downstream), asserted separately so a positionwise regression and
  // a selection regression are distinguishable in the output.
  int64_t set_mismatch = 0;
  for (int64_t r = 0; r < rows; ++r) {
    const std::set<int32_t> got = SelectedSet(got_topk.data() + r * out_w, out_w);
    const std::set<int32_t> want = SelectedSet(g::kTopkIndices + r * out_w, out_w);
    if (got != want) ++set_mismatch;
  }
  CHECK(set_mismatch == 0);

  // And against the host reference over the same packed history.
  const IndexerSelection host = SelectIndexerTopkFromPacked(
      d, w, hidden, q_resid, mask, packed, g::kBatch, g::kSeqLen, g::kSeqLen);
  int64_t host_mismatch = 0;
  for (size_t i = 0; i < host.topk_indices.size(); ++i)
    if (host.topk_indices[i] != got_topk[i]) ++host_mismatch;
  INFO("positionwise mismatches against the host reference: " << host_mismatch);
  CHECK(host_mismatch == 0);

  // THE MARGIN. Computed from the host reference's own candidate mask, printed,
  // and required to be strictly positive on the rows that actually prune — a
  // fixture whose boundary gap is zero decides the top-k by the tie rule alone
  // and cannot detect a scoring defect at all.
  const int64_t select_k = d.SelectK(P);
  double worst = std::numeric_limits<double>::infinity();
  int64_t pruning_rows = 0;
  for (int64_t b = 0; b < g::kBatch; ++b) {
    for (int64_t s = 0; s < g::kSeqLen; ++s) {
      const int64_t r = b * g::kSeqLen + s;
      std::vector<float> row(host.index_scores.begin() + r * P,
                             host.index_scores.begin() + (r + 1) * P);
      std::vector<uint8_t> cand(static_cast<size_t>(P), 0);
      for (int64_t p = 0; p < P; ++p) {
        const int32_t last =
            host.pooled.pool_indices[static_cast<size_t>((b * P + p) * d.index_kpool +
                                                         d.index_kpool - 1)];
        const int64_t safe = last < 0 ? 0 : (last >= g::kSeqLen ? g::kSeqLen - 1 : last);
        const bool vis = safe <= s && valid_keys[static_cast<size_t>(b * g::kSeqLen + safe)] != 0;
        cand[static_cast<size_t>(p)] =
            (vis && host.pooled.pool_valid[static_cast<size_t>(b * P + p)] != 0) ? 1 : 0;
      }
      const double m = RowMargin(row, cand, P, select_k);
      if (m < 0.0) continue;  // fewer candidates than the budget: nothing pruned
      ++pruning_rows;
      worst = std::min(worst, m);
    }
  }
  MESSAGE("pruning rows = " << pruning_rows << ", smallest decision margin = " << worst);
  MESSAGE("index_scores max|device - transformers| = " << score_delta);
  REQUIRE(pruning_rows > 0);
  CHECK(std::isfinite(worst));
  CHECK(worst > 0.0);

  // THE SCORE BOUND IS TIED TO THE MARGIN, not chosen. An absolute tolerance on
  // a score would be a number with no meaning: the golden magnitudes here run to
  // 45, so one f32 ULP is already 3.8e-6 and a tolerance below that fails on
  // arithmetic while a tolerance far above it stops bounding anything. The
  // question a selection gate can actually answer is whether the numerical
  // difference is small against the gap the top-k decides on, so that is what is
  // asserted — with a factor of four of room, and both numbers printed so a
  // future fixture that narrows the margin fails here instead of silently
  // becoming a coin flip.
  CHECK(score_delta * 4.0 < worst);
}

TEST_CASE("glm5_next k-pool device: P == 0 serves the raw visible tail alone") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend: the k-pool short-context device gate is SKIPPED");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(backend);

  const IndexerDims d = FixtureDims();
  const IndexerWeights w = FixtureWeights();
  const int64_t out_w = d.OutputWidth();
  const std::vector<float> ape(g::kKpoolApe, g::kKpoolApe + d.index_kpool * d.head_dim);

  int64_t h_off = 0, q_off = 0, m_off = 0, t_off = 0;
  for (int64_t c = 0; c < g::kShortCases; ++c) {
    const int64_t S = g::kShortSeqLen[c];
    const std::vector<float> hidden(g::kShortHidden + h_off,
                                    g::kShortHidden + h_off + S * g::kHidden);
    const std::vector<float> q_resid(g::kShortQResid + q_off,
                                     g::kShortQResid + q_off + S * g::kQLora);
    const std::vector<uint8_t> mask = MaskBytes(g::kShortMask + m_off, static_cast<size_t>(S));
    const std::vector<float> packed = PackIndexerStates(d, w, hidden, mask, 1, S);
    const int64_t np = (S + d.index_kpool - 1) / d.index_kpool;

    DeviceTensor dev_packed(backend, qg.q, DType::kF32, {1, S, 2 * d.head_dim + 1},
                            packed.data());
    DeviceTensor dev_ape(backend, qg.q, DType::kF32, {d.index_kpool, d.head_dim}, ape.data());
    DeviceTensor dev_keys(backend, qg.q, DType::kF32, {1, np, d.head_dim});
    DeviceTensor dev_idx(backend, qg.q, DType::kI32, {1, np, d.index_kpool});
    DeviceTensor dev_valid(backend, qg.q, DType::kI32, {1, np});
    DeviceTensor dev_np(backend, qg.q, DType::kI32, {1});
    vt::Glm5NextKpoolCompress(qg.q, dev_keys.tensor(), dev_idx.tensor(), dev_valid.tensor(),
                              dev_np.tensor(), dev_packed.tensor(), dev_ape.tensor());
    backend.Synchronize(qg.q);
    std::vector<int32_t> got_np(1, -1);
    dev_np.Download(qg.q, got_np.data());
    INFO("short case " << c << " (S = " << S << ")");
    CHECK(got_np[0] == static_cast<int32_t>(g::kShortNumPools[c]));

    const std::vector<float> q_states =
        LinearRows(w.wq_b, q_resid, S, d.n_heads * d.head_dim, d.q_lora_rank);
    const std::vector<float> head_w =
        LinearRows(w.weights_proj, hidden, S, d.n_heads, d.hidden_size);
    std::vector<int32_t> valid_keys(static_cast<size_t>(S), 0);
    for (int64_t i = 0; i < S; ++i)
      valid_keys[static_cast<size_t>(i)] =
          packed[static_cast<size_t>(i * (2 * d.head_dim + 1) + 2 * d.head_dim)] != 0.0f ? 1 : 0;
    std::vector<int32_t> q_mask(g::kShortMask + m_off, g::kShortMask + m_off + S);

    DeviceTensor dev_q(backend, qg.q, DType::kF32, {1, S, d.n_heads, d.head_dim},
                       q_states.data());
    DeviceTensor dev_hw(backend, qg.q, DType::kF32, {1, S, d.n_heads}, head_w.data());
    DeviceTensor dev_vk(backend, qg.q, DType::kI32, {1, S}, valid_keys.data());
    DeviceTensor dev_qm(backend, qg.q, DType::kI32, {1, S}, q_mask.data());
    DeviceTensor dev_topk(backend, qg.q, DType::kI32, {1, S, out_w});
    DeviceTensor dev_scores(backend, qg.q, DType::kF32, {1, S, np});

    vt::Glm5NextKpoolSelectArgs args;
    args.index_topk = d.index_topk;
    args.current_length = S;
    args.softmax_scale = d.softmax_scale();
    args.always_select_tail = true;
    vt::Glm5NextKpoolSelect(qg.q, dev_topk.tensor(), dev_scores.tensor(), dev_q.tensor(),
                            dev_hw.tensor(), dev_keys.tensor(), dev_idx.tensor(),
                            dev_valid.tensor(), dev_np.tensor(), dev_vk.tensor(),
                            dev_qm.tensor(), args);
    backend.Synchronize(qg.q);

    std::vector<int32_t> got(static_cast<size_t>(S * out_w));
    dev_topk.Download(qg.q, got.data());
    for (int64_t i = 0; i < S * out_w; ++i) {
      CHECK(got[static_cast<size_t>(i)] ==
            g::kShortTopk[static_cast<size_t>(t_off + i)]);
    }

    h_off += S * g::kHidden;
    q_off += S * g::kQLora;
    m_off += S;
    t_off += S * out_w;
  }
}
