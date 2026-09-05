// vllm.cpp original (vt runtime, inventory deviation §9.1). Wave TIEBREAK of
// MODEL-MM-QWEN4-EXP (spec .agents/specs/qwen4-exp-flash-next.md
// "## Wave TIEBREAK", issue #2586).
//
// THE QUESTION. Wave MOEDIV (#2552) measured that the `qwen4_exp` router's
// top-k boundary is an EXACT bf16 tie at 32.9% of prefill token-slots, and that
// the CPU and CUDA arms then select different experts at about the same rate.
// It concluded that this is tie-break order rather than a defect, and it
// verified that conclusion BY SOURCE READ of both kernels.
//
// A source read establishes the INTENDED tie-break. It cannot establish that a
// parallel argmax REALISES it. The CUDA arm's greedy top-k is a per-thread
// strided scan, a __shfl_down_sync butterfly and a cross-warp pass; if any level
// of that compares values without breaking the tie on the index, or breaks it
// with the wrong polarity, the arms disagree on BIT-IDENTICAL logits and that IS
// a defect.
//
// WHY THE STANDING GATE CANNOT ANSWER IT. tests/vt/test_ops_moe_grouped.cpp's
// "CUDA moe_router_topk parallel == serial byte-for-byte (adversarial)" sweep is
// real and it stays. Two limits keep it away from this question:
//
//   1. It sweeps E in {32,64,128,256} only. MoeRouterWarpValuesPerThread
//      (src/vt/cuda/moe_router_warp.h) returns 0 for E = 512, so the geometry
//      `qwen4_exp` routes -- E = 512, k = 10 -- falls through to
//      MoeRouterTopKKernel<Tin,false>, the BLOCK kernel at 2 experts per thread.
//      No case in this tree has ever executed that decomposition.
//   2. Its tie patterns are `(e / 4) % 5` and `e < 12 ? 3.0f : 0.0f`. Both keep
//      the tied set in a contiguous low prefix, so at kBlock = 256 it lives
//      inside warp 0 and the CROSS-WARP index comparison never breaks a tie.
//
// WHAT IS ASSERTED, AND WHY IT IS NOT ARM-VS-ARM. Every row below has a
// CLOSED-FORM correct answer: `h` experts at strictly higher distinct values,
// then a tied set S whose members carry the SAME bf16 bit pattern, then
// everything else strictly lower. The top-k is the h high experts in descending
// value order followed by the k-h LOWEST INDICES OF S. That is upstream's
// specified behaviour, not a local convention:
//
//   csrc/libtorch_stable/moe/topk_softmax_kernels.cu:536-537 @ vLLM 5559679229
//     "We want lower indices to \"win\" in every thread so we break ties this way"
//   :515-517  the per-thread scan, "only updated if > (not >=)"
//   :707-708  case 512: LAUNCH_TOPK(512, ...) -- E=512 is a registered width
//   :186,222,225  the fallback moeTopK's cub::ArgMax, same lowest-key contract
//   :465      "With 0s, the argmax uses index tie-breaking to pick [0,1,2,...,k-1]"
//
// So a failure here is a divergence from the oracle, and both arms being wrong
// the same way cannot read green.
//
// A DISCRETE SELECTION HAS BIMODAL ERROR. Indices are compared with `==` and the
// selected SET with sorted equality. No tolerance appears anywhere in this file.
// The boundary MARGIN is captured instead: `tied_seen` counts the rows whose
// boundary carries the same bf16 bit pattern on both sides, and it is CHECKed
// against the row count, so a case whose builder failed to produce a tie cannot
// report green while measuring nothing.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe_router_warp_env.h"
#include "vt/backend.h"
#include "vt/cuda/moe_decode_ref.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::MoeRouterTopKArgs;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

// ---------------------------------------------------------------------------
// The row builder. Every value it emits is EXACTLY representable in bf16, so the
// f32 and bf16 arms carry the identical mathematical row and the tie is a tie in
// both. `tie_v` sits strictly between the lowest high value and the low floor,
// separated by 0.5 in logit space (a factor of e^0.5 = 1.65 in probability
// space), so no rounding of exp()/sum can manufacture a tie this builder did not
// intend, and no intended tie can be broken by one.
struct TieRow {
  std::vector<float> logits;   // [E]
  std::vector<int32_t> expect; // [k] the closed-form correct selection, in order
  std::vector<int32_t> tied;   // S, ascending
  int h = 0;                   // how many strict winners precede the tie
};

// `pattern` selects WHERE the tied experts live, which is the whole point: at
// kBlock = 256 expert `x` is owned by thread `x % 256` and therefore by warp
// `(x % 256) / 32`, so the pattern decides WHICH LEVEL of the reduction is asked
// to break a tie.
//   0  scattered by (197i + 11) mod E      every thread and every warp
//   1  the upper half only                 the second strided slot
//   2  the last warp of block 0            the winner must come out of warp 7
//   3  lanes 28..31 of EVERY warp          cross-warp AND intra-warp at once
//   4  a contiguous block just under E     the standing sweep's shape moved up
//   5  BOTH slots of the same threads      the PER-THREAD strided scan, and the
//                                          only pattern that can see it: at
//                                          E <= 256 a thread holds ONE expert,
//                                          so no tie ever reaches that scan, and
//                                          at E = 512 it holds {t, t+256} and
//                                          the scan alone decides between them.
//
// Pattern 5 is why this file exists at a geometry the standing sweep does not
// cover. Nothing at E in {32,64,128,256} can exercise the intra-thread compare,
// because there is nothing to compare.
TieRow BuildTieRow(int64_t e, int k, int h, int m, int pattern, uint32_t seed) {
  REQUIRE(h >= 0);
  REQUIRE(h < k);
  TieRow r;
  r.h = h;
  r.logits.assign(static_cast<size_t>(e), 0.0f);  // the low floor

  // Each pattern emits its own index list. Nothing is truncated to a common
  // size: truncating to the m SMALLEST indices would quietly turn every pattern
  // back into a low contiguous prefix, which is the exact shape this file exists
  // to get away from.
  std::vector<int32_t> s;
  const int64_t nblocks = e > 256 ? e / 256 : 1;
  switch (pattern) {
    case 0:
      for (int i = 0; i < m; ++i)
        s.push_back(static_cast<int32_t>((static_cast<int64_t>(i) * 197 + 11) % e));
      break;
    case 1:
      for (int i = 0; i < m; ++i) s.push_back(static_cast<int32_t>(e / 2 + 3 * i));
      break;
    case 2:
      for (int64_t o = 224; o < 256; ++o) s.push_back(static_cast<int32_t>(o));
      break;
    case 3:
      for (int64_t i = 0; i < e; ++i)
        if ((i % 32) >= 28) s.push_back(static_cast<int32_t>(i));
      break;
    case 4:
      for (int64_t i = e - m; i < e; ++i) s.push_back(static_cast<int32_t>(i));
      break;
    default:
      // The SAME threads own every index here: 96+j and 96+j+256 (and +512,
      // +768 at E = 1024) are all thread 96+j.
      for (int64_t b = 0; b < nblocks; ++b)
        for (int i = 0; i < m; ++i) s.push_back(static_cast<int32_t>(96 + i + 256 * b));
      break;
  }
  std::sort(s.begin(), s.end());
  s.erase(std::unique(s.begin(), s.end()), s.end());
  REQUIRE(s.front() >= 0);
  REQUIRE(s.back() < e);
  REQUIRE(static_cast<int>(s.size()) > k - h);  // some tied candidate is REJECTED
  (void)seed;
  r.tied = s;

  const float tie_v = 1.0f;
  for (int32_t idx : s) r.logits[static_cast<size_t>(idx)] = tie_v;

  // The strict winners go at indices NOT in S, placed high in the row on
  // purpose: an implementation that returned "the first k experts it happened to
  // scan" would still have to reach them.
  std::vector<int32_t> highs;
  for (int64_t i = e - 1; i >= 0 && static_cast<int>(highs.size()) < h; --i) {
    if (std::binary_search(s.begin(), s.end(), static_cast<int32_t>(i))) continue;
    highs.push_back(static_cast<int32_t>(i));
  }
  REQUIRE(static_cast<int>(highs.size()) == h);
  for (int j = 0; j < h; ++j) {
    // 8.0, 7.5, 7.0, ... : strictly descending, distinct, bf16-exact, and all
    // at least 0.5 above tie_v.
    r.logits[static_cast<size_t>(highs[static_cast<size_t>(j)])] =
        8.0f - 0.5f * static_cast<float>(j);
  }

  r.expect = highs;  // already in descending-value order
  for (int j = 0; j < k - h; ++j) r.expect.push_back(s[static_cast<size_t>(j)]);
  REQUIRE(static_cast<int>(r.expect.size()) == k);
  return r;
}

// The counted property. Returns true when the row's top-k boundary carries ZERO
// bf16 margin -- the last selected and the first rejected logit are the SAME bf16
// bit pattern. It reads the LOGITS, so it is independent of every kernel under
// test, and a builder that silently stopped producing ties drives it to 0.
bool BoundaryIsExactTie(const TieRow& r, int k) {
  const int last_sel = r.expect[static_cast<size_t>(k - 1)];
  const int first_rej = r.tied[static_cast<size_t>(k - r.h)];
  const uint16_t a = vt::F32ToBF16(r.logits[static_cast<size_t>(last_sel)]);
  const uint16_t b = vt::F32ToBF16(r.logits[static_cast<size_t>(first_rej)]);
  return a == b;
}

std::vector<uint8_t> Pack(const std::vector<float>& f, DType dt) {
  std::vector<uint8_t> out(f.size() * vt::SizeOf(dt));
  if (dt == DType::kF32) {
    std::memcpy(out.data(), f.data(), out.size());
  } else {
    auto* p = reinterpret_cast<uint16_t*>(out.data());
    for (size_t i = 0; i < f.size(); ++i) p[i] = vt::F32ToBF16(f[i]);
  }
  return out;
}

// The geometries. E = 512 k = 10 is `qwen4_exp`'s own
// (num_experts = 512, num_experts_per_tok = 10); E = 256 is the width the warp
// kernel dispatches, so both CUDA kernels are covered by one sweep; E = 1024
// puts FOUR experts on every thread, which is the next decomposition up.
struct Geom {
  int64_t e;
  int k;
};
const Geom kGeoms[] = {{256, 8}, {512, 10}, {1024, 10}};

}  // namespace

// ---------------------------------------------------------------------------
// The CPU arm, everywhere. This runs in CI on every platform and is the red-first
// half of the pair: it pins the closed-form expectation on the reference the
// CUDA arms are required to match.
TEST_CASE("moe router CPU: an exact tie at the top-k boundary selects the lowest tied indices") {
  int tied_seen = 0;
  int rows = 0;
  for (const Geom& g : kGeoms) {
    for (int pattern = 0; pattern < 6; ++pattern) {
      for (int h : {0, 1, g.k - 1}) {
        const TieRow r = BuildTieRow(g.e, g.k, h, /*m=*/g.k + 20, pattern, 991u);
        ++rows;
        if (BoundaryIsExactTie(r, g.k)) ++tied_seen;
        for (DType dt : {DType::kF32, DType::kBF16}) {
          const auto lb = Pack(r.logits, dt);
          std::vector<float> w(static_cast<size_t>(g.k), 0.0f);
          std::vector<int32_t> ids(static_cast<size_t>(g.k), -1);
          Tensor tl = MakeT(const_cast<uint8_t*>(lb.data()), dt, Cpu(), {1, g.e});
          Tensor tw = MakeT(w.data(), DType::kF32, Cpu(), {1, g.k});
          Tensor ti = MakeT(ids.data(), DType::kI32, Cpu(), {1, g.k});
          Queue q{Cpu(), nullptr};
          vt::MoeRouterTopK(q, tw, ti, tl, MoeRouterTopKArgs{g.k, true});
          CAPTURE(g.e);
          CAPTURE(g.k);
          CAPTURE(pattern);
          CAPTURE(h);
          CAPTURE(static_cast<int>(dt));
          CHECK(ids == r.expect);
        }
      }
    }
  }
  // COUNTED PROPERTY: every row built above must actually carry a zero-margin
  // boundary, or this case measured something other than a tie.
  CAPTURE(rows);
  CHECK(rows == 54);
  CHECK(tied_seen == rows);
}

// ---------------------------------------------------------------------------
// CUDA. Guarded on VLLM_CPP_CUDA because the serial oracle
// vt::cuda::MoeRouterTopKSerialCuda is a CUDA-only symbol (a runtime HasCuda()
// skip is not enough -- it is undefined at link time on a CPU build), and on a
// runtime HasCuda() because a CUDA build still runs on hosts with no device.
//
// THAT RUNTIME SKIP IS PERMISSIVE, AND IT LETS A DEVICE-LESS CUDA BUILD READ
// GREEN. The three cases below print a MESSAGE and return, so such a build
// reports `4 cases | 488 assertions | 0 failed | rc 0` -- which this wave's
// orin:gpu0 lease did, with a BROKEN tie-break compiled in. Three states have
// to be told apart, and the CASE COUNT is half of what tells them apart:
//
//   CPU-only build (these cases #ifdef'd out) : 1 case  |  488 assertions
//   CUDA build, no device (they skip)         : 4 cases |  488 assertions
//   CUDA build with a device                  : 4 cases | 4652 assertions
//
// Read BOTH numbers; neither alone separates all three. The tree already has
// the idiom that refuses instead of skipping --
// tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:149 reads
// VT_REQUIRE_27N_FP8_GATE and turns absence into a hard FAILURE -- and 13 other
// test files share the permissive shape used here, so adopting it is a
// tree-wide change rather than this file's repair. #2603 owns it.
#ifdef VLLM_CPP_CUDA

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
    t_ = MakeT(p_, dt, Gpu(), shape);
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

// One launch, indices downloaded. `serial` picks the byte-exact single-threaded
// GPU oracle instead of the dispatched parallel kernel; everything else about
// the two calls -- the softmax, the memory, the device, the stream -- is
// identical, so a disagreement isolates the ARGMAX STRUCTURE and nothing else.
std::vector<int32_t> RunGpuIndices(Backend& gpu, const std::vector<uint8_t>& lb, DType dt,
                                   int64_t t, int64_t e, int k, bool serial) {
  QueueGuard gq(gpu);
  DeviceTensor dl(gpu, gq.q, dt, {t, e}, lb.data());
  DeviceTensor dw(gpu, gq.q, DType::kF32, {t, k});
  DeviceTensor di(gpu, gq.q, DType::kI32, {t, k});
  const MoeRouterTopKArgs args{k, true};
  if (serial) {
    vt::cuda::MoeRouterTopKSerialCuda(gq.q, dw.tensor(), di.tensor(), dl.tensor(), args);
  } else {
    vt::MoeRouterTopK(gq.q, dw.tensor(), di.tensor(), dl.tensor(), args);
  }
  gpu.Synchronize(gq.q);
  std::vector<int32_t> ids(static_cast<size_t>(t * k), -2);
  di.Download(gq.q, ids.data());
  return ids;
}

}  // namespace

// (1) The decisive experiment: bit-identical logits with an exact tie at the
//     boundary go to BOTH arms and the selections are compared as sets --
//     against each other AND against the closed-form answer.
TEST_CASE("CUDA moe router: an exact boundary tie selects the lowest tied indices on device") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  int tied_seen = 0;
  int rows = 0;
  auto sweep = [&](const char* arm) {
    for (const Geom& g : kGeoms) {
      for (int pattern = 0; pattern < 6; ++pattern) {
        for (int h : {0, 1, g.k - 1}) {
          const TieRow r = BuildTieRow(g.e, g.k, h, /*m=*/g.k + 20, pattern, 991u);
          ++rows;
          if (BoundaryIsExactTie(r, g.k)) ++tied_seen;
          for (DType dt : {DType::kF32, DType::kBF16}) {
            const auto lb = Pack(r.logits, dt);
            const auto par = RunGpuIndices(gpu, lb, dt, 1, g.e, g.k, /*serial=*/false);
            const auto ser = RunGpuIndices(gpu, lb, dt, 1, g.e, g.k, /*serial=*/true);
            CAPTURE(arm);
            CAPTURE(g.e);
            CAPTURE(g.k);
            CAPTURE(pattern);
            CAPTURE(h);
            CAPTURE(static_cast<int>(dt));
            CAPTURE(static_cast<int>(r.tied.size()));
            CHECK(par == r.expect);  // the closed-form answer, upstream's rule
            CHECK(ser == r.expect);  // the byte-exact GPU oracle agrees
            CHECK(par == ser);       // ... and so do the two structures
          }
        }
      }
    }
  };
  // BOTH ARMS ARE THE SAME KERNEL ABOVE E = 256, AND THAT HALVES THE COVERAGE
  // HERE. MoeRouterWarpValuesPerThread (moe_router_warp.h:96-98) returns 0 for
  // every E outside {32,64,128,256}, and LaunchRouterWarp (cuda_moe.cu:564)
  // returns false on vpt == 0 before it touches a tensor, so at E = 512 and
  // E = 1024 -- including the geometry `qwen4_exp` actually routes -- the
  // warp-on arm falls through to the SAME MoeRouterTopKKernel<Tin,false> the
  // warp-off arm runs. Only E = 256 compares two structures.
  //
  // The counts below are honest: every assertion really executes. But 36 of the
  // 72 rows at E > 256 are byte-identical repeats, so each real disagreement
  // there is counted TWICE -- the mutation table's `12` per cell is
  // 2 arms x 3 h x 2 dtypes, of which 6 are the second count of the same event.
  // Read "both VT_MOE_ROUTER_WARP arms" as two structures only at E = 256.
  // #2604 owns narrowing this; it is not fixed here because re-measuring it
  // costs the GPU time every record in this wave cites.
  {
    vt_test::ScopedMoeRouterWarp pin("1");
    REQUIRE(vt_test::ScopedMoeRouterWarp::EffectiveFlag());
    sweep("warp-on");
  }
  {
    vt_test::ScopedMoeRouterWarp pin("0");
    REQUIRE_FALSE(vt_test::ScopedMoeRouterWarp::EffectiveFlag());
    sweep("warp-off");
  }
  CAPTURE(rows);
  CHECK(rows == 108);
  CHECK(tied_seen == rows);
}

// (2) Determinism WITHIN the CUDA arm. Same bytes in; the same bytes must come
//     out every time. A selection that varies run to run on identical input is a
//     stronger finding than a disagreement between two arms.
TEST_CASE("CUDA moe router: a boundary tie is decided the same way on every repeat") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  constexpr int kRepeats = 32;
  int compared = 0;
  for (const Geom& g : kGeoms) {
    for (int pattern : {0, 3, 5}) {
      const TieRow r = BuildTieRow(g.e, g.k, /*h=*/1, /*m=*/g.k + 20, pattern, 991u);
      REQUIRE(BoundaryIsExactTie(r, g.k));
      const auto lb = Pack(r.logits, DType::kBF16);
      const auto first = RunGpuIndices(gpu, lb, DType::kBF16, 1, g.e, g.k, /*serial=*/false);
      for (int i = 1; i < kRepeats; ++i) {
        const auto again = RunGpuIndices(gpu, lb, DType::kBF16, 1, g.e, g.k, /*serial=*/false);
        CAPTURE(g.e);
        CAPTURE(pattern);
        CAPTURE(i);
        CHECK(again == first);
        ++compared;
      }
      CHECK(first == r.expect);
    }
  }
  CAPTURE(compared);
  CHECK(compared == 9 * (kRepeats - 1));
}

// (3) The grid must not decide the answer. One row's selection has to be the
//     same whether it is launched alone or as row `t` of a large batch: the
//     kernel is one block per token, so batching changes the grid and the
//     scheduling without changing any block's work. If a tie resolved
//     differently at T = 1 and at T = 257 the tie-break would be a function of
//     the launch shape.
TEST_CASE("CUDA moe router: a boundary tie resolves the same alone and inside a large batch") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  constexpr int64_t kBatch = 257;  // deliberately not a multiple of anything
  int compared = 0;
  for (const Geom& g : kGeoms) {
    for (int pattern : {0, 3, 5}) {
      const TieRow r = BuildTieRow(g.e, g.k, /*h=*/1, /*m=*/g.k + 20, pattern, 991u);
      REQUIRE(BoundaryIsExactTie(r, g.k));
      const auto one = Pack(r.logits, DType::kBF16);
      std::vector<float> wide;
      wide.reserve(static_cast<size_t>(kBatch * g.e));
      for (int64_t t = 0; t < kBatch; ++t)
        wide.insert(wide.end(), r.logits.begin(), r.logits.end());
      const auto many = Pack(wide, DType::kBF16);
      const auto alone = RunGpuIndices(gpu, one, DType::kBF16, 1, g.e, g.k, /*serial=*/false);
      const auto batch = RunGpuIndices(gpu, many, DType::kBF16, kBatch, g.e, g.k, /*serial=*/false);
      CHECK(alone == r.expect);
      for (int64_t t = 0; t < kBatch; ++t) {
        std::vector<int32_t> row(batch.begin() + static_cast<ptrdiff_t>(t * g.k),
                                 batch.begin() + static_cast<ptrdiff_t>((t + 1) * g.k));
        CAPTURE(g.e);
        CAPTURE(pattern);
        CAPTURE(t);
        CHECK(row == alone);
        ++compared;
      }
    }
  }
  CAPTURE(compared);
  CHECK(compared == 9 * static_cast<int>(kBatch));
}

#endif  // VLLM_CPP_CUDA
