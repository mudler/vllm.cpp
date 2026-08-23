// vllm.cpp — the block-scaled FP8 GEMM, CUDA arm, against the CPU reference arm
// of the SAME op.
//
// VT-MATMUL-FP8-BLOCK-CUDA (.agents/specs/vt-matmul-fp8-block-cuda.md), issue
// #1189 milestone M5. Pinned oracle: vLLM
// 5559679229bc961848b121ccdeaa8fa5d79bec98, asserted as the local checkout's
// HEAD before every anchor below was read.
//
// THIS FILE HAS RUN AGAINST A DEVICE, ON SEVEN SHAPES. The suite ran UNPATCHED
// on `dgx:gpu0` (NVIDIA GB10, driver 580.173.02, compute capability 12.1) in an
// `rc` lease on 2026-08-20, at tree 7481a2eecbb26b3d5c977e8707b0384994caf136 —
// an ancestor of `main` — and reported 5 cases, 136 assertions, 0 failed, with
// REFERENCE_TIER_LINES=0 and TEST_RC=0. A device-free run of this file prints
// 41, so 95 of those assertions ran on the board, and that delta is the
// discriminator between a run and a skip. #1189 M5's spec records it under
// `## Owed`, and the user-facing documentation for Qwen3.8-27B-FP8 states the
// same 5/136/0 result. No path to it is named here on purpose: a docs
// reorganisation does not open this file, and no checker ties the two. On a host with no
// CUDA device this file still SKIPS, and a skip is still not a pass — the four
// NO CUDA DEVICE messages below say exactly that, per case.
//
// WHAT THAT RUN DOES NOT ESTABLISH, written here so that no later reader widens
// it. THIS FILE is not a token gate, and it never was: it compares this arm
// against the CPU reference on seven shapes. The token gate is a different
// measurement, it RAN on 2026-08-23, and it PASSED — `Qwen/Qwen3.8-27B-FP8`
// decoded beside the pinned oracle on GB10, every first divergence an exact tie
// or in band, recorded in `.agents/specs/gate-qwen38-27b-fp8-block.md`. Nothing
// in this file measured that, and nothing in this file may be read as it.
// There is NO SPEED CLAIM of any kind: the lease took no clock control,
// recorded no contention and had no denominator, which are the three things
// `.agents/benchmarking.md` requires before a ratio means anything. And the
// correctness evidence covers the SEVEN SHAPES THAT WERE ACTUALLY RUN — G7's
// six servable grid entries, which between them span all three tile configs and
// both the swapped and the unswapped path, plus G2's {32,512,7168} under
// upstream's own fixture — and says nothing about a shape outside those seven.
// The capability gap STANDS: DSV3's `kv_a_proj_with_mqa` is N=576 and this arm
// cannot serve it on sm120 until CUTLASS's sm120 collective supports partial
// scale blocks, which is an upstream limitation and not a defect in this tree.
//
// WHAT IT MEASURES ON A DEVICE. The CUDA kernel against
// `vt::MatmulFp8BlockScaled`'s CPU arm — the reference #1189 M2 landed
// (`770e49486`) precisely to be this comparison's oracle. That arm is upstream's
// `native_w8a8_block_matmul` (tests/kernels/quant_utils.py) ported whole, so
// comparing against it IS comparing against upstream's reference, and the
// criterion below is upstream's own.
//
//   G2  upstream's `test_w8a8_block_fp8_cutlass_matmul`
//       (tests/kernels/quantization/test_block_fp8.py): M=32, N=576, K=7168,
//       block [128,128], bf16 out, scales U(0,1)*1e-2, compared at
//       `rel_diff < 0.001` computed by upstream's own formula. N=576 is
//       4*128 + 64 — a RAGGED final block, which is why upstream chose it
//       (DSV3's kv_a_proj_with_mqa) and which is exactly why THE sm120 ARM
//       CANNOT SERVE IT (#1437). The case therefore asserts the named refusal
//       for upstream's own shape and runs upstream's fixture and criterion on
//       the nearest servable N; the adaptation is documented at the case.
//   G6  the test's own precondition, and it runs on EVERY host including one
//       with no device: every shape the cases below use must get the refusal the
//       hand-written `expect` column names. It used to assert that every shape
//       is ACCEPTED, and it passed while three of them threw — a precondition
//       that agrees with the wrong side certifies a grid nothing can run.
//   G7  the M sweep across all three tile configs, with the dispatch counter
//       asserted to advance on the config the heuristic named, plus the named
//       refusal for each unservable shape.
//   G8  the f32 sink equals the bf16 result cast up.
//   G9  the refusals, on a device, by message.
//
// (G1, G3, G4 and G5 are the host-tier cases in
// `tests/vt/test_fp8_block_scaled_dispatch.cpp`, which DOES run here. The
// numbering is the spec's.)
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/cuda/fp8_block_scaled_dispatch.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using vt::cuda::Fp8BlockScaledConfig;
using vt::cuda::Fp8BlockScaledConfigFor;
using vt::cuda::Fp8BlockScaledRefusal;
using vt::cuda::Fp8BlockScaledRefusalFor;
using vt::cuda::Fp8BlockScaledStats;
using vt::cuda::Fp8BlockScaledStatsSnapshot;
using vt::cuda::kFp8BlockScaledScaleBlockN;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Device Cpu() { return Device{DeviceType::kCPU, 0}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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
    t_ = MakeTensor(p_, dt, Gpu(), shape);
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

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Random fp8-e4m3fn BYTES, not values. Both arms consume the same bytes, so
// nothing here needs an encoder and the comparison cannot be contaminated by one
// — upstream's CUTLASS test quantizes once and feeds both arms too. 0x7F/0xFF
// are the only NaN encodings of "fn" and are excluded.
std::vector<uint8_t> RandomFp8Bytes(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> u(0, 255);
  std::vector<uint8_t> v(n);
  for (auto& b : v) {
    int byte = u(rng);
    if ((byte & 0x7F) == 0x7F) byte &= ~0x7;
    b = static_cast<uint8_t>(byte);
  }
  return v;
}

// upstream: `As = torch.rand(M, k_tiles) * factor_for_scale` with
// `factor_for_scale = 1e-2` (test_block_fp8.py). Uniform on [0, 1e-2).
std::vector<float> RandomScales(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& s : v) s = u(rng) * 1e-2f;
  return v;
}

struct BlockCase {
  int64_t m = 0, n = 0, k = 0;
  int block_n = 128, block_k = 128;
  // The refusal this shape MUST get from `Fp8BlockScaledRefusalFor`, written out
  // by hand from the sm120 collective's `can_implement` and never computed from
  // the predicate. G6 asserts the predicate against this column; G7 then drives
  // each shape down the arm the column names.
  Fp8BlockScaledRefusal expect = Fp8BlockScaledRefusal::kNone;
};

// The CPU reference arm of the SAME op, run on a CPU queue in this process.
std::vector<float> ReferenceBf16(const BlockCase& c, const std::vector<uint8_t>& a_fp8,
                                 const std::vector<float>& a_scale,
                                 const std::vector<uint8_t>& b_fp8,
                                 const std::vector<float>& b_scale) {
  const int64_t k_tiles = CDiv(c.k, c.block_k);
  const int64_t n_tiles = CDiv(c.n, c.block_n);
  std::vector<uint16_t> out(static_cast<size_t>(c.m) * c.n, 0);

  Backend& b = vt::GetBackend(DeviceType::kCPU);
  QueueGuard g(b);
  Tensor ta = MakeTensor(const_cast<uint8_t*>(a_fp8.data()), DType::kI8, Cpu(), {c.m, c.k});
  Tensor tas = MakeTensor(const_cast<float*>(a_scale.data()), DType::kF32, Cpu(), {c.m, k_tiles});
  Tensor tb = MakeTensor(const_cast<uint8_t*>(b_fp8.data()), DType::kI8, Cpu(), {c.n, c.k});
  Tensor tbs =
      MakeTensor(const_cast<float*>(b_scale.data()), DType::kF32, Cpu(), {n_tiles, k_tiles});
  Tensor to = MakeTensor(out.data(), DType::kBF16, Cpu(), {c.m, c.n});
  vt::MatmulFp8BlockScaled(g.q, to, ta, tas, tb, tbs, c.block_n, c.block_k);
  b.Synchronize(g.q);

  std::vector<float> f(out.size());
  for (size_t i = 0; i < out.size(); ++i) f[i] = vt::BF16ToF32(out[i]);
  return f;
}

// The CUDA arm, same bytes, same scales, bf16 out.
std::vector<float> DeviceBf16(const BlockCase& c, const std::vector<uint8_t>& a_fp8,
                              const std::vector<float>& a_scale,
                              const std::vector<uint8_t>& b_fp8,
                              const std::vector<float>& b_scale) {
  const int64_t k_tiles = CDiv(c.k, c.block_k);
  const int64_t n_tiles = CDiv(c.n, c.block_n);
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor da(b, g.q, DType::kI8, {c.m, c.k}, a_fp8.data());
  DeviceTensor das(b, g.q, DType::kF32, {c.m, k_tiles}, a_scale.data());
  DeviceTensor db(b, g.q, DType::kI8, {c.n, c.k}, b_fp8.data());
  DeviceTensor dbs(b, g.q, DType::kF32, {n_tiles, k_tiles}, b_scale.data());
  DeviceTensor dout(b, g.q, DType::kBF16, {c.m, c.n});
  vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                           dbs.tensor(), c.block_n, c.block_k);
  std::vector<uint16_t> raw(static_cast<size_t>(c.m) * c.n);
  dout.Download(g.q, raw.data());
  std::vector<float> f(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) f[i] = vt::BF16ToF32(raw[i]);
  return f;
}

// upstream's criterion, its formula, verbatim
// (test_block_fp8.py::test_w8a8_block_fp8_cutlass_matmul):
//
//     rel_diff = mean(|out - ref|) / mean(|ref|);  assert rel_diff < 0.001
//
// It is the right tight criterion here for the reason the spec records: the two
// arms differ only in the K-reduction order inside a block and in associating
// `(part * a_s) * b_s` instead of `part * (a_s * b_s)` — at most one f32 ULP per
// K-block — and this is the bound upstream itself accepts between exactly these
// two implementations.
double RelDiff(const std::vector<float>& got, const std::vector<float>& want) {
  REQUIRE(got.size() == want.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    num += std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    den += std::fabs(static_cast<double>(want[i]));
  }
  REQUIRE(den > 0.0);
  return num / den;
}

// A MEAN CANNOT SEE ONE WRONG ELEMENT, and one wrong element is exactly what a
// ragged final block or an off-by-one scale index produces. This is a coarse
// per-element guard for that failure, not a precision bound: a mis-scaled
// element is wrong by a FACTOR, and this catches it, while the mean-relative
// criterion above is the tight one. The scale term is the run's own mean
// magnitude, because an element whose true value sits near zero after
// cancellation has an unbounded relative error and bounding it would only make
// the test flaky.
void CheckNoElementIsCatastrophic(const std::vector<float>& got, const std::vector<float>& want) {
  REQUIRE(got.size() == want.size());
  double mean_abs = 0.0;
  for (float w : want) mean_abs += std::fabs(static_cast<double>(w));
  mean_abs /= static_cast<double>(want.size());
  size_t bad = 0, first_bad = 0;
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    const double tol = 0.05 * mean_abs + 0.05 * std::fabs(static_cast<double>(want[i]));
    if (d > worst) worst = d;
    if (!(d <= tol)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(want[first_bad]);
    CAPTURE(worst);
    CAPTURE(mean_abs);
  }
  CHECK(bad == 0);
}

// AN ALL-ZERO OUTPUT PASSES EVERY RATIO. Both arms are asked to be non-vacuous.
void CheckNonVacuous(const std::vector<float>& v) {
  size_t nonzero = 0;
  for (float x : v)
    if (x != 0.0f) ++nonzero;
  CAPTURE(nonzero);
  CAPTURE(v.size());
  CHECK(nonzero * 10 > v.size() * 9);
}

// The full grid this file drives, in one place, with the refusal each shape must
// get. THREE OF THESE ARE REFUSED SINCE #1437, and the column is why this file
// no longer pretends otherwise: the sm120 blockwise collective's `can_implement`
// requires complete scale blocks (`N % 128 == 0`, on the collective's N
// unswapped and on its M under swap_ab) and full tiles in K (`K % 128 == 0`).
// A ragged N or K is expressible in the deduced scale layout and CUTLASS refuses
// it anyway. The sm90 sibling collective does not, which is why this is an sm120
// gap and not a property of block-wise FP8.
const std::vector<BlockCase>& Grid() {
  static const std::vector<BlockCase> grid = {
      // upstream's own CUTLASS case: DSV3 kv_a_proj_with_mqa, ragged N. 576 is
      // 4*128 + 64, and THE sm120 ARM CANNOT SERVE IT. This is the shape that
      // threw `cutlass Invalid status` on a GB10 on 2026-08-20.
      {32, 576, 7168, 128, 128, Fp8BlockScaledRefusal::kScaleBlockN},
      // the M sweep, one per tile config and both clauses of the swap_ab OR.
      {1, 512, 1024, 128, 128},     // swapab: M <= 64, and decode's own shape
      {7, 512, 1024, 128, 128},     // swapab: M <= 64
      // ragged N again, on the OTHER tile config, so the refusal is not pinned
      // to one config by accident.
      {8, 576, 1024, 128, 128, Fp8BlockScaledRefusal::kScaleBlockN},
      {83, 512, 1024, 128, 128},    // swapab: M > 64 but 83 % 4 != 0
      {200, 512, 1024, 128, 128},   // pingpong: M > 64, 200 % 4 == 0, M <= 256
      {512, 512, 1024, 128, 128},   // default: M > 256
      // THE N FLOOR, AND THE ONLY ENTRY THAT BINDS OVER-REFUSAL. Every other
      // servable N here is 512, and 512 is a multiple of 256, 128 and 64 alike,
      // so a `kFp8BlockScaledScaleBlockN` raised above 128 -- a predicate that
      // turns away shapes the collective CAN implement -- left this whole file
      // green while refusing real work. 128 is the smallest N the sm120
      // collective serves: one complete scale block, and exactly one N-tile of
      // the pingpong config. M = 200 is reused from the entry above on purpose
      // (200 > 64, 200 % 4 == 0, 200 <= 256, so `Fp8BlockScaledConfigFor` names
      // pingpong for both) -- this entry is an N-axis probe and adds no new M
      // behaviour to the sweep.
      {200, 128, 1024, 128, 128},   // pingpong, N at the collective's floor
      // ragged N on the UNSWAPPED path: M=512 takes `default`, where the ragged
      // extent is the collective's own N at ScaleGranularityN=128 rather than its
      // M under swap. Same number, different slot, same refusal — which is what
      // makes "the swap is the bug" falsifiable here.
      {512, 576, 1024, 128, 128, Fp8BlockScaledRefusal::kScaleBlockN},
      // a ragged K-block that is still 16-aligned: 1088 = 8*128 + 64. It fails a
      // DIFFERENT line of `can_implement` from the two above, which is what makes
      // it the independent check on the root cause rather than a repetition.
      {8, 512, 1088, 128, 128, Fp8BlockScaledRefusal::kTileK},
  };
  return grid;
}

// The servable half, which is the half G7 compares against the CPU reference.
std::vector<BlockCase> ServableGrid() {
  std::vector<BlockCase> v;
  for (const BlockCase& c : Grid())
    if (c.expect == Fp8BlockScaledRefusal::kNone) v.push_back(c);
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// G6 — the instrument's own precondition. RUNS EVERYWHERE, device or not.
// ---------------------------------------------------------------------------
//
// IT USED TO ASSERT THAT EVERY SHAPE HERE IS ACCEPTED, AND IT PASSED WHILE THE
// KERNEL THREW ON THREE OF THEM (#1437). The predicate and the collective's
// `can_implement` disagreed, and a precondition that agrees with the wrong side
// is worse than none: it certified the grid on the very machine that could not
// run it. So the case now asserts the PARTITION against the hand-written
// `expect` column, which is the same obligation stated in a form that can fail.
TEST_CASE("G6 every shape this file drives gets the refusal the collective gives it") {
  size_t servable = 0, scale_block_n = 0, tile_k = 0;
  for (const BlockCase& c : Grid()) {
    CAPTURE(c.m);
    CAPTURE(c.n);
    CAPTURE(c.k);
    CHECK(Fp8BlockScaledRefusalFor(c.n, c.k, c.block_n, c.block_k) == c.expect);
    // ... and the activation quant's own divisibility is not assumed: the op
    // tiles K with cdiv, so a ragged K is legal for THIS op even where
    // vt::QuantFp8Group would refuse it.
    CHECK(c.k > 0);
    CHECK(CDiv(c.k, c.block_k) >= 1);
    if (c.expect == Fp8BlockScaledRefusal::kNone) ++servable;
    if (c.expect == Fp8BlockScaledRefusal::kScaleBlockN) ++scale_block_n;
    if (c.expect == Fp8BlockScaledRefusal::kTileK) ++tile_k;
  }
  // BOTH LANES ARE NON-EMPTY. A grid that drifted to all-servable would silently
  // stop testing the refusal, and a grid that drifted to all-refused would leave
  // G7 with nothing to compare against the CPU reference while still reporting a
  // pass — which is the exact shape of the failure #1437 found.
  CHECK(servable >= 6);
  CHECK(scale_block_n >= 2);
  CHECK(tile_k >= 1);
  CHECK(servable + scale_block_n + tile_k == Grid().size());

  // All three tile configs are exercised by the SERVABLE half; a sweep whose
  // only entry for a config was a refused shape would compare nothing there.
  bool swap = false, ping = false, deflt = false;
  for (const BlockCase& c : ServableGrid()) {
    switch (Fp8BlockScaledConfigFor(c.m)) {
      case Fp8BlockScaledConfig::kSwapAb:
        swap = true;
        break;
      case Fp8BlockScaledConfig::kPingpong:
        ping = true;
        break;
      case Fp8BlockScaledConfig::kDefault:
      case Fp8BlockScaledConfig::kCount:
        deflt = true;
        break;
    }
  }
  CHECK(swap);
  CHECK(ping);
  CHECK(deflt);
  // AND THE SERVABLE LANE REACHES THE N FLOOR, which is what makes the partition
  // above bind in the OVER-refusal direction. A grid whose servable N is 512
  // everywhere cannot tell 128 from 256 or 64: raise
  // `kFp8BlockScaledScaleBlockN` to 256 and every check in this file still
  // passes, because 512 % 256 == 0. One servable entry at N = 128 fails the
  // moment the predicate refuses more than the collective does.
  bool servable_at_n_floor = false;
  for (const BlockCase& c : ServableGrid())
    if (c.n == kFp8BlockScaledScaleBlockN) servable_at_n_floor = true;
  CHECK(servable_at_n_floor);
  // The refused half spans the SWAPPED and the UNSWAPPED path, so "the swap is
  // the bug" is falsifiable rather than merely unasserted. A ragged N is refused
  // on both, because it is the collective's M under swap and its N without it,
  // at granularity 128 either way.
  bool refused_swapped = false, refused_unswapped = false;
  for (const BlockCase& c : Grid()) {
    if (c.expect == Fp8BlockScaledRefusal::kNone) continue;
    if (Fp8BlockScaledConfigFor(c.m) == Fp8BlockScaledConfig::kSwapAb)
      refused_swapped = true;
    else
      refused_unswapped = true;
  }
  CHECK(refused_swapped);
  CHECK(refused_unswapped);
}

// ---------------------------------------------------------------------------
// G2 — upstream's CUTLASS case, and the one adaptation the arch forces
// ---------------------------------------------------------------------------
//
// THE ADAPTATION, STATED IN FULL, because `.agents/porting.md` allows one only
// when it is unavoidable and only when it is documented. Upstream's
// `test_w8a8_block_fp8_cutlass_matmul` is M=32, N=576, K=7168, and N=576 is
// 4*128 + 64 because DSV3's `kv_a_proj_with_mqa` is that wide. On sm120 that
// shape CANNOT REACH THE GEMM: the collective's `can_implement` requires
// `N % ScaleGranularityN == 0` with the granularity at 128. Upstream gates the
// module only at `get_device_capability() < (9, 0)`
// (`tests/kernels/quantization/test_block_fp8.py`), so cc 12.1 passes its gate
// and upstream's own test cannot succeed on this arch either; the shape is
// served by the sm90/sm100 collectives, which have no such line.
//
// So this case does BOTH halves, and neither is the other's substitute:
//
//   1. upstream's exact shape must be REFUSED BY NAME. That is the fix #1437
//      owns — before it, the shape reached CUTLASS and came back as
//      `cutlass Invalid status`, which names no constraint and no dimension.
//   2. upstream's FIXTURE, CRITERION AND FORMULA are then run on the nearest
//      servable N — 512 instead of 576, everything else identical — so the
//      value comparison this case exists for still happens and still fails
//      loudly if the kernel regresses on a shape it CAN serve.
//
// Half 2 HAS now run on a device, once. Its {32,512,7168} was one of the seven
// shapes compared against the CPU reference in the 2026-08-20 GB10 run the row's
// `## Owed` records. That is evidence the kernel computes correct values on
// those seven shapes and on nothing else: it is not a token gate, it is not a
// speed number, and it does not reach a shape outside the grid. Nothing here
// softens any of the three.
TEST_CASE("G2 upstream's CUTLASS case is refused by name, and its criterion runs on the "
          "nearest servable N") {
  // Half 1's host tier runs EVERYWHERE, because the predicate is host-side and a
  // GPU-less box can hold it to the collective's line.
  const BlockCase up{32, 576, 7168, 128, 128, Fp8BlockScaledRefusal::kScaleBlockN};
  CHECK(Fp8BlockScaledRefusalFor(up.n, up.k, up.block_n, up.block_k) == up.expect);

  if (!HasCuda()) {
    MESSAGE("NO CUDA DEVICE: G2's device half did not run HERE, and a skip is not a pass. "
            "The on-hardware result is recorded in #1189 M5's `## Owed`: dgx:gpu0 (GB10, cc "
            "12.1), 2026-08-20, 5 cases / 136 assertions / 0 failed.");
    return;
  }

  // Half 1, on the device: the arm refuses before it allocates or launches, and
  // the message names the dimension, its value and the granularity — the three
  // things `cutlass Invalid status` did not say.
  {
    Backend& backend = vt::GetBackend(DeviceType::kCUDA);
    QueueGuard g(backend);
    const int64_t kt = CDiv(up.k, up.block_k), nt = CDiv(up.n, up.block_n);
    DeviceTensor da(backend, g.q, DType::kI8, {up.m, up.k});
    DeviceTensor das(backend, g.q, DType::kF32, {up.m, kt});
    DeviceTensor db(backend, g.q, DType::kI8, {up.n, up.k});
    DeviceTensor dbs(backend, g.q, DType::kF32, {nt, kt});
    DeviceTensor dout(backend, g.q, DType::kBF16, {up.m, up.n});
    const Fp8BlockScaledStats before = Fp8BlockScaledStatsSnapshot();
    std::string what;
    try {
      vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                               dbs.tensor(), up.block_n, up.block_k);
      FAIL("N = 576 must be refused by name on the sm120 arm");
    } catch (const std::runtime_error& e) {
      what = e.what();
    }
    CHECK(what.find("N is 576") != std::string::npos);
    CHECK(what.find("multiple of 128") != std::string::npos);
    CHECK(what.find("sm120") != std::string::npos);
    // NOT the bare CUTLASS status. That string is what a user got before #1437,
    // and a refusal that degraded back to it would pass every check above that
    // only looked for a throw.
    CHECK(what.find("cutlass Invalid status") == std::string::npos);
    const Fp8BlockScaledStats after = Fp8BlockScaledStatsSnapshot();
    CHECK(after.refused - before.refused == 1);
    CHECK(after.dispatched() == before.dispatched());
  }

  // Half 2: upstream's fixture and criterion, with N = 512. 512 is the largest
  // multiple of 128 below 576, so it is the nearest shape of upstream's own case
  // that this collective can implement.
  const BlockCase c{32, 512, 7168, 128, 128};
  REQUIRE(Fp8BlockScaledRefusalFor(c.n, c.k, c.block_n, c.block_k) ==
          Fp8BlockScaledRefusal::kNone);
  const auto a = RandomFp8Bytes(static_cast<size_t>(c.m) * c.k, 1);
  const auto b = RandomFp8Bytes(static_cast<size_t>(c.n) * c.k, 2);
  const auto as = RandomScales(static_cast<size_t>(c.m) * CDiv(c.k, c.block_k), 3);
  const auto bs = RandomScales(static_cast<size_t>(CDiv(c.n, c.block_n)) * CDiv(c.k, c.block_k), 4);

  const std::vector<float> ref = ReferenceBf16(c, a, as, b, bs);
  const std::vector<float> got = DeviceBf16(c, a, as, b, bs);

  CheckNonVacuous(ref);
  CheckNonVacuous(got);
  const double rel = RelDiff(got, ref);
  CAPTURE(rel);
  CHECK(rel < 0.001);  // upstream's own criterion, verbatim
  CheckNoElementIsCatastrophic(got, ref);
}

// ---------------------------------------------------------------------------
// G7 — the M sweep, with the dispatch counter
// ---------------------------------------------------------------------------
//
// SINCE #1437 the sweep has two arms, because the grid does. A shape the sm120
// collective cannot implement is driven to its NAMED REFUSAL, and a shape it can
// is compared against the CPU reference exactly as before. Neither arm is
// weakened by the other's presence: the servable half still covers all three
// tile configs (G6 asserts that it does), so a kernel that regressed on a shape
// it can serve still fails here.
TEST_CASE("G7 every tile config matches the CPU reference, and every unservable shape is "
          "refused by name") {
  if (!HasCuda()) {
    MESSAGE("NO CUDA DEVICE: G7 did not run HERE, and a skip is not a pass. The on-hardware "
            "result is recorded in #1189 M5's `## Owed`: dgx:gpu0 (GB10, cc 12.1), 2026-08-20, "
            "5 cases / 136 assertions / 0 failed.");
    return;
  }
  uint32_t seed = 100;
  for (const BlockCase& c : Grid()) {
    CAPTURE(c.m);
    CAPTURE(c.n);
    CAPTURE(c.k);

    // THE REFUSED ENTRIES ARE STILL DRIVEN, and that is deliberate. Skipping
    // them would leave the sweep proving nothing about the shapes the arm has to
    // turn away, and the previous version of this loop ABORTED AT Grid()[0] on
    // one of them, so seven of eight shapes were never attempted at all. Each
    // refused entry now asserts a named refusal, no dispatch, and one refusal
    // counted — the same three facts the servable entries assert in the mirror.
    if (c.expect != Fp8BlockScaledRefusal::kNone) {
      Backend& backend = vt::GetBackend(DeviceType::kCUDA);
      QueueGuard g(backend);
      const int64_t kt = CDiv(c.k, c.block_k), nt = CDiv(c.n, c.block_n);
      DeviceTensor da(backend, g.q, DType::kI8, {c.m, c.k});
      DeviceTensor das(backend, g.q, DType::kF32, {c.m, kt});
      DeviceTensor db(backend, g.q, DType::kI8, {c.n, c.k});
      DeviceTensor dbs(backend, g.q, DType::kF32, {nt, kt});
      DeviceTensor dout(backend, g.q, DType::kBF16, {c.m, c.n});
      const Fp8BlockScaledStats before = Fp8BlockScaledStatsSnapshot();
      std::string what;
      try {
        vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                                 dbs.tensor(), c.block_n, c.block_k);
        FAIL("this shape must be refused by name on the sm120 arm");
      } catch (const std::runtime_error& e) {
        what = e.what();
      }
      // The message names the RAGGED DIMENSION, not merely some dimension: a
      // kTileK that answered with the N text would name 512 here and pass a
      // check that only looked for "128".
      const std::string ragged =
          c.expect == Fp8BlockScaledRefusal::kTileK
              ? "K is " + std::to_string(c.k)
              : "N is " + std::to_string(c.n);
      CAPTURE(ragged);
      CHECK(what.find(ragged) != std::string::npos);
      CHECK(what.find("multiple of 128") != std::string::npos);
      CHECK(what.find("cutlass Invalid status") == std::string::npos);
      const Fp8BlockScaledStats after = Fp8BlockScaledStatsSnapshot();
      CHECK(after.refused - before.refused == 1);
      CHECK(after.dispatched() == before.dispatched());
      continue;
    }

    const auto a = RandomFp8Bytes(static_cast<size_t>(c.m) * c.k, seed++);
    const auto b = RandomFp8Bytes(static_cast<size_t>(c.n) * c.k, seed++);
    const auto as = RandomScales(static_cast<size_t>(c.m) * CDiv(c.k, c.block_k), seed++);
    const auto bs =
        RandomScales(static_cast<size_t>(CDiv(c.n, c.block_n)) * CDiv(c.k, c.block_k), seed++);

    const std::vector<float> ref = ReferenceBf16(c, a, as, b, bs);
    const Fp8BlockScaledStats before = Fp8BlockScaledStatsSnapshot();
    const std::vector<float> got = DeviceBf16(c, a, as, b, bs);
    const Fp8BlockScaledStats after = Fp8BlockScaledStatsSnapshot();

    CheckNonVacuous(got);
    const double rel = RelDiff(got, ref);
    CAPTURE(rel);
    CHECK(rel < 0.001);
    CheckNoElementIsCatastrophic(got, ref);

    // EXACTLY ONE dispatch, on the config the heuristic named. A kernel that
    // silently fell through to another config, or a fallback that ran instead
    // of this kernel, is invisible to every value comparison above — the
    // fallback would be numerically BETTER — and visible here.
    CHECK(after.dispatched() - before.dispatched() == 1);
    CHECK(after.refused == before.refused);
    switch (Fp8BlockScaledConfigFor(c.m)) {
      case Fp8BlockScaledConfig::kSwapAb:
        CHECK(after.swap_ab - before.swap_ab == 1);
        break;
      case Fp8BlockScaledConfig::kPingpong:
        CHECK(after.pingpong - before.pingpong == 1);
        break;
      case Fp8BlockScaledConfig::kDefault:
      case Fp8BlockScaledConfig::kCount:
        CHECK(after.deflt - before.deflt == 1);
        break;
    }
  }
}

// ---------------------------------------------------------------------------
// G8 — the f32 sink
// ---------------------------------------------------------------------------
TEST_CASE("G8 an f32 out is the bf16 epilogue value cast up") {
  if (!HasCuda()) {
    MESSAGE("NO CUDA DEVICE: G8 did not run HERE, and a skip is not a pass. The on-hardware "
            "result is recorded in #1189 M5's `## Owed`: dgx:gpu0 (GB10, cc 12.1), 2026-08-20, "
            "5 cases / 136 assertions / 0 failed.");
    return;
  }
  const BlockCase c{8, 512, 1024, 128, 128};
  const auto a = RandomFp8Bytes(static_cast<size_t>(c.m) * c.k, 11);
  const auto b = RandomFp8Bytes(static_cast<size_t>(c.n) * c.k, 12);
  const auto as = RandomScales(static_cast<size_t>(c.m) * CDiv(c.k, c.block_k), 13);
  const auto bs = RandomScales(static_cast<size_t>(CDiv(c.n, c.block_n)) * CDiv(c.k, c.block_k), 14);

  const std::vector<float> bf16 = DeviceBf16(c, a, as, b, bs);

  Backend& backend = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(backend);
  DeviceTensor da(backend, g.q, DType::kI8, {c.m, c.k}, a.data());
  DeviceTensor das(backend, g.q, DType::kF32, {c.m, CDiv(c.k, c.block_k)}, as.data());
  DeviceTensor db(backend, g.q, DType::kI8, {c.n, c.k}, b.data());
  DeviceTensor dbs(backend, g.q, DType::kF32,
                   {CDiv(c.n, c.block_n), CDiv(c.k, c.block_k)}, bs.data());
  DeviceTensor dout(backend, g.q, DType::kF32, {c.m, c.n});
  vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                           dbs.tensor(), c.block_n, c.block_k);
  std::vector<float> f32(static_cast<size_t>(c.m) * c.n);
  dout.Download(g.q, f32.data());

  CheckNonVacuous(f32);
  // EXACT, elementwise. The collective is instantiated for bf16 only, so an f32
  // out is a widening of the same value and nothing more; anything else would
  // mean a second, wider compute path had appeared.
  size_t mismatched = 0;
  for (size_t i = 0; i < f32.size(); ++i)
    if (f32[i] != bf16[i]) ++mismatched;
  CAPTURE(mismatched);
  CHECK(mismatched == 0);
}

// ---------------------------------------------------------------------------
// G9 — the refusals, on a device
// ---------------------------------------------------------------------------
TEST_CASE("G9 the CUDA arm refuses by name what cutlass cannot implement") {
  if (!HasCuda()) {
    MESSAGE("NO CUDA DEVICE: G9 did not run HERE, and a skip is not a pass. The on-hardware "
            "result is recorded in #1189 M5's `## Owed`: dgx:gpu0 (GB10, cc 12.1), 2026-08-20, "
            "5 cases / 136 assertions / 0 failed.");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(backend);

  // K = 3884 is upstream's own non-round K and 3884 % 16 == 12, so cutlass
  // cannot take it and upstream reroutes it to triton. The CPU arm of this same
  // op runs it; the message says so.
  {
    const BlockCase c{8, 512, 3884, 128, 128};
    const int64_t kt = CDiv(c.k, c.block_k), nt = CDiv(c.n, c.block_n);
    DeviceTensor da(backend, g.q, DType::kI8, {c.m, c.k});
    DeviceTensor das(backend, g.q, DType::kF32, {c.m, kt});
    DeviceTensor db(backend, g.q, DType::kI8, {c.n, c.k});
    DeviceTensor dbs(backend, g.q, DType::kF32, {nt, kt});
    DeviceTensor dout(backend, g.q, DType::kBF16, {c.m, c.n});
    const Fp8BlockScaledStats before = Fp8BlockScaledStatsSnapshot();
    std::string what;
    try {
      vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                               dbs.tensor(), c.block_n, c.block_k);
      FAIL("a K of 3884 must be refused on the CUDA arm");
    } catch (const std::runtime_error& e) {
      what = e.what();
    }
    CHECK(what.find("3884") != std::string::npos);
    CHECK(what.find("triton") != std::string::npos);
    const Fp8BlockScaledStats after = Fp8BlockScaledStatsSnapshot();
    // A REFUSAL IS NOT A DISPATCH.
    CHECK(after.refused - before.refused == 1);
    CHECK(after.dispatched() == before.dispatched());
  }

  // A block geometry the collective is not instantiated for.
  {
    const BlockCase c{8, 512, 1024, 64, 128};
    const int64_t kt = CDiv(c.k, c.block_k), nt = CDiv(c.n, c.block_n);
    DeviceTensor da(backend, g.q, DType::kI8, {c.m, c.k});
    DeviceTensor das(backend, g.q, DType::kF32, {c.m, kt});
    DeviceTensor db(backend, g.q, DType::kI8, {c.n, c.k});
    DeviceTensor dbs(backend, g.q, DType::kF32, {nt, kt});
    DeviceTensor dout(backend, g.q, DType::kBF16, {c.m, c.n});
    std::string what;
    try {
      vt::MatmulFp8BlockScaled(g.q, dout.tensor(), da.tensor(), das.tensor(), db.tensor(),
                               dbs.tensor(), c.block_n, c.block_k);
      FAIL("a block_n of 64 must be refused on the CUDA arm");
    } catch (const std::runtime_error& e) {
      what = e.what();
    }
    CHECK(what.find("block_n is 64") != std::string::npos);
  }
}
