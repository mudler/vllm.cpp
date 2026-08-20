// vllm.cpp original (vt runtime). Unit tests for vt::DFlashGroupedConv — the
// DFlash2 draft's GROUPED DYNAMIC DEPTHWISE CONVOLUTION (SPEC-DFLASH2 W2,
// #1314).
//
// BEYOND-PIN. The reference is `_grouped_conv`
// (vllm/model_executor/models/qwen3_dflash2.py @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`), and the sequential expectation
// below is upstream's OWN reference loop from
// `tests/v1/spec_decode/test_dflash2.py::test_grouped_conv_matches_reference` at
// that head, transcribed rather than reinvented:
//
//   for position in range(block_size):
//       for tap in range(min(taps, position + 1)):
//           expected[:, position] += (base[tap] + delta[:, position, tap, :, None])
//                                    * hidden_blocks[:, position - tap]
//
// WHAT IS GATED, and why each case exists.
//
//  * BOTH position-mask arms. Upstream special-cases a power-of-two block to
//    `position & (block-1)` and otherwise uses `position % block`. Upstream's own
//    parametrize covers 5 and 8; this file covers 5 (modulo arm), 8 and 16 —
//    8 and 16 because they are the blocks the two PUBLISHED checkpoints ship
//    (`z-lab/Qwen3.8-27B-DFlash2` block 8, `z-lab/Muse-Glimmer-30B-DFlash2`
//    block 16) and neither upstream parameter reaches 16.
//  * The BLOCK BOUNDARY. A tap must contribute NOTHING across it. The dedicated
//    case below feeds a row whose predecessor lives in the previous block and
//    asserts the output ignores it — that zeroing is the whole reason this is a
//    block convolution rather than a sequence one, and a token gate cannot see
//    it (a draft that leaks across the boundary still emits the target's tokens
//    and only loses acceptance).
//  * The GROUP map. `delta` is per GROUP and `base` is per CHANNEL; a
//    transposed or mis-divided `g(c) = c / group_size` still produces finite,
//    plausible output. The case asserts two channels of the same group take the
//    same delta and two channels of different groups do not.
//  * The SIDE. `base_kernel` dim 0 is prepare/finish, NOT a tap. The case
//    asserts side 1 reads base[1] and coefficients[:,1] and differs from side 0
//    on the same input.
//  * THE ROUNDING POLICY, in bf16 and on CPU. Every step of this op rounds to
//    the tensor dtype, exactly as upstream's bf16 chain materializes it. The
//    cases above the CUDA section run in f32, where that rounding is the
//    IDENTITY by construction and therefore invisible; the two bf16 cases pin it
//    directly, one with hand-computed literals that differ from the
//    round-once-at-the-end answer and one bit-exact against a reference that
//    rounds where upstream materializes.
//  * CUDA == CPU, BIT-FOR-BIT. The consequence of the policy above: no
//    reduction-order freedom and no envelope to hide behind. That case is
//    written and has NEVER RUN on this host (no `nvcc`, so it reports `no CUDA
//    backend; skipping`); it is `## Owed` O6 of the row's spec and is not
//    counted as coverage here.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DFlashGroupedConvArgs;
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

DFlashGroupedConvArgs Args(int64_t block, int64_t taps, int64_t groups, int64_t gsize,
                           int64_t side) {
  DFlashGroupedConvArgs a;
  a.block_size = block;
  a.taps = taps;
  a.num_groups = groups;
  a.group_size = gsize;
  a.side = side;
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

// bf16 helpers. Every published DFlash2 checkpoint stores this op's tensors in
// bf16, and bf16 is the ONLY arm on which the rounding policy is observable: on
// f32 the kernel's per-step rounding is the identity by construction, so no f32
// case can see it.
std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::F32ToBF16(v[i]);
  return o;
}
Tensor Bf16(std::vector<uint16_t>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kBF16, Cpu(), shape);
}
// Round an f32 through bf16 and back -- one MATERIALIZATION of an intermediate.
float RB(float v) { return vt::BF16ToF32(vt::F32ToBF16(v)); }

// UPSTREAM's reference loop again, at upstream's bf16 MATERIALIZATION points.
// Written from the upstream chain rather than from our kernel: `base + delta`,
// `coefficients * blocks` and `output += ...` each produce a tensor of the model
// dtype, so on a bf16 draft each of the three rounds, and the accumulation runs
// tap-ascending because upstream's `for tap in range(1, taps)` does.
//
// This is the whole reason the CUDA arm can be specified BIT-IDENTICAL rather
// than within an envelope, and it is not free: rounding ONCE at the end gives a
// different answer, which the hand-computed case below pins with literals.
std::vector<uint16_t> ReferenceBf16(const std::vector<uint16_t>& hidden,
                                    const std::vector<uint16_t>& delta,
                                    const std::vector<uint16_t>& base, int64_t batch,
                                    int64_t block, int64_t taps, int64_t groups,
                                    int64_t gsize, int64_t sides, int64_t side) {
  const int64_t H = groups * gsize;
  std::vector<uint16_t> out(static_cast<size_t>(batch * block * H), 0);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t pos = 0; pos < block; ++pos) {
      const int64_t row = b * block + pos;
      for (int64_t g = 0; g < groups; ++g) {
        for (int64_t j = 0; j < gsize; ++j) {
          const int64_t c = g * gsize + j;
          float acc = 0.0f;
          for (int64_t tap = 0; tap < taps && tap <= pos; ++tap) {
            const size_t di =
                static_cast<size_t>(((row * sides + side) * taps + tap) * groups + g);
            const size_t bi = static_cast<size_t>((side * taps + tap) * H + c);
            const float k = RB(vt::BF16ToF32(base[bi]) + vt::BF16ToF32(delta[di]));
            const float term =
                RB(k * vt::BF16ToF32(hidden[static_cast<size_t>((row - tap) * H + c)]));
            acc = (tap == 0) ? term : RB(acc + term);
          }
          out[static_cast<size_t>(row * H + c)] = vt::F32ToBF16(acc);
        }
      }
    }
  }
  return out;
}

// UPSTREAM's reference loop, transcribed from
// tests/v1/spec_decode/test_dflash2.py::test_grouped_conv_matches_reference @ the
// PR head. `hidden` is [batch*block, H] and every request block is contiguous and
// block-aligned, exactly as upstream's `hidden_states.unflatten` assumes. f32
// arithmetic: the op rounds to the tensor dtype at each step, and on f32 that
// rounding is the identity, so this is the exact expectation for an f32 run.
std::vector<float> Reference(const std::vector<float>& hidden, const std::vector<float>& delta,
                             const std::vector<float>& base, int64_t batch, int64_t block,
                             int64_t taps, int64_t groups, int64_t gsize, int64_t sides,
                             int64_t side) {
  const int64_t H = groups * gsize;
  std::vector<float> out(static_cast<size_t>(batch * block * H), 0.0f);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t pos = 0; pos < block; ++pos) {
      const int64_t row = b * block + pos;
      for (int64_t tap = 0; tap < taps && tap <= pos; ++tap) {
        for (int64_t g = 0; g < groups; ++g) {
          const size_t di = static_cast<size_t>(((row * sides + side) * taps + tap) * groups + g);
          for (int64_t j = 0; j < gsize; ++j) {
            const int64_t c = g * gsize + j;
            const size_t bi = static_cast<size_t>((side * taps + tap) * H + c);
            out[static_cast<size_t>(row * H + c)] +=
                (base[bi] + delta[di]) * hidden[static_cast<size_t>((row - tap) * H + c)];
          }
        }
      }
    }
  }
  return out;
}

// Drive the op over random f32 inputs at one shape and compare to Reference().
void RunReferenceCase(int64_t batch, int64_t block, int64_t taps, int64_t groups,
                      int64_t gsize, int64_t sides, int64_t side, uint32_t seed) {
  const int64_t H = groups * gsize;
  const int64_t T = batch * block;
  std::vector<float> hidden = RandF32(static_cast<size_t>(T * H), seed);
  std::vector<float> delta = RandF32(static_cast<size_t>(T * sides * taps * groups), seed + 1);
  std::vector<float> base = RandF32(static_cast<size_t>(sides * taps * H), seed + 2);
  std::vector<float> got(static_cast<size_t>(T * H), 0.0f);

  Tensor tx = F32(hidden, {T, H});
  Tensor tc = F32(delta, {T, sides, taps, groups});
  Tensor tb = F32(base, {sides, taps, H});
  Tensor to = F32(got, {T, H});
  Queue q = Q();
  vt::DFlashGroupedConv(q, to, tx, tc, tb, Args(block, taps, groups, gsize, side));

  const std::vector<float> want =
      Reference(hidden, delta, base, batch, block, taps, groups, gsize, sides, side);
  for (size_t i = 0; i < want.size(); ++i) {
    INFO("index ", i);
    CHECK(got[i] == doctest::Approx(want[i]).epsilon(1e-5));
  }
}

}  // namespace

TEST_CASE("dflash2-grouped-conv matches upstream's sequential reference at block 5, 8 and 16") {
  // Upstream's own parametrize shape (batch 3, taps 3, groups 4, group_size 2),
  // at its two block parameters plus the second published checkpoint's 16.
  //   block 5  -> the `position % block` arm
  //   block 8  -> the `position & (block-1)` arm, and z-lab/Qwen3.8-27B-DFlash2
  //   block 16 -> the same arm at the shape z-lab/Muse-Glimmer-30B-DFlash2 ships
  RunReferenceCase(/*batch=*/3, /*block=*/5, /*taps=*/3, /*groups=*/4, /*gsize=*/2,
                   /*sides=*/1, /*side=*/0, /*seed=*/11);
  RunReferenceCase(3, 8, 3, 4, 2, 1, 0, 22);
  RunReferenceCase(3, 16, 3, 4, 2, 1, 0, 33);
}

TEST_CASE("dflash2-grouped-conv matches the reference at BOTH published checkpoint shapes") {
  // taps 2 and conv_group_size 16 are what both published DFlash2 configs
  // declare. Hidden is reduced from 5120/6656 to keep the reference loop cheap;
  // what the case pins is the taps/group/block triple and the 2-SIDE buffer the
  // real `kernel_projection` produces (out = 2*taps*num_groups).
  RunReferenceCase(/*batch=*/2, /*block=*/8, /*taps=*/2, /*groups=*/5, /*gsize=*/16,
                   /*sides=*/2, /*side=*/0, /*seed=*/44);
  RunReferenceCase(2, 8, 2, 5, 16, 2, 1, 55);
  RunReferenceCase(2, 16, 2, 6, 16, 2, 0, 66);
  RunReferenceCase(2, 16, 2, 6, 16, 2, 1, 77);
}

TEST_CASE("dflash2-grouped-conv RED: a tap contributes NOTHING across the block boundary") {
  // Two blocks of 2 rows, taps=2, one group of one channel, base = 1 for both
  // taps, delta = 0. x = [10, 20, 30, 40].
  //   row 0 (pos 0): tap 1 masked  -> 10
  //   row 1 (pos 1): 20 + 10       -> 30
  //   row 2 (pos 0): tap 1 MASKED  -> 30   <-- x[1]=20 is in the PREVIOUS block
  //   row 3 (pos 1): 40 + 30       -> 70
  // Without the mask row 2 would read 20 and answer 50. That leak is invisible to
  // a token gate: the verify is lossless, so only acceptance moves.
  std::vector<float> x = {10, 20, 30, 40};
  std::vector<float> delta(4 * 1 * 2 * 1, 0.0f);
  std::vector<float> base = {1, 1};
  std::vector<float> got(4, 0.0f);
  Tensor tx = F32(x, {4, 1});
  Tensor tc = F32(delta, {4, 1, 2, 1});
  Tensor tb = F32(base, {1, 2, 1});
  Tensor to = F32(got, {4, 1});
  Queue q = Q();
  vt::DFlashGroupedConv(q, to, tx, tc, tb, Args(/*block=*/2, /*taps=*/2, 1, 1, 0));
  CHECK(got[0] == doctest::Approx(10.0f));
  CHECK(got[1] == doctest::Approx(30.0f));
  CHECK(got[2] == doctest::Approx(30.0f));  // 50.0f iff the boundary leaks
  CHECK(got[3] == doctest::Approx(70.0f));
}

TEST_CASE("dflash2-grouped-conv: delta is per GROUP and base is per CHANNEL") {
  // One row, taps=1, 2 groups of 2 channels. base = [1,2,3,4] (per channel),
  // delta = [10, 100] (per group). x = 1 everywhere.
  //   channel 0 (group 0): 1 + 10  = 11
  //   channel 1 (group 0): 2 + 10  = 12
  //   channel 2 (group 1): 3 + 100 = 103
  //   channel 3 (group 1): 4 + 100 = 104
  // A transposed group map, or `g(c) = c % group_size`, answers a different
  // permutation of the same four numbers and stays finite.
  std::vector<float> x = {1, 1, 1, 1};
  std::vector<float> delta = {10, 100};
  std::vector<float> base = {1, 2, 3, 4};
  std::vector<float> got(4, 0.0f);
  Tensor tx = F32(x, {1, 4});
  Tensor tc = F32(delta, {1, 1, 1, 2});
  Tensor tb = F32(base, {1, 1, 4});
  Tensor to = F32(got, {1, 4});
  Queue q = Q();
  vt::DFlashGroupedConv(q, to, tx, tc, tb, Args(/*block=*/1, /*taps=*/1, /*groups=*/2,
                                                /*gsize=*/2, 0));
  CHECK(got[0] == doctest::Approx(11.0f));
  CHECK(got[1] == doctest::Approx(12.0f));
  CHECK(got[2] == doctest::Approx(103.0f));
  CHECK(got[3] == doctest::Approx(104.0f));
}

TEST_CASE("dflash2-grouped-conv: base_kernel dim 0 is the SIDE, not a tap") {
  // taps=1, 1 group, 1 channel, 2 SIDES. base = [[5],[7]], delta = [[0],[0]].
  // x = 2. Side 0 must answer 10 and side 1 must answer 14. Reading dim 0 as a
  // tap would make side 1 unreachable and answer 10 twice — the shape defect the
  // 27B header ruled out (base_kernel is (2, taps=2, 5120) with taps ALSO 2, so
  // the two axes are indistinguishable by size on the real checkpoint).
  std::vector<float> x = {2};
  std::vector<float> delta = {0, 0};
  std::vector<float> base = {5, 7};
  std::vector<float> got0(1, 0.0f), got1(1, 0.0f);
  Tensor tx = F32(x, {1, 1});
  Tensor tc = F32(delta, {1, 2, 1, 1});
  Tensor tb = F32(base, {2, 1, 1});
  Tensor t0 = F32(got0, {1, 1});
  Tensor t1 = F32(got1, {1, 1});
  Queue q = Q();
  vt::DFlashGroupedConv(q, t0, tx, tc, tb, Args(1, 1, 1, 1, /*side=*/0));
  vt::DFlashGroupedConv(q, t1, tx, tc, tb, Args(1, 1, 1, 1, /*side=*/1));
  CHECK(got0[0] == doctest::Approx(10.0f));
  CHECK(got1[0] == doctest::Approx(14.0f));
}

// ===========================================================================
// bf16 — the ROUNDING POLICY, on the only arm that can see it.
//
// The cases above run in f32, where the kernel's per-step rounding is the
// identity by construction, so NOTHING above this line can tell per-step
// rounding from rounding once at the end. That distinction is not cosmetic: it
// is the reason `DFlashGroupedConvArgs` specifies the CUDA mirror as
// BIT-IDENTICAL to this CPU reference rather than within a tolerance, and it is
// what makes the op agree with upstream's bf16 chain element for element. Both
// cases below are CPU-ONLY and do not wait for a GPU.

TEST_CASE("dflash2-grouped-conv bf16: the answer rounds PER STEP, not once at the end") {
  // Hand-computed with literals, so the expectation is independent of any
  // reference loop in this file. bf16 carries 8 significand bits, so above 256
  // it steps by 2 and round-to-nearest-EVEN decides every halfway case.
  //
  // Two blocks of 2 rows; taps 2; TWO groups of one channel each (so each
  // channel gets its own delta); one side. x alternates a large row and a small
  // one, base = {3, 5} per channel for both taps, delta = 2^-9 everywhere.
  //
  //   k(ch0) = bf16(3 + 0.001953125) = 3        (the `base + delta` rounding)
  //   k(ch1) = bf16(5 + 0.001953125) = 5
  //
  //   row 0 / row 2  (pos 0, tap 1 masked by the block boundary)
  //     ch0: bf16(3*89) = bf16(267) = 268   (267 is halfway; 268 is the even one)
  //     ch1: bf16(5*53) = bf16(265) = 264   (265 is halfway; 264 is the even one)
  //   row 1 / row 3  (pos 1, both taps)
  //     ch0: bf16(bf16(3*1) + bf16(3*89)) = bf16(3 + 268) = bf16(271) = 272
  //     ch1: bf16(bf16(5*1) + bf16(5*53)) = bf16(5 + 264) = bf16(269) = 268
  //
  // Round ONCE at the end instead and six of these eight outputs move:
  // row 0/2 ch1 -> 266, row 1/3 ch0 -> 270 and ch1 -> 270. That is the whole
  // difference between the two policies, and it is why the numbers below are
  // 268/264/272/268 rather than 268/266/270/270.
  const float kDelta = 0.001953125f;  // 2^-9, exactly representable in bf16
  std::vector<uint16_t> x = ToBf16({89.0f, 53.0f, 1.0f, 1.0f, 89.0f, 53.0f, 1.0f, 1.0f});
  std::vector<uint16_t> delta = ToBf16(std::vector<float>(4 * 1 * 2 * 2, kDelta));
  std::vector<uint16_t> base = ToBf16({3.0f, 5.0f, 3.0f, 5.0f});
  std::vector<uint16_t> got(8, 0);
  Tensor tx = Bf16(x, {4, 2});
  Tensor tc = Bf16(delta, {4, 1, 2, 2});
  Tensor tb = Bf16(base, {1, 2, 2});
  Tensor to = Bf16(got, {4, 2});
  Queue q = Q();
  vt::DFlashGroupedConv(q, to, tx, tc, tb,
                        Args(/*block=*/2, /*taps=*/2, /*groups=*/2, /*gsize=*/1, /*side=*/0));
  const std::vector<float> want = {268.0f, 264.0f, 272.0f, 268.0f,
                                   268.0f, 264.0f, 272.0f, 268.0f};
  for (size_t i = 0; i < want.size(); ++i) {
    INFO("row ", i / 2, " channel ", i % 2, " got ", vt::BF16ToF32(got[i]));
    CHECK(got[i] == vt::F32ToBF16(want[i]));
  }
}

TEST_CASE("dflash2-grouped-conv bf16 is BIT-EXACT against the per-step reference") {
  // The same claim at shapes with real fan-in, against ReferenceBf16 — which
  // rounds where UPSTREAM materializes rather than where our kernel does. Both
  // published block shapes and both sides; taps 3 at block 16 so more than one
  // accumulate rounding is chained. Asserted on the STORED bf16 patterns, not
  // through a tolerance, because bit-identity is what the op promises.
  struct Case {
    int64_t batch, block, taps, groups, gsize, sides, side;
    uint32_t seed;
  };
  const Case cases[] = {
      {2, 8, 2, 4, 2, 2, 0, 44}, {2, 8, 2, 4, 2, 2, 1, 55}, {2, 16, 3, 4, 2, 1, 0, 66}};
  for (const Case& cs : cases) {
    const int64_t H = cs.groups * cs.gsize;
    const int64_t T = cs.batch * cs.block;
    std::vector<uint16_t> x = ToBf16(RandF32(static_cast<size_t>(T * H), cs.seed));
    std::vector<uint16_t> delta = ToBf16(
        RandF32(static_cast<size_t>(T * cs.sides * cs.taps * cs.groups), cs.seed + 1));
    std::vector<uint16_t> base =
        ToBf16(RandF32(static_cast<size_t>(cs.sides * cs.taps * H), cs.seed + 2));
    std::vector<uint16_t> got(static_cast<size_t>(T * H), 0);
    Tensor tx = Bf16(x, {T, H});
    Tensor tc = Bf16(delta, {T, cs.sides, cs.taps, cs.groups});
    Tensor tb = Bf16(base, {cs.sides, cs.taps, H});
    Tensor to = Bf16(got, {T, H});
    Queue q = Q();
    vt::DFlashGroupedConv(q, to, tx, tc, tb,
                          Args(cs.block, cs.taps, cs.groups, cs.gsize, cs.side));
    const std::vector<uint16_t> want = ReferenceBf16(x, delta, base, cs.batch, cs.block,
                                                     cs.taps, cs.groups, cs.gsize,
                                                     cs.sides, cs.side);
    for (size_t i = 0; i < want.size(); ++i) {
      INFO("block ", cs.block, " taps ", cs.taps, " side ", cs.side, " index ", i);
      CHECK(got[i] == want[i]);
    }
  }
}

// ===========================================================================
// CUDA parity. Unlike the attention ops, this one is elementwise with a rounding
// to the tensor dtype after each materialized step, so CPU and CUDA must agree
// BIT-FOR-BIT and the gate asserts equality rather than an envelope.
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

// One shape, run on CPU and CUDA in the SAME dtype, asserted BIT-EQUAL.
void RunCudaParity(int64_t batch, int64_t block, int64_t taps, int64_t groups, int64_t gsize,
                   int64_t sides, int64_t side, DType dt, uint32_t seed) {
  const int64_t H = groups * gsize;
  const int64_t T = batch * block;
  const std::vector<float> xf = RandF32(static_cast<size_t>(T * H), seed);
  const std::vector<float> cf = RandF32(static_cast<size_t>(T * sides * taps * groups), seed + 1);
  const std::vector<float> bf = RandF32(static_cast<size_t>(sides * taps * H), seed + 2);
  const DFlashGroupedConvArgs a = Args(block, taps, groups, gsize, side);

  std::vector<uint16_t> xh, ch, bh;
  const void* xp = xf.data();
  const void* cp = cf.data();
  const void* bp = bf.data();
  if (dt == DType::kBF16) {
    xh = ToBf16(xf);
    ch = ToBf16(cf);
    bh = ToBf16(bf);
    xp = xh.data();
    cp = ch.data();
    bp = bh.data();
  }
  const size_t esz = vt::SizeOf(dt);
  const size_t obytes = static_cast<size_t>(T * H) * esz;

  std::vector<uint8_t> cpu_out(obytes, 0);
  {
    Tensor tx = Contig(const_cast<void*>(xp), dt, Cpu(), {T, H});
    Tensor tc = Contig(const_cast<void*>(cp), dt, Cpu(), {T, sides, taps, groups});
    Tensor tb = Contig(const_cast<void*>(bp), dt, Cpu(), {sides, taps, H});
    Tensor to = Contig(cpu_out.data(), dt, Cpu(), {T, H});
    Queue q = Q();
    vt::DFlashGroupedConv(q, to, tx, tc, tb, a);
  }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dx(gpu, g.q, dt, {T, H}, xp);
  DeviceTensor dc(gpu, g.q, dt, {T, sides, taps, groups}, cp);
  DeviceTensor db(gpu, g.q, dt, {sides, taps, H}, bp);
  DeviceTensor dout(gpu, g.q, dt, {T, H});
  vt::DFlashGroupedConv(g.q, dout.tensor(), dx.tensor(), dc.tensor(), db.tensor(), a);
  std::vector<uint8_t> got(obytes, 0);
  dout.Download(g.q, got.data());

  CHECK(std::memcmp(got.data(), cpu_out.data(), obytes) == 0);
}

}  // namespace

TEST_CASE("dflash2-grouped-conv CUDA is BIT-IDENTICAL to the CPU reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping CUDA dflash2-grouped-conv parity");
    return;
  }
  // Both published block shapes, in bf16 (what both checkpoints store) and f32,
  // plus the modulo arm and both sides.
  RunCudaParity(/*batch=*/4, /*block=*/8, /*taps=*/2, /*groups=*/320, /*gsize=*/16,
                /*sides=*/2, /*side=*/0, DType::kBF16, 101);
  RunCudaParity(4, 8, 2, 320, 16, 2, 1, DType::kBF16, 202);
  RunCudaParity(4, 16, 2, 416, 16, 2, 0, DType::kBF16, 303);
  RunCudaParity(4, 16, 2, 416, 16, 2, 1, DType::kBF16, 404);
  RunCudaParity(3, 5, 3, 4, 2, 1, 0, DType::kF32, 505);
  RunCudaParity(3, 8, 3, 4, 2, 1, 0, DType::kF32, 606);
}
