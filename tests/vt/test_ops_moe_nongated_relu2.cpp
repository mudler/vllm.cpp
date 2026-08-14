// NemotronH's NON-GATED relu^2 MoE expert (row MODEL-TEXT-nemotron-h, W2).
// Spec: .agents/specs/nemotron-h-model.md §4 W2. Issue #517.
//
// Upstream mirror, all @ 5559679229bc961848b121ccdeaa8fa5d79bec98 (vLLM 0.26.0.dev0):
//   models/nemotron_h.py:126-256 `NemotronHMoE`
//     :220 ckpt_names=("up_proj", "down_proj", "") — the EMPTY third entry is the
//          absent gate half. The expert is up_proj -> relu^2 -> down_proj; there
//          is no gate_proj tensor anywhere in the checkpoint.
//     :227 activation=activation_without_mul(config.mlp_hidden_act) ("relu2" ->
//          "relu2_no_mul")
//     :234 apply_routed_scale_to_output=True, :233 routed_scaling_factor
//   layers/fused_moe/activation.py:98 `activation_without_mul`, :34
//     `MoEActivation.RELU2_NO_MUL`, and :184 `apply_moe_activation`'s
//     RELU2_NO_MUL branch: F.relu(input, inplace=True);
//     torch.square(input, out=output).
//   layers/activation.py:609-628 `ReLUSquaredActivation`
//     (forward_native = torch.square(F.relu(x))).
//   csrc/libtorch_stable/activation_kernels.cu:672-678 `relu_squared_kernel` —
//     the DTYPE/ROUNDING ORDER this file pins: widen to f32, clamp at 0 in f32,
//     square in f32, then ONE round back to the store dtype.
//   layers/fused_moe/runner/moe_runner.py:390-407
//     `_maybe_apply_routed_scale_to_output` — :402-406, `fused_output *=
//     routed_scaling_factor` on the ASSEMBLED tensor with the SHARED output left
//     UNSCALED, then :722-725 `result = shared_output + fused_output`. The
//     :403-406 fp16 arm (divide `shared_output` instead) is unreachable here;
//     see the f16-out refusal case below.
//   layers/fused_moe/layer.py:291-300 — with apply_routed_scale_to_output the
//     ROUTER's routed_scaling_factor is forced to 1.0 ("so it ends up being a nop"),
//     which is why the scale is NOT visible in the router weights here.
//
// CPU-only by construction: the two things W2 adds (vt::MoeRelu2 and the
// routed-output scale on vt::MoeCombine) are both registered on kCPU, and the
// non-gated expert composite below runs through vt::MatmulBT — the same
// per-expert reference loop the CPU/GGUF MoE path already uses. The CUDA arms
// (kMoeGroupedGemmBf16 / kMoeGroupedGemmNvfp4Marlin) are registered for kCUDA
// only and are exercised where a GPU exists.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor F32_1(std::vector<float>& v) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {static_cast<int64_t>(v.size())});
}
Tensor F32_2(std::vector<float>& v, int64_t a, int64_t b) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b});
}
Tensor F32_3(std::vector<float>& v, int64_t a, int64_t b, int64_t c) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b, c});
}
Tensor Bf16_1(std::vector<uint16_t>& v) {
  return Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {static_cast<int64_t>(v.size())});
}

// doctest::Approx carries a 1.19e-5 ABSOLUTE floor (its `scale` defaults to 1.0),
// which silently accepts a dropped term on small values. Every non-exact
// comparison here goes through this explicit relative+absolute comparator.
bool Close(float a, float b, float rel = 1e-6f, float abs_tol = 1e-30f) {
  const float d = std::fabs(a - b);
  return d <= abs_tol || d <= rel * std::fmax(std::fabs(a), std::fabs(b));
}

float SiluRef(float x) { return x / (1.0f + std::exp(-x)); }

// Raw f32 bit pattern. The routed-scale PLACEMENT test below distinguishes two
// expressions that agree in exact arithmetic and differ only in where they
// round, so it can only be gated bitwise — any tolerance loose enough to be a
// tolerance accepts both.
uint32_t Bits(float v) {
  uint32_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  return u;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The activation itself — ReLUSquaredActivation (activation.py:609-628).
// ---------------------------------------------------------------------------

// torch.square(F.relu(x)) on exactly-representable inputs, so the expected
// values are exact and the check needs no tolerance at all.
TEST_CASE("moe relu2: mirrors torch.square(F.relu(x)) on the f32 arm") {
  std::vector<float> x = {-3.0f, -0.5f, -0.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
  std::vector<float> out(x.size(), -1.0f);
  Tensor xt = F32_1(x);
  Tensor ot = F32_1(out);
  Queue q = Q();
  vt::MoeRelu2(q, ot, xt);

  const std::vector<float> want = {0.0f, 0.0f, 0.0f, 0.0f, 0.25f, 1.0f, 4.0f, 9.0f};
  for (size_t i = 0; i < want.size(); ++i) {
    CHECK(out[i] == want[i]);
  }
  // The negative half is exactly ZERO, not a small negative (a `x*|x|` or
  // `x*x*sign(x)` mis-port would keep the sign).
  CHECK(std::signbit(out[0]) == false);
}

// The two mis-ports a token gate would catch late: relu (forgot the square) and
// silu (took the GATED family's activation). Pinned at a single input where all
// three differ by a wide margin.
TEST_CASE("moe relu2: is neither relu nor silu") {
  std::vector<float> x = {2.0f, 3.0f, -1.5f};
  std::vector<float> out(x.size(), -1.0f);
  Tensor xt = F32_1(x);
  Tensor ot = F32_1(out);
  Queue q = Q();
  vt::MoeRelu2(q, ot, xt);

  CHECK(out[0] == 4.0f);   // relu(2)^2
  CHECK(out[0] != 2.0f);   // NOT relu(2)
  CHECK(!Close(out[0], SiluRef(2.0f)));  // NOT silu(2) == 1.7615942
  CHECK(out[1] == 9.0f);   // relu(3)^2
  CHECK(out[1] != 3.0f);   // NOT relu(3)
  CHECK(!Close(out[1], SiluRef(3.0f)));  // NOT silu(3) == 2.8577223
  CHECK(out[2] == 0.0f);   // relu(-1.5)^2
  CHECK(!Close(out[2], SiluRef(-1.5f)));  // silu(-1.5) == -0.27440965, NOT zero
}

// The dtype/rounding order of relu_squared_kernel (activation_kernels.cu:673-678):
// the square is computed in FP32 and stored ONCE. bf16 in, f32 out is the arm
// that catches a kernel narrowing the product back through bf16 before the store
// — the exact defect a bf16-out-only test absorbs (bf16 rounds it away).
//   x    = 1 + 1/128 = 1.0078125            (exactly representable in bf16)
//   x^2  = 16641/16384 = 1.01568603515625   (exact in f32)
//   bf16(x^2) = 1 + 2/128 = 1.015625        (what a narrowed square would store)
TEST_CASE("moe relu2: bf16 in / f32 out keeps the square in f32 (no narrowing)") {
  const float x0 = 1.0078125f;
  std::vector<uint16_t> x = {vt::F32ToBF16(x0)};
  REQUIRE(vt::BF16ToF32(x[0]) == x0);  // the input itself is exact in bf16
  std::vector<float> out(1, -1.0f);
  Tensor xt = Bf16_1(x);
  Tensor ot = F32_1(out);
  Queue q = Q();
  vt::MoeRelu2(q, ot, xt);

  CHECK(out[0] == x0 * x0);              // 1.01568603515625, the f32 square
  CHECK(out[0] != vt::BF16ToF32(vt::F32ToBF16(x0 * x0)));  // NOT the narrowed 1.015625
}

// The bf16 STORE arm: exactly one round-to-nearest-even of the f32 square. Raw
// bit comparison, not a tolerance — a second rounding step is invisible to any
// bf16 tolerance wide enough to be meaningful.
TEST_CASE("moe relu2: bf16 out rounds the f32 square exactly once") {
  const float x0 = 1.0078125f;
  std::vector<uint16_t> x = {vt::F32ToBF16(x0), vt::F32ToBF16(-2.5f)};
  std::vector<uint16_t> out(2, 0xFFFF);
  Tensor xt = Bf16_1(x);
  Tensor ot = Bf16_1(out);
  Queue q = Q();
  vt::MoeRelu2(q, ot, xt);

  CHECK(out[0] == vt::F32ToBF16(x0 * x0));
  CHECK(out[1] == vt::F32ToBF16(0.0f));
}

TEST_CASE("moe relu2: rejects a shape/dtype/device contract violation") {
  std::vector<float> x(8, 1.0f);
  std::vector<float> small(4, 0.0f);
  Tensor xt = F32_1(x);
  Tensor st = F32_1(small);
  Queue q = Q();
  CHECK_THROWS_AS(vt::MoeRelu2(q, st, xt), std::runtime_error);

  std::vector<int32_t> ints(8, 0);
  Tensor it = Tensor::Contiguous(ints.data(), DType::kI32, Cpu(), {8});
  CHECK_THROWS_AS(vt::MoeRelu2(q, it, xt), std::runtime_error);

  // The DEVICE half of the contract, which the name promises. Runnable without
  // a GPU: `vt::MoeRelu2` validates `out.device == q.device && x.device ==
  // q.device` BEFORE `GetOp` dispatches, so a tensor merely LABELLED kCUDA is
  // rejected on the host and its data pointer is never dereferenced. Both
  // operand positions are checked — a wrapper that validated only `out` would
  // pass the first and fail the second.
  std::vector<float> host(8, 1.0f);
  Tensor cuda_out = Tensor::Contiguous(host.data(), DType::kF32, Device{DeviceType::kCUDA, 0}, {8});
  Tensor cuda_x = Tensor::Contiguous(host.data(), DType::kF32, Device{DeviceType::kCUDA, 0}, {8});
  std::vector<float> out8(8, 0.0f);
  Tensor ot8 = F32_1(out8);
  CHECK_THROWS_AS(vt::MoeRelu2(q, cuda_out, xt), std::runtime_error);   // out on the wrong device
  CHECK_THROWS_AS(vt::MoeRelu2(q, ot8, cuda_x), std::runtime_error);    // x on the wrong device
}

// ---------------------------------------------------------------------------
// 2. routed_scaling_factor on the OUTPUT (apply_routed_scale_to_output=True).
// ---------------------------------------------------------------------------

// moe_runner.py:402-406 scales `fused_output` and leaves `shared_output` alone;
// :722-725 then adds them. So the combine is
//     out = routed_scale * sum_j w[t,j]*expert_out[t,j] + shared[t]
// and NOT routed_scale * (routed + shared), and NOT (routed + shared).
TEST_CASE("moe combine: routed_scale multiplies the ROUTED sum, not the shared term") {
  const int64_t T = 2, K = 2, H = 3;
  std::vector<float> expert_out = {
      // t=0
      1.0f, 2.0f, 3.0f,    // slot 0
      -1.0f, 0.5f, 4.0f,   // slot 1
      // t=1
      2.0f, -2.0f, 1.0f,   // slot 0
      0.25f, 1.5f, -3.0f,  // slot 1
  };
  std::vector<float> weights = {0.75f, 0.25f, 0.5f, 0.5f};
  std::vector<float> shared = {10.0f, 20.0f, 30.0f, -1.0f, -2.0f, -3.0f};
  std::vector<float> out(static_cast<size_t>(T * H), 0.0f);

  Tensor eo = F32_3(expert_out, T, K, H);
  Tensor wt = F32_2(weights, T, K);
  Tensor sh = F32_2(shared, T, H);
  Tensor ot = F32_2(out, T, H);
  Queue q = Q();
  const float scale = 2.5f;
  vt::MoeCombine(q, ot, eo, wt, &sh, scale);

  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < H; ++h) {
      float routed = 0.0f;
      for (int64_t j = 0; j < K; ++j) {
        routed += weights[static_cast<size_t>(t * K + j)] *
                  expert_out[static_cast<size_t>((t * K + j) * H + h)];
      }
      const float sv = shared[static_cast<size_t>(t * H + h)];
      const size_t i = static_cast<size_t>(t * H + h);
      CHECK(Close(out[i], scale * routed + sv));
      // The two misplacements: scaling the shared term too, and dropping the
      // scale entirely. Both are wrong by a wide margin on every element here.
      CHECK(!Close(out[i], scale * (routed + sv)));
      CHECK(!Close(out[i], routed + sv));
    }
  }
}

// The mis-port the case above CANNOT see, and the one most likely to be made:
// folding the factor into each router weight — `acc += (scale*w[j])*eo[j]` —
// instead of scaling the assembled sum — `acc = scale * sum_j w[j]*eo[j]`.
// The two agree in exact arithmetic, so every wide-margin check above passes
// under the fold; they differ in f32 because the fold rounds K times INSIDE the
// reduction instead of once at the end.
//
// Upstream scales the ASSEMBLED tensor: `_maybe_apply_routed_scale_to_output`
// (moe_runner.py:402-404) runs `fused_output *= self.routed_scaling_factor` on
// the finished `fused_output` — one multiply per output element, AFTER the
// K-term reduction that produced it. Laguna folds the same factor into the
// router weights by linearity (`laguna_ops.h:48`) and is entitled to; NemotronH
// takes the literal upstream form, so the fold is a defect here.
//
// The comparison is bitwise and portable: the project pins -ffp-contract=off
// for every C++ TU (CMakeLists.txt:42-56), so both expressions below are IEEE
// as-written, with no FMA contraction to make the reference disagree with the
// kernel. The REQUIRE is what stops this being a vacuous green — it proves the
// chosen data actually SEPARATES the two forms before the CHECKs assert which
// one we implement.
TEST_CASE("moe combine: routed_scale scales the ASSEMBLED sum, not each router weight") {
  const int64_t T = 2, K = 4, H = 1;
  const float scale = 2.5f;
  // Decimal-grid values: their f32 products carry full mantissas, so the K-term
  // reduction genuinely rounds and the two orderings separate (row 0 by 10 ULP,
  // row 1 by 4 ULP). A dyadic grid like 1/8 makes every product exact and both
  // forms bit-identical — which is exactly why the wide-margin case above, on
  // such data, cannot catch this.
  std::vector<float> expert_out = {2.2f,  -1.15f, -1.18f, -1.52f,   // t=0, slots 0..3
                                   2.72f, -2.55f, 2.06f,  -2.21f};  // t=1, slots 0..3
  std::vector<float> weights = {0.95f, 0.32f, 0.37f, 0.72f,
                                0.80f, 0.45f, 0.86f, 0.59f};
  std::vector<float> out(static_cast<size_t>(T * H), 0.0f);

  Tensor eo = F32_3(expert_out, T, K, H);
  Tensor wt = F32_2(weights, T, K);
  Tensor ot = F32_2(out, T, H);
  Queue q = Q();
  // No shared term: this case isolates the placement of the scale WITHIN the
  // routed reduction. The shared-term placement is the case above.
  vt::MoeCombine(q, ot, eo, wt, nullptr, scale);

  for (int64_t t = 0; t < T; ++t) {
    CAPTURE(t);
    // Upstream form: reduce the K terms, THEN scale once.
    float acc = 0.0f;
    for (int64_t j = 0; j < K; ++j) {
      acc += weights[static_cast<size_t>(t * K + j)] *
             expert_out[static_cast<size_t>(t * K + j)];
    }
    const float want = acc * scale;
    // Folded form: scale each router weight, then reduce.
    float folded = 0.0f;
    for (int64_t j = 0; j < K; ++j) {
      folded += (scale * weights[static_cast<size_t>(t * K + j)]) *
                expert_out[static_cast<size_t>(t * K + j)];
    }
    // The data separates the two forms — without this the CHECKs below prove
    // nothing.
    REQUIRE(Bits(want) != Bits(folded));
    CHECK(Bits(out[static_cast<size_t>(t)]) == Bits(want));
    CHECK(Bits(out[static_cast<size_t>(t)]) != Bits(folded));
  }
}

// The default keeps every landed caller byte-identical: the 5-argument call with
// routed_scale == 1.0f must produce the same bits as the 4-argument call.
TEST_CASE("moe combine: routed_scale defaults to 1.0 (landed callers unchanged)") {
  const int64_t T = 2, K = 3, H = 4;
  std::vector<float> expert_out(static_cast<size_t>(T * K * H));
  for (size_t i = 0; i < expert_out.size(); ++i) {
    expert_out[i] = 0.125f * static_cast<float>(i) - 1.5f;
  }
  std::vector<float> weights = {0.5f, 0.25f, 0.25f, 0.125f, 0.375f, 0.5f};
  std::vector<float> shared(static_cast<size_t>(T * H));
  for (size_t i = 0; i < shared.size(); ++i) shared[i] = 0.5f - 0.25f * static_cast<float>(i);

  std::vector<float> a(static_cast<size_t>(T * H), 0.0f);
  std::vector<float> b(static_cast<size_t>(T * H), 0.0f);
  Tensor eo = F32_3(expert_out, T, K, H);
  Tensor wt = F32_2(weights, T, K);
  Tensor sh = F32_2(shared, T, H);
  Tensor at = F32_2(a, T, H);
  Tensor bt = F32_2(b, T, H);
  Queue q = Q();
  vt::MoeCombine(q, at, eo, wt, &sh);            // landed 4-arg form
  vt::MoeCombine(q, bt, eo, wt, &sh, 1.0f);      // explicit no-op scale
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// The ONE arm of `_maybe_apply_routed_scale_to_output` this port does not
// mirror: moe_runner.py:403-406 keys on `fused_output.dtype == torch.float16`
// and, when a `shared_output` exists, divides the SHARED term by the factor
// instead of multiplying the fused one (fp16 overflow protection; the decoder
// layer compensates). It is not mirrored because it is UNREACHABLE, not because
// it was missed: the analogue of `fused_output` here is `out`, and MoeCombine
// gates `out.dtype` through `IsOutFloat` (src/vt/ops.cpp:22), which admits f32
// and bf16 ONLY. This case pins that gate, so the "unreachable" claim next to
// the kernel cannot quietly become false if `IsOutFloat` is ever widened to
// kF16 — that widening makes an unmirrored upstream branch reachable and must
// fail here rather than in a token gate.
TEST_CASE("moe combine: f16 out is refused, which is why upstream's fp16 arm is unreachable") {
  const int64_t T = 1, K = 2, H = 2;
  std::vector<float> expert_out = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> weights = {0.5f, 0.5f};
  std::vector<uint16_t> out_f16(static_cast<size_t>(T * H), 0);
  std::vector<float> shared(static_cast<size_t>(T * H), 1.0f);

  Tensor eo = F32_3(expert_out, T, K, H);
  Tensor wt = F32_2(weights, T, K);
  Tensor sh = F32_2(shared, T, H);
  Tensor ot = Tensor::Contiguous(out_f16.data(), DType::kF16, Cpu(), {T, H});
  Queue q = Q();
  CHECK_THROWS_AS(vt::MoeCombine(q, ot, eo, wt, &sh, 2.5f), std::runtime_error);
  // ... and with no shared term either: the refusal is on the OUT dtype, which
  // is what makes the fp16 branch unreachable regardless of `shared`.
  CHECK_THROWS_AS(vt::MoeCombine(q, ot, eo, wt, nullptr, 2.5f), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 3. The whole non-gated expert, end to end on the shared ops.
// ---------------------------------------------------------------------------

namespace {

// An independent scalar reference for ONE NemotronH MoE block. Deliberately
// written from the upstream formula rather than from any vt op, so it cannot
// agree with the implementation by sharing a helper.
//   h = x @ W_up[e]^T ; h = relu(h)^2 ; y = h @ W_down[e]^T
//   out = routed_scale * sum_j w[t,j] * y[t,j] + shared[t]
std::vector<float> NonGatedExpertRef(const std::vector<float>& x, int64_t T, int64_t H,
                                     int64_t I, int64_t E, const std::vector<float>& w_up,
                                     const std::vector<float>& w_down,
                                     const std::vector<int32_t>& ids,
                                     const std::vector<float>& weights, int64_t K,
                                     const std::vector<float>& shared, float routed_scale) {
  std::vector<float> out(static_cast<size_t>(T * H), 0.0f);
  (void)E;
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> acc(static_cast<size_t>(H), 0.0f);
    for (int64_t j = 0; j < K; ++j) {
      const int64_t e = ids[static_cast<size_t>(t * K + j)];
      std::vector<float> hbuf(static_cast<size_t>(I), 0.0f);
      for (int64_t i = 0; i < I; ++i) {
        float s = 0.0f;
        for (int64_t k = 0; k < H; ++k) {
          s += x[static_cast<size_t>(t * H + k)] *
               w_up[static_cast<size_t>((e * I + i) * H + k)];
        }
        const float r = s > 0.0f ? s : 0.0f;
        hbuf[static_cast<size_t>(i)] = r * r;  // relu^2, NOT relu, NOT silu
      }
      for (int64_t h = 0; h < H; ++h) {
        float s = 0.0f;
        for (int64_t i = 0; i < I; ++i) {
          s += hbuf[static_cast<size_t>(i)] *
               w_down[static_cast<size_t>((e * H + h) * I + i)];
        }
        acc[static_cast<size_t>(h)] += weights[static_cast<size_t>(t * K + j)] * s;
      }
    }
    for (int64_t h = 0; h < H; ++h) {
      out[static_cast<size_t>(t * H + h)] =
          routed_scale * acc[static_cast<size_t>(h)] + shared[static_cast<size_t>(t * H + h)];
    }
  }
  return out;
}

float Synth(int64_t a, int64_t b, float k) {
  return std::sin(static_cast<float>(a) * 0.7f + static_cast<float>(b) * 0.13f) * k;
}

}  // namespace

// The W2 deliverable: the expert has NO gate half, so it is the existing SINGLE
// grouped projection plus the relu^2 activation — up_proj -> relu^2 -> down_proj
// -> weighted combine with the routed scale on the OUTPUT. Every step is a
// shared vt:: op; nothing here is a NemotronH-specific MoE path.
TEST_CASE("nemotron-h non-gated expert: up -> relu^2 -> down -> scaled combine") {
  const int64_t T = 3, H = 6, I = 4, E = 4, K = 2;
  const float routed_scale = 2.5f;  // config.routed_scaling_factor

  std::vector<float> x(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < H; ++h) x[static_cast<size_t>(t * H + h)] = Synth(t, h, 0.9f);
  }
  // up_proj weight [E, I, H] and down_proj weight [E, H, I] — torch Linear
  // (out, in) orientation, which is exactly vt::MatmulBT's `b [N, K]`.
  std::vector<float> w_up(static_cast<size_t>(E * I * H));
  for (int64_t e = 0; e < E; ++e) {
    for (int64_t i = 0; i < I; ++i) {
      for (int64_t h = 0; h < H; ++h) {
        w_up[static_cast<size_t>((e * I + i) * H + h)] = Synth(e * I + i, h, 0.4f);
      }
    }
  }
  std::vector<float> w_down(static_cast<size_t>(E * H * I));
  for (int64_t e = 0; e < E; ++e) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t i = 0; i < I; ++i) {
        w_down[static_cast<size_t>((e * H + h) * I + i)] = Synth(e * H + h + 3, i, 0.3f);
      }
    }
  }
  const std::vector<int32_t> ids = {0, 1, 2, 3, 1, 0};
  const std::vector<float> weights = {0.6f, 0.4f, 0.7f, 0.3f, 0.5f, 0.5f};
  std::vector<float> shared(static_cast<size_t>(T * H));
  for (size_t i = 0; i < shared.size(); ++i) shared[i] = 0.05f * static_cast<float>(i) - 0.2f;

  Queue q = Q();
  // Per (token, slot) expert projection through the shared GEMM op, exactly as
  // the CPU/GGUF MoE reference loop does for the gated archs. The ONLY structural
  // difference from a SwiGLU expert is that there is one projection, not a merged
  // pair, and the epilogue is relu^2 instead of silu*up.
  std::vector<float> expert_out(static_cast<size_t>(T * K * H), 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < K; ++j) {
      const int64_t e = ids[static_cast<size_t>(t * K + j)];
      Tensor xt = Tensor::Contiguous(&x[static_cast<size_t>(t * H)], DType::kF32, Cpu(), {1, H});
      std::vector<float> hbuf(static_cast<size_t>(I), 0.0f);
      Tensor ht = F32_2(hbuf, 1, I);
      Tensor wu = Tensor::Contiguous(&w_up[static_cast<size_t>(e * I * H)], DType::kF32, Cpu(),
                                     {I, H});
      vt::MatmulBT(q, ht, xt, wu);
      std::vector<float> act(static_cast<size_t>(I), 0.0f);
      Tensor at = F32_2(act, 1, I);
      vt::MoeRelu2(q, at, ht);
      Tensor yt = Tensor::Contiguous(&expert_out[static_cast<size_t>((t * K + j) * H)],
                                     DType::kF32, Cpu(), {1, H});
      Tensor wd = Tensor::Contiguous(&w_down[static_cast<size_t>(e * H * I)], DType::kF32, Cpu(),
                                     {H, I});
      vt::MatmulBT(q, yt, at, wd);
    }
  }

  std::vector<float> out(static_cast<size_t>(T * H), 0.0f);
  Tensor eo = F32_3(expert_out, T, K, H);
  std::vector<float> wcopy = weights;
  Tensor wt = F32_2(wcopy, T, K);
  Tensor sh = F32_2(shared, T, H);
  Tensor ot = F32_2(out, T, H);
  vt::MoeCombine(q, ot, eo, wt, &sh, routed_scale);

  const std::vector<float> want =
      NonGatedExpertRef(x, T, H, I, E, w_up, w_down, ids, weights, K, shared, routed_scale);
  for (size_t i = 0; i < want.size(); ++i) {
    CHECK(Close(out[i], want[i], 1e-5f, 1e-6f));
  }

  // The same composite with the GATED family's activation is a DIFFERENT answer:
  // proves the block is genuinely sensitive to the activation choice and is not
  // dominated by the combine.
  const std::vector<float> silu_ref = [&] {
    std::vector<float> o(static_cast<size_t>(T * H), 0.0f);
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t j = 0; j < K; ++j) {
        const int64_t e = ids[static_cast<size_t>(t * K + j)];
        std::vector<float> hb(static_cast<size_t>(I), 0.0f);
        for (int64_t i = 0; i < I; ++i) {
          float s = 0.0f;
          for (int64_t k = 0; k < H; ++k) {
            s += x[static_cast<size_t>(t * H + k)] *
                 w_up[static_cast<size_t>((e * I + i) * H + k)];
          }
          hb[static_cast<size_t>(i)] = SiluRef(s);
        }
        for (int64_t h = 0; h < H; ++h) {
          float s = 0.0f;
          for (int64_t i = 0; i < I; ++i) {
            s += hb[static_cast<size_t>(i)] *
                 w_down[static_cast<size_t>((e * H + h) * I + i)];
          }
          o[static_cast<size_t>(t * H + h)] += weights[static_cast<size_t>(t * K + j)] * s;
        }
      }
    }
    for (size_t i = 0; i < o.size(); ++i) o[i] = routed_scale * o[i] + shared[i];
    return o;
  }();
  bool any_differs = false;
  for (size_t i = 0; i < want.size(); ++i) {
    if (!Close(out[i], silu_ref[i], 1e-3f, 1e-6f)) any_differs = true;
  }
  CHECK(any_differs);
}

// The scale lives on the OUTPUT, so the ROUTER runs with routed_scaling_factor
// 1.0 (layer.py:291-300) and its weights are the plain renormalized sigmoid
// scores. Scaling the LOGITS instead is a different answer entirely — sigmoid is
// non-linear, so it moves the weights (and can move the SELECTION).
TEST_CASE("nemotron-h routed scale: on the output, not on the router logits") {
  const int64_t T = 1, E = 6, K = 3;
  std::vector<float> logits = {0.2f, -0.4f, 1.1f, 0.05f, -1.3f, 0.7f};
  std::vector<float> scaled_logits(logits.size());
  const float routed_scale = 2.5f;
  for (size_t i = 0; i < logits.size(); ++i) scaled_logits[i] = routed_scale * logits[i];

  vt::MoeRouterTopKArgs args;
  args.top_k = static_cast<int>(K);
  args.renormalize = true;  // config.norm_topk_prob
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;
  args.num_expert_group = 1;  // config.n_group
  args.topk_group = 1;        // config.topk_group
  // apply_routed_scale_to_output=True => the ROUTER's factor is a nop (1.0).
  args.routed_scaling_factor = 1.0f;
  std::vector<float> bias(static_cast<size_t>(E), 0.0f);
  bias[2] = -5.0f;  // e_score_correction_bias: biases SELECTION only

  std::vector<float> w(static_cast<size_t>(T * K), 0.0f);
  std::vector<int32_t> ids(static_cast<size_t>(T * K), -1);
  Tensor lt = F32_2(logits, T, E);
  Tensor wt = F32_2(w, T, K);
  Tensor it = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {T, K});
  Tensor bt = F32_1(bias);
  Queue q = Q();
  vt::MoeRouterTopK(q, wt, it, lt, args, &bt);

  // Renormalized => the weights sum to 1: the routed scale is NOT in them.
  float sum = 0.0f;
  for (float v : w) sum += v;
  CHECK(Close(sum, 1.0f, 1e-6f));
  for (float v : w) CHECK(v <= 1.0f);

  // Scaling the logits produces DIFFERENT weights, so a mis-port that folds the
  // scale into the router input cannot pass the block-level gate above.
  std::vector<float> w2(static_cast<size_t>(T * K), 0.0f);
  std::vector<int32_t> ids2(static_cast<size_t>(T * K), -1);
  Tensor lt2 = F32_2(scaled_logits, T, E);
  Tensor wt2 = F32_2(w2, T, K);
  Tensor it2 = Tensor::Contiguous(ids2.data(), DType::kI32, Cpu(), {T, K});
  vt::MoeRouterTopK(q, wt2, it2, lt2, args, &bt);
  bool weights_differ = false;
  for (size_t i = 0; i < w.size(); ++i) {
    if (!Close(w[i], w2[i], 1e-4f, 1e-6f)) weights_differ = true;
  }
  CHECK(weights_differ);
}

// ---------------------------------------------------------------------------
// 4. The NVFP4 W4A16 arm's group size (spec §6 named risk).
// ---------------------------------------------------------------------------

// NemotronH's routed experts are W4A16_NVFP4 with group_size=16. The grouped
// Marlin path takes the group size as an explicit argument whose DEFAULT is the
// NVFP4 one, so 16 is the configuration it already runs (the alternative, 32,
// only comes with mxfp4=true). Pinned here so a later widening of the default
// cannot silently re-point NemotronH's experts at a group size the checkpoint
// does not carry.
TEST_CASE("nvfp4 grouped moe: group_size 16 is the NVFP4 default NemotronH needs") {
  vt::MoeMarlinArgs args;
  CHECK(args.group_size == 16);
  CHECK(args.mxfp4 == false);
  vt::MarlinDenseArgs dense;
  CHECK(dense.group_size == 16);
  CHECK(dense.mxfp4 == false);
}
