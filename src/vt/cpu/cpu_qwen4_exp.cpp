// CPU kernels for the Qwen4-Exp (`Qwen3.8-Flash-Next`) gated-residual
// hyper-connection stream — `vt::Qwen4ExpGatedResidual` and
// `vt::Qwen4ExpGatedResidualWriteBack`. Row MODEL-MM-QWEN4-EXP W5b (#2031),
// spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// This row splits its oracles by developer direction (spec `## Oracles`):
// transformers supplies the ALGORITHM, vLLM supplies the OP FORM. vLLM has never
// registered `qwen4_exp` at any revision, so there is no vLLM kernel to mirror
// for the gated residual itself; the grouped RMS norm at its front IS a vLLM op
// and is mirrored as one.
//
//   ALGORITHM  transformers v5.16.0 (the lane pin),
//              `models/qwen4_exp/modeling_qwen4_exp.py`
//                ::Qwen4ExpTextRMSNorm            (:158-181)
//                ::Qwen4ExpTextGatedResidual      (:941-969)
//                ::Qwen4ExpTextDecoderLayer.forward, the two write-back lines
//   OP FORM    vLLM @ origin/main 6a5e8f5979,
//              `model_executor/layers/layernorm.py::RMSNormGated` (:172),
//              `group_size` (:187), grouped branch (:243-244, :258-264)
//
// The landed HOST reference for the same arithmetic is
// `src/vllm/model_executor/models/qwen4_exp_hc.{h,cpp}` (W3, #1988), gated
// against goldens dumped by EXECUTING the two pinned oracle classes. These
// kernels are gated against THE SAME GOLDENS rather than against that reference,
// so the two arms answer to one oracle instead of to each other, and they are
// additionally required to agree with it at the model's real width
// (`tests/vllm/models/test_qwen4_exp_hc_device.cpp`).
//
// ─── DTYPE CONTRACT, AND ONE DELIBERATE DIVERGENCE FROM UPSTREAM ──────────────
// Widen on load, compute in f32, round ONCE on the store. That is this tree's
// house contract for every norm and activation — `vt::RmsNorm` states it in
// terms beside its own declaration ("the standard path keeps full f32 precision
// (no x.to(weight.dtype) rounding before the weight multiply); parity tests vs
// upstream bf16 need bf16-eps tolerance on the non-gemma path") and
// `vt::MoeRelu2` repeats it.
//
// Upstream narrows in the middle: `Qwen4ExpTextRMSNorm.forward` is
// `self._norm(x.float()) * (1.0 + weight.float())` followed by `.type_as(x)`, so
// on a bf16 stream the NORMED value is rounded to bf16 before the down
// projection and before the `mixed_input` product. This kernel does NOT
// reproduce that intermediate rounding, for the same reason `vt::RmsNorm` does
// not: consistency with every other norm in the tree beats a per-op exception,
// and the divergence is one rounding of one intermediate, in the direction of
// MORE precision. It is recorded in the spec's `## Owed` rather than left to be
// discovered, because it means a bf16 parity comparison against a running
// oracle carries a bf16-eps term that an f32 one does not. The f32 arm — the one
// the goldens are dumped at — is unaffected, and is what the gate measures.
//
// ─── PRECISION ────────────────────────────────────────────────────────────────
// The per-group sum of squares accumulates in DOUBLE, matching the host
// reference and the `deepseek_v4_mhc.cpp` house convention. It is not free
// accuracy nobody can see: at the model's real group size of 2560, on
// magnitude-separated data, a float accumulator and this one differ by 742x
// (`test_qwen4_exp_hc.cpp` measures 3.168e-06 against 2.352e-03). Every other
// reduction here — the three projections and the mean over branches — is f32, in
// the host reference's order, because that is the order the goldens were dumped
// in and a wider accumulator there would make the two arms disagree for a reason
// no oracle authorises.
//
// A CUDA ARM INHERITS A DECISION FROM THIS, and does not yet exist: a straight
// f32-accumulate block reduction will not meet the same bound on the norm, so
// that kernel must either accumulate wider than f32 or be gated against the
// oracle directly. Deciding which belongs to the wave that writes it. Nothing
// here is registered for any device but kCPU, so the dispatcher refuses by name
// on every other one rather than silently falling back.
#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

namespace vt::cpu {
namespace {

// Local dtype accessors, the `cpu_layernorm.cpp` arrangement. `cpu_ops.cpp`'s
// `LoadF32`/`StoreF32` are file-static there and hoisting them would edit a
// 3900-line translation unit several other rows are working in, which is the
// shared-file lock AGENTS.md "Records" names; a third copy of twelve lines is
// the cheaper of the two, and it is what every other cpu_*.cpp here already does.
float LoadF32At(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "qwen4_exp_gated_residual: unsupported input dtype"); return 0.0f;
  }
}

void StoreF32At(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kF16: t.Ptr<uint16_t>()[i] = F32ToF16(v); break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "qwen4_exp_gated_residual: unsupported output dtype");
  }
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y[o] = sum_i w[o*K + i] * x[i]. PyTorch `nn.Linear(bias=False)` weight layout,
// `(out_features, in_features)` row-major, accumulated in f32 in index order —
// the host reference's `LinearNoBias`, term for term.
void LinearNoBias(const Tensor& w, const float* x, int64_t out_dim, int64_t in_dim,
                  float* y) {
  for (int64_t o = 0; o < out_dim; ++o) {
    const int64_t row = o * in_dim;
    float acc = 0.0f;
    for (int64_t i = 0; i < in_dim; ++i) acc += LoadF32At(w, row + i) * x[i];
    y[o] = acc;
  }
}

void Qwen4ExpGatedResidualKernel(Queue&, Tensor& mixed, Tensor* injection,
                                 const Tensor& hyper, const Tensor& hc_norm_w,
                                 const Tensor& mix_down, const Tensor& mix_up,
                                 const Tensor* block_inject,
                                 const Qwen4ExpGatedResidualArgs& args) {
  const int64_t hc = args.hc_count;
  const int64_t H = args.hidden_size;
  const int64_t R = args.lowrank;
  const int64_t flat = hc * H;
  const int64_t T = hyper.shape[0];
  const float eps = args.eps;
  // A TRUE division, never a reciprocal multiply. Upstream spells
  // `/ self.hc_count` and 1/hc is inexact for any hc that is not a power of two,
  // so the shortcut would put a one-ulp wedge between this kernel and the oracle
  // at hc_count = 3 — which is exactly golden case B.
  const float hc_f = static_cast<float>(hc);

  std::vector<float> normed(static_cast<size_t>(flat));
  std::vector<float> low(static_cast<size_t>(R));
  std::vector<float> gate(static_cast<size_t>(flat));

  for (int64_t t = 0; t < T; ++t) {
    const int64_t base = t * flat;

    // ── grouped RMS norm, group_size == hidden_size ──────────────────────────
    // `x.reshape(*x.shape[:-1], -1, group_size)` then a mean over the LAST axis
    // (modeling_qwen4_exp.py:169-171): hc INDEPENDENT reductions of H elements,
    // which is what `RMSNormGated(group_size=)` buys and what the plain
    // `RMSNorm` — whose only related knob is a PREFIX reduction — cannot express.
    for (int64_t j = 0; j < hc; ++j) {
      const int64_t g0 = j * H;
      double ss = 0.0;
      for (int64_t h = 0; h < H; ++h) {
        const double v = LoadF32At(hyper, base + g0 + h);
        ss += v * v;
      }
      // eps is INSIDE the rsqrt, added to the MEAN SQUARE, never to the norm.
      const float r =
          1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(H)) + eps);
      for (int64_t h = 0; h < H; ++h) {
        normed[static_cast<size_t>(g0 + h)] =
            LoadF32At(hyper, base + g0 + h) * r * LoadF32At(hc_norm_w, g0 + h);
      }
    }

    // ── DIVISION 1: inside the SiLU, on the [R] low-rank intermediate, BEFORE
    // the activation. `F.silu(down(x) / hc_count)`. SiLU is not homogeneous, so
    // `silu(a)/hc` is a different function and the placement is load-bearing.
    LinearNoBias(mix_down, normed.data(), R, flat, low.data());
    for (int64_t r = 0; r < R; ++r) {
      const float a = low[static_cast<size_t>(r)] / hc_f;
      low[static_cast<size_t>(r)] = a * Sigmoid(a);
    }

    // ── NO division on the up projection: `torch.sigmoid(up(...))`, full stop.
    LinearNoBias(mix_up, low.data(), flat, R, gate.data());
    for (int64_t p = 0; p < flat; ++p) {
      gate[static_cast<size_t>(p)] = Sigmoid(gate[static_cast<size_t>(p)]);
    }

    // ── `.unflatten(-1, (hc, H))`, multiplied against the NORMED stream — not
    // the raw one — then `.mean(dim=-2)`. A MEAN over the branches, never a sum.
    for (int64_t h = 0; h < H; ++h) {
      float acc = 0.0f;
      for (int64_t j = 0; j < hc; ++j) {
        const size_t p = static_cast<size_t>(j * H + h);
        acc += gate[p] * normed[p];
      }
      StoreF32At(mixed, t * H + h, acc / hc_f);
    }

    // `block_inject_weight is None` returns `mixed_input` alone (:966-967), and
    // that early return IS `Qwen4ExpTextModel`'s terminal `use_combine=False`
    // mixer. The dispatcher has already refused a half-specified pair.
    if (block_inject == nullptr) continue;

    // ── DIVISION 2: inside the injection sigmoid, whole sigmoid scaled by 2.
    // `2 * sigmoid(inject(x) / hc_count)`. Range (0, 2), exactly 1.0 at a zero
    // logit, so an untrained branch is the identity rather than a half-scale.
    for (int64_t j = 0; j < hc; ++j) {
      const int64_t row = j * flat;
      float acc = 0.0f;
      for (int64_t i = 0; i < flat; ++i) {
        acc += LoadF32At(*block_inject, row + i) * normed[static_cast<size_t>(i)];
      }
      StoreF32At(*injection, t * hc + j, 2.0f * Sigmoid(acc / hc_f));
    }
  }
}

// The two verbatim lines of `Qwen4ExpTextDecoderLayer.forward`:
//   injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
//   hidden_states = hyper_input + injection.flatten(-2)
// A RANK-1 UPDATE, and this loop is why it stays one. Both llama.cpp
// implementations of this architecture materialise it as `repeat_4d` + `mul` —
// a dense [H, hc, T] broadcast built and thrown away at 48 layers x 2 sites.
void Qwen4ExpGatedResidualWriteBackKernel(Queue&, Tensor& hyper, const Tensor& block_out,
                                          const Tensor& injection,
                                          const Qwen4ExpGatedResidualArgs& args) {
  const int64_t hc = args.hc_count;
  const int64_t H = args.hidden_size;
  const int64_t flat = hc * H;
  const int64_t T = hyper.shape[0];
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < hc; ++j) {
      const float w = LoadF32At(injection, t * hc + j);
      const int64_t row = t * flat + j * H;
      for (int64_t h = 0; h < H; ++h) {
        StoreF32At(hyper, row + h,
                   LoadF32At(hyper, row + h) + LoadF32At(block_out, t * H + h) * w);
      }
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpGatedResidual, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpGatedResidualFn>(&Qwen4ExpGatedResidualKernel)));
    RegisterOp(OpId::kQwen4ExpGatedResidualWriteBack, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Qwen4ExpGatedResidualWriteBackFn>(
                   &Qwen4ExpGatedResidualWriteBackKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
