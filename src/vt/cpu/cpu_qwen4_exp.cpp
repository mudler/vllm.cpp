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
// A CUDA ARM OF `vt::Qwen4ExpGatedResidual` INHERITS A DECISION FROM THIS, and
// still does not exist: a straight f32-accumulate block reduction will not meet
// the same bound on the norm, so that kernel must either accumulate wider than
// f32 or be gated against the oracle directly. Deciding which belongs to the
// wave that writes it, and the spec's `## Owed` carries it.
//
// `vt::Qwen4ExpGatedResidualWriteBack` IS DIFFERENT AND W6-CUDA GAVE IT A
// DEVICE ARM (`src/vt/cuda/cuda_qwen4_exp.cu`). It has no reduction — every
// output element is one multiply and one add — so there is no width to choose,
// and the device kernel spells the host's two roundings with
// `__fmul_rn`/`__fadd_rn` against this build's `-ffp-contract=off`, making it
// BYTE-IDENTICAL rather than merely close. Nothing else here is registered for
// any device but kCPU, so the dispatcher refuses those by name on every other
// one rather than silently falling back.
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

// ONE PROJECTION OF ONE TOKEN, and the operand's dtype picks the route (W5p,
// #2031).
//
// THE FUSION IS WHAT FORCED FLOAT, and this is the repair. `LoadF32At` above is
// a SCALAR ELEMENT walk: it can read f32, f16 and bf16 and nothing else, so
// fusing the whole mixer — grouped norm, down, SiLU, up, sigmoid, stream mean,
// inject — into one kernel made every weight in it an elementwise tensor by
// construction. llama.cpp keeps the three projections as separate graph nodes
// (`src/models/qwen4exp.cpp:237-241`), which is why a block-typed weight falls
// through its generic quant-aware `mul_mat` there and died here.
//
// SO THE PROJECTIONS ROUTE THROUGH THE SHARED SEAM INSTEAD OF A NEW ONE.
// `vt::MatmulBT` is `out[M,N] = a[M,K] @ b^T` with `b` in `[N,K]` row-major —
// the SAME orientation `LinearNoBias` walks, which is also ggml's src0 layout
// and GGUF's disk order, so keeping the blocks needs no transpose (a block row
// cannot be transposed without requantizing). It auto-dispatches a block dtype
// to `kMatmulBTQuant`, whose CPU kernel is the 1:1 port of
// `ggml_compute_forward_mul_mat`: quantize the activation once to the weight's
// `vec_dot_type`, then one integer block-dot per output.
//
// WHY THAT SEAM AND NOT A NEW OpId — THE ARM COUNT DECIDED IT, and it is a
// measurement rather than a taste. `vt::MatmulBT`/`vt::MatmulBTQuant` take
// `(Queue&, Tensor& out, const Tensor& a, const Tensor& b)` and NO args struct:
// ONE arm, whose behaviour is fully determined by the operands, so there is no
// field a quantized route could silently ignore. Contrast the two decisions this
// row already made on the same test: `kRmsNorm` has SIX arms, so a grouped norm
// riding on it would have had a silently-ignored field returning a wrong answer
// with no crash, and `vt::RmsNormGroup` took its own OpId; the paged QSA read's
// op has one arm, so it took an address mode instead. This is the one-arm case.
//
// THE FLOAT PATH IS UNTOUCHED, and deliberately so: it stays `LinearNoBias`,
// term for term in index order, so every golden case above is bit-identical by
// construction rather than by re-measurement.
//
// COST, STATED RATHER THAN LEFT TO BE FOUND. This is a per-TOKEN matvec (M = 1),
// so a prefill of T tokens makes T `kMatmulBTQuant` calls per projection where
// llama.cpp makes one GEMM over the whole batch. Correct, and the arm the
// released file needs to run at all; batching the projections over a token tile
// is recorded under the spec's `## Owed` rather than done here, because it moves
// the fused kernel's loop structure and owes its own red-first measurement.
void ProjectRow(Queue& q, const Tensor& w, const float* x, int64_t out_dim,
                int64_t in_dim, float* y) {
  if (IsBlockQuant(w.dtype)) {
    Tensor a = Tensor::Contiguous(const_cast<float*>(x), DType::kF32, w.device,
                                  {1, in_dim});
    Tensor o = Tensor::Contiguous(y, DType::kF32, w.device, {1, out_dim});
    MatmulBTQuant(q, o, a, w);
    return;
  }
  LinearNoBias(w, x, out_dim, in_dim, y);
}

void Qwen4ExpGatedResidualKernel(Queue& q, Tensor& mixed, Tensor* injection,
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
  // Only allocated when the injection arm is live (`use_combine=True`).
  std::vector<float> inject_pre(
      block_inject != nullptr ? static_cast<size_t>(hc) : 0);

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
      // THE `1 +` IS THE OP'S, AND IT IS NOT AN ALTERNATIVE PARAMETERIZATION.
      // `Qwen4ExpTextRMSNorm.forward` is `output * (1.0 + self.weight.float())`
      // over a ZERO-initialised gamma (modeling_qwen4_exp.py:173-178), and
      // `hc_norm_w` is that gamma — the raw HuggingFace parameter, exactly as
      // `Qwen4ExpGdnWeights`/`Qwen4ExpQsaWeights`/`Qwen4ExpPleWeights` carry
      // every other gamma of this architecture and exactly as
      // `vt::RmsNorm(gemma=true)` and `vt::Qwen4ExpQsaCompress` already read
      // them. This op used to demand the FOLDED form instead, which made it the
      // one consumer in the model disagreeing with the loader, and handing it
      // the loaded weight scaled every hyper-connection norm by a gamma centred
      // on zero. See #2218 and the composition case in
      // tests/vllm/models/test_qwen4_exp_forward.cpp, which is the only gate
      // that can see the disagreement: both halves are individually correct.
      //
      // THE FOLD IS f32, and that is upstream's width and not a convenience:
      // `output * (1.0 + self.weight.float())` (:177) folds a Python weak `1.0`
      // into an fp32 tensor, so the promotion stays fp32. Every host reference
      // that widens its reduction to double folds in `float` first for the same
      // reason (`test_qwen4_exp_hc_device.cpp`'s wide-accumulator case), so the
      // widening isolates the reduction rather than also moving the multiplier.
      for (int64_t h = 0; h < H; ++h) {
        normed[static_cast<size_t>(g0 + h)] =
            LoadF32At(hyper, base + g0 + h) * r *
            (1.0f + LoadF32At(hc_norm_w, g0 + h));
      }
    }

    // ── DIVISION 1: inside the SiLU, on the [R] low-rank intermediate, BEFORE
    // the activation. `F.silu(down(x) / hc_count)`. SiLU is not homogeneous, so
    // `silu(a)/hc` is a different function and the placement is load-bearing.
    ProjectRow(q, mix_down, normed.data(), R, flat, low.data());
    for (int64_t r = 0; r < R; ++r) {
      const float a = low[static_cast<size_t>(r)] / hc_f;
      low[static_cast<size_t>(r)] = a * Sigmoid(a);
    }

    // ── NO division on the up projection: `torch.sigmoid(up(...))`, full stop.
    ProjectRow(q, mix_up, low.data(), flat, R, gate.data());
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
    // The projection and its activation are SPLIT so the projection can take the
    // quantized route. `ProjectRow` on a float weight is `LinearNoBias`, which
    // accumulates `sum_i w[j*flat + i] * normed[i]` in the same index order the
    // loop that used to be here did — so the float arm is bit-identical.
    ProjectRow(q, *block_inject, normed.data(), hc, flat, inject_pre.data());
    for (int64_t j = 0; j < hc; ++j) {
      StoreF32At(*injection, t * hc + j,
                 2.0f * Sigmoid(inject_pre[static_cast<size_t>(j)] / hc_f));
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
