// CUDA kernel for the Qwen4-Exp (`Qwen3.8-Flash-Next`) gated-residual
// hyper-connection stream — `vt::Qwen4ExpGatedResidualWriteBack`.
// Row MODEL-MM-QWEN4-EXP W6-CUDA, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`src/vt/cuda/cuda_layernorm.cu`, `src/vt/cuda/cuda_conv1d_general.cu`): no
// existing kernel TU and no shared op array is edited. The file name mirrors its
// CPU sibling `src/vt/cpu/cpu_qwen4_exp.cpp` one for one, as CMakeLists.txt
// already pairs the three `cpu_qwen4_exp*.cpp` units.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// The CPU arm, `src/vt/cpu/cpu_qwen4_exp.cpp::Qwen4ExpGatedResidualWriteBackKernel`,
// which is itself the port of the two verbatim lines of
// `Qwen4ExpTextDecoderLayer.forward` (transformers v5.16.0,
// `models/qwen4_exp/modeling_qwen4_exp.py`):
//
//     injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
//     hidden_states = hyper_input + injection.flatten(-2)
//
// vLLM has never registered `qwen4_exp` at any revision, so there is no vLLM
// kernel to mirror; the KERNEL STRUCTURE (self-registering TU, `Check`,
// `AsStream`, `GridFor`, grid-stride loop, one `Check(cudaGetLastError(), ...)`
// per launch site) is mirrored from `src/vt/cuda/cuda_layernorm.cu` and
// `src/vt/cuda/cuda_ops.cu`'s `MulScalarKernel`, which is this tree's template
// for a small elementwise op.
//
// ─── WHY THIS OP IS IN THE FIRST CUDA TRANCHE ────────────────────────────────
// It performs NO reduction across a parallel axis. Every output element is one
// multiply and one add over operands it alone reads, so there is no reduction
// ORDER for a device arm to choose and the CPU arm's numeric contract transfers
// unchanged. The four ops this row leaves owed (`vt::Qwen4ExpGatedResidual`,
// `vt::RmsNormGroup`, and the two QSA ops) each own a reduction whose width or
// whose visit order is a decision no wave has made; see the spec's `## Owed`.
//
// ─── BYTE IDENTITY, AND THE INTRINSICS IT REQUIRES ───────────────────────────
// The host provider is pinned to `-ffp-contract=off` (CMakeLists.txt:41-56), so
// the CPU arm evaluates `hyper + block_out * w` as a SEPARATE round-to-nearest
// multiply and a round-to-nearest add. nvcc's `-fmad` defaults to ON and is not
// pinned, so `a + b * w` written plainly here would contract into a single
// fma and produce a DIFFERENT — and slightly more accurate — answer, which is
// exactly the byte-agreement break `src/vt/cuda/cuda_conv1d_general.cu:138-141`
// records for the same reason. `__fmul_rn`/`__fadd_rn` spell the host's two
// roundings out, so this arm is BYTE-IDENTICAL to the CPU arm rather than merely
// close, and its gate is a `memcmp` and not a tolerance. A tolerance would be
// the wrong instrument here: a transposed `injection` axis or a dropped `hc`
// stride lands well inside any epsilon anyone would write.
//
// ─── DTYPE ────────────────────────────────────────────────────────────────────
// The op contract (`src/vt/ops.cpp::Qwen4ExpGatedResidualWriteBack`) admits
// f32/f16/bf16 inputs and f32/bf16 outputs, and this arm implements ALL of them
// rather than narrowing to the f32+bf16 set the tree's other CUDA arms use: a
// device arm that refused a dtype its CPU sibling accepts would be a divergence
// to record, and the tag design below makes admitting it free.
//
// The dtype is carried as a RUNTIME TAG rather than a template parameter, which
// is a deliberate divergence from the `<Tin, Tout>` house style of
// `cuda_ops.cu`. Three operands with independent dtypes is 8 to 18
// instantiations of a kernel whose body is four lines, and — the load-bearing
// half — the tag is a KERNEL-WIDE SCALAR, identical for every thread, so the
// switch is a warp-UNIFORM branch and costs a prediction rather than divergence.
// The spelling also reads as the CPU arm's own `LoadF32At`/`StoreF32At` line for
// line, which is what makes the two arms diffable by eye. Templated
// specialisations are recorded in the spec's `## Owed` as a SPEED item; no speed
// claim on this row is admissible before the MoE adapter hoist in any case.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

// The runtime dtype tag described in the header. Local to this TU for the same
// reason `cpu_qwen4_exp.cpp` keeps a local copy of `LoadF32At`/`StoreF32At`:
// hoisting it would edit a translation unit several other rows are working in,
// which is the shared-file lock AGENTS.md "Records" names.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("cuda qwen4_exp_gated_residual_write_back: unsupported ") +
                          what + " dtype (f32/f16/bf16)");
      return DTag::kF32;
  }
}

// `__float2bfloat16` / `__float2half` are round-to-nearest-even, identical to
// the host `vt::F32ToBF16` / `vt::F32ToF16` — the equality this arm's byte
// gate depends on, and the same one `cuda_ops.cu:48-54` asserts.
__device__ inline float LoadAt(const void* p, DTag tag, int64_t i) {
  switch (tag) {
    case DTag::kF32: return static_cast<const float*>(p)[i];
    case DTag::kF16: return __half2float(static_cast<const __half*>(p)[i]);
    default: return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
  }
}

__device__ inline void StoreAt(void* p, DTag tag, int64_t i, float v) {
  switch (tag) {
    case DTag::kF32: static_cast<float*>(p)[i] = v; break;
    case DTag::kF16: static_cast<__half*>(p)[i] = __float2half(v); break;
    default: static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(v); break;
  }
}

// ---------------------------------------------------------------------------
// vt::Qwen4ExpGatedResidualWriteBack — the rank-1 update, IN PLACE on `hyper`.
//
//   hyper[t, j*H + h] += block_out[t, h] * injection[t, j]
//
// One thread per OUTPUT element. `hyper` is `[T, hc*H]` contiguous, so the flat
// grid index IS the `hyper` offset and the decomposition below is exact:
// `idx = t*(hc*H) + j*H + h`. Both llama.cpp implementations of this
// architecture materialise this as `repeat_4d` + `mul` — a dense `[H, hc, T]`
// broadcast built and thrown away 96 times a step (48 layers x 2 sites); the op
// exists so neither arm has to, and the device arm keeps that property.
__global__ void GatedResidualWriteBackKernel(void* hyper, DTag hyper_tag, const void* block_out,
                                             DTag block_tag, const void* injection, DTag inj_tag,
                                             int64_t hc, int64_t hidden, int64_t n) {
  const int64_t flat = hc * hidden;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t t = idx / flat;
    const int64_t rem = idx - t * flat;
    const int64_t j = rem / hidden;
    const int64_t h = rem - j * hidden;
    const float w = LoadAt(injection, inj_tag, t * hc + j);
    const float base = LoadAt(hyper, hyper_tag, idx);
    const float add = LoadAt(block_out, block_tag, t * hidden + h);
    // TWO roundings, never one fma — see the header. This is the whole reason
    // the gate can be a memcmp.
    StoreAt(hyper, hyper_tag, idx, __fadd_rn(base, __fmul_rn(add, w)));
  }
}

void Qwen4ExpGatedResidualWriteBackKernelCuda(Queue& q, Tensor& hyper, const Tensor& block_out,
                                              const Tensor& injection,
                                              const Qwen4ExpGatedResidualArgs& args) {
  // `args.lowrank` and `args.eps` are DELIBERATELY not read. The args struct is
  // shared with `vt::Qwen4ExpGatedResidual` so a caller cannot describe the same
  // stream two ways (`include/vt/ops.h`, the Qwen4ExpGatedResidualArgs comment),
  // and the CPU arm reads only these two fields as well. Reading either here
  // would be reading a field with no meaning at this op.
  const int64_t hc = args.hc_count;
  const int64_t hidden = args.hidden_size;
  const int64_t T = hyper.shape[0];
  const int64_t n = T * hc * hidden;
  if (n == 0) return;  // empty-work early return, before the launch
  const DTag hyper_tag = TagOf(hyper.dtype, "hyper");
  const DTag block_tag = TagOf(block_out.dtype, "block output");
  const DTag inj_tag = TagOf(injection.dtype, "injection");
  GatedResidualWriteBackKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      hyper.data, hyper_tag, block_out.data, block_tag, injection.data, inj_tag, hc, hidden, n);
  Check(cudaGetLastError(), "qwen4_exp_gated_residual_write_back launch");
}

// Registers the CUDA Qwen4-Exp gated-residual write-back during static init
// (pre-main, like cuda_layernorm.cu's Registrar). Filling the op table is
// harmless on a machine without a GPU: the kCUDA backend never registers there,
// so no CUDA queue can exist to dispatch with.
//
// UNCONDITIONAL, at `if(VLLM_CPP_CUDA)` depth exactly one in CMakeLists.txt and
// at preprocessor depth 0 here. `scripts/check-cuda-op-arch-gate.py` requires
// both: an op registered from a feature-gated TU disappears on an arch that does
// not build it, the resolver falls to the portable CPU tier, and a CPU kernel
// then dereferences device pointers (#960/#844).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpGatedResidualWriteBack, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Qwen4ExpGatedResidualWriteBackFn>(
                   &Qwen4ExpGatedResidualWriteBackKernelCuda)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cuda
