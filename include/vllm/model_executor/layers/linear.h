// LinearMethod — the per-(scheme, device) linear-apply seam.
//
// UPSTREAM (ported FROM, ground-every-impl rule):
//   vllm/model_executor/layers/linear.py:141-181  class LinearMethodBase
//                                                  (create_weights + apply)
//   vllm/model_executor/layers/linear.py:184-230  class UnquantizedLinearMethod
//                                                  (apply == F.linear(x, weight))
//
// A LinearMethod owns HOW one logical linear's weights are laid out and
// multiplied on the running device. The model forward calls `method.Apply(x)`
// and never asks which scheme (bf16 / nvfp4 / …) or which device — exactly as a
// vLLM model calls `self.qkv_proj(x)` and the bound LinearMethod does the rest.
// The concrete method is chosen ONCE (MakeQkvMethod/MakeMlpDownMethod/… in the
// scheme headers), from the checkpoint's populated weights, not per forward call
// by a tensor-name probe.
//
// The UNQUANTIZED (bf16) method here is scheme-neutral and device-neutral: it
// runs the same vt::MatmulBT the inline path did, on whatever device the queue
// names. Quantized methods (compressed_tensors NVFP4 W4A16, …) live in the
// scheme headers under quantization/, and ask the vt::OpProvider table
// (`vt::OpRegistered`) which kernel is available here rather than testing
// `device == kCUDA` — see base_config.h for the policy/implementation split.
#pragma once

#include "vllm/model_executor/layers/quantization/base_config.h"
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev, DBuf, ResidentWeight
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace layers {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::ResidentWeight;

// LinearMethodBase (linear.py:141). y[M,N] = x[M,K] @ W[N,K]^T for the bound
// weight; out_dtype selects the f32/bf16 result buffer. The N (output width) is
// taken from the weight, so a merged qkv_proj (N = q+2kv) or gate_up (N = 2I)
// flows through the SAME contract as a plain o_proj/down_proj — the caller
// splits/activates afterwards exactly as before.
class LinearMethodBase : public QuantizeMethodBase {
 public:
  virtual DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const = 0;
  // The bound scheme's name (mirrors QuantizationConfig.get_name) — a diagnostic
  // + test hook proving which method the factory selected.
  virtual const char* Name() const = 0;
};

// UnquantizedLinearMethod (linear.py:184). apply == F.linear: one vt::MatmulBT
// over the resident raw-NK bf16 weight — byte-for-byte the inline
// `ResidentWeight + MatmulBT` the dense forward carried at every projection.
// Device-neutral: MatmulBT runs on whatever device the op table realizes it
// for, so nothing here says `kCUDA`.
class UnquantizedLinearMethod : public LinearMethodBase {
 public:
  explicit UnquantizedLinearMethod(const OwnedTensor* weight) : w_(weight) {}

  DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const override {
    const int64_t M = x.shape[0];
    const int64_t N = w_->shape[0];  // raw-NK [N=out, K=in]
    vt::Tensor rw = ResidentWeight(d, *w_);
    DBuf out(d, out_dtype, {M, N});
    vt::MatmulBT(d.q, out.t(), x, rw);
    return out;
  }

  const char* Name() const override { return "bf16-unquantized"; }

 private:
  const OwnedTensor* w_;
};

// The SwiGLU merged gate_up + SiluAndMul as one method: returns silu(gate)·up as
// [M,I]. This is a distinct method from a plain linear because a quantized
// scheme may FUSE the gate_up GEMM with the activation (one Marlin GEMM over the
// N-concatenated pair, dense_nvfp4::GateUpFusedMarlinD), which a plain
// linear+SiluAndMul cannot express. Mirrors vLLM's MergedColumnParallelLinear
// (gate_up_proj) apply followed by SiluAndMul, kept together so the fused-kernel
// scheme choice has a home.
class MlpGateUpMethodBase : public QuantizeMethodBase {
 public:
  virtual DBuf Apply(Dev d, const vt::Tensor& x) const = 0;
  virtual const char* Name() const = 0;
};

// Unquantized (bf16) gate_up: one MatmulBT over the merged [2I,H] weight then
// vt::SiluAndMul — byte-for-byte the inline bf16 MLP path.
class UnquantizedMlpGateUpMethod : public MlpGateUpMethodBase {
 public:
  UnquantizedMlpGateUpMethod(const OwnedTensor* gate_up, int64_t intermediate)
      : gate_up_(gate_up), I_(intermediate) {}

  DBuf Apply(Dev d, const vt::Tensor& x) const override {
    const int64_t M = x.shape[0];
    vt::Tensor wgu = ResidentWeight(d, *gate_up_);  // [2I, H] raw-NK
    DBuf gate_up(d, vt::DType::kBF16, {M, 2 * I_});
    vt::MatmulBT(d.q, gate_up.t(), x, wgu);
    DBuf act(d, vt::DType::kBF16, {M, I_});
    vt::SiluAndMul(d.q, act.t(), gate_up.t());  // silu(gate)*up
    return act;
  }

  const char* Name() const override { return "bf16-gate-up"; }

 private:
  const OwnedTensor* gate_up_;
  int64_t I_;
};

}  // namespace layers
}  // namespace vllm
