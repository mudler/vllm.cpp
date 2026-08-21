// Block-wise (fine-grained) FP8 (W8A8) LinearMethod + the scheme-selection
// factory — MODEL-FP8-BLOCK-LINEAR, #1189 milestone M4, spec
// `.agents/specs/model-fp8-block-linear.md`.
//
// UPSTREAM (ported FROM, ground-every-impl rule), pinned vLLM
// `5559679229bc961848b121ccdeaa8fa5d79bec98`:
//   vllm/model_executor/layers/quantization/fp8.py:297-298
//     `Fp8LinearMethod.block_quant = self.weight_block_size is not None` — the
//     whole dispatch, and the reason this is a distinct method rather than a
//     flag on the per-tensor one.
//   vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:97-135
//     the apply this wraps: dynamic per-token per-group activation quant, then
//     the block-scaled GEMM. `dense_fp8_block::MatmulFp8BlockScaledD` is that
//     body; this file is the policy layer over it.
//   vllm/model_executor/layers/quantization/base_config.py:180
//     `QuantizationConfig.get_quant_method(layer, prefix)` — the selection
//     `MakeLinearMethod` below mirrors: pick the method ONCE from the
//     checkpoint, not per call by a tensor-name probe.
//
// The sibling of `Fp8W8A8LinearMethod` (quantization/fp8.h, #940) and the same
// shape: a thin policy wrapper over one templated compute body, so the model
// layer and this seam are the same code rather than two copies of it.
//
// WHY A DISTINCT WEIGHT TYPE AND A DISTINCT METHOD. A block scheme has no
// `input_scale` at all — the activation scheme is `dynamic` and the target
// checkpoint ships zero such tensors — and its weight scale is a 2-D grid, so
// there is no value `Fp8Weight::alpha` could take. `Fp8BlockWeight` is
// therefore a sibling of `Fp8Weight` rather than an extension of it
// (`models/qwen3_5_weights.h`), and overloading on the weight TYPE here means a
// call site that pairs the wrong method with the wrong weight fails to COMPILE.
//
// DEVICE REACH — this method refuses on any device with no block-scaled GEMM,
// and that is inherited from the op table rather than chosen here:
// `vt::MatmulFp8BlockScaled` is a CPU correctness reference (#1189 M2) and the
// mainloop-scaled CUTLASS kernel for `sm_121a` is #1189 M5. The refusal names
// the device and quotes the issue.
#pragma once

#include <memory>

#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/models/dense_fp8_block_gemm.h"  // MatmulFp8BlockScaledD
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, Fp8BlockWeight
#include "vt/ops.h"

namespace vllm {
namespace layers {

// Block-wise FP8 W8A8 plain linear. `Apply` == dynamic per-token per-group
// activation quant + the mainloop-scaled block GEMM, byte-for-byte
// `dense_fp8_block::MatmulFp8BlockScaledD` — the same template body the
// Qwen3.5 dense forward instantiates with its own device glue.
class Fp8BlockLinearMethod : public LinearMethodBase {
 public:
  explicit Fp8BlockLinearMethod(const Fp8BlockWeight* weight) : w_(weight) {}

  DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const override {
    return dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x, *w_, out_dtype);
  }

  // The bound weight's block geometry, so a caller that has to allocate around
  // this method (a fused epilogue, a merged operand) reads it from the weight
  // rather than from a config it would have to keep in step.
  int64_t block_n() const { return w_->block_n; }
  int64_t block_k() const { return w_->block_k; }

  const char* Name() const override { return "fp8-w8a8-block"; }

 private:
  const Fp8BlockWeight* w_;
};

// Block-wise FP8 W8A8 merged `gate_up` + SiluAndMul — MODEL-FP8-BLOCK-MERGED
// (#1189 milestone M6, spec `.agents/specs/model-fp8-block-merged.md`).
//
// vLLM's `gate_up_proj` is ONE MergedColumnParallelLinear
// (`models/qwen3_5.py:288-298` names the merge, `layers/linear.py:660` is the
// loader, @ `5559679229`), so this runs
// ONE block-scaled GEMM over the N-concatenated `[2I,K]` operand and the SwiGLU
// tail over its halves, rather than two GEMMs and a two-input activation. The
// merge is exact because a block scale is indexed by `n / block_n`; see
// `dense_fp8_block::CheckFp8BlockMergeable` for the one geometry it refuses and
// why upstream cannot express that one either.
//
// The SwiGLU sibling of `Nvfp4W4A16MlpGateUpMethod` beside it, and the same
// shape: a thin policy wrapper over one templated compute body, so the model
// layer and this seam are the same code rather than two copies of it. The
// merged device operand lives on the weights, not here, because it is built
// once for the model's lifetime and this object is a view.
class Fp8BlockMlpGateUpMethod : public MlpGateUpMethodBase {
 public:
  Fp8BlockMlpGateUpMethod(const Fp8BlockWeight* gate, const Fp8BlockWeight* up,
                          const Fp8BlockMergedResident* merged,
                          int64_t intermediate)
      : gate_(gate), up_(up), merged_(merged), I_(intermediate) {}

  DBuf Apply(Dev d, const vt::Tensor& x) const override {
    const dense_fp8_block::Fp8BlockShard shards[2] = {{gate_, "gate_proj"},
                                                      {up_, "up_proj"}};
    const dense_fp8_block::Fp8BlockMergedView view =
        dense_fp8_block::ResidentFp8BlockMerged(
            d, vt::kFp8BlockGateUpSwiGLU, "gate_up_proj", shards, 2, *merged_);
    VT_CHECK(view.n_total == 2 * I_,
             "fp8-w8a8-block-gate-up: the merged operand's N does not match "
             "twice the intermediate size");
    // bf16, because that is upstream's `out_dtype` here -- the model dtype
    // (`fp8.py:284`) -- and it is what every other arm of this seam emits.
    return dense_fp8_block::Fp8BlockGateUpSwiGLUD<DBuf>(d, x, view,
                                                        vt::DType::kBF16);
  }

  const char* Name() const override { return "fp8-w8a8-block-gate-up"; }

 private:
  const Fp8BlockWeight* gate_;
  const Fp8BlockWeight* up_;
  const Fp8BlockMergedResident* merged_;
  int64_t I_;
};

// --- Selection factory (mirrors get_quant_method) ---------------------------
// The scheme is chosen ONCE, here, from the checkpoint's populated weights:
// M3's loader rung fills the block field and leaves the bf16 and per-tensor
// ones EMPTY (`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`), so a
// non-empty block weight selects the quantized method. Same shape as
// `MakeLinearMethod` for per-tensor FP8 (quantization/fp8.h) and for NVFP4
// (compressed_tensors/schemes/nvfp4.h); overloaded on the weight type, so a
// model including all three headers gets the right one by argument type.
inline std::unique_ptr<LinearMethodBase> MakeLinearMethod(
    const OwnedTensor& bf16_w, const Fp8BlockWeight& block_w) {
  if (!block_w.Empty()) return std::make_unique<Fp8BlockLinearMethod>(&block_w);
  return std::make_unique<UnquantizedLinearMethod>(&bf16_w);
}

// The merged `gate_up` sibling, chosen the same way and by the same rule: M3's
// loader fills the two block slots and leaves the bf16 merged owner EMPTY.
inline std::unique_ptr<MlpGateUpMethodBase> MakeMlpGateUpMethod(
    const OwnedTensor& bf16_gate_up, const Fp8BlockWeight& gate_block,
    const Fp8BlockWeight& up_block, const Fp8BlockMergedResident& merged,
    int64_t intermediate) {
  if (!gate_block.Empty())
    return std::make_unique<Fp8BlockMlpGateUpMethod>(&gate_block, &up_block,
                                                     &merged, intermediate);
  return std::make_unique<UnquantizedMlpGateUpMethod>(&bf16_gate_up,
                                                      intermediate);
}

}  // namespace layers
}  // namespace vllm
