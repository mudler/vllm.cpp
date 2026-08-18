// Per-tensor FP8 (W8A8) LinearMethod + the scheme-selection factory.
//
// UPSTREAM (ported FROM, ground-every-impl rule):
//   vllm/model_executor/layers/quantization/fp8.py:267,446
//     Fp8LinearMethod / Fp8LinearMethod.apply — the generic per-tensor fp8 W8A8
//     linear this file mirrors, and the reason this header sits directly under
//     quantization/ rather than under compressed_tensors/schemes/: upstream
//     `fp8.py` is a quantization-level module, not a compressed-tensors scheme.
//   vllm/model_executor/layers/quantization/compressed_tensors/schemes/
//     compressed_tensors_w8a8_fp8.py:60,201-207  (CompressedTensorsW8A8Fp8 —
//     the compressed-tensors spelling of the same checkpoint)
//   vllm/model_executor/layers/quantization/modelopt.py:444,531-537
//     (ModelOptFp8LinearMethod.apply — the ModelOpt spelling; note all three
//     delegate to the SAME `self.fp8_linear.apply_weights`, which is why one
//     method here covers every spelling our loader accepts)
//   vllm/model_executor/layers/quantization/base_config.py:180
//     QuantizationConfig.get_quant_method(layer, prefix) — the selection
//     MakeLinearMethod below mirrors: pick the method ONCE from the
//     checkpoint, not per call
//
// Our loader accepts both spellings (`LoadFp8Raw`,
// src/vllm/model_executor/models/qwen3_5_weights.cpp:423) and reduces them to
// one `Fp8Weight`: raw fp8-e4m3fn [N,K] bytes + per-tensor `weight_scale` +
// per-tensor `input_scale` + the folded `alpha = input_scale * weight_scale`.
// That is exactly the state upstream's shared `fp8_linear` consumes.
//
// These methods are a THIN policy wrapper over the byte-exact compute in
// dense_fp8_gemm.h (MatmulFp8CutlassD / MatmulFp8CutlassPreQuantD): identical
// vt:: op sequence, identical operands, identical order — the SAME template
// bodies src/vllm/model_executor/models/qwen3_5.cpp instantiates for the 35B
// production forward, instantiated here with the shared dense_attn::Dev/DBuf.
// Issue #940.
//
// DEVICE REACH — this method refuses on a host queue, and that is inherited,
// not chosen here. `MatmulFp8CutlassD`'s guard asks whether
// `vt::OpId::kMatmulFp8CublasLt` is registered for the running device, and it is
// registered for kCUDA only. #468/#842 registered CPU reference arms for
// `kQuantFp8Static` and `kMatmulFp8Cutlass`, so the OPS exist on the host, but
// the model-layer predicate still names the cuBLASLt op — the residual gap
// pinned at tests/vt/test_ops_fp8_cpu.cpp:445-453. Widening it is a dispatch
// change and belongs to its own row, not to the extraction that created this
// header.
#pragma once

#include <memory>

#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/models/dense_fp8_gemm.h"   // MatmulFp8Cutlass{,PreQuant}D
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, Fp8Weight
#include "vt/ops.h"

namespace vllm {
namespace layers {

// Per-tensor FP8 W8A8 plain linear (q/k/v/o_proj, GDN in_proj_qkv/z, out_proj,
// and the mamba in_proj/out_proj MODEL-NEMOTRON-H needs). `Apply` == static
// per-tensor activation quant + the folded-alpha fp8 GEMM, byte-for-byte
// dense_fp8::MatmulFp8CutlassD.
class Fp8W8A8LinearMethod : public LinearMethodBase {
 public:
  explicit Fp8W8A8LinearMethod(const Fp8Weight* weight) : w_(weight) {}

  DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const override {
    return dense_fp8::MatmulFp8CutlassD<DBuf>(d, x, *w_, out_dtype);
  }

  // The `QuantizedActivation` overload of the same upstream apply
  // (compressed_tensors_w8a8_fp8.py:201-207): `a_fp8` is ALREADY the static
  // per-tensor fp8 [M,K] a preceding fused epilogue produced, so the internal
  // QuantFp8Static is skipped and only the GEMM runs. Not on LinearMethodBase,
  // because a bf16 method has no meaning for a pre-quantized fp8 activation —
  // a caller reaches it only after selecting this scheme.
  DBuf ApplyPreQuantized(Dev d, const vt::Tensor& a_fp8,
                         vt::DType out_dtype) const {
    return dense_fp8::MatmulFp8CutlassPreQuantD<DBuf>(d, a_fp8, *w_, out_dtype);
  }

  const char* Name() const override { return "fp8-w8a8-per-tensor"; }

 private:
  const Fp8Weight* w_;
};

// --- Selection factory (mirrors get_quant_method) ---------------------------
// The scheme is chosen ONCE, here, from the checkpoint's populated weights:
// exactly one of {bf16, fp8} is present per projection (`LoadFp8Raw` fills the
// fp8 field and leaves the bf16 one EMPTY, and the dequant-at-load fallback does
// the reverse — qwen3_5_weights.cpp:391-405), so a non-empty fp8 weight selects
// the quantized method. Same shape as MakeLinearMethod for NVFP4 in
// compressed_tensors/schemes/nvfp4.h; overloaded on the weight type, so a model
// including both headers gets the right one by argument type.
inline std::unique_ptr<LinearMethodBase> MakeLinearMethod(
    const OwnedTensor& bf16_w, const Fp8Weight& fp8_w) {
  if (!fp8_w.Empty()) return std::make_unique<Fp8W8A8LinearMethod>(&fp8_w);
  return std::make_unique<UnquantizedLinearMethod>(&bf16_w);
}

}  // namespace layers
}  // namespace vllm
