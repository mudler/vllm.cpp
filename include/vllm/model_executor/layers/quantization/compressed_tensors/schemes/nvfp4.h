// Compressed-tensors NVFP4 W4A16 LinearMethods + the scheme-selection factory.
//
// UPSTREAM (ported FROM, ground-every-impl rule):
//   vllm/model_executor/layers/quantization/compressed_tensors/schemes/
//     compressed_tensors_w4a4_nvfp4.py:29-32,95-141  (CompressedTensorsW4A4Fp4,
//     use_a16=True — vLLM has NO separate W4A16 class; the a16 flag reuses this)
//   vllm/model_executor/layers/quantization/base_config.py:180
//     QuantizationConfig.get_quant_method(layer, prefix)  — the selection this
//     factory mirrors: pick the method ONCE from the checkpoint, not per call
//   vllm/model_executor/kernels/linear/__init__.py:879-881  (forced-Marlin for
//     the a16 / weight-only path)
//
// These methods are a THIN policy wrapper over the byte-exact compute already in
// dense_nvfp4_gemm.h (MatmulNvfp4W4A16D / GateUpFusedMarlinD): identical vt:: op
// sequence, identical operands, identical order. The ONLY change vs the inline
// per-model dispatch they replace is that scheme selection is centralized in
// MakeLinearMethod/MakeMlpGateUpMethod below (one place, mirroring
// get_quant_method) instead of a per-call `IsNvfp4()` tensor-name probe, and the
// device gate inside is a vt::OpRegistered op-availability query rather than a
// `device == kCUDA` test.
#pragma once

#include <memory>

#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // MatmulNvfp4W4A16D, GateUp…
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor, Nvfp4Weight
#include "vt/ops.h"

namespace vllm {
namespace layers {

// NVFP4 W4A16 plain linear (qkv_proj / o_proj / down_proj). apply == the forced-
// Marlin-else-fallback W4A16 GEMM. Byte-for-byte dense_nvfp4::MatmulNvfp4W4A16D.
class Nvfp4W4A16LinearMethod : public LinearMethodBase {
 public:
  explicit Nvfp4W4A16LinearMethod(const Nvfp4Weight* weight) : w_(weight) {}

  DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const override {
    return dense_nvfp4::MatmulNvfp4W4A16D(d, x, *w_, out_dtype);
  }

  const char* Name() const override { return "compressed-tensors-nvfp4-w4a16"; }

 private:
  const Nvfp4Weight* w_;
};

// NVFP4 W4A16 merged gate_up + SiluAndMul. Mirrors the inline fp4 MLP branch
// (qwen3.cpp MlpBlock): ONE fused Marlin gate_up GEMM (size_n = 2I) + SiluAndMul
// when eligible, else two split W4A16 GEMMs fed to MoeSiluMul. The device gate
// is the op-availability query (kMoeGroupedGemmNvfp4Marlin registered here?),
// byte-identical on the production build to the old `device == kCUDA`.
class Nvfp4W4A16MlpGateUpMethod : public MlpGateUpMethodBase {
 public:
  Nvfp4W4A16MlpGateUpMethod(const Nvfp4Weight* gate, const Nvfp4Weight* up,
                            int64_t intermediate)
      : gate_(gate), up_(up), I_(intermediate) {}

  DBuf Apply(Dev d, const vt::Tensor& x) const override {
    const int64_t M = x.shape[0];
#ifdef VT_MARLIN_NVFP4
    if (vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type) &&
        dense_nvfp4::MarlinW4A16Enabled() &&
        dense_nvfp4::GateUpFusedEligible(*gate_, *up_)) {
      // vLLM's shape: ONE Marlin GEMM over the merged gate_up operand
      // (size_n = 2I) + SiluAndMul on the halves.
      return dense_nvfp4::GateUpFusedMarlinD(d, x, *gate_, *up_);
    }
#endif
    // SPLIT A/B fallback (and the CPU reference path): two W4A16 GEMMs fed to the
    // two-input MoeSiluMul — bit-identical to the fused arm's SiluAndMul.
    DBuf gate = dense_nvfp4::MatmulNvfp4W4A16D(d, x, *gate_, vt::DType::kBF16);
    DBuf up = dense_nvfp4::MatmulNvfp4W4A16D(d, x, *up_, vt::DType::kBF16);
    DBuf a(d, vt::DType::kBF16, {M, I_});
    vt::MoeSiluMul(d.q, a.t(), gate.t(), up.t());
    return a;
  }

  const char* Name() const override {
    return "compressed-tensors-nvfp4-w4a16-gate-up";
  }

 private:
  const Nvfp4Weight* gate_;
  const Nvfp4Weight* up_;
  int64_t I_;
};

// --- Selection factories (mirror get_quant_method) --------------------------
// The scheme is chosen ONCE, here, from the checkpoint's populated weights:
// exactly one of {bf16, nvfp4} is present per projection (the loader probes
// `.weight_packed`), so a non-empty fp4 weight selects the quantized method.
// This replaces the per-call `IsNvfp4()` tensor-name probe + inline device gate
// scattered through the model forward with one factory in the shared layer.

inline std::unique_ptr<LinearMethodBase> MakeLinearMethod(
    const OwnedTensor& bf16_w, const Nvfp4Weight& fp4_w) {
  if (!fp4_w.Empty())
    return std::make_unique<Nvfp4W4A16LinearMethod>(&fp4_w);
  return std::make_unique<UnquantizedLinearMethod>(&bf16_w);
}

inline std::unique_ptr<MlpGateUpMethodBase> MakeMlpGateUpMethod(
    const OwnedTensor& bf16_gate_up, const Nvfp4Weight& gate_fp4,
    const Nvfp4Weight& up_fp4, int64_t intermediate) {
  if (!gate_fp4.Empty())
    return std::make_unique<Nvfp4W4A16MlpGateUpMethod>(&gate_fp4, &up_fp4,
                                                       intermediate);
  return std::make_unique<UnquantizedMlpGateUpMethod>(&bf16_gate_up, intermediate);
}

}  // namespace layers
}  // namespace vllm
