// Shared per-tensor **FP8 W8A8** (fp8-e4m3fn weights, fp8 activations) dense
// GEMM glue — the FP8 half of the QUANT-SCHEME additivity seam, and the sibling
// of dense_nvfp4_gemm.h.
//
// Extracted (behavior-preserving) from the anonymous namespace of
// src/vllm/model_executor/models/qwen3_5.cpp (`ResidentFp8`,
// `DenseCublasLtFp8Enabled`, `MatmulFp8CutlassD`, `MatmulFp8CutlassPreQuantD`)
// so a SECOND model needing an fp8 W8A8 projection can reach the residency +
// GEMM entry points without copying them. Before this header the only ways in
// were `#include`ing a `.cpp` (impossible) or re-typing the entry points into
// another translation unit — the hand-rolled parallel path AGENTS.md §"Shared
// seams" forbids. Issue #940.
//
// The forcing row is MODEL-NEMOTRON-H (#517): Nemotron-3.5-Lightning ships 46
// FP8 W8A8 projections (the mamba `in_proj`/`out_proj` of 23 layers), 36.6% of
// its decode bytes and 27.6% of its GEMM FLOPs. `in_proj` produces the fused
// `zxbcdt` the conv and the SSD scan consume, so that block cannot be split:
// without this seam its device path does not exist at all.
//
// UPSTREAM CHAIN (ported FROM, cited per the ground-every-impl rule; the
// checkpoint spelling our loader accepts is BOTH compressed-tensors and
// ModelOpt, and upstream both land on the same apply):
//   * scheme (compressed-tensors)  vllm/model_executor/layers/quantization/
//       compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:60,201-207
//       (`CompressedTensorsW8A8Fp8.apply_weights` -> `self.fp8_linear`)
//   * scheme (ModelOpt)            vllm/model_executor/layers/quantization/
//       modelopt.py:444,531-537  (`ModelOptFp8LinearMethod.apply` -> the SAME
//       `self.fp8_linear.apply_weights`)
//   * generic fp8 linear           vllm/model_executor/layers/quantization/
//       fp8.py:267,446  (`Fp8LinearMethod`)
//   * static per-tensor act quant  vllm/model_executor/layers/quantization/utils/
//       quant_utils.py:124 (`kFp8StaticTensorSym`), which modelopt.py:511-512
//       hands to `init_fp8_linear_kernel` — our vt::QuantFp8Static
//   * per-tensor scaled epilogue   the folded `alpha = input_scale *
//       weight_scale` scalar, mirroring vLLM's per-tensor ScaledEpilogue
// The local primitives this drives are vt::QuantFp8Static + vt::MatmulFp8CublasLt
// (default) / vt::MatmulFp8Cutlass (VT_DENSE_CUBLASLT_FP8=0), both realized in
// src/vt/cuda/cuda_matmul_fp8_cutlass.cu and (for the CPU reference arms of
// kQuantFp8Static / kMatmulFp8Cutlass) src/vt/cpu — see #468/#842.
//
// TEMPLATED ON Dev/DBuf ON PURPOSE — this is what makes the extraction REAL.
// `qwen3_5.cpp` carries its own anonymous-namespace `Dev`/`DBuf` (see the
// KNOWN-DUPLICATION note in dense_nvfp4_gemm.h: unifying the device-glue
// families is a separate, gate-model-touching refactor). A non-template header
// would therefore have had to be *copied* into qwen3_5.cpp rather than called by
// it — a seam that is dead alongside the production path proves nothing, which
// is precisely the failure #940 exists to prevent. Templating on the two glue
// types instead lets qwen3_5.cpp instantiate the ONE body with ITS types
// (generating byte-for-byte the code it had) while the shared layer
// (vllm::layers, dense_attn::Dev/DBuf) instantiates the SAME body with the
// shared ones. One definition, two instantiations, no copy.
//
// Both glue families satisfy the same tiny contract: `DevT` is `{Backend& b;
// Queue& q;}` and `DBufT` is constructible as `DBufT(d, dtype, shape)` with a
// `.t()` tensor view — dense_device_glue.h:42-56,108-130 and
// qwen3_5.cpp's copies of the same.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev/DBuf/MakeTensor
#include "vllm/model_executor/models/qwen3_5_weights.h"    // Fp8Weight
#include "vt/backend.h"
#include "vt/dtype.h"  // VT_CHECK
#include "vt/ops.h"

namespace vllm {
namespace dense_fp8 {

using vt::Backend;
using vt::DType;
using vt::Tensor;

// cuBLASLt FP8 dense GEMM toggle (VT_DENSE_CUBLASLT_FP8, DEFAULT ON when the fp8
// weights are resident). Routes the fp8 dense projections through vt::
// MatmulFp8CublasLt (cuBLASLt e4m3) instead of vt::MatmulFp8Cutlass (our cutlass
// sm120 fp8 GEMM, measured NEUTRAL vs bf16 at M=64/sm_121a). The activation
// quant + fp8-resident weight are IDENTICAL for both — only the GEMM backend
// differs, so both are the same fp8 W8A8 math (vLLM's scheme).
// VT_DENSE_CUBLASLT_FP8=0 restores the cutlass fp8 GEMM (the previous, validated
// path) for the parent's authoritative A/B.
//
// THE DEFAULT'S RECORDED JUSTIFICATION IS REFUTED, and the default has not
// moved (PERF-FP8-SMALL-M-DISPATCH, #1866). This comment used to call the
// cuBLASLt arm "the native equivalent of vLLM's measured-FASTER
// nvjet_sm121_qqtst fp8 kernels". At the pin vLLM runs `cutlass_scaled_mm` for
// this GEMM and no cuBLASLt at all, and #1857's artifact-verified GB10 profile
// measured our cuBLASLt arm resolving to `sm89_xmma ... tilesize32x64x64`, not
// nvjet — see the correction beside the kernel in `src/vt/cuda/cuda_matmul.cu`.
// The CUTLASS arm has since regained upstream's M16/M32 rungs, which is what
// makes the A/B this flag exists for a fair one for the first time at decode M.
// Flipping the default is a MEASUREMENT's decision and is `## Owed` in
// .agents/specs/perf-fp8-small-m-dispatch.md; nothing here presumes it.
inline bool DenseCublasLtFp8Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_DENSE_CUBLASLT_FP8");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Device-resident view over an Fp8Weight's raw fp8 [N,K] bytes, uploaded ONCE
// (lazily) and reused across every forward step (mirror ResidentNvfp4). The
// shared_ptr in the (const) weight owns the device buffer for the model lifetime.
template <class DevT>
inline Tensor ResidentFp8(DevT d, const Fp8Weight& w) {
  if (!w.d_packed) {
    const size_t pb = w.packed.bytes.size();
    void* p = d.b.Alloc(pb);
    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);
    Backend* bk = &d.b;
    w.d_packed = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return dense_attn::MakeTensor(w.d_packed.get(), DType::kI8, d.q.device,
                                {w.n, w.k});
}

// y[M,N] = x[M,K] (bf16/f32 device) @ dequant(w).T via a per-tensor W8A8 fp8
// GEMM: static per-tensor activation quant (vt::QuantFp8Static with the
// checkpoint input_scale) then an fp8 GEMM with the folded alpha
// (= input_scale·weight_scale). By DEFAULT the GEMM is cuBLASLt fp8 (vt::
// MatmulFp8CublasLt — a vt original, NOT a vLLM mirror; see the flag above);
// VT_DENSE_CUBLASLT_FP8=0 selects the cutlass sm120 fp8 GEMM, which IS the
// mirror (vt::MatmulFp8Cutlass). out dtype f32
// (q/k/v, in_proj_qkv/z sinks) or bf16 (o/out_proj residual sinks). CUDA-only
// (the 35B W8A8 path is CUDA-resident — fp8 fields are populated by DEFAULT on
// the CUDA+cutlass load, VT_DENSE_NATIVE).
//
// The CUDA-only refusal is UNCHANGED by the extraction and is deliberately
// keyed on `kMatmulFp8CublasLt`, which is registered for kCUDA only — a
// "cuBLASLt" kernel on the host would be a lie in the name. That is what the
// pinned expectation at tests/vt/test_ops_fp8_cpu.cpp:445-453 records, and it is
// why the CPU registrations of kQuantFp8Static / kMatmulFp8Cutlass (#468/#842)
// do NOT by themselves make this entry point run on a host queue. Nothing here
// widens it; see the header's issue for the residual gap.
template <class DBufT, class DevT>
inline DBufT MatmulFp8CutlassD(DevT d, const Tensor& x, const Fp8Weight& w,
                               DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  VT_CHECK(vt::OpRegistered(vt::OpId::kMatmulFp8CublasLt, d.q.device.type),
           "MatmulFp8CutlassD: the fp8 W8A8 path is CUDA-only");
  DBufT a_fp8(d, DType::kI8, {M, K});
  vt::QuantFp8Static(d.q, a_fp8.t(), x, w.input_scale);
  Tensor wdev = ResidentFp8(d, w);
  DBufT dout(d, out_dtype, {M, N});
  if (DenseCublasLtFp8Enabled())
    vt::MatmulFp8CublasLt(d.q, dout.t(), a_fp8.t(), wdev, w.alpha);
  else
    vt::MatmulFp8Cutlass(d.q, dout.t(), a_fp8.t(), wdev, w.alpha);
  return dout;
}

// Pre-quantized fp8 analog of MatmulFp8CutlassD: the activation is ALREADY the
// static-quant fp8 [M,K] (produced ONCE — either by RmsNormQuantFp8 or a shared
// quant — and fed to every projection reading it), so this SKIPS the internal
// QuantFp8Static and runs only the fp8 GEMM. The fp8 counterpart of
// MatmulNvfp4Fp4DirectD; each GEMM still applies its own folded alpha (= shared
// input_scale · this projection's weight_scale), so the result is identical to
// MatmulFp8CutlassD(x) when a_fp8 == QuantFp8Static(x, w.input_scale).
//
// Upstream this is the `x: torch.Tensor | QuantizedActivation` overload of the
// same apply (compressed_tensors_w8a8_fp8.py:201-207) — an activation that a
// preceding fused epilogue already quantized is handed to the GEMM as-is.
template <class DBufT, class DevT>
inline DBufT MatmulFp8CutlassPreQuantD(DevT d, const Tensor& a_fp8,
                                       const Fp8Weight& w, DType out_dtype) {
  const int64_t M = a_fp8.shape[0], N = w.n;
  VT_CHECK(vt::OpRegistered(vt::OpId::kMatmulFp8CublasLt, d.q.device.type),
           "MatmulFp8CutlassPreQuantD: the fp8 W8A8 path is CUDA-only");
  Tensor wdev = ResidentFp8(d, w);
  DBufT dout(d, out_dtype, {M, N});
  if (DenseCublasLtFp8Enabled())
    vt::MatmulFp8CublasLt(d.q, dout.t(), a_fp8, wdev, w.alpha);
  else
    vt::MatmulFp8Cutlass(d.q, dout.t(), a_fp8, wdev, w.alpha);
  return dout;
}

}  // namespace dense_fp8
}  // namespace vllm
